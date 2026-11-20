//! The server under concurrency, in process: `serve` with an embedded client and no listener, headless, each arch's
//! servable tiny fixture (`cooler_contract.serverDir`) loaded on demand by the first request, on Sushi Cooler. Each
//! connection is a socketpair served as `acceptLoop` serves an accepted one. The server's allocator writes 0xAA over
//! what it frees, so a read of freed memory reads the poison instead of the bytes that happened to stay.
const std = @import("std");
const testing = std.testing;
const mlx = @import("mlx.zig");
const server = @import("server.zig");
const server_state = @import("server_state.zig");
const model_load = @import("model_load.zig");
const registry = @import("arch/registry.zig");
const responses_api = @import("responses_api.zig");
const http_conn = @import("http_conn.zig");
const scheduler_mod = @import("scheduler.zig");
const test_support = @import("test_support.zig");
const bills = @import("bills.zig");
const model_mod = @import("model.zig");
const trunk = @import("trunk.zig");
const http = @import("http.zig");
const ModelRegistry = @import("model_registry.zig").ModelRegistry;
const nowMsMonotonic = @import("io_util.zig").nowMsMonotonic;

const io = std.testing.io;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

/// The allocator every server test serves with (`Poison`), for the client side's own buffers too.
var gpa: std.mem.Allocator = undefined;
var scenario: *const fn () anyerror!void = undefined;
var failed: ?anyerror = null;

const Poison = struct {
    const vt: std.mem.Allocator.VTable = .{ .alloc = alloc, .resize = resize, .remap = remap, .free = free };
    fn child() std.mem.Allocator {
        return testing.allocator;
    }
    fn alloc(_: *anyopaque, len: usize, al: std.mem.Alignment, ra: usize) ?[*]u8 {
        return child().rawAlloc(len, al, ra);
    }
    fn resize(_: *anyopaque, mem: []u8, al: std.mem.Alignment, new_len: usize, ra: usize) bool {
        if (!child().rawResize(mem, al, new_len, ra)) return false;
        if (new_len < mem.len) @memset(mem[new_len..], 0xAA);
        return true;
    }
    fn remap(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) ?[*]u8 {
        return null;
    }
    fn free(_: *anyopaque, mem: []u8, al: std.mem.Alignment, ra: usize) void {
        @memset(mem, 0xAA);
        child().rawFree(mem, al, ra);
    }
    fn allocator() std.mem.Allocator {
        return .{ .ptr = undefined, .vtable = &vt };
    }
};

fn clientMain() void {
    defer server.requestShutdown();
    scenario() catch |e| {
        failed = e;
    };
}

fn startClient() anyerror!std.Thread {
    return std.Thread.spawn(.{}, clientMain, .{});
}

fn startClientOnReady(_: u16) anyerror!std.Thread {
    return startClient();
}

/// `ssd_budget_gb`: what the fixture's load names in its model-settings.json (`cooler_contract.server_ssd_budget_gb`).
const Arch = struct { name: []const u8, tmp: testing.TmpDir, ssd_budget_gb: ?u32 = null };

/// What a test asks of a fixture beyond being served.
const Needs = struct {
    /// A chat or Responses reply that runs on for hundreds of tokens (`cooler_contract.server_long_replies`).
    long_replies: bool = false,
    /// A second copy of the fixture, served as `tiny2`.
    tiny2: bool = false,
    /// How many models the registry keeps resident.
    max_resident: u32 = 2,
    /// Serve on a real listener (an ephemeral port) with the client started by `on_ready`, so `acceptLoop` runs.
    listener: bool = false,
};

/// `body` as the embedded client of a server of `max_concurrent` rows, once per arch that declares a servable fixture
/// with what `needs` asks. A missing fixture skips the test before any load.
fn eachArch(max_concurrent: u32, needs: Needs, body: *const fn () anyerror!void) !void {
    try mlx.requireGpu();
    var archs: [std.meta.tags(registry.Tag).len]?Arch = @splat(null);
    defer for (&archs) |*a| if (a.*) |*x| x.tmp.cleanup();
    inline for (comptime std.meta.tags(registry.Tag), 0..) |t, i| {
        const M = registry.Mod(t);
        if (comptime @hasDecl(M, "cooler_contract") and @hasDecl(M.cooler_contract, "serverDir")) if (!needs.long_replies or M.cooler_contract.server_long_replies) {
            var tmp = testing.tmpDir(.{});
            archs[i] = .{ .name = @tagName(t), .tmp = tmp, .ssd_budget_gb = if (@hasDecl(M.cooler_contract, "server_ssd_budget_gb")) M.cooler_contract.server_ssd_budget_gb else null };
            const dirs = [_][]const u8{ "m", "m2" };
            for (dirs[0..if (needs.tiny2) 2 else 1]) |sub| {
                try tmp.dir.createDirPath(io, sub);
                var m = try tmp.dir.openDir(io, sub, .{});
                defer m.close(io);
                try M.cooler_contract.serverDir(m);
            }
        };
    }
    for (&archs) |*a| if (a.*) |*x| withServer(&x.tmp, x.ssd_budget_gb, max_concurrent, needs, body) catch |e| {
        if (e != error.SkipZigTest) std.debug.print("server test on {s}: {s}\n", .{ x.name, @errorName(e) });
        return e;
    };
}

fn withServer(tmp: *testing.TmpDir, ssd_budget_gb: ?u32, max_concurrent: u32, needs: Needs, body: *const fn () anyerror!void) !void {
    const a = Poison.allocator();
    try tmp.dir.createDirPath(io, "home/.sushi");
    try tmp.dir.createDirPath(io, "cache");
    var root_buf: [512]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(io, &root_buf)];
    var pb: [1024]u8 = undefined;
    const model_dir = try std.fmt.bufPrint(&pb, "{s}/m", .{root});
    var hb: [1024]u8 = undefined;
    const home = try std.fmt.bufPrintSentinel(&hb, "{s}/home", .{root}, 0);
    var cb: [1024]u8 = undefined;
    const cache = try std.fmt.bufPrintSentinel(&cb, "{s}/cache", .{root}, 0);
    var old_home_buf: [1024]u8 = undefined;
    const old_home: ?[:0]const u8 = if (std.c.getenv("HOME")) |h| try std.fmt.bufPrintSentinel(&old_home_buf, "{s}", .{std.mem.span(h)}, 0) else null;
    _ = setenv("HOME", home.ptr, 1);
    defer if (old_home) |h| {
        _ = setenv("HOME", h.ptr, 1);
    };
    _ = setenv("SUSHI_PREFIX_CACHE_DIR", cache.ptr, 1);
    defer _ = unsetenv("SUSHI_PREFIX_CACHE_DIR");
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);

    const reg = try ModelRegistry.init(a, io, null, needs.max_resident, 0, null);
    defer reg.deinit();
    var settings_buf: [1024]u8 = undefined;
    reg.settings_path = try std.fmt.bufPrint(&settings_buf, "{s}/.sushi/model-settings.json", .{home});
    var pb2: [1024]u8 = undefined;
    const model_dir2 = try std.fmt.bufPrint(&pb2, "{s}/m2", .{root});
    if (ssd_budget_gb) |gb| {
        var sb: [2400]u8 = undefined;
        const settings = if (needs.tiny2)
            try std.fmt.bufPrint(&sb, "{{\"{s}\":{{\"ssd_budget_gb\":{d}}},\"{s}\":{{\"ssd_budget_gb\":{d}}}}}", .{ model_dir, gb, model_dir2, gb })
        else
            try std.fmt.bufPrint(&sb, "{{\"{s}\":{{\"ssd_budget_gb\":{d}}}}}", .{ model_dir, gb });
        try tmp.dir.writeFile(io, .{ .sub_path = "home/.sushi/model-settings.json", .data = settings });
    }
    const entry = try reg.registerStub("tiny", model_dir, 90 << 20);
    if (needs.tiny2) _ = try reg.registerStub("tiny2", model_dir2, 90 << 20);
    try reg.setDefault("tiny");

    var stub = try model_load.headlessStubCpuState(a);
    defer model_load.freeCpuState(a, &stub);
    const params = model_load.LoadParams{
        .registry = reg,
        .entry = entry,
        .config = stub.config,
        .tok = stub.tok,
        .chat_config = stub.chat_config,
        .model_dir = "",
        .no_initial_load = true,
        .load_vision = false,
        .warmup_eager = false,
        .prefix_cache_capacity = server_state.prefix_cache_capacity,
        .prefix_cache_ram_enabled = server_state.prefix_cache_ram_enabled,
        .prefix_cache_mem_bytes = server_state.prefix_cache_mem_bytes,
        .prefix_cache_mem_resolver = server.prefixCacheMemForLoad,
        .prefix_cache_disk_resolver = server.prefixCacheDiskForLoad,
        .expert_cache_fit_resolver = server.expertCacheFitForLoad,
        .ssm_checkpoint_stride = server_state.ssm_checkpoint_stride,
        .ssm_checkpoint_max = server_state.ssm_checkpoint_max,
    };
    gpa = a;
    scenario = body;
    failed = null;
    const prev_max = server_state.max_concurrent;
    server_state.max_concurrent = max_concurrent;
    defer server_state.max_concurrent = prev_max;
    defer server_state.shutdown_requested.store(false, .release);
    server.accept_spawned_with_hooks = false;
    try server.serve(io, a, params, stub.config, "127.0.0.1", 0, .{
        .embedded_client = if (needs.listener) null else startClient,
        .on_ready = if (needs.listener) startClientOnReady else null,
        .request_timeout_sec = 300,
        .test_on_cooler = true,
    });
    if (failed) |e| return e;
}

/// One connection, served as `acceptLoop` serves an accepted one. Returns the client's end.
fn connect() !std.posix.fd_t {
    var sv: [2]std.posix.fd_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &sv) != 0) return error.SocketPairFailed;
    // Small buffers: a client that stops reading stops the server's writes after a few events.
    const small: c_int = 2048;
    _ = std.c.setsockopt(sv[0], std.posix.SOL.SOCKET, std.posix.SO.SNDBUF, std.mem.asBytes(&small), @sizeOf(c_int));
    _ = std.c.setsockopt(sv[1], std.posix.SOL.SOCKET, std.posix.SO.RCVBUF, std.mem.asBytes(&small), @sizeOf(c_int));
    // A read that waits past this fails the test instead of hanging it; it budgets a cold first load, whose kernel
    // compiles alone can pass half a minute on a loaded box.
    const limit: std.posix.timeval = .{ .sec = 120, .usec = 0 };
    _ = std.c.setsockopt(sv[1], std.posix.SOL.SOCKET, std.posix.SO.RCVTIMEO, std.mem.asBytes(&limit), @sizeOf(std.posix.timeval));
    server.spawnConnection(gpa, io, .{ .socket = .{ .handle = sv[0], .address = .{ .ip4 = .loopback(0) } } });
    return sv[1];
}

fn sendAll(fd: std.posix.fd_t, bytes: []const u8) !void {
    var off: usize = 0;
    while (off < bytes.len) {
        const n = std.c.write(fd, bytes[off..].ptr, bytes.len - off);
        if (n <= 0) return error.WriteFailed;
        off += @intCast(n);
    }
}

fn send(fd: std.posix.fd_t, method: []const u8, path: []const u8, body: []const u8) !void {
    var hb: [256]u8 = undefined;
    try sendAll(fd, try std.fmt.bufPrint(&hb, "{s} {s} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {d}\r\n\r\n", .{ method, path, body.len }));
    try sendAll(fd, body);
}

/// Everything until the server closes, or until `needle` arrives when one is given; the caller frees.
fn readUntil(fd: std.posix.fd_t, needle: ?[]const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    var buf: [16 * 1024]u8 = undefined;
    while (needle == null or std.mem.indexOf(u8, out.items, needle.?) == null) {
        const n = std.c.read(fd, &buf, buf.len);
        if (n == 0) break;
        if (n < 0) return if (std.posix.errno(n) == .AGAIN) error.ReadTimedOut else error.ReadFailed;
        try out.appendSlice(gpa, buf[0..@intCast(n)]);
    }
    return out.toOwnedSlice(gpa);
}

/// Whether `text` holds a Responses reply's terminal event: completed, or incomplete when max_output_tokens cut it.
fn hasResponsesTerminal(text: []const u8) bool {
    return std.mem.indexOf(u8, text, "\"type\":\"response.completed\"") != null or std.mem.indexOf(u8, text, "\"type\":\"response.incomplete\"") != null;
}

/// Everything read until the Responses terminal event.
fn readUntilResponsesTerminal(fd: std.posix.fd_t) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    var buf: [16 * 1024]u8 = undefined;
    while (!hasResponsesTerminal(out.items)) {
        const n = std.c.read(fd, &buf, buf.len);
        if (n == 0) break;
        if (n < 0) return if (std.posix.errno(n) == .AGAIN) error.ReadTimedOut else error.ReadFailed;
        try out.appendSlice(gpa, buf[0..@intCast(n)]);
    }
    return out.toOwnedSlice(gpa);
}

test "server test client: a read past its limit fails by name, never as a short answer" {
    gpa = testing.allocator;
    var sv: [2]std.posix.fd_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &sv) != 0) return error.SocketPairFailed;
    defer _ = std.c.close(sv[1]);
    const limit: std.posix.timeval = .{ .sec = 0, .usec = 50_000 };
    _ = std.c.setsockopt(sv[1], std.posix.SOL.SOCKET, std.posix.SO.RCVTIMEO, std.mem.asBytes(&limit), @sizeOf(std.posix.timeval));
    try sendAll(sv[0], "HTTP/1.1 200 OK\r\n");
    try testing.expectError(error.ReadTimedOut, readUntil(sv[1], "\r\n\r\n"));
    // A peer that closes ends the read with what it sent.
    try sendAll(sv[0], "{}");
    _ = std.c.close(sv[0]);
    const rest = try readUntil(sv[1], null);
    defer gpa.free(rest);
    try testing.expectEqualStrings("{}", rest);
}

fn request(method: []const u8, path: []const u8, body: []const u8) ![]u8 {
    const fd = try connect();
    defer _ = std.c.close(fd);
    try send(fd, method, path, body);
    return readUntil(fd, null);
}

/// The first `"id":"resp_..."` of a Responses answer.
fn respId(resp: []const u8) ![]const u8 {
    const at = std.mem.indexOf(u8, resp, "\"id\":\"resp_") orelse return error.NoResponseId;
    const start = at + "\"id\":\"".len;
    const end = std.mem.indexOfScalarPos(u8, resp, start, '"') orelse return error.NoResponseId;
    return resp[start..end];
}

/// The bytes a connection holds unread between the server's write and the client's read, the server's send buffer as
/// large as the client's receive buffer (`connect`): what the server writes past before it blocks.
fn buffered(client: std.posix.fd_t) !usize {
    var rcv: c_int = 0;
    var len: std.posix.socklen_t = @sizeOf(c_int);
    if (std.c.getsockopt(client, std.posix.SOL.SOCKET, std.posix.SO.RCVBUF, @ptrCast(&rcv), &len) != 0) return error.GetSockOpt;
    return 2 * @as(usize, @intCast(rcv));
}

test "server: a Responses continuation keeps the history it continues when that response is deleted mid-generation" {
    try eachArch(4, .{ .long_replies = true }, struct {
        fn run() !void {
            const r1 = try request("POST", "/v1/responses", "{\"model\":\"tiny\",\"input\":\"remember the word salmon please\",\"max_output_tokens\":4,\"temperature\":0}");
            defer gpa.free(r1);
            const id1 = try respId(r1);
            var b: [512]u8 = undefined;
            const fd = try connect();
            defer _ = std.c.close(fd);
            try send(fd, "POST", "/v1/responses", try std.fmt.bufPrint(&b, "{{\"model\":\"tiny\",\"previous_response_id\":\"{s}\",\"input\":\"which word was it\",\"max_output_tokens\":300,\"temperature\":0,\"stream\":true}}", .{id1}));
            const head = try readUntil(fd, "delta");
            defer gpa.free(head);
            // The client reads no more until the delete has answered, so the stream stands in a blocked write.
            const d = try request("DELETE", try std.fmt.bufPrint(&b, "/v1/responses/{s}", .{id1}), "");
            defer gpa.free(d);
            if (!std.mem.startsWith(u8, d, "HTTP/1.1 200")) return error.DeleteRefused;
            const rest = try readUntil(fd, null);
            defer gpa.free(rest);
            // More than the connection holds came after the delete: the continuation was still generating, not stored.
            if (rest.len <= try buffered(fd)) return error.DeleteNotMidReply;
            // The continuation's id rides its first event.
            const id2 = try respId(head);
            const store = responses_api.responseStore();
            const sr = store.acquire(id2) orelse return error.ContinuationNotStored;
            defer store.release(sr);
            var saw = false;
            for (sr.history) |m| saw = saw or std.mem.indexOf(u8, m.content, "salmon") != null;
            if (!saw) return error.ContinuationLostItsHistory;
        }
    }.run);
}

/// A WebSocket on `/v1/responses`, past its handshake. Returns the client's end.
fn wsConnect() !std.posix.fd_t {
    const fd = try connect();
    errdefer _ = std.c.close(fd);
    try sendAll(fd, "GET /v1/responses HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n");
    const head = try readUntil(fd, "\r\n\r\n");
    defer gpa.free(head);
    if (!std.mem.startsWith(u8, head, "HTTP/1.1 101")) return error.NoUpgrade;
    return fd;
}

/// One masked client text frame (a zero mask: the payload goes as is).
fn wsSend(fd: std.posix.fd_t, text: []const u8) !void {
    var head: [8]u8 = undefined;
    head[0] = 0x81;
    const n: usize = if (text.len < 126) blk: {
        head[1] = 0x80 | @as(u8, @intCast(text.len));
        break :blk 2;
    } else blk: {
        head[1] = 0x80 | 126;
        std.mem.writeInt(u16, head[2..4], @intCast(text.len), .big);
        break :blk 4;
    };
    try sendAll(fd, head[0..n]);
    try sendAll(fd, &.{ 0, 0, 0, 0 });
    try sendAll(fd, text);
}

test "server: a WebSocket store:false chain stays on its connection, never in the shared store, mid-turn too" {
    try eachArch(4, .{ .long_replies = true }, struct {
        fn run() !void {
            const fd = try wsConnect();
            defer _ = std.c.close(fd);
            try wsSend(fd, "{\"type\":\"response.create\",\"model\":\"tiny\",\"input\":\"remember the word salmon please\",\"max_output_tokens\":4,\"temperature\":0,\"store\":false}");
            const t1 = try readUntilResponsesTerminal(fd);
            defer gpa.free(t1);
            const id1 = try respId(t1);
            var b: [512]u8 = undefined;
            try wsSend(fd, try std.fmt.bufPrint(&b, "{{\"type\":\"response.create\",\"model\":\"tiny\",\"previous_response_id\":\"{s}\",\"input\":\"which word was it\",\"max_output_tokens\":300,\"temperature\":0,\"store\":false}}", .{id1}));
            const head = try readUntil(fd, "delta");
            defer gpa.free(head);
            // The client reads no more until the lookup has answered, so the turn stands in a blocked write.
            const got = try request("GET", try std.fmt.bufPrint(&b, "/v1/responses/{s}", .{id1}), "");
            defer gpa.free(got);
            if (!std.mem.startsWith(u8, got, "HTTP/1.1 404")) return error.LocalRecordInSharedStore;
            const rest = try readUntilResponsesTerminal(fd);
            defer gpa.free(rest);
            if (rest.len <= try buffered(fd)) return error.LookupNotMidTurn;
            if (std.mem.indexOf(u8, rest, "\"type\":\"error\"") != null) return error.TurnFailed;
        }
    }.run);
}

/// A greedy completion that runs its whole `n` tokens (`ignore_eos`): every tiny fixture's context holds 100.
fn completionBody(buf: []u8, text: []const u8, n: u32, stream: bool) ![]const u8 {
    return std.fmt.bufPrint(buf, "{{\"model\":\"tiny\",\"prompt\":\"{s}\",\"max_tokens\":{d},\"temperature\":0,\"ignore_eos\":true,\"stream\":{s}}}", .{ text, n, if (stream) "true" else "false" });
}

/// Polls `done` every 10 ms for at most `ms`; whether it came.
fn within(ms: u32, done: *const fn () bool) bool {
    var waited: u32 = 0;
    while (!done()) : (waited += 10) {
        if (waited >= ms) return false;
        var none = [_]std.posix.pollfd{};
        _ = std.posix.poll(&none, 10) catch {};
    }
    return true;
}

test "server: a headless boot starts accepting only once the request hooks are installed" {
    try eachArch(1, .{ .listener = true }, struct {
        fn run() !void {
            try testing.expect(server.accept_spawned_with_hooks);
        }
    }.run);
}

test "server: a shutdown with the submit queue full ends the request waiting for room with the shutdown error too" {
    try eachArch(1, .{}, struct {
        var next_id: u64 = 0;
        fn queueFull() bool {
            const sch = server_state.global_scheduler.?;
            sch.queue_mu.lockUncancelable(sch.io);
            defer sch.queue_mu.unlock(sch.io);
            return sch.in_flight == sch.queue_cap;
        }
        fn submitted() bool {
            const sch = server_state.global_scheduler.?;
            sch.queue_mu.lockUncancelable(sch.io);
            defer sch.queue_mu.unlock(sch.io);
            return sch.req_seq == next_id;
        }
        fn run() !void {
            var b: [512]u8 = undefined;
            // The model loads first, so every request below reaches submit.
            gpa.free(try request("POST", "/v1/completions", try completionBody(&b, "hi", 2, false)));
            const sch = server_state.global_scheduler.?;
            next_id = sch.req_seq;
            const first_id = next_id;
            // One more than the queue holds: the last waits in submit for room. Each submits before the next is
            // sent, so stream i carries request id first_id + i.
            var fds: [34]std.posix.fd_t = undefined;
            var n_open: usize = 0;
            defer for (fds[0..n_open]) |fd| {
                _ = std.c.close(fd);
            };
            for (&fds, 0..) |*fd, i| {
                fd.* = try connect();
                n_open += 1;
                try send(fd.*, "POST", "/v1/completions", try completionBody(&b, "write forever about rivers", 100, true));
                if (i + 1 == fds.len) break;
                next_id += 1;
                if (!within(5000, submitted)) return error.NeverSubmitted;
            }
            if (!within(5000, queueFull)) return error.QueueNeverFull;
            // Waiting at the shutdown: still pending, or not yet submitted. Read in the hold the shutdown is asked
            // in, as admission reads it, so no stream changes side between the two.
            var waiting: [fds.len]bool = @splat(true);
            {
                sch.queue_mu.lockUncancelable(sch.io);
                defer sch.queue_mu.unlock(sch.io);
                for (&waiting, 0..) |*w, i| {
                    const id = first_id + i;
                    if (id >= sch.req_seq) continue;
                    w.* = for (sch.pending.items) |slot| {
                        if (slot.request_id == id) break true;
                    } else false;
                }
                server.requestShutdown();
            }
            for (fds, waiting, 0..) |fd, w, i| {
                const all = try readUntil(fd, null);
                defer gpa.free(all);
                if (!w) continue;
                errdefer std.debug.print("stream {d} of {d} waiting at the shutdown: {s}\n", .{ i, fds.len, all[all.len -| 600..] });
                if (std.mem.indexOf(u8, all, "\"finish_reason\":\"length\"") != null or std.mem.indexOf(u8, all, "\"finish_reason\":\"stop\"") != null)
                    return error.ServedPastTheShutdown;
                if (std.mem.indexOf(u8, all, "shutting down") == null) return error.EndedWithoutTheShutdownError;
            }
        }
    }.run);
}

/// A POST on a thread of its own: `out` holds the whole answer once `done`; empty when it failed.
const Asked = struct {
    path: []const u8,
    body: []const u8,
    out: []u8 = &.{},
    done: std.atomic.Value(bool) = .init(false),
    fn run(q: *@This()) void {
        q.out = request("POST", q.path, q.body) catch &.{};
        q.done.store(true, .release);
    }
};

/// A stream the client stops reading after its first token, with its socket left open, an unload asked behind it and
/// then another request. The stall stays open until both answer: a hold by the stalled reader lasts until its socket
/// shuts. `ms` budgets the reload's served warm; the request behind must claim its model within `claim_ms` of the
/// unload's answer. Returns the two answers; the caller frees.
fn behindAStalledReader(ms: u32) ![2][]u8 {
    const claim_ms = 5_000;
    var b: [2][512]u8 = undefined;
    gpa.free(try request("POST", "/v1/completions", try completionBody(&b[0], "hi", 2, false)));
    const stalled = try connect();
    defer _ = std.c.close(stalled);
    try send(stalled, "POST", "/v1/completions", try completionBody(&b[1], "a long tale of the sea", 64, true));
    gpa.free(try readUntil(stalled, "\"choices\""));
    var asked = [_]Asked{ .{ .path = "/v1/unload-model", .body = "{\"model\":\"tiny\"}" }, .{ .path = "/v1/completions", .body = try completionBody(&b[0], "hello", 4, false) } };
    var threads: [2]std.Thread = undefined;
    var waited: u32 = 0;
    for (&asked, 0..) |*q, i| {
        threads[i] = try std.Thread.spawn(.{}, Asked.run, .{q});
        // The unload has marked the model before the other request asks for it.
        if (i == 0) while (!unloadPending()) : (waited += 10) {
            if (waited >= ms) break;
            var none = [_]std.posix.pollfd{};
            _ = std.posix.poll(&none, 10) catch {};
        };
    }
    var unclaimed: u32 = 0;
    while (!(asked[0].done.load(.acquire) and asked[1].done.load(.acquire)) and waited < ms and unclaimed < claim_ms) : (waited += 10) {
        var none = [_]std.posix.pollfd{};
        _ = std.posix.poll(&none, 10) catch {};
        const behind_waits = asked[0].done.load(.acquire) and !asked[1].done.load(.acquire);
        unclaimed = if (behind_waits and tinyState() == .unloaded) unclaimed + 10 else 0;
    }
    const wedged = !asked[1].done.load(.acquire);
    // The stall ends here either way, so the threads end.
    _ = std.c.shutdown(stalled, std.posix.SHUT.RDWR);
    for (threads) |t| t.join();
    if (wedged) {
        std.debug.print("{d} ms behind a stalled reader: the unload done={any} ({s}), the request behind it done={any}, its model unclaimed {d} ms\n", .{ waited, asked[0].done.load(.acquire), asked[0].out[0..@min(asked[0].out.len, 200)], asked[1].done.load(.acquire), unclaimed });
        for (asked) |q| gpa.free(q.out);
        return error.WedgedByAStalledReader;
    }
    return .{ asked[0].out, asked[1].out };
}

fn unloadPending() bool {
    return tinyState() == .evicting;
}

fn tinyState() @import("model_registry.zig").LoadState {
    const reg = server_state.global_registry.?;
    reg.mutex.lockUncancelable(reg.io);
    defer reg.mutex.unlock(reg.io);
    return reg.entries.get("tiny").?.state;
}

test "server: a client that stops reading its stream gives up its model, so an unload behind it and every request behind that go on" {
    const was = http_conn.send_limit_ms;
    http_conn.send_limit_ms = 200;
    defer http_conn.send_limit_ms = was;
    try eachArch(4, .{}, struct {
        fn run() !void {
            const got = try behindAStalledReader(120_000);
            defer for (got) |g| gpa.free(g);
            if (!std.mem.startsWith(u8, got[0], "HTTP/1.1 200")) return error.UnloadFailed;
            if (!std.mem.startsWith(u8, got[1], "HTTP/1.1 200")) return error.RequestBehindFailed;
        }
    }.run);
}

test "server: an unload whose model stays held past its drain bound is refused by name, and the request behind it is served" {
    const was = scheduler_mod.Scheduler.unload_drain_ms;
    scheduler_mod.Scheduler.unload_drain_ms = 200;
    defer scheduler_mod.Scheduler.unload_drain_ms = was;
    try eachArch(4, .{}, struct {
        fn run() !void {
            const got = try behindAStalledReader(120_000);
            defer for (got) |g| gpa.free(g);
            if (!std.mem.startsWith(u8, got[0], "HTTP/1.1 409") or std.mem.indexOf(u8, got[0], "model_busy") == null) return error.UnloadNotRefusedByName;
            if (!std.mem.startsWith(u8, got[1], "HTTP/1.1 200")) return error.RequestBehindFailed;
        }
    }.run);
}

test "server: a shutdown during an unload's drain ends the unload and the request waiting behind it with the shutdown error" {
    const was = http_conn.send_limit_ms;
    http_conn.send_limit_ms = 3000;
    defer http_conn.send_limit_ms = was;
    try eachArch(4, .{}, struct {
        var asked: [2]Asked = undefined;
        fn bothDone() bool {
            return asked[0].done.load(.acquire) and asked[1].done.load(.acquire);
        }
        fn run() !void {
            var b: [2][512]u8 = undefined;
            gpa.free(try request("POST", "/v1/completions", try completionBody(&b[0], "hi", 2, false)));
            // A stream the client stops reading holds the model through the unload's drain.
            const stalled = try connect();
            defer _ = std.c.close(stalled);
            try send(stalled, "POST", "/v1/completions", try completionBody(&b[1], "a long tale of the sea", 100, true));
            gpa.free(try readUntil(stalled, "\"choices\""));
            asked = .{ .{ .path = "/v1/unload-model", .body = "{\"model\":\"tiny\"}" }, .{ .path = "/v1/completions", .body = try completionBody(&b[0], "hello", 4, false) } };
            const unload = try std.Thread.spawn(.{}, Asked.run, .{&asked[0]});
            const marked = within(5000, unloadPending);
            const behind = try std.Thread.spawn(.{}, Asked.run, .{&asked[1]});
            // The request reaches its wait behind the unload before the shutdown is asked.
            var none = [_]std.posix.pollfd{};
            _ = std.posix.poll(&none, 300) catch {};
            server.requestShutdown();
            const answered = within(2000, bothDone);
            _ = std.c.shutdown(stalled, std.posix.SHUT.RDWR);
            unload.join();
            behind.join();
            defer for (asked) |q| gpa.free(q.out);
            if (!marked) return error.UnloadNeverMarked;
            if (!answered) return error.ShutdownLeftAWaiterUnanswered;
            // One shutdown error: the unload and the request behind it carry the same type too.
            const shutdown_type = "\"type\":\"" ++ http.shutting_down_type ++ "\"";
            for (asked) |q| if (!std.mem.startsWith(u8, q.out, "HTTP/1.1 503") or std.mem.indexOf(u8, q.out, http.shutting_down_message) == null or
                std.mem.indexOf(u8, q.out, shutdown_type) == null)
            {
                std.debug.print("{s} answered at the shutdown: {s}\n", .{ q.path, q.out[0..@min(q.out.len, 300)] });
                return error.NotTheShutdownError;
            };
        }
    }.run);
}

test "server: a load that fails after evicting the model in use leaves no view of the evicted model, so its retry serves" {
    const ram_was = server_state.prefix_cache_ram_enabled;
    defer server_state.prefix_cache_ram_enabled = ram_was;
    const Scenario = struct {
        var failed_loads: u32 = 0;
    };
    for ([_]bool{ false, true }) |ram| {
        server_state.prefix_cache_ram_enabled = ram;
        try eachArch(4, .{ .tiny2 = true, .max_resident = 1 }, struct {
            fn run() !void {
                var b: [512]u8 = undefined;
                gpa.free(try request("POST", "/v1/completions", try completionBody(&b, "the river runs to the sea and on and on", 8, false)));
                const sch = server_state.global_scheduler.?;
                // The load of tiny2 evicts tiny, then fails at its first pool buffer.
                mlx.engine.fault.arm(1);
                const failed_load = try request("POST", "/v1/completions", "{\"model\":\"tiny2\",\"prompt\":\"hi\",\"max_tokens\":2}");
                defer gpa.free(failed_load);
                const fired = !mlx.engine.fault.armed();
                mlx.engine.fault.disarm();
                if (!fired) return error.FaultNeverFired;
                // An arch whose load absorbs that refusal (Flash-Next's does) has no failed load to look past; the count
                // below fails the test when every arch absorbs it.
                if (std.mem.startsWith(u8, failed_load, "HTTP/1.1 200")) return;
                Scenario.failed_loads += 1;
                const reg = server_state.global_registry.?;
                reg.mutex.lockUncancelable(reg.io);
                const tiny = reg.entries.get("tiny").?;
                const evicted = tiny.state == .unloaded;
                reg.mutex.unlock(reg.io);
                if (!evicted) return error.VictimNotEvicted;
                // The inference loop runs its post-load revise between jobs.
                var none = [_]std.posix.pollfd{};
                _ = std.posix.poll(&none, 200) catch {};
                if (sch.current_model == tiny or sch.hot_prefix_cache != null) return error.EvictedModelStillInView;
                if (sch.resident_hot_cache_bytes.load(.monotonic) != 0) return error.EvictedModelHotCacheCredited;
                gpa.free(try request("POST", "/v1/load-model", "{\"model\":\"tiny2\"}"));
                const retry = try request("POST", "/v1/completions", "{\"model\":\"tiny2\",\"prompt\":\"hi\",\"max_tokens\":2}");
                defer gpa.free(retry);
                if (!std.mem.startsWith(u8, retry, "HTTP/1.1 200")) return error.RetryFailed;
            }
        }.run);
    }
    try testing.expect(Scenario.failed_loads > 0);
}

/// The next WebSocket frame the server sends (unmasked), its payload; the caller frees.
fn wsReadFrame(fd: std.posix.fd_t) ![]u8 {
    return (try wsReadFrameOf(fd)).payload;
}

/// The next WebSocket frame the server sends, its opcode and payload; the caller frees the payload.
fn wsReadFrameOf(fd: std.posix.fd_t) !struct { opcode: u4, payload: []u8 } {
    var head: [10]u8 = undefined;
    try readExact(fd, head[0..2]);
    var len: u64 = head[1] & 0x7f;
    if (len == 126) {
        try readExact(fd, head[2..4]);
        len = std.mem.readInt(u16, head[2..4], .big);
    } else if (len == 127) {
        try readExact(fd, head[2..10]);
        len = std.mem.readInt(u64, head[2..10], .big);
    }
    const payload = try gpa.alloc(u8, @intCast(len));
    errdefer gpa.free(payload);
    try readExact(fd, payload);
    return .{ .opcode = @truncate(head[0]), .payload = payload };
}

fn readExact(fd: std.posix.fd_t, out: []u8) !void {
    var got: usize = 0;
    while (got < out.len) {
        const n = std.c.read(fd, out[got..].ptr, out.len - got);
        if (n <= 0) return error.ConnectionEnded;
        got += @intCast(n);
    }
}

/// Declares an `iogpu.wired_limit_mb` whose floor (the limit less the margin) is `floor` bytes, refused when the bill
/// would read another: a cell under a floor it does not have tests nothing. The limit is the box's RAM, since a floor
/// is also at most the RAM less the margin (`bills.wiredLimitFloor`). `clearWiredFloor` restores both.
fn setWiredFloor(floor: u64) !void {
    const total = @import("status.zig").getTotalMemBytes();
    const limit_mb: u64 = total >> 20;
    bills.wired_limit_mb_override = limit_mb;
    server_state.wired_margin_flag_bytes = (limit_mb << 20) - floor;
    const unplanned = bills.wiredUnplannedBytes(server_state.wiredMarginBytes(), bills.gpu_floor_bytes.load(.monotonic));
    if (bills.wiredLimitFloor(bills.wiredLimitBytes(), total, unplanned) == floor) return;
    clearWiredFloor();
    return error.WiredFloorNotDeclared;
}

fn clearWiredFloor() void {
    bills.wired_limit_mb_override = null;
    server_state.wired_margin_flag_bytes = null;
}

test "server: a Responses WebSocket turn refused before it runs ends in the error frame, whichever check refused it" {
    try eachArch(4, .{}, struct {
        const Case = struct { name: []const u8, frame: []const u8 };
        const long_input = blk: {
            var s: []const u8 = "";
            for (0..300) |_| s = s ++ "salmon ";
            break :blk s;
        };
        const cases = [_]Case{
            .{ .name = "no input", .frame = "{\"type\":\"response.create\",\"model\":\"tiny\"}" },
            .{ .name = "schema without a schema", .frame = "{\"type\":\"response.create\",\"model\":\"tiny\",\"input\":\"hi\",\"text\":{\"format\":{\"type\":\"json_schema\",\"name\":\"x\"}}}" },
            .{ .name = "effort not a word", .frame = "{\"type\":\"response.create\",\"model\":\"tiny\",\"input\":\"hi\",\"reasoning\":{\"effort\":5}}" },
            .{ .name = "tool_choice undeclared", .frame = "{\"type\":\"response.create\",\"model\":\"tiny\",\"input\":\"hi\",\"tools\":[{\"type\":\"function\",\"name\":\"f\",\"parameters\":{\"type\":\"object\"}}],\"tool_choice\":{\"type\":\"function\",\"name\":\"g\"}}" },
            .{ .name = "media refused", .frame = "{\"type\":\"response.create\",\"model\":\"tiny\",\"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_image\",\"image_url\":\"data:image/png;base64,AAAA\"}]}]}" },
            .{ .name = "context overflow", .frame = "{\"type\":\"response.create\",\"model\":\"tiny\",\"input\":\"" ++ long_input ++ "\"}" },
            .{ .name = "memory admission", .frame = "{\"type\":\"response.create\",\"model\":\"tiny\",\"input\":\"hi\",\"max_output_tokens\":8,\"temperature\":0}" },
        };
        fn run() !void {
            const fd = try wsConnect();
            defer _ = std.c.close(fd);
            defer clearWiredFloor();
            const ctx_was = server_state.server_config.max_context_size;
            defer server_state.server_config.max_context_size = ctx_was;
            for (cases) |c| {
                if (std.mem.eql(u8, c.name, "memory admission")) try setWiredFloor(1 << 20);
                server_state.server_config.max_context_size = if (std.mem.eql(u8, c.name, "context overflow")) 64 else ctx_was;
                try wsSend(fd, c.frame);
                const frame = try wsReadFrame(fd);
                defer gpa.free(frame);
                if (!std.mem.startsWith(u8, frame, "{\"type\":\"error\",\"status\":4")) {
                    std.debug.print("refused turn ({s}) ended in: {s}\n", .{ c.name, frame[0..@min(frame.len, 400)] });
                    return error.RefusalNotTheTurnsErrorFrame;
                }
            }
            clearWiredFloor();
            // The connection serves the next turn.
            try wsSend(fd, "{\"type\":\"response.create\",\"model\":\"tiny\",\"input\":\"hi\",\"max_output_tokens\":2,\"temperature\":0}");
            while (true) {
                const frame = try wsReadFrame(fd);
                defer gpa.free(frame);
                if (hasResponsesTerminal(frame)) break;
                if (std.mem.startsWith(u8, frame, "{\"type\":\"error\"")) return error.NextTurnFailed;
            }
        }
    }.run);
}

/// The body of an HTTP answer.
fn bodyOf(resp: []const u8) []const u8 {
    const at = std.mem.indexOf(u8, resp, "\r\n\r\n") orelse return "";
    return resp[at + 4 ..];
}

test "server: an error on an Anthropic route takes Anthropic's envelope and status line, whichever check sent it" {
    try eachArch(4, .{}, struct {
        /// `status` is the head's status line, or its start.
        const Case = struct { name: []const u8, method: []const u8 = "POST", path: []const u8, body: []const u8, status: []const u8, anthropic: bool = true };
        const messages = "{\"model\":\"tiny\",\"max_tokens\":4,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
        const nested = blk: {
            var s: []const u8 = "";
            for (0..300) |_| s = s ++ "[";
            break :blk s;
        };
        fn check(c: Case) !void {
            const resp = try request(c.method, c.path, c.body);
            defer gpa.free(resp);
            var want: [64]u8 = undefined;
            const head = try std.fmt.bufPrint(&want, "HTTP/1.1 {s}", .{c.status});
            const shape = if (c.anthropic) "{\"type\":\"error\",\"error\":{\"type\":\"" else "{\"error\":{\"message\":";
            if (!std.mem.startsWith(u8, resp, head) or !std.mem.startsWith(u8, bodyOf(resp), shape)) {
                std.debug.print("{s}: {s}\n", .{ c.name, resp[0..@min(resp.len, 500)] });
                return error.WrongEnvelope;
            }
        }
        fn run() !void {
            // Refused loads, by whichever check refuses first: the model cannot fit under a declared wired limit's floor.
            try setWiredFloor(64 << 20);
            errdefer clearWiredFloor();
            const refused = [_]Case{
                .{ .name = "load refused, messages", .path = "/v1/messages", .body = messages, .status = "5" },
                .{ .name = "load refused, count_tokens", .path = "/v1/messages/count_tokens", .body = messages, .status = "5" },
                .{ .name = "load refused, chat", .path = "/v1/chat/completions", .body = "{\"model\":\"tiny\",\"max_tokens\":4,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", .status = "5", .anthropic = false },
            };
            for (refused) |c| try check(c);
            clearWiredFloor();
            const served = [_]Case{
                .{ .name = "method", .method = "GET", .path = "/v1/messages", .body = "", .status = "405 Method Not Allowed" },
                .{ .name = "nesting", .path = "/v1/messages", .body = "{\"model\":\"tiny\",\"x\":" ++ nested ++ "}", .status = "400 Bad Request" },
                .{ .name = "no messages", .path = "/v1/messages", .body = "{\"model\":\"tiny\",\"max_tokens\":4}", .status = "400 Bad Request" },
            };
            for (served) |c| try check(c);
        }
    }.run);
}

test "server: a completions stream that fails mid-reply ends in a completions chunk, not a chat delta" {
    try eachArch(4, .{}, struct {
        fn run() !void {
            defer mlx.engine.fault.disarm();
            var b: [512]u8 = undefined;
            gpa.free(try request("POST", "/v1/completions", try completionBody(&b, "hi", 2, false)));
            const fd = try connect();
            defer _ = std.c.close(fd);
            try send(fd, "POST", "/v1/completions", try completionBody(&b, "a long tale of the sea", 64, true));
            const head = try readUntil(fd, "\"choices\"");
            defer gpa.free(head);
            mlx.engine.fault.arm(1);
            const rest = try readUntil(fd, null);
            defer gpa.free(rest);
            const at = std.mem.lastIndexOf(u8, rest, "\"finish_reason\":\"error\"") orelse {
                std.debug.print("no error chunk: {s}\n", .{rest[rest.len -| 400..]});
                return error.NoErrorChunk;
            };
            const line = rest[(std.mem.lastIndexOf(u8, rest[0..at], "data: ") orelse 0)..at];
            if (std.mem.indexOf(u8, line, "\"delta\"") != null or std.mem.indexOf(u8, line, "\"text\":\"\"") == null) {
                std.debug.print("error chunk: {s}\n", .{line});
                return error.NotACompletionsChunk;
            }
        }
    }.run);
}

test "server: a WebSocket turn's response says the store the client asked for" {
    try eachArch(4, .{}, struct {
        fn run() !void {
            const fd = try wsConnect();
            defer _ = std.c.close(fd);
            for ([_][2][]const u8{ .{ ",\"store\":false", "\"store\":false" }, .{ "", "\"store\":true" } }) |c| {
                var b: [256]u8 = undefined;
                try wsSend(fd, try std.fmt.bufPrint(&b, "{{\"type\":\"response.create\",\"model\":\"tiny\",\"input\":\"hi\",\"max_output_tokens\":2,\"temperature\":0{s}}}", .{c[0]}));
                while (true) {
                    const frame = try wsReadFrame(fd);
                    defer gpa.free(frame);
                    if (std.mem.startsWith(u8, frame, "{\"type\":\"error\"")) return error.TurnFailed;
                    const created = std.mem.indexOf(u8, frame, "\"type\":\"response.created\"") != null;
                    const completed = hasResponsesTerminal(frame);
                    if ((created or completed) and std.mem.indexOf(u8, frame, c[1]) == null) {
                        std.debug.print("asked{s}: {s}\n", .{ c[0], frame[0..@min(frame.len, 600)] });
                        return error.StoreNotAsAsked;
                    }
                    if (completed) break;
                }
            }
        }
    }.run);
}

/// The `error.message` of a JSON error answer; the caller frees.
fn errorMessage(resp: []const u8) ![]u8 {
    const E = struct { @"error": struct { message: []const u8 } };
    const parsed = try std.json.parseFromSlice(E, gpa, bodyOf(resp), .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    return gpa.dupe(u8, parsed.value.@"error".message);
}

test "server: every request to a model whose load is refused for memory gets the whole refusal, re-read, as a named 503" {
    try eachArch(4, .{}, struct {
        fn run() !void {
            try setWiredFloor(64 << 20);
            defer clearWiredFloor();
            const chat = "{\"model\":\"tiny\",\"max_tokens\":4,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
            var first: ?[]u8 = null;
            defer if (first) |f| gpa.free(f);
            for ([_][2][]const u8{ .{ "/v1/load-model", "{\"model\":\"tiny\"}" }, .{ "/v1/chat/completions", chat }, .{ "/v1/chat/completions", chat } }) |c| {
                const resp = try request("POST", c[0], c[1]);
                defer gpa.free(resp);
                const message = try errorMessage(resp);
                errdefer gpa.free(message);
                errdefer std.debug.print("{s}: {s}\n", .{ c[0], resp[0..@min(resp.len, 600)] });
                if (!std.mem.startsWith(u8, resp, "HTTP/1.1 503")) return error.NotANamedRefusal;
                // The text the load's check wrote, its lines whole: a fixed fallback is one line.
                if (std.mem.indexOfScalar(u8, message, '\n') == null) return error.FallbackText;
                if (first) |f| {
                    const head = f[0 .. std.mem.indexOfScalar(u8, f, '\n') orelse f.len];
                    if (!std.mem.startsWith(u8, message, head)) return error.AnotherRefusal;
                    gpa.free(message);
                } else first = message;
            }
        }
    }.run);
}

/// The close code of the next frame, which must be a close.
fn wsReadCloseCode(fd: std.posix.fd_t) !u16 {
    const frame = try wsReadFrameOf(fd);
    defer gpa.free(frame.payload);
    if (frame.opcode != 0x8 or frame.payload.len < 2) {
        std.debug.print("expected a close frame, got opcode {d}: {s}\n", .{ frame.opcode, frame.payload[0..@min(frame.payload.len, 300)] });
        return error.NotAClose;
    }
    return std.mem.readInt(u16, frame.payload[0..2], .big);
}

/// One completed turn on `fd`, which leaves the WebSocket idle between turns.
fn wsCompletedTurn(fd: std.posix.fd_t) !void {
    try wsSend(fd, "{\"type\":\"response.create\",\"model\":\"tiny\",\"input\":\"hi\",\"max_output_tokens\":2,\"temperature\":0}");
    while (true) {
        const frame = try wsReadFrame(fd);
        defer gpa.free(frame);
        if (std.mem.startsWith(u8, frame, "{\"type\":\"error\"")) return error.TurnFailed;
        if (hasResponsesTerminal(frame)) return;
    }
}

test "server: a Responses WebSocket message sent as a binary frame is answered by an error frame and a 1003 close, never silence" {
    try eachArch(4, .{}, struct {
        fn run() !void {
            const fd = try wsConnect();
            defer _ = std.c.close(fd);
            const text = "{\"type\":\"response.create\",\"model\":\"tiny\",\"input\":\"hi\",\"max_output_tokens\":2,\"temperature\":0}";
            // One masked binary frame (opcode 2), a zero mask.
            try sendAll(fd, &.{ 0x82, 0x80 | @as(u8, @intCast(text.len)), 0, 0, 0, 0 });
            try sendAll(fd, text);
            const answer = try wsReadFrameOf(fd);
            defer gpa.free(answer.payload);
            if (answer.opcode != 0x1 or !std.mem.startsWith(u8, answer.payload, "{\"type\":\"error\",\"status\":400")) {
                std.debug.print("binary turn answered by opcode {d}: {s}\n", .{ answer.opcode, answer.payload[0..@min(answer.payload.len, 300)] });
                return error.NoErrorFrame;
            }
            if (try wsReadCloseCode(fd) != 1003) return error.NotUnsupportedData;
        }
    }.run);
}

test "server: a WebSocket upgrade the server cannot complete is answered 400, never closed bare" {
    try eachArch(4, .{}, struct {
        fn run() !void {
            const fd = try connect();
            defer _ = std.c.close(fd);
            // No Sec-WebSocket-Key.
            try sendAll(fd, "GET /v1/responses HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Version: 13\r\n\r\n");
            const answer = try readUntil(fd, null);
            defer gpa.free(answer);
            if (!std.mem.startsWith(u8, answer, "HTTP/1.1 400")) {
                std.debug.print("upgrade without a key: {d} bytes: {s}\n", .{ answer.len, answer[0..@min(answer.len, 400)] });
                return error.BareClose;
            }
        }
    }.run);
}

/// A read on `fd` that waits past `sec` fails instead of blocking.
fn readLimit(fd: std.posix.fd_t, sec: i64) void {
    const limit: std.posix.timeval = .{ .sec = @intCast(sec), .usec = 0 };
    _ = std.c.setsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.RCVTIMEO, std.mem.asBytes(&limit), @sizeOf(std.posix.timeval));
}

test "server: a shutdown closes a WebSocket idle between turns with 1001 and an idle connection at once, and serve runs its teardown" {
    try eachArch(4, .{}, struct {
        fn run() !void {
            // Between turns, as Codex keeps one.
            const ws = try wsConnect();
            defer _ = std.c.close(ws);
            try wsCompletedTurn(ws);
            // A connection that has sent nothing yet (a browser's preconnect).
            const idle = try connect();
            defer _ = std.c.close(idle);
            // Well under the request read limit (30 s) and the drain bound (35 s) that bounded these waits before.
            readLimit(ws, 10);
            readLimit(idle, 10);
            server.requestShutdown();
            if (try wsReadCloseCode(ws) != 1001) return error.NotGoingAway;
            var b: [64]u8 = undefined;
            if (std.c.read(idle, &b, b.len) != 0) return error.IdleConnectionOutlivesShutdown;
            // serve returns `ConnectionsOutliveShutdown` when a connection thread outlives the drain.
        }
    }.run);
}

test "server: a WebSocket idle between turns does not hold its model against an unload" {
    const was = scheduler_mod.Scheduler.unload_drain_ms;
    scheduler_mod.Scheduler.unload_drain_ms = 2000;
    defer scheduler_mod.Scheduler.unload_drain_ms = was;
    try eachArch(4, .{}, struct {
        fn run() !void {
            const ws = try wsConnect();
            defer _ = std.c.close(ws);
            try wsCompletedTurn(ws);
            const un = try request("POST", "/v1/unload-model", "{\"model\":\"tiny\"}");
            defer gpa.free(un);
            if (!std.mem.startsWith(u8, un, "HTTP/1.1 200")) {
                std.debug.print("unload beside an idle WebSocket: {s}\n", .{un[0..@min(un.len, 500)]});
                return error.IdleWebSocketPinsModel;
            }
            // The next turn loads the model again.
            try wsCompletedTurn(ws);
        }
    }.run);
}

test "server: a WebSocket turn routes its own model: an unregistered model path is refused in its error frame, as its HTTP route refuses it" {
    try eachArch(4, .{}, struct {
        fn run() !void {
            const http_answer = try request("POST", "/v1/responses", "{\"model\":\"/nonexistent/sushi-test/model\",\"input\":\"hi\",\"max_output_tokens\":2,\"temperature\":0}");
            defer gpa.free(http_answer);
            if (!std.mem.startsWith(u8, http_answer, "HTTP/1.1 404")) return error.HttpRouteServedIt;
            const fd = try wsConnect();
            defer _ = std.c.close(fd);
            try wsSend(fd, "{\"type\":\"response.create\",\"model\":\"/nonexistent/sushi-test/model\",\"input\":\"hi\",\"max_output_tokens\":2,\"temperature\":0}");
            const frame = try wsReadFrame(fd);
            defer gpa.free(frame);
            if (!std.mem.startsWith(u8, frame, "{\"type\":\"error\",\"status\":404")) {
                std.debug.print("turn for an unregistered path answered: {s}\n", .{frame[0..@min(frame.len, 300)]});
                return error.TurnModelNotRouted;
            }
            // The socket serves the next turn.
            try wsCompletedTurn(fd);
        }
    }.run);
}

/// Holds the inference thread in the next request's prefill, before its first forward (`prefill_request_chunk`),
/// until the test lets go: a request already decoding stands still between two tokens, the held one before its first.
const PrefillHold = struct {
    const ChunkFn = @typeInfo(@TypeOf(scheduler_mod.prefill_request_chunk)).optional.child;
    var served: ?ChunkFn = null;
    var armed: std.atomic.Value(bool) = .init(false);
    var held: std.atomic.Value(bool) = .init(false);
    var released: std.atomic.Value(bool) = .init(false);

    fn chunk(cfg: *const model_mod.ModelConfig, n: usize, max: u32, kv: trunk.KVQuantConfig, a: bool, b: u64, c: u64, d: bool, e: bool) u32 {
        if (armed.swap(false, .acq_rel)) {
            held.store(true, .release);
            // Bounds a test that fails before it lets go; the test itself releases at once.
            var waited: u32 = 0;
            while (!released.load(.acquire) and waited < 120_000) : (waited += 1) {
                var none = [_]std.posix.pollfd{};
                _ = std.posix.poll(&none, 1) catch {};
            }
        }
        return served.?(cfg, n, max, kv, a, b, c, d, e);
    }
    /// After a first request: serve installs the hook this one wraps.
    fn install() !void {
        served = scheduler_mod.prefill_request_chunk orelse return error.NoPrefillChunkHook;
        scheduler_mod.prefill_request_chunk = &chunk;
    }
    fn uninstall() void {
        released.store(true, .release);
        scheduler_mod.prefill_request_chunk = served;
    }
    fn arm() void {
        held.store(false, .release);
        released.store(false, .release);
        armed.store(true, .release);
    }
    fn isHeld() bool {
        return held.load(.acquire);
    }
    fn waitHeld() !void {
        if (!within(60_000, isHeld)) return error.PrefillNeverHeld;
    }
    fn release() void {
        released.store(true, .release);
    }
};

/// The one slot a scheduler list holds, read under its locks; null unless it holds exactly one.
const SlotSeen = struct { id: u64, tokens: usize, ended: bool };

fn onlySlot(comptime list: []const u8) ?SlotSeen {
    const sch = server_state.global_scheduler.?;
    sch.queue_mu.lockUncancelable(sch.io);
    defer sch.queue_mu.unlock(sch.io);
    const items = @field(sch, list).items;
    if (items.len != 1) return null;
    const s = items[0];
    s.out_mu.lockUncancelable(s.io);
    defer s.out_mu.unlock(s.io);
    return .{ .id = s.request_id, .tokens = s.out_buf.items.len, .ended = s.finished or s.error_code != null };
}

fn dumpLists(tag: []const u8) void {
    const sch = server_state.global_scheduler.?;
    sch.queue_mu.lockUncancelable(sch.io);
    defer sch.queue_mu.unlock(sch.io);
    std.debug.print("{s}: pending {d} prefilling {d} decoding {d} in_flight {d} held {any}\n", .{ tag, sch.pending.items.len, sch.prefilling.items.len, sch.decoding.items.len, sch.in_flight, PrefillHold.isHeld() });
}

fn pendingOne() bool {
    return onlySlot("pending") != null;
}

fn decodingWithTokens() bool {
    const seen = onlySlot("decoding") orelse return false;
    return seen.tokens > 0;
}

/// A request run on its own thread, for one whose answer waits on the test.
const Background = struct {
    path: []const u8,
    body: []const u8,
    out: []u8 = &.{},
    fn run(q: *@This()) void {
        q.out = request("POST", q.path, q.body) catch &.{};
    }
};

const Surface = enum { chat, completions, messages, responses, ws };

const CancelRow = struct {
    surface: Surface,
    stream: bool,
    path: []const u8,
    body: []const u8,
};

const cancel_rows = [_]CancelRow{
    .{ .surface = .chat, .stream = true, .path = "/v1/chat/completions", .body = "{\"model\":\"tiny\",\"max_tokens\":2000,\"temperature\":0,\"stream\":true,\"stream_options\":{\"include_usage\":true},\"messages\":[{\"role\":\"user\",\"content\":\"write a long story about rivers\"}]}" },
    .{ .surface = .chat, .stream = false, .path = "/v1/chat/completions", .body = "{\"model\":\"tiny\",\"max_tokens\":2000,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"write a long story about rivers\"}]}" },
    .{ .surface = .completions, .stream = true, .path = "/v1/completions", .body = "{\"model\":\"tiny\",\"prompt\":\"a long tale of the sea\",\"max_tokens\":64,\"temperature\":0,\"ignore_eos\":true,\"stream\":true,\"stream_options\":{\"include_usage\":true}}" },
    .{ .surface = .completions, .stream = false, .path = "/v1/completions", .body = "{\"model\":\"tiny\",\"prompt\":\"a long tale of the sea\",\"max_tokens\":64,\"temperature\":0,\"ignore_eos\":true}" },
    .{ .surface = .messages, .stream = true, .path = "/v1/messages", .body = "{\"model\":\"tiny\",\"max_tokens\":2000,\"temperature\":0,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"write a long story about rivers\"}]}" },
    .{ .surface = .messages, .stream = false, .path = "/v1/messages", .body = "{\"model\":\"tiny\",\"max_tokens\":2000,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"write a long story about rivers\"}]}" },
    .{ .surface = .responses, .stream = true, .path = "/v1/responses", .body = "{\"model\":\"tiny\",\"max_output_tokens\":2000,\"temperature\":0,\"stream\":true,\"input\":\"write a long story about rivers\"}" },
    .{ .surface = .responses, .stream = false, .path = "/v1/responses", .body = "{\"model\":\"tiny\",\"max_output_tokens\":2000,\"temperature\":0,\"input\":\"write a long story about rivers\"}" },
    .{ .surface = .ws, .stream = true, .path = "", .body = "{\"type\":\"response.create\",\"model\":\"tiny\",\"input\":\"write a long story about rivers\",\"max_output_tokens\":2000,\"temperature\":0}" },
};

/// `connect` with the buffers large, as a TCP socket's are: a stream writes its opening events before the test reads.
fn connectWide() !std.posix.fd_t {
    var sv: [2]std.posix.fd_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &sv) != 0) return error.SocketPairFailed;
    const wide: c_int = 256 * 1024;
    _ = std.c.setsockopt(sv[0], std.posix.SOL.SOCKET, std.posix.SO.SNDBUF, std.mem.asBytes(&wide), @sizeOf(c_int));
    _ = std.c.setsockopt(sv[1], std.posix.SOL.SOCKET, std.posix.SO.RCVBUF, std.mem.asBytes(&wide), @sizeOf(c_int));
    readLimit(sv[1], 120);
    server.spawnConnection(gpa, io, .{ .socket = .{ .handle = sv[0], .address = .{ .ip4 = .loopback(0) } } });
    return sv[1];
}

/// One in-flight request of a row: a socket for a stream, a thread for an answer the test waits on.
const InFlight = struct {
    row: CancelRow,
    fd: std.posix.fd_t = -1,
    asked: Background = .{ .path = "", .body = "" },
    thread: ?std.Thread = null,

    fn start(row: CancelRow) !InFlight {
        var f = InFlight{ .row = row };
        if (row.surface == .ws) {
            f.fd = try connectWide();
            errdefer _ = std.c.close(f.fd);
            try sendAll(f.fd, "GET /v1/responses HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n");
            const head = try readUntil(f.fd, "\r\n\r\n");
            defer gpa.free(head);
            if (!std.mem.startsWith(u8, head, "HTTP/1.1 101")) return error.NoUpgrade;
            try wsSend(f.fd, row.body);
        } else if (row.stream) {
            f.fd = try connectWide();
            try send(f.fd, "POST", row.path, row.body);
        }
        return f;
    }
    /// A non-stream request's thread starts once `f` sits where it stays.
    fn spawn(f: *InFlight) !void {
        if (f.fd >= 0) return;
        f.asked = .{ .path = f.row.path, .body = f.row.body };
        f.thread = try std.Thread.spawn(.{}, Background.run, .{&f.asked});
    }
    /// Everything the request answered, to its terminal; the caller frees.
    fn finish(f: *InFlight) ![]u8 {
        if (f.thread) |t| {
            t.join();
            return f.asked.out;
        }
        defer _ = std.c.close(f.fd);
        if (f.row.surface != .ws) return readUntil(f.fd, null);
        var all: std.ArrayList(u8) = .empty;
        errdefer all.deinit(gpa);
        while (true) {
            const frame = try wsReadFrameOf(f.fd);
            defer gpa.free(frame.payload);
            if (frame.opcode == 0x8) break;
            try all.appendSlice(gpa, frame.payload);
            try all.append(gpa, '\n');
            if (wsTerminal(frame.payload)) break;
        }
        return all.toOwnedSlice(gpa);
    }
};

fn wsTerminal(frame: []const u8) bool {
    return std.mem.indexOf(u8, frame, "\"type\":\"response.completed\"") != null or
        std.mem.indexOf(u8, frame, "\"type\":\"response.failed\"") != null or
        std.mem.indexOf(u8, frame, "\"type\":\"response.incomplete\"") != null or
        std.mem.startsWith(u8, frame, "{\"type\":\"error\"");
}

/// The number after the LAST `key` in `text` (the terminal's usage).
fn lastNumber(text: []const u8, key: []const u8) ?u64 {
    const at = std.mem.lastIndexOf(u8, text, key) orelse return null;
    const rest = text[at + key.len ..];
    var n: usize = 0;
    while (n < rest.len and std.ascii.isDigit(rest[n])) n += 1;
    return std.fmt.parseInt(u64, rest[0..n], 10) catch null;
}

/// What a row's answer must say once it was cancelled with `tokens` published: one terminal in the surface's own
/// cancelled wording, never a token cap, and usage that counts those tokens (and the prompt, once one was computed).
fn saysCancelled(row: CancelRow, answer: []const u8, tokens: usize, prompt_counted: bool) bool {
    const count = std.mem.count;
    const ok = switch (row.surface) {
        .chat, .completions => count(u8, answer, "\"finish_reason\":\"") == 1 and
            std.mem.indexOf(u8, answer, "\"finish_reason\":\"cancelled\"") != null and
            lastNumber(answer, "\"completion_tokens\":") == tokens and
            (!prompt_counted or lastNumber(answer, "\"prompt_tokens\":").? > 0) and
            (!row.stream or count(u8, answer, "data: [DONE]") == 1),
        .messages => count(u8, answer, "\"stop_reason\":\"") == 1 and
            std.mem.indexOf(u8, answer, "\"stop_reason\":\"end_turn\"") != null and
            lastNumber(answer, "\"output_tokens\":") == tokens and
            (!prompt_counted or lastNumber(answer, "\"input_tokens\":").? + lastNumber(answer, "\"cache_read_input_tokens\":").? > 0) and
            (!row.stream or count(u8, answer, "event: message_stop") == 1),
        .responses, .ws => std.mem.indexOf(u8, answer, "max_output_tokens\"}") == null and
            std.mem.indexOf(u8, answer, "\"incomplete_details\":null") != null and
            lastNumber(answer, "\"output_tokens\":") == tokens and
            (!prompt_counted or lastNumber(answer, "\"input_tokens\":").? + lastNumber(answer, "\"cached_tokens\":").? > 0) and
            (!row.stream or count(u8, answer, "\"type\":\"response.completed\"") == 1) and
            count(u8, answer, "\"type\":\"error\"") == 0,
    };
    if (!ok) {
        std.debug.print("{s} {s} cancelled with {d} tokens published answered: {s}\n", .{ @tagName(row.surface), if (row.stream) "stream" else "non-stream", tokens, answer[answer.len -| 1200..] });
    }
    return ok;
}

fn cancelById(id: u64) !void {
    var b: [64]u8 = undefined;
    const c = try request("POST", "/v1/slots/cancel", try std.fmt.bufPrint(&b, "{{\"id\":{d}}}", .{id}));
    defer gpa.free(c);
    if (!std.mem.startsWith(u8, c, "HTTP/1.1 200")) return error.CancelMissed;
}

/// The blocker whose prefill the hold stands in, while the request under test decodes.
const blocker_body = "{\"model\":\"tiny\",\"max_tokens\":1,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";

test "server: a request cancelled before its first token or mid-reply ends on every surface in one terminal that says cancelled, with usage counting what it returned" {
    try eachArch(4, .{ .long_replies = true }, struct {
        fn run() !void {
            gpa.free(try request("POST", "/v1/chat/completions", blocker_body));
            try PrefillHold.install();
            defer PrefillHold.uninstall();
            var wrong: u32 = 0;
            for ([_]bool{ true, false }) |queued| for (cancel_rows) |row| {
                // The cancel lands while the inference thread stands in the blocker's prefill: before the request's
                // first forward (queued behind it), or mid-reply between two tokens.
                var blocked = Background{ .path = "/v1/chat/completions", .body = blocker_body };
                var f = InFlight{ .row = row };
                var blocker: ?std.Thread = null;
                defer if (blocker) |t| {
                    t.join();
                    gpa.free(blocked.out);
                };
                defer PrefillHold.release();
                errdefer if (f.fd >= 0) {
                    _ = std.c.shutdown(f.fd, std.posix.SHUT.RDWR);
                };
                if (queued) {
                    PrefillHold.arm();
                    blocker = try std.Thread.spawn(.{}, Background.run, .{&blocked});
                    try PrefillHold.waitHeld();
                    f = try InFlight.start(row);
                    try f.spawn();
                    if (!within(20_000, pendingOne)) {
                        dumpLists(@tagName(row.surface));
                        return error.NeverQueued;
                    }
                } else {
                    f = try InFlight.start(row);
                    try f.spawn();
                    if (!within(60_000, decodingWithTokens)) {
                        dumpLists(@tagName(row.surface));
                        return error.NeverDecoded;
                    }
                    PrefillHold.arm();
                    blocker = try std.Thread.spawn(.{}, Background.run, .{&blocked});
                    try PrefillHold.waitHeld();
                }
                const seen = (if (queued) onlySlot("pending") else onlySlot("decoding")) orelse return error.NoSlotHeld;
                if (seen.ended) return error.EndedBeforeTheCancel;
                try cancelById(seen.id);
                PrefillHold.release();
                const answer = try f.finish();
                defer gpa.free(answer);
                if (!saysCancelled(row, answer, seen.tokens, !queued)) wrong += 1;
            };
            if (wrong > 0) return error.CancelMisreported;
        }
    }.run);
}

var decoding_want: usize = 0;

/// `decoding_want` slots decode, each with a token published.
fn allDecoding() bool {
    const sch = server_state.global_scheduler.?;
    sch.queue_mu.lockUncancelable(sch.io);
    defer sch.queue_mu.unlock(sch.io);
    if (sch.decoding.items.len != decoding_want) return false;
    for (sch.decoding.items) |s| {
        s.out_mu.lockUncancelable(s.io);
        defer s.out_mu.unlock(s.io);
        if (s.out_buf.items.len == 0) return false;
    }
    return true;
}

fn draining() bool {
    const sch = server_state.global_scheduler.?;
    sch.queue_mu.lockUncancelable(sch.io);
    defer sch.queue_mu.unlock(sch.io);
    return sch.draining;
}

/// One terminal, the surface's error naming the shutdown; never a token cap.
fn expectShutDown(row: CancelRow, answer: []const u8) !void {
    const count = std.mem.count;
    const says = std.mem.indexOf(u8, answer, http.shutting_down_message) != null;
    const ok = says and switch (row.surface) {
        .chat, .completions => count(u8, answer, "\"finish_reason\":\"") == 1 and
            std.mem.indexOf(u8, answer, "\"finish_reason\":\"error\"") != null,
        .messages => count(u8, answer, "event: error") == 1 and count(u8, answer, "\"stop_reason\":\"") == 0,
        .responses, .ws => count(u8, answer, "\"type\":\"error\"") == 1 and count(u8, answer, "\"type\":\"response.completed\"") == 0 and
            std.mem.indexOf(u8, answer, "max_output_tokens\"}") == null,
    };
    if (!ok) {
        std.debug.print("{s} stream at a shutdown mid-reply answered: {s}\n", .{ @tagName(row.surface), answer[answer.len -| 1200..] });
        return error.ShutdownMisreported;
    }
}

test "server: a shutdown mid-reply ends the stream of every surface in one terminal error naming it, never a token cap" {
    try eachArch(8, .{ .long_replies = true }, struct {
        fn run() !void {
            gpa.free(try request("POST", "/v1/chat/completions", blocker_body));
            try PrefillHold.install();
            defer PrefillHold.uninstall();
            // The long replies first; the completion, which ends within its context, joins them last.
            var flights: [5]InFlight = undefined;
            var n: usize = 0;
            // A test that fails before it reads lets the server's writes end, so serve's drain still ends.
            errdefer for (flights[0..n]) |f| if (f.fd >= 0) {
                _ = std.c.shutdown(f.fd, std.posix.SHUT.RDWR);
            };
            for ([_]bool{ false, true }) |short| {
                for (cancel_rows) |row| if (row.stream and (row.surface == .completions) == short) {
                    flights[n] = try InFlight.start(row);
                    n += 1;
                };
                decoding_want = n;
                if (!within(60_000, allDecoding)) {
                    dumpLists("shutdown");
                    return error.NeverDecoded;
                }
            }
            var blocked = Background{ .path = "/v1/chat/completions", .body = blocker_body };
            PrefillHold.arm();
            const blocker = try std.Thread.spawn(.{}, Background.run, .{&blocked});
            defer {
                blocker.join();
                gpa.free(blocked.out);
            }
            defer PrefillHold.release();
            try PrefillHold.waitHeld();
            if (!allDecoding()) return error.EndedBeforeTheShutdown;
            server.requestShutdown();
            if (!within(10_000, draining)) return error.NeverDrained;
            PrefillHold.release();
            for (flights[0..n]) |*f| {
                const answer = try f.finish();
                defer gpa.free(answer);
                try expectShutDown(f.row, answer);
            }
        }
    }.run);
}

/// `connect` with a 256 KB server receive buffer, so a request sent whole lands in one read, past the reader's buffer.
fn connectReceiveWide() !std.posix.fd_t {
    var sv: [2]std.posix.fd_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &sv) != 0) return error.SocketPairFailed;
    const wide: c_int = 256 * 1024;
    _ = std.c.setsockopt(sv[0], std.posix.SOL.SOCKET, std.posix.SO.RCVBUF, std.mem.asBytes(&wide), @sizeOf(c_int));
    _ = std.c.setsockopt(sv[1], std.posix.SOL.SOCKET, std.posix.SO.SNDBUF, std.mem.asBytes(&wide), @sizeOf(c_int));
    readLimit(sv[1], 10);
    server.spawnConnection(gpa, io, .{ .socket = .{ .handle = sv[0], .address = .{ .ip4 = .loopback(0) } } });
    return sv[1];
}

fn pause(ms: i64) void {
    std.Io.sleep(io, .fromMilliseconds(ms), .real) catch {};
}

/// Milliseconds until the server ends `fd` (or its read limit fires).
fn msUntilClosed(fd: std.posix.fd_t) !i64 {
    const t0 = nowMsMonotonic(io);
    const got = readUntil(fd, null) catch |e| switch (e) {
        error.ReadTimedOut => return nowMsMonotonic(io) - t0,
        else => return e,
    };
    gpa.free(got);
    return nowMsMonotonic(io) - t0;
}

test "server: a chunked upload sent whole, larger than the reader's buffer, is answered rather than left waiting in a read" {
    try eachArch(4, .{}, struct {
        fn run() !void {
            // Loads the model first, so the 10 s each answer may take is the request's own.
            var wb: [512]u8 = undefined;
            gpa.free(try request("POST", "/v1/completions", try completionBody(&wb, "hi", 2, false)));
            for ([_]usize{ 20_000, 30_000, 40_000 }) |pad| {
                var body: std.ArrayList(u8) = .empty;
                defer body.deinit(gpa);
                try body.appendSlice(gpa, "{\"model\":\"tiny\",\"prompt\":\"hi\",\"max_tokens\":2,\"temperature\":0,\"user\":\"");
                try body.appendNTimes(gpa, 'a', pad);
                try body.appendSlice(gpa, "\"}");
                var req: std.ArrayList(u8) = .empty;
                defer req.deinit(gpa);
                try req.appendSlice(gpa, "POST /v1/completions HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n");
                var off: usize = 0;
                while (off < body.items.len) {
                    const n = @min(5000, body.items.len - off);
                    var hb: [32]u8 = undefined;
                    try req.appendSlice(gpa, try std.fmt.bufPrint(&hb, "{x}\r\n", .{n}));
                    try req.appendSlice(gpa, body.items[off..][0..n]);
                    try req.appendSlice(gpa, "\r\n");
                    off += n;
                }
                try req.appendSlice(gpa, "0\r\n\r\n");
                const fd = try connectReceiveWide();
                defer _ = std.c.close(fd);
                try sendAll(fd, req.items);
                const resp = readUntil(fd, null) catch |e| {
                    std.debug.print("chunked upload of {d} bytes sent whole: no answer in 10 s ({s})\n", .{ req.items.len, @errorName(e) });
                    return error.ChunkedUploadHangs;
                };
                defer gpa.free(resp);
                if (resp.len == 0) return error.ChunkedUploadUnanswered;
                // The same body by Content-Length.
                const fd2 = try connectReceiveWide();
                defer _ = std.c.close(fd2);
                var hb2: [128]u8 = undefined;
                try sendAll(fd2, try std.fmt.bufPrint(&hb2, "POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Length: {d}\r\n\r\n", .{body.items.len}));
                try sendAll(fd2, body.items);
                const resp2 = readUntil(fd2, null) catch return error.ContentLengthUploadHangs;
                defer gpa.free(resp2);
                if (resp2.len == 0) return error.ContentLengthUploadUnanswered;
            }
        }
    }.run);
}

test "server: a shutdown while a body trickles, a chunked upload waits and a frame is half sent ends each connection at once" {
    try eachArch(4, .{}, struct {
        fn run() !void {
            const a1 = try connect();
            defer _ = std.c.close(a1);
            try sendAll(a1, "POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 50\r\n\r\n{\"mo");
            const a2 = try connect();
            defer _ = std.c.close(a2);
            try sendAll(a2, "POST /v1/completions HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n5\r\n{\"mod\r\n");
            // A frame of 20 payload bytes with 2 sent: the reader holds them, the rest never comes.
            const a3 = try wsConnect();
            defer _ = std.c.close(a3);
            try sendAll(a3, &.{ 0x81, 0x80 | 20, 0, 0, 0, 0, '{', '"' });
            for ([_]std.posix.fd_t{ a1, a2, a3 }) |fd| readLimit(fd, 5);
            pause(200);
            server.requestShutdown();
            for ([_]std.posix.fd_t{ a1, a2 }, [_][]const u8{ "trickled body", "waiting chunked upload" }) |fd, what| {
                const ms = try msUntilClosed(fd);
                if (ms >= 4500) {
                    std.debug.print("{s} at shutdown: open after {d} ms\n", .{ what, ms });
                    return error.ConnectionOutlivesShutdown;
                }
            }
            const code = wsReadCloseCode(a3) catch |e| {
                std.debug.print("half frame at shutdown: {s}\n", .{@errorName(e)});
                return e;
            };
            if (code != 1001) {
                std.debug.print("half frame at shutdown closed with {d}\n", .{code});
                return error.NotGoingAway;
            }
        }
    }.run);
}

test "server: a Content-Length body read past the reader's buffer, then stalled, ends at a shutdown" {
    try eachArch(4, .{}, struct {
        fn run() !void {
            const fd = try connectReceiveWide();
            defer _ = std.c.close(fd);
            readLimit(fd, 5);
            var req: std.ArrayList(u8) = .empty;
            defer req.deinit(gpa);
            try req.appendSlice(gpa, "POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 40000\r\n\r\n{\"model\":\"tiny\",\"user\":\"");
            try req.appendNTimes(gpa, 'a', 30_000);
            try sendAll(fd, req.items);
            pause(300);
            server.requestShutdown();
            const ms = try msUntilClosed(fd);
            if (ms >= 4500) {
                std.debug.print("stalled body at shutdown: open after {d} ms\n", .{ms});
                return error.StalledBodyOutlivesShutdown;
            }
        }
    }.run);
}

/// Every SSE event of a streamed body, in order: its `event:` name and `data:` line.
const SseEvent = struct { name: []const u8, data: []const u8 };

/// The events of `body`; the caller frees the slice only.
fn sseEvents(body: []const u8) ![]SseEvent {
    var out: std.ArrayList(SseEvent) = .empty;
    var blocks = std.mem.splitSequence(u8, body, "\n\n");
    while (blocks.next()) |blk| {
        var ev: SseEvent = .{ .name = "", .data = "" };
        var lines = std.mem.splitScalar(u8, blk, '\n');
        while (lines.next()) |l| {
            if (std.mem.startsWith(u8, l, "event: ")) ev.name = l["event: ".len..];
            if (std.mem.startsWith(u8, l, "data: ")) ev.data = l["data: ".len..];
        }
        if (ev.data.len > 0) try out.append(gpa, ev);
    }
    return out.toOwnedSlice(gpa);
}

fn parseBody(bytes: []const u8) !std.json.Parsed(std.json.Value) {
    return std.json.parseFromSlice(std.json.Value, gpa, bytes, .{}) catch |e| {
        std.debug.print("not JSON ({s}): {s}\n", .{ @errorName(e), bytes[0..@min(bytes.len, 600)] });
        return e;
    };
}

test "server: h21 a Responses envelope echoing service_tier is valid JSON" {
    try eachArch(4, .{}, struct {
        fn run() !void {
            // A tier with a quote and a backslash, and one that closes the string to inject a sibling field.
            const tiers = [_][]const u8{ "a\\\"b\\\\", "x\\\",\\\"injected\\\":\\\"1" };
            const want = [_][]const u8{ "a\"b\\", "x\",\"injected\":\"1" };
            for (tiers, want) |tier, sent| {
                var body: [256]u8 = undefined;
                const req = try std.fmt.bufPrint(&body, "{{\"model\":\"tiny\",\"max_output_tokens\":2,\"service_tier\":\"{s}\",\"input\":\"hi\"}}", .{tier});
                const resp = try request("POST", "/v1/responses", req);
                defer gpa.free(resp);
                const p = try parseBody(bodyOf(resp));
                defer p.deinit();
                try testing.expectEqualStrings(sent, p.value.object.get("service_tier").?.string);
                try testing.expect(p.value.object.get("injected") == null);

                const streamed = try std.fmt.bufPrint(&body, "{{\"model\":\"tiny\",\"max_output_tokens\":2,\"stream\":true,\"service_tier\":\"{s}\",\"input\":\"hi\"}}", .{tier});
                const sresp = try request("POST", "/v1/responses", streamed);
                defer gpa.free(sresp);
                const evs = try sseEvents(bodyOf(sresp));
                defer gpa.free(evs);
                var seen: usize = 0;
                for (evs) |e| {
                    if (!std.mem.eql(u8, e.name, "response.created") and !std.mem.eql(u8, e.name, "response.completed") and !std.mem.eql(u8, e.name, "response.incomplete")) continue;
                    const ep = try parseBody(e.data);
                    defer ep.deinit();
                    const r = ep.value.object.get("response").?.object;
                    try testing.expectEqualStrings(sent, r.get("service_tier").?.string);
                    try testing.expect(r.get("injected") == null);
                    seen += 1;
                }
                try testing.expectEqual(@as(usize, 2), seen);

                const id = try respId(resp);
                var path: [128]u8 = undefined;
                const got = try request("GET", try std.fmt.bufPrint(&path, "/v1/responses/{s}", .{id}), "");
                defer gpa.free(got);
                const gp = try parseBody(bodyOf(got));
                defer gp.deinit();
                try testing.expectEqualStrings(sent, gp.value.object.get("service_tier").?.string);
            }
        }
    }.run);
}

test "server: h21 a request field of the wrong type is refused by name, never read as absent" {
    try eachArch(4, .{}, struct {
        const Case = struct { path: []const u8, body: []const u8, field: []const u8 };
        const chat_tail = "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]";
        const cases = [_]Case{
            .{ .path = "/v1/chat/completions", .body = "{\"model\":\"tiny\",\"max_tokens\":2,\"stream\":\"true\"," ++ chat_tail ++ "}", .field = "'stream'" },
            .{ .path = "/v1/chat/completions", .body = "{\"model\":\"tiny\",\"max_tokens\":2,\"stream\":true,\"stream_options\":{\"include_usage\":\"true\"}," ++ chat_tail ++ "}", .field = "'stream_options.include_usage'" },
            .{ .path = "/v1/chat/completions", .body = "{\"model\":\"tiny\",\"max_tokens\":2,\"enable_mtp\":\"yes\"," ++ chat_tail ++ "}", .field = "'enable_mtp'" },
            .{ .path = "/v1/messages", .body = "{\"model\":\"tiny\",\"max_tokens\":2,\"stream\":\"true\"," ++ chat_tail ++ "}", .field = "'stream'" },
            .{ .path = "/v1/messages", .body = "{\"model\":\"tiny\",\"max_tokens\":2,\"enable_drafter\":1," ++ chat_tail ++ "}", .field = "'enable_drafter'" },
            .{ .path = "/v1/responses", .body = "{\"model\":\"tiny\",\"max_output_tokens\":2,\"stream\":1,\"input\":\"hi\"}", .field = "'stream'" },
            .{ .path = "/v1/responses", .body = "{\"model\":\"tiny\",\"max_output_tokens\":2,\"store\":\"false\",\"input\":\"hi\"}", .field = "'store'" },
            .{ .path = "/v1/responses", .body = "{\"model\":\"tiny\",\"max_output_tokens\":2,\"service_tier\":5,\"input\":\"hi\"}", .field = "'service_tier'" },
            .{ .path = "/v1/completions", .body = "{\"model\":\"tiny\",\"max_tokens\":2,\"echo\":\"true\",\"prompt\":\"hi\"}", .field = "'echo'" },
            .{ .path = "/v1/completions", .body = "{\"model\":\"tiny\",\"max_tokens\":2,\"stream\":true,\"stream_options\":\"all\",\"prompt\":\"hi\"}", .field = "'stream_options'" },
        };
        fn run() !void {
            for (cases) |c| {
                const resp = try request("POST", c.path, c.body);
                defer gpa.free(resp);
                if (!std.mem.startsWith(u8, resp, "HTTP/1.1 400") or std.mem.indexOf(u8, bodyOf(resp), c.field) == null) {
                    std.debug.print("wrong-typed field not refused by name: {s} {s}\n  -> {s}\n", .{ c.path, c.body, resp[0..@min(resp.len, 200)] });
                    return error.FieldReadAsAbsent;
                }
            }
            // A WebSocket turn forces `store` true in the body it reserializes: its own `store` is read before that.
            const fd = try wsConnect();
            defer _ = std.c.close(fd);
            try wsSend(fd, "{\"type\":\"response.create\",\"model\":\"tiny\",\"input\":\"hi\",\"max_output_tokens\":2,\"store\":\"false\"}");
            const frame = try wsReadFrame(fd);
            defer gpa.free(frame);
            if (!std.mem.startsWith(u8, frame, "{\"type\":\"error\",\"status\":400") or std.mem.indexOf(u8, frame, "'store'") == null) {
                std.debug.print("a WebSocket turn with store:\"false\" ended in: {s}\n", .{frame[0..@min(frame.len, 300)]});
                return error.FieldReadAsAbsent;
            }
        }
    }.run);
}

test "server: h21 a Responses stream cut by max_output_tokens ends in response.incomplete" {
    try eachArch(4, .{}, struct {
        fn run() !void {
            const resp = try request("POST", "/v1/responses", "{\"model\":\"tiny\",\"max_output_tokens\":1,\"temperature\":0,\"stream\":true,\"reasoning\":{\"effort\":\"none\"},\"input\":\"hi\"}");
            defer gpa.free(resp);
            const evs = try sseEvents(bodyOf(resp));
            defer gpa.free(evs);
            var terminals: usize = 0;
            for (evs) |e| {
                if (std.mem.eql(u8, e.name, "response.completed")) {
                    std.debug.print("a reply cut by max_output_tokens ended in response.completed: {s}\n", .{e.data[0..@min(e.data.len, 200)]});
                    return error.CutReplyEndedCompleted;
                }
                if (!std.mem.eql(u8, e.name, "response.incomplete")) continue;
                terminals += 1;
                const p = try parseBody(e.data);
                defer p.deinit();
                try testing.expectEqualStrings("response.incomplete", p.value.object.get("type").?.string);
                const r = p.value.object.get("response").?.object;
                try testing.expectEqualStrings("incomplete", r.get("status").?.string);
                try testing.expectEqualStrings("max_output_tokens", r.get("incomplete_details").?.object.get("reason").?.string);
            }
            try testing.expectEqual(@as(usize, 1), terminals);
            // `data: [DONE]` follows it, and nothing else.
            try testing.expectEqualStrings("[DONE]", evs[evs.len - 1].data);
            try testing.expectEqualStrings("response.incomplete", evs[evs.len - 2].name);

            // Not streamed: the same status and reason.
            const plain = try request("POST", "/v1/responses", "{\"model\":\"tiny\",\"max_output_tokens\":1,\"temperature\":0,\"reasoning\":{\"effort\":\"none\"},\"input\":\"hi\"}");
            defer gpa.free(plain);
            const pp = try parseBody(bodyOf(plain));
            defer pp.deinit();
            try testing.expectEqualStrings("incomplete", pp.value.object.get("status").?.string);
            try testing.expectEqualStrings("max_output_tokens", pp.value.object.get("incomplete_details").?.object.get("reason").?.string);

            // A WebSocket turn cut the same way ends in the same event.
            const fd = try wsConnect();
            defer _ = std.c.close(fd);
            try wsSend(fd, "{\"type\":\"response.create\",\"model\":\"tiny\",\"input\":\"hi\",\"max_output_tokens\":1,\"temperature\":0,\"reasoning\":{\"effort\":\"none\"}}");
            while (true) {
                const frame = try wsReadFrame(fd);
                defer gpa.free(frame);
                if (std.mem.indexOf(u8, frame, "\"type\":\"response.completed\"") != null) return error.CutTurnEndedCompleted;
                if (std.mem.indexOf(u8, frame, "\"type\":\"response.incomplete\"") != null) break;
                if (std.mem.startsWith(u8, frame, "{\"type\":\"error\"")) return error.TurnFailed;
            }
        }
    }.run);
}

/// The integer at `path` of `v`; null when the path ends elsewhere or on another type.
fn jsonInt(v: std.json.Value, path: []const []const u8) ?i64 {
    var cur = v;
    for (path) |k| {
        if (cur != .object) return null;
        cur = cur.object.get(k) orelse return null;
    }
    return if (cur == .integer) cur.integer else null;
}

test "server: h21 a null or zero token-budget field leaves the surface's alias in force" {
    try eachArch(4, .{}, struct {
        fn run() !void {
            const resp = try request("POST", "/v1/responses", "{\"model\":\"tiny\",\"max_output_tokens\":null,\"max_tokens\":1,\"temperature\":0,\"reasoning\":{\"effort\":\"none\"},\"input\":\"hi\"}");
            defer gpa.free(resp);
            const p = try parseBody(bodyOf(resp));
            defer p.deinit();
            try testing.expectEqual(@as(?i64, 1), jsonInt(p.value, &.{"max_output_tokens"}));
            try testing.expectEqual(@as(?i64, 1), jsonInt(p.value, &.{ "usage", "output_tokens" }));

            const chat = try request("POST", "/v1/chat/completions", "{\"model\":\"tiny\",\"max_completion_tokens\":0,\"max_tokens\":1,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");
            defer gpa.free(chat);
            const cp = try parseBody(bodyOf(chat));
            defer cp.deinit();
            try testing.expectEqual(@as(?i64, 1), jsonInt(cp.value, &.{ "usage", "completion_tokens" }));
        }
    }.run);
}

test "server: h21 a tool_choice the surface does not serve is refused by name, never read as auto" {
    try eachArch(4, .{}, struct {
        const Case = struct { path: []const u8, body: []const u8, field: []const u8 };
        const tail = "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]";
        const cases = [_]Case{
            .{ .path = "/v1/responses", .body = "{\"model\":\"tiny\",\"max_output_tokens\":2,\"tool_choice\":\"bogus\",\"input\":\"hi\"}", .field = "'tool_choice'" },
            .{ .path = "/v1/responses", .body = "{\"model\":\"tiny\",\"max_output_tokens\":2,\"tool_choice\":{\"type\":\"allowed_tools\"},\"input\":\"hi\"}", .field = "'tool_choice.type'" },
            .{ .path = "/v1/messages", .body = "{\"model\":\"tiny\",\"max_tokens\":2,\"tool_choice\":{\"type\":\"bogus\"}," ++ tail ++ "}", .field = "'tool_choice.type'" },
            .{ .path = "/v1/messages", .body = "{\"model\":\"tiny\",\"max_tokens\":2,\"tool_choice\":{\"type\":\"tool\"}," ++ tail ++ "}", .field = "'tool_choice.name'" },
            .{ .path = "/v1/messages/count_tokens", .body = "{\"model\":\"tiny\",\"tool_choice\":\"auto\"," ++ tail ++ "}", .field = "'tool_choice'" },
        };
        fn run() !void {
            for (cases) |c| {
                const resp = try request("POST", c.path, c.body);
                defer gpa.free(resp);
                if (!std.mem.startsWith(u8, resp, "HTTP/1.1 400") or std.mem.indexOf(u8, bodyOf(resp), c.field) == null) {
                    std.debug.print("tool_choice not refused by name: {s} {s}\n  -> {s}\n", .{ c.path, c.body, resp[0..@min(resp.len, 300)] });
                    return error.ToolChoiceReadAsAuto;
                }
            }
        }
    }.run);
}

/// A connection whose request is in its socket before the server's thread starts, so a shutdown already asked cannot
/// end the read first: the request reaches its route.
fn askedBeforeServed(path: []const u8, body: []const u8) ![]u8 {
    var sv: [2]std.posix.fd_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &sv) != 0) return error.SocketPairFailed;
    defer _ = std.c.close(sv[1]);
    readLimit(sv[1], 30);
    try send(sv[1], "POST", path, body);
    server.spawnConnection(gpa, io, .{ .socket = .{ .handle = sv[0], .address = .{ .ip4 = .loopback(0) } } });
    return readUntil(sv[1], null);
}

/// The OpenAI routes carry `shutting_down`, Anthropic's `api_error` (typed by status, by ruling), all the message.
fn saysShutdown(path: []const u8, answer: []const u8) bool {
    const anthropic = std.mem.startsWith(u8, path, "/v1/messages");
    const ty = if (anthropic) "\"type\":\"api_error\"" else "\"type\":\"" ++ http.shutting_down_type ++ "\"";
    const code = "\"code\":\"" ++ http.shutting_down_type ++ "\"";
    return std.mem.indexOf(u8, answer, http.shutting_down_message) != null and
        (std.mem.indexOf(u8, answer, ty) != null or (!anthropic and std.mem.indexOf(u8, answer, code) != null));
}

test "server: every route asked after the shutdown answers the one shutdown error" {
    try eachArch(4, .{ .tiny2 = true }, struct {
        fn run() !void {
            gpa.free(try request("POST", "/v1/completions", "{\"model\":\"tiny\",\"prompt\":\"hi\",\"max_tokens\":2}"));
            server.requestShutdown();
            const rows = [_][2][]const u8{
                .{ "/v1/completions", "{\"model\":\"tiny\",\"prompt\":\"hi\",\"max_tokens\":2}" },
                .{ "/v1/completions", "{\"model\":\"tiny\",\"prompt\":\"hi\",\"max_tokens\":2,\"stream\":true}" },
                .{ "/v1/chat/completions", "{\"model\":\"tiny\",\"max_tokens\":2,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" },
                .{ "/v1/chat/completions", "{\"model\":\"tiny\",\"max_tokens\":2,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" },
                .{ "/v1/messages", "{\"model\":\"tiny\",\"max_tokens\":2,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" },
                .{ "/v1/messages", "{\"model\":\"tiny\",\"max_tokens\":2,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" },
                .{ "/v1/responses", "{\"model\":\"tiny\",\"max_output_tokens\":2,\"input\":\"hi\"}" },
                .{ "/v1/responses", "{\"model\":\"tiny\",\"max_output_tokens\":2,\"stream\":true,\"input\":\"hi\"}" },
                .{ "/v1/completions", "{\"model\":\"tiny2\",\"prompt\":\"hi\",\"max_tokens\":2}" },
                .{ "/v1/load-model", "{\"model\":\"tiny2\"}" },
                .{ "/v1/load-model", "{\"model\":\"tiny\"}" },
                .{ "/v1/unload-model", "{\"model\":\"tiny\"}" },
            };
            var bad: usize = 0;
            for (rows) |row| {
                const answer = try askedBeforeServed(row[0], row[1]);
                defer gpa.free(answer);
                if (!saysShutdown(row[0], answer)) {
                    std.debug.print("after shutdown {s} {s}: {s}\n", .{ row[0], row[1], answer[0..@min(answer.len, 320)] });
                    bad += 1;
                }
            }
            if (bad > 0) return error.NotTheShutdownError;
        }
    }.run);
}

const Flood = struct {
    fd: std.posix.fd_t,
    bytes: []const u8,
    stop: std.atomic.Value(bool) = .init(false),
    /// Monotonic ms when a send first failed (the server closed); 0 while it takes bytes.
    ended_ms: std.atomic.Value(i64) = .init(0),
    fn run(f: *Flood) void {
        while (!f.stop.load(.acquire)) {
            const n = std.c.send(f.fd, f.bytes.ptr, f.bytes.len, std.posix.MSG.DONTWAIT);
            if (n < 0 and std.posix.errno(n) == .AGAIN) {
                pause(1);
                continue;
            }
            if (n <= 0) {
                f.ended_ms.store(nowMsMonotonic(io), .release);
                return;
            }
        }
    }
};

test "server: a shutdown ends a chunked upload whose client floods carriage returns, and a WebSocket fed empty continuation frames" {
    try eachArch(4, .{}, struct {
        fn run() !void {
            var cr: [16 * 1024]u8 = @splat('\r');
            const up = try connectReceiveWide();
            defer _ = std.c.close(up);
            try sendAll(up, "POST /v1/completions HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n");
            var empties: [6 * 2048]u8 = undefined;
            for (0..2048) |i| @memcpy(empties[i * 6 ..][0..6], &[_]u8{ 0x00, 0x80, 0, 0, 0, 0 });
            const ws = try wsConnect();
            defer _ = std.c.close(ws);
            try sendAll(ws, &.{ 0x01, 0x80 | 1, 0, 0, 0, 0, '{' });
            var floods = [_]Flood{ .{ .fd = up, .bytes = &cr }, .{ .fd = ws, .bytes = &empties } };
            var threads: [2]std.Thread = undefined;
            for (&floods, 0..) |*f, i| threads[i] = try std.Thread.spawn(.{}, Flood.run, .{f});
            pause(500);
            const t0 = nowMsMonotonic(io);
            server.requestShutdown();
            // The drain is 35 s; give each a 5 s bound, then let go so serve can finish.
            var waited: i64 = 0;
            while (waited < 5000 and (floods[0].ended_ms.load(.acquire) == 0 or floods[1].ended_ms.load(.acquire) == 0)) : (waited += 20) pause(20);
            for (&floods) |*f| f.stop.store(true, .release);
            for (threads) |th| th.join();
            var bad: usize = 0;
            for (floods, [_][]const u8{ "chunked CR flood", "WS empty-continuation flood" }) |f, what| {
                const end = f.ended_ms.load(.acquire);
                if (end == 0 or end - t0 > 1000) {
                    std.debug.print("{s}: {s}\n", .{ what, if (end == 0) "still taking bytes 5 s after the shutdown" else "ended late" });
                    bad += 1;
                }
            }
            // Stop the floods; the half-done requests now end at their next wait.
            _ = std.c.shutdown(up, std.posix.SHUT.WR);
            _ = std.c.shutdown(ws, std.posix.SHUT.WR);
            if (bad > 0) return error.FloodOutlivesShutdown;
        }
    }.run);
}

fn modelState(id: []const u8) @import("model_registry.zig").LoadState {
    const reg = server_state.global_registry.?;
    reg.mutex.lockUncancelable(reg.io);
    defer reg.mutex.unlock(reg.io);
    return reg.entries.get(id).?.state;
}

fn rawExchange(bytes: []const u8) ![]u8 {
    const fd = try connect();
    defer _ = std.c.close(fd);
    readLimit(fd, 30);
    try sendAll(fd, bytes);
    return readUntil(fd, null);
}

test "server: an unload whose Content-Length has no space after its colon unloads the model it names, or is refused, never another" {
    try eachArch(4, .{ .tiny2 = true }, struct {
        fn run() !void {
            gpa.free(try request("POST", "/v1/completions", "{\"model\":\"tiny\",\"prompt\":\"hi\",\"max_tokens\":2}"));
            gpa.free(try request("POST", "/v1/load-model", "{\"model\":\"tiny2\"}"));
            const before = .{ modelState("tiny"), modelState("tiny2") };
            // A valid HTTP/1.1 head: the field value's leading whitespace is optional (RFC 9110 §5.6.3).
            const answer = try rawExchange("POST /v1/unload-model HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length:17\r\n\r\n{\"model\":\"tiny2\"}");
            defer gpa.free(answer);
            const after = .{ modelState("tiny"), modelState("tiny2") };
            const refused = std.mem.startsWith(u8, answer, "HTTP/1.1 400 ");
            if (!refused and (after[0] != before[0] or after[1] == before[1])) {
                std.debug.print("unload of tiny2 by Content-Length:17: before tiny={s} tiny2={s}, after tiny={s} tiny2={s}; answer: {s}\n", .{ @tagName(before[0]), @tagName(before[1]), @tagName(after[0]), @tagName(after[1]), answer[0..@min(answer.len, 300)] });
                return error.BodyDroppedWrongModel;
            }
        }
    }.run);
}

test "server: an unload whose Content-Length has whitespace before its colon, is folded or follows a bare LF unloads the model it names, or is refused" {
    // RFC 9112 §5.1 (whitespace before the colon: MUST 400), §5.2 (obs-fold: 400 or SP), §2.2 (bare LF: a line end or
    // refused). The contract of h15-1's served test: unloads the model it names, or a 400, never another.
    try eachArch(4, .{ .tiny2 = true }, struct {
        fn run() !void {
            var bad: usize = 0;
            for ([_][]const u8{
                "Content-Length : 17\r\n",
                "Content-Length\t:17\r\n",
                "Content-Length: 1\r\n 7\r\n",
                "X-A: b\nContent-Length: 17\r\n",
            }) |field| {
                gpa.free(try request("POST", "/v1/completions", "{\"model\":\"tiny\",\"prompt\":\"hi\",\"max_tokens\":2}"));
                gpa.free(try request("POST", "/v1/load-model", "{\"model\":\"tiny2\"}"));
                const before = .{ modelState("tiny"), modelState("tiny2") };
                const sent = try std.mem.concat(gpa, u8, &.{ "POST /v1/unload-model HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n", field, "\r\n{\"model\":\"tiny2\"}" });
                defer gpa.free(sent);
                const answer = try rawExchange(sent);
                defer gpa.free(answer);
                const after = .{ modelState("tiny"), modelState("tiny2") };
                const refused = std.mem.startsWith(u8, answer, "HTTP/1.1 400 ");
                if (!refused and (after[0] != before[0] or after[1] == before[1])) {
                    std.debug.print("unload of tiny2 by '{f}': before tiny={s} tiny2={s}, after tiny={s} tiny2={s}; answer: {s}\n", .{ std.zig.fmtString(field), @tagName(before[0]), @tagName(before[1]), @tagName(after[0]), @tagName(after[1]), answer[0..@min(answer.len, 200)] });
                    bad += 1;
                }
            }
            if (bad != 0) return error.BodyDroppedWrongModel;
        }
    }.run);
}

test "server: a ping sent in the same write as the WebSocket upgrade gets its pong" {
    try eachArch(4, .{}, struct {
        fn run() !void {
            const fd = try connect();
            defer _ = std.c.close(fd);
            readLimit(fd, 5);
            try sendAll(fd, "GET /v1/responses HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n" ++ [_]u8{ 0x89, 0x80 | 2, 0, 0, 0, 0, 'h', 'i' });
            const head = try readUntil(fd, "\r\n\r\n");
            defer gpa.free(head);
            if (!std.mem.startsWith(u8, head, "HTTP/1.1 101")) return error.NoUpgrade;
            // The pong can arrive in the same read as the 101.
            const past = head[std.mem.indexOf(u8, head, "\r\n\r\n").? + 4 ..];
            if (past.len > 0) {
                if (past[0] != 0x8A) return error.NotAPong;
                return;
            }
            const f = wsReadFrameOf(fd) catch |e| {
                std.debug.print("eager ping: no frame in 5 s ({s})\n", .{@errorName(e)});
                return error.EagerFrameLost;
            };
            defer gpa.free(f.payload);
            if (f.opcode != 0xA) return error.NotAPong;
        }
    }.run);
}
