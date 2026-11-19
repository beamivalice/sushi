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
const ModelRegistry = @import("model_registry.zig").ModelRegistry;

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

const Arch = struct { name: []const u8, tmp: testing.TmpDir };

/// What a test asks of a fixture beyond being served.
const Needs = struct {
    /// A chat or Responses reply that runs on for hundreds of tokens (`cooler_contract.server_long_replies`).
    long_replies: bool = false,
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
            archs[i] = .{ .name = @tagName(t), .tmp = tmp };
            try tmp.dir.createDirPath(io, "m");
            var m = try tmp.dir.openDir(io, "m", .{});
            defer m.close(io);
            try M.cooler_contract.serverDir(m);
        };
    }
    for (&archs) |*a| if (a.*) |*x| withServer(&x.tmp, max_concurrent, body) catch |e| {
        if (e != error.SkipZigTest) std.debug.print("server test on {s}: {s}\n", .{ x.name, @errorName(e) });
        return e;
    };
}

fn withServer(tmp: *testing.TmpDir, max_concurrent: u32, body: *const fn () anyerror!void) !void {
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

    const reg = try ModelRegistry.init(a, io, null, 2, 0, null);
    defer reg.deinit();
    var settings_buf: [1024]u8 = undefined;
    reg.settings_path = try std.fmt.bufPrint(&settings_buf, "{s}/.sushi/model-settings.json", .{home});
    const entry = try reg.registerStub("tiny", model_dir, 90 << 20);
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
    try server.serve(io, a, params, stub.config, "127.0.0.1", 0, .{ .embedded_client = startClient, .request_timeout_sec = 300, .test_on_cooler = true });
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
    // A read that waits past this fails the test instead of hanging it.
    const limit: std.posix.timeval = .{ .sec = 30, .usec = 0 };
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
        if (n <= 0) break;
        try out.appendSlice(gpa, buf[0..@intCast(n)]);
    }
    return out.toOwnedSlice(gpa);
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
            const t1 = try readUntil(fd, "response.completed");
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
            const rest = try readUntil(fd, "response.completed");
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

/// Whether a token of a completion stream has reached `fd`'s client end: read without taking it.
fn hasToken(fd: std.posix.fd_t) !bool {
    var fds = [_]std.posix.pollfd{.{ .fd = fd, .events = std.posix.POLL.IN, .revents = 0 }};
    if (try std.posix.poll(&fds, 0) == 0) return false;
    var peek: [8192]u8 = undefined;
    const n = std.c.recv(fd, &peek, peek.len, std.posix.MSG.PEEK);
    return n > 0 and std.mem.indexOf(u8, peek[0..@intCast(n)], "\"choices\"") != null;
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

test "server: a shutdown with the submit queue full ends the request waiting for room with the shutdown error too" {
    try eachArch(1, .{}, struct {
        fn queueFull() bool {
            const sch = server_state.global_scheduler.?;
            sch.queue_mu.lockUncancelable(sch.io);
            defer sch.queue_mu.unlock(sch.io);
            return sch.in_flight == sch.queue_cap;
        }
        fn run() !void {
            var b: [512]u8 = undefined;
            // The model loads first, so every request below reaches submit.
            gpa.free(try request("POST", "/v1/completions", try completionBody(&b, "hi", 2, false)));
            // One more than the queue holds: the last waits in submit for room.
            var fds: [34]std.posix.fd_t = undefined;
            for (&fds) |*fd| {
                fd.* = try connect();
                try send(fd.*, "POST", "/v1/completions", try completionBody(&b, "write forever about rivers", 100, true));
            }
            defer for (fds) |fd| {
                _ = std.c.close(fd);
            };
            if (!within(5000, queueFull)) return error.QueueNeverFull;
            // The one stream already generating may have finished before the shutdown; every other had not started.
            var started: [fds.len]bool = undefined;
            for (fds, &started) |fd, *s| s.* = try hasToken(fd);
            server.requestShutdown();
            for (fds, started, 0..) |fd, s, i| {
                const all = try readUntil(fd, null);
                defer gpa.free(all);
                if (s) continue;
                errdefer std.debug.print("stream {d} of {d} at the shutdown: {s}\n", .{ i, fds.len, all[all.len -| 600..] });
                if (std.mem.indexOf(u8, all, "\"finish_reason\":\"length\"") != null or std.mem.indexOf(u8, all, "\"finish_reason\":\"stop\"") != null)
                    return error.ServedPastTheShutdown;
                if (std.mem.indexOf(u8, all, "shutting down") == null) return error.EndedWithoutTheShutdownError;
            }
        }
    }.run);
}

/// A stream the client stops reading after its first token, with its socket left open, an unload asked behind it and
/// then another request: each of the two answers within `ms`. Returns their answers; the caller frees.
fn behindAStalledReader(ms: u32) ![2][]u8 {
    var b: [2][512]u8 = undefined;
    gpa.free(try request("POST", "/v1/completions", try completionBody(&b[0], "hi", 2, false)));
    const stalled = try connect();
    defer _ = std.c.close(stalled);
    try send(stalled, "POST", "/v1/completions", try completionBody(&b[1], "a long tale of the sea", 64, true));
    gpa.free(try readUntil(stalled, "\"choices\""));
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
    while (!(asked[0].done.load(.acquire) and asked[1].done.load(.acquire)) and waited < ms) : (waited += 10) {
        var none = [_]std.posix.pollfd{};
        _ = std.posix.poll(&none, 10) catch {};
    }
    const wedged = !asked[1].done.load(.acquire);
    // The stall ends here either way, so the threads end.
    _ = std.c.shutdown(stalled, std.posix.SHUT.RDWR);
    for (threads) |t| t.join();
    if (wedged) {
        std.debug.print("{d} ms behind a stalled reader: the unload done={any} ({s}), the request behind it done={any}\n", .{ waited, asked[0].done.load(.acquire), asked[0].out[0..@min(asked[0].out.len, 200)], asked[1].done.load(.acquire) });
        for (asked) |q| gpa.free(q.out);
        return error.WedgedByAStalledReader;
    }
    return .{ asked[0].out, asked[1].out };
}

fn unloadPending() bool {
    const reg = server_state.global_registry.?;
    reg.mutex.lockUncancelable(reg.io);
    defer reg.mutex.unlock(reg.io);
    return reg.entries.get("tiny").?.state == .evicting;
}

test "server: a client that stops reading its stream gives up its model, so an unload behind it and every request behind that go on" {
    const was = http_conn.send_limit_ms;
    http_conn.send_limit_ms = 200;
    defer http_conn.send_limit_ms = was;
    try eachArch(4, .{}, struct {
        fn run() !void {
            const got = try behindAStalledReader(10_000);
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
            const got = try behindAStalledReader(10_000);
            defer for (got) |g| gpa.free(g);
            if (!std.mem.startsWith(u8, got[0], "HTTP/1.1 409") or std.mem.indexOf(u8, got[0], "model_busy") == null) return error.UnloadNotRefusedByName;
            if (!std.mem.startsWith(u8, got[1], "HTTP/1.1 200")) return error.RequestBehindFailed;
        }
    }.run);
}

/// The next WebSocket frame the server sends (unmasked), its payload; the caller frees.
fn wsReadFrame(fd: std.posix.fd_t) ![]u8 {
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
    return payload;
}

fn readExact(fd: std.posix.fd_t, out: []u8) !void {
    var got: usize = 0;
    while (got < out.len) {
        const n = std.c.read(fd, out[got..].ptr, out.len - got);
        if (n <= 0) return error.ConnectionEnded;
        got += @intCast(n);
    }
}

/// Sets a declared `iogpu.wired_limit_mb` whose floor (the limit less the margin) is `floor` bytes; null restores both.
fn setWiredFloor(floor: ?u64) void {
    const limit_mb: u64 = 120_000;
    bills.wired_limit_mb_override = if (floor != null) limit_mb else null;
    server_state.wired_limit_margin_bytes = if (floor) |f| (limit_mb << 20) - f else server_state.WIRED_LIMIT_MARGIN_BYTES;
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
            defer setWiredFloor(null);
            const ctx_was = server_state.server_config.max_context_size;
            defer server_state.server_config.max_context_size = ctx_was;
            for (cases) |c| {
                if (std.mem.eql(u8, c.name, "memory admission")) setWiredFloor(1 << 20);
                server_state.server_config.max_context_size = if (std.mem.eql(u8, c.name, "context overflow")) 64 else ctx_was;
                try wsSend(fd, c.frame);
                const frame = try wsReadFrame(fd);
                defer gpa.free(frame);
                if (!std.mem.startsWith(u8, frame, "{\"type\":\"error\",\"status\":4")) {
                    std.debug.print("refused turn ({s}) ended in: {s}\n", .{ c.name, frame[0..@min(frame.len, 400)] });
                    return error.RefusalNotTheTurnsErrorFrame;
                }
            }
            setWiredFloor(null);
            // The connection serves the next turn.
            try wsSend(fd, "{\"type\":\"response.create\",\"model\":\"tiny\",\"input\":\"hi\",\"max_output_tokens\":2,\"temperature\":0}");
            while (true) {
                const frame = try wsReadFrame(fd);
                defer gpa.free(frame);
                if (std.mem.indexOf(u8, frame, "\"type\":\"response.completed\"") != null) break;
                if (std.mem.startsWith(u8, frame, "{\"type\":\"error\"")) return error.NextTurnFailed;
            }
        }
    }.run);
}
