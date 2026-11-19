const std = @import("std");
const json_text = @import("json_text.zig");
const http = @import("http.zig");
const tokenizer_mod = @import("tokenizer.zig");
const chat_mod = @import("chat.zig");
const model_mod = @import("model.zig");
const log = @import("log.zig");
const responses_mod = @import("responses.zig");
const slot_admission = @import("slot_admission.zig");
const model_registry_mod = @import("model_registry.zig");
const instr = @import("metrics.zig");
const update_mod = @import("update.zig");
const Tokenizer = tokenizer_mod.Tokenizer;
const ModelRegistry = model_registry_mod.ModelRegistry;
const LoadedModel = model_registry_mod.LoadedModel;
const request_policy = @import("request_policy.zig");
const request_parse = @import("request_parse.zig");
const writeEffortNoticeHeader = request_policy.writeEffortNoticeHeader;
const io_util = @import("io_util.zig");
const ws_mod = @import("ws.zig");
const http_head = @import("http_head.zig");
const ByteBudget = @import("byte_budget.zig").ByteBudget;
const nowMsMonotonic = io_util.nowMsMonotonic;
const testing = std.testing;
const server_mod = @import("server.zig");
const server_state = @import("server_state.zig");
const generation_stream = @import("generation_stream.zig");
const responses_api = @import("responses_api.zig");
const completions = @import("completions.zig");
const anthropic = @import("anthropic.zig");
const EMPTY_PROMPT_MSG = generation_stream.EMPTY_PROMPT_MSG;
const SSE_ALLOW_HEADERS_DEFAULT = http.SSE_ALLOW_HEADERS_DEFAULT;
const bindAddress = http.bindAddress;
const chatPageUrl = server_mod.chatPageUrl;
const handleCompletions = completions.handleCompletions;
const handleConnection = server_mod.handleConnection;
const handleResponses = responses_api.handleResponses;
const jsonEscape = http.jsonEscape;
const mapGenerationError = generation_stream.mapGenerationError;
const readAllFromPair = server_mod.readAllFromPair;
const redactQueryKeys = http.redactQueryKeys;
const responseBody = http.responseBody;
const sendErrorResponse = http.sendErrorResponse;
const sendResponseFramed = http.sendResponseFramed;
const sendStreamIdle = generation_stream.sendStreamIdle;

pub fn startListener(host: []const u8, port: u16) (error{InvalidHost} || std.Io.net.IpAddress.ListenError)!std.Io.net.Server {
    var addr = try bindAddress(host, port);
    const ip4 = addr.ip4;
    const rc = std.posix.system.socket(std.posix.AF.INET, std.posix.SOCK.STREAM, 0);
    switch (std.posix.errno(rc)) {
        .SUCCESS => {},
        .MFILE => return error.ProcessFdQuotaExceeded,
        .NFILE => return error.SystemFdQuotaExceeded,
        .NOMEM, .NOBUFS => return error.SystemResources,
        else => return error.Unexpected,
    }
    const fd: std.posix.socket_t = @intCast(rc);
    errdefer _ = std.posix.system.close(fd);
    if (std.posix.errno(std.posix.system.fcntl(fd, std.posix.F.SETFD, @as(usize, std.posix.FD_CLOEXEC))) != .SUCCESS) return error.Unexpected;

    const on: c_int = 1;
    if (std.posix.errno(std.posix.system.setsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.REUSEADDR, @ptrCast(&on), @sizeOf(c_int))) != .SUCCESS) return error.Unexpected;

    const sa = std.posix.sockaddr.in{
        .port = std.mem.nativeToBig(u16, ip4.port),
        .addr = @bitCast(ip4.bytes),
    };
    switch (std.posix.errno(std.posix.system.bind(fd, @ptrCast(&sa), @sizeOf(std.posix.sockaddr.in)))) {
        .SUCCESS => {},
        .ADDRINUSE => return error.AddressInUse,
        .ACCES => return error.AccessDenied,
        .ADDRNOTAVAIL => return error.AddressUnavailable,
        .AFNOSUPPORT => return error.AddressFamilyUnsupported,
        .NOMEM, .NOBUFS => return error.SystemResources,
        else => return error.Unexpected,
    }
    switch (std.posix.errno(std.posix.system.listen(fd, 128))) {
        .SUCCESS => {},
        .ADDRINUSE => return error.AddressInUse,
        else => return error.Unexpected,
    }
    var bound: std.posix.sockaddr.in = undefined;
    var bound_len: std.posix.socklen_t = @sizeOf(@TypeOf(bound));
    if (std.posix.errno(std.posix.system.getsockname(fd, @ptrCast(&bound), &bound_len)) != .SUCCESS) return error.Unexpected;
    addr.ip4.port = std.mem.bigToNative(u16, bound.port);
    return .{ .socket = .{ .handle = fd, .address = addr }, .options = {} };
}

/// The error envelope a request's API takes: OpenAI's `{"error":{...}}` or Anthropic's `{"type":"error",...}`.
pub const Api = enum { openai, anthropic };

/// The one predicate of a request's API: every path under `/v1/messages` is Anthropic's, its 401 and 404 included.
pub fn apiOfPath(path: []const u8) Api {
    return if (std.mem.eql(u8, path, "/v1/messages") or std.mem.startsWith(u8, path, "/v1/messages/")) .anthropic else .openai;
}

/// The request line decides the error envelope and whether an answer carries a body, so it is read before the reader
/// can answer anything (a 431, a 413, the shutdown error).
fn noteRequestLine(stream: *Conn, head: []const u8) void {
    const line = http_head.requestLine(head);
    stream.api = apiOfPath(line.target[0 .. std.mem.indexOfScalar(u8, line.target, '?') orelse line.target.len]);
    stream.head_only = std.mem.eql(u8, line.method, "HEAD");
}

/// Bridge that lets a Conn route OpenAI-Responses output through a WebSocket
/// transport instead of HTTP/SSE. When `Conn.ws_mode` is set, `sendResponse`
/// and `sendAnthropicEvent` send the JSON payload as a single WS text frame
/// instead of an HTTP body or SSE event line.
pub const WsBridge = struct {
    impl: *anyopaque,
    sendTextFn: *const fn (impl: *anyopaque, data: []const u8) anyerror!void,
    allocator: ?std.mem.Allocator = null,
    /// Status emitted in the most recent `response.completed`-class event,
    /// or null if no terminal event was seen.
    captured_status: ?[]u8 = null,

    pub fn sendText(self: *WsBridge, data: []const u8) !void {
        if (self.allocator) |a| {
            if (extractCompletedStatus(a, data)) |st| {
                if (self.captured_status) |old| a.free(old);
                self.captured_status = st;
            }
        }
        try self.sendTextFn(self.impl, data);
    }

    pub fn reset(self: *WsBridge) void {
        if (self.allocator) |a| {
            if (self.captured_status) |st| a.free(st);
        }
        self.captured_status = null;
    }
};

/// Pull the value of `"status":"..."` (response-level, not item-level)
/// from a `response.completed`/`.failed`/`.incomplete` payload.
fn extractCompletedStatus(allocator: std.mem.Allocator, data: []const u8) ?[]u8 {
    // Only look in completed/failed/incomplete events.
    const is_terminal = std.mem.indexOf(u8, data, "\"type\":\"response.completed\"") != null or
        std.mem.indexOf(u8, data, "\"type\":\"response.failed\"") != null or
        std.mem.indexOf(u8, data, "\"type\":\"response.incomplete\"") != null;
    if (!is_terminal) return null;
    // Status field appears after the response object; first match is fine.
    const needle = "\"status\":\"";
    const start = std.mem.indexOf(u8, data, needle) orelse return null;
    const v_start = start + needle.len;
    const v_end = std.mem.indexOfScalarPos(u8, data, v_start, '"') orelse return null;
    return allocator.dupe(u8, data[v_start..v_end]) catch null;
}

/// Tracks how long a streaming connection has been SILENT so the token loops
/// can emit a protocol-level keepalive before a client's idle-body timeout
/// fires. The mirror image of `degenerate.StallClock`: that one protects the
/// server from a wedged model, this one protects the client from a wedged-
/// looking socket.
///
/// It exists because every streaming surface buffers tokens while it might be
/// looking at a tool call or an unclosed thinking block — during a large
/// tool call (an agent one-shotting a whole file into `write_file`) the
/// handler produces no SSE output at all, sometimes for minutes. Node's
/// `fetch`/undici drops such a body after 300 s (`TypeError: terminated`).
pub const StreamHeartbeat = struct {
    /// MONOTONIC milliseconds (`io_util.nowMsMonotonic`) — an NTP step must not
    /// be able to stall the keepalive, nor spam one per token.
    last_write_ms: i64 = 0,
    interval_ms: i64 = Conn.STREAM_KEEPALIVE_MS,

    pub fn noteWrite(self: *StreamHeartbeat, now_ms: i64) void {
        self.last_write_ms = now_ms;
    }

    /// True once `interval_ms` has elapsed with no bytes handed to the socket.
    pub fn due(self: *const StreamHeartbeat, now_ms: i64) bool {
        return now_ms - self.last_write_ms >= self.interval_ms;
    }
};

/// How long a connection's write waits for the client to take a byte, as its reads are bounded: a client that stops
/// reading with its socket open fails the write, so its request is cancelled and the model it holds is released.
const SEND_LIMIT_MS: i64 = 30_000;
/// `SEND_LIMIT_MS`, which a test shortens.
pub var send_limit_ms: i64 = SEND_LIMIT_MS;

/// `std.Io.net.Stream.Writer`'s socket writer, each send bounded by `send_limit_ms` of no progress (`err` is then
/// `error.SendTimedOut`), which the kernel's `SO_SNDTIMEO` enforces.
pub const SocketWriter = struct {
    interface: std.Io.Writer,
    fd: std.posix.fd_t,
    err: ?anyerror = null,

    fn init(fd: std.posix.fd_t, buffer: []u8) SocketWriter {
        const limit: std.posix.timeval = .{ .sec = @intCast(@divTrunc(send_limit_ms, 1000)), .usec = @intCast(@rem(send_limit_ms, 1000) * 1000) };
        std.posix.setsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.SNDTIMEO, std.mem.asBytes(&limit)) catch |e|
            log.warn("[http] a connection's writes stay unbounded: SO_SNDTIMEO refused ({s})\n", .{@errorName(e)});
        return .{ .fd = fd, .interface = .{ .vtable = &.{ .drain = drain }, .buffer = buffer } };
    }

    fn drain(io_w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
        const w: *SocketWriter = @alignCast(@fieldParentPtr("interface", io_w));
        var iov: [16]std.posix.iovec_const = undefined;
        var n: usize = 0;
        const parts = .{ io_w.buffered(), data[0 .. data.len - 1] };
        if (parts[0].len > 0) {
            iov[n] = .{ .base = parts[0].ptr, .len = parts[0].len };
            n += 1;
        }
        for (parts[1]) |d| if (d.len > 0 and n < iov.len) {
            iov[n] = .{ .base = d.ptr, .len = d.len };
            n += 1;
        };
        const pattern = data[data.len - 1];
        if (pattern.len > 0) for (0..splat) |_| {
            if (n == iov.len) break;
            iov[n] = .{ .base = pattern.ptr, .len = pattern.len };
            n += 1;
        };
        if (n == 0) return io_w.consume(0);
        const sent = w.sendWithin(iov[0..n]) catch |e| {
            w.err = e;
            return error.WriteFailed;
        };
        return io_w.consume(sent);
    }

    /// A send that moved some bytes returns their count; one that moved none in `send_limit_ms` fails.
    fn sendWithin(w: *SocketWriter, iov: []const std.posix.iovec_const) !usize {
        const msg: std.posix.msghdr_const = .{ .name = null, .namelen = 0, .iov = iov.ptr, .iovlen = @intCast(iov.len), .control = null, .controllen = 0, .flags = 0 };
        while (true) {
            const rc = std.posix.system.sendmsg(w.fd, &msg, std.posix.MSG.NOSIGNAL);
            switch (std.posix.errno(rc)) {
                .SUCCESS => return @intCast(rc),
                .INTR => continue,
                .AGAIN => return error.SendTimedOut,
                .PIPE => return error.BrokenPipe,
                .CONNRESET => return error.ConnectionResetByPeer,
                else => |e| return std.posix.unexpectedErrno(e),
            }
        }
    }
};

/// Connection wrapper bundling a TCP stream with its `Io` and per-connection
/// reader/writer buffers. Replaces `std.net.Stream` in 0.16 — methods that took
/// a bare stream now take `*Conn` so the IO interface and buffers travel together.
pub const Conn = struct {
    stream: std.Io.net.Stream,
    io: std.Io,
    write_buf: [16 * 1024]u8,
    read_buf: [16 * 1024]u8,
    write_state: SocketWriter,
    read_state: std.Io.net.Stream.Reader,
    /// Silence tracker for the streaming keepalive. Stamped by every write
    /// path below, so it measures client-visible silence rather than "time
    /// since the last token" — a buffered tool call keeps producing tokens
    /// while emitting nothing. (`writeAllNoFlush` stamps too: every SSE emit
    /// path flushes before returning, so a stamp can't outrun the socket by
    /// more than one event.)
    heartbeat: StreamHeartbeat = .{},
    /// Non-null when this connection is bridged onto a WebSocket. Output that
    /// would otherwise be HTTP/SSE is reshaped into WS text frames at the
    /// `sendResponse` / `sendAnthropicEvent` chokepoints.
    ws_mode: ?*WsBridge = null,
    /// The API this request speaks, from its route: `http.sendErrorResponse` shapes every error by it.
    api: Api = .openai,
    /// True once this connection has written a `text/event-stream` head: past it a generation
    /// failure can only be an SSE `error` event. Set only in `sendSseHeaders`.
    sse_headers_sent: bool = false,
    /// Bytes the request reader took past its request (a WebSocket frame sent with the upgrade): read before the socket.
    early: []const u8 = &.{},
    /// The request was HEAD: a response sends its head and never its body.
    head_only: bool = false,
    /// A Content-Length response ends by length and can release its socket immediately.
    length_framed: bool = false,
    /// Set for one request when Qwen `high`/`max` was served as `xhigh`. The
    /// text is the log line; `effort_notice_json` is `,"effort_notice":"..."`.
    effort_notice: ?[]const u8 = null,
    effort_notice_json: ?[]const u8 = null,
    /// `<requested> -> <granted>` when the window cut this request's `max_tokens`; empty otherwise.
    max_tokens_cut: [40]u8 = undefined,
    max_tokens_cut_len: u8 = 0,

    pub fn init(c: *Conn, stream: std.Io.net.Stream, io: std.Io) void {
        c.stream = stream;
        c.io = io;
        c.write_state = SocketWriter.init(stream.socket.handle, &c.write_buf);
        c.read_state = stream.reader(io, &c.read_buf);
        c.ws_mode = null;
        c.early = &.{};
        c.api = .openai;
        c.sse_headers_sent = false;
        c.head_only = false;
        c.length_framed = false;
        c.effort_notice = null;
        c.effort_notice_json = null;
        c.max_tokens_cut_len = 0;
        c.heartbeat = .{ .last_write_ms = nowMsMonotonic(io) };
    }

    /// True when the connection has been silent long enough that a streaming
    /// keepalive is due.
    pub fn keepaliveDue(c: *Conn) bool {
        return c.heartbeat.due(nowMsMonotonic(c.io));
    }

    pub fn writer(c: *Conn) *std.Io.Writer {
        return &c.write_state.interface;
    }

    /// Only `takeReadable`, `readArrived` and `awaitReadable` reach it (lint `conn-read-unbounded`).
    fn reader(c: *Conn) *std.Io.Reader {
        return &c.read_state.interface;
    }

    pub fn writeAll(c: *Conn, data: []const u8) !void {
        c.heartbeat.noteWrite(nowMsMonotonic(c.io));
        try c.writer().writeAll(data);
        try c.writer().flush();
    }

    pub fn writeAllNoFlush(c: *Conn, data: []const u8) !void {
        c.heartbeat.noteWrite(nowMsMonotonic(c.io));
        try c.writer().writeAll(data);
    }

    pub fn flush(c: *Conn) !void {
        c.heartbeat.noteWrite(nowMsMonotonic(c.io));
        try c.writer().flush();
    }

    /// The one read of a connection: up to buf.len bytes, maybe fewer, 0 at EOF; `error.ReadStalled` when nothing came
    /// within `limit_ms` (null: no limit). It waits in `awaitReadable` first, so a shutdown ends it (`error.Shutdown`)
    /// even with bytes waiting. Bytes the reader holds come back alone: a `readVec` that wants more blocks in the
    /// socket for the rest.
    pub fn read(c: *Conn, buf: []u8, limit_ms: ?i64) !usize {
        if (!try c.awaitReadable(limit_ms)) return error.ReadStalled;
        return c.takeReadable(buf);
    }

    /// Up to buf.len of the bytes already here, whatever the shutdown flag says, without waiting: 0 when none. Bounded
    /// by `buf`, so a shutdown still learns the request line of a request that arrived.
    pub fn readArrived(c: *Conn, buf: []u8) usize {
        if (c.early.len == 0 and c.reader().bufferedLen() == 0) {
            var fds = [_]std.posix.pollfd{.{ .fd = c.stream.socket.handle, .events = std.posix.POLL.IN, .revents = 0 }};
            if ((std.posix.poll(&fds, 0) catch 0) == 0) return 0;
        }
        return c.takeReadable(buf) catch 0;
    }

    /// The early bytes or the held bytes alone, else one read of a socket seen readable.
    fn takeReadable(c: *Conn, buf: []u8) !usize {
        if (c.early.len > 0) {
            const n = @min(c.early.len, buf.len);
            @memcpy(buf[0..n], c.early[0..n]);
            c.early = c.early[n..];
            return n;
        }
        const r = c.reader();
        if (r.bufferedLen() > 0) {
            const n = @min(r.bufferedLen(), buf.len);
            @memcpy(buf[0..n], r.buffered()[0..n]);
            r.toss(n);
            return n;
        }
        var bufs: [1][]u8 = .{buf};
        return r.readVec(&bufs) catch |err| switch (err) {
            error.EndOfStream => 0,
            else => |e| e,
        };
    }

    /// How often a wait for a client's bytes looks at the shutdown flag.
    const SHUTDOWN_POLL_MS: i64 = 100;

    /// Waits up to `limit_ms` (null: no limit) for bytes to read; false when none came. Every read of a connection
    /// waits here, and the shutdown flag is read first: a client that keeps bytes arriving cannot hold the drain, and
    /// an idle one is let go within `SHUTDOWN_POLL_MS`. Bytes the reader already buffered count as readable.
    fn awaitReadable(c: *Conn, limit_ms: ?i64) error{Shutdown}!bool {
        if (server_state.shutdown_requested.load(.acquire)) return error.Shutdown;
        if (c.early.len > 0 or c.reader().bufferedLen() > 0) return true;
        const deadline = if (limit_ms) |ms| nowMsMonotonic(c.io) +| ms else null;
        var fds = [_]std.posix.pollfd{.{ .fd = c.stream.socket.handle, .events = std.posix.POLL.IN, .revents = 0 }};
        var slice_ms: i64 = 0;
        while (true) {
            const n = std.posix.poll(&fds, @intCast(slice_ms)) catch return true;
            if (n > 0) return true;
            if (server_state.shutdown_requested.load(.acquire)) return error.Shutdown;
            const left = if (deadline) |d| d - nowMsMonotonic(c.io) else SHUTDOWN_POLL_MS;
            if (left <= 0) return false;
            slice_ms = @min(left, SHUTDOWN_POLL_MS);
        }
    }

    const CLOSE_WAIT_MS: i64 = 5 * 60 * 1000;
    /// How long a length-framed response waits for the client to close first: a client that sent bytes we never
    /// read (a refused body, a reused socket) would otherwise get a reset that can destroy the response.
    const LINGER_MS: i64 = 250;
    /// `LINGER_MS`, which a test shortens so it does not wait out the real linger.
    var linger_ms: i64 = LINGER_MS;

    /// Keep close-delimited responses owned until the reader closes: orphaned FIN_WAIT_2
    /// sockets can expire with a reset while a slow client is still consuming the body.
    pub fn close(c: *Conn) void {
        c.flush() catch {};
        const fd = c.stream.socket.handle;
        _ = std.posix.system.shutdown(fd, std.posix.SHUT.WR);
        const deadline = nowMsMonotonic(c.io) + if (c.length_framed) linger_ms else CLOSE_WAIT_MS;
        var fds = [_]std.posix.pollfd{.{ .fd = fd, .events = std.posix.POLL.IN, .revents = 0 }};
        var sink: [256]u8 = undefined;
        while (nowMsMonotonic(c.io) < deadline and !server_state.shutdown_requested.load(.acquire)) {
            const n = std.posix.poll(&fds, @intCast(std.math.clamp(deadline - nowMsMonotonic(c.io), 0, 500))) catch break;
            if (n == 0) continue;
            // EOF or reset ends the wait; discard any pipelined request bytes.
            if (std.c.recv(fd, &sink, sink.len, std.posix.MSG.DONTWAIT) <= 0) break;
        }
        c.stream.close(c.io);
    }

    /// Non-blocking probe: has the peer closed the connection? TCP send
    /// buffers absorb hundreds of SSE writes after the client FIN/RST so
    /// `writeAll` only fails seconds late; this surfaces the disconnect
    /// promptly so a long-running decode can be cancelled before the GPU
    /// burns more cycles.
    ///
    /// Uses `poll(timeout=0)` for HUP/ERR, and a zero-byte `recv(MSG_PEEK)`
    /// to disambiguate `POLLIN`-with-data (still alive, client sending) from
    /// `POLLIN`-with-FIN (peer closed, `recv` returns 0). Stray bytes from a
    /// pipelined client are not expected on an SSE response stream — if we
    /// see them, we conservatively treat the connection as live.
    /// Idle interval (ms) between keepalive probes while a streaming
    /// request waits on its first tokens (prefill). Short enough that an
    /// abandoned prefill is noticed and cancelled quickly and that clients
    /// never hit stream-idle timeouts; tiny traffic cost.
    pub const STREAM_KEEPALIVE_MS: i64 = 5000;

    pub fn peerClosed(c: *Conn) bool {
        const fd = c.stream.socket.handle;
        var fds = [_]std.posix.pollfd{.{
            .fd = fd,
            .events = std.posix.POLL.IN,
            .revents = 0,
        }};
        const n = std.posix.poll(&fds, 0) catch return false;
        if (n == 0) return false;
        const revents = fds[0].revents;
        if ((revents & (std.posix.POLL.HUP | std.posix.POLL.ERR | std.posix.POLL.NVAL)) != 0) return true;
        if ((revents & std.posix.POLL.IN) == 0) return false;
        var peek_buf: [1]u8 = undefined;
        const flags: c_int = std.posix.MSG.PEEK | std.posix.MSG.DONTWAIT;
        const r = std.c.recv(fd, &peek_buf, peek_buf.len, flags);
        if (r == 0) return true; // FIN: peer closed cleanly
        return false; // negative (EAGAIN) or data available → assume alive
    }
};

pub fn closeEmbeddedListener(io: std.Io, listener: *std.Io.net.Server, client: ?std.Thread) void {
    listener.deinit(io);
    if (client) |thread| thread.join();
}

const ModelRoute = union(enum) { entry: []const u8, default, not_found };

/// Where a request's `model` routes. An unknown NAME is the default, so SDK
/// names like "gpt-4" keep working; a path (`/…`, `~/…`, never `org/repo`)
/// names its own entry or nothing, since another model answering it would
/// pass for the pack the client asked for.
pub fn routeRequestModel(registry: *ModelRegistry, id: []const u8, home: ?[]const u8) ModelRoute {
    if (id.len == 0 or std.mem.eql(u8, id, "sushi")) return .default;
    if (registry.peek(id)) |e| return .{ .entry = e.id };
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = if (std.mem.startsWith(u8, id, "~/"))
        std.fmt.bufPrint(&buf, "{s}{s}", .{ std.mem.trimEnd(u8, home orelse return .not_found, "/"), id[1..] }) catch return .not_found
    else if (std.mem.startsWith(u8, id, "/"))
        id
    else
        return .default;
    const e = registry.peekByPath(path) orelse return .not_found;
    return .{ .entry = e.id };
}

test "routeRequestModel: a path names its own entry, an unknown name the default" {
    const t = std.testing;
    const reg = try ModelRegistry.init(t.allocator, t.io, null, 4, 0, null);
    defer reg.deinit();
    _ = try reg.registerStub("org/good", "/m/org/good", 1);
    // The exact path wins over another pack registered under the same basename.
    _ = try reg.registerStub("good", "/elsewhere/good", 1);
    // `--model` and `/v1/load-model` register a folder under its name.
    _ = try reg.registerStub("Pack", "/x/y/Pack", 1);
    _ = try reg.registerStub("home-pack", "/Users/t/.sushi/models/home-pack", 1);
    const home: ?[]const u8 = "/Users/t";

    try t.expectEqualDeep(ModelRoute{ .entry = "org/good" }, routeRequestModel(reg, "org/good", home));
    try t.expectEqualDeep(ModelRoute{ .entry = "org/good" }, routeRequestModel(reg, "/m/org/good", home));
    try t.expectEqualDeep(ModelRoute{ .entry = "org/good" }, routeRequestModel(reg, "/m/org/good/", home));
    try t.expectEqualDeep(ModelRoute{ .entry = "Pack" }, routeRequestModel(reg, "/x/y/Pack", home));
    // A folder no entry is at names none, whatever its name: another model answering would pass for it.
    try t.expectEqualDeep(ModelRoute.not_found, routeRequestModel(reg, "/elsewhere/Pack", home));
    try t.expectEqualDeep(ModelRoute{ .entry = "home-pack" }, routeRequestModel(reg, "~/.sushi/models/home-pack", home));

    try t.expectEqualDeep(ModelRoute.not_found, routeRequestModel(reg, "/m/org/missing", home));
    try t.expectEqualDeep(ModelRoute.not_found, routeRequestModel(reg, "~/.sushi/models/missing", home));
    try t.expectEqualDeep(ModelRoute.not_found, routeRequestModel(reg, "~/.sushi/models/home-pack", null));
    try t.expectEqualDeep(ModelRoute.not_found, routeRequestModel(reg, "/", home));

    try t.expectEqualDeep(ModelRoute.default, routeRequestModel(reg, "gpt-4", home));
    try t.expectEqualDeep(ModelRoute.default, routeRequestModel(reg, "org/unknown", home));
    try t.expectEqualDeep(ModelRoute.default, routeRequestModel(reg, "sushi", home));
    try t.expectEqualDeep(ModelRoute.default, routeRequestModel(reg, "", home));
}

/// Every path `handleConnection` dispatches. Kept beside the chain rather than
/// derived from it because one question has to be answerable BEFORE a model is
/// resolved: does this endpoint exist at all?
///
/// Resolution runs ahead of dispatch, so without this an unknown path on a
/// server with no default model returned 503 "No default model configured"
/// instead of 404 — and a path's existence has nothing to do with model state.
/// Every path an equality arm dispatches must be listed here.
const ROUTE_PATHS = [_][]const u8{
    "/",
    "/chat",
    "/dashboard",
    "/downloads",
    "/models",
    "/settings",
    "/detokenize",
    "/health",
    "/metrics",
    "/metrics.json",
    "/metrics/enable",
    "/props",
    "/tokenize",
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/downloads",
    "/v1/downloads/cancel",
    "/v1/load-model",
    "/v1/messages",
    "/v1/messages/count_tokens",
    "/v1/models",
    "/v1/models/rescan",
    "/v1/prefix-cache",
    "/v1/prefix-cache/clear",
    "/v1/responses",
    "/v1/responses/compact",
    "/v1/settings",
    "/v1/slots/cancel",
    "/v1/tools",
    "/v1/unload-model",
    "/v1/update",
};

/// Model load/unload/rescan and the Responses WebSocket change server state, so a request that carries an
/// Origin (a browser) must come from this server's own page; clients that send none (curl, SDKs) pass.
pub fn crossOriginRefused(method: []const u8, path: []const u8, headers: []const u8, host: []const u8, port: u16) bool {
    const guarded = if (std.mem.eql(u8, path, "/v1/responses"))
        std.mem.eql(u8, method, "GET") and ws_mod.isUpgrade(headers)
    else
        std.mem.eql(u8, path, "/v1/load-model") or std.mem.eql(u8, path, "/v1/unload-model") or
            std.mem.eql(u8, path, "/v1/models/rescan") or std.mem.eql(u8, path, "/v1/prefix-cache/clear") or
            std.mem.eql(u8, path, "/v1/prefix-cache") or std.mem.eql(u8, path, "/v1/settings") or std.mem.eql(u8, path, "/metrics/enable") or
            std.mem.eql(u8, path, "/v1/downloads") or std.mem.eql(u8, path, "/v1/downloads/cancel") or
            std.mem.eql(u8, path, "/v1/slots/cancel");
    if (!guarded) return false;
    const origin = http_head.value(headers, "origin") orelse return false;
    // A wildcard bind has no host of its own: a page opened at any IP address the server answers on is its own
    // page, so match the Origin against the Host the client reached. Only an IP literal counts: a DNS name can be
    // rebound to this Mac and would then send itself as both Origin and Host.
    const want = if (std.mem.eql(u8, host, "0.0.0.0")) hostHeaderIp(headers) orelse "127.0.0.1" else host;
    return !update_mod.originMatches(origin, want, port);
}

/// The `Host` header's host when it is an IP literal (IPv4, or bracketed IPv6), its port stripped; null otherwise.
fn hostHeaderIp(headers: []const u8) ?[]const u8 {
    const h = std.mem.trim(u8, http_head.value(headers, "host") orelse return null, " ");
    if (h.len == 0) return null;
    if (h[0] == '[') {
        const close = std.mem.indexOfScalar(u8, h, ']') orelse return null;
        for (h[1..close]) |c| if (!std.ascii.isHex(c) and c != ':' and c != '.') return null;
        return if (close > 1) h[0 .. close + 1] else null;
    }
    const name = if (std.mem.indexOfScalar(u8, h, ':')) |colon| h[0..colon] else h;
    var parts: usize = 0;
    var it = std.mem.splitScalar(u8, name, '.');
    while (it.next()) |part| : (parts += 1) {
        if (part.len == 0 or part.len > 3) return null;
        const v = std.fmt.parseInt(u16, part, 10) catch return null;
        if (v > 255) return null;
    }
    return if (parts == 4) name else null;
}

const ROUTES_GET = [_][]const u8{
    "/",             "/chat",    "/dashboard", "/downloads", "/models",    "/settings", "/health",
    "/metrics",      "/metrics.json", "/props", "/v1/models", "/v1/prefix-cache",
};
const ROUTES_POST = [_][]const u8{
    "/detokenize",         "/tokenize",              "/metrics/enable",      "/v1/chat/completions", "/v1/completions",
    "/v1/downloads/cancel", "/v1/load-model",        "/v1/messages",         "/v1/messages/count_tokens",
    "/v1/models/rescan",   "/v1/prefix-cache/clear", "/v1/responses/compact", "/v1/slots/cancel",   "/v1/tools",
    "/v1/unload-model",    "/v1/update",
};

/// The methods a known path answers (the `Allow` header of its 405), or null for a path that is no route.
/// `/v1/models/{id}` is matched by prefix, as `/v1/responses/{id}`.
pub fn allowedMethods(path: []const u8) ?[]const u8 {
    for (ROUTES_GET) |p| if (std.mem.eql(u8, path, p)) return "GET, HEAD";
    for (ROUTES_POST) |p| if (std.mem.eql(u8, path, p)) return "POST";
    if (std.mem.eql(u8, path, "/v1/settings") or std.mem.eql(u8, path, "/v1/downloads")) return "GET, HEAD, POST";
    if (std.mem.eql(u8, path, "/v1/responses")) return "GET, POST";
    if (std.mem.startsWith(u8, path, "/v1/responses/")) return "GET, HEAD, DELETE";
    if (std.mem.startsWith(u8, path, "/v1/models/")) return "GET, HEAD";
    return null;
}

/// Whether `method` is one of the comma-separated `allow` list.
pub fn methodAllowed(allow: []const u8, method: []const u8) bool {
    var it = std.mem.tokenizeAny(u8, allow, ", ");
    while (it.next()) |m| if (std.mem.eql(u8, m, method)) return true;
    return false;
}

test "allowedMethods names each route's methods, and HEAD rides GET" {
    const t = std.testing;
    try t.expectEqualStrings("GET, HEAD", allowedMethods("/health").?);
    try t.expectEqualStrings("POST", allowedMethods("/v1/chat/completions").?);
    try t.expectEqualStrings("POST", allowedMethods("/v1/models/rescan").?);
    try t.expectEqualStrings("GET, HEAD", allowedMethods("/v1/models/some-id").?);
    try t.expectEqualStrings("GET, HEAD, DELETE", allowedMethods("/v1/responses/resp_1").?);
    try t.expectEqualStrings("POST", allowedMethods("/v1/responses/compact").?);
    try t.expect(allowedMethods("/nope") == null);
    try t.expect(methodAllowed("GET, HEAD", "HEAD"));
    try t.expect(!methodAllowed("POST", "GET"));
    // Every served route has an Allow entry.
    for (ROUTE_PATHS) |p| try t.expect(allowedMethods(p) != null);
}

/// Is `path` an endpoint this server serves at all (any method)?
pub fn routeExists(path: []const u8) bool {
    // `/v1/responses/{id}` (GET/DELETE) is matched by prefix, not by a literal.
    if (std.mem.startsWith(u8, path, "/v1/responses/") or std.mem.startsWith(u8, path, "/v1/models/")) return true;
    for (ROUTE_PATHS) |p| if (std.mem.eql(u8, path, p)) return true;
    return false;
}

/// The 413 names BOTH numbers it compared (context-overflow-400 class): a
/// refusal that only names the cap reads as "this should have worked".
/// A chunked body stops at the cap, so its size is unknown (`got` null).
fn payloadTooLargeMessage(buf: []u8, got: ?usize, cap: usize) []const u8 {
    const mb = 1024 * 1024;
    const setting = "limit (server.max_request_mb)";
    const msg = if (got) |n|
        std.fmt.bufPrint(buf, "Request body too large: {d} MB exceeds the {d} MB " ++ setting, .{ n / mb + @intFromBool(n % mb != 0), cap / mb })
    else
        std.fmt.bufPrint(buf, "Request body too large: over the {d} MB " ++ setting, .{cap / mb});
    return msg catch "Request body too large";
}

/// Per-connection thread arguments. Heap-allocated so the spawned thread
/// owns its lifetime; freed in `handleConnectionThread` after the handler
/// returns. The accepted stream is moved into a stack-allocated `Conn`
/// inside the thread.
pub const ConnThreadArgs = struct {
    allocator: std.mem.Allocator,
    accepted_stream: std.Io.net.Stream,
    io: std.Io,
};

/// Copies the bytes already read into the request buffer; pipelined bytes past `buf.len` are dropped.
fn seedRequestBuf(buf: []u8, already_read: []const u8) usize {
    const n = @min(buf.len, already_read.len);
    @memcpy(buf[0..n], already_read[0..n]);
    return n;
}

/// One whole request, owned: its head (`buf[0..header_end]`), its complete body to `end`, then any bytes the read took
/// past it (a WebSocket frame sent with the upgrade). Built only once the body is whole, so a short body never reaches
/// the router.
pub const RawRequest = struct {
    buf: []u8,
    header_end: usize,
    end: usize,
};

/// What reading one request came to; only `.request` is routed.
pub const ReadOutcome = union(enum) {
    request: RawRequest,
    /// Nothing came within the read limit: the connection closes bare.
    stalled,
    /// The client closed before its head ended.
    closed,
    /// The client closed mid-body: what arrived is dropped, never routed (RFC 9112 §6.3).
    cut_short,
    /// The head ran past `MAX_HEAD_BYTES`: answered 431.
    head_too_large,
    /// Refused with its answer sent: a body over the cap (413), a transfer coding (501), an expectation (417), a bad
    /// chunk (400).
    refused,
    /// A body framing that is not one reading (RFC 9112 §6.3): a Content-Length not one number, Transfer-Encoding
    /// twice or beside Content-Length. Answered 400 and closed.
    bad_framing,
    /// A shutdown ended the read: a request whose line arrived got the shutdown error.
    shutdown,
};

/// The most a request head (its line and headers) may take.
pub const MAX_HEAD_BYTES = 16 * 1024;

/// Decodes a `Transfer-Encoding: chunked` body one piece at a time, however the bytes arrive.
const ChunkDecoder = struct {
    state: enum { size, data, data_end, trailer, done } = .size,
    remaining: usize = 0,
    line: [80]u8 = undefined,
    line_len: usize = 0,
    /// The bytes of the size or trailer line being read, its CR included.
    line_raw: usize = 0,
    cr: bool = false,

    /// Appends the decoded bytes of `input` to `out`; returns how much of `input` it used. `budget` takes the data as
    /// payload and, as idle framing, what a chunk's minimal size line and CRLFs do not need (extensions, leading
    /// zeros) and every byte of the last chunk and its trailers.
    fn feed(d: *ChunkDecoder, a: std.mem.Allocator, out: *std.ArrayList(u8), input: []const u8, budget: *ByteBudget) (ByteBudget.Error || error{ BadChunk, OutOfMemory })!usize {
        var i: usize = 0;
        while (i < input.len and d.state != .done) {
            if (d.state == .data) {
                const n = @min(d.remaining, input.len - i);
                try budget.take(n, 0);
                try out.appendSlice(a, input[i..][0..n]);
                i += n;
                d.remaining -= n;
                if (d.remaining == 0) d.state = .data_end;
                continue;
            }
            const b = input[i];
            i += 1;
            if (d.state != .data_end) d.line_raw += 1;
            // A CR only right before its LF: a skipped byte would count toward no cap.
            if (d.cr and b != '\n') return error.BadChunk;
            d.cr = b == '\r';
            if (d.cr) continue;
            if (b != '\n') {
                switch (d.state) {
                    .size => {
                        if (d.line_len == d.line.len) return error.BadChunk;
                        d.line[d.line_len] = b;
                        d.line_len += 1;
                    },
                    .trailer => d.line_len += 1,
                    else => return error.BadChunk,
                }
                continue;
            }
            switch (d.state) {
                .size => {
                    const text = d.line[0..d.line_len];
                    // chunk-size = 1*HEXDIG, then optional whitespace only before an extension's ';' (RFC 9112 §7.1).
                    const hex = if (std.mem.indexOfScalar(u8, text, ';')) |semi| std.mem.trimEnd(u8, text[0..semi], " \t") else text;
                    const size = http_head.digits(hex, 16) orelse return error.BadChunk;
                    // A chunk that carries data has its minimal size line and CRLFs free; the rest is idle framing.
                    const free: usize = if (size == 0) 0 else std.fmt.count("{x}", .{size}) + 2;
                    try budget.take(0, d.line_raw -| free);
                    d.line_len = 0;
                    d.line_raw = 0;
                    d.remaining = size;
                    d.state = if (size == 0) .trailer else .data;
                },
                .data_end => d.state = .size,
                .trailer => {
                    try budget.take(0, d.line_raw);
                    if (d.line_len == 0) d.state = .done;
                    d.line_len = 0;
                    d.line_raw = 0;
                },
                else => unreachable,
            }
        }
        return i;
    }
};


/// Answers `Expect: 100-continue` so a client holding its body back sends it; any other expectation is a 417.
/// False when the request was refused.
fn answerExpect(allocator: std.mem.Allocator, stream: *Conn, head: []const u8) !bool {
    const expect = http_head.value(head, "expect") orelse return true;
    if (std.ascii.eqlIgnoreCase(expect, "100-continue")) {
        try stream.writeAll("HTTP/1.1 100 Continue\r\n\r\n");
        return true;
    }
    try sendErrorResponse(allocator, stream, "417 Expectation Failed", "invalid_request_error", "Only Expect: 100-continue is supported", 417);
    return false;
}

/// A shutdown ended a request's read: one whose request line arrived gets the shutdown error, an idle connection
/// nothing.
fn refuseAtShutdown(allocator: std.mem.Allocator, stream: *Conn, head: []const u8) !ReadOutcome {
    if (std.mem.indexOf(u8, head, "\r\n") == null) return .shutdown;
    noteRequestLine(stream, head);
    try sendErrorResponse(allocator, stream, "503 Service Unavailable", http.shutting_down_type, http.shutting_down_message, 503);
    return .shutdown;
}

/// Reads a chunked body after `head` (the request line and headers through the blank line); `leftover` is what
/// already arrived past it. The result holds the head followed by the decoded body.
fn readChunkedRequest(allocator: std.mem.Allocator, stream: *Conn, limit_ms: i64, head: []const u8, leftover: []const u8, max_request_bytes: usize) !ReadOutcome {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, head);
    var decoder: ChunkDecoder = .{};
    var budget = ByteBudget.init(max_request_bytes -| head.len);
    var buf: [8 * 1024]u8 = undefined;
    var pending = leftover;
    var asked = false;
    while (true) {
        const used = decoder.feed(allocator, &out, pending, &budget) catch |err| switch (err) {
            error.OutOfMemory => return err,
            error.TooLarge => {
                var msg_buf: [128]u8 = undefined;
                const msg = payloadTooLargeMessage(&msg_buf, null, max_request_bytes);
                log.warn("[http] 413: {s}\n", .{msg});
                try sendErrorResponse(allocator, stream, "413 Payload Too Large", "invalid_request_error", msg, 413);
                out.deinit(allocator);
                return .refused;
            },
            error.FramingOverAllowance => {
                const msg = "The chunked body's framing that carries no data (chunk extensions, padding, the last chunk and trailers) is over its 8 KiB allowance";
                log.warn("[http] 413: {s}\n", .{msg});
                try sendErrorResponse(allocator, stream, "413 Payload Too Large", "invalid_request_error", msg, 413);
                out.deinit(allocator);
                return .refused;
            },
            error.BadChunk => {
                try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid chunked request body", 400);
                out.deinit(allocator);
                return .refused;
            },
        };
        if (decoder.state == .done) {
            const end = out.items.len;
            try out.appendSlice(allocator, pending[used..]);
            const owned = try out.toOwnedSlice(allocator);
            return .{ .request = .{ .buf = owned, .header_end = head.len, .end = end } };
        }
        if (!asked) {
            asked = true;
            if (!try answerExpect(allocator, stream, head)) {
                out.deinit(allocator);
                return .refused;
            }
        }
        const n = stream.read(&buf, limit_ms) catch |err| {
            out.deinit(allocator);
            return switch (err) {
                error.Shutdown => refuseAtShutdown(allocator, stream, head),
                error.ReadStalled => .stalled,
                else => err,
            };
        };
        if (n == 0) {
            out.deinit(allocator);
            return .cut_short;
        }
        pending = buf[0..n];
    }
}

/// Reads one request: the head (up to `MAX_HEAD_BYTES`), then its `Content-Length` or chunked body, whole. Sets
/// `Conn.api` from the head before it can send anything, so every error of the request takes its API's envelope.
pub fn readRequestWithin(allocator: std.mem.Allocator, stream: *Conn, limit_ms: i64) !ReadOutcome {
    const head_deadline = nowMsMonotonic(stream.io) + limit_ms;
    var hdr_buf: [MAX_HEAD_BYTES]u8 = undefined;
    var total_read: usize = 0;
    const header_end_pos = while (total_read < hdr_buf.len) {
        const n = stream.read(hdr_buf[total_read..], head_deadline - nowMsMonotonic(stream.io)) catch |err| switch (err) {
            error.Shutdown => {
                total_read += stream.readArrived(hdr_buf[total_read..]);
                return refuseAtShutdown(allocator, stream, hdr_buf[0..total_read]);
            },
            error.ReadStalled => return .stalled,
            else => return err,
        };
        if (n == 0) return .closed;
        total_read += n;
        if (std.mem.indexOf(u8, hdr_buf[0..total_read], "\r\n\r\n")) |he| break he + 4;
    } else {
        noteRequestLine(stream, hdr_buf[0..total_read]);
        log.warn("[http] 431: a request head over {d} KB\n", .{MAX_HEAD_BYTES / 1024});
        try sendErrorResponse(allocator, stream, "431 Request Header Fields Too Large", "invalid_request_error", "The request head (its line and headers) is over the 16 KB limit", 431);
        return .head_too_large;
    };
    const head = hdr_buf[0..header_end_pos];
    noteRequestLine(stream, head);
    http_head.check(head) catch {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "The request head is malformed: a request line, a field line or a line end the HTTP/1.1 grammar does not allow", 400);
        return .bad_framing;
    };
    const max_request_bytes = server_state.max_request_bytes.load(.monotonic);

    const length = switch (http_head.framing(head) catch |err| switch (err) {
        error.BadFraming => {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "The request body's framing is ambiguous: a Content-Length that is not one number, or Transfer-Encoding twice or beside Content-Length", 400);
            return .bad_framing;
        },
        error.UnsupportedCoding => {
            try sendErrorResponse(allocator, stream, "501 Not Implemented", "invalid_request_error", "Only Transfer-Encoding: chunked is supported", 501);
            return .refused;
        },
    }) {
        .chunked => return readChunkedRequest(allocator, stream, limit_ms, head, hdr_buf[header_end_pos..total_read], max_request_bytes),
        .length => |n| n,
    };
    const total_size = header_end_pos +| length;
    if (total_size > max_request_bytes) {
        var msg_buf: [128]u8 = undefined;
        const msg = payloadTooLargeMessage(&msg_buf, total_size, max_request_bytes);
        log.warn("[http] 413: {s}\n", .{msg});
        try sendErrorResponse(allocator, stream, "413 Payload Too Large", "invalid_request_error", msg, 413);
        return .refused;
    }

    const buf = try allocator.alloc(u8, @max(total_size, total_read));
    var whole = false;
    defer if (!whole) allocator.free(buf);
    var got = seedRequestBuf(buf, hdr_buf[0..total_read]);
    if (got < total_size and !try answerExpect(allocator, stream, head)) return .refused;
    while (got < total_size) {
        const n = stream.read(buf[got..total_size], limit_ms) catch |err| switch (err) {
            error.Shutdown => return refuseAtShutdown(allocator, stream, head),
            error.ReadStalled => return .stalled,
            else => return err,
        };
        if (n == 0) return .cut_short;
        got += n;
    }
    whole = true;
    return .{ .request = .{ .buf = buf, .header_end = header_end_pos, .end = total_size } };
}

/// True for a read-only route that reports on a model without using it.
/// `GET /props` is the only one: every other status endpoint either answers
/// before `ensureLoaded` (`/health`, `/metrics`, `/v1/models` return above it) or is a real load
/// (`/v1/load-model`). `/props` alone falls through to the shared
/// `ensureLoaded`/release pair, so without this a poll stamps `last_used_ms`
/// and a client polling faster than `--idle-evict-secs` pins every model.
pub fn isStatusRoute(method: []const u8, path: []const u8) bool {
    return std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/props");
}

/// True for the routes textGenRejectReason protects — used for the
/// pre-load peek so naming an encoder in a chat request doesn't
/// cold-load gigabytes just to earn its 400.
pub fn isTextGenRoute(method: []const u8, path: []const u8) bool {
    if (std.mem.eql(u8, method, "POST")) {
        const routes = [_][]const u8{
            "/v1/chat/completions", "/v1/completions", "/v1/messages",
            "/v1/messages/count_tokens", "/v1/responses",
        };
        for (routes) |r| if (std.mem.eql(u8, path, r)) return true;
        return false;
    }
    // WebSocket upgrade handshake for /v1/responses.
    return std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/v1/responses");
}

/// The one place a `text/event-stream` response head is written. Past this call a generation
/// failure is an SSE `error` event; before it the stream answers like its non-streaming twin.
pub fn sendSseHeaders(stream: *Conn, kind: []const u8, allow_headers: []const u8) !void {
    try stream.writeAllNoFlush("HTTP/1.1 200 OK\r\n" ++
        "Content-Type: text/event-stream\r\n" ++
        "Cache-Control: no-cache\r\n" ++
        "Connection: close\r\n" ++
        "Access-Control-Allow-Origin: *\r\n" ++
        "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n" ++
        "Access-Control-Allow-Headers: ");
    try stream.writeAllNoFlush(allow_headers);
    try stream.writeAllNoFlush("\r\n");
    try writeEffortNoticeHeader(stream);
    try request_policy.writeMaxTokensCutHeader(stream);
    try stream.writeAll("\r\n");
    stream.sse_headers_sent = true;
    logHttpStreamStart(kind);
}

fn logHttpStreamStart(kind: []const u8) void {
    if (!log.isDebug()) return;
    log.debug("[http] <- 200 OK text/event-stream ({s})\n", .{kind});
}


/// Whether the API-key auth gate covers this peer at all: a key must be set,
/// and the peer must be non-loopback — unless `--api-key-strict` removed the
/// loopback exemption. Pure so the truth table is hermetically testable.
pub fn apiKeyGateApplies(key_set: bool, strict: bool, peer_loopback: bool) bool {
    return key_set and (!peer_loopback or strict);
}

test "apiKeyGateApplies: strict removes exactly the loopback exemption" {
    // No key: fully open regardless of strictness or peer.
    try std.testing.expect(!apiKeyGateApplies(false, false, true));
    try std.testing.expect(!apiKeyGateApplies(false, true, false));
    // Key, default: loopback exempt, network gated.
    try std.testing.expect(!apiKeyGateApplies(true, false, true));
    try std.testing.expect(apiKeyGateApplies(true, false, false));
    // Key, strict: everyone gated.
    try std.testing.expect(apiKeyGateApplies(true, true, true));
    try std.testing.expect(apiKeyGateApplies(true, true, false));
}


/// Constant-time equality — avoids a timing oracle on the API key.
fn constTimeEql(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    var diff: u8 = 0;
    for (a, b) |x, y| diff |= x ^ y;
    return diff == 0;
}

/// Extract a query-string parameter from the RAW (un-stripped) request path,
/// e.g. `queryParamValue("/metrics?api_key=abc", "api_key") == "abc"`.
fn queryParamValue(raw_path: []const u8, name: []const u8) ?[]const u8 {
    const qpos = std.mem.indexOfScalar(u8, raw_path, '?') orelse return null;
    var it = std.mem.splitScalar(u8, raw_path[qpos + 1 ..], '&');
    while (it.next()) |pair| {
        const eq = std.mem.indexOfScalar(u8, pair, '=') orelse continue;
        if (std.mem.eql(u8, pair[0..eq], name)) return pair[eq + 1 ..];
    }
    return null;
}

/// True if an `Authorization: Basic base64(user:pass)` value carries the key as
/// its password (browser index/metrics pages send this after the Basic prompt).
/// A user-only entry (no colon) matching the key is also accepted.
fn basicAuthMatches(enc: []const u8, key: []const u8) bool {
    var dbuf: [1024]u8 = undefined;
    const dec_len = std.base64.standard.Decoder.calcSizeForSlice(enc) catch return false;
    if (dec_len > dbuf.len) return false;
    std.base64.standard.Decoder.decode(dbuf[0..dec_len], enc) catch return false;
    const creds = dbuf[0..dec_len];
    const pass = if (std.mem.indexOfScalar(u8, creds, ':')) |c| creds[c + 1 ..] else creds;
    return constTimeEql(pass, key);
}

/// True if the request is authorized for the configured `g_api_key` (or no key
/// is set). Accepts, in order: `Authorization: Bearer <key>`, HTTP Basic (key as
/// the password), `x-api-key: <key>` (Anthropic), and `?api_key=` / `?key=`.
pub fn apiKeyAuthorized(raw_headers: []const u8, raw_path: []const u8) bool {
    const key = server_state.g_api_key orelse return true;
    if (http_head.value(raw_headers, "authorization")) |auth| {
        if (std.ascii.startsWithIgnoreCase(auth, "Bearer ")) {
            if (constTimeEql(std.mem.trim(u8, auth["Bearer ".len..], " \t"), key)) return true;
        } else if (std.ascii.startsWithIgnoreCase(auth, "Basic ")) {
            if (basicAuthMatches(std.mem.trim(u8, auth["Basic ".len..], " \t"), key)) return true;
        }
    }
    if (http_head.value(raw_headers, "x-api-key")) |xk| {
        if (constTimeEql(xk, key)) return true;
    }
    if (queryParamValue(raw_path, "api_key")) |q| {
        if (constTimeEql(q, key)) return true;
    }
    if (queryParamValue(raw_path, "key")) |q| {
        if (constTimeEql(q, key)) return true;
    }
    return false;
}

/// A 401 with a Basic-auth challenge, so browsers prompt for the key on the index and metrics pages; API clients
/// read the error envelope of the request's API.
pub fn sendUnauthorized(allocator: std.mem.Allocator, stream: *Conn) !void {
    try http.sendErrorResponseWithHeaders(allocator, stream, "401 Unauthorized", "authentication_error", "missing or invalid API key", 401, "WWW-Authenticate: Basic realm=\"sushi\"\r\n");
}

test "apiKeyAuthorized accepts Bearer, x-api-key, Basic, and query param" {
    const prev = server_state.g_api_key;
    defer server_state.g_api_key = prev;
    server_state.g_api_key = "s3cret";

    // Bearer
    try std.testing.expect(apiKeyAuthorized("Authorization: Bearer s3cret\r\n", "/v1/chat/completions"));
    // x-api-key (Anthropic)
    try std.testing.expect(apiKeyAuthorized("x-api-key: s3cret\r\n", "/v1/messages"));
    // Basic — base64("user:s3cret") = "dXNlcjpzM2NyZXQ="
    try std.testing.expect(apiKeyAuthorized("Authorization: Basic dXNlcjpzM2NyZXQ=\r\n", "/"));
    // Query param (browser fallback)
    try std.testing.expect(apiKeyAuthorized("", "/metrics.json?api_key=s3cret"));
    try std.testing.expect(apiKeyAuthorized("", "/metrics?key=s3cret"));

    // Wrong / missing key is rejected
    try std.testing.expect(!apiKeyAuthorized("Authorization: Bearer wrong\r\n", "/"));
    try std.testing.expect(!apiKeyAuthorized("x-api-key: nope\r\n", "/"));
    // RFC 7235: the auth scheme is case-insensitive.
    try std.testing.expect(apiKeyAuthorized("authorization: bearer s3cret\r\n", "/v1/models"));
    try std.testing.expect(apiKeyAuthorized("Authorization: BASIC dXNlcjpzM2NyZXQ=\r\n", "/"));
    try std.testing.expect(!apiKeyAuthorized("", "/v1/models"));

    // No key configured ⇒ always authorized (open mode)
    server_state.g_api_key = null;
    try std.testing.expect(apiKeyAuthorized("", "/v1/chat/completions"));
}

/// One raw request through `handleConnection` from a loopback peer, over a
/// socketpair against an empty registry; returns the raw response (caller frees).
pub fn serveOneForTest(request: []const u8) ![]u8 {
    return serveSplitForTest(request, "");
}

/// `serveOneForTest` with the request sent in two parts: `rest` goes out once `head` has been answered with
/// `100 Continue` (when it asks for one), as a client holding its body back does.
pub fn serveSplitForTest(head: []const u8, rest: []const u8) ![]u8 {
    const reg = try ModelRegistry.init(std.testing.allocator, std.testing.io, null, 1, 0, null);
    defer reg.deinit();
    reg.settings_path = "/nonexistent/sushi-test/model-settings.json";
    return serveWithRegistryForTest(reg, head, rest);
}

/// `serveSplitForTest` with `reg` as the server's registry; null serves a server that has none yet.
fn serveWithRegistryForTest(reg: ?*ModelRegistry, head: []const u8, rest: []const u8) ![]u8 {
    const a = std.testing.allocator;
    const prev_registry = server_state.global_registry;
    defer server_state.global_registry = prev_registry;
    server_state.global_registry = reg;

    var sv: [2]std.posix.fd_t = undefined;
    if (std.c.socketpair(1, 1, 0, &sv) != 0) return error.SocketPairFailed; // AF_UNIX, SOCK_STREAM
    var client_open = true;
    defer if (client_open) {
        _ = std.c.close(sv[1]);
    };
    var conn: Conn = undefined;
    Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = .{ .ip4 = .loopback(0) } } }, std.testing.io);
    if (std.c.write(sv[1], head.ptr, head.len) != @as(isize, @intCast(head.len))) return error.WriteFailed;
    const server = try std.Thread.spawn(.{}, struct {
        fn run(c: *Conn) void {
            handleConnection(std.testing.allocator, c) catch {};
            c.close();
        }
    }.run, .{&conn});

    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(a);
    var buf: [16 * 1024]u8 = undefined;
    const interim = "HTTP/1.1 100 Continue\r\n\r\n";
    while (rest.len > 0) {
        const n = std.c.read(sv[1], &buf, buf.len);
        if (n <= 0) break;
        try out.appendSlice(a, buf[0..@intCast(n)]);
        if (std.mem.indexOf(u8, out.items, interim) != null or std.mem.startsWith(u8, out.items, "HTTP/1.1 4")) break;
    }
    if (rest.len > 0 and std.c.write(sv[1], rest.ptr, rest.len) != @as(isize, @intCast(rest.len))) return error.WriteFailed;
    while (true) {
        const n = std.c.read(sv[1], &buf, buf.len);
        if (n <= 0) break;
        try out.appendSlice(a, buf[0..@intCast(n)]);
    }
    // The client closes first, as a finished one does, so the server's close need not linger.
    _ = std.c.close(sv[1]);
    client_open = false;
    server.join();
    return out.toOwnedSlice(a);
}

test "request size limit: content-length and chunked bodies use the current limit" {
    const t = std.testing;
    const previous = server_state.max_request_bytes.swap(256, .monotonic);
    defer server_state.max_request_bytes.store(previous, .monotonic);
    const body: [256]u8 = @splat(' ');
    for ([_]bool{ false, true }) |chunked| {
        const request = if (chunked)
            try std.fmt.allocPrint(t.allocator, "GET /health HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n100\r\n{s}\r\n0\r\n\r\n", .{body})
        else
            try std.fmt.allocPrint(t.allocator, "GET /health HTTP/1.1\r\nContent-Length: 256\r\n\r\n{s}", .{body});
        defer t.allocator.free(request);
        const size = std.mem.indexOf(u8, request, "\r\n\r\n").? + 4 + body.len;
        server_state.max_request_bytes.store(size - 1, .monotonic);
        const rejected = try serveOneForTest(request);
        defer t.allocator.free(rejected);
        try t.expect(std.mem.startsWith(u8, rejected, "HTTP/1.1 413 "));
        try t.expect(std.mem.indexOf(u8, rejected, "limit (server.max_request_mb)") != null);
        server_state.max_request_bytes.store(size, .monotonic);
        const accepted = try serveOneForTest(request);
        defer t.allocator.free(accepted);
        try t.expect(std.mem.startsWith(u8, accepted, "HTTP/1.1 200 "));
    }
}

test "chat page: GET / and GET /chat serve the embedded page as HTML" {
    const t = std.testing;
    for ([_][]const u8{ "/", "/chat", "/?utm=x" }) |path| {
        const request = try std.fmt.allocPrint(t.allocator, "GET {s} HTTP/1.1\r\nHost: localhost\r\n\r\n", .{path});
        defer t.allocator.free(request);
        const response = try serveOneForTest(request);
        defer t.allocator.free(response);
        try t.expect(std.mem.startsWith(u8, response, "HTTP/1.1 200 OK\r\n"));
        try t.expect(std.mem.indexOf(u8, response, "\r\nContent-Type: text/html; charset=utf-8\r\n") != null);
        try t.expectEqualStrings(@embedFile("webui/index.html"), responseBody(response));
    }
}

test "dashboard page: GET /dashboard and GET /downloads serve the same embedded page as HTML" {
    const t = std.testing;
    for ([_][]const u8{ "/dashboard", "/dashboard?x=1", "/downloads" }) |path| {
        const request = try std.fmt.allocPrint(t.allocator, "GET {s} HTTP/1.1\r\nHost: localhost\r\n\r\n", .{path});
        defer t.allocator.free(request);
        const response = try serveOneForTest(request);
        defer t.allocator.free(response);
        try t.expect(std.mem.startsWith(u8, response, "HTTP/1.1 200 OK\r\n"));
        try t.expect(std.mem.indexOf(u8, response, "\r\nContent-Type: text/html; charset=utf-8\r\n") != null);
        try t.expectEqualStrings(@embedFile("webui/index.html"), responseBody(response));
    }
}

test "dashboard page: every other method is a 405 and /dashboard/x is not the page" {
    const t = std.testing;
    for ([_][]const u8{ "POST", "PUT", "DELETE", "PATCH" }) |method| {
        const request = try std.fmt.allocPrint(t.allocator, "{s} /dashboard HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{{}}", .{method});
        defer t.allocator.free(request);
        const response = try serveOneForTest(request);
        defer t.allocator.free(response);
        try t.expect(std.mem.startsWith(u8, response, "HTTP/1.1 405 Method Not Allowed\r\n"));
    }
    const response = try serveOneForTest("GET /dashboard/x HTTP/1.1\r\n\r\n");
    defer t.allocator.free(response);
    try t.expect(!std.mem.startsWith(u8, response, "HTTP/1.1 200 OK\r\n"));
}

test "dashboard page: --api-key leaves the page open and keeps the metrics feed behind the key" {
    const t = std.testing;
    const prev_key = server_state.g_api_key;
    const prev_strict = server_state.g_api_key_strict;
    defer {
        server_state.g_api_key = prev_key;
        server_state.g_api_key_strict = prev_strict;
    }
    server_state.g_api_key = "s3cret";
    server_state.g_api_key_strict = true;
    const Case = struct { request: []const u8, status: []const u8 };
    for ([_]Case{
        .{ .request = "GET /dashboard HTTP/1.1\r\n\r\n", .status = "HTTP/1.1 200 OK\r\n" },
        .{ .request = "POST /dashboard HTTP/1.1\r\n\r\n", .status = "HTTP/1.1 401 Unauthorized\r\n" },
        .{ .request = "GET /metrics.json HTTP/1.1\r\n\r\n", .status = "HTTP/1.1 401 Unauthorized\r\n" },
    }) |c| {
        const response = try serveOneForTest(c.request);
        defer t.allocator.free(response);
        try t.expect(std.mem.startsWith(u8, response, c.status));
    }
}

test "metrics enable: any other method is a 405 and leaves metrics off" {
    const t = std.testing;
    instr.process = .{};
    defer instr.process = .{};
    for ([_][]const u8{ "GET", "PUT", "DELETE" }) |method| {
        const request = try std.fmt.allocPrint(t.allocator, "{s} /metrics/enable HTTP/1.1\r\n\r\n", .{method});
        defer t.allocator.free(request);
        const response = try serveOneForTest(request);
        defer t.allocator.free(response);
        try t.expect(std.mem.startsWith(u8, response, "HTTP/1.1 405 "));
    }
    try t.expect(instr.current() == null);
}

test "metrics enable: --api-key binds the route, and a foreign Origin is refused before it acts" {
    const t = std.testing;
    instr.process = .{};
    defer instr.process = .{};
    const prev_key = server_state.g_api_key;
    const prev_strict = server_state.g_api_key_strict;
    const old_host = server_state.listen_host;
    const old_port = server_state.listen_port;
    defer {
        server_state.g_api_key = prev_key;
        server_state.g_api_key_strict = prev_strict;
        server_state.listen_host = old_host;
        server_state.listen_port = old_port;
    }
    server_state.g_api_key = "s3cret";
    server_state.g_api_key_strict = true;
    server_state.listen_host = "127.0.0.1";
    server_state.listen_port = 12345;
    const keyless = try serveOneForTest("POST /metrics/enable HTTP/1.1\r\nContent-Length: 0\r\n\r\n");
    defer t.allocator.free(keyless);
    try t.expect(std.mem.startsWith(u8, keyless, "HTTP/1.1 401 "));
    const wrong = try serveOneForTest("POST /metrics/enable HTTP/1.1\r\nAuthorization: Bearer nope\r\nContent-Length: 0\r\n\r\n");
    defer t.allocator.free(wrong);
    try t.expect(std.mem.startsWith(u8, wrong, "HTTP/1.1 401 "));
    try t.expect(instr.current() == null);

    const foreign = try serveOneForTest("POST /metrics/enable HTTP/1.1\r\nAuthorization: Bearer s3cret\r\nOrigin: http://evil.test\r\nContent-Length: 0\r\n\r\n");
    defer t.allocator.free(foreign);
    try t.expect(std.mem.startsWith(u8, foreign, "HTTP/1.1 403 "));
    try t.expect(instr.current() == null);

    const ok = try serveOneForTest("POST /metrics/enable HTTP/1.1\r\nAuthorization: Bearer s3cret\r\nOrigin: http://localhost:12345\r\nContent-Length: 0\r\n\r\n");
    defer t.allocator.free(ok);
    try t.expect(std.mem.startsWith(u8, ok, "HTTP/1.1 200 OK\r\n"));
    try t.expect(instr.current() != null);
}

test "routes: a wrong method is a 405 with Allow, HEAD is a GET without its body, an unserved sub-path says so" {
    const t = std.testing;
    const Case = struct { request: []const u8, status: []const u8, has: []const u8 };
    for ([_]Case{
        .{ .request = "PUT /v1/chat/completions HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}", .status = "HTTP/1.1 405 ", .has = "Allow: POST\r\n" },
        .{ .request = "GET /v1/chat/completions HTTP/1.1\r\n\r\n", .status = "HTTP/1.1 405 ", .has = "Allow: POST\r\n" },
        .{ .request = "POST /health HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}", .status = "HTTP/1.1 405 ", .has = "Allow: GET, HEAD\r\n" },
        .{ .request = "HEAD /v1/chat/completions HTTP/1.1\r\n\r\n", .status = "HTTP/1.1 405 ", .has = "Allow: POST\r\n" },
        .{ .request = "HEAD /health HTTP/1.1\r\n\r\n", .status = "HTTP/1.1 200 OK\r\n", .has = "Content-Length: 15\r\n" },
        .{ .request = "GET /v1/models/no-such HTTP/1.1\r\n\r\n", .status = "HTTP/1.1 404 ", .has = "Unknown model id" },
        .{ .request = "GET /v1/responses/resp_1/input_items HTTP/1.1\r\n\r\n", .status = "HTTP/1.1 404 ", .has = "sub-resource is not supported" },
    }) |c| {
        const response = try serveOneForTest(c.request);
        defer t.allocator.free(response);
        errdefer std.debug.print("\n{s}\n", .{response});
        try t.expect(std.mem.startsWith(u8, response, c.status));
        try t.expect(std.mem.indexOf(u8, response, c.has) != null);
        if (std.mem.startsWith(u8, c.request, "HEAD")) try t.expectEqual(@as(usize, 0), responseBody(response).len);
    }
}

test "chat page: another method on the page paths is refused before any model resolves" {
    const t = std.testing;
    const response = try serveOneForTest("POST / HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}");
    defer t.allocator.free(response);
    try t.expect(std.mem.startsWith(u8, response, "HTTP/1.1 405 Method Not Allowed\r\n"));
}

test "chat page: the API routes beside it answer as before" {
    const t = std.testing;
    const Case = struct { request: []const u8, body: []const u8 };
    for ([_]Case{
        .{ .request = "GET /health HTTP/1.1\r\n\r\n", .body = "{\"status\":\"ok\"}" },
        .{ .request = "GET /v1/models HTTP/1.1\r\n\r\n", .body = "{\"object\":\"list\",\"data\":[]}" },
    }) |c| {
        const response = try serveOneForTest(c.request);
        defer t.allocator.free(response);
        try t.expect(std.mem.startsWith(u8, response, "HTTP/1.1 200 OK\r\n"));
        try t.expectEqualStrings(c.body, responseBody(response));
    }
    // Only the two exact paths are the page.
    for ([_][]const u8{ "GET /index.html HTTP/1.1\r\n\r\n", "GET /chat/x HTTP/1.1\r\n\r\n" }) |request| {
        const response = try serveOneForTest(request);
        defer t.allocator.free(response);
        try t.expect(!std.mem.startsWith(u8, response, "HTTP/1.1 200 OK\r\n"));
    }
}

test "chat page: --api-key leaves the page open and keeps the API behind the key" {
    const t = std.testing;
    const prev_key = server_state.g_api_key;
    const prev_strict = server_state.g_api_key_strict;
    defer {
        server_state.g_api_key = prev_key;
        server_state.g_api_key_strict = prev_strict;
    }
    // Strict so the loopback test peer is gated like a network one.
    server_state.g_api_key = "s3cret";
    server_state.g_api_key_strict = true;
    const Case = struct { request: []const u8, status: []const u8 };
    for ([_]Case{
        .{ .request = "GET / HTTP/1.1\r\n\r\n", .status = "HTTP/1.1 200 OK\r\n" },
        .{ .request = "GET /chat HTTP/1.1\r\n\r\n", .status = "HTTP/1.1 200 OK\r\n" },
        .{ .request = "POST / HTTP/1.1\r\n\r\n", .status = "HTTP/1.1 401 Unauthorized\r\n" },
        .{ .request = "GET /v1/models HTTP/1.1\r\n\r\n", .status = "HTTP/1.1 401 Unauthorized\r\n" },
        .{ .request = "GET /v1/models HTTP/1.1\r\nAuthorization: Bearer s3cret\r\n\r\n", .status = "HTTP/1.1 200 OK\r\n" },
    }) |c| {
        const response = try serveOneForTest(c.request);
        defer t.allocator.free(response);
        try t.expect(std.mem.startsWith(u8, response, c.status));
    }
    // The key is refused in each API's own error envelope; the scheme word is any case.
    const messages_401 = try serveOneForTest("POST /v1/messages HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}");
    defer t.allocator.free(messages_401);
    try t.expect(std.mem.startsWith(u8, responseBody(messages_401), "{\"type\":\"error\",\"error\":{\"type\":\"authentication_error\""));
    const lower = try serveOneForTest("GET /v1/models HTTP/1.1\r\nAuthorization: bearer s3cret\r\n\r\n");
    defer t.allocator.free(lower);
    try t.expect(std.mem.startsWith(u8, lower, "HTTP/1.1 200 OK\r\n"));
}

test "self-update: /v1/update refuses by name over the wire and never shuts the server down" {
    const t = std.testing;
    const prev_key = server_state.g_api_key;
    const old_port = server_state.listen_port;
    defer {
        server_state.g_api_key = prev_key;
        server_state.listen_port = old_port;
    }
    // The Origin cases name this port; a server test earlier in the binary leaves its own bound port behind.
    server_state.listen_port = 12345;
    // A test binary is no release install, in a checkout's cache or a gate's outside one: refused by the reason its
    // place gives, before asking whether a release is newer.
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    var why_buf: [512]u8 = undefined;
    const install = update_mod.selfInstallRefusal(t.io, &path_buf, &why_buf) orelse return error.TestBinaryIsAnInstall;
    const Case = struct { request: []const u8, status: []const u8, says: []const u8 };
    for ([_]Case{
        .{ .request = "GET /v1/update HTTP/1.1\r\n\r\n", .status = "HTTP/1.1 405 ", .says = "POST only" },
        .{ .request = "POST /v1/update HTTP/1.1\r\n\r\n", .status = "HTTP/1.1 403 ", .says = "no Origin" },
        .{ .request = "POST /v1/update HTTP/1.1\r\nOrigin: http://evil.test\r\n\r\n", .status = "HTTP/1.1 403 ", .says = "Origin" },
        .{ .request = "POST /v1/update HTTP/1.1\r\nOrigin: http://localhost:12345\r\n\r\n", .status = "HTTP/1.1 409 ", .says = install },
    }) |c| {
        const response = try serveOneForTest(c.request);
        defer t.allocator.free(response);
        try t.expect(std.mem.startsWith(u8, response, c.status));
        try t.expect(std.mem.indexOf(u8, responseBody(response), c.says) != null);
    }
    // The key binds this route from loopback too: loopback is otherwise exempt.
    server_state.g_api_key = "s3cret";
    const keyless = try serveOneForTest("POST /v1/update HTTP/1.1\r\nOrigin: http://localhost:12345\r\n\r\n");
    defer t.allocator.free(keyless);
    try t.expect(std.mem.startsWith(u8, keyless, "HTTP/1.1 401 "));
    try t.expect(!update_mod.relaunch_requested.load(.acquire));
    try t.expect(!server_state.shutdown_requested.load(.acquire));
}

test "chat page URL: a wildcard bind is opened through loopback" {
    const t = std.testing;
    var buf: [64]u8 = undefined;
    try t.expectEqualStrings("http://127.0.0.1:12345/", chatPageUrl(&buf, "127.0.0.1", 12345));
    try t.expectEqualStrings("http://127.0.0.1:8080/", chatPageUrl(&buf, "0.0.0.0", 8080));
    try t.expectEqualStrings("http://localhost:1/", chatPageUrl(&buf, "localhost", 1));
    try t.expectEqualStrings("http://192.168.1.5:18800/", chatPageUrl(&buf, "192.168.1.5", 18800));
}

test "routeExists answers endpoint existence without consulting the model" {
    // Whether a path EXISTS has nothing to do with which models are loaded, but
    // model resolution runs ahead of dispatch — so on a headless boot (`--serve
    // --model-dir`, no `--model`, how the app always launches) an unknown path
    // never reached the 404 branch and came back 503 "No default model
    // configured". Any client that maps endpoints by probing them then concludes
    // the server implements EVERYTHING (llmprobe: "server answers unknown paths
    // with HTTP 503" → every surface scored absent, live 2026-07-25).
    try std.testing.expect(routeExists("/v1/chat/completions"));
    try std.testing.expect(!routeExists("/v1/embeddings")); // not served: an unknown path
    try std.testing.expect(routeExists("/health"));
    try std.testing.expect(routeExists("/downloads") and routeExists("/v1/downloads") and routeExists("/v1/downloads/cancel"));
    // `/v1/responses/{id}` is served by prefix, not by an exact literal.
    try std.testing.expect(routeExists("/v1/responses/resp_abc123"));
    try std.testing.expect(routeExists("/v1/responses/compact"));

    try std.testing.expect(!routeExists("/v1/__llmprobe_no_such_endpoint__"));
    try std.testing.expect(!routeExists("/v1/audio/transcriptions")); // real OpenAI route we don't serve
    try std.testing.expect(!routeExists("/v1/images/generations")); // media generation left this build
    try std.testing.expect(!routeExists("/nope"));
    try std.testing.expect(!routeExists(""));
}

// ─── WebSocket transport for /v1/responses ────────────────────────────────

const WsConnT = ws_mod.WsConn(Conn);

/// Connection-local cache for `store: false` continuations on a WS session.
/// Each entry owns its arena (StoredResponse.deinit frees both).
const WsLocalCache = struct {
    map: std.StringHashMapUnmanaged(*responses_mod.StoredResponse) = .{},
    gpa: std.mem.Allocator,

    fn put(self: *WsLocalCache, sr: *responses_mod.StoredResponse) !void {
        if (self.map.fetchRemove(sr.id)) |kv| kv.value.deinit();
        try self.map.put(self.gpa, sr.id, sr);
    }

    fn get(self: *WsLocalCache, id: []const u8) ?*responses_mod.StoredResponse {
        return self.map.get(id);
    }

    fn evict(self: *WsLocalCache, id: []const u8) void {
        if (self.map.fetchRemove(id)) |kv| kv.value.deinit();
    }

    fn deinit(self: *WsLocalCache) void {
        var it = self.map.valueIterator();
        while (it.next()) |sr_ptr| sr_ptr.*.deinit();
        self.map.deinit(self.gpa);
    }
};

/// Returns the call_id of the first `function_call_output` input item that
/// has no matching `function_call` in the prev response's history (or in
/// the same input list, for parallel tool-call replies). Returns null if
/// every output has a matching call.
fn orphanFunctionCallOutputId(input_val: std.json.Value, prev_history: ?[]const chat_mod.Message) ?[]const u8 {
    if (input_val != .array) return null;
    // Collect all known call_ids from (a) prev history and (b) current input's
    // function_call items.
    var known: std.StringHashMapUnmanaged(void) = .{};
    defer known.deinit(std.heap.page_allocator);
    for (prev_history orelse &.{}) |m| {
        if (m.tool_calls) |tcs| for (tcs) |tc| {
            _ = known.put(std.heap.page_allocator, tc.id, {}) catch {};
        };
    }
    for (input_val.array.items) |item| {
        if (item != .object) continue;
        const t_v = item.object.get("type") orelse continue;
        if (t_v != .string or !std.mem.eql(u8, t_v.string, "function_call")) continue;
        const cid_v = item.object.get("call_id") orelse continue;
        if (cid_v == .string) {
            _ = known.put(std.heap.page_allocator, cid_v.string, {}) catch {};
        }
    }
    // Now check each function_call_output.
    for (input_val.array.items) |item| {
        if (item != .object) continue;
        const t_v = item.object.get("type") orelse continue;
        if (t_v != .string or !std.mem.eql(u8, t_v.string, "function_call_output")) continue;
        const cid_v = item.object.get("call_id") orelse return "";
        if (cid_v != .string) return "";
        if (!known.contains(cid_v.string)) return cid_v.string;
    }
    return null;
}

fn wsBridgeSend(impl: *anyopaque, data: []const u8) anyerror!void {
    const ws_conn: *WsConnT = @ptrCast(@alignCast(impl));
    try ws_conn.writeText(data);
}

/// Send a JSON error frame for a single WS turn. The compliance suite's
/// frame handler treats a `{"type":"error"}` text frame as terminal —
/// emitting a trailing `[DONE]` would land in the *next* turn's bucket
/// (same bug as success-path: see `handleResponsesWebSocket`).
/// A WS turn's generation failure: the error frame, with the mapping every HTTP surface takes.
fn wsGenerationErrorTurn(allocator: std.mem.Allocator, ws_conn: *WsConnT, err: anyerror) !void {
    const ws_wire = try mapGenerationError(allocator, err);
    defer ws_wire.deinit(allocator);
    return wsSendErrorTurn(allocator, ws_conn, ws_wire.code, ws_wire.openai_type, ws_wire.message);
}

fn wsSendErrorTurn(allocator: std.mem.Allocator, ws_conn: *WsConnT, status: u32, code: []const u8, message: []const u8) !void {
    const err_frame = try http.wsErrorFrame(allocator, status, code, message);
    defer allocator.free(err_frame);
    try ws_conn.writeText(err_frame);
}

/// Drive a WebSocket connection that bridges to /v1/responses.
///
/// Each text frame is a `response.create`-shaped JSON message. We translate
/// it into an HTTP-like body and reuse `handleResponses` (with a
/// `Conn.ws_mode` bridge) so all SSE events become WS text frames. After
/// each turn we emit `[DONE]` and wait for the next message on the same
/// connection, supporting chained `response.create` calls and
/// `previous_response_id` continuations.
pub fn handleResponsesWebSocket(
    allocator: std.mem.Allocator,
    stream: *Conn,
    headers: []const u8,
    models: TurnModels,
) !void {
    ws_mod.handshake(stream, headers) catch |err| switch (err) {
        error.WsHandshake => {
            log.warn("WS handshake refused: no Sec-WebSocket-Key\n", .{});
            return sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "A WebSocket upgrade needs its Sec-WebSocket-Key header", 400);
        },
        else => return err,
    };
    log.info("WS /v1/responses connected\n", .{});

    var ws_conn = WsConnT.init(stream, &server_state.max_request_bytes);
    ws_conn.frame_limit_ms = server_mod.REQUEST_READ_LIMIT_MS;
    defer ws_conn.deinit(allocator);
    var reason_buf: [123]u8 = undefined;
    const end = wsServeTurns(allocator, stream, &ws_conn, models) catch |err| wsEndOf(err, ws_conn.message_limit, &reason_buf);
    if (end.code) |code| ws_conn.writeClose(code, end.reason) catch {};
}

/// How a WebSocket turn gets its model, as its HTTP twin does: `hold` routes and holds the turn's `model`, or sends
/// the turn's refusal and gives null; `release` ends the hold when the turn ends.
pub const TurnModels = struct {
    hold: *const fn (std.mem.Allocator, *Conn, []const u8) anyerror!?*LoadedModel,
    release: *const fn (*LoadedModel) void,
};

/// How a Responses WebSocket ends: the close frame `handleResponsesWebSocket` sends, none when the peer is gone.
const WsEnd = struct { code: ?u16, reason: []const u8 = "" };

fn wsEndOf(err: anyerror, message_limit: usize, reason_buf: *[123]u8) WsEnd {
    return switch (err) {
        error.WsClosed => .{ .code = null },
        error.Shutdown => .{ .code = 1001, .reason = "server shutting down" },
        error.WsProtocol => .{ .code = 1002, .reason = "protocol error" },
        error.WsFraming => .{ .code = 1009, .reason = "empty fragments over the 8 KiB framing allowance" },
        error.ReadStalled => .{ .code = 1008, .reason = "a frame stalled past the read limit" },
        error.WsTooLarge => blk: {
            const msg = payloadTooLargeMessage(reason_buf, null, message_limit);
            log.warn("[http] WS 1009: {s}\n", .{msg});
            break :blk .{ .code = 1009, .reason = msg };
        },
        else => blk: {
            log.warn("WS /v1/responses ended: {s}\n", .{@errorName(err)});
            break :blk .{ .code = 1011, .reason = "internal error" };
        },
    };
}

/// The frame loop, one iteration per inbound message, until the socket ends: every end returns or raises here, so
/// one sender closes it.
fn wsServeTurns(allocator: std.mem.Allocator, stream: *Conn, ws_conn: *WsConnT, models: TurnModels) !WsEnd {
    var local_cache: WsLocalCache = .{ .gpa = allocator };
    defer local_cache.deinit();

    var bridge: WsBridge = .{ .impl = ws_conn, .sendTextFn = &wsBridgeSend, .allocator = allocator };
    defer bridge.reset();

    while (true) {
        if (server_state.shutdown_requested.load(.acquire)) return error.Shutdown;
        const msg = try ws_conn.readMessage(allocator);
        defer allocator.free(msg.payload);

        switch (msg.opcode) {
            .close => return .{ .code = 1000 },
            .ping => {
                try ws_conn.writePong(msg.payload);
                continue;
            },
            .pong => continue,
            .text => {},
            .binary => {
                try wsSendErrorTurn(allocator, ws_conn, 400, "invalid_request_error", "A Responses WebSocket message is JSON in a text frame; a binary frame is not accepted");
                return .{ .code = 1003, .reason = "binary frames are not accepted" };
            },
            else => return error.WsProtocol,
        }

        if (chat_mod.jsonNestingExceeds(msg.payload, chat_mod.max_json_nesting)) {
            try wsSendErrorTurn(allocator, ws_conn, 400, "invalid_request_error", "JSON nesting is too deep");
            continue;
        }
        // Parse the request payload — must be {"type":"response.create", ...}
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, msg.payload, .{}) catch {
            try wsSendErrorTurn(allocator, ws_conn, 400, "invalid_request_error", "Invalid JSON in request body");
            continue;
        };
        defer parsed.deinit();
        if (parsed.value != .object) {
            try wsSendErrorTurn(allocator, ws_conn, 400, "invalid_request_error", "Request must be a JSON object");
            continue;
        }
        const root = parsed.value.object;
        const type_val = root.get("type");
        if (type_val == null or type_val.? != .string or !std.mem.eql(u8, type_val.?.string, "response.create")) {
            try wsSendErrorTurn(allocator, ws_conn, 400, "invalid_request_error", "Expected type=response.create");
            continue;
        }
        // The plan disallows stream/stream_options/background here — WS is
        // inherently streaming and stateless (no background jobs).
        if (root.get("stream") != null or root.get("stream_options") != null or root.get("background") != null) {
            try wsSendErrorTurn(allocator, ws_conn, 400, "invalid_request_error", "stream/stream_options/background are not allowed over WebSocket");
            continue;
        }

        // ── Resolve previous_response_id: a `store:false` chain root lives in `local_cache` on this connection only,
        //    a stored one in the shared store, held for the whole turn. On a miss, any stale local entry goes.
        const prev_id_owned: ?[]const u8 = if (root.get("previous_response_id")) |v|
            (if (v == .string) v.string else null)
        else
            null;
        const store = responses_api.responseStore();
        var prev_history: ?[]const chat_mod.Message = null;
        var prev_held: ?*responses_mod.StoredResponse = null;
        defer if (prev_held) |sr| store.release(sr);
        if (prev_id_owned) |pid| {
            if (local_cache.get(pid)) |sr| {
                prev_history = sr.history;
            } else if (store.acquire(pid)) |sr| {
                prev_held = sr;
                prev_history = sr.history;
            } else {
                local_cache.evict(pid);
                try wsSendErrorTurn(allocator, ws_conn, 404, "previous_response_not_found", "previous_response_id not found");
                continue;
            }
        }
        const prev_in_local = prev_id_owned != null and prev_held == null;

        // Validate any function_call_output items reference real call_ids
        // in the prev response's history. An orphan output means the user
        // is trying to feed back a tool result for a call the model never
        // made — the turn must fail and (per compliance) evict the chain
        // root from the local cache.
        if (prev_id_owned) |pid| {
            if (root.get("input")) |input_v| {
                if (orphanFunctionCallOutputId(input_v, prev_history)) |_| {
                    local_cache.evict(pid);
                    try wsSendErrorTurn(allocator, ws_conn, 400, "invalid_request_error", "function_call_output references a missing call_id");
                    continue;
                }
            }
        }

        // ── Reserialize the request as an HTTP body, forcing
        //    `stream: true` (WS is inherently streaming) and `store: true`
        //    so handleResponses always keeps the result: in the shared store,
        //    or on this connection when the user asked for `store: false`.
        // The body below forces `store` true, so the client's own is checked and read here.
        var fault_buf: [160]u8 = undefined;
        if (request_parse.boolFieldFault(&fault_buf, root)) |why| {
            try wsSendErrorTurn(allocator, ws_conn, 400, "invalid_request_error", why);
            continue;
        }
        const want_user_store: bool = request_parse.boolField(root, .store) orelse true;
        const body = try buildResponsesBodyFromWsRequest(allocator, root);
        defer allocator.free(body);

        stream.ws_mode = &bridge;
        defer stream.ws_mode = null;

        bridge.reset();
        const model_id = if (root.get("model")) |v| (if (v == .string) v.string else "") else "";
        const lm = (try models.hold(allocator, stream, model_id)) orelse continue;
        defer models.release(lm);
        var kept: ?*responses_mod.StoredResponse = null;
        const turn_result = handleResponses(allocator, stream, body, lm, .{ .prev_history = prev_history, .kept = if (want_user_store) null else &kept });
        if (kept) |sr| local_cache.put(sr) catch sr.deinit();
        turn_result catch |err| {
            log.warn("WS handleResponses error: {s}\n", .{@errorName(err)});
            wsGenerationErrorTurn(allocator, ws_conn, err) catch {};
            continue;
        };

        // Compliance: a continuation of a local chain that ended failed or incomplete evicts its chain root.
        if (prev_in_local) {
            const turn_failed = bridge.captured_status != null and !std.mem.eql(u8, bridge.captured_status.?, "completed");
            if (turn_failed) local_cache.evict(prev_id_owned.?);
        }

        // No `[DONE]` on the WS success path: over WebSocket the per-response
        // terminator is `response.completed` (or .failed/.incomplete), and the
        // compliance suite advances to the next turn the moment it sees one. A
        // trailing `[DONE]` would arrive *after* that advance and be misread
        // as the next turn's marker, killing chained sessions. HTTP SSE is the
        // opposite: one response per stream, and OpenAI ends it with
        // `data: [DONE]` (emitted in handleResponses, gated on ws_mode). We
        // still emit `[DONE]` for error fallbacks (see `wsSendErrorTurn`)
        // where no terminal event is sent.
    }
}

/// Reshape a WS `response.create` JSON object into an OpenAI-Responses HTTP
/// request body. We strip the `type` discriminator and force both
/// `stream: true` (WS is inherently streaming) and `store: true` (so
/// handleResponses persists the response in the global store, where the WS
/// handler can later move it into the connection-local cache when the user
/// requested `store: false`).
fn buildResponsesBodyFromWsRequest(allocator: std.mem.Allocator, root: std.json.ObjectMap) ![]u8 {
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);
    try buf.append(allocator, '{');
    var first = true;
    var stream_seen = false;
    var store_seen = false;
    var iter = root.iterator();
    while (iter.next()) |entry| {
        const k = entry.key_ptr.*;
        if (std.mem.eql(u8, k, "type")) continue;
        if (std.mem.eql(u8, k, "stream")) stream_seen = true;
        if (std.mem.eql(u8, k, "store")) store_seen = true;
        if (!first) try buf.append(allocator, ',');
        first = false;
        const ek = try jsonEscape(allocator, k);
        defer allocator.free(ek);
        try buf.appendSlice(allocator, ek);
        try buf.append(allocator, ':');
        if (std.mem.eql(u8, k, "stream") or std.mem.eql(u8, k, "store")) {
            try buf.appendSlice(allocator, "true");
        } else {
            try json_text.appendValue(allocator, &buf, entry.value_ptr.*);
        }
    }
    if (!stream_seen) {
        if (!first) try buf.append(allocator, ',');
        first = false;
        try buf.appendSlice(allocator, "\"stream\":true");
    }
    if (!store_seen) {
        if (!first) try buf.append(allocator, ',');
        try buf.appendSlice(allocator, "\"store\":true");
    }
    try buf.append(allocator, '}');
    return try buf.toOwnedSlice(allocator);
}

test "Conn.peerClosed: alive socket returns false, closed peer returns true" {
    // Create a connected socket pair (AF_UNIX SOCK_STREAM via socketpair).
    var sv: [2]std.posix.fd_t = undefined;
    const AF_UNIX: c_uint = 1;
    const SOCK_STREAM: c_uint = 1;
    const rc = std.c.socketpair(AF_UNIX, SOCK_STREAM, 0, &sv);
    try testing.expect(rc == 0);

    const server_fd = sv[0];
    const client_fd = sv[1];

    // Build a Conn that wraps the server-side fd. We only need
    // `stream.socket.handle` for peerClosed, so the Reader/Writer state
    // can stay zeroed.
    var conn: Conn = undefined;
    conn.stream = .{ .socket = .{ .handle = server_fd, .address = undefined } };

    // Healthy connection: no data pending, no FIN → peerClosed returns false.
    try testing.expect(!conn.peerClosed());

    // Client closes its side → server should observe FIN/HUP.
    _ = std.c.close(client_fd);

    // socketpair() returns connected sockets in the kernel; close-of-peer
    // is observable immediately on the other side without delay.
    const closed = conn.peerClosed();
    _ = std.c.close(server_fd);
    try testing.expect(closed);
}

/// Runs `Conn.close` after one response, with a peer that stays connected, and checks whether it
/// held the socket (`close_delimited`) or returned at once.
fn expectCloseWaitsForPeer(close_delimited: bool, stop_server: bool) !void {
    Conn.linger_ms = 20;
    defer Conn.linger_ms = Conn.LINGER_MS;
    const was_shutdown = server_state.shutdown_requested.swap(false, .acq_rel);
    defer server_state.shutdown_requested.store(was_shutdown, .release);
    var sv: [2]std.posix.fd_t = undefined;
    const AF_UNIX: c_uint = 1;
    const SOCK_STREAM: c_uint = 1;
    try testing.expect(std.c.socketpair(AF_UNIX, SOCK_STREAM, 0, &sv) == 0);

    var conn: Conn = undefined;
    Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = undefined } }, testing.io);
    const tail: []const u8 = if (close_delimited) "data: [DONE]\n\n" else "{}";
    if (close_delimited) {
        try sendSseHeaders(&conn, "test", SSE_ALLOW_HEADERS_DEFAULT);
        try conn.writeAll(tail);
    } else {
        try sendResponseFramed(&conn, "200 OK", "application/json", tail);
    }

    var closed = std.atomic.Value(bool).init(false);
    const closer = try std.Thread.spawn(.{}, struct {
        fn run(c: *Conn, done: *std.atomic.Value(bool)) void {
            c.close();
            done.store(true, .release);
        }
    }.run, .{ &conn, &closed });
    var client_open = true;
    defer closer.join();
    defer if (client_open) {
        _ = std.c.close(sv[1]);
    };

    // The reader gets the whole response and then EOF ...
    var buf: [512]u8 = undefined;
    var got: usize = 0;
    while (true) {
        const n = std.c.recv(sv[1], &buf[got], buf.len - got, 0);
        if (n <= 0) break;
        got += @intCast(n);
    }
    try testing.expect(std.mem.endsWith(u8, buf[0..got], tail));

    // ... a Content-Length body lingers only briefly, and only a close-delimited body keeps the server's end
    // open, so the kernel cannot reset a lagging reader. The held socket is watched for twice the linger; the
    // lingering one is given up to a second (the 5-minute close wait is what it must not take).
    var waited: u32 = 0;
    const watch_ms: u32 = if (close_delimited) @intCast(2 * Conn.linger_ms) else 1000;
    while (!closed.load(.acquire) and waited < watch_ms) : (waited += 1) {
        std.Io.sleep(testing.io, .fromMilliseconds(1), .real) catch {};
    }
    try testing.expectEqual(!close_delimited, closed.load(.acquire));

    if (stop_server) {
        server_state.shutdown_requested.store(true, .release);
        // A stray byte wakes the close's poll so it re-reads the flag now, not at the end of its 500 ms slice.
        _ = std.c.send(sv[1], "x", 1, 0);
    } else {
        _ = std.c.close(sv[1]);
        client_open = false;
    }
    waited = 0;
    while (!closed.load(.acquire) and waited < 3000) : (waited += 1) {
        std.Io.sleep(testing.io, .fromMilliseconds(1), .real) catch {};
    }
    try testing.expect(closed.load(.acquire));
}

test "Conn.close keeps the socket until the peer hangs up, after a close-delimited body" {
    try expectCloseWaitsForPeer(true, false);
}

test "Conn.close lingers only briefly for the peer after a Content-Length response" {
    try expectCloseWaitsForPeer(false, false);
}

test "Conn.close stops waiting when the server shuts down" {
    try expectCloseWaitsForPeer(true, true);
}

test "StreamHeartbeat: a write resets the deadline, silence expires it" {
    var hb = StreamHeartbeat{ .last_write_ms = 1_000, .interval_ms = 5_000 };

    // Silent, but not yet for a full interval.
    try testing.expect(!hb.due(1_000));
    try testing.expect(!hb.due(5_999));
    // Exactly the interval is due (the client's clock is not ours to trust).
    try testing.expect(hb.due(6_000));
    try testing.expect(hb.due(60_000));

    // Any byte written to the socket pushes the deadline out.
    hb.noteWrite(60_000);
    try testing.expect(!hb.due(60_001));
    try testing.expect(hb.due(65_000));
}

test "StreamHeartbeat: buffered tokens do NOT reset the deadline" {
    // The regression this guards: the old code beat only when NO token was
    // available (`.idle`). A tool call buffers tokens continuously while
    // writing nothing, so token arrival must not be mistaken for liveness —
    // only `noteWrite` (bytes on the socket) counts.
    var hb = StreamHeartbeat{ .last_write_ms = 0, .interval_ms = 5_000 };
    var now: i64 = 0;
    // 300 tokens arrive over 30s. Not one is written out (buffered tool call).
    var i: usize = 0;
    while (i < 300) : (i += 1) now += 100;
    try testing.expectEqual(@as(i64, 30_000), now);
    try testing.expect(hb.due(now));
}

test "effort notice rides the response header and does not replace the answer" {
    var sv: [2]std.posix.fd_t = undefined;
    try std.testing.expect(std.c.socketpair(1, 1, 0, &sv) == 0);
    defer _ = std.c.close(sv[1]);
    var conn: Conn = undefined;
    Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = undefined } }, std.testing.io);
    const notice = "Qwen3.8-Flash-Next has no `high` effort; served at `xhigh`";
    conn.effort_notice = notice;
    conn.effort_notice_json = ",\"effort_notice\":\"Qwen3.8-Flash-Next has no `high` effort; served at `xhigh`\"";
    try sendResponseFramed(&conn, "200 OK", "application/json", "{\"content\":\"page summary\"}");
    conn.close();
    var buf: [2048]u8 = undefined;
    var got: usize = 0;
    while (got < buf.len) {
        const n = std.c.recv(sv[1], buf[got..].ptr, buf.len - got, 0);
        if (n <= 0) break;
        got += @intCast(n);
    }
    const raw = buf[0..got];
    try std.testing.expect(std.mem.indexOf(u8, raw, "X-Sushi-Effort-Notice: " ++ notice ++ "\r\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, raw, "Access-Control-Expose-Headers: X-Sushi-Effort-Notice\r\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, raw, "\"content\":\"page summary\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, raw, "\"effort_notice\":\"" ++ notice ++ "\"") != null);
}

fn rawResponseFor(conn: *Conn, peer: std.posix.fd_t, buf: []u8) ![]u8 {
    // The peer ends its side first, as a client does: `close` waits for it, minutes on a close-delimited answer.
    _ = std.posix.system.shutdown(peer, std.posix.SHUT.WR);
    conn.close();
    var got: usize = 0;
    while (got < buf.len) {
        const n = std.c.recv(peer, buf[got..].ptr, buf.len - got, 0);
        if (n <= 0) break;
        got += @intCast(n);
    }
    return buf[0..got];
}

test "a client max_tokens the window cuts rides the response head of a JSON and an SSE answer; an uncut or filled one adds no header" {
    var sv: [2]std.posix.fd_t = undefined;
    try std.testing.expect(std.c.socketpair(1, 1, 0, &sv) == 0);
    defer _ = std.c.close(sv[1]);
    var conn: Conn = undefined;
    Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = undefined } }, std.testing.io);
    try std.testing.expectEqual(@as(u32, 96), request_policy.clampMaxTokensFor(&conn, 64000, true, 4000, 4096));
    try sendResponseFramed(&conn, "200 OK", "application/json", "{}");
    var buf: [2048]u8 = undefined;
    const json_raw = try rawResponseFor(&conn, sv[1], &buf);
    try std.testing.expect(std.mem.indexOf(u8, json_raw, "\r\nX-Sushi-Max-Tokens-Cut: 64000 -> 96\r\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, json_raw, "Access-Control-Expose-Headers: X-Sushi-Max-Tokens-Cut\r\n") != null);

    var sv2: [2]std.posix.fd_t = undefined;
    try std.testing.expect(std.c.socketpair(1, 1, 0, &sv2) == 0);
    defer _ = std.c.close(sv2[1]);
    var sse: Conn = undefined;
    Conn.init(&sse, .{ .socket = .{ .handle = sv2[0], .address = undefined } }, std.testing.io);
    _ = request_policy.clampMaxTokensFor(&sse, 64000, true, 4000, 4096);
    try sendSseHeaders(&sse, "chat", "Content-Type");
    const sse_raw = try rawResponseFor(&sse, sv2[1], &buf);
    try std.testing.expect(std.mem.indexOf(u8, sse_raw, "\r\nX-Sushi-Max-Tokens-Cut: 64000 -> 96\r\n") != null);

    for ([_]struct { u32, bool }{ .{ 50, true }, .{ request_policy.ENGINE_DEFAULT_MAX_TOKENS, false }, .{ 64000, false } }) |case| {
        var sv3: [2]std.posix.fd_t = undefined;
        try std.testing.expect(std.c.socketpair(1, 1, 0, &sv3) == 0);
        defer _ = std.c.close(sv3[1]);
        var plain: Conn = undefined;
        Conn.init(&plain, .{ .socket = .{ .handle = sv3[0], .address = undefined } }, std.testing.io);
        _ = request_policy.clampMaxTokensFor(&plain, case[0], case[1], 4000, 4096);
        try sendResponseFramed(&plain, "200 OK", "application/json", "{}");
        const raw = try rawResponseFor(&plain, sv3[1], &buf);
        try std.testing.expect(std.mem.indexOf(u8, raw, "X-Sushi-Max-Tokens-Cut") == null);
    }
}

test "isTextGenRoute covers exactly the guarded surfaces" {
    try std.testing.expect(isTextGenRoute("POST", "/v1/chat/completions"));
    try std.testing.expect(isTextGenRoute("POST", "/v1/completions"));
    try std.testing.expect(isTextGenRoute("POST", "/v1/messages"));
    try std.testing.expect(isTextGenRoute("POST", "/v1/responses"));
    try std.testing.expect(isTextGenRoute("GET", "/v1/responses")); // WS upgrade
    try std.testing.expect(!isTextGenRoute("GET", "/v1/models"));
}

test "isStatusRoute: only /props reaches the shared release" {
    try std.testing.expect(isStatusRoute("GET", "/props"));
    // Generation is use.
    try std.testing.expect(!isStatusRoute("POST", "/v1/chat/completions"));
    try std.testing.expect(!isStatusRoute("POST", "/v1/completions"));
    // An explicit load is use — it is the whole point of the request.
    try std.testing.expect(!isStatusRoute("POST", "/v1/load-model"));
    // Method matters: the path alone is not the predicate.
    try std.testing.expect(!isStatusRoute("POST", "/props"));
}

test "POST /v1/completions with a prompt that tokenizes to nothing answers 400 before any generation" {
    const t = std.testing;
    const a = t.allocator;
    var tok = Tokenizer.initEmpty(a);
    defer tok.deinit();
    var cfg = model_mod.ModelConfig{ .num_hidden_layers = 0, .pinned_context = 4096 };
    var lm: LoadedModel = undefined;
    lm.id = "";
    lm.config = &cfg;
    lm.tokenizer = &tok;
    lm.handle = null;
    lm.dflash = null;
    lm.mtp = null;

    var sv: [2]std.posix.fd_t = undefined;
    try t.expect(std.c.socketpair(1, 1, 0, &sv) == 0); // AF_UNIX, SOCK_STREAM
    defer _ = std.c.close(sv[1]);
    var conn: Conn = undefined;
    Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = undefined } }, t.io);
    try handleCompletions(a, &conn, "{\"prompt\":\"\",\"max_tokens\":4}", &lm);
    _ = std.c.close(sv[0]);

    var buf: [4096]u8 = undefined;
    var got: usize = 0;
    while (got < buf.len) {
        const n = std.c.read(sv[1], buf[got..].ptr, buf.len - got);
        if (n <= 0) break;
        got += @intCast(n);
    }
    try t.expect(std.mem.startsWith(u8, buf[0..got], "HTTP/1.1 400 Bad Request\r\n"));
    try t.expect(std.mem.indexOf(u8, buf[0..got], "\"invalid_request_error\"") != null);
    try t.expect(std.mem.indexOf(u8, buf[0..got], EMPTY_PROMPT_MSG) != null);
}

test "the 413 names both counts it compared, and the setting that moves the limit" {
    var buf: [128]u8 = undefined;
    try std.testing.expectEqualStrings("Request body too large: 98 MB exceeds the 64 MB limit (server.max_request_mb)", payloadTooLargeMessage(&buf, 97 * 1024 * 1024 + 1, 64 * 1024 * 1024));
    try std.testing.expectEqualStrings("Request body too large: over the 64 MB limit (server.max_request_mb)", payloadTooLargeMessage(&buf, null, 64 * 1024 * 1024));
    // At the widest counts both still fit the 123 bytes a WebSocket close reason carries, so no 413 or 1009 is cut.
    var close: [123]u8 = undefined;
    const most = std.math.maxInt(usize);
    try std.testing.expect(std.mem.endsWith(u8, payloadTooLargeMessage(&close, most, most), "(server.max_request_mb)"));
    try std.testing.expect(std.mem.endsWith(u8, payloadTooLargeMessage(&close, null, most), "(server.max_request_mb)"));
}

test "startListener: a second bind on the same port is refused" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var p: u16 = 23850;
    var a: std.Io.net.Server = undefined;
    var got = false;
    while (p < 23950) : (p += 1) {
        a = startListener("127.0.0.1", p) catch |e| switch (e) {
            error.AddressInUse => continue,
            else => return e,
        };
        got = true;
        break;
    }
    if (!got) return error.NoFreeTestPort;
    defer a.deinit(io);
    try std.testing.expectError(error.AddressInUse, startListener("127.0.0.1", p));
}

test "one-shot listener reports its private ephemeral port" {
    var listener = try startListener("127.0.0.1", 0);
    defer listener.deinit(std.testing.io);
    try std.testing.expect(listener.socket.address.ip4.port != 0);
    try std.testing.expectEqualSlices(u8, &.{ 127, 0, 0, 1 }, &listener.socket.address.ip4.bytes);
}

test "one-shot shutdown closes an unaccepted connection before joining its client" {
    const Waiter = struct {
        fd: std.posix.socket_t,
        disconnected: bool = false,
        fn run(self: *@This()) void {
            var fds = [_]std.posix.pollfd{.{ .fd = self.fd, .events = std.posix.POLL.IN, .revents = 0 }};
            self.disconnected = (std.posix.poll(&fds, 500) catch 0) > 0;
        }
    };
    var listener = try startListener("127.0.0.1", 0);
    const client = try listener.socket.address.connect(std.testing.io, .{ .mode = .stream });
    defer client.close(std.testing.io);
    var waiter: Waiter = .{ .fd = client.socket.handle };
    const thread = try std.Thread.spawn(.{}, Waiter.run, .{&waiter});
    closeEmbeddedListener(std.testing.io, &listener, thread);
    try std.testing.expect(waiter.disconnected);
}

test "web tools: list, confined execution, and browser origin guard" {
    const t = std.testing;
    const old_host = server_state.listen_host;
    const old_port = server_state.listen_port;
    defer {
        server_state.listen_host = old_host;
        server_state.listen_port = old_port;
    }
    server_state.listen_host = "127.0.0.1";
    server_state.listen_port = 12345;
    const Case = struct { origin: []const u8, body: []const u8, status: []const u8, contains: []const u8, lacks: []const u8 = "" };
    const write_call = "{\"name\":\"write_file\",\"arguments\":\"{\\\"path\\\":\\\"../outside.md\\\",\\\"content\\\":\\\"x\\\"}\",\"write\":true}";
    const cases = [_]Case{
        .{ .origin = "http://localhost:12345", .body = "{}", .status = "200 OK", .contains = "web_search" },
        .{ .origin = "http://evil.test", .body = "{}", .status = "403 Forbidden", .contains = "Origin" },
        .{ .origin = "null", .body = "{}", .status = "403 Forbidden", .contains = "Origin" },
        .{ .origin = "http://localhost:12345", .body = "[]", .status = "400 Bad Request", .contains = "object" },
        .{ .origin = "http://localhost:12345", .body = "{\"browse\":true}", .status = "200 OK", .contains = "directories" },
        .{ .origin = "http://localhost:12345", .body = "{\"directory\":3}", .status = "400 Bad Request", .contains = "absolute" },
        .{ .origin = "http://localhost:12345", .body = "{\"directory\":\"/sushi-folder-does-not-exist\"}", .status = "400 Bad Request", .contains = "no such folder" },
        .{ .origin = "http://localhost:12345", .body = "{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"../outside\\\"}\"}", .status = "200 OK", .contains = "refused" },
        .{ .origin = "http://localhost:12345", .body = "{\"name\":\"exec\",\"arguments\":\"{}\"}", .status = "200 OK", .contains = "unknown tool" },
        // The chat's own write flag is the switch: with it the call reaches the confinement check,
        // which is what keeps the test from writing anything; without it the call is refused.
        .{ .origin = "http://localhost:12345", .body = write_call, .status = "200 OK", .contains = "outside the folder" },
        .{ .origin = "http://localhost:12345", .body = "{\"name\":\"edit_file\",\"arguments\":\"{\\\"path\\\":\\\"/etc/hosts\\\",\\\"old_string\\\":\\\"a\\\",\\\"new_string\\\":\\\"b\\\"}\",\"write\":true}", .status = "200 OK", .contains = "outside the folder" },
        .{ .origin = "http://localhost:12345", .body = "{\"name\":\"write_file\",\"arguments\":\"{\\\"path\\\":\\\"x.md\\\",\\\"content\\\":\\\"x\\\"}\"}", .status = "200 OK", .contains = "writing files is off" },
        .{ .origin = "http://localhost:12345", .body = "{}", .status = "200 OK", .contains = "\"edit_default\":false", .lacks = "write_file" },
        .{ .origin = "http://localhost:12345", .body = "{\"write\":true,\"vision\":true}", .status = "200 OK", .contains = "edit_file" },
        .{ .origin = "http://localhost:12345", .body = "{\"write\":\"yes\"}", .status = "400 Bad Request", .contains = "boolean" },
        .{ .origin = "http://evil.test", .body = write_call, .status = "403 Forbidden", .contains = "Origin" },
    };
    const old_edit = server_state.g_web_edit;
    defer server_state.g_web_edit = old_edit;
    for ([_]bool{ false, true }) |edit_default| {
        server_state.g_web_edit = edit_default;
        // --edit on only changes what a new chat starts with; it never writes on its own.
        const extra = [_]Case{.{ .origin = "http://localhost:12345", .body = "{}", .status = "200 OK", .contains = if (edit_default) "\"edit_default\":true" else "\"edit_default\":false", .lacks = "write_file" }};
        for (if (edit_default) &extra else &cases) |c| {
            const req = try std.fmt.allocPrint(t.allocator, "POST /v1/tools HTTP/1.1\r\nOrigin: {s}\r\nContent-Length: {d}\r\n\r\n{s}", .{ c.origin, c.body.len, c.body });
            defer t.allocator.free(req);
            const response = try serveOneForTest(req);
            defer t.allocator.free(response);
            try t.expect(std.mem.indexOf(u8, response, c.status) != null);
            try t.expect(std.mem.indexOf(u8, responseBody(response), c.contains) != null);
            if (c.lacks.len > 0) try t.expect(std.mem.indexOf(u8, responseBody(response), c.lacks) == null);
        }
    }
}

test "cross-origin policy: a foreign Origin is refused on the state-changing routes only" {
    const t = std.testing;
    const ws = "Upgrade: websocket\r\nConnection: Upgrade\r\n";
    const Case = struct { method: []const u8, path: []const u8, headers: []const u8, host: []const u8 = "127.0.0.1", refused: bool };
    for ([_]Case{
        .{ .method = "POST", .path = "/v1/load-model", .headers = "Origin: http://evil.test\r\n", .refused = true },
        .{ .method = "POST", .path = "/v1/unload-model", .headers = "Origin: null\r\n", .refused = true },
        .{ .method = "POST", .path = "/v1/models/rescan", .headers = "ORIGIN: http://127.0.0.1:12346\r\n", .refused = true },
        .{ .method = "POST", .path = "/v1/prefix-cache/clear", .headers = "Origin: http://evil.test\r\n", .refused = true },
        .{ .method = "POST", .path = "/v1/prefix-cache/clear", .headers = "Origin: http://localhost:12345\r\n", .refused = false },
        .{ .method = "GET", .path = "/v1/responses", .headers = ws ++ "Origin: http://evil.test\r\n", .refused = true },
        .{ .method = "POST", .path = "/metrics/enable", .headers = "Origin: http://evil.test\r\n", .refused = true },
        .{ .method = "GET", .path = "/v1/downloads", .headers = "Origin: http://evil.test\r\n", .refused = true },
        .{ .method = "POST", .path = "/v1/downloads", .headers = "Origin: http://evil.test\r\n", .refused = true },
        .{ .method = "POST", .path = "/v1/downloads/cancel", .headers = "Origin: http://evil.test\r\n", .refused = true },
        .{ .method = "POST", .path = "/v1/slots/cancel", .headers = "Origin: http://evil.test\r\n", .refused = true },
        .{ .method = "POST", .path = "/v1/slots/cancel", .headers = "Origin: http://localhost:12345\r\n", .refused = false },
        .{ .method = "POST", .path = "/v1/downloads", .headers = "Origin: http://localhost:12345\r\n", .refused = false },
        .{ .method = "POST", .path = "/metrics/enable", .headers = "Origin: http://localhost:12345\r\n", .refused = false },
        .{ .method = "POST", .path = "/v1/load-model", .headers = "Origin: http://localhost:12345\r\n", .refused = false },
        .{ .method = "POST", .path = "/v1/models/rescan", .headers = "Origin: http://127.0.0.1:12345\r\n", .refused = false },
        .{ .method = "GET", .path = "/v1/responses", .headers = ws ++ "Origin: http://127.0.0.1:12345\r\n", .refused = false },
        .{ .method = "POST", .path = "/v1/load-model", .headers = "Origin: http://localhost:12345\r\n", .host = "0.0.0.0", .refused = false },
        .{ .method = "POST", .path = "/v1/load-model", .headers = "Origin: http://192.168.1.5:12345\r\nHost: 192.168.1.5:12345\r\n", .host = "0.0.0.0", .refused = false },
        .{ .method = "POST", .path = "/v1/load-model", .headers = "Origin: http://192.168.1.5:12345\r\nHost: 192.168.1.5\r\n", .host = "0.0.0.0", .refused = false },
        .{ .method = "POST", .path = "/v1/load-model", .headers = "Origin: http://[::1]:12345\r\nHost: [::1]:12345\r\n", .host = "0.0.0.0", .refused = false },
        .{ .method = "POST", .path = "/v1/load-model", .headers = "Origin: http://evil.test:12345\r\nHost: 192.168.1.5:12345\r\n", .host = "0.0.0.0", .refused = true },
        // DNS rebinding: a foreign name pointed at this Mac sends its own name as both Origin and Host.
        .{ .method = "POST", .path = "/v1/load-model", .headers = "Origin: http://evil.test:12345\r\nHost: evil.test:12345\r\n", .host = "0.0.0.0", .refused = true },
        .{ .method = "POST", .path = "/v1/settings", .headers = "Origin: http://mac.example:12345\r\nHost: mac.example:12345\r\n", .host = "0.0.0.0", .refused = true },
        .{ .method = "POST", .path = "/v1/load-model", .headers = "Origin: http://localhost:12345\r\nHost: localhost:12345\r\n", .host = "0.0.0.0", .refused = false },
        .{ .method = "POST", .path = "/v1/load-model", .headers = "Origin: http://192.168.1.5:12346\r\nHost: 192.168.1.5:12345\r\n", .host = "0.0.0.0", .refused = true },
        .{ .method = "POST", .path = "/v1/unload-model", .headers = "Origin: http://192.168.1.5:12345\r\nHost: 127.0.0.1:12345\r\n", .host = "0.0.0.0", .refused = true },
        .{ .method = "POST", .path = "/v1/load-model", .headers = "Origin: http://192.168.1.5:12345\r\n", .host = "0.0.0.0", .refused = true },
        .{ .method = "POST", .path = "/v1/load-model", .headers = "Origin: http://evil.test\r\nHost: 192.168.1.5:12345\r\n", .refused = true },
        .{ .method = "POST", .path = "/v1/unload-model", .headers = "Host: x\r\n", .refused = false },
        .{ .method = "GET", .path = "/v1/responses", .headers = ws, .refused = false },
        // The inference routes stay open to any page.
        .{ .method = "POST", .path = "/v1/chat/completions", .headers = "Origin: http://evil.test\r\n", .refused = false },
        .{ .method = "GET", .path = "/v1/responses", .headers = "Origin: http://evil.test\r\n", .refused = false },
    }) |c| try t.expectEqual(c.refused, crossOriginRefused(c.method, c.path, c.headers, c.host, 12345));
}

test "cross-origin policy: the routes answer 403 over the wire before any model resolves" {
    const t = std.testing;
    const old_host = server_state.listen_host;
    const old_port = server_state.listen_port;
    defer {
        server_state.listen_host = old_host;
        server_state.listen_port = old_port;
    }
    server_state.listen_host = "127.0.0.1";
    server_state.listen_port = 12345;
    const Case = struct { request: []const u8, status: []const u8 };
    for ([_]Case{
        .{ .request = "POST /v1/load-model HTTP/1.1\r\nOrigin: http://evil.test\r\nContent-Length: 2\r\n\r\n{}", .status = "HTTP/1.1 403 " },
        .{ .request = "POST /v1/unload-model HTTP/1.1\r\nOrigin: http://evil.test\r\nContent-Length: 2\r\n\r\n{}", .status = "HTTP/1.1 403 " },
        .{ .request = "POST /v1/slots/cancel HTTP/1.1\r\nOrigin: http://evil.test\r\nContent-Length: 10\r\n\r\n{\"id\":123}", .status = "HTTP/1.1 403 " },
        .{ .request = "POST /v1/models/rescan HTTP/1.1\r\nOrigin: http://evil.test\r\n\r\n", .status = "HTTP/1.1 403 " },
        .{ .request = "GET /v1/responses HTTP/1.1\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nOrigin: http://evil.test\r\n\r\n", .status = "HTTP/1.1 403 " },
        .{ .request = "POST /v1/models/rescan HTTP/1.1\r\nOrigin: http://localhost:12345\r\n\r\n", .status = "HTTP/1.1 200 " },
        .{ .request = "POST /v1/models/rescan HTTP/1.1\r\n\r\n", .status = "HTTP/1.1 200 " },
    }) |c| {
        const response = try serveOneForTest(c.request);
        defer t.allocator.free(response);
        try t.expect(std.mem.startsWith(u8, response, c.status));
    }
    server_state.listen_host = "0.0.0.0";
    for ([_]Case{
        .{ .request = "POST /v1/models/rescan HTTP/1.1\r\nHost: 192.168.1.5:12345\r\nOrigin: http://192.168.1.5:12345\r\n\r\n", .status = "HTTP/1.1 200 " },
        .{ .request = "POST /v1/models/rescan HTTP/1.1\r\nHost: 192.168.1.5:12345\r\nOrigin: http://evil.test:12345\r\n\r\n", .status = "HTTP/1.1 403 " },
        .{ .request = "POST /v1/models/rescan HTTP/1.1\r\nHost: evil.test:12345\r\nOrigin: http://evil.test:12345\r\n\r\n", .status = "HTTP/1.1 403 " },
        .{ .request = "POST /v1/models/rescan HTTP/1.1\r\nHost: 127.0.0.1:12345\r\nOrigin: http://127.0.0.1:12345\r\n\r\n", .status = "HTTP/1.1 200 " },
        .{ .request = "POST /v1/models/rescan HTTP/1.1\r\nOrigin: http://192.168.1.5:12345\r\n\r\n", .status = "HTTP/1.1 403 " },
    }) |c| {
        const response = try serveOneForTest(c.request);
        defer t.allocator.free(response);
        try t.expect(std.mem.startsWith(u8, response, c.status));
    }
}

test "downloads: the routes refuse a bad request by name before any model resolves" {
    const t = std.testing;
    const Case = struct { request: []const u8, status: []const u8, says: []const u8 };
    for ([_]Case{
        .{ .request = "POST /v1/downloads HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}", .status = "HTTP/1.1 400 ", .says = "Send {" },
        .{ .request = "POST /v1/downloads HTTP/1.1\r\nContent-Length: 21\r\n\r\n{\"repo\":\"other/Pack\"}", .status = "HTTP/1.1 400 ", .says = "org beamster only" },
        .{ .request = "POST /v1/downloads/cancel HTTP/1.1\r\nContent-Length: 24\r\n\r\n{\"repo\":\"beamster/None\"}", .status = "HTTP/1.1 404 ", .says = "queued or running" },
        .{ .request = "POST /v1/downloads HTTP/1.1\r\nContent-Length: 22\r\n\r\n{\"repo\":\"beamster/..\"}", .status = "HTTP/1.1 400 ", .says = "org beamster only" },
        .{ .request = "POST /v1/downloads HTTP/1.1\r\nContent-Length: 30\r\n\r\n{\"repo\":\"beamster/Not-A-Pack\"}", .status = "HTTP/1.1 404 ", .says = "published Sushi pack" },
        .{ .request = "DELETE /v1/downloads HTTP/1.1\r\n\r\n", .status = "HTTP/1.1 405 ", .says = "GET" },
        .{ .request = "GET /v1/downloads/cancel HTTP/1.1\r\n\r\n", .status = "HTTP/1.1 405 ", .says = "POST" },
    }) |c| {
        const response = try serveOneForTest(c.request);
        defer t.allocator.free(response);
        try t.expect(std.mem.startsWith(u8, response, c.status));
        try t.expect(std.mem.indexOf(u8, responseBody(response), c.says) != null);
    }
}

test "slots: cancel route validates request and method" {
    const t = std.testing;
    const Case = struct { request: []const u8, status: []const u8, says: []const u8 };
    for ([_]Case{
        .{ .request = "GET /v1/slots/cancel HTTP/1.1\r\n\r\n", .status = "HTTP/1.1 405 ", .says = "POST only" },
        .{ .request = "POST /v1/slots/cancel HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}", .status = "HTTP/1.1 400 ", .says = "Send {\\\"id\\\": <request_id>}" },
        .{ .request = "POST /v1/slots/cancel HTTP/1.1\r\nContent-Length: 12\r\n\r\n{\"id\":\"bad\"}", .status = "HTTP/1.1 400 ", .says = "Send {\\\"id\\\": <request_id>}" },
    }) |c| {
        const response = try serveOneForTest(c.request);
        defer t.allocator.free(response);
        try t.expect(std.mem.startsWith(u8, response, c.status));
        try t.expect(std.mem.indexOf(u8, responseBody(response), c.says) != null);
    }
}

test "seedRequestBuf drops bytes past the allocation" {
    const buf = try std.testing.allocator.alloc(u8, 4);
    defer std.testing.allocator.free(buf);
    try std.testing.expectEqual(@as(usize, 4), seedRequestBuf(buf, "abcdefgh"));
    try std.testing.expectEqualStrings("abcd", buf);
    try std.testing.expectEqual(@as(usize, 2), seedRequestBuf(buf, "xy"));
}

/// Runs `readRequestWithin(limit_ms)` against a client that sends `sent`, then one byte per
/// `drip_ms` (0 = nothing more), and never finishes; true when the reader gave up on its own.
fn readerGivesUp(sent: []const u8, limit_ms: i64, drip_ms: u64) !bool {
    const t = std.testing;
    var sv: [2]std.posix.fd_t = undefined;
    try t.expect(std.c.socketpair(1, 1, 0, &sv) == 0); // AF_UNIX, SOCK_STREAM
    // Wide enough that `sent` lands whole, so the first read can take more than the head.
    const wide: c_int = 256 * 1024;
    _ = std.c.setsockopt(sv[0], std.posix.SOL.SOCKET, std.posix.SO.RCVBUF, std.mem.asBytes(&wide), @sizeOf(c_int));
    _ = std.c.setsockopt(sv[1], std.posix.SOL.SOCKET, std.posix.SO.SNDBUF, std.mem.asBytes(&wide), @sizeOf(c_int));
    var conn: Conn = undefined;
    Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = undefined } }, t.io);
    defer conn.close();
    _ = std.c.send(sv[1], sent.ptr, sent.len, 0);

    var done = std.atomic.Value(bool).init(false);
    const reader = try std.Thread.spawn(.{}, struct {
        fn run(c: *Conn, limit: i64, d: *std.atomic.Value(bool)) void {
            if (readRequestWithin(t.allocator, c, limit) catch null) |o| if (o == .request) t.allocator.free(o.request.buf);
            d.store(true, .release);
        }
    }.run, .{ &conn, limit_ms, &done });
    var waited: u64 = 0;
    while (!done.load(.acquire) and waited < 2000) : (waited += 10) {
        std.Io.sleep(t.io, .fromMilliseconds(10), .real) catch {};
        if (drip_ms > 0 and waited % drip_ms == 0) _ = std.c.send(sv[1], "x", 1, 0);
    }
    const gave_up = done.load(.acquire);
    _ = std.c.close(sv[1]);
    reader.join();
    return gave_up;
}

test "readRequest: a client that stalls its head or body, or drips its head, is dropped" {
    const head = "POST /v1/chat/completions HTTP/1.1\r\n";
    try std.testing.expect(try readerGivesUp(head, 40, 0));
    try std.testing.expect(try readerGivesUp(head, 60, 10));
    try std.testing.expect(try readerGivesUp(head ++ "Content-Length: 100\r\n\r\n{}", 40, 0));
}

test "readRequest: a body read past the reader's buffer and then stalled is dropped at the limit, by length or chunked" {
    const pad: [24_000]u8 = @splat('a');
    try std.testing.expect(try readerGivesUp("POST /v1/completions HTTP/1.1\r\nContent-Length: 40000\r\n\r\n" ++ pad, 40, 0));
    try std.testing.expect(try readerGivesUp("POST /v1/completions HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n9c40\r\n" ++ pad, 40, 0));
}

test "readRequest: a Content-Length that wraps the size sum is a 413" {
    const response = try serveOneForTest("POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 18446744073709551615\r\n\r\n{}");
    defer std.testing.allocator.free(response);
    try std.testing.expect(std.mem.startsWith(u8, response, "HTTP/1.1 413"));
}

/// What an SSE parser keeps: the events, with every comment block dropped.
fn sseEventsOnly(allocator: std.mem.Allocator, raw: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var blocks = std.mem.splitSequence(u8, raw, "\n\n");
    while (blocks.next()) |block| {
        if (block.len == 0 or block[0] == ':') continue;
        try out.appendSlice(allocator, block);
        try out.appendSlice(allocator, "\n\n");
    }
    return out.toOwnedSlice(allocator);
}

test "a queued stream gets SSE comments while it waits and its events are the same bytes as when not queued" {
    const t = std.testing;
    const events = [_][]const u8{
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n",
        "data: [DONE]\n\n",
    };
    const wait = slot_admission.QueueWait{ .ahead = 2, .running_prompt = 3000, .running_generated = 8012, .running_max = 32768, .tps = 0 };
    var outs: [2][]u8 = undefined;
    var raws: [2][]u8 = undefined;
    for ([_]bool{ true, false }, 0..) |queued, i| {
        var sv: [2]std.posix.fd_t = undefined;
        try t.expect(std.c.socketpair(1, 1, 0, &sv) == 0);
        defer _ = std.c.close(sv[1]);
        var conn: Conn = undefined;
        Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = undefined } }, t.io);
        for (events) |ev| {
            if (queued) try sendStreamIdle(&conn, wait);
            try conn.writeAll(ev);
        }
        try sendStreamIdle(&conn, null);
        _ = std.c.close(sv[0]);
        var buf: [4096]u8 = undefined;
        raws[i] = try t.allocator.dupe(u8, readAllFromPair(sv[1], &buf));
        outs[i] = try sseEventsOnly(t.allocator, raws[i]);
    }
    defer for (outs) |o| t.allocator.free(o);
    defer for (raws) |r| t.allocator.free(r);
    const note = ": queued behind 2 request(s); running 8012 of 32768 tokens\n\n";
    try t.expect(std.mem.startsWith(u8, raws[0], note));
    try t.expectEqual(@as(usize, 3), std.mem.count(u8, raws[0], note));
    try t.expectEqual(@as(usize, 0), std.mem.count(u8, raws[1], "queued"));
    try t.expect(std.mem.endsWith(u8, raws[1], ": keepalive\n\n"));
    try t.expectEqualStrings(outs[1], outs[0]);
}

test "count_tokens: routed like /v1/messages, with Anthropic-shaped refusals, behind the key" {
    const t = std.testing;
    try t.expect(routeExists("/v1/messages/count_tokens"));
    try t.expect(isTextGenRoute("POST", "/v1/messages/count_tokens"));
    try t.expectEqual(Api.anthropic, apiOfPath("/v1/messages/count_tokens"));

    const prev_key = server_state.g_api_key;
    const prev_strict = server_state.g_api_key_strict;
    defer {
        server_state.g_api_key = prev_key;
        server_state.g_api_key_strict = prev_strict;
    }
    server_state.g_api_key = "s3cret";
    server_state.g_api_key_strict = true;
    const keyless = try serveOneForTest("POST /v1/messages/count_tokens?beta=true HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}");
    defer t.allocator.free(keyless);
    try t.expect(std.mem.startsWith(u8, keyless, "HTTP/1.1 401 "));
}

test "the debug log redacts api_key and key query values and keeps the rest" {
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("/metrics", redactQueryKeys(&buf, "/metrics"));
    try std.testing.expectEqualStrings("/metrics?api_key=***", redactQueryKeys(&buf, "/metrics?api_key=s3cret"));
    try std.testing.expectEqualStrings("/m?a=1&key=***&b=2", redactQueryKeys(&buf, "/m?a=1&key=abc&b=2"));
    try std.testing.expectEqualStrings("/m?api_key=***&key=***", redactQueryKeys(&buf, "/m?api_key=x&key=y"));
    try std.testing.expectEqualStrings("/m?apikey=1&flag", redactQueryKeys(&buf, "/m?apikey=1&flag"));
    try std.testing.expectEqualStrings("/m?", redactQueryKeys(&buf, "/m?" ++ "a=1&a=1&a=1&a=1&a=1&a=1&a=1&a=1&a=1&a=1&a=1&a=1&a=1&a=1&a=1&a=1&a=1&" ++ "key=zzzz"));
}

test "chunked decoder: any split of the bytes decodes the same body" {
    const t = std.testing;
    const wire = "5\r\nhello\r\n6;ext=1\r\n world\r\nA\r\n0123456789\r\n0\r\nX-Trailer: 1\r\n\r\n";
    var step: usize = 1;
    while (step <= wire.len) : (step += 1) {
        var d: ChunkDecoder = .{};
        var budget = ByteBudget.init(1024);
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(t.allocator);
        var at: usize = 0;
        while (at < wire.len) : (at += step) _ = try d.feed(t.allocator, &out, wire[at..@min(wire.len, at + step)], &budget);
        try t.expectEqualStrings("hello world0123456789", out.items);
        try t.expect(d.state == .done);
    }
}

test "chunked decoder: a bad size, an over-cap body and a bare line are refused" {
    const t = std.testing;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(t.allocator);
    var roomy = ByteBudget.init(1024);
    var bad: ChunkDecoder = .{};
    try t.expectError(error.BadChunk, bad.feed(t.allocator, &out, "zz\r\nab\r\n", &roomy));
    var small = ByteBudget.init(8);
    var big: ChunkDecoder = .{};
    try t.expectError(error.TooLarge, big.feed(t.allocator, &out, "20\r\n" ++ "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" ++ "\r\n", &small));
    var stray: ChunkDecoder = .{};
    try t.expectError(error.BadChunk, stray.feed(t.allocator, &out, "1\r\nab\r\n", &roomy));
}

test "request framing: a chunked body is decoded and routed like its Content-Length twin" {
    const t = std.testing;
    const chunked = "POST /v1/downloads HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n" ++
        "6\r\n{\"repo\r\n8\r\n\":\"other\r\n7\r\n/Pack\"}\r\n0\r\n\r\n";
    const response = try serveOneForTest(chunked);
    defer t.allocator.free(response);
    try t.expect(std.mem.startsWith(u8, response, "HTTP/1.1 400 "));
    try t.expect(std.mem.indexOf(u8, responseBody(response), "org beamster only") != null);

    // A coding the server does not implement before `chunked` is a 501; one that leaves the body unframed a 400.
    const gzip = try serveOneForTest("POST /v1/downloads HTTP/1.1\r\nTransfer-Encoding: gzip, chunked\r\n\r\n");
    defer t.allocator.free(gzip);
    try t.expect(std.mem.startsWith(u8, gzip, "HTTP/1.1 501 "));
    const bare = try serveOneForTest("POST /v1/downloads HTTP/1.1\r\nTransfer-Encoding: gzip\r\n\r\n");
    defer t.allocator.free(bare);
    try t.expect(std.mem.startsWith(u8, bare, "HTTP/1.1 400 "));
    const garbled = try serveOneForTest("POST /v1/downloads HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\nnot-hex\r\n");
    defer t.allocator.free(garbled);
    try t.expect(std.mem.startsWith(u8, garbled, "HTTP/1.1 400 "));
}

test "request framing: Expect 100-continue is answered before the body is read" {
    const t = std.testing;
    const body = "{\"repo\":\"other/Pack\"}";
    const sized = try serveSplitForTest("POST /v1/downloads HTTP/1.1\r\nExpect: 100-continue\r\nContent-Length: 21\r\n\r\n", body);
    defer t.allocator.free(sized);
    try t.expect(std.mem.startsWith(u8, sized, "HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 400 "));
    try t.expect(std.mem.indexOf(u8, responseBody(sized["HTTP/1.1 100 Continue\r\n\r\n".len..]), "org beamster only") != null);

    const chunked = try serveSplitForTest("POST /v1/downloads HTTP/1.1\r\nexpect: 100-Continue\r\nTransfer-Encoding: chunked\r\n\r\n", "15\r\n" ++ body ++ "\r\n0\r\n\r\n");
    defer t.allocator.free(chunked);
    try t.expect(std.mem.startsWith(u8, chunked, "HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 400 "));

    const other = try serveSplitForTest("POST /v1/downloads HTTP/1.1\r\nExpect: 200-ok\r\nContent-Length: 21\r\n\r\n", body);
    defer t.allocator.free(other);
    try t.expect(std.mem.startsWith(u8, other, "HTTP/1.1 417 "));
}

test "every response says Connection: close, because the server closes after one request" {
    const t = std.testing;
    for ([_][]const u8{
        "GET /health HTTP/1.1\r\n\r\n",
        "GET /v1/models HTTP/1.1\r\n\r\n",
        "GET /nope HTTP/1.1\r\n\r\n",
        "POST /v1/downloads HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}",
        "OPTIONS /v1/models HTTP/1.1\r\n\r\n",
        "POST /v1/downloads HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\n",
    }) |request| {
        const response = try serveOneForTest(request);
        defer t.allocator.free(response);
        const head = response[0 .. std.mem.indexOf(u8, response, "\r\n\r\n") orelse response.len];
        try t.expect(std.mem.indexOf(u8, head, "\r\nConnection: close") != null);
    }
}

test "a generation refusal reaches the client whole: the longest of each kind, in each dialect, streamed or not, and over the WebSocket" {
    const t = std.testing;
    const a = t.allocator;
    const forward_refusal = @import("forward_refusal.zig");
    const wide: u64 = std.math.maxInt(u64) >> 1;
    const most: u32 = std.math.maxInt(u32);
    const fits = model_registry_mod.MemoryContextRefusal{ .requested = wide, .maximum = most, .needed = wide, .available = wide, .kv_bits = wide, .chunk = wide, .asked_max_tokens = most, .fitting_max_tokens = most };
    var alone = fits;
    alone.fitting_max_tokens = 0;
    // A stored text at its store's whole capacity, with a quote and a line break the wire escapes.
    var forward: [@typeInfo(@FieldType(forward_refusal.Text, "buf")).array.len]u8 = @splat('f');
    forward[0] = '"';
    forward[1] = '\n';
    var media: @import("vision_queue.zig").MediaRefusal = .{ .server_error = false };
    @memset(&media.buf, 'm');
    media.len = media.buf.len;
    const Kind = struct { err: anyerror, context: ?model_registry_mod.MemoryContextRefusal = null, forward: bool = false, media: bool = false };
    var failures: usize = 0;
    var seq: u64 = 0;
    for ([_]Kind{ .{ .err = error.PrefillDoesNotFit, .context = fits }, .{ .err = error.PrefillDoesNotFit, .context = alone }, .{ .err = error.PrefillChunkDoesNotFit, .forward = true }, .{ .err = error.MediaRefused, .media = true } }) |kind| {
        const want = if (kind.context) |c| try server_mod.memoryContextRefusalMessage(a, c) else try a.dupe(u8, if (kind.forward) &forward else media.text());
        defer a.free(want);
        const esc = try http.jsonEscape(a, want);
        defer a.free(esc);
        for ([_]?generation_stream.ErrorSurface{ .openai, .anthropic, .{ .responses = &seq }, null }) |surface| for ([_]bool{ false, true }) |streamed| {
            if (surface == null and streamed) continue;
            var sv: [2]std.posix.fd_t = undefined;
            try t.expect(std.c.socketpair(1, 1, 0, &sv) == 0);
            defer _ = std.c.close(sv[1]);
            var conn: Conn = undefined;
            Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = undefined } }, t.io);
            conn.sse_headers_sent = streamed;
            server_state.slot_memory_context_refusal = kind.context;
            server_state.slot_forward_refusal = if (kind.forward) blk: {
                forward_refusal.record(&forward);
                break :blk forward_refusal.take();
            } else null;
            server_state.slot_media_refusal = if (kind.media) media else null;
            if (surface) |s| {
                try generation_stream.sendGenerationError(a, &conn, kind.err, s);
            } else {
                var limit = std.atomic.Value(usize).init(1 << 20);
                var ws_conn = WsConnT.init(&conn, &limit);
                try wsGenerationErrorTurn(a, &ws_conn, kind.err);
            }
            _ = std.c.close(sv[0]);
            var buf: [16384]u8 = undefined;
            const raw = server_mod.readAllFromPair(sv[1], &buf);
            if (std.mem.indexOf(u8, raw, esc) == null) {
                failures += 1;
                std.debug.print("{s} on {s}{s}: the wire lacks the refusal's {d}-byte text\n", .{ @errorName(kind.err), if (surface) |s| @tagName(s) else "websocket", if (streamed) " (streamed)" else "", want.len });
            }
        };
    }
    try t.expectEqual(@as(usize, 0), failures);
}

/// Whether `response` carries Anthropic's error envelope.
fn anthropicEnvelope(response: []const u8) bool {
    return std.mem.startsWith(u8, responseBody(response), "{\"type\":\"error\",\"error\":{\"type\":\"");
}

test "an error on an Anthropic route takes Anthropic's envelope, sent by the request reader or before any registry" {
    const t = std.testing;
    const previous = server_state.max_request_bytes.swap(4096, .monotonic);
    defer server_state.max_request_bytes.store(previous, .monotonic);
    var bad: usize = 0;
    for ([_][2][]const u8{
        .{ "413 oversize", "POST /v1/messages HTTP/1.1\r\nContent-Length: 999999\r\n\r\n{" },
        .{ "501 coding", "POST /v1/messages HTTP/1.1\r\nTransfer-Encoding: gzip\r\n\r\n" },
        .{ "417 expect", "POST /v1/messages HTTP/1.1\r\nContent-Length: 10\r\nExpect: 200-ok\r\n\r\n" },
        .{ "400 chunk", "POST /v1/messages HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\n" },
        .{ "413 count_tokens", "POST /v1/messages/count_tokens?beta=true HTTP/1.1\r\nContent-Length: 999999\r\n\r\n{" },
    }) |c| {
        const response = try serveOneForTest(c[1]);
        defer t.allocator.free(response);
        if (!anthropicEnvelope(response)) {
            bad += 1;
            std.debug.print("{s}: {s}\n", .{ c[0], response[0..@min(response.len, 400)] });
        }
    }
    const no_registry = try serveWithRegistryForTest(null, "POST /v1/messages HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}", "");
    defer t.allocator.free(no_registry);
    if (!std.mem.startsWith(u8, no_registry, "HTTP/1.1 503 ") or !anthropicEnvelope(no_registry)) {
        bad += 1;
        std.debug.print("no registry: {s}\n", .{no_registry[0..@min(no_registry.len, 400)]});
    }
    try t.expectEqual(@as(usize, 0), bad);
}

test "every path under /v1/messages takes Anthropic's envelope, its 401 and its other errors alike" {
    const t = std.testing;
    const prev_key = server_state.g_api_key;
    const prev_strict = server_state.g_api_key_strict;
    defer {
        server_state.g_api_key = prev_key;
        server_state.g_api_key_strict = prev_strict;
    }
    var bad: usize = 0;
    for ([_]?[]const u8{ null, "s3cret" }) |key| {
        server_state.g_api_key = key;
        server_state.g_api_key_strict = key != null;
        for ([_][]const u8{
            "POST /v1/messages/batches HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}",
            "GET /v1/messages/count_tokens HTTP/1.1\r\n\r\n",
            "POST /v1/messages/count_tokens HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}",
            "POST /v1/messages HTTP/1.1\r\nContent-Length: 8\r\n\r\nnot json",
        }) |request| {
            const response = try serveOneForTest(request);
            defer t.allocator.free(response);
            if (!anthropicEnvelope(response)) {
                bad += 1;
                std.debug.print("key={any}: {s}\n", .{ key != null, response[0..@min(response.len, 400)] });
            }
        }
    }
    try t.expectEqual(@as(usize, 0), bad);
}

test "the 401 takes the error envelope every other error takes, and a HEAD's 401 carries no body" {
    const t = std.testing;
    const prev_key = server_state.g_api_key;
    const prev_strict = server_state.g_api_key_strict;
    defer {
        server_state.g_api_key = prev_key;
        server_state.g_api_key_strict = prev_strict;
    }
    server_state.g_api_key = "s3cret";
    server_state.g_api_key_strict = true;
    const chat = try serveOneForTest("POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}");
    defer t.allocator.free(chat);
    try t.expectEqualStrings("{\"error\":{\"message\":\"missing or invalid API key\",\"type\":\"authentication_error\",\"param\":null,\"code\":401}}", responseBody(chat));
    try t.expect(std.mem.indexOf(u8, chat, "WWW-Authenticate: Basic realm=\"sushi\"\r\n") != null);
    const head = try serveOneForTest("HEAD /v1/models HTTP/1.1\r\n\r\n");
    defer t.allocator.free(head);
    try t.expect(std.mem.startsWith(u8, head, "HTTP/1.1 401 "));
    try t.expectEqualStrings("", responseBody(head));
}

test "the metrics feed's refusal carries the error envelope the page reads error.message from" {
    const t = std.testing;
    const was = instr.process;
    defer instr.process = was;
    instr.process = .{};
    const response = try serveOneForTest("GET /metrics.json HTTP/1.1\r\n\r\n");
    defer t.allocator.free(response);
    try t.expect(std.mem.startsWith(u8, response, "HTTP/1.1 503 "));
    const E = struct { @"error": struct { message: []const u8 } };
    const parsed = try std.json.parseFromSlice(E, t.allocator, responseBody(response), .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    try t.expect(std.mem.indexOf(u8, parsed.value.@"error".message, "--metrics") != null);
}

/// A client for the read tests: sends `sent` whole (a wide socket, so the first read can take past the head), then
/// drips, floods or stalls; the shutdown flag is set after `shutdown_after_ms`.
const ReadOpts = struct {
    sent: []const u8,
    /// Sent one byte per `drip_ms` after `sent`.
    drip: []const u8 = "",
    drip_ms: u64 = 0,
    limit_ms: i64,
    shutdown_after_ms: ?u64 = null,
    /// Sent forever after `sent` (and `drip`), as fast as the socket takes it.
    flood: ?[]const u8 = null,
    wait_ms: u64 = 2000,
    /// The WebSocket message cap `wsReadAgainst` reads with.
    ws_limit: usize = 1 << 20,
    /// The shutdown flag is set before the read starts, with every byte of `sent` already waiting.
    shutdown_first: bool = false,
    /// The client closes its side after `sent` and `drip`.
    half_close: bool = false,
    /// `WsConnT.frame_limit_ms` for `wsReadAgainst`.
    ws_frame_limit_ms: ?i64 = null,
};
const ReadResult = struct { returned: bool, ms: i64, outcome: ?std.meta.Tag(ReadOutcome), len: ?usize, reply: []u8, body: []u8 };

const FeedClient = struct {
    fd: std.posix.fd_t,
    o: ReadOpts,
    stop: *std.atomic.Value(bool),
    fn sendAllNb(fd: std.posix.fd_t, bytes: []const u8, stop: *std.atomic.Value(bool)) bool {
        var off: usize = 0;
        while (off < bytes.len) {
            if (stop.load(.acquire)) return false;
            const n = std.c.send(fd, bytes[off..].ptr, bytes.len - off, std.posix.MSG.DONTWAIT);
            if (n > 0) {
                off += @intCast(n);
                continue;
            }
            if (n < 0 and std.posix.errno(n) == .AGAIN) {
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .real) catch {};
                continue;
            }
            return false;
        }
        return true;
    }
    fn run(c: *FeedClient) void {
        if (!c.o.shutdown_first and !sendAllNb(c.fd, c.o.sent, c.stop)) return;
        for (c.o.drip) |b| {
            std.Io.sleep(std.testing.io, .fromMilliseconds(@intCast(c.o.drip_ms)), .real) catch {};
            if (!sendAllNb(c.fd, &.{b}, c.stop)) return;
        }
        if (c.o.half_close) _ = std.c.shutdown(c.fd, std.posix.SHUT.WR);
        if (c.o.flood) |f| {
            var block: [64 * 1024]u8 = undefined;
            var i: usize = 0;
            while (i + f.len <= block.len) : (i += f.len) @memcpy(block[i..][0..f.len], f);
            while (!c.stop.load(.acquire)) if (!sendAllNb(c.fd, block[0..i], c.stop)) return;
        }
    }
};

fn widePair() ![2]std.posix.fd_t {
    var sv: [2]std.posix.fd_t = undefined;
    if (std.c.socketpair(1, 1, 0, &sv) != 0) return error.SocketPairFailed;
    const wide: c_int = 256 * 1024;
    for (sv) |fd| {
        _ = std.c.setsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.RCVBUF, std.mem.asBytes(&wide), @sizeOf(c_int));
        _ = std.c.setsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.SNDBUF, std.mem.asBytes(&wide), @sizeOf(c_int));
    }
    // The client's send gives up after 20 ms when the server stops reading (MSG_DONTWAIT alone can still block here).
    const send_limit: std.posix.timeval = .{ .sec = 0, .usec = 20_000 };
    _ = std.c.setsockopt(sv[1], std.posix.SOL.SOCKET, std.posix.SO.SNDTIMEO, std.mem.asBytes(&send_limit), @sizeOf(std.posix.timeval));
    return sv;
}

fn drainReply(a: std.mem.Allocator, fd: std.posix.fd_t) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var buf: [4096]u8 = undefined;
    while (true) {
        const n = std.c.recv(fd, &buf, buf.len, std.posix.MSG.DONTWAIT);
        if (n <= 0) break;
        try out.appendSlice(a, buf[0..@intCast(n)]);
    }
    return out.toOwnedSlice(a);
}

/// `readRequestWithin(limit_ms)` against a client that sends `o.sent` whole (a wide socket, so the first read can take
/// past the head), then drips, floods or stalls; the shutdown flag set after `shutdown_after_ms`.
fn readAgainst(o: ReadOpts) !ReadResult {
    const t = std.testing;
    const sv = try widePair();
    var conn: Conn = undefined;
    Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = undefined } }, t.io);
    var stop = std.atomic.Value(bool).init(false);
    var client: FeedClient = .{ .fd = sv[1], .o = o, .stop = &stop };
    const State = struct {
        done: std.atomic.Value(bool) = .init(false),
        outcome: ?ReadOutcome = null,
        len: ?usize = null,
        body: []u8 = &.{},
        ms: i64 = 0,
    };
    var st: State = .{};
    if (o.shutdown_first) {
        _ = std.c.send(sv[1], o.sent.ptr, o.sent.len, 0);
        server_state.shutdown_requested.store(true, .release);
    }
    const t0 = nowMsMonotonic(t.io);
    const reader = try std.Thread.spawn(.{}, struct {
        fn run(c: *Conn, limit: i64, s: *State, start: i64) void {
            if (readRequestWithin(t.allocator, c, limit) catch null) |got| {
                s.outcome = got;
                if (got == .request) {
                    s.len = got.request.end;
                    s.body = t.allocator.dupe(u8, got.request.buf[got.request.header_end..got.request.end]) catch &.{};
                    t.allocator.free(got.request.buf);
                }
            }
            s.ms = nowMsMonotonic(t.io) - start;
            s.done.store(true, .release);
        }
    }.run, .{ &conn, o.limit_ms, &st, t0 });
    const sender = try std.Thread.spawn(.{}, FeedClient.run, .{&client});
    var shut_at: i64 = 0;
    var waited: u64 = 0;
    while (!st.done.load(.acquire) and waited < o.wait_ms) : (waited += 5) {
        std.Io.sleep(t.io, .fromMilliseconds(5), .real) catch {};
        if (o.shutdown_after_ms) |s| if (waited >= s and shut_at == 0) {
            server_state.shutdown_requested.store(true, .release);
            shut_at = nowMsMonotonic(t.io) - t0;
        };
    }
    const returned = st.done.load(.acquire);
    stop.store(true, .release);
    sender.join();
    const reply = try drainReply(t.allocator, sv[1]);
    _ = std.c.close(sv[1]);
    reader.join();
    server_state.shutdown_requested.store(false, .release);
    conn.close();
    return .{ .returned = returned, .ms = if (returned) st.ms - shut_at else -1, .outcome = if (st.outcome) |x| std.meta.activeTag(x) else null, .len = st.len, .reply = reply, .body = st.body };
}

fn freeRead(r: ReadResult) void {
    std.testing.allocator.free(r.reply);
    if (r.body.len > 0) std.testing.allocator.free(r.body);
}


/// `WsConnT.readMessage` in a loop (pings and pongs are messages) over a `Conn`, against a `ReadOpts` client.
const WsReadResult = struct { returned: bool, ms: i64, err: ?anyerror, messages: usize };

/// `readMessage` in a loop (pings and pongs are messages) until it fails, against a client that sends `o.sent` whole
/// and then stalls, drips or floods; the shutdown flag set after `shutdown_after_ms`.
fn wsReadAgainst(o: ReadOpts) !WsReadResult {
    const t = std.testing;
    const sv = try widePair();
    var conn: Conn = undefined;
    Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = undefined } }, t.io);
    var stop = std.atomic.Value(bool).init(false);
    var client: FeedClient = .{ .fd = sv[1], .o = o, .stop = &stop };
    const State = struct { done: std.atomic.Value(bool) = .init(false), err: ?anyerror = null, ms: i64 = 0, messages: usize = 0 };
    var st: State = .{};
    var limit = std.atomic.Value(usize).init(o.ws_limit);
    if (o.shutdown_first) {
        _ = std.c.send(sv[1], o.sent.ptr, o.sent.len, 0);
        server_state.shutdown_requested.store(true, .release);
    }
    const t0 = nowMsMonotonic(t.io);
    const reader = try std.Thread.spawn(.{}, struct {
        fn run(c: *Conn, lim: *std.atomic.Value(usize), frame_limit: ?i64, s: *State, start: i64) void {
            var ws = WsConnT.init(c, lim);
            ws.frame_limit_ms = frame_limit;
            defer ws.deinit(t.allocator);
            while (true) {
                const m = ws.readMessage(t.allocator) catch |e| {
                    s.err = e;
                    break;
                };
                t.allocator.free(m.payload);
                s.messages += 1;
            }
            s.ms = nowMsMonotonic(t.io) - start;
            s.done.store(true, .release);
        }
    }.run, .{ &conn, &limit, o.ws_frame_limit_ms, &st, t0 });
    const sender = try std.Thread.spawn(.{}, FeedClient.run, .{&client});
    var shut_at: i64 = 0;
    var waited: u64 = 0;
    while (!st.done.load(.acquire) and waited < o.wait_ms) : (waited += 5) {
        std.Io.sleep(t.io, .fromMilliseconds(5), .real) catch {};
        if (o.shutdown_after_ms) |s| if (waited >= s and shut_at == 0) {
            server_state.shutdown_requested.store(true, .release);
            shut_at = nowMsMonotonic(t.io) - t0;
        };
    }
    const returned = st.done.load(.acquire);
    stop.store(true, .release);
    sender.join();
    _ = std.c.close(sv[1]);
    reader.join();
    server_state.shutdown_requested.store(false, .release);
    conn.close();
    return .{ .returned = returned, .ms = if (returned) st.ms - shut_at else -1, .err = st.err, .messages = st.messages };
}

/// A masked client frame with a zero mask.
fn clientFrame(comptime b0: u8, comptime payload: []const u8) []const u8 {
    if (payload.len < 126) return &[_]u8{ b0, 0x80 | @as(u8, payload.len), 0, 0, 0, 0 } ++ payload;
    return &[_]u8{ b0, 0x80 | 126, @as(u8, payload.len >> 8), @as(u8, payload.len & 0xff), 0, 0, 0, 0 } ++ payload;
}


const chunked_head = "POST /v1/completions HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n";

test "read: a shutdown answers a request still arriving with the shutdown error, never reads it on" {
    // The whole request is waiting: a reader that takes waiting bytes before the flag would read it to its end.
    const r = try readAgainst(.{ .sent = chunked_head ++ "4e20\r\n" ++ @as([20_000]u8, @splat('a')) ++ "\r\n0\r\n\r\n", .limit_ms = 30_000, .shutdown_first = true });
    defer freeRead(r);
    if (!r.returned or r.len != null or !std.mem.startsWith(u8, r.reply, "HTTP/1.1 503 ") or std.mem.indexOf(u8, r.reply, http.shutting_down_type) == null) {
        std.debug.print("a request waiting at a shutdown: returned={any} len={?d} reply '{s}'\n", .{ r.returned, r.len, r.reply[0..@min(r.reply.len, 120)] });
        return error.ReadOnPastShutdown;
    }
}

test "read: a chunked body of carriage returns alone stops at the request cap" {
    const previous = server_state.max_request_bytes.swap(4096, .monotonic);
    defer server_state.max_request_bytes.store(previous, .monotonic);
    const r = try readAgainst(.{ .sent = chunked_head, .flood = "\r", .limit_ms = 30_000, .wait_ms = 3000 });
    defer freeRead(r);
    if (!r.returned or r.len != null or !(std.mem.startsWith(u8, r.reply, "HTTP/1.1 400 ") or std.mem.startsWith(u8, r.reply, "HTTP/1.1 413 "))) {
        std.debug.print("a CR flood under a 4 KB cap: returned={any} reply '{s}'\n", .{ r.returned, r.reply[0..@min(r.reply.len, 60)] });
        return error.UncappedRead;
    }
}

test "ws: a shutdown ends a fragmented message whose empty continuation frames are already waiting" {
    const a = std.testing.allocator;
    var frames: std.ArrayList(u8) = .empty;
    defer frames.deinit(a);
    try frames.appendSlice(a, comptime clientFrame(0x01, "{"));
    for (0..1000) |_| try frames.appendSlice(a, comptime clientFrame(0x00, ""));
    try frames.appendSlice(a, comptime clientFrame(0x80, "}"));
    const r = try wsReadAgainst(.{ .sent = frames.items, .limit_ms = 0, .shutdown_first = true });
    if (!r.returned or r.messages != 0 or r.err == null or r.err.? != error.Shutdown) {
        std.debug.print("a fragmented message waiting at a shutdown: returned={any} messages={d} err={s}\n", .{ r.returned, r.messages, if (r.err) |e| @errorName(e) else "none" });
        return error.ReadOnPastShutdown;
    }
}

test "ws: empty continuation frames end at the framing allowance" {
    const r = try wsReadAgainst(.{ .sent = comptime clientFrame(0x01, "{"), .flood = comptime clientFrame(0x00, ""), .limit_ms = 0, .wait_ms = 3000, .ws_limit = 4096 });
    if (!r.returned or r.err == null or r.err.? != error.WsFraming) {
        std.debug.print("an empty-continuation flood under a 4 KB cap: returned={any} err={s}\n", .{ r.returned, if (r.err) |e| @errorName(e) else "none" });
        return error.UncappedMessage;
    }
}

test "Conn.read: a shutdown ends a read with bytes waiting, held by the reader or in the socket" {
    const t = std.testing;
    const sv = try widePair();
    var conn: Conn = undefined;
    Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = undefined } }, t.io);
    // The peer goes first, so the close does not wait for it.
    defer conn.close();
    defer _ = std.c.close(sv[1]);
    const sent: [64]u8 = @splat('x');
    _ = std.c.send(sv[1], &sent, sent.len, 0);
    // The first read takes 8 bytes and leaves the rest held by the reader.
    var buf: [8]u8 = undefined;
    try t.expectEqual(@as(usize, 8), try conn.read(&buf, null));
    try t.expect(conn.reader().bufferedLen() > 0);
    server_state.shutdown_requested.store(true, .release);
    defer server_state.shutdown_requested.store(false, .release);
    try t.expectError(error.Shutdown, conn.read(&buf, null));
    // Bytes in the socket and none held.
    conn.reader().tossBuffered();
    _ = std.c.send(sv[1], &sent, sent.len, 0);
    try t.expectError(error.Shutdown, conn.read(&buf, 1000));
}

test "read: a head over 16 KB is answered 431, never closed bare" {
    const r = try readAgainst(.{ .sent = "GET /health HTTP/1.1\r\nX-Pad: " ++ @as([17_000]u8, @splat('p')) ++ "\r\n\r\n", .limit_ms = 1000 });
    defer freeRead(r);
    if (r.outcome != .head_too_large or !std.mem.startsWith(u8, r.reply, "HTTP/1.1 431 Request Header Fields Too Large\r\n") or std.mem.indexOf(u8, r.reply, "16 KB") == null) {
        std.debug.print("a 17 KB head: outcome={?} {d}-byte answer '{s}'\n", .{ r.outcome, r.reply.len, r.reply[0..@min(r.reply.len, 80)] });
        return error.HeadOverLimitUnanswered;
    }
}

test "read: a body cut short by the client's EOF is dropped, by length or chunked, never routed truncated" {
    var bad: usize = 0;
    for ([_][]const u8{
        "POST /v1/load-model HTTP/1.1\r\nContent-Length: 100\r\n\r\n{\"model\":\"tiny\"}",
        chunked_head ++ "40\r\n{\"model\":\"tiny\"}",
    }) |sent| {
        const r = try readAgainst(.{ .sent = sent, .limit_ms = 2000, .half_close = true });
        defer freeRead(r);
        if (!r.returned or r.outcome != .cut_short) {
            std.debug.print("EOF mid-body ({s}): returned={any} outcome={?} body='{s}'\n", .{ sent[0..20], r.returned, r.outcome, r.body });
            bad += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), bad);
}

test "read: a Content-Length the reader does not take whole (no space, a tab, junk, two values) is never routed with its body dropped" {
    var bad: usize = 0;
    for ([_][]const u8{
        "POST /v1/load-model HTTP/1.1\r\nContent-Length:16\r\n\r\n{\"model\":\"tiny\"}",
        "POST /v1/load-model HTTP/1.1\r\nContent-Length:\t16\r\n\r\n{\"model\":\"tiny\"}",
        "POST /v1/load-model HTTP/1.1\r\nContent-Length: 16\t\r\n\r\n{\"model\":\"tiny\"}",
        "POST /v1/load-model HTTP/1.1\r\nContent-Length: 1_6\r\n\r\n{\"model\":\"tiny\"}",
        "POST /v1/load-model HTTP/1.1\r\nContent-Length: 0\r\nContent-Length: 16\r\n\r\n{\"model\":\"tiny\"}",
    }) |sent| {
        const r = try readAgainst(.{ .sent = sent, .limit_ms = 500 });
        defer freeRead(r);
        // The contract: routed with its whole body, or refused (400). Never routed with the body left behind.
        const whole = r.outcome == .request and std.mem.eql(u8, r.body, "{\"model\":\"tiny\"}");
        const refused = r.outcome == .bad_framing and std.mem.startsWith(u8, r.reply, "HTTP/1.1 400 ");
        if (!whole and !refused) {
            const hl = std.mem.indexOf(u8, sent, "Content-Length") orelse 0;
            std.debug.print("'{s}': outcome={s} routed body='{s}' ({d} B)\n", .{ sent[hl .. std.mem.indexOf(u8, sent, "\r\n\r\n") orelse sent.len], if (r.outcome) |x| @tagName(x) else "none", r.body, r.body.len });
            bad += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), bad);
}

/// A request head of exactly `total` bytes: `line`, one pad header, the blank line.
fn headOfSize(a: std.mem.Allocator, line: []const u8, extra: []const u8, total: usize) ![]u8 {
    const fixed = line.len + "X-Pad: ".len + "\r\n".len + extra.len + "\r\n".len;
    const pad = total - fixed;
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, line);
    try out.appendSlice(a, "X-Pad: ");
    try out.appendNTimes(a, 'p', pad);
    try out.appendSlice(a, "\r\n");
    try out.appendSlice(a, extra);
    try out.appendSlice(a, "\r\n");
    std.debug.assert(out.items.len == total);
    return out.toOwnedSlice(a);
}

test "read: a HEAD refused by the reader (431, 413, 503 at a shutdown) carries no body" {
    const a = std.testing.allocator;
    var bad: usize = 0;
    const big = try headOfSize(a, "HEAD /health HTTP/1.1\r\n", "", 17_000);
    defer a.free(big);
    const previous = server_state.max_request_bytes.swap(4096, .monotonic);
    defer server_state.max_request_bytes.store(previous, .monotonic);
    for ([_]struct { what: []const u8, o: ReadOpts }{
        .{ .what = "431", .o = .{ .sent = big, .limit_ms = 1000 } },
        .{ .what = "413", .o = .{ .sent = "HEAD /v1/models HTTP/1.1\r\nContent-Length: 999999\r\n\r\n", .limit_ms = 1000 } },
        .{ .what = "503 shutdown", .o = .{ .sent = "HEAD /v1/models HTTP/1.1\r\nHost: x\r\n\r\n", .limit_ms = 1000, .shutdown_first = true } },
    }) |c| {
        const r = try readAgainst(c.o);
        defer freeRead(r);
        const body = responseBody(r.reply);
        if (!std.mem.startsWith(u8, r.reply, "HTTP/1.1 ") or body.len != 0) {
            std.debug.print("HEAD {s}: outcome={s} {d}-byte body: '{s}'\n", .{ c.what, if (r.outcome) |x| @tagName(x) else "none", body.len, body[0..@min(body.len, 120)] });
            bad += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), bad);
}

test "ws: a message of exactly the cap, its last fragment empty, is read, never refused as over it" {
    const a = std.testing.allocator;
    const pay: [4096]u8 = @splat('m');
    var bad: usize = 0;
    for ([_][]const []const u8{
        // 4096 bytes, then an empty final continuation.
        &.{ comptime clientFrame(0x01, &pay), comptime clientFrame(0x80, "") },
        // An empty first fragment, then 4096 bytes.
        &.{ comptime clientFrame(0x01, ""), comptime clientFrame(0x80, &pay) },
        // Control: 4096 bytes in two halves.
        &.{ comptime clientFrame(0x01, pay[0..2048]), comptime clientFrame(0x80, pay[2048..]) },
    }, 0..) |frames, i| {
        const sent = try std.mem.concat(a, u8, frames);
        defer a.free(sent);
        const r = try wsReadAgainst(.{ .sent = sent, .limit_ms = 0, .ws_limit = 4096, .half_close = true, .wait_ms = 2000 });
        if (r.messages != 1) {
            std.debug.print("ws exact-cap message {d}: messages={d} err={s}\n", .{ i, r.messages, if (r.err) |e| @errorName(e) else "none" });
            bad += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), bad);
}

test "read: chunk-size lines (extensions) count toward the request cap, so raw bytes stop near it" {
    const a = std.testing.allocator;
    const previous = server_state.max_request_bytes.swap(4096, .monotonic);
    defer server_state.max_request_bytes.store(previous, .monotonic);
    // Each chunk: an 80-byte size line (1 data byte and an extension), its byte, CRLF: 85 raw bytes per decoded byte.
    const chunk = "1;" ++ @as([78]u8, @splat('e')) ++ "\r\nA\r\n";
    var sent: std.ArrayList(u8) = .empty;
    defer sent.deinit(a);
    try sent.appendSlice(a, chunked_head);
    // 200 decoded bytes, far under the 4 KB cap, in 17 KB of raw bytes: 4x the cap.
    for (0..200) |_| try sent.appendSlice(a, chunk);
    try sent.appendSlice(a, "0\r\n\r\n");
    const r = try readAgainst(.{ .sent = sent.items, .limit_ms = 1000 });
    defer freeRead(r);
    if (r.outcome == .request) {
        std.debug.print("chunk extensions: {d} raw bytes under a 4096 B cap routed as a request of {d} B (body {d} B)\n", .{ sent.items.len, r.len orelse 0, r.body.len });
        return error.RawBytesUncapped;
    }
}

fn statusLineOf(reply: []const u8) []const u8 {
    return reply[0..@min(reply.len, std.mem.indexOf(u8, reply, "\r\n") orelse reply.len)];
}

test "read: a field name with whitespace before its colon, a folded line or a bare LF is a 400, never a body dropped" {
    // RFC 9112 §5.1: whitespace between a field name and its colon MUST be a 400. §5.2: an obs-fold MUST be a 400 or
    // read as SP ("1 6" is then no number, a 400). §2.2: a bare LF may end a line (then Content-Length is 16) or be
    // refused. The contract of the h15-1 test: routed with its whole body, or refused 400; never routed with it dropped.
    const body = "{\"model\":\"tiny\"}";
    var bad: usize = 0;
    for ([_][]const u8{
        "POST /v1/load-model HTTP/1.1\r\nContent-Length : 16\r\n\r\n" ++ body,
        "POST /v1/load-model HTTP/1.1\r\nContent-Length\t: 16\r\n\r\n" ++ body,
        "POST /v1/load-model HTTP/1.1\r\nContent-Length: 1\r\n 6\r\n\r\n" ++ body,
        "POST /v1/load-model HTTP/1.1\r\nContent-Length: 1\r\n\t6\r\n\r\n" ++ body,
        "POST /v1/load-model HTTP/1.1\r\nHost: x\nContent-Length: 16\r\n\r\n" ++ body,
        "POST /v1/load-model HTTP/1.1\r\nTransfer-Encoding : chunked\r\n\r\n10\r\n" ++ body ++ "\r\n0\r\n\r\n",
        "POST /v1/load-model HTTP/1.1\r\nHost: x\nTransfer-Encoding: chunked\r\n\r\n10\r\n" ++ body ++ "\r\n0\r\n\r\n",
    }) |sent| {
        const r = try readAgainst(.{ .sent = sent, .limit_ms = 500 });
        defer freeRead(r);
        const whole = r.outcome == .request and std.mem.eql(u8, r.body, body);
        const refused = std.mem.startsWith(u8, r.reply, "HTTP/1.1 400 ");
        if (!whole and !refused) {
            const he = std.mem.indexOf(u8, sent, "\r\n\r\n").?;
            std.debug.print("head '{f}': outcome={s} routed body='{s}' ({d} B) reply='{s}'\n", .{ std.zig.fmtString(sent[30..he]), if (r.outcome) |x| @tagName(x) else "none", r.body, r.body.len, statusLineOf(r.reply) });
            bad += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), bad);
}

test "read: a chunk size is hex digits alone: a sign, an underscore or a negative zero is a 400" {
    // RFC 9112 §7.1: chunk-size = 1*HEXDIG. The sibling rule http_head took for Content-Length ("1_6" and "+16" are a
    // 400, its own test): Zig's parseInt also reads '+', '_' and "-0".
    const body = "{\"model\":\"tiny\"}";
    var bad: usize = 0;
    for ([_][]const u8{
        "+10\r\n" ++ body ++ "\r\n0\r\n\r\n",
        "1_0\r\n" ++ body ++ "\r\n0\r\n\r\n",
        "10\r\n" ++ body ++ "\r\n-0\r\n\r\n",
        "10\r\n" ++ body ++ "\r\n+0\r\n\r\n",
        "0_10\r\n" ++ body ++ "\r\n0\r\n\r\n",
    }) |chunks| {
        const whole = try std.mem.concat(std.testing.allocator, u8, &.{ chunked_head, chunks });
        defer std.testing.allocator.free(whole);
        const r = try readAgainst(.{ .sent = whole, .limit_ms = 500 });
        defer freeRead(r);
        if (!std.mem.startsWith(u8, r.reply, "HTTP/1.1 400 ")) {
            std.debug.print("chunks '{f}': outcome={s} routed body='{s}' reply='{s}'\n", .{ std.zig.fmtString(chunks[0..@min(chunks.len, 8)]), if (r.outcome) |x| @tagName(x) else "none", r.body, statusLineOf(r.reply) });
            bad += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), bad);
}

test "read: a HEAD whose request line alone runs past 16 KB gets its 431 without a body" {
    // docs/server-http-apis.md: "a HEAD gets every refusal (431, 413, 501, 417, 400, the shutdown 503) without a body".
    const a = std.testing.allocator;
    const line = try std.mem.concat(a, u8, &.{ "HEAD /health?x=", &@as([17_000]u8, @splat('q')), " HTTP/1.1\r\n\r\n" });
    defer a.free(line);
    const r = try readAgainst(.{ .sent = line, .limit_ms = 1000 });
    defer freeRead(r);
    const body = responseBody(r.reply);
    if (!std.mem.startsWith(u8, r.reply, "HTTP/1.1 431 ") or body.len != 0) {
        std.debug.print("HEAD with a 17 KB target: outcome={s} reply='{s}' {d}-byte body: '{s}'\n", .{ if (r.outcome) |x| @tagName(x) else "none", statusLineOf(r.reply), body.len, body[0..@min(body.len, 120)] });
        return error.HeadBodySent;
    }
}

fn chunkedBodyOf(a: std.mem.Allocator, head: []const u8, payload: usize, chunk: usize) ![]u8 {
    var sent: std.ArrayList(u8) = .empty;
    errdefer sent.deinit(a);
    try sent.appendSlice(a, head);
    var left = payload;
    while (left > 0) {
        const n = @min(chunk, left);
        var size_buf: [20]u8 = undefined;
        try sent.appendSlice(a, try std.fmt.bufPrint(&size_buf, "{x}\r\n", .{n}));
        try sent.appendNTimes(a, 'b', n);
        try sent.appendSlice(a, "\r\n");
        left -= n;
    }
    try sent.appendSlice(a, "0\r\n\r\n");
    return sent.toOwnedSlice(a);
}

test "read: a chunked body of exactly the cap is read whatever its chunk size" {
    // docs/server-lifecycle.md: ByteBudget "charges decoded payload against the cap (a message of exactly the cap is
    // read)"; docs/server-http-apis.md: the cap includes "headers and the decoded chunked body". 1 MiB is the
    // smallest server.max_request_mb.
    const a = std.testing.allocator;
    const cap: usize = 1 << 20;
    const previous = server_state.max_request_bytes.swap(cap, .monotonic);
    defer server_state.max_request_bytes.store(previous, .monotonic);
    var bad: usize = 0;
    for ([_]usize{ 4096, 128, 64, 32, 16, 1 }) |chunk| {
        const sent = try chunkedBodyOf(a, chunked_head, cap - chunked_head.len, chunk);
        defer a.free(sent);
        const r = try readAgainst(.{ .sent = sent, .limit_ms = 2000, .wait_ms = 10_000 });
        defer freeRead(r);
        if (r.outcome != .request or r.body.len != cap - chunked_head.len) {
            std.debug.print("exact-cap chunked body in {d}-byte chunks ({d} raw B): outcome={s} reply='{s}'\n", .{ chunk, sent.len, if (r.outcome) |x| @tagName(x) else "none", statusLineOf(r.reply) });
            bad += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), bad);
}

test "read: a small chunked body in 1-byte chunks, far under the cap, is read" {
    // A 1 KiB body under a 4 KiB cap: the cap is on the decoded body (docs/server-http-apis.md).
    const a = std.testing.allocator;
    const previous = server_state.max_request_bytes.swap(4096, .monotonic);
    defer server_state.max_request_bytes.store(previous, .monotonic);
    var bad: usize = 0;
    for ([_]usize{ 1024, 2048 }) |n| {
        const sent = try chunkedBodyOf(a, chunked_head, n, 1);
        defer a.free(sent);
        const r = try readAgainst(.{ .sent = sent, .limit_ms = 1000 });
        defer freeRead(r);
        if (r.outcome != .request or r.body.len != n) {
            std.debug.print("{d}-byte body in 1-byte chunks under a 4096 B cap: outcome={s} reply='{s}'\n", .{ n, if (r.outcome) |x| @tagName(x) else "none", statusLineOf(r.reply) });
            bad += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), bad);
}

fn wsFragmentsOf(a: std.mem.Allocator, payload: usize, frag: usize) ![]u8 {
    var sent: std.ArrayList(u8) = .empty;
    errdefer sent.deinit(a);
    var left = payload;
    var first = true;
    while (left > 0) {
        const n = @min(frag, left);
        const fin: u8 = if (n == left) 0x80 else 0;
        const op: u8 = if (first) 0x01 else 0x00;
        first = false;
        if (n < 126) {
            try sent.appendSlice(a, &.{ fin | op, 0x80 | @as(u8, @intCast(n)), 0, 0, 0, 0 });
        } else {
            try sent.appendSlice(a, &.{ fin | op, 0x80 | 126, @as(u8, @intCast(n >> 8)), @as(u8, @intCast(n & 0xff)), 0, 0, 0, 0 });
        }
        try sent.appendNTimes(a, 'm', n);
        left -= n;
    }
    return sent.toOwnedSlice(a);
}

test "ws: a message of exactly the cap is read whatever its fragment size" {
    // docs/server-lifecycle.md: "a message of exactly the cap is read"; docs/server-http-apis.md: a WebSocket message
    // over the limit closes with 1009, so one at it must not.
    const a = std.testing.allocator;
    const cap: usize = 1 << 20;
    var bad: usize = 0;
    for ([_]usize{ 4096, 128, 64, 32, 1 }) |frag| {
        const sent = try wsFragmentsOf(a, cap, frag);
        defer a.free(sent);
        const r = try wsReadAgainst(.{ .sent = sent, .limit_ms = 0, .ws_limit = cap, .half_close = true, .wait_ms = 10_000 });
        if (r.messages != 1) {
            std.debug.print("ws exact-cap message in {d}-byte fragments ({d} raw B): messages={d} err={s}\n", .{ frag, sent.len, r.messages, if (r.err) |e| @errorName(e) else "none" });
            bad += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), bad);
}

test "serve: a request line the router cannot split, or one after a leading CRLF, is answered, never closed bare" {
    // RFC 9112 §2.2: a server SHOULD ignore at least one empty line before the request line; §3: an invalid
    // request line SHOULD be a 400. The brief's invariant: never closed bare when an answer is owed.
    var bad: usize = 0;
    for ([_][]const u8{
        "\r\nGET /health HTTP/1.1\r\n\r\n",
        "GET\r\n\r\n",
        "GET  /health HTTP/1.1\r\n\r\n",
    }) |req| {
        const reply = try serveOneForTest(req);
        defer std.testing.allocator.free(reply);
        if (!std.mem.startsWith(u8, reply, "HTTP/1.1 ")) {
            std.debug.print("'{f}': reply={d} B '{s}'\n", .{ std.zig.fmtString(req), reply.len, statusLineOf(reply) });
            bad += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), bad);
}

test "read: bytes sent past a request in the same write are kept for the connection, never dropped" {
    const frame = comptime clientFrame(0x89, "hi");
    for ([_][]const u8{
        "GET /v1/responses HTTP/1.1\r\nUpgrade: websocket\r\n\r\n",
        "POST /v1/completions HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}",
        chunked_head ++ "2\r\n{}\r\n0\r\n\r\n",
    }) |head| {
        const t = std.testing;
        const sv = try widePair();
        var conn: Conn = undefined;
        Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = undefined } }, t.io);
        defer conn.close();
        defer _ = std.c.close(sv[1]);
        const sent = try std.mem.concat(t.allocator, u8, &.{ head, frame });
        defer t.allocator.free(sent);
        _ = std.c.send(sv[1], sent.ptr, sent.len, 0);
        const got = try readRequestWithin(t.allocator, &conn, 1000);
        try t.expect(got == .request);
        defer t.allocator.free(got.request.buf);
        try t.expectEqualStrings(frame, got.request.buf[got.request.end..]);
    }
}

test "ws: a frame stalled mid-payload is dropped at the frame read limit" {
    const r = try wsReadAgainst(.{ .sent = comptime clientFrame(0x81, &@as([20_000]u8, @splat('w')))[0..18_000], .limit_ms = 0, .ws_frame_limit_ms = 200, .wait_ms = 3000 });
    if (!r.returned or r.err == null or r.err.? != error.ReadStalled) {
        std.debug.print("a half frame under a 200 ms frame limit: returned={any} err={s}\n", .{ r.returned, if (r.err) |e| @errorName(e) else "none" });
        return error.HalfFrameHeld;
    }
}

test "read: RFC 9112 framing statuses: chunked not last is a 400, Transfer-Encoding on HTTP/1.0 is faulty" {
    // RFC 9112 §6.3 (4): Transfer-Encoding whose final coding is not chunked: MUST 400 and close. §6.1: Transfer-Encoding
    // in an HTTP/1.0 request MUST be treated as faulty framing.
    var bad: usize = 0;
    for ([_][]const u8{
        "POST /v1/completions HTTP/1.1\r\nTransfer-Encoding: chunked, gzip\r\n\r\n2\r\n{}\r\n0\r\n\r\n",
        "POST /v1/completions HTTP/1.1\r\nTransfer-Encoding: gzip\r\n\r\n{}",
        "POST /v1/completions HTTP/1.0\r\nTransfer-Encoding: chunked\r\n\r\n2\r\n{}\r\n0\r\n\r\n",
    }) |sent| {
        const r = try readAgainst(.{ .sent = sent, .limit_ms = 500 });
        defer freeRead(r);
        if (!std.mem.startsWith(u8, r.reply, "HTTP/1.1 400 ")) {
            const he = std.mem.indexOf(u8, sent, "\r\n\r\n").?;
            std.debug.print("'{f}': outcome={s} reply='{s}'\n", .{ std.zig.fmtString(sent[0..he]), if (r.outcome) |x| @tagName(x) else "none", statusLineOf(r.reply) });
            bad += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), bad);
}
