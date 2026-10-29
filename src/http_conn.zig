const std = @import("std");
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
const writeEffortNoticeHeader = request_policy.writeEffortNoticeHeader;
const io_util = @import("io_util.zig");
const ws_mod = @import("ws.zig");
const nowMsMonotonic = io_util.nowMsMonotonic;
const testing = std.testing;
const server_mod = @import("server.zig");
const EMPTY_PROMPT_MSG = server_mod.EMPTY_PROMPT_MSG;
const SSE_ALLOW_HEADERS_DEFAULT = server_mod.SSE_ALLOW_HEADERS_DEFAULT;
const bindAddress = server_mod.bindAddress;
const chatPageUrl = server_mod.chatPageUrl;
const getOrInitResponseStore = server_mod.getOrInitResponseStore;
const handleCompletions = server_mod.handleCompletions;
const handleConnection = server_mod.handleConnection;
const handleResponses = server_mod.handleResponses;
const isAnthropicRoute = server_mod.isAnthropicRoute;
const jsonEscape = server_mod.jsonEscape;
const logHttpResponse = server_mod.logHttpResponse;
const mapGenerationError = server_mod.mapGenerationError;
const max_request_bytes = server_mod.max_request_bytes;
const readAllFromPair = server_mod.readAllFromPair;
const redactQueryKeys = server_mod.redactQueryKeys;
const responseBody = server_mod.responseBody;
const sendErrorResponse = server_mod.sendErrorResponse;
const sendResponseFramed = server_mod.sendResponseFramed;
const sendStreamIdle = server_mod.sendStreamIdle;

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

/// Bridge that lets a Conn route OpenAI-Responses output through a WebSocket
/// transport instead of HTTP/SSE. When `Conn.ws_mode` is set, `sendResponse`
/// and `sendAnthropicEvent` send the JSON payload as a single WS text frame
/// instead of an HTTP body or SSE event line.
pub const WsBridge = struct {
    impl: *anyopaque,
    sendTextFn: *const fn (impl: *anyopaque, data: []const u8) anyerror!void,
    /// Captured response id from the first event seen this turn (for moving
    /// the entry between global and connection-local caches afterwards).
    /// Owned by `allocator`; freed by the WS handler at end of turn.
    captured_resp_id: ?[]u8 = null,
    allocator: ?std.mem.Allocator = null,
    /// Status emitted in the most recent `response.completed`-class event,
    /// or null if no terminal event was seen.
    captured_status: ?[]u8 = null,

    pub fn sendText(self: *WsBridge, data: []const u8) !void {
        if (self.captured_resp_id == null) {
            if (self.allocator) |a| {
                if (extractResponseId(a, data)) |id| self.captured_resp_id = id;
            }
        }
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
            if (self.captured_resp_id) |id| a.free(id);
            if (self.captured_status) |st| a.free(st);
        }
        self.captured_resp_id = null;
        self.captured_status = null;
    }
};

/// Pull the value of the first `"id":"resp_..."` field out of a JSON
/// payload. Returns an owned copy or null. Used by `WsBridge.sendText` to
/// learn the response id without parsing the whole envelope.
fn extractResponseId(allocator: std.mem.Allocator, data: []const u8) ?[]u8 {
    const needle = "\"id\":\"resp_";
    const start = std.mem.indexOf(u8, data, needle) orelse return null;
    const v_start = start + "\"id\":\"".len;
    const v_end = std.mem.indexOfScalarPos(u8, data, v_start, '"') orelse return null;
    return allocator.dupe(u8, data[v_start..v_end]) catch null;
}

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

/// Connection wrapper bundling a TCP stream with its `Io` and per-connection
/// reader/writer buffers. Replaces `std.net.Stream` in 0.16 — methods that took
/// a bare stream now take `*Conn` so the IO interface and buffers travel together.
pub const Conn = struct {
    stream: std.Io.net.Stream,
    io: std.Io,
    write_buf: [16 * 1024]u8,
    read_buf: [16 * 1024]u8,
    write_state: std.Io.net.Stream.Writer,
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
    /// True once this connection has written a `text/event-stream` head: past it a generation
    /// failure can only be an SSE `error` event. Set only in `sendSseHeaders`.
    sse_headers_sent: bool = false,
    /// A Content-Length response ends by length and can release its socket immediately.
    length_framed: bool = false,
    /// Set for one request when Qwen `high`/`max` was served as `xhigh`. The
    /// text is the log line; `effort_notice_json` is `,"effort_notice":"..."`.
    effort_notice: ?[]const u8 = null,
    effort_notice_json: ?[]const u8 = null,

    pub fn init(c: *Conn, stream: std.Io.net.Stream, io: std.Io) void {
        c.stream = stream;
        c.io = io;
        c.write_state = stream.writer(io, &c.write_buf);
        c.read_state = stream.reader(io, &c.read_buf);
        c.ws_mode = null;
        c.sse_headers_sent = false;
        c.length_framed = false;
        c.effort_notice = null;
        c.effort_notice_json = null;
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

    pub fn reader(c: *Conn) *std.Io.Reader {
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

    /// Read up to buf.len bytes; may return fewer (HTTP-style short read). Returns 0 on EOF.
    /// Uses `readVec` because `readSliceShort` blocks until the buffer is full or EOF —
    /// wrong semantics for HTTP request parsing where we read until headers terminate.
    pub fn read(c: *Conn, buf: []u8) !usize {
        var bufs: [1][]u8 = .{buf};
        return c.reader().readVec(&bufs) catch |err| switch (err) {
            error.EndOfStream => 0,
            else => |e| e,
        };
    }

    /// Waits up to `ms` for bytes to read; false when none came. Bytes the reader already
    /// buffered count as readable.
    fn awaitReadable(c: *Conn, ms: i64) bool {
        if (c.reader().bufferedLen() > 0) return true;
        if (ms <= 0) return false;
        var fds = [_]std.posix.pollfd{.{ .fd = c.stream.socket.handle, .events = std.posix.POLL.IN, .revents = 0 }};
        const n = std.posix.poll(&fds, @intCast(@min(ms, std.math.maxInt(i32)))) catch return true;
        return n > 0;
    }

    const CLOSE_WAIT_MS: i64 = 5 * 60 * 1000;

    /// Keep close-delimited responses owned until the reader closes: orphaned FIN_WAIT_2
    /// sockets can expire with a reset while a slow client is still consuming the body.
    pub fn close(c: *Conn) void {
        c.flush() catch {};
        if (c.length_framed) return c.stream.close(c.io);
        const fd = c.stream.socket.handle;
        _ = std.posix.system.shutdown(fd, std.posix.SHUT.WR);
        const deadline = nowMsMonotonic(c.io) + CLOSE_WAIT_MS;
        var fds = [_]std.posix.pollfd{.{ .fd = fd, .events = std.posix.POLL.IN, .revents = 0 }};
        var sink: [256]u8 = undefined;
        while (nowMsMonotonic(c.io) < deadline and !server_mod.shutdown_requested.load(.acquire)) {
            const n = std.posix.poll(&fds, 500) catch break;
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
    const e = registry.peekPath(path) orelse return .not_found;
    return .{ .entry = e.id };
}

test "routeRequestModel: a path names its own entry, an unknown name the default" {
    const t = std.testing;
    const reg = try ModelRegistry.init(t.allocator, t.io, null, 4, 0, null);
    defer reg.deinit();
    _ = try reg.registerStub("org/good", "/m/org/good", 1);
    // The exact path wins over another pack registered under the same basename.
    _ = try reg.registerStub("good", "/elsewhere/good", 1);
    // `--model` and `/v1/load-model` register a path under its basename.
    _ = try reg.registerStub("Pack", "/x/y/Pack", 1);
    _ = try reg.registerStub("home-pack", "/Users/t/.sushi/models/home-pack", 1);
    const home: ?[]const u8 = "/Users/t";

    try t.expectEqualDeep(ModelRoute{ .entry = "org/good" }, routeRequestModel(reg, "org/good", home));
    try t.expectEqualDeep(ModelRoute{ .entry = "org/good" }, routeRequestModel(reg, "/m/org/good", home));
    try t.expectEqualDeep(ModelRoute{ .entry = "org/good" }, routeRequestModel(reg, "/m/org/good/", home));
    try t.expectEqualDeep(ModelRoute{ .entry = "Pack" }, routeRequestModel(reg, "/x/y/Pack", home));
    // The same basename-id resolution `/v1/load-model` applies.
    try t.expectEqualDeep(ModelRoute{ .entry = "Pack" }, routeRequestModel(reg, "/elsewhere/Pack", home));
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
    "/detokenize",
    "/health",
    "/metrics",
    "/metrics.json",
    "/metrics/enable",
    "/props",
    "/tokenize",
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/load-model",
    "/v1/messages",
    "/v1/messages/count_tokens",
    "/v1/models",
    "/v1/models/rescan",
    "/v1/responses",
    "/v1/responses/compact",
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
            std.mem.eql(u8, path, "/v1/models/rescan") or std.mem.eql(u8, path, "/metrics/enable");
    if (!guarded) return false;
    const origin = findHeaderValueCI(headers, "origin") orelse return false;
    return !update_mod.originMatches(origin, if (std.mem.eql(u8, host, "0.0.0.0")) "127.0.0.1" else host, port);
}

/// Is `path` an endpoint this server serves at all (any method)?
pub fn routeExists(path: []const u8) bool {
    // `/v1/responses/{id}` (GET/DELETE) is matched by prefix, not by a literal.
    if (std.mem.startsWith(u8, path, "/v1/responses/")) return true;
    for (ROUTE_PATHS) |p| if (std.mem.eql(u8, path, p)) return true;
    return false;
}

/// The 413 names BOTH numbers it compared (context-overflow-400 class): a
/// refusal that only names the cap reads as "this should have worked".
fn payloadTooLargeMessage(buf: []u8, got: usize, cap: usize) []const u8 {
    const mb = 1024 * 1024;
    return std.fmt.bufPrint(buf, "Request body too large: {d} MB exceeds the {d} MB limit", .{
        (got + mb - 1) / mb, cap / mb,
    }) catch "Request body too large";
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

const RawRequest = struct {
    /// Owned; `buf[0..len]` is the head plus as much of the body as arrived.
    buf: []u8,
    len: usize,
    header_end: usize,
};

/// Reads one request: the head (up to 16 KB), then its `Content-Length` body. Null when no
/// complete head arrived, the client stalled, or the body is over the cap (the 413 is already sent).
pub fn readRequestWithin(allocator: std.mem.Allocator, stream: *Conn, limit_ms: i64) !?RawRequest {
    const head_deadline = nowMsMonotonic(stream.io) + limit_ms;
    // Read HTTP headers first (up to 16KB), then allocate for the full body based on Content-Length.
    var hdr_buf: [16 * 1024]u8 = undefined;
    var total_read: usize = 0;
    var content_length: ?usize = null;
    var header_end_pos: usize = 0;

    // Phase 1: Read until we have complete headers
    while (total_read < hdr_buf.len) {
        if (!stream.awaitReadable(head_deadline - nowMsMonotonic(stream.io))) return null;
        const n = try stream.read(hdr_buf[total_read..]);
        if (n == 0) break;
        total_read += n;

        if (std.mem.indexOf(u8, hdr_buf[0..total_read], "\r\n\r\n")) |he| {
            header_end_pos = he + 4;
            content_length = findContentLength(hdr_buf[0..he]);
            break;
        }
    }

    if (header_end_pos == 0) return null;

    // Phase 2: Allocate buffer for full request and read remaining body
    const cl = content_length orelse 0;
    const total_size = header_end_pos +| cl;
    if (total_size > max_request_bytes) {
        var msg_buf: [128]u8 = undefined;
        const msg = payloadTooLargeMessage(&msg_buf, total_size, max_request_bytes);
        log.warn("[http] 413: {s}\n", .{msg});
        try sendErrorResponse(allocator, stream, "413 Payload Too Large", "invalid_request_error", msg, 413);
        return null;
    }

    const buf = try allocator.alloc(u8, total_size);
    errdefer allocator.free(buf);
    total_read = seedRequestBuf(buf, hdr_buf[0..total_read]);

    while (total_read < total_size) {
        if (!stream.awaitReadable(limit_ms)) {
            allocator.free(buf);
            return null;
        }
        const n = try stream.read(buf[total_read..total_size]);
        if (n == 0) break;
        total_read += n;
    }
    return .{ .buf = buf, .len = total_read, .header_end = header_end_pos };
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
    try stream.writeAll("\r\n");
    stream.sse_headers_sent = true;
    logHttpStreamStart(kind);
}

fn logHttpStreamStart(kind: []const u8) void {
    if (!log.isDebug()) return;
    log.debug("[http] <- 200 OK text/event-stream ({s})\n", .{kind});
}

fn findContentLength(headers: []const u8) ?usize {
    var lines = std.mem.splitSequence(u8, headers, "\r\n");
    while (lines.next()) |line| {
        const lower = "content-length: ";
        if (line.len >= lower.len) {
            var match = true;
            for (0..lower.len) |j| {
                if (std.ascii.toLower(line[j]) != lower[j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                return std.fmt.parseInt(usize, std.mem.trim(u8, line[lower.len..], " "), 10) catch null;
            }
        }
    }
    return null;
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

/// Case-insensitive HTTP header lookup in the raw header block. `name_lower`
/// must be lowercase (e.g. "authorization"). Returns the trimmed value, or null.
pub fn findHeaderValueCI(headers: []const u8, name_lower: []const u8) ?[]const u8 {
    var lines = std.mem.splitSequence(u8, headers, "\r\n");
    while (lines.next()) |line| {
        if (line.len <= name_lower.len + 1) continue;
        var match = true;
        for (0..name_lower.len) |j| {
            if (std.ascii.toLower(line[j]) != name_lower[j]) {
                match = false;
                break;
            }
        }
        if (!match) continue;
        if (line[name_lower.len] != ':') continue;
        return std.mem.trim(u8, line[name_lower.len + 1 ..], " \t");
    }
    return null;
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
    const key = server_mod.g_api_key orelse return true;
    if (findHeaderValueCI(raw_headers, "authorization")) |auth| {
        if (std.mem.startsWith(u8, auth, "Bearer ")) {
            if (constTimeEql(std.mem.trim(u8, auth["Bearer ".len..], " \t"), key)) return true;
        } else if (std.mem.startsWith(u8, auth, "Basic ")) {
            if (basicAuthMatches(std.mem.trim(u8, auth["Basic ".len..], " \t"), key)) return true;
        }
    }
    if (findHeaderValueCI(raw_headers, "x-api-key")) |xk| {
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

/// Send a 401 with a Basic-auth challenge so browsers prompt for the key on the
/// index + metrics pages; API clients read the JSON error body.
pub fn sendUnauthorized(stream: *Conn) !void {
    const body = "{\"error\":{\"message\":\"missing or invalid API key\",\"type\":\"authentication_error\"}}";
    logHttpResponse("401 Unauthorized", "application/json", body);
    if (stream.ws_mode) |bridge| {
        if (body.len > 0) try bridge.sendText(body);
        return;
    }
    var hdr_buf: [512]u8 = undefined;
    const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nWWW-Authenticate: Basic realm=\"sushi\"\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: Content-Type, Authorization, x-api-key\r\n\r\n", .{body.len}) catch return error.Overflow;
    try stream.writeAll(hdr);
    stream.length_framed = true;
    try stream.writeAll(body);
}

test "apiKeyAuthorized accepts Bearer, x-api-key, Basic, and query param" {
    const prev = server_mod.g_api_key;
    defer server_mod.g_api_key = prev;
    server_mod.g_api_key = "s3cret";

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
    try std.testing.expect(!apiKeyAuthorized("", "/v1/models"));

    // No key configured ⇒ always authorized (open mode)
    server_mod.g_api_key = null;
    try std.testing.expect(apiKeyAuthorized("", "/v1/chat/completions"));
}

/// One raw request through `handleConnection` from a loopback peer, over a
/// socketpair against an empty registry; returns the raw response (caller frees).
pub fn serveOneForTest(request: []const u8) ![]u8 {
    const a = std.testing.allocator;
    const reg = try ModelRegistry.init(a, std.testing.io, null, 1, 0, null);
    defer reg.deinit();
    const prev_registry = server_mod.global_registry;
    defer server_mod.global_registry = prev_registry;
    server_mod.global_registry = reg;

    var sv: [2]std.posix.fd_t = undefined;
    if (std.c.socketpair(1, 1, 0, &sv) != 0) return error.SocketPairFailed; // AF_UNIX, SOCK_STREAM
    defer _ = std.c.close(sv[1]);
    var conn: Conn = undefined;
    Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = .{ .ip4 = .loopback(0) } } }, std.testing.io);
    if (std.c.write(sv[1], request.ptr, request.len) != @as(isize, @intCast(request.len))) return error.WriteFailed;
    const server = try std.Thread.spawn(.{}, struct {
        fn run(c: *Conn) void {
            handleConnection(std.testing.allocator, c) catch {};
            c.close();
        }
    }.run, .{&conn});

    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(a);
    var buf: [16 * 1024]u8 = undefined;
    while (true) {
        const n = std.c.read(sv[1], &buf, buf.len);
        if (n <= 0) break;
        try out.appendSlice(a, buf[0..@intCast(n)]);
    }
    server.join();
    return out.toOwnedSlice(a);
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

test "dashboard page: GET /dashboard serves the same embedded page as HTML" {
    const t = std.testing;
    for ([_][]const u8{ "/dashboard", "/dashboard?x=1" }) |path| {
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
    const prev_key = server_mod.g_api_key;
    const prev_strict = server_mod.g_api_key_strict;
    defer {
        server_mod.g_api_key = prev_key;
        server_mod.g_api_key_strict = prev_strict;
    }
    server_mod.g_api_key = "s3cret";
    server_mod.g_api_key_strict = true;
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
    const prev_key = server_mod.g_api_key;
    const prev_strict = server_mod.g_api_key_strict;
    const old_host = server_mod.listen_host;
    const old_port = server_mod.listen_port;
    defer {
        server_mod.g_api_key = prev_key;
        server_mod.g_api_key_strict = prev_strict;
        server_mod.listen_host = old_host;
        server_mod.listen_port = old_port;
    }
    server_mod.g_api_key = "s3cret";
    server_mod.g_api_key_strict = true;
    server_mod.listen_host = "127.0.0.1";
    server_mod.listen_port = 12345;
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
    const prev_key = server_mod.g_api_key;
    const prev_strict = server_mod.g_api_key_strict;
    defer {
        server_mod.g_api_key = prev_key;
        server_mod.g_api_key_strict = prev_strict;
    }
    // Strict so the loopback test peer is gated like a network one.
    server_mod.g_api_key = "s3cret";
    server_mod.g_api_key_strict = true;
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
}

test "self-update: /v1/update refuses by name over the wire and never shuts the server down" {
    const t = std.testing;
    const prev_key = server_mod.g_api_key;
    defer server_mod.g_api_key = prev_key;
    const Case = struct { request: []const u8, status: []const u8, says: []const u8 };
    for ([_]Case{
        .{ .request = "GET /v1/update HTTP/1.1\r\n\r\n", .status = "HTTP/1.1 405 ", .says = "POST only" },
        .{ .request = "POST /v1/update HTTP/1.1\r\n\r\n", .status = "HTTP/1.1 403 ", .says = "no Origin" },
        .{ .request = "POST /v1/update HTTP/1.1\r\nOrigin: http://evil.test\r\n\r\n", .status = "HTTP/1.1 403 ", .says = "Origin" },
        // This test binary sits in a checkout: refused as a source build, before asking whether a release is newer.
        .{ .request = "POST /v1/update HTTP/1.1\r\nOrigin: http://localhost:12345\r\n\r\n", .status = "HTTP/1.1 409 ", .says = "built from source" },
    }) |c| {
        const response = try serveOneForTest(c.request);
        defer t.allocator.free(response);
        try t.expect(std.mem.startsWith(u8, response, c.status));
        try t.expect(std.mem.indexOf(u8, responseBody(response), c.says) != null);
    }
    // The key binds this route from loopback too: loopback is otherwise exempt.
    server_mod.g_api_key = "s3cret";
    const keyless = try serveOneForTest("POST /v1/update HTTP/1.1\r\nOrigin: http://localhost:12345\r\n\r\n");
    defer t.allocator.free(keyless);
    try t.expect(std.mem.startsWith(u8, keyless, "HTTP/1.1 401 "));
    try t.expect(!update_mod.relaunch_requested.load(.acquire));
    try t.expect(!server_mod.shutdown_requested.load(.acquire));
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
fn orphanFunctionCallOutputId(
    input_val: std.json.Value,
    prev_id: ?[]const u8,
    local: *WsLocalCache,
    global: *responses_mod.ResponseStore,
) ?[]const u8 {
    if (input_val != .array) return null;
    // Collect all known call_ids from (a) prev history and (b) current input's
    // function_call items.
    var known: std.StringHashMapUnmanaged(void) = .{};
    defer known.deinit(std.heap.page_allocator);
    if (prev_id) |pid| blk: {
        const sr = local.get(pid) orelse global.get(pid) orelse break :blk;
        for (sr.history) |m| {
            if (m.tool_calls) |tcs| for (tcs) |tc| {
                _ = known.put(std.heap.page_allocator, tc.id, {}) catch {};
            };
        }
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
fn wsSendErrorTurn(allocator: std.mem.Allocator, ws_conn: *WsConnT, status: u32, code: []const u8, message: []const u8) !void {
    const esc_code = try jsonEscape(allocator, code);
    defer allocator.free(esc_code);
    const esc_msg = try jsonEscape(allocator, message);
    defer allocator.free(esc_msg);
    const err_frame = try std.fmt.allocPrint(
        allocator,
        \\{{"type":"error","status":{d},"error":{{"code":{s},"message":{s}}}}}
    ,
        .{ status, esc_code, esc_msg },
    );
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
    lm: *LoadedModel,
) !void {
    ws_mod.handshake(stream, headers) catch |err| {
        log.warn("WS handshake failed: {s}\n", .{@errorName(err)});
        return;
    };
    log.info("WS /v1/responses connected\n", .{});

    var ws_conn = WsConnT.init(stream);
    defer ws_conn.deinit(allocator);

    var local_cache: WsLocalCache = .{ .gpa = allocator };
    defer local_cache.deinit();

    var bridge: WsBridge = .{ .impl = &ws_conn, .sendTextFn = &wsBridgeSend, .allocator = allocator };
    defer bridge.reset();

    // Frame loop — one iteration per inbound WS message.
    while (true) {
        const msg = ws_conn.readMessage(allocator) catch |err| switch (err) {
            error.WsClosed => return,
            error.WsProtocol => {
                ws_conn.writeClose(1002, "protocol error") catch {};
                return;
            },
            error.WsTooLarge => {
                ws_conn.writeClose(1009, "message too large") catch {};
                return;
            },
            else => return,
        };
        defer allocator.free(msg.payload);

        switch (msg.opcode) {
            .close => {
                ws_conn.writeClose(1000, "") catch {};
                return;
            },
            .ping => {
                try ws_conn.writePong(msg.payload);
                continue;
            },
            .pong => continue,
            .text => {},
            else => continue,
        }

        if (chat_mod.jsonNestingExceeds(msg.payload, chat_mod.max_json_nesting)) {
            try wsSendErrorTurn(allocator, &ws_conn, 400, "invalid_request_error", "JSON nesting is too deep");
            continue;
        }
        // Parse the request payload — must be {"type":"response.create", ...}
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, msg.payload, .{}) catch {
            try wsSendErrorTurn(allocator, &ws_conn, 400, "invalid_request_error", "Invalid JSON in request body");
            continue;
        };
        defer parsed.deinit();
        if (parsed.value != .object) {
            try wsSendErrorTurn(allocator, &ws_conn, 400, "invalid_request_error", "Request must be a JSON object");
            continue;
        }
        const root = parsed.value.object;
        const type_val = root.get("type");
        if (type_val == null or type_val.? != .string or !std.mem.eql(u8, type_val.?.string, "response.create")) {
            try wsSendErrorTurn(allocator, &ws_conn, 400, "invalid_request_error", "Expected type=response.create");
            continue;
        }
        // The plan disallows stream/stream_options/background here — WS is
        // inherently streaming and stateless (no background jobs).
        if (root.get("stream") != null or root.get("stream_options") != null or root.get("background") != null) {
            try wsSendErrorTurn(allocator, &ws_conn, 400, "invalid_request_error", "stream/stream_options/background are not allowed over WebSocket");
            continue;
        }

        // ── Resolve previous_response_id against per-conn cache first.
        //    For store:false continuations on this connection, the chain
        //    root is in `local_cache` only. For store:true responses we
        //    fall back to the global store (lets stored chains survive
        //    across connections). On miss, evict any stale local entry
        //    before reporting the error.
        const prev_id_owned: ?[]const u8 = if (root.get("previous_response_id")) |v|
            (if (v == .string) v.string else null)
        else
            null;
        var prev_in_local: bool = false;
        if (prev_id_owned) |pid| {
            if (local_cache.get(pid) != null) {
                prev_in_local = true;
            } else {
                const store = getOrInitResponseStore(stream.io, allocator);
                if (store.get(pid) == null) {
                    local_cache.evict(pid);
                    try wsSendErrorTurn(allocator, &ws_conn, 404, "previous_response_not_found", "previous_response_id not found");
                    continue;
                }
            }
        }

        // Validate any function_call_output items reference real call_ids
        // in the prev response's history. An orphan output means the user
        // is trying to feed back a tool result for a call the model never
        // made — the turn must fail and (per compliance) evict the chain
        // root from the local cache.
        if (prev_id_owned) |pid| {
            if (root.get("input")) |input_v| {
                if (orphanFunctionCallOutputId(input_v, prev_id_owned, &local_cache, getOrInitResponseStore(stream.io, allocator))) |_| {
                    local_cache.evict(pid);
                    try wsSendErrorTurn(allocator, &ws_conn, 400, "invalid_request_error", "function_call_output references a missing call_id");
                    continue;
                }
            }
        }

        // For store:false continuations on this connection, the chain root
        // lives in `local_cache` (not the global store). Move it into the
        // global store *just for this turn* so handleResponses' lookup at
        // line 4634 can find it. We restore the original location at end
        // of turn. (The kludge avoids modifying handleResponses.)
        const did_borrow_to_global = prev_id_owned != null and prev_in_local;
        if (did_borrow_to_global) {
            const sr = local_cache.map.get(prev_id_owned.?).?;
            // Remove from local without freeing.
            _ = local_cache.map.remove(prev_id_owned.?);
            const store = getOrInitResponseStore(stream.io, allocator);
            store.put(sr) catch {};
        }

        // ── Reserialize the request as an HTTP body, forcing
        //    `stream: true` (WS is inherently streaming) and `store: true`
        //    so handleResponses always persists the result. We move the
        //    entry to local_cache below if the user actually wanted
        //    `store: false`, achieving connection-scoped lifetime.
        const want_user_store: bool = if (root.get("store")) |v| (if (v == .bool) v.bool else true) else true;
        const body = try buildResponsesBodyFromWsRequest(allocator, root);
        defer allocator.free(body);

        // Reset sequence numbering per response per the OpenAI spec.
        // (handleResponses owns its own seq_num, fresh each call.)
        var seq: u64 = 0;
        _ = &seq;

        stream.ws_mode = &bridge;
        defer stream.ws_mode = null;

        bridge.reset();
        handleResponses(allocator, stream, body, lm) catch |err| {
            log.warn("WS handleResponses error: {s}\n", .{@errorName(err)});
            // Best-effort error frame; same mapping as every HTTP surface.
            var ws_err_buf: [192]u8 = undefined;
            const ws_wire = mapGenerationError(err, &ws_err_buf);
            wsSendErrorTurn(allocator, &ws_conn, ws_wire.code, ws_wire.openai_type, ws_wire.message) catch {};
            // Restore borrowed prev entry back to local cache on failure.
            if (did_borrow_to_global and prev_id_owned != null) {
                const store = getOrInitResponseStore(stream.io, allocator);
                if (store.map.fetchRemove(prev_id_owned.?)) |kv| {
                    store.lru.remove(&kv.value.list_node);
                    local_cache.map.put(local_cache.gpa, kv.value.id, kv.value) catch {};
                }
            }
            continue;
        };

        // Handle the borrowed prev: move it back from global to local
        // cache. On success, also do the eviction-on-failure check —
        // if the just-completed turn ended in a non-completed status
        // (failed/incomplete), evict the chain root from local cache.
        if (did_borrow_to_global and prev_id_owned != null) {
            const store = getOrInitResponseStore(stream.io, allocator);
            if (store.map.fetchRemove(prev_id_owned.?)) |kv| {
                store.lru.remove(&kv.value.list_node);
                const turn_failed = bridge.captured_status != null and !std.mem.eql(u8, bridge.captured_status.?, "completed");
                if (turn_failed) {
                    // Compliance: a failed continuation evicts the chain root.
                    kv.value.deinit();
                } else {
                    local_cache.map.put(local_cache.gpa, kv.value.id, kv.value) catch {};
                }
            }
        }

        // For store:false on this WS, move the freshly-stored response
        // from the global store into the connection-local cache so it
        // (a) survives the user's `store: false` semantics within this
        // connection (allowing previous_response_id chains) and
        // (b) does NOT leak across reconnects (other WS connections
        // looking up this id should get previous_response_not_found).
        if (!want_user_store) {
            if (bridge.captured_resp_id) |rid| {
                const store = getOrInitResponseStore(stream.io, allocator);
                if (store.map.fetchRemove(rid)) |kv| {
                    store.lru.remove(&kv.value.list_node);
                    local_cache.map.put(local_cache.gpa, kv.value.id, kv.value) catch {};
                }
            }
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
            try responses_mod.serializeJsonValue(allocator, &buf, entry.value_ptr.*);
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
    const was_shutdown = server_mod.shutdown_requested.swap(false, .acq_rel);
    defer server_mod.shutdown_requested.store(was_shutdown, .release);
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

    // ... and only a close-delimited body keeps the server's end open, so the kernel cannot
    // reset a lagging reader.
    var waited: u32 = 0;
    while (!closed.load(.acquire) and waited < 50) : (waited += 1) {
        std.Io.sleep(testing.io, .fromMilliseconds(1), .real) catch {};
    }
    try testing.expectEqual(!close_delimited, closed.load(.acquire));

    if (stop_server) {
        server_mod.shutdown_requested.store(true, .release);
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

test "Conn.close does not wait for the peer after a Content-Length response" {
    try expectCloseWaitsForPeer(false, false);
}

test "Conn.close stops waiting when the server shuts down" {
    try expectCloseWaitsForPeer(true, true);
}

test "findContentLength parses header" {
    try testing.expectEqual(@as(?usize, 42), findContentLength("Host: localhost\r\nContent-Length: 42\r\nAccept: */*"));
}

test "findContentLength case insensitive" {
    try testing.expectEqual(@as(?usize, 100), findContentLength("content-length: 100"));
    try testing.expectEqual(@as(?usize, 100), findContentLength("Content-Length: 100"));
    try testing.expectEqual(@as(?usize, 100), findContentLength("CONTENT-LENGTH: 100"));
}

test "findContentLength returns null when missing" {
    try testing.expect(findContentLength("Host: localhost\r\nAccept: */*") == null);
    try testing.expect(findContentLength("") == null);
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
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    var cfg = model_mod.ModelConfig{ .num_hidden_layers = 0, .pinned_context = 4096 };
    var lm: LoadedModel = undefined;
    lm.config = &cfg;
    lm.tokenizer = &tok;
    lm.transformer = null;
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

test "the 413 names both counts it compared" {
    var buf: [128]u8 = undefined;
    const msg = payloadTooLargeMessage(&buf, 97 * 1024 * 1024 + 1, 64 * 1024 * 1024);
    try std.testing.expect(std.mem.indexOf(u8, msg, "98 MB") != null);
    try std.testing.expect(std.mem.indexOf(u8, msg, "64 MB") != null);
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
    const old_host = server_mod.listen_host;
    const old_port = server_mod.listen_port;
    defer {
        server_mod.listen_host = old_host;
        server_mod.listen_port = old_port;
    }
    server_mod.listen_host = "127.0.0.1";
    server_mod.listen_port = 12345;
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
    const old_edit = server_mod.g_web_edit;
    defer server_mod.g_web_edit = old_edit;
    for ([_]bool{ false, true }) |edit_default| {
        server_mod.g_web_edit = edit_default;
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
        .{ .method = "GET", .path = "/v1/responses", .headers = ws ++ "Origin: http://evil.test\r\n", .refused = true },
        .{ .method = "POST", .path = "/metrics/enable", .headers = "Origin: http://evil.test\r\n", .refused = true },
        .{ .method = "POST", .path = "/metrics/enable", .headers = "Origin: http://localhost:12345\r\n", .refused = false },
        .{ .method = "POST", .path = "/v1/load-model", .headers = "Origin: http://localhost:12345\r\n", .refused = false },
        .{ .method = "POST", .path = "/v1/models/rescan", .headers = "Origin: http://127.0.0.1:12345\r\n", .refused = false },
        .{ .method = "GET", .path = "/v1/responses", .headers = ws ++ "Origin: http://127.0.0.1:12345\r\n", .refused = false },
        .{ .method = "POST", .path = "/v1/load-model", .headers = "Origin: http://localhost:12345\r\n", .host = "0.0.0.0", .refused = false },
        .{ .method = "POST", .path = "/v1/unload-model", .headers = "Host: x\r\n", .refused = false },
        .{ .method = "GET", .path = "/v1/responses", .headers = ws, .refused = false },
        // The inference routes stay open to any page.
        .{ .method = "POST", .path = "/v1/chat/completions", .headers = "Origin: http://evil.test\r\n", .refused = false },
        .{ .method = "GET", .path = "/v1/responses", .headers = "Origin: http://evil.test\r\n", .refused = false },
    }) |c| try t.expectEqual(c.refused, crossOriginRefused(c.method, c.path, c.headers, c.host, 12345));
}

test "cross-origin policy: the routes answer 403 over the wire before any model resolves" {
    const t = std.testing;
    const old_host = server_mod.listen_host;
    const old_port = server_mod.listen_port;
    defer {
        server_mod.listen_host = old_host;
        server_mod.listen_port = old_port;
    }
    server_mod.listen_host = "127.0.0.1";
    server_mod.listen_port = 12345;
    const Case = struct { request: []const u8, status: []const u8 };
    for ([_]Case{
        .{ .request = "POST /v1/load-model HTTP/1.1\r\nOrigin: http://evil.test\r\nContent-Length: 2\r\n\r\n{}", .status = "HTTP/1.1 403 " },
        .{ .request = "POST /v1/unload-model HTTP/1.1\r\nOrigin: http://evil.test\r\nContent-Length: 2\r\n\r\n{}", .status = "HTTP/1.1 403 " },
        .{ .request = "POST /v1/models/rescan HTTP/1.1\r\nOrigin: http://evil.test\r\n\r\n", .status = "HTTP/1.1 403 " },
        .{ .request = "GET /v1/responses HTTP/1.1\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nOrigin: http://evil.test\r\n\r\n", .status = "HTTP/1.1 403 " },
        .{ .request = "POST /v1/models/rescan HTTP/1.1\r\nOrigin: http://localhost:12345\r\n\r\n", .status = "HTTP/1.1 200 " },
        .{ .request = "POST /v1/models/rescan HTTP/1.1\r\n\r\n", .status = "HTTP/1.1 200 " },
    }) |c| {
        const response = try serveOneForTest(c.request);
        defer t.allocator.free(response);
        try t.expect(std.mem.startsWith(u8, response, c.status));
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
    var conn: Conn = undefined;
    Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = undefined } }, t.io);
    defer conn.close();
    _ = std.c.send(sv[1], sent.ptr, sent.len, 0);

    var done = std.atomic.Value(bool).init(false);
    const reader = try std.Thread.spawn(.{}, struct {
        fn run(c: *Conn, limit: i64, d: *std.atomic.Value(bool)) void {
            if (readRequestWithin(t.allocator, c, limit) catch null) |r| t.allocator.free(r.buf);
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
    try t.expect(isAnthropicRoute("/v1/messages/count_tokens"));

    const prev_key = server_mod.g_api_key;
    const prev_strict = server_mod.g_api_key_strict;
    defer {
        server_mod.g_api_key = prev_key;
        server_mod.g_api_key_strict = prev_strict;
    }
    server_mod.g_api_key = "s3cret";
    server_mod.g_api_key_strict = true;
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
