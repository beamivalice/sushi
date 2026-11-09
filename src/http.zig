//! The HTTP transport of the server: the bind address, the response writers and their debug log, the error body, and the
//! JSON string escape every body goes through. No request is interpreted here.
const chat_mod = @import("chat.zig");
const http_conn = @import("http_conn.zig");
const instr = @import("metrics.zig");
const log = @import("log.zig");
const request_policy = @import("request_policy.zig");
const responses_mod = @import("responses.zig");
const server_state = @import("server_state.zig");
const std = @import("std");
const testing = std.testing;

pub const BindConfig = struct {
    host: []const u8,
    port: u16,
};

/// `settings_port` is the settings file's `server.port` or the built-in default: `--port` wins over both.
pub fn resolveBind(host_flag: ?[]const u8, port_flag: ?u16, settings_port: u16) BindConfig {
    return .{
        .host = host_flag orelse server_state.default_host,
        .port = port_flag orelse settings_port,
    };
}

pub fn bindAddress(host: []const u8, port: u16) error{InvalidHost}!std.Io.net.IpAddress {
    var ip4_bytes: [4]u8 = undefined;
    if (std.mem.eql(u8, host, "localhost")) {
        ip4_bytes = .{ 127, 0, 0, 1 };
    } else if (std.mem.eql(u8, host, "0.0.0.0")) {
        ip4_bytes = .{ 0, 0, 0, 0 };
    } else {
        // Parse dotted-decimal IP
        var parts = std.mem.splitScalar(u8, host, '.');
        var idx: usize = 0;
        while (parts.next()) |part| {
            if (idx >= 4) return error.InvalidHost;
            ip4_bytes[idx] = std.fmt.parseInt(u8, part, 10) catch return error.InvalidHost;
            idx += 1;
        }
        if (idx != 4) return error.InvalidHost;
    }
    return .{ .ip4 = .{ .bytes = ip4_bytes, .port = port } };
}

pub fn ensurePortFree(io: std.Io, host: []const u8, port: u16) (error{InvalidHost} || std.Io.net.IpAddress.ListenError)!void {
    const addr = try bindAddress(host, port);
    if (addr.connect(io, .{ .mode = .stream })) |s| {
        s.close(io);
        return error.AddressInUse;
    } else |_| {}
    var probe = try http_conn.startListener(host, port);
    probe.deinit(io);
}

pub fn startupRefusal(err: anyerror, host: []const u8, port: u16, buf: []u8) ?[]const u8 {
    switch (err) {
        error.AddressInUse => return std.fmt.bufPrint(buf, "port {d} is already in use", .{port}) catch unreachable,
        error.InvalidHost => return std.fmt.bufPrint(buf, "invalid --host '{s}'", .{host}) catch "invalid --host",
        else => return null,
    }
}

/// A response whose head carries `extra_headers` (complete `Name: value\r\n` lines), then closes: the 405's
/// `Allow` and a 503's `Retry-After`.
pub fn sendResponseWithHeaders(stream: *http_conn.Conn, status: []const u8, content_type: []const u8, body: []const u8, extra_headers: []const u8) !void {
    logHttpResponse(status, content_type, body);
    if (stream.ws_mode) |bridge| {
        if (body.len > 0) try bridge.sendText(body);
        return;
    }
    var hdr_buf: [768]u8 = undefined;
    const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 {s}\r\nContent-Type: {s}\r\nContent-Length: {d}\r\n{s}Connection: close\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: POST, GET, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\n\r\n", .{
        status, content_type, body.len, extra_headers,
    }) catch return error.Overflow;
    try stream.writeAllNoFlush(hdr);
    stream.length_framed = true;
    if (body.len > 0 and !stream.head_only) try stream.writeAll(body) else try stream.flush();
}

/// A known path asked with a method it does not answer: 405 with the `Allow` list.
pub fn sendMethodNotAllowed(allocator: std.mem.Allocator, stream: *http_conn.Conn, path: []const u8, allow: []const u8) !void {
    const message = try std.fmt.allocPrint(allocator, "{s} answers {s} only", .{ path, allow });
    defer allocator.free(message);
    const escaped = try jsonEscape(allocator, message);
    defer allocator.free(escaped);
    const body = try std.fmt.allocPrint(allocator, "{{\"error\":{{\"message\":{s},\"type\":\"invalid_request_error\",\"param\":null,\"code\":405}}}}", .{escaped});
    defer allocator.free(body);
    var allow_line: [96]u8 = undefined;
    const header = std.fmt.bufPrint(&allow_line, "Allow: {s}\r\n", .{allow}) catch "";
    try sendResponseWithHeaders(stream, "405 Method Not Allowed", "application/json", body, header);
}

/// The body-less answer to a HEAD request: the head a GET would send, its Content-Length included.
fn sendHeadOnly(stream: *http_conn.Conn, status: []const u8, content_type: []const u8, body_len: usize) !void {
    var hdr_buf: [512]u8 = undefined;
    const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 {s}\r\nContent-Type: {s}\r\nContent-Length: {d}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n", .{ status, content_type, body_len }) catch return error.Overflow;
    try stream.writeAllNoFlush(hdr);
    stream.length_framed = true;
    try stream.flush();
}

pub fn sendResponse(stream: *http_conn.Conn, status: []const u8, content_type: []const u8, body: []const u8) !void {
    logHttpResponse(status, content_type, body);
    try sendResponseFramed(stream, status, content_type, body);
}

/// Like `sendResponse` but does NOT dump the response body to the debug log —
/// only a one-line status summary. For high-frequency, self-describing bodies
/// where the full dump is pure noise: the Prometheus scrape (`/metrics`, hit
/// every couple seconds) and the panel's JSON feed (`/metrics.json`, ~1 Hz).
pub fn sendResponseQuiet(stream: *http_conn.Conn, status: []const u8, content_type: []const u8, body: []const u8) !void {
    if (log.isDebug()) log.debug("[http] <- {s} {s} body={d}b (scrape; body not logged)\n", .{ status, content_type, body.len });
    try sendResponseFramed(stream, status, content_type, body);
}

pub fn sendResponseFramed(stream: *http_conn.Conn, status: []const u8, content_type: []const u8, body: []const u8) !void {
    if (stream.head_only) return sendHeadOnly(stream, status, content_type, body.len);
    const extra = stream.effort_notice_json orelse "";
    const inject = extra.len > 0 and body.len > 0 and body[body.len - 1] == '}' and std.mem.indexOf(u8, body, "\"effort_notice\":") == null;
    const out_len = body.len + if (inject) extra.len else 0;
    if (stream.ws_mode) |bridge| {
        if (body.len > 0) try bridge.sendText(body);
        return;
    }

    var hdr_buf: [512]u8 = undefined;
    const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 {s}\r\nContent-Type: {s}\r\nContent-Length: {d}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: POST, GET, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\n", .{
        status,
        content_type,
        out_len,
    }) catch return error.Overflow;
    try stream.writeAllNoFlush(hdr);
    try request_policy.writeEffortNoticeHeader(stream);
    try stream.writeAllNoFlush("\r\n");
    stream.length_framed = true;
    if (inject) {
        try stream.writeAllNoFlush(body[0 .. body.len - 1]);
        try stream.writeAllNoFlush(extra);
        try stream.writeAll("}");
    } else if (body.len > 0) {
        try stream.writeAll(body);
    } else try stream.flush();
}

/// `raw_path` with the value of every `api_key` / `key` query parameter replaced by `***`; a query too long for
/// `buf` is dropped whole rather than risk logging a key.
pub fn redactQueryKeys(buf: []u8, raw_path: []const u8) []const u8 {
    const qpos = std.mem.indexOfScalar(u8, raw_path, '?') orelse return raw_path;
    const fail_closed = raw_path[0 .. qpos + 1];
    if (qpos + 1 > buf.len) return fail_closed[0..@min(fail_closed.len, buf.len)];
    @memcpy(buf[0 .. qpos + 1], fail_closed);
    var n = qpos + 1;
    var it = std.mem.splitScalar(u8, raw_path[qpos + 1 ..], '&');
    var first = true;
    while (it.next()) |pair| {
        const eq = std.mem.indexOfScalar(u8, pair, '=');
        const secret = eq != null and (std.mem.eql(u8, pair[0..eq.?], "api_key") or std.mem.eql(u8, pair[0..eq.?], "key"));
        const kept = if (secret) pair[0 .. eq.? + 1] else pair;
        const tail = if (secret) "***" else "";
        const need = @intFromBool(!first) + kept.len + tail.len;
        if (n + need > buf.len) return fail_closed;
        if (!first) {
            buf[n] = '&';
            n += 1;
        }
        @memcpy(buf[n..][0..kept.len], kept);
        n += kept.len;
        @memcpy(buf[n..][0..tail.len], tail);
        n += tail.len;
        first = false;
    }
    return buf[0..n];
}

pub fn logHttpRequest(method: []const u8, raw_path: []const u8, body: []const u8) void {
    if (!log.isDebug()) return;
    var buf: [1024]u8 = undefined;
    log.debug("[http] -> {s} {s} body={d}b\n", .{ method, redactQueryKeys(&buf, raw_path), body.len });
    logHttpBody("[http] request body", body);
}

pub fn logHttpResponse(status: []const u8, content_type: []const u8, body: []const u8) void {
    if (!log.isDebug()) return;
    log.debug("[http] <- {s} {s} body={d}b\n", .{ status, content_type, body.len });
    logHttpBody("[http] response body", body);
}

/// The `Access-Control-Allow-Headers` value each SSE surface advertises.
pub const SSE_ALLOW_HEADERS_DEFAULT = "Content-Type, Authorization";

pub fn logHttpSseData(data: []const u8) void {
    if (!log.isDebug()) return;
    log.debug("[http] <- sse data={d}b\n", .{data.len});
    logHttpBody("[http] sse data", data);
}

/// Cap on how much of a BINARY body reaches the debug log.
const BODY_LOG_LIMIT = 4096;

/// True when every byte is printable or ordinary whitespace. Multibyte UTF-8
/// (≥0x80) counts as text, so an emoji in a chat body doesn't demote it.
fn bodyIsText(body: []const u8) bool {
    for (body) |c| {
        if (c == '\n' or c == '\r' or c == '\t') continue;
        if (c < 0x20 or c == 0x7f) return false;
    }
    return true;
}

/// Bounded, strictly-printable view of a body, copied into `out`.
fn bodyPreview(out: []u8, body: []const u8) []const u8 {
    const n = @min(@min(body.len, out.len), BODY_LOG_LIMIT);
    for (body[0..n], 0..) |c, i| {
        out[i] = if (c == '\n' or c == '\r' or c == '\t' or (c >= 0x20 and c < 0x7f)) c else '.';
    }
    return out[0..n];
}

pub fn logHttpBody(label: []const u8, body: []const u8) void {
    if (body.len == 0) return;
    // Text bodies go out WHOLE — reading a full request body out of the debug
    // log is the documented way to reproduce a tool-calling bug.
    if (bodyIsText(body)) {
        log.debug("{s} ({d}b):\n{s}\n", .{ label, body.len, body });
        return;
    }
    // Binary is a different story. `/v1/images/edits` is multipart, so one image
    // upload wrote raw PNG bytes into the log, and a body near the 64 MB request
    // cap blew through the 32 MB rotation — taking the post-mortem file with it.
    var buf: [BODY_LOG_LIMIT]u8 = undefined;
    const preview = bodyPreview(&buf, body);
    log.debug("{s} ({d}b binary, first {d} sanitized):\n{s}…\n", .{ label, body.len, preview.len, preview });
}

/// True if the connection's peer is a loopback address (127.0.0.0/8, ::1, or an
/// IPv4-mapped-IPv6 loopback). Loopback is the local machine → trusted and
/// exempt from the API key, so the same-host app never needs credentials.
pub fn ipIsLoopback(addr: std.Io.net.IpAddress) bool {
    return switch (addr) {
        .ip4 => |a4| a4.bytes[0] == 127, // 127.0.0.0/8
        .ip6 => |a6| blk: {
            const b = a6.bytes; // big-endian 16 bytes
            const zeros10: [10]u8 = @splat(0);
            // ::1
            const zeros15: [15]u8 = @splat(0);
            if (std.mem.eql(u8, b[0..15], &zeros15) and b[15] == 1) break :blk true;
            // ::ffff:127.x.x.x  (IPv4-mapped loopback)
            if (std.mem.eql(u8, b[0..10], &zeros10) and b[10] == 0xff and b[11] == 0xff and b[12] == 127) break :blk true;
            break :blk false;
        },
    };
}

pub fn peerIsLoopback(conn: *const http_conn.Conn) bool {
    return ipIsLoopback(conn.stream.socket.address);
}

pub fn responseBody(response: []const u8) []const u8 {
    const at = std.mem.indexOf(u8, response, "\r\n\r\n") orelse return "";
    return response[at + 4 ..];
}

test "ipIsLoopback exempts local addresses only" {
    // IPv4 loopback (whole 127.0.0.0/8) is exempt; a LAN address is not.
    try std.testing.expect(ipIsLoopback(.{ .ip4 = .{ .bytes = .{ 127, 0, 0, 1 }, .port = 0 } }));
    try std.testing.expect(ipIsLoopback(.{ .ip4 = .{ .bytes = .{ 127, 3, 2, 1 }, .port = 0 } }));
    try std.testing.expect(!ipIsLoopback(.{ .ip4 = .{ .bytes = .{ 192, 168, 1, 42 }, .port = 0 } }));
    try std.testing.expect(!ipIsLoopback(.{ .ip4 = .{ .bytes = .{ 10, 0, 0, 5 }, .port = 0 } }));
    // IPv6 ::1 is exempt; a routable v6 address is not.
    try std.testing.expect(ipIsLoopback(.{ .ip6 = .{ .bytes = .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }, .port = 0 } }));
    // IPv4-mapped loopback ::ffff:127.0.0.1
    try std.testing.expect(ipIsLoopback(.{ .ip6 = .{ .bytes = .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 127, 0, 0, 1 }, .port = 0 } }));
    try std.testing.expect(!ipIsLoopback(.{ .ip6 = .{ .bytes = .{ 0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }, .port = 0 } }));
}

/// Body text for a context-overflow 400, in a caller-owned buffer.
///
/// The COUNTS are the point. "Prompt exceeds maximum context length" tells a
/// client only that it failed; with both numbers the client can say how far
/// over it went and offer the one action that fixes it (raise the context to
/// N). Our own app renders exactly that, and the numbers are only knowable
/// here — the request is rejected before any usage is reported.
///
/// The legacy sentence stays the PREFIX: `APIError.looksLikeContextOverflow`
/// and third-party clients key on it, so appending must not move it. Output is
/// digits and spaces, so it carries no escaping hazard into the JSON sink, and
/// a bufPrint failure falls back to the bare sentence rather than sending no
/// body at all (the media-gen fixed-buffer class).
/// The `error.NotEnoughMemory` refusal, in ONE place because two identical
/// copies of it is how they drift.
///
/// The old wording ("retry after current requests complete") named concurrency,
/// which is the one thing it is never waiting for: the gate refuses against a
/// STATIC cap, so on an idle server with nothing loaded there is nothing to
/// retry after (#126). The counts are knowable only server-side and go to the
/// log; what the client needs is the knob.
///
/// ONE gate reaches this arm: the resident-model budget with no evictable
/// LRU victim (scheduler's `planEvictionsLocked` refusal). The free-memory
/// PREFLIGHT refuses as `error.InsufficientMemory` and gets its own message
/// below (#144) — blaming free memory here sends a user quitting apps when
/// the fix is unloading a model or moving the cap.
pub const not_enough_memory_message =
    "Not enough memory to load model: it would exceed the resident-model budget and no loaded model can be evicted. " ++
    "Unload the model you are chatting with (tray > Models > eject), " ++
    "or raise or disable the cap with --max-resident-mem <size>|0. " ++
    "The server log names the exact figures and the current cap.";

/// The count cap: `--max-resident-models` models are resident and each is serving a request.
pub const resident_models_cap_message =
    "Cannot load another model: the --max-resident-models cap is reached and every loaded model is serving a request. " ++
    "Retry when one finishes, or raise the cap with --max-resident-models <n>. The server log names the cap.";

/// Seconds a client waits before retrying a refusal that clears when a running request ends.
pub const BUSY_RETRY_AFTER_SECS: u32 = 5;

/// #144: the memory PREFLIGHT refusal (free RAM can't hold weights + warmup
/// headroom) is distinct from the resident-budget gate above and has a
/// different knob. Counts go to the log (#126); the client gets the remedy.
pub const insufficient_free_memory_message =
    "Not enough free memory to load model: weights + warmup headroom exceed what is currently available. " ++
    "Close other apps or models and retry (the server log names the peak estimate and available memory), or turn on \"Skip the memory check before loading\" in Settings (or pass --skip-mem-preflight) to override.";

/// A request refused before it owned a slot. A null sink (metrics off) is one branch.
pub fn countRejected() void {
    if (instr.current()) |m| m.recordRejected();
}

pub fn sendErrorResponse(allocator: std.mem.Allocator, stream: *http_conn.Conn, status: []const u8, err_type: []const u8, message: []const u8, code: ?u32) !void {
    return sendErrorResponseWithHeaders(allocator, stream, status, err_type, message, code, "");
}

/// `sendErrorResponse` whose head also carries `extra_headers` (complete `Name: value\r\n` lines).
pub fn sendErrorResponseWithHeaders(allocator: std.mem.Allocator, stream: *http_conn.Conn, status: []const u8, err_type: []const u8, message: []const u8, code: ?u32, extra_headers: []const u8) !void {
    const escaped_msg = try jsonEscape(allocator, message);
    defer allocator.free(escaped_msg);

    var code_buf: [16]u8 = undefined;
    const code_str = if (code) |c|
        std.fmt.bufPrint(&code_buf, "{d}", .{c}) catch "null"
    else
        "null";

    const body = try std.fmt.allocPrint(allocator,
        \\{{"error":{{"message":{s},"type":"{s}","param":null,"code":{s}}}}}
    , .{ escaped_msg, err_type, code_str });
    defer allocator.free(body);
    if (extra_headers.len == 0) return sendResponse(stream, status, "application/json", body);
    try sendResponseWithHeaders(stream, status, "application/json", body, extra_headers);
}

/// THE JSON string escaper for every response body. Every string here is model bytes
/// somewhere and a token is a BPE fragment, so invalid UTF-8 is sanitized INSIDE the
/// escaper (one U+FFFD per bad sequence); valid input passes through byte for byte.
pub fn jsonEscape(allocator: std.mem.Allocator, input: []const u8) ![]const u8 {
    if (std.unicode.utf8ValidateSlice(input)) return jsonEscapeValid(allocator, input);
    const clean = try chat_mod.utf8Sanitize(allocator, input);
    defer allocator.free(clean);
    return jsonEscapeValid(allocator, clean);
}

fn jsonEscapeValid(allocator: std.mem.Allocator, input: []const u8) ![]const u8 {
    var result = std.ArrayList(u8).empty;
    errdefer result.deinit(allocator);

    try result.append(allocator, '"');
    for (input) |c| {
        switch (c) {
            '"' => try result.appendSlice(allocator, "\\\""),
            '\\' => try result.appendSlice(allocator, "\\\\"),
            '\n' => try result.appendSlice(allocator, "\\n"),
            '\r' => try result.appendSlice(allocator, "\\r"),
            '\t' => try result.appendSlice(allocator, "\\t"),
            else => {
                if (c < 0x20) {
                    var esc_buf: [6]u8 = undefined;
                    const s = std.fmt.bufPrint(&esc_buf, "\\u{x:0>4}", .{c}) catch unreachable;
                    try result.appendSlice(allocator, s);
                } else {
                    try result.append(allocator, c);
                }
            },
        }
    }
    try result.append(allocator, '"');
    return result.toOwnedSlice(allocator);
}

test "EVERY JSON string escaper survives bytes that are not valid UTF-8" {
    // Class guard over the three escapers (OpenAI/Anthropic, Responses, the prompt
    // render): one passing bytes through loses a whole reply.
    const a = std.testing.allocator;
    const cases = [_][]const u8{
        "near\xe8\x91\xe4\xb8\x89 in", // 3-byte lead, one continuation, then another lead
        "\xff\xfe", // never-valid lead bytes
        "tail\xf0\x9f", // first half of a 4-byte emoji, cut by max_tokens
    };
    for (cases) |bad| {
        const a1 = try jsonEscape(a, bad);
        defer a.free(a1);
        try std.testing.expect(std.unicode.utf8ValidateSlice(a1));

        const a2 = try responses_mod.jsonEscape(a, bad);
        defer a.free(a2);
        try std.testing.expect(std.unicode.utf8ValidateSlice(a2));

        var buf = std.ArrayList(u8).empty;
        defer buf.deinit(a);
        try chat_mod.appendJsonString(a, &buf, bad);
        try std.testing.expect(std.unicode.utf8ValidateSlice(buf.items));
    }
    // Valid input is passed through byte for byte.
    const good = try jsonEscape(a, "héllo \u{1F600}");
    defer a.free(good);
    try std.testing.expectEqualStrings("\"héllo \u{1F600}\"", good);
}

test "debug body log bounds BINARY but never truncates text (multipart PNG class)" {
    // `/v1/images/edits` is the first binary body we serve. `logHttpBody` dumped
    // whatever it got verbatim, so at --log-level debug one image upload wrote
    // raw PNG bytes into the log — and a body near the 64 MB request cap blew
    // straight through the 32 MB rotation, taking the post-mortem file with it.
    //
    // Text bodies must stay WHOLE: reading a full request body out of the debug
    // log is the documented way to reproduce a tool-calling bug.
    try testing.expect(bodyIsText("{\"model\":\"x\",\"prompt\":\"hi\"}"));
    try testing.expect(bodyIsText("multi\nline\twith\r\nwhitespace"));
    try testing.expect(bodyIsText("utf8 stays text: é ✓ 🎉")); // multibyte, not control
    try testing.expect(!bodyIsText("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"));
    // The real shape: text multipart framing wrapped around a binary payload.
    try testing.expect(!bodyIsText("--B\r\nContent-Disposition: form-data; name=\"image\"\r\n\r\n\x89PNG\x00\x01\r\n--B--"));

    // A binary preview is bounded AND strictly printable ASCII.
    var buf: [64]u8 = undefined;
    const png = "\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\xffbinary tail";
    const p = bodyPreview(&buf, png);
    try testing.expect(p.len <= buf.len);
    for (p) |c| try testing.expect(c == '\n' or c == '\r' or c == '\t' or (c >= 0x20 and c < 0x7f));

    // Larger than the sink → truncated, never streamed.
    const big = try testing.allocator.alloc(u8, 1024 * 1024);
    defer testing.allocator.free(big);
    @memset(big, 0);
    try testing.expectEqual(@as(usize, buf.len), bodyPreview(&buf, big).len);
    // The cap is the SMALLER of the caller's buffer and the byte limit, so a
    // generous buffer can't reintroduce the megabyte dump.
    var huge: [BODY_LOG_LIMIT * 2]u8 = undefined;
    try testing.expectEqual(@as(usize, BODY_LOG_LIMIT), bodyPreview(&huge, big).len);
}

test "jsonEscape basic string" {
    const allocator = testing.allocator;
    const result = try jsonEscape(allocator, "hello");
    defer allocator.free(result);
    try testing.expectEqualStrings("\"hello\"", result);
}

test "jsonEscape special characters" {
    const allocator = testing.allocator;
    const result = try jsonEscape(allocator, "line1\nline2\t\"quoted\"\\back");
    defer allocator.free(result);
    try testing.expectEqualStrings("\"line1\\nline2\\t\\\"quoted\\\"\\\\back\"", result);
}

test "jsonEscape control characters" {
    const allocator = testing.allocator;
    const input = &[_]u8{ 0x01, 0x02 };
    const result = try jsonEscape(allocator, input);
    defer allocator.free(result);
    try testing.expectEqualStrings("\"\\u0001\\u0002\"", result);
}

test "jsonEscape empty string" {
    const allocator = testing.allocator;
    const result = try jsonEscape(allocator, "");
    defer allocator.free(result);
    try testing.expectEqualStrings("\"\"", result);
}

test "resolveBind: the port is the flag's, else the settings file's, else 12345" {
    try std.testing.expectEqual(@as(u16, 8080), resolveBind(null, null, 8080).port);
    try std.testing.expectEqual(@as(u16, 23817), resolveBind(null, 23817, 8080).port);
    try std.testing.expectEqual(server_state.default_port, resolveBind(null, null, server_state.default_port).port);
}

test "resolveBind: no flags bind 127.0.0.1:12345" {
    const d = resolveBind(null, null, server_state.default_port);
    try std.testing.expectEqualStrings("127.0.0.1", d.host);
    try std.testing.expectEqual(@as(u16, 12345), d.port);
    const e = resolveBind("0.0.0.0", 23817, server_state.default_port);
    try std.testing.expectEqualStrings("0.0.0.0", e.host);
    try std.testing.expectEqual(@as(u16, 23817), e.port);
    const p = resolveBind(null, 23817, server_state.default_port);
    try std.testing.expectEqualStrings("127.0.0.1", p.host);
    try std.testing.expectEqual(@as(u16, 23817), p.port);
    const h = resolveBind("192.168.7.9", null, server_state.default_port);
    try std.testing.expectEqualStrings("192.168.7.9", h.host);
    try std.testing.expectEqual(@as(u16, 12345), h.port);
}

test "startupRefusal: AddressInUse refuses naming the port" {
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("port 12345 is already in use", startupRefusal(error.AddressInUse, "127.0.0.1", 12345, &buf).?);
    try std.testing.expectEqualStrings("port 23817 is already in use", startupRefusal(error.AddressInUse, "127.0.0.1", 23817, &buf).?);
    try std.testing.expect(startupRefusal(error.AccessDenied, "127.0.0.1", 23817, &buf) == null);
    try std.testing.expect(startupRefusal(error.SystemResources, "127.0.0.1", 23817, &buf) == null);
}

test "bindAddress: localhost is loopback; wildcards stay" {
    const l = try bindAddress("localhost", 12345);
    try std.testing.expect(std.mem.eql(u8, &l.ip4.bytes, &.{ 127, 0, 0, 1 }));
    try std.testing.expectEqual(@as(u16, 12345), l.ip4.port);
    const e = try bindAddress("127.0.0.1", 23817);
    try std.testing.expect(std.mem.eql(u8, &e.ip4.bytes, &.{ 127, 0, 0, 1 }));
    try std.testing.expectEqual(@as(u16, 23817), e.ip4.port);
    const w = try bindAddress("0.0.0.0", 12345);
    try std.testing.expect(std.mem.eql(u8, &w.ip4.bytes, &.{ 0, 0, 0, 0 }));
    try std.testing.expectEqual(@as(u16, 12345), w.ip4.port);
}

test "bindAddress: garbage is refused by name" {
    try std.testing.expectError(error.InvalidHost, bindAddress("example.com", 80));
    try std.testing.expectError(error.InvalidHost, bindAddress("999.1.2.3", 80));
    try std.testing.expectError(error.InvalidHost, bindAddress("1.2.3", 80));
    var buf: [512]u8 = undefined;
    try std.testing.expectEqualStrings("invalid --host 'example.com'", startupRefusal(error.InvalidHost, "example.com", 80, &buf).?);
    try std.testing.expectEqualStrings("invalid --host '999.1.2.3'", startupRefusal(error.InvalidHost, "999.1.2.3", 80, &buf).?);
}
