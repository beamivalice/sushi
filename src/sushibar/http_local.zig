//! Minimal non-blocking POSIX HTTP/1.1 client for `sushibar`'s local/LAN server requests.
//! Avoids spawning `std.Io.Select` tasks or `std.http.Client` worker-pool threads, enforces a strict wall-clock
//! deadline via `poll(2)`, and distinguishes `ECONNREFUSED` from timeouts/errors so offline polling skips extra
//! probes.
const std = @import("std");

pub const Method = enum { GET, POST };

pub const Reply = struct {
    status: u16,
    body: []const u8,
};

pub const Outcome = union(enum) {
    reply: Reply,
    /// Loopback/host actively refused the TCP connection: nothing listens on that address and port.
    conn_refused,
    /// Timed out, reset, malformed response, or OS error.
    failed,
};

pub const ParsedUrl = struct {
    host: []const u8,
    port: u16,
    target: []const u8,
    ipv6: bool,
};

const max_body_bytes = 8 * 1024 * 1024;
const max_header_bytes = 32 * 1024;
const so_nosigpipe: c_int = 0x1022;

extern "c" fn inet_pton(af: c_int, src: [*:0]const u8, dst: *anyopaque) c_int;

pub fn parseUrl(url: []const u8) ?ParsedUrl {
    const prefix = "http://";
    if (!std.mem.startsWith(u8, url, prefix)) return null;
    const rest = url[prefix.len..];
    const slash = std.mem.indexOfScalar(u8, rest, '/');
    const authority = if (slash) |i| rest[0..i] else rest;
    const target = if (slash) |i| rest[i..] else "/";
    if (authority.len == 0) return null;

    if (authority[0] == '[') {
        const close = std.mem.indexOfScalar(u8, authority, ']') orelse return null;
        const host = authority[1..close];
        if (host.len == 0) return null;
        const after = authority[close + 1 ..];
        const port: u16 = if (after.len == 0) 80 else blk: {
            if (after[0] != ':') return null;
            const parsed = std.fmt.parseInt(u16, after[1..], 10) catch return null;
            if (parsed == 0) return null;
            break :blk parsed;
        };
        return .{ .host = host, .port = port, .target = target, .ipv6 = true };
    }

    const colon = std.mem.lastIndexOfScalar(u8, authority, ':');
    const host = if (colon) |c| authority[0..c] else authority;
    if (host.len == 0) return null;
    const port: u16 = if (colon) |c| blk: {
        const parsed = std.fmt.parseInt(u16, authority[c + 1 ..], 10) catch return null;
        if (parsed == 0) return null;
        break :blk parsed;
    } else 80;
    return .{ .host = host, .port = port, .target = target, .ipv6 = false };
}

fn monoMs() i64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(i64, @intCast(ts.sec)) * 1000 + @divTrunc(@as(i64, @intCast(ts.nsec)), 1_000_000);
}

fn remainingMs(deadline_ms: i64) c_int {
    const rem = deadline_ms - monoMs();
    if (rem <= 0) return 0;
    return @intCast(@min(rem, std.math.maxInt(c_int)));
}

fn waitFd(fd: c_int, events: i16, deadline_ms: i64) bool {
    while (true) {
        const rem = remainingMs(deadline_ms);
        if (rem <= 0) return false;
        var pfd = [_]std.c.pollfd{.{ .fd = fd, .events = events, .revents = 0 }};
        const rc = std.c.poll(&pfd, 1, rem);
        if (rc > 0) return true;
        if (rc == 0) return false;
        if (std.posix.errno(rc) != .INTR) return false;
    }
}

fn setNonBlockAndCloexec(fd: c_int) bool {
    const flags = std.c.fcntl(fd, std.c.F.GETFL, @as(c_int, 0));
    if (flags < 0) return false;
    const nonblock: c_int = @bitCast(std.c.O{ .NONBLOCK = true });
    if (std.c.fcntl(fd, std.c.F.SETFL, flags | nonblock) < 0) return false;
    _ = std.c.fcntl(fd, std.c.F.SETFD, @as(c_int, std.c.FD_CLOEXEC));
    var one: c_int = 1;
    _ = std.c.setsockopt(fd, std.c.SOL.SOCKET, so_nosigpipe, &one, @sizeOf(c_int));
    return true;
}

fn connectAddr(u: ParsedUrl, deadline_ms: i64) union(enum) { fd: c_int, conn_refused, failed } {
    var host_z: [128:0]u8 = undefined;
    if (u.host.len >= host_z.len) return .failed;
    @memcpy(host_z[0..u.host.len], u.host);
    host_z[u.host.len] = 0;

    var a4: std.c.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, u.port), .addr = 0 };
    var a6: std.c.sockaddr.in6 = .{ .port = std.mem.nativeToBig(u16, u.port), .addr = @splat(0), .flowinfo = 0, .scope_id = 0 };
    const af: c_uint = if (u.ipv6) std.c.AF.INET6 else std.c.AF.INET;
    if (u.ipv6) {
        if (inet_pton(@intCast(af), &host_z, &a6.addr) != 1) return .failed;
    } else {
        if (std.mem.eql(u8, u.host, "localhost")) {
            a4.addr = std.mem.nativeToBig(u32, 0x7f000001);
        } else if (inet_pton(@intCast(af), &host_z, &a4.addr) != 1) {
            return .failed;
        }
    }

    const fd = std.c.socket(af, std.c.SOCK.STREAM, 0);
    if (fd < 0) return .failed;
    if (!setNonBlockAndCloexec(fd)) {
        _ = std.c.close(fd);
        return .failed;
    }

    const sa: *const std.c.sockaddr = if (u.ipv6) @ptrCast(&a6) else @ptrCast(&a4);
    const sa_len: std.c.socklen_t = if (u.ipv6) @sizeOf(@TypeOf(a6)) else @sizeOf(@TypeOf(a4));
    const rc = std.c.connect(fd, sa, sa_len);
    if (rc == 0) return .{ .fd = fd };

    const err = std.posix.errno(rc);
    if (err == .CONNREFUSED) {
        _ = std.c.close(fd);
        return .conn_refused;
    }
    if (err != .INPROGRESS) {
        _ = std.c.close(fd);
        return .failed;
    }

    if (!waitFd(fd, std.c.POLL.OUT, deadline_ms)) {
        _ = std.c.close(fd);
        return .failed;
    }
    var so_err: c_int = 0;
    var opt_len: std.c.socklen_t = @sizeOf(c_int);
    if (std.c.getsockopt(fd, std.c.SOL.SOCKET, std.c.SO.ERROR, &so_err, &opt_len) != 0) {
        _ = std.c.close(fd);
        return .failed;
    }
    if (so_err != 0) {
        _ = std.c.close(fd);
        const e: std.c.E = @enumFromInt(so_err);
        return if (e == .CONNREFUSED) .conn_refused else .failed;
    }
    return .{ .fd = fd };
}

fn writeAll(fd: c_int, data: []const u8, deadline_ms: i64) bool {
    var off: usize = 0;
    while (off < data.len) {
        const n = std.c.write(fd, data[off..].ptr, data.len - off);
        if (n > 0) {
            off += @intCast(n);
            continue;
        }
        if (n < 0) {
            const e = std.posix.errno(n);
            if (e == .INTR) continue;
            if (e == .AGAIN) {
                if (!waitFd(fd, std.c.POLL.OUT, deadline_ms)) return false;
                continue;
            }
        }
        return false;
    }
    return true;
}

fn readChunk(fd: c_int, buf: []u8, deadline_ms: i64) ?usize {
    while (true) {
        const n = std.c.read(fd, buf.ptr, buf.len);
        if (n >= 0) return @intCast(n);
        const e = std.posix.errno(n);
        if (e == .INTR) continue;
        if (e == .AGAIN) {
            if (!waitFd(fd, std.c.POLL.IN, deadline_ms)) return null;
            continue;
        }
        return null;
    }
}

fn chunkedComplete(raw: []const u8) bool {
    var i: usize = 0;
    while (i < raw.len) {
        const crlf = std.mem.indexOfPos(u8, raw, i, "\r\n") orelse return false;
        const size_line = raw[i..crlf];
        const semi = std.mem.indexOfScalar(u8, size_line, ';') orelse size_line.len;
        const trimmed = std.mem.trim(u8, size_line[0..semi], " \t");
        const chunk_len = std.fmt.parseInt(usize, trimmed, 16) catch return false;
        i = crlf + 2;
        if (chunk_len == 0) {
            // Terminating chunk ends with empty trailer line ("\r\n") or trailers followed by "\r\n\r\n".
            if (i + 2 <= raw.len and raw[i] == '\r' and raw[i + 1] == '\n') return true;
            return std.mem.indexOfPos(u8, raw, i, "\r\n\r\n") != null;
        }
        if (chunk_len > max_body_bytes or i + chunk_len + 2 > raw.len) return false;
        if (raw[i + chunk_len] != '\r' or raw[i + chunk_len + 1] != '\n') return false;
        i += chunk_len + 2;
    }
    return false;
}

pub fn decodeChunked(a: std.mem.Allocator, raw: []const u8) ?[]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < raw.len) {
        const crlf = std.mem.indexOfPos(u8, raw, i, "\r\n") orelse return null;
        const size_line = raw[i..crlf];
        const semi = std.mem.indexOfScalar(u8, size_line, ';') orelse size_line.len;
        const trimmed = std.mem.trim(u8, size_line[0..semi], " \t");
        const chunk_len = std.fmt.parseInt(usize, trimmed, 16) catch return null;
        i = crlf + 2;
        if (chunk_len == 0) return out.items;
        if (chunk_len > max_body_bytes - out.items.len) return null;
        if (i + chunk_len + 2 > raw.len) return null;
        out.appendSlice(a, raw[i .. i + chunk_len]) catch return null;
        if (raw[i + chunk_len] != '\r' or raw[i + chunk_len + 1] != '\n') return null;
        i += chunk_len + 2;
    }
    return null;
}

fn parseStatus(line: []const u8) ?u16 {
    if (!std.mem.startsWith(u8, line, "HTTP/")) return null;
    const sp = std.mem.indexOfScalar(u8, line, ' ') orelse return null;
    const after = std.mem.trimStart(u8, line[sp + 1 ..], " ");
    if (after.len < 3) return null;
    return std.fmt.parseInt(u16, after[0..3], 10) catch null;
}

fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0 or needle.len > haystack.len) return false;
    for (0..haystack.len - needle.len + 1) |i| {
        if (std.ascii.eqlIgnoreCase(haystack[i .. i + needle.len], needle)) return true;
    }
    return false;
}

/// Sends a single HTTP/1.1 request to `url` with `timeout_ms` wall-clock budget.
/// Memory for the returned body is allocated from `a` (typically a per-poll arena).
pub fn fetch(
    a: std.mem.Allocator,
    method: Method,
    url: []const u8,
    origin: ?[]const u8,
    auth: []const u8,
    payload: []const u8,
    timeout_ms: i64,
) Outcome {
    const u = parseUrl(url) orelse return .failed;
    const deadline_ms = monoMs() + @max(timeout_ms, 1);

    const fd = switch (connectAddr(u, deadline_ms)) {
        .fd => |f| f,
        .conn_refused => return .conn_refused,
        .failed => return .failed,
    };
    defer _ = std.c.close(fd);

    var req: std.Io.Writer.Allocating = .init(a);
    defer req.deinit();
    const m_str = switch (method) {
        .GET => "GET",
        .POST => "POST",
    };
    const w = &req.writer;
    w.print("{s} {s} HTTP/1.1\r\n", .{ m_str, u.target }) catch return .failed;
    if (u.ipv6) {
        w.print("Host: [{s}]:{d}\r\n", .{ u.host, u.port }) catch return .failed;
    } else {
        w.print("Host: {s}:{d}\r\n", .{ u.host, u.port }) catch return .failed;
    }
    w.writeAll("Connection: close\r\n") catch return .failed;
    if (origin) |o| w.print("Origin: {s}\r\n", .{o}) catch return .failed;
    if (auth.len > 0) w.print("Authorization: {s}\r\n", .{auth}) catch return .failed;
    if (method == .POST) {
        if (payload.len > 0) w.writeAll("Content-Type: application/json\r\n") catch return .failed;
        w.print("Content-Length: {d}\r\n", .{payload.len}) catch return .failed;
    }
    w.writeAll("\r\n") catch return .failed;
    if (method == .POST and payload.len > 0) w.writeAll(payload) catch return .failed;

    if (!writeAll(fd, req.written(), deadline_ms)) return .failed;

    var raw: std.ArrayList(u8) = .empty;
    var tmp: [8192]u8 = undefined;
    var header_end: ?usize = null;
    var content_length: ?usize = null;
    var chunked = false;
    var status: u16 = 0;

    while (true) {
        const n = readChunk(fd, &tmp, deadline_ms) orelse return .failed;
        if (n == 0) break;
        if (raw.items.len + n > max_header_bytes + max_body_bytes) return .failed;
        raw.appendSlice(a, tmp[0..n]) catch return .failed;

        if (header_end == null) {
            if (std.mem.indexOf(u8, raw.items, "\r\n\r\n")) |idx| {
                header_end = idx + 4;
                const head = raw.items[0..idx];
                var lines = std.mem.splitSequence(u8, head, "\r\n");
                status = parseStatus(lines.next() orelse return .failed) orelse return .failed;
                while (lines.next()) |line| {
                    const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
                    const name = std.mem.trim(u8, line[0..colon], " \t");
                    const val = std.mem.trim(u8, line[colon + 1 ..], " \t");
                    if (std.ascii.eqlIgnoreCase(name, "content-length")) {
                        content_length = std.fmt.parseInt(usize, val, 10) catch return .failed;
                        if (content_length.? > max_body_bytes) return .failed;
                    } else if (std.ascii.eqlIgnoreCase(name, "transfer-encoding")) {
                        if (containsIgnoreCase(val, "chunked")) chunked = true;
                    }
                }
            } else if (raw.items.len > max_header_bytes) {
                return .failed;
            }
        }
        if (header_end) |he| {
            const body_have = raw.items.len - he;
            if (chunked) {
                if (chunkedComplete(raw.items[he..])) break;
            } else if (content_length) |cl| {
                if (body_have >= cl) break;
            }
        }
    }

    const he = header_end orelse return .failed;
    const raw_body = raw.items[he..];
    if (chunked) {
        const decoded = decodeChunked(a, raw_body) orelse return .failed;
        return .{ .reply = .{ .status = status, .body = decoded } };
    }
    if (content_length) |cl| {
        if (raw_body.len < cl) return .failed;
        return .{ .reply = .{ .status = status, .body = raw_body[0..cl] } };
    }
    return .{ .reply = .{ .status = status, .body = raw_body } };
}

test "http_local: parseUrl handles IPv4, IPv6, and rejects non-http" {
    const v4 = parseUrl("http://127.0.0.1:12345/metrics.json?since=1").?;
    try std.testing.expectEqualStrings("127.0.0.1", v4.host);
    try std.testing.expectEqual(@as(u16, 12345), v4.port);
    try std.testing.expectEqualStrings("/metrics.json?since=1", v4.target);
    try std.testing.expect(!v4.ipv6);

    const v6 = parseUrl("http://[::1]:8080/v1/models").?;
    try std.testing.expectEqualStrings("::1", v6.host);
    try std.testing.expectEqual(@as(u16, 8080), v6.port);
    try std.testing.expectEqualStrings("/v1/models", v6.target);
    try std.testing.expect(v6.ipv6);

    try std.testing.expect(parseUrl("https://127.0.0.1:8080/") == null);
    try std.testing.expect(parseUrl("http://127.0.0.1:0/") == null);
    try std.testing.expect(parseUrl("http://[::1]:0/") == null);
    try std.testing.expect(parseUrl("http://127.0.0.1:abc/") == null);
    try std.testing.expect(parseUrl("http://[::1/") == null);
}

test "http_local: decodeChunked joins chunks and rejects truncation" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expect(chunkedComplete("5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n"));
    try std.testing.expect(!chunkedComplete("5\r\nhello\r\n"));
    try std.testing.expectEqualStrings("hello world", decodeChunked(a, "5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n").?);
    try std.testing.expectEqualStrings("", decodeChunked(a, "0\r\n\r\n").?);
    try std.testing.expect(decodeChunked(a, "5\r\nhi") == null);
}

test "http_local: live loopback GET, POST, conn_refused, and timeout" {
    // Closed port returns conn_refused cleanly.
    const probe = std.c.socket(std.c.AF.INET, std.c.SOCK.STREAM, 0);
    try std.testing.expect(probe >= 0);
    var addr: std.c.sockaddr.in = .{ .port = 0, .addr = std.mem.nativeToBig(u32, 0x7f000001) };
    var alen: std.c.socklen_t = @sizeOf(@TypeOf(addr));
    try std.testing.expectEqual(@as(c_int, 0), std.c.bind(probe, @ptrCast(&addr), alen));
    try std.testing.expectEqual(@as(c_int, 0), std.c.getsockname(probe, @ptrCast(&addr), &alen));
    const closed_port = std.mem.bigToNative(u16, addr.port);
    _ = std.c.close(probe);

    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var ub: [64]u8 = undefined;
    const closed_url = try std.fmt.bufPrint(&ub, "http://127.0.0.1:{d}/health", .{closed_port});
    try std.testing.expectEqual(Outcome.conn_refused, fetch(arena.allocator(), .GET, closed_url, null, "", "", 1000));

    // Live loopback listener serving one Content-Length GET and one chunked POST.
    const srv = std.c.socket(std.c.AF.INET, std.c.SOCK.STREAM, 0);
    try std.testing.expect(srv >= 0);
    defer _ = std.c.close(srv);
    var saddr: std.c.sockaddr.in = .{ .port = 0, .addr = std.mem.nativeToBig(u32, 0x7f000001) };
    var salen: std.c.socklen_t = @sizeOf(@TypeOf(saddr));
    try std.testing.expectEqual(@as(c_int, 0), std.c.bind(srv, @ptrCast(&saddr), salen));
    try std.testing.expectEqual(@as(c_int, 0), std.c.listen(srv, 4));
    try std.testing.expectEqual(@as(c_int, 0), std.c.getsockname(srv, @ptrCast(&saddr), &salen));
    const port = std.mem.bigToNative(u16, saddr.port);

    const SrvCtx = struct {
        fd: c_int,
        fn run(self: @This()) void {
            var buf: [2048]u8 = undefined;
            // 1) GET -> Content-Length
            const c1 = std.c.accept(self.fd, null, null);
            if (c1 >= 0) {
                _ = std.c.read(c1, &buf, buf.len);
                const resp = "HTTP/1.1 200 OK\r\nContent-Length: 13\r\n\r\n{\"status\":ok}";
                _ = std.c.write(c1, resp.ptr, resp.len);
                _ = std.c.close(c1);
            }
            // 2) POST -> Chunked
            const c2 = std.c.accept(self.fd, null, null);
            if (c2 >= 0) {
                const n = std.c.read(c2, &buf, buf.len);
                const saw_auth = n > 0 and std.mem.indexOf(u8, buf[0..@intCast(n)], "Authorization: Bearer tok") != null;
                const resp = if (saw_auth)
                    "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2\r\nok\r\n0\r\n\r\n"
                else
                    "HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\n\r\n";
                _ = std.c.write(c2, resp.ptr, resp.len);
                _ = std.c.close(c2);
            }
            // 3) Hung accept -> never replies
            const c3 = std.c.accept(self.fd, null, null);
            if (c3 >= 0) {
                var pfd = [_]std.c.pollfd{.{ .fd = c3, .events = std.c.POLL.IN, .revents = 0 }};
                _ = std.c.poll(&pfd, 1, 400);
                _ = std.c.close(c3);
            }
        }
    };
    const th = try std.Thread.spawn(.{}, SrvCtx.run, .{SrvCtx{ .fd = srv }});
    defer th.join();

    const url = try std.fmt.bufPrint(&ub, "http://127.0.0.1:{d}/test", .{port});
    const r1 = fetch(arena.allocator(), .GET, url, null, "", "", 2000);
    try std.testing.expectEqual(@as(u16, 200), r1.reply.status);
    try std.testing.expectEqualStrings("{\"status\":ok}", r1.reply.body);

    const r2 = fetch(arena.allocator(), .POST, url, "http://127.0.0.1", "Bearer tok", "{\"a\":1}", 2000);
    try std.testing.expectEqual(@as(u16, 200), r2.reply.status);
    try std.testing.expectEqualStrings("ok", r2.reply.body);

    const r3 = fetch(arena.allocator(), .GET, url, null, "", "", 120);
    try std.testing.expectEqual(Outcome.failed, r3);
}
