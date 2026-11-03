//! `sushi repack --pack-revision`: stamps a pack with a revision id (docs/pack-format.md#pack-revision-writer).
//! A shard is replaced only after its rewrite was re-read and its data section hashed equal; `config.json` changes last,
//! so a run cut short leaves a pack the loader refuses and a rerun finishes.
const std = @import("std");
const repack = @import("repack.zig");
const expert_io = @import("../expert_io.zig");
const disk_placement = @import("../disk_placement.zig");
const log = @import("../log.zig");

const tmp_suffix = ".stamp-tmp";
const shard_suffix = ".safetensors";
const key = "pack_revision";
const chunk_bytes: usize = 8 << 20;
const spare_bytes: u64 = 1 << 30;
const max_id_len = 64;

pub const Options = struct {
    id: []const u8,
    /// Names or `*` / `?` globs; empty selects every shard.
    shards: []const []const u8 = &.{},
    /// Shard file names left without a stamp on purpose; a symlinked shard must be one.
    unstamped: []const []const u8 = &.{},
    /// Test seams: the free bytes of the pack's volume; count this process's own open files as users; stop after this
    /// many shards were replaced; flip a byte of each rewrite before it is verified.
    free_bytes: ?u64 = null,
    count_self: bool = false,
    fail_after_shards: ?usize = null,
    corrupt_tmp: bool = false,
};

/// What a run did, and the one line that names what it refused.
pub const Report = struct {
    stamped: usize = 0,
    skipped: usize = 0,
    buf: [320]u8 = undefined,
    len: usize = 0,

    fn say(self: *Report, comptime fmt: []const u8, args: anytype) void {
        const out = std.fmt.bufPrint(&self.buf, fmt, args) catch self.buf[0..];
        self.len = out.len;
    }

    pub fn text(self: *const Report) []const u8 {
        return self.buf[0..self.len];
    }
};

pub fn validId(id: []const u8) bool {
    if (id.len == 0 or id.len > max_id_len) return false;
    for (id) |c| if (!std.ascii.isAlphanumeric(c) and c != '.' and c != '_' and c != '-' and c != '+') return false;
    return true;
}

/// `*` matches any run of characters and `?` one; every other character matches itself.
pub fn globMatch(pattern: []const u8, name: []const u8) bool {
    if (pattern.len == 0) return false;
    var p: usize = 0;
    var n: usize = 0;
    var star: ?usize = null;
    var mark: usize = 0;
    while (n < name.len) {
        if (p < pattern.len and pattern[p] == '*') {
            star = p;
            mark = n;
            p += 1;
        } else if (p < pattern.len and (pattern[p] == '?' or pattern[p] == name[n])) {
            p += 1;
            n += 1;
        } else if (star) |s| {
            p = s + 1;
            mark += 1;
            n = mark;
        } else return false;
    }
    while (p < pattern.len and pattern[p] == '*') p += 1;
    return p == pattern.len;
}

fn selects(patterns: []const []const u8, name: []const u8) bool {
    if (patterns.len == 0) return true;
    for (patterns) |p| if (globMatch(p, name)) return true;
    return false;
}

fn contains(names: []const []const u8, name: []const u8) bool {
    for (names) |n| if (std.mem.eql(u8, n, name)) return true;
    return false;
}

// ── Headers ───────────────────────────────────────────────────────────────

/// The header of a shard with `__metadata__["pack_revision"] = id`: a safetensors length word and the JSON, every other
/// field as it was. The data section keeps its alignment: a page-aligned one stays page-aligned (the pair-major tiles
/// are read from page-aligned offsets), any other one stays 8-byte aligned.
pub fn stampedHead(a: std.mem.Allocator, header_json: []const u8, id: []const u8, old_data_start: u64) ![]u8 {
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, a, header_json, .{}) catch return error.InvalidSafetensorsHeader;
    if (parsed != .object) return error.InvalidSafetensorsHeader;
    var json: std.ArrayList(u8) = .empty;
    try json.appendSlice(a, "{\"__metadata__\":{");
    var seen = false;
    if (parsed.object.get("__metadata__")) |meta| {
        if (meta != .object) return error.InvalidSafetensorsHeader;
        for (meta.object.keys(), meta.object.values()) |k, v| {
            if (v != .string) return error.InvalidSafetensorsHeader;
            try repack.appendJsonString(a, &json, k);
            try json.append(a, ':');
            try repack.appendJsonString(a, &json, if (std.mem.eql(u8, k, key)) id else v.string);
            try json.append(a, ',');
            seen = seen or std.mem.eql(u8, k, key);
        }
    }
    if (!seen) {
        try repack.appendJsonString(a, &json, key);
        try json.append(a, ':');
        try repack.appendJsonString(a, &json, id);
    } else _ = json.pop();
    try json.append(a, '}');
    for (parsed.object.keys(), parsed.object.values()) |k, v| {
        if (std.mem.eql(u8, k, "__metadata__")) continue;
        try json.append(a, ',');
        try repack.appendJsonString(a, &json, k);
        try json.append(a, ':');
        try json.appendSlice(a, try std.json.Stringify.valueAlloc(a, v, .{}));
    }
    try json.append(a, '}');
    const alignment: usize = if (old_data_start % repack.page == 0) repack.page else 8;
    while ((8 + json.items.len) % alignment != 0) try json.append(a, ' ');
    const head = try a.alloc(u8, 8 + json.items.len);
    std.mem.writeInt(u64, head[0..8], json.items.len, .little);
    @memcpy(head[8..], json.items);
    return head;
}

/// The stamp a header carries, null when it has none.
fn headerStamp(header: std.json.Value) ?[]const u8 {
    if (header != .object) return null;
    const meta = header.object.get("__metadata__") orelse return null;
    if (meta != .object) return null;
    const v = meta.object.get(key) orelse return null;
    return if (v == .string) v.string else null;
}

/// Does `new_json` equal `old_json` plus the one stamp?
fn equalExceptStamp(a: std.mem.Allocator, old_json: []const u8, new_json: []const u8, id: []const u8) !bool {
    var old = std.json.parseFromSliceLeaky(std.json.Value, a, old_json, .{}) catch return false;
    var new = std.json.parseFromSliceLeaky(std.json.Value, a, new_json, .{}) catch return false;
    if (old != .object or new != .object) return false;
    const got = headerStamp(new) orelse return false;
    if (!std.mem.eql(u8, got, id)) return false;
    var new_meta = &new.object.getPtr("__metadata__").?.object;
    _ = new_meta.orderedRemove(key);
    if (old.object.getPtr("__metadata__")) |m| {
        _ = m.object.orderedRemove(key);
    } else {
        if (new_meta.count() != 0) return false;
        _ = new.object.orderedRemove("__metadata__");
    }
    return std.mem.eql(u8, try std.json.Stringify.valueAlloc(a, old, .{}), try std.json.Stringify.valueAlloc(a, new, .{}));
}

const Header = struct {
    json: []u8,
    value: std.json.Value,
    data_start: u64,

    fn read(a: std.mem.Allocator, fd: std.c.fd_t) !Header {
        var len_bytes: [8]u8 = undefined;
        try expert_io.readExact(fd, &len_bytes, 0);
        const len = std.mem.readInt(u64, &len_bytes, .little);
        if (len == 0 or len > 128 * 1024 * 1024) return error.InvalidSafetensorsHeader;
        const json = try a.alloc(u8, @intCast(len));
        try expert_io.readExact(fd, json, 8);
        const value = std.json.parseFromSliceLeaky(std.json.Value, a, json, .{}) catch return error.InvalidSafetensorsHeader;
        if (value != .object) return error.InvalidSafetensorsHeader;
        return .{ .json = json, .value = value, .data_start = 8 + len };
    }
};

// ── One shard ─────────────────────────────────────────────────────────────

const Sha = std.crypto.hash.sha2.Sha256;

fn hashRange(fd: std.c.fd_t, buf: []u8, start: u64, len: u64, copy_to: ?std.c.fd_t, dst_at: u64) ![Sha.digest_length]u8 {
    var h = Sha.init(.{});
    var done: u64 = 0;
    while (done < len) {
        const n: usize = @intCast(@min(buf.len, len - done));
        try expert_io.readExact(fd, buf[0..n], start + done);
        h.update(buf[0..n]);
        if (copy_to) |out| try repack.writeExact(out, buf[0..n], dst_at + done);
        done += n;
    }
    var digest: [Sha.digest_length]u8 = undefined;
    h.final(&digest);
    return digest;
}

/// `name` rewritten with the stamp through `<name>.stamp-tmp`, verified, then renamed over the original.
fn stampShard(gpa: std.mem.Allocator, dir: []const u8, name: []const u8, id: []const u8, opts: Options) !void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const final = try repack.pathZ(a, dir, name);
    const tmp = try std.fmt.allocPrintSentinel(a, "{s}{s}", .{ final, tmp_suffix }, 0);
    repack.unlinkQuiet(tmp);

    const src = try repack.openFd(final, false);
    defer _ = std.c.close(src);
    const header = try Header.read(a, src);
    const size = try repack.fileSize(src);
    if (header.data_start > size) return error.InvalidSafetensorsHeader;
    const data_len = size - header.data_start;
    const head = try stampedHead(a, header.json, id, header.data_start);
    const buf = try a.alloc(u8, chunk_bytes);

    errdefer repack.unlinkQuiet(tmp);
    var want: [Sha.digest_length]u8 = undefined;
    {
        const dst = try repack.openFd(tmp, true);
        defer _ = std.c.close(dst);
        var st: std.c.Stat = undefined;
        if (std.c.fstat(src, &st) == 0) _ = std.c.fchmod(dst, st.mode & 0o777);
        try repack.writeExact(dst, head, 0);
        want = try hashRange(src, buf, header.data_start, data_len, dst, head.len);
        if (opts.corrupt_tmp and data_len > 0) {
            var byte: [1]u8 = undefined;
            try expert_io.readExact(dst, &byte, head.len + data_len / 2);
            byte[0] ^= 0xff;
            try repack.writeExact(dst, &byte, head.len + data_len / 2);
        }
        try repack.fullSync(dst);
    }

    const check = try repack.openFd(tmp, false);
    defer _ = std.c.close(check);
    if (try repack.fileSize(check) != head.len + data_len or try repack.fileSize(src) != size) return error.RepackVerifyFailed;
    const back = try Header.read(a, check);
    if (back.data_start != head.len or !std.mem.eql(u8, back.json, head[8..]) or !try equalExceptStamp(a, header.json, back.json, id)) return error.RepackVerifyFailed;
    const got = try hashRange(check, buf, head.len, data_len, null, 0);
    if (!std.mem.eql(u8, &got, &want)) return error.RepackVerifyFailed;

    if (repack.inUse(final, opts.count_self)) return error.RepackPackInUse;
    if (std.c.rename(tmp.ptr, final.ptr) != 0) return error.RepackIoFailed;
    try repack.syncDir(a, dir);
}

// ── config.json ───────────────────────────────────────────────────────────

fn skipWs(raw: []const u8, from: usize) usize {
    var i = from;
    while (i < raw.len and std.ascii.isWhitespace(raw[i])) i += 1;
    return i;
}

fn skipString(raw: []const u8, from: usize) !usize {
    var i = from + 1;
    while (i < raw.len) : (i += 1) switch (raw[i]) {
        '\\' => i += 1,
        '"' => return i + 1,
        else => {},
    };
    return error.InvalidConfig;
}

fn skipValue(raw: []const u8, from: usize) !usize {
    if (from >= raw.len) return error.InvalidConfig;
    if (raw[from] == '"') return skipString(raw, from);
    if (raw[from] != '{' and raw[from] != '[') {
        var i = from;
        while (i < raw.len and !std.ascii.isWhitespace(raw[i]) and raw[i] != ',' and raw[i] != '}' and raw[i] != ']') i += 1;
        return i;
    }
    var depth: usize = 0;
    var i = from;
    while (i < raw.len) : (i += 1) switch (raw[i]) {
        '"' => i = (try skipString(raw, i)) - 1,
        '{', '[' => depth += 1,
        '}', ']' => {
            depth -= 1;
            if (depth == 0) return i + 1;
        },
        else => {},
    };
    return error.InvalidConfig;
}

/// `config.json` with a top-level `"pack_revision": {"id": ..., "unstamped": [...]}`: replaced where it stands, else the
/// last member; every other byte unchanged.
pub fn stampedConfig(a: std.mem.Allocator, raw: []const u8, id: []const u8, unstamped: []const []const u8) ![]u8 {
    var value: std.ArrayList(u8) = .empty;
    try value.appendSlice(a, "{\"id\": ");
    try repack.appendJsonString(a, &value, id);
    try value.appendSlice(a, ", \"unstamped\": [");
    for (unstamped, 0..) |name, i| {
        if (i > 0) try value.appendSlice(a, ", ");
        try repack.appendJsonString(a, &value, name);
    }
    try value.appendSlice(a, "]}");

    const open = skipWs(raw, 0);
    if (open >= raw.len or raw[open] != '{') return error.InvalidConfig;
    var j = open + 1;
    var lead: []const u8 = "";
    var last_end: ?usize = null;
    while (true) {
        const at = skipWs(raw, j);
        if (at >= raw.len) return error.InvalidConfig;
        if (raw[at] == '}') break;
        if (raw[at] != '"') return error.InvalidConfig;
        if (last_end == null and std.mem.indexOfScalar(u8, raw[open + 1 .. at], '\n') != null) lead = raw[open + 1 .. at];
        const key_end = try skipString(raw, at);
        const colon = skipWs(raw, key_end);
        if (colon >= raw.len or raw[colon] != ':') return error.InvalidConfig;
        const start = skipWs(raw, colon + 1);
        const end = try skipValue(raw, start);
        if (std.mem.eql(u8, raw[at + 1 .. key_end - 1], key)) return std.mem.concat(a, u8, &.{ raw[0..start], value.items, raw[end..] });
        last_end = end;
        j = skipWs(raw, end);
        if (j < raw.len and raw[j] == ',') j += 1;
    }
    var member: std.ArrayList(u8) = .empty;
    try member.appendSlice(a, if (last_end != null) "," else "");
    try member.appendSlice(a, lead);
    try member.appendSlice(a, "\"" ++ key ++ "\": ");
    try member.appendSlice(a, value.items);
    const at = last_end orelse open + 1;
    return std.mem.concat(a, u8, &.{ raw[0..at], member.items, raw[at..] });
}

const Revision = struct { id: []const u8, unstamped: []const []const u8 };

fn configRevision(a: std.mem.Allocator, config: std.json.Value) !?Revision {
    const v = config.object.get(key) orelse return null;
    if (v != .object) return error.InvalidConfig;
    const id = v.object.get("id") orelse return error.InvalidConfig;
    const un = v.object.get("unstamped") orelse return error.InvalidConfig;
    if (id != .string or un != .array) return error.InvalidConfig;
    const names = try a.alloc([]const u8, un.array.items.len);
    for (un.array.items, names) |item, *out| out.* = if (item == .string) item.string else return error.InvalidConfig;
    return .{ .id = id.string, .unstamped = names };
}

// ── The pack ──────────────────────────────────────────────────────────────

const Shard = struct { name: []const u8, link: bool, size: u64 };

fn listShards(a: std.mem.Allocator, io: std.Io, dir: []const u8) ![]Shard {
    var d = std.Io.Dir.openDirAbsolute(io, dir, .{ .iterate = true }) catch return error.RepackNotAPack;
    defer d.close(io);
    var out: std.ArrayList(Shard) = .empty;
    var it = d.iterate();
    while (try it.next(io)) |entry| {
        if (!std.mem.endsWith(u8, entry.name, shard_suffix)) continue;
        const path = try repack.pathZ(a, dir, entry.name);
        var st: std.c.Stat = undefined;
        if (repack.lstat(path.ptr, &st) != 0) continue;
        try out.append(a, .{ .name = try a.dupe(u8, entry.name), .link = std.c.S.ISLNK(st.mode), .size = @intCast(@max(st.size, 0)) });
    }
    std.mem.sort(Shard, out.items, {}, struct {
        fn lt(_: void, x: Shard, y: Shard) bool {
            return std.mem.lessThan(u8, x.name, y.name);
        }
    }.lt);
    return out.items;
}

const Context = struct {
    config_raw: []u8,
    config: std.json.Value,
    shards: []Shard,
};

fn loadContext(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Context {
    const config_raw = repack.readSmall(io, a, dir, "config.json") catch return error.RepackNotAPack;
    const config = std.json.parseFromSliceLeaky(std.json.Value, a, config_raw, .{}) catch return error.InvalidConfig;
    if (config != .object) return error.InvalidConfig;
    return .{ .config_raw = config_raw, .config = config, .shards = try listShards(a, io, dir) };
}

/// The refusals both modes share, each naming what it found: an unstamped name that is no shard, a symlink not listed,
/// a pattern that selects nothing.
fn checkNames(shards: []const Shard, unstamped: []const []const u8, patterns: []const []const u8, report: *Report) !void {
    for (unstamped) |name| {
        var found = false;
        for (shards) |s| found = found or std.mem.eql(u8, s.name, name);
        if (!found) {
            report.say("--unstamped {s}: no such shard in the pack", .{name});
            return error.RepackUnknownShard;
        }
    }
    for (shards) |s| if (s.link and !contains(unstamped, s.name)) {
        report.say("{s} is a symlink: it is never rewritten through the link; list it with --unstamped {s}", .{ s.name, s.name });
        return error.RepackShardIsLink;
    };
    for (patterns) |p| {
        var hit = false;
        for (shards) |s| hit = hit or globMatch(p, s.name);
        if (!hit) {
            report.say("--shards {s}: matches no shard", .{p});
            return error.RepackNoShardMatch;
        }
    }
}

fn shardStamp(a: std.mem.Allocator, dir: []const u8, name: []const u8) !?[]const u8 {
    const path = try repack.pathZ(a, dir, name);
    const fd = try repack.openFd(path, false);
    defer _ = std.c.close(fd);
    const header = try Header.read(a, fd);
    return headerStamp(header.value);
}

pub fn stampPack(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, opts: Options, report: *Report) !void {
    report.* = .{};
    if (!validId(opts.id)) {
        report.say("pack revision '{s}': 1 to {d} characters from letters, digits and . _ - +", .{ opts.id, max_id_len });
        return error.RepackStampBadId;
    }
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const ctx = try loadContext(a, io, dir);
    if (try configRevision(a, ctx.config)) |have| if (!std.mem.eql(u8, have.id, opts.id)) {
        report.say("config.json already says pack_revision {s}, not {s}; re-run with --pack-revision {s} or remove the member", .{ have.id, opts.id, have.id });
        return error.RepackStampConflict;
    };
    try checkNames(ctx.shards, opts.unstamped, opts.shards, report);

    var todo: std.ArrayList(Shard) = .empty;
    var biggest: u64 = 0;
    for (ctx.shards) |s| {
        if (contains(opts.unstamped, s.name) or !selects(opts.shards, s.name)) continue;
        if (try shardStamp(a, dir, s.name)) |have| {
            if (std.mem.eql(u8, have, opts.id)) {
                report.skipped += 1;
                continue;
            }
            report.say("{s} already carries pack_revision {s}, not {s}; re-run with --pack-revision {s}", .{ s.name, have, opts.id, have });
            return error.RepackStampConflict;
        }
        try todo.append(a, s);
        biggest = @max(biggest, s.size);
    }

    const new_config = try stampedConfig(a, ctx.config_raw, opts.id, opts.unstamped);
    if (todo.items.len == 0 and std.mem.eql(u8, new_config, ctx.config_raw)) return;
    if (todo.items.len > 0) {
        const free = opts.free_bytes orelse (disk_placement.volumeSpaceNear(dir) orelse return error.RepackNoSpace).free;
        if (free < biggest + spare_bytes) return error.RepackNoSpace;
    }
    const lock = try repack.Lock.acquire(gpa, io, dir);
    defer lock.release(gpa);
    for (todo.items) |s| {
        const path = try repack.pathZ(a, dir, s.name);
        if (repack.inUse(path, opts.count_self)) return error.RepackPackInUse;
    }

    for (todo.items) |s| {
        const started = std.Io.Timestamp.now(io, .awake);
        stampShard(gpa, dir, s.name, opts.id, opts) catch |err| {
            report.say("{s}: {s}", .{ s.name, @errorName(err) });
            return err;
        };
        report.stamped += 1;
        const secs = @as(f64, @floatFromInt(started.untilNow(io, .awake).nanoseconds)) / std.time.ns_per_s;
        log.info("[repack] stamped {s}  {d} bytes  {d:.1} s\n", .{ s.name, s.size, secs });
        if (opts.fail_after_shards == report.stamped) return error.RepackInterrupted;
    }
    if (!std.mem.eql(u8, new_config, ctx.config_raw)) try repack.replaceFile(a, dir, "config.json", new_config);
}

/// The loader's rule, read only: every shard not in `unstamped` carries the config's id.
pub fn checkPack(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, id: ?[]const u8, patterns: []const []const u8, report: *Report) !void {
    report.* = .{};
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const ctx = try loadContext(a, io, dir);
    const have = (try configRevision(a, ctx.config)) orelse {
        report.say("config.json has no pack_revision", .{});
        return error.RepackStampMissing;
    };
    if (id) |want| if (!std.mem.eql(u8, want, have.id)) {
        report.say("config.json says pack_revision {s}, not {s}", .{ have.id, want });
        return error.RepackStampConflict;
    };
    try checkNames(ctx.shards, have.unstamped, patterns, report);
    for (ctx.shards) |s| {
        if (contains(have.unstamped, s.name) or !selects(patterns, s.name)) continue;
        const got = (try shardStamp(a, dir, s.name)) orelse {
            report.say("{s} carries no pack_revision (config.json says {s})", .{ s.name, have.id });
            return error.RepackStampMissing;
        };
        if (!std.mem.eql(u8, got, have.id)) {
            report.say("{s} carries pack_revision {s}, config.json says {s}", .{ s.name, got, have.id });
            return error.RepackStampConflict;
        }
        report.stamped += 1;
    }
}

// ── Command line ──────────────────────────────────────────────────────────

pub const Cli = struct {
    pack: []const u8,
    id: ?[]const u8 = null,
    shards: []const []const u8 = &.{},
    unstamped: []const []const u8 = &.{},
    check: bool = false,

    pub fn stamping(self: Cli) bool {
        return self.id != null;
    }
};

/// The argument `parseArgs` refused, for the message.
pub var bad_arg: []const u8 = "";

fn bad(arg: []const u8) error{RepackBadArgs} {
    bad_arg = arg;
    return error.RepackBadArgs;
}

pub fn parseArgs(a: std.mem.Allocator, args: []const []const u8) !Cli {
    var cli: Cli = .{ .pack = "" };
    var shards: std.ArrayList([]const u8) = .empty;
    var unstamped: std.ArrayList([]const u8) = .empty;
    var have_pack = false;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (!std.mem.startsWith(u8, arg, "-")) {
            if (have_pack) return bad(arg);
            have_pack = true;
            cli.pack = arg;
        } else if (std.mem.eql(u8, arg, "--check")) {
            cli.check = true;
        } else if (std.mem.eql(u8, arg, "--pack-revision") or std.mem.eql(u8, arg, "--shards") or std.mem.eql(u8, arg, "--unstamped")) {
            i += 1;
            if (i >= args.len) return bad(arg);
            const value = args[i];
            if (std.mem.eql(u8, arg, "--pack-revision")) {
                if (cli.id != null or !validId(value)) return bad(value);
                cli.id = value;
            } else if (value.len == 0) return bad(arg) else try (if (std.mem.eql(u8, arg, "--shards")) shards else unstamped).append(a, value);
        } else return bad(arg);
    }
    if (!have_pack) return bad("<pack>");
    cli.shards = shards.items;
    cli.unstamped = unstamped.items;
    if (cli.shards.len > 0 and cli.id == null and !cli.check) return bad("--shards");
    if (cli.unstamped.len > 0 and (cli.id == null or cli.check)) return bad("--unstamped");
    return cli;
}

/// Is this a pack-order DeepSeek-V4.1 STQ pack, which `sushi repack` turns pair-major before it stamps?
pub fn wantsRepack(a: std.mem.Allocator, config_raw: []const u8) bool {
    const config = std.json.parseFromSliceLeaky(std.json.Value, a, config_raw, .{}) catch return false;
    if (config != .object) return false;
    const model_type = config.object.get("model_type") orelse return false;
    if (model_type != .string or !std.mem.eql(u8, model_type.string, "deepseek_v41")) return false;
    const quant = config.object.get("expert_quant") orelse return false;
    if (quant != .object or quant.object.get("tile_order") != null) return false;
    const format = quant.object.get("format") orelse return false;
    return format == .string and std.mem.eql(u8, format.string, "stq");
}
