//! The engine's Metal pipeline recipes on disk, so a load makes its pipelines without running a forward to find them.
//! A manifest lists every pipeline a complete warm made: the kernel's source text and entry point, or the metallib
//! function and its constants. It lives in `<root>/<scope>-<build>/manifest.bin`: `scope` names the pack and the shape
//! of its warm, `build` the binary, metal library, GPU and OS that turn a recipe into code. This file reads and writes
//! the format and the directory; it touches no MLX and no GPU.

const std = @import("std");
const metal = @import("metal.zig");

pub const version: u32 = 2;
const magic = "SUSHIPLM";
pub const file_name = "manifest.bin";
const temp_name = "manifest.bin.tmp";

pub const Kind = enum(u8) {
    /// A function of `mlx.metallib` made as it is: `function` is its name and `key` the engine's table key.
    stock = 1,
    /// A metallib function specialised by boolean and int constants: `key` is MLX's name for the set.
    stock_flags = 2,
    /// A kernel compiled from `text`, whose entry point is `function`.
    source = 3,
    /// No pipeline: `key` names a warm that the recipes before it make whole, so a load that made them all can skip it.
    marker = 4,
};

pub const Recipe = struct {
    kind: Kind,
    function: []const u8,
    key: []const u8 = "",
    text: []const u8 = "",
    flags: []const metal.Flag = &.{},
    ints: []const metal.IntConstant = &.{},

    pub fn eql(a: Recipe, b: Recipe) bool {
        if (a.kind != b.kind or a.flags.len != b.flags.len or a.ints.len != b.ints.len) return false;
        if (!std.mem.eql(u8, a.function, b.function) or !std.mem.eql(u8, a.key, b.key) or !std.mem.eql(u8, a.text, b.text)) return false;
        for (a.flags, b.flags) |x, y| if (x.index != y.index or x.value != y.value) return false;
        for (a.ints, b.ints) |x, y| if (x.index != y.index or x.value != y.value) return false;
        return true;
    }
};

pub const DecodeError = error{ ManifestTruncated, ManifestMagic, ManifestVersion, ManifestChecksum, ManifestCorrupt, ManifestKey, OutOfMemory };

const Sha256 = std.crypto.hash.sha2.Sha256;
const digest_len = Sha256.digest_length;

fn putU32(list: *std.ArrayList(u8), a: std.mem.Allocator, v: u32) !void {
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, v, .little);
    try list.appendSlice(a, &b);
}

fn putBytes(list: *std.ArrayList(u8), a: std.mem.Allocator, bytes: []const u8) !void {
    try putU32(list, a, @intCast(bytes.len));
    try list.appendSlice(a, bytes);
}

/// The file's bytes: header, `key`, the recipes, and a SHA-256 of everything before it.
pub fn encode(a: std.mem.Allocator, key: []const u8, recipes: []const Recipe) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, magic);
    try putU32(&out, a, version);
    try putBytes(&out, a, key);
    try putU32(&out, a, @intCast(recipes.len));
    for (recipes) |r| {
        try out.append(a, @intFromEnum(r.kind));
        try putBytes(&out, a, r.function);
        try putBytes(&out, a, r.key);
        try putBytes(&out, a, r.text);
        try putU32(&out, a, @intCast(r.flags.len));
        for (r.flags) |f| {
            try putU32(&out, a, @intCast(f.index));
            try out.append(a, @intFromBool(f.value));
        }
        try putU32(&out, a, @intCast(r.ints.len));
        for (r.ints) |c| {
            try putU32(&out, a, @intCast(c.index));
            try putU32(&out, a, @bitCast(c.value));
        }
    }
    var sum: [digest_len]u8 = undefined;
    Sha256.hash(out.items, &sum, .{});
    try out.appendSlice(a, &sum);
    return out.toOwnedSlice(a);
}

const Reader = struct {
    bytes: []const u8,
    at: usize = 0,

    fn take(r: *Reader, n: usize) DecodeError![]const u8 {
        if (n > r.bytes.len - r.at) return error.ManifestCorrupt;
        defer r.at += n;
        return r.bytes[r.at..][0..n];
    }

    fn u32_(r: *Reader) DecodeError!u32 {
        return std.mem.readInt(u32, (try r.take(4))[0..4], .little);
    }

    fn slice(r: *Reader) DecodeError![]const u8 {
        return r.take(try r.u32_());
    }
};

/// What a manifest file held; its strings point into `bytes`, which the arena keeps.
pub const Loaded = struct {
    arena: std.heap.ArenaAllocator,
    key: []const u8,
    recipes: []Recipe,

    pub fn deinit(l: *Loaded) void {
        l.arena.deinit();
    }
};

/// Reads the file `bytes` (copied into the result). Any damage is an error, never a partial list.
pub fn decode(a: std.mem.Allocator, bytes: []const u8) DecodeError!Loaded {
    if (bytes.len < magic.len + 4 + digest_len) return error.ManifestTruncated;
    if (!std.mem.eql(u8, bytes[0..magic.len], magic)) return error.ManifestMagic;
    const body = bytes[0 .. bytes.len - digest_len];
    var sum: [digest_len]u8 = undefined;
    Sha256.hash(body, &sum, .{});
    if (!std.mem.eql(u8, &sum, bytes[body.len..])) return error.ManifestChecksum;
    var loaded: Loaded = .{ .arena = .init(a), .key = "", .recipes = &.{} };
    errdefer loaded.arena.deinit();
    const arena = loaded.arena.allocator();
    const own = try arena.dupe(u8, body);
    var r: Reader = .{ .bytes = own, .at = magic.len };
    if (try r.u32_() != version) return error.ManifestVersion;
    loaded.key = try r.slice();
    const n = try r.u32_();
    // Every recipe is at least its kind byte and five lengths.
    if (n > own.len / 21) return error.ManifestCorrupt;
    const recipes = try arena.alloc(Recipe, n);
    for (recipes) |*rec| {
        const kind = std.enums.fromInt(Kind, (try r.take(1))[0]) orelse return error.ManifestCorrupt;
        rec.* = .{ .kind = kind, .function = try r.slice(), .key = try r.slice(), .text = try r.slice() };
        const nf = try r.u32_();
        if (nf > (own.len - r.at) / 5) return error.ManifestCorrupt;
        const flags = try arena.alloc(metal.Flag, nf);
        for (flags) |*f| {
            f.index = try r.u32_();
            f.value = (try r.take(1))[0] != 0;
        }
        rec.flags = flags;
        const ni = try r.u32_();
        if (ni > (own.len - r.at) / 8) return error.ManifestCorrupt;
        const ints = try arena.alloc(metal.IntConstant, ni);
        for (ints) |*c| {
            c.index = try r.u32_();
            c.value = @bitCast(try r.u32_());
        }
        rec.ints = ints;
    }
    if (r.at != own.len) return error.ManifestCorrupt;
    loaded.recipes = recipes;
    return loaded;
}

/// What a key is built from: labelled, length-prefixed values, so two lists that differ in where a value ends differ.
pub const Digest = struct {
    h: Sha256 = .init(.{}),

    pub fn add(d: *Digest, label: []const u8, value: []const u8) void {
        var len: [8]u8 = undefined;
        for ([_][]const u8{ label, value }) |part| {
            std.mem.writeInt(u64, &len, part.len, .little);
            d.h.update(&len);
            d.h.update(part);
        }
    }

    pub fn addInt(d: *Digest, label: []const u8, value: u64) void {
        var b: [8]u8 = undefined;
        std.mem.writeInt(u64, &b, value, .little);
        d.add(label, &b);
    }

    /// The first 16 hex digits of the digest; the digest is spent.
    pub fn hex(d: *Digest) [16]u8 {
        var sum: [digest_len]u8 = undefined;
        d.h.final(&sum);
        return std.fmt.bytesToHex(sum[0..8].*, .lower);
    }
};

pub const Key = struct {
    scope: [16]u8,
    build: [16]u8,

    pub const dir_len = 33;

    pub fn dirName(k: *const Key, buf: *[dir_len]u8) []const u8 {
        return std.fmt.bufPrint(buf, "{s}-{s}", .{ k.scope, k.build }) catch unreachable;
    }
};

extern "c" fn sysctlbyname(name: [*:0]const u8, oldp: ?*anyopaque, oldlenp: ?*usize, newp: ?*const anyopaque, newlen: usize) c_int;

/// What turns a recipe into code: the running binary (every recipe its code can emit), the metal library's size and
/// mtime (its stock kernels), the GPU architecture and the OS build (the compiler).
pub fn addBuild(d: *Digest, a: std.mem.Allocator, io: std.Io, gpu_arch: []const u8, metallib_path: []const u8) !void {
    d.addInt("manifest version", version);
    const exe = try std.process.executablePathAlloc(io, a);
    defer a.free(exe);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, exe, a, .limited(1 << 30));
    defer a.free(bytes);
    var sum: [digest_len]u8 = undefined;
    Sha256.hash(bytes, &sum, .{});
    d.add("binary", &sum);
    const lib = try std.Io.Dir.cwd().statFile(io, metallib_path, .{});
    d.addInt("metallib size", lib.size);
    d.addInt("metallib mtime", @truncate(@as(u96, @bitCast(lib.mtime.toNanoseconds()))));
    d.add("gpu", gpu_arch);
    var os: [64]u8 = undefined;
    var len: usize = os.len;
    if (sysctlbyname("kern.osversion", &os, &len, null, 0) != 0) return error.OsBuildUnknown;
    d.add("os build", os[0..len]);
}

/// Which pack: its config.json (which carries the pack revision), its index, and the name and size of each shard.
/// Not the shards' bytes.
pub fn addPack(d: *Digest, a: std.mem.Allocator, io: std.Io, dir_path: []const u8) !void {
    var dir = try std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true });
    defer dir.close(io);
    for ([_][]const u8{ "config.json", "model.safetensors.index.json" }) |name| {
        const bytes = dir.readFileAlloc(io, name, a, .limited(64 << 20)) catch |e| switch (e) {
            error.FileNotFound => {
                d.add(name, "absent");
                continue;
            },
            else => return e,
        };
        defer a.free(bytes);
        d.add(name, bytes);
    }
    var shards: std.ArrayList(struct { name: []u8, size: u64 }) = .empty;
    defer {
        for (shards.items) |s| a.free(s.name);
        shards.deinit(a);
    }
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        switch (entry.kind) {
            .file => try addShard(a, io, dir, "", entry.name, &shards),
            .directory => {
                var sub = dir.openDir(io, entry.name, .{ .iterate = true }) catch continue;
                defer sub.close(io);
                var sit = sub.iterate();
                while (try sit.next(io)) |se| if (se.kind == .file) try addShard(a, io, sub, entry.name, se.name, &shards);
            },
            else => {},
        }
    }
    std.mem.sort(@TypeOf(shards.items[0]), shards.items, {}, struct {
        fn less(_: void, x: @TypeOf(shards.items[0]), y: @TypeOf(shards.items[0])) bool {
            return std.mem.lessThan(u8, x.name, y.name);
        }
    }.less);
    for (shards.items) |s| {
        d.add("shard", s.name);
        d.addInt("shard size", s.size);
    }
}

fn addShard(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, parent: []const u8, name: []const u8, shards: anytype) !void {
    if (!std.mem.endsWith(u8, name, ".safetensors")) return;
    const st = try dir.statFile(io, name, .{});
    const shown = if (parent.len == 0) try a.dupe(u8, name) else try std.fmt.allocPrint(a, "{s}/{s}", .{ parent, name });
    errdefer a.free(shown);
    try shards.append(a, .{ .name = shown, .size = st.size });
}

/// `<home>/.sushi/cache/pipelines`.
pub fn rootPath(buf: []u8, home: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}/.sushi/cache/pipelines", .{home});
}

var configured_buf: [std.fs.max_path_bytes]u8 = undefined;
var configured_len: usize = 0;

/// Gives the process its manifest root, `rootPath(home)`: the server does this at start. Until then there is none, and
/// a load reads and writes no manifest.
pub fn setRootFromHome(home: []const u8) !void {
    configured_len = (try rootPath(&configured_buf, home)).len;
}

pub fn configuredRoot() ?[]const u8 {
    return if (configured_len == 0) null else configured_buf[0..configured_len];
}

pub const Lookup = union(enum) {
    none,
    /// The manifest of this very key.
    same: Loaded,
    /// The newest manifest of the same scope under another build.
    other: Loaded,
    /// A manifest that could not be trusted: its directory and why.
    unreadable: struct { dir: [Key.dir_len]u8, same: bool, reason: []const u8 },

    pub fn deinit(l: *Lookup) void {
        switch (l.*) {
            .same, .other => |*loaded| loaded.deinit(),
            else => {},
        }
    }
};

const max_manifest_bytes = 1 << 30;

fn readManifest(a: std.mem.Allocator, io: std.Io, root: std.Io.Dir, dir: []const u8) (DecodeError || error{ManifestMissing})!Loaded {
    var sub: [Key.dir_len + 1 + file_name.len]u8 = undefined;
    const path = std.fmt.bufPrint(&sub, "{s}/{s}", .{ dir, file_name }) catch unreachable;
    const bytes = root.readFileAlloc(io, path, a, .limited(max_manifest_bytes)) catch |e| switch (e) {
        error.FileNotFound => return error.ManifestMissing,
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.ManifestCorrupt,
    };
    defer a.free(bytes);
    return decode(a, bytes);
}

/// The manifest to start a load from: this key's, else the newest of the same scope, else none.
pub fn lookup(a: std.mem.Allocator, io: std.Io, root: std.Io.Dir, key: Key) !Lookup {
    var name_buf: [Key.dir_len]u8 = undefined;
    const mine = key.dirName(&name_buf);
    if (readManifest(a, io, root, mine)) |loaded_in| {
        var loaded = loaded_in;
        errdefer loaded.deinit();
        if (std.mem.eql(u8, loaded.key, mine)) return .{ .same = loaded };
        loaded.deinit();
        return .{ .unreadable = .{ .dir = name_buf, .same = true, .reason = @errorName(error.ManifestKey) } };
    } else |e| switch (e) {
        error.ManifestMissing => {},
        error.OutOfMemory => return error.OutOfMemory,
        else => return .{ .unreadable = .{ .dir = name_buf, .same = true, .reason = @errorName(e) } },
    }
    var newest: ?[Key.dir_len]u8 = null;
    var newest_ns: i96 = std.math.minInt(i96);
    var it = root.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .directory or entry.name.len != Key.dir_len) continue;
        if (!std.mem.startsWith(u8, entry.name, &key.scope) or entry.name[16] != '-') continue;
        var path_buf: [Key.dir_len + 1 + file_name.len]u8 = undefined;
        const path = std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ entry.name, file_name }) catch unreachable;
        const st = root.statFile(io, path, .{}) catch continue;
        if (st.mtime.toNanoseconds() > newest_ns) {
            newest_ns = st.mtime.toNanoseconds();
            newest = entry.name[0..Key.dir_len].*;
        }
    }
    const dir = newest orelse return .none;
    const loaded = readManifest(a, io, root, &dir) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return .{ .unreadable = .{ .dir = dir, .same = false, .reason = @errorName(e) } },
    };
    return .{ .other = loaded };
}

/// Writes the manifest of `key` through a temp file, so a reader sees the old file or the whole new one.
pub fn write(a: std.mem.Allocator, io: std.Io, root: std.Io.Dir, key: Key, recipes: []const Recipe) !void {
    var name_buf: [Key.dir_len]u8 = undefined;
    const dir_name = key.dirName(&name_buf);
    const bytes = try encode(a, dir_name, recipes);
    defer a.free(bytes);
    var dir = try root.createDirPathOpen(io, dir_name, .{});
    defer dir.close(io);
    try dir.writeFile(io, .{ .sub_path = temp_name, .data = bytes });
    errdefer dir.deleteFile(io, temp_name) catch {};
    try dir.rename(temp_name, dir, file_name, io);
}

/// Removes the other builds of `key`'s scope; the names removed go to `removed` (owned by the caller).
pub fn prune(a: std.mem.Allocator, io: std.Io, root: std.Io.Dir, key: Key, removed: *std.ArrayList([Key.dir_len]u8)) !void {
    var name_buf: [Key.dir_len]u8 = undefined;
    const mine = key.dirName(&name_buf);
    var found: std.ArrayList([Key.dir_len]u8) = .empty;
    defer found.deinit(a);
    var it = root.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .directory or entry.name.len != Key.dir_len) continue;
        if (!std.mem.startsWith(u8, entry.name, &key.scope) or entry.name[16] != '-') continue;
        if (std.mem.eql(u8, entry.name, mine)) continue;
        try found.append(a, entry.name[0..Key.dir_len].*);
    }
    for (found.items) |name| {
        try root.deleteTree(io, &name);
        try removed.append(a, name);
    }
}

// ── Tests ──

const testing = std.testing;

const sample_ints = [_]metal.IntConstant{.{ .index = 26, .value = 128 }};

fn sample(flags: []const metal.Flag) [4]Recipe {
    return .{
        .{ .kind = .stock, .function = "rmsbfloat16", .key = "rmsbfloat16" },
        .{ .kind = .stock_flags, .function = "steel_gemm", .key = "steel_gemm_hash", .flags = flags },
        .{ .kind = .stock_flags, .function = "sdpa_vector_2pass_1", .key = "sdpa_vector_2pass_1_128", .flags = flags, .ints = &sample_ints },
        .{ .kind = .source, .function = "custom_kernel_x", .text = "#include <metal_stdlib>\nkernel void custom_kernel_x() {}\n" },
    };
}

test "pipeline manifest: recipes come back from the file's bytes as they went in" {
    const flags = [_]metal.Flag{ .{ .index = 200, .value = true }, .{ .index = 201, .value = false } };
    const recipes = sample(&flags);
    const bytes = try encode(testing.allocator, "scope-build", &recipes);
    defer testing.allocator.free(bytes);
    var loaded = try decode(testing.allocator, bytes);
    defer loaded.deinit();
    try testing.expectEqualStrings("scope-build", loaded.key);
    try testing.expectEqual(recipes.len, loaded.recipes.len);
    for (recipes, loaded.recipes) |want, got| try testing.expect(want.eql(got));
    // An int constant is part of a recipe's identity: another value is another pipeline.
    var other = recipes[2];
    other.ints = &.{.{ .index = 26, .value = 64 }};
    try testing.expect(!recipes[2].eql(other));
    try testing.expectEqual(@as(i32, 128), loaded.recipes[2].ints[0].value);
}

test "pipeline manifest: a marker recipe and a manifest of the format before int constants" {
    const marker: Recipe = .{ .kind = .marker, .function = "vision tower", .key = "tower:1024:16" };
    const bytes = try encode(testing.allocator, "k", &.{marker});
    defer testing.allocator.free(bytes);
    var loaded = try decode(testing.allocator, bytes);
    defer loaded.deinit();
    try testing.expect(loaded.recipes[0].eql(marker));
    // The format number is in the file and in the build key: a file of the older format is refused, and its key differs.
    try testing.expectEqual(@as(u32, 2), version);
    const old = try testing.allocator.dupe(u8, bytes);
    defer testing.allocator.free(old);
    std.mem.writeInt(u32, old[magic.len..][0..4], 1, .little);
    var sum: [digest_len]u8 = undefined;
    Sha256.hash(old[0 .. old.len - digest_len], &sum, .{});
    @memcpy(old[old.len - digest_len ..], &sum);
    try testing.expectError(error.ManifestVersion, decode(testing.allocator, old));
    var now: Digest = .{};
    now.addInt("manifest version", version);
    var before: Digest = .{};
    before.addInt("manifest version", version - 1);
    try testing.expect(!std.mem.eql(u8, &now.hex(), &before.hex()));
}

test "pipeline manifest: a file cut short, changed in one byte or not a manifest is refused by name" {
    const flags = [_]metal.Flag{.{ .index = 1, .value = true }};
    const recipes = sample(&flags);
    const bytes = try encode(testing.allocator, "k", &recipes);
    defer testing.allocator.free(bytes);
    for (0..bytes.len) |cut| {
        const got = decode(testing.allocator, bytes[0..cut]);
        if (got) |*ok| {
            var l = ok.*;
            l.deinit();
            return error.TruncatedFileAccepted;
        } else |e| try testing.expect(e == error.ManifestTruncated or e == error.ManifestChecksum or e == error.ManifestMagic);
    }
    const copy = try testing.allocator.dupe(u8, bytes);
    defer testing.allocator.free(copy);
    copy[copy.len / 2] ^= 1;
    try testing.expectError(error.ManifestChecksum, decode(testing.allocator, copy));
    copy[copy.len / 2] ^= 1;
    copy[0] = 'X';
    try testing.expectError(error.ManifestMagic, decode(testing.allocator, copy));
    try testing.expectError(error.ManifestTruncated, decode(testing.allocator, "SUSHIPLM"));
}

test "pipeline manifest: a count that outruns the file is corrupt even when its checksum is right" {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(testing.allocator);
    try out.appendSlice(testing.allocator, magic);
    try putU32(&out, testing.allocator, version);
    try putBytes(&out, testing.allocator, "k");
    try putU32(&out, testing.allocator, 1 << 30);
    var sum: [digest_len]u8 = undefined;
    Sha256.hash(out.items, &sum, .{});
    try out.appendSlice(testing.allocator, &sum);
    try testing.expectError(error.ManifestCorrupt, decode(testing.allocator, out.items));
}

test "pipeline manifest: a key digest differs in any part, and in where a part ends" {
    var a: Digest = .{};
    a.add("ab", "c");
    var b: Digest = .{};
    b.add("a", "bc");
    var c: Digest = .{};
    c.add("ab", "c");
    const ha = a.hex();
    try testing.expect(!std.mem.eql(u8, &ha, &b.hex()));
    try testing.expectEqualSlices(u8, &ha, &c.hex());
    var d: Digest = .{};
    d.addInt("n", 1);
    var e: Digest = .{};
    e.addInt("n", 2);
    try testing.expect(!std.mem.eql(u8, &d.hex(), &e.hex()));
}

fn testKey(scope: []const u8, build: []const u8) Key {
    var k: Key = undefined;
    @memcpy(&k.scope, scope[0..16]);
    @memcpy(&k.build, build[0..16]);
    return k;
}

test "pipeline manifest: a lookup finds this key's file, else the newest build of the scope, else nothing" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const flags = [_]metal.Flag{};
    const recipes = sample(&flags);
    const mine = testKey("aaaaaaaaaaaaaaaa", "1111111111111111");
    var none = try lookup(testing.allocator, io, tmp.dir, mine);
    defer none.deinit();
    try testing.expect(none == .none);

    try write(testing.allocator, io, tmp.dir, testKey("aaaaaaaaaaaaaaaa", "2222222222222222"), &recipes);
    try write(testing.allocator, io, tmp.dir, testKey("bbbbbbbbbbbbbbbb", "1111111111111111"), &recipes);
    var other = try lookup(testing.allocator, io, tmp.dir, mine);
    defer other.deinit();
    try testing.expect(other == .other);
    try testing.expectEqual(recipes.len, other.other.recipes.len);

    try write(testing.allocator, io, tmp.dir, mine, recipes[0..1]);
    var same = try lookup(testing.allocator, io, tmp.dir, mine);
    defer same.deinit();
    try testing.expect(same == .same);
    try testing.expectEqual(@as(usize, 1), same.same.recipes.len);
}

test "pipeline manifest: a damaged file is reported with its reason and its directory, never read as a manifest" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const flags = [_]metal.Flag{};
    const recipes = sample(&flags);
    const mine = testKey("aaaaaaaaaaaaaaaa", "1111111111111111");
    try write(testing.allocator, io, tmp.dir, mine, &recipes);
    var name_buf: [Key.dir_len]u8 = undefined;
    const dir = mine.dirName(&name_buf);
    var path: [64]u8 = undefined;
    const file = try std.fmt.bufPrint(&path, "{s}/{s}", .{ dir, file_name });
    const whole = try tmp.dir.readFileAlloc(io, file, testing.allocator, .limited(1 << 20));
    defer testing.allocator.free(whole);
    try tmp.dir.writeFile(io, .{ .sub_path = file, .data = whole[0 .. whole.len - 7] });
    var got = try lookup(testing.allocator, io, tmp.dir, mine);
    defer got.deinit();
    try testing.expect(got == .unreadable);
    try testing.expect(got.unreadable.same);
    try testing.expectEqualStrings(dir, &got.unreadable.dir);
    try testing.expectEqualStrings("ManifestChecksum", got.unreadable.reason);
}

test "pipeline manifest: a write leaves no temp file, and a prune removes only the other builds of the scope" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const flags = [_]metal.Flag{};
    const recipes = sample(&flags);
    const mine = testKey("aaaaaaaaaaaaaaaa", "1111111111111111");
    const old = testKey("aaaaaaaaaaaaaaaa", "2222222222222222");
    const foreign = testKey("bbbbbbbbbbbbbbbb", "2222222222222222");
    for ([_]Key{ old, foreign, mine }) |k| try write(testing.allocator, io, tmp.dir, k, &recipes);
    var name_buf: [Key.dir_len]u8 = undefined;
    var temp_path: [64]u8 = undefined;
    const temp = try std.fmt.bufPrint(&temp_path, "{s}/{s}", .{ mine.dirName(&name_buf), temp_name });
    try testing.expectError(error.FileNotFound, tmp.dir.access(io, temp, .{}));

    var removed: std.ArrayList([Key.dir_len]u8) = .empty;
    defer removed.deinit(testing.allocator);
    try prune(testing.allocator, io, tmp.dir, mine, &removed);
    try testing.expectEqual(@as(usize, 1), removed.items.len);
    try testing.expectEqualStrings(old.dirName(&name_buf), &removed.items[0]);
    var gone = try lookup(testing.allocator, io, tmp.dir, old);
    defer gone.deinit();
    try testing.expect(gone == .other);
    var kept = try lookup(testing.allocator, io, tmp.dir, foreign);
    defer kept.deinit();
    try testing.expect(kept == .same);
}

test "pipeline manifest: a pack's key part follows its config and the size of each shard, not other files" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"a\":1}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "model-1.safetensors", .data = "12345" });
    try tmp.dir.createDirPath(io, "mtp");
    try tmp.dir.writeFile(io, .{ .sub_path = "mtp/head.safetensors", .data = "xyz" });
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &root_buf);
    const root = root_buf[0..n];

    const first = try packKey(root);
    try tmp.dir.writeFile(io, .{ .sub_path = "notes.txt", .data = "not part of the pack" });
    try testing.expectEqualSlices(u8, &first, &(try packKey(root)));
    try tmp.dir.writeFile(io, .{ .sub_path = "mtp/head.safetensors", .data = "xyzw" });
    try testing.expect(!std.mem.eql(u8, &first, &(try packKey(root))));
    try tmp.dir.writeFile(io, .{ .sub_path = "mtp/head.safetensors", .data = "xyz" });
    try testing.expectEqualSlices(u8, &first, &(try packKey(root)));
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"a\":2}" });
    try testing.expect(!std.mem.eql(u8, &first, &(try packKey(root))));
}

fn packKey(root: []const u8) ![16]u8 {
    var d: Digest = .{};
    try addPack(&d, testing.allocator, testing.io, root);
    return d.hex();
}

test "pipeline manifest: the build part of a key is stable in one process and follows the GPU and the metal library" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "mlx.metallib", .data = "lib" });
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &root_buf);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const lib = try std.fmt.bufPrint(&path_buf, "{s}/mlx.metallib", .{root_buf[0..n]});
    const keys = [_][16]u8{ try buildKey("applegpu_g17s", lib), try buildKey("applegpu_g17s", lib), try buildKey("applegpu_g16s", lib) };
    try testing.expectEqualSlices(u8, &keys[0], &keys[1]);
    try testing.expect(!std.mem.eql(u8, &keys[0], &keys[2]));
    try tmp.dir.writeFile(io, .{ .sub_path = "mlx.metallib", .data = "longer lib" });
    try testing.expect(!std.mem.eql(u8, &keys[0], &(try buildKey("applegpu_g17s", lib))));
}

fn buildKey(arch: []const u8, lib: []const u8) ![16]u8 {
    var d: Digest = .{};
    try addBuild(&d, testing.allocator, testing.io, arch, lib);
    return d.hex();
}

test "pipeline manifest: a process has no root until it is given the home" {
    try testing.expect(configuredRoot() == null);
    try setRootFromHome("/home/someone");
    defer configured_len = 0;
    try testing.expectEqualStrings("/home/someone/.sushi/cache/pipelines", configuredRoot().?);
}
