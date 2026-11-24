//! A pack's non-expert tensors taken as stored from another pack of the same model: every shard that holds no
//! routed-expert bank, the n-gram table, and the affine settings that pack's config states. A rate-grouped
//! layer's router rows follow its group order, so they are rewritten in place in the copy.
const std = @import("std");
const pack = @import("pack.zig");
const layer_stage = @import("layer_stage.zig");

pub const Error = error{ TrunkConfigInvalid, TrunkShardMixed, TrunkRouterInvalid, TrunkFromRateGrouped } || std.mem.Allocator.Error;

pub const Served = struct {
    arena: std.heap.ArenaAllocator,
    config: []const u8,
    bits: u32,
    group: u32,
    overrides: []const pack.Override,
    ngram: ?pack.NgramSpec,

    pub fn deinit(self: *Served) void {
        self.arena.deinit();
    }
};

fn positive(v: ?std.json.Value) ?u32 {
    const x = v orelse return null;
    if (x != .integer or x.integer <= 0 or x.integer > 1 << 16) return null;
    return @intCast(x.integer);
}

/// The trunk settings of the pack whose `config.json` is `text`.
pub fn parse(allocator: std.mem.Allocator, text: []const u8) Error!Served {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const a = arena.allocator();
    const root = std.json.parseFromSliceLeaky(std.json.Value, a, text, .{}) catch return error.TrunkConfigInvalid;
    if (root != .object) return error.TrunkConfigInvalid;
    const q = root.object.get("quantization") orelse return error.TrunkConfigInvalid;
    if (q != .object) return error.TrunkConfigInvalid;
    const bits = positive(q.object.get("bits")) orelse return error.TrunkConfigInvalid;
    const group = positive(q.object.get("group_size")) orelse return error.TrunkConfigInvalid;
    var overrides = std.ArrayList(pack.Override).empty;
    var it = q.object.iterator();
    while (it.next()) |e| {
        if (e.value_ptr.* != .object) continue;
        const o = e.value_ptr.object;
        if (o.get("mode")) |m| if (m != .string or !std.mem.eql(u8, m.string, "affine")) return error.TrunkConfigInvalid;
        const o_bits = positive(o.get("bits")) orelse return error.TrunkConfigInvalid;
        const o_group = positive(o.get("group_size")) orelse return error.TrunkConfigInvalid;
        const o_base = try a.dupe(u8, e.key_ptr.*);
        errdefer a.free(o_base);
        try overrides.append(a, .{ .base = o_base, .bits = o_bits, .group = o_group });
    }
    var ngram: ?pack.NgramSpec = null;
    if (root.object.get("ngram_table")) |n| {
        if (n != .object) return error.TrunkConfigInvalid;
        const ng_bits = positive(n.object.get("bits")) orelse return error.TrunkConfigInvalid;
        // A table served unquantized is BF16 rows with no groups: group 0.
        const ng_group: u32 = if (ng_bits == 16) blk: {
            const g = n.object.get("group_size") orelse return error.TrunkConfigInvalid;
            if (g != .integer or g.integer != 0) return error.TrunkConfigInvalid;
            break :blk 0;
        } else positive(n.object.get("group_size")) orelse return error.TrunkConfigInvalid;
        ngram = .{ .bits = ng_bits, .group = ng_group };
    }
    const config = try a.dupe(u8, text);
    return .{ .arena = arena, .config = config, .bits = bits, .group = group, .overrides = overrides.items, .ngram = ngram };
}

pub fn read(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !Served {
    const text = try dir.readFileAlloc(io, "config.json", allocator, .limited(16 << 20));
    defer allocator.free(text);
    return parse(allocator, text);
}

/// A rate-grouped bank's tensor name: `...switch_mlp.<proj>.g<N>.<part>`.
fn isGroupBank(name: []const u8) bool {
    const marker = ".switch_mlp.";
    const at = std.mem.indexOf(u8, name, marker) orelse return false;
    var it = std.mem.splitScalar(u8, name[at + marker.len ..], '.');
    _ = it.next();
    const group = it.next() orelse return false;
    return group.len > 1 and group[0] == 'g' and std.mem.indexOfNone(u8, group[1..], "0123456789") == null and it.next() != null;
}

/// A pack written with a rate plan stores its routers in group order; `copy` reorders a router assuming source
/// order, so such a pack cannot be a trunk source.
pub fn refuseRateGrouped(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, config: []const u8) !void {
    const cfg = std.json.parseFromSlice(std.json.Value, allocator, config, .{}) catch return error.TrunkConfigInvalid;
    defer cfg.deinit();
    if (cfg.value == .object) if (cfg.value.object.get("expert_quant")) |eq| {
        if (eq == .object and eq.object.contains("rate_plan_sha256")) return error.TrunkFromRateGrouped;
    };
    var it = dir.iterate();
    while (try it.next(io)) |e| {
        if (e.kind != .file or e.name[0] == '.' or !std.mem.endsWith(u8, e.name, ".safetensors")) continue;
        var h = try readHeader(allocator, io, dir, e.name);
        defer h.parsed.deinit();
        var tit = h.root.iterator();
        while (tit.next()) |t| if (isGroupBank(t.key_ptr.*)) return error.TrunkFromRateGrouped;
    }
}

extern "c" fn clonefile(src: [*:0]const u8, dst: [*:0]const u8, flags: u32) c_int;

/// A copy-on-write clone where the volume has them, else a byte copy.
pub fn cloneOrCopy(io: std.Io, from: std.Io.Dir, out: std.Io.Dir, name: []const u8) !void {
    var a_buf: [std.fs.max_path_bytes]u8 = undefined;
    var b_buf: [std.fs.max_path_bytes]u8 = undefined;
    const a_len = try from.realPath(io, &a_buf);
    const b_len = try out.realPath(io, &b_buf);
    var src_z: [std.fs.max_path_bytes + 1]u8 = undefined;
    var dst_z: [std.fs.max_path_bytes + 1]u8 = undefined;
    const src = try std.fmt.bufPrintSentinel(&src_z, "{s}/{s}", .{ a_buf[0..a_len], name }, 0);
    const dst = try std.fmt.bufPrintSentinel(&dst_z, "{s}/{s}", .{ b_buf[0..b_len], name }, 0);
    out.deleteFile(io, name) catch {};
    if (clonefile(src.ptr, dst.ptr, 0) == 0) return;
    try from.copyFile(name, out, name, io, .{});
}

pub const Header = struct {
    parsed: std.json.Parsed(std.json.Value),
    len: u64,
    /// The tensors and `__metadata__`: `readHeader` checks the document is an object.
    root: std.json.ObjectMap,
};

/// A header entry's dtype, shape (owned) and data range, every field checked.
pub const Tensor = struct {
    dtype: []const u8,
    shape: []usize,
    lo: u64,
    hi: u64,

    pub fn of(allocator: std.mem.Allocator, v: std.json.Value) !Tensor {
        if (v != .object) return error.BadSafetensorsHeader;
        const dtype = v.object.get("dtype") orelse return error.BadSafetensorsHeader;
        const shape = v.object.get("shape") orelse return error.BadSafetensorsHeader;
        const offs = v.object.get("data_offsets") orelse return error.BadSafetensorsHeader;
        if (dtype != .string or shape != .array or offs != .array or offs.array.items.len != 2) return error.BadSafetensorsHeader;
        const lo = intOf(u64, offs.array.items[0]) orelse return error.BadSafetensorsHeader;
        const hi = intOf(u64, offs.array.items[1]) orelse return error.BadSafetensorsHeader;
        if (hi < lo) return error.BadSafetensorsHeader;
        const dims = try allocator.alloc(usize, shape.array.items.len);
        errdefer allocator.free(dims);
        for (shape.array.items, dims) |d, *o| o.* = intOf(usize, d) orelse return error.BadSafetensorsHeader;
        return .{ .dtype = dtype.string, .shape = dims, .lo = lo, .hi = hi };
    }

    pub fn deinit(self: Tensor, allocator: std.mem.Allocator) void {
        allocator.free(self.shape);
    }
};

fn intOf(comptime T: type, v: std.json.Value) ?T {
    if (v != .integer) return null;
    return std.math.cast(T, v.integer);
}

/// Rewrites shard `file` of `dir` without the tensors `drop(ctx, name)` names, keeping the rest byte for byte and the
/// shard's string metadata; a shard left empty is deleted. Returns how many tensors it dropped (0: the file is untouched).
pub fn rewriteWithout(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, file: []const u8, ctx: anytype, comptime drop: fn (@TypeOf(ctx), []const u8) bool) !usize {
    var h = try readHeader(allocator, io, dir, file);
    defer h.parsed.deinit();
    var dropped: usize = 0;
    var it = h.root.iterator();
    while (it.next()) |e| {
        if (!std.mem.eql(u8, e.key_ptr.*, "__metadata__") and drop(ctx, e.key_ptr.*)) dropped += 1;
    }
    if (dropped == 0) return 0;
    var f = try dir.openFile(io, file, .{});
    defer f.close(io);
    var b = pack.ShardBuilder.init(allocator);
    defer b.deinit();
    var meta = std.ArrayList([2][]const u8).empty;
    defer meta.deinit(allocator);
    it = h.root.iterator();
    while (it.next()) |e| {
        const v = e.value_ptr.*;
        if (std.mem.eql(u8, e.key_ptr.*, "__metadata__")) {
            if (v != .object) return error.BadSafetensorsHeader;
            var mt = v.object.iterator();
            while (mt.next()) |m| if (m.value_ptr.* == .string) try meta.append(allocator, .{ m.key_ptr.*, m.value_ptr.string });
            continue;
        }
        if (drop(ctx, e.key_ptr.*)) continue;
        const t = try Tensor.of(allocator, v);
        defer t.deinit(allocator);
        const bytes = try allocator.alloc(u8, @intCast(t.hi - t.lo));
        defer allocator.free(bytes);
        if (try f.readPositionalAll(io, bytes, 8 + h.len + t.lo) != bytes.len) return error.TruncatedSafetensors;
        try b.add(e.key_ptr.*, t.dtype, t.shape, bytes);
    }
    if (b.count == 0) try dir.deleteFile(io, file) else try b.flush(io, dir, file, meta.items);
    return dropped;
}

pub fn readHeader(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, name: []const u8) !Header {
    var f = try dir.openFile(io, name, .{});
    defer f.close(io);
    var len_buf: [8]u8 = undefined;
    if (try f.readPositionalAll(io, &len_buf, 0) != 8) return error.TruncatedSafetensors;
    const len = std.mem.readInt(u64, &len_buf, .little);
    if (len > 1 << 30) return error.TruncatedSafetensors;
    const raw = try allocator.alloc(u8, len);
    defer allocator.free(raw);
    if (try f.readPositionalAll(io, raw, 8) != len) return error.TruncatedSafetensors;
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, raw, .{});
    if (parsed.value != .object) {
        parsed.deinit();
        return error.TruncatedSafetensors;
    }
    return .{ .parsed = parsed, .len = len, .root = parsed.value.object };
}

fn routerLayer(name: []const u8) ?u32 {
    const head = "language_model.model.layers.";
    const tail = ".mlp.gate.weight";
    if (!std.mem.startsWith(u8, name, head) or !std.mem.endsWith(u8, name, tail) or name.len <= head.len + tail.len) return null;
    return std.fmt.parseInt(u32, name[head.len .. name.len - tail.len], 10) catch null;
}

fn itemSize(dtype: []const u8) ?usize {
    if (std.mem.eql(u8, dtype, "BF16") or std.mem.eql(u8, dtype, "F16")) return 2;
    if (std.mem.eql(u8, dtype, "F32")) return 4;
    return null;
}

/// Rewrites the router `name` of the shard `file` so its row i is the source's row `perm[i]`.
fn permuteRouter(allocator: std.mem.Allocator, io: std.Io, out: std.Io.Dir, file: []const u8, header_len: u64, entry: std.json.ObjectMap, perm: []const u32) !void {
    const dtype = entry.get("dtype") orelse return error.TrunkRouterInvalid;
    const shape = entry.get("shape") orelse return error.TrunkRouterInvalid;
    const offsets = entry.get("data_offsets") orelse return error.TrunkRouterInvalid;
    if (dtype != .string or shape != .array or offsets != .array or shape.array.items.len != 2 or offsets.array.items.len != 2) return error.TrunkRouterInvalid;
    const item = itemSize(dtype.string) orelse return error.TrunkRouterInvalid;
    if (shape.array.items[0] != .integer or shape.array.items[1] != .integer or offsets.array.items[0] != .integer) return error.TrunkRouterInvalid;
    const rows = intOf(usize, shape.array.items[0]) orelse return error.TrunkRouterInvalid;
    const row_bytes = (intOf(usize, shape.array.items[1]) orelse return error.TrunkRouterInvalid) * item;
    if (rows != perm.len) return error.TrunkRouterInvalid;
    const at = 8 + header_len + (intOf(u64, offsets.array.items[0]) orelse return error.TrunkRouterInvalid);
    var f = try out.openFile(io, file, .{ .mode = .read_write });
    defer f.close(io);
    const old = try allocator.alloc(u8, rows * row_bytes);
    defer allocator.free(old);
    if (try f.readPositionalAll(io, old, at) != old.len) return error.TruncatedSafetensors;
    const new = try allocator.alloc(u8, old.len);
    defer allocator.free(new);
    for (perm, 0..) |src_row, i| {
        if (src_row >= rows) return error.TrunkRouterInvalid;
        @memcpy(new[i * row_bytes ..][0..row_bytes], old[src_row * row_bytes ..][0..row_bytes]);
    }
    try f.writePositionalAll(io, new, at);
}

/// Copies `from`'s shards that hold no routed-expert bank, and its n-gram table, into `out`; `perms[L]`, where
/// set, reorders layer L's router. `skip_rerounded` leaves its re-rounded `trunk-*` shards behind, for a run that
/// rounds the trunk itself. Returns the shards copied.
pub fn copy(allocator: std.mem.Allocator, io: std.Io, from: std.Io.Dir, out: std.Io.Dir, perms: []const ?[]const u32, ngram: bool, skip_rerounded: bool) !usize {
    var names = std.ArrayList([]u8).empty;
    defer {
        for (names.items) |n| allocator.free(n);
        names.deinit(allocator);
    }
    var it = from.iterate();
    while (try it.next(io)) |e| {
        if (e.kind != .file or e.name[0] == '.' or !std.mem.endsWith(u8, e.name, ".safetensors")) continue;
        if (skip_rerounded and layer_stage.isRerounded(e.name)) continue;
        const name = try allocator.dupe(u8, e.name);
        errdefer allocator.free(name);
        try names.append(allocator, name);
    }
    var copied: usize = 0;
    for (names.items) |name| {
        var h = try readHeader(allocator, io, from, name);
        defer h.parsed.deinit();
        var experts: usize = 0;
        var tensors: usize = 0;
        var tit = h.root.iterator();
        while (tit.next()) |t| {
            if (std.mem.eql(u8, t.key_ptr.*, "__metadata__")) continue;
            tensors += 1;
            if (std.mem.indexOf(u8, t.key_ptr.*, ".switch_mlp.") != null) experts += 1;
        }
        if (experts != 0) {
            if (experts != tensors) return error.TrunkShardMixed;
            continue;
        }
        try cloneOrCopy(io, from, out, name);
        copied += 1;
        tit = h.root.iterator();
        while (tit.next()) |t| {
            const layer = routerLayer(t.key_ptr.*) orelse continue;
            if (layer >= perms.len) continue;
            const perm = perms[layer] orelse continue;
            if (t.value_ptr.* != .object) return error.TrunkRouterInvalid;
            try permuteRouter(allocator, io, out, name, h.len, t.value_ptr.object, perm);
        }
    }
    if (ngram) try cloneOrCopy(io, from, out, "ngram_table.bin");
    return copied;
}

test "trunk from: the served settings parse, and a quantization block that is not affine is refused" {
    const t = std.testing;
    var s = try parse(t.allocator,
        \\{"quantization":{"group_size":64,"bits":8,"mode":"affine",
        \\ "language_model.model.layers.0.linear_attn.in_proj_a":{"bits":8,"group_size":128,"mode":"affine"}},
        \\ "ngram_table":{"file":"ngram_table.bin","bits":4,"group_size":32}}
    );
    defer s.deinit();
    try t.expectEqual(@as(u32, 8), s.bits);
    try t.expectEqual(@as(u32, 64), s.group);
    try t.expectEqual(@as(usize, 1), s.overrides.len);
    try t.expectEqualStrings("language_model.model.layers.0.linear_attn.in_proj_a", s.overrides[0].base);
    try t.expectEqual(@as(u32, 128), s.overrides[0].group);
    try t.expectEqual(@as(u32, 4), s.ngram.?.bits);
    try t.expectError(error.TrunkConfigInvalid, parse(t.allocator, "{\"quantization\":{\"bits\":8,\"group_size\":64,\"x\":{\"bits\":4,\"group_size\":32,\"mode\":\"mxfp4\"}}}"));
    try t.expectError(error.TrunkConfigInvalid, parse(t.allocator, "{}"));
}

test "trunk from: a pack that serves its n-gram table unquantized (BF16, group 0) parses; group 0 at fewer bits is refused" {
    const t = std.testing;
    const q = "{\"quantization\":{\"group_size\":64,\"bits\":8,\"mode\":\"affine\"},\"ngram_table\":";
    var s = try parse(t.allocator, q ++ "{\"file\":\"ngram_table.bin\",\"bits\":16,\"group_size\":0}}");
    defer s.deinit();
    try t.expectEqual(pack.NgramSpec{ .bits = 16, .group = 0 }, s.ngram.?);
    inline for (.{ "{\"bits\":4,\"group_size\":0}}", "{\"bits\":16,\"group_size\":32}}", "{\"bits\":16}}", "{\"bits\":0,\"group_size\":0}}" }) |bad| {
        try t.expectError(error.TrunkConfigInvalid, parse(t.allocator, q ++ bad));
    }
}

const TestTensor = struct { name: []const u8, dtype: []const u8, shape: []const usize, bytes: []const u8 };

fn writeTensors(io: std.Io, dir: std.Io.Dir, file: []const u8, tensors: []const TestTensor) !void {
    var b = pack.ShardBuilder.init(std.testing.allocator);
    defer b.deinit();
    for (tensors) |x| try b.add(x.name, x.dtype, x.shape, x.bytes);
    try b.flush(io, dir, file, &.{});
}

test "trunk from: expert shards stay behind, other shards copy byte for byte, and a planned layer's router rows follow its groups" {
    const t = std.testing;
    const io = t.io;
    var from = t.tmpDir(.{ .iterate = true });
    defer from.cleanup();
    var out = t.tmpDir(.{ .iterate = true });
    defer out.cleanup();
    const router = [_]u16{ 10, 11, 20, 21, 30, 31, 40, 41 };
    const other = [_]u16{ 7, 8, 9, 6 };
    const router1 = [_]u16{ 50, 51, 52, 53, 54, 55, 56, 57 };
    try writeTensors(io, from.dir, "model-trunk.safetensors", &.{
        .{ .name = "language_model.model.layers.0.mlp.gate.weight", .dtype = "BF16", .shape = &.{ 4, 2 }, .bytes = std.mem.sliceAsBytes(&router) },
        .{ .name = "language_model.model.layers.1.mlp.gate.weight", .dtype = "BF16", .shape = &.{ 4, 2 }, .bytes = std.mem.sliceAsBytes(&router1) },
        .{ .name = "language_model.model.norm.weight", .dtype = "BF16", .shape = &.{4}, .bytes = std.mem.sliceAsBytes(&other) },
    });
    try writeTensors(io, from.dir, "model-exl3-L00.safetensors", &.{
        .{ .name = "language_model.model.layers.0.mlp.switch_mlp.up_proj.suh", .dtype = "F16", .shape = &.{4}, .bytes = std.mem.sliceAsBytes(&other) },
    });
    try from.dir.writeFile(io, .{ .sub_path = "ngram_table.bin", .data = "ngram" });
    const perm = [_]u32{ 2, 0, 3, 1 };
    const perms = [_]?[]const u32{ &perm, null };
    try t.expectEqual(@as(usize, 1), try copy(t.allocator, io, from.dir, out.dir, &perms, true, false));
    try t.expectError(error.FileNotFound, out.dir.access(io, "model-exl3-L00.safetensors", .{}));
    const want = try from.dir.readFileAlloc(io, "model-trunk.safetensors", t.allocator, .limited(1 << 20));
    defer t.allocator.free(want);
    const got = try out.dir.readFileAlloc(io, "model-trunk.safetensors", t.allocator, .limited(1 << 20));
    defer t.allocator.free(got);
    try t.expectEqual(want.len, got.len);
    const permuted = [_]u16{ 30, 31, 10, 11, 40, 41, 20, 21 };
    const at = std.mem.indexOf(u8, want, std.mem.sliceAsBytes(&router)).?;
    try t.expectEqualSlices(u8, std.mem.sliceAsBytes(&permuted), got[at..][0..16]);
    try t.expectEqualSlices(u8, want[0..at], got[0..at]);
    try t.expectEqualSlices(u8, want[at + 16 ..], got[at + 16 ..]);
    const ng = try out.dir.readFileAlloc(io, "ngram_table.bin", t.allocator, .limited(16));
    defer t.allocator.free(ng);
    try t.expectEqualStrings("ngram", ng);
    const mixed = [_]u16{ 1, 2 };
    try writeTensors(io, from.dir, "model-mixed.safetensors", &.{
        .{ .name = "language_model.model.layers.0.mlp.switch_mlp.up_proj.svh", .dtype = "F16", .shape = &.{2}, .bytes = std.mem.sliceAsBytes(&mixed) },
        .{ .name = "language_model.model.layers.0.input_layernorm.weight", .dtype = "BF16", .shape = &.{2}, .bytes = std.mem.sliceAsBytes(&mixed) },
    });
    try t.expectError(error.TrunkShardMixed, copy(t.allocator, io, from.dir, out.dir, &perms, false, false));
}

test "trunk from: a pack written with a rate plan, by its config or by its group banks, is refused as a trunk source" {
    const t = std.testing;
    const io = t.io;
    const w = [_]u16{ 1, 2 };
    const plain = "{\"quantization\":{\"group_size\":64,\"bits\":8},\"expert_quant\":{\"k\":\"2.5\"}}";
    var clean = t.tmpDir(.{ .iterate = true });
    defer clean.cleanup();
    try writeTensors(io, clean.dir, "layer-0.safetensors", &.{
        .{ .name = "language_model.model.layers.0.mlp.switch_mlp.up_proj.trellis", .dtype = "U16", .shape = &.{2}, .bytes = std.mem.sliceAsBytes(&w) },
    });
    try refuseRateGrouped(t.allocator, io, clean.dir, plain);
    try t.expectError(error.TrunkFromRateGrouped, refuseRateGrouped(t.allocator, io, clean.dir, "{\"expert_quant\":{\"rate_plan_sha256\":\"ab\"}}"));

    var grouped = t.tmpDir(.{ .iterate = true });
    defer grouped.cleanup();
    try writeTensors(io, grouped.dir, "layer-0.safetensors", &.{
        .{ .name = "language_model.model.layers.0.mlp.switch_mlp.up_proj.g1.suh", .dtype = "U16", .shape = &.{2}, .bytes = std.mem.sliceAsBytes(&w) },
    });
    try t.expectError(error.TrunkFromRateGrouped, refuseRateGrouped(t.allocator, io, grouped.dir, plain));
    try t.expect(!isGroupBank("language_model.model.layers.0.mlp.switch_mlp.up_proj.trellis"));
    try t.expect(!isGroupBank("language_model.model.layers.0.self_attn.g1.weight"));
    try t.expect(isGroupBank("language_model.mtp.layers.0.mlp.switch_mlp.down_proj.g12.trellis"));
}

test "trunk from: a run that rounds the trunk itself leaves the pack's re-rounded shards behind" {
    const t = std.testing;
    const io = t.io;
    var from = t.tmpDir(.{ .iterate = true });
    defer from.cleanup();
    var out = t.tmpDir(.{ .iterate = true });
    defer out.cleanup();
    const w = [_]u16{ 1, 2 };
    for ([_][]const u8{ "model-trunk.safetensors", "trunk-L00.safetensors", "trunk-head.safetensors" }) |name| {
        try writeTensors(io, from.dir, name, &.{.{ .name = name, .dtype = "BF16", .shape = &.{2}, .bytes = std.mem.sliceAsBytes(&w) }});
    }
    try t.expectEqual(@as(usize, 1), try copy(t.allocator, io, from.dir, out.dir, &.{}, false, true));
    try out.dir.access(io, "model-trunk.safetensors", .{});
    try t.expectError(error.FileNotFound, out.dir.access(io, "trunk-head.safetensors", .{}));
    try t.expectEqual(@as(usize, 3), try copy(t.allocator, io, from.dir, out.dir, &.{}, false, false));
    try out.dir.access(io, "trunk-head.safetensors", .{});
}

test "trunk from: parse frees the duplicated config when it spills into a new arena buffer" {
    const t = std.testing;
    const pad = try t.allocator.alloc(u8, 16384);
    defer t.allocator.free(pad);
    @memset(pad, ' ');
    const text = try std.fmt.allocPrint(t.allocator, "{{\"quantization\":{{\"group_size\":64,\"bits\":8,\"mode\":\"affine\"}}}}{s}", .{pad});
    defer t.allocator.free(text);
    var s = try parse(t.allocator, text);
    defer s.deinit();
    try t.expectEqualStrings(text, s.config);
}
