//! Native Zig safetensors reader and writer, directly to and from Sushi Cooler Metal buffers (`mlx.engine.hostFilled`)
//! and MLX arrays (`mlx.newData`), with zero `mlx_load_safetensors`, `mlx_save_safetensors`, or `mlx_map_string_to_*`
//! calls. Header encoding reuses `disk_format.encodeHeader`.

const std = @import("std");
const mlx = @import("mlx.zig");
const disk_format = @import("disk_format.zig");

const Allocator = std.mem.Allocator;

pub const MetaPair = disk_format.MetaPair;
pub const MAX_HEADER_BYTES: u64 = 128 << 20;

/// One named array to write into a safetensors file or byte image.
pub const Tensor = struct { key: []const u8, arr: mlx.mlx_array };

/// One tensor's metadata as a streaming reader (`Sink`) hands it to its caller.
pub const Named = struct { key: []const u8, dtype: mlx.mlx_dtype, shape: []const c_int };

/// Where a streaming reader writes each tensor: a slice of exactly its byte length, or null to skip it.
/// `meta`, when set, runs once before any tensor is visited.
pub const Sink = struct {
    ctx: *anyopaque,
    dest: *const fn (ctx: *anyopaque, t: Named) anyerror!?[]u8,
    meta: ?*const fn (ctx: *anyopaque, meta: []const MetaPair) anyerror!void = null,

    pub fn begin(sink: Sink, meta: []const MetaPair) !void {
        if (sink.meta) |f| try f(sink.ctx, meta);
    }

    pub fn take(sink: Sink, t: Named, nbytes: usize) !?[]u8 {
        const out = (try sink.dest(sink.ctx, t)) orelse return null;
        if (out.len != nbytes) return error.SafetensorsCorrupt;
        return out;
    }
};

/// One tensor entry parsed from a safetensors JSON header.
pub const HeaderEntry = struct {
    key: []const u8,
    dtype: mlx.mlx_dtype,
    shape: []c_int,
    /// Byte offset relative to the start of the tensor data section (after the 8-byte len + header JSON).
    offset: u64,
    nbytes: usize,

    pub fn named(self: *const HeaderEntry) Named {
        return .{ .key = self.key, .dtype = self.dtype, .shape = self.shape };
    }
};

/// A parsed safetensors header allocated in the caller's arena.
pub const ParsedHeader = struct {
    data_start: usize,
    entries: []HeaderEntry,
    meta: []MetaPair,
};

/// Native Zig replacement for `mlx_map_string_to_array` + `mlx_map_string_to_string`.
/// Preserves tensor insertion order; owns every array handle inside `tensors` and all strings in `arena`.
pub const Loaded = struct {
    allocator: Allocator,
    arena: std.heap.ArenaAllocator,
    tensors: std.StringArrayHashMapUnmanaged(mlx.mlx_array) = .empty,
    meta: std.StringArrayHashMapUnmanaged([]const u8) = .empty,

    pub fn init(allocator: Allocator) Loaded {
        return .{
            .allocator = allocator,
            .arena = std.heap.ArenaAllocator.init(allocator),
        };
    }

    pub fn deinit(self: *Loaded) void {
        for (self.tensors.values()) |arr| {
            if (arr.ctx != null) _ = mlx.mlx_array_free(arr);
        }
        self.tensors.deinit(self.allocator);
        self.meta.deinit(self.allocator);
        self.arena.deinit();
    }

    pub fn count(self: *const Loaded) usize {
        return self.tensors.count();
    }

    /// Borrowed handle for `key`, or null if absent (or already taken via `take`).
    pub fn get(self: *const Loaded, key: []const u8) ?mlx.mlx_array {
        const arr = self.tensors.get(key) orelse return null;
        return if (arr.ctx != null) arr else null;
    }

    /// Transfers ownership of `key`'s handle to the caller without an extra refcount bump; returns null if absent.
    pub fn take(self: *Loaded, key: []const u8) ?mlx.mlx_array {
        const ptr = self.tensors.getPtr(key) orelse return null;
        if (ptr.ctx == null) return null;
        const arr = ptr.*;
        ptr.* = .{};
        return arr;
    }

    /// Returns a new owned (+1 ref) handle for `key`, or null if absent.
    pub fn getOwned(self: *const Loaded, key: []const u8) !?mlx.mlx_array {
        const arr = self.get(key) orelse return null;
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_array_set(&out, arr));
        return out;
    }

    /// Adopts ownership of `arr` under `key` (frees any prior handle at `key`).
    pub fn put(self: *Loaded, key: []const u8, arr: mlx.mlx_array) !void {
        if (self.tensors.getPtr(key)) |slot| {
            if (slot.ctx != null) _ = mlx.mlx_array_free(slot.*);
            slot.* = arr;
            return;
        }
        try self.tensors.ensureUnusedCapacity(self.allocator, 1);
        const duped = try self.arena.allocator().dupe(u8, key);
        self.tensors.putAssumeCapacity(duped, arr);
    }

    /// Retains (+1 ref) `arr` under `key`.
    pub fn putRetained(self: *Loaded, key: []const u8, arr: mlx.mlx_array) !void {
        var held = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(held);
        try mlx.check(mlx.mlx_array_set(&held, arr));
        try self.put(key, held);
    }

    pub fn putMeta(self: *Loaded, key: []const u8, value: []const u8) !void {
        const aa = self.arena.allocator();
        const val = try aa.dupe(u8, value);
        if (self.meta.getPtr(key)) |slot| {
            slot.* = val;
            return;
        }
        try self.meta.ensureUnusedCapacity(self.allocator, 1);
        const k = try aa.dupe(u8, key);
        self.meta.putAssumeCapacity(k, val);
    }

    pub fn getMeta(self: *const Loaded, key: []const u8) ?[]const u8 {
        return self.meta.get(key);
    }

    /// Writes all live tensors and metadata in `self` directly to `path`.
    pub fn writeFile(self: *const Loaded, io: std.Io, path: []const u8, s: mlx.mlx_stream) !u64 {
        var tensors: std.ArrayList(Tensor) = .empty;
        defer tensors.deinit(self.allocator);
        try tensors.ensureTotalCapacity(self.allocator, self.tensors.count());
        var tit = self.tensors.iterator();
        while (tit.next()) |e| {
            if (e.value_ptr.ctx == null) continue;
            tensors.appendAssumeCapacity(.{ .key = e.key_ptr.*, .arr = e.value_ptr.* });
        }
        var meta: std.ArrayList(MetaPair) = .empty;
        defer meta.deinit(self.allocator);
        try meta.ensureTotalCapacity(self.allocator, self.meta.count());
        var mit = self.meta.iterator();
        while (mit.next()) |e| {
            meta.appendAssumeCapacity(.{ .key = e.key_ptr.*, .value = e.value_ptr.* });
        }
        return saveFile(self.allocator, io, path, tensors.items, meta.items, s);
    }
};

pub fn dtypeName(d: mlx.mlx_dtype) ?[]const u8 {
    return switch (d) {
        .bool_ => "BOOL",
        .uint8 => "U8",
        .uint16 => "U16",
        .uint32 => "U32",
        .uint64 => "U64",
        .int8 => "I8",
        .int16 => "I16",
        .int32 => "I32",
        .int64 => "I64",
        .float16 => "F16",
        .float32 => "F32",
        .bfloat16 => "BF16",
        .complex64 => "C64",
        // .float64 is intentionally refused (unsupported on Metal GPU).
        else => null,
    };
}

/// Safetensors dtype string to `mlx_dtype` (matches MLX: FP8 codes load as `uint8`).
pub fn dtypeFromName(name: []const u8) ?mlx.mlx_dtype {
    const table = [_]struct { []const u8, mlx.mlx_dtype }{
        .{ "BOOL", .bool_ },       .{ "U8", .uint8 },      .{ "U16", .uint16 },
        .{ "U32", .uint32 },       .{ "U64", .uint64 },    .{ "I8", .int8 },
        .{ "I16", .int16 },        .{ "I32", .int32 },     .{ "I64", .int64 },
        .{ "F16", .float16 },      .{ "BF16", .bfloat16 }, .{ "F32", .float32 },
        .{ "F64", .float64 },      .{ "C64", .complex64 }, .{ "F8_E4M3", .uint8 },
        .{ "F8_E8M0", .uint8 },
    };
    for (table) |row| if (std.mem.eql(u8, row[0], name)) return row[1];
    return null;
}

pub fn itemSize(d: mlx.mlx_dtype) usize {
    return switch (d) {
        .bool_, .uint8, .int8 => 1,
        .uint16, .int16, .float16, .bfloat16 => 2,
        .uint32, .int32, .float32 => 4,
        .uint64, .int64, .float64, .complex64 => 8,
    };
}

pub fn nbytesOf(a: mlx.mlx_array) u64 {
    return @as(u64, mlx.mlx_array_size(a)) * @as(u64, mlx.mlx_array_itemsize(a));
}

/// Raw bytes of an evaluated, row-contiguous array (Cooler tensor or MLX array, any stored dtype).
pub fn bytesOf(arr: mlx.mlx_array) ?[]const u8 {
    const n: usize = @intCast(nbytesOf(arr));
    if (n == 0) return &.{};
    const p = mlx.mlx_array_data_uint8(arr) orelse return null;
    return p[0..n];
}

const HeaderSource = struct {
    tensors: []const Tensor,

    pub fn len(self: HeaderSource) usize {
        return self.tensors.len;
    }

    pub fn desc(self: HeaderSource, i: usize) !disk_format.TensorDesc {
        const t = self.tensors[i];
        return .{
            .key = t.key,
            .dtype = dtypeName(mlx.mlx_array_dtype(t.arr)) orelse return error.DiskCacheBadDtype,
            .shape = mlx.getShape(t.arr),
            .nbytes = nbytesOf(t.arr),
        };
    }
};

pub fn encodeHeader(allocator: Allocator, tensors: []const Tensor, meta: []const MetaPair) ![]u8 {
    return disk_format.encodeHeader(allocator, HeaderSource{ .tensors = tensors }, meta);
}

/// Parses a safetensors header (`head` = 8-byte little-endian length prefix + header JSON) in `aa`.
pub fn parseHeader(aa: Allocator, head: []const u8) !ParsedHeader {
    if (head.len < 8) return error.SafetensorsCorrupt;
    const hlen = std.mem.readInt(u64, head[0..8], .little);
    if (hlen != head.len - 8) return error.SafetensorsCorrupt;
    const root = std.json.parseFromSliceLeaky(std.json.Value, aa, head[8..], .{ .allocate = .alloc_always }) catch return error.SafetensorsCorrupt;
    if (root != .object) return error.SafetensorsCorrupt;

    var entries: std.ArrayList(HeaderEntry) = .empty;
    var meta: std.ArrayList(MetaPair) = .empty;
    var it = root.object.iterator();
    while (it.next()) |kv| {
        if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) {
            if (kv.value_ptr.* == .null) continue;
            if (kv.value_ptr.* != .object) return error.SafetensorsCorrupt;
            var mit = kv.value_ptr.object.iterator();
            while (mit.next()) |m| {
                if (m.value_ptr.* != .string) return error.SafetensorsCorrupt;
                try meta.append(aa, .{ .key = m.key_ptr.*, .value = m.value_ptr.string });
            }
            continue;
        }
        const obj = kv.value_ptr.*;
        if (obj != .object) return error.SafetensorsCorrupt;
        const dt = obj.object.get("dtype") orelse return error.SafetensorsCorrupt;
        const sh = obj.object.get("shape") orelse return error.SafetensorsCorrupt;
        const offs = obj.object.get("data_offsets") orelse return error.SafetensorsCorrupt;
        if (dt != .string or sh != .array or offs != .array or offs.array.items.len != 2) return error.SafetensorsCorrupt;
        const dtype = dtypeFromName(dt.string) orelse return error.SafetensorsCorrupt;
        const shape = try aa.alloc(c_int, sh.array.items.len);
        var nbytes: u64 = itemSize(dtype);
        for (sh.array.items, shape) |d, *slot| {
            if (d != .integer) return error.SafetensorsCorrupt;
            const v = std.math.cast(c_int, d.integer) orelse return error.SafetensorsCorrupt;
            if (v < 0) return error.SafetensorsCorrupt;
            slot.* = v;
            nbytes = std.math.mul(u64, nbytes, @intCast(v)) catch return error.SafetensorsCorrupt;
        }
        const lo_v = offs.array.items[0];
        const hi_v = offs.array.items[1];
        if (lo_v != .integer or hi_v != .integer or lo_v.integer < 0 or hi_v.integer < lo_v.integer) return error.SafetensorsCorrupt;
        const lo: u64 = @intCast(lo_v.integer);
        const hi: u64 = @intCast(hi_v.integer);
        if (hi - lo != nbytes) return error.SafetensorsCorrupt;
        const nbytes_usize = std.math.cast(usize, nbytes) orelse return error.SafetensorsCorrupt;
        try entries.append(aa, .{
            .key = kv.key_ptr.*,
            .dtype = dtype,
            .shape = shape,
            .offset = lo,
            .nbytes = nbytes_usize,
        });
    }
    return .{ .data_start = head.len, .entries = entries.items, .meta = meta.items };
}

/// Allocates a Cooler Metal-backed tensor (when `s` is Cooler's stream) or an MLX array (otherwise) and fills its bytes
/// through `fill(ctx, dst)`. On Cooler, `fill` writes straight into the `metal.Buffer` with zero intermediate copy.
pub fn makeArrayFilled(
    allocator: Allocator,
    dtype: mlx.mlx_dtype,
    shape: []const c_int,
    nbytes: usize,
    s: mlx.mlx_stream,
    ctx: anytype,
    comptime fill: fn (@TypeOf(ctx), []u8) anyerror!void,
) !mlx.mlx_array {
    if (mlx.engine.isStream(s)) {
        var dst: []u8 = undefined;
        const arr = mlx.engine.hostFilled(dtype, shape, &dst) orelse return error.MlxError;
        errdefer _ = mlx.mlx_array_free(arr);
        if (dst.len != nbytes) return error.SafetensorsCorrupt;
        try fill(ctx, dst);
        return arr;
    }
    const mk = mlx.engine.makers(false);
    defer mk.end();
    if (nbytes == 0) {
        var dummy: u8 = 0;
        try fill(ctx, &.{});
        return mlx.newData(&dummy, shape, dtype) catch return error.MlxError;
    }
    const buf = try allocator.alloc(u8, nbytes);
    defer allocator.free(buf);
    try fill(ctx, buf);
    return mlx.newData(buf.ptr, shape, dtype) catch return error.MlxError;
}

/// Batched `mlx_eval` calls performed by `materialize`, for regression tests.
pub var eval_count: std.atomic.Value(u64) = .init(0);

/// Makes every tensor row-contiguous and evaluates the batch in one `mlx_eval`. Caller frees each element and the slice.
pub fn materialize(allocator: Allocator, tensors: []const Tensor, s: mlx.mlx_stream) ![]mlx.mlx_array {
    const out = try allocator.alloc(mlx.mlx_array, tensors.len);
    var made: usize = 0;
    errdefer {
        for (out[0..made]) |a| _ = mlx.mlx_array_free(a);
        allocator.free(out);
    }
    const scope = mlx.engine.enter(s);
    defer scope.end();
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (tensors, out) |t, *slot| {
        slot.* = mlx.mlx_array_new();
        made += 1;
        try mlx.check(mlx.mlx_contiguous(slot, t.arr, false, s));
        try mlx.check(mlx.mlx_vector_array_append_value(vec, slot.*));
    }
    if (tensors.len > 0) {
        _ = eval_count.fetchAdd(1, .monotonic);
        try mlx.check(mlx.mlx_eval(vec));
    }
    return out;
}

pub fn freeArrays(allocator: Allocator, arrs: []mlx.mlx_array) void {
    for (arrs) |a| _ = mlx.mlx_array_free(a);
    allocator.free(arrs);
}

/// Serializes already-contiguous, evaluated `tensors` and `meta` into one safetensors byte buffer without evaluating.
pub fn encodeMaterialized(allocator: Allocator, tensors: []const Tensor, meta: []const MetaPair) ![]u8 {
    const head = try encodeHeader(allocator, tensors, meta);
    defer allocator.free(head);
    var total: usize = head.len;
    for (tensors) |t| total = std.math.add(usize, total, @intCast(nbytesOf(t.arr))) catch return error.OutOfMemory;
    const out = try allocator.alloc(u8, total);
    errdefer allocator.free(out);
    @memcpy(out[0..head.len], head);
    var at: usize = head.len;
    for (tensors) |t| {
        const nb: usize = @intCast(nbytesOf(t.arr));
        if (nb == 0) continue;
        const src = bytesOf(t.arr) orelse return error.DiskCacheUnreadable;
        @memcpy(out[at..][0..nb], src);
        at += nb;
    }
    return out;
}

/// Serializes `tensors` and `meta` into one safetensors byte buffer.
pub fn serialize(allocator: Allocator, tensors: []const Tensor, meta: []const MetaPair, s: mlx.mlx_stream) ![]u8 {
    const conts = try materialize(allocator, tensors, s);
    defer freeArrays(allocator, conts);
    const views = try allocator.alloc(Tensor, tensors.len);
    defer allocator.free(views);
    for (tensors, conts, views) |t, c, *v| v.* = .{ .key = t.key, .arr = c };
    return encodeMaterialized(allocator, views, meta);
}

/// Writes `tensors` and `meta` directly to `abs_path` without double-buffering the full payload. Returns total bytes.
pub fn saveFile(allocator: Allocator, io: std.Io, abs_path: []const u8, tensors: []const Tensor, meta: []const MetaPair, s: mlx.mlx_stream) !u64 {
    const conts = try materialize(allocator, tensors, s);
    defer freeArrays(allocator, conts);
    const views = try allocator.alloc(Tensor, tensors.len);
    defer allocator.free(views);
    for (tensors, conts, views) |t, c, *v| v.* = .{ .key = t.key, .arr = c };
    const head = try encodeHeader(allocator, views, meta);
    defer allocator.free(head);

    const f = try std.Io.Dir.cwd().createFile(io, abs_path, .{});
    defer f.close(io);
    var wb: [64 * 1024]u8 = undefined;
    var fw = f.writer(io, &wb);
    fw.interface.writeAll(head) catch |err| return fw.err orelse err;
    var total: u64 = head.len;
    for (conts) |c| {
        const nb: usize = @intCast(nbytesOf(c));
        if (nb == 0) continue;
        const src = bytesOf(c) orelse return error.DiskCacheUnreadable;
        fw.interface.writeAll(src) catch |err| return fw.err orelse err;
        total += nb;
    }
    fw.interface.flush() catch |err| return fw.err orelse err;
    return total;
}

/// Loads a safetensors file at `abs_path` directly into Cooler Metal buffers (on Cooler stream) or MLX arrays.
pub fn loadFile(allocator: Allocator, io: std.Io, abs_path: []const u8, s: mlx.mlx_stream) !Loaded {
    const f = try std.Io.Dir.cwd().openFile(io, abs_path, .{});
    defer f.close(io);
    const st = try f.stat(io);
    var word: [8]u8 = undefined;
    if (try f.readPositionalAll(io, &word, 0) != word.len) return error.SafetensorsCorrupt;
    const hlen = std.mem.readInt(u64, &word, .little);
    if (hlen > MAX_HEADER_BYTES or 8 + hlen > st.size) return error.SafetensorsCorrupt;

    var parse_arena = std.heap.ArenaAllocator.init(allocator);
    defer parse_arena.deinit();
    const pa = parse_arena.allocator();
    const head = try pa.alloc(u8, 8 + @as(usize, @intCast(hlen)));
    @memcpy(head[0..8], &word);
    if (try f.readPositionalAll(io, head[8..], 8) != hlen) return error.SafetensorsCorrupt;
    const parsed = try parseHeader(pa, head);
    const data_len = st.size - parsed.data_start;

    const scope = if (mlx.engine.isStream(s)) mlx.engine.enter(s) else null;
    defer if (scope) |sc| sc.end();
    var loaded = Loaded.init(allocator);
    errdefer loaded.deinit();

    const FileFill = struct {
        io: std.Io,
        f: std.Io.File,
        pos: u64,
        fn run(ctx: @This(), dst: []u8) anyerror!void {
            if (dst.len == 0) return;
            if (try ctx.f.readPositionalAll(ctx.io, dst, ctx.pos) != dst.len) return error.SafetensorsCorrupt;
        }
    };
    for (parsed.entries) |*e| {
        if (e.offset > data_len or e.nbytes > data_len - e.offset) return error.SafetensorsCorrupt;
        const arr = try makeArrayFilled(allocator, e.dtype, e.shape, e.nbytes, s, FileFill{
            .io = io,
            .f = f,
            .pos = parsed.data_start + e.offset,
        }, FileFill.run);
        errdefer _ = mlx.mlx_array_free(arr);
        try loaded.put(e.key, arr);
    }
    for (parsed.meta) |m| try loaded.putMeta(m.key, m.value);
    return loaded;
}

/// Loads an in-memory safetensors image directly into Cooler Metal buffers (on Cooler stream) or MLX arrays.
pub fn loadBytes(allocator: Allocator, bytes: []const u8, s: mlx.mlx_stream) !Loaded {
    if (bytes.len < 8) return error.SafetensorsCorrupt;
    const hlen = std.mem.readInt(u64, bytes[0..8], .little);
    if (hlen > bytes.len - 8) return error.SafetensorsCorrupt;

    var parse_arena = std.heap.ArenaAllocator.init(allocator);
    defer parse_arena.deinit();
    const pa = parse_arena.allocator();
    const parsed = try parseHeader(pa, bytes[0 .. 8 + @as(usize, @intCast(hlen))]);
    const data = bytes[parsed.data_start..];

    const scope = if (mlx.engine.isStream(s)) mlx.engine.enter(s) else null;
    defer if (scope) |sc| sc.end();
    var loaded = Loaded.init(allocator);
    errdefer loaded.deinit();

    const SliceFill = struct {
        src: []const u8,
        fn run(ctx: @This(), dst: []u8) anyerror!void {
            if (dst.len != 0) @memcpy(dst, ctx.src);
        }
    };
    for (parsed.entries) |*e| {
        if (e.offset > data.len or e.nbytes > data.len - e.offset) return error.SafetensorsCorrupt;
        const src = data[@intCast(e.offset)..][0..e.nbytes];
        const arr = try makeArrayFilled(allocator, e.dtype, e.shape, e.nbytes, s, SliceFill{ .src = src }, SliceFill.run);
        errdefer _ = mlx.mlx_array_free(arr);
        try loaded.put(e.key, arr);
    }
    for (parsed.meta) |m| try loaded.putMeta(m.key, m.value);
    return loaded;
}

/// Streams tensors from `abs_path` directly into caller-provided buffers (`sink`), making no arrays.
pub fn readFileInto(io: std.Io, abs_path: []const u8, aa: Allocator, sink: Sink) ![]const MetaPair {
    const f = try std.Io.Dir.cwd().openFile(io, abs_path, .{});
    defer f.close(io);
    const st = try f.stat(io);
    var word: [8]u8 = undefined;
    if (try f.readPositionalAll(io, &word, 0) != word.len) return error.SafetensorsCorrupt;
    const hlen = std.mem.readInt(u64, &word, .little);
    if (hlen > MAX_HEADER_BYTES or 8 + hlen > st.size) return error.SafetensorsCorrupt;
    const head = try aa.alloc(u8, 8 + @as(usize, @intCast(hlen)));
    @memcpy(head[0..8], &word);
    if (try f.readPositionalAll(io, head[8..], 8) != hlen) return error.SafetensorsCorrupt;
    const parsed = try parseHeader(aa, head);
    const data_len = st.size - parsed.data_start;
    try sink.begin(parsed.meta);
    for (parsed.entries) |*e| {
        if (e.offset > data_len or e.nbytes > data_len - e.offset) return error.SafetensorsCorrupt;
        const out = (try sink.take(e.named(), e.nbytes)) orelse continue;
        if (out.len > 0 and try f.readPositionalAll(io, out, parsed.data_start + e.offset) != out.len) return error.SafetensorsCorrupt;
    }
    return parsed.meta;
}

/// Streams tensors from an in-memory safetensors image directly into caller-provided buffers (`sink`).
pub fn readBytesInto(bytes: []const u8, aa: Allocator, sink: Sink) ![]const MetaPair {
    if (bytes.len < 8) return error.SafetensorsCorrupt;
    const hlen = std.mem.readInt(u64, bytes[0..8], .little);
    if (hlen > bytes.len - 8) return error.SafetensorsCorrupt;
    const parsed = try parseHeader(aa, bytes[0 .. 8 + @as(usize, @intCast(hlen))]);
    const data = bytes[parsed.data_start..];
    try sink.begin(parsed.meta);
    for (parsed.entries) |*e| {
        if (e.offset > data.len or e.nbytes > data.len - e.offset) return error.SafetensorsCorrupt;
        const out = (try sink.take(e.named(), e.nbytes)) orelse continue;
        if (out.len > 0) @memcpy(out, data[@intCast(e.offset)..][0..out.len]);
    }
    return parsed.meta;
}

// ── Tests ──

const testing = std.testing;

fn roundTripOnStream(s: mlx.mlx_stream) !void {
    const a = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const root = pbuf[0..try tmp.dir.realPath(io, &pbuf)];
    const path = try std.fmt.allocPrint(a, "{s}/sample.safetensors", .{root});
    defer a.free(path);

    const scope = if (mlx.engine.isStream(s)) mlx.engine.enter(s) else null;
    defer if (scope) |sc| sc.end();

    const f32_vals = [_]f32{ 1.5, -2.25, 3.0, 4.5, 5.125, -6.0 };
    const u8_vals = [_]u8{ 0, 1, 127, 255 };
    const i32_val: i32 = -42;
    const u16_vals = [_]u16{ 0x3f80, 0xbf80, 0x4000, 0x4040 }; // bf16 bits
    const bool_vals = [_]bool{ true, false, true };
    const u64_vals = [_]u64{ 1, 0xdeadbeefcafebabe };

    const a_f32 = try mlx.newData(&f32_vals, &.{ 2, 3 }, .float32);
    defer _ = mlx.mlx_array_free(a_f32);
    // Non-contiguous transpose view to verify materialization.
    var a_trans = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a_trans);
    try mlx.check(mlx.mlx_transpose_axes(&a_trans, a_f32, &[_]c_int{ 1, 0 }, 2, s));
    const a_u8 = try mlx.newData(&u8_vals, &.{4}, .uint8);
    defer _ = mlx.mlx_array_free(a_u8);
    const a_scalar = try mlx.newInt(i32_val);
    defer _ = mlx.mlx_array_free(a_scalar);
    const a_bf16 = try mlx.newData(&u16_vals, &.{ 2, 2 }, .bfloat16);
    defer _ = mlx.mlx_array_free(a_bf16);
    const a_bool = try mlx.newData(&bool_vals, &.{3}, .bool_);
    defer _ = mlx.mlx_array_free(a_bool);
    const a_u64 = try mlx.newData(&u64_vals, &.{2}, .uint64);
    defer _ = mlx.mlx_array_free(a_u64);
    var empty_byte: u8 = 0;
    const a_empty = try mlx.newData(&empty_byte, &.{ 0, 4 }, .float32);
    defer _ = mlx.mlx_array_free(a_empty);

    const tensors = [_]Tensor{
        .{ .key = "trans", .arr = a_trans },
        .{ .key = "u8", .arr = a_u8 },
        .{ .key = "scalar", .arr = a_scalar },
        .{ .key = "bf16", .arr = a_bf16 },
        .{ .key = "flag", .arr = a_bool },
        .{ .key = "u64", .arr = a_u64 },
        .{ .key = "empty", .arr = a_empty },
    };
    const meta = [_]MetaPair{ .{ .key = "layers", .value = "7" }, .{ .key = "tag", .value = "native" } };

    const written = try saveFile(a, io, path, &tensors, &meta, s);
    const serialized = try serialize(a, &tensors, &meta, s);
    defer a.free(serialized);
    try testing.expectEqual(@as(u64, serialized.len), written);

    const calls_before = mlx.engine.mlxCalls();
    var from_file = try loadFile(a, io, path, s);
    defer from_file.deinit();
    var from_bytes = try loadBytes(a, serialized, s);
    defer from_bytes.deinit();
    if (mlx.engine.isStream(s)) {
        try testing.expectEqual(calls_before, mlx.engine.mlxCalls());
    }

    for ([_]*Loaded{ &from_file, &from_bytes }) |loaded| {
        try testing.expectEqual(@as(usize, tensors.len), loaded.count());
        try testing.expectEqualStrings("7", loaded.getMeta("layers").?);
        try testing.expectEqualStrings("native", loaded.getMeta("tag").?);
        try testing.expect(loaded.getMeta("missing") == null);

        const gt = loaded.get("trans").?;
        try testing.expectEqual(mlx.engine.isStream(s), mlx.engine.isOurs(gt));
        try testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(gt));
        try testing.expectEqualSlices(c_int, &.{ 3, 2 }, mlx.getShape(gt));
        const want_t = [_]f32{ 1.5, 4.5, -2.25, 5.125, 3.0, -6.0 };
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&want_t), bytesOf(gt).?);

        try testing.expectEqualSlices(u8, &u8_vals, bytesOf(loaded.get("u8").?).?);
        try testing.expectEqualSlices(u8, std.mem.asBytes(&i32_val), bytesOf(loaded.get("scalar").?).?);
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&u16_vals), bytesOf(loaded.get("bf16").?).?);
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&bool_vals), bytesOf(loaded.get("flag").?).?);
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&u64_vals), bytesOf(loaded.get("u64").?).?);
        try testing.expectEqualSlices(c_int, &.{ 0, 4 }, mlx.getShape(loaded.get("empty").?));
    }

    // `take` transfers ownership without disturbing the other entries.
    const taken = from_file.take("u8").?;
    defer _ = mlx.mlx_array_free(taken);
    try testing.expect(from_file.get("u8") == null);
    try testing.expectEqualSlices(u8, &u8_vals, bytesOf(taken).?);

    // Streaming Sink reads only the requested tensor directly into caller bytes.
    const Into = struct {
        out: [4]u8 = @splat(0),
        layers: ?[]const u8 = null,
        fn onMeta(ctx: *anyopaque, pairs: []const MetaPair) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            for (pairs) |p| if (std.mem.eql(u8, p.key, "layers")) {
                self.layers = p.value;
            };
        }
        fn onDest(ctx: *anyopaque, t: Named) anyerror!?[]u8 {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            if (!std.mem.eql(u8, t.key, "u8")) return null;
            return &self.out;
        }
    };
    for (0..2) |arm| {
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        var into: Into = .{};
        const sink: Sink = .{ .ctx = &into, .dest = Into.onDest, .meta = Into.onMeta };
        _ = if (arm == 0)
            try readFileInto(io, path, arena.allocator(), sink)
        else
            try readBytesInto(serialized, arena.allocator(), sink);
        try testing.expectEqualStrings("7", into.layers.?);
        try testing.expectEqualSlices(u8, &u8_vals, &into.out);
    }
}

test "safetensors: native save/load round-trips all dtypes on both MLX and Sushi Cooler with zero MLX calls on Cooler" {
    try mlx.requireGpu();
    try roundTripOnStream(mlx.gpuStream());

    const was_enabled = mlx.engine.enabled;
    mlx.engine.enable();
    defer @import("test_support.zig").restoreEngine(was_enabled);
    try roundTripOnStream(mlx.engine.stream());

    // An MLX stream must still allocate MLX arrays even when called inside a Cooler scope.
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    const hdr_w = "{\"w\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}";
    var img: [8 + hdr_w.len + 8]u8 = undefined;
    std.mem.writeInt(u64, img[0..8], hdr_w.len, .little);
    @memcpy(img[8..][0..hdr_w.len], hdr_w);
    @memcpy(img[8 + hdr_w.len ..][0..8], std.mem.asBytes(&[_]f32{ 1.5, -2.5 }));
    const sc = mlx.engine.enter(mlx.engine.stream());
    defer sc.end();
    var on_cpu = try loadBytes(testing.allocator, &img, cpu);
    defer on_cpu.deinit();
    try testing.expect(!mlx.engine.isOurs(on_cpu.get("w").?));
}

test "safetensors: malformed headers, bad offsets, and truncated payloads are refused" {
    const a = testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    try testing.expectError(error.SafetensorsCorrupt, loadBytes(a, "short", s));
    // Header claims more bytes than exist.
    var bad_len = [_]u8{ 0xff, 0, 0, 0, 0, 0, 0, 0, '{', '}' };
    try testing.expectError(error.SafetensorsCorrupt, loadBytes(a, &bad_len, s));
    // Valid JSON header whose data_offsets exceed payload length.
    const hdr = "{\"w\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]}}";
    var buf: [8 + hdr.len + 4]u8 = undefined;
    std.mem.writeInt(u64, buf[0..8], hdr.len, .little);
    @memcpy(buf[8..][0..hdr.len], hdr);
    @memset(buf[8 + hdr.len ..], 0);
    try testing.expectError(error.SafetensorsCorrupt, loadBytes(a, &buf, s));
    // Mismatched byte length vs shape.
    const bad_span = "{\"w\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,8]}}";
    var buf2: [8 + bad_span.len + 8]u8 = undefined;
    std.mem.writeInt(u64, buf2[0..8], bad_span.len, .little);
    @memcpy(buf2[8..][0..bad_span.len], bad_span);
    @memset(buf2[8 + bad_span.len ..], 0);
    try testing.expectError(error.SafetensorsCorrupt, loadBytes(a, &buf2, s));

    // Null __metadata__ (as written by Python safetensors / MLX when metadata is absent) is accepted.
    const null_meta = "{\"__metadata__\":null,\"w\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}";
    var buf3: [8 + null_meta.len + 8]u8 = undefined;
    std.mem.writeInt(u64, buf3[0..8], null_meta.len, .little);
    @memcpy(buf3[8..][0..null_meta.len], null_meta);
    @memcpy(buf3[8 + null_meta.len ..][0..8], std.mem.asBytes(&[_]f32{ 1.5, -2.5 }));
    var ok = try loadBytes(a, &buf3, s);
    defer ok.deinit();
    try testing.expectEqual(@as(usize, 1), ok.count());
    try testing.expectEqual(@as(usize, 0), ok.meta.count());
    try testing.expectEqualSlices(u8, std.mem.asBytes(&[_]f32{ 1.5, -2.5 }), bytesOf(ok.get("w").?).?);
}
