const trf = @import("transformer.zig");
const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const io_util_mod = @import("io_util.zig");
const ModelConfig = model_mod.ModelConfig;
const diagEnvOn = io_util_mod.diagEnvOn;
const testing = std.testing;
const decodeProfReport = trf.decodeProfReport;

const DecodeProf = struct {
    calls: u64 = 0,
    embed_ns: u64 = 0,
    attn_ns: u64 = 0,
    mlp_ns: u64 = 0,
    lmhead_ns: u64 = 0,
    // MoE internals (summed across all MoE layers per token)
    moe_router_ns: u64 = 0,
    moe_experts_ns: u64 = 0,
    moe_shared_ns: u64 = 0,
};
/// One narrowed copy of a load-time constant table, keyed by dtype.
pub const DtypeCastCache = struct {
    arr: mlx.mlx_array = .{ .ctx = null },
    dtype: mlx.mlx_dtype = .float32,

    pub fn deinit(self: *DtypeCastCache) void {
        if (self.arr.ctx != null) _ = mlx.mlx_array_free(self.arr);
        self.arr = .{ .ctx = null };
    }
};

/// Hand back `table` in the dtype the activation it scales is in, building and
/// caching one narrowed copy on first use. Returns a BORROWED array — `cache`
/// owns it.
///
/// A constant table built f32 at load and multiplied straight into a bf16
/// activation promotes the PRODUCT to f32 (mlx's promotion rule). Where that
/// product feeds the residual stream, every projection and expert weight for
/// the rest of the forward is upcast on read — a ~3x token cost that shows up
/// in no op count, no kernel choice, and no pre-loop dtype log. Same class as
/// the MageFlow `scalarLike` rule.
pub fn constTableAs(table: mlx.mlx_array, want: mlx.mlx_dtype, cache: *DtypeCastCache, s: mlx.mlx_stream) !mlx.mlx_array {
    if (mlx.mlx_array_dtype(table) == want) return table;
    if (cache.arr.ctx != null and cache.dtype == want) return cache.arr;
    cache.deinit();
    var cast = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(cast);
    try mlx.check(mlx.mlx_astype(&cast, table, want, s));
    try mlx.check(mlx.mlx_array_eval(cast));
    cache.arr = cast;
    cache.dtype = want;
    return cast;
}

/// DIAGNOSTIC (SUSHI_LAYER_CAP): clamp the layer loop so a ms-vs-layers
/// sweep separates per-layer slope from fixed cost. Unset — the only shipped
/// state — returns the model's real depth.
var layer_cap_env: ?usize = null;
pub fn layerCap(n: usize) usize {
    if (layer_cap_env == null) {
        layer_cap_env = blk: {
            const raw = std.c.getenv("SUSHI_LAYER_CAP") orelse break :blk 0;
            break :blk std.fmt.parseInt(usize, std.mem.sliceTo(raw, 0), 10) catch 0;
        };
    }
    const cap = layer_cap_env.?;
    return if (cap == 0) n else @min(cap, n);
}

pub var decode_prof: DecodeProf = .{};
var decode_prof_enabled: ?bool = null;

var standin_ones: ?mlx.mlx_array = null;
pub fn standinOnes(shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    if (standin_ones) |a| {
        if (std.mem.eql(c_int, mlx.getShape(a), shape)) {
            var r = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&r, a));
            return r;
        }
        _ = mlx.mlx_array_free(a);
    }
    var a = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_ones(&a, shape.ptr, @intCast(shape.len), .bfloat16, s));
    try mlx.check(mlx.mlx_array_eval(a));
    standin_ones = a;
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&r, a));
    return r;
}
pub fn standinRef(x: mlx.mlx_array) !mlx.mlx_array {
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&r, x));
    return r;
}



pub const ProfClock = struct {
    io: std.Io,
    start: std.Io.Timestamp,
    mark_ns: u64 = 0,
    pub fn init() ProfClock {
        const io = std.Io.Threaded.global_single_threaded.io();
        return .{ .io = io, .start = std.Io.Timestamp.now(io, .boot) };
    }
    pub fn lap(self: *ProfClock) u64 {
        const cum: u64 = @intCast(self.start.untilNow(self.io, .boot).nanoseconds);
        const d = cum - self.mark_ns;
        self.mark_ns = cum;
        return d;
    }
};


pub fn decodeProfileEnabled() bool {
    if (decode_prof_enabled) |v| return v;
    const v = diagEnvOn("SUSHI_DECODE_PROFILE");
    decode_prof_enabled = v;
    return v;
}

/// Widest forward the profiler times: `SUSHI_DECODE_PROFILE=<rows>` also times verify widths.
pub fn decodeProfileRows() c_int {
    if (decode_prof_forced_rows > 0) return decode_prof_forced_rows;
    return decodeProfileRowsFrom(std.c.getenv("SUSHI_DECODE_PROFILE"));
}

fn decodeProfileRowsFrom(raw: ?[*:0]const u8) c_int {
    if (!diagEnvValueOn(raw)) return 0;
    return @max(1, std.fmt.parseInt(c_int, std.mem.sliceTo(raw.?, 0), 10) catch 1);
}
var decode_prof_forced_rows: c_int = 0;

/// The forward microbench's profiled pass: `rows > 0` times forwards up to that
/// width, `rows == 0` reports what was timed and restores the env's setting.
pub fn decodeProfileSession(rows: c_int) void {
    if (rows > 0) {
        decode_prof = .{};
        decode_prof_forced_rows = rows;
        decode_prof_enabled = true;
        return;
    }
    decodeProfReport();
    decode_prof_forced_rows = 0;
    decode_prof_enabled = null;
}

pub const diagEnvValueOn = io_util_mod.diagEnvValueOn;
/// SUSHI_MOE_DUMP=<dir> writes one forward's MoE tensors as raw f32, with shapes in filenames.
var moe_dump_dir: ?[]const u8 = null;
var moe_dump_asked: bool = false;
pub var moe_dump_layer: u32 = 0;
var moe_dump_done: bool = false;
var moe_dump_active: bool = false;
pub var moe_dump_warmup_depth: u32 = 0;

pub fn moeDumpBeginForward() bool {
    if (moe_dump_warmup_depth != 0 or moe_dump_done or moe_dump_active or moeDumpDir() == null) return false;
    moe_dump_active = true;
    return true;
}

pub fn moeDumpDir() ?[]const u8 {
    if (!moe_dump_asked) {
        moe_dump_asked = true;
        if (std.c.getenv("SUSHI_MOE_DUMP")) |p| {
            const v = std.mem.span(p);
            if (v.len > 0 and v[0] != '0') moe_dump_dir = v;
        }
    }
    return moe_dump_dir;
}

/// One forward only: the layer counter walks the MoE layers in order and the
/// dump latches off when it has seen a whole model's worth.
pub fn moeDumpForwardDone() void {
    moe_dump_active = false;
    moe_dump_done = true;
}

/// One tensor, evaluated and written as f32. Never on a serving path: the dump
/// is a host read per layer, which is a GPU barrier by construction.
pub fn moeDumpTensor(s: mlx.mlx_stream, tag: []const u8, layer: u32, a: mlx.mlx_array) void {
    const dir = moeDumpDir() orelse return;
    if (!moe_dump_active or moe_dump_done) return;
    var f32a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32a);
    if (mlx.mlx_astype(&f32a, a, .float32, s) != 0) return;
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    if (mlx.mlx_contiguous(&c, f32a, false, s) != 0) return;
    if (mlx.mlx_array_eval(c) != 0) return;
    const p = mlx.mlx_array_data_float32(c) orelse return;
    const shape = mlx.getShape(c);
    var name: [512]u8 = undefined;
    var used: usize = 0;
    used += (std.fmt.bufPrint(name[used..], "{s}/L{d:0>2}_{s}", .{ dir, layer, tag }) catch return).len;
    var n: usize = 1;
    for (shape) |d| {
        n *= @intCast(d);
        used += (std.fmt.bufPrint(name[used..], "_{d}", .{d}) catch return).len;
    }
    used += (std.fmt.bufPrint(name[used..], ".f32\x00", .{}) catch return).len;
    const path: [*:0]const u8 = @ptrCast(name[0 .. used - 1].ptr);
    const fd = std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return;
    defer _ = std.c.close(fd);
    const bytes = std.mem.sliceAsBytes(p[0..n]);
    var put: usize = 0;
    while (put < bytes.len) {
        const w = std.c.write(fd, bytes.ptr + put, bytes.len - put);
        if (w <= 0) return;
        put += @intCast(w);
    }
}

test "moe dump waits for a real forward before writing tensors" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(t.io, &path_buf);
    const saved_dir = moe_dump_dir;
    const saved_asked = moe_dump_asked;
    const saved_done = moe_dump_done;
    const saved_active = moe_dump_active;
    const saved_warmup = moe_dump_warmup_depth;
    const saved_layer = moe_dump_layer;
    defer {
        moe_dump_dir = saved_dir;
        moe_dump_asked = saved_asked;
        moe_dump_done = saved_done;
        moe_dump_active = saved_active;
        moe_dump_warmup_depth = saved_warmup;
        moe_dump_layer = saved_layer;
    }
    moe_dump_dir = path_buf[0..path_len];
    moe_dump_asked = true;
    moe_dump_done = false;
    moe_dump_active = false;
    moe_dump_warmup_depth = 0;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const values = [_]f32{ 1, 2, 3, 4 };
    const a = mlx.mlx_array_new_data(&values, &.{ 1, 2, 2 }, 3, .float32);
    defer _ = mlx.mlx_array_free(a);
    moeDumpTensor(s, "x", 1, a);
    try t.expectError(error.FileNotFound, tmp.dir.openFile(t.io, "L01_x_1_2_2.f32", .{}));

    const config = ModelConfig{ .model_type = "mimo_v2", .num_hidden_layers = 3, .first_k_dense_replace = 1 };
    const forward = struct {
        fn run(cfg: ModelConfig, stream: mlx.mlx_stream, x: mlx.mlx_array) void {
            const dumping = moeDumpBeginForward();
            defer if (dumping) moeDumpForwardDone();
            for (cfg.first_k_dense_replace..cfg.num_hidden_layers) |li| {
                if (dumping) moe_dump_layer = @intCast(li);
                moeDumpTensor(stream, "x", moe_dump_layer, x);
            }
        }
    }.run;
    moe_dump_warmup_depth += 1;
    forward(config, s, a);
    moe_dump_warmup_depth -= 1;
    try t.expectError(error.FileNotFound, tmp.dir.openFile(t.io, "L01_x_1_2_2.f32", .{}));
    try t.expect(!moe_dump_done);
    forward(config, s, a);
    for ([_][]const u8{ "L01_x_1_2_2.f32", "L02_x_1_2_2.f32" }) |name| {
        const bytes = try tmp.dir.readFileAlloc(t.io, name, t.allocator, .limited(128));
        defer t.allocator.free(bytes);
        try t.expectEqualSlices(u8, std.mem.sliceAsBytes(&values), bytes);
    }
    try t.expect(moe_dump_done and !moe_dump_active);
    const later_values = [_]f32{ 5, 6, 7, 8 };
    const later = mlx.mlx_array_new_data(&later_values, &.{ 1, 2, 2 }, 3, .float32);
    defer _ = mlx.mlx_array_free(later);
    forward(config, s, later);
    const unchanged = try tmp.dir.readFileAlloc(t.io, "L01_x_1_2_2.f32", t.allocator, .limited(128));
    defer t.allocator.free(unchanged);
    try t.expectEqualSlices(u8, std.mem.sliceAsBytes(&values), unchanged);
    var files = tmp.dir.iterate();
    var count: usize = 0;
    while (try files.next(t.io)) |_| count += 1;
    try t.expectEqual(@as(usize, 2), count);
}

test "constTableAs hands a load-time f32 table back in the activation dtype" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const vals = [_]f32{ 1.3465735, 1.3465735, 1.0, 1.0 };
    const shape = [_]c_int{4};
    const table = mlx.mlx_array_new_data(&vals, &shape, 1, .float32);
    defer _ = mlx.mlx_array_free(table);

    var cache: DtypeCastCache = .{};
    defer cache.deinit();

    // Same dtype in, same ARRAY out — no copy, no cache entry.
    const same = try constTableAs(table, .float32, &cache, s);
    try std.testing.expect(same.ctx == table.ctx);
    try std.testing.expect(cache.arr.ctx == null);

    // Narrower activation: the table comes back narrowed, and is cached.
    const bf = try constTableAs(table, .bfloat16, &cache, s);
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(bf));
    const again = try constTableAs(table, .bfloat16, &cache, s);
    try std.testing.expect(again.ctx == bf.ctx); // cached, not rebuilt
}

test "a constant table must not widen the activation it scales" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const mvals = [_]f32{ 1.3465735, 1.3465735, 1.0, 1.0 };
    const mshape = [_]c_int{4};
    const table = mlx.mlx_array_new_data(&mvals, &mshape, 1, .float32);
    defer _ = mlx.mlx_array_free(table);

    // A bf16 activation, exactly as post-RoPE q/k arrives.
    const avals = [_]f32{ 0.5, 0.25, 0.125, 0.0625 };
    const ashape = [_]c_int{4};
    const a32 = mlx.mlx_array_new_data(&avals, &ashape, 1, .float32);
    defer _ = mlx.mlx_array_free(a32);
    var act = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(act);
    try mlx.check(mlx.mlx_astype(&act, a32, .bfloat16, s));

    // The raw f32 table is what the bug shipped: the product widens.
    var widened = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(widened);
    try mlx.check(mlx.mlx_multiply(&widened, act, table, s));
    try std.testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(widened));

    // Routed through constTableAs it stays in the activation's dtype.
    var cache: DtypeCastCache = .{};
    defer cache.deinit();
    const matched = try constTableAs(table, mlx.mlx_array_dtype(act), &cache, s);
    var kept = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kept);
    try mlx.check(mlx.mlx_multiply(&kept, act, matched, s));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(kept));
}

test "decode profile rows: 0 and absent are off, a bare switch is one row" {
    try testing.expectEqual(@as(c_int, 0), decodeProfileRowsFrom(null));
    try testing.expectEqual(@as(c_int, 0), decodeProfileRowsFrom("0"));
    try testing.expectEqual(@as(c_int, 1), decodeProfileRowsFrom("on"));
    try testing.expectEqual(@as(c_int, 8), decodeProfileRowsFrom("8"));
}

test "diagEnvValueOn: absent, empty or 0 is off" {
    try testing.expectEqual(false, diagEnvValueOn(null));
    try testing.expectEqual(false, diagEnvValueOn("0"));
    try testing.expectEqual(false, diagEnvValueOn(""));
    try testing.expectEqual(true, diagEnvValueOn("1"));
    try testing.expectEqual(true, diagEnvValueOn("on"));
}
