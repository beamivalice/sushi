// Manual FFI bindings for mlx-c.
// We declare only the functions we need rather than @cImport to avoid
// potential issues with C++ headers and keep the dependency surface explicit.

const std = @import("std");
const builtin = @import("builtin");
const log = @import("log.zig");
const io_util = @import("io_util.zig");
const metal_minify = @import("metal_minify.zig");
pub const kernel_keys = @import("mlx_kernel_keys.zig");

// ── Opaque handle types ──
pub const mlx_array = extern struct { ctx: ?*anyopaque = null };
pub const mlx_stream = extern struct { ctx: ?*anyopaque = null };
pub const mlx_device = extern struct { ctx: ?*anyopaque = null };
pub const mlx_string = extern struct { ctx: ?*anyopaque = null };
pub const mlx_vector_array = extern struct { ctx: ?*anyopaque = null };

/// The engine of our own (`src/tensor`): a call whose stream is its stream, or that takes a tensor of its, goes to
/// the function of the same name there. Off (`engine.enabled`), every call is mlx-c's.
pub const engine = @import("tensor/route.zig");

/// mlx-c's own entry points, for the engine.
/// mlx-c's eval of an MLX array, whatever is routed.
pub fn rawArrayEval(arr: mlx_array) c_int {
    return raw.mlx_array_eval(arr);
}

pub const raw = struct {
    pub const mlx_array_new = emptyHandle(mlx_array);
    pub const mlx_array_free = counted("mlx_array_free", fn (arr: mlx_array) callconv(.c) c_int);
    pub const mlx_array_set = counted("mlx_array_set", fn (arr: *mlx_array, src: mlx_array) callconv(.c) c_int);
    pub const mlx_array_eval = counted("mlx_array_eval", fn (arr: mlx_array) callconv(.c) c_int);
    pub const mlx_array_new_data = counted("mlx_array_new_data", fn (data: ?*const anyopaque, shape: [*]const c_int, dim: c_int, dtype: mlx_dtype) callconv(.c) mlx_array);
    pub const mlx_array_new_int = counted("mlx_array_new_int", fn (val: c_int) callconv(.c) mlx_array);
    pub const mlx_array_new_float = counted("mlx_array_new_float", fn (val: f32) callconv(.c) mlx_array);
    pub const mlx_array_new_bool = counted("mlx_array_new_bool", fn (val: bool) callconv(.c) mlx_array);
    pub const mlx_vector_array_new = counted("mlx_vector_array_new", fn () callconv(.c) mlx_vector_array);
    pub const mlx_vector_array_new_data = counted("mlx_vector_array_new_data", fn (data: [*]const mlx_array, size: usize) callconv(.c) mlx_vector_array);
    pub const mlx_vector_array_new_value = counted("mlx_vector_array_new_value", fn (val: mlx_array) callconv(.c) mlx_vector_array);
    pub const mlx_vector_array_free = counted("mlx_vector_array_free", fn (vec: mlx_vector_array) callconv(.c) c_int);
    pub const mlx_vector_array_size = counted("mlx_vector_array_size", fn (vec: mlx_vector_array) callconv(.c) usize);
    pub const mlx_vector_array_get = counted("mlx_vector_array_get", fn (res: *mlx_array, vec: mlx_vector_array, idx: usize) callconv(.c) c_int);
    pub const mlx_vector_array_append_value = counted("mlx_vector_array_append_value", fn (vec: mlx_vector_array, val: mlx_array) callconv(.c) c_int);
    pub const mlx_stream_free = counted("mlx_stream_free", fn (s: mlx_stream) callconv(.c) c_int);
    pub const mlx_random_bits = counted("mlx_random_bits", fn (res: *mlx_array, shape: [*]const c_int, shape_num: usize, width: c_int, key: mlx_array, s: mlx_stream) callconv(.c) c_int);
    pub const mlx_array_new_data_managed_payload = counted("mlx_array_new_data_managed_payload", fn (data: ?*anyopaque, shape: [*]const c_int, dim: c_int, dtype: mlx_dtype, payload: ?*anyopaque, dtor: ?*const fn (?*anyopaque) callconv(.c) void) callconv(.c) mlx_array);
    pub const mlx_async_eval = counted("mlx_async_eval", fn (outputs: mlx_vector_array) callconv(.c) c_int);
    pub const mlx_synchronize = counted("mlx_synchronize", fn (s: mlx_stream) callconv(.c) c_int);
    pub const mlx_matmul = counted("mlx_matmul", fn (res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) callconv(.c) c_int);
    pub const mlx_clear_cache = counted("mlx_clear_cache", fn () callconv(.c) c_int);
    pub const mlx_set_memory_limit = counted("mlx_set_memory_limit", fn (res: *usize, limit: usize) callconv(.c) c_int);
    pub const mlx_set_cache_limit = counted("mlx_set_cache_limit", fn (res: *usize, limit: usize) callconv(.c) c_int);
    pub const mlx_set_wired_limit = counted("mlx_set_wired_limit", fn (res: *usize, limit: usize) callconv(.c) c_int);
    pub const mlx_get_active_memory = counted("mlx_get_active_memory", fn (res: *usize) callconv(.c) c_int);
    pub const mlx_get_cache_memory = counted("mlx_get_cache_memory", fn (res: *usize) callconv(.c) c_int);
    pub const mlx_get_peak_memory = counted("mlx_get_peak_memory", fn (res: *usize) callconv(.c) c_int);
    pub const mlx_reset_peak_memory = counted("mlx_reset_peak_memory", fn () callconv(.c) c_int);
    // The transient cap and the external reading are patched in (patches/mlx-transient-cap.patch), not upstream MLX.
    pub const mlx_set_transient_limit = counted("mlx_set_transient_limit", fn (res: *usize, limit: usize) callconv(.c) c_int);
    pub const mlx_get_external_memory = counted("mlx_get_external_memory", fn (res: *usize) callconv(.c) c_int);
    pub const mlx_metal_set_max_mb_per_buffer = counted("mlx_metal_set_max_mb_per_buffer", fn (prev: *c_int, mb: c_int) callconv(.c) c_int);
    pub const mlx_get_transient_peak = counted("mlx_get_transient_peak", fn (res: *usize) callconv(.c) c_int);
    pub const mlx_reset_transient_peak = counted("mlx_reset_transient_peak", fn () callconv(.c) c_int);
};

/// `name` of mlx-c with the engine's function of that name in front of it.
fn routed(comptime name: [:0]const u8, comptime F: type) Spread(F, Routed(name, F).go).Fn {
    return Spread(F, Routed(name, F).go).pick();
}

/// Calls the armed `fault` never takes: a test-only reading of MLX on the way to the call under test, which the fault
/// is owed to.
pub const past_fault = struct {
    const Matmul = fn (res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) callconv(.c) c_int;
    pub const mlx_matmul = Spread(Matmul, Counted("mlx_matmul", Matmul).go).pick();
};

/// `name` of mlx-c, counted in `calls`.
fn counted(comptime name: [:0]const u8, comptime F: type) Spread(F, CountedEntry(name, F).go).Fn {
    return Spread(F, CountedEntry(name, F).go).pick();
}

/// A counted call as callers reach it: the armed `fault` takes it at its entry (`faultAtEntry`).
fn CountedEntry(comptime name: [:0]const u8, comptime F: type) type {
    return struct {
        inline fn go(args: std.meta.ArgsTuple(F)) @typeInfo(F).@"fn".return_type.? {
            if (comptime faultable(name, F)) if (faultAtEntry()) return 1;
            return Counted(name, F).go(args);
        }
    };
}

/// A call `fault` may take: one that reports failure by its `c_int` and carries a stream, so it makes a result. A call
/// on the stream handle itself makes none: a fault there would skip a free and take the fault the next op is owed.
fn faultable(comptime name: []const u8, comptime F: type) bool {
    if (std.mem.startsWith(u8, name, "mlx_stream_")) return false;
    const info = @typeInfo(F).@"fn";
    if (info.return_type.? != c_int) return false;
    for (info.param_types) |T| if (T == mlx_stream) return true;
    return false;
}

/// TEST-ONLY: whether the armed `fault` fails this call. It fails at the call's entry, before anything runs, as a raise
/// in mlx-c or a refusal of the engine leaves a result slot unassigned.
inline fn faultAtEntry() bool {
    return if (comptime builtin.is_test) fault.hit() else false;
}

/// Calls into mlx-c since the process started, from any thread (the gauge sampler and request threads call MLX
/// beside the inference thread, hence atomic). A call the seam routes to the engine is not one.
pub var calls: std.atomic.Value(u64) = .init(0);

/// Calls that may have changed what MLX holds: every call but a reading of its allocator or of the device.
var changes: std.atomic.Value(u64) = .init(0);

fn readsOnly(comptime name: []const u8) bool {
    const readings = [_][]const u8{
        "mlx_get_active_memory",       "mlx_get_cache_memory",     "mlx_get_peak_memory",        "mlx_get_external_memory",
        "mlx_get_transient_peak",      "mlx_metal_is_available",   "mlx_version",                "mlx_string_data",
        "mlx_get_default_device",      "mlx_device_info_get",      "mlx_device_info_get_size",   "mlx_device_info_get_string",
        "mlx_device_info_free",        "mlx_device_free",          "mlx_metal_command_buffer_count",
    };
    for (readings) |r| if (std.mem.eql(u8, r, name)) return true;
    return false;
}

/// Calls that reached mlx-c while its allocator was at rest (`engine.hold`): work of a model on MLX done outside its
/// turn, or of a model on the engine that still asks MLX. Zero in a process that keeps its turns.
pub var rest_breaks: std.atomic.Value(u64) = .init(0);

/// Settings of MLX's allocator asked while it was at rest: a model on MLX sets its own inside its turn, and a model on
/// the engine has none to set. A pool clear and cap are owed to MLX's next turn; the rest reach nothing. Zero in a
/// process that keeps its turns.
pub var rest_swallows: std.atomic.Value(u64) = .init(0);

inline fn tally(comptime name: []const u8) void {
    _ = calls.fetchAdd(1, .monotonic);
    if (engine.mlx_at_rest.on.load(.monotonic)) _ = rest_breaks.fetchAdd(1, .monotonic);
    if (comptime !readsOnly(name)) _ = changes.fetchAdd(1, .monotonic);
    if (comptime builtin.is_test) by_name.add(name) else if (callNamesTraced()) by_name.add(name);
}

/// The tally's own reads of an array: not calls of the process it describes.
const untallied = struct {
    const ndim = @extern(*const fn (mlx_array) callconv(.c) usize, .{ .name = "mlx_array_ndim" });
    const shape = @extern(*const fn (mlx_array) callconv(.c) [*]const c_int, .{ .name = "mlx_array_shape" });
    const dtype = @extern(*const fn (mlx_array) callconv(.c) mlx_dtype, .{ .name = "mlx_array_dtype" });
};

/// An attention call MLX runs, tallied as `sdpa <route> ...` (`attention.tallyKey`), for the arms a load reaches.
fn noteAttention(q: mlx_array, k: mlx_array, v: mlx_array, mode: [*:0]const u8, mask: mlx_array, force_fused: bool) void {
    if (comptime !builtin.is_test) if (!callNamesTraced()) return;
    if (untallied.ndim(q) != 4 or untallied.ndim(k) != 4 or untallied.ndim(v) != 4) return;
    const qs = untallied.shape(q);
    const ks = untallied.shape(k);
    const kind: engine.attention.MaskKind = if (std.mem.eql(u8, std.mem.span(mode), "causal")) .causal else if (mask.ctx != null) .array else .none;
    var buf: [96]u8 = undefined;
    const key = engine.attention.tallyKey(&buf, .{
        .q_len = @intCast(qs[2]),
        .k_len = @intCast(ks[2]),
        .heads = @intCast(qs[1]),
        .kv_heads = @intCast(@max(ks[1], 1)),
        .q_dim = @intCast(qs[3]),
        .v_dim = @intCast(untallied.shape(v)[3]),
        .dtype = untallied.dtype(q),
        .mask = kind,
        .force_fused = force_fused,
    }) catch return;
    by_name.addJoined("sdpa ", key);
}

var call_names_env: ?bool = null;

/// `SUSHI_MLX_CALLS_TRACE`: a served process tallies its calls by entry point too, for the request log.
pub fn callNamesTraced() bool {
    return io_util.diagEnvOnCached(&call_names_env, "SUSHI_MLX_CALLS_TRACE");
}

/// Tallies per entry point, to split a count by kind: in tests, and in a process that traces them. An op that took a wrapped mapping of the engine's (a
/// weight) on MLX's side is also counted under `lend <op>`, which is no call.
const Tally = struct {
    const capacity = 1024;
    /// Counts past a full table land here, so a census reads that it missed some instead of losing them.
    const full = "(tally full)";
    names: [capacity][]const u8 = undefined,
    counts: [capacity]u64 = @splat(0),
    used: usize = 0,
    busy: std.atomic.Value(bool) = .init(false),

    fn add(self: *Tally, name: []const u8) void {
        while (self.busy.cmpxchgWeak(false, true, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
        defer self.busy.store(false, .release);
        for (self.names[0..self.used], self.counts[0..self.used]) |n, *c| {
            if (std.mem.eql(u8, n, name)) {
                c.* += 1;
                return;
            }
        }
        self.insert(name);
    }

    fn insert(self: *Tally, name: []const u8) void {
        if (self.used >= capacity - 1) {
            for (self.names[0..self.used], self.counts[0..self.used]) |n, *c| if (n.ptr == full.ptr) {
                c.* += 1;
                return;
            };
            self.names[self.used] = full;
            self.counts[self.used] = 1;
            self.used += 1;
            return;
        }
        self.names[self.used] = name;
        self.counts[self.used] = 1;
        self.used += 1;
    }

    /// `prefix ++ name` for a name known only at run time: copied for the process on its first count.
    fn addJoined(self: *Tally, comptime prefix: []const u8, name: []const u8) void {
        while (self.busy.cmpxchgWeak(false, true, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
        defer self.busy.store(false, .release);
        for (self.names[0..self.used], self.counts[0..self.used]) |n, *c| {
            if (n.len == prefix.len + name.len and std.mem.startsWith(u8, n, prefix) and std.mem.endsWith(u8, n, name)) {
                c.* += 1;
                return;
            }
        }
        if (self.used >= capacity - 1) return self.insert(full);
        self.insert(std.mem.concat(std.heap.page_allocator, u8, &.{ prefix, name }) catch full);
    }

    fn list(self: *Tally, out: *std.ArrayList(NameCount), gpa: std.mem.Allocator) !void {
        while (self.busy.cmpxchgWeak(false, true, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
        defer self.busy.store(false, .release);
        for (self.names[0..self.used], self.counts[0..self.used]) |n, c| try out.append(gpa, .{ .name = n, .count = c });
    }
};
var by_name: Tally = .{};

test "a full call tally counts what it cannot name under one entry, and writes nothing past its table" {
    const t = std.testing;
    const tally_box = try std.heap.page_allocator.create(Tally);
    defer std.heap.page_allocator.destroy(tally_box);
    tally_box.* = .{};
    var buf: [16]u8 = undefined;
    const extra = 77;
    for (0..Tally.capacity - 1 + extra) |i| tally_box.addJoined("n", try std.fmt.bufPrint(&buf, "{d}", .{i}));
    tally_box.add("one more");
    try t.expectEqual(@as(usize, Tally.capacity), tally_box.used);
    try t.expectEqualStrings(Tally.full, tally_box.names[Tally.capacity - 1]);
    try t.expectEqual(@as(u64, extra + 1), tally_box.counts[Tally.capacity - 1]);
}
var ops_by_name: Tally = .{};

/// TEST-ONLY: counts `a` under `lend <where>` when it is a wrapped mapping of the engine's taken on MLX's side.
pub inline fn noteWeightLend(comptime where: []const u8, a: mlx_array) void {
    if (!engine.isMapped(a)) return;
    if (comptime builtin.is_test) return by_name.add("lend " ++ where);
    const Said = struct {
        // One flag per op name: `where` is comptime, so each caller has its own.
        var once: bool = false;
        const op = where;
    };
    log.debug("[engine] weight lend: {s}\n", .{Said.op});
    if (Said.once) return;
    Said.once = true;
    log.warn("[engine] {s} took a weight the engine wraps onto MLX, which wires and bills it a second time for as long as it uses it; said once an op\n", .{Said.op});
}

pub const NameCount = struct { name: []const u8, count: u64 };

/// A custom kernel launched on MLX's side, counted under `kernel <name>`: which of a model's kernels a census reaches.
pub fn noteKernelLaunch(name: []const u8) void {
    if (comptime builtin.is_test) by_name.addJoined("kernel ", name) else if (callNamesTraced()) by_name.addJoined("kernel ", name);
}

/// TEST-ONLY: the calls so far, per entry point.
pub fn callsByName(out: *std.ArrayList(NameCount), gpa: std.mem.Allocator) !void {
    try by_name.list(out, gpa);
}

/// TEST-ONLY: the ops the seam sent to MLX so far (`engine.mlx_ops`), per entry point.
pub fn opsByName(out: *std.ArrayList(NameCount), gpa: std.mem.Allocator) !void {
    try ops_by_name.list(out, gpa);
}

/// An address inside the mlx-c library, to find the files beside it. Naming a symbol is no call.
pub fn libraryAnchor() *const anyopaque {
    return @ptrCast(@extern(*const fn (*mlx_string) callconv(.c) c_int, .{ .name = "mlx_version" }));
}

/// The `_new()` of a handle that mlx-c makes as a zero `ctx` and allocates nothing for: no call into mlx-c.
fn emptyHandle(comptime T: type) fn () T {
    return struct {
        fn f() T {
            return .{};
        }
    }.f;
}

fn Counted(comptime name: [:0]const u8, comptime F: type) type {
    return struct {
        inline fn go(args: std.meta.ArgsTuple(F)) @typeInfo(F).@"fn".return_type.? {
            tally(name);
            if (comptime std.mem.eql(u8, name, "mlx_fast_scaled_dot_product_attention")) noteAttention(args[1], args[2], args[3], args[5], args[6], args[8]);
            return @call(.auto, @extern(*const F, .{ .name = name }), args);
        }
    };
}

fn Routed(comptime name: [:0]const u8, comptime F: type) type {
    const info = @typeInfo(F).@"fn";
    const P = info.param_types;
    const R = info.return_type.?;
    return struct {
        const Args = std.meta.ArgsTuple(F);

        fn carriesStream() bool {
            for (P) |T| if (T == mlx_stream) return true;
            return false;
        }

        const viaMlx = Counted(name, F).go;

        inline fn go(args: Args) R {
            if (comptime faultable(name, F)) if (faultAtEntry()) return 1;
            if (!engine.enabled) {
                // A handle of the engine's can outlive the switch in a cache; MLX's free would read memory MLX never made.
                if (comptime std.mem.eql(u8, name, "mlx_array_free")) if (engine.isOurs(args[0])) return engine.mlx_array_free(args[0]);
                return viaMlx(args);
            }
            switch (engine.side(args)) {
                .engine => {
                    if (@hasDecl(engine, name)) return @call(.auto, @field(engine, name), args);
                    if (R != c_int) @compileError("routed: the engine has no " ++ name);
                    return engine.refuse(name, "the engine has no such op");
                },
                .leak => return if (R == c_int) blk: {
                    engine.traceRefusal();
                    break :blk engine.refuse(name, "an MLX stream inside a forward of the engine");
                } else unreachable,
                .mlx => {
                    // An op carries a stream; a free or a read of a handle does not.
                    if (comptime carriesStream()) {
                        engine.mlx_ops += 1;
                        if (comptime builtin.is_test) ops_by_name.add(name);
                    }
                    // MLX takes a tensor of ours as an array over the same bytes; a result slot that still holds
                    // one is emptied before mlx-c assigns into it.
                    var lent = args;
                    // Lends made for this call alone (`engine.lendsOnce`): mlx-c has its own reference once it returns.
                    var once: [P.len]mlx_array = @splat(.{ .ctx = null });
                    defer inline for (once) |a| {
                        if (a.ctx != null) _ = raw.mlx_array_free(a);
                    };
                    inline for (0..P.len) |i| switch (P[i].?) {
                        mlx_array => if (R == c_int and engine.isOurs(lent[i])) {
                            noteWeightLend(name, lent[i]);
                            const made_once = engine.lendsOnce(lent[i]);
                            lent[i] = engine.lend(lent[i]) orelse return 1;
                            if (made_once) once[i] = lent[i];
                        },
                        *mlx_array => if (engine.isOurs(lent[i].*)) {
                            _ = engine.mlx_array_free(lent[i].*);
                            lent[i].ctx = null;
                        },
                        mlx_vector_array => if (R == c_int and engine.isVec(lent[i])) return engine.refuse(name, "a vector of the engine on an MLX stream"),
                        else => {},
                    };
                    return viaMlx(lent);
                },
            }
        }
    };
}

/// A function of F's parameters that hands them to `go` as a tuple.
fn Spread(comptime F: type, comptime go: anytype) type {
    const info = @typeInfo(F).@"fn";
    const P = info.param_types;
    const R = info.return_type.?;
    return struct {
        const Fn = @TypeOf(pick());

        fn f0() R {
            return go(.{});
        }
        fn f1(a: P[0].?) R {
            return go(.{a});
        }
        fn f2(a: P[0].?, b: P[1].?) R {
            return go(.{ a, b });
        }
        fn f3(a: P[0].?, b: P[1].?, c: P[2].?) R {
            return go(.{ a, b, c });
        }
        fn f4(a: P[0].?, b: P[1].?, c: P[2].?, d: P[3].?) R {
            return go(.{ a, b, c, d });
        }
        fn f5(a: P[0].?, b: P[1].?, c: P[2].?, d: P[3].?, e: P[4].?) R {
            return go(.{ a, b, c, d, e });
        }
        fn f6(a: P[0].?, b: P[1].?, c: P[2].?, d: P[3].?, e: P[4].?, f: P[5].?) R {
            return go(.{ a, b, c, d, e, f });
        }
        fn f7(a: P[0].?, b: P[1].?, c: P[2].?, d: P[3].?, e: P[4].?, f: P[5].?, g: P[6].?) R {
            return go(.{ a, b, c, d, e, f, g });
        }
        fn f8(a: P[0].?, b: P[1].?, c: P[2].?, d: P[3].?, e: P[4].?, f: P[5].?, g: P[6].?, h: P[7].?) R {
            return go(.{ a, b, c, d, e, f, g, h });
        }
        fn f11(a: P[0].?, b: P[1].?, c: P[2].?, d: P[3].?, e: P[4].?, f: P[5].?, g: P[6].?, h: P[7].?, i: P[8].?, j: P[9].?, k: P[10].?) R {
            return go(.{ a, b, c, d, e, f, g, h, i, j, k });
        }
        fn f13(a: P[0].?, b: P[1].?, c: P[2].?, d: P[3].?, e: P[4].?, f: P[5].?, g: P[6].?, h: P[7].?, i: P[8].?, j: P[9].?, k: P[10].?, l: P[11].?, m: P[12].?) R {
            return go(.{ a, b, c, d, e, f, g, h, i, j, k, l, m });
        }
        fn f14(a: P[0].?, b: P[1].?, c: P[2].?, d: P[3].?, e: P[4].?, f: P[5].?, g: P[6].?, h: P[7].?, i: P[8].?, j: P[9].?, k: P[10].?, l: P[11].?, m: P[12].?, n: P[13].?) R {
            return go(.{ a, b, c, d, e, f, g, h, i, j, k, l, m, n });
        }
        fn f17(a: P[0].?, b: P[1].?, c: P[2].?, d: P[3].?, e: P[4].?, f: P[5].?, g: P[6].?, h: P[7].?, i: P[8].?, j: P[9].?, k: P[10].?, l: P[11].?, m: P[12].?, n: P[13].?, o: P[14].?, p: P[15].?, q: P[16].?) R {
            return go(.{ a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q });
        }
        fn f9(a: P[0].?, b: P[1].?, c: P[2].?, d: P[3].?, e: P[4].?, f: P[5].?, g: P[6].?, h: P[7].?, i: P[8].?) R {
            return go(.{ a, b, c, d, e, f, g, h, i });
        }
        fn f10(a: P[0].?, b: P[1].?, c: P[2].?, d: P[3].?, e: P[4].?, f: P[5].?, g: P[6].?, h: P[7].?, i: P[8].?, j: P[9].?) R {
            return go(.{ a, b, c, d, e, f, g, h, i, j });
        }
        fn pick() switch (P.len) {
            0 => @TypeOf(f0),
            1 => @TypeOf(f1),
            2 => @TypeOf(f2),
            3 => @TypeOf(f3),
            4 => @TypeOf(f4),
            5 => @TypeOf(f5),
            6 => @TypeOf(f6),
            7 => @TypeOf(f7),
            8 => @TypeOf(f8),
            9 => @TypeOf(f9),
            10 => @TypeOf(f10),
            11 => @TypeOf(f11),
            13 => @TypeOf(f13),
            14 => @TypeOf(f14),
            17 => @TypeOf(f17),
            else => @compileError("add an arity of " ++ std.fmt.comptimePrint("{d}", .{P.len})),
        } {
            return switch (P.len) {
                0 => f0,
                1 => f1,
                2 => f2,
                3 => f3,
                4 => f4,
                5 => f5,
                6 => f6,
                7 => f7,
                8 => f8,
                9 => f9,
                10 => f10,
                11 => f11,
                13 => f13,
                14 => f14,
                17 => f17,
                else => unreachable,
            };
        }
    };
}

pub const mlx_closure = extern struct { ctx: ?*anyopaque = null };

// ── Enums ──
pub const mlx_dtype = enum(c_int) {
    bool_ = 0,
    uint8 = 1,
    uint16 = 2,
    uint32 = 3,
    uint64 = 4,
    int8 = 5,
    int16 = 6,
    int32 = 7,
    int64 = 8,
    float16 = 9,
    float32 = 10,
    float64 = 11,
    bfloat16 = 12,
    complex64 = 13,
};

pub const mlx_device_type = enum(c_int) { cpu = 0, gpu = 1 };

// ── Optional types ──
pub const mlx_optional_int = extern struct {
    value: c_int = 0,
    has_value: bool = false,

    pub fn none() mlx_optional_int {
        return .{ .value = 0, .has_value = false };
    }
    pub fn some(v: c_int) mlx_optional_int {
        return .{ .value = v, .has_value = true };
    }
};

pub const mlx_optional_float = extern struct {
    value: f32 = 0,
    has_value: bool = false,

    pub fn none() mlx_optional_float {
        return .{ .value = 0, .has_value = false };
    }
    pub fn some(v: f32) mlx_optional_float {
        return .{ .value = v, .has_value = true };
    }
};

// ── Extern declarations ──
pub const mlx_version = counted("mlx_version", fn (str: *mlx_string) callconv(.c) c_int);

// String
pub const mlx_string_new = emptyHandle(mlx_string);
pub const mlx_string_data = counted("mlx_string_data", fn (str: mlx_string) callconv(.c) [*:0]const u8);
pub const mlx_string_free = counted("mlx_string_free", fn (str: mlx_string) callconv(.c) c_int);

// Device
pub const mlx_device_new = emptyHandle(mlx_device);
pub const mlx_device_new_type = counted("mlx_device_new_type", fn (dtype: mlx_device_type, index: c_int) callconv(.c) mlx_device);
pub const mlx_device_free = counted("mlx_device_free", fn (dev: mlx_device) callconv(.c) c_int);
pub const mlx_get_default_device = counted("mlx_get_default_device", fn (dev: *mlx_device) callconv(.c) c_int);
pub const mlx_set_default_device = counted("mlx_set_default_device", fn (dev: mlx_device) callconv(.c) c_int);

// Stream
pub const mlx_stream_new = emptyHandle(mlx_stream);
pub const mlx_stream_set = counted("mlx_stream_set", fn (stream: *mlx_stream, src: mlx_stream) callconv(.c) c_int);
pub const mlx_stream_equal = counted("mlx_stream_equal", fn (lhs: mlx_stream, rhs: mlx_stream) callconv(.c) bool);
pub const mlx_stream_new_device = counted("mlx_stream_new_device", fn (dev: mlx_device) callconv(.c) mlx_stream);
pub fn mlx_stream_free(s: mlx_stream) c_int {
    if (engine.enabled and engine.isStream(s)) return 0;
    return raw.mlx_stream_free(s);
}
pub const mlx_default_cpu_stream_new = counted("mlx_default_cpu_stream_new", fn () callconv(.c) mlx_stream);
pub const mlx_default_gpu_stream_new = counted("mlx_default_gpu_stream_new", fn () callconv(.c) mlx_stream);
pub const mlx_stream_get_device = counted("mlx_stream_get_device", fn (dev: *mlx_device, stream: mlx_stream) callconv(.c) c_int);
pub const mlx_device_get_type = counted("mlx_device_get_type", fn (dtype: *mlx_device_type, dev: mlx_device) callconv(.c) c_int);

/// True when the stream targets the GPU (custom Metal kernels require it).
pub fn streamIsGpu(s: mlx_stream) bool {
    if (engine.enabled and engine.isStream(s)) return true;
    var dev = mlx_device{ .ctx = null };
    if (mlx_stream_get_device(&dev, s) != 0) return false;
    defer _ = mlx_device_free(dev);
    var dt: mlx_device_type = .cpu;
    if (mlx_device_get_type(&dt, dev) != 0) return false;
    return dt == .gpu;
}
pub fn mlx_synchronize(s: mlx_stream) c_int {
    if (engine.enabled and engine.isStream(s)) return engine.mlx_synchronize(s);
    return raw.mlx_synchronize(s);
}

// Metal
pub const mlx_metal_is_available = counted("mlx_metal_is_available", fn (res: *bool) callconv(.c) c_int);
pub const _mlx_array_is_available = routed("_mlx_array_is_available", fn (res: *bool, arr: mlx_array) callconv(.c) c_int);

// Array creation
pub const mlx_array_new = emptyHandle(mlx_array);
pub fn mlx_array_new_int(val: c_int) mlx_array {
    return if (engine.active) engine.mlx_array_new_int(val) else raw.mlx_array_new_int(val);
}
pub fn mlx_array_new_float(val: f32) mlx_array {
    return if (engine.active) engine.mlx_array_new_float(val) else raw.mlx_array_new_float(val);
}
pub fn mlx_array_new_bool(val: bool) mlx_array {
    return if (engine.active) engine.mlx_array_new_bool(val) else raw.mlx_array_new_bool(val);
}
pub fn mlx_array_new_data(data: ?*const anyopaque, shape: [*]const c_int, dim: c_int, dtype: mlx_dtype) mlx_array {
    return if (engine.active) engine.mlx_array_new_data(data, shape, dim, dtype) else raw.mlx_array_new_data(data, shape, dim, dtype);
}
/// MLX's makers never fail; an engine's may (its pool refuses the buffer and hands back an empty handle), so a host
/// array a request or a cache makes comes from these.
pub const HostArrayRefused = error{HostArrayRefused};

/// A host array just made, or the error of a host that could not make it.
pub fn made(a: mlx_array) HostArrayRefused!mlx_array {
    return if (a.ctx == null) error.HostArrayRefused else a;
}
pub fn newData(data: ?*const anyopaque, shape: []const c_int, dtype: mlx_dtype) HostArrayRefused!mlx_array {
    return made(mlx_array_new_data(data, shape.ptr, @intCast(shape.len), dtype));
}
pub fn newInt(val: c_int) HostArrayRefused!mlx_array {
    return made(mlx_array_new_int(val));
}
pub fn newFloat(val: f32) HostArrayRefused!mlx_array {
    return made(mlx_array_new_float(val));
}
pub fn newBool(val: bool) HostArrayRefused!mlx_array {
    return made(mlx_array_new_bool(val));
}
pub const mlx_array_new_data_managed = counted("mlx_array_new_data_managed", fn (data: ?*anyopaque, shape: [*]const c_int, dim: c_int, dtype: mlx_dtype, dtor: ?*const fn (?*anyopaque) callconv(.c) void) callconv(.c) mlx_array);
pub const mlx_array_new_data_managed_payload = counted("mlx_array_new_data_managed_payload", fn (data: ?*anyopaque, shape: [*]const c_int, dim: c_int, dtype: mlx_dtype, payload: ?*anyopaque, dtor: ?*const fn (?*anyopaque) callconv(.c) void) callconv(.c) mlx_array);
/// The Metal buffer (`id<MTLBuffer>`, not retained) of an evaluated array and the byte offset of its data in it.
pub const _mlx_array_metal_buffer = counted("_mlx_array_metal_buffer", fn (buffer: *?*anyopaque, offset: *usize, arr: mlx_array) callconv(.c) c_int);
/// Whether this handle is the only holder of the array and of its buffer, as MLX asks before it writes in place.
pub const _mlx_array_is_donatable = counted("_mlx_array_is_donatable", fn (res: *bool, arr: mlx_array) callconv(.c) c_int);
/// An empty handle holds nothing, so its free is no call into mlx-c.
pub fn mlx_array_free(arr: mlx_array) c_int {
    return if (arr.ctx == null) 0 else routed("mlx_array_free", fn (arr: mlx_array) callconv(.c) c_int)(arr);
}
pub fn mlx_array_set(arr: *mlx_array, src: mlx_array) c_int {
    if (engine.enabled and (engine.isOurs(src) or engine.isOurs(arr.*))) return engine.mlx_array_set(arr, src);
    return raw.mlx_array_set(arr, src);
}

// Array info
pub const mlx_array_tostring = counted("mlx_array_tostring", fn (str: *mlx_string, arr: mlx_array) callconv(.c) c_int);
const array_info = struct {
    const mlx_array_ndim = counted("mlx_array_ndim", fn (arr: mlx_array) callconv(.c) usize);
    const mlx_array_shape = counted("mlx_array_shape", fn (arr: mlx_array) callconv(.c) [*]const c_int);
    const mlx_array_size = counted("mlx_array_size", fn (arr: mlx_array) callconv(.c) usize);
    const mlx_array_dtype = counted("mlx_array_dtype", fn (arr: mlx_array) callconv(.c) mlx_dtype);
};

/// TEST-ONLY count of the array shape, size and dtype queries (each is an FFI call); a release build carries no counter.
pub var query_count: u64 = 0;

pub inline fn mlx_array_ndim(arr: mlx_array) usize {
    if (comptime builtin.is_test) query_count += 1;
    if (engine.enabled and engine.isOurs(arr)) return engine.ndim(arr);
    return array_info.mlx_array_ndim(arr);
}
pub inline fn mlx_array_shape(arr: mlx_array) [*]const c_int {
    if (comptime builtin.is_test) query_count += 1;
    if (engine.enabled and engine.isOurs(arr)) return engine.shape(arr);
    return array_info.mlx_array_shape(arr);
}
pub inline fn mlx_array_size(arr: mlx_array) usize {
    if (comptime builtin.is_test) query_count += 1;
    if (engine.enabled and engine.isOurs(arr)) return engine.size(arr);
    return array_info.mlx_array_size(arr);
}
pub inline fn mlx_array_dtype(arr: mlx_array) mlx_dtype {
    if (comptime builtin.is_test) query_count += 1;
    if (engine.enabled and engine.isOurs(arr)) return engine.dtypeOf(arr);
    return array_info.mlx_array_dtype(arr);
}
pub const mlx_array_strides = routed("mlx_array_strides", fn (arr: mlx_array) callconv(.c) [*]const usize);
pub fn mlx_array_eval(arr: mlx_array) c_int {
    if (faultAtEntry()) return 1;
    // What a maker or an op that failed left its caller: mlx-c would read through it.
    if (arr.ctx == null) {
        latchError("[mlx] eval: an empty handle, the result of an op that failed");
        return 1;
    }
    if (engine.enabled and engine.isOurs(arr)) return engine.mlx_array_eval(arr);
    return eval_raw.mlx_array_eval(arr);
}
pub const mlx_array_itemsize = routed("mlx_array_itemsize", fn (arr: mlx_array) callconv(.c) usize);

// Scalar access
pub fn mlx_array_item_float32(res: *f32, arr: mlx_array) c_int {
    if (engine.enabled and engine.isOurs(arr)) return engine.item(f32, res, arr);
    return eval_raw.mlx_array_item_float32(res, arr);
}
pub fn mlx_array_item_int32(res: *i32, arr: mlx_array) c_int {
    if (engine.enabled and engine.isOurs(arr)) return engine.item(i32, res, arr);
    return eval_raw.mlx_array_item_int32(res, arr);
}

// Data access
pub const mlx_array_data_bool = routed("mlx_array_data_bool", fn (arr: mlx_array) callconv(.c) ?[*]const bool);
pub const mlx_array_data_float32 = routed("mlx_array_data_float32", fn (arr: mlx_array) callconv(.c) ?[*]const f32);
pub const mlx_array_data_float16 = routed("mlx_array_data_float16", fn (arr: mlx_array) callconv(.c) ?[*]const f16);
pub const mlx_array_data_bfloat16 = routed("mlx_array_data_bfloat16", fn (arr: mlx_array) callconv(.c) ?[*]const u16);
pub const mlx_array_data_int32 = routed("mlx_array_data_int32", fn (arr: mlx_array) callconv(.c) ?[*]const i32);
pub const mlx_array_data_uint16 = routed("mlx_array_data_uint16", fn (arr: mlx_array) callconv(.c) ?[*]const u16);
pub const mlx_array_data_uint32 = routed("mlx_array_data_uint32", fn (arr: mlx_array) callconv(.c) ?[*]const u32);
pub const mlx_array_data_uint8 = routed("mlx_array_data_uint8", fn (arr: mlx_array) callconv(.c) ?[*]const u8);

/// `arr`'s evaluated elements for a readback that steers no shape (docs/bill_redesign.md#recorder): in a dry recording
/// on the engine, made-up ones, each word below `below` when given (`engine.hostResult`); elsewhere its bytes.
pub fn hostResult(comptime T: type, arr: mlx_array, below: ?u32) ![]const T {
    if (engine.enabled and engine.isOurs(arr)) return engine.hostResult(T, arr, below);
    const n = mlx_array_size(arr);
    const p = switch (T) {
        i32 => mlx_array_data_int32(arr),
        u32 => mlx_array_data_uint32(arr),
        f32 => mlx_array_data_float32(arr),
        else => @compileError("hostResult reads i32, u32 or f32"),
    } orelse return error.HostRead;
    return p[0..n];
}

// Vector array
pub fn mlx_vector_array_new() mlx_vector_array {
    return if (engine.active) engine.mlx_vector_array_new() else raw.mlx_vector_array_new();
}
pub fn mlx_vector_array_new_data(data: [*]const mlx_array, size: usize) mlx_vector_array {
    if (engine.active) return engine.mlx_vector_array_new_data(data, size);
    if (engine.enabled) return engine.lentVector(data[0..size]);
    return raw.mlx_vector_array_new_data(data, size);
}
pub const mlx_vector_array_free = routed("mlx_vector_array_free", fn (vec: mlx_vector_array) callconv(.c) c_int);
pub const mlx_vector_array_size = routed("mlx_vector_array_size", fn (vec: mlx_vector_array) callconv(.c) usize);
pub const mlx_vector_array_get = routed("mlx_vector_array_get", fn (res: *mlx_array, vec: mlx_vector_array, idx: usize) callconv(.c) c_int);
pub fn mlx_vector_array_new_value(val: mlx_array) mlx_vector_array {
    if (engine.active) return engine.mlx_vector_array_new_value(val);
    if (engine.enabled) return engine.lentVector(&.{val});
    return raw.mlx_vector_array_new_value(val);
}
pub fn mlx_vector_array_append_value(vec: mlx_vector_array, val: mlx_array) c_int {
    if (!engine.enabled) return raw.mlx_vector_array_append_value(vec, val);
    if (engine.isVec(vec)) return engine.mlx_vector_array_append_value(vec, val);
    if (!engine.isOurs(val)) return raw.mlx_vector_array_append_value(vec, val);
    noteWeightLend("mlx_vector_array_append_value", val);
    const made_once = engine.lendsOnce(val);
    const lent = engine.lend(val) orelse return 1;
    defer if (made_once) {
        _ = raw.mlx_array_free(lent);
    };
    return raw.mlx_vector_array_append_value(vec, lent);
}

// Graph utils (diagnostics)
pub const mlx_node_namer = extern struct { ctx: ?*anyopaque = null };
pub const mlx_node_namer_new = counted("mlx_node_namer_new", fn () callconv(.c) mlx_node_namer);
pub const mlx_node_namer_free = counted("mlx_node_namer_free", fn (namer: mlx_node_namer) callconv(.c) c_int);
pub const mlx_print_graph = counted("mlx_print_graph", fn (os: *std.c.FILE, namer: mlx_node_namer, outputs: mlx_vector_array) callconv(.c) c_int);

// Closure + compile
pub const mlx_closure_new_func_payload = counted("mlx_closure_new_func_payload", fn (
    fun: *const fn (*mlx_vector_array, mlx_vector_array, ?*anyopaque) callconv(.c) c_int,
    payload: ?*anyopaque,
    dtor: ?*const fn (?*anyopaque) callconv(.c) void,
) callconv(.c) mlx_closure);
pub const mlx_closure_free = counted("mlx_closure_free", fn (cls: mlx_closure) callconv(.c) c_int);
pub const mlx_closure_apply = counted("mlx_closure_apply", fn (res: *mlx_vector_array, cls: mlx_closure, input: mlx_vector_array) callconv(.c) c_int);
pub const mlx_compile = counted("mlx_compile", fn (res: *mlx_closure, fun: mlx_closure, shapeless: bool) callconv(.c) c_int);
pub const mlx_detail_compile_clear_cache = counted("mlx_detail_compile_clear_cache", fn () callconv(.c) c_int);

// ── Ops ──
pub const mlx_add = routed("mlx_add", fn (res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_subtract = routed("mlx_subtract", fn (res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_multiply = routed("mlx_multiply", fn (res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_divide = routed("mlx_divide", fn (res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_floor_divide = routed("mlx_floor_divide", fn (res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_negative = routed("mlx_negative", fn (res: *mlx_array, a: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_maximum = routed("mlx_maximum", fn (res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) callconv(.c) c_int);
// Kokoro's SineGen takes the fractional part of a phase ramp (`x - floor(x)`).
pub const mlx_floor = routed("mlx_floor", fn (res: *mlx_array, a: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_ceil = routed("mlx_ceil", fn (res: *mlx_array, a: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_log2 = routed("mlx_log2", fn (res: *mlx_array, a: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_round = routed("mlx_round", fn (res: *mlx_array, a: mlx_array, decimals: c_int, s: mlx_stream) callconv(.c) c_int);
pub const mlx_sign = routed("mlx_sign", fn (res: *mlx_array, a: mlx_array, s: mlx_stream) callconv(.c) c_int);
// Complex plumbing for Kokoro's iSTFTNet head. mlx-c has NO "build a complex
// array from two real ones" op, so the spectrum is assembled as `re + im·i`
// using a complex SCALAR from `mlx_array_new_complex` (float ⊕ complex
// promotes to complex).
pub const mlx_real = routed("mlx_real", fn (res: *mlx_array, a: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_imag = routed("mlx_imag", fn (res: *mlx_array, a: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_arctan2 = routed("mlx_arctan2", fn (res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_array_new_complex = counted("mlx_array_new_complex", fn (real_val: f32, imag_val: f32) callconv(.c) mlx_array);
pub const mlx_minimum = routed("mlx_minimum", fn (res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_matmul = routed("mlx_matmul", fn (res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_block_masked_mm = routed("mlx_block_masked_mm", fn (res: *mlx_array, a: mlx_array, b: mlx_array, block_size: c_int, mask_out: mlx_array, mask_lhs: mlx_array, mask_rhs: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_square = routed("mlx_square", fn (res: *mlx_array, a: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_sqrt = routed("mlx_sqrt", fn (res: *mlx_array, a: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_rsqrt = routed("mlx_rsqrt", fn (res: *mlx_array, a: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_exp = routed("mlx_exp", fn (res: *mlx_array, a: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_log = routed("mlx_log", fn (res: *mlx_array, a: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_abs = routed("mlx_abs", fn (res: *mlx_array, a: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_tanh = routed("mlx_tanh", fn (res: *mlx_array, a: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_cos = routed("mlx_cos", fn (res: *mlx_array, a: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_sin = routed("mlx_sin", fn (res: *mlx_array, a: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_erf = routed("mlx_erf", fn (res: *mlx_array, a: mlx_array, s: mlx_stream) callconv(.c) c_int);

pub const mlx_reshape = routed("mlx_reshape", fn (res: *mlx_array, a: mlx_array, shape: [*]const c_int, shape_num: usize, s: mlx_stream) callconv(.c) c_int);
pub const mlx_transpose = routed("mlx_transpose", fn (res: *mlx_array, a: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_transpose_axes = routed("mlx_transpose_axes", fn (res: *mlx_array, a: mlx_array, axes: [*]const c_int, axes_num: usize, s: mlx_stream) callconv(.c) c_int);
pub const mlx_expand_dims = routed("mlx_expand_dims", fn (res: *mlx_array, a: mlx_array, axis: c_int, s: mlx_stream) callconv(.c) c_int);
pub const mlx_squeeze = routed("mlx_squeeze", fn (res: *mlx_array, a: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_broadcast_to = routed("mlx_broadcast_to", fn (res: *mlx_array, a: mlx_array, shape: [*]const c_int, shape_num: usize, s: mlx_stream) callconv(.c) c_int);

pub const mlx_take = routed("mlx_take", fn (res: *mlx_array, a: mlx_array, indices: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_take_axis = routed("mlx_take_axis", fn (res: *mlx_array, a: mlx_array, indices: mlx_array, axis: c_int, s: mlx_stream) callconv(.c) c_int);

pub const mlx_concatenate_axis = routed("mlx_concatenate_axis", fn (res: *mlx_array, arrays: mlx_vector_array, axis: c_int, s: mlx_stream) callconv(.c) c_int);
pub const mlx_pad = routed("mlx_pad", fn (res: *mlx_array, a: mlx_array, axes: [*]const c_int, axes_num: usize, low_pad: [*]const c_int, low_pad_num: usize, high_pad: [*]const c_int, high_pad_num: usize, pad_value: mlx_array, mode: [*:0]const u8, s: mlx_stream) callconv(.c) c_int);

pub const mlx_softmax_axis = routed("mlx_softmax_axis", fn (res: *mlx_array, a: mlx_array, axis: c_int, precise: bool, s: mlx_stream) callconv(.c) c_int);
pub const mlx_argmax_axis = routed("mlx_argmax_axis", fn (res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, s: mlx_stream) callconv(.c) c_int);

pub const mlx_copy = routed("mlx_copy", fn (res: *mlx_array, a: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_view = routed("mlx_view", fn (res: *mlx_array, a: mlx_array, dtype: mlx_dtype, s: mlx_stream) callconv(.c) c_int);
pub const mlx_sort_axis = routed("mlx_sort_axis", fn (res: *mlx_array, a: mlx_array, axis: c_int, s: mlx_stream) callconv(.c) c_int);
pub const mlx_argsort_axis = routed("mlx_argsort_axis", fn (res: *mlx_array, a: mlx_array, axis: c_int, s: mlx_stream) callconv(.c) c_int);
pub const mlx_topk = routed("mlx_topk", fn (res: *mlx_array, a: mlx_array, k: c_int, s: mlx_stream) callconv(.c) c_int);
pub const mlx_topk_axis = routed("mlx_topk_axis", fn (res: *mlx_array, a: mlx_array, k: c_int, axis: c_int, s: mlx_stream) callconv(.c) c_int);
pub const mlx_cumsum = routed("mlx_cumsum", fn (res: *mlx_array, a: mlx_array, axis: c_int, reverse: bool, inclusive: bool, s: mlx_stream) callconv(.c) c_int);

pub const mlx_mean_axis = routed("mlx_mean_axis", fn (res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, s: mlx_stream) callconv(.c) c_int);
// Variance along one axis — the InstanceNorm1d half of Kokoro's AdaIN1d
// (`src/kokoro.zig`). `ddof` 0 = biased/population variance, which is what
// torch's normalization layers use.
pub const mlx_var_axis = routed("mlx_var_axis", fn (res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, ddof: c_int, s: mlx_stream) callconv(.c) c_int);
pub const mlx_min_axis = routed("mlx_min_axis", fn (res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, s: mlx_stream) callconv(.c) c_int);

pub const mlx_astype = routed("mlx_astype", fn (res: *mlx_array, a: mlx_array, dtype: mlx_dtype, s: mlx_stream) callconv(.c) c_int);

pub const mlx_equal = routed("mlx_equal", fn (res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_array_equal = routed("mlx_array_equal", fn (res: *mlx_array, a: mlx_array, b: mlx_array, equal_nan: bool, s: mlx_stream) callconv(.c) c_int);
pub const mlx_remainder = routed("mlx_remainder", fn (res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_where = routed("mlx_where", fn (res: *mlx_array, condition: mlx_array, x: mlx_array, y: mlx_array, s: mlx_stream) callconv(.c) c_int);

pub const mlx_arange = routed("mlx_arange", fn (res: *mlx_array, start: f64, stop: f64, step: f64, dtype: mlx_dtype, s: mlx_stream) callconv(.c) c_int);
pub const mlx_full = routed("mlx_full", fn (res: *mlx_array, shape: [*]const c_int, shape_num: usize, val: mlx_array, dtype: mlx_dtype, s: mlx_stream) callconv(.c) c_int);
pub const mlx_zeros = routed("mlx_zeros", fn (res: *mlx_array, shape: [*]const c_int, shape_num: usize, dtype: mlx_dtype, s: mlx_stream) callconv(.c) c_int);
pub const mlx_ones = routed("mlx_ones", fn (res: *mlx_array, shape: [*]const c_int, shape_num: usize, dtype: mlx_dtype, s: mlx_stream) callconv(.c) c_int);

pub const mlx_slice = routed("mlx_slice", fn (res: *mlx_array, a: mlx_array, start: [*]const c_int, start_num: usize, stop: [*]const c_int, stop_num: usize, strides: [*]const c_int, strides_num: usize, s: mlx_stream) callconv(.c) c_int);
pub const mlx_slice_update = routed("mlx_slice_update", fn (res: *mlx_array, src: mlx_array, update: mlx_array, start: [*]const c_int, start_num: usize, stop: [*]const c_int, stop_num: usize, strides: [*]const c_int, strides_num: usize, s: mlx_stream) callconv(.c) c_int);
pub const mlx_scatter_single = routed("mlx_scatter_single", fn (res: *mlx_array, a: mlx_array, indices: mlx_array, updates: mlx_array, axis: c_int, s: mlx_stream) callconv(.c) c_int);

pub const mlx_triu = routed("mlx_triu", fn (res: *mlx_array, x: mlx_array, k: c_int, s: mlx_stream) callconv(.c) c_int);
pub const mlx_tril = routed("mlx_tril", fn (res: *mlx_array, x: mlx_array, k: c_int, s: mlx_stream) callconv(.c) c_int);

pub const mlx_power = routed("mlx_power", fn (res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_less = routed("mlx_less", fn (res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_greater = routed("mlx_greater", fn (res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_greater_equal = routed("mlx_greater_equal", fn (res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_less_equal = routed("mlx_less_equal", fn (res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) callconv(.c) c_int);

// Quantized matmul
pub const mlx_quantized_matmul = routed("mlx_quantized_matmul", fn (res: *mlx_array, x: mlx_array, w: mlx_array, scales: mlx_array, biases: mlx_array, transpose_w: bool, group_size: mlx_optional_int, bits: mlx_optional_int, mode: [*:0]const u8, s: mlx_stream) callconv(.c) c_int);

// Gathered quantized matmul (for MoE expert dispatch)
pub const mlx_gather_qmm = routed("mlx_gather_qmm", fn (res: *mlx_array, x: mlx_array, w: mlx_array, scales: mlx_array, biases: mlx_array, lhs_indices: mlx_array, rhs_indices: mlx_array, transpose_w: bool, group_size: mlx_optional_int, bits: mlx_optional_int, mode: [*:0]const u8, sorted_indices: bool, s: mlx_stream) callconv(.c) c_int);
// Dense gathered matmul — the unquantized analog of gather_qmm, used for bf16
// MoE expert dispatch. No transpose flag: `b` must already be [..., in, out].
pub const mlx_gather_mm = routed("mlx_gather_mm", fn (res: *mlx_array, a: mlx_array, b: mlx_array, lhs_indices: mlx_array, rhs_indices: mlx_array, sorted_indices: bool, s: mlx_stream) callconv(.c) c_int);

// Dequantize (fallback)
pub const mlx_dequantize = routed("mlx_dequantize", fn (res: *mlx_array, w: mlx_array, scales: mlx_array, biases: mlx_array, group_size: mlx_optional_int, bits: mlx_optional_int, mode: [*:0]const u8, global_scale: mlx_array, dtype: mlx_optional_dtype, s: mlx_stream) callconv(.c) c_int);

// Quantize (affine group-wise). Returns a vector_array of [q, scales, biases].
// Used by the KV cache quantization path; see src/kv_quant.zig.
pub const mlx_quantize = routed("mlx_quantize", fn (res: *mlx_vector_array, w: mlx_array, group_size: mlx_optional_int, bits: mlx_optional_int, mode: [*:0]const u8, global_scale: mlx_array, s: mlx_stream) callconv(.c) c_int);

// Additional ops for MoE / GatedDeltaNet
pub const mlx_sigmoid = routed("mlx_sigmoid", fn (res: *mlx_array, a: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_sum_axis = routed("mlx_sum_axis", fn (res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, s: mlx_stream) callconv(.c) c_int);
pub const mlx_max_axis = routed("mlx_max_axis", fn (res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, s: mlx_stream) callconv(.c) c_int);
pub const mlx_sum = routed("mlx_sum", fn (res: *mlx_array, a: mlx_array, keepdims: bool, s: mlx_stream) callconv(.c) c_int);
pub const mlx_conv1d = routed("mlx_conv1d", fn (res: *mlx_array, input: mlx_array, weight: mlx_array, stride: c_int, padding: c_int, dilation: c_int, groups: c_int, s: mlx_stream) callconv(.c) c_int);
// FFT — real-input forward transform for the TTS speaker-encoder mel/STFT.
// `mlx_fft_norm`: BACKWARD=0 (no scaling on forward; matches mx.fft.rfft default).
pub const mlx_fft_norm = c_int;
pub const MLX_FFT_NORM_BACKWARD: mlx_fft_norm = 0;
pub const mlx_fft_rfft = routed("mlx_fft_rfft", fn (res: *mlx_array, a: mlx_array, n: c_int, axis: c_int, norm: mlx_fft_norm, s: mlx_stream) callconv(.c) c_int);
// Real-output inverse transform — the iSTFT head of Kokoro's iSTFTNet generator
// (`src/kokoro.zig`). `n` is the OUTPUT length in samples, so the caller states
// the frame size (20) rather than inferring it from the 11-bin input.
pub const mlx_fft_irfft = routed("mlx_fft_irfft", fn (res: *mlx_array, a: mlx_array, n: c_int, axis: c_int, norm: mlx_fft_norm, s: mlx_stream) callconv(.c) c_int);
// 2D/3D conv + transposed conv for the media-generation decoders (VAE/vocoder).
// MLX layout: input NHWC / NDHWC, weight OHWI / ODHWI (out-channels first, in-channels last).
pub const mlx_conv2d = routed("mlx_conv2d", fn (res: *mlx_array, input: mlx_array, weight: mlx_array, stride_0: c_int, stride_1: c_int, padding_0: c_int, padding_1: c_int, dilation_0: c_int, dilation_1: c_int, groups: c_int, s: mlx_stream) callconv(.c) c_int);
pub const mlx_conv3d = routed("mlx_conv3d", fn (res: *mlx_array, input: mlx_array, weight: mlx_array, stride_0: c_int, stride_1: c_int, stride_2: c_int, padding_0: c_int, padding_1: c_int, padding_2: c_int, dilation_0: c_int, dilation_1: c_int, dilation_2: c_int, groups: c_int, s: mlx_stream) callconv(.c) c_int);
pub const mlx_conv_transpose1d = routed("mlx_conv_transpose1d", fn (res: *mlx_array, input: mlx_array, weight: mlx_array, stride: c_int, padding: c_int, dilation: c_int, output_padding: c_int, groups: c_int, s: mlx_stream) callconv(.c) c_int);
pub const mlx_conv_transpose2d = routed("mlx_conv_transpose2d", fn (res: *mlx_array, input: mlx_array, weight: mlx_array, stride_0: c_int, stride_1: c_int, padding_0: c_int, padding_1: c_int, dilation_0: c_int, dilation_1: c_int, output_padding_0: c_int, output_padding_1: c_int, groups: c_int, s: mlx_stream) callconv(.c) c_int);
pub const mlx_conv_transpose3d = routed("mlx_conv_transpose3d", fn (res: *mlx_array, input: mlx_array, weight: mlx_array, stride_0: c_int, stride_1: c_int, stride_2: c_int, padding_0: c_int, padding_1: c_int, padding_2: c_int, dilation_0: c_int, dilation_1: c_int, dilation_2: c_int, output_padding_0: c_int, output_padding_1: c_int, output_padding_2: c_int, groups: c_int, s: mlx_stream) callconv(.c) c_int);
// Seed-exact normal noise: pair with mlx_random_key(seed).
pub const mlx_random_normal = routed("mlx_random_normal", fn (res: *mlx_array, shape: [*]const c_int, shape_num: usize, dtype: mlx_dtype, loc: f32, scale: f32, key: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_argpartition_axis = routed("mlx_argpartition_axis", fn (res: *mlx_array, a: mlx_array, kth: c_int, axis: c_int, s: mlx_stream) callconv(.c) c_int);
pub const mlx_take_along_axis = routed("mlx_take_along_axis", fn (res: *mlx_array, a: mlx_array, indices: mlx_array, axis: c_int, s: mlx_stream) callconv(.c) c_int);
pub const mlx_put_along_axis = routed("mlx_put_along_axis", fn (res: *mlx_array, a: mlx_array, indices: mlx_array, values: mlx_array, axis: c_int, s: mlx_stream) callconv(.c) c_int);
pub const mlx_logical_and = routed("mlx_logical_and", fn (res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_logical_or = routed("mlx_logical_or", fn (res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_repeat_axis = routed("mlx_repeat_axis", fn (res: *mlx_array, arr: mlx_array, repeats: c_int, axis: c_int, s: mlx_stream) callconv(.c) c_int);
pub const mlx_tile = routed("mlx_tile", fn (res: *mlx_array, arr: mlx_array, reps: [*]const c_int, reps_num: usize, s: mlx_stream) callconv(.c) c_int);
pub const mlx_log1p = routed("mlx_log1p", fn (res: *mlx_array, a: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_logaddexp = routed("mlx_logaddexp", fn (res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_stack_axis = routed("mlx_stack_axis", fn (res: *mlx_array, arrays: mlx_vector_array, axis: c_int, s: mlx_stream) callconv(.c) c_int);
pub const mlx_split = routed("mlx_split", fn (res: *mlx_vector_array, a: mlx_array, num_splits: c_int, axis: c_int, s: mlx_stream) callconv(.c) c_int);

pub const mlx_optional_dtype = extern struct {
    value: mlx_dtype = .float32,
    has_value: bool = false,
};

// ── Fast ops ──
pub const mlx_fast_rms_norm = routed("mlx_fast_rms_norm", fn (res: *mlx_array, x: mlx_array, weight: mlx_array, eps: f32, s: mlx_stream) callconv(.c) c_int);
pub const mlx_fast_layer_norm = routed("mlx_fast_layer_norm", fn (res: *mlx_array, x: mlx_array, weight: mlx_array, bias: mlx_array, eps: f32, s: mlx_stream) callconv(.c) c_int);
pub const mlx_fast_rope = routed("mlx_fast_rope", fn (res: *mlx_array, x: mlx_array, dims: c_int, traditional: bool, base: mlx_optional_float, scale: f32, offset: c_int, freqs: mlx_array, s: mlx_stream) callconv(.c) c_int);
// `mlx_fast_rope_dynamic` accepts a per-row offset array (shape [B], int32) so a single
// kernel launch handles N requests at different KV positions during batched decode.
pub const mlx_fast_rope_dynamic = routed("mlx_fast_rope_dynamic", fn (res: *mlx_array, x: mlx_array, dims: c_int, traditional: bool, base: mlx_optional_float, scale: f32, offset: mlx_array, freqs: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_fast_scaled_dot_product_attention = routed("mlx_fast_scaled_dot_product_attention", fn (res: *mlx_array, queries: mlx_array, keys: mlx_array, values: mlx_array, scale: f32, mask_mode: [*:0]const u8, mask_arr: mlx_array, sinks: mlx_array, force_fused: bool, s: mlx_stream) callconv(.c) c_int);

// ── Vector of strings (for custom metal kernels) ──
pub const mlx_vector_string = extern struct { ctx: ?*anyopaque = null };
/// A list of names made while the engine is on is the engine's (`engine.isStrings`); only a kernel's maker reads one.
pub fn mlx_vector_string_new() mlx_vector_string {
    if (engine.enabled) return engine.stringsNew(null, 0);
    return raw_kernel.mlx_vector_string_new();
}
pub fn mlx_vector_string_new_data(data: [*]const [*:0]const u8, size: usize) mlx_vector_string {
    if (engine.enabled) return engine.stringsNew(data, size);
    return raw_kernel.mlx_vector_string_new_data(data, size);
}
pub fn mlx_vector_string_free(vec: mlx_vector_string) c_int {
    if (engine.isStrings(vec)) {
        engine.stringsFree(vec);
        return 0;
    }
    return raw_kernel.mlx_vector_string_free(vec);
}
pub fn mlx_vector_string_size(vec: mlx_vector_string) usize {
    if (engine.isStrings(vec)) return engine.stringsItems(vec).len;
    return raw_kernel.mlx_vector_string_size(vec);
}
pub fn mlx_vector_string_get(res: *[*:0]u8, vec: mlx_vector_string, idx: usize) c_int {
    if (engine.isStrings(vec)) {
        const items = engine.stringsItems(vec);
        if (idx >= items.len) return engine.refuse("mlx_vector_string_get", "an index past the list's end");
        res.* = items[idx].ptr;
        return 0;
    }
    return raw_kernel.mlx_vector_string_get(res, vec, idx);
}

// ── Custom Metal kernels ──
pub const mlx_fast_metal_kernel_config = extern struct { ctx: ?*anyopaque = null };
/// With the engine on, a config of the engine's own (`engine.isConfig`): filling it calls nothing in MLX, and MLX's
/// config of the same contents (its twin) is made when it first launches on MLX's side. With the engine off, MLX's
/// config; a test process mirrors that one for the engine as it is filled.
pub fn mlx_fast_metal_kernel_config_new() mlx_fast_metal_kernel_config {
    if (engine.enabled) return engine.configNew();
    return raw_kernel.mlx_fast_metal_kernel_config_new();
}

/// Frees the MLX twin of a config of the engine's own: its contents changed, or it is freed.
fn dropConfigTwin(c: *engine.kernel.Config) void {
    const twin: mlx_fast_metal_kernel_config = .{ .ctx = c.twin orelse return };
    kernel_keys.configFree(twin);
    c.twin = null;
    // A model on the engine drops a config a model on MLX launched: MLX frees its twin at its next turn.
    if (atRest() != null) if (owed.configs.append(std.heap.c_allocator, twin)) |_| return else |_| {};
    _ = raw_kernel.mlx_fast_metal_kernel_config_free(twin);
}

/// MLX work asked for while its allocator is at rest, done when the inference thread next turns to a model on MLX or
/// the last hold ends (`payOwed`): the free of a config of MLX's (a twin, or one a cache shared by both sides
/// evicts), the wired limit a load or an unload moved, and the pool's cap and clear.
const owed = struct {
    var configs: std.ArrayList(mlx_fast_metal_kernel_config) = .empty;
    var wired: bool = false;
    var cache_limit: ?usize = null;
    var clear: bool = false;
};

/// Inference thread, MLX's allocator not at rest.
pub fn payOwed() void {
    for (owed.configs.items) |twin| _ = raw_kernel.mlx_fast_metal_kernel_config_free(twin);
    owed.configs.clearRetainingCapacity();
    if (owed.cache_limit) |limit| {
        owed.cache_limit = null;
        var prev: usize = 0;
        _ = raw.mlx_set_cache_limit(&prev, limit);
    }
    if (owed.clear) {
        owed.clear = false;
        _ = raw.mlx_clear_cache();
    }
    if (!owed.wired) return;
    owed.wired = false;
    _ = applyWiredPolicy();
}

/// Whether MLX is owed work for its next turn.
pub fn owes() bool {
    return owed.wired or owed.cache_limit != null or owed.clear or owed.configs.items.len != 0;
}

/// MLX's config for a launch on MLX's side: `cls` itself, or the twin of a config of the engine's own, filled in the
/// order its maker filled it.
fn mlxConfig(cls: mlx_fast_metal_kernel_config) ?mlx_fast_metal_kernel_config {
    if (!engine.isConfig(cls)) {
        // A config of MLX's whose grid was last set inside the engine's scope gets it now.
        if (engine.enabled) if (engine.caughtUp(cls)) |m| {
            if (raw_kernel.mlx_fast_metal_kernel_config_set_grid(cls, m.grid[0], m.grid[1], m.grid[2]) != 0) return null;
            if (raw_kernel.mlx_fast_metal_kernel_config_set_thread_group(cls, m.group[0], m.group[1], m.group[2]) != 0) return null;
        };
        return cls;
    }
    const c = engine.configOf(cls);
    if (c.twin) |twin| {
        if (c.behind) {
            c.behind = false;
            if (raw_kernel.mlx_fast_metal_kernel_config_set_grid(.{ .ctx = twin }, c.grid[0], c.grid[1], c.grid[2]) != 0) return null;
            if (raw_kernel.mlx_fast_metal_kernel_config_set_thread_group(.{ .ctx = twin }, c.group[0], c.group[1], c.group[2]) != 0) return null;
        }
        return .{ .ctx = twin };
    }
    const twin = raw_kernel.mlx_fast_metal_kernel_config_new();
    if (twin.ctx == null) return null;
    var out: usize = 0;
    var targ: usize = 0;
    var failed = false;
    for (c.order[0 .. c.n_out + c.n_targs]) |is_output| {
        if (is_output) {
            kernel_keys.configNote(twin, " ->{t}", .{c.out_dtypes[out]});
            failed = failed or raw_kernel.mlx_fast_metal_kernel_config_add_output_arg(twin, &c.out_shapes[out], c.out_ndims[out], c.out_dtypes[out]) != 0;
            out += 1;
            continue;
        }
        const arg = &c.targs[targ];
        targ += 1;
        var name_buf: [arg.name.len + 1]u8 = undefined;
        @memcpy(name_buf[0..arg.name_len], arg.name[0..arg.name_len]);
        name_buf[arg.name_len] = 0;
        const name: [*:0]const u8 = @ptrCast(&name_buf);
        failed = failed or switch (arg.value) {
            .int => |v| blk: {
                kernel_keys.configNote(twin, " {s}={d}", .{ name, v });
                break :blk raw_kernel.mlx_fast_metal_kernel_config_add_template_arg_int(twin, name, v);
            },
            .dtype => |v| blk: {
                kernel_keys.configNote(twin, " {s}={t}", .{ name, v });
                break :blk raw_kernel.mlx_fast_metal_kernel_config_add_template_arg_dtype(twin, name, v);
            },
            .boolean => |v| raw_kernel.mlx_fast_metal_kernel_config_add_template_arg_bool(twin, name, v),
        } != 0;
    }
    failed = failed or raw_kernel.mlx_fast_metal_kernel_config_set_grid(twin, c.grid[0], c.grid[1], c.grid[2]) != 0;
    failed = failed or raw_kernel.mlx_fast_metal_kernel_config_set_thread_group(twin, c.group[0], c.group[1], c.group[2]) != 0;
    if (failed) {
        kernel_keys.configFree(twin);
        _ = raw_kernel.mlx_fast_metal_kernel_config_free(twin);
        return null;
    }
    c.twin = twin.ctx;
    return twin;
}

pub fn mlx_fast_metal_kernel_config_free(cls: mlx_fast_metal_kernel_config) c_int {
    if (engine.isConfig(cls)) {
        dropConfigTwin(engine.configOf(cls));
        engine.configFree(cls);
        return 0;
    }
    engine.forgetConfig(cls);
    kernel_keys.configFree(cls);
    if (atRest() != null) if (owed.configs.append(std.heap.c_allocator, cls)) |_| return 0 else |_| {};
    return raw_kernel.mlx_fast_metal_kernel_config_free(cls);
}
pub fn mlx_fast_metal_kernel_config_add_output_arg(cls: mlx_fast_metal_kernel_config, shape: [*]const c_int, size: usize, dtype: mlx_dtype) c_int {
    if (engine.isConfig(cls)) {
        dropConfigTwin(engine.configOf(cls));
        return engine.configOutput(cls, shape, size, dtype);
    }
    engine.noteOutput(cls, shape, size, dtype);
    kernel_keys.configNote(cls, " ->{t}", .{dtype});
    return raw_kernel.mlx_fast_metal_kernel_config_add_output_arg(cls, shape, size, dtype);
}
pub fn mlx_fast_metal_kernel_config_set_grid(cls: mlx_fast_metal_kernel_config, g1: c_int, g2: c_int, g3: c_int) c_int {
    if (engine.isConfig(cls)) {
        const c = engine.configOf(cls);
        c.grid = .{ g1, g2, g3 };
        const twin = c.twin orelse return 0;
        // Inside the engine's scope MLX's twin waits for its next launch on MLX's side (`mlxConfig`).
        if (engine.active) {
            c.behind = true;
            return 0;
        }
        return raw_kernel.mlx_fast_metal_kernel_config_set_grid(.{ .ctx = twin }, g1, g2, g3);
    }
    engine.noteGrid(cls, g1, g2, g3);
    if (engine.keepsBehind(cls)) return 0;
    return raw_kernel.mlx_fast_metal_kernel_config_set_grid(cls, g1, g2, g3);
}
pub fn mlx_fast_metal_kernel_config_set_thread_group(cls: mlx_fast_metal_kernel_config, t1: c_int, t2: c_int, t3: c_int) c_int {
    if (engine.isConfig(cls)) {
        const c = engine.configOf(cls);
        c.group = .{ t1, t2, t3 };
        const twin = c.twin orelse return 0;
        if (engine.active) {
            c.behind = true;
            return 0;
        }
        return raw_kernel.mlx_fast_metal_kernel_config_set_thread_group(.{ .ctx = twin }, t1, t2, t3);
    }
    engine.noteGroup(cls, t1, t2, t3);
    if (engine.keepsBehind(cls)) return 0;
    return raw_kernel.mlx_fast_metal_kernel_config_set_thread_group(cls, t1, t2, t3);
}
pub fn mlx_fast_metal_kernel_config_add_template_arg_dtype(cls: mlx_fast_metal_kernel_config, name: [*:0]const u8, dtype: mlx_dtype) c_int {
    if (engine.isConfig(cls)) {
        dropConfigTwin(engine.configOf(cls));
        return engine.configTemplate(cls, name, .{ .dtype = dtype });
    }
    engine.noteTemplate(cls, name, .{ .dtype = dtype });
    kernel_keys.configNote(cls, " {s}={t}", .{ name, dtype });
    return raw_kernel.mlx_fast_metal_kernel_config_add_template_arg_dtype(cls, name, dtype);
}
pub const mlx_any_axes = routed("mlx_any_axes", fn (res: *mlx_array, a: mlx_array, axes: [*]const c_int, axes_num: usize, keepdims: bool, s: mlx_stream) callconv(.c) c_int);
pub fn mlx_fast_metal_kernel_config_add_template_arg_int(cls: mlx_fast_metal_kernel_config, name: [*:0]const u8, value: c_int) c_int {
    if (engine.isConfig(cls)) {
        dropConfigTwin(engine.configOf(cls));
        return engine.configTemplate(cls, name, .{ .int = value });
    }
    engine.noteTemplate(cls, name, .{ .int = value });
    kernel_keys.configNote(cls, " {s}={d}", .{ name, value });
    return raw_kernel.mlx_fast_metal_kernel_config_add_template_arg_int(cls, name, value);
}
/// No kernel of the engine's serving takes a bool template arg, so on MLX's config this one notes nothing for the
/// engine's mirror or the keys.
pub fn mlx_fast_metal_kernel_config_add_template_arg_bool(cls: mlx_fast_metal_kernel_config, name: [*:0]const u8, value: bool) c_int {
    if (engine.isConfig(cls)) {
        dropConfigTwin(engine.configOf(cls));
        return engine.configTemplate(cls, name, .{ .boolean = value });
    }
    return raw_kernel.mlx_fast_metal_kernel_config_add_template_arg_bool(cls, name, value);
}

pub const mlx_fast_metal_kernel = extern struct { ctx: ?*anyopaque = null };
var raw_source_env: ?bool = null;

/// Every custom kernel enters MLX here with its source and header minified once, the header without the helpers the
/// source never reaches: MLX rebuilds and hashes both on every launch. `SUSHI_KERNEL_RAW_SOURCE=1` passes them as
/// written, so a Metal compile error names the source lines.
pub fn mlx_fast_metal_kernel_new(name: [*:0]const u8, input_names: mlx_vector_string, output_names: mlx_vector_string, source: [*:0]const u8, header: [*:0]const u8, ensure_row_contiguous: bool, atomic_outputs: bool) mlx_fast_metal_kernel {
    if (io_util.diagEnvOnCached(&raw_source_env, "SUSHI_KERNEL_RAW_SOURCE")) return kernelNewSource(name, input_names, output_names, source, header, ensure_row_contiguous, atomic_outputs);
    const alloc = std.heap.page_allocator;
    const text = minifiedText(alloc, source, header) catch return kernelNewSource(name, input_names, output_names, source, header, ensure_row_contiguous, atomic_outputs);
    defer alloc.free(text.source);
    defer alloc.free(text.header);
    return kernelNewSource(name, input_names, output_names, text.source, text.header, ensure_row_contiguous, atomic_outputs);
}

/// A kernel's source and header minified, the header without the helpers the source never reaches.
fn minifiedText(alloc: std.mem.Allocator, source: [*:0]const u8, header: [*:0]const u8) !struct { source: [:0]u8, header: [:0]u8 } {
    const src = try metal_minify.minify(alloc, std.mem.span(source));
    errdefer alloc.free(src);
    const whole = try metal_minify.minify(alloc, std.mem.span(header));
    defer alloc.free(whole);
    return .{ .source = src, .header = try metal_minify.shake(alloc, whole, src) };
}

/// The kernel exactly as given, its strings copied. With the engine on it is a kernel of the engine's own
/// (`engine.isKernel`), and MLX's kernel of the same source (its twin) is made when it first launches on MLX's side.
pub fn kernelNewSource(name: [*:0]const u8, input_names: mlx_vector_string, output_names: mlx_vector_string, source: [*:0]const u8, header: [*:0]const u8, ensure_row_contiguous: bool, atomic_outputs: bool) mlx_fast_metal_kernel {
    if (engine.enabled) return engine.kernelNative(name, input_names, output_names, source, header, ensure_row_contiguous, atomic_outputs);
    const ins = MlxStrings.of(input_names) orelse return .{};
    defer ins.deinit();
    const outs = MlxStrings.of(output_names) orelse return .{};
    defer outs.deinit();
    const k = raw_kernel.mlx_fast_metal_kernel_new(name, ins.vec, outs.vec, source, header, ensure_row_contiguous, atomic_outputs);
    engine.kernelNew(k, name, input_names, output_names, source, header, ensure_row_contiguous, atomic_outputs);
    kernel_keys.kernelNew(k, name, source, header, ensure_row_contiguous, atomic_outputs);
    return k;
}

/// A list of names as MLX's own vector: the list itself, or a copy of one the engine holds.
const MlxStrings = struct {
    vec: mlx_vector_string,
    copied: bool,

    fn of(v: mlx_vector_string) ?MlxStrings {
        if (!engine.isStrings(v)) return .{ .vec = v, .copied = false };
        return ofItems(engine.stringsItems(v));
    }

    fn ofItems(items: anytype) ?MlxStrings {
        var ptrs: [engine.kernel.max_inputs][*:0]const u8 = undefined;
        if (items.len > ptrs.len) return null;
        for (items, 0..) |text, i| ptrs[i] = text.ptr;
        const vec = raw_kernel.mlx_vector_string_new_data(&ptrs, items.len);
        return if (vec.ctx == null) null else .{ .vec = vec, .copied = true };
    }

    fn deinit(m: MlxStrings) void {
        if (m.copied) _ = raw_kernel.mlx_vector_string_free(m.vec);
    }
};

/// MLX's kernel for a launch on MLX's side: `cls` itself, or the twin of a kernel of the engine's own.
fn mlxKernel(cls: mlx_fast_metal_kernel) ?mlx_fast_metal_kernel {
    if (!engine.isKernel(cls)) return cls;
    const def = engine.kernelOf(cls);
    if (def.twin) |twin| return .{ .ctx = twin };
    const alloc = std.heap.page_allocator;
    var texts: [3][:0]u8 = undefined;
    for ([_][]const u8{ def.name, def.source, def.header }, 0..) |text, i| {
        texts[i] = alloc.dupeSentinel(u8, text, 0) catch {
            for (texts[0..i]) |held| alloc.free(held);
            return null;
        };
    }
    defer for (texts) |held| alloc.free(held);
    var names: [2][engine.kernel.max_inputs][:0]u8 = undefined;
    var counts = [2]usize{ 0, 0 };
    defer for (names, counts) |list, n| for (list[0..n]) |held| alloc.free(held);
    for ([_][]const []u8{ def.inputs, def.outputs }, 0..) |list, which| {
        if (list.len > names[which].len) return null;
        for (list) |text| {
            names[which][counts[which]] = alloc.dupeSentinel(u8, text, 0) catch return null;
            counts[which] += 1;
        }
    }
    const ins = MlxStrings.ofItems(names[0][0..counts[0]]) orelse return null;
    defer ins.deinit();
    const outs = MlxStrings.ofItems(names[1][0..counts[1]]) orelse return null;
    defer outs.deinit();
    const twin = raw_kernel.mlx_fast_metal_kernel_new(texts[0].ptr, ins.vec, outs.vec, texts[1].ptr, texts[2].ptr, def.ensure_row_contiguous, def.atomic_outputs);
    if (twin.ctx == null) return null;
    kernel_keys.kernelNew(twin, texts[0].ptr, texts[1].ptr, texts[2].ptr, def.ensure_row_contiguous, def.atomic_outputs);
    def.twin = twin.ctx;
    return twin;
}

pub fn mlx_fast_metal_kernel_free(cls: mlx_fast_metal_kernel) c_int {
    if (engine.isKernel(cls)) {
        const def = engine.kernelOf(cls);
        const twin: mlx_fast_metal_kernel = .{ .ctx = def.twin orelse return 0 };
        def.twin = null;
        kernel_keys.kernelFree(twin);
        return raw_kernel.mlx_fast_metal_kernel_free(twin);
    }
    engine.kernelFree(cls);
    kernel_keys.kernelFree(cls);
    return raw_kernel.mlx_fast_metal_kernel_free(cls);
}
/// A kernel's one output written over `into` on the engine's stream (`engine.kernelApplyInto`); null where the stream
/// is MLX's, which writes only fresh outputs.
pub fn kernelApplyInto(outputs: *mlx_vector_array, cls: mlx_fast_metal_kernel, inputs: mlx_vector_array, config: mlx_fast_metal_kernel_config, s: mlx_stream, into: mlx_array) ?c_int {
    if (!engine.enabled) return null;
    return switch (engine.side(.{ inputs, config, s })) {
        .engine => if (faultAtEntry()) 1 else engine.kernelApplyInto(outputs, cls, inputs, config, into),
        else => null,
    };
}

/// A custom kernel applied into `dsts` on the engine's stream (`engine.kernelApplyInPlace`); null when the call is
/// not the engine's, so the caller takes MLX's path.
pub fn kernelApplyInPlace(cls: mlx_fast_metal_kernel, inputs: mlx_vector_array, config: mlx_fast_metal_kernel_config, s: mlx_stream, dsts: []const mlx_array) ?c_int {
    if (!engine.enabled) return null;
    return switch (engine.side(.{ inputs, config, s })) {
        .engine => if (faultAtEntry()) 1 else engine.kernelApplyInPlace(cls, inputs, config, dsts),
        else => null,
    };
}

/// A kernel whose last output is the engine's counter page on the engine's stream (`engine.kernelApplyCounted`); null
/// where the stream is MLX's, which has no such page.
pub fn kernelApplyCounted(outputs: *mlx_vector_array, cls: mlx_fast_metal_kernel, inputs: mlx_vector_array, config: mlx_fast_metal_kernel_config, s: mlx_stream) ?c_int {
    if (!engine.enabled) return null;
    return switch (engine.side(.{ inputs, config, s })) {
        .engine => if (faultAtEntry()) 1 else engine.kernelApplyCounted(outputs, cls, inputs, config),
        else => null,
    };
}

pub fn mlx_fast_metal_kernel_apply(outputs: *mlx_vector_array, cls: mlx_fast_metal_kernel, inputs: mlx_vector_array, config: mlx_fast_metal_kernel_config, s: mlx_stream) c_int {
    if (engine.enabled) switch (engine.side(.{ inputs, config, s })) {
        // The MLX path's entry is counted (`raw_kernel`); the engine's is this one.
        .engine => return if (faultAtEntry()) 1 else engine.mlx_fast_metal_kernel_apply(outputs, cls, inputs, config, s),
        .leak => return engine.refuse("mlx_fast_metal_kernel_apply", "an MLX stream inside a forward of the engine"),
        .mlx => if (engine.isVec(inputs)) return engine.refuse("mlx_fast_metal_kernel_apply", "a vector of the engine on an MLX stream"),
    };
    const k = mlxKernel(cls) orelse return engine.refuse("mlx_fast_metal_kernel_apply", "MLX could not make the kernel");
    const c = mlxConfig(config) orelse return engine.refuse("mlx_fast_metal_kernel_apply", "MLX could not make the config");
    kernel_keys.apply(k, inputs, c);
    return raw_kernel.mlx_fast_metal_kernel_apply(outputs, k, inputs, c, s);
}

/// The custom-kernel entry points under the wrappers above, which note each kernel's library key.
const raw_kernel = struct {
    const mlx_vector_string_new = counted("mlx_vector_string_new", fn () callconv(.c) mlx_vector_string);
    const mlx_vector_string_new_data = counted("mlx_vector_string_new_data", fn (data: [*]const [*:0]const u8, size: usize) callconv(.c) mlx_vector_string);
    const mlx_vector_string_free = counted("mlx_vector_string_free", fn (vec: mlx_vector_string) callconv(.c) c_int);
    const mlx_vector_string_size = counted("mlx_vector_string_size", fn (vec: mlx_vector_string) callconv(.c) usize);
    const mlx_vector_string_get = counted("mlx_vector_string_get", fn (res: *[*:0]u8, vec: mlx_vector_string, idx: usize) callconv(.c) c_int);
    const mlx_fast_metal_kernel_config_new = counted("mlx_fast_metal_kernel_config_new", fn () callconv(.c) mlx_fast_metal_kernel_config);
    const mlx_fast_metal_kernel_config_free = counted("mlx_fast_metal_kernel_config_free", fn (cls: mlx_fast_metal_kernel_config) callconv(.c) c_int);
    const mlx_fast_metal_kernel_config_set_grid = counted("mlx_fast_metal_kernel_config_set_grid", fn (cls: mlx_fast_metal_kernel_config, g1: c_int, g2: c_int, g3: c_int) callconv(.c) c_int);
    const mlx_fast_metal_kernel_config_set_thread_group = counted("mlx_fast_metal_kernel_config_set_thread_group", fn (cls: mlx_fast_metal_kernel_config, t1: c_int, t2: c_int, t3: c_int) callconv(.c) c_int);
    const mlx_fast_metal_kernel_config_add_output_arg = counted("mlx_fast_metal_kernel_config_add_output_arg", fn (cls: mlx_fast_metal_kernel_config, shape: [*]const c_int, size: usize, dtype: mlx_dtype) callconv(.c) c_int);
    const mlx_fast_metal_kernel_config_add_template_arg_dtype = counted("mlx_fast_metal_kernel_config_add_template_arg_dtype", fn (cls: mlx_fast_metal_kernel_config, name: [*:0]const u8, dtype: mlx_dtype) callconv(.c) c_int);
    const mlx_fast_metal_kernel_config_add_template_arg_int = counted("mlx_fast_metal_kernel_config_add_template_arg_int", fn (cls: mlx_fast_metal_kernel_config, name: [*:0]const u8, value: c_int) callconv(.c) c_int);
    const mlx_fast_metal_kernel_config_add_template_arg_bool = counted("mlx_fast_metal_kernel_config_add_template_arg_bool", fn (cls: mlx_fast_metal_kernel_config, name: [*:0]const u8, value: bool) callconv(.c) c_int);
    const mlx_fast_metal_kernel_new = counted("mlx_fast_metal_kernel_new", fn (name: [*:0]const u8, input_names: mlx_vector_string, output_names: mlx_vector_string, source: [*:0]const u8, header: [*:0]const u8, ensure_row_contiguous: bool, atomic_outputs: bool) callconv(.c) mlx_fast_metal_kernel);
    const mlx_fast_metal_kernel_free = counted("mlx_fast_metal_kernel_free", fn (cls: mlx_fast_metal_kernel) callconv(.c) c_int);
    const mlx_fast_metal_kernel_apply = counted("mlx_fast_metal_kernel_apply", fn (outputs: *mlx_vector_array, cls: mlx_fast_metal_kernel, inputs: mlx_vector_array, config: mlx_fast_metal_kernel_config, s: mlx_stream) callconv(.c) c_int);
};

// ── Random ──
pub const mlx_random_categorical = routed("mlx_random_categorical", fn (res: *mlx_array, logits: mlx_array, axis: c_int, key: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_random_key = counted("mlx_random_key", fn (res: *mlx_array, seed: u64) callconv(.c) c_int);
// Uniform noise — Kokoro's SineGen draws a random initial phase per harmonic.
// Bounds are ARRAYS, unlike mlx_random_normal's scalar loc/scale.
pub const mlx_random_uniform = routed("mlx_random_uniform", fn (res: *mlx_array, low: mlx_array, high: mlx_array, shape: [*]const c_int, shape_num: usize, dtype: mlx_dtype, key: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_random_seed = counted("mlx_random_seed", fn (seed: u64) callconv(.c) c_int);
pub const mlx_random_bits = routed("mlx_random_bits", fn (res: *mlx_array, shape: [*]const c_int, shape_num: usize, width: c_int, key: mlx_array, s: mlx_stream) callconv(.c) c_int);
// Uniform random integers in [low, high).
pub const mlx_random_randint = routed("mlx_random_randint", fn (res: *mlx_array, low: mlx_array, high: mlx_array, shape: [*]const c_int, shape_num: usize, dtype: mlx_dtype, key: mlx_array, s: mlx_stream) callconv(.c) c_int);

// ── Reductions and elementwise ops ──
pub const mlx_logsumexp_axis = routed("mlx_logsumexp_axis", fn (res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, s: mlx_stream) callconv(.c) c_int);
pub const mlx_all = routed("mlx_all", fn (res: *mlx_array, a: mlx_array, keepdims: bool, s: mlx_stream) callconv(.c) c_int);
pub const mlx_isfinite = routed("mlx_isfinite", fn (res: *mlx_array, a: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_clip = routed("mlx_clip", fn (res: *mlx_array, a: mlx_array, a_min: mlx_array, a_max: mlx_array, s: mlx_stream) callconv(.c) c_int);
pub const mlx_contiguous = routed("mlx_contiguous", fn (res: *mlx_array, a: mlx_array, allow_col_major: bool, s: mlx_stream) callconv(.c) c_int);
pub const mlx_max = routed("mlx_max", fn (res: *mlx_array, a: mlx_array, keepdims: bool, s: mlx_stream) callconv(.c) c_int);
pub fn mlx_array_item_bool(res: *bool, arr: mlx_array) c_int {
    if (engine.enabled and engine.isOurs(arr)) return engine.item(bool, res, arr);
    return eval_raw.mlx_array_item_bool(res, arr);
}

// ── Batch eval ──
pub fn mlx_eval(outputs: mlx_vector_array) c_int {
    if (faultAtEntry()) return 1;
    if (engine.enabled and engine.isVec(outputs)) return engine.mlx_eval(outputs);
    return eval_raw.mlx_eval(outputs);
}
pub fn mlx_async_eval(outputs: mlx_vector_array) c_int {
    if (faultAtEntry()) return 1;
    if (engine.enabled and engine.isVec(outputs)) return engine.mlx_async_eval(outputs);
    return eval_raw.mlx_async_eval(outputs);
}
const eval_raw = struct {
    const mlx_array_eval = counted("mlx_array_eval", fn (arr: mlx_array) callconv(.c) c_int);
    const mlx_array_item_float32 = counted("mlx_array_item_float32", fn (res: *f32, arr: mlx_array) callconv(.c) c_int);
    const mlx_array_item_int32 = counted("mlx_array_item_int32", fn (res: *i32, arr: mlx_array) callconv(.c) c_int);
    const mlx_array_item_bool = counted("mlx_array_item_bool", fn (res: *bool, arr: mlx_array) callconv(.c) c_int);
    const mlx_eval = counted("mlx_eval", fn (outputs: mlx_vector_array) callconv(.c) c_int);
    const mlx_async_eval = counted("mlx_async_eval", fn (outputs: mlx_vector_array) callconv(.c) c_int);
};

// ── Memory management ──
// ── MLX's allocator ──
// Under a held scope (`engine.hold`) none of these calls MLX: a reading is the one MLX was last left with, a pool
// clear and the pool's cap are owed to MLX's next turn (`payOwed`), and a peak reset and the other limits do nothing,
// since MLX allocates nothing there (`rest_swallows` counts them all). A limit hands back the value given.

fn atRest() ?*engine.MlxAtRest {
    return if (engine.mlx_at_rest.on.load(.acquire)) &engine.mlx_at_rest else null;
}

/// A setting asked of MLX at rest: counted, and nothing else.
fn swallowedAtRest(comptime name: []const u8) bool {
    if (atRest() == null) return false;
    _ = rest_swallows.fetchAdd(1, .monotonic);
    if (comptime builtin.is_test) by_name.add("at rest: " ++ name);
    return true;
}

var allocator_read_at: ?u64 = null;
var pool_emptied_at: ?u64 = null;

/// MLX's allocator read into `r`, its pool emptied first when `empty`, unless nothing has changed in MLX since the
/// last time: then what `r` holds is still true and no call is made. A process that never called MLX has nothing
/// there to read.
fn readAllocator(r: *engine.MlxAtRest, empty: bool) void {
    const at = changes.load(.monotonic);
    if (allocator_read_at == at and (!empty or pool_emptied_at == at)) return;
    var now: [4]usize = @splat(0);
    if (calls.load(.monotonic) != 0) {
        if (empty) _ = raw.mlx_clear_cache();
        _ = raw.mlx_get_active_memory(&now[0]);
        _ = raw.mlx_get_cache_memory(&now[1]);
        _ = raw.mlx_get_peak_memory(&now[2]);
        now[2] = @max(now[2], now[0]);
        _ = raw.mlx_get_external_memory(&now[3]);
    }
    r.active.store(now[0], .monotonic);
    r.cache.store(now[1], .monotonic);
    r.peak.store(now[2], .monotonic);
    r.external.store(now[3], .monotonic);
    allocator_read_at = changes.load(.monotonic);
    if (empty) pool_emptied_at = allocator_read_at;
}

/// MLX's allocator goes to rest (inference thread): for the first hold with its pool emptied (`empty`), after a turn
/// of a model on MLX as that model left it, since the pool is that model's to reuse at its next turn.
pub fn restAllocator(r: *engine.MlxAtRest, empty: bool) void {
    engine.settleMlxAliases();
    readAllocator(r, empty);
    r.on.store(true, .release);
}

/// The end of a process's MLX setup, on the thread that loads: a load that is the engine's and finds MLX as it was
/// left here asks it nothing.
pub fn noteAllocator() void {
    readAllocator(&engine.mlx_at_rest, true);
}

/// At rest the clear is owed to MLX's next turn (`payOwed`): a pool a caller empties is never kept by a held scope.
pub fn mlx_clear_cache() c_int {
    if (!swallowedAtRest("mlx_clear_cache")) return raw.mlx_clear_cache();
    owed.clear = true;
    return 0;
}
pub fn mlx_get_active_memory(res: *usize) c_int {
    const r = atRest() orelse return raw.mlx_get_active_memory(res);
    res.* = r.active.load(.monotonic);
    return 0;
}
pub fn mlx_get_cache_memory(res: *usize) c_int {
    const r = atRest() orelse return raw.mlx_get_cache_memory(res);
    res.* = r.cache.load(.monotonic);
    return 0;
}
/// MLX's reset starts its peak from zero, not from what is active: the high-water read here never sits under it.
pub fn mlx_get_peak_memory(res: *usize) c_int {
    const r = atRest() orelse return livePeak(res);
    res.* = r.peak.load(.monotonic);
    return 0;
}
fn livePeak(res: *usize) c_int {
    var active: usize = 0;
    const rc = raw.mlx_get_peak_memory(res);
    if (rc != 0) return rc;
    if (raw.mlx_get_active_memory(&active) == 0) res.* = @max(res.*, active);
    return 0;
}
pub fn mlx_reset_peak_memory() c_int {
    const r = atRest() orelse return raw.mlx_reset_peak_memory();
    r.peak.store(r.active.load(.monotonic), .monotonic);
    return 0;
}
pub fn mlx_get_external_memory(res: *usize) c_int {
    const r = atRest() orelse return raw.mlx_get_external_memory(res);
    res.* = r.external.load(.monotonic);
    return 0;
}
/// High-water of MLX's own active bytes; at rest, what it holds.
pub fn mlx_get_transient_peak(res: *usize) c_int {
    const r = atRest() orelse return raw.mlx_get_transient_peak(res);
    res.* = r.active.load(.monotonic) -| r.external.load(.monotonic);
    return 0;
}
pub fn mlx_reset_transient_peak() c_int {
    return if (swallowedAtRest("mlx_reset_transient_peak")) 0 else raw.mlx_reset_transient_peak();
}
fn limitAtRest(comptime name: []const u8, res: *usize, limit: usize) bool {
    if (!swallowedAtRest(name)) return false;
    res.* = limit;
    return true;
}
pub fn mlx_set_memory_limit(res: *usize, limit: usize) c_int {
    return if (limitAtRest("mlx_set_memory_limit", res, limit)) 0 else raw.mlx_set_memory_limit(res, limit);
}
/// At rest the pool's cap is owed to MLX's next turn, the last one asked.
pub fn mlx_set_cache_limit(res: *usize, limit: usize) c_int {
    if (!limitAtRest("mlx_set_cache_limit", res, limit)) return raw.mlx_set_cache_limit(res, limit);
    owed.cache_limit = limit;
    return 0;
}
pub fn mlx_set_wired_limit(res: *usize, limit: usize) c_int {
    return if (limitAtRest("mlx_set_wired_limit", res, limit)) 0 else raw.mlx_set_wired_limit(res, limit);
}
pub fn mlx_set_transient_limit(res: *usize, limit: usize) c_int {
    return if (limitAtRest("mlx_set_transient_limit", res, limit)) 0 else raw.mlx_set_transient_limit(res, limit);
}
pub fn mlx_metal_set_max_mb_per_buffer(prev: *c_int, mb: c_int) c_int {
    if (!swallowedAtRest("mlx_metal_set_max_mb_per_buffer")) return raw.mlx_metal_set_max_mb_per_buffer(prev, mb);
    prev.* = mb;
    return 0;
}

/// Command buffers committed since the process started: each costs host time in Metal's submit and completion queues.
pub const mlx_metal_command_buffer_count = counted("mlx_metal_command_buffer_count", fn (res: *usize) callconv(.c) c_int);

/// A decode tick's command buffers commit on their op count alone. MLX also commits a buffer once the inputs of its
/// ops pass a size budget, which bounds the memory a buffer keeps from being freed; at decode widths what passes it is
/// a weight or the KV the engine holds anyway, so the budget only split the forward into more buffers, each one host
/// time in Metal's submit and completion queues. Prefill keeps the budget: its memory bill is fitted to it.
pub const DecodeBuffers = struct {
    budget_mb: c_int,

    pub fn begin() DecodeBuffers {
        var prev: c_int = 0;
        _ = mlx_metal_set_max_mb_per_buffer(&prev, std.math.maxInt(c_int));
        return .{ .budget_mb = prev };
    }

    pub fn end(self: DecodeBuffers) void {
        var lifted: c_int = 0;
        _ = mlx_metal_set_max_mb_per_buffer(&lifted, self.budget_mb);
    }
};

// ── Device info ──
pub const mlx_device_info = extern struct { ctx: ?*anyopaque = null };
pub const mlx_device_info_new = emptyHandle(mlx_device_info);
pub const mlx_device_info_get = counted("mlx_device_info_get", fn (info: *mlx_device_info, dev: mlx_device) callconv(.c) c_int);
pub const mlx_device_info_free = counted("mlx_device_info_free", fn (info: mlx_device_info) callconv(.c) c_int);
pub const mlx_device_info_get_size = counted("mlx_device_info_get_size", fn (res: *usize, info: mlx_device_info, key: [*:0]const u8) callconv(.c) c_int);
pub const mlx_device_info_get_string = counted("mlx_device_info_get_string", fn (res: *[*:0]const u8, info: mlx_device_info, key: [*:0]const u8) callconv(.c) c_int);

// ── Error handler ──
pub const mlx_error_handler_func = ?*const fn ([*:0]const u8, ?*anyopaque) callconv(.c) void;
pub const mlx_set_error_handler = counted("mlx_set_error_handler", fn (handler: mlx_error_handler_func, data: ?*anyopaque, dtor: ?*const fn (?*anyopaque) callconv(.c) void) callconv(.c) void);

// ── Zig helper wrappers ──

/// Get the default compute stream. Normally the GPU (Metal) stream. When MLX
/// was built without a Metal backend (e.g. the iOS Simulator slice, where MLX's
/// Metal device can't be constructed), there is no GPU stream — fall back to the
/// CPU stream so the engine runs on the Accelerate backend. The check is cached;
/// on real hardware Metal is always available, so this is the GPU stream as
/// before with no measurable overhead.
var no_gpu_backend_cache: ?bool = null;

/// True when MLX has no Metal/GPU backend (e.g. the iOS Simulator slice built
/// with MLX_BUILD_METAL=OFF). Callers use this to take CPU-only paths and skip
/// GPU-only work (kernel-fusion JIT compilation). Always false on real hardware.
pub fn noGpuBackend() bool {
    if (no_gpu_backend_cache == null) {
        var avail: bool = false;
        _ = mlx_metal_is_available(&avail);
        no_gpu_backend_cache = !avail or (builtin.is_test and testNoGpuEnv());
    }
    return no_gpu_backend_cache.?;
}

/// A test build reads `SUSHI_TEST_NO_GPU` (empty or a leading `0` is off) as a machine whose MLX has no Metal backend.
fn testNoGpuEnv() bool {
    const v = std.c.getenv("SUSHI_TEST_NO_GPU") orelse return false;
    return v[0] != 0 and v[0] != '0';
}

/// The first line of a test that needs Metal kernels: it skips where MLX has no GPU backend.
pub fn requireGpu() !void {
    if (noGpuBackend()) return error.SkipZigTest;
}

var test_error_handler_installed = std.atomic.Value(bool).init(false);

/// A test reaches the GPU through this stream, so it latches MLX's errors like `main()` does instead of mlx-c's
/// exiting the process: a retry that reads a refusal must not depend on which test ran first.
pub fn gpuStream() mlx_stream {
    if (builtin.is_test and !test_error_handler_installed.swap(true, .acq_rel)) installErrorHandler();
    return if (noGpuBackend()) mlx_default_cpu_stream_new() else mlx_default_gpu_stream_new();
}

/// One synced element-op on the GPU stream: a queue the driver sees in use. A failure stays unlatched. The engine's
/// queue gets a pass too when it keeps weights resident, and it alone while MLX is at rest.
pub fn gpuTouch() void {
    engine.touch();
    if (atRest() != null) return;
    const had_error = errorPending();
    defer dropLatchedErrorUnless(had_error);
    const s = gpuStream();
    defer _ = mlx_stream_free(s);
    const one = mlx_array_new_float(1);
    defer _ = mlx_array_free(one);
    var out = mlx_array_new();
    defer _ = mlx_array_free(out);
    if (mlx_multiply(&out, one, one, s) != 0) return;
    _ = mlx_array_eval(out);
}

/// Print an array for debugging
pub fn printArray(label: []const u8, arr: mlx_array) void {
    _ = mlx_array_eval(arr);
    var str = mlx_string_new();
    _ = mlx_array_tostring(&str, arr);
    const data = mlx_string_data(str);
    log.debug("{s}: {s}\n", .{ label, data });
    _ = mlx_string_free(str);
}

/// Get array shape as a Zig slice
/// The dims of `arr`, which must have `rank` of them: an empty handle (a refused host array) or another rank is an
/// error, where indexing `getShape` would read past its slice.
pub fn shapeOf(arr: mlx_array, comptime rank: usize) error{ EmptyArray, ShapeRank }![rank]c_int {
    if (arr.ctx == null) return error.EmptyArray;
    const dims = getShape(arr);
    if (dims.len != rank) return error.ShapeRank;
    return dims[0..rank].*;
}

pub fn getShape(arr: mlx_array) []const c_int {
    if (arr.ctx == null) return &.{};
    const ndim = mlx_array_ndim(arr);
    if (ndim == 0) return &.{};
    return mlx_array_shape(arr)[0..ndim];
}

/// How a residual's dtype moved between two observation points.
pub const DtypeStep = enum { quiet, widened, narrowed, changed };

/// Bytes per element, for deciding whether a dtype change WIDENED the stream.
/// Only the dtypes a residual can actually hold are named; anything else is
/// reported as a plain change rather than guessed at.
pub fn dtypeBytes(d: mlx_dtype) ?u8 {
    return switch (d) {
        .float16, .bfloat16 => 2,
        .float32 => 4,
        .float64 => 8,
        else => null,
    };
}

pub fn dtypeStep(prev: mlx_dtype, cur: mlx_dtype) DtypeStep {
    if (prev == cur) return .quiet;
    const pb = dtypeBytes(prev) orelse return .changed;
    const cb = dtypeBytes(cur) orelse return .changed;
    if (cb > pb) return .widened;
    if (cb < pb) return .narrowed;
    return .changed; // same width, different kernel selection (bf16 vs f16)
}

/// One-shot latches for `[dtype-trace]`, keyed by forward-path name. A vision
/// tower and its text trunk are different paths and latch independently.
var dtype_trace_seen: [12]?[]const u8 = @splat(null);

/// True the FIRST time `path` is seen, false forever after. After the first
/// forward the trace costs nothing — a per-layer FFI dtype read on every token
/// would be a real (if small) tax, and identical repeated lines would bury the
/// one line that matters.
pub fn dtypeTraceArm(path: []const u8) bool {
    for (&dtype_trace_seen) |*slot| {
        if (slot.*) |seen| {
            if (std.mem.eql(u8, seen, path)) return false;
            continue;
        }
        slot.* = path;
        return true;
    }
    return false; // table full — stop tracing rather than spam
}

pub fn resetDtypeTraceForTest() void {
    dtype_trace_seen = @splat(null);
}

/// DIAGNOSTIC (`[dtype-trace]`): watch the residual stream's dtype across one
/// forward pass.
///
/// A residual wider than the weights promotes EVERY projection's weight on
/// read, so the cost lands as a UNIFORM multiple across attention, MLP and
/// lm_head — which reads like a platform problem, not a bug. A YaRN
/// mscale table (f32 constant multiplied into bf16 q/k) cost 3x and was
/// invisible to op count, kernel choice, KV ablation and the CPU/GPU split.
/// Endpoints alone say THAT it widened; per-layer lines on every layer are
/// unreadable. So: log both endpoints, and in between log only where the dtype
/// actually moves — one line that names the layer responsible.
pub const DtypeTrace = struct {
    path: []const u8,
    on: bool,
    last: mlx_dtype,

    pub fn begin(path: []const u8, h: mlx_array, weight: ?mlx_array) DtypeTrace {
        const on = dtypeTraceArm(path);
        const d = mlx_array_dtype(h);
        if (on) {
            log.info("[dtype-trace] {s}: residual in = {s}, first weight = {s}\n", .{
                path,
                @tagName(d),
                if (weight) |w| (if (w.ctx == null) "null" else @tagName(mlx_array_dtype(w))) else "n/a",
            });
        }
        return .{ .path = path, .on = on, .last = d };
    }

    pub fn layer(self: *DtypeTrace, h: mlx_array, layer_idx: usize) void {
        if (!self.on) return;
        const d = mlx_array_dtype(h);
        const step = dtypeStep(self.last, d);
        if (step == .quiet) return;
        log.info("[dtype-trace] {s}: residual {s} at layer {d}: {s} -> {s}\n", .{
            self.path,
            @tagName(step),
            layer_idx,
            @tagName(self.last),
            @tagName(d),
        });
        self.last = d;
    }

    pub fn end(self: *DtypeTrace, h: mlx_array) void {
        if (!self.on) return;
        log.info("[dtype-trace] {s}: residual out = {s}\n", .{ self.path, @tagName(mlx_array_dtype(h)) });
    }
};

/// Check if an mlx-c call succeeded (returns 0 on success)
/// DIAGNOSTIC op counter. Every graph-building op funnels through `check`, so
/// this counts ops ISSUED (not kernels dispatched — MLX fuses some). Only read
/// by the decode forward probe; the increment is a single relaxed add on a
/// path that already does an FFI call, so it does not move any benchmark.
pub var op_count: std.atomic.Value(u64) = .init(0);

/// TEST-ONLY fault injection: `fault.arm(k)` makes the k-th SUBSEQUENT op (a call that carries a stream and reports
/// by its `c_int`) or eval fail at its entry, without running, so an ownership audit can walk every error path of a
/// function without patching it. The result slot is left as a real failure leaves it: unassigned. Compiled only under
/// `builtin.is_test`, so a release build carries neither the branch nor the symbols.
/// TEST-ONLY: a fault's recording, for a sweep by site. While `list` is set, every call the fault counts appends its site
/// in call order and fails none; between a `disarm` and the next `arm` it notes nothing, as an armed fault counts nothing. A site is the stack of the call, hashed over `frames` frames: from the call itself
/// (`.leaf`), so each op under each caller is a site, or from the sweep's root down (`.root`), so a site is where in the
/// swept code the call sits, `frames` calls deep, and the frames below are the units' own tests' to sweep.
pub const FaultSites = struct {
    list: ?*std.ArrayList(u64) = null,
    gpa: std.mem.Allocator = undefined,
    frames: usize = 4,
    from: enum { leaf, root } = .leaf,
    /// Each site's last call is armed too, not only its first: worth it where the run is short.
    lasts: bool = true,
    /// The return address of the sweep's root frame: a call's stack holds it just outside the swept code.
    root_ret: usize = 0,
    failed: bool = false,
    paused: bool = false,
    /// Armed by site (`armAt`): the call that fails is the target site's n-th, whatever order the run's calls come in.
    target: ?SitePoint = null,
    target_seen: u64 = 0,

    /// A site and which of its calls (from 1): a sweep's point where a run's call order is not the same every time.
    pub const SitePoint = struct {
        site: u64,
        n: u64,
        /// Every run reaches it: a point that does not fire is a run that made other calls than recorded.
        strict: bool = true,
    };

    pub fn armAt(self: *FaultSites, p: SitePoint) void {
        self.target = p;
        self.target_seen = 0;
    }
    /// What a fault does with a call: recording notes it; armed by site, the target's n-th call fails and any other
    /// passes; otherwise the fault's own count decides. One entry for both, so a call's site is the same stack either way.
    pub const Take = enum { count, noted, fail, pass };
    pub noinline fn take(self: *FaultSites) Take {
        if (self.list == null and self.target == null) return .count;
        if (self.list != null and self.paused) return .noted;
        // One walk, from one place in this frame, so recording and matching hash the same stack.
        const site = self.siteHere();
        if (self.list) |list| {
            list.append(self.gpa, site) catch {
                self.failed = true;
            };
            return .noted;
        }
        const t = self.target.?;
        if (site != t.site) return .pass;
        self.target_seen += 1;
        if (self.target_seen != t.n) return .pass;
        self.target = null;
        return .fail;
    }

    noinline fn siteHere(self: *const FaultSites) u64 {
        var addrs: [256]usize = undefined;
        const all = std.debug.captureCurrentStackTrace(.{}, &addrs).return_addresses;
        const st = switch (self.from) {
            .leaf => all[0..@min(self.frames, all.len)],
            .root => blk: {
                // The trampoline's own frame sits just inside the root's return address; the swept code inside that.
                const at = std.mem.indexOfScalar(usize, all, self.root_ret) orelse all.len;
                const below = at -| 1;
                break :blk all[below -| self.frames..below];
            },
        };
        return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(st));
    }
    /// A fault's `arm` while recording: calls count from here, as the armed fault's do.
    pub fn armed(self: *FaultSites) void {
        if (self.list) |l| l.clearRetainingCapacity();
        self.paused = false;
    }
    pub fn disarmed(self: *FaultSites) void {
        self.paused = true;
        self.target = null;
    }
};

/// A fault sweep's points: the call number (from 1) of the first and the last call of each distinct site in `sites`,
/// ascending. The first proves the site's unwinding; the last proves it again with all the run built before it held,
/// as a later prefill chunk or layer holds more than the first.
pub fn sweepPointsOf(gpa: std.mem.Allocator, sites: []const u64, lasts: bool) ![]u64 {
    var span: std.AutoArrayHashMapUnmanaged(u64, [2]u64) = .empty;
    defer span.deinit(gpa);
    for (sites, 1..) |site, k| {
        const g = try span.getOrPut(gpa, site);
        if (!g.found_existing) g.value_ptr.* = .{ k, k } else g.value_ptr[1] = k;
    }
    var points: std.ArrayList(u64) = .empty;
    errdefer points.deinit(gpa);
    for (span.values()) |ends| {
        try points.append(gpa, ends[0]);
        if (lasts and ends[1] != ends[0]) try points.append(gpa, ends[1]);
    }
    std.mem.sort(u64, points.items, {}, std.sort.asc(u64));
    return points.toOwnedSlice(gpa);
}

/// Records the calls `run(ctx)` makes through `rec` (from the leaf), none failed, twice, and returns each site's first
/// call and, with `rec.lasts`, its last, as site points for `armAt`. A site whose count the two runs agree on is
/// `strict`: its points must fire. One they do not (a speculative request's rounds follow its load's acceptance so far)
/// has its last at the fewer, and that point, or the first of a site one run never reached, may not fire.
pub fn sweepSites(gpa: std.mem.Allocator, rec: *FaultSites, ctx: anytype, comptime run: fn (@TypeOf(ctx)) anyerror!void) ![]FaultSites.SitePoint {
    std.debug.assert(rec.from == .leaf);
    var counts: [2]std.AutoArrayHashMapUnmanaged(u64, u64) = .{ .empty, .empty };
    defer for (&counts) |*c| c.deinit(gpa);
    for (&counts) |*count| {
        var list: std.ArrayList(u64) = .empty;
        defer list.deinit(gpa);
        rec.* = .{ .list = &list, .gpa = gpa, .frames = rec.frames, .lasts = rec.lasts };
        const r = run(ctx);
        const failed = rec.failed;
        rec.* = .{ .frames = rec.frames, .lasts = rec.lasts };
        try r;
        if (failed) return error.OutOfMemory;
        for (list.items) |site| {
            const g = try count.getOrPut(gpa, site);
            g.value_ptr.* = if (g.found_existing) g.value_ptr.* + 1 else 1;
        }
    }
    var points: std.ArrayList(FaultSites.SitePoint) = .empty;
    errdefer points.deinit(gpa);
    for (counts, 0..) |count, i| {
        const other = counts[1 - i];
        var it = count.iterator();
        while (it.next()) |e| {
            const site = e.key_ptr.*;
            const theirs = other.get(site);
            if (i == 1 and theirs != null) continue; // Both runs reached it: the first pass took it.
            const n = @min(e.value_ptr.*, theirs orelse e.value_ptr.*);
            const strict = theirs == e.value_ptr.*;
            try points.append(gpa, .{ .site = site, .n = 1, .strict = theirs != null });
            if (rec.lasts and n > 1) try points.append(gpa, .{ .site = site, .n = n, .strict = strict });
        }
    }
    return points.toOwnedSlice(gpa);
}

/// Records the sites of the calls `run(ctx)` makes through `rec` (`mlx.fault.sites`, or the engine pool's
/// `fault.sites`), none failed, and returns their `sweepPointsOf`. `run` must make the same calls in the same order each
/// time; the sweep proves it, as every armed point must fire.
pub fn sweepPoints(gpa: std.mem.Allocator, rec: *FaultSites, ctx: anytype, comptime run: fn (@TypeOf(ctx)) anyerror!void) ![]u64 {
    var list: std.ArrayList(u64) = .empty;
    defer list.deinit(gpa);
    const Root = struct {
        // A frame of its own whose return address marks, in every call's stack, where the swept code begins.
        noinline fn go(r: *FaultSites, list_: *std.ArrayList(u64), gpa_: std.mem.Allocator, c: @TypeOf(ctx)) anyerror!void {
            r.* = .{ .list = list_, .gpa = gpa_, .frames = r.frames, .from = r.from, .lasts = r.lasts, .root_ret = @returnAddress() };
            // Work after the call keeps it from being a tail call, which would take this frame off the stack.
            defer std.mem.doNotOptimizeAway(r.root_ret);
            return @call(.never_inline, run, .{c});
        }
    };
    const r = Root.go(rec, &list, gpa, ctx);
    const failed = rec.failed;
    rec.* = .{ .frames = rec.frames, .from = rec.from, .lasts = rec.lasts };
    try r;
    if (failed) return error.OutOfMemory;
    return sweepPointsOf(gpa, list.items, rec.lasts);
}

test "a fault sweep arms each site's first call, and its last unless told not to, once each, in call order" {
    const a = std.testing.allocator;
    const points = try sweepPointsOf(a, &.{ 7, 9, 7, 7, 4, 9, 5 }, true);
    defer a.free(points);
    // 7 at 1..4, 9 at 2..6, 4 once at 5, 5 once at 7.
    try std.testing.expectEqualSlices(u64, &.{ 1, 2, 4, 5, 6, 7 }, points);
    const firsts = try sweepPointsOf(a, &.{ 7, 9, 7, 7, 4, 9, 5 }, false);
    defer a.free(firsts);
    try std.testing.expectEqualSlices(u64, &.{ 1, 2, 5, 7 }, firsts);
    const none = try sweepPointsOf(a, &.{}, true);
    defer a.free(none);
    try std.testing.expectEqual(@as(usize, 0), none.len);
}

test "a fault armed at a site fails that site's n-th call, whatever calls come before it" {
    const a = std.testing.allocator;
    const Run = struct {
        var run_calls: u64 = 0;
        var failed_at: u64 = 0;
        var extra: bool = false;
        var vary: bool = false;
        var runs: usize = 0;
        fn at(r: *FaultSites) void {
            run_calls += 1;
            if (r.take() == .fail) failed_at = run_calls;
        }
        noinline fn viaOne(r: *FaultSites) void {
            at(r);
            std.mem.doNotOptimizeAway(r.frames);
        }
        noinline fn viaTwo(r: *FaultSites) void {
            // A body of its own, so the linker does not fold it into viaOne.
            at(r);
            std.mem.doNotOptimizeAway(r.frames *% 3);
        }
        fn run(r: *FaultSites) anyerror!void {
            run_calls = 0;
            // A run may make a call another did not: the site's n-th call is still the one armed.
            if (extra) viaTwo(r);
            // With `vary`, each run takes one more step than the last, as a speculative request's rounds may.
            var n: usize = 3 + (if (vary) runs else 0);
            runs += 1;
            std.mem.doNotOptimizeAway(&n);
            var i: usize = 0;
            while (i < n) : (i += 1) step(r, i);
        }
        noinline fn step(r: *FaultSites, i: usize) void {
            viaOne(r);
            if (i == 1) viaTwo(r);
            std.mem.doNotOptimizeAway(i);
        }
    };
    var rec: FaultSites = .{};
    const points = try sweepSites(a, &rec, &rec, Run.run);
    defer a.free(points);
    // viaOne at calls 1, 2 and 4 (first and last), viaTwo at 3 (once); both runs agree, so every point is strict.
    try std.testing.expectEqual(@as(usize, 3), points.len);
    for (points) |p| try std.testing.expect(p.strict);
    {
        // Runs that disagree on a site's count: its last is the fewer, and need not fire.
        Run.vary = true;
        Run.runs = 0;
        defer Run.vary = false;
        const varied = try sweepSites(a, &rec, &rec, Run.run);
        defer a.free(varied);
        try std.testing.expectEqual(@as(usize, 3), varied.len);
        try std.testing.expectEqual(FaultSites.SitePoint{ .site = varied[0].site, .n = 1, .strict = true }, varied[0]);
        try std.testing.expectEqual(FaultSites.SitePoint{ .site = varied[0].site, .n = 3, .strict = false }, varied[1]);
        try std.testing.expect(varied[2].strict and varied[2].n == 1);
    }
    Run.extra = true;
    defer Run.extra = false;
    // With a viaTwo call first, from another caller: viaOne's first call is call 2, its third call 5, and the step's
    // viaTwo is call 4.
    const want = [_]u64{ 2, 5, 4 };
    for (points, want) |p, w| {
        Run.failed_at = 0;
        rec.armAt(p);
        try Run.run(&rec);
        try std.testing.expectEqual(w, Run.failed_at);
        try std.testing.expect(rec.target == null);
    }
}

test "a fault's recording fails no call and tells a call's site by its callers" {
    const a = std.testing.allocator;
    var rec: FaultSites = .{};
    const Run = struct {
        fn at(r: *FaultSites) void {
            _ = r.take();
        }
        // Each call is followed by work, so none is a tail call that takes its caller's frame away.
        noinline fn viaOne(r: *FaultSites) void {
            at(r);
            std.mem.doNotOptimizeAway(r.frames);
        }
        noinline fn viaTwo(r: *FaultSites) void {
            // A body of its own, so the linker does not fold it into viaOne.
            at(r);
            std.mem.doNotOptimizeAway(r.frames *% 3);
        }
        fn run(r: *FaultSites) anyerror!void {
            // A trip count the compiler cannot see, so the loop's one call stays one site.
            var n: usize = 3;
            std.mem.doNotOptimizeAway(&n);
            var i: usize = 0;
            while (i < n) : (i += 1) step(r, i);
        }
        noinline fn step(r: *FaultSites, i: usize) void {
            viaOne(r);
            if (i == 1) viaTwo(r);
            std.mem.doNotOptimizeAway(i);
        }
    };
    const points = try sweepPoints(a, &rec, &rec, Run.run);
    defer a.free(points);
    // viaOne's site at calls 1, 2 and 4, viaTwo's at 3.
    try std.testing.expectEqualSlices(u64, &.{ 1, 3, 4 }, points);
    // From the root: one frame down every call sits in the loop's step, one site; two frames down, viaOne's and viaTwo's.
    rec = .{ .from = .root, .frames = 1 };
    const one = try sweepPoints(a, &rec, &rec, Run.run);
    defer a.free(one);
    try std.testing.expectEqualSlices(u64, &.{ 1, 4 }, one);
    rec = .{ .from = .root, .frames = 2 };
    const two = try sweepPoints(a, &rec, &rec, Run.run);
    defer a.free(two);
    try std.testing.expectEqualSlices(u64, &.{ 1, 3, 4 }, two);
    try std.testing.expect(rec.list == null);
}

pub const fault = struct {
    var remaining: u64 = 0; // 0 = disarmed
    var fired: bool = false;
    /// Print the stack of the call the fault fails, so a sweep that finds a leak names the op.
    pub var trace: bool = false;
    /// Recording for a sweep by site (`sweepPoints`).
    pub var sites: FaultSites = .{};

    /// Fail the k-th checked call from here (k >= 1). `arm(0)` is a no-op arm.
    pub fn arm(k: u64) void {
        remaining = k;
        fired = false;
        sites.armed();
    }
    pub fn disarm() void {
        remaining = 0;
        sites.disarmed();
    }
    pub fn armed() bool {
        return remaining != 0;
    }
    /// Did the armed fault actually fire? False means the call under test
    /// returned before reaching its k-th checked op — an assertion the test
    /// needs, or the sweep silently stops covering anything.
    pub fn didFire() bool {
        return fired;
    }
    fn hit() bool {
        switch (sites.take()) {
            .noted, .pass => return false,
            .fail => {
                fired = true;
                if (trace) std.debug.dumpCurrentStackTrace(.{});
                return true;
            },
            .count => {},
        }
        if (remaining == 0) return false;
        remaining -= 1;
        if (remaining != 0) return false;
        fired = true;
        if (trace) std.debug.dumpCurrentStackTrace(.{});
        return true;
    }
};

/// Test-only: like `fault`, but the injected failure also LATCHES the process-wide
/// error — the state a real mlx-c raise leaves behind, which a best-effort caller
/// must drop before returning.
const LATCH_FAULT_MSG = "injected latching mlx-c raise (test only)";
const LATCH_MEMORY_FAULT_MSG = "[METAL] Command buffer execution failed: Insufficient Memory (injected latching raise, test only)";
var latch_remaining: u64 = 0;
var latch_msg: [*:0]const u8 = LATCH_FAULT_MSG.ptr;
var latch_fired = false;

fn latchFaultHit() bool {
    if (latch_remaining == 0) return false;
    latch_remaining -= 1;
    if (latch_remaining != 0) return false;
    latch_fired = true;
    return true;
}

pub fn armLatchingFaultForTest(k: u64) void {
    latch_remaining = k;
    latch_fired = false;
    latch_msg = LATCH_FAULT_MSG.ptr;
}
/// `armLatchingFaultForTest` whose raise is a Metal out-of-memory.
pub fn armLatchingMemoryFaultForTest(k: u64) void {
    armLatchingFaultForTest(k);
    latch_msg = LATCH_MEMORY_FAULT_MSG.ptr;
}
pub fn latchingFaultFiredForTest() bool {
    return latch_fired;
}

pub fn check(ret: c_int) !void {
    _ = op_count.fetchAdd(1, .monotonic);
    if (ret != 0) return error.MlxError;
    if (comptime builtin.is_test) {
        if (latchFaultHit()) {
            latchMlxError(latch_msg, null);
            return error.MlxError;
        }
    }
}

// ---------------------------------------------------------------------------
// Wired-residency policy (mlxfast notes/47 class).
//
// MLX's Metal residency set has a capacity set by `mlx_set_wired_limit`. At
// capacity 0 (MLX default) nothing is wired, and the driver re-establishes
// residency for the whole RAM-resident model on every command buffer; the
// policy wires up to `max_recommended_working_set_size`.

/// What the default device says of itself. It does not change, so MLX is asked once a process (a failed query is
/// asked again).
const DeviceFacts = struct {
    working_set: usize = 0,
};
var device_facts: DeviceFacts = .{};
var device_known: std.atomic.Value(bool) = .init(false);

/// A test's process as one that has not asked the device yet.
pub fn forgetDeviceForTest() void {
    no_gpu_backend_cache = null;
    device_known.store(false, .release);
}

fn deviceFacts() ?*const DeviceFacts {
    if (device_known.load(.acquire)) return &device_facts;
    var dev = mlx_device{ .ctx = null };
    if (mlx_get_default_device(&dev) != 0) return null;
    defer _ = mlx_device_free(dev);
    var info = mlx_device_info_new();
    defer _ = mlx_device_info_free(info);
    if (mlx_device_info_get(&info, dev) != 0) return null;
    var facts: DeviceFacts = .{};
    if (mlx_device_info_get_size(&facts.working_set, info, "max_recommended_working_set_size") != 0) return null;
    device_facts = facts;
    device_known.store(true, .release);
    return &device_facts;
}

pub fn maxRecommendedWorkingSet() usize {
    return (deviceFacts() orelse return 0).working_set;
}

/// Apply the wired-residency policy. Call on the inference thread AFTER a
/// model load or unload completes (and from the offline run path after load).
/// Returns the limit set, or null when there is no GPU or the device query
/// failed. The caller logs the result.
pub fn applyWiredPolicy() ?usize {
    if (noGpuBackend()) return null;
    // Under a held scope MLX's limit is not set now: it is owed to MLX's next turn (`payOwed`), where a model on MLX
    // beside the engine's gets the limit less what the engine wraps. No limit is reported as set.
    if (atRest() != null) {
        owed.wired = true;
        return null;
    }
    const max_rec = maxRecommendedWorkingSet();
    if (max_rec == 0) return null;
    // A mapping the engine wraps is wired by the engine's sets: MLX's limit gives those bytes up, so the two together
    // stay under the working set.
    const limit = max_rec -| @as(usize, @intCast(engine.mappedTotal()));
    var prev: usize = 0;
    _ = mlx_set_wired_limit(&prev, limit);
    return limit;
}

// mlx-c error latch. mlx-c's default error handler calls exit(-1) after the C boundary
// already caught the exception, so a Metal working-set OOM killed the server (#353).
// Our handler latches the message and returns; the engine raises a named error at its
// next checkpoint and the process stays serviceable.
var mlx_error_latched = std.atomic.Value(bool).init(false);
var mlx_error_buf: [512]u8 = undefined;
var mlx_error_len: usize = 0;
/// pthread mutex: mlx-c calls the handler from whatever thread raised, none carry an `Io`.
var mlx_error_mtx: std.c.pthread_mutex_t = .{};
fn lockErrBuf() void {
    _ = std.c.pthread_mutex_lock(&mlx_error_mtx);
}
fn unlockErrBuf() void {
    _ = std.c.pthread_mutex_unlock(&mlx_error_mtx);
}

/// Opens the message of an allocation the transient cap refused.
pub const transient_cap_tag = "[mlx-transient-cap]";

/// In the message of an allocation the engine's pool refused under its cap (`engine.fail`).
pub const pool_cap_tag = "PoolOverBudget";

/// A refusal under a forward's cap, MLX's transient cap or the engine's pool cap: the bill did not cover the forward.
pub fn mlxErrorIsTransientCap(msg: []const u8) bool {
    return std.mem.startsWith(u8, msg, transient_cap_tag) or std.mem.indexOf(u8, msg, pool_cap_tag) != null;
}

/// Cap on the bytes MLX allocates itself, imported buffers left out; 0 turns it off. Returns the previous cap.
pub fn setTransientLimit(limit: usize) usize {
    var prev: usize = 0;
    _ = mlx_set_transient_limit(&prev, limit);
    return prev;
}

/// Active bytes that are imported (no-copy) buffers.
pub fn externalMemory() usize {
    var n: usize = 0;
    _ = mlx_get_external_memory(&n);
    return n;
}

/// High-water of MLX's own active bytes since `resetTransientPeak`.
pub fn transientPeak() usize {
    var n: usize = 0;
    _ = mlx_get_transient_peak(&n);
    return n;
}

pub fn resetTransientPeak() void {
    _ = mlx_reset_transient_peak();
}

/// Memory-class latched messages become a memory 503 (`[METAL] ... Insufficient Memory`,
/// `[malloc]`/`[metal::malloc]` failures, `Resource limit exceeded`).
pub fn mlxErrorIsMemory(msg: []const u8) bool {
    const needles = [_][]const u8{
        "Insufficient Memory",
        "insufficient memory",
        "Unable to allocate",
        "Resource limit",
        "out of memory",
        "Out of memory",
        "maximum allowed buffer size",
    };
    for (needles) |n| {
        if (std.mem.indexOf(u8, msg, n) != null) return true;
    }
    return false;
}

/// MLX raises of the memory class, counted whether or not they won the latch.
var memory_raises = std.atomic.Value(u64).init(0);

/// Refusals for memory so far, MLX's raises and the engine's pool both: a step that saw this move failed for room,
/// whatever error it returned and whoever settled the latch since.
pub fn memoryRefusals() u64 {
    return memory_raises.load(.monotonic) + engine.poolRefusals();
}

fn latchMlxError(msg: [*:0]const u8, data: ?*anyopaque) callconv(.c) void {
    _ = data;
    const span = std.mem.span(msg);
    if (mlxErrorIsTransientCap(span) or mlxErrorIsMemory(span)) _ = memory_raises.fetchAdd(1, .monotonic);
    lockErrBuf();
    defer unlockErrBuf();
    // First error wins: follow-on noise from a poisoned stream must not hide the cause.
    if (!mlx_error_latched.load(.acquire)) {
        const n = @min(span.len, mlx_error_buf.len);
        @memcpy(mlx_error_buf[0..n], span[0..n]);
        mlx_error_len = n;
        mlx_error_latched.store(true, .release);
    }
    log.atLevel(errorLogLevel(span), "[mlx] {s}\n", .{span});
}

/// Text of the refusal a probe on this thread is about to provoke and will settle itself (it names its step in a
/// line of its own): MLX's raise of it is logged at debug, not as an error.
pub threadlocal var expected_refusal: ?[]const u8 = null;

fn errorLogLevel(span: []const u8) log.Level {
    const want = expected_refusal orelse return .err;
    return if (std.mem.indexOf(u8, span, want) != null) .debug else .err;
}

/// Install the latching handler (mlx-c's default exits the process). Called once from `main()`.
pub fn installErrorHandler() void {
    mlx_set_error_handler(latchMlxError, null, null);
}

/// True when an MLX call has failed since the last `takeError`.
pub fn errorPending() bool {
    return mlx_error_latched.load(.acquire);
}

/// Peek the latch without consuming it: the name of the error `checkError` would raise, or
/// null. A slot finish must not answer 200 over a failed forward, yet must leave the latch
/// for the decode-tick wrapper that fails the rest of a batched group.
pub fn peekErrorName() ?[]const u8 {
    if (!mlx_error_latched.load(.acquire)) return null;
    lockErrBuf();
    defer unlockErrBuf();
    const msg = mlx_error_buf[0..mlx_error_len];
    if (mlxErrorIsTransientCap(msg)) return "TransientCapExceeded";
    return if (mlxErrorIsMemory(msg)) "OutOfMemory" else "MlxFailure";
}

/// Consume the latched message into `buf` (copied), clearing the latch.
pub fn takeError(buf: []u8) ?[]const u8 {
    if (!mlx_error_latched.load(.acquire)) return null;
    lockErrBuf();
    defer unlockErrBuf();
    const n = @min(mlx_error_len, buf.len);
    @memcpy(buf[0..n], mlx_error_buf[0..n]);
    mlx_error_len = 0;
    mlx_error_latched.store(false, .release);
    return buf[0..n];
}

/// Whether an error is latched, left for its reader.
pub fn errorLatched() bool {
    return mlx_error_latched.load(.acquire);
}

/// The latched message copied into `buf` without clearing it, for a log line that names the cause of a failure.
pub fn peekErrorMessage(buf: []u8) ?[]const u8 {
    if (!mlx_error_latched.load(.acquire)) return null;
    lockErrBuf();
    defer unlockErrBuf();
    const n = @min(mlx_error_len, buf.len);
    @memcpy(buf[0..n], mlx_error_buf[0..n]);
    return buf[0..n];
}

/// The limit MLX names when it refuses a custom-kernel launch that asks for more threads per threadgroup than
/// the compiled pipeline allows; it varies with the kernel's register and threadgroup-memory use.
pub fn threadgroupLimit(msg: []const u8) ?u32 {
    const marker = "maximum allowed threads per threadgroup (";
    const at = std.mem.indexOf(u8, msg, marker) orelse return null;
    const digits = msg[at + marker.len ..];
    const end = std.mem.indexOfScalar(u8, digits, ')') orelse return null;
    return std.fmt.parseInt(u32, digits[0..end], 10) catch null;
}

/// Consume the latch and return its limit when it is a threadgroup-size refusal; any other error stays latched.
pub fn takeThreadgroupLimit() ?u32 {
    if (!mlx_error_latched.load(.acquire)) return null;
    lockErrBuf();
    defer unlockErrBuf();
    const limit = threadgroupLimit(mlx_error_buf[0..mlx_error_len]) orelse return null;
    mlx_error_len = 0;
    mlx_error_latched.store(false, .release);
    return limit;
}

/// Consume the latch only when its message contains `needle`; any other error stays latched.
pub fn takeErrorIf(needle: []const u8) bool {
    if (!mlx_error_latched.load(.acquire)) return false;
    lockErrBuf();
    defer unlockErrBuf();
    if (std.mem.indexOf(u8, mlx_error_buf[0..mlx_error_len], needle) == null) return false;
    mlx_error_len = 0;
    mlx_error_latched.store(false, .release);
    return true;
}

/// Drop a latch a best-effort op raised and its caller already reported, so an
/// optional write or a diagnostic can never become an unrelated request's
/// `MlxFailure`. `had_error` is the caller's `errorPending()` from BEFORE the op:
/// an error that was already latched belongs to someone else and stays.
pub fn dropLatchedErrorUnless(had_error: bool) void {
    if (!had_error and errorPending()) {
        var buf: [512]u8 = undefined;
        _ = takeError(&buf);
    }
}

/// Release-build fault injection: `SUSHI_MLX_FAULT_CHUNK=<n>` latches a synthetic Metal
/// OOM at the n-th `checkError` of the process, then disarms (`tests/test_mlx_error_recovery.sh`).
const FAULT_CHUNK_MSG = "[METAL] Command buffer execution failed: Insufficient Memory (injected by SUSHI_MLX_FAULT_CHUNK). at transforms.cpp:15";
const FAULT_STEP_MSG = "[METAL] Command buffer execution failed: Insufficient Memory (injected by SUSHI_MLX_FAULT_STEP). at transforms.cpp:15";

/// One armed, self-disarming injector; prefill and decode checkpoints count separately.
const FaultSite = struct {
    env: [:0]const u8,
    msg: [:0]const u8,
    at: ?u64 = null,
    seen: u64 = 0,

    fn target(self: *FaultSite) u64 {
        if (self.at) |v| return v;
        var v: u64 = 0;
        if (std.c.getenv(self.env.ptr)) |pz| {
            v = std.fmt.parseInt(u64, std.mem.span(pz), 10) catch 0;
        }
        self.at = v;
        if (v > 0) log.warn("[mlx] FAULT INJECTION armed: {s}=#{d}\n", .{ self.env, v });
        return v;
    }

    fn maybeFire(self: *FaultSite) void {
        const t = self.target();
        if (t == 0) return;
        self.seen += 1;
        if (self.seen != t) return;
        self.at = 0;
        latchErrorForTest(self.msg);
    }

    fn reset(self: *FaultSite) void {
        self.at = null;
        self.seen = 0;
    }
};

var fault_chunk = FaultSite{ .env = "SUSHI_MLX_FAULT_CHUNK", .msg = FAULT_CHUNK_MSG };
var fault_step = FaultSite{ .env = "SUSHI_MLX_FAULT_STEP", .msg = FAULT_STEP_MSG };

/// Consume a latched failure: `error.TransientCapExceeded` for a refused allocation, `error.OutOfMemory`
/// for the memory class, else `error.MlxFailure`.
fn consumeLatch() !void {
    if (!mlx_error_latched.load(.acquire)) return;
    var buf: [512]u8 = undefined;
    const msg = takeError(&buf) orelse return;
    if (mlxErrorIsTransientCap(msg)) return error.TransientCapExceeded;
    if (mlxErrorIsMemory(msg)) return error.OutOfMemory;
    return error.MlxFailure;
}

/// The error a caught failure is reported as, its latch settled whatever the error: on the inference thread a latch
/// pending at a failure is that failure's cause (`check`'s `MlxError`, an engine maker's `HostArrayRefused`), and left
/// set it fails the next request's first checkpoint. A refusal or a Metal out-of-memory behind it is that failure.
pub fn settledError(err: anyerror) anyerror {
    var buf: [512]u8 = undefined;
    const msg = takeError(&buf);
    // The engine's pool refused the buffer: the same refusal its latched message names.
    if (err == error.PoolOverBudget) return error.TransientCapExceeded;
    const m = msg orelse return err;
    if (mlxErrorIsTransientCap(m)) return error.TransientCapExceeded;
    return if (mlxErrorIsMemory(m)) error.OutOfMemory else err;
}

pub fn settledErrorName(err: anyerror) []const u8 {
    return @errorName(settledError(err));
}

/// The end of a unit of a request's work that returned no error (a prefill, a decode tick): a latch a step of it left
/// fails the unit here, not the next request.
pub fn checkLatch() !void {
    return consumeLatch();
}

/// The end of a step that handles its own failure (a commit or an SSD flush skipped): takes a latch it left, logged
/// under `what`. Called only where no latch was pending as the step began, so the latch is the step's own.
pub fn settleHandled(what: []const u8) void {
    var buf: [512]u8 = undefined;
    const msg = takeError(&buf) orelse return;
    log.warn("{s} left an MLX error, settled: {s}\n", .{ what, msg });
}

/// Prefill checkpoint: once per chunk plus once after the last.
pub fn checkError() !void {
    fault_chunk.maybeFire();
    return consumeLatch();
}

/// Decode checkpoint: once per decode tick, after the forward. Without it a failed decode
/// forward finished 200 with garbage and handed the latch to the NEXT request's prefill.
pub fn checkErrorDecode() !void {
    fault_step.maybeFire();
    return consumeLatch();
}

/// Latches an error of the engine's own as an mlx-c error is latched.
pub fn latchError(msg: [:0]const u8) void {
    latchMlxError(msg.ptr, null);
}

pub fn latchErrorForTest(msg: [:0]const u8) void {
    latchMlxError(msg.ptr, null);
}

pub fn resetFaultChunkForTest() void {
    fault_chunk.reset();
    fault_step.reset();
    var buf: [512]u8 = undefined;
    _ = takeError(&buf);
}

test "a refusal the thread's probe expects logs at debug, any other error and an unexpected one as errors" {
    const t = std.testing;
    const refusal = "[engine] Thread group size (1024) is greater than the maximum allowed threads per threadgroup (832).";
    try t.expectEqual(log.Level.err, errorLogLevel(refusal));
    expected_refusal = "Thread group size";
    defer expected_refusal = null;
    try t.expectEqual(log.Level.debug, errorLogLevel(refusal));
    try t.expectEqual(log.Level.err, errorLogLevel("[matmul] Last dimension of first input must match."));
}

test "a threadgroup-size refusal names the pipeline limit and only it is consumed" {
    const t = std.testing;
    const refusal = "Thread group size (1024) is greater than  the maximum allowed threads per threadgroup (832).";
    try t.expectEqual(@as(?u32, 832), threadgroupLimit(refusal));
    try t.expectEqual(@as(?u32, null), threadgroupLimit("[slice_update] Invalid slice sizes."));
    try t.expectEqual(@as(?u32, null), threadgroupLimit("maximum allowed threads per threadgroup (x)"));

    // A refusal an earlier test left latched is that test's failure, not this one's.
    var buf: [512]u8 = undefined;
    _ = takeError(&buf);
    try t.expectEqual(@as(?u32, null), takeThreadgroupLimit());
    latchErrorForTest("[matmul] Last dimension of first input must match.");
    try t.expectEqual(@as(?u32, null), takeThreadgroupLimit());
    try t.expect(errorPending());
    _ = takeError(&buf);
    latchErrorForTest(refusal);
    try t.expectEqual(@as(?u32, 832), takeThreadgroupLimit());
    try t.expect(!errorPending());
}

test "an MLX memory failure is classified as a memory error, an argument error is not" {
    const t = std.testing;
    try t.expect(mlxErrorIsMemory("[METAL] Command buffer execution failed: Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory). at transforms.cpp:15"));
    try t.expect(mlxErrorIsMemory("[malloc] Unable to allocate 8589934592 bytes."));
    try t.expect(mlxErrorIsMemory("[metal::malloc] Resource limit (128000) exceeded."));
    try t.expect(mlxErrorIsMemory("[metal::malloc] Attempting to allocate 99 bytes which is greater than the maximum allowed buffer size of 5 bytes."));
    try t.expect(!mlxErrorIsMemory("[slice_update] Invalid slice sizes."));
    try t.expect(!mlxErrorIsMemory("[matmul] Last dimension of first input must match."));
}

test "the latch turns an mlx-c error into a Zig error instead of exiting, once" {
    const t = std.testing;
    try t.expect(!errorPending());
    try checkError();

    latchErrorForTest("[METAL] Command buffer execution failed: Insufficient Memory. at transforms.cpp:15");
    try t.expect(errorPending());
    latchErrorForTest("[slice_update] Invalid slice sizes.");
    try t.expectError(error.OutOfMemory, checkError());
    try t.expect(!errorPending());
    try checkError();

    latchErrorForTest("[matmul] Last dimension of first input must match.");
    try t.expectError(error.MlxFailure, checkError());
    try t.expect(!errorPending());
}

test "the injected MLX error costs ONE checkError and the engine keeps working after it" {
    const t = std.testing;
    resetFaultChunkForTest();
    defer resetFaultChunkForTest();

    try checkError();
    try checkError();

    latchErrorForTest("[METAL] Command buffer execution failed: Insufficient Memory. at transforms.cpp:15");
    try t.expectError(error.OutOfMemory, checkError());

    try t.expect(!errorPending());
    try checkError();
    const s = gpuStream();
    const shape = [_]c_int{2};
    const data = [_]f32{ 1.5, 2.5 };
    const a = mlx_array_new_data(@ptrCast(&data), &shape, 1, .float32);
    defer _ = mlx_array_free(a);
    var sum = mlx_array_new();
    defer _ = mlx_array_free(sum);
    try t.expectEqual(@as(c_int, 0), mlx_sum(&sum, a, false, s));
    _ = mlx_array_eval(sum);
    var out: f32 = 0;
    _ = mlx_array_item_float32(&out, sum);
    try t.expectApproxEqAbs(@as(f32, 4.0), out, 1e-6);
    try checkError();
}

test "peekErrorName names the class WITHOUT consuming the latch" {
    const t = std.testing;
    resetFaultChunkForTest();
    defer resetFaultChunkForTest();

    try t.expect(peekErrorName() == null);

    latchErrorForTest("[METAL] Command buffer execution failed: Insufficient Memory. at transforms.cpp:15");
    try t.expectEqualStrings("OutOfMemory", peekErrorName().?);
    try t.expectEqualStrings("OutOfMemory", peekErrorName().?);
    try t.expect(errorPending());
    try t.expectError(error.OutOfMemory, checkErrorDecode());
    try t.expect(peekErrorName() == null);

    latchErrorForTest("[matmul] Last dimension of first input must match.");
    try t.expectEqualStrings("MlxFailure", peekErrorName().?);
    try t.expectError(error.MlxFailure, checkErrorDecode());
    try t.expect(peekErrorName() == null);
}

test "a DECODE-time MLX failure is attributed to the decoding request, not the next one" {
    const t = std.testing;
    resetFaultChunkForTest();
    defer resetFaultChunkForTest();

    try checkError();
    try checkErrorDecode();
    latchErrorForTest("[METAL] Command buffer execution failed: Insufficient Memory. at transforms.cpp:15");
    try t.expectError(error.OutOfMemory, checkErrorDecode());
    try t.expect(!errorPending());
    try checkError();

    try checkError();
    try checkError();
    try checkErrorDecode();
    try t.expect(!errorPending());
}


test "a failure caught as MlxError settles the latch, so the next request's first checkpoint is clean" {
    const t = std.testing;
    resetFaultChunkForTest();
    defer resetFaultChunkForTest();
    installErrorHandler();

    // The real route: an mlx-c call raises, the handler latches, `check` returns MlxError.
    const s = gpuStream();
    defer _ = mlx_stream_free(s);
    const shape = [_]c_int{4};
    const data = [_]f32{ 1, 2, 3, 4 };
    const a = mlx_array_new_data(@ptrCast(&data), &shape, 1, .float32);
    defer _ = mlx_array_free(a);
    const bad = [_]c_int{ 3, 3 };
    var out = mlx_array_new();
    defer _ = mlx_array_free(out);
    try t.expectError(error.MlxError, check(mlx_reshape(&out, a, &bad, 2, s)));
    try t.expect(errorPending());

    try t.expectEqualStrings("MlxError", settledErrorName(error.MlxError));
    try t.expect(!errorPending());
    try checkError();
}

test "a Metal out-of-memory behind an MlxError is a memory failure and leaves no latch" {
    const t = std.testing;
    resetFaultChunkForTest();
    defer resetFaultChunkForTest();

    latchErrorForTest("[METAL] Command buffer execution failed: Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory). at transforms.cpp:73");
    try t.expectEqualStrings("OutOfMemory", settledErrorName(error.MlxError));
    try t.expect(!errorPending());
    try checkError();
    try checkErrorDecode();
}

test "settledErrorName takes the latch whatever the error, and names a refusal behind an engine maker's error by it" {
    const t = std.testing;
    resetFaultChunkForTest();
    defer resetFaultChunkForTest();

    try t.expectEqualStrings("MlxError", settledErrorName(error.MlxError));
    try t.expectEqualStrings("Cancelled", settledErrorName(error.Cancelled));
    latchErrorForTest("[matmul] Last dimension of first input must match.");
    try t.expectEqualStrings("Cancelled", settledErrorName(error.Cancelled));
    try t.expect(!errorPending());
    latchErrorForTest("[engine] array_new_data: PoolOverBudget: a buffer of 16 KB under a limit of 4 MB");
    try t.expectEqualStrings("TransientCapExceeded", settledErrorName(error.HostArrayRefused));
    try t.expect(!errorPending());
    try t.expectEqualStrings("TransientCapExceeded", settledErrorName(error.PoolOverBudget));
    try checkError();
}

test "a transient-cap refusal is its own failure, at every checkpoint, and is consumed once" {
    const t = std.testing;
    resetFaultChunkForTest();
    defer resetFaultChunkForTest();

    const refusal = transient_cap_tag ++ " Allocating 4096 bytes needs 8192 bytes of transient memory; the limit is 4096.";
    try t.expect(!mlxErrorIsMemory(refusal));
    try t.expect(mlxErrorIsTransientCap(refusal));
    try t.expect(mlxErrorIsTransientCap("[engine] kernel_apply: PoolOverBudget: a buffer of 16 KB with 10 MB held for the pool, 4 MB of it asked by tensors in use, under a limit of 4 MB"));
    try t.expect(!mlxErrorIsTransientCap("[METAL] Insufficient Memory"));

    latchErrorForTest(refusal);
    try t.expectEqualStrings("TransientCapExceeded", peekErrorName().?);
    try t.expectError(error.TransientCapExceeded, checkError());
    try t.expect(!errorPending());
    try checkError();

    latchErrorForTest(refusal);
    try t.expectError(error.TransientCapExceeded, checkErrorDecode());
    try t.expect(!errorPending());

    latchErrorForTest(refusal);
    try t.expectEqualStrings("TransientCapExceeded", settledErrorName(error.MlxError));
    try t.expect(!errorPending());
}

test "takeErrorIf leaves unrelated errors latched" {
    var buf: [512]u8 = undefined;
    _ = takeError(&buf);
    defer _ = takeError(&buf);
    latchErrorForTest("unrelated allocation failure");
    try std.testing.expect(!takeErrorIf("maximum allowed threads per threadgroup"));
    try std.testing.expectEqualStrings("unrelated allocation failure", takeError(&buf).?);
    latchErrorForTest("exceeds maximum allowed threads per threadgroup");
    try std.testing.expect(takeErrorIf("maximum allowed threads per threadgroup"));
    try std.testing.expect(!errorPending());
}

const cap_test_mib: usize = 1 << 20;
const cap_test_region_len: usize = 64 * 16384;

fn capTestUnmap(data: ?*anyopaque) callconv(.c) void {
    _ = std.c.munmap(@ptrCast(@alignCast(data.?)), cap_test_region_len);
}

fn capTestTransient() usize {
    var active: usize = 0;
    _ = mlx_get_active_memory(&active);
    return active - externalMemory();
}

fn capTestCache() usize {
    var cache: usize = 0;
    _ = mlx_get_cache_memory(&cache);
    return cache;
}

/// A lazy float32 array of `mib` MiB; its buffer is allocated at eval.
fn capTestOnes(mib: usize, s: mlx_stream) !mlx_array {
    const shape = [_]c_int{@intCast(mib * cap_test_mib / 4)};
    var out = mlx_array_new();
    errdefer _ = mlx_array_free(out);
    try check(mlx_ones(&out, &shape, 1, .float32, s));
    return out;
}

/// Frees an evaluated array and waits until its buffer is back in the pool.
fn capTestRelease(arr: mlx_array, s: mlx_stream) void {
    _ = mlx_array_free(arr);
    _ = mlx_synchronize(s);
}

/// Leaves exactly one `mib`-MiB buffer parked in the pool.
fn capTestPool(mib: usize, s: mlx_stream) !void {
    _ = mlx_clear_cache();
    const arr = try capTestOnes(mib, s);
    try check(mlx_array_eval(arr));
    capTestRelease(arr, s);
}

test "the transient cap: external bytes follow a no-copy import and nothing else" {
    if (noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    const before = externalMemory();

    const mapped = std.c.mmap(null, cap_test_region_len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    try t.expect(mapped != std.c.MAP_FAILED);
    const shape = [_]c_int{@intCast(cap_test_region_len)};
    const arr = mlx_array_new_data_managed(mapped, &shape, 1, .uint8, capTestUnmap);
    try t.expect(arr.ctx != null);
    try check(mlx_array_eval(arr));
    try t.expect(mlx_array_data_uint8(arr) == @as(?[*]const u8, @ptrCast(mapped)));
    try t.expectEqual(before + cap_test_region_len, externalMemory());

    const s = gpuStream();
    defer _ = mlx_stream_free(s);
    const own = try capTestOnes(2, s);
    try check(mlx_array_eval(own));
    try t.expectEqual(before + cap_test_region_len, externalMemory());
    capTestRelease(own, s);

    _ = mlx_array_free(arr);
    _ = mlx_synchronize(s);
    var tries: usize = 0;
    while (externalMemory() != before and tries < 100) : (tries += 1) {
        var ts = std.c.timespec{ .sec = 0, .nsec = 20 * std.time.ns_per_ms };
        _ = std.c.nanosleep(&ts, null);
    }
    try t.expectEqual(before, externalMemory());
}

test "the transient cap trims the pool first, then refuses with the tagged error" {
    if (noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    var buf: [512]u8 = undefined;
    _ = takeError(&buf);
    const s = gpuStream();
    defer _ = mlx_stream_free(s);
    defer _ = setTransientLimit(0);

    // Four pooled MiB, then a two-MiB request that cannot reuse them.
    try capTestPool(4, s);
    try t.expect(capTestCache() >= 4 * cap_test_mib);

    // Control: no cap, the request is served and the pool is left alone.
    try t.expectEqual(@as(usize, 0), setTransientLimit(0));
    const free_run = try capTestOnes(2, s);
    try check(mlx_array_eval(free_run));
    capTestRelease(free_run, s);
    try t.expect(capTestCache() >= 4 * cap_test_mib);

    // Room once the pool is trimmed: served, pool released.
    try capTestPool(4, s);
    const fits = try capTestOnes(2, s);
    _ = setTransientLimit(capTestTransient() + 3 * cap_test_mib);
    try check(mlx_array_eval(fits));
    try t.expect(capTestCache() < 4 * cap_test_mib);
    capTestRelease(fits, s);

    // No room even with an empty pool: refused, tagged, and a latched error.
    _ = mlx_clear_cache();
    const over = try capTestOnes(2, s);
    _ = setTransientLimit(capTestTransient() + cap_test_mib);
    try t.expectError(error.MlxError, check(mlx_array_eval(over)));
    try t.expect(errorPending());
    try t.expectEqualStrings("TransientCapExceeded", settledErrorName(error.MlxError));
    try t.expect(!errorPending());
    _ = mlx_array_free(over);
    _ = mlx_synchronize(s);
}

test "the transient peak is the high-water of MLX's own bytes and restarts on reset" {
    if (noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    const s = gpuStream();
    defer _ = mlx_stream_free(s);

    _ = mlx_clear_cache();
    resetTransientPeak();
    const base = capTestTransient();
    try t.expect(transientPeak() >= base);
    try t.expect(transientPeak() < base + cap_test_mib);

    const big = try capTestOnes(4, s);
    try check(mlx_array_eval(big));
    try t.expect(transientPeak() >= base + 4 * cap_test_mib);
    capTestRelease(big, s);
    try t.expect(transientPeak() >= base + 4 * cap_test_mib);

    resetTransientPeak();
    try t.expect(transientPeak() < base + 4 * cap_test_mib);
}

test "an op after a refused allocation is correct once the stream is drained and the latch cleared" {
    if (noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    var buf: [512]u8 = undefined;
    _ = takeError(&buf);
    const s = gpuStream();
    defer _ = mlx_stream_free(s);
    defer _ = setTransientLimit(0);

    const shape = [_]c_int{4};
    const data = [_]f32{ 1, 2, 3, 4 };
    const x = mlx_array_new_data(@ptrCast(&data), &shape, 1, .float32);
    defer _ = mlx_array_free(x);
    try check(mlx_array_eval(x));

    const big = try capTestOnes(2, s);
    defer _ = mlx_array_free(big);
    _ = mlx_clear_cache();
    _ = setTransientLimit(capTestTransient() + cap_test_mib);
    try t.expectError(error.MlxError, check(mlx_array_eval(big)));
    _ = mlx_synchronize(s);
    try t.expectEqualStrings("TransientCapExceeded", settledErrorName(error.MlxError));
    _ = setTransientLimit(0);

    var doubled = mlx_array_new();
    defer _ = mlx_array_free(doubled);
    try check(mlx_add(&doubled, x, x, s));
    try check(mlx_array_eval(doubled));
    const got = mlx_array_data_float32(doubled).?;
    try t.expectEqualSlices(f32, &.{ 2, 4, 6, 8 }, got[0..4]);

    // The array that failed evaluates once there is room.
    try check(mlx_array_eval(big));
    var sum = mlx_array_new();
    defer _ = mlx_array_free(sum);
    try check(mlx_sum(&sum, big, false, s));
    var total: f32 = 0;
    try check(mlx_array_item_float32(&total, sum));
    try t.expectEqual(@as(f32, 2 * cap_test_mib / 4), total);
    try t.expect(!errorPending());
}

test "requireGpu skips a test when MLX has no GPU backend" {
    const saved = no_gpu_backend_cache;
    defer no_gpu_backend_cache = saved;
    no_gpu_backend_cache = true;
    try std.testing.expectError(error.SkipZigTest, requireGpu());
    no_gpu_backend_cache = false;
    try requireGpu();
}

test "a minified kernel gives the bytes of the same kernel as written" {
    try requireGpu();
    const header =
        \\// a comment with an open /* and a // inside
        \\#define SCALE(v) \
        \\    ((v) * 1.5f + /* inline */ 0.25f)
        \\template <typename T, int N>
        \\struct Acc { T v; };
        \\static inline float blend(float a, float b) {
        \\    /* two blank lines follow */
        \\
        \\
        \\    return SCALE(a) - b * 0.5f;
        \\}
        \\#if 1
        \\#define K 3
        \\#else
        \\#define K 4
        \\#endif
        \\struct Unused { float v; };
        \\static inline float never_called(float v) { return Unused{v}.v * 7.0f; }
        \\
    ;
    const source =
        \\uint i = thread_position_in_grid.x;   // lane
        \\Acc<float, K> acc;
        \\acc.v = blend(x[i], 2.0f) + float(K);
        \\out[i] = acc.v - -x[i] * 0.5f;
    ;
    const t = std.testing;
    const s = gpuStream();
    defer _ = mlx_stream_free(s);
    const names = [_][*:0]const u8{"x"};
    const outs = [_][*:0]const u8{"out"};
    const iv = mlx_vector_string_new_data(&names, names.len);
    defer _ = mlx_vector_string_free(iv);
    const ov = mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx_vector_string_free(ov);
    const as_written = kernelNewSource("sushi_minify_probe_raw", iv, ov, source, header, true, false);
    defer _ = mlx_fast_metal_kernel_free(as_written);
    const minified = mlx_fast_metal_kernel_new("sushi_minify_probe_min", iv, ov, source, header, true, false);
    defer _ = mlx_fast_metal_kernel_free(minified);
    try t.expect(as_written.ctx != null and minified.ctx != null);

    var data: [256]f32 = undefined;
    for (&data, 0..) |*v, i| v.* = 0.37 * @as(f32, @floatFromInt(i)) - 11.0;
    const x = mlx_array_new_data(&data, &[_]c_int{256}, 1, .float32);
    defer _ = mlx_array_free(x);
    const inputs = mlx_vector_array_new_data(&[_]mlx_array{x}, 1);
    defer _ = mlx_vector_array_free(inputs);
    var results: [2][256]f32 = undefined;
    for ([_]mlx_fast_metal_kernel{ as_written, minified }, 0..) |k, which| {
        const cfg = mlx_fast_metal_kernel_config_new();
        defer _ = mlx_fast_metal_kernel_config_free(cfg);
        try check(mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{256}, 1, .float32));
        try check(mlx_fast_metal_kernel_config_set_grid(cfg, 256, 1, 1));
        try check(mlx_fast_metal_kernel_config_set_thread_group(cfg, 64, 1, 1));
        var outs_vec = mlx_vector_array_new();
        defer _ = mlx_vector_array_free(outs_vec);
        try check(mlx_fast_metal_kernel_apply(&outs_vec, k, inputs, cfg, s));
        var y = mlx_array_new();
        defer _ = mlx_array_free(y);
        try check(mlx_vector_array_get(&y, outs_vec, 0));
        try check(mlx_array_eval(y));
        @memcpy(&results[which], mlx_array_data_float32(y).?[0..256]);
    }
    try t.expectEqualSlices(u8, std.mem.sliceAsBytes(&results[0]), std.mem.sliceAsBytes(&results[1]));
    try t.expectApproxEqAbs(1.5 * data[5] + 0.25 - 1.0 + 3.0 + 0.5 * data[5], results[0][5], 1e-4);
}

test "one kernel runs each instantiation on its own source: template values, input dtype, how an input is passed" {
    const source =
        \\uint i = thread_position_in_grid.x;
        \\out[i] = float(x[i]) * float(SCALE) + float(w[i % uint(WN)]);
    ;
    const t = std.testing;
    const s = gpuStream();
    defer _ = mlx_stream_free(s);
    const names = [_][*:0]const u8{ "x", "w" };
    const outs = [_][*:0]const u8{"out"};
    const iv = mlx_vector_string_new_data(&names, names.len);
    defer _ = mlx_vector_string_free(iv);
    const ov = mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx_vector_string_free(ov);
    const kernel = mlx_fast_metal_kernel_new("sushi_instance_probe", iv, ov, source, "", true, false);
    defer _ = mlx_fast_metal_kernel_free(kernel);
    try t.expect(kernel.ctx != null);

    var xf: [64]f32 = undefined;
    var xi: [64]i32 = undefined;
    var w: [16]f32 = undefined;
    for (&xf, &xi, 0..) |*f, *n, i| {
        n.* = @as(i32, @intCast(i)) - 20;
        f.* = 0.25 * @as(f32, @floatFromInt(n.*));
    }
    for (&w, 0..) |*v, i| v.* = 100.0 * @as(f32, @floatFromInt(i + 1));

    // A repeat of an earlier case must still run that case's source, whatever ran in between.
    const Case = struct { scale: c_int, int_x: bool, wn: c_int };
    const cases = [_]Case{
        .{ .scale = 2, .int_x = false, .wn = 16 },
        .{ .scale = 3, .int_x = false, .wn = 16 },
        .{ .scale = 2, .int_x = false, .wn = 16 },
        .{ .scale = 2, .int_x = true, .wn = 16 },
        .{ .scale = 2, .int_x = false, .wn = 4 }, // under 8 elements: passed in the constant address space
        .{ .scale = 3, .int_x = false, .wn = 16 },
        .{ .scale = 2, .int_x = true, .wn = 4 },
        .{ .scale = 2, .int_x = true, .wn = 16 },
    };
    for (cases) |c| {
        const x = if (c.int_x) mlx_array_new_data(&xi, &[_]c_int{64}, 1, .int32) else mlx_array_new_data(&xf, &[_]c_int{64}, 1, .float32);
        defer _ = mlx_array_free(x);
        const wa = mlx_array_new_data(&w, &[_]c_int{c.wn}, 1, .float32);
        defer _ = mlx_array_free(wa);
        const inputs = mlx_vector_array_new_data(&[_]mlx_array{ x, wa }, 2);
        defer _ = mlx_vector_array_free(inputs);
        const cfg = mlx_fast_metal_kernel_config_new();
        defer _ = mlx_fast_metal_kernel_config_free(cfg);
        try check(mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{64}, 1, .float32));
        try check(mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "SCALE", c.scale));
        try check(mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "WN", c.wn));
        try check(mlx_fast_metal_kernel_config_set_grid(cfg, 64, 1, 1));
        try check(mlx_fast_metal_kernel_config_set_thread_group(cfg, 64, 1, 1));
        var outs_vec = mlx_vector_array_new();
        defer _ = mlx_vector_array_free(outs_vec);
        try check(mlx_fast_metal_kernel_apply(&outs_vec, kernel, inputs, cfg, s));
        var y = mlx_array_new();
        defer _ = mlx_array_free(y);
        try check(mlx_vector_array_get(&y, outs_vec, 0));
        try check(mlx_array_eval(y));
        const got = mlx_array_data_float32(y).?[0..64];
        for (got, 0..) |g, i| {
            const xv: f32 = if (c.int_x) @floatFromInt(xi[i]) else xf[i];
            try t.expectEqual(xv * @as(f32, @floatFromInt(c.scale)) + w[i % @as(usize, @intCast(c.wn))], g);
        }
    }
}

test "a decode tick's buffers commit on op count: an input past the size budget commits a buffer only outside it" {
    const t = std.testing;
    const s = gpuStream();
    defer _ = mlx_stream_free(s);
    // 64M one-byte items: past MLX's per-buffer size budget on every chip, so an op that reads it commits its buffer.
    var big = mlx_array_new();
    defer _ = mlx_array_free(big);
    try check(mlx_zeros(&big, &[_]c_int{64 << 20}, 1, .uint8, s));
    try check(mlx_array_eval(big));
    const picks = [_]i32{ 3, 1 << 20, 63 << 20 };
    const idx = mlx_array_new_data(&picks, &[_]c_int{picks.len}, 1, .int32);
    defer _ = mlx_array_free(idx);

    const ops = 12;
    const Run = struct {
        fn buffers(big_: mlx_array, idx_: mlx_array, s_: mlx_stream) !usize {
            var acc = mlx_array_new();
            defer _ = mlx_array_free(acc);
            try check(mlx_take(&acc, big_, idx_, s_));
            for (1..ops) |_| {
                var row = mlx_array_new();
                defer _ = mlx_array_free(row);
                try check(mlx_take(&row, big_, idx_, s_));
                var sum = mlx_array_new();
                try check(mlx_add(&sum, acc, row, s_));
                _ = mlx_array_free(acc);
                acc = sum;
            }
            var before: usize = 0;
            try check(mlx_metal_command_buffer_count(&before));
            try check(mlx_array_eval(acc));
            var after: usize = 0;
            try check(mlx_metal_command_buffer_count(&after));
            const got = mlx_array_data_uint8(acc).?[0..3];
            for (got) |v| if (v != 0) return error.TestUnexpectedResult;
            return after - before;
        }
    };
    try t.expect(try Run.buffers(big, idx, s) >= ops);
    {
        const tick = DecodeBuffers.begin();
        defer tick.end();
        try t.expect(try Run.buffers(big, idx, s) <= 2);
    }
    try t.expect(try Run.buffers(big, idx, s) >= ops);
}
