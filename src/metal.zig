//! Metal from Zig through the Objective-C runtime: the GPU MLX runs on, a command queue of our own, pipelines compiled
//! once and held, and buffers over memory someone else maps. Nothing here calls MLX.
//!
//! What a caller owns that MLX did for it: a buffer outlives every command buffer that binds it (wait before the
//! release), a dispatch that reads what an earlier one wrote relies on encode order (the encoder is serial), and a
//! failed command buffer is reported at its `wait`.

const std = @import("std");

pub const id = ?*anyopaque;
const SEL = ?*anyopaque;

extern fn sel_registerName(name: [*:0]const u8) SEL;
extern fn objc_getClass(name: [*:0]const u8) id;
extern fn objc_msgSend() void;
extern fn objc_autoreleasePoolPush() ?*anyopaque;
extern fn objc_autoreleasePoolPop(pool: ?*anyopaque) void;
extern fn MTLCreateSystemDefaultDevice() id;

/// MTLSize, also the grid and the threadgroup of a dispatch.
pub const Size = extern struct { x: usize, y: usize = 1, z: usize = 1 };

/// One selector per name, registered at its first use.
fn sel(comptime name: [:0]const u8) SEL {
    const S = struct {
        // Naming `name` makes this struct, and so its variable, one per selector.
        const of = name;
        var cached: std.atomic.Value(SEL) = .init(null);
    };
    if (S.cached.load(.monotonic)) |s| return s;
    const s = sel_registerName(S.of);
    S.cached.store(s, .monotonic);
    return s;
}

/// `[target name:args...]` with the argument types taken from `args` and the result type `R`.
fn send(comptime R: type, target: id, comptime name: [:0]const u8, args: anytype) R {
    const T = @typeInfo(@TypeOf(args)).@"struct".field_types;
    const F = switch (T.len) {
        0 => *const fn (id, SEL) callconv(.c) R,
        1 => *const fn (id, SEL, T[0]) callconv(.c) R,
        2 => *const fn (id, SEL, T[0], T[1]) callconv(.c) R,
        3 => *const fn (id, SEL, T[0], T[1], T[2]) callconv(.c) R,
        4 => *const fn (id, SEL, T[0], T[1], T[2], T[3]) callconv(.c) R,
        else => @compileError("metal.send: more arguments than a Metal call here takes"),
    };
    const f: F = @ptrCast(&objc_msgSend);
    return @call(.auto, f, .{ target, sel(name) } ++ args);
}

pub const Error = error{ MetalUnavailable, MetalCompile, MetalBuffer, MetalCommand };

var error_text: [2048]u8 = undefined;
var error_len: usize = 0;

/// What Metal said about the last `Error` on this thread's device calls; empty when it said nothing.
pub fn lastError() []const u8 {
    return error_text[0..error_len];
}

fn noteError(ns_error: id, what: []const u8) void {
    error_len = 0;
    const text: ?[*:0]const u8 = if (ns_error) |e| send(?[*:0]const u8, send(id, e, "localizedDescription", .{}), "UTF8String", .{}) else null;
    const said = if (text) |t| std.mem.span(t) else what;
    error_len = @min(said.len, error_text.len);
    @memcpy(error_text[0..error_len], said[0..error_len]);
}

/// A buffer over memory someone else owns takes MLX's options: shared storage, no hazard tracking. A pass of ours
/// reads such a buffer or writes it in the order its dispatches are encoded, and MLX's queue is idle meanwhile.
const alias_options: usize = 1 << 8;

var shared: ?Device = null;

/// The system GPU, which is the one MLX opens, and our queue on it. Opened once and kept for the process.
pub fn device() Error!*const Device {
    if (shared == null) shared = try Device.open();
    return &shared.?;
}

/// Buffers were added to or taken from the residency set since its last commit.
var resident_dirty = false;

/// A boolean function constant of a compiled kernel, by its index.
pub const Flag = struct { index: usize, value: bool };

pub const Device = struct {
    raw: id,
    queue: id,
    options: id,
    /// The residency set every buffer of ours is in, attached to the queue: a command buffer then finds its buffers
    /// resident instead of making each one so at commit, which for a buffer over mapped weights costs its size.
    resident: id,

    fn open() Error!Device {
        const raw = MTLCreateSystemDefaultDevice() orelse return error.MetalUnavailable;
        const queue = send(id, raw, "newCommandQueue", .{}) orelse return error.MetalUnavailable;
        // MLX's compile settings, so a kernel source gives the code it gives under MLX: safe math and the newest
        // language version the OS has.
        const options = send(id, send(id, objc_getClass("MTLCompileOptions"), "alloc", .{}), "init", .{});
        send(void, options, "setMathMode:", .{@as(isize, 0)});
        const OsVersion = extern struct { major: isize, minor: isize, patch: isize };
        const os = send(OsVersion, send(id, objc_getClass("NSProcessInfo"), "processInfo", .{}), "operatingSystemVersion", .{});
        send(void, options, "setLanguageVersion:", .{@as(usize, if (os.major >= 27) (4 << 16) + 1 else 4 << 16)});
        var ns_error: id = null;
        const descriptor = send(id, send(id, objc_getClass("MTLResidencySetDescriptor"), "alloc", .{}), "init", .{});
        defer send(void, descriptor, "release", .{});
        const resident = send(id, raw, "newResidencySetWithDescriptor:error:", .{ descriptor, &ns_error }) orelse {
            noteError(ns_error, "Metal refused a residency set");
            return error.MetalUnavailable;
        };
        send(void, resident, "requestResidency", .{});
        send(void, queue, "addResidencySet:", .{resident});
        return .{ .raw = raw, .queue = queue, .options = options, .resident = resident };
    }

    /// The compute pipeline of `function` in `source`, compiled now. `lastError` holds the compiler's text on failure.
    pub fn compile(dev: *const Device, source: [:0]const u8, function: [:0]const u8) Error!Pipeline {
        const pool = objc_autoreleasePoolPush();
        defer objc_autoreleasePoolPop(pool);
        var ns_error: id = null;
        const text = send(id, objc_getClass("NSString"), "stringWithUTF8String:", .{source.ptr});
        const library = send(id, dev.raw, "newLibraryWithSource:options:error:", .{ text, dev.options, &ns_error }) orelse {
            noteError(ns_error, "the library did not build");
            return error.MetalCompile;
        };
        defer send(void, library, "release", .{});
        const name = send(id, objc_getClass("NSString"), "stringWithUTF8String:", .{function.ptr});
        const entry = send(id, library, "newFunctionWithName:", .{name}) orelse {
            noteError(null, "the library has no such kernel function");
            return error.MetalCompile;
        };
        defer send(void, entry, "release", .{});
        const state = send(id, dev.raw, "newComputePipelineStateWithFunction:error:", .{ entry, &ns_error }) orelse {
            noteError(ns_error, "the pipeline did not build");
            return error.MetalCompile;
        };
        return .{ .raw = state, .max_threads = send(usize, state, "maxTotalThreadsPerThreadgroup", .{}) };
    }

    /// The compiled library at `path` (a `.metallib`).
    pub fn load(dev: *const Device, path: [:0]const u8) Error!Library {
        const pool = objc_autoreleasePoolPush();
        defer objc_autoreleasePoolPop(pool);
        var ns_error: id = null;
        const text = send(id, objc_getClass("NSString"), "stringWithUTF8String:", .{path.ptr});
        const url = send(id, objc_getClass("NSURL"), "fileURLWithPath:", .{text});
        const raw = send(id, dev.raw, "newLibraryWithURL:error:", .{ url, &ns_error }) orelse {
            noteError(ns_error, "the library did not load");
            return error.MetalCompile;
        };
        return .{ .raw = raw };
    }

    /// The compute pipeline of `function`, a kernel `lib` holds.
    pub fn pipelineOf(dev: *const Device, lib: Library, function: [:0]const u8) Error!Pipeline {
        const pool = objc_autoreleasePoolPush();
        defer objc_autoreleasePoolPop(pool);
        var ns_error: id = null;
        const name = send(id, objc_getClass("NSString"), "stringWithUTF8String:", .{function.ptr});
        const entry = send(id, lib.raw, "newFunctionWithName:", .{name}) orelse {
            noteError(null, function);
            return error.MetalCompile;
        };
        defer send(void, entry, "release", .{});
        const state = send(id, dev.raw, "newComputePipelineStateWithFunction:error:", .{ entry, &ns_error }) orelse {
            noteError(ns_error, "the pipeline did not build");
            return error.MetalCompile;
        };
        return .{ .raw = state, .max_threads = send(usize, state, "maxTotalThreadsPerThreadgroup", .{}) };
    }

    /// `pipelineOf` for a function specialized by boolean function constants, as MLX builds its matmul kernels.
    pub fn pipelineOfFlags(dev: *const Device, lib: Library, function: [:0]const u8, flags: []const Flag) Error!Pipeline {
        const pool = objc_autoreleasePoolPush();
        defer objc_autoreleasePoolPop(pool);
        var ns_error: id = null;
        const values = send(id, send(id, objc_getClass("MTLFunctionConstantValues"), "alloc", .{}), "init", .{});
        defer send(void, values, "release", .{});
        const mtl_data_type_bool: usize = 53;
        for (flags) |f| {
            var v = f.value;
            send(void, values, "setConstantValue:type:atIndex:", .{ @as(*const anyopaque, &v), mtl_data_type_bool, f.index });
        }
        const name = send(id, objc_getClass("NSString"), "stringWithUTF8String:", .{function.ptr});
        const entry = send(id, lib.raw, "newFunctionWithName:constantValues:error:", .{ name, values, &ns_error }) orelse {
            noteError(ns_error, function);
            return error.MetalCompile;
        };
        defer send(void, entry, "release", .{});
        const state = send(id, dev.raw, "newComputePipelineStateWithFunction:error:", .{ entry, &ns_error }) orelse {
            noteError(ns_error, "the pipeline did not build");
            return error.MetalCompile;
        };
        return .{ .raw = state, .max_threads = send(usize, state, "maxTotalThreadsPerThreadgroup", .{}) };
    }

    /// The GPU's architecture name (`applegpu_g17s`), written into `buf`.
    pub fn architecture(dev: *const Device, buf: []u8) []const u8 {
        const pool = objc_autoreleasePoolPush();
        defer objc_autoreleasePoolPop(pool);
        const name = send(id, send(id, dev.raw, "architecture", .{}), "name", .{});
        const text = std.mem.span(send([*:0]const u8, name, "UTF8String", .{}));
        const n = @min(text.len, buf.len);
        @memcpy(buf[0..n], text[0..n]);
        return buf[0..n];
    }

    /// A zeroed buffer of `len` bytes that Metal allocates.
    pub fn alloc(dev: *const Device, len: usize) Error!Buffer {
        const raw = send(id, dev.raw, "newBufferWithLength:options:", .{ @max(len, 1), alias_options }) orelse {
            noteError(null, "Metal refused a buffer");
            return error.MetalBuffer;
        };
        dev.keepResident(raw);
        return .{ .raw = raw };
    }

    pub fn fence(dev: *const Device) Error!Fence {
        return .{ .raw = send(id, dev.raw, "newFence", .{}) orelse return error.MetalUnavailable };
    }

    fn keepResident(dev: *const Device, buffer: id) void {
        send(void, dev.resident, "addAllocation:", .{buffer});
        resident_dirty = true;
    }

    /// A command buffer with one compute encoder whose dispatches run in the order they are encoded.
    pub fn begin(dev: *const Device) Pass {
        const pool = objc_autoreleasePoolPush();
        const cmd = send(id, dev.queue, "commandBuffer", .{});
        return .{ .pool = pool, .cmd = cmd, .encoder = send(id, cmd, "computeCommandEncoder", .{}) };
    }

    /// A command buffer with one compute encoder whose dispatches may overlap: the encoder's caller puts a
    /// `barrier` before a dispatch that reads what an earlier one wrote, or writes what an earlier one read.
    pub fn beginConcurrent(dev: *const Device) Pass {
        const pool = objc_autoreleasePoolPush();
        const cmd = send(id, dev.queue, "commandBuffer", .{});
        return .{ .pool = pool, .cmd = cmd, .encoder = send(id, cmd, "computeCommandEncoderWithDispatchType:", .{@as(usize, 1)}) };
    }
};

pub const Library = struct { raw: id };

/// Orders two passes whose buffers Metal does not track: one updates it, a later one waits for it.
pub const Fence = struct {
    raw: id,

    pub fn release(f: Fence) void {
        send(void, f.raw, "release", .{});
    }
};

pub const Pipeline = struct {
    raw: id,
    /// The most threads a threadgroup of this pipeline may hold.
    max_threads: usize,

    pub fn release(p: Pipeline) void {
        send(void, p.raw, "release", .{});
    }
};

pub const Buffer = struct {
    raw: id,

    pub fn bytes(b: Buffer) []u8 {
        return send([*]u8, b.raw, "contents", .{})[0..send(usize, b.raw, "length", .{})];
    }

    /// The owner's release: the buffer leaves the residency set. The inference thread only.
    pub fn release(b: Buffer) void {
        if (shared) |dev| {
            send(void, dev.resident, "removeAllocation:", .{b.raw});
            resident_dirty = true;
        }
        send(void, b.raw, "release", .{});
    }

    /// One more holder of the buffer, who lets go with `unretain`.
    pub fn retain(b: Buffer) Buffer {
        _ = send(id, b.raw, "retain", .{});
        return b;
    }

    pub fn unretain(b: Buffer) void {
        send(void, b.raw, "release", .{});
    }
};

/// A buffer and the byte in it where a tensor's data starts.
pub const Bound = struct { buffer: Buffer, offset: usize };

/// The pipeline the latest `Pass.pipeline` set, whichever pass: the one thread that encodes asks it at a dispatch.
pub var bound_pipeline: id = null;

pub const Pass = struct {
    pool: ?*anyopaque,
    cmd: id,
    encoder: id,

    pub fn pipeline(p: Pass, state: Pipeline) void {
        bound_pipeline = state.raw;
        send(void, p.encoder, "setComputePipelineState:", .{state.raw});
    }

    pub fn buffer(p: Pass, b: Buffer, offset: usize, index: usize) void {
        send(void, p.encoder, "setBuffer:offset:atIndex:", .{ b.raw, offset, index });
    }

    /// A small constant copied into the command buffer.
    pub fn constant(p: Pass, value: anytype, index: usize) void {
        const v = value;
        send(void, p.encoder, "setBytes:length:atIndex:", .{ @as(*const anyopaque, &v), @as(usize, @sizeOf(@TypeOf(v))), index });
    }

    /// A few constant bytes copied into the command buffer.
    pub fn bytes(p: Pass, data: []const u8, index: usize) void {
        send(void, p.encoder, "setBytes:length:atIndex:", .{ @as(*const anyopaque, data.ptr), data.len, index });
    }

    pub fn dispatch(p: Pass, grid: Size, group: Size) void {
        send(void, p.encoder, "dispatchThreads:threadsPerThreadgroup:", .{ grid, group });
    }

    /// The pass does not start before the pass that updates `fence` has run.
    pub fn waitFence(p: Pass, fence: Fence) void {
        send(void, p.encoder, "waitForFence:", .{fence.raw});
    }

    /// `fence` is reached when this pass has run.
    pub fn updateFence(p: Pass, fence: Fence) void {
        send(void, p.encoder, "updateFence:", .{fence.raw});
    }

    /// Every dispatch encoded so far finishes its buffer writes before the next one starts.
    pub fn barrier(p: Pass) void {
        send(void, p.encoder, "memoryBarrierWithScope:", .{@as(usize, 1)});
    }

    /// `grid` counts threadgroups, not threads.
    pub fn dispatchGroups(p: Pass, grid: Size, group: Size) void {
        send(void, p.encoder, "dispatchThreadgroups:threadsPerThreadgroup:", .{ grid, group });
    }

    /// Hands the command buffer to the GPU. Every buffer it binds lives until the pass has run (`Committed.wait`,
    /// `waitDone`). No shared event is signaled: one costs about 8 us of CPU per command buffer on Metal's threads.
    pub fn commit(p: Pass) Committed {
        send(void, p.encoder, "endEncoding", .{});
        if (resident_dirty) {
            resident_dirty = false;
            send(void, shared.?.resident, "commit", .{});
        }
        _ = send(id, p.cmd, "retain", .{});
        send(void, p.cmd, "commit", .{});
        objc_autoreleasePoolPop(p.pool);
        return .{ .cmd = p.cmd };
    }
};

pub const Committed = struct {
    cmd: id,

    /// Whether the command buffer failed; Metal's text is then in `lastError`.
    pub fn failed(c: Committed) bool {
        const errored = 5;
        if (send(usize, c.cmd, "status", .{}) != errored) return false;
        noteError(send(id, c.cmd, "error", .{}), "the command buffer failed");
        return true;
    }

    /// Whether the command buffer has run, or failed.
    pub fn ended(c: Committed) bool {
        const completed = 4;
        return send(usize, c.cmd, "status", .{}) >= completed;
    }

    /// Blocks until the command buffer has run, or failed; the caller still holds it.
    pub fn waitDone(c: Committed) void {
        send(void, c.cmd, "waitUntilCompleted", .{});
    }

    pub fn release(c: Committed) void {
        send(void, c.cmd, "release", .{});
    }

    /// When the GPU began and ended the command buffer, seconds on Metal's clock; zeros until it has completed.
    pub fn gpuSpan(c: Committed) [2]f64 {
        return .{ send(f64, c.cmd, "GPUStartTime", .{}), send(f64, c.cmd, "GPUEndTime", .{}) };
    }

    /// Blocks until the GPU has run the command buffer; its failure is this call's error, with Metal's text in
    /// `lastError`.
    pub fn wait(c: Committed) Error!void {
        defer send(void, c.cmd, "release", .{});
        send(void, c.cmd, "waitUntilCompleted", .{});
        const completed = 4;
        if (send(usize, c.cmd, "status", .{}) == completed) return;
        noteError(send(id, c.cmd, "error", .{}), "the command buffer failed");
        return error.MetalCommand;
    }
};

const probe_source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\[[kernel]] void sushi_metal_probe(
    \\  const device float* x [[buffer(0)]],
    \\  device float* out [[buffer(1)]],
    \\  const constant float& scale [[buffer(2)]],
    \\  uint3 at [[thread_position_in_grid]]) {
    \\  out[at.x] = x[at.x] * scale;
    \\}
;

test "a kernel compiled and dispatched from Zig reads one buffer and writes another" {
    const t = std.testing;
    const dev = device() catch return error.SkipZigTest;
    const state = try dev.compile(probe_source, "sushi_metal_probe");
    defer state.release();
    try t.expect(state.max_threads >= 64);

    const n = 1000;
    const x = try dev.alloc(n * 4);
    defer x.release();
    const out = try dev.alloc(n * 4);
    defer out.release();
    const xs: []f32 = @alignCast(std.mem.bytesAsSlice(f32, x.bytes()[0 .. n * 4]));
    for (xs, 0..) |*v, i| v.* = @floatFromInt(i);

    const pass = dev.begin();
    pass.pipeline(state);
    pass.buffer(x, 0, 0);
    pass.buffer(out, 0, 1);
    pass.constant(@as(f32, 1.5), 2);
    pass.dispatch(.{ .x = n }, .{ .x = 64 });
    try pass.commit().wait();

    const got: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, out.bytes()[0 .. n * 4]));
    for (got, 0..) |v, i| try t.expectEqual(@as(f32, @floatFromInt(i)) * 1.5, v);
}

test "a dispatch reads what the dispatch before it wrote" {
    const t = std.testing;
    const dev = device() catch return error.SkipZigTest;
    const state = try dev.compile(probe_source, "sushi_metal_probe");
    defer state.release();
    const n = 4096;
    const a = try dev.alloc(n * 4);
    defer a.release();
    const b = try dev.alloc(n * 4);
    defer b.release();
    const as: []f32 = @alignCast(std.mem.bytesAsSlice(f32, a.bytes()[0 .. n * 4]));
    for (as, 0..) |*v, i| v.* = @floatFromInt(i % 97);

    // a -> b, b -> a, a -> b in one command buffer: 2 * 3 * 5 times the start.
    const pass = dev.begin();
    pass.pipeline(state);
    for ([_]f32{ 2, 3, 5 }, 0..) |scale, step| {
        pass.buffer(if (step % 2 == 0) a else b, 0, 0);
        pass.buffer(if (step % 2 == 0) b else a, 0, 1);
        pass.constant(scale, 2);
        pass.dispatch(.{ .x = n }, .{ .x = 256 });
    }
    try pass.commit().wait();
    const got: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, b.bytes()[0 .. n * 4]));
    for (got, 0..) |v, i| try t.expectEqual(@as(f32, @floatFromInt(i % 97)) * 30, v);
}

test "a source that does not compile is an error that carries the compiler's text" {
    const t = std.testing;
    const dev = device() catch return error.SkipZigTest;
    try t.expectError(error.MetalCompile, dev.compile("[[kernel]] void broken(uint3 at [[thread_position_in_grid]]) { not_a_thing(at); }", "broken"));
    try t.expect(std.mem.indexOf(u8, lastError(), "not_a_thing") != null);
    try t.expectError(error.MetalCompile, dev.compile(probe_source, "no_such_function"));
}
