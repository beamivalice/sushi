//! Where a model load spends its wall time, phase by phase, printed once as the `[load]` line. A phase is timed
//! where it runs; one that cannot be separated stays in its parent (the build includes its view evals, the warm
//! forwards include MLX's own kernel compiles). Loads run on the inference thread, one at a time.

const std = @import("std");

pub const Phases = struct {
    started: u64 = 0,
    running: bool = false,
    headers_ns: u64 = 0,
    read_ns: u64 = 0,
    read_bytes: u64 = 0,
    build_ns: u64 = 0,
    pipelines_ns: u64 = 0,
    compiled: u64 = 0,
    warm_ns: u64 = 0,
    wired_ns: u64 = 0,
    cache_ns: u64 = 0,
    total_ns: u64 = 0,
    /// The warm's whole span, pipelines included: what the build must not count.
    warm_span_ns: u64 = 0,

    /// What the phases leave uncounted: the load's own bookkeeping outside them.
    pub fn otherNs(p: *const Phases) u64 {
        const counted = p.headers_ns + p.read_ns + p.build_ns + p.pipelines_ns + p.warm_ns + p.wired_ns + p.cache_ns;
        return p.total_ns -| counted;
    }
};

pub var load: Phases = .{};

pub fn now() u64 {
    var t: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &t);
    return @as(u64, @intCast(t.sec)) * std.time.ns_per_s + @as(u64, @intCast(t.nsec));
}

pub fn begin() void {
    load = .{ .started = now(), .running = true };
}

pub fn end() void {
    if (!load.running) return;
    load.total_ns = now() - load.started;
    load.running = false;
}

pub fn headersDone(t0: u64) void {
    if (load.running) load.headers_ns += now() - t0;
}

pub fn readDone(t0: u64, bytes: u64) void {
    if (!load.running) return;
    load.read_ns += now() - t0;
    load.read_bytes += bytes;
}

pub fn wiredDone(t0: u64) void {
    if (load.running) load.wired_ns += now() - t0;
}

pub fn cacheDone(t0: u64) void {
    if (load.running) load.cache_ns += now() - t0;
}

/// The trunk's construction since `t0`, less the warm that ran inside it.
pub fn buildDone(t0: u64) void {
    if (load.running) load.build_ns += (now() - t0) -| load.warm_span_ns;
}

/// The warm since `t0`; `pipelines_ns` of it went to making engine pipelines, `compiled` of them by a compiler.
pub fn warmDone(t0: u64, pipelines_ns: u64, compiled: u64) void {
    if (!load.running) return;
    const span = now() - t0;
    load.warm_span_ns += span;
    load.pipelines_ns += pipelines_ns;
    load.warm_ns += span -| pipelines_ns;
    load.compiled += compiled;
}

fn ms(ns: u64) u64 {
    return ns / std.time.ns_per_ms;
}

pub fn line(buf: []u8, p: *const Phases) []const u8 {
    const gb = @as(f64, @floatFromInt(p.read_bytes)) / 1e9;
    const secs = @as(f64, @floatFromInt(p.read_ns)) / 1e9;
    return std.fmt.bufPrint(buf, "[load] headers {d} ms, read {d} ms ({d:.1} GB, {d:.1} GB/s), build {d} ms, pipelines {d} ms (compiled {d}), warm forwards {d} ms, wired {d} ms, expert cache {d} ms, other {d} ms, total {d} ms", .{
        ms(p.headers_ns),   ms(p.read_ns),   gb,           if (secs > 0) gb / secs else 0,
        ms(p.build_ns),     ms(p.pipelines_ns), p.compiled,
        ms(p.warm_ns),      ms(p.wired_ns), ms(p.cache_ns), ms(p.otherNs()),
        ms(p.total_ns),
    }) catch buf[0..0];
}
