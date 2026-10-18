const std = @import("std");

/// Persistent worker threads for `run`: one LDLQ step dispatches two parallel loops, and spawning
/// fresh threads for each cost more than the loop bodies. One caller at a time per pool.
pub const Pool = struct {
    mutex: std.c.pthread_mutex_t = .{},
    wake: std.c.pthread_cond_t = .{},
    idle: std.c.pthread_cond_t = .{},
    threads: []std.Thread = &.{},
    allocator: std.mem.Allocator,
    generation: u64 = 0,
    stop: bool = false,
    busy: usize = 0,
    count: usize = 0,
    next: std.atomic.Value(usize) = .init(0),
    ctx: *anyopaque = undefined,
    body: *const fn (*anyopaque, usize) void = undefined,

    pub fn init(allocator: std.mem.Allocator, workers: usize) !*Pool {
        const self = try allocator.create(Pool);
        errdefer allocator.destroy(self);
        self.* = .{ .allocator = allocator };
        const threads = try allocator.alloc(std.Thread, workers);
        errdefer allocator.free(threads);
        var spawned: usize = 0;
        errdefer {
            self.shutdown(threads[0..spawned]);
        }
        while (spawned < workers) : (spawned += 1) threads[spawned] = try std.Thread.spawn(.{}, loop, .{self});
        self.threads = threads;
        return self;
    }

    fn shutdown(self: *Pool, threads: []std.Thread) void {
        _ = std.c.pthread_mutex_lock(&self.mutex);
        self.stop = true;
        _ = std.c.pthread_cond_broadcast(&self.wake);
        _ = std.c.pthread_mutex_unlock(&self.mutex);
        for (threads) |t| t.join();
    }

    pub fn deinit(self: *Pool) void {
        self.shutdown(self.threads);
        self.allocator.free(self.threads);
        self.allocator.destroy(self);
    }

    fn drain(self: *Pool) void {
        while (true) {
            const i = self.next.fetchAdd(1, .monotonic);
            if (i >= self.count) return;
            self.body(self.ctx, i);
        }
    }

    fn loop(self: *Pool) void {
        var seen: u64 = 0;
        while (true) {
            _ = std.c.pthread_mutex_lock(&self.mutex);
            while (!self.stop and self.generation == seen) _ = std.c.pthread_cond_wait(&self.wake, &self.mutex);
            if (self.stop) {
                _ = std.c.pthread_mutex_unlock(&self.mutex);
                return;
            }
            seen = self.generation;
            _ = std.c.pthread_mutex_unlock(&self.mutex);
            self.drain();
            _ = std.c.pthread_mutex_lock(&self.mutex);
            self.busy -= 1;
            if (self.busy == 0) _ = std.c.pthread_cond_signal(&self.idle);
            _ = std.c.pthread_mutex_unlock(&self.mutex);
        }
    }

    /// body(ctx, i) for i in 0..count on the workers and the calling thread; returns when all are done.
    pub fn run(self: *Pool, count: usize, ctx: anytype, comptime body: fn (@TypeOf(ctx), usize) void) void {
        if (count == 0) return;
        if (count == 1 or self.threads.len == 0) {
            for (0..count) |i| body(ctx, i);
            return;
        }
        self.dispatch(count, ctx, body);
        self.drain();
        self.wait();
    }

    /// `run` on the workers alone: returns at once, and `wait` blocks until the loop is done. `ctx` is a
    /// pointer the caller keeps alive until then.
    pub fn start(self: *Pool, count: usize, ctx: anytype, comptime body: fn (@TypeOf(ctx), usize) void) void {
        if (count == 0) return;
        if (self.threads.len == 0) {
            for (0..count) |i| body(ctx, i);
            return;
        }
        self.dispatch(count, ctx, body);
    }

    pub fn wait(self: *Pool) void {
        _ = std.c.pthread_mutex_lock(&self.mutex);
        while (self.busy != 0) _ = std.c.pthread_cond_wait(&self.idle, &self.mutex);
        _ = std.c.pthread_mutex_unlock(&self.mutex);
    }

    fn dispatch(self: *Pool, count: usize, ctx: anytype, comptime body: fn (@TypeOf(ctx), usize) void) void {
        const Ctx = @TypeOf(ctx);
        comptime std.debug.assert(@typeInfo(Ctx) == .pointer);
        const Wrap = struct {
            fn call(raw: *anyopaque, i: usize) void {
                const c: Ctx = @ptrCast(@alignCast(raw));
                body(c, i);
            }
        };
        _ = std.c.pthread_mutex_lock(&self.mutex);
        self.ctx = @ptrCast(@constCast(ctx));
        self.body = Wrap.call;
        self.count = count;
        self.next.store(0, .monotonic);
        self.busy = self.threads.len;
        self.generation += 1;
        _ = std.c.pthread_cond_broadcast(&self.wake);
        _ = std.c.pthread_mutex_unlock(&self.mutex);
    }
};

test "pool: every index runs exactly once across repeated dispatches" {
    const t = std.testing;
    const pool = try Pool.init(t.allocator, 5);
    defer pool.deinit();
    var hits: [1000]std.atomic.Value(u32) = undefined;
    for (&hits) |*h| h.* = .init(0);
    const Body = struct {
        fn f(h: *[1000]std.atomic.Value(u32), i: usize) void {
            _ = h[i].fetchAdd(1, .monotonic);
        }
    };
    for (0..50) |_| pool.run(hits.len, &hits, Body.f);
    for (0..10) |_| {
        pool.start(hits.len, &hits, Body.f);
        pool.wait();
    }
    for (&hits) |*h| try t.expectEqual(@as(u32, 60), h.load(.monotonic));
}

/// A one-shot flag a thread can wait on.
pub const Event = struct {
    mutex: std.c.pthread_mutex_t = .{},
    cond: std.c.pthread_cond_t = .{},
    set: bool = false,

    pub fn signal(self: *Event) void {
        _ = std.c.pthread_mutex_lock(&self.mutex);
        self.set = true;
        _ = std.c.pthread_cond_broadcast(&self.cond);
        _ = std.c.pthread_mutex_unlock(&self.mutex);
    }

    pub fn wait(self: *Event) void {
        _ = std.c.pthread_mutex_lock(&self.mutex);
        while (!self.set) _ = std.c.pthread_cond_wait(&self.cond, &self.mutex);
        _ = std.c.pthread_mutex_unlock(&self.mutex);
    }
};
