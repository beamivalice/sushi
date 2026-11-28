//! DeepSeek-V4.1 persistent request state, row stores, fixed owners, point slots, and zero-pool restore targets.
//! Separated from forward orchestration (`dsv41_model.zig`) so state layout, reservation, eager in-place landing, and
//! point slot ownership live in one place.
const std = @import("std");
const mlx = @import("../mlx.zig");
const mlx_pages = @import("../mlx_pages.zig");
const cfg_mod = @import("dsv41_config.zig");
const sc_mod = @import("dsv41_scope.zig");
const qat_mod = @import("dsv41_qat.zig");
const exl3 = @import("sushi_exl3");

pub const A = sc_mod.A;
pub const S = sc_mod.S;
pub const Scope = sc_mod.Scope;
pub const none = sc_mod.none;
pub const dim = sc_mod.dim;
pub const ci = sc_mod.ci;

/// Appends that had to grow a store: none while a request that reserved/bound its stores runs.
pub var store_grows: u64 = 0;

/// The arrays one slot of a restore point keeps; the prefix cache stores them in an SSM snapshot's four fields.
pub const Slot = [4]A;
pub const nil_slot: Slot = @splat(none);

pub const exclusive_point_slots: usize = 12;
pub const joined_point_slots: usize = 4;
pub const max_point_slots: usize = exclusive_point_slots;
pub const narrow_cap: usize = cfg_mod.verify_rows;

/// Rows appended one chunk at a time, in a buffer made at the rows its request ends at, or for a store no request has
/// bound, grown by `rows_grow_div`'s fraction.
pub const Rows = struct {
    /// The rows `[cap, width]` in `dtype`, or packed: E2M1 codes `[cap, width/2]` u8.
    buf: mlx_pages.InPlace = .none,
    /// Packed: the scale bytes `[cap, width/group]`.
    sbuf: mlx_pages.InPlace = .none,
    used: usize = 0,
    cap: usize = 0,
    /// The rows the request that holds this store ends at (`bindRequest`); 0 = no request has bound it, and it grows by
    /// `rows_grow_div`'s fraction.
    goal: usize = 0,
    width: usize,
    dtype: mlx.mlx_dtype,
    /// The rows are an FP4 round trip's output, stored in its own format (`qat_mod.packFp4`).
    pack: ?Pack = null,

    pub const Pack = struct { mode: qat_mod.Qat, group: usize };

    pub fn deinit(self: *Rows) void {
        self.buf.deinit();
        self.sbuf.deinit();
    }

    /// A new buffer of `cap` rows of `cols` (`cap` from `pagedCap`) holding `buf`'s first `@min(used, cap)` rows; `buf`
    /// is left as it was, so `growTo` builds every part before it frees any old one.
    fn grown(buf: mlx_pages.InPlace, used: usize, cap: usize, cols: usize, dt: mlx.mlx_dtype, s: S) !mlx_pages.InPlace {
        var z = try mlx_pages.inPlaceBuffer(&.{ ci(cap), ci(cols) }, dt, s);
        errdefer z.deinit();
        const keep = @min(used, cap);
        if (keep > 0) {
            var old = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(old);
            try mlx.check(mlx.mlx_slice(&old, buf.array, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ ci(keep), ci(cols) }, 2, &[_]c_int{ 1, 1 }, 2, s));
            var up = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(up);
            const next = try z.write(&up, old, &.{ 0, 0 }, &.{ ci(keep), ci(cols) }, s);
            z.deinit();
            z = next;
        }
        return z;
    }

    /// Every part made `cap` rows long with the rows held kept, all or none: each part's new buffer is built (and, with
    /// `written`, evaluated) before any old one is freed, so a failure at any part frees what was built and leaves every
    /// part and `cap` as they were, never a part and `cap` that disagree.
    pub fn growTo(self: *Rows, cap: usize, s: S, written: bool) !void {
        const owned = if (mlx.engine.isStream(s)) mlx.engine.widePhase() else null;
        defer if (owned) |o| o.end();
        const Shape = struct { cols: usize, dtype: mlx.mlx_dtype };
        const olds = [2]*mlx_pages.InPlace{ &self.buf, &self.sbuf };
        var shapes: [2]Shape = undefined;
        var n: usize = 1;
        shapes[0] = .{ .cols = self.width, .dtype = self.dtype };
        if (self.pack) |pk| {
            n = 2;
            shapes = .{ .{ .cols = self.width / 2, .dtype = .uint8 }, .{ .cols = self.width / pk.group, .dtype = .uint8 } };
        }
        var next: [2]mlx_pages.InPlace = @splat(.none);
        errdefer for (&next) |*b| b.deinit();
        for (olds[0..n], shapes[0..n], next[0..n]) |old, sh, *dst| dst.* = try grown(old.*, self.used, cap, sh.cols, sh.dtype, s);
        if (written) {
            // A grow left pending would take its buffer inside the forward that first reads it.
            var i = n;
            while (i > 0) : (i -= 1) try mlx.check(mlx.mlx_array_eval(next[i - 1].array));
        }
        for (olds[0..n], next[0..n]) |old, *z| {
            old.deinit();
            old.* = z.*;
            z.* = .none;
        }
        self.cap = cap;
    }

    pub fn put(buf: *mlx_pages.InPlace, x: A, at: usize, s: S) !void {
        const n = dim(x, 0);
        const cols = dim(x, 1);
        const next = blk: {
            var up = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(up);
            break :blk try buf.write(&up, x, &.{ ci(at), 0 }, &.{ ci(at + n), ci(cols) }, s);
        };
        _ = mlx.mlx_array_free(buf.array);
        buf.* = next;
        if (@import("../tensor/plan/plan_api.zig").planned()) try next.landed(s);
    }

    /// `count` rows on MLX's page grid for every buffer that grows with these rows (`mlx_pages`), so each append writes in
    /// place; the engine of our own appends in place at any size.
    pub fn pagedCap(self: *const Rows, count: usize, s: S) usize {
        if (mlx.engine.isStream(s)) return count;
        const item: u64 = @import("../tensor/core.zig").itemSize(self.dtype);
        const parts: []const u64 = if (self.pack) |pk| &.{ self.width / 2, self.width / pk.group } else &.{self.width * item};
        return @intCast(mlx_pages.pagedCountAll(count, parts));
    }

    /// Whether this store is no larger than a request billed `billed` rows pays for: the bill on MLX's page grid, which
    /// `Config.mlxForwardCap` adds (`rowsGridBytes`) and `pagedCap` sizes every store by.
    pub fn fitsBill(self: *const Rows, billed: usize, s: S) bool {
        return self.cap <= self.pagedCap(billed, s);
    }

    /// The grow the next append would make, made now: a buffer of this state's own with that append's slack.
    pub fn reserve(self: *Rows, s: S) !void {
        if (self.used == 0) return;
        try self.regrow(self.pagedCap(cfg_mod.reservedRows(self.used), s), s);
    }

    /// The store of a request that ends at `goal` rows: its buffers made that long now with the rows it holds kept (none
    /// yet, when it holds none), so no append makes or grows one.
    pub fn bind(self: *Rows, goal: usize, s: S) !void {
        self.goal = goal;
        const cap = self.pagedCap(@max(goal, self.used), s);
        if (cap != self.cap) try self.regrow(cap, s);
    }

    /// Cooler reservation: size the store to `cap` rows (growing or shrinking cleanly between requests), or free when 0.
    pub fn reserveFor(self: *Rows, cap: usize, s: S) !void {
        self.goal = cap;
        const target = self.pagedCap(@max(cap, self.used), s);
        if (target == 0) {
            self.deinit();
            self.cap = 0;
            self.used = 0;
            return;
        }
        const need_sole = (self.buf.array.ctx != null and !try mlx.engine.sole(self.buf.array)) or
            (self.sbuf.array.ctx != null and !try mlx.engine.sole(self.sbuf.array));
        if (target != self.cap or need_sole) try self.regrow(target, s);
    }

    /// The buffers made `cap` rows long with the rows held kept, written now.
    pub fn regrow(self: *Rows, cap: usize, s: S) !void {
        try self.growTo(cap, s, true);
    }

    pub fn truncate(self: *Rows, used: usize) void {
        self.used = @min(self.used, used);
    }

    /// The buffer cut back to the rows it holds with `reserve`'s slack, or gone when it holds none.
    pub fn trim(self: *Rows, s: S) !void {
        if (self.used > 0) return self.reserve(s);
        self.deinit();
        self.cap = 0;
    }

    pub fn append(self: *Rows, sc: *Scope, x: A) !void {
        const n = dim(x, 0);
        if (n == 0) return;
        const s = sc.s;
        if (self.used + n > self.cap) {
            store_grows += 1;
            const cap = self.pagedCap(cfg_mod.grownRows(self.cap, self.used, n, self.goal), s);
            try self.growTo(cap, s, false);
        }
        if (self.pack) |pk| {
            const p = try qat_mod.packFp4(sc, try sc.astype(x, self.dtype), pk.mode, pk.group);
            try put(&self.buf, p.codes, self.used, s);
            try put(&self.sbuf, p.scales, self.used, s);
        } else try put(&self.buf, try sc.astype(x, self.dtype), self.used, s);
        self.used += n;
    }

    /// Rows [0, n) as values.
    pub fn rows(self: *const Rows, sc: *Scope, n: usize) !A {
        return self.rowsIn(sc, 0, n);
    }

    /// Rows [lo, hi) as values.
    pub fn rowsIn(self: *const Rows, sc: *Scope, lo: usize, hi: usize) !A {
        const head = try sc.slice(self.buf.array, 0, lo, hi);
        const pk = self.pack orelse return head;
        return qat_mod.unpackFp4(sc, .{ .codes = head, .scales = try sc.slice(self.sbuf.array, 0, lo, hi) }, pk.mode, pk.group, self.dtype);
    }

    /// Rows `idx` (flat ids below `used`) as values.
    pub fn gather(self: *const Rows, sc: *Scope, idx: A) !A {
        const pk = self.pack orelse return sc.take(self.buf.array, idx, 0);
        return qat_mod.unpackFp4(sc, .{ .codes = try sc.take(self.buf.array, idx, 0), .scales = try sc.take(self.sbuf.array, idx, 0) }, pk.mode, pk.group, self.dtype);
    }
};

/// The window's rows by absolute position, in a ring of `cap` rows.
pub const Ring = struct {
    buf: mlx_pages.InPlace,
    cap: usize,
    width: usize,
    /// The buffer's dtype, so a launch that takes the ring reads nothing off it.
    dt: mlx.mlx_dtype,

    pub fn init(cap: usize, width: usize, dt: mlx.mlx_dtype, s: S) !Ring {
        if (mlx.engine.isStream(s)) {
            return .{ .buf = try mlx_pages.ringBuffer(&.{ ci(cap), ci(width) }, dt, s), .cap = cap, .width = width, .dt = dt };
        }
        var z = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(z);
        try mlx.check(mlx.mlx_zeros(&z, &[_]c_int{ ci(cap), ci(width) }, 2, dt, s));
        return .{ .buf = mlx_pages.InPlace.adopt(z, .ring), .cap = cap, .width = width, .dt = dt };
    }

    pub fn deinit(self: *Ring) void {
        self.buf.deinit();
    }

    pub fn landed(self: Ring, s: S) !void {
        try self.buf.landed(s);
    }

    /// Rows for positions [p0, p0 + n); only the last `cap` survive.
    pub fn write(self: *Ring, x: A, p0: usize, s: S) !void {
        const n = dim(x, 0);
        const keep_n = @min(n, self.cap);
        const first = p0 + n - keep_n;
        var done: usize = 0;
        while (done < keep_n) {
            const slot = (first + done) % self.cap;
            const run = @min(keep_n - done, self.cap - slot);
            var part = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(part);
            try mlx.check(mlx.mlx_slice(&part, x, &[_]c_int{ ci(n - keep_n + done), 0 }, 2, &[_]c_int{ ci(n - keep_n + done + run), ci(self.width) }, 2, &[_]c_int{ 1, 1 }, 2, s));
            var cast = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(cast);
            try mlx.check(mlx.mlx_astype(&cast, part, mlx.mlx_array_dtype(self.buf.array), s));
            const next = blk: {
                var up = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(up);
                break :blk try self.buf.write(&up, cast, &.{ ci(slot), 0 }, &.{ ci(slot + run), ci(self.width) }, s);
            };
            _ = mlx.mlx_array_free(self.buf.array);
            self.buf = next;
            if (@import("../tensor/plan/plan_api.zig").planned()) try next.landed(s);
            done += run;
        }
    }

    /// The ring as `write` of rows `x` at `p0` would leave it, made by another launch: `next` replaces the buffer.
    pub fn replace(self: *Ring, next: A) void {
        const kept = self.buf.replaced(next);
        _ = mlx.mlx_array_free(self.buf.array);
        self.buf = kept;
    }

    /// Writes `src` (`[cap, width]`) directly into the ring's buffer in place on Cooler (`landed`).
    pub fn writeAll(self: *Ring, src: A, s: S) !void {
        var arr: ?A = self.buf.array;
        try writeFixed(&arr, src, &.{ ci(self.cap), ci(self.width) }, s);
        self.buf = mlx_pages.InPlace.adopt(arr.?, .ring);
    }

    /// Rows for positions [lo, hi) in order; `hi - lo <= cap`.
    pub fn read(self: *const Ring, sc: *Scope, lo: usize, hi: usize) !A {
        std.debug.assert(hi - lo <= self.cap);
        var idx: [512]i32 = undefined;
        for (lo..hi, 0..) |p, k| idx[k] = @intCast(p % self.cap);
        return sc.take(self.buf.array, try sc.ints(idx[0 .. hi - lo], &.{ci(hi - lo)}), 0);
    }
};

pub const LState = struct {
    win: Ring,
    comp: ?Rows = null,
    idx_k: ?Rows = null,
    /// Ratio > 1 compressors: views (or dynamic MLX arrays) of the open group's kv and gate rows, f32.
    pend_kv: A = none,
    pend_gate: A = none,
    /// Fixed Cooler owner blocks `[r - 1, head_dim]` f32 for `pend_kv` / `pend_gate`.
    pend_kv_buf: ?A = null,
    pend_gate_buf: ?A = null,

    pub fn deinit(self: *LState) void {
        self.win.deinit();
        if (self.comp) |*r| r.deinit();
        if (self.idx_k) |*r| r.deinit();
        if (self.pend_kv.ctx != null) _ = mlx.mlx_array_free(self.pend_kv);
        if (self.pend_gate.ctx != null) _ = mlx.mlx_array_free(self.pend_gate);
        if (self.pend_kv_buf) |b| _ = mlx.mlx_array_free(b);
        if (self.pend_gate_buf) |b| _ = mlx.mlx_array_free(b);
    }
};

/// What a verify chunk keeps to give back its rejected rows: each compressor's row counts and open pooling group
/// as the chunk began, and the main rows the stages' windows take only for the rows that stay.
pub const Verify = struct {
    n0: usize,
    rows: usize,
    comp_used: []usize,
    idx_used: []usize,
    /// A pooling compressor's rows from its open group's first token on, as the chunk left them before it cut a new group.
    pend_kv: []A,
    pend_gate: []A,
    mh: A = none,
};

/// Pre-allocated fixed restore-point slots on `State` (`exclusive_point_slots` or `joined_point_slots`) so live points
/// and rollbacks copy into pre-owned buffers with zero pool allocations and keep live state buffers `sole`.
pub const Points = struct {
    cap: usize,
    slots: [max_point_slots]PointSlot = @splat(.{}),

    pub const PointSlot = struct {
        /// Per layer: `[win, pend_kv, pend_gate]`.
        layers: [cfg_mod.max_layers][3]?A = @splat(@splat(null)),
        /// Per DSpark stage: stage window ring.
        ds: [cfg_mod.max_targets]?A = @splat(null),

        fn deinit(self: *PointSlot, n_layers: usize, n_ds: usize) void {
            for (self.layers[0..n_layers]) |*row| for (row) |*x| if (x.*) |a| {
                _ = mlx.mlx_array_free(a);
                x.* = null;
            };
            for (self.ds[0..n_ds]) |*x| if (x.*) |a| {
                _ = mlx.mlx_array_free(a);
                x.* = null;
            };
        }

        fn isFree(self: *const PointSlot, n_layers: usize, n_ds: usize) !bool {
            for (self.layers[0..n_layers]) |row| for (row) |x| if (x) |a| {
                if (!try mlx.engine.sole(a)) return false;
            };
            for (self.ds[0..n_ds]) |x| if (x) |a| {
                if (!try mlx.engine.sole(a)) return false;
            };
            return true;
        }
    };

    pub fn deinit(self: *Points, n_layers: usize, n_ds: usize) void {
        for (self.slots[0..self.cap]) |*sl| sl.deinit(n_layers, n_ds);
    }

    pub fn allocSlot(self: *Points, n_layers: usize, n_ds: usize) !usize {
        for (self.slots[0..self.cap], 0..) |*sl, i| {
            if (try sl.isFree(n_layers, n_ds)) return i;
        }
        return error.Dsv41PointsFull;
    }
};

pub const SpecUndo = struct {
    win: Ring = .{ .buf = .none, .cap = 0, .width = 0, .dt = .bfloat16 },
    pend_kv_buf: ?A = null,
    pend_gate_buf: ?A = null,
    verify_kv_buf: ?A = null,
    verify_gate_buf: ?A = null,
    max_ratio: usize = 0,

    pub fn deinit(self: *SpecUndo) void {
        self.win.deinit();
        for (&[_]*?A{ &self.pend_kv_buf, &self.pend_gate_buf, &self.verify_kv_buf, &self.verify_gate_buf }) |slot| {
            if (slot.*) |a| {
                _ = mlx.mlx_array_free(a);
                slot.* = null;
            }
        }
        self.max_ratio = 0;
    }
};

pub const State = struct {
    gpa: std.mem.Allocator,
    /// Which state this is, for a record of it (`dsv41_prefix.Live`): a made state takes a new one, and the state keeps
    /// it as it rolls back or moves between its model and a request.
    id: u64 = 0,
    /// The KV cache of the request whose forward last moved this state; 0 once a rollback hands it to the next one.
    /// Only that request publishes a record of it (`ownedBy`).
    owner: usize = 0,
    n: usize = 0,
    reach: usize = 0,
    layers: []LState,
    /// The request's compressed token ids (Engram look-back).
    hist: std.ArrayList(u32) = .empty,
    /// Per DSpark stage: the window of the main stream's keys for the committed positions.
    ds: []Ring = &.{},
    /// The stage rings hold a commit no pass has submitted yet (`commitMain` lazy); `flushMain` submits it.
    main_lazy: bool = false,
    verify: ?Verify = null,
    /// Fixed Cooler verify buffers (`r - 1 + ring_slack` rows) and main hidden rows (`ring_slack` rows).
    verify_pend_kv: [cfg_mod.max_layers]?A = @splat(null),
    verify_pend_gate: [cfg_mod.max_layers]?A = @splat(null),
    verify_mh: ?A = null,
    /// Pre-allocated Cooler point slots for zero-pool live points & rollbacks.
    points: ?*Points = null,
    /// Pre-allocated single-layer undo scratch for zero-pool `specOn` rollbacks on Cooler.
    spec_undo: SpecUndo = .{},
    /// `bindRequest` made this state for a request that has not run yet: its first forward keeps it, where it would
    /// reset a state left by another (`takeBoundState`).
    bound_fresh: bool = false,

    pub fn deinit(self: *State) void {
        const nl = self.layers.len;
        const n_ds = self.ds.len;
        if (self.points) |pts| {
            pts.deinit(nl, n_ds);
            self.gpa.destroy(pts);
            self.points = null;
        }
        for (self.verify_pend_kv[0..nl]) |*x| if (x.*) |a| {
            _ = mlx.mlx_array_free(a);
            x.* = null;
        };
        for (self.verify_pend_gate[0..nl]) |*x| if (x.*) |a| {
            _ = mlx.mlx_array_free(a);
            x.* = null;
        };
        if (self.verify_mh) |a| {
            _ = mlx.mlx_array_free(a);
            self.verify_mh = null;
        }
        self.spec_undo.deinit();
        for (self.layers) |*l| l.deinit();
        self.gpa.free(self.layers);
        for (self.ds) |*r| r.deinit();
        self.gpa.free(self.ds);
        self.hist.deinit(self.gpa);
        self.dropVerify();
    }

    pub fn dropVerify(self: *State) void {
        const v = self.verify orelse return;
        for (v.pend_kv) |x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x);
        };
        for (v.pend_gate) |x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x);
        };
        if (v.mh.ctx != null) _ = mlx.mlx_array_free(v.mh);
        self.gpa.free(v.comp_used);
        self.gpa.free(v.idx_used);
        self.gpa.free(v.pend_kv);
        self.gpa.free(v.pend_gate);
        self.verify = null;
    }
};

pub fn arrayBytes(a: A) u64 {
    if (a.ctx == null) return 0;
    return @as(u64, mlx.mlx_array_size(a)) * @as(u64, mlx.mlx_array_itemsize(a));
}

/// The state's resident bytes as allocated: every layer's window ring, compressed and index rows
/// (their capacity) and open pooling group.
pub fn stateBytes(st: *const State) u64 {
    var bytes: u64 = 0;
    for (st.layers) |*l| {
        bytes += arrayBytes(l.win.buf.array) + arrayBytes(l.pend_kv) + arrayBytes(l.pend_gate);
        if (l.comp) |r| bytes += arrayBytes(r.buf.array) + arrayBytes(r.sbuf.array);
        if (l.idx_k) |r| bytes += arrayBytes(r.buf.array) + arrayBytes(r.sbuf.array);
    }
    for (st.ds) |r| bytes += arrayBytes(r.buf.array);
    return bytes;
}

/// The first `n` rows of `x` as a view.
pub fn headRows(x: A, n: usize, s: S) !A {
    if (x.ctx == null or n == 0) return none;
    const sh = mlx.getShape(x);
    var head = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(head);
    try mlx.check(mlx.mlx_slice(&head, x, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ ci(n), sh[1] }, 2, &[_]c_int{ 1, 1 }, 2, s));
    return head;
}

pub fn rowsOf(x: A) usize {
    return if (x.ctx != null) dim(x, 0) else 0;
}

/// The bytes of rows [0, n) of `x`.
pub fn rowBytes(x: A, n: usize) usize {
    if (x.ctx == null or n == 0) return 0;
    return n * (mlx.mlx_array_size(x) / dim(x, 0)) * mlx.mlx_array_itemsize(x);
}

/// An owned copy of rows [0, n) that keeps every bit (a float add of zero would turn -0.0 into +0.0).
pub fn ownedRows(x: A, n: usize, s: S) !A {
    if (x.ctx == null or n == 0) return none;
    const head = try headRows(x, n, s);
    defer _ = mlx.mlx_array_free(head);
    const dt = mlx.mlx_array_dtype(x);
    const raw_dt: mlx.mlx_dtype = switch (dt) {
        .bfloat16, .float16 => .uint16,
        .float32 => .uint32,
        else => dt,
    };
    var raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(raw);
    try mlx.check(mlx.mlx_view(&raw, head, raw_dt, s));
    var copy = mlx.mlx_array_new();
    {
        errdefer _ = mlx.mlx_array_free(copy);
        if (mlx.engine.isStream(s)) {
            // The engine copies into one buffer of the pool; MLX's own copy shares, so there an add of zero makes one.
            try mlx.check(mlx.engine.copyOwned(&copy, raw));
        } else {
            var zero = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(zero);
            try mlx.check(mlx.mlx_zeros(&zero, &[_]c_int{1}, 1, raw_dt, s));
            try mlx.check(mlx.mlx_add(&copy, raw, zero, s));
        }
    }
    if (raw_dt == dt) return copy;
    defer _ = mlx.mlx_array_free(copy);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_view(&out, copy, dt, s));
    return out;
}

/// A copy of `src` in a buffer of its own, or null.
pub fn copyOf(src: ?A, s: S) !?A {
    const a = src orelse return null;
    if (a.ctx == null) return null;
    var c = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(c);
    try mlx.check(if (mlx.engine.isOurs(a)) mlx.engine.copyOwned(&c, a) else mlx.mlx_copy(&c, a, s));
    return c;
}

/// `value` written over the fixed block `buf` from its first element, in place; the block made zeros of `full` (and
/// `value`'s dtype) when it has none or another: it is the state's own, so the forward that writes it leaves nothing of
/// its own behind (docs/bill_redesign.md part 8).
pub fn writeFixed(buf: *?A, value: A, full: []const c_int, s: S) !void {
    const dt = mlx.mlx_array_dtype(value);
    if (buf.*) |b| if (mlx.mlx_array_dtype(b) != dt or !std.mem.eql(c_int, mlx.getShape(b), full)) {
        _ = mlx.mlx_array_free(b);
        buf.* = null;
    };
    if (buf.* == null) buf.* = (try mlx_pages.ringBuffer(full, dt, s)).array;
    const sh = mlx.getShape(value);
    const start: [4]c_int = @splat(0);
    const next = blk: {
        var up = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(up);
        break :blk try mlx_pages.InPlace.adopt(buf.*.?, .ring).write(&up, value, start[0..sh.len], sh, s);
    };
    _ = mlx.mlx_array_free(buf.*.?);
    buf.* = next.array;
    // The block holds the write from here: one that fails to land is still the block's to release.
    try next.landed(s);
}

/// Copies `value` into the first `size(value)` elements of fixed 2-D owner `buf` (`[cap_elems, 1]` of `value`'s
/// dtype) and returns a zero-copy view reshaped to `value`'s exact shape. Callers must free any prior view referencing
/// `buf.*` before calling so the owner stays sole.
pub fn storeFixed(buf: *?A, value: A, cap_elems: c_int, s: S) !A {
    const sh = mlx.getShape(value);
    const n: c_int = @intCast(mlx.mlx_array_size(value));
    var cont = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cont);
    try mlx.check(mlx.mlx_contiguous(&cont, value, false, s));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, cont, &[_]c_int{ n, 1 }, 2, s));
    try writeFixed(buf, flat, &.{ cap_elems, 1 }, s);
    const head = try headRows(buf.*.?, @intCast(n), s);
    if (sh.len == 2 and sh[1] == 1) return head;
    defer _ = mlx.mlx_array_free(head);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, head, sh.ptr, sh.len, s));
    return out;
}

/// Picks a sole slot from a 2-slot logits ring (`blocks`), preferring `turn_ptr.*` and falling back to `turn_ptr.* ^ 1`
/// when `turn_ptr.*` is still referenced by a caller view (e.g. after `queuedHead` is discarded on a last-layer
/// speculative rollback). Advances `turn_ptr.*` past the chosen slot.
pub fn pickSolePair(blocks: *[2]?A, turn_ptr: *u1, s: S) !u1 {
    const first = turn_ptr.*;
    if (mlx.engine.isStream(s)) {
        for ([2]u1{ first, first ^ 1 }) |idx| {
            const b = blocks[idx] orelse {
                turn_ptr.* = idx ^ 1;
                return idx;
            };
            if (!mlx.engine.isOurs(b) or try mlx.engine.sole(b)) {
                turn_ptr.* = idx ^ 1;
                return idx;
            }
        }
        if (@import("../tensor/plan/plan_api.zig").planned()) {
            return error.Dsv41LogitsBlocksHeld;
        }
    }
    turn_ptr.* = first ^ 1;
    return first;
}

/// One request's logits block (`blocks[turn]`) written with rows `[start, stop)` of `joined` (`[rows, vocab]`) and
/// handed back as `[1, stop - start, vocab]` view.
pub fn writeGroupBlock(blocks: *[2]?A, turn_ptr: *u1, joined: A, start: c_int, stop: c_int, rows: c_int, s: S) !A {
    const vocab: c_int = mlx.getShape(joined)[1];
    var row = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(row);
    try mlx.check(mlx.mlx_slice(&row, joined, &[_]c_int{ start, 0 }, 2, &[_]c_int{ stop, vocab }, 2, &[_]c_int{ 1, 1 }, 2, s));
    const turn = try pickSolePair(blocks, turn_ptr, s);
    const block = &blocks[turn];
    try writeFixed(block, row, &.{ rows, vocab }, s);
    var view = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(view);
    try mlx.check(mlx.mlx_slice(&view, block.*.?, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ stop - start, vocab }, 2, &[_]c_int{ 1, 1 }, 2, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const shape = [3]c_int{ 1, stop - start, vocab };
    try mlx.check(mlx.mlx_reshape(&out, view, &shape, 3, s));
    return out;
}

/// A slot array the checkpoint holds; every real one is 2-D, and an absent one is stored as a one-byte placeholder.
pub fn present(a: A) bool {
    return a.ctx != null and mlx.mlx_array_ndim(a) == 2 and mlx.mlx_array_size(a) > 0;
}

/// `shareInto` for a buffer later rows are written into: its record is read off `src`, a restore's own exact copy.
pub fn shareBuf(dst: *mlx_pages.InPlace, src: A, grid: mlx_pages.Grid) !void {
    dst.deinit();
    if (!present(src)) return;
    var copy = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(copy);
    try mlx.check(mlx.mlx_array_set(&copy, src));
    dst.* = mlx_pages.InPlace.adopt(copy, grid);
}

pub fn shareInto(dst: *A, src: A) !void {
    if (dst.ctx != null) _ = mlx.mlx_array_free(dst.*);
    dst.* = none;
    if (!present(src)) return;
    dst.* = mlx.mlx_array_new();
    errdefer {
        _ = mlx.mlx_array_free(dst.*);
        dst.* = none;
    }
    try mlx.check(mlx.mlx_array_set(dst, src));
}

/// Tokens per row of layer `li`'s compressed rows and of its index keys; 0 where the layer keeps none. A restore
/// point at position `n` holds `n / ratio` rows of each.
pub fn rowRatios(c: *const cfg_mod.Config, li: usize) [2]u32 {
    const r: u32 = c.ratios[li];
    return .{ if (c.isKvSource(li)) r else 0, if (c.isOwner(li)) r else 0 };
}

/// `slotCount` for a config.
pub fn slotCountFor(c: *const cfg_mod.Config, dspark: bool) usize {
    return 2 * c.layers + (if (dspark) c.n_mtp else 0);
}

/// Ensures `st` has every fixed owner (`pend_kv_buf`, `pend_gate_buf`, `verify_*`, `Points` slots) and row stores
/// sized for a request reaching `reach` tokens. Enters `widePhase()` on Cooler so persistent owner allocations pass a
/// closed pool cleanly.
pub fn reserveState(st: *State, c: *const cfg_mod.Config, act: mlx.mlx_dtype, has_dspark: bool, on_engine: bool, joined: bool, reach: usize, s: S) !void {
    st.reach = reach;
    const owned = mlx.engine.widePhase();
    defer owned.end();
    for (st.layers, 0..) |*ls, li| {
        const goal: usize = @intCast(c.storeRows(li, reach));
        if (ls.comp) |*rows| {
            if (on_engine) try rows.reserveFor(goal, s) else try rows.bind(goal, s);
        }
        if (ls.idx_k) |*rows| {
            if (on_engine) try rows.reserveFor(goal, s) else try rows.bind(goal, s);
        }
    }
    if (on_engine) try ensureFixedOwners(st, c, act, has_dspark, joined, s);
}

pub fn ensureFixedOwners(st: *State, c: *const cfg_mod.Config, act: mlx.mlx_dtype, has_dspark: bool, joined: bool, s: S) !void {
    const owned = mlx.engine.widePhase();
    defer owned.end();
    for (st.layers, 0..) |*ls, li| {
        const r = c.ratios[li];
        if (r > 1) {
            if (ls.pend_kv_buf == null) ls.pend_kv_buf = (try mlx_pages.ringBuffer(&.{ ci(r - 1), ci(c.head_dim) }, .float32, s)).array;
            if (ls.pend_gate_buf == null) ls.pend_gate_buf = (try mlx_pages.ringBuffer(&.{ ci(r - 1), ci(c.head_dim) }, .float32, s)).array;
            if (has_dspark) {
                if (st.verify_pend_kv[li] == null) st.verify_pend_kv[li] = (try mlx_pages.ringBuffer(&.{ ci(r - 1 + cfg_mod.ring_slack), ci(c.head_dim) }, .float32, s)).array;
                if (st.verify_pend_gate[li] == null) st.verify_pend_gate[li] = (try mlx_pages.ringBuffer(&.{ ci(r - 1 + cfg_mod.ring_slack), ci(c.head_dim) }, .float32, s)).array;
            }
        }
    }
    if (has_dspark and st.verify_mh == null) {
        st.verify_mh = (try mlx_pages.ringBuffer(&.{ ci(cfg_mod.ring_slack), ci(c.dspark.n_targets * c.dim) }, act, s)).array;
    }
    if (st.spec_undo.win.cap == 0) {
        const win_cap = c.window + cfg_mod.ring_slack;
        st.spec_undo.win = try Ring.init(win_cap, c.head_dim, act, s);
        var max_r: usize = 1;
        for (0..c.layers) |li| max_r = @max(max_r, c.ratios[li]);
        st.spec_undo.max_ratio = max_r;
        if (max_r > 1) {
            st.spec_undo.pend_kv_buf = (try mlx_pages.ringBuffer(&.{ ci(max_r - 1), ci(c.head_dim) }, .float32, s)).array;
            st.spec_undo.pend_gate_buf = (try mlx_pages.ringBuffer(&.{ ci(max_r - 1), ci(c.head_dim) }, .float32, s)).array;
            if (has_dspark) {
                st.spec_undo.verify_kv_buf = (try mlx_pages.ringBuffer(&.{ ci(max_r - 1 + cfg_mod.ring_slack), ci(c.head_dim) }, .float32, s)).array;
                st.spec_undo.verify_gate_buf = (try mlx_pages.ringBuffer(&.{ ci(max_r - 1 + cfg_mod.ring_slack), ci(c.head_dim) }, .float32, s)).array;
            }
        }
    }
    if (st.points == null) {
        const pts = try st.gpa.create(Points);
        pts.* = .{ .cap = if (joined) joined_point_slots else exclusive_point_slots };
        errdefer {
            pts.deinit(c.layers, st.ds.len);
            st.gpa.destroy(pts);
        }
        const slack: usize = if (has_dspark) cfg_mod.ring_slack else 0;
        for (pts.slots[0..pts.cap]) |*slot| {
            for (0..c.layers) |li| {
                slot.layers[li][0] = (try mlx_pages.ringBuffer(&.{ ci(c.window + slack), ci(c.head_dim) }, act, s)).array;
                const r = c.ratios[li];
                if (r > 1) {
                    slot.layers[li][1] = (try mlx_pages.ringBuffer(&.{ ci(r - 1), ci(c.head_dim) }, .float32, s)).array;
                    slot.layers[li][2] = (try mlx_pages.ringBuffer(&.{ ci(r - 1), ci(c.head_dim) }, .float32, s)).array;
                }
            }
            for (st.ds, 0..) |_, si| {
                slot.ds[si] = (try mlx_pages.ringBuffer(&.{ ci(c.window), ci(c.head_dim) }, act, s)).array;
            }
        }
        st.points = pts;
    }
}

/// Copies the small state (`win`, `pend_kv`, `pend_gate`, `ds`) of `st` into an unused `Points` slot on `st` and
/// returns refcounted handles in `out` (`slotCountFor` slots, `nil_slot`-initialized by caller).
pub fn captureFixedSlot(st: *State, c: *const cfg_mod.Config, s: S, out: []Slot) !void {
    const pts = st.points orelse return error.Dsv41NoPointSlots;
    const slot_idx = try pts.allocSlot(c.layers, st.ds.len);
    const slot = &pts.slots[slot_idx];
    errdefer for (out) |*sl| for (sl) |*x| {
        if (x.ctx != null) _ = mlx.mlx_array_free(x.*);
        x.* = none;
    };
    for (st.layers, 0..) |*ls, li| {
        const dst_a = &out[2 * li];
        try writeFixed(&slot.layers[li][0], ls.win.buf.array, &.{ ci(ls.win.cap), ci(ls.win.width) }, s);
        try shareInto(&dst_a[0], slot.layers[li][0].?);
        const r = c.ratios[li];
        if (r > 1) {
            const pend_n = rowsOf(ls.pend_kv);
            if (pend_n > 0) {
                try writeFixed(&slot.layers[li][1], ls.pend_kv, &.{ ci(r - 1), ci(c.head_dim) }, s);
                dst_a[1] = try headRows(slot.layers[li][1].?, pend_n, s);
            }
            const gate_n = rowsOf(ls.pend_gate);
            if (gate_n > 0) {
                try writeFixed(&slot.layers[li][2], ls.pend_gate, &.{ ci(r - 1), ci(c.head_dim) }, s);
                dst_a[2] = try headRows(slot.layers[li][2].?, gate_n, s);
            }
        }
    }
    for (st.ds, 0..) |*ring, si| {
        try writeFixed(&slot.ds[si], ring.buf.array, &.{ ci(ring.cap), ci(ring.width) }, s);
        try shareInto(&out[2 * c.layers + si][0], slot.ds[si].?);
    }
}

/// Copies `slots` (an SSM checkpoint's small arrays `[win, pend_kv, pend_gate, ds]`) into an unused `Points` slot on
/// `st`, returning refcounted views into that slot in `out` so the source arrays can be released immediately.
pub fn copyCheckpointIntoFixedSlot(st: *State, c: *const cfg_mod.Config, src_slots: []const Slot, s: S, out: []Slot) !void {
    const pts = st.points orelse return error.Dsv41NoPointSlots;
    const slot_idx = try pts.allocSlot(c.layers, st.ds.len);
    const slot = &pts.slots[slot_idx];
    errdefer for (out) |*sl| for (sl) |*x| {
        if (x.ctx != null) _ = mlx.mlx_array_free(x.*);
        x.* = none;
    };
    for (0..c.layers) |li| {
        const src_a = src_slots[2 * li];
        const dst_a = &out[2 * li];
        const w = c.window + (if (st.ds.len > 0) cfg_mod.ring_slack else 0);
        try writeFixed(&slot.layers[li][0], src_a[0], &.{ ci(w), ci(c.head_dim) }, s);
        try shareInto(&dst_a[0], slot.layers[li][0].?);
        const r = c.ratios[li];
        if (r > 1) {
            const pend_n = if (present(src_a[1])) rowsOf(src_a[1]) else 0;
            if (pend_n > 0) {
                try writeFixed(&slot.layers[li][1], src_a[1], &.{ ci(r - 1), ci(c.head_dim) }, s);
                dst_a[1] = try headRows(slot.layers[li][1].?, pend_n, s);
            }
            const gate_n = if (present(src_a[2])) rowsOf(src_a[2]) else 0;
            if (gate_n > 0) {
                try writeFixed(&slot.layers[li][2], src_a[2], &.{ ci(r - 1), ci(c.head_dim) }, s);
                dst_a[2] = try headRows(slot.layers[li][2].?, gate_n, s);
            }
        }
    }
    for (0..st.ds.len) |si| {
        const src_d = src_slots[2 * c.layers + si][0];
        try writeFixed(&slot.ds[si], src_d, &.{ ci(c.window), ci(c.head_dim) }, s);
        try shareInto(&out[2 * c.layers + si][0], slot.ds[si].?);
    }
}

/// Empties `st` in place for `restoreBegin`: keeps all ring/row/point/fixed allocations so a subsequent
/// `DiskTier.restorePointInto` writes into `st`'s own buffers with zero pool asks.
pub fn emptyForRestore(st: *State, c: *const cfg_mod.Config, act: mlx.mlx_dtype, has_dspark: bool, on_engine: bool, joined: bool, pos: usize, s: S) !void {
    st.dropVerify();
    st.n = 0;
    st.owner = 0;
    st.main_lazy = false;
    st.bound_fresh = false;
    st.hist.clearRetainingCapacity();
    for (st.layers) |*ls| {
        if (ls.comp) |*r| r.truncate(0);
        if (ls.idx_k) |*r| r.truncate(0);
        if (ls.pend_kv.ctx != null) _ = mlx.mlx_array_free(ls.pend_kv);
        if (ls.pend_gate.ctx != null) _ = mlx.mlx_array_free(ls.pend_gate);
        ls.pend_kv = none;
        ls.pend_gate = none;
    }
    const need_reach = @max(pos, st.reach);
    try reserveState(st, c, act, has_dspark, on_engine, joined, need_reach, s);
    // Replace any shared live rings so hostWritable writes into sole buffers.
    const owned = mlx.engine.widePhase();
    defer owned.end();
    for (st.layers) |*ls| {
        if (!try mlx.engine.sole(ls.win.buf.array)) {
            const fresh = try mlx_pages.ringBuffer(&.{ ci(ls.win.cap), ci(ls.win.width) }, ls.win.dt, s);
            ls.win.deinit();
            ls.win.buf = fresh;
        }
    }
    for (st.ds) |*ring| {
        if (!try mlx.engine.sole(ring.buf.array)) {
            const fresh = try mlx_pages.ringBuffer(&.{ ci(ring.cap), ci(ring.width) }, ring.dt, s);
            ring.deinit();
            ring.buf = fresh;
        }
    }
}

/// Target slice inside `st` for `DiskTier.restorePointInto` at snapshot layer `snap_li`, `field` (`0` conv / `1` ssm),
/// starting at `row`, with element `dtype` and `shape`. Zero pool asks.
pub fn restoreDest(st: *State, c: *const cfg_mod.Config, snap_li: usize, field: u1, row: usize, dtype: mlx.mlx_dtype, shape: []const c_int) ![]u8 {
    if (shape.len != 2 or snap_li >= 2 * (2 * c.layers + st.ds.len)) return error.Dsv41CheckpointMismatch;
    const slot_idx = snap_li / 2;
    const half = snap_li % 2;
    const k = 2 * half + @as(usize, field);
    var block: ?A = null;
    var whole = false;
    var exact_rows = false;
    if (slot_idx < 2 * c.layers) {
        const li = slot_idx / 2;
        const ls = &st.layers[li];
        if (slot_idx % 2 == 0) {
            whole = true;
            switch (k) {
                0 => {
                    block = ls.win.buf.array;
                    exact_rows = true;
                },
                1 => block = ls.pend_kv_buf,
                2 => block = ls.pend_gate_buf,
                else => return error.Dsv41CheckpointMismatch,
            }
        } else {
            switch (k) {
                0 => block = if (ls.comp) |r| r.buf.array else null,
                1 => block = if (ls.comp) |r| r.sbuf.array else null,
                2 => block = if (ls.idx_k) |r| r.buf.array else null,
                3 => block = if (ls.idx_k) |r| r.sbuf.array else null,
                else => unreachable,
            }
        }
    } else {
        const si = slot_idx - 2 * c.layers;
        if (k != 0) return error.Dsv41CheckpointMismatch;
        whole = true;
        exact_rows = true;
        block = st.ds[si].buf.array;
    }
    const b = block orelse return error.Dsv41CheckpointMismatch;
    const bs = mlx.getShape(b);
    const n: usize = @intCast(shape[0]);
    if (mlx.mlx_array_dtype(b) != dtype or bs.len != 2 or bs[1] != shape[1]) return error.Dsv41CheckpointMismatch;
    if ((whole and row != 0) or (exact_rows and n != @as(usize, @intCast(bs[0]))) or row + n > @as(usize, @intCast(bs[0]))) return error.Dsv41CheckpointMismatch;
    const bytes = try mlx.engine.hostWritable(b);
    const row_bytes = bytes.len / @as(usize, @intCast(bs[0]));
    return bytes[row * row_bytes ..][0 .. n * row_bytes];
}

/// Validates every tensor field written by `DiskTier.restorePointInto` and advances `st.n = pos`, populating `st.hist`
/// and setting open pooling group views (`pend_kv`/`pend_gate`) + row store `.used` counts.
pub fn restoreEnd(st: *State, c: *const cfg_mod.Config, has_hash: bool, compress_id_ctx: anytype, comptime compressFn: fn (@TypeOf(compress_id_ctx), u32) u32, pos: usize, ids: []const u32, counts: []const [2]usize, s: S) !void {
    const n_slots = 2 * c.layers + st.ds.len;
    if (ids.len < pos or counts.len != 2 * n_slots) return error.Dsv41CheckpointMismatch;
    for (st.layers, 0..) |*ls, li| {
        const a0 = counts[4 * li];
        const a1 = counts[4 * li + 1];
        const r = c.ratios[li];
        const open_want: usize = if (ls.comp != null and r > 1) pos % r else 0;
        if (a0[0] != ls.win.cap or a0[1] != open_want or a1[0] != open_want or a1[1] != 0) return error.Dsv41CheckpointMismatch;
        if (open_want > 0) {
            ls.pend_kv = try headRows(ls.pend_kv_buf.?, open_want, s);
            ls.pend_gate = try headRows(ls.pend_gate_buf.?, open_want, s);
        }
        const b0 = counts[4 * li + 2];
        const b1 = counts[4 * li + 3];
        const want_rows = if (r > 0) pos / r else 0;
        if (ls.comp) |*rows| {
            const want_s = if (rows.pack != null) want_rows else 0;
            if (b0[0] != want_rows or b0[1] != want_s) return error.Dsv41CheckpointMismatch;
            rows.used = want_rows;
        } else if (b0[0] != 0 or b0[1] != 0) return error.Dsv41CheckpointMismatch;
        if (ls.idx_k) |*rows| {
            const want_s = if (rows.pack != null) want_rows else 0;
            if (b1[0] != want_rows or b1[1] != want_s) return error.Dsv41CheckpointMismatch;
            rows.used = want_rows;
        } else if (b1[0] != 0 or b1[1] != 0) return error.Dsv41CheckpointMismatch;
    }
    for (st.ds, 0..) |*ring, si| {
        const d0 = counts[2 * (2 * c.layers + si)];
        const d1 = counts[2 * (2 * c.layers + si) + 1];
        if (d0[0] != ring.cap or d0[1] != 0 or d1[0] != 0 or d1[1] != 0) return error.Dsv41CheckpointMismatch;
    }
    st.n = pos;
    if (has_hash) {
        try st.hist.ensureTotalCapacity(st.gpa, pos);
        for (ids[0..pos]) |t| st.hist.appendAssumeCapacity(compressFn(compress_id_ctx, t));
    }
}

fn freeOpt(buf: *?A) void {
    if (buf.*) |a| {
        _ = mlx.mlx_array_free(a);
        buf.* = null;
    }
}

fn optBytes(buf: ?A) u64 {
    return if (buf) |a| arrayBytes(a) else 0;
}

/// Fixed owners for one speculative unresolved MoE layer (`Unresolved`) across all narrow widths.
pub const UnresFixed = struct {
    h: ?A = null,
    idx: ?A = null,
    wts: ?A = null,
    x1: ?A = null,
    post: ?A = null,
    comb: ?A = null,
    pre: ?A = null,
    pred: ?A = null,
    kept_down: ?A = null,
    kept_shared: ?A = null,
    kept_shared_f32: ?A = null,

    pub fn deinit(self: *UnresFixed) void {
        inline for (.{ "h", "idx", "wts", "x1", "post", "comb", "pre", "pred", "kept_down", "kept_shared", "kept_shared_f32" }) |f| {
            freeOpt(&@field(self, f));
        }
    }

    pub fn bytes(self: *const UnresFixed) u64 {
        var total: u64 = 0;
        inline for (.{ "h", "idx", "wts", "x1", "post", "comb", "pre", "pred", "kept_down", "kept_shared", "kept_shared_f32" }) |f| {
            total += optBytes(@field(self, f));
        }
        return total;
    }
};

/// Fixed Cooler owners for every cross-layer narrow tensor (`Pass.x`, `Pass.cpre`, `Pass.targets`,
/// `Pass.engram_rows`, `Unresolved`, and `Shared.{topk,uids,urows,cand}`) so narrow frames export zero arena
/// allocations across layers (`docs/bill_redesign.md`).
pub const PassFixed = struct {
    x: [2]?A = .{ null, null },
    cpre: [2]?A = .{ null, null },
    targets: [cfg_mod.max_targets]?A = @splat(null),
    engram_rows: [cfg_mod.max_engram]?A = @splat(null),
    unres: [2]UnresFixed = .{ .{}, .{} },
    topk: [narrow_cap]?A = @splat(null),
    uids: [narrow_cap]?A = @splat(null),
    urows: [narrow_cap]?A = @splat(null),
    cand_i32: [narrow_cap]?A = @splat(null),
    cand_bool: [narrow_cap]?A = @splat(null),
    x_cap: c_int = 0,
    cpre_cap: c_int = 0,
    target_cap: c_int = 0,
    engram_cap: c_int = 0,
    unres_h_cap: c_int = 0,
    unres_idx_cap: c_int = 0,
    unres_wts_cap: c_int = 0,
    unres_x1_cap: c_int = 0,
    unres_post_cap: c_int = 0,
    unres_comb_cap: c_int = 0,
    unres_pre_cap: c_int = 0,
    unres_pred_cap: c_int = 0,
    unres_kept_down_cap: c_int = 0,
    unres_kept_shared_cap: c_int = 0,
    unres_kept_shared_f32_cap: c_int = 0,
    topk_cap: c_int = 0,
    urows_cap: [narrow_cap]c_int = @splat(0),
    cand_i32_cap: c_int = 0,
    cand_bool_cap: c_int = 0,

    pub fn init(c: *const cfg_mod.Config, act: mlx.mlx_dtype, has_dspark: bool, has_spec: bool, has_engram: bool, engram_cols: usize, serves_rows: bool, reach: usize, s: S) !PassFixed {
        const owned = mlx.engine.widePhase();
        defer owned.end();
        var self: PassFixed = .{};
        errdefer self.deinit();

        self.x_cap = ci(narrow_cap * c.hc_mult * c.dim);
        self.cpre_cap = ci(narrow_cap * c.hc_mult);
        for (&self.x) |*b| b.* = (try mlx_pages.ringBuffer(&.{ self.x_cap, 1 }, act, s)).array;
        for (&self.cpre) |*b| b.* = (try mlx_pages.ringBuffer(&.{ self.cpre_cap, 1 }, .float32, s)).array;

        if (has_dspark) {
            self.target_cap = ci(narrow_cap * c.dim);
            for (0..c.dspark.n_targets) |t| {
                self.targets[t] = (try mlx_pages.ringBuffer(&.{ self.target_cap, 1 }, act, s)).array;
            }
        }
        if (has_engram and engram_cols > 0) {
            self.engram_cap = ci(narrow_cap * engram_cols * c.engram.head_dim);
            for (0..c.engram.n) |ei| {
                self.engram_rows[ei] = (try mlx_pages.ringBuffer(&.{ self.engram_cap, 1 }, act, s)).array;
            }
        }
        if (has_spec) {
            self.unres_h_cap = ci(narrow_cap * c.dim);
            self.unres_idx_cap = ci(narrow_cap * c.top_k);
            self.unres_wts_cap = ci(narrow_cap * c.top_k);
            self.unres_x1_cap = ci(narrow_cap * c.hc_mult * c.dim);
            self.unres_post_cap = ci(narrow_cap * c.hc_mult);
            self.unres_comb_cap = ci(narrow_cap * c.hc_mult * c.hc_mult);
            self.unres_pre_cap = ci(narrow_cap * c.hc_mult);
            self.unres_pred_cap = ci(narrow_cap * c.experts);
            self.unres_kept_down_cap = ci(narrow_cap * c.top_k * c.dim);
            self.unres_kept_shared_cap = ci(narrow_cap * c.dim);
            const nax_rows = @min(narrow_cap, @as(usize, @intCast(exl3.project_rows.NAX_ROWS_MAX)));
            const shared_nsplit: usize = @intCast(exl3.rate_groups.sharedDownNsplit(ci(c.moe_inter)));
            self.unres_kept_shared_f32_cap = ci(nax_rows * shared_nsplit * c.dim);
            for (&self.unres) |*u| {
                u.h = (try mlx_pages.ringBuffer(&.{ self.unres_h_cap, 1 }, act, s)).array;
                u.idx = (try mlx_pages.ringBuffer(&.{ self.unres_idx_cap, 1 }, .int32, s)).array;
                u.wts = (try mlx_pages.ringBuffer(&.{ self.unres_wts_cap, 1 }, .float32, s)).array;
                u.x1 = (try mlx_pages.ringBuffer(&.{ self.unres_x1_cap, 1 }, act, s)).array;
                u.post = (try mlx_pages.ringBuffer(&.{ self.unres_post_cap, 1 }, .float32, s)).array;
                u.comb = (try mlx_pages.ringBuffer(&.{ self.unres_comb_cap, 1 }, .float32, s)).array;
                u.pre = (try mlx_pages.ringBuffer(&.{ self.unres_pre_cap, 1 }, .float32, s)).array;
                u.pred = (try mlx_pages.ringBuffer(&.{ self.unres_pred_cap, 1 }, .float32, s)).array;
                if (c.expert_dec != null) {
                    u.kept_down = (try mlx_pages.ringBuffer(&.{ self.unres_kept_down_cap, 1 }, .float16, s)).array;
                    u.kept_shared = (try mlx_pages.ringBuffer(&.{ self.unres_kept_shared_cap, 1 }, .float16, s)).array;
                    if (exl3.kernels.gemmNaxOn()) {
                        u.kept_shared_f32 = (try mlx_pages.ringBuffer(&.{ self.unres_kept_shared_f32_cap, 1 }, .float32, s)).array;
                    }
                }
            }
        }

        const n_slots: usize = if (serves_rows) narrow_cap else 1;
        if (c.index_topk > 0) {
            self.topk_cap = ci(narrow_cap * c.index_topk);
            for (0..n_slots) |slot| {
                const row_mul: usize = if (slot == 0) narrow_cap else 1;
                const ur_cap = ci(row_mul * c.index_topk * c.head_dim);
                self.urows_cap[slot] = ur_cap;
                self.topk[slot] = (try mlx_pages.ringBuffer(&.{ self.topk_cap, 1 }, .int32, s)).array;
                self.uids[slot] = (try mlx_pages.ringBuffer(&.{ self.topk_cap, 1 }, .int32, s)).array;
                self.urows[slot] = (try mlx_pages.ringBuffer(&.{ ur_cap, 1 }, act, s)).array;
            }
        }
        if (c.cand_source >= 0 and c.cand_blocks > 0 and c.cand_block_size > 0) {
            self.cand_i32_cap = ci(narrow_cap * c.cand_blocks);
            const cs: usize = @intCast(c.cand_source);
            const r_cs = @max(@as(usize, c.ratios[cs]), 1);
            const max_cand_blocks = @max((reach / r_cs + c.cand_block_size - 1) / c.cand_block_size, 1);
            self.cand_bool_cap = ci(narrow_cap * max_cand_blocks);
            for (0..n_slots) |slot| {
                self.cand_i32[slot] = (try mlx_pages.ringBuffer(&.{ self.cand_i32_cap, 1 }, .int32, s)).array;
                self.cand_bool[slot] = (try mlx_pages.ringBuffer(&.{ self.cand_bool_cap, 1 }, .bool_, s)).array;
            }
        }
        return self;
    }

    pub fn deinit(self: *PassFixed) void {
        for (&self.x) |*b| freeOpt(b);
        for (&self.cpre) |*b| freeOpt(b);
        for (&self.targets) |*b| freeOpt(b);
        for (&self.engram_rows) |*b| freeOpt(b);
        for (&self.unres) |*u| u.deinit();
        for (&self.topk) |*b| freeOpt(b);
        for (&self.uids) |*b| freeOpt(b);
        for (&self.urows) |*b| freeOpt(b);
        for (&self.cand_i32) |*b| freeOpt(b);
        for (&self.cand_bool) |*b| freeOpt(b);
    }

    pub fn bytes(self: *const PassFixed) u64 {
        var total: u64 = 0;
        for (self.x) |b| total += optBytes(b);
        for (self.cpre) |b| total += optBytes(b);
        for (self.targets) |b| total += optBytes(b);
        for (self.engram_rows) |b| total += optBytes(b);
        for (self.unres) |*u| total += u.bytes();
        for (self.topk) |b| total += optBytes(b);
        for (self.uids) |b| total += optBytes(b);
        for (self.urows) |b| total += optBytes(b);
        for (self.cand_i32) |b| total += optBytes(b);
        for (self.cand_bool) |b| total += optBytes(b);
        return total;
    }
};
