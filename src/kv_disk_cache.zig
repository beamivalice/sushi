//! SSD tier for the hot prefix cache — chunked KV persistence.
//!
//! Committed KV prefixes are persisted to disk as position-chunked
//! safetensors, so previously-seen prefixes survive server restarts and RAM
//! evictions and are RESTORED instead of recomputed. Two-tier flow:
//!
//!   commit  → RAM entry (refcount snapshot, unchanged) + chunk-APPEND to
//!             disk. Only chunks not yet on disk are written — a multi-turn
//!             agent session pays one bounded partial-chunk rewrite + the new
//!             tail per turn, never a full re-serialize.
//!   lookup  → longest-prefix match across RAM entries as before; the disk
//!             index is consulted when it can beat the RAM match by at least
//!             one chunk (fresh boot, post-eviction). The entry is rebuilt
//!             into the live cache and the normal truncate-then-prefill path
//!             continues.
//!
//! Layout (one root per model fingerprint — path + config.json identity):
//!   <base>/<fingerprint>/e<id>/meta.json    commit point (written tmp+rename
//!                                           LAST; an entry without it is a
//!                                           crash leftover and is GC'd)
//!   <base>/<fingerprint>/e<id>/tokens.bin   LE u32 token ids (prompt ++ gen)
//!   <base>/<fingerprint>/e<id>/c000000.safetensors   KV positions [0, chunk_tokens)
//!   <base>/<fingerprint>/e<id>/c000001.safetensors   ...
//!
//! Chunk files hold per-layer K/V slices keyed "l{i}.k"/"l{i}.v" (plus
//! ".ks/.kb/.vs/.vb" scale/bias triples in affine mode). The final chunk may
//! be partial; a commit that extends the entry rewrites ONLY that chunk and
//! appends new ones. A chunk file holding MORE positions than meta.json
//! claims (crash between chunk write and meta rename) is sliced down at
//! restore, never trusted.
//!
//! Phase 3 — hybrid SSM archs (GatedDeltaNet):
//! the RAM tier's per-position `SSMCheckpoint`s persist beside the chunks as
//!   <base>/<fingerprint>/e<id>/s0002048.safetensors   SSM state at pos 2048
//! keyed "l{i}.conv"/"l{i}.ssm" (absent key = null state — plain-attention
//! layers in the hybrid have neither), with the per-layer `initialized` flags in the safetensors
//! metadata map ("init"). Checkpoint files are immutable once written (keyed
//! by position); extend commits append only NEW positions, bounded per entry
//! by `SSM_DISK_MAX_PER_ENTRY` (evict-lowest — the newest positions are where
//! multi-turn warm requests match). A hybrid restore rebuilds the KV prefix
//! [0, cp_pos) AND the SSM state at cp_pos (`restoreIntoHybrid`) — mirroring
//! the RAM tier's rewind-both semantics.
//!
//! Ringed sliding layers hold a window, not a prefix, so the chunks carry only the other
//! layers; each ring restore point persists as
//!   <base>/<fingerprint>/e<id>/r0000700.safetensors   ringed layers' rows below pos 700
//! and a ringed entry restores only at one of them (`restoreIntoRinged`).
//!
//! Scope: B==1 slot caches. All array work runs on the inference thread;
//! native Zig safetensors reads land directly into Cooler Metal buffers or MLX arrays.

const std = @import("std");
const mlx = @import("mlx.zig");
const kv_quant = @import("kv_quant.zig");
const trunk = @import("trunk.zig");
const prefix_arch = @import("arch/prefix.zig");
const io_util = @import("io_util.zig");
const disk_writer = @import("kv_disk_writer.zig");
const write_stats = @import("disk_write_stats.zig");
const log = @import("log.zig");
const disk_placement = @import("disk_placement.zig");
const disk_format = @import("disk_format.zig");
const kv_disk_fs = @import("kv_disk_fs.zig");
const kv_disk_restore = @import("kv_disk_restore.zig");
const kv_disk_persist = @import("kv_disk_persist.zig");
const kv_disk_index = @import("kv_disk_index.zig");
const kv_disk_helpers_test = @import("kv_disk_helpers_test.zig");

/// Restoring from disk only happens when it beats the best RAM match by at
/// least this many tokens — a disk read + rebuild is only worth it when it
/// replaces a meaningful amount of prefill.
pub const MIN_DISK_ADVANTAGE_TOKENS: u32 = 256;

/// Entries shorter than this are never persisted (a short prefix re-prefills
/// in well under the restore cost).
pub const MIN_PERSIST_TOKENS: u32 = 512;

pub const DEFAULT_CHUNK_TOKENS: u32 = 1024;

/// Budget floor for a decline-spill's flush: the client is already gone, so
/// the only cost is the synchronous write itself (~2-4 s at SSD speeds); the
/// tier's byte budget and LRU eviction are the real bounds. Sized to bank a
/// 122k-token hybrid candidate's KV in one spill.
pub const DECLINE_SPILL_FLUSH_FLOOR: u64 = 4 * 1024 * 1024 * 1024;

/// Tokens between the restore points every arch keeps on SSD: the absolute multiples, beside each prompt's end.
pub const POINT_GRID: u32 = 8192;

/// Max persisted SSM checkpoint positions per entry: the whole grid to 1M tokens plus the prompt-end points.
/// Thinning past it is span-preserving (`trunk.positionDropIndex`): the lowest and the newest survive.
/// The tier's byte budget, not this count, bounds what a long conversation keeps (`DiskTier.pointBudget`).
pub const SSM_DISK_MAX_PER_ENTRY: usize = 160;

/// Rows per whole segment of a growing sidecar: the MTP history's rows go to `p<j>.safetensors` files of this many,
/// written once and kept while their bytes hold, and the sidecar keeps the rows past the last one.
pub const SIDE_SEG_SPAN: u32 = 8192;
pub const SideSeg = disk_format.SideSeg;

/// The cap every arch outside the long-context gate keeps.
pub const SSM_DISK_MAX_PER_ENTRY_LEGACY: usize = 8;

/// Ring restore points one entry keeps on disk (~16 MiB each at kv8), thinned (`thinRingPositions`).
pub const RING_DISK_MAX_PER_ENTRY: usize = 8;

/// How a tier stores a restore point whose arrays grow with the prompt (compressed rows). Every `span`
/// tokens of rows go once per entry into `g<n>.safetensors`; a point's own file keeps its fixed arrays and the rows
/// past its last whole span, so a restore reads the point and the first `pos / span` segments.
pub const GrownSpec = struct {
    span: u32,
    /// Tokens per row of each snapshot layer's arrays; 0 marks a fixed layer.
    divs: []const u32,
    /// Row bytes one token adds across the growing layers.
    row_bytes: u64,
};

pub const IndexEntry = disk_format.IndexEntry;

/// The most a tier keeps of its off-grid restore points in RAM, as their files' bytes, instead of writing them
/// (`DiskTier.hold_stride`; docs/engine-prefix-cache.md#held-points).
/// 250 MB (owner): the newest two points of the STQ 4bpw pack (2 x 115.7 MB), a prompt end and the newer point over it.
pub const HELD_POINTS_MAX_BYTES: u64 = 250_000_000;
/// The held points take at most the writer's permit over this: they are pageable host bytes beside the writer's
/// staged ones, and give way to a staged write first (docs/engine-memory-admission.md).
pub const HELD_POINTS_PERMIT_DIVISOR: u64 = 2;

/// One restore point kept in RAM as its file's bytes (`serializeSafetensors`), listed in entry `id`'s positions.
/// `spare`: a point a newer one of its conversation extends, kept so a regenerate of its turn, or a prompt that
/// re-renders the reply after it, restores from it; the first written when the held points pass the cap.
pub const HeldPoint = struct { id: u64, pos: u32, bytes: []u8, spare: bool = false };
pub const PointKey = struct { id: u64, pos: u32 };
pub const RingFile = disk_format.RingFile;

/// What a commit of a ringed cache adds: its sliding layers hold only a window, so the entry
/// persists restore points instead, at the persisted length off the cache's own ring and at
/// each checkpoint's `step`.
pub const RingCommit = struct {
    window: u32,
    cps: []const trunk.KVCacheSnapshot = &.{},
};

/// `bestRingMatch`'s result: the entry, the ring file, and the length it restores to.
pub const RingMatch = struct { idx: usize, pos: u32, at: u32 };

pub const SpecMeta = disk_format.SpecMeta;
pub const SpecHeadMeta = disk_format.SpecHeadMeta;

/// What `appendCommitWithSpec` reads to persist one spec snapshot — the same
/// snapshot-shaped parts the trunk flush takes, plus the base position.
pub const SpecCommit = struct {
    entries: []const trunk.KVCacheEntry,
    step: usize,
    config: kv_quant.KVQuantConfig,
    base_pos: usize,
    /// In-checkpoint MTP head: the QSA aux half, persisted alongside the KV.
    head_aux: ?*const trunk.SSMCacheEntrySnapshot = null,
    head_pos_base: c_int = 0,
    head_marks: []const trunk.QsaHeadMark = &.{},
};

/// What a commit actually achieved on disk. The old bool meant "nothing more to write", and
/// every silent skip returned it too, so `spillIdleEntries` dropped RAM copies that had no
/// disk copy at all. Only one of the three outcomes is a promise.
pub const PersistOutcome = enum {
    /// The tier holds the full prefix. The only value that may license discarding the RAM copy.
    persisted,
    /// Real bytes landed but the entry is not whole yet; the next commit resumes.
    partial,
    /// Nothing was written and nothing is promised.
    skipped,

    /// "Nothing more for the caller to write", the old bool's meaning.
    pub fn nothingPending(self: PersistOutcome) bool {
        return self != .partial;
    }
};

/// The KV extent one commit would persist, in tokens: clamped to the token record (the cache
/// runs 1-2 positions ahead on EOS turns), from the initialized layers' offset, not `step`
/// (0 on GDN hybrids). Shared by the commit and `holdsFullPrefix`.
pub fn anyInitialized(kv_entries: []const trunk.KVCacheEntry) bool {
    for (kv_entries) |*entry| if (entry.initialized) return true;
    return false;
}

pub fn persistTargetLen(
    kv_entries: []const trunk.KVCacheEntry,
    step: usize,
    tokens_len: usize,
) usize {
    var max_off: usize = 0;
    var any = false;
    for (kv_entries) |*entry| {
        if (!entry.initialized) continue;
        any = true;
        if (!entry.ringed and entry.offset > max_off) max_off = entry.offset;
    }
    // A cache whose state lives elsewhere (a request-held state) holds no rows to persist.
    if (!any) return 0;
    return @min(@max(step, max_off), tokens_len);
}

pub const SpecKind = enum { dflash, mtp };

/// What a lookup may share from each persisted entry, parallel to `DiskTier.entries`. The media
/// rule has ONE owner, `prefix_cache.crossKeyBoundary`; the tier stores the identity (v10) and
/// applies the ceilings that rule produces, so it never learns what a pixel is. An empty slice
/// leaves every entry unconstrained, which is what a text-only request wants.
pub const MediaLimits = struct {
    /// Null leaves every entry unconstrained, which is what a text-only request wants.
    rule: ?Rule = null,

    pub const UNBOUNDED: u32 = std.math.maxInt(u32);

    /// The caller's rule, closed over the request's own media chain.
    pub const Rule = struct {
        ctx: *const anyopaque,
        /// What a record holding media `(vision_key, media_start)` may share; null = nothing.
        limitFor: *const fn (ctx: *const anyopaque, vision_key: u64, media_start: ?u32) ?u32,
    };

    /// The tokens a record holding media `(vision_key, media_start)` may share with the request: 0 when none.
    pub fn ceilingFor(self: MediaLimits, vision_key: u64, media_start: ?u32) usize {
        const r = self.rule orelse return UNBOUNDED;
        return r.limitFor(r.ctx, vision_key, media_start) orelse 0;
    }

    /// `e`'s ceiling, or null when it may share nothing at all.
    pub fn at(self: MediaLimits, e: *const IndexEntry) ?u32 {
        const r = self.rule orelse return UNBOUNDED;
        return r.limitFor(r.ctx, e.vision_key, e.media_start);
    }
};

/// A request's prefix as an SSD restore left it: rows `[0, rows)` read from entry `id`'s chunk files.
pub const RestoredRows = struct { id: u64, rows: u32 };

pub const Match = struct {
    idx: usize,
    /// Shared-prefix length clamped to kv_len — the positions a restore can
    /// actually rebuild.
    usable: u32,
};

/// `bestHybridMatch`'s result: the winning entry, its usable prefix, and the
/// restorable checkpoint position (≤ usable) that won it the race.
pub const HybridMatch = struct { idx: usize, usable: u32, cp: u32 };

pub fn nbytesOf(a: mlx.mlx_array) u64 {
    return @as(u64, mlx.mlx_array_size(a)) * @as(u64, mlx.mlx_array_itemsize(a));
}

/// The header allowance per file and per tensor in `DiskTier.entryBytesAt`'s estimate.
pub const FILE_OVERHEAD_BYTES: u64 = 512;
pub const TENSOR_OVERHEAD_BYTES: u64 = 128;

pub fn mib(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / (1024.0 * 1024.0);
}

/// A KV layer's buffers: keys, values and, when quantized, their scales and biases.
pub fn layerArrays(e: *const trunk.KVCacheEntry) [6]mlx.mlx_array {
    return .{ e.keys.array, e.values.array, e.keys_scales.array, e.keys_biases.array, e.values_scales.array, e.values_biases.array };
}

/// Bytes per token row of a `[1, H, rows, D]` buffer; 0 for an empty one.
pub fn rowBytes(a: mlx.mlx_array) u64 {
    if (a.ctx == null) return 0;
    const sh = mlx.getShape(a);
    if (sh.len != 4 or sh[2] <= 0) return 0;
    return nbytesOf(a) / @as(u64, @intCast(sh[2]));
}

pub fn checkpointFileBytes(cp: *const trunk.SSMCheckpoint) u64 {
    var n: u64 = FILE_OVERHEAD_BYTES + cp.released_bytes;
    for (cp.layers) |*l| for ([_]mlx.mlx_array{ l.conv_state, l.ssm_state, l.aux_state, l.qsa_pooled }) |a| {
        if (a.ctx != null) n += nbytesOf(a) + TENSOR_OVERHEAD_BYTES;
    };
    return n;
}

pub fn specFileBytes(sc_opt: ?SpecCommit) u64 {
    const sc = sc_opt orelse return 0;
    var n: u64 = FILE_OVERHEAD_BYTES;
    for (sc.entries) |*e| {
        if (!e.initialized) continue;
        for (layerArrays(e)) |a| n += rowBytes(a) * sc.step + TENSOR_OVERHEAD_BYTES;
    }
    if (sc.head_aux) |h| for ([_]mlx.mlx_array{ h.aux_state, h.qsa_pooled }) |a| {
        if (a.ctx != null) n += nbytesOf(a) + TENSOR_OVERHEAD_BYTES;
    };
    return n;
}

/// Rows `[r0, r1)` of `src` along `axis`.
pub fn sliceAxis(out: *mlx.mlx_array, src: mlx.mlx_array, axis: usize, r0: c_int, r1: c_int, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(src);
    var st: [8]c_int = @splat(0);
    var sp: [8]c_int = undefined;
    const sd: [8]c_int = @splat(1);
    for (shape, 0..) |d, i| sp[i] = d;
    st[axis] = r0;
    sp[axis] = r1;
    try mlx.check(mlx.mlx_slice(out, src, &st, shape.len, &sp, shape.len, &sd, shape.len, s));
}

pub const DiskTier = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    /// Where a read makes and joins its arrays when the model's state is not MLX's: the stream the model's arch names
    /// as the tier is armed. Null is MLX's CPU stream.
    arrays_on: ?mlx.mlx_stream = null,
    /// Absolute root for this model's entries (`<base>/<fingerprint>`). Owned.
    root: []u8,
    /// Byte budget across all entries. 0 = unbounded.
    max_bytes: u64,
    chunk_tokens: u32,
    /// Max bytes one appendCommit piece writes. Without the writer a piece is synchronous on the
    /// inference thread (a 4 GB first commit stalled the next request), so the entry persists one
    /// 512 MB piece per turn; with it, a commit runs `FLUSH_PIECE_BYTES` pieces until it is whole.
    max_flush_bytes: u64 = 512 * 1024 * 1024,
    /// SSD-first mode (mirrored from `HotPrefixCache.ssd_first`): SSM checkpoints ride outside
    /// the per-flush byte budget, beside the chunk that closes their position.
    ssd_first: bool = false,
    /// Checkpoint-retention policy for the persisted position set, mirrored from
    /// `HotPrefixCache.cp_thin`. The default is the previous behaviour (keep the highest N).
    cp_thin: trunk.ThinPolicy = .oldest,
    /// How many checkpoint positions one entry may keep on disk; the default is the previous cap.
    ssm_max_per_entry: usize = SSM_DISK_MAX_PER_ENTRY_LEGACY,
    /// Set by an arch whose restore points carry rows that grow with the prompt; null stores every point whole.
    grown: ?GrownSpec = null,
    /// The loaded arch's restore points hold its whole request state (Hy4-Flash, whose trunk cache holds no rows): a
    /// commit with no KV rows persists its points over its tokens. Elsewhere such a commit persists nothing (GLM's
    /// points hold only its KDA state; its rows persist through its own record).
    points_hold_state: bool = false,
    /// Bytes this commit's checkpoint files may take: nonzero only while one conversation alone outgrows
    /// the tier, when the oldest points go first (`ssmTargetPositions`). Set by each commit.
    ssm_budget: u64 = 0,
    /// Points the last `ssmTargetPositions` dropped for `ssm_budget`.
    ssm_budget_drops: usize = 0,
    /// Bytes of restore-point and row-segment files staged so far, for the write-through log line.
    point_bytes_written: u64 = 0,
    /// The restore-point grid: a point off it (a prompt end, a reply end) is held in RAM, not written, until a flush
    /// or a newer point that extends it (Qwen under SSD-first). 0 writes every point.
    hold_stride: u32 = 0,
    /// Whether the last restore point `loadSsmFile` loaded was a held one (RAM), not a file (SSD).
    last_point_held: bool = false,
    /// The off-grid points held in RAM, oldest first, and their bytes.
    held: std.ArrayList(HeldPoint) = .empty,
    held_bytes: u64 = 0,
    /// Held points a failed write took out of RAM during the current commit: its record may still name them, so they
    /// are unlisted again once the index holds it (`settleHeld`).
    lost_points: std.ArrayList(PointKey) = .empty,
    /// Tests size the cap in points of a fixture (`heldCap`).
    held_cap_for_test: ?u64 = null,
    /// Tests fail every write of a held point (`flushHeldAt`) before it stages.
    fail_held_flush_for_test: bool = false,
    /// Rows per whole segment of a growing sidecar (`SIDE_SEG_SPAN`).
    seg_span: u32 = SIDE_SEG_SPAN,
    /// What the last `persisted` line reports: every file the commit wrote but the manifest and the token record.
    last_persist_bytes: u64 = 0,
    /// Rows below this position are what the committing request took from a stored entry (a restore or a rollback);
    /// the rows past it it computed itself. A row segment is linked from another entry only when its span ends at or
    /// before it, so a restore of the heir is its own state's bytes.
    rows_base: u32 = std.math.maxInt(u32),
    /// Bytes this tier wrote, by kind. Heap-allocated: the background writer holds the pointer.
    stats: *write_stats.Stats,
    /// Background writer (heap-allocated so the mutex survives `init`'s by-value
    /// return). Null = the synchronous path.
    writer: ?*disk_writer.Writer = null,
    /// The permit `enableBackgroundWriter` arms the writer with (the scheduler sets the host's).
    writer_permit_bytes: u64 = disk_writer.DEFAULT_PERMIT_BYTES,
    /// The ceiling `max_bytes` is re-derived under before every store: the flag, or the formula / 20 GB cap;
    /// never a free-space number (`init` seeds it with `max_bytes`; the loader sets the real one).
    operator_cap: u64 = 0,
    /// The free-space probe; tests arm a fixed answer with `armTestSpace`.
    space_probe: disk_placement.SpaceProbeFn = disk_placement.volumeSpace,
    /// The volume has nothing left after `DISK_FREE_RESERVE`: no new entry persists, existing ones stay restorable.
    store_declined: bool = false,
    /// `<base>` (the parent of `root`), for the root-wide sweep. Null when the dupe failed.
    base_dir: ?[]u8 = null,
    entries: std.ArrayList(IndexEntry),
    next_id: u64,
    total_bytes: u64,
    counter: u64,
    /// Chunk count read by the most recent restore. Diagnostics + a
    /// red-on-revert guard that a short-prefix restore reads only the chunks
    /// covering the usable prefix, not the whole stored entry. Not persisted.
    chunks_loaded_last: u32 = 0,
    /// The entry the most recent append committed; null when it committed none.
    last_store_id: ?u64 = null,
    /// `<root>/.lock`, flocked for the tier's lifetime: one live tier owns a root.
    root_lock: ?std.Io.File = null,
    /// `root` is a private root under the shared one, which another live tier owns; removed at deinit.
    private_root: bool = false,
    /// Every file operation under `root`: the plain files or the content-addressed layer.
    fs: kv_disk_fs.Fs,

    /// Create the tier rooted at `<base>/<fingerprint>` and scan whatever
    /// already exists there. Crash leftovers (no meta.json) are deleted. When another live tier
    /// (another process on the same pack) owns that root, this one keeps a private root under it.
    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        base_dir: []const u8,
        fingerprint: []const u8,
        max_bytes: u64,
        chunk_tokens: u32,
    ) !DiskTier {
        if (base_dir.len == 0 or !std.fs.path.isAbsolute(base_dir)) return error.BadDiskCacheDir;
        const shared = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ base_dir, fingerprint });
        var root = shared;
        defer if (root.ptr != shared.ptr) allocator.free(shared);
        errdefer allocator.free(root);
        try io_util.makeDirPath(io, base_dir);
        // The gate keeps a stale clear from deleting the namespace between this tier's mkdir and its lock.
        const gate = try disk_placement.lockNamespaceGate(io, base_dir, fingerprint);
        var lock: ?std.Io.File = null;
        {
            defer gate.close(io);
            try io_util.makeDirPath(io, root);
            lock = try disk_placement.lockRoot(io, shared);
            if (lock == null) {
                const own = try disk_placement.ownPrivateRoot(allocator, io, shared);
                root = own.root;
                lock = own.lock;
            }
        }
        if (root.ptr != shared.ptr) {
            log.info("[disk-cache] {s} is in use by another sushi process: this one keeps a private SSD tier at {s}, removed at exit\n", .{ shared, root });
        }
        disk_placement.reapPrivateRoots(allocator, io, shared);
        var fs = try kv_disk_fs.Fs.open(allocator, io, root);
        errdefer fs.close();
        const stats = try allocator.create(write_stats.Stats);
        errdefer allocator.destroy(stats);
        stats.* = .{ .parent = &write_stats.process };
        var self: DiskTier = .{
            .allocator = allocator,
            .io = io,
            .root = root,
            .stats = stats,
            .max_bytes = max_bytes,
            .operator_cap = max_bytes,
            .chunk_tokens = if (chunk_tokens == 0) DEFAULT_CHUNK_TOKENS else chunk_tokens,
            .entries = std.ArrayList(IndexEntry).empty,
            .next_id = 1,
            .total_bytes = 0,
            .counter = 0,
            .root_lock = lock,
            .private_root = root.ptr != shared.ptr,
            .fs = fs,
        };
        self.fs.stats = stats;
        self.dropPlainRecords();
        self.scan() catch |err| {
            log.warn("[disk-cache] scan failed: {s} — starting empty\n", .{@errorName(err)});
        };
        self.enforceBudget();
        self.base_dir = allocator.dupe(u8, base_dir) catch null;
        return self;
    }

    /// Test hook: answer every free-space probe with these numbers. Every SSD-first test must call this.
    pub fn armTestSpace(self: *DiskTier, free: u64, total: u64) void {
        disk_placement.test_space = .{ .free = free, .total = total };
        self.space_probe = disk_placement.testSpaceProbe;
    }


    /// Arm the background writer. A spawn failure keeps the synchronous path.
    pub fn enableBackgroundWriter(self: *DiskTier) void {
        if (self.writer != null) return;
        const w = self.allocator.create(disk_writer.Writer) catch return;
        w.* = disk_writer.Writer.init(self.allocator, self.io);
        w.permit_bytes = self.writer_permit_bytes;
        w.stats = self.stats;
        w.start() catch {
            self.allocator.destroy(w);
            log.warn("[disk-cache] background writer unavailable — writing synchronously\n", .{});
            return;
        };
        self.writer = w;
        self.fs.land = &w.land_mutex;
        self.max_flush_bytes = disk_placement.FLUSH_PIECE_BYTES;
        log.info("[disk-cache] background writer armed (permit {d} MB, {d} MB pieces, whole entries per commit)\n", .{
            w.permit_bytes / (1024 * 1024),
            self.max_flush_bytes / (1024 * 1024),
        });
    }

    /// Wait for every staged file to land.
    pub fn drainWriter(self: *DiskTier) void {
        if (self.writer) |w| w.drain();
    }

    /// Host bytes staged for the writer and not yet written. Zero when the writer is not armed.
    pub fn stagedHostBytes(self: *DiskTier) u64 {
        const w = self.writer orelse return 0;
        return w.pendingBytes();
    }

    /// Background write failures so far.
    pub fn writeErrors(self: *DiskTier) u64 {
        const w = self.writer orelse return 0;
        return w.writeErrorCount();
    }

    /// Non-blocking: does entry `id` still have files staged or in flight? An unarmed tier
    /// answers false (its writes were synchronous).
    pub fn entryWritesPending(self: *DiskTier, id: u64) bool {
        const w = self.writer orelse return false;
        const pre = std.fmt.allocPrint(self.allocator, "{s}/e{d}/", .{ self.root, id }) catch return true;
        defer self.allocator.free(pre);
        return w.pendingPrefix(pre);
    }

    /// Frees the RAM state of every checkpoint in `cps` (ascending by position) but the newest one still
    /// held, after waiting for its file to leave the writer: staged for entry `id`, the file is the
    /// restore point from then on. A checkpoint the tier did not take, or whose write failed, is not
    /// a restore point (the failure poisons the entry on the next commit) and its state is not kept.
    pub fn settleCheckpoints(self: *DiskTier, id: ?u64, cps: []trunk.SSMCheckpoint) void {
        var newest: ?usize = null;
        for (cps, 0..) |*cp, i| {
            if (!cp.isReleased()) newest = i;
        }
        for (cps, 0..) |*cp, i| {
            if (cp.isReleased() or i == newest) continue;
            if (self.writer) |w| if (id) |entry| {
                if (std.fmt.allocPrint(self.allocator, "{s}/e{d}/s{d:0>7}.safetensors", .{ self.root, entry, cp.pos })) |path| {
                    defer self.allocator.free(path);
                    w.drainPrefix(path);
                } else |_| {}
            };
            cp.release();
        }
    }


    pub fn deinit(self: *DiskTier) void {
        for (self.held.items) |h| self.allocator.free(h.bytes);
        self.held.deinit(self.allocator);
        self.lost_points.deinit(self.allocator);
        if (self.grown) |g| self.allocator.free(g.divs);
        if (self.base_dir) |b| self.allocator.free(b);
        self.base_dir = null;
        if (self.writer) |w| {
            // Teardown must never block on a paused writer: lift the pause so the queue drains for real.
            w.setPaused(false);
            w.drain();
            w.deinit();
            self.fs.land = null;
            self.allocator.destroy(w);
            self.writer = null;
        }
        for (self.entries.items) |*e| {
            self.freeIndexEntryOwned(e);
        }
        self.entries.deinit(self.allocator);
        self.fs.close();
        // Removed under its lock, so no other tier reaps it mid-delete.
        if (self.private_root) disk_placement.deleteTreeAbsolute(self.io, self.root);
        if (self.root_lock) |f| f.close(self.io);
        self.root_lock = null;
        self.allocator.destroy(self.stats);
        self.allocator.free(self.root);
    }


    /// Splits the restore points this tier writes from now on (`GrownSpec`); `divs` is copied.
    pub fn setGrownSpec(self: *DiskTier, span: u32, divs: []const u32, row_bytes: u64) !void {
        const own = try self.allocator.dupe(u32, divs);
        if (self.grown) |g| self.allocator.free(g.divs);
        self.grown = .{ .span = span, .divs = own, .row_bytes = row_bytes };
    }


    pub fn entryCount(self: *const DiskTier) usize {
        return self.entries.items.len;
    }


    // ── Lookup & Restore (kv_disk_restore.zig) ──
    pub const bestMatch = kv_disk_restore.bestMatch;
    pub const bestHybridMatch = kv_disk_restore.bestHybridMatch;
    pub const bestRingMatch = kv_disk_restore.bestRingMatch;
    pub const ringPointsHeld = kv_disk_restore.ringPointsHeld;
    pub const restoreIntoRinged = kv_disk_restore.restoreIntoRinged;
    pub const loadRingFile = kv_disk_restore.loadRingFile;
    pub const restoreInto = kv_disk_restore.restoreInto;
    pub const restorePrefixInto = kv_disk_restore.restorePrefixInto;
    pub const restoreIntoHybrid = kv_disk_restore.restoreIntoHybrid;
    pub const restoreIntoKda = kv_disk_restore.restoreIntoKda;
    pub const restoreCheckpointInto = kv_disk_restore.restoreCheckpointInto;
    pub const PointSink = kv_disk_restore.PointSink;
    pub const restorePointInto = kv_disk_restore.restorePointInto;
    pub const PointRead = kv_disk_restore.PointRead;
    pub const metaValue = kv_disk_restore.metaValue;
    pub const readPointInto = kv_disk_restore.readPointInto;
    pub const highestSsmPosAtOrBelow = kv_disk_restore.highestSsmPosAtOrBelow;
    pub const tokenRecordMatches = kv_disk_restore.tokenRecordMatches;
    pub const restoreKvInto = kv_disk_restore.restoreKvInto;
    pub const restoredRows = kv_disk_restore.restoredRows;
    pub const joinGrownSegments = kv_disk_restore.joinGrownSegments;
    pub const Reading = kv_disk_restore.Reading;
    pub const reading = kv_disk_restore.reading;
    pub const loadSsmFile = kv_disk_restore.loadSsmFile;
    pub const specStampMatches = kv_disk_restore.specStampMatches;
    pub const loadSpecSnap = kv_disk_restore.loadSpecSnap;
    pub const joinSegments = kv_disk_restore.joinSegments;
    pub const getSpecArray = kv_disk_restore.getSpecArray;

    // ── Commit & Persistence (kv_disk_persist.zig) ──
    pub const appendCommit = kv_disk_persist.appendCommit;
    pub const CommitParams = kv_disk_persist.CommitParams;
    pub const CommitOpts = kv_disk_persist.CommitOpts;
    pub const appendCommitWith = kv_disk_persist.appendCommitWith;
    pub const appendCommitBounded = kv_disk_persist.appendCommitBounded;
    pub const appendCommitWithSpec = kv_disk_persist.appendCommitWithSpec;
    pub const appendCommitWithRing = kv_disk_persist.appendCommitWithRing;
    pub const appendCommitWhole = kv_disk_persist.appendCommitWhole;
    pub const appendCommitPieces = kv_disk_persist.appendCommitPieces;
    pub const storeProgress = kv_disk_persist.storeProgress;
    pub const budgetTarget = kv_disk_persist.budgetTarget;
    pub const pointBudget = kv_disk_persist.pointBudget;
    pub const pointFileBytes = kv_disk_persist.pointFileBytes;
    pub const entryBytesAt = kv_disk_persist.entryBytesAt;
    pub const appendCommitWithSpecBounded = kv_disk_persist.appendCommitWithSpecBounded;
    pub const holdsFullPrefix = kv_disk_persist.holdsFullPrefix;
    pub const fullPrefixEntryId = kv_disk_persist.fullPrefixEntryId;
    pub const appendSsmOnly = kv_disk_persist.appendSsmOnly;
    pub const SpecSidecarResult = kv_disk_persist.SpecSidecarResult;
    pub const specWorkPending = kv_disk_persist.specWorkPending;
    pub const writeSpecSidecar = kv_disk_persist.writeSpecSidecar;
    pub const SegPrev = kv_disk_persist.SegPrev;
    pub const SegResult = kv_disk_persist.SegResult;
    pub const Growth = kv_disk_persist.Growth;
    pub const specGrowth = kv_disk_persist.specGrowth;
    pub const writeSegments = kv_disk_persist.writeSegments;
    pub const rowsFrom = kv_disk_persist.rowsFrom;
    pub const imageHash = kv_disk_persist.imageHash;
    pub const linkSeg = kv_disk_persist.linkSeg;
    pub const linkSegFile = kv_disk_persist.linkSegFile;
    pub const deleteSegFiles = kv_disk_persist.deleteSegFiles;
    pub const appendSpecTensors = kv_disk_persist.appendSpecTensors;
    pub const appendSpecArray = kv_disk_persist.appendSpecArray;
    pub const appendSpecSlice = kv_disk_persist.appendSpecSlice;
    pub const NamedTensor = kv_disk_persist.NamedTensor;
    pub const freeNamed = kv_disk_persist.freeNamed;
    pub const writeChunkFile = kv_disk_persist.writeChunkFile;
    pub const appendLayerSlices = kv_disk_persist.appendLayerSlices;
    pub const saveNamed = kv_disk_persist.saveNamed;
    pub const TensorViews = kv_disk_persist.TensorViews;
    pub const tensorViews = kv_disk_persist.tensorViews;
    pub const saveSync = kv_disk_persist.saveSync;
    pub const appendSlice = kv_disk_persist.appendSlice;
    pub const MetaPair = kv_disk_persist.MetaPair;
    pub const no_meta = kv_disk_persist.no_meta;
    pub const serializeNamed = kv_disk_persist.serializeNamed;
    pub const encodeNamed = kv_disk_persist.encodeNamed;
    pub const dataLen = kv_disk_persist.dataLen;
    pub const SsmPersistResult = kv_disk_persist.SsmPersistResult;
    pub const growable = kv_disk_persist.growable;
    pub const pointSegments = kv_disk_persist.pointSegments;
    pub const grownRows = kv_disk_persist.grownRows;
    pub const sliceRows = kv_disk_persist.sliceRows;
    pub const writeGrownSegment = kv_disk_persist.writeGrownSegment;
    pub const growSegments = kv_disk_persist.growSegments;
    pub const findCp = kv_disk_persist.findCp;
    pub const ssmTargetPositions = kv_disk_persist.ssmTargetPositions;
    pub const StoreTarget = kv_disk_persist.StoreTarget;
    pub const CommitMedia = kv_disk_persist.CommitMedia;
    pub const commitShares = kv_disk_persist.commitShares;
    pub const commitSharesWithChain = kv_disk_persist.commitSharesWithChain;
    pub const selectStoreTarget = kv_disk_persist.selectStoreTarget;
    pub const ssmWorkPending = kv_disk_persist.ssmWorkPending;
    pub const persistSsmCheckpoints = kv_disk_persist.persistSsmCheckpoints;
    pub const writeSsmFile = kv_disk_persist.writeSsmFile;
    pub const writePoint = kv_disk_persist.writePoint;
    pub const encodePoint = kv_disk_persist.encodePoint;
    pub const QsaHistoryResult = kv_disk_persist.QsaHistoryResult;
    pub const deleteSsmFile = kv_disk_persist.deleteSsmFile;
    pub const RingSource = kv_disk_persist.RingSource;
    pub const ringFileBytes = kv_disk_persist.ringFileBytes;
    pub const ringRowsAt = kv_disk_persist.ringRowsAt;
    pub const ringSources = kv_disk_persist.ringSources;
    pub const ringSourceAt = kv_disk_persist.ringSourceAt;
    pub const ringTargetPositions = kv_disk_persist.ringTargetPositions;
    pub const thinRingPositions = kv_disk_persist.thinRingPositions;
    pub const ringWorkPending = kv_disk_persist.ringWorkPending;
    pub const ringsEqual = kv_disk_persist.ringsEqual;
    pub const persistRings = kv_disk_persist.persistRings;
    pub const writeRingFile = kv_disk_persist.writeRingFile;
    pub const deleteRingFile = kv_disk_persist.deleteRingFile;
    pub const heldCap = kv_disk_persist.heldCap;
    pub const holdsPosition = kv_disk_persist.holdsPosition;
    pub const heldIndex = kv_disk_persist.heldIndex;
    pub const holdPoint = kv_disk_persist.holdPoint;
    pub const heldExtendedBy = kv_disk_persist.heldExtendedBy;
    pub const dropHeldAt = kv_disk_persist.dropHeldAt;
    pub const dropHeldOf = kv_disk_persist.dropHeldOf;
    pub const unlistPoint = kv_disk_persist.unlistPoint;
    pub const flushHeldAt = kv_disk_persist.flushHeldAt;
    pub const settleHeld = kv_disk_persist.settleHeld;
    pub const fitHeld = kv_disk_persist.fitHeld;
    pub const flushHeld = kv_disk_persist.flushHeld;
    pub const chunkShareDonor = kv_disk_persist.chunkShareDonor;
    pub const segmentDonor = kv_disk_persist.segmentDonor;
    pub const restoredSegmentSource = kv_disk_persist.restoredSegmentSource;
    pub const ChunkDonor = kv_disk_persist.ChunkDonor;
    pub const restoredSource = kv_disk_persist.restoredSource;
    pub const linkInheritedChunks = kv_disk_persist.linkInheritedChunks;
    pub const linkInheritedSegments = kv_disk_persist.linkInheritedSegments;
    pub const unlinkSegment = kv_disk_persist.unlinkSegment;
    pub const chunkLanded = kv_disk_persist.chunkLanded;
    pub const unlinkChunkRange = kv_disk_persist.unlinkChunkRange;
    pub const bequeathPoint = kv_disk_persist.bequeathPoint;
    pub const sameRowsBelow = kv_disk_persist.sameRowsBelow;
    pub const linkPointInto = kv_disk_persist.linkPointInto;

    // ── Index, Budget, GC & Scan (kv_disk_index.zig) ──
    pub const dropPlainRecords = kv_disk_index.dropPlainRecords;
    pub const harvestWriteFailures = kv_disk_index.harvestWriteFailures;
    pub const entryIdFromPath = kv_disk_index.entryIdFromPath;
    pub const holdsPoint = kv_disk_index.holdsPoint;
    pub const poisonId = kv_disk_index.poisonId;
    pub const poisonEntry = kv_disk_index.poisonEntry;
    pub const poisonAll = kv_disk_index.poisonAll;
    pub const dropPoisonedEntries = kv_disk_index.dropPoisonedEntries;
    pub const chunkOnDisk = kv_disk_index.chunkOnDisk;
    pub const entryWholeOnDisk = kv_disk_index.entryWholeOnDisk;
    pub const entryPoisoned = kv_disk_index.entryPoisoned;
    pub const drainEntry = kv_disk_index.drainEntry;
    pub const freeIndexEntryOwned = kv_disk_index.freeIndexEntryOwned;
    pub const refreshDiskBudget = kv_disk_index.refreshDiskBudget;
    pub const sweepSiblings = kv_disk_index.sweepSiblings;
    pub const claimEntryId = kv_disk_index.claimEntryId;
    pub const scanSegs = kv_disk_index.scanSegs;
    pub const invalidateAll = kv_disk_index.invalidateAll;
    pub const invalidateNewest = kv_disk_index.invalidateNewest;
    pub const bump = kv_disk_index.bump;
    pub const removeAt = kv_disk_index.removeAt;
    pub const bytesFreedByRemoving = kv_disk_index.bytesFreedByRemoving;
    pub const deleteEntryDir = kv_disk_index.deleteEntryDir;
    pub const diskBytes = kv_disk_index.diskBytes;
    pub const enforceBudget = kv_disk_index.enforceBudget;
    pub const entryCutTarget = kv_disk_index.entryCutTarget;
    pub const cutBytes = kv_disk_index.cutBytes;
    pub const cutEntryTo = kv_disk_index.cutEntryTo;
    pub const deleteCountFreed = kv_disk_index.deleteCountFreed;
    pub const gcToBudget = kv_disk_index.gcToBudget;
    pub const writeMeta = kv_disk_index.writeMeta;
    pub const metaVersionFor = kv_disk_index.metaVersionFor;
    pub const renderMeta = kv_disk_index.renderMeta;
    pub const scan = kv_disk_index.scan;
    pub const billChunksOnce = kv_disk_index.billChunksOnce;
    pub const loadEntry = kv_disk_index.loadEntry;

    pub const linkInheritedQsa = prefix_arch.disk.linkInheritedQsa;
    pub const loadQsaHistoryFile = prefix_arch.disk.loadQsaHistoryFile;
    pub const newestQsaCheckpoint = prefix_arch.disk.newestQsaCheckpoint;
    pub const persistQsaHistory = prefix_arch.disk.persistQsaHistory;
    pub const qsaHistoryRowsOf = prefix_arch.disk.qsaHistoryRowsOf;
    pub const writeQsaHistoryFile = prefix_arch.disk.writeQsaHistoryFile;
};

/// Bytes of an entry's files that are never shared: tokens.bin, the SSM checkpoints and the spec sidecar.
pub fn nonChunkBytes(e: *const IndexEntry) u64 {
    var n: u64 = @as(u64, e.tokens.len) * 4 + e.spec_bytes;
    for (e.spec_segs[@min(e.spec_segs_inherited, e.spec_segs.len)..]) |sg| n += sg.bytes;
    for (e.qsa_segs[@min(e.qsa_segs_inherited, e.qsa_segs.len)..]) |sg| n += sg.bytes;
    for (e.ssm_bytes) |b| n += b;
    for (e.grown[@min(e.inherited_grown, e.grown.len)..]) |b| n += b;
    for (e.rings) |r| n += r.bytes;
    if (!e.inherited_qsa) n += e.qsa_history_bytes;
    return n;
}

pub fn clampAdd(base: u64, delta: i64) u64 {
    const v: i128 = @as(i128, base) + @as(i128, delta);
    return if (v < 0) 0 else @intCast(v);
}

/// Length of the longest common prefix of two token slices.
pub fn commonPrefixLen(a: []const u32, b: []const u32) usize {
    const n = @min(a.len, b.len);
    var i: usize = 0;
    while (i < n and a[i] == b[i]) : (i += 1) {}
    return i;
}

/// Test seam: false makes every commit write all its chunks instead of hard-linking a donor's.
pub var chunk_share_override: ?bool = null;

pub fn chunkShareEnabled() bool {
    return chunk_share_override orelse true;
}

pub const statFile = kv_disk_fs.statFile;

pub fn fileSize(io: std.Io, abs_path: []const u8) ?u64 {
    const st = statFile(io, abs_path) orelse return null;
    return st.size;
}

pub const readFileAlloc = kv_disk_fs.readFileAlloc;


// ── Shared test helper re-exports (kv_disk_helpers_test.zig) ──
pub const fillCache = kv_disk_helpers_test.fillCache;
pub const cacheValueAt = kv_disk_helpers_test.cacheValueAt;
pub const tmpRoot = kv_disk_helpers_test.tmpRoot;
pub const makeArange = kv_disk_helpers_test.makeArange;
pub const ssmArrVal = kv_disk_helpers_test.ssmArrVal;
pub const buildHybridEntries = kv_disk_helpers_test.buildHybridEntries;
pub const freeHybridEntries = kv_disk_helpers_test.freeHybridEntries;
pub const chunkShareTokens = kv_disk_helpers_test.chunkShareTokens;

pub var test_evict_tier: ?*DiskTier = null;
pub var test_evict_id: ?u64 = null;

pub fn evictingSpaceProbe(path: []const u8) ?disk_placement.VolumeSpace {
    _ = path;
    if (test_evict_tier) |t| {
        if (test_evict_id) |id| {
            test_evict_id = null;
            for (t.entries.items, 0..) |e, i| if (e.id == id) {
                t.removeAt(i);
                break;
            };
        }
    }
    return .{ .free = 1024 * 1024 * 1024 * 1024, .total = 2 * 1024 * 1024 * 1024 * 1024 };
}
