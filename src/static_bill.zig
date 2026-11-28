//! A load on the static bill's ledger (docs/bill_redesign.md parts 4 and 9): its budget, read once while its weights are
//! resident and before its chunk is chosen; its model bill, committed at the seal from the engine's own counts; its
//! context in closed form; and each request's admission and end, decided on the ledger's lines. A load is on it when
//! its arch answers `ledgerTerms` (it plans) and it warms (its plans are built in the warm).
const std = @import("std");
const ledger = @import("ledger.zig");
const bills = @import("bills.zig");
const gpu_mem = @import("gpu_mem.zig");
const model_mod = @import("model.zig");
const expert_io = @import("expert_io.zig");
const sched = @import("arch/sched.zig");
const log = @import("log.zig");
const metrics = @import("status.zig");
const scheduler_mod = @import("scheduler.zig");
const forward_refusal = @import("forward_refusal.zig");
const request_media = @import("request_media.zig");
const vision_queue = @import("vision_queue.zig");
const arch_media = @import("arch/media.zig");
const slot_admission = @import("slot_admission.zig");
const model_registry = @import("model_registry.zig");
const kv_quant = @import("kv_quant.zig");
const arch_flags = @import("arch/flags.zig");

const ModelConfig = model_mod.ModelConfig;
const mb: u64 = 1 << 20;

/// The engine's counts at a load's start: what the load holds is what they gained since.
pub const Before = struct {
    held: gpu_mem.EngineHeld,
    weights: u64,
    slots: u64,
    metal: u64,
    mlx: u64,
    /// The active bytes (`gpu_mem.activeBytes`), MLX's and the engine's.
    active: u64,

    pub fn now() Before {
        const m = gpu_mem.metalAllocated();
        return .{ .held = gpu_mem.engineHeld(), .weights = model_mod.owned_bytes.load(.monotonic), .slots = expert_io.slab_bytes.load(.monotonic), .metal = m.metal, .mlx = m.mlx, .active = gpu_mem.activeBytes() };
    }

    /// What Metal allocated for the load past every buffer the engine and MLX count: the pipelines the warm compiled
    /// before the seal, with whatever else the driver keeps for them. A remainder, so never below 0. The pool's free
    /// buffers from before the load that it released (the warm's trim) left Metal's count, not the load's.
    fn pipelinesSince(b: Before) u64 {
        const m = gpu_mem.metalAllocated();
        const released = b.held.free -| gpu_mem.engineHeld().free;
        const counted = @as(i128, b.heldSince()) + (@as(i128, m.mlx) - @as(i128, b.mlx)) - @as(i128, released);
        const rest = @as(i128, m.metal) - @as(i128, b.metal) - counted;
        return if (rest > 0) @intCast(rest) else 0;
    }

    /// What this load holds as Metal holds it: the pool buffers it took into use and the free ones added since (a free
    /// buffer it reused was the pool's before it), its arena, and the regions the loader mapped and the expert slabs,
    /// whole pages (the engine's mapped count is the arrays' bytes).
    pub fn heldSince(b: Before) u64 {
        const h = gpu_mem.engineHeld();
        return (h.pool -| b.held.pool) +| (h.free -| b.held.free) +| (h.arena -| b.held.arena) +| (model_mod.owned_bytes.load(.monotonic) -| b.weights) +| (expert_io.slab_bytes.load(.monotonic) -| b.slots);
    }

    /// What this load added to the active bytes (`gpu_mem.activeBytes`), whose mapped part is the arrays' bytes.
    fn activeSince(b: Before) u64 {
        return gpu_mem.engineHeld().total() -| b.held.total();
    }

    /// The load's model bill from the engine's own counts (part 4): the weights the loader mapped, the expert slabs,
    /// the arena, what the pool keeps (`static`), and what Metal allocated past them (`pipelines`); the margin is
    /// `bills.BILL_MARGIN_BYTES` until one is measured per GPU class (D7).
    pub fn modelBill(b: Before) ledger.ModelBill {
        const h = gpu_mem.engineHeld();
        return .{
            .weights = model_mod.owned_bytes.load(.monotonic) -| b.weights,
            .slots = expert_io.slab_bytes.load(.monotonic) -| b.slots,
            .static = h.pool -| b.held.pool,
            .arena = h.arena -| b.held.arena,
            .pipelines = b.pipelinesSince(),
            .margin = bills.BILL_MARGIN_BYTES,
        };
    }
};

/// Whether a load of `config` serves on the ledger: its arch bills only there (`bills.ledgerArch`) and it warms.
pub fn onLedger(config: *const ModelConfig, warms: bool) bool {
    return warms and bills.ledgerArch(config);
}

/// B (part 9): the static-bill planning ceiling (`bills.ceilingNowStatic`, 0 GiB default wired margin unless
/// overridden) less what the process held before this load: what MLX holds, its cache too, and the other resident
/// models (the engine's free buffers among them, each load on the ledger at its whole budget), and the decoder pools
/// lent out of it.
pub fn budgetFor(b: Before) u64 {
    const mlx_cache = gpu_mem.cacheBytes() -| gpu_mem.engineFree();
    const others = (bills.promisedActive(gpu_mem.activeBytes(), null) -| b.activeSince()) +| mlx_cache +| b.held.free;
    return bills.ceilingNowStatic().planning() -| others;
}

/// The load's ledger opens with its budget, read once here. Returns the room its chunk and context are chosen in: the
/// budget less what the load holds now and its margin.
pub fn open(config: *ModelConfig, b: Before) error{OutOfMemory}!u64 {
    const l = try std.heap.c_allocator.create(ledger.Ledger);
    l.* = .{};
    l.setBudget(budgetFor(b));
    config.static_ledger = l;
    const held = b.heldSince();
    log.info("[ledger] budget {d} MB (the planning ceiling less what MLX and the other models hold); the load holds {d} MB so far\n", .{ l.budget() / mb, held / mb });
    return l.budget() -| held -| bills.BILL_MARGIN_BYTES;
}

/// The load ends or fails: its ledger goes with it.
pub fn close(config: *ModelConfig) void {
    const l = config.static_ledger orelse return;
    l.letGo();
    std.heap.c_allocator.destroy(l);
    config.static_ledger = null;
}

/// The load's warm ended: its plans are built and its pool keeps only what the load made. Its model bill is committed
/// on the ledger, and its context computed once in closed form beside it (`ledger.contextFor`) and kept on the ledger
/// (`Ledger.context`). The static and pipelines the warm added were not known when the chunk was chosen, so a fixed
/// context (`--ctx-size`, the model's `ctx_size`, a pinned one) the lines no longer hold refuses the load by name, with
/// the widest narrower chunk that holds it (`bills.chunkHoldingContext`, its arena measured dry from the plans).
pub fn seal(config: *ModelConfig, b: Before, arena: bills.ArenaAt) !void {
    const l = config.static_ledger.?;
    const bill = b.modelBill();
    var why: ledger.Refusal = undefined;
    bill.reserveOn(l, &why) catch {
        log.err("[ledger] the model bill of {d} MB is past the budget of {d} MB ({t} refused at {d} MB committed)\n", .{ bill.sum() / mb, l.budget() / mb, why.line, why.committed / mb });
        return error.ModelBillPastBudget;
    };
    const terms = l.terms.?;
    const room = l.room();
    const fits = ledger.contextFor(room, terms, bills.ledgerWindow(config));
    const ctx = bills.ledgerContext(config, fits);
    if (ctx.tokens == 0 or ctx.tokens > fits) {
        var rung: ?u32 = null;
        var chunks_buf: [256]u8 = undefined;
        var min: u32 = 0;
        switch (bills.prefillChunkBounds(&chunks_buf, config)) {
            .bounds => |bounds| {
                min = bounds.min;
                rung = try bills.chunkHoldingContext(config, bounds, config.static_prefill_chunk, @max(ctx.tokens, 1), bill.sum() - bill.arena, l.budget(), arena);
            },
            .refused => {},
        }
        var line: [640]u8 = undefined;
        log.err("{s}\n", .{contextRefusalLine(&line, .{ .context = ctx, .needed = terms.at(@max(ctx.tokens, 1)), .bill = bill.sum(), .budget = l.budget(), .room = room, .fits = fits, .chunk = config.static_prefill_chunk, .min = min, .rung = rung })});
        return error.ContextDoesNotFit;
    }
    if (ctx.source == .auto) config.pinned_context = ctx.tokens;
    l.context.store(fits, .release);
    l.holdWhole(gpu_mem.activeBytes() -| b.active);
    log.info("[ledger] model bill {d} MB: weights {d} MB, slots {d} MB, static {d} MB, pipelines {d} MB, arena {d} MB, host {d} MB, margin {d} MB (BILL_MARGIN_BYTES); budget {d} MB; context {d} tokens ({t}) at {d} B a token beside {d} MB fixed and {d} MB of restore points a request, chunk {d}; Metal holds {d} MB for the process, its footprint is {d} MB\n", .{
        bill.sum() / mb,  bill.weights / mb, bill.slots / mb, bill.static / mb, bill.pipelines / mb,         bill.arena / mb,                     bill.host / mb,
        bill.margin / mb, l.budget() / mb,   ctx.tokens,      ctx.source,       terms.per_token,             terms.fixed / mb,                    terms.points / mb,
        config.static_prefill_chunk, gpu_mem.metalAllocated().metal / mb, metrics.getAppMemFootprintMb(),
    });
}

/// What a seal's refusal of a context says: the context and where it came from, what it needs beside the model bill at
/// the load's chunk, the room the budget leaves and the tokens that hold, and the widest narrower chunk (down to `min`)
/// that would hold it.
const ContextRefusal = struct { context: bills.LedgerContext, needed: u64, bill: u64, budget: u64, room: u64, fits: u32, chunk: u64, min: u32, rung: ?u32 };

fn contextRefusalLine(buf: []u8, r: ContextRefusal) []const u8 {
    var remedy_buf: [192]u8 = undefined;
    const label = r.context.source.label();
    const remedy = if (r.rung) |rung|
        std.fmt.bufPrint(&remedy_buf, "a prefill chunk of {d} rows holds it: pass --prefill-chunk-max {d}", .{ rung, rung }) catch "a narrower prefill chunk holds it"
    else switch (r.context.source) {
        .flag, .setting => std.fmt.bufPrint(&remedy_buf, "no prefill chunk down to the min of {d} rows holds it: lower {s}", .{ r.min, label }) catch "lower the context",
        .pinned => "no prefill chunk holds it, and the session keeps the context it advertised: free memory, or restart the server",
        .auto => "not one token fits at any prefill chunk: free memory",
    };
    return std.fmt.bufPrint(buf, "[ledger] {s} of {d} tokens needs {d} MB beside the model bill of {d} MB at a prefill chunk of {d} rows; the budget of {d} MB leaves {d} MB, which holds {d} tokens; {s}", .{ label, r.context.tokens, bills.mbUp(r.needed), r.bill / mb, r.chunk, r.budget / mb, r.room / mb, r.fits, remedy }) catch "[ledger] the context does not fit";
}

/// Why a request's `kv_quant` is refused, in one text (D103), or null when the load takes it. A load on the ledger serves
/// only its own width, since it sized its context and its region and planned its forwards there: a request naming another
/// is refused by that width, and by the arch's own words too when its cache cannot store the width at all. Any other load
/// is refused only by its arch. The text is written into `buf`.
pub fn kvRefusal(buf: []u8, config: *const ModelConfig, kv: kv_quant.KVQuantConfig) !?[]const u8 {
    const arch_says = arch_flags.kvQuantRequestRefusal(config, kv);
    if (config.static_ledger == null) return arch_says;
    const own = bills.configuredKvQuantFor(config);
    if (kv.scheme == own.scheme and (kv.scheme == .off or kv.bits == own.bits)) return arch_says;
    return try std.fmt.bufPrint(buf, "This model serves only kv_quant {s}, the width it was loaded with (--kv-quant): a request may not name another.{s}{s}", .{ own.wireName(), if (arch_says != null) " " else "", arch_says orelse "" });
}

/// What a request reserves on its load's ledger, priced once by `reservation`: every reader of a request's bill (its
/// admission, the connection thread's check, the scheduler's hold, a refusal's figures) reads this.
pub const Reservation = struct {
    /// The load's terms at the request's reach (prompt and max_tokens), device.
    reach: u64,
    /// The rows its images' encode writes, device.
    encode: u64 = 0,
    /// Its images' decoded pixels, host.
    media: u64 = 0,

    /// What goes on the ledger's `state` line.
    pub fn state(r: Reservation) u64 {
        return r.reach +| r.encode;
    }

    pub fn total(r: Reservation) u64 {
        return r.state() +| r.media;
    }

    fn take(r: Reservation, l: *ledger.Ledger, why: *ledger.Refusal) error{OverBudget}!void {
        try l.reserve(.state, r.state(), why);
        l.reserve(.media, r.media, why) catch |e| {
            l.release(.state, r.state());
            return e;
        };
    }

    fn give(r: Reservation, l: *ledger.Ledger) void {
        l.release(.state, r.state());
        l.release(.media, r.media);
    }
};

/// A request's bill on a load on the ledger, from the terms the load fixed (`Ledger.terms`) and the media it brings.
pub fn reservation(config: *const ModelConfig, prompt_len: usize, max_tokens: u32, media: ?*const vision_queue.DeferredMedia) Reservation {
    var r: Reservation = .{ .reach = config.static_ledger.?.terms.?.at(@as(u64, prompt_len) +| max_tokens) };
    if (media) |d| {
        const in = request_media.mediaInputs(config, d.images, d.videos);
        var rows: usize = 0;
        for (d.blocks) |b| rows += b.rows;
        r.encode = arch_media.mediaBill(config, in.worst, 0, rows) orelse 0;
        r.media = in.input_bytes;
    }
    return r;
}

/// The connection thread's answer for a request on a load on the ledger: what admission would decide on the lines now.
pub const Check = union(enum) {
    /// Admit, release the parked state, or wait: the scheduler holds a wait.
    proceed,
    /// It does not fit even once the running request ends: the figures the client is told.
    refused: model_registry.MemoryContextRefusal,
};

pub fn check(config: *const ModelConfig, prompt_len: usize, max_tokens: u32, media: ?*const vision_queue.DeferredMedia) Check {
    const l = config.static_ledger.?;
    const r = reservation(config, prompt_len, max_tokens, media);
    const s = l.snapshot();
    if (ledger.decide(s, r.total()) != .refuse) return .proceed;
    const room = s.roomAfter(.running);
    // The reach that fits is what the room leaves past the encode's rows and the pixels.
    const maximum = ledger.contextFor(room -| r.encode -| r.media, l.terms.?, bills.ledgerWindow(config));
    const prompt: u64 = prompt_len;
    log.warn("  prompt {d} tokens needs {d} MB, {d} MB is free once the running request ends and the parked state is released - rejecting\n", .{ prompt_len, r.total() / mb, room / mb });
    return .{ .refused = .{
        .requested = prompt +| max_tokens,
        .maximum = maximum,
        .needed = r.total(),
        .available = room,
        .kv_bits = bills.defaultKvBits(config),
        .chunk = config.static_prefill_chunk,
        .asked_max_tokens = max_tokens,
        .fitting_max_tokens = @intCast(@min(@as(u64, max_tokens), @as(u64, maximum) -| prompt)),
        .smaller_pack = config.smaller_pack,
    } };
}

/// The scheduler's hold of a pending request on a load on the ledger: its reservation against the room `ledger.decide`
/// admits it in, the parked state released. Null for a load off the ledger.
pub fn holdBill(config: *const ModelConfig, prompt_len: usize, max_tokens: u32, media: ?*const vision_queue.DeferredMedia) ?slot_admission.MemoryBill {
    const l = config.static_ledger orelse return null;
    const r = reservation(config, prompt_len, max_tokens, media);
    return .{ .needed = r.total(), .available = l.snapshot().roomAfter(.parked), .held = r.total() };
}

const too_big = "This request does not fit in this model's memory budget; nothing ran.\nTo serve it:\n- Lower max_tokens, shorten the prompt or send fewer or smaller images.";
const must_wait = "This request fits this model's memory budget only once the running request ends; nothing ran.\nTo serve it:\n- Retry once the running request ends.";

/// The request in `slot` begins its prefill on a load on the ledger: its reservation is admitted on the ledger's lines
/// (`ledger.decide`), the parked state released when it must be, the reservation taken, and free RAM read as a gate that
/// can only refuse. Inference thread, before the request allocates anything, its encode included.
pub fn admit(slot: *scheduler_mod.Slot) error{PrefillDoesNotFit}!void {
    const config = slot.model.config orelse return;
    const l = config.static_ledger orelse return;
    // A slot holds one reservation: a prefill run again (a retry) is admitted again from nothing.
    if (slot.ledger) |held| held.give(l);
    slot.ledger = null;
    const r = reservation(config, slot.full_prompt.len, slot.max_tokens, slot.deferred_media);
    const s = l.snapshot();
    const verdict = ledger.decide(s, r.total());
    log.info("[admission] needed={d} MB committed={d} MB budget={d} MB verdict={t}\n", .{ r.total() / mb, s.committed / mb, s.budget / mb, verdict });
    switch (verdict) {
        .admit => {},
        .release_parked => {
            sched.releaseParked(slot.model);
            l.release(.parked, s.of(.parked));
            l.publish();
        },
        // One request runs a load on the ledger at a time and the scheduler holds the next behind it, so a wait here is
        // a request the ledger cannot cover while another still holds its reservation.
        .wait => return refuse(must_wait),
        .refuse => return refuse(too_big),
    }
    var why: ledger.Refusal = undefined;
    r.take(l, &why) catch return refuse(too_big);
    const free_ram = metrics.getAvailableMemBytes();
    if (ledger.freeRamRefuses(free_ram, r.total())) {
        r.give(l);
        log.warn("[admission] free RAM {d} MB is below the {d} MB this request allocates: memory the budget counted is held outside this server; refused\n", .{ free_ram / mb, r.total() / mb });
        return refuse("This request fits this model's memory budget, but free RAM is below what it allocates: memory is held outside this server. Nothing ran.\nTo serve it:\n- Quit other apps, then retry.");
    }
    slot.ledger = r;
}

fn refuse(text: []const u8) error{PrefillDoesNotFit} {
    forward_refusal.record(text);
    return error.PrefillDoesNotFit;
}

/// The request in `slot` ended on the inference thread (finished, failed, cancelled), or its slot is cleaned up (a client
/// that left mid-decode never reaches the first): its reservation and the state it took over give way to what the model
/// now parks for the next one (`sched.parkedStateBytes`), in one step. A slot that holds no reservation (never
/// admitted, or already ended) changes nothing. State held past the bill is said, by name.
pub fn requestEnded(slot: *scheduler_mod.Slot) void {
    const config = slot.model.config orelse return;
    const l = config.static_ledger orelse return;
    const r = slot.ledger orelse return;
    slot.ledger = null;
    l.release(.media, r.media);
    const was = l.of(.parked);
    const held = sched.parkedStateBytes(slot.model);
    if (held > r.state() +| was) log.err("[ledger] the model parks {d} MB, past the {d} MB its request and the state it took over were billed\n", .{ held / mb, (r.state() +| was) / mb });
    const committed = l.snapshot().committed;
    l.replace(&.{ .{ .state, r.state() }, .{ .parked, was } }, .parked, held);
    l.publish();
    log.info("[ledger] request end: {d} tokens; it reserved {d} MB and the model parks {d} MB; {d} MB committed at its end of the {d} MB budget; Metal holds {d} MB for the process, its footprint is {d} MB\n", .{ slot.full_prompt.len +| slot.completion_tokens, r.state() / mb, held / mb, committed / mb, l.budget() / mb, gpu_mem.metalAllocated().metal / mb, metrics.getAppMemFootprintMb() });
}

/// `/props` `memory.ledger` of a load on the ledger; null for any other.
pub fn propsJson(allocator: std.mem.Allocator, config: *const ModelConfig) !?[]u8 {
    const l = config.static_ledger orelse return null;
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    try ledger.writeJson(l.snapshot(), &out.writer);
    return try out.toOwnedSlice();
}

const testing = std.testing;

test "static bill: a request at another KV width than its load's is refused naming the load's width on a load on the ledger, and passes at the load's width or on a load off it" {
    var config: ModelConfig = .{};
    var buf: [384]u8 = undefined;
    const own = bills.configuredKvQuantFor(&config);
    const other: kv_quant.KVQuantConfig = if (own.scheme == .off) .affine(8) else .dense;
    try testing.expect(try kvRefusal(&buf, &config, other) == null);
    var l: ledger.Ledger = .{};
    config.static_ledger = &l;
    try testing.expect(try kvRefusal(&buf, &config, own) == null);
    const why = (try kvRefusal(&buf, &config, other)).?;
    var want: [32]u8 = undefined;
    try testing.expect(std.mem.indexOf(u8, why, try std.fmt.bufPrint(&want, "serves only kv_quant {s},", .{own.wireName()})) != null);
}

test "static bill: a refused context names where it came from, and the widest chunk that holds it or that none does" {
    var buf: [640]u8 = undefined;
    const base: ContextRefusal = .{ .context = .{ .tokens = 65536, .source = .setting }, .needed = 9000 * mb, .bill = 40000 * mb, .budget = 48000 * mb, .room = 8000 * mb, .fits = 60000, .chunk = 8192, .min = 512, .rung = 2048 };
    const setting = contextRefusalLine(&buf, base);
    try testing.expect(std.mem.indexOf(u8, setting, "the model's ctx_size setting of 65536 tokens") != null);
    try testing.expect(std.mem.indexOf(u8, setting, "--ctx-size") == null);
    try testing.expect(std.mem.indexOf(u8, setting, "a prefill chunk of 2048 rows holds it: pass --prefill-chunk-max 2048") != null);
    var flag = base;
    flag.context.source = .flag;
    flag.rung = null;
    const none = contextRefusalLine(&buf, flag);
    try testing.expect(std.mem.indexOf(u8, none, "--ctx-size of 65536 tokens") != null);
    try testing.expect(std.mem.indexOf(u8, none, "no prefill chunk down to the min of 512 rows holds it: lower --ctx-size") != null);
}

test "static bill: a context one byte short of the room names a need above the room it names" {
    var buf: [640]u8 = undefined;
    const line = contextRefusalLine(&buf, .{ .context = .{ .tokens = 65536, .source = .setting }, .needed = 8000 * mb + 1, .bill = 40000 * mb, .budget = 48000 * mb, .room = 8000 * mb, .fits = 65535, .chunk = 8192, .min = 512, .rung = null });
    try testing.expect(std.mem.indexOf(u8, line, "needs 8001 MB beside the model bill") != null);
    try testing.expect(std.mem.indexOf(u8, line, "leaves 8000 MB") != null);
}
