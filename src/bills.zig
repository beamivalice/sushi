//! Bills, admission, context and chunk math. It reads the GPU only through gpu_mem.zig, so a test that reaches no
//! live reading runs in the host tier.
const std = @import("std");
const arch_config = @import("arch/config.zig");
const settings_resolve = @import("settings_resolve.zig");
const trunk = @import("trunk.zig");
const verify_moe = @import("verify_moe.zig");
const attn_kernels = @import("attn_kernels.zig");
const nax_gate = @import("nax_gate.zig");
const qwen4_qsa_select = @import("qwen4/qwen4_qsa_select.zig");
const qwen4_qsa_gather = @import("qwen4/qwen4_qsa_gather.zig");
const qwen4_qsa_attn = @import("qwen4/qwen4_qsa_attn.zig");
const generate_mod = @import("generate.zig");
const fp8_block = @import("fp8_block.zig");
const sushi_exl3 = @import("sushi_exl3");
const model_mod = @import("model.zig");
const dsv41_config = @import("dsv41/dsv41_config.zig");
const glm5_prefix = @import("glm5/glm5_prefix.zig");
const dsv41_prefix = @import("dsv41/dsv41_prefix.zig");
const dsv41_borrow = @import("dsv41/dsv41_borrow.zig");
const log = @import("log.zig");
const prefix_cache_mod = @import("prefix_cache.zig");
const kv_disk_cache = @import("kv_disk_cache.zig");
const disk_placement = @import("disk_placement.zig");
const scheduler_mod = @import("scheduler.zig");
const model_load = @import("model_load.zig");
const slot_admission = @import("slot_admission.zig");
const model_registry_mod = @import("model_registry.zig");
const model_settings = @import("model_settings.zig");
const metrics = @import("status.zig");
const qwen4_mod = @import("qwen4/qwen4_exp.zig");
const Trunk = trunk.Trunk;
const request_policy = @import("request_policy.zig");
const glm5_generate = @import("glm5/glm5_generate.zig");
const mtp_plan = @import("mtp_plan.zig");
const prefill_plan = @import("prefill_plan.zig");
const testing = std.testing;
const expert_stream_mod = @import("expert_stream.zig");
const mtp_mod = @import("mtp.zig");
const gpu_mem = @import("gpu_mem.zig");
const fixtures = @import("bills_fixtures_test.zig");
const arch_spec = @import("arch/spec.zig");
const arch_registry = @import("arch/registry.zig");
pub const launch_state = @import("server_state.zig");

/// An arch's own bill terms live in its `bill` module (`src/<arch>/<arch>_bill.zig`, declared in its `arch.zig`). Each hook
/// takes the config first and may return null; one the arch omits leaves the generic term.
fn archHook(comptime name: []const u8, comptime T: type, config: *const model_mod.ModelConfig, args: anytype) ?T {
    const claimed = arch_spec.of(config) orelse return null;
    switch (claimed) {
        inline else => |_, tag| {
            const Bill = arch_registry.Mod(tag).bill;
            if (!@hasDecl(Bill, name)) return null;
            const out = @call(.auto, @field(Bill, name), .{config} ++ args);
            return if (@typeInfo(@TypeOf(out)) == .optional) out else @as(?T, out);
        },
    }
}

/// `--mtp` process-wide, or the model's own `"mtp": true` in `model-settings.json` when unflagged.
pub fn forceMtpFor(config: *const model_mod.ModelConfig) bool {
    return mtpChoiceFor(config).forced();
}

/// Will a request that omits `enable_mtp` run the head on this model? The load-time bills
/// (`sessionBytesPerToken`) price its KV whenever it does, the request-time bill per request.
fn mtpHeadDefaultOn(config: *const model_mod.ModelConfig) bool {
    if (config.isGlm5() or config.isDsv41()) return false;
    // A loaded DFlash2 drafter outranks the head: the default round never builds its cache.
    if (config.dflash_loaded) return false;
    const choice = mtpChoiceFor(config);
    if (choice.forced()) return true;
    return choice.on and model_mod.isServedArch(config.model_type) and (!config.expert_streaming or config.stream_mtp_head);
}

/// THIS model's MTP decision and its source, the answer the load log and `/props` read.
pub fn mtpChoiceFor(config: *const model_mod.ModelConfig) model_settings.MtpChoice {
    if (launch_state.global_scheduler) |sch| return model_load.mtpChoiceFor(sch.mtp, config);
    const flag: ?bool = launch_state.configured_mtp orelse (if (launch_state.server_config.default_force_mtp) @as(?bool, true) else null);
    return model_load.mtpChoiceFor(flag, config);
}

/// THIS model's KV scheme and its source, the one answer the load log, `/props` and every bill read.
pub fn kvCacheFor(config: *const model_mod.ModelConfig) trunk.KvCacheChoice {
    if (launch_state.global_scheduler) |sch| return kvCacheForLaunch(config, sch.kv_quant);
    return kvCacheForLaunch(config, launch_state.configured_kv_quant);
}

/// `kvCacheFor` under a given launch choice. deepseek_v41 keeps its own compressed state; `--kv-quant` has nothing to quantize.
pub fn kvCacheForLaunch(config: *const model_mod.ModelConfig, flag: ?trunk.KVQuantConfig) trunk.KvCacheChoice {
    if (config.isDsv41()) return .{ .config = .dense, .source = .default };
    return trunk.KvCacheChoice.resolve(flag, config.kv_quant_override);
}

pub fn configuredKvQuantFor(config: *const model_mod.ModelConfig) trunk.KVQuantConfig {
    return kvCacheFor(config).config;
}

/// The explicit context for THIS model: `--ctx-size`, else its `model-settings.json` `ctx_size`.
/// 0 = auto. Every reader of the manual context goes through here, never `server_config.max_context_size`.
pub fn manualContext(config: *const model_mod.ModelConfig) u32 {
    return model_settings.contextPick(launch_state.server_config.max_context_size, config.ctx_override).value;
}

/// What the hot cache was actually given for the loaded model, after `clampedPrefixCacheMem`.
/// Every post-load reserve reads it through `resolvedPrefixCacheMem()`. Atomic: written on the
/// inference thread, read from connection threads. `maxInt(u64)` is the unresolved sentinel,
/// not 0 (`--prefix-cache-mem 0` under SSD-first publishes a real zero).
const HOT_CACHE_MEM_UNRESOLVED: u64 = std.math.maxInt(u64);

var hot_cache_mem_resolved = std.atomic.Value(u64).init(HOT_CACHE_MEM_UNRESOLVED);

/// The hot-cache byte budget every post-load reserve must bill: the clamp's answer once loaded, the raw ask before.
pub fn resolvedPrefixCacheMem() u64 {
    if (launch_state.prefix_cache_capacity == 0 or !launch_state.prefix_cache_ram_enabled) return 0;
    const v = hot_cache_mem_resolved.load(.monotonic);
    return if (v != HOT_CACHE_MEM_UNRESOLVED) v else launch_state.prefix_cache_mem_bytes;
}

/// Publish the clamp's answer (inference thread, end of a load); `clearResolvedPrefixCacheMem`
/// resets it on unload so the next model does not inherit this one's budget.
pub fn publishResolvedPrefixCacheMem(v: u64) void {
    hot_cache_mem_resolved.store(v, .monotonic);
}

pub fn clearResolvedPrefixCacheMem() void {
    hot_cache_mem_resolved.store(HOT_CACHE_MEM_UNRESOLVED, .monotonic);
}

pub const DiskAsk = struct {
    bytes: u64,
    source: settings_resolve.Source,

    /// An operator named the number, by flag or in the file: it is the budget as given.
    pub fn explicit(self: DiskAsk) bool {
        return self.source != .default;
    }

    pub fn sourceName(self: DiskAsk) []const u8 {
        return if (self.source == .model_settings) "model-settings.json" else "--prefix-cache-disk";
    }
};

/// The ask as of now. `POST /v1/settings` changes it from a connection thread while loads and bills read it, so
/// every read after boot comes through here.
pub fn diskAsk() DiskAsk {
    const source = @atomicLoad(settings_resolve.Source, &launch_state.prefix_cache_disk_source, .acquire);
    return .{ .bytes = @atomicLoad(u64, &launch_state.prefix_cache_disk_bytes, .monotonic), .source = source };
}

/// Does THIS model run with an SSD tier? The launch asked for one and its load did not decline it
/// (`ModelConfig.prefix_cache_disk_declined`); every bill, budget and checkpoint policy reads this.
pub fn diskTierOn(config: *const model_mod.ModelConfig) bool {
    return diskAsk().bytes > 0 and !config.prefix_cache_disk_declined;
}

/// The disk bytes the SSD-tier-dependent policies read: the launch ask, or 0 for a model whose tier is off.
pub fn diskBytesFor(config: *const model_mod.ModelConfig) u64 {
    return if (diskTierOn(config)) diskAsk().bytes else 0;
}

/// SSM checkpoints feed reusable prefix tiers; zero capacity disables all reuse.
pub fn effectiveSsmCheckpointStride(stride: u32, cache_capacity: u32, ram_enabled: bool, disk_bytes: u64) u32 {
    if (cache_capacity == 0 or (!ram_enabled and disk_bytes == 0)) return 0;
    return stride;
}

/// Percentage of the memory-derived ceiling we actually admit as context when
/// `--ctx-size` is absent.
///
/// `computeMaxSafeContext` returns the largest context that FITS right now.
/// Serving at exactly that leaves nothing for the prefix cache to grow into, a
/// second model to load beside this one, or another app to take RAM — and the
/// Metal OOM it would eventually hit is uncatchable (see the auto-context
/// gotcha). Reserve headroom instead.
pub const auto_ctx_safety_pct: u32 = 93;

/// PURE: apply the safety margin and round down to a 1024 boundary so the
/// number reads sanely in logs and client configs. Never returns 0.
///
/// Takes the MEMORY-derived ceiling only. The model's own `max_position_embeddings`
/// is applied afterwards, un-margined: when the checkpoint's max is the binding
/// constraint there is nothing to reserve memory headroom against, and shaving
/// the margin off a 131,072-token model that comfortably fits in RAM just throws
/// context away.
pub fn safeAutoContext(raw: u32) u32 {
    const scaled: u64 = (@as(u64, raw) * auto_ctx_safety_pct) / 100;
    const rounded: u64 = (scaled / 1024) * 1024;
    if (rounded == 0) return @intCast(@max(scaled, 1));
    return @intCast(rounded);
}

/// This model's auto-context: memory ceiling with headroom, then capped by what
/// the checkpoint actually supports. `contextCap` is the rope-derived window:
/// `max_position_embeddings`, or `original_max_position_embeddings × factor` for
/// a YaRN-scaled checkpoint — the same derivation vLLM applies to
/// `max_model_len`, because a position past the scaled window aliases back
/// inside the ramp rather than reading as a longer distance.
/// The context the server advertises, from the largest context memory alone allows (93%
/// margin on the memory number, checkpoint cap afterwards). Shared by the load-time session
/// bill and the boot-time sizer so the two cannot spell the relation two ways.
pub fn autoContextFrom(memory_ctx: u32, ctx_cap: u32, ctx_default: u32) u32 {
    const with_headroom = safeAutoContext(memory_ctx);
    // The default is served only where the margined memory context reaches it: the headroom stays.
    if (ctx_default > 0 and with_headroom >= ctx_default) return ctx_default;
    return if (ctx_cap > 0) @min(with_headroom, ctx_cap) else with_headroom;
}

pub fn autoContextFor(config: *const model_mod.ModelConfig) u32 {
    if (billedAutoContext(config)) return billedAdvertisedContext(config, computeMemoryContext(config));
    return autoContextFrom(computeMemoryContext(config), config.contextCap(), config.defaultContext());
}

/// Freeze this model's auto-context at load time. Idempotent; a no-op (and
/// never a write) when `--ctx-size` pinned it explicitly. Returns the value
/// `getEffectiveContextLength` will report from now on.
///
/// Called once per model, at startup for the `--model` primary and right after
/// an on-demand load. The advertised context must not drift: agent CLIs read it
/// once (pi/opencode never re-read `/v1/models`) and budget their own
/// `max_tokens` against it for the rest of the session.
pub fn pinAutoContext(config: *model_mod.ModelConfig) u32 {
    // Order is load-bearing: the sizer bills THIS chunk's transient reserve and
    // the admission guard bills the same frozen value, so the width has to be
    // resolved before any context is computed from it. Pinned even under an
    // explicit `--ctx-size`, because the guard reads it on every request.
    _ = pinPrefillChunk(config);
    if (manualContext(config) > 0) return manualContext(config);
    if (config.pinned_context == 0) {
        config.pinned_context = autoContextFor(config);
        const want = config.defaultContext();
        if (want > 0 and config.pinned_context < want)
            log.warn("[context] this Mac's memory bill admits {d} tokens, below the {d} the model defaults to; serving {d}\n", .{ computeMemoryContext(config), want, config.pinned_context });
    }
    return config.pinned_context;
}

pub fn getEffectiveContextLength(config: *const model_mod.ModelConfig) u32 {
    if (manualContext(config) > 0) return manualContext(config);
    if (config.pinned_context > 0) return config.pinned_context;
    // Not pinned yet (a discovery stub that was never loaded): compute from
    // current GPU memory rather than a fixed 16K cap.
    return autoContextFor(config);
}

/// PURE (unit-testable): the real ceiling a NEW MLX allocation must fit under.
///
/// `working_set_limit` (Metal's `max_recommended_working_set_size`) is a STATIC
/// device maximum — it assumes the whole GPU working set is MLX's to claim and
/// is BLIND to memory held by anything else on the machine. `mlx_footprint`
/// (MLX active + reclaimable cache) + `free_system` (physically free RAM) is
/// what MLX can actually reach RIGHT NOW; when another process holds a big
/// chunk of unified memory the second term binds and the ceiling collapses.
///
/// #64 (2026-07): a Claude Code session on a 128 GB Mac ran a docker-compose
/// stack (firecrawl/rabbitmq/postgres/playwright) holding tens of GB. The guard
/// budgeted against the static 115 GB max, admitted a 90 K-token MoE prefill,
/// and Metal threw `Insufficient Memory` from the command-buffer completion
/// handler — an UNCATCHABLE async C++ throw that terminates the process.
/// Capping by real free RAM makes the two prefill guards
/// reject/clamp before that allocation is ever submitted.
pub fn physicalMemoryCeiling(working_set_limit: u64, mlx_footprint: u64, free_system: u64) u64 {
    return @min(working_set_limit, mlx_footprint +| free_system);
}

/// PURE: the floor an explicitly raised `iogpu.wired_limit_mb` puts under the ceiling; 0 when
/// the sysctl is absent or at the macOS default (75% of RAM), which leaves the ceiling as is.
/// A floor, not a cap: the working-set term already tracks the sysctl. The term it lifts is
/// the free-RAM one, which counts other processes' pages as gone; the operator who raised the
/// enforced limit has said the GPU may claim that much. `margin` (`--wired-margin-gib`)
/// is what stays unplanned under the limit: other processes' wired GPU memory, the
/// allocator cache and the estimator's slack; past the limit Metal returns zeros.
pub fn wiredLimitFloor(wired_limit: u64, total_ram: u64, margin: u64) u64 {
    if (wired_limit == 0 or total_ram == 0) return 0;
    if (wired_limit <= total_ram * 75 / 100) return 0; // the macOS default: no declaration
    return @min(wired_limit -| margin, total_ram -| margin);
}

pub fn parseWiredMarginGib(raw: []const u8) error{InvalidWiredMargin}!u64 {
    const n = std.fmt.parseInt(u32, raw, 10) catch return error.InvalidWiredMargin;
    if (n < 1 or n > 32) return error.InvalidWiredMargin;
    return @as(u64, n) << 30;
}

pub fn parseSsdBudgetGb(raw: []const u8) error{InvalidSsdBudgetGb}!u64 {
    const gb = std.fmt.parseInt(u64, raw, 10) catch return error.InvalidSsdBudgetGb;
    if (gb == 0 or gb > 512) return error.InvalidSsdBudgetGb;
    return std.math.mul(u64, gb, 1 << 30) catch return error.InvalidSsdBudgetGb;
}

/// PURE: the residency limit the SSD budget is admitted against — `iogpu.wired_limit_mb`
/// when the owner declared one, else Metal's recommended working set.
pub fn ssdBudgetCeiling(wired_limit: u64, working_set: u64) u64 {
    return if (wired_limit > 0) wired_limit else working_set;
}

pub fn ssdBudgetAdmits(budget_bytes: u64, kv_bytes: u64, limit: u64) bool {
    return budget_bytes +| kv_bytes <= limit;
}

/// PURE: the `iogpu.wired_limit_mb` value that would admit this plan — quoted by the refusal.
pub fn wiredLimitMbForBudget(budget_bytes: u64, kv_bytes: u64) u64 {
    const MB: u64 = 1024 * 1024;
    return (budget_bytes +| kv_bytes +| (MB - 1)) / MB;
}

/// GPU memory the wired limit counts that MLX does not own: other processes' Metal allocations
/// (WindowServer, every app with a layer) and the driver's. `AGXAccelerator` "In use system
/// memory" is system-wide, so this is that counter less MLX's active + cache, read once at
/// startup before MLX holds a model (`measureGpuFloor`). A command buffer fails once MLX's peak
/// passes the wired limit less this floor; above it the counter is elastic.
pub var gpu_floor_bytes = std.atomic.Value(u64).init(0);

/// What the bill may still be wrong by once the floor is planned, including sushi's own non-MLX
/// Metal objects that arrive after the startup reading.
pub const GPU_FLOOR_SLACK_BYTES: u64 = 512 << 20;

/// PURE: what stays unplanned under a declared wired limit: the operator's margin, or the
/// measured floor plus the bill's slack when that is larger.
pub fn wiredUnplannedBytes(margin: u64, gpu_floor: u64) u64 {
    if (gpu_floor == 0) return margin;
    return @max(margin, gpu_floor +| GPU_FLOOR_SLACK_BYTES);
}

/// PURE: the floor from one idle reading; 0 when IOKit gave nothing.
pub fn gpuFloorFrom(in_use_system: u64, mlx_active: u64, mlx_cache: u64) u64 {
    if (in_use_system == 0) return 0;
    return in_use_system -| (mlx_active +| mlx_cache);
}

/// The MB-per-buffer the prefill terms were measured at (`prefillStreamBytesPerToken` on an
/// M4 Max, MLX's Max/Ultra default). MLX commits a command buffer once its distinct inputs pass
/// `MLX_MAX_MB_PER_BUFFER`, and every op's inputs live until its buffer completes, up to 10
/// buffers in flight (`mlx/backend/metal/eval.cpp`, `MAX_ACTIVE_TASKS`).
pub const BILL_FIT_MB_PER_BUFFER: u64 = 50;

/// MLX's per-buffer defaults by architecture suffix (`mlx/backend/metal/device.cpp`).
pub fn mlxDefaultMbPerBuffer(arch: []const u8) u64 {
    if (arch.len == 0) return 40;
    return switch (arch[arch.len - 1]) {
        'p', 'g' => 40,
        's', 'd' => 50,
        else => 40,
    };
}

pub fn mlxDefaultOpsPerBuffer(arch: []const u8) u64 {
    if (arch.len == 0) return 40;
    return switch (arch[arch.len - 1]) {
        'p' => 20,
        'g' => 40,
        's', 'd' => 50,
        else => 40,
    };
}

/// PURE: MLX's reading of its env override (a positive integer replaces the default).
pub fn mlxPerBufferFromEnv(raw: ?[]const u8, default: u64) u64 {
    const s = raw orelse return default;
    const n = std.fmt.parseInt(u64, std.mem.trim(u8, s, " \t"), 10) catch return default;
    return if (n > 0) n else default;
}

/// PURE: prefill bytes the command-buffer cadence keeps alive beyond what the bill's fitted
/// terms already hold; ops per buffer does not move the peak. A narrower cap is not credited.
pub fn cadenceInflightBytes(mb_per_buffer: u64) u64 {
    return (mb_per_buffer -| BILL_FIT_MB_PER_BUFFER) * (1 << 20) * 9 / 8;
}

/// The effective `MLX_MAX_MB_PER_BUFFER`, resolved once at startup.
pub var mlx_mb_per_buffer: u64 = BILL_FIT_MB_PER_BUFFER;

/// PURE: the ceiling with the wired-limit floor applied. `wired_floor == 0` returns
/// `physicalMemoryCeiling` byte for byte. A declared limit IS the ceiling, its margin included:
/// the working-set term is the limit itself, so on a quiet box it would plan into the margin.
pub fn gpuCeilingWithWiredFloor(
    working_set_limit: u64,
    mlx_footprint: u64,
    free_system: u64,
    wired_floor: u64,
) u64 {
    if (wired_floor > 0) return wired_floor;
    return physicalMemoryCeiling(working_set_limit, mlx_footprint, free_system);
}

/// Test seam: `iogpu.wired_limit_mb` in MB; `null` reads the machine.
pub var wired_limit_mb_override: ?u64 = null;

var wired_limit_bytes_cached: u64 = 0;

var wired_limit_read: bool = false;

/// `iogpu.wired_limit_mb` in bytes, read ONCE per process: re-reading it per admission would
/// let the ceiling move under a live request. 0 when the OID is absent.
pub fn wiredLimitBytes() u64 {
    if (wired_limit_mb_override) |mb| return mb *| (1024 * 1024);
    if (wired_limit_read) return wired_limit_bytes_cached;
    var v: u32 = 0;
    var len: usize = @sizeOf(u32);
    wired_limit_bytes_cached = if (sysctlbyname("iogpu.wired_limit_mb", @ptrCast(&v), &len, null, 0) == 0)
        @as(u64, v) * 1024 * 1024
    else
        0;
    wired_limit_read = true;
    return wired_limit_bytes_cached;
}

/// Bytes MLX wires beside the owned weights and slabs: the decode buffer pool (`dsv41_pool_bytes`), the one
/// ledger term the owned-bytes count lacks.
pub const USER_WIRE_MARGIN_BYTES: u64 = 1 << 30;

/// RAM a suggested user-wire limit leaves to the rest of the system.
const USER_WIRE_RAM_FLOOR_BYTES: u64 = 12_000_000_000;

/// PURE: the one-line warning for a load whose wired bytes pass macOS's user-wire cap, else null.
/// The one way sushi tells a user to raise the GPU limit: the two user-wire limits move with it, or the lower of them
/// binds and prefill slows. `iogpu.wired_limit_mb` counts MiB, so the user-wire limits get `mb` MiB in bytes.
pub fn wireLimitsCommand(buf: []u8, mb: u64) []const u8 {
    const bytes = mb *| (1 << 20);
    return std.fmt.bufPrint(buf, "sudo sysctl iogpu.wired_limit_mb={d} vm.global_user_wire_limit={d} vm.user_wire_limit={d}", .{ mb, bytes, bytes }) catch "";
}

/// The GPU limit, in MB, that holds `bytes`: rounded up to 100 so the command reads cleanly.
pub fn wireLimitMbFor(bytes: u64) u64 {
    const step: u64 = 100 << 20;
    return (bytes +| (step - 1)) / step * 100;
}

/// `limits` are `vm.global_user_wire_limit` and `vm.user_wire_limit` (0 = unreadable); the lower binds.
/// Past it prefill slows down with no error. The suggestion matches a declared `iogpu.wired_limit_mb` that holds the
/// load; otherwise it is the need rounded up to 100 MB, never above `total_ram` less a floor.
pub fn userWireWarning(buf: []u8, owned_bytes: u64, limits: [2]u64, wired_limit: u64, total_ram: u64) ?[]const u8 {
    const names = [2][]const u8{ "vm.global_user_wire_limit", "vm.user_wire_limit" };
    const bind: usize = if (limits[0] == 0 or (limits[1] != 0 and limits[1] < limits[0])) 1 else 0;
    const limit = limits[bind];
    if (limit == 0) return null;
    const need = owned_bytes +| USER_WIRE_MARGIN_BYTES;
    if (need <= limit) return null;
    const head = "[mem] warning: this load wires ~{d:.1} GB but {s} is {d:.1} GB: prefill will run slower. ";
    const args = .{ @as(f64, @floatFromInt(need)) / 1e9, names[bind], @as(f64, @floatFromInt(limit)) / 1e9 };
    // A declared GPU limit that holds the load is the user's choice: the user-wire limits come up to match it.
    const mb = if (wired_limit >= need) wired_limit >> 20 else wireLimitMbFor(need);
    if (wired_limit < need and mb << 20 > total_ram -| USER_WIRE_RAM_FLOOR_BYTES)
        return std.fmt.bufPrint(buf, head ++ "This Mac cannot wire that much: use a smaller pack or a lower --ssd-budget-gb", args) catch null;
    var cmd: [192]u8 = undefined;
    return std.fmt.bufPrint(buf, head ++ "Raise it: {s} (resets at reboot)", args ++ .{wireLimitsCommand(&cmd, mb)}) catch null;
}

fn readSysctlU64(name: [*:0]const u8) u64 {
    var v: u64 = 0;
    var len: usize = @sizeOf(u64);
    return if (sysctlbyname(name, @ptrCast(&v), &len, null, 0) == 0 and len == @sizeOf(u64)) v else 0;
}

var user_wire_warned = std.atomic.Value(bool).init(false);

/// Warns once per process, after the weights and expert slabs are mapped, when they pass the user-wire cap.
pub fn warnUserWireLimit() void {
    if (user_wire_warned.cmpxchgStrong(false, true, .monotonic, .monotonic) != null) return;
    var buf: [384]u8 = undefined;
    const limits = [2]u64{ readSysctlU64("vm.global_user_wire_limit"), readSysctlU64("vm.user_wire_limit") };
    if (userWireWarning(&buf, model_mod.ownedBytesTotal(), limits, wiredLimitBytes(), metrics.getTotalMemBytes())) |line|
        log.warn("{s}\n", .{line});
}

/// The limits a default `--ssd-budget-gb` must fit: the residency limit (the declared wired limit, else Metal's working
/// set) and the lower user-wire sysctl, each less what stays unplanned under it.
pub fn defaultBudgetLimits() model_load.BudgetLimits {
    const wire = [2]u64{ readSysctlU64("vm.global_user_wire_limit"), readSysctlU64("vm.user_wire_limit") };
    const wire_bind: usize = if (wire[0] == 0 or (wire[1] != 0 and wire[1] < wire[0])) 1 else 0;
    return .{
        .wired_limit = wiredLimitBytes(),
        .ceiling = ssdBudgetCeiling(wiredLimitBytes(), staticGpuMemoryCeiling()),
        .ceiling_name = if (wiredLimitBytes() > 0) "iogpu.wired_limit_mb" else "the GPU working-set limit",
        .margin = launch_state.wired_limit_margin_bytes,
        .user_wire = wire[wire_bind],
        .user_wire_name = if (wire_bind == 0) "vm.global_user_wire_limit" else "vm.user_wire_limit",
        .user_wire_margin = USER_WIRE_MARGIN_BYTES,
    };
}

/// What a request at the model's default context bills beside the ssd budget with its prefill borrowed from the decoder
/// pools (`dsv41-memory`): the state, the restore points in flight and a decode block's working set.
pub fn defaultRequestBill(config: *const model_mod.ModelConfig) u64 {
    var probe = config.*;
    probe.expert_streaming = false;
    probe.expert_borrow_bytes = std.math.maxInt(u64) / 2;
    return prefillNeededAtChunk(&probe, config.defaultContext(), 0, 16, PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1], .{});
}

var wired_floor_logged = std.atomic.Value(bool).init(false);

/// Long-context models and engaged expert streams honor the operator's wired limit.
/// `null` config (no model resolved yet) takes no floor.
pub fn wiredCeilingFloorFor(config: ?*const model_mod.ModelConfig) u64 {
    return wiredCeilingFloorForRam(config, metrics.getTotalMemBytes());
}

/// PURE: the floor for a machine with `total_ram` bytes; the wrapper above reads the machine.
pub fn wiredCeilingFloorForRam(config: ?*const model_mod.ModelConfig, total_ram: u64) u64 {
    const c = config orelse return 0;
    // Resident MiMo fills a 128 GB Mac, so its ceiling is the declared limit, not what other apps leave free.
    if (!c.longCtxGated() and !c.expert_streaming and !c.isMimo() and !c.isGlm5()) return 0;
    const unplanned = wiredUnplannedBytes(launch_state.wired_limit_margin_bytes, gpu_floor_bytes.load(.monotonic));
    const floor = wiredLimitFloor(wiredLimitBytes(), total_ram, unplanned);
    if (floor > 0 and wired_floor_logged.cmpxchgStrong(false, true, .monotonic, .monotonic) == null) {
        log.info("[mem] ceiling {d} MB from iogpu.wired_limit_mb={d} (working set {d} MB, margin {d} MB, GPU floor {d} MB, unplanned {d} MB)\n", .{
            floor >> 20,
            wiredLimitBytes() >> 20,
            gpu_mem.getGpuWorkingSetLimit() >> 20,
            launch_state.wired_limit_margin_bytes >> 20,
            gpu_floor_bytes.load(.monotonic) >> 20,
            unplanned >> 20,
        });
    }
    return floor;
}

/// PURE (unit-testable): the cap to put on MLX's reclaimable buffer pool for a
/// machine with `total_ram` bytes of physical memory. 0 when the RAM query
/// failed — never clamp on bad data.
///
/// MLX's own default is `min(1.5 x working_set, 0.95 x RAM)` with a GC limit at
/// `0.95 x working_set` (`backend/metal/allocator.cpp`) — ~121 GB / ~91 GB on a
/// 128 GB Mac, i.e. it does not trim until the machine is already dead. Freed
/// buffers are parked in that pool instead of going back to the OS, so anything
/// that frees never-repeating sizes in a loop (KV growth, decode transients)
/// grows the process footprint without moving `active_bytes` at all. Issue #110:
/// 81.4 GB in Activity Monitor against 19.6 GB in the panel.
///
/// RAM/16 is big enough for the step-to-step buffer reuse the pool exists for
/// (8 GB on a 128 GB Mac) and small enough that it can never be the footprint.
/// The 2 GB floor keeps 8/16 GB machines from thrashing the allocator.
pub fn mlxCacheLimitBytes(total_ram: u64) u64 {
    if (total_ram == 0) return 0;
    const GB: u64 = 1 << 30;
    return @max(2 * GB, @min(8 * GB, total_ram / 16));
}

/// Impure wrapper: current GPU allocation ceiling from live MLX + system
/// counters. `mlx_footprint` = active (in-use) + cache (reclaimable) so an
/// idle machine's ceiling stays ≈ the static device max (no auto-context
/// regression), while external pressure — reflected in `getAvailableMemBytes`
/// (free + file-backed pages, capped by the RAM outside wired + compressor) — tightens it. See
/// `physicalMemoryCeiling`.
/// The ceiling term that does not move with instantaneous free RAM: Metal's recommended
/// working set (or the wired limit). The load-time hot-cache clamp bills against this and
/// nothing else: two boots 11 minutes apart resolved the same ask to 1076 and 9757 MB off the
/// live term. Request-time admission still reads live memory.
/// Tests only: stand in for the machine's working-set limit (CI runners have 7 GB).
pub var static_ceiling_override: ?u64 = null;

pub fn staticGpuMemoryCeiling() u64 {
    return static_ceiling_override orelse gpu_mem.getGpuWorkingSetLimit();
}

/// THE ceiling helper: `available`, `/props`, the hot-cache clamp, the auto-context pin and the
/// admission bill all read it, so the wired-limit floor cannot reach some of them and not others.
pub fn currentGpuMemoryCeiling(config: ?*const model_mod.ModelConfig, active_mem: u64) u64 {
    const cache_mem: usize = gpu_mem.cacheBytes();
    return gpuCeilingWithWiredFloor(
        gpu_mem.getGpuWorkingSetLimit(),
        active_mem +| @as(u64, cache_mem),
        metrics.getAvailableMemBytes(),
        wiredCeilingFloorFor(config),
    );
}

pub fn expertStreamingLoadFits(ceiling: u64, active_mem: u64, resident_bytes: u64, cache_bytes: u64, bounce_bytes: u64, fill_peak_bytes: u64, serving_bytes: u64) bool {
    const needed = active_mem +| resident_bytes +| cache_bytes +| bounce_bytes +| fill_peak_bytes +| serving_bytes;
    return needed <= ceiling;
}

pub fn expertStreamingPlannedSeq(config: *const model_mod.ModelConfig) u64 {
    const manual = manualContext(config);
    return if (manual > 0) manual else @min(config.contextCap(), 65_536);
}

/// The KV a streamed load plans to hold: the second term of the
/// budget-vs-wired-limit admission.
pub fn expertStreamingPlannedKvBytes(config: *const model_mod.ModelConfig) u64 {
    const kv_bits = defaultKvBits(config);
    return sessionBytesPerToken(config, kv_bits) *| expertStreamingPlannedSeq(config) +| slotRingBytes(config, kv_bits);
}

fn expertStreamingServingBytes(config: *const model_mod.ModelConfig) u64 {
    const seq: u64 = expertStreamingPlannedSeq(config);
    const explicit = explicitPrefillChunk();
    // A request picks its own rung against free memory, and an unpinned arch's load-time pin is memory-sized after
    // this check, so the load must only prove the ladder's floor fits.
    const floor: u64 = PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1];
    const chunk: u64 = if (explicit > 0) (perRequestFloorWidth(config) orelse explicit) else if (config.pinned_prefill_chunk > 0) config.pinned_prefill_chunk else floor;
    // The load's own staging term already holds the decode pool.
    return prefillNeededAtChunk(config, @max(seq, chunk), 2048, defaultKvBits(config), chunk, .{}) -| config.expert_fill_peak_bytes -| config.expertPoolBytes();
}

/// What a streamed load holds beside its ssd budget: the planned KV and, under `--vision`, its largest image encode.
pub fn expertStreamingBesideBudgetBytes(config: *const model_mod.ModelConfig) u64 {
    return expertStreamingPlannedKvBytes(config) +| config.expert_vision_encode_bytes;
}

pub fn expertCacheFitForLoad(config: *const model_mod.ModelConfig, resident_bytes: u64) !void {
    if (!config.expert_streaming) return;
    if (config.expert_ssd_budget_bytes > 0) {
        const beside = expertStreamingBesideBudgetBytes(config);
        const limit = ssdBudgetCeiling(wiredLimitBytes(), staticGpuMemoryCeiling());
        if (!ssdBudgetAdmits(config.expert_ssd_budget_bytes, beside, limit)) {
            var cmd: [192]u8 = undefined;
            log.err("--ssd-budget-gb {d} plus {d:.2} GB of planned KV and {d:.2} GB of image encode exceeds the {d:.2} GB residency limit; raise the limits ({s}) or lower the budget\n", .{
                config.expert_ssd_budget_bytes >> 30,
                @as(f64, @floatFromInt(expertStreamingPlannedKvBytes(config))) / 1e9,
                @as(f64, @floatFromInt(config.expert_vision_encode_bytes)) / 1e9,
                @as(f64, @floatFromInt(limit)) / 1e9,
                wireLimitsCommand(&cmd, wireLimitMbFor(wiredLimitMbForBudget(config.expert_ssd_budget_bytes, beside) << 20)),
            });
            return error.SsdBudgetExceedsWiredLimit;
        }
    }
    const active_mem: usize = gpu_mem.activeBytes();
    const ceiling = currentGpuMemoryCeiling(config, active_mem);
    if (!expertStreamingConfigFits(config, ceiling, active_mem, resident_bytes)) {
        log.err("the streamed load bills {d} MB (active {d}, resident {d}, cache {d}, workspace and bounce {d}, serving {d}) against {d} MB; lower --ssd-budget-gb\n", .{
            (active_mem + resident_bytes + config.expert_cache_bytes + config.expert_bounce_bytes + model_load.loadStagingBytes(config, config.expert_fill_peak_bytes, config.expert_workspace_bytes) + @max(expertStreamingServingBytes(config), config.expert_vision_encode_bytes)) >> 20,
            active_mem >> 20,
            resident_bytes >> 20,
            config.expert_cache_bytes >> 20,
            (config.expert_bounce_bytes + model_load.loadStagingBytes(config, config.expert_fill_peak_bytes, config.expert_workspace_bytes)) >> 20,
            @max(expertStreamingServingBytes(config), config.expert_vision_encode_bytes) >> 20,
            ceiling >> 20,
        });
        return error.ExpertCacheDoesNotFit;
    }
}

fn expertStreamingConfigFits(config: *const model_mod.ModelConfig, ceiling: u64, active_mem: u64, resident_bytes: u64) bool {
    return expertStreamingLoadFits(
        ceiling,
        active_mem,
        resident_bytes,
        config.expert_cache_bytes,
        config.expert_bounce_bytes,
        model_load.loadStagingBytes(config, config.expert_fill_peak_bytes, config.expert_workspace_bytes),
        @max(expertStreamingServingBytes(config), config.expert_vision_encode_bytes),
    );
}

pub fn rowSsdBudgetGb(effective_bytes: u64, setting_gb: u32) u32 {
    if (effective_bytes > 0) return @intCast(effective_bytes >> 30);
    return setting_gb;
}

/// PURE budget math (no MLX/Metal calls — unit-testable): the largest context
/// length whose KV cache + per-token working set fits under the GPU
/// working-set ceiling, after subtracting what's already resident
/// (`active_mem` — model weights + any resident hot-cache KV) AND the hot
/// prefix cache's FULL byte budget (`hot_cache_reserve`).
///
/// Reserving the whole cache budget — not just its current residency — budgets
/// for STEADY STATE. Over an agentic session the hot cache fills to its cap, so
/// an auto-context computed against an empty cache (24k tokens on a 16 GB Mac)
/// later collides with the filled 2 GB cache plus a
/// large cold MoE prefill and the command buffer OOMs. Subtracting it up front
/// keeps the reported context stable across the session and within the ceiling
/// once the cache is full. The 0.64 factor (two ×4/5 margins) absorbs the
/// prefill compute spike that lands on top of `active_mem`.
/// PURE: the raw largest context the budget fits — no 1024 floor, no
/// `max_pos` clamp. Split out of `safeContextForBudget` because
/// `resolvePrefillChunk` compares rungs against each other and the floor makes
/// a rung that fits NOTHING look identical to one that fits 1024 tokens.
/// The 0.64 factor is the same two x4/5 margins documented below.
fn contextForBudget(working_set_limit: u64, active_mem: u64, hot_cache_reserve: u64, per_tok: u64) u64 {
    if (per_tok == 0) return 0;
    const spoken_for: u64 = active_mem +| hot_cache_reserve;
    const usable: u64 = if (working_set_limit > spoken_for) working_set_limit - spoken_for else 0;
    return ((usable * 4 / 5) * 4 / 5) / per_tok;
}

fn safeContextForBudget(
    working_set_limit: u64,
    active_mem: u64,
    hot_cache_reserve: u64,
    per_tok: u64,
    max_pos: u32,
) u32 {
    if (per_tok == 0) return 1024;
    const max_seq: u64 = contextForBudget(working_set_limit, active_mem, hot_cache_reserve, per_tok);
    if (max_seq < 1024) return 1024;

    var result: u32 = if (max_seq > std.math.maxInt(u32)) std.math.maxInt(u32) else @intCast(max_seq);
    // Cap at model's max position embeddings.
    if (max_pos > 0) result = @min(result, max_pos);
    return result;
}

/// The process-wide KV width a request gets when it names none — the same
/// expression `checkAttentionMemory` falls back to, so the sizer and the
/// admission guard read one answer.
///
/// Through `configuredKvQuant`, so a load-time bill gets the boot's `--kv-quant`, not the dense fallback.
pub fn defaultKvBits(config: *const model_mod.ModelConfig) u64 {
    const cfg: trunk.KVQuantConfig = configuredKvQuantFor(config);
    return if (cfg.scheme == .off) 16 else cfg.bits;
}

/// The prefill working set that is NOT the KV cache: score scratch, the
/// quantized-KV dequant, and the QKV/MLP transient envelope — all bounded by
/// the prefill CHUNK, at the widest chunk any prompt on this model can run.
///
/// It is a one-off reserve, not a per-token cost. `computeMemoryContext` used
/// to bill the MLP envelope (`8 × max(hidden, ffn) × 2`) against every token of
/// CONTEXT: 272 KB/token beside a 64 KB/token KV cache, i.e.
/// 81% of the budget spent on a transient that never scales with the context
/// length, which caps a 16 GB Mac under 4k tokens whatever the weights cost.
/// The peak above steady state is flat in prompt length and tracks the chunk.
///
/// Same estimator as the admission guard with the KV term zeroed, because the
/// KV cache is exactly what the sizer is solving for.
pub fn prefillTransientReserve(config: *const model_mod.ModelConfig, kv_bits: u64, chunk: u64) u64 {
    return prefillTransientReserveAtKv(config, kv_bits, chunk, chunk);
}

/// `prefillTransientReserve` with the KV length made explicit. The load-time sizer asks at
/// `kv = chunk`; a per-chunk decision has a KV, and the attention score sheet and the QSA
/// sheet are linear in it (~0.5 GB under-billed at the default sheet, ~9 GB at 4096 MB).
pub fn prefillTransientReserveAtKv(
    config: *const model_mod.ModelConfig,
    kv_bits: u64,
    chunk: u64,
    kv_len: u64,
) u64 {
    const seq: u64 = @max(kv_len, chunk);
    if (archHook("transientReserve", u64, config, .{ kv_bits, chunk, seq })) |reserve| return reserve;
    return prefillMemoryNeeded(
        seq,
        config.num_attention_heads,
        config.num_key_value_heads,
        0,
        config.head_dim,
        config.prefillScoreHeadDim(),
        config.hidden_size,
        prefillFfnWidth(config),
        kv_bits,
        chunk,
        config.prefillAttnKeys(seq),
        prefillStreamBytesPerToken(config),
        prefillDequantWeightBytes(config),
        .{
            .state_bytes = if (mtpHeadDefaultOn(config)) mtpHeadRingBytes(config) else 0,
            .qsa_ring_bytes = slotRingBytes(config, kv_bits),
            .dequant_scratch_bytes = kvDequantScratchBytes(config, seq, @min(chunk, seq)),
            .uplift_16ths = billUplift16ths(config, seq),
        },
    ) + qsaMaskBytes(config, @min(chunk, @max(seq, 1)), seq) +
        slidingBandScoreBytes(config, @min(chunk, @max(seq, 1))) +
        fp8DequantScratchBytes(config, @min(chunk, @max(seq, 1))) +
        trunkDecodeScratchBytes(config, @min(chunk, @max(seq, 1))) +
        (if (config.expert_streaming) config.expert_fill_peak_bytes else 0) +
        cadenceInflightBytes(mlx_mb_per_buffer);
}

/// The composed score sheet a ringed arch's SLIDING layers build where
/// `sushi_attn_pd` does not serve their band + sink attention. Their attention
/// is a band, so one layer's sheet is `[heads, fwd, window + fwd - 1]` whatever
/// the prompt length. Billed at `MOE_PREFILL_COEXIST` layers, the same
/// coexistence the other per-layer prefill transients are billed at. Zero
/// exactly where the dispatch's own predicate (`slidingPrefillFused`) holds.
pub fn slidingBandScoreBytes(config: *const model_mod.ModelConfig, fwd: u64) u64 {
    if (config.swaRingTokens() == 0) return 0;
    if (trunk.slidingPrefillFused(config, fwd)) return 0;
    const keys: u64 = @as(u64, config.sliding_window) +| fwd -| 1;
    return MOE_PREFILL_COEXIST *| @as(u64, config.num_attention_heads) *| fwd *| keys *| 2;
}

/// The bf16 copy a source FP8 linear dequantizes into when a forward is wider than its GEMV.
pub fn fp8DequantScratchBytes(config: *const model_mod.ModelConfig, fwd: u64) u64 {
    return archHook("fp8Scratch", u64, config, .{fwd}) orelse 0;
}

/// The f16 weight a trellis trunk linear decodes into when a forward is wide enough for MLX's GEMM
/// (`trunk_gemm.decodedServes`).
pub fn trunkDecodeScratchBytes(config: *const model_mod.ModelConfig, fwd: u64) u64 {
    if (config.trunk_quant == null or fwd < @as(u64, @intCast(sushi_exl3.trunk_gemm.DECODED_MIN_ROWS))) return 0;
    return sushi_exl3.trunk_gemm.DECODED_SCRATCH_BYTES;
}

/// The most keys a decode tick rebuilds dense, below the packed decode arms' floor; each arch whose decode kernels
/// have such a floor pins its own against it.
pub const DECODE_REBUILD_KEYS_MAX: u64 = 4095;

/// Dense bytes ONE layer's `KVCache.denseView` rebuild costs at this prompt
/// length — the quantized-KV prefill transient, which the eval cadence bounds
/// to a layer at a time. Zero unless the generic `2 · seq · kv_heads · hdim ·
/// 2` cannot say it: on a ringed arch the sliding layers rebuild a WINDOW, and
/// the layers that rebuild the sequence have their own head count and two
/// different K and V widths (mimo_v2: 4 heads of 192+128, against the generic
/// 8 heads of 192 twice — 2.4x).
pub fn kvDequantScratchBytes(config: *const model_mod.ModelConfig, seq: u64, fwd: u64) u64 {
    // A QSA layer rebuilds only what a forward of this width cannot gather in place.
    const qsa_rows: ?u64 = if (config.longCtxGated())
        qwen4_qsa_attn.qsaDenseRebuildRows(config.indexer_budget, config.indexer_compress_ratio, fwd)
    else
        null;
    if (config.swaRingTokens() == 0 and qsa_rows == null) return 0;
    // A layer whose prefill attention reads the PACKED cache one dispatch at a
    // time (`transformer.fusedSdpaPrefillKv`) never rebuilds more than a
    // dispatch's keys, at `MOE_PREFILL_COEXIST` of them — the dispatches are
    // serially dependent through the softmax carry, so this is the ceiling.
    // The kill switch takes the width off `prefillHeadDimFused` and the whole
    // rebuild comes back with it. A decode rebuilds whole below the packed arms' floor.
    const seq_rows: u64 = if (qsa_rows) |rows| rows else if (trunk.prefillHeadDimFused(config.prefillScoreHeadDim()))
        @max(MOE_PREFILL_COEXIST *| @as(u64, @intCast(trunk.PACKED_KV_SLICE_MAX)), DECODE_REBUILD_KEYS_MAX)
    else
        seq;
    var widest: u64 = 0;
    var li: u32 = 0;
    while (li < config.num_hidden_layers) : (li += 1) {
        const rows: u64 = if (config.isKvPerTokenLayer(li)) @min(seq, seq_rows) else @min(seq, config.swaRingTokens());
        widest = @max(widest, rows *| config.layerKvBytes(li));
    }
    return widest;
}

/// Bytes per (query, key) the QSA prefill holds for ONE live layer past the
/// indexer budget: the `[S, kv]` bool mask, its additive copy for the sdpa
/// arm, and the f32 indexer scores over kv/4 blocks (4 bytes/block = 1/key).
const QSA_MASK_BYTES_PER_KEY: u64 = 4;

/// qwen4_exp sparse attention materializes a `[S, kv]` mask per layer that
/// no envelope term models: 4096 x 25k keys = 410 MB, and a tight box met it
/// as a Metal OOM at 25k+. Zero for every arch without an indexer. The sizer
/// bills it at kv = chunk (no prompt yet); the admission guard at the real
/// prompt length.
pub fn qsaMaskBytes(config: *const model_mod.ModelConfig, fwd: u64, kv: u64) u64 {
    if (config.indexer_budget == 0) return 0;
    // Prefill widths gather by block index (row-chunked score sheet, no
    // [S, kv] mask); decode/verify widths still build the dense mask.
    if (fwd >= trunk.FUSED256_MIN_Q_LEN and qwen4_qsa_gather.qsaGatherEnabled())
        return qwen4_qsa_attn.qsaPrefillTransientBytes(config.indexer_n_heads, fwd, kv, config.indexer_compress_ratio, config.indexer_head_dim);
    return QSA_MASK_BYTES_PER_KEY * fwd * kv * 5 / 4;
}

/// Widths `resolvePrefillChunk` will step down through. Descending, floored at
/// `prefill_plan.PREFILL_CHUNK_FLOOR` — below that the score-budget path refuses to
/// go either, and a 256-token forward stops amortizing the per-chunk sweeps.
pub const PREFILL_CHUNK_LADDER = [_]u32{ 8192, 4096, 2048, 1024, 512 };

/// How much of the post-weights serving budget a ONE-OFF prefill transient may
/// claim before the chunk steps down. A quarter: past that the machine is
/// trading the whole session's context for one forward's speed, which is the
/// trade that reported a 1024-token context on a 16 GB Mac.
const PREFILL_RESERVE_BUDGET_SHARE: u64 = 4;

/// The hot prefix cache's smallest useful budget: a prefill chunk that would price the cache
/// under this steps down instead. Stops `requested == 0` from resolving to the 1-byte floor.
const HOT_CACHE_FLOOR_BYTES: u64 = 1024 * 1024 * 1024;

/// PURE: the widest prefill chunk THIS machine can afford for THIS model.
///
/// The chunk multiplies the biggest term in the memory bill, and until now
/// nothing tied it to the machine: `--prefill-chunk` defaults to 8192, the app
/// never passes the flag, so a 16 GB Mac reserved the same 5-7 GB MLP envelope
/// a 128 GB one does. Measured on a 16 GB profile (Mistral-7B-4bit, weights
/// 3.80 GB against an 11.9 GB Metal ceiling): the reserve at chunk 8192 alone
/// exceeded what was left after the weights, so the sizer reported 1024 tokens
/// and the admission guard refused a 10k-token prompt for "needing" 9.2 GB —
/// a prompt whose measured peak is 2.39 GB. Stepping to chunk 512 costs prefill
/// throughput and buys 22k tokens of context back.
///
/// Never raises anything: `effectivePrefillChunk` takes the MIN of this and the
/// launch chunk, and an explicit `--prefill-chunk` outranks it entirely.
///
/// It does not read the hot-cache ask: the clamp downstream subtracts this chunk's reserve
/// from the cache's headroom, so feeding the ask in here was non-monotone (a bigger
/// `--prefix-cache-mem` stepped the chunk down and handed the cache MORE bytes). The chunk is
/// sized against the serving budget alone; the cache is the residual claimant. `ctx_kv_bytes`
/// is the KV a pinned context has already spoken for (0 under auto-context).
/// The byte cap one prefill forward's transient may claim, and the one place the sizer's two
/// changes (dropping the ask, adding the ctx bar) are gated by arch; every other arch keeps
/// the previous arithmetic.
pub fn prefillChunkCap(
    config: *const model_mod.ModelConfig,
    ceiling: u64,
    active_mem: u64,
    ctx_kv_bytes: u64,
    hot_cache_ask: u64,
) u64 {
    if (!config.longCtxGated()) {
        // Ungated: the ask is spoken for, and a quarter of what is left is the whole cap.
        const spoken_for: u64 = active_mem +| hot_cache_ask;
        return (ceiling -| spoken_for) / PREFILL_RESERVE_BUDGET_SHARE;
    }
    const serving_budget: u64 = ceiling -| active_mem;
    // Two ask-independent bars: a share of the serving budget, and whatever the pinned context leaves less a cache worth having.
    const share_bar: u64 = serving_budget / PREFILL_RESERVE_BUDGET_SHARE;
    // The ctx bar can only ever narrow; `--ctx-size` is a cap, not an allocation, so it can pin
    // chunk 512 for the life of a process. The narrowing is logged.
    const ctx_bar: u64 = (serving_budget -| ctx_kv_bytes) -| HOT_CACHE_FLOOR_BYTES;
    return @min(share_bar, ctx_bar);
}

pub fn resolvePrefillChunk(
    config: *const model_mod.ModelConfig,
    kv_bits: u64,
    ceiling: u64,
    active_mem: u64,
    ctx_kv_bytes: u64,
    hot_cache_ask: u64,
) u32 {
    if (archHook("neutralChunkWidth", u32, config, .{})) |width| return contextNeutralChunk(config, kv_bits, ceiling, active_mem, width);
    const cap: u64 = prefillChunkCap(config, ceiling, active_mem, ctx_kv_bytes, hot_cache_ask);
    for (PREFILL_CHUNK_LADDER) |chunk| {
        if (prefillTransientReserve(config, kv_bits, chunk) <= cap) return chunk;
    }
    // Nothing fits the share — the model barely fits at all. Take the narrowest
    // rung: it is the smallest bill this box can be asked for, and returning 0
    // would read as "not pinned" and hand the forward the launch width.
    return PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1];
}

/// The widest rung, at most `width` (the arch's numerics width), that advertises the same auto context as the
/// narrowest: the bill `max_safe_context` and admission charge decides, not a share of free memory. GLM and DeepSeek,
/// whose per-token state is small beside their prefill transient.
fn contextNeutralChunk(config: *const model_mod.ModelConfig, kv_bits: u64, ceiling: u64, active_mem: u64, width: u32) u32 {
    const narrowest = PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1];
    const floor = advertisedContextAt(config, kv_bits, ceiling, active_mem, narrowest);
    for (PREFILL_CHUNK_LADDER) |chunk| {
        if (chunk > width) continue;
        if (advertisedContextAt(config, kv_bits, ceiling, active_mem, chunk) >= floor) return chunk;
    }
    return narrowest;
}

fn advertisedContextAt(config: *const model_mod.ModelConfig, kv_bits: u64, ceiling: u64, active_mem: u64, chunk: u64) u32 {
    return autoContextFrom(memoryContextAtChunk(config, kv_bits, ceiling, active_mem, chunk), config.contextCap(), config.defaultContext());
}

/// The width `--prefill-chunk` asked for, or 0. The bill must let it outrank the pin as the
/// forward does: `--prefill-chunk 4096` used to run against a 512-token reserve.
pub fn explicitPrefillChunk() u32 {
    if (settings_resolve.prefill_chunk.source != .flag) return 0;
    const raw: usize = @min(settings_resolve.prefill_chunk.value, @as(usize, std.math.maxInt(u32)));
    return @intCast(raw);
}

/// The chunk this model is billed at, and therefore runs at. `chunk_override` (0 = none)
/// mirrors `prefill_plan.effectivePrefillChunk`'s precedence.
pub fn billedPrefillChunk(
    config: *const model_mod.ModelConfig,
    kv_bits: u64,
    ceiling: u64,
    active_mem: u64,
    ctx_kv_bytes: u64,
    hot_cache_ask: u64,
    chunk_override: u32,
) u32 {
    if (chunk_override > 0) return chunk_override;
    return resolvePrefillChunk(config, kv_bits, ceiling, active_mem, ctx_kv_bytes, hot_cache_ask);
}

/// The context KV the chunk sizer must leave standing: the pinned context's bill under an
/// explicit `--ctx-size`, 0 while the context is auto (the auto sizer adapts to the rung).
pub fn sizerCtxKvBytes(config: *const model_mod.ModelConfig, kv_bits: u64) u64 {
    if (manualContext(config) == 0) return 0;
    return sessionBytesPerToken(config, kv_bits) *| manualContext(config) +| slotRingBytes(config, kv_bits);
}

/// The full-context admission bill at the width a requested launch can run.
/// An explicit prefill width is priced as requested; auto picks the per-request floor.
fn sushiContextBytesAt(config: *const model_mod.ModelConfig, context: u32, kv_bits: u64) struct { bytes: u64, chunk: u64 } {
    const explicit = explicitPrefillChunk();
    const rung: u32 = if (explicit > 0) explicit else perRequestFloorWidth(config) orelse
        (if (config.pinned_prefill_chunk > 0) config.pinned_prefill_chunk else 8192);
    const chunk = rungWidth(config, context, rung, config.longCtxGated());
    return .{
        .bytes = prefillNeededAtChunk(config, context, 0, kv_bits, chunk, .{ .mtp_on = mtpHeadDefaultOn(config) }),
        .chunk = chunk,
    };
}

/// PURE with respect to memory: solve the same admission bill used to check the requested context.
pub fn maxSushiContextForBudget(config: *const model_mod.ModelConfig, kv_bits: u64, budget: u64) u32 {
    return maxSushiContextAtWidth(config, kv_bits, budget, null, .{ .mtp_on = mtpHeadDefaultOn(config) });
}

fn maxSushiContextAtWidth(config: *const model_mod.ModelConfig, kv_bits: u64, budget: u64, fixed_chunk: ?u64, warm: WarmPrefix) u32 {
    return maxSushiContextUpTo(config, if (config.contextCap() > 0) config.contextCap() else std.math.maxInt(u32), kv_bits, budget, fixed_chunk, warm);
}

/// The smallest memory context whose margined value (`safeAutoContext`) reaches `target`.
pub fn memoryContextForMargined(target: u32) u32 {
    return target / auto_ctx_safety_pct * 100 + 1024;
}

pub fn maxSushiContextUpTo(config: *const model_mod.ModelConfig, upper: u32, kv_bits: u64, budget: u64, fixed_chunk: ?u64, warm: WarmPrefix) u32 {
    var low: u32 = 0;
    var high: u32 = upper;
    // Before the auto context is pinned the bill's reservation reads `getEffectiveContextLength`,
    // which is this very search: bill each probe as the context it is asking about.
    const unpinned = manualContext(config) == 0 and config.pinned_context == 0;
    while (low < high) {
        const mid = low + @as(u32, @intCast((@as(u64, high) - low + 1) / 2));
        var probe = config.*;
        if (unpinned) probe.pinned_context = mid;
        const chunk = fixed_chunk orelse sushiContextBytesAt(&probe, mid, kv_bits).chunk;
        if (prefillNeededAtChunk(&probe, mid, 0, kv_bits, chunk, warm) <= budget)
            low = mid
        else
            high = mid - 1;
    }
    return low;
}

pub fn requestContextRefusal(config: *const model_mod.ModelConfig, prompt_len: usize, max_tokens: u32, kv: trunk.KVQuantConfig, bill: AdmissionBill, warm: WarmPrefix) ?model_registry_mod.MemoryContextRefusal {
    if (!model_mod.usesSushiQuantMemoryBill(config)) return null;
    const bits: u64 = if (kv.scheme == .off) 16 else kv.bits;
    const budget = bill.available +| bill.evictionCredit();
    return .{
        .requested = @as(u64, @intCast(prompt_len)) +| @min(max_tokens, getEffectiveContextLength(config) -| @as(u64, @intCast(prompt_len))),
        .maximum = maxSushiContextAtWidth(config, bits, budget, bill.chunk, warm),
        .needed = bill.needed,
        .available = budget,
        .kv_bits = bits,
        .chunk = bill.chunk,
    };
}

pub fn memoryContextRefusalMessage(buf: []u8, refusal: model_registry_mod.MemoryContextRefusal) []const u8 {
    return std.fmt.bufPrint(buf, "Context {d} does not fit available GPU memory at KV{d}, prefill chunk {d}. Maximum context: {d} tokens. Reduce context or free memory.", .{
        refusal.requested, refusal.kv_bits, refusal.chunk, refusal.maximum,
    }) catch "The requested context does not fit in available GPU memory.";
}

pub fn loadServingBill(config: *const model_mod.ModelConfig, budget: u64) ?model_load.LoadServingBill {
    if (!model_mod.usesSushiQuantMemoryBill(config)) return null;
    const context = manualContext(config);
    if (context == 0) return null;
    const bits = defaultKvBits(config);
    const full = sushiContextBytesAt(config, context, bits);
    return .{
        .requested_context = context,
        .needed = full.bytes,
        .max_context = maxSushiContextForBudget(config, bits, budget),
        .kv_bits = bits,
        .chunk = full.chunk,
    };
}

/// Freeze this model's prefill chunk at load, from live memory. Idempotent.
/// Must run BEFORE the auto-context sizer: the sizer bills this chunk's
/// transient reserve, and `checkAttentionMemory` and
/// `prefill_plan.effectivePrefillChunk` then read the same frozen value, so the
/// bill and the forward cannot drift.
pub fn pinPrefillChunk(config: *model_mod.ModelConfig) u32 {
    if (config.pinned_prefill_chunk == 0) {
        const active_mem: usize = gpu_mem.activeBytes();
        const kv_bits: u64 = defaultKvBits(config);
        // Which ask: the gated arch reads the resolved budget; every other arch reads the raw
        // ask as before (the accessor would pin a second model against the first's budget).
        const hot_cache_ask = if (config.longCtxGated()) resolvedPrefixCacheMem() else legacyPrefixCacheAsk(config);
        // The override's precedence is the second gated input: ungated, `--prefill-chunk`
        // still outranks the pin only at forward time, as before (a known inconsistency).
        const pin_override: u32 = if (config.longCtxGated()) explicitPrefillChunk() else 0;
        config.pinned_prefill_chunk = billedPrefillChunk(
            config,
            kv_bits,
            currentGpuMemoryCeiling(config, active_mem),
            active_mem,
            sizerCtxKvBytes(config, kv_bits),
            hot_cache_ask,
            pin_override,
        );
        {
            const cache_mem: usize = gpu_mem.cacheBytes();
            log.debug("[prefill] pinned {d}: ceiling {d} MB, active {d} MB, mlx cache {d} MB, system available {d} MB, working set {d} MB\n", .{
                config.pinned_prefill_chunk,
                currentGpuMemoryCeiling(config, active_mem) >> 20,
                active_mem >> 20,
                cache_mem >> 20,
                metrics.getAvailableMemBytes() >> 20,
                gpu_mem.getGpuWorkingSetLimit() >> 20,
            });
        }
        // Say when the ctx bar is the binding one.
        if (config.longCtxGated() and explicitPrefillChunk() == 0) {
            const share_only = resolvePrefillChunk(config, kv_bits, currentGpuMemoryCeiling(config, active_mem), active_mem, 0, hot_cache_ask);
            if (share_only > config.pinned_prefill_chunk) {
                log.info("[prefill] chunk {d} (ctx bar: the --ctx-size KV bill; share-only would allow {d})\n", .{ config.pinned_prefill_chunk, share_only });
            }
        }
        // Say it once per model, wherever the model was pinned from (startup
        // primary or on-demand load) — a narrowed prefill otherwise reads as an
        // unexplained slowdown.
        if (settings_resolve.prefill_chunk.source != .flag) {
            var buf: [256]u8 = undefined;
            if (prefillChunkLoadLine(&buf, config, config.pinned_prefill_chunk, settings_resolve.prefill_chunk.value)) |line| log.info("{s}", .{line});
        }
    }
    return config.pinned_prefill_chunk;
}

/// The load line for the frozen width, null when there is nothing to say. On a per-request arch every
/// request prices its own rung against live memory, so the pin is only the fallback.
pub fn prefillChunkLoadLine(buf: []u8, config: *const model_mod.ModelConfig, pinned: u32, launch: usize) ?[]const u8 {
    if (perRequestPrefillChunkEnabled(config)) {
        const widest = rungWidth(config, 1, cappedRung(PREFILL_CHUNK_LADDER[0], ladderTop(config, explicitPrefillChunk())), config.longCtxGated());
        return std.fmt.bufPrint(buf, "Prefill chunk: per request, up to {d} at a short prompt (the widest rung each request's bill admits); load-time fallback {d}\n", .{ widest, pinned }) catch null;
    }
    if (config.isDsv41() and launch > dsv41_config.prefill_width) {
        if (pinned == dsv41_config.prefill_width)
            return std.fmt.bufPrint(buf, "Prefill chunk: {d} tokens (DeepSeek's default width; --prefill-chunk overrides)\n", .{pinned}) catch null;
        return std.fmt.bufPrint(buf, "Prefill chunk: {d} tokens (memory-sized down from DeepSeek's {d}; --prefill-chunk overrides)\n", .{ pinned, dsv41_config.prefill_width }) catch null;
    }
    if (pinned >= launch) return null;
    if (config.isGlm5() and pinned == @import("glm5/glm5_forward.zig").prefill_chunk)
        return std.fmt.bufPrint(buf, "Prefill chunk: {d} tokens (GLM's numerics width; --prefill-chunk overrides)\n", .{pinned}) catch null;
    return std.fmt.bufPrint(buf, "Prefill chunk: {d} tokens (memory-sized down from {d}; --prefill-chunk overrides)\n", .{ pinned, launch }) catch null;
}

/// PURE: clamp the hot prefix cache's byte budget to what the loaded weights
/// leave under the GPU ceiling after the serving context's KV and the prefill
/// transient reserve. A 40 GB `--prefix-cache-mem` beside a ~70 GB pack was
/// never validated against this headroom: the cache filled toward its cap and
/// a 143k prefill died in an uncatchable Metal OOM (2026-08-30, Flash-Next on
/// 128 GB). `requested == 0` (byte cap disabled) is bounded too — an uncapped
/// cache beside a large model is exactly that crash. Never returns 0:
/// `initWithMem` reads 0 as "no byte cap", the opposite of no headroom.
pub fn clampedPrefixCacheMem(
    requested: u64,
    gpu_ceiling: u64,
    active_weights: u64,
    ctx_kv_bytes: u64,
    transient_reserve: u64,
) u64 {
    const headroom = @max(gpu_ceiling -| (active_weights +| ctx_kv_bytes +| transient_reserve), 1);
    if (requested == 0) return headroom;
    return @min(requested, headroom);
}

/// What the hot cache gets and what it was charged for getting it.
pub const HotCachePlan = struct {
    /// The width the sizer pinned.
    chunk: u32,
    /// The width the clamp reserved for; not `chunk` on an arch that re-bills per request (`clampReserveWidth`).
    reserve_chunk: u32,
    reserve: u64,
    ctx_kv: u64,
    /// The clamped byte budget handed to `initWithMem`.
    budget: u64,
};

/// The whole chain in the one order that is monotone in the ask: chunk first
/// (ask-independent), then the cache takes the residual.
/// The prefill width the load-time clamp reserves for: the ladder floor on an arch that
/// re-bills its width per request (the load-time reserve is a promise to the first request
/// only). Billing the sizer's rung made the clamp non-monotone in the ceiling (a bigger
/// ceiling bought a wider rung and a smaller cache: 3873 vs 1076 MB). An explicit
/// `--prefill-chunk` only lowers that floor; ungated archs keep the sizer's rung.
pub fn clampReserveWidth(config: *const model_mod.ModelConfig, pinned_width: u32) u32 {
    return perRequestFloorWidth(config) orelse pinned_width;
}

/// The narrowest width a request can step down to on the per-request ladder: its floor, or a
/// narrower `--prefill-chunk`. Null where the width is pinned (no ladder).
fn perRequestFloorWidth(config: *const model_mod.ModelConfig) ?u32 {
    if (!perRequestPrefillChunkEnabled(config)) return null;
    return cappedRung(PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1], explicitPrefillChunk());
}

pub fn planHotCache(
    config: *const model_mod.ModelConfig,
    kv_bits: u64,
    ceiling: u64,
    active_weights: u64,
    ctx_tokens: u64,
    sizer_ctx_kv: u64,
    requested: u64,
    chunk_override: u32,
) HotCachePlan {
    const chunk = billedPrefillChunk(config, kv_bits, ceiling, active_weights, sizer_ctx_kv, requested, chunk_override);
    const reserve_chunk = clampReserveWidth(config, chunk);
    const reserve = prefillTransientReserve(config, kv_bits, reserve_chunk);
    const ctx_kv: u64 = sessionBytesPerToken(config, kv_bits) *| ctx_tokens +| slotRingBytes(config, kv_bits);
    return .{
        .chunk = chunk,
        .reserve_chunk = reserve_chunk,
        .reserve = reserve,
        .ctx_kv = ctx_kv,
        .budget = clampedPrefixCacheMem(requested, ceiling, active_weights, ctx_kv, reserve),
    };
}

/// SSD-first budget semantics. The RAM-first formula treats the hot cache as what is left
/// after the live session's reserve, which double-counts (the resident entry's arrays are
/// shared with the live KV). Here the budget floors at one entry at the working context and
/// `--prefix-cache-mem` is the RAM allowance for idle entries on top (0 = none idle).
pub fn ssdFirstPrefixCacheMem(
    idle_requested: u64,
    gpu_ceiling: u64,
    active_weights: u64,
    ctx_kv_bytes: u64,
    transient_reserve: u64,
) u64 {
    const headroom = gpu_ceiling -| (active_weights +| ctx_kv_bytes +| transient_reserve);
    const idle = @min(idle_requested, headroom);
    // Never 0: `initWithMem` reads 0 as "no byte cap".
    return @max(ctx_kv_bytes +| idle, 1);
}

/// The context a load-time bill must use, before `pinAutoContext` has run: on an auto boot
/// `getEffectiveContextLength` re-derives through `computeMemoryContext`, which reserves the
/// very ask being resolved and answered a 1024-token placeholder. `cache_reserve` is
/// `CTX_SIZING_CACHE_RESERVE` on both arms so the billed session is the advertised one; it
/// must never be the ask or anything derived from it. Runs `autoContextFrom`, the same helper
/// `autoContextFor` runs.
pub fn resolvedContextForLoad(
    explicit_ctx: u32,
    pinned_ctx: u32,
    ceiling: u64,
    active_mem: u64,
    cache_reserve: u64,
    transient_reserve: u64,
    per_tok: u64,
    ctx_cap: u32,
    ctx_default: u32,
) u32 {
    if (explicit_ctx > 0) return explicit_ctx;
    if (pinned_ctx > 0) return pinned_ctx;
    return autoContextFrom(safeContextForBudget(
        ceiling,
        active_mem,
        cache_reserve +| transient_reserve,
        per_tok,
        0,
    ), ctx_cap, ctx_default);
}

/// The SSD-first call site. Takes the static ceiling (the budget is a property of the machine),
/// the same reserve constant the RAM arm and `computeMemoryContext` pass, and the boot's
/// configured KV width (billing at bf16 on a `--kv-quant 8` boot was the 22,464-vs-13,824 MB defect).
pub fn ssdFirstSessionTokensNow(config: *const model_mod.ModelConfig, kv_bits: u64, ceiling: u64, active_mem: u64, chunk: u32) u32 {
    if (billedAutoContext(config)) return billedAdvertisedContext(config, memoryContextAtChunk(config, kv_bits, ceiling, active_mem, chunk));
    return resolvedContextForLoad(
        manualContext(config),
        config.pinned_context,
        ceiling,
        active_mem,
        ctxCacheShare(ceiling, active_mem, ctxSizingCacheReserve(config)),
        prefillTransientReserve(config, kv_bits, chunk),
        sessionBytesPerToken(config, kv_bits),
        config.contextCap(),
        config.defaultContext(),
    );
}

/// The bytes the SSD-first budget floors at: one session at the working context, at the width the cache stores.
pub fn ssdFirstSessionKvBytes(config: *const model_mod.ModelConfig, kv_bits: u64, ceiling: u64, active_mem: u64, chunk: u32) u64 {
    return sessionBytesPerToken(config, kv_bits) *|
        ssdFirstSessionTokensNow(config, kv_bits, ceiling, active_mem, chunk) +| slotRingBytes(config, kv_bits);
}

/// The SSD-first arm of `prefixCacheMemForLoad`, gate and log included. Null when the arch
/// gate is off; the caller falls through to `clampedPrefixCacheMem`.
pub fn ssdFirstBudgetForLoad(
    config: *model_mod.ModelConfig,
    requested: u64,
    ceiling: u64,
    active_mem: usize,
    ctx_kv: u64,
    transient_reserve: u64,
    idle_out: *u64,
    quiet: bool,
) ?u64 {
    // The predicate, shared with the spill site: without a disk tier the mode's floor would be RAM the server cannot use.
    if (launch_state.prefix_cache_capacity == 0 or !prefix_cache_mod.ssdFirstActive(config, diskTierOn(config), launch_state.prefix_cache_ram_enabled)) return null;
    const budget = ssdFirstPrefixCacheMem(
        requested,
        ceiling,
        active_mem,
        ctx_kv,
        transient_reserve,
    );
    // Publish the cache's cost BEYOND the live session, not the whole budget: `budget` includes
    // `ctx_kv`, and `computeMemoryContext` subtracts the published figure, so publishing the
    // whole would bill the session twice and halve the auto-context.
    publishResolvedPrefixCacheMem(budget -| ctx_kv);
    // The idle allowance, handed to the cache for `spillIdleEntries` to enforce.
    idle_out.* = budget -| ctx_kv;
    // On this arch `--prefix-cache-mem` is the idle allowance, not the whole cache; say so once.
    if (quiet) return budget;
    log.info("[hot-cache] SSD-first budget {d} MB = one session at the working context ({d} MB) + {d} MB idle (--prefix-cache-mem {d} MB = the IDLE allowance on this arch; 0 = no idle entries)\n", .{
        budget >> 20,
        ctx_kv >> 20,
        (budget -| ctx_kv) >> 20,
        requested >> 20,
    });
    return budget;
}

/// Impure wrapper for the model-load site (`model_load.doLoadOnInferenceThread`,
/// reached through the LoadParams/LoadRequest resolver pointer — the scheduler
/// deliberately has no server.zig import): the weights are resident there, so
/// `mlx_get_active_memory` is honest. Logs one line when the clamp bites, and publishes the
/// answer so the auto-context sizer stops reserving a budget the cache never got.
/// The hot-cache reserve the context is sized against at load time on the RAM-first arm: a
/// constant (the server's own `--prefix-cache-mem` default), never the ask. Reading the ask
/// let a 60GB ask collapse the advertised context to 870 tokens; context is the primary
/// claimant and the cache the residual.
pub const CTX_SIZING_CACHE_RESERVE: u64 = 2 * 1024 * 1024 * 1024;

/// The previous context-sizing cache reserve: the raw `--prefix-cache-mem` ask. Kept for
/// ungated archs so their advertised `context_length` does not move. None when RAM retention
/// is off or the model's hot cache never loads.
fn legacyPrefixCacheAsk(config: *const model_mod.ModelConfig) u64 {
    if (launch_state.prefix_cache_capacity == 0 or !launch_state.prefix_cache_ram_enabled) return 0;
    if (!prefix_cache_mod.HotPrefixCache.shouldUse(config, launch_state.ssm_checkpoint_stride > 0)) return 0;
    return launch_state.prefix_cache_mem_bytes; // legacy_ask_read
}

/// An arch whose auto context is the largest one its request bill admits, and that has none pinned yet.
/// Flash-Next's advertised context from the largest one its bill admits: within the measured range the bill
/// (at least 1.1x the live peak) is the one reserve, so no margin goes on top; past it the margin stays.
fn billedAdvertisedContext(config: *const model_mod.ModelConfig, memory_ctx: u32) u32 {
    const measured: u32 = @intCast(@min(memory_ctx, MEASURED_NET_MAX_TOKENS) / 1024 * 1024);
    const ctx = @max(safeAutoContext(memory_ctx), measured);
    return if (config.contextCap() > 0) @min(ctx, config.contextCap()) else ctx;
}

fn billedAutoContext(config: *const model_mod.ModelConfig) bool {
    return config.longCtxGated() and manualContext(config) == 0 and config.pinned_context == 0;
}

/// The context the RAM-first clamp bills, through the one shared pure resolver.
fn ramFirstContextForLoad(config: *const model_mod.ModelConfig, kv_bits: u64, active_mem: u64, chunk: u32) u32 {
    const ceiling = currentGpuMemoryCeiling(config, active_mem);
    if (billedAutoContext(config)) return billedAdvertisedContext(config, memoryContextAtChunk(config, kv_bits, ceiling, active_mem, chunk));
    return resolvedContextForLoad(
        manualContext(config),
        config.pinned_context,
        ceiling,
        active_mem,
        ctxCacheShare(ceiling, active_mem, ctxSizingCacheReserve(config)),
        prefillTransientReserve(config, kv_bits, chunk),
        sessionBytesPerToken(config, kv_bits),
        config.contextCap(),
        config.defaultContext(),
    );
}

/// The hot-cache ask when nobody named one: `session` (`OneSession.ask`), never under 2 GB. Below
/// one session the cache keeps only a prefix of the longest conversations, the ones whose reuse
/// saves the most prefill. Any other ask (an operator's, even one equal to the default) stands.
pub fn defaultPrefixCacheAsk(requested: u64, explicit: bool, session: u64) u64 {
    if (explicit or requested != launch_state.PREFIX_CACHE_MEM_DEFAULT) return requested;
    return @max(requested, session);
}

/// GLM's unnamed RAM tier: the last few turns, given back to a long prefill by admission.
/// Long reuse belongs on the SSD tier (`--prefix-cache-disk`).
pub const GLM_PREFIX_CACHE_MEM_DEFAULT: u64 = 1 << 30;

/// `defaultPrefixCacheAsk`, with GLM's unnamed ask pinned at `GLM_PREFIX_CACHE_MEM_DEFAULT`.
pub fn defaultPrefixCacheAskFor(config: *const model_mod.ModelConfig, requested: u64, explicit: bool, session: u64) u64 {
    if (archHook("unnamedCacheAsk", u64, config, .{ requested, explicit })) |ask| return ask;
    return defaultPrefixCacheAsk(requested, explicit, session);
}

/// Bytes one cached session at `ctx_tokens` holds: the capacity a request there reserves (one
/// widest rung past it, `reservedCacheTokens`), the rings, every ring checkpoint an entry keeps,
/// and the SSM checkpoints a cold prefill of that length retains.
pub fn oneSessionEntryBytes(config: *const model_mod.ModelConfig, kv_bits: u64, ctx_tokens: u64, chunk: u64) u64 {
    const ring = config.swaRingBytes() +| @as(u64, prefix_cache_mod.RING_CHECKPOINT_MAX) *| config.swaRingCheckpointBytes();
    return sessionBytesPerToken(config, kv_bits) *| (ctx_tokens +| @as(u64, PREFILL_CHUNK_LADDER[0])) +|
        config.qsaRingBytes() +| kvBytesPerTokenAtBits(ring, kv_bits) +| retainedSsmCheckpointBytes(config, ctx_tokens, 0, chunk);
}

/// What an unnamed hot-cache budget grows toward, and the room the machine leaves it.
pub const OneSession = struct {
    /// One cached session at the working context (`oneSessionEntryBytes`).
    session: u64,
    /// The ceiling less the weights, `claim` and `bill`.
    room: u64,
    /// Admission's bill for a cold prompt that fills the working context.
    bill: u64,
    /// Page cache that must stay resident beside the weights.
    claim: u64,

    /// The session when the room holds it. Else the room, at most half the bill: a session that
    /// outgrows the cache is trimmed by a real copy beside the entry it replaces and its live KV.
    pub fn ask(self: OneSession) u64 {
        if (self.session <= self.room) return self.session;
        return @min(self.room, self.bill / 2);
    }
};

/// `claim` (the qwen4_exp n-gram table, `qwen4_exp.page_cache_claim`) is billed as GPU memory: the
/// wired KV would otherwise evict the rows every decode step gathers. `chunk` is the width the
/// clamp reserves, so a full cache still admits a cold full-context prompt there.
pub fn oneSessionFor(config: *const model_mod.ModelConfig, kv_bits: u64, ceiling: u64, active_mem: u64, claim: u64, ctx_tokens: u64, chunk: u64) OneSession {
    const bill = prefillNeededAtChunk(config, ctx_tokens, 0, kv_bits, chunk, .{ .mtp_on = mtpHeadDefaultOn(config) });
    return .{
        .session = oneSessionEntryBytes(config, kv_bits, ctx_tokens, chunk),
        .room = ceiling -| (active_mem +| claim +| bill),
        .bill = bill,
        .claim = claim,
    };
}

fn logOneSessionAsk(ask: u64, ctx_tokens: u64, one: OneSession) void {
    log.info("[hot-cache] budget {d} MB for one session at the working context ({d} tokens, {d} MB; room {d} MB beside the weights, {d} MB of n-gram page cache and a full-context prompt's {d} MB) — --prefix-cache-mem overrides\n", .{
        ask >> 20, ctx_tokens, one.session >> 20, one.room >> 20, one.claim >> 20, one.bill >> 20,
    });
}

/// `revise` marks a post-load re-clamp (#364): the cache's own resident bytes come off the
/// machine read so it never bills itself, and the budget lines stay quiet.
pub fn prefixCacheMemForLoad(config: *model_mod.ModelConfig, requested: u64, revise: scheduler_mod.BudgetRevise, idle_out: *u64) u64 {
    // RAM-first default: no idle allowance. Written first so every early return carries it.
    idle_out.* = 0;
    var active_mem: usize = gpu_mem.activeBytes();
    active_mem -|= @as(usize, @intCast(@min(revise.exclude_bytes, std.math.maxInt(usize))));
    const kv_bits: u64 = defaultKvBits(config);
    // Arch gate: the budget resolver changed three inputs at once (static ceiling, the
    // load-time context resolver, the floor-width reserve), all measured on qwen4_exp alone.
    // Ungated is the previous `prefixCacheMemForLoad`.
    if (!config.longCtxGated()) {
        const chunk: u64 = pinPrefillChunk(config);
        const ctx_tokens: u64 = getEffectiveContextLength(config);
        const ceiling = currentGpuMemoryCeiling(config, active_mem);
        // `statePerTokenBilled` is 0 off qwen4_exp.
        const ctx_kv: u64 = (kvBytesPerTokenAtBits(config.kvBytesPerToken(), kv_bits) +| statePerTokenBilled(config)) *|
            ctx_tokens +| slotRingBytes(config, kv_bits);
        // Context sizing and the chunk pin read the raw ask, so neither moves with this one.
        const one = oneSessionFor(config, kv_bits, ceiling, active_mem, qwen4_mod.page_cache_claim.load(.acquire), ctx_tokens, chunk);
        const ask = defaultPrefixCacheAskFor(config, requested, launch_state.prefix_cache_mem_source != .default, one.ask());
        const clamped = clampedPrefixCacheMem(
            ask,
            ceiling,
            active_mem,
            ctx_kv,
            prefillTransientReserve(config, kv_bits, chunk),
        );
        publishResolvedPrefixCacheMem(clamped);
        if (revise.quiet) return clamped;
        if (ask != requested) logOneSessionAsk(ask, ctx_tokens, one);
        if (ask > 0 and clamped < ask) {
            log.info("[hot-cache] budget clamped {d} -> {d} MB (weights + ctx KV + prefill reserve vs GPU ceiling)\n", .{ ask >> 20, clamped >> 20 });
        } else if (requested == 0) {
            log.info("[hot-cache] budget capped at {d} MB (no --prefix-cache-mem; weights + ctx KV + prefill reserve vs GPU ceiling)\n", .{clamped >> 20});
        }
        return clamped;
    }
    // SSD-first takes the budget before the RAM-first plan: its resident entry IS the live KV.
    const ssd_chunk: u64 = pinPrefillChunk(config);
    // Not `getEffectiveContextLength`: on an auto boot it is still unpinned here.
    const ssd_ctx_kv: u64 = ssdFirstSessionKvBytes(config, kv_bits, staticGpuMemoryCeiling(), active_mem, @intCast(ssd_chunk));
    // Static ceiling on both arms (the budget is a property of the machine), and the clamp
    // reserve at the same width `planHotCache` reserves for: the ladder floor on an arch that
    // re-bills per request. The session is billed at the pinned width, the budget at the floor.
    const ssd_clamp_reserve: u64 = prefillTransientReserve(config, kv_bits, clampReserveWidth(config, @intCast(ssd_chunk)));
    if (ssdFirstBudgetForLoad(config, requested, staticGpuMemoryCeiling(), active_mem, ssd_ctx_kv, ssd_clamp_reserve, idle_out, revise.quiet)) |b| return b;
    // Pin first, then hand the pinned width in as the override.
    const pinned: u32 = pinPrefillChunk(config);
    // Not `getEffectiveContextLength`: still a placeholder on an auto boot.
    const ctx_tokens = ramFirstContextForLoad(config, kv_bits, active_mem, pinned);
    // Static ceiling, not the live one: the budget must be reproducible boot to boot.
    const one = oneSessionFor(config, kv_bits, staticGpuMemoryCeiling(), active_mem, qwen4_mod.page_cache_claim.load(.acquire), ctx_tokens, clampReserveWidth(config, pinned));
    const ask = defaultPrefixCacheAskFor(config, requested, launch_state.prefix_cache_mem_source != .default, one.ask());
    const plan = planHotCache(
        config,
        kv_bits,
        staticGpuMemoryCeiling(),
        active_mem,
        ctx_tokens,
        sizerCtxKvBytes(config, kv_bits),
        ask,
        pinned,
    );
    publishResolvedPrefixCacheMem(plan.budget);
    if (revise.quiet) return plan.budget;
    // Say so when the box is under external pressure right now, but do not shrink the budget for it.
    const live_ceiling: u64 = currentGpuMemoryCeiling(config, active_mem);
    if (live_ceiling < staticGpuMemoryCeiling()) {
        const free_gb = @as(f64, @floatFromInt(live_ceiling -| active_mem)) / (1024.0 * 1024.0 * 1024.0);
        log.info("[hot-cache] budget {d} MB (static ceiling); free at load {d:.1} GB — live admission will evict as needed\n", .{ plan.budget >> 20, free_gb });
    }
    if (ask != requested) logOneSessionAsk(ask, ctx_tokens, one);
    if (ask > 0 and plan.budget < ask) {
        log.info("[hot-cache] budget clamped {d} -> {d} MB (chunk {d}, reserve at width {d} = {d} MB, ctx KV {d} MB)\n", .{ ask >> 20, plan.budget >> 20, plan.chunk, plan.reserve_chunk, plan.reserve >> 20, plan.ctx_kv >> 20 });
    } else if (requested == 0) {
        log.info("[hot-cache] budget capped at {d} MB (no --prefix-cache-mem; chunk {d}, reserve at width {d} = {d} MB, ctx KV {d} MB)\n", .{ plan.budget >> 20, plan.chunk, plan.reserve_chunk, plan.reserve >> 20, plan.ctx_kv >> 20 });
    }
    return plan.budget;
}

/// The SSD bytes one cached token costs this model at these settings, from the serialized geometry:
/// the KV rows at the served width, the one pooled QSA history the entry writes, and the MTP head's KV
/// and history where the head is on. Execution state (the QSA score bank) is
/// RAM-only and never enters it. A recurrent state is kept on every `POINT_GRID` tokens, so its share per token
/// is billed; DFlash2's window and MiMo's rings are fixed per entry.
pub fn diskBytesPerToken(config: *const model_mod.ModelConfig, kv_bits: u64) u64 {
    if (archHook("diskTokenBytes", u64, config, .{kv_bits})) |bytes| return bytes;
    const points = (config.ssmCheckpointBytes() +| kv_disk_cache.POINT_GRID - 1) / kv_disk_cache.POINT_GRID;
    const rows = kvBytesPerTokenAtBits(config.kvBytesPerToken(), kv_bits) +| config.qsaHistoryBytesPerToken() +| points;
    if (config.isGlm5() or !config.longCtxGated() or !mtpHeadDefaultOn(config)) return rows;
    const layers = config.attnCacheLayerCount();
    if (layers == 0) return rows;
    return rows +| mtpHeadKvBytesPerToken(config) +| config.qsaHistoryBytesPerToken() / layers;
}

/// The context the SSD budget is sized for: the model's pinned served context (`pinAutoContext`, idempotent
/// and the one startup and `/v1/load-model` run anyway), so the Allocating line, `/props` and `/v1/models`
/// name the same number. The weights are resident at this point, so the memory read is the one they would take.
pub fn diskContextForLoad(config: *model_mod.ModelConfig) u64 {
    return pinAutoContext(config);
}

/// The SSD budget the model gets at load, in bytes (0 = tier off); logs the `Allocating` line, or why
/// there is none. Runs on the inference thread beside `prefixCacheMemForLoad`.
/// `held` is what this model's tier already occupies under `dir`: it counts as available, or a restart on a
/// nearly full volume would evict what the last boot stored.
pub fn prefixCacheDiskForLoad(config: *model_mod.ModelConfig, dir: []const u8, held: u64) ?disk_placement.DefaultDiskBudget {
    const kv_bits = defaultKvBits(config);
    const bpt = diskBytesPerToken(config, kv_bits);
    const ask = diskAsk();
    const ctx = if (ask.explicit()) 0 else diskContextForLoad(config);
    // An unreadable volume is no reason to refuse: the tier re-checks before every store.
    const free: u64 = if (disk_placement.volumeSpaceNear(dir)) |v| v.free else std.math.maxInt(u64);
    var buf: [384]u8 = undefined;
    const plan = disk_placement.resolveDiskBudget(ask.explicit(), ask.bytes, launch_state.prefix_cache_capacity, ctx, bpt, free, held) orelse {
        log.info("{s}\n", .{disk_placement.diskOffLine(&buf, free, dir, if (ask.explicit() and ask.bytes == 0) ask.sourceName() else null)});
        return null;
    };
    log.info("{s}\n", .{disk_placement.allocatingDiskLine(&buf, plan, launch_state.prefix_cache_capacity, ctx, bpt, dir, ask.sourceName())});
    return plan;
}

/// The width a ladder rung actually forwards at for this prompt.
pub fn widthForRung(cfg: *const model_mod.ModelConfig, seq: u64, rung: u32) u32 {
    return @intCast(rungWidth(cfg, seq, rung, cfg.longCtxGated()));
}

/// The largest context this model's per-token footprint fits into RAM right now, ignoring the checkpoint's own maximum.
pub fn computeMemoryContext(config: *const model_mod.ModelConfig) u32 {
    const active_mem: usize = gpu_mem.activeBytes();
    return memoryContextAt(config, currentGpuMemoryCeiling(config, active_mem), active_mem);
}

/// `computeMemoryContext` at one reading of the ceiling and MLX's active memory.
pub fn memoryContextAt(config: *const model_mod.ModelConfig, ceiling: u64, active_mem: u64) u32 {
    // `total_ctx = 0` asks for the unshrunk cap: the widest forward any prompt can run.
    const chunk: u64 = @intCast(prefill_plan.effectivePrefillChunk(
        config.prefillScoreHeadDim(),
        config.num_attention_heads,
        0,
        config.has_sliding_window,
        config.isMoe(),
        config.longCtxGated(),
        config.pinned_prefill_chunk,
    ));
    return memoryContextAtChunk(config, defaultKvBits(config), ceiling, active_mem, chunk);
}

/// `memoryContextAt` for a given prefill width, before the checkpoint's position cap.
pub fn memoryContextAtChunk(config: *const model_mod.ModelConfig, kv_bits: u64, ceiling: u64, active_mem: u64, chunk: u64) u32 {
    if (config.num_attention_heads == 0) return 16384;
    // Flash-Next advertises the context its own request bill admits at the floor rung, from the
    // headroom admission reads (the writers' host bytes included), so the two cannot disagree.
    if (config.longCtxGated()) {
        const budget = admissionAvailable(ceiling, active_mem) -| ctxCacheShare(ceiling, active_mem, ctxSizingCacheReserve(config));
        return maxSushiContextAtWidth(config, kv_bits, budget, null, .{ .mtp_on = mtpHeadDefaultOn(config) });
    }
    if (archHook("memoryContext", u32, config, .{ kv_bits, ceiling, active_mem, chunk })) |ctx| return ctx;
    //   KV cache: the arch's own caching-layer count and K/V widths, billed at the active kv-quant width.
    const per_tok: u64 = sessionBytesPerToken(config, kv_bits);
    return safeContextForBudget(
        // Real reachable ceiling, so auto-context shrinks under external memory pressure (#64).
        ceiling,
        active_mem,
        // The hot prefix cache fills to this cap over a session, and a prefill has to land on
        // top of whatever context we report. `CTX_SIZING_CACHE_RESERVE`, not the resolved
        // budget and not the ask: the ask starved the context, the resolved budget was a
        // one-step loop. Gated: off the gate this is the raw ask, so `/v1/models` is unchanged.
        ctxCacheShare(ceiling, active_mem, ctxSizingCacheReserve(config)) +| prefillTransientReserve(config, kv_bits, chunk),
        per_tok,
        // 0 = do not clamp to the checkpoint's max here; the caller applies it after the margin.
        0,
    );
}

/// The share of the room `reserve` may claim: the context is the primary claimant, so the cache never takes more than half.
pub fn ctxCacheShare(ceiling: u64, active_mem: u64, reserve: u64) u64 {
    return @min(reserve, admissionAvailable(ceiling, active_mem) / 2);
}

/// The cache reserve the context sizer bills. Gated: the ask-independent constant. Ungated:
/// the raw `--prefix-cache-mem`. Both load-time wrappers must pass the same value.
pub fn ctxSizingCacheReserve(config: *const model_mod.ModelConfig) u64 {
    if (archHook("cacheReserve", u64, config, .{})) |reserve| return reserve;
    if (launch_state.prefix_cache_capacity == 0 or !launch_state.prefix_cache_ram_enabled) return 0;
    return if (config.longCtxGated()) CTX_SIZING_CACHE_RESERVE else legacyPrefixCacheAsk(config);
}

/// Pure memory model behind checkAttentionMemory. All quantities in bytes,
/// 25% safety margin included. `chunk` must be the SAME prefill chunk
/// generate.zig will pick (prefill_plan.effectivePrefillChunk) so the
/// admission guard and the real prefill cannot drift. Terms:
///   - kv: persistent for the request; billed at the ACTIVE kv-quant width
///     (+0.5 bit/elem group scale/bias overhead when quantized), not a
///     hardcoded fp16.
///   - scores: the composed-SDPA scratch [heads, chunk, seq] — materialized
///     only for head_dims no fused kernel covers (transformer.
///     prefillHeadDimFused: <= 128 via MLX, 256 via sushi_attn_p256 — unfused
///     only for an exotic dim).
///     Bounded to ~one layer by the adaptive eval cadence
///     (transformer.prefillEvalCadence).
///   - dequant: dense-fp16 rebuild of the FULL quantized cache each layer
///     (KVCache.denseView under --kv-quant).
///   - mlp: QKV/MLP transient envelope; ~3 layers coexist between eval
///     points, and it is CHUNK-bounded, never seq-scaled — the old
///     8×seq×ffn envelope over-billed a 255K prompt by ~60 GB and rejected
///     requests that fit comfortably (PR #69).
///
/// `attn_keys` is how many keys one query actually reads (config.
/// prefillAttnKeys): `seq` for dense causal attention, a much smaller bound for
/// a sparse arch. It scales the SCORE term only — KV/dequant are storage, and
/// a sparse reader still stores every latent it might later attend to.
/// `kv_per_tok` is the whole-model DENSE (f16) KV bytes for one token —
/// `ModelConfig.kvBytesPerToken()`. It replaced a `layers` parameter because
/// neither the layer count nor the per-head width is uniform once an arch
/// interleaves linear-attention layers or stores keys wider than values.
///
/// `hdim` and `score_hdim` are two DIFFERENT widths and only coincide on a
/// symmetric arch. `hdim` is the STORED width the quantized-KV dequant
/// transient is read at; `score_hdim` is the width the score is contracted at
/// (`ModelConfig.prefillScoreHeadDim`), which is what decides whether a fused
/// kernel exists — MLA scores over 192 while storing values at 128, and billing
/// the score term against `head_dim` 128 declared it fused and billed ZERO for
/// a composed path that really does materialize [heads, chunk, seq].
/// KV bytes for ONE token at the width the cache actually STORES. `dense` is
/// `ModelConfig.kvBytesPerToken()` — the arch's own caching-layer count and its
/// own K and V widths, at fp16. A quantized cache stores `kv_bits` per element
/// plus the group scale/bias pair, which is the `+1` in `(2·bits + 1)/32`
/// (~0.5 bit/elem at group 64 with bf16 params).
///
/// Both the auto-context sizer and the prefill admission guard bill through
/// here. The sizer used to bill fp16 unconditionally, so a `--kv-quant 4`
/// server reported — and served — under a third of the context it can hold.
pub fn kvBytesPerTokenAtBits(dense: u64, kv_bits: u64) u64 {
    if (kv_bits >= 16) return dense;
    return dense * (2 * kv_bits + 1) / 32;
}

/// Context-INDEPENDENT bytes one live slot holds beside its per-token KV: the
/// qwen4 QSA raw-key ring (not kv-quantized) and a ringed sliding layer's
/// retained window plus every checkpoint a slot holds (`SLOT_RING_CHECKPOINTS`;
/// stored like any other cache row, so they follow `kv_bits`).
/// Every sizer that used to add `qsaRingBytes` adds this instead — a second
/// per-slot constant billed at only some of them is the under-bill class.
pub fn slotRingBytes(config: *const model_mod.ModelConfig, kv_bits: u64) u64 {
    if (archHook("slotRing", u64, config, .{kv_bits})) |bytes| return bytes;
    const swa = config.swaRingBytes() +| prefix_cache_mod.SLOT_RING_CHECKPOINTS *| config.swaRingCheckpointBytes();
    return config.qsaRingBytes() +| kvBytesPerTokenAtBits(swa, kv_bits) +| batchedDecodeRowsBytes(config);
}

/// What the other rows of a batched decode hold beside this slot's tick; zero where decode does not batch.
pub fn batchedDecodeRowsBytes(config: *const model_mod.ModelConfig) u64 {
    return archHook("decodeRows", u64, config, .{}) orelse 0;
}

/// The chunk-independent floor every prefill pays: MLX runtime scratch, the
/// KV cache's proportional capacity growth (old + new buffer coexist across a
/// grow), and the graph live-set no other term models. MEASURED as the
/// intercept of peak-above-steady-state against the chunk. It
/// does NOT scale with the weights — a dense bf16 checkpoint pays it too — so
/// it is billed as what it measures as: a constant.
pub const PREFILL_RUNTIME_FLOOR_BYTES: u64 = 512 * 1024 * 1024;

/// Prompt length up to which Flash-Next's summed terms stand without a net: measured at least a tenth over
/// the live peak (engine-memory-admission.md#bill-net). Past it the bill keeps the quarter its terms were fitted under.
const MEASURED_NET_MAX_TOKENS: u64 = 86_016;

/// The net the bill adds over its summed terms, in sixteenths (16 = none, 20 = the quarter).
fn billUplift16ths(config: *const model_mod.ModelConfig, seq: u64) u64 {
    return if (config.longCtxGated() and seq <= MEASURED_NET_MAX_TOKENS) 16 else 20;
}

/// How many MoE layers' gather-sort expansions coexist between eval points.
/// Mirrors `Trunk.MOE_EVAL_EVERY_N_LAYERS` — the MoE prefill loop evals
/// every 4th layer, so 4 layers' worth is what the peak holds.
const MOE_PREFILL_COEXIST: u64 = trunk.Trunk.PREFILL_EVAL_CADENCE_DEFAULT;

/// What a warm turn already holds. A RAM hot-cache restore hands the slot the entry's KV
/// buffers without allocating another copy, so `available` is already net of them and
/// billing them again refused a 786k warm turn whose own entry was resident. KV only: the
/// QSA history is privatised on restore. Credited only while the prefix fits the destination
/// capacity (a grow allocates a fresh buffer beside the protected entry).
pub const WarmPrefix = struct {
    /// Prompt rows the hot cache restored. 0 = cold.
    matched_tokens: u64 = 0,
    /// Tokens of capacity the restored buffers hold (`KVCache.residentCapacityTokens`). The
    /// connection thread passes 0 and defers the warm decision to the inference thread.
    capacity_tokens: u64 = 0,
    /// Did the restore check out its entry (`prefix_cache.checkoutEligible`)? A shared restore
    /// is copied whole by the first append, so crediting it is an under-bill.
    will_donate: bool = false,
    mtp_on: bool = false,
    /// KDA checkpoints a GLM request keeps for the prefix cache. Only the inference thread's
    /// admission pass bills them, and it keeps fewer rather than refuse.
    checkpoints: u32 = 0,
    /// The prefill moves each checkpoint to SSD as it is produced and frees the RAM copy, so at most
    /// `glm5_prefix.checkpoint_ram_held` states are in RAM at once.
    released: bool = false,
    /// The request takes a thought point mid-decode: one more checkpoint state, held to its commit.
    thought: bool = false,

    /// Rows of KV this request will not allocate: the rows the restored buffer already holds.
    /// Capacity is a ceiling on the credit, never a switch: a growing chain outruns its capacity
    /// on nearly every turn. The grow's own cost is `reserved - credited` plus
    /// `PrefillRequestTerms.grow_coexist_bytes`. `will_donate` gates the whole credit.
    pub fn creditedRows(self: WarmPrefix, reserved: u64) u64 {
        if (!self.will_donate) return 0;
        if (self.matched_tokens == 0) return 0;
        return @min(@min(self.matched_tokens, self.capacity_tokens), reserved);
    }

    /// Does this append grow the buffers at all? The per-layer grow fires on `offset + new_len
    /// > capacity`; a reservation above the prompt allocates nothing by itself.
    pub fn grows(self: WarmPrefix, seq: u64) bool {
        if (!self.will_donate or self.matched_tokens == 0) return false;
        return seq > self.capacity_tokens;
    }

    /// Does the reservation outgrow restored buffers that hold the whole prompt? Then no append
    /// in the prefill grows them; `KVCache.growToReservation` does, before the first chunk.
    pub fn decodeOutgrows(self: WarmPrefix, seq: u64, reserved: u64) bool {
        if (self.matched_tokens == 0) return false;
        return seq <= self.capacity_tokens and reserved > self.capacity_tokens;
    }
};

/// PURE: the most KV-caching layers that fall inside one window of `window` consecutive layers,
/// i.e. how many old buffers can be alive at once at that eval cadence.
pub fn attnLayersPerEvalWindow(config: *const model_mod.ModelConfig, window: u32) u32 {
    const n = config.num_hidden_layers;
    if (window == 0 or n == 0) return 0;
    if (window >= n) return config.attnCacheLayerCount();
    var best: u32 = 0;
    var start: u32 = 0;
    while (start + window <= n) : (start += 1) {
        var count: u32 = 0;
        var i: u32 = start;
        while (i < start + window) : (i += 1) {
            if (config.isKvPerTokenLayer(i)) count += 1;
        }
        if (count > best) best = count;
    }
    return @min(best, config.kvPerTokenLayerCount());
}

/// PURE: bytes of OLD KV buffer alive beside the new ones while a warm append grows the cache.
/// `growQuantBuf` drops our reference to the old buffer at once, but the graph node holds it
/// until it EVALUATES, so the coexistence window is the prefill loop's eval cadence rather than
/// the size of the cache. A forward narrower than `prefillEvalCadenceApplies` runs one eval for
/// the whole layer loop and therefore does pay the whole old cache.
fn growCoexistBytes(config: *const model_mod.ModelConfig, warm: WarmPrefix, seq: u64, reserved: u64, kv_per_tok: u64) u64 {
    if (warm.grows(seq)) return oldBuffersInEvalWindow(config, warm.capacity_tokens, seq -| warm.matched_tokens, kv_per_tok);
    if (warm.decodeOutgrows(seq, reserved)) {
        // The prefill grows each layer first (`KVCache.growToReservation`), one eval at a
        // time; a share's old rows stay the entry's.
        return if (warm.will_donate) oldBuffersInEvalWindow(config, warm.capacity_tokens, 1 << 20, kv_per_tok) else 0;
    }
    // An SSD restore installs each layer at exactly its restored rows, so the first append grows
    // it beside the restored buffer. A ringed arch's admission never sees the restore (it bills a
    // warm request cold), so with a disk tier it bills that coexistence at the prompt's length.
    if (diskTierOn(config) and config.swaRingTokens() > 0 and !config.longCtxGated())
        return oldBuffersInEvalWindow(config, seq, seq, kv_per_tok);
    return 0;
}

/// Bytes of `tokens`-row old buffers alive beside their grown copies: one eval window of the
/// caching layers when a `span`-wide forward runs the cadence, every caching layer otherwise.
fn oldBuffersInEvalWindow(config: *const model_mod.ModelConfig, tokens: u64, span: u64, kv_per_tok: u64) u64 {
    // The layers `kv_per_tok` is the sum over, not every caching layer: on a
    // ringed arch those are nine of forty-eight.
    const attn = config.kvPerTokenLayerCount();
    if (attn == 0) return 0;
    const window: u32 = if (trunk.Trunk.prefillEvalCadenceApplies(@intCast(@min(span, 1 << 20))))
        attnLayersPerEvalWindow(config, trunk.Trunk.MOE_EVAL_EVERY_N_LAYERS)
    else
        attn;
    if (window == 0) return 0;
    // The old buffer's own size, spread over the layers that cache.
    return tokens *| kv_per_tok / attn *| window;
}

/// Per-request bytes that scale with the prompt rather than the chunk.
pub const PrefillRequestTerms = struct {
    /// Cache capacity reserved at admission beyond `seq * kv_per_tok`: the generation headroom
    /// plus a chunk of slack, so no buffer grows during the prefill (#353).
    reserved_kv_bytes: u64 = 0,
    /// Per-request state the arch keeps outside the KV cache (the qwen4_exp QSA indexer history), at its reserved size.
    state_bytes: u64 = 0,
    /// SSM checkpoints the prefill is still holding when it reaches its last chunk.
    checkpoint_bytes: u64 = 0,
    /// Bytes of the KV terms above already resident in the buffer the restore handed this slot
    /// (`WarmPrefix`). KV only; subtracted once, in `prefillMemoryNeeded`.
    shared_resident_bytes: u64 = 0,
    /// Old KV buffers that coexist with the new ones while a warm append, or the grow to the
    /// reservation before it, GROWS the cache. Zero when nothing grows, and on every arch that
    /// does not reserve its capacity.
    grow_coexist_bytes: u64 = 0,
    qsa_ring_bytes: u64 = 0,
    mtp_head_kv_bytes: u64 = 0,
    /// The DFlash2 assistant's window and capture rows (`dflashRequestBytes`).
    dflash_bytes: u64 = 0,
    /// The bill's net over the summed terms, in sixteenths (20 = the historical quarter).
    uplift_16ths: u64 = 20,
    /// The dense rebuild ONE quantized-KV layer really makes at this prompt
    /// length (`kvDequantScratchBytes`). Zero = the generic
    /// `2 · seq · kv_heads · hdim · 2` below, which is exact for every arch
    /// whose caching layers share one head count and one width.
    dequant_scratch_bytes: u64 = 0,
};

pub fn prefillMemoryNeeded(seq: u64, heads: u64, kv_heads: u64, kv_per_tok: u64, hdim: u64, score_hdim: u64, hidden: u64, ffn: u64, kv_bits: u64, chunk: u64, attn_keys: u64, stream_per_tok: u64, dequant_weights: u64, req: PrefillRequestTerms) u64 {
    // A forward is never wider than the prompt: a prompt shorter than the
    // chunk runs ONE seq-wide forward, so the per-chunk transient envelope
    // scales with min(chunk, seq) — billing the raw chunk cap rejected a
    // 31-token prompt for "needing" 6.2 GB on a memory-squeezed host
    // (live 2026-07-14). No-op for prompts >= one chunk.
    const fwd: u64 = @min(chunk, @max(seq, 1));
    const kv_bytes: u64 = seq * kvBytesPerTokenAtBits(kv_per_tok, kv_bits);
    const scores: u64 = if (!trunk.prefillHeadDimFused(@intCast(score_hdim))) heads * fwd * @min(attn_keys, seq) * 2 else 0;
    const dequant: u64 = if (kv_bits >= 16) 0 else if (req.dequant_scratch_bytes > 0)
        req.dequant_scratch_bytes
    else
        2 * seq * kv_heads * hdim * 2;
    const mlp: u64 = 8 * fwd * @max(hidden, ffn) * 2;
    // The MLP envelope alone is not the whole per-token working set: an arch
    // with its own prefill streams (linear-attention layers, MoE gather-sort
    // expansion — `prefillStreamBytesPerToken`) adds to it. Floored at the
    // historical 3-envelope bill so no arch's admission ever LOOSENS: that
    // figure is conservative on plain attention archs and would be a 33%
    // UNDER-bill on a GatedDeltaNet hybrid.
    const envelope: u64 = @max(3 * mlp, mlp + fwd * stream_per_tok);
    // The dequant+GEMM prefill route (verify_moe.prefillDqGemm) materializes
    // a bf16 copy of the weight it is about to multiply plus its transpose,
    // and only at forwards wide enough to take that route. Measured against
    // the route off: ~0 at chunk 8192 where the envelope already dominates.
    const dq_weights: u64 = if (fwd >= trunk.PREFILL_DQ_GEMM_MIN_M) dequant_weights else 0;
    const gross: u64 = kv_bytes + req.reserved_kv_bytes + req.state_bytes + req.checkpoint_bytes +
        req.grow_coexist_bytes + req.qsa_ring_bytes + req.mtp_head_kv_bytes + req.dflash_bytes + scores + dequant + envelope + dq_weights + PREFILL_RUNTIME_FLOOR_BYTES;
    // The one place a warm turn's resident rows leave the bill, inside the 5/4.
    return (gross -| req.shared_resident_bytes) * req.uplift_16ths / 16;
}

/// Per-TOKEN bytes of prefill working set that live OUTSIDE the MLP envelope,
/// because the arch runs streams the envelope does not model. MEASURED as the
/// slope of peak-above-steady-state against the prefill chunk.
///
/// Two families need it. A linear-attention hybrid holds one chunk-wide q/k/v
/// stream per LINEAR layer — all of them, not the ~3 the eval cadence bounds.
/// A MoE prefill sorts and gathers,
/// which replicates the hidden stream `top_k` times per layer alongside the
/// expert rows, for the 4 layers the MoE eval cadence lets coexist.
///
/// The linear term bills only `cadence + 1` layers: since the conv_state tail is
/// an owned copy evaluated at each cadence point, older layers' chunk inputs are
/// released inside the loop (the old all-layers figure WAS that pin).
///
/// Attention-only archs return 0 and keep exactly the bill they had.
pub fn prefillStreamBytesPerToken(config: *const model_mod.ModelConfig) u64 {
    var per_tok: u64 = 0;
    const linear_layers: u64 = @as(u64, config.num_hidden_layers) -| config.attnCacheLayerCount();
    if (config.linear_num_value_heads > 0 and linear_layers > 0) {
        const qkv: u64 = 2 * @as(u64, config.linear_num_key_heads) * config.linear_key_head_dim +
            @as(u64, config.linear_num_value_heads) * config.linear_value_head_dim;
        const cadence_window: u64 = @as(u64, trunk.Trunk.PREFILL_EVAL_CADENCE_DEFAULT) + 1;
        per_tok += @min(linear_layers, cadence_window) * qkv * 2;
    }
    if (config.isMoe()) {
        const top_k: u64 = @max(@as(u64, config.num_experts_per_tok), 1);
        per_tok += MOE_PREFILL_COEXIST * top_k * 2 *
            (@as(u64, config.hidden_size) + config.moe_intermediate_size) * 2;
    }
    // A ringed sliding layer stages the WHOLE chunk before it compacts back to
    // its window, so the rows exist for the width of the forward and the
    // per-token KV term (global layers only) does not carry them. Bounded by
    // the same eval-cadence window as the linear term above: older layers'
    // staging is released inside the loop, so five coexist, not thirty-nine.
    per_tok += config.swaStreamBytesPerToken(
        @as(u64, trunk.Trunk.PREFILL_EVAL_CADENCE_DEFAULT) + 1,
    );
    return per_tok;
}

/// The dequantized-weight working set `verify_moe.prefillDqGemm` materializes
/// (a bf16 copy of the weight plus its transpose, ~3 alive across the layer
/// loop). Chunk-INDEPENDENT — it is weights, not activations — and billed only
/// where the route can fire: affine-quantized weights, and (inside
/// `prefillMemoryNeeded`) forwards at least `PREFILL_DQ_GEMM_MIN_M` wide.
/// Reads the same kill switch the route does, so an A/B with the route off
/// does not get billed for it.
fn prefillDequantWeightBytes(config: *const model_mod.ModelConfig) u64 {
    if (config.quant_bits == 0 or config.quant_mode != .affine) return 0;
    if (!trunk.prefillDqGemmEnabled()) return 0;
    return 3 * prefillFfnWidth(config) * @as(u64, config.hidden_size) * 2;
}

pub fn prefillFfnWidth(config: *const model_mod.ModelConfig) u64 {
    const per_tok: u64 = @max(@as(u64, config.num_experts_per_tok), 1);
    const moe_w: u64 = @as(u64, config.moe_intermediate_size) * per_tok +
        config.shared_expert_intermediate_size;
    const dense_w: u64 = if (config.intermediate_size_declared) config.intermediate_size else 0;
    const w = @max(dense_w, moe_w);
    return if (w > 0) w else config.intermediate_size;
}

/// Estimate peak GPU memory for prefill and reject if it would exceed the Metal
/// working-set ceiling. Exceeding it throws an uncatchable C++ exception on a
/// Metal completion-handler thread and kills the process, so PREVENTION is the
/// only lever — this guard must track what the prefill actually allocates
/// (prefillMemoryNeeded above; chunk choice from prefill_plan.effectivePrefillChunk).
/// `kv_override` is the per-request `kv_quant` body field where the surface
/// parses one (chat/messages/responses); null falls back to the process default.
/// Per-token bytes of the arch's out-of-cache request state as billed: the QSA indexer history
/// at the one copy a live slot holds, plus the f32 block-score bank. One helper because the
/// auto-context sizer and the admission guard must agree.
pub fn statePerTokenBilled(config: *const model_mod.ModelConfig) u64 {
    if (archHook("stateBilled", u64, config, .{})) |bytes| return bytes;
    const one = config.qsaHistoryBytesPerToken();
    if (one == 0) return 0;
    return one + config.qsaScoreBankBytesPerToken();
}

/// Tokens of cache capacity a request reserves at admission (prompt + generation + one chunk
/// of slack); `KVCache.reserve_tokens` allocates it on the first grow so the prefill never
/// runs a second grow. The guard and the allocator are the same number by construction.
pub fn reservedCacheTokens(seq: u64, max_tokens: u64, chunk: u64, ctx: u64) u64 {
    return trunk.KVCache.reservedTokens(seq, max_tokens, chunk, ctx);
}

/// Bytes of SSM checkpoints a prefill is still holding at its last chunk (~1.9 GB on a 36-layer
/// GDN trunk, billed at zero before #353). Mirrors the prefill loop's stride. `matched` is the
/// restored prefix: a warm prefill captures only over `seq - matched` (the entry's own
/// checkpoints are already resident and merged, never re-allocated).
pub fn retainedSsmCheckpointBytes(config: *const model_mod.ModelConfig, seq: u64, matched: u64, chunk: u64) u64 {
    const per_cp = config.ssmCheckpointBytes();
    if (per_cp == 0 or launch_state.ssm_checkpoint_stride == 0) return 0;
    if (archHook("retainedCheckpoints", u64, config, .{ seq, matched })) |bytes| return bytes;
    const stride: u64 = prefill_plan.effectiveSsmCheckpointStride(launch_state.ssm_checkpoint_stride, @max(@as(usize, @intCast(chunk)), settings_resolve.prefill_chunk.value));
    if (stride == 0) return 0;
    // Stride-aligned capture positions in `(matched, seq]` plus the always-on end-of-prompt
    // snapshot, the only capture a warm append shorter than one stride makes.
    const span: u64 = seq -| @min(matched, seq);
    var n: u64 = seq / stride -| @min(matched, seq) / stride;
    if (span > prefill_plan.SSM_SNAPSHOT_BACKOFF) n += 1;
    if (launch_state.ssm_checkpoint_max > 0) n = @min(n, launch_state.ssm_checkpoint_max);
    if (ssmResidentCap(config)) |cap| n = @min(n, cap);
    return n * per_cp;
}

/// The checkpoint state a thinking request holds past its prefill's, for the arch that takes one.
pub fn thoughtPointBytes(config: *const model_mod.ModelConfig, warm: WarmPrefix) u64 {
    return archHook("thoughtPoint", u64, config, .{warm}) orelse 0;
}

/// The restore points a request's prefill holds in RAM, for the arch that keeps them.
pub fn checkpointReserve(config: *const model_mod.ModelConfig, seq: u64, chunk: u64) u64 {
    return archHook("checkpointReserve", u64, config, .{ seq, chunk }) orelse 0;
}

/// The most checkpoint states a request holds at once when its arch runs an SSD tier; null = every one stays in RAM.
pub fn ssmResidentCap(config: *const model_mod.ModelConfig) ?u64 {
    return archHook("residentCheckpointCap", u64, config, .{});
}

/// Per-token bytes of the MTP head's own KV, billed once per request when MTP is on (the arch that has a head says).
pub fn mtpHeadKvBytesPerToken(config: *const model_mod.ModelConfig) u64 {
    return archHook("headKvPerToken", u64, config, .{}) orelse 0;
}

pub fn mtpHeadQsaRingBytes(config: *const model_mod.ModelConfig) u64 {
    return archHook("headQsaRing", u64, config, .{}) orelse 0;
}

/// The fixed rings and catch-up rows of the arch's MTP head, billed once per request.
fn mtpHeadRingBytes(config: *const model_mod.ModelConfig) u64 {
    return archHook("headRings", u64, config, .{}) orelse 0;
}

pub fn mtpHeadStateBytesPerToken(config: *const model_mod.ModelConfig) u64 {
    return archHook("headStatePerToken", u64, config, .{}) orelse 0;
}

pub fn sessionBytesPerToken(config: *const model_mod.ModelConfig, kv_bits: u64) u64 {
    if (archHook("sessionBytes", u64, config, .{kv_bits})) |bytes| return bytes;
    const head: u64 = if (mtpHeadDefaultOn(config)) mtpHeadKvBytesPerToken(config) +| mtpHeadStateBytesPerToken(config) else 0;
    return kvBytesPerTokenAtBits(config.kvBytesPerToken(), kv_bits) +| statePerTokenBilled(config) +| head;
}

/// The sliding DFlash2 assistant's per-request bytes on a non-GLM target: the physical window and
/// its trim copies, plus the tap and encoder rows of one prefill chunk.
pub fn dflashRequestBytes(config: *const model_mod.ModelConfig, chunk: u64) u64 {
    if (!config.dflash_loaded or config.isGlm5()) return 0;
    return config.dflash_window_bytes *| 4 +| config.dflash_capture_bytes_per_token *| chunk +| 64 * 1024 * 1024;
}

/// What a GLM DFlash2 request may still allocate when it reserves its verifier capacity right after
/// prefill: its admission bill at the narrowest width (no width bills less) less what it holds.
pub fn glmDflashReserveBudget(config: *const model_mod.ModelConfig, seq: u64, max_tokens: u32, kv_bits: u64, held: u64) u64 {
    // `max_tokens` is already clamped to admission's context; the caller's config copy is not pinned to it.
    var sized = config.*;
    sized.pinned_context = @intCast(@min(seq +| max_tokens, std.math.maxInt(u32)));
    return prefillNeededAtChunk(&sized, seq, max_tokens, kv_bits, 1, .{}) -| held;
}

/// The per-request terms of the admission bill, in one place.
pub fn prefillRequestTerms(config: *const model_mod.ModelConfig, seq: u64, max_tokens: u64, kv_bits: u64, chunk: u64, warm: WarmPrefix) PrefillRequestTerms {
    // Arch gate for every term (all new, all measured on qwen4_exp alone; the reservation's
    // allocator twin is gated too, so an ungated guard billed memory never reserved). `.{}` is
    // the identity: `prefillMemoryNeeded` then reduces to the previous expression. A ringed arch
    // joins through the narrower `reservesKvCapacity`; every term it does not have stays zero on
    // its own (no QSA history, no SSM checkpoints); its MTP heads bill `mimo_mtp.State.billedBytes`.
    if (!config.reservesKvCapacity()) return .{};
    // `reservedTokens` returns 0 below its length threshold; floor the reserved length at `seq`.
    const reserved = @max(reservedCacheTokens(seq, max_tokens, chunk, getEffectiveContextLength(config)), seq);
    // Only the headroom is new here: the prompt's own rows are already billed.
    const kv_per_tok = kvBytesPerTokenAtBits(config.kvBytesPerToken(), kv_bits);
    const mtp_on = warm.mtp_on or (forceMtpFor(config) and !config.dflash_loaded);
    const head_per_tok: u64 = if (mtp_on) mtpHeadKvBytesPerToken(config) else 0;
    const head_qsa_ring: u64 = if (mtp_on) mtpHeadQsaRingBytes(config) else 0;
    const head_state_per_tok: u64 = if (mtp_on) mtpHeadStateBytesPerToken(config) else 0;
    // Rows the restore hands over without allocating another copy. KV only: the history is
    // privatised on restore (`seedCapBuf`), so crediting it cancelled the one-copy bill outright.
    const credited = warm.creditedRows(reserved);
    return .{
        .reserved_kv_bytes = (reserved -| seq) * kv_per_tok,
        .state_bytes = reserved * (statePerTokenBilled(config) +| head_state_per_tok) +|
            (if (mtp_on) mtpHeadRingBytes(config) else 0),
        // The warm span, not the prompt; read regardless of `will_donate` (a shared restore skips the same rows).
        .checkpoint_bytes = retainedSsmCheckpointBytes(config, seq, warm.matched_tokens, chunk) +| thoughtPointBytes(config, warm),
        .shared_resident_bytes = credited *| kv_per_tok,
        .grow_coexist_bytes = growCoexistBytes(config, warm, seq, reserved, kv_per_tok),
        .qsa_ring_bytes = slotRingBytes(config, kv_bits) +| head_qsa_ring,
        .mtp_head_kv_bytes = reserved *| head_per_tok,
        .dflash_bytes = dflashRequestBytes(config, @min(chunk, @max(seq, 1))),
        .dequant_scratch_bytes = kvDequantScratchBytes(config, seq, @min(chunk, seq)),
        .uplift_16ths = billUplift16ths(config, seq),
    };
}

/// The body of the pre-flight memory 400. Two arms: with the credits present the message
/// quotes them; with them structurally zero (every arch outside `admissionEvictsHotCache`)
/// the previous sentence, which named no cache. The discriminator is the arch gate, not `bill.evictable`,
/// so the qwen4_exp bytes stay exactly as they were. Caller owns the returned bytes.
pub fn memoryRefusalMessage(
    allocator: std.mem.Allocator,
    prompt_len: usize,
    needed_mb: u64,
    avail_mb: u64,
    bill: AdmissionBill,
    carries_cache: bool,
) ![]u8 {
    if (!carries_cache) {
        return std.fmt.allocPrint(allocator, "Prompt ({d} tokens) requires ~{d}MB GPU memory but only ~{d}MB available. Reduce prompt size or use a smaller model.", .{ prompt_len, needed_mb, avail_mb });
    }
    return std.fmt.allocPrint(allocator, "Prompt ({d} tokens) requires ~{d}MB GPU memory but only ~{d}MB is available and the hot prefix cache holds ~{d}MB more, all of which can be reclaimed (~{d}MB — the shortfall is elsewhere), which is not enough. Reduce prompt size, lower --ctx-size, or use a smaller model.", .{ prompt_len, needed_mb, avail_mb, bill.evictable / (1024 * 1024), bill.evictionCredit() / (1024 * 1024) });
}

/// The admission bill and the headroom it is compared against, so the connection thread's
/// guard and the inference thread's evict-or-refuse decision cannot compute them differently.
pub const AdmissionBill = struct {
    needed: u64 = 0,
    available: u64 = 0,
    /// Bytes the hot prefix cache is holding. Reported in a refusal, never credited (see `reclaimable`).
    evictable: u64 = 0,
    /// The part of `evictable` an eviction can prove it will return: residency minus the
    /// largest entry (a restore pins at most one).
    reclaimable: u64 = 0,
    /// The prefill width this bill was taken at, so a refusal can name it.
    chunk: u64 = 0,

    pub fn fits(self: AdmissionBill) bool {
        return self.needed <= self.available;
    }

    /// The one credit the decision and both sentences read: `reclaimable`, never `evictable`
    /// (a refusal once quoted the slot's own checked-out prefix as "evicted").
    pub fn evictionCredit(self: AdmissionBill) u64 {
        return self.reclaimable;
    }

    /// Fits once the cache has given back what it provably can. Credits `reclaimable`, not
    /// `evictable`: a 383k prompt was admitted on one entry it then restored from and refused
    /// after the eviction pass returned nothing.
    pub fn fitsAfterEviction(self: AdmissionBill) bool {
        return self.needed <= self.available +| self.evictionCredit();
    }
};

/// Unloads idle resident models, least recently used first, until `fits` holds; false when
/// nothing idle is left. An idle model is memory its next request reloads.
pub fn fitAfterUnloadingIdle(ctx: anytype, fits: fn (@TypeOf(ctx)) bool, unloadOne: fn (@TypeOf(ctx)) bool) bool {
    while (!fits(ctx)) if (!unloadOne(ctx)) return false;
    return true;
}

/// Everything the admission guard bills for one request at one prefill width, so the
/// per-request chunk chooser prices a rung with the same function that admits it. `chunk` is
/// the width the forward will run, already through `prefill_plan.effectivePrefillChunk`.
pub fn prefillNeededAtChunk(
    config: *const model_mod.ModelConfig,
    seq: u64,
    max_tokens: u32,
    kv_bits: u64,
    chunk: u64,
    warm: WarmPrefix,
) u64 {
    if (archHook("requestBill", u64, config, .{ seq, max_tokens, kv_bits, chunk, warm })) |bill| return bill;
    const heads: u64 = config.num_attention_heads;
    const kv_heads: u64 = config.num_key_value_heads;
    const hdim: u64 = config.head_dim;
    const hidden: u64 = config.hidden_size;
    const ffn: u64 = prefillFfnWidth(config);
    return prefillMemoryNeeded(seq, heads, kv_heads, config.kvBytesPerToken(), hdim, config.prefillScoreHeadDim(), hidden, ffn, kv_bits, chunk, config.prefillAttnKeys(seq), prefillStreamBytesPerToken(config), prefillDequantWeightBytes(config), prefillRequestTerms(config, seq, max_tokens, kv_bits, chunk, warm)) +
        qsaMaskBytes(config, @min(chunk, @max(seq, 1)), seq) +
        slidingBandScoreBytes(config, @min(chunk, @max(seq, 1))) +
        fp8DequantScratchBytes(config, @min(chunk, @max(seq, 1))) +
        trunkDecodeScratchBytes(config, @min(chunk, @max(seq, 1))) +
        (if (config.expert_streaming) config.expert_fill_peak_bytes else 0) +
        cadenceInflightBytes(mlx_mb_per_buffer);
}

/// Headroom a rung must clear beyond its own bill when the long-context gate is what widened
/// it: past that width the machine cap frozen at load no longer bounds the forward, and
/// `prefillMemoryNeeded` is documented to UNDER-bill quantized prefill by about this much.
/// Narrower rungs keep the plain bill, so no other admission decision moves.
pub const PREFILL_WIDE_RUNG_MARGIN_PCT: u64 = 22;

/// Test seam: false makes the load-time pin the width again.
pub var per_request_chunk_override: ?bool = null;

pub fn perRequestPrefillChunkEnabled(config: *const model_mod.ModelConfig) bool {
    if (!config.perRequestPrefillChunk()) return false;
    return per_request_chunk_override orelse true;
}

/// The widest ladder rung this request can afford, priced by the same estimator that admits
/// it against post-eviction memory. Candidates are priced at the width the forward will run
/// (`effectivePrefillChunk` caps by arch), and the width is returned, not the rung. An explicit
/// `--prefill-chunk` (`chunk_override`) caps every rung; nothing fitting means the capped floor.
pub fn chooseRequestPrefillChunk(
    config: *const model_mod.ModelConfig,
    seq: u64,
    max_tokens: u32,
    kv_bits: u64,
    available: u64,
    load_time_pin: u32,
    chunk_override: u32,
    warm: WarmPrefix,
) u32 {
    // The arch gate first: an ungated arch's answer is the load-time pin, as before.
    if (!perRequestPrefillChunkEnabled(config)) return load_time_pin;
    const top = ladderTop(config, chunk_override);
    // GLM's checkpoints yield to the width: a request keeps those the chosen width leaves room for.
    var rung_warm = warm;
    rung_warm.checkpoints = 0;
    for (PREFILL_CHUNK_LADDER) |r| {
        const rung = cappedRung(r, top);
        const width: u64 = rungWidth(config, seq, rung, config.longCtxGated());
        const bill = prefillNeededAtChunk(config, seq, max_tokens, kv_bits, width, rung_warm);
        const needed = if (width > rungWidth(config, seq, rung, false))
            bill *| (100 + PREFILL_WIDE_RUNG_MARGIN_PCT) / 100
        else
            bill;
        if (needed <= available) return @intCast(width);
    }
    return @intCast(rungWidth(config, seq, cappedRung(PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1], top), config.longCtxGated()));
}

fn cappedRung(rung: u32, cap: u32) u32 {
    return if (cap == 0) rung else @min(rung, cap);
}

/// The widest rung a request may take: an explicit `--prefill-chunk`, else the arch's start width.
pub fn ladderTop(config: *const model_mod.ModelConfig, chunk_override: u32) u32 {
    return if (chunk_override > 0) chunk_override else config.prefillStartWidth();
}

/// The width one ladder rung forwards at. `long_ctx_gated = false` asks what the rung would be
/// without the long-context gate, i.e. the width the load-time machine cap still bounds.
pub fn rungWidth(config: *const model_mod.ModelConfig, seq: u64, rung: u32, long_ctx_gated: bool) u64 {
    // `--prefill-chunk` is unbounded at its source; clamp before narrowing.
    return @min(
        @as(u64, prefill_plan.requestPrefillChunk(
            config.prefillScoreHeadDim(),
            config.num_attention_heads,
            @intCast(seq),
            config.has_sliding_window,
            config.isMoe(),
            long_ctx_gated,
            rung,
        )),
        @as(u64, std.math.maxInt(u32)),
    );
}

/// The width handed to the forward (`Generator.pinned_prefill_chunk`): one rule for the
/// connection thread's bill and the scheduler's pick. Off the ladder (the vision kill switch, an
/// arch that does not pick per request) an explicit width outranks the load-time pin.
pub fn requestPrefillPick(config: *const model_mod.ModelConfig, seq: u64, max_tokens: u32, kv_bits: u64, available: u64, unchunked_prefill: bool, warm: WarmPrefix) u32 {
    const explicit = explicitPrefillChunk();
    if (unchunked_prefill or !perRequestPrefillChunkEnabled(config))
        return if (explicit > 0) explicit else config.pinned_prefill_chunk;
    return chooseRequestPrefillChunk(config, seq, max_tokens, kv_bits, available, config.pinned_prefill_chunk, explicit, warm);
}

/// Impure wrapper the scheduler reaches through `prefill_request_chunk`, mirroring
/// `prefillFitsNow`'s inputs. Reads live memory, so it must run after the eviction pass.
pub fn requestPrefillChunkNow(
    config: *const model_mod.ModelConfig,
    prompt_len: usize,
    max_tokens: u32,
    kv_cfg: trunk.KVQuantConfig,
    unchunked_prefill: bool,
    warm_matched: u64,
    warm_capacity: u64,
    warm_will_donate: bool,
    enable_mtp: bool,
) u32 {
    const pin: u32 = config.pinned_prefill_chunk;
    const seq: u64 = @intCast(prompt_len);
    const kv_bits: u64 = if (kv_cfg.scheme == .off) 16 else kv_cfg.bits;
    const active_mem: usize = gpu_mem.activeBytes();
    const available: u64 = currentGpuMemoryCeiling(config, active_mem) -| active_mem;
    const warm = WarmPrefix{ .matched_tokens = warm_matched, .capacity_tokens = warm_capacity, .will_donate = warm_will_donate, .mtp_on = enable_mtp };
    const chosen = requestPrefillPick(config, seq, max_tokens, kv_bits, available, unchunked_prefill, warm);
    if (chosen != pin and !unchunked_prefill and perRequestPrefillChunkEnabled(config)) {
        const width: u64 = chosen;
        const terms = prefillRequestTerms(config, seq, max_tokens, kv_bits, width, warm);
        const kv_bytes: u64 = seq *| kvBytesPerTokenAtBits(config.kvBytesPerToken(), kv_bits) +| terms.reserved_kv_bytes;
        log.debug("[prefill] chunk {d} for this request (reserve {d} MB beside KV {d} MB)\n", .{ width, prefillTransientReserve(config, kv_bits, width) >> 20, kv_bytes >> 20 });
    }
    return chosen;
}

/// Test seam: false turns the per-chunk adaptive width off.
pub var adaptive_chunk_override: ?bool = null;

/// The per-chunk adaptive width: subordinate to the per-request gate, plus "no operator pinned a width".
pub fn adaptivePrefillChunkEnabled(config: *const model_mod.ModelConfig) bool {
    // Arch first: the cheapest and most selective, and this runs once per chunk boundary.
    if (!config.perRequestPrefillChunk()) return false;
    if (explicitPrefillChunk() > 0) return false;
    if (!perRequestPrefillChunkEnabled(config)) return false;
    return adaptive_chunk_override orelse true;
}

/// Consecutive supporting probes a widening needs: the probe is taken after `mlx_clear_cache`
/// and never contains a chunk's own peak, so the second reading pays for that blind spot.
pub const PREFILL_WIDEN_PROBES: u8 = 2;

/// What one chunk of `width` costs beyond what is already resident: its transient envelope
/// plus the per-token state the chunk is about to write. The growth term is billed even under
/// the reservation (short prompts do grow per chunk).
pub fn prefillChunkCost(
    config: *const model_mod.ModelConfig,
    kv_bits: u64,
    width: u64,
    kv_len: u64,
) u64 {
    const per_tok: u64 = kvBytesPerTokenAtBits(config.kvBytesPerToken(), kv_bits) +|
        statePerTokenBilled(config);
    return (prefillTransientReserveAtKv(config, kv_bits, width, kv_len) -| dsv41BorrowableBytes(config)) +| (width *| per_tok);
}

/// The decoder pools a DeepSeek prefill may still release for its working set (`dsv41_borrow`); 0 elsewhere.
pub fn dsv41BorrowableBytes(config: *const model_mod.ModelConfig) u64 {
    return @atomicLoad(u64, &config.expert_borrow_bytes, .monotonic);
}

/// `dsv41_borrow.headroom_fn`: what admission reads as available, live.
pub fn dsv41Headroom(config: *const model_mod.ModelConfig) u64 {
    const active_mem: usize = gpu_mem.activeBytes();
    return dsv41HeadroomOf(admissionAvailable(currentGpuMemoryCeiling(config, active_mem), active_mem), config);
}

/// The decode pool the ledger parks in MLX's cache is no part of its active bytes, so every live check (decode and
/// prefill width) takes it off here once; `prefillNeededAtChunk` bills it once for the same reason.
fn dsv41HeadroomOf(available: u64, config: *const model_mod.ModelConfig) u64 {
    return available -| config.expertPoolBytes();
}

/// The width the next chunk runs at. Asymmetric because a Metal abort is uncatchable: HOLD
/// while `headroom >= cost(current)` (the estimator carries its own 5/4); STEP DOWN at once,
/// by as many rungs as it takes, floored at `PREFILL_CHUNK_FLOOR`; WIDEN one rung only at
/// 1.25x its cost, after `PREFILL_WIDEN_PROBES` supporting probes, up to `cap`, never after a
/// step-down. `cap` is the widest width this arch forwards for this prompt, not a raw rung.
pub fn adaptivePrefillWidth(
    config: *const model_mod.ModelConfig,
    kv_bits: u64,
    kv_len: u64,
    headroom: u64,
    current: u32,
    cap: u32,
    st: *mtp_plan.AdaptiveWidthState,
) u32 {
    if (cap == 0 or current == 0) return current;
    const floor: u32 = @min(cap, @as(u32, @intCast(prefill_plan.PREFILL_CHUNK_FLOOR)));

    if (prefillChunkCost(config, kv_bits, current, kv_len) > headroom) {
        st.supporting = 0;
        st.ratcheted = true;
        var best: u32 = floor;
        for (PREFILL_CHUNK_LADDER) |rung| {
            const w: u32 = @min(rung, cap);
            if (w >= current) continue;
            if (prefillChunkCost(config, kv_bits, w, kv_len) <= headroom) {
                best = w;
                break;
            }
        }
        return @min(best, current);
    }

    var next: u32 = 0;
    for (PREFILL_CHUNK_LADDER) |rung| {
        const w: u32 = @min(rung, cap);
        if (w > current and (next == 0 or w < next)) next = w;
    }
    if (next == 0 or st.ratcheted) {
        st.supporting = 0;
        return current;
    }
    const asked: u64 = prefillChunkCost(config, kv_bits, next, kv_len) *| 5 / 4;
    if (asked > headroom) {
        st.supporting = 0;
        return current;
    }
    st.supporting +|= 1;
    if (st.supporting < PREFILL_WIDEN_PROBES) return current;
    st.supporting = 0;
    return next;
}

/// The live headroom one prefill chunk has to fit in: the GPU ceiling less what MLX's
/// allocator holds, read once per chunk boundary after its `mlx_clear_cache`. The same
/// expression `prefillAdmissionBill` compares against.
pub fn prefillHeadroomNow(config: *const model_mod.ModelConfig, staged_host: u64) u64 {
    const active_mem: usize = gpu_mem.activeBytes();
    // The SSD write-through stages chunk blobs as host bytes behind a ~1 GiB permit that
    // `mlx_get_active_memory` does not count; on unified memory they are the same pool. Read at
    // the hook site (inference thread) and passed in.
    return prefillHeadroomFrom(currentGpuMemoryCeiling(config, active_mem), active_mem, staged_host);
}

/// Pure half of `prefillHeadroomNow`.
pub fn prefillHeadroomFrom(ceiling: u64, active: u64, staged_host: u64) u64 {
    return ceiling -| active -| staged_host;
}

/// Does a widen the pre-tick probe chose still fit after the interleave tick allocated for
/// co-tenant slots? Same 1.25 bar, same estimator, same live KV.
pub fn adaptivePrefillWidenStillFits(
    config: *const model_mod.ModelConfig,
    kv_bits: u64,
    pos: usize,
    want: u32,
    staged_host: u64,
) bool {
    if (!adaptivePrefillChunkEnabled(config)) return false;
    const headroom = prefillHeadroomNow(config, staged_host);
    const asked: u64 = prefillChunkCost(config, kv_bits, want, pos) *| 5 / 4;
    if (asked <= headroom) return true;
    log.info("[prefill] widen to {d} withdrawn at pos {d} (headroom {d} MB after the decode tick, cost {d} MB)\n", .{
        want,
        pos,
        headroom >> 20,
        asked >> 20,
    });
    return false;
}

/// Impure wrapper the prefill loop reaches through `scheduler.prefill_chunk_adapt`. Logs one
/// line per change, never per boundary (the bench harness greps it).
pub fn adaptivePrefillWidthNow(
    config: *const model_mod.ModelConfig,
    kv_bits: u64,
    pos: usize,
    current: u32,
    cap: u32,
    st: *mtp_plan.AdaptiveWidthState,
    staged_host: u64,
) u32 {
    if (!adaptivePrefillChunkEnabled(config)) return current;
    const headroom = prefillHeadroomNow(config, staged_host);
    // `pos` is the live KV this chunk will attend to (see `prefillTransientReserveAtKv`).
    const next = adaptivePrefillWidth(config, kv_bits, pos, headroom, current, cap, st);
    if (next != current) {
        // Quote the number it compared: `prefillChunkCost` at this KV.
        log.info("[prefill] width {d} -> {d} at pos {d} (headroom {d} MB, cost {d} MB)\n", .{
            current,
            next,
            pos,
            headroom >> 20,
            prefillChunkCost(config, kv_bits, next, pos) >> 20,
        });
    }
    return next;
}

/// What a request may allocate: the ceiling less MLX's allocations. The SSD writers' host bytes are
/// pageable memory outside the wired limit, so the ceiling's margin under it is the only reserve.
pub fn admissionAvailable(total_limit: u64, active_mem: u64) u64 {
    return total_limit -| active_mem;
}

/// The hot cache is read once, as the published evictable atomic, never billed on top.
pub fn prefillAdmissionBill(config: *const model_mod.ModelConfig, prompt_len: usize, max_tokens: u32, kv_override: ?trunk.KVQuantConfig, unchunked_prefill: bool, prompt_tokens: ?[]const u32, warm: WarmPrefix) AdmissionBill {
    const heads = config.num_attention_heads;
    if (heads == 0) return .{ .needed = 0, .available = std.math.maxInt(u64) };
    const seq: u64 = @intCast(prompt_len);

    const kv_cfg: trunk.KVQuantConfig = kv_override orelse configuredKvQuantFor(config);
    const kv_bits: u64 = if (kv_cfg.scheme == .off) 16 else kv_cfg.bits;
    // Available = the lesser of Metal's static working-set max and what is physically reachable
    // now (#64). Read before the width is chosen: on a per-request arch the width is a function of it.
    const active_mem: usize = gpu_mem.activeBytes();
    const total_limit: u64 = currentGpuMemoryCeiling(config, active_mem);
    const available = admissionAvailable(total_limit, active_mem);

    // A vision prefill chunks like text since issue #197 (the splice resumes
    // its row index across chunks), so it bills the chunk-bounded envelope —
    // UNLESS the SUSHI_VISION_CHUNKED=0 kill switch restored the
    // whole-prompt forward, in which case `unchunked_prefill` bills the real
    // width. Call sites pass prefill_plan.visionPrefillUnchunked(has_vision)
    // so the guard and the prefill loop cannot disagree.
    //
    // Otherwise the width comes from `requestPrefillPick`, the same rule the scheduler pins with.
    const chosen: u32 = if (unchunked_prefill) 0 else requestPrefillPick(config, seq, max_tokens, kv_bits, available, false, warm);
    const chunk: u64 = if (unchunked_prefill)
        @max(seq, 1)
    else
        @intCast(prefill_plan.requestPrefillChunk(config.prefillScoreHeadDim(), config.num_attention_heads, prompt_len, config.has_sliding_window, config.isMoe(), config.longCtxGated(), chosen));
    // RAM hot-cache restores rebind MLX array handles by refcount; they do not
    // allocate another copy of the cached buffers. `active_mem` above already
    // includes the resident entry, while `prefillMemoryNeeded` bills the full
    // destination KV capacity that may be allocated when the restored cache
    // grows. Adding the resident entry here again would invent a third copy
    // and reject long warm prompts (and even cache misses) spuriously.
    const needed: u64 = prefillNeededAtChunk(config, seq, max_tokens, kv_bits, chunk, warm);
    // Published number, never the pointer: `hot_prefix_cache` is inference-thread state, freed
    // on every model switch. Stale by at most one entry, which is all a hint has to be.
    const evictable: u64 = if (launch_state.global_scheduler) |sch|
        sch.resident_hot_cache_bytes.load(.monotonic)
    else
        0;
    // Prompt-aware when the caller has the ids: an entry this prompt could not restore from is
    // genuinely evictable. Still a published snapshot, never the cache pointer.
    const reclaimable: u64 = if (launch_state.global_scheduler) |sch|
        if (prompt_tokens) |toks|
            scheduler_mod.reclaimableHotCacheBytesFor(sch, toks)
        else
            sch.reclaimable_hot_cache_bytes.load(.monotonic)
    else
        0;
    return creditedAdmissionBill(config, needed, available, evictable, reclaimable, chunk);
}

/// PURE: the bill with the hot-cache credits on an arch that evicts to admit
/// (`admissionEvictsHotCache`). Elsewhere both credits are zero, so both admit arms go dead and
/// the single `needed > available` refusal stands. The publishers stay on every arch.
pub fn creditedAdmissionBill(config: *const model_mod.ModelConfig, needed: u64, available: u64, evictable: u64, reclaimable: u64, chunk: u64) AdmissionBill {
    if (!config.admissionEvictsHotCache()) {
        return .{ .needed = needed, .available = available, .chunk = chunk };
    }
    return .{ .needed = needed, .available = available, .evictable = evictable, .reclaimable = reclaimable, .chunk = chunk };
}

/// Bytes of the resident hot cache this prompt pins: residency less what an eviction can prove
/// it will return. Non-zero means an entry this prompt would restore from is holding memory
/// that neither goes back nor has to be allocated again; the connection thread cannot price
/// how much, so it defers to the inference thread's `WarmPrefix` bill.
pub fn pinnedResidentBytes(bill: AdmissionBill) u64 {
    return bill.evictable -| bill.evictionCredit();
}

/// What the admission guard decided, from one bill so the log line and the arm agree: fits
/// outright, fits once the hot cache gives back what it provably can, or does not fit.
pub const AdmissionVerdict = enum {
    admit,
    evict,
    refuse,

    pub fn name(self: AdmissionVerdict) []const u8 {
        return switch (self) {
            .admit => "admit",
            .evict => "evict",
            .refuse => "refuse",
        };
    }
};

pub fn admissionVerdict(bill: AdmissionBill) AdmissionVerdict {
    if (bill.fits()) return .admit;
    if (bill.fitsAfterEviction()) return .evict;
    return .refuse;
}

/// The generation of `markReadyLocked` the last info-level admission line was emitted for.
var admission_logged_generation = std.atomic.Value(u64).init(std.math.maxInt(u64));

/// Has a model become resident since the last time this returned true? Consuming, so N racing connection threads produce one line.
fn firstRequestAfterLoad() bool {
    const gen = model_registry_mod.load_generation.load(.monotonic);
    const last = admission_logged_generation.load(.monotonic);
    if (gen == last) return false;
    return admission_logged_generation.cmpxchgStrong(last, gen, .monotonic, .monotonic) == null;
}

/// The level one admission line is emitted at: info on the first request after a load and
/// whenever the bill is within 10% of what is available; debug otherwise.
pub fn admissionLogLevel(needed: u64, available: u64, first_after_load: bool) log.Level {
    if (first_after_load) return .info;
    // `needed > 0.9 * available` without floating point or overflow.
    if (needed *| 10 > available *| 9) return .info;
    return .debug;
}

/// One line per request at the admission decision, on admitted requests too. Every field comes
/// off the same `bill` the arms act on. The level check runs first.
pub fn logAdmissionDecision(bill: AdmissionBill) void {
    // `info` is the loudest level this line is emitted at, and `firstRequestAfterLoad` consumes its token.
    if (!log.enabled(.info)) return;
    const level = admissionLogLevel(bill.needed, bill.available, firstRequestAfterLoad());
    if (!log.enabled(level)) return;
    const mb = 1024 * 1024;
    log.atLevel(level, "[admission] needed={d} MB available={d} MB reclaimable={d} MB width={d} verdict={s}\n", .{
        bill.needed / mb,
        bill.available / mb,
        bill.evictionCredit() / mb,
        bill.chunk,
        admissionVerdict(bill).name(),
    });
}

/// Inference-thread predicate behind the scheduler's admission hook: does this request fit
/// right now, with live memory re-read? Every input is forwarded, none defaulted (a hardcoded
/// `kv_override = null` priced a `kv_quant: 4` request at fp16 and evicted a cache for nothing).
pub fn prefillFitsNow(config: *const model_mod.ModelConfig, prompt_len: usize, max_tokens: u32, kv_cfg: trunk.KVQuantConfig, unchunked_prefill: bool, warm_matched: u64, warm_capacity: u64, warm_will_donate: bool, enable_mtp: bool, checkpoints: scheduler_mod.CheckpointPlan) bool {
    return prefillAdmissionBill(config, prompt_len, max_tokens, kv_cfg, unchunked_prefill, null, .{
        .matched_tokens = warm_matched,
        .capacity_tokens = warm_capacity,
        .will_donate = warm_will_donate,
        .mtp_on = enable_mtp,
        .checkpoints = checkpoints.keep,
        .released = checkpoints.released,
        .thought = checkpoints.thought,
    }).fits();
}

/// The cold bill, live memory re-read, for the scheduler's sibling hold.
pub fn prefillBillNumbersNow(config: *const model_mod.ModelConfig, prompt_len: usize, max_tokens: u32, kv_cfg: trunk.KVQuantConfig, unchunked_prefill: bool, enable_mtp: bool) slot_admission.MemoryBill {
    const bill = prefillAdmissionBill(config, prompt_len, max_tokens, kv_cfg, unchunked_prefill, null, .{ .mtp_on = enable_mtp });
    const kv_bits: u64 = if (kv_cfg.scheme == .off) 16 else kv_cfg.bits;
    return .{ .needed = bill.needed, .available = bill.available, .commit = admissionCommitBytes(config, prompt_len, max_tokens, kv_bits) };
}

/// Bytes the bill prices for a request's cache that it allocates AFTER admission. A native GLM
/// MLA cache grows row by row (a DFlash2 request reserves it after prefill); a Qwen or MiMo
/// request takes its reserved capacity at its first grow, so it is resident before the next admission.
pub fn admissionCommitBytes(config: *const model_mod.ModelConfig, prompt_len: usize, max_tokens: u32, kv_bits: u64) u64 {
    return archHook("commitBytes", u64, config, .{ prompt_len, max_tokens, kv_bits }) orelse 0;
}

/// The inference thread's refusal, quoting the numbers it compared.
pub fn logPrefillRefusal(config: *const model_mod.ModelConfig, prompt_len: usize, max_tokens: u32, kv_cfg: trunk.KVQuantConfig, unchunked_prefill: bool, warm_matched: u64, warm_capacity: u64, warm_will_donate: bool, enable_mtp: bool, checkpoints: scheduler_mod.CheckpointPlan) ?model_registry_mod.MemoryContextRefusal {
    // The same warm inputs the probe was refused on, the checkout decision included.
    const warm = WarmPrefix{
        .matched_tokens = warm_matched,
        .capacity_tokens = warm_capacity,
        .will_donate = warm_will_donate,
        .mtp_on = enable_mtp,
        .checkpoints = checkpoints.keep,
        .released = checkpoints.released,
        .thought = checkpoints.thought,
    };
    const bill = prefillAdmissionBill(config, prompt_len, max_tokens, kv_cfg, unchunked_prefill, null, warm);
    const mb = 1024 * 1024;
    // "Narrowest width tried" is only true where the ladder was walked.
    const width_note: []const u8 = if (perRequestPrefillChunkEnabled(config)) "the narrowest width tried" else "the pinned width";
    // The comparison first (`fitsAfterEviction` spelled out), then the residency as context, with
    // the part eviction cannot take named.
    log.warn("  prompt {d} tokens needs ~{d}MB at prefill chunk {d} ({s}), ~{d}MB available + ~{d}MB the hot cache can give back (~{d}MB resident, ~{d}MB of it pinned by this request's own prefix) — refused before prefill\n", .{
        prompt_len,
        bill.needed / mb,
        bill.chunk,
        width_note,
        bill.available / mb,
        bill.evictionCredit() / mb,
        bill.evictable / mb,
        pinnedResidentBytes(bill) / mb,
    });
    const context = requestContextRefusal(config, prompt_len, max_tokens, kv_cfg, bill, warm);
    if (context) |r| log.warn("  maximum context at KV{d}, prefill chunk {d}: {d} tokens\n", .{ r.kv_bits, r.chunk, r.maximum });
    return context;
}

extern "c" fn sysctlbyname(name: [*:0]const u8, oldp: ?*anyopaque, oldlenp: ?*usize, newp: ?*const anyopaque, newlen: usize) c_int;

/// Get the Metal max buffer allocation limit (~75% of system unified memory).
pub fn getMetalBufferLimit() u64 {
    var mem: u64 = 0;
    var len: usize = @sizeOf(u64);
    _ = sysctlbyname("hw.memsize", @ptrCast(&mem), &len, null, 0);
    if (mem == 0) return 8 * 1024 * 1024 * 1024; // fallback 8GB
    return mem * 75 / 100;
}

/// Pins the cache tiers a bill test is about. The defaults (no RAM tier, an SSD tier) bill differently:
/// SSD-first stages the writer's flush and a ringed arch bills its restore's coexistence.
pub const TierGuard = struct {
    ram: bool,
    disk: u64,

    pub fn pin(ram: bool, disk: u64) TierGuard {
        const saved: TierGuard = .{ .ram = launch_state.prefix_cache_ram_enabled, .disk = launch_state.prefix_cache_disk_bytes };
        launch_state.prefix_cache_ram_enabled = ram;
        launch_state.prefix_cache_disk_bytes = disk;
        return saved;
    }

    pub fn restore(self: TierGuard) void {
        launch_state.prefix_cache_ram_enabled = self.ram;
        launch_state.prefix_cache_disk_bytes = self.disk;
    }
};

/// The generic admission bill for `qwen4ExpOomConfig`, minus the live-memory read a unit test cannot make.
pub fn oomBillFor(cfg: *const model_mod.ModelConfig, seq: u64, chunk: u64, terms: PrefillRequestTerms) u64 {
    return prefillMemoryNeeded(seq, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.kvBytesPerToken(), cfg.head_dim, cfg.prefillScoreHeadDim(), cfg.hidden_size, prefillFfnWidth(cfg), 8, chunk, cfg.prefillAttnKeys(seq), prefillStreamBytesPerToken(cfg), prefillDequantWeightBytes(cfg), terms) +
        qsaMaskBytes(cfg, @min(chunk, @max(seq, 1)), seq);
}

/// The widest width the adaptive chooser may take for a prompt on this config.
fn adaptCapFor(cfg: *const model_mod.ModelConfig, seq: u64) u32 {
    return widthForRung(cfg, seq, PREFILL_CHUNK_LADDER[0]);
}

/// The live 768k scenario's checkout inputs, as `prefix_cache.checkoutEligible` takes them.
pub const CheckoutCase = struct {
    ssd_first: bool = true,
    move_enabled: bool = true,
    pending_disk: bool = false,
    entry_tokens: usize = 786_695,
    shared: usize = 786_695,
    prompt_len: usize = 786_707,
    has_slot: bool = true,

    /// The WarmPrefix the scheduler builds, with `will_donate` from restore-move's own predicate.
    pub fn warm(self: CheckoutCase, matched: u64, capacity: u64) WarmPrefix {
        return .{
            .matched_tokens = matched,
            .capacity_tokens = capacity,
            .will_donate = prefix_cache_mod.HotPrefixCache.checkoutEligible(
                self.ssd_first,
                self.move_enabled,
                self.pending_disk,
                self.entry_tokens,
                self.shared,
                self.prompt_len,
                self.has_slot,
            ),
        };
    }
};

pub const PinCfg = model_mod.ModelConfig;

/// A test that parses an arch's config needs that arch in the build (`-Darchs`); without it the test skips.
pub fn archBuilt(comptime name: []const u8) bool {
    return @hasField(arch_registry.Tag, name);
}

pub const PinGot = struct {
    buf: [2048]u64 = undefined,
    n: usize = 0,

    pub fn add(self: *PinGot, v: anytype) void {
        self.buf[self.n] = switch (@typeInfo(@TypeOf(v))) {
            .bool => @intFromBool(v),
            .@"enum" => @intFromEnum(v),
            .optional => if (v) |x| @as(u64, @intCast(x)) else 0,
            else => @intCast(v),
        };
        self.n += 1;
    }

    pub fn slice(self: *const PinGot) []const u64 {
        return self.buf[0..self.n];
    }
};

pub fn pinEq(comptime name: []const u8, want: []const u64, got: []const u64) !void {
    std.testing.expectEqualSlices(u64, want, got) catch |err| {
        std.debug.print("pin {s} moved\n", .{name});
        return err;
    };
}

const pin_archs = [_][]const u8{ "qwen4", "mimo2", "glm5", "dsv41" };

/// A pin lists one run of values per served pack, in `pin_archs` order, equal runs unless `lens` says otherwise. The runs
/// of an arch `-Darchs` leaves out are not checked; a build with none of them skips.
fn pinRows(comptime name: []const u8, want: []const u64, lens: ?[4]usize, cfgs: [4]*const model_mod.ModelConfig, comptime run: fn (*const model_mod.ModelConfig, *PinGot) void) !void {
    try testing.expect(lens != null or want.len % pin_archs.len == 0);
    var start: usize = 0;
    var checked: usize = 0;
    inline for (pin_archs, 0..) |arch, i| {
        const n = if (lens) |l| l[i] else want.len / pin_archs.len;
        if (archBuilt(arch)) {
            var got = PinGot{};
            run(cfgs[i], &got);
            try pinEq(name, want[start..][0..n], got.slice());
            checked += 1;
        }
        start += n;
    }
    try testing.expectEqual(want.len, start);
    if (checked == 0) return error.SkipZigTest;
}

pub fn pinConfigs(comptime name: []const u8, want: []const u64, comptime run: fn (*const PinCfg, *PinGot) void) !void {
    var fx = try fixtures.PinFx.init();
    defer fx.deinit();
    try pinRows(name, want, null, fx.all(), run);
}

/// `pinConfigs` for a bill whose run adds a different number of values per pack.
pub fn pinConfigsRagged(comptime name: []const u8, want: []const u64, lens: [4]usize, comptime run: fn (*const PinCfg, *PinGot) void) !void {
    var fx = try fixtures.PinFx.init();
    defer fx.deinit();
    try pinRows(name, want, lens, fx.all(), run);
}

pub const pin_bits = [_]u64{ 8, 16 };

pub const pin_gib: u64 = 1 << 30;

pub fn pinWarms() [4]WarmPrefix {
    return .{
        .{},
        .{ .mtp_on = true },
        .{ .matched_tokens = 50_000, .capacity_tokens = 60_000, .will_donate = true },
        .{ .checkpoints = 4, .released = true, .thought = true, .mtp_on = true },
    };
}

test "every enabled arch's bill is part of the build" {
    inline for (std.enums.values(arch_registry.Tag)) |tag| _ = arch_registry.Mod(tag).bill;
}

test "effectiveSsmCheckpointStride: zero capacity disables both tiers, RAM-off preserves disk" {
    try std.testing.expectEqual(@as(u32, 0), effectiveSsmCheckpointStride(256, 0, true, 4 << 30));
    try std.testing.expectEqual(@as(u32, 0), effectiveSsmCheckpointStride(256, 0, false, 4 << 30));
    try std.testing.expectEqual(@as(u32, 256), effectiveSsmCheckpointStride(256, 32, true, 0));
    try std.testing.expectEqual(@as(u32, 256), effectiveSsmCheckpointStride(256, 32, false, 4 << 30));
    try std.testing.expectEqual(@as(u32, 0), effectiveSsmCheckpointStride(256, 32, false, 0));
    try std.testing.expectEqual(@as(u32, 0), effectiveSsmCheckpointStride(0, 32, true, 0));
}

test "the bill plans the non-MLX GPU floor and the command-buffer size" {
    const t = std.testing;
    const mb: u64 = 1 << 20;
    // An idle reading: 28.96 GB in use, 28.28 active + 0.06 cache -> a 0.62 GB floor.
    try t.expectEqual(@as(u64, 620 * mb), gpuFloorFrom(28_960 * mb, 28_280 * mb, 60 * mb));
    try t.expectEqual(@as(u64, 0), gpuFloorFrom(0, 28_280 * mb, 0));
    try t.expectEqual(@as(u64, 0), gpuFloorFrom(100 * mb, 200 * mb, 0));
    // No reading keeps the operator's margin; a floor above margin - slack widens it.
    try t.expectEqual(launch_state.WIRED_LIMIT_MARGIN_BYTES, wiredUnplannedBytes(launch_state.WIRED_LIMIT_MARGIN_BYTES, 0));
    try t.expectEqual(launch_state.WIRED_LIMIT_MARGIN_BYTES, wiredUnplannedBytes(launch_state.WIRED_LIMIT_MARGIN_BYTES, 256 * mb));
    try t.expectEqual(620 * mb + GPU_FLOOR_SLACK_BYTES, wiredUnplannedBytes(launch_state.WIRED_LIMIT_MARGIN_BYTES, 620 * mb));
    // The default cadence bills nothing extra; 1000 MB bills about 1.07 GB.
    try t.expectEqual(@as(u64, 0), cadenceInflightBytes(BILL_FIT_MB_PER_BUFFER));
    try t.expectEqual(@as(u64, 0), cadenceInflightBytes(40));
    try t.expectEqual(@as(u64, 950 * mb * 9 / 8), cadenceInflightBytes(1000));
    // MLX's table and env reading.
    try t.expectEqual(@as(u64, 50), mlxDefaultMbPerBuffer("applegpu_g13s"));
    try t.expectEqual(@as(u64, 50), mlxDefaultMbPerBuffer("applegpu_g13d"));
    try t.expectEqual(@as(u64, 40), mlxDefaultMbPerBuffer("applegpu_g13g"));
    try t.expectEqual(@as(u64, 20), mlxDefaultOpsPerBuffer("applegpu_g14p"));
    try t.expectEqual(@as(u64, 1000), mlxPerBufferFromEnv("1000", 50));
    try t.expectEqual(@as(u64, 50), mlxPerBufferFromEnv("0", 50));
    try t.expectEqual(@as(u64, 50), mlxPerBufferFromEnv("lots", 50));
    try t.expectEqual(@as(u64, 50), mlxPerBufferFromEnv(null, 50));
}

test "Sushi quant memory production context CPU audit" {
    const raw = std.c.getenv("SUSHI_MEMORY_AUDIT_MODEL") orelse return error.SkipZigTest;
    const path = std.mem.span(raw);
    var cfg = try model_mod.parseConfig(std.testing.io, std.testing.allocator, path);
    defer cfg.deinit(std.testing.allocator);
    const saved_ctx = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = saved_ctx;
    const saved_width = settings_resolve.prefill_chunk.value;
    defer settings_resolve.prefill_chunk.value = saved_width;
    const saved_source = settings_resolve.prefill_chunk.source;
    defer settings_resolve.prefill_chunk.source = saved_source;
    launch_state.server_config.max_context_size = 500000;
    settings_resolve.prefill_chunk.value = 2048;
    settings_resolve.prefill_chunk.source = .flag;
    const budget: u64 = 16 << 30;
    const bill = loadServingBill(&cfg, budget).?;
    try std.testing.expectEqual(@as(u64, 8), bill.kv_bits);
    try std.testing.expect(sushiContextBytesAt(&cfg, bill.max_context, 8).bytes <= budget);
    if (bill.max_context < cfg.contextCap()) try std.testing.expect(sushiContextBytesAt(&cfg, bill.max_context + 1, 8).bytes > budget);
}

test "Sushi quant memory rejects an oversized context with an exact maximum at KV8" {
    if (!archBuilt("mimo2")) return error.SkipZigTest;
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const saved_ctx = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = saved_ctx;
    const saved_mtp = launch_state.configured_mtp;
    defer launch_state.configured_mtp = saved_mtp;
    const saved_kv = launch_state.configured_kv_quant;
    defer launch_state.configured_kv_quant = saved_kv;
    const saved_width = settings_resolve.prefill_chunk.value;
    defer settings_resolve.prefill_chunk.value = saved_width;
    const saved_source = settings_resolve.prefill_chunk.source;
    defer settings_resolve.prefill_chunk.source = saved_source;
    launch_state.server_config.max_context_size = 500000;
    launch_state.configured_kv_quant = null;
    launch_state.configured_mtp = false;
    settings_resolve.prefill_chunk.value = 2048;
    settings_resolve.prefill_chunk.source = .flag;
    var cfg = fixtures.qwen4RequestTestConfig();
    cfg.expert_layout = .exl3_k4;
    const budget = sushiContextBytesAt(&cfg, 128000, 8).bytes;
    const bill = loadServingBill(&cfg, budget).?;
    try std.testing.expectEqual(@as(u64, 8), bill.kv_bits);
    try std.testing.expectEqual(@as(u32, 500000), bill.requested_context);
    try std.testing.expect(bill.needed > budget);
    try std.testing.expectEqual(@as(u32, 128000), bill.max_context);
    try std.testing.expect(sushiContextBytesAt(&cfg, bill.max_context, 8).bytes <= budget);
    try std.testing.expect(sushiContextBytesAt(&cfg, bill.max_context + 1, 8).bytes > budget);
    try std.testing.expectEqual(@as(u64, 2048), sushiContextBytesAt(&cfg, 128000, 8).chunk);
    try std.testing.expect(maxSushiContextForBudget(&cfg, 4, budget) > bill.max_context);
    try std.testing.expectEqual(@as(u32, 0), maxSushiContextForBudget(&cfg, 8, 0));
    launch_state.configured_mtp = true;
    try std.testing.expect(loadServingBill(&cfg, budget).?.max_context < bill.max_context);
    var mimo = try model_mod.parseConfigFromJson(std.testing.allocator, arch_config.fixtureConfig("mimo2") orelse return error.SkipZigTest);
    defer mimo.deinit(std.testing.allocator);
    mimo.expert_layout = .exl3_k4;
    const mimo_budget = sushiContextBytesAt(&mimo, 64000, 8).bytes;
    const mimo_bill = loadServingBill(&mimo, mimo_budget).?;
    try std.testing.expectEqual(@as(u32, 64000), mimo_bill.max_context);
    launch_state.server_config.max_context_size = 0;
    cfg.ctx_override = 4096;
    cfg.kv_quant_override = trunk.KVQuantConfig.affine(4);
    const settings = loadServingBill(&cfg, budget).?;
    try std.testing.expectEqual(@as(u32, 4096), settings.requested_context);
    try std.testing.expectEqual(@as(u64, 4), settings.kv_bits);
    launch_state.server_config.max_context_size = 8192;
    launch_state.configured_kv_quant = trunk.KVQuantConfig.affine(8);
    try std.testing.expectEqual(@as(u32, 8192), loadServingBill(&cfg, budget).?.requested_context);
    try std.testing.expectEqual(@as(u64, 8), loadServingBill(&cfg, budget).?.kv_bits);
    cfg.expert_layout = .mxfp4_individual;
    try std.testing.expectEqual(@as(?model_load.LoadServingBill, null), loadServingBill(&cfg, budget));
    cfg.expert_layout = .exl3_k4;
    cfg.expert_streaming = true;
    try std.testing.expectEqual(@as(?model_load.LoadServingBill, null), loadServingBill(&cfg, budget));
}

test "diskBytesPerToken: the serialized geometry, whatever the QSA execution switches say" {
    if (!archBuilt("qwen4") or !archBuilt("mimo2") or !archBuilt("glm5")) return error.SkipZigTest;
    const t = std.testing;
    const saved_mtp = launch_state.configured_mtp;
    defer launch_state.configured_mtp = saved_mtp;
    const saved_fused = qwen4_qsa_select.qsa_score_fused_override;
    defer qwen4_qsa_select.qsa_score_fused_override = saved_fused;
    var q = fixtures.qwen4DeployedTestConfig();
    // kv8 rows plus the one pooled history; the head is off.
    launch_state.configured_mtp = false;
    // Plus its share of the recurrent state kept on every grid point.
    const points = (q.ssmCheckpointBytes() + kv_disk_cache.POINT_GRID - 1) / kv_disk_cache.POINT_GRID;
    try t.expect(points > 0);
    try t.expectEqual(@as(u64, 13_824) + points, diskBytesPerToken(&q, 8));
    const head = mtpHeadKvBytesPerToken(&q) + q.qsaHistoryBytesPerToken() / q.attnCacheLayerCount();
    try t.expect(head > 0);
    launch_state.configured_mtp = true;
    try t.expectEqual(@as(u64, 13_824) + points + head, diskBytesPerToken(&q, 8));
    try t.expect(diskBytesPerToken(&q, 16) > diskBytesPerToken(&q, 8));
    // The score bank is execution state, never in an entry.
    for ([_]?bool{ true, false }) |fused| {
        qwen4_qsa_select.qsa_score_fused_override = fused;
        for ([_]bool{ false, true }) |mtp| {
            launch_state.configured_mtp = mtp;
            try t.expectEqual(@as(u64, 13_824) + points + (if (mtp) head else 0), diskBytesPerToken(&q, 8));
        }
    }
    qwen4_qsa_select.qsa_score_fused_override = false;
    launch_state.configured_mtp = false;
    try t.expect(sessionBytesPerToken(&q, 8) > diskBytesPerToken(&q, 8) - points);

    // A ringed arch pays per token only on its global layers; its head is a fixed window.
    const mimo = fixtures.mimoV2FlashBillConfig();
    for ([_]bool{ false, true }) |mtp| {
        launch_state.configured_mtp = mtp;
        try t.expectEqual(kvBytesPerTokenAtBits(mimo.kvBytesPerToken(), 8), diskBytesPerToken(&mimo, 8));
        try t.expectEqual(kvBytesPerTokenAtBits(mimo.kvBytesPerToken(), 16), diskBytesPerToken(&mimo, 16));
    }

    // GLM: the latent rows (6,688 at kv8, 11,968 BF16) and the pooled index; DFlash2's window is fixed per entry.
    var glm = try model_mod.parseConfigFromJson(t.allocator, @embedFile("fixtures/glm5_config.json"));
    glm.dflash_loaded = false;
    const glm_points = (glm.ssmCheckpointBytes() + kv_disk_cache.POINT_GRID - 1) / kv_disk_cache.POINT_GRID;
    try t.expect(glm_points > 0);
    try t.expectEqual(@as(u64, 6_688) + glm_points, diskBytesPerToken(&glm, 8));
    try t.expectEqual(@as(u64, 5_280) + glm_points, diskBytesPerToken(&glm, 6));
    try t.expectEqual(@as(u64, 11_968) + glm_points, diskBytesPerToken(&glm, 16));
    glm.dflash_loaded = true;
    glm.dflash_window_bytes = 5 * 8 * 128 * 4 * 2304;
    try t.expectEqual(@as(u64, 6_688) + glm_points, diskBytesPerToken(&glm, 8));
}

test "admission's headroom is the ceiling less MLX's bytes, and never wraps" {
    const t = std.testing;
    const gib: u64 = 1 << 30;
    try t.expectEqual(@as(u64, 21 * gib), admissionAvailable(100 * gib, 79 * gib));
    try t.expectEqual(@as(u64, 0), admissionAvailable(100 * gib, 101 * gib));
}

test "defaultPrefixCacheAsk: an unnamed budget holds one session at the working context, never under 2 GB" {
    const t = std.testing;
    const gb: u64 = 1024 * 1024 * 1024;
    const session: u64 = 262144 * 24576; // 262k-token qwen4_exp session, ~6 GB
    try t.expectEqual(session, defaultPrefixCacheAsk(launch_state.PREFIX_CACHE_MEM_DEFAULT, false, session));
    // A short working context keeps the old floor.
    try t.expectEqual(launch_state.PREFIX_CACHE_MEM_DEFAULT, defaultPrefixCacheAsk(launch_state.PREFIX_CACHE_MEM_DEFAULT, false, gb));
    // An operator's number stands, even one equal to the default.
    try t.expectEqual(launch_state.PREFIX_CACHE_MEM_DEFAULT, defaultPrefixCacheAsk(launch_state.PREFIX_CACHE_MEM_DEFAULT, true, session));
    try t.expectEqual(32 * gb, defaultPrefixCacheAsk(32 * gb, true, session));
    try t.expectEqual(@as(u64, 0), defaultPrefixCacheAsk(0, true, session));
    // Any other ask is someone's choice and stands.
    try t.expectEqual(256 * 1024 * 1024, defaultPrefixCacheAsk(256 * 1024 * 1024, false, session));
}

test "defaultPrefixCacheAsk: the machine's headroom still caps the defaulted ask" {
    const t = std.testing;
    const gb: u64 = 1024 * 1024 * 1024;
    const session: u64 = 262144 * 24576;
    const ask = defaultPrefixCacheAsk(launch_state.PREFIX_CACHE_MEM_DEFAULT, false, session);
    // 16 GB ceiling, 11 GB weights, 1 GB live context, 1.5 GB prefill reserve: 2.5 GB left.
    try t.expectEqual(gb * 5 / 2, clampedPrefixCacheMem(ask, 16 * gb, 11 * gb, gb, gb * 3 / 2));
    // A big machine gets the whole session.
    try t.expectEqual(session, clampedPrefixCacheMem(ask, 200 * gb, 80 * gb, gb, 2 * gb));
}

test "oneSessionEntryBytes: a cached session is billed with its SSM checkpoints and the capacity it reserved" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const t = std.testing;
    const guard = fixtures.qsaScoreFusedOffGuard();
    defer guard.deinit();
    var cfg = fixtures.qwen4DeployedTestConfig();
    const ctx: u64 = 262_144;
    const chunk: u64 = 8192;
    cfg.pinned_context = @intCast(ctx);
    const entry = oneSessionEntryBytes(&cfg, 16, ctx, chunk);
    try t.expect(cfg.ssmCheckpointBytes() > 0);
    const fixed = cfg.qsaRingBytes() + retainedSsmCheckpointBytes(&cfg, ctx, 0, chunk);
    // A request at the working context reserves past it (`reservedCacheTokens`), and the entry
    // shares that capacity: no session the context admits outgrows the default.
    for ([_]u64{ ctx - 4000, ctx }) |seq| {
        const reserved = reservedCacheTokens(seq, 8192, PREFILL_CHUNK_LADDER[0], ctx);
        try t.expect(reserved > ctx);
        try t.expect(sessionBytesPerToken(&cfg, 16) * reserved + fixed <= entry);
    }
    try t.expect(defaultPrefixCacheAsk(launch_state.PREFIX_CACHE_MEM_DEFAULT, false, entry) >= entry);
}

test "oneSessionEntryBytes: a ringed session is billed with its ring and every checkpoint an entry keeps" {
    if (!archBuilt("mimo2")) return error.SkipZigTest;
    const t = std.testing;
    var cfg = fixtures.mimoV2FlashBillConfig();
    const ctx: u64 = 454_656;
    cfg.pinned_context = @intCast(ctx);
    const ring = cfg.swaRingBytes() + prefix_cache_mod.RING_CHECKPOINT_MAX * cfg.swaRingCheckpointBytes();
    try t.expect(ring > 0);
    const entry = oneSessionEntryBytes(&cfg, 8, ctx, 2048);
    try t.expectEqual(sessionBytesPerToken(&cfg, 8) * (ctx + PREFILL_CHUNK_LADDER[0]) + kvBytesPerTokenAtBits(ring, 8), entry);
    // Past the 2 GB default at kv8: the sessions that cold-prefilled every turn.
    try t.expect(entry > launch_state.PREFIX_CACHE_MEM_DEFAULT);
}

test "oneSessionFor: an unnamed budget grows only into room the n-gram table and a full-context prompt leave" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const t = std.testing;
    const guard = fixtures.qsaScoreFusedOffGuard();
    defer guard.deinit();
    const GiB: u64 = 1 << 30;
    var cfg = fixtures.qwen4DeployedTestConfig();
    const ctx: u64 = 1_048_576;
    cfg.pinned_context = @intCast(ctx);
    const chunk: u64 = 512;
    const weights: u64 = 51 * GiB;
    const table: u64 = 30 * GiB;

    const roomy = oneSessionFor(&cfg, 8, 400 * GiB, weights, table, ctx, chunk);
    try t.expectEqual(roomy.session, roomy.ask());

    // This box: the session fits beside the weights, not beside the table as well.
    const ceiling: u64 = 111_808 * (1 << 20);
    const bare = oneSessionFor(&cfg, 8, ceiling, weights, 0, ctx, chunk);
    try t.expectEqual(bare.session, bare.ask());
    const with_table = oneSessionFor(&cfg, 8, ceiling, weights, table, ctx, chunk);
    const ask = with_table.ask();
    try t.expect(ask > launch_state.PREFIX_CACHE_MEM_DEFAULT and ask < with_table.session);
    try t.expect(weights + table + ask + with_table.bill <= ceiling);

    // A room past half the prompt's bill stops at the half: the trim a session makes when it
    // outgrows the cache copies up to the budget beside the entry it replaces and the live KV.
    const wide = oneSessionFor(&cfg, 8, weights + roomy.bill + roomy.session - 1, weights, 0, ctx, chunk);
    try t.expect(wide.bill / 2 < wide.room);
    try t.expectEqual(wide.bill / 2, wide.ask());

    // No room: nothing grows and the 2 GB default stands.
    const none = oneSessionFor(&cfg, 8, weights + table, weights, table, ctx, chunk);
    try t.expectEqual(@as(u64, 0), none.ask());
    try t.expectEqual(launch_state.PREFIX_CACHE_MEM_DEFAULT, defaultPrefixCacheAsk(launch_state.PREFIX_CACHE_MEM_DEFAULT, false, none.ask()));
}

test "oneSessionFor: a full MiMo cache leaves a cold full-context prompt its bill at the pinned width" {
    if (!archBuilt("mimo2")) return error.SkipZigTest;
    const t = std.testing;
    attn_kernels.fused256_override = true;
    defer attn_kernels.fused256_override = null;
    const GiB: u64 = 1 << 30;
    var cfg = fixtures.mimoV2FlashBillConfig();
    const ctx: u64 = 454_656;
    cfg.pinned_context = @intCast(ctx);
    const chunk: u64 = 2048;
    const weights: u64 = 97 * GiB;
    const ceiling: u64 = weights + 14 * GiB;
    const saved = launch_state.server_config.default_force_mtp;
    defer launch_state.server_config.default_force_mtp = saved;
    launch_state.server_config.default_force_mtp = false;
    // Off, then the served default: the heads on for every request that leaves MTP out.
    for ([_]?bool{ false, null }) |mtp| {
        cfg.mtp_override = mtp;
        const ask = defaultPrefixCacheAsk(launch_state.PREFIX_CACHE_MEM_DEFAULT, false, oneSessionFor(&cfg, 8, ceiling, weights, 0, ctx, chunk).ask());
        try t.expect(ask > launch_state.PREFIX_CACHE_MEM_DEFAULT);
        // The ungated arch refuses rather than evicts to admit, so the cache must not take that bill.
        try t.expect(prefillNeededAtChunk(&cfg, ctx, 0, 8, chunk, .{ .mtp_on = mtp == null }) <= ceiling - weights - ask);
    }
    try t.expect(oneSessionFor(&cfg, 8, ceiling, weights, 0, ctx, chunk).bill > prefillNeededAtChunk(&cfg, ctx, 0, 8, chunk, .{}));
}

test "a qwen4_exp SSD restore is credited: its first grow bills one eval window of the restored rows" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    // The SSD tier fills buffers the slot owns at exactly the restored length (`LookupResult.slot_owned`),
    // so the first append grows every layer beside its old buffer and frees that buffer at the layer's
    // eval window. A share is billed the whole copy instead.
    const t = std.testing;
    const guard = fixtures.qsaScoreFusedOffGuard();
    defer guard.deinit();
    var cfg = fixtures.qwen4DeployedTestConfig();
    cfg.pinned_context = 1 << 20;
    const seq: u64 = 400_000;
    const restored: u64 = seq - 2000;
    const disk = WarmPrefix{ .matched_tokens = restored, .capacity_tokens = restored, .will_donate = true };
    const shared = WarmPrefix{ .matched_tokens = restored, .capacity_tokens = restored };
    const d = prefillRequestTerms(&cfg, seq, 2048, 8, 1024, disk);
    try t.expectEqual(restored * kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), 8), d.shared_resident_bytes);
    // Twelve caching layers, one inside each four-layer eval window.
    try t.expectEqual(d.shared_resident_bytes / 12, d.grow_coexist_bytes);
    try t.expectEqual(
        prefillNeededAtChunk(&cfg, seq, 2048, 8, 1024, shared) - (d.shared_resident_bytes - d.grow_coexist_bytes) * 5 / 4,
        prefillNeededAtChunk(&cfg, seq, 2048, 8, 1024, disk),
    );
}

test "clampedPrefixCacheMem: the budget never exceeds what the weights leave under the ceiling" {
    const t = std.testing;
    const GB: u64 = 1 << 30;
    // The crash's numbers: 96 GB ceiling, ~70 GB weights, ~6.4 GB of 262k-ctx
    // KV, ~4 GB prefill reserve, --prefix-cache-mem 40 GB → ~15.6 GB budget.
    const crash = clampedPrefixCacheMem(40 * GB, 96 * GB, 70 * GB, 6 * GB + (400 << 20), 4 * GB);
    try t.expect(crash >= 10 * GB and crash < 16 * GB);
    // Small model with plenty of headroom: the request is untouched.
    try t.expectEqual(8 * GB, clampedPrefixCacheMem(8 * GB, 96 * GB, 20 * GB, 2 * GB, 2 * GB));
    // requested == 0 (byte cap disabled today): headroom still bounds it.
    try t.expectEqual(96 * GB - 80 * GB, clampedPrefixCacheMem(0, 96 * GB, 70 * GB, 6 * GB, 4 * GB));
    // No headroom at all: never 0 — initWithMem reads 0 as "no byte cap".
    try t.expectEqual(@as(u64, 1), clampedPrefixCacheMem(40 * GB, 64 * GB, 70 * GB, 6 * GB, 4 * GB));
}

test "bf16 expert cache admission bills cache workspace bounce and resident trunk" {
    const gb: u64 = 1_000_000_000;
    const fill_peak: u64 = 2 * 5_033_164_800 + 127 * 9_830_400 + 536_870_912;
    const serving: u64 = 9 * gb;
    const needed = 10 * gb + 16 * gb + 60 * gb + 536_870_912 + fill_peak + serving;
    try std.testing.expect(expertStreamingLoadFits(needed, 10 * gb, 16 * gb, 60 * gb, 536_870_912, fill_peak, serving));
    try std.testing.expect(!expertStreamingLoadFits(needed - 1, 10 * gb, 16 * gb, 60 * gb, 536_870_912, fill_peak, serving));
}

test "DeepSeek: the streamed request bill brackets the prefill peaks measured on the M5 Max" {
    if (!archBuilt("dsv41")) return error.SkipZigTest;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.max_position_embeddings = 1_048_576;
    config.expert_fill_peak_bytes = model_load.billedFillPeakBytes(&config, 9_911 << 20);
    // No decoder pool to lend: the bill is the working set's, as measured.
    config.expert_borrow_bytes = 0;
    try std.testing.expectEqual(prefill_plan.TAIL_MERGE_MAX - 1, dsv41_config.tail_merge_rows);
    // Live per-chunk `peak_bytes - active_bytes` (`SUSHI_PREFILL_CHUNK_PEAK`), CED prefill, `--ssd-budget-gb 94`, the STQ d4
    // pack, DSpark on: `chunk` is the configured width, `seq` where the chunk ended (a last chunk absorbs its tail).
    const measured = [_]struct { seq: u64, chunk: u64, peak_mb: u64 }{
        .{ .seq = 7168, .chunk = 1024, .peak_mb = 3_560 },
        .{ .seq = 8675, .chunk = 1024, .peak_mb = 4_049 },
        .{ .seq = 18432, .chunk = 512, .peak_mb = 2_635 },
        .{ .seq = 34625, .chunk = 512, .peak_mb = 2_997 },
        .{ .seq = 108032, .chunk = 512, .peak_mb = 2_600 },
        .{ .seq = 108675, .chunk = 512, .peak_mb = 2_740 },
        // Transient peaks of MLX's own bytes, state included, binary 26a28040, budget 108.
        .{ .seq = 2048, .chunk = 2048, .peak_mb = 4_907 },
        .{ .seq = 225_000, .chunk = 2048, .peak_mb = 5_124 },
        // The two-layer prefill pipeline holds 0.25 GB more of a 2048-row chunk's transient.
        .{ .seq = 2048, .chunk = 2048, .peak_mb = 4_907 + 250 },
        .{ .seq = 225_000, .chunk = 2048, .peak_mb = 5_124 + 250 },
    };
    for (measured) |m| {
        const bill = prefillNeededAtChunk(&config, m.seq, 16, 16, m.chunk, .{});
        const peak = m.peak_mb * 1_000_000;
        // Under-billing is a Metal OOM: at least a tenth over the peak. The upper bar keeps the net small.
        try std.testing.expect(bill >= peak + peak / 10);
        try std.testing.expect(bill <= peak * 4);
    }
}

test "DeepSeek prices its chunk per request: 2048 while the bill fits, then 1024, then 512, never wider" {
    if (!archBuilt("dsv41")) return error.SkipZigTest;
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.max_position_embeddings = 1_048_576;
    config.num_attention_heads = 64;
    config.expert_fill_peak_bytes = model_load.billedFillPeakBytes(&config, 9_911 << 20);
    // No decoder pool to lend: the bill is the working set's, as measured.
    config.expert_borrow_bytes = 0;
    const MB: u64 = 1 << 20;
    // What `--ssd-budget-gb 96` on the 128 GB M5 Max leaves to serve, as its pin measured it.
    const at96: u64 = 110_100 * MB - 97_080 * MB;
    try t.expectEqual(@as(u32, 2048), chooseRequestPrefillChunk(&config, 8192, 2048, 16, at96, 512, 0, .{}));
    // Its attention scores only the picked keys, so a long prompt keeps the width while the bill fits.
    try t.expectEqual(@as(u32, 2048), chooseRequestPrefillChunk(&config, 400_000, 2048, 16, at96, 512, 0, .{}));
    const long: u64 = 524_288;
    const at1024 = prefillNeededAtChunk(&config, long, 2048, 16, 1024, .{});
    const only_narrow = prefillNeededAtChunk(&config, long, 2048, 16, 512, .{});
    try t.expect(prefillNeededAtChunk(&config, long, 2048, 16, 2048, .{}) > at1024 and at1024 > only_narrow);
    try t.expectEqual(@as(u32, 1024), chooseRequestPrefillChunk(&config, long, 2048, 16, at1024, 512, 0, .{}));
    try t.expectEqual(@as(u32, 512), chooseRequestPrefillChunk(&config, long, 2048, 16, only_narrow, 512, 0, .{}));
    try t.expectEqual(@as(u32, 512), chooseRequestPrefillChunk(&config, 8192, 2048, 16, at96, 512, 512, .{}));
    var buf: [256]u8 = undefined;
    try t.expectEqualStrings("Prefill chunk: per request, up to 2048 at a short prompt (the widest rung each request's bill admits); load-time fallback 512\n", prefillChunkLoadLine(&buf, &config, 512, 8192).?);
    // The ladder never narrows the advertised context: it is priced at the 512 floor.
    try t.expectEqual(@as(u32, 1_000_000), advertisedContextAt(&config, 16, 110_100 * MB, 97_080 * MB, 512));
}

test "DeepSeek advertises the longest context its own request bill admits" {
    if (!archBuilt("dsv41")) return error.SkipZigTest;
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.max_position_embeddings = 1_048_576;
    config.num_attention_heads = 64;
    config.expert_fill_peak_bytes = model_load.billedFillPeakBytes(&config, 9_911 << 20);
    // No decoder pool to lend: the bill is the working set's, as measured.
    config.expert_borrow_bytes = 0;
    const GB: u64 = 1_000_000_000;
    // The request bill grows with the prompt (the state and the rows' growth), so a serving budget of exactly one
    // prompt's bill advertises that prompt and not one token less.
    for ([_]u64{ 8192, 65_536, 131_072, 262_144, 524_288, 1_048_576 }) |seq| {
        for ([_]u64{ 512, 1024 }) |chunk| {
            const bill = prefillNeededAtChunk(&config, seq, 0, 16, chunk, .{});
            try t.expect(memoryContextAtChunk(&config, 16, 100 * GB + bill, 100 * GB, chunk) >= seq);
            try t.expect(memoryContextAtChunk(&config, 16, 100 * GB + bill - 1, 100 * GB, chunk) < seq);
        }
    }
    // At 96 the whole 1M context fits the fallback width's bill, with the memory the margined default needs to spare.
    const MB: u64 = 1 << 20;
    try t.expect(memoryContextAtChunk(&config, 16, 110_100 * MB, 97_080 * MB, 512) >= memoryContextForMargined(1_000_000));
}

test "a streamed DeepSeek bills its workspace at load and no fill peak at load or per request" {
    if (!archBuilt("dsv41")) return error.SkipZigTest;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.max_position_embeddings = 1_048_576;
    config.expert_cache_bytes = 93_993 << 20;
    config.expert_bounce_bytes = 512 << 20;
    config.expert_workspace_bytes = 3_525 << 20;
    config.expert_fill_peak_bytes = model_load.billedFillPeakBytes(&config, 9_911 << 20);
    const mib: u64 = 1 << 20;
    // No decoder pool to lend: the bill is the working set's, as measured.
    config.expert_borrow_bytes = 0;
    // A 100-token request pays its own transient, not the cache layer and a second workspace.
    try std.testing.expect(prefillNeededAtChunk(&config, 100, 512, 16, 512, .{}) < 2048 * mib);
    const resident: u64 = 8_295 * mib;
    const held = resident + config.expert_cache_bytes + config.expert_bounce_bytes + config.expert_workspace_bytes + config.expertPoolBytes();
    const serving = expertStreamingServingBytes(&config);
    try std.testing.expect(expertStreamingConfigFits(&config, held + serving, 0, resident));
    try std.testing.expect(!expertStreamingConfigFits(&config, held + serving - 1, 0, resident));
}

test "DeepSeek's live headroom and its load fit each count the decode pool the ledger parks once" {
    if (!archBuilt("dsv41")) return error.SkipZigTest;
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.max_position_embeddings = 1_048_576;
    // The parked pool is in MLX's cache, not its active bytes, so `ceiling - active` does not see it.
    try t.expectEqual(@as(u64, 10 << 30) - config.expertPoolBytes(), dsv41HeadroomOf(10 << 30, &config));
    try t.expectEqual(@as(u64, 0), dsv41HeadroomOf(1 << 20, &config));
    // The load fit holds the pool in its staging term, so the serving term it adds is the bill without it.
    config.expert_borrow_bytes = 60_000_000_000;
    const bill = prefillNeededAtChunk(&config, expertStreamingPlannedSeq(&config), 2048, defaultKvBits(&config), 512, .{});
    try t.expectEqual(bill - config.expertPoolBytes(), expertStreamingServingBytes(&config));
}

test "DeepSeek's request bill takes its prefill working set out of the decoder pools a prefill may release" {
    if (!archBuilt("dsv41")) return error.SkipZigTest;
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.max_position_embeddings = 1_048_576;
    const c = config.dsv41.?;
    const seq: u64 = 524_288;
    const state = ((seq + 2048) * c.stateBytesPerToken() + c.stateFixedBytes(true) + checkpointReserve(&config, seq, 2048)) * 5 / 4;
    const work = c.chunkCap(2048, seq) + config.expertPoolBytes();
    const decode = c.forwardCap(dsv41_borrow.verify_rows, seq);
    // No pool to lend: the whole working set is billed, as before.
    try t.expectEqual(state + work, prefillNeededAtChunk(&config, seq, 2048, 16, 2048, .{}));
    // Pools that cover part of it: admission pays only the rest.
    config.expert_borrow_bytes = work / 2;
    try t.expectEqual(state + work - work / 2, prefillNeededAtChunk(&config, seq, 2048, 16, 2048, .{}));
    // Pools that cover all of it: the state, a decode forward and the decode pool it leaves parked, at any chunk.
    config.expert_borrow_bytes = 60_000_000_000;
    for ([_]u64{ 512, 1024, 2048 }) |chunk| {
        const ck = ((seq + 2048) * c.stateBytesPerToken() + c.stateFixedBytes(true) + checkpointReserve(&config, seq, chunk)) * 5 / 4;
        try t.expectEqual(ck + decode + config.expertPoolBytes(), prefillNeededAtChunk(&config, seq, 2048, 16, chunk, .{}));
    }
    // The widest rung is then always affordable, and a chunk boundary never steps down for the working set.
    const MB: u64 = 1 << 20;
    try t.expectEqual(@as(u32, 2048), chooseRequestPrefillChunk(&config, seq, 2048, 16, state + decode + config.expertPoolBytes(), 512, 0, .{}));
    var st: mtp_plan.AdaptiveWidthState = .{};
    try t.expectEqual(@as(u32, 2048), adaptivePrefillWidth(&config, 16, seq, 64 * MB, 2048, 2048, &st));
    config.expert_borrow_bytes = 0;
    var st0: mtp_plan.AdaptiveWidthState = .{};
    try t.expect(adaptivePrefillWidth(&config, 16, seq, 64 * MB, 2048, 2048, &st0) < 2048);
}

test "a default context is served when the margined bill admits it, else the margined memory ceiling" {
    const t = std.testing;
    const dflt: u32 = 1_000_000;
    // The margined memory context reaches the default: it is served whole.
    const reach = memoryContextForMargined(dflt);
    try t.expect(safeAutoContext(reach) >= dflt);
    try t.expectEqual(dflt, autoContextFrom(reach, 1_048_576, dflt));
    try t.expectEqual(dflt, autoContextFrom(4_000_000, 1_048_576, dflt));
    // Just below it, and at the checkpoint's window, the margined memory value stands, never the default.
    const below = reach - 6 * 1024;
    try t.expect(safeAutoContext(below) < dflt);
    try t.expectEqual(safeAutoContext(below), autoContextFrom(below, 1_048_576, dflt));
    try t.expectEqual(@as(u32, 974_848), autoContextFrom(1_048_576, 1_048_576, dflt));
    try t.expectEqual(safeAutoContext(390_144), autoContextFrom(390_144, 1_048_576, dflt));
    // Another architecture has no default: the margin applies as before.
    try t.expectEqual(@as(u32, 974_848), autoContextFrom(1_048_576, 1_048_576, 0));
    // The same relation on the load-time bill, so the bill covers what is advertised.
    const MiB: u64 = 1 << 20;
    const roomy = resolvedContextForLoad(0, 0, 4_000_000 * MiB, 0, 0, 0, 890, 1_048_576, dflt);
    try t.expectEqual(dflt, roomy);
    try t.expectEqual(@as(u32, 1_048_576), resolvedContextForLoad(0, 0, 4_000_000 * MiB, 0, 0, 0, 890, 1_048_576, 0));
    // An explicit context, from the flag or the pin, outranks it.
    try t.expectEqual(@as(u32, 250_000), resolvedContextForLoad(250_000, 0, 4_000_000 * MiB, 0, 0, 0, 890, 1_048_576, dflt));
    try t.expectEqual(@as(u32, 300_000), resolvedContextForLoad(0, 300_000, 4_000_000 * MiB, 0, 0, 0, 890, 1_048_576, dflt));
}

test "the default request bill at 1M tokens is the borrowed bill plus the decode pool" {
    if (!archBuilt("dsv41")) return error.SkipZigTest;
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.max_position_embeddings = 1_048_576;
    config.num_attention_heads = 64;
    const saved = .{ launch_state.prefix_cache_ram_enabled, launch_state.prefix_cache_disk_bytes };
    defer {
        launch_state.prefix_cache_ram_enabled = saved[0];
        launch_state.prefix_cache_disk_bytes = saved[1];
    }
    launch_state.prefix_cache_ram_enabled = false;
    launch_state.prefix_cache_disk_bytes = 20 << 30;
    const bill = defaultRequestBill(&config);
    // dsv41-memory: 3.4 GB at 1M with the prefill borrowed (state, writer in flight, rings, x5/4, the decode cap), plus the
    // 1 GiB decode pool the ledger parks beside it.
    try t.expectApproxEqAbs(@as(f64, 4_400_000_000), @as(f64, @floatFromInt(bill)), 300_000_000);
    // The bill leaves the config's own streaming fields as they were.
    try t.expect(config.expert_streaming);
    try t.expectEqual(@as(u64, 0), config.expert_borrow_bytes);
}

test "with its decoder pools lendable, DeepSeek at --ssd-budget-gb 117 with DSpark on wired 124000 advertises its whole context" {
    if (!archBuilt("dsv41")) return error.SkipZigTest;
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.max_position_embeddings = 1_048_576;
    config.num_attention_heads = 64;
    const saved = .{ launch_state.prefix_cache_ram_enabled, launch_state.prefix_cache_disk_bytes };
    defer {
        launch_state.prefix_cache_ram_enabled = saved[0];
        launch_state.prefix_cache_disk_bytes = saved[1];
    }
    // The served default: no RAM tier, the SSD tier on, so restore points go to SSD as a prefill takes them.
    launch_state.prefix_cache_ram_enabled = false;
    launch_state.prefix_cache_disk_bytes = 20 << 30;
    const MB: u64 = 1 << 20;
    // The budget-100 boot on wired 124000 read a 122,170 MiB ceiling and 100,328 MiB active; 17 GiB more at 117.
    const ceiling: u64 = 122_170 * MB;
    // The budget now holds the decode pool, parked in MLX's cache: the same budget leaves active that much lower.
    const active: u64 = (100_328 + 17 * 1024) * MB - (1 << 30);
    // Its expert cache, 13.57 GB under the budget (94% of the experts), split evenly: the decoder half is lendable.
    config.expert_borrow_bytes = (117 * (1 << 30) - 13_570_000_000) / 2;
    try t.expectEqual(@as(u32, 1_000_000), advertisedContextAt(&config, 16, ceiling, active, 2048));
    // Billing the working set to the request, with nothing to lend, advertises well under the whole context.
    config.expert_borrow_bytes = 0;
    try t.expectEqual(@as(u32, 390_144), advertisedContextAt(&config, 16, ceiling, active, 512));
}

test "bf16 expert load admission includes 65k KV and QSA serving terms" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const fused_guard = fixtures.qsaScoreFusedOffGuard();
    defer fused_guard.deinit();
    var config = fixtures.qwen4RequestTestConfig();
    config.expert_streaming = true;
    config.expert_fill_peak_bytes = 11_852_000_000;
    const saved_ctx = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = saved_ctx;
    launch_state.server_config.max_context_size = 65_536;
    const saved_kv = launch_state.configured_kv_quant;
    defer launch_state.configured_kv_quant = saved_kv;
    launch_state.configured_kv_quant = trunk.KVQuantConfig.affine(8);
    const serving = expertStreamingServingBytes(&config);
    try std.testing.expect(serving >= kvBytesPerTokenAtBits(config.kvBytesPerToken(), 8) * 65_536);
    const fixed: u64 = 10_000_000_000 + 16_000_000_000 + 60_000_000_000 + 536_870_912 + config.expert_fill_peak_bytes;
    try std.testing.expect(expertStreamingLoadFits(fixed + serving, 10_000_000_000, 16_000_000_000, 60_000_000_000, 536_870_912, config.expert_fill_peak_bytes, serving));
    try std.testing.expect(!expertStreamingLoadFits(fixed + serving - 1, 10_000_000_000, 16_000_000_000, 60_000_000_000, 536_870_912, config.expert_fill_peak_bytes, serving));
}

test "a streamed load bills the narrowest rung the per-request ladder can take" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const fused_guard = fixtures.qsaScoreFusedOffGuard();
    defer fused_guard.deinit();
    var config = fixtures.qwen4RequestTestConfig();
    config.expert_streaming = true;
    // The fit check runs before the post-load sizer pins a chunk.
    config.pinned_prefill_chunk = 0;
    const saved_ctx = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = saved_ctx;
    launch_state.server_config.max_context_size = 32_768;
    const saved_kv = launch_state.configured_kv_quant;
    defer launch_state.configured_kv_quant = saved_kv;
    launch_state.configured_kv_quant = trunk.KVQuantConfig.affine(8);
    const saved_source = settings_resolve.prefill_chunk.source;
    const saved_chunk = settings_resolve.prefill_chunk.value;
    defer {
        settings_resolve.prefill_chunk.source = saved_source;
        settings_resolve.prefill_chunk.value = saved_chunk;
    }
    const saved_per_request = per_request_chunk_override;
    defer per_request_chunk_override = saved_per_request;
    settings_resolve.prefill_chunk.source = .default;
    per_request_chunk_override = true;
    const floor: u64 = PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1];
    const at = struct {
        fn bill(c: *const model_mod.ModelConfig, chunk: u64) u64 {
            return prefillNeededAtChunk(c, @max(32_768, chunk), 2048, 8, chunk, .{}) -| c.expert_fill_peak_bytes;
        }
    }.bill;
    try std.testing.expectEqual(at(&config, floor), expertStreamingServingBytes(&config));
    try std.testing.expect(at(&config, floor) < at(&config, 8192));
    // An explicit width caps the ladder, so the load proves the floor, or a narrower flag.
    settings_resolve.prefill_chunk.source = .flag;
    settings_resolve.prefill_chunk.value = 4096;
    try std.testing.expectEqual(at(&config, floor), expertStreamingServingBytes(&config));
    settings_resolve.prefill_chunk.value = 256;
    try std.testing.expectEqual(at(&config, 256), expertStreamingServingBytes(&config));
    // With the ladder off the flag is a pin again.
    settings_resolve.prefill_chunk.value = 4096;
    per_request_chunk_override = false;
    try std.testing.expectEqual(at(&config, 4096), expertStreamingServingBytes(&config));
    per_request_chunk_override = true;
    settings_resolve.prefill_chunk.source = .default;
    config.pinned_prefill_chunk = 1024;
    try std.testing.expectEqual(at(&config, 1024), expertStreamingServingBytes(&config));
    config.pinned_prefill_chunk = 0;
    per_request_chunk_override = false;
    try std.testing.expectEqual(at(&config, floor), expertStreamingServingBytes(&config));
}

test "a loaded streamed row reports its effective budget, an unloaded one the setting" {
    try std.testing.expectEqual(@as(u32, 60), rowSsdBudgetGb(60 << 30, 0));
    try std.testing.expectEqual(@as(u32, 48), rowSsdBudgetGb(0, 48));
    try std.testing.expectEqual(@as(u32, 60), rowSsdBudgetGb(60 << 30, 48));
    try std.testing.expectEqual(@as(u32, 0), rowSsdBudgetGb(0, 0));
}

test "raising the GPU limit always raises both user-wire limits to match, never below the need" {
    var buf: [192]u8 = undefined;
    // iogpu.wired_limit_mb counts MiB: the user-wire limits get the same amount in bytes.
    try std.testing.expectEqualStrings("sudo sysctl iogpu.wired_limit_mb=125000 vm.global_user_wire_limit=131072000000 vm.user_wire_limit=131072000000", wireLimitsCommand(&buf, 125000));
    try std.testing.expectEqual(@as(u64, 86100), wireLimitMbFor(84 << 30));
    try std.testing.expectEqual(@as(u64, 125000), wireLimitMbFor(125000 << 20));
}

test "the user-wire warning fires when the wired bytes pass the lower macOS limit" {
    const t = std.testing;
    const GiB: u64 = 1 << 30;
    // M5 Max 128 GB: `vm.global_user_wire_limit` defaults to memsize less 20.62 GB.
    const ram: u64 = 137_438_953_472;
    const default_limit: u64 = 116_820_000_000;
    // A streamed DeepSeek load's owned bytes are its ssd budget less the 1 GiB MLX pool.
    const owned = struct {
        fn at(budget_gib: u64) u64 {
            return (budget_gib - 1) * (1 << 30);
        }
    }.at;
    var buf: [384]u8 = undefined;
    try t.expect(userWireWarning(&buf, owned(108), .{ default_limit, default_limit }, 125_000 << 20, ram) == null);
    for ([_]u64{ 111, 114 }) |b| {
        const line = userWireWarning(&buf, owned(b), .{ default_limit, default_limit }, 125_000 << 20, ram) orelse return error.TestExpectedWarning;
        try t.expect(std.mem.indexOf(u8, line, "vm.global_user_wire_limit is 116.8 GB") != null);
        // The declared GPU limit covers the load: the user-wire limits match it.
        try t.expect(std.mem.indexOf(u8, line, "sudo sysctl iogpu.wired_limit_mb=125000 vm.global_user_wire_limit=131072000000 vm.user_wire_limit=131072000000") != null);
    }
    const line = userWireWarning(&buf, owned(111), .{ default_limit, default_limit }, 0, ram).?;
    try t.expect(std.mem.indexOf(u8, line, "wires ~119.2 GB") != null);
    try t.expect(std.mem.indexOf(u8, line, "iogpu.wired_limit_mb=113700 vm.global_user_wire_limit=119223091200 ") != null);
    try t.expect(userWireWarning(&buf, owned(114), .{ 125_000_000_000, 125_000_000_000 }, 125_000 << 20, ram) == null);
    // The per-process limit binds when it is the lower one; unreadable limits never warn.
    try t.expect(std.mem.indexOf(u8, userWireWarning(&buf, owned(111), .{ 125_000_000_000, default_limit }, 0, ram).?, "vm.user_wire_limit is") != null);
    try t.expect(userWireWarning(&buf, owned(114), .{ 0, 0 }, 0, ram) == null);
    // A 96 GB Mac: the default limit is 15% under memsize; the suggestion stays 12 GB under RAM.
    const ram96: u64 = 103_079_215_104;
    const limit96: u64 = 87_600_000_000;
    try t.expect(userWireWarning(&buf, 80 * GiB, .{ limit96, limit96 }, 0, ram96) == null);
    const l96 = userWireWarning(&buf, 83 * GiB, .{ limit96, limit96 }, 0, ram96).?;
    try t.expect(std.mem.indexOf(u8, l96, "iogpu.wired_limit_mb=86100 vm.global_user_wire_limit=90282393600 ") != null);
    // Past what the Mac may wire, raising the limit cannot help: the line says so.
    const none = userWireWarning(&buf, 95 * GiB, .{ limit96, limit96 }, 0, ram96).?;
    try t.expect(std.mem.indexOf(u8, none, "sudo") == null);
    try t.expect(std.mem.indexOf(u8, none, "--ssd-budget-gb") != null);
}

test "ssd budget parses GiB and admits against the wired limit" {
    const t = std.testing;
    const GiB: u64 = 1 << 30;
    try t.expectEqual(100 * GiB, try parseSsdBudgetGb("100"));
    try t.expectError(error.InvalidSsdBudgetGb, parseSsdBudgetGb("0"));
    try t.expectError(error.InvalidSsdBudgetGb, parseSsdBudgetGb("513"));
    try t.expectError(error.InvalidSsdBudgetGb, parseSsdBudgetGb("100.5"));
    // The sysctl is the limit when it is set; otherwise Metal's recommended working set.
    try t.expectEqual(120 * GiB, ssdBudgetCeiling(120 * GiB, 96 * GiB));
    try t.expectEqual(96 * GiB, ssdBudgetCeiling(0, 96 * GiB));
    try t.expect(ssdBudgetAdmits(80 * GiB, 8 * GiB, 96 * GiB));
    try t.expect(!ssdBudgetAdmits(100 * GiB, 8 * GiB, 96 * GiB));
    try t.expectEqual(@as(u64, 110_592), wiredLimitMbForBudget(100 * GiB, 8 * GiB));
    try t.expectEqual(@as(u64, 1), wiredLimitMbForBudget(1, 0));
}

test "planHotCache: the budget is MONOTONE in the ask" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    // The defect: a bigger ask stepped the chunk down, shrank the reserve and bought MORE cache
    // (10GB -> 3873 MB, 24GB -> 5703 MB, same box).
    const t = std.testing;
    const GiB: u64 = 1 << 30;
    const cfg = fixtures.longCtxTestConfig();
    const kv_bits: u64 = 8;
    const ctx_tokens: u64 = 262144;
    const weights: u64 = 69 * GiB;
    const ceiling: u64 = weights + 28_909 * (1 << 20);

    var prev: u64 = 0;
    var ask: u64 = GiB;
    while (ask <= 64 * GiB) : (ask += GiB) {
        const plan = planHotCache(&cfg, kv_bits, ceiling, weights, ctx_tokens, 0, ask, 0);
        try t.expect(plan.budget >= prev);
        try t.expect(plan.budget <= @max((ceiling -| weights) -| plan.ctx_kv -| plan.reserve, 1));
        prev = plan.budget;
    }

    // The two live asks now agree: the sizer no longer sees the ask.
    const at10 = planHotCache(&cfg, kv_bits, ceiling, weights, ctx_tokens, 0, 10 * GiB, 0);
    const at24 = planHotCache(&cfg, kv_bits, ceiling, weights, ctx_tokens, 0, 24 * GiB, 0);
    try t.expect(at24.budget >= at10.budget);
    try t.expectEqual(at10.chunk, at24.chunk);
}

test "planHotCache: an out-of-the-box boot gets the whole remaining headroom, never the 1-byte floor" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    // One boot with no flags: `requested == 0` used to hand the sizer a zero reserve, so it took
    // a wide rung and the clamp returned its 1-byte floor.
    const t = std.testing;
    const GiB: u64 = 1 << 30;
    const cfg = fixtures.longCtxTestConfig();
    const kv_bits: u64 = 8;
    const weights: u64 = 69 * GiB;
    const ceiling: u64 = weights + 28_909 * (1 << 20);
    const ctx_tokens: u64 = 262144;
    const per_tok: u64 = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits) + statePerTokenBilled(&cfg);
    const sizer_ctx_kv: u64 = per_tok * ctx_tokens + cfg.qsaRingBytes();

    const boot = planHotCache(&cfg, kv_bits, ceiling, weights, ctx_tokens, sizer_ctx_kv, 0, 0);
    // Everything left over: the cache is the residual claimant.
    try t.expectEqual(@max((ceiling -| weights) -| boot.ctx_kv -| boot.reserve, 1), boot.budget);
    // A cache worth having, or the ladder bottomed out and the box is too small.
    try t.expect(boot.budget >= HOT_CACHE_FLOOR_BYTES or
        boot.chunk == PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1]);

    // The rung is the widest that clears both ask-independent bars.
    const serving: u64 = ceiling - weights;
    const cap: u64 = @min(serving / PREFILL_RESERVE_BUDGET_SHARE, (serving -| sizer_ctx_kv) -| HOT_CACHE_FLOOR_BYTES);
    for (PREFILL_CHUNK_LADDER) |c| {
        if (c <= boot.chunk) break;
        try t.expect(prefillTransientReserve(&cfg, kv_bits, c) > cap);
    }

    // An explicit ask can only take a slice of the same headroom.
    const huge = planHotCache(&cfg, kv_bits, ceiling, weights, ctx_tokens, sizer_ctx_kv, 1024 * GiB, 0);
    try t.expectEqual(boot.budget, huge.budget);
    try t.expect(planHotCache(&cfg, kv_bits, ceiling, weights, ctx_tokens, sizer_ctx_kv, 4 * GiB, 0).budget <= boot.budget);
}

test "the clamp's live inversion, as arithmetic" {
    // The reported boot reduced to `clampedPrefixCacheMem`: ceiling - weights = 28,909 MiB, ctx KV 20,736 MiB.
    const t = std.testing;
    const MiB: u64 = 1 << 20;
    const GiB: u64 = 1 << 30;
    const serving: u64 = 28_909 * MiB;
    const ctx_kv: u64 = 20_736 * MiB;
    const reserve_1024: u64 = 4_300 * MiB;
    const reserve_512: u64 = 2_470 * MiB;

    const ask10 = clampedPrefixCacheMem(10 * GiB, serving, 0, ctx_kv, reserve_1024);
    const ask24 = clampedPrefixCacheMem(24 * GiB, serving, 0, ctx_kv, reserve_512);
    try t.expectEqual(3_873 * MiB, ask10);
    try t.expectEqual(5_703 * MiB, ask24);
    try t.expect(ask10 < ask24); // the inversion, in the numbers that shipped

    // One ask-independent reserve: the clamp is `@min(ask, headroom)` over a headroom no ask can move.
    var prev: u64 = 0;
    var ask: u64 = GiB;
    while (ask <= 64 * GiB) : (ask += GiB) {
        const got = clampedPrefixCacheMem(ask, serving, 0, ctx_kv, reserve_1024);
        try t.expect(got >= prev);
        try t.expect(got <= @max((serving -| ctx_kv) -| reserve_1024, 1));
        prev = got;
    }
    try t.expectEqual(3_873 * MiB, clampedPrefixCacheMem(0, serving, 0, ctx_kv, reserve_1024));
}

test "an explicit --prefill-chunk is the chunk that gets BILLED" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    // `--prefill-chunk 4096` used to run a 4096-token forward against a 512-token reserve.
    const t = std.testing;
    const GiB: u64 = 1 << 30;
    var cfg = fixtures.longCtxTestConfig();
    cfg.head_dim = 128; // fused dim: boundedPrefillChunk returns its base untouched
    const kv_bits: u64 = 8;
    const weights: u64 = 69 * GiB;
    // Squeezed so the machine sizer would step below the flag.
    const ceiling: u64 = weights + 6 * GiB;

    settings_resolve.prefill_chunk.value = 4096;
    settings_resolve.prefill_chunk.source = .flag;
    defer {
        settings_resolve.prefill_chunk.value = 8192;
        settings_resolve.prefill_chunk.source = .default;
    }
    try t.expectEqual(@as(u32, 4096), explicitPrefillChunk());
    const sized = resolvePrefillChunk(&cfg, kv_bits, ceiling, weights, 0, 0);
    try t.expect(sized < 4096); // the machine would have narrowed it

    const billed = billedPrefillChunk(&cfg, kv_bits, ceiling, weights, 0, 0, explicitPrefillChunk());
    try t.expectEqual(@as(u32, 4096), billed);

    // The other reader agrees.
    try t.expectEqual(@as(usize, 4096), prefill_plan.effectivePrefillChunk(
        cfg.prefillScoreHeadDim(),
        cfg.num_attention_heads,
        40_000,
        cfg.has_sliding_window,
        cfg.isMoe(),
        cfg.longCtxGated(),
        billed,
    ));

    // With the per-request ladder off the flag is a pin, and the clamp reserves it; on the ladder it
    // only caps a width every request re-bills (`clampReserveWidth`).
    per_request_chunk_override = false;
    defer per_request_chunk_override = null;
    const plan = planHotCache(&cfg, kv_bits, ceiling, weights, 8192, 0, 2 * GiB, billed);
    try t.expectEqual(@as(u32, 4096), plan.chunk);
    try t.expectEqual(prefillTransientReserve(&cfg, kv_bits, 4096), plan.reserve);
    try t.expect(plan.reserve > prefillTransientReserve(&cfg, kv_bits, sized));
}

test "qwen4RequestTestConfig does not leak qsa_score_fused_override" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const saved = qwen4_qsa_select.qsa_score_fused_override;
    defer qwen4_qsa_select.qsa_score_fused_override = saved;
    qwen4_qsa_select.qsa_score_fused_override = null;
    _ = fixtures.qwen4RequestTestConfig();
    try std.testing.expectEqual(@as(?bool, null), qwen4_qsa_select.qsa_score_fused_override);
}

test "Flash-Next's request bill brackets the peaks measured on the M1 Max, with no padding beyond a small net" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const saved_ctx = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = saved_ctx;
    launch_state.server_config.max_context_size = 100000;
    const saved_width = settings_resolve.prefill_chunk.value;
    defer settings_resolve.prefill_chunk.value = saved_width;
    settings_resolve.prefill_chunk.value = 512;
    const cfg = fixtures.qwen4MeasuredTestConfig();
    // Live `peak_bytes - active_bytes` after one prompt per boot (512 rung, kv8, no MTP, SSD tier on, 16 new
    // tokens), with each landed checkpoint already moved to the SSD.
    const measured = [_]struct { tokens: u64, peak_mib: u64 }{
        .{ .tokens = 2857, .peak_mib = 935 },
        .{ .tokens = 8505, .peak_mib = 984 },
        .{ .tokens = 16874, .peak_mib = 1196 },
        .{ .tokens = 33958, .peak_mib = 1381 },
        .{ .tokens = 59314, .peak_mib = 1976 },
    };
    for (measured) |m| {
        const bill = prefillNeededAtChunk(&cfg, m.tokens, 16, 8, 512, .{});
        const peak = m.peak_mib * (1 << 20);
        // Under-billing is a Metal OOM: at least a tenth over the peak. The upper bar keeps the net small.
        try std.testing.expect(bill >= peak + peak / 10);
        try std.testing.expect(bill <= peak + peak * 2 / 5);
    }
}

test "Flash-Next's streamed request bill brackets the streamed peaks measured on the M1 Max" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const saved_ctx = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = saved_ctx;
    launch_state.server_config.max_context_size = 49152;
    var cfg = fixtures.qwen4MeasuredTestConfig();
    cfg.expert_streaming = true;
    cfg.expert_fill_peak_bytes = model_load.billedFillPeakBytes(&cfg, 1_848_000_000);
    // Live `peak_bytes - active_bytes` after one cold prompt per width, `--ssd-budget-gb 23`, kv8, no MTP, SSD tier on.
    const measured = [_]struct { tokens: u64, chunk: u64, peak_mib: u64 }{
        .{ .tokens = 7987, .chunk = 1024, .peak_mib = 1224 },
        .{ .tokens = 45106, .chunk = 512, .peak_mib = 1683 },
    };
    for (measured) |m| {
        const bill = prefillNeededAtChunk(&cfg, m.tokens, 16, 8, m.chunk, .{});
        const peak = m.peak_mib << 20;
        try std.testing.expect(bill >= peak + peak / 10);
        try std.testing.expect(bill <= peak + peak * 2 / 5);
    }
}

test "the advertised Flash-Next context is the largest one the request bill admits" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const saved_ctx = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = saved_ctx;
    launch_state.server_config.max_context_size = 0;
    const saved_disk = launch_state.prefix_cache_disk_bytes;
    defer launch_state.prefix_cache_disk_bytes = saved_disk;
    launch_state.prefix_cache_disk_bytes = 20 << 30;
    const saved_mtp = launch_state.configured_mtp;
    defer launch_state.configured_mtp = saved_mtp;
    launch_state.configured_mtp = false;
    var cfg = fixtures.qwen4MeasuredTestConfig();
    cfg.pinned_prefill_chunk = 512;
    const MiB: u64 = 1 << 20;
    const ceiling: u64 = 28976 * MiB; // 30000 MiB wired limit less the 1 GiB margin
    const active: u64 = 27_378_205_274; // the measured lean pack's resident bytes
    // The SSD writer's host bytes sit outside the wired limit: the 1 GiB margin under it is the only reserve.
    const budget = ceiling - active;
    const raw = memoryContextAt(&cfg, ceiling, active);
    const fits = struct {
        fn at(c: *const model_mod.ModelConfig, ctx: u64, avail: u64) bool {
            var sized = c.*;
            sized.pinned_context = @intCast(ctx);
            return prefillNeededAtChunk(&sized, ctx, 0, 8, 512, .{}) <= avail;
        }
    };
    // What admission accepts is what the sizer advertises, to within the search step.
    try std.testing.expect(raw > 0);
    try std.testing.expect(fits.at(&cfg, raw, budget));
    try std.testing.expect(!fits.at(&cfg, @as(u64, raw) + 1024, budget));
}

test "the load-time budget is reproducible: free RAM at load does not move it" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // Two boots 11 minutes apart resolved the same 10 GB ask to 1076 and 9757 MB off the live ceiling.
    const t = std.testing;
    const cfg = fixtures.qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const GB: u64 = 1000 * 1000 * 1000;
    const weights: u64 = 70 * GB;
    const ctx_tokens: u64 = 1_048_576; // the deployed --ctx-size
    const static_ceiling: u64 = 120_000 * 1024 * 1024; // the wired limit, both boots
    // `requested == 0` is the arm where the clamp alone decides: the budget IS the headroom.
    const uncapped: u64 = 0;

    const busy = physicalMemoryCeiling(static_ceiling, weights, 108_700 * 1000 * 1000 -| weights);
    const quiet = physicalMemoryCeiling(static_ceiling, weights, 116_600 * 1000 * 1000 -| weights);
    try t.expect(busy < quiet);

    const on_busy = planHotCache(&cfg, kv_bits, busy, weights, ctx_tokens, 0, uncapped, 0);
    const on_quiet = planHotCache(&cfg, kv_bits, quiet, weights, ctx_tokens, 0, uncapped, 0);
    try t.expect(on_busy.budget != on_quiet.budget);
    try t.expect(on_quiet.budget > on_busy.budget);

    // Billed against the static term, both boots agree.
    const a = planHotCache(&cfg, kv_bits, static_ceiling, weights, ctx_tokens, 0, uncapped, 0);
    const b = planHotCache(&cfg, kv_bits, static_ceiling, weights, ctx_tokens, 0, uncapped, 0);
    try t.expectEqual(a.budget, b.budget);
    try t.expect(a.budget >= on_quiet.budget);

    const capped_a = planHotCache(&cfg, kv_bits, static_ceiling, weights, ctx_tokens, 0, 24 * GB, 0);
    const capped_b = planHotCache(&cfg, kv_bits, static_ceiling, weights, ctx_tokens, 0, 24 * GB, 0);
    try t.expectEqual(capped_a.budget, capped_b.budget);
    try t.expect(capped_a.budget <= a.budget);

    try t.expectEqual(static_ceiling, physicalMemoryCeiling(static_ceiling, weights, 400 * GB));
}

test "an auto boot sizes the SAME context whatever the cache ask (live check #6)" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // `--prefix-cache-mem 60GB` collapsed the advertised context to 870 tokens (ctx KV billed at 26 MB).
    const t = std.testing;
    const cfg = fixtures.qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const active: u64 = 69_827 * MiB; // backed out of the live clamp line
    const live_ceiling: u64 = 109_395 * MiB;
    const per_tok: u64 = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits) +| statePerTokenBilled(&cfg);
    const transient: u64 = prefillTransientReserve(&cfg, kv_bits, 4096);
    const cap: u32 = cfg.contextCap();

    // The sizing reserve is a constant, so the answer cannot move with the ask.
    const ctx_default = resolvedContextForLoad(0, 0, live_ceiling, active, CTX_SIZING_CACHE_RESERVE, transient, per_tok, cap, 0);
    const ctx_big_ask = resolvedContextForLoad(0, 0, live_ceiling, active, CTX_SIZING_CACHE_RESERVE, transient, per_tok, cap, 0);
    try t.expectEqual(ctx_default, ctx_big_ask);
    try t.expect(ctx_default > 900_000); // a real context, not the floor

    // The defect, both spellings: sizing against the granted budget...
    const vs_granted = resolvedContextForLoad(0, 0, live_ceiling, active, 48_673 * MiB, transient, per_tok, cap, 0);
    try t.expect(vs_granted <= 1024);
    // ...and against the raw ask, which saturates usable to zero.
    const vs_raw_ask = resolvedContextForLoad(0, 0, live_ceiling, active, 60 * 1024 * MiB, transient, per_tok, cap, 0);
    try t.expect(vs_raw_ask <= 1024);
    try t.expect(ctx_default > vs_granted * 500);
}

test "the clamp bills the context that will be SERVED, not the placeholder" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    // With a placeholder context the clamp hands back nearly the whole ask.
    const t = std.testing;
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    var cfg = fixtures.qwen4RequestTestConfig();
    cfg.mtp_override = false; // the MTP head's own term: `the load-time bill prices the MTP head ...`
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const active: u64 = 69_827 * MiB;
    const static_ceiling: u64 = 120_000 * MiB;
    const ask: u64 = 60 * 1024 * MiB;

    const placeholder = planHotCache(&cfg, kv_bits, static_ceiling, active, 1024, 0, ask, 4096);
    const real = planHotCache(&cfg, kv_bits, static_ceiling, active, 1_048_576, 0, ask, 4096);

    try t.expect(placeholder.budget > 45_000 * MiB); // reproduces the live 48,673
    // 13,056 KV + 5,376 state (one history copy + the f32 score bank).
    try t.expectEqual(@as(u64, 1_048_576) * (13_056 + 2_304) + cfg.qsaRingBytes(), real.ctx_kv);

    // Both budgets are their headroom, so the whole difference is the session each billed.
    try t.expectEqual(placeholder.budget - real.budget, real.ctx_kv - placeholder.ctx_kv);
    try t.expect(real.budget < placeholder.budget);
    try t.expect(placeholder.ctx_kv < 30 * MiB);
    try t.expect(real.ctx_kv > 15_000 * MiB);
    try t.expect(real.budget > 0);
}

test "manualContext: an explicit --ctx-size outranks the model's ctx_size, which outranks auto" {
    const t = std.testing;
    const original = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = original;
    var cfg = model_mod.ModelConfig{};
    cfg.ctx_override = 4096;
    launch_state.server_config.max_context_size = 16384;
    try t.expectEqual(@as(u32, 16384), manualContext(&cfg));
    launch_state.server_config.max_context_size = 0;
    try t.expectEqual(@as(u32, 4096), manualContext(&cfg));
    cfg.ctx_override = 0;
    try t.expectEqual(@as(u32, 0), manualContext(&cfg));
}

test "with no --kv-quant, the per-token KV bill and the auto-context follow the kv8 engine default" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    var cfg = fixtures.qwen4RequestTestConfig();
    const saved_kv = launch_state.configured_kv_quant;
    defer launch_state.configured_kv_quant = saved_kv;

    // One fixed memory reading for every arm: live free RAM moves between two reads.
    const ceiling: u64 = 96 << 30;
    const active: u64 = 90 << 30; // tight enough that the checkpoint's own cap does not bind
    launch_state.configured_kv_quant = null;
    try t.expectEqual(@as(u64, 8), defaultKvBits(&cfg));
    try t.expectEqual(sessionBytesPerToken(&cfg, 8), sessionBytesPerToken(&cfg, defaultKvBits(&cfg)));
    const ctx_default = memoryContextAt(&cfg, ceiling, active);

    launch_state.configured_kv_quant = trunk.KVQuantConfig.affine(8);
    try t.expectEqual(ctx_default, memoryContextAt(&cfg, ceiling, active));
    launch_state.configured_kv_quant = trunk.KVQuantConfig.dense;
    try t.expect(memoryContextAt(&cfg, ceiling, active) < ctx_default);

    // A per-model setting still outranks the default.
    launch_state.configured_kv_quant = null;
    cfg.kv_quant_override = trunk.KVQuantConfig.dense;
    try t.expectEqual(@as(u64, 16), defaultKvBits(&cfg));
}

test "an explicit --ctx-size keeps the load-time context byte-identical" {
    const t = std.testing;
    const MiB: u64 = 1 << 20;
    const per_tok: u64 = 20_736;
    try t.expectEqual(@as(u32, 1_048_576), resolvedContextForLoad(1_048_576, 0, 0, 0, 99_000 * MiB, 99_000 * MiB, per_tok, 262_144, 0));
    try t.expectEqual(@as(u32, 262_144), resolvedContextForLoad(0, 262_144, 0, 0, 99_000 * MiB, 0, per_tok, 1_048_576, 0));
}

test "clampReserveWidth: the load-time reserve is a promise to the FIRST request" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // The ceiling inversion: same ask, `iogpu.wired_limit_mb 120000` got a 3.6x smaller cache
    // because the wider rung's reserve grew faster than the headroom. Billing the ladder floor
    // makes the budget monotone in the ceiling.
    const t = std.testing;
    const cfg = fixtures.qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const GiB: u64 = 1 << 30;
    const weights: u64 = 70 * GiB;
    const ctx_tokens: u64 = 1_048_576;
    const floor_rung = PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1];

    const low = planHotCache(&cfg, kv_bits, weights + 28_909 * MiB, weights, ctx_tokens, 0, 10 * GiB, 0);
    const high = planHotCache(&cfg, kv_bits, weights + 43_000 * MiB, weights, ctx_tokens, 0, 10 * GiB, 0);
    try t.expectEqual(floor_rung, low.reserve_chunk);
    try t.expectEqual(floor_rung, high.reserve_chunk);
    try t.expectEqual(prefillTransientReserve(&cfg, kv_bits, floor_rung), low.reserve);
    // A bigger ceiling never buys a smaller cache.
    try t.expect(high.budget >= low.budget);
    try t.expect(low.chunk >= low.reserve_chunk);

    var prev: u64 = 0;
    var mb: u64 = 24_000;
    while (mb <= 60_000) : (mb += 1_000) {
        const p = planHotCache(&cfg, kv_bits, weights + mb * MiB, weights, ctx_tokens, 0, 10 * GiB, 0);
        try t.expect(p.budget >= prev);
        prev = p.budget;
    }

    // An explicit --prefill-chunk caps the ladder, so the clamp still reserves the floor.
    settings_resolve.prefill_chunk.value = 4096;
    settings_resolve.prefill_chunk.source = .flag;
    const capped = planHotCache(&cfg, kv_bits, weights + 43_000 * MiB, weights, ctx_tokens, 0, 10 * GiB, explicitPrefillChunk());
    try t.expectEqual(@as(u32, 4096), capped.chunk);
    try t.expectEqual(floor_rung, capped.reserve_chunk);
    try t.expectEqual(prefillTransientReserve(&cfg, kv_bits, floor_rung), capped.reserve);
    // With the ladder off it is a pin, billed as-is.
    per_request_chunk_override = false;
    const pinned = planHotCache(&cfg, kv_bits, weights + 43_000 * MiB, weights, ctx_tokens, 0, 10 * GiB, explicitPrefillChunk());
    per_request_chunk_override = null;
    settings_resolve.prefill_chunk.value = 8192;
    settings_resolve.prefill_chunk.source = .default;
    try t.expectEqual(@as(u32, 4096), pinned.reserve_chunk);
    try t.expectEqual(prefillTransientReserve(&cfg, kv_bits, 4096), pinned.reserve);

    // Gate off: the clamp reserves at the sizer's rung.
    per_request_chunk_override = false;
    defer per_request_chunk_override = null;
    const off = planHotCache(&cfg, kv_bits, weights + 28_909 * MiB, weights, ctx_tokens, 0, 10 * GiB, 0);
    try t.expectEqual(off.chunk, off.reserve_chunk);
    try t.expectEqual(prefillTransientReserve(&cfg, kv_bits, off.chunk), off.reserve);
    per_request_chunk_override = null;

    var other = fixtures.qwen4RequestTestConfig();
    other.model_type = "not_a_served_arch";
    const neutral = planHotCache(&other, kv_bits, weights + 28_909 * MiB, weights, ctx_tokens, 0, 10 * GiB, 0);
    try t.expectEqual(neutral.chunk, neutral.reserve_chunk);
}

test "the per-token widths the request chooser is built on" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    // Pin the two constants every number in the story is derived from.
    const t = std.testing;
    const cfg = fixtures.qwen4RequestTestConfig();
    try t.expectEqual(@as(u32, 12), cfg.attnCacheLayerCount());
    try t.expectEqual(@as(u64, 13_056), kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), 8));
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    try t.expectEqual(@as(u64, 768), cfg.qsaHistoryBytesPerToken());
    try t.expectEqual(@as(u64, 1_536), cfg.qsaScoreBankBytesPerToken());
    try t.expectEqual(@as(u64, 2_304), statePerTokenBilled(&cfg));
    try t.expectEqual(@as(u64, 15_360 * (1 << 20)), (13_056 + 2_304) * @as(u64, 1_048_576));
}

test "prefillMemoryNeeded: the wider chunk bills every chunk-scaled term at the wider width" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const t = std.testing;
    const cfg = fixtures.qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const seq: u64 = 300_000;
    const ffn: u64 = prefillFfnWidth(&cfg);
    const stream: u64 = prefillStreamBytesPerToken(&cfg);
    const dq: u64 = prefillDequantWeightBytes(&cfg);
    const keys: u64 = cfg.prefillAttnKeys(seq);

    const narrow = prefillMemoryNeeded(seq, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.kvBytesPerToken(), cfg.head_dim, cfg.prefillScoreHeadDim(), cfg.hidden_size, ffn, kv_bits, 4096, keys, stream, dq, .{});
    const wide = prefillMemoryNeeded(seq, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.kvBytesPerToken(), cfg.head_dim, cfg.prefillScoreHeadDim(), cfg.hidden_size, ffn, kv_bits, 8192, keys, stream, dq, .{});

    const mlp_step: u64 = 8 * 4096 * @max(@as(u64, cfg.hidden_size), ffn) * 2;
    const envelope_step: u64 = @max(3 * mlp_step, mlp_step + 4096 * stream);
    try t.expect(wide >= narrow + envelope_step * 5 / 4);
    try t.expect(qsaMaskBytes(&cfg, 8192, seq) > qsaMaskBytes(&cfg, 4096, seq));
}

test "an explicit --prefill-chunk is the widest width a request may run, not a pin" {
    if (!archBuilt("mimo2")) return error.SkipZigTest;
    const t = std.testing;
    attn_kernels.fused256_override = true;
    defer attn_kernels.fused256_override = null;
    const saved_chunk = settings_resolve.prefill_chunk.value;
    const saved_source = settings_resolve.prefill_chunk.source;
    defer {
        settings_resolve.prefill_chunk.value = saved_chunk;
        settings_resolve.prefill_chunk.source = saved_source;
    }
    // The live shape: MiMo, `--prefill-chunk 4096`, a 139,717-token turn.
    const cfg = fixtures.mimoV2FlashBillConfig();
    const kv_bits: u64 = 8;
    const seq: u64 = 139_717;
    const pin: u32 = 2048;
    settings_resolve.prefill_chunk.value = 4096;
    settings_resolve.prefill_chunk.source = .flag;
    const n = explicitPrefillChunk();
    const at_n = prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, 4096, .{});
    const at_2048 = prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, 2048, .{});
    const at_floor = prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, 512, .{});
    try t.expect(at_2048 < at_n);

    // N when it fits, however roomy: never wider.
    try t.expectEqual(@as(u32, 4096), chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, at_n, pin, n, .{}));
    try t.expectEqual(@as(u32, 4096), chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, 200 << 30, pin, n, .{}));
    // Memory that fits only 2048 runs 2048 instead of refusing.
    try t.expectEqual(@as(u32, 2048), chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, at_n - 1, pin, n, .{}));
    try t.expectEqual(@as(u32, 2048), chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, at_2048, pin, n, .{}));
    // Nothing fits: the floor, whose bill the guard then refuses.
    const floor_w = chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, at_floor - 1, pin, n, .{});
    try t.expectEqual(@as(u32, 512), floor_w);
    try t.expectEqual(AdmissionVerdict.refuse, admissionVerdict(.{ .needed = prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, floor_w, .{}), .available = at_floor - 1 }));

    // The bill and the forward run the one width the pick names.
    const pick = requestPrefillPick(&cfg, seq, 2048, kv_bits, at_2048, false, .{});
    try t.expectEqual(@as(u32, 2048), pick);
    try t.expectEqual(@as(usize, pick), prefill_plan.requestPrefillChunk(cfg.prefillScoreHeadDim(), cfg.num_attention_heads, seq, cfg.has_sliding_window, cfg.isMoe(), cfg.longCtxGated(), pick));

    // A narrower N is the cap for every rung; one below the floor is the floor.
    settings_resolve.prefill_chunk.value = 3000;
    try t.expectEqual(@as(u32, 3000), chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, 200 << 30, pin, explicitPrefillChunk(), .{}));
    settings_resolve.prefill_chunk.value = 256;
    try t.expectEqual(@as(u32, 256), chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, 0, pin, explicitPrefillChunk(), .{}));

    // Per-request picking off: the explicit width is the pin again, on the bill and the forward.
    settings_resolve.prefill_chunk.value = 4096;
    per_request_chunk_override = false;
    defer per_request_chunk_override = null;
    const off = requestPrefillPick(&cfg, seq, 2048, kv_bits, 0, false, .{});
    try t.expectEqual(@as(u32, 4096), off);
    try t.expectEqual(@as(usize, 4096), prefill_plan.requestPrefillChunk(cfg.prefillScoreHeadDim(), cfg.num_attention_heads, seq, cfg.has_sliding_window, cfg.isMoe(), cfg.longCtxGated(), off));
}

test "admissionLogLevel: free on the first request after a load and at the edge, quiet in the middle" {
    const t = std.testing;
    const GiB: u64 = 1 << 30;

    // A fresh load is always free.
    try t.expectEqual(log.Level.info, admissionLogLevel(1 * GiB, 100 * GiB, true));

    // Inside 10% of what is available.
    try t.expectEqual(log.Level.info, admissionLogLevel(95 * GiB, 100 * GiB, false));
    try t.expectEqual(log.Level.info, admissionLogLevel(120 * GiB, 100 * GiB, false)); // already over
    // The boundary itself: exactly 0.9 is quiet.
    try t.expectEqual(log.Level.debug, admissionLogLevel(90 * GiB, 100 * GiB, false));
    try t.expectEqual(log.Level.info, admissionLogLevel(90 * GiB + 1, 100 * GiB, false));

    try t.expectEqual(log.Level.debug, admissionLogLevel(1 * GiB, 100 * GiB, false));

    try t.expectEqual(log.Level.info, admissionLogLevel(1 * GiB, 0, false));

    try t.expectEqual(log.Level.info, admissionLogLevel(std.math.maxInt(u64), 1, false));
    try t.expectEqual(log.Level.debug, admissionLogLevel(1, std.math.maxInt(u64), false));
}

test "admissionVerdict names the same three arms the guard acts on" {
    const t = std.testing;
    try t.expectEqual(AdmissionVerdict.admit, admissionVerdict(.{ .needed = 10, .available = 20 }));
    try t.expectEqual(AdmissionVerdict.admit, admissionVerdict(.{ .needed = 20, .available = 20 }));
    // Fits once the cache gives back what it can prove: `reclaimable`, never `evictable`.
    try t.expectEqual(AdmissionVerdict.evict, admissionVerdict(.{ .needed = 30, .available = 20, .reclaimable = 10 }));
    try t.expectEqual(AdmissionVerdict.refuse, admissionVerdict(.{ .needed = 30, .available = 20, .evictable = 1000, .reclaimable = 0 }));
    try t.expectEqual(AdmissionVerdict.refuse, admissionVerdict(.{ .needed = 100, .available = 20, .reclaimable = 10 }));

    try t.expectEqualStrings("admit", AdmissionVerdict.admit.name());
    try t.expectEqualStrings("evict", AdmissionVerdict.evict.name());
    try t.expectEqualStrings("refuse", AdmissionVerdict.refuse.name());
}

test "a prompt that does not fit unloads idle models, oldest first, until it fits" {
    const t = std.testing;
    const Fake = struct {
        needed: u64,
        available: u64,
        idle: []const u64, // resident bytes, least recently used first
        unloaded: usize = 0,
        fn fits(f: *@This()) bool {
            return f.needed <= f.available;
        }
        fn unloadOne(f: *@This()) bool {
            if (f.unloaded == f.idle.len) return false;
            f.available += f.idle[f.unloaded];
            f.unloaded += 1;
            return true;
        }
    };
    // Already fits: nothing is unloaded.
    var a = Fake{ .needed = 5, .available = 9, .idle = &.{ 15, 3 } };
    try t.expect(fitAfterUnloadingIdle(&a, Fake.fits, Fake.unloadOne));
    try t.expectEqual(@as(usize, 0), a.unloaded);
    // One idle model is enough: the second stays resident.
    var b = Fake{ .needed = 10_558, .available = 9_466, .idle = &.{ 14_730, 8_000 } };
    try t.expect(fitAfterUnloadingIdle(&b, Fake.fits, Fake.unloadOne));
    try t.expectEqual(@as(usize, 1), b.unloaded);
    // Every idle model goes and it still does not fit: refused.
    var c = Fake{ .needed = 50, .available = 9, .idle = &.{ 15, 3 } };
    try t.expect(!fitAfterUnloadingIdle(&c, Fake.fits, Fake.unloadOne));
    try t.expectEqual(@as(usize, 2), c.unloaded);
}

test "a refusal names the credit it COMPARED, never the residency it cannot take" {
    const t = std.testing;
    const mb: u64 = 1024 * 1024;

    // The live shape: the whole residency is the checked-out entry.
    const pinned_only = AdmissionBill{
        .needed = 13_664 * mb,
        .available = 13_374 * mb,
        .evictable = 7_802 * mb,
        .reclaimable = 0,
        .chunk = 512,
    };
    try t.expectEqual(@as(u64, 0), pinned_only.evictionCredit());
    try t.expectEqual(7_802 * mb, pinnedResidentBytes(pinned_only));
    try t.expect(!pinned_only.fits());
    try t.expect(!pinned_only.fitsAfterEviction());
    try t.expectEqual(
        pinned_only.needed <= pinned_only.available +| pinned_only.evictionCredit(),
        pinned_only.fitsAfterEviction(),
    );

    // The same residency, genuinely reclaimable: the credit is real and the request is admitted to evict.
    var reclaimable = pinned_only;
    reclaimable.reclaimable = 7_802 * mb;
    try t.expectEqual(7_802 * mb, reclaimable.evictionCredit());
    try t.expectEqual(@as(u64, 0), pinnedResidentBytes(reclaimable));
    try t.expect(reclaimable.fitsAfterEviction());
}

test "safeAutoContext leaves headroom below the memory ceiling and rounds to 1024" {
    // The raw ceiling is the largest context that FITS. Running at exactly that
    // leaves nothing for the prefix cache to grow into, a second model, or
    // another app — so admit only `auto_ctx_safety_pct` of it.
    try testing.expectEqual(@as(u32, 88064), safeAutoContext(94729)); // 93% = 88098 -> 1024-floor
    try testing.expectEqual(@as(u32, 29696), safeAutoContext(32768)); // 93% = 30474 -> 1024-floor
    // Never rounds down to zero on a tiny ceiling.
    try testing.expect(safeAutoContext(1000) > 0);
    try testing.expect(safeAutoContext(1) > 0);
    // Monotonic, and always strictly inside the ceiling for real-sized contexts.
    var prev: u32 = 0;
    for ([_]u32{ 4096, 8192, 16384, 32768, 65536, 94729, 262144 }) |raw| {
        const got = safeAutoContext(raw);
        try testing.expect(got <= raw);
        try testing.expect(got >= prev);
        prev = got;
    }
}

test "every served model advertises 93% of the memory ceiling, capped at its own maximum" {
    if (!archBuilt("qwen4") or !archBuilt("mimo2") or !archBuilt("glm5")) return error.SkipZigTest;
    const t = testing;
    var qwen = try model_mod.parseConfigFromJson(t.allocator, @embedFile("fixtures/model-configs/qwen4_exp.json"));
    defer qwen.deinit(t.allocator);
    var mimo = try model_mod.parseConfigFromJson(t.allocator, arch_config.fixtureConfig("mimo2") orelse return error.SkipZigTest);
    defer mimo.deinit(t.allocator);
    var glm = try model_mod.parseConfigFromJson(t.allocator, @embedFile("fixtures/glm5_config.json"));
    defer glm.deinit(t.allocator);
    for ([_]*const model_mod.ModelConfig{ &qwen, &mimo, &glm }) |cfg| {
        try t.expectEqual(@as(u32, 1_048_576), cfg.contextCap());
        try t.expectEqual(@as(u32, 88064), autoContextFrom(94729, cfg.contextCap(), 0));
        try t.expectEqual(cfg.contextCap(), autoContextFrom(4_000_000, cfg.contextCap(), 0));
        try t.expectEqual(@as(u32, 974_848), autoContextFrom(1_048_576, cfg.contextCap(), 0));
    }
}

test "safeContextForBudget reserves the hot-cache budget (2026-06-19 OOM regression)" {
    const GB: u64 = 1 << 30;
    // A footprint that OOM'd a 16 GB Mac: 32 layers, 4 kv heads, head_dim 256.
    const kv_per_tok: u64 = 32 * 2 * 4 * 256 * 2; // 131072
    const work_per_tok: u64 = 8 * 9216 * 2; //        147456
    const per_tok: u64 = kv_per_tok + work_per_tok;

    // Mid-session: ~6 GB resident (model + a partly-filled hot cache), 2 GB cache cap.
    const ws: u64 = 12 * GB; // ~Metal recommended working set on a 16 GB Mac
    const active: u64 = 6 * GB;
    const cache: u64 = 2 * GB;

    const with_reserve = safeContextForBudget(ws, active, cache, per_tok, 0);
    const without_reserve = safeContextForBudget(ws, active, 0, per_tok, 0);

    // The whole point of the fix: subtracting the hot-cache budget shrinks the
    // reported context so it still fits once the cache fills (it didn't before
    // — auto-ctx was computed against an empty cache and later collided with it).
    try testing.expect(with_reserve < without_reserve);
    // Still usable, never floored to nothing.
    try testing.expect(with_reserve > 1024);
}

test "parseWiredMarginGib accepts 1..32 and names a bad value" {
    const t = std.testing;
    try t.expectEqual(@as(u64, 1) << 30, try parseWiredMarginGib("1"));
    try t.expectEqual(@as(u64, 8) << 30, try parseWiredMarginGib("8"));
    try t.expectEqual(@as(u64, 6) << 30, try parseWiredMarginGib("6"));
    try t.expectEqual(@as(u64, 2) << 30, try parseWiredMarginGib("2"));
    try t.expectEqual(@as(u64, 32) << 30, try parseWiredMarginGib("32"));
    try t.expectError(error.InvalidWiredMargin, parseWiredMarginGib("0"));
    try t.expectError(error.InvalidWiredMargin, parseWiredMarginGib("33"));
    try t.expectError(error.InvalidWiredMargin, parseWiredMarginGib("nope"));
    try t.expectError(error.InvalidWiredMargin, parseWiredMarginGib(""));
}

test "physicalMemoryCeiling caps the static GPU max by real free RAM (#64 docker OOM)" {
    const GB: u64 = 1 << 30;
    const static_max: u64 = 115 * GB; // Metal max_recommended_working_set_size, 128 GB Mac

    // Idle: MLX holds 26 GB (active) + 5 GB (reclaimable cache), ~84 GB free.
    // 31 + 84 = 115 ⇒ the static device max still binds — NO regression when the
    // machine is uncontended (auto-context stays generous).
    try testing.expectEqual(static_max, physicalMemoryCeiling(static_max, 31 * GB, 84 * GB));

    // #64: a docker-compose stack (firecrawl/rabbitmq/postgres/playwright) holds
    // ~45 GB. MLX holds 26 GB, only ~35 GB is physically free. The static max is
    // BLIND to docker, but the ceiling must collapse to 26 + 35 = 61 GB — otherwise
    // a large MoE prefill is admitted and Metal throws an uncatchable OOM.
    try testing.expectEqual(@as(u64, 61 * GB), physicalMemoryCeiling(static_max, 26 * GB, 35 * GB));

    // Saturating add: a bogus/huge footprint reading can never wrap to a tiny ceiling.
    try testing.expectEqual(static_max, physicalMemoryCeiling(static_max, std.math.maxInt(u64), GB));
}

test "auto-context tightens under external memory pressure vs the pre-fix static ceiling (#64)" {
    const GB: u64 = 1 << 30;
    // A 35B-A3B MoE footprint (per-token KV + working envelope).
    const per_tok: u64 = 328 * 1024;
    const static_max: u64 = 115 * GB;
    const active: u64 = 26 * GB;
    const mlx_cache: u64 = 4 * GB;
    const hot_cache_reserve: u64 = 2 * GB;
    const max_pos: u32 = 262144;

    // Pre-fix behaviour: budget straight against the static device max, blind to
    // free RAM — the exact code path that OOM'd #64.
    const prefix_ctx = safeContextForBudget(static_max, active, hot_cache_reserve, per_tok, max_pos);

    // Post-fix, idle (~84 GB free): ceiling ≈ static max ⇒ context essentially unchanged.
    const idle_ceiling = physicalMemoryCeiling(static_max, active + mlx_cache, 84 * GB);
    const idle_ctx = safeContextForBudget(idle_ceiling, active, hot_cache_reserve, per_tok, max_pos);
    try testing.expect(idle_ctx >= prefix_ctx * 9 / 10); // within 10% of the old generous value

    // Post-fix, docker holds ~45 GB (~35 GB free): ceiling collapses, so the safe
    // context is far smaller than the pre-fix static-max computation would allow.
    const busy_ceiling = physicalMemoryCeiling(static_max, active + mlx_cache, 35 * GB);
    const busy_ctx = safeContextForBudget(busy_ceiling, active, hot_cache_reserve, per_tok, max_pos);
    try testing.expect(busy_ctx < prefix_ctx / 2);
    try testing.expect(busy_ctx > 1024); // still usable, never floored to nothing
}

test "safeContextForBudget floors when memory is exhausted and caps at max_pos" {
    const GB: u64 = 1 << 30;
    const per_tok: u64 = 278528;
    // working set already below what's resident → no room → minimum floor.
    try testing.expectEqual(@as(u32, 1024), safeContextForBudget(4 * GB, 5 * GB, 0, per_tok, 0));
    // active + reserve exceeds the ceiling → floor.
    try testing.expectEqual(@as(u32, 1024), safeContextForBudget(8 * GB, 6 * GB, 4 * GB, per_tok, 0));
    // Plenty of memory, but capped at the model's max position embeddings.
    try testing.expectEqual(@as(u32, 4096), safeContextForBudget(256 * GB, 0, 0, per_tok, 4096));
    // Degenerate per_tok never divides by zero.
    try testing.expectEqual(@as(u32, 1024), safeContextForBudget(8 * GB, 0, 0, 0, 0));
}

test "kvBytesPerTokenAtBits: the KV bill follows the CONFIGURED quant width" {
    const t = std.testing;
    // Dense: 16 caching layers x 4 kv heads x 256 head_dim x (K+V) x 2 B.
    const dense: u64 = 16 * 4 * 256 * 2 * 2; // 65536

    // Dense/off bills fp16 verbatim — no change for anyone not using --kv-quant.
    try t.expectEqual(dense, kvBytesPerTokenAtBits(dense, 16));
    try t.expectEqual(dense, kvBytesPerTokenAtBits(dense, 32));

    // Quantized widths carry the group scale/bias overhead (+0.5 bit/elem), the
    // SAME expression prefillMemoryNeeded bills — the sizer and the admission
    // guard must not disagree about what one token costs.
    try t.expectEqual(dense * 9 / 32, kvBytesPerTokenAtBits(dense, 4)); // 18432
    try t.expectEqual(dense * 13 / 32, kvBytesPerTokenAtBits(dense, 6));
    try t.expectEqual(dense * 17 / 32, kvBytesPerTokenAtBits(dense, 8));
    // The bill is what the cache stores: `bits` per element plus a bf16 scale and bias per group of 64.
    for ([_]u64{ 4, 6, 8 }) |bits| {
        const elems = dense / 2;
        try t.expectEqual(elems * bits / 8 + elems / 64 * 4, kvBytesPerTokenAtBits(dense, bits));
    }
    // 4-bit is what makes the 16 GB tier reachable: under a third of dense.
    try t.expect(kvBytesPerTokenAtBits(dense, 4) * 3 < dense);
}

test "auto-context on a 16 GB profile: activations are a chunk reserve, not a per-token multiplier" {
    const t = std.testing;
    // Qwen3.8-27B on a 16 GB Mac: ~10.6 GB reachable Metal working set, an
    // 8.6 GB iQ-MLX pack resident, --kv-quant 4, and the small prefill chunk
    // the machine can actually afford.
    const ceiling: u64 = 10_600 * 1000 * 1000;
    const weights: u64 = 8_600 * 1000 * 1000;
    const dense_kv: u64 = 16 * 4 * 256 * 2 * 2;
    const per_tok = kvBytesPerTokenAtBits(dense_kv, 4);

    var cfg = model_mod.ModelConfig{ .model_type = "not_a_served_arch" };
    cfg.num_attention_heads = 24;
    cfg.num_key_value_heads = 4;
    cfg.head_dim = 256;
    cfg.hidden_size = 5120;
    cfg.intermediate_size = 17408;
    cfg.intermediate_size_declared = true;
    // The real checkpoint's hybrid geometry: 64 layers, every 4th full
    // attention, 48 GatedDeltaNet layers whose chunk-wide q/k/v streams are
    // two thirds of this model's measured prefill transient.
    cfg.num_hidden_layers = 64;
    cfg.full_attention_interval = 4;
    cfg.linear_num_key_heads = 16;
    cfg.linear_num_value_heads = 48;

    const reserve = prefillTransientReserve(&cfg, 4, 512);
    const got = safeContextForBudget(ceiling, weights, reserve, per_tok, 262144);

    // The pre-fix formula billed 8 x max(hidden, ffn) x 2 = 272 KB PER TOKEN on
    // top of a 64 KB/token dense KV — 344 KB/token, which caps this machine at
    // under 4k tokens no matter what the weights cost.
    const old_per_tok: u64 = dense_kv + 8 * 17408 * 2;
    const old = safeContextForBudget(ceiling, weights, 0, old_per_tok, 262144);
    try t.expect(old < 4096);

    // Corrected, the same machine holds a real working context. The band is
    // what the RESERVE is worth, and the reserve is measured: this checkpoint
    // peaks 1.06 GB above steady state at chunk 512 (M4 Max, 2026-08-14), so
    // a sizer that reported the ~51k tokens the first cut of this fix did was
    // spending memory the prefill needs. Halve the chunk and it comes back.
    try t.expect(got > 15_000);
    try t.expect(got < 40_000);

    // The reserve is a CONSTANT of the chunk, so halving the chunk buys context
    // back rather than being invisible.
    const narrow = safeContextForBudget(ceiling, weights, prefillTransientReserve(&cfg, 4, 256), per_tok, 262144);
    try t.expect(narrow > got);
}

test "auto-context never sizes past the ceiling as weights grow" {
    const t = std.testing;
    const ceiling: u64 = 10_600 * 1000 * 1000;
    const dense_kv: u64 = 16 * 4 * 256 * 2 * 2;
    const per_tok = kvBytesPerTokenAtBits(dense_kv, 4);
    var cfg = model_mod.ModelConfig{};
    cfg.num_attention_heads = 24;
    cfg.num_key_value_heads = 4;
    cfg.head_dim = 256;
    cfg.hidden_size = 5120;
    cfg.intermediate_size = 17408;
    cfg.intermediate_size_declared = true;
    const reserve = prefillTransientReserve(&cfg, 4, 512);

    // Under-billing here ends in an uncatchable Metal OOM, not a 400: at every
    // weight size the reported context plus what is already spoken for has to
    // stay inside the ceiling, and the reported figure must fall as the pack
    // grows rather than staying pinned to the checkpoint's max.
    var prev: u32 = std.math.maxInt(u32);
    var weights: u64 = 2_000 * 1000 * 1000;
    while (weights < ceiling) : (weights += 1_000 * 1000 * 1000) {
        const ctx = safeContextForBudget(ceiling, weights, reserve, per_tok, 262144);
        try t.expect(weights + reserve + @as(u64, ctx) * per_tok <= ceiling or ctx == 1024);
        try t.expect(ctx <= prev);
        prev = ctx;
    }
    // Squeezed to nothing it floors rather than reporting a context that cannot exist.
    try t.expectEqual(@as(u32, 1024), safeContextForBudget(ceiling, ceiling, reserve, per_tok, 262144));
}

test "expertPoolBytes: only DeepSeek-V4.1 takes a ledger pool, within the RAM-wide cap" {
    if (!archBuilt("qwen4") or !archBuilt("mimo2") or !archBuilt("glm5") or !archBuilt("dsv41")) return error.SkipZigTest;
    const ds = model_mod.ModelConfig{ .model_type = "deepseek_v41" };
    try testing.expect(ds.expertPoolBytes() > 0);
    try testing.expect(ds.expertPoolBytes() <= mlxCacheLimitBytes(128 << 30));
    for ([_][]const u8{ "qwen4_exp", "mimo_v2", "glm5_next" }) |arch| {
        const c = model_mod.ModelConfig{ .model_type = arch };
        try testing.expectEqual(@as(u64, 0), c.expertPoolBytes());
    }
}

test "mlxCacheLimitBytes: RAM-proportional cap, 2 GB floor, 8 GB ceiling" {
    const GB: u64 = 1 << 30;
    // MLX's own default is min(1.5 x working_set, 0.95 x RAM) — ~121 GB on a
    // 128 GB Mac (`backend/metal/allocator.cpp`), i.e. it will not trim until
    // the machine is already dead. RAM/16 keeps the pool big enough for
    // step-to-step buffer reuse without letting it become the footprint.
    try testing.expectEqual(8 * GB, mlxCacheLimitBytes(128 * GB));
    try testing.expectEqual(4 * GB, mlxCacheLimitBytes(64 * GB));
    try testing.expectEqual(2 * GB, mlxCacheLimitBytes(32 * GB)); // 2 GB exactly
    try testing.expectEqual(2 * GB, mlxCacheLimitBytes(16 * GB)); // floor
    try testing.expectEqual(2 * GB, mlxCacheLimitBytes(8 * GB)); // floor
    // Ceiling holds above 128 GB (Mac Studio / M3 Ultra 512 GB).
    try testing.expectEqual(8 * GB, mlxCacheLimitBytes(512 * GB));
    // A failed `hw.memsize` read is 0 — never clamp on bad data.
    try testing.expectEqual(@as(u64, 0), mlxCacheLimitBytes(0));
}

test "prefix cache defaults: no RAM retention, an unnamed SSD tier on" {
    try testing.expect(!launch_state.prefix_cache_ram_enabled);
    try testing.expectEqual(@as(u64, 0), resolvedPrefixCacheMem());
    try testing.expect(launch_state.prefix_cache_mem_source == .default and launch_state.prefix_cache_disk_source == .default);
    try testing.expectEqual(disk_placement.DEFAULT_DISK_CAP, launch_state.prefix_cache_disk_bytes);
}

test "kvBytesPerToken bills only the CACHING layers, at the arch's own K and V widths" {
    const t = std.testing;
    // Uniform arch: every layer caches, K and V are both head_dim wide.
    var dense = model_mod.ModelConfig{ .model_type = "not_a_served_arch" };
    dense.num_hidden_layers = 24;
    dense.num_key_value_heads = 8;
    dense.head_dim = 128;
    try t.expectEqual(@as(u32, 24), dense.attnCacheLayerCount());
    try t.expectEqual(@as(u64, 24 * 8 * 256 * 2), dense.kvBytesPerToken());

    // A hybrid with 6 of 24 layers attention, whose MLA stores
    // a 192-wide key against a 128-wide value. Billing it as a uniform arch
    // charged 196608 B/token against a real 61440 — a 3.2x over-bill that
    // pinned auto-context to under a third of what fits.
    var ling = model_mod.ModelConfig{ .model_type = "not_a_served_arch" };
    ling.num_hidden_layers = 24;
    ling.num_attention_heads = 16;
    ling.num_key_value_heads = 16;
    ling.head_dim = 128;
    ling.full_attention_interval = 4;
    ling.mla_kv_lora_rank = 512;
    ling.mla_qk_nope_head_dim = 128;
    ling.mla_qk_rope_head_dim = 64;
    ling.mla_v_head_dim = 128;
    try t.expectEqual(@as(u32, 6), ling.attnCacheLayerCount());
    try t.expectEqual(@as(u64, 6 * 16 * (192 + 128) * 2), ling.kvBytesPerToken());
    try t.expect(ling.kvBytesPerToken() * 3 < 24 * 2 * 16 * 128 * 2);

    // MLA caches per ATTENTION head — it decompresses the latent to all of
    // them, so there is no grouping to bill against. Ling 3.0 ships 16/16, so
    // only an asymmetric config can tell the two spellings apart, and reading
    // `num_key_value_heads` here UNDER-bills (2 of 16 heads = 8x light) —
    // which surfaces as an uncatchable Metal OOM, not a refusal.
    var mla_grouped = ling;
    mla_grouped.num_key_value_heads = 2;
    try t.expectEqual(@as(u64, 6 * 16 * (192 + 128) * 2), mla_grouped.kvBytesPerToken());
    // ...while a NON-MLA arch still bills its KV heads, grouping and all.
    var gqa = dense;
    gqa.num_attention_heads = 32;
    gqa.num_key_value_heads = 8;
    try t.expectEqual(@as(u64, 24 * 8 * 256 * 2), gqa.kvBytesPerToken());

    // A hybrid WITHOUT MLA still drops its linear layers from the bill.
    var gdn = model_mod.ModelConfig{ .model_type = "not_a_served_arch" };
    gdn.num_hidden_layers = 40;
    gdn.num_key_value_heads = 2;
    gdn.head_dim = 256;
    gdn.full_attention_interval = 4;
    try t.expectEqual(@as(u32, 10), gdn.attnCacheLayerCount());
    try t.expectEqual(@as(u64, 10 * 2 * 512 * 2), gdn.kvBytesPerToken());
}

test "prefillMemoryNeeded: a prompt shorter than the chunk bills the prompt-width forward, not the chunk cap" {
    // Live 2026-07-14: with 111 GB of weights resident (~4 GB headroom),
    // a 31-TOKEN prompt was rejected "needs ~6246MB" — the MLP envelope was
    // billed at the hd-128 8192 chunk cap while the actual prefill runs ONE
    // 31-token forward (~25 MB). A forward is never wider than the prompt:
    // the envelope must use min(chunk, seq). Guard-tracks-reality rule.
    const small = prefillMemoryNeeded(31, 64, 8, 327680, 128, 128, 4096, 13312, 8, 31, 31, 0, 0, .{});
    const capped = prefillMemoryNeeded(31, 64, 8, 327680, 128, 128, 4096, 13312, 8, 8192, 31, 0, 0, .{});
    try std.testing.expectEqual(small, capped);
    // Sanity: the clamped estimate for the live 31-token case sits far below
    // the ~4 GB that was actually available. It is not zero — every prefill
    // pays PREFILL_RUNTIME_FLOOR_BYTES — but a tiny prompt pays little else,
    // and the dequant-weight term never fires this far under the route's
    // minimum forward width.
    try std.testing.expect(capped < 1024 * 1024 * 1024);
}

test "prefillMemoryNeeded: quantized KV is billed at its real width, not fp16" {
    const t = std.testing;
    attn_kernels.fused256_override = false;
    defer attn_kernels.fused256_override = null;
    // 26B-A4B shape at 100K ctx, chunk 512 (what the bounded chunk picks there).
    // fp16: kv 30x2x100000x8x256x2 = 24.576 GB; scores 16x512x100000x2 = 1.6384 GB;
    //       mlp 3x8x512x2816x2 = 69.2 MB; x1.25 margin.
    try t.expectEqual(
        @as(u64, 32_854_507_520 + 671_088_640),
        prefillMemoryNeeded(100_000, 16, 8, 245760, 256, 256, 2816, 2112, 16, 512, 100_000, 0, 0, .{}),
    );
    // 4-bit: kv billed at (2*4+1)/16 bytes/elem (payload + group scale/bias)
    // = 6.912 GB, plus the 0.8192 GB per-layer dense dequant transient.
    try t.expectEqual(
        @as(u64, 11_798_507_520 + 671_088_640),
        prefillMemoryNeeded(100_000, 16, 8, 245760, 256, 256, 2816, 2112, 4, 512, 100_000, 0, 0, .{}),
    );
    // Direction: quantized admission must be under half the fp16 bill.
    try t.expect(prefillMemoryNeeded(100_000, 16, 8, 245760, 256, 256, 2816, 2112, 4, 512, 100_000, 0, 0, .{}) * 2 <
        prefillMemoryNeeded(100_000, 16, 8, 245760, 256, 256, 2816, 2112, 16, 512, 100_000, 0, 0, .{}));
}

test "prefillMemoryNeeded: working set is chunk-bounded — the 255K MoE prompt is admittable" {
    const t = std.testing;
    attn_kernels.fused256_override = false;
    defer attn_kernels.fused256_override = null;
    // The PR-#69 failure case: Qwen3.6-35B-A3B shape (whose checkpoint OMITS
    // intermediate_size, so the struct default 15360 leaks into ffn) at 262144
    // tokens with 4-bit KV. kv 6.04 GB + scores 4.29 GB + dequant 0.54 GB +
    // mlp 0.377 GB, x1.25 = ~13.1 GiB — admittable on a big Mac.
    const needed = prefillMemoryNeeded(262_144, 16, 2, 81920, 256, 256, 2048, 15360, 4, 512, 262_144, 0, 0, .{});
    try t.expectEqual(@as(u64, 14_061_404_160 + 671_088_640), needed);
    try t.expect(needed < 16 << 30);
    // A seq-scaled envelope (8 x seq x ffn x 2 working bytes, ~64 GB, on top
    // of fp16 KV, ~21.5 GB) would refuse this prompt with a spurious 400.
    const old_estimate: u64 = (40 * 2 * 262_144 * 2 * 256 * 2 + 8 * 262_144 * 15360 * 2) * 5 / 4;
    try t.expect(needed < old_estimate / 7);
}

test "prefillMemoryNeeded: unfused head dims bill the materialized score scratch" {
    const t = std.testing;
    attn_kernels.fused256_override = false;
    defer attn_kernels.fused256_override = null;
    // Same shape, head_dim 128 vs 256 (kv scaled by hdim too, so compare the
    // full bills computed by hand): hd=128 fp16 -> kv 12.288 GB + 0 scores +
    // mlp 69.2 MB, x1.25.
    try t.expectEqual(
        @as(u64, 15_446_507_520 + 671_088_640),
        prefillMemoryNeeded(100_000, 16, 8, 122880, 128, 128, 2816, 2112, 16, 512, 100_000, 0, 0, .{}),
    );
    // hd=256 adds the [heads, chunk, seq] score tensor (the guard must see
    // what the composed SDPA path actually allocates). Decomposition: the
    // hd-128 bill + the KV doubling from 128->256 (12.288 GB x1.25) + the
    // score tensor (16x512x100000x2 = 1.6384 GB x1.25).
    const with_scores = prefillMemoryNeeded(100_000, 16, 8, 245760, 256, 256, 2816, 2112, 16, 512, 100_000, 0, 0, .{});
    try t.expectEqual(@as(u64, 15_446_507_520 + 671_088_640 + 15_360_000_000 + 2_048_000_000), with_scores);
}

test "prefillMemoryNeeded: the SCORE width decides the score term, not the stored width" {
    // MLA contracts scores over qk_head_dim 192 while storing
    // values at v_head_dim 128 and declaring head_dim 128. mlx has a fused
    // vector kernel for that pair at DECODE and falls back to the composed
    // path at prefill widths, so the [heads, chunk, seq] score tensor is real —
    // but `prefillHeadDimFused(128)` is true, so passing the stored width for
    // both billed it at ZERO. Under-billing ends in an uncatchable Metal OOM
    // instead of a 400, which is the whole reason this guard exists.
    const t = std.testing;
    const kv_per_tok: u64 = 6 * 16 * (192 + 128) * 2; // 6 caching layers of 24
    const stored: u64 = 128;
    const scored: u64 = 192;
    // `sushi_attn_pd` serves qk 192 now, so the composed sheet survives only
    // under the kill switch — which is precisely where confusing the two
    // widths would zero a sheet that is really there.
    attn_kernels.fused256_override = false;
    defer attn_kernels.fused256_override = null;
    const honest = prefillMemoryNeeded(32_768, 16, 16, kv_per_tok, stored, scored, 1536, 4608, 16, 4096, 32_768, 0, 0, .{});
    const blind = prefillMemoryNeeded(32_768, 16, 16, kv_per_tok, stored, stored, 1536, 4608, 16, 4096, 32_768, 0, 0, .{});
    // The difference is exactly the score tensor: heads x chunk x seq x 2, x1.25.
    const score_bytes: u64 = 16 * 4096 * 32_768 * 2;
    try t.expectEqual(honest - blind, score_bytes * 5 / 4);
    try t.expect(honest > 2 * blind);
    // The two widths are independent: at fp16 (no dequant transient) the
    // stored width touches nothing the score term reads.
    try t.expectEqual(
        honest,
        prefillMemoryNeeded(32_768, 16, 16, kv_per_tok, scored, scored, 1536, 4608, 16, 4096, 32_768, 0, 0, .{}),
    );
    // With the kernel serving the width there is no sheet to attribute, so the
    // two widths agree — at zero, not by accident.
    attn_kernels.fused256_override = true;
    try t.expectEqual(
        blind,
        prefillMemoryNeeded(32_768, 16, 16, kv_per_tok, stored, scored, 1536, 4608, 16, 4096, 32_768, 0, 0, .{}),
    );
}

test "prefillMemoryNeeded: fused hd-256 kernel drops the score bill, keeps KV + dequant" {
    const t = std.testing;
    // Default (sushi_attn_p256 active): the composed score scratch never exists,
    // so hd 256 bills exactly the hd-256 KV + mlp — the unfused bill minus the
    // 1.6384 GB x1.25 score term from the test above.
    attn_kernels.fused256_override = true;
    defer attn_kernels.fused256_override = null;
    try t.expectEqual(
        @as(u64, 32_854_507_520 + 671_088_640 - 2_048_000_000),
        prefillMemoryNeeded(100_000, 16, 8, 245760, 256, 256, 2816, 2112, 16, 512, 100_000, 0, 0, .{}),
    );
    // The quantized-KV dequant transient is a denseView property, NOT a score
    // property — it must survive the fused kernel (fires on every kv-quant
    // request regardless of head_dim).
    try t.expectEqual(
        @as(u64, 11_798_507_520 + 671_088_640 - 2_048_000_000),
        prefillMemoryNeeded(100_000, 16, 8, 245760, 256, 256, 2816, 2112, 4, 512, 100_000, 0, 0, .{}),
    );
}

test "resolvePrefillChunk: a squeezed box steps the chunk down; a roomy one keeps it" {
    // The reserve at the widest chunk must never claim more than a quarter of
    // what is left after the weights. Numbers are the measured 16 GB profile: Metal's working
    // set ~11.9 GiB, Mistral-7B-4bit 3.80 GiB resident. The hot-cache budget is no longer an
    // input; the former `2 * GiB` argument is the pinned-context KV bar and is 0 under auto-context.
    const t = std.testing;
    const GiB: u64 = 1024 * 1024 * 1024;
    var mistral = model_mod.ModelConfig{};
    mistral.num_hidden_layers = 32;
    mistral.num_attention_heads = 32;
    mistral.num_key_value_heads = 8;
    mistral.head_dim = 128;
    mistral.hidden_size = 4096;
    mistral.intermediate_size = 14336;
    mistral.intermediate_size_declared = true;
    mistral.quant_bits = 4;
    mistral.max_position_embeddings = 32768;

    const ceiling_16: u64 = 11_918_000_000;
    const weights: u64 = 4_080_000_000;
    const narrowed = resolvePrefillChunk(&mistral, 16, ceiling_16, weights, 0, 0);
    try t.expect(narrowed < 8192);
    try t.expect(narrowed >= 512);
    // It is the WIDEST rung that fits, not simply the floor.
    try t.expect(prefillTransientReserve(&mistral, 16, narrowed) <=
        (ceiling_16 - weights) / 4);
    for (PREFILL_CHUNK_LADDER) |c| {
        if (c <= narrowed) break;
        try t.expect(prefillTransientReserve(&mistral, 16, c) > (ceiling_16 - weights) / 4);
    }

    // Same model, a machine with room: nothing narrows.
    try t.expectEqual(@as(u32, 8192), resolvePrefillChunk(&mistral, 16, 95 * GiB, weights, 0, 0));

    // A model that barely fits gets the narrowest rung rather than a wide one
    // it cannot pay for — and never 0, which would read as "not pinned".
    try t.expectEqual(@as(u32, 512), resolvePrefillChunk(&mistral, 16, weights + 64 * 1024 * 1024, weights, 0, 0));

    // Every rung is a real ladder entry, descending, floored at generate's own
    // minimum — a chunk generate would refuse is a chunk the bill cannot model.
    var prev: u32 = std.math.maxInt(u32);
    for (PREFILL_CHUNK_LADDER) |c| {
        try t.expect(c < prev);
        try t.expect(c >= prefill_plan.PREFILL_CHUNK_FLOOR);
        prev = c;
    }
}

test "qsaMaskBytes: a qwen4_exp twin bills the QSA mask and steps a rung the same trunk without qwen4_exp keeps" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const t = std.testing;
    var twin = model_mod.ModelConfig{};
    twin.num_hidden_layers = 40;
    twin.num_attention_heads = 24;
    twin.num_key_value_heads = 2;
    twin.head_dim = 256;
    twin.hidden_size = 2560;
    twin.intermediate_size = 9216;
    twin.intermediate_size_declared = true;
    twin.quant_bits = 4;
    twin.max_position_embeddings = 262144;
    var q4 = twin;
    q4.indexer_budget = 2048;
    q4.indexer_n_heads = 4;
    q4.indexer_head_dim = 128;
    q4.indexer_compress_ratio = 4;
    try t.expectEqual(@as(u64, 0), qsaMaskBytes(&twin, 4096, 25000));
    // Decode/verify widths build the dense mask; prefill widths gather by
    // block and bill the bounded score sheet instead of 4 B x rows x keys.
    try t.expectEqual(@as(u64, 4 * 8 * 25000 * 5 / 4), qsaMaskBytes(&q4, 8, 25000));
    qwen4_qsa_gather.qsa_gather_override = true;
    defer qwen4_qsa_gather.qsa_gather_override = null;
    const saved_fused = qwen4_qsa_select.qsa_score_fused_override;
    const saved_nax = nax_gate.vqmm_nax_probe_override;
    defer {
        qwen4_qsa_select.qsa_score_fused_override = saved_fused;
        nax_gate.vqmm_nax_probe_override = saved_nax;
    }
    nax_gate.vqmm_nax_probe_override = false;
    qwen4_qsa_select.qsa_score_fused_override = true;
    const gathered_composed = qsaMaskBytes(&q4, 4096, 250_000);
    try t.expect(gathered_composed > 0);
    try t.expect(gathered_composed < 4 * 4096 * 250_000);
    try t.expectEqual(qwen4_qsa_attn.qsaPrefillTransientBytes(4, 4096, 250_000, 4, 128), gathered_composed);
    nax_gate.vqmm_nax_probe_override = null;
    qwen4_qsa_select.qsa_score_fused_override = true;
    const gathered_fused = qsaMaskBytes(&q4, 4096, 250_000);
    try t.expectEqual(qwen4_qsa_attn.qsaPrefillTransientBytes(4, 4096, 250_000, 4, 128), gathered_fused);
    if (qwen4_qsa_select.qsaScoreFusedActive()) {
        try t.expect(gathered_fused < gathered_composed);
    }
    qwen4_qsa_select.qsa_score_fused_override = false;
    nax_gate.vqmm_nax_probe_override = false;
    const gathered = qsaMaskBytes(&q4, 4096, 250_000);
    try t.expectEqual(gathered_composed, gathered);
    const twin_bill = prefillTransientReserve(&twin, 16, 8192);
    const q4_bill = prefillTransientReserve(&q4, 16, 8192);
    try t.expectEqual(twin_bill + qsaMaskBytes(&q4, 8192, 8192) + q4.qsaRingBytes() * 5 / 4, q4_bill);
    // A ceiling whose quarter-share sits between the two bills at 8192.
    const weights: u64 = 70_000_000_000;
    const ceiling = weights + (twin_bill + q4_bill) / 2 * 4;
    try t.expectEqual(@as(u32, 8192), resolvePrefillChunk(&twin, 16, ceiling, weights, 0, 0));
    try t.expectEqual(@as(u32, 4096), resolvePrefillChunk(&q4, 16, ceiling, weights, 0, 0));
}

test "resolvePrefillChunk: the sizer and the guard bill the chunk that was pinned" {
    // The whole point of pinning: `prefillTransientReserve` (sizer) and
    // `prefillMemoryNeeded` (guard) must move together with the resolved width.
    // A narrower pin has to produce a SMALLER bill for the same prompt, or the
    // step-down bought nothing; and generate must resolve to the same width or
    // the forward runs wider than the bill and the Metal OOM is uncatchable.
    const t = std.testing;
    var cfg = model_mod.ModelConfig{};
    cfg.num_hidden_layers = 32;
    cfg.num_attention_heads = 32;
    cfg.num_key_value_heads = 8;
    cfg.head_dim = 128;
    cfg.hidden_size = 4096;
    cfg.intermediate_size = 14336;
    cfg.intermediate_size_declared = true;

    const wide = prefillTransientReserve(&cfg, 16, 8192);
    const narrow = prefillTransientReserve(&cfg, 16, 512);
    try t.expect(narrow * 2 < wide);

    cfg.pinned_prefill_chunk = 512;
    try t.expectEqual(@as(usize, 512), prefill_plan.effectivePrefillChunk(
        cfg.prefillScoreHeadDim(),
        cfg.num_attention_heads,
        40_000,
        cfg.has_sliding_window,
        cfg.isMoe(),
        cfg.longCtxGated(),
        cfg.pinned_prefill_chunk,
    ));
    // Unpinned keeps the launch width — a model the sizer never ran on must
    // not silently get a narrower forward than the guard bills.
    const unpinned_prefill_chunk: usize = 0;
    try t.expectEqual(@as(usize, 8192), prefill_plan.effectivePrefillChunk(
        cfg.prefillScoreHeadDim(),
        cfg.num_attention_heads,
        40_000,
        cfg.has_sliding_window,
        cfg.isMoe(),
        cfg.longCtxGated(),
        unpinned_prefill_chunk,
    ));
    // An explicit --prefill-chunk outranks the pin in BOTH directions.
    settings_resolve.prefill_chunk.value = 2048;
    settings_resolve.prefill_chunk.source = .flag;
    defer {
        settings_resolve.prefill_chunk.value = 8192;
        settings_resolve.prefill_chunk.source = .default;
    }
    try t.expectEqual(@as(usize, 2048), prefill_plan.effectivePrefillChunk(
        cfg.prefillScoreHeadDim(),
        cfg.num_attention_heads,
        40_000,
        cfg.has_sliding_window,
        cfg.isMoe(),
        cfg.longCtxGated(),
        cfg.pinned_prefill_chunk,
    ));
}

test "prefillMemoryNeeded: a GDN trunk's prefill peaks stay covered" {
    // Peak GPU bytes above steady state for ONE request on a clean boot
    // (`--prefix-cache-entries 0 --no-mtp --no-drafter`, /props
    // peak_bytes minus active_bytes, M4 Max) — byte-identical across repeat
    // boots, so these are exact figures, not samples. An estimate BELOW a peak
    // we have actually seen ends in an uncatchable Metal OOM instead of a 400.
    // The rows are the measured peaks minus the exact conv_state pin the
    // compaction removed ((linear_layers - 5) x chunk x qkv x 2).
    const t = std.testing;
    attn_kernels.fused256_override = true;
    defer attn_kernels.fused256_override = null;

    // Hidden 5120, ffn 17408, 64 layers / 16 caching, hd 256,
    // 48 GatedDeltaNet layers x (2x16x128 + 48x128) elems x 2 B of stream.
    const q27_stream: u64 = 5 * (2 * 16 * 128 + 48 * 128) * 2;
    const q27_dq: u64 = 3 * 17408 * 5120 * 2;
    try t.expect(prefillMemoryNeeded(9320, 24, 4, 65536, 256, 256, 5120, 17408, 16, 2048, 9320, q27_stream, q27_dq, .{}) >= 2_174_449_280);
    try t.expect(prefillMemoryNeeded(9270, 24, 4, 65536, 256, 256, 5120, 17408, 16, 8192, 9270, q27_stream, q27_dq, .{}) >= 3_623_797_120);
    // Hidden 2560, ffn 9216, 32 layers / 8 caching, 24 GDN layers.
    const q4_stream: u64 = 5 * (2 * 16 * 128 + 32 * 128) * 2;
    const q4_dq: u64 = 3 * 9216 * 2560 * 2;
    try t.expect(prefillMemoryNeeded(9340, 16, 4, 32768, 256, 256, 2560, 9216, 16, 1024, 9340, q4_stream, q4_dq, .{}) >= 1_329_232_896);
    try t.expect(prefillMemoryNeeded(9270, 16, 4, 32768, 256, 256, 2560, 9216, 16, 8192, 9270, q4_stream, q4_dq, .{}) >= 2_559_863_168);
    // A MoE trunk: hidden 2816, top_k 8 x moe 704.
    const g4_stream: u64 = 4 * 8 * 2 * (2816 + 704) * 2;
    const g4_dq: u64 = 3 * 5632 * 2816 * 2;
    try t.expect(prefillMemoryNeeded(8204, 16, 8, 245760, 256, 256, 2816, 5632, 16, 4096, 8204, g4_stream, g4_dq, .{}) >= 3_960_000_000);
}

test "prefillMemoryNeeded: the new terms fire only where the measurement put them" {
    const t = std.testing;
    attn_kernels.fused256_override = true;
    defer attn_kernels.fused256_override = null;
    const q27 = .{ .seq = 9320, .h = 24, .kvh = 4, .kvpt = 65536, .hd = 256, .hidden = 5120, .ffn = 17408 };
    const stream: u64 = 48 * (2 * 16 * 128 + 48 * 128) * 2;
    const dq: u64 = 3 * 17408 * 5120 * 2;

    // The stream term is per CHUNK-TOKEN: it scales with the forward width and
    // not with the prompt, which is what the whole chunk-bounded model says.
    const at2k = prefillMemoryNeeded(q27.seq, q27.h, q27.kvh, q27.kvpt, q27.hd, q27.hd, q27.hidden, q27.ffn, 16, 2048, q27.seq, stream, 0, .{});
    const at8k = prefillMemoryNeeded(q27.seq, q27.h, q27.kvh, q27.kvpt, q27.hd, q27.hd, q27.hidden, q27.ffn, 16, 8192, q27.seq, stream, 0, .{});
    const flat2k = prefillMemoryNeeded(q27.seq, q27.h, q27.kvh, q27.kvpt, q27.hd, q27.hd, q27.hidden, q27.ffn, 16, 2048, q27.seq, 0, 0, .{});
    const mlp2k: u64 = 8 * 2048 * q27.ffn * 2;
    const floor: u64 = 512 * 1024 * 1024;
    // The stream arm wins on this arch — that IS the fix — so the bill is one
    // MLP envelope plus the streams, not the three-envelope floor.
    try t.expectEqual(@as(u64, (q27.seq * q27.kvpt + mlp2k + 2048 * stream + floor) * 5 / 4), at2k);
    try t.expectEqual(@as(u64, (q27.seq * q27.kvpt + 3 * mlp2k + floor) * 5 / 4), flat2k);
    try t.expect(at2k > flat2k);
    try t.expect(at8k > at2k);

    // A longer prompt at the same chunk adds KV only — the stream term does
    // not move, or it would be the per-token multiplier this class is about.
    const long = prefillMemoryNeeded(4 * q27.seq, q27.h, q27.kvh, q27.kvpt, q27.hd, q27.hd, q27.hidden, q27.ffn, 16, 2048, 4 * q27.seq, stream, 0, .{});
    try t.expectEqual(long - at2k, 3 * q27.seq * q27.kvpt * 5 / 4);

    // The dequant-weight term is chunk-INDEPENDENT (it is weights) but only
    // bills where the route can fire: at or above PREFILL_DQ_GEMM_MIN_M.
    const dq_wide = prefillMemoryNeeded(q27.seq, q27.h, q27.kvh, q27.kvpt, q27.hd, q27.hd, q27.hidden, q27.ffn, 16, 2048, q27.seq, 0, dq, .{});
    try t.expectEqual(dq_wide - flat2k, dq * 5 / 4);
    const narrow_with = prefillMemoryNeeded(q27.seq, q27.h, q27.kvh, q27.kvpt, q27.hd, q27.hd, q27.hidden, q27.ffn, 16, 1024, q27.seq, 0, dq, .{});
    const narrow_without = prefillMemoryNeeded(q27.seq, q27.h, q27.kvh, q27.kvpt, q27.hd, q27.hd, q27.hidden, q27.ffn, 16, 1024, q27.seq, 0, 0, .{});
    try t.expectEqual(narrow_with, narrow_without);
    // A dense checkpoint passes 0 and is billed nothing for it.
    try t.expectEqual(flat2k, prefillMemoryNeeded(q27.seq, q27.h, q27.kvh, q27.kvpt, q27.hd, q27.hd, q27.hidden, q27.ffn, 16, 2048, q27.seq, 0, 0, .{}));

    // An arch with neither keeps the 3-envelope bill it had, plus the floor —
    // no plain-attention arch's admission tightens by more than the floor.
    const dense_only = prefillMemoryNeeded(9827, 32, 2, 53248, 128, 128, 6656, 19968, 16, 2048, 9827, 0, 0, .{});
    const pre_fix: u64 = (9827 * 53248 + 3 * 8 * 2048 * 19968 * 2) * 5 / 4;
    try t.expectEqual(dense_only - pre_fix, 512 * 1024 * 1024 * 5 / 4);
}

test "prefillStreamBytesPerToken: keyed on the arch's own geometry, zero for plain attention" {
    const t = std.testing;
    // 64 layers, every 4th full — 48 GatedDeltaNet layers, each
    // holding a chunk-wide q/k/v stream (2 x 16 x 128 key + 48 x 128 value).
    var q27 = model_mod.ModelConfig{ .model_type = "not_a_served_arch" };
    q27.num_hidden_layers = 64;
    q27.full_attention_interval = 4;
    q27.linear_num_key_heads = 16;
    q27.linear_num_value_heads = 48;
    try t.expectEqual(@as(u64, 5 * (2 * 16 * 128 + 48 * 128) * 2), prefillStreamBytesPerToken(&q27));

    var q4 = model_mod.ModelConfig{ .model_type = "not_a_served_arch" };
    q4.num_hidden_layers = 32;
    q4.full_attention_interval = 4;
    q4.linear_num_key_heads = 16;
    q4.linear_num_value_heads = 32;
    try t.expectEqual(@as(u64, 5 * (2 * 16 * 128 + 32 * 128) * 2), prefillStreamBytesPerToken(&q4));

    // No linear layers, but a MoE prefill gathers a top_k-fold
    // copy of the hidden stream beside the expert rows, for the 4 layers the
    // MoE eval cadence lets coexist.
    var g4 = model_mod.ModelConfig{ .model_type = "not_a_served_arch" };
    g4.num_hidden_layers = 30;
    g4.hidden_size = 2816;
    g4.num_experts = 128;
    g4.num_experts_per_tok = 8;
    g4.moe_intermediate_size = 704;
    try t.expectEqual(@as(u64, 4 * 8 * 2 * (2816 + 704) * 2), prefillStreamBytesPerToken(&g4));

    // A plain attention arch declares neither and is
    // billed exactly what it was billed before this term existed.
    var dense = model_mod.ModelConfig{ .model_type = "not_a_served_arch" };
    dense.num_hidden_layers = 52;
    dense.hidden_size = 6656;
    try t.expectEqual(@as(u64, 0), prefillStreamBytesPerToken(&dense));

    // Non-attention layers without linear head geometry are NOT
    // linear-attention layers: the term must not invent a stream for them.
    var conv = model_mod.ModelConfig{ .model_type = "not_a_served_arch" };
    conv.num_hidden_layers = 30;
    conv.full_attention_interval = 4;
    conv.hidden_size = 2048;
    try t.expectEqual(@as(u64, 0), prefillStreamBytesPerToken(&conv));
}

test "prefillDequantWeightBytes: affine-quantized weights only, and it reads the route's kill switch" {
    const t = std.testing;
    var cfg = model_mod.ModelConfig{ .model_type = "not_a_served_arch" };
    cfg.hidden_size = 6656;
    cfg.intermediate_size = 19968;
    cfg.intermediate_size_declared = true;
    cfg.quant_bits = 4;
    cfg.quant_mode = .affine;
    try t.expectEqual(@as(u64, 3 * 19968 * 6656 * 2), prefillDequantWeightBytes(&cfg));

    // Dense bf16: the route never runs, and the measurement agrees — a bf16
    // checkpoint's prefill peak sits below the quantized one by this term.
    var dense = cfg;
    dense.quant_bits = 0;
    try t.expectEqual(@as(u64, 0), prefillDequantWeightBytes(&dense));

    // Non-affine (mxfp4/mxfp8) stays in mlx_quantized_matmul — no dequantized
    // copy exists to bill.
    var mxfp8 = cfg;
    mxfp8.quant_mode = .mxfp8;
    try t.expectEqual(@as(u64, 0), prefillDequantWeightBytes(&mxfp8));

    // The estimator must not bill an allocation the kill switch removed.
    verify_moe.prefill_dq_gemm_override = false;
    defer verify_moe.prefill_dq_gemm_override = null;
    try t.expectEqual(@as(u64, 0), prefillDequantWeightBytes(&cfg));
}

test "prefillMemoryNeeded: a sparse-attention arch bills its KEY BOUND, not the whole prompt" {
    const t = std.testing;
    // DeepSeek-V4-Flash shape (live 2026-07-31): 64 heads, ONE latent kv head,
    // 43 layers, head_dim 512 — outside every fused kernel, so the score term
    // fires. But DSV4 attends to sliding_window(128) + index_topk(512) + sink
    // keys per query, NEVER the whole prompt (`tk = wk + n_sel`). Billing a dense 5806-wide key axis put a 5806-token prompt at
    // ~10.3 GB and 400-rejected it on a 1M-context model with 8.6 GB free.
    // The live request verbatim: 5806 tokens at kv-quant 8, chunk 5632, with
    // 8629 MB free. Pre-fix it billed 10277 MB — a dense 5806-wide key axis
    // PLUS the 15360 intermediate_size struct default — and 400'd.
    const available: u64 = 8629 * 1024 * 1024;
    const before = prefillMemoryNeeded(5806, 64, 1, 88064, 512, 512, 4096, 15360, 8, 5632, 5806, 0, 0, .{});
    try t.expectEqual(@as(u64, 10277 + 640), before / (1024 * 1024));
    try t.expect(before > available);
    // Fixed: 641 keys (window 128 + top-512 + sink) and the width the config
    // actually states (top_k 6 x moe_intermediate 2048 = 12288, what
    // prefillFfnWidth resolves for this checkpoint) — 4848 MB, admitted.
    const after = prefillMemoryNeeded(5806, 64, 1, 88064, 512, 512, 4096, 12288, 8, 5632, 641, 0, 0, .{});
    try t.expectEqual(@as(u64, 4848 + 640), after / (1024 * 1024));
    try t.expect(after < available);
    // The score term is the ONLY one the key bound may touch: same call with a
    // fused head_dim has no score scratch, so the bound changes nothing.
    attn_kernels.fused256_override = true;
    defer attn_kernels.fused256_override = null;
    try t.expectEqual(
        prefillMemoryNeeded(5806, 64, 1, 44032, 256, 256, 4096, 2048, 8, 5632, 5806, 0, 0, .{}),
        prefillMemoryNeeded(5806, 64, 1, 44032, 256, 256, 4096, 2048, 8, 5632, 641, 0, 0, .{}),
    );
}

test "resolvePrefillChunk: the 16 GB tier gets a working context instead of the 1024 floor" {
    // The reported bug, end to end. Mistral-7B-4bit on a 16 GB Mac: ~11.9 GiB
    // Metal working set, 3.80 GiB of weights (measured), the hot prefix cache
    // at its 2 GB default. At the launch chunk the transient reserve alone is
    // larger than what is left, so the sizer floored the advertised context at
    // 1024 and the guard 400'd a 10,348-token prompt for "needing" 9.18 GiB —
    // a prompt whose peak measures 2.391 GiB (M4 Max, one request per boot).
    const t = std.testing;
    const GiB: u64 = 1024 * 1024 * 1024;
    const ceiling: u64 = 11_918_000_000;
    const weights: u64 = 4_080_000_000;
    var cfg = model_mod.ModelConfig{};
    cfg.num_hidden_layers = 32;
    cfg.num_attention_heads = 32;
    cfg.num_key_value_heads = 8;
    cfg.head_dim = 128;
    cfg.hidden_size = 4096;
    cfg.intermediate_size = 14336;
    cfg.intermediate_size_declared = true;
    cfg.quant_bits = 4;
    cfg.max_position_embeddings = 32768;
    const per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), 16);

    const launch = safeContextForBudget(ceiling, weights, 2 * GiB +| prefillTransientReserve(&cfg, 16, 8192), per_tok, 32768);
    try t.expectEqual(@as(u32, 1024), launch); // the floor: nothing fits

    // The hot-cache budget is no longer an input; auto-context means the pinned-context bar is 0.
    const chunk = resolvePrefillChunk(&cfg, 16, ceiling, weights, 0, 0);
    const sized = safeContextForBudget(ceiling, weights, 2 * GiB +| prefillTransientReserve(&cfg, 16, chunk), per_tok, 32768);
    try t.expect(sized > 12_000);

    // And the guard now admits a real agent prompt at that context, billed
    // against the same chunk. 8.10 GiB is what a 16 GB Mac has free with only
    // the weights resident.
    const needed = prefillMemoryNeeded(
        10_348,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.kvBytesPerToken(),
        cfg.head_dim,
        cfg.prefillScoreHeadDim(),
        cfg.hidden_size,
        prefillFfnWidth(&cfg),
        16,
        chunk,
        cfg.prefillAttnKeys(10_348),
        prefillStreamBytesPerToken(&cfg),
        prefillDequantWeightBytes(&cfg),
        .{},
    );
    try t.expect(needed < ceiling - weights);
    // Still comfortably above the measured 2.391 GiB peak — narrowing the chunk
    // must not turn a conservative bill into an optimistic one.
    try t.expect(needed > 2_391_000_000);
}

test "prefillFfnWidth: a declared MoE width beats the struct default, which is only a fallback" {
    const t = std.testing;
    // DSV4's config ships NO intermediate_size at all, so the struct default
    // (15360) was billed as though declared. The width the config DOES state is
    // top_k(6) x moe_intermediate(2048) — a chunk gathers that many expert rows
    // per token, so billing the bare 2048 would UNDER-bill 3x, and under-
    // billing ends in an uncatchable Metal OOM rather than a 400.
    var moe = model_mod.ModelConfig{};
    moe.intermediate_size = 15360; // struct default, never seen in the JSON
    moe.intermediate_size_declared = false;
    moe.moe_intermediate_size = 2048;
    moe.num_experts_per_tok = 6;
    moe.shared_expert_intermediate_size = 0;
    try t.expectEqual(@as(u64, 12288), prefillFfnWidth(&moe));

    // The shared expert runs on every token too, so it adds rather than maxes.
    var shared = moe;
    shared.shared_expert_intermediate_size = 1024;
    try t.expectEqual(@as(u64, 13312), prefillFfnWidth(&shared));

    // A dense checkpoint is unaffected: no MoE width, declared ffn wins.
    var dense = model_mod.ModelConfig{};
    dense.intermediate_size = 13312;
    dense.intermediate_size_declared = true;
    dense.moe_intermediate_size = 0;
    dense.num_experts_per_tok = 0;
    try t.expectEqual(@as(u64, 13312), prefillFfnWidth(&dense));

    // A checkpoint that DECLARES a dense ffn keeps the conservative @max — a
    // hybrid with dense layers really does run the wider one.
    var declared = moe;
    declared.intermediate_size_declared = true;
    try t.expectEqual(@as(u64, 15360), prefillFfnWidth(&declared));

    // Declares nothing: the struct default is all we have, so it still wins.
    var bare = model_mod.ModelConfig{};
    bare.intermediate_size = 15360;
    bare.intermediate_size_declared = false;
    bare.moe_intermediate_size = 0;
    bare.shared_expert_intermediate_size = 0;
    try t.expectEqual(@as(u64, 15360), prefillFfnWidth(&bare));
}

test "the load-time bill prices the MTP head a served pack runs by default" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const t = std.testing;
    const saved_force = launch_state.server_config.default_force_mtp;
    defer launch_state.server_config.default_force_mtp = saved_force;
    launch_state.server_config.default_force_mtp = false;
    var cfg = fixtures.qwen4ExpOomConfig();
    cfg.mtp_override = null;
    const head = mtpHeadKvBytesPerToken(&cfg) + mtpHeadStateBytesPerToken(&cfg);
    try t.expect(head > 0);
    try t.expect(mtpHeadDefaultOn(&cfg));
    var off = cfg;
    off.mtp_override = false;
    try t.expect(!mtpHeadDefaultOn(&off));
    try t.expectEqual(sessionBytesPerToken(&off, 8) + head, sessionBytesPerToken(&cfg, 8));
    var streamed = cfg;
    streamed.expert_streaming = true;
    try t.expect(!mtpHeadDefaultOn(&streamed));
    // A streamed load that kept its head bills it like a resident one.
    streamed.stream_mtp_head = true;
    try t.expect(mtpHeadDefaultOn(&streamed));
    try t.expectEqual(sessionBytesPerToken(&off, 8) + head, sessionBytesPerToken(&streamed, 8));
}

test "the load-time bill drops the MTP head when Qwen's default round is DFlash2" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const t = std.testing;
    const saved_force = launch_state.server_config.default_force_mtp;
    defer launch_state.server_config.default_force_mtp = saved_force;
    launch_state.server_config.default_force_mtp = false;
    var cfg = fixtures.qwen4ExpOomConfig();
    cfg.mtp_override = null;
    const head = mtpHeadKvBytesPerToken(&cfg) + mtpHeadStateBytesPerToken(&cfg);
    const with_head = sessionBytesPerToken(&cfg, 8);
    var dflash = cfg;
    dflash.dflash_loaded = true;
    try t.expect(!mtpHeadDefaultOn(&dflash));
    try t.expectEqual(with_head - head, sessionBytesPerToken(&dflash, 8));
    // The auto-context bill reads the same default: no head terms at the rung it prices.
    const a = sushiContextBytesAt(&cfg, 100_000, 8).bytes;
    const b = sushiContextBytesAt(&dflash, 100_000, 8).bytes;
    try t.expect(b + 100_000 * head <= a + dflashRequestBytes(&dflash, 4096));
    try t.expect(b < a + dflashRequestBytes(&dflash, 8192));
    // A forced head still yields to the drafter by default, and `enable_drafter:false` bills it again.
    launch_state.server_config.default_force_mtp = true;
    try t.expectEqual(@as(u64, 0), prefillRequestTerms(&dflash, 100_000, 4096, 8, 4096, .{}).mtp_head_kv_bytes);
    try t.expect(prefillRequestTerms(&dflash, 100_000, 4096, 8, 4096, .{ .mtp_on = true }).mtp_head_kv_bytes > 0);
    try t.expect(prefillRequestTerms(&cfg, 100_000, 4096, 8, 4096, .{}).mtp_head_kv_bytes > 0);
}

test "qwen4ExpOomConfig does not leak qsa_score_fused_override" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const saved = qwen4_qsa_select.qsa_score_fused_override;
    defer qwen4_qsa_select.qsa_score_fused_override = saved;
    qwen4_qsa_select.qsa_score_fused_override = null;
    _ = fixtures.qwen4ExpOomConfig();
    try std.testing.expectEqual(@as(?bool, null), qwen4_qsa_select.qsa_score_fused_override);
}

test "the 458k prefill's two unbilled terms are billed: retained checkpoints and reserved capacity" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    // These pins price a cache with no SSD tier behind it: every checkpoint stays in RAM.
    const no_tier = TierGuard.pin(false, 0);
    defer no_tier.restore();
    const whole_rebuild = fixtures.qsaWholeRebuildGuard();
    defer whole_rebuild.deinit();
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    // A 458,832-token cold prefill was admitted against 26.6 GB of headroom and died mid-prefill.
    const cfg = fixtures.qwen4ExpOomConfig();
    const seq: u64 = 458_832;
    const chunk: u64 = 4096;

    // 1. SSM checkpoints: 36 GDN layers x (state + conv window), up to --ssm-checkpoint-max.
    const per_cp = cfg.ssmCheckpointBytes();
    // 36 x (3,145,728 + 61,440) B = 115.46 MB: the state is f32.
    try t.expect(per_cp > 115_000_000 and per_cp < 116_000_000);
    const saved_cp_max = launch_state.ssm_checkpoint_max;
    launch_state.ssm_checkpoint_max = 32;
    defer launch_state.ssm_checkpoint_max = saved_cp_max;
    const held = retainedSsmCheckpointBytes(&cfg, seq, 0, chunk);
    try t.expectEqual(@as(u64, launch_state.ssm_checkpoint_max) * per_cp, held);
    try t.expect(held > 1_800_000_000);
    // It saturates: 393k holds the same 32 as 458k.
    try t.expectEqual(held, retainedSsmCheckpointBytes(&cfg, 393_216, 0, chunk));
    // An arch with no linear layers is billed nothing.
    var dense = model_mod.ModelConfig{};
    dense.num_hidden_layers = 32;
    dense.num_attention_heads = 32;
    try t.expectEqual(@as(u64, 0), dense.ssmCheckpointBytes());
    try t.expectEqual(@as(u64, 0), retainedSsmCheckpointBytes(&dense, seq, 0, chunk));

    // 2. Reserved capacity: prompt + max_tokens + one chunk; the guard bills the headroom beyond
    // the prompt's own rows, not a second copy of the cache.
    const terms = prefillRequestTerms(&cfg, seq, 2048, 8, chunk, .{});
    const kv_per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), 8);
    try t.expectEqual((2048 + chunk) * kv_per_tok, terms.reserved_kv_bytes);
    try t.expect(terms.reserved_kv_bytes * 4 < seq * kv_per_tok); // headroom, not a copy
    // The indexer history is billed at one copy plus the score bank (`statePerTokenBilled`).
    try t.expectEqual(cfg.qsaHistoryBytesPerToken() + cfg.qsaScoreBankBytesPerToken(), statePerTokenBilled(&cfg));
    try t.expectEqual((seq + 2048 + chunk) * statePerTokenBilled(&cfg), terms.state_bytes);
    try t.expectEqual(held, terms.checkpoint_bytes);

    // 3. The bill moved by more than 3 GB, toward a 503 instead of a corpse.
    const with_terms = prefillMemoryNeeded(seq, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.kvBytesPerToken(), cfg.head_dim, cfg.prefillScoreHeadDim(), cfg.hidden_size, prefillFfnWidth(&cfg), 8, chunk, cfg.prefillAttnKeys(seq), prefillStreamBytesPerToken(&cfg), prefillDequantWeightBytes(&cfg), terms);
    const without = prefillMemoryNeeded(seq, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.kvBytesPerToken(), cfg.head_dim, cfg.prefillScoreHeadDim(), cfg.hidden_size, prefillFfnWidth(&cfg), 8, chunk, cfg.prefillAttnKeys(seq), prefillStreamBytesPerToken(&cfg), prefillDequantWeightBytes(&cfg), .{});
    try t.expect(with_terms > without + 3_000_000_000);
    // The sentinel runs through the same path: an omitted max_tokens must not move the bill past the context.
    const sentinel: u64 = request_policy.omittedMaxTokensDefault(&cfg);
    const with_sentinel = prefillRequestTerms(&cfg, seq, sentinel, 8, chunk, .{});
    const with_explicit = prefillRequestTerms(&cfg, seq, 2048, 8, chunk, .{});
    try t.expect(with_sentinel.reserved_kv_bytes < 8 * 1024 * 1024 * 1024);
    try t.expect(with_sentinel.state_bytes < 4 * with_explicit.state_bytes);
    try t.expectEqual(with_explicit.checkpoint_bytes, with_sentinel.checkpoint_bytes);

    // A short prompt is untouched.
    const short = prefillRequestTerms(&cfg, 4096, 2048, 8, chunk, .{});
    try t.expectEqual(@as(u64, 0), short.reserved_kv_bytes);
    try t.expectEqual(4096 * statePerTokenBilled(&cfg), short.state_bytes);
}

test "the guard credits only PROVABLY reclaimable cache bytes, never the entry a restore will pin" {
    const t = std.testing;
    // The guard admitted on `available + evictable >= needed`, but the evictable 1,564 MB was
    // the one entry this prompt then restored from; the provable credit is residency minus the largest entry.
    const MB: u64 = 1024 * 1024;
    var bill = AdmissionBill{ .needed = 37_394 * MB, .available = 37_130 * MB, .evictable = 1_564 * MB, .reclaimable = 0 };
    try t.expect(!bill.fits());
    try t.expect(!bill.fitsAfterEviction()); // it DID, before this fix
    bill.reclaimable = 1_564 * MB;
    try t.expect(bill.fitsAfterEviction());
}

test "the QSA history is billed at the one copy a slot HOLDS, plus the score bank" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const t = std.testing;
    const cfg = fixtures.qwen4ExpOomConfig();
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    const one = cfg.qsaHistoryBytesPerToken();
    const bank = cfg.qsaScoreBankBytesPerToken();
    try t.expect(one > 0);
    try t.expect(bank > 0);
    try t.expectEqual(one + bank, statePerTokenBilled(&cfg));

    const seq: u64 = 458_832;
    const terms = prefillRequestTerms(&cfg, seq, 2048, 8, 4096, .{});
    try t.expectEqual((seq + 2048 + 4096) * (one + bank), terms.state_bytes);

    var dense = model_mod.ModelConfig{};
    dense.num_hidden_layers = 32;
    dense.num_attention_heads = 32;
    try t.expectEqual(@as(u64, 0), statePerTokenBilled(&dense));
}

test "ssdFirstPrefixCacheMem: the active session's KV is billed ONCE and is the floor" {
    const t = std.testing;
    const GB: u64 = 1 << 30;
    // The 1M qwen4 case: 107.5 GB ceiling, ~70 GB weights, ~24 GB of 1M-ctx KV, ~3 GB reserve.
    const ceiling: u64 = 107 * GB;
    const weights: u64 = 70 * GB;
    const ctx_kv: u64 = 24 * GB;
    const transient: u64 = 3 * GB;

    // RAM-first clamps the cache to the leftover: less than half an entry.
    const ram_first = clampedPrefixCacheMem(0, ceiling, weights, ctx_kv, transient);
    try t.expect(ram_first < ctx_kv);

    // SSD-first floors at one entry and hands `--prefix-cache-mem` to idle entries on top.
    try t.expectEqual(ctx_kv, ssdFirstPrefixCacheMem(0, ceiling, weights, ctx_kv, transient));
    try t.expectEqual(ctx_kv + 4 * GB, ssdFirstPrefixCacheMem(4 * GB, ceiling, weights, ctx_kv, transient));
    const huge = ssdFirstPrefixCacheMem(500 * GB, ceiling, weights, ctx_kv, transient);
    try t.expectEqual(ctx_kv + (ceiling - weights - ctx_kv - transient), huge);
    try t.expectEqual(ctx_kv, ssdFirstPrefixCacheMem(8 * GB, 80 * GB, weights, ctx_kv, transient));
    try t.expect(ssdFirstPrefixCacheMem(0, 0, 0, 0, 0) >= 1);
}

test "adaptivePrefillWidth: a step-down is immediate, by as many rungs as it takes, and never 0" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // The down direction does not wait for a trend or a margin.
    const t = std.testing;
    const cfg = fixtures.qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const seq: u64 = 1_048_576;
    const kv: u64 = seq;
    const cap = adaptCapFor(&cfg, seq);

    {
        var st: mtp_plan.AdaptiveWidthState = .{};
        const h = prefillChunkCost(&cfg, kv_bits, 2048, kv);
        try t.expect(h < prefillChunkCost(&cfg, kv_bits, cap, kv));
        try t.expectEqual(@as(u32, 2048), adaptivePrefillWidth(&cfg, kv_bits, kv, h, cap, cap, &st));
        try t.expect(st.ratcheted);
    }
    // Several rungs in one call.
    {
        var st: mtp_plan.AdaptiveWidthState = .{};
        const h = prefillChunkCost(&cfg, kv_bits, 512, kv);
        try t.expectEqual(@as(u32, 512), adaptivePrefillWidth(&cfg, kv_bits, kv, h, cap, cap, &st));
    }
    // No headroom at all: the ladder floor, never 0 (the loop reads 0 as "no change").
    {
        var st: mtp_plan.AdaptiveWidthState = .{};
        const got = adaptivePrefillWidth(&cfg, kv_bits, kv, 0, cap, cap, &st);
        try t.expectEqual(@as(u32, @intCast(prefill_plan.PREFILL_CHUNK_FLOOR)), got);
        try t.expect(got != 0);
    }
    {
        var st: mtp_plan.AdaptiveWidthState = .{};
        const floor: u32 = @intCast(prefill_plan.PREFILL_CHUNK_FLOOR);
        try t.expectEqual(floor, adaptivePrefillWidth(&cfg, kv_bits, kv, 0, floor, cap, &st));
    }
}

test "adaptivePrefillWidth: HOLD is margin 1.0 — the admitted width is not second-guessed" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // `prefillMemoryNeeded` already carries its own 5/4; a second 1.25 on the hold stepped the
    // 768k prompt down from the width admission admitted.
    const t = std.testing;
    const cfg = fixtures.qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const kv: u64 = 786_432;
    const cap = adaptCapFor(&cfg, 786_432);
    const cost_2k = prefillChunkCost(&cfg, kv_bits, 2048, kv);

    var st: mtp_plan.AdaptiveWidthState = .{};
    try t.expectEqual(@as(u32, 2048), adaptivePrefillWidth(&cfg, kv_bits, kv, cost_2k, 2048, cap, &st));
    try t.expect(!st.ratcheted);
    try t.expect(cost_2k * 5 / 4 > cost_2k);
    var st2: mtp_plan.AdaptiveWidthState = .{};
    try t.expectEqual(@as(u32, 2048), adaptivePrefillWidth(&cfg, kv_bits, kv, cost_2k * 5 / 4 - 1, 2048, cap, &st2));
    var st3: mtp_plan.AdaptiveWidthState = .{};
    try t.expect(adaptivePrefillWidth(&cfg, kv_bits, kv, cost_2k - 1, 2048, cap, &st3) < 2048);
}

test "adaptivePrefillWidth: a widen costs 1.25x AND two consecutive supporting probes" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // Holding is free, growing is a bet: two consecutive probes.
    const t = std.testing;
    const cfg = fixtures.qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const kv: u64 = 524_288;
    const cap = adaptCapFor(&cfg, 524_288);
    // Starts at half the cap so one widen step lands exactly on the cap.
    const from: u32 = cap / 2;
    const cost_up = prefillChunkCost(&cfg, kv_bits, cap, kv);

    var st: mtp_plan.AdaptiveWidthState = .{};
    try t.expectEqual(from, adaptivePrefillWidth(&cfg, kv_bits, kv, cost_up * 4, from, cap, &st));
    try t.expectEqual(@as(u8, 1), st.supporting);
    try t.expectEqual(cap, adaptivePrefillWidth(&cfg, kv_bits, kv, cost_up * 4, from, cap, &st));
    try t.expectEqual(@as(u8, 0), st.supporting); // and the count restarts at the new width

    // Room for the wider chunk's bill but not its margin: never.
    var st2: mtp_plan.AdaptiveWidthState = .{};
    const between = cost_up * 5 / 4 - 1;
    try t.expect(between >= cost_up);
    var i: usize = 0;
    while (i < 8) : (i += 1) {
        try t.expectEqual(from, adaptivePrefillWidth(&cfg, kv_bits, kv, between, from, cap, &st2));
    }

    // A single unsupporting probe resets the run.
    var st3: mtp_plan.AdaptiveWidthState = .{};
    try t.expectEqual(from, adaptivePrefillWidth(&cfg, kv_bits, kv, cost_up * 4, from, cap, &st3));
    try t.expectEqual(from, adaptivePrefillWidth(&cfg, kv_bits, kv, between, from, cap, &st3));
    try t.expectEqual(@as(u8, 0), st3.supporting);
    try t.expectEqual(from, adaptivePrefillWidth(&cfg, kv_bits, kv, cost_up * 4, from, cap, &st3));
}

test "the tail-merge bound scales ONLY where the per-chunk adaptive width is live" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // Every arch but an adaptive-width qwen4_exp keeps the flat `TAIL_MERGE_MAX` at every chunk width.
    const t = std.testing;

    var other = fixtures.qwen4RequestTestConfig();
    other.model_type = "not_a_served_arch";
    try t.expect(!other.perRequestPrefillChunk());
    try t.expect(!adaptivePrefillChunkEnabled(&other));
    const off = adaptivePrefillChunkEnabled(&other);
    try t.expectEqual(prefill_plan.TAIL_MERGE_MAX, prefill_plan.tailMergeMaxFor(512, off));
    try t.expectEqual(prefill_plan.TAIL_MERGE_MAX, prefill_plan.tailMergeMaxFor(2048, off));
    try t.expectEqual(@as(usize, 812), prefill_plan.nextChunkEnd(0, 812, 512, false, 0, 0, off));

    const cfg = fixtures.qwen4RequestTestConfig();
    try t.expect(cfg.perRequestPrefillChunk());
    per_request_chunk_override = true;
    defer per_request_chunk_override = null;
    adaptive_chunk_override = true;
    defer adaptive_chunk_override = null;
    const unpinned = explicitPrefillChunk() == 0;
    try t.expectEqual(unpinned, adaptivePrefillChunkEnabled(&cfg));
    try t.expectEqual(@as(usize, 64), prefill_plan.tailMergeMaxFor(512, true));
    try t.expectEqual(@as(usize, 512), prefill_plan.nextChunkEnd(0, 812, 512, false, 0, 0, true));
    // An operator pin turns the scaling back off.
    adaptive_chunk_override = false;
    try t.expect(!adaptivePrefillChunkEnabled(&cfg));
    try t.expectEqual(prefill_plan.TAIL_MERGE_MAX, prefill_plan.tailMergeMaxFor(512, adaptivePrefillChunkEnabled(&cfg)));
}

test "adaptivePrefillWidth: the ratchet is one-way, and the cap is a ceiling" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    const cfg = fixtures.qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const kv: u64 = 1_048_576;
    const cap = adaptCapFor(&cfg, 1_048_576);

    // Step down once, then hand it the whole machine: it stays (a re-widen would be a metronome).
    var st: mtp_plan.AdaptiveWidthState = .{};
    const narrowed = adaptivePrefillWidth(&cfg, kv_bits, kv, prefillChunkCost(&cfg, kv_bits, 1024, kv), cap, cap, &st);
    try t.expect(narrowed < cap);
    try t.expect(st.ratcheted);
    var i: usize = 0;
    while (i < 8) : (i += 1) {
        try t.expectEqual(narrowed, adaptivePrefillWidth(&cfg, kv_bits, kv, std.math.maxInt(u64) / 2, narrowed, cap, &st));
    }

    // Nothing above the arch cap, ever.
    var st2: mtp_plan.AdaptiveWidthState = .{};
    i = 0;
    while (i < 8) : (i += 1) {
        try t.expectEqual(cap, adaptivePrefillWidth(&cfg, kv_bits, kv, std.math.maxInt(u64) / 2, cap, cap, &st2));
    }
    // A cap of 0 (the unchunked vision arm) is "do not touch the width".
    var st3: mtp_plan.AdaptiveWidthState = .{};
    try t.expectEqual(@as(u32, 2048), adaptivePrefillWidth(&cfg, kv_bits, kv, 0, 2048, 0, &st3));
}

test "adaptivePrefillWidth: a 1M prompt walks DOWN as the box fills, and a small one never moves" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // Sweep: the width is the widest rung that fits, only narrows within one prefill, and is always forwardable.
    const t = std.testing;
    const cfg = fixtures.qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const kv: u64 = 1_048_576;
    const cap = adaptCapFor(&cfg, 1_048_576);

    var st: mtp_plan.AdaptiveWidthState = .{};
    var width: u32 = cap;
    var seen_narrower: usize = 0;
    var headroom: u64 = 50 * 1024 * MiB; // the wired-limit box, quiet
    var step: usize = 0;
    while (step < 64) : (step += 1) {
        const next = adaptivePrefillWidth(&cfg, kv_bits, kv, headroom, width, cap, &st);
        try t.expect(next <= width); // never widens after the first step-down
        try t.expect(next >= @as(u32, @intCast(prefill_plan.PREFILL_CHUNK_FLOOR)));
        try t.expect(next <= cap);
        try t.expect(prefillChunkCost(&cfg, kv_bits, next, kv) <= headroom or
            next == @as(u32, @intCast(prefill_plan.PREFILL_CHUNK_FLOOR)));
        if (next < width) seen_narrower += 1;
        width = next;
        headroom -|= 1024 * MiB; // a co-tenant, or the accreting state
    }
    try t.expect(seen_narrower >= 2);
    try t.expectEqual(@as(u32, @intCast(prefill_plan.PREFILL_CHUNK_FLOOR)), width);

    // A 32k prompt on the same box: the width never moves.
    var st2: mtp_plan.AdaptiveWidthState = .{};
    const cap32 = adaptCapFor(&cfg, 32_768);
    var w32: u32 = cap32;
    step = 0;
    while (step < 16) : (step += 1) {
        w32 = adaptivePrefillWidth(&cfg, kv_bits, 32_768, 50 * 1024 * MiB, w32, cap32, &st2);
        try t.expectEqual(cap32, w32);
    }
    try t.expect(!st2.ratcheted);
}

test "prefillHeadroomNow subtracts the SSD writer's staged HOST bytes" {
    // Staged host bytes come out of the same unified pool the widen bar reads.
    const t = std.testing;
    const GiB: u64 = 1 << 30;
    try t.expectEqual(90 * GiB, prefillHeadroomFrom(120 * GiB, 30 * GiB, 0));
    try t.expectEqual(89 * GiB, prefillHeadroomFrom(120 * GiB, 30 * GiB, GiB));
    // Saturating: a permit larger than the headroom refuses every widen.
    try t.expectEqual(@as(u64, 0), prefillHeadroomFrom(120 * GiB, 119 * GiB, 8 * GiB));
}

test "adaptivePrefillWidth: the widen prices the QSA sheet at the LIVE KV, not at one chunk" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const whole_rebuild = fixtures.qsaWholeRebuildGuard();
    defer whole_rebuild.deinit();
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // `prefillTransientReserve` bills the QSA sheet at `kv = chunk`; past the indexer budget the
    // sheet costs its whole budget, so a widen decided on that number is unguarded.
    const t = std.testing;
    const cfg = fixtures.qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const cap = adaptCapFor(&cfg, 786_432);
    const short_kv: u64 = 4096;
    const long_kv: u64 = 400_000;

    // The bill is strictly larger at the live KV.
    const at_short = prefillChunkCost(&cfg, kv_bits, cap, short_kv);
    const at_long = prefillChunkCost(&cfg, kv_bits, cap, long_kv);
    try t.expect(at_long > at_short);
    // `prefillTransientReserve` is exactly the `kv = chunk` case.
    try t.expectEqual(
        prefillTransientReserve(&cfg, kv_bits, cap),
        prefillTransientReserveAtKv(&cfg, kv_bits, cap, cap),
    );

    // Headroom that clears the short bill's 1.25 bar but not the long one: widen at 4k, refuse at 400k.
    const headroom: u64 = at_short * 5 / 4 + 1;
    try t.expect(headroom < at_long * 5 / 4);

    // Starts at half the cap so one widen step lands exactly on the cap.
    const from: u32 = cap / 2;
    var st_short: mtp_plan.AdaptiveWidthState = .{};
    try t.expectEqual(from, adaptivePrefillWidth(&cfg, kv_bits, short_kv, headroom, from, cap, &st_short));
    try t.expectEqual(cap, adaptivePrefillWidth(&cfg, kv_bits, short_kv, headroom, from, cap, &st_short));

    var st_long: mtp_plan.AdaptiveWidthState = .{};
    var i: usize = 0;
    while (i < 8) : (i += 1) {
        try t.expectEqual(from, adaptivePrefillWidth(&cfg, kv_bits, long_kv, headroom, from, cap, &st_long));
    }
    try t.expectEqual(@as(u8, 0), st_long.supporting);

    // The hold keeps its margin-1.0 semantics at the live KV.
    const hold = prefillChunkCost(&cfg, kv_bits, 2048, long_kv);
    var st_hold: mtp_plan.AdaptiveWidthState = .{};
    try t.expectEqual(@as(u32, 2048), adaptivePrefillWidth(&cfg, kv_bits, long_kv, hold, 2048, cap, &st_hold));
    var st_down: mtp_plan.AdaptiveWidthState = .{};
    try t.expect(adaptivePrefillWidth(&cfg, kv_bits, long_kv, hold - 1, 2048, cap, &st_down) < 2048);
}

test "adaptivePrefillChunkEnabled: the arch, both kill switches, and any pinned width" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    const cfg = fixtures.qwen4RequestTestConfig();
    var other = fixtures.qwen4RequestTestConfig();
    other.model_type = "not_a_served_arch";

    try t.expect(adaptivePrefillChunkEnabled(&cfg));
    // Another arch never adapts; one switch disables both.
    try t.expect(!adaptivePrefillChunkEnabled(&other));

    per_request_chunk_override = false;
    try t.expect(!adaptivePrefillChunkEnabled(&cfg));
    per_request_chunk_override = null;
    try t.expect(adaptivePrefillChunkEnabled(&cfg));

    adaptive_chunk_override = false;
    try t.expect(!adaptivePrefillChunkEnabled(&cfg));
    adaptive_chunk_override = null;

    // An operator who pinned a width pinned every forward.
    settings_resolve.prefill_chunk.value = 4096;
    settings_resolve.prefill_chunk.source = .flag;
    defer {
        settings_resolve.prefill_chunk.value = 8192;
        settings_resolve.prefill_chunk.source = .default;
    }
    try t.expect(!adaptivePrefillChunkEnabled(&cfg));
}

test "resolvedContextForLoad: an auto boot bills the session it will serve, not the placeholder" {
    // On an auto boot the effective-context accessor answered a 1024-token placeholder during the load.
    const t = std.testing;
    const MiB: u64 = 1 << 20;
    // This box: ~28,909 MiB after the weights, 20,736 B/token at 1M/kv8, ~3 GiB transient.
    const per_tok: u64 = 20_736;
    const active: u64 = 70_000 * MiB;
    const ceiling: u64 = active + 28_909 * MiB;
    const transient: u64 = 3 * 1024 * MiB;
    const cap: u32 = 1_048_576;

    // Auto boot, nothing pinned: the session is what the machine can serve. `cache_reserve = 0`
    // is a test isolation.
    const auto = resolvedContextForLoad(0, 0, ceiling, active, 0, transient, per_tok, cap, 0);
    try t.expect(auto > 100_000);
    try t.expect(auto <= cap);

    // The placeholder bills ~26 MB of session and hands the idle allowance almost the whole machine.
    const placeholder_kv: u64 = per_tok * 1024;
    const real_kv: u64 = per_tok * auto;
    try t.expect(placeholder_kv < 30 * MiB);
    try t.expect(real_kv > 100 * placeholder_kv);
    const ask: u64 = 24 * 1024 * MiB;
    const bogus = ssdFirstPrefixCacheMem(ask, ceiling, active, placeholder_kv, transient);
    const fixed = ssdFirstPrefixCacheMem(ask, ceiling, active, real_kv, transient);
    try t.expect(bogus -| placeholder_kv == ask);
    // Under the real session bill the same ask is clamped to what is left after the weights, one
    // session and the transient (the helper's own contract, not a hand-picked byte count).
    const idle_headroom: u64 = ceiling - (active + real_kv + transient);
    try t.expectEqual(real_kv + idle_headroom, fixed);
    try t.expect(fixed -| real_kv < ask);
    try t.expect(fixed -| real_kv < (bogus -| placeholder_kv) / 2);

    // An explicit --ctx-size wins outright, and a pinned context is used as-is.
    try t.expectEqual(@as(u32, 262_144), resolvedContextForLoad(262_144, 0, ceiling, active, 0, transient, per_tok, cap, 0));
    try t.expectEqual(@as(u32, 131_072), resolvedContextForLoad(0, 131_072, ceiling, active, 0, transient, per_tok, cap, 0));
    try t.expectEqual(cap, resolvedContextForLoad(0, 0, active + 900_000 * MiB, active, 0, transient, per_tok, cap, 0));
}

test "the advertised context does not move with the cache ask" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // Sizing the context against the ask collapsed it (60GB ask -> 870 tokens); sizing against the
    // resolved budget was a one-step loop. A constant reserve closes both.
    const t = std.testing;
    const cfg = fixtures.qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const active: u64 = 69_827 * MiB;
    const live_ceiling: u64 = 109_395 * MiB;
    const per_tok: u64 = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits) +| statePerTokenBilled(&cfg);
    const transient: u64 = prefillTransientReserve(&cfg, kv_bits, 4096);

    // `max_pos` is 0: the checkpoint cap is applied by the caller after the margin.
    const memory_ctx = safeContextForBudget(live_ceiling, active, CTX_SIZING_CACHE_RESERVE +| transient, per_tok, 0);
    try t.expect(memory_ctx > 900_000);

    // The two defect spellings.
    const vs_ask = safeContextForBudget(live_ceiling, active, 60 * 1024 * MiB +| transient, per_tok, 0);
    const vs_resolved = safeContextForBudget(live_ceiling, active, 48_673 * MiB +| transient, per_tok, 0);
    try t.expect(vs_ask <= 1024);
    try t.expect(vs_resolved <= 1024);

    // The pair agrees at the advertised number: both sides go through `autoContextFrom`.
    const advertised = autoContextFrom(memory_ctx, cfg.contextCap(), cfg.defaultContext());
    const clamp_ctx = resolvedContextForLoad(0, 0, live_ceiling, active, CTX_SIZING_CACHE_RESERVE, transient, per_tok, cfg.contextCap(), cfg.defaultContext());
    try t.expectEqual(advertised, clamp_ctx);
}

test "the published hot-cache budget is retired when the cache is dropped" {
    // The budget is per model, the global is process-wide.
    const t = std.testing;
    const tiers = TierGuard.pin(true, 0);
    defer tiers.restore();
    publishResolvedPrefixCacheMem(4096);
    try t.expectEqual(@as(u64, 4096), resolvedPrefixCacheMem());
    clearResolvedPrefixCacheMem();
    // Back to the ask, the honest pre-load answer.
    try t.expectEqual(launch_state.prefix_cache_mem_bytes, resolvedPrefixCacheMem());

    // A resolved zero is a real budget (`--prefix-cache-mem 0` under SSD-first); the sentinel is maxInt.
    publishResolvedPrefixCacheMem(0);
    try t.expectEqual(@as(u64, 0), resolvedPrefixCacheMem());
    clearResolvedPrefixCacheMem();
    try t.expectEqual(launch_state.prefix_cache_mem_bytes, resolvedPrefixCacheMem());
}

test "pinnedResidentBytes names the entry a restore would share, and only that" {
    const t = std.testing;
    // Residency the eviction pass cannot return: the connection thread's one signal that this prompt is warm.
    try t.expectEqual(@as(u64, 0), pinnedResidentBytes(.{ .evictable = 0, .reclaimable = 0 }));
    try t.expectEqual(@as(u64, 0), pinnedResidentBytes(.{ .evictable = 4096, .reclaimable = 4096 }));
    // The live 768k reading: 24,826 MB resident, 10,294 MB reclaimable.
    const mb: u64 = 1024 * 1024;
    try t.expectEqual(@as(u64, 14_532 * mb), pinnedResidentBytes(.{ .evictable = 24_826 * mb, .reclaimable = 10_294 * mb }));
    // Saturating: the two numbers are published from different sites.
    try t.expectEqual(@as(u64, 0), pinnedResidentBytes(.{ .evictable = 1, .reclaimable = 2 }));
}

test "prefillChunkCap: the chunk sizer's two long-context changes are qwen4_exp-only" {
    if (!archBuilt("qwen4") or !archBuilt("glm5")) return error.SkipZigTest;
    // Both sizer changes (dropping the ask, the ctx bar) moved the prefill width on every arch;
    // the ungated arithmetic is the previous `resolvePrefillChunk`.
    const t = std.testing;
    const GiB: u64 = 1 << 30;
    const ceiling: u64 = 120 * GiB;
    const weights: u64 = 69 * GiB;
    const ask: u64 = 10 * GiB;
    const ctx_kv: u64 = 20 * GiB;

    // Every other arch: the ask is spoken for, the context is not, no HOT_CACHE_FLOOR.
    for ([_][]const u8{ "glm5_next", "not_a_served_arch" }) |mt| {
        var cfg = fixtures.longCtxTestConfig();
        cfg.model_type = mt;
        const baseline: u64 = (ceiling - (weights + ask)) / PREFILL_RESERVE_BUDGET_SHARE;
        try t.expectEqual(baseline, prefillChunkCap(&cfg, ceiling, weights, ctx_kv, ask));
        try t.expectEqual(baseline, prefillChunkCap(&cfg, ceiling, weights, 48 * GiB, ask));
        try t.expect(prefillChunkCap(&cfg, ceiling, weights, ctx_kv, 0) > baseline);
    }

    // qwen4_exp: the ask is ignored, and the ctx bar binds.
    var q4 = fixtures.longCtxTestConfig();
    q4.model_type = "qwen4_exp";
    const serving: u64 = ceiling - weights;
    const share_bar: u64 = serving / PREFILL_RESERVE_BUDGET_SHARE;
    const want: u64 = @min(share_bar, (serving - ctx_kv) -| HOT_CACHE_FLOOR_BYTES);
    try t.expectEqual(want, prefillChunkCap(&q4, ceiling, weights, ctx_kv, ask));
    try t.expectEqual(
        prefillChunkCap(&q4, ceiling, weights, ctx_kv, 0),
        prefillChunkCap(&q4, ceiling, weights, ctx_kv, 24 * GiB),
    );
    try t.expect(prefillChunkCap(&q4, ceiling, weights, 48 * GiB, ask) < share_bar);
}

test "prefillRequestTerms: the admission bill's new terms are qwen4_exp-only" {
    if (!archBuilt("qwen4") or !archBuilt("glm5")) return error.SkipZigTest;
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // The allocator side of the reservation is gated (`prefill_plan.reservedPrefillTokens`); the
    // guard side was not, so off qwen4_exp the bill charged headroom never reserved.
    const t = std.testing;
    const seq: u64 = 200_000; // well past RESERVE_MIN_TOKENS
    const chunk: u64 = 4096;
    const max_tokens: u64 = 8192;
    const kv_bits: u64 = 8;

    // A 27B-class GDN pack: `ssmCheckpointBytes()` is non-zero here.
    var q35 = fixtures.qwen4ExpOomConfig();
    q35.model_type = "not_a_served_arch";
    try t.expect(q35.ssmCheckpointBytes() > 0);
    try t.expect(retainedSsmCheckpointBytes(&q35, seq, 0, chunk) > 0);

    // Supplies the donating case so the qwen4_exp assertion below is falsifiable; the ungated loop reads 0 through `.{}`.
    const warm = WarmPrefix{ .matched_tokens = 100_000, .capacity_tokens = 400_000, .will_donate = true };
    for ([_][]const u8{ "glm5_next", "not_a_served_arch" }) |mt| {
        var cfg = fixtures.qwen4ExpOomConfig();
        cfg.model_type = mt;
        const terms = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, warm);
        try t.expectEqual(@as(u64, 0), terms.reserved_kv_bytes);
        try t.expectEqual(@as(u64, 0), terms.checkpoint_bytes);
        try t.expectEqual(@as(u64, 0), terms.state_bytes);
        try t.expectEqual(@as(u64, 0), terms.shared_resident_bytes);
        // `.{}` is the identity: the whole bill is the previous expression.
        try t.expectEqual(
            prefillMemoryNeeded(seq, 24, 2, cfg.kvBytesPerToken(), 256, 256, 2560, 9216, kv_bits, chunk, cfg.prefillAttnKeys(seq), prefillStreamBytesPerToken(&cfg), prefillDequantWeightBytes(&cfg), .{}),
            prefillMemoryNeeded(seq, 24, 2, cfg.kvBytesPerToken(), 256, 256, 2560, 9216, kv_bits, chunk, cfg.prefillAttnKeys(seq), prefillStreamBytesPerToken(&cfg), prefillDequantWeightBytes(&cfg), terms),
        );
    }

    const q4 = fixtures.qwen4ExpOomConfig();
    const gated = prefillRequestTerms(&q4, seq, max_tokens, kv_bits, chunk, warm);
    try t.expect(gated.reserved_kv_bytes > 0);
    try t.expect(gated.checkpoint_bytes > 0);
    try t.expect(gated.state_bytes > 0);
    try t.expect(gated.shared_resident_bytes > 0);
}

test "prefillRequestTerms: a loaded DFlash2 assistant bills its window and chunk rows on qwen4 only" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const t = std.testing;
    var cfg = fixtures.qwen4ExpOomConfig();
    const seq: u64 = 200_000;
    const chunk: u64 = 2048;
    const before = prefillRequestTerms(&cfg, seq, 8192, 8, chunk, .{});
    try t.expectEqual(@as(u64, 0), before.dflash_bytes);
    cfg.dflash_loaded = true;
    cfg.dflash_window_bytes = 5 * 8 * 128 * 4 * 2304;
    cfg.dflash_capture_bytes_per_token = (5 * 2560 + 2 * 2560) * 2 + 5 * 8 * 128 * 4;
    const on = prefillRequestTerms(&cfg, seq, 8192, 8, chunk, .{});
    try t.expectEqual(cfg.dflash_window_bytes * 4 + cfg.dflash_capture_bytes_per_token * chunk + 64 * 1024 * 1024, on.dflash_bytes);
    const short = prefillRequestTerms(&cfg, 100, 8192, 8, chunk, .{});
    try t.expectEqual(cfg.dflash_window_bytes * 4 + cfg.dflash_capture_bytes_per_token * 100 + 64 * 1024 * 1024, short.dflash_bytes);
    var other = cfg;
    other.model_type = "not_a_served_arch";
    try t.expectEqual(@as(u64, 0), prefillRequestTerms(&other, seq, 8192, 8, chunk, .{}).dflash_bytes);
}

test "the 64 GB Sushi-STQ-2.75bpw recipe's bills at both GDN widths: 250000 context, kv8, MTP, 32000 new tokens" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const t = std.testing;
    const mb: u64 = 1024 * 1024;
    var cfg = try model_mod.parseConfigFromJson(t.allocator, @embedFile("fixtures/model-configs/qwen4_exp.json"));
    defer cfg.deinit(t.allocator);
    cfg.expert_layout = .exl3_k4;
    const saved_head = trunk.Trunk.mtp_head_kv_quant_flag;
    defer trunk.Trunk.mtp_head_kv_quant_flag = saved_head;
    trunk.Trunk.mtp_head_kv_quant_flag = true;
    const saved_ctx = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = saved_ctx;
    launch_state.server_config.max_context_size = 250_000;
    const saved_kv = launch_state.configured_kv_quant;
    defer launch_state.configured_kv_quant = saved_kv;
    launch_state.configured_kv_quant = trunk.KVQuantConfig.affine(8);
    const saved_mtp = launch_state.configured_mtp;
    defer launch_state.configured_mtp = saved_mtp;
    launch_state.configured_mtp = true;
    const warm = WarmPrefix{ .mtp_on = true };
    // The load bill's context term is the narrowest rung; at this pack's weights the 59,000 MB ceiling admits
    // the 2048 rung. With the SSD tier up (the default) the prefill holds 3 checkpoints, not the 16 it captures.
    try t.expectEqual(@as(u64, 7_844), prefillNeededAtChunk(&cfg, 250_000, 32_000, 8, 512, warm) / mb);
    try t.expectEqual(@as(u64, 9_337), prefillNeededAtChunk(&cfg, 250_000, 32_000, 8, 2048, warm) / mb);
    try t.expectEqual(@as(u64, 11_157), prefillNeededAtChunk(&cfg, 250_000, 32_000, 8, 4096, warm) / mb);
    cfg.gdn_state_f32 = false;
    try t.expectEqual(@as(u64, 7_641), prefillNeededAtChunk(&cfg, 250_000, 32_000, 8, 512, warm) / mb);
    try t.expectEqual(@as(u64, 9_135), prefillNeededAtChunk(&cfg, 250_000, 32_000, 8, 2048, warm) / mb);
    try t.expectEqual(@as(u64, 10_954), prefillNeededAtChunk(&cfg, 250_000, 32_000, 8, 4096, warm) / mb);
}

test "an SSD tier bounds the checkpoints a Qwen prefill holds, at both GDN widths" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const t = std.testing;
    var cfg = try model_mod.parseConfigFromJson(t.allocator, @embedFile("fixtures/model-configs/qwen4_exp.json"));
    defer cfg.deinit(t.allocator);
    const saved = .{ launch_state.ssm_checkpoint_stride, launch_state.ssm_checkpoint_max };
    defer {
        launch_state.ssm_checkpoint_stride = saved[0];
        launch_state.ssm_checkpoint_max = saved[1];
    }
    launch_state.ssm_checkpoint_stride = 256;
    launch_state.ssm_checkpoint_max = 16;
    const seq: u64 = 250_000;
    for ([_]bool{ true, false }) |f32_state| {
        cfg.gdn_state_f32 = f32_state;
        const per_cp = cfg.ssmCheckpointBytes();
        try t.expectEqual(if (f32_state) @as(u64, 115_458_048) else 58_834_944, per_cp);

        const no_tier = TierGuard.pin(false, 0);
        try t.expectEqual(@as(u64, 16) * per_cp, retainedSsmCheckpointBytes(&cfg, seq, 0, 512));
        no_tier.restore();

        // A write-through prefill holds two states; one it cannot serve is thinned to two and holds a third
        // while one is captured: the bill covers the larger.
        const ssd_only = TierGuard.pin(false, 20 << 30);
        try t.expectEqual(@as(u64, 3) * per_cp, retainedSsmCheckpointBytes(&cfg, seq, 0, 512));
        // A prompt under one stride captures its prompt-end snapshot alone and bills just that.
        try t.expectEqual(per_cp, retainedSsmCheckpointBytes(&cfg, 300, 0, 512));
        // A tier that did not come up bills as without one.
        cfg.prefix_cache_disk_declined = true;
        try t.expectEqual(@as(u64, 16) * per_cp, retainedSsmCheckpointBytes(&cfg, seq, 0, 512));
        cfg.prefix_cache_disk_declined = false;
        ssd_only.restore();

        // A RAM tier beside the SSD one keeps its entry's newest two out of the same states.
        const both = TierGuard.pin(true, 20 << 30);
        try t.expectEqual(@as(u64, 3) * per_cp, retainedSsmCheckpointBytes(&cfg, seq, 0, 512));
        both.restore();
    }
}

test "ctxSizingCacheReserve: the advertised context is unchanged on every other arch" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    // The constant changes the advertised `context_length` at every ask except the 2 GiB default,
    // and an agent CLI budgets against that number for a whole session.
    const t = std.testing;
    const GiB: u64 = 1 << 30;
    const saved = launch_state.prefix_cache_mem_bytes;
    defer launch_state.prefix_cache_mem_bytes = saved;
    const tiers = TierGuard.pin(true, 0);
    defer tiers.restore();

    var q4 = fixtures.longCtxTestConfig(); // declares qwen4_exp
    var other = fixtures.longCtxTestConfig();
    other.model_type = "not_a_served_arch";

    for ([_]u64{ 0, 2 * GiB, 10 * GiB, 60 * GiB }) |ask| {
        launch_state.prefix_cache_mem_bytes = ask;
        try t.expectEqual(CTX_SIZING_CACHE_RESERVE, ctxSizingCacheReserve(&q4));
        try t.expectEqual(ask, ctxSizingCacheReserve(&other));
    }

    // At the default ask the two agree: a default boot is unchanged on every arch.
    launch_state.prefix_cache_mem_bytes = CTX_SIZING_CACHE_RESERVE;
    try t.expectEqual(ctxSizingCacheReserve(&q4), ctxSizingCacheReserve(&other));

    // The reserve really moves the advertised context on the ungated arch.
    const kv_bits: u64 = 8;
    const per_tok = kvBytesPerTokenAtBits(other.kvBytesPerToken(), kv_bits) +| statePerTokenBilled(&other);
    const ceiling: u64 = 120 * GiB;
    const active: u64 = 69 * GiB;
    const transient = prefillTransientReserve(&other, kv_bits, 4096);
    const small = safeContextForBudget(ceiling, active, 2 * GiB +| transient, per_tok, 0);
    const big = safeContextForBudget(ceiling, active, 40 * GiB +| transient, per_tok, 0);
    try t.expect(big < small);
}

test "prefillAdmissionBill: the evict-to-admit arms are driven by the two credits alone" {
    // Both admit arms are driven entirely by the two credits, so zeroing them restores the
    // previous single `needed > available` refusal.
    const t = std.testing;
    const MB: u64 = 1024 * 1024;

    const gated = AdmissionBill{ .needed = 30 * MB, .available = 20 * MB, .evictable = 15 * MB, .reclaimable = 15 * MB };
    try t.expect(!gated.fits());
    try t.expect(gated.fitsAfterEviction()); // evict-and-admit
    try t.expectEqual(AdmissionVerdict.evict, admissionVerdict(gated));

    const deferral = AdmissionBill{ .needed = 30 * MB, .available = 20 * MB, .evictable = 15 * MB, .reclaimable = 0 };
    try t.expect(!deferral.fitsAfterEviction());
    try t.expect(pinnedResidentBytes(deferral) > 0); // warm deferral

    // Where the arch does not evict, the bill carries neither credit, and both arms go dead.
    const ungated = AdmissionBill{ .needed = 30 * MB, .available = 20 * MB };
    try t.expectEqual(ungated.fits(), ungated.fitsAfterEviction());
    try t.expectEqual(@as(u64, 0), pinnedResidentBytes(ungated));
    try t.expectEqual(AdmissionVerdict.refuse, admissionVerdict(ungated));
    // A bill that fits still admits on both arms.
    const roomy = AdmissionBill{ .needed = 10 * MB, .available = 20 * MB };
    try t.expectEqual(AdmissionVerdict.admit, admissionVerdict(roomy));
}

test "memoryRefusalMessage: the 400 names the hot cache only where the bill carries it" {
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // A non-qwen4 refusal read "the hot prefix cache holds ~0MB ..." on a box with a multi-GB resident cache.
    const t = std.testing;
    const MB: u64 = 1024 * 1024;

    // Ungated: the previous sentence, byte for byte.
    const bare = try memoryRefusalMessage(t.allocator, 12_000, 4096, 2048, .{ .needed = 30 * MB, .available = 20 * MB }, false);
    defer t.allocator.free(bare);
    try t.expectEqualStrings(
        "Prompt (12000 tokens) requires ~4096MB GPU memory but only ~2048MB available. Reduce prompt size or use a smaller model.",
        bare,
    );

    // Gated: the message quotes what the guard compared.
    const rich = try memoryRefusalMessage(t.allocator, 12_000, 4096, 2048, .{
        .needed = 30 * MB,
        .available = 20 * MB,
        .evictable = 1500 * MB,
        .reclaimable = 900 * MB,
    }, true);
    defer t.allocator.free(rich);
    try t.expect(std.mem.indexOf(u8, rich, "holds ~1500MB more") != null);
    try t.expect(std.mem.indexOf(u8, rich, "(~900MB") != null);

    // The gated arch reaches this arm with an empty cache too and keeps its bytes, zeroes and all.
    const q4_cold = try memoryRefusalMessage(t.allocator, 12_000, 4096, 2048, .{ .needed = 30 * MB, .available = 20 * MB }, true);
    defer t.allocator.free(q4_cold);
    try t.expectEqualStrings(
        "Prompt (12000 tokens) requires ~4096MB GPU memory but only ~2048MB is available and the hot prefix cache holds ~0MB more, all of which can be reclaimed (~0MB — the shortfall is elsewhere), which is not enough. Reduce prompt size, lower --ctx-size, or use a smaller model.",
        q4_cold,
    );
    const ungated_rich = try memoryRefusalMessage(t.allocator, 12_000, 4096, 2048, .{
        .needed = 30 * MB,
        .available = 20 * MB,
        .evictable = 1500 * MB,
        .reclaimable = 900 * MB,
    }, false);
    defer t.allocator.free(ungated_rich);
    try t.expectEqualStrings(bare, ungated_rich);
}

test "the live 364k session's admission numbers reproduce, to the megabyte" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    // These pins price a cache with no SSD tier behind it: every checkpoint stays in RAM.
    const no_tier = TierGuard.pin(false, 0);
    defer no_tier.restore();
    const whole_rebuild = fixtures.qsaWholeRebuildGuard();
    defer whole_rebuild.deinit();
    // The anchor: fed the deployed config and the incident's prompt lengths, the estimator prints
    // the incident's own numbers (Turn A admitted at 16,233 MB; Turn B refused at 13,664 MB).
    const t = std.testing;
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    const cfg = fixtures.qwen4ExpLive364kConfig();
    const mb: u64 = 1024 * 1024;
    const kv_bits: u64 = 8;
    // Turn A, width 2048, prompt 364,509, max_gen 421,923.
    try t.expectEqual(
        @as(u64, 13_269),
        prefillNeededAtChunk(&cfg, 364_509, 421_923, kv_bits, 2048, .{}) / mb,
    );
    try t.expectEqual(
        @as(u64, 11_854),
        prefillNeededAtChunk(&cfg, 368_208, 418_224, kv_bits, 512, .{}) / mb,
    );
}

test "the MLX buffer pool is inside `available` by construction, so a refusal cannot buy it back" {
    // `currentGpuMemoryCeiling` adds the pool to the footprint before the physical minimum, so on
    // the physical arm `available` is `cache_bytes + free_system` and a clear moves it by zero.
    const t = std.testing;
    const wired: u64 = 120 << 30;
    const active: u64 = 71_310 << 20;
    const pool: u64 = 8_268 << 20;
    const free_system: u64 = 12_546 << 20;
    const ceiling = physicalMemoryCeiling(wired, active + pool, free_system);
    try t.expect(ceiling < wired); // the physical arm, as on the box
    try t.expectEqual(pool + free_system, ceiling - active);
    const cleared = physicalMemoryCeiling(wired, active, free_system + pool);
    try t.expectEqual(ceiling, cleared);
}

test "a MiMo slot bills the decode rebuilds the other rows of its batched decode hold" {
    if (!archBuilt("mimo2")) return error.SkipZigTest;
    const t = std.testing;
    var cfg = fixtures.mimoV2BillConfig();
    // One global layer's dense K/V below the packed arms' floor, for every other row of a full group.
    const rebuild = DECODE_REBUILD_KEYS_MAX * cfg.layerKvBytes(0);
    try t.expect(rebuild > 0);
    try t.expectEqual(@as(u64, @intCast(slot_admission.batchGroupCap(&cfg) - 1)) * rebuild, batchedDecodeRowsBytes(&cfg));
    try t.expectEqual(@as(usize, 4), slot_admission.batchGroupCap(&cfg));
    // A streamed load decodes serial and bills nothing for it.
    cfg.expert_streaming = true;
    try t.expectEqual(@as(u64, 0), batchedDecodeRowsBytes(&cfg));
    var plain = fixtures.mimoV2BillConfig();
    plain.model_type = "not_a_served_arch";
    try t.expectEqual(@as(u64, 0), batchedDecodeRowsBytes(&plain));
}

test "the sliding ring is billed once per slot and staged per chunk token" {
    if (!archBuilt("mimo2")) return error.SkipZigTest;
    const t = std.testing;
    const cfg = fixtures.mimoV2BillConfig();
    // Once per slot, at the width the cache stores — the twin of the QSA ring,
    // which this arch does not have.
    try t.expectEqual(@as(u64, 0), cfg.qsaRingBytes());
    // The slot also holds its ring checkpoints until the commit takes them: where it restored
    // to, the message starts its prefill crossed, and its prompt end.
    const ring_and_cp = cfg.swaRingBytes() + prefix_cache_mod.SLOT_RING_CHECKPOINTS * cfg.swaRingCheckpointBytes();
    const rows = batchedDecodeRowsBytes(&cfg);
    try t.expectEqual(ring_and_cp + rows, slotRingBytes(&cfg, 16));
    try t.expectEqual(kvBytesPerTokenAtBits(ring_and_cp, 8) + rows, slotRingBytes(&cfg, 8));
    try t.expect(slotRingBytes(&cfg, 8) < slotRingBytes(&cfg, 16));
    // Per token of context only the two global layers count.
    try t.expectEqual(@as(u64, 2 * 2 * (192 + 128) * 2), sessionBytesPerToken(&cfg, 16));
    // The chunk the ring has not compacted yet rides the prefill envelope.
    try t.expect(prefillStreamBytesPerToken(&cfg) >= cfg.swaStreamBytesPerToken(5));
    // At a 48-layer geometry the cadence window is the bound: five sliding
    // layers stage a chunk at once, not forty.
    var full = cfg;
    full.num_hidden_layers = 48;
    var li: u32 = 0;
    while (li < 48) : (li += 1) full.layer_is_global[li] = (li % 6) == 0;
    try t.expectEqual(
        @as(u64, 5 * 3 * (192 + 128) * 2),
        full.swaStreamBytesPerToken(@as(u64, trunk.Trunk.PREFILL_EVAL_CADENCE_DEFAULT) + 1),
    );
    // The ring itself still holds every sliding layer.
    try t.expectEqual(full.swaRingTokens() * 40 * 3 * (192 + 128) * 2, full.swaRingBytes());
    // An arch that stores every layer full-length keeps a zero ring.
    var plain = cfg;
    plain.model_type = "not_a_served_arch";
    try t.expectEqual(@as(u64, 0), slotRingBytes(&plain, 16));
    try t.expectEqual(@as(u64, 0), prefillStreamBytesPerToken(&plain));
}

test "a QSA arch's kv-quant dequant scratch follows the widest forward that still gathers from a rebuild" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const t = std.testing;
    const cfg = fixtures.qwen4RequestTestConfig();
    const seq: u64 = 1024 * 1024;
    const layer_bytes: u64 = 2 * (256 + 256) * 2;
    const span: u64 = 2048 + 4 - 1 + trunk.FUSED256_MIN_Q_LEN - 1;
    const split_k: u64 = @intCast(qwen4_qsa_attn.QSA_ATTN_PACKED_MAX_S);
    // Split-K widths read the packed cache at every kv: only the dense span rebuilds.
    try t.expectEqual(span * layer_bytes, kvDequantScratchBytes(&cfg, seq, split_k));
    try t.expectEqual(span * layer_bytes, kvDequantScratchBytes(&cfg, 2 * seq, 16));
    // A wider forward rebuilds only while its rows would stage each packed row over 4 times.
    try t.expectEqual(512 * 2048 / 4 * layer_bytes, kvDequantScratchBytes(&cfg, seq, 512));
    try t.expectEqual(seq * layer_bytes, kvDequantScratchBytes(&cfg, seq, 8192));
    // A short prompt never rebuilds more than it stores.
    try t.expectEqual(@as(u64, 1000) * layer_bytes, kvDequantScratchBytes(&cfg, 1000, 16));
    // Either kill switch brings the whole-cache rebuild (the generic term) back.
    qwen4_qsa_attn.qsa_attn_kernel_override = false;
    try t.expectEqual(@as(u64, 0), kvDequantScratchBytes(&cfg, seq, 16));
    qwen4_qsa_attn.qsa_attn_kernel_override = null;
    qwen4_qsa_gather.qsa_gather_override = false;
    defer qwen4_qsa_gather.qsa_gather_override = null;
    try t.expectEqual(@as(u64, 0), kvDequantScratchBytes(&cfg, seq, 16));
}

test "a ringed arch's kv-quant dequant scratch is one layer's rebuild, not the model's" {
    if (!archBuilt("mimo2")) return error.SkipZigTest;
    const t = std.testing;
    const cfg = fixtures.mimoV2BillConfig();
    const seq: u64 = 512 * 1024;

    // Composed: the widest rebuild is a GLOBAL layer at the prompt length,
    // 2 kv heads of qk 192 + v 128. The sliding layers rebuild their ring.
    attn_kernels.fused256_override = false;
    defer attn_kernels.fused256_override = null;
    try t.expectEqual(seq * 2 * (192 + 128) * 2, kvDequantScratchBytes(&cfg, seq, 1024));
    // The generic expression this replaces reads the SLIDING head count and
    // doubles the qk width: 1.8x here, 2.4x at the shipped 4/8-head geometry.
    const generic: u64 = 2 * seq * cfg.num_key_value_heads * cfg.head_dim * 2;
    try t.expect(generic > kvDequantScratchBytes(&cfg, seq, 1024));

    // Fused: the global layers read the packed cache per dispatch, so their
    // rebuild stops scaling with the prompt at all.
    attn_kernels.fused256_override = true;
    const fused = kvDequantScratchBytes(&cfg, seq, 1024);
    try t.expectEqual(
        MOE_PREFILL_COEXIST * @as(u64, @intCast(trunk.PACKED_KV_SLICE_MAX)) * 2 * (192 + 128) * 2,
        fused,
    );
    try t.expectEqual(fused, kvDequantScratchBytes(&cfg, 2 * seq, 1024));
    attn_kernels.fused256_override = false;

    // A decode forward bills whatever the global-layer decode rebuilds whole: a floor's worth of keys.
    attn_kernels.fused256_override = true;
    try t.expectEqual(fused, kvDequantScratchBytes(&cfg, seq, 1));
    attn_kernels.fused256_override = false;

    // A short prompt never rebuilds more than it stores, and below the ring the
    // widest layer is a SLIDING one: it carries more KV heads than a global.
    try t.expectEqual(@as(u64, 256) * 3 * (192 + 128) * 2, kvDequantScratchBytes(&cfg, 256, 1024));

    // Every arch whose caching layers share one geometry keeps the generic term.
    var plain = cfg;
    plain.model_type = "not_a_served_arch";
    try t.expectEqual(@as(u64, 0), kvDequantScratchBytes(&plain, seq, 1024));

    // fp16 KV never rebuilds anything, whatever the arch.
    const dense_kv = prefillMemoryNeeded(seq, 4, 3, cfg.kvBytesPerToken(), 192, 192, 384, 1024, 16, 1024, seq, 0, 0, .{ .dequant_scratch_bytes = 1 << 30 });
    try t.expectEqual(
        dense_kv,
        prefillMemoryNeeded(seq, 4, 3, cfg.kvBytesPerToken(), 192, 192, 384, 1024, 16, 1024, seq, 0, 0, .{}),
    );
}

test "a ringed arch reserves its KV capacity once and bills the ring it holds" {
    if (!archBuilt("mimo2")) return error.SkipZigTest;
    const t = std.testing;
    const tiers = TierGuard.pin(true, 0);
    defer tiers.restore();
    const cfg = fixtures.mimoV2BillConfig();
    const seq: u64 = 512 * 1024;
    const chunk: u64 = 1024;

    // The allocator side: the prefill loop asks for the whole request's capacity
    // before the first chunk writes, so no buffer grows mid-prefill.
    try t.expect(prefill_plan.reservedPrefillTokens(&cfg, seq, 2048, chunk) > seq);
    try t.expectEqual(@as(u64, 0), prefill_plan.reservedPrefillTokens(&cfg, 1024, 2048, chunk));

    // The bill side: headroom and the per-slot ring, nothing else acquired.
    const terms = prefillRequestTerms(&cfg, seq, 2048, 8, chunk, .{});
    try t.expect(terms.reserved_kv_bytes > 0);
    try t.expectEqual(slotRingBytes(&cfg, 8), terms.qsa_ring_bytes);
    try t.expect(terms.qsa_ring_bytes > 0);
    try t.expectEqual(@as(u64, 0), terms.state_bytes);
    try t.expectEqual(@as(u64, 0), terms.checkpoint_bytes);
    try t.expectEqual(@as(u64, 0), terms.mtp_head_kv_bytes);
    // A cold prefill reserves; nothing coexists because nothing grows.
    try t.expectEqual(@as(u64, 0), terms.grow_coexist_bytes);

    // A warm append that outruns its restored capacity DOES grow, and the old
    // buffer it carries is spread over the layers that store the sequence —
    // nine of forty-eight here, not all of them.
    try t.expectEqual(@as(u32, 2), cfg.kvPerTokenLayerCount());
    try t.expectEqual(cfg.num_hidden_layers, cfg.attnCacheLayerCount());
    const warm = WarmPrefix{ .matched_tokens = seq / 2, .capacity_tokens = seq / 2, .will_donate = true };
    const grown = prefillRequestTerms(&cfg, seq, 2048, 8, chunk, warm);
    try t.expect(grown.grow_coexist_bytes > 0);
    const kv_per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), 8);
    try t.expectEqual(
        (seq / 2) * kv_per_tok / cfg.kvPerTokenLayerCount() * attnLayersPerEvalWindow(&cfg, trunk.Trunk.MOE_EVAL_EVERY_N_LAYERS),
        grown.grow_coexist_bytes,
    );

    // Every arch outside the gate keeps exactly the terms it had.
    var plain = cfg;
    plain.model_type = "not_a_served_arch";
    try t.expect(!plain.reservesKvCapacity());
    try t.expectEqual(PrefillRequestTerms{}, prefillRequestTerms(&plain, seq, 2048, 8, chunk, .{}));
    try t.expectEqual(@as(u64, 0), prefill_plan.reservedPrefillTokens(&plain, seq, 2048, chunk));
}

test "mimo_v2 prefill bill: fusing qk 192 drops the global and the band score sheets" {
    if (!archBuilt("mimo2")) return error.SkipZigTest;
    const t = std.testing;
    const tiers = TierGuard.pin(true, 0);
    defer tiers.restore();
    const cfg = fixtures.mimoV2BillConfig();
    const seq: u64 = 512 * 1024;
    const chunk: u64 = 1024;
    // Pin the arm, not the host: the guard and the dispatch read one switch.
    attn_kernels.fused256_override = true;
    defer attn_kernels.fused256_override = null;

    // One global layer's composed sheet at this prompt, the term the kernel removes.
    const global_sheet: u64 = @as(u64, cfg.num_attention_heads) * chunk * seq * 2;
    const fused_bill = prefillNeededAtChunk(&cfg, seq, 2048, 8, chunk, .{});
    try t.expect(fused_bill < global_sheet);

    // The sliding layers' band + sink attention runs in the same kernel: no sheet left.
    try t.expectEqual(@as(u64, 0), slidingBandScoreBytes(&cfg, chunk));

    // Kill switch: composed attention restores both sheets, and the bill with them. The
    // band sheet is prompt-length independent, `window + fwd - 1` keys wide, at the same
    // coexistence the other per-layer prefill transients are billed at.
    attn_kernels.fused256_override = false;
    try t.expectEqual(MOE_PREFILL_COEXIST * @as(u64, cfg.num_attention_heads) * chunk * (128 + chunk - 1) * 2, slidingBandScoreBytes(&cfg, chunk));
    try t.expect(prefillNeededAtChunk(&cfg, seq, 2048, 8, chunk, .{}) > fused_bill +| global_sheet);

    // An arch with no ring never acquires the band term.
    var plain = cfg;
    plain.model_type = "not_a_served_arch";
    attn_kernels.fused256_override = true;
    try t.expectEqual(@as(u64, 0), slidingBandScoreBytes(&plain, chunk));
}

test "a trellis trunk's prefill bill carries one decoded bank pair from the decoded path's width" {
    if (!archBuilt("mimo2")) return error.SkipZigTest;
    const t = std.testing;
    var plain = fixtures.mimoV2BillConfig();
    var stq = plain;
    stq.trunk_quant = sushi_exl3.format.Decode.tabled(.spec1, .w15);
    const min: u64 = @intCast(sushi_exl3.trunk_gemm.DECODED_MIN_ROWS);
    try t.expectEqual(sushi_exl3.trunk_gemm.DECODED_SCRATCH_BYTES, prefillTransientReserve(&stq, 8, 4096) - prefillTransientReserve(&plain, 8, 4096));
    try t.expectEqual(sushi_exl3.trunk_gemm.DECODED_SCRATCH_BYTES, trunkDecodeScratchBytes(&stq, min));
    try t.expectEqual(@as(u64, 0), trunkDecodeScratchBytes(&stq, min - 1));
    try t.expectEqual(@as(u64, 0), trunkDecodeScratchBytes(&plain, 4096));
}

test "mimo_v2 prefill bill carries the FP8 trunk's dequant scratch past the GEMV width" {
    if (!archBuilt("mimo2")) return error.SkipZigTest;
    const t = std.testing;
    var cfg = fixtures.mimoV2BillConfig();
    cfg.intermediate_size = 1536;
    cfg.expert_layout = .exl3_k4;
    try t.expect(cfg.usesMimoSourceTrunk());
    // The widest source FP8 linear is a sliding layer's QKV: 4*192 + 3*192 + 3*128 rows.
    const scratch: u64 = 2 * 1728 * 384 * 2;
    try t.expectEqual(scratch, fp8DequantScratchBytes(&cfg, 512));
    try t.expectEqual(@as(u64, 0), fp8DequantScratchBytes(&cfg, @intCast(fp8_block.gemv_max_rows)));
    // Both admission paths carry it, and a trunk with no FP8 linears does not.
    var dense = cfg;
    dense.expert_layout = .bf16_fused;
    try t.expectEqual(@as(u64, 0), fp8DequantScratchBytes(&dense, 512));
    try t.expectEqual(scratch, prefillNeededAtChunk(&cfg, 4096, 256, 8, 512, .{}) - prefillNeededAtChunk(&dense, 4096, 256, 8, 512, .{}));
    try t.expectEqual(scratch, prefillTransientReserveAtKv(&cfg, 8, 512, 4096) - prefillTransientReserveAtKv(&dense, 8, 512, 4096));
}

test "mimo_v2 with an SSD tier bills a restored global layer beside its first grow" {
    if (!archBuilt("mimo2")) return error.SkipZigTest;
    // An SSD restore installs each global layer at exactly the restored rows, and the first
    // append grows it while the restored buffer is still alive. MiMo's admission never sees
    // the restore, so with a disk tier it bills one eval window of that coexistence.
    const t = std.testing;
    const cfg = fixtures.mimoV2FlashBillConfig();
    const saved = launch_state.prefix_cache_disk_bytes;
    defer launch_state.prefix_cache_disk_bytes = saved;
    const seq: u64 = 400_000;
    const kv_bits: u64 = 8;
    const kv_per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits);
    const one_layer = seq * kv_per_tok / cfg.kvPerTokenLayerCount();
    launch_state.prefix_cache_disk_bytes = 0;
    try t.expectEqual(@as(u64, 0), prefillRequestTerms(&cfg, seq, 8192, kv_bits, 2048, .{}).grow_coexist_bytes);
    launch_state.prefix_cache_disk_bytes = 20 << 30;
    const window = attnLayersPerEvalWindow(&cfg, trunk.Trunk.MOE_EVAL_EVERY_N_LAYERS);
    try t.expect(window >= 1);
    try t.expectEqual(one_layer * window, prefillRequestTerms(&cfg, seq, 8192, kv_bits, 2048, .{}).grow_coexist_bytes);
}

test "mimo_v2 prefills at the widest width its request bill admits, not at the load-time pin" {
    if (!archBuilt("mimo2")) return error.SkipZigTest;
    const t = std.testing;
    attn_kernels.fused256_override = true;
    defer attn_kernels.fused256_override = null;
    const cfg = fixtures.mimoV2FlashBillConfig();
    const kv_bits: u64 = 8;
    const seq: u64 = 500_000;
    // What load-time sizing pins beside the default hot-cache ask.
    const pin: u32 = 512;
    const roomy: u64 = 200 << 30;

    try t.expect(cfg.perRequestPrefillChunk());
    try t.expectEqual(@as(u32, 2048), chooseRequestPrefillChunk(&cfg, 4096, 256, kv_bits, roomy, pin, 0, .{}));
    try t.expectEqual(@as(u32, 2048), chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, roomy, pin, 0, .{}));

    // Admitted at its exact bill; a byte under steps down the ladder, not back to the pin.
    const need = prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, 2048, .{});
    try t.expectEqual(@as(u32, 2048), chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, need, pin, 0, .{}));
    try t.expectEqual(@as(u32, 1024), chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, need - 1, pin, 0, .{}));

    // The forward starts at 2048 and steps down per chunk only where a chunk stops fitting.
    try t.expectEqual(@as(usize, 2048), prefill_plan.effectivePrefillChunk(cfg.prefillScoreHeadDim(), cfg.num_attention_heads, seq, cfg.has_sliding_window, cfg.isMoe(), cfg.longCtxGated(), 0));
    try t.expectEqual(explicitPrefillChunk() == 0, adaptivePrefillChunkEnabled(&cfg));

    // 2048 is a default: an explicit `--prefill-chunk 4096` raises the ladder's top rung, billed at that width.
    const saved_source = settings_resolve.prefill_chunk.source;
    const saved_chunk = settings_resolve.prefill_chunk.value;
    defer {
        settings_resolve.prefill_chunk.source = saved_source;
        settings_resolve.prefill_chunk.value = saved_chunk;
    }
    settings_resolve.prefill_chunk.source = .flag;
    settings_resolve.prefill_chunk.value = 4096;
    try t.expectEqual(@as(u32, 4096), chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, roomy, pin, 4096, .{}));
    const need4096 = prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, 4096, .{});
    try t.expectEqual(@as(u32, 2048), chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, need4096 - 1, pin, 4096, .{}));
}

test "mimo_v2: a 64k prompt keeps 2048 rows at every chunk where its box has room" {
    if (!archBuilt("mimo2")) return error.SkipZigTest;
    // The per-chunk estimator once stepped a 64k MiMo prompt down to 512 on a box that had room.
    const t = std.testing;
    attn_kernels.fused256_override = true;
    defer attn_kernels.fused256_override = null;
    const cfg = fixtures.mimoV2FlashBillConfig();
    const kv_bits: u64 = 8;
    const GiB: u64 = 1 << 30;
    // A 93 GiB resident MiMo under the 120,000 MB wired floor, beside its 64k request's reserved
    // KV, ring and MTP state: the live headroom each chunk boundary reads.
    const ceiling: u64 = 115_904 * 1024 * 1024;
    const resident: u64 = 93 * GiB;
    const seq: u64 = 65_536;
    try t.expectEqual(@as(u32, 2048), chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, ceiling - resident, 2048, 0, .{}));
    const terms = prefillRequestTerms(&cfg, seq, 2048, kv_bits, 2048, .{});
    const held = seq * kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits) + terms.reserved_kv_bytes + terms.state_bytes + terms.qsa_ring_bytes;
    const headroom = ceiling - resident - held;
    var st: mtp_plan.AdaptiveWidthState = .{};
    var pos: u64 = 0;
    while (pos < seq) : (pos += 2048) try t.expectEqual(@as(u32, 2048), adaptivePrefillWidth(&cfg, kv_bits, pos, headroom, 2048, 2048, &st));
}

test "the load line calls a per-request arch's pin the fallback, not the width it prefills at" {
    if (!archBuilt("mimo2")) return error.SkipZigTest;
    const t = std.testing;
    attn_kernels.fused256_override = true;
    defer attn_kernels.fused256_override = null;
    const saved = per_request_chunk_override;
    defer per_request_chunk_override = saved;
    const cfg = fixtures.mimoV2FlashBillConfig();
    var buf: [256]u8 = undefined;

    per_request_chunk_override = true;
    const line = prefillChunkLoadLine(&buf, &cfg, 1024, 8192).?;
    try t.expect(std.mem.indexOf(u8, line, "per request, up to 2048 at a short prompt") != null);
    try t.expect(std.mem.indexOf(u8, line, "fallback 1024") != null);

    // With the ladder off the pin is the width every request runs.
    per_request_chunk_override = false;
    try t.expectEqualStrings("Prefill chunk: 1024 tokens (memory-sized down from 8192; --prefill-chunk overrides)\n", prefillChunkLoadLine(&buf, &cfg, 1024, 8192).?);
    try t.expectEqual(@as(?[]const u8, null), prefillChunkLoadLine(&buf, &cfg, 8192, 8192));
}

test "mimo_v2 admission credits the hot cache: a gap the cache covers is an evict, not a refusal" {
    if (!archBuilt("qwen4") or !archBuilt("mimo2")) return error.SkipZigTest;
    const t = std.testing;
    const MB: u64 = 1024 * 1024;
    const cfg = fixtures.mimoV2FlashBillConfig();
    // The live refusal: 8924 MB needed, 8027 MB available, 1995 MB of hot cache this prompt does not pin.
    const bill = creditedAdmissionBill(&cfg, 8924 * MB, 8027 * MB, 1995 * MB, 1995 * MB, 4096);
    try t.expectEqual(1995 * MB, bill.evictionCredit());
    try t.expectEqual(AdmissionVerdict.evict, admissionVerdict(bill));
    // A credit short of the gap is still a refusal.
    const short = creditedAdmissionBill(&cfg, 8924 * MB, 8027 * MB, 1995 * MB, 800 * MB, 4096);
    try t.expectEqual(AdmissionVerdict.refuse, admissionVerdict(short));

    // Both halves read one predicate: the credits ride exactly where the inference thread evicts.
    var other = cfg;
    other.model_type = "not_a_served_arch";
    const q4 = fixtures.qwen4RequestTestConfig();
    for ([_]*const model_mod.ModelConfig{ &cfg, &other, &q4 }) |c| {
        const b = creditedAdmissionBill(c, 8924 * MB, 8027 * MB, 1995 * MB, 1995 * MB, 4096);
        try t.expectEqual(slot_admission.admissionPassArmed(c), b.evictionCredit() > 0);
    }
    try t.expectEqual(AdmissionVerdict.refuse, admissionVerdict(creditedAdmissionBill(&other, 8924 * MB, 8027 * MB, 1995 * MB, 1995 * MB, 4096)));
}

test "mimo_v2 warm bill credits only the restored global rows, never the ring" {
    if (!archBuilt("mimo2")) return error.SkipZigTest;
    const t = std.testing;
    const cfg = fixtures.mimoV2FlashBillConfig();
    const kv_bits: u64 = 8;
    const seq: u64 = 140_000;
    const matched: u64 = 120_000;
    const chunk: u64 = 4096;
    const kv_per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits);
    // `kvBytesPerToken` is the nine global layers; `residentCapacityTokens` skips the ringed ones.
    try t.expectEqual(@as(u32, 9), cfg.kvPerTokenLayerCount());

    const cold = prefillRequestTerms(&cfg, seq, 32_000, kv_bits, chunk, .{});
    const donated = prefillRequestTerms(&cfg, seq, 32_000, kv_bits, chunk, .{ .matched_tokens = matched, .capacity_tokens = matched, .will_donate = true });
    try t.expectEqual(matched * kv_per_tok, donated.shared_resident_bytes);
    // The restored ring holds only its retained rows; the first append regrows it, so the ring
    // and both checkpoint copies are billed whole on a warm turn too.
    try t.expect(donated.qsa_ring_bytes > 0);
    try t.expectEqual(cold.qsa_ring_bytes, donated.qsa_ring_bytes);
    // A share is copied by its first append: nothing credited.
    const shared = prefillRequestTerms(&cfg, seq, 32_000, kv_bits, chunk, .{ .matched_tokens = matched, .capacity_tokens = matched, .will_donate = false });
    try t.expectEqual(@as(u64, 0), shared.shared_resident_bytes);
}

test "mimo_v2 at 500k, kv8, chunk 2048: the fused sliding arm drops exactly the band sheet" {
    if (!archBuilt("mimo2")) return error.SkipZigTest;
    const t = std.testing;
    const cfg = fixtures.mimoV2FlashBillConfig();
    const seq: u64 = 500_000;
    const chunk: u64 = 2048;
    attn_kernels.fused256_override = true;
    defer attn_kernels.fused256_override = null;
    defer attn_kernels.sliding_prefill_fused_override = null;

    attn_kernels.sliding_prefill_fused_override = false;
    const band = slidingBandScoreBytes(&cfg, chunk);
    try t.expectEqual(MOE_PREFILL_COEXIST * 64 * chunk * (128 + chunk - 1) * 2, band);
    const composed = prefillNeededAtChunk(&cfg, seq, 2048, 8, chunk, .{});
    attn_kernels.sliding_prefill_fused_override = null;
    try t.expectEqual(composed - band, prefillNeededAtChunk(&cfg, seq, 2048, 8, chunk, .{}));

    // One predicate decides the dispatch and the bill, at every width and switch state.
    for ([_]?bool{ null, false }) |ov| {
        attn_kernels.sliding_prefill_fused_override = ov;
        for ([_]u64{ 1, 8, 15, 16, 512, 2048, 4096 }) |fwd| {
            try t.expectEqual(trunk.slidingPrefillFused(&cfg, fwd), slidingBandScoreBytes(&cfg, fwd) == 0);
        }
    }
}

test "MiMo MTP state bill is constant and follows retained storage" {
    if (!archBuilt("mimo2")) return error.SkipZigTest;
    const t = std.testing;
    const saved = launch_state.server_config.default_force_mtp;
    defer launch_state.server_config.default_force_mtp = saved;
    launch_state.server_config.default_force_mtp = false;
    var cfg = fixtures.mimoV2BillConfig();
    cfg.hidden_size = 4096;
    cfg.num_key_value_heads = 8;
    cfg.mtp_override = false;
    const kv_row: u64 = 8 * (192 + 128) * 2;
    const kv: u64 = 3 * (2 * 128 - 1) * kv_row;
    const hiddens: u64 = 256 * 4096 * 2;
    const catchup: u64 = 3 * 128 * 4096 * 2;
    const expected: u64 = 2 * (kv + hiddens) + catchup;
    for ([_]u64{ 1024, 131072 }) |seq| {
        const off = prefillRequestTerms(&cfg, seq, 2048, 8, 512, .{});
        const on = prefillRequestTerms(&cfg, seq, 2048, 8, 512, .{ .mtp_on = true });
        try t.expectEqual(@as(u64, 0), off.state_bytes);
        try t.expectEqual(expected, on.state_bytes);
        try t.expectEqual(@as(u64, 0), on.mtp_head_kv_bytes);
    }
    const off_reserve = prefillTransientReserve(&cfg, 8, 512);
    const off_per_token = sessionBytesPerToken(&cfg, 8);
    cfg.mtp_override = true;
    try t.expectEqual(expected * 5 / 4, prefillTransientReserve(&cfg, 8, 512) - off_reserve);
    try t.expectEqual(off_per_token, sessionBytesPerToken(&cfg, 8));
    cfg.hidden_size = 2048;
    cfg.num_key_value_heads = 4;
    cfg.sliding_window = 64;
    const smaller: u64 = 2 * (3 * (2 * 64 - 1) * 4 * (192 + 128) * 2 + 256 * 2048 * 2) + 3 * 64 * 2048 * 2;
    try t.expectEqual(smaller, prefillRequestTerms(&cfg, 1024, 2048, 4, 512, .{}).state_bytes);
}

test "resolvedPrefixCacheMem: RAM-off bills zero despite a configured disk tier" {
    const capacity = launch_state.prefix_cache_capacity;
    const ram = launch_state.prefix_cache_ram_enabled;
    defer launch_state.prefix_cache_capacity = capacity;
    defer launch_state.prefix_cache_ram_enabled = ram;
    launch_state.prefix_cache_capacity = 8;
    launch_state.prefix_cache_ram_enabled = false;
    try testing.expectEqual(@as(u64, 0), resolvedPrefixCacheMem());
    launch_state.prefix_cache_ram_enabled = true;
    launch_state.prefix_cache_capacity = 0;
    try testing.expectEqual(@as(u64, 0), resolvedPrefixCacheMem());
}

test "disabled prefix cache: sizing releases the cache reserve on every arch" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    if (!archBuilt("glm5")) return error.SkipZigTest;
    const saved_capacity = launch_state.prefix_cache_capacity;
    const saved_ask = launch_state.prefix_cache_mem_bytes;
    defer launch_state.prefix_cache_capacity = saved_capacity;
    defer launch_state.prefix_cache_mem_bytes = saved_ask;
    const saved_ram = launch_state.prefix_cache_ram_enabled;
    defer launch_state.prefix_cache_ram_enabled = saved_ram;
    var gated = fixtures.longCtxTestConfig();
    var other = fixtures.longCtxTestConfig();
    other.model_type = "not_a_served_arch";
    const configs = [_]*model_mod.ModelConfig{ &gated, &other };
    for ([_]u64{ 0, 2 << 30, 60 << 30 }) |ask| {
        launch_state.prefix_cache_mem_bytes = ask;
        launch_state.prefix_cache_capacity = 0;
        for (configs) |cfg| {
            try testing.expectEqual(@as(u64, 0), ctxSizingCacheReserve(cfg));
        }
        try testing.expectEqual(@as(u64, 0), legacyPrefixCacheAsk(&other));
        launch_state.prefix_cache_capacity = 32;
        launch_state.prefix_cache_ram_enabled = false;
        for (configs) |cfg| {
            try testing.expectEqual(@as(u64, 0), ctxSizingCacheReserve(cfg));
        }
        try testing.expectEqual(@as(u64, 0), legacyPrefixCacheAsk(&other));
        launch_state.prefix_cache_ram_enabled = true;
    }
    launch_state.prefix_cache_capacity = 32;
    launch_state.prefix_cache_ram_enabled = true;
    launch_state.prefix_cache_mem_bytes = 10 << 30;
    try testing.expectEqual(CTX_SIZING_CACHE_RESERVE, ctxSizingCacheReserve(&gated));
    try testing.expectEqual(launch_state.prefix_cache_mem_bytes, ctxSizingCacheReserve(&other));
    try testing.expectEqual(launch_state.prefix_cache_mem_bytes, legacyPrefixCacheAsk(&other));
    // GLM's context is sized with no cache reserve (its admission evicts); its unnamed ask is
    // its 1 GiB tier and a named one stands.
    const glm = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    try testing.expectEqual(@as(u64, 0), ctxSizingCacheReserve(&glm));
    try testing.expectEqual(GLM_PREFIX_CACHE_MEM_DEFAULT, defaultPrefixCacheAskFor(&glm, launch_state.PREFIX_CACHE_MEM_DEFAULT, false, 8 << 30));
    try testing.expectEqual(launch_state.PREFIX_CACHE_MEM_DEFAULT, defaultPrefixCacheAskFor(&glm, launch_state.PREFIX_CACHE_MEM_DEFAULT, true, 8 << 30));
    try testing.expectEqual(@as(u64, 6 << 30), defaultPrefixCacheAskFor(&glm, 6 << 30, true, 8 << 30));
    try testing.expectEqual(@as(u64, 8 << 30), defaultPrefixCacheAskFor(&other, launch_state.PREFIX_CACHE_MEM_DEFAULT, false, 8 << 30));
    // `--prefix-cache-mem off --prefix-cache-disk 12GB` keeps no idle RAM.
    const saved_disk = launch_state.prefix_cache_disk_bytes;
    defer launch_state.prefix_cache_disk_bytes = saved_disk;
    launch_state.prefix_cache_disk_bytes = 12 << 30;
    launch_state.prefix_cache_ram_enabled = false;
    try testing.expectEqual(@as(u64, 0), legacyPrefixCacheAsk(&other));
}

test "a Qwen or MiMo request owes no growth after admission, its reserved capacity is taken at its first grow" {
    if (!archBuilt("qwen4") or !archBuilt("mimo2")) return error.SkipZigTest;
    for ([_][]const u8{ "qwen4_exp", "mimo_v2" }) |model_type| {
        const cfg = model_mod.ModelConfig{ .model_type = model_type };
        try std.testing.expectEqual(@as(u64, 0), admissionCommitBytes(&cfg, 40_000, 4096, 8));
    }
}

test "GLM prefill bills the packed selection planes of its history" {
    if (!archBuilt("glm5")) return error.SkipZigTest;
    const cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    const attention = @import("glm5/glm5_attention.zig");
    const short: u64 = 1 << 19;
    const long: u64 = 1 << 20;
    // Beyond selection planes, only the growing kv8 latent (544) and pooled index (64) scale with history.
    const growth = (long - short) * (544 + 64);
    const planes = try attention.selectionScratchBytes(long) - try attention.selectionScratchBytes(short);
    try std.testing.expectEqual(growth + planes, prefillTransientReserveAtKv(&cfg, 8, 2048, long) - prefillTransientReserveAtKv(&cfg, 8, 2048, short));
}

test "a streamed Qwen plans its QSA history beside its KV for every planned token" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const fused_guard = fixtures.qsaScoreFusedOffGuard();
    defer fused_guard.deinit();
    var cfg = fixtures.qwen4RequestTestConfig();
    cfg.expert_streaming = true;
    const saved = launch_state.configured_kv_quant;
    defer launch_state.configured_kv_quant = saved;
    launch_state.configured_kv_quant = trunk.KVQuantConfig.affine(8);
    const state = statePerTokenBilled(&cfg);
    try std.testing.expect(state > 0);
    const per_token = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), 8) + state;
    try std.testing.expectEqual(per_token * expertStreamingPlannedSeq(&cfg) + slotRingBytes(&cfg, 8), expertStreamingPlannedKvBytes(&cfg));
}

test "GLM release defaults advertise the full context at the widest prefill chunk that keeps it" {
    if (!archBuilt("glm5")) return error.SkipZigTest;
    var cfg = try fixtures.glmAssistantFixture();
    cfg.pinned_prefill_chunk = 0;
    const kv_bits: u64 = 8;
    const cap = cfg.contextCap();
    // Sushi-2.5bpw with vision and the A4 assistant, as its boot measured: 104.35 GB active, 115.9 GB ceiling.
    const ceiling: u64 = 115_904 * 1024 * 1024;
    const active: u64 = 104_350_000_000;
    try std.testing.expectEqual(@as(u32, 2048), resolvePrefillChunk(&cfg, kv_bits, ceiling, active, 0, 0));
    try std.testing.expectEqual(cap, advertisedContextAt(&cfg, kv_bits, ceiling, active, 2048));
    // A 1 GiB idle cache reserve takes the same bytes as a 1 GiB lower ceiling: 2048 rows would cost
    // advertised context there, 1024 keeps all of it.
    const reserved = ceiling - 1024 * 1024 * 1024;
    try std.testing.expect(advertisedContextAt(&cfg, kv_bits, reserved, active, 2048) < cap);
    try std.testing.expectEqual(@as(u32, 1024), resolvePrefillChunk(&cfg, kv_bits, reserved, active, 0, 0));
    try std.testing.expectEqual(cap, advertisedContextAt(&cfg, kv_bits, reserved, active, 1024));
    // The least free memory at which 1024 rows still advertise the position cap keeps 1024.
    var low: u64 = 0;
    var high: u64 = ceiling - active;
    while (low < high) {
        const mid = low + (high - low) / 2;
        if (advertisedContextAt(&cfg, kv_bits, active + mid, active, 1024) >= cap) high = mid else low = mid + 1;
    }
    try std.testing.expect(advertisedContextAt(&cfg, kv_bits, active + low, active, 2048) < cap);
    try std.testing.expectEqual(@as(u32, 1024), resolvePrefillChunk(&cfg, kv_bits, active + low, active, 0, 0));
    // Far less memory: every widening costs advertised context.
    try std.testing.expectEqual(@as(u32, 512), resolvePrefillChunk(&cfg, kv_bits, active + 8 * 1024 * 1024 * 1024, active, 0, 0));
}

test "GLM prefill chunk never widens past the width its numerics are built for" {
    if (!archBuilt("glm5")) return error.SkipZigTest;
    var cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    const roomy: u64 = 1 << 40;
    for ([_]bool{ false, true }) |assistant| {
        cfg.dflash_loaded = assistant;
        for ([_]u64{ 16, 8 }) |kv_bits| try std.testing.expectEqual(@as(u32, 2048), resolvePrefillChunk(&cfg, kv_bits, roomy, 0, 0, 0));
    }
}

test "GLM DFlash2 reserve at prefill end fits the admission bill at every request shape" {
    if (!archBuilt("glm5")) return error.SkipZigTest;
    const reserve = @import("glm5/glm5_dflash_reserve.zig");
    var cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    cfg.dflash_loaded = true;
    const mla_layers: usize = cfg.num_hidden_layers / cfg.full_attention_interval;
    const pool_row: usize = @as(usize, cfg.indexer_head_dim) * 2;
    const shapes = [_][2]u64{ .{ 16, std.math.maxInt(u32) }, .{ 18, 32 }, .{ 100, 920 }, .{ 1000, 24 }, .{ 1022, 2 }, .{ 2008, 2400 }, .{ 32768, 32768 }, .{ 131072, 4096 }, .{ 2008, std.math.maxInt(u32) }, .{ 500000, std.math.maxInt(u32) } };
    for ([_]u32{ 1024, 8192, 65536, 1048576 }) |ctx| {
        cfg.pinned_context = ctx;
        for ([_]u64{ 16, 8, 6 }) |kv_bits| {
            const latent_row: usize = if (kv_bits == 16) @as(usize, cfg.mla_kv_lora_rank) * 2 else @import("glm5/glm5_latent.zig").rowBytes(cfg.mla_kv_lora_rank, @intCast(kv_bits));
            for (shapes) |shape| {
                const seq: usize = @intCast(shape[0]);
                if (seq >= ctx) continue;
                const max_tokens: u32 = @intCast(@min(shape[1], ctx - seq));
                const lc = try reserve.capacity(seq);
                const pc = try reserve.capacity(seq / 4);
                const p = try reserve.plan(seq, lc, pc, latent_row, cfg.indexer_head_dim, 2, seq + max_tokens + 3);
                const held: u64 = mla_layers * (lc * latent_row + pc * pool_row) + cfg.ssmCheckpointBytes();
                try std.testing.expect(try reserve.statesPeak(mla_layers, p) <= glmDflashReserveBudget(&cfg, seq, max_tokens, kv_bits, held));
            }
        }
    }
}

test "GLM DFlash2 reserve fits the admission bill when max_tokens fills the advertised context" {
    if (!archBuilt("glm5")) return error.SkipZigTest;
    const reserve = @import("glm5/glm5_dflash_reserve.zig");
    var cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    cfg.dflash_loaded = true;
    var mla_layers: usize = 0;
    for (0..cfg.num_hidden_layers) |l| mla_layers += @intFromBool(!cfg.isLinearLayer(@intCast(l)));
    const pool_row: usize = @as(usize, cfg.indexer_head_dim) * 2;
    for ([_]struct { bits: u64, ctx: u32 }{ .{ .bits = 8, .ctx = 1048576 }, .{ .bits = 6, .ctx = 1048576 }, .{ .bits = 16, .ctx = 524288 } }) |arm| {
        cfg.pinned_context = arm.ctx;
        const latent_row: usize = if (arm.bits == 16) @as(usize, cfg.mla_kv_lora_rank) * 2 else @import("glm5/glm5_latent.zig").rowBytes(cfg.mla_kv_lora_rank, @intCast(arm.bits));
        for ([_]usize{ 16, 2008, 100000 }) |seq| {
            const max_tokens: u32 = @intCast(arm.ctx - seq);
            const lc = try reserve.capacity(seq);
            const pc = try reserve.capacity(seq / 4);
            const p = try reserve.plan(seq, lc, pc, latent_row, cfg.indexer_head_dim, 2, seq + max_tokens + 3);
            const held: u64 = mla_layers * (lc * latent_row + pc * pool_row) + cfg.ssmCheckpointBytes();
            const peak = try reserve.statesPeak(mla_layers, p);
            // The generator's config copy is never pinned: it resolves its own, smaller context.
            var unpinned = cfg;
            unpinned.pinned_context = arm.ctx / 4;
            try std.testing.expect(peak <= glmDflashReserveBudget(&unpinned, seq, max_tokens, arm.bits, held));
        }
    }
}

test "GLM admission credits the hot cache, so an eviction lets a long prefill in" {
    if (!archBuilt("glm5")) return error.SkipZigTest;
    const cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    const gb: u64 = 1 << 30;
    const bill = creditedAdmissionBill(&cfg, 20 * gb, 19 * gb, 3 * gb / 2, 3 * gb / 2, 2048);
    try std.testing.expectEqual(AdmissionVerdict.evict, admissionVerdict(bill));
    try std.testing.expectEqual(AdmissionVerdict.refuse, admissionVerdict(creditedAdmissionBill(&cfg, 21 * gb, 19 * gb, 3 * gb / 2, 3 * gb / 2, 2048)));
}

test "auto context on a room of about 1.8 GB neither advertises a few tokens nor lets the cache reserve eat it" {
    const GiB: u64 = 1 << 30;
    const per_tok: u64 = 278528;
    const room: u64 = GiB * 18 / 10;
    // The 2 GiB cache reserve alone exceeds the room: the context, not the cache, gets the half.
    const share = ctxCacheShare(room + 5 * GiB, 5 * GiB, 2 * GiB);
    try testing.expectEqual(room / 2, share);
    try testing.expectEqual(@as(u64, GiB), ctxCacheShare(64 * GiB, 5 * GiB, GiB));
    const auto = resolvedContextForLoad(0, 0, room + 5 * GiB, 5 * GiB, share, 0, per_tok, 0, 0);
    try testing.expectEqual(@as(u32, 2048), auto);
    // A room that fits a few tokens past the reserve floors at 1024 instead of advertising them.
    try testing.expectEqual(@as(u32, 1024), safeContextForBudget(2 * GiB + 100 * (1 << 20), 0, 2 * GiB, per_tok, 0));
}

test "bill pin: kvBytesPerTokenAtBits" {
    var got = PinGot{};
    for ([_]u64{ 1, 98_304, 12_345_678 }) |dense| for ([_]u64{ 4, 6, 8, 16 }) |bits| got.add(kvBytesPerTokenAtBits(dense, bits));
    try pinEq("kvBytesPerTokenAtBits", &.{
        0, 0, 0, 1, 27648, 39936, 52224, 98304,
        3472221, 5015431, 6558641, 12345678,
    }, got.slice());
}

test "bill pin: slotRingBytes" {
    try pinConfigs("slotRingBytes", &.{ 98304, 98304, 199904640, 348541440, 147636736, 147636736, 6303744, 6303744 }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            for (pin_bits) |b| g.add(slotRingBytes(c, b));
        }
    }.run);
}

test "bill pin: batchedDecodeRowsBytes" {
    try pinConfigs("batchedDecodeRowsBytes", &.{ 0, 31449600, 0, 0 }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            g.add(batchedDecodeRowsBytes(c));
        }
    }.run);
}

test "bill pin: statePerTokenBilled" {
    try pinConfigs("statePerTokenBilled", &.{ 2304, 0, 704, 0 }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            g.add(statePerTokenBilled(c));
        }
    }.run);
}

test "bill pin: sessionBytesPerToken" {
    try pinConfigs("sessionBytesPerToken", &.{ 17600, 29120, 12240, 23040, 6688, 11968, 890, 890 }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            for (pin_bits) |b| g.add(sessionBytesPerToken(c, b));
        }
    }.run);
}

test "bill pin: the MTP head terms" {
    try pinConfigs("mtpHead", &.{
        2048, 8192, 192, 1, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0,
    }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            g.add(mtpHeadKvBytesPerToken(c));
            g.add(mtpHeadQsaRingBytes(c));
            g.add(mtpHeadStateBytesPerToken(c));
            g.add(mtpHeadDefaultOn(c));
        }
    }.run);
}

test "bill pin: prefillFfnWidth prefillStreamBytesPerToken prefillDequantWeightBytes" {
    try pinConfigs("prefillWidths", &.{
        7040, 614400, 108134400, 16384, 812032, 0, 18432, 1032192,
        0, 15360, 0, 0,
    }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            g.add(prefillFfnWidth(c));
            g.add(prefillStreamBytesPerToken(c));
            g.add(prefillDequantWeightBytes(c));
        }
    }.run);
}

test "bill pin: prefillMemoryNeeded" {
    try pinConfigs("prefillMemoryNeeded", &.{
        1155274240, 945764388, 1167114240, 939458596, 1598768640, 1300559908, 1610608640, 1294254116,
        2457994240, 1846628388, 3286794240, 2635202596, 3989079040, 3071496228, 4817879040, 3860070436,
        1381541280, 1122682020, 1387361280, 1115656228, 2036788640, 1646879908, 2042608640, 1639854116,
        2967161280, 1967242020, 3374561280, 2705416228, 5029579200, 3617176356, 5436979200, 4355350564,
        1530475200, 1245925156, 1534515200, 1233379364, 2340008640, 1893551908, 2344048640, 1881006116,
        2223235200, 1658821156, 2506035200, 2010595364, 4771274880, 3697252900, 5054074880, 4049027108,
        1153777840, 938423268, 1144060340, 921015476, 1603518640, 1298215908, 1593801140, 1280808116,
        1901047840, 970991268, 1220822840, 982425476, 3316625440, 2103453348, 2636400440, 2114887556,
    }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            const terms = PrefillRequestTerms{ .reserved_kv_bytes = 1 << 20, .state_bytes = 2 << 20, .checkpoint_bytes = 3 << 20, .shared_resident_bytes = 1 << 19, .grow_coexist_bytes = 5, .qsa_ring_bytes = 7, .mtp_head_kv_bytes = 11, .dflash_bytes = 13, .uplift_16ths = 16, .dequant_scratch_bytes = 17 << 20 };
            for ([_]u64{ 1000, 70_000 }) |seq| for ([_]u64{ 512, 2048 }) |chunk| for (pin_bits) |b| {
                for ([_]PrefillRequestTerms{ .{}, terms }) |t|
                    g.add(prefillMemoryNeeded(seq, c.num_attention_heads, c.num_key_value_heads, c.kvBytesPerToken(), c.head_dim, c.prefillScoreHeadDim(), c.hidden_size, prefillFfnWidth(c), b, chunk, c.prefillAttnKeys(seq), prefillStreamBytesPerToken(c), prefillDequantWeightBytes(c), t));
            };
        }
    }.run);
}

test "bill pin: prefillTransientReserve and prefillTransientReserveAtKv" {
    try pinConfigs("prefillTransientReserve", &.{
        913670144, 2007965696, 3841163264, 5598412928, 912621568, 1751965696, 3832774656, 5342412928,
        1630685920, 1837124320, 6452824800, 6649432800, 1813205120, 1813205120, 6625513600, 6625513600,
        1637207552, 1701999104, 5399432704, 5462045184, 1635074560, 1747620352, 5397070336, 5505716736,
        1841562496, 3462466456, 12853451776, 14472514456, 1841562496, 3462466456, 12853451776, 14472514456,
    }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            for (pin_bits) |b| for ([_]u64{ 512, 4096 }) |chunk| {
                g.add(prefillTransientReserve(c, b, chunk));
                g.add(prefillTransientReserveAtKv(c, b, chunk, 100_000));
            };
        }
    }.run);
}

test "bill pin: the per-forward scratch terms" {
    try pinConfigs("scratchTerms", &.{
        0, 0, 0, 4231168, 350000, 0, 0, 0,
        143360000, 431128576, 0, 0, 0, 143360000, 813108608, 65536,
        0, 0, 167772160, 0, 0, 0, 0, 167772160,
        0, 0, 0, 0, 167772160, 0, 0, 0,
        0, 0, 350000, 0, 0, 0, 0, 567488576,
        0, 0, 0, 0, 574828608, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            for ([_]u64{ 1, 512, 4096 }) |fwd| {
                g.add(slidingBandScoreBytes(c, fwd));
                g.add(fp8DequantScratchBytes(c, fwd));
                g.add(trunkDecodeScratchBytes(c, fwd));
                g.add(kvDequantScratchBytes(c, 70_000, fwd));
                g.add(qsaMaskBytes(c, fwd, 70_000));
            }
        }
    }.run);
}

test "bill pin: prefillChunkCap" {
    try pinConfigs("prefillChunkCap", &.{
        10737418240, 10737418240, 3221225472, 3221225472, 10737418240, 9663676416, 3221225472, 2147483648,
        10737418240, 9663676416, 3221225472, 2147483648, 10737418240, 9663676416, 3221225472, 2147483648,
    }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            for ([_][2]u64{ .{ 100 * pin_gib, 60 * pin_gib }, .{ 32 * pin_gib, 20 * pin_gib } }) |m| for ([_]u64{ 0, 4 * pin_gib }) |ask|
                g.add(prefillChunkCap(c, m[0], m[1], 2 * pin_gib, ask));
        }
    }.run);
}

test "bill pin: resolvePrefillChunk and billedPrefillChunk" {
    try pinConfigs("resolvePrefillChunk", &.{
        8192, 8192, 1024, 8192, 8192, 1024, 2048, 2048,
        1024, 2048, 2048, 1024, 4096, 4096, 1024, 4096,
        4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024,
        2048, 2048, 1024, 2048, 2048, 1024, 512, 512,
        1024, 512, 512, 1024, 2048, 2048, 1024, 2048,
        2048, 1024, 2048, 2048, 1024, 2048, 2048, 1024,
    }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            for ([_][2]u64{ .{ 100 * pin_gib, 60 * pin_gib }, .{ 32 * pin_gib, 20 * pin_gib } }) |m| for (pin_bits) |b| {
                g.add(resolvePrefillChunk(c, b, m[0], m[1], 0, 2 * pin_gib));
                g.add(billedPrefillChunk(c, b, m[0], m[1], 0, 2 * pin_gib, 0));
                g.add(billedPrefillChunk(c, b, m[0], m[1], 0, 2 * pin_gib, 1024));
            };
        }
    }.run);
}

test "bill pin: contextNeutralChunk and advertisedContextAt" {
    try pinConfigs("advertisedContextAt", &.{
        974848, 2048, 974848, 2048, 1048576, 2048, 1048576, 2048,
        1048576, 2048, 1048576, 2048, 1000000, 2048, 1000000, 2048,
    }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            for ([_]u64{ 512, 2048 }) |chunk| {
                g.add(advertisedContextAt(c, 8, 100 * pin_gib, 60 * pin_gib, chunk));
                g.add(contextNeutralChunk(c, 8, 100 * pin_gib, 60 * pin_gib, 2048));
            }
        }
    }.run);
}

test "bill pin: sizerCtxKvBytes" {
    const saved = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = saved;
    launch_state.server_config.max_context_size = 100_000;
    try pinConfigs("sizerCtxKvBytes", &.{ 1760098304, 2912098304, 1423904640, 2652541440, 816436736, 1344436736, 95303744, 95303744 }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            for (pin_bits) |b| g.add(sizerCtxKvBytes(c, b));
        }
    }.run);
}

test "bill pin: sushiContextBytesAt and the max-context search" {
    const saved = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = saved;
    launch_state.server_config.max_context_size = 100_000;
    try pinConfigs("maxSushiContext", &.{
        4652207616, 512, 1048576, 1048576, 200000, 3544957920, 512, 1048576,
        1048576, 200000, 2966441600, 512, 1048576, 1048576, 200000, 4845384760,
        512, 1048576, 1048576, 200000,
    }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            const at = sushiContextBytesAt(c, 100_000, 8);
            g.add(at.bytes);
            g.add(at.chunk);
            g.add(maxSushiContextForBudget(c, 8, 40 * pin_gib));
            g.add(maxSushiContextAtWidth(c, 8, 40 * pin_gib, 1024, .{}));
            g.add(maxSushiContextUpTo(c, 200_000, 16, 40 * pin_gib, null, .{ .mtp_on = true }));
        }
    }.run);
}

test "bill pin: loadServingBill" {
    const saved = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = saved;
    launch_state.server_config.max_context_size = 100_000;
    try pinConfigsRagged("loadServingBill", &.{
        0, 1, 100000, 4652207616, 1048576, 8, 512, 0,
        1, 100000, 3813393376, 1048576, 8, 512, 0, 0,
        0, 0,
    }, .{ 7, 7, 2, 2 }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            var exl3 = c.*;
            exl3.expert_layout = .exl3_k4;
            for ([_]*const PinCfg{ c, &exl3 }) |cfg| {
                const bill = loadServingBill(cfg, 40 * pin_gib);
                g.add(bill != null);
                if (bill) |b| {
                    g.add(b.requested_context);
                    g.add(b.needed);
                    g.add(b.max_context);
                    g.add(b.kv_bits);
                    g.add(b.chunk);
                }
            }
        }
    }.run);
}

test "bill pin: memoryContextAt and memoryContextAtChunk" {
    try pinConfigs("memoryContextAt", &.{
        1048576, 1048576, 446723, 446723, 2052459, 2160469, 480445, 588455,
        3799051, 3953345, 922039, 1076334, 1076224, 1076224, 1076224, 1076224,
    }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            for ([_][2]u64{ .{ 100 * pin_gib, 60 * pin_gib }, .{ 32 * pin_gib, 20 * pin_gib } }) |m| {
                g.add(memoryContextAt(c, m[0], m[1]));
                g.add(memoryContextAtChunk(c, 8, m[0], m[1], 512));
            }
        }
    }.run);
}

test "bill pin: the cache reserve and the context margins" {
    try pinConfigs("ctxReserve", &.{
        0, 2147483648, 536870912, 0, 930, 930, 930, 185344,
        185344, 185344, 1048576, 2789376, 1048576, 1024, 0, 2147483648,
        536870912, 0, 930, 930, 930, 185344, 185344, 185344,
        1048576, 2789376, 1048576, 1024, 0, 2147483648, 536870912, 0,
        930, 930, 930, 185344, 185344, 185344, 1048576, 2789376,
        1048576, 1024, 0, 2147483648, 536870912, 0, 930, 930,
        930, 185344, 185344, 185344, 1048576, 2789376, 1000000, 1076224,
    }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            g.add(ctxSizingCacheReserve(c));
            g.add(ctxCacheShare(100 * pin_gib, 60 * pin_gib, 2 * pin_gib));
            g.add(ctxCacheShare(100 * pin_gib, 99 * pin_gib, 2 * pin_gib));
            g.add(billedAutoContext(c));
            for ([_]u32{ 1000, 200_000, 3_000_000 }) |m| {
                g.add(billedAdvertisedContext(c, m));
                g.add(safeAutoContext(m));
                g.add(autoContextFrom(m, c.contextCap(), c.defaultContext()));
            }
            g.add(memoryContextForMargined(c.defaultContext()));
        }
    }.run);
}

test "bill pin: contextForBudget and safeContextForBudget" {
    var got = PinGot{};
    for ([_]u64{ 0, 1 << 20 }) |per_tok| for ([_]u64{ 0, 2 * pin_gib }) |reserve| {
        got.add(contextForBudget(100 * pin_gib, 60 * pin_gib, reserve, per_tok));
        got.add(safeContextForBudget(100 * pin_gib, 60 * pin_gib, reserve, per_tok, 131_072));
        got.add(safeContextForBudget(100 * pin_gib, 99 * pin_gib, reserve, per_tok, 0));
    };
    try pinEq("safeContextForBudget", &.{
        0, 1024, 1024, 0, 1024, 1024, 26214, 26214,
        1024, 24903, 24903, 1024,
    }, got.slice());
}

test "bill pin: clampedPrefixCacheMem and ssdFirstPrefixCacheMem" {
    var got = PinGot{};
    for ([_]u64{ 0, 2 * pin_gib, 80 * pin_gib }) |req| {
        got.add(clampedPrefixCacheMem(req, 100 * pin_gib, 60 * pin_gib, 4 * pin_gib, pin_gib));
        got.add(clampedPrefixCacheMem(req, 60 * pin_gib, 60 * pin_gib, 4 * pin_gib, pin_gib));
        got.add(ssdFirstPrefixCacheMem(req, 100 * pin_gib, 60 * pin_gib, 4 * pin_gib, pin_gib));
        got.add(ssdFirstPrefixCacheMem(req, 60 * pin_gib, 60 * pin_gib, 4 * pin_gib, pin_gib));
    }
    try pinEq("clampedPrefixCacheMem", &.{
        37580963840, 1, 4294967296, 4294967296, 2147483648, 1, 6442450944, 4294967296,
        37580963840, 1, 41875931136, 4294967296,
    }, got.slice());
}

test "bill pin: planHotCache" {
    try pinConfigs("planHotCache", &.{
        8192, 512, 913670144, 1760098304, 40275904512, 1024, 512, 913670144,
        1760098304, 40275904512, 8192, 512, 913670144, 1760098304, 2147483648, 1024,
        512, 913670144, 1760098304, 2147483648, 4096, 512, 1630685920, 1423904640,
        39895082400, 1024, 512, 1630685920, 1423904640, 39895082400, 4096, 512,
        1630685920, 1423904640, 2147483648, 1024, 512, 1630685920, 1423904640, 2147483648,
        2048, 512, 1637207552, 816436736, 40496028672, 1024, 512, 1637207552,
        816436736, 40496028672, 2048, 512, 1637207552, 816436736, 2147483648, 1024,
        512, 1637207552, 816436736, 2147483648, 2048, 512, 1841562496, 95303744,
        41012806720, 1024, 512, 1841562496, 95303744, 41012806720, 2048, 512,
        1841562496, 95303744, 2147483648, 1024, 512, 1841562496, 95303744, 2147483648,
    }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            for ([_]u64{ 0, 2 * pin_gib }) |req| for ([_]u32{ 0, 1024 }) |over| {
                const p = planHotCache(c, 8, 100 * pin_gib, 60 * pin_gib, 100_000, 0, req, over);
                g.add(p.chunk);
                g.add(p.reserve_chunk);
                g.add(p.reserve);
                g.add(p.ctx_kv);
                g.add(p.budget);
            };
        }
    }.run);
}

test "bill pin: resolvedContextForLoad" {
    var got = PinGot{};
    got.add(resolvedContextForLoad(50_000, 0, 100 * pin_gib, 60 * pin_gib, pin_gib, pin_gib, 1 << 20, 262_144, 0));
    got.add(resolvedContextForLoad(0, 40_000, 100 * pin_gib, 60 * pin_gib, pin_gib, pin_gib, 1 << 20, 262_144, 0));
    got.add(resolvedContextForLoad(0, 0, 100 * pin_gib, 60 * pin_gib, pin_gib, pin_gib, 1 << 20, 262_144, 0));
    got.add(resolvedContextForLoad(0, 0, 100 * pin_gib, 60 * pin_gib, pin_gib, pin_gib, 1 << 20, 16_000, 20_000));
    got.add(resolvedContextForLoad(0, 0, 100 * pin_gib, 60 * pin_gib, pin_gib, pin_gib, 0, 262_144, 0));
    try pinEq("resolvedContextForLoad", &.{ 50000, 40000, 22528, 20000, 952 }, got.slice());
}

test "bill pin: clampReserveWidth and perRequestFloorWidth" {
    try pinConfigs("clampReserveWidth", &.{
        512, 512, 4096, 1024, 512, 512, 2048, 1024,
        512, 512, 2048, 1024, 512, 512, 2048, 1024,
    }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            g.add(clampReserveWidth(c, 2048));
            g.add(perRequestFloorWidth(c));
            g.add(ladderTop(c, 0));
            g.add(ladderTop(c, 1024));
        }
    }.run);
}

test "bill pin: the unnamed hot-cache ask" {
    try pinConfigs("defaultPrefixCacheAsk", &.{
        0, 0, 0, 0, 3221225472, 3221225472, 2147483648, 2147483648,
        5368709120, 5368709120, 5368709120, 5368709120, 0, 0, 0, 0,
        3221225472, 3221225472, 2147483648, 2147483648, 5368709120, 5368709120, 5368709120, 5368709120,
        0, 0, 0, 0, 3221225472, 1073741824, 2147483648, 2147483648,
        5368709120, 5368709120, 5368709120, 5368709120, 0, 0, 0, 0,
        3221225472, 3221225472, 2147483648, 2147483648, 5368709120, 5368709120, 5368709120, 5368709120,
    }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            for ([_]u64{ 0, launch_state.PREFIX_CACHE_MEM_DEFAULT, 5 * pin_gib }) |req| for ([_]bool{ false, true }) |explicit| {
                g.add(defaultPrefixCacheAsk(req, explicit, 3 * pin_gib));
                g.add(defaultPrefixCacheAskFor(c, req, explicit, 3 * pin_gib));
            };
        }
    }.run);
}

test "bill pin: oneSessionEntryBytes and oneSessionFor" {
    try pinConfigs("oneSession", &.{
        2250651648, 2250651648, 36555305344, 5320625792, 1073741824, 2250651648, 1526246400, 1526246400,
        37635666976, 4240264160, 1073741824, 1526246400, 2052183552, 2052183552, 38238052736, 3637878400,
        1073741824, 2052183552, 96290880, 96290880, 35457682376, 6418248760, 1073741824, 96290880,
    }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            g.add(oneSessionEntryBytes(c, 8, 100_000, 1024));
            const one = oneSessionFor(c, 8, 100 * pin_gib, 60 * pin_gib, pin_gib, 100_000, 1024);
            g.add(one.session);
            g.add(one.room);
            g.add(one.bill);
            g.add(one.claim);
            g.add(one.ask());
        }
    }.run);
}

test "bill pin: diskBytesPerToken" {
    try pinConfigs("diskBytesPerToken", &.{ 30030, 41550, 12240, 23040, 24708, 29988, 1660, 1660 }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            for (pin_bits) |b| g.add(diskBytesPerToken(c, b));
        }
    }.run);
}

test "bill pin: reservedCacheTokens" {
    var got = PinGot{};
    for ([_]u64{ 100, 5000, 200_000 }) |seq| for ([_]u64{ 0, 4096 }) |max| got.add(reservedCacheTokens(seq, max, 2048, 262_144));
    try pinEq("reservedCacheTokens", &.{ 0, 0, 0, 0, 202048, 206144 }, got.slice());
}

test "bill pin: prefillRequestTerms" {
    try pinConfigs("prefillRequestTerms", &.{
        0, 2304000, 115458048, 0, 0, 98304, 0, 0,
        16, 2048000, 0, 2496000, 115458048, 0, 0, 106496,
        2048000, 0, 16, 2048000, 0, 2304000, 0, 13056000,
        0, 98304, 0, 0, 16, 2048000, 0, 2496000,
        230916096, 0, 0, 106496, 2048000, 0, 16, 2048000,
        0, 2304000, 115458048, 0, 0, 98304, 0, 0,
        16, 2048000, 0, 2496000, 115458048, 0, 0, 106496,
        2048000, 0, 16, 2048000, 0, 2304000, 0, 24576000,
        0, 98304, 0, 0, 16, 2048000, 0, 2496000,
        230916096, 0, 0, 106496, 2048000, 0, 16, 2048000,
        80216064, 244555776, 346374144, 0, 0, 98304, 0, 0,
        20, 204800000, 80216064, 264935424, 346374144, 0, 0, 106496,
        217382912, 0, 20, 204800000, 80216064, 244555776, 346374144, 652800000,
        65280000, 98304, 0, 0, 20, 204800000, 80216064, 264935424,
        461832192, 0, 0, 106496, 217382912, 0, 20, 204800000,
        150994944, 244555776, 346374144, 0, 0, 98304, 0, 0,
        20, 204800000, 150994944, 264935424, 346374144, 0, 0, 106496,
        217382912, 0, 20, 204800000, 150994944, 244555776, 346374144, 1228800000,
        122880000, 98304, 0, 0, 20, 204800000, 150994944, 264935424,
        461832192, 0, 0, 106496, 217382912, 0, 20, 204800000,
        0, 0, 0, 0, 1360000, 199904640, 0, 0,
        20, 3276800, 0, 15173632, 0, 0, 1360000, 199904640,
        0, 0, 20, 3276800, 0, 0, 0, 12240000,
        1360000, 199904640, 0, 0, 20, 3276800, 0, 15173632,
        0, 0, 1360000, 199904640, 0, 0, 20, 3276800,
        0, 0, 0, 0, 2560000, 348541440, 0, 0,
        20, 3276800, 0, 15173632, 0, 0, 2560000, 348541440,
        0, 0, 20, 3276800, 0, 0, 0, 23040000,
        2560000, 348541440, 0, 0, 20, 3276800, 0, 15173632,
        0, 0, 2560000, 348541440, 0, 0, 20, 3276800,
        75202560, 0, 0, 0, 136000000, 199904640, 0, 0,
        20, 167772160, 75202560, 15173632, 0, 0, 136000000, 199904640,
        0, 0, 20, 167772160, 75202560, 0, 0, 612000000,
        81600000, 199904640, 0, 0, 20, 167772160, 75202560, 15173632,
        0, 0, 136000000, 199904640, 0, 0, 20, 167772160,
        141557760, 0, 0, 0, 256000000, 348541440, 0, 0,
        20, 167772160, 141557760, 15173632, 0, 0, 256000000, 348541440,
        0, 0, 20, 167772160, 141557760, 0, 0, 1152000000,
        153600000, 348541440, 0, 0, 20, 167772160, 141557760, 15173632,
        0, 0, 256000000, 348541440, 0, 0, 20, 167772160,
        0, 0, 0, 0, 0, 0, 0, 0,
        20, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 20, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 20, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 20, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        20, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 20, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 20, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 20, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        20, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 20, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 20, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 20, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        20, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 20, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 20, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 20, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        20, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 20, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 20, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 20, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        20, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 20, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 20, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 20, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        20, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 20, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 20, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 20, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        20, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 20, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 20, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 20, 0,
    }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            for ([_]u64{ 1000, 100_000 }) |seq| for (pin_bits) |b| for (pinWarms()) |w| {
                const t = prefillRequestTerms(c, seq, 4096, b, 2048, w);
                g.add(t.reserved_kv_bytes);
                g.add(t.state_bytes);
                g.add(t.checkpoint_bytes);
                g.add(t.shared_resident_bytes);
                g.add(t.grow_coexist_bytes);
                g.add(t.qsa_ring_bytes);
                g.add(t.mtp_head_kv_bytes);
                g.add(t.dflash_bytes);
                g.add(t.uplift_16ths);
                g.add(t.dequant_scratch_bytes);
            };
        }
    }.run);
}

test "bill pin: growCoexistBytes and the eval-window terms" {
    try pinConfigs("growCoexist", &.{
        0, 0, 491520000, 0, 409600000, 4915200000, 1, 16,
        20, 1092266666, 1092266666, 655360000, 1092266666, 546133333, 4915199997, 1,
        20, 20, 0, 0, 536203636, 0, 446836363, 4915199993,
        1, 20, 20, 0, 0, 491520000, 0, 409600000,
        4915200000, 4, 20, 20,
    }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            for (pinWarms()) |w| g.add(growCoexistBytes(c, w, 100_000, 120_000, 98_304));
            g.add(oldBuffersInEvalWindow(c, 50_000, 2048, 98_304));
            g.add(oldBuffersInEvalWindow(c, 50_000, 1, 98_304));
            g.add(attnLayersPerEvalWindow(c, 4));
            g.add(billUplift16ths(c, 1000));
            g.add(billUplift16ths(c, 200_000));
        }
    }.run);
}

test "bill pin: prefillNeededAtChunk" {
    try pinConfigs("prefillNeededAtChunk", &.{
        1048735744, 1050983936, 920221696, 1166441984, 1409875264, 1412123456, 1281361216, 1527581504,
        1058207744, 1060455936, 918173696, 1175913984, 1419347264, 1421595456, 1279313216, 1537053504,
        4449406976, 4742319616, 3715006976, 4886642176, 6203928704, 6501142144, 5469528704, 6645464704,
        5699762176, 5992674816, 4317362176, 6136997376, 7476402304, 7773615744, 6094002304, 7917938304,
        1629538080, 1648505120, 1614238080, 1648505120, 2284785440, 2303752480, 2269485440, 2303752480,
        1826238080, 1845205120, 1797438080, 1845205120, 2481485440, 2500452480, 2452685440, 2500452480,
        3588659680, 3607626720, 2755659680, 3607626720, 5674578400, 5693545440, 4841578400, 5693545440,
        5126948480, 5145915520, 3558948480, 5145915520, 7233603200, 7252570240, 5665603200, 7252570240,
        2089683520, 2089683520, 2089683520, 2274208320, 2729646720, 2729646720, 2729646720, 2914171520,
        2121102080, 2121102080, 2121102080, 2305626880, 2760733440, 2760733440, 2760733440, 2945258240,
        3000684160, 3000684160, 3000684160, 3369733760, 5014994560, 5014994560, 5014994560, 5384044160,
        3747067520, 3747067520, 3747067520, 4116117120, 5760333440, 5760333440, 5760333440, 6129383040,
        4508149310, 4508149310, 4508149310, 4508149310, 4508149310, 4508149310, 4508149310, 4508149310,
        4508149310, 4508149310, 4508149310, 4508149310, 4508149310, 4508149310, 4508149310, 4508149310,
        4849941560, 4849941560, 4849941560, 4849941560, 9568533560, 9568533560, 9568533560, 9568533560,
        4849941560, 4849941560, 4849941560, 4849941560, 9568533560, 9568533560, 9568533560, 9568533560,
    }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            for ([_]u64{ 1000, 100_000 }) |seq| for (pin_bits) |b| for ([_]u64{ 512, 2048 }) |chunk| for (pinWarms()) |w|
                g.add(prefillNeededAtChunk(c, seq, 4096, b, chunk, w));
        }
    }.run);
}

test "bill pin: defaultRequestBill" {
    try pinConfigs("defaultRequestBill", &.{ 537696256, 1108173696, 1037824640, 4407603760 }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            g.add(defaultRequestBill(c));
        }
    }.run);
}

test "bill pin: chooseRequestPrefillChunk and requestPrefillPick" {
    try pinConfigs("chooseRequestPrefillChunk", &.{
        4096, 1024, 4096, 2048, 4096, 1024, 4096, 2048,
        2048, 1024, 2048, 2048, 4096, 1024, 4096, 2048,
        2048, 1024, 2048, 2048, 2048, 1024, 2048, 2048,
        2048, 1024, 2048, 2048, 2048, 1024, 2048, 2048,
        2048, 1024, 2048, 2048, 2048, 1024, 2048, 2048,
        2048, 1024, 2048, 2048, 2048, 1024, 2048, 2048,
        2048, 1024, 2048, 2048, 2048, 1024, 2048, 2048,
        1024, 1024, 1024, 2048, 2048, 1024, 2048, 2048,
    }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            for ([_]u64{ 1000, 100_000 }) |seq| for ([_]u64{ 6 * pin_gib, 40 * pin_gib }) |avail| {
                g.add(chooseRequestPrefillChunk(c, seq, 4096, 8, avail, 2048, 0, .{}));
                g.add(chooseRequestPrefillChunk(c, seq, 4096, 8, avail, 2048, 1024, .{ .mtp_on = true }));
                g.add(requestPrefillPick(c, seq, 4096, 8, avail, false, .{}));
                g.add(requestPrefillPick(c, seq, 4096, 8, avail, true, .{}));
            };
        }
    }.run);
}

test "bill pin: rungWidth and cappedRung" {
    try pinConfigs("rungWidth", &.{
        4096, 2048, 8192, 8192, 2048, 8192, 4096, 2048,
        4096, 4096, 2048, 4096, 2048, 2048, 2048, 2048,
        2048, 2048, 1024, 1024, 1024, 1024, 1024, 1024,
        512, 512, 512, 512, 512, 512, 2048, 2048,
        8192, 2048, 2048, 8192, 2048, 2048, 4096, 2048,
        2048, 4096, 2048, 2048, 2048, 2048, 2048, 2048,
        1024, 1024, 1024, 1024, 1024, 1024, 512, 512,
        512, 512, 512, 512, 4096, 2048, 8192, 8192,
        2048, 8192, 4096, 2048, 4096, 4096, 2048, 4096,
        2048, 2048, 2048, 2048, 2048, 2048, 1024, 1024,
        1024, 1024, 1024, 1024, 512, 512, 512, 512,
        512, 512, 8192, 2048, 8192, 8192, 2048, 8192,
        4096, 2048, 4096, 4096, 2048, 4096, 2048, 2048,
        2048, 2048, 2048, 2048, 1024, 1024, 1024, 1024,
        1024, 1024, 512, 512, 512, 512, 512, 512,
    }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            for (PREFILL_CHUNK_LADDER) |rung| for ([_]bool{ false, true }) |gated| {
                g.add(rungWidth(c, 100_000, rung, gated));
                g.add(cappedRung(rung, 2048));
                g.add(cappedRung(rung, 0));
            };
        }
    }.run);
}

test "bill pin: prefillChunkCost and the adaptive width" {
    try pinConfigs("adaptivePrefillWidth", &.{
        2015830016, 5661327488, 512, 512, 0, 1, 512, 512,
        0, 1, 512, 1024, 0, 0, 2048, 2048,
        0, 0, 512, 1024, 0, 0, 2048, 4096,
        0, 0, 1, 1, 1843391200, 6699567840, 512, 512,
        0, 1, 512, 512, 0, 1, 512, 1024,
        0, 0, 2048, 2048, 0, 0, 512, 1024,
        0, 0, 2048, 4096, 0, 0, 1, 1,
        1705423360, 5489439232, 512, 512, 0, 1, 512, 512,
        0, 1, 512, 1024, 0, 0, 2048, 2048,
        0, 0, 512, 1024, 0, 0, 2048, 4096,
        0, 0, 1, 1, 3462708120, 14474447768, 512, 512,
        0, 1, 512, 512, 0, 1, 512, 1024,
        0, 0, 1024, 1024, 0, 1, 512, 1024,
        0, 0, 2048, 4096, 0, 0, 1, 1,
    }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            for ([_]u64{ 512, 4096 }) |width| g.add(prefillChunkCost(c, 8, width, 100_000));
            for ([_]u64{ pin_gib, 6 * pin_gib, 40 * pin_gib }) |headroom| for ([_]u32{ 512, 2048 }) |current| {
                var st = mtp_plan.AdaptiveWidthState{};
                g.add(adaptivePrefillWidth(c, 8, 100_000, headroom, current, 8192, &st));
                g.add(adaptivePrefillWidth(c, 8, 100_000, headroom, current, 8192, &st));
                g.add(st.supporting);
                g.add(st.ratcheted);
            };
            g.add(adaptivePrefillChunkEnabled(c));
            g.add(perRequestPrefillChunkEnabled(c));
        }
    }.run);
}

test "bill pin: admissionAvailable and prefillHeadroomFrom" {
    var got = PinGot{};
    got.add(admissionAvailable(100 * pin_gib, 60 * pin_gib));
    got.add(admissionAvailable(60 * pin_gib, 100 * pin_gib));
    got.add(prefillHeadroomFrom(100 * pin_gib, 60 * pin_gib, pin_gib));
    got.add(prefillHeadroomFrom(60 * pin_gib, 60 * pin_gib, pin_gib));
    try pinEq("admissionAvailable", &.{ 42949672960, 0, 41875931136, 0 }, got.slice());
}

test "bill pin: creditedAdmissionBill and its verdict" {
    try pinConfigs("creditedAdmissionBill", &.{
        10737418240, 12884901888, 5368709120, 3221225472, 1024, 1, 1, 0,
        2147483648, 12884901888, 10737418240, 5368709120, 3221225472, 1024, 0, 1,
        1, 2147483648, 12884901888, 11811160064, 5368709120, 3221225472, 1024, 0,
        1, 1, 2147483648, 0, 10737418240, 12884901888, 5368709120, 3221225472,
        1024, 1, 1, 0, 2147483648, 12884901888, 10737418240, 5368709120,
        3221225472, 1024, 0, 1, 1, 2147483648, 12884901888, 11811160064,
        5368709120, 3221225472, 1024, 0, 1, 1, 2147483648, 0,
        10737418240, 12884901888, 5368709120, 3221225472, 1024, 1, 1, 0,
        2147483648, 12884901888, 10737418240, 5368709120, 3221225472, 1024, 0, 1,
        1, 2147483648, 12884901888, 11811160064, 5368709120, 3221225472, 1024, 0,
        1, 1, 2147483648, 209273344, 10737418240, 12884901888, 0, 0,
        1024, 1, 1, 0, 0, 12884901888, 10737418240, 0,
        0, 1024, 0, 0, 2, 0, 12884901888, 11811160064,
        0, 0, 1024, 0, 0, 2, 0, 0,
    }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            for ([_][2]u64{ .{ 10 * pin_gib, 12 * pin_gib }, .{ 12 * pin_gib, 10 * pin_gib }, .{ 12 * pin_gib, 11 * pin_gib } }) |m| {
                const bill = creditedAdmissionBill(c, m[0], m[1], 5 * pin_gib, 3 * pin_gib, 1024);
                g.add(bill.needed);
                g.add(bill.available);
                g.add(bill.evictable);
                g.add(bill.reclaimable);
                g.add(bill.chunk);
                g.add(bill.fits());
                g.add(bill.fitsAfterEviction());
                g.add(admissionVerdict(bill));
                g.add(pinnedResidentBytes(bill));
            }
            g.add(admissionCommitBytes(c, 5000, 4096, 8));
        }
    }.run);
}

test "bill pin: admissionLogLevel" {
    var got = PinGot{};
    for ([_]bool{ false, true }) |first| for ([_]u64{ 5, 9, 10 }) |needed| got.add(admissionLogLevel(needed, 10, first));
    try pinEq("admissionLogLevel", &.{ 3, 3, 2, 2, 2, 2 }, got.slice());
}

test "bill pin: the refusal sentences" {
    const a = std.testing.allocator;
    const bill = AdmissionBill{ .needed = 13_664 << 20, .available = 13_374 << 20, .evictable = 7_802 << 20, .reclaimable = 2_000 << 20, .chunk = 512 };
    const plain = try memoryRefusalMessage(a, 5000, 13_664, 13_374, bill, false);
    defer a.free(plain);
    try std.testing.expectEqualStrings("Prompt (5000 tokens) requires ~13664MB GPU memory but only ~13374MB available. Reduce prompt size or use a smaller model.", plain);
    const cached = try memoryRefusalMessage(a, 5000, 13_664, 13_374, bill, true);
    defer a.free(cached);
    try std.testing.expectEqualStrings("Prompt (5000 tokens) requires ~13664MB GPU memory but only ~13374MB is available and the hot prefix cache holds ~7802MB more, all of which can be reclaimed (~2000MB — the shortfall is elsewhere), which is not enough. Reduce prompt size, lower --ctx-size, or use a smaller model.", cached);
    var buf: [256]u8 = undefined;
    try std.testing.expectEqualStrings(
        "Context 160000 does not fit available GPU memory at KV4, prefill chunk 1024. Maximum context: 64000 tokens. Reduce context or free memory.",
        memoryContextRefusalMessage(&buf, .{ .requested = 160_000, .maximum = 64_000, .needed = 1, .available = 1, .kv_bits = 4, .chunk = 1024 }),
    );
}

test "bill pin: requestContextRefusal" {
    const saved = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = saved;
    launch_state.server_config.max_context_size = 100_000;
    try pinConfigsRagged("requestContextRefusal", &.{
        1, 94096, 955581, 32212254720, 22548578304, 8, 1024, 1,
        94096, 1048576, 32212254720, 22548578304, 8, 1024, 0, 0,
    }, .{ 7, 7, 1, 1 }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            var exl3 = c.*;
            exl3.expert_layout = .exl3_k4;
            const bill = AdmissionBill{ .needed = 30 * pin_gib, .available = 20 * pin_gib, .reclaimable = pin_gib, .chunk = 1024 };
            const r = requestContextRefusal(&exl3, 90_000, 4096, trunk.KVQuantConfig.affine(8), bill, .{});
            g.add(r != null);
            if (r) |x| {
                g.add(x.requested);
                g.add(x.maximum);
                g.add(x.needed);
                g.add(x.available);
                g.add(x.kv_bits);
                g.add(x.chunk);
            }
        }
    }.run);
}

test "bill pin: the machine-limit arithmetic" {
    var got = PinGot{};
    const mb: u64 = 1 << 20;
    got.add(physicalMemoryCeiling(100 * pin_gib, 10 * pin_gib, 20 * pin_gib));
    got.add(physicalMemoryCeiling(100 * pin_gib, 90 * pin_gib, 20 * pin_gib));
    got.add(wiredLimitFloor(120_000 * mb, 128 * pin_gib, pin_gib));
    got.add(wiredLimitFloor(90 * pin_gib, 128 * pin_gib, pin_gib));
    got.add(wiredLimitFloor(0, 128 * pin_gib, pin_gib));
    got.add(ssdBudgetCeiling(0, 100 * pin_gib));
    got.add(ssdBudgetCeiling(110 * pin_gib, 100 * pin_gib));
    got.add(ssdBudgetAdmits(80 * pin_gib, 10 * pin_gib, 100 * pin_gib));
    got.add(ssdBudgetAdmits(95 * pin_gib, 10 * pin_gib, 100 * pin_gib));
    got.add(wiredLimitMbForBudget(80 * pin_gib, 10 * pin_gib));
    got.add(wiredUnplannedBytes(pin_gib, 0));
    got.add(wiredUnplannedBytes(pin_gib, 900 * mb));
    got.add(gpuFloorFrom(28_960 * mb, 28_280 * mb, 60 * mb));
    got.add(gpuCeilingWithWiredFloor(100 * pin_gib, 10 * pin_gib, 20 * pin_gib, 0));
    got.add(gpuCeilingWithWiredFloor(100 * pin_gib, 10 * pin_gib, 20 * pin_gib, 90 * pin_gib));
    got.add(cadenceInflightBytes(50));
    got.add(cadenceInflightBytes(1000));
    got.add(mlxDefaultMbPerBuffer("applegpu_g13s"));
    got.add(mlxDefaultOpsPerBuffer("applegpu_g14p"));
    got.add(mlxPerBufferFromEnv("1000", 50));
    got.add(mlxCacheLimitBytes(128 * pin_gib));
    got.add(mlxCacheLimitBytes(16 * pin_gib));
    got.add(wireLimitMbFor(100 * pin_gib));
    got.add(rowSsdBudgetGb(96 * pin_gib, 80));
    got.add(rowSsdBudgetGb(0, 80));
    got.add(effectiveSsmCheckpointStride(256, 32, false, 4 * pin_gib));
    got.add(effectiveSsmCheckpointStride(256, 0, true, 4 * pin_gib));
    try pinEq("machineLimits", &.{
        32212254720, 107374182400, 124755378176, 0, 0, 107374182400, 118111600640, 1,
        0, 92160, 1073741824, 1480589312, 650117120, 32212254720, 96636764160, 0,
        1120665600, 50, 20, 1000, 8589934592, 2147483648, 102400, 96,
        80, 256, 0,
    }, got.slice());
}

test "bill pin: the wire-limit sentences" {
    var buf: [384]u8 = undefined;
    var cmd: [192]u8 = undefined;
    try std.testing.expectEqualStrings("sudo sysctl iogpu.wired_limit_mb=120000 vm.global_user_wire_limit=125829120000 vm.user_wire_limit=125829120000", wireLimitsCommand(&cmd, 120_000));
    const cannot = userWireWarning(&buf, 117 * pin_gib, .{ 116 * pin_gib, 116 * pin_gib }, 0, 128 * pin_gib).?;
    try std.testing.expectEqualStrings("[mem] warning: this load wires ~126.7 GB but vm.global_user_wire_limit is 124.6 GB: prefill will run slower. This Mac cannot wire that much: use a smaller pack or a lower --ssd-budget-gb", cannot);
    const raise = userWireWarning(&buf, 100 * pin_gib, .{ 96 * pin_gib, 96 * pin_gib }, 0, 128 * pin_gib).?;
    try std.testing.expectEqualStrings("[mem] warning: this load wires ~108.4 GB but vm.global_user_wire_limit is 103.1 GB: prefill will run slower. Raise it: sudo sysctl iogpu.wired_limit_mb=103500 vm.global_user_wire_limit=108527616000 vm.user_wire_limit=108527616000 (resets at reboot)", raise);
    try std.testing.expect(userWireWarning(&buf, 10 * pin_gib, .{ 116 * pin_gib, 116 * pin_gib }, 0, 128 * pin_gib) == null);
}

test "bill pin: the streamed-load fit" {
    try pinConfigs("expertStreaming", &.{
        65536, 1006731264, 4493721600, 3154214912, 1, 0, 65536, 970615680,
        5018044000, 3118099328, 1, 0, 65536, 585941504, 4677861760, 2733425152,
        1, 0, 65536, 64630784, 8398125056, 2212114432, 1, 0,
    }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            var s = c.*;
            s.expert_streaming = true;
            s.expert_cache_bytes = 10 * pin_gib;
            s.expert_bounce_bytes = pin_gib;
            s.expert_workspace_bytes = pin_gib;
            s.expert_vision_encode_bytes = 2 * pin_gib;
            g.add(expertStreamingPlannedSeq(&s));
            g.add(expertStreamingPlannedKvBytes(&s));
            g.add(expertStreamingServingBytes(&s));
            g.add(expertStreamingBesideBudgetBytes(&s));
            g.add(expertStreamingConfigFits(&s, 100 * pin_gib, 5 * pin_gib, 20 * pin_gib));
            g.add(expertStreamingConfigFits(&s, 20 * pin_gib, 5 * pin_gib, 20 * pin_gib));
        }
    }.run);
}

test "bill pin: expertStreamingLoadFits" {
    var got = PinGot{};
    got.add(expertStreamingLoadFits(100 * pin_gib, 10 * pin_gib, 16 * pin_gib, 60 * pin_gib, pin_gib, pin_gib, 9 * pin_gib));
    got.add(expertStreamingLoadFits(80 * pin_gib, 10 * pin_gib, 16 * pin_gib, 60 * pin_gib, pin_gib, pin_gib, 9 * pin_gib));
    try pinEq("expertStreamingLoadFits", &.{ 1, 0 }, got.slice());
}

test "bill pin: the SSD ask and tier" {
    try pinConfigs("diskAsk", &.{
        21474836480, 0, 0, 1, 21474836480, 0, 21474836480, 0,
        0, 1, 21474836480, 0, 21474836480, 0, 0, 1,
        21474836480, 0, 21474836480, 0, 0, 1, 21474836480, 0,
    }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            const ask = diskAsk();
            g.add(ask.bytes);
            g.add(ask.explicit());
            g.add(ask.source == .model_settings);
            g.add(diskTierOn(c));
            g.add(diskBytesFor(c));
            g.add(resolvedPrefixCacheMem());
        }
    }.run);
}

test "a default context never exceeds the checkpoint's window, and --ctx-size or model-settings still win" {
    if (!archBuilt("dsv41")) return error.SkipZigTest;
    const t = std.testing;
    const original = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = original;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .max_position_embeddings = 262_144, .serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults) };
    try t.expectEqual(@as(u32, 262_144), autoContextFrom(1_048_576, config.contextCap(), config.defaultContext()));

    config.max_position_embeddings = 1_048_576;
    config.pinned_context = 1_000_000;
    launch_state.server_config.max_context_size = 0;
    try t.expectEqual(@as(u32, 1_000_000), getEffectiveContextLength(&config));
    config.ctx_override = 300_000;
    try t.expectEqual(@as(u32, 300_000), getEffectiveContextLength(&config));
    launch_state.server_config.max_context_size = 250_000;
    try t.expectEqual(@as(u32, 250_000), getEffectiveContextLength(&config));
}

test "a 32 GB Mac's lean Flash-Next budget advertises and admits an 80k prompt" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const saved_ctx = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = saved_ctx;
    launch_state.server_config.max_context_size = 0;
    const saved_disk = launch_state.prefix_cache_disk_bytes;
    defer launch_state.prefix_cache_disk_bytes = saved_disk;
    launch_state.prefix_cache_disk_bytes = 20 << 30;
    const saved_mtp = launch_state.configured_mtp;
    defer launch_state.configured_mtp = saved_mtp;
    launch_state.configured_mtp = false;
    var cfg = fixtures.qwen4MeasuredTestConfig();
    cfg.pinned_prefill_chunk = 512;
    const ceiling: u64 = 28976 << 20; // 30000 MiB wired limit less the 1 GiB margin
    const active: u64 = 27_378_205_274; // the measured lean pack's resident bytes
    const prompt: u64 = 80_000;
    const new_tokens: u32 = 16;
    // What the boot advertises: within the measured range the bill is the one reserve, no margin on top.
    var sized = cfg;
    sized.pinned_context = ssdFirstSessionTokensNow(&cfg, 8, ceiling, active, 512);
    try std.testing.expect(sized.pinned_context >= prompt + new_tokens);
    const budget = admissionAvailable(ceiling, active);
    try std.testing.expect(prefillNeededAtChunk(&sized, prompt, new_tokens, 8, 512, .{}) <= budget);
}

test "the SSD-first budget bills the FLOOR reserve, at the deployed pack's live numbers" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // `ssdFirstBudgetForLoad` short-circuits above `planHotCache`, so the floor-width reserve never ran on qwen4_exp.
    const t = std.testing;
    var cfg = fixtures.qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const active: u64 = 69_827 * MiB;
    const ceiling: u64 = 109_395 * MiB;

    const orig_ctx = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = orig_ctx;
    launch_state.server_config.max_context_size = 0;
    cfg.pinned_context = 0;
    per_request_chunk_override = true;
    defer per_request_chunk_override = null;
    const orig_source = settings_resolve.prefill_chunk.source;
    defer settings_resolve.prefill_chunk.source = orig_source;
    settings_resolve.prefill_chunk.source = .default;

    // The pure chunk resolver: `pinPrefillChunk` reads live memory.
    const chunk: u32 = @intCast(prefill_plan.effectivePrefillChunk(
        cfg.prefillScoreHeadDim(),
        cfg.num_attention_heads,
        0,
        cfg.has_sliding_window,
        cfg.isMoe(),
        cfg.longCtxGated(),
        cfg.pinned_prefill_chunk,
    ));
    try t.expect(perRequestPrefillChunkEnabled(&cfg));
    const floor_w = clampReserveWidth(&cfg, chunk);
    try t.expectEqual(PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1], floor_w);
    try t.expect(floor_w < chunk);

    const reserve_pinned = prefillTransientReserve(&cfg, kv_bits, chunk);
    const reserve_floor = prefillTransientReserve(&cfg, kv_bits, floor_w);
    try t.expect(reserve_floor < reserve_pinned);

    const ctx_kv: u64 = (kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits) +| statePerTokenBilled(&cfg)) *|
        ssdFirstSessionTokensNow(&cfg, kv_bits, ceiling, active, chunk);
    const ask: u64 = 10 * 1024 * MiB;

    // The idle allowance is the residual after the reserve, so the whole difference lands there.
    const now = ssdFirstPrefixCacheMem(ask, ceiling, active, ctx_kv, reserve_floor);
    const before = ssdFirstPrefixCacheMem(ask, ceiling, active, ctx_kv, reserve_pinned);
    try t.expect(now >= before);

    // The RAM arm reserves the same floor for the same pack.
    const plan = planHotCache(&cfg, kv_bits, ceiling, active, 1_048_576, sizerCtxKvBytes(&cfg, kv_bits), ask, chunk);
    try t.expectEqual(floor_w, plan.reserve_chunk);
    try t.expectEqual(reserve_floor, plan.reserve);

    // The session the floor holds is the one the server advertises.
    const advertised = autoContextFrom(memoryContextAt(&cfg, ceiling, active), cfg.contextCap(), cfg.defaultContext());
    try t.expectEqual(advertised, ssdFirstSessionTokensNow(&cfg, kv_bits, ceiling, active, chunk));
}

test "an auto boot advertises the session the SSD-first budget floor was billed for" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // The load-time bill and the pinned context must be one session: the SSD arm billed with
    // `cache_reserve = 0` while `pinAutoContext` sized against `CTX_SIZING_CACHE_RESERVE`, so
    // the floor held RAM for a larger session than the one advertised.
    const t = std.testing;
    var cfg = fixtures.qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const active: u64 = 69_827 * MiB;
    const ceiling: u64 = 109_395 * MiB;

    const original = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = original;
    launch_state.server_config.max_context_size = 0;
    cfg.pinned_context = 0;

    const chunk: u32 = @intCast(prefill_plan.effectivePrefillChunk(
        cfg.prefillScoreHeadDim(),
        cfg.num_attention_heads,
        0,
        cfg.has_sliding_window,
        cfg.isMoe(),
        cfg.longCtxGated(),
        cfg.pinned_prefill_chunk,
    ));
    const per_tok: u64 = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits) +| statePerTokenBilled(&cfg);
    const transient: u64 = prefillTransientReserve(&cfg, kv_bits, chunk);

    const advertised = autoContextFrom(memoryContextAt(&cfg, ceiling, active), cfg.contextCap(), cfg.defaultContext());
    const billed = ssdFirstSessionTokensNow(&cfg, kv_bits, ceiling, active, chunk);
    try t.expectEqual(advertised, billed);

    // And on a tighter box where the checkpoint cap does not bind.
    const tight: u64 = active + 24_000 * MiB;
    const billed_t = ssdFirstSessionTokensNow(&cfg, kv_bits, tight, active, chunk);
    try t.expectEqual(autoContextFrom(memoryContextAt(&cfg, tight, active), cfg.contextCap(), cfg.defaultContext()), billed_t);
    try t.expect(billed > 500_000);
    try t.expect(billed_t > 300_000);
    try t.expect(billed_t < billed);

    // The floor holds RAM for exactly the advertised session.
    const idle: u64 = 4 * 1024 * MiB;
    const floor_t = ssdFirstPrefixCacheMem(idle, tight, active, per_tok *| @as(u64, billed_t), transient);
    const floor_more = ssdFirstPrefixCacheMem(idle, tight, active, per_tok *| @as(u64, billed_t + 1024), transient);
    try t.expectEqual(per_tok * 1024, floor_more - floor_t);
}

test "the load-time session bill is billed at the boot's --kv-quant, not bf16" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    // Live: the SSD-first budget billed the session at 22,464 MB with `--kv-quant 8` and with
    // `off` alike (dense 29,952 B/tok): `defaultKvBits` asked `global_scheduler`, which `serve`
    // assigns only after the load has run its preflight.
    const t = std.testing;
    var cfg = fixtures.qwen4RequestTestConfig();
    cfg.mtp_override = false; // the MTP head's own term: `the load-time bill prices the MTP head ...`
    const MiB: u64 = 1 << 20;
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    try t.expectEqual(@as(u64, 2_304), statePerTokenBilled(&cfg));
    try t.expectEqual(@as(u64, 13_056), kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), 8));
    try t.expectEqual(@as(u64, 24_576), kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), 16));

    // An explicit `--ctx-size`, so the two numbers differ in the width and nothing else.
    const saved_ctx = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = saved_ctx;
    launch_state.server_config.max_context_size = 786_432;
    const ceiling: u64 = 109_395 * MiB;
    const active: u64 = 69_827 * MiB;
    const chunk: u32 = cfg.pinned_prefill_chunk;

    const saved_kv = launch_state.configured_kv_quant;
    defer launch_state.configured_kv_quant = saved_kv;

    // `--kv-quant 8`, no scheduler yet: the state the load runs in.
    launch_state.configured_kv_quant = trunk.KVQuantConfig.affine(8);
    try t.expectEqual(@as(u64, 8), defaultKvBits(&cfg));
    try t.expectEqual(
        (13_056 + 2_304) * 786_432 + cfg.qsaRingBytes(),
        ssdFirstSessionKvBytes(&cfg, defaultKvBits(&cfg), ceiling, active, chunk),
    );

    launch_state.configured_kv_quant = trunk.KVQuantConfig.dense;
    try t.expectEqual(@as(u64, 16), defaultKvBits(&cfg));
    try t.expectEqual(
        (24_576 + 2_304) * 786_432 + cfg.qsaRingBytes(),
        ssdFirstSessionKvBytes(&cfg, defaultKvBits(&cfg), ceiling, active, chunk),
    );
    launch_state.configured_kv_quant = null;
    try t.expectEqual(@as(u64, 8), defaultKvBits(&cfg));

    const idle: u64 = 10 * 1024 * MiB;
    const transient: u64 = prefillTransientReserve(&cfg, 8, chunk);
    const at8 = ssdFirstPrefixCacheMem(idle, ceiling, active, 13_824 * MiB, transient);
    const at16 = ssdFirstPrefixCacheMem(idle, ceiling, active, 22_464 * MiB, transient);
    try t.expectEqual(@as(u64, 8_640 * MiB), at16 -| at8);
}

test "an explicit --ctx-size boot never consults the session reserve" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // `--ctx-size` wins in the resolver's first branch; asserted as invariance in the reserve.
    const t = std.testing;
    var cfg = fixtures.qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const per_tok: u64 = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits) +| statePerTokenBilled(&cfg);

    const original = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = original;

    launch_state.server_config.max_context_size = 262_144;
    cfg.pinned_context = 0;
    for ([_]u64{ 0, 40_000 * MiB, 109_395 * MiB, 900_000 * MiB }) |ceiling| {
        for ([_]u64{ 0, 69_827 * MiB }) |active| {
            for ([_]u32{ 512, 1024, 4096, 8192 }) |chunk| {
                try t.expectEqual(@as(u32, 262_144), ssdFirstSessionTokensNow(&cfg, kv_bits, ceiling, active, chunk));
            }
        }
    }
    for ([_]u64{ 0, CTX_SIZING_CACHE_RESERVE, 99_000 * MiB }) |reserve| {
        try t.expectEqual(
            @as(u32, 262_144),
            resolvedContextForLoad(262_144, 0, 109_395 * MiB, 69_827 * MiB, reserve, 3 * 1024 * MiB, per_tok, cfg.contextCap(), cfg.defaultContext()),
        );
    }

    launch_state.server_config.max_context_size = 0;
    cfg.pinned_context = 131_072;
    try t.expectEqual(@as(u32, 131_072), ssdFirstSessionTokensNow(&cfg, kv_bits, 109_395 * MiB, 69_827 * MiB, 4096));
}

// Bar: the ring is billed once per slot, independent of prompt length.
test "a 512k QSA session drops the raw-key overbill and bills the ring once" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    // These pins price a cache with no SSD tier behind it: every checkpoint stays in RAM.
    const no_tier = TierGuard.pin(false, 0);
    defer no_tier.restore();
    const whole_rebuild = fixtures.qsaWholeRebuildGuard();
    defer whole_rebuild.deinit();
    const t = std.testing;
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    const cfg = fixtures.qwen4ExpOomConfig();
    const seq: u64 = 512 * 1024;
    const short = prefillRequestTerms(&cfg, 1024, 2048, 8, 4096, .{});
    const long = prefillRequestTerms(&cfg, seq, 2048, 8, 4096, .{});
    try t.expectEqual(cfg.qsaRingBytes(), short.qsa_ring_bytes);
    try t.expectEqual(cfg.qsaRingBytes(), long.qsa_ring_bytes);
    const reserved = @max(reservedCacheTokens(seq, 2048, 4096, getEffectiveContextLength(&cfg)), seq);
    const new_needed = prefillNeededAtChunk(&cfg, seq, 2048, 8, 4096, .{});
    const old_needed = new_needed +| (reserved * 3_072 * 5 / 4);
    const ceiling: u64 = 26_000 * 1024 * 1024;
    try t.expect(old_needed > ceiling);
    try t.expect(new_needed <= ceiling);
}

test "getEffectiveContextLength returns the PINNED value, not a fresh memory reading" {
    const original = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = original;
    launch_state.server_config.max_context_size = 0; // auto

    var config = model_mod.ModelConfig{};
    config.max_position_embeddings = 131072;
    config.num_attention_heads = 8;
    config.num_hidden_layers = 42;
    config.num_key_value_heads = 2;
    config.head_dim = 256;

    // Once pinned, the advertised context must NOT move — clients budget
    // against it, and a fresh memory reading drifts with system load.
    config.pinned_context = 12345;
    try testing.expectEqual(@as(u32, 12345), getEffectiveContextLength(&config));
    try testing.expectEqual(@as(u32, 12345), getEffectiveContextLength(&config));

    // ...but an explicit --ctx-size still wins.
    launch_state.server_config.max_context_size = 4096;
    try testing.expectEqual(@as(u32, 4096), getEffectiveContextLength(&config));
}

test "getEffectiveContextLength uses ctx_size override" {
    const original = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = original;

    launch_state.server_config.max_context_size = 4096;
    var config = model_mod.ModelConfig{};
    config.max_position_embeddings = 32768;
    try testing.expectEqual(@as(u32, 4096), getEffectiveContextLength(&config));
}

test "a warm append is billed the rows it ALLOCATES, not the rows it already holds" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const whole_rebuild = fixtures.qsaWholeRebuildGuard();
    defer whole_rebuild.deinit();
    // Bar: a checked-out restore credits the rows it holds even when the append outgrows the
    // buffer; the grow bills the new capacity plus one eval window of old buffers.
    const t = std.testing;
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    const mb: u64 = 1024 * 1024;
    const cfg = fixtures.qwen4ExpLive364kConfig();
    const kv_bits: u64 = 8;
    const seq: u64 = 448_604;
    const matched: u64 = 448_531;
    const chunk: u64 = 512;
    const max_tokens: u32 = @intCast(getEffectiveContextLength(&cfg) - seq);
    const kv_per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits);
    const reserved = @max(reservedCacheTokens(seq, max_tokens, chunk, getEffectiveContextLength(&cfg)), seq);

    // Pessimal reading of the entry: capacity exactly the rows it matched, so the append DOES
    // grow every attention layer.
    const grows = WarmPrefix{ .matched_tokens = matched, .capacity_tokens = matched, .will_donate = true };
    const g = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, grows);
    try t.expectEqual(matched *| kv_per_tok, g.shared_resident_bytes);
    try t.expectEqual((reserved - matched) *| kv_per_tok, g.reserved_kv_bytes +| (seq - matched) *| kv_per_tok);
    // ...plus one eval window of old buffers: qwen4_exp caches 12 of 48 layers, one per window.
    try t.expectEqual(@as(u32, 1), attnLayersPerEvalWindow(&cfg, trunk.Trunk.MOE_EVAL_EVERY_N_LAYERS));
    try t.expectEqual(@as(u64, 465), g.grow_coexist_bytes / mb);
    // The bill at the narrowest and a wider width.
    try t.expectEqual(@as(u64, 5_057), prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, grows) / mb);
    try t.expectEqual(@as(u64, 6_548), prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, 2048, grows) / mb);

    // A buffer the previous turn sized above this prompt but below the reservation is grown to
    // it before the prefill writes, one attention layer per eval: one window of old rows.
    const fits = WarmPrefix{ .matched_tokens = matched, .capacity_tokens = 450_048, .will_donate = true };
    const f = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, fits);
    try t.expect(reserved > 450_048);
    try t.expectEqual(450_048 *| kv_per_tok / 12, f.grow_coexist_bytes);
    try t.expectEqual(@as(u64, 5_059), prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, fits) / mb);

    // Byte-identical arms: a SHARED restore is copied whole by the first append, so it credits
    // nothing and bills no window either; a COLD prompt has nothing to credit.
    const shared = WarmPrefix{ .matched_tokens = matched, .capacity_tokens = matched, .will_donate = false };
    const sh = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, shared);
    try t.expectEqual(@as(u64, 0), sh.shared_resident_bytes);
    try t.expectEqual(@as(u64, 0), sh.grow_coexist_bytes);
    const cold = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, .{});
    try t.expectEqual(@as(u64, 0), cold.shared_resident_bytes);
    try t.expectEqual(@as(u64, 0), cold.grow_coexist_bytes);
}

test "an omitted max_tokens reserves a generation HEADROOM, not the rest of the context" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const t = std.testing;
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    // A 383,067-token prompt with no `max_tokens` was admitted by eviction and then refused on
    // the inference thread: `clampMaxTokens` made the reservation cover the whole context
    // (~13 GB for a generation that would never run). The reservation exists to stop buffer
    // coexistence during prefill, not to pre-buy a generation.
    const KVCache = trunk.KVCache;
    const MB: u64 = 1024 * 1024;
    const cfg = fixtures.qwen4ExpOomConfig();
    const ctx: u64 = getEffectiveContextLength(&cfg);
    try t.expectEqual(@as(u64, 1_048_576), ctx);
    const seq: u64 = 383_067;
    const chunk: u64 = 4096;
    const clamped: u64 = request_policy.clampMaxTokens(@intCast(request_policy.omittedMaxTokensDefault(&cfg)), @intCast(seq), @intCast(ctx));
    try t.expectEqual(ctx - seq, clamped); // 665,509 — the live number

    // The reservation is bounded by the headroom, not by the context.
    const reserved = KVCache.reservedTokens(seq, clamped, chunk, ctx);
    try t.expectEqual(seq + KVCache.RESERVE_GEN_HEADROOM + chunk, reserved);
    try t.expect(reserved < seq + ctx / 8);
    try t.expectEqual(seq + 512 + chunk, KVCache.reservedTokens(seq, 512, chunk, ctx));
    try t.expectEqual(ctx + chunk, KVCache.reservedTokens(ctx - 16, clamped, chunk, ctx));

    const kv_per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), 8);
    const terms = prefillRequestTerms(&cfg, seq, clamped, 8, chunk, .{});
    try t.expectEqual((KVCache.RESERVE_GEN_HEADROOM + chunk) * kv_per_tok, terms.reserved_kv_bytes);
    try t.expectEqual(reserved * statePerTokenBilled(&cfg), terms.state_bytes);

    // ...and the whole bill drops by more than 10 GB, from above the live headroom to under it.
    const ctx_bounded: PrefillRequestTerms = .{
        .reserved_kv_bytes = (ctx + chunk - seq) * kv_per_tok,
        .state_bytes = (ctx + chunk) * statePerTokenBilled(&cfg),
        .checkpoint_bytes = terms.checkpoint_bytes,
    };
    const bill = oomBillFor(&cfg, seq, chunk, terms);
    const old_bill = oomBillFor(&cfg, seq, chunk, ctx_bounded);
    try t.expect(old_bill > bill + 10 * 1024 * MB);
    try t.expect(old_bill > 28_000 * MB);
    try t.expect(bill < 30_000 * MB); // under the live "~37130MB available"
}

test "the tail-merge gate reads the ARCH, not the installed hook (serve installs it for everyone)" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // `serve` installs `prefill_chunk_adapt` unconditionally, so `chunk_width_hook != null` is
    // true on every arch and reading it as the gate put the scaled bound back everywhere. This
    // builds the slot state the way `serve` does and asserts the two answers differ.
    const t = std.testing;
    scheduler_mod.prefill_chunk_adapt = &adaptivePrefillWidthNow;
    defer scheduler_mod.prefill_chunk_adapt = null;
    scheduler_mod.prefill_chunk_widen_ok = &adaptivePrefillWidenStillFits;
    defer scheduler_mod.prefill_chunk_widen_ok = null;
    scheduler_mod.prefill_chunk_adaptive_enabled = &adaptivePrefillChunkEnabled;
    defer scheduler_mod.prefill_chunk_adaptive_enabled = null;

    const hook_installed = scheduler_mod.prefill_chunk_adapt != null;
    try t.expect(hook_installed);

    var other = fixtures.qwen4RequestTestConfig();
    other.model_type = "not_a_served_arch";
    const adaptive = slot_admission.adaptiveChunkWidthFor(&other);
    try t.expect(!adaptive);
    try t.expect(hook_installed != adaptive);

    // A slot without the adaptive hook keeps the flat bound.
    try t.expectEqual(prefill_plan.TAIL_MERGE_MAX, prefill_plan.tailMergeMaxFor(512, adaptive));
    try t.expectEqual(@as(usize, 812), prefill_plan.nextChunkEnd(0, 812, 512, false, 0, 0, adaptive));
    try t.expectEqual(@as(usize, 1), prefill_plan.prefillChunkCount(812, 512, false, 0, 0, adaptive));

    const cfg = fixtures.qwen4RequestTestConfig();
    per_request_chunk_override = true;
    defer per_request_chunk_override = null;
    adaptive_chunk_override = true;
    defer adaptive_chunk_override = null;
    const unpinned = explicitPrefillChunk() == 0;
    try t.expectEqual(unpinned, slot_admission.adaptiveChunkWidthFor(&cfg));
    if (unpinned) {
        try t.expectEqual(@as(usize, 64), prefill_plan.tailMergeMaxFor(512, slot_admission.adaptiveChunkWidthFor(&cfg)));
        try t.expectEqual(@as(usize, 512), prefill_plan.nextChunkEnd(0, 812, 512, false, 0, 0, slot_admission.adaptiveChunkWidthFor(&cfg)));
    }
    try t.expect(!slot_admission.adaptiveChunkWidthFor(null));
    scheduler_mod.prefill_chunk_adaptive_enabled = null;
    try t.expect(!slot_admission.adaptiveChunkWidthFor(&cfg));
    try t.expect(scheduler_mod.prefill_chunk_adapt != null); // ...with the hook still installed
}

test "a WARM turn is not billed for the prefix it is about to SHARE (786,707 @ 19,032 MB)" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const whole_rebuild = fixtures.qsaWholeRebuildGuard();
    defer whole_rebuild.deinit();
    const t = std.testing;
    // Live: a 786,707-token warm turn restored its own 786,676-token entry and was then refused
    // by name; the restore hands over the entry's buffers without a second copy, so those
    // ~14.5 GB were already inside `mlx_get_active_memory` and `needed` billed them again.
    const cfg = fixtures.qwen4ExpOomConfig();
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    const seq: u64 = 786_707;
    const chunk: u64 = 512; // the width the ladder came down to that evening
    const kv_bits: u64 = 8;
    const max_tokens: u32 = 200;
    const mb: u64 = 1024 * 1024;
    const available: u64 = 19_032 * mb;

    // 13,056 B of 8-bit KV and 5,376 B of state per token (one history copy + the score bank).
    // The credit prices KV only: the history is privatised on restore.
    const kv_per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits);
    try t.expectEqual(@as(u64, 13_056), kv_per_tok);
    try t.expectEqual(@as(u64, 768), cfg.qsaHistoryBytesPerToken());
    try t.expectEqual(@as(u64, 2_304), statePerTokenBilled(&cfg));

    // Cold: the refusal, reproduced.
    const cold = prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, .{});
    try t.expect(cold / mb > 18_000 and cold / mb < 23_000);
    try t.expect(cold > available);

    // Warm: the entry's buffers hold the capacity the previous turn reserved, which still covers
    // this turn's reservation, so nothing grows.
    const matched: u64 = 786_676;
    const capacity: u64 = 787_456;
    const reserved = reservedCacheTokens(seq, max_tokens, chunk, getEffectiveContextLength(&cfg));
    try t.expectEqual(@as(u64, 787_419), reserved);
    try t.expect(reserved <= capacity);

    // `will_donate`: this restore checked out its entry, so the buffers are the slot's own.
    const warm_prefix = WarmPrefix{ .matched_tokens = matched, .capacity_tokens = capacity, .will_donate = true };
    try t.expectEqual(matched, warm_prefix.creditedRows(reserved));

    // The credit is the handed-over rows at this arch's KV width and nothing else, inside the 5/4.
    const credit = matched * kv_per_tok;
    try t.expectEqual(@as(u64, 786_676 * 13_056), credit);
    try t.expect(credit < matched * (kv_per_tok + cfg.qsaHistoryBytesPerToken()));
    const warm = prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, warm_prefix);
    // Isolated against a shared restore of the same span: `checkpoint_bytes` follows `matched` too.
    const shared_same_span = prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, .{ .matched_tokens = matched, .capacity_tokens = capacity });
    try t.expect(shared_same_span < cold);
    try t.expectEqual(shared_same_span - credit * 5 / 4, warm);

    // The verdict flips: admitted outright, no eviction pass. Under-crediting is the safe direction.
    try t.expect(warm < available);
    try t.expect(warm / mb > 6_500 and warm / mb < 8_500);

    // A cold prompt on the same box keeps the old refusal.
    try t.expect(prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, .{}) > available);
    try t.expect(prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, .{ .matched_tokens = 0, .capacity_tokens = capacity, .will_donate = true }) > available);
}

test "a chain extension that OUTGROWS the resident entry is credited the rows it holds" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    // Past the restored capacity the cache grows per layer: the resident rows stay credited and
    // the grow bills the new capacity beyond the old plus one eval window of old buffers.
    const cfg = fixtures.qwen4ExpOomConfig();
    const seq: u64 = 1_047_556;
    const chunk: u64 = 512;
    const kv_bits: u64 = 8;
    const max_tokens: u32 = 200;
    const capacity: u64 = 787_456; // what the 786k turn reserved

    const reserved = reservedCacheTokens(seq, max_tokens, chunk, getEffectiveContextLength(&cfg));
    try t.expectEqual(@as(u64, 1_048_268), reserved);
    const extend = WarmPrefix{ .matched_tokens = 786_707, .capacity_tokens = capacity, .will_donate = true };
    try t.expect(reserved > capacity);
    try t.expect(extend.grows(seq));
    try t.expectEqual(@as(u64, 786_707), extend.creditedRows(reserved));
    const per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits);
    const ext_terms = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, extend);
    const window = attnLayersPerEvalWindow(&cfg, trunk.Trunk.MOE_EVAL_EVERY_N_LAYERS);
    try t.expectEqual(capacity * per_tok / cfg.attnCacheLayerCount() * window, ext_terms.grow_coexist_bytes);
    try t.expectEqual(
        prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, .{}) -
            (786_707 * per_tok - ext_terms.grow_coexist_bytes) * 5 / 4,
        prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, extend),
    );

    // A buffer that already covers the reservation credits the full shared prefix: the gate is capacity, not prompt length.
    const roomy = WarmPrefix{ .matched_tokens = 786_707, .capacity_tokens = 1_048_576, .will_donate = true };
    try t.expectEqual(@as(u64, 786_707), roomy.creditedRows(reserved));
    const credit = 786_707 * kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits);
    try t.expectEqual(
        prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, .{}) - credit * 5 / 4,
        prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, roomy),
    );
}

test "the warm KV credit fires ONLY where the restore checked its entry out" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const whole_rebuild = fixtures.qsaWholeRebuildGuard();
    defer whole_rebuild.deinit();
    const t = std.testing;
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    // `KVCache.restore` binds by refcount and mlx donates only at `use_count() == 1`, so on a
    // shared restore the first `writeAtOffset` copies the whole prefix: the credited bytes are
    // allocated. Only the checkout makes the credit true; it used to fire on `matched > 0` alone.
    const cfg = fixtures.qwen4ExpOomConfig();
    const seq: u64 = 786_707;
    const chunk: u64 = 512;
    const kv_bits: u64 = 8;
    const max_tokens: u32 = 200;
    const mb: u64 = 1024 * 1024;
    const matched: u64 = 786_676;
    const capacity: u64 = 787_456;
    const reserved = reservedCacheTokens(seq, max_tokens, chunk, getEffectiveContextLength(&cfg));
    const cold = prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, .{});

    // (1) qwen4_exp, SSD-first, full-prefix hit, switch on: the credit is real.
    const donating = (CheckoutCase{}).warm(matched, capacity);
    try t.expect(donating.will_donate);
    try t.expectEqual(matched, donating.creditedRows(reserved));
    const warm = prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, donating);
    // Baseline = a shared restore of the same span, so the two credits are not folded into one comparison.
    const shared_same_span = prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, .{ .matched_tokens = matched, .capacity_tokens = capacity });
    try t.expectEqual(shared_same_span - matched * kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits) * 5 / 4, warm);
    try t.expect(warm / mb > 6_500 and warm / mb < 8_500);

    // Every arm below is the same restore minus one conjunct of the checkout, and every one shares.
    const shared_arms = [_]WarmPrefix{
        // (2) a partial hit: the entry describes a prefix this prompt left.
        (CheckoutCase{ .shared = 700_000 }).warm(matched, capacity),
        // (3) a non-qwen4 arch: a checkout can never be taken there.
        (CheckoutCase{ .ssd_first = false }).warm(matched, capacity),
        // (4) restore by move off: the refcount share.
        (CheckoutCase{ .move_enabled = false }).warm(matched, capacity),
        // (5) a pending disk record shares the same buffers.
        (CheckoutCase{ .pending_disk = true }).warm(matched, capacity),
        // (6) no slot to hand the buffers to (the connection thread's shape).
        (CheckoutCase{ .has_slot = false }).warm(matched, capacity),
    };
    for (shared_arms, 0..) |arm, i| {
        errdefer std.debug.print("shared arm {d} credited\n", .{i});
        try t.expect(!arm.will_donate);
        try t.expectEqual(@as(u64, 0), arm.creditedRows(reserved));
        try t.expectEqual(shared_same_span, prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, arm));
    }
    // ...and `cold` is still the number a prompt with NO resident prefix pays,
    // which is what the `.{}` arm above is for.
    try t.expect(cold > shared_same_span);

    // Capacity is a CEILING on the credit, not a switch: a buffer holding 700,000 rows donates
    // them and the request is billed the growth on top.
    const outgrown = (CheckoutCase{}).warm(matched, 700_000);
    try t.expect(outgrown.will_donate);
    try t.expectEqual(@as(u64, 700_000), outgrown.creditedRows(reserved));
    try t.expect(outgrown.grows(seq));
    const kv_per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits);
    const og_terms = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, outgrown);
    // 31 tokens wide, BELOW `prefillEvalCadenceApplies`: one eval for the whole layer loop, so
    // every old buffer is alive at once and the coexistence term cancels the credit exactly.
    try t.expect(seq - matched < 32);
    try t.expectEqual(700_000 * kv_per_tok, og_terms.grow_coexist_bytes);
    try t.expectEqual(shared_same_span, prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, outgrown));
    // An append wide enough to take the cadence pays ONE eval window instead, and that is the
    // difference between the old bill and the new one.
    const wide = WarmPrefix{ .matched_tokens = 700_000, .capacity_tokens = 700_000, .will_donate = true };
    const wide_terms = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, wide);
    const window = attnLayersPerEvalWindow(&cfg, trunk.Trunk.MOE_EVAL_EVERY_N_LAYERS);
    try t.expect(window < cfg.attnCacheLayerCount());
    try t.expectEqual(700_000 * kv_per_tok / cfg.attnCacheLayerCount() * window, wide_terms.grow_coexist_bytes);
    try t.expect(prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, wide) < shared_same_span);
}

test "prefillRequestTerms: qwen4 MTP head KV is billed when MTP is on and zero when off" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    const saved_force = launch_state.server_config.default_force_mtp;
    defer launch_state.server_config.default_force_mtp = saved_force;
    launch_state.server_config.default_force_mtp = false;
    const saved_ov = trunk.Trunk.mtp_head_kv_quant_override;
    defer trunk.Trunk.mtp_head_kv_quant_override = saved_ov;
    trunk.Trunk.mtp_head_kv_quant_override = false;
    const saved_boot = launch_state.configured_kv_quant;
    defer launch_state.configured_kv_quant = saved_boot;
    launch_state.configured_kv_quant = trunk.KVQuantConfig.affine(8);
    var cfg = fixtures.qwen4ExpOomConfig();
    cfg.mtp_override = null;
    try t.expect(!forceMtpFor(&cfg));
    const seq: u64 = 200_000;
    const chunk: u64 = 4096;
    const max_tokens: u64 = 8192;
    const kv_bits: u64 = 8;
    try t.expectEqual(@as(u64, 2048), mtpHeadKvBytesPerToken(&cfg));
    trunk.Trunk.mtp_head_kv_quant_override = true;
    try t.expectEqual(@as(u64, 1088), mtpHeadKvBytesPerToken(&cfg));
    trunk.Trunk.mtp_head_kv_quant_override = false;
    const off = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, .{});
    try t.expectEqual(@as(u64, 0), off.mtp_head_kv_bytes);
    const on = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, .{ .mtp_on = true });
    const reserved = @max(reservedCacheTokens(seq, max_tokens, chunk, getEffectiveContextLength(&cfg)), seq);
    try t.expectEqual(reserved * 2048, on.mtp_head_kv_bytes);
    const n: u64 = cfg.attnCacheLayerCount();
    try t.expectEqual(cfg.qsaRingBytes() + cfg.qsaRingBytes() / n, on.qsa_ring_bytes);
    try t.expectEqual(reserved * (statePerTokenBilled(&cfg) + statePerTokenBilled(&cfg) / n), on.state_bytes);
    try t.expectEqual(cfg.qsaRingBytes(), off.qsa_ring_bytes);
    try t.expectEqual(reserved * statePerTokenBilled(&cfg), off.state_bytes);
    var other = cfg;
    other.model_type = "not_a_served_arch";
    const other_on = prefillRequestTerms(&other, seq, max_tokens, kv_bits, chunk, .{ .mtp_on = true });
    try t.expectEqual(@as(u64, 0), other_on.mtp_head_kv_bytes);
}

test "prefillRequestTerms: MTP head KV bills the boot scheme, not a request kv_quant override" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    const saved_ov = trunk.Trunk.mtp_head_kv_quant_override;
    defer trunk.Trunk.mtp_head_kv_quant_override = saved_ov;
    trunk.Trunk.mtp_head_kv_quant_override = true;
    const saved_boot = launch_state.configured_kv_quant;
    defer launch_state.configured_kv_quant = saved_boot;
    launch_state.configured_kv_quant = trunk.KVQuantConfig.affine(8);
    var cfg = fixtures.qwen4ExpOomConfig();
    cfg.mtp_override = null;
    const seq: u64 = 200_000;
    const chunk: u64 = 4096;
    const max_tokens: u64 = 8192;
    const request_bits: u64 = 4;
    const n: u64 = cfg.attnCacheLayerCount();
    const request_head = kvBytesPerTokenAtBits(cfg.kvBytesPerToken() / n, request_bits);
    try t.expectEqual(@as(u64, 576), request_head);
    try t.expectEqual(@as(u64, 1088), mtpHeadKvBytesPerToken(&cfg));
    const on = prefillRequestTerms(&cfg, seq, max_tokens, request_bits, chunk, .{ .mtp_on = true });
    const reserved = @max(reservedCacheTokens(seq, max_tokens, chunk, getEffectiveContextLength(&cfg)), seq);
    try t.expectEqual(reserved * 1088, on.mtp_head_kv_bytes);
}

test "a WARM append bills the checkpoints its prefill CAPTURES, not the prompt's" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    // These pins price a cache with no SSD tier behind it: every checkpoint stays in RAM.
    const no_tier = TierGuard.pin(false, 0);
    defer no_tier.restore();
    const whole_rebuild = fixtures.qsaWholeRebuildGuard();
    defer whole_rebuild.deinit();
    // Turn B: prompt 368,208, restored 364,478 by move, refused by 290 MB at width 512. 1,795 MB
    // of the bill was 32 SSM checkpoints; a warm prefill forwards a 3,730-token tail and captures
    // at most one, and the entry's own checkpoints are already resident and merged at commit.
    const t = std.testing;
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    const cfg = fixtures.qwen4ExpLive364kConfig();
    const mb: u64 = 1024 * 1024;
    const per_cp = cfg.ssmCheckpointBytes();
    const seq: u64 = 368_208;
    const matched: u64 = 364_478;
    const chunk: u64 = 512;
    // The stride follows the uncapped base chunk (`settings_resolve.prefill_chunk`), not the adaptive width.
    const saved_override = settings_resolve.prefill_chunk.value;
    defer settings_resolve.prefill_chunk.value = saved_override;
    settings_resolve.prefill_chunk.value = 8192;
    const stride: u64 = 8192;
    try t.expectEqual(stride, prefill_plan.effectiveSsmCheckpointStride(launch_state.ssm_checkpoint_stride, 8192));

    // Cold: `seq / stride` saturates at --ssm-checkpoint-max as before.
    try t.expectEqual(@as(u64, launch_state.ssm_checkpoint_max) * per_cp, retainedSsmCheckpointBytes(&cfg, seq, 0, chunk));
    // Warm: a 3,730-token append at stride 8192 crosses no boundary, so the end snapshot is the whole of it.
    try t.expectEqual(@as(u64, 1) * per_cp, retainedSsmCheckpointBytes(&cfg, seq, matched, chunk));
    // A chain extension long enough to fill the ring still bills the cap: the
    // credit is proportional to the SPAN, not a blanket warm discount.
    try t.expectEqual(@as(u64, launch_state.ssm_checkpoint_max) * per_cp, retainedSsmCheckpointBytes(&cfg, seq, 60_000, chunk));
    // A span shorter than the backoff window snapshots nothing.
    try t.expectEqual(@as(u64, 0), retainedSsmCheckpointBytes(&cfg, seq, seq - 4, chunk));

    // The bill term by term, from the same inputs `prefillMemoryNeeded` reads.
    const kv_bits: u64 = 8;
    const max_tokens: u32 = 418_224;
    // The restored buffer is smaller than the reservation, so it grows and the rows it holds
    // are still credited. This block keeps the original turn-B decomposition by pricing the
    // SHARED restore of the same span, whose credit is zero for the other reason.
    const turn_b_donating = WarmPrefix{ .matched_tokens = matched, .capacity_tokens = 365_568, .will_donate = true };
    const turn_b = WarmPrefix{ .matched_tokens = matched, .capacity_tokens = 365_568, .will_donate = false };
    const reserve_rows = reservedCacheTokens(seq, max_tokens, chunk, getEffectiveContextLength(&cfg));
    try t.expectEqual(seq + trunk.KVCache.RESERVE_GEN_HEADROOM + chunk, reserve_rows);
    try t.expect(reserve_rows > turn_b.capacity_tokens);
    try t.expectEqual(@as(u64, 0), turn_b.creditedRows(reserve_rows));
    try t.expectEqual(@as(u64, 364_478), turn_b_donating.creditedRows(reserve_rows));

    const kv_per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits);
    const terms = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, turn_b);
    // On a GDN MoE the envelope is the stream arm, not 3 x MLP.
    const mlp: u64 = 8 * chunk * @max(cfg.hidden_size, prefillFfnWidth(&cfg)) * 2;
    const envelope: u64 = @max(3 * mlp, mlp + chunk * prefillStreamBytesPerToken(&cfg));
    try t.expect(envelope > 3 * mlp);
    const gross =
        seq * kv_per_tok +
        terms.reserved_kv_bytes +
        terms.state_bytes +
        terms.checkpoint_bytes +
        terms.qsa_ring_bytes +
        2 * seq * cfg.num_key_value_heads * cfg.head_dim * 2 +
        envelope +
        PREFILL_RUNTIME_FLOOR_BYTES;
    // No score scratch (hd 256 is fused) and no dequant-GEMM weights (fwd 512
    // is below PREFILL_DQ_GEMM_MIN_M) at this width.
    try t.expect(trunk.prefillHeadDimFused(@intCast(cfg.prefillScoreHeadDim())));
    try t.expect(chunk < trunk.PREFILL_DQ_GEMM_MIN_M);
    const needed = prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, turn_b);
    try t.expectEqual(gross * 5 / 4 + qsaMaskBytes(&cfg, chunk, seq), needed);

    // AND THE VERDICT MOVES. 11,854 MB refused against 10,000 MB available;
    // the honest checkpoint term brings it to 9,789 MB, so the turn is served.
    const available: u64 = 10_000 * mb;
    try t.expect(needed < available);
    try t.expectEqual(@as(u64, 9_789), needed / mb);
    const before = gross - terms.checkpoint_bytes + @as(u64, launch_state.ssm_checkpoint_max) * per_cp;
    try t.expectEqual(@as(u64, 11_854), (before * 5 / 4 + qsaMaskBytes(&cfg, chunk, seq)) / mb);
    try t.expect((before * 5 / 4 + qsaMaskBytes(&cfg, chunk, seq)) > available);
}

test "a warm restore whose buffers hold the prompt but not the reservation bills one window of old rows" {
    if (!archBuilt("qwen4") or !archBuilt("mimo2")) return error.SkipZigTest;
    // seq <= C < R: the prefill grows each KV layer to R before it writes, one eval at a time
    // (`KVCache.growToReservation`), so only one window's C-row buffers are alive beside the new.
    const t = std.testing;
    const saved_disk = launch_state.prefix_cache_disk_bytes;
    defer launch_state.prefix_cache_disk_bytes = saved_disk;
    launch_state.prefix_cache_disk_bytes = 0;
    var q4 = fixtures.qwen4RequestTestConfig();
    q4.pinned_context = 1 << 20;
    const mimo = fixtures.mimoV2FlashBillConfig();
    const kv_bits: u64 = 8;
    const seq: u64 = 130_000;
    const matched: u64 = 125_000;
    const capacity: u64 = 131_072;
    const chunk: u64 = 2048;
    for ([_]*const model_mod.ModelConfig{ &q4, &mimo }) |cfg| {
        const kv = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits);
        const reserved = @max(reservedCacheTokens(seq, 32_000, chunk, getEffectiveContextLength(cfg)), seq);
        try t.expect(seq <= capacity and capacity < reserved);
        const window = capacity * kv / cfg.kvPerTokenLayerCount() * attnLayersPerEvalWindow(cfg, trunk.Trunk.MOE_EVAL_EVERY_N_LAYERS);
        try t.expect(window > 0 and window < capacity * kv);
        for ([_]bool{ true, false }) |donate| {
            const warm = WarmPrefix{ .matched_tokens = matched, .capacity_tokens = capacity, .will_donate = donate };
            const terms = prefillRequestTerms(cfg, seq, 32_000, kv_bits, chunk, warm);
            const kv_bill = seq * kv + terms.reserved_kv_bytes + terms.grow_coexist_bytes - terms.shared_resident_bytes;
            // A donated grow frees each old buffer after its window; a share's stay with the entry.
            const peak = if (donate) reserved * kv - capacity * kv + window else reserved * kv;
            try t.expect(kv_bill >= peak);
            try t.expectEqual(if (donate) window else 0, terms.grow_coexist_bytes);
        }
        // Nothing grows for a restore that already holds the reservation.
        const roomy = prefillRequestTerms(cfg, seq, 32_000, kv_bits, chunk, .{ .matched_tokens = matched, .capacity_tokens = reserved, .will_donate = true });
        try t.expectEqual(@as(u64, 0), roomy.grow_coexist_bytes);
    }
}

test "bill pin: the manual context and the KV scheme" {
    const saved = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = saved;
    try pinConfigs("manualContext", &.{
        0, 8, 8, 1, 262144, 0, 8, 8,
        1, 262144, 0, 8, 8, 1, 262144, 0,
        16, 0, 0, 262144,
    }, struct {
        fn run(c: *const PinCfg, g: *PinGot) void {
            g.add(manualContext(c));
            g.add(defaultKvBits(c));
            const kv = kvCacheFor(c);
            g.add(kv.config.bits);
            g.add(kv.config.scheme);
            g.add(getEffectiveContextLength(c));
        }
    }.run);
}

/// FP8 experts compute from bf16 copies of a layer's routed slots (`glm5_stream.fp8Routed`): at most
/// every expert, plus the gathered codes when the routed slots of a cache slab are not contiguous.
pub fn fp8ExpertScratchBytes(layout: expert_stream_mod.quant.Layout, geometry: expert_stream_mod.Geometry) u64 {
    if (layout != .fp8_individual) return 0;
    const codes = 3 *| @as(u64, geometry.hidden) *| geometry.intermediate;
    return @as(u64, geometry.experts) *| codes *| (2 + 1);
}

/// The expert fill peak a streamed load and its requests bill. Flash-Next's streamed prefill peaked no higher than
/// its resident one, the union workspace and bounce already inside the budget (engine-expert-streaming.md#budget).
/// DeepSeek fills preallocated slabs in place: its fill stages nothing beyond the workspace and bounce the budget holds.
pub fn billedFillPeakBytes(config: *const model_mod.ModelConfig, prefill_peak_bytes: u64) u64 {
    return if (config.longCtxGated() or config.isDsv41()) 0 else prefill_peak_bytes;
}

/// What a streamed load holds beside its cache and bounce buffers while it fills. A request bill never carries the
/// workspace: it is allocated at boot and already counted in the active memory a request reads.
pub fn loadStagingBytes(config: *const model_mod.ModelConfig, billed_fill_peak_bytes: u64, workspace_bytes: u64) u64 {
    return billed_fill_peak_bytes +| (if (config.isDsv41()) workspace_bytes else 0) +| config.expertPoolBytes();
}

/// The 2-bit lm_head copy a MiMo source trunk keeps, resident or streamed: the heads draft on it
/// and a greedy readout shortlists on it.
pub fn coarseHeadBytes(config: *const model_mod.ModelConfig) u64 {
    return archHook("coarseHead", u64, config, .{}) orelse 0;
}

/// The post-load residency bill the eviction gate reserves, in bytes: the
/// weights plus 10% headroom for KV / vision / drafter overhead.
pub fn gateEstimateBytes(bytes_on_disk: ?u64, num_hidden_layers: u32, hidden_size: u32) u64 {
    const base: u64 = if (bytes_on_disk) |b|
        b
    else
        @as(u64, num_hidden_layers) * @as(u64, hidden_size) * 4 * 4;
    return base + base / 10;
}

/// A boot `--model` entry carries no discovery `bytes_on_disk`, so the measured shard sum is
/// the answer before the layers x hidden guess (which read 1966080 for a ~64 GiB pack).
pub fn residentWeightBytes(billed: ?u64, bytes_on_disk: ?u64, measured_disk: u64, num_hidden_layers: u32, hidden_size: u32) u64 {
    if (billed) |b| return b;
    if (bytes_on_disk) |b| return b;
    if (measured_disk > 0) return measured_disk;
    return @as(u64, num_hidden_layers) * @as(u64, hidden_size) * 4 * 4;
}

pub fn expertStreamingGateBytes(resident_bytes: u64, cache_bytes: u64, fill_peak_bytes: u64, bounce_bytes: u64) u64 {
    return resident_bytes +| cache_bytes +| fill_peak_bytes +| bounce_bytes;
}

/// The draft head's coarse lm_head copy: built only with MTP on, from an affine head (a trellis head has no packed rows to shrink).
pub fn rerankCoarseBill(split: model_mod.ResidentSplit, config: *const model_mod.ModelConfig, mtp_on: bool, coarse_bits: u32) u64 {
    if (!mtp_on or split.mtp == 0 or coarse_bits == 0 or split.stq_head) return 0;
    return mtp_mod.rerankCoarseBytes(@intCast(config.vocab_size), @intCast(config.hidden_size), coarse_bits);
}

/// Which "available memory" figure the preflight should trust. On iOS the
/// host-wide number is meaningless — the OS keeps RAM full (file cache,
/// jetsam-evictable background apps) and will evict on our behalf, so the
/// per-PROCESS jetsam headroom (`os_proc_available_memory`, nonzero only on
/// iOS) is the figure that decides whether the load survives. Live bug: an
/// 8 GB iPhone reported ~4 GB host-free and the preflight refused a 3.6 GB
/// model that fit comfortably inside the ~6.4 GB process limit.
/// `gpu_limit` = Metal's working-set limit (0 = unknown): a lowered `iogpu.wired_limit_mb` binds
/// below free RAM, and weights past it OOM in warmup instead of refusing by name.
pub fn effectiveAvailableBytes(host_avail: u64, proc_avail: u64, gpu_limit: u64) u64 {
    const avail = if (proc_avail > 0) proc_avail else host_avail;
    return if (gpu_limit > 0) @min(avail, gpu_limit) else avail;
}

/// A just-exited process returns its Metal memory over seconds: while a short reading keeps
/// rising, poll every 500 ms; stop at enough, after 2 s without growth, or after 15 s.
pub fn settleAvailableBytes(first: u64, needed: u64, reader: anytype) u64 {
    var avail = first;
    var stalled: u32 = 0;
    var polls: u32 = 0;
    while (avail > 0 and avail < needed and stalled < 4 and polls < 30) : (polls += 1) {
        reader.sleep();
        const next = reader.read();
        stalled = if (next > avail) 0 else stalled + 1;
        avail = next;
    }
    return avail;
}

/// A resident Sushi load shares the working-set cap with already active model buffers. A declared
/// wired limit (`wired_floor`, 0 = none) lifts the free-RAM figure to what that limit leaves the
/// GPU, the same floor request admission reads.
pub fn effectiveSushiAvailableBytes(host_avail: u64, proc_avail: u64, gpu_limit: u64, active: u64, wired_floor: u64) u64 {
    const host = if (wired_floor > 0) @max(host_avail, wired_floor -| active) else host_avail;
    const available = effectiveAvailableBytes(host, proc_avail, gpu_limit);
    return if (gpu_limit > 0) @min(available, gpu_limit -| active) else available;
}

/// Free memory a load keeps beyond everything it bills: allocator slack and pages macOS has not yet handed back.
pub const LOAD_SAFETY_NET_BYTES: u64 = 1 << 30;

/// What a load's warmup allocates above its billed weights: the measured peak over the billed bytes at
/// each architecture's worst setting, rounded up (engine-memory-admission.md#load-warmup).
pub fn loadWarmupBytes(config: *const model_mod.ModelConfig, mtp_on: bool) u64 {
    const gib: u64 = 1 << 30;
    if (archHook("loadWarmup", u64, config, .{mtp_on})) |bytes| return bytes;
    // Flash-Next: reading the MTP head peaks about a GiB above its resident bytes, and quantizing the
    // drafter holds its dense linears beside the quantized ones until the load ends.
    return gib / 8 +| (if (mtp_on) gib * 9 / 8 else 0) +| config.dflash_load_transient_bytes;
}

/// A streamed load holds the cache, the whole-layer fill union and the bounce buffers beside the resident trunk.
pub fn streamedLoadRequirementBytes(resident_bytes: u64, cache_bytes: u64, fill_peak_bytes: u64, bounce_bytes: u64) u64 {
    return expertStreamingGateBytes(resident_bytes, cache_bytes, fill_peak_bytes, bounce_bytes) +| LOAD_SAFETY_NET_BYTES;
}

/// Free memory a load demands: its billed weights, then the larger of the warmup transient plus the safety net and
/// the requested context's admission bill (the warmup ends before any request allocates).
pub fn loadRequirementBytes(config: *const model_mod.ModelConfig, weights_bytes: u64, mtp_on: bool, context_bill: u64) u64 {
    if (config.expert_streaming) return streamedLoadRequirementBytes(weights_bytes, config.expert_cache_bytes, loadStagingBytes(config, config.expert_fill_peak_bytes, config.expert_workspace_bytes), config.expert_bounce_bytes);
    return weights_bytes +| @max(loadWarmupBytes(config, mtp_on) +| LOAD_SAFETY_NET_BYTES, context_bill);
}

/// Unknown figures (a failed memory query, an unmeasurable size) never block a load.
pub fn memInsufficientForLoad(weights_bytes: u64, avail_bytes: u64, required_bytes: u64) bool {
    if (weights_bytes == 0 or avail_bytes == 0) return false;
    return avail_bytes < required_bytes;
}

pub fn pinLoadConfigs(comptime name: []const u8, want: []const u64, comptime run: fn (*const model_mod.ModelConfig, *PinGot) void) !void {
    var fx = try model_load.LoadBillFixtures.init();
    defer fx.deinit();
    var dsv = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    dsv.dsv41 = try @import("dsv41/dsv41_config.zig").parseText(@import("dsv41/dsv41_config.zig").release_config);
    try pinRows(name, want, null, .{ &fx.qwen, &fx.mimo, &fx.glm, &dsv }, run);
}

test "residentWeightBytes: a boot entry without a disk hint reports its measured shards" {
    const gib: u64 = 1 << 30;
    try testing.expectEqual(64 * gib, residentWeightBytes(null, null, 64 * gib, 48, 2560));
    try testing.expectEqual(@as(u64, 97) * gib, residentWeightBytes(97 * gib, 64 * gib, 64 * gib, 48, 2560));
    try testing.expectEqual(@as(u64, 5), residentWeightBytes(null, 5, 64 * gib, 48, 2560));
    try testing.expectEqual(@as(u64, 48 * 2560 * 16), residentWeightBytes(null, null, 0, 48, 2560));
}

test "the Sushi resident bill leaves out the coarse rerank copy of a trellis head" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    var config = model_mod.ModelConfig{ .model_type = "qwen4_exp" };
    config.vocab_size = 128;
    config.hidden_size = 64;
    const affine: model_mod.ResidentSplit = .{ .trunk = 1000, .mtp = 100 };
    const trellis: model_mod.ResidentSplit = .{ .trunk = 1000, .mtp = 100, .stq_head = true };
    const coarse = mtp_mod.rerankCoarseBytes(128, 64, 3);
    try testing.expect(coarse > 0);
    try testing.expectEqual(coarse, rerankCoarseBill(affine, &config, true, 3));
    try testing.expectEqual(@as(u64, 0), rerankCoarseBill(trellis, &config, true, 3));
    try testing.expectEqual(@as(u64, 0), rerankCoarseBill(affine, &config, false, 3));
}

test "bf16 streaming registry gate uses resident plan not checkpoint disk size" {
    try testing.expectEqual(@as(u64, 81), expertStreamingGateBytes(10, 60, 5, 6));
    try testing.expect(expertStreamingGateBytes(15_000_000_000, 60_000_000_000, 5_033_164_800, 536_870_912) < 360_000_000_000);
}

test "load preflight waits for a prior process's memory to come back, and only while it does" {
    const gib: u64 = 1 << 30;
    const Fake = struct {
        seq: []const u64,
        i: usize = 0,
        fn read(f: *@This()) u64 {
            const v = f.seq[@min(f.i, f.seq.len - 1)];
            f.i += 1;
            return v;
        }
        fn sleep(_: *@This()) void {}
    };
    var rising = Fake{ .seq = &.{ 80 * gib, 95 * gib, 110 * gib } };
    try testing.expectEqual(@as(u64, 110 * gib), settleAvailableBytes(69 * gib, 100 * gib, &rising));
    var short = Fake{ .seq = &.{ 70 * gib, 70 * gib, 70 * gib, 70 * gib, 70 * gib, 200 * gib } };
    try testing.expectEqual(@as(u64, 70 * gib), settleAvailableBytes(69 * gib, 100 * gib, &short));
    try testing.expectEqual(@as(usize, 5), short.i);
    var never = Fake{ .seq = &.{200 * gib} };
    try testing.expectEqual(@as(u64, 120 * gib), settleAvailableBytes(120 * gib, 100 * gib, &never));
    try testing.expectEqual(@as(usize, 0), never.i);
}

test "Sushi quant memory subtracts existing resident buffers from the GPU limit" {
    const gib: u64 = 1 << 30;
    try testing.expectEqual(@as(u64, 54 * gib), effectiveSushiAvailableBytes(88 * gib, 0, 94 * gib, 40 * gib, 0));
    try testing.expectEqual(@as(u64, 30 * gib), effectiveSushiAvailableBytes(30 * gib, 0, 94 * gib, 40 * gib, 0));
    try testing.expectEqual(@as(u64, 0), effectiveSushiAvailableBytes(88 * gib, 0, 40 * gib, 40 * gib, 0));
    try testing.expectEqual(@as(u64, 88 * gib), effectiveSushiAvailableBytes(88 * gib, 0, 0, 40 * gib, 0));
}

test "a declared wired limit lifts the load preflight's free-RAM figure, never past the working set" {
    const gib: u64 = 1 << 30;
    // 32 GB box, iogpu.wired_limit_mb raised: file cache and other apps' pages read as gone,
    // but the declared limit says the GPU may claim the floor.
    try testing.expectEqual(@as(u64, 28 * gib), effectiveSushiAvailableBytes(25 * gib, 0, 29 * gib, 0, 28 * gib));
    // Resident buffers come off the floor, as they come off the working set.
    try testing.expectEqual(@as(u64, 18 * gib), effectiveSushiAvailableBytes(15 * gib, 0, 29 * gib, 10 * gib, 28 * gib));
    // More free RAM than the floor keeps the larger figure; the working set still caps it.
    try testing.expectEqual(@as(u64, 29 * gib), effectiveSushiAvailableBytes(40 * gib, 0, 29 * gib, 0, 28 * gib));
    // A floor above the working set never raises the cap.
    try testing.expectEqual(@as(u64, 29 * gib), effectiveSushiAvailableBytes(25 * gib, 0, 29 * gib, 0, 31 * gib));
}

test "effectiveAvailableBytes is capped by the GPU working-set limit" {
    const GB: u64 = 1024 * 1024 * 1024;
    try std.testing.expectEqual(36 * GB, effectiveAvailableBytes(98 * GB, 0, 36 * GB));
    try std.testing.expect(memInsufficientForLoad(70 * GB, effectiveAvailableBytes(98 * GB, 0, 36 * GB), 71 * GB));
}

test "effectiveAvailableBytes prefers the per-process jetsam headroom when present" {
    const GB: u64 = 1024 * 1024 * 1024;
    try std.testing.expectEqual(6 * GB, effectiveAvailableBytes(4 * GB, 6 * GB, 0)); // iOS: proc wins
    try std.testing.expectEqual(4 * GB, effectiveAvailableBytes(4 * GB, 0, 0)); // macOS: proc query = 0 → host
    try std.testing.expectEqual(@as(u64, 0), effectiveAvailableBytes(0, 0, 0)); // both unknown → 0 (never blocks)
}

test "a streamed load bills its trunk, expert cache, fill union and bounce buffers plus the net" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const gib: u64 = 1 << 30;
    const cfg = model_mod.ModelConfig{ .model_type = "qwen4_exp", .expert_streaming = true, .expert_cache_bytes = 12 * gib, .expert_fill_peak_bytes = 2 * gib, .expert_bounce_bytes = gib / 2 };
    try testing.expectEqual(6 * gib + 12 * gib + 2 * gib + gib / 2 + gib, loadRequirementBytes(&cfg, 6 * gib, true, 0));
    try testing.expectEqual(loadRequirementBytes(&cfg, 6 * gib, true, 0), streamedLoadRequirementBytes(6 * gib, 12 * gib, 2 * gib, gib / 2));
}

test "the exact load bill admits a MiMo load the flat 7 GiB headroom refused" {
    if (!archBuilt("qwen4") or !archBuilt("glm5")) return error.SkipZigTest;
    if (!archBuilt("mimo2")) return error.SkipZigTest;
    const gib: u64 = 1 << 30;
    var f = try model_load.LoadBillFixtures.init();
    defer f.deinit();
    const weights = 90 * gib;
    const available = 92 * gib;
    // The retired headroom demanded weights + min(weights / 8, 6 GiB) + 1 GiB.
    try testing.expect(memInsufficientForLoad(weights, available, weights + 7 * gib));
    const exact = loadRequirementBytes(&f.mimo, weights, true, 0);
    try testing.expect(!memInsufficientForLoad(weights, available, exact));
    try testing.expect(!memInsufficientForLoad(weights, exact, exact));
    try testing.expect(memInsufficientForLoad(weights, exact - 1, exact));
    // Unknown figures (a failed memory query, an unmeasurable size) never block a load.
    try testing.expect(!memInsufficientForLoad(0, available, exact));
    try testing.expect(!memInsufficientForLoad(weights, 0, exact));
}

test "the eviction gate bills weights plus 10% headroom" {
    try testing.expectEqual(@as(u64, 1100), gateEstimateBytes(1000, 0, 0));
    try testing.expectEqual(@as(u64, 0), gateEstimateBytes(0, 0, 0));
    // No bytes_on_disk → the layers × hidden × 16 fallback.
    const fallback: u64 = 32 * 4096 * 16;
    try testing.expectEqual(fallback + fallback / 10, gateEstimateBytes(null, 32, 4096));
}

test "bill pin: billedFillPeakBytes and loadStagingBytes" {
    try pinLoadConfigs("fillPeak", &.{ 0, 3221225472, 3221225472, 3221225472, 3221225472, 3221225472, 0, 6442450944 }, struct {
        fn run(c: *const model_mod.ModelConfig, g: *PinGot) void {
            g.add(billedFillPeakBytes(c, 3 << 30));
            g.add(loadStagingBytes(c, 3 << 30, 2 << 30));
        }
    }.run);
}

test "bill pin: loadWarmupBytes" {
    try pinLoadConfigs("loadWarmupBytes", &.{ 134217728, 1342177280, 268435456, 268435456, 562545152, 562545152, 134217728, 1342177280 }, struct {
        fn run(c: *const model_mod.ModelConfig, g: *PinGot) void {
            g.add(loadWarmupBytes(c, false));
            g.add(loadWarmupBytes(c, true));
        }
    }.run);
}

test "bill pin: loadRequirementBytes" {
    try pinLoadConfigs("loadRequirementBytes", &.{
        65632468992, 85899345920, 66840428544, 85899345920, 77309411328, 77309411328, 77309411328, 77309411328,
        65766686720, 85899345920, 65766686720, 85899345920, 77309411328, 77309411328, 77309411328, 77309411328,
        66060796416, 85899345920, 66060796416, 85899345920, 77309411328, 77309411328, 77309411328, 77309411328,
        66571993088, 66571993088, 66571993088, 66571993088, 79456894976, 79456894976, 79456894976, 79456894976,
    }, struct {
        fn run(c: *const model_mod.ModelConfig, g: *PinGot) void {
            var streamed = c.*;
            streamed.expert_streaming = true;
            streamed.expert_cache_bytes = 10 << 30;
            streamed.expert_bounce_bytes = 1 << 30;
            streamed.expert_workspace_bytes = 1 << 30;
            for ([_]*const model_mod.ModelConfig{ c, &streamed }) |cfg| for ([_]bool{ false, true }) |mtp| for ([_]u64{ 0, 20 << 30 }) |context|
                g.add(loadRequirementBytes(cfg, 60 << 30, mtp, context));
        }
    }.run);
}

test "bill pin: streamedLoadRequirementBytes and expertStreamingGateBytes" {
    var got = PinGot{};
    got.add(streamedLoadRequirementBytes(5 << 30, 10 << 30, 2 << 30, 1 << 30));
    got.add(expertStreamingGateBytes(5 << 30, 10 << 30, 2 << 30, 1 << 30));
    got.add(expertStreamingGateBytes(std.math.maxInt(u64), 1, 1, 1));
    try pinEq("streamedLoadRequirementBytes", &.{ 20401094656, 19327352832, 18446744073709551615 }, got.slice());
}

test "bill pin: rerankCoarseBill and the coarse head bytes" {
    try pinLoadConfigs("rerankCoarseBill", &.{
        0, 0, 0, 0, 0, 0, 278118400, 0,
        0, 0, 0, 0, 0, 0, 0, 273416192,
        0, 0, 0, 0, 0, 0, 0, 0,
        277544960, 0, 0, 0, 0, 0, 0, 0,
        0, 440509440, 0, 0,
    }, struct {
        fn run(c: *const model_mod.ModelConfig, g: *PinGot) void {
            const affine: model_mod.ResidentSplit = .{ .trunk = 1000, .mtp = 100 };
            const trellis: model_mod.ResidentSplit = .{ .trunk = 1000, .mtp = 100, .stq_head = true };
            for ([_]bool{ false, true }) |mtp| for ([_]u32{ 0, 3 }) |bits| {
                g.add(rerankCoarseBill(affine, c, mtp, bits));
                g.add(rerankCoarseBill(trellis, c, mtp, bits));
            };
            g.add(coarseHeadBytes(c));
        }
    }.run);
}

test "bill pin: the registry gate estimates" {
    var got = PinGot{};
    got.add(gateEstimateBytes(null, 48, 2560));
    got.add(gateEstimateBytes(64 << 30, 48, 2560));
    got.add(residentWeightBytes(null, null, 64 << 30, 48, 2560));
    got.add(residentWeightBytes(97 << 30, 64 << 30, 64 << 30, 48, 2560));
    got.add(residentWeightBytes(null, null, 0, 48, 2560));
    try pinEq("gateEstimateBytes", &.{ 2162688, 75591424409, 68719476736, 104152956928, 1966080 }, got.slice());
}

test "bill pin: the preflight's available figures" {
    var got = PinGot{};
    const gib: u64 = 1 << 30;
    got.add(effectiveAvailableBytes(98 * gib, 0, 36 * gib));
    got.add(effectiveAvailableBytes(4 * gib, 6 * gib, 0));
    got.add(effectiveSushiAvailableBytes(88 * gib, 0, 94 * gib, 40 * gib, 0));
    got.add(effectiveSushiAvailableBytes(25 * gib, 0, 29 * gib, 0, 28 * gib));
    got.add(memInsufficientForLoad(70 * gib, 36 * gib, 71 * gib));
    got.add(memInsufficientForLoad(0, 36 * gib, 71 * gib));
    try pinEq("effectiveAvailableBytes", &.{ 38654705664, 6442450944, 57982058496, 30064771072, 1, 0 }, got.slice());
}

test "bill pin: fp8ExpertScratchBytes" {
    var got = PinGot{};
    const geometry = expert_stream_mod.Geometry{ .layers = 4, .first_moe_layer = 1, .experts = 16, .hidden = 128, .intermediate = 64 };
    got.add(fp8ExpertScratchBytes(.fp8_individual, geometry));
    got.add(fp8ExpertScratchBytes(.mxfp4_individual, geometry));
    try pinEq("fp8ExpertScratchBytes", &.{ 1179648, 0 }, got.slice());
}
