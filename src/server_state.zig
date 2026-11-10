//! The process globals of the server: what `main.zig` and the settings API set at launch and the handlers, bills and the
//! scheduler read. Written before `serve` accepts a connection unless a field says otherwise.
const std = @import("std");
const settings_resolve = @import("settings_resolve.zig");
const model_settings = @import("model_settings.zig");
const trunk = @import("trunk.zig");
const scheduler_mod = @import("scheduler.zig");
const model_registry_mod = @import("model_registry.zig");
const vision_queue = @import("vision_queue.zig");
const disk_placement = @import("disk_placement.zig");
const kv_disk_writer = @import("kv_disk_writer.zig");

const ModelRegistry = model_registry_mod.ModelRegistry;

/// Global flag set by signal handler for graceful shutdown.
pub var shutdown_requested = std.atomic.Value(bool).init(false);

pub var completion_id_counter = std.atomic.Value(i64).init(0);

pub var max_request_bytes = std.atomic.Value(usize).init(@as(usize, model_settings.default_max_request_mb) << 20);

pub fn setMaxRequestMb(value: ?u32) void {
    max_request_bytes.store(@as(usize, value orelse model_settings.default_max_request_mb) << 20, .monotonic);
}

/// Optional global API key (`--api-key`). When set, every NON-LOOPBACK request
/// (i.e. from another machine over the network) except the `/health` probe and
/// CORS preflight requires the key — the OpenAI/Anthropic APIs AND the
/// metrics feed. Loopback (127.0.0.0/8, ::1) is TRUSTED and
/// exempt, so the local app + a local browser keep working with no credentials;
/// the key protects the surface that actually matters — network exposure
/// (`--host 0.0.0.0`). Accepted as `Authorization: Bearer <key>`, `x-api-key:
/// <key>` (Anthropic), HTTP Basic (browser pages — the key is the password), or
/// an `?api_key=` / `?key=` query param. null ⇒ fully open (default, unchanged).
pub var g_api_key: ?[]const u8 = null;

/// `--api-key-strict`: drop the loopback exemption, so the key is required
/// from 127.0.0.1 too (`/health` and CORS preflight stay open). For hosts
/// where localhost is NOT one trust domain — an embedding application that
/// generates a fresh key per start wants "only the process holding the key
/// can drive inference", and loopback exemption made any local process (and
/// any browser page via a simple no-preflight POST) a key holder. No effect
/// unless `--api-key` is set.
pub var g_api_key_strict: bool = false;

/// Tool-call ARGUMENT auto-correct. When true (default), parsed tool-call
/// arguments are coerced to the types the request's tool schema declares
/// (`tool_schema.coerceToolArgsToSchema` — e.g. Python's `False` → JSON `false`, an
/// `edits` array shipped as a string → a real array). `--no-tool-autocorrect`
/// turns it OFF: arguments pass through exactly as the model emitted them (still
/// valid JSON — the parse-repair + safety net always run; this gates ONLY the
/// schema-type coercion, the opinionated layer). Off means a weak model's
/// mistyped value reaches the client verbatim, which strict clients reject.
pub var g_tool_autocorrect: bool = true;

/// `--parent-pid`: a host runs this engine, so the host (not `/v1/update`) updates it.
pub var host_managed: bool = false;
/// The bind `serve` listens on; `/v1/update` checks the page's Origin against it.
pub var listen_host: []const u8 = default_host;
pub var listen_port: u16 = default_port;

pub const default_host: []const u8 = "127.0.0.1";
pub const default_port: u16 = @import("default_port.zig").port;

/// Server-level configuration. Single source of truth for all process-wide
/// defaults that handlers might consult. Populated once by `serve()` from
/// its CLI args; read-only afterwards (no synchronization required, the
/// values don't change for the server's lifetime).
pub const ServerConfig = struct {
    /// Starts an embedded client only after this process owns its listening socket.
    on_ready: ?*const fn (u16) anyerror!std.Thread = null,
    /// Starts a client that submits to the scheduler directly, once the model is loaded and the context is
    /// pinned. The server binds no socket; the client ends the run with `requestShutdown`.
    embedded_client: ?*const fn () anyerror!std.Thread = null,
    /// Maximum context size (0 = unlimited). `--ctx-size N`.
    max_context_size: u32 = 0,
    /// Request timeout in seconds (0 = no timeout). `--timeout N`.
    request_timeout_sec: u32 = 300,
    /// Default reasoning budget in tokens (-1 = unlimited).
    /// Defaults for sampling fields the request OMITS, set by `--temp` /
    /// `--top-p` / `--top-k` in serve mode (the macOS app passes its Settings
    /// values so external clients like Claude Code — which send no sampling
    /// params at all — inherit them). null = flag not given; resolution falls
    /// through to the model's generation_config.json recommendation, then the
    /// hardcoded fallback. Explicit request body fields always win. See
    /// `resolveSamplingDefault`.
    default_temperature: ?f32 = null,
    default_top_p: ?f32 = null,
    default_top_k: ?u32 = null,
    /// Default `max_tokens` for a request that omits the field (`--max-tokens N` in serve mode;
    /// 0 = flag not given). Agent clients send nothing, and an omitted budget is the rest of
    /// the window, which the admission bill reserves for. Read through `launchMaxTokensDefault`.
    default_max_tokens: u32 = 0,
    /// `--mtp`: force the native MTP head ON for MoE targets too. The
    /// per-request default is otherwise `sidecar loaded and !isMoe()` (the
    /// verify-forward expert-routing caution the drafter shares), which makes
    /// a MoE MTP checkpoint unreachable from any client that doesn't send
    /// `enable_mtp:true` in the body. Per-request `enable_mtp` still wins.
    default_force_mtp: bool = false,
};

/// Explicit launch choice, for the bills a load runs before `global_scheduler` is published.
pub var configured_mtp: ?bool = null;

/// Live process-wide config. Mutated by `serve()` at startup from its
/// parameters; never written from a handler after that. Public so main.zig
/// can supply defaults (notably the per-request override CLI flags don't
/// flow through `serve()` arguments).
pub var server_config: ServerConfig = .{};

/// Plan 01 Phase 2 — continuous-batching scheduler. Always non-null in
/// serve mode (set by `serve()`); null only before `serve()` runs. Every
/// inference request handler routes through this; the scheduler's
/// inference thread is the single mlx-call site.
pub var global_scheduler: ?*scheduler_mod.Scheduler = null;

/// The boot's explicit `--kv-quant`, published by `serve()` before `Scheduler.init` so a
/// load-time bill sees it (the scheduler does not exist yet during the load; asking it billed
/// the SSD-first session at bf16, ~8.6 GB of idle allowance lost at 786k). Null = unflagged.
pub var configured_kv_quant: ?trunk.KVQuantConfig = null;

/// Plan 05 — model registry. Always non-null in serve mode (set by
/// `serve()`); handleConnection resolves `model` body fields against this
/// per request via `ensureLoaded`/`release`.
pub var global_registry: ?*ModelRegistry = null;

/// `--edit on|off`: whether a new chat on the page starts with its write tools on. Each chat's
/// Edit chip then switches them for itself.
pub var g_web_edit: bool = false;

/// Module-level capacity for plan 03 hot prefix cache. main.zig writes this
/// from `--prefix-cache-entries N` before calling `serve()`.
// Default 32: agent clients (Claude Code, the app's agent loop) interleave
// requests from several conversation roots — main thread, subagents, title
// generation — and a single-entry cache gets its long system-prompt prefix
// evicted by every interleaved request, forcing a full re-prefill each
// turn. The count cap is pure retention metadata; the SSD byte budget
// (`prefixCacheDiskForLoad`) is what bounds the bytes. 0 disables.
pub var prefix_cache_capacity: u32 = 32;
/// RAM retention is opt-in: `--prefix-cache-mem` turns it on, `0` or `off` keeps it off.
pub var prefix_cache_ram_enabled: bool = false;

/// The RAM tier's byte budget, read only where RAM retention is on (`--prefix-cache-mem`). The cache
/// evicts LRU entries on commit until `current_kv_bytes + new_bytes <= prefix_cache_mem_bytes`. 0
/// disables the byte budget (the machine's headroom and the count cap from `--prefix-cache-entries`
/// still apply).
pub var prefix_cache_mem_bytes: u64 = PREFIX_CACHE_MEM_DEFAULT;
pub const PREFIX_CACHE_MEM_DEFAULT: u64 = 2 * 1024 * 1024 * 1024;
/// `.flag` once `--prefix-cache-mem` is given: the operator's number is used as given, even when it equals the default.
pub var prefix_cache_mem_source: settings_resolve.Source = .default;

/// `--prefix-cache-mem 0` (or `off`) is "no RAM retention".
pub fn ramRetentionForMemArg(mem_bytes: u64) bool {
    return mem_bytes > 0;
}

test "ramRetentionForMemArg: a zero budget keeps RAM retention off" {
    try std.testing.expect(!ramRetentionForMemArg(0));
    try std.testing.expect(ramRetentionForMemArg(1 << 30));
}

/// The SSD tier's byte ask (`--prefix-cache-disk`, 0 = off). Committed KV prefixes persist as chunked
/// safetensors under `~/.sushi/kv-cache/<model-fingerprint>` and are restored across restarts instead
/// of recomputed, LRU-evicted to the budget. Unnamed, the ask is the 20 GB ceiling and the tier is on;
/// `prefixCacheDiskForLoad` sizes the real budget per model (the Swift app's Settings toggle passes the flag).
pub var prefix_cache_disk_bytes: u64 = disk_placement.DEFAULT_DISK_CAP;
/// Who named the ask: `--prefix-cache-disk` (`.flag`) or `server.prefix_cache_disk_gb` (`.model_settings`) make the
/// number the budget as given; `.default` leaves it to the per-model sizing.
pub var prefix_cache_disk_source: settings_resolve.Source = .default;

/// The settings API's write. Publishes the bytes before a named source, so a reader that sees the source sees its number.
pub fn setDiskAsk(bytes: u64, source: settings_resolve.Source) void {
    if (source != .default) {
        @atomicStore(u64, &prefix_cache_disk_bytes, bytes, .monotonic);
        @atomicStore(settings_resolve.Source, &prefix_cache_disk_source, source, .release);
    } else {
        @atomicStore(settings_resolve.Source, &prefix_cache_disk_source, source, .release);
        @atomicStore(u64, &prefix_cache_disk_bytes, bytes, .monotonic);
    }
}

/// The SSD writer's permit, sized once from host RAM at startup (`kv_disk_writer.permitForRam`); the
/// scheduler arms every writer with it.
pub var disk_writer_permit_bytes: u64 = kv_disk_writer.DEFAULT_PERMIT_BYTES;

/// Phase 1 (performance-plan): SSM/conv state snapshot stride during prefill,
/// in tokens. Non-zero values enable multi-turn warm reuse on hybrid SSM
/// architectures (Flash-Next GatedDeltaNet). Set to 0 to disable (fall back to pre-Phase-1 behavior:
/// hybrid models bypass the hot prefix cache entirely). Override via
/// `--ssm-checkpoint-stride N`.
///
/// Default 256. The stride forces a prefill CHUNK boundary at every multiple
/// (to snapshot SSM/sliding-window state mid-prompt). On dense / non-MoE-hybrid
/// models prefill is compute-bound, so a fine 256 stride is ~free and buys
/// finer warm mid-prompt reuse. **MoE models are different**: their prefill is
/// memory-bound on the per-expert weights, and every extra chunk re-streams
/// ~all expert weights from HBM, so a fine stride silently costs cold prefill
/// on a MoE (an 850-token prompt = 4 chunks = ~4x expert-weight traffic). To
/// avoid that, the prefill loop
/// **coarsens this stride to >= PREFILL_CHUNK for MoE models only** (see
/// `generate.effectiveSsmCheckpointStride`), so MoE prefill is never
/// over-chunked at any prompt length while non-MoE keeps the fine stride.
/// The always-on end-of-prompt snapshot preserves append-growth multi-turn
/// reuse for MoE regardless. 0 disables (hybrid models then bypass the hot cache);
/// only tests move it off the default.
pub const SSM_CHECKPOINT_STRIDE_DEFAULT: u32 = 256;
pub var ssm_checkpoint_stride: u32 = SSM_CHECKPOINT_STRIDE_DEFAULT;

/// Phase 1: cap on the number of checkpoints retained per request. Snapshots
/// past this drop the oldest (front of list) to keep memory bounded on very
/// long prompts. 0 = unlimited (rely on the prefix-cache byte budget alone).
pub var ssm_checkpoint_max: u32 = 16;

/// Plan 01 — continuous batching: maximum concurrent in-flight requests sharing
/// the inference thread's batched-decode pass. Set via `--max-concurrent N`.
/// Default 1 = legacy single-slot behavior (no scheduler engagement, every
/// existing test bit-identical). Values >1 require Phase 2's scheduler wiring
/// (handler refactor + per-connection threads), which the `serve()` startup
/// guard checks before enabling.
pub var max_concurrent: u32 = 1;

/// How far under the enforced wired limit a plan may reach: past the limit Metal returns
/// zeros before an uncatchable abort, so a real transient's worth of margin stays unplanned.
pub const WIRED_LIMIT_MARGIN_BYTES: u64 = 1 << 30;

/// `--wired-margin-gib` in bytes: what stays unplanned under a raised wired limit.
pub var wired_limit_margin_bytes: u64 = WIRED_LIMIT_MARGIN_BYTES;

// Numeric slot diagnostics cross to error mapping only on this connection's thread.
pub threadlocal var slot_memory_context_refusal: ?model_registry_mod.MemoryContextRefusal = null;
pub threadlocal var slot_media_refusal: ?vision_queue.MediaRefusal = null;
