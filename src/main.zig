const std = @import("std");
const build_options = @import("build_options");
// pub: src/exl3 reaches mlx, log and io_util through its host root.
pub const mlx = @import("mlx.zig");
pub const io_util = @import("io_util.zig");
const model_mod = @import("model.zig");
const tokenizer_mod = @import("tokenizer.zig");
const transformer_mod = @import("transformer.zig");
const round_cost_mod = @import("round_cost.zig");
const generate_mod = @import("generate.zig");
const mtp_acceptance = @import("mtp_acceptance.zig");
const model_discovery = @import("model_discovery.zig");
const model_registry_mod = @import("model_registry.zig");
const mtp_mod = @import("mtp.zig");
const depth_bounds = @import("mtp_depth_bounds.zig");
const chat_mod = @import("chat.zig");
const server_mod = @import("server.zig");
const scheduler_mod = @import("scheduler.zig");
const expert_stream_mod = @import("expert_stream.zig");
const model_settings_mod = @import("model_settings.zig");
const vision_mod = @import("vision.zig");
const cli_mod = @import("cli.zig");
const kld_mod = @import("kld.zig");
const launch_mod = @import("launch.zig");
pub const log = @import("log.zig");
const metrics_mod = @import("metrics.zig");
const sleep_inhibit_mod = @import("sleep_inhibit.zig");
const version_mod = @import("version.zig");
const parent_watch = @import("parent_watch.zig");
const update_mod = @import("update.zig");

pub const VERSION: []const u8 = build_options.version;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

const DEFAULT_MODEL_DIR = ""; // pass --model <path> to specify

var expert_cache_bytes: u64 = 0;
var ssd_budget_bytes: u64 = 0;
// Serve-mode default for requests that omit max_tokens (0 = flag not given).
var serve_default_max_tokens: u32 = 0;

/// `sushi run` REPL thread: chats against the in-process server over
/// its own Ollama /api/chat endpoint, then brings the server down cleanly
/// (SIGTERM → the serve loop's shutdown path) when the user exits.
fn replThreadMain(allocator: std.mem.Allocator, io: std.Io, port: u16, opts: cli_mod.ReplOptions) void {
    cli_mod.runRepl(allocator, io, port, opts) catch |err| {
        log.warn("chat REPL exited: {s}\n", .{@errorName(err)});
    };
    std.posix.raise(std.posix.SIG.TERM) catch {};
}

const PromptClient = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    body: []const u8,
    stream: bool,
    err: ?anyerror = null,

    fn ready(port: u16) !std.Thread {
        return std.Thread.spawn(.{}, run, .{ prompt_client.?, port });
    }

    fn run(self: *PromptClient, port: u16) void {
        defer server_mod.requestShutdown();
        cli_mod.runPrompt(self.allocator, self.io, port, self.body, self.stream, server_mod.g_api_key) catch |err| {
            self.err = err;
        };
    }
};
var prompt_client: ?*PromptClient = null;

fn printUsage(io: std.Io) void {
    var stdout_buf: [4096]u8 = undefined;
    var stdout_w = std.Io.File.stdout().writer(io, &stdout_buf);
    stdout_w.interface.writeAll(
        \\sushi — MLX inference server for Apple Silicon
        \\
        \\Usage: sushi <command> [options]
        \\       sushi [options]
        \\
        \\Commands:
        \\  run <model>         Download if needed, serve it, and chat right here
        \\                      (a local model name or a HuggingFace "org/repo")
        \\                      --think <value>: GLM low|high|max (default high);
        \\                      Qwen off|low|medium|xhigh; MiMo off, or any
        \\                      other level as on.
        \\                      /think <effort> changes it in the chat;
        \\                      --tool on (or /tool on) lets the model search the
        \\                      web, fetch pages and read files in the current
        \\                      folder, read-only (off by default); --edit on (or
        \\                      /edit on) adds write_file and edit_file, still
        \\                      inside that folder and off by default; /cd
        \\                      <folder> moves that folder; /image <path> shows a
        \\                      vision model an image (a relative path is read
        \\                      from that folder). The prompt shows the folder and
        \\                      whether tools are on. The same chat opens in a
        \\                      browser at the URL it prints. /update installs
        \\                      a newer release and restarts the chat.
        \\  pull <model>        Download a model into ~/.sushi/models
        \\  list                Show downloaded models
        \\  serve               Start the server over ~/.sushi/models
        \\                      (every pulled model loads on demand by name)
        \\  launch <agent>      Configure + launch a coding agent CLI against the
        \\                      local server (claude, pi, omp, opencode, codex,
        \\                      hermes, aider, zcode). `sushi launch <agent> -h`
        \\                      for options
        \\  kld capture|compare Write a teacher fixture (full-vocab logits at
        \\                      every greedy position), or teacher-force one
        \\                      through a model and report KLD / top-1 / NLL.
        \\                      `sushi kld --help` for options
        \\  update              Replace this install with the newest release,
        \\                      after checking its SHA-256, its signature and
        \\                      that it runs; the old one stays as
        \\                      <install>.previous. --check only reports,
        \\                      --rollback swaps the previous one back.
        \\                      `sushi update --help` for options
        \\
        \\Options:
        \\  --model <dir>       Path to MLX model directory
        \\  --serve             Start HTTP server mode
        \\  --host <ip>         Bind address (default: 127.0.0.1 — this Mac only;
        \\                      0.0.0.0 opens it to the local network)
        \\  --port <n>          Bind port (default: 12345)
        \\  --ctx-size <n>      Maximum context length (default: model max)
        \\  --config-overrides <json>   JSON object deep-merged into EVERY
        \\                      model's config.json this process loads or
        \\                      discovers (alias: --hf-overrides). Generic keys
        \\                      like max_position_embeddings hit everything.
        \\                      HF attention_factor replaces the computed YaRN
        \\                      mscale; vLLM attn_factor multiplies it.
        \\                      e.g. '{"text_config":{"rope_parameters":{"rope_type":
        \\                      "yarn","factor":4.0,"original_max_position_embeddings":
        \\                      262144},"max_position_embeddings":1048576}}'
        \\  --embedding-max-length <n>  Per-input token ceiling for /v1/embeddings
        \\                      (default auto = the model's declared window; over-limit
        \\                      inputs get a 400 naming index/count/limit, never truncation)
        \\  --prompt, -p <text> Run one prompt and exit (also with run <model>)
        \\  --stream            Stream tokens as they are generated (with --prompt)
        \\  --max-tokens <n>    Max tokens to generate (default: 100); in --serve
        \\                      mode, the default for requests that omit the field
        \\  --temp <f>          Temperature. Offline: sampling temp (default 0.0).
        \\                      Serve: default for requests that omit `temperature`
        \\                      (otherwise the model's generation_config.json, then 1.0)
        \\  --top-p <f>         Serve-mode default top_p for requests that omit it
        \\                      (otherwise generation_config.json, then 1.0 = off)
        \\  --top-k <n>         Serve-mode default top_k for requests that omit it
        \\                      (otherwise generation_config.json, then 0 = off)
        \\  --timeout <n>       Stall timeout in seconds: abort a request after n seconds
        \\                      WITHOUT producing a token (default: 300, 0=none). A request
        \\                      that keeps generating never times out, however long it runs.
        \\  --reasoning-budget <n>  Max thinking tokens per request (default: unlimited)
        \\  --preserve-thinking on|off  Templates that read `preserve_thinking` (Qwen3.8)
        \\                      keep every turn's thinking (on, the template default)
        \\                      or only the latest user turn's (off). Request
        \\                      chat_template_kwargs > this flag > model-settings.json
        \\  --logit-bias-file <path> Experimental JSON/CSV scoped token penalties and rewards.
        \\  --think-penalty <f> Lower the logits of ~50 overthinking markers ("Wait",
        \\                      "But", "Alternatively", ...) by f inside the reasoning
        \\                      span (default 0 = off). Request think_penalty > this
        \\                      flag > model-settings.json
        \\  --no-vision         Disable vision encoder (saves memory)
        \\  --vision            Load the vision tower on an SSD-streamed load too (off by
        \\                      default there: its weights come out of the expert cache)
        \\  --no-prevent-sleep  Allow Mac idle sleep during inference and model
        \\                      loads. Display sleep is always allowed.
        \\  --skip-mem-preflight  Bypass the model-load free-RAM pre-flight that
        \\                        refuses a load whose weights + warmup headroom
        \\                        look too big for current free memory. The check
        \\                        is conservative (macOS reclaims file cache as
        \\                        MLX allocates); use this if a load you know fits
        \\                        is being refused. A genuine over-commit can
        \\                        hard-crash the server.
        \\  --pld               Enable Prompt Lookup Decoding (default: ON).
        \\                        Model-agnostic speculative decoding via n-gram
        \\                        matches in the prompt + generated tokens. Big
        \\                        wins on echo-heavy workloads (code editing, RAG,
        \\                        agentic loops). Adaptive prompt-time gate
        \\                        auto-disables it on novel content. Pass
        \\                        --no-pld to force-disable.
        \\  --no-pld            Force-disable standalone Prompt Lookup Decoding
        \\                        (the lookups inside MTP rounds stay on).
        \\  --no-mtp-lookup     Turn off the prompt-lookup drafts inside MTP
        \\                        rounds and GLM DFlash2 (default: ON); standalone
        \\                        PLD (--pld/--no-pld) is unaffected.
        \\  --pld-draft-len <n> Max draft tokens per PLD step (default: 5).
        \\  --pld-key-len <n>   N-gram match key length for PLD (default: 3).
        \\  --fast              MTP with typical acceptance and greedy tail, plus kv8: lossy
        \\                        for sampled requests, greedy requests unchanged.
        \\  --no-mtp            Disable the native MTP head. Both served models
        \\                        load it and run it by default.
        \\  --mtp               Force the MTP head ON. Both served models run it
        \\                        by default; an SSD-streamed pack loads with it
        \\                        off and refuses --mtp.
        \\  --mtp-head-kv-quant Quantize the qwen4 MTP head's own KV with
        \\                        --kv-quant (default OFF: the head keeps
        \\                        dense bf16 KV). No effect on MiMo, whose
        \\                        heads keep a dense sliding window.
        \\  --decode-attn-quant / --no-decode-attn-quant
        \\                      Serve decode from quantized side copies of
        \\                      DENSE (bf16/f16) attention projection weights:
        \\                      INT8 group-32 for most layers, NVFP4 for the
        \\                      last 20% (late layers amplify quantization
        \\                      error far less). Cuts their per-token weight
        \\                      read by half or more on models that ship dense
        \\                      attention.
        \\                      LOSSY: a real requantization, applied to
        \\                      decode/verify steps only; prefill keeps the
        \\                      dense weights. Default ON; --no-… restores
        \\                      exact dense decode. Env tuning:
        \\                      SUSHI_DECODE_ATTN_QUANT_NVFP4_FROM=<layer>
        \\                      moves the 4-bit boundary, =off keeps the whole
        \\                      stack INT8.
        \\  --mtp-min-depth <n> Fewest tokens an MTP round drafts, 1..8
        \\                        (default 1). Every depth the planner picks
        \\                        stays inside --mtp-min-depth..--mtp-max-depth;
        \\                        equal values pin one depth.
        \\  --mtp-max-depth <n> Most tokens an MTP round drafts, 1..8 (default:
        \\                        adaptive — the EV controller plans depth
        \\                        per round up to 8 on eligible M5 NAX targets,
        \\                        otherwise 6, lower on silicon with a measured
        \\                        verify-width cliff; SUSHI_MTP_ADAPTIVE=0
        \\                        reverts to the fixed windowed controller,
        \\                        cap 3). A --mtp-min-depth above that default
        \\                        lifts it. Each machine finds its own range.
        \\  --mtp-typical <d>  Opt-in lossy typical MTP acceptance (d > 0).
        \\                        Use 0.2 for the Qwen3.8 matched comparison.
        \\  --mtp-tokenv3 <a>  Opt-in lossy TokenV3 cascade (0 <= a <= 1).
        \\                        Alias: --mtp-cascade. Use 0.95 for the
        \\                        Qwen3.8 matched comparison. Exclusive with
        \\                        --mtp-typical; exact is the default.
        \\  --mtp-greedy-tail   Sampled requests draft only the first MTP token
        \\                        from the draft sampler, every later one by
        \\                        argmax. Pays beside --mtp-typical (faster,
        \\                        slightly more predictable text); greedy
        \\                        requests are unchanged. Default off.
        \\  --max-mtp-ctx <n>   Keep MTP speculative decoding OFF past <n>
        \\                        context tokens (default: 0 = no ceiling).
        \\                        A verify row is BYTES, so on a long-context
        \\                        trunk a round can cost more than the serial
        \\                        steps it replaces. A request whose prompt is
        \\                        past <n> decodes serially, and one that
        \\                        GENERATES past it switches mid-flight. The
        \\                        bound is inclusive (<n> itself still drafts)
        \\                        and it outranks `enable_mtp:true` in the
        \\                        request body. MTP only — PLD is unaffected.
        \\  --mtp-history-window <n>
        \\                      MTP prefill-history window: prompts forwarding
        \\                        more than 16384 tokens only build head history
        \\                        for the last <n> (default: 0 = full history;
        \\                        windowing costs acceptance on stock Qwen heads).
        \\  --kv-quant <mode>   KV-cache quantization scheme:
        \\                        8 or 4 — affine group quant; default 8 on
        \\                          every model (GLM refuses 4). 16 (or `off`)
        \\                          keeps dense bf16 KV. It outranks a
        \\                          model's model-settings.json `kv_quant`.
        \\                          Per-request override via the `kv_quant`
        \\                          body field.
        \\  --kv-attn-mode {{auto|dense|fused}}
        \\                      Decode read path for quantized KV. `dense`
        \\                        dequantizes K/V before SDPA; `fused` reads
        \\                        the packed cache in place at decode width
        \\                        (spec verify + prefill always read dense);
        \\                        `auto` (default) picks fused from 8K prompt
        \\                        tokens. Only effective at --kv-quant 4 or 8;
        \\                        per-request `kv_attn_mode` field overrides.
        \\                        MiMo's global layers read packed from 4096
        \\                        cached keys in every mode.
        \\  --prefill-chunk <n> Maximum tokens forwarded per prefill chunk
        \\                        (default: 8192). Auto-capped further per model
        \\                        so one layer's attention scores stay within
        \\                        budget; this flag is the ceiling, not a floor.
        \\                        When a request does not fit at <n>, it runs
        \\                        at the widest narrower width that fits (down
        \\                        to 512) instead of being refused.
        \\  --prefill-decode-share <s>
        \\                      Target decode wall-time share during another
        \\                        request's prefill (0..0.9); narrows chunks too.
        \\                        Default 0; env SUSHI_PREFILL_DECODE_SHARE.
        \\  --prefix-cache-entries <n>
        \\                      Hot prefix cache LRU capacity in entries
        \\                        (default: 32). 0 disables all prefix reuse.
        \\  --no-prefix-cache-ram
        \\                      Keep RAM retention off even when
        \\                        --prefix-cache-mem is given (RAM is off by
        \\                        default).
        \\  --prefix-cache-mem <n>{{KB,MB,GB}}
        \\                      Turn on RAM retention of idle prefix-cache
        \\                        entries with this KV-bytes budget (default:
        \\                        off; prefixes live on the SSD tier).
        \\                      Evicts LRU entries until the budget fits.
        \\                      Pass 0 (or off) to keep RAM retention off.
        \\  --prefix-cache-disk <n>{{KB,MB,GB}}
        \\                      SSD tier byte budget for the prefix cache
        \\                        (default: on, min(entries x context x bytes
        \\                        per token + 2GB, 20GB, free disk - 4GB)).
        \\                      Sizes are binary: 1GB = 1 GiB.
        \\                      Seen prefixes persist under ~/.sushi/kv-cache
        \\                        (SUSHI_PREFIX_CACHE_DIR) and are restored
        \\                        across restarts instead of recomputed. 0/off
        \\                        disables.
        \\  --ssm-checkpoint-stride <n>
        \\                      Hybrid SSM architectures only (Qwen3.8-Flash-Next's
        \\                        GDN layers): capture an SSM/conv state checkpoint every
        \\                        <n> tokens during chunked prefill, so a later
        \\                        request sharing a prefix can restore mid-prompt
        \\                        instead of re-prefilling (default: 256). 0
        \\                        disables capture — hybrid models then bypass the
        \\                        hot prefix cache entirely. On MoE targets the
        \\                        effective stride is raised to the prefill chunk,
        \\                        because each checkpoint forces a chunk boundary
        \\                        and every extra chunk re-streams the expert
        \\                        weights; see --prefill-chunk.
        \\  --ssm-checkpoint-max <n>
        \\                      Cap on SSM checkpoints retained per cache entry
        \\                        (default: 16). The first stride-aligned position
        \\                        is always kept; beyond the cap the oldest are
        \\                        dropped. 0 = unlimited, bounded only by the
        \\                        prefix cache's byte budget.
        \\  --wired-margin-gib <n>
        \\                      How far under iogpu.wired_limit_mb a plan may
        \\                        reach (default: 1, integers 1..32).
        \\  --expert-pick-tolerance <n>
        \\                      LOSSY, streamed packs only (default: 0 = off,
        \\                        exact routing). A routed expert missing from the
        \\                        cache is replaced by the best cached expert the
        \\                        router did not pick, when that expert's probability
        \\                        is at least (1 - n) x the missed one's. 0..0.6;
        \\                        0.3 is a mild setting. A sigmoid router (MiMo)
        \\                        compares sigmoid probabilities.
        \\  --tokenize-cache-entries <n>
        \\                      Per-model LRU cache of chat-template render +
        \\                        tokenize results (default: 4). Skips re-
        \\                        rendering identical messages on warm reuse.
        \\                        0 disables.
        \\  --expert-cache-gb <n>
        \\                      Stream routed experts from SSD (Sushi pack or
        \\                        source checkpoint) with a decimal-GB cache.
        \\  --ssd-budget-gb <n> Stream routed experts from SSD (Sushi pack or
        \\                        source checkpoint) with a
        \\                        TOTAL resident target of <n> GiB; the expert
        \\                        cache is what is left after the trunk, the
        \\                        prefill union and the fill buffers.
        \\                        --expert-cache-gb wins when both are given.
        \\  --model-dir <dir>   Directory of MLX models to discover at startup.
        \\                        Discovered siblings appear in /v1/models and
        \\                        can be loaded on-demand via /v1/load-model
        \\                        (or by sending a request with model=<id>).
        \\                        REPEATABLE (up to 8) — pass it once per folder
        \\                        your models live in. Scanned in order; the
        \\                        first folder wins a repeated model id, and a
        \\                        folder that can't be opened is skipped.
        \\  --max-resident-models <n>
        \\                      Maximum loaded models in memory (default: 3).
        \\                        ensureLoaded evicts LRU before exceeding.
        \\  --max-resident-mem <n>{{KB,MB,GB}}|auto
        \\                      Summed resident-bytes cap across all loaded
        \\                        models. Default 'auto' = 80% of MLX wired
        \\                        limit at startup. Pass 0 to disable.
        \\  --idle-evict-secs <n>
        \\                      Evict .ready entries with refcount==0 if
        \\                        idle for this many seconds. Default: off.
        \\  --gpu-warm-secs <n> Keep the GPU awake for this many seconds after
        \\                        the last request, so the next one starts
        \\                        without a wake-up delay (default: 60, 0 = off).
        \\  --metrics           Enable Prometheus metrics at GET /metrics (opt-in;
        \\                        zero cost when off). Also GET /metrics.json.
        \\  --no-tool-autocorrect
        \\                      Disable tool-call ARGUMENT auto-correct — the
        \\                        coercion of parsed args to the tool schema's
        \\                        declared types (e.g. Python `False` -> JSON
        \\                        `false`). Args then pass through as the model
        \\                        emitted them (still valid JSON). Default: on.
        \\  --api-key <token>   Require this key on every request (OpenAI/
        \\                        Anthropic/Ollama APIs + index page + metrics).
        \\                        Accepts Authorization: Bearer, x-api-key, HTTP
        \\                        Basic (key = password), or ?api_key=. /health
        \\                        stays open. Unset = no auth (default).
        \\  --api-key-strict    Require the key from loopback too (localhost is
        \\                        exempt by default). For embedders that want
        \\                        "only the key holder drives inference" on a
        \\                        shared machine. No effect without --api-key.
        \\  --api-key-env <VAR> Read the key from environment variable VAR
        \\                        instead of argv (the process table is
        \\                        world-readable). Unset/empty VAR = no auth.
        \\  --log-level <lvl>   Log level: error, warn, info, debug (default: info)
        \\  --log-file <path>   Persist the server log ("off" disables).
        \\                      Default: ~/.sushi/logs/sushi-<port>.log
        \\  --parent-pid <pid>  Shut down when process <pid> exits (for a host
        \\                        that runs sushi as its engine).
        \\  --no-update-check   Never ask GitHub for a newer release (a server
        \\                        otherwise checks at most once a day; also
        \\                        SUSHI_NO_UPDATE_CHECK=1)
        \\  --version           Print version and exit
        \\  --guest-manifest    Print the JSON a host checks before running this
        \\                        build as its engine (guest.json), and exit
        \\  --help              Show this help
        \\
    ) catch {};
    stdout_w.interface.flush() catch {};
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    // Bound MLX's reclaimable buffer pool before anything can allocate. ONCE,
    // above every subcommand branch — a per-serve-path call is how
    // `runHeadlessServe` (the mode the app always launches) silently ate the
    // --pld* flags. See server.mlxCacheLimitBytes for why MLX's own default
    // (~121 GB on a 128 GB Mac) is no defense.
    server_mod.applyMlxCacheLimit();
    // Resolve lazily-cached env reads on the main thread before other threads exist.
    @import("transformer.zig").warmQsaEnvCaches();
    @import("prefix_cache.zig").warmEnvCaches();

    // mlx-c's default handler exits the process; latch MLX failures instead (#353).
    mlx.installErrorHandler();

    // Materialize CLI args from the iterator API into a flat slice
    var args_iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args_iter.deinit();
    var args_list: std.ArrayList([]const u8) = .empty;
    defer {
        for (args_list.items) |a| allocator.free(a);
        args_list.deinit(allocator);
    }
    while (args_iter.next()) |arg| {
        try args_list.append(allocator, try allocator.dupe(u8, arg));
    }
    const args = args_list.items;
    // Runs after every other defer: an update asked for by /v1/update or the REPL replaces this process only once
    // the server has shut down the way a SIGTERM exit does.
    defer update_mod.relaunchIfRequested(allocator, io, args);

    if (args.len == 1) {
        printUsage(io);
        return;
    }

    // ── Subcommands (Ollama-grade CLI): `sushi run|pull|list|serve` ──
    // `pull` and `list` finish here; `run` and `serve` fall through into the
    // normal flag parse (skipping the consumed positionals) and serve path.
    var arg_start: usize = 1;
    var run_model_dir: ?[]u8 = null;
    defer if (run_model_dir) |d| allocator.free(d);
    var use_default_models_root = false;
    var repl_after_serve = false;
    var run_opts: cli_mod.ReplOptions = .{};
    if (args.len >= 2 and args[1].len > 0 and args[1][0] != '-') {
        const cmd = args[1];
        if (std.mem.eql(u8, cmd, "pull")) {
            if (args.len < 3) {
                log.err("usage: sushi pull <model>\n", .{});
                std.process.exit(1);
            }
            try cli_mod.cmdPull(allocator, io, args[2]);
            return;
        } else if (std.mem.eql(u8, cmd, "list")) {
            try cli_mod.cmdList(allocator, io);
            return;
        } else if (std.mem.eql(u8, cmd, "run")) {
            if (args.len < 3) {
                log.err("usage: sushi run <model> [options]\n", .{});
                std.process.exit(1);
            }
            run_model_dir = try cli_mod.ensureModelAvailable(allocator, io, args[2]);
            // `run` is the chat UX — refuse non-chat models up front instead
            // of booting a server whose chat surface can only 400.
            if (model_discovery.classifyModelPath(io, allocator, run_model_dir.?)) |kind| {
                if (kind != .chat) {
                    log.err("'{s}' is {s} — `sushi run` starts a chat REPL, which it can't serve.\n", .{ args[2], kind.describe() });
                    std.process.exit(1);
                }
            }
            arg_start = 3;
            use_default_models_root = true;
            repl_after_serve = std.Io.File.stdin().isTty(io) catch false;
        } else if (std.mem.eql(u8, cmd, "serve")) {
            arg_start = 2;
            use_default_models_root = true;
        } else if (std.mem.eql(u8, cmd, "launch")) {
            if (args.len < 3) {
                log.err("usage: sushi launch <agent> — supported: {s}\n", .{launch_mod.AgentKind.names});
                std.process.exit(1);
            }
            try launch_mod.cmdLaunch(allocator, io, args[2..]);
            return;
        } else if (std.mem.eql(u8, cmd, "kld")) {
            try kld_mod.cmdKld(allocator, io, args[2..]);
            return;
        } else if (std.mem.eql(u8, cmd, "update")) {
            try update_mod.cmdUpdate(allocator, io, args[2..]);
            return;
        } else {
            log.err("unknown command '{s}' (expected run, pull, list, launch, kld, update, or serve)\n", .{cmd});
            std.process.exit(1);
        }
    }

    var model_dir: []const u8 = DEFAULT_MODEL_DIR;
    var models_root: ?[]const u8 = null; // --model-dir for plan 05 discovery
    // Additional `--model-dir` folders, scanned after the first. Fixed-size:
    // a handful of library folders is the shape this serves, and a bound the
    // parser enforces beats an allocation the arg loop has to unwind.
    var extra_roots: [7][]const u8 = undefined;
    var extra_roots_n: usize = 0;
    var port_flag: ?u16 = null;
    var host_flag: ?[]const u8 = null;
    // `--log-file <path|off>`. null = default (`~/.sushi/logs/sushi-<port>.log`).
    var log_file_arg: ?[]const u8 = null;
    var parent_pid: ?std.posix.pid_t = null;
    var no_update_check = false;
    var serve_mode = false;
    var serve_explicit = false;
    var stream_mode = false;
    var prompt: ?[]const u8 = null;
    var max_tokens: u32 = 100;
    var temperature: f32 = 0.0;
    // Serve-mode sampling defaults for requests that omit the field
    // (request > flag > model generation_config.json > hardcoded). `--temp`
    // doubles as the offline --prompt sampling temp, so track whether it was
    // explicitly given — only then does it become the serve default.
    var temp_explicit = false;
    var top_p_flag: ?f32 = null;
    var top_k_flag: ?u32 = null;
    var ctx_size: u32 = 0; // 0 = use model default
    var timeout: u32 = 300; // seconds, 0 = no timeout
    var reasoning_budget: i32 = -1; // -1 = unlimited
    var no_vision = false;
    var vision = false;
    var enable_pld = true; // Prompt Lookup Decoding (on by default; --no-pld to disable)
    var pld_explicit = false;
    var pld_draft_len: u32 = 5;
    var pld_key_len: u32 = 3;
    var drafter_dir: ?[]const u8 = null; // Path to Gemma 4 assistant drafter checkpoint
    var no_drafter = false; // --no-drafter: never load one, merged-in ones included
    var draft_block_size: u32 = 4;
    var draft_block_size_explicit: bool = false; // user passed --draft-block-size?
    var enable_mtp = true; // Qwen native MTP head (auto when sidecar present; --no-mtp to disable)
    // --mtp: force the head ON where requests do not default to it: an SSD-streamed
    // pack and inherited MoE arches (server.defaultEnableMtp).
    var force_mtp = false;
    // Either flag given: it outranks the per-model `mtp` (the last one wins).
    var mtp_explicit = false;
    var mtp_head_kv_quant = false;
    var mtp_typical_raw: ?[]const u8 = if (std.c.getenv("SUSHI_MTP_TYPICAL")) |v| std.mem.span(v) else null;
    var mtp_tokenv3_raw: ?[]const u8 = if (std.c.getenv("SUSHI_MTP_TOKENV3")) |v| std.mem.span(v) else null;
    // Plan 04 Phase 1: pre-fault weights and pre-compile kernels at boot.
    // Default ON in serve mode — small boot-time cost, big cold-prefill win.
    // --no-warmup-eager opts out for benchmarking / minimal-footprint deployments.
    var warmup_eager: bool = true;
    var kv_quant_config: transformer_mod.KVQuantConfig = transformer_mod.KVQuantConfig.engine_default;
    var kv_quant_explicit = false;
    // Phase 2 (Plan ricky): fused attention reads K/V triples directly via
    // mlx_quantized_matmul instead of dequantizing through DenseKVView.
    // Off by default — only the `.affine` cache scheme has a fused path.
    var kv_attn_mode: server_mod.KvAttnMode = .auto;
    // Plan 05 Phase D: multi-model caps. Defaults aim for "comfortable on
    // 32–64 GB systems running Gemma 4 E4B-class models". Override via the
    // CLI flags below; the Swift app exposes them under Advanced settings.
    var max_resident_models: u32 = 3;
    var max_resident_mem: u64 = 0; // 0 = auto (80% of wired limit at startup)
    var max_resident_mem_explicit: bool = false;
    var idle_evict_secs: ?u32 = null;
    var metrics_enabled = false;
    var log_level_explicit = false;
    var decode_share_flag: ?[]const u8 = null;
    var no_prefix_cache_ram = false;
    var i: usize = arg_start;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--version")) {
            // Report app + every embedded engine version WITHOUT booting the
            // server (the macOS app spawns this and parses it — src/version.zig,
            // Swift EngineVersions). MLX self-reports at runtime; mlx-c has no
            // runtime API and rides build options.
            var mlx_ver = mlx.mlx_string_new();
            defer _ = mlx.mlx_string_free(mlx_ver);
            _ = mlx.mlx_version(&mlx_ver);
            const info = version_mod.Info{
                .app = VERSION,
                .mlx = std.mem.span(mlx.mlx_string_data(mlx_ver)),
                .mlx_c = build_options.mlx_c_version,
                .nax = transformer_mod.naxStatus(),
            };
            var ver_buf: [512]u8 = undefined;
            var ver_w = std.Io.File.stdout().writer(io, &ver_buf);
            version_mod.writeReport(&ver_w.interface, info) catch {};
            ver_w.interface.flush() catch {};
            return;
        } else if (std.mem.eql(u8, args[i], "--guest-manifest")) {
            var mlx_ver = mlx.mlx_string_new();
            defer _ = mlx.mlx_string_free(mlx_ver);
            try mlx.check(mlx.mlx_version(&mlx_ver));
            var out_buf: [1024]u8 = undefined;
            var out_w = std.Io.File.stdout().writer(io, &out_buf);
            try version_mod.writeGuestManifest(&out_w.interface, allocator, .{
                .version = VERSION,
                .commit = build_options.git_sha,
                .mlx = std.mem.span(mlx.mlx_string_data(mlx_ver)),
                .mlx_sha = build_options.mlx_sha,
                .mlx_c_sha = build_options.mlx_c_version,
                .min_macos = @import("builtin").os.version_range.semver.min,
            });
            try out_w.interface.flush();
            return;
        } else if (std.mem.eql(u8, args[i], "--help") or std.mem.eql(u8, args[i], "-h")) {
            printUsage(io);
            return;
        } else if (std.mem.eql(u8, args[i], "--model") and i + 1 < args.len) {
            i += 1;
            model_dir = args[i];
        } else if (std.mem.eql(u8, args[i], "--port") and i + 1 < args.len) {
            i += 1;
            port_flag = try std.fmt.parseInt(u16, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--host") and i + 1 < args.len) {
            i += 1;
            host_flag = args[i];
        } else if (std.mem.eql(u8, args[i], "--serve")) {
            serve_mode = true;
            serve_explicit = true;
        } else if (std.mem.eql(u8, args[i], "--think")) {
            const f = cli_mod.parseThinkFlag(if (i + 1 < args.len) args[i + 1] else null);
            if (!f.consumed) {
                log.err("--think requires a model-supported value: GLM low|high|max, Qwen off|low|medium|xhigh, MiMo off or any level (on)\n", .{});
                std.process.exit(1);
            }
            run_opts.think = f.think;
            if (f.consumed) i += 1;
        } else if (std.mem.eql(u8, args[i], "--tool") and i + 1 < args.len) {
            i += 1;
            run_opts.tools = cli_mod.parseToolSwitch(args[i]) orelse {
                log.err("--tool takes on or off, not '{s}'\n", .{args[i]});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, args[i], "--edit") and i + 1 < args.len) {
            i += 1;
            run_opts.edit = cli_mod.parseToolSwitch(args[i]) orelse {
                log.err("--edit takes on or off, not '{s}'\n", .{args[i]});
                std.process.exit(1);
            };
            server_mod.g_web_edit = run_opts.edit;
        } else if (std.mem.eql(u8, args[i], "--stream")) {
            stream_mode = true;
        } else if (cli_mod.isPromptFlag(args[i]) and i + 1 < args.len) {
            i += 1;
            prompt = args[i];
        } else if (std.mem.eql(u8, args[i], "--max-tokens") and i + 1 < args.len) {
            i += 1;
            max_tokens = try std.fmt.parseInt(u32, args[i], 10);
            serve_default_max_tokens = max_tokens;
        } else if (std.mem.eql(u8, args[i], "--temp") and i + 1 < args.len) {
            i += 1;
            temperature = try std.fmt.parseFloat(f32, args[i]);
            temp_explicit = true;
        } else if (std.mem.eql(u8, args[i], "--top-p") and i + 1 < args.len) {
            i += 1;
            top_p_flag = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, args[i], "--top-k") and i + 1 < args.len) {
            i += 1;
            top_k_flag = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--ctx-size") and i + 1 < args.len) {
            i += 1;
            ctx_size = try std.fmt.parseInt(u32, args[i], 10);
        } else if ((std.mem.eql(u8, args[i], "--config-overrides") or
            std.mem.eql(u8, args[i], "--hf-overrides")) and i + 1 < args.len)
        {
            // vLLM's `--hf-overrides` analogue. Applied the moment the flag is
            // seen, so no config.json is ever parsed without it — the loaded
            // model, the registry stubs and `/v1/models` all agree on the
            // resulting document. Validated here so a typo names the flag
            // instead of surfacing as a model-load parse error.
            i += 1;
            if (!configOverridesJsonValid(args[i])) {
                log.err("--config-overrides: expected a JSON object; got '{s}'\n", .{args[i]});
                std.process.exit(1);
            }
            model_mod.setConfigOverrides(args[i]);
            log.info("[args] config-overrides: {s}\n", .{args[i]});
        } else if (std.mem.eql(u8, args[i], "--embedding-max-length") and i + 1 < args.len) {
            i += 1;
            // Module global (like --max-concurrent): every serve path reads it,
            // so a hand-rolled ServerConfig can't eat it (the runHeadlessServe
            // class). "auto" = 0 = bound only by the model's declared window.
            server_mod.embedding_max_length = if (std.mem.eql(u8, args[i], "auto"))
                0
            else
                try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--timeout") and i + 1 < args.len) {
            i += 1;
            timeout = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--no-vision")) {
            no_vision = true;
        } else if (std.mem.eql(u8, args[i], "--vision")) {
            vision = true;
        } else if (std.mem.eql(u8, args[i], "--no-prevent-sleep")) {
            sleep_inhibit_mod.setEnabled(false);
        } else if (std.mem.eql(u8, args[i], "--skip-mem-preflight")) {
            scheduler_mod.skip_mem_preflight = true;
        } else if (std.mem.eql(u8, args[i], "--no-safety")) {
            // Accepted as a no-op so launchers that pass it keep booting.
        } else if (std.mem.eql(u8, args[i], "--pld")) {
            enable_pld = true;
            pld_explicit = true;
        } else if (std.mem.eql(u8, args[i], "--no-tool-autocorrect")) {
            server_mod.g_tool_autocorrect = false;
        } else if (cli_mod.isNoMtpLookupFlag(args[i])) {
            generate_mod.mtp_lookup_disabled = true;
        } else if (std.mem.eql(u8, args[i], "--no-pld")) {
            enable_pld = false;
            pld_explicit = true;
        } else if (std.mem.eql(u8, args[i], "--pld-draft-len") and i + 1 < args.len) {
            i += 1;
            pld_draft_len = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--pld-key-len") and i + 1 < args.len) {
            i += 1;
            pld_key_len = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--drafter") and i + 1 < args.len) {
            i += 1;
            drafter_dir = args[i];
        } else if (std.mem.eql(u8, args[i], "--draft-block-size") and i + 1 < args.len) {
            i += 1;
            draft_block_size = try std.fmt.parseInt(u32, args[i], 10);
            draft_block_size_explicit = true;
        } else if (std.mem.eql(u8, args[i], "--metrics")) {
            metrics_enabled = true;
        } else if (std.mem.eql(u8, args[i], "--api-key") and i + 1 < args.len) {
            i += 1;
            // Borrowed from argv (lives for the process). Empty ⇒ leave open.
            if (args[i].len > 0) server_mod.g_api_key = args[i];
        } else if (std.mem.eql(u8, args[i], "--api-key-strict")) {
            server_mod.g_api_key_strict = true;
        } else if (std.mem.eql(u8, args[i], "--api-key-env") and i + 1 < args.len) {
            i += 1;
            // The key read from a named environment variable instead of argv:
            // the process table is world-readable, argv with it. Same
            // borrow-for-the-process lifetime as --api-key; empty/unset
            // leaves the server open, exactly like an empty --api-key.
            // getenv needs a null-terminated name; args[i] is a plain
            // slice, so print a `:0` copy (process-lifetime, like the argv
            // borrow --api-key uses). std.c.getenv is how every other env
            // read in this codebase works. An unset var leaves the server
            // open, exactly like an empty --api-key.
            const name = std.fmt.allocPrintSentinel(allocator, "{s}", .{args[i]}, 0) catch null;
            if (name) |name_z| {
                if (std.c.getenv(name_z.ptr)) |value| {
                    const key = std.mem.span(value);
                    if (key.len > 0) server_mod.g_api_key = key;
                }
            }
        } else if (std.mem.eql(u8, args[i], "--no-drafter")) {
            no_drafter = true;
        } else if (std.mem.eql(u8, args[i], "--no-mtp")) {
            enable_mtp = false;
            force_mtp = false;
            mtp_explicit = true;
        } else if (std.mem.eql(u8, args[i], "--mtp")) {
            enable_mtp = true;
            force_mtp = true;
            mtp_explicit = true;
        } else if (std.mem.eql(u8, args[i], "--mtp-head-kv-quant")) {
            mtp_head_kv_quant = true;
        } else if (std.mem.eql(u8, args[i], "--dspark")) {
            // DSpark (DeepSeek-V4 draft stages) is OPT-IN: the stages cost
            // ~11 GB resident, so the default leaves them lazy and serves
            // serial. deepseek_v4.initModel reads the env at model load.
            _ = setenv("SUSHI_DSV4_DSPARK", "1", 1);
        } else if (std.mem.eql(u8, args[i], "--decode-attn-quant")) {
            transformer_mod.decode_attn_quant_flag = true;
        } else if (std.mem.eql(u8, args[i], "--no-decode-attn-quant")) {
            transformer_mod.decode_attn_quant_flag = false;
        } else if (std.mem.eql(u8, args[i], "--mtp-depth")) {
            log.err("{s}\n", .{depth_bounds.removed_flag_message});
            std.process.exit(1);
        } else if (std.mem.eql(u8, args[i], "--mtp-min-depth") or std.mem.eql(u8, args[i], "--mtp-max-depth")) {
            const is_min = std.mem.eql(u8, args[i], "--mtp-min-depth");
            i += 1;
            const text: []const u8 = if (i < args.len) args[i] else "";
            const n = depth_bounds.parseDepth(text, mtp_mod.MAX_DEPTH) catch {
                log.err("{s}: expected an integer in 1..{d}; got '{s}'\n", .{ args[i - 1], mtp_mod.MAX_DEPTH, text });
                std.process.exit(1);
            };
            if (is_min) depth_bounds.active.min = n else depth_bounds.active.max = n;
        } else if (std.mem.eql(u8, args[i], "--mtp-typical") and i + 1 < args.len) {
            i += 1;
            mtp_typical_raw = args[i];
        } else if ((std.mem.eql(u8, args[i], "--mtp-tokenv3") or std.mem.eql(u8, args[i], "--mtp-cascade")) and i + 1 < args.len) {
            i += 1;
            mtp_tokenv3_raw = args[i];
        } else if (std.mem.eql(u8, args[i], "--mtp-greedy-tail")) {
            generate_mod.mtp_greedy_tail_explicit = true;
        } else if (std.mem.eql(u8, args[i], "--fast")) {
            model_settings_mod.fast = true;
        } else if (std.mem.eql(u8, args[i], "--max-mtp-ctx") and i + 1 < args.len) {
            i += 1;
            generate_mod.max_mtp_ctx = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--mtp-history-window") and i + 1 < args.len) {
            i += 1;
            // 0 = full history; otherwise the last-N-token window applied
            // above mtp.HISTORY_WINDOW_THRESHOLD (set-once module override,
            // same contract as --prefill-chunk).
            generate_mod.mtp_history_window_override = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--reasoning-budget") and i + 1 < args.len) {
            i += 1;
            reasoning_budget = try std.fmt.parseInt(i32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--log-level") and i + 1 < args.len) {
            i += 1;
            if (log.Level.fromString(args[i])) |level| {
                log.setLevel(level);
                log_level_explicit = true;
            }
        } else if (std.mem.eql(u8, args[i], "--log-file") and i + 1 < args.len) {
            i += 1;
            log_file_arg = args[i];
        } else if (std.mem.eql(u8, args[i], "--parent-pid") and i + 1 < args.len) {
            i += 1;
            parent_pid = parent_watch.parseArg(args[i]) catch {
                log.err("--parent-pid: expected a process id above 1; got '{s}'\n", .{args[i]});
                std.process.exit(1);
            };
            server_mod.host_managed = true;
        } else if (std.mem.eql(u8, args[i], "--no-update-check")) {
            no_update_check = true;
        } else if (std.mem.eql(u8, args[i], "--warmup-eager")) {
            warmup_eager = true;
        } else if (std.mem.eql(u8, args[i], "--no-warmup-eager")) {
            warmup_eager = false;
        } else if (std.mem.eql(u8, args[i], "--prefill-chunk") and i + 1 < args.len) {
            i += 1;
            // `explicit` disables the machine-sized pin, so only a real width
            // earns it — a typo'd value keeps the defaults (flag-absent
            // behavior), never a silent 8192 that also switches sizing off.
            if (std.fmt.parseInt(usize, args[i], 10)) |v| {
                if (v > 0) {
                    generate_mod.prefill_chunk_override = v;
                    generate_mod.prefill_chunk_explicit = true;
                }
            } else |_| {}
        } else if (std.mem.eql(u8, args[i], "--prefill-trace")) {
            generate_mod.prefill_trace_force = true;
        } else if (std.mem.eql(u8, args[i], "--no-prefix-cache-ram")) {
            no_prefix_cache_ram = true;
            server_mod.prefix_cache_ram_enabled = server_mod.ramRetentionFor(server_mod.prefix_cache_mem_explicit, true);
        } else if (std.mem.eql(u8, args[i], "--prefix-cache-entries") and i + 1 < args.len) {
            i += 1;
            server_mod.prefix_cache_capacity = std.fmt.parseInt(u32, args[i], 10) catch 1;
        } else if (std.mem.eql(u8, args[i], "--prefix-cache-mem") and i + 1 < args.len) {
            // Wave 1.B — KV-bytes budget for the hot prefix cache. Accepts
            // bare numbers (bytes), a suffix of `MB`/`GB`/`KB` (case-
            // insensitive); `0`/`off` keeps RAM retention off.
            i += 1;
            server_mod.prefix_cache_mem_bytes = parseSizeArg(args[i]) catch {
                log.err("--prefix-cache-mem: expected '<n>{{MB,GB,KB}}' or '0'/'off'; got '{s}'\n", .{args[i]});
                std.process.exit(1);
            };
            server_mod.prefix_cache_mem_explicit = true;
            server_mod.prefix_cache_ram_enabled = server_mod.ramRetentionForMemArg(server_mod.prefix_cache_mem_bytes, no_prefix_cache_ram);
        } else if (std.mem.eql(u8, args[i], "--prefix-cache-disk") and i + 1 < args.len) {
            // SSD tier for the hot prefix cache: previously-seen prefixes are
            // persisted as chunked safetensors and restored across restarts
            // and RAM evictions instead of recomputed. Byte budget with the
            // same size grammar as --prefix-cache-mem; `0`/`off` disables.
            i += 1;
            server_mod.prefix_cache_disk_bytes = parseSizeArg(args[i]) catch {
                log.err("--prefix-cache-disk: expected '<n>{{MB,GB,KB}}' or '0'/'off'; got '{s}'\n", .{args[i]});
                std.process.exit(1);
            };
            server_mod.prefix_cache_disk_explicit = true;
        } else if (std.mem.eql(u8, args[i], "--logit-bias-file") and i + 1 < args.len) {
            i += 1;
            model_settings_mod.logit_bias_file_flag = args[i];
        } else if (std.mem.eql(u8, args[i], "--think-penalty") and i + 1 < args.len) {
            i += 1;
            const lambda = std.fmt.parseFloat(f32, args[i]) catch -1;
            if (!(lambda >= 0 and lambda <= model_settings_mod.think_penalty_max)) {
                log.err("--think-penalty: expected a number from 0 to {d}; got '{s}'\n", .{ model_settings_mod.think_penalty_max, args[i] });
                std.process.exit(1);
            }
            model_settings_mod.think_penalty_flag = lambda;
        } else if (std.mem.eql(u8, args[i], "--preserve-thinking") and i + 1 < args.len) {
            i += 1;
            model_settings_mod.preserve_thinking_flag = server_mod.parseOnOff(args[i]) orelse {
                log.err("--preserve-thinking: expected on or off; got '{s}'\n", .{args[i]});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, args[i], "--tokenize-cache-entries") and i + 1 < args.len) {
            // Iteration 2 (perf-plan Phase 4 #3): caps the per-LoadedModel
            // chat-template tokenize cache. 0 = off (every request re-
            // renders+re-tokenizes, mirrors pre-Iteration-2 behavior).
            i += 1;
            server_mod.tokenize_cache_entries = std.fmt.parseInt(u32, args[i], 10) catch 4;
        } else if (std.mem.eql(u8, args[i], "--ssm-checkpoint-stride") and i + 1 < args.len) {
            // Phase 1 (perf-plan): per-position SSM/conv state snapshots during
            // chunked prefill enable multi-turn warm reuse on hybrid SSM
            // architectures. 0 disables (legacy behavior: hybrid bypasses the
            // hot prefix cache); default 128.
            i += 1;
            server_mod.ssm_checkpoint_stride = std.fmt.parseInt(u32, args[i], 10) catch 128;
        } else if (std.mem.eql(u8, args[i], "--ssm-checkpoint-max") and i + 1 < args.len) {
            i += 1;
            server_mod.ssm_checkpoint_max = std.fmt.parseInt(u32, args[i], 10) catch 16;
        } else if (std.mem.eql(u8, args[i], "--wired-margin-gib") and i + 1 < args.len) {
            i += 1;
            server_mod.wired_limit_margin_bytes = server_mod.parseWiredMarginGib(args[i]) catch {
                log.err("--wired-margin-gib: expected an integer 1..32, got '{s}'\n", .{args[i]});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, args[i], "--expert-pick-tolerance") and i + 1 < args.len) {
            i += 1;
            expert_stream_mod.pick_tolerance = expert_stream_mod.parsePickTolerance(args[i]) catch {
                log.err("--expert-pick-tolerance: expected a number from 0 to 0.6, got '{s}'\n", .{args[i]});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, args[i], "--prefill-decode-share") and i + 1 < args.len) {
            i += 1;
            decode_share_flag = args[i];
        } else if (std.mem.eql(u8, args[i], "--max-concurrent") and i + 1 < args.len) {
            i += 1;
            server_mod.max_concurrent = std.fmt.parseInt(u32, args[i], 10) catch 1;
        } else if (std.mem.eql(u8, args[i], "--model-dir") and i + 1 < args.len) {
            // REPEATABLE. A user's library can live in more than one place (the
            // app's download folder, an external drive, an LM Studio tree), and
            // with one root the others are invisible to /v1/models even though
            // the picker lists them. Extras past the cap are refused loudly —
            // silently dropping a folder the user asked us to scan is the
            // silent-flag-eater class.
            i += 1;
            if (models_root == null) {
                models_root = args[i];
            } else if (extra_roots_n < extra_roots.len) {
                extra_roots[extra_roots_n] = args[i];
                extra_roots_n += 1;
            } else {
                log.err("--model-dir: at most {d} folders (got one more: {s})\n", .{ extra_roots.len + 1, args[i] });
                std.process.exit(1);
            }
        } else if (std.mem.eql(u8, args[i], "--max-resident-models") and i + 1 < args.len) {
            // Plan 05 Phase D: cap on .ready entries in the registry.
            // ensureLoaded evicts LRU before loading when this would be exceeded.
            i += 1;
            max_resident_models = std.fmt.parseInt(u32, args[i], 10) catch 3;
            if (max_resident_models == 0) max_resident_models = 1;
        } else if (std.mem.eql(u8, args[i], "--max-resident-mem") and i + 1 < args.len) {
            // Plan 05 Phase D: cap on summed resident bytes. Accepts the
            // same suffixes as --prefix-cache-mem. Special string "auto"
            // (or default 0) → 80% of mlx_set_wired_limit at server start.
            i += 1;
            if (std.mem.eql(u8, args[i], "auto")) {
                max_resident_mem = 0;
            } else {
                max_resident_mem = parseSizeArg(args[i]) catch {
                    log.err("--max-resident-mem: expected '<n>{{MB,GB,KB}}' or 'auto'; got '{s}'\n", .{args[i]});
                    std.process.exit(1);
                };
                max_resident_mem_explicit = true;
            }
        } else if (std.mem.eql(u8, args[i], "--gpu-warm-secs") and i + 1 < args.len) {
            i += 1;
            scheduler_mod.gpu_warm_secs = std.fmt.parseInt(u32, args[i], 10) catch {
                log.err("--gpu-warm-secs: expected a whole number of seconds, got '{s}'\n", .{args[i]});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, args[i], "--idle-evict-secs") and i + 1 < args.len) {
            // Idle eviction window. When set, `server.idleEvictLoop` unloads
            // .ready entries (refcount==0) whose last_used_ms is older than
            // this. Default off — eviction is on-demand only.
            i += 1;
            const n = std.fmt.parseInt(u32, args[i], 10) catch 0;
            idle_evict_secs = if (n > 0) n else null;
        } else if (std.mem.eql(u8, args[i], "--kv-quant") and i + 1 < args.len) {
            i += 1;
            kv_quant_explicit = true;
            kv_quant_config = transformer_mod.KVQuantConfig.fromJsonValue(.{ .string = args[i] }) orelse {
                log.err("--kv-quant: expected one of {{16, 8, 4}} (16 = BF16, also `off`); got '{s}'\n", .{args[i]});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, args[i], "--expert-cache-gb") and i + 1 < args.len) {
            i += 1;
            expert_cache_bytes = server_mod.parseExpertCacheGb(args[i]) catch {
                log.err("--expert-cache-gb: expected an integer in 1..360; got '{s}'\n", .{args[i]});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, args[i], "--ssd-budget-gb") and i + 1 < args.len) {
            i += 1;
            ssd_budget_bytes = server_mod.parseSsdBudgetGb(args[i]) catch {
                log.err("--ssd-budget-gb: expected an integer in 1..512; got '{s}'\n", .{args[i]});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, args[i], "--kv-attn-mode") and i + 1 < args.len) {
            i += 1;
            if (std.mem.eql(u8, args[i], "dense")) {
                kv_attn_mode = .dense;
            } else if (std.mem.eql(u8, args[i], "fused")) {
                kv_attn_mode = .fused;
            } else if (std.mem.eql(u8, args[i], "auto")) {
                kv_attn_mode = .auto;
            } else {
                log.err("--kv-attn-mode: expected 'dense', 'fused' or 'auto'; got '{s}'\n", .{args[i]});
                std.process.exit(1);
            }
        } else {
            // Nothing above consumed it. This loop used to end here with no
            // else at all, so an unrecognized argument was dropped in SILENCE
            // — `--model=<path>` (the '='-joined form none of the arms match)
            // booted a clean-looking headless server that then auto-picked
            // some other model. A launcher that ignores what it was asked for
            // is worse than one that refuses to start.
            const reason = cli_mod.classifyUnparsedArg(args[i], i + 1 == args.len);
            log.err("unrecognized argument '{s}' — {s}\n", .{ args[i], reason.hint() });
            std.process.exit(1);
        }
    }
    model_settings_mod.vision_flag = model_settings_mod.visionFlagFrom(vision, no_vision) catch {
        log.err("--vision and --no-vision conflict; pass one\n", .{});
        std.process.exit(1);
    };

    if (prompt != null and (serve_explicit or (use_default_models_root and run_model_dir == null) or host_flag != null or port_flag != null or run_opts.tools)) {
        log.err("--prompt/-p cannot be combined with serve, --serve, --host, --port, or --tool on; it uses a private local listener and exits after one reply.\n", .{});
        return error.PromptModeConflict;
    }
    const bind = server_mod.resolveBind(host_flag, if (prompt != null) @as(u16, 0) else port_flag);
    const host = bind.host;
    const port = bind.port;

    const decode_share_env: ?[]const u8 = if (std.c.getenv("SUSHI_PREFILL_DECODE_SHARE")) |r| std.mem.span(r) else null;
    scheduler_mod.prefill_decode_share = scheduler_mod.resolveDecodeShare(decode_share_flag, decode_share_env) catch {
        log.err("--prefill-decode-share / SUSHI_PREFILL_DECODE_SHARE: expected a number >= 0 (above 0.9 clamps), got '{s}'\n", .{decode_share_flag orelse decode_share_env.?});
        std.process.exit(1);
    };
    const effective_decode_share = scheduler_mod.prefillDecodeShare();
    if (decode_share_flag != null or decode_share_env != null) {
        log.info("[prefill] decode share: configured={d}, effective={d} ({s})\n", .{
            scheduler_mod.prefill_decode_share, effective_decode_share,
            if (decode_share_flag != null) "--prefill-decode-share" else "SUSHI_PREFILL_DECODE_SHARE",
        });
    }

    transformer_mod.Transformer.mtp_head_kv_quant_flag = mtp_head_kv_quant;
    generate_mod.mtp_acceptance_default = mtp_acceptance.parse(mtp_typical_raw, mtp_tokenv3_raw) catch |err| {
        log.err("MTP acceptance settings: {s} (--mtp-typical needs d > 0; --mtp-tokenv3 needs 0 <= a <= 1; choose one)\n", .{@errorName(err)});
        std.process.exit(1);
    };
    generate_mod.mtp_acceptance_explicit = mtp_typical_raw != null or mtp_tokenv3_raw != null;
    depth_bounds.validate(depth_bounds.active) catch {
        log.err("--mtp-min-depth {d} is above --mtp-max-depth {d}\n", .{ depth_bounds.active.min, depth_bounds.active.max });
        std.process.exit(1);
    };

    // Subcommand plumbing: `run <model>` supplies the model dir + serve
    // mode; `run`/`serve` default the discovery root to ~/.sushi/models
    // so every pulled model is loadable by name (Ollama-style).
    var default_models_root_storage: ?[]u8 = null;
    defer if (default_models_root_storage) |r| allocator.free(r);
    if (run_model_dir) |d| {
        model_dir = d;
        serve_mode = true;
    }
    if (use_default_models_root) serve_mode = true;
    if (prompt != null) {
        if (model_dir.len == 0) {
            log.err("--prompt/-p requires --model <path> or run <model>\n", .{});
            return error.PromptModelRequired;
        }
        serve_mode = true;
        repl_after_serve = false;
    }
    model_mod.think_effort_flag = if (run_opts.think == .effort) run_opts.think.effort else null;
    const prompt_body = if (prompt) |text| try cli_mod.buildPromptBody(allocator, text, run_opts.think, .{ .max_tokens = max_tokens, .temperature = temperature, .top_p = top_p_flag orelse 1.0, .top_k = top_k_flag orelse 0 }) else null;
    defer if (prompt_body) |body| allocator.free(body);
    var prompt_state: PromptClient = .{ .allocator = allocator, .io = io, .body = prompt_body orelse "", .stream = stream_mode };
    prompt_client = if (prompt != null) &prompt_state else null;
    defer prompt_client = null;
    // An unspecified `--model-dir` falls back to the shared models root that
    // `pull`/`list`/the app already agree on. Gated so `--model <path> --serve`
    // still serves exactly the one model it named (cli.shouldDefaultModelsRoot).
    if (models_root == null and cli_mod.shouldDefaultModelsRoot(.{
        .subcommand = use_default_models_root,
        .serve_mode = serve_mode,
        .has_explicit_model = model_dir.len > 0,
    })) {
        const home = std.mem.span(std.c.getenv("HOME") orelse "/tmp");
        default_models_root_storage = try cli_mod.modelsRootPath(allocator, home);
        models_root = default_models_root_storage;
    }

    // `sushi run` on a TTY quiets logs to warn (unless --log-level was
    // given) BEFORE the models-root scan below — discovery's per-directory
    // `[discovery] skip …` info lines would otherwise spam the chat REPL.
    if (repl_after_serve and !log_level_explicit) log.setLevel(.warn);

    // Persist the server log. The macOS app only keeps stderr in a 64 KB
    // in-memory ring, so a server that crashed or was restarted takes its
    // history with it — exactly when you need it (see the 2026-07-08 pi
    // session post-mortem). Serving paths only; `pull`/`list` stay quiet.
    //
    // NOTE: opened AFTER `--log-level` is parsed (so the level gates what
    // reaches disk) and BEFORE model discovery/loading (so weight-load and
    // auto-context lines land in the file).
    if (serve_mode or repl_after_serve) {
        var log_path_buf: [1024]u8 = undefined;
        const chosen: ?[]const u8 = if (log_file_arg) |a|
            (if (std.mem.eql(u8, a, "off") or std.mem.eql(u8, a, "none")) null else a)
        else if (std.c.getenv("HOME")) |h|
            log.defaultLogPath(&log_path_buf, std.mem.span(h), port) catch null
        else
            null;
        if (chosen) |p| {
            if (log.openFile(p, log.default_max_bytes)) |_| {
                log.info("Logging to {s} (rotates at {d} MB)\n", .{ p, log.default_max_bytes / (1024 * 1024) });
            } else |e| {
                log.warn("could not open log file {s}: {s} (stderr only)\n", .{ p, @errorName(e) });
            }
        }
    }
    defer log.closeFile();
    if (serve_mode and prompt == null) update_mod.startDailyCheck(io, no_update_check, transformer_mod.diagEnvOn("SUSHI_NO_UPDATE_CHECK"));

    if (parent_pid) |pid| {
        parent_watch.start(pid) catch |err| {
            log.err("--parent-pid: cannot start the watchdog: {s}\n", .{@errorName(err)});
            std.process.exit(1);
        };
    }

    // Plan 05 Phase 1: model discovery. When --model-dir is passed, scan
    // the directory for subdirectories containing config.json. The
    // discovered list is published via /v1/models. v1: routing still goes
    // to a single loaded model — if --model isn't set, pick the first
    // discovered. v2 (plan 05 phases 2-5) adds on-demand load and LRU.
    var discovery_storage: ?model_discovery.DiscoveryResult = null;
    defer if (discovery_storage) |*d| d.deinit();
    if (models_root) |root| {
        // Every `--model-dir`, first-wins on a repeated id (see
        // `discoverModelsMany` for why de-dup is not optional here).
        var roots_buf: [8][]const u8 = undefined;
        roots_buf[0] = root;
        for (extra_roots[0..extra_roots_n], 0..) |r, n| roots_buf[n + 1] = r;
        const roots = roots_buf[0 .. 1 + extra_roots_n];
        discovery_storage = model_discovery.discoverModelsMany(io, allocator, roots) catch |err| blk: {
            log.warn("--model-dir scan failed: {s}\n", .{@errorName(err)});
            break :blk null;
        };
        if (discovery_storage) |*d| {
            if (d.models.len > 0) {
                if (roots.len == 1) {
                    log.info("Discovered {d} model(s) under {s}:\n", .{ d.models.len, root });
                } else {
                    log.info("Discovered {d} model(s) under {d} folders:\n", .{ d.models.len, roots.len });
                    for (roots) |r| log.info("  (scanning {s})\n", .{r});
                }
                for (d.models) |m| {
                    if (m.bytes_on_disk) |b| {
                        log.info("  - {s} ({d:.1} GB)\n", .{ m.id, @as(f64, @floatFromInt(b)) / 1_073_741_824.0 });
                    } else {
                        log.info("  - {s}\n", .{m.id});
                    }
                }
                // No auto-select: when `--model` is omitted but `--model-dir` is
                // present, the server starts HEADLESS (no primary model). All
                // discovered models are registered as stubs and load on demand via
                // `/v1/load-model` — chat OR media. The headless branch in the
                // serve block below handles this.
            }
        }
    }
    // In serve mode, check if the port is already in use before loading the model
    // (model loading takes seconds — fail fast instead of wasting time)
    if (serve_mode) {
        server_mod.ensurePortFree(io, host, port) catch |err| {
            var msg_buf: [512]u8 = undefined;
            if (server_mod.startupRefusal(err, host, port, &msg_buf)) |msg| {
                log.err("{s}\n", .{msg});
                std.process.exit(1);
            }
            return err;
        };
    }

    // `sushi run` on a TTY: chat REPL on a side thread. It polls
    // /health until the model is up, then drives the server's own
    // /v1/chat/completions (SSE) endpoint. (Logs were already quieted to warn above,
    // before discovery, so streamed tokens aren't interleaved with [info]
    // lines.)
    if (repl_after_serve and serve_mode) {
        const t = std.Thread.spawn(.{}, replThreadMain, .{ allocator, io, port, run_opts }) catch |err| blk: {
            log.warn("could not start chat REPL: {s}\n", .{@errorName(err)});
            break :blk null;
        };
        if (t) |thread| thread.detach();
    }

    // Observability: allocate the metrics core once (when --metrics is on) and
    // publish it via the server-global `g_metrics`. Declared here — above every
    // serve-dispatch path (headless and the primary MLX
    // path) — so `server_mod.serve()` spawns the gauge sampler + routes /metrics
    // regardless of engine, and each LoadParams builder reads it back into the
    // scheduler's per-request sink via `.metrics = server_mod.g_metrics`. The
    // instance lives on this stack frame for the whole process lifetime; the
    // defer clears the global so an early serve() failure can't leave it
    // dangling. Off (the default) → null: a single per-request branch, no cost.
    var metrics_instance: ?metrics_mod.Metrics = if (metrics_enabled) metrics_mod.Metrics.init() else null;
    if (metrics_instance) |*m| server_mod.g_metrics = m;
    defer server_mod.g_metrics = null;

    // An unsupported file format is refused by name before any config.json read.
    if (isGgufPath(io, model_dir)) {
        log.err("[load] {s}: unsupported model format. Serve an MLX safetensors checkpoint (qwen4_exp or mimo_v2).\n", .{model_dir});
        std.process.exit(1);
    }

    // Print MLX version
    var ver = mlx.mlx_string_new();
    defer _ = mlx.mlx_string_free(ver);
    try mlx.check(mlx.mlx_version(&ver));
    log.info("sushi {s} (MLX {s})\n", .{ VERSION, mlx.mlx_string_data(ver) });

    // Every text-gen serve path takes the PLD defaults from this ONE value —
    // see `server.PldDefaults`. Built after arg parsing so it can't capture a
    // pre-flag default, and passed whole so a path can't honor `--pld` while
    // dropping the two lengths next to it (which is precisely what headless
    // mode did).
    var cli_pld = server_mod.PldDefaults.fromCli(enable_pld, pld_draft_len, pld_key_len);
    cli_pld.explicit = pld_explicit;
    const launch_serve = LaunchServe{
        .ctx_size = ctx_size,
        .timeout = timeout,
        .reasoning_budget = reasoning_budget,
        .max_tokens = serve_default_max_tokens,
        .temperature = if (temp_explicit) temperature else null,
        .top_p = top_p_flag,
        .top_k = top_k_flag,
        .pld = cli_pld,
        .kv_attn_mode = kv_attn_mode,
    };

    // Echo the resolved arguments — makes drafter/target mismatches obvious
    // from the log without having to scroll through the whole launch line in
    // the parent's process listing.
    log.info("[args] model: {s}\n", .{model_dir});
    if (drafter_dir) |dir| {
        log.info("[args] drafter: {s} (block_size={d}{s})\n", .{
            dir,
            draft_block_size,
            if (draft_block_size_explicit) "" else ", auto",
        });
    } else {
        log.debug("[args] drafter: <none>\n", .{});
    }
    if (serve_mode) {
        log.info("[args] serve: {s}:{d}, ctx-size={d}, vision={s}, prevent-sleep={}\n", .{
            host,
            port,
            ctx_size,
            if (vision) "--vision" else if (no_vision) "--no-vision" else "default",
            sleep_inhibit_mod.isEnabled(),
        });
    }
    if (!kv_quant_explicit) {
        log.info("[args] kv-quant: model default (resolved at load)\n", .{});
    } else switch (kv_quant_config.scheme) {
        .off => log.info("[args] kv-quant: off\n", .{}),
        .affine => log.info("[args] kv-quant: affine {d}-bit (group={d})\n", .{ kv_quant_config.bits, kv_quant_config.group_size }),
    }
    log.info("[args] kv-attn-mode: {s}\n", .{@tagName(kv_attn_mode)});
    if (model_settings_mod.fast) {
        // The launch layer only: a model's own settings resolve at its load.
        const mtp = model_settings_mod.MtpChoice.resolve(model_settings_mod.launchFlag(bool, enable_mtp, mtp_explicit), null, true);
        const acceptance = generate_mod.mtpAcceptanceFor(null);
        const tail = generate_mod.mtpGreedyTailFor(null);
        const kv = transformer_mod.KvCacheChoice.resolve(null, kv_quant_config, kv_quant_explicit);
        log.info("[args] fast: mtp {s} ({s}); acceptance {s} ({s}); greedy tail {s} ({s}); {s} ({s})\n", .{
            mtp.label(),                           mtp.sourceName(),
            mtp_acceptance.name(acceptance.value), model_settings_mod.sourceLabel(acceptance.source, model_settings_mod.acceptanceFlagName(acceptance.value)),
            if (tail.value) "on" else "off",       model_settings_mod.sourceLabel(tail.source, "--mtp-greedy-tail"),
            kv.label(),                            kv.sourceName(),
        });
    }

    // Set GPU as default
    var metal_avail: bool = false;
    try mlx.check(mlx.mlx_metal_is_available(&metal_avail));
    log.info("Metal GPU: {}\n", .{metal_avail});

    if (metal_avail) {
        const gpu_dev = mlx.mlx_device_new_type(.gpu, 0);
        defer _ = mlx.mlx_device_free(gpu_dev);
        try mlx.check(mlx.mlx_set_default_device(gpu_dev));
    }

    // Seed MLX RNG with current wall-clock time for non-deterministic sampling
    _ = mlx.mlx_random_seed(@intCast(std.Io.Timestamp.now(io, .real).toMilliseconds()));

    if (serve_mode) {
        // Headless boot: `--model-dir` given, no `--model`. Start with no
        // primary model; everything (chat + media) loads on demand through the
        // registry. The app uses this so a single server hosts chat + image +
        // audio + video, coexisting under one memory budget.
        if (model_dir.len == 0) {
            const discovery_for_registry = discovery_storage;
            discovery_storage = null; // ownership moves to the registry
            try runHeadlessServe(io, allocator, discovery_for_registry, host, port, launch_serve, drafter_dir orelse "", no_drafter, max_resident_models, max_resident_mem, max_resident_mem_explicit, idle_evict_secs, kv_quant_config, kv_quant_explicit, enable_mtp, mtp_explicit);
            return;
        }
    }

    // Parse config — heap allocate so the LoadedModel can take ownership
    // (Plan 05). Free path in serve_mode = registry.deinit; offline mode =
    // explicit defer on `config_storage`.
    const config_storage = try allocator.create(model_mod.ModelConfig);
    var config_owned_by_registry = false;
    // defer-only, NOT errdefer + defer: a plain `defer` already runs on the
    // error-return path, so pairing it with an errdefer that has the same body
    // frees the resource twice on error (double-free / SIGSEGV). The runtime
    // `owned_by_registry` guard makes the single defer correct on every exit.
    // `create` hands back UNINITIALIZED memory and the defer below READS a
    // field, so the struct gets a valid value before that defer can ever run:
    // a `parseConfig` failure would otherwise free a garbage pointer.
    config_storage.* = std.mem.zeroes(model_mod.ModelConfig);
    defer if (!config_owned_by_registry) {
        config_storage.deinit(allocator);
        allocator.destroy(config_storage);
    };
    config_storage.* = try model_mod.parseConfig(io, allocator, model_dir);
    const config = config_storage;
    scheduler_mod.applyModelSettings(config, model_settings_mod.overrideFor(allocator, io, model_dir));
    if (model_mod.think_effort_flag) |e| if (model_mod.effortArms(config.model_type)) |arms| {
        if (model_mod.armForWord(arms, @tagName(e)) == null) {
            var choices: [std.enums.values(model_mod.Effort).len]model_mod.Effort = undefined;
            for (arms, 0..) |arm, ai| choices[ai] = arm.effort;
            var w = std.Io.Writer.Allocating.init(allocator);
            defer w.deinit();
            try cli_mod.writeEffortRefusal(&w.writer, @tagName(e), config.model_type, choices[0..arms.len]);
            log.err("--think: {s}\n", .{w.written()});
            std.process.exit(1);
        }
    };
    if (config.glm_fp8_trunk) {
        log.info("Model: {s} ({d} layers, {d}-dim, raw FP8 E4M3FN block128 trunk)\n", .{ config.model_type, config.num_hidden_layers, config.hidden_size });
    } else {
        log.info("Model: {s} ({d} layers, {d}-dim, head_dim={d}, {d}h/{d}kv, {d}-bit {s} quant)\n", .{
            config.model_type,
            config.num_hidden_layers,
            config.hidden_size,
            config.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.quant_bits,
            @tagName(config.quant_mode),
        });
    }

    // Load tokenizer — heap-allocated, ownership transfers to registry on serve_mode.
    log.info("Loading tokenizer...\n", .{});
    const tok = try allocator.create(tokenizer_mod.Tokenizer);
    var tok_owned_by_registry = false;
    tok.* = tokenizer_mod.loadTokenizer(io, allocator, model_dir) catch |err| {
        // Raw memory only — nothing initialized to deinit. The cleanup defer
        // below must NOT be registered yet: deinit on the undefined pointee
        // was a live SIGSEGV on a partially-downloaded model dir (the
        // preloadCpuState errdefer-after-init pattern applies here too).
        allocator.destroy(tok);
        log.err("failed to load tokenizer from {s}: {s} (incomplete download? `sushi pull` the model again to resume, or delete the dir)\n", .{ model_dir, @errorName(err) });
        return err;
    };
    // defer-only (see config note above): errdefer + defer with the same body
    // double-frees on the error-return path.
    defer if (!tok_owned_by_registry) {
        tok.deinit();
        allocator.destroy(tok);
    };

    // Load chat config — heap-allocated, ownership transfers to registry on serve_mode.
    const chat_config = try allocator.create(chat_mod.ChatConfig);
    var chat_config_owned_by_registry = false;
    chat_config.* = chat_mod.loadChatConfig(io, allocator, model_dir) catch |err| {
        // Raw memory only — see the tokenizer catch above.
        allocator.destroy(chat_config);
        return err;
    };
    // defer-only (see config note above): errdefer + defer with the same body
    // double-frees on the error-return path — this is the one that crashed in
    // the #45 GPU-OOM pre-flight refusal (ChatConfig.deinit ran twice).
    defer if (!chat_config_owned_by_registry) {
        chat_config.deinit();
        allocator.destroy(chat_config);
    };

    // Merge the tokenizer's chat-terminator EOS into the stop set — ALWAYS,
    // even when config.json already specified an eos_token_id. Some checkpoints
    // (e.g. Qwen2.5-Coder-7B) set config.json eos_token_id to <|endoftext|>
    // (151643) but their chat template ends turns with <|im_end|> (151645);
    // stopping only on config's id leaks <|im_end|> into the output (breaks
    // structured-JSON / tool-calling). Additive + dedup-guarded: this can only
    // ADD a model-declared stop token, never remove one.
    if (chat_config.eos_token) |eos_str| {
        if (tok.special_tokens.get(eos_str)) |eos_id| {
            if (!config.isEosToken(eos_id)) {
                config.addEosToken(eos_id);
                log.info("EOS token from tokenizer: {s} (id={d})\n", .{ eos_str, eos_id });
            }
        }
    }
    // Also add <|endoftext|> if it exists and wasn't already added.
    if (tok.special_tokens.get("<|endoftext|>")) |eot_id| {
        if (!config.isEosToken(eot_id)) {
            config.addEosToken(eot_id);
        }
    }

    // Treat <pad> as a stop token, but only if it's not token ID 0
    // (ID 0 can be produced spuriously by models under long/confusing prompts)
    if (tok.special_tokens.get("<pad>")) |pad_id| {
        if (pad_id > 0 and !config.isEosToken(pad_id)) {
            config.addEosToken(pad_id);
            log.info("Added <pad> as stop token (id={d})\n", .{pad_id});
        }
    }

    const load_vision = config.has_vision and !no_vision;

    if (serve_mode) {
        // ── Plan 05: build the ModelRegistry, register a stub for the
        //    loaded model, and pass everything to serve(). The registry
        //    takes ownership of `discovery_storage` (if any) and, once
        //    the inference thread completes loading, ownership of
        //    config/tok/chat_config too.
        const model_id = blk: {
            var p = model_dir;
            while (p.len > 0 and p[p.len - 1] == '/') p = p[0 .. p.len - 1];
            if (p.len == 0) break :blk config.model_type;
            if (std.mem.lastIndexOfScalar(u8, p, '/')) |slash_idx| break :blk p[slash_idx + 1 ..];
            break :blk p;
        };

        const discovery_for_registry = discovery_storage;
        discovery_storage = null; // ownership moves to the registry

        // Plan 05 Phase D: compute the effective max_resident_mem. When the
        // user didn't pass an explicit cap, derive 80% of mlx's wired limit
        // (mlx_set_wired_limit returns a value the platform considers safe
        // for sustained GPU work). The wired limit was already applied in
        // the inference thread's load path; here we mirror that calculation
        // so the registry's eviction gate stays in sync. 0 disables the cap.
        const effective_max_resident_mem: u64 = if (max_resident_mem_explicit)
            max_resident_mem
        else blk: {
            var dev = mlx.mlx_device{ .ctx = null };
            _ = mlx.mlx_get_default_device(&dev);
            var info = mlx.mlx_device_info_new();
            defer _ = mlx.mlx_device_info_free(info);
            if (mlx.mlx_device_info_get(&info, dev) != 0) break :blk 0;
            var max_rec: usize = 0;
            if (mlx.mlx_device_info_get_size(&max_rec, info, "max_recommended_working_set_size") != 0 or max_rec == 0) break :blk 0;
            break :blk @as(u64, max_rec) * 4 / 5;
        };
        if (effective_max_resident_mem > 0) {
            log.info("[registry] max_resident_models={d}, max_resident_mem={d:.1} GB\n", .{
                max_resident_models,
                @as(f64, @floatFromInt(effective_max_resident_mem)) / 1_073_741_824.0,
            });
        } else {
            log.info("[registry] max_resident_models={d}, max_resident_mem=unlimited\n", .{max_resident_models});
        }

        const registry = try model_registry_mod.ModelRegistry.init(
            allocator,
            io,
            discovery_for_registry,
            max_resident_models,
            effective_max_resident_mem,
            idle_evict_secs,
        );
        defer registry.deinit();
        registry.mem_cap_binds_alone = max_resident_mem_explicit;

        // Register the loaded model. Use the pre-registered discovery entry
        // when available (so id/path/bytes_on_disk are consistent across
        // /v1/models listings); otherwise create a fresh stub.
        const entry = if (registry.peek(model_id)) |e|
            e
        else if (registry.peekByPath(model_dir)) |e|
            // Discovered under an org/name id whose basename differs from
            // model_id — reuse it, never register the same path twice.
            e
        else
            try registry.registerStub(model_id, model_dir, null);
        try registry.setDefault(entry.id);

        // Ownership-transfer defer: registry takes ownership of
        // config/tok/chat_config IF the inference-thread load installed
        // them on `entry` (entry.config != null). Declared AFTER
        // registry.deinit so it fires BEFORE it on scope exit — by the
        // time registry.deinit walks the entry we've already decided who
        // owns the heap pointers, so the early defers can no-op.
        defer if (entry.config != null) {
            config_owned_by_registry = true;
            tok_owned_by_registry = true;
            chat_config_owned_by_registry = true;
        };

        const params = scheduler_mod.LoadParams{
            .registry = registry,
            .entry = entry,
            .config = config,
            .tok = tok,
            .chat_config = chat_config,
            .model_dir = model_dir,
            .ctx_size = ctx_size,
            .drafter_dir = drafter_dir orelse "",
            .no_drafter = no_drafter,
            .mtp_enabled = enable_mtp,
            .mtp_explicit = mtp_explicit,
            .mtp_head_kv_quant = mtp_head_kv_quant,
            .mtp_depth = depth_bounds.active.max,
            .load_vision = load_vision,
            .warmup_eager = warmup_eager,
            .draft_block_size = draft_block_size,
            .draft_block_size_explicit = draft_block_size_explicit,
            .kv_quant_config = kv_quant_config,
            .kv_quant_explicit = kv_quant_explicit,
            .prefix_cache_capacity = server_mod.prefix_cache_capacity,
            .prefix_cache_ram_enabled = server_mod.prefix_cache_ram_enabled,
            .prefix_cache_mem_bytes = server_mod.prefix_cache_mem_bytes,
            .prefix_cache_mem_resolver = server_mod.prefixCacheMemForLoad,
            .prefix_cache_disk_bytes = server_mod.prefix_cache_disk_bytes,
            .prefix_cache_disk_resolver = server_mod.prefixCacheDiskForLoad,
            .expert_cache_bytes = expert_cache_bytes,
            .ssd_budget_bytes = ssd_budget_bytes,
            .expert_cache_fit_resolver = server_mod.expertCacheFitForLoad,
            .ssm_checkpoint_stride = server_mod.effectiveSsmCheckpointStride(server_mod.ssm_checkpoint_stride, server_mod.prefix_cache_capacity, server_mod.prefix_cache_ram_enabled, server_mod.prefix_cache_disk_bytes),
            .ssm_checkpoint_max = server_mod.ssm_checkpoint_max,
            .tokenize_cache_entries = server_mod.tokenize_cache_entries,
            .metrics = server_mod.g_metrics,
        };
        var serve_cfg = launch_serve.serverConfig();
        serve_cfg.on_ready = if (prompt != null) PromptClient.ready else null;
        serve_cfg.default_force_mtp = force_mtp;
        try server_mod.serve(io, allocator, params, config, host, port, serve_cfg);
        if (prompt_state.err) |err| return err;
    } else {
        // ── Offline single-prompt mode. mlx ops run on this thread, no
        //    scheduler. The same load path as pre-A1.
        if (config.expertStreamingRequired()) {
            log.err("This checkpoint requires expert streaming. Use `run <model>` or `--serve` with --ssd-budget-gb or --expert-cache-gb.\n", .{});
            return error.ExpertStreamingRequired;
        }
        log.info("Loading weights...\n", .{});
        var weights = try model_mod.loadWeightsForConfig(io, allocator, model_dir, config, load_vision);
        defer weights.deinit();
        model_mod.resolveWeightPrefix(config, &weights);

        var xfm = try transformer_mod.Transformer.init(io, allocator, config.*, &weights);
        defer xfm.deinit();

        xfm.round_cost.layout = round_cost_mod.layoutFor(config);

        // Reserved-token suppression, same derivation as the serve path.
        generate_mod.installSuppressMask(&xfm, tok, chat_config.chat_template, config.eosTokenSlice());
        generate_mod.installThinkMarkers(&xfm, tok);
        try generate_mod.installLogitBias(io, &xfm, tok);

        // Honor --kv-quant in offline mode too. The serve path threads this
        // through Slot caches via the scheduler; here we swap the
        // Transformer's own legacy cache to match.
        const kv_cache = transformer_mod.KvCacheChoice.resolve(config.kv_quant_override, kv_quant_config, kv_quant_explicit);
        log.info("[kv-cache] {s} ({s})\n", .{ kv_cache.label(), kv_cache.sourceName() });
        const mtp_choice = scheduler_mod.mtpChoiceFor(enable_mtp, mtp_explicit, config);
        log.info("[mtp] {s} ({s})\n", .{ mtp_choice.label(), mtp_choice.sourceName() });
        if (kv_cache.config.scheme != .off) {
            try xfm.cache.reinit(config.num_hidden_layers, kv_cache.config);
        }
        try xfm.qwen4MtpApplyKvQuant(kv_cache.config);

        // JIT-compile + wire memory limits (policy: mlx.applyWiredPolicy).
        {
            const wired = mlx.applyWiredPolicy();
            if (wired.target) |t| log.debug("[wired] mode={s} limit={d} MB\n", .{ @tagName(wired.mode), t / (1024 * 1024) });
        }
        if (config.hidden_act == .gelu_approx) {
            xfm.compileGelu();
            xfm.compileGeglu();
        }
        if (xfm.moe_layers != null) {
            xfm.compileMoeRouting();
        }
        if (config.linear_num_key_heads > 0) {
            xfm.compileGdnGate();
        }
        log.info("Model ready.\n", .{});

        // Qwen native MTP head — auto-load when the model ships one (sidecar
        // file or in-checkpoint tensors in the trunk shards).
        var mtp_head: ?mtp_mod.MtpModel = null;
        defer if (mtp_head) |*h| h.deinit();
        if (mtp_choice.on and mtp_mod.hasMtpHead(io, allocator, model_dir)) {
            // A failed load (e.g. a sidecar layout we can't bind yet) only
            // disables the head — mirrors the serve path's graceful degrade.
            if (mtp_mod.loadMtp(io, allocator, xfm.s, model_dir)) |loaded| {
                mtp_head = loaded;
                mtp_head.?.bind(&xfm) catch |err| {
                    log.warn("[mtp] sidecar incompatible with target ({any}) — disabled\n", .{err});
                    mtp_head.?.deinit();
                    mtp_head = null;
                };
            } else |err| {
                log.warn("[mtp] failed to load sidecar ({any}) — disabled\n", .{err});
            }
        }

        const user_prompt = prompt orelse "What is 2+2? Answer in one sentence.";
        const messages = [_]chat_mod.Message{
            .{ .role = "user", .content = user_prompt },
        };

        const prompt_ids = try chat_mod.formatChat(allocator, tok, &messages, chat_config, null, null, false, null, false);
        defer allocator.free(prompt_ids);

        // Reset peak memory before generation
        _ = mlx.mlx_reset_peak_memory();

        const eos_slice = config.eosTokenSlice();
        const sampling = generate_mod.SamplingParams{ .temperature = temperature };

        var stdout_buf: [16 * 1024]u8 = undefined;
        var stdout_w_state = std.Io.File.stdout().writer(io, &stdout_buf);
        const stdout_w = &stdout_w_state.interface;
        defer stdout_w.flush() catch {};

        if (stream_mode) {
            // Streaming: print tokens as they're generated
            const prefill_start = std.Io.Timestamp.now(io, .awake);
            var gen = try generate_mod.Generator.init(io, allocator, &xfm, tok, prompt_ids, max_tokens, sampling, eos_slice);
            defer gen.deinit(allocator);

            const prefill_ns: u64 = @intCast(prefill_start.untilNow(io, .awake).nanoseconds);
            const prefill_tps: f64 = if (prefill_ns > 0)
                @as(f64, @floatFromInt(prompt_ids.len)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(prefill_ns))
            else
                0.0;

            try stdout_w.writeAll("==========\n");
            const decode_start = std.Io.Timestamp.now(io, .awake);
            var completion_tokens: u32 = 0;
            while (try gen.next(allocator)) |token_id| {
                const ids = [_]u32{token_id};
                const piece = try tok.decode(allocator, &ids, completion_tokens == 0);
                defer allocator.free(piece);
                if (piece.len > 0) {
                    try stdout_w.writeAll(piece);
                    try stdout_w.flush();
                }
                completion_tokens += 1;
            }
            const decode_ns: u64 = @intCast(decode_start.untilNow(io, .awake).nanoseconds);
            const decode_tps: f64 = if (decode_ns > 0)
                @as(f64, @floatFromInt(completion_tokens)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(decode_ns))
            else
                0.0;

            try stdout_w.writeAll("\n==========\n");
            try stdout_w.print("Prompt: {d} tokens, {d:.3} tokens-per-sec\n", .{ prompt_ids.len, prefill_tps });
            try stdout_w.print("Generation: {d} tokens, {d:.3} tokens-per-sec\n", .{ completion_tokens, decode_tps });
        } else {
            // Non-streaming: generate all tokens then print
            const result = if (mtp_head) |*h|
                try generate_mod.generateMtp(io, allocator, &xfm, h, tok, prompt_ids, max_tokens, sampling, eos_slice, 0, depth_bounds.active.max, null)
            else
                try generate_mod.generate(io, allocator, &xfm, tok, prompt_ids, max_tokens, sampling, eos_slice, 0, 0);
            defer allocator.free(result.text);
            defer allocator.free(result.token_ids);

            try stdout_w.writeAll("==========\n");
            try stdout_w.writeAll(result.text);
            try stdout_w.writeAll("\n==========\n");
            try stdout_w.print("Prompt: {d} tokens, {d:.3} tokens-per-sec\n", .{ result.prompt_tokens, result.prefill_tps });
            try stdout_w.print("Generation: {d} tokens, {d:.3} tokens-per-sec\n", .{ result.completion_tokens, result.decode_tps });
        }

        var peak_mem: usize = 0;
        _ = mlx.mlx_get_peak_memory(&peak_mem);
        const peak_gb = @as(f64, @floatFromInt(peak_mem)) / (1024.0 * 1024.0 * 1024.0);
        try stdout_w.print("Peak memory: {d:.3} GB\n", .{peak_gb});
    }
}

const isGgufPath = model_discovery.isGgufModelPath;

/// Registry resident-memory cap: the user's explicit value, or 80% of mlx's
/// wired limit at startup (mirrors the MLX serve block). 0 = query failed →
/// unlimited (the count cap still applies).
fn autoResidentMemBytes(explicit: bool, val: u64) u64 {
    if (explicit) return val;
    var dev = mlx.mlx_device{ .ctx = null };
    _ = mlx.mlx_get_default_device(&dev);
    var info = mlx.mlx_device_info_new();
    defer _ = mlx.mlx_device_info_free(info);
    if (mlx.mlx_device_info_get(&info, dev) != 0) return 0;
    var max_rec: usize = 0;
    if (mlx.mlx_device_info_get_size(&max_rec, info, "max_recommended_working_set_size") != 0 or max_rec == 0) return 0;
    return @as(u64, max_rec) * 4 / 5;
}

/// The ServerConfig fields both boot modes take from the parsed launch flags; a mode
/// adds only what is its own (`on_ready`, `default_force_mtp`).
const LaunchServe = struct {
    ctx_size: u32,
    timeout: u32,
    reasoning_budget: i32,
    max_tokens: u32,
    temperature: ?f32,
    top_p: ?f32,
    top_k: ?u32,
    pld: server_mod.PldDefaults,
    kv_attn_mode: server_mod.KvAttnMode,

    fn serverConfig(self: LaunchServe) server_mod.ServerConfig {
        return .{
            .max_context_size = self.ctx_size,
            .request_timeout_sec = self.timeout,
            .default_reasoning_budget = self.reasoning_budget,
            .default_max_tokens = self.max_tokens,
            .default_temperature = self.temperature,
            .default_top_p = self.top_p,
            .default_top_k = self.top_k,
            .default_enable_pld = self.pld.enable,
            .pld_explicit = self.pld.explicit,
            .default_pld_draft_len = self.pld.draft_len,
            .default_pld_key_len = self.pld.key_len,
            .kv_attn_mode = self.kv_attn_mode,
        };
    }
};

/// Headless serve mode: start with NO primary model. The registry holds all
/// discovery stubs; chat AND media models load on demand via `/v1/load-model`
/// (or a request targeting a discovered id), coexisting under one memory
/// budget. The scheduler's borrowed-view fields are seeded from a throwaway
/// stub that's never installed on an entry (`no_initial_load`).
fn runHeadlessServe(
    io: std.Io,
    allocator: std.mem.Allocator,
    discovery: ?model_discovery.DiscoveryResult,
    host: []const u8,
    port: u16,
    launch_serve: LaunchServe,
    drafter_dir: []const u8,
    no_drafter: bool,
    max_resident_models: u32,
    max_resident_mem: u64,
    max_resident_mem_explicit: bool,
    idle_evict_secs: ?u32,
    kv_quant_config: transformer_mod.KVQuantConfig,
    kv_quant_explicit: bool,
    enable_mtp: bool,
    mtp_explicit: bool,
) !void {
    log.info("sushi {s} (headless — models load on demand)\n", .{VERSION});
    log.info("[args] serve: {s}:{d}\n", .{ host, port });

    var stub = try scheduler_mod.headlessStubCpuState(allocator);
    defer scheduler_mod.freeCpuState(allocator, &stub);

    const effective_max_resident_mem = autoResidentMemBytes(max_resident_mem_explicit, max_resident_mem);
    if (effective_max_resident_mem > 0) {
        log.info("[registry] max_resident_models={d}, max_resident_mem={d:.1} GB\n", .{ max_resident_models, @as(f64, @floatFromInt(effective_max_resident_mem)) / 1_073_741_824.0 });
    } else {
        log.info("[registry] max_resident_models={d}, max_resident_mem=unlimited\n", .{max_resident_models});
    }

    const registry = try model_registry_mod.ModelRegistry.init(allocator, io, discovery, max_resident_models, effective_max_resident_mem, idle_evict_secs);
    defer registry.deinit();
    registry.mem_cap_binds_alone = max_resident_mem_explicit;

    // Carrier entry for LoadParams (required field), never loaded here
    // (`no_initial_load`). Prefer a discovered stub (so it's listed in
    // /v1/models); else a throwaway placeholder — headless is valid with empty
    // discovery because the app loads media/chat models by ABSOLUTE PATH via
    // /v1/load-model (registerByPath), regardless of what --model-dir scans.
    // No default is set, so a request that omits `model` gets a clean 503
    // until a model is loaded.
    var placeholder = model_registry_mod.LoadedModel{
        .allocator = allocator,
        .id = "",
        .path = "",
        .bytes_on_disk = null,
        .arch_hint = "",
        .config = null,
        .weights = null,
        .transformer = null,
        .tokenizer = null,
        .chat_config = null,
        .vision_encoder = null,
        .drafter_path = "",
        .drafter_block_size = 0,
        .prefix_cache = null,
        .refcount = std.atomic.Value(u32).init(0),
        .last_used_ns = 0,
        .bytes_resident = 0,
        .state = .unloaded,
        .error_name = null,
    };
    const carrier: *model_registry_mod.LoadedModel = blk: {
        var it = registry.entries.valueIterator();
        if (it.next()) |e| break :blk e.*;
        log.info("Headless: no models under --model-dir; load by path via /v1/load-model.\n", .{});
        break :blk &placeholder;
    };

    const params = scheduler_mod.LoadParams{
        .registry = registry,
        .entry = carrier,
        .config = stub.config,
        .tok = stub.tok,
        .chat_config = stub.chat_config,
        .model_dir = "",
        .ctx_size = launch_serve.ctx_size,
        .drafter_dir = drafter_dir,
        .no_drafter = no_drafter,
        .no_initial_load = true,
        .load_vision = false,
        .warmup_eager = false,
        .draft_block_size = 0,
        .kv_quant_config = kv_quant_config,
        .kv_quant_explicit = kv_quant_explicit,
        .mtp_enabled = enable_mtp,
        .mtp_explicit = mtp_explicit,
        .mtp_head_kv_quant = transformer_mod.Transformer.mtp_head_kv_quant_flag,
        // Seed the scheduler's prefix-cache config from the server globals so
        // on-demand (headless/discover-mode) loads get the SAME hot prefix
        // cache as a `--model` startup load. Previously hardcoded to 0, which
        // left `Scheduler.prefix_cache_capacity == 0` → every model loaded via
        // `ensureLoaded` skipped `HotPrefixCache` init → cross-turn KV reuse
        // was silently dead for the entire headless serving mode (the default
        // `serve` path). Mirrors the LoadParams built in `main()`.
        .prefix_cache_capacity = server_mod.prefix_cache_capacity,
        .prefix_cache_ram_enabled = server_mod.prefix_cache_ram_enabled,
        .prefix_cache_mem_bytes = server_mod.prefix_cache_mem_bytes,
        .prefix_cache_mem_resolver = server_mod.prefixCacheMemForLoad,
        .prefix_cache_disk_bytes = server_mod.prefix_cache_disk_bytes,
        .prefix_cache_disk_resolver = server_mod.prefixCacheDiskForLoad,
        .expert_cache_bytes = expert_cache_bytes,
        .ssd_budget_bytes = ssd_budget_bytes,
        .expert_cache_fit_resolver = server_mod.expertCacheFitForLoad,
        .ssm_checkpoint_stride = server_mod.effectiveSsmCheckpointStride(server_mod.ssm_checkpoint_stride, server_mod.prefix_cache_capacity, server_mod.prefix_cache_ram_enabled, server_mod.prefix_cache_disk_bytes),
        .ssm_checkpoint_max = server_mod.ssm_checkpoint_max,
        .tokenize_cache_entries = server_mod.tokenize_cache_entries,
        .metrics = server_mod.g_metrics,
    };

    var serve_cfg = launch_serve.serverConfig();
    // On-demand MLX loads auto-attach an MTP sidecar (LoadParams.mtp_enabled
    // defaults true), so the MoE force flag has to reach this path too.
    serve_cfg.default_force_mtp = enable_mtp and mtp_explicit;
    try server_mod.serve(io, allocator, params, stub.config, host, port, serve_cfg);
}

/// Parse a size-style CLI argument: bare integer = bytes, suffix `KB`/`MB`/
/// `GB` (case-insensitive) multiplies by 1024^N, "0"/"off" = 0. Used by
/// `--prefix-cache-mem`; returns `error.InvalidSize` on malformed input.
/// `--config-overrides` takes a JSON OBJECT (every field it names is merged into
/// config.json). Parsed here purely to fail the launch on a typo; the merge
/// itself happens per-config in `model.parseConfigFromJson`.
fn configOverridesJsonValid(raw: []const u8) bool {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const v = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), raw, .{}) catch return false;
    return v == .object;
}

fn parseSizeArg(s: []const u8) !u64 {
    if (std.mem.eql(u8, s, "off") or std.mem.eql(u8, s, "0")) return 0;
    var end: usize = s.len;
    var mult: u64 = 1;
    if (std.mem.endsWith(u8, s, "GB") or std.mem.endsWith(u8, s, "gb")) {
        end -= 2;
        mult = 1024 * 1024 * 1024;
    } else if (std.mem.endsWith(u8, s, "MB") or std.mem.endsWith(u8, s, "mb")) {
        end -= 2;
        mult = 1024 * 1024;
    } else if (std.mem.endsWith(u8, s, "KB") or std.mem.endsWith(u8, s, "kb")) {
        end -= 2;
        mult = 1024;
    } else if (std.mem.endsWith(u8, s, "B") or std.mem.endsWith(u8, s, "b")) {
        end -= 1;
    }
    if (end == 0) return error.InvalidSize;
    const n = std.fmt.parseInt(u64, s[0..end], 10) catch return error.InvalidSize;
    return n * mult;
}

test "both boot modes build their server defaults from the same launch flags" {
    const launch = LaunchServe{
        .ctx_size = 32768,
        .timeout = 77,
        .reasoning_budget = 512,
        .max_tokens = 4096,
        .temperature = 0.3,
        .top_p = 0.8,
        .top_k = 20,
        .pld = server_mod.PldDefaults.fromCli(true, 7, 4),
        .kv_attn_mode = .dense,
    };
    const cfg = launch.serverConfig();
    try std.testing.expectEqual(@as(?f32, 0.3), cfg.default_temperature);
    try std.testing.expectEqual(@as(?f32, 0.8), cfg.default_top_p);
    try std.testing.expectEqual(@as(?u32, 20), cfg.default_top_k);
    try std.testing.expectEqual(server_mod.KvAttnMode.dense, cfg.kv_attn_mode);
    try std.testing.expectEqual(@as(u32, 4096), cfg.default_max_tokens);
    try std.testing.expectEqual(@as(u32, 7), cfg.default_pld_draft_len);
    try std.testing.expectEqual(@as(u32, 4), cfg.default_pld_key_len);
    try std.testing.expectEqual(@as(u32, 32768), cfg.max_context_size);

    var unset = launch;
    unset.temperature = null;
    unset.top_p = null;
    unset.top_k = null;
    const none = unset.serverConfig();
    try std.testing.expectEqual(@as(?f32, null), none.default_temperature);
    try std.testing.expectEqual(@as(?u32, null), none.default_top_k);
}
