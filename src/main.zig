const std = @import("std");
const settings_resolve = @import("settings_resolve.zig");
const http = @import("http.zig");
const build_options = @import("build_options");
// pub: src/exl3 reaches mlx, log and io_util through its host root.
pub const mlx = @import("mlx.zig");
pub const io_util = @import("io_util.zig");
const model_mod = @import("model.zig");
const effort_mod = @import("effort.zig");
const tokenizer_mod = @import("tokenizer.zig");
pub const tokenizer = tokenizer_mod;
pub const kld = kld_mod;
pub const bench = bench_mod;
pub const imatrix = @import("imatrix.zig");
pub const model_handle = @import("model_handle.zig");
const trunk = @import("trunk.zig");
const round_cost_mod = @import("round_cost.zig");
const generate_mod = @import("generate.zig");
const mtp_acceptance = @import("mtp_acceptance.zig");
const model_discovery = @import("model_discovery.zig");
const model_registry_mod = @import("model_registry.zig");
const mtp_mod = @import("mtp.zig");
const depth_bounds = @import("mtp_depth_bounds.zig");
const dflash_mod = @import("dflash.zig");
const chat_mod = @import("chat.zig");
const server_mod = @import("server.zig");
const server_state = @import("server_state.zig");
const scheduler_mod = @import("scheduler.zig");
const arch_state = @import("arch/arch_state.zig");
const model_load = @import("model_load.zig");
const expert_stream_mod = @import("expert_stream.zig");
const model_settings_mod = @import("model_settings.zig");
const cli_mod = @import("cli.zig");
const kld_mod = @import("kld.zig");
const help = @import("help.zig");
const arch_flags = @import("arch/flags.zig");
const arch_packs = @import("arch/packs.zig");
const bench_mod = @import("bench.zig");
const encoder_cli = @import("sushi_encoder").cli;
const launch_mod = @import("launch.zig");
pub const log = @import("log.zig");
const metrics_mod = @import("metrics.zig");
const metrics_history = @import("metrics_history.zig");
const sleep_inhibit_mod = @import("sleep_inhibit.zig");
const version_mod = @import("version.zig");
const parent_watch = @import("parent_watch.zig");
const update_mod = @import("update.zig");
const sampler = @import("sampler.zig");

pub const VERSION: []const u8 = build_options.version;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

const DEFAULT_MODEL_DIR = ""; // pass --model <path> to specify

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
        cli_mod.runPrompt(self.allocator, self.io, port, self.body, self.stream, server_state.g_api_key) catch |err| {
            self.err = err;
        };
    }
};
var prompt_client: ?*PromptClient = null;

fn printUsage(io: std.Io) void {
    var stdout_buf: [4096]u8 = undefined;
    var stdout_w = std.Io.File.stdout().writerStreaming(io, &stdout_buf);
    stdout_w.interface.writeAll(help.text) catch {};
    stdout_w.interface.flush() catch {};
}

/// False for the commands and flags that never touch MLX, so they do not wake Metal.
fn commandUsesGpu(proc_args: std.process.Args) bool {
    var it = std.process.Args.Iterator.init(proc_args);
    _ = it.next();
    var first = true;
    while (it.next()) |a| {
        if (first and (std.mem.eql(u8, a, "pull") or std.mem.eql(u8, a, "list") or std.mem.eql(u8, a, "launch") or
            std.mem.eql(u8, a, "update") or arch_flags.commandSkipsGpu(a))) return false;
        first = false;
        if (std.mem.eql(u8, a, "--version") or std.mem.eql(u8, a, "--guest-manifest") or
            std.mem.eql(u8, a, "--help") or std.mem.eql(u8, a, "-h")) return false;
    }
    return true;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    metrics_history.process.markStarted(io, io_util.nowMs(io));

    // Bound MLX's reclaimable buffer pool before anything can allocate. ONCE,
    // above every GPU subcommand branch — a per-serve-path call is how
    // `runHeadlessServe` (the mode the app always launches) silently ate the
    // launch flags. See server.mlxCacheLimitBytes for why MLX's own default
    // (~121 GB on a 128 GB Mac) is no defense.
    if (commandUsesGpu(init.minimal.args)) {
        server_mod.applyMlxCacheLimit();
        server_mod.resolveMlxCadence();
    }
    // Resolve lazily-cached env reads on the main thread before other threads exist.
    @import("arch/kernels.zig").warmEnvCaches();

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
    var args: []const []const u8 = args_list.items;
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
    var bench_run: ?*bench_mod.InProcess = null;
    defer if (bench_run) |r| r.finish();
    var repl_after_serve = false;
    var run_opts: cli_mod.ReplOptions = .{};
    if (args.len >= 2 and args[1].len > 0 and args[1][0] != '-') {
        const cmd = args[1];
        if (std.mem.eql(u8, cmd, "pull")) {
            const model = cli_mod.parsePullArgs(args[2..]) catch |err| {
                switch (err) {
                    error.MissingModel => log.err("usage: sushi pull <model>\n", .{}),
                    error.UnexpectedArgument => log.err("sushi pull: unexpected argument '{s}' (usage: sushi pull <model>)\n", .{args[3]}),
                }
                std.process.exit(1);
            };
            try cli_mod.cmdPull(allocator, io, model);
            return;
        } else if (std.mem.eql(u8, cmd, "list")) {
            cli_mod.parseListArgs(args[2..]) catch {
                log.err("sushi list: unexpected argument '{s}' (usage: sushi list)\n", .{args[2]});
                std.process.exit(1);
            };
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
        } else if (std.mem.eql(u8, cmd, "bench")) {
            switch (try bench_mod.start(allocator, io, args[2..])) {
                .done => return,
                .in_process => |run| {
                    bench_run = run;
                    bench_mod.setCurrent(run);
                    // The serve parser below reads the flags the bench left, as `sushi serve --model` does.
                    args = try run.argv(args[0]);
                    arg_start = 1;
                },
            }
        } else if (std.mem.eql(u8, cmd, "convert")) {
            try encoder_cli.cmdConvert(allocator, io, args[2..]);
            return;
        } else if (std.mem.eql(u8, cmd, "update")) {
            try update_mod.cmdUpdate(allocator, io, args[2..]);
            return;
        } else if (try arch_flags.runCommand(allocator, io, cmd, args[2..])) {
            return;
        } else {
            log.err("unknown command '{s}' (expected " ++ help.known_commands ++ ")\n", .{cmd});
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
    var skip_mem_preflight = false;
    var serve_mode = false;
    var serve_explicit = false;
    var stream_mode = false;
    var tool_flag_given = false;
    var prompt: ?[]const u8 = null;
    var max_tokens: u32 = 100;
    // Serve-mode sampling defaults for requests that omit the field
    // (request > flag > model generation_config.json > hardcoded). `--temp`
    // doubles as the offline --prompt sampling temp, so track whether it was
    // explicitly given — only then does it become the serve default.
    var temp_flag: ?f32 = null;
    var top_p_flag: ?f32 = null;
    var top_k_flag: ?u32 = null;
    var ctx_size: u32 = 0; // 0 = use model default
    var timeout: u32 = 300; // seconds, 0 = no timeout
    var no_vision = false;
    var vision = false;
    var drafter_dir: ?[]const u8 = null; // Path to a DFlash2 assistant checkpoint
    var no_drafter = false; // --no-drafter: never load one, merged-in ones included
    // `--mtp` / `--no-mtp`, the last one given. `--mtp` also forces the head on where requests do not default to
    // it: an SSD-streamed pack and inherited MoE arches (server.defaultEnableMtp).
    var mtp_flag: ?bool = null;
    var mtp_typical_raw: ?[]const u8 = null;
    var mtp_tokenv3_raw: ?[]const u8 = null;
    var kv_quant_flag: ?trunk.KVQuantConfig = null;
    // Plan 05 Phase D: multi-model caps. Defaults aim for "comfortable on
    // 32–64 GB systems running small models". Override via the
    // CLI flags below; the Swift app exposes them under Advanced settings.
    var max_resident_models: u32 = 3;
    var max_resident_mem_flag: ?u64 = null; // null = auto (80% of wired limit at startup)
    var idle_evict_secs: ?u32 = null;
    var metrics_enabled = true;
    var log_level_flag: ?log.Level = null;
    var decode_share_flag: ?[]const u8 = null;
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
                .nax = trunk.naxStatus(),
            };
            var ver_buf: [512]u8 = undefined;
            var ver_w = std.Io.File.stdout().writerStreaming(io, &ver_buf);
            version_mod.writeReport(&ver_w.interface, info) catch {};
            ver_w.interface.flush() catch {};
            return;
        } else if (std.mem.eql(u8, args[i], "--guest-manifest")) {
            var mlx_ver = mlx.mlx_string_new();
            defer _ = mlx.mlx_string_free(mlx_ver);
            try mlx.check(mlx.mlx_version(&mlx_ver));
            var out_buf: [1024]u8 = undefined;
            var out_w = std.Io.File.stdout().writerStreaming(io, &out_buf);
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
            model_dir = cli_mod.resolveRelativeDir(allocator, io, args[i]) orelse args[i];
        } else if (std.mem.eql(u8, args[i], "--port") and i + 1 < args.len) {
            i += 1;
            port_flag = flagInt(u16, "--port", args[i], 0);
        } else if (std.mem.eql(u8, args[i], "--host") and i + 1 < args.len) {
            i += 1;
            host_flag = args[i];
        } else if (std.mem.eql(u8, args[i], "--serve")) {
            serve_mode = true;
            serve_explicit = true;
        } else if (std.mem.eql(u8, args[i], "--think")) {
            const f = cli_mod.parseThinkFlag(if (i + 1 < args.len) args[i + 1] else null) catch {
                log.err("--think requires a model-supported value: " ++ help.think_values ++ "\n", .{});
                std.process.exit(1);
            };
            run_opts.think = f.think;
            if (f.consumed) i += 1;
        } else if (std.mem.eql(u8, args[i], "--tool") and i + 1 < args.len) {
            i += 1;
            run_opts.tools = cli_mod.parseToolSwitch(args[i]) orelse {
                log.err("--tool takes on or off, not '{s}'\n", .{args[i]});
                std.process.exit(1);
            };
            tool_flag_given = true;
        } else if (std.mem.eql(u8, args[i], "--edit") and i + 1 < args.len) {
            i += 1;
            run_opts.edit = cli_mod.parseToolSwitch(args[i]) orelse {
                log.err("--edit takes on or off, not '{s}'\n", .{args[i]});
                std.process.exit(1);
            };
            server_state.g_web_edit = run_opts.edit;
        } else if (std.mem.eql(u8, args[i], "--stream")) {
            stream_mode = true;
        } else if (cli_mod.isPromptFlag(args[i]) and i + 1 < args.len) {
            i += 1;
            prompt = args[i];
        } else if (std.mem.eql(u8, args[i], "--max-tokens") and i + 1 < args.len) {
            i += 1;
            max_tokens = flagInt(u32, "--max-tokens", args[i], 0);
            serve_default_max_tokens = max_tokens;
        } else if (std.mem.eql(u8, args[i], "--temp") and i + 1 < args.len) {
            i += 1;
            temp_flag = flagFloat("--temp", args[i]);
        } else if (std.mem.eql(u8, args[i], "--top-p") and i + 1 < args.len) {
            i += 1;
            top_p_flag = flagFloat("--top-p", args[i]);
        } else if (std.mem.eql(u8, args[i], "--top-k") and i + 1 < args.len) {
            i += 1;
            top_k_flag = flagInt(u32, "--top-k", args[i], 0);
        } else if (std.mem.eql(u8, args[i], "--ctx-size") and i + 1 < args.len) {
            i += 1;
            ctx_size = flagInt(u32, "--ctx-size", args[i], 0);
        } else if (std.mem.eql(u8, args[i], "--config-overrides") and i + 1 < args.len) {
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
        } else if (std.mem.eql(u8, args[i], "--timeout") and i + 1 < args.len) {
            i += 1;
            timeout = flagInt(u32, "--timeout", args[i], 0);
        } else if (std.mem.eql(u8, args[i], "--no-vision")) {
            no_vision = true;
        } else if (std.mem.eql(u8, args[i], "--vision")) {
            vision = true;
        } else if (std.mem.eql(u8, args[i], "--vision-bf16")) {
            @import("vision_common.zig").vision_bf16 = true;
        } else if (std.mem.eql(u8, args[i], "--no-prevent-sleep")) {
            sleep_inhibit_mod.setEnabled(false);
        } else if (std.mem.eql(u8, args[i], "--skip-mem-preflight")) {
            skip_mem_preflight = true;
        } else if (std.mem.eql(u8, args[i], "--no-tool-autocorrect")) {
            server_state.g_tool_autocorrect = false;
        } else if (cli_mod.isNoMtpLookupFlag(args[i])) {
            generate_mod.mtp_lookup_disabled = true;
        } else if (std.mem.eql(u8, args[i], "--drafter") and i + 1 < args.len) {
            i += 1;
            drafter_dir = args[i];
        } else if (std.mem.eql(u8, args[i], "--metrics")) {
            metrics_enabled = true;
        } else if (std.mem.eql(u8, args[i], "--no-metrics")) {
            metrics_enabled = false;
        } else if (std.mem.eql(u8, args[i], "--api-key") and i + 1 < args.len) {
            i += 1;
            // Borrowed from argv (lives for the process). Empty ⇒ leave open.
            if (args[i].len > 0) server_state.g_api_key = args[i];
        } else if (std.mem.eql(u8, args[i], "--api-key-strict")) {
            server_state.g_api_key_strict = true;
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
                    if (key.len > 0) server_state.g_api_key = key;
                }
            }
        } else if (std.mem.eql(u8, args[i], "--drafter-min-depth") or std.mem.eql(u8, args[i], "--drafter-max-depth")) {
            const is_min = std.mem.eql(u8, args[i], "--drafter-min-depth");
            i += 1;
            const text: []const u8 = if (i < args.len) args[i] else "";
            const n = dflash_mod.parseDrafterDepth(text) catch {
                log.err("{s}: expected an integer in {d}..{d}; got '{s}'\n", .{ args[i - 1], dflash_mod.QWEN_MIN_ROWS, dflash_mod.DRAFTER_DEPTH_MAX, text });
                std.process.exit(1);
            };
            if (is_min) dflash_mod.drafter_min_depth = n else dflash_mod.drafter_max_depth = n;
        } else if (std.mem.eql(u8, args[i], "--no-drafter")) {
            no_drafter = true;
        } else if (std.mem.eql(u8, args[i], "--no-mtp")) {
            mtp_flag = false;
        } else if (std.mem.eql(u8, args[i], "--mtp")) {
            mtp_flag = true;
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
        } else if (std.mem.eql(u8, args[i], "--mtp-tokenv3") and i + 1 < args.len) {
            i += 1;
            mtp_tokenv3_raw = args[i];
        } else if (std.mem.eql(u8, args[i], "--mtp-greedy-tail")) {
            settings_resolve.launch.mtp_greedy_tail = true;
        } else if (std.mem.eql(u8, args[i], "--max-mtp-ctx") and i + 1 < args.len) {
            i += 1;
            generate_mod.max_mtp_ctx = flagInt(u32, "--max-mtp-ctx", args[i], 0);
        } else if (std.mem.eql(u8, args[i], "--log-level") and i + 1 < args.len) {
            i += 1;
            const level = log.Level.fromString(args[i]) orelse {
                log.err("--log-level: expected error, warn, info or debug; got '{s}'\n", .{args[i]});
                std.process.exit(1);
            };
            log.setLevel(level);
            log_level_flag = level;
        } else if (std.mem.eql(u8, args[i], "--log-file") and i + 1 < args.len) {
            i += 1;
            log_file_arg = args[i];
        } else if (std.mem.eql(u8, args[i], "--parent-pid") and i + 1 < args.len) {
            i += 1;
            parent_pid = parent_watch.parseArg(args[i]) catch {
                log.err("--parent-pid: expected a process id above 1; got '{s}'\n", .{args[i]});
                std.process.exit(1);
            };
            server_state.host_managed = true;
        } else if (std.mem.eql(u8, args[i], "--no-update-check")) {
            no_update_check = true;
        } else if (std.mem.eql(u8, args[i], "--prefill-chunk") and i + 1 < args.len) {
            i += 1;
            // `explicit` disables the machine-sized pin, so only a real width earns it.
            settings_resolve.prefill_chunk = .{ .value = flagInt(usize, "--prefill-chunk", args[i], 1), .source = .flag };
        } else if (std.mem.eql(u8, args[i], "--prefix-cache-entries") and i + 1 < args.len) {
            i += 1;
            server_state.prefix_cache_capacity = flagInt(u32, "--prefix-cache-entries", args[i], 0);
        } else if (std.mem.eql(u8, args[i], "--prefix-cache-mem") and i + 1 < args.len) {
            // Wave 1.B — KV-bytes budget for the hot prefix cache. Accepts
            // bare numbers (bytes), a suffix of `MB`/`GB`/`KB` (case-
            // insensitive); `0`/`off` keeps RAM retention off.
            i += 1;
            server_state.prefix_cache_mem_bytes = parseSizeArg(args[i]) catch {
                log.err("--prefix-cache-mem: expected '<n>{{MB,GB,KB}}' or '0'/'off'; got '{s}'\n", .{args[i]});
                std.process.exit(1);
            };
            server_state.prefix_cache_mem_source = .flag;
            server_state.prefix_cache_ram_enabled = server_state.ramRetentionForMemArg(server_state.prefix_cache_mem_bytes);
        } else if (std.mem.eql(u8, args[i], "--prefix-cache-disk") and i + 1 < args.len) {
            // SSD tier for the hot prefix cache: previously-seen prefixes are
            // persisted as chunked safetensors and restored across restarts
            // and RAM evictions instead of recomputed. Byte budget with the
            // same size grammar as --prefix-cache-mem; `0`/`off` disables.
            i += 1;
            server_state.prefix_cache_disk_bytes = parseSizeArg(args[i]) catch {
                log.err("--prefix-cache-disk: expected '<n>{{MB,GB,KB}}' or '0'/'off'; got '{s}'\n", .{args[i]});
                std.process.exit(1);
            };
            server_state.prefix_cache_disk_source = .flag;
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
        } else if (std.mem.eql(u8, args[i], "--ssm-checkpoint-max") and i + 1 < args.len) {
            i += 1;
            server_state.ssm_checkpoint_max = flagInt(u32, "--ssm-checkpoint-max", args[i], 0);
        } else if (std.mem.eql(u8, args[i], "--wired-margin-gib") and i + 1 < args.len) {
            i += 1;
            server_state.wired_limit_margin_bytes = server_mod.parseWiredMarginGib(args[i]) catch {
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
            server_state.max_concurrent = flagInt(u32, "--max-concurrent", args[i], 0);
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
            max_resident_models = flagInt(u32, "--max-resident-models", args[i], 1);
        } else if (std.mem.eql(u8, args[i], "--max-resident-mem") and i + 1 < args.len) {
            // Plan 05 Phase D: cap on summed resident bytes. Accepts the
            // same suffixes as --prefix-cache-mem. Special string "auto"
            // (or default 0) → 80% of mlx_set_wired_limit at server start.
            i += 1;
            if (std.mem.eql(u8, args[i], "auto")) {
                if (max_resident_mem_flag != null) max_resident_mem_flag = 0;
            } else {
                max_resident_mem_flag = parseSizeArg(args[i]) catch {
                    log.err("--max-resident-mem: expected '<n>{{MB,GB,KB}}' or 'auto'; got '{s}'\n", .{args[i]});
                    std.process.exit(1);
                };
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
            const n = flagInt(u32, "--idle-evict-secs", args[i], 0);
            idle_evict_secs = if (n > 0) n else null;
        } else if (std.mem.eql(u8, args[i], "--kv-quant") and i + 1 < args.len) {
            i += 1;
            kv_quant_flag = trunk.KVQuantConfig.fromJsonValue(.{ .string = args[i] }) orelse {
                log.err("--kv-quant: expected one of {{16, 8, 6, 4}} (16 = BF16, also `off`); got '{s}'\n", .{args[i]});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, args[i], "--ssd-budget-gb") and i + 1 < args.len) {
            i += 1;
            ssd_budget_bytes = server_mod.parseSsdBudgetGb(args[i]) catch {
                log.err("--ssd-budget-gb: expected an integer in 1..512; got '{s}'\n", .{args[i]});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, args[i], "--ssd-prefetch") and i + 1 < args.len) {
            i += 1;
            model_mod.ssd_prefetch = model_mod.parseSsdPrefetch(args[i]) orelse {
                log.err("--ssd-prefetch: expected an integer in 0..100; got '{s}'\n", .{args[i]});
                std.process.exit(1);
            };
        } else if (arch_flags.parseOption(args, &i)) |parsed| {
            switch (parsed) {
                .applied => {},
                .refused => |r| {
                    log.err("{s}: {s}; got '{s}'\n", .{ r.name, r.expected, r.got });
                    std.process.exit(1);
                },
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

    if (bench_run != null and (host_flag != null or port_flag != null or prompt != null)) {
        log.err("sushi bench --model: --host, --port and --prompt do not apply, it starts no server\n", .{});
        std.process.exit(1);
    }
    if (tool_flag_given and run_model_dir == null) {
        log.err("--tool: only valid with `sushi run <model>`\n", .{});
        std.process.exit(1);
    }
    if (stream_mode and prompt == null) {
        log.err("--stream: only valid with --prompt\n", .{});
        std.process.exit(1);
    }

    if (prompt != null and (serve_explicit or (use_default_models_root and run_model_dir == null) or host_flag != null or port_flag != null or run_opts.tools)) {
        log.err("--prompt/-p cannot be combined with serve, --serve, --host, --port, or --tool on; it uses a private local listener and exits after one reply.\n", .{});
        return error.PromptModeConflict;
    }
    const bind = if (prompt != null or bench_run != null)
        http.resolveBind(host_flag, 0, server_state.default_port)
    else
        http.resolveBind(host_flag, port_flag, model_settings_mod.defaultPort(allocator, io));
    const host = bind.host;
    const port = bind.port;

    if (decode_share_flag) |text| {
        scheduler_mod.prefill_decode_share = @import("sched_policy.zig").parseDecodeShare(text) catch {
            log.err("--prefill-decode-share: expected a number >= 0 (above 0.9 clamps), got '{s}'\n", .{text});
            std.process.exit(1);
        };
        log.info("[prefill] decode share: configured={d}, effective={d} (--prefill-decode-share)\n", .{
            scheduler_mod.prefill_decode_share, scheduler_mod.prefillDecodeShare(),
        });
    }

    const acceptance_flag = mtp_acceptance.parse(mtp_typical_raw, mtp_tokenv3_raw) catch |err| {
        log.err("MTP acceptance settings: {s} (--mtp-typical needs d > 0; --mtp-tokenv3 needs 0 <= a <= 1; choose one)\n", .{@errorName(err)});
        std.process.exit(1);
    };
    if (mtp_typical_raw != null or mtp_tokenv3_raw != null) settings_resolve.launch.mtp_acceptance = acceptance_flag;
    const temperature: f32 = temp_flag orelse 0.0;
    var acceptance_flag_buf: [128]u8 = undefined;
    pinLaunchFlags(.{
        .mtp_acceptance_flag = model_settings_mod.acceptanceFlagText(&acceptance_flag_buf, mtp_typical_raw, mtp_tokenv3_raw),
        .ctx_size = ctx_size,
        .kv_quant = if (kv_quant_flag) |q| q.wireName() else null,
        .mtp = mtp_flag,
        .vision = if (vision) true else if (no_vision) false else null,
        .models_dir = models_root,
        .no_update_check = no_update_check,
        .skip_mem_preflight = skip_mem_preflight,
        .port = port_flag,
    });
    if (dflash_mod.drafter_min_depth) |lo| if (dflash_mod.drafter_max_depth) |hi| if (lo > hi) {
        log.err("--drafter-min-depth {d} is above --drafter-max-depth {d}\n", .{ lo, hi });
        std.process.exit(1);
    };
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
    effort_mod.think_effort_flag = if (run_opts.think == .effort) run_opts.think.effort else null;
    model_settings_mod.loadEffortTables(allocator, io);
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
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        var file = model_settings_mod.load(allocator, io, model_settings_mod.defaultPath(&path_buf));
        defer file.deinit();
        if (model_settings_mod.server(&file).models_dir) |dir| {
            if (std.Io.Dir.openDirAbsolute(io, dir, .{})) |d| {
                d.close(io);
                allocator.free(default_models_root_storage.?);
                default_models_root_storage = try allocator.dupe(u8, dir);
            } else |_| log.warn("[model-settings] server.models_dir {s} cannot be opened, using the default models directory\n", .{dir});
        }
        models_root = default_models_root_storage;
    }
    {
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        var file = model_settings_mod.load(allocator, io, model_settings_mod.defaultPath(&path_buf));
        defer file.deinit();
        if (server_state.prefix_cache_disk_source == .default) if (model_settings_mod.server(&file).prefix_cache_disk_gb) |gb| {
            server_state.prefix_cache_disk_bytes = @as(u64, gb) << 30;
            server_state.prefix_cache_disk_source = .model_settings;
            log.info("[model-settings] server.prefix_cache_disk_gb {d} sets the SSD prefix cache budget\n", .{gb});
        };
    }

    // `sushi run` on a TTY quiets logs to warn (unless --log-level was
    // given) BEFORE the models-root scan below — discovery's per-directory
    // `[discovery] skip …` info lines would otherwise spam the chat REPL.
    if (repl_after_serve and log_level_flag == null) log.setLevel(.warn);

    // Persist the server log. The macOS app only keeps stderr in a 64 KB
    // in-memory ring, so a server that crashed or was restarted takes its
    // history with it — exactly when you need it (see the 2026-07-08 pi
    // session post-mortem). Serving paths only; `pull`/`list` stay quiet.
    //
    // NOTE: opened AFTER `--log-level` is parsed (so the level gates what
    // reaches disk) and BEFORE model discovery/loading (so weight-load and
    // auto-context lines land in the file).
    if (bench_run) |r| {
        try r.prepare();
        if (log_file_arg == null) log_file_arg = try r.logPath();
    }
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
    {
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        var file = model_settings_mod.load(allocator, io, model_settings_mod.defaultPath(&path_buf));
        defer file.deinit();
        const srv = model_settings_mod.server(&file);
        scheduler_mod.setSkipMemPreflight(model_settings_mod.skipMemPreflight(skip_mem_preflight, srv.skip_mem_preflight).value);
        const serving = serve_mode and prompt == null and bench_run == null;
        if (serving) update_mod.startDailyCheck(io, no_update_check, srv.update_check);
        cli_mod.configurePackUpdates(io, serving and !no_update_check, model_settings_mod.autoCheck(no_update_check, srv.pack_update_check).value);
    }

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
    if (serve_mode and bench_run == null) {
        http.ensurePortFree(io, host, port) catch |err| {
            var msg_buf: [512]u8 = undefined;
            if (http.startupRefusal(err, host, port, &msg_buf)) |msg| {
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

    // Observability: the process-wide core is on before serving; `--no-metrics` keeps it off
    // until POST /metrics/enable. Every LoadParams builder reads it back as the scheduler's initial sink.
    if (metrics_enabled) _ = metrics_mod.process.enable();

    // An unsupported file format is refused by name before any config.json read.
    if (isGgufPath(io, model_dir)) {
        log.err("[load] {s}: unsupported model format. Serve an MLX safetensors checkpoint (" ++ arch_packs.served_types_or ++ ").\n", .{model_dir});
        std.process.exit(1);
    }

    // Print MLX version
    var ver = mlx.mlx_string_new();
    defer _ = mlx.mlx_string_free(ver);
    try mlx.check(mlx.mlx_version(&ver));
    log.info("sushi {s} (MLX {s})\n", .{ VERSION, mlx.mlx_string_data(ver) });

    const launch_serve = LaunchServe{
        .ctx_size = ctx_size,
        .timeout = timeout,
        .max_tokens = serve_default_max_tokens,
        .temperature = temp_flag,
        .top_p = top_p_flag,
        .top_k = top_k_flag,
    };

    // Echo the resolved arguments — makes drafter/target mismatches obvious
    // from the log without having to scroll through the whole launch line in
    // the parent's process listing.
    log.info("[args] model: {s}\n", .{model_dir});
    if (drafter_dir) |dir| {
        log.info("[args] drafter: {s}\n", .{dir});
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
    if (kv_quant_flag) |kv| switch (kv.scheme) {
        .off => log.info("[args] kv-quant: off\n", .{}),
        .affine => log.info("[args] kv-quant: affine {d}-bit (group={d})\n", .{ kv.bits, kv.group_size }),
    } else {
        log.info("[args] kv-quant: model default (resolved at load)\n", .{});
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
    server_state.disk_writer_permit_bytes = @import("kv_disk_writer.zig").permitForRam(@import("status.zig").getTotalMemBytes());

    if (serve_mode) {
        // Headless boot: `--model-dir` given, no `--model`. Start with no
        // primary model; everything (chat + media) loads on demand through the
        // registry. The app uses this so a single server hosts chat + image +
        // audio + video, coexisting under one memory budget.
        if (model_dir.len == 0) {
            const discovery_for_registry = discovery_storage;
            discovery_storage = null; // ownership moves to the registry
            try runHeadlessServe(io, allocator, discovery_for_registry, host, port, launch_serve, drafter_dir orelse "", no_drafter, max_resident_models, max_resident_mem_flag, idle_evict_secs, kv_quant_flag, mtp_flag);
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
    model_load.applyModelSettings(config, model_settings_mod.overrideFor(allocator, io, model_dir));
    if (effort_mod.think_effort_flag) |e| if (effort_mod.servedArms(config.model_type)) |served| {
        var choices: [std.enums.values(effort_mod.Effort).len]effort_mod.Effort = undefined;
        if (effort_mod.resolve(served.slice(), e)) |pick| {
            if (pick.rerouted_from) |from| log.info("[effort] " ++ effort_mod.effort_reroute_notice ++ "\n", .{ config.model_type, @tagName(from), @tagName(pick.arm.effort) });
        } else {
            var w = std.Io.Writer.Allocating.init(allocator);
            defer w.deinit();
            try cli_mod.writeEffortRefusal(&w.writer, @tagName(e), config.model_type, effort_mod.listedWords(served.slice(), &choices));
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
        const effective_max_resident_mem = autoResidentMemBytes(max_resident_mem_flag);
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
        registry.mem_cap_binds_alone = max_resident_mem_flag != null;

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

        const params = model_load.LoadParams{
            .registry = registry,
            .entry = entry,
            .config = config,
            .tok = tok,
            .chat_config = chat_config,
            .model_dir = model_dir,
            .ctx_size = ctx_size,
            .drafter_dir = drafter_dir orelse "",
            .no_drafter = no_drafter,
            .mtp = mtp_flag,
            .mtp_head_kv_quant = trunk.Trunk.mtp_head_kv_quant_flag,
            .mtp_depth = depth_bounds.active.max,
            .load_vision = load_vision,
            .kv_quant = kv_quant_flag,
            .prefix_cache_capacity = server_state.prefix_cache_capacity,
            .prefix_cache_ram_enabled = server_state.prefix_cache_ram_enabled,
            .prefix_cache_mem_bytes = server_state.prefix_cache_mem_bytes,
            .prefix_cache_mem_resolver = server_mod.prefixCacheMemForLoad,
            .prefix_cache_disk_bytes = server_state.prefix_cache_disk_bytes,
            .prefix_cache_disk_resolver = server_mod.prefixCacheDiskForLoad,
            .ssd_budget_bytes = ssd_budget_bytes,
            .expert_cache_fit_resolver = server_mod.expertCacheFitForLoad,
            .ssm_checkpoint_stride = server_mod.effectiveSsmCheckpointStride(server_state.ssm_checkpoint_stride, server_state.prefix_cache_capacity, server_state.prefix_cache_ram_enabled, server_state.prefix_cache_disk_bytes),
            .ssm_checkpoint_max = server_state.ssm_checkpoint_max,
            .tokenize_cache_entries = server_mod.tokenize_cache_entries,
            .metrics = metrics_mod.current(),
        };
        var serve_cfg = launch_serve.serverConfig();
        serve_cfg.on_ready = if (prompt != null) PromptClient.ready else null;
        serve_cfg.embedded_client = if (bench_run != null) bench_mod.embeddedClient else null;
        serve_cfg.default_force_mtp = mtp_flag == true;
        try server_mod.serve(io, allocator, params, config, host, port, serve_cfg);
        if (prompt_state.err) |err| return err;
        if (bench_run) |r| if (r.err) |err| return err;
    } else {
        // ── Offline single-prompt mode. mlx ops run on this thread, no
        //    scheduler. The same load path as pre-A1.
        if (config.expertStreamingRequired()) {
            log.err("This checkpoint requires expert streaming. Use `run <model>` or `--serve` with --ssd-budget-gb.\n", .{});
            return error.ExpertStreamingRequired;
        }
        log.info("Loading weights...\n", .{});
        var weights = try model_mod.loadWeightsForConfig(io, allocator, model_dir, config, load_vision);
        defer weights.deinit();
        model_mod.resolveWeightPrefix(config, &weights);

        var xfm = try trunk.Trunk.init(io, allocator, config.*, &weights);
        defer xfm.deinit();

        xfm.round_cost.layout = round_cost_mod.layoutFor(config);

        // Reserved-token suppression, same derivation as the serve path.
        config.mergeTokenizerStops(tok);
        sampler.installSuppressMask(&xfm, tok, chat_config.chat_template, config.eosTokenSlice());
        generate_mod.installThinkMarkers(&xfm, tok);
        try generate_mod.installLogitBias(io, &xfm, tok);

        // Honor --kv-quant in offline mode too. The serve path threads this
        // through Slot caches via the scheduler; here we swap the
        // Trunk's own legacy cache to match.
        const kv_cache = server_mod.kvCacheForLaunch(config, kv_quant_flag);
        log.info("[kv-cache] {s} ({s})\n", .{ kv_cache.label(), kv_cache.sourceName() });
        const mtp_choice = model_load.mtpChoiceFor(mtp_flag, config);
        log.info("[mtp] {s} ({s})\n", .{ mtp_choice.label(), mtp_choice.sourceName() });
        if (kv_cache.config.scheme != .off) {
            try xfm.cache.reinit(config.num_hidden_layers, kv_cache.config);
        }
        try arch_state.applyHeadKvQuant(&xfm, kv_cache.config);

        // JIT-compile + wire memory limits (policy: mlx.applyWiredPolicy).
        {
            if (mlx.applyWiredPolicy()) |limit| log.debug("[wired] limit={d} MB\n", .{limit / (1024 * 1024)});
        }
        if (config.hidden_act == .gelu_approx) {
            xfm.compileGelu();
            xfm.compileGeglu();
        }
        if (xfm.moe_layers != null) {
            xfm.compileMoeRouting();
        }
        arch_state.compile(&xfm, .gates);
        log.info("Model ready.\n", .{});

        const user_prompt = prompt orelse "What is 2+2? Answer in one sentence.";
        const messages = [_]chat_mod.Message{
            .{ .role = "user", .content = user_prompt },
        };

        const prompt_ids = try chat_mod.formatChat(allocator, tok, &messages, chat_config, null, null, false, null, false);
        defer allocator.free(prompt_ids);

        // Reset peak memory before generation
        _ = mlx.mlx_reset_peak_memory();

        const eos_slice = config.eosTokenSlice();
        const sampling = sampler.SamplingParams{ .temperature = temperature };

        var stdout_buf: [16 * 1024]u8 = undefined;
        var stdout_w_state = std.Io.File.stdout().writerStreaming(io, &stdout_buf);
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
                const piece = try tok.decode(allocator, &ids);
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
            const result = try generate_mod.generate(io, allocator, &xfm, tok, prompt_ids, max_tokens, sampling, eos_slice, 0, 0);
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
fn autoResidentMemBytes(flag: ?u64) u64 {
    if (flag) |bytes| return bytes;
    var dev = mlx.mlx_device{ .ctx = null };
    _ = mlx.mlx_get_default_device(&dev);
    defer _ = mlx.mlx_device_free(dev);
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
    max_tokens: u32,
    temperature: ?f32,
    top_p: ?f32,
    top_k: ?u32,

    fn serverConfig(self: LaunchServe) server_state.ServerConfig {
        return .{
            .max_context_size = self.ctx_size,
            .request_timeout_sec = self.timeout,
            .default_max_tokens = self.max_tokens,
            .default_temperature = self.temperature,
            .default_top_p = self.top_p,
            .default_top_k = self.top_k,
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
    max_resident_mem_flag: ?u64,
    idle_evict_secs: ?u32,
    kv_quant_flag: ?trunk.KVQuantConfig,
    mtp_flag: ?bool,
) !void {
    log.info("sushi {s} (headless — models load on demand)\n", .{VERSION});
    log.info("[args] serve: {s}:{d}\n", .{ host, port });

    var stub = try model_load.headlessStubCpuState(allocator);
    defer model_load.freeCpuState(allocator, &stub);

    const effective_max_resident_mem = autoResidentMemBytes(max_resident_mem_flag);
    if (effective_max_resident_mem > 0) {
        log.info("[registry] max_resident_models={d}, max_resident_mem={d:.1} GB\n", .{ max_resident_models, @as(f64, @floatFromInt(effective_max_resident_mem)) / 1_073_741_824.0 });
    } else {
        log.info("[registry] max_resident_models={d}, max_resident_mem=unlimited\n", .{max_resident_models});
    }

    const registry = try model_registry_mod.ModelRegistry.init(allocator, io, discovery, max_resident_models, effective_max_resident_mem, idle_evict_secs);
    defer registry.deinit();
    registry.mem_cap_binds_alone = max_resident_mem_flag != null;

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
        .handle = null,
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

    const params = model_load.LoadParams{
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
        .kv_quant = kv_quant_flag,
        .mtp = mtp_flag,
        .mtp_head_kv_quant = trunk.Trunk.mtp_head_kv_quant_flag,
        // Seed the scheduler's prefix-cache config from the server globals so
        // on-demand (headless/discover-mode) loads get the SAME hot prefix
        // cache as a `--model` startup load. Previously hardcoded to 0, which
        // left `Scheduler.prefix_cache_capacity == 0` → every model loaded via
        // `ensureLoaded` skipped `HotPrefixCache` init → cross-turn KV reuse
        // was silently dead for the entire headless serving mode (the default
        // `serve` path). Mirrors the LoadParams built in `main()`.
        .prefix_cache_capacity = server_state.prefix_cache_capacity,
        .prefix_cache_ram_enabled = server_state.prefix_cache_ram_enabled,
        .prefix_cache_mem_bytes = server_state.prefix_cache_mem_bytes,
        .prefix_cache_mem_resolver = server_mod.prefixCacheMemForLoad,
        .prefix_cache_disk_resolver = server_mod.prefixCacheDiskForLoad,
        .ssd_budget_bytes = ssd_budget_bytes,
        .expert_cache_fit_resolver = server_mod.expertCacheFitForLoad,
        .ssm_checkpoint_stride = server_state.ssm_checkpoint_stride,
        .ssm_checkpoint_max = server_state.ssm_checkpoint_max,
        .tokenize_cache_entries = server_mod.tokenize_cache_entries,
        .metrics = metrics_mod.current(),
    };

    var serve_cfg = launch_serve.serverConfig();
    // On-demand MLX loads auto-attach an MTP sidecar (MTP is on unless a flag or the
    // file says off), so the MoE force flag has to reach this path too.
    serve_cfg.default_force_mtp = mtp_flag == true;
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

/// A flag's whole-number value, or a named error and exit: a bad value never falls back to a default.
/// Records which launch flags were given, so the settings API shows their keys locked: a flag outranks the file.
fn pinLaunchFlags(given: struct { mtp_acceptance_flag: ?[]const u8, ctx_size: u32, kv_quant: ?[]const u8, mtp: ?bool, vision: ?bool, models_dir: ?[]const u8, no_update_check: bool, skip_mem_preflight: bool = false, port: ?u16 = null }) void {
    const locks = &model_settings_mod.launch_locks;
    if (given.ctx_size > 0) locks.modelMut("ctx_size").pin("{d}", .{given.ctx_size});
    if (given.kv_quant) |q| locks.modelMut("kv_quant").pin("{s}", .{q});
    if (given.mtp) |m| locks.modelMut("mtp").pin("{s}", .{if (m) "on" else "off"});
    if (given.mtp_acceptance_flag) |f| locks.modelMut("mtp_acceptance").pin("{s}", .{f});
    if (settings_resolve.launch.mtp_greedy_tail != null) locks.modelMut("mtp_greedy_tail").pin("--mtp-greedy-tail", .{});
    if (ssd_budget_bytes != 0) locks.modelMut("ssd_budget_gb").pin("{d}", .{ssd_budget_bytes >> 30});
    if (model_mod.ssd_prefetch) |p| locks.modelMut("ssd_prefetch").pin("{d}", .{p});
    if (model_settings_mod.preserve_thinking_flag) |p| locks.modelMut("preserve_thinking").pin("{s}", .{if (p) "on" else "off"});
    if (model_settings_mod.think_penalty_flag) |l| locks.modelMut("think_penalty").pin("{d}", .{l});
    if (serve_default_max_tokens > 0) locks.modelMut("max_tokens").pin("{d}", .{serve_default_max_tokens});
    if (given.vision) |v| locks.modelMut("vision").pin("{s}", .{if (v) "on" else "off"});
    if (server_state.prefix_cache_disk_source != .default) {
        var size_buf: [32]u8 = undefined;
        locks.prefix_cache_disk_gb.pin("{s}", .{model_settings_mod.sizeText(&size_buf, server_state.prefix_cache_disk_bytes)});
    }
    if (given.models_dir) |d| locks.models_dir.pin("{s}", .{d});
    if (given.port) |p| locks.port.pin("{d}", .{p});
    if (given.skip_mem_preflight) locks.skip_mem_preflight.pin("--skip-mem-preflight", .{});
    if (given.no_update_check) {
        locks.update_check.pin("--no-update-check", .{});
        locks.pack_update_check.pin("--no-update-check", .{});
    }
}

fn flagInt(comptime T: type, flag: []const u8, text: []const u8, min: T) T {
    return cli_mod.parseFlagInt(T, text, min) orelse {
        log.err("{s}: expected a whole number of at least {d}; got '{s}'\n", .{ flag, min, text });
        std.process.exit(1);
    };
}

fn flagFloat(flag: []const u8, text: []const u8) f32 {
    return cli_mod.parseFlagFloat(text) orelse {
        log.err("{s}: expected a number; got '{s}'\n", .{ flag, text });
        std.process.exit(1);
    };
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
        .max_tokens = 4096,
        .temperature = 0.3,
        .top_p = 0.8,
        .top_k = 20,
    };
    const cfg = launch.serverConfig();
    try std.testing.expectEqual(@as(?f32, 0.3), cfg.default_temperature);
    try std.testing.expectEqual(@as(?f32, 0.8), cfg.default_top_p);
    try std.testing.expectEqual(@as(?u32, 20), cfg.default_top_k);
    try std.testing.expectEqual(@as(u32, 4096), cfg.default_max_tokens);
    try std.testing.expectEqual(@as(u32, 32768), cfg.max_context_size);

    var unset = launch;
    unset.temperature = null;
    unset.top_p = null;
    unset.top_k = null;
    const none = unset.serverConfig();
    try std.testing.expectEqual(@as(?f32, null), none.default_temperature);
    try std.testing.expectEqual(@as(?u32, null), none.default_top_k);
}

test "launch flags: --port pins server.port as typed, and no flag leaves it free" {
    const saved = model_settings_mod.launch_locks;
    defer model_settings_mod.launch_locks = saved;
    model_settings_mod.launch_locks = .{};
    pinLaunchFlags(.{ .mtp_acceptance_flag = null, .ctx_size = 0, .kv_quant = null, .mtp = null, .vision = null, .models_dir = null, .no_update_check = false });
    try std.testing.expect(!model_settings_mod.launch_locks.port.set);
    pinLaunchFlags(.{ .mtp_acceptance_flag = null, .ctx_size = 0, .kv_quant = null, .mtp = null, .vision = null, .models_dir = null, .no_update_check = false, .port = 8080 });
    try std.testing.expectEqualStrings("8080", model_settings_mod.launch_locks.port.text());
}

test "launch flags: --no-update-check pins both automatic update checks, and no flag pins neither" {
    const saved = model_settings_mod.launch_locks;
    defer model_settings_mod.launch_locks = saved;
    model_settings_mod.launch_locks = .{};
    pinLaunchFlags(.{ .mtp_acceptance_flag = null, .ctx_size = 0, .kv_quant = null, .mtp = null, .vision = null, .models_dir = null, .no_update_check = false });
    try std.testing.expect(!model_settings_mod.launch_locks.update_check.set);
    try std.testing.expect(!model_settings_mod.launch_locks.pack_update_check.set);
    pinLaunchFlags(.{ .mtp_acceptance_flag = null, .ctx_size = 0, .kv_quant = null, .mtp = null, .vision = null, .models_dir = null, .no_update_check = true });
    try std.testing.expectEqualStrings("--no-update-check", model_settings_mod.launch_locks.update_check.text());
    try std.testing.expectEqualStrings("--no-update-check", model_settings_mod.launch_locks.pack_update_check.text());
}

test "launch flags: --skip-mem-preflight pins server.skip_mem_preflight, and no flag leaves it free" {
    const saved = model_settings_mod.launch_locks;
    defer model_settings_mod.launch_locks = saved;
    model_settings_mod.launch_locks = .{};
    pinLaunchFlags(.{ .mtp_acceptance_flag = null, .ctx_size = 0, .kv_quant = null, .mtp = null, .vision = null, .models_dir = null, .no_update_check = false });
    try std.testing.expect(!model_settings_mod.launch_locks.skip_mem_preflight.set);
    pinLaunchFlags(.{ .mtp_acceptance_flag = null, .ctx_size = 0, .kv_quant = null, .mtp = null, .vision = null, .models_dir = null, .no_update_check = false, .skip_mem_preflight = true });
    try std.testing.expectEqualStrings("--skip-mem-preflight", model_settings_mod.launch_locks.skip_mem_preflight.text());
}
