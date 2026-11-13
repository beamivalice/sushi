const std = @import("std");
const mlx = @import("mlx.zig");
const gpu_mem = @import("gpu_mem.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const chat_mod = @import("chat.zig");
const tokenizer_mod = @import("tokenizer.zig");
const model_handle = @import("model_handle.zig");
const trunk = @import("trunk.zig");
const expert_stream_mod = @import("expert_stream.zig");
const model_load = @import("model_load.zig");
const model_settings_mod = @import("model_settings.zig");
const server_mod = @import("server.zig");
const arch_flags = @import("arch/flags.zig");
const arch_state = @import("arch/arch_state.zig");
const server_state = @import("server_state.zig");
const hidden_capture = @import("hidden_capture.zig");
const fp8_block = @import("fp8_block.zig");
const offline = @import("arch/offline.zig");
const fixture_mod = @import("kld_fixture.zig");
const cli_error = @import("cli_error.zig");
const testing = std.testing;

pub const Command = enum { capture, compare };

pub const Options = struct {
    command: Command = .capture,
    help: bool = false,
    model_dir: []const u8 = "",
    prompts: []const u8 = "",
    out_dir: []const u8 = "",
    fixture: []const u8 = "",
    json_out: []const u8 = "",
    label: []const u8 = "sushi",
    tokens: u32 = 64,
    top_k: u32 = 10,
    limit: u32 = 0,
    no_template: bool = false,
    ctx_size: u32 = 0,
    /// Dense, not the serving default: the teacher adds no quantization of its own.
    kv_quant_config: trunk.KVQuantConfig = trunk.KVQuantConfig.dense,
    /// `--qwen-gdn 16` stores the GDN state in bf16, so a student arm can be scored against the f32 teacher.
    qwen_gdn_f32: bool = true,
    /// DeepSeek only, compare only: the student drops this many routed experts a layer by the pack's usage list.
    drop_expert: u16 = 0,
    ssd_budget_bytes: u64 = 0,
    ssd_prefetch: ?u8 = null,
    pick_tolerance: f32 = 0,
    wired_margin_bytes: u64 = 0,
    /// `--mtp` / `--no-mtp`; null = not given (the head stays off).
    mtp: ?bool = null,
    /// `SUSHI_HIDDEN_OUT`: capture appends each prompt forward's block boundaries here.
    hidden_out: []const u8 = "",
    /// `SUSHI_IMATRIX_OUT`: a native GLM capture also accumulates the experts' imatrix and writes it here.
    imatrix_out: []const u8 = "",
    /// Layer-major: the imatrix covers only the first N windows (0 = all); later windows still run for the boundaries.
    imatrix_windows: u32 = 0,
    /// Native GLM teacher capture: windows run in batches, layer by layer (resumable).
    layer_major: bool = false,
    /// Windows per layer-major batch; 0 = chosen from the budget.
    batch_windows: u32 = 0,
    /// Layer-major: while this file exists the capture idles after its current batch.
    pause_file: []const u8 = "",
};

pub const ArgError = error{
    MissingSubcommand,
    UnknownSubcommand,
    UnknownFlag,
    MissingFlagValue,
    BadFlagValue,
    MissingModel,
    MissingPrompts,
    MissingOut,
    MissingFixture,
    ConflictingFlags,
    TeacherMustBeLossless,
    LayerMajorFlagWithoutLayerMajor,
};

const ValueFlag = enum {
    model,
    prompts,
    out,
    fixture,
    json,
    label,
    tokens,
    top_k,
    limit,
    ctx_size,
    kv_quant,
    qwen_gdn,
    drop_expert,
    ssd_budget_gb,
    ssd_prefetch,
    expert_pick_tolerance,
    wired_margin_gib,
    batch_windows,
    imatrix_windows,
    pause_file,
};

const value_flags = [_]struct { []const u8, ValueFlag }{
        .{ "--model", .model },
        .{ "--prompts", .prompts },
        .{ "--out", .out },
        .{ "--fixture", .fixture },
        .{ "--json", .json },
        .{ "--label", .label },
        .{ "--tokens", .tokens },
        .{ "--top-k", .top_k },
        .{ "--limit", .limit },
        .{ "--ctx-size", .ctx_size },
        .{ "--kv-quant", .kv_quant },
        .{ "--qwen-gdn", .qwen_gdn },
        .{ "--drop-expert", .drop_expert },
        .{ "--ssd-budget-gb", .ssd_budget_gb },
        .{ "--ssd-prefetch", .ssd_prefetch },
        .{ "--expert-pick-tolerance", .expert_pick_tolerance },
        .{ "--wired-margin-gib", .wired_margin_gib },
        .{ "--batch-windows", .batch_windows },
        .{ "--imatrix-windows", .imatrix_windows },
        .{ "--pause-file", .pause_file },
};

fn valueFlag(name: []const u8) ?ValueFlag {
    for (value_flags) |row| if (std.mem.eql(u8, name, row[0])) return row[1];
    return null;
}

pub fn parseArgs(args: []const []const u8) ArgError!Options {
    if (args.len == 0) return error.MissingSubcommand;
    var o = Options{};
    if (std.mem.eql(u8, args[0], "capture")) {
        o.command = .capture;
    } else if (std.mem.eql(u8, args[0], "compare")) {
        o.command = .compare;
    } else if (std.mem.eql(u8, args[0], "--help") or std.mem.eql(u8, args[0], "-h")) {
        o.help = true;
        return o;
    } else {
        return error.UnknownSubcommand;
    }

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--help") or std.mem.eql(u8, a, "-h")) {
            o.help = true;
            return o;
        } else if (std.mem.eql(u8, a, "--no-template")) {
            o.no_template = true;
        } else if (std.mem.eql(u8, a, "--no-mtp")) {
            if (o.mtp == true) return error.ConflictingFlags;
            o.mtp = false;
        } else if (std.mem.eql(u8, a, "--mtp")) {
            if (o.mtp == false) return error.ConflictingFlags;
            o.mtp = true;
        } else if (std.mem.eql(u8, a, "--layer-major")) {
            o.layer_major = true;
        } else if (valueFlag(a)) |flag| {
            if (i + 1 >= args.len) return error.MissingFlagValue;
            i += 1;
            const v = args[i];
            switch (flag) {
                .model => o.model_dir = v,
                .prompts => o.prompts = v,
                .out => o.out_dir = v,
                .fixture => o.fixture = v,
                .json => o.json_out = v,
                .label => o.label = v,
                .tokens => o.tokens = std.fmt.parseInt(u32, v, 10) catch return error.BadFlagValue,
                .top_k => o.top_k = std.fmt.parseInt(u32, v, 10) catch return error.BadFlagValue,
                .limit => o.limit = std.fmt.parseInt(u32, v, 10) catch return error.BadFlagValue,
                .ctx_size => o.ctx_size = std.fmt.parseInt(u32, v, 10) catch return error.BadFlagValue,
                .kv_quant => o.kv_quant_config = trunk.KVQuantConfig.fromJsonValue(.{ .string = v }) orelse return error.BadFlagValue,
                .qwen_gdn => o.qwen_gdn_f32 = model_mod.parseQwenGdn(v) orelse return error.BadFlagValue,
                .drop_expert => o.drop_expert = @import("expert_usage.zig").parseCount(v) orelse return error.BadFlagValue,
                .ssd_budget_gb => o.ssd_budget_bytes = server_mod.parseSsdBudgetGb(v) catch return error.BadFlagValue,
                .ssd_prefetch => o.ssd_prefetch = model_mod.parseSsdPrefetch(v) orelse return error.BadFlagValue,
                .expert_pick_tolerance => o.pick_tolerance = expert_stream_mod.parsePickTolerance(v) catch return error.BadFlagValue,
                .wired_margin_gib => o.wired_margin_bytes = server_mod.parseWiredMarginGib(v) catch return error.BadFlagValue,
                .batch_windows => o.batch_windows = std.fmt.parseInt(u32, v, 10) catch return error.BadFlagValue,
                .imatrix_windows => o.imatrix_windows = std.fmt.parseInt(u32, v, 10) catch return error.BadFlagValue,
                .pause_file => o.pause_file = v,
            }
        } else {
            return error.UnknownFlag;
        }
    }
    if (o.tokens == 0) return error.BadFlagValue;
    if (!o.layer_major and (o.batch_windows != 0 or o.imatrix_windows != 0 or o.pause_file.len != 0)) return error.LayerMajorFlagWithoutLayerMajor;
    if (o.model_dir.len == 0) return error.MissingModel;
    switch (o.command) {
        .capture => {
            if (o.prompts.len == 0) return error.MissingPrompts;
            if (o.out_dir.len == 0) return error.MissingOut;
            if (o.pick_tolerance > 0 or o.drop_expert != 0) return error.TeacherMustBeLossless;
        },
        .compare => {
            if (o.fixture.len == 0) return error.MissingFixture;
        },
    }
    return o;
}

pub const USAGE =
    \\usage:
    \\  sushi kld capture --model <dir> --prompts <src> --out <dir> [options]
    \\  sushi kld compare --model <dir> --fixture <dir> [options]
    \\
    \\  <src> may be standard4 (bundled two code and two prose prompts),
    \\  a captured fixture dir (its prompts are reused), a directory of
    \\  *.txt files (one prompt each, sorted by name), or a .jsonl of
    \\  {"id":...,"prompt":...} lines; a line may give "prompt_ids":[...]
    \\  (token ids, used as given) instead of "prompt", and with it
    \\  "continuation_ids":[...] that capture scores teacher-forced
    \\  instead of generating a continuation.
    \\
    \\  SUSHI_HIDDEN_OUT=<dir> makes capture append each prompt forward's
    \\  residual at every block boundary to <dir> (boundary-XX.bin, tokens.bin).
    \\
    \\options:
    \\  --model <dir>         pack or checkpoint to capture from / score
    \\  --prompts <src>       capture: the prompt source (see above)
    \\  --out <dir>           capture: fixture output directory
    \\  --fixture <dir>       compare: teacher fixture directory
    \\  --tokens <n>          greedy tokens per prompt to capture (default 64)
    \\  --top-k <n>           top_k recorded in baseline.json (default 10)
    \\  --label <s>           label/run recorded in the output (default sushi)
    \\  --limit <n>           only the first n prompts (0 = all)
    \\  --no-template         feed the raw prompt text, no chat template
    \\  --json <file>         compare: write the numbers as JSON
    \\  --ctx-size <n>        context length override
    \\  --kv-quant <16|8|6|4> KV cache quantization (16 = BF16, also off)
    \\  --qwen-gdn <32|16>    Qwen only: bits of the GatedDeltaNet recurrent state (default 32 =
    \\                        f32; 16 = bf16, the decay gate is f32 either way). A student at 16 is
    \\                        scored against the f32 teacher. A GLM pack has no GDN arm and refuses it.
    \\  --drop-expert <n>     compare only, DeepSeek-V4.1 only: drop the n least-used routed experts of every layer by
    \\                        the pack's expert_usage.json (a multiple of 8; 0 = off). The teacher is never pruned.
    \\  --ssd-budget-gb <n>   expert streaming budget (GiB)
    \\  --ssd-prefetch <n>    streamed-expert prefetch read cap: percent (0..100, default 0) of the SSD read rate; 0 = no prefetch
    \\  --mtp                 keep the MTP head resident (refused under streaming)
    \\  --no-mtp              do not keep it (the default; the teacher runs without a head)
    \\  --expert-pick-tolerance <n>  compare only, LOSSY: swap a missed streamed expert for a cached one within n (0..0.6)
    \\  --wired-margin-gib <n>  headroom under iogpu.wired_limit_mb (integers 1..32)
    \\  --layer-major         native GLM BF16 teacher or a streamed checkpoint's calibration capture, --tokens 1:
    \\                        batches of windows run layer by layer, each layer's experts read once per batch;
    \\                        resumable by rerunning
    \\  --imatrix-windows <n> layer-major: SUSHI_IMATRIX_OUT counts only the first n windows (0 = all)
    \\  --batch-windows <n>   windows per layer-major batch (default: from the budget, at most 32)
    \\  --pause-file <path>   layer-major: idle after the current batch while <path> exists (<path>.ack)
    \\
;

pub fn kvCacheFormat(cfg: trunk.KVQuantConfig) []const u8 {
    return switch (cfg.scheme) {
        .off => "bf16",
        .affine => switch (cfg.bits) {
            4 => "affine4",
            6 => "affine6",
            8 => "affine8",
            else => "affine",
        },
    };
}

pub const Loaded = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    config: model_mod.ModelConfig,
    tok: tokenizer_mod.Tokenizer,
    chat_config: chat_mod.ChatConfig,
    weights: model_mod.Weights,
    xfm: trunk.Trunk,

    pub fn deinit(self: *Loaded) void {
        const allocator = self.allocator;
        self.xfm.deinit();
        self.weights.deinit();
        self.chat_config.deinit();
        self.tok.deinit();
        self.config.deinit(allocator);
        allocator.destroy(self);
    }
};

pub fn loadModel(io: std.Io, allocator: std.mem.Allocator, opts: Options) !*Loaded {
    const self = try allocator.create(Loaded);
    errdefer allocator.destroy(self);
    model_mod.qwen_gdn_f32 = opts.qwen_gdn_f32;
    model_mod.drop_expert = opts.drop_expert;
    model_mod.ssd_prefetch = opts.ssd_prefetch;
    self.* = .{
        .allocator = allocator,
        .io = io,
        .config = try model_mod.parseConfig(io, allocator, opts.model_dir),
        .tok = undefined,
        .chat_config = undefined,
        .weights = undefined,
        .xfm = undefined,
    };
    errdefer self.config.deinit(allocator);
    try offline.refuse(&self.config, opts);
    if (arch_flags.archFlagRefusal(&self.config, .{ .drop_expert = opts.drop_expert != 0, .ssd_prefetch = opts.ssd_prefetch != null })) |why| {
        log.err("{s}\n", .{why});
        return error.LaunchFlagUnsupported;
    }
    model_load.applyModelSettings(&self.config, model_settings_mod.overrideFor(allocator, io, opts.model_dir));
    self.config.ctx_override = model_settings_mod.contextPick(opts.ctx_size, self.config.ctx_override).value;

    const kld_budget = model_load.resolveSsdBudget(opts.ssd_budget_bytes, self.config.ssd_budget_gb_override, self.config.supportsExpertStreaming());
    if (expert_stream_mod.expertStreamingEngaged(
        self.config.supportsExpertStreaming(),
        self.config.expertStreamingRequired(),
        kld_budget.bytes,
    )) {
        const budget = kld_budget;
        if (budget.bytes == 0) return error.ExpertStreamingRequired;
        const mtp = model_settings_mod.MtpChoice.resolve(opts.mtp, self.config.mtp_override, false);
        switch (expert_stream_mod.mtpUnderStreaming(mtp.on, mtp.source == .model_settings, mtp.source == .default, false)) {
            .keep, .refuse => {
                log.err("[expert-stream] KLD scoring under expert streaming runs without the MTP head; drop --mtp\n", .{});
                return error.ExpertStreamingMtpUnsupported;
            },
            .drop_settings => {
                log.info("[expert-stream] model-settings mtp=true ignored: {s}\n", .{expert_stream_mod.MTP_UNSUPPORTED});
                self.config.mtp_override = false;
            },
            .drop_default, .off => {},
        }
        const plan = try model_load.planExpertStreaming(io, allocator, &self.config, opts.model_dir, budget.bytes, false);
        model_load.applyStreamingPlan(&self.config, plan, budget.bytes);
        if (self.config.expert_source_dir == null) self.config.expert_source_dir = try allocator.dupe(u8, opts.model_dir);
        log.info("[kld] expert streaming: cache {d:.2} GB, {d} slots/layer\n", .{
            @as(f64, @floatFromInt(plan.cache.cache_bytes)) / 1e9,
            plan.cache.slots_per_layer,
        });
    }

    var metal_available: bool = false;
    try mlx.check(mlx.mlx_metal_is_available(&metal_available));
    if (metal_available) {
        const gpu = mlx.mlx_device_new_type(.gpu, 0);
        defer _ = mlx.mlx_device_free(gpu);
        try mlx.check(mlx.mlx_set_default_device(gpu));
    }

    self.tok = try tokenizer_mod.loadTokenizer(io, allocator, opts.model_dir);
    errdefer self.tok.deinit();
    self.chat_config = try chat_mod.loadChatConfig(io, allocator, opts.model_dir, self.config.model_type);
    errdefer self.chat_config.deinit();

    self.weights = try model_mod.loadWeightsForConfig(io, allocator, opts.model_dir, &self.config, false);
    errdefer self.weights.deinit();
    model_mod.resolveWeightPrefix(&self.config, &self.weights);

    self.xfm = try trunk.Trunk.init(io, allocator, self.config, &self.weights);
    errdefer self.xfm.deinit();
    if (opts.kv_quant_config.scheme != .off) {
        try self.xfm.cache.reinit(self.config.num_hidden_layers, opts.kv_quant_config);
    }
    _ = mlx.applyWiredPolicy();
    if (self.config.hidden_act == .gelu_approx) {
        self.xfm.compileGelu();
        self.xfm.compileGeglu();
    }
    if (self.xfm.moe_layers != null) self.xfm.compileMoeRouting();
    arch_state.compile(&self.xfm, .gates);
    try offline.afterLoad(&self.xfm, self.config, opts.kv_quant_config);
    return self;
}

fn logitsVocab(logits: mlx.mlx_array) !usize {
    const shape = mlx.getShape(logits);
    if (shape.len != 3) return error.KldLogitsShape;
    return @intCast(shape[2]);
}

pub fn readLastRow(xfm: *trunk.Trunk, logits: mlx.mlx_array, dst: []f32) !void {
    const shape = mlx.getShape(logits);
    if (shape.len != 3 or @as(usize, @intCast(shape[2])) != dst.len) return error.KldLogitsShape;
    var row = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(row);
    if (shape[1] == 1) {
        try mlx.check(mlx.mlx_reshape(&row, logits, &[_]c_int{shape[2]}, 1, xfm.s));
    } else {
        var sliced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced);
        try mlx.check(mlx.mlx_slice(&sliced, logits, &[_]c_int{ 0, shape[1] - 1, 0 }, 3, &[_]c_int{ 1, shape[1], shape[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, xfm.s));
        try mlx.check(mlx.mlx_reshape(&row, sliced, &[_]c_int{shape[2]}, 1, xfm.s));
    }
    var row_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(row_f32);
    try mlx.check(mlx.mlx_astype(&row_f32, row, .float32, xfm.s));
    var contiguous = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contiguous);
    try mlx.check(mlx.mlx_contiguous(&contiguous, row_f32, false, xfm.s));
    try mlx.check(mlx.mlx_array_eval(contiguous));
    const data = mlx.mlx_array_data_float32(contiguous) orelse return error.KldLogitsUnreadable;
    @memcpy(dst, data[0..dst.len]);
}

pub const Out = struct {
    silent: bool = false,

    pub fn print(self: *Out, comptime fmt: []const u8, args: anytype) void {
        if (self.silent) return;
        var buf: [32 * 1024]u8 = undefined;
        const line = std.fmt.bufPrint(&buf, fmt, args) catch return;
        fixture_mod.writeAllFd(1, line) catch return;
    }
};

fn promptIds(allocator: std.mem.Allocator, l: *Loaded, opts: Options, p: fixture_mod.Prompt) ![]u32 {
    if (p.ids) |ids| return allocator.dupe(u32, ids);
    const text = p.text;
    if (opts.no_template) return l.tok.encode(allocator, text);
    const messages = [_]chat_mod.Message{.{ .role = "user", .content = text }};
    return chat_mod.formatChat(allocator, &l.tok, &messages, &l.chat_config, null, null, false, null, false);
}

fn renderPrompt(allocator: std.mem.Allocator, l: *Loaded, opts: Options, p: fixture_mod.Prompt) ![]const u8 {
    if (p.ids) |ids| return l.tok.decode(allocator, ids);
    const text = p.text;
    if (opts.no_template) return allocator.dupe(u8, text);
    const messages = [_]chat_mod.Message{.{ .role = "user", .content = text }};
    return chat_mod.renderChatTemplate(allocator, &messages, &l.chat_config, null, null, false, null, false);
}

/// Prompts prefill in serving-width chunks, identically at capture and compare: the chunk width is part of
/// the numerics, and a long prompt's whole-prompt forward outgrows memory. A prompt within one chunk is one forward.
/// The width is the architecture's serving start width, so no box and no prompt length changes it.
pub fn promptChunk(config: *const model_mod.ModelConfig) usize {
    return config.prefillStartWidth();
}

pub fn forwardPrompt(allocator: std.mem.Allocator, l: *Loaded, ctx: *model_handle.ForwardCtx, ids: []const u32) !mlx.mlx_array {
    return forwardPromptChunks(allocator, l, ctx, ids, promptChunk(&l.config));
}

pub fn forwardPromptChunks(allocator: std.mem.Allocator, l: *Loaded, ctx: *model_handle.ForwardCtx, ids: []const u32, chunk: usize) !mlx.mlx_array {
    if (ids.len == 0 or chunk == 0) return error.EmptyPrompt;
    const prompt_i32 = try allocator.alloc(i32, ids.len);
    defer allocator.free(prompt_i32);
    for (ids, 0..) |t, i| prompt_i32[i] = @intCast(t);
    var logits = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(logits);
    var start: usize = 0;
    while (start < ids.len) {
        const end = @min(ids.len, start + chunk);
        const part = mlx.mlx_array_new_data(prompt_i32[start..end].ptr, &[_]c_int{ 1, @intCast(end - start) }, 2, .int32);
        defer _ = mlx.mlx_array_free(part);
        const next = try l.xfm.forwardWith(ctx, part);
        _ = mlx.mlx_array_free(logits);
        logits = next;
        start = end;
    }
    return logits;
}

/// The prompt forward; with `hidden`, every block boundary of it is appended there.
fn forwardPromptCapture(allocator: std.mem.Allocator, l: *Loaded, ctx: *model_handle.ForwardCtx, ids: []const u32, hidden: ?*hidden_capture.Writer) !mlx.mlx_array {
    const w = hidden orelse return forwardPrompt(allocator, l, ctx, ids);
    if (try offline.capturePrompt(allocator, l, ctx, ids, w, promptChunk(&l.config))) |logits| return logits;
    const layers = l.config.num_hidden_layers;
    const layer_ids = try allocator.alloc(u32, layers);
    defer allocator.free(layer_ids);
    for (layer_ids, 0..) |*id, i| id.* = @intCast(i);
    const rows = try allocator.alloc(mlx.mlx_array, layers + 1);
    defer allocator.free(rows);
    for (rows) |*r| r.* = mlx.mlx_array_new();
    defer for (rows) |r| {
        _ = mlx.mlx_array_free(r);
    };
    var cl: model_handle.CaptureLayers = .{ .ids = layer_ids, .out = rows[1..], .input = &rows[0] };
    ctx.capture_layers = &cl;
    defer ctx.capture_layers = null;
    // Boundary rows are captured from one forward, so a hidden capture takes the prompt whole.
    const logits = try forwardPromptChunks(allocator, l, ctx, ids, ids.len);
    errdefer _ = mlx.mlx_array_free(logits);
    try w.append(l.xfm.s, ids, rows);
    return logits;
}

fn forwardOne(l: *Loaded, ctx: *model_handle.ForwardCtx, token: u32) !mlx.mlx_array {
    const data = [_]i32{@intCast(token)};
    const input = mlx.mlx_array_new_data(&data, &[_]c_int{ 1, 1 }, 2, .int32);
    defer _ = mlx.mlx_array_free(input);
    return l.xfm.forwardWith(ctx, input);
}

/// The RoPE the loaded model runs under, as a capture stamps it and a compare reports it.
pub fn ropeOf(config: *const model_mod.ModelConfig) fixture_mod.Rope {
    const flag = model_mod.getConfigOverrides() != null;
    return .{
        .yarn = config.rope_yarn,
        .factor = if (config.rope_yarn) config.yarn_factor else 1,
        .original = if (config.rope_yarn) config.yarn_orig_max_pos else config.max_position_embeddings,
        .window = config.max_position_embeddings,
        .source = if (config.yarn_factor_override != null) (if (flag) .both else .model_settings) else if (flag) .launch_flag else .none,
    };
}

fn ropeLine(out: *Out, who: []const u8, rope: ?fixture_mod.Rope) void {
    var buf: [160]u8 = undefined;
    out.print("[kld] RoPE {s}: {s}\n", .{ who, if (rope) |r| r.text(&buf) else "not recorded" });
}

fn writeRopeField(w: *std.Io.Writer, key: []const u8, rope: ?fixture_mod.Rope) !void {
    try w.print(",\n  \"{s}\": ", .{key});
    if (rope) |r| try r.writeJson(w) else try w.writeAll("null");
}

pub fn capturedSsdBudgetGb(config: *const model_mod.ModelConfig, flag_bytes: u64) u64 {
    if (!config.expert_streaming) return 0;
    if (config.expert_ssd_budget_bytes > 0) return config.expert_ssd_budget_bytes >> 30;
    return flag_bytes >> 30;
}

pub fn runCapture(io: std.Io, allocator: std.mem.Allocator, l: *Loaded, opts: Options, out: *Out) !void {
    fp8_block.reference_route = true;
    defer fp8_block.reference_route = false;
    var prompts = try fixture_mod.loadPrompts(allocator, io, opts.prompts, opts.limit);
    defer prompts.deinit();
    if (prompts.items.len == 0) return error.NoPromptsFound;
    try std.Io.Dir.cwd().createDirPath(io, opts.out_dir);
    ropeLine(out, "capture", ropeOf(&l.config));
    const hidden = if (opts.hidden_out.len > 0)
        try hidden_capture.Writer.open(allocator, io, opts.hidden_out, l.config.num_hidden_layers, offline.hiddenCaptureWidth(&l.config))
    else
        null;
    defer if (hidden) |w| w.close();

    var records: std.ArrayList(fixture_mod.PromptRecord) = .empty;
    defer {
        for (records.items) |r| r.deinit(allocator);
        records.deinit(allocator);
    }
    const started = std.Io.Timestamp.now(io, .awake);
    for (prompts.items, 0..) |p, index| {
        try l.xfm.resetCache();
        const rendered = try renderPrompt(allocator, l, opts, p);
        defer allocator.free(rendered);
        const ids = try promptIds(allocator, l, opts, p);
        defer allocator.free(ids);
        if (ids.len == 0) return error.EmptyPrompt;

        var writer = try fixture_mod.PromptWriter.begin(allocator, io, opts.out_dir, index, p.id);
        defer writer.deinit();
        var generated: std.ArrayList(u32) = .empty;
        defer generated.deinit(allocator);

        var ctx = l.xfm.defaultCtx();
        var logits = try forwardPromptCapture(allocator, l, &ctx, ids, hidden);
        defer _ = mlx.mlx_array_free(logits);
        const vocab = try logitsVocab(logits);
        const row = try allocator.alloc(f32, vocab);
        defer allocator.free(row);

        const steps: u32 = if (p.force) |f| @intCast(f.len) else opts.tokens;
        var step: u32 = 0;
        while (step < steps) : (step += 1) {
            try readLastRow(&l.xfm, logits, row);
            const chosen = fixture_mod.pickToken(p.force, step, row);
            try writer.appendRow(row, chosen);
            try generated.append(allocator, chosen);
            if (step + 1 == steps) break;
            const next = try forwardOne(l, &ctx, chosen);
            _ = mlx.mlx_array_free(logits);
            logits = next;
        }

        const record = try writer.finish(p.text, rendered, ids, generated.items);
        try records.append(allocator, record);
        out.print("[kld] {d}/{d} {s}: prompt={d} tok, generated={d}, nll={d:.9}, ppl={d:.6}\n", .{
            index + 1,
            prompts.items.len,
            record.id,
            record.prompt_tokens,
            record.generated_tokens,
            record.strict_nll_mean,
            record.strict_perplexity,
        });
    }
    const elapsed_ns: u64 = @intCast(started.untilNow(io, .awake).nanoseconds);
    const elapsed_secs = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;

    var nll_sum: f64 = 0;
    for (records.items) |r| nll_sum += r.strict_nll_mean;
    const mean_nll = nll_sum / @as(f64, @floatFromInt(records.items.len));

    try fixture_mod.writeBaseline(allocator, io, opts.out_dir, .{
        .rope = ropeOf(&l.config),
        .label = opts.label,
        .model = opts.model_dir,
        .run = opts.label,
        .kv_cache_format = kvCacheFormat(opts.kv_quant_config),
        .inference_profile = "greedy",
        .prompt_set = opts.prompts,
        .ssd_budget_gb = capturedSsdBudgetGb(&l.config, opts.ssd_budget_bytes),
        .tokens_per_prompt = opts.tokens,
        .top_k = opts.top_k,
        .elapsed_secs = elapsed_secs,
    }, records.items);
    const peak_bytes = gpu_mem.peakBytes();
    out.print("[kld] captured {d} prompts x {d} tokens into {s} in {d:.1}s (mean strict NLL {d:.9}, peak active {d:.2} GB)\n", .{
        records.items.len,
        opts.tokens,
        opts.out_dir,
        elapsed_secs,
        mean_nll,
        @as(f64, @floatFromInt(peak_bytes)) / 1e9,
    });
}

pub const PromptScore = struct {
    id: []const u8,
    kld: f64,
    top1: usize,
    positions: usize,
    nll: f64,
    cosine_similarity: f64,
    cosine_loss: f64,
    eos_pos: ?usize = null,
    kld_to_eos: f64 = 0,
    top1_to_eos: usize = 0,
    positions_to_eos: usize = 0,
    nll_to_eos: f64 = 0,
    cosine_similarity_to_eos: f64 = 0,
    cosine_loss_to_eos: f64 = 0,
    worst_cosine_loss: f64 = 0,
    worst_cosine_loss_to_eos: f64 = 0,
    per_position_kld: []const f64 = &.{},
};

// MLX allocator counters intentionally exclude mmap/page-cache-backed n-gram storage.
const MemoryReport = struct {
    model_active_bytes: usize,
    peak_active_bytes: usize,
    final_active_bytes: usize,
    final_cache_bytes: usize,
};

fn writeMemoryReportFields(w: *std.Io.Writer, report: MemoryReport) !void {
    try w.print("\"model_active_bytes\":{d},\"peak_active_bytes\":{d},\"final_active_bytes\":{d},\"final_cache_bytes\":{d}", .{
        report.model_active_bytes,
        report.peak_active_bytes,
        report.final_active_bytes,
        report.final_cache_bytes,
    });
}

pub fn runCompare(io: std.Io, allocator: std.mem.Allocator, l: *Loaded, opts: Options, out: *Out) !void {
    var memory: MemoryReport = undefined;
    memory.model_active_bytes = gpu_mem.activeBytes();
    gpu_mem.resetPeak();

    var base = try fixture_mod.readBaseline(allocator, io, opts.fixture);
    defer base.deinit();
    const count = if (opts.limit > 0 and opts.limit < base.prompts.len) opts.limit else base.prompts.len;
    if (count == 0) return error.NoPromptsFound;
    const student_rope = ropeOf(&l.config);
    ropeLine(out, "teacher", base.rope);
    ropeLine(out, "student", student_rope);

    var scores: std.ArrayList(PromptScore) = .empty;
    defer scores.deinit(allocator);
    defer for (scores.items) |sc| allocator.free(sc.per_position_kld);
    var total_kld: f64 = 0;
    var total_kld_eos: f64 = 0;
    var total_nll_eos: f64 = 0;
    var total_cosine_similarity: f64 = 0;
    var total_cosine_loss: f64 = 0;
    var total_cosine_similarity_eos: f64 = 0;
    var total_cosine_loss_eos: f64 = 0;
    var worst_cosine_loss: f64 = 0;
    var worst_cosine_loss_eos: f64 = 0;
    var total_top1_eos: usize = 0;
    var total_positions_eos: usize = 0;
    var eos_set: std.ArrayList(u32) = .empty;
    defer eos_set.deinit(allocator);
    for (l.config.eos_token_ids[0..l.config.num_eos_tokens]) |e| try eos_set.append(allocator, e);
    if (l.tok.encode(allocator, "<|im_end|>")) |im_end| {
        defer allocator.free(im_end);
        if (im_end.len == 1) try eos_set.append(allocator, im_end[0]);
    } else |_| {}
    var total_nll: f64 = 0;
    var total_top1: usize = 0;
    var total_positions: usize = 0;

    out.print("{s:<48} {s:>14} {s:>10} {s:>14}\n", .{ "prompt", "KLD", "top1", "NLL" });
    for (base.prompts[0..count]) |fp| {
        try l.xfm.resetCache();
        const tokens = try fixture_mod.readPromptTokens(allocator, io, opts.fixture, fp.dir);
        defer tokens.deinit(allocator);
        const generated = tokens.generated;

        const teacher = try fixture_mod.TeacherLogits.open(allocator, opts.fixture, fp.dir, generated.len);
        defer teacher.close();

        const teacher_row = try allocator.alloc(f32, teacher.vocab);
        defer allocator.free(teacher_row);
        const model_row = try allocator.alloc(f32, teacher.vocab);
        defer allocator.free(model_row);

        var ctx = l.xfm.defaultCtx();
        var logits = try forwardPrompt(allocator, l, &ctx, tokens.prompt);
        defer _ = mlx.mlx_array_free(logits);

        var prompt_kld: f64 = 0;
        var prompt_nll: f64 = 0;
        var prompt_cosine_similarity: f64 = 0;
        var prompt_cosine_loss: f64 = 0;
        var prompt_worst_cosine_loss: f64 = 0;
        var prompt_top1: usize = 0;
        const eos_pos = fixture_mod.firstEosPosition(generated, eos_set.items);
        const n_eos = fixture_mod.positionsToEos(generated, eos_set.items);
        var kld_eos: f64 = 0;
        var nll_eos: f64 = 0;
        var cosine_similarity_eos: f64 = 0;
        var cosine_loss_eos: f64 = 0;
        var worst_cosine_loss_eos_prompt: f64 = 0;
        var top1_eos: usize = 0;
        const per_pos = try allocator.alloc(f64, generated.len);
        for (generated, 0..) |token, position| {
            try teacher.readRow(position, teacher_row);
            if (position > 0) {
                const next = try forwardOne(l, &ctx, generated[position - 1]);
                _ = mlx.mlx_array_free(logits);
                logits = next;
            }
            try readLastRow(&l.xfm, logits, model_row);
            const score = try fixture_mod.scoreRow(teacher_row, model_row, token);
            prompt_kld += score.kld;
            prompt_nll += score.nll;
            prompt_cosine_similarity += score.cosine_similarity;
            prompt_cosine_loss += score.cosine_loss;
            if (score.cosine_loss > prompt_worst_cosine_loss) prompt_worst_cosine_loss = score.cosine_loss;
            if (score.top1) prompt_top1 += 1;
            per_pos[position] = score.kld;
            if (position < n_eos) {
                kld_eos += score.kld;
                nll_eos += score.nll;
                cosine_similarity_eos += score.cosine_similarity;
                cosine_loss_eos += score.cosine_loss;
                if (score.cosine_loss > worst_cosine_loss_eos_prompt) worst_cosine_loss_eos_prompt = score.cosine_loss;
                if (score.top1) top1_eos += 1;
            }
        }
        const positions: f64 = @floatFromInt(generated.len);
        const positions_eos: f64 = @floatFromInt(n_eos);
        try scores.append(allocator, .{
            .id = fp.id,
            .kld = prompt_kld / positions,
            .top1 = prompt_top1,
            .positions = generated.len,
            .nll = prompt_nll / positions,
            .cosine_similarity = prompt_cosine_similarity / positions,
            .cosine_loss = prompt_cosine_loss / positions,
            .eos_pos = eos_pos,
            .kld_to_eos = kld_eos / positions_eos,
            .top1_to_eos = top1_eos,
            .positions_to_eos = n_eos,
            .nll_to_eos = nll_eos / positions_eos,
            .cosine_similarity_to_eos = cosine_similarity_eos / positions_eos,
            .cosine_loss_to_eos = cosine_loss_eos / positions_eos,
            .worst_cosine_loss = prompt_worst_cosine_loss,
            .worst_cosine_loss_to_eos = worst_cosine_loss_eos_prompt,
            .per_position_kld = per_pos,
        });
        total_kld += prompt_kld;
        total_nll += prompt_nll;
        total_cosine_similarity += prompt_cosine_similarity;
        total_cosine_loss += prompt_cosine_loss;
        if (prompt_worst_cosine_loss > worst_cosine_loss) worst_cosine_loss = prompt_worst_cosine_loss;
        total_top1 += prompt_top1;
        total_positions += generated.len;
        total_kld_eos += kld_eos;
        total_nll_eos += nll_eos;
        total_cosine_similarity_eos += cosine_similarity_eos;
        total_cosine_loss_eos += cosine_loss_eos;
        if (worst_cosine_loss_eos_prompt > worst_cosine_loss_eos) worst_cosine_loss_eos = worst_cosine_loss_eos_prompt;
        total_top1_eos += top1_eos;
        total_positions_eos += n_eos;
        out.print("{s:<48} {d:>14.9} {d:>6}/{d:<3} {d:>14.9}   to-eos {d:>12.9} {d:>3}/{d:<3} {d:>12.9}\n", .{
            fp.id,
            prompt_kld / positions,
            prompt_top1,
            generated.len,
            prompt_nll / positions,
            kld_eos / positions_eos,
            top1_eos,
            n_eos,
            nll_eos / positions_eos,
        });
    }

    const positions_f: f64 = @floatFromInt(total_positions);
    const mean_kld = total_kld / positions_f;
    const mean_nll = total_nll / positions_f;
    const mean_top1 = @as(f64, @floatFromInt(total_top1)) / positions_f;
    const positions_eos_f: f64 = @floatFromInt(@max(total_positions_eos, 1));
    const mean_kld_eos = total_kld_eos / positions_eos_f;
    const mean_nll_eos = total_nll_eos / positions_eos_f;
    const mean_cosine_similarity = total_cosine_similarity / positions_f;
    const mean_cosine_loss = total_cosine_loss / positions_f;
    const mean_cosine_similarity_eos = total_cosine_similarity_eos / positions_eos_f;
    const mean_cosine_loss_eos = total_cosine_loss_eos / positions_eos_f;
    const mean_top1_eos = @as(f64, @floatFromInt(total_top1_eos)) / positions_eos_f;
    memory.peak_active_bytes = gpu_mem.peakBytes();
    memory.final_active_bytes = gpu_mem.activeBytes();
    memory.final_cache_bytes = gpu_mem.cacheBytes();

    out.print("{s:<48} {d:>14.9} {d:>6}/{d:<3} {d:>14.9}   to-eos {d:>12.9} {d:>3}/{d:<3} {d:>12.9}\n", .{ "mean", mean_kld, total_top1, total_positions, mean_nll, mean_kld_eos, total_top1_eos, total_positions_eos, mean_nll_eos });
    out.print("[kld] {s}: to-first-EOS mean KLD={d:.9} top1={d:.6} NLL={d:.9} cosine={d:.9} loss={d:.9} worst_loss={d:.9} over {d} positions\n", .{
        opts.label,
        mean_kld_eos,
        mean_top1_eos,
        mean_nll_eos,
        mean_cosine_similarity_eos,
        mean_cosine_loss_eos,
        worst_cosine_loss_eos,
        total_positions_eos,
    });
    out.print("[kld] {s}: mean KLD={d:.9} top1={d:.6} NLL={d:.9} cosine={d:.9} loss={d:.9} worst_loss={d:.9} over {d} prompts / {d} positions\n", .{
        opts.label,
        mean_kld,
        mean_top1,
        mean_nll,
        mean_cosine_similarity,
        mean_cosine_loss,
        worst_cosine_loss,
        scores.items.len,
        total_positions,
    });

    if (opts.json_out.len > 0) {
        var f = try std.Io.Dir.cwd().createFile(io, opts.json_out, .{});
        defer f.close(io);
        var buf: [64 * 1024]u8 = undefined;
        var fw = f.writer(io, &buf);
        const w = &fw.interface;
        try w.writeAll("{\n  \"schema\": ");
        try fixture_mod.writeJsonString(w, fixture_mod.COMPARE_SCHEMA);
        try w.writeAll(",\n  \"tool\": ");
        try fixture_mod.writeJsonString(w, fixture_mod.TOOL);
        try w.writeAll(",\n  \"label\": ");
        try fixture_mod.writeJsonString(w, opts.label);
        try w.writeAll(",\n  \"model\": ");
        try fixture_mod.writeJsonString(w, opts.model_dir);
        try w.writeAll(",\n  \"fixture\": ");
        try fixture_mod.writeJsonString(w, opts.fixture);
        try w.writeAll(",\n  \"kv_cache_format\": ");
        try fixture_mod.writeJsonString(w, kvCacheFormat(opts.kv_quant_config));
        try w.print(",\n  \"expert_pick_tolerance\": {d}", .{opts.pick_tolerance});
        try writeRopeField(w, "teacher_rope", base.rope);
        try writeRopeField(w, "student_rope", student_rope);
        try w.writeAll(",\n  ");
        try writeMemoryReportFields(w, memory);
        try w.print(",\n  \"mean_kld\": {d},\n  \"mean_top1\": {d},\n  \"mean_nll\": {d},\n  \"mean_cosine_similarity\": {d},\n  \"mean_cosine_loss\": {d},\n  \"worst_cosine_loss\": {d},\n  \"positions\": {d}", .{
            mean_kld,
            mean_top1,
            mean_nll,
            mean_cosine_similarity,
            mean_cosine_loss,
            worst_cosine_loss,
            total_positions,
        });
        try w.print(",\n  \"mean_kld_to_eos\": {d},\n  \"mean_top1_to_eos\": {d},\n  \"mean_nll_to_eos\": {d},\n  \"mean_cosine_similarity_to_eos\": {d},\n  \"mean_cosine_loss_to_eos\": {d},\n  \"worst_cosine_loss_to_eos\": {d},\n  \"positions_to_eos\": {d}", .{
            mean_kld_eos,
            mean_top1_eos,
            mean_nll_eos,
            mean_cosine_similarity_eos,
            mean_cosine_loss_eos,
            worst_cosine_loss_eos,
            total_positions_eos,
        });
        try w.writeAll(",\n  \"prompts\": [\n");
        for (scores.items, 0..) |s, i| {
            try w.writeAll("    {\"id\": ");
            try fixture_mod.writeJsonString(w, s.id);
            try w.print(", \"kld\": {d}, \"top1\": {d}, \"positions\": {d}, \"nll\": {d}, \"cosine_similarity\": {d}, \"cosine_loss\": {d}, \"worst_cosine_loss\": {d}", .{
                s.kld,
                s.top1,
                s.positions,
                s.nll,
                s.cosine_similarity,
                s.cosine_loss,
                s.worst_cosine_loss,
            });
            if (s.eos_pos) |e| try w.print(", \"eos_pos\": {d}", .{e}) else try w.writeAll(", \"eos_pos\": null");
            try w.print(", \"kld_to_eos\": {d}, \"top1_to_eos\": {d}, \"positions_to_eos\": {d}, \"nll_to_eos\": {d}, \"cosine_similarity_to_eos\": {d}, \"cosine_loss_to_eos\": {d}, \"worst_cosine_loss_to_eos\": {d}, \"per_position_kld\": [", .{
                s.kld_to_eos,
                s.top1_to_eos,
                s.positions_to_eos,
                s.nll_to_eos,
                s.cosine_similarity_to_eos,
                s.cosine_loss_to_eos,
                s.worst_cosine_loss_to_eos,
            });
            for (s.per_position_kld, 0..) |k, j| {
                if (j > 0) try w.writeAll(",");
                try w.print("{d}", .{k});
            }
            try w.writeAll("]}");
            if (i + 1 < scores.items.len) try w.writeAll(",");
            try w.writeAll("\n");
        }
        try w.writeAll("  ]\n}\n");
        try w.flush();
    }

    for (scores.items) |s| {
        if (!std.math.isFinite(s.kld) or !std.math.isFinite(s.nll) or
            !std.math.isFinite(s.cosine_similarity) or !std.math.isFinite(s.cosine_loss) or
            !std.math.isFinite(s.cosine_similarity_to_eos) or !std.math.isFinite(s.cosine_loss_to_eos) or
            !std.math.isFinite(s.worst_cosine_loss) or !std.math.isFinite(s.worst_cosine_loss_to_eos))
        {
            return error.NonFiniteKld;
        }
    }
}

/// What a bad --fixture (compare) or --prompts (capture) says, found before the model loads; null when both read.
fn inputRefusal(allocator: std.mem.Allocator, io: std.Io, opts: Options, buf: []u8) ?[]const u8 {
    switch (opts.command) {
        .compare => {
            var base = fixture_mod.readBaseline(allocator, io, opts.fixture) catch |err|
                return std.fmt.bufPrint(buf, "--fixture {s}: {s}", .{ opts.fixture, cli_error.describe(err) orelse @errorName(err) }) catch "--fixture: unreadable";
            base.deinit();
        },
        .capture => {
            var prompts = fixture_mod.loadPrompts(allocator, io, opts.prompts, opts.limit) catch |err|
                return std.fmt.bufPrint(buf, "--prompts {s}: {s}", .{ opts.prompts, cli_error.describe(err) orelse @errorName(err) }) catch "--prompts: unreadable";
            prompts.deinit();
        },
    }
    return null;
}

pub fn cmdKld(allocator: std.mem.Allocator, io: std.Io, args: []const []const u8) !void {
    var out: Out = .{};
    var opts = parseArgs(args) catch |err| {
        out.print("sushi kld: {s}\n\n{s}", .{ cli_error.describe(err) orelse @errorName(err), USAGE });
        std.process.exit(1);
    };
    if (opts.help) {
        out.print("{s}", .{USAGE});
        return;
    }
    if (opts.command == .capture) if (hidden_capture.envPath()) |dir| {
        opts.hidden_out = dir;
        log.info("[kld] {s}: appending every prompt forward's block boundaries to {s}\n", .{ hidden_capture.ENV_VAR, dir });
    };
    if (opts.command == .capture) if (@import("imatrix.zig").envPath()) |path| {
        opts.imatrix_out = path;
    };
    var refusal_buf: [1024]u8 = undefined;
    if (inputRefusal(allocator, io, opts, &refusal_buf)) |why| {
        log.err("sushi kld: {s}\n", .{why});
        std.process.exit(1);
    }
    if (try offline.tryRun(allocator, io, opts, &out)) return;
    if (opts.layer_major) return error.LayerMajorNeedsNativeGlmTeacher;
    expert_stream_mod.pick_tolerance = opts.pick_tolerance;
    if (opts.wired_margin_bytes > 0) server_state.wired_limit_margin_bytes = opts.wired_margin_bytes;
    const loaded = try loadModel(io, allocator, opts);
    defer loaded.deinit();
    switch (opts.command) {
        .capture => try runCapture(io, allocator, loaded, opts, &out),
        .compare => try runCompare(io, allocator, loaded, opts, &out),
    }
}

test "kld: --help lists every flag the parser accepts" {
    for (value_flags) |row| try testing.expect(std.mem.indexOf(u8, USAGE, row[0]) != null);
    for ([_][]const u8{ "--no-template", "--mtp", "--no-mtp", "--layer-major" }) |name|
        try testing.expect(std.mem.indexOf(u8, USAGE, name) != null);
}

test "kld: compare memory report serializes aggregate MLX byte counters" {
    const report = MemoryReport{
        .model_active_bytes = 11_000,
        .peak_active_bytes = 22_000,
        .final_active_bytes = 13_000,
        .final_cache_bytes = 4_000,
    };
    var buf: [256]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try writeMemoryReportFields(&w, report);
    try testing.expectEqualStrings(
        "\"model_active_bytes\":11000,\"peak_active_bytes\":22000,\"final_active_bytes\":13000,\"final_cache_bytes\":4000",
        w.buffered(),
    );
}

test "kld: the argument parser reads every flag and refuses an unknown one" {
    const capture = try parseArgs(&.{
        "capture",
        "--model",
        "/models/pack",
        "--prompts",
        "/fixtures/teacher",
        "--out",
        "/out/run",
        "--tokens",
        "32",
        "--top-k",
        "5",
        "--label",
        "run-a",
        "--limit",
        "3",
        "--no-template",
        "--ctx-size",
        "8192",
        "--kv-quant",
        "8",
        "--ssd-budget-gb",
        "94",
        "--no-mtp",
    });
    try testing.expectEqual(Command.capture, capture.command);
    try testing.expectEqualStrings("/models/pack", capture.model_dir);
    try testing.expectEqualStrings("/fixtures/teacher", capture.prompts);
    try testing.expectEqualStrings("/out/run", capture.out_dir);
    try testing.expectEqual(@as(u32, 32), capture.tokens);
    try testing.expectEqual(@as(u32, 5), capture.top_k);
    try testing.expectEqualStrings("run-a", capture.label);
    try testing.expectEqual(@as(u32, 3), capture.limit);
    try testing.expect(capture.no_template);
    try testing.expectEqual(@as(u32, 8192), capture.ctx_size);
    try testing.expectEqual(@as(u8, 8), capture.kv_quant_config.bits);
    try testing.expectEqual(@as(u64, 94) << 30, capture.ssd_budget_bytes);
    try testing.expect(capture.mtp == false);

    const compare = try parseArgs(&.{ "compare", "--model", "/models/pack", "--fixture", "/fixtures/teacher", "--json", "/tmp/out.json" });
    try testing.expectEqual(Command.compare, compare.command);
    try testing.expectEqualStrings("/fixtures/teacher", compare.fixture);
    try testing.expectEqualStrings("/tmp/out.json", compare.json_out);
    try testing.expect(compare.mtp == null);
    try testing.expectEqual(@as(u32, 64), compare.tokens);
    const with_mtp = try parseArgs(&.{ "compare", "--model", "/models/pack", "--fixture", "/fixtures/teacher", "--mtp" });
    try testing.expect(with_mtp.mtp == true);
    try testing.expectError(error.ConflictingFlags, parseArgs(&.{ "compare", "--model", "/m", "--fixture", "/f", "--mtp", "--no-mtp" }));
    try testing.expectError(error.ConflictingFlags, parseArgs(&.{ "compare", "--model", "/m", "--fixture", "/f", "--no-mtp", "--mtp" }));

    const routed = try parseArgs(&.{ "compare", "--model", "/m", "--fixture", "/f", "--expert-pick-tolerance", "0.3", "--wired-margin-gib", "5" });
    try testing.expectEqual(@as(f32, 0.3), routed.pick_tolerance);
    try testing.expectEqual(@as(u64, 5) << 30, routed.wired_margin_bytes);
    try testing.expectError(error.BadFlagValue, parseArgs(&.{ "compare", "--model", "/m", "--fixture", "/f", "--expert-pick-tolerance", "0.7" }));
    try testing.expectError(error.BadFlagValue, parseArgs(&.{ "compare", "--model", "/m", "--fixture", "/f", "--wired-margin-gib", "0" }));

    try testing.expectError(error.TeacherMustBeLossless, parseArgs(&.{ "capture", "--model", "/m", "--prompts", "/p", "--out", "/o", "--expert-pick-tolerance", "0.3" }));
    const exact_capture = try parseArgs(&.{ "capture", "--model", "/m", "--prompts", "/p", "--out", "/o", "--expert-pick-tolerance", "0" });
    try testing.expectEqual(@as(f32, 0), exact_capture.pick_tolerance);

    try testing.expectError(error.UnknownFlag, parseArgs(&.{ "compare", "--model", "/m", "--fixture", "/f", "--nope" }));
    try testing.expectError(error.UnknownFlag, parseArgs(&.{ "capture", "--model=/m" }));
    try testing.expectError(error.UnknownSubcommand, parseArgs(&.{"replay"}));
    try testing.expectError(error.MissingSubcommand, parseArgs(&.{}));
    try testing.expectError(error.MissingFlagValue, parseArgs(&.{ "compare", "--model" }));
    try testing.expectError(error.BadFlagValue, parseArgs(&.{ "capture", "--model", "/m", "--prompts", "/p", "--out", "/o", "--tokens", "zero" }));
    try testing.expectError(error.BadFlagValue, parseArgs(&.{ "capture", "--model", "/m", "--prompts", "/p", "--out", "/o", "--kv-quant", "3" }));
    try testing.expectError(error.BadFlagValue, parseArgs(&.{ "capture", "--model", "/m", "--prompts", "/p", "--out", "/o", "--qwen-gdn", "fp32" }));
    try testing.expectError(error.MissingModel, parseArgs(&.{ "capture", "--prompts", "/p", "--out", "/o" }));
    try testing.expectError(error.MissingPrompts, parseArgs(&.{ "capture", "--model", "/m", "--out", "/o" }));
    try testing.expectError(error.MissingOut, parseArgs(&.{ "capture", "--model", "/m", "--prompts", "/p" }));
    try testing.expectError(error.MissingFixture, parseArgs(&.{ "compare", "--model", "/m" }));
    try testing.expect((try parseArgs(&.{ "capture", "--help" })).help);
}

test "kld: an unflagged capture keeps a full-width teacher KV, never the serving kv8 default" {
    const capture = try parseArgs(&.{ "capture", "--model", "/m", "--prompts", "/p", "--out", "/o" });
    try testing.expectEqual(trunk.KVQuantConfig.dense, capture.kv_quant_config);
    try testing.expect(trunk.KVQuantConfig.engine_default.isQuant());
    try testing.expectEqualStrings("bf16", kvCacheFormat(capture.kv_quant_config));
    try testing.expect(capture.qwen_gdn_f32);
}

test "kld: --qwen-gdn 16 stores the GDN state in bf16, 32 is the default width" {
    const narrow = try parseArgs(&.{ "capture", "--model", "/m", "--prompts", "/p", "--out", "/o", "--qwen-gdn", "16" });
    try testing.expect(!narrow.qwen_gdn_f32);
    const wide = try parseArgs(&.{ "compare", "--model", "/m", "--fixture", "/f", "--qwen-gdn", "32" });
    try testing.expect(wide.qwen_gdn_f32);
}

test "kld: --ssd-prefetch is an integer 0..100" {
    try testing.expectEqual(@as(?u8, null), (try parseArgs(&.{ "compare", "--model", "/m", "--fixture", "/f" })).ssd_prefetch);
    for ([_][]const u8{ "0", "65", "100" }) |ok| {
        const o = try parseArgs(&.{ "compare", "--model", "/m", "--fixture", "/f", "--ssd-prefetch", ok });
        try testing.expectEqual(@as(?u8, try std.fmt.parseInt(u8, ok, 10)), o.ssd_prefetch);
    }
    for ([_][]const u8{ "101", "-1", "abc", "" }) |bad|
        try testing.expectError(error.BadFlagValue, parseArgs(&.{ "compare", "--model", "/m", "--fixture", "/f", "--ssd-prefetch", bad }));
}

test "kld: --drop-expert is a student-side multiple of 8, and a capture refuses it because the teacher is exact" {
    const cmp = try parseArgs(&.{ "compare", "--model", "/m", "--fixture", "/f", "--drop-expert", "16" });
    try testing.expectEqual(@as(u16, 16), cmp.drop_expert);
    try testing.expectEqual(@as(u16, 0), (try parseArgs(&.{ "compare", "--model", "/m", "--fixture", "/f" })).drop_expert);
    try testing.expectError(error.BadFlagValue, parseArgs(&.{ "compare", "--model", "/m", "--fixture", "/f", "--drop-expert", "12" }));
    try testing.expectError(error.BadFlagValue, parseArgs(&.{ "compare", "--model", "/m", "--fixture", "/f", "--drop-expert", "many" }));
    try testing.expectError(error.TeacherMustBeLossless, parseArgs(&.{ "capture", "--model", "/m", "--prompts", "/p", "--out", "/o", "--drop-expert", "8" }));
}

test "kld: kvCacheFormat names the effective KV width" {
    try testing.expectEqualStrings("bf16", kvCacheFormat(trunk.KVQuantConfig.dense));
    try testing.expectEqualStrings("affine4", kvCacheFormat(trunk.KVQuantConfig.affine(4)));
    try testing.expectEqualStrings("affine6", kvCacheFormat(trunk.KVQuantConfig.affine(6)));
    try testing.expectEqualStrings("affine8", kvCacheFormat(trunk.KVQuantConfig.affine(8)));
}

test "kld records the budget that shaped the load, never the raw flag" {
    const GiB: u64 = 1 << 30;
    var moe = model_mod.ModelConfig{ .model_type = "not_a_served_arch" };
    try testing.expectEqual(@as(u64, 0), capturedSsdBudgetGb(&moe, 60 * GiB));
    moe.expert_ssd_budget_bytes = 60 * GiB;
    try testing.expectEqual(@as(u64, 0), capturedSsdBudgetGb(&moe, 60 * GiB));

    var q4 = model_mod.ModelConfig{ .model_type = "not_a_served_arch", .expert_streaming = true };
    q4.expert_ssd_budget_bytes = 60 * GiB;
    try testing.expectEqual(@as(u64, 60), capturedSsdBudgetGb(&q4, 0));
    q4.expert_ssd_budget_bytes = 0;
    try testing.expectEqual(@as(u64, 48), capturedSsdBudgetGb(&q4, 48 * GiB));
}

test {
    _ = offline;
}

test "kld: a missing fixture or prompt source is found before any model loads" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var buf: [1024]u8 = undefined;
    const compare = try parseArgs(&.{ "compare", "--model", "/m", "--fixture", "/no/such/fixture" });
    const why = inputRefusal(testing.allocator, io, compare, &buf).?;
    try testing.expect(std.mem.indexOf(u8, why, "--fixture /no/such/fixture") != null);
    try testing.expect(std.mem.indexOf(u8, why, "baseline.json") != null);
    const capture = try parseArgs(&.{ "capture", "--model", "/m", "--prompts", "/no/such/prompts", "--out", "/o" });
    const why_prompts = inputRefusal(testing.allocator, io, capture, &buf).?;
    try testing.expect(std.mem.indexOf(u8, why_prompts, "--prompts /no/such/prompts") != null);
    const bundled = try parseArgs(&.{ "capture", "--model", "/m", "--prompts", "standard4", "--out", "/o" });
    try testing.expect(inputRefusal(testing.allocator, io, bundled, &buf) == null);
}

test "kld: ropeOf names the RoPE a model runs under and what overrode it" {
    defer model_mod.setConfigOverrides(null);
    var yarn = model_mod.ModelConfig{ .max_position_embeddings = 524_288, .rope_yarn = true, .yarn_factor = 2, .yarn_orig_max_pos = 262_144, .yarn_factor_override = 2 };
    const native = model_mod.ModelConfig{ .max_position_embeddings = 262_144 };

    const stamped = ropeOf(&yarn);
    try testing.expect(stamped.yarn);
    try testing.expectEqual(@as(f32, 2), stamped.factor);
    try testing.expectEqual(@as(u32, 262_144), stamped.original);
    try testing.expectEqual(@as(u32, 524_288), stamped.window);
    try testing.expectEqual(fixture_mod.Rope.Source.model_settings, stamped.source);

    const plain = ropeOf(&native);
    try testing.expect(!plain.yarn);
    try testing.expectEqual(@as(f32, 1), plain.factor);
    try testing.expectEqual(@as(u32, 262_144), plain.original);
    try testing.expectEqual(fixture_mod.Rope.Source.none, plain.source);

    model_mod.setConfigOverrides("{}");
    try testing.expectEqual(fixture_mod.Rope.Source.both, ropeOf(&yarn).source);
    try testing.expectEqual(fixture_mod.Rope.Source.launch_flag, ropeOf(&native).source);
}
