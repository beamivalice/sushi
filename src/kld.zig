const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const chat_mod = @import("chat.zig");
const tokenizer_mod = @import("tokenizer.zig");
const transformer_mod = @import("transformer.zig");
const expert_stream_mod = @import("expert_stream.zig");
const scheduler_mod = @import("scheduler.zig");
const model_settings_mod = @import("model_settings.zig");
const server_mod = @import("server.zig");
const hidden_capture = @import("hidden_capture.zig");
const testing = std.testing;

pub const SCHEMA = "mlx-serve-kld-baseline-v1";
pub const COMPARE_SCHEMA = "mlx-serve-kld-compare-v1";
pub const TOOL = "sushi";

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
    kv_quant_config: transformer_mod.KVQuantConfig = transformer_mod.KVQuantConfig.dense,
    expert_cache_bytes: u64 = 0,
    ssd_budget_bytes: u64 = 0,
    pick_tolerance: f32 = 0,
    wired_margin_bytes: u64 = 0,
    enable_mtp: bool = false,
    mtp_explicit: bool = false,
    /// `SUSHI_HIDDEN_OUT`: capture appends each prompt forward's block boundaries here.
    hidden_out: []const u8 = "",
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
    TeacherMustBeLossless,
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
    ssd_budget_gb,
    expert_cache_gb,
    expert_pick_tolerance,
    wired_margin_gib,
};

fn valueFlag(name: []const u8) ?ValueFlag {
    const table = [_]struct { []const u8, ValueFlag }{
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
        .{ "--ssd-budget-gb", .ssd_budget_gb },
        .{ "--expert-cache-gb", .expert_cache_gb },
        .{ "--expert-pick-tolerance", .expert_pick_tolerance },
        .{ "--wired-margin-gib", .wired_margin_gib },
    };
    for (table) |row| if (std.mem.eql(u8, name, row[0])) return row[1];
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
            o.enable_mtp = false;
            o.mtp_explicit = true;
        } else if (std.mem.eql(u8, a, "--mtp")) {
            o.enable_mtp = true;
            o.mtp_explicit = true;
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
                .kv_quant => o.kv_quant_config = transformer_mod.KVQuantConfig.fromJsonValue(.{ .string = v }) orelse return error.BadFlagValue,
                .ssd_budget_gb => o.ssd_budget_bytes = server_mod.parseSsdBudgetGb(v) catch return error.BadFlagValue,
                .expert_cache_gb => o.expert_cache_bytes = server_mod.parseExpertCacheGb(v) catch return error.BadFlagValue,
                .expert_pick_tolerance => o.pick_tolerance = expert_stream_mod.parsePickTolerance(v) catch return error.BadFlagValue,
                .wired_margin_gib => o.wired_margin_bytes = server_mod.parseWiredMarginGib(v) catch return error.BadFlagValue,
            }
        } else {
            return error.UnknownFlag;
        }
    }
    if (o.tokens == 0) return error.BadFlagValue;
    if (o.model_dir.len == 0) return error.MissingModel;
    switch (o.command) {
        .capture => {
            if (o.prompts.len == 0) return error.MissingPrompts;
            if (o.out_dir.len == 0) return error.MissingOut;
            if (o.pick_tolerance > 0) return error.TeacherMustBeLossless;
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
    \\  (token ids, used as given) instead of "prompt".
    \\
    \\  SUSHI_HIDDEN_OUT=<dir> makes capture append each prompt forward's
    \\  residual at every block boundary to <dir> (boundary-XX.bin, tokens.bin).
    \\
    \\options:
    \\  --tokens <n>          greedy tokens per prompt to capture (default 64)
    \\  --top-k <n>           top_k recorded in baseline.json (default 10)
    \\  --label <s>           label/run recorded in the output (default sushi)
    \\  --limit <n>           only the first n prompts (0 = all)
    \\  --no-template         feed the raw prompt text, no chat template
    \\  --json <file>         compare: write the numbers as JSON
    \\  --ctx-size <n>        context length override
    \\  --kv-quant <16|8|4>   KV cache quantization (16 = BF16, also off)
    \\  --ssd-budget-gb <n>   expert streaming budget (GiB)
    \\  --expert-cache-gb <n> streamed expert cache size (GB), outranks --ssd-budget-gb
    \\  --mtp                 keep the MTP head resident (refused under streaming)
    \\  --expert-pick-tolerance <n>  compare only, LOSSY: swap a missed streamed expert for a cached one within n (0..0.6)
    \\  --wired-margin-gib <n>  headroom under iogpu.wired_limit_mb (integers 2..32)
    \\
;

pub const RowScore = struct {
    kld: f64,
    nll: f64,
    top1: bool,
    cosine_similarity: f64,
    cosine_loss: f64,
};

pub fn scoreRow(teacher: []const f32, model: []const f32, token: u32) !RowScore {
    if (model.len != teacher.len or token >= teacher.len) return error.KldShapeMismatch;
    var teacher_max = -std.math.inf(f64);
    var candidate_max = -std.math.inf(f64);
    var candidate_top: usize = 0;
    var dot: f64 = 0;
    var teacher_norm_sq: f64 = 0;
    var candidate_norm_sq: f64 = 0;
    for (teacher, 0..) |value, i| {
        const teacher_value = @as(f64, value);
        const candidate_value = @as(f64, model[i]);
        if (!std.math.isFinite(teacher_value) or !std.math.isFinite(candidate_value)) return error.NonFinite;
        if (value > teacher_max) teacher_max = value;
        if (model[i] > candidate_max) {
            candidate_max = model[i];
            candidate_top = i;
        }
        dot += teacher_value * candidate_value;
        teacher_norm_sq += teacher_value * teacher_value;
        candidate_norm_sq += candidate_value * candidate_value;
    }
    if (!std.math.isFinite(dot) or !std.math.isFinite(teacher_norm_sq) or !std.math.isFinite(candidate_norm_sq)) return error.NonFinite;
    if (teacher_norm_sq == 0 or candidate_norm_sq == 0) return error.ZeroNorm;
    const cosine_similarity = dot / (@sqrt(teacher_norm_sq) * @sqrt(candidate_norm_sq));
    if (!std.math.isFinite(cosine_similarity)) return error.NonFinite;
    var teacher_sum: f64 = 0;
    var candidate_sum: f64 = 0;
    for (teacher, 0..) |value, i| {
        teacher_sum += @exp(@as(f64, value) - teacher_max);
        candidate_sum += @exp(@as(f64, model[i]) - candidate_max);
    }
    const teacher_log_z = teacher_max + @log(teacher_sum);
    const candidate_log_z = candidate_max + @log(candidate_sum);
    var kld: f64 = 0;
    for (teacher, 0..) |value, i| {
        const log_p = @as(f64, value) - teacher_log_z;
        const log_q = @as(f64, model[i]) - candidate_log_z;
        kld += @exp(log_p) * (log_p - log_q);
    }
    return .{
        .kld = kld,
        .nll = candidate_log_z - model[token],
        .top1 = candidate_top == token,
        .cosine_similarity = cosine_similarity,
        .cosine_loss = 1.0 - cosine_similarity,
    };
}

fn rowNll(row: []const f32, chosen: u32) f64 {
    var max = -std.math.inf(f64);
    for (row) |v| {
        if (v > max) max = v;
    }
    var sum: f64 = 0;
    for (row) |v| sum += @exp(@as(f64, v) - max);
    return max + @log(sum) - @as(f64, row[chosen]);
}

fn argmaxOf(row: []const f32) u32 {
    var best: usize = 0;
    var best_v = -std.math.inf(f64);
    for (row, 0..) |v, i| {
        if (v > best_v) {
            best_v = v;
            best = i;
        }
    }
    return @intCast(best);
}

/// `ids` set: the prompt is these token ids as given (no template, no tokenizer).
pub const Prompt = struct { id: []u8, text: []u8, ids: ?[]u32 = null };

pub const PromptList = struct {
    allocator: std.mem.Allocator,
    items: []Prompt = &.{},

    pub fn deinit(self: *PromptList) void {
        for (self.items) |p| {
            self.allocator.free(p.id);
            self.allocator.free(p.text);
            if (p.ids) |ids| self.allocator.free(ids);
        }
        self.allocator.free(self.items);
        self.items = &.{};
    }
};

pub const SourceKind = enum { fixture, text_dir, jsonl };

pub fn classifySource(io: std.Io, path: []const u8) !SourceKind {
    if (path.len == 0) return error.PromptSourceUnreadable;
    if (std.mem.endsWith(u8, path, ".jsonl")) return .jsonl;
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const baseline = std.fmt.bufPrint(&buf, "{s}/baseline.json", .{path}) catch return error.KldPathTooLong;
    if (std.Io.Dir.cwd().statFile(io, baseline, .{})) |_| {
        return .fixture;
    } else |_| {}
    const st = std.Io.Dir.cwd().statFile(io, path, .{}) catch return error.PromptSourceUnreadable;
    if (st.kind != .directory) return error.PromptSourceUnreadable;
    return .text_dir;
}

const MAX_PROMPT_BYTES = 8 * 1024 * 1024;
/// A jsonl of token-id windows runs to several MiB per thousand windows.
const MAX_JSONL_BYTES = 256 * 1024 * 1024;

pub fn loadPrompts(allocator: std.mem.Allocator, io: std.Io, path: []const u8, limit: u32) !PromptList {
    var list = PromptList{ .allocator = allocator };
    var items: std.ArrayList(Prompt) = .empty;
    errdefer {
        for (items.items) |p| {
            allocator.free(p.id);
            allocator.free(p.text);
            if (p.ids) |ids| allocator.free(ids);
        }
        items.deinit(allocator);
    }
    const bundled = std.mem.eql(u8, path, "standard4");
    switch (if (bundled) SourceKind.jsonl else try classifySource(io, path)) {
        .fixture => {
            var base = try readBaseline(allocator, io, path);
            defer base.deinit();
            for (base.prompts) |fp| {
                if (limit > 0 and items.items.len >= limit) break;
                const text_path = try std.fmt.allocPrint(allocator, "{s}/{s}/prompt.txt", .{ path, fp.dir });
                defer allocator.free(text_path);
                const text = try std.Io.Dir.cwd().readFileAlloc(io, text_path, allocator, .limited(MAX_PROMPT_BYTES));
                errdefer allocator.free(text);
                try items.append(allocator, .{ .id = try allocator.dupe(u8, fp.id), .text = text });
            }
        },
        .text_dir => {
            var names: std.ArrayList([]u8) = .empty;
            defer {
                for (names.items) |n| allocator.free(n);
                names.deinit(allocator);
            }
            {
                var dir = std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true }) catch return error.PromptSourceUnreadable;
                defer dir.close(io);
                var it = dir.iterate();
                while (try it.next(io)) |dent| {
                    if (dent.kind == .directory) continue;
                    if (!std.mem.endsWith(u8, dent.name, ".txt")) continue;
                    try names.append(allocator, try allocator.dupe(u8, dent.name));
                }
            }
            std.mem.sort([]u8, names.items, {}, struct {
                fn lt(_: void, a: []u8, b: []u8) bool {
                    return std.mem.lessThan(u8, a, b);
                }
            }.lt);
            for (names.items) |name| {
                if (limit > 0 and items.items.len >= limit) break;
                const text_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ path, name });
                defer allocator.free(text_path);
                const text = try std.Io.Dir.cwd().readFileAlloc(io, text_path, allocator, .limited(MAX_PROMPT_BYTES));
                errdefer allocator.free(text);
                const id = try allocator.dupe(u8, name[0 .. name.len - 4]);
                errdefer allocator.free(id);
                try items.append(allocator, .{ .id = id, .text = text });
            }
        },
        .jsonl => {
            const body: []const u8 = if (bundled) @embedFile("fixtures/kld-standard4.jsonl") else std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(MAX_JSONL_BYTES)) catch return error.PromptSourceUnreadable;
            defer if (!bundled) allocator.free(body);
            var lines = std.mem.splitScalar(u8, body, '\n');
            var seq: usize = 0;
            while (lines.next()) |raw_line| {
                const line = std.mem.trim(u8, raw_line, " \t\r\n");
                if (line.len == 0) continue;
                if (limit > 0 and items.items.len >= limit) break;
                var parsed = std.json.parseFromSlice(std.json.Value, allocator, line, .{}) catch return error.BadPromptJsonl;
                defer parsed.deinit();
                const obj = switch (parsed.value) {
                    .object => |o| o,
                    else => return error.BadPromptJsonl,
                };
                const ids = if (obj.get("prompt_ids")) |v| try tokenIdsField(allocator, v) else null;
                errdefer if (ids) |i| allocator.free(i);
                const text = if (ids != null) try allocator.dupe(u8, "") else blk: {
                    const text_value = obj.get("prompt") orelse return error.BadPromptJsonl;
                    break :blk switch (text_value) {
                        .string => |s| try allocator.dupe(u8, s),
                        else => return error.BadPromptJsonl,
                    };
                };
                errdefer allocator.free(text);
                const id = if (obj.get("id")) |v| switch (v) {
                    .string => |s| try allocator.dupe(u8, s),
                    .integer => |n| try std.fmt.allocPrint(allocator, "{d}", .{n}),
                    else => try std.fmt.allocPrint(allocator, "prompt-{d:0>2}", .{seq}),
                } else try std.fmt.allocPrint(allocator, "prompt-{d:0>2}", .{seq});
                errdefer allocator.free(id);
                try items.append(allocator, .{ .id = id, .text = text, .ids = ids });
                seq += 1;
            }
        },
    }
    list.items = try items.toOwnedSlice(allocator);
    return list;
}

fn tokenIdsField(allocator: std.mem.Allocator, v: std.json.Value) ![]u32 {
    const arr = switch (v) {
        .array => |a| a,
        else => return error.BadPromptJsonl,
    };
    if (arr.items.len == 0) return error.BadPromptJsonl;
    const ids = try allocator.alloc(u32, arr.items.len);
    errdefer allocator.free(ids);
    for (arr.items, ids) |item, *id| id.* = switch (item) {
        .integer => |n| std.math.cast(u32, n) orelse return error.BadPromptJsonl,
        else => return error.BadPromptJsonl,
    };
    return ids;
}

pub const PromptRecord = struct {
    id: []u8,
    dir: []u8,
    prompt_tokens: usize,
    generated_tokens: usize,
    strict_nll_mean: f64,
    strict_perplexity: f64,

    pub fn deinit(self: *const PromptRecord, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        allocator.free(self.dir);
    }
};

pub const BaselineMeta = struct {
    label: []const u8,
    model: []const u8,
    run: []const u8,
    kv_cache_format: []const u8,
    inference_profile: []const u8,
    prompt_set: []const u8,
    ssd_budget_gb: u64,
    tokens_per_prompt: u32,
    top_k: u32,
    elapsed_secs: f64,
};

pub const FixturePrompt = struct {
    id: []u8,
    dir: []u8,
    prompt_tokens: usize = 0,
    generated_tokens: usize = 0,
    strict_nll_mean: f64 = 0,
    strict_perplexity: f64 = 0,
};

pub const Baseline = struct {
    allocator: std.mem.Allocator,
    schema: []u8 = &.{},
    tool: []u8 = &.{},
    label: []u8 = &.{},
    model: []u8 = &.{},
    kv_cache_format: []u8 = &.{},
    inference_profile: []u8 = &.{},
    prompt_set: []u8 = &.{},
    tokens_per_prompt: u32 = 0,
    top_k: u32 = 0,
    ssd_budget_gb: u64 = 0,
    prompts: []FixturePrompt = &.{},

    pub fn deinit(self: *Baseline) void {
        const a = self.allocator;
        a.free(self.schema);
        a.free(self.tool);
        a.free(self.label);
        a.free(self.model);
        a.free(self.kv_cache_format);
        a.free(self.inference_profile);
        a.free(self.prompt_set);
        for (self.prompts) |p| {
            a.free(p.id);
            a.free(p.dir);
        }
        a.free(self.prompts);
        self.prompts = &.{};
    }
};

fn dupStringField(allocator: std.mem.Allocator, obj: std.json.ObjectMap, key: []const u8) ![]u8 {
    if (obj.get(key)) |v| switch (v) {
        .string => |s| return allocator.dupe(u8, s),
        else => {},
    };
    return allocator.dupe(u8, "");
}

fn floatField(obj: std.json.ObjectMap, key: []const u8) f64 {
    if (obj.get(key)) |v| switch (v) {
        .float => |f| return f,
        .integer => |n| return @floatFromInt(n),
        .number_string => |t| return std.fmt.parseFloat(f64, t) catch 0,
        else => {},
    };
    return 0;
}

fn intField(obj: std.json.ObjectMap, key: []const u8) u64 {
    if (obj.get(key)) |v| switch (v) {
        .integer => |n| return if (n < 0) 0 else @intCast(n),
        .float => |f| return if (f < 0) 0 else @intFromFloat(f),
        else => {},
    };
    return 0;
}

pub fn readBaseline(allocator: std.mem.Allocator, io: std.Io, dir: []const u8) !Baseline {
    const path = try std.fmt.allocPrint(allocator, "{s}/baseline.json", .{dir});
    defer allocator.free(path);
    const body = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(64 * 1024 * 1024)) catch return error.BaselineUnreadable;
    defer allocator.free(body);
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch return error.BadBaselineJson;
    defer parsed.deinit();
    const root = switch (parsed.value) {
        .object => |o| o,
        else => return error.BadBaselineJson,
    };
    var b = Baseline{ .allocator = allocator };
    errdefer b.deinit();
    b.schema = try dupStringField(allocator, root, "schema");
    b.tool = try dupStringField(allocator, root, "tool");
    b.label = try dupStringField(allocator, root, "label");
    b.model = try dupStringField(allocator, root, "model");
    b.kv_cache_format = try dupStringField(allocator, root, "kv_cache_format");
    b.inference_profile = try dupStringField(allocator, root, "inference_profile");
    b.prompt_set = try dupStringField(allocator, root, "prompt_set");
    b.tokens_per_prompt = @intCast(intField(root, "tokens_per_prompt"));
    b.top_k = @intCast(intField(root, "top_k"));
    b.ssd_budget_gb = intField(root, "ssd_budget_gb");
    const arr = switch (root.get("prompts") orelse return error.BadBaselineJson) {
        .array => |a| a,
        else => return error.BadBaselineJson,
    };
    var prompts: std.ArrayList(FixturePrompt) = .empty;
    errdefer {
        for (prompts.items) |p| {
            allocator.free(p.id);
            allocator.free(p.dir);
        }
        prompts.deinit(allocator);
    }
    for (arr.items) |item| {
        const obj = switch (item) {
            .object => |o| o,
            else => return error.BadBaselineJson,
        };
        const id = try dupStringField(allocator, obj, "id");
        errdefer allocator.free(id);
        const sub = try dupStringField(allocator, obj, "dir");
        errdefer allocator.free(sub);
        if (sub.len == 0) return error.BadBaselineJson;
        try prompts.append(allocator, .{
            .id = id,
            .dir = sub,
            .prompt_tokens = @intCast(intField(obj, "prompt_tokens")),
            .generated_tokens = @intCast(intField(obj, "generated_tokens")),
            .strict_nll_mean = floatField(obj, "strict_nll_mean"),
            .strict_perplexity = floatField(obj, "strict_perplexity"),
        });
    }
    b.prompts = try prompts.toOwnedSlice(allocator);
    return b;
}

fn writeJsonString(w: *std.Io.Writer, s: []const u8) !void {
    try w.writeAll("\"");
    for (s) |c| switch (c) {
        '"' => try w.writeAll("\\\""),
        '\\' => try w.writeAll("\\\\"),
        '\n' => try w.writeAll("\\n"),
        '\r' => try w.writeAll("\\r"),
        '\t' => try w.writeAll("\\t"),
        else => {
            if (c < 0x20) {
                try w.print("\\u{x:0>4}", .{c});
            } else {
                try w.writeByte(c);
            }
        },
    };
    try w.writeAll("\"");
}

pub fn writeBaseline(
    allocator: std.mem.Allocator,
    io: std.Io,
    out_root: []const u8,
    meta: BaselineMeta,
    records: []const PromptRecord,
) !void {
    _ = allocator;
    try std.Io.Dir.cwd().createDirPath(io, out_root);
    var dir = try std.Io.Dir.cwd().openDir(io, out_root, .{});
    defer dir.close(io);
    var f = try dir.createFile(io, "baseline.json", .{});
    defer f.close(io);
    var buf: [64 * 1024]u8 = undefined;
    var fw = f.writer(io, &buf);
    const w = &fw.interface;
    try w.writeAll("{\n  \"schema\": ");
    try writeJsonString(w, SCHEMA);
    try w.writeAll(",\n  \"tool\": ");
    try writeJsonString(w, TOOL);
    try w.writeAll(",\n  \"label\": ");
    try writeJsonString(w, meta.label);
    try w.writeAll(",\n  \"model\": ");
    try writeJsonString(w, meta.model);
    try w.writeAll(",\n  \"run\": ");
    try writeJsonString(w, meta.run);
    try w.writeAll(",\n  \"kv_cache_format\": ");
    try writeJsonString(w, meta.kv_cache_format);
    try w.writeAll(",\n  \"inference_profile\": ");
    try writeJsonString(w, meta.inference_profile);
    try w.print(",\n  \"ssd_budget_gb\": {d}", .{meta.ssd_budget_gb});
    try w.writeAll(",\n  \"prompt_set\": ");
    try writeJsonString(w, meta.prompt_set);
    try w.print(",\n  \"tokens_per_prompt\": {d}", .{meta.tokens_per_prompt});
    try w.print(",\n  \"top_k\": {d}", .{meta.top_k});
    try w.print(",\n  \"elapsed_secs\": {d}", .{meta.elapsed_secs});
    try w.writeAll(",\n  \"prompts\": [\n");
    for (records, 0..) |r, i| {
        try w.writeAll("    {\"id\": ");
        try writeJsonString(w, r.id);
        try w.writeAll(", \"dir\": ");
        try writeJsonString(w, r.dir);
        try w.print(", \"prompt_tokens\": {d}, \"generated_tokens\": {d}, \"strict_nll_mean\": {d}, \"strict_perplexity\": {d}}}", .{
            r.prompt_tokens,
            r.generated_tokens,
            r.strict_nll_mean,
            r.strict_perplexity,
        });
        if (i + 1 < records.len) try w.writeAll(",");
        try w.writeAll("\n");
    }
    try w.writeAll("  ]\n}\n");
    try w.flush();
}

pub fn readIdList(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]u32 {
    const raw = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(64 * 1024 * 1024)) catch return error.TokenListUnreadable;
    defer allocator.free(raw);
    var ids: std.ArrayList(u32) = .empty;
    errdefer ids.deinit(allocator);
    var parts = std.mem.splitScalar(u8, std.mem.trim(u8, raw, " \t\r\n"), ',');
    while (parts.next()) |part| {
        const t = std.mem.trim(u8, part, " \t\r\n");
        if (t.len == 0) continue;
        try ids.append(allocator, std.fmt.parseInt(u32, t, 10) catch return error.BadTokenList);
    }
    return ids.toOwnedSlice(allocator);
}

fn writeTextFile(io: std.Io, dir: std.Io.Dir, name: []const u8, body: []const u8) !void {
    var f = try dir.createFile(io, name, .{});
    defer f.close(io);
    var buf: [16 * 1024]u8 = undefined;
    var fw = f.writer(io, &buf);
    try fw.interface.writeAll(body);
    try fw.interface.flush();
}

fn writeIdListFile(io: std.Io, dir: std.Io.Dir, name: []const u8, ids: []const u32) !void {
    var f = try dir.createFile(io, name, .{});
    defer f.close(io);
    var buf: [16 * 1024]u8 = undefined;
    var fw = f.writer(io, &buf);
    for (ids, 0..) |id, i| {
        if (i > 0) try fw.interface.writeAll(",");
        try fw.interface.print("{d}", .{id});
    }
    try fw.interface.flush();
}

fn sanitizeDirName(allocator: std.mem.Allocator, id: []const u8) ![]u8 {
    const out = try allocator.alloc(u8, id.len);
    for (id, 0..) |c, i| {
        out[i] = switch (c) {
            'a'...'z', 'A'...'Z', '0'...'9', '-', '_' => c,
            else => '_',
        };
    }
    return out;
}

fn writeAllFd(fd: std.c.fd_t, bytes: []const u8) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const got = std.c.write(fd, bytes[done..].ptr, bytes.len - done);
        if (got < 0) {
            if (std.c._errno().* == @backingInt(std.c.E.INTR)) continue;
            return error.LogitsWriteFailed;
        }
        if (got == 0) return error.LogitsWriteFailed;
        done += @intCast(got);
    }
}

pub const PromptWriter = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    rel_dir: []u8,
    abs_dir: []u8,
    id: []u8,
    fd: std.c.fd_t,
    rows: usize = 0,
    vocab: usize = 0,
    nll_sum: f64 = 0,

    pub fn begin(
        allocator: std.mem.Allocator,
        io: std.Io,
        out_root: []const u8,
        index: usize,
        id: []const u8,
    ) !PromptWriter {
        const safe = try sanitizeDirName(allocator, id);
        defer allocator.free(safe);
        const rel_dir = try std.fmt.allocPrint(allocator, "prompts/{d:0>2}_{s}", .{ index, safe });
        errdefer allocator.free(rel_dir);
        const abs_dir = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ out_root, rel_dir });
        errdefer allocator.free(abs_dir);
        const own_id = try allocator.dupe(u8, id);
        errdefer allocator.free(own_id);
        try std.Io.Dir.cwd().createDirPath(io, abs_dir);
        const logits_path = try std.fmt.allocPrintSentinel(allocator, "{s}/logits.f32", .{abs_dir}, 0);
        defer allocator.free(logits_path);
        const fd = std.c.open(logits_path.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
        if (fd < 0) return error.LogitsCreateFailed;
        return .{
            .allocator = allocator,
            .io = io,
            .rel_dir = rel_dir,
            .abs_dir = abs_dir,
            .id = own_id,
            .fd = fd,
        };
    }

    pub fn appendRow(self: *PromptWriter, row: []const f32, chosen: u32) !void {
        if (self.rows == 0) {
            self.vocab = row.len;
        } else if (row.len != self.vocab) {
            return error.LogitsRowWidthChanged;
        }
        if (chosen >= row.len) return error.ChosenTokenOutOfRange;
        try writeAllFd(self.fd, std.mem.sliceAsBytes(row));
        self.nll_sum += rowNll(row, chosen);
        self.rows += 1;
    }

    pub fn finish(
        self: *PromptWriter,
        prompt_text: []const u8,
        rendered: []const u8,
        prompt_ids: []const u32,
        generated_ids: []const u32,
    ) !PromptRecord {
        if (self.fd >= 0) {
            _ = std.c.close(self.fd);
            self.fd = -1;
        }
        var dir = try std.Io.Dir.cwd().openDir(self.io, self.abs_dir, .{});
        defer dir.close(self.io);
        try writeTextFile(self.io, dir, "id.txt", self.id);
        try writeTextFile(self.io, dir, "prompt.txt", prompt_text);
        try writeTextFile(self.io, dir, "rendered_prompt.txt", rendered);
        try writeIdListFile(self.io, dir, "prompt_tokens.txt", prompt_ids);
        try writeIdListFile(self.io, dir, "generated_tokens.txt", generated_ids);
        const mean = if (self.rows == 0) 0 else self.nll_sum / @as(f64, @floatFromInt(self.rows));
        const id = try self.allocator.dupe(u8, self.id);
        errdefer self.allocator.free(id);
        const rel = try self.allocator.dupe(u8, self.rel_dir);
        return .{
            .id = id,
            .dir = rel,
            .prompt_tokens = prompt_ids.len,
            .generated_tokens = generated_ids.len,
            .strict_nll_mean = mean,
            .strict_perplexity = @exp(mean),
        };
    }

    pub fn deinit(self: *PromptWriter) void {
        if (self.fd >= 0) {
            _ = std.c.close(self.fd);
            self.fd = -1;
        }
        self.allocator.free(self.rel_dir);
        self.allocator.free(self.abs_dir);
        self.allocator.free(self.id);
    }
};

pub fn kvCacheFormat(cfg: transformer_mod.KVQuantConfig) []const u8 {
    return switch (cfg.scheme) {
        .off => "bf16",
        .affine => switch (cfg.bits) {
            4 => "affine4",
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
    xfm: transformer_mod.Transformer,

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
    scheduler_mod.applyModelSettings(&self.config, model_settings_mod.overrideFor(allocator, io, opts.model_dir));
    self.config.ctx_override = model_settings_mod.contextPick(opts.ctx_size, self.config.ctx_override).value;

    const kld_budget = scheduler_mod.resolveSsdBudget(opts.ssd_budget_bytes, self.config.ssd_budget_gb_override, self.config.supportsExpertStreaming());
    if (expert_stream_mod.expertStreamingEngaged(
        self.config.supportsExpertStreaming(),
        self.config.expertStreamingRequired(),
        opts.expert_cache_bytes,
        kld_budget.bytes,
    )) {
        const budget = kld_budget;
        if (opts.expert_cache_bytes == 0 and budget.bytes == 0) return error.ExpertStreamingRequired;
        const mtp = model_settings_mod.MtpChoice.resolve(model_settings_mod.launchFlag(bool, opts.enable_mtp, opts.mtp_explicit), self.config.mtp_override, false);
        switch (expert_stream_mod.mtpUnderStreaming(mtp.on, mtp.source == .model_settings, mtp.source == .default)) {
            .keep => {
                log.err("[expert-stream] KLD scoring under expert streaming runs without the MTP head; drop --mtp\n", .{});
                return error.ExpertStreamingMtpUnsupported;
            },
            .drop_settings => {
                log.info("[expert-stream] model-settings mtp=true ignored: {s}\n", .{expert_stream_mod.MTP_UNSUPPORTED});
                self.config.mtp_override = false;
            },
            .drop_default, .off => {},
        }
        const plan = try scheduler_mod.planExpertStreaming(io, allocator, &self.config, opts.model_dir, opts.expert_cache_bytes, budget.bytes);
        scheduler_mod.applyStreamingPlan(&self.config, plan, budget.bytes);
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
    self.chat_config = try chat_mod.loadChatConfig(io, allocator, opts.model_dir);
    errdefer self.chat_config.deinit();

    self.weights = try model_mod.loadWeightsForConfig(io, allocator, opts.model_dir, &self.config, false);
    errdefer self.weights.deinit();
    model_mod.resolveWeightPrefix(&self.config, &self.weights);

    self.xfm = try transformer_mod.Transformer.init(io, allocator, self.config, &self.weights);
    errdefer self.xfm.deinit();
    if (opts.kv_quant_config.scheme != .off) {
        try self.xfm.cache.reinit(self.config.num_hidden_layers, opts.kv_quant_config);
    }
    try self.xfm.qwen4MtpApplyKvQuant(opts.kv_quant_config);
    _ = mlx.applyWiredPolicy();
    if (self.config.hidden_act == .gelu_approx) {
        self.xfm.compileGelu();
        self.xfm.compileGeglu();
    }
    if (self.config.final_logit_softcapping > 0.0) self.xfm.compileSoftcap();
    if (self.xfm.moe_layers != null) self.xfm.compileMoeRouting();
    if (self.config.linear_num_key_heads > 0) self.xfm.compileGdnGate();
    return self;
}

fn logitsVocab(logits: mlx.mlx_array) !usize {
    const shape = mlx.getShape(logits);
    if (shape.len != 3) return error.KldLogitsShape;
    return @intCast(shape[2]);
}

fn readLastRow(xfm: *transformer_mod.Transformer, logits: mlx.mlx_array, dst: []f32) !void {
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
        writeAllFd(1, line) catch return;
    }
};

fn promptIds(allocator: std.mem.Allocator, l: *Loaded, opts: Options, p: Prompt) ![]u32 {
    if (p.ids) |ids| return allocator.dupe(u32, ids);
    const text = p.text;
    if (opts.no_template) return l.tok.encode(allocator, text);
    const messages = [_]chat_mod.Message{.{ .role = "user", .content = text }};
    return chat_mod.formatChat(allocator, &l.tok, &messages, &l.chat_config, null, null, false, null, false);
}

fn renderPrompt(allocator: std.mem.Allocator, l: *Loaded, opts: Options, p: Prompt) ![]const u8 {
    if (p.ids) |ids| return l.tok.decode(allocator, ids, false);
    const text = p.text;
    if (opts.no_template) return allocator.dupe(u8, text);
    const messages = [_]chat_mod.Message{.{ .role = "user", .content = text }};
    return chat_mod.renderChatTemplate(allocator, &messages, &l.chat_config, null, null, false, null, false);
}

/// GLM prefills in serving-width chunks, identically at capture and compare: the chunk width is
/// part of GLM's numerics, and a 128K-token prompt does not fit one forward.
const glm_prompt_chunk: usize = 2048;

fn forwardPrompt(allocator: std.mem.Allocator, l: *Loaded, ctx: *transformer_mod.ForwardCtx, ids: []const u32) !mlx.mlx_array {
    return forwardPromptChunks(allocator, l, ctx, ids, if (l.config.isGlm5()) glm_prompt_chunk else ids.len);
}

fn forwardPromptChunks(allocator: std.mem.Allocator, l: *Loaded, ctx: *transformer_mod.ForwardCtx, ids: []const u32, chunk: usize) !mlx.mlx_array {
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
fn forwardPromptCapture(allocator: std.mem.Allocator, l: *Loaded, ctx: *transformer_mod.ForwardCtx, ids: []const u32, hidden: ?*hidden_capture.Writer) !mlx.mlx_array {
    const w = hidden orelse return forwardPrompt(allocator, l, ctx, ids);
    // Each chunk would replace the previous chunk's captured boundaries.
    if (l.config.isGlm5() and ids.len > glm_prompt_chunk) return error.GlmHiddenCaptureNeedsOneChunk;
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
    var cl: transformer_mod.CaptureLayers = .{ .ids = layer_ids, .out = rows[1..], .input = &rows[0] };
    ctx.capture_layers = &cl;
    defer ctx.capture_layers = null;
    const logits = try forwardPrompt(allocator, l, ctx, ids);
    errdefer _ = mlx.mlx_array_free(logits);
    try w.append(l.xfm.s, ids, rows);
    return logits;
}

fn forwardOne(l: *Loaded, ctx: *transformer_mod.ForwardCtx, token: u32) !mlx.mlx_array {
    const data = [_]i32{@intCast(token)};
    const input = mlx.mlx_array_new_data(&data, &[_]c_int{ 1, 1 }, 2, .int32);
    defer _ = mlx.mlx_array_free(input);
    return l.xfm.forwardWith(ctx, input);
}

pub fn capturedSsdBudgetGb(config: *const model_mod.ModelConfig, flag_bytes: u64) u64 {
    if (!config.expert_streaming) return 0;
    if (config.expert_ssd_budget_bytes > 0) return config.expert_ssd_budget_bytes >> 30;
    return flag_bytes >> 30;
}

pub fn runCapture(io: std.Io, allocator: std.mem.Allocator, l: *Loaded, opts: Options, out: *Out) !void {
    var prompts = try loadPrompts(allocator, io, opts.prompts, opts.limit);
    defer prompts.deinit();
    if (prompts.items.len == 0) return error.NoPromptsFound;
    try std.Io.Dir.cwd().createDirPath(io, opts.out_dir);
    const hidden = if (opts.hidden_out.len > 0)
        try hidden_capture.Writer.open(allocator, io, opts.hidden_out, l.config.num_hidden_layers, hiddenCaptureWidth(&l.config))
    else
        null;
    defer if (hidden) |w| w.close();

    var records: std.ArrayList(PromptRecord) = .empty;
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

        var writer = try PromptWriter.begin(allocator, io, opts.out_dir, index, p.id);
        defer writer.deinit();
        var generated: std.ArrayList(u32) = .empty;
        defer generated.deinit(allocator);

        var ctx = l.xfm.defaultCtx();
        var logits = try forwardPromptCapture(allocator, l, &ctx, ids, hidden);
        defer _ = mlx.mlx_array_free(logits);
        const vocab = try logitsVocab(logits);
        const row = try allocator.alloc(f32, vocab);
        defer allocator.free(row);

        var step: u32 = 0;
        while (step < opts.tokens) : (step += 1) {
            try readLastRow(&l.xfm, logits, row);
            const chosen = argmaxOf(row);
            try writer.appendRow(row, chosen);
            try generated.append(allocator, chosen);
            if (step + 1 == opts.tokens) break;
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

    try writeBaseline(allocator, io, opts.out_dir, .{
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
    out.print("[kld] captured {d} prompts x {d} tokens into {s} in {d:.1}s (mean strict NLL {d:.9})\n", .{
        records.items.len,
        opts.tokens,
        opts.out_dir,
        elapsed_secs,
        mean_nll,
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

pub fn firstEosPosition(generated: []const u32, eos: []const u32) ?usize {
    for (generated, 0..) |t, i| {
        for (eos) |e| if (t == e) return i;
    }
    return null;
}

pub fn runCompare(io: std.Io, allocator: std.mem.Allocator, l: *Loaded, opts: Options, out: *Out) !void {
    var memory: MemoryReport = undefined;
    try mlx.check(mlx.mlx_get_active_memory(&memory.model_active_bytes));
    try mlx.check(mlx.mlx_reset_peak_memory());

    var base = try readBaseline(allocator, io, opts.fixture);
    defer base.deinit();
    const count = if (opts.limit > 0 and opts.limit < base.prompts.len) opts.limit else base.prompts.len;
    if (count == 0) return error.NoPromptsFound;

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
        const prompt_path = try std.fmt.allocPrint(allocator, "{s}/{s}/prompt_tokens.txt", .{ opts.fixture, fp.dir });
        defer allocator.free(prompt_path);
        const prompt_ids = try readIdList(allocator, io, prompt_path);
        defer allocator.free(prompt_ids);
        const gen_path = try std.fmt.allocPrint(allocator, "{s}/{s}/generated_tokens.txt", .{ opts.fixture, fp.dir });
        defer allocator.free(gen_path);
        const generated = try readIdList(allocator, io, gen_path);
        defer allocator.free(generated);
        if (prompt_ids.len == 0 or generated.len == 0) return error.EmptyFixturePrompt;

        const logits_path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}/logits.f32", .{ opts.fixture, fp.dir }, 0);
        defer allocator.free(logits_path);
        const teacher_fd = std.c.open(logits_path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (teacher_fd < 0) return error.TeacherLogitsMissing;
        defer _ = std.c.close(teacher_fd);
        var st: std.c.Stat = undefined;
        if (std.c.fstat(teacher_fd, &st) != 0 or st.size < 0) return error.TeacherLogitsStatFailed;
        const size: u64 = @intCast(st.size);
        const row_bytes = generated.len * @sizeOf(f32);
        if (row_bytes == 0 or size % row_bytes != 0) return error.TeacherLogitsSizeMismatch;
        const vocab: usize = @intCast(size / row_bytes);

        const teacher_row = try allocator.alloc(f32, vocab);
        defer allocator.free(teacher_row);
        const model_row = try allocator.alloc(f32, vocab);
        defer allocator.free(model_row);

        var ctx = l.xfm.defaultCtx();
        var logits = try forwardPrompt(allocator, l, &ctx, prompt_ids);
        defer _ = mlx.mlx_array_free(logits);

        var prompt_kld: f64 = 0;
        var prompt_nll: f64 = 0;
        var prompt_cosine_similarity: f64 = 0;
        var prompt_cosine_loss: f64 = 0;
        var prompt_worst_cosine_loss: f64 = 0;
        var prompt_top1: usize = 0;
        const eos_pos = firstEosPosition(generated, eos_set.items);
        const n_eos: usize = if (eos_pos) |e| e + 1 else generated.len;
        var kld_eos: f64 = 0;
        var nll_eos: f64 = 0;
        var cosine_similarity_eos: f64 = 0;
        var cosine_loss_eos: f64 = 0;
        var worst_cosine_loss_eos_prompt: f64 = 0;
        var top1_eos: usize = 0;
        const per_pos = try allocator.alloc(f64, generated.len);
        for (generated, 0..) |token, position| {
            try expert_stream_mod.readExact(teacher_fd, std.mem.sliceAsBytes(teacher_row), position * vocab * @sizeOf(f32));
            if (position > 0) {
                const next = try forwardOne(l, &ctx, generated[position - 1]);
                _ = mlx.mlx_array_free(logits);
                logits = next;
            }
            try readLastRow(&l.xfm, logits, model_row);
            const score = try scoreRow(teacher_row, model_row, token);
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
    try mlx.check(mlx.mlx_get_peak_memory(&memory.peak_active_bytes));
    try mlx.check(mlx.mlx_get_active_memory(&memory.final_active_bytes));
    try mlx.check(mlx.mlx_get_cache_memory(&memory.final_cache_bytes));

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
        try writeJsonString(w, COMPARE_SCHEMA);
        try w.writeAll(",\n  \"tool\": ");
        try writeJsonString(w, TOOL);
        try w.writeAll(",\n  \"label\": ");
        try writeJsonString(w, opts.label);
        try w.writeAll(",\n  \"model\": ");
        try writeJsonString(w, opts.model_dir);
        try w.writeAll(",\n  \"fixture\": ");
        try writeJsonString(w, opts.fixture);
        try w.writeAll(",\n  \"kv_cache_format\": ");
        try writeJsonString(w, kvCacheFormat(opts.kv_quant_config));
        try w.print(",\n  \"expert_pick_tolerance\": {d}", .{opts.pick_tolerance});
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
            try writeJsonString(w, s.id);
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

pub fn cmdKld(allocator: std.mem.Allocator, io: std.Io, args: []const []const u8) !void {
    var out: Out = .{};
    var opts = parseArgs(args) catch |err| {
        out.print("sushi kld: {s}\n\n{s}", .{ @errorName(err), USAGE });
        return err;
    };
    if (opts.help) {
        out.print("{s}", .{USAGE});
        return;
    }
    if (opts.command == .capture) if (hidden_capture.envPath()) |dir| {
        opts.hidden_out = dir;
        log.info("[kld] {s}: appending every prompt forward's block boundaries to {s}\n", .{ hidden_capture.ENV_VAR, dir });
    };
    if (try @import("glm5_kld_capture.zig").tryRun(allocator, io, opts, &out)) return;
    expert_stream_mod.pick_tolerance = opts.pick_tolerance;
    if (opts.wired_margin_bytes > 0) server_mod.wired_limit_margin_bytes = opts.wired_margin_bytes;
    const loaded = try loadModel(io, allocator, opts);
    defer loaded.deinit();
    switch (opts.command) {
        .capture => try runCapture(io, allocator, loaded, opts, &out),
        .compare => try runCompare(io, allocator, loaded, opts, &out),
    }
}

/// The 60x64 teacher fixture under the models root, or null when it is absent.
fn teacherFixture(buf: []u8, io: std.Io) ?[]const u8 {
    const path = @import("test_models.zig").packPath(buf, "kld-teacher/Qwen3.8-Flash-Next-wikitext2-60x64") catch return null;
    var dir = std.Io.Dir.openDirAbsolute(io, path, .{}) catch return null;
    defer dir.close(io);
    _ = dir.statFile(io, "baseline.json", .{}) catch return null;
    return path;
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

test "kld: scoreRow on identical logits is zero divergence and the plain NLL" {
    const logits = [_]f32{ 0.5, -1.0, 2.0, 0.0, 3.5, -2.5, 1.0, 0.25 };
    const s = try scoreRow(&logits, &logits, 4);
    try testing.expectApproxEqAbs(@as(f64, 0), s.kld, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1), s.cosine_similarity, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0), s.cosine_loss, 1e-12);
    try testing.expect(s.top1);
    var sum: f64 = 0;
    for (logits) |v| sum += @exp(@as(f64, v) - 3.5);
    try testing.expectApproxEqAbs(@log(sum), s.nll, 1e-12);
}

test "kld: scoreRow reports raw-logit cosine for identical, orthogonal, opposite, and scaling rows" {
    const teacher = [_]f32{ 3.0, 4.0 };
    const identical = [_]f32{ 3.0, 4.0 };
    const scaled = [_]f32{ 30.0, 40.0 };
    const orthogonal = [_]f32{ 4.0, -3.0 };
    const opposite = [_]f32{ -3.0, -4.0 };

    const same = try scoreRow(&teacher, &identical, 0);
    try testing.expectApproxEqAbs(@as(f64, 1), same.cosine_similarity, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0), same.cosine_loss, 1e-12);

    const scale = try scoreRow(&teacher, &scaled, 0);
    try testing.expectApproxEqAbs(@as(f64, 1), scale.cosine_similarity, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0), scale.cosine_loss, 1e-12);

    const right_angle = try scoreRow(&teacher, &orthogonal, 0);
    try testing.expectApproxEqAbs(@as(f64, 0), right_angle.cosine_similarity, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1), right_angle.cosine_loss, 1e-12);

    const reverse = try scoreRow(&teacher, &opposite, 0);
    try testing.expectApproxEqAbs(@as(f64, -1), reverse.cosine_similarity, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 2), reverse.cosine_loss, 1e-12);
}

test "kld: scoreRow rejects nonfinite and zero-norm cosine rows" {
    const valid = [_]f32{ 1.0, 2.0 };
    const zero = [_]f32{ 0.0, 0.0 };
    const nan_row = [_]f32{ std.math.nan(f32), 2.0 };
    const inf_row = [_]f32{ 1.0, std.math.inf(f32) };

    try testing.expectError(error.NonFinite, scoreRow(&nan_row, &valid, 0));
    try testing.expectError(error.NonFinite, scoreRow(&valid, &inf_row, 0));
    try testing.expectError(error.ZeroNorm, scoreRow(&zero, &valid, 0));
    try testing.expectError(error.ZeroNorm, scoreRow(&valid, &zero, 0));
}

fn refScore(teacher: []const f32, model: []const f32, token: usize) RowScore {
    var tmax: f64 = -std.math.inf(f64);
    var mmax: f64 = -std.math.inf(f64);
    var mtop: usize = 0;
    var dot: f64 = 0;
    var teacher_norm_sq: f64 = 0;
    var model_norm_sq: f64 = 0;
    for (teacher, 0..) |v, i| {
        if (v > tmax) tmax = v;
        if (model[i] > mmax) {
            mmax = model[i];
            mtop = i;
        }
        const tv = @as(f64, v);
        const mv = @as(f64, model[i]);
        dot += tv * mv;
        teacher_norm_sq += tv * tv;
        model_norm_sq += mv * mv;
    }
    var tsum: f64 = 0;
    var msum: f64 = 0;
    for (teacher, 0..) |v, i| {
        tsum += @exp(@as(f64, v) - tmax);
        msum += @exp(@as(f64, model[i]) - mmax);
    }
    const tlz = tmax + @log(tsum);
    const mlz = mmax + @log(msum);
    var kld: f64 = 0;
    for (teacher, 0..) |v, i| {
        const lp = @as(f64, v) - tlz;
        const lq = @as(f64, model[i]) - mlz;
        kld += @exp(lp) * (lp - lq);
    }
    const cosine_similarity = dot / (@sqrt(teacher_norm_sq) * @sqrt(model_norm_sq));
    return .{
        .kld = kld,
        .nll = mlz - model[token],
        .top1 = mtop == token,
        .cosine_similarity = cosine_similarity,
        .cosine_loss = 1.0 - cosine_similarity,
    };
}

test "kld: scoreRow matches the closed form on two 8-vocab rows" {
    const teacher = [_]f32{ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };
    const model = [_]f32{ 1.0, 0.5, -0.5, 2.0, 0.0, -1.0, 0.25, 3.0 };

    var msum: f64 = 0;
    for (model) |v| msum += @exp(@as(f64, v));
    const log_z = @log(msum);
    var mean_logit: f64 = 0;
    for (model) |v| mean_logit += @as(f64, v) / 8.0;
    const expected_kld = -@log(@as(f64, 8.0)) - mean_logit + log_z;
    const expected_nll = log_z - 2.0;

    const s = try scoreRow(&teacher, &model, 3);
    try testing.expectApproxEqAbs(expected_kld, s.kld, 1e-12);
    try testing.expectApproxEqAbs(expected_nll, s.nll, 1e-12);
    try testing.expect(!s.top1);
    const s7 = try scoreRow(&teacher, &model, 7);
    try testing.expect(s7.top1);
    try testing.expectApproxEqAbs(refScore(&teacher, &model, 7).kld, s7.kld, 1e-12);
}

test "kld: scoreRow is log-sum-exp stable on large logits" {
    var teacher: [8]f32 = .{ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };
    var model: [8]f32 = .{ 1.0, 0.5, -0.5, 2.0, 0.0, -1.0, 0.25, 3.0 };
    const base = try scoreRow(&teacher, &model, 3);
    for (&teacher) |*v| v.* += 90000.0;
    for (&model) |*v| v.* += 90000.0;
    const shifted = try scoreRow(&teacher, &model, 3);
    try testing.expect(std.math.isFinite(shifted.kld));
    try testing.expect(std.math.isFinite(shifted.nll));
    try testing.expectApproxEqAbs(base.kld, shifted.kld, 1e-9);
    try testing.expectApproxEqAbs(base.nll, shifted.nll, 1e-9);
}

test "kld: a captured fixture round-trips through the reader byte for byte" {
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &path_buf);
    const out_root = path_buf[0..root_len];

    const ids = [_][]const u32{ &.{ 5, 6, 7 }, &.{ 1, 2 } };
    const gen = [_][]const u32{ &.{ 0, 3, 6 }, &.{ 7, 1, 2 } };
    const texts = [_][]const u8{ "first prompt", "second prompt" };
    const rendered = [_][]const u8{ "<|im_start|>user\nfirst prompt<|im_end|>\n", "<|im_start|>user\nsecond prompt<|im_end|>\n" };
    var rows_written: [2][3][8]f32 = undefined;

    var records: [2]PromptRecord = undefined;
    for (0..2) |p| {
        var w = try PromptWriter.begin(allocator, io, out_root, p, if (p == 0) "alpha" else "beta");
        defer w.deinit();
        for (0..3) |r| {
            var row: [8]f32 = undefined;
            for (&row, 0..) |*v, i| v.* = @as(f32, @floatFromInt(p * 100 + r * 10 + i)) * 0.125;
            row[gen[p][r]] = 9.0;
            rows_written[p][r] = row;
            try w.appendRow(&row, gen[p][r]);
        }
        records[p] = try w.finish(texts[p], rendered[p], ids[p], gen[p]);
    }
    defer for (&records) |*r| r.deinit(allocator);

    try writeBaseline(allocator, io, out_root, .{
        .label = "round-trip",
        .model = "/models/fake",
        .run = "round-trip",
        .kv_cache_format = "bf16",
        .inference_profile = "greedy",
        .prompt_set = "unit-test",
        .ssd_budget_gb = 0,
        .tokens_per_prompt = 3,
        .top_k = 10,
        .elapsed_secs = 1.5,
    }, &records);

    var base = try readBaseline(allocator, io, out_root);
    defer base.deinit();
    try testing.expectEqualStrings(SCHEMA, base.schema);
    try testing.expectEqualStrings(TOOL, base.tool);
    try testing.expectEqualStrings("round-trip", base.label);
    try testing.expectEqualStrings("bf16", base.kv_cache_format);
    try testing.expectEqualStrings("greedy", base.inference_profile);
    try testing.expectEqual(@as(u32, 3), base.tokens_per_prompt);
    try testing.expectEqual(@as(usize, 2), base.prompts.len);
    try testing.expectEqualStrings("alpha", base.prompts[0].id);
    try testing.expectEqualStrings("prompts/00_alpha", base.prompts[0].dir);
    try testing.expectEqualStrings("prompts/01_beta", base.prompts[1].dir);
    try testing.expectEqual(@as(usize, 3), base.prompts[0].prompt_tokens);
    try testing.expectEqual(@as(usize, 3), base.prompts[0].generated_tokens);
    try testing.expectEqual(@as(usize, 2), base.prompts[1].prompt_tokens);
    try testing.expectApproxEqAbs(records[0].strict_nll_mean, base.prompts[0].strict_nll_mean, 1e-9);
    try testing.expectApproxEqAbs(records[0].strict_perplexity, base.prompts[0].strict_perplexity, 1e-9);

    for (0..2) |p| {
        const dir_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ out_root, base.prompts[p].dir });
        defer allocator.free(dir_path);
        const tok_path = try std.fmt.allocPrint(allocator, "{s}/prompt_tokens.txt", .{dir_path});
        defer allocator.free(tok_path);
        const got_prompt = try readIdList(allocator, io, tok_path);
        defer allocator.free(got_prompt);
        try testing.expectEqualSlices(u32, ids[p], got_prompt);

        const gen_path = try std.fmt.allocPrint(allocator, "{s}/generated_tokens.txt", .{dir_path});
        defer allocator.free(gen_path);
        const got_gen = try readIdList(allocator, io, gen_path);
        defer allocator.free(got_gen);
        try testing.expectEqualSlices(u32, gen[p], got_gen);

        const text_path = try std.fmt.allocPrint(allocator, "{s}/prompt.txt", .{dir_path});
        defer allocator.free(text_path);
        const got_text = try std.Io.Dir.cwd().readFileAlloc(io, text_path, allocator, .limited(1 << 16));
        defer allocator.free(got_text);
        try testing.expectEqualStrings(texts[p], got_text);

        const rend_path = try std.fmt.allocPrint(allocator, "{s}/rendered_prompt.txt", .{dir_path});
        defer allocator.free(rend_path);
        const got_rend = try std.Io.Dir.cwd().readFileAlloc(io, rend_path, allocator, .limited(1 << 16));
        defer allocator.free(got_rend);
        try testing.expectEqualStrings(rendered[p], got_rend);

        const id_path = try std.fmt.allocPrint(allocator, "{s}/id.txt", .{dir_path});
        defer allocator.free(id_path);
        const got_id = try std.Io.Dir.cwd().readFileAlloc(io, id_path, allocator, .limited(1 << 16));
        defer allocator.free(got_id);
        try testing.expectEqualStrings(base.prompts[p].id, got_id);

        const logits_path = try std.fmt.allocPrint(allocator, "{s}/logits.f32", .{dir_path});
        defer allocator.free(logits_path);
        const got_logits = try std.Io.Dir.cwd().readFileAlloc(io, logits_path, allocator, .limited(1 << 20));
        defer allocator.free(got_logits);
        try testing.expectEqual(@as(usize, 3 * 8 * 4), got_logits.len);
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&rows_written[p]), got_logits);

        var nll: f64 = 0;
        for (0..3) |r| nll += refScore(&rows_written[p][r], &rows_written[p][r], gen[p][r]).nll;
        try testing.expectApproxEqAbs(nll / 3.0, records[p].strict_nll_mean, 1e-9);
        try testing.expectApproxEqAbs(@exp(nll / 3.0), records[p].strict_perplexity, 1e-9);
    }
}

test "kld: the real teacher fixture reads back at full width" {
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var fixture_buf: [std.fs.max_path_bytes]u8 = undefined;
    const fixture = teacherFixture(&fixture_buf, io) orelse return error.SkipZigTest;

    var base = try readBaseline(allocator, io, fixture);
    defer base.deinit();
    try testing.expectEqual(@as(usize, 60), base.prompts.len);
    try testing.expectEqual(@as(u32, 64), base.tokens_per_prompt);
    try testing.expectEqualStrings("wikitext2-test-00-robert-boulter", base.prompts[0].id);
    try testing.expectEqualStrings("prompts/00_wikitext2-test-00-robert-boulter", base.prompts[0].dir);

    const gen_path = try std.fmt.allocPrint(allocator, "{s}/prompts/00_wikitext2-test-00-robert-boulter/generated_tokens.txt", .{fixture});
    defer allocator.free(gen_path);
    const gen = try readIdList(allocator, io, gen_path);
    defer allocator.free(gen);
    try testing.expectEqual(@as(usize, 64), gen.len);

    const tok_path = try std.fmt.allocPrint(allocator, "{s}/prompts/00_wikitext2-test-00-robert-boulter/prompt_tokens.txt", .{fixture});
    defer allocator.free(tok_path);
    const ptok = try readIdList(allocator, io, tok_path);
    defer allocator.free(ptok);
    try testing.expectEqual(@as(usize, 343), ptok.len);

    const logits_path = try std.fmt.allocPrint(allocator, "{s}/prompts/00_wikitext2-test-00-robert-boulter/logits.f32", .{fixture});
    defer allocator.free(logits_path);
    const st = try std.Io.Dir.cwd().statFile(io, logits_path, .{});
    const vocab = st.size / (64 * @sizeOf(f32));
    try testing.expectEqual(@as(u64, 248320), vocab);
    try testing.expectEqual(@as(u64, 0), st.size % (64 * @sizeOf(f32)));
}

test "kld: prompt sources parse from a fixture dir, a text dir and a jsonl file" {
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &path_buf);
    const root = path_buf[0..root_len];

    try tmp.dir.createDirPath(io, "texts");
    {
        var d = try tmp.dir.openDir(io, "texts", .{});
        defer d.close(io);
        const names = [_][]const u8{ "b_second.txt", "a_first.txt", "skipme.md" };
        const bodies = [_][]const u8{ "second body", "first body", "not a prompt" };
        for (names, bodies) |n, b| {
            var f = try d.createFile(io, n, .{});
            defer f.close(io);
            var wb: [256]u8 = undefined;
            var w = f.writer(io, &wb);
            try w.interface.writeAll(b);
            try w.interface.flush();
        }
    }
    {
        var f = try tmp.dir.createFile(io, "prompts.jsonl", .{});
        defer f.close(io);
        var wb: [512]u8 = undefined;
        var w = f.writer(io, &wb);
        try w.interface.writeAll(
            \\{"id":"one","prompt":"prompt one"}
            \\
            \\{"id":"two","prompt":"prompt two"}
            \\{"id":"ids","prompt_ids":[3,1,2]}
        );
        try w.interface.flush();
    }

    const texts_dir = try std.fmt.allocPrint(allocator, "{s}/texts", .{root});
    defer allocator.free(texts_dir);
    try testing.expectEqual(SourceKind.text_dir, try classifySource(io, texts_dir));
    var from_dir = try loadPrompts(allocator, io, texts_dir, 0);
    defer from_dir.deinit();
    try testing.expectEqual(@as(usize, 2), from_dir.items.len);
    try testing.expectEqualStrings("a_first", from_dir.items[0].id);
    try testing.expectEqualStrings("first body", from_dir.items[0].text);
    try testing.expectEqualStrings("b_second", from_dir.items[1].id);

    const jsonl = try std.fmt.allocPrint(allocator, "{s}/prompts.jsonl", .{root});
    defer allocator.free(jsonl);
    try testing.expectEqual(SourceKind.jsonl, try classifySource(io, jsonl));
    var from_jsonl = try loadPrompts(allocator, io, jsonl, 0);
    defer from_jsonl.deinit();
    try testing.expectEqual(@as(usize, 3), from_jsonl.items.len);
    try testing.expectEqualStrings("one", from_jsonl.items[0].id);
    try testing.expectEqualStrings("prompt two", from_jsonl.items[1].text);
    try testing.expect(from_jsonl.items[1].ids == null);
    // Token ids are taken as given: no template, no tokenizer.
    try testing.expectEqualSlices(u32, &.{ 3, 1, 2 }, from_jsonl.items[2].ids.?);
    try testing.expectEqualStrings("", from_jsonl.items[2].text);

    var limited = try loadPrompts(allocator, io, jsonl, 1);
    defer limited.deinit();
    try testing.expectEqual(@as(usize, 1), limited.items.len);

    var fixture_buf: [std.fs.max_path_bytes]u8 = undefined;
    if (teacherFixture(&fixture_buf, io)) |fixture| {
        try testing.expectEqual(SourceKind.fixture, try classifySource(io, fixture));
        var from_fixture = try loadPrompts(allocator, io, fixture, 2);
        defer from_fixture.deinit();
        try testing.expectEqual(@as(usize, 2), from_fixture.items.len);
        try testing.expectEqualStrings("wikitext2-test-00-robert-boulter", from_fixture.items[0].id);
        try testing.expect(std.mem.startsWith(u8, from_fixture.items[0].text, "= Robert Boulter ="));
    }
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
        "--expert-cache-gb",
        "40",
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
    try testing.expectEqual(@as(u64, 40) * 1_000_000_000, capture.expert_cache_bytes);
    try testing.expect(!capture.enable_mtp);

    const compare = try parseArgs(&.{ "compare", "--model", "/models/pack", "--fixture", "/fixtures/teacher", "--json", "/tmp/out.json" });
    try testing.expectEqual(Command.compare, compare.command);
    try testing.expectEqualStrings("/fixtures/teacher", compare.fixture);
    try testing.expectEqualStrings("/tmp/out.json", compare.json_out);
    try testing.expect(!compare.enable_mtp);
    try testing.expectEqual(@as(u32, 64), compare.tokens);
    const with_mtp = try parseArgs(&.{ "compare", "--model", "/models/pack", "--fixture", "/fixtures/teacher", "--mtp" });
    try testing.expect(with_mtp.enable_mtp);

    const routed = try parseArgs(&.{ "compare", "--model", "/m", "--fixture", "/f", "--expert-pick-tolerance", "0.3", "--wired-margin-gib", "5" });
    try testing.expectEqual(@as(f32, 0.3), routed.pick_tolerance);
    try testing.expectEqual(@as(u64, 5) << 30, routed.wired_margin_bytes);
    try testing.expectError(error.BadFlagValue, parseArgs(&.{ "compare", "--model", "/m", "--fixture", "/f", "--expert-pick-tolerance", "0.7" }));
    try testing.expectError(error.BadFlagValue, parseArgs(&.{ "compare", "--model", "/m", "--fixture", "/f", "--wired-margin-gib", "1" }));

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
    try testing.expectError(error.MissingModel, parseArgs(&.{ "capture", "--prompts", "/p", "--out", "/o" }));
    try testing.expectError(error.MissingPrompts, parseArgs(&.{ "capture", "--model", "/m", "--out", "/o" }));
    try testing.expectError(error.MissingOut, parseArgs(&.{ "capture", "--model", "/m", "--prompts", "/p" }));
    try testing.expectError(error.MissingFixture, parseArgs(&.{ "compare", "--model", "/m" }));
    try testing.expect((try parseArgs(&.{ "capture", "--help" })).help);
}

test "kld: an unflagged capture keeps a full-width teacher KV, never the serving kv8 default" {
    const capture = try parseArgs(&.{ "capture", "--model", "/m", "--prompts", "/p", "--out", "/o" });
    try testing.expectEqual(transformer_mod.KVQuantConfig.dense, capture.kv_quant_config);
    try testing.expect(transformer_mod.KVQuantConfig.engine_default.isQuant());
    try testing.expectEqualStrings("bf16", kvCacheFormat(capture.kv_quant_config));
}

test "kld: the recorded strict NLL is the teacher's own log-softmax, as the capture wrote it" {
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var fixture_buf: [std.fs.max_path_bytes]u8 = undefined;
    const fixture = teacherFixture(&fixture_buf, io) orelse return error.SkipZigTest;

    var base = try readBaseline(allocator, io, fixture);
    defer base.deinit();
    try testing.expectEqual(@as(usize, 343), base.prompts[0].prompt_tokens);

    const vocab: usize = 248320;
    const row = try allocator.alloc(f32, vocab);
    defer allocator.free(row);
    for (base.prompts[0..3]) |fp| {
        try testing.expectEqual(@as(usize, 64), fp.generated_tokens);
        const gen_path = try std.fmt.allocPrint(allocator, "{s}/{s}/generated_tokens.txt", .{ fixture, fp.dir });
        defer allocator.free(gen_path);
        const generated = try readIdList(allocator, io, gen_path);
        defer allocator.free(generated);

        const logits_path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}/logits.f32", .{ fixture, fp.dir }, 0);
        defer allocator.free(logits_path);
        const fd = std.c.open(logits_path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        try testing.expect(fd >= 0);
        defer _ = std.c.close(fd);

        var nll_sum: f64 = 0;
        for (generated, 0..) |token, position| {
            try expert_stream_mod.readExact(fd, std.mem.sliceAsBytes(row), position * vocab * @sizeOf(f32));
            try testing.expectEqual(token, argmaxOf(row));
            const self_score = try scoreRow(row, row, token);
            try testing.expectApproxEqAbs(@as(f64, 0), self_score.kld, 1e-12);
            try testing.expect(self_score.top1);
            try testing.expectApproxEqAbs(self_score.nll, rowNll(row, token), 1e-12);
            nll_sum += self_score.nll;
        }
        const mean = nll_sum / @as(f64, @floatFromInt(generated.len));
        try testing.expectApproxEqAbs(fp.strict_nll_mean, mean, 1e-8);
        try testing.expectApproxEqAbs(fp.strict_perplexity, @exp(mean), 1e-7);
    }
}

test "kld: kvCacheFormat names the effective KV width" {
    try testing.expectEqualStrings("bf16", kvCacheFormat(transformer_mod.KVQuantConfig.dense));
    try testing.expectEqualStrings("affine4", kvCacheFormat(transformer_mod.KVQuantConfig.affine(4)));
    try testing.expectEqualStrings("affine8", kvCacheFormat(transformer_mod.KVQuantConfig.affine(8)));
}

test "kld: a prompt directory is named by index and id" {
    const allocator = testing.allocator;
    const plain = try sanitizeDirName(allocator, "wikitext2-test-00-robert-boulter");
    defer allocator.free(plain);
    try testing.expectEqualStrings("wikitext2-test-00-robert-boulter", plain);
    const messy = try sanitizeDirName(allocator, "a b/c.d:e");
    defer allocator.free(messy);
    try testing.expectEqualStrings("a_b_c_d_e", messy);
}

test "kld: the first EOS position bounds the scored span, and no EOS means the whole span" {
    const eos = [_]u32{ 248044, 248046 };
    try std.testing.expectEqual(@as(?usize, 2), firstEosPosition(&[_]u32{ 5, 6, 248046, 7 }, &eos));
    try std.testing.expectEqual(@as(?usize, 0), firstEosPosition(&[_]u32{ 248044, 1 }, &eos));
    try std.testing.expectEqual(@as(?usize, null), firstEosPosition(&[_]u32{ 1, 2, 3 }, &eos));
    try std.testing.expectEqual(@as(?usize, null), firstEosPosition(&[_]u32{}, &eos));
}

test "kld records the budget that shaped the load, never the raw flag" {
    const GiB: u64 = 1 << 30;
    var moe = model_mod.ModelConfig{ .model_type = "qwen3_5_moe" };
    try testing.expectEqual(@as(u64, 0), capturedSsdBudgetGb(&moe, 60 * GiB));
    moe.expert_ssd_budget_bytes = 60 * GiB;
    try testing.expectEqual(@as(u64, 0), capturedSsdBudgetGb(&moe, 60 * GiB));

    var q4 = model_mod.ModelConfig{ .model_type = "qwen4_exp", .expert_streaming = true };
    q4.expert_ssd_budget_bytes = 60 * GiB;
    try testing.expectEqual(@as(u64, 60), capturedSsdBudgetGb(&q4, 0));
    q4.expert_ssd_budget_bytes = 0;
    try testing.expectEqual(@as(u64, 48), capturedSsdBudgetGb(&q4, 48 * GiB));
}

fn hiddenCaptureWidth(config: *const model_mod.ModelConfig) usize {
    // Block boundaries precede the final mixer and retain all HC streams.
    const streams: usize = if (std.mem.eql(u8, config.model_type, "qwen4_exp"))
        @max(config.hc_count, 1)
    else
        1;
    return @as(usize, config.hidden_size) * streams;
}

test "kld hidden capture keeps the complete Qwen4 residual stream" {
    const q4 = model_mod.ModelConfig{ .model_type = "qwen4_exp", .hidden_size = 2560, .hc_count = 4 };
    const mimo = model_mod.ModelConfig{ .model_type = "mimo_v2", .hidden_size = 4096 };
    try testing.expectEqual(@as(usize, 10240), hiddenCaptureWidth(&q4));
    try testing.expectEqual(@as(usize, 4096), hiddenCaptureWidth(&mimo));
}

const TinyMimo = struct {
    const hidden: usize = 128;
    const vocab: usize = 8;
    const experts: usize = 4;
    const layers: usize = 2;
    const qkv_rows: usize = 384; // (8 heads + 8 kv + 8 v) * 16
    const tiles: usize = hidden / 16;
    const trellis_n: usize = 40; // halfwords per 16x16 tile at k 2.5
};

const TinyTensor = struct {
    key: []const u8,
    dtype: []const u8,
    shape: []const u64,
    bytes: []const u8,
};

fn tinyBf16(a: std.mem.Allocator, count: usize, seed: u64) ![]u8 {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const out = try a.alloc(u8, count * 2);
    for (0..count) |i| {
        const v: f32 = r.float(f32) - 0.5;
        const bits: u32 = @bitCast(v);
        std.mem.writeInt(u16, out[i * 2 ..][0..2], @truncate(bits >> 16), .little);
    }
    return out;
}

/// e4m3 codes in [0.5, 0.94] with alternating signs — every byte a finite value,
/// so a pack that reads them raw still forwards instead of dying on a NaN.
fn tinyFp8(a: std.mem.Allocator, count: usize, seed: u64) ![]u8 {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const out = try a.alloc(u8, count);
    for (out) |*b| b.* = (if (r.boolean()) @as(u8, 0xB0) else @as(u8, 0x30)) | r.uintLessThan(u8, 8);
    return out;
}

/// f16 axis scales of magnitude [0.5, 1) with random signs.
fn tinyF16Axis(a: std.mem.Allocator, count: usize, seed: u64) ![]u8 {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const out = try a.alloc(u8, count * 2);
    for (0..count) |i| {
        const bits: u16 = (if (r.boolean()) @as(u16, 0x8000) else 0) | 0x3800 | r.uintLessThan(u16, 0x400);
        std.mem.writeInt(u16, out[i * 2 ..][0..2], bits, .little);
    }
    return out;
}

fn tinyU32(a: std.mem.Allocator, count: usize, seed: u64) ![]u8 {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const out = try a.alloc(u8, count * 4);
    for (0..count) |i| std.mem.writeInt(u32, out[i * 4 ..][0..4], r.int(u32), .little);
    return out;
}

fn tinyF32(a: std.mem.Allocator, values: []const f32) ![]u8 {
    const out = try a.alloc(u8, values.len * 4);
    for (values, 0..) |v, i| std.mem.writeInt(u32, out[i * 4 ..][0..4], @bitCast(v), .little);
    return out;
}

fn writeTinyShard(io: std.Io, a: std.mem.Allocator, dir: std.Io.Dir, tensors: []const TinyTensor) !void {
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(a);
    try header.append(a, '{');
    var payload: usize = 0;
    for (tensors, 0..) |t, i| {
        if (i != 0) try header.append(a, ',');
        try header.print(a, "\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ t.key, t.dtype });
        for (t.shape, 0..) |dim, j| {
            if (j != 0) try header.append(a, ',');
            try header.print(a, "{d}", .{dim});
        }
        try header.print(a, "],\"data_offsets\":[{d},{d}]}}", .{ payload, payload + t.bytes.len });
        payload += t.bytes.len;
    }
    try header.append(a, '}');

    const file_bytes = try a.alloc(u8, 8 + header.items.len + payload);
    defer a.free(file_bytes);
    std.mem.writeInt(u64, file_bytes[0..8], header.items.len, .little);
    @memcpy(file_bytes[8..][0..header.items.len], header.items);
    var at: usize = 8 + header.items.len;
    for (tensors) |t| {
        @memcpy(file_bytes[at..][0..t.bytes.len], t.bytes);
        at += t.bytes.len;
    }
    try dir.writeFile(io, .{ .sub_path = "model-00001.safetensors", .data = file_bytes });

    var index: std.ArrayList(u8) = .empty;
    defer index.deinit(a);
    try index.appendSlice(a, "{\"weight_map\":{");
    for (tensors, 0..) |t, i| {
        if (i != 0) try index.append(a, ',');
        try index.print(a, "\"{s}\":\"model-00001.safetensors\"", .{t.key});
    }
    try index.appendSlice(a, "}}");
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
}

/// A two-layer mimo_v2 pack shaped like a converted MiMo: the FP8 source trunk
/// (layer 0 dense) beside resident EXL3 routed banks (layer 1). `expert_seed`
/// is the only thing that differs between two arms.
fn writeTinyMimoResidentPack(io: std.Io, a: std.mem.Allocator, dir: std.Io.Dir, expert_seed: u64) !void {
    return writeTinyMimoPackWith(io, a, dir, expert_seed, &.{});
}

/// `stored_affine` names `.weight` tensors the pack stores 8-bit g64 affine
/// (MLX's own packer over the same bf16), the way a converted pack carries them.
fn writeTinyMimoPackWith(io: std.Io, a: std.mem.Allocator, dir: std.Io.Dir, expert_seed: u64, stored_affine: []const []const u8) !void {
    const H = TinyMimo.hidden;
    try dir.writeFile(io, .{ .sub_path = "config.json", .data = try std.mem.concat(a, u8, &.{
        \\{"model_type":"mimo_v2","vocab_size":8,"hidden_size":128,
        \\ "num_hidden_layers":2,"intermediate_size":128,
        \\ "moe_intermediate_size":128,"n_routed_experts":4,
        \\ "num_experts_per_tok":2,"n_group":1,"topk_group":1,
        \\ "num_attention_heads":8,"num_key_value_heads":8,"head_dim":16,
        \\ "v_head_dim":16,"swa_num_attention_heads":8,
        \\ "swa_num_key_value_heads":8,"swa_head_dim":16,"swa_v_head_dim":16,
        \\ "hybrid_layer_pattern":[0,0],"moe_layer_freq":[0,1],
        \\ "attention_projection_layout":"fused_qkv",
        \\ "add_swa_attention_sink_bias":false,
        \\ "add_full_attention_sink_bias":false,
        \\ "expert_quant":{"format":"exl3","k":2.5,"codebook":"mcg"}
        ,
        "}",
    }) });
    try dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = "{}" });
    try dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data =
        \\{"pre_tokenizer":{"type":"ByteLevel"},"model":{"type":"BPE",
        \\ "vocab":{"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7},"merges":[]}}
    });

    var tensors: std.ArrayList(TinyTensor) = .empty;
    defer tensors.deinit(a);
    const vec_shape = try a.dupe(u64, &[_]u64{H});
    try tensors.append(a, .{ .key = "model.embed_tokens.weight", .dtype = "BF16", .shape = try a.dupe(u64, &[_]u64{ TinyMimo.vocab, H }), .bytes = try tinyBf16(a, TinyMimo.vocab * H, 11) });
    try tensors.append(a, .{ .key = "lm_head.weight", .dtype = "BF16", .shape = try a.dupe(u64, &[_]u64{ TinyMimo.vocab, H }), .bytes = try tinyBf16(a, TinyMimo.vocab * H, 12) });
    try tensors.append(a, .{ .key = "model.norm.weight", .dtype = "BF16", .shape = vec_shape, .bytes = try tinyBf16(a, H, 13) });

    for (0..TinyMimo.layers) |li| {
        const p = try std.fmt.allocPrint(a, "model.layers.{d}", .{li});
        const seed: u64 = 100 + li;
        try tensors.append(a, .{ .key = try std.fmt.allocPrint(a, "{s}.input_layernorm.weight", .{p}), .dtype = "BF16", .shape = vec_shape, .bytes = try tinyBf16(a, H, seed) });
        try tensors.append(a, .{ .key = try std.fmt.allocPrint(a, "{s}.post_attention_layernorm.weight", .{p}), .dtype = "BF16", .shape = vec_shape, .bytes = try tinyBf16(a, H, seed + 1) });
        try tensors.append(a, .{ .key = try std.fmt.allocPrint(a, "{s}.self_attn.qkv_proj.weight", .{p}), .dtype = "F8_E4M3", .shape = try a.dupe(u64, &[_]u64{ TinyMimo.qkv_rows, H }), .bytes = try tinyFp8(a, TinyMimo.qkv_rows * H, seed + 2) });
        try tensors.append(a, .{ .key = try std.fmt.allocPrint(a, "{s}.self_attn.qkv_proj.weight_scale_inv", .{p}), .dtype = "F32", .shape = try a.dupe(u64, &[_]u64{ 4, 1 }), .bytes = try tinyF32(a, &[_]f32{ 1.0, 0.75, 1.25, 0.5 }) });
        try tensors.append(a, .{ .key = try std.fmt.allocPrint(a, "{s}.self_attn.o_proj.weight", .{p}), .dtype = "BF16", .shape = try a.dupe(u64, &[_]u64{ H, H }), .bytes = try tinyBf16(a, H * H, seed + 3) });

        if (li == 0) {
            for ([_][]const u8{ "gate", "up", "down" }, 0..) |proj, j| {
                try tensors.append(a, .{ .key = try std.fmt.allocPrint(a, "{s}.mlp.{s}_proj.weight", .{ p, proj }), .dtype = "F8_E4M3", .shape = try a.dupe(u64, &[_]u64{ H, H }), .bytes = try tinyFp8(a, H * H, seed + 10 + j) });
                try tensors.append(a, .{ .key = try std.fmt.allocPrint(a, "{s}.mlp.{s}_proj.weight_scale_inv", .{ p, proj }), .dtype = "F32", .shape = try a.dupe(u64, &[_]u64{ 1, 1 }), .bytes = try tinyF32(a, &[_]f32{1.0}) });
            }
            continue;
        }
        try tensors.append(a, .{ .key = try std.fmt.allocPrint(a, "{s}.mlp.gate.weight", .{p}), .dtype = "BF16", .shape = try a.dupe(u64, &[_]u64{ TinyMimo.experts, H }), .bytes = try tinyBf16(a, TinyMimo.experts * H, seed + 20) });
        try tensors.append(a, .{ .key = try std.fmt.allocPrint(a, "{s}.mlp.gate.e_score_correction_bias", .{p}), .dtype = "F32", .shape = try a.dupe(u64, &[_]u64{TinyMimo.experts}), .bytes = try tinyF32(a, &[_]f32{ 0.0, 0.1, -0.1, 0.05 }) });
        for ([_][]const u8{ "gate", "up", "down" }, 0..) |proj, j| {
            const E = TinyMimo.experts;
            const halfwords = E * TinyMimo.tiles * TinyMimo.tiles * TinyMimo.trellis_n;
            try tensors.append(a, .{ .key = try std.fmt.allocPrint(a, "{s}.mlp.switch_mlp.{s}_proj.trellis", .{ p, proj }), .dtype = "U16", .shape = try a.dupe(u64, &[_]u64{ E, TinyMimo.tiles, TinyMimo.tiles, TinyMimo.trellis_n }), .bytes = try tinyU32(a, halfwords / 2, expert_seed * 1000 + j) });
            try tensors.append(a, .{ .key = try std.fmt.allocPrint(a, "{s}.mlp.switch_mlp.{s}_proj.suh", .{ p, proj }), .dtype = "F16", .shape = try a.dupe(u64, &[_]u64{ E, H }), .bytes = try tinyF16Axis(a, E * H, expert_seed * 1000 + 10 + j) });
            try tensors.append(a, .{ .key = try std.fmt.allocPrint(a, "{s}.mlp.switch_mlp.{s}_proj.svh", .{ p, proj }), .dtype = "F16", .shape = try a.dupe(u64, &[_]u64{ E, H }), .bytes = try tinyF16Axis(a, E * H, expert_seed * 1000 + 20 + j) });
        }
    }
    var stored: std.ArrayList(TinyTensor) = .empty;
    defer stored.deinit(a);
    for (tensors.items) |t| {
        for (stored_affine) |name| {
            if (std.mem.eql(u8, name, t.key)) break;
        } else {
            try stored.append(a, t);
            continue;
        }
        try appendAffine8(a, &stored, t);
    }
    try writeTinyShard(io, a, dir, stored.items);
}

fn appendAffine8(a: std.mem.Allocator, out: *std.ArrayList(TinyTensor), t: TinyTensor) !void {
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    const dense = mlx.mlx_array_new_data(t.bytes.ptr, &[_]c_int{ @intCast(t.shape[0]), @intCast(t.shape[1]) }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(dense);
    var parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    try mlx.check(mlx.mlx_quantize(&parts, dense, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", .{ .ctx = null }, s));
    const base = t.key[0 .. t.key.len - ".weight".len];
    for ([_][]const u8{ "weight", "scales", "biases" }, [_][]const u8{ "U32", "BF16", "BF16" }, 0..) |part, dtype, i| {
        var arr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(arr);
        try mlx.check(mlx.mlx_vector_array_get(&arr, parts, i));
        try mlx.check(mlx.mlx_array_eval(arr));
        const shape = mlx.getShape(arr);
        const bytes = mlx.mlx_array_size(arr) * @as(usize, if (i == 0) 4 else 2);
        const src: [*]const u8 = if (i == 0)
            @ptrCast(mlx.mlx_array_data_uint32(arr) orelse return error.KldLogitsUnreadable)
        else
            @ptrCast(mlx.mlx_array_data_bfloat16(arr) orelse return error.KldLogitsUnreadable);
        try out.append(a, .{
            .key = try std.fmt.allocPrint(a, "{s}.{s}", .{ base, part }),
            .dtype = dtype,
            .shape = try a.dupe(u64, &[_]u64{ @intCast(shape[0]), @intCast(shape[1]) }),
            .bytes = try a.dupe(u8, src[0..bytes]),
        });
    }
}

test "kld: a resident mimo_v2 load takes the source trunk and its logits follow the routed experts" {
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var metal: bool = false;
    mlx.check(mlx.mlx_metal_is_available(&metal)) catch return error.SkipZigTest;
    if (!metal) return error.SkipZigTest;

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const ids = [_]u32{ 1, 3, 5, 2, 7, 0 };
    var rows: [2][TinyMimo.vocab]f32 = undefined;
    for ([_][]const u8{ "pack_a", "pack_b" }, 0..) |sub, arm| {
        try tmp.dir.createDirPath(io, sub);
        var dir = try tmp.dir.openDir(io, sub, .{});
        defer dir.close(io);
        try writeTinyMimoResidentPack(io, arena, dir, @as(u64, arm) + 1);
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path_len = try dir.realPath(io, &path_buf);

        const loaded = try loadModel(io, allocator, .{ .model_dir = path_buf[0..path_len] });
        defer loaded.deinit();
        // The served QKV is the source's rank-local FP8 bytes, which only the
        // FP8 kernels read; no generic split of it is bound.
        const qkv = loaded.weights.get("model.layers.0.self_attn.qkv_proj.weight") orelse return error.MissingWeight;
        try testing.expectEqual(mlx.mlx_dtype.uint8, mlx.mlx_array_dtype(qkv));
        try testing.expect(loaded.weights.get("model.layers.0.self_attn.q_proj.weight") == null);

        var ctx = loaded.xfm.defaultCtx();
        const logits = try forwardPrompt(allocator, loaded, &ctx, &ids);
        defer _ = mlx.mlx_array_free(logits);
        try readLastRow(&loaded.xfm, logits, &rows[arm]);
        for (rows[arm]) |v| try testing.expect(std.math.isFinite(v));
    }
    try testing.expect(!std.mem.eql(f32, &rows[0], &rows[1]));
}

test "kld: an imatrix capture records every mimo_v2 o_proj input and the lm_head input the forward read" {
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var metal: bool = false;
    mlx.check(mlx.mlx_metal_is_available(&metal)) catch return error.SkipZigTest;
    if (!metal) return error.SkipZigTest;

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    try tmp.dir.createDirPath(io, "pack");
    var dir = try tmp.dir.openDir(io, "pack", .{});
    defer dir.close(io);
    try writeTinyMimoResidentPack(io, arena, dir, 1);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try dir.realPath(io, &path_buf);

    const loaded = try loadModel(io, allocator, .{ .model_dir = path_buf[0..path_len] });
    defer loaded.deinit();
    const imatrix = @import("imatrix.zig");
    const col = try imatrix.Collector.init(allocator, loaded.xfm.s, "/dev/null", TinyMimo.layers, TinyMimo.experts, .mimo_v2);
    defer col.deinit();
    loaded.xfm.imatrix = col;
    defer loaded.xfm.imatrix = null;

    const ids = [_]u32{ 1, 3, 5, 2, 7, 0 };
    var hidden = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hidden);
    var ctx = loaded.xfm.defaultCtx();
    ctx.capture_hidden_all = &hidden;
    const logits = try forwardPrompt(allocator, loaded, &ctx, &ids);
    defer _ = mlx.mlx_array_free(logits);
    try mlx.check(mlx.mlx_array_eval(logits));

    for (col.o_proj) |d| try testing.expectEqual(@as(u64, ids.len), d.rows);
    try testing.expectEqual(@as(u64, ids.len), col.lm_head.rows);

    // lm_head's statistic is the per-channel sum of squares of the final
    // normed hidden rows, the same rows the logits were projected from.
    var h32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(h32);
    try mlx.check(mlx.mlx_astype(&h32, hidden, .float32, loaded.xfm.s));
    try mlx.check(mlx.mlx_array_eval(h32));
    try mlx.check(mlx.mlx_array_eval(col.lm_head.acc));
    const h = mlx.mlx_array_data_float32(h32) orelse return error.KldLogitsUnreadable;
    const acc = mlx.mlx_array_data_float32(col.lm_head.acc) orelse return error.KldLogitsUnreadable;
    for (0..TinyMimo.hidden) |c| {
        var want: f32 = 0;
        for (0..ids.len) |r| want += h[r * TinyMimo.hidden + c] * h[r * TinyMimo.hidden + c];
        try testing.expectApproxEqRel(want, acc[c], 1e-5);
    }
}

fn bf16HostBytes(a: std.mem.Allocator, s: mlx.mlx_stream, arr: mlx.mlx_array) ![]u8 {
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(arr));
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, arr, false, s));
    try mlx.check(mlx.mlx_array_eval(c));
    const p = mlx.mlx_array_data_bfloat16(c) orelse return error.KldLogitsUnreadable;
    return a.dupe(u8, std.mem.sliceAsBytes(p[0..mlx.mlx_array_size(c)]));
}

fn bf16At(bytes: []const u8, i: usize) f32 {
    return @bitCast(@as(u32, std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little)) << 16);
}

test "kld: a hidden capture appends every block boundary of each prompt and leaves the teacher bit-identical" {
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var metal: bool = false;
    mlx.check(mlx.mlx_metal_is_available(&metal)) catch return error.SkipZigTest;
    if (!metal) return error.SkipZigTest;

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    try tmp.dir.createDirPath(io, "pack");
    {
        var dir = try tmp.dir.openDir(io, "pack", .{});
        defer dir.close(io);
        try writeTinyMimoResidentPack(io, arena, dir, 1);
    }
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = try arena.dupe(u8, path_buf[0..try tmp.dir.realPath(io, &path_buf)]);
    try tmp.dir.writeFile(io, .{ .sub_path = "windows.jsonl", .data =
        \\{"id":"w0","prompt_ids":[1,3,5,2,7,0]}
        \\{"id":"w1","prompt_ids":[4,4,6,1,2]}
    });
    const windows = [_][]const u32{ &.{ 1, 3, 5, 2, 7, 0 }, &.{ 4, 4, 6, 1, 2 } };
    const total = windows[0].len + windows[1].len;

    const pack = try std.fmt.allocPrint(arena, "{s}/pack", .{root});
    const loaded = try loadModel(io, allocator, .{ .model_dir = pack });
    defer loaded.deinit();
    var quiet: Out = .{ .silent = true };
    const base: Options = .{
        .model_dir = pack,
        .prompts = try std.fmt.allocPrint(arena, "{s}/windows.jsonl", .{root}),
        .tokens = 3,
        .no_template = true,
    };
    var plain = base;
    plain.out_dir = try std.fmt.allocPrint(arena, "{s}/plain", .{root});
    try runCapture(io, allocator, loaded, plain, &quiet);
    // Off: nothing is written.
    if (tmp.dir.statFile(io, "hidden", .{})) |_| return error.TestUnexpectedResult else |_| {}

    var armed = base;
    armed.out_dir = try std.fmt.allocPrint(arena, "{s}/armed", .{root});
    armed.hidden_out = try std.fmt.allocPrint(arena, "{s}/hidden", .{root});
    try runCapture(io, allocator, loaded, armed, &quiet);

    // The teacher is untouched: every logits row and greedy token, byte for byte.
    for ([_][]const u8{ "prompts/00_w0", "prompts/01_w1" }) |sub| {
        for ([_][]const u8{ "logits.f32", "generated_tokens.txt" }) |name| {
            const a = try tmp.dir.readFileAlloc(io, try std.fmt.allocPrint(arena, "plain/{s}/{s}", .{ sub, name }), arena, .limited(1 << 20));
            const b = try tmp.dir.readFileAlloc(io, try std.fmt.allocPrint(arena, "armed/{s}/{s}", .{ sub, name }), arena, .limited(1 << 20));
            try testing.expect(a.len > 0);
            try testing.expectEqualSlices(u8, a, b);
        }
    }

    const H = TinyMimo.hidden;
    const toks = try tmp.dir.readFileAlloc(io, "hidden/tokens.bin", arena, .limited(1 << 20));
    try testing.expectEqual(total * 4, toks.len);
    var files: [TinyMimo.layers + 1][]u8 = undefined;
    for (&files, 0..) |*f, b| {
        f.* = try tmp.dir.readFileAlloc(io, try std.fmt.allocPrint(arena, "hidden/boundary-{d:0>2}.bin", .{b}), arena, .limited(1 << 20));
        try testing.expectEqual(total * H * 2, f.len);
    }

    const embed = try tinyBf16(arena, TinyMimo.vocab * H, 11);
    const norm_w = try tinyBf16(arena, H, 13);
    const s = loaded.xfm.s;
    var row: usize = 0;
    for (windows) |w| {
        for (w, row..) |t, r| {
            try testing.expectEqual(t, std.mem.readInt(u32, toks[r * 4 ..][0..4], .little));
            // Boundary 0 is the token's own embedding row.
            try testing.expectEqualSlices(u8, embed[t * H * 2 ..][0 .. H * 2], files[0][r * H * 2 ..][0 .. H * 2]);
        }

        // Every boundary holds the forward's own residual at that depth.
        try loaded.xfm.resetCache();
        const layer_ids = [_]u32{ 0, 1 };
        var outs = [_]mlx.mlx_array{ mlx.mlx_array_new(), mlx.mlx_array_new() };
        defer for (outs) |o| {
            _ = mlx.mlx_array_free(o);
        };
        var input = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(input);
        var final = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(final);
        var cl: transformer_mod.CaptureLayers = .{ .ids = &layer_ids, .out = &outs, .input = &input };
        var ctx = loaded.xfm.defaultCtx();
        ctx.capture_layers = &cl;
        ctx.capture_hidden_all = &final;
        const logits = try forwardPrompt(allocator, loaded, &ctx, w);
        defer _ = mlx.mlx_array_free(logits);
        try mlx.check(mlx.mlx_array_eval(logits));
        for ([_]mlx.mlx_array{ input, outs[0], outs[1] }, 0..) |arr, b| {
            const got = try bf16HostBytes(arena, s, arr);
            try testing.expectEqualSlices(u8, got, files[b][row * H * 2 ..][0 .. w.len * H * 2]);
        }

        // The last boundary is the residual the final norm reads.
        const fin = try bf16HostBytes(arena, s, final);
        const last = files[TinyMimo.layers][row * H * 2 ..][0 .. w.len * H * 2];
        for (0..w.len) |r| {
            var ms: f64 = 0;
            for (0..H) |c| ms += @as(f64, bf16At(last, r * H + c)) * bf16At(last, r * H + c);
            const inv = 1.0 / @sqrt(ms / @as(f64, H) + loaded.config.rms_norm_eps);
            for (0..H) |c| {
                const want: f64 = bf16At(last, r * H + c) * inv * bf16At(norm_w, c);
                try testing.expectApproxEqAbs(want, @as(f64, bf16At(fin, r * H + c)), 1e-2 + 1e-2 * @abs(want));
            }
        }
        row += w.len;
    }
}

test "kld: a pack storing o_proj, lm_head and embed_tokens affine serves them packed beside its bf16 twin" {
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var metal: bool = false;
    mlx.check(mlx.mlx_metal_is_available(&metal)) catch return error.SkipZigTest;
    if (!metal) return error.SkipZigTest;

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const o_proj = [_][]const u8{ "model.layers.0.self_attn.o_proj.weight", "model.layers.1.self_attn.o_proj.weight" };
    const tables = [_][]const u8{ "lm_head.weight", "model.embed_tokens.weight" };
    const arms = [_][]const []const u8{ &.{}, &o_proj, &tables, tables[1..] };
    const ids = [_]u32{ 1, 3, 5, 2, 7, 0 };
    var rows: [arms.len][TinyMimo.vocab]f32 = undefined;
    for (arms, 0..) |stored, arm| {
        const sub = ([_][]const u8{ "source", "o_proj", "tables", "embed" })[arm];
        try tmp.dir.createDirPath(io, sub);
        var dir = try tmp.dir.openDir(io, sub, .{});
        defer dir.close(io);
        try writeTinyMimoPackWith(io, arena, dir, 1, stored);
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path_len = try dir.realPath(io, &path_buf);

        const loaded = try loadModel(io, allocator, .{ .model_dir = path_buf[0..path_len] });
        defer loaded.deinit();
        for (o_proj ++ tables) |name| {
            const packed_here = for (stored) |s| {
                if (std.mem.eql(u8, s, name)) break true;
            } else false;
            const w = loaded.weights.get(name) orelse return error.MissingWeight;
            try testing.expectEqual(if (packed_here) mlx.mlx_dtype.uint32 else mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(w));
        }

        var ctx = loaded.xfm.defaultCtx();
        const logits = try forwardPrompt(allocator, loaded, &ctx, &ids);
        defer _ = mlx.mlx_array_free(logits);
        try readLastRow(&loaded.xfm, logits, &rows[arm]);
        for (rows[arm]) |v| try testing.expect(std.math.isFinite(v));
    }
    // 8-bit weights move the logits only by their own rounding; a bf16 head
    // beside a packed embedding keeps its own (absent) scales.
    for (1..arms.len) |arm| {
        try testing.expect(!std.mem.eql(f32, &rows[0], &rows[arm]));
        var dot: f64 = 0;
        var na: f64 = 0;
        var nb: f64 = 0;
        for (rows[0], rows[arm]) |a, b| {
            dot += a * b;
            na += a * a;
            nb += b * b;
        }
        try testing.expect(dot / @sqrt(na * nb) > 0.98);
    }
}

test "kld standard4 bundled texts identifiers and limits" {
    const a = testing.allocator;
    var full = try loadPrompts(a, testing.io, "standard4", 0);
    defer full.deinit();
    try testing.expectEqual(@as(usize, 4), full.items.len);
    const ids = [_][]const u8{ "code-python-topological-sort", "code-zig-byte-reader", "prose-water-cycle", "prose-navigation" };
    const digests = [_][]const u8{ "524e3d4590d0935d3a8e40f29bb75f502eba2f39eca0684d3de36a04ddf6fd30", "af2dc27bd24d4643ecaf0fd849ade54d56a8eb42c6434fcd61b47db2cd8e1d35", "a54cbe6b15eb667b5d87cfff019d5430dd84239d119a9cbf2dfac79d09efe00e", "882dd467214d1b696fa5ac04dd1277a3bb7720d2c88ce7e6a278008696adaf09" };
    for (full.items, ids, digests) |prompt, id, digest| {
        try testing.expectEqualStrings(id, prompt.id);
        try testing.expect(prompt.ids == null);
        var hash: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(prompt.text, &hash, .{});
        try testing.expectEqualStrings(digest, &std.fmt.bytesToHex(hash, .lower));
    }
    var limited = try loadPrompts(a, testing.io, "standard4", 2);
    defer limited.deinit();
    try testing.expectEqual(@as(usize, 2), limited.items.len);
    for (limited.items, full.items[0..2]) |actual, expected| {
        try testing.expectEqualStrings(expected.id, actual.id);
        try testing.expectEqualStrings(expected.text, actual.text);
    }
    var above = try loadPrompts(a, testing.io, "standard4", 5);
    defer above.deinit();
    try testing.expectEqual(@as(usize, 4), above.items.len);
}

test {
    _ = @import("glm5_kld_capture.zig");
}

fn jsonNumber(value: std.json.Value) !f64 {
    return switch (value) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => error.NotANumber,
    };
}

test "kld: a GLM pack captures its BF16-latent reference in prompt chunks and compare scores kv8 against it" {
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var metal: bool = false;
    mlx.check(mlx.mlx_metal_is_available(&metal)) catch return error.SkipZigTest;
    if (!metal) return error.SkipZigTest;
    const glm = @import("glm5_forward.zig");
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = "{}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data =
        \\{"pre_tokenizer":{"type":"ByteLevel"},"model":{"type":"BPE",
        \\ "vocab":{"a":0,"b":1,"c":2,"d":3},"merges":[]}}
    });
    try tmp.dir.writeFile(io, .{ .sub_path = "prompts.jsonl", .data =
        \\{"id":"p0","prompt_ids":[1,3,0,2,2,1,3]}
        \\{"id":"p1","prompt_ids":[2,0,1]}
    });
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = try arena.dupe(u8, path_buf[0..try tmp.dir.realPath(io, &path_buf)]);

    var weights = model_mod.Weights.init(allocator);
    defer weights.deinit();
    var cfg = try glm.completeFixture(&weights);
    cfg.max_position_embeddings = 64;
    var loaded = Loaded{ .allocator = allocator, .io = io, .config = cfg, .tok = try tokenizer_mod.loadTokenizer(io, allocator, root), .chat_config = undefined, .weights = undefined, .xfm = undefined };
    defer loaded.tok.deinit();
    loaded.xfm = try transformer_mod.Transformer.init(io, allocator, cfg, &weights);
    defer loaded.xfm.deinit();

    // Chunked prompt forwards continue one request exactly as the native model run in the same chunks.
    var reference = try glm.Model.load(allocator, cfg, &weights, loaded.xfm.s);
    defer reference.deinit();
    var request = try glm.Request.init(allocator, cfg.num_hidden_layers);
    defer request.deinit();
    request.dense_prefill = true;
    request.prefill_async = true;
    const ids = [_]u32{ 1, 3, 0, 2, 2, 1, 3 };
    try loaded.xfm.resetCache();
    var ctx = loaded.xfm.defaultCtx();
    const chunked = try forwardPromptChunks(allocator, &loaded, &ctx, &ids, 3);
    defer _ = mlx.mlx_array_free(chunked);
    var expected = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(expected);
    for ([_][2]usize{ .{ 0, 3 }, .{ 3, 6 }, .{ 6, 7 } }) |r| {
        const part = mlx.mlx_array_new_data(ids[r[0]..r[1]].ptr, &[_]c_int{ 1, @intCast(r[1] - r[0]) }, 2, .uint32);
        defer _ = mlx.mlx_array_free(part);
        _ = mlx.mlx_array_free(expected);
        expected = try reference.forwardLast(&request, part, true);
    }
    var same = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(same);
    try mlx.check(mlx.mlx_array_equal(&same, chunked, expected, true, loaded.xfm.s));
    var equal = false;
    try mlx.check(mlx.mlx_array_item_bool(&equal, same));
    try testing.expect(equal);
    try testing.expectEqual(@as(usize, 7), loaded.xfm.glm5_request.?.offset);

    var quiet: Out = .{ .silent = true };
    const fixture = try std.fmt.allocPrint(arena, "{s}/reference", .{root});
    try runCapture(io, allocator, &loaded, .{ .model_dir = root, .prompts = try std.fmt.allocPrint(arena, "{s}/prompts.jsonl", .{root}), .out_dir = fixture, .tokens = 4, .no_template = true }, &quiet);
    for ([_]transformer_mod.KVQuantConfig{ .dense, .affine(8) }) |kv| {
        // As `loadModel` applies the flag to the loaded cache.
        if (kv.isQuant()) try loaded.xfm.cache.reinit(cfg.num_hidden_layers, kv);
        const json_path = try std.fmt.allocPrint(arena, "{s}/compare-{s}.json", .{ root, kvCacheFormat(kv) });
        try runCompare(io, allocator, &loaded, .{ .command = .compare, .model_dir = root, .fixture = fixture, .json_out = json_path, .kv_quant_config = kv }, &quiet);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, json_path, arena, .limited(1 << 20));
        const parsed = try std.json.parseFromSlice(std.json.Value, arena, text, .{});
        const kld = try jsonNumber(parsed.value.object.get("mean_kld").?);
        try testing.expectEqual(@as(f64, 8), try jsonNumber(parsed.value.object.get("positions").?));
        try testing.expect(std.math.isFinite(kld));
        if (!kv.isQuant()) try testing.expectEqual(@as(f64, 0), kld);
        try testing.expectEqual(@as(u8, if (kv.isQuant()) 8 else 0), loaded.xfm.glm5_request.?.latent_bits);
    }
}
