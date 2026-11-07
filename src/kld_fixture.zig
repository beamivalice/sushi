//! The fixture a `sushi kld capture` writes and `compare` reads: prompt sources, the `baseline.json` record, the
//! per-prompt files, the teacher's logits rows, and the scoring of a student row against one. Links no MLX, so a
//! host-tier test or a plugin's parity harness reads and scores a fixture without loading a model.
const std = @import("std");
const io_util = @import("io_util.zig");
const testing = std.testing;

pub const SCHEMA = "mlx-serve-kld-baseline-v1";
pub const COMPARE_SCHEMA = "mlx-serve-kld-compare-v1";
pub const TOOL = "sushi";

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

pub fn rowNll(row: []const f32, chosen: u32) f64 {
    var max = -std.math.inf(f64);
    for (row) |v| {
        if (v > max) max = v;
    }
    var sum: f64 = 0;
    for (row) |v| sum += @exp(@as(f64, v) - max);
    return max + @log(sum) - @as(f64, row[chosen]);
}

/// The token row `step` is scored against: the forced continuation's, else the greedy one.
pub fn pickToken(force: ?[]const u32, step: usize, row: []const f32) u32 {
    return if (force) |f| f[step] else argmaxOf(row);
}

pub fn argmaxOf(row: []const f32) u32 {
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

/// `ids` set: the prompt is these token ids as given (no template, no tokenizer). `force` set (needs `ids`): capture
/// scores this continuation teacher-forced instead of generating one.
pub const Prompt = struct { id: []u8, text: []u8, ids: ?[]u32 = null, force: ?[]u32 = null };

pub const PromptList = struct {
    allocator: std.mem.Allocator,
    items: []Prompt = &.{},

    pub fn deinit(self: *PromptList) void {
        for (self.items) |p| {
            self.allocator.free(p.id);
            self.allocator.free(p.text);
            if (p.ids) |ids| self.allocator.free(ids);
            if (p.force) |f| self.allocator.free(f);
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
            if (p.force) |f| allocator.free(f);
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
                const force = if (obj.get("continuation_ids")) |v| try tokenIdsField(allocator, v) else null;
                errdefer if (force) |f| allocator.free(f);
                if (force != null and ids == null) return error.BadPromptJsonl;
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
                try items.append(allocator, .{ .id = id, .text = text, .ids = ids, .force = force });
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

pub fn writeJsonString(w: *std.Io.Writer, s: []const u8) !void {
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
    try writeIdList(&fw.interface, ids);
    try fw.interface.flush();
}

/// The text of a fixture's `prompt_tokens.txt` / `generated_tokens.txt`: comma-separated decimal ids.
pub fn writeIdList(w: *std.Io.Writer, ids: []const u32) !void {
    for (ids, 0..) |id, i| {
        if (i > 0) try w.writeAll(",");
        try w.print("{d}", .{id});
    }
}

pub fn sanitizeDirName(allocator: std.mem.Allocator, id: []const u8) ![]u8 {
    const out = try allocator.alloc(u8, id.len);
    for (id, 0..) |c, i| {
        out[i] = switch (c) {
            'a'...'z', 'A'...'Z', '0'...'9', '-', '_' => c,
            else => '_',
        };
    }
    return out;
}

pub fn writeAllFd(fd: std.c.fd_t, bytes: []const u8) !void {
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

pub fn firstEosPosition(generated: []const u32, eos: []const u32) ?usize {
    for (generated, 0..) |t, i| {
        for (eos) |e| if (t == e) return i;
    }
    return null;
}

/// A prompt's `prompt_tokens.txt` and `generated_tokens.txt`, both non-empty.
pub const PromptTokens = struct {
    prompt: []u32,
    generated: []u32,

    pub fn deinit(self: PromptTokens, allocator: std.mem.Allocator) void {
        allocator.free(self.prompt);
        allocator.free(self.generated);
    }
};

pub fn readPromptTokens(allocator: std.mem.Allocator, io: std.Io, fixture: []const u8, prompt_dir: []const u8) !PromptTokens {
    const prompt_path = try std.fmt.allocPrint(allocator, "{s}/{s}/prompt_tokens.txt", .{ fixture, prompt_dir });
    defer allocator.free(prompt_path);
    const prompt = try readIdList(allocator, io, prompt_path);
    errdefer allocator.free(prompt);
    const gen_path = try std.fmt.allocPrint(allocator, "{s}/{s}/generated_tokens.txt", .{ fixture, prompt_dir });
    defer allocator.free(gen_path);
    const generated = try readIdList(allocator, io, gen_path);
    errdefer allocator.free(generated);
    if (prompt.len == 0 or generated.len == 0) return error.EmptyFixturePrompt;
    return .{ .prompt = prompt, .generated = generated };
}

/// The positions a prompt is scored over up to its first EOS, inclusive; no EOS scores the whole span.
pub fn positionsToEos(generated: []const u32, eos: []const u32) usize {
    return if (firstEosPosition(generated, eos)) |e| e + 1 else generated.len;
}

/// A prompt's `logits.f32`: one full-vocabulary f32 row per generated token, read by position.
pub const TeacherLogits = struct {
    fd: std.c.fd_t,
    vocab: usize,

    /// The vocabulary is the file size over `rows` rows of f32.
    pub fn open(allocator: std.mem.Allocator, fixture: []const u8, prompt_dir: []const u8, rows: usize) !TeacherLogits {
        const path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}/logits.f32", .{ fixture, prompt_dir }, 0);
        defer allocator.free(path);
        const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.TeacherLogitsMissing;
        errdefer _ = std.c.close(fd);
        var st: std.c.Stat = undefined;
        if (std.c.fstat(fd, &st) != 0 or st.size < 0) return error.TeacherLogitsStatFailed;
        const size: u64 = @intCast(st.size);
        const row_bytes = rows * @sizeOf(f32);
        if (row_bytes == 0 or size % row_bytes != 0) return error.TeacherLogitsSizeMismatch;
        return .{ .fd = fd, .vocab = @intCast(size / row_bytes) };
    }

    pub fn close(self: TeacherLogits) void {
        _ = std.c.close(self.fd);
    }

    pub fn readRow(self: TeacherLogits, position: usize, dst: []f32) !void {
        std.debug.assert(dst.len == self.vocab);
        const bytes = std.mem.sliceAsBytes(dst);
        try io_util.readExact(self.fd, bytes, position * self.vocab * @sizeOf(f32));
    }
};

test "kld fixture: scoreRow on identical logits is zero divergence and the plain NLL" {
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

test "kld fixture: scoreRow reports raw-logit cosine for identical, orthogonal, opposite, and scaling rows" {
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

test "kld fixture: scoreRow rejects nonfinite and zero-norm cosine rows" {
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

test "kld fixture: scoreRow matches the closed form on two 8-vocab rows" {
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

test "kld fixture: scoreRow is log-sum-exp stable on large logits" {
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

test "kld fixture: a captured fixture round-trips through the reader byte for byte" {
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

test "kld fixture: the real teacher fixture reads back at full width" {
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

test "kld fixture: prompt sources parse from a fixture dir, a text dir and a jsonl file" {
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

test "kld fixture: the recorded strict NLL is the teacher's own log-softmax, as the capture wrote it" {
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var fixture_buf: [std.fs.max_path_bytes]u8 = undefined;
    const fixture = teacherFixture(&fixture_buf, io) orelse return error.SkipZigTest;

    var base = try readBaseline(allocator, io, fixture);
    defer base.deinit();
    try testing.expectEqual(@as(usize, 343), base.prompts[0].prompt_tokens);

    const row = try allocator.alloc(f32, 248320);
    defer allocator.free(row);
    for (base.prompts[0..3]) |fp| {
        try testing.expectEqual(@as(usize, 64), fp.generated_tokens);
        const tokens = try readPromptTokens(allocator, io, fixture, fp.dir);
        defer tokens.deinit(allocator);
        const generated = tokens.generated;

        const teacher = try TeacherLogits.open(allocator, fixture, fp.dir, generated.len);
        defer teacher.close();
        try testing.expectEqual(row.len, teacher.vocab);

        var nll_sum: f64 = 0;
        for (generated, 0..) |token, position| {
            try teacher.readRow(position, row);
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

test "kld fixture: a prompt directory is named by index and id" {
    const allocator = testing.allocator;
    const plain = try sanitizeDirName(allocator, "wikitext2-test-00-robert-boulter");
    defer allocator.free(plain);
    try testing.expectEqualStrings("wikitext2-test-00-robert-boulter", plain);
    const messy = try sanitizeDirName(allocator, "a b/c.d:e");
    defer allocator.free(messy);
    try testing.expectEqualStrings("a_b_c_d_e", messy);
}

test "kld fixture: the first EOS position bounds the scored span, and no EOS means the whole span" {
    const eos = [_]u32{ 248044, 248046 };
    try std.testing.expectEqual(@as(?usize, 2), firstEosPosition(&[_]u32{ 5, 6, 248046, 7 }, &eos));
    try std.testing.expectEqual(@as(?usize, 0), firstEosPosition(&[_]u32{ 248044, 1 }, &eos));
    try std.testing.expectEqual(@as(?usize, null), firstEosPosition(&[_]u32{ 1, 2, 3 }, &eos));
    try std.testing.expectEqual(@as(?usize, null), firstEosPosition(&[_]u32{}, &eos));
}

test "kld fixture: standard4 bundled texts identifiers and limits" {
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

test "kld fixture: a jsonl line may force its continuation, and only after prompt_ids" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeTextFile(io, tmp.dir, "ok.jsonl", "{\"id\":\"t\",\"prompt_ids\":[1,2,3],\"continuation_ids\":[4,5]}\n");
    try writeTextFile(io, tmp.dir, "bad.jsonl", "{\"id\":\"t\",\"prompt\":\"x\",\"continuation_ids\":[4,5]}\n");
    var path_buf: [512]u8 = undefined;
    const root = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const ok_path = try std.fmt.allocPrint(a, "{s}/ok.jsonl", .{root});
    defer a.free(ok_path);
    var list = try loadPrompts(a, io, ok_path, 0);
    defer list.deinit();
    try testing.expectEqualSlices(u32, &.{ 4, 5 }, list.items[0].force.?);
    const bad_path = try std.fmt.allocPrint(a, "{s}/bad.jsonl", .{root});
    defer a.free(bad_path);
    try testing.expectError(error.BadPromptJsonl, loadPrompts(a, io, bad_path, 0));
}

test "kld fixture: pickToken scores the forced token and falls back to the argmax" {
    const row = [_]f32{ 0.1, 2.0, 0.3 };
    try testing.expectEqual(@as(u32, 1), pickToken(null, 0, &row));
    try testing.expectEqual(@as(u32, 2), pickToken(&.{ 2, 0 }, 0, &row));
    try testing.expectEqual(@as(u32, 0), pickToken(&.{ 2, 0 }, 1, &row));
}

/// The 60x64 teacher fixture under the models root, or null when it is absent.
fn teacherFixture(buf: []u8, io: std.Io) ?[]const u8 {
    const path = @import("test_models.zig").packPath(buf, "kld-teacher/Qwen3.8-Flash-Next-wikitext2-60x64") catch return null;
    var dir = std.Io.Dir.openDirAbsolute(io, path, .{}) catch return null;
    defer dir.close(io);
    _ = dir.statFile(io, "baseline.json", .{}) catch return null;
    return path;
}

test "kld fixture: scoreRow on known rows is the hand-computed divergence, NLL and top-1" {
    const teacher = [_]f32{ @log(2.0), 0 };
    const model = [_]f32{ @log(3.0), 0 };
    const hit = try scoreRow(&teacher, &model, 0);
    try testing.expectApproxEqAbs(2.0 / 3.0 * @log(8.0 / 9.0) + 1.0 / 3.0 * @log(4.0 / 3.0), hit.kld, 1e-6);
    try testing.expectApproxEqAbs(@log(4.0 / 3.0), hit.nll, 1e-6);
    try testing.expect(hit.top1);
    try testing.expectApproxEqAbs(@as(f64, 1), hit.cosine_similarity, 1e-12);
    const miss = try scoreRow(&teacher, &model, 1);
    try testing.expectApproxEqAbs(@log(4.0), miss.nll, 1e-6);
    try testing.expect(!miss.top1);
    try testing.expectError(error.KldShapeMismatch, scoreRow(&teacher, model[0..1], 0));
    try testing.expectError(error.KldShapeMismatch, scoreRow(&teacher, &model, 2));
}

test "kld fixture: the first EOS cuts the scored span after the EOS token itself" {
    const eos = [_]u32{ 7, 9 };
    try testing.expectEqual(@as(usize, 3), positionsToEos(&.{ 1, 2, 9, 4, 5 }, &eos));
    try testing.expectEqual(@as(usize, 1), positionsToEos(&.{ 7, 2, 3 }, &eos));
    try testing.expectEqual(@as(usize, 4), positionsToEos(&.{ 1, 2, 3, 4 }, &eos));
}

test "kld fixture: a tiny fixture reads back through the sources, the token lists and the teacher rows" {
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [512]u8 = undefined;
    const root = path_buf[0..try tmp.dir.realPath(io, &path_buf)];

    const generated = [_]u32{ 2, 0, 1 };
    const rows = [3][4]f32{ .{ 0, 1, 5, 2 }, .{ 6, 1, 0, 2 }, .{ 1, 4, 0, 2 } };
    var writer = try PromptWriter.begin(allocator, io, root, 0, "tiny one");
    defer writer.deinit();
    for (rows, generated) |row, token| try writer.appendRow(&row, token);
    const record = try writer.finish("tiny prompt", "tiny prompt", &.{ 9, 8 }, &generated);
    defer record.deinit(allocator);
    try writeBaseline(allocator, io, root, .{
        .label = "tiny",
        .model = "none",
        .run = "tiny",
        .kv_cache_format = "bf16",
        .inference_profile = "greedy",
        .prompt_set = "tiny",
        .ssd_budget_gb = 0,
        .tokens_per_prompt = 3,
        .top_k = 1,
        .elapsed_secs = 0,
    }, &.{record});

    try testing.expectEqual(SourceKind.fixture, try classifySource(io, root));
    var base = try readBaseline(allocator, io, root);
    defer base.deinit();
    try testing.expectEqual(@as(usize, 1), base.prompts.len);
    try testing.expectEqualStrings("prompts/00_tiny_one", base.prompts[0].dir);

    const tokens = try readPromptTokens(allocator, io, root, base.prompts[0].dir);
    defer tokens.deinit(allocator);
    try testing.expectEqualSlices(u32, &.{ 9, 8 }, tokens.prompt);
    try testing.expectEqualSlices(u32, &generated, tokens.generated);

    const teacher = try TeacherLogits.open(allocator, root, base.prompts[0].dir, generated.len);
    defer teacher.close();
    try testing.expectEqual(@as(usize, 4), teacher.vocab);
    var row: [4]f32 = undefined;
    for (rows, 0..) |want, position| {
        try teacher.readRow(position, &row);
        try testing.expectEqualSlices(f32, &want, &row);
        const self_score = try scoreRow(&row, &row, generated[position]);
        try testing.expectEqual(argmaxOf(&row) == generated[position], self_score.top1);
    }
    try testing.expectError(error.TeacherLogitsSizeMismatch, TeacherLogits.open(allocator, root, base.prompts[0].dir, 5));
    try testing.expectError(error.TeacherLogitsMissing, TeacherLogits.open(allocator, root, "prompts/99_none", 3));
}
