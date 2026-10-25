const std = @import("std");
const host = @import("mlx_host");
const format = @import("sushi_exl3").format;
const calib = @import("calib.zig");
const capture_mod = @import("capture.zig");
const driver = @import("driver.zig");
const encode = @import("encode.zig");
const gpu_factor = @import("gpu_factor.zig");
const source = @import("source.zig");

pub const version = "sushi-encoder-1";
const capture_budget_bytes: u64 = 8 << 30;

pub const ArgError = error{
    MissingSubcommand,
    UnknownSubcommand,
    UnknownFlag,
    MissingFlagValue,
    BadFlagValue,
    MissingModel,
    MissingOut,
    CalibFileMissing,
};

pub const Options = struct {
    help: bool = false,
    model: []const u8 = "",
    out: []const u8 = "",
    experts_n: u32 = 40,
    expert_codebook: format.Codebook = .mcg,
    window: format.Window = .w15,
    calib: []const u8 = "",
    calib_tokens: u32 = 65536,
    resuming: bool = false,
    trunk_bits: ?u32 = null,
    trunk_group: ?u32 = null,
    batch: u32 = 16,
    factor_precision: gpu_factor.Precision = .f64,
};

pub const Diag = struct { flag: []const u8 = "" };

const Flag = enum { model, out, experts_k, expert_codebook, window, calib, calib_tokens, trunk_bits, trunk_group, batch, factor_precision };

pub const value_flags = [_]struct { []const u8, Flag }{
    .{ "--model", .model },
    .{ "--out", .out },
    .{ "--experts-k", .experts_k },
    .{ "--expert-codebook", .expert_codebook },
    .{ "--window", .window },
    .{ "--calib", .calib },
    .{ "--calib-tokens", .calib_tokens },
    .{ "--trunk-bits", .trunk_bits },
    .{ "--trunk-group", .trunk_group },
    .{ "--batch", .batch },
    .{ "--factor-precision", .factor_precision },
};

pub const switch_flags = [_][]const u8{ "--resume", "--help", "-h" };

pub const min_calib_tokens: u32 = 2048;
pub const max_batch: u32 = 64;

pub const USAGE =
    \\usage:
    \\  sushi convert exl3 --model <dir> --out <dir> [options]
    \\
    \\  Converts a Qwen3.8-Flash-Next BF16 checkpoint into a Sushi pack: the
    \\  routed experts as EXL3, the trunk as MLX affine. Any other architecture
    \\  is refused by name.
    \\
    \\options:
    \\  --model <dir>         source checkpoint directory
    \\  --out <dir>           new pack directory; refused when it already holds
    \\                        files, except with --resume
    \\  --experts-k <K>       routed-expert rate in bits per weight, 1.5 to 8 in
    \\                        steps of 0.125 (default 2.5)
    \\  --expert-codebook <name>
    \\                        the routed experts' codebook: exl3-mcg (default) or
    \\                        exl3-mul1
    \\  --window <n>          trellis search window, 12 to 16 (default 15)
    \\  --calib <file>        calibration text (default: a built-in 64k-token mix
    \\                        of prose and code)
    \\  --calib-tokens <n>    calibration tokens, at least 2048 (default 65536)
    \\  --trunk-bits <n>      trunk affine bits: 2, 3, 4, 5, 6 or 8 (default: the
    \\                        width the model's packs serve)
    \\  --trunk-group <n>     trunk affine group size: 32, 64 or 128 (default: the
    \\                        group the model's packs serve)
    \\  --batch <n>           experts quantized together per bank, 1 to 64 (default
    \\                        16); the pack bytes do not depend on it
    \\  --factor-precision <p>
    \\                        Hessian factor: 64 on the CPU in float64 (default), or 32
    \\                        on the GPU in float32 (faster, changes the pack bytes)
    \\  --resume              continue an interrupted conversion in --out
    \\  -h, --help            print this text
    \\
;

/// The CLI spelling of a codebook is `exl3-<name>`; the pack stamps keep the bare name.
fn parseCodec(text: []const u8) ?format.Codebook {
    if (!std.mem.startsWith(u8, text, "exl3-")) return null;
    const cb = format.Codebook.fromName(text["exl3-".len..]) orelse return null;
    return if (encode.searchable(cb)) cb else null;
}

fn parseExpertsN(text: []const u8) ?u32 {
    const k = std.fmt.parseFloat(f64, text) catch return null;
    if (!std.math.isFinite(k) or k < 1.5 or k > 8.0) return null;
    const n = k * 16.0;
    const rounded = @round(n);
    if (@abs(n - rounded) > 1e-6) return null;
    const as_int: u32 = @intFromFloat(rounded);
    if (as_int % 2 != 0) return null;
    return as_int;
}

fn parseTrunkBits(text: []const u8) ?u32 {
    const v = std.fmt.parseInt(u32, text, 10) catch return null;
    return switch (v) {
        2, 3, 4, 5, 6, 8 => v,
        else => null,
    };
}

fn parseTrunkGroup(text: []const u8) ?u32 {
    const v = std.fmt.parseInt(u32, text, 10) catch return null;
    return switch (v) {
        32, 64, 128 => v,
        else => null,
    };
}

fn parseFactorPrecision(text: []const u8) ?gpu_factor.Precision {
    if (std.mem.eql(u8, text, "64")) return .f64;
    if (std.mem.eql(u8, text, "32")) return .f32;
    return null;
}

pub fn parseArgs(args: []const []const u8, diag: ?*Diag) ArgError!Options {
    if (args.len == 0) return error.MissingSubcommand;
    if (std.mem.eql(u8, args[0], "--help") or std.mem.eql(u8, args[0], "-h")) return .{ .help = true };
    if (!std.mem.eql(u8, args[0], "exl3")) return error.UnknownSubcommand;
    var o = Options{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--help") or std.mem.eql(u8, a, "-h")) return .{ .help = true };
        if (std.mem.eql(u8, a, "--resume")) {
            o.resuming = true;
            continue;
        }
        var flag: ?Flag = null;
        for (value_flags) |row| if (std.mem.eql(u8, a, row[0])) {
            flag = row[1];
        };
        const f = flag orelse {
            if (diag) |d| d.flag = a;
            return error.UnknownFlag;
        };
        if (i + 1 >= args.len) {
            if (diag) |d| d.flag = a;
            return error.MissingFlagValue;
        }
        i += 1;
        const v = args[i];
        const bad = switch (f) {
            .model => blk: {
                o.model = v;
                break :blk false;
            },
            .out => blk: {
                o.out = v;
                break :blk false;
            },
            .experts_k => blk: {
                o.experts_n = parseExpertsN(v) orelse break :blk true;
                break :blk false;
            },
            .expert_codebook => blk: {
                o.expert_codebook = parseCodec(v) orelse break :blk true;
                break :blk false;
            },
            .window => blk: {
                const w = std.fmt.parseInt(i64, v, 10) catch break :blk true;
                if (w < 12 or w > 16) break :blk true;
                o.window = format.Window.fromBits(w) orelse break :blk true;
                break :blk false;
            },
            .calib => blk: {
                o.calib = v;
                break :blk false;
            },
            .calib_tokens => blk: {
                o.calib_tokens = std.fmt.parseInt(u32, v, 10) catch break :blk true;
                if (o.calib_tokens < min_calib_tokens) break :blk true;
                break :blk false;
            },
            .trunk_bits => blk: {
                o.trunk_bits = parseTrunkBits(v) orelse break :blk true;
                break :blk false;
            },
            .trunk_group => blk: {
                o.trunk_group = parseTrunkGroup(v) orelse break :blk true;
                break :blk false;
            },
            .factor_precision => blk: {
                o.factor_precision = parseFactorPrecision(v) orelse break :blk true;
                break :blk false;
            },
            .batch => blk: {
                o.batch = std.fmt.parseInt(u32, v, 10) catch break :blk true;
                if (o.batch < 1 or o.batch > max_batch) break :blk true;
                break :blk false;
            },
        };
        if (bad) {
            if (diag) |d| d.flag = a;
            return error.BadFlagValue;
        }
    }
    if (o.model.len == 0) return error.MissingModel;
    if (o.out.len == 0) return error.MissingOut;
    return o;
}

fn writeStdout(bytes: []const u8) void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = std.c.write(1, bytes[done..].ptr, bytes.len - done);
        if (n <= 0) return;
        done += @intCast(n);
    }
}

fn printOut(comptime fmt: []const u8, args: anytype) void {
    var buf: [8192]u8 = undefined;
    const text = std.fmt.bufPrint(&buf, fmt, args) catch return;
    writeStdout(text);
}

fn progress(_: ?*anyopaque, line: []const u8) void {
    host.log.info("[convert] {s}\n", .{line});
}

pub fn cmdConvert(allocator: std.mem.Allocator, io: std.Io, args: []const []const u8) !void {
    var diag = Diag{};
    const opts = parseArgs(args, &diag) catch |err| {
        if (diag.flag.len != 0) printOut("sushi convert: {s} ({s})\n\n{s}", .{ @errorName(err), diag.flag, USAGE }) else printOut("sushi convert: {s}\n\n{s}", .{ @errorName(err), USAGE });
        return err;
    };
    if (opts.help) {
        printOut("{s}", .{USAGE});
        return;
    }
    const cwd = std.Io.Dir.cwd();
    if (opts.calib.len != 0) cwd.access(io, opts.calib, .{}) catch {
        printOut("sushi convert: CalibFileMissing ({s})\n", .{opts.calib});
        return error.CalibFileMissing;
    };
    var model_dir = cwd.openDir(io, opts.model, .{ .iterate = true }) catch |err| {
        printOut("sushi convert: cannot open --model {s}: {s}\n", .{ opts.model, @errorName(err) });
        return err;
    };
    defer model_dir.close(io);
    const arch = source.identifySourceDir(allocator, io, model_dir) catch |err| {
        printOut("sushi convert: {s}: {s} is not a checkpoint this tool converts (Qwen3.8-Flash-Next, MiMo-V2.6-Flash, GLM-5.3-Flash)\n", .{ @errorName(err), opts.model });
        return err;
    };
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const abs_len = try model_dir.realPath(io, &path_buf);
    const model_abs = path_buf[0..abs_len];

    var windows: ?calib.Windows = null;
    defer if (windows) |*w| w.deinit();
    var src: ?source.SafetensorsSource = null;
    defer if (src) |*v| v.deinit();
    var capture: ?capture_mod.Capture = null;
    defer if (capture) |*c| c.deinit();
    if (arch == .qwen4_exp) {
        var tokenizer = try host.tokenizer.loadTokenizerAny(io, allocator, model_abs);
        defer tokenizer.deinit();
        if (opts.calib.len != 0) {
            windows = try calib.fileWindows(allocator, io, &tokenizer, opts.calib, opts.calib_tokens);
        } else {
            const home = std.c.getenv("HOME") orelse return error.HomeNotSet;
            var cache_buf: [std.fs.max_path_bytes]u8 = undefined;
            const cache = try std.fmt.bufPrint(&cache_buf, "{s}/.sushi/cache", .{std.mem.span(home)});
            windows = try calib.defaultWindows(allocator, io, &tokenizer, cache, opts.calib_tokens, .{ .fetch = calib.curlFetch });
        }
        src = try source.SafetensorsSource.open(allocator, io, model_dir);
        capture = capture_mod.Capture.init(allocator, io, host.mlx.gpuStream(), &windows.?, &src.?, capture_budget_bytes);
        progress(null, "calibration windows ready");
    }

    const summary = driver.run(allocator, io, host.mlx.gpuStream(), .{
        .model_dir = model_abs,
        .out_dir = opts.out,
        .rate = .{ .n = opts.experts_n },
        .dec = .{ .codebook = opts.expert_codebook, .window = opts.window },
        .trunk_bits = opts.trunk_bits,
        .trunk_group = opts.trunk_group,
        .batch = opts.batch,
        .factor_precision = opts.factor_precision,
        .calib_sha256 = if (windows) |*w| &w.sha256 else "none",
        .calib_tokens = if (windows) |*w| w.tokens() else 0,
        .resuming = opts.resuming,
        .version = version,
    }, if (capture) |*c| c.calibrator() else null, .{ .write = progress }) catch |err| {
        printOut("sushi convert: {s} ({s})\n", .{ @errorName(err), @tagName(arch) });
        return err;
    };
    printOut("converted {d} layers ({d} reused){s} into {s}\n", .{ summary.layers_written + summary.layers_reused, summary.layers_reused, if (summary.mtp_written) " and the MTP head" else "", opts.out });
}

fn expectParse(args: []const []const u8) !Options {
    return parseArgs(args, null);
}

test "convert cli: defaults and every flag parse" {
    const t = std.testing;
    const o = try expectParse(&.{ "exl3", "--model", "m", "--out", "o" });
    try t.expectEqual(@as(u32, 40), o.experts_n);
    try t.expectEqual(format.Codebook.mcg, o.expert_codebook);
    try t.expectEqual(format.Window.w15, o.window);
    try t.expectEqual(@as(u32, 65536), o.calib_tokens);
    try t.expect(!o.resuming);
    try t.expect(o.trunk_bits == null and o.trunk_group == null);
    try t.expectEqual(@as(u32, 16), o.batch);
    try t.expectEqual(gpu_factor.Precision.f64, o.factor_precision);

    const all = try expectParse(&.{ "exl3", "--model", "m", "--out", "o", "--experts-k", "1.5", "--expert-codebook", "exl3-mul1", "--window", "12", "--calib", "c.txt", "--calib-tokens", "4096", "--trunk-bits", "6", "--trunk-group", "128", "--batch", "4", "--factor-precision", "32", "--resume" });
    try t.expectEqual(@as(u32, 24), all.experts_n);
    try t.expectEqual(format.Codebook.mul1, all.expert_codebook);
    try t.expectEqual(format.Window.w12, all.window);
    try t.expectEqualStrings("c.txt", all.calib);
    try t.expectEqual(@as(u32, 4096), all.calib_tokens);
    try t.expectEqual(@as(?u32, 6), all.trunk_bits);
    try t.expectEqual(@as(?u32, 128), all.trunk_group);
    try t.expect(all.resuming);
    try t.expectEqual(@as(u32, 4), all.batch);
    try t.expectEqual(gpu_factor.Precision.f32, all.factor_precision);
    try t.expectEqual(gpu_factor.Precision.f64, (try expectParse(&.{ "exl3", "--model", "m", "--out", "o", "--factor-precision", "64" })).factor_precision);
    try t.expectEqual(@as(u32, 128), (try expectParse(&.{ "exl3", "--model", "m", "--out", "o", "--experts-k", "8" })).experts_n);
    try t.expectEqual(@as(u32, 42), (try expectParse(&.{ "exl3", "--model", "m", "--out", "o", "--experts-k", "2.625" })).experts_n);
    try t.expectEqual(format.Window.w16, (try expectParse(&.{ "exl3", "--model", "m", "--out", "o", "--window", "16" })).window);
}

test "convert cli: a bad value, an unknown flag or a missing piece is refused before anything runs" {
    const t = std.testing;
    const base = [_][]const u8{ "exl3", "--model", "m", "--out", "o" };
    const bad_values = [_][2][]const u8{
        .{ "--experts-k", "1.25" },       .{ "--experts-k", "8.125" },   .{ "--experts-k", "2.51" },      .{ "--experts-k", "x" },
        .{ "--experts-k", "nan" },        .{ "--expert-codebook", "tiny" }, .{ "--expert-codebook", "x" }, .{ "--window", "11" },
        .{ "--window", "17" },            .{ "--window", "x" },          .{ "--calib-tokens", "2047" },   .{ "--calib-tokens", "-1" },
        .{ "--trunk-bits", "7" },         .{ "--trunk-bits", "1" },      .{ "--trunk-group", "48" },      .{ "--batch", "0" },
        .{ "--batch", "65" },             .{ "--batch", "x" },           .{ "--factor-precision", "16" }, .{ "--factor-precision", "x" },
        .{ "--factor-precision", "032" }, .{ "--factor-precision", "" },
    };
    for (bad_values) |pair| {
        var diag = Diag{};
        const args = base ++ [_][]const u8{ pair[0], pair[1] };
        try t.expectError(error.BadFlagValue, parseArgs(&args, &diag));
        try t.expectEqualStrings(pair[0], diag.flag);
    }
    var diag = Diag{};
    try t.expectError(error.UnknownFlag, parseArgs(&(base ++ [_][]const u8{"--attn-k"}), &diag));
    try t.expectEqualStrings("--attn-k", diag.flag);
    try t.expectError(error.MissingFlagValue, parseArgs(&(base ++ [_][]const u8{"--window"}), null));
    try t.expectError(error.MissingModel, parseArgs(&.{ "exl3", "--out", "o" }, null));
    try t.expectError(error.MissingOut, parseArgs(&.{ "exl3", "--model", "m" }, null));
    try t.expectError(error.UnknownSubcommand, parseArgs(&.{"gguf"}, null));
    try t.expectError(error.MissingSubcommand, parseArgs(&.{}, null));
}

test "convert cli: codec values carry exl3-, and the old flag name is gone" {
    const t = std.testing;
    const base = [_][]const u8{ "exl3", "--model", "m", "--out", "o" };
    try t.expectEqual(format.Codebook.mcg, (try expectParse(&(base ++ [_][]const u8{ "--expert-codebook", "exl3-mcg" }))).expert_codebook);
    try t.expectEqual(format.Codebook.mul1, (try expectParse(&(base ++ [_][]const u8{ "--expert-codebook", "exl3-mul1" }))).expert_codebook);
    for ([_][]const u8{ "mcg", "mul1", "stq-mcg", "stq-mul1", "exl3-std1", "exl3-spec1", "exl3-pair1", "exl3-alpha", "exl3-" }) |bad| {
        var diag = Diag{};
        try t.expectError(error.BadFlagValue, parseArgs(&(base ++ [_][]const u8{ "--expert-codebook", bad }), &diag));
        try t.expectEqualStrings("--expert-codebook", diag.flag);
    }
    var diag = Diag{};
    try t.expectError(error.UnknownFlag, parseArgs(&(base ++ [_][]const u8{ "--codebook", "mcg" }), &diag));
    try t.expectEqualStrings("--codebook", diag.flag);
}

test "convert cli: --help works after convert and after exl3, and lists every flag" {
    const t = std.testing;
    try t.expect((try expectParse(&.{"--help"})).help);
    try t.expect((try expectParse(&.{ "exl3", "--help" })).help);
    try t.expect((try expectParse(&.{ "exl3", "--model", "m", "-h" })).help);
    for (value_flags) |row| try t.expect(std.mem.indexOf(u8, USAGE, row[0]) != null);
    for (switch_flags) |name| try t.expect(std.mem.indexOf(u8, USAGE, name) != null);
}
