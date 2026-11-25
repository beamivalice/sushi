const std = @import("std");
const host = @import("mlx_host");
const format = @import("sushi_exl3").format;
const calib = @import("calib.zig");
const capture_mod = @import("capture.zig");
const driver = @import("driver.zig");
const pool = @import("pool.zig");
const encode = @import("encode.zig");
const codebook_tables = @import("sushi_exl3").codebook_tables;
const rate_plan = @import("rate_plan.zig");
const gpu_factor = @import("gpu_factor.zig");
const source = @import("source.zig");
const dsv41 = @import("dsv41.zig");
const hy4 = @import("hy4.zig");
const trunk = @import("trunk.zig");
const trunk_stq = @import("trunk_stq.zig");
const vision_affine = @import("vision_affine.zig");

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
    RatePlanFileMissing,
    TargetNoneWithBoundaries,
    CodebookTableMismatch,
    TrunkCodebookNeedsStq,
    TrunkKNeedsStq,
    TrunkCodebookUnsupported,
    TrunkCodebookTableUnsupported,
    Pair1WindowNot16,
    Pair1RateUnsupported,
    Pair1DrafterRateUnsupported,
    CalibWindowsMissing,
    CalibSourceConflict,
};

pub const Subcommand = enum { exl3, vision_affine8 };

pub const Options = struct {
    help: bool = false,
    subcommand: Subcommand = .exl3,
    model: []const u8 = "",
    out: []const u8 = "",
    experts_n: u32 = 40,
    expert_codebook: format.Codebook = .mcg,
    expert_codebook_table: ?*const codebook_tables.Table = null,
    /// Null when the command line does not say: `--trunk stq` then takes SPEC1 `spec1-w15`.
    trunk_codebook: ?format.Codebook = null,
    trunk_codebook_table: ?*const codebook_tables.Table = null,
    /// The trunk codec after the defaults and the refusals (`trunkDecode`).
    trunk_decode: format.Decode = (trunk.Stq{}).decode,
    /// With `--trunk stq`, the rate of the trunk banks (`--trunk-k`); `lm_head` stays K6.
    trunk_rate: format.Rate = (trunk.Stq{}).rate,
    trunk_k_given: bool = false,
    /// The drafter's bank rate (`--drafter-k`): Hy4-Flash's DFly2, DeepSeek-V4.1's DSpark trunk; null: the trunk's.
    drafter_rate: ?format.Rate = null,
    /// DeepSeek-V4.1 only: the DSpark stages' routed-expert rate (`--drafter-experts-k`); null: the converter's own.
    drafter_experts_n: ?u32 = null,
    window: format.Window = .w15,
    window_given: bool = false,
    calib: []const u8 = "",
    calib_tokens: u32 = 65536,
    calib_ids: []const u8 = "",
    /// A DeepSeek-V4.1 calibration capture's `calib/` directory.
    capture: []const u8 = "",
    calib_windows: []const u8 = "",
    rate_plan: []const u8 = "",
    trunk_from: []const u8 = "",
    trunk: driver.TrunkMode = .gptq,
    fisher_dir: []const u8 = "",
    target_boundaries: []const u8 = "",
    /// Null when the command line does not say: `--target-boundaries` then implies the compensated target.
    target: ?Target = null,
    resuming: bool = false,
    /// Hy4-Flash only: random trellis codes and RMS-matched scales, no calibration (a dev pack).
    synthetic_codes: bool = false,
    /// Hy4-Flash only: every bank rounded against an identity Hessian (no calibration capture).
    identity_hessian: bool = false,
    trunk_bits: ?u32 = null,
    trunk_group: ?u32 = null,
    batch: u32 = 16,
    factor_precision: gpu_factor.Precision = .f64,
    pin_pass: u32 = encode.default_pin_pass,
    layer_cooldown: u32 = 0,
    cpu_threads: ?u32 = null,
};

pub const Target = enum { none, compensated };

pub const Diag = struct { flag: []const u8 = "" };

const Flag = enum { model, out, experts_k, expert_codebook, expert_codebook_table, trunk_codebook, trunk_codebook_table, trunk_k, drafter_k, drafter_experts_k, window, calib, calib_tokens, calib_ids, calib_windows, capture, rate_plan, trunk, trunk_from, fisher_dir, target, target_boundaries, trunk_bits, trunk_group, batch, factor_precision, pin_pass, layer_cooldown, cpu_threads };

pub const value_flags = [_]struct { []const u8, Flag }{
    .{ "--model", .model },
    .{ "--out", .out },
    .{ "--experts-k", .experts_k },
    .{ "--expert-codebook", .expert_codebook },
    .{ "--expert-codebook-table", .expert_codebook_table },
    .{ "--trunk-codebook", .trunk_codebook },
    .{ "--trunk-codebook-table", .trunk_codebook_table },
    .{ "--trunk-k", .trunk_k },
    .{ "--drafter-k", .drafter_k },
    .{ "--drafter-experts-k", .drafter_experts_k },
    .{ "--window", .window },
    .{ "--calib", .calib },
    .{ "--calib-tokens", .calib_tokens },
    .{ "--calib-ids", .calib_ids },
    .{ "--capture", .capture },
    .{ "--calib-windows", .calib_windows },
    .{ "--rate-plan", .rate_plan },
    .{ "--trunk", .trunk },
    .{ "--trunk-from", .trunk_from },
    .{ "--fisher-dir", .fisher_dir },
    .{ "--target", .target },
    .{ "--target-boundaries", .target_boundaries },
    .{ "--trunk-bits", .trunk_bits },
    .{ "--trunk-group", .trunk_group },
    .{ "--batch", .batch },
    .{ "--factor-precision", .factor_precision },
    .{ "--pin-pass", .pin_pass },
    .{ "--layer-cooldown", .layer_cooldown },
    .{ "--cpu-threads", .cpu_threads },
};

pub const switch_flags = [_][]const u8{ "--resume", "--synthetic-codes", "--identity-hessian", "--help", "-h" };

pub const min_calib_tokens: u32 = 2048;
pub const max_batch: u32 = 64;

pub const USAGE =
    \\usage:
    \\  sushi convert exl3 --model <dir> --out <dir> [options]
    \\  sushi convert vision-affine8 --model <pack> --out <dir>
    \\
    \\  Converts a Qwen3.8-Flash-Next, MiMo-V2.6-Flash or GLM-5.3-Flash source
    \\  checkpoint (BF16, FP8 block or MXFP4) into a Sushi pack: the routed
    \\  experts as EXL3, every other tensor in the format that model's packs
    \\  serve. Any other architecture or format is refused by name.
    \\
    \\options:
    \\  --model <dir>         source checkpoint directory
    \\  --out <dir>           new pack directory; refused when it already holds
    \\                        files, except with --resume
    \\  --experts-k <K>       routed-expert rate in bits per weight, 1.5 to 8 in
    \\                        steps of 0.125 (default 2.5); with --rate-plan, the
    \\                        rate of every layer the plan does not name
    \\  --rate-plan <file>    per-expert gate/up and down rates per layer (JSON);
    \\                        a layer is written as rate groups
    \\  --expert-codebook <name>
    \\                        the routed experts' codebook: exl3-mcg (default),
    \\                        exl3-mul1, stq-std1, stq-spec1 or stq-pair1; stq-pair1
    \\                        decodes two weights from one window, at window 16
    \\                        and an --experts-k of 1.5 to 4 in steps of 0.125
    \\  --expert-codebook-table <name>
    \\                        the named table a stq-std1 or stq-spec1 pack decodes
    \\                        with; stq-pair1's is pair1-w16-v1
    \\  --trunk-codebook <name>
    \\                        with --trunk stq, the trunk and lm_head codec:
    \\                        stq-spec1 (default, the only one the trunk serves)
    \\  --trunk-codebook-table <name>
    \\                        with --trunk stq, the trunk's table (default
    \\                        spec1-w15, the only one the trunk serves)
    \\  --trunk-k <4|5|6>     with --trunk stq, the rate of the trunk banks in bits
    \\                        per weight (default 4; Hy4-Flash: 6, the head and
    \\                        embedding included); lm_head is otherwise always K6
    \\  --drafter-k <4|5|6>   Hy4-Flash and DeepSeek-V4.1 only: the rate of the
    \\                        drafter's banks (DFly2, DSpark's trunk) in bits per
    \\                        weight (default: the trunk's rate)
    \\  --drafter-experts-k <K>
    \\                        DeepSeek-V4.1 only: the rate of DSpark's routed
    \\                        experts, 1.5 to 4 in steps of 0.125 (default 2)
    \\  --window <n>          trellis search window, 12 to 16 (default 15; 16
    \\                        with stq-pair1)
    \\  --calib <file>        calibration text (default: a built-in 64k-token mix
    \\                        of prose and code)
    \\  --calib-tokens <n>    calibration tokens, at least 2048 (default 65536)
    \\  --calib-ids <file>    calibration token ids (little-endian u32), cut into
    \\                        the windows of --calib-windows (JSON lengths)
    \\  --calib-windows <file>
    \\  --capture <dir>       DeepSeek-V4.1 and Hy4-Flash: the calib/ directory of
    \\                        a layer-major calibration capture (rows, routes and
    \\                        trunk sums); the experts default to PAIR1, the
    \\                        trunk is SPEC1 at --trunk-k (default 4; Hy4-Flash:
    \\                        6), the head K6
    \\  --trunk <mode>        how the trunk's linears are rounded: gptq (default)
    \\                        rounds each by GPTQ on its inputs as the quantized
    \\                        model computes them, layer by layer, in the
    \\                        layout of --trunk-from, else at --trunk-bits and
    \\                        --trunk-group; stored takes them from --trunk-from
    \\                        as they are; rtn rounds each to nearest
    \\  --trunk-from <dir>    take every non-expert tensor from this pack (its
    \\                        config's quantization), routers reordered to the
    \\                        rate groups; its width is the pack's, so it
    \\                        refuses --trunk-bits and --trunk-group, and a pack
    \\                        written with a rate plan is refused (not with
    \\                        --trunk stq, which keeps its experts as stored)
    \\  --fisher-dir <dir>    per-layer output Fishers (fisher-LL.npz): routed
    \\                        experts take Fisher-weighted output scales
    \\  --target <t>          none (default) or compensated: routed experts round
    \\                        toward the BF16 model's layer, its inputs computed
    \\                        in the run one boundary at a time
    \\  --target-boundaries <dir>
    \\                        the compensated target with the BF16 model's stored
    \\                        layer inputs over the same calibration
    \\                        (boundary-LL.u16) instead; refused beside
    \\                        --target none
    \\  --trunk-bits <n>      trunk affine bits: 2, 3, 4, 5, 6 or 8 (default: the
    \\                        width the model's packs serve); --trunk gptq rounds
    \\                        2, 4 and 8 only
    \\  --trunk-group <n>     trunk affine group size: 32, 64 or 128 (default: the
    \\                        group the model's packs serve)
    \\  --batch <n>           experts quantized together per bank, 1 to 64 (default
    \\                        16); the pack bytes do not depend on it
    \\  --factor-precision <p>
    \\                        Hessian factor: 64 on the CPU in float64 (default), or 32
    \\                        on the GPU in float32 (faster, changes the pack bytes)
    \\  --pin-pass <A>        positions on each side of the wrap that the search's
    \\                        first pass covers, a multiple of 16 from 16 to 128
    \\                        (default 64; 128 is the full pass); pair1 counts
    \\                        pairs, half as many; changes the pack bytes
    \\  --layer-cooldown <s>  seconds to rest, CPU and GPU idle, after each layer so
    \\                        the chassis cools (default 0, no rest); the pack
    \\                        bytes do not depend on it
    \\  --cpu-threads <n>     cap on the CPU threads the encoder's pools use, at
    \\                        least 1; above the core count it is the core count
    \\                        (default: all cores); the pack bytes do not
    \\                        depend on it
    \\  --synthetic-codes     Hy4-Flash only, a dev pack: every trellis bank gets
    \\                        random codes and scales matched to its source
    \\                        tensor's RMS, with no calibration or search; the
    \\                        pack is the real format and its config says
    \\                        "synthetic": true. Needs --expert-codebook
    \\                        stq-pair1; the trunk, drafter, embedding and head
    \\                        are SPEC1 K6
    \\  --identity-hessian    Hy4-Flash only, a correctness fixture: every bank is
    \\                        rounded by LDLQ and the trellis search against an
    \\                        identity Hessian, with no calibration capture; the
    \\                        config's quant blocks say "hessian": "identity"
    \\  --resume              continue an interrupted conversion in --out
    \\  -h, --help            print this text
    \\
    \\vision-affine8:
    \\  Writes a copy of a Hy4-Flash pack whose vision tower is MLX affine 8
    \\  (model-vision.safetensors, config "vision_quant"). --model is the pack;
    \\  --out a new folder on the same volume: the pack's other files are
    \\  cloned. The pack itself is not changed.
    \\
;

/// The CLI spelling of a codebook is `<family>-<name>`: `exl3-` for the scalar codebooks, `stq-` for the table ones; the
/// pack stamps keep the bare name.
fn parseCodec(text: []const u8) ?format.Codebook {
    const families = [_]struct { []const u8, []const format.Codebook }{
        .{ "exl3-", &.{ .mcg, .mul1 } },
        .{ "stq-", &.{ .std1, .spec1, .pair1 } },
    };
    for (families) |f| {
        if (!std.mem.startsWith(u8, text, f[0])) continue;
        const cb = format.Codebook.fromName(text[f[0].len..]) orelse return null;
        return if (std.mem.indexOfScalar(format.Codebook, f[1], cb) != null) cb else null;
    }
    return null;
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

fn parseTable(text: []const u8) ?*const codebook_tables.Table {
    const t = codebook_tables.lookup(text) orelse return null;
    return if (encode.tableSearchableAny(t)) t else null;
}

/// What `--trunk stq` can round and the engine serves: SPEC1 `spec1-w15`, the only table with a row at the head's K6.
const trunk_codec = format.Codebook.spec1;
const trunk_table_name = "spec1-w15";

fn trunkDecode(o: Options) ArgError!format.Decode {
    const given = o.trunk_codebook != null or o.trunk_codebook_table != null;
    if (given and o.trunk != .stq) return error.TrunkCodebookNeedsStq;
    const cb = o.trunk_codebook orelse trunk_codec;
    if (cb != trunk_codec) return error.TrunkCodebookUnsupported;
    const table = o.trunk_codebook_table orelse codebook_tables.lookup(trunk_table_name).?;
    if (table.codebook != cb) return error.CodebookTableMismatch;
    if (!std.mem.eql(u8, table.name, trunk_table_name)) return error.TrunkCodebookTableUnsupported;
    return .{ .codebook = cb, .window = table.window, .table = codebook_tables.index(table) };
}

fn parseTrunkK(text: []const u8) ?format.Rate {
    if (std.mem.eql(u8, text, "4")) return .{ .n = 64 };
    if (std.mem.eql(u8, text, "5")) return .{ .n = 80 };
    if (std.mem.eql(u8, text, "6")) return .{ .n = 96 };
    return null;
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

/// `vision-affine8`: `--model` and `--out` only.
fn parseVisionArgs(args: []const []const u8, diag: ?*Diag) ArgError!Options {
    var o = Options{ .subcommand = .vision_affine8 };
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--help") or std.mem.eql(u8, a, "-h")) return .{ .help = true, .subcommand = .vision_affine8 };
        const dst = if (std.mem.eql(u8, a, "--model")) &o.model else if (std.mem.eql(u8, a, "--out")) &o.out else {
            if (diag) |d| d.flag = a;
            return error.UnknownFlag;
        };
        if (i + 1 >= args.len) {
            if (diag) |d| d.flag = a;
            return error.MissingFlagValue;
        }
        i += 1;
        dst.* = args[i];
    }
    if (o.model.len == 0) return error.MissingModel;
    if (o.out.len == 0) return error.MissingOut;
    return o;
}

pub fn parseArgs(args: []const []const u8, diag: ?*Diag) ArgError!Options {
    if (args.len == 0) return error.MissingSubcommand;
    if (std.mem.eql(u8, args[0], "--help") or std.mem.eql(u8, args[0], "-h")) return .{ .help = true };
    if (std.mem.eql(u8, args[0], "vision-affine8")) return parseVisionArgs(args[1..], diag);
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
        if (std.mem.eql(u8, a, "--synthetic-codes")) {
            o.synthetic_codes = true;
            continue;
        }
        if (std.mem.eql(u8, a, "--identity-hessian")) {
            o.identity_hessian = true;
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
            .expert_codebook_table => blk: {
                o.expert_codebook_table = parseTable(v) orelse break :blk true;
                break :blk false;
            },
            .trunk_codebook => blk: {
                o.trunk_codebook = parseCodec(v) orelse break :blk true;
                break :blk false;
            },
            .trunk_codebook_table => blk: {
                o.trunk_codebook_table = parseTable(v) orelse break :blk true;
                break :blk false;
            },
            .trunk_k => blk: {
                o.trunk_rate = parseTrunkK(v) orelse break :blk true;
                o.trunk_k_given = true;
                break :blk false;
            },
            .drafter_k => blk: {
                o.drafter_rate = parseTrunkK(v) orelse break :blk true;
                break :blk false;
            },
            .drafter_experts_k => blk: {
                o.drafter_experts_n = parseExpertsN(v) orelse break :blk true;
                break :blk false;
            },
            .capture => blk: {
                o.capture = v;
                break :blk false;
            },
            .calib_ids => blk: {
                o.calib_ids = v;
                break :blk false;
            },
            .calib_windows => blk: {
                o.calib_windows = v;
                break :blk false;
            },
            .rate_plan => blk: {
                o.rate_plan = v;
                break :blk false;
            },
            .trunk_from => blk: {
                o.trunk_from = v;
                break :blk false;
            },
            .trunk => blk: {
                o.trunk = std.meta.stringToEnum(driver.TrunkMode, v) orelse break :blk true;
                break :blk false;
            },
            .fisher_dir => blk: {
                o.fisher_dir = v;
                break :blk false;
            },
            .target_boundaries => blk: {
                o.target_boundaries = v;
                break :blk false;
            },
            .target => blk: {
                o.target = std.meta.stringToEnum(Target, v) orelse break :blk true;
                break :blk false;
            },
            .window => blk: {
                const w = std.fmt.parseInt(i64, v, 10) catch break :blk true;
                if (w < 12 or w > 16) break :blk true;
                o.window = format.Window.fromBits(w) orelse break :blk true;
                o.window_given = true;
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
            .pin_pass => blk: {
                o.pin_pass = std.fmt.parseInt(u32, v, 10) catch break :blk true;
                break :blk !encode.validPinPass(o.pin_pass);
            },
            .layer_cooldown => blk: {
                o.layer_cooldown = std.fmt.parseInt(u32, v, 10) catch break :blk true;
                break :blk false;
            },
            .cpu_threads => blk: {
                const n = std.fmt.parseInt(u32, v, 10) catch break :blk true;
                if (n == 0) break :blk true;
                o.cpu_threads = n;
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
    if (o.expert_codebook == .pair1) {
        if (o.window_given and o.window != .w16) return error.Pair1WindowNot16;
        o.window = .w16;
        if (format.pair1Pair(.{ .n = o.experts_n }) == null) return error.Pair1RateUnsupported;
        if (o.expert_codebook_table == null) o.expert_codebook_table = codebook_tables.lookup("pair1-w16-v1");
    }
    if (o.drafter_experts_n) |n| if (format.pair1Pair(.{ .n = n }) == null) return error.Pair1DrafterRateUnsupported;
    if (o.expert_codebook_table) |t| {
        if (t.codebook != o.expert_codebook) return error.CodebookTableMismatch;
    } else if (encode.needsTable(o.expert_codebook)) return error.CodebookTableMismatch;
    o.trunk_decode = try trunkDecode(o);
    if (o.trunk_k_given and o.trunk != .stq) return error.TrunkKNeedsStq;
    if (o.target == .none and o.target_boundaries.len != 0) return error.TargetNoneWithBoundaries;
    if ((o.calib_ids.len == 0) != (o.calib_windows.len == 0)) return error.CalibWindowsMissing;
    if (o.calib_ids.len != 0 and o.calib.len != 0) return error.CalibSourceConflict;
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

const MissingFile = struct { err: ArgError, path: []const u8 };

fn missingFile(io: std.Io, cwd: std.Io.Dir, opts: Options) ?MissingFile {
    for ([_]MissingFile{
        .{ .err = error.CalibFileMissing, .path = opts.calib },
        .{ .err = error.CalibFileMissing, .path = opts.calib_ids },
        .{ .err = error.CalibFileMissing, .path = opts.calib_windows },
        .{ .err = error.RatePlanFileMissing, .path = opts.rate_plan },
    }) |f| {
        if (f.path.len != 0) cwd.access(io, f.path, .{}) catch return f;
    }
    return null;
}

/// The sentence for an argument error. This module cannot import `src/cli_error.zig`, so it keeps its own.
fn argMessage(err: ArgError) []const u8 {
    return switch (err) {
        error.MissingSubcommand => "a subcommand is required: `sushi convert exl3 ...`",
        error.UnknownSubcommand => "unknown subcommand: `exl3` or `vision-affine8` (see `sushi convert --help`)",
        error.UnknownFlag => "unrecognized flag",
        error.MissingFlagValue => "the flag is missing its value",
        error.BadFlagValue => "the flag has a value it cannot take",
        error.MissingModel => "--model <source checkpoint folder> is required",
        error.MissingOut => "--out <new pack folder> is required",
        error.CalibFileMissing => "the calibration file does not exist",
        error.RatePlanFileMissing => "the --rate-plan file does not exist",
        error.TargetNoneWithBoundaries => "--target-boundaries needs a --target other than none",
        error.CodebookTableMismatch => "the codebook table is for another codebook, or the codebook needs a table",
        error.TrunkCodebookNeedsStq => "--trunk-codebook and --trunk-codebook-table need --trunk stq",
        error.TrunkKNeedsStq => "--trunk-k needs --trunk stq",
        error.TrunkCodebookUnsupported => "the STQ trunk rounds only with codebook stq-spec1",
        error.TrunkCodebookTableUnsupported => "the STQ trunk rounds only with table spec1-w15",
        error.Pair1WindowNot16 => "PAIR1 decodes only at window 16",
        error.Pair1RateUnsupported => "PAIR1 has no table row at this --experts-k",
        error.Pair1DrafterRateUnsupported => "PAIR1 has no table row at this --drafter-experts-k",
        error.CalibWindowsMissing => "--calib-ids and --calib-windows go together",
        error.CalibSourceConflict => "--calib-ids and --calib are two calibration sources: give one",
    };
}

pub fn cmdConvert(allocator: std.mem.Allocator, io: std.Io, args: []const []const u8) !void {
    var diag = Diag{};
    const opts = parseArgs(args, &diag) catch |err| {
        if (diag.flag.len != 0) printOut("sushi convert: {s}: {s}\n\n{s}", .{ diag.flag, argMessage(err), USAGE }) else printOut("sushi convert: {s}\n\n{s}", .{ argMessage(err), USAGE });
        std.process.exit(1);
    };
    if (opts.help) {
        printOut("{s}", .{USAGE});
        return;
    }
    if (opts.subcommand == .vision_affine8) return convertVisionAffine(allocator, io, opts);
    const cwd = std.Io.Dir.cwd();
    if (missingFile(io, cwd, opts)) |missing| {
        printOut("sushi convert: {s}: {s}\n", .{ missing.path, argMessage(missing.err) });
        std.process.exit(1);
    }
    var model_dir = cwd.openDir(io, opts.model, .{ .iterate = true }) catch |err| {
        printOut("sushi convert: cannot open --model {s}: {s}\n", .{ opts.model, if (err == error.FileNotFound) "no such folder" else @errorName(err) });
        std.process.exit(1);
    };
    defer model_dir.close(io);
    const arch = source.identifySourceDir(allocator, io, model_dir) catch |err| {
        printOut("sushi convert: {s}: {s} is not a checkpoint this tool converts (Qwen3.8-Flash-Next, MiMo-V2.6-Flash, GLM-5.3-Flash)\n", .{ @errorName(err), opts.model });
        return err;
    };
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const abs_len = try model_dir.realPath(io, &path_buf);
    const model_abs = path_buf[0..abs_len];

    var plan: ?rate_plan.Plan = null;
    defer if (plan) |*p| p.deinit();
    if (opts.rate_plan.len != 0) {
        const cfg = try model_dir.readFileAlloc(io, "config.json", allocator, .limited(16 << 20));
        defer allocator.free(cfg);
        const shape = try driver.planShape(allocator, cfg);
        const text = try cwd.readFileAlloc(io, opts.rate_plan, allocator, .limited(64 << 20));
        defer allocator.free(text);
        plan = rate_plan.parse(allocator, text, shape.experts, shape.layers) catch |err| {
            printOut("sushi convert: {s} ({s})\n", .{ @errorName(err), opts.rate_plan });
            return err;
        };
    }
    var trunk_buf: [std.fs.max_path_bytes]u8 = undefined;
    var trunk_abs: ?[]const u8 = null;
    if (opts.trunk_from.len != 0) {
        var d = cwd.openDir(io, opts.trunk_from, .{}) catch |err| {
            printOut("sushi convert: cannot open --trunk-from {s}: {s}\n", .{ opts.trunk_from, @errorName(err) });
            return err;
        };
        defer d.close(io);
        trunk_abs = trunk_buf[0..try d.realPath(io, &trunk_buf)];
    }

    if (archRefusal(opts, arch)) |text| {
        printOut("sushi convert: {s}\n", .{text});
        std.process.exit(1);
    }
    if (arch == .hy_v4_flash) return convertHy4(allocator, io, opts, model_abs, if (plan) |*p| p else null);
    if (arch == .deepseek_v41) return convertDsv41(allocator, io, opts, model_abs, if (plan) |*p| p else null);
    if (opts.trunk == .stq) return convertStqTrunk(allocator, io, opts, model_abs, arch, trunk_abs);

    var windows: ?calib.Windows = null;
    defer if (windows) |*w| w.deinit();
    var src: ?source.SafetensorsSource = null;
    defer if (src) |*v| v.deinit();
    var capture: ?capture_mod.Capture = null;
    defer if (capture) |*c| c.deinit();
    if (arch == .qwen4_exp) {
        var tokenizer = try host.tokenizer.loadTokenizerAny(io, allocator, model_abs);
        defer tokenizer.deinit();
        if (opts.calib_ids.len != 0) {
            windows = try calib.idsFileWindows(allocator, io, opts.calib_ids, opts.calib_windows);
        } else if (opts.calib.len != 0) {
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

    pool.setCpuCap(opts.cpu_threads);
    const summary = driver.run(allocator, io, host.mlx.gpuStream(), .{
        .model_dir = model_abs,
        .out_dir = opts.out,
        .rate = .{ .n = opts.experts_n },
        .dec = .{ .codebook = opts.expert_codebook, .window = opts.window, .table = if (opts.expert_codebook_table) |t| codebook_tables.index(t) else null },
        .plan = if (plan) |*p| p else null,
        .trunk_from = trunk_abs,
        .fisher_dir = if (opts.fisher_dir.len != 0) opts.fisher_dir else null,
        .target_boundaries = if (opts.target_boundaries.len != 0) opts.target_boundaries else null,
        .target_in_run = opts.target == .compensated and opts.target_boundaries.len == 0,
        .trunk = opts.trunk,
        // The re-rounded trunk's reference keeps a measured Hessian for every expert any row reaches.
        .min_rows = if (opts.trunk == .gptq) 1 else 64,
        .trunk_bits = opts.trunk_bits,
        .trunk_group = opts.trunk_group,
        .batch = opts.batch,
        .factor_precision = opts.factor_precision,
        .pin_pass = opts.pin_pass,
        .calib_sha256 = if (windows) |*w| &w.sha256 else "none",
        .calib_tokens = if (windows) |*w| w.tokens() else 0,
        .resuming = opts.resuming,
        .layer_cooldown = opts.layer_cooldown,
        .version = version,
    }, if (capture) |*c| c.calibrator() else null, .{ .write = progress }) catch |err| {
        printOut("sushi convert: {s} ({s})\n", .{ @errorName(err), @tagName(arch) });
        return err;
    };
    printOut("converted {d} layers ({d} reused){s} into {s}\n", .{ summary.layers_written + summary.layers_reused, summary.layers_reused, if (summary.mtp_written) " and the MTP head" else "", opts.out });
}

/// `--trunk stq`: the trunk of the `--trunk-from` pack rounded onto the trellis, its experts kept.
/// The refusal of a flag the source's arch does not honor, or null: every flag a convert takes is either used by
/// that arch's writer or refused by name before anything runs.
fn archRefusal(opts: Options, arch: source.Arch) ?[]const u8 {
    if ((opts.synthetic_codes or opts.identity_hessian) and arch != .hy_v4_flash)
        return "Hy4CodesOnly (--synthetic-codes and --identity-hessian write Hy4-Flash packs only)";
    if (opts.drafter_rate != null and arch != .hy_v4_flash and arch != .deepseek_v41)
        return "DrafterKUnsupported (--drafter-k sets the drafter's rate of Hy4-Flash and DeepSeek-V4.1)";
    if (opts.drafter_experts_n != null and arch != .deepseek_v41)
        return "DrafterExpertsKDsv41Only (--drafter-experts-k sets the DeepSeek-V4.1 DSpark experts' rate)";
    if (opts.trunk_from.len != 0 and (arch == .hy_v4_flash or arch == .deepseek_v41))
        return "TrunkFromUnsupported (--trunk-from takes a Qwen3.8-Flash-Next pack's trunk; Hy4-Flash and DeepSeek-V4.1 write theirs from the source)";
    return null;
}

fn convertStqTrunk(allocator: std.mem.Allocator, io: std.Io, opts: Options, model_abs: []const u8, arch: source.Arch, trunk_abs: ?[]const u8) !void {
    const fail = struct {
        fn with(err: anyerror, why: []const u8) anyerror {
            printOut("sushi convert: {s} ({s})\n", .{ @errorName(err), why });
            return err;
        }
    }.with;
    if (arch != .qwen4_exp) return fail(error.ArchitectureNotConvertible, "--trunk stq converts Qwen3.8-Flash-Next");
    const base_abs = trunk_abs orelse return fail(error.TrunkFromMissing, "--trunk stq rounds the trunk of the --trunk-from pack");
    if (opts.calib_ids.len == 0) return fail(error.CalibWindowsMissing, "--trunk stq reads --calib-ids and --calib-windows");
    var model_dir = try std.Io.Dir.cwd().openDir(io, model_abs, .{ .iterate = true });
    defer model_dir.close(io);
    var windows = try calib.idsFileWindows(allocator, io, opts.calib_ids, opts.calib_windows);
    defer windows.deinit();
    var src = try source.SafetensorsSource.open(allocator, io, model_dir);
    defer src.deinit();
    var capture = capture_mod.Capture.init(allocator, io, host.mlx.gpuStream(), &windows, &src, capture_budget_bytes);
    defer capture.deinit();
    pool.setCpuCap(opts.cpu_threads);
    // Dev lever: round only the first N layers (a timing run); the pack is left without `config.json`.
    const only: ?u32 = if (std.c.getenv("SUSHI_ENCODER_STQ_LAYERS")) |raw| try std.fmt.parseInt(u32, std.mem.span(raw), 10) else null;
    const summary = trunk_stq.run(allocator, io, .{
        .model_dir = model_abs,
        .base_dir = base_abs,
        .out_dir = opts.out,
        .stq = .{ .rate = opts.trunk_rate, .pin_pass = opts.pin_pass, .decode = opts.trunk_decode },
        .calib_sha256 = &windows.sha256,
        .calib_tokens = windows.tokens(),
        .resuming = opts.resuming,
        .version = version,
        .only_layers = only,
    }, &capture, .{ .write = progress }) catch |err| return fail(err, "qwen4_exp");
    printOut("rounded {d} layers ({d} resumed) and {d} banks into {s}\n", .{ summary.layers_written, summary.layers_resumed, summary.banks, opts.out });
}

fn convertVisionAffine(allocator: std.mem.Allocator, io: std.Io, opts: Options) !void {
    const s = vision_affine.run(allocator, io, .{ .model_dir = opts.model, .out_dir = opts.out, .version = version }) catch |err| {
        printOut("sushi convert: {s} (vision-affine8 of {s})\n", .{ @errorName(err), opts.model });
        return err;
    };
    printOut("wrote {s} into {s}: {d} matrices affine 8 at group {d}, {d} matrices kept BF16 ({d} bytes, named in vision_quant.bf16_kept), {d} tensors as stored in all, {d} -> {d} bytes; {d} shards rewritten without the tower, {d} files cloned\n", .{ vision_affine.vision_file, opts.out, s.quantized, s.group, s.kept, s.kept_bytes, s.stored, s.bytes_before, s.bytes_after, s.rewritten, s.cloned });
}

fn convertHy4(allocator: std.mem.Allocator, io: std.Io, opts: Options, model_abs: []const u8, plan: ?*const rate_plan.Plan) !void {
    const given = @as(u32, @intFromBool(opts.synthetic_codes)) + @intFromBool(opts.identity_hessian) + @intFromBool(opts.capture.len > 0);
    if (given != 1) {
        printOut("sushi convert: Hy4NeedsCodes (a hy_v4_flash checkpoint converts with exactly one of --capture <calib dir>, --identity-hessian and --synthetic-codes)\n", .{});
        return error.Hy4NeedsCodes;
    }
    var cap_buf: [std.fs.max_path_bytes]u8 = undefined;
    var cap_abs: []const u8 = "";
    if (opts.capture.len > 0) {
        var cap = std.Io.Dir.cwd().openDir(io, opts.capture, .{}) catch |err| {
            printOut("sushi convert: cannot open --capture {s}: {s}\n", .{ opts.capture, @errorName(err) });
            return err;
        };
        defer cap.close(io);
        cap_abs = cap_buf[0..try cap.realPath(io, &cap_buf)];
    }
    const expert_dec: format.Decode = switch (opts.expert_codebook) {
        .pair1 => hy4.default_expert_dec,
        .spec1 => .{ .codebook = .spec1, .window = opts.window, .table = codebook_tables.index(opts.expert_codebook_table.?) },
        else => {
            printOut("sushi convert: Hy4ExpertCodec (--expert-codebook stq-pair1, or stq-spec1 with its table)\n", .{});
            return error.Hy4ExpertCodec;
        },
    };
    // Dev lever: write only these layers' shards (comma list), for a one-layer timing.
    var only = std.ArrayList(u32).empty;
    defer only.deinit(allocator);
    if (std.c.getenv("SUSHI_ENCODER_HY4_LAYERS")) |raw| {
        var it = std.mem.splitScalar(u8, std.mem.span(raw), ',');
        while (it.next()) |part| try only.append(allocator, try std.fmt.parseInt(u32, std.mem.trim(u8, part, " "), 10));
    }
    pool.setCpuCap(opts.cpu_threads);
    const started = std.Io.Timestamp.now(io, .awake);
    const summary = hy4.run(allocator, io, .{
        .model_dir = model_abs,
        .out_dir = opts.out,
        .rate = .{ .n = opts.experts_n },
        .plan = plan,
        .codes = if (opts.synthetic_codes) .synthetic else if (opts.identity_hessian) .identity_hessian else .capture,
        .capture_dir = cap_abs,
        .pin_pass = opts.pin_pass,
        .batch = opts.batch,
        .expert_dec = expert_dec,
        .trunk_rate = if (opts.trunk_k_given) opts.trunk_rate else hy4.default_trunk_rate,
        .drafter_rate = opts.drafter_rate,
        .only_layers = if (only.items.len > 0) only.items else null,
        .version = version,
        .resuming = opts.resuming,
    }, .{ .write = progress }) catch |err| {
        printOut("sushi convert: {s} (hy_v4_flash)\n", .{@errorName(err)});
        return err;
    };
    const secs = @as(f64, @floatFromInt(started.untilNow(io, .awake).nanoseconds)) / 1e9;
    printOut("wrote pack {s} into {s} in {d:.1} s: {d} expert layers ({d} rate groups), {d} trunk banks, {d} tensors as stored, {d} Engram tables cloned, {d} bytes written\n", .{ summary.revision(), opts.out, secs, summary.expert_layers, summary.expert_groups, summary.trunk_banks, summary.stored, summary.engram_tables, summary.bytes_written });
    if (opts.resuming) printOut("resumed: kept {d} expert layers and {d} trunk, drafter or globals shards of {s}\n", .{ summary.experts_resumed, summary.units_resumed, summary.revision() });
    if (opts.capture.len > 0) printOut("against the identity: {d} banks the capture taps no input of (the embedding, the DFly2 drafter), {d} banks under {d} rows\n", .{ summary.untapped_banks, summary.short_banks, hy4.default_min_rows });
}

const Dsv41Rates = struct { trunk: format.Rate, drafter: ?format.Rate, drafter_experts: format.Rate };

/// What the rate flags give a DeepSeek-V4.1 run; the converter's own rates where a flag is unset.
fn dsv41Rates(opts: Options) Dsv41Rates {
    return .{
        .trunk = if (opts.trunk_k_given) opts.trunk_rate else dsv41.default_trunk_rate,
        .drafter = opts.drafter_rate,
        .drafter_experts = if (opts.drafter_experts_n) |n| .{ .n = n } else dsv41.default_stage_rate,
    };
}

fn convertDsv41(allocator: std.mem.Allocator, io: std.Io, opts: Options, model_abs: []const u8, plan: ?*const rate_plan.Plan) !void {
    if (opts.capture.len == 0) {
        printOut("sushi convert: CaptureMissing (a deepseek_v41 checkpoint converts from --capture <calib dir>)\n", .{});
        return error.CaptureMissing;
    }
    if (opts.expert_codebook != .pair1) {
        printOut("sushi convert: Dsv41ExpertsArePair1 (--expert-codebook stq-pair1)\n", .{});
        return error.Dsv41ExpertsArePair1;
    }
    var cap_buf: [std.fs.max_path_bytes]u8 = undefined;
    var cap = std.Io.Dir.cwd().openDir(io, opts.capture, .{}) catch |err| {
        printOut("sushi convert: cannot open --capture {s}: {s}\n", .{ opts.capture, @errorName(err) });
        return err;
    };
    const cap_abs = cap_buf[0..try cap.realPath(io, &cap_buf)];
    cap.close(io);
    // Dev lever: convert only these backbone layers (comma list), for a timing run.
    var only = std.ArrayList(u32).empty;
    defer only.deinit(allocator);
    if (std.c.getenv("SUSHI_ENCODER_DSV41_LAYERS")) |raw| {
        var it = std.mem.splitScalar(u8, std.mem.span(raw), ',');
        while (it.next()) |part| try only.append(allocator, try std.fmt.parseInt(u32, std.mem.trim(u8, part, " "), 10));
    }
    pool.setCpuCap(opts.cpu_threads);
    const rates = dsv41Rates(opts);
    const summary = dsv41.run(allocator, io, host.mlx.gpuStream(), .{
        .model_dir = model_abs,
        .out_dir = opts.out,
        .capture_dir = cap_abs,
        .rate = .{ .n = opts.experts_n },
        .plan = plan,
        .pin_pass = opts.pin_pass,
        .batch = opts.batch,
        .resuming = opts.resuming,
        .version = version,
        .only_layers = if (only.items.len > 0) only.items else null,
        .trunk_rate = rates.trunk,
        .drafter_rate = rates.drafter,
        .drafter_experts_rate = rates.drafter_experts,
    }, .{ .write = progress }) catch |err| {
        printOut("sushi convert: {s} (deepseek_v41)\n", .{@errorName(err)});
        return err;
    };
    printOut("converted {d} layers ({d} reused) and {d} DSpark stages into {s}; identity Hessians: gate/up {d}, down {d}, trunk {d}\n", .{ summary.layers_written, summary.layers_reused, summary.stages_written, opts.out, summary.gate_up_identity, summary.down_identity, summary.trunk_identity });
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
    try t.expectEqual(driver.TrunkMode.gptq, o.trunk);
    try t.expectEqual(driver.TrunkMode.stored, (try expectParse(&.{ "exl3", "--model", "m", "--out", "o", "--trunk", "stored" })).trunk);
    try t.expectEqual(driver.TrunkMode.rtn, (try expectParse(&.{ "exl3", "--model", "m", "--out", "o", "--trunk", "rtn" })).trunk);
    try t.expectError(error.BadFlagValue, expectParse(&.{ "exl3", "--model", "m", "--out", "o", "--trunk", "served" }));
    try t.expectEqual(@as(u32, 16), o.batch);
    try t.expectEqual(gpu_factor.Precision.f64, o.factor_precision);
    try t.expectEqual(@as(u32, 64), o.pin_pass);
    try t.expectEqual(@as(u32, 128), (try expectParse(&.{ "exl3", "--model", "m", "--out", "o", "--pin-pass", "128" })).pin_pass);

    const all = try expectParse(&.{ "exl3", "--model", "m", "--out", "o", "--experts-k", "1.5", "--expert-codebook", "exl3-mul1", "--window", "12", "--calib", "c.txt", "--calib-tokens", "4096", "--trunk-bits", "6", "--trunk-group", "128", "--batch", "4", "--factor-precision", "32", "--pin-pass", "96", "--resume" });
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
    try t.expectEqual(@as(u32, 96), all.pin_pass);
    try t.expectEqual(@as(u32, 16), (try expectParse(&.{ "exl3", "--model", "m", "--out", "o", "--pin-pass", "16" })).pin_pass);
    try t.expectEqual(gpu_factor.Precision.f64, (try expectParse(&.{ "exl3", "--model", "m", "--out", "o", "--factor-precision", "64" })).factor_precision);
    try t.expectEqual(@as(u32, 128), (try expectParse(&.{ "exl3", "--model", "m", "--out", "o", "--experts-k", "8" })).experts_n);
    try t.expectEqual(@as(u32, 42), (try expectParse(&.{ "exl3", "--model", "m", "--out", "o", "--experts-k", "2.625" })).experts_n);
    try t.expectEqual(format.Window.w16, (try expectParse(&.{ "exl3", "--model", "m", "--out", "o", "--window", "16" })).window);

    const stq = try expectParse(&.{ "exl3", "--model", "m", "--out", "o", "--expert-codebook-table", "std1", "--expert-codebook", "stq-std1", "--rate-plan", "p.json", "--trunk-from", "served", "--fisher-dir", "fd", "--target-boundaries", "tb", "--calib-ids", "t.u32", "--calib-windows", "w.json" });
    try t.expectEqual(format.Codebook.std1, stq.expert_codebook);
    try t.expectEqualStrings("std1", stq.expert_codebook_table.?.name);
    try t.expectEqualStrings("p.json", stq.rate_plan);
    try t.expectEqualStrings("served", stq.trunk_from);
    try t.expectEqualStrings("fd", stq.fisher_dir);
    try t.expectEqualStrings("tb", stq.target_boundaries);
    try t.expect(stq.target == null);
    try t.expectEqual(Target.compensated, (try expectParse(&.{ "exl3", "--model", "m", "--out", "o", "--target", "compensated" })).target);
    try t.expectEqualStrings("t.u32", stq.calib_ids);
    try t.expectEqualStrings("w.json", stq.calib_windows);
}

test "convert cli: a bad value, an unknown flag or a missing piece is refused before anything runs" {
    const t = std.testing;
    const base = [_][]const u8{ "exl3", "--model", "m", "--out", "o" };
    const bad_values = [_][2][]const u8{
        .{ "--experts-k", "1.25" },            .{ "--experts-k", "8.125" },                    .{ "--experts-k", "2.51" },                   .{ "--experts-k", "x" },
        .{ "--experts-k", "nan" },             .{ "--expert-codebook", "stq-beta" },           .{ "--expert-codebook", "stq-s1" },           .{ "--expert-codebook", "stq-dd1" },
        .{ "--expert-codebook", "stq-pairs" }, .{ "--expert-codebook-table", "pair1-w16-v2" }, .{ "--expert-codebook-table", "spec1-nope" }, .{ "--window", "11" },
        .{ "--window", "17" },                 .{ "--window", "x" },                           .{ "--calib-tokens", "2047" },                .{ "--calib-tokens", "-1" },
        .{ "--trunk-bits", "7" },              .{ "--trunk-bits", "1" },                       .{ "--trunk-group", "48" },                   .{ "--batch", "0" },
        .{ "--batch", "65" },                  .{ "--batch", "x" },                            .{ "--factor-precision", "16" },              .{ "--factor-precision", "x" },
        .{ "--factor-precision", "032" },      .{ "--factor-precision", "" },                  .{ "--pin-pass", "0" },                       .{ "--pin-pass", "8" },
        .{ "--pin-pass", "24" },               .{ "--pin-pass", "129" },                       .{ "--pin-pass", "144" },                     .{ "--pin-pass", "-16" },
        .{ "--pin-pass", "x" },                .{ "--target", "stored" },                      .{ "--target", "" },
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
    try t.expectError(error.CodebookTableMismatch, parseArgs(&(base ++ [_][]const u8{ "--expert-codebook", "stq-spec1" }), null));
    try t.expectError(error.CodebookTableMismatch, parseArgs(&(base ++ [_][]const u8{ "--expert-codebook-table", "std1" }), null));
    try t.expectError(error.CalibWindowsMissing, parseArgs(&(base ++ [_][]const u8{ "--calib-ids", "t.u32" }), null));
    try t.expectError(error.CalibSourceConflict, parseArgs(&(base ++ [_][]const u8{ "--calib-ids", "t.u32", "--calib-windows", "w.json", "--calib", "c.txt" }), null));
    try t.expectError(error.UnknownSubcommand, parseArgs(&.{"gguf"}, null));
    try t.expectError(error.MissingSubcommand, parseArgs(&.{}, null));
}

test "convert cli: pair1 runs at window 16 under pair1-w16-v1, and refuses another window or rate by name" {
    const t = std.testing;
    const base = [_][]const u8{ "exl3", "--model", "m", "--out", "o", "--expert-codebook", "stq-pair1" };
    const o = try expectParse(&(base ++ [_][]const u8{ "--experts-k", "1.75" }));
    try t.expectEqual(format.Codebook.pair1, o.expert_codebook);
    try t.expectEqual(format.Window.w16, o.window);
    try t.expectEqualStrings("pair1-w16-v1", o.expert_codebook_table.?.name);
    try t.expectEqual(@as(u32, 28), o.experts_n);
    _ = try expectParse(&(base ++ [_][]const u8{ "--window", "16", "--expert-codebook-table", "pair1-w16-v1", "--experts-k", "4" }));
    try t.expectError(error.Pair1WindowNot16, parseArgs(&(base ++ [_][]const u8{ "--window", "15" }), null));
    try t.expectError(error.Pair1RateUnsupported, parseArgs(&(base ++ [_][]const u8{ "--experts-k", "4.5" }), null));
    try t.expectError(error.Pair1RateUnsupported, parseArgs(&(base ++ [_][]const u8{ "--experts-k", "8" }), null));
    try t.expectError(error.CodebookTableMismatch, parseArgs(&(base ++ [_][]const u8{ "--expert-codebook-table", "spec1-w16" }), null));
    try t.expectError(error.CodebookTableMismatch, parseArgs(&.{ "exl3", "--model", "m", "--out", "o", "--expert-codebook", "stq-spec1", "--expert-codebook-table", "pair1-w16-v1" }, null));
}

test "convert cli: codec values carry their family, and the old flag names are gone" {
    const t = std.testing;
    const base = [_][]const u8{ "exl3", "--model", "m", "--out", "o" };
    try t.expectEqual(format.Codebook.mcg, (try expectParse(&(base ++ [_][]const u8{ "--expert-codebook", "exl3-mcg" }))).expert_codebook);
    try t.expectEqual(format.Codebook.mul1, (try expectParse(&(base ++ [_][]const u8{ "--expert-codebook", "exl3-mul1" }))).expert_codebook);
    try t.expectEqual(format.Codebook.std1, (try expectParse(&(base ++ [_][]const u8{ "--expert-codebook", "stq-std1", "--expert-codebook-table", "std1" }))).expert_codebook);
    try t.expectEqual(format.Codebook.spec1, (try expectParse(&(base ++ [_][]const u8{ "--expert-codebook", "stq-spec1", "--expert-codebook-table", "spec1-w15" }))).expert_codebook);
    try t.expectEqual(format.Codebook.pair1, (try expectParse(&(base ++ [_][]const u8{ "--expert-codebook", "stq-pair1" }))).expert_codebook);
    for ([_][]const u8{ "mcg", "mul1", "std1", "spec1", "pair1", "stq-mcg", "stq-mul1", "exl3-std1", "exl3-spec1", "exl3-pair1", "exl3-alpha", "stq-alpha", "exl3-", "stq-" }) |bad| {
        var diag = Diag{};
        try t.expectError(error.BadFlagValue, parseArgs(&(base ++ [_][]const u8{ "--expert-codebook", bad }), &diag));
        try t.expectEqualStrings("--expert-codebook", diag.flag);
        try t.expectError(error.BadFlagValue, parseArgs(&(base ++ [_][]const u8{ "--trunk", "stq", "--trunk-codebook", bad }), null));
    }
    for ([_][]const u8{ "--codebook", "--codebook-table" }) |old| {
        var diag = Diag{};
        try t.expectError(error.UnknownFlag, parseArgs(&(base ++ [_][]const u8{ old, "mcg" }), &diag));
        try t.expectEqualStrings(old, diag.flag);
    }
}

test "convert cli: the trunk codec defaults to stq-spec1 spec1-w15, needs --trunk stq, and refuses what the trunk cannot serve" {
    const t = std.testing;
    const base = [_][]const u8{ "exl3", "--model", "m", "--out", "o" };
    const d = try expectParse(&base);
    try t.expectEqual(format.Codebook.spec1, d.trunk_decode.codebook);
    try t.expectEqualStrings("spec1-w15", codebook_tables.tables[d.trunk_decode.table.?].name);
    const stq = base ++ [_][]const u8{ "--trunk", "stq" };
    const given = try expectParse(&(stq ++ [_][]const u8{ "--trunk-codebook", "stq-spec1", "--trunk-codebook-table", "spec1-w15", "--expert-codebook", "stq-pair1", "--experts-k", "2" }));
    try t.expectEqual(format.Codebook.spec1, given.trunk_decode.codebook);
    try t.expectEqual(format.Codebook.pair1, given.expert_codebook);
    try t.expectEqual(format.Window.w15, given.trunk_decode.window);

    try t.expectError(error.TrunkCodebookNeedsStq, parseArgs(&(base ++ [_][]const u8{ "--trunk-codebook", "stq-spec1" }), null));
    try t.expectError(error.TrunkCodebookNeedsStq, parseArgs(&(base ++ [_][]const u8{ "--trunk-codebook-table", "spec1-w15" }), null));
    try t.expectError(error.TrunkCodebookNeedsStq, parseArgs(&(base ++ [_][]const u8{ "--trunk", "gptq", "--trunk-codebook", "stq-spec1" }), null));
    try t.expectError(error.TrunkCodebookUnsupported, parseArgs(&(stq ++ [_][]const u8{ "--trunk-codebook", "stq-std1", "--trunk-codebook-table", "std1" }), null));
    try t.expectError(error.TrunkCodebookUnsupported, parseArgs(&(stq ++ [_][]const u8{ "--trunk-codebook", "stq-pair1" }), null));
    try t.expectError(error.TrunkCodebookUnsupported, parseArgs(&(stq ++ [_][]const u8{ "--trunk-codebook", "exl3-mcg" }), null));
    try t.expectError(error.CodebookTableMismatch, parseArgs(&(stq ++ [_][]const u8{ "--trunk-codebook-table", "std1" }), null));
    try t.expectError(error.TrunkCodebookTableUnsupported, parseArgs(&(stq ++ [_][]const u8{ "--trunk-codebook-table", "spec1-w16" }), null));
    try t.expectError(error.BadFlagValue, parseArgs(&(stq ++ [_][]const u8{ "--trunk-codebook-table", "spec1-nope" }), null));
}

test "convert cli: --trunk-k picks the STQ trunk's rate, 4 by default, and needs --trunk stq" {
    const t = std.testing;
    const base = [_][]const u8{ "exl3", "--model", "m", "--out", "o" };
    const stq = base ++ [_][]const u8{ "--trunk", "stq" };
    try t.expectEqual(@as(u32, 64), (try expectParse(&stq)).trunk_rate.n);
    try t.expectEqual(@as(u32, 64), (try expectParse(&(stq ++ [_][]const u8{ "--trunk-k", "4" }))).trunk_rate.n);
    try t.expectEqual(@as(u32, 80), (try expectParse(&(stq ++ [_][]const u8{ "--trunk-k", "5" }))).trunk_rate.n);
    try t.expectEqual(@as(u32, 96), (try expectParse(&(stq ++ [_][]const u8{ "--trunk-k", "6" }))).trunk_rate.n);
    for ([_][]const u8{ "3", "7", "8", "6.0", "x" }) |bad| {
        var diag = Diag{};
        try t.expectError(error.BadFlagValue, parseArgs(&(stq ++ [_][]const u8{ "--trunk-k", bad }), &diag));
        try t.expectEqualStrings("--trunk-k", diag.flag);
    }
    try t.expectError(error.TrunkKNeedsStq, parseArgs(&(base ++ [_][]const u8{ "--trunk-k", "6" }), null));
    try t.expectError(error.TrunkKNeedsStq, parseArgs(&(base ++ [_][]const u8{ "--trunk", "gptq", "--trunk-k", "4" }), null));
}

test "convert cli: --drafter-k picks the drafter's bank rate, 4, 5 or 6, unset by default" {
    const t = std.testing;
    const base = [_][]const u8{ "exl3", "--model", "m", "--out", "o" };
    try t.expect((try expectParse(&base)).drafter_rate == null);
    try t.expectEqual(@as(u32, 64), (try expectParse(&(base ++ [_][]const u8{ "--drafter-k", "4" }))).drafter_rate.?.n);
    try t.expectEqual(@as(u32, 80), (try expectParse(&(base ++ [_][]const u8{ "--drafter-k", "5" }))).drafter_rate.?.n);
    try t.expectEqual(@as(u32, 96), (try expectParse(&(base ++ [_][]const u8{ "--drafter-k", "6" }))).drafter_rate.?.n);
    for ([_][]const u8{ "3", "7", "x" }) |bad| {
        var diag = Diag{};
        try t.expectError(error.BadFlagValue, parseArgs(&(base ++ [_][]const u8{ "--drafter-k", bad }), &diag));
        try t.expectEqualStrings("--drafter-k", diag.flag);
    }
}

test "convert cli: --drafter-experts-k picks the drafter's routed-expert rate on the PAIR1 grid, unset by default" {
    const t = std.testing;
    const base = [_][]const u8{ "exl3", "--model", "m", "--out", "o", "--expert-codebook", "stq-pair1" };
    try t.expect((try expectParse(&base)).drafter_experts_n == null);
    try t.expectEqual(@as(u32, 40), (try expectParse(&(base ++ [_][]const u8{ "--drafter-experts-k", "2.5" }))).drafter_experts_n.?);
    try t.expectEqual(@as(u32, 32), (try expectParse(&(base ++ [_][]const u8{ "--drafter-experts-k", "2" }))).drafter_experts_n.?);
    for ([_][]const u8{ "1.25", "2.51", "x" }) |bad| {
        var diag = Diag{};
        try t.expectError(error.BadFlagValue, parseArgs(&(base ++ [_][]const u8{ "--drafter-experts-k", bad }), &diag));
        try t.expectEqualStrings("--drafter-experts-k", diag.flag);
    }
    try t.expectError(error.Pair1DrafterRateUnsupported, parseArgs(&(base ++ [_][]const u8{ "--drafter-experts-k", "4.5" }), null));
}

test "convert cli: the rate flags become the recipe's rates, and the recipe's own where unset" {
    const t = std.testing;
    const base = [_][]const u8{ "exl3", "--model", "m", "--out", "o", "--expert-codebook", "stq-pair1", "--trunk", "stq" };
    const unset = dsv41Rates(try expectParse(&base));
    try t.expectEqual(dsv41.default_trunk_rate.n, unset.trunk.n);
    try t.expect(unset.drafter == null);
    try t.expectEqual(dsv41.default_stage_rate.n, unset.drafter_experts.n);
    const set = dsv41Rates(try expectParse(&(base ++ [_][]const u8{ "--trunk-k", "5", "--drafter-k", "6", "--drafter-experts-k", "2.5" })));
    try t.expectEqual(@as(u32, 80), set.trunk.n);
    try t.expectEqual(@as(u32, 96), set.drafter.?.n);
    try t.expectEqual(@as(u32, 40), set.drafter_experts.n);
}

test "convert cli: --help works after convert and after exl3, and lists every flag" {
    const t = std.testing;
    try t.expect((try expectParse(&.{"--help"})).help);
    try t.expect((try expectParse(&.{ "exl3", "--help" })).help);
    try t.expect((try expectParse(&.{ "exl3", "--model", "m", "-h" })).help);
    for (value_flags) |row| try t.expect(std.mem.indexOf(u8, USAGE, row[0]) != null);
    for (switch_flags) |name| try t.expect(std.mem.indexOf(u8, USAGE, name) != null);
}

test "convert cli: --layer-cooldown and --cpu-threads default to off and parse whole positive counts" {
    const t = std.testing;
    const base = [_][]const u8{ "exl3", "--model", "m", "--out", "o" };
    const d = try expectParse(&base);
    try t.expectEqual(@as(u32, 0), d.layer_cooldown);
    try t.expect(d.cpu_threads == null);
    try t.expectEqual(@as(u32, 15), (try expectParse(&(base ++ [_][]const u8{ "--layer-cooldown", "15" }))).layer_cooldown);
    try t.expectEqual(@as(u32, 0), (try expectParse(&(base ++ [_][]const u8{ "--layer-cooldown", "0" }))).layer_cooldown);
    try t.expectEqual(@as(?u32, 6), (try expectParse(&(base ++ [_][]const u8{ "--cpu-threads", "6" }))).cpu_threads);
    const bad_values = [_][2][]const u8{
        .{ "--layer-cooldown", "-1" }, .{ "--layer-cooldown", "x" }, .{ "--layer-cooldown", "1.5" }, .{ "--layer-cooldown", "" },
        .{ "--cpu-threads", "0" },     .{ "--cpu-threads", "-2" },   .{ "--cpu-threads", "x" },      .{ "--cpu-threads", "2.5" },
    };
    for (bad_values) |pair| {
        var diag = Diag{};
        try t.expectError(error.BadFlagValue, parseArgs(&(base ++ [_][]const u8{ pair[0], pair[1] }), &diag));
        try t.expectEqualStrings(pair[0], diag.flag);
    }
}

test "convert cli: stored boundaries beside an explicit --target none are refused, beside no --target or compensated they are not" {
    const t = std.testing;
    const base = [_][]const u8{ "exl3", "--model", "m", "--out", "o", "--target-boundaries", "tb" };
    try t.expectError(error.TargetNoneWithBoundaries, parseArgs(&(base ++ [_][]const u8{ "--target", "none" }), null));
    try t.expectError(error.TargetNoneWithBoundaries, parseArgs(&.{ "exl3", "--target", "none", "--model", "m", "--out", "o", "--target-boundaries", "tb" }, null));
    try t.expect((try expectParse(&base)).target == null);
    try t.expectEqual(Target.compensated, (try expectParse(&(base ++ [_][]const u8{ "--target", "compensated" }))).target.?);
    try t.expectEqual(Target.none, (try expectParse(&.{ "exl3", "--model", "m", "--out", "o", "--target", "none" })).target.?);
}

test "convert cli: a missing rate plan file is named as one, not as a calibration file" {
    const t = std.testing;
    const io = t.io;
    var work = t.tmpDir(.{});
    defer work.cleanup();
    try work.dir.writeFile(io, .{ .sub_path = "calib.txt", .data = "x" });
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = buf[0..try work.dir.realPath(io, &buf)];
    const calib_path = try std.fmt.allocPrint(t.allocator, "{s}/calib.txt", .{dir});
    defer t.allocator.free(calib_path);
    const plan_path = try std.fmt.allocPrint(t.allocator, "{s}/plan.json", .{dir});
    defer t.allocator.free(plan_path);
    const cwd = std.Io.Dir.cwd();
    var opts = try expectParse(&.{ "exl3", "--model", "m", "--out", "o", "--calib", calib_path, "--rate-plan", plan_path });
    const missing = missingFile(io, cwd, opts).?;
    try t.expectEqual(@as(ArgError, error.RatePlanFileMissing), missing.err);
    try t.expectEqualStrings(plan_path, missing.path);
    opts.calib = plan_path;
    try t.expectEqual(@as(ArgError, error.CalibFileMissing), missingFile(io, cwd, opts).?.err);
    try work.dir.writeFile(io, .{ .sub_path = "plan.json", .data = "{}" });
    opts.calib = calib_path;
    try t.expect(missingFile(io, cwd, opts) == null);
}

test "convert cli: vision-affine8 takes --model and --out, and no flag of exl3's" {
    const t = std.testing;
    const o = try expectParse(&.{ "vision-affine8", "--model", "m", "--out", "o" });
    try t.expectEqual(Subcommand.vision_affine8, o.subcommand);
    try t.expectEqualStrings("m", o.model);
    try t.expectEqualStrings("o", o.out);
    try t.expectEqual(Subcommand.exl3, (try expectParse(&.{ "exl3", "--model", "m", "--out", "o" })).subcommand);
    try t.expect((try expectParse(&.{ "vision-affine8", "--help" })).help);
    var diag = Diag{};
    try t.expectError(error.UnknownFlag, parseArgs(&.{ "vision-affine8", "--model", "m", "--out", "o", "--experts-k", "4" }, &diag));
    try t.expectEqualStrings("--experts-k", diag.flag);
    try t.expectError(error.UnknownFlag, parseArgs(&.{ "vision-affine8", "--model", "m", "--out", "o", "--resume" }, null));
    try t.expectError(error.MissingOut, parseArgs(&.{ "vision-affine8", "--model", "m" }, null));
    try t.expectError(error.MissingModel, parseArgs(&.{ "vision-affine8", "--out", "o" }, null));
    try t.expectError(error.MissingFlagValue, parseArgs(&.{ "vision-affine8", "--model" }, null));
    try t.expect(std.mem.indexOf(u8, USAGE, "sushi convert vision-affine8 --model <pack> --out <dir>") != null);
}

test "convert cli: --drafter-experts-k outside PAIR1's grid is refused even when --expert-codebook is omitted" {
    const t = std.testing;
    try t.expectError(error.Pair1DrafterRateUnsupported, parseArgs(&.{ "exl3", "--model", "m", "--out", "o", "--drafter-experts-k", "4.5" }, null));
}

test "convert cli: a flag the source's arch does not honor, --trunk-from among them, is refused by name before anything runs" {
    const t = std.testing;
    var o: Options = .{};
    o.trunk_from = "served";
    try t.expect(std.mem.startsWith(u8, archRefusal(o, .hy_v4_flash).?, "TrunkFromUnsupported"));
    try t.expect(std.mem.startsWith(u8, archRefusal(o, .deepseek_v41).?, "TrunkFromUnsupported"));
    try t.expect(archRefusal(o, .qwen4_exp) == null);
    o = .{};
    o.drafter_experts_n = 40;
    try t.expect(std.mem.startsWith(u8, archRefusal(o, .hy_v4_flash).?, "DrafterExpertsKDsv41Only"));
    try t.expect(archRefusal(o, .deepseek_v41) == null);
    try t.expect(archRefusal(.{}, .hy_v4_flash) == null);
}
