const std = @import("std");
const mlx = @import("../mlx.zig");
const model = @import("../model.zig");
const load = @import("../model_load.zig");
const report = @import("../ubench_report.zig");
const Trunk = @import("../trunk.zig").Trunk;
const dsv = @import("dsv41_model.zig");
const serving = @import("dsv41_serving.zig");

pub const bench: report.Bench = .{
    .name = "dsv41.joined_decode",
    .scope = .dsv41,
    .kind = .pack,
    .what = "resident decode with one prefilled state per request and a solo logit check at each width",
    .unit = .ms,
    .steps = 100,
    .axes = &.{.{ .name = "rows", .values = &.{ "1", "2", "4", "8", "16" } }},
    .arms = &.{"forward"},
    .ref = "forward",
    .opts = &.{.{ .name = "kv", .help = "tokens prefilled per request before each sample (default 4096)" }},
    .run = runMeter,
};

fn solo(xfm: *Trunk, req: *dsv.Request, ids: []const u32) !mlx.mlx_array {
    var ctx = xfm.defaultCtx();
    ctx.arch_ctx.dsv41.request = req;
    ctx.cache.step = req.position();
    const input = blk: {
        const side = mlx.engine.enter(xfm.outStream());
        defer side.end();
        break :blk try mlx.newData(ids.ptr, &.{ 1, @intCast(ids.len) }, .int32);
    };
    defer _ = mlx.mlx_array_free(input);
    return xfm.forwardWith(&ctx, input);
}

const Sample = struct {
    xfm: *Trunk,
    vocab: u32,
    rows: usize,
    kv: usize,
    steps: usize,
    verified: usize = 0,
    dispatches: f64 = 0,
    differing: usize = 0,
    logits_crc: u32 = 0,

    fn seed(self: *const Sample, req: *dsv.Request, i: usize) !void {
        var prng = std.Random.DefaultPrng.init(0x7541 + i);
        var ids: [2048]u32 = undefined;
        var left = self.kv;
        while (left > 0) {
            const chunk = ids[0..@min(left, ids.len)];
            for (chunk) |*id| id.* = prng.random().uintLessThan(u32, self.vocab);
            _ = mlx.mlx_array_free(try solo(self.xfm, req, chunk));
            left -= chunk.len;
        }
    }

    fn forward(self: *const Sample, reqs: []dsv.Request, ids: []const u32) !mlx.mlx_array {
        if (reqs.len == 1) return solo(self.xfm, &reqs[0], ids);
        var segs: [16]dsv.Seg = undefined;
        for (reqs, ids, segs[0..reqs.len]) |*req, *id, *sg| sg.* = .{ .request = req, .ids = @as(*const [1]u32, id) };
        return serving.forwardRows(self.xfm, segs[0..reqs.len]);
    }

    fn check(self: *Sample, logits: mlx.mlx_array, ids: []const u32) !void {
        try mlx.check(mlx.mlx_array_eval(logits));
        const stride = @as(usize, self.vocab) * @sizeOf(f32);
        if (mlx.mlx_array_dtype(logits) != .float32 or mlx.mlx_array_size(logits) != self.rows * self.vocab) return error.BadLogitShape;
        const bytes = mlx.mlx_array_data_uint8(logits) orelse return error.NoLogits;
        self.logits_crc = std.hash.Crc32.hash(bytes[0 .. self.rows * stride]);
        for (ids, 0..) |*id, i| {
            var req: dsv.Request = .{};
            defer req.deinit();
            try self.seed(&req, i);
            const one = try solo(self.xfm, &req, @as(*const [1]u32, id));
            defer _ = mlx.mlx_array_free(one);
            try mlx.check(mlx.mlx_array_eval(one));
            const want = mlx.mlx_array_data_uint8(one) orelse return error.NoLogits;
            if (!std.mem.eql(u8, want[0..stride], bytes[i * stride ..][0..stride])) self.differing += 1;
        }
        self.verified = self.rows * stride;
    }
};

fn sampleForward(s: *Sample, _: usize) !f64 {
    var reqs: [16]dsv.Request = @splat(.{});
    defer for (reqs[0..s.rows]) |*req| req.deinit();
    var ids: [16]u32 = undefined;
    var prng = std.Random.DefaultPrng.init(0xd541);
    for (reqs[0..s.rows], 0..) |*req, i| try s.seed(req, i);
    const profile = s.verified > 0 and @import("../tensor/core.zig").kernelProfOn();
    if (profile) mlx.engine.resetKernelProfile();
    const start = mlx.engine.stats();
    var clock = @import("../expert_stream.zig").Clock.init();
    var elapsed: u64 = 0;
    for (0..s.steps) |step| {
        for (ids[0..s.rows]) |*id| id.* = prng.random().uintLessThan(u32, s.vocab);
        const out = try s.forward(reqs[0..s.rows], ids[0..s.rows]);
        defer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_array_eval(out));
        elapsed += clock.lap();
        if (step == 0 and s.verified == 0) {
            try s.check(out, ids[0..s.rows]);
            _ = clock.lap();
        }
    }
    const steps: f64 = @floatFromInt(s.steps);
    if (profile) mlx.engine.logKernelProfile(steps);
    s.dispatches = @as(f64, @floatFromInt(mlx.engine.stats().dispatches - start.dispatches)) / steps;
    return @as(f64, @floatFromInt(elapsed)) / steps;
}

fn runMeter(run: *report.Run) !void {
    const a = std.heap.page_allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = run.pack.?;
    const kv = if (run.opt("kv")) |v| try std.fmt.parseUnsigned(usize, v, 10) else 4096;
    if (kv == 0) return error.BadOpt;
    var cfg = try model.parseConfig(io, a, dir);
    defer cfg.deinit(a);
    if (cfg.dsv41 == null or !cfg.dsv41.?.widthExactLayout()) return error.Dsv41RowsNotWidthExact;
    cfg.stream_mtp_head = cfg.dsv41.?.dspark.block > 0;
    const job = load.JobTurn.begin(null, &cfg);
    defer job.end();
    const tier = load.ramDefaultTierFor(&cfg) orelse return error.Dsv41RowsStreamed;
    var settle: @import("../bills.zig").SettleBudget = .{};
    const budget = load.ramDefaultFor(io, a, &cfg, dir, tier, false, 0, &settle);
    if (!budget.resident) return error.Dsv41RowsStreamed;
    const plan = try load.planExpertStreaming(io, a, &cfg, dir, budget.bytes, false);
    if (!plan.cache.resident) return error.Dsv41RowsStreamed;
    load.applyStreamingPlan(&cfg, plan, budget.bytes);
    cfg.expert_source_dir = try a.dupe(u8, dir);
    var weights = try model.loadWeightsForConfig(io, a, dir, &cfg, false);
    defer weights.deinit();
    var xfm = try load.initTrunk(io, a, &cfg, &weights);
    defer xfm.deinit();
    const m = @import("dsv41_state.zig").modelOf(&xfm).?;
    if (!m.serves_rows) return error.Dsv41RowsStreamed;
    for (run.axis("rows")) |v| {
        const rows = try std.fmt.parseUnsigned(usize, v, 10);
        if (rows == 0 or rows > m.maxJoinedRows()) return error.Dsv41RowsWidth;
        var sample: Sample = .{ .xfm = &xfm, .vocab = @intCast(m.cfg.vocab), .rows = rows, .kv = kv, .steps = run.steps(100) };
        const label = try std.fmt.allocPrint(run.a, "{d} requests, {d} keys", .{ rows, kv });
        try run.interleave(label, &sample, sampleForward);
        try run.metric("forward", "dispatches", "", sample.dispatches);
        try run.metric("forward", "solo logit bytes", "B", @floatFromInt(sample.verified));
        try run.metric("forward", "rows differing", "", @floatFromInt(sample.differing));
        try run.metric("forward", "logits CRC32", "", @floatFromInt(sample.logits_crc));
        if (sample.differing > 0) run.invalid(try std.fmt.allocPrint(run.a, "{s}: joined logits differ from solo", .{label}));
        try run.metric("forward", "aggregate", "tok/s", @as(f64, @floatFromInt(rows)) * 1e9 / run.lastMedianNs("forward"));
    }
}
