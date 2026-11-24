//! The Sushi Cooler contract's prefill bill cells: what each kind of layer asks at its peak, and the chunks before
//! the tail. The runner and the shared helpers are in `cooler_contract_test.zig`.

const std = @import("std");
const testing = std.testing;
const mlx = @import("mlx.zig");
const arch_handle = @import("arch/handle.zig");
const arch_sched = @import("arch/sched.zig");
const io = std.Io.Threaded.global_single_threaded.io();
const prefill_plan = @import("prefill_plan.zig");
const contract = @import("cooler_contract_test.zig");
const Load = contract.Load;
const eachArch = contract.eachArch;
const eachFixture = contract.eachFixture;
const greedy = contract.greedy;
const qwenTinyServed = contract.qwenTinyServed;


test "cooler contract: a Cooler prefill's chunks before the tail project no logits" {
    // An eager engine evaluates what MLX's lazy graph drops: a chunk's logits are rows x vocab x 2 bytes.
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            var ld: Load = undefined;
            try ld.openLight(C, dir, true);
            defer _ = ld.close();
            var long: [700]u32 = undefined;
            for (&long, 0..) |*t, i| t.* = ids[i % ids.len];
            const before = @import("trunk.zig").head_rows_for_test;
            var got = try greedy(&ld, &long, 2);
            got.deinit(testing.allocator);
            const rows = @import("trunk.zig").head_rows_for_test - before;
            // An arch with a head of its own projects none through the trunk's.
            if (rows == 0) return error.SkipZigTest;
            errdefer std.debug.print("cooler contract: a 700-token prompt and 2 tokens projected {d} head rows\n", .{rows});
            try testing.expect(rows <= prefill_plan.SSM_SNAPSHOT_BACKOFF + 1 + 2);
        }
    }.run);
}

test "cooler contract: an exact Cooler prefill bill is what each kind of layer asks at its peak, within the one margin either way" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            if (!@hasDecl(C, "prefill_kind_sets")) return error.SkipZigTest;
            const bills = @import("bills.zig");
            var ld: Load = undefined;
            try ld.openLight(C, dir, true);
            defer _ = ld.close();
            if (bills.coolerPrefillRowBytes(&ld.config) == 0) return error.SkipZigTest;
            const kv = bills.configuredKvQuantFor(&ld.config);
            const kv_bits: u64 = if (kv.scheme == .off) 16 else kv.bits;
            const a = testing.allocator;
            defer C.standInAllBut(null);
            defer bills.cooler_rows_for_test = null;
            defer bills.qsa_off_for_test = false;
            for (C.prefill_kind_sets) |kinds| {
                C.standInAllBut(kinds);
                bills.cooler_rows_for_test = C.prefillRowsOf(&ld.config, kinds);
                bills.qsa_off_for_test = !C.selectsBlocks(kinds);
                for ([_]usize{ 512, 8192 }) |width| {
                    const wide = try a.alloc(u32, width);
                    defer a.free(wide);
                    for (wide, 0..) |*t, i| t.* = ids[i % ids.len];
                    const grew = try chunkAsked(&ld, wide);
                    const bill = bills.prefillChunkCost(&ld.config, kv_bits, width, width, .{});
                    // The margin covers the small tensors no term names, and no term may bill more than a margin it does not
                    // hold beyond the allowance the bill names: the rows' in-flight planes, and what a layer's moments (a
                    // decoded trunk weight at its matmul, NAX only, the dense KV rebuild, and the MTP head's attention layer where the load
                    // runs the head) add past the rows' peak (`bills.forwardMomentsBilled`).
                    const allowance: u64 = (if (@hasDecl(C, "prefillAllowanceRows")) width * C.prefillAllowanceRows(&ld.config) else 0) +
                        bills.forwardMomentsBilled(&ld.config, kv_bits, width, width, bills.mtpHeadDefaultOn(&ld.config));
                    if (grew > bill + bills.BILL_MARGIN_BYTES or bill > grew + allowance + bills.BILL_MARGIN_BYTES) std.debug.print("cooler contract: a {d}-row chunk running {any} asked {d} KB against its {d} KB bill\n", .{ width, kinds, grew >> 10, bill >> 10 });
                    try testing.expect(grew <= bill + bills.BILL_MARGIN_BYTES);
                    try testing.expect(bill <= grew + allowance + bills.BILL_MARGIN_BYTES);
                }
            }
        }
    }.run);
}

/// The bytes the engine's pool has in use at its fullest over one prefill chunk of `ids` from an empty cache, past what
/// it had in use before.
fn chunkAsked(ld: *Load, ids: []const u32) !u64 {
    const turn = arch_handle.turnFor(ld.h, &ld.config);
    defer turn.end();
    defer arch_sched.endRequest(&ld.config);
    try ld.xfm.resetCache();
    ld.xfm.bindOwnRows(ids.len);
    mlx.engine.close();
    mlx.engine.trim();
    const in0 = mlx.engine.poolTotal() - mlx.engine.poolFree();
    mlx.engine.resetPoolPeak(.process);
    const arr = mlx.mlx_array_new_data(@ptrCast(ids.ptr), &[_]c_int{ 1, @intCast(ids.len) }, 2, .int32);
    defer _ = mlx.mlx_array_free(arr);
    var ctx = ld.xfm.defaultCtx();
    // A served chunk reads out no row of a prompt it continues.
    ctx.skip_lm_head = true;
    const hidden = try @import("model_handle.zig").forwardWith(ld.xfm, &ctx, arr);
    defer _ = mlx.mlx_array_free(hidden);
    try mlx.check(mlx.mlx_array_eval(hidden));
    mlx.engine.close();
    return mlx.engine.poolPeak(.process) -| in0;
}

/// One launch of the prefix cache, as `main` hands it to a load.
const CacheLaunch = struct { capacity: u32, ram: bool, disk: u64 };

/// What a load's prefill took and what its bill held, from the same launch: the checkpoints a 700-token prefill took at
/// the stride the scheduler's walk reads off the entry the launch attached (`walkCheckpointStride`), and the checkpoint
/// term of the request's bill.
fn checkpointsTakenAndBilled(ld: *Load, dir: []const u8, ids: []const u32, launch: CacheLaunch) !struct { taken: usize, billed: u64, walk: u32, bill_stride: u32 } {
    const a = testing.allocator;
    const bills = @import("bills.zig");
    const server_state = @import("server_state.zig");
    const registry_mod = @import("model_registry.zig");
    const saved = .{ server_state.prefix_cache_capacity, server_state.prefix_cache_ram_enabled, server_state.prefix_cache_disk_bytes };
    defer {
        server_state.prefix_cache_capacity = saved[0];
        server_state.prefix_cache_ram_enabled = saved[1];
        server_state.prefix_cache_disk_bytes = saved[2];
    }
    server_state.prefix_cache_capacity = launch.capacity;
    server_state.prefix_cache_ram_enabled = launch.ram;
    server_state.prefix_cache_disk_bytes = launch.disk;

    var reg = try registry_mod.ModelRegistry.init(a, io, null, 3, 0, null);
    defer reg.deinit();
    const entry = try reg.registerStub("contract", dir, 1);
    entry.config = &ld.config;
    entry.handle = ld.h;
    defer {
        if (entry.prefix_cache) |*hc| hc.deinit();
        entry.prefix_cache = null;
        entry.config = null;
        entry.handle = null;
    }
    // The stride `main` hands the load; no SSD tier is attached here, so a launch with one is out of this cell.
    std.debug.assert(launch.disk == 0);
    @import("model_load.zig").attachPrefixCache(a, io, entry, .{
        .capacity = launch.capacity,
        .ram_enabled = launch.ram,
        .mem_bytes = if (launch.ram) 1 << 30 else 0,
        .mem_resolver = null,
        .disk_resolver = null,
        .ssm_checkpoint_stride = @import("server.zig").effectiveSsmCheckpointStride(server_state.ssm_checkpoint_stride, launch.capacity, launch.ram, launch.disk),
        .ssm_checkpoint_max = server_state.ssm_checkpoint_max,
    }, launch.disk, .load);

    const walk = @import("scheduler.zig").walkCheckpointStride(entry);
    var long: [700]u32 = undefined;
    for (&long, 0..) |*t, i| t.* = ids[i % ids.len];
    const turn = arch_handle.turnFor(ld.h, &ld.config);
    defer turn.end();
    defer arch_sched.endRequest(&ld.config);
    var tok = @import("tokenizer.zig").Tokenizer.initEmpty(a);
    defer tok.deinit();
    try ld.xfm.resetCache();
    var gen = try @import("generate.zig").Generator.initWithOptions(io, a, ld.xfm, &tok, &long, 1, .{ .temperature = 0.0 }, &.{}, .{
        .ssm_checkpoint_stride = walk,
        .ssm_checkpoint_max = server_state.ssm_checkpoint_max,
    });
    defer gen.deinit(a);
    return .{
        .taken = gen.ssm_checkpoints.items.len,
        .billed = bills.retainedSsmCheckpointBytes(&ld.config, long.len, 0, 512),
        .walk = walk,
        .bill_stride = bills.launchCheckpointStride(&ld.config),
    };
}

test "cooler contract: a hybrid prefill with the prefix cache off takes no checkpoint and its bill holds none, and with a RAM cache both hold the ones it takes" {
    try qwenTinyServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const server_state = @import("server_state.zig");
            var ld: Load = undefined;
            try ld.openLight(Q, dir, true);
            defer _ = ld.close();
            const per_cp = ld.config.ssmCheckpointBytes();
            if (per_cp == 0) return error.SkipZigTest;
            // The launch settings the bill reads are the ones `main` handed the load.
            const saved = .{ server_state.prefix_cache_capacity, server_state.prefix_cache_ram_enabled, server_state.prefix_cache_disk_bytes };
            defer {
                server_state.prefix_cache_capacity = saved[0];
                server_state.prefix_cache_ram_enabled = saved[1];
                server_state.prefix_cache_disk_bytes = saved[2];
            }
            for ([_]CacheLaunch{
                .{ .capacity = 4, .ram = true, .disk = 0 },
                .{ .capacity = 0, .ram = true, .disk = 0 },
                .{ .capacity = 4, .ram = false, .disk = 0 },
            }) |launch| {
                const got = try checkpointsTakenAndBilled(&ld, dir, ids, launch);
                errdefer std.debug.print("cooler contract: capacity {d} ram {} disk {d}: the walk strides {d}, took {d}, the bill strides {d} and holds {d} B of {d} B a state\n", .{ launch.capacity, launch.ram, launch.disk, got.walk, got.taken, got.bill_stride, got.billed, per_cp });
                try testing.expectEqual(got.walk, got.bill_stride);
                if (launch.capacity > 0 and launch.ram) {
                    try testing.expect(got.taken > 0);
                    try testing.expect(got.billed >= got.taken * per_cp);
                } else {
                    try testing.expectEqual(@as(usize, 0), got.taken);
                    try testing.expectEqual(@as(u64, 0), got.billed);
                }
            }
        }
    }.run);
}

test "cooler contract: the load's sizer reserves at a width what a request's bill holds at that width, the MTP head's attention moment included" {
    try qwenTinyServed(struct {
        fn run(comptime Q: type, dir: []const u8, _: []const u32) !void {
            const bills = @import("bills.zig");
            var ld: Load = undefined;
            try ld.openLight(Q, dir, true);
            defer _ = ld.close();
            // A load that runs the head: the sizer prices every width for it.
            if (!bills.mtpHeadDefaultOn(&ld.config)) return error.SkipZigTest;
            const kv = bills.configuredKvQuantFor(&ld.config);
            const kv_bits: u64 = if (kv.scheme == .off) 16 else kv.bits;
            var wide = ld.config;
            wide.pinned_context = 1 << 20;
            var over: usize = 0;
            for ([_]u64{ 2048, 4096, 6000, 8192 }) |width| {
                const warm: bills.WarmPrefix = .{ .mtp_on = true };
                const terms = bills.prefillRequestTerms(&wide, width, 64, kv_bits, width, warm);
                const phases = bills.requestPhasesFor(&wide, width, 64, kv_bits, width, warm);
                // What the request holds through its prefill, past the chunk's own forward.
                const held = width * bills.kvBytesPerTokenAtBits(wide.kvBytesPerToken(), kv_bits) + terms.reserved_kv_bytes + terms.state_bytes +
                    terms.checkpoint_bytes + terms.qsa_ring_bytes + terms.mtp_head_kv_bytes + terms.dflash_bytes;
                const transient = phases.prefill - held;
                const reserve = bills.prefillTransientReserveAtKv(&wide, kv_bits, width, width);
                if (transient > reserve) {
                    over += 1;
                    std.debug.print("cooler contract: a {d}-row chunk's request transient is {d} B against the sizer's reserve of {d} B\n", .{ width, transient, reserve });
                }
            }
            try testing.expectEqual(@as(usize, 0), over);
        }
    }.run);
}
