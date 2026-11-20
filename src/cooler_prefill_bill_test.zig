//! The Sushi Cooler contract's prefill bill cells: what each kind of layer asks at its peak, and the chunks before
//! the tail. The runner and the shared helpers are in `cooler_contract_test.zig`.

const std = @import("std");
const testing = std.testing;
const mlx = @import("mlx.zig");
const arch_handle = @import("arch/handle.zig");
const arch_sched = @import("arch/sched.zig");
const prefill_plan = @import("prefill_plan.zig");
const contract = @import("cooler_contract_test.zig");
const Load = contract.Load;
const eachArch = contract.eachArch;
const eachFixture = contract.eachFixture;
const greedy = contract.greedy;


test "cooler contract: a Cooler prefill's chunks before the tail project no logits" {
    // An eager engine evaluates what MLX's lazy graph drops: a chunk's logits are rows x vocab x 2 bytes.
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            var ld: Load = undefined;
            try ld.open(C, dir, true);
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
            try ld.open(C, dir, true);
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
                    const bill = bills.prefillChunkCost(&ld.config, kv_bits, width, width);
                    // The margin covers the small tensors no term names, and no term may bill more than a margin it does not
                    // hold beyond the allowance the bill names: the rows' in-flight planes, and the f16 weight a decoded
                    // trunk linear holds at its own matmul (`bills.trunkDecodeScratchBytes`, NAX only), never at a chunk's
                    // peak: the stq4 cut's 8192-row attention chunk asks 1,933,856 KB on the M5 and on the M1 alike.
                    const allowance: u64 = (if (@hasDecl(C, "prefillAllowanceRows")) width * C.prefillAllowanceRows(&ld.config) else 0) +
                        bills.trunkDecodeScratchBytes(&ld.config, width);
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
