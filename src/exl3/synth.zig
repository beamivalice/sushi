//! Synthetic random banks for tests and the microbenches: test-only API, not part of the serving contract.
const std = @import("std");
const mlx = @import("mlx_host").mlx;
const exl3 = @import("expert_exl3.zig");
const root = @import("root.zig");

pub fn randomWords(rnd: std.Random, out: []u16) void {
    const w64: []align(1) u64 = std.mem.bytesAsSlice(u64, std.mem.sliceAsBytes(out[0 .. out.len / 4 * 4]));
    for (w64) |*v| v.* = rnd.int(u64);
    for (out[out.len / 4 * 4 ..]) |*v| v.* = @truncate(rnd.int(u32));
}

/// A bank of `experts` random trellises (gate, up, down at `n`'s rates) with sign-only scales of magnitude `suh_mag`
/// (the hidden-side input scale) and `svh_mag`; the caller frees it with `freeBank`.
pub fn makeBank(rnd: std.Random, hidden: usize, inter: usize, n: [3]u32, experts: u32, suh_mag: f32, svh_mag: f32) !root.Bank {
    var bank: root.Bank = undefined;
    const alloc = std.heap.page_allocator;
    const projs = [_]*root.Proj{ &bank.gate, &bank.up, &bank.down };
    for (projs, 0..) |p, pi| {
        const in_d: c_int = @intCast(if (pi == 2) inter else hidden);
        const out_d: c_int = @intCast(if (pi == 2) hidden else inter);
        const e: c_int = @intCast(experts);
        const shape = [_]c_int{ e, @divExact(in_d, 16), @divExact(out_d, 16), @intCast(n[pi]) };
        const words = try alloc.alloc(u16, @as(usize, @intCast(e * shape[1] * shape[2])) * n[pi]);
        defer alloc.free(words);
        randomWords(rnd, words);
        const su = try alloc.alloc(u16, @intCast(e * in_d));
        defer alloc.free(su);
        const sv = try alloc.alloc(u16, @intCast(e * out_d));
        defer alloc.free(sv);
        for (su) |*v| v.* = exl3.f32ToF16Bits(if (rnd.boolean()) suh_mag else -suh_mag);
        for (sv) |*v| v.* = exl3.f32ToF16Bits(if (rnd.boolean()) svh_mag else -svh_mag);
        p.* = .{
            .trellis = mlx.mlx_array_new_data(words.ptr, &shape, 4, .uint16),
            .suh = mlx.mlx_array_new_data(su.ptr, &[_]c_int{ e, in_d }, 2, .float16),
            .svh = mlx.mlx_array_new_data(sv.ptr, &[_]c_int{ e, out_d }, 2, .float16),
        };
    }
    return bank;
}

pub fn freeBank(b: *root.Bank) void {
    for ([_]*root.Proj{ &b.gate, &b.up, &b.down }) |p| {
        _ = mlx.mlx_array_free(p.trellis);
        _ = mlx.mlx_array_free(p.suh);
        _ = mlx.mlx_array_free(p.svh);
    }
}
