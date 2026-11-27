//! Unified Sushi Cooler precompiled custom-kernel catalog (`sushi.metallib`).
//! Aggregates shared kernels, EXL3 kernels, and every built architecture's metallib catalog
//! through `src/arch/registry.zig` (never naming any architecture here).
//!
//! Pure Zig — imported by:
//! - `src/metallib_gen_main.zig` (build-time `.metal` generator),
//! - `src/tensor/kernel.zig` (runtime resolution — zero Metal source compilation),
//! - `src/tensor/stock.zig` (completeness test verifying every function in `sushi.metallib` by construction).

const std = @import("std");
const registry = @import("arch/registry.zig");
const custom = @import("tensor/metallib_custom.zig");
const shared = @import("tensor/metallib_shared.zig");
const exl3 = @import("tensor/metallib_exl3.zig");

pub const Dtype = custom.Dtype;
pub const CtVal = custom.CtVal;
pub const Resolved = custom.Resolved;
pub const RuntimeInput = custom.RuntimeInput;
pub const RuntimeTarg = custom.RuntimeTarg;
pub const max_fc_ints = custom.max_fc_ints;
pub const max_fc_bools = custom.max_fc_bools;

/// Resolves a custom kernel launch against every registered Cooler kernel table.
/// Returns `true` if found and valid, `false` if unknown or unprecompiled (caller refuses by name).
pub fn resolve(
    name: []const u8,
    inputs: []const RuntimeInput,
    out_dtypes: []const Dtype,
    targs: []const RuntimeTarg,
    out: *Resolved,
) bool {
    if (custom.resolveSpecs(shared.specs, name, inputs, out_dtypes, targs, out)) return true;
    if (exl3.resolve(name, inputs, out_dtypes, targs, out)) return true;
    inline for (comptime std.meta.tags(registry.Tag)) |t| {
        if (comptime registry.hasMetallib(t)) {
            const M = registry.Metallib(t);
            if (custom.resolveSpecs(M.specs, name, inputs, out_dtypes, targs, out)) return true;
        }
    }
    return false;
}

/// Visits every precompiled custom kernel function name across shared, EXL3, and all built archs.
/// Driven by the exact same tables `gen-metallib` emits from — completeness by construction.
pub fn visitAll(ctx: anytype, comptime onName: fn (@TypeOf(ctx), []const u8) anyerror!void) !void {
    try custom.visitSpecs(shared.specs, ctx, onName);
    try exl3.visitFunctions(ctx, onName);
    inline for (comptime std.meta.tags(registry.Tag)) |t| {
        if (comptime registry.hasMetallib(t)) {
            try custom.visitSpecs(registry.Metallib(t).specs, ctx, onName);
        }
    }
}

/// Emits one custom `.metal` unit (`shared_{simd,nax}`, `exl3_*`, `arch_<name>_{simd,nax}`, `nax_small`),
/// optionally sharded via `_p<idx>of<count>` for parallel Metal compilation.
pub fn emitCustomUnit(out: *std.ArrayList(u8), a: std.mem.Allocator, unit_name: []const u8) !bool {
    const sh = custom.parseShardUnit(unit_name);
    if (std.mem.eql(u8, sh.base, "nax_small")) {
        // Combine small NAX units (shared_nax, exl3_gemm_nax, enabled arch_*_nax) into one `.air`
        // to avoid paying per-module `air-lld` link overhead on sub-second units.
        var lists: std.ArrayList([]const custom.KernelSpec) = .empty;
        defer lists.deinit(a);
        try lists.append(a, shared.specs);
        try lists.append(a, exl3.gemmNaxSpecs());
        inline for (comptime std.meta.tags(registry.Tag)) |t| {
            if (comptime registry.hasMetallib(t)) {
                try lists.append(a, registry.Metallib(t).specs);
            }
        }
        try custom.emitSpecListsSharded(out, a, true, lists.items, .by_spec, sh.idx, sh.count);
        return true;
    }
    if (std.mem.eql(u8, sh.base, "shared_simd")) {
        try custom.emitSpecsUnitSharded(out, a, shared.specs, false, .by_spec, sh.idx, sh.count);
        return true;
    }
    if (std.mem.eql(u8, sh.base, "shared_nax")) {
        try custom.emitSpecsUnitSharded(out, a, shared.specs, true, .by_spec, sh.idx, sh.count);
        return true;
    }
    if (std.mem.startsWith(u8, sh.base, "exl3_")) {
        return exl3.emitUnit(out, a, unit_name);
    }
    if (std.mem.startsWith(u8, sh.base, "arch_")) {
        const rest = sh.base["arch_".len..];
        const is_nax = std.mem.endsWith(u8, rest, "_nax");
        const is_simd = std.mem.endsWith(u8, rest, "_simd");
        if (!is_nax and !is_simd) return false;
        const arch_name = rest[0 .. rest.len - (if (is_nax) "_nax".len else "_simd".len)];
        inline for (comptime std.meta.tags(registry.Tag)) |t| {
            if (comptime registry.hasMetallib(t)) {
                if (std.mem.eql(u8, @tagName(t), arch_name)) {
                    try custom.emitSpecsUnitSharded(out, a, registry.Metallib(t).specs, is_nax, .by_spec, sh.idx, sh.count);
                    return true;
                }
            }
        }
    }
    return false;
}
