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

/// Emits one custom `.metal` unit (`shared_simd`, `shared_nax`, `exl3_*`, `arch_<name>_{simd,nax}`).
pub fn emitCustomUnit(out: *std.ArrayList(u8), a: std.mem.Allocator, unit_name: []const u8) !bool {
    if (std.mem.eql(u8, unit_name, "shared_simd")) {
        try custom.emitSpecsUnit(out, a, shared.specs, false);
        return true;
    }
    if (std.mem.eql(u8, unit_name, "shared_nax")) {
        try custom.emitSpecsUnit(out, a, shared.specs, true);
        return true;
    }
    if (std.mem.startsWith(u8, unit_name, "exl3_")) {
        return exl3.emitUnit(out, a, unit_name);
    }
    if (std.mem.startsWith(u8, unit_name, "arch_")) {
        const rest = unit_name["arch_".len..];
        const is_nax = std.mem.endsWith(u8, rest, "_nax");
        const is_simd = std.mem.endsWith(u8, rest, "_simd");
        if (!is_nax and !is_simd) return false;
        const arch_name = rest[0 .. rest.len - (if (is_nax) "_nax".len else "_simd".len)];
        inline for (comptime std.meta.tags(registry.Tag)) |t| {
            if (comptime registry.hasMetallib(t)) {
                if (std.mem.eql(u8, @tagName(t), arch_name)) {
                    try custom.emitSpecsUnit(out, a, registry.Metallib(t).specs, is_nax);
                    return true;
                }
            }
        }
    }
    return false;
}
