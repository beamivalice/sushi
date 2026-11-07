//! Per-request state of the loaded arch. Inference thread only.
const registry = @import("registry.zig");
const Arch = registry.Arch;

/// Whether the model keeps decode state of its own, so admission lets one request in at a time.
pub fn ownsModuleDecodeState(a: Arch) bool {
    return switch (a) {
        inline else => |m, t| registry.Mod(t).ownsModuleDecodeState(m),
    };
}
