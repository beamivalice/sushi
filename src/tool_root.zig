//! The root of an offline tool an arch folder declares (`build_roots.zon` `.tools`, `tools` in its `arch.zig`). A tool
//! lives in its arch's folder and reaches the engine by relative path, which Zig allows only from a root file in `src/`;
//! this file is that root. It carries what the shared modules read from their root (`mlx_host`) and runs the tool the
//! build option names.
const std = @import("std");
const registry = @import("arch/registry.zig");
pub const mlx = @import("mlx.zig");
pub const log = @import("log.zig");
pub const io_util = @import("io_util.zig");

pub const main = blk: {
    const name = @import("build_options").tool;
    for (std.meta.tags(registry.Tag)) |t| {
        const M = registry.Mod(t);
        if (@hasDecl(M, "tools") and @hasDecl(M.tools, name)) break :blk @field(M.tools, name).main;
    }
    @compileError("no enabled arch builds the tool " ++ name);
};
