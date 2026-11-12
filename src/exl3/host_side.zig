//! The side of the host that makes arrays, for the module's process-wide caches of them. A host with an engine of
//! its own beside MLX (`mlx.engine`) has two; any other host has one, and nothing here asks more of it.
const mlx = @import("mlx_host").mlx;

const two = @hasDecl(mlx, "engine");

/// Arrays made between `enter(s)` and `end` belong to the side of stream `s`.
pub const Scope = if (two) @TypeOf(mlx.engine.enter(undefined)) else struct {
    pub fn end(_: @This()) void {}
};

pub fn enter(s: mlx.mlx_stream) Scope {
    return if (two) mlx.engine.enter(s) else .{};
}

/// The side of stream `s`: 1 the engine's, 0 MLX's.
pub fn of(s: mlx.mlx_stream) u1 {
    return if (two) @intFromBool(mlx.engine.isStream(s)) else 0;
}

/// The side an array made now, with no stream to say, belongs to.
pub fn making() u1 {
    return if (two) @intFromBool(mlx.engine.active) else 0;
}

/// A host array just made, or the error of a host that could not make it: MLX's maker never fails, an engine's may
/// (its pool refuses the buffer and hands back an empty handle). Nothing unchecked goes into a cache.
pub fn made(a: mlx.mlx_array) error{HostArrayRefused}!mlx.mlx_array {
    return if (a.ctx == null) error.HostArrayRefused else a;
}
