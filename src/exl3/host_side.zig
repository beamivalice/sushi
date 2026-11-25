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

/// Whether the host engine records a forward dry (`mlx.engine.dryOn`): its arrays then have shapes and no bytes, so a
/// process-wide cache that filled now would hand a later real forward a tensor with nothing behind it.
pub fn dry() bool {
    return if (two and @hasDecl(mlx.engine, "dryOn")) mlx.engine.dryOn() else false;
}

/// The side an array made now, with no stream to say, belongs to.
pub fn making() u1 {
    return if (two) @intFromBool(mlx.engine.active) else 0;
}

/// A host array just made, or the error of a host that could not make it: MLX's maker never fails, an engine's may
/// (its pool refuses the buffer and hands back an empty handle). Nothing unchecked goes into a cache (`side_cache`).
pub fn made(a: mlx.mlx_array) error{HostArrayRefused}!mlx.mlx_array {
    return if (a.ctx == null) error.HostArrayRefused else a;
}

const counted = @hasDecl(mlx, "kernelApplyCounted");

/// Words of the host engine's counter page, which a counted launch's last output is.
pub const counter_page_words: c_int = if (counted) mlx.engine.counter_page_words else 0;

/// The counters at the head of the host engine's counter page.
pub const counter_words: usize = if (counted) mlx.engine.counter_words else 0;

/// The counters a counted launch on stream `s` may use: the host engine's page's on its own stream, else none.
pub fn counters(s: mlx.mlx_stream) usize {
    return if (counted) (if (mlx.engine.enabled and mlx.engine.isStream(s)) counter_words else 0) else 0;
}

/// A launch whose last output is the host engine's counter page (`mlx.kernelApplyCounted`); null where the stream has none.
pub fn applyCounted(outputs: *mlx.mlx_vector_array, kernel: mlx.mlx_fast_metal_kernel, inputs: mlx.mlx_vector_array, cfg: mlx.mlx_fast_metal_kernel_config, s: mlx.mlx_stream) ?c_int {
    return if (counted) mlx.kernelApplyCounted(outputs, kernel, inputs, cfg, s) else null;
}
