//! Hy4-Flash's committed configs, MLX-free, for its own tests and the shared ones (`arch_config.fixtureConfig`),
//! reached through its chat face.

/// The checkpoint's `config.json`.
pub const config = @import("hy4_config.zig").release_file;
