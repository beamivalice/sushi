//! DeepSeek-V4.1's committed release config, MLX-free, for its own tests and the shared ones (`arch_config.fixtureConfig`,
//! the config pins), reached through its chat face.
const config_file = @import("dsv41_config.zig");

/// The release's `config.json`.
pub const config = config_file.release_config;

pub const config_pins = [_][2][]const u8{.{ "dsv41 release", config_file.release_config }};
