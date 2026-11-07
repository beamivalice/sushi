//! The EXL3 kernel microbenches `sushi-ubench` runs: a module of its own, so `sushi_exl3` carries no bench code.
//! Its `mlx_host` supplies `mlx`, `io_util` and `report`.
pub const ubench = @import("ubench.zig");
pub const bench_util = @import("bench_util.zig");
