//! Minimal pure-Zig `mlx_host` stub for the build-time `gen-metallib` tool.
//! Only provides the three top-level declarations `sushi_exl3` expects (`mlx`, `log`, `io_util`)
//! so `gen-metallib` can read `sushi_exl3`'s Metal source constants and codebook tables with
//! zero MLX, Metal, or Apple framework linkage.

pub const mlx = struct {
    pub const mlx_dtype = enum(u32) {
        bool_ = 0,
        uint8 = 1,
        uint16 = 2,
        uint32 = 3,
        uint64 = 4,
        int8 = 5,
        int16 = 6,
        int32 = 7,
        int64 = 8,
        float16 = 9,
        float32 = 10,
        float64 = 11,
        bfloat16 = 12,
        complex64 = 13,
    };
    pub const mlx_array = extern struct { ctx: ?*anyopaque = null };
    pub const mlx_stream = extern struct { ctx: ?*anyopaque = null };
    pub const mlx_fast_metal_kernel = extern struct { ctx: ?*anyopaque = null };
    pub const mlx_fast_metal_kernel_config = extern struct { ctx: ?*anyopaque = null };
    pub const mlx_vector_string = extern struct { ctx: ?*anyopaque = null };
    pub const mlx_vector_array = extern struct { ctx: ?*anyopaque = null };
};

pub const log = struct {};
pub const io_util = struct {};
