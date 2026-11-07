//! `arch_host.zig` for a host that carries only the services an arch needs: each name resolves to a file the host
//! supplies beside this one, under the same name. It declares exactly the names of `arch_host.zig`.
pub const log = @import("log.zig");
pub const io_util = @import("io_util.zig");
pub const mlx = @import("mlx.zig");
pub const nax = @import("nax_gate.zig");
pub const kernel_keys = @import("mlx_kernel_keys.zig");
pub const forward_diag = @import("forward_diag.zig");
pub const round_cost = @import("round_cost.zig");
pub const imatrix = @import("imatrix.zig");
pub const expert_stream = @import("expert_stream.zig");
pub const ExpertEngine = expert_stream.Engine;
pub const KVCache = @import("kv_cache.zig").KVCache;
pub const KVQuantConfig = @import("kv_quant.zig").KVQuantConfig;
pub const SSMCacheEntry = @import("ssm_cache.zig").SSMCacheEntry;
pub const disk_write_stats = @import("disk_write_stats.zig");
pub const vision_common = @import("vision_common.zig");
pub const request_media = @import("request_media.zig");

test "the stub declares exactly the names of the host" {
    const std = @import("std");
    const host = @import("arch_host.zig");
    const here = @This();
    inline for (comptime std.meta.declarations(host)) |name| try std.testing.expect(@hasDecl(here, name));
    inline for (comptime std.meta.declarations(here)) |name| try std.testing.expect(@hasDecl(host, name));
}
