//! Flash-Next on the offline tools: the names its source checkpoint gives the imatrix, the width of a hidden-capture
//! row, and the state it sets up once a KLD model is loaded.
const std = @import("std");
const qwen4_forward = @import("qwen4_forward.zig");
const model = @import("../model.zig");
const imatrix = @import("../imatrix.zig");
const trunk_mod = @import("../trunk.zig");

pub const imatrix_naming = imatrix.Naming{ .layer_prefix = "model.language_model.layers." };

/// Block boundaries precede the final mixer and keep every hyper-connection stream.
pub fn hiddenCaptureWidth(config: *const model.ModelConfig) usize {
    return @as(usize, config.hidden_size) * @max(config.hc_count, 1);
}

pub fn afterLoad(xfm: *trunk_mod.Trunk, config: model.ModelConfig, kv: trunk_mod.KVQuantConfig) !void {
    _ = config;
    try qwen4_forward.qwen4MtpApplyKvQuant(xfm, kv);
    qwen4_forward.compileQwen4Hc(xfm);
}

test "a hidden-capture row is the hidden size times every hyper-connection stream" {
    const config = model.ModelConfig{ .model_type = "qwen4_exp", .hidden_size = 2560, .hc_count = 4 };
    try std.testing.expectEqual(@as(usize, 10240), hiddenCaptureWidth(&config));
    const plain = model.ModelConfig{ .model_type = "qwen4_exp", .hidden_size = 2560 };
    try std.testing.expectEqual(@as(usize, 2560), hiddenCaptureWidth(&plain));
}
