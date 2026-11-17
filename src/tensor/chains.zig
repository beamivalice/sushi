//! Elementwise chains the engine runs as one kernel (`route.fusedChain`) where MLX runs a compiled closure or the ops.
//! Each expression rounds every step to the inputs' dtype as its op does, so a chain gives the ops' bytes: a bfloat
//! product keeps float precision even cast back to bfloat, so a step rounds as `$0(float(a) * float(b))`. Inputs read
//! as `x`, `y`, `z`; `$0` names their type. The ops' own forms stay at the call site, for MLX.

pub const Chain = struct { tag: []const u8, expr: []const u8 };

/// `x + sigmoid(y) * z`: the shared expert's gated add to the routed sum.
pub const sigmoid_gated_add: Chain = .{ .tag = "sigmoid_gated_add", .expr = "x + $0(float(sushi_sigmoid(y)) * float(z))" };
/// `2 * sigmoid(x)`: a hyper-connection's inject weights.
pub const twice_sigmoid: Chain = .{ .tag = "twice_sigmoid", .expr = "sushi_sigmoid(x) * $0(2)" };
/// `x + y * z`: a hyper-connection's write of a layer's output into the streams.
pub const multiply_add: Chain = .{ .tag = "multiply_add", .expr = "x + $0(float(y) * float(z))" };
/// `x * sigmoid(x)`.
/// The attention output gate: `x * sigmoid(y)`, rounding as the Sigmoid and Multiply ops do.
pub const sigmoid_gate: Chain = .{ .tag = "sigmoid_gate", .expr = "$0(float(x) * float(sushi_sigmoid(y)))" };
/// Qwen's PLE gate over `x` with `y` the scalar `1/sqrt(hidden)`: `sigmoid(sqrt(max(|g|, 1e-6)) * sign(g))`, `g = x*y`,
/// each step as the engine's op of it rounds (abs and sqrt in float, MLX's Maximum and Sign).
pub const ple_gate: Chain = .{
    .tag = "ple_gate",
    .expr = "sushi_sigmoid($0(float($0(metal::precise::sqrt(sushi_maximum($0(metal::abs(float($0(float(x) * float(y))))), $0(1e-6f))))) * " ++
        "float($0(($0(float(x) * float(y)) > $0(0)) - ($0(float(x) * float(y)) < $0(0))))))",
};
pub const silu: Chain = .{ .tag = "silu", .expr = "x * sushi_sigmoid(x)" };
