//! `sushi --version` report — the versions of the app and every embedded
//! engine, WITHOUT booting the HTTP server. The macOS app spawns
//! `sushi --version` as a one-shot subprocess and parses this so Settings
//! can show engine versions without a running server (Swift side:
//! `EngineVersions.parse`). Keep this a pure formatter — main.zig gathers the
//! runtime value (`mlx_version()`) and the build-time pins (`build_options`)
//! and calls `writeReport`. `writeGuestManifest` is the same idea in JSON
//! for `sushi --guest-manifest` (the release tarball's `guest.json`).

const std = @import("std");
const expert_exl3 = @import("sushi_exl3").format;

/// A host that runs sushi as an out-of-process engine refuses a `guest_api` it does not know.
pub const guest_api = 1;
/// The model types a host may route to this release line.
pub const guest_model_types = [_][]const u8{ "qwen4_exp", "mimo_v2", "glm5_next" };

/// The build facts `sushi --guest-manifest` reports (shipped as `guest.json` in the release tarball).
pub const GuestPins = struct {
    version: []const u8,
    /// Empty outside a release build (`-Dgit-sha`).
    commit: []const u8,
    /// MLX's own version, from `mlx_version()` at runtime.
    mlx: []const u8,
    mlx_sha: []const u8,
    mlx_c_sha: []const u8,
    min_macos: std.SemanticVersion,
};

pub fn writeGuestManifest(w: *std.Io.Writer, allocator: std.mem.Allocator, pins: GuestPins) !void {
    var arena_state: std.heap.ArenaAllocator = .init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const exl3_codebooks = try codebooksOf(arena, false);
    const stq_codebooks = try codebooksOf(arena, true);

    const v = pins.min_macos;
    const manifest = .{
        .version = pins.version,
        .commit = val(pins.commit),
        .guest_api = guest_api,
        .mlx = try std.fmt.allocPrint(arena, "v{s} @{s}", .{ pins.mlx, shortSha(pins.mlx_sha) }),
        .mlx_c = shortSha(pins.mlx_c_sha),
        .min_macos = if (v.patch == 0)
            try std.fmt.allocPrint(arena, "{d}.{d}", .{ v.major, v.minor })
        else
            try std.fmt.allocPrint(arena, "{d}.{d}.{d}", .{ v.major, v.minor, v.patch }),
        .model_types = &guest_model_types,
        .expert_quant = .{
            .format = "exl3",
            .codebooks = exl3_codebooks,
            .windows = [2]u32{ expert_exl3.Window.min_bits, 16 },
            .stq_codebooks = stq_codebooks,
        },
    };
    try std.json.Stringify.value(manifest, .{ .whitespace = .indent_2 }, w);
    try w.writeByte('\n');
}

/// The codebooks `parseExpertQuant` admits under format `stq` or, when `stq` is false, `exl3`: STQ is the ones that name a table.
fn codebooksOf(arena: std.mem.Allocator, stq: bool) ![]const []const u8 {
    var names: std.ArrayList([]const u8) = .empty;
    for (std.enums.values(expert_exl3.Codebook)) |cb| {
        if (cb.hasTable() == stq) try names.append(arena, @tagName(cb));
    }
    std.mem.sort([]const u8, names.items, {}, lessThan);
    return names.items;
}

fn shortSha(sha: []const u8) []const u8 {
    return val(sha[0..@min(sha.len, 7)]);
}

fn lessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

/// Every version string surfaced by `--version`. `mlx` comes from the linked
/// library at runtime; the rest are build-time pins.
pub const Info = struct {
    /// sushi app version (`build_options.version`).
    app: []const u8,
    /// MLX core, from `mlx_version()` at runtime.
    mlx: []const u8,
    /// mlx-c C bindings, the pinned submodule revision (no runtime API).
    mlx_c: []const u8,
    /// M5 NAX (neural accelerator) status: "on (...)" / "off (<reason>)",
    /// from `transformer.naxStatus()` (GPU gen + macOS floor; the bundled
    /// MLX always ships the NAX kernels — asserted at build time).
    nax: []const u8,
};

/// Render one `name value` line per component in a stable order. Machine-
/// parseable: the first whitespace-delimited token is the component name, the
/// remainder is its version (which may itself contain spaces, e.g.
/// `nax on (M5 neural accelerators)`). A pin with no value collapses to
/// `unknown` so every line always has a value token.
pub fn writeReport(w: *std.Io.Writer, info: Info) !void {
    try w.print("sushi {s}\n", .{val(info.app)});
    try w.print("mlx {s}\n", .{val(info.mlx)});
    try w.print("mlx-c {s}\n", .{val(info.mlx_c)});
    try w.print("nax {s}\n", .{val(info.nax)});
}

/// Allocate the report as a string (test/caller convenience).
pub fn report(allocator: std.mem.Allocator, info: Info) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    try writeReport(&out.writer, info);
    return allocator.dupe(u8, out.written());
}

/// A blank pin reads as `unknown` — never an empty value token, so the Swift
/// parser always gets `name` + `version`.
fn val(s: []const u8) []const u8 {
    return if (s.len == 0) "unknown" else s;
}

test "version: report renders one name-value line per component" {
    const s = try report(std.testing.allocator, .{
        .app = "1.0.0",
        .mlx = "0.32.0",
        .mlx_c = "0.6.0",
        .nax = "on (M5 neural accelerators)",
    });
    defer std.testing.allocator.free(s);
    try std.testing.expectEqualStrings(
        \\sushi 1.0.0
        \\mlx 0.32.0
        \\mlx-c 0.6.0
        \\nax on (M5 neural accelerators)
        \\
    , s);
}

test "version: blank pins read as unknown" {
    const s = try report(std.testing.allocator, .{
        .app = "1.0.0",
        .mlx = "0.32.0",
        .mlx_c = "", // build.sh couldn't resolve it (dev build)
        .nax = "",
    });
    defer std.testing.allocator.free(s);
    try std.testing.expectEqualStrings(
        \\sushi 1.0.0
        \\mlx 0.32.0
        \\mlx-c unknown
        \\nax unknown
        \\
    , s);
}

fn guestManifestFor(pins: GuestPins) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    try writeGuestManifest(&out.writer, std.testing.allocator, pins);
    return std.testing.allocator.dupe(u8, out.written());
}

test "version: the guest manifest names the build's pins and the pack contract it serves" {
    const s = try guestManifestFor(.{
        .version = "1.0.0",
        .commit = "3ba7272a3c0d9e1f",
        .mlx = "0.32.2",
        .mlx_sha = "1f8e74e3f12f",
        .mlx_c_sha = "56b2d39fc831",
        .min_macos = .{ .major = 26, .minor = 2, .patch = 0 },
    });
    defer std.testing.allocator.free(s);
    try std.testing.expectEqualStrings(
        \\{
        \\  "version": "1.0.0",
        \\  "commit": "3ba7272a3c0d9e1f",
        \\  "guest_api": 1,
        \\  "mlx": "v0.32.2 @1f8e74e",
        \\  "mlx_c": "56b2d39",
        \\  "min_macos": "26.2",
        \\  "model_types": [
        \\    "qwen4_exp",
        \\    "mimo_v2",
        \\    "glm5_next"
        \\  ],
        \\  "expert_quant": {
        \\    "format": "exl3",
        \\    "codebooks": [
        \\      "mcg",
        \\      "mul1"
        \\    ],
        \\    "windows": [
        \\      8,
        \\      16
        \\    ],
        \\    "stq_codebooks": [
        \\      "pair1",
        \\      "spec1",
        \\      "std1"
        \\    ]
        \\  }
        \\}
        \\
    , s);
}

test "version: a guest manifest from a non-release build says its commit is unknown" {
    const s = try guestManifestFor(.{
        .version = "1.0.0",
        .commit = "",
        .mlx = "0.32.2",
        .mlx_sha = "1f8e74e3f12f",
        .mlx_c_sha = "56b2d39fc831",
        .min_macos = .{ .major = 26, .minor = 2, .patch = 1 },
    });
    defer std.testing.allocator.free(s);
    try std.testing.expect(std.mem.indexOf(u8, s, "\"commit\": \"unknown\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "\"min_macos\": \"26.2.1\"") != null);
}

test "version: every guest model type is one the loader serves" {
    const served = @import("model.zig").served_model_types;
    for (guest_model_types) |t| {
        const found = for (served) |s| {
            if (std.mem.eql(u8, s, t)) break true;
        } else false;
        try std.testing.expect(found);
    }
}

test "version: the guest manifest lists exactly the (format, codebook) pairs the pack parser admits" {
    const t = std.testing;
    const s = try guestManifestFor(.{
        .version = "1.0.0",
        .commit = "",
        .mlx = "0.32.2",
        .mlx_sha = "1f8e74e3f12f",
        .mlx_c_sha = "56b2d39fc831",
        .min_macos = .{ .major = 26, .minor = 2, .patch = 0 },
    });
    defer t.allocator.free(s);
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, s, .{});
    defer parsed.deinit();
    const block = parsed.value.object.get("expert_quant").?.object;
    const listed = [_]struct { format: []const u8, key: []const u8 }{ .{ .format = "exl3", .key = "codebooks" }, .{ .format = "stq", .key = "stq_codebooks" } };
    for (listed) |entry| for (std.enums.values(expert_exl3.Codebook)) |cb| {
        var buf: [128]u8 = undefined;
        const raw = try std.fmt.bufPrint(&buf, "{{\"expert_quant\":{{\"format\":\"{s}\",\"k\":2,\"codebook\":\"{s}\"}}}}", .{ entry.format, @tagName(cb) });
        const config = try std.json.parseFromSlice(std.json.Value, t.allocator, raw, .{});
        defer config.deinit();
        // A table-keyed config without its table is still refused, but past the format gate.
        const admitted = if (@import("sushi_exl3").parseExpertQuant(config.value.object)) |_| true else |err| err != error.ExpertLayoutUnsupported;
        var in_manifest = false;
        if (block.get(entry.key)) |list| for (list.array.items) |name| {
            if (std.mem.eql(u8, name.string, @tagName(cb))) in_manifest = true;
        };
        try t.expectEqual(admitted, in_manifest);
    };
}
