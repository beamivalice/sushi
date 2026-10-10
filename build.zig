const std = @import("std");
const builtin = @import("builtin");

comptime {
    // 0.16.0's bundled libc++ fails to compile against the macOS 27 SDK
    // (`use of undeclared identifier 'INFINITY'` in its vendored <random>).
    if (builtin.zig_version.major == 0 and builtin.zig_version.minor < 17) {
        @compileError(std.fmt.comptimePrint(
            "sushi requires Zig 0.17 (have {d}.{d}.{d}). Run ./scripts/fetch-zig.sh, or grab 0.17.0 from https://ziglang.org/download/.",
            .{ builtin.zig_version.major, builtin.zig_version.minor, builtin.zig_version.patch },
        ));
    }
}

pub fn build(b: *std.Build) void {
    // Match libmlx's macOS 26.2 deployment target, required by its NAX kernels.
    const target = b.standardTargetOptions(.{
        .default_target = .{
            .os_version_min = .{ .semver = .{ .major = 26, .minor = 2, .patch = 0 } },
        },
    });
    const optimize = b.standardOptimizeOption(.{});

    // Setting any non-default target field disables Zig's native macOS SDK detection,
    // so we resolve the SDK path ourselves and surface its frameworks dir.
    const macos_sdk_frameworks: ?[]const u8 = blk: {
        if (target.result.os.tag != .macos) break :blk null;
        var code: u8 = undefined;
        const stdout = b.runAllowFail(
            &.{ "xcrun", "--sdk", "macosx", "--show-sdk-path" },
            &code,
            .inherit,
        ) catch break :blk null;
        const sdk = std.mem.trim(u8, stdout, " \n\r\t");
        if (sdk.len == 0) break :blk null;
        break :blk b.fmt("{s}/System/Library/Frameworks", .{sdk});
    };

    if (target.result.os.tag == .macos) {
        verifyBrewDeps(b);
        verifyMlxStage(b);
    }

    if (builtin.os.tag != .macos) return;

    // Version: SemVer, build.zig.zon's `.version` unless the release workflow
    // passes the tag's version (release.sh checks the two agree).
    const version = b.option([]const u8, "version", "SemVer version string (default: build.zig.zon)") orelse @import("build.zig.zon").version;
    _ = std.SemanticVersion.parse(version) catch {
        std.debug.print("[sushi] -Dversion={s} is not a SemVer version (MAJOR.MINOR.PATCH[-pre])\n", .{version});
        std.process.exit(1);
    };

    // MLX reports its version at runtime; mlx-c and the guest manifest need build-time pins.
    const mlx_c_version = b.option([]const u8, "mlx-c-version", "Pinned mlx-c version") orelse readMlxPin(b, "mlxc=") orelse "unknown";
    const mlx_sha = readMlxPin(b, "mlx=") orelse "";

    const build_options = b.addOptions();
    build_options.addOption([]const u8, "version", version);
    build_options.addOption([]const u8, "mlx_c_version", mlx_c_version);
    build_options.addOption([]const u8, "mlx_sha", mlx_sha);
    const git_sha = b.option([]const u8, "git-sha", "Engine build id for the round-cost table: a release sha stands for the executable bytes, which are then not hashed; the MLX dylib and metallib fingerprints are always mixed in") orelse "";
    build_options.addOption([]const u8, "git_sha", git_sha);

    const mod = b.createModule(.{
        .root_source_file = b.path(if (b.option(bool, "glm-ffn-replay", "Build offline GLM frozen-prefix extractor") orelse false) "src/glm5_ffn_replay.zig" else if (b.option(bool, "glm-trunk-cov", "Build offline GLM trunk input-covariance extractor") orelse false) "src/glm5_trunk_cov.zig" else "src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libcpp = true,
        .imports = &.{
            .{ .name = "build_options", .module = build_options.createModule() },
            .{ .name = "jinja_c", .module = addCHeaderModule(b, b.path("lib/jinja_cpp/jinja_wrapper.h"), b.path("lib/jinja_cpp"), target, optimize) },
            .{ .name = "stb", .module = addCHeaderModule(b, b.path("lib/stb_image.h"), b.path("lib"), target, optimize) },
            .{ .name = "webp", .module = addCHeaderModule(b, .{ .cwd_relative = "/opt/homebrew/include/webp/decode.h" }, .{ .cwd_relative = "/opt/homebrew/include" }, target, optimize) },
        },
    });

    // Jinja2 template engine (wangzhaode/jinja.cpp + nlohmann/json; see NOTICE).
    // Precompiled with system clang++ for C++17's system libc++; rebuild instructions in CLAUDE.md.
    mod.addObjectFile(b.path("lib/jinja_cpp/libjinja.a"));
    mod.addIncludePath(b.path("lib/jinja_cpp"));

    // stb_image for JPEG/PNG decoding in the vision pipeline
    mod.addCSourceFile(.{ .file = b.path("lib/stb_image_impl.c"), .flags = &.{"-O2"} });
    mod.addCSourceFile(.{ .file = b.path("lib/dflash_cache_space.c"), .flags = &.{"-O2"} });
    mod.addCSourceFile(.{ .file = b.path("lib/volume_space.m"), .flags = &.{ "-O2", "-fobjc-arc" } });
    mod.addIncludePath(b.path("lib"));

    // The staged MLX library path must precede Homebrew's.
    addMlxLib(b, mod);
    _ = addExl3Module(b, mod, target, optimize);
    // webp include/lib paths (homebrew)
    mod.addIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });
    mod.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });
    mod.linkSystemLibrary("webp", .{});

    if (macos_sdk_frameworks) |fw_path| {
        mod.addFrameworkPath(.{ .cwd_relative = fw_path });
    }
    mod.linkFramework("IOKit", .{});
    mod.linkFramework("CoreFoundation", .{});
    mod.linkFramework("Foundation", .{});
    mod.linkFramework("Metal", .{});
    mod.linkFramework("IOSurface", .{});

    const exe = b.addExecutable(.{
        .name = "sushi",
        .root_module = mod,
    });

    // Ensure Mach-O header has room for install_name_tool path changes — the
    // release tarball rewires @rpath/libmlxc.dylib to @executable_path.
    exe.headerpad_max_install_names = true;

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();

    const run_step = b.step("run", "Run sushi");
    run_step.dependOn(&run_cmd.step);

    // Unit tests — reuses the same module config (mlx-c, jinja_cpp, etc.)
    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/tests.zig"),
        .target = target,
        .optimize = optimize,
        .link_libcpp = true,
        .imports = &.{
            .{ .name = "build_options", .module = build_options.createModule() },
            .{ .name = "jinja_c", .module = addCHeaderModule(b, b.path("lib/jinja_cpp/jinja_wrapper.h"), b.path("lib/jinja_cpp"), target, optimize) },
            .{ .name = "stb", .module = addCHeaderModule(b, b.path("lib/stb_image.h"), b.path("lib"), target, optimize) },
            .{ .name = "webp", .module = addCHeaderModule(b, .{ .cwd_relative = "/opt/homebrew/include/webp/decode.h" }, .{ .cwd_relative = "/opt/homebrew/include" }, target, optimize) },
        },
    });

    test_mod.addObjectFile(b.path("lib/jinja_cpp/libjinja.a"));
    test_mod.addIncludePath(b.path("lib/jinja_cpp"));
    test_mod.addCSourceFile(.{ .file = b.path("lib/stb_image_impl.c"), .flags = &.{"-O2"} });
    test_mod.addCSourceFile(.{ .file = b.path("lib/dflash_cache_space.c"), .flags = &.{"-O2"} });
    test_mod.addCSourceFile(.{ .file = b.path("lib/volume_space.m"), .flags = &.{ "-O2", "-fobjc-arc" } });
    test_mod.addIncludePath(b.path("lib"));
    test_mod.linkSystemLibrary("c++", .{});
    addMlxLib(b, test_mod);
    const exl3_test_mod = addExl3Module(b, test_mod, target, optimize);
    test_mod.addIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });
    test_mod.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });
    test_mod.linkSystemLibrary("webp", .{});

    if (macos_sdk_frameworks) |fw_path| {
        test_mod.addFrameworkPath(.{ .cwd_relative = fw_path });
    }
    test_mod.linkFramework("IOKit", .{});
    test_mod.linkFramework("CoreFoundation", .{});
    test_mod.linkFramework("Foundation", .{});
    test_mod.linkFramework("Metal", .{});
    test_mod.linkFramework("IOSurface", .{});

    const test_filter = b.option([]const u8, "test-filter", "Only run tests whose name contains this substring");
    const qwen_preprocess_fixture = b.option(
        []const u8,
        "qwen-preprocess-fixture",
        "CPU reference fixture for the gated Qwen preprocessing parity test",
    );
    const test_build = b.step("test-build", "Compile unit tests without running them");
    const test_step = b.step("test", "Run unit tests");
    // A filtered run stays one binary: bucket filters OR with the user's, they cannot narrow it.
    const unit_tests: []const *std.Build.Step.Compile = if (test_filter) |f|
        b.allocator.dupe(*std.Build.Step.Compile, &.{b.addTest(.{ .root_module = test_mod, .filters = &.{f} })}) catch @panic("OOM")
    else
        addTestBuckets(b, test_mod);
    for (unit_tests) |unit_test| {
        test_build.dependOn(&b.addInstallArtifact(unit_test, .{ .dest_dir = .{ .override = .{ .custom = "tests" } } }).step);
        const run_unit_tests = b.addRunArtifact(unit_test);
        if (qwen_preprocess_fixture) |fixture| {
            run_unit_tests.setEnvironmentVariable("QWEN_PREPROCESS_FIXTURE", fixture);
            run_unit_tests.addFileInput(.{ .cwd_relative = b.fmt("{s}/manifest.json", .{fixture}) });
            run_unit_tests.addFileInput(.{ .cwd_relative = b.fmt("{s}/source_rgb.bin", .{fixture}) });
            run_unit_tests.addFileInput(.{ .cwd_relative = b.fmt("{s}/pixel_values.bin", .{fixture}) });
        }
        test_step.dependOn(&run_unit_tests.step);
    }

    // Zig collects tests from an artifact's root module only, so src/exl3 is its own.
    const exl3_tests = b.addTest(.{
        .name = "exl3-test",
        .root_module = exl3_test_mod,
        .filters = if (test_filter) |f| &.{f} else &.{},
    });
    test_build.dependOn(&b.addInstallArtifact(exl3_tests, .{ .dest_dir = .{ .override = .{ .custom = "tests" } } }).step);
    test_step.dependOn(&b.addRunArtifact(exl3_tests).step);
}

/// The unit tests compile as several binaries in parallel: one LLVM module is single-threaded.
/// A test's name starts with `<file stem>.`, so a bucket is a list of stems used as filters, and
/// the last bucket takes every src/*.zig stem the others do not name.
const test_buckets = [_][]const []const u8{
    &.{ "transformer", "gdn_decode", "cli", "repl_tools", "repl_input", "update", "launch" },
    &.{ "generate", "scheduler", "prefix_cache", "kv_disk_cache", "kv_disk_writer", "restore_dump", "mtp", "dflash" },
    &.{ "server", "chat", "format_corpus_test", "tool_traffic_replay_test", "responses", "ws", "reasoning_protocol", "json_grammar", "json_schema", "regex", "token_mask", "tokenizer" },
};

fn addTestBuckets(b: *std.Build, root: *std.Build.Module) []const *std.Build.Step.Compile {
    const io = b.graph.io;
    var stems: std.ArrayList([]const u8) = .empty;
    var src = buildRootHandle(b).openDir(io, "src", .{ .iterate = true }) catch |e| std.debug.panic("src/: {t}", .{e});
    defer src.close(io);
    var it = src.iterate();
    while (it.next(io) catch |e| std.debug.panic("src/: {t}", .{e})) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".zig")) continue;
        stems.append(b.allocator, b.fmt("{s}.", .{entry.name[0 .. entry.name.len - ".zig".len]})) catch @panic("OOM");
    }
    // Directory order varies; the filters key the compile step's cache.
    std.mem.sortUnstable([]const u8, stems.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
    const bucket = b.allocator.alloc(usize, stems.items.len) catch @panic("OOM");
    for (stems.items, bucket) |stem, *k| k.* = namedBucket(stem[0 .. stem.len - 1]);
    // A filter matches anywhere in a name, so "dflash." also selects glm5_dflash's tests: stems
    // that contain one another must share a bucket, or their tests would run twice.
    var moved = true;
    while (moved) {
        moved = false;
        for (stems.items, bucket) |a, *ka| {
            for (stems.items, bucket) |s, *ks| {
                if (ka.* == ks.* or std.mem.indexOf(u8, a, s) == null) continue;
                if (ka.* != test_buckets.len and ks.* != test_buckets.len)
                    std.debug.panic("test buckets {d} and {d} split overlapping stems {s} and {s}", .{ ka.*, ks.*, a, s });
                ka.* = @min(ka.*, ks.*);
                ks.* = ka.*;
                moved = true;
            }
        }
    }
    var out: std.ArrayList(*std.Build.Step.Compile) = .empty;
    for (0..test_buckets.len + 1) |i| {
        var filters: std.ArrayList([]const u8) = .empty;
        for (stems.items, bucket) |stem, k| if (k == i) filters.append(b.allocator, stem) catch @panic("OOM");
        if (filters.items.len == 0) continue;
        const name = if (i < test_buckets.len) b.fmt("test-{s}", .{test_buckets[i][0]}) else "test-rest";
        out.append(b.allocator, b.addTest(.{ .name = name, .root_module = root, .filters = filters.items })) catch @panic("OOM");
    }
    return out.items;
}

fn namedBucket(stem: []const u8) usize {
    for (test_buckets, 0..) |names, i| {
        for (names) |name| if (std.mem.eql(u8, stem, name)) return i;
    }
    return test_buckets.len;
}

fn addCHeaderModule(
    b: *std.Build,
    header_path: std.Build.LazyPath,
    include_dir: std.Build.LazyPath,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Module {
    const translate = b.addTranslateC(.{
        .root_source_file = header_path,
        .target = target,
        .optimize = optimize,
    });
    translate.addIncludePath(include_dir);
    return translate.createModule();
}

fn buildRootHandle(b: *std.Build) std.Io.Dir {
    return b.root.root_dir.handle;
}

/// src/exl3: the EXL3 expert engine, a module so another MLX host (mlx-serve)
/// can root it too. It reaches mlx, log and io_util through `mlx_host`, so the
/// host root file must expose them as `pub const`.
fn addExl3Module(b: *std.Build, host: *std.Build.Module, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const exl3 = b.createModule(.{
        .root_source_file = b.path("src/exl3/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "mlx_host", .module = host }},
    });
    host.addImport("sushi_exl3", exl3);
    return exl3;
}

/// Link the NAX-enabled libraries staged by scripts/build-mlx.sh.
/// Release packaging rewrites their @rpath install names to @executable_path.
fn addMlxLib(b: *std.Build, module: *std.Build.Module) void {
    module.addIncludePath(b.path("lib/mlx/include"));
    module.addLibraryPath(b.path("lib/mlx/lib"));
    // Prevent Homebrew's mlx-c.pc from overriding the staged libraries.
    module.linkSystemLibrary("mlxc", .{ .use_pkg_config = .no });
    // Binary-relative paths cover zig-out/bin and .zig-cache/o/<hash> respectively.
    module.addRPath(.{ .cwd_relative = "@loader_path/../../lib/mlx/lib" });
    module.addRPath(.{ .cwd_relative = "@loader_path/../../../lib/mlx/lib" });
}

/// Fail before linking if the local MLX stage is missing.
fn verifyMlxStage(b: *std.Build) void {
    const stage_ok = blk: {
        buildRootHandle(b).access(b.graph.io, "lib/mlx/lib/libmlxc.dylib", .{}) catch break :blk false;
        buildRootHandle(b).access(b.graph.io, "lib/mlx/lib/mlx.metallib", .{}) catch break :blk false;
        buildRootHandle(b).access(b.graph.io, "lib/mlx/.version", .{}) catch break :blk false;
        break :blk true;
    };
    if (!stage_ok) {
        std.debug.print(
            "\n[sushi] lib/mlx is not staged (self-built mlx + mlx-c). Run:\n" ++
                "  git submodule update --init lib/mlx-src lib/mlxc-src && ./scripts/build-mlx.sh\n\n",
            .{},
        );
        std.process.exit(1);
    }
}

/// A pinned revision (`key` "mlx=" or "mlxc=") from lib/mlx/.version, written
/// by scripts/build-mlx.sh as "mlx=<sha> mlxc=<sha> target=<ver>". Returns
/// null when not staged yet.
fn readMlxPin(b: *std.Build, key: []const u8) ?[]const u8 {
    const bytes = buildRootHandle(b).readFileAlloc(
        b.graph.io,
        "lib/mlx/.version",
        b.allocator,
        .limited(256),
    ) catch return null;
    var it = std.mem.tokenizeScalar(u8, std.mem.trim(u8, bytes, " \t\r\n"), ' ');
    while (it.next()) |tok| {
        if (std.mem.startsWith(u8, tok, key)) return b.dupe(tok[key.len..]);
    }
    return null;
}

const BrewDep = struct { name: []const u8, min: std.SemanticVersion };

const required_brew_deps = [_]BrewDep{
    .{ .name = "webp", .min = .{ .major = 1, .minor = 6, .patch = 0 } },
};

fn verifyBrewDeps(b: *std.Build) void {
    for (required_brew_deps) |dep| {
        var code: u8 = undefined;
        const stdout = b.runAllowFail(
            &.{ "brew", "list", "--versions", dep.name },
            &code,
            .inherit,
        ) catch {
            std.debug.print(
                "\n[sushi] missing Homebrew dependency '{s}' (>= {d}.{d}.{d}). Install with: brew install webp\n\n",
                .{ dep.name, dep.min.major, dep.min.minor, dep.min.patch },
            );
            std.process.exit(1);
        };
        const trimmed = std.mem.trim(u8, stdout, " \n\r\t");
        const space = std.mem.indexOfScalar(u8, trimmed, ' ') orelse {
            std.debug.print("[sushi] cannot parse `brew list --versions {s}` output: {s}\n", .{ dep.name, trimmed });
            std.process.exit(1);
        };
        var ver_str = trimmed[space + 1 ..];
        // Strip Homebrew revision suffix (e.g., "0.6.0_2" -> "0.6.0").
        if (std.mem.indexOfScalar(u8, ver_str, '_')) |us| ver_str = ver_str[0..us];
        const have = std.SemanticVersion.parse(ver_str) catch {
            std.debug.print("[sushi] cannot parse '{s}' version '{s}'\n", .{ dep.name, ver_str });
            std.process.exit(1);
        };
        if (have.order(dep.min) == .lt) {
            std.debug.print(
                "\n[sushi] Homebrew '{s}' is {d}.{d}.{d}; need >= {d}.{d}.{d}. Run: brew upgrade {s}\n\n",
                .{ dep.name, have.major, have.minor, have.patch, dep.min.major, dep.min.minor, dep.min.patch, dep.name },
            );
            std.process.exit(1);
        }
    }
}
