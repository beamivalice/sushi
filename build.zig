const std = @import("std");
const builtin = @import("builtin");

comptime {
    // 0.16.0's bundled libc++ fails to compile against the macOS 27 SDK
    // (`use of undeclared identifier 'INFINITY'` in its vendored <random>).
    if (builtin.zig_version.major == 0 and builtin.zig_version.minor < 17) {
        @compileError(std.fmt.comptimePrint(
            "sushi requires Zig 0.17 (have {d}.{d}.{d}). Run ./scripts/fetch-zig.sh, or grab 0.17.1 from https://ziglang.org/download/.",
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

    const build_roots = archBuildRoots(b);
    const tool = toolRoot(b, build_roots);
    const build_options = b.addOptions();
    build_options.addOption([]const u8, "tool", tool orelse "");
    build_options.addOption([]const u8, "version", version);
    build_options.addOption([]const u8, "mlx_c_version", mlx_c_version);
    build_options.addOption([]const u8, "mlx_sha", mlx_sha);
    const git_sha = b.option([]const u8, "git-sha", "Engine build id for the round-cost table: a release sha stands for the executable bytes, which are then not hashed; the MLX dylib and metallib fingerprints are always mixed in") orelse "";
    build_options.addOption([]const u8, "git_sha", git_sha);
    const archs = archList(b);
    build_options.addOption([]const []const u8, "archs", archs);
    build_options.addOption(bool, "host_tier", false);
    // The host test root links no MLX, so gpu_mem.zig refuses a live GPU read under it.
    const host_options = b.addOptions();
    host_options.addOption([]const u8, "version", version);
    host_options.addOption([]const u8, "mlx_c_version", mlx_c_version);
    host_options.addOption([]const u8, "mlx_sha", mlx_sha);
    host_options.addOption([]const u8, "git_sha", git_sha);
    host_options.addOption([]const []const u8, "archs", archs);
    host_options.addOption(bool, "host_tier", true);

    const mod = b.createModule(.{
        .root_source_file = b.path(if (tool != null) "src/tool_root.zig" else "src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "build_options", .module = build_options.createModule() },
            .{ .name = "stb", .module = addCHeaderModule(b, b.path("lib/stb_image.h"), b.path("lib"), target, optimize) },
            .{ .name = "webp", .module = addCHeaderModule(b, .{ .cwd_relative = "/opt/homebrew/include/webp/decode.h" }, .{ .cwd_relative = "/opt/homebrew/include" }, target, optimize) },
        },
    });

    // stb_image for JPEG/PNG decoding in the vision pipeline
    mod.addCSourceFile(.{ .file = b.path("lib/stb_image_impl.c"), .flags = &.{"-O2"} });
    mod.addCSourceFile(.{ .file = b.path("lib/dflash_cache_space.c"), .flags = &.{"-O2"} });
    mod.addCSourceFile(.{ .file = b.path("lib/volume_space.m"), .flags = &.{ "-O2", "-fobjc-arc" } });
    mod.addIncludePath(b.path("lib"));

    // The staged MLX library path must precede Homebrew's.
    addMlxLib(b, mod);
    const exl3_mod = addExl3Module(b, mod, target, optimize);
    mod.addImport("sushi_encoder", addEncoderModule(b, mod, exl3_mod, target, optimize));
    // webp include/lib paths (homebrew)
    mod.addIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });
    mod.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });
    mod.linkSystemLibrary("webp", .{});

    linkApple(mod, macos_sdk_frameworks);

    const exe = b.addExecutable(.{
        .name = "sushi",
        .root_module = mod,
    });

    // Ensure Mach-O header has room for install_name_tool path changes — the
    // release tarball rewires @rpath/libmlxc.dylib to @executable_path.
    exe.headerpad_max_install_names = true;

    b.installArtifact(exe);
    b.installArtifact(addSushibar(b, target, optimize, macos_sdk_frameworks));

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
    const encoder_mod = addEncoderModule(b, test_mod, exl3_test_mod, target, optimize);
    test_mod.addImport("sushi_encoder", encoder_mod);
    test_mod.addIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });
    test_mod.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });
    test_mod.linkSystemLibrary("webp", .{});

    linkApple(test_mod, macos_sdk_frameworks);

    const test_filter = b.option([]const []const u8, "test-filter", "Only run tests whose name contains this substring (repeat for several)");
    const test_model = b.option(TestModel, "model", "Run one model folder's tests plus the shared ones (qwen4, mimo2, glm5, dsv41), or only the exl3 or encoder module's");
    const qwen_preprocess_fixture = b.option(
        []const u8,
        "qwen-preprocess-fixture",
        "CPU reference fixture for the gated Qwen preprocessing parity test",
    );
    const test_build = b.step("test-build", "Compile unit tests without running them");
    const test_step = b.step("test", "Run unit tests");
    // The host root links no MLX: a file in it that reaches MLX fails the link, so this tier can run beside a live server.
    const test_host_step = b.step("test-host", "Run the unit tests that link no MLX");
    const lint_step = addLint(b, target);
    test_step.dependOn(lint_step);
    test_host_step.dependOn(lint_step);
    const host_mod = b.createModule(.{
        .root_source_file = b.path("src/tests_host.zig"),
        .target = target,
        .optimize = optimize,
        .link_libcpp = true,
        .imports = &.{
            .{ .name = "build_options", .module = host_options.createModule() },
            .{ .name = "jinja_c", .module = addCHeaderModule(b, b.path("lib/jinja_cpp/jinja_wrapper.h"), b.path("lib/jinja_cpp"), target, optimize) },
        },
    });
    // bills.zig reaches the transformer's and the model's pure helpers, which import these three modules; no MLX symbol is linked.
    _ = addExl3Module(b, host_mod, target, optimize);
    host_mod.addImport("stb", addCHeaderModule(b, b.path("lib/stb_image.h"), b.path("lib"), target, optimize));
    host_mod.addCSourceFile(.{ .file = b.path("lib/stb_image_impl.c"), .flags = &.{"-O2"} });
    host_mod.addIncludePath(b.path("lib"));
    host_mod.addImport("webp", addCHeaderModule(b, .{ .cwd_relative = "/opt/homebrew/include/webp/decode.h" }, .{ .cwd_relative = "/opt/homebrew/include" }, target, optimize));
    linkApple(host_mod, macos_sdk_frameworks);
    host_mod.addObjectFile(b.path("lib/jinja_cpp/libjinja.a"));
    host_mod.addIncludePath(b.path("lib/jinja_cpp"));
    host_mod.linkSystemLibrary("c++", .{});
    // A user's filter narrows the host tier to its own tests that the filter selects: a bare substring would also select the
    // MLX-tier tests of the files the host files reach, and those do not link. A filter that selects none builds no binary.
    const host_filters = if (test_filter) |user| hostUserFilters(b, user) else hostFilters(b);
    const count_guard = addCountGuard(b, target);
    if (host_filters.len > 0) {
        const host_tests = b.addTest(.{ .name = "test-host", .root_module = host_mod, .filters = host_filters });
        test_build.dependOn(&b.addInstallArtifact(host_tests, .{ .dest_dir = .{ .override = .{ .custom = "tests" } } }).step);
        const run_host_tests = b.addRunArtifact(host_tests);
        test_host_step.dependOn(&run_host_tests.step);
        test_step.dependOn(&run_host_tests.step);
        if (test_filter == null) {
            const host_count = countGuard(b, count_guard, host_tests);
            test_host_step.dependOn(host_count);
            test_step.dependOn(host_count);
        }
    }
    const unit_tests: []const *std.Build.Step.Compile = if (test_model == .exl3 or test_model == .enc)
        &.{}
    else
        addTestBuckets(b, test_mod, test_model, test_filter);
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
        if (std.mem.eql(u8, unit_test.name, "test-" ++ tier_bucket[0])) {
            const on_files = b.addRunArtifact(unit_test);
            on_files.setName("run test-" ++ tier_bucket[0] ++ " (SUSHI_PREFIX_FS=real)");
            on_files.setEnvironmentVariable("SUSHI_PREFIX_FS", "real");
            test_step.dependOn(&on_files.step);
        }
        test_step.dependOn(countGuard(b, count_guard, unit_test));
    }

    // Zig collects tests from an artifact's root module only, so src/exl3 is its own.
    const exl3_tests: []const *std.Build.Step.Compile = if (test_model != null and test_model != .exl3)
        &.{}
    else
        addExl3TestBins(b, exl3_test_mod, test_filter);
    for (exl3_tests) |exl3_test| {
        test_build.dependOn(&b.addInstallArtifact(exl3_test, .{ .dest_dir = .{ .override = .{ .custom = "tests" } } }).step);
        test_step.dependOn(&b.addRunArtifact(exl3_test).step);
        test_step.dependOn(countGuard(b, count_guard, exl3_test));
    }

    const encoder_tests = b.addTest(.{
        .name = "encoder-test",
        .root_module = encoder_mod,
        .filters = test_filter orelse &.{},
    });
    if (test_model == null or test_model == .enc) {
        test_build.dependOn(&b.addInstallArtifact(encoder_tests, .{ .dest_dir = .{ .override = .{ .custom = "tests" } } }).step);
        test_step.dependOn(&b.addRunArtifact(encoder_tests).step);
    }

    guardBenchFiles(b, build_roots);
    const ubench_step = b.step("ubench", "Build zig-out/bin/sushi-ubench (always ReleaseFast): the microbenches, which no test step builds");
    ubench_step.dependOn(&b.addInstallArtifact(addUbench(b, build_options, target, macos_sdk_frameworks), .{}).step);

    const host_check_step = b.step("exl3-host-check", "Compile the sushi_exl3 module and its tests against the minimal host root src/exl3/host_min.zig (nothing links or runs)");
    for (addExl3HostCheck(b, target)) |check| host_check_step.dependOn(&check.step);

    addArchSlice(b, target, optimize, build_options);
}

/// `-Darchs=qwen4,glm5`: the architectures this build serves, named as `src/arch/registry.zig` names them; empty is
/// every registered one. An arch left out has no prong in the dispatch unions, so none of its code is analysed.
fn archList(b: *std.Build) []const []const u8 {
    const raw = b.option([]const u8, "archs", "Architectures to build, comma separated (default: every registered one)") orelse return &.{};
    var out: std.ArrayList([]const u8) = .empty;
    var it = std.mem.tokenizeScalar(u8, raw, ',');
    while (it.next()) |name| out.append(b.allocator, name) catch @panic("OOM");
    return out.items;
}

/// `sushibar`: the menu bar app. It links no MLX, jinja or server module, only the Objective-C runtime and AppKit.
fn addSushibar(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, macos_sdk_frameworks: ?[]const u8) *std.Build.Step.Compile {
    const mod = b.createModule(.{
        .root_source_file = b.path("src/sushibar_main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    if (macos_sdk_frameworks) |fw_path| mod.addFrameworkPath(.{ .cwd_relative = fw_path });
    mod.linkFramework("AppKit", .{});
    mod.linkFramework("Foundation", .{});
    return b.addExecutable(.{ .name = "sushibar", .root_module = mod });
}

/// `sushi-ubench`: the microbench harness and the bench bodies, a binary of its own. It stamps the commit it was
/// built from so a stale or dirty binary refuses to take a number.
fn addUbench(b: *std.Build, build_options: *std.Build.Step.Options, target: std.Build.ResolvedTarget, macos_sdk_frameworks: ?[]const u8) *std.Build.Step.Compile {
    // A run step that always runs: a configure-time read would stay cached across commits.
    const stamp_run = b.addSystemCommand(&.{ "sh", "-c", stamp_script });
    stamp_run.setCwd(b.path("."));
    stamp_run.has_side_effects = true;
    const stamp = b.createModule(.{ .root_source_file = stamp_run.captureStdOut(.{ .basename = "ubench_stamp.zig" }) });
    const mod = b.createModule(.{
        .root_source_file = b.path("src/ubench_main.zig"),
        .target = target,
        .optimize = .ReleaseFast,
        .link_libcpp = true,
        .imports = &.{
            .{ .name = "build_options", .module = build_options.createModule() },
            .{ .name = "ubench_stamp", .module = stamp },
        },
    });
    addMlxLib(b, mod);
    const exl3 = addExl3Module(b, mod, target, .ReleaseFast);
    mod.addImport("sushi_encoder", addEncoderModule(b, mod, exl3, target, .ReleaseFast));
    mod.addImport("sushi_exl3_ubench", b.createModule(.{
        .root_source_file = b.path("src/exl3_ubench/root.zig"),
        .target = target,
        .optimize = .ReleaseFast,
        .imports = &.{
            .{ .name = "mlx_host", .module = mod },
            .{ .name = "sushi_exl3", .module = exl3 },
        },
    }));
    linkApple(mod, macos_sdk_frameworks);
    return b.addExecutable(.{ .name = "sushi-ubench", .root_module = mod });
}

const stamp_script =
    \\printf 'pub const commit = "%s";\npub const dirty = %s;\npub const repo = "%s";\n' \
    \\  "$(git rev-parse HEAD)" "$(test -n "$(git status --porcelain --untracked-files=no)" && echo true || echo false)" "$(pwd -P)"
;

/// What an arch folder declares to the build in `src/<arch>/build_roots.zon`, so the shared build names no arch: an
/// offline tool built in place of the server by `-D<option>=true` (the tool itself is in the arch's `tools`), and the
/// arch's microbench files, by path from the repo root. A dropped folder takes its options with it.
const BuildRoots = struct {
    tools: []const struct { option: []const u8, help: []const u8 } = &.{},
    benches: []const []const u8 = &.{},
};

fn parseBuildRoots(arena: std.mem.Allocator, source: [:0]const u8, diag: *std.zon.parse.Diagnostics) !BuildRoots {
    return std.zon.parse.fromSlice(BuildRoots, .{ .gpa = arena, .arena = arena, .source = source, .diagnostics = diag });
}

/// The folders of `src/`, in name order.
fn srcFolders(b: *std.Build) []const []const u8 {
    const io = b.graph.io;
    var src = buildRootHandle(b).openDir(io, "src", .{ .iterate = true }) catch |e| std.debug.panic("src: {t}", .{e});
    defer src.close(io);
    b.dependOnDirectoryContents(b.path("src"));
    var folders: std.ArrayList([]const u8) = .empty;
    var it = src.iterate();
    while (it.next(io) catch |e| std.debug.panic("src: {t}", .{e})) |entry| {
        if (entry.kind == .directory or entry.kind == .sym_link) folders.append(b.allocator, b.dupe(entry.name)) catch @panic("OOM");
    }
    std.mem.sortUnstable([]const u8, folders.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
    return folders.items;
}

/// Every `src/<folder>/build_roots.zon`, in folder order.
fn archBuildRoots(b: *std.Build) []const BuildRoots {
    const io = b.graph.io;
    var out: std.ArrayList(BuildRoots) = .empty;
    for (srcFolders(b)) |folder| {
        b.dependOnDirectoryContents(b.path(b.fmt("src/{s}", .{folder})));
        const path = b.fmt("src/{s}/build_roots.zon", .{folder});
        const source = buildRootHandle(b).readFileAllocOptions(io, path, b.allocator, .limited(1 << 16), .of(u8), 0) catch |e| switch (e) {
            error.FileNotFound, error.NotDir => continue,
            else => std.debug.panic("{s}: {t}", .{ path, e }),
        };
        b.dependOnFileContents(b.path(path));
        var diag: std.zon.parse.Diagnostics = undefined;
        const roots = parseBuildRoots(b.allocator, source, &diag) catch |e| {
            if (e == error.ParseZon) diag.log(path);
            std.debug.panic("{s}: {t}", .{ path, e });
        };
        out.append(b.allocator, roots) catch @panic("OOM");
    }
    return out.items;
}

/// The first declared offline tool whose `-D` option is on; null builds the server. A tool is run through
/// `src/tool_root.zig`, which finds it by this name in the declaring arch's `tools`.
fn toolRoot(b: *std.Build, roots: []const BuildRoots) ?[]const u8 {
    var picked: ?[]const u8 = null;
    for (roots) |r| for (r.tools) |tool| {
        if (b.option(bool, tool.option, tool.help) orelse false) picked = picked orelse tool.option;
    };
    return picked;
}

/// A microbench file holds no `test` block: a bench is timed by `sushi-ubench`, never run by the suite.
/// `ubench_report.zig` is the harness's host tier and keeps its tests.
fn guardBenchFiles(b: *std.Build, roots: []const BuildRoots) void {
    const io = b.graph.io;
    const shared = [_][]const u8{ "src/ubench_shared.zig", "src/exl3_ubench/ubench.zig", "src/encoder/ubench.zig" };
    var files: std.ArrayList([]const u8) = .empty;
    files.appendSlice(b.allocator, &shared) catch @panic("OOM");
    for (roots) |r| files.appendSlice(b.allocator, r.benches) catch @panic("OOM");
    for (files.items) |path| {
        const source = buildRootHandle(b).readFileAlloc(io, path, b.allocator, .limited(8 << 20)) catch |e| std.debug.panic("{s}: {t}", .{ path, e });
        b.dependOnFileContents(b.path(path));
        var lines = std.mem.splitScalar(u8, source, '\n');
        while (lines.next()) |line| {
            const at = std.mem.indexOf(u8, line, "test \"") orelse continue;
            if (std.mem.trim(u8, line[0..at], " ").len == 0)
                std.debug.panic("{s} holds a test block: a microbench is never a test ({s})", .{ path, line });
        }
    }
}

fn linkApple(module: *std.Build.Module, macos_sdk_frameworks: ?[]const u8) void {
    if (macos_sdk_frameworks) |fw_path| module.addFrameworkPath(.{ .cwd_relative = fw_path });
    module.linkFramework("IOKit", .{});
    module.linkFramework("CoreFoundation", .{});
    module.linkFramework("Foundation", .{});
    module.linkFramework("Metal", .{});
    module.linkFramework("IOSurface", .{});
}

/// The unit tests compile as several binaries in parallel: one LLVM module is single-threaded.
/// A test's name is `<file stem>.test.<name>`, or `<file stem>.<Struct>.test.<name>` inside a struct, so a bucket is
/// a list of `<stem>.test.` filters plus one filter per nested test; the last bucket takes every MLX stem the others
/// do not name.
const test_buckets = [_][]const []const u8{
    &.{ "trunk", "gdn_decode", "cli", "launch" },
    &.{ "generate", "mtp_policy", "mtp_round", "scheduler", "restore_dump", "mtp", "dflash" },
    &.{ "server", "responses" },
    tier_bucket,
};

/// The SSD tier's tests, which run once on each file backend (`SUSHI_PREFIX_FS`).
const tier_bucket: []const []const u8 = &.{ "kv_disk_cache", "kv_disk_fs", "prefix_cache", "disk_placement", "prefix_cache_admin", "prefix_vfs_tensors_test", "qwen4_disk_qsa", "qwen4_prefix_cache_test", "glm5_prefix_cache_test", "dsv41_prefix_cache_test" };

/// The model folders hold their own test files beside the root's; src/exl3 and src/encoder are modules of their own.
const test_dirs = [_][]const u8{ "src", "src/arch", "src/tensor", "src/qwen4", "src/mimo2", "src/glm5", "src/dsv41" };

/// A test's model is its file's folder; the src root is shared.
const TestModel = enum { qwen4, mimo2, glm5, dsv41, exl3, enc };

/// Zig keeps a test when its fully qualified name holds any filter, and filters OR, so a user's filter cannot narrow
/// a bucket's: the scan names each test in full and keeps those the user's filters select.
fn selectedByUser(user: ?[]const []const u8, full: []const u8) bool {
    const filters = user orelse return true;
    for (filters) |f| if (std.mem.indexOf(u8, full, f) != null) return true;
    return false;
}

/// The filter that selects a user-matched test. Filters reach `zig test` in a response file whose parser reads a quote
/// or backslash as syntax, so a name holding one is cut there and its longest piece that no skipped test also holds
/// stands for it; null when every piece would select a test the user's filters skip.
fn quoteFreeFilter(full: []const u8, all: []const []const u8, user: []const []const u8) ?[]const u8 {
    if (std.mem.indexOfAny(u8, full, "'\"\\") == null) return full;
    var best: ?[]const u8 = null;
    var pieces = std.mem.tokenizeAny(u8, full, "'\"\\");
    while (pieces.next()) |piece| {
        if (best) |b| if (b.len >= piece.len) continue;
        for (all) |other| {
            if (std.mem.indexOf(u8, other, piece) != null and !selectedByUser(user, other)) break;
        } else best = piece;
    }
    return best;
}

fn nameFilter(full: []const u8, all: []const []const u8, user: []const []const u8) []const u8 {
    return quoteFreeFilter(full, all, user) orelse
        std.debug.panic("rename the test \"{s}\": no quote-free part of it avoids the tests your filter skips", .{full});
}

fn addTestBuckets(b: *std.Build, root: *std.Build.Module, model: ?TestModel, user: ?[]const []const u8) []const *std.Build.Step.Compile {
    const io = b.graph.io;
    const host_stems = hostStems(b);
    const host_paths = hostPaths(b);
    var stems: std.ArrayList([]const u8) = .empty;
    var nested: std.ArrayList(struct { stem: []const u8, filter: []const u8, full: []const u8 }) = .empty;
    // A scoped run names each test of a root file that another model's file name ends with ("model" in
    // "dsv41_model"), since a filter matches anywhere in a test name.
    var skipped: std.ArrayList([]const u8) = .empty;
    var all_names: std.ArrayList([]const u8) = .empty;
    var top: std.ArrayList(struct { stem: []const u8, filter: []const u8 }) = .empty;
    for (test_dirs) |dir_path| {
        const out_of_scope = if (model) |m| !std.mem.eql(u8, dir_path, "src") and !std.mem.eql(u8, dir_path, "src/arch") and !std.mem.eql(u8, dir_path, "src/tensor") and !std.mem.eql(u8, dir_path["src/".len..], @tagName(m)) else false;
        // A dropped architecture has no folder.
        var src = buildRootHandle(b).openDir(io, dir_path, .{ .iterate = true }) catch |e| switch (e) {
            error.FileNotFound => continue,
            else => std.debug.panic("{s}: {t}", .{ dir_path, e }),
        };
        // The scan runs at configure time, whose cache would otherwise miss a new or edited test file.
        b.dependOnDirectoryContents(b.path(dir_path));
        defer src.close(io);
        var it = src.iterate();
        while (it.next(io) catch |e| std.debug.panic("{s}: {t}", .{ dir_path, e })) |entry| {
            if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".zig")) continue;
            const stem = entry.name[0 .. entry.name.len - ".zig".len];
            if (isHostFile(host_paths, dir_path, stem)) continue;
            if (std.mem.startsWith(u8, stem, "ubench_")) continue;
            if (out_of_scope) {
                skipped.append(b.allocator, b.allocator.dupe(u8, stem) catch @panic("OOM")) catch @panic("OOM");
                continue;
            }
            stems.append(b.allocator, b.fmt("{s}.test.", .{stem})) catch @panic("OOM");
            b.dependOnFileContents(b.path(b.fmt("{s}/{s}", .{ dir_path, entry.name })));
            const source = src.readFileAlloc(io, entry.name, b.allocator, .limited(16 << 20)) catch |e| std.debug.panic("{s}: {t}", .{ entry.name, e });
            var lines = std.mem.splitScalar(u8, source, '\n');
            var owner: []const u8 = "";
            while (lines.next()) |line| {
                if ((std.mem.startsWith(u8, line, "pub const ") or std.mem.startsWith(u8, line, "const ")) and std.mem.endsWith(u8, line, " = struct {"))
                    owner = line[std.mem.indexOf(u8, line, "const ").? + "const ".len .. line.len - " = struct {".len];
                const indent = std.mem.indexOf(u8, line, "test \"") orelse continue;
                if (indent == 0 and (model != null or user != null) and std.mem.endsWith(u8, line, "\" {")) {
                    const full = b.fmt("{s}.test.{s}", .{ stem, line["test \"".len .. line.len - "\" {".len] });
                    all_names.append(b.allocator, full) catch @panic("OOM");
                    top.append(b.allocator, .{ .stem = b.fmt("{s}.test.", .{stem}), .filter = full }) catch @panic("OOM");
                }
                if (indent == 0 or std.mem.trim(u8, line[0..indent], " ").len != 0) continue;
                if (!std.mem.endsWith(u8, line, "\" {") or std.mem.indexOfScalar(u8, line, '\\') != null)
                    std.debug.panic("{s}: write `test \"name\" {{` on one line without escapes: {s}", .{ entry.name, line });
                const name = line[indent + "test \"".len .. line.len - "\" {".len];
                // A nested test is `<stem>.<Struct>.test.<name>`; only a struct opened at column 0 is understood.
                if (user != null and (indent != 4 or owner.len == 0)) std.debug.panic("{s}: cannot name the nested test: {s}", .{ entry.name, line });
                const full = b.fmt("{s}.{s}.test.{s}", .{ stem, owner, name });
                nested.append(b.allocator, .{ .stem = b.fmt("{s}.test.", .{stem}), .filter = b.fmt(".test.{s}", .{name}), .full = full }) catch @panic("OOM");
                all_names.append(b.allocator, full) catch @panic("OOM");
            }
        }
    }
    // Directory order varies; the filters key the compile step's cache.
    std.mem.sortUnstable([]const u8, stems.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
    const bucket = b.allocator.alloc(usize, stems.items.len) catch @panic("OOM");
    for (stems.items, bucket) |stem, *k| k.* = namedBucket(stem[0 .. stem.len - ".test.".len]);
    // A filter matches anywhere in a name, so "ws.test." also selects dspark_rows's tests: stems
    // that end with one another must share a bucket, or their tests would run twice.
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
    // A host file's tests run only in test-host, so no MLX filter may select them. A host file is found by its path, so
    // an MLX file that shares its stem (a later src/render.zig, an arch's value.zig) lands here, loudly.
    for (host_stems) |h| for (stems.items) |a| {
        if (std.mem.endsWith(u8, h, a[0 .. a.len - ".test.".len])) std.debug.panic("host test file {s} would also run in the MLX bucket of {s}", .{ h, a });
    };
    var out: std.ArrayList(*std.Build.Step.Compile) = .empty;
    for (0..test_buckets.len + 1) |i| {
        var filters: std.ArrayList([]const u8) = .empty;
        for (stems.items, bucket) |stem, k| if (k == i) {
            if (user) |u| {
                for (top.items) |t| if (std.mem.eql(u8, t.stem, stem) and selectedByUser(u, t.filter)) filters.append(b.allocator, nameFilter(t.filter, all_names.items, u)) catch @panic("OOM");
                continue;
            }
            const bare = stem[0 .. stem.len - ".test.".len];
            const shadowed = for (skipped.items) |sk| {
                if (std.mem.endsWith(u8, sk, bare)) break true;
            } else false;
            if (!shadowed) {
                filters.append(b.allocator, stem) catch @panic("OOM");
                continue;
            }
            for (top.items) |t| if (std.mem.eql(u8, t.stem, stem)) filters.append(b.allocator, t.filter) catch @panic("OOM");
        };
        for (nested.items) |n| {
            const owner = for (stems.items, bucket) |stem, k| {
                if (std.mem.eql(u8, stem, n.stem)) break k;
            } else unreachable;
            if (owner != i) continue;
            if (user) |u| {
                if (selectedByUser(u, n.full)) filters.append(b.allocator, nameFilter(n.full, all_names.items, u)) catch @panic("OOM");
            } else filters.append(b.allocator, n.filter) catch @panic("OOM");
        }
        if (filters.items.len == 0) continue;
        const name = if (i < test_buckets.len) b.fmt("test-{s}", .{test_buckets[i][0]}) else "test-rest";
        out.append(b.allocator, b.addTest(.{ .name = name, .root_module = root, .filters = filters.items })) catch @panic("OOM");
    }
    return out.items;
}

/// The host tier's files, as `src/`-relative paths without `.zig`: the imports of src/tests_host.zig, and each arch
/// folder's `host_tests` in its `arch_tests.zig`.
fn hostPaths(b: *std.Build) []const []const u8 {
    b.dependOnFileContents(b.path("src/tests_host.zig"));
    const source = buildRootHandle(b).readFileAlloc(b.graph.io, "src/tests_host.zig", b.allocator, .limited(1 << 20)) catch |e| std.debug.panic("src/tests_host.zig: {t}", .{e});
    var paths: std.ArrayList([]const u8) = .empty;
    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |line| {
        const head = "    _ = @import(\"";
        const tail = ".zig\");";
        if (!std.mem.startsWith(u8, line, head) or !std.mem.endsWith(u8, line, tail)) continue;
        paths.append(b.allocator, line[head.len .. line.len - tail.len]) catch @panic("OOM");
    }
    // A folder left out of -Darchs is listed too: its host files stay out of the MLX buckets whichever arch is built.
    for (srcFolders(b)) |folder| {
        const arch_zig = b.fmt("src/{s}/arch_tests.zig", .{folder});
        const arch_source = buildRootHandle(b).readFileAlloc(b.graph.io, arch_zig, b.allocator, .limited(1 << 20)) catch |e| switch (e) {
            error.FileNotFound, error.NotDir => continue,
            else => std.debug.panic("{s}: {t}", .{ arch_zig, e }),
        };
        b.dependOnFileContents(b.path(arch_zig));
        appendArchList(b.allocator, &paths, folder, arch_source, "host_tests") catch |e|
            std.debug.panic("{s}: `pub const host_tests = .{{` takes one `@import(\"<file>.zig\"),` a line: {t}", .{ arch_zig, e });
    }
    return paths.items;
}

/// Appends `<folder>/<file>` for each `@import("<file>.zig"),` line of `pub const <decl> = .{ ... };` in an arch_tests.zig. The
/// files sit in the arch's folder; a comment or blank line is skipped, any other line is an error.
fn appendArchList(gpa: std.mem.Allocator, out: *std.ArrayList([]const u8), folder: []const u8, source: []const u8, decl: []const u8) !void {
    const open = try std.fmt.allocPrint(gpa, "pub const {s} = .{{", .{decl});
    var inside = false;
    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (!inside) {
            inside = std.mem.eql(u8, line, open);
            continue;
        }
        if (std.mem.eql(u8, line, "};")) return;
        if (line.len == 0 or std.mem.startsWith(u8, line, "//")) continue;
        const head = "@import(\"";
        const tail = ".zig\"),";
        if (!std.mem.startsWith(u8, line, head) or !std.mem.endsWith(u8, line, tail)) return error.UnreadableTestList;
        const file = line[head.len .. line.len - tail.len];
        if (std.mem.indexOfScalar(u8, file, '/') != null) return error.UnreadableTestList;
        try out.append(gpa, try std.fmt.allocPrint(gpa, "{s}/{s}", .{ folder, file }));
    }
    if (inside) return error.UnreadableTestList;
}

fn hostStems(b: *std.Build) []const []const u8 {
    var stems: std.ArrayList([]const u8) = .empty;
    for (hostPaths(b)) |path| stems.append(b.allocator, path[if (std.mem.lastIndexOfScalar(u8, path, '/')) |i| i + 1 else 0..]) catch @panic("OOM");
    return stems.items;
}

/// An arch's `<arch>_bill.zig` is MLX-free and runs in the host tier; the bills test reaches each enabled arch's through the registry.
const arch_bill_suffix = "_bill";

/// The host tier's tests by name. Zig keeps a test by its name and collects the tests of every file the root reaches:
/// bills.zig reaches the transformer and the model, whose own tests need MLX and must not join this binary.
fn hostFilters(b: *std.Build) []const []const u8 {
    const io = b.graph.io;
    var out: std.ArrayList([]const u8) = .empty;
    for (hostPaths(b)) |path| {
        const stem = path[if (std.mem.lastIndexOfScalar(u8, path, '/')) |i| i + 1 else 0..];
        // A dropped arch's file is missing: the import then fails the compile, naming it, instead of the configure.
        const file = buildRootHandle(b).readFileAlloc(io, b.fmt("src/{s}.zig", .{path}), b.allocator, .limited(16 << 20)) catch |e| switch (e) {
            error.FileNotFound => continue,
            else => std.debug.panic("src/{s}.zig: {t}", .{ path, e }),
        };
        out.append(b.allocator, b.fmt("{s}.test.", .{stem})) catch @panic("OOM");
        b.dependOnFileContents(b.path(b.fmt("src/{s}.zig", .{path})));
        var owner: []const u8 = "";
        var rows = std.mem.splitScalar(u8, file, '\n');
        while (rows.next()) |row| {
            if ((std.mem.startsWith(u8, row, "pub const ") or std.mem.startsWith(u8, row, "const ")) and std.mem.endsWith(u8, row, " = struct {"))
                owner = row[std.mem.indexOf(u8, row, "const ").? + "const ".len .. row.len - " = struct {".len];
            if (std.mem.startsWith(u8, row, "    test \"") and owner.len != 0)
                out.append(b.allocator, b.fmt("{s}.{s}.test.", .{ stem, owner })) catch @panic("OOM");
        }
    }
    out.append(b.allocator, arch_bill_suffix ++ ".test.") catch @panic("OOM");
    return out.items;
}

/// Appends the full name of each test in `source`, a file whose stem is `stem`: `<stem>.test.<name>`, or
/// `<stem>.<Struct>.test.<name>` inside a struct opened at column 0 (the same reading the buckets' scan makes).
fn appendTestNames(b: *std.Build, out: *std.ArrayList([]const u8), stem: []const u8, source: []const u8) void {
    var owner: []const u8 = "";
    var rows = std.mem.splitScalar(u8, source, '\n');
    while (rows.next()) |row| {
        if ((std.mem.startsWith(u8, row, "pub const ") or std.mem.startsWith(u8, row, "const ")) and std.mem.endsWith(u8, row, " = struct {"))
            owner = row[std.mem.indexOf(u8, row, "const ").? + "const ".len .. row.len - " = struct {".len];
        if (!std.mem.endsWith(u8, row, "\" {")) continue;
        if (std.mem.startsWith(u8, row, "test \""))
            out.append(b.allocator, b.fmt("{s}.test.{s}", .{ stem, row["test \"".len .. row.len - "\" {".len] })) catch @panic("OOM")
        else if (std.mem.startsWith(u8, row, "    test \"") and owner.len != 0)
            out.append(b.allocator, b.fmt("{s}.{s}.test.{s}", .{ stem, owner, row["    test \"".len .. row.len - "\" {".len] })) catch @panic("OOM");
    }
}

/// The host tier's tests that a user's `-Dtest-filter` selects, each by its own name (`selectedByUser`, as the buckets do).
fn hostUserFilters(b: *std.Build, user: []const []const u8) []const []const u8 {
    const io = b.graph.io;
    var names: std.ArrayList([]const u8) = .empty;
    for (hostPaths(b)) |path| {
        const stem = path[if (std.mem.lastIndexOfScalar(u8, path, '/')) |i| i + 1 else 0..];
        const file = buildRootHandle(b).readFileAlloc(io, b.fmt("src/{s}.zig", .{path}), b.allocator, .limited(16 << 20)) catch |e| switch (e) {
            error.FileNotFound => continue,
            else => std.debug.panic("src/{s}.zig: {t}", .{ path, e }),
        };
        b.dependOnFileContents(b.path(b.fmt("src/{s}.zig", .{path})));
        appendTestNames(b, &names, stem, file);
    }
    for (test_dirs) |dir_path| {
        var dir = buildRootHandle(b).openDir(io, dir_path, .{ .iterate = true }) catch |e| switch (e) {
            error.FileNotFound => continue,
            else => std.debug.panic("{s}: {t}", .{ dir_path, e }),
        };
        defer dir.close(io);
        var it = dir.iterate();
        while (it.next(io) catch |e| std.debug.panic("{s}: {t}", .{ dir_path, e })) |entry| {
            if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, arch_bill_suffix ++ ".zig")) continue;
            const file = dir.readFileAlloc(io, entry.name, b.allocator, .limited(16 << 20)) catch |e| std.debug.panic("{s}: {t}", .{ entry.name, e });
            b.dependOnFileContents(b.path(b.fmt("{s}/{s}", .{ dir_path, entry.name })));
            appendTestNames(b, &names, entry.name[0 .. entry.name.len - ".zig".len], file);
        }
    }
    var out: std.ArrayList([]const u8) = .empty;
    for (names.items) |full| if (selectedByUser(user, full)) out.append(b.allocator, nameFilter(full, names.items, user)) catch @panic("OOM");
    return out.items;
}

/// Whether `<dir_path>/<stem>.zig` is a host-tier file, matched by its path under src/, never by its stem alone.
fn isHostFile(host_paths: []const []const u8, dir_path: []const u8, stem: []const u8) bool {
    if (std.mem.endsWith(u8, stem, arch_bill_suffix)) return true;
    const folder = if (std.mem.eql(u8, dir_path, "src")) "" else dir_path["src/".len..];
    for (host_paths) |h| {
        const sep = std.mem.lastIndexOfScalar(u8, h, '/');
        const h_folder = if (sep) |i| h[0..i] else "";
        const h_stem = if (sep) |i| h[i + 1 ..] else h;
        if (std.mem.eql(u8, h_folder, folder) and std.mem.eql(u8, h_stem, stem)) return true;
    }
    return false;
}

fn namedBucket(stem: []const u8) usize {
    for (test_buckets, 0..) |names, i| {
        for (names) |name| if (std.mem.eql(u8, stem, name)) return i;
    }
    return test_buckets.len;
}

/// Seconds of the slow src/exl3 tests with the binaries running side by side; any other test weighs 1. Only the
/// balance of the exl3 binaries reads this: a stale entry costs speed, never coverage.
const exl3_test_weights = [_]struct { []const u8, u16 }{
    .{ "group2 half4 composition preserves all-rate", 13 },
    .{ "every admitted rate takes the fast arms", 23 },
    .{ "every K1.5 to K2 rate takes the fast arms", 30 },
    .{ "group2 cooperative projections preserve F16 bits", 17 },
    .{ "prefill grid serves every admitted rate, mixed projections", 4 },
    .{ "prefill rows hold the f32-truth bar for every codebook, past 24 groups", 39 },
    .{ "the grouped chain at 5120x2304", 108 },
    .{ "SPEC1 per-rate table: every reader decodes", 13 },
    .{ "SPEC1 below K1.5 and above K4: the prefill GEMM", 40 },
    .{ "SPEC1 below K1.5 and above K4: every reader decodes", 2 },
    .{ "serves spec1 and std1 on the prefill grid", 36 },
    .{ "one-bank and grouped trellis linears", 35 },
    .{ "SIMD multi-row: the split row loops", 15 },
    .{ "every slot writes the served kernels' bytes", 35 },
    .{ "spec1's 2^12 folded into the stores", 28 },
    .{ "PAIR1 decode and prefill rows hold the f32-truth bar", 27 },
    .{ "PAIR1's 2^12 folded into the stores", 26 },
    .{ "SPEC1 per-rate table: the 2^12 folded into the stores", 23 },
    .{ "BF16 prefill on the simdgroup-matrix body", 20 },
    .{ "Sushi GPU decode and prefill match scalar truth", 19 },
    .{ "the shared expert as a one-expert fused chain", 19 },
    .{ "GLM serves spec1 and std1 on group2's routed chain", 18 },
    .{ "GLM serves spec1 and std1 on its lane chain", 17 },
    .{ "GLM GPU windows preserve 288 expert bytes at every admitted rate", 15 },
    .{ "multi-row projection holds the bar at every admitted rate", 14 },
    .{ "PAIR1 weight readers decode every K1.5 to K4 rate", 14 },
    .{ "SPEC1 per-rate table: decode and prefill rows hold", 12 },
    .{ "GLM down lane exact staged and fused bytes", 11 },
    .{ "the simdgroup-matrix body and the scalar body both match", 10 },
    .{ "multi-row projection matches projectIndexed per row", 10 },
    .{ "MCG n48 BF16 decode and NAX preserve f32 truth", 9 },
};
const exl3_test_bins = 6;

const Exl3Test = struct { filter: []const u8, weight: u16, bin: usize = 0 };

/// The exl3 tests compile and run as several binaries. A filter matches `<file stem>.test.<name>`,
/// so each test is a filter of its own, found by scanning src/exl3, and the tests are balanced by weight.
fn addExl3TestBins(b: *std.Build, root: *std.Build.Module, user: ?[]const []const u8) []const *std.Build.Step.Compile {
    const io = b.graph.io;
    var tests: std.ArrayList(Exl3Test) = .empty;
    var all_names: std.ArrayList([]const u8) = .empty;
    var dir = buildRootHandle(b).openDir(io, "src/exl3", .{ .iterate = true }) catch |e| std.debug.panic("src/exl3/: {t}", .{e});
    defer dir.close(io);
    b.dependOnDirectoryContents(b.path("src/exl3"));
    var files = dir.iterate();
    while (files.next(io) catch |e| std.debug.panic("src/exl3/: {t}", .{e})) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".zig")) continue;
        const stem = entry.name[0 .. entry.name.len - ".zig".len];
        const source = dir.readFileAlloc(io, entry.name, b.allocator, .limited(8 << 20)) catch |e| std.debug.panic("{s}: {t}", .{ entry.name, e });
        b.dependOnFileContents(b.path(b.fmt("src/exl3/{s}", .{entry.name})));
        var lines = std.mem.splitScalar(u8, source, '\n');
        while (lines.next()) |line| {
            if (!std.mem.startsWith(u8, line, "test \"")) continue;
            // A name this scan cannot cut out would drop its test from every binary.
            if (!std.mem.endsWith(u8, line, "\" {") or std.mem.indexOfScalar(u8, line, '\\') != null)
                std.debug.panic("{s}: write `test \"name\" {{` on one line without escapes: {s}", .{ entry.name, line });
            const name = line["test \"".len .. line.len - "\" {".len];
            var weight: u16 = 1;
            for (exl3_test_weights) |w| if (std.mem.indexOf(u8, name, w[0]) != null) {
                weight = w[1];
            };
            const full = b.fmt("{s}.test.{s}", .{ stem, name });
            all_names.append(b.allocator, full) catch @panic("OOM");
            if (selectedByUser(user, full)) tests.append(b.allocator, .{ .filter = full, .weight = weight }) catch @panic("OOM");
        }
    }
    if (user) |u| for (tests.items) |*t| {
        t.filter = nameFilter(t.filter, all_names.items, u);
    };
    std.mem.sortUnstable(Exl3Test, tests.items, {}, struct {
        fn lt(_: void, x: Exl3Test, y: Exl3Test) bool {
            return if (x.weight != y.weight) x.weight > y.weight else std.mem.lessThan(u8, x.filter, y.filter);
        }
    }.lt);
    var load: [exl3_test_bins]u32 = @splat(0);
    for (tests.items) |*t| {
        t.bin = std.mem.indexOfMin(u32, &load);
        load[t.bin] += t.weight;
    }
    // A filter matches anywhere in a name, so a test whose name holds another's runs in that one's binary.
    var moved = true;
    while (moved) {
        moved = false;
        for (tests.items, 0..) |a, i| {
            for (tests.items, 0..) |*c, j| {
                if (i == j or a.bin == c.bin or std.mem.indexOf(u8, c.filter, a.filter) == null) continue;
                c.bin = a.bin;
                moved = true;
            }
        }
    }
    var out: std.ArrayList(*std.Build.Step.Compile) = .empty;
    for (0..exl3_test_bins) |i| {
        var filters: std.ArrayList([]const u8) = .empty;
        for (tests.items) |t| if (t.bin == i) filters.append(b.allocator, t.filter) catch @panic("OOM");
        if (filters.items.len == 0) continue;
        out.append(b.allocator, b.addTest(.{ .name = b.fmt("exl3-test-{d}", .{i + 1}), .root_module = root, .filters = filters.items })) catch @panic("OOM");
    }
    return out.items;
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
        .imports = &.{
            .{ .name = "mlx_host", .module = host },
        },
    });
    host.addImport("sushi_exl3", exl3);
    return exl3;
}

/// `sushi_exl3` compiled against `src/exl3/host_min.zig`, whose `mlx`, `log` and `io_util` are sushi's own files. A
/// module that reaches any other host name fails here instead of in a host that lacks it. Two objects: the module's
/// own tests, and host_min's walk over the module's public declarations.
fn addExl3HostCheck(b: *std.Build, target: std.Build.ResolvedTarget) [2]*std.Build.Step.Compile {
    const files = b.createModule(.{ .root_source_file = b.path("src/arch_host_stub.zig"), .target = target, .optimize = .Debug });
    const host = b.createModule(.{
        .root_source_file = b.path("src/exl3/host_min.zig"),
        .target = target,
        .optimize = .Debug,
        .imports = &.{.{ .name = "host_files", .module = files }},
    });
    const exl3 = b.createModule(.{
        .root_source_file = b.path("src/exl3/root.zig"),
        .target = target,
        .optimize = .Debug,
        .link_libc = true,
        .imports = &.{
            .{ .name = "mlx_host", .module = host },
        },
    });
    host.addImport("sushi_exl3", exl3);
    return .{
        b.addTest(.{ .name = "exl3-host-check-tests", .root_module = exl3, .emit_object = true }),
        b.addTest(.{ .name = "exl3-host-check-decls", .root_module = host, .emit_object = true }),
    };
}

/// `zig build lint`: scripts/lint/ over src/, every rule on every file. It always runs, so a new or edited file is read.
/// ReleaseFast: the Debug walk of src/ alone costs over half a second.
fn addLint(b: *std.Build, target: std.Build.ResolvedTarget) *std.Build.Step {
    const exe = b.addExecutable(.{ .name = "lint", .root_module = b.createModule(.{
        .root_source_file = b.path("scripts/lint/main.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    }) });
    const run = b.addRunArtifact(exe);
    run.setCwd(b.path("."));
    run.addArgs(&.{ "src", "tests/guards/lint.allow" });
    run.has_side_effects = true;
    const step = b.step("lint", "Run the source lint rules (scripts/lint/) over src/");
    step.dependOn(&run.step);
    return step;
}

/// Zig reports a test binary whose filters select nothing as a pass. This helper fails such a binary, naming it.
fn addCountGuard(b: *std.Build, target: std.Build.ResolvedTarget) *std.Build.Step.Compile {
    return b.addExecutable(.{ .name = "test-count-guard", .root_module = b.createModule(.{
        .root_source_file = b.path("scripts/test_count_guard.zig"),
        .target = target,
        .optimize = .Debug,
    }) });
}

fn countGuard(b: *std.Build, guard: *std.Build.Step.Compile, unit_test: *std.Build.Step.Compile) *std.Build.Step {
    const run = b.addRunArtifact(guard);
    run.addArg(unit_test.name);
    run.addArtifactArg(unit_test);
    return &run.step;
}

/// `zig build arch-slice -Dslice-arch=<name>`: the arch's folder as a module of its own, rooted at its `arch.zig`, against
/// `arch_host_stub.zig`. Zig refuses an import that leaves a module's folder, so every engine file the folder reaches
/// other than through `arch_host` is a compile error; nothing links or runs.
fn addArchSlice(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, build_options: *std.Build.Step.Options) void {
    const step = b.step("arch-slice", "Compile one arch's folder alone against the host stub (-Dslice-arch=<name>); nothing links or runs");
    const arch = b.option([]const u8, "slice-arch", "The architecture folder src/<name> that arch-slice builds") orelse {
        step.dependOn(&b.addFail("arch-slice needs -Dslice-arch=<name>, the folder under src/ that holds an arch.zig").step);
        return;
    };
    const root = b.fmt("src/{s}/arch.zig", .{arch});
    const named = arch.len > 0 and for (arch) |c| {
        if (!std.ascii.isLower(c) and !std.ascii.isDigit(c) and c != '_') break false;
    } else true;
    if (!named or std.meta.isError(buildRootHandle(b).access(b.graph.io, root, .{}))) {
        step.dependOn(&b.addFail(b.fmt("-Dslice-arch={s}: {s} does not exist", .{ arch, root })).step);
        return;
    }
    const host = b.createModule(.{
        .root_source_file = b.path("src/arch_host_stub.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "build_options", .module = build_options.createModule() }},
    });
    addMlxLib(b, host);
    const exl3 = addExl3Module(b, host, target, optimize);
    const folder = b.createModule(.{
        .root_source_file = b.path(root),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "build_options", .module = build_options.createModule() },
            .{ .name = "arch_host", .module = host },
            .{ .name = "sushi_exl3", .module = exl3 },
        },
    });
    const check = b.createModule(.{
        .root_source_file = b.path("src/arch/slice_check.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "arch_slice", .module = folder }},
    });
    step.dependOn(&b.addTest(.{ .name = b.fmt("arch-slice-{s}", .{arch}), .root_module = check, .emit_object = true }).step);
}

/// The encoder behind `sushi convert exl3`, also tested alone as `encoder-test`.
fn addEncoderModule(b: *std.Build, host: *std.Build.Module, exl3: *std.Build.Module, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const encoder = b.createModule(.{
        .root_source_file = b.path("src/encoder/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "mlx_host", .module = host },
            .{ .name = "sushi_exl3", .module = exl3 },
        },
    });
    return encoder;
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

test "a test name with an apostrophe keeps a quote-free filter that skips no other test" {
    const all = [_][]const u8{
        "server.test.Flash-Next's request bill brackets the peaks",
        "server.test.Flash-Next's streamed request bill brackets the streamed peaks",
        "server.test.plain name",
    };
    const user = [_][]const u8{"streamed request bill brackets"};
    const f = quoteFreeFilter(all[1], &all, &user).?;
    try std.testing.expect(std.mem.indexOfAny(u8, f, "'\"\\") == null);
    for (all, 0..) |name, i| try std.testing.expectEqual(i == 1, std.mem.indexOf(u8, name, f) != null);
}

test "a name without a quote is its own filter, and an unavoidable quote yields none" {
    const all = [_][]const u8{ "a.test.x's y", "a.test.x and s y" };
    const user = [_][]const u8{"x's y"};
    try std.testing.expectEqualStrings("a.test.plain", quoteFreeFilter("a.test.plain", &all, &user).?);
    try std.testing.expect(quoteFreeFilter(all[0], &all, &user) == null);
}

test "a build_roots.zon declares tools and benches, and a misspelt key is refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var diag: std.zon.parse.Diagnostics = undefined;
    const roots = try parseBuildRoots(arena.allocator(),
        \\.{
        \\    .tools = .{ .{ .option = "a-tool", .help = "Build a" } },
        \\    .benches = .{ "src/a/ubench.zig", "src/b.zig" },
        \\}
    , &diag);
    try std.testing.expectEqual(@as(usize, 1), roots.tools.len);
    try std.testing.expectEqualStrings("a-tool", roots.tools[0].option);
    try std.testing.expectEqualStrings("Build a", roots.tools[0].help);
    try std.testing.expectEqual(@as(usize, 2), roots.benches.len);
    try std.testing.expectEqual(@as(usize, 0), (try parseBuildRoots(arena.allocator(), ".{}", &diag)).benches.len);
    try std.testing.expectError(error.ParseZon, parseBuildRoots(arena.allocator(), ".{ .bench = .{} }", &diag));
}

test "an arch_tests.zig lists its host tests one import a line, and a stray line is refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var out: std.ArrayList([]const u8) = .empty;
    try appendArchList(arena.allocator(), &out, "zz",
        \\pub const tests = .{
        \\    @import("not_host.zig"),
        \\};
        \\pub const host_tests = .{
        \\    @import("a_rows.zig"),
        \\    // a note
        \\
        \\    @import("b_tokens.zig"),
        \\};
        \\pub const after = 1;
    , "host_tests");
    try std.testing.expectEqual(@as(usize, 2), out.items.len);
    try std.testing.expectEqualStrings("zz/a_rows", out.items[0]);
    try std.testing.expectEqualStrings("zz/b_tokens", out.items[1]);
    try appendArchList(arena.allocator(), &out, "zz", "pub const tests = .{};\n", "host_tests");
    try std.testing.expectEqual(@as(usize, 2), out.items.len);
    try std.testing.expectError(error.UnreadableTestList, appendArchList(arena.allocator(), &out, "zz",
        \\pub const host_tests = .{
        \\    @import("../up.zig"),
        \\};
    , "host_tests"));
    try std.testing.expectError(error.UnreadableTestList, appendArchList(arena.allocator(), &out, "zz",
        \\pub const host_tests = .{
        \\    @import("a.zig")
        \\};
    , "host_tests"));
}
