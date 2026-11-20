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
    built_archs = archs;
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

    // Unit tests — reuses the same module config (mlx-c, stb, webp, etc.)
    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/tests.zig"),
        .target = target,
        .optimize = optimize,
        .link_libcpp = true,
        .imports = &.{
            .{ .name = "build_options", .module = build_options.createModule() },
            .{ .name = "stb", .module = addCHeaderModule(b, b.path("lib/stb_image.h"), b.path("lib"), target, optimize) },
            .{ .name = "webp", .module = addCHeaderModule(b, .{ .cwd_relative = "/opt/homebrew/include/webp/decode.h" }, .{ .cwd_relative = "/opt/homebrew/include" }, target, optimize) },
        },
    });

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
    const test_log = b.option([]const u8, "test-log", "Append a line per test (UTC, step, PASS/SKIP/FAIL, ms, name) and each step's DONE totals to this absolute path");
    const fixture_cut = b.option(bool, "fixture-cut", "QWEN4_TEST_MODEL names a Qwen cut or pack: the test steps that load it declare 16 GiB; a test step that sees it without this fails") orelse false;
    var runs: TestRuns = .{
        .b = b,
        .log = test_log,
        .fixture_cut = fixture_cut,
        .skip = b.option([]const []const u8, "test-skip-step", "Leave out this test run step (repeat for several): tests/gate.sh --continue reuses its tests' results") orelse &.{},
    };
    if (test_log) |path| if (!std.fs.path.isAbsolute(path)) std.debug.panic("-Dtest-log={s} is not an absolute path", .{path});
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
        .imports = &.{
            .{ .name = "build_options", .module = host_options.createModule() },
        },
    });
    // bills.zig reaches the transformer's and the model's pure helpers, which import these three modules; no MLX symbol is linked.
    _ = addExl3Module(b, host_mod, target, optimize);
    host_mod.addImport("stb", addCHeaderModule(b, b.path("lib/stb_image.h"), b.path("lib"), target, optimize));
    host_mod.addCSourceFile(.{ .file = b.path("lib/stb_image_impl.c"), .flags = &.{"-O2"} });
    host_mod.addIncludePath(b.path("lib"));
    host_mod.addImport("webp", addCHeaderModule(b, .{ .cwd_relative = "/opt/homebrew/include/webp/decode.h" }, .{ .cwd_relative = "/opt/homebrew/include" }, target, optimize));
    linkApple(host_mod, macos_sdk_frameworks);
    // A user's filter narrows the host tier to its own tests that the filter selects: a bare substring would also select the
    // MLX-tier tests of the files the host files reach, and those do not link. A filter that selects none builds no binary.
    const host_filters = if (test_filter) |user| hostUserFilters(b, user) else hostFilters(b);
    const count_guard = addCountGuard(b, target);
    if (host_filters.len > 0) {
        const host_tests = b.addTest(.{ .name = "test-host", .root_module = host_mod, .filters = host_filters, .test_runner = testRunner(b), .max_rss = 4 * gib });
        test_build.dependOn(&b.addInstallArtifact(host_tests, .{ .dest_dir = .{ .override = .{ .custom = "tests" } } }).step);
        if (runs.add(host_tests, host_tests.name, &.{}, false, 1)) |run_host_tests| {
            test_host_step.dependOn(&run_host_tests.step);
            test_step.dependOn(&run_host_tests.step);
        }
        if (test_filter == null) {
            const host_count = countGuard(b, count_guard, host_tests);
            test_host_step.dependOn(host_count);
            test_step.dependOn(host_count);
        }
    }
    const domains: []const DomainBuild = if (test_model == .exl3 or test_model == .enc)
        &.{}
    else
        addTestDomains(b, test_mod, test_model, test_filter);
    for (domains) |domain| {
        test_build.dependOn(&b.addInstallArtifact(domain.compile, .{ .dest_dir = .{ .override = .{ .custom = "tests" } } }).step);
        for (domain.steps) |step| {
            var env: std.ArrayList([2][]const u8) = .empty;
            env.append(b.allocator, .{ "SUSHI_TEST_SELECT", step.select }) catch @panic("OOM");
            if (step.env) |kv| env.append(b.allocator, kv) catch @panic("OOM");
            if (qwen_preprocess_fixture) |fixture| env.append(b.allocator, .{ "QWEN_PREPROCESS_FIXTURE", fixture }) catch @panic("OOM");
            const run = runs.add(domain.compile, step.name, env.items, domain.domain.cut, domain.domain.run_gib) orelse continue;
            if (qwen_preprocess_fixture) |fixture| {
                run.addFileInput(.{ .cwd_relative = b.fmt("{s}/manifest.json", .{fixture}) });
                run.addFileInput(.{ .cwd_relative = b.fmt("{s}/source_rgb.bin", .{fixture}) });
                run.addFileInput(.{ .cwd_relative = b.fmt("{s}/pixel_values.bin", .{fixture}) });
            }
            test_step.dependOn(&run.step);
        }
        test_step.dependOn(countGuard(b, count_guard, domain.compile));
    }

    // Zig collects tests from an artifact's root module only, so src/exl3 is its own.
    const exl3_tests: []const *std.Build.Step.Compile = if (test_model != null and test_model != .exl3)
        &.{}
    else
        addExl3TestBins(b, exl3_test_mod, test_filter);
    for (exl3_tests) |exl3_test| {
        test_build.dependOn(&b.addInstallArtifact(exl3_test, .{ .dest_dir = .{ .override = .{ .custom = "tests" } } }).step);
        const area = for (exl3_test_bins) |area| {
            if (std.mem.eql(u8, exl3_test.name, b.fmt("test-exl3-{s}", .{area.area}))) break area;
        } else unreachable;
        if (runs.add(exl3_test, exl3_test.name, &.{}, area.cut, area.run_gib)) |run_exl3| test_step.dependOn(&run_exl3.step);
        test_step.dependOn(countGuard(b, count_guard, exl3_test));
    }

    const encoder_tests = b.addTest(.{
        .test_runner = testRunner(b),
        .max_rss = 5 * gib,
        .name = "test-encoder",
        .root_module = encoder_mod,
        .filters = test_filter orelse &.{},
    });
    if ((test_model == null or test_model == .enc) and encoderSelected(b, test_filter)) {
        test_build.dependOn(&b.addInstallArtifact(encoder_tests, .{ .dest_dir = .{ .override = .{ .custom = "tests" } } }).step);
        if (runs.add(encoder_tests, encoder_tests.name, &.{}, false, 6)) |run_encoder| test_step.dependOn(&run_encoder.step);
        test_step.dependOn(countGuard(b, count_guard, encoder_tests));
    }

    if (test_model == null) if (sushibarFilters(b, test_filter)) |filters| {
        const bar_tests = b.addTest(.{
            .test_runner = testRunner(b),
            .max_rss = 1 * gib,
            .name = "test-sushibar",
            .root_module = sushibarModule(b, target, optimize, macos_sdk_frameworks),
            .filters = filters,
        });
        test_build.dependOn(&b.addInstallArtifact(bar_tests, .{ .dest_dir = .{ .override = .{ .custom = "tests" } } }).step);
        if (runs.add(bar_tests, bar_tests.name, &.{}, false, 1)) |run_bar| test_step.dependOn(&run_bar.step);
        test_step.dependOn(countGuard(b, count_guard, bar_tests));
    };

    runs.finish();

    guardBenchFiles(b, build_roots);
    const ubench_step = b.step("ubench", "Build zig-out/bin/sushi-ubench (always ReleaseFast): the microbenches, which no test step builds");
    ubench_step.dependOn(&b.addInstallArtifact(addUbench(b, build_options, target, macos_sdk_frameworks), .{}).step);

    const host_check_step = b.step("exl3-host-check", "Compile the sushi_exl3 module and its tests against the minimal host root src/exl3/host_min.zig (nothing links or runs)");
    for (addExl3HostCheck(b, target)) |check| host_check_step.dependOn(&check.step);

    addArchSlice(b, target, optimize, build_options);
}

/// `-Darchs=qwen4,glm5`: the architectures this build serves, named as `src/arch/registry.zig` names them; empty is
/// every registered one. An arch left out has no prong in the dispatch unions, so none of its code is analysed.
var built_archs: []const []const u8 = &.{};

/// An arch folder (one with `build_roots.zon`) whose arch `-Darchs` leaves out: its tests have no step.
fn archLeftOut(b: *std.Build, folder: []const u8) bool {
    if (built_archs.len == 0 or folder.len == 0) return false;
    for (built_archs) |a| if (std.mem.eql(u8, a, folder)) return false;
    buildRootHandle(b).access(b.graph.io, b.fmt("src/{s}/build_roots.zon", .{folder}), .{}) catch return false;
    return true;
}

fn archList(b: *std.Build) []const []const u8 {
    const raw = b.option([]const u8, "archs", "Architectures to build, comma separated (default: every registered one)") orelse return &.{};
    var out: std.ArrayList([]const u8) = .empty;
    var it = std.mem.tokenizeScalar(u8, raw, ',');
    while (it.next()) |name| out.append(b.allocator, name) catch @panic("OOM");
    return out.items;
}

/// `sushibar`: the menu bar app. It links no MLX or server module, only the Objective-C runtime and AppKit.
fn addSushibar(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, macos_sdk_frameworks: ?[]const u8) *std.Build.Step.Compile {
    return b.addExecutable(.{ .name = "sushibar", .root_module = sushibarModule(b, target, optimize, macos_sdk_frameworks) });
}

fn sushibarModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, macos_sdk_frameworks: ?[]const u8) *std.Build.Module {
    const mod = b.createModule(.{
        .root_source_file = b.path("src/sushibar_main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    if (macos_sdk_frameworks) |fw_path| mod.addFrameworkPath(.{ .cwd_relative = fw_path });
    mod.linkFramework("AppKit", .{});
    mod.linkFramework("Foundation", .{});
    return mod;
}

/// The filters of `test-sushibar`, sushibar's AppKit tests (those of `sushibar_main.zig`; its logic and proc files are
/// host-tier); null when the user's filters select none of them.
fn sushibarFilters(b: *std.Build, user: ?[]const []const u8) ?[]const []const u8 {
    const path = "src/sushibar_main.zig";
    const source = buildRootHandle(b).readFileAlloc(b.graph.io, path, b.allocator, .limited(16 << 20)) catch |e| std.debug.panic("{s}: {t}", .{ path, e });
    b.dependOnFileContents(b.path(path));
    var names: std.ArrayList([]const u8) = .empty;
    appendTestNames(b, &names, "sushibar_main", source);
    const filters = user orelse return &.{"sushibar_main.test."};
    var out: std.ArrayList([]const u8) = .empty;
    for (names.items) |name| if (selectedByUser(filters, name)) out.append(b.allocator, nameFilter(name, names.items, filters)) catch @panic("OOM");
    return if (out.items.len == 0) null else out.items;
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

/// The MLX tests compile as one binary per domain, `test-<name>`, named by what its tests prove. A domain holds every
/// test file of its `folders` (`src/<folder>/`) and the files it lists (`src/`-relative, without `.zig`); a listed file
/// leaves its folder's domain. A test file that no domain holds, or that two list, fails the configure by name.
/// `compile_gib` bounds the compile's RSS, `run_gib` each run step's (`--maxrss` schedules both); a `cut` domain's
/// steps can hold the Qwen cut, which `-Dfixture-cut=true` declares at `cut_run_gib`.
const test_domains = [_]TestDomain{
    .{
        .name = "cooler",
        .cut = true,
        .files = &.{
            "cooler_contract_test",     "cooler_late_test",     "cooler_load_test",    "cooler_pool_test",
            "cooler_prefill_bill_test", "cooler_requests_test", "cooler_restore_test", "cooler_held_requests_test",
            "cooler_held_restore_test",
        },
        .steps = &.{
            .{ .name = "cooler-load", .files = &.{ "cooler_contract_test", "cooler_load_test" } },
            .{ .name = "cooler-pool", .files = &.{"cooler_pool_test"} },
            .{ .name = "cooler-prefill-bill", .files = &.{"cooler_prefill_bill_test"} },
            .{ .name = "cooler-requests", .files = &.{"cooler_requests_test"} },
            .{ .name = "cooler-restore", .files = &.{"cooler_restore_test"} },
            // A served load per fixture held across the step's request-level cells (`heldLoad`), apart from the cells
            // that make their own loads.
            .{ .name = "cooler-held-requests", .files = &.{"cooler_held_requests_test"} },
            .{ .name = "cooler-held-restore", .files = &.{"cooler_held_restore_test"} },
            // Every test of the file runs alone (`alone_tests`).
            .{ .name = "cooler-late", .files = &.{"cooler_late_test"} },
        },
    },
    .{ .name = "deepseek", .folders = &.{"dsv41"}, .compile_gib = 8, .run_gib = 9 },
    .{ .name = "qwen", .folders = &.{"qwen4"}, .cut = true, .run_gib = 6 },
    .{ .name = "glm", .folders = &.{"glm5"} },
    .{ .name = "mimo", .folders = &.{"mimo2"} },
    .{ .name = "kernels", .folders = &.{"tensor"}, .run_gib = 6, .files = &.{
        "attn_kernels",       "dflash_conv",    "dflash_qmv",      "expert_bf16_kernels", "fp8_block",    "linear",     "metal",
        "metal_mlx",          "mlx",            "mlx_kernel_keys", "mlx_pages",           "mlx_util",     "moe_gqmv",   "moe_router",
        "moe_stream_compute", "mtp_dense_rows", "mtp_qmv",         "nax_gate",            "sampler_draw", "verify_moe", "verify_qmm",
    } },
    .{ .name = "generate", .files = &.{
        "arm_trial",     "degenerate",   "dflash",             "forward_diag", "generate",       "hc_prefill",   "kv_cache",
        "kv_quant",      "logprobs",     "mtp",                "mtp_plan",     "mtp_policy",     "mtp_round",    "mtp_tick",
        "prefill_plan",  "restore_dump", "sampler",            "scheduler",    "slot_admission", "slot_outcome", "ssm_cache",
        "think_penalty", "trunk",        "trunk_helpers_test", "trunk_stq",
    } },
    // `tests` is the MLX root: its own test only names the files the binaries reach.
    .{ .name = "load", .folders = &.{"arch"}, .files = &.{
        "arch_host_stub", "bills_live_test", "exl3_flat_layout_test", "expert_io",     "expert_store", "expert_stream",
        "hidden_capture", "imatrix",         "kld",                   "layer_weights", "model",        "model_discovery",
        "model_load",     "model_registry",  "owned_weights",         "pack_index",    "tests",        "transient_peak",
        "vision_common",
    } },
    .{ .name = "server", .compile_gib = 10, .files = &.{
        "anthropic",     "bench",          "byte_budget", "cli",           "completions", "downloads",      "generation_stream",
        "http_conn",     "http_head",      "launch",      "main",          "metrics",     "model_settings", "models_api",
        "request_media", "request_policy", "responses",   "responses_api", "server",      "server_test",    "settings_api",
        "sleep_inhibit", "status",         "version",     "vision_queue",
    } },
    // The SSD tier's tests run once on each file backend (`SUSHI_PREFIX_FS`); a `-Dtest-filter` run, on the default.
    .{ .name = "prefix-cache", .files = &.{
        "disk_placement",       "dsv41/dsv41_prefix_cache_test", "glm5/glm5_prefix_cache_test", "kv_disk_cache",
        "kv_disk_fs",           "prefix_cache",                  "prefix_cache_admin",          "prefix_vfs_tensors_test",
        "qwen4/qwen4_disk_qsa", "qwen4/qwen4_prefix_cache_test",
    }, .steps = &.{
        .{ .name = "test-prefix-cache" },
        .{ .name = "test-prefix-cache (SUSHI_PREFIX_FS=real)", .env = .{ "SUSHI_PREFIX_FS", "real" }, .again = true },
    } },
};

/// Tests that run alone, each in a process of its own (a step named `step`), left out of their file's step: what they
/// prove holds only on state the process starts with. `name` is the test's name in `src/<file>.zig`.
const alone_tests = [_]AloneTest{
    .{ .step = "cooler-late-widest-tinies", .file = "cooler_late_test", .name = "cooler contract: after a Cooler load's served warm, a prompt at the widest chunk and a repetitive one past a sparse indexer's budget, greedy, speculative and sampled, compile no kernel, on the tiny fixtures" },
    .{ .step = "cooler-late-widest-cut", .file = "cooler_late_test", .name = "cooler contract: after a Cooler load's served warm, a prompt at the widest chunk and a repetitive one past a sparse indexer's budget, greedy, speculative and sampled, compile no kernel, on the cut" },
    .{ .step = "cooler-late-lengths-tinies", .file = "cooler_late_test", .name = "cooler contract: after a Cooler load's served warm, greedy requests of every prompt length to 40, and speculative and sampled ones, compile no kernel, on the tiny fixtures" },
    .{ .step = "cooler-late-lengths-cut", .file = "cooler_late_test", .name = "cooler contract: after a Cooler load's served warm, greedy requests of every prompt length to 40, and speculative and sampled ones, compile no kernel, on the cut" },
    .{ .step = "cooler-late-restore", .file = "cooler_late_test", .name = "cooler contract: after a Cooler load's served warm, a served request that restores the live state behind its reply compiles no kernel" },
    .{ .step = "cooler-late-bench", .file = "cooler_late_test", .name = "cooler contract: after a Cooler load's served warm, the release bench's first request through the scheduler compiles no kernel" },
    .{ .step = "deepseek-late-new-row-count", .file = "dsv41/dsv41_kernels", .name = "dsv41 kernels: a new row count, pick count, split count or context width compiles no new Metal library" },
    .{ .step = "deepseek-late-image-splice", .file = "dsv41/dsv41_generate_test", .name = "dsv41 engine: a forward that splices image rows runs on the engine at a prefill width and at a decode width, with no op on MLX and no kernel compiled after the warm (DSV41_STQ_TINY)" },
    .{ .step = "deepseek-late-nax-gemm", .file = "dsv41/dsv41_generate_test", .name = "dsv41 engine: a load with decode on the engine keeps the NAX GEMM arm, and after its warm nothing compiles a kernel: a forward of every width from 1 to 16 and a DSpark round on either side, a round whose rows take two attention split counts, a restore point captured at any position, a step or round of any width, the split top-k a one-row step takes over 131072 keys, a draft at any fill of the stages' windows; a decode forward, a draft and a verify round of the same width build no kernel instance key once they have run (DSV41_STQ_TINY)" },
    .{ .step = "deepseek-late-no-drafter", .file = "dsv41/dsv41_generate_test", .name = "dsv41 engine: a load with no drafter compiles nothing after the load at any forward width from 1 to 16 from any position, STQ or the release's layout (DSV41_STQ_TINY, DSV41_TINY)" },
    .{ .step = "deepseek-late-cut-widths", .file = "dsv41/dsv41_generate_test", .name = "dsv41 engine: after the load's wide warm, prompts cut at other widths compile no kernel, through the union (DSV41_STQ_TINY)" },
    .{ .step = "deepseek-late-sampled-dspark", .file = "dsv41/dsv41_generate_test", .name = "dsv41 engine: after the load's warm a sampled DSpark request, seeded or not, compiles no kernel, at any filter setting or with a suppress mask (DSV41_STQ_TINY)" },
    .{ .step = "deepseek-late-release-layout", .file = "dsv41/dsv41_generate_test", .name = "dsv41 engine: the release's layout (FP8 trunk, MXFP4 experts) on Sushi Cooler gives the same logits and DSpark tokens at each round-width mix, every expert cached or read at each forward, and a load that reads them asks MLX nothing and compiles nothing after the load (DSV41_TINY)" },
    .{ .step = "deepseek-late-joined-rows", .file = "dsv41/dsv41_generate_test", .name = "dsv41 engine: a load that joins rows, with DSpark or without, compiles nothing after its warm at joined rows of every width (DSV41_STQ_TINY)" },
    .{ .step = "deepseek-late-manifest", .file = "dsv41/dsv41_pipelines", .name = "dsv41 pipelines: a load with no manifest writes one with the wide chunks' recipes, and a load from it skips the engine's forwards, the wide ones included, serves the same tokens and compiles none inside a prompt (DSV41_STQ_TINY)" },
    .{ .step = "deepseek-late-topk-tiles", .file = "dsv41/dsv41_pipelines", .name = "dsv41 pipelines: a chunk whose index top-k streams over key tiles compiles nothing inside a prompt, after a full warm and after a load from the manifest (DSV41_STQ_TINY)" },
    .{ .step = "deepseek-late-stream", .file = "dsv41/dsv41_stream", .name = "dsv41 kernels: after the load's warm, prompts of any length, every verify width and a growing context compile no Metal library on the STQ pack, streamed and resident, DSpark on and off (DSV41_STQ_TINY)" },
    .{ .step = "qwen-late-gdn-captured-verify", .file = "qwen4/gdn_decode", .name = "gdn recur: on the engine before the seal a captured verify makes the replay kernel at each count up to its own" },
    .{ .step = "qwen-late-gdn-final-state", .file = "qwen4/gdn_decode", .name = "gdn recur: on the engine before the seal a verify that keeps only its final state makes the replay at each count up to its own" },
    .{ .step = "qwen-late-vision-tower", .file = "qwen4/qwen_vision", .name = "qwen vision: a Cooler load's tower warms before the seal, so images and videos of any grid compile nothing inside a request" },
    .{ .step = "qwen-late-qsa-split", .file = "qwen4/qwen4_qsa_select", .name = "qsa select on the engine before the seal makes the split's pipelines, so a long select after it compiles none" },
    .{ .step = "qwen-late-qsa-row-class", .file = "qwen4/qwen4_qsa_select", .name = "qsa select before the seal, at the one row class a warm's wide chunk takes, makes every row class's, so a chunk's remainder of any width compiles none" },
    .{ .step = "kernels-late-f32-simd", .file = "tensor/route", .name = "engine f32 matmul on the SIMD tiles: the first use of a tile before the seal makes every alignment of it, so no length compiles inside a request" },
    .{ .step = "kernels-late-bf16-nax", .file = "tensor/route", .name = "engine bf16 matmul on the NAX tiles: a weight's first product before the seal makes what every row count to the widest chunk picks, so no length compiles inside a request" },
    .{ .step = "kernels-late-by-size", .file = "tensor/route", .name = "engine kernels a request's lengths pick by size: an elementwise op's and a matrix-vector product's first use before the seal makes every size's, so a long prompt compiles nothing" },
};

/// What every `alone_tests` entry proves: no kernel compiles after a load's warm or seal, which only a process whose
/// pipeline cache that load filled can show (an earlier test's compiles would hide a late one).
const AloneTest = struct { step: []const u8, file: []const u8, name: []const u8 };

const TestDomain = struct {
    name: []const u8,
    folders: []const []const u8 = &.{},
    files: []const []const u8 = &.{},
    compile_gib: u64 = 6,
    run_gib: u64 = 4,
    cut: bool = false,
    /// The run steps that share the binary, each a process of its own; none means one, `test-<name>`, over every file.
    steps: []const RunStep = &.{},
};

/// One process of a domain's binary: the tests of `files` (none: every file of the domain), under `env`. An `again`
/// step runs tests another step runs, on another backend: only an unfiltered run takes it.
const RunStep = struct {
    name: []const u8,
    files: []const []const u8 = &.{},
    env: ?[2][]const u8 = null,
    again: bool = false,
};

const cut_run_gib: u64 = 16;
const gib: u64 = 1 << 30;

/// The model folders hold their own test files beside the root's; src/exl3 and src/encoder are modules of their own.
const test_dirs = [_][]const u8{ "src", "src/arch", "src/tensor", "src/qwen4", "src/mimo2", "src/glm5", "src/dsv41" };

/// A test's model is its file's folder; the src root is shared.
const TestModel = enum { qwen4, mimo2, glm5, dsv41, exl3, enc };

/// Zig keeps a test when its fully qualified name holds any filter, and filters OR, so a user's filter cannot narrow
/// a domain's: the scan names each test in full and keeps those the user's filters select.
fn selectedByUser(user: ?[]const []const u8, full: []const u8) bool {
    const filters = user orelse return true;
    for (filters) |f| if (std.mem.indexOf(u8, full, f) != null) return true;
    return false;
}

/// The longest piece of `full` between quotes and backslashes that no name of `others` holds. Filters reach `zig test`
/// in a response file whose parser reads a quote or backslash as syntax; null when every piece selects another test.
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

/// One test as the runner names it: `<path>.test.<name>`, `<path>.<Struct>.test.<name>` in a struct opened at column 0,
/// or `<path>.test_<n>` for an anonymous block, `<path>` being the file's `src/`-relative path with `.` for `/`.
const TestName = struct {
    full: []const u8,
    /// An anonymous block survives every filter, so it is never one.
    anonymous: bool = false,
    /// A test nested where the scan cannot name it; its file's prefix still selects it.
    unnamed: bool = false,
};

const TestFile = struct {
    /// `src/`-relative, without `.zig`.
    path: []const u8,
    /// The prefix every test of the file carries: `<path with . for />.`.
    prefix: []const u8,
    names: []const TestName,
    in_scope: bool,
    host: bool,
    domain: usize = 0,
};

fn testFolder(path: []const u8) []const u8 {
    return if (std.mem.lastIndexOfScalar(u8, path, '/')) |i| path[0..i] else "";
}

/// The tests of one file, as `TestName`s; in an MLX file (`strict`), a test line the build cannot cut out fails, naming it.
fn scanTestNames(b: *std.Build, path: []const u8, prefix: []const u8, source: []const u8, strict: bool) []const TestName {
    var out: std.ArrayList(TestName) = .empty;
    var owner: []const u8 = "";
    var anonymous: usize = 0;
    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |line| {
        if ((std.mem.startsWith(u8, line, "pub const ") or std.mem.startsWith(u8, line, "const ")) and std.mem.endsWith(u8, line, " = struct {"))
            owner = line[std.mem.indexOf(u8, line, "const ").? + "const ".len .. line.len - " = struct {".len];
        const trimmed = std.mem.trimStart(u8, line, " ");
        const indent = line.len - trimmed.len;
        if (std.mem.startsWith(u8, trimmed, "test {")) {
            out.append(b.allocator, .{ .full = b.fmt("{s}test_{d}", .{ prefix, anonymous }), .anonymous = true, .unnamed = indent != 0 }) catch @panic("OOM");
            if (indent == 0) anonymous += 1;
            continue;
        }
        if (!std.mem.startsWith(u8, trimmed, "test \"")) continue;
        // A nested name this scan cannot cut out would drop its test from every binary.
        const clean = std.mem.endsWith(u8, line, "\" {");
        if (strict and indent != 0 and (!clean or std.mem.indexOfScalar(u8, line, '\\') != null))
            std.debug.panic("src/{s}.zig: write `test \"name\" {{` on one line without escapes: {s}", .{ path, line });
        if (!clean) {
            out.append(b.allocator, .{ .full = b.fmt("{s}test.?", .{prefix}), .unnamed = true }) catch @panic("OOM");
            continue;
        }
        const name = trimmed["test \"".len .. trimmed.len - "\" {".len];
        if (indent == 0) {
            out.append(b.allocator, .{ .full = b.fmt("{s}test.{s}", .{ prefix, name }) }) catch @panic("OOM");
        } else {
            const named = indent == 4 and owner.len != 0;
            out.append(b.allocator, .{ .full = b.fmt("{s}{s}.test.{s}", .{ prefix, if (named) owner else "?", name }), .unnamed = !named }) catch @panic("OOM");
        }
    }
    return out.items;
}

/// Every file whose tests the MLX root reaches, the host tier's included (their names keep a domain's filters off
/// them), with its domain; a file no domain holds fails here.
fn scanTestFiles(b: *std.Build, model: ?TestModel) []TestFile {
    const io = b.graph.io;
    const host_paths = hostPaths(b);
    var files: std.ArrayList(TestFile) = .empty;
    for (test_dirs) |dir_path| {
        const folder = if (std.mem.eql(u8, dir_path, "src")) "" else dir_path["src/".len..];
        const in_scope = !archLeftOut(b, folder) and if (model) |m| folder.len == 0 or std.mem.eql(u8, folder, "arch") or std.mem.eql(u8, folder, "tensor") or std.mem.eql(u8, folder, @tagName(m)) else true;
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
            if (std.mem.startsWith(u8, stem, "ubench_")) continue;
            // sushibar's own binary, `test-sushibar`, runs its tests; tests_host roots the host tier.
            if (folder.len == 0 and (std.mem.eql(u8, stem, "sushibar_main") or std.mem.eql(u8, stem, "tests_host"))) continue;
            const path = if (folder.len == 0) b.dupe(stem) else b.fmt("{s}/{s}", .{ folder, stem });
            b.dependOnFileContents(b.path(b.fmt("{s}/{s}", .{ dir_path, entry.name })));
            const source = src.readFileAlloc(io, entry.name, b.allocator, .limited(16 << 20)) catch |e| std.debug.panic("{s}: {t}", .{ entry.name, e });
            const prefix = b.fmt("{s}.", .{std.mem.replaceOwned(u8, b.allocator, path, "/", ".") catch @panic("OOM")});
            const host = isHostFile(host_paths, dir_path, stem);
            const names = scanTestNames(b, path, prefix, source, !host);
            if (names.len == 0) continue;
            files.append(b.allocator, .{ .path = path, .prefix = prefix, .names = names, .in_scope = in_scope, .host = host }) catch @panic("OOM");
        }
    }
    // Host files outside the scanned folders (src/template/) are reached by the MLX root too.
    for (host_paths) |path| {
        const folder = testFolder(path);
        const scanned = for (test_dirs) |d| {
            if (std.mem.eql(u8, d, if (folder.len == 0) "src" else b.fmt("src/{s}", .{folder}))) break true;
        } else false;
        if (scanned) continue;
        const source = buildRootHandle(b).readFileAlloc(io, b.fmt("src/{s}.zig", .{path}), b.allocator, .limited(16 << 20)) catch |e| switch (e) {
            error.FileNotFound => continue,
            else => std.debug.panic("src/{s}.zig: {t}", .{ path, e }),
        };
        const prefix = b.fmt("{s}.", .{std.mem.replaceOwned(u8, b.allocator, path, "/", ".") catch @panic("OOM")});
        files.append(b.allocator, .{ .path = path, .prefix = prefix, .names = scanTestNames(b, path, prefix, source, false), .in_scope = false, .host = true }) catch @panic("OOM");
    }
    // Directory order varies; the filters key the compile step's cache.
    std.mem.sortUnstable(TestFile, files.items, {}, struct {
        fn lt(_: void, x: TestFile, y: TestFile) bool {
            return std.mem.lessThan(u8, x.path, y.path);
        }
    }.lt);
    for (files.items) |*f| {
        if (f.host) continue;
        f.domain = domainOf(f.path) orelse
            std.debug.panic("src/{s}.zig holds tests that no test_domains entry holds: list it in build.zig", .{f.path});
    }
    for (test_domains) |d| for (d.files) |listed| {
        const found = for (files.items) |f| {
            if (!f.host and std.mem.eql(u8, f.path, listed)) break true;
        } else false;
        if (found) continue;
        // A dropped architecture's folder is gone with its files.
        const folder = testFolder(listed);
        if (folder.len != 0) {
            var dir = buildRootHandle(b).openDir(io, b.fmt("src/{s}", .{folder}), .{}) catch continue;
            dir.close(io);
        }
        std.debug.panic("test_domains lists {s} under test-{s}, which is no MLX test file", .{ listed, d.name });
    };
    return files.items;
}

/// The domain that lists `path`, else the one whose folder holds it; two that list it fail.
fn domainOf(path: []const u8) ?usize {
    var listed: ?usize = null;
    for (test_domains, 0..) |d, i| for (d.files) |f| if (std.mem.eql(u8, f, path)) {
        if (listed) |first| std.debug.panic("test_domains lists {s} under test-{s} and test-{s}", .{ path, test_domains[first].name, d.name });
        listed = i;
    };
    if (listed) |i| return i;
    const folder = testFolder(path);
    if (folder.len == 0) return null;
    for (test_domains, 0..) |d, i| for (d.folders) |f| if (std.mem.eql(u8, f, folder)) return i;
    return null;
}

/// A domain's binary and the run steps that share it.
const DomainBuild = struct {
    compile: *std.Build.Step.Compile,
    domain: TestDomain,
    steps: []const PlannedStep,
};

const PlannedStep = struct {
    name: []const u8,
    /// `SUSHI_TEST_SELECT`: the prefix of each file the step runs.
    select: []const u8,
    env: ?[2][]const u8,
};

fn addTestDomains(b: *std.Build, root: *std.Build.Module, model: ?TestModel, user: ?[]const []const u8) []const DomainBuild {
    const files = scanTestFiles(b, model);
    var all_names: std.ArrayList([]const u8) = .empty;
    for (files) |f| for (f.names) |n| all_names.append(b.allocator, n.full) catch @panic("OOM");

    var planned: std.ArrayList(struct { domain: usize, step: PlannedStep, again: bool }) = .empty;
    for (test_domains, 0..) |d, di| {
        const steps: []const RunStep = if (d.steps.len > 0) d.steps else &.{.{ .name = b.fmt("test-{s}", .{d.name}) }};
        for (steps) |s| {
            if (s.again and user != null) continue;
            var select: std.ArrayList(u8) = .empty;
            for (files) |f| {
                if (f.host or f.domain != di or !f.in_scope or !stepHolds(s, f.path)) continue;
                select.print(b.allocator, "{s}\n", .{f.prefix}) catch @panic("OOM");
                for (alone_tests) |t| if (std.mem.eql(u8, t.file, f.path))
                    select.print(b.allocator, "!{s}test.{s}\n", .{ f.prefix, t.name }) catch @panic("OOM");
            }
            // A step is made when it runs a test the user's filters select (any named test without filters).
            var runs_one = false;
            for (files) |f| {
                if (f.host or f.domain != di or !f.in_scope or !stepHolds(s, f.path)) continue;
                for (f.names) |n| runs_one = runs_one or (!n.anonymous and selectedByUser(user, n.full) and selects(select.items, n.full));
            }
            if (runs_one) planned.append(b.allocator, .{ .domain = di, .step = .{ .name = s.name, .select = select.items, .env = s.env }, .again = s.again }) catch @panic("OOM");
        }
        for (alone_tests) |t| {
            const f = for (files) |f| {
                if (!f.host and std.mem.eql(u8, f.path, t.file)) break f;
            } else std.debug.panic("alone_tests names src/{s}.zig, which holds no MLX test", .{t.file});
            if (f.domain != di or !f.in_scope) continue;
            const full = b.fmt("{s}test.{s}", .{ f.prefix, t.name });
            var held: usize = 0;
            for (f.names) |n| held += @intFromBool(std.mem.startsWith(u8, n.full, full));
            if (held != 1) std.debug.panic("alone_tests' step {s} would run {d} tests, not one: {s}", .{ t.step, held, full });
            if (!selectedByUser(user, full)) continue;
            planned.append(b.allocator, .{ .domain = di, .step = .{ .name = t.step, .select = b.fmt("{s}\n", .{full}), .env = null }, .again = false }) catch @panic("OOM");
        }
        for (d.steps) |s| for (s.files) |sf| if (domainOf(sf) != di)
            std.debug.panic("test-{s}'s step {s} names {s}, a file of another domain", .{ d.name, s.name, sf });
    }
    // Every test the steps can see runs in exactly one of them (a second backend's run aside).
    for (files) |f| {
        if (f.host or !f.in_scope) continue;
        for (f.names) |n| {
            if (user != null and (n.anonymous or !selectedByUser(user, n.full))) continue;
            var runs: usize = 0;
            for (planned.items) |p| {
                if (!p.again and selects(p.step.select, n.full)) runs += 1;
            }
            if (runs != 1) std.debug.panic("the test '{s}' runs in {d} test steps, not one: check test_domains", .{ n.full, runs });
        }
    }

    var out: std.ArrayList(DomainBuild) = .empty;
    for (test_domains, 0..) |d, di| {
        var steps: std.ArrayList(PlannedStep) = .empty;
        for (planned.items) |p| if (p.domain == di) steps.append(b.allocator, p.step) catch @panic("OOM");
        if (steps.items.len == 0) continue;
        // A file's prefix filter is kept when it selects only the domain's tests; else its tests are named one by one.
        var others: std.ArrayList([]const u8) = .empty;
        for (files) |f| if (f.host or f.domain != di or !f.in_scope) for (f.names) |n| others.append(b.allocator, n.full) catch @panic("OOM");
        var filters: std.ArrayList([]const u8) = .empty;
        for (files) |f| {
            if (f.host or f.domain != di or !f.in_scope) continue;
            if (user == null) {
                const exclusive = for (others.items) |o| {
                    if (std.mem.indexOf(u8, o, f.prefix) != null) break false;
                } else true;
                if (exclusive) {
                    filters.append(b.allocator, f.prefix) catch @panic("OOM");
                    continue;
                }
            }
            for (f.names) |n| {
                if (n.anonymous or !selectedByUser(user, n.full)) continue;
                if (n.unnamed) std.debug.panic("src/{s}.zig: cannot name the nested test {s}; nest it in a struct opened at column 0", .{ f.path, n.full });
                filters.append(b.allocator, if (user) |u| nameFilter(n.full, all_names.items, u) else domainFilter(n.full, others.items)) catch @panic("OOM");
            }
        }
        const compile = b.addTest(.{ .name = b.fmt("test-{s}", .{d.name}), .root_module = root, .filters = filters.items, .test_runner = testRunner(b), .max_rss = d.compile_gib * gib });
        out.append(b.allocator, .{ .compile = compile, .domain = d, .steps = steps.items }) catch @panic("OOM");
    }
    return out.items;
}

fn stepHolds(step: RunStep, path: []const u8) bool {
    if (step.files.len == 0) return true;
    for (step.files) |f| if (std.mem.eql(u8, f, path)) return true;
    return false;
}

/// The runner's reading of `SUSHI_TEST_SELECT` (scripts/test_runner.zig `isSelected`).
fn selects(select: []const u8, name: []const u8) bool {
    var kept = false;
    var it = std.mem.splitScalar(u8, select, '\n');
    while (it.next()) |line| {
        if (line.len == 0) continue;
        if (line[0] == '!') {
            if (std.mem.startsWith(u8, name, line[1..])) return false;
        } else if (std.mem.startsWith(u8, name, line)) kept = true;
    }
    return kept;
}

/// A filter for one test that selects no name of `others`: its full name, or its longest quote-free piece.
fn domainFilter(full: []const u8, others: []const []const u8) []const u8 {
    var best: ?[]const u8 = null;
    var pieces = std.mem.tokenizeAny(u8, full, "'\"\\");
    while (pieces.next()) |piece| {
        if (best) |b| if (b.len >= piece.len) continue;
        for (others) |other| {
            if (std.mem.indexOf(u8, other, piece) != null) break;
        } else best = piece;
    }
    return best orelse std.debug.panic("rename the test \"{s}\": every quote-free part of it is in a test of another domain", .{full});
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

/// The exl3 tests run as one binary per area, `test-exl3-<area>`: the tests of the src/exl3 files listed for it, with
/// its compile's and its run's RSS bounds in GiB (as `test_domains`). Every file that holds a test is listed once, and
/// every listed file holds one.
const exl3_test_bins = [_]Exl3Bin{
    .{ .area = "experts", .compile_gib = 3, .run_gib = 8, .files = &.{ "expert_exl3_kernels", "expert_exl3", "rate_groups", "project_rows", "coop_rows2", "prefill_grid", "sorted_windows", "geometry_5120x2304_test" } },
    .{ .area = "trunk", .compile_gib = 2, .run_gib = 8, .cut = true, .files = &.{ "linear", "trunk", "trunk_gemm" } },
    .{ .area = "format", .compile_gib = 2, .run_gib = 4, .files = &.{ "root", "device", "kernel_source", "codebook_tables", "side_cache", "layout", "pack" } },
};

const Exl3Bin = struct { area: []const u8, compile_gib: u64, run_gib: u64, cut: bool = false, files: []const []const u8 };

/// The binary `exl3_test_bins` lists the file `stem` under.
fn exl3Bin(stem: []const u8) usize {
    var bin: ?usize = null;
    for (exl3_test_bins, 0..) |area, i| for (area.files) |listed| if (std.mem.eql(u8, listed, stem)) {
        if (bin != null) std.debug.panic("exl3_test_bins lists src/exl3/{s}.zig under two binaries", .{stem});
        bin = i;
    };
    return bin orelse std.debug.panic("src/exl3/{s}.zig holds tests that exl3_test_bins lists under no binary", .{stem});
}

const Exl3Test = struct { filter: []const u8, bin: usize };

/// The exl3 tests compile and run as the binaries of `exl3_test_bins`. A filter matches `<file stem>.test.<name>`,
/// so each test is a filter of its own, found by scanning src/exl3.
fn addExl3TestBins(b: *std.Build, root: *std.Build.Module, user: ?[]const []const u8) []const *std.Build.Step.Compile {
    const io = b.graph.io;
    var tests: std.ArrayList(Exl3Test) = .empty;
    var all_names: std.ArrayList([]const u8) = .empty;
    var all_bins: std.ArrayList(usize) = .empty;
    var with_tests: std.ArrayList([]const u8) = .empty;
    var dir = buildRootHandle(b).openDir(io, "src/exl3", .{ .iterate = true }) catch |e| std.debug.panic("src/exl3/: {t}", .{e});
    defer dir.close(io);
    b.dependOnDirectoryContents(b.path("src/exl3"));
    var files = dir.iterate();
    while (files.next(io) catch |e| std.debug.panic("src/exl3/: {t}", .{e})) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".zig")) continue;
        // host_min.zig roots exl3-host-check, which compiles its test; the exl3 root reaches it from no binary.
        if (std.mem.eql(u8, entry.name, "host_min.zig")) continue;
        const stem = b.dupe(entry.name[0 .. entry.name.len - ".zig".len]);
        const source = dir.readFileAlloc(io, entry.name, b.allocator, .limited(8 << 20)) catch |e| std.debug.panic("{s}: {t}", .{ entry.name, e });
        b.dependOnFileContents(b.path(b.fmt("src/exl3/{s}", .{entry.name})));
        var bin: ?usize = null;
        var lines = std.mem.splitScalar(u8, source, '\n');
        while (lines.next()) |line| {
            if (!std.mem.startsWith(u8, line, "test \"")) continue;
            // A name this scan cannot cut out would drop its test from every binary.
            if (!std.mem.endsWith(u8, line, "\" {") or std.mem.indexOfScalar(u8, line, '\\') != null)
                std.debug.panic("{s}: write `test \"name\" {{` on one line without escapes: {s}", .{ entry.name, line });
            const name = line["test \"".len .. line.len - "\" {".len];
            if (bin == null) {
                bin = exl3Bin(stem);
                with_tests.append(b.allocator, stem) catch @panic("OOM");
            }
            const full = b.fmt("{s}.test.{s}", .{ stem, name });
            all_names.append(b.allocator, full) catch @panic("OOM");
            all_bins.append(b.allocator, bin.?) catch @panic("OOM");
            if (selectedByUser(user, full)) tests.append(b.allocator, .{ .filter = full, .bin = bin.? }) catch @panic("OOM");
        }
    }
    for (exl3_test_bins) |area| for (area.files) |listed| {
        for (with_tests.items) |stem| {
            if (std.mem.eql(u8, stem, listed)) break;
        } else std.debug.panic("exl3_test_bins lists {s}, which is no src/exl3 file with tests", .{listed});
    };
    if (user) |u| for (tests.items) |*t| {
        t.filter = nameFilter(t.filter, all_names.items, u);
    };
    // A filter matches anywhere in a name: one that also selects a test of another binary would run that test twice.
    for (tests.items) |t| for (all_names.items, all_bins.items) |other, bin| {
        if (bin != t.bin and std.mem.indexOf(u8, other, t.filter) != null)
            std.debug.panic("the exl3 filter '{s}' also selects '{s}' of another binary in exl3_test_bins", .{ t.filter, other });
    };
    var out: std.ArrayList(*std.Build.Step.Compile) = .empty;
    for (exl3_test_bins, 0..) |area, i| {
        var filters: std.ArrayList([]const u8) = .empty;
        for (tests.items) |t| if (t.bin == i) filters.append(b.allocator, t.filter) catch @panic("OOM");
        if (filters.items.len == 0) continue;
        out.append(b.allocator, b.addTest(.{ .name = b.fmt("test-exl3-{s}", .{area.area}), .root_module = root, .filters = filters.items, .test_runner = testRunner(b), .max_rss = area.compile_gib * gib })) catch @panic("OOM");
    }
    return out.items;
}

/// Whether the user's filters select a test of src/encoder, the encoder binary's root module; with none, no binary.
fn encoderSelected(b: *std.Build, user: ?[]const []const u8) bool {
    const filters = user orelse return true;
    const io = b.graph.io;
    var dir = buildRootHandle(b).openDir(io, "src/encoder", .{ .iterate = true }) catch |e| std.debug.panic("src/encoder/: {t}", .{e});
    defer dir.close(io);
    var walker = dir.walk(b.allocator) catch @panic("OOM");
    defer walker.deinit();
    var names: std.ArrayList([]const u8) = .empty;
    while (walker.next(io) catch |e| std.debug.panic("src/encoder/: {t}", .{e})) |entry| {
        if (entry.kind == .directory) b.dependOnDirectoryContents(b.path(b.fmt("src/encoder/{s}", .{entry.path})));
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.path, ".zig")) continue;
        const source = dir.readFileAlloc(io, entry.path, b.allocator, .limited(8 << 20)) catch |e| std.debug.panic("{s}: {t}", .{ entry.path, e });
        b.dependOnFileContents(b.path(b.fmt("src/encoder/{s}", .{entry.path})));
        const stem = std.mem.replaceOwned(u8, b.allocator, entry.path[0 .. entry.path.len - ".zig".len], "/", ".") catch @panic("OOM");
        appendTestNames(b, &names, stem, source);
    }
    b.dependOnDirectoryContents(b.path("src/encoder"));
    for (names.items) |name| if (selectedByUser(filters, name)) return true;
    return false;
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
    // Sushi's own files behind the three names reach the module too (the engine's stock ops ask its device profile).
    files.addImport("sushi_exl3", exl3);
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

/// A test reads its fixtures and engine switches from the environment, which neither the configure cache nor a run's
/// cache key sees: the run always executes, so a pass made without a fixture never stands for a run with it.
fn runEveryTime(run: *std.Build.Step.Run) void {
    run.has_side_effects = true;
}

/// The test run steps, each a process of one test binary, and their plan, which `zig build test-plan` writes to
/// `zig-out/test-plan.json`: each step's name, its binary's path beside the plan (`zig build test-build`) and the env it sets. tests/gate.sh
/// keys a step by them (tests/lib/test_keys.py) and leaves out, with `-Dtest-skip-step`, the steps it reuses.
const TestRuns = struct {
    b: *std.Build,
    log: ?[]const u8,
    fixture_cut: bool,
    skip: []const []const u8,
    plan: std.ArrayList(PlanRow) = .empty,

    const PlanRow = struct { step: []const u8, binary: []const u8, env: []const [2][]const u8 };

    /// The run step `name` of `compile` with `env` set, or null when `-Dtest-skip-step` names it. Its RSS bound for
    /// `--maxrss` is `run_gib` (Metal is not in RSS, so it is generous); a step that can load the Qwen cut holds
    /// `cut_run_gib` under `-Dfixture-cut=true`, and without it its runner refuses QWEN4_TEST_MODEL.
    fn add(self: *TestRuns, compile: *std.Build.Step.Compile, name: []const u8, env: []const [2][]const u8, holds_cut: bool, run_gib: u64) ?*std.Build.Step.Run {
        const b = self.b;
        var all: std.ArrayList([2][]const u8) = .empty;
        all.append(b.allocator, .{ "SUSHI_TEST_STEP", name }) catch @panic("OOM");
        all.appendSlice(b.allocator, env) catch @panic("OOM");
        if (holds_cut) all.append(b.allocator, .{ "SUSHI_TEST_FIXTURE_CUT", if (self.fixture_cut) "1" else "0" }) catch @panic("OOM");
        self.plan.append(b.allocator, .{ .step = name, .binary = b.fmt("tests/{s}", .{compile.name}), .env = all.items }) catch @panic("OOM");
        for (self.skip) |s| if (std.mem.eql(u8, s, name)) return null;
        const run = b.addRunArtifact(compile);
        runEveryTime(run);
        run.setName(b.fmt("run {s}", .{name}));
        for (all.items) |kv| run.setEnvironmentVariable(kv[0], kv[1]);
        if (self.log) |path| run.setEnvironmentVariable("SUSHI_TEST_LOG", path);
        run.step.max_rss = (if (self.fixture_cut and holds_cut) cut_run_gib else run_gib) * gib;
        return run;
    }

    /// Writes the plan for `zig build test-plan`; a `-Dtest-skip-step` that names no run step fails.
    fn finish(self: *TestRuns) void {
        const b = self.b;
        for (self.skip) |s| {
            for (self.plan.items) |row| {
                if (std.mem.eql(u8, row.step, s)) break;
            } else std.debug.panic("-Dtest-skip-step={s} names no test run step", .{s});
        }
        const json = std.json.Stringify.valueAlloc(b.allocator, self.plan.items, .{}) catch @panic("OOM");
        const file = b.addWriteFiles().add("test-plan.json", json);
        const step = b.step("test-plan", "Write zig-out/test-plan.json: each test run step's name, binary and env");
        step.dependOn(&b.addInstallFileWithDir(file, .prefix, "test-plan.json").step);
    }
};

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

/// The encoder behind `sushi convert exl3`, also tested alone as `test-encoder`.
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

/// Every test binary runs under sushi's runner: std's, plus the check after each test that the engine is at rest.
fn testRunner(b: *std.Build) std.Build.Step.Compile.TestRunner {
    return .{ .path = b.path("scripts/test_runner.zig"), .mode = .server };
}
