//! CLI subcommands — `sushi run|pull|list <model>` — Ollama-grade
//! ergonomics for the terminal.
//!
//!   sushi run qwen3.8-flash-next          # download if missing, serve, drop into a REPL
//!   sushi pull qwen3.8-flash-next:2.6bpw  # download only
//!   sushi run mimo-v2.6-flash             # MiMo-V2.6-Flash (128 GB)
//!   sushi list                            # what's on disk
//!
//! Short names resolve through the published Sushi packs (`aliases`);
//! anything containing '/' is treated as a HuggingFace repo id directly ("org/repo", with optional
//! "hf.co/" prefix and ":tag" suffix). Downloads land in
//! `~/.sushi/models/<org>/<repo>` — the single source of truth shared
//! with the app's DownloadManager and the server's media-dep resolution.
//!
//! Downloads use system curl for TLS and resume. The embedded REPL uses an
//! in-process HTTP client: spawning curl would fork the resident MLX process.

const std = @import("std");
const chat = @import("chat.zig");
const model = @import("model.zig");
const model_discovery = @import("model_discovery.zig");
const log = @import("log.zig");
const dflash = @import("dflash.zig");
const status = @import("status.zig");
const repl_tools = @import("repl_tools.zig");
const repl_input = @import("repl_input.zig");
const update = @import("update.zig");

// ── Unparsed-argument reporting ─────────────────────────────────────────

/// Why main.zig's flag loop could not consume an argument.
///
/// The loop matches every flag by EXACT name and reads its value from the
/// NEXT argv slot, and it used to end with no else branch at all — so a
/// misspelled flag, or the `--flag=value` shape it never accepted, was
/// dropped in silence. That is the worst possible outcome for a launcher: the
/// flag parses as far as the user can tell, `--help` documents it, and the
/// server boots clean while ignoring what was asked for. Rejecting loudly is
/// the whole point; the variants exist only to make the message actionable.
pub const ArgReject = enum {
    /// `--model=/path` — value welded to the flag name.
    equals_form,
    /// A flag in the LAST argv slot, so its value never arrived.
    missing_value,
    /// Not a flag we know.
    unknown,

    /// Trailing advice for the error message. Never empty.
    pub fn hint(self: ArgReject) []const u8 {
        return switch (self) {
            .equals_form => "flags take their value as a separate argument (--model <path>, not --model=<path>)",
            .missing_value => "this flag expects a value after it, or it is misspelled",
            .unknown => "see --help for the flag list",
        };
    }
};

/// Classify an argument the flag loop fell through on. `is_last` is true when
/// it occupied the final argv slot — the only way a known value-taking flag
/// can reach the else branch (every such arm is guarded on `i + 1 < args.len`).
pub fn classifyUnparsedArg(arg: []const u8, is_last: bool) ArgReject {
    const is_flag = std.mem.startsWith(u8, arg, "-");
    if (is_flag and std.mem.indexOfScalar(u8, arg, '=') != null) return .equals_form;
    if (is_flag and is_last) return .missing_value;
    return .unknown;
}

// ── Alias table ─────────────────────────────────────────────────────────

pub const Alias = struct {
    /// Short name before the ':', e.g. "qwen3.8-flash-next".
    name: []const u8,
    /// Tag after the ':'; empty = selectable only by full name:tag.
    tag: []const u8,
    repo: []const u8,
    /// Picked when the user gives the bare name with no tag.
    is_default: bool = false,
    /// Non-empty: restrict the download to this single .gguf artifact.
    gguf_file: []const u8 = "",
};

/// The published Sushi packs. The bare Qwen name picks 3bpw: 2.6bpw needs the same
/// 64 GB and scores worse on KLD; 4bpw needs 96 GB. MiMo's and GLM's one pack each
/// need 128 GB.
pub const aliases = [_]Alias{
    .{ .name = "qwen3.8-flash-next", .tag = "2bpw", .repo = "beamster/Qwen3.8-Flash-Next-Sushi-2bpw" },
    .{ .name = "qwen3.8-flash-next", .tag = "2.6bpw", .repo = "beamster/Qwen3.8-Flash-Next-Sushi-2.6bpw" },
    .{ .name = "qwen3.8-flash-next", .tag = "3bpw", .repo = "beamster/Qwen3.8-Flash-Next-Sushi-3bpw", .is_default = true },
    .{ .name = "qwen3.8-flash-next", .tag = "4bpw", .repo = "beamster/Qwen3.8-Flash-Next-Sushi-4bpw" },
    .{ .name = "mimo-v2.6-flash", .tag = "2.3bpw", .repo = "beamster/MiMo-V2.6-Flash-Sushi-2.3bpw", .is_default = true },
    .{ .name = "glm-5.3-flash", .tag = "2.4bpw", .repo = "beamster/GLM-5.3-Flash-Sushi-2.4bpw", .is_default = true },
};

pub const Resolved = struct {
    repo: []const u8,
    gguf_file: []const u8 = "",
};

/// "name:tag" → "name"; a ':' before a '/' is part of the name (a host:port).
fn stripTag(name: []const u8) []const u8 {
    if (std.mem.lastIndexOfScalar(u8, name, ':')) |i| {
        if (std.mem.indexOfScalarPos(u8, name, i, '/') == null) return name[0..i];
    }
    return name;
}

/// Short name / repo ref → HF repo id. Accepts:
///   "qwen3.8-flash-next" / "qwen3.8-flash-next:2.6bpw"  (alias table)
///   "org/repo" / "org/repo:tag"       (direct, tag stripped)
///   "hf.co/org/repo", "huggingface.co/org/repo"
/// Returns null for unknown alias-shaped names (no '/').
pub fn resolveShortName(name: []const u8) ?Resolved {
    var n = name;
    for ([_][]const u8{ "hf.co/", "huggingface.co/", "https://huggingface.co/" }) |prefix| {
        if (std.ascii.startsWithIgnoreCase(n, prefix)) {
            n = n[prefix.len..];
            break;
        }
    }
    n = stripTag(n);
    if (n.len == 0) return null;
    if (std.mem.indexOfScalar(u8, n, '/') != null) {
        // Direct org/repo reference.
        return .{ .repo = n };
    }
    // Alias lookup: "name" or "name:tag" (tag was stripped above — redo the
    // split on the ORIGINAL string so alias tags still work).
    var base = name;
    var tag: []const u8 = "";
    if (std.mem.lastIndexOfScalar(u8, name, ':')) |ci| {
        base = name[0..ci];
        tag = name[ci + 1 ..];
    }
    if (std.mem.eql(u8, tag, "latest")) tag = "";
    for (aliases) |a| {
        if (!std.ascii.eqlIgnoreCase(a.name, base)) continue;
        if (tag.len == 0) {
            if (a.is_default) return .{ .repo = a.repo, .gguf_file = a.gguf_file };
        } else if (std.ascii.eqlIgnoreCase(a.tag, tag)) {
            return .{ .repo = a.repo, .gguf_file = a.gguf_file };
        }
    }
    return null;
}

/// `~/.sushi/models/<org>/<repo>` — the single models root shared with
/// the app's DownloadManager.
pub fn modelDestPath(allocator: std.mem.Allocator, home: []const u8, repo: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}/.sushi/models/{s}", .{ home, repo });
}

pub fn modelsRootPath(allocator: std.mem.Allocator, home: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}/.sushi/models", .{home});
}

/// Inputs to "which models should this boot discover?" — see
/// `shouldDefaultModelsRoot`.
pub const RootDefaulting = struct {
    /// Invoked as `sushi serve` / `sushi run <model>` rather than flags.
    subcommand: bool,
    /// The process will serve HTTP (subcommand, or the `--serve` flag).
    serve_mode: bool,
    /// `--model <path>` (or `run <model>`) named a specific checkpoint.
    has_explicit_model: bool,
};

/// Should an unspecified `--model-dir` fall back to `~/.sushi/models`?
///
/// That path is already the single source of truth everywhere else — `pull`
/// writes there, `list` reads there, the app's DownloadManager and both
/// resolvers agree on it — so a server told to serve, but given neither a model
/// nor a directory, has exactly one sensible place to look. Without this a bare
/// `sushi --serve` discovered nothing and answered 503 to everything.
///
/// The one case that must NOT default: `--model <path> --serve`, which asked
/// for one specific model. Registering everything else on disk beside it is a
/// different server than the one requested.
pub fn shouldDefaultModelsRoot(in: RootDefaulting) bool {
    if (!in.serve_mode) return false;
    if (in.subcommand) return true; // `serve` / `run` always populate the picker
    return !in.has_explicit_model;
}

fn homeDir() []const u8 {
    return std.mem.span(std.c.getenv("HOME") orelse return "/tmp");
}

// ── HF tree listing ─────────────────────────────────────────────────────

pub const RepoFile = struct {
    path: []u8,
    size: u64,
};

pub fn freeRepoFiles(allocator: std.mem.Allocator, files: []RepoFile) void {
    for (files) |f| allocator.free(f.path);
    allocator.free(files);
}

/// Parse the HF `/api/models/<repo>/tree/main?recursive=true` JSON array.
/// LFS entries report the real artifact size under `lfs.size`.
pub fn parseTreeJson(allocator: std.mem.Allocator, json: []const u8) ![]RepoFile {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, json, .{}) catch return error.InvalidTree;
    defer parsed.deinit();
    if (parsed.value != .array) return error.InvalidTree;

    var files = std.ArrayList(RepoFile).empty;
    errdefer {
        for (files.items) |f| allocator.free(f.path);
        files.deinit(allocator);
    }
    for (parsed.value.array.items) |item| {
        if (item != .object) continue;
        const obj = item.object;
        const t = obj.get("type") orelse continue;
        if (t != .string or !std.mem.eql(u8, t.string, "file")) continue;
        const p = obj.get("path") orelse continue;
        if (p != .string) continue;
        var size: u64 = 0;
        if (obj.get("lfs")) |lfs| {
            if (lfs == .object) {
                if (lfs.object.get("size")) |s| {
                    if (s == .integer and s.integer > 0) size = @intCast(s.integer);
                }
            }
        }
        if (size == 0) {
            if (obj.get("size")) |s| {
                if (s == .integer and s.integer > 0) size = @intCast(s.integer);
            }
        }
        try files.append(allocator, .{ .path = try allocator.dupe(u8, p.string), .size = size });
    }
    return files.toOwnedSlice(allocator);
}

/// Chat-default file selection (mirrors the app's `FileSelection.chatDefault`):
/// top-level files + the `mtp/` spec-decode sidecar + GLM's DFlash2 assistant
/// folder (the one the loader auto-detects); repo housekeeping and demo assets
/// are skipped.
/// `pytorch_model.bin` / `pytorch_model-0000N-of-0000M.bin` — the HF torch
/// weights that sit beside the safetensors copy. Shared rule with the app's
/// `DownloadManager.selectNeededFiles`; keep them in sync.
pub fn isTorchShadowBin(path: []const u8) bool {
    if (!std.ascii.endsWithIgnoreCase(path, ".bin")) return false;
    const base = if (std.mem.lastIndexOfScalar(u8, path, '/')) |i| path[i + 1 ..] else path;
    return std.ascii.startsWithIgnoreCase(base, "pytorch_model") or
        std.ascii.startsWithIgnoreCase(base, "rust_model") or
        std.ascii.startsWithIgnoreCase(base, "tf_model");
}

pub fn shouldDownload(full_path: []const u8) bool {
    if (full_path.len == 0 or full_path[0] == '.') return false;
    var path = full_path;
    if (std.mem.indexOfScalar(u8, path, '/')) |slash| {
        if (std.mem.startsWith(u8, path, "mtp/")) return true;
        if (!std.mem.eql(u8, path[0..slash], dflash.SHIPPED_GLM_SUBDIR)) return false;
        path = path[slash + 1 ..];
        if (path.len == 0 or path[0] == '.' or std.mem.indexOfScalar(u8, path, '/') != null) return false;
    }
    const skip_exact = [_][]const u8{ "README.md", "LICENSE", "LICENSE.txt", "USE_POLICY.md" };
    for (skip_exact) |s| {
        if (std.ascii.eqlIgnoreCase(path, s)) return false;
    }
    // Torch/flax shadow weights are a second copy of the same model in a format
    // the server never reads — a doubled download. `.bin` itself stays allowed:
    // qwen4_exp's `ngram_table.bin` is an engine-read sidecar (mmapped at serve
    // time), and dropping it is what made app-downloaded packs fail to load.
    const skip_ext = [_][]const u8{ ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".md", ".pth", ".h5", ".msgpack", ".ckpt" };
    for (skip_ext) |ext| {
        if (path.len > ext.len and std.ascii.eqlIgnoreCase(path[path.len - ext.len ..], ext)) return false;
    }
    if (isTorchShadowBin(path)) return false;
    return true;
}

// ── Pull ────────────────────────────────────────────────────────────────

pub const Reporter = struct {
    impl: *anyopaque,
    /// One human-readable status line per call (no trailing newline).
    reportFn: *const fn (impl: *anyopaque, line: []const u8) void,

    pub fn say(self: Reporter, comptime fmt: []const u8, args: anytype) void {
        var buf: [512]u8 = undefined;
        const line = std.fmt.bufPrint(&buf, fmt, args) catch return;
        self.reportFn(self.impl, line);
    }
};

/// Appends the shared curl argv prefix; returns the owned Authorization
/// header string when HF_TOKEN is set (caller frees).
fn curlBaseArgs(list: *std.ArrayList([]const u8), allocator: std.mem.Allocator) !?[]u8 {
    try list.appendSlice(allocator, &.{ "curl", "-fL", "--retry", "3", "--retry-delay", "2" });
    if (std.c.getenv("HF_TOKEN")) |tok| {
        const header = try std.fmt.allocPrint(allocator, "Authorization: Bearer {s}", .{std.mem.span(tok)});
        try list.appendSlice(allocator, &.{ "-H", header });
        return header;
    }
    return null;
}

/// GET a small HTTPS document (the tree listing) via curl; returns stdout.
fn curlFetch(allocator: std.mem.Allocator, io: std.Io, url: []const u8) ![]u8 {
    var argv = std.ArrayList([]const u8).empty;
    defer argv.deinit(allocator);
    const header_storage = try curlBaseArgs(&argv, allocator);
    defer if (header_storage) |h| allocator.free(h);
    try argv.appendSlice(allocator, &.{ "-s", url });
    const result = std.process.run(allocator, io, .{
        .argv = argv.items,
        .stdout_limit = .limited(64 * 1024 * 1024),
    }) catch return error.FetchFailed;
    defer allocator.free(result.stderr);
    errdefer allocator.free(result.stdout);
    switch (result.term) {
        // Quiet on failure — callers report (the REPL health poll EXPECTS
        // failures while the server boots).
        .exited => |code| if (code != 0) return error.FetchFailed,
        else => return error.FetchFailed,
    }
    return result.stdout;
}

/// Download one file to `<dest_dir>/<file>` via curl (`-C -` resume onto a
/// .partial, atomic rename on success). `show_progress` inherits stderr so
/// the terminal gets curl's progress bar; the server-side pull passes false.
fn curlDownload(allocator: std.mem.Allocator, io: std.Io, url: []const u8, dest_path: []const u8, show_progress: bool) !void {
    const partial = try std.fmt.allocPrint(allocator, "{s}.partial", .{dest_path});
    defer allocator.free(partial);

    var argv = std.ArrayList([]const u8).empty;
    defer argv.deinit(allocator);
    const header_storage = try curlBaseArgs(&argv, allocator);
    defer if (header_storage) |h| allocator.free(h);
    try argv.appendSlice(allocator, &.{ "--create-dirs", "-C", "-", "-o", partial });
    if (show_progress) {
        try argv.append(allocator, "--progress-bar");
    } else {
        try argv.append(allocator, "-sS");
    }
    try argv.append(allocator, url);

    var child = std.process.spawn(io, .{
        .argv = argv.items,
        .stdin = .ignore,
        .stdout = .ignore,
        .stderr = if (show_progress) .inherit else .ignore,
    }) catch return error.DownloadFailed;
    const term = child.wait(io) catch return error.DownloadFailed;
    switch (term) {
        .exited => |code| if (code != 0) return error.DownloadFailed,
        else => return error.DownloadFailed,
    }
    std.Io.Dir.renameAbsolute(partial, dest_path, io) catch return error.DownloadFailed;
}

fn fileSizeAt(io: std.Io, dir_path: []const u8, rel: []const u8) ?u64 {
    if (dir_path.len == 0 or !std.fs.path.isAbsolute(dir_path)) return null;
    var dir = std.Io.Dir.openDirAbsolute(io, dir_path, .{}) catch return null;
    defer dir.close(io);
    const st = dir.statFile(io, rel, .{}) catch return null;
    if (st.kind != .file) return null;
    return st.size;
}

/// True when the model directory already holds a COMPLETE, loadable
/// checkpoint. "config.json exists" is NOT enough: an interrupted `pull`
/// (Ctrl-C mid-weights) leaves config.json + *.partial, and treating that
/// as present skipped the resume and fed a weightless dir to the loader
/// (live SIGSEGV — see tests/test_partial_download.sh). Complete means: no
/// .partial leftovers anywhere (top level or one subdir deep, e.g.
/// mtp/weights.safetensors.partial), plus config.json AND at least one
/// .safetensors for MLX dirs — or any .gguf, which is self-contained.
pub fn modelPresent(io: std.Io, dir_path: []const u8) bool {
    if (dir_path.len == 0 or !std.fs.path.isAbsolute(dir_path)) return false;
    var dir = std.Io.Dir.openDirAbsolute(io, dir_path, .{ .iterate = true }) catch return false;
    defer dir.close(io);
    return modelPresentInDir(io, dir);
}

fn modelPresentInDir(io: std.Io, dir: std.Io.Dir) bool {
    var has_config = false;
    var has_safetensors = false;
    var has_gguf = false;
    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        switch (entry.kind) {
            .file => {
                if (std.mem.endsWith(u8, entry.name, ".partial")) return false;
                if (std.mem.eql(u8, entry.name, "config.json")) has_config = true;
                if (std.mem.endsWith(u8, entry.name, ".safetensors")) has_safetensors = true;
                if (std.mem.endsWith(u8, entry.name, ".gguf")) has_gguf = true;
            },
            .directory => {
                // One level deep is enough for the pull layouts (mtp/ is the
                // only subdir the chat-default selection downloads into).
                var sub = dir.openDir(io, entry.name, .{ .iterate = true }) catch continue;
                defer sub.close(io);
                var sit = sub.iterate();
                while (sit.next(io) catch null) |se| {
                    if (se.kind == .file and std.mem.endsWith(u8, se.name, ".partial")) return false;
                }
            },
            else => {},
        }
    }
    if (has_gguf) return true;
    return has_config and has_safetensors;
}

/// Download `resolved.repo` into `dest_dir`. Skips files already complete
/// on disk (size match), resumes partials, reports per-file progress.
pub fn pullRepo(allocator: std.mem.Allocator, io: std.Io, resolved: Resolved, dest_dir: []const u8, reporter: Reporter, show_progress: bool) !void {
    reporter.say("pulling manifest for {s}", .{resolved.repo});
    const tree_url = try std.fmt.allocPrint(allocator, "https://huggingface.co/api/models/{s}/tree/main?recursive=true", .{resolved.repo});
    defer allocator.free(tree_url);
    const tree_json = curlFetch(allocator, io, tree_url) catch {
        reporter.say("error: could not list {s} (check the name, your network, or HF_TOKEN for gated repos)", .{resolved.repo});
        return error.PullFailed;
    };
    defer allocator.free(tree_json);
    const files = parseTreeJson(allocator, tree_json) catch {
        reporter.say("error: unexpected listing for {s}", .{resolved.repo});
        return error.PullFailed;
    };
    defer freeRepoFiles(allocator, files);

    var wanted: usize = 0;
    var total_bytes: u64 = 0;
    for (files) |f| {
        if (!wantedFile(resolved, f.path)) continue;
        wanted += 1;
        total_bytes += f.size;
    }
    if (wanted == 0) {
        reporter.say("error: {s} has no downloadable model files", .{resolved.repo});
        return error.PullFailed;
    }
    reporter.say("{d} files, {d} MB total", .{ wanted, total_bytes / (1024 * 1024) });

    var idx: usize = 0;
    for (files) |f| {
        if (!wantedFile(resolved, f.path)) continue;
        idx += 1;
        const dest_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ dest_dir, f.path });
        defer allocator.free(dest_path);
        if (f.size > 0) {
            if (fileSizeAt(io, dest_dir, f.path)) |have| {
                if (have == f.size) {
                    reporter.say("[{d}/{d}] {s} — already complete", .{ idx, wanted, f.path });
                    continue;
                }
            }
        }
        reporter.say("[{d}/{d}] pulling {s} ({d} MB)", .{ idx, wanted, f.path, f.size / (1024 * 1024) });
        const url = try std.fmt.allocPrint(allocator, "https://huggingface.co/{s}/resolve/main/{s}", .{ resolved.repo, f.path });
        defer allocator.free(url);
        curlDownload(allocator, io, url, dest_path, show_progress) catch {
            reporter.say("error: download failed for {s} (partial kept — rerun to resume)", .{f.path});
            return error.PullFailed;
        };
    }
    reporter.say("success: {s} ready", .{resolved.repo});
}

fn wantedFile(resolved: Resolved, path: []const u8) bool {
    if (resolved.gguf_file.len > 0) {
        // Single-artifact GGUF repos: just that file (plus nothing else).
        return std.mem.eql(u8, path, resolved.gguf_file);
    }
    return shouldDownload(path);
}

// ── Commands ────────────────────────────────────────────────────────────

fn stderrReport(impl: *anyopaque, line: []const u8) void {
    _ = impl;
    log.info("{s}\n", .{line});
}

var stderr_reporter_dummy: u8 = 0;
const stderr_reporter = Reporter{ .impl = &stderr_reporter_dummy, .reportFn = &stderrReport };

/// Resolve + download-if-missing; returns the local model dir (owned).
/// Exits the process with a friendly message on unknown names.
pub fn ensureModelAvailable(allocator: std.mem.Allocator, io: std.Io, name: []const u8) ![]u8 {
    // A path that exists locally is used as-is.
    if (std.fs.path.isAbsolute(name)) return allocator.dupe(u8, name);
    const resolved = resolveShortName(name) orelse {
        log.err("unknown model '{s}'\n", .{name});
        printKnownAliases(io);
        std.process.exit(1);
    };
    const dest = try modelDestPath(allocator, homeDir(), resolved.repo);
    errdefer allocator.free(dest);
    if (modelPresent(io, dest)) return dest;
    try pullRepo(allocator, io, resolved, dest, stderr_reporter, true);
    return dest;
}

pub fn cmdPull(allocator: std.mem.Allocator, io: std.Io, name: []const u8) !void {
    const dir = try ensureModelAvailable(allocator, io, name);
    defer allocator.free(dir);
    log.info("model at {s}\n", .{dir});
    log.info("run it: sushi run {s}\n", .{name});
}

fn printKnownAliases(io: std.Io) void {
    _ = io;
    log.err("known short names (or use any HuggingFace 'org/repo'):\n", .{});
    for (aliases) |a| {
        if (a.is_default) {
            log.err("  {s} (= {s}:{s}) -> {s}\n", .{ a.name, a.name, a.tag, a.repo });
        } else {
            log.err("  {s}:{s} -> {s}\n", .{ a.name, a.tag, a.repo });
        }
    }
}

/// `sushi list` — models on disk under ~/.sushi/models.
pub fn cmdList(allocator: std.mem.Allocator, io: std.Io) !void {
    const root = try modelsRootPath(allocator, homeDir());
    defer allocator.free(root);

    var out_buf: [4096]u8 = undefined;
    var stdout_w = std.Io.File.stdout().writer(io, &out_buf);
    const w = &stdout_w.interface;
    defer w.flush() catch {};

    var dir = std.Io.Dir.openDirAbsolute(io, root, .{ .iterate = true }) catch {
        try w.print("no models yet (looked in {s})\ntry: sushi pull qwen3.8-flash-next\n", .{root});
        return;
    };
    defer dir.close(io);

    try w.print("{s: <56} {s: <12} {s: >10}\n", .{ "NAME", "TYPE", "SIZE" });
    var count: usize = 0;
    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        if (!treeEntryDescends(entry.kind)) continue;
        if (entry.name.len == 0 or entry.name[0] == '.') continue;
        var sub = dir.openDir(io, entry.name, .{ .iterate = true }) catch continue;
        defer sub.close(io);
        if (isModelDir(io, &sub)) {
            try printModelRow(io, allocator, w, &sub, entry.name, root);
            count += 1;
            continue;
        }
        // org/ level: one more hop down.
        var sub_it = sub.iterate();
        while (sub_it.next(io) catch null) |sub_entry| {
            if (!treeEntryDescends(sub_entry.kind)) continue;
            var leaf = sub.openDir(io, sub_entry.name, .{ .iterate = true }) catch continue;
            defer leaf.close(io);
            if (!isModelDir(io, &leaf)) continue;
            var name_buf: [512]u8 = undefined;
            const full = std.fmt.bufPrint(&name_buf, "{s}/{s}", .{ entry.name, sub_entry.name }) catch continue;
            try printModelRow(io, allocator, w, &leaf, full, root);
            count += 1;
        }
    }
    if (count == 0) {
        try w.print("(none) — try: sushi pull qwen3.8-flash-next\n", .{});
    }
}

/// A tree-walk entry worth descending into: a real directory OR a symlink
/// (a checkpoint moved to an external drive and linked back — the H3 mirrors
/// live that way; openDir resolves the link, and model_discovery's own walk
/// already accepts both kinds).
fn treeEntryDescends(kind: std.Io.File.Kind) bool {
    return kind == .directory or kind == .sym_link;
}

fn isModelDir(io: std.Io, dir: *std.Io.Dir) bool {
    if (dir.statFile(io, "config.json", .{})) |st| {
        if (st.kind == .file) return true;
    } else |_| {}
    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        if (entry.kind != .file and entry.kind != .sym_link) continue;
        if (!std.mem.endsWith(u8, entry.name, ".gguf")) continue;
        const st = dir.statFile(io, entry.name, .{}) catch continue;
        if (st.kind == .file) return true;
    }
    return false;
}

fn printModelRow(io: std.Io, allocator: std.mem.Allocator, w: *std.Io.Writer, dir: *std.Io.Dir, name: []const u8, root: []const u8) !void {
    const bytes = dirBytesOneLevel(io, dir);
    // TYPE from the same classification serving uses (gguf → chat via the
    // embedded engines, media modalities, embed, drafter, unsupported) so
    // the list is honest about which rows `run` can actually chat with.
    const abs = std.fmt.allocPrint(allocator, "{s}/{s}", .{ root, name }) catch null;
    defer if (abs) |a| allocator.free(a);
    const kind_label: []const u8 = blk: {
        const a = abs orelse break :blk "?";
        const kind = model_discovery.classifyModelPath(io, allocator, a) orelse break :blk "?";
        break :blk kind.label();
    };
    var size_buf: [32]u8 = undefined;
    try w.print("{s: <56} {s: <12} {s: >10}\n", .{ name, kind_label, formatSize(&size_buf, bytes) });
}

/// Sum file bytes in a model dir INCLUDING one level of subdirectories —
/// media bundles keep their weights in transformer/ vae/ text_encoder/ etc.
/// (the same one-level layout assumption `modelPresent` makes). Top-level-
/// only summing showed a 7 GB FLUX bundle as "6 KB".
fn dirBytesOneLevel(io: std.Io, dir: *std.Io.Dir) u64 {
    var bytes: u64 = 0;
    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        switch (entry.kind) {
            .file, .sym_link => {
                const st = dir.statFile(io, entry.name, .{}) catch continue;
                if (st.kind == .file) bytes += @intCast(st.size);
            },
            .directory => {
                var sub = dir.openDir(io, entry.name, .{ .iterate = true }) catch continue;
                defer sub.close(io);
                var sit = sub.iterate();
                while (sit.next(io) catch null) |se| {
                    if (se.kind != .file and se.kind != .sym_link) continue;
                    const st = sub.statFile(io, se.name, .{}) catch continue;
                    if (st.kind != .file) continue;
                    bytes += @intCast(st.size);
                }
            },
            else => {},
        }
    }
    return bytes;
}

pub fn formatSize(buf: []u8, bytes: u64) []const u8 {
    const gb = 1024 * 1024 * 1024;
    const mb = 1024 * 1024;
    if (bytes >= gb) {
        const whole = bytes / gb;
        const tenth = (bytes % gb) * 10 / gb;
        return std.fmt.bufPrint(buf, "{d}.{d} GB", .{ whole, tenth }) catch "?";
    }
    if (bytes >= mb) return std.fmt.bufPrint(buf, "{d} MB", .{bytes / mb}) catch "?";
    return std.fmt.bufPrint(buf, "{d} KB", .{bytes / 1024}) catch "?";
}

/// The post-load line `sushi run` prints, from the status getters:
/// `real_bytes` is THIS process's physical footprint (Activity Monitor's
/// "Real Memory" — what the load just grew), `free_bytes` is what the system
/// would grant a new large allocation, `total_bytes` is physical RAM.
pub fn formatMemorySummary(buf: []u8, real_bytes: u64, free_bytes: u64, total_bytes: u64) ![]const u8 {
    var real_sz: [24]u8 = undefined;
    var free_sz: [24]u8 = undefined;
    var total_sz: [24]u8 = undefined;
    return std.fmt.bufPrint(buf, "[mem] real {s}, free {s}, total {s}", .{
        formatSize(&real_sz, real_bytes),
        formatSize(&free_sz, free_bytes),
        formatSize(&total_sz, total_bytes),
    });
}

// ── REPL (sushi run) ────────────────────────────────────────────────
//
// The REPL is deliberately a real HTTP client against the server's own
// /v1/chat/completions endpoint (streaming SSE) — it dogfoods the API on
// every keystroke instead of poking internal functions.

pub const Turn = struct {
    role: []const u8,
    /// Owned by the REPL history, like every field below.
    content: []const u8,
    /// Assistant turn: its OpenAI `tool_calls` array, as JSON.
    tool_calls_json: ?[]const u8 = null,
    /// Tool turn: the call it answers.
    tool_call_id: ?[]const u8 = null,
    /// data: URLs sent as image parts after the text.
    images: []const []const u8 = &.{},

    pub fn deinit(t: Turn, allocator: std.mem.Allocator) void {
        allocator.free(t.content);
        if (t.tool_calls_json) |j| allocator.free(j);
        if (t.tool_call_id) |id| allocator.free(id);
        for (t.images) |i| allocator.free(i);
        if (t.images.len > 0) allocator.free(t.images);
    }
};

/// The REPL's thinking request. `model_default` sends no thinking field.
pub const Think = union(enum) { model_default, on, effort: model.Effort };

pub const ReplOptions = struct {
    think: Think = .model_default,
    /// Client-side research tools (`repl_tools`); off unless asked for.
    tools: bool = false,
    /// `write_file` and `edit_file` inside the tools' folder; off unless asked for, and only ever
    /// offered together with `tools`.
    edit: bool = false,

    pub fn toolsJson(o: ReplOptions, vision: bool) ?[]const u8 {
        return if (o.tools) repl_tools.definitionsJson(vision, o.edit) else null;
    }
};

/// The one switch for the prompt-lookup drafts in MTP rounds and GLM DFlash2; `--pld`/`--no-pld`
/// (standalone PLD) and `--no-mtp` are separate flags and never imply it.
pub fn isNoMtpLookupFlag(arg: []const u8) bool {
    return std.mem.eql(u8, arg, "--no-mtp-lookup");
}

/// `on`/`off` for `--tool` and `/tool`.
pub fn parseToolSwitch(word: []const u8) ?bool {
    if (std.mem.eql(u8, word, "on")) return true;
    if (std.mem.eql(u8, word, "off")) return false;
    return null;
}

pub const ToolCommand = union(enum) { show, set: bool, refuse: []const u8 };

/// `/tool [on|off]`; null when the line is not that command.
pub fn parseToolCommand(line: []const u8) ?ToolCommand {
    const word = commandArg(line, "/tool") orelse return null;
    if (word.len == 0) return .show;
    return .{ .set = parseToolSwitch(word) orelse return .{ .refuse = word } };
}

/// `/edit [on|off]`: the same switch as `/tool`, for the file-writing pair.
pub fn parseEditCommand(line: []const u8) ?ToolCommand {
    const word = commandArg(line, "/edit") orelse return null;
    if (word.len == 0) return .show;
    return .{ .set = parseToolSwitch(word) orelse return .{ .refuse = word } };
}

/// `/image <path>`: the raw argument ("" when missing), null for other lines.
pub fn parseImageCommand(line: []const u8) ?[]const u8 {
    return commandArg(line, "/image");
}

/// `/cd [folder]`: the raw argument ("" when missing), null for other lines.
pub fn parseCdCommand(line: []const u8) ?[]const u8 {
    return commandArg(line, "/cd");
}

fn commandArg(line: []const u8, name: []const u8) ?[]const u8 {
    if (!std.mem.startsWith(u8, line, name)) return null;
    const rest = line[name.len..];
    if (rest.len > 0 and rest[0] != ' ') return null;
    return std.mem.trim(u8, rest, " \t");
}

/// A path as a terminal pastes it: surrounding quotes dropped, `\ ` unescaped.
pub fn unquotePath(allocator: std.mem.Allocator, arg: []const u8) ![]u8 {
    if (arg.len >= 2 and (arg[0] == '\'' or arg[0] == '"') and arg[arg.len - 1] == arg[0])
        return allocator.dupe(u8, arg[1 .. arg.len - 1]);
    var out = try std.ArrayList(u8).initCapacity(allocator, arg.len);
    errdefer out.deinit(allocator);
    var i: usize = 0;
    while (i < arg.len) : (i += 1) {
        if (arg[i] == '\\' and i + 1 < arg.len) i += 1;
        out.appendAssumeCapacity(arg[i]);
    }
    return out.toOwnedSlice(allocator);
}

pub const ThinkFlag = struct { think: Think, consumed: bool };

/// `--think [effort]`: the next argument is taken only when it is an effort word.
pub fn parseThinkFlag(next: ?[]const u8) ThinkFlag {
    if (next) |w| if (model.parseEffort(w)) |e| return .{ .think = .{ .effort = e }, .consumed = true };
    return .{ .think = .on, .consumed = false };
}

pub const ThinkCommand = union(enum) { show, set: Think, refuse: []const u8 };

/// `/think [effort]`; null when the line is not that command. An empty
/// `accepted` (the server listed none) leaves the whole vocabulary to the server.
pub fn parseThinkCommand(line: []const u8, accepted: []const model.Effort) ?ThinkCommand {
    if (!std.mem.startsWith(u8, line, "/think")) return null;
    const rest = line["/think".len..];
    if (rest.len > 0 and rest[0] != ' ') return null;
    const word = std.mem.trim(u8, rest, " \t");
    if (word.len == 0) return .show;
    const e = model.parseEffort(word) orelse return .{ .refuse = word };
    return .{ .set = .{ .effort = acceptedEffort(e, accepted) orelse return .{ .refuse = word } } };
}

/// The effort `e` selects among the server's listed words (`model.selectEffort`); null = refused.
pub fn acceptedEffort(e: model.Effort, accepted: []const model.Effort) ?model.Effort {
    if (accepted.len == 0) return e;
    return model.selectEffort(accepted, e);
}

fn writeEffortOptions(w: *std.Io.Writer, accepted: []const model.Effort) !void {
    const all = std.enums.values(model.Effort);
    const list = if (accepted.len > 0) accepted else all;
    for (list, 0..) |e, i| try w.print("{s}{s}", .{ if (i > 0) ", " else "", @tagName(e) });
}

/// Same text as the server's 400 (`server.effortRefusal`).
pub fn writeEffortRefusal(w: *std.Io.Writer, word: []const u8, model_id: []const u8, accepted: []const model.Effort) !void {
    try w.print("reasoning effort '{s}' is not supported by {s}; use one of: ", .{ word, model_id });
    try writeEffortOptions(w, accepted);
}

pub fn writeThinkSetting(w: *std.Io.Writer, think: Think, accepted: []const model.Effort) !void {
    try w.writeAll("thinking: ");
    try w.writeAll(switch (think) {
        .model_default => "model default",
        .on => "on",
        .effort => |e| @tagName(e),
    });
    try w.writeAll(" (options: ");
    try writeEffortOptions(w, accepted);
    try w.writeByte(')');
}

pub const ModelEfforts = struct {
    id: []u8,
    /// Empty when the row lists none.
    efforts: []model.Effort,
    /// The row lists the `vision` capability.
    vision: bool = false,

    pub fn deinit(self: ModelEfforts, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        allocator.free(self.efforts);
    }
};

/// The first `/v1/models` row (the default model): its id and `reasoning_efforts`.
pub fn parseModelEfforts(allocator: std.mem.Allocator, body: []const u8) !ModelEfforts {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();
    const data = if (parsed.value == .object) parsed.value.object.get("data") else null;
    if (data == null or data.? != .array or data.?.array.items.len == 0 or data.?.array.items[0] != .object) return error.ReplNoModel;
    const row = data.?.array.items[0].object;
    const id = if (row.get("id")) |v| (if (v == .string) v.string else "") else "";
    var efforts = std.ArrayList(model.Effort).empty;
    errdefer efforts.deinit(allocator);
    if (row.get("reasoning_efforts")) |list| if (list == .array) for (list.array.items) |v| {
        if (v != .string) continue;
        if (model.parseEffort(v.string)) |e| try efforts.append(allocator, e);
    };
    var vision = false;
    if (row.get("capabilities")) |caps| if (caps == .array) for (caps.array.items) |v| {
        if (v == .string and std.mem.eql(u8, v.string, "vision")) vision = true;
    };
    const efforts_owned = try efforts.toOwnedSlice(allocator);
    errdefer allocator.free(efforts_owned);
    return .{ .id = try allocator.dupe(u8, id), .efforts = efforts_owned, .vision = vision };
}

/// /v1/chat/completions request body for the REPL conversation so far;
/// `tools` is the OpenAI tools array to offer, null for none.
pub fn buildReplChatBody(allocator: std.mem.Allocator, history: []const Turn, think: Think, tools: ?[]const u8) ![]u8 {
    return buildChatBody(allocator, history, think, tools, null);
}

pub const PromptSampling = struct { max_tokens: u32 = 100, temperature: f32 = 0, top_p: f32 = 1, top_k: u32 = 0 };

fn buildChatBody(allocator: std.mem.Allocator, history: []const Turn, think: Think, tools: ?[]const u8, sampling: ?PromptSampling) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"model\":\"sushi\",\"stream\":true,\"stream_options\":{\"include_usage\":true},");
    if (sampling) |v| try out.print(allocator, "\"max_tokens\":{d},\"temperature\":{d},\"top_p\":{d},\"top_k\":{d},\"repeat_penalty\":1,\"presence_penalty\":0,", .{ v.max_tokens, v.temperature, v.top_p, v.top_k });
    switch (think) {
        .model_default => {},
        .on => try out.appendSlice(allocator, "\"enable_thinking\":true,"),
        .effort => |e| try out.print(allocator, "\"reasoning_effort\":\"{s}\",", .{@tagName(e)}),
    }
    if (tools) |t| try out.print(allocator, "\"tools\":{s},", .{t});
    try out.appendSlice(allocator, "\"messages\":[");
    for (history, 0..) |turn, i| {
        if (i > 0) try out.append(allocator, ',');
        try out.appendSlice(allocator, "{\"role\":");
        try chat.appendJsonString(allocator, &out, turn.role);
        try out.appendSlice(allocator, ",\"content\":");
        if (turn.images.len == 0) {
            try chat.appendJsonString(allocator, &out, turn.content);
        } else {
            try out.appendSlice(allocator, "[{\"type\":\"text\",\"text\":");
            try chat.appendJsonString(allocator, &out, turn.content);
            try out.append(allocator, '}');
            for (turn.images) |url| {
                try out.appendSlice(allocator, ",{\"type\":\"image_url\",\"image_url\":{\"url\":");
                try chat.appendJsonString(allocator, &out, url);
                try out.appendSlice(allocator, "}}");
            }
            try out.append(allocator, ']');
        }
        if (turn.tool_calls_json) |calls| try out.print(allocator, ",\"tool_calls\":{s}", .{calls});
        if (turn.tool_call_id) |id| {
            try out.appendSlice(allocator, ",\"tool_call_id\":");
            try chat.appendJsonString(allocator, &out, id);
        }
        try out.append(allocator, '}');
    }
    try out.appendSlice(allocator, "]}");
    return out.toOwnedSlice(allocator);
}

/// The server's own `timings` for a turn; a client cannot time our stream.
pub const ReplStats = struct {
    /// Whole prompt, cached prefix included.
    prompt_n: u64 = 0,
    cached_n: u64 = 0,
    /// Over the tokens actually prefilled (prompt_n - cached_n).
    prompt_per_second: f64 = 0,
    eval_count: u64 = 0,
    eval_duration_ns: u64 = 0,
};

pub const ToolCall = struct {
    /// Position in the assistant turn; stream deltas with the same index extend one call.
    index: usize = 0,
    id: []u8,
    name: []u8,
    arguments: []u8,

    pub fn deinit(c: ToolCall, allocator: std.mem.Allocator) void {
        allocator.free(c.id);
        allocator.free(c.name);
        allocator.free(c.arguments);
    }
};

fn freeToolCalls(allocator: std.mem.Allocator, calls: []ToolCall) void {
    for (calls) |c| c.deinit(allocator);
    allocator.free(calls);
}

pub const ReplDelta = struct {
    /// Owned by caller.
    content: []u8,
    /// Owned by caller; the thought, streamed before the answer.
    reasoning: ?[]u8 = null,
    done: bool,
    stats: ReplStats = .{},
    err: ?[]u8 = null,
    /// Owned by caller.
    tool_calls: []ToolCall = &.{},

    pub fn deinit(d: ReplDelta, allocator: std.mem.Allocator) void {
        allocator.free(d.content);
        if (d.reasoning) |r| allocator.free(r);
        if (d.err) |e| allocator.free(e);
        freeToolCalls(allocator, d.tool_calls);
    }
};

/// One model reply: the streamed text and the tool calls it ended with.
pub const Reply = struct {
    content: []u8,
    tool_calls: []ToolCall,

    pub fn deinit(r: Reply, allocator: std.mem.Allocator) void {
        allocator.free(r.content);
        freeToolCalls(allocator, r.tool_calls);
    }
};

fn dupeJsonString(allocator: std.mem.Allocator, v: ?std.json.Value) ![]u8 {
    return allocator.dupe(u8, if (v) |s| (if (s == .string) s.string else "") else "");
}

fn parseToolCallDeltas(allocator: std.mem.Allocator, list: std.json.Value) ![]ToolCall {
    if (list != .array) return &.{};
    var calls = std.ArrayList(ToolCall).empty;
    errdefer {
        for (calls.items) |c| c.deinit(allocator);
        calls.deinit(allocator);
    }
    for (list.array.items, 0..) |item, i| {
        if (item != .object) continue;
        const f = item.object.get("function");
        const fo: ?std.json.ObjectMap = if (f) |v| (if (v == .object) v.object else null) else null;
        const id = try dupeJsonString(allocator, item.object.get("id"));
        errdefer allocator.free(id);
        const name = try dupeJsonString(allocator, if (fo) |o| o.get("name") else null);
        errdefer allocator.free(name);
        const args = try dupeJsonString(allocator, if (fo) |o| o.get("arguments") else null);
        errdefer allocator.free(args);
        const index = jsonCount(item.object.get("index"));
        try calls.append(allocator, .{ .index = if (item.object.get("index") != null) index else i, .id = id, .name = name, .arguments = args });
    }
    return calls.toOwnedSlice(allocator);
}

/// `[prefill P tok (C cached), R tok/s | N tokens, D tok/s]`; either half drops
/// out when the server did not report it, "" when neither was reported.
pub fn formatReplStats(buf: []u8, s: ReplStats) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    const has_prefill = s.prompt_n > 0;
    const has_decode = s.eval_count > 0 and s.eval_duration_ns > 0;
    if (!has_prefill and !has_decode) return "";
    w.writeByte('[') catch return "";
    if (has_prefill) {
        w.print("prefill {d} tok", .{s.prompt_n -| s.cached_n}) catch return "";
        if (s.cached_n > 0) w.print(" ({d} cached)", .{s.cached_n}) catch return "";
        if (s.prompt_per_second > 0) w.print(", {d:.1} tok/s", .{s.prompt_per_second}) catch return "";
        if (has_decode) w.writeAll(" | ") catch return "";
    }
    if (has_decode) {
        const tok_s = @as(f64, @floatFromInt(s.eval_count)) * 1e9 / @as(f64, @floatFromInt(s.eval_duration_ns));
        w.print("{d} tokens, {d:.1} tok/s", .{ s.eval_count, tok_s }) catch return "";
    }
    w.writeByte(']') catch return "";
    return w.buffered();
}

/// One SSE line from /v1/chat/completions → the piece the REPL prints.
/// Non-event lines (blank, `:` comments) return null.
pub fn parseReplLine(allocator: std.mem.Allocator, line: []const u8) ?ReplDelta {
    const trimmed = std.mem.trimEnd(u8, line, "\r");
    if (!std.mem.startsWith(u8, trimmed, "data:")) return null;
    const payload = std.mem.trim(u8, trimmed["data:".len..], " ");
    if (std.mem.eql(u8, payload, "[DONE]")) return .{ .content = allocator.dupe(u8, "") catch return null, .done = true };
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, payload, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const root = parsed.value.object;
    if (root.get("error")) |e| {
        const msg = switch (e) {
            .string => |str| str,
            .object => |o| if (o.get("message")) |m| (if (m == .string) m.string else "error") else "error",
            else => "error",
        };
        return .{
            .content = allocator.dupe(u8, "") catch return null,
            .done = true,
            .err = allocator.dupe(u8, msg) catch return null,
        };
    }
    var content: []const u8 = "";
    var reasoning: ?[]const u8 = null;
    var tool_calls: []ToolCall = &.{};
    if (root.get("choices")) |choices| {
        if (choices == .array and choices.array.items.len > 0 and choices.array.items[0] == .object) {
            if (choices.array.items[0].object.get("delta")) |d| {
                if (d == .object) if (d.object.get("content")) |c| {
                    if (c == .string) content = c.string;
                };
                if (d == .object) if (d.object.get("reasoning_content")) |c| {
                    if (c == .string and c.string.len > 0) reasoning = c.string;
                };
                if (d == .object) if (d.object.get("tool_calls")) |list| {
                    tool_calls = parseToolCallDeltas(allocator, list) catch return null;
                };
            }
        }
    }
    var stats: ReplStats = .{};
    if (root.get("timings")) |t| if (t == .object) {
        stats.prompt_n = jsonCount(t.object.get("prompt_n"));
        stats.cached_n = jsonCount(t.object.get("cached_n"));
        stats.prompt_per_second = jsonNumber(t.object.get("prompt_per_second"));
        stats.eval_count = jsonCount(t.object.get("predicted_n"));
        const ms = jsonNumber(t.object.get("predicted_ms"));
        if (ms > 0) stats.eval_duration_ns = @intFromFloat(ms * 1e6);
    };
    const reasoning_owned = if (reasoning) |r| allocator.dupe(u8, r) catch {
        freeToolCalls(allocator, tool_calls);
        return null;
    } else null;
    return .{
        .content = allocator.dupe(u8, content) catch {
            if (reasoning_owned) |r| allocator.free(r);
            freeToolCalls(allocator, tool_calls);
            return null;
        },
        .reasoning = reasoning_owned,
        .done = false,
        .stats = stats,
        .tool_calls = tool_calls,
    };
}

fn jsonCount(v: ?std.json.Value) u64 {
    const n = v orelse return 0;
    return if (n == .integer and n.integer > 0) @intCast(n.integer) else 0;
}

fn jsonNumber(v: ?std.json.Value) f64 {
    const n = v orelse return 0;
    return switch (n) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => 0,
    };
}

pub const max_tool_rounds = 8;

pub const tool_cap_nudge = "You have used every tool round for this question. Answer now from the results above, without calling tools.";

/// Where a tool turn's requests go and its tool calls run; stubbed in tests.
pub const TurnDriver = struct {
    ptr: *anyopaque,
    complete: *const fn (ptr: *anyopaque, body: []const u8) anyerror!Reply,
    runTool: *const fn (ptr: *anyopaque, name: []const u8, args_json: []const u8) anyerror!repl_tools.Output,
};

/// Answers the user turn at the end of `history`: requests, runs the tool
/// calls, appends their results, and repeats until the model answers without
/// one. `tools` null offers none.
pub fn runToolTurn(allocator: std.mem.Allocator, history: *std.ArrayList(Turn), think: Think, tools: ?[]const u8, driver: TurnDriver) !void {
    var round: usize = 0;
    while (true) : (round += 1) {
        const offered = if (round < max_tool_rounds) tools else null;
        if (tools != null and round == max_tool_rounds)
            try history.append(allocator, .{ .role = "user", .content = try allocator.dupe(u8, tool_cap_nudge) });
        const body = try buildReplChatBody(allocator, history.items, think, offered);
        defer allocator.free(body);
        const reply = try driver.complete(driver.ptr, body);
        if (offered == null or reply.tool_calls.len == 0) {
            freeToolCalls(allocator, reply.tool_calls);
            errdefer allocator.free(reply.content);
            try history.append(allocator, .{ .role = "assistant", .content = reply.content });
            return;
        }
        defer freeToolCalls(allocator, reply.tool_calls);
        {
            errdefer allocator.free(reply.content);
            const calls_json = try toolCallsJson(allocator, reply.tool_calls);
            errdefer allocator.free(calls_json);
            try history.append(allocator, .{ .role = "assistant", .content = reply.content, .tool_calls_json = calls_json });
        }

        var images = std.ArrayList([]const u8).empty;
        defer {
            for (images.items) |i| allocator.free(i);
            images.deinit(allocator);
        }
        for (reply.tool_calls) |call| {
            const out = try driver.runTool(driver.ptr, call.name, call.arguments);
            if (out.image) |img| {
                images.append(allocator, img) catch |err| {
                    out.deinit(allocator);
                    return err;
                };
            }
            errdefer allocator.free(out.text);
            const id = try allocator.dupe(u8, call.id);
            errdefer allocator.free(id);
            try history.append(allocator, .{ .role = "tool", .content = out.text, .tool_call_id = id });
        }
        // The server shows a model only the images of user turns.
        if (images.items.len > 0) {
            const text = try std.fmt.allocPrint(allocator, "(the image{s} from the tool results above)", .{if (images.items.len > 1) "s" else ""});
            errdefer allocator.free(text);
            const owned = try images.toOwnedSlice(allocator);
            errdefer {
                for (owned) |i| allocator.free(i);
                allocator.free(owned);
            }
            try history.append(allocator, .{ .role = "user", .content = text, .images = owned });
        }
    }
}

/// An assistant turn's OpenAI `tool_calls` array.
fn toolCallsJson(allocator: std.mem.Allocator, calls: []const ToolCall) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.append(allocator, '[');
    for (calls, 0..) |c, i| {
        if (i > 0) try out.append(allocator, ',');
        try out.appendSlice(allocator, "{\"id\":");
        try chat.appendJsonString(allocator, &out, c.id);
        try out.appendSlice(allocator, ",\"type\":\"function\",\"function\":{\"name\":");
        try chat.appendJsonString(allocator, &out, c.name);
        try out.appendSlice(allocator, ",\"arguments\":");
        try chat.appendJsonString(allocator, &out, c.arguments);
        try out.appendSlice(allocator, "}}");
    }
    try out.append(allocator, ']');
    return out.toOwnedSlice(allocator);
}

/// Interactive loop on the calling thread. Waits for the server to answer
/// /health, then reads prompts from stdin and streams /v1/chat/completions.
/// Returns when the user exits (/bye or EOF); caller shuts the server down.
pub fn runRepl(allocator: std.mem.Allocator, io: std.Io, port: u16, launch: ReplOptions) !void {
    var opts = launch;
    const health_url = try std.fmt.allocPrint(allocator, "http://127.0.0.1:{d}/health", .{port});
    defer allocator.free(health_url);
    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    // Big checkpoints take a while to fault in; poll patiently.
    var waited_ms: u64 = 0;
    var ready = false;
    while (waited_ms < 15 * 60 * 1000) {
        if (client.fetch(.{ .location = .{ .url = health_url }, .keep_alive = false })) |response| {
            if (response.status == .ok) {
                ready = true;
                break;
            }
        } else |_| {}
        std.Io.sleep(io, .fromMilliseconds(500), .real) catch {};
        waited_ms += 500;
    }
    if (!ready) return error.ReplServerNotReady;

    const chat_url = try std.fmt.allocPrint(allocator, "http://127.0.0.1:{d}/v1/chat/completions", .{port});
    defer allocator.free(chat_url);

    var out_buf: [4096]u8 = undefined;
    var stdout_w = std.Io.File.stdout().writer(io, &out_buf);
    const w = &stdout_w.interface;

    const models_url = try std.fmt.allocPrint(allocator, "http://127.0.0.1:{d}/v1/models", .{port});
    defer allocator.free(models_url);
    var models_body: std.Io.Writer.Allocating = .init(allocator);
    defer models_body.deinit();
    const models_info: ?ModelEfforts = if (client.fetch(.{ .location = .{ .url = models_url }, .keep_alive = false, .response_writer = &models_body.writer })) |_|
        parseModelEfforts(allocator, models_body.written()) catch null
    else |_|
        null;
    defer if (models_info) |m| m.deinit(allocator);
    const accepted: []const model.Effort = if (models_info) |m| m.efforts else &.{};
    var think = opts.think;
    if (think == .effort) think = .{ .effort = acceptedEffort(think.effort, accepted) orelse {
        try writeEffortRefusal(w, @tagName(think.effort), if (models_info) |m| m.id else "this model", accepted);
        try w.writeAll("\n");
        try w.flush();
        return error.ReplThinkUnsupported;
    } };
    // The initial load finishes BEFORE the listener binds, so an answering
    // /health is the post-load moment. `run` quieted the log to warn on a TTY,
    // so this one line is printed here rather than logged.
    var mem_buf: [160]u8 = undefined;
    if (formatMemorySummary(
        &mem_buf,
        @as(u64, status.getAppMemFootprintMb()) * 1024 * 1024,
        status.getAvailableMemBytes(),
        status.getTotalMemBytes(),
    )) |line| {
        try w.print("{s}\n", .{line});
    } else |_| {}
    const vision = if (models_info) |m| m.vision else false;

    var input: repl_input.Input = .{ .tty = std.Io.File.stdin().isTty(io) catch false };
    defer input.deinit(allocator);

    // File tools start confined to the folder `sushi run` started in; `/cd` moves them.
    var driver: ReplDriver = .{
        .allocator = allocator,
        .io = io,
        .url = chat_url,
        .w = w,
        .input = &input,
        .tools = .{ .allocator = allocator, .io = io, .root = try std.Io.Dir.cwd().realPathFileAlloc(io, ".", allocator), .vision = vision, .writable = opts.tools and opts.edit },
    };
    defer allocator.free(driver.tools.root);
    var state_buf: [512]u8 = undefined;
    try writeReadyBanner(w, vision, port, formatPromptStatus(&state_buf, driver.tools.root, homeDir(), opts.tools));
    var version_buf: [64]u8 = undefined;
    try writeUpdateNotice(w, update.availableVersion(&version_buf), update.homebrewInstall());
    try w.flush();

    var history = std.ArrayList(Turn).empty;
    defer {
        for (history.items) |t| t.deinit(allocator);
        history.deinit(allocator);
    }
    var pending_images = std.ArrayList([]const u8).empty;
    defer {
        for (pending_images.items) |i| allocator.free(i);
        pending_images.deinit(allocator);
    }

    while (true) {
        try writePrompt(w, formatPromptStatus(&state_buf, driver.tools.root, homeDir(), opts.tools));
        try w.flush();
        const line = input.readLine(allocator, w) catch break orelse break;
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len == 0) continue;
        if (std.mem.eql(u8, trimmed, "/bye") or std.mem.eql(u8, trimmed, "/exit") or std.mem.eql(u8, trimmed, "/quit")) break;
        if (parseThinkCommand(trimmed, accepted)) |cmd| {
            switch (cmd) {
                .show => try writeThinkSetting(w, think, accepted),
                .set => |t| {
                    think = t;
                    try writeThinkSetting(w, think, accepted);
                },
                .refuse => |word| try writeEffortRefusal(w, word, if (models_info) |m| m.id else "this model", accepted),
            }
            try w.writeAll("\n");
            continue;
        }
        if (parseToolCommand(trimmed)) |cmd| {
            switch (cmd) {
                .show => {},
                .set => |on| {
                    opts.tools = on;
                    driver.tools.writable = on and opts.edit;
                },
                .refuse => |word| try w.print("/tool takes on or off, not '{s}'\n", .{word}),
            }
            try w.print("tools: {s} ({s}; files under {s}, /cd <folder> moves them)\n", .{ if (opts.tools) "on" else "off", repl_tools.toolNames(vision, opts.tools and opts.edit), driver.tools.root });
            continue;
        }
        if (parseEditCommand(trimmed)) |cmd| {
            switch (cmd) {
                .show => {},
                .set => |on| {
                    opts.edit = on;
                    driver.tools.writable = opts.tools and on;
                },
                .refuse => |word| try w.print("/edit takes on or off, not '{s}'\n", .{word}),
            }
            try w.print("edit: {s} (write_file, edit_file under {s}{s})\n", .{
                if (opts.edit) "on" else "off",
                driver.tools.root,
                if (opts.edit and !opts.tools) "; the file tools are off, turn them on with /tool on" else "",
            });
            continue;
        }
        if (parseCdCommand(trimmed)) |arg| {
            try changeToolFolder(allocator, io, w, &driver.tools, arg);
            continue;
        }
        if (parseImageCommand(trimmed)) |arg| {
            try attachImage(allocator, w, driver.tools, arg, &pending_images);
            continue;
        }
        if (std.mem.eql(u8, trimmed, "/update")) {
            if (try replUpdate(allocator, io, w)) break;
            continue;
        }

        const mark = history.items.len;
        const images = try pending_images.toOwnedSlice(allocator);
        try history.append(allocator, .{ .role = "user", .content = try allocator.dupe(u8, trimmed), .images = images });
        driver.requests = 0;
        runToolTurn(allocator, &history, think, opts.toolsJson(vision), driver.turnDriver()) catch |err| {
            try w.print("\n[error: {s}]\n", .{@errorName(err)});
            try w.flush();
            while (history.items.len > mark) history.pop().?.deinit(allocator);
            continue;
        };
        try w.writeAll("\n");
        try w.flush();
    }
}

/// The lines `sushi run` prints once the model answers; `state` is `formatPromptStatus`.
pub fn writeReadyBanner(w: *std.Io.Writer, vision: bool, port: u16, state: []const u8) !void {
    try w.writeAll("\n>>> chat is live — /bye to exit, /tool on for web search and file tools, /edit on to let the model write files there");
    try w.writeAll(if (vision) ", /image <path> to show an image\n" else "\n");
    try w.print(">>> {s} (shown before each prompt); /cd <folder> moves the folder the file tools{s} read\n", .{ state, if (vision) " and relative /image paths" else "" });
    try w.print(">>> chat in your browser: http://127.0.0.1:{d}/\n", .{port});
}

/// The REPL's line for a newer release found by the daily check; nothing without one.
pub fn writeUpdateNotice(w: *std.Io.Writer, latest: ?[]const u8, brew: bool) !void {
    const v = latest orelse return;
    if (brew) return w.print(">>> sushi {s} is available: run {s}\n", .{ v, update.brew_upgrade });
    try w.print(">>> sushi {s} is available: /update installs it and restarts this chat on the same model\n", .{v});
}

/// `/update`: a fresh check; with a newer release the REPL ends, and main replaces the process with the updater,
/// which restarts `sushi run` on the same model. True when the REPL should end.
fn replUpdate(allocator: std.mem.Allocator, io: std.Io, w: *std.Io.Writer) !bool {
    var arena_state: std.heap.ArenaAllocator = .init(allocator);
    defer arena_state.deinit();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    var why_buf: [512]u8 = undefined;
    if (update.selfInstallRefusal(io, &path_buf, &why_buf)) |why| {
        try w.print("update: {s}\n", .{why});
        return false;
    }
    const found = update.checkNow(arena_state.allocator(), io, false) catch {
        try w.print("update: could not read the release list\n", .{});
        return false;
    };
    const r = found orelse {
        try w.print("update: no release carries {s}\n", .{update.asset});
        return false;
    };
    if (!update.isNewer(r.version, update.version)) {
        try w.print("sushi {s} is up to date\n", .{update.version});
        return false;
    }
    try w.print("updating to sushi {s}; this chat restarts on the same model\n", .{r.version});
    try w.flush();
    update.relaunch_requested.store(true, .release);
    return true;
}

const max_status_path = 32;

/// The prompt's status: the tools' folder (`~` for `home`, `…` and the tail past
/// `max_status_path` characters) and whether the tools are on.
pub fn formatPromptStatus(buf: []u8, root: []const u8, home: []const u8, tools_on: bool) []const u8 {
    const under_home = home.len > 1 and std.mem.startsWith(u8, root, home) and (root.len == home.len or root[home.len] == '/');
    var lead: []const u8 = if (under_home) "~" else "";
    var path = if (under_home) root[home.len..] else root;
    if (@min(lead.len, 1) + path.len > max_status_path) {
        const from = path.len - (max_status_path - 1);
        const cut = std.mem.indexOfScalarPos(u8, path, from, '/') orelse std.mem.lastIndexOfScalar(u8, path, '/') orelse 0;
        lead = "…";
        path = path[cut..];
    }
    return std.fmt.bufPrint(buf, "{s}{s} · tools {s}", .{ lead, path, if (tools_on) "on" else "off" }) catch root;
}

/// The REPL prompt: the dim status, then `>>> `.
pub fn writePrompt(w: *std.Io.Writer, state: []const u8) !void {
    try w.print("\x1b[2m{s}\x1b[0m >>> ", .{state});
}

/// `/cd [folder]`: shows the file tools' folder, or moves them to another one.
fn changeToolFolder(allocator: std.mem.Allocator, io: std.Io, w: *std.Io.Writer, tools: *repl_tools.Context, arg: []const u8) !void {
    if (arg.len > 0) {
        const path = try unquotePath(allocator, arg);
        defer allocator.free(path);
        switch (try repl_tools.changeRoot(allocator, io, tools.root, homeDir(), path)) {
            .refused => |msg| return w.print("cannot /cd to {s}: {s}\n", .{ path, msg }),
            .ok => |root| {
                allocator.free(tools.root);
                tools.root = root;
            },
        }
    }
    try w.print("file tools read under {s}\n", .{tools.root});
}

/// `/image <path>`: a relative path reads in the tools' folder (`repl_tools.loadUserImage`).
fn attachImage(allocator: std.mem.Allocator, w: *std.Io.Writer, tools: repl_tools.Context, arg: []const u8, pending: *std.ArrayList([]const u8)) !void {
    if (arg.len == 0) return w.writeAll("usage: /image <path>\n");
    if (!tools.vision) return w.writeAll("this model cannot see images\n");
    const path = try unquotePath(allocator, arg);
    defer allocator.free(path);
    const url = switch (try repl_tools.loadUserImage(allocator, tools.io, tools.root, homeDir(), path)) {
        .ok => |u| u,
        .refused => |msg| return w.print("cannot attach {s}: {s}\n", .{ path, msg }),
    };
    errdefer allocator.free(url);
    try pending.append(allocator, url);
    try w.print("attached {s}; it goes with your next message\n", .{path});
}

/// The live `TurnDriver`: streams from the in-process server and prints one
/// dim line per tool call.
const ReplDriver = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    url: []const u8,
    w: *std.Io.Writer,
    input: *repl_input.Input,
    tools: repl_tools.Context,
    requests: usize = 0,
    /// Set once a file tool has run: from then on a `fetch_url` that can carry data out needs the user's yes.
    file_content_seen: bool = false,

    fn complete(ptr: *anyopaque, body: []const u8) anyerror!Reply {
        const d: *ReplDriver = @ptrCast(@alignCast(ptr));
        if (d.requests > 0) try d.w.writeAll("\n");
        d.requests += 1;
        return streamOneTurn(d.allocator, d.io, d.url, body, d.w);
    }

    fn runTool(ptr: *anyopaque, name: []const u8, args_json: []const u8) anyerror!repl_tools.Output {
        const d: *ReplDriver = @ptrCast(@alignCast(ptr));
        const trace = try repl_tools.traceLine(d.allocator, name, args_json);
        defer d.allocator.free(trace);
        try d.w.print("\n\x1b[2m  {s}\x1b[0m", .{trace});
        try d.w.flush();
        if (try repl_tools.fetchNeedingApproval(d.allocator, name, args_json, d.file_content_seen)) |url| {
            defer d.allocator.free(url);
            const question = try std.fmt.allocPrint(d.allocator, "\nThe model wants to fetch {s}\nThis URL can carry what it read from your files to that site. Allow?", .{url});
            defer d.allocator.free(question);
            if (!try d.input.confirm(d.allocator, d.w, question))
                return .{ .text = try d.allocator.dupe(u8, "refused: the user did not allow this fetch because the URL could send file contents out; use a URL without a query string and with a short path") };
        }
        if (repl_tools.isFileTool(name)) d.file_content_seen = true;
        return repl_tools.run(d.tools, name, args_json);
    }

    fn turnDriver(d: *ReplDriver) TurnDriver {
        return .{ .ptr = d, .complete = complete, .runTool = runTool };
    }
};

/// POST the body, stream SSE, print content deltas as they arrive.
/// Returns the full assistant reply and its tool calls (owned).
fn streamOneTurn(allocator: std.mem.Allocator, io: std.Io, url: []const u8, body: []const u8, w: *std.Io.Writer) !Reply {
    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    var req = try client.request(.POST, try std.Uri.parse(url), .{
        .keep_alive = false,
        .redirect_behavior = .unhandled,
        .headers = .{
            .content_type = .{ .override = "application/json" },
            .accept_encoding = .{ .override = "identity" },
        },
    });
    defer req.deinit();
    req.transfer_encoding = .{ .content_length = body.len };
    var request_body = try req.sendBodyUnflushed(&.{});
    try request_body.writer.writeAll(body);
    try request_body.end();
    try req.connection.?.flush();
    var head_buffer: [8192]u8 = undefined;
    var response = try req.receiveHead(&head_buffer);
    var out_buf: [64 * 1024]u8 = undefined;
    const r = response.reader(&out_buf);
    if (response.head.status != .ok) {
        const detail = try r.allocRemaining(allocator, .limited(64 * 1024));
        defer allocator.free(detail);
        try w.print("[server error HTTP {d}: ", .{@backingInt(response.head.status)});
        var detail_filter: repl_tools.TermFilter = .{};
        try detail_filter.write(w, detail);
        try w.writeAll("]\n");
        try w.flush();
        return error.ReplHttpStatus;
    }

    var full = std.ArrayList(u8).empty;
    errdefer full.deinit(allocator);
    var calls = std.ArrayList(ToolCall).empty;
    errdefer {
        for (calls.items) |c| c.deinit(allocator);
        calls.deinit(allocator);
    }
    var stats: ReplStats = .{};
    // The thought prints dim so a thinking turn never looks frozen.
    var in_thought = false;
    defer if (in_thought) w.writeAll("\x1b[0m") catch {};
    // The model's text reaches the terminal only through these: a page it read can steer what it emits.
    var thought_filter: repl_tools.TermFilter = .{};
    var answer_filter: repl_tools.TermFilter = .{};

    while (true) {
        const line = r.takeDelimiter('\n') catch break orelse break;
        if (line.len == 0) continue;
        const delta = parseReplLine(allocator, line) orelse continue;
        defer delta.deinit(allocator);
        if (delta.err) |e| {
            var err_filter: repl_tools.TermFilter = .{};
            try w.writeAll("[server error: ");
            try err_filter.write(w, e);
            try w.writeAll("]");
            try w.flush();
            break;
        }
        for (delta.tool_calls) |tc| try mergeToolCallDelta(allocator, &calls, tc);
        if (delta.reasoning) |t| {
            if (!in_thought) try w.writeAll("\x1b[2m");
            in_thought = true;
            try thought_filter.write(w, t);
            try w.flush();
        }
        if (delta.content.len > 0) {
            if (in_thought) try w.writeAll("\x1b[0m\n");
            in_thought = false;
            try answer_filter.write(w, delta.content);
            try w.flush();
            try full.appendSlice(allocator, delta.content);
        }
        if (delta.stats.eval_count > 0 or delta.stats.prompt_n > 0) stats = delta.stats;
        if (delta.done) {
            var stats_buf: [160]u8 = undefined;
            const stats_line = formatReplStats(&stats_buf, stats);
            if (stats_line.len > 0) try w.print("\n{s}", .{stats_line});
            break;
        }
    }
    const content = try full.toOwnedSlice(allocator);
    errdefer allocator.free(content);
    return .{ .content = content, .tool_calls = try calls.toOwnedSlice(allocator) };
}

/// Folds one streamed tool-call delta into the calls so far, by index.
fn mergeToolCallDelta(allocator: std.mem.Allocator, calls: *std.ArrayList(ToolCall), tc: ToolCall) !void {
    for (calls.items) |*c| if (c.index == tc.index) {
        const args = try std.mem.concat(allocator, u8, &.{ c.arguments, tc.arguments });
        allocator.free(c.arguments);
        c.arguments = args;
        if (c.id.len == 0 and tc.id.len > 0) {
            allocator.free(c.id);
            c.id = try allocator.dupe(u8, tc.id);
        }
        if (c.name.len == 0 and tc.name.len > 0) {
            allocator.free(c.name);
            c.name = try allocator.dupe(u8, tc.name);
        }
        return;
    };
    const id = try allocator.dupe(u8, tc.id);
    errdefer allocator.free(id);
    const name = try allocator.dupe(u8, tc.name);
    errdefer allocator.free(name);
    const args = try allocator.dupe(u8, tc.arguments);
    errdefer allocator.free(args);
    try calls.append(allocator, .{ .index = tc.index, .id = id, .name = name, .arguments = args });
}

// ── Tests ───────────────────────────────────────────────────────────────

const testing = std.testing;

test "cli: --no-mtp-lookup is its own flag, not --no-pld or --no-mtp" {
    try testing.expect(isNoMtpLookupFlag("--no-mtp-lookup"));
    for ([_][]const u8{ "--no-pld", "--pld", "--no-mtp", "--mtp", "--no-mtp-lookups", "--mtp-lookup" }) |other| {
        try testing.expect(!isNoMtpLookupFlag(other));
    }
}

test "cli: resolveShortName aliases, tags, org/repo, hf.co, unknown" {
    // Bare alias picks the family default.
    try testing.expectEqualStrings("beamster/Qwen3.8-Flash-Next-Sushi-3bpw", resolveShortName("qwen3.8-flash-next").?.repo);
    // Tagged alias, case-insensitive.
    try testing.expectEqualStrings("beamster/Qwen3.8-Flash-Next-Sushi-2.6bpw", resolveShortName("qwen3.8-flash-next:2.6bpw").?.repo);
    try testing.expectEqualStrings("beamster/Qwen3.8-Flash-Next-Sushi-4bpw", resolveShortName("QWEN3.8-FLASH-NEXT:4BPW").?.repo);
    // :latest behaves like bare.
    try testing.expectEqualStrings("beamster/Qwen3.8-Flash-Next-Sushi-3bpw", resolveShortName("qwen3.8-flash-next:latest").?.repo);
    // MiMo's one published pack, bare or tagged.
    try testing.expectEqualStrings("beamster/MiMo-V2.6-Flash-Sushi-2.3bpw", resolveShortName("mimo-v2.6-flash").?.repo);
    try testing.expectEqualStrings("beamster/MiMo-V2.6-Flash-Sushi-2.3bpw", resolveShortName("MiMo-V2.6-Flash:2.3bpw").?.repo);
    try testing.expectEqualStrings("beamster/Qwen3.8-Flash-Next-Sushi-2bpw", resolveShortName("qwen3.8-flash-next:2bpw").?.repo);
    try testing.expectEqualStrings("beamster/GLM-5.3-Flash-Sushi-2.4bpw", resolveShortName("glm-5.3-flash").?.repo);
    try testing.expectEqualStrings("beamster/GLM-5.3-Flash-Sushi-2.4bpw", resolveShortName("GLM-5.3-Flash:2.4bpw").?.repo);
    // Direct org/repo passthrough, tag stripped, hf.co prefixes stripped.
    try testing.expectEqualStrings("org/repo", resolveShortName("org/repo").?.repo);
    try testing.expectEqualStrings("org/repo", resolveShortName("org/repo:latest").?.repo);
    try testing.expectEqualStrings("org/repo", resolveShortName("hf.co/org/repo").?.repo);
    try testing.expectEqualStrings("org/repo", resolveShortName("https://huggingface.co/org/repo").?.repo);
    // Unknown alias-shaped name → null; upstream families are not served.
    try testing.expect(resolveShortName("doesnotexist") == null);
    try testing.expect(resolveShortName("qwen3.8-flash-next:nosuchtag") == null);
    try testing.expect(resolveShortName("gemma4") == null);
}

test "cli: every short name is a published Sushi pack" {
    for (aliases) |a| try testing.expect(std.mem.startsWith(u8, a.repo, "beamster/Qwen3.8-Flash-Next-Sushi-") or
        std.mem.startsWith(u8, a.repo, "beamster/MiMo-V2.6-Flash-Sushi-") or
        std.mem.startsWith(u8, a.repo, "beamster/GLM-5.3-Flash-Sushi-"));
}

test "cli: modelDestPath layout" {
    const allocator = testing.allocator;
    const p = try modelDestPath(allocator, "/Users/x", "org/repo");
    defer allocator.free(p);
    try testing.expectEqualStrings("/Users/x/.sushi/models/org/repo", p);
}

test "cli: shouldDownload chat-default selection" {
    try testing.expect(shouldDownload("config.json"));
    try testing.expect(shouldDownload("model.safetensors"));
    try testing.expect(shouldDownload("model-00001-of-00002.safetensors"));
    try testing.expect(shouldDownload("tokenizer.json"));
    try testing.expect(shouldDownload("chat_template.jinja"));
    try testing.expect(shouldDownload("mtp/weights.safetensors"));
    try testing.expect(shouldDownload("GLM-5.3-Flash-DFlash2/model.safetensors"));
    try testing.expect(shouldDownload("GLM-5.3-Flash-DFlash2/config.json"));
    try testing.expect(!shouldDownload("GLM-5.3-Flash-DFlash2/README.md"));
    try testing.expect(!shouldDownload("GLM-5.3-Flash-DFlash2/assets/demo.png"));
    try testing.expect(!shouldDownload("GLM-5.3-Flash-DFlash2-other/model.safetensors"));
    try testing.expect(!shouldDownload("dflash2/model.safetensors")); // the derived runtime cache is never downloaded
    try testing.expect(!shouldDownload(".gitattributes"));
    try testing.expect(!shouldDownload("README.md"));
    try testing.expect(!shouldDownload("assets/demo.png"));
    try testing.expect(!shouldDownload("banner.png"));
    try testing.expect(!shouldDownload("vae/weights.safetensors")); // media subdirs are app-bundle territory
    // A `.bin` the engine READS (qwen4_exp ngram_table) is needed; torch-format
    // shadow weights are a second copy of the same model. Same rule as the app's
    // `DownloadManager.selectNeededFiles` — keep them in sync.
    try testing.expect(shouldDownload("ngram_table.bin"));
    try testing.expect(!shouldDownload("pytorch_model-00001-of-00002.bin"));
    try testing.expect(!shouldDownload("consolidated.pth"));
    try testing.expect(!shouldDownload("flax_model.msgpack"));
}

test "cli: modelPresentInDir requires a COMPLETE checkpoint" {
    // Regression: an interrupted `pull` (Ctrl-C mid-weights) leaves
    // config.json + model.safetensors.partial. modelPresent used to return
    // true on config.json alone, so the rerun skipped the resume and fed a
    // weightless dir to the loader (SIGSEGV). Present now means: no .partial
    // leftovers anywhere (top level or one subdir deep), and config.json +
    // >=1 .safetensors (MLX) or any .gguf.
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    // config.json alone (weights never started): not present.
    try tmp.dir.createDirPath(io, "a");
    try tmp.dir.writeFile(io, .{ .sub_path = "a/config.json", .data = "{}" });
    {
        var d = try tmp.dir.openDir(io, "a", .{ .iterate = true });
        defer d.close(io);
        try testing.expect(!modelPresentInDir(io, d));
    }

    // config.json + interrupted weights: not present (the user's live repro).
    try tmp.dir.writeFile(io, .{ .sub_path = "a/model.safetensors.partial", .data = "x" });
    {
        var d = try tmp.dir.openDir(io, "a", .{ .iterate = true });
        defer d.close(io);
        try testing.expect(!modelPresentInDir(io, d));
    }

    // Complete single-file checkpoint: present.
    try tmp.dir.createDirPath(io, "b");
    try tmp.dir.writeFile(io, .{ .sub_path = "b/config.json", .data = "{}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "b/model.safetensors", .data = "x" });
    {
        var d = try tmp.dir.openDir(io, "b", .{ .iterate = true });
        defer d.close(io);
        try testing.expect(modelPresentInDir(io, d));
    }

    // Complete weights but another file still partial (e.g. tokenizer.json):
    // not present — resume must finish the pull.
    try tmp.dir.writeFile(io, .{ .sub_path = "b/tokenizer.json.partial", .data = "x" });
    {
        var d = try tmp.dir.openDir(io, "b", .{ .iterate = true });
        defer d.close(io);
        try testing.expect(!modelPresentInDir(io, d));
    }

    // Interrupted sidecar one subdir deep (mtp/weights.safetensors.partial):
    // not present.
    try tmp.dir.createDirPath(io, "c/mtp");
    try tmp.dir.writeFile(io, .{ .sub_path = "c/config.json", .data = "{}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "c/model.safetensors", .data = "x" });
    try tmp.dir.writeFile(io, .{ .sub_path = "c/mtp/weights.safetensors.partial", .data = "x" });
    {
        var d = try tmp.dir.openDir(io, "c", .{ .iterate = true });
        defer d.close(io);
        try testing.expect(!modelPresentInDir(io, d));
    }

    // GGUF: the file itself is the checkpoint (no config.json needed)…
    try tmp.dir.createDirPath(io, "g");
    try tmp.dir.writeFile(io, .{ .sub_path = "g/model-Q4_K_M.gguf", .data = "x" });
    {
        var d = try tmp.dir.openDir(io, "g", .{ .iterate = true });
        defer d.close(io);
        try testing.expect(modelPresentInDir(io, d));
    }

    // …but a partial GGUF is not.
    try tmp.dir.createDirPath(io, "h");
    try tmp.dir.writeFile(io, .{ .sub_path = "h/model-Q4_K_M.gguf.partial", .data = "x" });
    {
        var d = try tmp.dir.openDir(io, "h", .{ .iterate = true });
        defer d.close(io);
        try testing.expect(!modelPresentInDir(io, d));
    }
}

test "cli: parseTreeJson uses lfs size and skips directories" {
    const allocator = testing.allocator;
    const files = try parseTreeJson(allocator,
        \\[{"type":"directory","path":"mtp","size":0},
        \\ {"type":"file","path":"config.json","size":1234},
        \\ {"type":"file","path":"model.safetensors","size":135,"lfs":{"size":5300000000,"sha256":"x"}}]
    );
    defer freeRepoFiles(allocator, files);
    try testing.expectEqual(@as(usize, 2), files.len);
    try testing.expectEqualStrings("config.json", files[0].path);
    try testing.expectEqual(@as(u64, 1234), files[0].size);
    try testing.expectEqual(@as(u64, 5_300_000_000), files[1].size);
}

test "cli: buildReplChatBody and parseReplLine speak /v1/chat/completions SSE" {
    const allocator = testing.allocator;
    const history = [_]Turn{
        .{ .role = "user", .content = "hi \"there\"\n" },
        .{ .role = "assistant", .content = "hello" },
    };
    const body = try buildReplChatBody(allocator, &history, .model_default, null);
    defer allocator.free(body);
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();
    const msgs = parsed.value.object.get("messages").?.array.items;
    try testing.expectEqual(@as(usize, 2), msgs.len);
    try testing.expectEqualStrings("hi \"there\"\n", msgs[0].object.get("content").?.string);
    try testing.expect(parsed.value.object.get("stream").?.bool);
    try testing.expect(parsed.value.object.get("stream_options").?.object.get("include_usage").?.bool);

    const d1 = parseReplLine(allocator, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hey\"},\"finish_reason\":null}]}").?;
    defer allocator.free(d1.content);
    try testing.expectEqualStrings("Hey", d1.content);
    try testing.expect(!d1.done);

    const d2 = parseReplLine(allocator, "data: {\"choices\":[],\"usage\":{\"completion_tokens\":50},\"timings\":{\"predicted_n\":50,\"predicted_ms\":2000.0}}").?;
    defer allocator.free(d2.content);
    try testing.expect(!d2.done);
    try testing.expectEqual(@as(u64, 50), d2.stats.eval_count);
    try testing.expectEqual(@as(u64, 2_000_000_000), d2.stats.eval_duration_ns);
    var stats_buf: [128]u8 = undefined;
    try testing.expectEqualStrings("[50 tokens, 25.0 tok/s]", formatReplStats(&stats_buf, d2.stats));

    const cold = parseReplLine(allocator, "data: {\"choices\":[],\"timings\":{\"prompt_n\":4204,\"cached_n\":0,\"prompt_ms\":4934.272,\"prompt_per_second\":852.3,\"predicted_n\":128,\"predicted_ms\":4309.764}}").?;
    defer allocator.free(cold.content);
    try testing.expectEqual(@as(u64, 4204), cold.stats.prompt_n);
    try testing.expectEqualStrings("[prefill 4204 tok, 852.3 tok/s | 128 tokens, 29.7 tok/s]", formatReplStats(&stats_buf, cold.stats));

    const warm = parseReplLine(allocator, "data: {\"choices\":[],\"timings\":{\"prompt_n\":4204,\"cached_n\":4192,\"prompt_ms\":40.0,\"prompt_per_second\":300.0,\"predicted_n\":128,\"predicted_ms\":4309.764}}").?;
    defer allocator.free(warm.content);
    try testing.expectEqualStrings("[prefill 12 tok (4192 cached), 300.0 tok/s | 128 tokens, 29.7 tok/s]", formatReplStats(&stats_buf, warm.stats));
    try testing.expectEqualStrings("", formatReplStats(&stats_buf, d1.stats));

    const d3 = parseReplLine(allocator, "data: [DONE]").?;
    defer allocator.free(d3.content);
    try testing.expect(d3.done);

    const d4 = parseReplLine(allocator, "data: {\"error\":{\"message\":\"boom\",\"type\":\"x\"}}").?;
    defer allocator.free(d4.content);
    defer if (d4.err) |e| allocator.free(e);
    try testing.expect(d4.done);
    try testing.expectEqualStrings("boom", d4.err.?);

    // SSE comments and keepalives carry no event.
    try testing.expect(parseReplLine(allocator, ": keepalive") == null);
}

test "cli: --think takes the next argument only when it is an effort word" {
    try testing.expectEqual(ThinkFlag{ .think = .on, .consumed = false }, parseThinkFlag(null));
    try testing.expectEqual(ThinkFlag{ .think = .on, .consumed = false }, parseThinkFlag("--port"));
    try testing.expectEqual(ThinkFlag{ .think = .{ .effort = .xhigh }, .consumed = true }, parseThinkFlag("xhigh"));
    try testing.expectEqual(ThinkFlag{ .think = .{ .effort = .off }, .consumed = true }, parseThinkFlag("off"));
    try testing.expectEqual(ThinkFlag{ .think = .{ .effort = .off }, .consumed = true }, parseThinkFlag("none"));
}

test "cli: /think shows, sets an accepted word, and refuses the rest" {
    const qwen = [_]model.Effort{ .off, .low, .medium, .xhigh };
    try testing.expectEqual(@as(?ThinkCommand, null), parseThinkCommand("/thinking about it", &qwen));
    try testing.expectEqual(@as(?ThinkCommand, null), parseThinkCommand("hello", &qwen));
    try testing.expectEqual(@as(?ThinkCommand, .show), parseThinkCommand("/think", &qwen));
    try testing.expectEqual(@as(?ThinkCommand, .{ .set = .{ .effort = .medium } }), parseThinkCommand("/think medium", &qwen));
    try testing.expectEqual(@as(?ThinkCommand, .{ .set = .{ .effort = .off } }), parseThinkCommand("/think  off ", &qwen));
    try testing.expectEqualStrings("high", parseThinkCommand("/think high", &qwen).?.refuse);
    try testing.expectEqualStrings("hgih", parseThinkCommand("/think hgih", &qwen).?.refuse);
    // A server that lists no efforts leaves the whole vocabulary to it.
    try testing.expectEqual(@as(?ThinkCommand, .{ .set = .{ .effort = .max } }), parseThinkCommand("/think max", &.{}));

    var buf: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try writeEffortRefusal(&w, "high", "Qwen3.8-Flash-Next-EXL3-K4", &qwen);
    try testing.expectEqualStrings("reasoning effort 'high' is not supported by Qwen3.8-Flash-Next-EXL3-K4; use one of: off, low, medium, xhigh", w.buffered());
    w = .fixed(&buf);
    try writeThinkSetting(&w, .{ .effort = .low }, &qwen);
    try testing.expectEqualStrings("thinking: low (options: off, low, medium, xhigh)", w.buffered());
    w = .fixed(&buf);
    try writeThinkSetting(&w, .model_default, &qwen);
    try testing.expectEqualStrings("thinking: model default (options: off, low, medium, xhigh)", w.buffered());
}

test "cli: the REPL chat body carries the thinking setting" {
    const allocator = testing.allocator;
    const history = [_]Turn{.{ .role = "user", .content = "hi" }};
    const Case = struct { think: Think, enable: ?bool, effort: ?[]const u8 };
    for ([_]Case{
        .{ .think = .model_default, .enable = null, .effort = null },
        .{ .think = .on, .enable = true, .effort = null },
        .{ .think = .{ .effort = .low }, .enable = null, .effort = "low" },
        .{ .think = .{ .effort = .off }, .enable = null, .effort = "off" },
    }) |c| {
        const body = try buildReplChatBody(allocator, &history, c.think, null);
        defer allocator.free(body);
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
        defer parsed.deinit();
        const root = parsed.value.object;
        if (c.enable) |e| try testing.expectEqual(e, root.get("enable_thinking").?.bool) else try testing.expect(root.get("enable_thinking") == null);
        if (c.effort) |e| try testing.expectEqualStrings(e, root.get("reasoning_effort").?.string) else try testing.expect(root.get("reasoning_effort") == null);
    }
}

test "cli: the REPL reads the model's efforts from /v1/models and streams reasoning deltas" {
    const allocator = testing.allocator;
    const info = try parseModelEfforts(allocator,
        \\{"object":"list","data":[{"id":"MiMo","capabilities":["chat","vision"],"reasoning_efforts":["off","low","max"]},{"id":"other"}]}
    );
    defer info.deinit(allocator);
    try testing.expectEqualStrings("MiMo", info.id);
    try testing.expectEqualSlices(model.Effort, &.{ .off, .low, .max }, info.efforts);
    try testing.expect(info.vision);
    const bare = try parseModelEfforts(allocator, "{\"data\":[{\"id\":\"old\",\"capabilities\":[\"chat\"]}]}");
    defer bare.deinit(allocator);
    try testing.expectEqual(@as(usize, 0), bare.efforts.len);
    try testing.expect(!bare.vision);

    const d = parseReplLine(allocator, "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"hmm\"}}]}").?;
    defer allocator.free(d.content);
    defer if (d.reasoning) |r| allocator.free(r);
    try testing.expectEqualStrings("hmm", d.reasoning.?);
    try testing.expectEqualStrings("", d.content);
}

test "cli: tools and editing are off by default; --tool, --edit, /tool, /edit and /image parse" {
    const defaults: ReplOptions = .{};
    try testing.expect(!defaults.tools);

    try testing.expectEqual(@as(?bool, true), parseToolSwitch("on"));
    try testing.expectEqual(@as(?bool, false), parseToolSwitch("off"));
    try testing.expectEqual(@as(?bool, null), parseToolSwitch("yes"));

    try testing.expectEqual(@as(?ToolCommand, .show), parseToolCommand("/tool"));
    try testing.expectEqual(@as(?ToolCommand, .{ .set = true }), parseToolCommand("/tool on"));
    try testing.expectEqual(@as(?ToolCommand, .{ .set = false }), parseToolCommand("/tool  off "));
    try testing.expectEqualStrings("maybe", parseToolCommand("/tool maybe").?.refuse);
    try testing.expectEqual(@as(?ToolCommand, null), parseToolCommand("/tools on"));
    try testing.expectEqual(@as(?ToolCommand, null), parseToolCommand("/toolbox"));
    try testing.expectEqual(@as(?ToolCommand, null), parseToolCommand("tool on"));

    try testing.expectEqual(@as(?ToolCommand, .show), parseEditCommand("/edit"));
    try testing.expectEqual(@as(?ToolCommand, .{ .set = true }), parseEditCommand("/edit on"));
    try testing.expectEqual(@as(?ToolCommand, .{ .set = false }), parseEditCommand("/edit  off "));
    try testing.expectEqualStrings("maybe", parseEditCommand("/edit maybe").?.refuse);
    try testing.expectEqual(@as(?ToolCommand, null), parseEditCommand("/editor on"));
    try testing.expectEqual(@as(?ToolCommand, null), parseEditCommand("edit on"));
    try testing.expect(!@as(ReplOptions, .{}).edit);
    // The pack the REPL offers grows the write pair only when both switches are on.
    try testing.expectEqual(@as(?[]const u8, null), @as(ReplOptions, .{}).toolsJson(false));
    try testing.expect(std.mem.indexOf(u8, @as(ReplOptions, .{ .tools = true }).toolsJson(false).?, "write_file") == null);
    try testing.expect(std.mem.indexOf(u8, @as(ReplOptions, .{ .tools = true, .edit = true }).toolsJson(false).?, "edit_file") != null);

    try testing.expectEqualStrings("shot.png", parseImageCommand("/image shot.png").?);
    try testing.expectEqualStrings("", parseImageCommand("/image").?);
    try testing.expectEqual(@as(?[]const u8, null), parseImageCommand("/images x"));
    try testing.expectEqual(@as(?[]const u8, null), parseImageCommand("look at shot.png"));
    const allocator = testing.allocator;
    for ([_][2][]const u8{
        .{ "'My Shot.png'", "My Shot.png" },
        .{ "\"My Shot.png\"", "My Shot.png" },
        .{ "My\\ Shot\\ 2.png", "My Shot 2.png" },
        .{ "plain.png", "plain.png" },
    }) |c| {
        const got = try unquotePath(allocator, c[0]);
        defer allocator.free(got);
        try testing.expectEqualStrings(c[1], got);
    }
}

test "cli: the ready banner shows the folder, the tools state, /cd and the browser chat page" {
    for ([_]bool{ false, true }) |vision| {
        var buf: [512]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buf);
        try writeReadyBanner(&w, vision, 18800, "~/project · tools off");
        const text = w.buffered();
        try testing.expect(std.mem.indexOf(u8, text, "chat in your browser: http://127.0.0.1:18800/\n") != null);
        try testing.expect(std.mem.indexOf(u8, text, "/tool on") != null);
        try testing.expect(std.mem.indexOf(u8, text, "~/project · tools off") != null);
        try testing.expect(std.mem.indexOf(u8, text, "/cd <folder>") != null);
        try testing.expectEqual(vision, std.mem.indexOf(u8, text, "/image") != null);
    }
}

test "cli: the REPL names a newer release and /update, or brew for a Homebrew install, and says nothing without one" {
    var buf: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try writeUpdateNotice(&w, null, false);
    try testing.expectEqualStrings("", w.buffered());
    try writeUpdateNotice(&w, "1.1.0", false);
    try testing.expectEqualStrings(">>> sushi 1.1.0 is available: /update installs it and restarts this chat on the same model\n", w.buffered());
    w = .fixed(&buf);
    try writeUpdateNotice(&w, "1.1.0", true);
    try testing.expectEqualStrings(">>> sushi 1.1.0 is available: run brew upgrade sushi\n", w.buffered());
}

test "cli: the prompt status shows the tools' folder (~ for home, the tail when long) and the tools state" {
    const Case = struct { root: []const u8, tools: bool, want: []const u8 };
    for ([_]Case{
        .{ .root = "/Users/me", .tools = false, .want = "~ · tools off" },
        .{ .root = "/Users/me/project", .tools = true, .want = "~/project · tools on" },
        .{ .root = "/Users/meg/project", .tools = false, .want = "/Users/meg/project · tools off" },
        .{ .root = "/opt/data", .tools = true, .want = "/opt/data · tools on" },
        .{ .root = "/", .tools = false, .want = "/ · tools off" },
        .{ .root = "/Users/me/work/clients/acme/backend/services/billing", .tools = true, .want = "…/acme/backend/services/billing · tools on" },
        .{ .root = "/Volumes/x/a-single-folder-name-longer-than-the-limit-allows", .tools = false, .want = "…/a-single-folder-name-longer-than-the-limit-allows · tools off" },
    }) |c| {
        var buf: [256]u8 = undefined;
        try testing.expectEqualStrings(c.want, formatPromptStatus(&buf, c.root, "/Users/me", c.tools));
    }
    var buf: [256]u8 = undefined;
    try testing.expectEqualStrings("/Users/me/p · tools off", formatPromptStatus(&buf, "/Users/me/p", "", false));

    var out: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&out);
    try writePrompt(&w, "~/project · tools on");
    try testing.expectEqualStrings("\x1b[2m~/project · tools on\x1b[0m >>> ", w.buffered());
}

test "cli: /cd parses its folder argument" {
    try testing.expectEqualStrings("", parseCdCommand("/cd").?);
    try testing.expectEqualStrings("src", parseCdCommand("/cd  src ").?);
    try testing.expectEqualStrings("~/My Folder", parseCdCommand("/cd ~/My Folder").?);
    try testing.expectEqual(@as(?[]const u8, null), parseCdCommand("/cdx"));
    try testing.expectEqual(@as(?[]const u8, null), parseCdCommand("cd src"));
}

test "cli: /cd moves the file tools' folder, and a refused /cd keeps it" {
    const allocator = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "proj/my sub");
    var tools: repl_tools.Context = .{ .allocator = allocator, .io = io, .root = try tmp.dir.realPathFileAlloc(io, "proj", allocator), .vision = false };
    defer allocator.free(tools.root);

    var buf: [2048]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try changeToolFolder(allocator, io, &w, &tools, "'my sub'");
    try testing.expect(std.mem.endsWith(u8, tools.root, "/proj/my sub"));
    try changeToolFolder(allocator, io, &w, &tools, "missing");
    try testing.expect(std.mem.endsWith(u8, tools.root, "/proj/my sub"));
    try changeToolFolder(allocator, io, &w, &tools, "..");
    try testing.expect(std.mem.endsWith(u8, tools.root, "/proj"));
    try changeToolFolder(allocator, io, &w, &tools, "");
    try testing.expect(std.mem.endsWith(u8, tools.root, "/proj"));

    const text = w.buffered();
    try testing.expect(std.mem.indexOf(u8, text, "cannot /cd to missing: no such folder\n") != null);
    try testing.expectEqual(@as(usize, 3), std.mem.count(u8, text, "file tools read under "));
}

test "cli: a relative /image follows /cd and stays in the folder; an absolute one may leave it" {
    const allocator = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "proj/shots");
    try tmp.dir.writeFile(io, .{ .sub_path = "proj/shots/pic.png", .data = "\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR" });
    try tmp.dir.writeFile(io, .{ .sub_path = "desk.png", .data = "\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR" });
    const base = try tmp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(base);
    const desk_abs = try std.fmt.allocPrint(allocator, "{s}/desk.png", .{base});
    defer allocator.free(desk_abs);
    var tools: repl_tools.Context = .{ .allocator = allocator, .io = io, .root = try tmp.dir.realPathFileAlloc(io, "proj", allocator), .vision = true };
    defer allocator.free(tools.root);
    var pending = std.ArrayList([]const u8).empty;
    defer {
        for (pending.items) |i| allocator.free(i);
        pending.deinit(allocator);
    }

    var buf: [2048]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try attachImage(allocator, &w, tools, "pic.png", &pending);
    try testing.expectEqual(@as(usize, 0), pending.items.len);
    try changeToolFolder(allocator, io, &w, &tools, "shots");
    try attachImage(allocator, &w, tools, "pic.png", &pending);
    try testing.expectEqual(@as(usize, 1), pending.items.len);
    try attachImage(allocator, &w, tools, "../../proj/shots/pic.png", &pending);
    try testing.expectEqual(@as(usize, 1), pending.items.len);
    try attachImage(allocator, &w, tools, desk_abs, &pending);
    try testing.expectEqual(@as(usize, 2), pending.items.len);

    const text = w.buffered();
    try testing.expect(std.mem.indexOf(u8, text, "cannot attach pic.png: no such file") != null);
    try testing.expect(std.mem.indexOf(u8, text, "attached pic.png; it goes with your next message\n") != null);
    try testing.expect(std.mem.indexOf(u8, text, "cannot attach ../../proj/shots/pic.png: ") != null);
}

test "cli: the chat body carries tools only while they are on, plus tool turns and image parts" {
    const allocator = testing.allocator;
    const images = [_][]const u8{"data:image/png;base64,AAAA"};
    const history = [_]Turn{
        .{ .role = "user", .content = "what is this?", .images = &images },
        .{ .role = "assistant", .content = "", .tool_calls_json = "[{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"web_search\",\"arguments\":\"{\\\"query\\\":\\\"x\\\"}\"}}]" },
        .{ .role = "tool", .content = "1. result", .tool_call_id = "call_1" },
    };
    const tools = repl_tools.definitionsJson(false, false);
    for ([_]?[]const u8{ tools, null }) |offered| {
        const body = try buildReplChatBody(allocator, &history, .model_default, offered);
        defer allocator.free(body);
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
        defer parsed.deinit();
        const root = parsed.value.object;
        if (offered != null) {
            try testing.expectEqual(@as(usize, 5), root.get("tools").?.array.items.len);
        } else {
            try testing.expect(root.get("tools") == null);
        }
        const msgs = root.get("messages").?.array.items;
        const parts = msgs[0].object.get("content").?.array.items;
        try testing.expectEqualStrings("what is this?", parts[0].object.get("text").?.string);
        try testing.expectEqualStrings("data:image/png;base64,AAAA", parts[1].object.get("image_url").?.object.get("url").?.string);
        const call = msgs[1].object.get("tool_calls").?.array.items[0].object;
        try testing.expectEqualStrings("web_search", call.get("function").?.object.get("name").?.string);
        try testing.expectEqualStrings("tool", msgs[2].object.get("role").?.string);
        try testing.expectEqualStrings("call_1", msgs[2].object.get("tool_call_id").?.string);
        try testing.expectEqualStrings("1. result", msgs[2].object.get("content").?.string);
    }
    // A REPL started with tools off sends none.
    const plain = try buildReplChatBody(allocator, history[2..], .model_default, (ReplOptions{}).toolsJson(true));
    defer allocator.free(plain);
    try testing.expect(std.mem.indexOf(u8, plain, "\"tools\"") == null);
}

test "cli: parseReplLine reads streamed tool calls" {
    const allocator = testing.allocator;
    const d = parseReplLine(allocator,
        \\data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_7_0","type":"function","function":{"name":"fetch_url","arguments":"{\"url\":\"https://ziglang.org\"}"}}]}}]}
    ).?;
    defer d.deinit(allocator);
    try testing.expectEqual(@as(usize, 1), d.tool_calls.len);
    try testing.expectEqualStrings("call_7_0", d.tool_calls[0].id);
    try testing.expectEqualStrings("fetch_url", d.tool_calls[0].name);
    try testing.expectEqualStrings("{\"url\":\"https://ziglang.org\"}", d.tool_calls[0].arguments);
}

/// Scripted server + tool runner for the tool loop.
const StubServer = struct {
    allocator: std.mem.Allocator,
    /// Tool calls returned while tools are offered, per round; then a final answer.
    rounds_with_calls: usize,
    call_name: []const u8 = "web_search",
    tool_image: bool = false,
    requests: usize = 0,
    requests_with_tools: usize = 0,
    ran: std.ArrayList(u8) = .empty,
    last_body: ?[]u8 = null,

    fn complete(ptr: *anyopaque, body: []const u8) anyerror!Reply {
        const s: *StubServer = @ptrCast(@alignCast(ptr));
        s.requests += 1;
        if (s.last_body) |b| s.allocator.free(b);
        s.last_body = try s.allocator.dupe(u8, body);
        const offered = std.mem.indexOf(u8, body, "\"tools\":[") != null;
        if (offered) s.requests_with_tools += 1;
        if (offered and s.requests <= s.rounds_with_calls) {
            const calls = try s.allocator.alloc(ToolCall, 1);
            calls[0] = .{
                .id = try std.fmt.allocPrint(s.allocator, "call_{d}", .{s.requests}),
                .name = try s.allocator.dupe(u8, s.call_name),
                .arguments = try s.allocator.dupe(u8, "{\"query\":\"q\"}"),
            };
            return .{ .content = try s.allocator.dupe(u8, ""), .tool_calls = calls };
        }
        return .{ .content = try s.allocator.dupe(u8, "final answer"), .tool_calls = &.{} };
    }

    fn runTool(ptr: *anyopaque, name: []const u8, args: []const u8) anyerror!repl_tools.Output {
        const s: *StubServer = @ptrCast(@alignCast(ptr));
        _ = args;
        try s.ran.appendSlice(s.allocator, name);
        try s.ran.append(s.allocator, ' ');
        return .{
            .text = try s.allocator.dupe(u8, "result text"),
            .image = if (s.tool_image) try s.allocator.dupe(u8, "data:image/png;base64,BBBB") else null,
        };
    }

    fn driver(s: *StubServer) TurnDriver {
        return .{ .ptr = s, .complete = complete, .runTool = runTool };
    }

    fn deinit(s: *StubServer) void {
        s.ran.deinit(s.allocator);
        if (s.last_body) |b| s.allocator.free(b);
    }
};

fn freeHistory(allocator: std.mem.Allocator, history: *std.ArrayList(Turn)) void {
    for (history.items) |t| t.deinit(allocator);
    history.deinit(allocator);
}

fn roles(allocator: std.mem.Allocator, history: []const Turn) ![]u8 {
    var out = std.ArrayList(u8).empty;
    for (history) |t| {
        try out.appendSlice(allocator, t.role);
        try out.append(allocator, ' ');
    }
    return out.toOwnedSlice(allocator);
}

test "cli: the tool loop runs calls client-side until the model answers" {
    const allocator = testing.allocator;
    var stub: StubServer = .{ .allocator = allocator, .rounds_with_calls = 2 };
    defer stub.deinit();
    var history = std.ArrayList(Turn).empty;
    defer freeHistory(allocator, &history);
    try history.append(allocator, .{ .role = "user", .content = try allocator.dupe(u8, "latest zig?") });
    try runToolTurn(allocator, &history, .model_default, repl_tools.definitionsJson(false, false), stub.driver());

    try testing.expectEqual(@as(usize, 3), stub.requests);
    try testing.expectEqual(@as(usize, 3), stub.requests_with_tools);
    try testing.expectEqualStrings("web_search web_search ", stub.ran.items);
    const r = try roles(allocator, history.items);
    defer allocator.free(r);
    try testing.expectEqualStrings("user assistant tool assistant tool assistant ", r);
    try testing.expectEqualStrings("call_1", history.items[2].tool_call_id.?);
    try testing.expectEqualStrings("result text", history.items[2].content);
    try testing.expectEqualStrings("final answer", history.items[5].content);
    try testing.expect(history.items[5].tool_calls_json == null);
}

test "cli: the tool loop stops offering tools after 8 rounds and asks for an answer" {
    const allocator = testing.allocator;
    var stub: StubServer = .{ .allocator = allocator, .rounds_with_calls = 100 };
    defer stub.deinit();
    var history = std.ArrayList(Turn).empty;
    defer freeHistory(allocator, &history);
    try history.append(allocator, .{ .role = "user", .content = try allocator.dupe(u8, "dig forever") });
    try runToolTurn(allocator, &history, .model_default, repl_tools.definitionsJson(false, false), stub.driver());

    try testing.expectEqual(@as(usize, max_tool_rounds + 1), stub.requests);
    try testing.expectEqual(@as(usize, max_tool_rounds), stub.requests_with_tools);
    try testing.expect(std.mem.indexOf(u8, stub.last_body.?, "\"tools\"") == null);
    try testing.expectEqualStrings("final answer", history.items[history.items.len - 1].content);
    try testing.expectEqualStrings("user", history.items[history.items.len - 2].role);
    try testing.expectEqualStrings(tool_cap_nudge, history.items[history.items.len - 2].content);
}

test "cli: a tool image reaches the next request as a user image part; tools off sends none" {
    const allocator = testing.allocator;
    var stub: StubServer = .{ .allocator = allocator, .rounds_with_calls = 1, .call_name = "view_image", .tool_image = true };
    defer stub.deinit();
    var history = std.ArrayList(Turn).empty;
    defer freeHistory(allocator, &history);
    try history.append(allocator, .{ .role = "user", .content = try allocator.dupe(u8, "look at cat.png") });
    try runToolTurn(allocator, &history, .model_default, repl_tools.definitionsJson(true, false), stub.driver());
    const r = try roles(allocator, history.items);
    defer allocator.free(r);
    try testing.expectEqualStrings("user assistant tool user assistant ", r);
    try testing.expectEqualStrings("data:image/png;base64,BBBB", history.items[3].images[0]);
    try testing.expect(std.mem.indexOf(u8, stub.last_body.?, "data:image/png;base64,BBBB") != null);

    var off: StubServer = .{ .allocator = allocator, .rounds_with_calls = 5 };
    defer off.deinit();
    var h2 = std.ArrayList(Turn).empty;
    defer freeHistory(allocator, &h2);
    try h2.append(allocator, .{ .role = "user", .content = try allocator.dupe(u8, "hi") });
    try runToolTurn(allocator, &h2, .model_default, null, off.driver());
    try testing.expectEqual(@as(usize, 1), off.requests);
    try testing.expectEqual(@as(usize, 0), off.requests_with_tools);
    try testing.expectEqualStrings("", off.ran.items);
}

test "cli: formatSize" {
    var buf: [32]u8 = undefined;
    try testing.expectEqualStrings("5.2 GB", formatSize(&buf, 5_600_000_000));
    try testing.expectEqualStrings("35 MB", formatSize(&buf, 36_700_160));
    try testing.expectEqualStrings("2 KB", formatSize(&buf, 2048));
}

test "cli: formatMemorySummary reports real, free and total on one line" {
    var buf: [160]u8 = undefined;
    const gb: u64 = 1024 * 1024 * 1024;
    // Post-load line: process footprint ("real"), headroom for a new large
    // allocation ("free"), physical RAM ("total"), each through formatSize.
    const line = try formatMemorySummary(&buf, 20 * gb + gb / 2, 6 * gb + gb / 2, 32 * gb);
    try testing.expectEqualStrings("[mem] real 20.5 GB, free 6.5 GB, total 32.0 GB", line);
    // Sub-GB figures ride formatSize's MB arm instead of reading "0.0 GB".
    const small = try formatMemorySummary(&buf, 512 * 1024 * 1024, gb, 2 * gb);
    try testing.expectEqualStrings("[mem] real 512 MB, free 1.0 GB, total 2.0 GB", small);
}

test "cli: dirBytesOneLevel counts weight subdirs (FLUX bundle showed 6 KB)" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    // Media-bundle shape: small top-level config + the actual weights one
    // level down in transformer/ and vae/ (the modelPresent layout).
    try tmp.dir.createDirPath(io, "m/transformer");
    try tmp.dir.createDirPath(io, "m/vae");
    try tmp.dir.writeFile(io, .{ .sub_path = "m/config.json", .data = "{}" }); // 2 bytes
    try tmp.dir.writeFile(io, .{ .sub_path = "m/transformer/w.safetensors", .data = "0123456789" }); // 10
    try tmp.dir.writeFile(io, .{ .sub_path = "m/vae/w.safetensors", .data = "0123" }); // 4

    var m = try tmp.dir.openDir(io, "m", .{ .iterate = true });
    defer m.close(io);
    try testing.expectEqual(@as(u64, 16), dirBytesOneLevel(io, &m));
}

test "cli: a serving boot with no model named falls back to the shared models root" {
    // `sushi serve` and `sushi run <m>` already default the discovery
    // root; the `--serve` FLAG form did not, so a bare `sushi --serve`
    // booted a server that had discovered nothing and could only answer 503 —
    // never what anyone meant by "serve".
    try testing.expect(shouldDefaultModelsRoot(.{ .subcommand = true, .serve_mode = true, .has_explicit_model = false }));
    try testing.expect(shouldDefaultModelsRoot(.{ .subcommand = false, .serve_mode = true, .has_explicit_model = false }));

    // `--model <path> --serve` asked for ONE model. Quietly registering the
    // other 28 on disk is a different server than the one requested.
    try testing.expect(!shouldDefaultModelsRoot(.{ .subcommand = false, .serve_mode = true, .has_explicit_model = true }));
    // `sushi run <model>` names a model AND wants the picker populated —
    // the subcommand's existing behavior, which this must not change.
    try testing.expect(shouldDefaultModelsRoot(.{ .subcommand = true, .serve_mode = true, .has_explicit_model = true }));

    // Not serving at all (one-shot `--prompt`) never scans a root.
    try testing.expect(!shouldDefaultModelsRoot(.{ .subcommand = false, .serve_mode = false, .has_explicit_model = true }));
    try testing.expect(!shouldDefaultModelsRoot(.{ .subcommand = false, .serve_mode = false, .has_explicit_model = false }));
}

test "cli: an unparsed argument is classified, never silently ignored" {
    // `--flag=value`: main.zig's flag loop matches names EXACTLY and takes the
    // value as the NEXT argument, so the '='-joined form is not a near-miss,
    // it is a shape we have never accepted. It used to fall out of the loop in
    // silence — `--model=<path>` booted a healthy-looking headless server that
    // then auto-picked a different model and crashed on it.
    try testing.expectEqual(ArgReject.equals_form, classifyUnparsedArg("--model=/tmp/m", false));
    try testing.expectEqual(ArgReject.equals_form, classifyUnparsedArg("--port=1234", false));
    // Even in last position the '=' hint is the useful one.
    try testing.expectEqual(ArgReject.equals_form, classifyUnparsedArg("--model=/tmp/m", true));

    // A flag in the LAST position fell out because its value is missing —
    // the loop's `i + 1 < args.len` guard is the only way to reach here.
    try testing.expectEqual(ArgReject.missing_value, classifyUnparsedArg("--model", true));
    try testing.expectEqual(ArgReject.missing_value, classifyUnparsedArg("-h", true));

    // Anything else is simply not a flag we know.
    try testing.expectEqual(ArgReject.unknown, classifyUnparsedArg("--frobnicate", false));
    try testing.expectEqual(ArgReject.unknown, classifyUnparsedArg("stray", false));
    // An '=' outside a flag is not the equals form.
    try testing.expectEqual(ArgReject.unknown, classifyUnparsedArg("a=b", false));
    try testing.expectEqual(ArgReject.unknown, classifyUnparsedArg("a=b", true));

    // Every reason carries actionable advice, and the equals-form one names
    // the shape that actually works.
    for ([_]ArgReject{ .equals_form, .missing_value, .unknown }) |r| {
        try testing.expect(r.hint().len > 0);
    }
    try testing.expect(std.mem.indexOf(u8, ArgReject.equals_form.hint(), "separate argument") != null);
}

test "cli: list tree walk descends into symlinked model dirs" {
    // Moving a big checkpoint to an external drive and symlinking it back is
    // a supported layout (the H3 mirrors live that way): model_discovery's
    // walk accepts .sym_link entries, but `list` had its own private walk
    // that silently skipped them — both MiniMax mirrors vanished from `list`
    // while the server kept serving them. Both loops route through ONE
    // predicate now.
    try testing.expect(treeEntryDescends(.directory));
    try testing.expect(treeEntryDescends(.sym_link));
    try testing.expect(!treeEntryDescends(.file));

    // Source scan: the org-level and leaf-level loops in listModels must both
    // consult the predicate — a reintroduced raw `!= .directory` check is the
    // regression this pins. Needle split so the scan cannot match itself.
    const src = @embedFile("cli.zig");
    const needle = "treeEntry" ++ "Descends(";
    var found: usize = 0;
    var idx: usize = 0;
    while (std.mem.indexOfPos(u8, src, idx, needle)) |p| {
        found += 1;
        idx = p + needle.len;
    }
    // 1 definition + 3 in this test + at least 2 call sites in the walk.
    try testing.expect(found >= 6);
}

pub fn isPromptFlag(arg: []const u8) bool {
    return std.mem.eql(u8, arg, "--prompt") or std.mem.eql(u8, arg, "-p");
}

pub fn buildPromptBody(allocator: std.mem.Allocator, prompt: []const u8, think: Think, sampling: PromptSampling) ![]u8 {
    return buildChatBody(allocator, &.{.{ .role = "user", .content = prompt }}, think, null, sampling);
}

test "cli: one-shot prompt aliases" {
    try testing.expect(isPromptFlag("--prompt"));
    try testing.expect(isPromptFlag("-p"));
    try testing.expect(!isPromptFlag("--prompt=hi"));
}

test "cli: one-shot request preserves prompt and sampling with every thinking arm" {
    const a = testing.allocator;
    const prompt = "\n  literal \\\" text\t\x00🦀\n";
    const arms = [_]Think{ .model_default, .on, .{ .effort = .off }, .{ .effort = .low }, .{ .effort = .medium }, .{ .effort = .xhigh } };
    for (arms) |think| {
        const body = try buildPromptBody(a, prompt, think, .{ .max_tokens = 37, .temperature = 0.25, .top_p = 0.75, .top_k = 12 });
        defer a.free(body);
        const parsed = try std.json.parseFromSlice(std.json.Value, a, body, .{});
        defer parsed.deinit();
        const root = parsed.value.object;
        try testing.expectEqualStrings(prompt, root.get("messages").?.array.items[0].object.get("content").?.string);
        try testing.expect(root.get("max_tokens") != null);
        try testing.expectEqual(@as(i64, 37), root.get("max_tokens").?.integer);
        try testing.expectEqual(@as(f64, 0.25), root.get("temperature").?.float);
        try testing.expect(root.get("stream").?.bool);
        try testing.expectEqual(@as(f64, 0.75), root.get("top_p").?.float);
        try testing.expectEqual(@as(i64, 12), root.get("top_k").?.integer);
        switch (think) {
            .model_default => {
                try testing.expect(root.get("enable_thinking") == null);
                try testing.expect(root.get("reasoning_effort") == null);
            },
            .on => try testing.expect(root.get("enable_thinking").?.bool),
            .effort => |e| try testing.expectEqualStrings(@tagName(e), root.get("reasoning_effort").?.string),
        }
    }
}

/// The one-shot client uses the same request and decode policy as the HTTP API.
/// Buffering changes delivery only; both modes consume the same SSE bytes.
pub fn runPrompt(allocator: std.mem.Allocator, io: std.Io, port: u16, body: []const u8, stream: bool, api_key: ?[]const u8) !void {
    const url = try std.fmt.allocPrint(allocator, "http://127.0.0.1:{d}/v1/chat/completions", .{port});
    defer allocator.free(url);
    const auth = if (api_key) |key| try std.fmt.allocPrint(allocator, "Bearer {s}", .{key}) else null;
    defer if (auth) |v| allocator.free(v);
    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    var req = try client.request(.POST, try std.Uri.parse(url), .{
        .keep_alive = false,
        .redirect_behavior = .unhandled,
        .headers = .{
            .content_type = .{ .override = "application/json" },
            .accept_encoding = .{ .override = "identity" },
            .authorization = if (auth) |v| .{ .override = v } else .omit,
        },
    });
    defer req.deinit();
    req.transfer_encoding = .{ .content_length = body.len };
    var request_body = try req.sendBodyUnflushed(&.{});
    try request_body.writer.writeAll(body);
    try request_body.end();
    try req.connection.?.flush();
    var head_buffer: [8192]u8 = undefined;
    var response = try req.receiveHead(&head_buffer);
    var read_buffer: [64 * 1024]u8 = undefined;
    const r = response.reader(&read_buffer);
    if (response.head.status != .ok) {
        const detail = try r.allocRemaining(allocator, .limited(64 * 1024));
        defer allocator.free(detail);
        log.err("prompt HTTP {d}: {s}\n", .{ @backingInt(response.head.status), detail });
        return error.PromptHttpStatus;
    }
    var out_buf: [16 * 1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &out_buf);
    var buffered: std.Io.Writer.Allocating = .init(allocator);
    defer buffered.deinit();
    const target = if (stream) &stdout.interface else &buffered.writer;
    defer stdout.interface.flush() catch {};
    try readPromptStream(allocator, r, target);
    if (!stream) try stdout.interface.writeAll(buffered.written());
}

fn readPromptStream(allocator: std.mem.Allocator, r: *std.Io.Reader, w: *std.Io.Writer) !void {
    var in_thought = false;
    while (try r.takeDelimiter('\n')) |line| {
        const delta = parseReplLine(allocator, line) orelse continue;
        defer delta.deinit(allocator);
        if (delta.err) |e| {
            if (!@import("builtin").is_test) log.err("prompt: {s}\n", .{e});
            return error.PromptGenerationFailed;
        }
        if (delta.reasoning) |thought| {
            if (!in_thought) try w.writeAll("<think>\n");
            in_thought = true;
            try w.writeAll(thought);
        }
        if (delta.content.len > 0) {
            if (in_thought) try w.writeAll("\n</think>\n\n");
            in_thought = false;
            try w.writeAll(delta.content);
        }
        try w.flush();
        if (delta.done) return;
    }
    return error.PromptStreamIncomplete;
}

test "cli: one-shot stream preserves reasoning and content without terminal markup" {
    var r: std.Io.Reader = .fixed(
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"hmm\"}}]}\n\n" ++
            "data: {\"choices\":[{\"delta\":{\"content\":\"  answer\\n\"}}]}\n\n" ++
            "data: [DONE]\n\n",
    );
    var w: std.Io.Writer.Allocating = .init(testing.allocator);
    defer w.deinit();
    try readPromptStream(testing.allocator, &r, &w.writer);
    try testing.expectEqualStrings("<think>\nhmm\n</think>\n\n  answer\n", w.written());
}

test "cli: one-shot incomplete stream is a failure" {
    var r: std.Io.Reader = .fixed("data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n");
    var w: std.Io.Writer.Allocating = .init(testing.allocator);
    defer w.deinit();
    try testing.expectError(error.PromptStreamIncomplete, readPromptStream(testing.allocator, &r, &w.writer));
}

test "cli: one-shot generation error is a failure even with a done marker" {
    var r: std.Io.Reader = .fixed("data: {\"error\":{\"message\":\"decode failed\"}}\n\ndata: [DONE]\n\n");
    var w: std.Io.Writer.Allocating = .init(testing.allocator);
    defer w.deinit();
    try testing.expectError(error.PromptGenerationFailed, readPromptStream(testing.allocator, &r, &w.writer));
}

test "cli: one-shot and REPL omission keep the model default" {
    const a = testing.allocator;
    const prompt = try buildPromptBody(a, "hello", .model_default, .{});
    defer a.free(prompt);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, prompt, .{});
    defer parsed.deinit();
    const effort = parsed.value.object.get("reasoning_effort");
    try testing.expect(effort == null);
    const repl = try buildReplChatBody(a, &.{.{ .role = "user", .content = "hello" }}, .model_default, null);
    defer a.free(repl);
    const parsed_repl = try std.json.parseFromSlice(std.json.Value, a, repl, .{});
    defer parsed_repl.deinit();
    try testing.expect(parsed_repl.value.object.get("reasoning_effort") == null);
    try testing.expect(parsed_repl.value.object.get("enable_thinking") == null);
}


test "thinking policy CLI: MiMo takes every thinking level as on, Qwen and GLM only their own words" {
    try testing.expectEqual(ThinkFlag{ .think = .{ .effort = .on }, .consumed = true }, parseThinkFlag("on"));
    try testing.expectEqual(ThinkFlag{ .think = .{ .effort = .minimal }, .consumed = true }, parseThinkFlag("minimal"));
    const mimo = [_]model.Effort{ .off, .on };
    for ([_][]const u8{ "/think on", "/think minimal", "/think low", "/think medium", "/think high", "/think xhigh", "/think max" }) |line| {
        try testing.expectEqual(@as(?ThinkCommand, .{ .set = .{ .effort = .on } }), parseThinkCommand(line, &mimo));
    }
    try testing.expectEqual(@as(?ThinkCommand, .{ .set = .{ .effort = .off } }), parseThinkCommand("/think none", &mimo));
    try testing.expectEqual(@as(?model.Effort, .on), acceptedEffort(.high, &mimo));
    try testing.expect(parseThinkCommand("/think off", &.{ .low, .high, .max }).? == .refuse);
    try testing.expect(parseThinkCommand("/think minimal", &.{ .low, .high, .max }).? == .refuse);
    try testing.expect(parseThinkCommand("/think high", &.{ .off, .low, .medium, .xhigh }).? == .refuse);
    try testing.expect(parseThinkCommand("/think minimal", &.{ .off, .low, .medium, .xhigh }).? == .refuse);
    try testing.expectEqual(@as(?model.Effort, null), acceptedEffort(.on, &.{ .off, .low, .medium, .xhigh }));
}
