//! Plan 05 — model discovery (Phase 1 minimal).
//!
//! Walks a directory looking for subdirectories that contain `config.json`,
//! treats each as a discoverable model. Used by `--model-dir` to enumerate
//! candidate models for `/v1/models` listing without loading them.
//!
//! v1 scope: discovery + listing only. Request routing still goes to the
//! single `--model` loaded at startup; if the user passes `--model-dir`
//! WITHOUT `--model`, we pick the first discovered model as the loaded one
//! and surface the rest as `loaded:false` siblings.
//!
//! On-demand load + LRU eviction live in plan 05 phases 2-5 and depend on
//! plan 01 Phase 0 (detangling Transformer state).

const std = @import("std");
const log = @import("log.zig");
// Only the pure JSON contract predicate is referenced — lazy analysis keeps
// dflash.zig's mlx FFI out of this filesystem-only module.
const dflash = @import("dflash.zig");
const mtp = @import("mtp.zig");
const expert_stream = @import("expert_stream.zig");
const expert_quant = @import("expert_quant.zig");
const sushi_exl3 = @import("sushi_exl3");

/// Architecture allow-list for discovery: the four served archs. Discovery
/// silently skips any subdirectory whose `config.json` declares another
/// `model_type`, so `--model-dir` never picks up a partially-downloaded or
/// unsupported checkpoint that would crash the tokenizer load.
const supported_model_types = [_][]const u8{
    "qwen4_exp", // Qwen3.8-Flash-Next
    "qwen4_exp_text",
    "mimo_v2", // MiMo-V2.6-Flash: resident EXL3 packs (text + image), or the original checkpoint streamed.
    "glm5_next", // GLM-5.3-Flash
    "deepseek_v41", // DeepSeek-V4.1-Flash: the original checkpoint, experts streamed.
};

fn isSupportedModelType(model_type: []const u8) bool {
    for (supported_model_types) |t| {
        if (std.mem.eql(u8, model_type, t)) return true;
    }
    return false;
}

/// Quantization modes the MLX loader supports. Must stay in sync with
/// `model.zig:QuantMode` (discovery deliberately avoids importing model.zig,
/// which would drag the mlx FFI into this filesystem-only module).
const supported_quant_modes = [_][]const u8{ "affine", "mxfp4", "mxfp8" };

fn isSupportedQuantMode(mode: []const u8) bool {
    for (supported_quant_modes) |m| {
        if (std.mem.eql(u8, mode, m)) return true;
    }
    return false;
}

/// Outcome of reading a candidate's config.json. Discovery treats any
/// non-`.supported` result as "skip this directory."
const ConfigPeek = union(enum) {
    supported: []const u8, // owned dupe of model_type
    unsupported_arch: []const u8, // owned dupe of model_type
    unsupported_quant: []const u8, // owned dupe of quantization.mode
    /// Declares the DFlash config contract — a spec-decode sidecar whatever
    /// its `model_type` says (DFlash2 ships a bare "qwen3" with no embed
    /// weights; registering it as chat dies at cold load).
    drafter,
    missing_or_unparseable,
};

/// Peek at a candidate's `config.json`: classify by `model_type` and
/// `quantization.mode`. Discovery uses this to filter out:
///   - unsupported archs (which crash the tokenizer)
///   - unsupported quantization modes (anything outside
///     `supported_quant_modes` — affine, mxfp4, mxfp8).
///
/// Returned strings are owned by `allocator`; the caller frees them via
/// the helpers in `freeConfigPeek`.
fn peekConfig(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir, entry_name: []const u8) ConfigPeek {
    var sub = dir.openDir(io, entry_name, .{}) catch return .missing_or_unparseable;
    defer sub.close(io);
    var file = sub.openFile(io, "config.json", .{}) catch return .missing_or_unparseable;
    defer file.close(io);
    var rbuf: [4096]u8 = undefined;
    var rs = file.reader(io, &rbuf);
    const bytes = rs.interface.allocRemaining(allocator, .limited(4 * 1024 * 1024)) catch return .missing_or_unparseable;
    defer allocator.free(bytes);
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, bytes, .{}) catch return .missing_or_unparseable;
    defer parsed.deinit();
    if (parsed.value != .object) return .missing_or_unparseable;
    const root = parsed.value.object;
    // The DFlash contract outranks model_type: a DFlash2 sidecar is
    // config-indistinguishable from its trunk family without this probe (one
    // predicate, shared with the loader's own detection).
    if (dflash.isDflashConfigJson(root)) return .drafter;
    const mt_val = root.get("model_type") orelse return .missing_or_unparseable;
    if (mt_val != .string) return .missing_or_unparseable;
    if (!isSupportedModelType(mt_val.string)) {
        const dup = allocator.dupe(u8, mt_val.string) catch return .missing_or_unparseable;
        return .{ .unsupported_arch = dup };
    }
    // Quantization gate: if a model declares a `quantization.mode`, accept
    // only the schemes the loader supports. Models without a quantization
    // block (bf16 / unquantized) pass through.
    if (root.get("quantization")) |q_val| {
        if (q_val == .object) {
            if (q_val.object.get("mode")) |mode_val| {
                if (mode_val == .string and !isSupportedQuantMode(mode_val.string)) {
                    const dup = allocator.dupe(u8, mode_val.string) catch return .missing_or_unparseable;
                    return .{ .unsupported_quant = dup };
                }
            }
        }
    }
    return .{ .supported = allocator.dupe(u8, mt_val.string) catch return .missing_or_unparseable };
}

/// The set of shard basenames in `model.safetensors.index.json`'s `weight_map`,
/// or null when there is no usable index (single-file packs, media packs).
pub fn indexShardSet(io: std.Io, dir: std.Io.Dir) ?std.StringHashMapUnmanaged(void) {
    const a = std.heap.page_allocator;
    const raw = dir.readFileAlloc(io, "model.safetensors.index.json", a, .limited(16 * 1024 * 1024)) catch return null;
    defer a.free(raw);
    var parsed = std.json.parseFromSlice(std.json.Value, a, raw, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const wm = parsed.value.object.get("weight_map") orelse return null;
    if (wm != .object) return null;
    var set: std.StringHashMapUnmanaged(void) = .empty;
    for (wm.object.values()) |v| {
        if (v != .string) continue;
        if (set.contains(v.string)) continue;
        const key = a.dupe(u8, v.string) catch continue;
        set.put(a, key, {}) catch {
            a.free(key);
            continue;
        };
    }
    // An index none of whose shards exist is stale (the repo was re-sharded
    // after this index was written); the directory is then the set.
    var any_present = false;
    var keys = set.keyIterator();
    while (keys.next()) |k| {
        _ = dir.statFile(io, k.*, .{}) catch continue;
        any_present = true;
        break;
    }
    if (!any_present) {
        log.warn("model.safetensors.index.json names no shard in this directory; loading every *.safetensors instead\n", .{});
        freeShardSet(&set);
        return null;
    }
    return set;
}

/// Free a set from `indexShardSet`.
pub fn freeShardSet(set: *std.StringHashMapUnmanaged(void)) void {
    var keys = set.keyIterator();
    while (keys.next()) |k| std.heap.page_allocator.free(k.*);
    set.deinit(std.heap.page_allocator);
}

/// The routed-expert layout this index declares, or null when it declares
/// neither. A dense (fused) index must also name every PLE shard; the quantized
/// pack keeps its PLE table in `ngram_table.bin` and has none.
pub fn streamingIndexLayout(allocator: std.mem.Allocator, model_type: []const u8, raw: []const u8, layers: u16, ple_shards: u16) ?expert_quant.Layout {
    const first_moe: u16 = if (std.mem.eql(u8, model_type, "mimo_v2")) 1 else 0;
    return streamingIndexLayoutWithFirstMoe(allocator, model_type, raw, layers, first_moe, ple_shards);
}

fn streamingIndexLayoutWithFirstMoe(allocator: std.mem.Allocator, model_type: []const u8, raw: []const u8, layers: u16, first_moe: u16, ple_shards: u16) ?expert_quant.Layout {
    const layout = expert_quant.layoutFromIndexJsonWithFirstMoe(allocator, model_type, raw, layers, first_moe) orelse return null;
    if (layout != .bf16_fused) return layout;
    if (!fusedPleShardsComplete(allocator, raw, ple_shards)) return null;
    return layout;
}

fn fusedPleShardsComplete(allocator: std.mem.Allocator, raw: []const u8, ple_shards: u16) bool {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, raw, .{}) catch return false;
    defer parsed.deinit();
    if (parsed.value != .object) return false;
    const weight_map_value = parsed.value.object.get("weight_map") orelse return false;
    if (weight_map_value != .object) return false;
    const weight_map = weight_map_value.object;
    const seen = allocator.alloc(bool, ple_shards) catch return false;
    defer allocator.free(seen);
    @memset(seen, false);
    const prefix = "model.language_model.layers.";
    const middle = ".ple.ple_embedding.ngram_embedding.shard_";
    const suffix = ".weight";
    var ple_layer: ?u16 = null;
    var found: usize = 0;
    var it = weight_map.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        if (!std.mem.startsWith(u8, key, prefix) or !std.mem.endsWith(u8, key, suffix)) continue;
        const body = key[prefix.len .. key.len - suffix.len];
        const split = std.mem.indexOf(u8, body, middle) orelse continue;
        const layer = std.fmt.parseInt(u16, body[0..split], 10) catch return false;
        const shard = std.fmt.parseInt(u16, body[split + middle.len ..], 10) catch return false;
        if (shard >= ple_shards or seen[shard] or entry.value_ptr.* != .string) return false;
        if (ple_layer) |known| {
            if (known != layer) return false;
        } else ple_layer = layer;
        seen[shard] = true;
        found += 1;
    }
    return found == ple_shards;
}

pub fn qwen4StreamingIndexComplete(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) ?expert_quant.Layout {
    const meta = readStubMeta(io, allocator, model_dir);
    if (!expert_quant.isExpertStreamingArch(meta.modelType())) return null;
    if (!meta.found or meta.num_hidden_layers == 0 or meta.num_experts == 0 or
        meta.hidden_size == 0 or meta.moe_intermediate_size == 0) return null;
    if (meta.num_hidden_layers > std.math.maxInt(u16) or meta.num_experts > std.math.maxInt(u16)) return null;
    const geometry = expert_quant.Geometry{
        .layers = @intCast(meta.num_hidden_layers),
        .experts = @intCast(meta.num_experts),
        .hidden = meta.hidden_size,
        .intermediate = meta.moe_intermediate_size,
        .first_moe_layer = @intCast(meta.first_moe_layer),
    };
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{}) catch return null;
    defer dir.close(io);
    const raw = dir.readFileAlloc(io, "model.safetensors.index.json", allocator, .limited(64 * 1024 * 1024)) catch return null;
    defer allocator.free(raw);
    const layout = streamingIndexLayoutWithFirstMoe(allocator, meta.modelType(), raw, geometry.layers, geometry.first_moe_layer, 128) orelse return null;
    var shards = indexShardSet(io, dir) orelse return null;
    defer freeShardSet(&shards);
    var keys = shards.keyIterator();
    while (keys.next()) |name| {
        const stat = dir.statFile(io, name.*, .{}) catch return null;
        if (stat.kind != .file) return null;
    }
    switch (layout) {
        .bf16_fused => {
            var experts = expert_stream.ExpertStore.open(allocator, model_dir, .{
                .layers = geometry.layers,
                .experts = geometry.experts,
                .hidden = geometry.hidden,
                .intermediate = geometry.intermediate,
            }) catch return null;
            experts.deinit();
            var table = expert_stream.Bf16NgramStore.open(allocator, model_dir) catch return null;
            defer table.deinit();
            if (table.dim != 160 or table.rows != 320_001_536) return null;
        },
        .quantized_split => {
            const stat = dir.statFile(io, "ngram_table.bin", .{}) catch return null;
            if (stat.kind != .file) return null;
            var experts = expert_quant.QuantStore.open(allocator, model_dir, geometry) catch return null;
            experts.deinit();
        },
        .bf16_individual, .fp8_individual, .mxfp4_split, .mxfp4_individual, .mxfp4_w123 => {
            var experts = expert_quant.QuantStore.openForLayout(allocator, model_dir, geometry, layout) catch return null;
            experts.deinit();
        },
        .exl3_k4 => {
            const stat = dir.statFile(io, "ngram_table.bin", .{}) catch return null;
            if (stat.kind != .file) return null;
        },
    }
    return layout;
}

/// True if `path` names a `.gguf` file or a directory holding one. The engine
/// refuses these by name (`ModelFormatUnsupported`) instead of reading a
/// config.json that is not there.
///
/// Empty / non-absolute paths (e.g. headless boot with no --model) return
/// false — guarded BEFORE `openDirAbsolute`, which ASSERTS the path is
/// absolute (`unreachable` on "") and in ReleaseFast that's UB that
/// miscompiles the caller (see the openDirAbsolute rule in docs/engine-mlx-gotchas.md).
pub fn isGgufModelPath(io: std.Io, path: []const u8) bool {
    if (path.len == 0 or !std.fs.path.isAbsolute(path)) return false;
    if (std.mem.endsWith(u8, path, ".gguf")) return true;
    var dir = std.Io.Dir.openDirAbsolute(io, path, .{ .iterate = true }) catch return false;
    defer dir.close(io);
    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        if (!std.mem.endsWith(u8, entry.name, ".gguf")) continue;
        const st = dir.statFile(io, entry.name, .{}) catch continue;
        if (st.kind == .file) return true;
    }
    return false;
}

/// Coarse classification of a model for UX surfaces — the `sushi list`
/// TYPE column and the `run` chat-REPL preflight. Mirrors how serving
/// actually routes the directory (drafter sidecars).
pub const ModelKind = enum {
    chat,
    drafter,
    unsupported,

    /// Short label for the `list` TYPE column.
    pub fn label(self: ModelKind) []const u8 {
        return switch (self) {
            .chat => "chat",
            .drafter => "drafter",
            .unsupported => "unsupported",
        };
    }

    /// Human phrase for refusal messages ("'X' is <describe>").
    pub fn describe(self: ModelKind) []const u8 {
        return switch (self) {
            .chat => "a chat model",
            .drafter => "a speculative-decoding drafter sidecar, not a standalone model (load it via --drafter beside its Qwen3.8-Flash-Next or GLM-5.3-Flash target)",
            .unsupported => "an architecture sushi does not support",
        };
    }

};

/// Map a config.json `model_type` to its ModelKind.
pub fn modelKindFromType(model_type: []const u8) ModelKind {
    if (std.mem.endsWith(u8, model_type, "_assistant")) return .drafter;
    if (isSupportedModelType(model_type)) return .chat;
    return .unsupported;
}

/// Classify an ABSOLUTE model dir by config.json's model_type; null when the
/// dir is not model-shaped at all.
pub fn classifyModelPath(io: std.Io, allocator: std.mem.Allocator, abs_path: []const u8) ?ModelKind {
    if (abs_path.len == 0 or !std.fs.path.isAbsolute(abs_path)) return null;
    const trimmed = trimTrailingSlash(abs_path);
    const base = std.fs.path.basename(trimmed);
    const parent = std.fs.path.dirname(trimmed) orelse return null;
    if (base.len == 0 or parent.len == 0) return null;
    var dir = std.Io.Dir.openDirAbsolute(io, parent, .{}) catch return null;
    defer dir.close(io);
    return switch (peekConfig(io, allocator, dir, base)) {
        .missing_or_unparseable => null,
        // Raw model_type still classifies the KIND even when serving would
        // skip it (drafters, vit, ...) — that's exactly what the label is for.
        .unsupported_arch => |mt| blk: {
            defer allocator.free(mt);
            break :blk modelKindFromType(mt);
        },
        .unsupported_quant => |mode| blk: {
            allocator.free(mode);
            break :blk .unsupported;
        },
        .drafter => .drafter,
        .supported => |mt| blk: {
            defer allocator.free(mt);
            break :blk modelKindFromType(mt);
        },
    };
}

pub const DiscoveredModel = struct {
    /// Model id (subdirectory basename, e.g. "Qwen3.8-Flash-Next-4bit").
    id: []const u8,
    /// Absolute path to the model directory.
    path: []const u8,
    /// Approximate weight size on disk in bytes (sum of *.safetensors). Used
    /// later by eviction; null if scan failed.
    bytes_on_disk: ?u64,
    /// `model_type` peeked from config.json (e.g. "qwen4_exp"), so registry stubs
    /// can advertise arch-derived capabilities before a cold load. Empty
    /// when unknown.
    model_type: []const u8 = "",
    streaming_index_complete: bool = false,
};

pub const DiscoveryResult = struct {
    models: []DiscoveredModel,
    allocator: std.mem.Allocator,
    /// The roots this result was scanned from (owned dupes; set by
    /// `discoverModelsMany`). Kept so `ModelRegistry.rescan` can re-walk
    /// them at runtime — models downloaded after boot are invisible to a
    /// boot-only scan.
    roots: []const []const u8 = &.{},

    pub fn deinit(self: *DiscoveryResult) void {
        for (self.models) |*m| {
            self.allocator.free(m.id);
            self.allocator.free(m.path);
            if (m.model_type.len > 0) self.allocator.free(m.model_type);
        }
        self.allocator.free(self.models);
        if (self.roots.len > 0) {
            for (self.roots) |r| self.allocator.free(r);
            self.allocator.free(self.roots);
        }
    }
};

/// Scan `model_dir` for subdirectories containing `config.json`.
/// Returns DiscoveryResult; caller owns memory via deinit().
/// Symlinks followed; permission errors on individual subdirs skipped silently.
pub fn discoverModels(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !DiscoveryResult {
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true }) catch |err| {
        return err;
    };
    defer dir.close(io);
    return discoverModelsInDir(io, allocator, dir, model_dir);
}

/// Scan several roots and merge them into ONE result, first-root-wins on a
/// repeated id.
///
/// De-dup is load-bearing, not tidiness: `registerStubWithArch` answers
/// `error.DuplicateId` and `ModelRegistry.registerDiscovered` does `try`, so
/// two roots holding the same `org/name` would fail the whole registry init —
/// the server would not start. First wins because the caller orders the roots
/// and the first is where downloads land, so a stale copy on a second disk must
/// never shadow the live one.
///
/// A root that cannot be opened is SKIPPED with a warning rather than failing:
/// the second folder can live on an external drive, and unplugging it must not
/// stop the server serving everything else.
pub fn discoverModelsMany(io: std.Io, allocator: std.mem.Allocator, roots: []const []const u8) !DiscoveryResult {
    // Dupe the roots first so the result can carry them (see
    // `DiscoveryResult.roots`).
    var roots_owned: []const []const u8 = &.{};
    if (roots.len > 0) {
        const dupes = try allocator.alloc([]const u8, roots.len);
        var done: usize = 0;
        errdefer {
            for (dupes[0..done]) |r| allocator.free(r);
            allocator.free(dupes);
        }
        for (roots) |r| {
            dupes[done] = try allocator.dupe(u8, r);
            done += 1;
        }
        roots_owned = dupes;
    }
    errdefer if (roots_owned.len > 0) {
        for (roots_owned) |r| allocator.free(r);
        allocator.free(roots_owned);
    };

    var merged = std.ArrayList(DiscoveredModel).empty;
    errdefer {
        for (merged.items) |*m| {
            allocator.free(m.id);
            allocator.free(m.path);
            if (m.model_type.len > 0) allocator.free(m.model_type);
        }
        merged.deinit(allocator);
    }

    for (roots) |root| {
        const one = discoverModels(io, allocator, root) catch |err| {
            log.warn("--model-dir scan failed ({s}): {s}\n", .{ root, @errorName(err) });
            continue;
        };
        // Transfer ownership per model, freeing only the ones we drop — the
        // whole result's `deinit` would free the strings we just handed over.
        defer allocator.free(one.models);
        for (one.models) |m| {
            var dup = false;
            for (merged.items) |seen| {
                if (std.mem.eql(u8, seen.id, m.id)) {
                    dup = true;
                    break;
                }
            }
            if (dup) {
                log.warn("[discovery] {s}: already found in an earlier --model-dir, skipping {s}\n", .{ m.id, m.path });
                allocator.free(m.id);
                allocator.free(m.path);
                if (m.model_type.len > 0) allocator.free(m.model_type);
                continue;
            }
            try merged.append(allocator, m);
        }
    }

    return .{ .models = try merged.toOwnedSlice(allocator), .allocator = allocator, .roots = roots_owned };
}

/// Core scan over an already-open root. Two layouts are recognized:
///   <root>/<model>/config.json               → id "<model>"
///   <root>/<org>/<model>/config.json         → id "<org>/<model>"
/// The second is the HF-style layout `~/.sushi/models` uses (the app's
/// DownloadManager and `sushi pull` both write there).
pub fn discoverModelsInDir(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir, model_dir: []const u8) !DiscoveryResult {
    var found = std.ArrayList(DiscoveredModel).empty;
    errdefer {
        for (found.items) |*m| {
            allocator.free(m.id);
            allocator.free(m.path);
            if (m.model_type.len > 0) allocator.free(m.model_type);
        }
        found.deinit(allocator);
    }

    var iter = dir.iterate();
    while (try iter.next(io)) |entry| {
        if (entry.kind != .directory and entry.kind != .sym_link) continue;
        if (entry.name.len == 0 or entry.name[0] == '.') continue;

        const is_model_dir = try tryAddModel(io, allocator, dir, entry.name, "", model_dir, &found);
        if (is_model_dir) continue;

        // No config.json at this level — maybe an org dir (org/repo layout).
        var org = dir.openDir(io, entry.name, .{ .iterate = true }) catch continue;
        defer org.close(io);
        var org_iter = org.iterate();
        while (org_iter.next(io) catch null) |org_entry| {
            if (org_entry.kind != .directory and org_entry.kind != .sym_link) continue;
            if (org_entry.name.len == 0 or org_entry.name[0] == '.') continue;
            _ = try tryAddModel(io, allocator, org, org_entry.name, entry.name, model_dir, &found);
        }
    }

    // Stable order: by id ascending, so listing is deterministic.
    std.sort.pdq(DiscoveredModel, found.items, {}, lessThanById);

    return .{
        .models = try found.toOwnedSlice(allocator),
        .allocator = allocator,
    };
}

/// Inspect `parent/<name>` as a model-dir candidate; append to `found` if
/// it holds a supported config.json. Returns true when the dir was
/// model-shaped (config.json present — even if unsupported), so callers know
/// not to descend into it looking for an org layout.
fn tryAddModel(
    io: std.Io,
    allocator: std.mem.Allocator,
    parent: std.Io.Dir,
    name: []const u8,
    id_prefix: []const u8,
    model_dir: []const u8,
    found: *std.ArrayList(DiscoveredModel),
) !bool {
    var sub = parent.openDir(io, name, .{ .iterate = true }) catch return false;
    defer sub.close(io);

    var bytes: u64 = 0;
    var bytes_ok = false;

    const model_type: []const u8 = blk: {
        const has_config = if (sub.statFile(io, "config.json", .{})) |st| st.kind == .file else |_| false;
        if (!has_config) return false;

        // Filter by supported model_type AND quantization scheme. Catches:
        //   - partially-downloaded checkpoints (missing/garbage config)
        //   - unsupported arches
        //   - unsupported quants (modes outside supported_quant_modes)
        // before they reach the tokenizer/weight loaders.
        break :blk switch (peekConfig(io, allocator, parent, name)) {
            .missing_or_unparseable => {
                log.info("[discovery] skip {s}: config.json missing or unparseable", .{name});
                return true;
            },
            .unsupported_arch => |mt| {
                defer allocator.free(mt);
                log.info("[discovery] skip {s}: unsupported model_type '{s}'", .{ name, mt });
                return true;
            },
            .unsupported_quant => |mode| {
                defer allocator.free(mode);
                log.info("[discovery] skip {s}: unsupported quantization mode '{s}' (supported: affine, mxfp4, mxfp8)", .{ name, mode });
                return true;
            },
            .drafter => {
                log.info("[discovery] skip {s}: DFlash drafter sidecar, not a standalone model", .{name});
                return true;
            },
            .supported => |mt| mt, // ownership moves to the DiscoveredModel
        };
    };
    errdefer if (model_type.len > 0) allocator.free(model_type);

    // Compute weight bytes (sum of *.safetensors sizes) — best-effort.
    if (!bytes_ok) {
        var sub_iter_dir = parent.openDir(io, name, .{ .iterate = true }) catch null;
        if (sub_iter_dir) |*sd| {
            defer sd.close(io);
            var referenced = indexShardSet(io, sd.*);
            defer if (referenced) |*r| freeShardSet(r);
            var sd_iter = sd.iterate();
            while (sd_iter.next(io) catch null) |sub_entry| {
                if (sub_entry.kind != .file and sub_entry.kind != .sym_link) continue;
                if (!std.mem.endsWith(u8, sub_entry.name, ".safetensors")) continue;
                if (referenced) |r| if (!r.contains(sub_entry.name)) continue;
                const st = sd.statFile(io, sub_entry.name, .{}) catch continue;
                if (st.kind != .file) continue;
                bytes += @intCast(st.size);
                bytes_ok = true;
            }
        }
    }

    const id = if (id_prefix.len > 0)
        try std.fmt.allocPrint(allocator, "{s}/{s}", .{ id_prefix, name })
    else
        try allocator.dupe(u8, name);
    errdefer allocator.free(id);
    const path = if (id_prefix.len > 0)
        try std.fmt.allocPrint(allocator, "{s}/{s}/{s}", .{ trimTrailingSlash(model_dir), id_prefix, name })
    else
        try std.fmt.allocPrint(allocator, "{s}/{s}", .{ trimTrailingSlash(model_dir), name });
    const streaming_index_complete = expert_quant.isExpertStreamingArch(model_type) and
        qwen4StreamingIndexComplete(io, allocator, path) != null;
    try found.append(allocator, .{
        .id = id,
        .path = path,
        .bytes_on_disk = if (bytes_ok) bytes else null,
        .model_type = model_type,
        .streaming_index_complete = streaming_index_complete,
    });
    return true;
}

fn trimTrailingSlash(s: []const u8) []const u8 {
    var p = s;
    while (p.len > 0 and p[p.len - 1] == '/') p = p[0 .. p.len - 1];
    return p;
}

pub const ProbeResult = struct {
    /// Owned dupe of the supported model_type.
    model_type: []const u8,
    /// Sum of *.safetensors bytes; null if the scan failed.
    bytes_on_disk: ?u64,
};

/// Validate an arbitrary absolute model directory the way discovery would
/// (config.json present, supported model_type and quant mode) and report its
/// weight bytes. Used by /v1/load-model's register-by-path branch for models
/// OUTSIDE the --model-dir scan — e.g. a pack in the app's download root,
/// whichever org dir the chat model came from.
pub fn probeModelDir(io: std.Io, allocator: std.mem.Allocator, abs_path: []const u8) !ProbeResult {
    const trimmed = trimTrailingSlash(abs_path);
    const base = std.fs.path.basename(trimmed);
    const parent = std.fs.path.dirname(trimmed) orelse return error.InvalidModelPath;
    if (base.len == 0 or parent.len == 0) return error.InvalidModelPath;

    var dir = std.Io.Dir.openDirAbsolute(io, parent, .{}) catch return error.ModelDirNotFound;
    defer dir.close(io);

    const model_type: []const u8 = switch (peekConfig(io, allocator, dir, base)) {
        .missing_or_unparseable => return error.ModelDirNotFound,
        .unsupported_arch => |mt| {
            allocator.free(mt);
            return error.UnsupportedArch;
        },
        .unsupported_quant => |mode| {
            allocator.free(mode);
            return error.UnsupportedQuantMode;
        },
        .drafter => return error.UnsupportedArch,
        .supported => |mt| mt,
    };
    errdefer allocator.free(model_type);

    var bytes: u64 = 0;
    var bytes_ok = false;
    var sub = dir.openDir(io, base, .{ .iterate = true }) catch null;
    if (sub) |*sd| {
        defer sd.close(io);
        var it = sd.iterate();
        while (it.next(io) catch null) |entry| {
            if (entry.kind != .file and entry.kind != .sym_link) continue;
            if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
            const st = sd.statFile(io, entry.name, .{}) catch continue;
            if (st.kind != .file) continue;
            bytes += @intCast(st.size);
            bytes_ok = true;
        }
    }
    return .{ .model_type = model_type, .bytes_on_disk = if (bytes_ok) bytes else null };
}

/// Metadata an `unloaded` stub can advertise via `/v1/models` without faulting
/// in weights — all sourced from `config.json` (+ chat-template presence). Lets
/// clients see context window, dims, MoE-ness, and capabilities (tools/vision)
/// before a cold load. `found=false` means config.json couldn't be read/parsed.
pub const StubMeta = struct {
    found: bool = false,
    vocab_size: u32 = 0,
    hidden_size: u32 = 0,
    num_hidden_layers: u32 = 0,
    max_position_embeddings: u32 = 0,
    quant_bits: u32 = 0,
    expert_quant_rate: ?expert_quant.expert_exl3.Rate = null,
    /// The pack carries an `expert_quant` block: its routed experts are already
    /// packed, at widths this config does not state one number for.
    quantized_experts: bool = false,
    is_moe: bool = false,
    /// The dir ships an MTP head (sidecar or in-checkpoint) the server can load.
    has_mtp: bool = false,
    num_experts: u32 = 0,
    num_experts_per_tok: u32 = 0,
    moe_intermediate_size: u32 = 0,
    first_moe_layer: u32 = 0,
    has_vision: bool = false,
    /// Qwen3-VL-family video input: a `video_token_id` alongside `has_vision`
    /// (video piggybacks the vision tower — see src/qwen_vision.zig).
    has_video: bool = false,
    has_chat: bool = false,
    /// generation_config.json's thinking default (`ModelConfig.gen_enable_thinking`).
    enable_thinking: ?bool = null,
    model_type_buf: [64]u8 = @splat(0),
    model_type_len: u8 = 0,

    pub fn modelType(self: *const StubMeta) []const u8 {
        return self.model_type_buf[0..self.model_type_len];
    }
};

fn jsonU32(obj: std.json.ObjectMap, key: []const u8) u32 {
    if (obj.get(key)) |v| {
        if (v == .integer and v.integer > 0) return std.math.cast(u32, v.integer) orelse 0;
    }
    return 0;
}

/// Pure: extract `StubMeta` from raw config.json bytes. `has_chat_template` is
/// supplied by the caller (the template lives outside config.json), and
/// determines chat/tool capabilities. Mirrors the loaded-path capability rules in `models_api.renderModelEntry`
/// (chat-template presence ⇒ chat/tool_use/streaming/json_schema). Returns
/// `.found=false` on any parse failure.
pub fn parseStubMeta(allocator: std.mem.Allocator, config_json: []const u8, has_chat_template: bool) StubMeta {
    var meta: StubMeta = .{};
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, config_json, .{}) catch return meta;
    defer parsed.deinit();
    if (parsed.value != .object) return meta;
    const root = parsed.value.object;
    meta.found = true;

    // A multimodal checkpoint puts EVERY text dim under `text_config` and
    // leaves only model_type / vision_config / quantization at the root. Read
    // the nested block first and fall back to the root per field (a minimal
    // `text_config` may omit fields the root still carries) — the same merge
    // `model.zig parseConfigFromJson` does. Reading the root alone made
    // /v1/models report hidden=0 / layers=0 / ctx=0 / is_moe=false for every
    // unloaded model with a nested `text_config`.
    const text_cfg: ?std.json.ObjectMap = if (root.get("text_config")) |tc|
        (if (tc == .object) tc.object else null)
    else
        null;
    const cfgU32 = struct {
        fn get(r: std.json.ObjectMap, tc: ?std.json.ObjectMap, key: []const u8) u32 {
            if (tc) |t| {
                const v = jsonU32(t, key);
                if (v > 0) return v;
            }
            return jsonU32(r, key);
        }
    }.get;

    meta.vocab_size = cfgU32(root, text_cfg, "vocab_size");
    meta.hidden_size = cfgU32(root, text_cfg, "hidden_size");
    meta.num_hidden_layers = cfgU32(root, text_cfg, "num_hidden_layers");
    meta.max_position_embeddings = cfgU32(root, text_cfg, "max_position_embeddings");
    if (root.get("quantization")) |q| {
        if (q == .object) meta.quant_bits = jsonU32(q.object, "bits");
    }
    if (root.get("expert_quant")) |q| meta.quantized_experts = q == .object;
    if (meta.quantized_experts) meta.expert_quant_rate = if (sushi_exl3.parseExpertQuant(root)) |spec| spec.rate else |_| null;
    meta.is_moe = cfgU32(root, text_cfg, "num_experts") > 0 or
        cfgU32(root, text_cfg, "num_local_experts") > 0 or
        cfgU32(root, text_cfg, "n_routed_experts") > 0;
    meta.num_experts = cfgU32(root, text_cfg, "num_experts");
    if (meta.num_experts == 0) meta.num_experts = cfgU32(root, text_cfg, "n_routed_experts");
    if (meta.num_experts == 0) meta.num_experts = cfgU32(root, text_cfg, "num_local_experts");
    meta.num_experts_per_tok = cfgU32(root, text_cfg, "num_experts_per_tok");
    meta.moe_intermediate_size = cfgU32(root, text_cfg, "moe_intermediate_size");
    const mt: []const u8 = if (root.get("model_type")) |v|
        (if (v == .string) v.string else "")
    else
        "";
    if (mt.len <= meta.model_type_buf.len) {
        @memcpy(meta.model_type_buf[0..mt.len], mt);
        meta.model_type_len = @intCast(mt.len);
    }
    if (std.mem.eql(u8, mt, "mimo_v2")) {
        const cfg = text_cfg orelse root;
        const freq = cfg.get("moe_layer_freq") orelse return .{};
        meta.first_moe_layer = denseMoePrefix(freq, meta.num_hidden_layers) catch return .{};
    }
    // Vision: a `vision_config` block on a non-`_text` arch (the `_text` guard
    // skips text-only quantized checkpoints with a vestigial block).
    meta.has_vision = root.get("vision_config") != null and !std.mem.endsWith(u8, mt, "_text");
    // MiMo serves images only; its video pads are not wired.
    meta.has_video = meta.has_vision and cfgU32(root, text_cfg, "video_token_id") > 0 and
        !std.mem.eql(u8, mt, "mimo_v2");
    meta.has_chat = has_chat_template;
    return meta;
}

/// The dense-prefix representation cannot describe interspersed dense/MoE layers.
pub fn denseMoePrefix(freq: std.json.Value, layers: u32) !u32 {
    if (freq != .array or freq.array.items.len != layers) return error.UnsupportedMimoV2Config;
    var dense: u32 = 0;
    var seen_moe = false;
    for (freq.array.items) |v| {
        if (v != .integer or (v.integer != 0 and v.integer != 1)) return error.UnsupportedMimoV2Config;
        if (v.integer == 1) {
            seen_moe = true;
        } else {
            if (seen_moe) return error.UnsupportedMimoV2Config;
            dense += 1;
        }
    }
    return dense;
}

/// Read `StubMeta` for the model directory at `abs_path` (config.json + a
/// chat-template-presence check). Best-effort: any I/O / parse failure yields
/// `.found=false`. Called per unloaded entry from `/v1/models` — cheap (small
/// JSON files) and that endpoint isn't hot.
pub fn readStubMeta(io: std.Io, allocator: std.mem.Allocator, abs_path: []const u8) StubMeta {
    const trimmed = trimTrailingSlash(abs_path);
    const base = std.fs.path.basename(trimmed);
    const parent = std.fs.path.dirname(trimmed) orelse return .{};
    if (base.len == 0 or parent.len == 0) return .{};
    var parent_dir = std.Io.Dir.openDirAbsolute(io, parent, .{}) catch return .{};
    defer parent_dir.close(io);
    var dir = parent_dir.openDir(io, base, .{}) catch return .{};
    defer dir.close(io);

    var file = dir.openFile(io, "config.json", .{}) catch return .{};
    defer file.close(io);
    var rbuf: [4096]u8 = undefined;
    var rs = file.reader(io, &rbuf);
    const bytes = rs.interface.allocRemaining(allocator, .limited(4 * 1024 * 1024)) catch return .{};
    defer allocator.free(bytes);

    var meta = parseStubMeta(allocator, bytes, hasChatTemplate(io, allocator, dir));
    meta.has_mtp = mtp.dirAdvertisesMtp(io, allocator, dir);
    if (dir.openFile(io, "generation_config.json", .{})) |gen| {
        defer gen.close(io);
        var gbuf: [4096]u8 = undefined;
        var gr = gen.reader(io, &gbuf);
        if (gr.interface.allocRemaining(allocator, .limited(1024 * 1024))) |gen_bytes| {
            defer allocator.free(gen_bytes);
            meta.enable_thinking = @import("model.zig").parseGenerationDefaultsFromJson(gen_bytes).enable_thinking;
        } else |_| {}
    } else |_| {}
    return meta;
}

/// True if the model dir ships a chat template — a `chat_template.jinja` file,
/// or a `tokenizer_config.json` that carries a `chat_template` key. Cheap proxy
/// for "this is an instruct/chat model" used to gate chat/tool capabilities on
/// unloaded stubs.
fn hasChatTemplate(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir) bool {
    if (dir.statFile(io, "chat_template.jinja", .{})) |st| {
        if (st.kind == .file) return true;
    } else |_| {}
    var f = dir.openFile(io, "tokenizer_config.json", .{}) catch return false;
    defer f.close(io);
    var rbuf: [4096]u8 = undefined;
    var rs = f.reader(io, &rbuf);
    const bytes = rs.interface.allocRemaining(allocator, .limited(8 * 1024 * 1024)) catch return false;
    defer allocator.free(bytes);
    return std.mem.indexOf(u8, bytes, "\"chat_template\"") != null;
}

fn lessThanById(_: void, a: DiscoveredModel, b: DiscoveredModel) bool {
    return std.mem.lessThan(u8, a.id, b.id);
}

// ── Tests ──

const testing = std.testing;

test "discoverModels finds flat and org/repo model dirs" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    try tmp.dir.createDirPath(io, "flat-model");
    try tmp.dir.writeFile(io, .{ .sub_path = "flat-model/config.json", .data = "{\"model_type\":\"mimo_v2\"}" });
    // HF-style org/repo layout — the DownloadManager / `sushi pull`
    // convention. Must be discovered with id "org/name".
    try tmp.dir.createDirPath(io, "mlx-community/nested-model");
    try tmp.dir.writeFile(io, .{ .sub_path = "mlx-community/nested-model/config.json", .data = "{\"model_type\":\"glm5_next\"}" });
    // Junk that must not surface: an org dir with a non-model child, and a
    // dot-dir.
    try tmp.dir.createDirPath(io, "empty-org/not-a-model");
    try tmp.dir.createDirPath(io, ".hidden/whatever");

    var result = try discoverModelsInDir(io, allocator, tmp.dir, "/models-root");
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 2), result.models.len);
    try std.testing.expectEqualStrings("flat-model", result.models[0].id);
    try std.testing.expectEqualStrings("/models-root/flat-model", result.models[0].path);
    try std.testing.expectEqualStrings("mimo_v2", result.models[0].model_type);
    try std.testing.expectEqualStrings("mlx-community/nested-model", result.models[1].id);
    try std.testing.expectEqualStrings("/models-root/mlx-community/nested-model", result.models[1].path);
    try std.testing.expectEqualStrings("glm5_next", result.models[1].model_type);
}

fn writeStreamingQuantFixture(io: std.Io, dir: std.Io.Dir, model_type: []const u8) !void {
    const allocator = std.testing.allocator;
    const experts: u64 = 2;
    const hidden: u64 = 64;
    const intermediate: u64 = 32;
    const Spec = struct { key: []const u8, dtype: []const u8, d1: u64, d2: u64, elem: u64 };
    const specs = [_]Spec{
        .{ .key = "gate_proj.weight", .dtype = "U32", .d1 = intermediate, .d2 = 8, .elem = 4 },
        .{ .key = "gate_proj.scales", .dtype = "BF16", .d1 = intermediate, .d2 = 2, .elem = 2 },
        .{ .key = "gate_proj.biases", .dtype = "BF16", .d1 = intermediate, .d2 = 2, .elem = 2 },
        .{ .key = "up_proj.weight", .dtype = "U32", .d1 = intermediate, .d2 = 8, .elem = 4 },
        .{ .key = "up_proj.scales", .dtype = "BF16", .d1 = intermediate, .d2 = 2, .elem = 2 },
        .{ .key = "up_proj.biases", .dtype = "BF16", .d1 = intermediate, .d2 = 2, .elem = 2 },
        .{ .key = "down_proj.weight", .dtype = "U32", .d1 = hidden, .d2 = 4, .elem = 4 },
        .{ .key = "down_proj.scales", .dtype = "BF16", .d1 = hidden, .d2 = 1, .elem = 2 },
        .{ .key = "down_proj.biases", .dtype = "BF16", .d1 = hidden, .d2 = 1, .elem = 2 },
    };

    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(allocator);
    var index: std.ArrayList(u8) = .empty;
    defer index.deinit(allocator);
    try header.append(allocator, '{');
    try index.appendSlice(allocator, "{\"weight_map\":{");
    var at: u64 = 0;
    for (specs, 0..) |spec, i| {
        const bytes = experts * spec.d1 * spec.d2 * spec.elem;
        const row = try std.fmt.allocPrint(
            allocator,
            "{s}\"language_model.model.layers.0.mlp.switch_mlp.{s}\":{{\"dtype\":\"{s}\",\"shape\":[{d},{d},{d}],\"data_offsets\":[{d},{d}]}}",
            .{ if (i == 0) "" else ",", spec.key, spec.dtype, experts, spec.d1, spec.d2, at, at + bytes },
        );
        defer allocator.free(row);
        try header.appendSlice(allocator, row);
        const mapping = try std.fmt.allocPrint(
            allocator,
            "{s}\"language_model.model.layers.0.mlp.switch_mlp.{s}\":\"model.safetensors\"",
            .{ if (i == 0) "" else ",", spec.key },
        );
        defer allocator.free(mapping);
        try index.appendSlice(allocator, mapping);
        at += bytes;
    }
    try header.append(allocator, '}');
    try index.appendSlice(allocator, "}}");

    const file_bytes = try allocator.alloc(u8, 8 + header.items.len + @as(usize, @intCast(at)));
    defer allocator.free(file_bytes);
    std.mem.writeInt(u64, file_bytes[0..8], header.items.len, .little);
    @memcpy(file_bytes[8 .. 8 + header.items.len], header.items);
    @memset(file_bytes[8 + header.items.len ..], 0);
    try dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = file_bytes });
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
    try dir.writeFile(io, .{ .sub_path = "ngram_table.bin", .data = "0" });
    const config = try std.fmt.allocPrint(
        allocator,
        "{{\"model_type\":\"{s}\",\"hidden_size\":{d},\"num_hidden_layers\":1,\"num_experts\":{d},\"num_experts_per_tok\":2,\"moe_intermediate_size\":{d}}}",
        .{ model_type, hidden, experts, intermediate },
    );
    defer allocator.free(config);
    try dir.writeFile(io, .{ .sub_path = "config.json", .data = config });
}

test "expert streaming is qwen4_exp only: the same switch_mlp pack is no candidate under another model_type" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var root = std.testing.tmpDir(.{ .iterate = true });
    defer root.cleanup();
    try root.dir.createDirPath(io, "q4");
    try root.dir.createDirPath(io, "moe");
    var q4_dir = try root.dir.openDir(io, "q4", .{ .iterate = true });
    defer q4_dir.close(io);
    var moe_dir = try root.dir.openDir(io, "moe", .{ .iterate = true });
    defer moe_dir.close(io);
    try writeStreamingQuantFixture(io, q4_dir, "qwen4_exp");
    try writeStreamingQuantFixture(io, moe_dir, "not_qwen4");

    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try root.dir.realPath(io, &root_buf);
    const root_path = root_buf[0..root_len];
    var q4_buf: [std.fs.max_path_bytes]u8 = undefined;
    const q4_len = try q4_dir.realPath(io, &q4_buf);
    const q4_path = q4_buf[0..q4_len];
    var moe_buf: [std.fs.max_path_bytes]u8 = undefined;
    const moe_len = try moe_dir.realPath(io, &moe_buf);
    const moe_path = moe_buf[0..moe_len];

    try std.testing.expectEqual(
        expert_quant.Layout.quantized_split,
        expert_quant.layoutOfDir(allocator, io, "qwen4_exp", q4_path, 1).?,
    );
    try std.testing.expect(expert_quant.layoutOfDir(allocator, io, "not_qwen4", moe_path, 1) == null);

    try std.testing.expectEqual(
        expert_quant.Layout.quantized_split,
        qwen4StreamingIndexComplete(io, allocator, q4_path).?,
    );
    try std.testing.expect(qwen4StreamingIndexComplete(io, allocator, moe_path) == null);

    var result = try discoverModels(io, allocator, root_path);
    defer result.deinit();
    // The foil's type is no served arch, so discovery lists only the qwen4 pack.
    try std.testing.expectEqual(@as(usize, 1), result.models.len);
    try std.testing.expectEqualStrings("qwen4_exp", result.models[0].model_type);
    try std.testing.expect(result.models[0].streaming_index_complete);
}

test "qwen4 streaming index completeness requires every target bank and PLE shard" {
    const complete = "{\"weight_map\":{\"model.language_model.layers.0.mlp.experts.gate_up_proj\":\"a\",\"model.language_model.layers.0.mlp.experts.down_proj\":\"b\",\"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight\":\"c\",\"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_1.weight\":\"d\"}}";
    const missing = "{\"weight_map\":{\"model.language_model.layers.0.mlp.experts.gate_up_proj\":\"a\",\"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight\":\"c\"}}";
    try std.testing.expectEqual(expert_quant.Layout.bf16_fused, streamingIndexLayout(std.testing.allocator, "qwen4_exp", complete, 1, 2).?);
    try std.testing.expect(streamingIndexLayout(std.testing.allocator, "qwen4_exp", missing, 1, 2) == null);
}

test "a quantized index is a streaming candidate without any PLE shard" {
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(std.testing.allocator);
    try buf.appendSlice(std.testing.allocator, "{\"weight_map\":{");
    var first = true;
    for ([_][]const u8{ "gate", "up", "down" }) |proj| {
        for ([_][]const u8{ "weight", "scales", "biases" }) |part| {
            if (!first) try buf.append(std.testing.allocator, ',');
            first = false;
            const row = try std.fmt.allocPrint(std.testing.allocator, "\"language_model.model.layers.0.mlp.switch_mlp.{s}_proj.{s}\":\"a\"", .{ proj, part });
            defer std.testing.allocator.free(row);
            try buf.appendSlice(std.testing.allocator, row);
        }
    }
    try buf.appendSlice(std.testing.allocator, "}}");
    try std.testing.expectEqual(expert_quant.Layout.quantized_split, streamingIndexLayout(std.testing.allocator, "qwen4_exp", buf.items, 1, 128).?);
    try std.testing.expect(streamingIndexLayout(std.testing.allocator, "qwen4_exp", buf.items, 2, 128) == null);
}

test "the real quantized pack is a complete streaming discovery candidate" {
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try @import("test_models.zig").packPath(&path_buf, "Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit");
    var dir = std.Io.Dir.openDirAbsolute(std.testing.io, path, .{}) catch return error.SkipZigTest;
    dir.close(std.testing.io);
    try std.testing.expectEqual(expert_quant.Layout.quantized_split, qwen4StreamingIndexComplete(std.testing.io, std.testing.allocator, path).?);
    const meta = readStubMeta(std.testing.io, std.testing.allocator, path);
    try std.testing.expect(meta.found and meta.quant_bits == 4 and meta.num_experts == 512);
}

test "real qwen checkpoint is a complete streaming discovery candidate" {
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try @import("test_models.zig").packPath(&path_buf, "Qwen/Qwen3.8-Flash-Next");
    var dir = std.Io.Dir.openDirAbsolute(std.testing.io, path, .{}) catch return error.SkipZigTest;
    dir.close(std.testing.io);
    try std.testing.expectEqual(expert_quant.Layout.bf16_fused, qwen4StreamingIndexComplete(std.testing.io, std.testing.allocator, path).?);
    const meta = readStubMeta(std.testing.io, std.testing.allocator, path);
    try std.testing.expect(meta.found and meta.quant_bits == 0 and meta.hidden_size == 2560 and meta.num_hidden_layers == 48);
    try std.testing.expect(meta.num_experts == 512 and meta.num_experts_per_tok == 10 and meta.moe_intermediate_size == 640);
}

test "discoverModelsMany merges roots in order and de-dups by id" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var a_dir = std.testing.tmpDir(.{ .iterate = true });
    defer a_dir.cleanup();
    var b_dir = std.testing.tmpDir(.{ .iterate = true });
    defer b_dir.cleanup();

    // Same id in BOTH roots. The registry answers `error.DuplicateId` and
    // `registerDiscovered` does `try`, so an un-deduped merge does not produce
    // a duplicate entry — it fails the whole registry init, i.e. the server
    // does not start. De-dup is not tidiness here.
    try a_dir.dir.createDirPath(io, "org/shared");
    try a_dir.dir.writeFile(io, .{ .sub_path = "org/shared/config.json", .data = "{\"model_type\":\"glm5_next\"}" });
    try a_dir.dir.createDirPath(io, "org/only-in-a");
    try a_dir.dir.writeFile(io, .{ .sub_path = "org/only-in-a/config.json", .data = "{\"model_type\":\"mimo_v2\"}" });
    try b_dir.dir.createDirPath(io, "org/shared");
    try b_dir.dir.writeFile(io, .{ .sub_path = "org/shared/config.json", .data = "{\"model_type\":\"deepseek_v41\"}" });
    try b_dir.dir.createDirPath(io, "org/only-in-b");
    try b_dir.dir.writeFile(io, .{ .sub_path = "org/only-in-b/config.json", .data = "{\"model_type\":\"qwen4_exp\"}" });

    // `discoverModels` opens by ABSOLUTE path (the openDirAbsolute UB class),
    // and a testing tmpDir is `<cwd>/.zig-cache/tmp/<sub_path>`.
    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd_ptr = std.c.getcwd(&cwd_buf, cwd_buf.len) orelse return error.NoCwd;
    const cwd = std.mem.span(@as([*:0]const u8, @ptrCast(cwd_ptr)));
    const a_path = try std.fmt.allocPrint(allocator, "{s}/.zig-cache/tmp/{s}", .{ cwd, a_dir.sub_path });
    defer allocator.free(a_path);
    const b_path = try std.fmt.allocPrint(allocator, "{s}/.zig-cache/tmp/{s}", .{ cwd, b_dir.sub_path });
    defer allocator.free(b_path);

    var result = try discoverModelsMany(io, allocator, &.{ a_path, b_path });
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 3), result.models.len);
    // FIRST root wins the shared id: roots are ordered by the caller and the
    // first is the one downloads land in, so a stale copy elsewhere must not
    // shadow the live one.
    var shared_type: []const u8 = "";
    var saw_a = false;
    var saw_b = false;
    for (result.models) |m| {
        if (std.mem.eql(u8, m.id, "org/shared")) shared_type = m.model_type;
        if (std.mem.eql(u8, m.id, "org/only-in-a")) saw_a = true;
        if (std.mem.eql(u8, m.id, "org/only-in-b")) saw_b = true;
        // Every path must be absolute and under the root it came from, or the
        // registry stores a path nothing can open.
        try std.testing.expect(m.path.len > 0 and m.path[0] == '/');
    }
    try std.testing.expectEqualStrings("glm5_next", shared_type);
    try std.testing.expect(saw_a and saw_b);

    // One root behaves exactly like `discoverModels` — the multi path is the
    // only path, so the single-root case cannot be left behind.
    var one = try discoverModelsMany(io, allocator, &.{a_path});
    defer one.deinit();
    try std.testing.expectEqual(@as(usize, 2), one.models.len);

    // A root that does not exist is SKIPPED, not fatal: a user can unplug the
    // external drive their second folder lives on, and that must not stop the
    // server from serving everything else.
    var with_missing = try discoverModelsMany(io, allocator, &.{ a_path, "/nope/not/here", b_path });
    defer with_missing.deinit();
    try std.testing.expectEqual(@as(usize, 3), with_missing.models.len);

    // No roots at all is an empty result, never an error.
    var none = try discoverModelsMany(io, allocator, &.{});
    defer none.deinit();
    try std.testing.expectEqual(@as(usize, 0), none.models.len);
}

test "discovery measures a SYMLINKED (HF hub cache) model dir's real bytes" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    // HF hub-cache snapshot shape: every file is a symlink into ../../blobs.
    // bytes_on_disk feeds /v1/models, the app's RAM column AND
    // model_load.gateEstimateBytes — all of which saw ~0 for a 121 GB model.
    try tmp.dir.createDirPath(io, "blobs");
    try tmp.dir.writeFile(io, .{ .sub_path = "blobs/cfg", .data = "{\"model_type\":\"mimo_v2\"}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "blobs/w1", .data = "0123456789" });
    try tmp.dir.createDirPath(io, "org/snap-model");
    try tmp.dir.symLink(io, "../../blobs/cfg", "org/snap-model/config.json", .{});
    try tmp.dir.symLink(io, "../../blobs/w1", "org/snap-model/model.safetensors", .{});

    var result = try discoverModelsInDir(io, allocator, tmp.dir, "/root");
    defer result.deinit();

    try testing.expectEqual(@as(usize, 1), result.models.len);
    try testing.expectEqualStrings("org/snap-model", result.models[0].id);
    try testing.expectEqual(@as(?u64, 10), result.models[0].bytes_on_disk);
}

test "modelKindFromType labels every family (list TYPE column + run preflight)" {
    try testing.expectEqual(ModelKind.chat, modelKindFromType("qwen4_exp"));
    try testing.expectEqual(ModelKind.chat, modelKindFromType("mimo_v2"));
    try testing.expectEqual(ModelKind.unsupported, modelKindFromType("gemma4"));
    try testing.expectEqual(ModelKind.unsupported, modelKindFromType("qwen3_5_moe"));
    // Drafter sidecars, and genuinely unsupported archs (media
    // generation included: it is no longer served).
    try testing.expectEqual(ModelKind.unsupported, modelKindFromType("flux2-klein-4b"));
    // DFlash sidecars ride the same `*_assistant` suffix rule — never listed
    // as primary models.
    try testing.expectEqual(ModelKind.drafter, modelKindFromType("glm5_next_assistant"));
    try testing.expectEqual(ModelKind.unsupported, modelKindFromType("vit"));
    // Labels stay column-friendly.
    try testing.expectEqualStrings("chat", ModelKind.chat.label());
}

test "every served arch classifies as chat, so `sushi run` accepts it" {
    for (@import("model.zig").served_model_types) |t| try testing.expectEqual(ModelKind.chat, modelKindFromType(t));
}

test "classifyModelPath: drafter dirs classify; junk is null" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    try tmp.dir.createDirPath(io, "drafter");
    try tmp.dir.writeFile(io, .{ .sub_path = "drafter/config.json", .data = "{\"model_type\":\"glm5_next_assistant\"}" });
    try tmp.dir.createDirPath(io, "junk");

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &path_buf);
    const root = path_buf[0..root_len];

    const cases = .{
        .{ "drafter", ModelKind.drafter },
    };
    inline for (cases) |c| {
        const p = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ root, c[0] });
        defer allocator.free(p);
        try testing.expectEqual(c[1], classifyModelPath(io, allocator, p).?);
    }
    const junk = try std.fmt.allocPrint(allocator, "{s}/junk", .{root});
    defer allocator.free(junk);
    try testing.expect(classifyModelPath(io, allocator, junk) == null);
    try testing.expect(classifyModelPath(io, allocator, "") == null);
    try testing.expect(classifyModelPath(io, allocator, "rel/path") == null);
}

test "a DFlash2 sidecar (bare chat model_type + dflash_config) classifies as drafter, never listed" {
    // incoai/Qwen3.8-27B-DFlash2 ships `model_type: "qwen3"` — the
    // `*_assistant` suffix rule alone would register it as a standalone chat
    // model and die at cold load (no embed weights). Classification consults
    // the DFlash config contract too (dflash.isDflashConfigJson — the ONE
    // predicate the loader itself keys on).
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    const d2_config =
        \\{"model_type":"qwen3",
        \\ "dflash_config":{"block_size":8,"mask_token_id":248070,"target_layer_ids":[5,19],
        \\   "conv_kernel_size":2,"conv_group_size":16,"selector_rank":256,"selector_top_k":16},
        \\ "hidden_size":5120,"num_hidden_layers":5,"num_attention_heads":32,"head_dim":128,
        \\ "intermediate_size":17408,"rms_norm_eps":1e-6,"sliding_window":2048,
        \\ "layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention"]}
    ;
    try tmp.dir.createDirPath(io, "d2");
    try tmp.dir.writeFile(io, .{ .sub_path = "d2/config.json", .data = d2_config });

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &path_buf);
    const root = path_buf[0..root_len];
    const p = try std.fmt.allocPrint(allocator, "{s}/d2", .{root});
    defer allocator.free(p);
    try testing.expectEqual(ModelKind.drafter, classifyModelPath(io, allocator, p).?);

    // The discovery scan skips it entirely.
    var result = try discoverModels(io, allocator, root);
    defer result.deinit();
    try testing.expectEqual(@as(usize, 0), result.models.len);
}

test "a .gguf path or a directory holding one is a GGUF path, refused by name" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    try tmp.dir.createDirPath(io, "quants");
    try tmp.dir.writeFile(io, .{ .sub_path = "quants/model-Q4_K_M.gguf", .data = "GGUF" });
    try tmp.dir.createDirPath(io, "mlx");
    try tmp.dir.writeFile(io, .{ .sub_path = "mlx/config.json", .data = "{}" });

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &path_buf);
    const root = path_buf[0..root_len];
    const quants = try std.fmt.allocPrint(allocator, "{s}/quants", .{root});
    defer allocator.free(quants);
    const mlx_dir = try std.fmt.allocPrint(allocator, "{s}/mlx", .{root});
    defer allocator.free(mlx_dir);

    try testing.expect(isGgufModelPath(io, quants));
    try testing.expect(isGgufModelPath(io, "/models/x.gguf"));
    try testing.expect(!isGgufModelPath(io, mlx_dir));
    try testing.expect(!isGgufModelPath(io, ""));
    try testing.expect(!isGgufModelPath(io, "relative/dir"));

    // Discovery does not list it: no config.json, so it is not model-shaped.
    var result = try discoverModelsInDir(io, allocator, tmp.dir, "/root");
    defer result.deinit();
    for (result.models) |m| try testing.expect(!std.mem.eql(u8, m.id, "quants"));
}

test "trimTrailingSlash" {
    try testing.expectEqualStrings("foo", trimTrailingSlash("foo/"));
    try testing.expectEqualStrings("foo", trimTrailingSlash("foo//"));
    try testing.expectEqualStrings("foo", trimTrailingSlash("foo"));
    try testing.expectEqualStrings("", trimTrailingSlash("//"));
}

test "lessThanById sorts ascending" {
    const a: DiscoveredModel = .{ .id = "a", .path = "x", .bytes_on_disk = null };
    const b: DiscoveredModel = .{ .id = "b", .path = "x", .bytes_on_disk = null };
    try testing.expect(lessThanById({}, a, b));
    try testing.expect(!lessThanById({}, b, a));
    try testing.expect(!lessThanById({}, a, a));
}

test "the supported model_type list is the served four" {
    for (@import("model.zig").served_model_types) |t| try testing.expect(isSupportedModelType(t));
    try testing.expect(isSupportedModelType("qwen4_exp_text"));
    try testing.expectEqual(@as(usize, 5), supported_model_types.len);
    for ([_][]const u8{ "qwen3", "qwen3_moe", "qwen3_5_moe", "gemma3_text", "gemma4", "llama", "lfm2", "lfm2_vl", "bert", "totally_made_up_arch" }) |t| {
        try testing.expect(!isSupportedModelType(t));
    }
}

test "isSupportedQuantMode accepts affine, mxfp4 and mxfp8, rejects nvfp4 and unknown" {
    try testing.expect(isSupportedQuantMode("affine"));
    try testing.expect(isSupportedQuantMode("mxfp4"));
    try testing.expect(isSupportedQuantMode("mxfp8"));
    try testing.expect(!isSupportedQuantMode("nvfp4"));
    try testing.expect(!isSupportedQuantMode("fp99"));
}

test "parseStubMeta mimo_v2 reports routed count and image input without video" {
    const m = parseStubMeta(testing.allocator,
        \\{"model_type":"mimo_v2", "num_hidden_layers":4, "hidden_size":384,
        \\ "n_routed_experts":16, "num_experts_per_tok":4, "moe_intermediate_size":192,
        \\ "moe_layer_freq":[0,1,1,1], "quantization":{"bits":4,"mode":"mxfp4"},
        \\ "vision_config":{}, "audio_config":{}, "video_token_id":151656}
    , true);
    try testing.expect(m.found and m.has_chat and m.is_moe);
    try testing.expectEqual(@as(u32, 16), m.num_experts);
    try testing.expectEqual(@as(u32, 1), m.first_moe_layer);
    try testing.expectEqual(@as(u32, 4), m.quant_bits);
    try testing.expect(m.has_vision and !m.has_video and !m.has_mtp);
}

test "mimo_v2 streaming discovery validates MXFP4 headers without a PLE table" {
    const a = testing.allocator;
    try testing.expect(isSupportedModelType("mimo_v2"));
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const bytes = try expert_stream.writeTinyMxfp4Checkpoint(a, tmp.dir, 4, 64, 32);
    defer a.free(bytes);
    try tmp.dir.writeFile(testing.io, .{
        .sub_path = "config.json",
        .data =
        \\{"model_type":"mimo_v2", "num_hidden_layers":2, "hidden_size":64,
        \\ "n_routed_experts":4, "num_experts_per_tok":2, "moe_intermediate_size":32,
        \\ "moe_layer_freq":[0,1], "quantization":{"bits":4,"mode":"mxfp4"}}
        ,
    });
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(testing.io, &path_buf);
    const path = path_buf[0..n];
    try testing.expectEqual(expert_quant.Layout.mxfp4_split, qwen4StreamingIndexComplete(testing.io, a, path).?);
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "mxfp4.safetensors", .data = bytes[0..16] });
    try testing.expect(qwen4StreamingIndexComplete(testing.io, a, path) == null);
}

test "real mimo_v2 original checkpoint discovers complete streamed expert headers" {
    const source = std.c.getenv("MIMO_V2_SOURCE") orelse return error.SkipZigTest;
    const layout = qwen4StreamingIndexComplete(testing.io, testing.allocator, std.mem.span(source)) orelse
        return error.IncompleteMimoSource;
    try testing.expectEqual(expert_quant.Layout.mxfp4_individual, layout);
}

test "mimo_v2 original source discovery validates individual MXFP4 tensors" {
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const bytes = try expert_quant.writeTinyMxfp4IndividualCheckpoint(a, tmp.dir, 4, 64, 32);
    defer a.free(bytes);
    try tmp.dir.writeFile(testing.io, .{
        .sub_path = "config.json",
        .data =
        \\{"model_type":"mimo_v2", "num_hidden_layers":2, "hidden_size":64,
        \\ "n_routed_experts":4, "num_experts_per_tok":2, "moe_intermediate_size":32,
        \\ "moe_layer_freq":[0,1], "quantization_config":{"quant_method":"fp8","store_dtype":"mxfp4"}}
        ,
    });
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(testing.io, &path_buf);
    try testing.expectEqual(expert_quant.Layout.mxfp4_individual, qwen4StreamingIndexComplete(testing.io, a, path_buf[0..n]).?);
}

test "parseStubMeta extracts dims/ctx/quant/MoE + chat/vision capabilities" {
    const a = testing.allocator;
    // MoE chat model (Qwen3-Coder-30B-A3B shape), chat template present.
    {
        const json =
            \\{"model_type":"qwen4_exp","vocab_size":151936,"hidden_size":2048,
            \\"num_hidden_layers":48,"max_position_embeddings":262144,
            \\"num_experts":128,"num_experts_per_tok":8,"quantization":{"bits":8,"group_size":64}}
        ;
        const m = parseStubMeta(a, json, true);
        try testing.expect(m.found);
        try testing.expectEqual(@as(u32, 151936), m.vocab_size);
        try testing.expectEqual(@as(u32, 2048), m.hidden_size);
        try testing.expectEqual(@as(u32, 48), m.num_hidden_layers);
        try testing.expectEqual(@as(u32, 262144), m.max_position_embeddings);
        try testing.expectEqual(@as(u32, 8), m.quant_bits);
        try testing.expect(m.is_moe);
        try testing.expect(m.has_chat); // template present, not encoder
        try testing.expect(!m.has_vision);
    }
    // Dense model, no template → no chat caps; no quant block → 0 bits.
    {
        const json =
            \\{"model_type":"qwen4_exp","hidden_size":5120,"num_attention_heads":40,
            \\"max_position_embeddings":32768}
        ;
        const m = parseStubMeta(a, json, false);
        try testing.expect(m.found);
        try testing.expect(!m.is_moe);
        try testing.expect(!m.has_chat);
        try testing.expectEqual(@as(u32, 0), m.quant_bits);
        try testing.expectEqual(@as(u32, 32768), m.max_position_embeddings);
    }
    // Vision: vision_config on a non-_text arch → has_vision.
    {
        const m = parseStubMeta(a, "{\"model_type\":\"qwen4_exp\",\"vision_config\":{\"hidden_size\":1152}}", true);
        try testing.expect(m.has_vision);
    }
    // …but a `_text` arch with a vestigial vision_config must NOT report vision.
    {
        const m = parseStubMeta(a, "{\"model_type\":\"qwen4_exp_text\",\"vision_config\":{}}", true);
        try testing.expect(!m.has_vision);
    }
    // A MULTIMODAL checkpoint keeps every text dim under `text_config` — the
    // root carries only model_type / vision_config / quantization. Reading the
    // root alone reported hidden=0, layers=0, ctx=0, is_moe=false on /v1/models
    // for EVERY unloaded multimodal model (which is most of them),
    // so a client couldn't tell a 128-expert MoE from a dense model without
    // cold-loading 16 GB of weights. Same class as the "Config fields omitted
    // by nested text_config" gotcha, different parser.
    {
        const json =
            \\{"model_type":"qwen4_exp","vision_config":{"hidden_size":1152},
            \\"quantization":{"bits":4,"group_size":32},
            \\"text_config":{"vocab_size":262144,"hidden_size":2560,"num_hidden_layers":62,
            \\"max_position_embeddings":131072,"num_experts":128,"top_k_experts":8}}
        ;
        const m = parseStubMeta(a, json, true);
        try testing.expect(m.found);
        try testing.expectEqual(@as(u32, 262144), m.vocab_size);
        try testing.expectEqual(@as(u32, 2560), m.hidden_size);
        try testing.expectEqual(@as(u32, 62), m.num_hidden_layers);
        try testing.expectEqual(@as(u32, 131072), m.max_position_embeddings);
        try testing.expectEqual(@as(u32, 4), m.quant_bits); // still root-level
        try testing.expect(m.is_moe);
        try testing.expect(m.has_vision);
    }
    // A flat checkpoint whose text_config is absent keeps reading the root, and
    // a nested block that OMITS a field falls back to the root rather than
    // reporting 0 (mirrors the model.zig text_config merge).
    {
        const json =
            \\{"model_type":"qwen4_exp","max_position_embeddings":8192,
            \\"text_config":{"hidden_size":3840}}
        ;
        const m = parseStubMeta(a, json, true);
        try testing.expectEqual(@as(u32, 3840), m.hidden_size);
        try testing.expectEqual(@as(u32, 8192), m.max_position_embeddings);
    }
    // A MoE nests its expert count too.
    {
        const m = parseStubMeta(a, "{\"model_type\":\"qwen4_exp\",\"text_config\":{\"num_experts\":256}}", true);
        try testing.expect(m.is_moe);
    }
    // Malformed → found=false.
    {
        const m = parseStubMeta(a, "not json", true);
        try testing.expect(!m.found);
    }
}

test "readStubMeta: has_mtp follows the checkpoint's MTP head" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "m");
    try tmp.dir.writeFile(io, .{ .sub_path = "m/config.json", .data = "{\"model_type\":\"qwen4_exp\",\"hidden_size\":8}" });
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &path_buf);
    const model_dir = try std.fmt.allocPrint(allocator, "{s}/m", .{path_buf[0..root_len]});
    defer allocator.free(model_dir);

    try std.testing.expect(!readStubMeta(io, allocator, model_dir).has_mtp);
    // qwen4_exp's head is the checkpoint's own layer.
    try tmp.dir.writeFile(io, .{ .sub_path = "m/model.safetensors.index.json", .data =
        \\{"weight_map":{"language_model.mtp.fc_hidden.weight":"model-00002.safetensors"}}
    });
    try std.testing.expect(readStubMeta(io, allocator, model_dir).has_mtp);
}

test "discovery skips a config.json whose root is not an object and drops a size past u32" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    try std.testing.expectEqual(@as(u32, 0), parseStubMeta(allocator, "{\"hidden_size\":4294967297}", false).hidden_size);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    for ([_][]const u8{ "[]", "null", "false", "17", "\"qwen4_exp\"", "[{}]" }) |content| {
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = content });
        try std.testing.expect(peekConfig(io, allocator, tmp.dir, ".") == .missing_or_unparseable);
    }
}
