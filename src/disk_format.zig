//! The SSD record's wire format, apart from the files it lives in: the safetensors header a tensor
//! file starts with, and the `meta.json` manifest an entry directory is described by. A cache
//! written by an older binary restores through this code, so a byte here changes only with a
//! manifest version (`docs/engine-prefix-cache.md`, "SSD record format"). Host tier: no MLX array.

const std = @import("std");
const kv_quant = @import("kv_quant.zig");

/// One persisted ring restore point: the ringed layers' rows `[pos - rows, pos)`.
pub const RingFile = struct { pos: u32, rows: u32, bytes: u64 };

/// v4 spec-snapshot metadata for one speculative-side cache (dflash assistant
/// context or MTP committed history). The tensors live in the entry's ONE
/// `spec.safetensors` file, keyed `d{layer}.*` / `m{layer}.*`.
pub const SpecMeta = struct {
    /// Absolute trunk position the snapshot's index 0 represents.
    base: u64,
    /// Positions persisted (the snapshot's logical length).
    step: u32,
    /// Layer count of the source cache — a restore target with a different
    /// count declines (KVCache.restore asserts equal lengths).
    layers: u32,
    quant: kv_quant.KVQuantConfig,
    /// v5, in-checkpoint MTP head only: the head's QSA aux half in the same sidecar. Null = a
    /// head-only miss at restore; the trunk entry is unaffected.
    head: ?SpecHeadMeta = null,
};

/// v12: one whole segment of a growing sidecar's rows, as written: its size and a Wyhash of its file image,
/// which lets a later commit holding the same bytes keep or link the file instead of writing it again.
pub const SideSeg = struct { bytes: u64, hash: u64 };

/// Marks a head sidecar records. The length is part of the record, so the format owns it and the arch that
/// keeps the marks (`QSA_HEAD_MARKS_MAX`) takes it from here.
pub const HEAD_MARKS_MAX: usize = 32;

/// v5 head half of a `SpecMeta`: the scalars the head's position bookkeeping needs.
pub const SpecHeadMeta = struct {
    /// Absolute position of the head's key row 0 (the head's `pos_base`).
    pos_base: i32,
    ratio: i32,
    pooled: bool,
    rows: i32 = 0,
    /// v8: head rows the sidecar carries a QSA leftover for, ascending — the trunk's
    /// checkpoint positions. Tensor `<prefix>h.lv<i>` holds the i-th. Inline: an IndexEntry
    /// is scanned and copied by value.
    marks: [HEAD_MARKS_MAX]i32 = @splat(0),
    mark_count: u8 = 0,
};

pub const IndexEntry = struct {
    /// Directory id — the `e<id>` component.
    id: u64,
    /// Full committed token sequence (prompt ++ generated). Owned.
    tokens: []u32,
    /// KV positions actually persisted (== snapshot `step` at commit; may be
    /// tokens.len - 1 when the final sampled token was never forwarded).
    kv_len: u32,
    has_tools: bool,
    quant: kv_quant.KVQuantConfig,
    /// Total on-disk bytes (chunks + tokens.bin + meta).
    bytes: u64,
    /// Per-chunk file sizes recorded at commit (meta.json "chunk_bytes").
    /// The scan validates actual file sizes against these and clamps kv_len
    /// to the last contiguous valid chunk — a kill -9 mid-flush truncates a
    /// chunk, and restoring it would poison the cache. Owned.
    chunk_bytes: []u64,
    /// The first `inherited_chunks` chunk files are hard links into a donor entry's chunks;
    /// this entry never billed them. meta.json v6 `inherited_chunks`; 0 on older manifests.
    inherited_chunks: u32 = 0,
    /// Phase 3: persisted SSM checkpoint positions (sorted ascending; empty
    /// for pure-attention entries) and per-file byte sizes (parallel array —
    /// the same kill -9 salvage role as `chunk_bytes`: the scan drops
    /// individual positions whose file size mismatches). Owned.
    ssm_positions: []u32,
    ssm_bytes: []u64,
    /// v4: spec-snapshot sidecar (`spec.safetensors`) byte size; 0 = none.
    /// The same kill -9 salvage as chunks — a size mismatch at scan drops the
    /// SPEC only (a restore then starts blind), never the entry.
    spec_bytes: u64 = 0,
    /// v4: dflash assistant context / MTP committed history persisted in the
    /// spec sidecar. DRAFT-side state — a missing or dropped snap costs
    /// acceptance on the first reused turn, never a token.
    spec_dflash: ?SpecMeta = null,
    spec_mtp: ?SpecMeta = null,
    /// v12: the MTP history's whole segments `p<j>.safetensors`, in order, ahead of the rows the sidecar keeps; the
    /// first `spec_segs_inherited` are hard links into a donor entry's, which this entry never billed. Owned.
    spec_segs: []SideSeg = &.{},
    spec_segs_inherited: u32 = 0,
    /// A background write for this entry failed: never matched, never a donor, never reported
    /// complete; the next commit reclaims its directory. Not persisted (`scan` re-validates).
    poisoned: bool = false,
    qsa_history_bytes: u64 = 0,
    qsa_history_rows: u32 = 0,
    inherited_qsa: bool = false,
    /// v11: bytes of each row segment `g<n>.safetensors`, in order, and the tokens a segment spans. Empty for
    /// an entry whose restore points hold their own rows.
    grown: []u64 = &.{},
    grown_span: u32 = 0,
    /// The first `inherited_grown` of them are hard links into a donor entry's: this entry never billed them.
    inherited_grown: u32 = 0,
    /// v9: the ringed layers' restore points (`r{pos}.safetensors`), ascending; empty for an
    /// entry whose layers all hold the prefix. A ringed entry restores only at one of them. Owned.
    rings: []RingFile = &.{},
    /// v10: the media chain this record's rows were computed under (`prefix_cache.MediaSpan`), as
    /// its last block's key, and the first media placeholder in `tokens`. 0 / null = no media row,
    /// which is what every pre-v10 record is. A lookup hands the tier per-entry limits
    /// (`MediaLimits`) computed from these by the one owner of the rule,
    /// `prefix_cache.crossKeyBoundary`; a commit applies that rule with no chain (`commitShares`).
    vision_key: u64 = 0,
    media_start: ?u32 = null,
    /// In-process LRU stamp; seeded from meta.json mtime order at scan.
    last_used: u64,
};

/// The newest manifest version this reader restores; an older one just carries none of the newer state.
pub const META_VERSION_MAX: u64 = 12;

/// The lowest manifest version that describes this entry. The version is a compatibility
/// claim: an older reader accepts only 2..4, so stamping v6 unconditionally made a binary
/// downgrade discard the whole tier. v6 = inherited chunks, v5 = the MTP head's QSA half.
pub fn metaVersionFor(e: IndexEntry) u8 {
    // v12: an older reader would take the sidecar's tail rows for the whole history.
    if (e.spec_segs.len > 0) return 12;
    // v11: an older reader would look for rows in the point files, which keep only the ones past the segments.
    if (e.grown.len > 0) return 11;
    // v10: an older reader knows no media identity, so it would hand a text prompt rows that
    // only this record's pixels explain. Dropping the record is the only safe read.
    if (e.vision_key != 0) return 10;
    // v9: an older reader would restore the chunks alone, leaving the ringed layers empty.
    if (e.rings.len > 0) return 9;
    if (e.qsa_history_rows > 0 or e.qsa_history_bytes > 0) return 8;
    if (e.inherited_chunks > 0) return 6;
    if (e.spec_mtp) |m| if (m.head) |h| return if (h.mark_count > 0) 8 else 5;
    return 4;
}

/// The meta.json body; one renderer for the synchronous and the staged path.
pub fn renderMeta(a: std.mem.Allocator, out: *std.ArrayList(u8), e: IndexEntry, chunk_tokens: u32) !void {
    try out.print(
        a,
        "{{\"v\":{d},\"kv_len\":{d},\"tokens\":{d},\"has_tools\":{},\"scheme\":\"{s}\",\"bits\":{d},\"group_size\":{d},\"chunk_tokens\":{d},\"inherited_chunks\":{d},\"bytes\":{d},\"chunk_bytes\":[",
        .{
            metaVersionFor(e),
            e.kv_len,
            e.tokens.len,
            e.has_tools,
            @tagName(e.quant.scheme),
            e.quant.bits,
            e.quant.group_size,
            chunk_tokens,
            e.inherited_chunks,
            e.bytes,
        },
    );
    for (e.chunk_bytes, 0..) |cb, i| {
        if (i > 0) try out.appendSlice(a, ",");
        try out.print(a, "{d}", .{cb});
    }
    // v3: SSM checkpoints as [{pos,bytes},...] (sorted ascending).
    try out.appendSlice(a, "],\"ssm\":[");
    for (e.ssm_positions, e.ssm_bytes, 0..) |pos, sz, i| {
        if (i > 0) try out.appendSlice(a, ",");
        try out.print(a, "{{\"pos\":{d},\"bytes\":{d}}}", .{ pos, sz });
    }
    try out.appendSlice(a, "]");
    if (e.grown.len > 0) {
        try out.print(a, ",\"grown\":{{\"span\":{d},\"inherited\":{d},\"bytes\":[", .{ e.grown_span, e.inherited_grown });
        for (e.grown, 0..) |gb, i| {
            if (i > 0) try out.appendSlice(a, ",");
            try out.print(a, "{d}", .{gb});
        }
        try out.appendSlice(a, "]}");
    }
    if (e.rings.len > 0) {
        try out.appendSlice(a, ",\"ring\":[");
        for (e.rings, 0..) |r, i| {
            if (i > 0) try out.appendSlice(a, ",");
            try out.print(a, "{{\"pos\":{d},\"rows\":{d},\"bytes\":{d}}}", .{ r.pos, r.rows, r.bytes });
        }
        try out.appendSlice(a, "]");
    }
    // v10: the media chain the rows were computed under. `start` is absent when the record
    // holds no media row, which is every record an older binary wrote.
    if (e.vision_key != 0) {
        try out.print(a, ",\"media\":{{\"key\":\"{x}\"", .{e.vision_key});
        if (e.media_start) |ms| try out.print(a, ",\"start\":{d}", .{ms});
        try out.appendSlice(a, "}");
    }
    if (e.qsa_history_rows > 0 or e.qsa_history_bytes > 0) {
        try out.print(a, ",\"qsa_history\":{{\"bytes\":{d},\"rows\":{d},\"inherited\":{}}}", .{
            e.qsa_history_bytes,
            e.qsa_history_rows,
            e.inherited_qsa,
        });
    }
    // v4: spec snapshots (dflash context / MTP history); a size mismatch drops only the spec.
    if ((e.spec_bytes > 0 and (e.spec_dflash != null or e.spec_mtp != null)) or e.spec_segs.len > 0) {
        try out.print(a, ",\"spec\":{{\"bytes\":{d}", .{e.spec_bytes});
        if (e.spec_segs.len > 0) try writeSegsJson(a, out, e.spec_segs, e.spec_segs_inherited);
        if (e.spec_dflash) |sm| try writeSpecMetaJson(a, out, "dflash", sm);
        if (e.spec_mtp) |sm| try writeSpecMetaJson(a, out, "mtp", sm);
        try out.appendSlice(a, "}");
    }
    try out.appendSlice(a, "}");
}

/// `"segs":{"inherited":n,"files":[[bytes,"hash"],...]}`; the hash is hex, past JSON's i64.
fn writeSegsJson(a: std.mem.Allocator, out: *std.ArrayList(u8), segs: []const SideSeg, inherited: u32) !void {
    try out.print(a, ",\"segs\":{{\"inherited\":{d},\"files\":[", .{inherited});
    for (segs, 0..) |sg, i| {
        if (i > 0) try out.appendSlice(a, ",");
        try out.print(a, "[{d},\"{x}\"]", .{ sg.bytes, sg.hash });
    }
    try out.appendSlice(a, "]}");
}

/// One `[bytes, "hash"]` item of a `segs` list; null for a malformed one, which ends the list's read there.
pub fn segItem(v: std.json.Value) ?SideSeg {
    if (v != .array or v.array.items.len != 2) return null;
    const bytes = valueU64(v.array.items[0]) orelse return null;
    const hv = v.array.items[1];
    if (hv != .string) return null;
    return .{ .bytes = bytes, .hash = std.fmt.parseInt(u64, hv.string, 16) catch return null };
}

/// A `segs` section: the files as listed and how many lead as hard links.
pub const SegsMeta = struct { files: []const std.json.Value, inherited: u32 };

fn parseSegs(obj: std.json.ObjectMap) ?SegsMeta {
    const v = obj.get("segs") orelse return null;
    if (v != .object) return null;
    const files = v.object.get("files") orelse return null;
    if (files != .array) return null;
    return .{ .files = files.array.items, .inherited = jsonInt(u32, v.object, "inherited") orelse return null };
}

pub fn writeSpecMetaJson(a: std.mem.Allocator, w: *std.ArrayList(u8), name: []const u8, sm: SpecMeta) !void {
    try w.print(a, ",\"{s}\":{{\"base\":{d},\"step\":{d},\"layers\":{d},\"scheme\":\"{s}\",\"bits\":{d},\"group_size\":{d}", .{
        name, sm.base, sm.step, sm.layers, @tagName(sm.quant.scheme), sm.quant.bits, sm.quant.group_size,
    });
    if (sm.head) |h| {
        try w.print(a, ",\"head\":{{\"pos_base\":{d},\"ratio\":{d},\"pooled\":{s},\"rows\":{d}", .{
            h.pos_base, h.ratio, if (h.pooled) "true" else "false", h.rows,
        });
        if (h.mark_count > 0) {
            try w.appendSlice(a, ",\"marks\":[");
            for (h.marks[0..h.mark_count], 0..) |pos, i| {
                if (i > 0) try w.appendSlice(a, ",");
                try w.print(a, "{d}", .{pos});
            }
            try w.appendSlice(a, "]");
        }
        try w.appendSlice(a, "}");
    }
    try w.appendSlice(a, "}");
}

fn jsonU64(obj: std.json.ObjectMap, key: []const u8) ?u64 {
    const v = obj.get(key) orelse return null;
    return valueU64(v);
}

/// A manifest integer that is not negative.
pub fn valueU64(v: std.json.Value) ?u64 {
    if (v != .integer) return null;
    if (v.integer < 0) return null;
    return @intCast(v.integer);
}

/// A manifest scalar narrowed to the field that will hold it. meta.json is on disk and
/// hand-editable, so a value that does not fit drops its record rather than being cast.
fn jsonInt(comptime T: type, obj: std.json.ObjectMap, key: []const u8) ?T {
    const v = obj.get(key) orelse return null;
    if (v != .integer) return null;
    return std.math.cast(T, v.integer);
}

/// The KV quant config a manifest describes, or null when it is not one this build can hold.
fn manifestQuant(obj: std.json.ObjectMap) ?kv_quant.KVQuantConfig {
    const scheme_v = obj.get("scheme") orelse return null;
    if (scheme_v != .string) return null;
    const scheme = std.meta.stringToEnum(kv_quant.Scheme, scheme_v.string) orelse return null;
    // A dense entry may still name a width: a latent-row layout keys its kv8 and kv6 rows so.
    if (scheme == .off) return .{ .scheme = .off, .bits = jsonInt(u8, obj, "bits") orelse 0, .group_size = jsonInt(u32, obj, "group_size") orelse 0 };
    if (scheme != .affine) return null;
    const bits = jsonInt(u8, obj, "bits") orelse return null;
    if (!kv_quant.KVQuantConfig.supportedBits(bits)) return null;
    const gs = jsonInt(u32, obj, "group_size") orelse return null;
    if (gs == 0) return null;
    var q = kv_quant.KVQuantConfig.affine(bits);
    q.group_size = gs;
    return q;
}

pub fn parseSpecMeta(obj: std.json.ObjectMap, key: []const u8) ?SpecMeta {
    const v = obj.get(key) orelse return null;
    if (v != .object) return null;
    const o = v.object;
    const base = jsonU64(o, "base") orelse return null;
    const step = jsonInt(u32, o, "step") orelse return null;
    const layers = jsonInt(u32, o, "layers") orelse return null;
    if (step == 0 or layers == 0) return null;
    const quant = manifestQuant(o) orelse return null;
    // v5 head half; absent on every earlier manifest.
    var head: ?SpecHeadMeta = null;
    if (o.get("head")) |hv| head_blk: {
        if (hv != .object) break :head_blk;
        const ho = hv.object;
        const pb = jsonInt(i32, ho, "pos_base") orelse break :head_blk;
        const ratio = jsonInt(i32, ho, "ratio") orelse break :head_blk;
        if (ratio == 0) break :head_blk;
        const pooled_v = ho.get("pooled") orelse break :head_blk;
        if (pooled_v != .bool) break :head_blk;
        const rows = jsonInt(i32, ho, "rows") orelse 0;
        var hm: SpecHeadMeta = .{ .pos_base = pb, .ratio = ratio, .pooled = pooled_v.bool, .rows = rows };
        // v8 leftovers; absent on every earlier manifest.
        if (ho.get("marks")) |mv| {
            if (mv == .array) {
                for (mv.array.items) |iv| {
                    if (iv != .integer or hm.mark_count == HEAD_MARKS_MAX) break;
                    hm.marks[hm.mark_count] = std.math.cast(i32, iv.integer) orelse break;
                    hm.mark_count += 1;
                }
            }
        }
        head = hm;
    }
    return .{ .base = base, .step = step, .layers = layers, .quant = quant, .head = head };
}

/// The row segments a v11 manifest lists. `inherited` is null when the field is missing or does not fit, which
/// ends the read of the section there: the segments are listed but none is validated.
pub const GrownMeta = struct { span: u32, bytes: []const std.json.Value, inherited: ?u32 };

pub const QsaHistoryMeta = struct { bytes: u64, rows: u32, inherited: bool };

/// The spec sidecar's recorded size and the two snapshots it holds; either may be null.
pub const SpecSidecar = struct { bytes: u64, dflash: ?SpecMeta, mtp: ?SpecMeta, segs: ?SegsMeta = null };

pub const MediaMeta = struct { key: u64, start: u32 };

/// What a manifest says, before any file is checked against it. The lists borrow the parsed JSON.
pub const Manifest = struct {
    version: u64,
    inherited_chunks: u64,
    kv_len: u64,
    tokens: u64,
    chunk_tokens: u64,
    has_tools: bool,
    chunk_bytes: []const std.json.Value,
    quant: kv_quant.KVQuantConfig,
    grown: ?GrownMeta,
    ssm: []const std.json.Value,
    ring: []const std.json.Value,
    qsa_history: ?QsaHistoryMeta,
    spec: ?SpecSidecar,
    media: ?MediaMeta,
};

/// Null when the manifest is not one this build can restore: an unknown version, a missing or ill-typed
/// required field, or a media identity that names no boundary.
pub fn parseManifest(obj: std.json.ObjectMap) ?Manifest {
    const version = jsonU64(obj, "v") orelse return null;
    if (version < 2 or version > META_VERSION_MAX) return null;
    const kv_len = jsonU64(obj, "kv_len") orelse return null;
    const n_tokens = jsonU64(obj, "tokens") orelse return null;
    const chunk_tokens = jsonU64(obj, "chunk_tokens") orelse return null;
    const has_tools_v = obj.get("has_tools") orelse return null;
    if (has_tools_v != .bool) return null;
    const chunk_bytes_v = obj.get("chunk_bytes") orelse return null;
    if (chunk_bytes_v != .array) return null;
    const quant = manifestQuant(obj) orelse return null;

    var grown: ?GrownMeta = null;
    if (obj.get("grown")) |gv| grown_blk: {
        if (gv != .object) break :grown_blk;
        const span = jsonInt(u32, gv.object, "span") orelse break :grown_blk;
        const list_v = gv.object.get("bytes") orelse break :grown_blk;
        if (list_v != .array or span == 0) break :grown_blk;
        grown = .{ .span = span, .bytes = list_v.array.items, .inherited = jsonInt(u32, gv.object, "inherited") };
    }

    var ssm: []const std.json.Value = &.{};
    if (obj.get("ssm")) |v| if (v == .array) {
        ssm = v.array.items;
    };
    var ring: []const std.json.Value = &.{};
    if (obj.get("ring")) |v| if (v == .array) {
        ring = v.array.items;
    };

    var qsa_history: ?QsaHistoryMeta = null;
    if (obj.get("qsa_history")) |qh_v| qh_blk: {
        if (qh_v != .object) break :qh_blk;
        const qo = qh_v.object;
        const bytes = jsonU64(qo, "bytes") orelse break :qh_blk;
        const rows = jsonInt(u32, qo, "rows") orelse break :qh_blk;
        const inh_v = qo.get("inherited");
        qsa_history = .{ .bytes = bytes, .rows = rows, .inherited = inh_v != null and inh_v.? == .bool and inh_v.?.bool };
    }

    var spec: ?SpecSidecar = null;
    if (obj.get("spec")) |spec_v| spec_blk: {
        if (spec_v != .object) break :spec_blk;
        const so = spec_v.object;
        const bytes = jsonU64(so, "bytes") orelse break :spec_blk;
        spec = .{ .bytes = bytes, .dflash = parseSpecMeta(so, "dflash"), .mtp = parseSpecMeta(so, "mtp"), .segs = parseSegs(so) };
    }

    // A record claiming media must say where it starts: a key without a boundary would be matched by
    // nothing (`crossKeyBoundary` needs one of the two), so the pair is read together or neither is.
    var media: ?MediaMeta = null;
    if (obj.get("media")) |mv| {
        if (mv != .object) return null;
        // Hex, not a number: a Wyhash key runs past JSON's i64 and `std.json` hands those
        // back as `.number_string`, which would have dropped half of all media records.
        const key_v = mv.object.get("key") orelse return null;
        if (key_v != .string) return null;
        const key = std.fmt.parseInt(u64, key_v.string, 16) catch return null;
        const start = jsonInt(u32, mv.object, "start") orelse return null;
        if (key == 0 or start >= n_tokens) return null;
        media = .{ .key = key, .start = start };
    }

    return .{
        .version = version,
        .inherited_chunks = jsonU64(obj, "inherited_chunks") orelse 0,
        .kv_len = kv_len,
        .tokens = n_tokens,
        .chunk_tokens = chunk_tokens,
        .has_tools = has_tools_v.bool,
        .chunk_bytes = chunk_bytes_v.array.items,
        .quant = quant,
        .grown = grown,
        .ssm = ssm,
        .ring = ring,
        .qsa_history = qsa_history,
        .spec = spec,
        .media = media,
    };
}

/// One `{pos, bytes}` item of the manifest's `ssm` list; null for a malformed item, which the scan skips.
pub fn ssmItem(v: std.json.Value) ?struct { pos: u64, bytes: u64 } {
    if (v != .object) return null;
    const pos = jsonU64(v.object, "pos") orelse return null;
    const bytes = jsonU64(v.object, "bytes") orelse return null;
    return .{ .pos = pos, .bytes = bytes };
}

/// One item of the manifest's `ring` list; null for a malformed item, which the scan skips.
pub fn ringItem(v: std.json.Value) ?RingFile {
    if (v != .object) return null;
    const pos = jsonInt(u32, v.object, "pos") orelse return null;
    const rows = jsonInt(u32, v.object, "rows") orelse return null;
    const bytes = jsonU64(v.object, "bytes") orelse return null;
    return .{ .pos = pos, .rows = rows, .bytes = bytes };
}

/// One `__metadata__` entry. The encoder refuses anything needing JSON escaping.
pub const MetaPair = struct { key: []const u8, value: []const u8 };

/// Is `v` safe inside a JSON string with no escaping?
fn plainJsonAscii(v: []const u8) bool {
    for (v) |c| {
        if (c < 0x20 or c > 0x7e or c == '"' or c == '\\') return false;
    }
    return true;
}

/// What a safetensors header says of one tensor.
pub const TensorDesc = struct {
    key: []const u8,
    dtype: []const u8,
    shape: []const c_int,
    nbytes: u64,
};

/// The 8-byte little-endian header length and the header JSON, as `mlx::core::save_safetensors` writes
/// them. `src` has `len()` and `desc(i)` over the file's tensors in order.
pub fn encodeHeader(allocator: std.mem.Allocator, src: anytype, meta: []const MetaPair) ![]u8 {
    var header = std.ArrayList(u8).empty;
    errdefer header.deinit(allocator);
    try header.appendNTimes(allocator, 0, 8);
    try header.appendSlice(allocator, "{\"__metadata__\":{");
    for (meta, 0..) |m, mi| {
        if (!plainJsonAscii(m.key) or !plainJsonAscii(m.value)) return error.DiskCacheBadMetadata;
        if (mi > 0) try header.appendSlice(allocator, ",");
        try header.print(allocator, "\"{s}\":\"{s}\"", .{ m.key, m.value });
    }
    try header.appendSlice(allocator, "}");
    var off: u64 = 0;
    for (0..src.len()) |ti| {
        const t = try src.desc(ti);
        try header.print(allocator, ",\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ t.key, t.dtype });
        for (t.shape, 0..) |d, i| {
            if (i > 0) try header.appendSlice(allocator, ",");
            try header.print(allocator, "{d}", .{d});
        }
        try header.print(allocator, "],\"data_offsets\":[{d},{d}]}}", .{ off, off + t.nbytes });
        off += t.nbytes;
    }
    try header.appendSlice(allocator, "}");
    std.mem.writeInt(u64, header.items[0..8], @intCast(header.items.len - 8), .little);
    return header.toOwnedSlice(allocator);
}

const testing = std.testing;

fn testEntry() IndexEntry {
    return .{
        .id = 7,
        .tokens = @constCast(&[_]u32{ 1, 2, 3 }),
        .kv_len = 600,
        .has_tools = true,
        .quant = kv_quant.KVQuantConfig.affine(8),
        .bytes = 9000,
        .chunk_bytes = @constCast(&[_]u64{ 4096, 2048 }),
        .ssm_positions = @constCast(&[_]u32{512}),
        .ssm_bytes = @constCast(&[_]u64{777}),
        .last_used = 1,
    };
}

const full_manifest =
    "{\"v\":11,\"kv_len\":600,\"tokens\":3,\"has_tools\":true,\"scheme\":\"affine\",\"bits\":8,\"group_size\":64," ++
    "\"chunk_tokens\":1024,\"inherited_chunks\":1,\"bytes\":9000,\"chunk_bytes\":[4096,2048]," ++
    "\"ssm\":[{\"pos\":512,\"bytes\":777}]," ++
    "\"grown\":{\"span\":8192,\"inherited\":1,\"bytes\":[100,200]}," ++
    "\"ring\":[{\"pos\":600,\"rows\":64,\"bytes\":50}]," ++
    "\"media\":{\"key\":\"abc\",\"start\":2}," ++
    "\"qsa_history\":{\"bytes\":10,\"rows\":5,\"inherited\":true}," ++
    "\"spec\":{\"bytes\":30," ++
    "\"dflash\":{\"base\":1,\"step\":2,\"layers\":3,\"scheme\":\"off\",\"bits\":0,\"group_size\":0}," ++
    "\"mtp\":{\"base\":9,\"step\":4,\"layers\":1,\"scheme\":\"off\",\"bits\":0,\"group_size\":0," ++
    "\"head\":{\"pos_base\":1,\"ratio\":4,\"pooled\":true,\"rows\":2,\"marks\":[64,128]}}}}";

fn fullEntry() IndexEntry {
    var e = testEntry();
    e.inherited_chunks = 1;
    e.grown = @constCast(&[_]u64{ 100, 200 });
    e.grown_span = 8192;
    e.inherited_grown = 1;
    e.rings = @constCast(&[_]RingFile{.{ .pos = 600, .rows = 64, .bytes = 50 }});
    e.vision_key = 0xabc;
    e.media_start = 2;
    e.qsa_history_bytes = 10;
    e.qsa_history_rows = 5;
    e.inherited_qsa = true;
    e.spec_bytes = 30;
    e.spec_dflash = .{ .base = 1, .step = 2, .layers = 3, .quant = kv_quant.KVQuantConfig.dense };
    var head: SpecHeadMeta = .{ .pos_base = 1, .ratio = 4, .pooled = true, .rows = 2, .mark_count = 2 };
    head.marks[0] = 64;
    head.marks[1] = 128;
    e.spec_mtp = .{ .base = 9, .step = 4, .layers = 1, .quant = kv_quant.KVQuantConfig.dense, .head = head };
    return e;
}

test "manifest: the bytes of a plain and of a fully loaded entry are pinned" {
    var out = std.ArrayList(u8).empty;
    defer out.deinit(testing.allocator);
    var plain = testEntry();
    plain.inherited_chunks = 1;
    try renderMeta(testing.allocator, &out, plain, 1024);
    try testing.expectEqualStrings(
        "{\"v\":6,\"kv_len\":600,\"tokens\":3,\"has_tools\":true,\"scheme\":\"affine\",\"bits\":8,\"group_size\":64," ++
            "\"chunk_tokens\":1024,\"inherited_chunks\":1,\"bytes\":9000,\"chunk_bytes\":[4096,2048],\"ssm\":[{\"pos\":512,\"bytes\":777}]}",
        out.items,
    );
    out.clearRetainingCapacity();
    try renderMeta(testing.allocator, &out, fullEntry(), 1024);
    try testing.expectEqualStrings(full_manifest, out.items);
}

test "manifest: the lowest version that describes an entry is stamped" {
    var e = testEntry();
    e.ssm_positions = &.{};
    e.ssm_bytes = &.{};
    const kv_only = SpecMeta{ .base = 0, .step = 600, .layers = 2, .quant = kv_quant.KVQuantConfig.dense };
    const with_head = SpecMeta{
        .base = 0,
        .step = 600,
        .layers = 1,
        .quant = kv_quant.KVQuantConfig.dense,
        .head = .{ .pos_base = 1, .ratio = 4, .pooled = true },
    };
    try testing.expectEqual(@as(u8, 4), metaVersionFor(e));
    // A dflash sidecar is v4 shape too.
    e.spec_bytes = 4096;
    e.spec_dflash = kv_only;
    try testing.expectEqual(@as(u8, 4), metaVersionFor(e));
    // A KV-only MTP snap is still v4; the in-checkpoint head's QSA half lifts it to v5.
    e.spec_mtp = kv_only;
    try testing.expectEqual(@as(u8, 4), metaVersionFor(e));
    e.spec_mtp = with_head;
    try testing.expectEqual(@as(u8, 5), metaVersionFor(e));
    // Inherited (hard-linked) chunks are v6: an older reader would bill and delete a donor's files.
    e.inherited_chunks = 4;
    try testing.expectEqual(@as(u8, 6), metaVersionFor(e));
    e.spec_mtp = null;
    try testing.expectEqual(@as(u8, 6), metaVersionFor(e));
    e.qsa_history_rows = 256;
    e.qsa_history_bytes = 4096;
    try testing.expectEqual(@as(u8, 8), metaVersionFor(e));
    e.rings = @constCast(&[_]RingFile{.{ .pos = 600, .rows = 64, .bytes = 50 }});
    try testing.expectEqual(@as(u8, 9), metaVersionFor(e));
    e.vision_key = 1;
    try testing.expectEqual(@as(u8, 10), metaVersionFor(e));
    e.grown = @constCast(&[_]u64{1});
    try testing.expectEqual(@as(u8, 11), metaVersionFor(e));
}

test "manifest: v12 spec segments render, stamp v12 and parse back with their hashes past i64" {
    var e = testEntry();
    e.spec_bytes = 30;
    e.spec_mtp = .{ .base = 0, .step = 9000, .layers = 1, .quant = kv_quant.KVQuantConfig.dense };
    e.spec_segs = @constCast(&[_]SideSeg{ .{ .bytes = 100, .hash = 0xffff_0000_aaaa_5555 }, .{ .bytes = 200, .hash = 7 } });
    e.spec_segs_inherited = 1;
    try testing.expectEqual(@as(u8, 12), metaVersionFor(e));
    var out = std.ArrayList(u8).empty;
    defer out.deinit(testing.allocator);
    try renderMeta(testing.allocator, &out, e, 1024);
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, out.items, .{});
    defer parsed.deinit();
    const m = parseManifest(parsed.value.object) orelse return error.TestUnexpectedResult;
    const segs = m.spec.?.segs orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(u32, 1), segs.inherited);
    try testing.expectEqual(SideSeg{ .bytes = 100, .hash = 0xffff_0000_aaaa_5555 }, segItem(segs.files[0]).?);
    try testing.expectEqual(SideSeg{ .bytes = 200, .hash = 7 }, segItem(segs.files[1]).?);
}

test "manifest: what renderMeta writes parseManifest reads back" {
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, full_manifest, .{});
    defer parsed.deinit();
    const m = parseManifest(parsed.value.object) orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(u64, 11), m.version);
    try testing.expectEqual(@as(u64, 1), m.inherited_chunks);
    try testing.expectEqual(@as(u64, 600), m.kv_len);
    try testing.expectEqual(@as(u64, 3), m.tokens);
    try testing.expectEqual(@as(u64, 1024), m.chunk_tokens);
    try testing.expect(m.has_tools);
    try testing.expectEqual(@as(usize, 2), m.chunk_bytes.len);
    try testing.expectEqual(kv_quant.KVQuantConfig.affine(8), m.quant);
    const g = m.grown orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(u32, 8192), g.span);
    try testing.expectEqual(@as(?u32, 1), g.inherited);
    try testing.expectEqual(@as(?u64, 200), valueU64(g.bytes[1]));
    const s = ssmItem(m.ssm[0]) orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(u64, 512), s.pos);
    try testing.expectEqual(@as(u64, 777), s.bytes);
    const r = ringItem(m.ring[0]) orelse return error.TestUnexpectedResult;
    try testing.expectEqual(RingFile{ .pos = 600, .rows = 64, .bytes = 50 }, r);
    try testing.expectEqual(QsaHistoryMeta{ .bytes = 10, .rows = 5, .inherited = true }, m.qsa_history.?);
    try testing.expectEqual(MediaMeta{ .key = 0xabc, .start = 2 }, m.media.?);
    const sp = m.spec orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(u64, 30), sp.bytes);
    try testing.expectEqual(@as(u64, 1), sp.dflash.?.base);
    try testing.expectEqualSlices(i32, &.{ 64, 128 }, sp.mtp.?.head.?.marks[0..sp.mtp.?.head.?.mark_count]);
}

test "manifest: a record this build cannot restore parses to null" {
    const base = "\"kv_len\":600,\"tokens\":3,\"has_tools\":false,\"scheme\":\"off\",\"chunk_tokens\":1024,\"chunk_bytes\":[]";
    const bad = [_][]const u8{
        "{\"v\":13," ++ base ++ "}",
        "{\"v\":1," ++ base ++ "}",
        "{" ++ base ++ "}",
        "{\"v\":4,\"tokens\":3,\"has_tools\":false,\"scheme\":\"off\",\"chunk_tokens\":1024,\"chunk_bytes\":[]}",
        "{\"v\":4,\"kv_len\":600,\"tokens\":3,\"has_tools\":0,\"scheme\":\"off\",\"chunk_tokens\":1024,\"chunk_bytes\":[]}",
        "{\"v\":4," ++ base ++ ",\"media\":{\"key\":\"0\",\"start\":1}}",
        "{\"v\":4," ++ base ++ ",\"media\":{\"key\":\"ff\"}}",
        "{\"v\":4," ++ base ++ ",\"media\":{\"key\":\"ff\",\"start\":3}}",
        "{\"v\":4,\"kv_len\":600,\"tokens\":3,\"has_tools\":false,\"scheme\":\"affine\",\"bits\":5,\"group_size\":64,\"chunk_tokens\":1024,\"chunk_bytes\":[]}",
    };
    for (bad) |text| {
        var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, text, .{});
        defer parsed.deinit();
        try testing.expect(parseManifest(parsed.value.object) == null);
    }
    var ok = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{\"v\":2," ++ base ++ "}", .{});
    defer ok.deinit();
    const m = parseManifest(ok.value.object) orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(u64, 0), m.inherited_chunks);
    try testing.expect(m.grown == null and m.spec == null and m.media == null and m.qsa_history == null);
}

test "manifest: a grown section with no usable inherited count is still listed, a spec with no snapshot still has its size" {
    const text = "{\"v\":11,\"kv_len\":600,\"tokens\":3,\"has_tools\":false,\"scheme\":\"off\",\"chunk_tokens\":1024,\"chunk_bytes\":[]," ++
        "\"grown\":{\"span\":4,\"bytes\":[1]},\"spec\":{\"bytes\":9}}";
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, text, .{});
    defer parsed.deinit();
    const m = parseManifest(parsed.value.object) orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(?u32, null), m.grown.?.inherited);
    try testing.expectEqual(@as(usize, 1), m.grown.?.bytes.len);
    try testing.expectEqual(@as(u64, 9), m.spec.?.bytes);
    try testing.expect(m.spec.?.dflash == null and m.spec.?.mtp == null);
}

test "spec meta json: the v5 head half round-trips and a v4 record parses without one" {
    var w = std.ArrayList(u8).empty;
    defer w.deinit(testing.allocator);
    const sm: SpecMeta = .{
        .base = 62_000,
        .step = 700,
        .layers = 1,
        .quant = kv_quant.KVQuantConfig.dense,
        .head = .{ .pos_base = 1, .ratio = 4, .pooled = true },
    };
    try writeSpecMetaJson(testing.allocator, &w, "mtp", sm);
    const rec = try std.fmt.allocPrint(testing.allocator, "{{\"bytes\":1{s}}}", .{w.items});
    defer testing.allocator.free(rec);
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, rec, .{});
    defer parsed.deinit();
    const back = parseSpecMeta(parsed.value.object, "mtp") orelse return error.TestUnexpectedResult;
    try testing.expectEqual(sm.base, back.base);
    try testing.expectEqual(sm.step, back.step);
    const h = back.head orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(i32, 1), h.pos_base);
    try testing.expectEqual(@as(i32, 4), h.ratio);
    try testing.expect(h.pooled);

    // A v4-shaped record (no "head") parses, with the head half absent.
    const v4 = "{\"mtp\":{\"base\":5,\"step\":9,\"layers\":1,\"scheme\":\"off\",\"bits\":0,\"group_size\":0}}";
    var p4 = try std.json.parseFromSlice(std.json.Value, testing.allocator, v4, .{});
    defer p4.deinit();
    const old = parseSpecMeta(p4.value.object, "mtp") orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(u64, 5), old.base);
    try testing.expectEqual(@as(?SpecHeadMeta, null), old.head);
}

test "spec meta json: a head sidecar keeps at most HEAD_MARKS_MAX marks" {
    var text = std.ArrayList(u8).empty;
    defer text.deinit(testing.allocator);
    try text.appendSlice(testing.allocator, "{\"mtp\":{\"base\":5,\"step\":9,\"layers\":1,\"scheme\":\"off\",\"head\":{\"pos_base\":1,\"ratio\":4,\"pooled\":true,\"marks\":[");
    for (0..HEAD_MARKS_MAX + 8) |i| try text.print(testing.allocator, "{s}{d}", .{ if (i > 0) "," else "", i + 1 });
    try text.appendSlice(testing.allocator, "]}}}");
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, text.items, .{});
    defer parsed.deinit();
    const sm = parseSpecMeta(parsed.value.object, "mtp") orelse return error.TestUnexpectedResult;
    const head = sm.head orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(usize, HEAD_MARKS_MAX), head.mark_count);
    try testing.expectEqual(@as(i32, HEAD_MARKS_MAX), head.marks[HEAD_MARKS_MAX - 1]);
}

const TestTensors = struct {
    list: []const TensorDesc,

    fn len(self: TestTensors) usize {
        return self.list.len;
    }

    fn desc(self: TestTensors, i: usize) !TensorDesc {
        return self.list[i];
    }
};

test "safetensors header: the length prefix, the metadata and the offsets are pinned" {
    const a = [_]TensorDesc{
        .{ .key = "a", .dtype = "F16", .shape = &.{ 2, 3 }, .nbytes = 12 },
        .{ .key = "b.x", .dtype = "I32", .shape = &.{}, .nbytes = 4 },
    };
    const meta = [_]MetaPair{ .{ .key = "init", .value = "0" }, .{ .key = "k", .value = "v" } };
    const head = try encodeHeader(testing.allocator, TestTensors{ .list = &a }, &meta);
    defer testing.allocator.free(head);
    const json = "{\"__metadata__\":{\"init\":\"0\",\"k\":\"v\"},\"a\":{\"dtype\":\"F16\",\"shape\":[2,3],\"data_offsets\":[0,12]}," ++
        "\"b.x\":{\"dtype\":\"I32\",\"shape\":[],\"data_offsets\":[12,16]}}";
    try testing.expectEqual(@as(u64, json.len), std.mem.readInt(u64, head[0..8], .little));
    try testing.expectEqualStrings(json, head[8..]);

    const bare = try encodeHeader(testing.allocator, TestTensors{ .list = &.{} }, &.{});
    defer testing.allocator.free(bare);
    try testing.expectEqualStrings("{\"__metadata__\":{}}", bare[8..]);
    try testing.expectEqual(@as(u64, bare.len - 8), std.mem.readInt(u64, bare[0..8], .little));
}

test "safetensors header: a metadata value that needs JSON escaping is refused" {
    const bad = [_]MetaPair{.{ .key = "init", .value = "0\",\"x" }};
    try testing.expectError(error.DiskCacheBadMetadata, encodeHeader(testing.allocator, TestTensors{ .list = &.{} }, &bad));
    const ctl = [_]MetaPair{.{ .key = "k", .value = "a\nb" }};
    try testing.expectError(error.DiskCacheBadMetadata, encodeHeader(testing.allocator, TestTensors{ .list = &.{} }, &ctl));
}
