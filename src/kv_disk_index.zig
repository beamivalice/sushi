//! SSD prefix cache index lifecycle: manifest I/O, scan, poisoning, writer failure harvesting, budget/GC/cut.

const std = @import("std");
const cache_stats = @import("cache_stats.zig");
const log = @import("log.zig");
const disk_placement = @import("disk_placement.zig");
const disk_format = @import("disk_format.zig");
const kv_disk_fs = @import("kv_disk_fs.zig");
const disk = @import("kv_disk_cache.zig");

const DiskTier = disk.DiskTier;
const IndexEntry = disk.IndexEntry;
const SideSeg = disk.SideSeg;
const RingFile = disk.RingFile;
const SpecMeta = disk.SpecMeta;
const MIN_PERSIST_TOKENS = disk.MIN_PERSIST_TOKENS;
const mib = disk.mib;

/// Deletes the plain-file `e<id>` records a `real` tier left under this root: the content-addressed layer never
/// reads or frees them, and they are not converted. Runs under the root lock, before the first scan.
pub fn dropPlainRecords(self: *DiskTier) void {
    var d = std.Io.Dir.openDirAbsolute(self.io, self.root, .{ .iterate = true }) catch return;
    defer d.close(self.io);
    var names = std.ArrayList([]u8).empty;
    defer {
        for (names.items) |n| self.allocator.free(n);
        names.deinit(self.allocator);
    }
    var it = d.iterate();
    while (it.next(self.io) catch null) |dent| {
        if (dent.kind != .directory or dent.name.len < 2 or dent.name[0] != 'e') continue;
        _ = std.fmt.parseInt(u64, dent.name[1..], 10) catch continue;
        const name = self.allocator.dupe(u8, dent.name) catch continue;
        names.append(self.allocator, name) catch self.allocator.free(name);
    }
    var seen = disk_placement.InodeSet.init(self.allocator);
    defer seen.deinit();
    var removed: usize = 0;
    var freed: u64 = 0;
    for (names.items) |name| {
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const abs = std.fmt.bufPrint(&buf, "{s}/{s}", .{ self.root, name }) catch continue;
        const bytes = disk_placement.dirBytes(self.io, abs, &seen);
        d.deleteTree(self.io, name) catch continue;
        removed += 1;
        freed += bytes;
    }
    if (removed > 0) log.info("[disk-cache] removed {d} plain-file entries ({d:.1} MB) from {s}: the content-addressed layer does not read them\n", .{
        removed,
        @as(f64, @floatFromInt(freed)) / (1024.0 * 1024.0),
        self.root,
    });
}


/// Attribute the writer's failed blobs to the entries that staged them and poison those
/// entries. A counter alone misses the failure that lands between two spill passes.
pub fn harvestWriteFailures(self: *DiskTier) usize {
    const w = self.writer orelse return 0;
    const fails = w.takeFailures();
    defer {
        for (fails) |f| self.allocator.free(f.path);
        self.allocator.free(fails);
    }
    var poisoned: usize = 0;
    for (fails) |f| {
        const id = self.entryIdFromPath(f.path) orelse {
            poisoned += self.poisonAll(f.err_name);
            continue;
        };
        for (self.entries.items) |*e| {
            if (e.id != id or e.poisoned) continue;
            self.poisonEntry(e, f.err_name);
            poisoned += 1;
        }
    }
    if (w.takeUnattributed()) poisoned += self.poisonAll("unrecorded");
    return poisoned;
}

/// The `e<id>` this absolute path belongs to, or null.
pub fn entryIdFromPath(self: *const DiskTier, path: []const u8) ?u64 {
    if (!std.mem.startsWith(u8, path, self.root)) return null;
    var rest = path[self.root.len..];
    if (rest.len == 0 or rest[0] != '/') return null;
    rest = rest[1..];
    if (rest.len < 2 or rest[0] != 'e') return null;
    rest = rest[1..];
    const slash = std.mem.indexOfScalar(u8, rest, '/') orelse rest.len;
    return std.fmt.parseInt(u64, rest[0..slash], 10) catch null;
}

/// Whether entry `id` is still on the tier, unpoisoned, with a restore point at `pos`: the budget, a cache clear,
/// a poisoned read or a thinning of the entry's points can each take one away after it was written.
pub fn holdsPoint(self: *const DiskTier, id: u64, pos: usize) bool {
    for (self.entries.items) |*e| {
        if (e.id != id) continue;
        return !e.poisoned and std.mem.indexOfScalar(u32, e.ssm_positions, @intCast(pos)) != null;
    }
    return false;
}

pub fn poisonId(self: *DiskTier, id: u64, err_name: []const u8) bool {
    for (self.entries.items) |*e| {
        if (e.id != id) continue;
        if (!e.poisoned) self.poisonEntry(e, err_name);
        return true;
    }
    return false;
}

/// Mark one entry dead. `chunk_bytes` stays its billing estimate: the removal path credits what the
/// commit billed, and every whole-ness reader checks `poisoned` itself.
pub fn poisonEntry(self: *DiskTier, e: *IndexEntry, err_name: []const u8) void {
    _ = self;
    e.poisoned = true;
    log.warn("  [disk-cache] e{d} invalidated ({s})\n", .{ e.id, err_name });
}

pub fn poisonAll(self: *DiskTier, err_name: []const u8) usize {
    var n: usize = 0;
    for (self.entries.items) |*e| {
        if (e.poisoned) continue;
        self.poisonEntry(e, err_name);
        n += 1;
    }
    return n;
}

/// Reclaim the directories of poisoned entries. Runs on the commit path, never on the spill's check.
pub fn dropPoisonedEntries(self: *DiskTier) void {
    var i: usize = self.entries.items.len;
    while (i > 0) {
        i -= 1;
        if (!self.entries.items[i].poisoned) continue;
        log.info("  [disk-cache] dropping invalidated entry e{d}\n", .{self.entries.items[i].id});
        // `removeAt` swap-removes; the element moved into `i` was already checked.
        self.removeAt(i);
    }
}

pub fn chunkOnDisk(self: *DiskTier, e: *const IndexEntry, chunk: u32) bool {
    if (chunk >= e.chunk_bytes.len) return true;
    const cp = std.fmt.allocPrint(self.allocator, "{s}/e{d}/c{d:0>6}.safetensors", .{ self.root, e.id, chunk }) catch return true;
    defer self.allocator.free(cp);
    const st = self.fs.stat(cp) orelse return false;
    return st.size == e.chunk_bytes[chunk];
}

/// Does the filesystem agree with the index about entry `id`? One stat per chunk; catches
/// a byte that went missing with no failed write at all.
pub fn entryWholeOnDisk(self: *DiskTier, id: u64) bool {
    for (self.entries.items) |*e| {
        if (e.id != id) continue;
        if (e.poisoned) return false;
        for (e.chunk_bytes, 0..) |want, i| {
            if (want == 0) return false;
            const cp = std.fmt.allocPrint(self.allocator, "{s}/e{d}/c{d:0>6}.safetensors", .{ self.root, id, i }) catch return false;
            defer self.allocator.free(cp);
            const st = self.fs.stat(cp) orelse return false;
            if (st.size != want) return false;
        }
        return true;
    }
    return false;
}

/// Is entry `id` poisoned, or gone?
pub fn entryPoisoned(self: *const DiskTier, id: u64) bool {
    for (self.entries.items) |*e| {
        if (e.id == id) return e.poisoned;
    }
    return true;
}

/// Wait only for the files of entry `id`.
pub fn drainEntry(self: *DiskTier, id: u64) void {
    const w = self.writer orelse return;
    const pre = std.fmt.allocPrint(self.allocator, "{s}/e{d}/", .{ self.root, id }) catch {
        w.drain();
        return;
    };
    defer self.allocator.free(pre);
    w.drainPrefix(pre);
}


/// Free everything an IndexEntry owns. Every removal path (deinit,
/// eviction, invalidation, scan-append failure, extend-replace) must go
/// through this so a new owned field can't leak on one of them.
pub fn freeIndexEntryOwned(self: *DiskTier, e: *IndexEntry) void {
    self.allocator.free(e.tokens);
    self.allocator.free(e.chunk_bytes);
    self.allocator.free(e.ssm_positions);
    self.allocator.free(e.ssm_bytes);
    self.allocator.free(e.rings);
    self.allocator.free(e.grown);
    self.allocator.free(e.spec_segs);
    self.allocator.free(e.qsa_segs);
}


/// Re-derive `max_bytes` from the volume. A failed probe keeps the operator cap; a volume with
/// nothing left after the reserve declines new stores without touching what is already persisted.
pub fn refreshDiskBudget(self: *DiskTier) void {
    const vs = self.space_probe(self.root) orelse return;
    // Our own entries are already counted in `used`; add what the tier holds back.
    const budget = disk_placement.diskBudgetFromFreeSpace(self.operator_cap, self.fs.freeAsFiles(vs.free) +| self.total_bytes);
    if (budget) |b| {
        self.store_declined = false;
        if (b != self.max_bytes) self.max_bytes = b;
        // The cap may have moved under a resident that already holds more (a lowered `--prefix-cache-disk`,
        // `set_cap`); apply it now, not at the next allocation.
        if (self.total_bytes > self.max_bytes) self.enforceBudget();
    } else if (!self.store_declined) {
        self.store_declined = true;
        log.warn("[disk-cache] {s}: {d} MB free leaves nothing after the {d} MB reserve — no NEW entries persist (already-persisted entries stay restorable)\n", .{
            self.root,
            vs.free >> 20,
            disk_placement.DISK_FREE_RESERVE >> 20,
        });
    }
}

/// Sweep other models' fingerprints under the same base: strays always, LRU once they
/// collectively exceed one budget's worth. SSD-first only.
pub fn sweepSiblings(self: *DiskTier) void {
    if (!self.ssd_first) return;
    const base = self.base_dir orelse return;
    self.refreshDiskBudget();
    disk_placement.sweepBase(self.allocator, self.io, base, self.root, self.max_bytes);
}


/// A new entry's id, claimed by creating its `e<id>`: an existing directory is never adopted.
pub fn claimEntryId(self: *DiskTier) !u64 {
    while (true) {
        const id = self.next_id;
        self.next_id += 1;
        const dir = try std.fmt.allocPrint(self.allocator, "{s}/e{d}", .{ self.root, id });
        defer self.allocator.free(dir);
        self.fs.mkdirExclusive(dir) catch |err| switch (err) {
            error.PathAlreadyExists => continue,
            else => return err,
        };
        return id;
    }
}


/// A manifest's segment list checked against the files: null unless every one is there at its size.
pub fn scanSegs(self: *DiskTier, id: u64, letter: u8, sm: disk_format.SegsMeta) ?struct { segs: []SideSeg, inherited: u32, own_bytes: u64 } {
    const list = self.allocator.alloc(SideSeg, sm.files.len) catch return null;
    var own: u64 = 0;
    for (sm.files, list, 0..) |fv, *dst, j| {
        const sg = disk_format.segItem(fv) orelse break;
        const pp = std.fmt.allocPrint(self.allocator, "{s}/e{d}/{c}{d:0>5}.safetensors", .{ self.root, id, letter, j }) catch break;
        defer self.allocator.free(pp);
        if (self.fs.size(pp) != sg.bytes) break;
        dst.* = sg;
        if (j >= sm.inherited) own += sg.bytes;
    } else return .{ .segs = list, .inherited = @min(sm.inherited, @as(u32, @intCast(list.len))), .own_bytes = own };
    self.allocator.free(list);
    return null;
}


// ── Invalidation (mirrors the RAM cache API) ──

pub fn invalidateAll(self: *DiskTier) void {
    while (self.held.items.len > 0) self.dropHeldAt(0);
    for (self.entries.items) |*e| {
        self.deleteEntryDir(e.id);
        self.freeIndexEntryOwned(e);
    }
    self.entries.clearRetainingCapacity();
    self.total_bytes = 0;
    // An empty layer keeps only its snapshot's state record, not the journal of every delete.
    self.fs.compact();
}

pub fn invalidateNewest(self: *DiskTier) void {
    if (self.entries.items.len == 0) return;
    var newest_idx: usize = 0;
    var newest_used: u64 = 0;
    for (self.entries.items, 0..) |*e, i| {
        if (e.last_used >= newest_used) {
            newest_used = e.last_used;
            newest_idx = i;
        }
    }
    self.removeAt(newest_idx);
}


// ── Internals ──

pub fn bump(self: *DiskTier) u64 {
    self.counter += 1;
    return self.counter;
}

pub fn removeAt(self: *DiskTier, idx: usize) void {
    var e = self.entries.swapRemove(idx);
    self.dropHeldOf(e.id);
    self.total_bytes -|= self.bytesFreedByRemoving(&e);
    self.deleteEntryDir(e.id);
    self.freeIndexEntryOwned(&e);
}

/// Bytes deleting `e`'s directory returns to the volume: its non-chunk files plus every
/// chunk file nobody else links (`nlink == 1`). The filesystem is the refcount.
pub fn bytesFreedByRemoving(self: *DiskTier, e: *const IndexEntry) u64 {
    const dir_abs = std.fmt.allocPrint(self.allocator, "{s}/e{d}/", .{ self.root, e.id }) catch return e.bytes;
    defer self.allocator.free(dir_abs);
    if (self.writer) |w| w.fence(dir_abs);
    var freed: u64 = @as(u64, e.tokens.len) * 4 + e.spec_bytes;
    for ([_][]const SideSeg{ e.spec_segs, e.qsa_segs }, [_]u32{ e.spec_segs_inherited, e.qsa_segs_inherited }, "pq") |segs, inh, letter| {
        for (segs, 0..) |sg, j| {
            const sp = std.fmt.allocPrint(self.allocator, "{s}{c}{d:0>5}.safetensors", .{ dir_abs, letter, j }) catch {
                freed += sg.bytes;
                continue;
            };
            defer self.allocator.free(sp);
            if (self.fs.stat(sp)) |st| {
                if (st.nlink <= 1) freed += st.size;
            } else if (j >= inh) {
                freed += sg.bytes;
            }
        }
    }
    for (e.ssm_bytes) |b| freed += b;
    for (e.grown, 0..) |gb, i| {
        const gp = std.fmt.allocPrint(self.allocator, "{s}g{d:0>5}.safetensors", .{ dir_abs, i + 1 }) catch {
            freed += gb;
            continue;
        };
        defer self.allocator.free(gp);
        if (self.fs.stat(gp)) |st| {
            if (st.nlink <= 1) freed += st.size;
        } else if (i >= e.inherited_grown) {
            freed += gb;
        }
    }
    for (e.rings) |r| freed += r.bytes;
    if (e.qsa_history_bytes > 0) {
        if (std.fmt.allocPrint(self.allocator, "{s}qsa.safetensors", .{dir_abs})) |qp| {
            defer self.allocator.free(qp);
            if (self.fs.stat(qp)) |st| {
                if (st.nlink <= 1) freed += st.size;
            } else if (!e.inherited_qsa) {
                freed += e.qsa_history_bytes;
            }
        } else |_| {
            if (!e.inherited_qsa) freed += e.qsa_history_bytes;
        }
    }
    for (e.chunk_bytes, 0..) |cb, i| {
        const cp = std.fmt.allocPrint(self.allocator, "{s}c{d:0>6}.safetensors", .{ dir_abs, i }) catch {
            freed += cb;
            continue;
        };
        defer self.allocator.free(cp);
        if (self.fs.stat(cp)) |st| {
            if (st.nlink <= 1) freed += st.size;
        } else if (i >= e.inherited_chunks) {
            freed += cb;
        }
    }
    return freed;
}


pub fn deleteEntryDir(self: *DiskTier, id: u64) void {
    // Epoch fence, the one removal site: staged bytes for this directory are discarded.
    const dir_abs = std.fmt.allocPrint(self.allocator, "{s}/e{d}/", .{ self.root, id }) catch null;
    defer if (dir_abs) |da| self.allocator.free(da);
    if (self.writer) |w| w.fence(dir_abs);
    const dir = std.fmt.allocPrint(self.allocator, "{s}/e{d}", .{ self.root, id }) catch return;
    defer self.allocator.free(dir);
    self.fs.deleteTree(dir);
    self.fs.reclaim();
}


/// What the tier holds on the volume: its files' bytes, which on the content-addressed layer are the packs,
/// journal and snapshot.
pub fn diskBytes(self: *const DiskTier) u64 {
    return self.fs.heldBytes();
}

/// The one owner of the byte cap, called wherever a budget is applied (scan, `refreshDiskBudget`,
/// a commit): evict LRU residents, then a sole resident over the cap is cut to the highest restore
/// point that fits, or dropped whole when none does, so the tier is never left over its cap.
pub fn enforceBudget(self: *DiskTier) void {
    self.gcToBudget();
    if (self.max_bytes == 0 or self.total_bytes <= self.max_bytes) return;
    const e = &self.entries.items[0];
    // A poisoned entry cannot be restored; honouring the cap means dropping it, not keeping a cut corpse.
    if (e.poisoned) {
        log.info("  [disk-cache] dropped the last entry e{d} (byte budget; it is invalid)\n", .{e.id});
        self.removeAt(0);
        return;
    }
    // Its own commit is still writing it; the store's pre-flight refresh applies the cap first.
    if (self.entryWritesPending(e.id)) return;
    const target = self.entryCutTarget(e) orelse {
        log.info("  [disk-cache] evicted the last entry (byte budget; {d:.1} MB; no restore point fits)\n", .{mib(self.total_bytes)});
        cache_stats.process.recordEviction(.ssd, e.bytes);
        self.removeAt(0);
        return;
    };
    log.info("  [disk-cache] cut the last entry e{d} to {d} tokens for the {d:.1} MB byte budget\n", .{ e.id, target, mib(self.max_bytes) });
    self.cutEntryTo(0, target);
    if (self.total_bytes <= self.max_bytes) return;
    log.info("  [disk-cache] evicted the last entry (byte budget; {d:.1} MB; the cut left it over)\n", .{mib(self.total_bytes)});
    cache_stats.process.recordEviction(.ssd, self.entries.items[0].bytes);
    self.removeAt(0);
}

/// The highest restore point of `e` whose kept files fit `max_bytes`: a ring or checkpoint point, or a
/// chunk boundary on plain attention. Null when none does.
pub fn entryCutTarget(self: *const DiskTier, e: *const IndexEntry) ?u32 {
    var best: u32 = 0;
    if (e.rings.len > 0) {
        for (e.rings) |r| {
            if (r.pos < e.kv_len and r.pos > best and self.cutBytes(e, r.pos) <= self.max_bytes) best = r.pos;
        }
    } else if (e.ssm_positions.len > 0) {
        for (e.ssm_positions) |p| {
            if (p < e.kv_len and p > best and self.cutBytes(e, p) <= self.max_bytes) best = p;
        }
    } else {
        var k: usize = e.chunk_bytes.len;
        while (k > 0) : (k -= 1) {
            const p: u32 = @intCast(@min(@as(u64, k) * self.chunk_tokens, e.kv_len));
            if (p < e.kv_len and self.cutBytes(e, p) <= self.max_bytes) {
                best = p;
                break;
            }
        }
    }
    return if (best >= MIN_PERSIST_TOKENS) best else null;
}

/// What `e` bills after a cut to `p`: the files it keeps at the sizes it billed them, its spec state and
/// its token tail gone. Inherited (linked) files it never billed do not count; the QSA history stays whole.
pub fn cutBytes(self: *const DiskTier, e: *const IndexEntry, p: u32) u64 {
    var b: u64 = e.bytes;
    const keep: usize = (@as(usize, p) + self.chunk_tokens - 1) / self.chunk_tokens;
    for (e.chunk_bytes[keep..], keep..) |cb, i| {
        if (i >= e.inherited_chunks) b -|= cb;
    }
    for (e.ssm_positions, e.ssm_bytes) |pos, cb| {
        if (pos > p) b -|= cb;
    }
    for (e.rings) |r| {
        if (r.pos > p) b -|= r.bytes;
    }
    if (e.grown.len > 0 and e.grown_span > 0) {
        var top: u32 = 0;
        for (e.ssm_positions) |pos| {
            if (pos <= p) top = pos;
        }
        for (e.grown[top / e.grown_span ..], top / e.grown_span..) |gb, i| {
            if (i >= e.inherited_grown) b -|= gb;
        }
    }
    b -|= e.spec_bytes;
    for (e.spec_segs[e.spec_segs_inherited..]) |sg| b -|= sg.bytes;
    b -|= @as(u64, e.tokens.len - p) * 4;
    return b;
}

/// Cut entry `idx` to `p` tokens: drop every file the restore point no longer reaches, shorten the token
/// record and its manifest, and credit the volume and the tier's bill with exactly what left.
pub fn cutEntryTo(self: *DiskTier, idx: usize, p: u32) void {
    const e = &self.entries.items[idx];
    const dir_abs = std.fmt.allocPrint(self.allocator, "{s}/e{d}/", .{ self.root, e.id }) catch return;
    defer self.allocator.free(dir_abs);
    if (self.writer) |w| w.fence(dir_abs);
    self.dropHeldOf(e.id);
    var freed: u64 = 0;
    const keep: usize = (@as(usize, p) + self.chunk_tokens - 1) / self.chunk_tokens;
    var name_buf: [32]u8 = undefined;
    for (keep..e.chunk_bytes.len) |i| {
        const name = std.fmt.bufPrint(&name_buf, "c{d:0>6}.safetensors", .{i}) catch continue;
        freed += self.deleteCountFreed(dir_abs, name);
        if (i >= e.inherited_chunks) e.bytes -|= e.chunk_bytes[i];
    }
    e.chunk_bytes = self.allocator.realloc(e.chunk_bytes, keep) catch e.chunk_bytes;
    if (e.inherited_chunks > e.chunk_bytes.len) e.inherited_chunks = @intCast(e.chunk_bytes.len);
    var w: usize = 0;
    for (e.ssm_positions, e.ssm_bytes) |pos, cb| {
        if (pos > p) {
            const name = std.fmt.bufPrint(&name_buf, "s{d:0>7}.safetensors", .{pos}) catch continue;
            freed += self.deleteCountFreed(dir_abs, name);
            e.bytes -|= cb;
            continue;
        }
        e.ssm_positions[w] = pos;
        e.ssm_bytes[w] = cb;
        w += 1;
    }
    e.ssm_positions = self.allocator.realloc(e.ssm_positions, w) catch e.ssm_positions;
    e.ssm_bytes = self.allocator.realloc(e.ssm_bytes, w) catch e.ssm_bytes;
    w = 0;
    for (e.rings) |r| {
        if (r.pos > p) {
            const name = std.fmt.bufPrint(&name_buf, "r{d:0>7}.safetensors", .{r.pos}) catch continue;
            freed += self.deleteCountFreed(dir_abs, name);
            e.bytes -|= r.bytes;
            continue;
        }
        e.rings[w] = r;
        w += 1;
    }
    e.rings = self.allocator.realloc(e.rings, w) catch e.rings;
    if (e.grown.len > 0 and e.grown_span > 0) {
        const segs = p / e.grown_span;
        for (e.grown[segs..], segs..) |gb, i| {
            const name = std.fmt.bufPrint(&name_buf, "g{d:0>5}.safetensors", .{i + 1}) catch continue;
            freed += self.deleteCountFreed(dir_abs, name);
            if (i >= e.inherited_grown) e.bytes -|= gb;
        }
        e.grown = self.allocator.realloc(e.grown, segs) catch e.grown;
        if (e.inherited_grown > e.grown.len) e.inherited_grown = @intCast(e.grown.len);
    }
    // The spec describes the whole record; a cut record carries none (as the commit-time cut does).
    freed += self.deleteCountFreed(dir_abs, "spec.safetensors");
    for (e.spec_segs[e.spec_segs_inherited..], e.spec_segs_inherited..) |sg, j| {
        _ = sg;
        const name = std.fmt.bufPrint(&name_buf, "p{d:0>5}.safetensors", .{j}) catch continue;
        freed += self.deleteCountFreed(dir_abs, name);
    }
    e.bytes -|= e.spec_bytes;
    for (e.spec_segs[e.spec_segs_inherited..]) |sg| e.bytes -|= sg.bytes;
    self.allocator.free(e.spec_segs);
    e.spec_segs = &.{};
    e.spec_segs_inherited = 0;
    e.spec_bytes = 0;
    e.spec_dflash = null;
    e.spec_mtp = null;
    const tail: u64 = @as(u64, e.tokens.len - p) * 4;
    const tokens = self.allocator.dupe(u32, e.tokens[0..p]) catch return;
    self.allocator.free(e.tokens);
    e.tokens = tokens;
    e.bytes -|= tail;
    e.kv_len = p;
    // Cutting below the first media row leaves a pure-text prefix; keeping the pixel key would bar text prompts.
    if (e.media_start) |ms| if (ms >= p) {
        e.vision_key = 0;
        e.media_start = null;
    };
    e.last_used = self.bump();
    const tpath = std.fmt.allocPrint(self.allocator, "{s}tokens.bin", .{dir_abs}) catch return;
    defer self.allocator.free(tpath);
    self.fs.writeAt(tpath, true, 0, std.mem.sliceAsBytes(e.tokens), e.tokens.len * 4) catch {};
    self.total_bytes -|= freed +| tail;
    self.writeMeta(e.*) catch {};
}

/// Bytes deleting `<dir_abs><name>` returns to the volume: its size when it is the last link, else 0.
pub fn deleteCountFreed(self: *DiskTier, dir_abs: []const u8, name: []const u8) u64 {
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = std.fmt.bufPrint(&path_buf, "{s}{s}", .{ dir_abs, name }) catch return 0;
    const st = self.fs.stat(path) orelse return 0;
    self.fs.deleteFile(path);
    return if (st.nlink <= 1) st.size else 0;
}

pub fn gcToBudget(self: *DiskTier) void {
    // Files the writer replaced since the last GC left dead bytes; an eviction reclaims its own.
    self.fs.reclaim();
    if (self.max_bytes == 0) return;
    while (self.total_bytes > self.max_bytes and self.entries.items.len > 1) {
        var lru_idx: usize = 0;
        var lru_used: u64 = std.math.maxInt(u64);
        for (self.entries.items, 0..) |*e, i| {
            if (e.last_used < lru_used) {
                lru_used = e.last_used;
                lru_idx = i;
            }
        }
        const mb = @as(f64, @floatFromInt(self.entries.items[lru_idx].bytes)) / (1024.0 * 1024.0);
        log.info("  [disk-cache] evicted LRU entry (byte budget; {d:.1} MB)\n", .{mb});
        cache_stats.process.recordEviction(.ssd, self.entries.items[lru_idx].bytes);
        self.bequeathPoint(lru_idx);
        self.removeAt(lru_idx);
    }
}

pub fn writeMeta(self: *DiskTier, e: IndexEntry) !void {
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(self.allocator);
    try self.renderMeta(&buf, e);

    const final_path = try std.fmt.allocPrint(self.allocator, "{s}/e{d}/meta.json", .{ self.root, e.id });
    // One owner per branch: the staged branch hands `final_path` over on every path (a `defer`
    // does not cancel an enclosing `errdefer`, and that was a double free on ENOSPC).
    if (self.writer) |w| {
        // The index rides the same FIFO queue as the chunks, so it is the last file to land.
        // Its rendering is handed over, not copied, once the permit has room for it.
        w.waitForRoom(buf.items.len);
        const bytes = buf.toOwnedSlice(self.allocator) catch |err| {
            self.allocator.free(final_path);
            return err;
        };
        return self.fs.stageBytes(w, final_path, bytes, .meta);
    }
    defer self.allocator.free(final_path);
    const tmp_path = try std.fmt.allocPrint(self.allocator, "{s}/e{d}/meta.json.{d}.tmp", .{ self.root, e.id, std.c.getpid() });
    defer self.allocator.free(tmp_path);
    errdefer self.stats.recordFailed(buf.items.len);
    try self.fs.writeRenamed(tmp_path, final_path, buf.items);
    self.stats.recordWritten(.meta, buf.items.len);
}

pub const metaVersionFor = disk_format.metaVersionFor;

pub fn renderMeta(self: *DiskTier, out: *std.ArrayList(u8), e: IndexEntry) !void {
    return disk_format.renderMeta(self.allocator, out, e, self.chunk_tokens);
}

pub fn scan(self: *DiskTier) !void {
    const dents = self.fs.list(self.allocator, self.root) catch return;
    defer kv_disk_fs.Fs.freeEntries(self.allocator, dents);

    // Collected (entry, mtime) pairs; sorted by mtime → LRU order.
    const Pending = struct { e: IndexEntry, mtime: i128 };
    var pending = std.ArrayList(Pending).empty;
    defer pending.deinit(self.allocator);

    for (dents) |dent| {
        if (!dent.is_dir) continue;
        if (dent.name.len < 2 or dent.name[0] != 'e') continue;
        const id = std.fmt.parseInt(u64, dent.name[1..], 10) catch continue;
        // Never reuse an id that has ever existed on disk — even a
        // dropped leftover's delete could fail and leave a dirty dir.
        if (id >= self.next_id) self.next_id = id + 1;
        if (self.loadEntry(id)) |loaded| {
            pending.append(self.allocator, .{ .e = loaded.e, .mtime = loaded.mtime }) catch {
                var le = loaded.e;
                self.freeIndexEntryOwned(&le);
                continue;
            };
        } else {
            // Crash leftover / corrupt — remove it. No age bar: `root_lock` keeps every other
            // writer out of this root, so an index-less entry has no live writer.
            log.info("  [disk-cache] dropping incomplete entry e{d}\n", .{id});
            self.deleteEntryDir(id);
        }
    }

    std.mem.sort(Pending, pending.items, {}, struct {
        fn lessThan(_: void, a: Pending, b: Pending) bool {
            return a.mtime < b.mtime;
        }
    }.lessThan);

    // Shared chunk files are counted once.
    var seen = std.AutoHashMap(u64, void).init(self.allocator);
    defer seen.deinit();
    for (pending.items) |*p| {
        p.e.last_used = self.bump();
        self.billChunksOnce(&p.e, &seen);
        self.entries.append(self.allocator, p.e) catch {
            self.freeIndexEntryOwned(&p.e);
            continue;
        };
        self.total_bytes += p.e.bytes;
    }
    if (self.entries.items.len > 0) {
        log.info("  [disk-cache] scanned {d} persisted entries ({d:.1} MB) at {s}\n", .{
            self.entries.items.len,
            @as(f64, @floatFromInt(self.total_bytes)) / (1024.0 * 1024.0),
            self.root,
        });
    }
}

/// Re-bill a scanned entry's chunk files against the inodes already counted this scan: the
/// first entry to see an inode pays for it.
pub fn billChunksOnce(self: *DiskTier, e: *IndexEntry, seen: *std.AutoHashMap(u64, void)) void {
    var billed: u64 = @as(u64, e.tokens.len) * 4 + e.spec_bytes;
    for ([_][]const SideSeg{ e.spec_segs, e.qsa_segs }, "pq") |segs, letter| {
        for (segs, 0..) |sg, j| {
            const sp = std.fmt.allocPrint(self.allocator, "{s}/e{d}/{c}{d:0>5}.safetensors", .{ self.root, e.id, letter, j }) catch {
                billed += sg.bytes;
                continue;
            };
            defer self.allocator.free(sp);
            billed += billSharedFile(&self.fs, sp, sg.bytes, seen);
        }
    }
    for (e.ssm_bytes) |b| billed += b;
    for (e.grown, 0..) |gb, i| {
        const gp = std.fmt.allocPrint(self.allocator, "{s}/e{d}/g{d:0>5}.safetensors", .{ self.root, e.id, i + 1 }) catch {
            billed += gb;
            continue;
        };
        defer self.allocator.free(gp);
        billed += billSharedFile(&self.fs, gp, gb, seen);
    }
    for (e.rings) |r| billed += r.bytes;
    if (e.qsa_history_bytes > 0) {
        if (std.fmt.allocPrint(self.allocator, "{s}/e{d}/qsa.safetensors", .{ self.root, e.id })) |qp| {
            defer self.allocator.free(qp);
            if (self.fs.stat(qp)) |st| {
                if (st.nlink <= 1) {
                    billed += st.size;
                } else {
                    const ino: u64 = @intCast(st.inode);
                    if (seen.getOrPut(ino)) |gop| {
                        if (!gop.found_existing) billed += st.size;
                    } else |_| {
                        billed += st.size;
                    }
                }
            } else if (!e.inherited_qsa) {
                billed += e.qsa_history_bytes;
            }
        } else |_| {
            if (!e.inherited_qsa) billed += e.qsa_history_bytes;
        }
    }
    for (e.chunk_bytes, 0..) |cb, i| {
        const cp = std.fmt.allocPrint(self.allocator, "{s}/e{d}/c{d:0>6}.safetensors", .{ self.root, e.id, i }) catch {
            billed += cb;
            continue;
        };
        defer self.allocator.free(cp);
        billed += billSharedFile(&self.fs, cp, cb, seen);
    }
    e.bytes = billed;
}

pub fn loadEntry(self: *DiskTier, id: u64) ?struct { e: IndexEntry, mtime: i128 } {
    const meta_path = std.fmt.allocPrint(self.allocator, "{s}/e{d}/meta.json", .{ self.root, id }) catch return null;
    defer self.allocator.free(meta_path);

    const stat = self.fs.stat(meta_path) orelse return null;
    const content = self.fs.readAlloc(self.allocator, meta_path, 64 * 1024) orelse return null;
    defer self.allocator.free(content);

    var parsed = std.json.parseFromSlice(std.json.Value, self.allocator, content, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const obj = parsed.value.object;

    // v2 = pure-attention (no ssm field); v3 adds SSM checkpoints; v4
    // adds optional spec snapshots; v5 the in-checkpoint MTP head's QSA half; v6 inherited
    // chunks; v9 ring restore points; v10 the media identity; v11 row segments. All restore; a
    // lower-version entry just carries none of the newer state, and a pre-v10 one holds no media row.
    const man = disk_format.parseManifest(obj) orelse return null;
    // v6: the leading `inherited_chunks` chunk files are hard links into a donor's.
    const inherited_rec = man.inherited_chunks;
    var kv_len = man.kv_len;
    const n_tokens = man.tokens;
    const chunk_tokens = man.chunk_tokens;
    const chunk_bytes_items = man.chunk_bytes;

    // Chunk geometry must match this tier's configuration — a stale root
    // written under a different chunk size can't be extended coherently.
    if (chunk_tokens != self.chunk_tokens) return null;
    if (kv_len == 0 or kv_len > n_tokens) return null;

    const n_chunks: u64 = (kv_len + chunk_tokens - 1) / chunk_tokens;
    if (chunk_bytes_items.len != n_chunks) return null;

    // Validate each chunk file's size against the recorded one. A kill -9
    // mid-flush truncates the chunk being (re)written while meta still
    // describes the previous valid state — restoring it would poison the
    // cache (live: MLX "invalid data offsets exceeding the size of the
    // file"). Clamp to the last contiguous valid chunk and salvage the
    // prefix.
    var valid_chunks: u64 = 0;
    while (valid_chunks < n_chunks) : (valid_chunks += 1) {
        const want = disk_format.valueU64(chunk_bytes_items[@intCast(valid_chunks)]) orelse break;
        const cp = std.fmt.allocPrint(self.allocator, "{s}/e{d}/c{d:0>6}.safetensors", .{ self.root, id, valid_chunks }) catch return null;
        defer self.allocator.free(cp);
        const have = self.fs.size(cp) orelse break;
        if (have != want) break;
    }
    if (valid_chunks < n_chunks) {
        const salvaged = valid_chunks * chunk_tokens;
        log.info("  [disk-cache] e{d}: chunk {d} invalid — salvaging {d}/{d} tokens\n", .{ id, valid_chunks, salvaged, kv_len });
        kv_len = salvaged;
        if (kv_len < MIN_PERSIST_TOKENS) return null;
    }

    const chunk_bytes = self.allocator.alloc(u64, @intCast(valid_chunks)) catch return null;
    for (chunk_bytes, 0..) |*cb, i| cb.* = disk_format.valueU64(chunk_bytes_items[i]).?;

    // Token record.
    const tokens_path = std.fmt.allocPrint(self.allocator, "{s}/e{d}/tokens.bin", .{ self.root, id }) catch {
        self.allocator.free(chunk_bytes);
        return null;
    };
    defer self.allocator.free(tokens_path);
    const raw = self.fs.readAlloc(self.allocator, tokens_path, 64 * 1024 * 1024) orelse {
        self.allocator.free(chunk_bytes);
        return null;
    };
    defer self.allocator.free(raw);
    // Longer is an extend's append that died before its manifest landed: the first `n_tokens` are the record.
    if (raw.len < n_tokens * 4) {
        self.allocator.free(chunk_bytes);
        return null;
    }
    const tokens = self.allocator.alloc(u32, n_tokens) catch {
        self.allocator.free(chunk_bytes);
        return null;
    };
    for (tokens, 0..) |*t, i| {
        t.* = std.mem.readInt(u32, raw[i * 4 ..][0..4], .little);
    }

    const inherited: u32 = @intCast(@min(inherited_rec, chunk_bytes.len));
    var total: u64 = @as(u64, tokens.len) * 4;
    for (chunk_bytes[inherited..]) |cb| total += cb;

    // v11 row segments, kept up to the first one whose file is missing or the wrong size: a restore point
    // leaning on a later segment is dropped below.
    var grown_recorded: []const std.json.Value = &.{};
    var grown_valid: usize = 0;
    var grown_span: u32 = 0;
    var grown_rec_span: u32 = 1;
    var grown_inherited: u32 = 0;
    if (man.grown) |gm| grown_blk: {
        grown_recorded = gm.bytes;
        grown_rec_span = gm.span;
        grown_inherited = gm.inherited orelse break :grown_blk;
        for (grown_recorded, 1..) |bv, j| {
            const want = disk_format.valueU64(bv) orelse break;
            const gp = std.fmt.allocPrint(self.allocator, "{s}/e{d}/g{d:0>5}.safetensors", .{ self.root, id, j }) catch break;
            defer self.allocator.free(gp);
            const have = self.fs.size(gp) orelse break;
            if (have != want) break;
            grown_valid = j;
            total += have;
        }
        if (grown_valid > 0) grown_span = gm.span;
    }

    // v3 SSM checkpoints (v2 entries have no "ssm" field → pure-attention,
    // stays empty). Each file's size is validated against the recorded one
    // — the same kill -9 salvage as chunks: a position whose file mismatches
    // (or now sits beyond a salvaged-down kv_len) is dropped individually.
    var ssm_positions: []u32 = &[_]u32{};
    var ssm_bytes: []u64 = &[_]u64{};
    const had_ssm_listed = man.ssm.len > 0;
    var pos_list = std.ArrayList(u32).empty;
    defer pos_list.deinit(self.allocator);
    var byte_list = std.ArrayList(u64).empty;
    defer byte_list.deinit(self.allocator);
    for (man.ssm) |it_v| {
        const item = disk_format.ssmItem(it_v) orelse continue;
        const pos = item.pos;
        const szrec = item.bytes;
        if (pos == 0 or pos > kv_len) continue; // beyond the salvaged KV → unusable
        if (grown_recorded.len > 0 and pos / grown_rec_span > grown_valid) continue; // its rows are in a lost segment
        const sp = std.fmt.allocPrint(self.allocator, "{s}/e{d}/s{d:0>7}.safetensors", .{ self.root, id, pos }) catch continue;
        defer self.allocator.free(sp);
        const have = self.fs.size(sp) orelse continue;
        if (have != szrec) continue; // truncated mid-flush — drop this position
        pos_list.append(self.allocator, @intCast(pos)) catch continue;
        byte_list.append(self.allocator, szrec) catch {
            _ = pos_list.pop();
            continue;
        };
    }
    if (pos_list.items.len > 0) {
        // meta lists positions ascending, but re-sort defensively so
        // highestSsmPosAtOrBelow / retention can trust the order.
        const Pair = struct { pos: u32, bytes: u64 };
        const pairs = self.allocator.alloc(Pair, pos_list.items.len) catch {
            self.allocator.free(tokens);
            self.allocator.free(chunk_bytes);
            return null;
        };
        defer self.allocator.free(pairs);
        for (pairs, 0..) |*pr, i| pr.* = .{ .pos = pos_list.items[i], .bytes = byte_list.items[i] };
        std.mem.sort(Pair, pairs, {}, struct {
            fn lt(_: void, a: Pair, b: Pair) bool {
                return a.pos < b.pos;
            }
        }.lt);
        const sp_arr = self.allocator.alloc(u32, pairs.len) catch {
            self.allocator.free(tokens);
            self.allocator.free(chunk_bytes);
            return null;
        };
        const sb_arr = self.allocator.alloc(u64, pairs.len) catch {
            self.allocator.free(sp_arr);
            self.allocator.free(tokens);
            self.allocator.free(chunk_bytes);
            return null;
        };
        for (pairs, 0..) |pr, i| {
            sp_arr[i] = pr.pos;
            sb_arr[i] = pr.bytes;
            total += pr.bytes;
        }
        ssm_positions = sp_arr;
        ssm_bytes = sb_arr;
    }
    // A hybrid entry (SSM listed in meta) whose checkpoints ALL failed
    // validation is unusable — KV without any SSM state can't restore a
    // recurrent arch (the RAM path resets to cold in that case too). Drop
    // it wholesale.
    if (had_ssm_listed and ssm_positions.len == 0) {
        log.info("  [disk-cache] e{d}: all SSM checkpoints invalid — dropping hybrid entry\n", .{id});
        self.allocator.free(tokens);
        self.allocator.free(chunk_bytes);
        return null;
    }

    // v9 ring restore points, salvaged per file like the SSM checkpoints. A ringed entry whose
    // files all fail restores nothing (its chunks lack the ringed layers): dropped wholesale.
    var rings: []RingFile = &.{};
    if (man.ring.len > 0) {
        var list = std.ArrayList(RingFile).empty;
        defer list.deinit(self.allocator);
        for (man.ring) |it_v| {
            const item = disk_format.ringItem(it_v) orelse continue;
            const pos = item.pos;
            const rows = item.rows;
            const szrec = item.bytes;
            if (pos > kv_len or rows == 0 or rows > pos) continue;
            const rp = std.fmt.allocPrint(self.allocator, "{s}/e{d}/r{d:0>7}.safetensors", .{ self.root, id, pos }) catch continue;
            defer self.allocator.free(rp);
            const have = self.fs.size(rp) orelse continue;
            if (have != szrec) continue; // truncated mid-flush — drop this position
            list.append(self.allocator, .{ .pos = pos, .rows = rows, .bytes = szrec }) catch continue;
        }
        std.mem.sort(RingFile, list.items, {}, struct {
            fn lt(_: void, a: RingFile, b: RingFile) bool {
                return a.pos < b.pos;
            }
        }.lt);
        rings = if (list.items.len == 0) &.{} else list.toOwnedSlice(self.allocator) catch &.{};
        if (rings.len == 0) {
            log.info("  [disk-cache] e{d}: no valid ring file — dropping ringed entry\n", .{id});
            self.allocator.free(tokens);
            self.allocator.free(chunk_bytes);
            self.allocator.free(ssm_positions);
            self.allocator.free(ssm_bytes);
            return null;
        }
        for (rings) |r| total += r.bytes;
    }

    var qsa_history_bytes: u64 = 0;
    var qsa_history_rows: u32 = 0;
    var inherited_qsa = false;
    var qsa_segs: []SideSeg = &.{};
    var qsa_segs_inherited: u32 = 0;
    if (man.qsa_history) |qh| qh_blk: {
        const rec_bytes = qh.bytes;
        const rec_rows = qh.rows;
        inherited_qsa = qh.inherited;
        const qp = std.fmt.allocPrint(self.allocator, "{s}/e{d}/qsa.safetensors", .{ self.root, id }) catch break :qh_blk;
        defer self.allocator.free(qp);
        const have = self.fs.size(qp) orelse break :qh_blk;
        if (have != rec_bytes) {
            log.info("  [disk-cache] e{d}: qsa history size mismatch — dropping the history\n", .{id});
            break :qh_blk;
        }
        if (qh.segs) |sm| {
            const got = self.scanSegs(id, 'q', sm) orelse {
                log.info("  [disk-cache] e{d}: qsa history segment missing or short — dropping the history\n", .{id});
                break :qh_blk;
            };
            qsa_segs = got.segs;
            qsa_segs_inherited = got.inherited;
            total += got.own_bytes;
        }
        qsa_history_bytes = rec_bytes;
        qsa_history_rows = rec_rows;
        if (!inherited_qsa) total += rec_bytes;
    }

    // v4 spec snapshots: validated by recorded file size (kill -9
    // salvage), but a bad spec drops only the SPEC — a restore then
    // starts blind, which is today's v2/v3 behavior anyway.
    var spec_bytes: u64 = 0;
    var spec_dflash: ?SpecMeta = null;
    var spec_mtp: ?SpecMeta = null;
    var spec_segs: []SideSeg = &.{};
    var spec_segs_inherited: u32 = 0;
    if (man.spec) |sc| parse_spec: {
        const rec_bytes = sc.bytes;
        // Spare segments a spec-less commit kept: no sidecar, no spec.
        const spare = rec_bytes == 0;
        if (!spare) {
            const sp = std.fmt.allocPrint(self.allocator, "{s}/e{d}/spec.safetensors", .{ self.root, id }) catch break :parse_spec;
            defer self.allocator.free(sp);
            const have = self.fs.size(sp) orelse break :parse_spec;
            if (have != rec_bytes) {
                log.info("  [disk-cache] e{d}: spec sidecar size mismatch — dropping the spec\n", .{id});
                break :parse_spec;
            }
        }
        // Every listed segment must be whole, or the history is not.
        var seg_total: u64 = 0;
        if (sc.segs) |sm| {
            const got = self.scanSegs(id, 'p', sm) orelse {
                log.info("  [disk-cache] e{d}: spec segment missing or short — dropping the spec\n", .{id});
                break :parse_spec;
            };
            spec_segs = got.segs;
            spec_segs_inherited = got.inherited;
            seg_total = got.own_bytes;
        }
        if (spare) {
            total += seg_total;
            break :parse_spec;
        }
        spec_dflash = sc.dflash;
        spec_mtp = sc.mtp;
        if (spec_dflash == null and spec_mtp == null) {
            self.allocator.free(spec_segs);
            spec_segs = &.{};
            break :parse_spec;
        }
        spec_bytes = rec_bytes;
        total += rec_bytes + seg_total;
    }

    const vision_key: u64 = if (man.media) |md| md.key else 0;
    const media_start: ?u32 = if (man.media) |md| md.start else null;

    const grown = self.allocator.alloc(u64, grown_valid) catch {
        self.allocator.free(tokens);
        self.allocator.free(chunk_bytes);
        self.allocator.free(ssm_positions);
        self.allocator.free(ssm_bytes);
        self.allocator.free(rings);
        self.allocator.free(spec_segs);
        self.allocator.free(qsa_segs);
        return null;
    };
    for (grown, grown_recorded[0..grown_valid]) |*g, rec| g.* = disk_format.valueU64(rec).?;

    return .{
        .e = .{
            .id = id,
            .tokens = tokens,
            .grown = grown,
            .grown_span = grown_span,
            .inherited_grown = @min(grown_inherited, @as(u32, @intCast(grown_valid))),
            .kv_len = @intCast(kv_len),
            .has_tools = man.has_tools,
            .vision_key = vision_key,
            .media_start = media_start,
            .quant = man.quant,
            .bytes = total,
            .inherited_chunks = inherited,
            .chunk_bytes = chunk_bytes,
            .ssm_positions = ssm_positions,
            .ssm_bytes = ssm_bytes,
            .spec_bytes = spec_bytes,
            .spec_dflash = spec_dflash,
            .spec_mtp = spec_mtp,
            .spec_segs = spec_segs,
            .spec_segs_inherited = spec_segs_inherited,
            .qsa_history_bytes = qsa_history_bytes,
            .qsa_history_rows = qsa_history_rows,
            .inherited_qsa = inherited_qsa,
            .qsa_segs = qsa_segs,
            .qsa_segs_inherited = qsa_segs_inherited,
            .rings = rings,
            .last_used = 0,
        },
        .mtime = stat.mtime_ns,
    };
}

/// What a scanned file adds to the bill: its size the first time its inode is seen, 0 after (hard links), `missing` if gone.
fn billSharedFile(fs: *const kv_disk_fs.Fs, path: []const u8, missing: u64, seen: *std.AutoHashMap(u64, void)) u64 {
    const st = fs.stat(path) orelse return missing;
    if (st.nlink <= 1) return st.size;
    const gop = seen.getOrPut(@intCast(st.inode)) catch return st.size;
    return if (gop.found_existing) 0 else st.size;
}

