const std = @import("std");
const tok = @import("mlx_host").tokenizer;

pub const window_len: usize = 2048;
pub const wikitext_url = "https://wikitext.smerity.com/wikitext-103-raw-v1.zip";
pub const wikitext_sha256 = "91c00ae287f0d699e18605c84afc9e45c192bc6b7797ff8837e5474655a33794";
pub const wikitext_entry = "wikitext-103-raw/wiki.train.raw";
pub const wikitext_cache_name = "wikitext-103-raw-v1.zip";
pub const code_text = @embedFile("calib/code.txt");

pub const Error = error{
    CalibTooShort,
    WikitextChecksumMismatch,
    WikitextFetchFailed,
    WikitextEntryMissing,
    WikitextUnsupportedCompression,
};

pub const Windows = struct {
    allocator: std.mem.Allocator,
    ids: []u32,
    count: usize,
    sha256: [64]u8,

    pub fn deinit(self: *Windows) void {
        self.allocator.free(self.ids);
    }

    pub fn window(self: *const Windows, i: usize) []const u32 {
        return self.ids[i * window_len ..][0..window_len];
    }

    pub fn tokens(self: *const Windows) usize {
        return self.count * window_len;
    }
};

fn idsDigest(ids: []const u32) [64]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (ids) |id| {
        var le: [4]u8 = undefined;
        std.mem.writeInt(u32, &le, id, .little);
        h.update(&le);
    }
    var raw: [32]u8 = undefined;
    h.final(&raw);
    return std.fmt.bytesToHex(raw, .lower);
}

pub fn assemble(allocator: std.mem.Allocator, wiki_ids: []const u32, code_ids: []const u32, total_tokens: usize) !Windows {
    const total = total_tokens / window_len;
    if (total == 0) return error.CalibTooShort;
    var wiki_n = (total + 1) / 2;
    var code_n = total / 2;
    const code_have = code_ids.len / window_len;
    if (code_n > code_have) {
        wiki_n += code_n - code_have;
        code_n = code_have;
    }
    if (wiki_n > wiki_ids.len / window_len) return error.CalibTooShort;
    const ids = try allocator.alloc(u32, (wiki_n + code_n) * window_len);
    errdefer allocator.free(ids);
    @memcpy(ids[0 .. wiki_n * window_len], wiki_ids[0 .. wiki_n * window_len]);
    @memcpy(ids[wiki_n * window_len ..], code_ids[0 .. code_n * window_len]);
    return .{ .allocator = allocator, .ids = ids, .count = wiki_n + code_n, .sha256 = idsDigest(ids) };
}

pub fn fromTokens(allocator: std.mem.Allocator, ids: []const u32, total_tokens: usize) !Windows {
    const total = @min(total_tokens, ids.len) / window_len;
    if (total == 0) return error.CalibTooShort;
    const owned = try allocator.dupe(u32, ids[0 .. total * window_len]);
    return .{ .allocator = allocator, .ids = owned, .count = total, .sha256 = idsDigest(owned) };
}

pub const Fetcher = struct {
    ctx: ?*anyopaque = null,
    fetch: *const fn (ctx: ?*anyopaque, io: std.Io, url: []const u8, dest_path: []const u8) anyerror!void,
};

pub fn curlFetch(_: ?*anyopaque, io: std.Io, url: []const u8, dest_path: []const u8) anyerror!void {
    var child = std.process.spawn(io, .{
        .argv = &.{ "curl", "-fL", "--create-dirs", "-sS", "-o", dest_path, url },
        .stdin = .ignore,
        .stdout = .ignore,
        .stderr = .inherit,
    }) catch return error.WikitextFetchFailed;
    const term = child.wait(io) catch return error.WikitextFetchFailed;
    switch (term) {
        .exited => |code| if (code != 0) return error.WikitextFetchFailed,
        else => return error.WikitextFetchFailed,
    }
}

fn fileSha256(io: std.Io, dir: std.Io.Dir, name: []const u8) !?[64]u8 {
    var file = dir.openFile(io, name, .{}) catch return null;
    defer file.close(io);
    var buf: [1 << 16]u8 = undefined;
    var reader = file.reader(io, &buf);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var chunk: [1 << 16]u8 = undefined;
    while (true) {
        const n = reader.interface.readSliceShort(&chunk) catch return error.ReadFailed;
        if (n == 0) break;
        h.update(chunk[0..n]);
    }
    var raw: [32]u8 = undefined;
    h.final(&raw);
    return std.fmt.bytesToHex(raw, .lower);
}

pub fn ensureWikitext(
    allocator: std.mem.Allocator,
    io: std.Io,
    cache_dir: []const u8,
    expect_sha256: []const u8,
    fetcher: Fetcher,
) ![]u8 {
    const cwd = std.Io.Dir.cwd();
    try cwd.createDirPath(io, cache_dir);
    var dir = try cwd.openDir(io, cache_dir, .{});
    defer dir.close(io);
    const final = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ cache_dir, wikitext_cache_name });
    errdefer allocator.free(final);
    if (try fileSha256(io, dir, wikitext_cache_name)) |have| {
        if (std.mem.eql(u8, &have, expect_sha256)) return final;
        dir.deleteFile(io, wikitext_cache_name) catch {};
    }
    const partial = try std.fmt.allocPrint(allocator, "{s}.partial", .{final});
    defer allocator.free(partial);
    const partial_name = try std.fmt.allocPrint(allocator, "{s}.partial", .{wikitext_cache_name});
    defer allocator.free(partial_name);
    dir.deleteFile(io, partial_name) catch {};
    try fetcher.fetch(fetcher.ctx, io, wikitext_url, partial);
    const got = (try fileSha256(io, dir, partial_name)) orelse return error.WikitextFetchFailed;
    if (!std.mem.eql(u8, &got, expect_sha256)) {
        dir.deleteFile(io, partial_name) catch {};
        return error.WikitextChecksumMismatch;
    }
    try dir.rename(partial_name, dir, wikitext_cache_name, io);
    return final;
}

pub fn zipEntryPrefix(allocator: std.mem.Allocator, io: std.Io, zip_path: []const u8, entry_name: []const u8, max_bytes: usize) ![]u8 {
    var file = try std.Io.Dir.cwd().openFile(io, zip_path, .{});
    defer file.close(io);
    var read_buf: [4096]u8 = undefined;
    var reader = file.reader(io, &read_buf);
    var it = try std.zip.Iterator.init(&reader);
    var name_buf: [512]u8 = undefined;
    while (try it.next()) |entry| {
        const name = try entry.getFilename(&reader, &name_buf, .{});
        if (!std.mem.eql(u8, name, entry_name)) continue;
        try reader.seekTo(entry.file_offset);
        const local = try reader.interface.takeStruct(std.zip.LocalFileHeader, .little);
        if (!std.mem.eql(u8, &local.signature, &std.zip.local_file_header_sig)) return error.ZipBadFileOffset;
        const data_at = entry.file_offset + @sizeOf(std.zip.LocalFileHeader) + local.filename_len + local.extra_len;
        try reader.seekTo(data_at);
        const want: usize = @intCast(@min(@as(u64, max_bytes), entry.uncompressed_size));
        const out = try allocator.alloc(u8, want);
        errdefer allocator.free(out);
        switch (entry.compression_method) {
            .store => try reader.interface.readSliceAll(out),
            .deflate => {
                var window: [std.compress.flate.max_window_len]u8 = undefined;
                var d: std.compress.flate.Decompress = .init(&reader.interface, .raw, &window);
                try d.reader.readSliceAll(out);
            },
            else => return error.WikitextUnsupportedCompression,
        }
        return out;
    }
    return error.WikitextEntryMissing;
}

fn upToLastNewline(bytes: []u8) []u8 {
    const at = std.mem.lastIndexOfScalar(u8, bytes, '\n') orelse return bytes;
    return bytes[0 .. at + 1];
}

pub fn tokenizeText(allocator: std.mem.Allocator, tokenizer: *const tok.Tokenizer, text: []const u8) ![]u32 {
    return tokenizer.encode(allocator, text);
}

pub fn defaultWindows(
    allocator: std.mem.Allocator,
    io: std.Io,
    tokenizer: *const tok.Tokenizer,
    cache_dir: []const u8,
    total_tokens: usize,
    fetcher: Fetcher,
) !Windows {
    const zip_path = try ensureWikitext(allocator, io, cache_dir, wikitext_sha256, fetcher);
    defer allocator.free(zip_path);
    const want_wiki_tokens = (total_tokens + 1) / 2 + window_len;
    const raw = try zipEntryPrefix(allocator, io, zip_path, wikitext_entry, want_wiki_tokens * 8);
    defer allocator.free(raw);
    const wiki_ids = try tokenizeText(allocator, tokenizer, upToLastNewline(raw));
    defer allocator.free(wiki_ids);
    const code_ids = try tokenizeText(allocator, tokenizer, code_text);
    defer allocator.free(code_ids);
    return assemble(allocator, wiki_ids, code_ids, total_tokens);
}

pub fn fileWindows(allocator: std.mem.Allocator, io: std.Io, tokenizer: *const tok.Tokenizer, path: []const u8, total_tokens: usize) !Windows {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1 << 30));
    defer allocator.free(text);
    const ids = try tokenizeText(allocator, tokenizer, text);
    defer allocator.free(ids);
    return fromTokens(allocator, ids, total_tokens);
}

fn tinyIds(allocator: std.mem.Allocator, n: usize, base: u32) ![]u32 {
    const ids = try allocator.alloc(u32, n);
    for (ids, 0..) |*v, i| v.* = base + @as(u32, @intCast(i));
    return ids;
}

test "calib: windows split the tokens half and half, spill to prose when code runs out, and hash the ids" {
    const t = std.testing;
    const wiki = try tinyIds(t.allocator, window_len * 10, 0);
    defer t.allocator.free(wiki);
    const code = try tinyIds(t.allocator, window_len * 3 + 5, 1_000_000);
    defer t.allocator.free(code);

    var even = try assemble(t.allocator, wiki, code, window_len * 4);
    defer even.deinit();
    try t.expectEqual(@as(usize, 4), even.count);
    try t.expectEqual(@as(u32, 0), even.window(0)[0]);
    try t.expectEqual(@as(u32, window_len), even.window(1)[0]);
    try t.expectEqual(@as(u32, 1_000_000), even.window(2)[0]);
    try t.expectEqual(@as(usize, window_len * 4), even.tokens());

    var spill = try assemble(t.allocator, wiki, code, window_len * 9);
    defer spill.deinit();
    try t.expectEqual(@as(usize, 9), spill.count);
    try t.expectEqual(@as(u32, 1_000_000), spill.window(6)[0]);
    try t.expectEqual(@as(u32, 1_000_000 + 2 * window_len), spill.window(8)[0]);

    var again = try assemble(t.allocator, wiki, code, window_len * 4);
    defer again.deinit();
    try t.expectEqualSlices(u8, &even.sha256, &again.sha256);
    try t.expect(!std.mem.eql(u8, &even.sha256, &spill.sha256));

    try t.expectError(error.CalibTooShort, assemble(t.allocator, wiki, code, window_len - 1));
    try t.expectError(error.CalibTooShort, assemble(t.allocator, wiki[0 .. window_len * 2], code, window_len * 12));

    var capped = try fromTokens(t.allocator, wiki, window_len * 100);
    defer capped.deinit();
    try t.expectEqual(@as(usize, 10), capped.count);
    try t.expectError(error.CalibTooShort, fromTokens(t.allocator, wiki[0 .. window_len - 1], window_len));
}

const FakeFetch = struct {
    body: []const u8,
    calls: usize = 0,

    fn fetch(ctx: ?*anyopaque, io: std.Io, _: []const u8, dest: []const u8) anyerror!void {
        const self: *FakeFetch = @ptrCast(@alignCast(ctx.?));
        self.calls += 1;
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = dest, .data = self.body });
    }

    fn fetcher(self: *FakeFetch) Fetcher {
        return .{ .ctx = self, .fetch = fetch };
    }
};

fn hexOf(bytes: []const u8) [64]u8 {
    var raw: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &raw, .{});
    return std.fmt.bytesToHex(raw, .lower);
}

test "calib: the WikiText archive is fetched once, checked against its sha256, and a bad copy is refused or replaced" {
    const t = std.testing;
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &path_buf);
    const cache = try std.fmt.allocPrint(t.allocator, "{s}/cache", .{path_buf[0..n]});
    defer t.allocator.free(cache);

    const good = "pretend archive";
    const sha = hexOf(good);
    var fake = FakeFetch{ .body = good };
    const first = try ensureWikitext(t.allocator, io, cache, &sha, fake.fetcher());
    defer t.allocator.free(first);
    try t.expectEqual(@as(usize, 1), fake.calls);
    const second = try ensureWikitext(t.allocator, io, cache, &sha, fake.fetcher());
    defer t.allocator.free(second);
    try t.expectEqual(@as(usize, 1), fake.calls);

    var cache_dir = try tmp.dir.openDir(io, "cache", .{});
    defer cache_dir.close(io);
    try cache_dir.writeFile(io, .{ .sub_path = wikitext_cache_name, .data = "corrupt" });
    const third = try ensureWikitext(t.allocator, io, cache, &sha, fake.fetcher());
    defer t.allocator.free(third);
    try t.expectEqual(@as(usize, 2), fake.calls);

    var bad = FakeFetch{ .body = "tampered" };
    try cache_dir.deleteFile(io, wikitext_cache_name);
    try t.expectError(error.WikitextChecksumMismatch, ensureWikitext(t.allocator, io, cache, &sha, bad.fetcher()));
    try t.expectError(error.FileNotFound, cache_dir.access(io, wikitext_cache_name, .{}));
    try t.expectError(error.FileNotFound, cache_dir.access(io, wikitext_cache_name ++ ".partial", .{}));
}

fn zipWith(allocator: std.mem.Allocator, name: []const u8, payload: []const u8, uncompressed: usize, method: u16) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    const crc = std.hash.Crc32.hash(payload[if (method == 8) 5 else 0..]);
    try out.appendSlice(allocator, &std.zip.local_file_header_sig);
    try out.appendSlice(allocator, &std.mem.toBytes(@as(u16, 20)));
    try out.appendSlice(allocator, &std.mem.toBytes(@as(u16, 0)));
    try out.appendSlice(allocator, &std.mem.toBytes(method));
    try out.appendSlice(allocator, &std.mem.toBytes(@as(u16, 0)));
    try out.appendSlice(allocator, &std.mem.toBytes(@as(u16, 0x21)));
    try out.appendSlice(allocator, &std.mem.toBytes(crc));
    try out.appendSlice(allocator, &std.mem.toBytes(@as(u32, @intCast(payload.len))));
    try out.appendSlice(allocator, &std.mem.toBytes(@as(u32, @intCast(uncompressed))));
    try out.appendSlice(allocator, &std.mem.toBytes(@as(u16, @intCast(name.len))));
    try out.appendSlice(allocator, &std.mem.toBytes(@as(u16, 0)));
    try out.appendSlice(allocator, name);
    try out.appendSlice(allocator, payload);
    const central_at = out.items.len;
    try out.appendSlice(allocator, &std.zip.central_file_header_sig);
    try out.appendSlice(allocator, &std.mem.toBytes(@as(u16, 20)));
    try out.appendSlice(allocator, &std.mem.toBytes(@as(u16, 20)));
    try out.appendSlice(allocator, &std.mem.toBytes(@as(u16, 0)));
    try out.appendSlice(allocator, &std.mem.toBytes(method));
    try out.appendSlice(allocator, &std.mem.toBytes(@as(u16, 0)));
    try out.appendSlice(allocator, &std.mem.toBytes(@as(u16, 0x21)));
    try out.appendSlice(allocator, &std.mem.toBytes(crc));
    try out.appendSlice(allocator, &std.mem.toBytes(@as(u32, @intCast(payload.len))));
    try out.appendSlice(allocator, &std.mem.toBytes(@as(u32, @intCast(uncompressed))));
    try out.appendSlice(allocator, &std.mem.toBytes(@as(u16, @intCast(name.len))));
    try out.appendSlice(allocator, &std.mem.toBytes(@as(u16, 0)));
    try out.appendSlice(allocator, &std.mem.toBytes(@as(u16, 0)));
    try out.appendSlice(allocator, &std.mem.toBytes(@as(u16, 0)));
    try out.appendSlice(allocator, &std.mem.toBytes(@as(u16, 0)));
    try out.appendSlice(allocator, &std.mem.toBytes(@as(u32, 0)));
    try out.appendSlice(allocator, &std.mem.toBytes(@as(u32, 0)));
    try out.appendSlice(allocator, name);
    const central_len = out.items.len - central_at;
    try out.appendSlice(allocator, &std.zip.end_record_sig);
    try out.appendSlice(allocator, &std.mem.toBytes(@as(u16, 0)));
    try out.appendSlice(allocator, &std.mem.toBytes(@as(u16, 0)));
    try out.appendSlice(allocator, &std.mem.toBytes(@as(u16, 1)));
    try out.appendSlice(allocator, &std.mem.toBytes(@as(u16, 1)));
    try out.appendSlice(allocator, &std.mem.toBytes(@as(u32, @intCast(central_len))));
    try out.appendSlice(allocator, &std.mem.toBytes(@as(u32, @intCast(central_at))));
    try out.appendSlice(allocator, &std.mem.toBytes(@as(u16, 0)));
    return out.toOwnedSlice(allocator);
}

test "calib: the first bytes of a stored and of a deflated zip entry are read without extracting the rest" {
    const t = std.testing;
    const io = t.io;
    const text = "alpha beta\ngamma delta\nepsilon";
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &path_buf);

    const stored = try zipWith(t.allocator, "wikitext-103-raw/wiki.train.raw", text, text.len, 0);
    defer t.allocator.free(stored);
    try tmp.dir.writeFile(io, .{ .sub_path = "stored.zip", .data = stored });
    const stored_path = try std.fmt.allocPrint(t.allocator, "{s}/stored.zip", .{path_buf[0..n]});
    defer t.allocator.free(stored_path);
    const head = try zipEntryPrefix(t.allocator, io, stored_path, wikitext_entry, 14);
    defer t.allocator.free(head);
    try t.expectEqualStrings("alpha beta\ngam", head);
    const whole = try zipEntryPrefix(t.allocator, io, stored_path, wikitext_entry, 1000);
    defer t.allocator.free(whole);
    try t.expectEqualStrings(text, whole);

    var block: [5 + text.len]u8 = undefined;
    block[0] = 0x01;
    std.mem.writeInt(u16, block[1..3], text.len, .little);
    std.mem.writeInt(u16, block[3..5], ~@as(u16, text.len), .little);
    @memcpy(block[5..], text);
    const deflated = try zipWith(t.allocator, "wikitext-103-raw/wiki.train.raw", &block, text.len, 8);
    defer t.allocator.free(deflated);
    try tmp.dir.writeFile(io, .{ .sub_path = "deflated.zip", .data = deflated });
    const deflated_path = try std.fmt.allocPrint(t.allocator, "{s}/deflated.zip", .{path_buf[0..n]});
    defer t.allocator.free(deflated_path);
    const dhead = try zipEntryPrefix(t.allocator, io, deflated_path, wikitext_entry, 14);
    defer t.allocator.free(dhead);
    try t.expectEqualStrings("alpha beta\ngam", dhead);
    try t.expectError(error.WikitextEntryMissing, zipEntryPrefix(t.allocator, io, deflated_path, "other.raw", 4));
}

fn failingFetch(_: ?*anyopaque, _: std.Io, _: []const u8, _: []const u8) anyerror!void {
    return error.WikitextFetchFailed;
}

test "calib: the default 64k mix tokenizes a real source into 2048-token windows, twice the same" {
    const t = std.testing;
    const io = t.io;
    const root = std.c.getenv("SUSHI_MODELS_DIR") orelse return error.SkipZigTest;
    const cache = std.c.getenv("SUSHI_TEST_WIKITEXT_DIR") orelse return error.SkipZigTest;
    var model_buf: [std.fs.max_path_bytes]u8 = undefined;
    const model = std.fmt.bufPrint(&model_buf, "{s}/Qwen/Qwen3.8-Flash-Next", .{std.mem.span(root)}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, model, .{}) catch return error.SkipZigTest;
    var tokenizer = try tok.loadTokenizerAny(io, t.allocator, model);
    defer tokenizer.deinit();
    const fetcher = Fetcher{ .fetch = failingFetch };
    var a = try defaultWindows(t.allocator, io, &tokenizer, std.mem.span(cache), 65536, fetcher);
    defer a.deinit();
    var b = try defaultWindows(t.allocator, io, &tokenizer, std.mem.span(cache), 65536, fetcher);
    defer b.deinit();
    try t.expectEqual(@as(usize, 32), a.count);
    try t.expectEqualSlices(u8, &a.sha256, &b.sha256);
}
