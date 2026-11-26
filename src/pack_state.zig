//! What a pack folder is: the one reader of the pull's marker and record, and the one place the content rule lives.
//! Every decision (load gate, boot, discovery, `sushi list`, `sushi run`) asks `of` and nothing else. A pull verifies
//! every file's content before it renames it into place and writes its record last, so a record whose weights still
//! stat as they did then (`hf_pull.Seal`) seals the folder: no content is read to load it. Links no MLX.
const std = @import("std");
const hf_pull = @import("hf_pull.zig");

/// Present for the whole of a pull and removed only after the record is written: a refused request leaves no
/// `.partial`, so the files alone cannot tell an interrupted pull from a finished one.
pub const pull_marker = ".pull-incomplete";

pub const Kind = enum {
    /// The marker is there: a pull is running, stopped or failed, and the files are a mixture of revisions.
    pulling,
    /// A record seals the folder and every sealed weights file still stats as it did when the pull verified it.
    sealed,
    /// A record seals the folder, but a sealed file is missing or differs.
    changed,
    /// No marker and nothing sealing it (a hand copy, `hf download`, a converter's output, an older pull): the content
    /// rule decides.
    unsealed,
    /// No config.json, or no readable folder.
    not_a_pack,
};

pub const State = struct {
    kind: Kind,
    /// `unsealed` only: the content rule found the folder whole.
    whole: bool = false,
    reason_buf: [1024]u8 = undefined,
    reason_len: usize = 0,

    pub fn loadable(self: *const State) bool {
        return switch (self.kind) {
            .sealed => true,
            .unsealed => self.whole,
            .pulling, .changed, .not_a_pack => false,
        };
    }

    /// Why the folder cannot load, ending in what to do; empty when it can.
    pub fn reason(self: *const State) []const u8 {
        return self.reason_buf[0..self.reason_len];
    }

    /// The word `sushi list` prints for a pack that cannot load.
    pub fn label(self: *const State) []const u8 {
        return switch (self.kind) {
            .sealed => "sealed",
            .changed => "changed",
            .pulling => "partial",
            .unsealed => if (self.whole) "unsealed" else "partial",
            .not_a_pack => "not a pack",
        };
    }

    fn refuse(kind: Kind, comptime fmt: []const u8, args: anytype) State {
        var s: State = .{ .kind = kind };
        const text = std.fmt.bufPrint(&s.reason_buf, fmt, args) catch blk: {
            @memcpy(s.reason_buf[0..generic_reason.len], generic_reason);
            break :blk s.reason_buf[0..generic_reason.len];
        };
        s.reason_len = text.len;
        return s;
    }
};

const generic_reason = "this pack's files are not all there: resume it on Downloads, or run `sushi pull` for it again, then load it";
const redo = "resume it on Downloads, or run `sushi pull` for it again, then load it";

/// The state of the folder at `dir_path` (absolute, or relative to the working directory).
pub fn of(io: std.Io, dir_path: []const u8) State {
    var dir = std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true }) catch
        return State.refuse(.not_a_pack, "{s} is not a readable folder", .{dir_path});
    defer dir.close(io);
    return ofDir(io, dir, std.fs.path.basename(std.mem.trimEnd(u8, dir_path, "/")));
}

/// The state of the open folder `dir` (opened for iteration), which `name` calls in its reasons.
pub fn ofDir(io: std.Io, dir: std.Io.Dir, name: []const u8) State {

    if (dir.statFile(io, pull_marker, .{})) |_| {
        return State.refuse(.pulling, "The last download or update of {s} did not finish, so its files are mixed: " ++ redo, .{name});
    } else |_| {}
    const config = dir.statFile(io, "config.json", .{}) catch return State.refuse(.not_a_pack, "{s} has no config.json", .{name});
    if (config.kind != .file) return State.refuse(.not_a_pack, "{s} has no config.json", .{name});

    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const record: ?hf_pull.Manifest = switch (hf_pull.readRecord(arena, io, dir)) {
        .absent => null,
        .unreadable => return State.refuse(.changed, "{s}: the pull's record (" ++ hf_pull.manifest_name ++ ") is unreadable or from a newer sushi: run `sushi pull <repo>` to re-verify", .{name}),
        .present => |m| m,
    };
    if (record) |m| {
        var sealed_rows: usize = 0;
        for (m.files) |f| {
            if (f.seal == null) continue;
            sealed_rows += 1;
            switch (hf_pull.sealCheck(io, dir, f)) {
                .holds => {},
                .missing => return State.refuse(.changed, "{s} is missing {s}, which its last pull verified: " ++ redo, .{ name, f.path }),
                .differs => return State.refuse(.changed, "{s}/{s} was replaced or edited since its last pull verified it: " ++ redo, .{ name, f.path }),
            }
        }
        if (sealed_rows > 0) return .{ .kind = .sealed };
    }
    return unsealed(io, arena, dir, name, record);
}

/// The content rule: a config.json, every weights file the index names (else at least one top-level `.safetensors`),
/// and no `.partial` of a file the folder needs. `record`, from a pull before seals, or the index tells a `.partial`
/// of a file the repo dropped, which a finished pull leaves behind, from the one a pull is still filling.
fn unsealed(io: std.Io, arena: std.mem.Allocator, dir: std.Io.Dir, name: []const u8, record: ?hf_pull.Manifest) State {
    const indexed: ?[]const []const u8 = hf_pull.indexedFiles(arena, io, dir) catch |err| switch (err) {
        error.NoIndex => null,
        error.InvalidIndex => return State.refuse(.unsealed, "{s}'s " ++ hf_pull.weights_index_name ++ " is unreadable: " ++ redo, .{name}),
        error.OutOfMemory => return State.refuse(.unsealed, "{s} could not be read: out of memory", .{name}),
    };
    if (indexed) |files| {
        for (files) |f| {
            const st = dir.statFile(io, f, .{}) catch return State.refuse(.unsealed, "{s} is missing {s}, which its weights index names: " ++ redo, .{ name, f });
            if (st.kind != .file) return State.refuse(.unsealed, "{s} is missing {s}, which its weights index names: " ++ redo, .{ name, f });
        }
    } else if (!hasTopLevelWeights(io, dir)) {
        return State.refuse(.unsealed, "{s} has no weights (.safetensors file): " ++ redo, .{name});
    }
    if (unfinishedPartial(io, arena, dir, record, indexed)) |partial| {
        return State.refuse(.unsealed, "{s} holds {s}, an unfinished download: " ++ redo, .{ name, partial });
    }
    return .{ .kind = .unsealed, .whole = true };
}

fn hasTopLevelWeights(io: std.Io, dir: std.Io.Dir) bool {
    var it = dir.iterate();
    while (it.next(io) catch null) |e| {
        if (!std.mem.endsWith(u8, e.name, ".safetensors")) continue;
        // A shard may be a link (a Hugging Face cache snapshot): what it points at must be a file.
        const st = dir.statFile(io, e.name, .{}) catch continue;
        if (st.kind == .file) return true;
    }
    return false;
}

/// The first `.partial` (top level or one folder down, where `mtp/` pulls) that is not provably the leftover of a file
/// this revision does not have; null when there is none.
fn unfinishedPartial(io: std.Io, arena: std.mem.Allocator, dir: std.Io.Dir, record: ?hf_pull.Manifest, indexed: ?[]const []const u8) ?[]const u8 {
    var it = dir.iterate();
    while (it.next(io) catch null) |e| {
        switch (e.kind) {
            .file => if (blocks(e.name, record, indexed)) return arena.dupe(u8, e.name) catch e.name,
            .directory => {
                var sub = dir.openDir(io, e.name, .{ .iterate = true }) catch continue;
                defer sub.close(io);
                var sit = sub.iterate();
                while (sit.next(io) catch null) |se| {
                    if (se.kind != .file) continue;
                    const rel = std.fmt.allocPrint(arena, "{s}/{s}", .{ e.name, se.name }) catch continue;
                    if (blocks(rel, record, indexed)) return rel;
                }
            },
            else => {},
        }
    }
    return null;
}

fn blocks(rel: []const u8, record: ?hf_pull.Manifest, indexed: ?[]const []const u8) bool {
    if (!std.mem.endsWith(u8, rel, ".partial")) return false;
    const final = rel[0 .. rel.len - ".partial".len];
    if (record) |m| {
        for (m.files) |f| if (std.mem.eql(u8, f.path, final)) return true;
        return false;
    }
    if (indexed) |files| {
        if (!std.mem.endsWith(u8, final, ".safetensors")) return true;
        for (files) |f| if (std.mem.eql(u8, f, final)) return true;
        return false;
    }
    return true;
}

const testing = std.testing;

const Fixture = struct {
    tmp: testing.TmpDir,
    path: [:0]u8,

    fn init() !Fixture {
        var tmp = testing.tmpDir(.{ .iterate = true });
        errdefer tmp.cleanup();
        return .{ .tmp = tmp, .path = try tmp.dir.realPathFileAlloc(testing.io, ".", testing.allocator) };
    }

    fn deinit(self: *Fixture) void {
        testing.allocator.free(self.path);
        self.tmp.cleanup();
    }

    fn write(self: *Fixture, sub: []const u8, data: []const u8) !void {
        if (std.fs.path.dirname(sub)) |d| try self.tmp.dir.createDirPath(testing.io, d);
        try self.tmp.dir.writeFile(testing.io, .{ .sub_path = sub, .data = data });
    }

    fn state(self: *Fixture) State {
        return of(testing.io, self.path);
    }

    /// The record a finished pull writes for `rows` (config.json unsealed, each `.safetensors` sealed from disk).
    fn writeRecord(self: *Fixture, rows: []const []const u8, sealed: bool) !void {
        var files: [8]hf_pull.RepoFile = undefined;
        for (rows, 0..) |path, i| {
            const st = try self.tmp.dir.statFile(testing.io, path, .{});
            files[i] = .{ .path = @constCast(path), .size = st.size, .oid = @constCast("aa") };
            if (sealed and hf_pull.sealsPath(path)) files[i].seal = hf_pull.Seal.of(st);
        }
        try hf_pull.writeManifest(testing.allocator, testing.io, self.path, .{ .repo = "beamster/X", .files = files[0..rows.len] });
    }
};

test "pack state: a folder with weights and no marker or record is unsealed and loadable" {
    var fx = try Fixture.init();
    defer fx.deinit();
    try testing.expectEqual(Kind.not_a_pack, fx.state().kind);
    try fx.write("config.json", "{}");
    try testing.expect(!fx.state().loadable()); // config alone: no weights
    try fx.write("model.safetensors", "12345");
    const s = fx.state();
    try testing.expectEqual(Kind.unsealed, s.kind);
    try testing.expect(s.loadable());
    try testing.expectEqualStrings("", s.reason());
}

test "pack state: the marker makes a folder pulling whatever else it holds" {
    var fx = try Fixture.init();
    defer fx.deinit();
    try fx.write("config.json", "{}");
    try fx.write("model.safetensors", "12345");
    try fx.writeRecord(&.{ "config.json", "model.safetensors" }, true);
    try fx.write(pull_marker, "");
    const s = fx.state();
    try testing.expectEqual(Kind.pulling, s.kind);
    try testing.expect(!s.loadable());
    try testing.expect(std.mem.indexOf(u8, s.reason(), "sushi pull") != null);
}

test "pack state: a .partial weights file leaves a folder partial, and a dropped file's orphan does not" {
    var fx = try Fixture.init();
    defer fx.deinit();
    try fx.write("config.json", "{}");
    try fx.write("model.safetensors.partial", "12");
    try testing.expect(!fx.state().loadable());
    try fx.write("model.safetensors", "12345");
    try testing.expect(!fx.state().loadable()); // no record, no index: any partial may be a shard in flight
    try fx.write("mtp/w.safetensors.partial", "1");
    try fx.tmp.dir.deleteFile(testing.io, "model.safetensors.partial");
    try testing.expect(!fx.state().loadable()); // one folder down counts too
    try fx.tmp.dir.deleteFile(testing.io, "mtp/w.safetensors.partial");
    try fx.write("dropped.safetensors.partial", "12");
    try fx.write("model.safetensors.index.json", "{\"weight_map\":{\"a\":\"model.safetensors\"}}");
    try testing.expect(fx.state().loadable()); // the index does not name it
    try fx.write("model-2.safetensors.partial", "1");
    try fx.write("model.safetensors.index.json", "{\"weight_map\":{\"a\":\"model.safetensors\",\"b\":\"model-2.safetensors\"}}");
    const s = fx.state();
    try testing.expect(!s.loadable()); // named, missing, and its partial is there
    try testing.expect(std.mem.indexOf(u8, s.reason(), "model-2.safetensors") != null);
}

test "pack state: a shard the index names must be there" {
    var fx = try Fixture.init();
    defer fx.deinit();
    try fx.write("config.json", "{}");
    try fx.write("a.safetensors", "1");
    try fx.write("model.safetensors.index.json", "{\"weight_map\":{\"x\":\"a.safetensors\",\"y\":\"b.safetensors\"}}");
    try testing.expect(!fx.state().loadable());
    try fx.write("b.safetensors", "2");
    try testing.expect(fx.state().loadable());
    try fx.write("model.safetensors.index.json", "nope");
    try testing.expect(!fx.state().loadable());
}

test "pack state: a record from before seals leaves the folder unsealed, and lists what its pull needed" {
    var fx = try Fixture.init();
    defer fx.deinit();
    try fx.write("config.json", "{}");
    try fx.write("model.safetensors", "12345");
    try fx.write("dropped.safetensors.partial", "12");
    try fx.writeRecord(&.{ "config.json", "model.safetensors" }, false);
    const s = fx.state();
    try testing.expectEqual(Kind.unsealed, s.kind);
    try testing.expect(s.loadable());
    try fx.write("config.json.partial", "{");
    try testing.expect(!fx.state().loadable()); // a listed file's partial is a pull still filling it
}

test "pack state: a sealed folder loads from stats alone and a hand edit of a small file never refuses it" {
    var fx = try Fixture.init();
    defer fx.deinit();
    try fx.write("config.json", "{}");
    try fx.write("tokenizer.json", "{}");
    try fx.write("model.safetensors", "12345");
    try fx.writeRecord(&.{ "config.json", "tokenizer.json", "model.safetensors" }, true);
    try fx.write("orphan.safetensors.partial", "1"); // a file the record does not list
    try testing.expectEqual(Kind.sealed, fx.state().kind);
    try fx.write("config.json", "{\"hand\":\"edited\"}");
    try fx.write("notes.txt", "mine");
    const s = fx.state();
    try testing.expectEqual(Kind.sealed, s.kind);
    try testing.expect(s.loadable());
}

test "pack state: a replaced or removed weights file reads changed and names the file" {
    var fx = try Fixture.init();
    defer fx.deinit();
    try fx.write("config.json", "{}");
    try fx.write("model.safetensors", "12345");
    try fx.writeRecord(&.{ "config.json", "model.safetensors" }, true);
    try fx.write("model.safetensors.new", "12345"); // same size, another file
    try fx.tmp.dir.rename("model.safetensors.new", fx.tmp.dir, "model.safetensors", testing.io);
    var s = fx.state();
    try testing.expectEqual(Kind.changed, s.kind);
    try testing.expect(!s.loadable());
    try testing.expect(std.mem.indexOf(u8, s.reason(), "model.safetensors") != null);
    try testing.expect(std.mem.indexOf(u8, s.reason(), "sushi pull") != null);
    try fx.tmp.dir.deleteFile(testing.io, "model.safetensors");
    s = fx.state();
    try testing.expectEqual(Kind.changed, s.kind);
    try testing.expect(std.mem.indexOf(u8, s.reason(), "missing") != null);
}

test "pack state: a record parses with and without seals, and a half-sealed row is refused" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const v1 = try hf_pull.parseManifest(arena.allocator(), "{\"version\":1,\"repo\":\"o/r\",\"files\":[{\"path\":\"m.safetensors\",\"size\":5,\"sha256\":\"aa\"}]}");
    try testing.expect(v1.files[0].seal == null);
    const v2 = try hf_pull.parseManifest(arena.allocator(), "{\"version\":2,\"files\":[{\"path\":\"m.safetensors\",\"size\":5,\"sha256\":\"aa\",\"inode\":7,\"mtime_ns\":9}]}");
    try testing.expectEqual(@as(u64, 7), v2.files[0].seal.?.inode);
    try testing.expectError(error.InvalidManifest, hf_pull.parseManifest(arena.allocator(), "{\"version\":2,\"files\":[{\"path\":\"m\",\"inode\":7}]}"));
}

test "pack state: a record that does not parse, or is from a newer sushi, reads changed and never the content rule" {
    var fx = try Fixture.init();
    defer fx.deinit();
    try fx.write("config.json", "{}");
    try fx.write("model.safetensors", "12345");
    try testing.expect(fx.state().loadable());
    for ([_][]const u8{ "nope", "{\"version\":3,\"files\":[]}", "{\"files\":[]}" }) |bytes| {
        try fx.write(hf_pull.manifest_name, bytes);
        const s = fx.state();
        try testing.expectEqual(Kind.changed, s.kind);
        try testing.expect(!s.loadable());
        try testing.expect(std.mem.indexOf(u8, s.reason(), "unreadable or from a newer sushi") != null);
        try testing.expect(std.mem.indexOf(u8, s.reason(), "sushi pull <repo>") != null);
    }
}
