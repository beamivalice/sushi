//! The one identity of a model folder. Every comparison of two folders (the registry, the download queue, discovery,
//! `--model`) compares `FolderId`s, never path bytes: a trailing slash, `//`, `.`, a symlink and a case spelling
//! (APFS folds case) are all one folder, and only the filesystem knows that.
const std = @import("std");

pub const FolderId = struct {
    buf: [std.fs.max_path_bytes]u8 = undefined,
    len: usize = 0,

    /// Matches no folder, itself included: an entry built without registering has no identity.
    pub const unset: FolderId = .{};

    pub const Error = error{InvalidFolderPath};

    /// The canonical path the filesystem gives `path` (absolute, or relative to the working directory). A folder that
    /// does not exist yet (a download target) is its deepest existing ancestor's canonical path plus the missing
    /// names, spelled as given: nothing exists under another spelling, and it gets its identity once made.
    pub fn of(io: std.Io, path: []const u8) Error!FolderId {
        var id: FolderId = .{};
        if (std.Io.Dir.cwd().realPathFile(io, path, &id.buf)) |n| {
            id.len = n;
            return id;
        } else |_| {}

        // Lexical names: `.` and `//` vanish, `..` drops the name before it. Safe only because none of these exist.
        var names: [64][]const u8 = undefined;
        var n: usize = 0;
        var it = std.mem.tokenizeScalar(u8, path, '/');
        while (it.next()) |name| {
            if (std.mem.eql(u8, name, ".")) continue;
            if (std.mem.eql(u8, name, "..")) {
                if (n == 0) {
                    if (std.fs.path.isAbsolute(path)) continue;
                    return error.InvalidFolderPath;
                }
                n -= 1;
                continue;
            }
            if (n == names.len) return error.InvalidFolderPath;
            names[n] = name;
            n += 1;
        }
        const absolute = std.fs.path.isAbsolute(path);
        var prefix: [std.fs.max_path_bytes]u8 = undefined;
        var keep = n;
        while (true) : (keep -= 1) {
            var w = std.Io.Writer.fixed(&prefix);
            if (absolute) w.writeByte('/') catch return error.InvalidFolderPath;
            for (names[0..keep], 0..) |name, i| {
                if (i > 0) w.writeByte('/') catch return error.InvalidFolderPath;
                w.writeAll(name) catch return error.InvalidFolderPath;
            }
            const head = if (w.buffered().len == 0) "." else w.buffered();
            if (std.Io.Dir.cwd().realPathFile(io, head, &id.buf)) |len| {
                id.len = len;
                break;
            } else |_| {}
            if (keep == 0) return error.InvalidFolderPath;
        }
        id.len = std.mem.trimEnd(u8, id.buf[0..id.len], "/").len;
        for (names[keep..n]) |name| {
            if (id.len + 1 + name.len > id.buf.len) return error.InvalidFolderPath;
            id.buf[id.len] = '/';
            @memcpy(id.buf[id.len + 1 ..][0..name.len], name);
            id.len += 1 + name.len;
        }
        return id;
    }

    pub fn bytes(self: *const FolderId) []const u8 {
        return self.buf[0..self.len];
    }

    pub fn eql(a: *const FolderId, b: *const FolderId) bool {
        return a.len != 0 and std.mem.eql(u8, a.bytes(), b.bytes());
    }
};

const testing = std.testing;

fn idAt(root: []const u8, comptime fmt: []const u8) !FolderId {
    var pb: [std.fs.max_path_bytes]u8 = undefined;
    return FolderId.of(testing.io, try std.fmt.bufPrint(&pb, fmt, .{root}));
}

test "FolderId: every spelling of one folder is one identity, and a missing folder keeps its tail" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "org/M");
    var rb: [std.fs.max_path_bytes]u8 = undefined;
    const root = rb[0..try tmp.dir.realPath(io, &rb)];
    const exact = try idAt(root, "{s}/org/M");
    inline for (.{ "{s}/org/M/", "{s}//org/M", "{s}/org/./M", "{s}/org/x/../M" }) |spelling| {
        const other = try idAt(root, spelling);
        try testing.expect(exact.eql(&other));
    }
    // A download target that is not there yet: the same names under the same canonical ancestor.
    const made = try idAt(root, "{s}//org/new/");
    var eb: [std.fs.max_path_bytes]u8 = undefined;
    try testing.expectEqualStrings(try std.fmt.bufPrint(&eb, "{s}/org/new", .{root}), made.bytes());
    try testing.expect(!exact.eql(&made));
    try testing.expect(!FolderId.unset.eql(&FolderId.unset));
}
