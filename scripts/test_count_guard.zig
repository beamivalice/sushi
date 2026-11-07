//! `test_count_guard <bucket> <test binary>`: exit 1, naming the bucket, when the binary holds no named test. It asks
//! the test runner for its test list over the build protocol and runs nothing, so `zig build test` can follow each
//! scanned binary with it. A filter set that selects nothing otherwise passes: Zig reports "0 tests passed" as success.
const std = @import("std");

/// The root file's anonymous `test { ... }` block survives every filter and is named `<module>.test_<n>`.
fn isAnonymous(name: []const u8) bool {
    const dot = std.mem.lastIndexOfScalar(u8, name, '.') orelse return false;
    const tail = name[dot + 1 ..];
    if (!std.mem.startsWith(u8, tail, "test_")) return false;
    _ = std.fmt.parseInt(u32, tail["test_".len..], 10) catch return false;
    return true;
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa);
    _ = args.next();
    const bucket = args.next() orelse return error.MissingBucketName;
    const binary = args.next() orelse return error.MissingTestBinary;

    var child = try std.process.spawn(io, .{
        .argv = &.{ binary, "--listen=-" },
        .stdin = .pipe,
        .stdout = .pipe,
        .stderr = .ignore,
    });
    defer child.kill(io);
    var read_buf: [4096]u8 = undefined;
    var write_buf: [64]u8 = undefined;
    var reader = child.stdout.?.readerStreaming(io, &read_buf);
    var writer = child.stdin.?.writerStreaming(io, &write_buf);
    const client: std.zig.Client = .{ .in = &reader.interface, .out = &writer.interface };
    try client.serveBodylessMessage(.query_test_metadata);

    var named: usize = 0;
    while (true) {
        const header = try client.receiveMessage();
        const body = try client.in.readAlloc(init.gpa, header.bytes_len);
        defer init.gpa.free(body);
        if (header.tag != .test_metadata) continue;
        var fixed: std.Io.Reader = .fixed(body);
        const meta = try fixed.takeStruct(std.zig.Server.Message.TestMetadata, .little);
        const strings = body[body.len - meta.string_bytes_len ..];
        for (0..meta.tests_len) |_| {
            const at = try fixed.takeInt(u32, .little);
            if (!isAnonymous(std.mem.sliceTo(strings[at..], 0))) named += 1;
        }
        break;
    }
    try client.serveBodylessMessage(.exit);
    if (named == 0) {
        std.debug.print("test bucket {s} ran 0 tests: its filters select nothing\n", .{bucket});
        std.process.exit(1);
    }
}
