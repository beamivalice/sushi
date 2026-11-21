//! The environment a test process runs with, read back from outside: tests/test_build_cache_env.sh runs `zig build` twice
//! with one `SUSHI_*` variable changed and reads what each process saw.
const std = @import("std");
const testing = std.testing;

test "test env: SUSHI_TEST_ENV_DUMP names a file the run writes its SUSHI_ variables to, as NAME=VALUE lines" {
    const path = std.c.getenv("SUSHI_TEST_ENV_DUMP") orelse return error.SkipZigTest;
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(testing.allocator);
    var i: usize = 0;
    while (std.c.environ[i]) |entry| : (i += 1) {
        const line = std.mem.span(entry);
        if (std.mem.startsWith(u8, line, "SUSHI_")) try text.print(testing.allocator, "{s}\n", .{line});
    }
    const file = try std.Io.Dir.createFileAbsolute(testing.io, std.mem.span(path), .{});
    defer file.close(testing.io);
    try file.writeStreamingAll(testing.io, text.items);
}
