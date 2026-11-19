//! A forward an arch refused for memory partway through a request: the text the client's answer carries. The arch
//! records it on the inference thread before it returns `error.PrefillChunkDoesNotFit`; the slot takes it as it fails.

pub const Text = struct {
    buf: [1024]u8 = undefined,
    len: usize = 0,

    pub fn text(self: *const Text) []const u8 {
        return self.buf[0..self.len];
    }
};

threadlocal var last: ?Text = null;

pub fn record(text: []const u8) void {
    var t: Text = .{};
    t.len = @min(text.len, t.buf.len);
    @memcpy(t.buf[0..t.len], text[0..t.len]);
    last = t;
}

pub fn take() ?Text {
    defer last = null;
    return last;
}
