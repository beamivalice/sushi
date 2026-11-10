//! XXH3-64 with seed 0 and the default secret, the values of `std.hash.XxHash3.hash(0, bytes)`. The standard
//! library's long-input loop multiplies 64-bit vector lanes, which NEON has no instruction for; this one takes the
//! 32x32 products as widening multiplies, the form the reference NEON path uses.
const std = @import("std");

const V2 = @Vector(2, u64);
const H2 = @Vector(2, u32);

const prime32_1: u64 = 0x9E3779B1;
const prime32_2: u64 = 0x85EBCA77;
const prime32_3: u64 = 0xC2B2AE3D;
const prime64_1: u64 = 0x9E3779B185EBCA87;
const prime64_2: u64 = 0xC2B2AE3D27D4EB4F;
const prime64_3: u64 = 0x165667B19E3779F9;
const prime64_4: u64 = 0x85EBCA77C2B2AE63;
const prime64_5: u64 = 0x27D4EB2F165667C5;

const secret = [192]u8{
    0xb8, 0xfe, 0x6c, 0x39, 0x23, 0xa4, 0x4b, 0xbe, 0x7c, 0x01, 0x81, 0x2c, 0xf7, 0x21, 0xad, 0x1c, 0xde, 0xd4, 0x6d, 0xe9, 0x83, 0x90, 0x97, 0xdb, 0x72, 0x40, 0xa4, 0xa4, 0xb7, 0xb3, 0x67, 0x1f,
    0xcb, 0x79, 0xe6, 0x4e, 0xcc, 0xc0, 0xe5, 0x78, 0x82, 0x5a, 0xd0, 0x7d, 0xcc, 0xff, 0x72, 0x21, 0xb8, 0x08, 0x46, 0x74, 0xf7, 0x43, 0x24, 0x8e, 0xe0, 0x35, 0x90, 0xe6, 0x81, 0x3a, 0x26, 0x4c,
    0x3c, 0x28, 0x52, 0xbb, 0x91, 0xc3, 0x00, 0xcb, 0x88, 0xd0, 0x65, 0x8b, 0x1b, 0x53, 0x2e, 0xa3, 0x71, 0x64, 0x48, 0x97, 0xa2, 0x0d, 0xf9, 0x4e, 0x38, 0x19, 0xef, 0x46, 0xa9, 0xde, 0xac, 0xd8,
    0xa8, 0xfa, 0x76, 0x3f, 0xe3, 0x9c, 0x34, 0x3f, 0xf9, 0xdc, 0xbb, 0xc7, 0xc7, 0x0b, 0x4f, 0x1d, 0x8a, 0x51, 0xe0, 0x4b, 0xcd, 0xb4, 0x59, 0x31, 0xc8, 0x9f, 0x7e, 0xc9, 0xd9, 0x78, 0x73, 0x64,
    0xea, 0xc5, 0xac, 0x83, 0x34, 0xd3, 0xeb, 0xc3, 0xc5, 0x81, 0xa0, 0xff, 0xfa, 0x13, 0x63, 0xeb, 0x17, 0x0d, 0xdd, 0x51, 0xb7, 0xf0, 0xda, 0x49, 0xd3, 0x16, 0x55, 0x26, 0x29, 0xd4, 0x68, 0x9e,
    0x2b, 0x16, 0xbe, 0x58, 0x7d, 0x47, 0xa1, 0xfc, 0x8f, 0xf8, 0xb8, 0xd1, 0x7a, 0xd0, 0x31, 0xce, 0x45, 0xcb, 0x3a, 0x8f, 0x95, 0x16, 0x04, 0x28, 0xaf, 0xd7, 0xfb, 0xca, 0xbb, 0x4b, 0x40, 0x7e,
};

const stripe = 64;
const stripes_per_block = (secret.len - stripe) / 8;
const block = stripe * stripes_per_block;
/// Inputs up to here take the short paths, which the standard library runs well.
const short_max = 240;

const Acc = [4]V2;

/// The four lane pairs of the secret at byte `at`.
fn keyAt(comptime at: usize) Acc {
    var k: Acc = undefined;
    for (0..4) |i| k[i] = .{
        std.mem.readInt(u64, secret[at + 16 * i ..][0..8], .little),
        std.mem.readInt(u64, secret[at + 16 * i + 8 ..][0..8], .little),
    };
    return k;
}

/// One key per stripe of a block. Stripe keys overlap in the secret by 56 bytes; a table keeps each a plain load.
const stripe_keys: [stripes_per_block]Acc = blk: {
    var t: [stripes_per_block]Acc = undefined;
    for (&t, 0..) |*k, s| k.* = keyAt(s * 8);
    break :blk t;
};
const scramble_key = keyAt(secret.len - stripe);
const last_key = keyAt(secret.len - stripe - 7);
const merge_key = keyAt(11);

inline fn load(p: [*]const u8) V2 {
    return @as(*align(1) const V2, @ptrCast(p)).*;
}

inline fn accumulate(acc: *Acc, in: [*]const u8, key: *const Acc) void {
    inline for (0..4) |i| {
        const data = load(in + 16 * i);
        const mixed = data ^ key[i];
        const lo: H2 = @truncate(mixed);
        const hi: H2 = @truncate(mixed >> @splat(32));
        acc[i] +%= @as(V2, lo) * @as(V2, hi);
        acc[i] +%= @shuffle(u64, data, undefined, [2]i32{ 1, 0 });
    }
}

inline fn scramble(acc: *Acc) void {
    inline for (0..4) |i| {
        var v = acc[i];
        v ^= v >> @splat(47);
        v ^= scramble_key[i];
        const lo: H2 = @truncate(v);
        const hi: H2 = @truncate(v >> @splat(32));
        const p: H2 = @splat(@intCast(prime32_1));
        acc[i] = @as(V2, lo) * @as(V2, p) +% ((@as(V2, hi) * @as(V2, p)) << @splat(32));
    }
}

inline fn fold(a: u64, b: u64) u64 {
    const m = @as(u128, a) * b;
    return @as(u64, @truncate(m)) ^ @as(u64, @truncate(m >> 64));
}

const acc_init: Acc = .{
    .{ prime32_3, prime64_1 },
    .{ prime64_2, prime64_3 },
    .{ prime64_4, prime32_2 },
    .{ prime64_5, prime32_1 },
};

inline fn wholeBlock(acc: *Acc, base: [*]const u8) void {
    for (0..stripes_per_block) |s| accumulate(acc, base + s * stripe, &stripe_keys[s]);
    scramble(acc);
}

inline fn merge(acc: *const Acc, len: u64) u64 {
    var h: u64 = len *% prime64_1;
    inline for (0..4) |i| {
        const v = acc[i] ^ merge_key[i];
        h +%= fold(v[0], v[1]);
    }
    h ^= h >> 37;
    h *%= 0x165667919E3779F9;
    h ^= h >> 32;
    return h;
}

pub fn hash(bytes: []const u8) u64 {
    if (bytes.len <= short_max) return std.hash.XxHash3.hash(0, bytes);
    var acc = acc_init;
    const in = bytes.ptr;
    const blocks = (bytes.len - 1) / block;
    for (0..blocks) |n| wholeBlock(&acc, in + n * block);
    const tail = in + blocks * block;
    const stripes = ((bytes.len - 1) - blocks * block) / stripe;
    for (0..stripes) |s| accumulate(&acc, tail + s * stripe, &stripe_keys[s]);
    accumulate(&acc, in + bytes.len - stripe, &last_key);
    return merge(&acc, bytes.len);
}

/// The hash of bytes handed over in pieces: `final` is `hash` of their concatenation.
pub const Stream = struct {
    acc: Acc = acc_init,
    total: u64 = 0,
    /// The bytes not yet consumed: a block is consumed only once a byte follows it, as `hash` does.
    held: [block]u8 = undefined,
    n: usize = 0,
    /// The last stripe of the block consumed last, for a final stripe that reaches back into it.
    before: [stripe]u8 = undefined,

    pub fn update(self: *Stream, bytes: []const u8) void {
        var rest = bytes;
        self.total += rest.len;
        if (self.n > 0) {
            const take = @min(block - self.n, rest.len);
            @memcpy(self.held[self.n..][0..take], rest[0..take]);
            self.n += take;
            rest = rest[take..];
            if (rest.len == 0) return;
            wholeBlock(&self.acc, &self.held);
            self.before = self.held[block - stripe ..].*;
            self.n = 0;
        }
        while (rest.len > block) {
            wholeBlock(&self.acc, rest.ptr);
            self.before = rest[block - stripe ..][0..stripe].*;
            rest = rest[block..];
        }
        @memcpy(self.held[0..rest.len], rest);
        self.n = rest.len;
    }

    pub fn final(self: *const Stream) u64 {
        if (self.total <= short_max) return std.hash.XxHash3.hash(0, self.held[0..self.n]);
        var acc = self.acc;
        const stripes = (self.n - 1) / stripe;
        for (0..stripes) |s| accumulate(&acc, self.held[s * stripe ..].ptr, &stripe_keys[s]);
        var last: [stripe]u8 = undefined;
        if (self.n >= stripe) {
            last = self.held[self.n - stripe ..][0..stripe].*;
        } else {
            @memcpy(last[0 .. stripe - self.n], self.before[self.n..]);
            @memcpy(last[stripe - self.n ..], self.held[0..self.n]);
        }
        accumulate(&acc, &last, &last_key);
        return merge(&acc, self.total);
    }
};

test "the same value as the standard library at every length class" {
    var prng = std.Random.DefaultPrng.init(7);
    const buf = try std.testing.allocator.alloc(u8, 70_000);
    defer std.testing.allocator.free(buf);
    prng.random().bytes(buf);
    for (0..2200) |n| try std.testing.expectEqual(std.hash.XxHash3.hash(0, buf[0..n]), hash(buf[0..n]));
    for ([_]usize{ 4095, 4096, 4097, 16 << 10, (16 << 10) + 1, 65_535, 65_536, 65_537, 70_000 }) |n| {
        try std.testing.expectEqual(std.hash.XxHash3.hash(0, buf[0..n]), hash(buf[0..n]));
        try std.testing.expectEqual(std.hash.XxHash3.hash(0, buf[3..n]), hash(buf[3..n]));
    }
}

test "pieces hash as their concatenation, wherever they are cut" {
    var prng = std.Random.DefaultPrng.init(11);
    const r = prng.random();
    const buf = try std.testing.allocator.alloc(u8, 9000);
    defer std.testing.allocator.free(buf);
    r.bytes(buf);
    for ([_]usize{ 0, 1, 63, 64, 240, 241, 1023, 1024, 1025, 1088, 2048, 2049, 4096, 5000, 9000 }) |n| {
        const want = hash(buf[0..n]);
        for (0..40) |_| {
            var st: Stream = .{};
            var at: usize = 0;
            while (at < n) {
                const take = @min(n - at, r.uintLessThan(usize, 1500));
                st.update(buf[at..][0..take]);
                at += take;
            }
            try std.testing.expectEqual(want, st.final());
        }
        var whole: Stream = .{};
        whole.update(buf[0..n]);
        try std.testing.expectEqual(want, whole.final());
    }
}
