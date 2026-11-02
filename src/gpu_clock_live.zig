//! The GPU clock of the running box, read without sudo: IOReport's "GPU Performance States" residency, weighted by
//! the P-state frequencies of the `pmgr` registry entry. Fails soft: anything missing reads as 0 MHz.
//! Docs: server-http-apis.md#gpu-clock.
const std = @import("std");
const builtin = @import("builtin");
const testing = std.testing;

const is_macos = builtin.os.tag == .macos;
const max_states = 32;

pub const Table = struct {
    /// MHz of P1, P2, ...; the OFF state has no entry.
    mhz: [max_states]u32 = @splat(0),
    len: usize = 0,

    pub fn top(t: Table) u32 {
        return if (t.len == 0) 0 else t.mhz[t.len - 1];
    }
};

/// `voltage-states9`: little-endian (frequency, voltage) u32 pairs. The frequency is in Hz on some chips and kHz on
/// others; a zero frequency is the OFF row.
pub fn decodeTable(bytes: []const u8) Table {
    var t: Table = .{};
    var i: usize = 0;
    while (i + 8 <= bytes.len and t.len < max_states) : (i += 8) {
        const f = std.mem.readInt(u32, bytes[i..][0..4], .little);
        if (f == 0) continue;
        t.mhz[t.len] = if (f >= 50_000_000) f / 1_000_000 else if (f >= 50_000) f / 1000 else f;
        t.len += 1;
    }
    return t;
}

/// "P7" is the 7th table entry (1-based) and clamps to the top one when the table is shorter; OFF and anything else is null.
pub fn stateMhz(t: Table, name: []const u8) ?u32 {
    if (t.len == 0 or name.len < 2 or name[0] != 'P') return null;
    const n = std.fmt.parseInt(usize, name[1..], 10) catch return null;
    if (n == 0) return null;
    return t.mhz[@min(n, t.len) - 1];
}

/// Residency-weighted mean frequency over the powered states; OFF time does not count.
pub const Average = struct {
    weighted: f64 = 0,
    residency: f64 = 0,

    pub fn add(a: *Average, t: Table, name: []const u8, residency: i64) void {
        const f = stateMhz(t, name) orelse return;
        if (residency <= 0) return;
        const r: f64 = @floatFromInt(residency);
        a.weighted += r * @as(f64, @floatFromInt(f));
        a.residency += r;
    }

    pub fn mhz(a: Average) u32 {
        return if (a.residency <= 0) 0 else @intFromFloat(@round(a.weighted / a.residency));
    }
};

// IOReport has no header or framework to link, so its symbols are bound at runtime; CoreFoundation and IOKit are linked.
const CF = ?*anyopaque;
extern "c" fn CFStringCreateWithCString(alloc: CF, s: [*:0]const u8, enc: u32) CF;
extern "c" fn CFStringGetCString(s: CF, buf: [*]u8, len: isize, enc: u32) bool;
extern "c" fn CFDictionaryGetValue(d: CF, k: CF) CF;
extern "c" fn CFArrayGetCount(a: CF) isize;
extern "c" fn CFArrayGetValueAtIndex(a: CF, i: isize) CF;
extern "c" fn CFDataGetLength(d: CF) isize;
extern "c" fn CFDataGetBytePtr(d: CF) [*]const u8;
extern "c" fn CFRelease(o: CF) void;
extern "c" fn IOServiceNameMatching(name: [*:0]const u8) CF;
extern "c" fn IOServiceGetMatchingServices(port: u32, matching: CF, iter: *u32) i32;
extern "c" fn IOIteratorNext(iter: u32) u32;
extern "c" fn IOObjectRelease(obj: u32) i32;
extern "c" fn IORegistryEntryCreateCFProperty(entry: u32, key: CF, alloc: CF, opts: u32) CF;

const utf8: u32 = 0x08000100;

const Fns = struct {
    copyGroup: *const fn (CF, CF, u64, u64, u64) callconv(.c) CF,
    createSub: *const fn (CF, CF, *CF, u64, CF) callconv(.c) CF,
    samples: *const fn (CF, CF, CF) callconv(.c) CF,
    delta: *const fn (CF, CF, CF) callconv(.c) CF,
    chName: *const fn (CF) callconv(.c) CF,
    stCount: *const fn (CF) callconv(.c) i32,
    stName: *const fn (CF, i32) callconv(.c) CF,
    stRes: *const fn (CF, i32) callconv(.c) i64,
};

fn cfs(s: [*:0]const u8) CF {
    return CFStringCreateWithCString(null, s, utf8);
}

fn readTable() Table {
    const matching = IOServiceNameMatching("pmgr") orelse return .{};
    var iter: u32 = 0;
    if (IOServiceGetMatchingServices(0, matching, &iter) != 0) return .{};
    defer _ = IOObjectRelease(iter);
    const entry = IOIteratorNext(iter);
    if (entry == 0) return .{};
    defer _ = IOObjectRelease(entry);
    const key = cfs("voltage-states9");
    defer CFRelease(key);
    const data = IORegistryEntryCreateCFProperty(entry, key, null, 0) orelse return .{};
    defer CFRelease(data);
    const len: usize = @intCast(@max(CFDataGetLength(data), 0));
    return decodeTable(CFDataGetBytePtr(data)[0..len]);
}

/// One IOReport subscription and its previous sample. Not thread-safe: each caller owns its own.
pub const Reader = struct {
    fns: Fns,
    sub: CF,
    subbed: CF,
    prev: CF,
    table: Table,

    pub fn open() ?Reader {
        if (!is_macos) return null;
        var lib = std.DynLib.open("/usr/lib/libIOReport.dylib") catch return null;
        const f: Fns = .{
            .copyGroup = lib.lookup(@FieldType(Fns, "copyGroup"), "IOReportCopyChannelsInGroup") orelse return null,
            .createSub = lib.lookup(@FieldType(Fns, "createSub"), "IOReportCreateSubscription") orelse return null,
            .samples = lib.lookup(@FieldType(Fns, "samples"), "IOReportCreateSamples") orelse return null,
            .delta = lib.lookup(@FieldType(Fns, "delta"), "IOReportCreateSamplesDelta") orelse return null,
            .chName = lib.lookup(@FieldType(Fns, "chName"), "IOReportChannelGetChannelName") orelse return null,
            .stCount = lib.lookup(@FieldType(Fns, "stCount"), "IOReportStateGetCount") orelse return null,
            .stName = lib.lookup(@FieldType(Fns, "stName"), "IOReportStateGetNameForIndex") orelse return null,
            .stRes = lib.lookup(@FieldType(Fns, "stRes"), "IOReportStateGetResidency") orelse return null,
        };
        const table = readTable();
        if (table.len == 0) return null;
        const group = cfs("GPU Stats");
        defer CFRelease(group);
        const subgroup = cfs("GPU Performance States");
        defer CFRelease(subgroup);
        const chans = f.copyGroup(group, subgroup, 0, 0, 0) orelse return null;
        defer CFRelease(chans);
        var subbed: CF = null;
        const sub = f.createSub(null, chans, &subbed, 0, null) orelse return null;
        const prev = f.samples(sub, subbed, null) orelse return null;
        return .{ .fns = f, .sub = sub, .subbed = subbed, .prev = prev, .table = table };
    }

    /// Active-weighted MHz over the window since the previous call (or since `open`); 0 when the GPU was off throughout.
    pub fn sample(l: *Reader) ?u32 {
        const now = l.fns.samples(l.sub, l.subbed, null) orelse return null;
        defer {
            CFRelease(l.prev);
            l.prev = now;
        }
        const delta = l.fns.delta(l.prev, now, null) orelse return null;
        defer CFRelease(delta);
        const key = cfs("IOReportChannels");
        defer CFRelease(key);
        const arr = CFDictionaryGetValue(delta, key) orelse return null;
        var avg: Average = .{};
        var i: isize = 0;
        while (i < CFArrayGetCount(arr)) : (i += 1) {
            const ch = CFArrayGetValueAtIndex(arr, i);
            var cbuf: [32]u8 = undefined;
            if (!CFStringGetCString(l.fns.chName(ch), &cbuf, cbuf.len, utf8)) continue;
            if (!std.mem.eql(u8, std.mem.sliceTo(&cbuf, 0), "GPUPH")) continue;
            var k: i32 = 0;
            while (k < l.fns.stCount(ch)) : (k += 1) {
                var nbuf: [16]u8 = undefined;
                if (!CFStringGetCString(l.fns.stName(ch, k), &nbuf, nbuf.len, utf8)) continue;
                avg.add(l.table, std.mem.sliceTo(&nbuf, 0), l.fns.stRes(ch, k));
            }
        }
        return avg.mhz();
    }
};

// The server's gauge sampler is the only caller of the process-wide reader.
var shared: ?Reader = null;
var tried = false;

fn sharedReader() ?*Reader {
    if (!tried) {
        tried = true;
        shared = Reader.open();
    }
    return if (shared) |*r| r else null;
}

/// The highest P-state frequency, the chart's fixed axis; 0 when the clock cannot be read.
pub fn maxMhz() u32 {
    const r = sharedReader() orelse return 0;
    return r.table.top();
}

/// Active-weighted MHz since the previous call; 0 when the GPU was off in that window or the clock is unavailable.
pub fn sampleMhz() u32 {
    const r = sharedReader() orelse return 0;
    return r.sample() orelse 0;
}

fn pair(buf: []u8, f: u32) void {
    std.mem.writeInt(u32, buf[0..4], f, .little);
    std.mem.writeInt(u32, buf[4..8], 700, .little);
}

test "decodeTable: Hz and kHz tables both read as MHz and the zero row is dropped" {
    var hz: [24]u8 = undefined;
    pair(hz[0..8], 0);
    pair(hz[8..16], 338_000_000);
    pair(hz[16..24], 1_620_000_000);
    const a = decodeTable(&hz);
    try testing.expectEqual(@as(usize, 2), a.len);
    try testing.expectEqual(@as(u32, 338), a.mhz[0]);
    try testing.expectEqual(@as(u32, 1620), a.top());

    var khz: [16]u8 = undefined;
    pair(khz[0..8], 444_000);
    pair(khz[8..16], 1_296_000);
    const b = decodeTable(&khz);
    try testing.expectEqual(@as(u32, 444), b.mhz[0]);
    try testing.expectEqual(@as(u32, 1296), b.top());
}

test "decodeTable: a short or empty property is an empty table" {
    try testing.expectEqual(@as(usize, 0), decodeTable(&.{}).len);
    try testing.expectEqual(@as(usize, 0), decodeTable(&.{ 1, 2, 3 }).len);
    try testing.expectEqual(@as(u32, 0), (Table{}).top());
}

test "stateMhz: P-states index the table from 1, clamp past its end, and OFF has none" {
    var t: Table = .{};
    t.mhz[0] = 338;
    t.mhz[1] = 486;
    t.len = 2;
    try testing.expectEqual(@as(?u32, 338), stateMhz(t, "P1"));
    try testing.expectEqual(@as(?u32, 486), stateMhz(t, "P2"));
    try testing.expectEqual(@as(?u32, 486), stateMhz(t, "P15"));
    try testing.expectEqual(@as(?u32, null), stateMhz(t, "OFF"));
    try testing.expectEqual(@as(?u32, null), stateMhz(t, "P0"));
    try testing.expectEqual(@as(?u32, null), stateMhz(t, "Px"));
    try testing.expectEqual(@as(?u32, null), stateMhz(.{}, "P1"));
}

test "Average: weighted by residency, OFF excluded, empty and all-OFF read 0" {
    var t: Table = .{};
    t.mhz[0] = 1000;
    t.mhz[1] = 2000;
    t.len = 2;
    var a: Average = .{};
    try testing.expectEqual(@as(u32, 0), a.mhz());
    a.add(t, "OFF", 900);
    try testing.expectEqual(@as(u32, 0), a.mhz());
    a.add(t, "P1", 3);
    a.add(t, "P2", 1);
    try testing.expectEqual(@as(u32, 1250), a.mhz());
}

test "live clock: on macOS the table has a top frequency and a sample is 0 or within it" {
    if (!is_macos) return error.SkipZigTest;
    const top = maxMhz();
    if (top == 0) return;
    try testing.expect(top >= 100 and top < 10_000);
    var r = Reader.open() orelse return error.TestUnexpectedResult;
    try testing.expect((r.sample() orelse return error.TestUnexpectedResult) <= top);
}
