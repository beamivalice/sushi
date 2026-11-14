//! The GPU this process runs on, read once: its generation and class letter, the compute units its kernels may use, and
//! the OS it runs under. Every kernel arm is chosen from this profile; nothing else reads the GPU's name, the OS version
//! or the QA switch (`zig build lint`, rule `device-probe`).
const std = @import("std");

/// A compute unit of a GPU that a kernel arm is written for. A later chip's own unit is a new tag here and a row in
/// `unit_rows`: no generation number implies a unit anywhere else.
pub const Unit = enum {
    /// The shader cores' SIMD groups: every Apple GPU has them, and every kernel has this arm.
    simd,
    /// The matrix units that arrived with generation 17 (M5), reached through MetalPerformancePrimitives' `matmul2d`.
    matrix_g17,
};

pub const Units = std.EnumSet(Unit);

/// The letter that ends an architecture name (`applegpu_g17s`). MLX's tile choices switch on it, so the stock ops held
/// to MLX's bytes do too: `p` is a phone, `d` the widest desktop class, and an M5 Max reports `s`.
pub const Class = enum {
    g,
    s,
    c,
    d,
    p,
    other,

    fn of(letter: u8) Class {
        return switch (letter) {
            'g' => .g,
            's' => .s,
            'c' => .c,
            'd' => .d,
            'p' => .p,
            else => .other,
        };
    }
};

pub const Os = struct {
    major: u32 = 0,
    minor: u32 = 0,

    /// "26.4" or "26.4.1". A part that is not a number reads as 0, so garbage never meets a floor.
    pub fn parse(text: []const u8) Os {
        var it = std.mem.splitScalar(u8, text, '.');
        const major = std.fmt.parseInt(u32, it.first(), 10) catch 0;
        const minor: u32 = if (it.next()) |part| (std.fmt.parseInt(u32, part, 10) catch 0) else 0;
        return .{ .major = major, .minor = minor };
    }

    pub fn atLeast(self: Os, major: u32, minor: u32) bool {
        return self.major > major or (self.major == major and self.minor >= minor);
    }
};

const UnitRow = struct {
    unit: Unit,
    /// The first generation that has the unit, on a desktop class and on a phone.
    from: u32,
    from_phone: u32,
    /// The last generation whose unit this arm's kernels run on; null while every later one still does.
    until: ?u32 = null,
    /// The first OS whose Metal reaches the unit.
    os: Os,
};

/// The only place a chip generation is named. A generation past this table keeps the newest row's units, which is what
/// MLX's own dispatch assumes; a chip whose unit the `matrix_g17` kernels cannot use closes that row with `until` and
/// opens its own.
const unit_rows = [_]UnitRow{
    .{ .unit = .matrix_g17, .from = 17, .from_phone = 18, .os = .{ .major = 26, .minor = 2 } },
};

/// The newest generation a kernel here was run on. A load on a later one says so.
pub const newest_known: u32 = 17;

fn row(unit: Unit) ?UnitRow {
    for (unit_rows) |r| if (r.unit == unit) return r;
    return null;
}

pub const Profile = struct {
    /// 0 when the name names no Apple GPU generation.
    generation: u32,
    class: Class,
    /// What the chip has, under any OS and whatever the QA switch says.
    chip: Units,
    /// What a kernel may use: the chip's units that this OS reaches, down to `simd` alone when they are withheld.
    units: Units,
    os: Os,
    /// The QA switch: every unit past `simd` is withheld, so an M5 runs the arms an M1 to M4 runs.
    withheld: bool,

    /// `arch` is Metal's architecture name, any case: `applegpu_g` or `g`, the generation's digits, at most one class
    /// letter. Anything else (a compiler target such as `air64_v27`) names no generation.
    pub fn from(arch: []const u8, os: Os, withheld: bool) Profile {
        const parsed = parseArch(arch);
        var chip = Units.initOne(.simd);
        var units = Units.initOne(.simd);
        for (unit_rows) |r| {
            const first = if (parsed.class == .p) r.from_phone else r.from;
            if (parsed.generation < first) continue;
            if (r.until) |last| if (parsed.generation > last) continue;
            chip.insert(r.unit);
            if (!withheld and os.atLeast(r.os.major, r.os.minor)) units.insert(r.unit);
        }
        return .{ .generation = parsed.generation, .class = parsed.class, .chip = chip, .units = units, .os = os, .withheld = withheld };
    }

    pub fn has(self: Profile, unit: Unit) bool {
        return self.units.contains(unit);
    }

    /// The chip has `unit` and only this OS keeps a kernel from it.
    pub fn osTooOldFor(self: Profile, unit: Unit) bool {
        const r = row(unit) orelse return false;
        return self.chip.contains(unit) and !self.os.atLeast(r.os.major, r.os.minor);
    }

    /// No kernel here has run on this generation: it is served as the newest one known.
    pub fn untested(self: Profile) bool {
        return self.generation > newest_known;
    }
};

/// A kernel's threadgroup sizes, the widest first. A GPU may grant a register-heavy kernel fewer threads than it asks;
/// the launch it refuses for size takes the next rung, and the kernel is written to give the same bytes at each.
pub const ThreadLadder = struct {
    rungs: []const u32,
    at: usize = 0,

    pub fn threads(self: ThreadLadder) u32 {
        return self.rungs[self.at];
    }

    /// The GPU refused the current rung: false when it was the last.
    pub fn stepDown(self: *ThreadLadder) bool {
        if (self.at + 1 >= self.rungs.len) return false;
        self.at += 1;
        return true;
    }
};

const Arch = struct { generation: u32 = 0, class: Class = .other };

fn parseArch(arch: []const u8) Arch {
    const none: Arch = .{};
    var buf: [64]u8 = undefined;
    if (arch.len == 0 or arch.len > buf.len) return none;
    const name = std.ascii.lowerString(&buf, arch);
    const tail = if (std.mem.startsWith(u8, name, "applegpu_g")) name[10..] else if (name[0] == 'g') name[1..] else return none;
    var digits: usize = 0;
    while (digits < tail.len and std.ascii.isDigit(tail[digits])) digits += 1;
    if (digits == 0 or tail.len - digits > 1) return none;
    if (digits < tail.len and !std.ascii.isAlphabetic(tail[digits])) return none;
    return .{
        .generation = std.fmt.parseInt(u32, tail[0..digits], 10) catch return none,
        .class = if (digits < tail.len) Class.of(tail[digits]) else .other,
    };
}

const State = enum(u8) { unread, reading, known };
var state = std.atomic.Value(State).init(.unread);
var known: Profile = undefined;
var replaced: ?Profile = null;

/// The profile of this process's GPU. The first call reads the device; any thread may make it.
pub fn current() Profile {
    if (replaced) |p| return p;
    while (true) switch (state.load(.acquire)) {
        .known => return known,
        .reading => std.atomic.spinLoopHint(),
        .unread => if (state.cmpxchgStrong(.unread, .reading, .acquire, .acquire) == null) {
            known = read();
            state.store(.known, .release);
        },
    };
}

/// A test's GPU: `current` answers `p` until the returned scope ends. Tests only, one thread.
pub fn withProfile(p: Profile) Scope {
    const before = replaced;
    replaced = p;
    return .{ .before = before };
}

pub const Scope = struct {
    before: ?Profile,

    pub fn end(self: Scope) void {
        replaced = self.before;
    }
};

/// This process's GPU with every unit past `simd` withheld, as the QA switch gives it.
pub fn simdOnly() Profile {
    const p = current();
    return .{ .generation = p.generation, .class = p.class, .chip = p.chip, .units = Units.initOne(.simd), .os = p.os, .withheld = true };
}

fn read() Profile {
    var arch_buf: [64]u8 = undefined;
    var os_buf: [64]u8 = undefined;
    return .from(archName(&arch_buf), Os.parse(osVersion(&os_buf)), envIsOne("SUSHI_FORCE_GPU_FAMILY_FALLBACK"));
}

fn envIsOne(name: [*:0]const u8) bool {
    const value = std.mem.span(std.c.getenv(name) orelse return false);
    return value.len > 0 and value[0] == '1';
}

const id = ?*anyopaque;
extern "c" fn MTLCreateSystemDefaultDevice() id;
extern "c" fn objc_msgSend() void;
extern "c" fn sel_registerName(name: [*:0]const u8) ?*anyopaque;
extern "c" fn objc_autoreleasePoolPush() ?*anyopaque;
extern "c" fn objc_autoreleasePoolPop(pool: ?*anyopaque) void;
extern "c" fn sysctlbyname(name: [*:0]const u8, oldp: ?*anyopaque, oldlenp: ?*usize, newp: ?*const anyopaque, newlen: usize) c_int;

fn send(comptime R: type, target: id, name: [*:0]const u8) R {
    const f: *const fn (id, ?*anyopaque) callconv(.c) R = @ptrCast(&objc_msgSend);
    return f(target, sel_registerName(name));
}

/// The name MLX dispatches on: its own `MLX_METAL_GPU_ARCH` when that is set, else the system GPU's. Empty without a GPU.
/// For a log line or a store's identity; a decision asks `current`.
pub fn archName(buf: []u8) []const u8 {
    if (std.c.getenv("MLX_METAL_GPU_ARCH")) |forced| {
        const text = std.mem.span(forced);
        if (text.len > 0) return copyInto(buf, text);
    }
    const pool = objc_autoreleasePoolPush();
    defer objc_autoreleasePoolPop(pool);
    const gpu = MTLCreateSystemDefaultDevice() orelse return "";
    defer send(void, gpu, "release");
    const arch = send(id, gpu, "architecture") orelse return "";
    const name = send(id, arch, "name") orelse return "";
    const text = send(?[*:0]const u8, name, "UTF8String") orelse return "";
    return copyInto(buf, std.mem.span(text));
}

/// The OS product version, whole ("26.4.1"); empty when it cannot be read.
pub fn osVersion(buf: []u8) []const u8 {
    var len: usize = buf.len;
    if (sysctlbyname("kern.osproductversion", buf.ptr, &len, null, 0) != 0) return "";
    return std.mem.sliceTo(buf[0..@min(len, buf.len)], 0);
}

fn copyInto(buf: []u8, text: []const u8) []const u8 {
    const n = @min(text.len, buf.len);
    @memcpy(buf[0..n], text[0..n]);
    return buf[0..n];
}

const testing = std.testing;
const ok_os: Os = .{ .major = 26, .minor = 6 };

fn matrix(arch: []const u8, os: Os, held: bool) bool {
    return Profile.from(arch, os, held).has(.matrix_g17);
}

test "device: a name that is no Apple GPU generation has the SIMD arm alone" {
    for ([_][]const u8{ "", "garbage", "air64_v27", "AIR64_V27", "metal_27", "foo17", "gfoobar27", "g17dd", "applegpu_g", "applegpu_g17_extra" }) |arch| {
        const p = Profile.from(arch, ok_os, false);
        try testing.expectEqual(@as(u32, 0), p.generation);
        try testing.expect(p.units.eql(Units.initOne(.simd)));
        try testing.expect(p.chip.eql(Units.initOne(.simd)));
    }
}

test "device: the matrix units start at generation 17, at 18 on a phone, whatever the name's case or prefix" {
    for ([_][]const u8{ "applegpu_g17s", "AppleGPU_G17S", "applegpu_g17", "g17", "applegpu_g17d", "applegpu_g18s", "g18", "applegpu_g18p", "g18p", "applegpu_g20c" }) |arch| {
        try testing.expect(matrix(arch, ok_os, false));
    }
    for ([_][]const u8{ "applegpu_g13c", "applegpu_g14g", "applegpu_g15s", "applegpu_g16s", "applegpu_g16", "g16", "applegpu_g17p", "g17p" }) |arch| {
        try testing.expect(!matrix(arch, ok_os, false));
        try testing.expect(Profile.from(arch, ok_os, false).has(.simd));
    }
}

test "device: the generation and the class letter are read off the name" {
    const p = Profile.from("applegpu_g17s", ok_os, false);
    try testing.expectEqual(@as(u32, 17), p.generation);
    try testing.expectEqual(Class.s, p.class);
    try testing.expectEqual(Class.d, Profile.from("AppleGPU_G13D", ok_os, false).class);
    try testing.expectEqual(Class.p, Profile.from("g18p", ok_os, false).class);
    try testing.expectEqual(Class.other, Profile.from("applegpu_g17", ok_os, false).class);
    try testing.expectEqual(Class.other, Profile.from("applegpu_g17x", ok_os, false).class);
    try testing.expectEqual(@as(u32, 17), Profile.from("applegpu_g17x", ok_os, false).generation);
}

test "device: an OS older than a unit's floor keeps the chip's unit from its kernels, and the profile says which leg is missing" {
    try testing.expect(matrix("applegpu_g17d", Os.parse("26.2"), false));
    try testing.expect(matrix("applegpu_g17d", Os.parse("27.0"), false));
    const old = Profile.from("applegpu_g17s", Os.parse("26.1"), false);
    try testing.expect(!old.has(.matrix_g17));
    try testing.expect(old.chip.contains(.matrix_g17));
    try testing.expect(old.osTooOldFor(.matrix_g17));
    const m4 = Profile.from("applegpu_g16s", Os.parse("26.1"), false);
    try testing.expect(!m4.chip.contains(.matrix_g17));
    try testing.expect(!m4.osTooOldFor(.matrix_g17));
}

test "device: the QA switch withholds every unit past SIMD and leaves the chip's own list alone" {
    const p = Profile.from("applegpu_g17d", Os.parse("26.4"), true);
    try testing.expect(p.withheld);
    try testing.expect(p.units.eql(Units.initOne(.simd)));
    try testing.expect(p.chip.contains(.matrix_g17));
    try testing.expect(!p.osTooOldFor(.matrix_g17));
}

test "device: an OS version reads major and minor, and a part that is no number is 0" {
    try testing.expect(Os.parse("26.2").atLeast(26, 2));
    try testing.expect(Os.parse("26.4.1").atLeast(26, 2));
    try testing.expect(Os.parse("27").atLeast(26, 2));
    try testing.expect(!Os.parse("26.1").atLeast(26, 2));
    try testing.expect(!Os.parse("26").atLeast(26, 2));
    try testing.expect(!Os.parse("25.5").atLeast(26, 2));
    for ([_][]const u8{ "", "garbage", "26.x" }) |text| try testing.expect(!Os.parse(text).atLeast(26, 2));
    try testing.expect(Os.parse("26.3").atLeast(26, 3));
    try testing.expect(!Os.parse("26.2").atLeast(26, 3));
}

test "device: a generation past the newest known keeps its units and is reported as untested" {
    const later = Profile.from("applegpu_g19s", ok_os, false);
    try testing.expect(later.has(.matrix_g17));
    try testing.expect(later.untested());
    try testing.expect(!Profile.from("applegpu_g17s", ok_os, false).untested());
}

test "device: a test's profile answers until its scope ends, and scopes nest" {
    const real = current();
    const m1 = Profile.from("applegpu_g13c", ok_os, false);
    {
        const outer = withProfile(m1);
        defer outer.end();
        try testing.expectEqual(@as(u32, 13), current().generation);
        {
            const inner = withProfile(Profile.from("applegpu_g17s", ok_os, false));
            defer inner.end();
            try testing.expect(current().has(.matrix_g17));
        }
        try testing.expect(!current().has(.matrix_g17));
    }
    try testing.expectEqual(real.generation, current().generation);
    try testing.expect(real.units.eql(current().units));
}

test "device: on a box with a GPU the process's profile holds the generation of its name and the OS it runs" {
    var buf: [64]u8 = undefined;
    const name = archName(&buf);
    if (name.len == 0) return error.SkipZigTest;
    const p = current();
    try testing.expectEqual(parseArch(name).generation, p.generation);
    try testing.expect(p.generation >= 13);
    try testing.expect(p.os.major >= 26);
    try testing.expect(p.has(.simd));
}

test "device: the process's own profile with its units withheld keeps the chip and the OS" {
    const p = simdOnly();
    try testing.expect(p.withheld);
    try testing.expect(p.units.eql(Units.initOne(.simd)));
    try testing.expectEqual(current().generation, p.generation);
    try testing.expect(p.chip.eql(current().chip));
}

test "device: a thread ladder starts on its widest rung, steps down on a refusal and stops on the last" {
    var l: ThreadLadder = .{ .rungs = &.{ 1024, 512, 256 } };
    try testing.expectEqual(@as(u32, 1024), l.threads());
    try testing.expect(l.stepDown());
    try testing.expectEqual(@as(u32, 512), l.threads());
    try testing.expect(l.stepDown());
    try testing.expect(!l.stepDown());
    try testing.expectEqual(@as(u32, 256), l.threads());
}
