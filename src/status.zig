const std = @import("std");
const builtin = @import("builtin");
const is_macos = builtin.os.tag == .macos;

// ── macOS Mach externs ──

extern "c" var mach_task_self_: u32;
extern "c" fn mach_host_self() u32;
extern "c" fn task_info(task: u32, flavor: u32, info: [*]i32, cnt: *u32) i32;
extern "c" fn host_statistics(host: u32, flavor: u32, info: [*]i32, cnt: *u32) i32;
extern "c" fn host_statistics64(host: u32, flavor: u32, info: [*]i32, cnt: *u32) i32;
extern "c" fn host_page_size(host: u32, out: *usize) i32;
extern "c" fn sysctlbyname(name: [*:0]const u8, oldp: ?*anyopaque, oldlenp: ?*usize, newp: ?*const anyopaque, newlen: usize) c_int;

// ── IOKit / CoreFoundation externs ──

extern "c" fn IOServiceMatching(name: [*:0]const u8) ?*anyopaque;
extern "c" fn IOServiceGetMatchingServices(port: u32, matching: ?*anyopaque, iter: *u32) i32;
extern "c" fn IOIteratorNext(iter: u32) u32;
extern "c" fn IORegistryEntryCreateCFProperties(entry: u32, props: *?*anyopaque, alloc: ?*anyopaque, opts: u32) i32;
extern "c" fn IOObjectRelease(obj: u32) i32;
extern "c" fn CFDictionaryGetValue(dict: ?*const anyopaque, key: ?*const anyopaque) ?*const anyopaque;
extern "c" fn CFStringCreateWithCString(alloc: ?*anyopaque, s: [*:0]const u8, enc: u32) ?*const anyopaque;
extern "c" fn CFNumberGetValue(num: ?*const anyopaque, typ: u32, out: *anyopaque) u8;
extern "c" fn CFRelease(cf: ?*const anyopaque) void;

// ── Mach struct layouts (extern = C ABI) ──

const TaskBasicInfo = extern struct {
    virtual_size: u64,
    resident_size: u64,
    resident_size_max: u64,
    user_time_sec: i32,
    user_time_usec: i32,
    sys_time_sec: i32,
    sys_time_usec: i32,
    policy: i32,
    suspend_count: i32,
};

/// task_vm_info truncated through phys_footprint (rev1). Field order matches
/// <mach/task_info.h> exactly; @sizeOf(TaskVmInfo)/@sizeOf(i32) == 38 ==
/// TASK_VM_INFO_REV1_COUNT, so the kernel fills through phys_footprint without
/// overrunning the buffer.
const TaskVmInfo = extern struct {
    virtual_size: u64,
    region_count: i32,
    page_size: i32,
    resident_size: u64,
    resident_size_peak: u64,
    device: u64,
    device_peak: u64,
    internal: u64,
    internal_peak: u64,
    external: u64,
    external_peak: u64,
    reusable: u64,
    reusable_peak: u64,
    purgeable_volatile_pmap: u64,
    purgeable_volatile_resident: u64,
    purgeable_volatile_virtual: u64,
    compressed: u64,
    compressed_peak: u64,
    compressed_lifetime: u64,
    phys_footprint: u64,
};

const CpuLoadInfo = extern struct {
    ticks: [4]u32, // user, system, idle, nice
};

const VmStats64 = extern struct {
    free_count: u32,
    active_count: u32,
    inactive_count: u32,
    wire_count: u32,
    zero_fill_count: u64,
    reactivations: u64,
    pageins: u64,
    pageouts: u64,
    faults: u64,
    cow_faults: u64,
    lookups: u64,
    hits: u64,
    purges: u64,
    purgeable_count: u32,
    speculative_count: u32,
    decompressions: u64,
    compressions: u64,
    swapins: u64,
    swapouts: u64,
    compressor_page_count: u32,
    throttled_count: u32,
    external_page_count: u32,
    internal_page_count: u32,
    total_uncompressed_pages_in_compressor: u64,
};

// ── CPU delta tracking (module-level state) ──
var prev_ticks: [4]u64 = @splat(0);

// ── Public metric helpers ──

pub fn getAppRssMb() u32 {
    var info = std.mem.zeroes(TaskBasicInfo);
    var count: u32 = @sizeOf(TaskBasicInfo) / @sizeOf(i32);
    if (task_info(mach_task_self_, 20, @ptrCast(&info), &count) != 0) return 0;
    return @intCast(info.resident_size / (1024 * 1024));
}

/// Process physical memory footprint in MB (TASK_VM_INFO flavor 22). Unlike
/// resident_size, this includes MLX's Metal/IOKit + compressed memory — the
/// only figure that reflects a loaded model's true footprint on Apple Silicon.
pub fn getAppMemFootprintMb() u32 {
    var info = std.mem.zeroes(TaskVmInfo);
    var count: u32 = @sizeOf(TaskVmInfo) / @sizeOf(i32); // 38 = TASK_VM_INFO_REV1_COUNT
    if (task_info(mach_task_self_, 22, @ptrCast(&info), &count) != 0) return 0;
    return @intCast(info.phys_footprint / (1024 * 1024));
}

/// Bytes the OS will grant a new wiring: free (less speculative, which `external` carries
/// back), file-backed and purgeable pages (volatile anonymous memory the OS drops on demand), capped
/// by the RAM outside wired + compressor. Mach's page classes overlap: a resident model's GPU-wired pages sit in `wired` AND in the anonymous
/// set, so subtracting `wired + compressor + internal` charges them twice. Anonymous pressure
/// still shows, as it consumes free pages and raises wired. 0 on a failed query.
fn computeAvailableBytes(total_mem: u64, free_pages: u64, speculative_pages: u64, external_pages: u64, wire_pages: u64, compressor_pages: u64, purgeable_pages: u64, page: u64) u64 {
    if (total_mem == 0) return 0;
    const reclaimable: u64 = (free_pages -| speculative_pages) +| external_pages +| purgeable_pages;
    const outside: u64 = total_mem -| (wire_pages +| compressor_pages) *| page;
    return @min(reclaimable *| page, outside);
}

extern "c" fn os_proc_available_memory() usize;

/// Per-process memory headroom before jetsam (iOS only — the entitlement-
/// aware figure that actually governs whether a big allocation survives).
/// Returns 0 on macOS, where the concept doesn't apply; the symbol is only
/// referenced on iOS builds so macOS links are unaffected.
pub fn getProcAvailableMemBytes() u64 {
    if (comptime builtin.os.tag != .ios) return 0;
    return @intCast(os_proc_available_memory());
}

/// Total physical RAM (hw.memsize; works on macOS and iOS). 0 on failure.
pub fn getTotalMemBytes() u64 {
    var total_mem: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (sysctlbyname("hw.memsize", @ptrCast(&total_mem), &len, null, 0) != 0) return 0;
    return total_mem;
}

/// Bytes the kernel holds wired right now (Metal's resident buffers among them). 0 on failure.
pub fn getWiredMemBytes() u64 {
    var page: usize = 0;
    if (host_page_size(mach_host_self(), &page) != 0) return 0;
    var vm = std.mem.zeroes(VmStats64);
    var count: u32 = @sizeOf(VmStats64) / @sizeOf(i32);
    if (host_statistics64(mach_host_self(), 4, @ptrCast(&vm), &count) != 0) return 0;
    return @as(u64, vm.wire_count) * page;
}

pub fn getAvailableMemBytes() u64 {
    var total_mem: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (sysctlbyname("hw.memsize", @ptrCast(&total_mem), &len, null, 0) != 0) return 0;

    var page: usize = 0;
    if (host_page_size(mach_host_self(), &page) != 0) return 0;

    var vm = std.mem.zeroes(VmStats64);
    var count: u32 = @sizeOf(VmStats64) / @sizeOf(i32);
    if (host_statistics64(mach_host_self(), 4, @ptrCast(&vm), &count) != 0) return 0;

    return computeAvailableBytes(total_mem, vm.free_count, vm.speculative_count, vm.external_page_count, vm.wire_count, vm.compressor_page_count, vm.purgeable_count, page);
}

test "computeAvailableBytes sums the reclaimable classes, capped by wired headroom" {
    const GB: u64 = 1024 * 1024 * 1024;
    const page: u64 = 16384;
    const ppg: u64 = GB / page; // pages per GB

    // free_count already contains the speculative pages, which are file-backed inside external.
    try std.testing.expectEqual(@as(u64, 32 * GB), computeAvailableBytes(128 * GB, 8 * ppg, 2 * ppg, 26 * ppg, 83 * ppg, ppg, 0, page));

    // The cap: wired + compressor bound the figure.
    try std.testing.expectEqual(@as(u64, 6 * GB), computeAvailableBytes(16 * GB, 6 * ppg, 0, 6 * ppg, 9 * ppg, ppg, 0, page));

    // A 7 GB model resident in the anonymous set consumed the free and file pages of a 16 GB
    // box (9 GB anonymous, 3 wired, 1 compressed), so a second 7 GB load is still refused.
    try std.testing.expectEqual(@as(u64, 3 * GB), computeAvailableBytes(16 * GB, 2 * ppg, 0, 1 * ppg, 3 * ppg, 1 * ppg, 0, page));

    // Disagreeing counters never underflow; a failed query or a box with no headroom reports 0.
    try std.testing.expectEqual(@as(u64, 4 * GB), computeAvailableBytes(16 * GB, ppg, 5 * ppg, 4 * ppg, 0, 0, 0, page));
    try std.testing.expectEqual(@as(u64, 0), computeAvailableBytes(0, 1, 1, 1, 1, 1, 0, page));
    try std.testing.expectEqual(@as(u64, 0), computeAvailableBytes(16 * GB, 6 * ppg, 0, 6 * ppg, 16 * ppg, 0, 0, page));
}

test "computeAvailableBytes counts purgeable pages as reclaimable, still capped by wired headroom" {
    const GB: u64 = 1024 * 1024 * 1024;
    const page: u64 = 16384;
    const ppg: u64 = GB / page;
    try std.testing.expectEqual(@as(u64, 5 * GB), computeAvailableBytes(64 * GB, 2 * ppg, 0, 1 * ppg, 8 * ppg, ppg, 2 * ppg, page));
    try std.testing.expectEqual(@as(u64, 4 * GB), computeAvailableBytes(16 * GB, 2 * ppg, 0, 1 * ppg, 12 * ppg, 0, 3 * ppg, page));
}

test "computeAvailableBytes: a resident model's wired pages are not charged a second time" {
    // Counters read on a 128 GiB Mac with GLM-5.3-Flash resident (a 93 GB process). Wired,
    // anonymous, free and file-backed pages sum to 9,376,543 pages of 8,388,608: the classes overlap.
    const page: u64 = 16384;
    const free_count: u64 = 924_691; // vm_stat prints it less the 19,860 speculative pages
    const speculative: u64 = 19_860;
    const external: u64 = 356_617;
    const wired: u64 = 6_259_311;
    const compressor: u64 = 32_238;
    // What the OS grants a new wiring: the free and file-backed pages, 19.25 GiB. Charging the
    // anonymous pages (1,823,546) on top of `wired` left 4.54 GiB.
    try std.testing.expectEqual((free_count - speculative + external) * page, computeAvailableBytes(8_388_608 * page, free_count, speculative, external, wired, compressor, 0, page));
}

pub fn getSysMemPct() u32 {
    var total_mem: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (sysctlbyname("hw.memsize", @ptrCast(&total_mem), &len, null, 0) != 0) return 0;

    var page: usize = 0;
    if (host_page_size(mach_host_self(), &page) != 0) return 0;

    var vm = std.mem.zeroes(VmStats64);
    var count: u32 = @sizeOf(VmStats64) / @sizeOf(i32);
    if (host_statistics64(mach_host_self(), 4, @ptrCast(&vm), &count) != 0) return 0;

    const used: u64 = (@as(u64, vm.active_count) + vm.wire_count + vm.compressor_page_count) * page;
    if (total_mem == 0) return 0;
    return @intCast(used * 100 / total_mem);
}

pub fn getCpuPct() u32 {
    var info = std.mem.zeroes(CpuLoadInfo);
    var count: u32 = 4;
    if (host_statistics(mach_host_self(), 3, @ptrCast(&info), &count) != 0) return 0;

    var total: u64 = 0;
    var idle: u64 = 0;
    for (0..4) |i| {
        const cur: u64 = info.ticks[i];
        const delta = cur -| prev_ticks[i];
        total += delta;
        if (i == 2) idle = delta;
        prev_ticks[i] = cur;
    }
    if (total == 0) return 0;
    return @intCast((total - idle) * 100 / total);
}

pub fn getGpuPct() u32 {
    // IOKit's IOServiceMatching/AGXAccelerator path is macOS-only; the symbols
    // aren't in the public iOS SDK (and apps are sandboxed from the GPU service
    // registry anyway). On iOS we report 0 — the value is only a log-line stat.
    if (comptime !is_macos) return 0;
    const matching = IOServiceMatching("AGXAccelerator") orelse return 0;
    var iter: u32 = 0;
    if (IOServiceGetMatchingServices(0, matching, &iter) != 0) return 0;
    defer _ = IOObjectRelease(iter);

    const entry = IOIteratorNext(iter);
    if (entry == 0) return 0;
    defer _ = IOObjectRelease(entry);

    var props: ?*anyopaque = null;
    if (IORegistryEntryCreateCFProperties(entry, &props, null, 0) != 0) return 0;
    defer if (props) |p| CFRelease(p);

    const perf = cfDictGet(props, "PerformanceStatistics") orelse return 0;
    const util = cfDictGet(perf, "Device Utilization %") orelse return 0;

    var value: i64 = 0;
    _ = CFNumberGetValue(util, 4, @ptrCast(&value));
    return if (value >= 0 and value <= 100) @intCast(value) else 0;
}

/// System-wide GPU memory in use (`AGXAccelerator` "In use system memory"): every process's
/// Metal allocations plus the driver's, the quantity `iogpu.wired_limit_mb` is enforced on.
/// 0 when IOKit is unavailable.
pub fn getGpuInUseSystemMemBytes() u64 {
    if (comptime !is_macos) return 0;
    const matching = IOServiceMatching("AGXAccelerator") orelse return 0;
    var iter: u32 = 0;
    if (IOServiceGetMatchingServices(0, matching, &iter) != 0) return 0;
    defer _ = IOObjectRelease(iter);

    const entry = IOIteratorNext(iter);
    if (entry == 0) return 0;
    defer _ = IOObjectRelease(entry);

    var props: ?*anyopaque = null;
    if (IORegistryEntryCreateCFProperties(entry, &props, null, 0) != 0) return 0;
    defer if (props) |p| CFRelease(p);

    const perf = cfDictGet(props, "PerformanceStatistics") orelse return 0;
    const in_use = cfDictGet(perf, "In use system memory") orelse return 0;

    var value: i64 = 0;
    _ = CFNumberGetValue(in_use, 4, @ptrCast(&value));
    return if (value > 0) @intCast(value) else 0;
}

fn cfDictGet(dict: ?*const anyopaque, key_name: [*:0]const u8) ?*const anyopaque {
    const key = CFStringCreateWithCString(null, key_name, 0x08000100) orelse return null;
    defer CFRelease(key);
    return CFDictionaryGetValue(dict, key);
}

test "getAppMemFootprintMb returns a plausible nonzero footprint" {
    const fp = getAppMemFootprintMb();
    // The test process itself footprints several MB; a wrong flavor/offset
    // would yield 0 or absurd garbage.
    try std.testing.expect(fp > 0);
    try std.testing.expect(fp < 1024 * 1024); // < 1 TB sanity bound
}
