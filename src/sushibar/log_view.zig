//! The log window: the live server log of the watched port in one reused window. Which file, how much to read,
//! when it was truncated and when to follow the end are decided in `sushibar_logic.zig`; this file reads the file
//! and drives the text view. Single instance, main thread only.
const std = @import("std");
const logic = @import("sushibar_logic.zig");
const objc = @import("objc.zig");

const msg = objc.msg;
const id = objc.id;
const NSRect = objc.NSRect;
const NSSize = objc.NSSize;

/// Lines shown on open; a window past twice that drops its oldest lines in place.
const keep_lines: usize = 2000;
/// The most of a file read at once; an opening read takes its tail.
const read_cap: usize = 1 << 20;
const marker = "── log restarted ──\n";

const win_w: f64 = 780;
const win_h: f64 = 480;
const bar_h: f64 = 38;

const Range = extern struct { location: usize, length: usize };

fn Buf(comptime N: usize) type {
    return struct {
        data: [N]u8 = undefined,
        len: usize = 0,
        fn set(self: *@This(), s: []const u8) void {
            self.len = @min(s.len, N);
            @memcpy(self.data[0..self.len], s[0..self.len]);
        }
        fn get(self: *const @This()) []const u8 {
            return self.data[0..self.len];
        }
    };
}

var window: id = null;
var text_view: id = null;
var scroll: id = null;
var path_label: id = null;
var controller: id = null;

var shown_port: u16 = 0;
var shown_kind: logic.LogKind = .standard;
var shown_path: Buf(1024) = .{};
/// Nothing is on screen for the current source yet, or the window was closed since.
var stale = true;
var follow: logic.Follow = .{};
var lines: usize = 0;
var has_file = false;
var shown_has_file: ?bool = null;
var copy_button: id = null;
var console_button: id = null;
var reveal_button: id = null;
var new_lines: id = null;
var new_lines_shown = false;

fn z(buf: []u8, s: []const u8) [*:0]const u8 {
    const n = @min(s.len, buf.len - 1);
    @memcpy(buf[0..n], s[0..n]);
    buf[n] = 0;
    return @ptrCast(buf.ptr);
}

fn nsData(bytes: []const u8) id {
    const utf8: usize = 4;
    const latin1: usize = 5;
    for ([_]usize{ utf8, latin1 }) |enc| {
        const s = msg(id, msg(id, objc.class("NSString"), "alloc", .{}), "initWithBytes:length:encoding:", .{ bytes.ptr, bytes.len, enc });
        if (s != null) return s;
    }
    return null;
}

fn color(name: [*:0]const u8) id {
    return msg(id, objc.class("NSColor"), name, .{});
}

fn control(cls: [*:0]const u8, frame: NSRect) id {
    return msg(id, msg(id, objc.class(cls), "alloc", .{}), "initWithFrame:", .{frame});
}

fn textLength() usize {
    return msg(usize, msg(id, text_view, "textStorage", .{}), "length", .{});
}

fn followingEnd() bool {
    const visible = msg(NSRect, msg(id, scroll, "contentView", .{}), "bounds", .{});
    const doc = msg(NSRect, msg(id, scroll, "documentView", .{}), "frame", .{});
    return logic.atBottom(visible.y + visible.h, doc.h);
}

fn setBody(bytes: []const u8) void {
    const s = nsData(bytes);
    defer if (s != null) objc.release(s);
    msg(void, text_view, "setString:", .{s orelse objc.nsString("")});
    msg(void, text_view, "scrollToEndOfDocument:", .{@as(id, null)});
    showNewLines(false);
}

fn append(bytes: []const u8) void {
    const s = nsData(bytes) orelse return;
    defer objc.release(s);
    const follow_end = followingEnd();
    msg(void, text_view, "replaceCharactersInRange:withString:", .{ Range{ .location = textLength(), .length = 0 }, s });
    if (follow_end) msg(void, text_view, "scrollToEndOfDocument:", .{@as(id, null)});
    showNewLines(logic.newLinesPill(new_lines_shown, true, follow_end));
}

fn showNewLines(on: bool) void {
    if (on == new_lines_shown or new_lines == null) return;
    new_lines_shown = on;
    objc.fadeNext(msg(id, new_lines, "superview", .{}), 0.2);
    msg(void, new_lines, "setHidden:", .{!on});
}

fn newLinesAction(_: id, _: objc.SEL, _: id) callconv(.c) void {
    msg(void, text_view, "scrollToEndOfDocument:", .{@as(id, null)});
    showNewLines(false);
}

fn findAction(_: id, _: objc.SEL, sender: id) callconv(.c) void {
    msg(void, window, "makeFirstResponder:", .{text_view});
    msg(void, text_view, "performTextFinderAction:", .{sender});
}

// ── Window ──

/// Removes the oldest `n` lines without moving what the reader is looking at: the scroll position shifts by the
/// height removed, a follower stays at the end, and a selection below the cut stays selected.
fn trimTop(n: usize) void {
    const ns = msg(id, text_view, "string", .{});
    var cut: usize = 0;
    for (0..n) |_| {
        const r = msg(Range, ns, "lineRangeForRange:", .{Range{ .location = cut, .length = 0 }});
        cut = r.location + r.length;
    }
    const follow_end = followingEnd();
    const top = msg(NSRect, msg(id, scroll, "contentView", .{}), "bounds", .{}).y;
    const before = textHeight();
    msg(void, text_view, "replaceCharactersInRange:withString:", .{ Range{ .location = 0, .length = cut }, objc.nsString("") });
    if (follow_end) return msg(void, text_view, "scrollToEndOfDocument:", .{@as(id, null)});
    msg(void, text_view, "scrollPoint:", .{objc.NSPoint{ .x = 0, .y = @max(0, top - (before - textHeight())) }});
}

fn textHeight() f64 {
    const lm = msg(id, text_view, "layoutManager", .{});
    const tc = msg(id, text_view, "textContainer", .{});
    msg(void, lm, "ensureLayoutForTextContainer:", .{tc});
    return msg(NSRect, lm, "usedRectForTextContainer:", .{tc}).h;
}

const copied_flash_s: f64 = 1.2;

/// The selection, or the whole log when nothing is selected.
fn copyAction(_: id, _: objc.SEL, _: id) callconv(.c) void {
    const sel = msg(Range, text_view, "selectedRange", .{});
    const all = msg(id, text_view, "string", .{});
    objc.copyString(if (sel.length > 0) msg(id, all, "substringWithRange:", .{sel}) else all);
    objc.fadeNext(copy_button, 0.15);
    msg(void, copy_button, "setTitle:", .{objc.nsString("Copied")});
    msg(void, controller, "performSelector:withObject:afterDelay:", .{ objc.sel("copyDone:"), @as(id, null), copied_flash_s });
}

fn copyDone(_: id, _: objc.SEL, _: id) callconv(.c) void {
    objc.fadeNext(copy_button, 0.15);
    msg(void, copy_button, "setTitle:", .{objc.nsString("Copy")});
}

fn fileUrlPath() ?id {
    if (!has_file) return null;
    var b: [1100]u8 = undefined;
    return objc.nsString(z(&b, shown_path.get()));
}

fn consoleAction(_: id, _: objc.SEL, _: id) callconv(.c) void {
    const p = fileUrlPath() orelse return;
    const ws = msg(id, objc.class("NSWorkspace"), "sharedWorkspace", .{});
    _ = msg(bool, ws, "openFile:withApplication:", .{ p, objc.nsString("Console") });
}

fn revealAction(_: id, _: objc.SEL, _: id) callconv(.c) void {
    const p = fileUrlPath() orelse return;
    const ws = msg(id, objc.class("NSWorkspace"), "sharedWorkspace", .{});
    _ = msg(bool, ws, "selectFile:inFileViewerRootedAtPath:", .{ p, objc.nsString("") });
}

fn pushButton(title: [*:0]const u8, frame: NSRect, selector: [*:0]const u8) id {
    const b = control("NSButton", frame);
    msg(void, b, "setTitle:", .{objc.nsString(title)});
    msg(void, b, "setBezelStyle:", .{@as(isize, 1)});
    msg(void, b, "setControlSize:", .{@as(isize, 1)});
    msg(void, b, "setFont:", .{msg(id, objc.class("NSFont"), "systemFontOfSize:", .{@as(f64, 12)})});
    msg(void, b, "setTarget:", .{controller});
    msg(void, b, "setAction:", .{objc.sel(selector)});
    msg(void, b, "setAutoresizingMask:", .{@as(usize, 1 | 8)});
    return b;
}

fn build() void {
    if (window != null) return;
    controller = objc.defineClass("SushiBarLogController", &.{
        .{ .selector = "copyLog:", .imp = @ptrCast(&copyAction), .types = "v@:@" },
        .{ .selector = "copyDone:", .imp = @ptrCast(&copyDone), .types = "v@:@" },
        .{ .selector = "openConsole:", .imp = @ptrCast(&consoleAction), .types = "v@:@" },
        .{ .selector = "revealLog:", .imp = @ptrCast(&revealAction), .types = "v@:@" },
        .{ .selector = "newLines:", .imp = @ptrCast(&newLinesAction), .types = "v@:@" },
        .{ .selector = "findLog:", .imp = @ptrCast(&findAction), .types = "v@:@" },
    });
    controller = objc.new(controller);
    const style: usize = 1 | 2 | 4 | 8;
    window = msg(id, msg(id, objc.class("NSWindow"), "alloc", .{}), "initWithContentRect:styleMask:backing:defer:", .{ NSRect{ .x = 0, .y = 0, .w = win_w, .h = win_h }, style, @as(usize, 2), false });
    msg(void, window, "setReleasedWhenClosed:", .{false});
    msg(void, window, "setMinSize:", .{NSSize{ .width = 420, .height = 240 }});
    msg(void, window, "center", .{});
    _ = msg(bool, window, "setFrameAutosaveName:", .{objc.nsString("SushiBarLog")});
    const content = msg(id, window, "contentView", .{});

    scroll = msg(id, objc.class("NSTextView"), "scrollableTextView", .{});
    msg(void, scroll, "setFrame:", .{NSRect{ .x = 0, .y = 0, .w = win_w, .h = win_h - bar_h }});
    msg(void, scroll, "setAutoresizingMask:", .{@as(usize, 2 | 16)});
    text_view = msg(id, scroll, "documentView", .{});
    msg(void, text_view, "setEditable:", .{false});
    msg(void, text_view, "setSelectable:", .{true});
    msg(void, text_view, "setRichText:", .{false});
    msg(void, text_view, "setUsesFindBar:", .{true});
    msg(void, text_view, "setIncrementalSearchingEnabled:", .{true});
    msg(void, text_view, "setTextContainerInset:", .{NSSize{ .width = 8, .height = 6 }});
    msg(void, text_view, "setFont:", .{msg(id, objc.class("NSFont"), "monospacedSystemFontOfSize:weight:", .{ @as(f64, 11.5), @as(f64, 0) })});
    msg(void, content, "addSubview:", .{scroll});

    // An accent capsule over the text's lower right: a box draws the fill, a borderless button inside takes the click.
    const pill_w: f64 = 108;
    new_lines = control("NSBox", .{ .x = win_w - 20 - pill_w, .y = 16, .w = pill_w, .h = 26 });
    msg(void, new_lines, "setBoxType:", .{@as(isize, 4)});
    msg(void, new_lines, "setBorderWidth:", .{@as(f64, 0)});
    msg(void, new_lines, "setCornerRadius:", .{@as(f64, 13)});
    msg(void, new_lines, "setFillColor:", .{color("controlAccentColor")});
    msg(void, new_lines, "setContentViewMargins:", .{NSSize{ .width = 0, .height = 0 }});
    msg(void, new_lines, "setAutoresizingMask:", .{@as(usize, 1 | 32)});
    msg(void, new_lines, "setHidden:", .{true});
    const jump = pushButton("New lines", .{ .x = 0, .y = 0, .w = pill_w, .h = 26 }, "newLines:");
    msg(void, jump, "setBordered:", .{false});
    const arrow = msg(id, objc.class("NSImage"), "imageWithSystemSymbolName:accessibilityDescription:", .{ objc.nsString("arrow.down"), @as(id, null) });
    const weight = msg(id, objc.class("NSImageSymbolConfiguration"), "configurationWithPointSize:weight:", .{ @as(f64, 11), @as(f64, 0.3) });
    msg(void, jump, "setImage:", .{msg(id, arrow, "imageWithSymbolConfiguration:", .{weight})});
    msg(void, jump, "setImagePosition:", .{@as(isize, 2)});
    msg(void, jump, "setImageHugsTitle:", .{true});
    msg(void, jump, "setContentTintColor:", .{color("whiteColor")});
    msg(void, jump, "setAutoresizingMask:", .{@as(usize, 0)});
    msg(void, jump, "setToolTip:", .{objc.nsString("Lines arrived below: jump to the end and follow it")});
    msg(void, new_lines, "addSubview:", .{jump});
    msg(void, content, "addSubview:", .{new_lines});

    const y = win_h - bar_h + 7;
    path_label = msg(id, objc.class("NSTextField"), "labelWithString:", .{objc.nsString("")});
    msg(void, path_label, "setFrame:", .{NSRect{ .x = 12, .y = y + 2, .w = win_w - 24 - 4 * 90, .h = 16 }});
    msg(void, path_label, "setFont:", .{msg(id, objc.class("NSFont"), "systemFontOfSize:", .{@as(f64, 11)})});
    msg(void, path_label, "setTextColor:", .{color("secondaryLabelColor")});
    msg(void, path_label, "setLineBreakMode:", .{@as(isize, 5)});
    msg(void, path_label, "setAutoresizingMask:", .{@as(usize, 2 | 8)});
    msg(void, content, "addSubview:", .{path_label});

    var x = win_w - 12;
    const buttons = [_]struct { title: [*:0]const u8, w: f64, selector: [*:0]const u8, tip: [*:0]const u8 }{
        .{ .title = "Reveal", .w = 70, .selector = "revealLog:", .tip = "Show the log file in Finder" },
        .{ .title = "Open in Console", .w = 120, .selector = "openConsole:", .tip = "Open the log file in Console" },
        .{ .title = "Copy", .w = 60, .selector = "copyLog:", .tip = "Copy the selection, or the whole log when nothing is selected" },
        .{ .title = "Find", .w = 56, .selector = "findLog:", .tip = "Search the log (⌘F)" },
    };
    var find_button: id = null;
    for (buttons, [_]*id{ &reveal_button, &console_button, &copy_button, &find_button }) |b, slot| {
        x -= b.w;
        slot.* = pushButton(b.title, .{ .x = x, .y = y, .w = b.w, .h = 24 }, b.selector);
        msg(void, slot.*, "setToolTip:", .{objc.nsString(b.tip)});
        msg(void, content, "addSubview:", .{slot.*});
        x -= 6;
    }
    msg(void, find_button, "setTag:", .{@as(isize, 1)});
}

fn titleFor(port: u16, src: logic.LogSource) void {
    var tb: [64]u8 = undefined;
    const title = std.fmt.bufPrint(&tb, "sushi log — port {d}", .{port}) catch "sushi log";
    var zb: [96]u8 = undefined;
    msg(void, window, "setTitle:", .{objc.nsString(z(&zb, title))});
    var pb: [1100]u8 = undefined;
    const shown = switch (src.kind) {
        .standard, .given => src.path,
        .off => "no log file (--log-file off)",
        .relative => src.path,
    };
    msg(void, path_label, "setStringValue:", .{objc.nsString(z(&pb, shown))});
    msg(void, path_label, "setToolTip:", .{objc.nsString(z(&pb, shown))});
}

// ── Reading ──

const Opened = struct { fd: c_int, size: u64, file: u64 };

fn openLog(path: []const u8) ?Opened {
    var zb: [1100]u8 = undefined;
    const fd = std.c.open(z(&zb, path), .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return null;
    var st: std.c.Stat = undefined;
    if (std.c.fstat(fd, &st) != 0) {
        _ = std.c.close(fd);
        return null;
    }
    // The inode alone names the file: a rotation moves a new one into the path.
    return .{ .fd = fd, .size = @intCast(st.size), .file = @intCast(st.ino) };
}

fn readAt(fd: c_int, offset: u64, want: usize, buf: []u8) []u8 {
    const n = @min(want, buf.len);
    const got = std.c.pread(fd, buf.ptr, n, @intCast(offset));
    return buf[0..@intCast(@max(got, 0))];
}

/// Replaces the text with the last lines of the file.
fn reload(o: Opened, restarted: bool) void {
    const a = std.heap.page_allocator;
    const start: u64 = if (o.size > read_cap) o.size - read_cap else 0;
    const buf = a.alloc(u8, read_cap) catch return;
    defer a.free(buf);
    var chunk = readAt(o.fd, start, @intCast(o.size - start), buf);
    var chunk_at = start;
    // A read that began mid-file starts mid-line.
    if (start > 0) {
        const skip = if (std.mem.indexOfScalar(u8, chunk, '\n')) |nl| nl + 1 else chunk.len;
        chunk = chunk[skip..];
        chunk_at += skip;
    }
    const whole = chunk[0..logic.completeLen(chunk)];
    const tail = whole[logic.tailStart(whole, keep_lines)..];
    if (restarted) {
        var out = std.ArrayList(u8).initCapacity(a, marker.len + tail.len) catch return;
        defer out.deinit(a);
        out.appendSliceAssumeCapacity(marker);
        out.appendSliceAssumeCapacity(tail);
        setBody(out.items);
    } else setBody(tail);
    lines = logic.countLines(tail) + @intFromBool(restarted);
    follow = .{ .offset = chunk_at + whole.len, .file = o.file };
}

fn extend(o: Opened) void {
    const a = std.heap.page_allocator;
    const buf = a.alloc(u8, read_cap) catch return;
    defer a.free(buf);
    const chunk = readAt(o.fd, follow.offset, @intCast(o.size - follow.offset), buf);
    var take = logic.completeLen(chunk);
    // A line longer than one read would otherwise never be shown.
    if (take == 0 and chunk.len == read_cap) take = chunk.len;
    if (take == 0) return;
    append(chunk[0..take]);
    follow.offset += take;
    lines += logic.countLines(chunk[0..take]);
    const excess = logic.excessLines(lines, keep_lines);
    if (excess > 0) {
        trimTop(excess);
        lines -= excess;
    }
}

fn showNotice(port: u16, src: logic.LogSource) void {
    var nb: [600]u8 = undefined;
    setBody(logic.logNotice(&nb, port, src));
    has_file = false;
    follow = .{};
    lines = 0;
}

/// Brings the window up to date with `src`, the log of `port`: a new source or a first look loads the file's tail,
/// after that only what the file grew by is appended.
pub fn sync(port: u16, src: logic.LogSource) void {
    syncBody(port, src);
    showFileButtons();
}

fn syncBody(port: u16, src: logic.LogSource) void {
    const moved = stale or port != shown_port or src.kind != shown_kind or !std.mem.eql(u8, src.path, shown_path.get());
    if (moved) {
        shown_port = port;
        shown_kind = src.kind;
        shown_path.set(src.path);
        titleFor(port, src);
        follow = .{};
    }
    if (src.kind == .off or src.kind == .relative) {
        if (moved) showNotice(port, src);
        stale = false;
        return;
    }
    const o = openLog(src.path) orelse {
        if (moved or has_file) showNotice(port, src);
        stale = false;
        return;
    };
    defer _ = std.c.close(o.fd);
    has_file = true;
    const step = if (moved) logic.FollowStep.reload else logic.followStep(follow, o.size, o.file);
    switch (step) {
        .idle => {},
        .append => extend(o),
        .reload => reload(o, !moved and follow.file != 0),
    }
    if (new_lines_shown) showNewLines(logic.newLinesPill(true, false, followingEnd()));
    stale = false;
}

/// Open in Console and Reveal need a file to act on.
fn showFileButtons() void {
    if (shown_has_file == has_file) return;
    shown_has_file = has_file;
    msg(void, console_button, "setEnabled:", .{has_file});
    msg(void, reveal_button, "setEnabled:", .{has_file});
}

/// For a render: the reader scrolled away from the end.
pub fn scrollToTop() void {
    msg(void, text_view, "scrollToBeginningOfDocument:", .{@as(id, null)});
}

pub fn isVisible() bool {
    return window != null and msg(bool, window, "isVisible", .{});
}

/// Shows the window, in front, reading `src` afresh.
pub fn open(port: u16, src: logic.LogSource) void {
    build();
    stale = true;
    sync(port, src);
    const app = msg(id, objc.class("NSApplication"), "sharedApplication", .{});
    msg(void, app, "activateIgnoringOtherApps:", .{true});
    msg(void, window, "makeKeyAndOrderFront:", .{@as(id, null)});
}

/// Draws the window into a PNG at 2x in `dark` or light appearance: with `sample` as the log of `port`, or as it
/// stands after `sync` calls.
pub fn renderPng(path: [*:0]const u8, dark: bool, shown: ?struct { port: u16, src: logic.LogSource, sample: []const u8, has_file: bool = true }) bool {
    build();
    if (shown) |s| {
        titleFor(s.port, s.src);
        setBody(s.sample);
        has_file = s.has_file;
        showFileButtons();
    }
    const name = if (dark) "NSAppearanceNameDarkAqua" else "NSAppearanceNameAqua";
    msg(void, window, "setAppearance:", .{msg(id, objc.class("NSAppearance"), "appearanceNamed:", .{objc.nsString(name)})});
    // The frame view carries the title bar and the window background the content view does not draw itself.
    const content = msg(id, msg(id, window, "contentView", .{}), "superview", .{});
    msg(void, content, "layoutSubtreeIfNeeded", .{});
    msg(void, msg(id, text_view, "layoutManager", .{}), "ensureLayoutForTextContainer:", .{msg(id, text_view, "textContainer", .{})});
    const bounds = msg(NSRect, content, "bounds", .{});
    const rep = msg(id, content, "bitmapImageRepForCachingDisplayInRect:", .{bounds});
    if (rep == null) return false;
    msg(void, content, "cacheDisplayInRect:toBitmapImageRep:", .{ bounds, rep });
    const data = msg(id, rep, "representationUsingType:properties:", .{ @as(isize, 4), @as(id, null) });
    if (data == null) return false;
    return msg(bool, data, "writeToFile:atomically:", .{ objc.nsString(path), true });
}
