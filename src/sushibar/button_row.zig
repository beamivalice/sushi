//! The menu's last row: Log, Update and Quit as three rounded buttons in one custom menu-item view. A button
//! highlights under the pointer and while pressed; a click closes the menu and then sends its selector to the
//! app's target. What the Update button says and does is decided in `sushibar_logic.zig` (`updateButton`).
//! Single instance: its state is module-level.
const std = @import("std");
const objc = @import("objc.zig");

const msg = objc.msg;
const id = objc.id;
const NSRect = objc.NSRect;

extern var NSFontAttributeName: id;
extern var NSForegroundColorAttributeName: id;

pub const width: f64 = 300;
pub const height: f64 = 40;
const pad: f64 = 10;
const gap: f64 = 6;
const button_h: f64 = 28;
const radius: f64 = 7;
const symbol_h: f64 = 14;
const symbol_gap: f64 = 5;

const tracking_enter_exit_always_visible: usize = 0x1 | 0x80 | 0x200;

pub const Look = struct { label: []const u8, symbol: []const u8, tip: []const u8 = "", accent: bool = false, enabled: bool = true };

const Btn = struct {
    view: id = null,
    selector: [*:0]const u8,
    label: [32:0]u8 = @splat(0),
    symbol: [48:0]u8 = @splat(0),
    accent: bool = false,
    enabled: bool = true,
    hover: bool = false,
    pressed: bool = false,
};

const log_i = 0;
const update_i = 1;
const quit_i = 2;

var btns = [3]Btn{
    .{ .selector = "showLog:" },
    .{ .selector = "updateButton:" },
    .{ .selector = "quit:" },
};
var target: id = null;
var menu: id = null;
var opaque_bg = false;

fn setZ(dst: []u8, s: []const u8) void {
    const n = @min(s.len, dst.len - 1);
    @memcpy(dst[0..n], s[0..n]);
    dst[n] = 0;
}

fn same(z: []const u8, s: []const u8) bool {
    return std.mem.eql(u8, std.mem.sliceTo(z, 0), s);
}

fn btnOf(view: id) ?*Btn {
    for (&btns) |*b| if (b.view == view) return b;
    return null;
}

const color = objc.color;
const alpha = objc.withAlpha;

fn fill(b: *const Btn) id {
    if (b.accent and b.enabled) {
        const accent = color("controlAccentColor");
        const shade: f64 = if (b.pressed) 0.28 else if (b.hover) 0.14 else 0;
        return if (shade == 0) accent else msg(id, accent, "blendedColorWithFraction:ofColor:", .{ shade, color("blackColor") });
    }
    const a: f64 = if (!b.enabled) 0.04 else if (b.pressed) 0.24 else if (b.hover) 0.15 else 0.07;
    return alpha(color("labelColor"), a);
}

fn ink(b: *const Btn) id {
    if (!b.enabled) return color("tertiaryLabelColor");
    return if (b.accent) color("whiteColor") else color("labelColor");
}

fn drawRect(self: id, _: objc.SEL, _: NSRect) callconv(.c) void {
    const b = btnOf(self) orelse return;
    const bounds = msg(NSRect, self, "bounds", .{});
    if (opaque_bg) {
        msg(void, color("windowBackgroundColor"), "setFill", .{});
        msg(void, msg(id, objc.class("NSBezierPath"), "bezierPathWithRect:", .{bounds}), "fill", .{});
    }
    msg(void, fill(b), "setFill", .{});
    msg(void, msg(id, objc.class("NSBezierPath"), "bezierPathWithRoundedRect:xRadius:yRadius:", .{ bounds, radius, radius }), "fill", .{});

    const c = ink(b);
    const font = msg(id, objc.class("NSFont"), "systemFontOfSize:weight:", .{ @as(f64, 12.5), @as(f64, 0.23) });
    var keys = [_]id{ NSFontAttributeName, NSForegroundColorAttributeName };
    var vals = [_]id{ font, c };
    const attrs = msg(id, objc.class("NSDictionary"), "dictionaryWithObjects:forKeys:count:", .{ @as([*]id, &vals), @as([*]id, &keys), @as(usize, 2) });
    const text = objc.nsString(&b.label);
    const tsize = msg(objc.NSSize, text, "sizeWithAttributes:", .{attrs});
    const img = objc.symbolImage(&b.symbol, c);
    const isize_ = if (img != null) msg(objc.NSSize, img, "size", .{}) else objc.NSSize{ .width = 0, .height = 0 };
    const iw = if (isize_.height > 0) isize_.width * symbol_h / isize_.height else 0;
    const total = iw + (if (iw > 0) symbol_gap else 0) + tsize.width;
    var x = (bounds.w - total) / 2;
    if (img != null) {
        msg(void, img, "drawInRect:", .{NSRect{ .x = x, .y = (bounds.h - symbol_h) / 2, .w = iw, .h = symbol_h }});
        x += iw + symbol_gap;
    }
    msg(void, text, "drawAtPoint:withAttributes:", .{ objc.NSPoint{ .x = x, .y = (bounds.h - tsize.height) / 2 }, attrs });
}

fn redraw(b: *const Btn) void {
    msg(void, b.view, "setNeedsDisplay:", .{true});
}

fn mouseEntered(self: id, _: objc.SEL, _: id) callconv(.c) void {
    const b = btnOf(self) orelse return;
    b.hover = true;
    redraw(b);
}

fn mouseExited(self: id, _: objc.SEL, _: id) callconv(.c) void {
    const b = btnOf(self) orelse return;
    b.hover = false;
    redraw(b);
}

fn acceptsFirstMouse(_: id, _: objc.SEL, _: id) callconv(.c) bool {
    return true;
}

fn pressedWhileInside(view: id, inside: bool) void {
    const b = btnOf(view) orelse return;
    b.pressed = inside;
    msg(void, view, "display", .{});
}

/// A button stays pressed only while the pointer is over it; releasing over it runs the action.
fn mouseDown(self: id, _: objc.SEL, _: id) callconv(.c) void {
    const b = btnOf(self) orelse return;
    if (!b.enabled) return;
    if (!objc.trackClick(self, &pressedWhileInside)) return redraw(b);
    b.hover = false;
    msg(void, self, "display", .{});
    msg(void, menu, "cancelTracking", .{});
    // After the menu has closed: an action that opens a window or an alert must not run inside menu tracking.
    msg(void, target, "performSelector:withObject:afterDelay:", .{ objc.sel(b.selector), self, @as(f64, 0) });
}

/// `target` receives `showLog:`, `updateButton:` and `quit:`; `top_menu` is closed when one is clicked.
pub fn create(app_target: id, top_menu: id) id {
    target = app_target;
    menu = top_menu;
    const cls = objc.defineSubclass("NSView", "SushiBarMenuButton", &.{
        .{ .selector = "drawRect:", .imp = @ptrCast(&drawRect), .types = "v@:{CGRect={CGPoint=dd}{CGSize=dd}}" },
        .{ .selector = "mouseEntered:", .imp = @ptrCast(&mouseEntered), .types = "v@:@" },
        .{ .selector = "mouseExited:", .imp = @ptrCast(&mouseExited), .types = "v@:@" },
        .{ .selector = "mouseDown:", .imp = @ptrCast(&mouseDown), .types = "v@:@" },
        .{ .selector = "acceptsFirstMouse:", .imp = @ptrCast(&acceptsFirstMouse), .types = "c@:@" },
    });
    const row = msg(id, msg(id, objc.class("NSView"), "alloc", .{}), "initWithFrame:", .{NSRect{ .x = 0, .y = 0, .w = width, .h = height }});
    const bw = (width - 2 * pad - 2 * gap) / 3;
    const looks = [_]Look{
        .{ .label = "Log", .symbol = "doc.text", .tip = "Show the server log" },
        .{ .label = "Check", .symbol = "arrow.triangle.2.circlepath", .tip = "Check for Updates" },
        .{ .label = "Quit", .symbol = "power", .tip = "Quit sushibar (⌘Q)" },
    };
    for (&btns, looks, 0..) |*b, look, i| {
        const frame = NSRect{ .x = pad + @as(f64, @floatFromInt(i)) * (bw + gap), .y = (height - button_h) / 2, .w = bw, .h = button_h };
        b.view = msg(id, msg(id, cls, "alloc", .{}), "initWithFrame:", .{frame});
        const area = msg(id, msg(id, objc.class("NSTrackingArea"), "alloc", .{}), "initWithRect:options:owner:userInfo:", .{ NSRect{ .x = 0, .y = 0, .w = 0, .h = 0 }, tracking_enter_exit_always_visible, b.view, @as(id, null) });
        msg(void, b.view, "addTrackingArea:", .{area});
        msg(void, row, "addSubview:", .{b.view});
        set(i, look);
    }
    return row;
}

fn set(i: usize, look: Look) void {
    const b = &btns[i];
    if (same(&b.label, look.label) and same(&b.symbol, look.symbol) and b.accent == look.accent and b.enabled == look.enabled) return;
    setZ(&b.label, look.label);
    setZ(&b.symbol, look.symbol);
    b.accent = look.accent;
    b.enabled = look.enabled;
    var tip: [128:0]u8 = undefined;
    setZ(&tip, look.tip);
    msg(void, b.view, "setToolTip:", .{objc.nsString(&tip)});
    redraw(b);
}

pub fn setUpdate(look: Look) void {
    set(update_i, look);
}

/// A menu that closed under the pointer sends no exit: the highlight is dropped once the row is off screen.
pub fn dropStaleHover() void {
    for (&btns) |*b| {
        if (!b.hover or b.view == null) continue;
        const win = msg(id, b.view, "window", .{});
        if (win != null and msg(bool, win, "isVisible", .{})) continue;
        b.hover = false;
        redraw(b);
    }
}

/// The three buttons in `dark` or light appearance at 2x on an opaque background; `hover` and `pressed` pick a
/// button (0 Log, 1 Update, 2 Quit) to draw in that state.
pub fn renderPng(row: id, path: [*:0]const u8, dark: bool, hover: ?usize, pressed: ?usize) bool {
    opaque_bg = true;
    defer opaque_bg = false;
    for (&btns, 0..) |*b, i| {
        b.hover = hover == i;
        b.pressed = pressed == i;
    }
    defer for (&btns) |*b| {
        b.hover = false;
        b.pressed = false;
    };
    const name = if (dark) "NSAppearanceNameDarkAqua" else "NSAppearanceNameAqua";
    msg(void, row, "setAppearance:", .{msg(id, objc.class("NSAppearance"), "appearanceNamed:", .{objc.nsString(name)})});
    const bounds = NSRect{ .x = 0, .y = 0, .w = width, .h = height };
    const rep = msg(id, row, "bitmapImageRepForCachingDisplayInRect:", .{bounds});
    if (rep == null) return false;
    msg(void, row, "cacheDisplayInRect:toBitmapImageRep:", .{ bounds, rep });
    const data = msg(id, rep, "representationUsingType:properties:", .{ @as(isize, 4), @as(id, null) });
    if (data == null) return false;
    return msg(bool, data, "writeToFile:atomically:", .{ objc.nsString(path), true });
}
