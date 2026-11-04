//! The "Port" field at the bottom of the Port submenu: a label, a text field and a Switch button in one custom
//! menu-item view. Parsing and the wording live in `sushibar_logic.zig`; applying a port is the app's (the
//! `target`'s `applyPort:`). Single instance: its controls are module-level.
const std = @import("std");
const logic = @import("sushibar_logic.zig");
const objc = @import("objc.zig");

const msg = objc.msg;
const id = objc.id;
const NSRect = objc.NSRect;

pub const width: f64 = 216;
pub const height: f64 = 54;

var field: id = null;
var hint: id = null;
var opaque_bg = false;
var invalid_shown = false;

fn color(name: [*:0]const u8) id {
    return msg(id, objc.class("NSColor"), name, .{});
}

fn systemFont(size: f64, mono_digits: bool) id {
    return msg(id, objc.class("NSFont"), if (mono_digits) "monospacedDigitSystemFontOfSize:weight:" else "systemFontOfSize:weight:", .{ size, @as(f64, 0) });
}

fn label(str: [*:0]const u8, frame: NSRect, size: f64, c: id) id {
    const l = msg(id, objc.class("NSTextField"), "labelWithString:", .{objc.nsString(str)});
    msg(void, l, "setFrame:", .{frame});
    msg(void, l, "setFont:", .{systemFont(size, false)});
    msg(void, l, "setTextColor:", .{c});
    return l;
}

fn drawRect(_: id, _: objc.SEL, _: NSRect) callconv(.c) void {
    if (!opaque_bg) return;
    msg(void, color("windowBackgroundColor"), "setFill", .{});
    msg(void, msg(id, objc.class("NSBezierPath"), "bezierPathWithRect:", .{NSRect{ .x = 0, .y = 0, .w = width, .h = height }}), "fill", .{});
}

/// A text field in a menu is only typed into once its window is key and it is first responder.
fn mouseDown(self: id, _: objc.SEL, _: id) callconv(.c) void {
    const win = msg(id, self, "window", .{});
    msg(void, win, "makeKeyWindow", .{});
    _ = msg(bool, win, "makeFirstResponder:", .{field});
}

fn acceptsFirstResponder(_: id, _: objc.SEL) callconv(.c) bool {
    return true;
}

/// `target` receives `applyPort:` when Return is pressed in the field or Switch is clicked, and
/// `controlTextDidChange:` as the user types.
pub fn create(target: id) id {
    const cls = objc.defineSubclass("NSView", "SushiBarPortField", &.{
        .{ .selector = "drawRect:", .imp = @ptrCast(&drawRect), .types = "v@:{CGRect={CGPoint=dd}{CGSize=dd}}" },
        .{ .selector = "mouseDown:", .imp = @ptrCast(&mouseDown), .types = "v@:@" },
        .{ .selector = "acceptsFirstResponder", .imp = @ptrCast(&acceptsFirstResponder), .types = "c@:" },
    });
    const view = msg(id, msg(id, cls, "alloc", .{}), "initWithFrame:", .{NSRect{ .x = 0, .y = 0, .w = width, .h = height }});
    msg(void, view, "addSubview:", .{label("Port", .{ .x = 14, .y = 28, .w = 34, .h = 17 }, 13, color("labelColor"))});

    field = msg(id, msg(id, objc.class("NSTextField"), "alloc", .{}), "initWithFrame:", .{NSRect{ .x = 50, .y = 24, .w = 76, .h = 24 }});
    msg(void, field, "setPlaceholderString:", .{objc.nsString(logic.port_placeholder)});
    msg(void, field, "setFont:", .{systemFont(13, true)});
    msg(void, field, "setTarget:", .{target});
    msg(void, field, "setAction:", .{objc.sel("applyPort:")});
    msg(void, field, "setDelegate:", .{target});
    msg(void, view, "addSubview:", .{field});

    const button = msg(id, msg(id, objc.class("NSButton"), "alloc", .{}), "initWithFrame:", .{NSRect{ .x = 134, .y = 23, .w = 70, .h = 26 }});
    msg(void, button, "setTitle:", .{objc.nsString("Switch")});
    msg(void, button, "setBezelStyle:", .{@as(isize, 1)});
    msg(void, button, "setControlSize:", .{@as(isize, 1)});
    msg(void, button, "setFont:", .{systemFont(12, false)});
    msg(void, button, "setTarget:", .{target});
    msg(void, button, "setAction:", .{objc.sel("applyPort:")});
    msg(void, view, "addSubview:", .{button});

    hint = label(logic.port_hint, .{ .x = 52, .y = 6, .w = 120, .h = 14 }, 11, color("systemRedColor"));
    msg(void, hint, "setHidden:", .{true});
    msg(void, view, "addSubview:", .{hint});
    return view;
}

/// What the user typed.
pub fn text() []const u8 {
    const s = msg(id, field, "stringValue", .{});
    return std.mem.span(msg([*:0]const u8, s, "UTF8String", .{}));
}

pub fn setText(s: [*:0]const u8) void {
    msg(void, field, "setStringValue:", .{objc.nsString(s)});
}

pub fn showInvalid() void {
    invalid_shown = true;
    msg(void, field, "setTextColor:", .{color("systemRedColor")});
    msg(void, hint, "setHidden:", .{false});
}

/// Back to the neutral look, after typing or after a port was taken.
pub fn clearInvalid() void {
    if (!invalid_shown) return;
    invalid_shown = false;
    msg(void, field, "setTextColor:", .{color("textColor")});
    msg(void, hint, "setHidden:", .{true});
}

pub fn clear() void {
    setText("");
    clearInvalid();
}

pub fn isInvalid() bool {
    return invalid_shown;
}

/// Draws the field in `dark` or light appearance into a PNG at 2x on an opaque background.
pub fn renderPng(view: id, path: [*:0]const u8, dark: bool) bool {
    opaque_bg = true;
    defer opaque_bg = false;
    const name = if (dark) "NSAppearanceNameDarkAqua" else "NSAppearanceNameAqua";
    msg(void, view, "setAppearance:", .{msg(id, objc.class("NSAppearance"), "appearanceNamed:", .{objc.nsString(name)})});
    const bounds = NSRect{ .x = 0, .y = 0, .w = width, .h = height };
    const rep = msg(id, view, "bitmapImageRepForCachingDisplayInRect:", .{bounds});
    if (rep == null) return false;
    msg(void, view, "cacheDisplayInRect:toBitmapImageRep:", .{ bounds, rep });
    const data = msg(id, rep, "representationUsingType:properties:", .{ @as(isize, 4), @as(id, null) });
    if (data == null) return false;
    return msg(bool, data, "writeToFile:atomically:", .{ objc.nsString(path), true });
}
