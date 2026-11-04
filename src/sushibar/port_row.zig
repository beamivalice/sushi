//! One row of the Port submenu as a custom menu-item view: a check for the watched port, its label, and, on a
//! port that can be forgotten, a small remove button that shows while the pointer is over the row; the default port
//! carries a "Default" tag instead. The row under the pointer is highlighted as a system menu item is. A click
//! closes the menu and then sends the app's target `pickRow:` or `removeRow:` with the row's view; the port is
//! read back with `portOf`. Rows are rebuilt together (`reset`), so the state pool is module-level.
const std = @import("std");
const objc = @import("objc.zig");

const msg = objc.msg;
const id = objc.id;
const NSRect = objc.NSRect;

extern var NSFontAttributeName: id;
extern var NSForegroundColorAttributeName: id;
extern var NSParagraphStyleAttributeName: id;

pub const min_width: f64 = 216;
const max_width: f64 = 440;
pub const height: f64 = 24;
const x_size: f64 = 16;
const tracking_enter_exit_always_visible: usize = 0x1 | 0x80 | 0x200;

pub const Row = struct { port: u16, label: []const u8, checked: bool, removable: bool };

const State = struct {
    view: id = null,
    x: id = null,
    port: u16 = 0,
    label: [112:0]u8 = @splat(0),
    checked: bool = false,
    removable: bool = false,
    hover: bool = false,
    x_hover: bool = false,
};

var rows: [32]State = @splat(.{});
var row_n: usize = 0;
var target: id = null;
var menu: id = null;
var opaque_bg = false;

fn stateOf(view: id) ?*State {
    for (rows[0..row_n]) |*r| if (r.view == view or r.x == view) return r;
    return null;
}

/// The port of the row whose view sent an action.
pub fn portOf(view: id) ?u16 {
    return if (stateOf(view)) |r| r.port else null;
}

/// Forgets every row: call before the rows' menu items are removed.
pub fn reset() void {
    row_n = 0;
}

var middle_style: id = null;

/// The label: a model name too long for the widest row loses its middle, not its end.
fn textAttrs(c: id) id {
    if (middle_style == null) {
        middle_style = objc.new(objc.class("NSMutableParagraphStyle"));
        msg(void, middle_style, "setLineBreakMode:", .{@as(isize, 5)});
    }
    const font = msg(id, objc.class("NSFont"), "systemFontOfSize:", .{@as(f64, 13)});
    var keys = [_]id{ NSFontAttributeName, NSForegroundColorAttributeName, NSParagraphStyleAttributeName };
    var vals = [_]id{ font, c, middle_style };
    return msg(id, objc.class("NSDictionary"), "dictionaryWithObjects:forKeys:count:", .{ @as([*]id, &vals), @as([*]id, &keys), @as(usize, 3) });
}

fn attrsOf(size: f64, c: id) id {
    const font = msg(id, objc.class("NSFont"), "systemFontOfSize:", .{size});
    var keys = [_]id{ NSFontAttributeName, NSForegroundColorAttributeName };
    var vals = [_]id{ font, c };
    return msg(id, objc.class("NSDictionary"), "dictionaryWithObjects:forKeys:count:", .{ @as([*]id, &vals), @as([*]id, &keys), @as(usize, 2) });
}

const tag = "Default";
const tag_size: f64 = 11;

fn tagWidth() f64 {
    return msg(objc.NSSize, objc.nsString(tag), "sizeWithAttributes:", .{attrsOf(tag_size, objc.color("labelColor"))}).width;
}

/// What the right end of a row holds: the remove button, or the default port's tag.
fn trailing(removable: bool) f64 {
    return if (removable) x_size + 12 else tagWidth() + 24;
}

/// The widest a row needs for its label, so a long model name is not cut when there is room.
pub fn widthFor(label: []const u8, removable: bool) f64 {
    var z: [112:0]u8 = undefined;
    const n = @min(label.len, z.len - 1);
    @memcpy(z[0..n], label[0..n]);
    z[n] = 0;
    const w = msg(objc.NSSize, objc.nsString(&z), "sizeWithAttributes:", .{textAttrs(objc.color("labelColor"))}).width;
    return std.math.clamp(w + 30 + 8 + trailing(removable), min_width, max_width);
}

fn drawRect(self: id, _: objc.SEL, _: NSRect) callconv(.c) void {
    const r = stateOf(self) orelse return;
    const bounds = msg(NSRect, self, "bounds", .{});
    if (opaque_bg) {
        msg(void, objc.color("windowBackgroundColor"), "setFill", .{});
        msg(void, msg(id, objc.class("NSBezierPath"), "bezierPathWithRect:", .{bounds}), "fill", .{});
    }
    if (r.hover) {
        msg(void, objc.color("controlAccentColor"), "setFill", .{});
        const inner = NSRect{ .x = 5, .y = 0, .w = bounds.w - 10, .h = bounds.h };
        msg(void, msg(id, objc.class("NSBezierPath"), "bezierPathWithRoundedRect:xRadius:yRadius:", .{ inner, @as(f64, 4), @as(f64, 4) }), "fill", .{});
    }
    const ink = objc.color(if (r.hover) "selectedMenuItemTextColor" else "labelColor");
    if (r.checked) {
        if (objc.symbolImage("checkmark", ink)) |img| msg(void, img, "drawInRect:", .{NSRect{ .x = 12, .y = (bounds.h - 11) / 2, .w = 11, .h = 11 }});
    }
    const text = objc.nsString(&r.label);
    const attrs = textAttrs(ink);
    const th = msg(objc.NSSize, text, "sizeWithAttributes:", .{attrs}).height;
    const room = bounds.w - 30 - 8 - trailing(r.removable);
    msg(void, text, "drawInRect:withAttributes:", .{ NSRect{ .x = 30, .y = (bounds.h - th) / 2, .w = room, .h = th }, attrs });
    if (r.removable) return;
    const tag_attrs = attrsOf(tag_size, if (r.hover) objc.withAlpha(ink, 0.8) else objc.color("secondaryLabelColor"));
    const tw = tagWidth();
    const tag_h = msg(objc.NSSize, objc.nsString(tag), "sizeWithAttributes:", .{tag_attrs}).height;
    msg(void, objc.nsString(tag), "drawAtPoint:withAttributes:", .{ objc.NSPoint{ .x = bounds.w - 14 - tw, .y = (bounds.h - tag_h) / 2 }, tag_attrs });
}

fn xDrawRect(self: id, _: objc.SEL, _: NSRect) callconv(.c) void {
    const r = stateOf(self) orelse return;
    const bounds = msg(NSRect, self, "bounds", .{});
    // It shows only on the highlighted row, so it is drawn in the highlight's text colour.
    const ink = objc.color("selectedMenuItemTextColor");
    msg(void, objc.withAlpha(ink, if (r.x_hover) 0.42 else 0.22), "setFill", .{});
    msg(void, msg(id, objc.class("NSBezierPath"), "bezierPathWithOvalInRect:", .{bounds}), "fill", .{});
    if (objc.symbolImage("xmark", ink)) |img| msg(void, img, "drawInRect:", .{NSRect{ .x = 4.5, .y = 4.5, .w = 7, .h = 7 }});
}

fn showX(r: *State) void {
    msg(void, r.x, "setHidden:", .{!(r.hover and r.removable)});
}

fn mouseEntered(self: id, _: objc.SEL, _: id) callconv(.c) void {
    const r = stateOf(self) orelse return;
    r.hover = true;
    showX(r);
    msg(void, r.view, "setNeedsDisplay:", .{true});
}

fn mouseExited(self: id, _: objc.SEL, _: id) callconv(.c) void {
    const r = stateOf(self) orelse return;
    r.hover = false;
    showX(r);
    msg(void, r.view, "setNeedsDisplay:", .{true});
}

fn xEntered(self: id, _: objc.SEL, _: id) callconv(.c) void {
    const r = stateOf(self) orelse return;
    r.x_hover = true;
    msg(void, self, "setNeedsDisplay:", .{true});
}

fn xExited(self: id, _: objc.SEL, _: id) callconv(.c) void {
    const r = stateOf(self) orelse return;
    r.x_hover = false;
    msg(void, self, "setNeedsDisplay:", .{true});
}

fn ignoreMove(_: id, _: bool) void {}

fn fire(self: id, selector: [*:0]const u8) void {
    msg(void, menu, "cancelTracking", .{});
    // After the menu has closed: the action rebuilds the menu these views belong to.
    msg(void, target, "performSelector:withObject:afterDelay:", .{ objc.sel(selector), self, @as(f64, 0) });
}

fn mouseDown(self: id, _: objc.SEL, _: id) callconv(.c) void {
    if (objc.trackClick(self, &ignoreMove)) fire(self, "pickRow:");
}

fn xMouseDown(self: id, _: objc.SEL, _: id) callconv(.c) void {
    if (objc.trackClick(self, &ignoreMove)) fire(self, "removeRow:");
}

fn acceptsFirstMouse(_: id, _: objc.SEL, _: id) callconv(.c) bool {
    return true;
}

fn trackingArea(view: id) void {
    const area = msg(id, msg(id, objc.class("NSTrackingArea"), "alloc", .{}), "initWithRect:options:owner:userInfo:", .{ NSRect{ .x = 0, .y = 0, .w = 0, .h = 0 }, tracking_enter_exit_always_visible, view, @as(id, null) });
    msg(void, view, "addTrackingArea:", .{area});
    objc.release(area);
}

var classes_made = false;
var row_class: id = null;
var x_class: id = null;

fn makeClasses() void {
    if (classes_made) return;
    classes_made = true;
    const draw_types = "v@:{CGRect={CGPoint=dd}{CGSize=dd}}";
    row_class = objc.defineSubclass("NSView", "SushiBarPortRow", &.{
        .{ .selector = "drawRect:", .imp = @ptrCast(&drawRect), .types = draw_types },
        .{ .selector = "mouseEntered:", .imp = @ptrCast(&mouseEntered), .types = "v@:@" },
        .{ .selector = "mouseExited:", .imp = @ptrCast(&mouseExited), .types = "v@:@" },
        .{ .selector = "mouseDown:", .imp = @ptrCast(&mouseDown), .types = "v@:@" },
        .{ .selector = "acceptsFirstMouse:", .imp = @ptrCast(&acceptsFirstMouse), .types = "c@:@" },
    });
    x_class = objc.defineSubclass("NSView", "SushiBarPortRowRemove", &.{
        .{ .selector = "drawRect:", .imp = @ptrCast(&xDrawRect), .types = draw_types },
        .{ .selector = "mouseEntered:", .imp = @ptrCast(&xEntered), .types = "v@:@" },
        .{ .selector = "mouseExited:", .imp = @ptrCast(&xExited), .types = "v@:@" },
        .{ .selector = "mouseDown:", .imp = @ptrCast(&xMouseDown), .types = "v@:@" },
        .{ .selector = "acceptsFirstMouse:", .imp = @ptrCast(&acceptsFirstMouse), .types = "c@:@" },
    });
}

/// A row view `w` points wide. `app_target` receives `pickRow:` and `removeRow:`; `top_menu` is closed by a click.
pub fn create(app_target: id, top_menu: id, row: Row, w: f64) id {
    makeClasses();
    target = app_target;
    menu = top_menu;
    if (row_n == rows.len) return null;
    const r = &rows[row_n];
    row_n += 1;
    r.* = .{ .port = row.port, .checked = row.checked, .removable = row.removable };
    const n = @min(row.label.len, r.label.len);
    @memcpy(r.label[0..n], row.label[0..n]);
    r.label[n] = 0;
    r.view = msg(id, msg(id, row_class, "alloc", .{}), "initWithFrame:", .{NSRect{ .x = 0, .y = 0, .w = w, .h = height }});
    trackingArea(r.view);
    r.x = msg(id, msg(id, x_class, "alloc", .{}), "initWithFrame:", .{NSRect{ .x = w - 12 - x_size, .y = (height - x_size) / 2, .w = x_size, .h = x_size }});
    trackingArea(r.x);
    msg(void, r.x, "setToolTip:", .{objc.nsString("Forget this port")});
    msg(void, r.x, "setHidden:", .{true});
    msg(void, r.view, "addSubview:", .{r.x});
    objc.release(r.x);
    return r.view;
}

/// The rows stacked in `dark` or light appearance at 2x on an opaque background; `hover` is the row the pointer
/// is over and `x_hover` whether it is over that row's remove button.
pub fn renderPng(list: []const Row, path: [*:0]const u8, dark: bool, hover: ?usize, x_hover: bool) bool {
    opaque_bg = true;
    defer opaque_bg = false;
    reset();
    var w: f64 = min_width;
    for (list) |r| w = @max(w, widthFor(r.label, r.removable));
    const total = height * @as(f64, @floatFromInt(list.len));
    const stack = msg(id, msg(id, objc.class("NSView"), "alloc", .{}), "initWithFrame:", .{NSRect{ .x = 0, .y = 0, .w = w, .h = total }});
    for (list, 0..) |r, i| {
        const v = create(null, null, r, w) orelse continue;
        msg(void, v, "setFrameOrigin:", .{objc.NSPoint{ .x = 0, .y = total - height * @as(f64, @floatFromInt(i + 1)) }});
        msg(void, stack, "addSubview:", .{v});
        const st = &rows[row_n - 1];
        st.hover = hover == i;
        st.x_hover = st.hover and x_hover;
        showX(st);
    }
    const name = if (dark) "NSAppearanceNameDarkAqua" else "NSAppearanceNameAqua";
    msg(void, stack, "setAppearance:", .{msg(id, objc.class("NSAppearance"), "appearanceNamed:", .{objc.nsString(name)})});
    const bounds = NSRect{ .x = 0, .y = 0, .w = w, .h = total };
    const rep = msg(id, stack, "bitmapImageRepForCachingDisplayInRect:", .{bounds});
    if (rep == null) return false;
    msg(void, stack, "cacheDisplayInRect:toBitmapImageRep:", .{ bounds, rep });
    const data = msg(id, rep, "representationUsingType:properties:", .{ @as(isize, 4), @as(id, null) });
    if (data == null) return false;
    return msg(bool, data, "writeToFile:atomically:", .{ objc.nsString(path), true });
}
