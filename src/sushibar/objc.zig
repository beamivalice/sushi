//! The Objective-C runtime from Zig, with no Objective-C source: `msg` types each `objc_msgSend` call, and a
//! class whose methods are Zig `callconv(.c)` functions is registered at run time.

const std = @import("std");

pub const id = ?*anyopaque;
pub const SEL = ?*anyopaque;

extern fn objc_getClass(name: [*:0]const u8) id;
extern fn sel_registerName(name: [*:0]const u8) SEL;
extern fn objc_msgSend() void;
extern fn objc_allocateClassPair(super: id, name: [*:0]const u8, extra: usize) id;
extern fn objc_registerClassPair(cls: id) void;
extern fn class_addMethod(cls: id, sel: SEL, imp: *const anyopaque, types: [*:0]const u8) bool;
extern var NSRunLoopCommonModes: id;
extern var NSPasteboardTypeString: id;
extern var NSEventTrackingRunLoopMode: id;

pub fn class(name: [*:0]const u8) id {
    return objc_getClass(name);
}

pub fn sel(name: [*:0]const u8) SEL {
    return sel_registerName(name);
}

/// `[target sel:args...]` with the argument types taken from `args` and the result type `R`.
pub fn msg(comptime R: type, target: id, selector: [*:0]const u8, args: anytype) R {
    const T = @typeInfo(@TypeOf(args)).@"struct".field_types;
    const F = switch (T.len) {
        0 => *const fn (id, SEL) callconv(.c) R,
        1 => *const fn (id, SEL, T[0]) callconv(.c) R,
        2 => *const fn (id, SEL, T[0], T[1]) callconv(.c) R,
        3 => *const fn (id, SEL, T[0], T[1], T[2]) callconv(.c) R,
        4 => *const fn (id, SEL, T[0], T[1], T[2], T[3]) callconv(.c) R,
        5 => *const fn (id, SEL, T[0], T[1], T[2], T[3], T[4]) callconv(.c) R,
        10 => *const fn (id, SEL, T[0], T[1], T[2], T[3], T[4], T[5], T[6], T[7], T[8], T[9]) callconv(.c) R,
        else => @compileError("objc.msg: more arguments than a sushibar call uses"),
    };
    const f: F = @ptrCast(&objc_msgSend);
    return @call(.auto, f, .{ target, sel_registerName(selector) } ++ args);
}

/// An autoreleased NSString copy of `s`.
pub fn nsString(s: [*:0]const u8) id {
    return msg(id, class("NSString"), "stringWithUTF8String:", .{s});
}

/// An autoreleased NSString of all of the UTF-8 `s`. Text shown to the user goes through this, never a fixed C-string
/// buffer: a cut loses the end, and a cut inside a character makes the whole string nil.
pub fn nsText(s: []const u8) id {
    const utf8: usize = 4;
    const str = msg(id, msg(id, class("NSString"), "alloc", .{}), "initWithBytes:length:encoding:", .{ s.ptr, s.len, utf8 });
    return msg(id, str, "autorelease", .{});
}

/// `s` retained in `slot`, whose old string is released; false when `slot` already held the same text.
pub fn keepText(slot: *id, s: []const u8) bool {
    const t = nsText(s);
    if (slot.* != null and msg(bool, slot.*, "isEqualToString:", .{t})) return false;
    if (slot.* != null) release(slot.*);
    slot.* = msg(id, t, "retain", .{});
    return true;
}

pub const Method = struct { selector: [*:0]const u8, imp: *const anyopaque, types: [*:0]const u8 };

pub const NSPoint = extern struct { x: f64, y: f64 };
pub const NSSize = extern struct { width: f64, height: f64 };
pub const NSRect = extern struct { x: f64, y: f64, w: f64, h: f64 };

/// A subclass of NSObject named `name` whose instance methods are `methods`.
pub fn defineClass(name: [*:0]const u8, methods: []const Method) id {
    return defineSubclass("NSObject", name, methods);
}

pub fn defineSubclass(super: [*:0]const u8, name: [*:0]const u8, methods: []const Method) id {
    const cls = objc_allocateClassPair(class(super), name, 0);
    for (methods) |m| _ = class_addMethod(cls, sel_registerName(m.selector), m.imp, m.types);
    objc_registerClassPair(cls);
    return cls;
}

pub fn release(o: id) void {
    msg(void, o, "release", .{});
}

pub fn new(cls: id) id {
    return msg(id, msg(id, cls, "alloc", .{}), "init", .{});
}

/// Calls `[target selector:timer]` every `seconds` on the main run loop, menu tracking included; `invalidate` stops it.
/// Sets a 20% timer tolerance so macOS can coalesce wakeups across processes.
pub fn addTimer(seconds: f64, target: id, selector: [*:0]const u8) id {
    const timer = msg(id, class("NSTimer"), "timerWithTimeInterval:target:selector:userInfo:repeats:", .{ seconds, target, sel_registerName(selector), @as(id, null), true });
    msg(void, timer, "setTolerance:", .{seconds * 0.2});
    msg(void, msg(id, class("NSRunLoop"), "mainRunLoop", .{}), "addTimer:forMode:", .{ timer, NSRunLoopCommonModes });
    return timer;
}

pub fn copyText(text: [*:0]const u8) void {
    copyString(nsString(text));
}

pub fn copyString(s: id) void {
    const pb = msg(id, class("NSPasteboard"), "generalPasteboard", .{});
    _ = msg(isize, pb, "clearContents", .{});
    _ = msg(bool, pb, "setString:forType:", .{ s, NSPasteboardTypeString });
}

/// An NSImage from SVG bytes, `w` x `h` points; a template image is tinted by the system.
pub fn loadSvg(svg: []const u8, template: bool, w: f64, h: f64) id {
    const data = msg(id, class("NSData"), "dataWithBytes:length:", .{ svg.ptr, svg.len });
    const image = msg(id, msg(id, class("NSImage"), "alloc", .{}), "initWithData:", .{data});
    msg(void, image, "setSize:", .{NSSize{ .width = w, .height = h }});
    msg(void, image, "setTemplate:", .{template});
    return image;
}

/// Rasterizes SVG bytes once into a retained 2x Retina `NSBitmapImageRep` (`w` x `h` points).
/// Retaining the rep keeps its backing `[rep CGImage]` valid for GPU `CALayer.contents` without re-parsing SVG.
pub fn rasterizeSvgRep(svg: []const u8, w: f64, h: f64) id {
    const img = loadSvg(svg, false, w, h);
    defer release(img);
    const px_w: isize = @intFromFloat(@round(w * 2.0));
    const px_h: isize = @intFromFloat(@round(h * 2.0));
    const rep = msg(id, msg(id, class("NSBitmapImageRep"), "alloc", .{}), "initWithBitmapDataPlanes:pixelsWide:pixelsHigh:bitsPerSample:samplesPerPixel:hasAlpha:isPlanar:colorSpaceName:bytesPerRow:bitsPerPixel:", .{
        @as(?*anyopaque, null),
        px_w,
        px_h,
        @as(isize, 8),
        @as(isize, 4),
        true,
        false,
        nsString("NSCalibratedRGBColorSpace"),
        @as(isize, 0),
        @as(isize, 0),
    });
    if (rep == null) return null;
    msg(void, rep, "setSize:", .{NSSize{ .width = w, .height = h }});
    const gc_cls = class("NSGraphicsContext");
    msg(void, gc_cls, "saveGraphicsState", .{});
    const ctx = msg(id, gc_cls, "graphicsContextWithBitmapImageRep:", .{rep});
    if (ctx != null) {
        msg(void, gc_cls, "setCurrentContext:", .{ctx});
        const zero = NSRect{ .x = 0, .y = 0, .w = 0, .h = 0 };
        msg(void, img, "drawInRect:fromRect:operation:fraction:", .{ NSRect{ .x = 0, .y = 0, .w = w, .h = h }, zero, @as(usize, 2), @as(f64, 1.0) });
    }
    msg(void, gc_cls, "restoreGraphicsState", .{});
    return rep;
}

/// Returns the `CGImage` owned by a retained `NSBitmapImageRep`.
pub fn repCgImage(rep: id) id {
    return if (rep != null) msg(id, rep, "CGImage", .{}) else null;
}

/// Wraps a rasterized `NSBitmapImageRep` in an `NSImage` of logical size `w` x `h` (e.g. for `renderBadge`).
pub fn imageFromRep(rep: id, w: f64, h: f64) id {
    const img = msg(id, msg(id, class("NSImage"), "alloc", .{}), "initWithSize:", .{NSSize{ .width = w, .height = h }});
    if (rep != null) msg(void, img, "addRepresentation:", .{rep});
    return img;
}

/// A transparent `w` x `h` `NSImage` that holds layout space on `NSStatusBarButton` while a `CALayer` draws the icon.
pub fn makeBlankImage(w: f64, h: f64) id {
    return msg(id, msg(id, class("NSImage"), "alloc", .{}), "initWithSize:", .{NSSize{ .width = w, .height = h }});
}

/// Runs a discrete repeating `CAKeyframeAnimation` on `layer.contents` entirely inside WindowServer's GPU compositor
/// (zero app timers, zero CPU wakeups, zero `NSStatusBarButton` redraws).
pub fn startContentsAnimation(layer: id, key: [*:0]const u8, cg_images: []const id, duration: f64) void {
    if (layer == null or cg_images.len == 0) return;
    const arr = msg(id, class("NSArray"), "arrayWithObjects:count:", .{ cg_images.ptr, cg_images.len });
    const anim = msg(id, class("CAKeyframeAnimation"), "animationWithKeyPath:", .{nsString("contents")});
    msg(void, anim, "setValues:", .{arr});
    msg(void, anim, "setDuration:", .{duration});
    msg(void, anim, "setCalculationMode:", .{nsString("discrete")});
    msg(void, anim, "setRepeatCount:", .{std.math.inf(f32)});
    msg(void, anim, "setRemovedOnCompletion:", .{false});
    msg(void, layer, "addAnimation:forKey:", .{ anim, nsString(key) });
}

pub fn stopLayerAnimation(layer: id, key: [*:0]const u8) void {
    if (layer == null) return;
    msg(void, layer, "removeAnimationForKey:", .{nsString(key)});
}

/// Disables implicit Core Animation transitions on `layer.contents` so manual frame steps snap cleanly.
pub fn disableContentsAction(layer: id) void {
    if (layer == null) return;
    const null_val = msg(id, class("NSNull"), "null", .{});
    const dict = msg(id, class("NSDictionary"), "dictionaryWithObject:forKey:", .{ null_val, nsString("contents") });
    msg(void, layer, "setActions:", .{dict});
}

/// The user asked the system for less motion (Accessibility > Display > Reduce motion).
pub fn reduceMotion() bool {
    return msg(bool, msg(id, class("NSWorkspace"), "sharedWorkspace", .{}), "accessibilityDisplayShouldReduceMotion", .{});
}

/// The next change `view` draws, its subviews' included, cross-fades in over `seconds` instead of snapping; nothing
/// under Reduce motion. Core Animation, which AppKit already loads.
pub fn fadeNext(view: id, seconds: f64) void {
    if (view == null or reduceMotion()) return;
    msg(void, view, "setWantsLayer:", .{true});
    const layer = msg(id, view, "layer", .{});
    if (layer == null) return;
    const t = msg(id, class("CATransition"), "animation", .{});
    msg(void, t, "setDuration:", .{seconds});
    msg(void, t, "setTimingFunction:", .{msg(id, class("CAMediaTimingFunction"), "functionWithName:", .{nsString("easeInEaseOut")})});
    msg(void, layer, "addAnimation:forKey:", .{ t, nsString("fade") });
}

pub fn color(name: [*:0]const u8) id {
    return msg(id, class("NSColor"), name, .{});
}

pub fn withAlpha(c: id, a: f64) id {
    return msg(id, c, "colorWithAlphaComponent:", .{a});
}

/// A system symbol drawn in the single colour `c`, or null when the system has no such symbol.
pub fn symbolImage(name: id, c: id) id {
    const img = msg(id, class("NSImage"), "imageWithSystemSymbolName:accessibilityDescription:", .{ name, @as(id, null) });
    if (img == null) return null;
    const colors = msg(id, class("NSArray"), "arrayWithObject:", .{c});
    const cfg = msg(id, class("NSImageSymbolConfiguration"), "configurationWithPaletteColors:", .{colors});
    return msg(id, img, "imageWithSymbolConfiguration:", .{cfg});
}

/// Follows the pointer from a mouse-down in `view` to the button release, as a control does, calling `moved(view,
/// inside)` as it goes. True when the release was inside the view.
pub fn trackClick(view: id, moved: *const fn (id, bool) void) bool {
    const app = msg(id, class("NSApplication"), "sharedApplication", .{});
    const forever = msg(id, class("NSDate"), "distantFuture", .{});
    const bounds = msg(NSRect, view, "bounds", .{});
    const mask_up_or_drag: u64 = (1 << 2) | (1 << 6);
    const type_up: isize = 2;
    var inside = true;
    while (true) {
        moved(view, inside);
        const ev = msg(id, app, "nextEventMatchingMask:untilDate:inMode:dequeue:", .{ mask_up_or_drag, forever, NSEventTrackingRunLoopMode, true });
        if (ev == null) return false;
        const p = msg(NSPoint, view, "convertPoint:fromView:", .{ msg(NSPoint, ev, "locationInWindow", .{}), @as(id, null) });
        inside = p.x >= 0 and p.y >= 0 and p.x < bounds.w and p.y < bounds.h;
        if (msg(isize, ev, "type", .{}) == type_up) break;
    }
    moved(view, false);
    return inside;
}
