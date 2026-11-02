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
        else => @compileError("objc.msg: more arguments than a sushibar call uses"),
    };
    const f: F = @ptrCast(&objc_msgSend);
    return @call(.auto, f, .{ target, sel_registerName(selector) } ++ args);
}

/// An autoreleased NSString copy of `s`.
pub fn nsString(s: [*:0]const u8) id {
    return msg(id, class("NSString"), "stringWithUTF8String:", .{s});
}

pub const Method = struct { selector: [*:0]const u8, imp: *const anyopaque, types: [*:0]const u8 };

/// A subclass of NSObject named `name` whose instance methods are `methods`.
pub fn defineClass(name: [*:0]const u8, methods: []const Method) id {
    const cls = objc_allocateClassPair(class("NSObject"), name, 0);
    for (methods) |m| _ = class_addMethod(cls, sel_registerName(m.selector), m.imp, m.types);
    objc_registerClassPair(cls);
    return cls;
}

pub fn new(cls: id) id {
    return msg(id, msg(id, cls, "alloc", .{}), "init", .{});
}

/// Calls `[target selector:timer]` every `seconds` on the main run loop, menu tracking included.
pub fn addTimer(seconds: f64, target: id, selector: [*:0]const u8) void {
    const timer = msg(id, class("NSTimer"), "timerWithTimeInterval:target:selector:userInfo:repeats:", .{ seconds, target, sel_registerName(selector), @as(id, null), true });
    msg(void, msg(id, class("NSRunLoop"), "mainRunLoop", .{}), "addTimer:forMode:", .{ timer, NSRunLoopCommonModes });
}
