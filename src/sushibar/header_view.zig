//! The menu's header card: one custom NSView drawn in Zig. What it shows is decided in `sushibar_logic.zig`
//! (`buildHeader`, `sparkPoints`); this file only lays the strings out and draws them with system fonts and colours,
//! so light and dark appearance both work. Single instance: its state is module-level.
const std = @import("std");
const logic = @import("sushibar_logic.zig");
const objc = @import("objc.zig");

const msg = objc.msg;
const id = objc.id;
const NSRect = objc.NSRect;
const NSPoint = objc.NSPoint;

extern var NSFontAttributeName: id;
extern var NSForegroundColorAttributeName: id;
extern var NSParagraphStyleAttributeName: id;

pub const width: f64 = 300;
pub const height_full: f64 = 168;
pub const height_short: f64 = 84;
const pad: f64 = 12;
const spark_y: f64 = 130;
const spark_h: f64 = 30;
pub const spark_window_ms: i64 = 5 * 60 * 1000;

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

const Shown = struct {
    version: Buf(96) = .{},
    version_tail: Buf(48) = .{},
    pill_label: Buf(24) = .{},
    pill_tone: logic.Tone = .grey,
    line1: Buf(160) = .{},
    line2: Buf(96) = .{},
    show_stats: bool = false,
    values: [3]Buf(16) = .{ .{}, .{}, .{} },
    labels: [3]Buf(24) = .{ .{}, .{}, .{} },
    spark: [160]logic.SparkPoint = undefined,
    spark_n: usize = 0,
};

var shown: Shown = .{};
var opaque_bg = false;
var view: id = null;
var image_view: id = null;
var icon_color: id = null;
var icon_outline: id = null;
var shown_color: ?bool = null;

fn str(s: []const u8) id {
    var z: [320:0]u8 = undefined;
    const n = @min(s.len, z.len - 1);
    @memcpy(z[0..n], s[0..n]);
    z[n] = 0;
    return objc.nsString(&z);
}

fn color(name: [*:0]const u8) id {
    return msg(id, objc.class("NSColor"), name, .{});
}

fn font(size: f64, weight: f64, mono_digits: bool) id {
    return msg(id, objc.class("NSFont"), if (mono_digits) "monospacedDigitSystemFontOfSize:weight:" else "systemFontOfSize:weight:", .{ size, weight });
}

const weight_regular: f64 = 0.0;
const weight_medium: f64 = 0.23;
const weight_semibold: f64 = 0.3;

/// `{NSFontAttributeName: font, NSForegroundColorAttributeName: color, NSParagraphStyleAttributeName: style}`.
fn attrs(f: id, c: id, line_break: isize, alignment: isize) id {
    const style = objc.new(objc.class("NSMutableParagraphStyle"));
    msg(void, style, "setLineBreakMode:", .{line_break});
    msg(void, style, "setAlignment:", .{alignment});
    var keys = [_]id{ NSFontAttributeName, NSForegroundColorAttributeName, NSParagraphStyleAttributeName };
    var vals = [_]id{ f, c, style };
    return msg(id, objc.class("NSDictionary"), "dictionaryWithObjects:forKeys:count:", .{ @as([*]id, &vals), @as([*]id, &keys), @as(usize, 3) });
}

const truncate_tail: isize = 4;
const truncate_middle: isize = 5;
const clip: isize = 2;
const align_left: isize = 0;
const align_right: isize = 1;

fn text(s: []const u8, r: NSRect, f: id, c: id, line_break: isize, alignment: isize) void {
    msg(void, str(s), "drawInRect:withAttributes:", .{ r, attrs(f, c, line_break, alignment) });
}

fn textWidth(s: []const u8, f: id) f64 {
    const size = msg(objc.NSSize, str(s), "sizeWithAttributes:", .{attrs(f, color("labelColor"), clip, align_left)});
    return @ceil(size.width);
}

fn toneColor(t: logic.Tone) id {
    return color(switch (t) {
        .green => "systemGreenColor",
        .orange => "systemOrangeColor",
        .grey => "systemGrayColor",
    });
}

fn fillRound(r: NSRect, radius: f64, c: id) void {
    const path = msg(id, objc.class("NSBezierPath"), "bezierPathWithRoundedRect:xRadius:yRadius:", .{ r, radius, radius });
    msg(void, c, "setFill", .{});
    msg(void, path, "fill", .{});
}

fn drawSpark(accent: id) void {
    var raw: [160]logic.PlotPoint = undefined;
    const pts = logic.plotSpark(&raw, shown.spark[0..shown.spark_n], pad, width - 2 * pad, spark_y + 4, spark_y + spark_h - 2);
    if (pts.len < 2) return;
    const bottom = spark_y + spark_h - 2;
    msg(void, objc.class("NSBezierPath"), "clipRect:", .{NSRect{ .x = pad, .y = spark_y, .w = width - 2 * pad, .h = spark_h }});
    const fill_color = msg(id, accent, "colorWithAlphaComponent:", .{@as(f64, 0.12)});
    var s: usize = 0;
    while (s < pts.len) {
        var e = s + 1;
        while (e < pts.len and !pts[e].gap) e += 1;
        if (e - s >= 2) {
            const line = msg(id, objc.class("NSBezierPath"), "bezierPath", .{});
            const area = msg(id, objc.class("NSBezierPath"), "bezierPath", .{});
            for (pts[s..e], 0..) |p, i| {
                const pt = NSPoint{ .x = p.x, .y = p.y };
                if (i == 0) {
                    msg(void, line, "moveToPoint:", .{pt});
                    msg(void, area, "moveToPoint:", .{NSPoint{ .x = p.x, .y = bottom }});
                }
                if (i > 0) msg(void, line, "lineToPoint:", .{pt});
                msg(void, area, "lineToPoint:", .{pt});
            }
            msg(void, area, "lineToPoint:", .{NSPoint{ .x = pts[e - 1].x, .y = bottom }});
            msg(void, area, "closePath", .{});
            msg(void, fill_color, "setFill", .{});
            msg(void, area, "fill", .{});
            msg(void, line, "setLineWidth:", .{@as(f64, 1.5)});
            msg(void, line, "setLineJoinStyle:", .{@as(isize, 1)});
            msg(void, line, "setLineCapStyle:", .{@as(isize, 1)});
            msg(void, accent, "setStroke", .{});
            msg(void, line, "stroke", .{});
        }
        s = e;
    }
}

fn drawRect(_: id, _: objc.SEL, _: NSRect) callconv(.c) void {
    if (opaque_bg) {
        msg(void, color("windowBackgroundColor"), "setFill", .{});
        msg(void, msg(id, objc.class("NSBezierPath"), "bezierPathWithRect:", .{NSRect{ .x = 0, .y = 0, .w = width, .h = height_full }}), "fill", .{});
    }
    const label = color("labelColor");
    const secondary = color("secondaryLabelColor");

    // Row 1: icon (a subview), name and version, the status pill.
    const name_font = font(13, weight_semibold, false);
    text("sushi", .{ .x = 50, .y = 15, .w = 60, .h = 17 }, name_font, label, clip, align_left);
    const pill_font = font(11, weight_medium, false);
    const pill_label = shown.pill_label.get();
    const pill_w = 7 + 7 + 5 + textWidth(pill_label, pill_font) + 8;
    const pill_x = width - pad - pill_w;
    const name_w = textWidth("sushi", name_font);
    const version_x = 50 + name_w + 5;
    const small = font(11, weight_regular, false);
    const head_w = textWidth(shown.version.get(), small);
    text(shown.version.get(), .{ .x = version_x, .y = 17, .w = @max(0, pill_x - 8 - version_x), .h = 14 }, small, secondary, truncate_tail, align_left);
    text(shown.version_tail.get(), .{ .x = version_x + head_w, .y = 17, .w = @max(0, pill_x - 8 - version_x - head_w), .h = 14 }, small, color("controlAccentColor"), truncate_tail, align_left);
    const tone = toneColor(shown.pill_tone);
    fillRound(.{ .x = pill_x, .y = 14, .w = pill_w, .h = 20 }, 10, msg(id, tone, "colorWithAlphaComponent:", .{@as(f64, 0.16)}));
    fillRound(.{ .x = pill_x + 7, .y = 20.5, .w = 7, .h = 7 }, 3.5, tone);
    text(pill_label, .{ .x = pill_x + 7 + 7 + 5, .y = 17, .w = pill_w - 19, .h = 14 }, pill_font, label, clip, align_left);

    // Row 2: the model and where it is.
    text(shown.line1.get(), .{ .x = pad, .y = 46, .w = width - 2 * pad, .h = 16 }, font(12, weight_medium, false), label, truncate_middle, align_left);
    text(shown.line2.get(), .{ .x = pad, .y = 63, .w = width - 2 * pad, .h = 14 }, font(11, weight_regular, false), secondary, truncate_tail, align_left);
    if (!shown.show_stats) return;

    // Row 3: three equal columns.
    const col_w = (width - 2 * pad) / 3;
    const value_font = font(17, weight_semibold, true);
    const label_font = font(10, weight_medium, false);
    for (0..3) |i| {
        const x = pad + @as(f64, @floatFromInt(i)) * col_w;
        text(shown.values[i].get(), .{ .x = x, .y = 88, .w = col_w - 4, .h = 22 }, value_font, label, clip, align_left);
        text(shown.labels[i].get(), .{ .x = x, .y = 111, .w = col_w - 4, .h = 12 }, label_font, secondary, truncate_tail, align_left);
    }

    // Row 4: decode throughput over the last five minutes.
    drawSpark(color("controlAccentColor"));
}

fn isFlipped(_: id, _: objc.SEL) callconv(.c) bool {
    return true;
}

pub fn create(color_icon: id, outline_icon: id) id {
    icon_color = color_icon;
    icon_outline = outline_icon;
    const cls = objc.defineSubclass("NSView", "SushiBarHeader", &.{
        .{ .selector = "drawRect:", .imp = @ptrCast(&drawRect), .types = "v@:{CGRect={CGPoint=dd}{CGSize=dd}}" },
        .{ .selector = "isFlipped", .imp = @ptrCast(&isFlipped), .types = "c@:" },
    });
    view = msg(id, msg(id, cls, "alloc", .{}), "initWithFrame:", .{NSRect{ .x = 0, .y = 0, .w = width, .h = height_full }});
    image_view = msg(id, msg(id, objc.class("NSImageView"), "alloc", .{}), "initWithFrame:", .{NSRect{ .x = pad, .y = 10, .w = 34, .h = 28 }});
    msg(void, image_view, "setImageScaling:", .{@as(isize, 3)});
    msg(void, view, "addSubview:", .{image_view});
    msg(void, view, "setAccessibilityElement:", .{true});
    msg(void, view, "setAccessibilityRole:", .{str("AXStaticText")});
    return view;
}

/// Copies `h` and the sparkline in and redraws; the view keeps no pointer into the caller's memory.
pub fn update(h: logic.Header, spark: []const logic.SparkPoint) void {
    shown.version.set(h.version[0 .. h.version.len - h.version_tail.len]);
    shown.version_tail.set(h.version_tail);
    shown.pill_label.set(h.pill.label);
    shown.pill_tone = h.pill.tone;
    shown.line1.set(h.line1);
    shown.line2.set(h.line2);
    shown.show_stats = h.show_stats;
    for (h.stats, 0..) |st, i| {
        shown.values[i].set(st.value);
        shown.labels[i].set(st.label);
    }
    shown.spark_n = @min(spark.len, shown.spark.len);
    @memcpy(shown.spark[0..shown.spark_n], spark[0..shown.spark_n]);

    if (shown_color != h.color_icon) {
        shown_color = h.color_icon;
        msg(void, image_view, "setImage:", .{if (h.color_icon) icon_color else icon_outline});
        msg(void, image_view, "setContentTintColor:", .{if (h.color_icon) @as(id, null) else color("labelColor")});
    }
    const want_h = if (h.show_stats) height_full else height_short;
    msg(void, view, "setFrameSize:", .{objc.NSSize{ .width = width, .height = want_h }});
    msg(void, view, "setAccessibilityLabel:", .{str(h.a11y)});
    msg(void, view, "setNeedsDisplay:", .{true});
}

/// Renders the header at 2x for `dark` or light appearance into a PNG at `path`; false when it could not.
pub fn renderPng(path: [*:0]const u8, dark: bool) bool {
    opaque_bg = true;
    defer opaque_bg = false;
    const h: f64 = if (shown.show_stats) height_full else height_short;
    const name = if (dark) "NSAppearanceNameDarkAqua" else "NSAppearanceNameAqua";
    msg(void, view, "setAppearance:", .{msg(id, objc.class("NSAppearance"), "appearanceNamed:", .{objc.nsString(name)})});
    const bounds = NSRect{ .x = 0, .y = 0, .w = width, .h = h };
    const rep = msg(id, view, "bitmapImageRepForCachingDisplayInRect:", .{bounds});
    if (rep == null) return false;
    msg(void, view, "cacheDisplayInRect:toBitmapImageRep:", .{ bounds, rep });
    const data = msg(id, rep, "representationUsingType:properties:", .{ @as(isize, 4), @as(id, null) });
    if (data == null) return false;
    return msg(bool, data, "writeToFile:atomically:", .{ objc.nsString(path), true });
}

/// The attributes of the menu bar button's title: 12 pt monospaced digits, dim for a past reading.
pub fn titleAttributes(dim: bool) id {
    return attrs(font(12, weight_regular, true), color(if (dim) "secondaryLabelColor" else "labelColor"), clip, align_left);
}
