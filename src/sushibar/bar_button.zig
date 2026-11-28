//! Menu-bar button visuals for `sushibar`: GPU-composited `CALayer` icon + `CAKeyframeAnimation` states,
//! stacked fixed-width readout labels, orange status badge, blue update dot, and tooltip deduplication.
//! Main thread only.
const std = @import("std");
const logic = @import("sushibar_logic.zig");
const objc = @import("objc.zig");
const header_view = @import("header_view.zig");
const main_bar = @import("main_bar.zig");

const Str = logic.FixedStr;

pub const icon_w: f64 = 22;
pub const icon_h: f64 = 18;
const badge_size: f64 = 7;
const update_dot_size: f64 = 6;
/// How far the value's line rises into the caption's, so the pair fits the bar cleanly.
const readout_overlap: f64 = 2;
/// The room between the icon and the readout, beyond the title's own.
const readout_gap: f64 = 2;

/// The title that reserves the readout's room while it shows: as wide as `readout_max - 1` digits of the value font,
/// so a number that changes never moves the icon.
const readout_spacer = blk: {
    var spaces: []const u8 = "";
    for (0..logic.readout_max - 1) |_| spaces = spaces ++ "\u{2007}";
    break :blk spaces;
};

/// Template outline icon (drawn by NSButton for system menu-bar tinting) + transparent spacer for colored icons.
var outline_icon: objc.id = null;
var blank_icon: objc.id = null;
/// Pre-rasterized 2x NSBitmapImageReps for colored icons; loading/reading/dim frames are rasterized lazily on first use.
var static_reps: [std.enums.values(logic.Icon).len]objc.id = @splat(null);
var loading_reps: [logic.loading_frames]objc.id = @splat(null);
var loading_reps_ready = false;
var reading_reps: [logic.reading_steps + 1]objc.id = @splat(null);
var generating_dim_rep: objc.id = null;

var icon_view: objc.id = null;
var icon_layer: objc.id = null;
var placed_icon_rect: ?objc.NSRect = null;
var button_uses_blank = false;
var shown_icon: ?logic.Icon = null;
/// The reading frame shown, while a prompt read shows; `showIcon` clears it.
var shown_reading: ?usize = null;
const LayerAnim = enum { none, loading, blink };
var active_anim: LayerAnim = .none;
var want_blink = false;

var badge_view: objc.id = null;
var shown_badge: ?bool = null;
var update_dot: objc.id = null;
var shown_update_dot: ?bool = null;

var readout_caption: objc.id = null;
var readout_value: objc.id = null;
var readout_fixed_w: f64 = 44.0;
var shown_readout: [2 * logic.readout_max + 8]u8 = undefined;
var shown_readout_len: usize = std.math.maxInt(usize);
var shown_readout_visible: ?bool = null;
var shown_caption: Str(16) = .{};

var shown_tooltip: Str(200) = .{};
var shown_tooltip_set = false;

pub fn shownIcon() ?logic.Icon {
    return shown_icon;
}

pub fn shownBadge() bool {
    return shown_badge == true;
}

pub fn loadIcons() void {
    outline_icon = objc.loadSvg(@embedFile("icon-offline.svg"), true, icon_w, icon_h);
    blank_icon = objc.makeBlankImage(icon_w, icon_h);
    static_reps[@intFromEnum(logic.Icon.empty)] = objc.rasterizeSvgRep(@embedFile("icon-empty.svg"), icon_w, icon_h);
    static_reps[@intFromEnum(logic.Icon.loading)] = objc.rasterizeSvgRep(@embedFile("icon-loading.svg"), icon_w, icon_h);
    static_reps[@intFromEnum(logic.Icon.ready)] = objc.rasterizeSvgRep(@embedFile("icon-online.svg"), icon_w, icon_h);
}

fn ensureLoadingReps() void {
    if (loading_reps_ready) return;
    var buf: [4096]u8 = undefined;
    for (&loading_reps, 0..) |*rep, i| {
        const svg = logic.loadingSvg(&buf, @embedFile("icon-loading.svg"), i) catch |e| std.debug.panic("loading icon frame {d}: {t}", .{ i, e });
        rep.* = objc.rasterizeSvgRep(svg, icon_w, icon_h);
    }
    loading_reps_ready = true;
}

fn ensureGeneratingDimRep() objc.id {
    if (generating_dim_rep == null) {
        var buf: [4096]u8 = undefined;
        const dim = logic.generatingSvg(&buf, @embedFile("icon-online.svg"), false) catch |e| std.debug.panic("generating icon: {t}", .{e});
        generating_dim_rep = objc.rasterizeSvgRep(dim, icon_w, icon_h);
    }
    return generating_dim_rep;
}

fn ensureReadingRep(step: usize) objc.id {
    if (reading_reps[step] == null) {
        var buf: [4096]u8 = undefined;
        const svg = logic.readingSvg(&buf, @embedFile("icon-online.svg"), step) catch |e| std.debug.panic("reading icon step {d}: {t}", .{ step, e });
        reading_reps[step] = objc.rasterizeSvgRep(svg, icon_w, icon_h);
    }
    return reading_reps[step];
}

pub const PreviewImage = struct { img: objc.id, owned: bool };

/// Returns an `NSImage` for offline PNG gallery renders (`owned` is false only for the shared `outline_icon`).
pub fn previewImage(icon: logic.Icon, reading: ?usize, dim: bool) PreviewImage {
    if (reading) |step| return .{ .img = objc.imageFromRep(ensureReadingRep(step), icon_w, icon_h), .owned = true };
    if (dim) return .{ .img = objc.imageFromRep(ensureGeneratingDimRep(), icon_w, icon_h), .owned = true };
    if (icon == .outline) return .{ .img = outline_icon, .owned = false };
    return .{ .img = objc.imageFromRep(static_reps[@intFromEnum(icon)], icon_w, icon_h), .owned = true };
}

fn makeIconView(button: objc.id) void {
    icon_view = objc.msg(objc.id, objc.msg(objc.id, objc.class("NSView"), "alloc", .{}), "initWithFrame:", .{objc.NSRect{ .x = 0, .y = 0, .w = icon_w, .h = icon_h }});
    objc.msg(void, icon_view, "setWantsLayer:", .{true});
    icon_layer = objc.msg(objc.id, icon_view, "layer", .{});
    objc.msg(void, icon_layer, "setContentsScale:", .{@as(f64, 2.0)});
    objc.msg(void, icon_layer, "setContentsGravity:", .{objc.nsString("resizeAspect")});
    objc.disableContentsAction(icon_layer);
    objc.msg(void, icon_view, "setHidden:", .{true});
    objc.msg(void, button, "addSubview:", .{icon_view});
    objc.release(icon_view);
    placed_icon_rect = null;
    button_uses_blank = false;
    active_anim = .none;
}

fn placeIconView(button: objc.id) void {
    if (icon_view == null or button == null) return;
    const bounds = objc.msg(objc.NSRect, button, "bounds", .{});
    if (bounds.w <= 0 or bounds.h <= 0) return;
    const r = objc.msg(objc.NSRect, objc.msg(objc.id, button, "cell", .{}), "imageRectForBounds:", .{bounds});
    const want = objc.NSRect{
        .x = r.x + (r.w - icon_w) / 2.0,
        .y = r.y + (r.h - icon_h) / 2.0,
        .w = icon_w,
        .h = icon_h,
    };
    if (placed_icon_rect) |cur| {
        if (cur.x == want.x and cur.y == want.y and cur.w == want.w and cur.h == want.h) return;
    }
    placed_icon_rect = want;
    objc.msg(void, icon_view, "setFrame:", .{want});
}

fn placeDot(r: objc.NSRect, flipped: bool, dot: objc.id, size: f64, top: bool) void {
    const y = if (top)
        (if (flipped) r.y - 1 else r.y + r.h - size + 1)
    else
        (if (flipped) r.y + r.h - size + 2 else r.y - 2);
    objc.msg(void, dot, "setFrameOrigin:", .{objc.NSPoint{ .x = r.x + r.w - size + 1, .y = y }});
}

fn placeDots(button: objc.id) void {
    if (button == null) return;
    if (shown_badge != true and shown_update_dot != true) return;
    const r = objc.msg(objc.NSRect, objc.msg(objc.id, button, "cell", .{}), "imageRectForBounds:", .{objc.msg(objc.NSRect, button, "bounds", .{})});
    const flipped = objc.msg(bool, button, "isFlipped", .{});
    if (shown_badge == true and badge_view != null) placeDot(r, flipped, badge_view, badge_size, false);
    if (shown_update_dot == true and update_dot != null) placeDot(r, flipped, update_dot, update_dot_size, true);
}

pub fn syncAnimation() void {
    const want: LayerAnim = if (objc.reduceMotion()) .none else if (shown_reading != null) .none else if (shown_icon == .loading) .loading else if (shown_icon == .ready and want_blink) .blink else .none;
    if (want == active_anim) return;
    objc.stopLayerAnimation(icon_layer, "iconAnim");
    active_anim = want;
    switch (want) {
        .none => {},
        .loading => {
            ensureLoadingReps();
            var cgs: [logic.loading_frames]objc.id = undefined;
            for (loading_reps, 0..) |rep, i| cgs[i] = objc.repCgImage(rep);
            objc.startContentsAnimation(icon_layer, "iconAnim", &cgs, @as(f64, @floatFromInt(logic.loading_frames)) * logic.loading_frame_s);
        },
        .blink => {
            const cgs = [_]objc.id{
                objc.repCgImage(static_reps[@intFromEnum(logic.Icon.ready)]),
                objc.repCgImage(ensureGeneratingDimRep()),
            };
            objc.startContentsAnimation(icon_layer, "iconAnim", &cgs, 2.0 * logic.generating_blink_s);
        },
    }
}

fn ensureBlankButtonImage(button: objc.id) void {
    if (!button_uses_blank) {
        objc.msg(void, button, "setImage:", .{blank_icon});
        button_uses_blank = true;
        placeIconView(button);
    }
}

pub fn showIcon(button: objc.id, icon: logic.Icon) void {
    if (shown_icon == icon and shown_reading == null) return;
    shown_reading = null;
    if (icon != .ready) want_blink = false;
    const was_outline = shown_icon == .outline;
    if (shown_icon != null) objc.fadeNext(if (was_outline or icon == .outline) button else icon_view, 0.3);
    shown_icon = icon;
    if (icon == .outline) {
        objc.stopLayerAnimation(icon_layer, "iconAnim");
        active_anim = .none;
        objc.msg(void, icon_view, "setHidden:", .{true});
        objc.msg(void, button, "setImage:", .{outline_icon});
        button_uses_blank = false;
        return;
    }
    ensureBlankButtonImage(button);
    objc.msg(void, icon_layer, "setContents:", .{objc.repCgImage(static_reps[@intFromEnum(icon)])});
    objc.msg(void, icon_view, "setHidden:", .{false});
    placeIconView(button);
    syncAnimation();
}

/// The reading icon at `step`: the ready sushi's fish filled green that far. A model reading a prompt is ready, so
/// it replaces the ready icon, and `showIcon` puts that back when the read ends.
pub fn showReading(button: objc.id, step: usize) void {
    if (shown_reading == step) return;
    const was_outline = shown_icon == .outline;
    if (shown_reading == null and shown_icon != null) {
        objc.fadeNext(if (was_outline) button else icon_view, 0.3);
    }
    want_blink = false;
    shown_reading = step;
    shown_icon = .ready;
    ensureBlankButtonImage(button);
    objc.stopLayerAnimation(icon_layer, "iconAnim");
    active_anim = .none;
    objc.msg(void, icon_layer, "setContents:", .{objc.repCgImage(ensureReadingRep(step))});
    objc.msg(void, icon_view, "setHidden:", .{false});
    placeIconView(button);
}

/// Blinks the ready icon's slot while a reply generates; under Reduce motion the slot stays lit.
pub fn setGeneratingBlink(on: bool) void {
    if (want_blink == on) return;
    want_blink = on;
    syncAnimation();
}

fn makeDot(button: objc.id, size: f64, color_name: [*:0]const u8) objc.id {
    const dot = objc.msg(objc.id, objc.msg(objc.id, objc.class("NSView"), "alloc", .{}), "initWithFrame:", .{objc.NSRect{ .x = 0, .y = 0, .w = size, .h = size }});
    objc.msg(void, dot, "setWantsLayer:", .{true});
    const layer = objc.msg(objc.id, dot, "layer", .{});
    objc.msg(void, layer, "setBackgroundColor:", .{objc.msg(objc.id, objc.color(color_name), "CGColor", .{})});
    objc.msg(void, layer, "setCornerRadius:", .{size / 2});
    objc.msg(void, dot, "setHidden:", .{true});
    objc.msg(void, button, "addSubview:", .{dot});
    objc.release(dot);
    return dot;
}

/// The orange dot on the menu bar icon, at its lower right: it sits on the colour icon where the sushi does not hide it.
pub fn showBadge(button: objc.id, on: bool) void {
    if (shown_badge == on) return;
    if (shown_badge != null) objc.fadeNext(button, 0.25);
    shown_badge = on;
    if (on) placeDots(button);
    objc.msg(void, badge_view, "setHidden:", .{!on});
}

/// The blue dot on the menu bar icon's upper right while an update waits.
pub fn showUpdateDot(button: objc.id, on: bool) void {
    if (shown_update_dot == on) return;
    if (shown_update_dot != null) objc.fadeNext(button, 0.25);
    shown_update_dot = on;
    if (on) placeDots(button);
    objc.msg(void, update_dot, "setHidden:", .{!on});
}

fn readoutLabel(f: objc.id) objc.id {
    const l = objc.msg(objc.id, objc.class("NSTextField"), "labelWithString:", .{objc.nsString("")});
    objc.msg(void, l, "setFont:", .{f});
    objc.msg(void, l, "setTextColor:", .{objc.color("labelColor")});
    return l;
}

const ReadoutLabels = struct { caption: objc.id, value: objc.id, fixed_w: f64 };

fn createReadoutLabels(button: objc.id) ReadoutLabels {
    const cap = readoutLabel(header_view.readoutCaptionFont());
    const val = readoutLabel(header_view.readoutValueFont());
    objc.msg(void, cap, "setStringValue:", .{objc.nsString("TOK/S")});
    objc.msg(void, val, "setStringValue:", .{objc.nsString("00000")});
    objc.msg(void, cap, "sizeToFit", .{});
    objc.msg(void, val, "sizeToFit", .{});
    const cf = objc.msg(objc.NSRect, cap, "frame", .{});
    const vf = objc.msg(objc.NSRect, val, "frame", .{});
    const fixed_w = @max(44.0, @max(cf.w, vf.w) + 4.0);
    objc.msg(void, cap, "setFrameSize:", .{objc.NSSize{ .width = fixed_w, .height = cf.h }});
    objc.msg(void, val, "setFrameSize:", .{objc.NSSize{ .width = fixed_w, .height = vf.h }});
    objc.msg(void, cap, "setStringValue:", .{objc.nsString("")});
    objc.msg(void, val, "setStringValue:", .{objc.nsString("")});
    objc.msg(void, cap, "setHidden:", .{true});
    objc.msg(void, val, "setHidden:", .{true});
    objc.msg(void, button, "addSubview:", .{cap});
    objc.msg(void, button, "addSubview:", .{val});
    return .{ .caption = cap, .value = val, .fixed_w = fixed_w };
}

/// The stacked readout right of the icon: two fixed-width labels sized once from prototype strings over a clear
/// title spacer, shown only while there is a readout.
fn makeReadout(button: objc.id) void {
    const labels = createReadoutLabels(button);
    readout_caption = labels.caption;
    readout_value = labels.value;
    readout_fixed_w = labels.fixed_w;
    shown_readout_len = std.math.maxInt(usize);
    shown_readout_visible = null;
    shown_caption = .{};
}

/// The two lines stacked in the title's room, the pair centred on the bar's height a half point low.
fn placeReadoutViews(button: objc.id, cap: objc.id, val: objc.id) void {
    const bounds = objc.msg(objc.NSRect, button, "bounds", .{});
    const t = objc.msg(objc.NSRect, objc.msg(objc.id, button, "cell", .{}), "titleRectForBounds:", .{bounds});
    const cf = objc.msg(objc.NSRect, cap, "frame", .{});
    const vf = objc.msg(objc.NSRect, val, "frame", .{});
    const bottom = (bounds.h - (vf.h + cf.h - readout_overlap)) / 2 - 0.5;
    const flipped = objc.msg(bool, button, "isFlipped", .{});
    const at = struct {
        fn y(fl: bool, h: f64, from_bottom: f64, height: f64) f64 {
            return if (fl) h - from_bottom - height else from_bottom;
        }
    }.y;
    const x = t.x + readout_gap;
    objc.msg(void, val, "setFrameOrigin:", .{objc.NSPoint{ .x = x, .y = at(flipped, bounds.h, bottom, vf.h) }});
    objc.msg(void, cap, "setFrameOrigin:", .{objc.NSPoint{ .x = x, .y = at(flipped, bounds.h, bottom + vf.h - readout_overlap, cf.h) }});
}

fn placeReadout(button: objc.id) void {
    placeReadoutViews(button, readout_caption, readout_value);
}

pub fn setReadout(button: objc.id, r: logic.Readout) void {
    const n = r.caption.len + 1 + r.value.len;
    var key: [shown_readout.len]u8 = undefined;
    const k = std.fmt.bufPrint(&key, "{s}\n{s}", .{ r.caption, r.value }) catch return;
    if (shown_readout_len == n and std.mem.eql(u8, shown_readout[0..n], k)) return;
    shown_readout_len = n;
    @memcpy(shown_readout[0..n], k);
    const shown = r.caption.len > 0;
    const vis_changed = shown_readout_visible == null or shown_readout_visible.? != shown;
    if (vis_changed) {
        shown_readout_visible = shown;
        // Only mutate NSStatusBarButton's attributedTitle when visibility toggles, not on every digit change:
        // setAttributedTitle triggers _adjustLength and an FBSScene XPC round-trip to ControlCenter on macOS 26.
        const title = objc.msg(objc.id, objc.msg(objc.id, objc.class("NSAttributedString"), "alloc", .{}), "initWithString:attributes:", .{ objc.nsText(if (shown) readout_spacer else ""), header_view.readoutSpacerAttributes() });
        defer objc.release(title);
        objc.msg(void, button, "setAttributedTitle:", .{title});
        objc.msg(void, readout_caption, "setHidden:", .{!shown});
        objc.msg(void, readout_value, "setHidden:", .{!shown});
    }
    if (shown) {
        if (!std.mem.eql(u8, shown_caption.get(), r.caption)) {
            shown_caption.set(r.caption);
            objc.msg(void, readout_caption, "setStringValue:", .{objc.nsText(r.caption)});
        }
        objc.msg(void, readout_value, "setStringValue:", .{objc.nsText(r.value)});
        if (vis_changed) {
            if (main_bar.mode() == .main_window) {
                main_bar.setContentWidth(icon_w + readout_gap + readout_fixed_w + 10.0);
            }
            placeReadout(button);
        }
    } else if (vis_changed and main_bar.mode() == .main_window) {
        main_bar.setContentWidth(30.0);
    }
    if (vis_changed) {
        placeIconView(button);
        placeDots(button);
    }
}

pub fn setToolTip(button: objc.id, text: []const u8) void {
    if (shown_tooltip_set and std.mem.eql(u8, shown_tooltip.get(), text)) return;
    shown_tooltip.set(text);
    shown_tooltip_set = true;
    objc.msg(void, button, "setToolTip:", .{objc.nsText(text)});
}

/// Attaches the icon layer, badge, update dot, and readout subviews onto `button` (used at startup and when
/// switching between `NSStatusItem` and the main-monitor-only panel on display hotplug).
pub fn attach(button: objc.id) void {
    shown_icon = null;
    shown_reading = null;
    shown_badge = null;
    shown_update_dot = null;
    shown_tooltip_set = false;
    want_blink = false;
    active_anim = .none;
    button_uses_blank = false;
    makeIconView(button);
    // Display the outline until the first refresh without setting `shown_icon`, so a hotplug rebuild + refresh
    // snaps straight to the live icon instead of cross-fading from offline.
    objc.msg(void, button, "setImage:", .{outline_icon});
    badge_view = makeDot(button, badge_size, "systemOrangeColor");
    update_dot = makeDot(button, update_dot_size, "systemBlueColor");
    makeReadout(button);
}

/// Realigns subviews on `button` after a menu-bar height change or Space switch in `.main_window` mode.
pub fn realign(button: objc.id) void {
    if (button == null) return;
    placeIconView(button);
    placeDots(button);
    if (shown_readout_visible == true) placeReadout(button);
}

/// Attaches self-contained readout, update dot, and badge subviews onto a standalone preview button for
/// `SUSHIBAR_RENDER_HEADER`, without mutating live module globals or attaching `CATransition` animations.
pub fn attachPreview(button: objc.id, readout: logic.Readout, badge: bool, update: bool) void {
    if (readout.caption.len > 0) {
        const title = objc.msg(objc.id, objc.msg(objc.id, objc.class("NSAttributedString"), "alloc", .{}), "initWithString:attributes:", .{ objc.nsText(readout_spacer), header_view.readoutSpacerAttributes() });
        defer objc.release(title);
        objc.msg(void, button, "setAttributedTitle:", .{title});
        const labels = createReadoutLabels(button);
        objc.msg(void, labels.caption, "setStringValue:", .{objc.nsText(readout.caption)});
        objc.msg(void, labels.value, "setStringValue:", .{objc.nsText(readout.value)});
        objc.msg(void, labels.caption, "setHidden:", .{false});
        objc.msg(void, labels.value, "setHidden:", .{false});
        placeReadoutViews(button, labels.caption, labels.value);
    }
    if (!update and !badge) return;
    const r = objc.msg(objc.NSRect, objc.msg(objc.id, button, "cell", .{}), "imageRectForBounds:", .{objc.msg(objc.NSRect, button, "bounds", .{})});
    const flipped = objc.msg(bool, button, "isFlipped", .{});
    if (update) {
        const dot = makeDot(button, update_dot_size, "systemBlueColor");
        placeDot(r, flipped, dot, update_dot_size, true);
        objc.msg(void, dot, "setHidden:", .{false});
    }
    if (badge) {
        const dot = makeDot(button, badge_size, "systemOrangeColor");
        placeDot(r, flipped, dot, badge_size, false);
        objc.msg(void, dot, "setHidden:", .{false});
    }
}
