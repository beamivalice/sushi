//! Hosts `sushibar`'s menu-bar item: a standard `NSStatusItem` when a single monitor (or unified Spaces) has the
//! menu bar, and a borderless `NSPanel` at `NSStatusWindowLevel` pinned strictly to `NSScreen.screens[0]` (the
//! primary/main monitor) when multiple monitors have separate menu bars.
//!
//! macOS has no public `NSStatusItem` API to hide an item from secondary monitors when "Displays have separate
//! Spaces" is enabled (`NSSceneStatusItem` on macOS 26 lets ControlCenter replicate every status item to every
//! monitor). Removing the `NSStatusItem` and placing a status-level panel on `NSScreen.screens[0]` keeps
//! `sushibar` exclusively on the main monitor using 100% public AppKit and CoreGraphics APIs.
const objc = @import("objc.zig");
const logic = @import("sushibar_logic.zig");

const id = objc.id;
const msg = objc.msg;
const NSRect = objc.NSRect;
const NSPoint = objc.NSPoint;

const CFArrayRef = ?*const anyopaque;
const CFDictionaryRef = ?*const anyopaque;
const CFNumberRef = ?*const anyopaque;
const CFStringRef = ?*const anyopaque;
const CFTypeRef = ?*const anyopaque;

const cg_window_list_on_screen_only: u32 = 1 << 0;
const cg_null_window_id: u32 = 0;
const cf_number_s32: isize = 3;
const cf_number_f64: isize = 13;

extern "CoreGraphics" fn CGWindowListCopyWindowInfo(option: u32, relative_to: u32) CFArrayRef;
extern "CoreFoundation" fn CFArrayGetCount(arr: CFArrayRef) isize;
extern "CoreFoundation" fn CFArrayGetValueAtIndex(arr: CFArrayRef, idx: isize) CFTypeRef;
extern "CoreFoundation" fn CFDictionaryGetValue(dict: CFDictionaryRef, key: CFTypeRef) CFTypeRef;
extern "CoreFoundation" fn CFNumberGetValue(num: CFNumberRef, the_type: isize, value_ptr: *anyopaque) bool;
extern "CoreFoundation" fn CFRelease(cf: CFTypeRef) void;
extern "CoreGraphics" var kCGWindowLayer: CFStringRef;
extern "CoreGraphics" var kCGWindowBounds: CFStringRef;
extern "CoreGraphics" var kCGWindowNumber: CFStringRef;

const style_borderless: usize = 0;
const style_nonactivating_panel: usize = 1 << 7;
const backing_buffered: usize = 2;
const status_window_level: isize = 25;
const menu_bar_layer: i32 = 24;
const status_layer: i32 = 25;
// CanJoinAllSpaces (1) | Stationary (16) | IgnoresCycle (64). Intentionally omits FullScreenAuxiliary so the
// panel hides automatically in full-screen spaces when the menu bar hides.
const panel_collection_behavior: usize = (1 << 0) | (1 << 4) | (1 << 6);

var g_mode: logic.BarMode = .status_item;
var g_status_item: id = null;
var g_panel: id = null;
var g_panel_button: id = null;
var g_highlight: id = null;
var g_target: id = null;
var g_menu: id = null;
var g_force_main_window = false;
var g_item_w: f64 = 30;
var g_panel_hidden = false;

pub fn setForceMainWindow(force: bool) void {
    g_force_main_window = force;
}

pub fn mode() logic.BarMode {
    return g_mode;
}

/// The active NSButton (`statusItem.button` in `.status_item` mode, or the panel's button in `.main_window` mode).
pub fn button() id {
    return switch (g_mode) {
        .status_item => if (g_status_item != null) msg(id, g_status_item, "button", .{}) else null,
        .main_window => g_panel_button,
    };
}

/// Evaluates connected displays and decides whether to use `NSStatusItem` or the main-monitor-only panel.
pub fn desiredMode() logic.BarMode {
    if (g_force_main_window) return .main_window;
    const screens = msg(id, objc.class("NSScreen"), "screens", .{});
    const count: usize = if (screens != null) msg(usize, screens, "count", .{}) else 1;
    const sep = msg(bool, objc.class("NSScreen"), "screensHaveSeparateSpaces", .{});
    return logic.barMode(count, sep);
}

/// Initializes the bar surface. Returns the active button and attaches `menu` (and `target` for panel clicks).
pub fn init(target: id, menu: id) id {
    g_target = target;
    g_menu = menu;
    g_mode = desiredMode();
    buildActive();
    return button();
}

/// Re-evaluates display configuration (e.g., on `NSApplicationDidChangeScreenParametersNotification`).
/// Returns true if the active button instance changed and needs its subviews/state re-attached.
pub fn syncScreens() bool {
    const want = desiredMode();
    if (want == g_mode) {
        if (g_mode == .main_window) reposition();
        return false;
    }
    teardownActive();
    g_mode = want;
    buildActive();
    return true;
}

/// Updates panel width when the readout shows or hides (no-op for `NSStatusItem`, which auto-sizes).
pub fn setContentWidth(w: f64) void {
    g_item_w = @max(w, 28);
    if (g_mode == .main_window) reposition();
}

/// Shows or clears the subtle rounded menu-bar selection pill on the main-window panel while the menu is open.
pub fn setHighlight(on: bool) void {
    if (g_highlight == null) return;
    if (on) {
        const hlayer = msg(id, g_highlight, "layer", .{});
        if (hlayer != null) {
            msg(void, hlayer, "setBackgroundColor:", .{msg(id, objc.withAlpha(objc.color("labelColor"), 0.16), "CGColor", .{})});
        }
    }
    msg(void, g_highlight, "setHidden:", .{!on});
}

/// Opens `g_menu` anchored cleanly below the main-monitor panel button.
pub fn popupMenu() void {
    if (g_mode != .main_window or g_panel_button == null or g_menu == null) return;
    setHighlight(true);
    const bounds = msg(NSRect, g_panel_button, "bounds", .{});
    const flipped = msg(bool, g_panel_button, "isFlipped", .{});
    // Flipped NSButton has y=0 at the top and y=bounds.h at the bottom; unflipped has y=0 at the bottom.
    const anchor_y = if (flipped) bounds.h else 0.0;
    _ = msg(bool, g_menu, "popUpMenuPositioningItem:atLocation:inView:", .{
        @as(id, null),
        NSPoint{ .x = 0, .y = anchor_y },
        g_panel_button,
    });
    setHighlight(false);
}

fn buildActive() void {
    switch (g_mode) {
        .status_item => {
            const bar = msg(id, objc.class("NSStatusBar"), "systemStatusBar", .{});
            g_status_item = msg(id, bar, "statusItemWithLength:", .{@as(f64, -1)});
            const btn = msg(id, g_status_item, "button", .{});
            msg(void, btn, "setImagePosition:", .{@as(isize, 2)});
            if (g_menu != null) msg(void, g_status_item, "setMenu:", .{g_menu});
        },
        .main_window => {
            const panel_rect = NSRect{ .x = 0, .y = 0, .w = g_item_w, .h = 24 };
            g_panel = msg(id, msg(id, objc.class("NSPanel"), "alloc", .{}), "initWithContentRect:styleMask:backing:defer:", .{
                panel_rect,
                style_borderless | style_nonactivating_panel,
                backing_buffered,
                false,
            });
            msg(void, g_panel, "setReleasedWhenClosed:", .{false});
            msg(void, g_panel, "setLevel:", .{status_window_level});
            msg(void, g_panel, "setOpaque:", .{false});
            msg(void, g_panel, "setBackgroundColor:", .{objc.color("clearColor")});
            msg(void, g_panel, "setHasShadow:", .{false});
            msg(void, g_panel, "setCollectionBehavior:", .{panel_collection_behavior});
            msg(void, g_panel, "setIgnoresMouseEvents:", .{false});
            msg(void, g_panel, "setHidesOnDeactivate:", .{false});

            const content = msg(id, g_panel, "contentView", .{});
            g_highlight = msg(id, msg(id, objc.class("NSView"), "alloc", .{}), "initWithFrame:", .{NSRect{ .x = 1, .y = 3, .w = g_item_w - 2, .h = 18 }});
            msg(void, g_highlight, "setWantsLayer:", .{true});
            const hlayer = msg(id, g_highlight, "layer", .{});
            msg(void, hlayer, "setBackgroundColor:", .{msg(id, objc.withAlpha(objc.color("labelColor"), 0.16), "CGColor", .{})});
            msg(void, hlayer, "setCornerRadius:", .{@as(f64, 4)});
            msg(void, g_highlight, "setHidden:", .{true});
            msg(void, g_highlight, "setAutoresizingMask:", .{@as(usize, 2 | 16)});
            msg(void, content, "addSubview:", .{g_highlight});

            g_panel_button = msg(id, msg(id, objc.class("NSButton"), "alloc", .{}), "initWithFrame:", .{NSRect{ .x = 0, .y = 0, .w = g_item_w, .h = 24 }});
            msg(void, g_panel_button, "setBordered:", .{false});
            msg(void, g_panel_button, "setButtonType:", .{@as(isize, 0)});
            const cell = msg(id, g_panel_button, "cell", .{});
            if (cell != null) msg(void, cell, "setHighlightsBy:", .{@as(usize, 0)});
            // Fire immediately on left-mouse-down (mask 1 << 1), matching native NSStatusItem responsiveness.
            _ = msg(isize, g_panel_button, "sendActionOn:", .{@as(usize, 1 << 1)});
            msg(void, g_panel_button, "setImagePosition:", .{@as(isize, 2)});
            msg(void, g_panel_button, "setContentTintColor:", .{objc.color("labelColor")});
            msg(void, g_panel_button, "setAutoresizingMask:", .{@as(usize, 2 | 16)});
            if (g_target != null) {
                msg(void, g_panel_button, "setTarget:", .{g_target});
                msg(void, g_panel_button, "setAction:", .{objc.sel("popupMenu:")});
            }
            msg(void, content, "addSubview:", .{g_panel_button});
            g_panel_hidden = true;
            reposition();
        },
    }
}

fn teardownActive() void {
    switch (g_mode) {
        .status_item => {
            if (g_status_item != null) {
                const bar = msg(id, objc.class("NSStatusBar"), "systemStatusBar", .{});
                msg(void, bar, "removeStatusItem:", .{g_status_item});
                g_status_item = null;
            }
        },
        .main_window => {
            if (g_panel != null) {
                msg(void, g_panel, "orderOut:", .{@as(id, null)});
                objc.release(g_panel);
                g_panel = null;
            }
            if (g_panel_button != null) {
                objc.release(g_panel_button);
                g_panel_button = null;
            }
            if (g_highlight != null) {
                objc.release(g_highlight);
                g_highlight = null;
            }
            g_panel_hidden = true;
        },
    }
}

fn cfNumberI32(dict: CFDictionaryRef, key: CFTypeRef) ?i32 {
    const val = CFDictionaryGetValue(dict, key) orelse return null;
    var out: i32 = 0;
    return if (CFNumberGetValue(val, cf_number_s32, &out)) out else null;
}

fn cfNumberF64(dict: CFDictionaryRef, key: CFTypeRef) ?f64 {
    const val = CFDictionaryGetValue(dict, key) orelse return null;
    var out: f64 = 0;
    return if (CFNumberGetValue(val, cf_number_f64, &out)) out else null;
}

const MenuBarScan = struct {
    occupied_x: [64]f64 = undefined,
    occupied_n: usize = 0,
    bar_h: f64 = 0,
    bar_visible: bool = false,
};

fn scanMainMenuBar(main_w: f64, self_win_num: i32) MenuBarScan {
    var out: MenuBarScan = .{};
    const arr = CGWindowListCopyWindowInfo(cg_window_list_on_screen_only, cg_null_window_id) orelse return out;
    defer CFRelease(arr);
    const n = CFArrayGetCount(arr);
    const k_x: CFTypeRef = objc.nsString("X");
    const k_y: CFTypeRef = objc.nsString("Y");
    const k_w: CFTypeRef = objc.nsString("Width");
    const k_h: CFTypeRef = objc.nsString("Height");

    var i: isize = 0;
    while (i < n) : (i += 1) {
        const dict = CFArrayGetValueAtIndex(arr, i) orelse continue;
        const layer = cfNumberI32(dict, kCGWindowLayer) orelse continue;
        if (layer != status_layer and layer != menu_bar_layer) continue;
        const win_num = cfNumberI32(dict, kCGWindowNumber) orelse 0;
        if (self_win_num != 0 and win_num == self_win_num) continue;
        const bounds = CFDictionaryGetValue(dict, kCGWindowBounds) orelse continue;
        const x = cfNumberF64(bounds, k_x) orelse continue;
        const y = cfNumberF64(bounds, k_y) orelse continue;
        const w = cfNumberF64(bounds, k_w) orelse continue;
        const h = cfNumberF64(bounds, k_h) orelse continue;
        // Main display always has CG origin (0, 0). Ignore secondary displays (x < 0 or x >= main_w) and offscreen bars.
        if (y < -1 or y > 48 or h < 16 or h > 64 or w <= 0) continue;
        if (x < -1 or x >= main_w) continue;
        out.bar_visible = true;
        if (layer == menu_bar_layer and h > out.bar_h) out.bar_h = h;
        if (layer == status_layer) {
            if (h > out.bar_h) out.bar_h = h;
            if (out.occupied_n < out.occupied_x.len) {
                out.occupied_x[out.occupied_n] = x;
                out.occupied_n += 1;
            }
        }
    }
    return out;
}

/// Repositions the main-monitor panel onto `NSScreen.screens[0]`'s menu bar, hiding it if the bar is hidden.
pub fn reposition() void {
    if (g_mode != .main_window or g_panel == null) return;
    const screens = msg(id, objc.class("NSScreen"), "screens", .{});
    if (screens == null or msg(usize, screens, "count", .{}) == 0) return;
    const main = msg(id, screens, "objectAtIndex:", .{@as(usize, 0)});
    const frame = msg(NSRect, main, "frame", .{});
    const vis = msg(NSRect, main, "visibleFrame", .{});
    var aux_right: NSRect = .{ .x = 0, .y = 0, .w = 0, .h = 0 };
    if (msg(bool, main, "respondsToSelector:", .{objc.sel("auxiliaryTopRightArea")})) {
        aux_right = msg(NSRect, main, "auxiliaryTopRightArea", .{});
    }
    const self_win: i32 = @intCast(msg(isize, g_panel, "windowNumber", .{}));
    const scan = scanMainMenuBar(frame.w, self_win);
    const fallback_h = frame.h - (vis.y - frame.y) - vis.h;
    const bar_h = if (scan.bar_h >= 20) scan.bar_h else if (fallback_h >= 20 and fallback_h <= 48) fallback_h else 24;
    const placed = logic.mainBarFrame(
        .{ .x = frame.x, .y = frame.y, .w = frame.w, .h = frame.h },
        .{ .x = aux_right.x, .y = aux_right.y, .w = aux_right.w, .h = aux_right.h },
        bar_h,
        scan.occupied_x[0..scan.occupied_n],
        g_item_w,
        scan.bar_visible,
    );
    if (placed) |r| {
        msg(void, g_panel, "setFrame:display:", .{ NSRect{ .x = r.x, .y = r.y, .w = r.w, .h = r.h }, true });
        if (g_panel_button != null) {
            msg(void, g_panel_button, "setFrame:", .{NSRect{ .x = 0, .y = 0, .w = r.w, .h = r.h }});
        }
        if (g_highlight != null) {
            const hh = @min(22.0, @max(18.0, r.h - 6.0));
            msg(void, g_highlight, "setFrame:", .{NSRect{ .x = 1, .y = (r.h - hh) / 2.0, .w = @max(1.0, r.w - 2.0), .h = hh }});
        }
        if (g_panel_hidden) {
            msg(void, g_panel, "orderFrontRegardless", .{});
            g_panel_hidden = false;
        }
    } else if (!g_panel_hidden) {
        msg(void, g_panel, "orderOut:", .{@as(id, null)});
        g_panel_hidden = true;
    }
}
