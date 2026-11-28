//! `SUSHIBAR_RENDER_HEADER=<dir>`: offline PNG screenshot harness for checking every `sushibar` view state
//! (header cards, button rows, port rows, port field, log window, menu-bar icons/readouts) in light and dark
//! appearance at 2x without a screen.
const std = @import("std");
const io_util = @import("../io_util.zig");
const logic = @import("sushibar_logic.zig");
const objc = @import("objc.zig");
const header_view = @import("header_view.zig");
const port_field = @import("port_field.zig");
const button_row = @import("button_row.zig");
const log_view = @import("log_view.zig");
const port_row = @import("port_row.zig");
const bar_button = @import("bar_button.zig");

fn buttonLook(b: logic.UpdateButton) button_row.Look {
    return .{ .label = b.label, .symbol = b.symbol, .tip = b.tip, .accent = b.accent, .enabled = b.enabled, .closes_menu = b.closes_menu, .spinner = b.spinner };
}

pub fn run(io: std.Io, gpa: std.mem.Allocator, dir: []const u8) noreturn {
    _ = objc.msg(objc.id, objc.class("NSApplication"), "sharedApplication", .{});
    _ = header_view.create(
        objc.loadSvg(@embedFile("icon-online.svg"), false, 34, 28),
        objc.loadSvg(@embedFile("icon-offline.svg"), true, 34, 28),
        null,
        null,
    );
    io_util.makeDirPath(io, dir) catch {};
    const t0: i64 = 1_700_000_000_000;
    // A busy session with a gap; one history for the card with a short outage in it, one without.
    var hist: [150]logic.Hist = undefined;
    var hist_gap: [150]logic.Hist = undefined;
    var n: usize = 0;
    var n_gap: usize = 0;
    var gen: u64 = 0;
    for (0..150) |i| {
        const f: f64 = @floatFromInt(i);
        const rate: f64 = if (i >= 25) 95 + 20 * @sin(f * 0.7) + 10 * @sin(f * 0.23) else 0;
        gen += @intFromFloat(rate * 2);
        const sample: logic.Hist = .{ .t_ms = t0 - @as(i64, @intCast(149 - i)) * 2000, .gen = gen };
        hist[n] = sample;
        n += 1;
        if (i >= 100 and i < 103) continue;
        hist_gap[n_gap] = sample;
        n_gap += 1;
    }
    const States = struct { name: []const u8, in: logic.HeaderIn, sparkline: bool = false, outage: bool = false };
    const deep = "Example-Flash-1017-Sushi-STQ-1.75bpw";
    const states = [_]States{
        .{ .name = "stopped", .in = .{ .link = .offline, .port = 12345, .bar_version = "1.5.0-dev1" } },
        .{ .name = "stopped-last-model", .in = .{ .link = .offline, .port = 12345, .bar_version = "1.5.0-dev1", .last_model = deep } },
        .{ .name = "port-taken", .in = .{ .link = .offline, .port = 8080, .bar_version = "1.5.0-dev1", .port_taken = true } },
        .{ .name = "no-sushi", .in = .{ .link = .offline, .port = 12345, .bar_version = "1.5.0-dev1", .sushi_found = false } },
        .{ .name = "starting", .in = .{ .link = .offline, .port = 12345, .bar_version = "1.5.0-dev1", .starting = true, .last_model = deep } },
        .{ .name = "connecting", .in = .{ .link = .offline, .port = 18803, .bar_version = "1.5.0-dev1", .connecting = true } },
        .{ .name = "silent", .in = .{ .link = .silent, .port = 12345, .bar_version = "1.5.0-dev1" } },
        .{ .name = "ready-to-load", .in = .{ .link = .up, .port = 12345, .bar_version = "1.5.0-dev1", .last_model = deep, .gpu_mhz = 0 } },
        .{ .name = "loading", .in = .{ .link = .up, .port = 12345, .bar_version = "1.5.0-dev1", .loading = true, .model = deep, .mem_mb = 12000 } },
        .{ .name = "no-model", .in = .{ .link = .up, .port = 12345, .bar_version = "1.5.0-dev1", .gpu_mhz = 0 } },
        .{ .name = "generating", .sparkline = true, .outage = true, .in = .{
            .link = .up, .port = 12345, .model = deep, .mem_mb = 49971, .bar_version = "1.5.0-dev1",
            .decoding = true, .live = 105.27, .last = 90, .last_prefill = 5843.6, .gpu_mhz = 1195, .gpu_pct = 62,
        } },
        .{ .name = "key-needed", .in = .{ .link = .unauthorized, .port = 12345, .bar_version = "1.5.0-dev1" } },
        .{ .name = "update", .sparkline = true, .in = .{
            .link = .up, .port = 12345, .model = "Example-Flash-Next-Sushi-STQ-2.75bpw", .mem_mb = 31200, .bar_version = "1.5.0",
            .update_available = true, .latest = "1.5.1", .last = 142.6, .last_prefill = 3120, .gpu_mhz = 1380, .gpu_pct = 8,
        } },
        .{ .name = "pack-update", .sparkline = true, .in = .{
            .link = .up, .port = 12345, .model = deep, .mem_mb = 49971, .bar_version = "1.5.0",
            .pack_update = .{ .current = "1.0", .latest = "2.0", .command = "sushi pull example-pack" },
            .last = 90, .last_prefill = 5843.6, .gpu_mhz = 1195, .gpu_pct = 62,
        } },
    };
    for (states) |st| {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const h = logic.buildHeader(arena.allocator(), st.in) catch continue;
        var spark: [160]logic.SparkPoint = undefined;
        const pts = if (st.sparkline) logic.sparkPoints(&spark, if (st.outage) hist_gap[0..n_gap] else hist[0..n], t0, header_view.spark_window_ms) else spark[0..0];
        header_view.update(h, pts);
        for ([_]bool{ false, true }) |dark| {
            var path: [std.fs.max_path_bytes:0]u8 = undefined;
            const pz = std.fmt.bufPrintSentinel(&path, "{s}/header-{s}-{s}.png", .{ dir, st.name, if (dark) "dark" else "light" }, 0) catch continue;
            if (!header_view.renderPng(pz.ptr, dark)) std.debug.print("sushibar: could not render {s}\n", .{pz});
        }
    }
    const field = port_field.create(null);
    for ([_][]const u8{ "empty", "invalid" }) |state| {
        if (std.mem.eql(u8, state, "invalid")) {
            port_field.setText("99999");
            port_field.showInvalid();
        }
        for ([_]bool{ false, true }) |dark| {
            var path: [std.fs.max_path_bytes:0]u8 = undefined;
            const pz = std.fmt.bufPrintSentinel(&path, "{s}/portfield-{s}-{s}.png", .{ dir, state, if (dark) "dark" else "light" }, 0) catch continue;
            if (!port_field.renderPng(field, pz.ptr, dark)) std.debug.print("sushibar: could not render {s}\n", .{pz});
        }
    }
    renderButtonsAndLog(io, dir);
    renderBadge(dir);
    std.process.exit(0);
}

/// The menu bar item in its states, on a plain button of the status button's shape over a menu-bar-like fill.
fn renderBadge(dir: []const u8) void {
    bar_button.loadIcons();
    const Case = struct { name: []const u8, icon: logic.Icon, readout: logic.Readout = logic.readout_none, badge: bool = false, update: bool = false, reading: ?usize = null, dim: bool = false };
    const cases = [_]Case{
        .{ .name = "live", .icon = .ready, .readout = .{ .caption = "TOK/S", .value = "105" } },
        .{ .name = "live-slow", .icon = .ready, .readout = .{ .caption = "TOK/S", .value = "40.0" } },
        .{ .name = "idle", .icon = .ready },
        .{ .name = "update-waiting", .icon = .ready, .update = true },
        .{ .name = "update-waiting-live", .icon = .ready, .update = true, .readout = .{ .caption = "TOK/S", .value = "40.0" } },
        .{ .name = "reading", .icon = .ready, .reading = logic.reading_steps * 3 / 5, .readout = .{ .caption = "READ", .value = "12s" } },
        .{ .name = "reading-start", .icon = .ready, .reading = logic.reading_steps / 10, .readout = .{ .caption = "READ", .value = logic.readout_pending } },
        .{ .name = "reading-minutes", .icon = .ready, .reading = logic.reading_steps / 4, .readout = .{ .caption = "READ", .value = "59:59" } },
        .{ .name = "generating-dim", .icon = .ready, .dim = true, .readout = .{ .caption = "TOK/S", .value = "105" } },
        .{ .name = "badge", .icon = .ready, .badge = true },
        .{ .name = "no-model", .icon = .empty },
        .{ .name = "loading", .icon = .loading },
        .{ .name = "key-needed", .icon = .empty, .badge = true },
        .{ .name = "offline", .icon = .outline },
    };
    for (cases) |c| for ([_]bool{ false, true }) |dark| {
        const box = objc.msg(objc.id, objc.msg(objc.id, objc.class("NSBox"), "alloc", .{}), "initWithFrame:", .{objc.NSRect{ .x = 0, .y = 0, .w = 168, .h = 24 }});
        defer objc.release(box);
        objc.msg(void, box, "setBoxType:", .{@as(isize, 4)});
        objc.msg(void, box, "setBorderWidth:", .{@as(f64, 0)});
        objc.msg(void, box, "setContentViewMargins:", .{objc.NSSize{ .width = 0, .height = 0 }});
        objc.msg(void, box, "setFillColor:", .{objc.msg(objc.id, objc.class("NSColor"), "colorWithWhite:alpha:", .{ @as(f64, if (dark) 0.16 else 0.92), @as(f64, 1) })});
        const button = objc.msg(objc.id, objc.msg(objc.id, objc.class("NSButton"), "alloc", .{}), "initWithFrame:", .{objc.NSRect{ .x = 4, .y = 0, .w = 160, .h = 24 }});
        objc.msg(void, button, "setBordered:", .{false});
        objc.msg(void, button, "setImagePosition:", .{@as(isize, 2)});
        const preview = bar_button.previewImage(c.icon, c.reading, c.dim);
        defer if (preview.owned) objc.release(preview.img);
        objc.msg(void, button, "setImage:", .{preview.img});
        // The status bar draws a template image in the bar's text colour; a plain button would draw it dimmed.
        objc.msg(void, button, "setContentTintColor:", .{objc.color("labelColor")});
        bar_button.attachPreview(button, c.readout, c.badge, c.update);
        objc.msg(void, box, "addSubview:", .{button});
        objc.release(button);
        var path: [std.fs.max_path_bytes:0]u8 = undefined;
        const pz = std.fmt.bufPrintSentinel(&path, "{s}/menubar-{s}-{s}.png", .{ dir, c.name, if (dark) "dark" else "light" }, 0) catch return;
        _ = objc.renderViewPng(box, objc.msg(objc.NSRect, box, "bounds", .{}), pz.ptr, dark);
    };
}

fn renderButtonsAndLog(io: std.Io, dir: []const u8) void {
    const row = button_row.createFooter();
    const Case = struct { name: []const u8, uv: logic.UpdateView = .{}, hover: ?usize = null, pressed: ?usize = null };
    const cases = [_]Case{
        .{ .name = "normal" },
        .{ .name = "hover-log", .hover = 0 },
        .{ .name = "pressed-quit", .hover = 2, .pressed = 2 },
        .{ .name = "update", .uv = .{ .available = true, .latest = "1.5.1", .path = .server } },
        .{ .name = "update-hover", .uv = .{ .available = true, .latest = "1.5.1", .path = .server }, .hover = 1 },
        .{ .name = "update-pressed", .uv = .{ .available = true, .latest = "1.5.1", .path = .server }, .pressed = 1 },
        .{ .name = "brew", .uv = .{ .available = true, .latest = "1.5.1", .path = .brew, .command = "brew upgrade sushi" } },
        .{ .name = "brew-copied", .uv = .{ .available = true, .latest = "1.5.1", .path = .brew, .command = "brew upgrade sushi", .copied = true } },
        .{ .name = "checking", .uv = .{ .busy = .checking } },
        .{ .name = "up-to-date", .uv = .{ .checked = .up_to_date } },
        .{ .name = "failed", .uv = .{ .checked = .failed } },
        .{ .name = "failed-hover", .uv = .{ .checked = .failed }, .hover = 1 },
    };
    var tip_b: [160]u8 = undefined;
    for (cases) |c| {
        var tip: [160]u8 = undefined;
        const b = logic.updateButton(&tip, c.uv);
        button_row.setUpdate(buttonLook(b));
        for ([_]bool{ false, true }) |dark| {
            var path: [std.fs.max_path_bytes:0]u8 = undefined;
            const pz = std.fmt.bufPrintSentinel(&path, "{s}/buttons-{s}-{s}.png", .{ dir, c.name, if (dark) "dark" else "light" }, 0) catch continue;
            if (!button_row.renderPng(row, pz.ptr, dark, c.hover, c.pressed, "")) std.debug.print("sushibar: could not render {s}\n", .{pz});
        }
    }
    button_row.setUpdate(buttonLook(logic.updateButton(&tip_b, .{})));
    var qb: [128]u8 = undefined;
    const q = logic.quitButton(&qb, true, 1);
    button_row.setQuit(.{ .label = q.label, .symbol = if (q.stops) "" else "power", .tip = q.tip, .selector = if (q.stops) "quitStop:" else null });
    for ([_]bool{ false, true }) |dark| {
        var path: [std.fs.max_path_bytes:0]u8 = undefined;
        const pz = std.fmt.bufPrintSentinel(&path, "{s}/buttons-quit-stop-{s}.png", .{ dir, if (dark) "dark" else "light" }, 0) catch continue;
        if (!button_row.renderPng(row, pz.ptr, dark, 2, null, "")) std.debug.print("sushibar: could not render {s}\n", .{pz});
    }
    renderServerRow(dir);
    renderPortRows(dir);
    const sample =
        \\2026-11-04 09:12:01 [info] Logging to /Users/me/.sushi/logs/sushi-11250.log (rotates at 100 MB)
        \\2026-11-04 09:12:02 [info] sushi 1.5.0 serving on http://127.0.0.1:11250
        \\2026-11-04 09:12:04 [info] [kv-cache] affine8 KV, context 262144
        \\2026-11-04 09:12:09 [info] Loaded Example-Flash-Next-Sushi-STQ-2.75bpw in 4.8 s
        \\2026-11-04 09:13:20 [info] <- 412+0 tokens (chat, stream)
        \\2026-11-04 09:13:20 [warn] [admission] prefill chunk lowered to 2048 for the free memory
        \\2026-11-04 09:13:31 [info] [spec-stats] accepted 2.31 of 3 drafted
        \\2026-11-04 09:13:34 [info] -> 288 tokens, 142.6 tok/s decode
        \\2026-11-04 09:14:02 [error] [load] chat template: line 1 column 12: unsupported filter 'tojson_pretty'
        \\2026-11-04 09:14:40 [info] [hot-cache] kept 3 entries, 1.2 GB
        \\
    ;
    const std_path = "/Users/me/.sushi/logs/sushi-11250.log";
    for ([_]bool{ false, true }) |dark| {
        var path: [std.fs.max_path_bytes:0]u8 = undefined;
        const pz = std.fmt.bufPrintSentinel(&path, "{s}/log-window-{s}.png", .{ dir, if (dark) "dark" else "light" }, 0) catch continue;
        if (!log_view.renderPng(pz.ptr, dark, .{ .port = 11250, .src = .{ .kind = .standard, .path = std_path }, .sample = sample })) std.debug.print("sushibar: could not render {s}\n", .{pz});
    }
    var nb: [600]u8 = undefined;
    const notice = logic.logNotice(&nb, 11250, .{ .kind = .standard, .path = std_path });
    var path: [std.fs.max_path_bytes:0]u8 = undefined;
    if (std.fmt.bufPrintSentinel(&path, "{s}/log-window-missing-light.png", .{dir}, 0)) |pz| {
        if (!log_view.renderPng(pz.ptr, false, .{ .port = 11250, .src = .{ .kind = .standard, .path = std_path }, .sample = notice, .has_file = false })) std.debug.print("sushibar: could not render {s}\n", .{pz});
    } else |_| {}
    renderFollowedLog(io, dir);
}

/// Start/Stop beside Load/Unload for each state the pair can be in, with the line under them.
fn renderServerRow(dir: []const u8) void {
    const row = button_row.createServerRow();
    const m = "Example-Flash-1017-Sushi-STQ-1.75bpw";
    const Case = struct { name: []const u8, s: logic.Server, m: logic.ModelIn, stopping: bool = false, hover: ?usize = null, pressed: ?usize = null };
    const cases = [_]Case{
        .{ .name = "stopped", .s = .{ .link = .offline }, .m = .{ .link = .offline, .last_model = m } },
        .{ .name = "stopped-no-last", .s = .{ .link = .offline }, .m = .{ .link = .offline } },
        .{ .name = "starting", .s = .{ .link = .offline, .ours_alive = true }, .m = .{ .link = .offline, .starting = true, .last_model = m } },
        .{ .name = "ready-to-load", .s = .{ .link = .up, .ours_alive = true }, .m = .{ .link = .up, .last_model = m } },
        .{ .name = "ready-hover-load", .s = .{ .link = .up, .ours_alive = true }, .m = .{ .link = .up, .last_model = m }, .hover = 4 },
        .{ .name = "no-last-model", .s = .{ .link = .up, .ours_alive = true }, .m = .{ .link = .up } },
        .{ .name = "loading", .s = .{ .link = .up, .ours_alive = true }, .m = .{ .link = .up, .last_model = m, .loading = true } },
        .{ .name = "loaded", .s = .{ .link = .up, .ours_alive = true }, .m = .{ .link = .up, .model = m, .last_model = m } },
        .{ .name = "loaded-hover-unload", .s = .{ .link = .up, .ours_alive = true }, .m = .{ .link = .up, .model = m, .last_model = m }, .hover = 4 },
        .{ .name = "loaded-pressed-stop", .s = .{ .link = .up, .ours_alive = true }, .m = .{ .link = .up, .model = m, .last_model = m }, .hover = 3, .pressed = 3 },
        .{ .name = "unloading", .s = .{ .link = .up, .ours_alive = true }, .m = .{ .link = .up, .model = m, .last_model = m, .unloading = true } },
        .{ .name = "stopping", .s = .{ .link = .up, .ours_alive = true }, .m = .{ .link = .up, .model = m, .last_model = m }, .stopping = true },
    };
    for (cases) |c| {
        const tg = logic.toggle(c.s, c.stopping);
        var tip: [160]u8 = undefined;
        const mb = logic.modelButton(&tip, c.m);
        button_row.setStart(.{ .label = tg.label, .symbol = tg.symbol, .tip = tg.tip, .enabled = tg.enabled, .closes_menu = false });
        button_row.setLoad(.{ .label = mb.label, .symbol = mb.symbol, .tip = mb.tip, .enabled = mb.enabled, .closes_menu = false, .spinner = mb.spinner });
        for ([_]bool{ false, true }) |dark| {
            var path: [std.fs.max_path_bytes:0]u8 = undefined;
            const pz = std.fmt.bufPrintSentinel(&path, "{s}/serverrow-{s}-{s}.png", .{ dir, c.name, if (dark) "dark" else "light" }, 0) catch continue;
            if (!button_row.renderPng(row, pz.ptr, dark, c.hover, c.pressed, mb.why)) std.debug.print("sushibar: could not render {s}\n", .{pz});
        }
    }
}

/// The Port submenu's rows: the default (not removable), then added ports.
fn renderPortRows(dir: []const u8) void {
    const rows = [_]port_row.Row{
        .{ .port = 12345, .label = "12345 — Example-Flash-Next-Sushi-STQ-2.75bpw", .checked = true, .removable = false },
        .{ .port = 18803, .label = "18803 — not running", .checked = false, .removable = true },
        .{ .port = 18802, .label = "18802 — Example-Flash-1017", .checked = false, .removable = true },
        .{ .port = 7000, .label = "7000 — no model", .checked = false, .removable = true },
    };
    const Case = struct { name: []const u8, hover: ?usize = null, x_hover: bool = false };
    for ([_]Case{ .{ .name = "normal" }, .{ .name = "hover-default", .hover = 0 }, .{ .name = "hover-added", .hover = 1 }, .{ .name = "hover-remove", .hover = 2, .x_hover = true } }) |c| {
        for ([_]bool{ false, true }) |dark| {
            var path: [std.fs.max_path_bytes:0]u8 = undefined;
            const pz = std.fmt.bufPrintSentinel(&path, "{s}/portrows-{s}-{s}.png", .{ dir, c.name, if (dark) "dark" else "light" }, 0) catch continue;
            if (!port_row.renderPng(&rows, pz.ptr, dark, c.hover, c.x_hover)) std.debug.print("sushibar: could not render {s}\n", .{pz});
        }
    }
}

/// The window reading a real file: its last lines on open, then what the file grew by while the reader was scrolled
/// up, which the New lines pill announces.
fn renderFollowedLog(io: std.Io, dir: []const u8) void {
    var path: [std.fs.max_path_bytes:0]u8 = undefined;
    const file = std.fmt.bufPrint(&path, "{s}/followed.log", .{dir}) catch return;
    var text: std.Io.Writer.Allocating = .init(std.heap.page_allocator);
    defer text.deinit();
    const w = &text.writer;
    for (1..3001) |i| w.print("line {d} of the first read\n", .{i}) catch return;
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = file, .data = text.written() }) catch return;
    const src: logic.LogSource = .{ .kind = .given, .path = file };
    log_view.sync(11250, src);
    log_view.scrollToTop();
    for (3001..3006) |i| w.print("line {d} appended later\n", .{i}) catch return;
    w.writeAll("a line still being writ") catch return;
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = file, .data = text.written() }) catch return;
    log_view.sync(11250, src);
    for ([_]bool{ false, true }) |dark| {
        var png: [std.fs.max_path_bytes:0]u8 = undefined;
        const pz = std.fmt.bufPrintSentinel(&png, "{s}/log-window-new-lines-{s}.png", .{ dir, if (dark) "dark" else "light" }, 0) catch return;
        if (!log_view.renderPng(pz.ptr, dark, null)) std.debug.print("sushibar: could not render {s}\n", .{pz});
    }
}
