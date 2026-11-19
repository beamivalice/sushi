//! The one place a command-line failure becomes a sentence a user can act on. MLX-free.
//! A caller names the flag or path it was reading beside the sentence; the sentence says what to do.
const std = @import("std");

/// What `err` means at the command line. Never empty: a site that returns an error without logging still exits with this
/// line, so a short repeat of a line logged above is the price of never exiting silent.
pub fn describe(err: anyerror) []const u8 {
    return switch (err) {
        error.PromptHttpStatus => "the server refused the prompt",
        error.LoadFailed => "the model did not load",
        error.LaunchFlagUnsupported => "a launch flag has no effect on this model: drop it",
        error.ExpertStreamingRequired => "this checkpoint streams its experts from SSD: pass --ssd-budget-gb <n> or set \"ssd_budget_gb\" in model-settings.json",
        error.PromptModeConflict => "--prompt runs alone: drop serve, --serve, --host, --port and --tool",
        error.PromptModelRequired => "--prompt needs --model <path> or run <model>",
        error.BenchServerDown => "no server answers at --url: start one or pass its <host:port>",
        error.BenchNoModel => "the server lists no model: pass --model <id>",
        error.FileNotFound => "no such file or folder",
        error.MissingSubcommand => "a subcommand is required (see --help)",
        error.UnknownSubcommand => "unknown subcommand (see --help)",
        error.UnknownFlag, error.UnknownArgument => "unrecognized argument (see --help)",
        error.MissingFlagValue, error.MissingValue => "a flag is missing its value",
        error.BadFlagValue, error.BadValue => "a flag has a value it cannot take",
        error.ConflictingFlags => "two flags conflict; pass one",
        error.MissingRequired => "a required flag is missing",
        error.MissingModel => "--model <pack folder> is required",
        error.MissingPrompts => "--prompts <fixture folder, .jsonl, or standard4> is required",
        error.MissingOut => "--out <folder> is required",
        error.MissingFixture => "--fixture <teacher fixture folder> is required",
        error.TeacherMustBeLossless => "a teacher capture is lossless: drop --expert-pick-tolerance, --expert-cache-bias and --drop-expert",
        error.LayerMajorFlagWithoutLayerMajor => "--batch-windows, --imatrix-windows and --pause-file need --layer-major",
        error.LayerMajorNeedsNativeGlmTeacher => "--layer-major needs the native GLM BF16 teacher as --model",
        error.BaselineUnreadable => "no baseline.json there: pass the folder `sushi kld capture` wrote",
        error.BadBaselineJson => "baseline.json is not a teacher fixture",
        error.PromptSourceUnreadable => "cannot read the prompt source",
        error.BadPromptJsonl => "a line of the prompt file is not a prompt",
        error.CalibFileMissing => "the --calib file does not exist",
        else => @errorName(err),
    };
}

test "cli error: a raw error name becomes a sentence" {
    const t = std.testing;
    try t.expect(std.mem.indexOf(u8, describe(error.UnknownSubcommand), "unknown subcommand") != null);
    try t.expect(std.mem.indexOf(u8, describe(error.BaselineUnreadable), "baseline.json") != null);
    try t.expect(std.mem.indexOf(u8, describe(error.MissingModel), "--model") != null);
    try t.expectEqualStrings("OutOfMemory", describe(error.OutOfMemory));
}

test "cli error: every error a command returns prints a line naming the fix, none exits silent" {
    const t = std.testing;
    try t.expect(std.mem.indexOf(u8, describe(error.ExpertStreamingRequired), "--ssd-budget-gb") != null);
    try t.expect(std.mem.indexOf(u8, describe(error.LaunchFlagUnsupported), "drop") != null);
}
