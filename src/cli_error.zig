//! The one place a command-line failure becomes a sentence a user can act on. MLX-free.
//! A caller names the flag or path it was reading beside the sentence; the sentence says what to do.
const std = @import("std");

/// What `err` means at the command line; null when the failure was already reported in full above, so a second line
/// would only repeat it.
pub fn describe(err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PromptHttpStatus, error.LoadFailed, error.LaunchFlagUnsupported, error.ExpertStreamingRequired, error.PromptModeConflict, error.PromptModelRequired, error.BenchServerDown => null,
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
        error.TeacherMustBeLossless => "a teacher capture is lossless: drop --expert-pick-tolerance and --drop-expert",
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

test "cli error: a raw error name becomes a sentence, a reported failure stays silent" {
    const t = std.testing;
    try t.expect(std.mem.indexOf(u8, describe(error.UnknownSubcommand).?, "unknown subcommand") != null);
    try t.expect(std.mem.indexOf(u8, describe(error.BaselineUnreadable).?, "baseline.json") != null);
    try t.expect(std.mem.indexOf(u8, describe(error.MissingModel).?, "--model") != null);
    try t.expect(describe(error.PromptHttpStatus) == null);
    try t.expect(describe(error.LoadFailed) == null);
    try t.expectEqualStrings("OutOfMemory", describe(error.OutOfMemory).?);
}
