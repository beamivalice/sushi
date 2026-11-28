//! DeepSeek-V4.1's DSpark rounds: the shared round body (`drafter_round`) over DeepSeek's model, its stages drafting a
//! block and its trunk verifying and cutting back (`dsv41_serving`).
const drafter_round = @import("../drafter_round.zig");
const dsv41_config = @import("dsv41_config.zig");
const dsv41_model = @import("dsv41_model.zig");
const dsv41_serving = @import("dsv41_serving.zig");
const dsv41_state = @import("dsv41_state.zig");
const arch_drafter_rounds = @import("../arch/drafter_rounds.zig");
const Trunk = @import("../trunk.zig").Trunk;

const Hooks = struct {
    pub const Draft = dsv41_model.Draft;
    pub const row_grammar = @import("../row_grammar.zig");
    /// A ring slack's rows.
    pub const max_verify_rows: u8 = @intCast(dsv41_config.ring_slack);
    pub const diag_env = "SUSHI_DSV41_DSPARK_ROWS";
    /// Drafts of the widest pass that still queues its routed chains: a copy round past them pays a wait a layer.
    pub const copy_narrow: usize = dsv41_config.route_spec_rows - 1;
    pub const draft = dsv41_serving.dsv41DraftImpl;
    pub const draftMeasured = dsv41_serving.dsv41DraftMeasured;
    pub const verify = dsv41_serving.dsv41VerifyImpl;
    pub const finish = dsv41_serving.dsv41FinishImpl;
    pub const verdictStream = dsv41_serving.verdictStream;
    pub const sampledDraws = dsv41_serving.sampledDraws;

    /// A row of this load is the same bytes at every verify width (`Model.widthExact`: trellis banks only), so a greedy or
    /// seeded request keeps the adaptive width.
    pub fn widthExact(xfm: *Trunk) bool {
        return if (dsv41_state.modelOf(xfm)) |m| m.widthExact() else false;
    }

    pub fn vocab(xfm: *Trunk) usize {
        return if (dsv41_state.modelOf(xfm)) |m| m.cfg.vocab else 0;
    }

    /// The policy's default where DeepSeek's model is not loaded.
    pub fn stageBlock(xfm: *Trunk) u32 {
        return if (dsv41_state.modelOf(xfm)) |m| m.cfg.dspark.block else arch_drafter_rounds.default_block;
    }

    pub fn copyArm(xfm: *Trunk) ?drafter_round.CopyArm {
        const m = dsv41_state.modelOf(xfm) orelse return null;
        return .{ .exact_rows = m.exactVerifyRows(), .block = m.cfg.dspark.block };
    }
};

const R = drafter_round.Rounds(Hooks);
pub const drafterRowsForRoom = R.drafterRowsForRoom;
pub const drafterFixedRows = R.drafterFixedRows;
pub const dsv41_max_drafts = R.max_drafts;
pub const dsv41_max_verify_rows = R.max_verify_rows;
pub const DrafterDiag = R.DrafterDiag;
pub const parseDrafterDiag = R.parseDrafterDiag;
pub const drafterDiag = R.drafterDiag;
pub const stageBlock = R.stageBlock;
pub const drafterEligible = R.drafterEligible;
pub const nextDrafterRound = R.nextDrafterRound;
pub const measuredRound = R.measuredRound;
pub const RoundMeter = drafter_round.RoundMeter;
pub const ConstrainedEntry = R.ConstrainedEntry;
pub const Pending = R.Pending;
pub const rowView = R.rowView;
pub const drafterNoteSerial = R.drafterNoteSerial;
pub const drafterConstrainedEntry = R.drafterConstrainedEntry;
pub const drafterSamplePending = R.drafterSamplePending;
pub const drafterJudgePending = R.drafterJudgePending;
pub const drafterObserve = R.drafterObserve;
pub const drafterEmitPending = R.drafterEmitPending;
pub const DrafterResume = R.DrafterResume;
pub const drafterResumeAfterForced = R.drafterResumeAfterForced;
pub const roundRows = R.roundRows;
pub const Verdict = R.Verdict;
pub const VerdictIn = R.VerdictIn;
pub const LazyVerdict = R.LazyVerdict;
pub const verdictBegin = R.verdictBegin;
pub const verdictInFor = R.verdictInFor;
pub const verdictOf = R.verdictOf;
pub const commitVerdict = R.commitVerdict;
pub const settleDrafted = R.settleDrafted;

/// Whether a decoding slot can join a grouped DSpark speculative tick.
pub fn groupEligible(gen: *const @import("../generate.zig").Generator) bool {
    if (!gen.drafter_on or gen.done or gen.spec_disabled_runtime) return false;
    if (gen.sampling.constraint != null or @import("../sampler.zig").penaltyActive(gen.sampling)) return false;
    if (gen.sampling.think_bound != null or gen.sampling.call_force != null) return false;
    if (gen.max_tokens <= gen.completion_tokens + 1) return false;
    if (!@import("../mtp.zig").mtpCtxWithinLimit(gen.opts.max_mtp_ctx, gen.mtpKvLen())) return false;
    return !gen.samplesSync();
}
