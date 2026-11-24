//! DeepSeek-V4.1's DSpark rounds: the shared round body (`dspark_round`) over DeepSeek's model, its stages drafting a
//! block and its trunk verifying and cutting back (`dsv41_serving`).
const dspark_round = @import("../dspark_round.zig");
const dsv41_config = @import("dsv41_config.zig");
const dsv41_model = @import("dsv41_model.zig");
const dsv41_serving = @import("dsv41_serving.zig");
const dsv41_state = @import("dsv41_state.zig");
const arch_dspark = @import("../arch/dspark.zig");
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
        return if (dsv41_state.modelOf(xfm)) |m| m.cfg.dspark.block else arch_dspark.default_block;
    }

    pub fn copyArm(xfm: *Trunk) ?dspark_round.CopyArm {
        const m = dsv41_state.modelOf(xfm) orelse return null;
        return .{ .exact_rows = m.exactVerifyRows(), .block = m.cfg.dspark.block };
    }
};

const R = dspark_round.Rounds(Hooks);
pub const dsparkRowsForRoom = R.dsparkRowsForRoom;
pub const dsparkFixedRows = R.dsparkFixedRows;
pub const dsv41_max_drafts = R.max_drafts;
pub const dsv41_max_verify_rows = R.max_verify_rows;
pub const DsparkDiag = R.DsparkDiag;
pub const parseDsparkDiag = R.parseDsparkDiag;
pub const dsparkDiag = R.dsparkDiag;
pub const stageBlock = R.stageBlock;
pub const dsparkEligible = R.dsparkEligible;
pub const nextDspark = R.nextDspark;
pub const measuredRound = R.measuredRound;
pub const RoundMeter = dspark_round.RoundMeter;
pub const ConstrainedEntry = R.ConstrainedEntry;
pub const Pending = R.Pending;
pub const rowView = R.rowView;
pub const dsparkNoteSerial = R.dsparkNoteSerial;
pub const dsparkConstrainedEntry = R.dsparkConstrainedEntry;
pub const dsparkSamplePending = R.dsparkSamplePending;
pub const dsparkJudgePending = R.dsparkJudgePending;
pub const dsparkObserve = R.dsparkObserve;
pub const dsparkEmitPending = R.dsparkEmitPending;
pub const DsparkResume = R.DsparkResume;
pub const dsparkResumeAfterForced = R.dsparkResumeAfterForced;
