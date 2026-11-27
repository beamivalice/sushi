#!/bin/bash
# sushi-test: tier=lib
# Sourced by tests/lib/boot.sh and tests/run.sh: the model keys and the flag set of each shared boot.

# model key (qwen4, mimo2, glm5, dsv41, hy4) from a key or a config.json model_type; fails on anything else
sushi_model_key() {
    case "${1:-}" in
        qwen4 | qwen4_exp) echo qwen4 ;;
        mimo2 | mimo_v2) echo mimo2 ;;
        glm5 | glm5_next) echo glm5 ;;
        dsv41 | deepseek_v41) echo dsv41 ;;
        hy4 | hy_v4_flash) echo hy4 ;;
        *) return 1 ;;
    esac
}

# Flags of a profile for a model, one line. A streamed DeepSeek never batches, so it serves one slot.
# Only qwen4 and mimo2 carry an MTP head to pin; Hy4-Flash's slots are held to its load's row fit.
sushi_profile_flags() {
    local profile="${1:?sushi_profile_flags <api|exact|ram|knobs> <model>}" model
    model=$(sushi_model_key "${2:-}") || { echo "sushi_profile_flags: unknown model '${2:-}'" >&2; return 1; }
    local common="--metrics --log-level debug" slots="--max-concurrent 4"
    [ "$model" = dsv41 ] && slots="--ssd-budget-gb ${SUSHI_TEST_DSV41_BUDGET:-96} --max-concurrent 1"
    case "$profile" in
        api) echo "$common $slots" ;;
        exact)
            local depth=""
            case "$model" in qwen4 | mimo2) depth=" --mtp-min-depth 3 --mtp-max-depth 3" ;; esac
            echo "$common $slots --prefix-cache-mem off --prefix-cache-disk off$depth"
            ;;
        ram) echo "$common $slots --prefix-cache-mem 2GB --prefix-cache-disk off" ;;
        knobs) echo "$common --think medium --prefix-cache-mem off --prefix-cache-disk off" ;;
        *) echo "sushi_profile_flags: unknown profile '$profile'" >&2; return 1 ;;
    esac
}
