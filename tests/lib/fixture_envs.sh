#!/bin/bash
# sushi-test: tier=lib
# The tiny-fixture envs a full `zig build test` sets (docs/testsuite.md, the pack tier's envs). Sourced by tests/gate.sh.
# SUSHI_FIXTURE_ENVS lists the envs that each name a fixture folder. Sourcing fails while the tiny models still sit in
# the old ${SUSHI_MODELS_DIR}/test-fixtures folder, naming the mv.

export SUSHI_MODELS_DIR="${SUSHI_MODELS_DIR:-$HOME/.sushi/models}"
SUSHI_FIXTURES="$SUSHI_MODELS_DIR/tiny-models"
# The tiny models were in test-fixtures/ before; a box that still keeps them there is told the move, not read from it.
if [ -d "$SUSHI_MODELS_DIR/test-fixtures" ] && [ ! -d "$SUSHI_FIXTURES" ]; then
  echo "fixture_envs.sh: the tiny models moved to $SUSHI_FIXTURES; run: mv \"$SUSHI_MODELS_DIR/test-fixtures\" \"$SUSHI_FIXTURES\"" >&2
  return 1
fi
export QWEN4_STQ_TINY="$SUSHI_FIXTURES/qwen4-stq-tiny-hd256"
export QWEN4_BF16_TINY="$SUSHI_FIXTURES/qwen4-bf16-tiny-hd256"
export DSV41_STQ_TINY="$SUSHI_FIXTURES/dsv41-stq-tiny"
export DSV41_STQ_TINY_PM="$SUSHI_FIXTURES/dsv41-stq-tiny-pm"
export DSV41_TINY="$SUSHI_FIXTURES/dsv41-dspark-tiny"
export DSV41_EPI_TINY="$SUSHI_FIXTURES/dsv41-epi-tiny"
export HY4_BF16_TINY="$SUSHI_FIXTURES/hy4-bf16-tiny"
export HY4_STQ_TINY="$SUSHI_FIXTURES/hy4-stq-tiny"
export HY4_BF16_TINY_STQ="$SUSHI_FIXTURES/hy4-bf16-tiny-stq"
export QWEN4_TEST_MODEL="$SUSHI_FIXTURES/qwen4-stq-cut"
SUSHI_FIXTURE_ENVS="QWEN4_STQ_TINY QWEN4_BF16_TINY DSV41_STQ_TINY DSV41_STQ_TINY_PM DSV41_TINY DSV41_EPI_TINY HY4_BF16_TINY HY4_STQ_TINY HY4_BF16_TINY_STQ QWEN4_TEST_MODEL"
