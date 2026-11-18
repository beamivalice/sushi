#!/bin/bash
# sushi-test: tier=lib
# The tiny-fixture envs a full `zig build test` sets (docs/testsuite.md, the pack tier's envs). Sourced by tests/gate.sh.
# SUSHI_FIXTURE_ENVS lists the envs that each name a fixture folder.

export SUSHI_MODELS_DIR="${SUSHI_MODELS_DIR:-$HOME/.sushi/models}"
SUSHI_FIXTURES="$SUSHI_MODELS_DIR/test-fixtures"
export QWEN4_STQ_TINY="$SUSHI_FIXTURES/qwen4-stq-tiny-hd256"
export QWEN4_BF16_TINY="$SUSHI_FIXTURES/qwen4-bf16-tiny-hd256"
export DSV41_STQ_TINY="$SUSHI_FIXTURES/dsv41-stq-tiny"
export DSV41_STQ_TINY_PM="$SUSHI_FIXTURES/dsv41-stq-tiny-pm"
export DSV41_TINY="$SUSHI_FIXTURES/dsv41-dspark-tiny"
export DSV41_EPI_TINY="$SUSHI_FIXTURES/dsv41-epi-tiny"
export QWEN4_TEST_MODEL="$SUSHI_FIXTURES/qwen4-stq-cut"
SUSHI_FIXTURE_ENVS="QWEN4_STQ_TINY QWEN4_BF16_TINY DSV41_STQ_TINY DSV41_STQ_TINY_PM DSV41_TINY DSV41_EPI_TINY QWEN4_TEST_MODEL"
