#!/bin/bash
# sushi-test: tier=tool
# The hermetic guards CI's host tier runs, as one list, so a local landing runs exactly what CI runs.
# A skip is a failure here: test_release_workflow_gates needs PyYAML (python3 -m pip install pyyaml).
set -uo pipefail
cd "$(dirname "$0")/.."

fail=0
note() { if [ -n "${GITHUB_ACTIONS:-}" ]; then echo "::error::$1"; else echo "FAIL $1" >&2; fi; fail=1; }
for t in test_release test_release_workflow_gates test_brewfile_covers_build_tools test_build_mlx_pins \
         test_private_cache test_gpu_lock test_run test_doc_links; do
  [ -n "${GITHUB_ACTIONS:-}" ] && echo "::group::$t"
  bash "tests/$t.sh" || note "tests/$t.sh failed (exit $?)"
  [ -n "${GITHUB_ACTIONS:-}" ] && echo "::endgroup::"
done
python3 tests/test_suite_guards.py || note "tests/test_suite_guards.py failed"
python3 tests/test_shell_guards.py || note "tests/test_shell_guards.py failed"
bash tests/run.sh --check || note "tests/run.sh --check failed"
TEST_STEP=test-host bash tests/test_test_runner_quiet.sh || note "tests/test_test_runner_quiet.sh failed"
exit $fail
