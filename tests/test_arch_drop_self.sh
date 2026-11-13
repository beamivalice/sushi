#!/bin/bash
# sushi-test: tier=hermetic needs=git
# test_arch_drop_self.sh — `tests/test_arch_drop.sh` itself, in a scratch repository of four toy architectures with a
# stand-in for zig, so nothing is built:
#   - which architectures it takes: all by default, with `--since <ref>` only those whose folder changed after it;
#   - every job it starts gets its own arch list, a failing job fails the run and names itself, the others pass;
#   - stopped by its PID, it leaves no step running and no worktree behind.
set -u
HERE=$(cd "$(dirname "$0")" && pwd -P)
T=$(mktemp -d "${TMPDIR:-/tmp}/arch-drop-plan.XXXXXX")
trap 'rm -rf "$T"' EXIT
FAIL=0
check() { # <description> <want> <got>
  if [ "$2" = "$3" ]; then echo "  PASS: $1"; else echo "  FAIL: $1"; echo "    want: $2"; echo "    got:  $3"; FAIL=1; fi
}

mkdir -p "$T/tests" "$T/src/arch" "$T/src/mimo2" "$T/src/dsv41" "$T/src/glm5" "$T/src/qwen4"
cp "$HERE/test_arch_drop.sh" "$T/tests/"
for a in mimo2 dsv41 glm5 qwen4; do
  echo "const x = 1;" > "$T/src/$a/arch.zig"
  echo "    .{ \"$a\", @import(\"../$a/arch.zig\") }, // arch:$a" >> "$T/src/arch/registry.zig"
done
echo "const shared = 1;" > "$T/src/shared.zig"
mkdir -p "$T/lib" "$T/tests/guards"
: > "$T/lib/.keep"
: > "$T/tests/guards/lint.allow"
printf '#!/bin/bash\nexit 0\n' > "$T/tests/test_doc_links.sh"
chmod +x "$T/tests/test_doc_links.sh"
git -C "$T" init -q
git -C "$T" -c user.name=t -c user.email=t@example.invalid add -A
git -C "$T" -c user.name=t -c user.email=t@example.invalid commit -q -m base
echo "const x = 2;" > "$T/src/dsv41/arch.zig"
echo "const shared = 2;" > "$T/src/shared.zig"
git -C "$T" -c user.name=t -c user.email=t@example.invalid commit -q -am "one arch and a shared file"
echo "const y = 1;" > "$T/src/glm5/new.zig"
git -C "$T" -c user.name=t -c user.email=t@example.invalid add -A
git -C "$T" -c user.name=t -c user.email=t@example.invalid commit -q -m "a second arch"

plan() { (cd "$T" && bash tests/test_arch_drop.sh --plan "$@" 2>&1 | grep '^  plan:'); }
check "with no argument every architecture is dropped and built alone" \
  "  plan: drop mimo2 dsv41 glm5 qwen4; alone mimo2 dsv41 glm5 qwen4" "$(plan)"
check "--since takes the architectures whose folder changed after the commit" \
  "  plan: drop dsv41 glm5; alone dsv41 glm5" "$(plan --since HEAD~2)"
check "--since one commit back takes the one touched there" \
  "  plan: drop glm5; alone glm5" "$(plan --since HEAD~1)"
check "--since HEAD takes none: a shared file alone drops nothing" \
  "  plan: drop (none); alone (none)" "$(plan --since HEAD)"
(cd "$T" && bash tests/test_arch_drop.sh --plan --since no-such-ref > "$T/out" 2>&1); rc=$?
check "a ref that is no commit is refused before anything runs" "2" "$rc"
(cd "$T" && bash tests/test_arch_drop.sh --frobnicate > "$T/out" 2>&1); rc=$?
check "an unknown argument is refused" "2" "$rc"
(cd "$T" && DROP_ONLY=qwen4 bash tests/test_arch_drop.sh --plan 2>&1 | grep '^  plan:' > "$T/out")
check "DROP_ONLY names the one architecture" "  plan: drop qwen4; alone qwen4" "$(cat "$T/out")"

# The jobs, with a stand-in zig that records its call and fails where told to.
cat > "$T/zig" <<'ZIG'
#!/bin/bash
echo "$$ $*" >> "$CALLS"
case " $* " in *" $FAIL_ON "*) exit 1 ;; esac
[ -z "${HOLD:-}" ] || exec sleep 300
exit 0
ZIG
chmod +x "$T/zig"
run() { CALLS=$T/calls ZIG=$T/zig TMPDIR=$T/tmp bash "$T/tests/test_arch_drop.sh" "$@" > "$T/out" 2>&1; }
mkdir -p "$T/tmp"

: > "$T/calls"
FAIL_ON=never run; rc=$?
check "every job passing, the run passes" "0" "$rc"
check "each arch is dropped with the other three and built alone, the host tier in each drop" \
  "4 4 4 4" "$(grep -c -- '^[0-9]* build -Darchs=[a-z0-9]*,[a-z0-9]*,[a-z0-9]*$' "$T/calls") $(grep -c 'test-build' "$T/calls") $(grep -c 'test-host' "$T/calls") $(grep -c -- '-Doptimize=ReleaseFast -Darchs=[a-z0-9]* -p' "$T/calls")"
check "a drop's builds never name the arch dropped" "0" "$(grep -c -- '-Darchs=.*,.*,.*,' "$T/calls")"
check "no worktree or run directory is left" "" "$(ls "$T/tmp"; git -C "$T" worktree list | sed 1d)"

: > "$T/calls"
FAIL_ON='-Darchs=mimo2,glm5,qwen4' run; rc=$?
check "one failing job fails the run" "1" "$rc"
check "the failed drop is named and the other seven jobs pass" "1 7" \
  "$(grep -c 'FAIL: dropping dsv41' "$T/out") $(grep -c '  PASS: \(dropping\|building only\)' "$T/out")"

: > "$T/calls"
CALLS=$T/calls ZIG=$T/zig TMPDIR=$T/tmp FAIL_ON=never HOLD=1 bash "$T/tests/test_arch_drop.sh" > "$T/out" 2>&1 &
script=$!
for _ in $(seq 1 100); do [ "$(wc -l < "$T/calls" | tr -d ' ')" -ge 8 ] && break; sleep 0.2; done
check "the eight jobs run at once: each has a step open before any ends" "8" "$(wc -l < "$T/calls" | tr -d ' ')"
kill $script 2>/dev/null
wait $script 2>/dev/null
alive=0
while read -r pid _; do kill -0 "$pid" 2>/dev/null && alive=$((alive + 1)); done < "$T/calls"
check "stopped by its PID, it leaves no step running" "0" "$alive"
check "and no worktree or run directory" "" "$(ls "$T/tmp"; git -C "$T" worktree list | sed 1d)"

[ $FAIL -eq 0 ] && { echo "PASS"; exit 0; }
echo "FAIL"; exit 1
