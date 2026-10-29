#!/bin/bash
# sushi-test: tier=tool
# The test runner: reads each script's header and runs a tier, one model pass at a time, on shared boots.
#   tests/run.sh [--tier hermetic,host,pack,live|all] [--model qwen4|mimo2|glm5|dsv41|shared|all]
#                [--only <glob>] [--list] [--check] [--no-matrix] [--keep-going] [--no-build] [--out <dir>]
# Header, scope, boots and exit codes: docs/testsuite.md, "Integration scripts". Written for bash 3.2.
# Every boot goes through lib/boot.sh, which gives it a private HOME and SUSHI_PREFIX_CACHE_DIR.
set -u
export LC_ALL=C

SELF_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SELF_DIR/.." && pwd)
TESTS_DIR="${SUSHI_TESTS_DIR:-$SELF_DIR}"
. "$SELF_DIR/lib/profiles.sh"

TIERS=hermetic,host
MODELS=all
ONLY=
MODE=run
KEEP_GOING=0
NO_BUILD=0
NO_MATRIX=0
OUT=

usage() { sed -n '4,5p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
while [ $# -gt 0 ]; do
    case "$1" in
        --tier) TIERS="${2:?--tier needs a value}"; shift ;;
        --model) MODELS="${2:?--model needs a value}"; shift ;;
        --only) ONLY="${2:?--only needs a glob}"; shift ;;
        --out) OUT="${2:?--out needs a dir}"; shift ;;
        --list) MODE=list ;;
        --check) MODE=check ;;
        --no-matrix) NO_MATRIX=1 ;;
        --keep-going) KEEP_GOING=1 ;;
        --no-build) NO_BUILD=1 ;;
        -h | --help) usage ;;
        *) echo "run.sh: unknown argument '$1'" >&2; usage ;;
    esac
    shift
done
[ "$TIERS" = all ] && TIERS=hermetic,host,pack,live
for t in $(echo "$TIERS" | tr , ' '); do
    case "$t" in hermetic | host | pack | live) ;; *) echo "run.sh: --tier '$t' is not hermetic|host|pack|live" >&2; exit 2 ;; esac
done
case "$MODELS" in
    all) PASSES="qwen4 mimo2 glm5 dsv41 shared" ;;
    qwen4 | mimo2 | glm5 | dsv41 | shared) PASSES="$MODELS" ;;
    *) echo "run.sh: --model '$MODELS' is not qwen4|mimo2|glm5|dsv41|shared|all" >&2; exit 2 ;;
esac

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
ENTRIES="$WORK/entries.tsv"
ERRORS="$WORK/errors.txt"

# parse_header <file>: sets H_TIER H_BOOT H_PACK H_BOOTS H_NEEDS, or H_ERR
parse_header() {
    local f="$1" l1="" l2="" l3="" cm="#" hdr more prefix kv tok key val
    H_TIER= H_BOOT= H_PACK= H_BOOTS= H_NEEDS= H_ERR=
    { IFS= read -r l1; IFS= read -r l2; IFS= read -r l3; } < "$f" || true
    case "$f" in *.cjs | *.mjs | *.js) cm="//" ;; esac
    case "$l1" in
        '#!'*) hdr="$l2"; more="$l3" ;;
        *) hdr="$l1"; more="$l2" ;;
    esac
    prefix="$cm sushi-test: "
    case "$hdr" in
        "$prefix"*) ;;
        *) H_ERR="no '$prefix...' header line right after the shebang"; return ;;
    esac
    set -f
    kv="${hdr#"$prefix"}"
    for tok in $kv; do
        case "$tok" in *=?*) ;; *) H_ERR="header token '$tok' is not key=value"; set +f; return ;; esac
        key="${tok%%=*}"
        val="${tok#*=}"
        case "$key" in
            tier) [ -z "$H_TIER" ] || H_ERR="tier given twice"; H_TIER="$val" ;;
            boot) [ -z "$H_BOOT" ] || H_ERR="boot given twice"; H_BOOT="$val" ;;
            pack) [ -z "$H_PACK" ] || H_ERR="pack given twice"; H_PACK="$val" ;;
            boots) [ -z "$H_BOOTS" ] || H_ERR="boots given twice"; H_BOOTS="$val" ;;
            needs) H_NEEDS="$val" ;;
            *) H_ERR="unknown header key '$key'" ;;
        esac
    done
    case "$more" in
        "$prefix"*)
            for tok in ${more#"$prefix"}; do
                case "$tok" in needs=?*) H_NEEDS="${H_NEEDS:+$H_NEEDS,}${tok#needs=}" ;; *) H_ERR="a second header line may only hold needs=" ;; esac
            done
            ;;
    esac
    set +f
    [ -z "$H_ERR" ] || return
    case "$H_TIER" in
        hermetic | host | pack | live | bench | tool | lib) ;;
        *) H_ERR="tier '${H_TIER:-}' is not hermetic|host|pack|live|bench|tool|lib"; return ;;
    esac
    if [ "$H_TIER" = live ]; then
        case "$H_BOOT" in
            api | exact | ram | knobs | own) ;;
            *) H_ERR="a live script needs boot=api|exact|ram|knobs|own"; return ;;
        esac
    elif [ -n "$H_BOOT" ]; then
        H_ERR="boot= is for tier=live only"
        return
    fi
    if [ -n "$H_PACK" ] && [ "$H_BOOT" != own ] && [ "$H_TIER" != pack ]; then
        H_ERR="pack= is for boot=own and tier=pack only"
        return
    fi
    if [ -n "$H_BOOTS" ]; then
        case "$H_BOOTS" in
            *[!0-9]* | 0*) H_ERR="boots=$H_BOOTS is not a positive integer"; return ;;
        esac
        [ "$H_BOOT" = own ] || { H_ERR="boots= is for boot=own only"; return; }
    fi
    case "$H_NEEDS" in *[!A-Za-z0-9_.,-]*) H_ERR="needs= takes a comma list of command names" ;; esac
}

# scan: fills ENTRIES (rel folder tier boot pack boots needs, tab separated) and ERRORS
scan() {
    local f rel folder
    : > "$ENTRIES"
    : > "$ERRORS"
    find "$TESTS_DIR" \( -name fixtures -o -name __pycache__ -o -name node_modules -o -name '.*' \) -prune -o \
        -type f \( -name '*.sh' -o -name '*.py' -o -name '*.cjs' -o -name '*.mjs' \) -print | sort > "$WORK/files.txt"
    while IFS= read -r f; do
        rel="${f#"$TESTS_DIR"/}"
        folder=
        case "$rel" in */*) folder="${rel%%/*}" ;; esac
        case "$folder" in
            "" | qwen4 | mimo2 | glm5 | dsv41 | lib) ;;
            *) echo "$rel: folder '$folder' is not qwen4|mimo2|glm5|dsv41|lib" >> "$ERRORS"; continue ;;
        esac
        parse_header "$f"
        if [ -n "$H_ERR" ]; then
            echo "$rel: $H_ERR" >> "$ERRORS"
            continue
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$rel" "${folder:--}" "$H_TIER" "${H_BOOT:--}" "${H_PACK:--}" "${H_BOOTS:--}" "${H_NEEDS:--}" >> "$ENTRIES"
    done < "$WORK/files.txt"
}

# check_matrix: every runnable script has a row in tests/CLAUDE.md, every row names a file that exists
check_matrix() {
    local matrix="$TESTS_DIR/CLAUDE.md" rel p
    if [ ! -f "$matrix" ]; then echo "CLAUDE.md: the matrix file is missing" >> "$ERRORS"; return; fi
    grep '^|' "$matrix" | cut -d'|' -f2 | grep -oE 'tests/[A-Za-z0-9_./-]+\.(sh|py|cjs|mjs)' | sed 's#^tests/##' | grep -v '^fixtures/' | sort -u > "$WORK/rows.txt"
    while IFS=$'\t' read -r rel _ tier _; do
        case "$tier" in hermetic | host | pack | live) ;; *) continue ;; esac
        grep -qxF "$rel" "$WORK/rows.txt" || echo "$rel: no row in tests/CLAUDE.md" >> "$ERRORS"
    done < "$ENTRIES"
    while IFS= read -r p; do
        [ -e "$TESTS_DIR/$p" ] || echo "$p: a tests/CLAUDE.md row names a file that is gone" >> "$ERRORS"
    done < "$WORK/rows.txt"
}

# check_attach: an attach script (boot=api|exact|ram|knobs) never starts a server
check_attach() {
    local rel boot
    while IFS=$'\t' read -r rel _ tier boot _; do
        [ "$tier" = live ] || continue
        case "$boot" in api | exact | ram | knobs) ;; *) continue ;; esac
        if grep -v '^[[:space:]]*#' "$TESTS_DIR/$rel" | grep -Eq 'zig-out/bin/sushi|\$\{?(SUSHI_BIN|SUSHI_BINARY|BINARY|BIN)\b|sushi_boot|taskpolicy|--serve([^a-z-]|$)'; then
            echo "$rel: boot=$boot is an attach script but launches sushi" >> "$ERRORS"
        fi
    done < "$ENTRIES"
}

# entries_of <pass>: this pass's runnable scripts in run order: phase, folder before root, name, sleep_inhibit last
entries_of() {
    awk -F'\t' -v OFS='\t' -v pass="$1" -v tiers=",$TIERS," '
    {
        rel = $1; folder = $2; tier = $3; boot = $4; pack = $5; boots = $6; needs = $7
        if (tier != "hermetic" && tier != "host" && tier != "pack" && tier != "live") next
        if (index(tiers, "," tier ",") == 0) next
        root = (folder == "-"); sel = 0
        if (pass != "shared") {
            if (folder == pass) sel = 1
            else if (root && tier == "live" && (boot == "api" || boot == "exact" || boot == "ram")) sel = 1
            else if (root && tier == "live" && boot == "own" && pack == "each") sel = 1
        } else {
            if (root && (tier == "hermetic" || tier == "host" || tier == "pack")) sel = 1
            else if (root && tier == "live" && boot == "knobs") sel = 1
            else if (root && tier == "live" && boot == "own" && pack != "each") sel = 1
        }
        if (!sel) next
        phase = (tier == "hermetic") ? 1 : (tier == "host") ? 2 : (tier == "pack") ? 3 : \
            (boot == "api") ? 4 : (boot == "exact") ? 5 : (boot == "ram") ? 6 : (boot == "knobs") ? 7 : 8
        key = rel; n = split(rel, parts, "/")
        if (parts[n] == "test_sleep_inhibit.sh") key = "~" rel
        print phase, (root ? 1 : 0), key, rel, tier, boot, pack, boots, needs
    }' "$ENTRIES" | sort -t"$(printf '\t')" -k1,1n -k2,2n -k3,3 | cut -f1,4-
}

apply_only() {
    local rel
    [ -n "$ONLY" ] || return 0
    : > "$WORK/entries.only"
    while IFS= read -r line; do
        rel="${line%%$'\t'*}"
        # shellcheck disable=SC2254
        case "$rel" in $ONLY | */$ONLY) printf '%s\n' "$line" >> "$WORK/entries.only" ;; esac
    done < "$ENTRIES"
    cp "$WORK/entries.only" "$ENTRIES"
}

pass_model() { if [ "$1" = shared ]; then echo qwen4; else echo "$1"; fi; }
phase_profile() { case "$1" in 4) echo api ;; 5) echo exact ;; 6) echo ram ;; 7) echo knobs ;; esac; }

print_plan() {
    local pass shared own phase rel tier boot pack boots needs last
    for pass in $PASSES; do
        entries_of "$pass" > "$WORK/pass.$pass"
        [ -s "$WORK/pass.$pass" ] || continue
        echo "pass $pass"
        shared=0
        own=0
        last=
        while IFS=$'\t' read -r phase rel tier boot pack boots needs; do
            case "$phase" in
                1 | 2 | 3)
                    [ "$needs" != - ] || needs=
                    echo "  $tier $rel${needs:+ needs=$needs}"
                    ;;
                4 | 5 | 6 | 7)
                    if [ "$last" != "$phase" ]; then
                        echo "  boot $(phase_profile "$phase"): $(sushi_profile_flags "$(phase_profile "$phase")" "$(pass_model "$pass")")"
                        shared=$((shared + 1))
                        last="$phase"
                    fi
                    echo "    $rel"
                    ;;
                8)
                    [ "$boots" != - ] || boots=1
                    [ "$pack" != - ] || pack=default
                    echo "  own $rel pack=$pack boots=$boots"
                    own=$((own + boots))
                    ;;
            esac
        done < "$WORK/pass.$pass"
        echo "  boots: $shared shared + $own own"
    done
}

# ---- run mode ----

RESULTS=
FAILED=0
HALT_NOTE=0

record() { # status pass rel secs
    printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$RESULTS"
    echo "  $1 $3${4:+ (${4}s)}"
    case "$1" in FAIL | CRASH) FAILED=1 ;; esac
}

halted() { [ "$FAILED" = 1 ] && [ "$KEEP_GOING" = 0 ]; }

# run_one <pass> <rel> <tier> <needs>: PASS, SKIP (exit 3) or FAIL; a failure shows the log tail
run_one() {
    local pass="$1" rel="$2" tier="$3" needs="$4" f="$TESTS_DIR/$2" log t0 rc interp n owner
    log="$OUT/logs/$pass-$(echo "$rel" | tr / _).log"
    if [ "$needs" != - ]; then
        for n in $(echo "$needs" | tr , ' '); do
            command -v "$n" > /dev/null || { echo "SKIP $rel: needs $n" > "$log"; record SKIP "$pass" "$rel" 0; echo "    needs $n"; return 3; }
        done
    fi
    case "$rel" in
        *.sh) interp=/bin/bash ;;
        *.py) interp=python3 ;;
        *) interp=node ;;
    esac
    case "$(head -1 "$f")" in *bun*) interp=bun ;; esac
    owner="${GPU_LOCK_OWNER:-tests}-$(basename "$rel")"
    [ "$tier" != pack ] || "$ROOT/scripts/gpu-lock.sh" acquire "$owner"
    t0=$(date +%s)
    (cd "$ROOT" && SUSHI_PASS_MODEL="$(pass_model "$pass")" $interp "$f" < /dev/null > "$log" 2>&1)
    rc=$?
    [ "$tier" != pack ] || "$ROOT/scripts/gpu-lock.sh" release "$owner"
    t0=$(($(date +%s) - t0))
    case "$rc" in
        0) record PASS "$pass" "$rel" "$t0" ;;
        3) record SKIP "$pass" "$rel" "$t0"; grep -m1 SKIP "$log" | sed 's/^/    /' ;;
        *) record FAIL "$pass" "$rel" "$t0"; tail -20 "$log" | sed 's/^/    | /' ;;
    esac
    return "$rc"
}

# run_boot_group <pass> <profile> <listfile>: one shared boot, its scripts, a reboot once after a crash
run_boot_group() {
    local pass="$1" profile="$2" list="$3" model pack rc flags rel tier needs t0 rebooted=0
    model=$(pass_model "$pass")
    pack=$(sushi_pack "$model") || {
        rc=$?
        while IFS=$'\t' read -r _ rel _; do record SKIP "$pass" "$rel" 0; done < "$list"
        return "$rc"
    }
    flags=$(sushi_profile_flags "$profile" "$model")
    t0=$(date +%s)
    echo " boot $pass-$profile"
    export SUSHI_PROFILE="$profile"
    # shellcheck disable=SC2086
    if ! sushi_boot "$pass-$profile" "$pack" $flags; then
        while IFS=$'\t' read -r _ rel _; do record FAIL "$pass" "$rel" 0; done < "$list"
        echo "    the boot failed; scripts not run"
        return 1
    fi
    while IFS=$'\t' read -r phase rel tier _ _ _ needs <&3; do
        run_one "$pass" "$rel" "$tier" "$needs"
        if ! sushi_alive; then
            awk -F'\t' -v OFS='\t' '{ a[NR] = $0 } END { for (i = 1; i < NR; i++) print a[i]; $0 = a[NR]; $1 = "CRASH"; print }' "$RESULTS" > "$RESULTS.new" && mv "$RESULTS.new" "$RESULTS"
            echo "    server died during $rel"
            sushi_stop 1
            FAILED=1
            [ "$rebooted" = 0 ] || break
            rebooted=1
            # shellcheck disable=SC2086
            sushi_boot "$pass-$profile" "$pack" $flags || break
        fi
    done 3< "$list"
    sushi_stop 0
    printf '%s\t%s\t%s\n' "$pass" "$profile" "$(($(date +%s) - t0))" >> "$OUT/boots.tsv"
}

run_pass() {
    local pass="$1" phase rel tier boot pack boots needs
    entries_of "$pass" > "$WORK/pass.$pass"
    [ -s "$WORK/pass.$pass" ] || return 0
    echo "pass $pass"
    for phase in 1 2 3; do
        awk -F'\t' -v p="$phase" '$1 == p' "$WORK/pass.$pass" > "$WORK/phase.list"
        [ -s "$WORK/phase.list" ] || continue
        halted && { HALT_NOTE=1; return; }
        while IFS=$'\t' read -r _ rel tier _ _ _ needs <&3; do run_one "$pass" "$rel" "$tier" "$needs"; done 3< "$WORK/phase.list"
    done
    for phase in 4 5 6 7; do
        awk -F'\t' -v p="$phase" '$1 == p' "$WORK/pass.$pass" > "$WORK/phase.list"
        [ -s "$WORK/phase.list" ] || continue
        halted && { HALT_NOTE=1; return; }
        run_boot_group "$pass" "$(phase_profile "$phase")" "$WORK/phase.list"
    done
    awk -F'\t' '$1 == 8' "$WORK/pass.$pass" > "$WORK/phase.list"
    if [ -s "$WORK/phase.list" ]; then
        halted && { HALT_NOTE=1; return; }
        while IFS=$'\t' read -r _ rel tier boot pack boots needs <&3; do run_one "$pass" "$rel" "$tier" "$needs"; done 3< "$WORK/phase.list"
    fi
}

finish() {
    local rc=$?
    trap - EXIT
    if type sushi_stop > /dev/null 2>&1; then sushi_stop "$rc" || { [ "$rc" -ne 0 ] || rc=1; }; fi
    if [ -n "$RESULTS" ] && [ -d "$OUT" ]; then
        {
            echo "commit: $(git -C "$ROOT" rev-parse --short HEAD 2> /dev/null || echo unknown)   binary: $SUSHI_BIN $(stat -f %Sm "$SUSHI_BIN" 2> /dev/null || echo missing)"
            echo "args: tier=$TIERS model=$MODELS only=${ONLY:-all}"
            cat "$RESULTS"
            [ ! -s "$OUT/boots.tsv" ] || { echo "boots (pass profile seconds):"; sed 's/^/  /' "$OUT/boots.tsv"; }
            [ "$HALT_NOTE" = 0 ] || echo "stopped after a failure; later boots and scripts did not run"
            echo "totals: $(for s in PASS FAIL SKIP CRASH; do printf '%s=%s ' "$s" "$(grep -c "^$s	" "$RESULTS")"; done) wall=$(($(date +%s) - START))s"
        } > "$OUT/summary.txt"
        cat "$OUT/summary.txt"
        echo "exit=$rc" > "$OUT/END"
    fi
    rm -rf "$WORK"
    exit "$rc"
}

# ---- main ----

scan
if [ "$MODE" = check ]; then
    check_attach
    [ "$NO_MATRIX" = 1 ] || check_matrix
    if [ -s "$ERRORS" ]; then
        sort -u "$ERRORS" | sed 's/^/FAIL /'
        echo "run.sh --check: $(sort -u "$ERRORS" | wc -l | tr -d ' ') problem(s)"
        exit 1
    fi
    echo "run.sh --check: $(wc -l < "$ENTRIES" | tr -d ' ') scripts with headers, no problems"
    exit 0
fi
if [ -s "$ERRORS" ]; then
    echo "run.sh: scripts without a valid header (run --check):" >&2
    sort -u "$ERRORS" | sed 's/^/  /' >&2
    exit 1
fi
apply_only
if [ "$MODE" = list ]; then
    print_plan
    exit 0
fi

OUT="${OUT:-$HOME/.sushi/runs/tests-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT/logs"
RESULTS="$OUT/results.tsv"
: > "$RESULTS"
: > "$OUT/boots.tsv"
START=$(date +%s)
export SUSHI_RUN_DIR="$OUT"
. "$SELF_DIR/lib/boot.sh"
SUSHI_TRAP_ARMED=1
trap finish EXIT

if [ "$NO_BUILD" = 0 ] && echo ",$TIERS," | grep -Eq ',(host|pack|live),'; then
    echo "build: ReleaseFast"
    (cd "$ROOT" && taskpolicy -a .zig-toolchain/zig build -Doptimize=ReleaseFast) || { echo "run.sh: the build failed" >&2; exit 1; }
fi

for pass in $PASSES; do
    run_pass "$pass"
    halted && break
done
[ "$FAILED" = 0 ]
