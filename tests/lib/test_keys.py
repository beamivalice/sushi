#!/usr/bin/env python3
# sushi-test: tier=lib
"""Run-step keys for tests/gate.sh --continue: a step's tests are reused from a prior gate only when everything they
can read is the same, and a test run then would have run the same code on the same inputs.

  test_keys.py keys <test-plan.json> <repo>
      one `<step>\t<key>` line per run step of the plan (`zig build test-plan`; its binaries sit beside it after
      `zig build test-build`); the key hashes the step's test binary
      (its code and data, read as `normalized` reads it), the step's own env, and what every step shares: the box
      (macOS build, chip), lib/mlx's files, the tracked files of <repo> that are neither Zig source nor docs, the
      environment's SUSHI_/MLX_/fixture variables and the contents of every fixture folder SUSHI_FIXTURE_ENVS names
  test_keys.py reuse <prior logs dir> <keys file> <reused log>
      the steps whose key equals the prior gate's and whose prior run ended whole (one DONE, no FAIL, its counts
      matching its lines), one a line; their prior log lines are written to <reused log>
  test_keys.py normalized <binary>
      the sha256 of what a Mach-O binary runs (any other file: its bytes): its code, data and dyld info, without its
      UUID, code signature and symbol tables, the bytes a rebuild of the same code changes without changing what runs
"""
import hashlib
import json
import os
import platform
import re
import struct
import subprocess
import sys

LC_SEGMENT_64, LC_SYMTAB, LC_DYSYMTAB, LC_UUID, LC_CODE_SIGNATURE = 0x19, 0x2, 0xB, 0x1B, 0x1D
LC_DYLD_INFO, LC_DYLD_INFO_ONLY = 0x22, 0x80000022
# linkedit_data_command: function starts, data in code, the exports trie, chained fixups.
LINKEDIT_DATA = (0x26, 0x29, 0x80000033, 0x80000034)
# S_ZEROFILL, S_GB_ZEROFILL, S_THREAD_LOCAL_ZEROFILL: no bytes in the file.
ZEROFILL = (0x1, 0xC, 0x12)
ENV_PREFIXES = ("SUSHI_", "MLX_", "QWEN", "DSV41", "GLM", "MIMO")
# Per-run paths and the gate's own switches, which no test reads.
ENV_SKIP = re.compile(r"^(SUSHI_TEST_LOG|SUSHI_GATE_.*)$")


def normalized(path):
    """What dyld maps and runs, hashed: the header, every load command but the UUID, the code signature and the
    symbol tables (with __LINKEDIT's sizes zeroed), every section's bytes and the dyld info each command points at.
    The symbol table holds the debug map, whose object paths and mtimes change on every build of the same code."""
    b = open(path, "rb").read()
    if len(b) < 32 or struct.unpack_from("<I", b, 0)[0] != 0xFEEDFACF:
        return hashlib.sha256(b).hexdigest()
    h = hashlib.sha256(b[:32])
    ncmds = struct.unpack_from("<I", b, 16)[0]
    off = 32
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<II", b, off)
        body = bytearray(b[off : off + size])
        if cmd in (LC_UUID, LC_CODE_SIGNATURE, LC_SYMTAB, LC_DYSYMTAB):
            off += size
            continue
        if cmd == LC_SEGMENT_64:
            if body[8:24].rstrip(b"\0") == b"__LINKEDIT":
                body[32:40] = bytes(8)  # vmsize
                body[48:56] = bytes(8)  # filesize
            h.update(body)
            nsects = struct.unpack_from("<I", body, 64)[0]
            for i in range(nsects):
                _, size_s, offset, _, _, _, flags = struct.unpack_from("<QQIIIII", body, 72 + 80 * i + 32)
                if flags & 0xFF not in ZEROFILL:
                    h.update(b[offset : offset + size_s])
        elif cmd in LINKEDIT_DATA:
            dataoff, datasize = struct.unpack_from("<II", body, 8)
            h.update(struct.pack("<I", cmd) + b[dataoff : dataoff + datasize])
        elif cmd in (LC_DYLD_INFO, LC_DYLD_INFO_ONLY):
            ranges = struct.unpack_from("<10I", body, 8)
            h.update(struct.pack("<I", cmd))
            for i in range(0, 10, 2):
                h.update(b[ranges[i] : ranges[i] + ranges[i + 1]])
        else:
            h.update(body)
        off += size
    return h.hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_sha(root):
    """Every regular file under root, by its relative path and contents."""
    h = hashlib.sha256()
    root = os.path.realpath(root)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames.sort()
        for name in sorted(filenames):
            p = os.path.join(dirpath, name)
            h.update(os.path.relpath(p, root).encode() + b"\0" + file_sha(p).encode() + b"\n")
    return h.hexdigest()


def shared(repo):
    h = hashlib.sha256()
    out = lambda *cmd: subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    h.update(f"macos {platform.mac_ver()[0]} {out('sw_vers', '-buildVersion').strip()}\n".encode())
    h.update(f"chip {out('sysctl', '-n', 'machdep.cpu.brand_string').strip()}\n".encode())
    h.update(f"mlx {tree_sha(os.path.join(repo, 'lib', 'mlx', 'lib'))}\n".encode())
    for line in out("git", "-C", repo, "ls-files", "-s").splitlines():
        path = line.split("\t", 1)[1]
        if path.endswith((".zig", ".md")) or path.startswith("docs/"):
            continue
        h.update(line.encode() + b"\n")
    for name in sorted(os.environ):
        if name.startswith(ENV_PREFIXES) and not ENV_SKIP.match(name):
            h.update(f"env {name}={os.environ[name]}\n".encode())
    for name in os.environ.get("SUSHI_FIXTURE_ENVS", "").split():
        h.update(f"fixture {name} {tree_sha(os.environ[name])}\n".encode())
    return h.hexdigest()


def keys(plan_path, repo):
    """The plan's binaries sit beside it (`tests/<binary>`) once `zig build test-build` has installed them."""
    plan = json.load(open(plan_path, encoding="utf-8"))
    out_dir = os.path.dirname(os.path.abspath(plan_path))
    base = shared(repo)
    sushi_metallib = os.path.join(out_dir, "lib", "sushi.metallib")
    if os.path.isfile(sushi_metallib):
        base = hashlib.sha256(f"{base}\nsushi.metallib {file_sha(sushi_metallib)}\n".encode()).hexdigest()
    for step in plan:
        h = hashlib.sha256()
        h.update(f"{base}\n{step['step']}\n".encode())
        h.update(normalized(os.path.join(out_dir, step["binary"])).encode())
        h.update(json.dumps(step["env"], sort_keys=True).encode())
        print(f"{step['step']}\t{h.hexdigest()}")


def reuse(prior_dir, keys_path, out_path):
    read_keys = lambda p: dict(line.rstrip("\n").split("\t", 1) for line in open(p, encoding="utf-8") if "\t" in line)
    prior, now = read_keys(os.path.join(prior_dir, "step-keys.tsv")), read_keys(keys_path)
    lines = {}
    with open(os.path.join(prior_dir, "tests.tsv"), encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 4)
            if len(parts) == 5:
                lines.setdefault(parts[1], []).append(line)
    reused = []
    for step, key in now.items():
        if prior.get(step) != key or step not in lines:
            continue
        rows = [l.rstrip("\n").split("\t", 4) for l in lines[step]]
        done = [r for r in rows if r[2] == "DONE"]
        counts = {s: sum(r[2] == s for r in rows) for s in ("PASS", "SKIP", "FAIL")}
        if len(done) != 1 or counts["FAIL"] or done[0][4] != f"{counts['PASS']} pass, {counts['SKIP']} skip, 0 fail":
            continue
        reused.append(step)
    with open(out_path, "w", encoding="utf-8") as out:
        for step in reused:
            out.writelines(lines[step])
    print("\n".join(reused))


def main(argv):
    if len(argv) == 3 and argv[0] == "keys":
        keys(argv[1], argv[2])
    elif len(argv) == 4 and argv[0] == "reuse":
        reuse(argv[1], argv[2], argv[3])
    elif len(argv) == 2 and argv[0] == "normalized":
        print(normalized(argv[1]))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
