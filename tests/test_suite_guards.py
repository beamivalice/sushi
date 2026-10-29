#!/usr/bin/env python3
# sushi-test: tier=hermetic
"""Guard for tests/suite_guards.py: each rule fails on a planted violation in a scratch tree, and the repo
passes its own allowlists. Stdlib only; a passing run prints nothing.

    python3 tests/test_suite_guards.py
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "suite_guards.py")


def run(root, *args):
    p = subprocess.run([sys.executable, "-I", TOOL, "--root", root, *args], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def put(root, rel, text):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def expect(cond, what, out=""):
    if not cond:
        sys.exit(f"FAIL: {what}\n{out}")


def scratch(root):
    put(root, "src/engine.zig", 'pub fn dispatch() void {}\ntest "ring cache keeps order" {}\n')
    put(root, "src/glm5/glm5_model.zig", 'test "GLM router picks top k" {}\n')
    put(root, "src/mimo2/mimo_model.zig", 'test "mimo sinks" {}\n')
    expect(run(root, "--write")[0] == 0, "--write on a clean scratch tree")
    rc, out = run(root)
    expect(rc == 0 and out == "", "a clean scratch tree passes silently", out)


def arch_rules(root):
    engine = os.path.join(root, "src/engine.zig")
    with open(engine) as f:
        clean = f.read()
    with open(engine, "a") as f:
        f.write('test "GLM planted outside its folder" {}\n')
    rc, out = run(root)
    expect(rc == 1 and "GLM planted outside its folder" in out, "a model-named test in a shared file fails", out)
    with open(engine, "w") as f:
        f.write(clean)

    put(root, "src/mimo2/mimo_other.zig", 'test "qwen4 planted in another folder" {}\n')
    rc, out = run(root)
    expect(rc == 1 and "qwen4 planted in another folder" in out, "a test named for another model fails", out)
    os.remove(os.path.join(root, "src/mimo2/mimo_other.zig"))

    with open(engine, "a") as f:
        f.write('test "GLM planted outside its folder" {}\n')
    expect(run(root, "--write")[0] == 0, "--write absorbs a listed test")
    rc, out = run(root)
    expect(rc == 0, "a listed shared-at-arch-geometry test passes", out)
    with open(engine, "w") as f:
        f.write(clean)
    put(root, "src/engine2.zig", 'test "GLM planted outside its folder" {}\n')
    rc, out = run(root)
    expect(rc == 0, "the allowlist keys the name, so a move between shared files passes", out)
    os.remove(os.path.join(root, "src/engine2.zig"))
    expect(run(root, "--write")[0] == 0, "--write drops the entry")


def footprint_rules(root):
    engine = os.path.join(root, "src/engine.zig")
    with open(engine) as f:
        clean = f.read()
    with open(engine, "a") as f:
        f.write('// glm in a comment\ntest "ring" { _ = "mimo in a test body"; }\n')
    put(root, "src/glm5/glm5_more.zig", 'const a = "glm5 inside its own folder";\n')
    rc, out = run(root)
    expect(rc == 0 and out == "", "comments, test bodies and a model's own folder are not footprint", out)

    with open(engine, "a") as f:
        f.write('const planted = "glm5_next";\n')
    rc, out = run(root)
    expect(rc == 1 and "glm5" in out and "1 mention" in out and "allows 0" in out, "a new model mention in a shared file fails", out)

    expect(run(root, "--write")[0] == 0, "--write takes the new count")
    with open(engine, "w") as f:
        f.write(clean)
    rc, out = run(root)
    expect(rc == 0 and "lower glm5 to 0" in out, "a count below its entry passes and names the number to lower", out)
    with open(engine, "w") as f:
        f.write(clean)
    put(root, "src/exl3/glm_fold.zig", 'const a = "glm5";\n')
    rc, out = run(root)
    expect(rc == 0 and out == "", "a mention moved to another shared file keeps the model's count", out)
    put(root, "src/exl3/glm_fold.zig", 'const a = "glm5";\nconst b = "GLM";\n')
    rc, out = run(root)
    expect(rc == 1 and "2 mentions" in out and "allows 1" in out, "a second mention in any shared file fails", out)
    os.remove(os.path.join(root, "src/exl3/glm_fold.zig"))
    expect(run(root, "--write")[0] == 0, "--write lowers the entry")


def main():
    with tempfile.TemporaryDirectory() as root:
        scratch(root)
        arch_rules(root)
        footprint_rules(root)
    rc, out = run(os.path.dirname(HERE))
    expect(rc == 0, "the repo passes its own allowlists (run tests/suite_guards.py --write after a landing)", out)


main()
