#!/usr/bin/env python3
# sushi-test: tier=tool
"""The chat-template engine against Python Jinja2, case by case, under transformers' chat-template rules.

Each case (template, messages, extra context, tools, add_generation_prompt) is rendered by `src/template/` through
the `template differential` test and by transformers' `_compile_jinja_template`, and the two results are compared:
the same bytes, or both failing in the same class (a `raise_exception` with the same text). From the repo root:

    zig build test-build -Doptimize=ReleaseFast -Dtest-filter="template differential"
    uv run --python 3.14 --with transformers==5.22.2 --with jinja2==3.1.6 \\
        python3 -I tests/template_differential.py fuzz --n 5000 --seed 1
    uv run ... python3 -I tests/template_differential.py cases <file.jsonl>

`fuzz` makes structure-aware random templates (nested if/for/set/macro/call/filter/set-block, every whitespace
control, unicode whitespace, CR and CRLF); `cases` reads JSONL lines `{"t", "m", "x", "tools", "agp"}`. Each case
whose results differ is printed with both; the counts per class end the output. A case that crashes the engine is
reported as `CRASH` and the run resumes after it.
"""
import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / 'zig-out/tests/test-host'


def python_render(case):
    import jinja2
    from transformers.utils.chat_template_utils import _compile_jinja_template
    try:
        compiled = _compile_jinja_template(case['t'])
    except (jinja2.exceptions.TemplateError, SyntaxError) as e:  # Python's compile of the generated code raises SyntaxError
        return 'err', f'{type(e).__name__}: {e}'
    ctx = {'messages': json.loads(case['m']), 'add_generation_prompt': case['agp']}
    if case['tools']:
        ctx['tools'] = json.loads(case['tools'])
    ctx.update(json.loads(case['x']))
    try:
        return 'ok', compiled.render(**ctx)
    except RecursionError:
        return 'err', 'RecursionError'
    except Exception as e:  # what a template's own arithmetic or a builtin raises
        return 'err', f'{type(e).__name__}: {e}'


def engine_render(cases, timeout):
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / 'cases.jsonl'
        inp.write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in cases))
        start = 0
        while start < len(cases):
            env = {'SUSHI_TEMPLATE_DIFF': str(inp), 'SUSHI_TEMPLATE_DIFF_START': str(start)}
            try:
                rc = subprocess.run([str(BINARY)], env=env, capture_output=True, timeout=timeout, cwd=ROOT).returncode
            except subprocess.TimeoutExpired:
                rc = 'timeout'
            last = start - 1
            out = Path(str(inp) + '.out')
            if out.exists():
                for line in out.read_text().split('\n'):
                    if not line:
                        continue
                    r = json.loads(line)
                    results[r['i']] = r
                    last = max(last, r['i'])
            if rc == 0:
                break
            results[last + 1] = {'i': last + 1, 'ok': False, 'kind': f'CRASH {rc}', 'msg': ''}
            start = last + 2
    return results


def classify(z, py):
    if z is None:
        return 'NORESULT'
    if not z['ok'] and z['kind'].startswith('CRASH'):
        return 'CRASH'
    if z['ok'] and py[0] == 'ok':
        return 'same' if z['out'] == py[1] else 'DIFF'
    if z['ok']:
        return 'ENGINE_OK_PYTHON_ERR'
    if py[0] == 'ok':
        return 'ENGINE_ERR_PYTHON_OK'
    if py[1].startswith('TemplateError: '):
        return 'both_err' if z['kind'] == 'TemplateRaised' and z['msg'] == py[1][len('TemplateError: '):] else 'RAISE_MISMATCH'
    syntax = py[1].startswith(('TemplateSyntaxError', 'TemplateAssertionError', 'SyntaxError'))
    return 'both_err' if z['kind'] == ('TemplateSyntax' if syntax else 'TemplateRuntime') else 'KIND_MISMATCH'


def compare(cases, timeout):
    results = engine_render(cases, timeout)
    counts = {}
    for i, c in enumerate(cases):
        z, py = results.get(i), python_render(c)
        cat = classify(z, py)
        counts[cat] = counts.get(cat, 0) + 1
        if cat in ('same', 'both_err'):
            continue
        print(f'--- [{cat}] #{i}\n  T: {c["t"][:300]!r}')
        print('  PYTHON:', repr(py[1][:300]) if py[0] == 'ok' else 'ERR ' + py[1][:200])
        if z:
            print('  ENGINE:', repr(z['out'][:300]) if z['ok'] else f"ERR {z['kind']} {z.get('msg', '')[:200]}")
    print(counts)


class Fuzz:
    WS = ['', ' ', '  ', '\n', '\n\n', ' \n', '\n ', '\t', '\r\n', ' \t ', '\x0b', '\x0c', '\u3000', '\x85', '\r', '\x00']
    TXT = ['a', 'b c', 'x\ny', '<|im_start|>', '{', '}', '%', '#', '{ {', '% }', 'é', '😀', '-', '+', '{{', '}}', '{%',
           '%}', '{#', '#}', "'", '"', '\\', 'raw', '{ %', ' - ']
    ATOMS = ['1', '0', "'a'", "''", 'x', 'y', 'n', 's', 'l', 'd', 'u', 'none', 'true', 'false', '2.5', "' b '", "'é'",
             'messages|length', 'loop.index', 'i', 'ns.v', 'l[0]', 'd.a', 'd.b', 's[1:]', 'l|first']
    FILTERS = ['string', 'length', 'trim', 'upper', 'lower', 'tojson', 'default(1)', 'list', 'first', 'last', 'int',
               'float', 'join(",")', 'reverse', 'sort', 'title', 'capitalize', 'abs']
    TESTS = ['defined', 'none', 'string', 'number', 'odd', 'even', 'mapping', 'iterable', 'sequence', 'true', 'false',
             'undefined']
    CTX = {'x': 1, 'y': 'q', 'n': None, 's': 'héllo\nwörld', 'l': [1, 2, 3], 'd': {'a': 1, 'b': [1, 2]}, 'u': ' u '}

    def __init__(self, seed):
        self.r = random.Random(seed)

    def ws(self):
        return self.r.choice(self.WS)

    def txt(self):
        return ''.join(self.r.choice(self.TXT) if self.r.random() < 0.5 else self.ws() for _ in range(self.r.randint(0, 3)))

    def expr(self, depth=0):
        r, R = self.r.random(), self.r
        if depth > 2 or r < 0.3:
            return R.choice(self.ATOMS)
        if r < 0.45:
            return f'({self.expr(depth + 1)} {R.choice(["+", "-", "*", "~", "==", "!=", "<", "and", "or", "in", "%"])} {self.expr(depth + 1)})'
        if r < 0.6:
            return f'{self.expr(depth + 1)}|{R.choice(self.FILTERS)}'
        if r < 0.7:
            return f'({self.expr(depth + 1)} if {self.expr(depth + 1)} else {self.expr(depth + 1)})'
        if r < 0.78:
            return f'(not {self.expr(depth + 1)})'
        if r < 0.85:
            return f'{self.expr(depth + 1)} is {R.choice(self.TESTS)}'
        if r < 0.9:
            return f'[{self.expr(depth + 1)}, {self.expr(depth + 1)}]'
        if r < 0.95:
            return f"{{'k': {self.expr(depth + 1)}}}"
        return f'range({R.randint(0, 4)})'

    def tag(self, body):
        R = self.r
        return '{%' + R.choice(['', '-', '+']) + R.choice([' ', '']) + body + R.choice([' ', '']) + R.choice(['', '-']) + '%}'

    def block(self, depth):
        return ''.join(self.node(depth) for _ in range(self.r.randint(0, 4)))

    def node(self, depth):
        r, R, t = self.r.random(), self.r, self.tag
        if depth > 3 or r < 0.3:
            return self.txt()
        if r < 0.45:
            return '{{' + R.choice(['', '-']) + ' ' + self.expr() + ' ' + R.choice(['', '-']) + '}}'
        if r < 0.5:
            return '{#' + R.choice(['', '-']) + ' c ' + R.choice(['', '-']) + '#}'
        if r < 0.62:
            return (t(f'if {self.expr()}') + self.block(depth + 1)
                    + (t(f'elif {self.expr()}') + self.block(depth + 1) if R.random() < 0.3 else '')
                    + (t('else') + self.block(depth + 1) if R.random() < 0.4 else '') + t('endif'))
        if r < 0.74:
            over = R.choice(['range(3)', 'l', '[1,2]', 'd|items', 'messages', '[]', 's'])
            return (t(f'for i in {over}' + (' if i' if R.random() < 0.2 else '')) + self.block(depth + 1)
                    + (t('else') + self.block(depth + 1) if R.random() < 0.25 else '') + t('endfor'))
        if r < 0.8:
            return t(f'set v = {self.expr()}') + ('{{ v }}' if R.random() < 0.5 else '')
        if r < 0.84:
            return t(f'set ns.v = {self.expr()}')
        if r < 0.88:
            return t('set w') + self.block(depth + 1) + t('endset') + '{{ w }}'
        if r < 0.93:
            return t('macro m(a, b=1)') + self.block(depth + 1) + t('endmacro') + '{{ m(1) }}'
        if r < 0.96:
            return t(f'filter {R.choice(["upper", "trim", "indent(2)", "lower"])}') + self.block(depth + 1) + t('endfilter')
        return t('macro k()') + '{{ caller() }}' + t('endmacro') + t('call k()') + self.block(depth + 1) + t('endcall')

    def cases(self, n):
        out = []
        messages = json.dumps([{'role': 'user', 'content': 'u'}, {'role': 'user', 'content': 'v'}])
        for _ in range(n):
            t = '{% set ns = namespace(v=0) %}' * self.r.randint(0, 1) + self.block(0)
            if 'ns.v' in t and '{% set ns' not in t:
                t = '{% set ns = namespace(v=0) %}' + t
            out.append({'t': t, 'm': messages, 'x': json.dumps(self.CTX, ensure_ascii=False), 'tools': None, 'agp': False})
        return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--timeout', type=int, default=300, help='seconds the engine may take over the cases left')
    sub = parser.add_subparsers(dest='step', required=True)
    fuzz = sub.add_parser('fuzz')
    fuzz.add_argument('--n', type=int, default=5000)
    fuzz.add_argument('--seed', type=int, default=1)
    sub.add_parser('cases').add_argument('file')
    args = parser.parse_args()
    if not BINARY.exists():
        sys.exit(f'{BINARY} is missing: build it with the test-build step above')
    if args.step == 'fuzz':
        compare(Fuzz(args.seed).cases(args.n), args.timeout)
    else:
        lines = Path(args.file).read_text().split('\n')
        compare([dict({'x': '{}', 'tools': None, 'agp': False}, **json.loads(l)) for l in lines if l], args.timeout)
