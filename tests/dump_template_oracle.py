#!/usr/bin/env python3
# sushi-test: tier=tool
"""Reference renders of the chat-template corpus, by Python Jinja2 under transformers' chat-template rules.

`src/template_oracle_test.zig` captures every call `chat.formatChat` makes into the template engine over its
request corpus (template, messages JSON, tools JSON, extra-context JSON, add_generation_prompt) and compares
jinja.cpp's render of each with the render this script commits. The environment is transformers' own
(`_compile_jinja_template`: sandboxed, trim_blocks and lstrip_blocks on, loopcontrols, its `tojson`,
`raise_exception` and `strftime_now`); the context is built the way `lib/jinja_cpp/jinja_wrapper.cpp` builds it
(no `tools` key without tools, the extra context merged last).

Three steps, from the repo root:

    # 1. The packs' added tokens, which formatChat marks in message text (once per tokenizer change):
    python3 -I tests/dump_template_oracle.py tokens --models "${SUSHI_MODELS_DIR:-$HOME/.sushi/models}"
    # 2. The engine inputs of the corpus, from the real code path:
    SUSHI_TEMPLATE_ORACLE_DUMP=<abs path>/inputs.json \\
        zig build test-host -Doptimize=ReleaseFast -Dtest-filter="template oracle: the corpus"
    # 3. Python's renders, with a coverage check over every builtin and statement the templates use:
    uv run --python 3.14 --with transformers==5.22.2 --with jinja2==3.1.6 \\
        python3 -I tests/dump_template_oracle.py render --inputs <abs path>/inputs.json

Step 3 refuses to write the fixture when a filter, test, call, statement or operator a template uses is reached by
no case, and names it.
"""
import argparse
import functools
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKENS = ROOT / 'src/fixtures/chat_added_tokens.json'
ORACLE = ROOT / 'src/fixtures/template_oracle.json'
# What a template uses that no request formatChat can build reaches, and why. Anything else unreached fails.
UNREACHABLE = {
    'call endswith': 'Qwen tests a user turn for `<tool_response>` with startswith first, and formatChat marks that '
                     'token text in user text (U+FDD0 before it), so startswith never holds and endswith never runs',
    'call emit_audio': 'GLM renders an audio content part; formatChat sends only text, image and video parts',
    'call tool_references_to_response': 'GLM renders `tool_reference` content parts; formatChat sends none',
}
# The release pack of each served template whose tokenizer formatChat reads. MiMo-V2.6 has none on the release box.
PACKS = {
    'qwen38_flash_next': 'Qwen3.8-Flash-Next-Sushi-4bpw',
    'glm53': 'GLM-5.3-Flash-Sushi-2.4bpw',
    'dsv41': 'DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw',
}


def dump_tokens(models):
    out = {}
    for name, pack in PACKS.items():
        added = json.loads((Path(models) / pack / 'tokenizer.json').read_text())['added_tokens']
        out[name] = [t['content'] for t in sorted(added, key=lambda t: t['id'])]
    TOKENS.write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')


def context_of(case):
    ctx = {'messages': json.loads(case['messages']), 'add_generation_prompt': case['add_generation_prompt']}
    if case['tools']:
        ctx['tools'] = json.loads(case['tools'])
    ctx.update(json.loads(case['extra']))
    return ctx


def render(template, case):
    import jinja2
    try:
        return template.render(**context_of(case)), None
    except jinja2.exceptions.TemplateError as e:
        return None, f'{type(e).__name__}: {e}'
    except (TypeError, ValueError, KeyError, IndexError, AttributeError) as e:
        return None, f'{type(e).__name__}: {e}'


class Coverage:
    """What a template uses (from its tree) and what the corpus reached (counted calls, executed lines)."""

    def __init__(self, env_template, source):
        import jinja2
        from jinja2 import nodes
        env = env_template.environment.overlay()
        env.filters, env.tests = dict(env.filters), dict(env.tests)
        self.called = {}
        for table, kind in ((env.filters, 'filter'), (env.tests, 'test')):
            for name, fn in list(table.items()):
                table[name] = self._counting(kind, name, fn)
        base_call = env.call
        global_names = {id(v): k for k, v in env.globals.items()}

        def call(context, obj, *args, **kwargs):
            name = global_names.get(id(obj)) or (obj.name if isinstance(obj, jinja2.runtime.Macro) else obj.__name__)
            self.called[('call', name)] = self.called.get(('call', name), 0) + 1
            return base_call(context, obj, *args, **kwargs)
        env.call = call
        self.template = env.from_string(source)
        self.lines = set()
        self.uses = []
        for node in env.parse(source).find_all(nodes.Node):
            for kind, name in self._features(node, nodes):
                self.uses.append((kind, name, node.lineno))

    def _counting(self, kind, name, fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            self.called[(kind, name)] = self.called.get((kind, name), 0) + 1
            return fn(*args, **kwargs)
        return wrapped

    @staticmethod
    def _features(node, nodes):
        if isinstance(node, nodes.Filter):
            yield 'filter', node.name
        elif isinstance(node, nodes.Test):
            yield 'test', node.name
        elif isinstance(node, nodes.Call):
            yield 'call', node.node.attr if isinstance(node.node, nodes.Getattr) else getattr(node.node, 'name', '?')
        elif isinstance(node, nodes.Compare):
            for op in node.ops:
                yield 'operator', op.op
        elif isinstance(node, nodes.Getattr) and isinstance(node.node, nodes.Name) and node.node.name == 'loop':
            yield 'loop', node.attr
        elif isinstance(node, nodes.Getitem):
            yield 'operator', 'slice' if isinstance(node.arg, nodes.Slice) else 'getitem'
        elif isinstance(node, (nodes.BinExpr, nodes.UnaryExpr, nodes.Concat, nodes.CondExpr)):
            yield 'operator', type(node).__name__
        elif isinstance(node, nodes.Assign) and isinstance(node.target, nodes.NSRef):
            yield 'statement', 'set namespace attribute'
        elif isinstance(node, (nodes.For, nodes.If, nodes.Assign, nodes.AssignBlock, nodes.Macro, nodes.CallBlock,
                               nodes.FilterBlock, nodes.Break, nodes.Continue, nodes.With)):
            yield 'statement', type(node).__name__

    def run(self, case):
        def tracer(frame, event, arg):
            if frame.f_code.co_filename == '<template>':
                self.lines.add(self.template.get_corresponding_lineno(frame.f_lineno))
                return tracer
            return None
        sys.settrace(tracer)
        try:
            render(self.template, case)
        finally:
            sys.settrace(None)

    def reached(self, kind, name, line):
        if kind in ('filter', 'test', 'call'):
            return self.called.get((kind, name), 0) > 0
        return line in self.lines


def dump_render(inputs):
    import jinja2
    import transformers
    from transformers.utils.chat_template_utils import _compile_jinja_template

    doc = json.loads(Path(inputs).read_text())
    compiled = {tid: _compile_jinja_template(text) for tid, text in doc['templates'].items()}
    coverage = {tid: Coverage(compiled[tid], text) for tid, text in doc['templates'].items()}
    cases = []
    for case in doc['cases']:
        out, err = render(compiled[case['template']], case)
        coverage[case['template']].run(case)
        cases.append(dict(case, python=out, python_error=err))

    # A served template's adapted copy (`+...`) is the same template to a reader: its uses count together.
    unreached = {}
    for tid, cov in coverage.items():
        family = tid.split('+')[0]
        for kind, name, line in cov.uses:
            key = (family, kind, name)
            hit = cov.reached(kind, name, line)
            unreached[key] = unreached.get(key, True) and not hit
    by_feature = {}
    for (family, kind, name), missed in unreached.items():
        by_feature.setdefault(f'{kind} {name}', []).append(missed)
    never = {feature for feature, missed in by_feature.items() if all(missed)}
    for tid, cov in coverage.items():
        if '+' not in tid:
            for kind, name, line in sorted(set(cov.uses), key=lambda u: (u[2], u[0], u[1])):
                if not cov.reached(kind, name, line):
                    print(f'unreached site: {tid}:{line} {kind} {name}', file=sys.stderr)
    if never - UNREACHABLE.keys():
        sys.exit('no case reaches: ' + ', '.join(sorted(never - UNREACHABLE.keys())))
    if UNREACHABLE.keys() - never:
        sys.exit('listed as unreachable but reached: ' + ', '.join(sorted(UNREACHABLE.keys() - never)))

    ORACLE.write_text(json.dumps({
        'generator': f'tests/dump_template_oracle.py: transformers {transformers.__version__} chat-template '
                     f'environment, jinja2 {jinja2.__version__}, Python {platform.python_version()}',
        'templates': doc['templates'],
        'cases': cases,
    }, ensure_ascii=False, indent=0) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='step', required=True)
    sub.add_parser('tokens').add_argument('--models', required=True)
    sub.add_parser('render').add_argument('--inputs', required=True)
    args = parser.parse_args()
    dump_tokens(args.models) if args.step == 'tokens' else dump_render(args.inputs)
