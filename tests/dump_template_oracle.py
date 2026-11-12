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

The builtin table (`src/fixtures/template_builtins.json`): one or more small templates per filter, test, method,
global, loop field, operator and statement the engine supports, each rendered by the same environment over a
context of its own. `src/template/` is held to it name by name:

    uv run --python 3.14 --with transformers==5.22.2 --with jinja2==3.1.6 \\
        python3 -I tests/dump_template_oracle.py builtins
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
BUILTINS = ROOT / 'src/fixtures/template_builtins.json'
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


# (covers, template, context). `covers` is the name the Zig test requires a case for: `<kind> <name>`.
D = {'d': {'b': 2, 'a': 1, 'c': None}}
PEOPLE = {'people': [{'name': 'bo', 'age': 31, 'tags': ['x']}, {'name': 'Al', 'age': 25, 'tags': []},
                     {'name': 'cy', 'age': 31, 'tags': ['y', 'z']}]}
BUILTIN_CASES = [
    # filters
    ('filter abs', '{{ -3|abs }} {{ (-2.5)|abs }} {{ n|abs }}', {'n': -7}),
    ('filter capitalize', "{{ 'hELLO wORLD'|capitalize }}|{{ ''|capitalize }}", {}),
    ('filter default', "{{ missing|default('dflt') }} {{ ''|default('e', true) }} {{ ''|default('e') }} {{ none|default('n') }}", {}),
    ('filter first', "{{ [3, 1, 2]|first }} {{ 'abc'|first }} {{ d|first }}", D),
    ('filter first', "{{ []|first }}|", {}),
    ('filter float', "{{ '3.5'|float }} {{ 2|float }} {{ 'x'|float }} {{ 'x'|float(1.5) }} {{ ' 1e3 '|float }}", {}),
    ('filter format', "{{ '%s-%d-%r'|format('a', 5, 'q') }} {{ '%(k)s'|format(k='v') }} {{ '%.2f%%'|format(1.5) }}", {}),
    ('filter format', "{{ '%.2f|%.0f|%f'|format(2.675, 0.5, 123456789.125) }}", {}),
    ('filter format', "{{ '<b>%s</b>'|safe|format('<') }} {{ '<%s>'|format('<') }}", {}),
    ('filter indent', "{{ 'a\\nb\\n\\nc'|indent }}|{{ 'a\\nb\\n\\nc'|indent(2, true, true) }}|{{ 'a\\nb'|indent('> ') }}", {}),
    ('filter int', "{{ '42'|int }} {{ 3.9|int }} {{ 'x'|int }} {{ 'x'|int(7) }} {{ '0x1A'|int(base=16) }} {{ '2.5'|int }} {{ true|int }}", {}),
    ('filter int', "{{ '99999999999999999999'|int }}", {}),
    ('filter int', "{{ ' -0x1a '|int(0, 16) }} {{ '0b101'|int(base=0) }} {{ '0o17'|int(0, 8) }} {{ '1_000'|int }} {{ '_1'|int(5) }} {{ '1__0'|int(5) }} {{ '017'|int(5, 0) }} {{ 'z'|int(base=36) }}", {}),
    ('filter int', "{{ 'inf'|int }} {{ 'nan'|int(7) }} {{ '-inf'|int }} {{ '1e400'|int }} {{ ' 1e3 '|int }} {{ 'infinity'|int(3) }}", {}),
    ('filter int', "{{ (big * 10 - big * 10)|int(5) }}", {'big': 1e308}),
    ('filter int', "{{ (big * 10)|int(5) }}", {'big': 1e308}),
    ('filter items', "{% for k, v in d|items %}{{ k }}={{ v }};{% endfor %}", D),
    ('filter join', "{{ [1, 'a', none]|join(', ') }} {{ people|join('/', attribute='name') }} {{ 'abc'|join('.') }}", PEOPLE),
    ('filter last', "{{ [3, 1, 2]|last }} {{ 'abc'|last }}", {}),
    ('filter length', "{{ [1, 2]|length }} {{ 'héllo'|length }} {{ d|length }} {{ ''|length }}", D),
    ('filter list', "{{ 'ab'|list }} {{ d|list }} {{ (1, 2)|list }}", D),
    ('filter lower', "{{ 'MiXeD Case'|lower }}", {}),
    ('filter lower', "{{ 'ÉCOLE ÀÖ'|lower }}", {}),
    ('filter map', "{{ people|map(attribute='name')|join(',') }} {{ ['a', 'b']|map('upper')|list }} {{ people|map(attribute='missing', default='-')|list }}", PEOPLE),
    ('filter map', "{{ ['ab', 'c']|map('map', 'upper')|map('list')|list }} {{ ['a']|map('map', 'map', 'upper')|map('map', 'list')|map('list')|list }}", {}),
    ('filter max', "{{ [3, 9, 2]|max }} {{ ['b', 'A', 'c']|max }} {{ ['b', 'A', 'c']|max(case_sensitive=true) }} {{ people|max(attribute='age')|tojson }}", PEOPLE),
    ('filter min', "{{ [3, 9, 2]|min }} {{ ['b', 'A', 'c']|min }} {{ people|min(attribute='age')|tojson }}", PEOPLE),
    ('filter reject', "{{ [0, 1, 2, 3, none]|reject|list }} {{ [1, 2, 3, 4]|reject('odd')|list }} {{ [1, 2, 3]|reject('==', 2)|list }}", {}),
    ('filter rejectattr', "{{ people|rejectattr('tags')|map(attribute='name')|list }} {{ people|rejectattr('age', 'equalto', 31)|map(attribute='name')|list }}", PEOPLE),
    ('filter replace', "{{ 'aaa'|replace('a', 'b') }} {{ 'aaa'|replace('a', 'b', 2) }} {{ 'abc'|replace('', '-') }}", {}),
    ('filter replace', "{{ 'aaa'|replace('a', 'b', -9223372036854775807 - 1) }} {{ 'ab'|replace('', '-', -9223372036854775807 - 1) }}", {}),
    ('filter reverse', "{{ 'abc'|reverse }} {{ [1, 2, 3]|reverse|list }} {% for x in [1, 2]|reverse %}{{ x }}{% endfor %}", {}),
    ('filter safe', "{{ '<b>'|safe }} {{ ('<a>'|safe + '<i>') }}", {}),
    ('filter select', "{{ [0, 1, 2, '', 'x']|select|list }} {{ [1, 2, 3, 4]|select('even')|list }} {{ [1, 5, 9]|select('>', 4)|list }} {{ ['a', 1]|select('string')|list }}", {}),
    ('filter selectattr', "{{ people|selectattr('tags')|map(attribute='name')|list }} {{ people|selectattr('age', 'gt', 30)|map(attribute='name')|list }} {{ people|selectattr('name', 'in', ['Al', 'cy'])|map(attribute='age')|list }}", PEOPLE),
    ('filter sort', "{{ [3, 1, 2]|sort }} {{ ['b', 'A', 'c']|sort }} {{ ['b', 'A', 'c']|sort(case_sensitive=true) }} {{ [3, 1, 2]|sort(reverse=true) }} {{ people|sort(attribute='age,name')|map(attribute='name')|list }}", PEOPLE),
    ('filter sort', "{{ [none, none]|sort }} {{ [{'a': 1}, {'a': 1}]|sort }} {{ [[1], [1]]|sort }} {{ [none]|sort }}", {}),
    ('filter sort', "{{ [{'a': 1}, {'a': 2}]|sort }}", {}),
    ('filter string', "{{ 5|string }} {{ none|string }} {{ [1, 'a']|string }} {{ d|string }} {{ 1.0|string }}", D),
    ('filter string', r'''{{ ['tab\t', 'q\'', 'both\'"', '\x7f', '\u00ad', 'é', '\u0378', '\U0001F600'] }}''', {}),
    ('filter strip', "{{ '  x  '|strip }}|{{ 'xxaxx'|strip('x') }}", {}),
    ('filter sum', "{{ [1, 2, 3]|sum }} {{ [1.5, 2]|sum }} {{ people|sum(attribute='age') }} {{ [1, 2]|sum(start=10) }}", PEOPLE),
    ('filter title', "{{ 'hello wORLD-foo (bar) x_y'|title }}", {}),
    ('filter tojson', "{{ d|tojson }} {{ [1, 'é', none, true, 1.5]|tojson }} {{ 'a<b>&\\'c'|tojson }}", D),
    ('filter tojson', r'''{{ '\x7fé'|tojson(ensure_ascii=true) }} {{ '\x7f'|tojson }}''', {}),
    ('filter tojson', "{{ d|tojson(indent=2) }}|{{ d|tojson(sort_keys=true) }}|{{ x|tojson }}", dict(D, x={'n': [1, 2.0, 1e-05, 1e100, 9007199254740993, -0.0]})),
    ('filter tojson', "{{ x|tojson(ensure_ascii=true) }} {{ x|tojson(separators=(',', ':')) }}", {'x': {'k': 'é中😀\u0001'}}),
    ('filter trim', "{{ '  x \\n'|trim }}|{{ '--x--'|trim('-') }}", {}),
    ('filter truncate', "{{ 'hello world, how are you'|truncate(9) }}|{{ 'hello world, how are you'|truncate(9, true) }}|{{ 'hello world, how are you'|truncate(12, false, '..', 0) }}|{{ 'short'|truncate(10) }}", {}),
    ('filter truncate', "{{ 'abc'|truncate(9223372036854775807) }} {{ 'abcdefghij'|truncate(5, leeway=9223372036854775807) }}", {}),
    ('filter unique', "{{ [1, 2, 1, 3, 2]|unique|list }} {{ ['a', 'A', 'b']|unique|list }} {{ ['a', 'A']|unique(case_sensitive=true)|list }} {{ people|unique(attribute='age')|map(attribute='name')|list }}", PEOPLE),
    ('filter upper', "{{ 'MiXeD case'|upper }}", {}),
    ('filter upper', "{{ 'école àö ß'|upper }}", {}),
    ('filter wordcount', "{{ 'one two  three-four'|wordcount }} {{ ''|wordcount }} {{ 'a_b c'|wordcount }}", {}),
    ('filter wordcount', "{{ 'naïve café 中文 x'|wordcount }}", {}),
    ('filter wordcount', "{{ 'a → b'|wordcount }}", {}),
    # tests
    ('test boolean', "{{ true is boolean }} {{ 1 is boolean }}", {}),
    ('test callable', "{{ range is callable }} {{ 1 is callable }} {{ 'x'.upper is callable }}", {}),
    ('test defined', "{{ x is defined }} {{ missing is defined }} {{ d.a is defined }} {{ d.zz is defined }} {{ d.c is defined }}", dict(D, x=1)),
    ('test divisibleby', "{{ 9 is divisibleby 3 }} {{ 9 is divisibleby(4) }}", {}),
    ('test eq', "{{ 1 is eq 1 }} {{ 'a' is eq 'b' }}", {}),
    ('test equalto', "{{ 1 is equalto 1.0 }}", {}),
    ('test escaped', "{{ 'x'|safe is escaped }} {{ 'x' is escaped }}", {}),
    ('test even', "{{ 2 is even }} {{ 3 is even }}", {}),
    ('test false', "{{ false is false }} {{ 0 is false }}", {}),
    ('test filter', "{{ 'upper' is filter }} {{ 'nope' is filter }}", {}),
    ('test float', "{{ 1.0 is float }} {{ 1 is float }}", {}),
    ('test ge', "{{ 2 is ge 2 }} {{ 1 is ge 2 }}", {}),
    ('test gt', "{{ 3 is gt 2 }} {{ 2 is gt 2 }}", {}),
    ('test greaterthan', "{{ 3 is greaterthan 2 }}", {}),
    ('test in', "{{ 1 is in [1, 2] }} {{ 'a' is in 'bcd' }}", {}),
    ('test integer', "{{ 1 is integer }} {{ 1.0 is integer }} {{ true is integer }}", {}),
    ('test iterable', "{{ [] is iterable }} {{ 'x' is iterable }} {{ 1 is iterable }} {{ d is iterable }}", D),
    ('test lower', "{{ 'abc' is lower }} {{ 'aBc' is lower }}", {}),
    ('test lt', "{{ 1 is lt 2 }} {{ 2 is lt 1 }}", {}),
    ('test lessthan', "{{ 1 is lessthan 2 }}", {}),
    ('test mapping', "{{ d is mapping }} {{ [] is mapping }}", D),
    ('test ne', "{{ 1 is ne 2 }} {{ 1 is ne 1 }}", {}),
    ('test none', "{{ none is none }} {{ d.c is none }} {{ 0 is none }}", D),
    ('test number', "{{ 1 is number }} {{ 1.5 is number }} {{ '1' is number }}", {}),
    ('test odd', "{{ 3 is odd }} {{ 2 is odd }}", {}),
    ('test sameas', "{{ none is sameas none }} {{ true is sameas true }} {{ x is sameas x }} {{ [] is sameas [] }}", {'x': [1]}),
    ('test sameas', "{{ 'a' is sameas 'a' }} {{ 1000 is sameas 1000 }} {{ 1.5 is sameas 1.5 }}", {}),
    ('test sequence', "{{ [] is sequence }} {{ 'x' is sequence }} {{ 1 is sequence }} {{ d is sequence }}", D),
    ('test string', "{{ 'x' is string }} {{ 1 is string }}", {}),
    ('test test', "{{ 'odd' is test }} {{ 'nope' is test }}", {}),
    ('test true', "{{ true is true }} {{ 1 is true }}", {}),
    ('test undefined', "{{ missing is undefined }} {{ 1 is undefined }}", {}),
    ('test upper', "{{ 'ABC' is upper }} {{ 'AbC' is upper }}", {}),
    ('test not', "{{ 1 is not none }} {{ x is not defined }}", {}),
    # methods
    ('method upper', "{{ 'ab'.upper() }}", {}),
    ('method lower', "{{ 'AB'.lower() }}", {}),
    ('method title', "{{ 'hello wORLD x1y'.title() }}", {}),
    ('method title', "{{ 'xéy'.title() }}", {}),
    ('method capitalize', "{{ 'hELLO'.capitalize() }}", {}),
    ('method strip', "{{ '  x '.strip() }}|{{ 'xyx'.strip('x') }}", {}),
    ('method lstrip', "{{ '  x '.lstrip() }}|{{ 'xxy'.lstrip('x') }}", {}),
    ('method rstrip', "{{ '  x '.rstrip() }}|{{ 'yxx'.rstrip('x') }}", {}),
    ('method startswith', "{{ 'abc'.startswith('ab') }} {{ 'abc'.startswith(('x', 'a')) }} {{ 'abc'.startswith('b', 1) }}", {}),
    ('method startswith', "{{ ''.startswith('', 1) }} {{ 'ab'.startswith('', 2) }} {{ 'ab'.startswith('', 3) }} {{ 'ab'.startswith('', -9) }}", {}),
    ('method endswith', "{{ 'abc'.endswith('bc') }} {{ 'abc'.endswith(('x', 'y')) }} {{ 'abc'.endswith('b', 0, 2) }}", {}),
    ('method endswith', "{{ 'ab'.endswith('', 3) }} {{ 'ab'.endswith('', 2) }} {{ 'ab'.endswith('b', -1) }} {{ 'ab'.endswith('', 1, 0) }}", {}),
    ('method split', "{{ 'a,b,,c'.split(',') }} {{ ' a  b '.split() }} {{ 'a,b,c'.split(',', 1) }} {{ 'a b c'.split(maxsplit=1) }} {{ ' a  b  c '.split(None, 1) }}", {}),
    ('method split', "{{ 'a,b,c'.split(',', -9223372036854775807 - 1) }} {{ 'a b c'.split(None, -9223372036854775807 - 1) }}", {}),
    ('method rsplit', "{{ 'a,b,c'.rsplit(',', 1) }} {{ ' a  b c '.rsplit(None, 1) }}", {}),
    ('method rsplit', "{{ 'a,b,c'.rsplit(',', -9223372036854775807 - 1) }} {{ ' a b '.rsplit(None, -9223372036854775807 - 1) }}", {}),
    ('method replace', "{{ 'aaa'.replace('a', 'b', 1) }}", {}),
    ('method replace', "{{ 'aaa'.replace('a', 'b', -9223372036854775807 - 1) }} {{ 'ab'.replace('', '-', -9223372036854775807 - 1) }}", {}),
    ('method get', "{{ d.get('a') }} {{ d.get('zz') }} {{ d.get('zz', 9) }}", D),
    ('method keys', "{{ d.keys()|list }} {{ d.keys()|length }}", D),
    ('method values', "{{ d.values()|list }}", D),
    ('method items', "{% for k, v in d.items() %}{{ k }}:{{ v }} {% endfor %}{{ d.items()|list }}", D),
    ('method indent', "{{ 'a\\nb\\n\\nc\\n'.indent(2) }}|{{ 'a\\nb'.indent(2, true) }}|{{ 'a\\n\\nb'.indent('>', false, true) }}", {}),
    ('method append', "{% set xs = [1] %}{% set _ = xs.append(2) %}{{ xs|length }} {{ xs|join(',') }} {{ xs.append(3)|join(',') }}", {}),
    ('method append', "{% set xs = [1, 2] %}{% for x in xs %}{{ x }}{% if x < 3 %}{% set _ = xs.append(x + 2) %}{% endif %}{% endfor %}[{{ xs|join(',') }}]", {}),
    # globals
    ('global range', "{{ range(3)|list }} {{ range(1, 7, 2)|list }} {{ range(5, 0, -2)|list }} {% for i in range(2) %}{{ i }}{% endfor %}", {}),
    ('global range', "{{ range(-9223372036854775807 - 1, 9223372036854775807, 9223372036854775807)|list }} {{ range(9223372036854775807, -9223372036854775807 - 1, -9223372036854775807 - 1)|list }}", {}),
    ('global range', "{{ range(3) }} {{ range(1, 5, 2) }} {{ range(0) }} {{ range(5, 1, -1) }}", {}),
    ('global range', "{{ range(10)[::3] }} {{ range(10)[1:3] }} {{ range(5)[::-1] }} {{ range(0, 10, 3)[-1:] }} {{ range(4)[9:] }} {{ [range(2)] }}", {}),
    ('global namespace', "{% set ns = namespace(n=0, s='a') %}{% for i in range(3) %}{% set ns.n = ns.n + i %}{% endfor %}{{ ns.n }}{{ ns.s }}", {}),
    ('global raise_exception', "{{ raise_exception('stop here') }}", {}),
    ('global strftime_now', "{{ strftime_now('fixed text, no field') }}|{{ strftime_now('100%%') }}", {}),
    # loop fields
    ('loop index', "{% for x in 'abc' %}{{ loop.index }}{% endfor %}", {}),
    ('loop index0', "{% for x in 'abc' %}{{ loop.index0 }}{% endfor %}", {}),
    ('loop revindex', "{% for x in 'abc' %}{{ loop.revindex }}{% endfor %}", {}),
    ('loop revindex0', "{% for x in 'abc' %}{{ loop.revindex0 }}{% endfor %}", {}),
    ('loop first', "{% for x in 'abc' %}{{ loop.first }} {% endfor %}", {}),
    ('loop last', "{% for x in 'abc' %}{{ loop.last }} {% endfor %}", {}),
    ('loop length', "{% for x in 'abc' %}{{ loop.length }}{% endfor %}", {}),
    ('loop previtem', "{% for x in 'abc' %}[{{ loop.previtem }}]{% endfor %}", {}),
    ('loop nextitem', "{% for x in 'abc' %}[{{ loop.nextitem }}]{% endfor %}", {}),
    # operators
    ('operator in', "{{ 1 in [1, 2] }} {{ 'b' in 'abc' }} {{ 'a' in d }} {{ 'zz' in d }}", D),
    ('operator in', "{{ [1] in d }}", D),
    ('operator in', "{{ {'a': 1}.get([1]) }}", {}),
    ('operator in', "{{ {[1]: 2} }}", {}),
    ('operator in', "[{{ d[[1]] }}]", D),
    ('operator in', "{{ (1, [2]) in d }}", D),
    ('operator in', "{{ {namespace(): 1}|length }} {{ {(1, 'a'): 1}|length }} {{ {range(2): 1}|length }} {{ {none: 1, 1.5: 2, true: 3}|length }}", {}),
    ('operator in', "{{ {x.keys(): 1} }}", {'x': {'a': 1}}),
    ('operator in', "{{ {x.values(): 1}|length }}", {'x': {'a': 1}}),
    ('filter unique', "{{ [[1], [1]]|unique|list }}", {}),
    ('operator not in', "{{ 3 not in [1, 2] }} {{ 'a' not in d }}", D),
    ('operator ~', "{{ 'a' ~ 1 ~ none ~ [2] }}", {}),
    ('operator ==', "{{ 1 == 1.0 }} {{ 'a' == 'a' }} {{ [1] == [1] }} {{ none == false }}", {}),
    ('operator !=', "{{ 1 != 2 }} {{ d != d }}", D),
    ('operator <', "{{ 1 < 2 }} {{ 'a' < 'b' }} {{ [1, 2] < [1, 3] }}", {}),
    ('operator <=', "{{ 2 <= 2 }} {{ 1 < 2 <= 2 }}", {}),
    ('operator >', "{{ 2 > 1 }} {{ 1.5 > 2 }}", {}),
    ('operator >=', "{{ 2 >= 3 }}", {}),
    ('operator <', "{% set n = big * 10 - big * 10 %}{{ n < 1 }} {{ n > 1 }} {{ n <= n }} {{ n >= 1 }} {{ 1 < n }} "
                   "{{ [1, n] < [1, 2] }} {{ ([n]|sort)[0] }}", {'big': 1e308}),
    ('operator <', "{{ 9007199254740993 > 9007199254740992.0 }} {{ 9007199254740992.0 < 9007199254740993 }} "
                   "{{ 9223372036854775807 < 9223372036854775808.0 }} {{ -9223372036854775808 > -9223372036854775808.0 }} "
                   "{{ 2 > 1.5 }} {{ -2 < -1.5 }}", {}),
    ('filter sort', "{% set n = big * 10 - big * 10 %}{{ [n, 1, n]|sort }}", {'big': 1e308}),
    ('operator and', "{{ 1 and 'x' }} {{ 0 and 'x' }} {{ [] and 1 }}|", {}),
    ('operator or', "{{ 0 or 'x' }} {{ 1 or 'x' }} {{ none or [] }}", {}),
    ('operator not', "{{ not 0 }} {{ not 'x' }} {{ not none }}", {}),
    ('operator +', "{{ 1 + 2 }} {{ 1 + 0.5 }} {{ 'a' + 'b' }} {{ [1] + [2] }}", {}),
    ('operator -', "{{ 5 - 7 }} {{ 1.5 - 1 }}", {}),
    ('operator +', "{{ x }} {{ x + 1 }}", {'x': 18446744073709551616}),
    ('operator *', "{{ 3 * 4 }} {{ 'ab' * 3 }} {{ [0] * 2 }} {{ 1.5 * 2 }}", {}),
    ('operator /', "{{ 7 / 2 }} {{ 6 / 3 }} {{ 1 / 3 }}", {}),
    ('operator /', "{{ 0 / 9007199254740993 }} {{ 0 / -9007199254740993 }} {{ -0 / 9007199254740993 }} {{ -0 / -9007199254740993 }} "
                   "{{ (-9223372036854775807 - 1) / 9007199254740993 }} {{ (-9223372036854775807 - 1) / -9007199254740993 }} "
                   "{{ 0 / (-9223372036854775807 - 1) }} {{ (-9223372036854775807 - 1) / (-9223372036854775807 - 1) }} {{ 1 / 9223372036854775807 }}", {}),
    ('operator /', "{{ 18014398509481985 / 3 }} {{ 9007199254740993 / 1 }} {{ -9223372036854775807 / 7 }} {{ 9223372036854775807 / -3 }} "
                   "{{ 9007199254740993 / 9007199254740992 }} {{ 1 / 9223372036854775807 }}", {}),
    ('operator /', "{{ 7553705426322969113 / 929034 }} {{ -4024571214244900419 / 3 }} {{ 845035163197522276 / 721185 }} {{ 6737799718809409591 / 9075052138599801624 }} {{ 3803124323149175780 / 8866492710033342388 }} {{ 630775572003593558 / -250236853049 }} {{ 6624828931508799992 / 3 }} {{ 3960464187922662128 / 10 }} {{ 5176960997028201545 / 10 }} {{ 8167369317587610177 / 1348282716345341916 }} {{ -4170956898410604572 / 2080350270456257537 }} {{ 7680770348320607716 / 3 }} {{ -4923639403098242023 / 990257 }} {{ 1317653624395294099 / -705612325586 }} {{ 9071690235867802094 / 3 }} {{ 1479425574464320135 / -904039931603 }} {{ -6767389531502128954 / 7 }} {{ -2588341384320151708 / 278333 }} {{ 2467830691614586993 / 3 }} {{ 4449338550679644969 / 3 }} {{ 8941867459866163772 / 3 }} {{ 3118797910300725590 / 10 }} {{ 4817483597098638109 / -146154148479 }} {{ 7487016406521098487 / 256445 }} {{ 8154973488651719718 / 10 }} {{ -2123646268900458661 / 10 }} {{ 5884347578626357340 / -568963763138 }} {{ -3603404109804220084 / 10 }} {{ 319425140002548654 / 10 }} {{ 5681260045197463057 / 3 }} {{ -3312766790882648609 / 135277 }} {{ 4440462185195170651 / 3 }} {{ 2231394784865157940 / 6598030653182758576 }} {{ 6511301892412530921 / 3 }} {{ 2737248902196432997 / -691038876585 }} {{ -4994553524825823619 / 10 }} {{ -4172655392929477050 / -864140256871 }} {{ -1144348507175027107 / 3 }} {{ 3529398522193778747 / 3 }} {{ -1866377842881345303 / 7 }}", {}),
    ('operator <', "{{ 642753998708299439892463 < 6.427539987082994e+23 }}{{ 642753998708299439892463 == 6.427539987082994e+23 }} {{ 949466351188907686881201 < 9.494663511889074e+23 }}{{ 949466351188907686881201 == 9.494663511889074e+23 }} {{ -482915017715844009995815 < -4.829150177158441e+23 }}{{ -482915017715844009995815 == -4.829150177158441e+23 }} {{ -900847596202032899796327 < -9.008475962020329e+23 }}{{ -900847596202032899796327 == -9.008475962020329e+23 }} {{ -14158789038519150330827 < -1.4158789038519148e+22 }}{{ -14158789038519150330827 == -1.4158789038519148e+22 }} {{ -1000329225958895646043093 < -1.0003292259588954e+24 }}{{ -1000329225958895646043093 == -1.0003292259588954e+24 }} {{ -211064566021775290963957 < -2.1106456602177532e+23 }}{{ -211064566021775290963957 == -2.1106456602177532e+23 }} {{ -433026300841319576901397 < -4.330263008413195e+23 }}{{ -433026300841319576901397 == -4.330263008413195e+23 }} {{ 546215503529557539222417 < 5.4621550352955765e+23 }}{{ 546215503529557539222417 == 5.4621550352955765e+23 }} {{ -294293842617904827679533 < -2.9429384261790483e+23 }}{{ -294293842617904827679533 == -2.9429384261790483e+23 }} {{ 892822975724787589052807 < 8.928229757247877e+23 }}{{ 892822975724787589052807 == 8.928229757247877e+23 }} {{ -18025537850313306408188 < -1.802553785031331e+22 }}{{ -18025537850313306408188 == -1.802553785031331e+22 }} {{ -738538394959388516640948 < -7.385383949593884e+23 }}{{ -738538394959388516640948 == -7.385383949593884e+23 }} {{ -260127383319125153836337 < -2.601273833191251e+23 }}{{ -260127383319125153836337 == -2.601273833191251e+23 }} {{ 690141392371859763891065 < 6.901413923718598e+23 }}{{ 690141392371859763891065 == 6.901413923718598e+23 }} {{ -1003119745054903110374226 < -1.0031197450549028e+24 }}{{ -1003119745054903110374226 == -1.0031197450549028e+24 }} {{ 972821265147317145827744 < 9.728212651473174e+23 }}{{ 972821265147317145827744 == 9.728212651473174e+23 }} {{ 1010574352992422631008909 < 1.0105743529924224e+24 }}{{ 1010574352992422631008909 == 1.0105743529924224e+24 }} {{ -226984074321792254703469 < -2.2698407432179232e+23 }}{{ -226984074321792254703469 == -2.2698407432179232e+23 }} {{ 520151534780550678026784 < 5.201515347805508e+23 }}{{ 520151534780550678026784 == 5.201515347805508e+23 }} {{ -938706240211981996583205 < -9.38706240211982e+23 }}{{ -938706240211981996583205 == -9.38706240211982e+23 }} {{ 756363345633110549383049 < 7.563633456331107e+23 }}{{ 756363345633110549383049 == 7.563633456331107e+23 }} {{ -163982613949203472410755 < -1.639826139492035e+23 }}{{ -163982613949203472410755 == -1.639826139492035e+23 }} {{ 679329962532905644328485 < 6.793299625329055e+23 }}{{ 679329962532905644328485 == 6.793299625329055e+23 }} {{ -964161866125000661814369 < -9.641618661250003e+23 }}{{ -964161866125000661814369 == -9.641618661250003e+23 }} {{ -63284378059865588370791 < -6.3284378059865604e+22 }}{{ -63284378059865588370791 == -6.3284378059865604e+22 }} {{ 729452652869525919688514 < 7.294526528695259e+23 }}{{ 729452652869525919688514 == 7.294526528695259e+23 }} {{ 44382484143188898641580 < 4.438248414318889e+22 }}{{ 44382484143188898641580 == 4.438248414318889e+22 }} {{ 1111569193078051617541419 < 1.1115691930780516e+24 }}{{ 1111569193078051617541419 == 1.1115691930780516e+24 }} {{ 302058235990530834453403 < 3.0205823599053083e+23 }}{{ 302058235990530834453403 == 3.0205823599053083e+23 }}", {}),
    ('operator %', "{{ 7 % 3 }} {{ -7 % 3 }} {{ 7.5 % 2 }} {{ 'x=%s' % 5 }}", {}),
    ('operator %', "{{ '%.2f' % 2.675 }} {{ '%.0f' % 0.5 }} {{ '%.0f' % 1.5 }} {{ '%.0f' % 2.5 }} {{ '%.3f' % -0.0 }} {{ '%f' % 1e300 }} {{ '%.20f' % 0.1 }} {{ '%f' % 3 }} {{ '%.1f' % 0.05 }} {{ '%.2f' % 1.005 }} {{ '%f' % 5e-324 }}", {}),
    ('operator %', "{{ ('%.400f' % 1.0)|length }} {{ ('%.400f' % 0.1)[-5:] }} {{ '%f %f %f' % (1e308 * 10, -1e308 * 10, (1e308 * 10) - (1e308 * 10)) }} {{ '%.3f' % true }}", {}),
    ('operator %', "{{ '<b>%s</b>'|safe % '<' }} {{ '<b>%s %s</b>'|safe % ('<', '&') }} {{ '<%s>' % '<' }}", {}),
    ('operator <', "{{ 9223372036854775807 < 9223372036854775808 }} {{ 9223372036854775807 == 9223372036854775808 }} "
                   "{{ 18446744073709551616 < 18446744073709551617 }} {{ 18446744073709551617 > 18446744073709551616 }} "
                   "{{ 18446744073709551617 == 18446744073709551616.0 }} {{ 18446744073709551616 == 18446744073709551616.0 }} "
                   "{{ -18446744073709551617 < -18446744073709551616 }} {{ -18446744073709551617 > -18446744073709551616.0 }} "
                   "{{ 18446744073709551617 < 18446744073709555712.0 }} {{ 18446744073709551617 > 18446744073709551616.0 }} "
                   "{{ -9223372036854775809 < -9223372036854775808 }} {{ 1e300 > 99999999999999999999999 }} {{ -1e400 < -99999999999999999999 }}", {}),
    ('operator <', "{{ -9223372036854775808 == -9223372036854775807 - 1 }} {{ -9223372036854775808 + 1 }} "
                   "{{ x == -9223372036854775807 - 1 }} {{ -0x8000000000000000 + 1 }}", {'x': -9223372036854775808}),
    ('operator unary -', "{{ (-9223372036854775808)|abs }} {{ -(-9223372036854775808) }}", {}),
    ('filter sort', "{{ [18446744073709551617, 18446744073709551616.0, 9223372036854775807, -18446744073709551617]|sort }} "
                    "{{ [18446744073709551617, 18446744073709551616.0]|unique|list }} {{ [18446744073709551617, 18446744073709551616.0]|max }} "
                    "{{ {18446744073709551616: 'a'}[18446744073709551616.0] }}", {}),
    ('operator <', "{{ 99999999999999999999 < 1e20 }} {{ 100000000000000000001 > 1e20 }} {{ 100000000000000000001 == 1e20 }} "
                   "{{ 100000000000000000000 == 1e20 }} {{ -99999999999999999999 > -1e20 }} {{ 99999999999999999999 > 1.5 }}", {}),
    ('operator unary -', "{{ -x }} {{ -(1.5) }} {{ - -2 }}", {'x': 3}),
    ('operator unary +', "{{ +x }}", {'x': 3}),
    ('operator slice', "{{ 'abcdef'[1:4] }} {{ 'abcdef'[::-1] }} {{ [1, 2, 3, 4][-2:] }} {{ [1, 2, 3, 4][::2] }} {{ 'héllo'[1:3] }} {{ x[:-1] }}", {'x': [1, 2, 3]}),
    ('operator slice', "{{ 'abc'[1::9223372036854775807] }} {{ 'abc'[::-9223372036854775807 - 1] }} {{ [1, 2, 3][-9223372036854775807 - 1:9223372036854775807] }} {{ 'abc'[9223372036854775807::-1] }}", {}),
    ('operator slice', "{{ x[:10] }}", {'x': None}),
    ('operator slice', "{{ x[:1] }}", {'x': 5}),
    ('operator slice', "{{ x[:1] }}", {'x': True}),
    ('operator slice', "{{ x[:1] }}", {'x': {'a': 1}}),
    ('operator slice', "{{ x.items()[:1] }}", {'x': {'a': 1}}),
    ('operator slice', "{{ (x|map('upper'))[:1] }}", {'x': ['a']}),
    ('operator slice', "{% for i in [1] %}{{ loop[:1] }}{% endfor %}", {}),
    ('operator slice', "{{ 'abc'['a':] }}", {}),
    ('operator slice', "{{ 'abc'[1.0:] }}", {}),
    ('operator slice', "{{ [1, 2][:x] }}", {'x': [1]}),
    ('operator slice', "{{ 'abc'[true:] }} {{ 'abc'[-99999999999999999999:99999999999999999999] }} "
                       "{{ [1, 2, 3][none:none:-1] }} {{ range(5)[1:3]|list }} {{ 'abc'[99999999999999999999:] }}|", {}),
    ('operator subscript', "{{ x[0] }} {{ x[-1] }} {{ d['a'] }} {{ x.0 }} {{ d.b }} {{ y.1.0 }} {{ x[9] }}|", dict(D, x=[1, 2, 3], y=[0, [5]])),
    ('operator if else', "{{ 'y' if 1 else 'n' }} {{ 'y' if 0 else 'n' }} {{ 'y' if 0 }}|", {}),
    ('operator literals', "{{ [1, 'a', none, true, false, 1.5, -2] }} {{ {'k': 'v', 1: 2} }} {{ (1,) }} {{ (1, 2) }} {{ 1_000 }} {{ 1e3 }} {{ \"dq\" 'sq' }}", {}),
    ('operator literals', "{{ 0x0 }} {{ 0x00000000000000000000000001 }} {{ 0b0 }} {{ 0o0 }} {{ 0xFFFFFFFFFFFFFFFF }} {{ -0x8000000000000001 }}", {}),
    ('operator literals', "{{ 0xFFFFFFFFFFFFFFFFFF }} {{ 0o7777777777777777777777777 }} {{ 0b11111111111111111111111111111111111111111111111111111111111111111111 }} "
                          "{{ 0x7FFF_FFFF_FFFF_FFFF }} {{ 0x8000000000000000 }} {{ -0x8000000000000000 }}", {}),
    ('operator literals', "{{ '\\x+1' }}", {}),
    ('operator literals', "{{ '\\u+041' }}", {}),
    ('operator literals', "{{ '\\u0_41' }}", {}),
    ('operator literals', "{{ '\\U+0000041' }}", {}),
    ('operator literals', "{{ '\\x41\\u0042\\U00000043' }}", {}),
    # statements
    ('statement if', "{% if 0 %}a{% elif none %}b{% elif 'x' %}c{% else %}d{% endif %}", {}),
    ('statement for', "{% for x in [] %}a{% else %}empty{% endfor %} {% for x in [1, 2, 3, 4] if x is even %}{{ x }}{% endfor %} {% for a, b in [[1, 2], [3, 4]] %}{{ a + b }}{% endfor %}", {}),
    ('statement for', "{% set q = 0 %}{% for i in [] %}x{% else %}{% set q = 1 %}{{ q }}{% endfor %}{{ q }}{% for i in [] %}{% else %}{% set z = 1 %}{% endfor %}{{ z is defined }}{% for x in [1, 2] if x > 5 %}{% else %}[{{ x }}]{% endfor %}", {}),
    ('statement for', "{% for i in [1, 2] %}{% break %}{% else %}E{% endfor %}|{% for i in [1, 2] %}{% continue %}{% else %}E{% endfor %}|"
                      "{% for i in [1, 2] %}{% if i == 2 %}{% continue %}{% endif %}{{ i }}{% else %}E{% endfor %}|"
                      "{% for i in [1, 2] %}{{ i }}{% if i == 2 %}{% break %}{% endif %}{% else %}E{% endfor %}|"
                      "{% for i in [1, 2] %}{% if i == 1 %}{% continue %}{% endif %}{{ i }}{% else %}E{% endfor %}", {}),
    ('statement for', "{% set i = 7 %}{% for i in [1, 2] %}{% break %}{% else %}[{{ i }}{{ loop is defined }}]{% endfor %}", {}),
    ('statement set', "{% if 1 %}{% set q = 1 %}{% endif %}{{ q }}{% for i in [1] %}{% if 1 %}{% set w = 2 %}{% endif %}{{ w }}{% endfor %}[{{ w }}]", {}),
    ('statement break', "{% for x in range(9) %}{% if x == 3 %}{% break %}{% endif %}{{ x }}{% endfor %}", {}),
    ('statement continue', "{% for x in range(5) %}{% if x is odd %}{% continue %}{% endif %}{{ x }}{% endfor %}", {}),
    ('statement set', "{% set x = 1 %}{% for i in [1] %}{% set x = 2 %}{{ x }}{% endfor %}{{ x }}{% set a, b = 3, 4 %}{{ a }}{{ b }}", {}),
    ('statement set block', "{% set x %}  body {{ 1 }} {% endset %}[{{ x }}]{% set y | upper %}up{% endset %}{{ y }}", {}),
    ('statement macro', "{% macro m(a, b='B') %}<{{ a }}{{ b }}>{% endmacro %}{{ m(1) }}{{ m(1, 2) }}{{ m(b=3, a=4) }}", {}),
    ('statement macro', "{% set x = 3 %}{% macro m(x=x) %}[{{ x }}]{% endmacro %}{{ m() }}{% macro n(a, b=a) %}[{{ b }}]{% endmacro %}{{ n(1) }}", {}),
    ('statement macro', "{% macro m(a) %}{{ a }}{{ varargs }}{{ kwargs }}{% endmacro %}{{ m(1, 2, k=3) }}|{{ m(1) }}", {}),
    ('statement macro', "{% macro m(a) %}{{ a }}{% endmacro %}{{ m(1, 2) }}", {}),
    ('statement macro', "{% macro m(a) %}{{ a }}{% endmacro %}{{ m(1, k=2) }}", {}),
    ('statement macro', "{% macro m() %}{{ varargs|length }}{% endmacro %}{{ m(1, 2) }}", {}),
    ('statement call', "{% macro w(t) %}[{{ t }}:{{ caller() }}]{% endmacro %}{% call w('x') %}inner{% endcall %}", {}),
    ('statement call', "{% macro m() %}x{% endmacro %}{% call m() %}y{% endcall %}", {}),
    ('statement call', "{% macro m() %}{{ caller() if caller is defined }}{% endmacro %}{% call m() %}y{% endcall %}", {}),
    ('statement filter', "{% filter upper %}shout {{ 'this' }}{% endfilter %}", {}),
    ('statement set block', "{% for x in [1, 2, 3] %}{% set y %}{% if x == 2 %}{% break %}{% endif %}{{ x }}{% endset %}[{{ y }}]{% endfor %}", {}),
    ('statement set block', "{% for x in [1, 2, 3] %}{% set y %}{% if x == 2 %}{% continue %}{% endif %}{{ x }}{% endset %}[{{ y }}]{% endfor %}", {}),
    ('statement filter', "{% for x in [1, 2, 3] %}{% filter upper %}{% if x == 2 %}{% break %}{% endif %}a{{ x }}{% endfilter %}{% endfor %}", {}),
    ('statement filter', "{% for x in [1, 2, 3] %}{% filter upper %}a{% if x == 2 %}{% continue %}{% endif %}{{ x }}{% endfilter %}|{% endfor %}", {}),
    ('statement generation', "a{% generation %}b{{ 1 }}{% endgeneration %}c", {}),
    ('statement comment', "a{# note #}b", {}),
    ('statement whitespace', "  {% if 1 %}\n  x\n  {%- if 1 -%}  y  {%- endif %}\n{%+ if 1 %}z{% endif %}\n{% endif %}\n{{- ' w ' -}}\n.\n", {}),
    ('statement whitespace', "x\n　{% if 1 %}y{% endif %}|x\n\f{% if 1 %}y{% endif %}|x\n\v{% if 1 %}y{% endif %}|x\n {% if 1 %}y{% endif %}|x\n \t{% if 1 %}y{% endif %}|x\nz {% if 1 %}y{% endif %}", {}),
    ('statement whitespace', "a\r\n  {# c #}\r\nb {%+ if 1 +%} c\n{% endif %}\t{{ 'd' }}\n  {{ 'e' }}\n", {}),
    # undefined values and failures
    ('undefined prints', "[{{ missing }}][{{ d.zz }}][{{ x[5] }}]", dict(D, x=[1])),
    ('undefined fails', "{{ missing.attr }}", {}),
    ('undefined fails', "{{ missing + 1 }}", {}),
    ('undefined fails', "{{ 1 / 0 }}", {}),
]


def dump_builtins():
    import jinja2
    import transformers
    from transformers.utils.chat_template_utils import _compile_jinja_template

    cases, seen = [], {}
    for covers, text, ctx in BUILTIN_CASES:
        seen[covers] = seen.get(covers, 0) + 1
        extra = json.dumps(ctx, ensure_ascii=False)
        case = {'covers': covers, 'label': f'{covers} #{seen[covers]}', 'template': text, 'extra': extra}
        try:
            compiled = _compile_jinja_template(text)
        except jinja2.exceptions.TemplateError as e:
            cases.append(dict(case, python=None, python_error=f'{type(e).__name__}: {e}'))
            continue
        try:
            out, err = render(compiled, {'messages': '[]', 'add_generation_prompt': False, 'tools': None, 'extra': extra})
        except Exception as e:  # what a template's own arithmetic raises (ZeroDivisionError, OverflowError)
            out, err = None, f'{type(e).__name__}: {e}'
        cases.append(dict(case, python=out, python_error=err))
    BUILTINS.write_text(json.dumps({
        'generator': f'tests/dump_template_oracle.py builtins: transformers {transformers.__version__} chat-template '
                     f'environment, jinja2 {jinja2.__version__}, Python {platform.python_version()}',
        'cases': cases,
    }, ensure_ascii=False, indent=0) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='step', required=True)
    sub.add_parser('tokens').add_argument('--models', required=True)
    sub.add_parser('render').add_argument('--inputs', required=True)
    sub.add_parser('builtins')
    args = parser.parse_args()
    if args.step == 'tokens':
        dump_tokens(args.models)
    elif args.step == 'render':
        dump_render(args.inputs)
    else:
        dump_builtins()
