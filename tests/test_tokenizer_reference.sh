#!/bin/bash
# CPU-only real-pack parity. No server restart or model weights needed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_ROOT="${SUSHI_MODELS_DIR:-$HOME/.sushi/models}"
FIXTURE="$(mktemp -t sushi-tokenizer-parity)"
trap 'rm -f "$FIXTURE"' EXIT
python3 - "$ROOT" "$MODEL_ROOT" "$FIXTURE" <<'PY'
import sys,json,os
from pathlib import Path
from tokenizers import Tokenizer
root,model_root,fixture=map(Path,sys.argv[1:])
sys.path.insert(0,str(root/'tests'))
from dump_tokenizer_rule_fixtures import TEXTS, CODE_TEXTS
# Code and agent-history strings: the traffic where a wrong grammar (camelCase, `//!`, `iOS`) shows.
AGENT_TEXTS=["<|system|>\nYou have tools.\n<tools>\n{\"name\": \"getWeather\", \"parameters\": {\"cityName\": {\"type\": \"string\"}}}\n</tools><|user|>Weather in Paris?<|assistant|><think></think><tool_call>getWeather<arg_key>cityName</arg_key><arg_value>Paris</arg_value></tool_call><|observation|><tool_response>{\"tempC\": 21}</tool_response><|assistant|>It is 21 degrees.",
 "<|im_start|>user\nlistFiles?<|im_end|>\n<|im_start|>assistant\n<tool_call>\n{\"name\": \"listFiles\", \"arguments\": {\"dirPath\": \"/usr/local/bin\"}}\n</tool_call><|im_end|>"]
texts=list(TEXTS)+CODE_TEXTS+AGENT_TEXTS+[(root/f).read_text()[:40000] for f in ['src/server.zig','src/chat.zig','tests/test_tokenizer_reference.sh']]
base=len(texts)
if os.environ.get('TOKENIZER_CASES_JSON'):
    texts += [r['prompt'] for r in json.loads(Path(os.environ['TOKENIZER_CASES_JSON']).read_text())['cases']]
rows=[]
for name in ['GLM-5.3-Flash-Sushi-2.5bpw','Qwen3.8-Flash-Next-Sushi-4bpw','MiMo-V2.6-Flash-Sushi-2.3bpw','deepseek-ai/DeepSeek-V4.1-Flash-1017']:
    path=model_root/name
    if not (path/'tokenizer.json').exists():
        print(f'SKIP: {name} not on this box'); continue
    tok=Tokenizer.from_file(str(path/'tokenizer.json'))
    # Isolate Split/BPE parity from the separate, pre-existing NFC-normalizer gap.
    # Original benchmark prompts are already NFC; decomposed synthetic cases are normalized.
    normalized=[tok.normalizer.normalize_str(t) if tok.normalizer else t for t in texts]
    if os.environ.get('TOKENIZER_CASES_JSON'):
        assert normalized[base:] == texts[base:], 'Benchmark contains non-normalized prompts'
    rows.append({'model_dir':str(path.resolve()),'cases':[{'text':t,'ids':tok.encode(t,add_special_tokens=False).ids} for t in normalized]})
fixture.write_text(json.dumps(rows,ensure_ascii=False))
print(f'Checking {len(texts)} texts per model against Hugging Face tokenizers')
PY
cd "$ROOT"
SUSHI_TOKENIZER_PARITY_FIXTURE="$FIXTURE" "${ZIG:-$ROOT/.zig-toolchain/zig}" build test -Doptimize=ReleaseFast -Dtest-filter='real tokenizer reference parity'
