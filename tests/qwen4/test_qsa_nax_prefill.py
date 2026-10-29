#!/usr/bin/env python3
# sushi-test: tier=live boot=exact
"""A prefill past 8192 tokens through the NAX QSA gather: the passphrase is recalled and `[qsa-gather] engaged:
sushi_qsa_nax_precise` is logged. The default boot runs NAX; the prefix cache is off so no prompt is restored."""
import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'lib'))
import sushi_test as st

run_nonce = st.attach()
url = st.url()
model = 'sushi'
source = (Path(__file__).resolve().parents[2] / 'src/transformer.zig').read_text()
corpus = '\n'.join(source.splitlines()[100:900])
for nonce in ('81492017', '52839106', '90371648'):
    prompt = (f'Nonce {run_nonce}-{nonce}. Remember the passphrase MAGNOLIA-7731.\n'
              + (corpus * 4)[:40000]
              + '\nWhat was the passphrase? Answer with the passphrase only.')
    body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0, 'seed': 1234, 'max_tokens': 32,
            'enable_mtp': False, 'enable_pld': False, 'stream': False}
    req = urllib.request.Request(url + '/v1/chat/completions',
                                 data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=300) as response:
        result = json.load(response)
    usage = result['usage']
    assert usage['prompt_tokens'] > 8192, usage
    assert usage.get('prompt_tokens_details', {}).get('cached_tokens', 0) == 0, usage
    answer = result['choices'][0]['message']['content']
    assert 'MAGNOLIA-7731' in answer, result
expected = '[qsa-gather] engaged: sushi_qsa_nax_precise'
assert expected in st.log_since(), f'Missing engagement: {expected}'
