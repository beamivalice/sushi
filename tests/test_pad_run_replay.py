#!/usr/bin/env python3
# sushi-test: tier=live boot=api
"""Replay a supplied nondegenerate request in warm concurrent waves; never execute tools."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent / 'lib'))
import sushi_test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', default=os.environ.get('SUSHI_PAD_REPLAY_REQUEST'))
    parser.add_argument('--output', default=os.environ.get('SUSHI_RUN_DIR'))
    args = parser.parse_args()
    if not args.request:
        sushi_test.skip('set SUSHI_PAD_REPLAY_REQUEST to a local captured request')
    raw = json.loads(Path(args.request).read_text())
    body = raw.get('request', raw)
    body = dict(body, stream=True, stream_options={'include_usage': True}, enable_mtp=True)
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('SUSHI_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['SUSHI_API_KEY']
    sushi_test.attach()
    with urlopen(Request(sushi_test.url() + '/props', headers=headers), timeout=30) as response:
        if not json.load(response)['batching']['supported']:
            sushi_test.skip('the boot does not support batching')
    output = Path(args.output) if args.output else Path(tempfile.mkdtemp(prefix='sushi-pad-replay-'))
    output.mkdir(parents=True, exist_ok=True)

    def replay(barrier=None):
        if barrier:
            barrier.wait(timeout=30)
        req = Request(sushi_test.url() + '/v1/chat/completions',
                      data=json.dumps(body).encode(), headers=headers)
        done, finish, usage, visible, tools = False, None, {}, 0, 0
        with urlopen(req, timeout=900) as response:
            for line in response:
                if not line.startswith(b'data:'):
                    continue
                data = line[5:].strip()
                if data == b'[DONE]':
                    done = True
                    break
                event = json.loads(data)
                assert 'error' not in event, event.get('error')
                if event.get('usage'):
                    usage = event['usage']
                for choice in event.get('choices', []):
                    delta = choice.get('delta', {})
                    visible += len(delta.get('content') or '')
                    tools += len(delta.get('tool_calls') or [])
                    finish = choice.get('finish_reason') or finish
        assert done and finish, 'incomplete SSE response'
        return dict(finish=finish, visible_chars=visible, tool_fragments=tools, usage=usage)

    # Keep the captured prefix unchanged: a new nonce would remove the restore path.
    replay()
    sushi_test.attach()
    replies = []
    for _ in range(2):
        barrier = threading.Barrier(3)
        with ThreadPoolExecutor(max_workers=3) as pool:
            replies.extend(pool.map(lambda _: replay(barrier), range(3)))
    log = sushi_test.log_since()
    stops = [line for line in log.splitlines() if '[stop] reason=' in line]
    (output / 'pad-run-replay.json').write_text(json.dumps(dict(replies=replies, stops=stops), indent=2))
    assert not any('reason=pad-run' in line for line in stops), 'pad-run stop on replay fixture'
    if not any('path=batch' in line for line in stops):
        sushi_test.skip('the supplied request did not engage batched decoding')
    if not all(r['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0) > 0 for r in replies):
        sushi_test.skip('the supplied request did not restore a cached prefix')


if __name__ == '__main__':
    main()
