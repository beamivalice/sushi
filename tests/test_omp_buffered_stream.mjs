#!/usr/bin/env bun
// sushi-test: tier=hermetic needs=node
// Run with bun and OMP_AI_MODULE pointing to omp's installed pi-ai package.
// The public client API receives mocked SSE; no model, tool or network runs.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const spec = process.env.OMP_AI_MODULE ?? process.argv[2] ?? '@oh-my-pi/pi-ai';
let entry;
if (fs.existsSync(spec)) {
  const resolved = path.resolve(spec);
  entry = fs.statSync(resolved).isDirectory()
    ? path.join(resolved, JSON.parse(fs.readFileSync(path.join(resolved, 'package.json'), 'utf8')).main)
    : resolved;
} else entry = Bun.resolveSync(spec, process.cwd());
// Reject accidental transport fallbacks, including during dependency loading.
globalThis.fetch = async () => { throw new Error('Real network is forbidden in this test'); };
// Isolate this proof before dependencies inspect environment defaults.
for (const key of ['PI_OPENAI_STREAM_IDLE_TIMEOUT_MS', 'PI_STREAM_IDLE_TIMEOUT_MS',
  'PI_OPENAI_STREAM_FIRST_EVENT_TIMEOUT_MS', 'PI_STREAM_FIRST_EVENT_TIMEOUT_MS']) delete process.env[key];
const ai = await import(pathToFileURL(entry).href);
const { buildModel } = await import(pathToFileURL(Bun.resolveSync('@oh-my-pi/pi-catalog/build', path.dirname(entry))).href);
let packageDir = path.dirname(entry);
while (!fs.existsSync(path.join(packageDir, 'package.json'))) {
  const parent = path.dirname(packageDir);
  assert.notEqual(parent, packageDir, 'Cannot locate package metadata');
  packageDir = parent;
}
const packageInfo = JSON.parse(fs.readFileSync(path.join(packageDir, 'package.json'), 'utf8'));

const timeoutMs = Number(process.env.MOCK_IDLE_MS ?? 100);
const delayMs = Number(process.env.MOCK_TOOL_DELAY_MS ?? 400);
assert(delayMs > timeoutMs * 2);
const html = '<!doctype html>\n<html><head><title>Exact</title></head><body>hello "world" & café</body></html>\n';
const args = { path: 'page.html', content: html };
const argsJson = JSON.stringify(args);
const context = {
  messages: [{ role: 'user', content: 'Write the supplied HTML file.', timestamp: 0 }],
  tools: [{ name: 'write', description: 'Write a file.', parameters: ai.type({ path: 'string', content: 'string' }) }],
};

async function arm(label, compatMs, explicitMs, expectSuccess, initialProgress = true) {
  const model = buildModel({
    id: 'mock-sushi', name: 'Mock Sushi', api: 'openai-completions', provider: 'sushi',
    baseUrl: 'http://127.0.0.1:1/v1', reasoning: false, input: ['text'],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 4096, maxTokens: 1024,
    compat: { supportsDeveloperRole: false, maxTokensField: 'max_tokens', streamIdleTimeoutMs: compatMs },
  });
  assert.equal(model.compat.streamIdleTimeoutMs, compatMs);
  if (!initialProgress) assert.equal(model.compat.streamFirstEventTimeoutMs, 0, 'Loopback first-event policy changed');
  const stats = { fetches: 0, comments: 0, emptyDeltas: 0, completeToolSent: false, requestAborted: false };
  let cleanTransport = () => {};
  const encoder = new TextEncoder();
  const mockedFetch = async (input, init = {}) => {
    stats.fetches++;
    assert.equal(stats.fetches, 1, 'Unexpected retry');
    const url = typeof input === 'string' || input instanceof URL ? String(input) : input.url;
    assert.equal(url, model.baseUrl + '/chat/completions');
    const signal = init.signal ?? (input instanceof Request ? input.signal : undefined);
    const chunk = (delta, finish_reason = null) => ({
      id: 'chatcmpl-mock', object: 'chat.completion.chunk', created: 1, model: model.id,
      choices: [{ index: 0, delta, finish_reason }],
    });
    let interval, timer, abortListener, ended = false;
    const body = new ReadableStream({
      start(controller) {
        const send = data => { if (!ended) controller.enqueue(encoder.encode(data)); };
        const event = data => send('data: ' + JSON.stringify(data) + '\n\n');
        const cleanup = () => {
          clearInterval(interval); clearTimeout(timer);
          if (abortListener) signal?.removeEventListener('abort', abortListener);
        };
        abortListener = () => {
          if (ended) return;
          ended = true; stats.requestAborted = true; cleanup();
          controller.error(signal?.reason ?? new Error('Mock request aborted'));
        };
        cleanTransport = () => { cleanup(); if (!ended) { ended = true; controller.close(); } };
        signal?.addEventListener('abort', abortListener, { once: true });
        // Real model output begins the idle interval; no tool arguments exist yet.
        event(chunk(initialProgress ? { role: 'assistant', content: 'Preparing file.' } : { role: 'assistant' }));
        interval = setInterval(() => {
          stats.comments++; send(': keepalive\n\n');
          stats.emptyDeltas++; event(chunk({}));
        }, 5);
        timer = setTimeout(() => {
          if (ended) return;
          stats.completeToolSent = true;
          event(chunk({ tool_calls: [{ index: 0, id: 'call_mock', type: 'function',
            function: { name: 'write', arguments: argsJson } }] }));
          event(chunk({}, 'tool_calls'));
          send('data: [DONE]\n\n');
          ended = true; cleanup(); controller.close();
        }, delayMs);
        if (signal?.aborted) abortListener();
      },
      cancel() { cleanTransport(); },
    });
    return new Response(body, { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
  };
  const controller = new AbortController();
  const hardLimit = setTimeout(() => controller.abort(new Error('Test hard deadline')), 3000);
  const started = performance.now();
  const events = [];
  let result;
  try {
    const stream = ai.stream(model, context, {
      apiKey: 'mock-only', temperature: 0, maxTokens: 100, fetch: mockedFetch,
      signal: controller.signal,
      ...(initialProgress ? { streamFirstEventTimeoutMs: 1000 } : {}),
      ...(explicitMs === undefined ? {} : { streamIdleTimeoutMs: explicitMs }),
    });
    for await (const event of stream) events.push(event);
    result = await stream.result();
  } finally { clearTimeout(hardLimit); cleanTransport(); }
  assert(stats.comments > 0 && stats.emptyDeltas > 0, 'Keepalive interval was not exercised');
  const calls = result.content.filter(c => c.type === 'toolCall');
  if (expectSuccess) {
    assert.equal(result.stopReason, 'toolUse');
    assert.equal(stats.requestAborted, false);
    assert.equal(stats.completeToolSent, true);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].name, 'write');
    assert.deepEqual(calls[0].arguments, args);
    assert.equal(Buffer.compare(Buffer.from(calls[0].arguments.content), Buffer.from(html)), 0);
    const deltas = events.filter(e => e.type === 'toolcall_delta').map(e => e.delta).join('');
    assert.equal(deltas, argsJson, 'Wire argument fragments changed');
  } else {
    assert(['error', 'aborted'].includes(result.stopReason), JSON.stringify(result));
    assert.match(result.errorMessage ?? '', /stalled|timed out|timeout/i);
    assert.equal(stats.requestAborted, true);
    assert.equal(stats.completeToolSent, false);
    assert.equal(calls.length, 0);
  }
  return { label, compatMs, explicitMs: explicitMs ?? null, initialProgress, elapsedMs: Math.round(performance.now() - started),
    stopReason: result.stopReason, ...stats, exactHtmlArguments: expectSuccess, error: result.errorMessage ?? null };
}

const results = [];
results.push(await arm('short-compat-control', timeoutMs, undefined, false));
results.push(await arm('sushi-compat-disabled', 0, undefined, true));
results.push(await arm('explicit-timeout-wins', 0, timeoutMs, false));
results.push(await arm('loopback-initial-buffered-call', 0, undefined, true, false));
console.log(JSON.stringify({ package: packageInfo.name, version: packageInfo.version,
  publicAPI: 'stream', timeoutMs, delayMs, results, passed: true }, null, 2));
