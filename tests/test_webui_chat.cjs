// sushi-test: tier=hermetic needs=node,playwright
// The chat tab in a real headless browser against a scriptable stub of the API (a local http server, so a stream
// can be cut, delayed or sent in pieces): what a broken stream, a deleted or shared conversation, a full browser
// store or a damaged saved chat does to the user's text, plus the markdown the models write.
// Needs Playwright with a Chromium build (global or local install); skips without one.
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const { execSync } = require('node:child_process');
const assert = require('node:assert/strict');

function loadPlaywright() {
  const roots = [undefined];
  try { roots.push(path.join(execSync('npm root -g', { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim(), 'playwright')); } catch {}
  for (const root of roots) {
    try { return require(root ?? 'playwright'); } catch {}
  }
  return null;
}

const PAGE = 'src/webui/index.html';
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const PNG = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=', 'base64');
const model = (o = {}) => ({ id: 'Chat-A', state: 'ready', loaded: true, capabilities: ['chat'], context_length: 8192, bytes_on_disk: 2 ** 30, input_modalities: ['text'], ...o });
const sse = (res, obj) => res.write(`data: ${JSON.stringify(obj)}\n\n`);
const delta = (res, content) => sse(res, { choices: [{ delta: { content } }] });
const finish = (res) => { sse(res, { choices: [{ delta: {}, finish_reason: 'stop' }] }); res.end('data: [DONE]\n\n'); };
const head = (res) => { res.writeHead(200, { 'content-type': 'text/event-stream' }); res.flushHeaders(); };
const replying = (text) => (req, res) => { head(res); delta(res, text); finish(res); };
const trickling = (count, gap = 100) => async (req, res) => {
  head(res);
  for (let i = 0; i < count && !res.destroyed; i++) { delta(res, `tok${i} `); await sleep(gap); }
  if (!res.destroyed) finish(res);
};

async function startStub({ models = [model()], chat = replying('hello'), tools = () => ({ edit_default: false }), port = 0 } = {}) {
  const state = { models, chat, tools, calls: [] };
  const server = http.createServer(async (req, res) => {
    const p = new URL(req.url, 'http://x').pathname;
    let body = ''; for await (const piece of req) body += piece;
    state.calls.push({ method: req.method, path: p, body });
    const json = (o, status = 200) => { res.writeHead(status, { 'content-type': 'application/json' }); res.end(JSON.stringify(o)); };
    if (req.method === 'GET' && /^\/(chat|dashboard|models|settings)?$/.test(p)) { res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' }); return res.end(fs.readFileSync(PAGE)); }
    switch (p) {
      case '/v1/models': return json({ data: state.models });
      case '/props': return json({ settings: { version: '9.9.9', mtp: {}, prefix_cache: {} }, memory: { available_bytes: 5e9, total_bytes: 2 ** 37 }, batching: {} });
      case '/v1/tools': return json(await state.tools(JSON.parse(body || '{}')));
      case '/v1/chat/completions': return state.chat(req, res, JSON.parse(body || '{}'));
      case '/v1/load-model': return json({ ok: true });
      case '/metrics.json': return json({ counters: {}, gauges: {}, histograms: {}, sessions: [] });
      default: return json({}, 404);
    }
  });
  await new Promise((resolve) => server.listen(port, '127.0.0.1', resolve));
  state.port = server.address().port;
  state.url = `http://127.0.0.1:${state.port}`;
  state.close = () => { server.closeAllConnections?.(); return new Promise((resolve) => server.close(resolve)); };
  opened.push(state.close);
  state.posts = () => state.calls.filter((c) => c.path === '/v1/chat/completions');
  return state;
}

async function open(browser, stub, { viewport = { width: 1440, height: 900 }, storage = {}, scheme = 'dark', clock = false, context = null, ready = true } = {}) {
  const ctx = context ?? await browser.newContext({ viewport, colorScheme: scheme });
  opened.push(() => ctx.close());
  const page = await ctx.newPage();
  page.setDefaultTimeout(4000);
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('dialog', (dialog) => dialog.dismiss());
  await page.addInitScript((seed) => {
    if (sessionStorage.getItem('__seeded')) return;
    sessionStorage.setItem('__seeded', '1');
    try { localStorage.setItem('sushi-ui-mode', 'advanced'); for (const [k, v] of Object.entries(seed)) localStorage.setItem(k, v); } catch {}
  }, storage);
  if (clock) await page.clock.install();
  await page.goto(`${stub.url}/chat`);
  if (ready) await page.waitForSelector('#modelSelect option', { state: 'attached' });
  return { page, context: ctx, errors };
}

const chatOf = (id, title, messages) => ({ id, title, updated: Date.now(), messages });
const seedChats = (...chats) => ({ 'sushi-chats': JSON.stringify(chats) });
const send = async (page, text) => { await page.fill('#promptInput', text); await page.keyboard.press('Enter'); };
const stored = (page) => page.evaluate(() => JSON.parse(localStorage.getItem('sushi-chats') || '[]'));
const titles = (page) => page.$$eval('.conv-item .c-title', (nodes) => nodes.map((n) => n.textContent));
// A control nested in another (a button in a role=button row) is unreachable or announced wrongly by a screen reader.
const nestedControls = (page, root) => page.evaluate((selector) => {
  const control = 'button, a[href], input, select, textarea, summary, [tabindex], [role=button], [role=option], [role=link], [role=menuitem]';
  return [...document.querySelectorAll(`${selector} :is(${control})`)].filter((node) => node.parentElement.closest(`${selector} :is(${control})`)).map((node) => node.outerHTML.slice(0, 60));
}, root);
const idle = (page) => page.waitForSelector('#sendButton:not([hidden])');

// Everything a test opens is closed after it, pass or fail, so a failure cannot leave a server holding the process.
const opened = [];
const tests = [];
const test = (name, fn) => tests.push([name, fn]);

test('the generation fields say what the model uses unset, and a model that cannot say keeps the bare word', async ({ browser }) => {
  const known = model({ id: 'Known', meta: { default_max_tokens: 256000, default_temperature: 1, default_top_p: 0.95, default_top_k: 20 } });
  const windowed = model({ id: 'Windowed', meta: { default_max_tokens: 0, default_temperature: 0.6, default_top_p: 1, default_top_k: 0 } });
  const cold = model({ id: 'Cold', state: 'unloaded', loaded: false });
  const stub = await startStub({ models: [known, windowed, cold] });
  const { page, errors } = await open(browser, stub);
  const placeholders = () => page.evaluate(() => ['optMaxTokens', 'optTemperature', 'optTopP', 'optTopK'].map((id) => document.getElementById(id).placeholder));
  await page.selectOption('#modelSelect', 'Known');
  assert.deepEqual(await placeholders(), ['Default (256,000)', 'Default (1.0)', 'Default (0.95)', 'Default (20)']);
  await page.selectOption('#modelSelect', 'Windowed');
  assert.deepEqual(await placeholders(), ['Default (rest of the context window)', 'Default (0.6)', 'Default (1.0)', 'Default (off)']);
  await page.selectOption('#modelSelect', 'Cold');
  assert.deepEqual(await placeholders(), ['Default', 'Default', 'Default', 'Default']);
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
  assert.deepEqual(errors, []);
});

// "Reply" as a speed, length or state word is retired; it stays only where it means the answer's text.
const RETIRED_REPLY = /\\brepl(?:y|ies)\\s+(?:speed|length|tok|done|[\\d-])|\\breplying\\b|\\blast reply\\b|(?:faster|slower|slowly)\\s+repl|repl(?:y|ies)\\s+(?:come\\s+)?(?:faster|more slowly)|\\bspeeds?\\s+(?:up\\s+)?repl|\\breply tokens\\b/i;

test('every plain-worded chat label keeps its engineering name in the tooltip', async ({ browser }) => {
  const stub = await startStub({ models: [model({ input_modalities: ['text', 'image'] })] });
  const { page, errors } = await open(browser, stub);
  await page.waitForSelector('.perf-row .k');
  const terms = await page.$$eval('#chatRightbar [data-tip], #chatRightbar ~ * [data-tip], .badge[data-tip], .perf-row .k, #chatRightbar summary', (ns) =>
    Object.fromEntries(ns.map((n) => [n.textContent.trim().replace(/\s+/g, ' '), n.dataset.term ?? ''])));
  const want = { 'First token': /TTFT/, Prompt: /prompt_tokens/, 'Prompt speed': /prefill/, 'Generation speed': /decode/, Generated: /output tokens|completion_tokens/, 'Last request': /^$/,
    'Prompt cache (SSD)': /prefix cache/, '8,192-token context': /ctx_size|context window/, 'sees images': /vision/, 'Thinking budget': /reasoning_budget_tokens/ };
  for (const [label, term] of Object.entries(want)) {
    const key = Object.keys(terms).find((k) => k === label || k.startsWith(label));
    assert.ok(key, `${label} is a tooltip anchor (${Object.keys(terms)})`);
    assert.match(terms[key], term, `${label} names its technical term`);
  }
  const words = await page.evaluate(() => [...document.querySelectorAll('body *:not(script):not(style)')].flatMap((n) => [...n.childNodes].filter((c) => c.nodeType === 3).map((c) => c.textContent)
    .concat(['tip', 'term'].map((k) => n.dataset[k]), n.getAttribute('aria-label'), n.title)).filter(Boolean).join('\n'));
  assert.doesNotMatch(words, /guess|conversation cache/i, 'no page says guess or Conversation cache');
  assert.doesNotMatch(words, RETIRED_REPLY, 'no speed, length or state says reply: thinking and answer are the same generated tokens');
  assert.deepEqual(errors, []);
});

test('a refused request shows the server reason and what to do next, not a bare status code', async ({ browser }) => {
  const cases = [[503, 'Requested model is not currently loaded', /on Models/], [400, 'Prompt does not fit', /new chat|settings/i], [500, 'boom', /server log/]];
  for (const [status, message, next] of cases) {
    const stub = await startStub({ chat: (req, res) => { res.writeHead(status, { 'content-type': 'application/json' }); res.end(JSON.stringify({ error: { message } })); } });
    const { page, context } = await open(browser, stub);
    await send(page, 'question');
    const text = await (await page.waitForSelector('.chat-error span')).textContent();
    assert.ok(text.includes(message), `${status}: keeps the server's reason`);
    assert.match(text, next, `${status}: says what to do next`);
    assert.doesNotMatch(text, new RegExp(`^${status}:`), `${status}: no bare status prefix`);
    await context.close(); await stub.close();
  }
});

test('a stream that breaks keeps the words already shown, saved as an interrupted reply with Retry', async ({ browser }) => {
  for (const mode of ['destroy', 'errorEvent', 'cleanClose']) {
    const stub = await startStub({
      chat: async (req, res) => {
        head(res);
        delta(res, 'A partial answer the user already read. '); await sleep(120);
        delta(res, 'And a little more.'); await sleep(120);
        if (mode === 'destroy') res.socket.destroy();
        else if (mode === 'errorEvent') { sse(res, { error: { message: 'Metal OOM' } }); res.end(); } else res.end();
      },
    });
    const { page, context } = await open(browser, stub);
    await send(page, 'question');
    await page.waitForSelector('.chat-error');
    await idle(page);
    const view = await page.evaluate(() => ({ shown: [...document.querySelectorAll('.msg.assistant .content')].map((n) => n.textContent).join('|'), retry: !!document.querySelector('.chat-error .btn') }));
    assert.match(view.shown, /A partial answer the user already read\. And a little more\./, `${mode}: the partial stays on screen`);
    assert.equal(view.retry, true, `${mode}: Retry is offered`);
    const saved = (await stored(page))[0].messages;
    assert.deepEqual(saved.map((m) => m.role), ['user', 'assistant'], `${mode}: the partial reply is saved`);
    assert.match(saved[1].content, /And a little more/, `${mode}: saved text`);
    assert.match(saved[1].meta, /interrupted/, `${mode}: marked as interrupted`);
    await context.close(); await stub.close();
  }
});

test('deleting the conversation whose reply is streaming leaves it deleted', async ({ browser }) => {
  const stub = await startStub({ chat: trickling(30) });
  const { page, context } = await open(browser, stub, { storage: seedChats(chatOf('a', 'Doomed', [{ role: 'user', content: 'hi' }, { role: 'assistant', content: 'yo' }])) });
  await page.click('.conv-item');
  await send(page, 'follow-up');
  await page.waitForSelector('#stopButton:not([hidden])');
  await page.waitForTimeout(300);
  await page.click('.conv-item.selected [aria-label="Delete conversation"]');
  await page.click('.conv-item.selected .btn.danger');
  await page.waitForTimeout(1200);
  assert.deepEqual(await titles(page), [], 'the row stays gone');
  assert.equal((await stored(page)).length, 0, 'and stays out of the saved list');
  await context.close(); await stub.close();
});

test('two tabs share conversations instead of overwriting each other', async ({ browser }) => {
  const stub = await startStub();
  const context = await browser.newContext();
  const a = (await open(browser, stub, { context })).page;
  const b = (await open(browser, stub, { context })).page;
  await send(a, 'question from tab A'); await a.waitForSelector('.msg.assistant .content'); await idle(a);
  await b.waitForFunction(() => document.querySelectorAll('.conv-item').length === 1);
  await send(b, 'question from tab B'); await b.waitForSelector('.msg.assistant .content'); await idle(b);
  await a.waitForFunction(() => document.querySelectorAll('.conv-item').length === 2);
  await a.reload(); await a.waitForSelector('#modelSelect option', { state: 'attached' });
  assert.deepEqual((await titles(a)).sort(), ['question from tab A', 'question from tab B']);
  await a.hover('.conv-item:has-text("tab A")'); await a.click('.conv-item:has-text("tab A") [aria-label="Delete conversation"]'); await a.click('.conv-item .btn.danger');
  await b.waitForFunction(() => document.querySelectorAll('.conv-item').length === 1);
  await b.click('#newChatButton'); await send(b, 'third'); await b.waitForSelector('.msg.assistant .content'); await idle(b);
  await b.reload(); await b.waitForSelector('#modelSelect option', { state: 'attached' });
  assert.deepEqual((await titles(b)).sort(), ['question from tab B', 'third'], 'a deleted conversation does not come back from the other tab');
  await context.close(); await stub.close();
});

test('a full browser store says so once and keeps the newest conversations', async ({ browser }) => {
  const stub = await startStub();
  const filler = 'x'.repeat(1500);
  const old = [1, 2, 3, 4, 5, 6].map((i) => chatOf(`old${i}`, `Old ${i}`, [{ role: 'user', content: filler }]));
  const { page, context } = await open(browser, stub, { storage: seedChats(...old) });
  await page.evaluate(() => {
    const original = Storage.prototype.setItem;
    Storage.prototype.setItem = function (key, value) {
      if (key === 'sushi-chats' && value.length > 4500) { const e = new Error('quota'); e.name = 'QuotaExceededError'; throw e; }
      return original.call(this, key, value);
    };
  });
  await send(page, 'important question'); await page.waitForSelector('.msg.assistant .content'); await idle(page);
  assert.equal(await page.$$eval('.toast.err', (n) => n.length), 1, 'one warning, not one per save');
  assert.match(await page.textContent('#toastHost'), /storage is full/i);
  await page.evaluate(() => { delete Storage.prototype.setItem; });
  await page.reload(); await page.waitForSelector('#modelSelect option', { state: 'attached' });
  assert.ok((await titles(page)).includes('important question'), 'the conversation just held survives a reload');
  await context.close(); await stub.close();
});

test('Send with no chat model listed keeps the typed prompt', async ({ browser }) => {
  const stub = await startStub({ models: [] });
  const { page, context } = await open(browser, stub, { ready: false });
  await page.waitForTimeout(500);
  await send(page, 'A long prompt I spent ten minutes writing');
  await page.waitForTimeout(200);
  assert.equal(await page.inputValue('#promptInput'), 'A long prompt I spent ten minutes writing');
  assert.equal(stub.posts().length, 0);
  await context.close(); await stub.close();
});

test('a damaged saved conversation does not stop the chat page', async ({ browser }) => {
  const stub = await startStub();
  const cases = {
    'null message': [{ id: 'a', messages: [null, { role: 'user', content: 'x' }] }],
    'content is a number': [{ id: 'a', messages: [{ role: 'user', content: 42 }, { role: 'assistant', content: { x: 1 } }] }],
    'tool_calls is not a list': [{ id: 'a', messages: [{ role: 'user', content: 'x' }, { role: 'assistant', content: 'y', tool_calls: 'oops' }] }],
    'a null tool call': [{ id: 'a', messages: [{ role: 'user', content: 'x' }, { role: 'assistant', content: 'y', tool_calls: [null] }] }],
    'title is an object': [{ id: 'a', title: { a: 1 }, messages: [{ role: 'user', content: 'x' }] }],
    'images is a string': [{ id: 'a', messages: [{ role: 'user', content: 'x', images: 'data:' }] }],
    'no id': [{ messages: [{ role: 'user', content: 'x' }] }],
  };
  for (const [name, value] of Object.entries(cases)) {
    const { page, context, errors } = await open(browser, stub, { storage: { 'sushi-chats': JSON.stringify(value) } });
    assert.equal(await page.evaluate(() => document.getElementById('modelSelect').options.length), 1, `${name}: models load`);
    await page.click('.conv-item');
    await send(page, 'hello'); await page.waitForFunction(() => /hello/.test(document.getElementById('chatMessages').textContent)); await idle(page);
    assert.deepEqual(errors, [], name);
    await context.close();
  }
  await stub.close();
});

test('a reloaded image turn is never sent as an empty message, and is not regenerated blind', async ({ browser }) => {
  const stub = await startStub({ models: [model({ input_modalities: ['text', 'image'], capabilities: ['chat', 'vision'] })] });
  const { page, context } = await open(browser, stub, { storage: seedChats(chatOf('a', '', [{ role: 'user', content: '', droppedImages: 1 }, { role: 'assistant', content: 'A cat.' }])) });
  await page.click('.conv-item');
  await page.click('[aria-label="Regenerate the reply"]');
  await page.waitForTimeout(300);
  assert.equal(stub.posts().length, 0, 'regenerating a reply to an image that is gone sends nothing');
  assert.match(await page.textContent('#toastHost'), /image/i);
  await send(page, 'and the colour?'); await page.waitForFunction(() => /hello/.test(document.getElementById('chatMessages').textContent)); await idle(page);
  const wire = JSON.parse(stub.posts()[0].body).messages;
  assert.ok(wire.every((m) => m.role !== 'user' || (typeof m.content === 'string' ? m.content.trim() : m.content.length)), 'no empty user message on the wire');
  await context.close(); await stub.close();
});

test('the keyboard reaches Rename and Delete on a conversation row', async ({ browser }) => {
  const stub = await startStub();
  const { page, context } = await open(browser, stub, { storage: seedChats(chatOf('a', 'First', [{ role: 'user', content: 'hi' }]), chatOf('b', 'Second', [{ role: 'user', content: 'yo' }])) });
  assert.deepEqual(await nestedControls(page, '#conversationList'), [], 'no control sits inside another');
  assert.deepEqual(await page.evaluate(() => [...document.querySelectorAll('#conversationList, #conversationList [role]')].map((n) => n.getAttribute('role'))), ['list', 'listitem', 'listitem'], 'rows are list items, the open control is a real button');
  assert.equal(await page.evaluate(() => document.querySelectorAll('#conversationList .conv-item > button.c-open').length), 2);
  await page.focus('.conv-item:nth-child(2) .c-open');
  await page.keyboard.press('Tab');
  assert.equal(await page.evaluate(() => document.activeElement.getAttribute('aria-label')), 'Rename conversation', 'Tab from a row reaches its Rename button');
  await page.keyboard.press('Shift+Tab');
  await page.keyboard.press('Tab'); await page.keyboard.press('Tab');
  assert.equal(await page.evaluate(() => document.activeElement.getAttribute('aria-label')), 'Delete conversation');
  await page.keyboard.press('Enter');
  assert.ok(await page.$('.conv-item .btn.danger'), 'Enter on Delete asks for confirmation instead of opening the chat');
  assert.equal(await page.evaluate(() => document.activeElement.textContent), 'Delete?', 'the confirmation takes the focus');
  await page.keyboard.press('Enter');
  assert.deepEqual(await titles(page), ['First'], 'Enter again deletes');
  await page.focus('.conv-item .c-open'); await page.keyboard.press('Tab'); await page.keyboard.press('Enter');
  await page.keyboard.type('Kept'); await page.keyboard.press('Enter');
  assert.deepEqual(await titles(page), ['Kept']);
  assert.equal(await page.evaluate(() => document.activeElement.matches('.conv-item .c-open')), true, 'a rename ends on its row, not on the chat it would open');
  await page.evaluate(() => { document.activeElement.blur(); });
  await page.focus('.conv-item:nth-child(1) .c-open');
  await page.keyboard.press('Space');
  assert.ok(await page.$('.conv-item.selected'), 'Space on a row opens it');
  await context.close(); await stub.close();
});

test('renaming a conversation: leaving the box saves, Escape cancels', async ({ browser }) => {
  const stub = await startStub();
  const { page, context } = await open(browser, stub, { storage: seedChats(chatOf('a', 'First', [{ role: 'user', content: 'hi' }])) });
  await page.hover('.conv-item'); await page.click('[aria-label="Rename conversation"]');
  await page.fill('.conv-item input', 'Renamed');
  await page.evaluate(() => document.activeElement.blur());
  await page.waitForTimeout(100);
  assert.deepEqual(await titles(page), ['Renamed']);
  assert.equal((await stored(page))[0].title, 'Renamed');
  await page.hover('.conv-item'); await page.click('[aria-label="Rename conversation"]');
  await page.fill('.conv-item input', 'Discarded'); await page.keyboard.press('Escape');
  await page.waitForTimeout(100);
  assert.deepEqual(await titles(page), ['Renamed']);
  await context.close(); await stub.close();
});

const CASES = [
  ['a space before the language', '``` python\nprint(1)\n```\n\nAfter', { blocks: [['python', 'print(1)']], paras: ['After'] }],
  ['an info string after the language', '```python title="a.py"\nprint(1)\n```\n\nAfter', { blocks: [['python', 'print(1)']], paras: ['After'] }],
  ['a four-backtick fence around a fence', '````markdown\n```js\nx\n```\n````\n\nAfter', { blocks: [['markdown', '```js\nx\n```']], paras: ['After'] }],
  ['a tilde fence', '~~~\ncode\n~~~\n\nAfter', { blocks: [['code', 'code']], paras: ['After'] }],
  ['a fence inside a list item', '1. step\n   ```sh\n   ls\n   ```\n2. next', { blocks: [['sh', 'ls']], items: 2 }],
  ['a link whose address has a parenthesis', 'See [Foo](https://en.wikipedia.org/wiki/Foo_(bar)) now', { links: [['https://en.wikipedia.org/wiki/Foo_(bar)', 'Foo']], paras: ['See Foo now'] }],
  ['a pipe inside a code span in a table', '| a | b |\n|---|---|\n| `x|y` | 2 |', { cells: ['a', 'b', 'x|y', '2'] }],
  ['an escaped star', 'a \\* b \\* c', { paras: ['a * b * c'], ems: 0 }],
  ['an unterminated fence', 'text\n```js\nconst a = 1;', { blocks: [['js', 'const a = 1;']] }],
  ['stars with spaces around them are arithmetic', '2 * 3 * 4 = 24 and x ** 2 ** y', { paras: ['2 * 3 * 4 = 24 and x ** 2 ** y'], ems: 0, strongs: 0 }],
  ['emphasis beside a spaced star', 'Costs 5 * 2, *really* cheap', { paras: ['Costs 5 * 2, really cheap'], ems: 1 }],
  ['a numbered list inside a numbered list', '1. one\n   1. a\n   2. b\n2. two', { items: 2, sub: ['OL', 2] }],
  ['a list three levels deep', '- a\n  - b\n    - c\n      - d\n- e', { items: 2, depth: 4 }],
  ['a numbered list keeps its first number below the top', '1. one\n   5. five\n   6. six\n2. two', { sub: ['OL', 2], subStart: 5 }],
  ['a bullet inside a numbered inside a bullet', '- a\n  1. b\n     - c\n  2. d', { depth: 3, items: 1 }],
  ['triple stars are emphasis around strong', 'plain ***both*** and **bold** and *it*', { strongs: 2, ems: 2, nestedStrong: 1 }],
  ['triple stars with spaces are text', '2 *** 3 *** 4', { paras: ['2 *** 3 *** 4'], strongs: 0, ems: 0 }],
];
test('the markdown the models write: fences, links, tables and escapes', async ({ browser }) => {
  const stub = await startStub();
  const { page, context } = await open(browser, stub);
  for (const [name, markdown, want] of CASES) {
    const got = await page.evaluate((md) => {
      const host = document.createElement('div');
      host.appendChild(renderMarkdown(md));
      return {
        blocks: [...host.querySelectorAll('.code-block')].map((b) => [b.querySelector('.cb-head span').textContent, b.querySelector('pre').textContent]),
        paras: [...host.querySelectorAll(':scope > p')].map((p) => p.textContent),
        items: host.querySelectorAll(':scope > ol > li, :scope > ul > li').length,
        links: [...host.querySelectorAll('a')].map((a) => [a.href, a.textContent]),
        cells: [...host.querySelectorAll('th, td')].map((c) => c.textContent),
        ems: host.querySelectorAll('em').length,
        strongs: host.querySelectorAll('strong').length,
        sub: [...host.querySelectorAll('li > ol, li > ul')].map((list) => [list.tagName, list.children.length])[0] ?? null,
        subStart: host.querySelector('li > ol')?.start ?? null,
        depth: Math.max(0, ...[...host.querySelectorAll('li')].map((li) => { let d = 0; for (let n = li; n && n !== host; n = n.parentElement) if (n.tagName === 'LI') d++; return d; })),
        nestedStrong: host.querySelectorAll('em > strong').length,
      };
    }, markdown);
    for (const [key, value] of Object.entries(want)) assert.deepEqual(got[key], value, `${name}: ${key}`);
  }
  await context.close(); await stub.close();
});

test('a model that is loading, unloading or failed is not offered "loads on first message"', async ({ browser }) => {
  for (const [state, want, button] of [['loading', /loading/i, false], ['evicting', /unloading/i, false], ['error', /failed/i, true]]) {
    const stub = await startStub({ models: [model({ state, loaded: false })] });
    const { page, context } = await open(browser, stub);
    await page.waitForSelector('#modelStatus .badge');
    const text = await page.innerText('#modelStatus');
    assert.match(text, want, state);
    assert.doesNotMatch(text, /loads on first message/, state);
    assert.equal(await page.$$eval('#modelStatus button', (n) => n.length), button ? 1 : 0, `${state}: button`);
    await context.close(); await stub.close();
  }
});

test('Offline clears within seconds of the server coming back', async ({ browser }) => {
  let stub = await startStub();
  const { port } = stub;
  const { page, context } = await open(browser, stub);
  await page.waitForFunction(() => document.getElementById('serverDotLabel').textContent === 'Chat-A');
  await stub.close();
  await send(page, 'hello?');
  await page.waitForSelector('.chat-error');
  await page.waitForFunction(() => document.getElementById('serverDotLabel').textContent === 'Offline');
  stub = await startStub({ port });
  await page.waitForFunction(() => document.getElementById('serverDotLabel').textContent !== 'Offline', null, { timeout: 12000 });
  await context.close(); await stub.close();
});

test('a failed reply shows one inline notice, and error toasts leave on their own', async ({ browser }) => {
  const stub = await startStub({ chat: (req, res) => { res.writeHead(500, { 'content-type': 'application/json' }); res.end('{"error":{"message":"boom"}}'); } });
  const { page, context } = await open(browser, stub, { clock: true });
  await send(page, 'q');
  await page.waitForSelector('.chat-error');
  assert.equal(await page.$$eval('.toast.err', (n) => n.length), 0, 'the notice in the chat is the one message');
  await page.evaluate(() => { toast('Same trouble', 'err'); toast('Same trouble', 'err'); });
  assert.equal(await page.$$eval('.toast.err', (n) => n.length), 1, 'one toast per message');
  await page.clock.runFor(9000);
  assert.equal(await page.$$eval('.toast', (n) => n.length), 0, 'and it leaves on its own');
  await context.close(); await stub.close();
});

test('the SVG preview reads on a dark page', async ({ browser }) => {
  const svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40" width="80" height="80"><circle cx="20" cy="20" r="15" fill="none" stroke="black"/></svg>';
  const stub = await startStub({ chat: replying('```svg\n' + svg + '\n```') });
  const { page, context } = await open(browser, stub, { scheme: 'dark' });
  await send(page, 'draw'); await page.waitForSelector('.code-block .btn');
  await page.click('.code-block .btn');
  const rgb = await page.$eval('.svg-preview img', (img) => getComputedStyle(img).backgroundColor.match(/\d+/g).slice(0, 3).map(Number));
  assert.ok(rgb.reduce((a, b) => a + b, 0) > 600, `light backing, got rgb(${rgb})`);
  await context.close(); await stub.close();
});

test('Stop before the first token still offers Retry', async ({ browser }) => {
  const stub = await startStub({ chat: async (req, res) => { head(res); await sleep(4000); res.end(); } });
  const { page, context } = await open(browser, stub);
  await send(page, 'second');
  await page.waitForSelector('#stopButton:not([hidden])');
  await page.waitForTimeout(300);
  await page.click('#stopButton');
  await idle(page);
  assert.ok(await page.waitForSelector('.chat-error .btn', { timeout: 2000 }), 'a Retry beside the unanswered message');
  await context.close(); await stub.close();
});

test('a new conversation is in the sidebar as soon as it is sent', async ({ browser }) => {
  const stub = await startStub({ chat: trickling(20) });
  const { page, context } = await open(browser, stub);
  await send(page, 'brand new question');
  await page.waitForSelector('#stopButton:not([hidden])');
  assert.deepEqual(await titles(page), ['brand new question']);
  await page.click('#stopButton');
  await context.close(); await stub.close();
});

test('attached images that a model switch drops are announced', async ({ browser }) => {
  const stub = await startStub({ models: [model({ input_modalities: ['text', 'image'], capabilities: ['chat', 'vision'] }), model({ id: 'TextOnly' })] });
  const { page, context } = await open(browser, stub);
  await page.setInputFiles('#attachInput', { name: 'a.png', mimeType: 'image/png', buffer: PNG });
  await page.waitForSelector('.attach-chip');
  await page.selectOption('#modelSelect', 'TextOnly');
  assert.equal(await page.$$eval('.attach-chip', (n) => n.length), 0);
  assert.match(await page.textContent('#toastHost'), /image/i);
  await context.close(); await stub.close();
});

test('the chosen model comes back when it returns to the list', async ({ browser }) => {
  const stub = await startStub({ models: [model({ id: 'A' }), model({ id: 'B' })] });
  const { page, context } = await open(browser, stub, { storage: { 'sushi-model': 'B' } });
  assert.equal(await page.inputValue('#modelSelect'), 'B');
  stub.models = [model({ id: 'A' })];
  await page.evaluate(() => modelStore.refresh());
  await page.waitForFunction(() => document.getElementById('modelSelect').value === 'A');
  stub.models = [model({ id: 'A' }), model({ id: 'B' })];
  await page.evaluate(() => modelStore.refresh());
  await page.waitForFunction(() => document.getElementById('modelSelect').value === 'B');
  await context.close(); await stub.close();
});

test('a tool round follows the bottom only for a reader who is at the bottom', async ({ browser }) => {
  const filler = Array.from({ length: 30 }, (_, i) => [{ role: 'user', content: `question ${i}` }, { role: 'assistant', content: `answer ${i}\n\n` + 'line\n\n'.repeat(4) }]).flat();
  let release;
  let held = new Promise((resolve) => { release = resolve; });
  let rounds = 0;
  const stub = await startStub({
    chat: (req, res) => {
      head(res);
      if (rounds++ % 2 === 0) {
        sse(res, { choices: [{ delta: { tool_calls: [{ index: 0, id: `c${rounds}`, function: { name: 'list_dir', arguments: '{}' } }] } }] });
        sse(res, { choices: [{ delta: {}, finish_reason: 'tool_calls' }] }); res.end('data: [DONE]\n\n');
      } else { delta(res, 'done'); finish(res); }
    },
    tools: async (body) => {
      if (!body.name) return { edit_default: false, tools: [{ type: 'function', function: { name: 'list_dir', description: 'x', parameters: { type: 'object', properties: {} } } }] };
      await held;
      return { text: 'listing' };
    },
  });
  const { page, context, errors } = await open(browser, stub, { storage: { ...seedChats(chatOf('a', 'Long', filler)), 'sushi-tools': 'on' } });
  await page.click('.conv-item');
  const geometry = () => page.evaluate(() => { const s = document.getElementById('chatScroll'); return { top: s.scrollTop, gap: s.scrollHeight - s.scrollTop - s.clientHeight, room: s.scrollHeight - s.clientHeight }; });
  assert.ok((await geometry()).room > 1000, 'the chat is taller than the view');
  const settled = async () => { await page.waitForFunction(() => !document.getElementById('sendButton').hidden); await page.waitForFunction(() => /done/.test(document.getElementById('chatMessages').lastElementChild?.textContent ?? '')); };

  await send(page, 'list it');
  await page.waitForFunction(() => /list_dir/.test(document.getElementById('chatMessages').textContent));
  assert.ok((await geometry()).gap < 24, 'sending a message scrolls to the bottom');
  await page.evaluate(() => { document.getElementById('chatScroll').scrollTop = 300; });
  const before = (await geometry()).top;
  release();
  await settled();
  assert.equal((await geometry()).top, before, 'a reader who scrolled up stays put through the tool result and the reply');

  held = new Promise((resolve) => { release = resolve; });
  await page.evaluate(() => { const s = document.getElementById('chatScroll'); s.scrollTop = s.scrollHeight; });
  await send(page, 'again');
  await page.waitForFunction(() => document.getElementById('chatMessages').textContent.split('list_dir').length > 2);
  release();
  await settled();
  assert.ok((await geometry()).gap < 24, 'at the bottom the view follows the tool result and the reply');
  assert.deepEqual(errors, []);
  await context.close(); await stub.close();
});

test('after a click on Send the cursor is back in the prompt', async ({ browser }) => {
  const stub = await startStub();
  const { page, context } = await open(browser, stub);
  await page.fill('#promptInput', 'q'); await page.click('#sendButton');
  await page.waitForSelector('.msg.assistant .content'); await idle(page);
  assert.equal(await page.evaluate(() => document.activeElement.id), 'promptInput');
  await context.close(); await stub.close();
});

async function main() {
  const playwright = loadPlaywright();
  if (!playwright) { console.log('Web UI chat: skipped (Playwright is not installed)'); return; }
  let browser;
  try { browser = await playwright.chromium.launch(); } catch (error) {
    console.log(`Web UI chat: skipped (no Chromium: ${String(error.message).split('\n')[0]})`);
    return;
  }
  const only = process.argv[2];
  let failed = 0;
  try {
    for (const [name, fn] of tests) {
      if (only && !name.includes(only)) continue;
      let timer;
      try {
        await Promise.race([fn({ browser }), new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('timed out after 90 s')), 90000); })]);
      } catch (error) { failed++; console.error(`FAIL ${name}\n  ${String(error.message).split('\n').join('\n  ')}`); }
      finally {
        clearTimeout(timer);
        await Promise.allSettled(opened.splice(0).map((close) => close()));
      }
    }
  } finally {
    await browser.close();
  }
  if (failed) { process.exitCode = 1; return; }
  console.log('Web UI chat: passed');
}
main().catch((error) => { console.error(error); process.exitCode = 1; }).finally(() => process.exit());
