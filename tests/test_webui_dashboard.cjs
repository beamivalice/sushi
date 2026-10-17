// The page's Chat/Dashboard navigator, driven in a real headless browser against stubbed endpoints:
// no sushi server, no model. Needs Playwright with a Chromium build (global or local install); skips without one.
const fs = require('node:fs');
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

const html = fs.readFileSync('src/webui/index.html', 'utf8');
const ORIGIN = 'http://sushi.test';

// Runs in the page before its own scripts: every endpoint the page calls is answered here, and the chat
// stream pauses after its first words until the test opens the gate.
function installStubs(config) {
  const stub = window.__stub = { calls: [], aborted: 0, streams: 0, metricsStatus: config.metricsStatus, openGate: null, ...config };
  const gate = new Promise((resolve) => { stub.openGate = resolve; });
  window.__clears = 0;
  const realClear = CanvasRenderingContext2D.prototype.clearRect;
  CanvasRenderingContext2D.prototype.clearRect = function (...args) { window.__clears += 1; return realClear.apply(this, args); };
  if (config.key) localStorage.setItem('sushi-api-key', config.key);
  const json = (body, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
  const realFetch = window.fetch.bind(window);
  window.fetch = async (input, init = {}) => {
    const url = new URL(typeof input === 'string' ? input : input.url, location.href);
    if (url.origin !== location.origin) return realFetch(input, init);
    const headers = new Headers(init.headers ?? {});
    stub.calls.push({ path: url.pathname, search: url.search, auth: headers.get('authorization'), method: init.method ?? 'GET' });
    switch (url.pathname) {
      case '/health': return json({ status: 'ok' });
      case '/v1/models': return json({ data: [{ id: 'org/test-model', state: 'ready', loaded: true, capabilities: ['chat'], context_length: 8192, bytes_resident: 3e9, meta: { kv_quant: 8 } }] });
      case '/props': return json({ settings: { version: '9.9.9', mtp: {}, prefix_cache: {} }, memory: { available_bytes: 5e9 }, batching: { supported: true, max_group: 4 }, default_generation_settings: { n_ctx: 8192 } });
      case '/v1/tools': return json({ edit_default: false });
      case '/metrics.json':
        if (stub.metricsStatus !== 200) return json({ error: { message: 'metrics off' } }, stub.metricsStatus);
        return json({ counters: { prompt_tokens_total: 10 }, gauges: { requests_running: 1, generation_tokens_live: stub.metricsTick = (stub.metricsTick ?? 0) + 40, gpu_utilization_pct: 55, memory_mb: 4096 }, histograms: {}, sessions: [] });
      case '/v1/chat/completions': {
        stub.streams += 1;
        const signal = init.signal;
        const encoder = new TextEncoder();
        const sse = (text) => encoder.encode(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: text } }] })}\n\n`);
        const body = new ReadableStream({
          async start(controller) {
            signal?.addEventListener('abort', () => { stub.aborted += 1; controller.error(new DOMException('aborted', 'AbortError')); });
            controller.enqueue(sse(stub.firstWords));
            await gate;
            controller.enqueue(sse(' and the rest'));
            controller.enqueue(encoder.encode('data: [DONE]\n\n'));
            controller.close();
          },
        });
        return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } });
      }
      default: return json({}, 404);
    }
  };
}

async function openPage(browser, urlPath, config = {}, fakeClock = false) {
  const context = await browser.newContext({ viewport: { width: 1280, height: 800 } });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.route(`${ORIGIN}/**`, (route) => route.fulfill({ status: 200, contentType: 'text/html; charset=utf-8', body: html }));
  await page.addInitScript(installStubs, { metricsStatus: 200, firstWords: 'Streaming '.repeat(400), ...config });
  if (fakeClock) await page.clock.install();
  await page.goto(`${ORIGIN}${urlPath}`);
  return { page, errors, context };
}

async function main() {
  const playwright = loadPlaywright();
  if (!playwright) { console.log('Web UI dashboard: skipped (Playwright is not installed)'); return; }
  let browser;
  try { browser = await playwright.chromium.launch(); } catch (error) {
    console.log(`Web UI dashboard: skipped (no Chromium: ${String(error.message).split('\n')[0]})`);
    return;
  }
  try {
    {
      const { page, errors, context } = await openPage(browser, '/chat');
      await page.waitForSelector('#modelSelect option', { state: 'attached' });
      assert.equal(await page.isVisible('#chatPanel'), true, 'chat is the view at /chat');
      assert.equal(await page.isVisible('#dashPanel'), false, 'the dashboard starts hidden');
      assert.equal(await page.getAttribute('#navChat', 'aria-current'), 'page');
      assert.equal(await page.evaluate(() => document.getElementById('dashCTps').width), 300, 'a hidden dashboard draws nothing');

      await page.fill('#promptInput', 'tell me a story');
      await page.click('#sendButton');
      await page.waitForFunction(() => document.querySelector('#chatScroll')?.textContent.includes('Streaming Streaming'));
      await page.fill('#promptInput', 'a half-typed follow-up');
      const before = await page.evaluate(() => {
        const scroll = document.getElementById('chatScroll');
        scroll.scrollTop = Math.floor(scroll.scrollHeight / 3);
        window.__column = document.querySelector('.chat-column');
        window.__marked = scroll.querySelector('.chat-column *:last-child');
        return { top: scroll.scrollTop, html: window.__column.innerHTML, history: document.getElementById('conversationList').innerHTML };
      });
      assert.ok(before.top > 0, 'the transcript is long enough to scroll');
      assert.ok(await page.isDisabled('#sendButton') || await page.isVisible('#stopButton'), 'the turn is in flight');

      await page.click('#navDashboard');
      assert.equal(new URL(page.url()).pathname, '/dashboard');
      assert.equal(await page.isVisible('#dashPanel'), true);
      assert.equal(await page.isVisible('#chatPanel'), false);
      assert.equal(await page.getAttribute('#navDashboard', 'aria-current'), 'page');
      const hiddenChat = await page.evaluate(() => { const panel = document.getElementById('chatPanel'); const rect = panel.getBoundingClientRect(); return { display: getComputedStyle(panel).display, width: rect.width, height: rect.height }; });
      assert.notEqual(hiddenChat.display, 'none', 'the hidden chat keeps its layout');
      assert.ok(hiddenChat.width > 0 && hiddenChat.height > 0, 'and its size');
      await page.waitForFunction(() => /Live/.test(document.getElementById('dashStatus').textContent));
      await page.waitForFunction(() => document.getElementById('dashModels').textContent.includes('test-model'), null, { timeout: 5000 });
      assert.notEqual(await page.evaluate(() => document.getElementById('dashCTps').width), 300, 'a visible dashboard draws its charts');

      await page.evaluate(() => window.__stub.openGate());
      await page.waitForTimeout(300);
      assert.equal(await page.evaluate(() => window.__stub.aborted), 0, 'the switch aborted nothing');

      await page.click('#navChat');
      assert.equal(new URL(page.url()).pathname, '/chat');
      assert.equal(await page.isVisible('#chatPanel'), true);
      assert.equal(await page.isVisible('#dashPanel'), false);
      await page.waitForFunction(() => document.querySelector('#chatScroll').textContent.includes('and the rest'));
      const after = await page.evaluate(() => ({
        sameColumn: window.__column === document.querySelector('.chat-column') && document.contains(window.__marked),
        composer: document.getElementById('promptInput').value,
        text: document.getElementById('chatScroll').textContent,
        busy: document.getElementById('stopButton') && !document.getElementById('stopButton').hidden && getComputedStyle(document.getElementById('stopButton')).display !== 'none',
        streams: window.__stub.streams,
        aborted: window.__stub.aborted,
      }));
      assert.equal(after.sameColumn, true, 'the conversation DOM is the same nodes');
      assert.equal(after.composer, 'a half-typed follow-up', 'the composer text survives');
      assert.ok(after.text.includes('tell me a story'), 'the question is still shown');
      assert.match(after.text, /(Streaming\s+){400}and the rest/, 'the whole answer arrived');
      assert.equal(after.streams, 1, 'one request, never repeated');
      assert.equal(after.aborted, 0, 'nothing aborted');
      assert.equal(after.busy, false, 'the turn finished');
      assert.equal(await page.evaluate(() => document.getElementById('chatPanel').inert), false, 'the chat is not left inert');
      const box = await page.evaluate(() => {
        const panel = document.getElementById('chatPanel');
        const rect = panel.getBoundingClientRect();
        return { display: getComputedStyle(panel).display, visibility: getComputedStyle(panel).visibility, width: rect.width, height: rect.height };
      });
      assert.notEqual(box.display, 'none', 'the chat panel is laid out');
      assert.equal(box.visibility, 'visible');
      assert.ok(box.width > 0 && box.height > 0, 'the chat panel has a size');
      await page.click('#promptInput');
      await page.keyboard.type('x');
      assert.equal(await page.inputValue('#promptInput'), 'a half-typed follow-upx', 'the composer takes typing again');
      assert.deepEqual(errors, [], 'no page errors');
      await context.close();
    }

    {
      const { page, context } = await openPage(browser, '/chat');
      await page.waitForSelector('#modelSelect option', { state: 'attached' });
      await page.focus('#promptInput');
      await page.click('#navDashboard');
      await page.click('#navChat');
      assert.equal(await page.evaluate(() => document.activeElement?.id), 'promptInput', 'focus returns to what had it');
      await page.evaluate(() => document.getElementById('promptInput').blur());
      await page.click('#navDashboard');
      await page.click('#navChat');
      assert.notEqual(await page.evaluate(() => document.activeElement?.id), 'promptInput', 'focus is not forced onto the composer');
      await context.close();
    }

    {
      const { page, context } = await openPage(browser, '/chat');
      await page.waitForSelector('#modelSelect option', { state: 'attached' });
      await page.fill('#promptInput', 'scroll test');
      await page.click('#sendButton');
      await page.waitForFunction(() => document.querySelector('#chatScroll')?.textContent.includes('Streaming Streaming'));
      const top = await page.evaluate(() => {
        const scroll = document.getElementById('chatScroll');
        scroll.scrollTop = 120;
        return scroll.scrollTop;
      });
      await page.click('#navDashboard');
      await page.click('#navChat');
      assert.equal(await page.evaluate(() => document.getElementById('chatScroll').scrollTop), top, 'scroll position survives');
      await context.close();
    }

    {
      const { page, errors, context } = await openPage(browser, '/chat');
      await page.waitForSelector('#modelSelect option', { state: 'attached' });
      await page.click('#navDashboard');
      await page.goBack();
      assert.equal(new URL(page.url()).pathname, '/chat');
      assert.equal(await page.isVisible('#chatPanel'), true, 'Back returns to chat without a reload');
      assert.equal(await page.isVisible('#dashPanel'), false);
      await page.goForward();
      assert.equal(new URL(page.url()).pathname, '/dashboard');
      assert.equal(await page.isVisible('#dashPanel'), true, 'Forward returns to the dashboard');
      assert.equal(await page.isVisible('#chatPanel'), false);
      assert.equal(await page.evaluate(() => performance.getEntriesByType('navigation').length), 1);
      assert.deepEqual(errors, []);
      await context.close();
    }

    {
      const { page, context } = await openPage(browser, '/dashboard');
      assert.equal(await page.isVisible('#dashPanel'), true, 'a link or reload lands on the dashboard');
      assert.equal(await page.isVisible('#chatPanel'), false);
      await page.click('#navChat');
      assert.equal(await page.isVisible('#chatPanel'), true);
      await context.close();
    }

    {
      const { page, context } = await openPage(browser, '/dashboard', { key: 'k-12345' });
      await page.waitForFunction(() => window.__stub.calls.some((call) => call.path === '/metrics.json'));
      await page.waitForFunction(() => window.__stub.calls.some((call) => call.path === '/props'));
      const calls = await page.evaluate(() => window.__stub.calls.filter((call) => call.path !== '/health'));
      for (const call of calls) {
        assert.equal(call.auth, 'Bearer k-12345', `${call.path} carries the chat's key`);
        assert.ok(!/api_key/.test(call.search), `${call.path} has no key in the query`);
      }
      assert.equal(await page.isVisible('#dashKey'), false, 'no second key field');
      const select = await page.evaluate(() => { const style = getComputedStyle(document.getElementById('dashInterval')); return { image: style.backgroundImage, right: style.paddingRight }; });
      assert.notEqual(select.image, 'none', 'the dashboard selects keep their chevron');
      assert.equal(select.right, '28px');
      await context.close();
    }

    {
      const { page, context } = await openPage(browser, '/dashboard', { metricsStatus: 503 });
      await page.waitForFunction(() => /--metrics/.test(document.getElementById('dashStatus').textContent));
      await page.waitForFunction(() => document.getElementById('dashModels').textContent.includes('test-model'));
      await context.close();
    }

    {
      const { page, context } = await openPage(browser, '/chat', {}, true);
      await page.waitForSelector('#modelSelect option', { state: 'attached' });
      await page.fill('#promptInput', 'poll test');
      await page.click('#sendButton');
      await page.waitForFunction(() => window.__stub.streams === 1);
      const count = () => page.evaluate(() => window.__stub.calls.filter((call) => call.path === '/metrics.json').length);
      const base = await count();
      await page.clock.runFor(30_000);
      assert.equal(await count(), base, 'no background poll competes with a chat request');
      await page.evaluate(() => window.__stub.openGate());
      await page.waitForFunction(() => document.querySelector('#chatScroll').textContent.includes('and the rest'));
      await page.clock.runFor(10_000);
      const idle = await count();
      assert.ok(idle - base >= 1 && idle - base <= 5, `a light background poll runs while chat is idle (saw ${idle - base})`);
      await page.evaluate(() => {
        Object.defineProperty(document, 'hidden', { value: true, configurable: true });
        document.dispatchEvent(new Event('visibilitychange'));
      });
      await page.clock.runFor(30_000);
      assert.equal(await count(), idle, 'no poll while the page tab is hidden');
      await context.close();
    }
    {
      const { page, context } = await openPage(browser, '/chat', {}, true);
      await page.waitForSelector('#modelSelect option', { state: 'attached' });
      await page.clock.pauseAt(Date.now() + 5000);
      const dashCalls = () => page.evaluate(() => window.__stub.calls.filter((call) => call.path === '/metrics.json' || (call.path === '/props' && call.search.includes('model='))).length);
      const base = await dashCalls();
      for (let i = 0; i < 10; i++) { await page.click('#navDashboard'); await page.click('#navChat'); }
      await page.click('#navDashboard');
      assert.ok(await dashCalls() - base <= 2, `rapid switches do not burst (sent ${await dashCalls() - base})`);
      await page.clock.runFor(2500);
      const later = await dashCalls();
      assert.ok(later > base, 'the poll waits out the minimum, then runs');
      assert.ok(later - base <= 4, `and still only once per interval (sent ${later - base})`);
      await context.close();
    }

    {
      const { page, context } = await openPage(browser, '/chat', { metricsStatus: 500 }, true);
      await page.waitForSelector('#modelSelect option', { state: 'attached' });
      const count = () => page.evaluate(() => window.__stub.calls.filter((call) => call.path === '/metrics.json').length);
      await page.clock.runFor(60_000);
      const n = await count();
      assert.ok(n >= 4 && n <= 12, `a hidden dashboard keeps retrying a failing feed with backoff (${n} calls)`);
      await page.evaluate(() => { window.__stub.metricsStatus = 200; });
      await page.clock.runFor(60_000);
      assert.ok(await count() > n, 'and is still polling after the feed recovers');
      await context.close();
    }

    for (const status of [503, 401]) {
      const { page, context } = await openPage(browser, '/chat', { metricsStatus: status }, true);
      await page.waitForSelector('#modelSelect option', { state: 'attached' });
      const count = () => page.evaluate(() => window.__stub.calls.filter((call) => call.path === '/metrics.json').length);
      await page.clock.runFor(60_000);
      assert.equal(await count(), 1, `a hidden dashboard stops after a ${status}`);
      await context.close();
    }

    {
      const { page, context } = await openPage(browser, '/dashboard', {}, true);
      await page.waitForFunction(() => /Live/.test(document.getElementById('dashStatus').textContent));
      await page.clock.pauseAt(Date.now() + 1000);
      const before = await page.evaluate(() => window.__clears);
      await page.evaluate(() => { document.documentElement.dataset.theme = document.documentElement.dataset.theme === 'night' ? 'day' : 'night'; });
      await page.waitForFunction((was) => window.__clears > was, before, { timeout: 2000 });
      await context.close();
    }
    console.log('Web UI dashboard: passed');
  } finally {
    await browser.close();
  }
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
