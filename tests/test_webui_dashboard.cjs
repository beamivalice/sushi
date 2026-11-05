// sushi-test: tier=hermetic needs=node,playwright
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
  const stub = window.__stub = { calls: [], aborted: 0, streams: 0, metricsStatus: config.metricsStatus, enableStatus: 200, openGate: null, ...config };
  const gate = new Promise((resolve) => { stub.openGate = resolve; });
  window.__clears = 0;
  window.__axis = {};
  const realClear = CanvasRenderingContext2D.prototype.clearRect;
  CanvasRenderingContext2D.prototype.clearRect = function (...args) { window.__clears += 1; window.__axis[this.canvas.id] = []; return realClear.apply(this, args); };
  const realText = CanvasRenderingContext2D.prototype.fillText;
  CanvasRenderingContext2D.prototype.fillText = function (text, ...args) { window.__axis[this.canvas.id]?.push(String(text)); return realText.call(this, text, ...args); };
  window.__dots = {};
  const realArc = CanvasRenderingContext2D.prototype.arc;
  CanvasRenderingContext2D.prototype.arc = function (x, y, r, ...rest) { (window.__dots[this.canvas.id] ??= []).push({ x, y, r }); return realArc.call(this, x, y, r, ...rest); };
  const realClearDots = CanvasRenderingContext2D.prototype.clearRect;
  CanvasRenderingContext2D.prototype.clearRect = function (...args) { window.__dots[this.canvas.id] = []; return realClearDots.apply(this, args); };
  if (config.key) localStorage.setItem('sushi-api-key', config.key);
  for (const [k, v] of Object.entries(config.storage ?? {})) localStorage.setItem(k, v);
  if (config.mockDpr) {
    // Chromium's emulation never fires the resolution query, so the test fires it: window.__setDpr(ratio).
    let dpr = 1;
    const listeners = new Set(), realMatch = window.matchMedia.bind(window);
    Object.defineProperty(window, 'devicePixelRatio', { get: () => dpr, configurable: true });
    window.matchMedia = (query) => {
      const mq = realMatch(query);
      if (/resolution/.test(query)) { mq.addEventListener = (type, fn) => { if (type === 'change') listeners.add(fn); }; mq.removeEventListener = (type, fn) => { listeners.delete(fn); }; }
      return mq;
    };
    window.__setDpr = (value) => { dpr = value; for (const fn of [...listeners]) fn(new Event('change')); };
  }
  if (config.sim) {
    // A server the test steers: finished requests, a restart, a skewed clock. Times are the server's own (browser time + skew).
    const skew = config.skew ?? 0, srv = () => Date.now() + skew;
    const fresh = () => ({ last: 0, started: null, next: 0, samples: [], reqs: [], seq: 0, running: 0, freeze: false, c: { prefill_tokens_total: 0, generation_tokens_total: 0, prompt_tokens_total: 0, prefix_cache_tokens_total: 0 }, h: { pre: { count: 0, sum: 0 }, dec: { count: 0, sum: 0 } } });
    const sim = window.__sim = {
      ...fresh(),
      init() { if (this.started == null) { this.started = srv() - (config.age ?? 0); this.first = this.next = srv(); } },
      tick() { this.init(); while (!this.freeze && this.next <= srv()) { this.samples.push([this.next, 0, 0, this.running, 0, 0, 0, 50, 50000, 2 ** 30, 2 ** 30]); this.next += 2000; } },
      finish({ computed, restored = 0, prefillS, gen = 0, decodeS = 0 }) {
        this.tick();
        this.c.prefill_tokens_total += computed; this.c.prompt_tokens_total += computed + restored; this.c.prefix_cache_tokens_total += restored; this.c.generation_tokens_total += gen;
        this.h.pre.count++; this.h.pre.sum += prefillS; this.h.dec.count++; this.h.dec.sum += decodeS;
        this.last = Math.max(this.last + 1, srv());
        this.reqs.push({ request_id: ++this.seq, model: 'org/test-model', t_ms: this.last, prompt_tokens: computed, cached_tokens: restored, prefill_seconds: prefillS, generated_tokens: gen, decode_seconds: decodeS, outcome: 'ok' });
      },
      restart() { this.tick(); Object.assign(this, fresh()); },
      body(since) {
        this.tick();
        const hist = (x) => ({ count: x.count, sum: x.sum, bounds: [1], bucket_counts: [x.count, x.count] });
        return {
          counters: { ...this.c },
          gauges: { requests_running: this.running, requests_waiting: 0, memory_mb: 50000, mlx_active_bytes: 2 ** 30, mlx_cache_bytes: 2 ** 30 },
          histograms: { prefill_time_seconds: hist(this.h.pre), decode_time_seconds: hist(this.h.dec) }, sessions: [],
          history: { interval_ms: 2000, capacity: 1800, started_ms: this.started, samples: this.samples.filter((r) => r[0] > since) },
          requests: this.reqs.filter((r) => r.t_ms > since),
        };
      },
    };
  }
  const json = (body, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
  const realFetch = window.fetch.bind(window);
  window.fetch = async (input, init = {}) => {
    const url = new URL(typeof input === 'string' ? input : input.url, location.href);
    if (url.origin !== location.origin) return realFetch(input, init);
    if (stub.down) throw new TypeError('Failed to fetch');
    const headers = new Headers(init.headers ?? {});
    stub.calls.push({ path: url.pathname, search: url.search, auth: headers.get('authorization'), method: init.method ?? 'GET', body: init.body });
    switch (url.pathname) {
      case '/health': return json({ status: 'ok' });
      case '/v1/models':
        if (stub.modelsDelay) await new Promise((r) => setTimeout(r, stub.modelsDelay));
        return json({ data: stub.noModels ? [] : [{ id: stub.modelId ?? 'org/test-model', state: 'ready', loaded: true, capabilities: ['chat'], context_length: 8192, bytes_resident: 3e9, meta: { kv_quant: 8 } }] });
      case '/props': return json({ settings: { version: '9.9.9', mtp: {}, prefix_cache: { disk_bytes: 20 * 2 ** 30, disk_used_bytes: 2 * 2 ** 30, disk_written_bytes: 4 * 2 ** 30, disk_entries: 3 } }, memory: { available_bytes: 5e9, total_bytes: 128 * 2 ** 30 }, batching: { supported: true, max_group: 4 }, default_generation_settings: { n_ctx: 8192 } });
      case '/v1/slots/cancel': return stub.cancelStatus === 404 ? json({ error: { message: 'No in-flight request matches that id' } }, 404) : json({ ok: true });
      case '/v1/tools': return json({ edit_default: false });
      case '/metrics/enable':
        if (stub.enableStatus !== 200) return json({ error: { message: 'refused' } }, stub.enableStatus);
        stub.metricsStatus = 200;
        return json({ metrics: true, already_enabled: false });
      case '/metrics.json':
        if (stub.metricsStatus !== 200) return json({ error: { message: 'metrics off' } }, stub.metricsStatus);
        if (stub.hold) await stub.hold;
        if (stub.sim) return json(window.__sim.body(Number(url.searchParams.get('since') || 0)));
        if (stub.compact) {
          // A decoding (or idle) server whose fourth poll sees one 12,000-token prefill finish after 10 s of prefilling.
          const polls = stub.polls = (stub.polls ?? 0) + 1, now = Date.now(), origin = stub.origin ??= now, done = polls >= 4 ? 1 : 0;
          const since = Number(url.searchParams.get('since') || 0);
          const samples = [];
          for (let k = -39; origin + 2000 * k <= now; k++) {
            const t = origin + 2000 * k, live = stub.compact === 'decode' && k >= -10;
            samples.push([t, live ? 100 * (k + 10) : 0, 0, +live, 0, 0, 0, 60, 106787, 113 * 2 ** 30, 2 ** 28]);
          }
          const hist = (count, sum) => ({ count, sum, bounds: [1], bucket_counts: [count, count] });
          return json({
            counters: { prefill_tokens_total: 40000 + 12000 * done, generation_tokens_total: 9000 },
            gauges: { requests_running: stub.compact === 'decode' ? 1 : 0, requests_waiting: 0, memory_mb: 106787, mlx_active_bytes: 113 * 2 ** 30, mlx_cache_bytes: 2 ** 28 },
            histograms: { prefill_time_seconds: hist(15 + done, 60 + 10 * done), decode_time_seconds: hist(15, 180) }, sessions: [],
            history: { interval_ms: 2000, capacity: 1800, started_ms: now - 3 * 3600e3 - 12 * 60e3, samples: samples.filter((r) => r[0] > since) },
            requests: done ? (stub.reqAt ??= now - 1000) > since ? [{ request_id: 12, model: 'org/test-model', t_ms: stub.reqAt, prompt_tokens: 12000, cached_tokens: 0, prefill_seconds: 10, generated_tokens: 0, decode_seconds: 0, outcome: 'ok' }] : [] : [],
          });
        }
        if (stub.history) {
          const since = Number(url.searchParams.get('since') || 0), t0 = Date.now() - 20000;
          const fields = ['t_ms', 'gen_tokens', 'prefill_tokens', 'running', 'waiting', 'prefilling', 'batched', 'gpu_pct', 'memory_mb', 'mlx_active', 'mlx_cache', 'gpu_mhz'];
          const samples = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((i) => [t0 + (i < 5 ? i : i + 15) * 1000, i * 40, 0, 1, 0, 0, 0, 50, 4096, 2 ** 30, 2 ** 30, stub.noClock ? 0 : 1200]).filter((r) => r[0] > since);
          const clock = stub.noClock ? { gpu_clock_mhz: 0, gpu_clock_max_mhz: 0 } : { gpu_clock_mhz: 1200, gpu_clock_max_mhz: 1620 };
          return json({ counters: { prompt_tokens_total: 10 }, gauges: { requests_running: 1, gpu_utilization_pct: 50, ...clock }, histograms: {}, sessions: [], history: { interval_ms: 2000, capacity: 1800, fields, samples } });
        }
        return json({ counters: { prompt_tokens_total: 10 }, gauges: { requests_running: 1, generation_tokens_live: stub.metricsTick = (stub.metricsTick ?? 0) + 40, prefill_tokens_live: stub.prefillTick = (stub.prefillTick ?? 0) + (stub.prefillStep ?? 0), gpu_utilization_pct: 55, memory_mb: 4096 }, histograms: {}, sessions: stub.sessions ?? [], prefix_cache_writes: { written_bytes: { kv: 2 ** 30 * stub.metricsTick, ssm: 2 ** 20, qsa: 0, spec: 0, ring: 0, meta: 1024 }, failed_bytes: 3 * 2 ** 20 } });
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

// Live intervals by delay and a count of armed timeouts; installed after load so it wraps the fake clock's timers.
function trackTimers() {
  const live = window.__intervals = new Map(), realSet = window.setInterval.bind(window), realClearI = window.clearInterval.bind(window), realTimeout = window.setTimeout.bind(window);
  window.__timeouts = 0;
  window.setInterval = (fn, ms, ...rest) => { const id = realSet(fn, ms, ...rest); live.set(id, ms); return id; };
  window.clearInterval = (id) => { live.delete(id); return realClearI(id); };
  window.setTimeout = (...args) => { window.__timeouts += 1; return realTimeout(...args); };
}

async function openPage(browser, urlPath, config = {}, fakeClock = false) {
  const context = await browser.newContext({ viewport: { width: 1280, height: 800 }, ...(config.contextOptions ?? {}) });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.route(`${ORIGIN}/**`, (route) => route.fulfill({ status: 200, contentType: 'text/html; charset=utf-8', body: html }));
  // The page opens in Compact; these checks read the Advanced cards unless a test asks for the other mode.
  await page.addInitScript((mode) => { try { localStorage.setItem('sushi-ui-mode', mode); } catch { /* storage off */ } }, config.mode ?? 'advanced');
  await page.addInitScript(installStubs, { metricsStatus: 200, firstWords: 'Streaming '.repeat(400), ...config });
  if (fakeClock) await page.clock.install();
  await page.goto(`${ORIGIN}${urlPath}`);
  return { page, errors, context };
}


// Steps the page's fake clock in short slices so each fetch lands between them.
async function adv(page, ms, step = 1000) {
  for (let left = ms; left > 0;) { const s = Math.min(step, left); left -= s; await page.clock.runFor(s); await page.waitForTimeout(30); }
}
const dotsOf = (page, id) => page.evaluate((i) => window.__dots[i] ?? [], id);
const textOf = (page, id) => page.evaluate((i) => document.getElementById(i).innerText.replace(/\s+/g, ' ').trim(), id);
const axisOf = (page, id) => page.evaluate((i) => window.__axis[i] ?? [], id);
const setHidden = (page, hidden) => page.evaluate((h) => { Object.defineProperty(document, 'hidden', { value: h, configurable: true }); document.dispatchEvent(new Event('visibilitychange')); }, hidden);
const metricsCalls = (page) => page.evaluate(() => window.__stub.calls.filter((c) => c.path === '/metrics.json'));
async function simPage(browser, config = {}, urlPath = '/dashboard') {
  const opened = await openPage(browser, urlPath, { mode: 'compact', sim: true, ...config }, true);
  await adv(opened.page, 4000);
  return opened;
}

// The Dashboard's honesty bugs, against a server the test steers (see `sim` in installStubs).
async function dashboardBugs(browser) {
  {
    // One dot per finished request, at its own finish time, however many finish between two polls.
    const { page, errors, context } = await simPage(browser);
    await page.selectOption('#dashWindow', '60');
    await page.evaluate(() => { window.__stub.hold = new Promise((r) => { window.__stub.release = r; }); });
    await adv(page, 2200, 200);
    await page.evaluate(() => window.__sim.finish({ computed: 3001, prefillS: 0.08 }));
    await adv(page, 1000, 200);
    await page.evaluate(() => { window.__sim.finish({ computed: 2001, prefillS: 0.5 }); window.__stub.hold = null; window.__stub.release(); });
    await adv(page, 3000);
    const dots = await dotsOf(page, 'dashCCpre');
    assert.equal(dots.length, 2, 'two requests between polls are two dots');
    assert.ok(dots[1].x > dots[0].x, 'each at its own finish time');
    assert.equal(await textOf(page, 'dashCmPre'), '4,002 tok/s', 'the tile is the newest request, not a pool');
    assert.match(await textOf(page, 'dashCmPreS'), /2,001 tokens/);
    await adv(page, 6000);
    assert.equal((await dotsOf(page, 'dashCCpre')).length, 2, 'later polls add no copies');
    assert.deepEqual(errors, []);
    await context.close();
  }
  {
    // A request finishing after the newest sample is delivered again until a sample passes it: still one dot.
    const { page, context } = await simPage(browser);
    await page.evaluate(() => { window.__sim.freeze = true; window.__sim.finish({ computed: 3001, prefillS: 0.08 }); });
    await adv(page, 8000);
    assert.equal((await dotsOf(page, 'dashCCpre')).length, 1, 'a re-sent request is one dot');
    await context.close();
  }
  {
    // A tab hidden for minutes keeps every request that finished meanwhile as its own dot.
    const { page, context } = await simPage(browser);
    await setHidden(page, true);
    for (let i = 0; i < 5; i++) { await page.evaluate(() => window.__sim.finish({ computed: 2000 + 100 * window.__sim.seq, prefillS: 1 })); await page.clock.runFor(30_000); }
    await setHidden(page, false);
    await adv(page, 3000);
    const xs = (await dotsOf(page, 'dashCCpre')).map((d) => d.x);
    assert.equal(xs.length, 5, 'five requests, five dots');
    assert.deepEqual(xs, [...xs].sort((a, b) => a - b));
    assert.equal(new Set(xs).size, 5);
    await context.close();
  }
  {
    // Pausing stops the polls; resuming fills in every request that finished meanwhile.
    const { page, context } = await simPage(browser);
    await page.click('#dashPause');
    const before = (await metricsCalls(page)).length;
    await page.evaluate(() => window.__sim.finish({ computed: 3000, prefillS: 1 }));
    await adv(page, 6000, 2000);
    await page.evaluate(() => window.__sim.finish({ computed: 4000, prefillS: 1 }));
    await adv(page, 6000, 2000);
    assert.equal((await metricsCalls(page)).length, before, 'no poll while paused');
    await page.click('#dashPause');
    await adv(page, 3000);
    assert.equal((await dotsOf(page, 'dashCCpre')).length, 2, 'both requests are there after resuming');
    await context.close();
  }
  {
    // A restart is the process start time changing, whatever the new counters say; the old server's request is gone.
    const { page, context } = await simPage(browser, { skew: 600_000 });
    await page.evaluate(() => window.__sim.finish({ computed: 5000, prefillS: 5, gen: 100, decodeS: 2 }));
    await adv(page, 3000);
    assert.equal(await textOf(page, 'dashCmPre'), '1,000 tok/s');
    assert.equal(await textOf(page, 'dashCmDec'), '50.0 tok/s');
    await page.evaluate(() => window.__sim.restart());
    await adv(page, 4000);
    assert.equal(await textOf(page, 'dashCmPre'), '-', 'the old server\'s last request is not shown');
    assert.equal(await textOf(page, 'dashCmDec'), '-');
    assert.equal((await dotsOf(page, 'dashCCpre')).length, 0, 'and its dot is gone');
    assert.match(await textOf(page, 'dashStatus'), /restarted/);
    assert.ok((await axisOf(page, 'dashCMemc')).includes('restart'), 'the marker is drawn at the server\'s time (this server\'s clock is 10 min ahead)');
    await context.close();
  }
  {
    // New counters above the old ones are still a restart: no dot pairs one server's request with the other's.
    const { page, context } = await simPage(browser);
    await page.evaluate(() => window.__sim.finish({ computed: 5000, prefillS: 5 }));
    await adv(page, 3000);
    await page.evaluate(() => { window.__sim.restart(); window.__sim.finish({ computed: 9000, prefillS: 3 }); });
    await adv(page, 4000);
    const dots = await dotsOf(page, 'dashCCpre');
    assert.equal(dots.length, 1, 'only the new server\'s request');
    assert.equal(await textOf(page, 'dashCmPre'), '3,000 tok/s');
    assert.match(await textOf(page, 'dashStatus'), /restarted/);
    await adv(page, 70_000, 5000);
    assert.equal(await textOf(page, 'dashStatus'), 'Live', 'the restart note does not stay for the session');
    await context.close();
  }
  {
    // Uptime is the process's, not the time since metrics (and the history) began.
    const { page, context } = await simPage(browser, { age: 3 * 3600e3 + 5 * 60e3 });
    assert.equal(await textOf(page, 'dashCmUp'), 'up 3h 5m');
    await context.close();
  }
  {
    // A prompt mostly restored from the cache is labelled as such and is not a prefill speed.
    const { page, context } = await simPage(browser);
    await page.evaluate(() => window.__sim.finish({ computed: 3001, prefillS: 0.08 }));
    await adv(page, 3000);
    await page.evaluate(() => window.__sim.finish({ computed: 71, restored: 2970, prefillS: 0.073 }));
    await adv(page, 3000);
    const dots = await dotsOf(page, 'dashCCpre');
    assert.equal(dots.length, 2);
    assert.equal(await textOf(page, 'dashCmPre'), '37,513 tok/s', 'the tile keeps the last real prefill');
    const box = await page.locator('#dashCCpre').boundingBox();
    await page.mouse.move(box.x + dots[1].x, box.y + 40);
    await adv(page, 100, 50);
    const tip = await textOf(page, 'dashTip');
    assert.match(tip, /71 computed, 2,970 restored/);
    assert.doesNotMatch(tip, /tok\/s/, 'no speed is claimed for it');
    await context.close();
  }
  {
    // A short reply still sets the Decode tile.
    const { page, context } = await simPage(browser);
    await page.evaluate(() => window.__sim.finish({ computed: 100, prefillS: 0.1, gen: 5, decodeS: 0.1 }));
    await adv(page, 3000);
    assert.equal(await textOf(page, 'dashCmDec'), '50.0 tok/s');
    assert.match(await textOf(page, 'dashCmDecS'), /last request/);
    await context.close();
  }
  {
    // Offline: the tiles say so instead of the last live numbers, and the charts dim.
    const { page, context } = await simPage(browser);
    await page.evaluate(() => { window.__sim.running = 1; });
    await adv(page, 4000);
    assert.equal(await textOf(page, 'dashCmReq'), '1 running', 'the gauge is shown while the server answers');
    await page.evaluate(() => { window.__stub.down = true; });
    await adv(page, 6000);
    for (const id of ['dashCmDec', 'dashCmPre', 'dashCmMem', 'dashCmSsd', 'dashCmReq']) assert.equal(await textOf(page, id), '-', `${id} shows nothing`);
    for (const id of ['dashCmDecS', 'dashCmPreS']) assert.equal(await textOf(page, id), 'server unreachable');
    assert.equal(await textOf(page, 'dashCmState'), 'Offline');
    assert.ok(await page.evaluate(() => +getComputedStyle(document.getElementById('dashCCdec')).opacity < 1), 'the charts are dimmed');
    await page.evaluate(() => { window.__stub.down = false; });
    await adv(page, 6000);
    assert.notEqual(await textOf(page, 'dashCmMem'), '-', 'and come back with the server');
    assert.equal(await page.evaluate(() => +getComputedStyle(document.getElementById('dashCCdec')).opacity), 1);
    await context.close();
  }
  {
    // Claims need a source: metrics off does not make the server Idle; no model does not make the SSD cache off.
    const off = await openPage(browser, '/dashboard', { mode: 'compact', metricsStatus: 503 }, true);
    await adv(off.page, 4000);
    assert.notEqual(await textOf(off.page, 'dashCmState'), 'Idle', 'idle is unknown without metrics');
    await off.context.close();
    const none = await openPage(browser, '/dashboard', { mode: 'compact', sim: true, noModels: true }, true);
    await adv(none.page, 4000);
    assert.equal(await textOf(none.page, 'dashCmSsd'), '-');
    assert.equal(await textOf(none.page, 'dashCmSsdS'), 'no model loaded');
    await none.context.close();
  }
  {
    // The window can widen again, also while paused, and an unknown stored window falls back.
    const { page, context } = await simPage(browser);
    await page.selectOption('#dashWindow', '60');
    await page.evaluate(() => window.__sim.finish({ computed: 3000, prefillS: 1 }));
    await adv(page, 90_000, 5000);
    assert.equal((await dotsOf(page, 'dashCCpre')).length, 0, 'the 1 min window forgets the request');
    await page.selectOption('#dashWindow', '900');
    await adv(page, 3000);
    assert.equal((await dotsOf(page, 'dashCCpre')).length, 1, 'widening brings it back');
    await page.click('#dashPause');
    await page.selectOption('#dashWindow', '3600');
    await adv(page, 100, 50);
    assert.ok((await axisOf(page, 'dashCCpre')).includes('-1h'), `the chart redraws while paused, labelled -1h (${await axisOf(page, 'dashCCpre')})`);
    assert.equal((await dotsOf(page, 'dashCCpre')).length, 1);
    await context.close();
    const bad = await openPage(browser, '/dashboard', { mode: 'compact', sim: true, storage: { 'sushi-dash:window': '120' } }, true);
    assert.equal(await bad.page.inputValue('#dashWindow'), '300', 'a window the list lacks falls back to 5 min');
    await bad.context.close();
  }
  {
    // A change made while a poll is in flight is not lost.
    const { page, context } = await simPage(browser, { mode: 'advanced' });
    await page.evaluate(() => { window.__stub.hold = new Promise((r) => { window.__stub.release = r; }); });
    const n = (await metricsCalls(page)).length;
    await adv(page, 2500, 500);
    assert.equal((await metricsCalls(page)).length, n + 1, 'a poll is in flight');
    await page.selectOption('#dashWindow', '900');
    await page.evaluate(() => { window.__stub.hold = null; window.__stub.release(); });
    await adv(page, 600, 100);
    const calls = await metricsCalls(page);
    assert.equal(calls[calls.length - 1].search, '?since=0', 'the wider window is fetched at once');
    await page.evaluate(() => { window.__stub.hold = new Promise((r) => { window.__stub.release = r; }); });
    await adv(page, 2500, 500);
    const m = (await metricsCalls(page)).length;
    await page.selectOption('#dashInterval', '10000');
    await page.evaluate(() => { window.__stub.hold = null; window.__stub.release(); });
    await adv(page, 6000, 500);
    assert.equal((await metricsCalls(page)).length, m + 1, 'one refresh at once, then the new interval');
    await context.close();
  }
  for (const height of [720, 600]) {
    // A tile's fly-over opens below the top bar and inside the window, whatever the window height.
    const { page, context } = await openPage(browser, '/dashboard', { contextOptions: { viewport: { width: 1280, height } } });
    await page.waitForSelector('[data-tile="mem"]');
    for (const tile of ['mem', 'cache']) {
      await page.hover(`[data-tile="${tile}"]`);
      await page.waitForFunction(() => getComputedStyle(document.getElementById('dashTip')).display === 'block', null, { timeout: 5000 });
      const box = await page.evaluate(() => { const b = document.getElementById('dashTip').getBoundingClientRect(); return { top: b.top, bottom: b.bottom, left: b.left, right: b.right, bar: document.getElementById('appTopbar').getBoundingClientRect().bottom, w: innerWidth, h: innerHeight }; });
      assert.ok(box.top >= box.bar, `${tile} at ${height}px: the fly-over starts below the top bar (${JSON.stringify(box)})`);
      assert.ok(box.bottom <= box.h && box.left >= 0 && box.right <= box.w, `${tile} at ${height}px: and stays inside the window (${JSON.stringify(box)})`);
      await page.mouse.move(2, height - 2);
    }
    await context.close();
  }
  {
    // Charts answer a tap and the keyboard, not only a mouse.
    const { page, context } = await simPage(browser, { contextOptions: { hasTouch: true } });
    await page.evaluate(() => window.__sim.finish({ computed: 3000, prefillS: 1 }));
    await adv(page, 3000);
    const dots = await dotsOf(page, 'dashCCpre');
    const box = await page.locator('#dashCCpre').boundingBox();
    const tip = () => page.evaluate(() => document.getElementById('dashTip').style.display);
    await page.touchscreen.tap(box.x + dots[0].x, box.y + dots[0].y);
    await adv(page, 100, 50);
    assert.equal(await tip(), 'block', 'a tap on a dot shows its tooltip');
    await adv(page, 2000);
    assert.equal(await tip(), 'block', 'and it stays until another tap');
    await page.touchscreen.tap(20, 20);
    await adv(page, 100, 50);
    assert.equal(await tip(), 'none', 'a tap elsewhere hides it');
    assert.equal(await page.evaluate(() => document.getElementById('dashCCpre').tabIndex), 0, 'charts take focus');
    await page.focus('#dashCCpre');
    await page.keyboard.press('ArrowLeft');
    await adv(page, 100, 50);
    assert.equal(await tip(), 'block', 'an arrow key steps to a point and shows it');
    await page.keyboard.press('Escape');
    await adv(page, 100, 50);
    assert.equal(await tip(), 'none');
    await context.close();
  }
  {
    // Nothing ticks once per second in a hidden tab.
    const { page, context } = await openPage(browser, '/chat', {}, true);
    await page.waitForSelector('#modelSelect option', { state: 'attached' });
    await page.evaluate(trackTimers);
    const fast = () => page.evaluate(() => [...window.__intervals.values()].filter((ms) => ms <= 1000).length);
    await page.click('#navDashboard');
    assert.equal(await fast(), 1);
    await setHidden(page, true);
    assert.equal(await fast(), 0, 'the hidden tab runs no 1 s interval');
    await setHidden(page, false);
    assert.equal(await fast(), 1, 'and it is back when the tab is');
    await context.close();
  }
  for (const mode of ['compact', 'advanced']) {
    // A poll that brings the same numbers rewrites no part of the page.
    const { page, context } = await simPage(browser, { mode });
    await adv(page, 20_000, 2000);
    await page.evaluate(() => {
      window.__mutations = [];
      new MutationObserver((rs) => { for (const r of rs) window.__mutations.push(`${r.type}:${r.attributeName ?? ''}:${r.target.id || r.target.className || r.target.nodeName}`); }).observe(document.getElementById('dashPanel'), { subtree: true, childList: true, attributes: true, characterData: true });
    });
    await adv(page, 10_000, 2000);
    const writes = (await page.evaluate(() => window.__mutations)).filter((m) => m !== 'attributes:title:dashDot');
    assert.deepEqual(writes, [], `${mode}: an unchanged poll writes nothing`);
    await context.close();
  }
  {
    // A device pixel ratio change redraws the charts sharply without a resize or a poll; a fractional one does not reallocate each frame.
    const { page, context } = await simPage(browser, { mockDpr: true });
    await page.click('#dashPause');
    const size = () => page.evaluate(() => { const c = document.getElementById('dashCCdec'); return [c.width, c.clientWidth]; });
    await page.evaluate(() => window.__setDpr(2));
    await adv(page, 300, 100);
    let [w, css] = await size();
    assert.equal(w, css * 2, 'the backing store follows the ratio');
    await page.evaluate(() => window.__setDpr(1.25));
    await adv(page, 300, 100);
    [w, css] = await size();
    assert.equal(w, Math.round(css * 1.25));
    await page.evaluate(() => {
      const c = document.getElementById('dashCCdec'), d = Object.getOwnPropertyDescriptor(HTMLCanvasElement.prototype, 'width');
      window.__widthSets = 0;
      Object.defineProperty(c, 'width', { get() { return d.get.call(this); }, set(v) { window.__widthSets += 1; d.set.call(this, v); } });
    });
    const box = await page.locator('#dashCCdec').boundingBox();
    for (let i = 0; i < 5; i++) { await page.mouse.move(box.x + 100 + i * 10, box.y + 50); await adv(page, 50, 50); }
    assert.equal(await page.evaluate(() => window.__widthSets), 0, 'a redraw at a fractional ratio keeps the canvas');
    await context.close();
  }
  {
    // The server's reason for a refused enable is shown.
    const { page, context } = await openPage(browser, '/dashboard', { metricsStatus: 503, enableStatus: 403 });
    await page.waitForFunction(() => !document.getElementById('dashOff').hidden);
    await page.click('#dashEnableMetrics');
    await page.waitForFunction(() => /refused/.test(document.getElementById('dashOffError').textContent));
    await context.close();
  }
  {
    // The first /props read runs before the model list arrives; the read for the loaded model must not wait out the 2 s throttle.
    const { page, context } = await openPage(browser, '/dashboard', { mode: 'compact', modelsDelay: 300 });
    await page.waitForFunction(() => /^2\.00 GiB \/ 20\.00 GiB$/.test(document.getElementById('dashCmSsd').innerText.trim()), null, { timeout: 1000 });
    await context.close();
  }
  {
    // On a phone a tile value keeps its unit on its line.
    const { page, context } = await simPage(browser, { contextOptions: { viewport: { width: 390, height: 844 } } });
    for (const id of ['dashCmMem', 'dashCmSsd']) {
      const fit = await page.evaluate((i) => { const v = document.getElementById(i); return { lines: Math.round(v.getBoundingClientRect().height / parseFloat(getComputedStyle(v).lineHeight)), over: v.scrollWidth - v.clientWidth }; }, id);
      assert.deepEqual(fit, { lines: 1, over: 0 }, `${id} fits one line at 390 px`);
    }
    await context.close();
  }
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
    await dashboardBugs(browser);
    {
      // A running request has a Cancel button that posts its id, with the page's API key; a cached conversation has none.
      const row = (request_id, phase) => ({ request_id, model: 'org/test-model', phase, context_length: 8192, context_tokens: 100, cached_tokens: 0, generated_tokens: 5, elapsed_seconds: 3, state_bytes: 1 << 20 });
      const sessions = [row(7, 'decode'), row(8, 'cached')];
      const { page, errors, context } = await openPage(browser, '/dashboard', { sessions, key: 'k-12345' });
      await page.waitForSelector('#dashSessions button[data-cancel]');
      assert.equal(await page.locator('#dashSessions button').count(), 1, 'only the decode row has a button');
      await page.click('#dashSessions button[data-cancel="7"]');
      await page.waitForFunction(() => document.querySelector('#dashSessions button')?.textContent === 'Cancelled');
      const posts = await page.evaluate(() => window.__stub.calls.filter((c) => c.path === '/v1/slots/cancel'));
      assert.deepEqual(posts.map((c) => [c.method, JSON.parse(c.body), c.auth]), [['POST', { id: 7 }, 'Bearer k-12345']]);
      assert.equal(await page.isDisabled('#dashSessions button'), true, 'the button stays off once clicked');
      assert.deepEqual(errors, []);
      await context.close();
    }
    {
      // A duration rounds as a whole: 119.7 s is 2m 0s, never 1m 60s.
      const row = (request_id, elapsed_seconds) => ({ request_id, model: 'org/test-model', phase: 'decode', context_length: 8192, generated_tokens: 5, elapsed_seconds });
      const { page, context } = await openPage(browser, '/dashboard', { sessions: [row(1, 119.7), row(2, 59.996), row(3, 0.9996)] });
      await page.waitForSelector('#dashSessions button[data-cancel]');
      assert.deepEqual(await page.$$eval('#dashSessions tr', (rows) => rows.map((r) => r.children[7].textContent)), ['2m 0s', '1m 0s', '1.00 s']);
      await context.close();
    }
    {
      // A gridline's label is that line's value: a 0 to 1 scale reads 0.25 and 0.75, a 2,500 one 1.875k, never a rounded neighbour.
      const { page, context } = await openPage(browser, '/dashboard', { history: true });
      await page.waitForFunction(() => (window.__axis.dashCReq ?? []).includes('now'));
      assert.deepEqual((await axisOf(page, 'dashCReq')).slice(0, 5), ['0', '0.25', '0.5', '0.75', '1']);
      await context.close();
      const sim = await simPage(browser);
      await sim.page.evaluate(() => window.__sim.finish({ computed: 5000, prefillS: 2 }));
      await adv(sim.page, 3000);
      assert.deepEqual((await axisOf(sim.page, 'dashCCpre')).slice(0, 5), ['0', '625', '1.25k', '1.875k', '2.5k']);
      await sim.context.close();
    }
    {
      const { page, context } = await openPage(browser, '/dashboard', { sessions: [{ request_id: 9, model: 'org/test-model', phase: 'prefill', context_length: 8192 }], cancelStatus: 404 });
      await page.click('#dashSessions button[data-cancel="9"]');
      await page.waitForFunction(() => document.querySelector('#dashSessions button')?.textContent === 'Failed');
      assert.equal(await page.getAttribute('#dashSessions button', 'title'), 'No in-flight request matches that id', 'the server\'s message rides the title');
      await context.close();
    }
    {
      const { page, errors, context } = await openPage(browser, '/chat');
      await page.waitForSelector('#modelSelect option', { state: 'attached' });
      assert.equal(await page.isVisible('#chatPanel'), true, 'chat is the view at /chat');
      assert.equal(await page.isVisible('#dashPanel'), false, 'the dashboard starts hidden');
      assert.equal(await page.getAttribute('#navChat', 'aria-current'), 'page');
      assert.equal(await page.evaluate(() => document.getElementById('dashCPre').width), 300, 'a hidden dashboard draws nothing');

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
      await page.waitForFunction(() => document.getElementById('serverDotLabel').textContent.includes('test-model'), null, { timeout: 5000 });
      assert.notEqual(await page.evaluate(() => document.getElementById('dashCPre').width), 300, 'a visible dashboard draws its charts');

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
      await page.waitForFunction(() => { const o = document.getElementById('dashOff'); return !o.hidden && /--no-metrics/.test(o.textContent); });
      await page.waitForFunction(() => document.getElementById('serverDotLabel').textContent.includes('test-model'));
      await context.close();
    }

    {
      const { page, errors, context } = await openPage(browser, '/dashboard', { metricsStatus: 503, key: 'k-12345' });
      await page.waitForFunction(() => { const o = document.getElementById('dashOff'); return !o.hidden && /--no-metrics/.test(o.textContent); });
      assert.equal(await page.isVisible('#dashEnableMetrics'), true, 'the button shows while metrics are off');
      const enables = () => page.evaluate(() => window.__stub.calls.filter((call) => call.path === '/metrics/enable'));
      assert.equal((await enables()).length, 0, 'nothing is enabled by itself');
      await page.click('#dashEnableMetrics');
      await page.waitForFunction(() => /Live/.test(document.getElementById('dashStatus').textContent));
      const sent = await enables();
      assert.equal(sent.length, 1, 'one enable request');
      assert.equal(sent[0].method, 'POST');
      assert.equal(sent[0].auth, 'Bearer k-12345', 'it carries the chat key');
      assert.equal(await page.isVisible('#dashEnableMetrics'), false, 'the button is gone once metrics are on');
      await page.waitForFunction(() => document.getElementById('dashTGpu').textContent.includes('55'));
      assert.equal(await page.evaluate(() => performance.getEntriesByType('navigation').length), 1, 'no page reload');
      assert.deepEqual(errors, []);
      await context.close();
    }

    {
      const { page, context } = await openPage(browser, '/dashboard', { metricsStatus: 503, enableStatus: 500 });
      await page.waitForFunction(() => { const o = document.getElementById('dashOff'); return !o.hidden && /--no-metrics/.test(o.textContent); });
      await page.click('#dashEnableMetrics');
      await page.waitForFunction(() => /Could not enable metrics/.test(document.getElementById('dashStatus').textContent));
      assert.equal(await page.isVisible('#dashEnableMetrics'), true, 'a refused enable leaves the button for another try');
      assert.equal(await page.isEnabled('#dashEnableMetrics'), true);
      await context.close();
    }

    for (const metricsStatus of [200, 401]) {
      const { page, context } = await openPage(browser, '/dashboard', { metricsStatus });
      await page.waitForFunction(() => window.__stub.calls.some((call) => call.path === '/metrics.json'));
      await page.waitForTimeout(300);
      assert.equal(await page.isVisible('#dashEnableMetrics'), false, `no button when the feed answers ${metricsStatus}`);
      await context.close();
    }

    {
      const { page, context } = await openPage(browser, '/chat', { metricsStatus: 503 });
      await page.waitForSelector('#modelSelect option', { state: 'attached' });
      await page.fill('#promptInput', 'keep me');
      await page.click('#navDashboard');
      await page.waitForFunction(() => { const o = document.getElementById('dashOff'); return !o.hidden && /--no-metrics/.test(o.textContent); });
      await page.click('#dashEnableMetrics');
      await page.waitForFunction(() => /Live/.test(document.getElementById('dashStatus').textContent));
      await page.click('#navChat');
      assert.equal(await page.inputValue('#promptInput'), 'keep me', 'enabling from the dashboard leaves the chat composer alone');
      assert.equal(await page.evaluate(() => document.getElementById('chatPanel').inert), false);
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
      const { page, errors, context } = await openPage(browser, '/dashboard');
      const rows = () => page.evaluate(() => Object.fromEntries([...document.querySelectorAll('#dashWrites span')].reduce((acc, el, i, all) => (i % 2 ? acc : [...acc, [el.textContent, all[i + 1].textContent]]), [])));
      await page.waitForFunction(() => /Live/.test(document.getElementById('dashStatus').textContent));
      // The rate needs two polls and the ratio needs /props: wait for both, then read one consistent render.
      await page.waitForFunction(() => {
        const spans = [...document.querySelectorAll('#dashWrites span')];
        const value = (label) => spans[spans.findIndex((el) => el.textContent === label) + 1]?.textContent ?? '';
        return /\/s$/.test(value('Write rate')) && /x$/.test(value('Written per byte on disk'));
      }, null, { timeout: 15000 });
      const card = await rows();
      for (const label of ['Cache location', 'Written', 'Write rate', 'Written per byte on disk', 'KV chunks', 'Restore points', 'QSA history', 'Speculative state', 'Ring points', 'Manifests and token records', 'Failed or dropped']) {
        assert.ok(label in card, `the card has a ${label} row`);
      }
      assert.equal(card['Failed or dropped'], '3.0 MiB', 'failed bytes are named apart from the written ones');
      const gib = (text) => parseFloat(text) * (/GiB/.test(text) ? 1 : /MiB/.test(text) ? 1 / 1024 : 1 / 1048576);
      assert.ok(Math.abs(parseFloat(card['Written per byte on disk']) - gib(card['Written']) / 2) < 0.06, 'written over the 2 GiB on disk');
      assert.deepEqual(errors, []);
      await context.close();
    }

    {
      const { page, errors, context } = await openPage(browser, '/dashboard', { compact: 'decode' });
      const axisTop = (id) => page.evaluate((canvas) => Math.max(0, ...(window.__axis[canvas] ?? []).filter((t) => /^\d+(\.\d+)?k?$/.test(t)).map((t) => parseFloat(t) * (t.endsWith('k') ? 1000 : 1))), id);
      await page.waitForFunction(() => /Live/.test(document.getElementById('dashStatus').textContent));
      const cards = await page.evaluate(() => [...document.querySelectorAll('#dashPanel .charts.adv .card')].map((card) => ({ title: card.querySelector('h2').firstChild.textContent, legends: card.querySelectorAll('.legend').length, canvas: card.querySelector('canvas').id })));
      assert.deepEqual(cards.slice(0, 2).map((card) => card.title), ['Prefill (tok/s)', 'Decode (tok/s)'], 'prefill and decode are separate cards');
      assert.deepEqual(cards.slice(0, 2).map((card) => card.legends), [0, 0], 'a single series needs no legend');
      assert.equal(new Set(cards.map((card) => card.canvas)).size, cards.length, 'every card draws on its own canvas');
      assert.equal(await page.evaluate(() => /\bgeneration\b/i.test(document.getElementById('dashPanel').textContent)), false, 'the dashboard says decode, not generation');
      // The Advanced view draws prefill and decode the way the Compact view does: one dot per finished request, decode on its own scale.
      assert.deepEqual(await page.evaluate(() => [...document.querySelectorAll('#dashPanel .charts.adv .card .sub')].map((e) => e.textContent).slice(0, 2)), ['one point per finished request', 'while generating']);
      await page.waitForFunction((id) => (window.__axis[id] ?? []).includes('2k'), cards[0].canvas, { timeout: 15000 });
      await page.waitForFunction((id) => (window.__axis[id] ?? []).length > 5, cards[1].canvas, { timeout: 15000 });
      const axis = await page.evaluate(() => window.__axis.dashCMem ?? []);
      assert.ok(axis.some((t) => /GiB machine memory/.test(t)), `the memory chart marks the machine's memory (${axis})`);
      const pre = await axisTop(cards[0].canvas), dec = await axisTop(cards[1].canvas);
      assert.ok(dec > 0 && dec * 4 <= pre, `each chart scales to its own series (prefill top ${pre}, decode top ${dec})`);
      assert.deepEqual(errors, []);
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
    {
      // Compact is the default: the essentials are visible, the detailed cards are not, and the header toggle flips it.
      const { page, errors, context } = await openPage(browser, '/dashboard', { mode: 'compact' });
      await page.waitForFunction(() => /Live/.test(document.getElementById('dashStatus').textContent));
      const visible = (sel) => page.evaluate((s) => { const n = document.querySelector(s); return !!n && n.getClientRects().length > 0; }, sel);
      assert.equal(await visible('#dashCmDec'), true, 'the compact tiles show');
      assert.equal(await visible('#dashCCpre') && await visible('#dashCCdec'), true, 'and the prefill and decode charts');
      assert.equal(await visible('#dashSessions'), false, 'the sessions table is an Advanced detail');
      assert.equal(await visible('#dashCReq'), false, 'so are the request and GPU charts');
      assert.equal(await page.getAttribute('#modeToggle [data-mode="compact"]', 'aria-pressed'), 'true', 'the switch shows the current mode');
      await page.click('#modeToggle [data-mode="advanced"]');
      assert.equal(await page.evaluate(() => document.body.dataset.mode), 'advanced');
      assert.equal(await visible('#dashSessions'), true);
      assert.equal(await visible('#dashCmDec'), false);
      assert.equal(await page.evaluate(() => localStorage.getItem('sushi-ui-mode')), 'advanced', 'the choice is remembered');
      assert.deepEqual(errors, []);
      await context.close();
    }

    for (const compact of ['decode', 'idle']) {
      // The Compact view: a one-line status strip, five equal tiles, and prefill and decode on charts of their own.
      const name = 'DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw-with-a-name-longer-than-any-strip-could-hold-on-one-line';
      const { page, errors, context } = await openPage(browser, '/dashboard', { mode: 'compact', compact, modelId: `org/${name}` });
      await page.setViewportSize({ width: 1440, height: 900 });
      await page.waitForFunction(() => /Live/.test(document.getElementById('dashStatus').textContent));
      await page.waitForFunction(() => document.getElementById('dashCmUp').textContent.startsWith('up 3h'), null, { timeout: 15000 });
      await page.waitForFunction(() => document.getElementById('dashCmState').textContent !== 'Connecting', null, { timeout: 15000 });
      const box = (sel) => page.evaluate((s) => { const r = document.querySelector(s).getBoundingClientRect(); return { w: Math.round(r.width), h: Math.round(r.height), x: Math.round(r.x), y: Math.round(r.y) }; }, sel);
      const text = (id) => page.evaluate((i) => document.getElementById(i).innerText.replace(/\s+/g, ' ').trim(), id);

      const strip = await box('.cm-strip');
      assert.ok(strip.h < 70, `a long model name does not grow the strip (${strip.h} px)`);
      assert.equal(await page.getAttribute('#dashCmModel', 'title'), name, 'the full name rides the title');
      assert.equal(await page.evaluate(() => getComputedStyle(document.getElementById('dashCmModel')).textOverflow), 'ellipsis');
      assert.equal(await text('dashCmState'), compact === 'decode' ? 'Decoding' : 'Idle');

      const tiles = await page.evaluate(() => [...document.querySelectorAll('.tiles-cmp .tile')].map((t) => { const r = t.getBoundingClientRect(); return { h: Math.round(r.height), y: Math.round(r.y) }; }));
      assert.equal(tiles.length, 5);
      assert.equal(new Set(tiles.map((t) => t.h)).size, 1, `the tiles are one height (${tiles.map((t) => t.h)})`);
      assert.equal(new Set(tiles.map((t) => t.y)).size, 1, 'on one row at 1440 px');
      assert.ok(tiles[0].h < 160, `and compact (${tiles[0].h} px)`);

      const pre = await box('#dashCCpre'), dec = await box('#dashCCdec');
      assert.deepEqual([pre.w, pre.h], [dec.w, dec.h], 'the prefill and decode charts are the same size');
      assert.ok(Math.abs(pre.y - dec.y) < 2 && dec.x > pre.x, 'side by side on a wide screen');
      assert.equal(await page.evaluate(() => !!document.getElementById('dashCSpd')), false, 'no chart mixes the two speeds');

      // Decode: live from the history while generating; idle falls back to the since-start mean, labelled as such.
      assert.equal(await text('dashCmDec'), '50.0 tok/s');
      assert.equal(await text('dashCmDecS'), compact === 'decode' ? 'live' : 'average since start');
      // Prefill is a per-request speed, never an instantaneous 0: the mean until a request is seen to finish (40,000 tokens
      // in 60 s), then that request's own (12,000 tokens in 10 s).
      assert.notEqual(await text('dashCmPre'), '0 tok/s');
      assert.match(await text('dashCmPre'), /^667 tok\/s$/);
      await page.waitForFunction(() => /^1,200 tok\/s$/.test(document.getElementById('dashCmPre').innerText.trim()), null, { timeout: 15000 });
      assert.match(await text('dashCmPreS'), /last request, .*12,000 tokens/);
      await page.waitForFunction(() => (window.__axis.dashCCpre ?? []).includes('2k'), null, { timeout: 5000 });
      assert.deepEqual(await page.evaluate(() => (window.__axis.dashCCpre ?? []).filter((t) => /request finished/.test(t))), [], 'the prefill chart plots the finished request');

      assert.match(await text('dashCmMem'), /^104\.3 \/ 128 GiB$/);
      assert.ok(await page.evaluate(() => parseFloat(document.getElementById('dashCmMemBar').style.width)) > 80, 'the memory bar is used over the machine total');
      assert.match(await text('dashCmSsd'), /^2\.00 GiB \/ 20\.00 GiB$/);
      assert.match(await text('dashCmReq'), compact === 'decode' ? /^1 running$/ : /^0 running$/);
      const axis = await page.evaluate(() => window.__axis.dashCMemc ?? []);
      assert.ok(axis.includes('128') && !axis.includes('200'), `the memory axis ends at the machine's memory (${axis})`);
      assert.ok(axis.some((t) => /128 GiB machine memory/.test(t)), 'and the ceiling is labelled');

      // Narrow: decode and prefill share the first row, the rest stack; the strip still one line, nothing wider than the screen.
      await page.setViewportSize({ width: 390, height: 844 });
      await page.waitForTimeout(300);
      const narrow = await page.evaluate(() => ({ strip: document.querySelector('.cm-strip').getBoundingClientRect().height, wide: document.getElementById('dashPanel').scrollWidth - document.getElementById('dashPanel').clientWidth,
        tiles: new Set([...document.querySelectorAll('.tiles-cmp .tile')].map((t) => Math.round(t.getBoundingClientRect().x))).size,
        firstRow: [...document.querySelectorAll('.tiles-cmp .tile')].filter((t, _, all) => Math.round(t.getBoundingClientRect().y) === Math.round(all[0].getBoundingClientRect().y)).length,
        charts: [...document.querySelectorAll('.charts-cmp .card')].map((c) => Math.round(c.getBoundingClientRect().x)) }));
      assert.ok(narrow.strip < 70 && narrow.wide <= 0 && narrow.tiles === 2 && narrow.firstRow === 2, `390 px stacks without overflow (${JSON.stringify(narrow)})`);
      assert.equal(new Set(narrow.charts).size, 1, 'the charts stack');
      assert.deepEqual(errors, []);
      await context.close();
    }

    {
      // The charts take their points from the server's history: a page that never polled while hidden still shows every
      // sample, each at the server's time, and a long silence between two samples stays a gap.
      const { page, errors, context } = await openPage(browser, '/dashboard', { mode: 'advanced', history: true });
      await page.waitForFunction(() => /Live/.test(document.getElementById('dashStatus').textContent));
      await page.waitForFunction(() => window.__stub.calls.some((call) => call.path === '/metrics.json' && /since=\d{6,}/.test(call.search)), null, { timeout: 15000 });
      assert.equal(await page.evaluate(() => window.__stub.calls.find((call) => call.path === '/metrics.json').search), '?since=0', 'the first poll asks for everything');
      assert.deepEqual(errors, []);
      await context.close();
    }
    for (const mode of ['advanced', 'compact']) {
      // The GPU chart pairs the clock (left axis, up to the table's top) with utilization (right axis, 0 to 100%); the
      // history backfill carries both, and a box with no readable clock says so and keeps the utilization.
      const id = mode === 'advanced' ? 'dashCGpu' : 'dashCGpuc';
      const hoverTip = async (page) => {
        const box = await page.evaluate((i) => { const b = document.getElementById(i).getBoundingClientRect(); return { x: b.left, y: b.top, w: b.width, h: b.height }; }, id);
        await page.mouse.move(box.x + box.w - 60, box.y + box.h / 2);
        await page.waitForFunction(() => getComputedStyle(document.getElementById('dashTip')).display === 'block', null, { timeout: 5000 });
        return page.evaluate(() => document.getElementById('dashTip').innerText.replace(/\s+/g, ' '));
      };
      const { page, errors, context } = await openPage(browser, '/dashboard', { mode, history: true });
      await page.waitForFunction((i) => (window.__axis[i] ?? []).includes('1,620'), id, { timeout: 15000 });
      const axis = await axisOf(page, id);
      assert.ok(axis.includes('100%') && axis.includes('0%'), `${mode}: the right axis is utilization 0 to 100% (${axis})`);
      assert.ok(await page.evaluate(() => document.getElementById('dashCGpu').closest('.card').querySelector('.legend').innerText.includes('clock')), `${mode}: the legend names the clock`);
      const tip = await hoverTip(page);
      assert.match(tip, /GPU clock\s*1,200 MHz/, `${mode}: the tooltip names the clock from the backfilled samples (${tip})`);
      assert.match(tip, /Utilization\s*50%/, `${mode}: and the utilization (${tip})`);
      assert.equal(await textOf(page, mode === 'advanced' ? 'dashCGpuL' : 'dashCGpucL'), '', `${mode}: no unavailable note`);
      assert.deepEqual(errors, []);
      await context.close();

      const off = await openPage(browser, '/dashboard', { mode, history: true, noClock: true });
      await off.page.waitForFunction((i) => (off => off.length > 4)(window.__axis[i] ?? []), id, { timeout: 15000 });
      const offAxis = await axisOf(off.page, id);
      assert.ok(!offAxis.includes('1,620') && offAxis.includes('100'), `${mode}: without a clock the axis is utilization alone (${offAxis})`);
      assert.match(await textOf(off.page, mode === 'advanced' ? 'dashCGpuL' : 'dashCGpucL'), /clock unavailable/, `${mode}: the chart says the clock is unavailable`);
      const offTip = await hoverTip(off.page);
      assert.ok(!/GPU clock/.test(offTip) && /Utilization\s*50%/.test(offTip), `${mode}: the tooltip shows utilization only (${offTip})`);
      assert.deepEqual(off.errors, []);
      await off.context.close();
    }

    {
      // Hovering a chart costs one redraw per frame and reads no style per colour.
      const { page, errors, context } = await openPage(browser, '/dashboard', { compact: 'decode', mode: 'compact' });
      await page.waitForFunction(() => /Live/.test(document.getElementById('dashStatus').textContent));
      await page.waitForTimeout(300);
      const cost = await page.evaluate(async () => {
        let styleReads = 0;
        const real = window.getComputedStyle;
        window.getComputedStyle = (...args) => { styleReads += 1; return real.apply(window, args); };
        const clears0 = window.__clears, canvas = document.getElementById('dashCCdec'), box = canvas.getBoundingClientRect();
        for (let i = 0; i < 20; i++) canvas.dispatchEvent(new PointerEvent('pointermove', { clientX: box.left + 100 + i, clientY: box.top + 60, bubbles: true }));
        await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        window.getComputedStyle = real;
        return { styleReads, clears: window.__clears - clears0 };
      });
      assert.ok(cost.clears <= 2, `20 pointer moves redraw the hovered chart once (${cost.clears} clears)`);
      assert.ok(cost.styleReads <= 3, `a redraw reads no style per colour (${cost.styleReads} reads)`);
      assert.deepEqual(errors, []);
      await context.close();
    }

    {
      // A Compact poll does not rebuild the Advanced cards; switching to Advanced fills them at once.
      const { page, errors, context } = await openPage(browser, '/dashboard', { compact: 'decode', mode: 'compact' });
      await page.waitForFunction(() => /Live/.test(document.getElementById('dashStatus').textContent));
      await page.evaluate(() => { window.__sanitized = 0; const real = DOMPurify.sanitize.bind(DOMPurify); DOMPurify.sanitize = (...args) => { window.__sanitized += 1; return real(...args); }; });
      const seen = await page.evaluate(() => window.__stub.calls.filter((call) => call.path === '/metrics.json').length);
      await page.waitForFunction((n) => window.__stub.calls.filter((call) => call.path === '/metrics.json').length > n + 1, seen, { timeout: 10000 });
      assert.equal(await page.evaluate(() => window.__sanitized), 0, 'Compact polls build no Advanced markup');
      await page.click('#modeToggle [data-mode="advanced"]');
      assert.ok(await page.evaluate(() => document.getElementById('dashLat').children.length > 0), 'Advanced is filled the moment it is shown');
      assert.deepEqual(errors, []);
      await context.close();
    }

    {
      // Nothing ticks for a view that is not shown, and a hidden tab arms no timers.
      const { page, context } = await openPage(browser, '/chat', {}, true);
      await page.waitForSelector('#modelSelect option', { state: 'attached' });
      await page.evaluate(trackTimers);
      const fast = () => page.evaluate(() => [...window.__intervals.values()].filter((ms) => ms <= 1000).length);
      assert.equal(await fast(), 0, 'the chat view runs no 1 s interval');
      await page.click('#navDashboard');
      assert.equal(await fast(), 1, 'the dashboard runs its server poll while shown');
      await page.click('#navChat');
      assert.equal(await fast(), 0, 'and drops it when it is hidden');
      await page.evaluate(() => {
        Object.defineProperty(document, 'hidden', { value: true, configurable: true });
        document.dispatchEvent(new Event('visibilitychange'));
        window.__timeouts = 0; window.__healthBefore = window.__stub.calls.filter((call) => call.path === '/health').length;
      });
      await page.clock.runFor(60_000);
      const health = await page.evaluate(() => [window.__stub.calls.filter((call) => call.path === '/health').length, window.__healthBefore, window.__timeouts]);
      assert.equal(health[0], health[1], 'a hidden tab does not ping /health');
      assert.ok(health[2] <= 3, `a hidden tab arms no polling timers (${health[2]})`);
      await context.close();
    }

    for (const [scheme, theme, bg, chart] of [['dark', 'system', 'rgb(15, 17, 21)', 'rgb(255, 156, 126)'], ['light', 'system', 'rgb(247, 245, 242)', 'rgb(220, 90, 54)'], ['light', 'night', 'rgb(15, 17, 21)', 'rgb(255, 156, 126)'], ['dark', 'day', 'rgb(247, 245, 242)', 'rgb(220, 90, 54)']]) {
      // The stored theme wins over the system scheme; the page and the chart palette follow it.
      const context = await browser.newContext({ colorScheme: scheme });
      const page = await context.newPage();
      await page.route(`${ORIGIN}/**`, (route) => route.fulfill({ status: 200, contentType: 'text/html; charset=utf-8', body: html }));
      await page.addInitScript((value) => { try { localStorage.setItem('sushi-theme', value); } catch { /* storage off */ } }, theme);
      await page.addInitScript(installStubs, { metricsStatus: 200 });
      await page.goto(`${ORIGIN}/dashboard`);
      const seen = await page.evaluate(() => { const probe = document.createElement('i'); probe.style.color = 'var(--c1)'; document.getElementById('dashPanel').append(probe); return { bg: getComputedStyle(document.body).backgroundColor, chart: getComputedStyle(probe).color, meta: document.querySelector('meta[name=theme-color]').content }; });
      assert.deepEqual(seen, { bg, chart, meta: bg }, `${theme} theme under a ${scheme} system`);
      await context.close();
    }
    console.log('Web UI dashboard: passed');
  } finally {
    await browser.close();
  }
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
