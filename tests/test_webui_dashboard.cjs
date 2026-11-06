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
            gauges: { requests_running: ['idle', 'waiting'].includes(stub.compact) ? 0 : 1, requests_prefilling: stub.compact === 'prefill' ? 1 : 0, requests_waiting: stub.compact === 'waiting' ? 1 : 0, memory_mb: 106787, mlx_active_bytes: 113 * 2 ** 30, mlx_cache_bytes: 2 ** 28 },
            histograms: { prefill_time_seconds: hist(15 + done, 60 + 10 * done), decode_time_seconds: hist(15, 180) }, sessions: [],
            history: { interval_ms: 2000, capacity: 1800, started_ms: now - 3 * 3600e3 - 12 * 60e3, samples: samples.filter((r) => r[0] > since) },
            requests: done ? (stub.reqAt ??= now - 1000) > since ? [{ request_id: 12, model: 'org/test-model', t_ms: stub.reqAt, prompt_tokens: 12000, cached_tokens: 0, prefill_seconds: 10, generated_tokens: 0, decode_seconds: 0, outcome: 'ok' }] : [] : [],
          });
        }
        if (stub.history) {
          const since = Number(url.searchParams.get('since') || 0), t0 = Date.now() - 20000;
          const fields = ['t_ms', 'gen_tokens', 'prefill_tokens', 'running', 'waiting', 'prefilling', 'batched', 'gpu_pct', 'memory_mb', 'mlx_active', 'mlx_cache', 'gpu_mhz'];
          const samples = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((i) => [t0 + (i < 5 ? i : i + 15) * 1000, i * 40, 0, 1, 0, 0, 0, 50, 4096, 2 ** 30, 2 ** 30, stub.noClock ? 0 : 1200]).map((r) => (r[3] = stub.runningCount ?? 1, r)).filter((r) => r[0] > since);
          const clock = stub.noClock ? { gpu_clock_mhz: 0, gpu_clock_max_mhz: 0 } : { gpu_clock_mhz: 1200, gpu_clock_max_mhz: 1620 };
          return json({ counters: { prompt_tokens_total: 10 }, gauges: { requests_running: 1, gpu_utilization_pct: 50, ...clock }, histograms: {}, sessions: [], history: { interval_ms: 2000, capacity: 1800, fields, samples } });
        }
        return json({ counters: { prompt_tokens_total: 10 }, gauges: { requests_running: 1, generation_tokens_live: stub.metricsTick = (stub.metricsTick ?? 0) + 40, prefill_tokens_live: stub.prefillTick = (stub.prefillTick ?? 0) + (stub.prefillStep ?? 0), gpu_utilization_pct: 55, memory_mb: 4096 }, histograms: {}, sessions: stub.sessions ?? [], prefix_cache_writes: { written_bytes: { kv: 2 ** 30 * stub.metricsTick, ssm: 2 ** 20, qsa: 0, spec: 0, ring: 0, meta: 1024 }, failed_bytes: 3 * 2 ** 20 }, ...(stub.specData ?? {}) });
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

// The speculative, tier and request-log members a server with drafting and a prefix cache reports.
function specFixture() {
  const zero = { requests: 0, rounds: 0, drafted: 0, accepted: 0, plain_rounds: 0 };
  return {
    speculative: { mtp: { requests: 4, rounds: 40, drafted: 120, accepted: 84, plain_rounds: 3 }, dflash: zero, dspark: { requests: 2, rounds: 20, drafted: 100, accepted: 60, plain_rounds: 5, rows: { 1: 5, 3: 10, 5: 5 } } },
    prefix_cache_tiers: { hits: { hot: 3, ssd: 9 }, evictions: { hot: 1, ssd: 2 }, evicted_bytes: { hot: 2 ** 20, ssd: 8 * 2 ** 20 }, hot_bytes: 2 ** 30 },
    requests: [{ request_id: 1, model: 'org/test-model', t_ms: Date.now() - 3000, prompt_tokens: 100, cached_tokens: 40, prefill_seconds: 0.1, generated_tokens: 50, decode_seconds: 1, outcome: 'ok', spec: { mode: 'mtp', rounds: 10, drafted: 30, accepted: 21 } }],
  };
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
    // A four-digit speed in the eight-tile Advanced row keeps its unit on the number's line at 1440 px.
    const { page, context } = await simPage(browser, { mode: 'advanced', contextOptions: { viewport: { width: 1440, height: 900 } } });
    await page.evaluate(() => window.__sim.finish({ computed: 5600, prefillS: 4, gen: 100, decodeS: 2 }));
    await adv(page, 3000);
    for (const id of ['dashTPre', 'dashTGen']) {
      const drift = await page.evaluate((id) => {
        const v = document.getElementById(id), u = v.querySelector('small'), r = document.createRange();
        r.selectNodeContents(v.firstChild);
        return Math.abs(r.getBoundingClientRect().bottom - u.getBoundingClientRect().bottom);
      }, id);
      assert.ok(drift < 6, `${id}'s unit stays on the number's line (${drift}px apart)`);
    }
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
    assert.ok((await axisOf(page, 'dashCMem')).includes('restart'), 'the marker is drawn at the server\'s time (this server\'s clock is 10 min ahead)');
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
    assert.match(tip, /71 read, 2,970 reused/);
    assert.doesNotMatch(tip, /tokens\/s/, 'no speed is claimed for it');
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

// The chart picker: which charts show and in what order is a browser preference; every chart is half a row.
async function chartPicker(browser) {
  const shown = (page) => page.evaluate(() => [...document.querySelectorAll('#dashCharts .chart:not([hidden])')].map((c) => c.dataset.chart));
  {
    const { page, errors, context } = await openPage(browser, '/dashboard', { mode: 'advanced' });
    await page.waitForFunction(() => /Live/.test(document.getElementById('dashStatus').textContent));
    const all = await shown(page);
    assert.equal(all.length, 13, 'Advanced shows all but the DSpark-only chart until a choice is made');
    const widths = await page.evaluate(() => {
      const grid = document.getElementById('dashCharts').getBoundingClientRect(), gap = parseFloat(getComputedStyle(document.getElementById('dashCharts')).columnGap);
      return { half: (grid.width - gap) / 2, cards: [...document.querySelectorAll('#dashCharts .chart:not([hidden])')].map((c) => [c.dataset.chart, c.getBoundingClientRect().width]) };
    });
    for (const [id, w] of widths.cards) assert.ok(Math.abs(w - widths.half) < 1.5, `${id} is half the row at 1280 px (${w} vs ${widths.half})`);
    const rows = await page.evaluate(() => new Set([...document.querySelectorAll('#dashCharts .chart:not([hidden])')].map((c) => Math.round(c.getBoundingClientRect().x))).size);
    assert.equal(rows, 2, 'two columns on a desktop width');
    await page.setViewportSize({ width: 600, height: 800 });
    const narrow = await page.evaluate(() => new Set([...document.querySelectorAll('#dashCharts .chart:not([hidden])')].map((c) => Math.round(c.getBoundingClientRect().x))).size);
    assert.equal(narrow, 1, 'one column on a narrow screen');
    await page.setViewportSize({ width: 1280, height: 800 });

    assert.equal(await page.isVisible('#dashPicker'), false, 'the picker starts closed');
    await page.click('#dashChartsBtn');
    assert.equal(await page.isVisible('#dashPicker'), true);
    assert.equal(await page.getAttribute('#dashChartsBtn', 'aria-expanded'), 'true');
    await page.uncheck('#dashPickList input[data-chart="latency"]');
    await page.uncheck('#dashPickList input[data-chart="cachereuse"]');
    const after = await shown(page);
    assert.deepEqual(all.filter((id) => !after.includes(id)).sort(), ['cachereuse', 'latency'], 'unticking hides exactly those charts');
    assert.equal(await page.isVisible('#dashLat'), false);
    await page.check('#dashPickList input[data-chart="latency"]');
    assert.ok((await shown(page)).includes('latency'), 'ticking brings it back');
    await page.uncheck('#dashPickList input[data-chart="latency"]');
    await page.keyboard.press('Escape');
    assert.equal(await page.isVisible('#dashPicker'), false, 'Escape closes the picker');
    await page.reload();
    await page.waitForFunction(() => /Live/.test(document.getElementById('dashStatus').textContent));
    assert.deepEqual(await shown(page), after.filter((id) => id !== 'latency'), 'the choice survives a reload');
    await page.click('#dashChartsBtn');
    assert.equal(await page.isChecked('#dashPickList input[data-chart="latency"]'), false, 'and the picker shows it');
    await page.click('#dashPickReset');
    assert.deepEqual(await shown(page), all, 'Reset restores the default for the view');
    assert.deepEqual(errors, []);
    await context.close();
  }
  {
    // Compact starts with four charts; a choice made there holds in both views.
    const { page, errors, context } = await openPage(browser, '/dashboard', { mode: 'compact' });
    await page.waitForFunction(() => /Live/.test(document.getElementById('dashStatus').textContent));
    assert.deepEqual(await shown(page), ['prefill', 'decode', 'gpu', 'memory'], 'Compact shows the four essentials');
    await page.click('#dashChartsBtn');
    await page.check('#dashPickList input[data-chart="outcomes"]');
    await page.click('#dashPickDone');
    await page.click('#modeToggle [data-mode="advanced"]');
    assert.equal((await shown(page)).length, 5, 'a stored choice no longer depends on the view');
    assert.deepEqual(errors, []);
    await context.close();
  }
  {
    // Config mode: dragging a chart onto another moves it, the arrows nudge it, and the order outlives a reload.
    const { page, errors, context } = await openPage(browser, '/dashboard', { mode: 'advanced' });
    await page.waitForFunction(() => /Live/.test(document.getElementById('dashStatus').textContent));
    const first = (await shown(page)).slice(0, 4);
    assert.equal(await page.evaluate(() => getComputedStyle(document.querySelector('#dashCharts .cfg')).display), 'none', 'no drag strip or arrows while the picker is closed');
    await page.click('#dashChartsBtn');
    await page.click('#dashPickDone');
    await page.click('#dashChartsBtn');
    await page.locator('#dashCharts .chart[data-chart="decode"]').scrollIntoViewIfNeeded();
    await page.dragAndDrop('#dashCharts .chart[data-chart="prefill"] .grip', '#dashCharts .chart[data-chart="requests"] .grip');
    const dragged = await shown(page);
    assert.deepEqual(dragged.slice(0, 3), [first[1], first[2], first[0]], `dragging prefill onto requests moves it after (${dragged.slice(0, 4)})`);
    await page.click(`#dashCharts .chart[data-chart="${first[3]}"] [data-move="-1"]`);
    assert.deepEqual((await shown(page)).slice(0, 4), [first[1], first[2], first[3], first[0]], 'the earlier arrow swaps with the previous chart');
    assert.equal(await page.evaluate(() => document.activeElement.dataset.move), '-1', 'and keeps the focus on the arrow');
    await page.click('#dashPickDone');
    const saved = await shown(page);
    await page.reload();
    await page.waitForFunction(() => /Live/.test(document.getElementById('dashStatus').textContent));
    assert.deepEqual(await shown(page), saved, 'the order survives a reload');
    assert.deepEqual(errors, []);
    await context.close();
  }
  {
    // The arrows at the ends of the order are disabled, Escape from anywhere in config mode closes it, and closing it drops a drag.
    const { page, errors, context } = await openPage(browser, '/dashboard', { mode: 'advanced' });
    await page.waitForFunction(() => /Live/.test(document.getElementById('dashStatus').textContent));
    await page.click('#dashChartsBtn');
    const ends = () => page.evaluate(() => {
      const cards = [...document.querySelectorAll('#dashCharts .chart:not([hidden])')];
      const state = (c, dir) => { const b = c.querySelector(`[data-move="${dir}"]`); return { aria: b.getAttribute('aria-disabled'), opacity: getComputedStyle(b).opacity }; };
      return { first: state(cards[0], -1), second: state(cards[1], -1), last: state(cards.at(-1), 1), beforeLast: state(cards.at(-2), 1), firstDown: state(cards[0], 1) };
    });
    let e = await ends();
    assert.equal(e.first.aria, 'true', 'the first chart cannot move earlier');
    assert.ok(Number(e.first.opacity) < 1, 'and its arrow reads as disabled');
    assert.equal(e.last.aria, 'true', 'the last chart cannot move later');
    assert.equal(e.second.aria, null, 'a chart with room before it keeps its arrow');
    assert.equal(e.beforeLast.aria, null);
    assert.equal(e.firstDown.aria, null);
    const order = () => page.evaluate(() => [...document.querySelectorAll('#dashCharts .chart:not([hidden])')].map((c) => c.dataset.chart));
    const was = await order();
    await page.click(`#dashCharts .chart[data-chart="${was[0]}"] [data-move="-1"]`, { force: true });
    assert.deepEqual(await order(), was, 'a disabled arrow moves nothing');
    await page.click(`#dashCharts .chart[data-chart="${was[1]}"] [data-move="-1"]`);
    e = await ends();
    assert.equal(e.first.aria, 'true', 'and the new first chart is the disabled one');
    assert.equal(e.second.aria, null, 'the arrow of the chart that moved down is live again');
    await page.uncheck(`#dashPickList input[data-chart="${was[0]}"]`);
    assert.equal((await ends()).second.aria, null, 'hiding a chart re-reads the ends');
    await page.keyboard.press('Escape');
    await page.click('#dashChartsBtn');
    await page.focus('#dashCharts .chart[data-chart="memory"] [data-move="1"]');
    await page.keyboard.press('Escape');
    assert.equal(await page.isVisible('#dashPicker'), false, 'Escape from a chart arrow closes config mode');
    assert.equal(await page.evaluate(() => document.activeElement.id), 'dashChartsBtn', 'and returns the focus to the Charts button');
    await page.click('#dashChartsBtn');
    const before = await order();
    await page.locator(`#dashCharts .chart[data-chart="${before[1]}"]`).scrollIntoViewIfNeeded();
    const point = (id) => page.evaluate((i) => { const r = document.querySelector(`#dashCharts .chart[data-chart="${i}"] .grip`).getBoundingClientRect(); return { x: r.left + 20, y: r.top + r.height / 2 }; }, id);
    const from = await point(before[0]), to = await point(before[1]);
    await page.mouse.move(from.x, from.y);
    await page.mouse.down();
    await page.mouse.move(to.x, to.y, { steps: 6 });
    assert.equal(await page.evaluate(() => document.querySelectorAll('#dashCharts .chart.over').length), 1, 'the target is marked mid-drag');
    await page.keyboard.press('Escape');
    assert.equal(await page.evaluate(() => document.querySelectorAll('#dashCharts .chart.dragging, #dashCharts .chart.over').length), 0, 'closing the panel clears the drag marks');
    await page.mouse.up();
    assert.deepEqual(await order(), before, 'and the release drops nothing');
    assert.deepEqual(errors, []);
    await context.close();
  }
  {
    // The picker follows the theme and draws no motion when reduced.
    const { page, context } = await openPage(browser, '/dashboard', { mode: 'advanced', contextOptions: { reducedMotion: 'reduce' } });
    await page.waitForFunction(() => /Live/.test(document.getElementById('dashStatus').textContent));
    const transitions = await page.evaluate(() => [...document.querySelectorAll('#dashCharts .hb .tr > i')].map((i) => getComputedStyle(i).transitionDuration));
    assert.ok(transitions.length > 0 && transitions.every((d) => d === '0s'), `bars do not animate under reduced motion (${transitions})`);
    await context.close();
  }
}

// The picker is an inline panel, a touch drags a chart too, count charts keep whole-number gridlines, and the drafting and cache charts read the exported fields.
async function chartExtras(browser) {
  {
    const { page, errors, context } = await openPage(browser, '/dashboard', { mode: 'advanced' });
    await page.waitForFunction(() => /Live/.test(document.getElementById('dashStatus').textContent));
    const box = (sel) => page.evaluate((q) => { const r = document.querySelector(q).getBoundingClientRect(); return { top: r.top, bottom: r.bottom }; }, sel);
    const before = await box('#dashCharts');
    await page.click('#dashChartsBtn');
    const picker = await box('#dashPicker'), after = await box('#dashCharts');
    assert.ok(after.top >= picker.bottom - 1, 'the open picker sits above the charts instead of covering them');
    assert.ok(after.top > before.top + 40, `and pushes them down (${before.top} to ${after.top})`);
    await page.click('#dashPickDone');
    assert.ok(Math.abs((await box('#dashCharts')).top - before.top) < 2, 'Done gives the room back');
    assert.deepEqual(errors, []);
    await context.close();
  }
  {
    // A finger drags a chart by its strip: the same reorder as the mouse, no scrolling taken from it.
    const { page, errors, context } = await openPage(browser, '/dashboard', { mode: 'advanced', contextOptions: { hasTouch: true } });
    await page.waitForFunction(() => /Live/.test(document.getElementById('dashStatus').textContent));
    await page.click('#dashChartsBtn');
    await page.click('#dashPickDone');
    await page.click('#dashChartsBtn');
    await page.locator('#dashCharts .chart[data-chart="prefill"]').scrollIntoViewIfNeeded();
    const order = () => page.evaluate(() => [...document.querySelectorAll('#dashCharts .chart:not([hidden])')].map((c) => c.dataset.chart));
    const was = await order();
    const grip = (id) => page.evaluate((i) => { const r = document.querySelector(`#dashCharts .chart[data-chart="${i}"] .grip`).getBoundingClientRect(); return { x: r.left + 20, y: r.top + r.height / 2 }; }, id);
    const from = await grip('prefill'), to = await grip('decode');
    const cdp = await context.newCDPSession(page);
    const touch = (type, point) => cdp.send('Input.dispatchTouchEvent', { type, touchPoints: point ? [{ x: point.x, y: point.y, id: 1 }] : [] });
    await touch('touchStart', from);
    // The browser takes up the first move as touch slop and may hold the last, so the path runs on a little inside the target.
    for (let i = 1; i <= 9; i++) await touch('touchMove', { x: from.x + (to.x - from.x) * i / 6 + (i > 6 ? 4 * (i - 6) : 0), y: from.y + (to.y - from.y) * i / 6 });
    assert.equal(await page.evaluate(() => document.querySelectorAll('#dashCharts .chart.over').length), 1, 'the chart under the finger is marked');
    await touch('touchEnd');
    const now = await order();
    assert.deepEqual(now.slice(0, 2), [was[1], was[0]], `a touch drag moves prefill after decode (${now.slice(0, 3)})`);
    assert.deepEqual(await page.evaluate(() => JSON.parse(localStorage.getItem('sushi-dash:charts')).order.slice(0, 2)), [was[1], was[0]], 'and the order is saved');
    assert.deepEqual(errors, []);
    await context.close();
  }
  for (const [running, labels] of [[1, ['0', '1']], [2, ['0', '1', '2']], [7, ['0', '2', '4', '6', '8']]]) {
    const { page, context } = await openPage(browser, '/dashboard', { mode: 'advanced', history: true, runningCount: running });
    // The scale eases to its new top; the labels are read once it has landed.
    await page.waitForFunction((top) => (window.__axis.dashCReq ?? []).includes(top), labels[labels.length - 1], { timeout: 15000 });
    const axis = (await axisOf(page, 'dashCReq')).filter((t) => /^[\d.,]+$/.test(t));
    assert.deepEqual(axis, labels, `the Requests chart counts in whole numbers at a peak of ${running} (${axis})`);
    await context.close();
  }
  {
    const { page, errors, context } = await openPage(browser, '/dashboard', { mode: 'advanced', specData: specFixture(), storage: ALL_ON });
    await page.waitForFunction(() => /Live/.test(document.getElementById('dashStatus').textContent));
    await page.waitForFunction(() => document.getElementById('dashSpecModes').textContent.includes('MTP'), null, { timeout: 15000 });
    const rows = (id) => page.evaluate((i) => [...document.querySelectorAll(`#${i} .hb`)].map((r) => [r.querySelector('span').textContent, r.querySelector('b').textContent]), id);
    assert.deepEqual(await rows('dashSpecModes'), [['MTP', '70%, 2.1 per round'], ['DSpark', '60%, 3.0 per round'], ['DSpark rounds without a draft', '25%']], 'draft acceptance by method: acceptance and the mean kept per round');
    assert.deepEqual(await rows('dashSpecRows'), [['1 token (no draft)', '5'], ['3 tokens', '10'], ['5 tokens', '5']], 'DSpark rounds by tokens verified');
    await page.hover('#dashSpecModes .hb');
    await page.waitForFunction(() => document.getElementById('tip').classList.contains('show'), null, { timeout: 5000 });
    assert.match(await page.evaluate(() => document.getElementById('tip').textContent), /MTP\nAcceptance: 70\.0%\nAccepted \/ drafted: 84 \/ 120\n/, 'a bar row explains its figure in the page\'s one tooltip');
    await page.waitForFunction(() => (window.__dots.dashCSpec ?? []).length > 0, null, { timeout: 15000 });
    const axis = await axisOf(page, 'dashCSpec');
    assert.ok(axis.includes('100%') && axis.includes('0%'), `the acceptance chart is a percentage (${axis})`);
    const reuse = await textOf(page, 'dashReuse');
    assert.match(reuse, /Reused from memory\s*3/);
    assert.match(reuse, /Reused from SSD\s*9/);
    const tiers = await textOf(page, 'dashTiers');
    assert.match(tiers, /Dropped from memory\s*1 \(1\.0 MiB\)/);
    assert.match(tiers, /Dropped from SSD\s*2 \(8\.0 MiB\)/);
    assert.match(tiers, /In memory/);
    assert.deepEqual(errors, []);
    await context.close();
  }
  {
    // An older server that exports no drafting counters is not read as one where nothing drafted.
    const { page, errors, context } = await openPage(browser, '/dashboard', { mode: 'advanced', storage: ALL_ON });
    await page.waitForFunction(() => /Live/.test(document.getElementById('dashStatus').textContent));
    assert.equal(await textOf(page, 'dashSpecModes'), '', 'no method rows: the chart above says why');
    assert.match(await textOf(page, 'dashSpecRows'), /Not reported by this server/);
    await page.waitForFunction(() => (window.__axis.dashCSpec ?? []).length > 0);
    assert.ok((await axisOf(page, 'dashCSpec')).includes('Not reported by this server'), 'nor is the acceptance chart');
    assert.deepEqual(errors, []);
    await context.close();
  }
  {
    // A server that drafted nothing says so.
    const zero = { requests: 0, rounds: 0, drafted: 0, accepted: 0, plain_rounds: 0 };
    const { page, context } = await openPage(browser, '/dashboard', { mode: 'advanced', specData: { speculative: { mtp: zero, dflash: zero, dspark: zero } }, storage: ALL_ON });
    await page.waitForFunction(() => (window.__axis.dashCSpec ?? []).some((t) => /No request used speculative decoding/.test(t)));
    assert.equal(await textOf(page, 'dashSpecModes'), '', 'and draws no empty method rows');
    assert.match(await textOf(page, 'dashSpecRows'), /No DSpark round yet/);
    await context.close();
  }
}

// The bar, column and histogram charts work from the keyboard, on a phone, and say only what the server reported.
async function chartAudit(browser) {
  const tipText = (page) => page.evaluate(() => document.getElementById('tip').classList.contains('show') ? document.getElementById('tip').textContent : null);
  const latency = (per) => { let c = 0; return { count: per.reduce((a, b) => a + b, 0), sum: 3, bounds: [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10], bucket_counts: per.map((v) => c += v) }; };
  const hists = { time_to_first_token_seconds: latency([0, 0, 1, 2, 3, 1, 0, 0, 0, 0, 0]), e2e_request_latency_seconds: latency(Array(11).fill(0)) };
  {
    // A card's bars are one Tab stop; the arrows walk its rows; the focused row keeps the focus and its tip through a poll that changes it.
    const { page, errors, context } = await openPage(browser, '/dashboard', { mode: 'advanced', specData: { histograms: hists } });
    await page.waitForFunction(() => document.querySelectorAll('#dashWrites .hb').length > 2);
    assert.deepEqual(await page.$$eval('#dashWrites .hb', (rows) => rows.map((r) => r.tabIndex)).then((t) => [t[0], t.slice(1).every((x) => x === -1)]), [0, true], 'the first row takes Tab, the others wait for the arrows');
    await page.locator('#dashReuseBars .hb').first().focus();
    await page.keyboard.press('Tab');
    assert.equal(await page.evaluate(() => document.activeElement === document.querySelector('[data-chart="cachetiers"] h2 [data-tip]')), true, 'Tab goes from one card\'s bars to the next card\'s title');
    await page.keyboard.press('Tab');
    assert.equal(await page.evaluate(() => document.activeElement === document.querySelector('#dashTiers .hb')), true, 'and then to its first row');
    await page.locator('#dashWrites .hb').first().focus();
    await page.keyboard.press('ArrowDown');
    assert.equal(await page.evaluate(() => document.activeElement === document.querySelectorAll('#dashWrites .hb')[1]), true);
    await page.keyboard.press('ArrowUp');
    assert.equal(await page.evaluate(() => document.activeElement === document.querySelector('#dashWrites .hb')), true, 'the arrows move along the rows');
    assert.match(await tipText(page) ?? '', /^Attention memory\nWritten: /, 'a focused row shows its tip');
    const before = await page.$eval('#dashWrites .hb b', (b) => b.textContent);
    await page.waitForFunction((was) => document.querySelector('#dashWrites .hb b').textContent !== was, before, { timeout: 8000 });
    assert.equal(await page.evaluate(() => document.activeElement === document.querySelector('#dashWrites .hb')), true, 'a poll that rewrites the row keeps the focus on it');
    const now = await page.$eval('#dashWrites .hb b', (b) => b.textContent);
    await page.waitForFunction((v) => document.getElementById('tip').textContent.includes(v), now, { timeout: 3000 });
    // A histogram steps through its buckets from the keyboard and lets go on blur.
    await page.focus('#dashHLat');
    await page.keyboard.press('ArrowRight'); await page.keyboard.press('ArrowRight'); await page.keyboard.press('ArrowRight');
    assert.match(await page.textContent('#dashTip'), /25 ms to 50 ms\s*First token\s*1 \(14%\)\s*14% got their first token within 50 ms\s*Last token\s*0 \(-\)/, 'the third bucket, both series named in the chart tooltip');
    assert.equal(await page.$$eval('#dashHLat > div', (b) => b.findIndex((x) => x.classList.contains('on'))), 2);
    await page.keyboard.press('Tab');
    assert.equal(await page.isVisible('#dashTip'), false, 'blur hides it');
    assert.match(await textOf(page, 'dashHLen'), /Not reported by this server/, 'a histogram the server does not export says so instead of drawing nothing');
    assert.equal(await page.getByRole('group', { name: 'Prompt cache reuse' }).count(), 1, 'each chart card is a named group');
    const named = await page.$$eval('#dashCharts .chart[role="group"]', (cards) => cards.map((c) => [c.getAttribute('aria-label'), c.querySelector('h2 [data-tip]').textContent.replace(/ \([^)]*\)$/, '')]));
    assert.equal(named.length, 14, 'chart cards');
    assert.deepEqual(named.filter(([a, t]) => a !== t), [], 'a chart has one name: the picker, the card and its title agree');
    assert.deepEqual(errors, []);
    await context.close();
  }
  {
    // Every chart title and every Sessions column explains itself, and the keyboard reaches each explanation.
    const { page, errors, context } = await openPage(browser, '/dashboard', { mode: 'advanced', specData: specFixture() });
    await page.waitForFunction(() => /Live/.test(document.getElementById('dashStatus').textContent));
    const titles = await page.$$eval('#dashCharts .chart', (cards) => cards.map((c) => { const a = c.querySelector('h2 [data-tip]'); return { id: c.dataset.chart, tip: a?.dataset.tip ?? '', tab: a?.tabIndex }; }));
    assert.equal(titles.length, 14, 'every chart was checked');
    for (const t of titles) assert.ok(t.tip.length > 20 && t.tab === 0, `${t.id}: its title has a focusable tooltip`);
    const heads = await page.$$eval('table.sessions thead th', (ths) => ths.filter((th) => th.textContent.trim()).map((th) => ({ text: th.textContent, tip: th.querySelector('[data-tip]')?.dataset.tip ?? '' })));
    for (const h of heads) assert.ok(h.tip.length > 10, `Sessions column ${h.text} has a tooltip`);
    await page.keyboard.press('Tab');
    await page.locator('#dashCharts .chart h2 [data-tip]').first().focus();
    await page.waitForFunction(() => document.getElementById('tip').classList.contains('show'), null, { timeout: 3000 });
    assert.equal(await page.evaluate(() => document.getElementById('tip').firstChild.textContent), titles[0].tip, 'keyboard focus opens a chart title\'s tooltip');
    assert.deepEqual(errors, []);
    await context.close();
  }
  {
    // Everything the plain-language pass renamed keeps its engineering name in its tooltip's "Technical name" line.
    const lines = (page, scope) => page.$$eval(`${scope} [data-tip]`, (ns) => Object.fromEntries(ns.map((n) => [(n.querySelector('.k') ?? [...n.childNodes].find((c) => c.nodeType === 3 && c.textContent.trim()) ?? n).textContent.trim(), n.dataset.term ?? ''])));
    const expectTerms = (got, want, where) => {
      for (const [label, term] of Object.entries(want)) {
        assert.ok(label in got, `${where}: ${label} is an anchor (${Object.keys(got)})`);
        assert.match(got[label], term, `${where}: ${label} names its technical term`);
      }
    };
    const compact = await openPage(browser, '/dashboard', { mode: 'compact' });
    await compact.page.waitForSelector('.tiles-cmp .tile[data-tip]');
    expectTerms(await lines(compact.page, '.tiles-cmp'), {
      'Generation speed': /decode/, 'Prompt speed': /prefill/, Memory: /MLX active.*MLX cache/, 'Prompt cache': /prefix cache/, Requests: /running.*waiting/,
    }, 'Compact tiles');
    await compact.context.close();

    const { page, errors, context } = await openPage(browser, '/dashboard', { mode: 'advanced', sessions: [{ request_id: 7, model: 'org/test-model', phase: 'prefill', context_length: 8192, prefill_done: 10, prefill_expected: 100 }, { request_id: 8, model: 'org/test-model', phase: 'decode', context_length: 8192 }],
      specData: { ...specFixture(), decode_serial: { spec_active: 4, pad_waste: 2 } }, storage: ALL_ON });
    await page.waitForFunction(() => /Live/.test(document.getElementById('dashStatus').textContent));
    await page.waitForFunction(() => document.getElementById('dashSpecModes').textContent.includes('MTP') && document.getElementById('dashSerial').textContent.includes('Speculative decoding'), null, { timeout: 15000 });
    expectTerms(await lines(page, 'section.tiles.feed'), {
      'Gen speed': /decode/, 'Prompt speed': /prefill/, Requests: /running.*waiting/, GPU: /utilization/, 'First token': /TTFT/, Served: /generation_tokens_total/,
    }, 'Advanced tiles');
    expectTerms(await lines(page, 'table.sessions thead th'), { Generated: /generated_tokens/, Step: /phase.*prefill.*decode.*cached/, Reused: /cached/, Speed: /rate/, Memory: /state_bytes/ }, 'Sessions columns');
    assert.match(await page.getAttribute('h2 > span.tip-u:text-is("Sessions")', 'data-term'), /sessions/, 'the Sessions heading');
    assert.deepEqual(await page.$$eval('label[data-tip]:has(#dashInterval, #dashWindow)', (ns) => ns.map((n) => n.dataset.term)), ['poll interval', 'chart window'], 'the Every and Window selects');
    const charts = await page.$$eval('#dashCharts .chart', (cards) => Object.fromEntries(cards.map((c) => [c.dataset.chart, c.querySelector('h2 [data-tip]')?.dataset.term ?? ''])));
    const chartTerms = { prefill: /prefill/, decode: /decode/, requests: /prefill.*batched/, gpu: /utilization/, memory: /footprint.*MLX active.*MLX cache/, specaccept: /draft acceptance.*MTP.*DFlash2.*DSpark/,
      specrows: /verifier rows/, cachereuse: /prefix cache.*hit.*computed/, cachetiers: /evict/, cachewrites: /prefix cache/, outcomes: /requests_/, latency: /TTFT.*prefill.*decode.*e2e/,
      length: /prompt_tokens.*output_tokens/, serial: /decode_serial/ };
    assert.equal(Object.keys(charts).length, Object.keys(chartTerms).length, 'every chart title was checked');
    for (const [id, term] of Object.entries(chartTerms)) assert.match(charts[id] ?? '', term, `chart ${id}: its title names the technical term`);
    const rowTerms = (id) => page.$$eval(`#${id} .hb`, (rows) => rows.map((r) => [r.querySelector('span').textContent, r.dataset.term ?? '']));
    assert.deepEqual((await rowTerms('dashSerial')).map(([, t]) => t), ['spec_active', 'pad_waste'], 'a serial reason shows the server\'s own name');
    assert.match((await rowTerms('dashSpecRows'))[0][1], /verifier rows/, 'a verifier-row bar');
    assert.match((await rowTerms('dashSpecModes'))[0][1], /mtp_acceptance|draft acceptance/, 'a method bar');
    assert.deepEqual((await rowTerms('dashTiers')).filter(([l]) => /^Dropped/.test(l)).map(([, t]) => /evict/.test(t)), [true, true], 'Dropped is evicted');
    assert.match((await rowTerms('dashTiers')).find(([l]) => l === 'In memory')[1], /hot/, 'In memory is the hot tier');
    assert.match(await page.getAttribute('#dashReuseHero', 'data-term'), /prefix_cache_tokens/, 'the reuse share');
    // A phase chip's fly-over carries the engine's phase name.
    await page.hover('#dashSessions [data-phase="prefill"]');
    await page.waitForFunction(() => document.getElementById('dashTip').style.display === 'block');
    assert.match(await page.textContent('#dashTip'), /Technical name:\s*prefill/, 'Reading is prefill');
    await page.hover('#dashSessions [data-phase="decode"]');
    await page.waitForFunction(() => /Generating/.test(document.getElementById('dashTip').textContent));
    assert.match(await page.textContent('#dashTip'), /Technical name:\s*decode/, 'Generating is decode');
    assert.deepEqual(await page.$$eval('#dashSessions .phase', (ns) => ns.map((n) => n.textContent)), ['Reading prompt', 'Generating'], 'the step chips use the usual words');
    assert.deepEqual(await page.$$eval('#dashSessions .phase', (ns) => ns.map((n) => n.innerText)), ['Reading prompt', 'Generating'], 'and show them in sentence case');
    const titles = await page.$$eval('#dashCharts .chart h2 [data-tip]', (ns) => ns.map((n) => n.textContent));
    for (const t of ['Draft acceptance', 'Response time', 'Prompt and generated length', 'Generation speed (tokens/s)', 'Tokens verified per round', 'Why requests weren\'t batched', 'Prompt cache reuse', 'Prompt cache storage', 'Prompt cache writes']) assert.ok(titles.includes(t), `chart ${t} (${titles})`);
    assert.match(await page.textContent('[data-chart="memory"] .legend'), /in use.*reusable cache.*other.*machine memory/, 'the memory legend');
    const words = await page.evaluate(() => [...document.querySelectorAll('body *:not(script):not(style)')].flatMap((n) => [...n.childNodes].filter((c) => c.nodeType === 3).map((c) => c.textContent)
      .concat(['tip', 'term'].map((k) => n.dataset[k]), n.getAttribute('aria-label'), n.title)).filter(Boolean).join('\n'));
    assert.doesNotMatch(words, /guess|conversation cache/i, 'no page says guess or Conversation cache');
    assert.doesNotMatch(words, RETIRED_REPLY, 'no speed, length or state says reply: thinking and answer are the same generated tokens');
    assert.deepEqual(errors, []);
    await context.close();
  }
  {
    // A refused request computed nothing and restored nothing: it is no bar in the cached vs computed chart.
    const t = Date.now() - 3000, row = (request_id, outcome, dt) => ({ request_id, model: 'org/test-model', t_ms: t + dt, prompt_tokens: 900, cached_tokens: 100, prefill_seconds: 0, generated_tokens: 0, decode_seconds: 0, outcome });
    const { page, context } = await openPage(browser, '/dashboard', { mode: 'advanced', specData: { requests: [row(1, 'ok', 0), row(2, 'rejected', 1)] } });
    await page.waitForFunction(() => document.querySelectorAll('#dashStk > div').length > 0);
    await page.waitForTimeout(500);
    assert.equal(await page.locator('#dashStk > div').count(), 1);
    await context.close();
  }
  {
    // With no model loaded the hot tier's setting is unknown, not off.
    const { page, context } = await openPage(browser, '/dashboard', { mode: 'advanced', noModels: true, specData: specFixture() });
    await page.waitForFunction(() => /In memory/.test(document.getElementById('dashTiers').textContent));
    assert.match(await textOf(page, 'dashTiers'), /In memory\s*-\s*SSD\s*-/);
    await context.close();
  }
  {
    // The open picker ticks what the view shows after a Compact/Advanced switch.
    const { page, context } = await openPage(browser, '/dashboard', { mode: 'compact' });
    await page.click('#dashChartsBtn');
    await page.click('#modeToggle [data-mode="advanced"]');
    const shown = await page.$$eval('#dashCharts .chart:not([hidden])', (c) => c.map((x) => x.dataset.chart).sort());
    assert.deepEqual(await page.$$eval('#dashPickList input:checked', (c) => c.map((x) => x.dataset.chart).sort()), shown);
    await context.close();
  }
  {
    // On a 320 px phone every meter keeps a visible track, no value is cut off, and every printed bucket label is whole.
    const { page, errors, context } = await openPage(browser, '/dashboard', { mode: 'advanced', specData: { ...specFixture(), histograms: hists }, contextOptions: { viewport: { width: 320, height: 800 } } });
    await page.waitForFunction(() => document.querySelectorAll('#dashTiers .hb').length >= 4 && document.querySelectorAll('#dashHlLat span').length === 11);
    const layout = await page.evaluate(() => {
      const inside = (el) => { const c = el.closest('.chart').getBoundingClientRect(), r = el.getBoundingClientRect(); return r.left >= c.left - 0.5 && r.right <= c.right + 0.5; };
      document.getElementById('dashHlLat').scrollIntoView({ block: 'center' });
      const labels = [...document.querySelectorAll('#dashHlLat span')].filter((s) => getComputedStyle(s).visibility === 'visible').map((s) => {
        const range = document.createRange(); range.selectNodeContents(s); const r = range.getBoundingClientRect();
        return document.elementFromPoint(r.right - 1, r.top + r.height / 2) === s;
      });
      return {
        tracks: [...document.querySelectorAll('#dashTiers .hb .tr, #dashReuseBars .hb .tr')].map((t) => t.getBoundingClientRect().width),
        values: [...document.querySelectorAll('#dashTiers .hb b, #dashReuseBars .hb b, #dashLat .k b, #dashLat .parts span')].map(inside),
        labels,
      };
    });
    assert.ok(layout.tracks.every((w) => w > 80), `every meter keeps a track (${layout.tracks.map(Math.round)})`);
    assert.ok(layout.values.every(Boolean), 'every value sits inside its card');
    assert.ok(layout.labels.length >= 5 && layout.labels.every(Boolean), `every shown bucket label is printed whole (${layout.labels})`);
    assert.deepEqual(errors, []);
    await context.close();
  }
}

// "Reply" as a speed, length or state word is retired; it stays only where it means the answer's text.
const RETIRED_REPLY = /\\brepl(?:y|ies)\\s+(?:speed|length|tok|done|[\\d-])|\\breplying\\b|\\blast reply\\b|(?:faster|slower|slowly)\\s+repl|repl(?:y|ies)\\s+(?:come\\s+)?(?:faster|more slowly)|\\bspeeds?\\s+(?:up\\s+)?repl|\\breply tokens\\b/i;
// A stored choice that shows every chart, the DSpark-only one included.
const ALL_ON = { 'sushi-dash:charts': JSON.stringify({ order: [], hidden: [] }) };
// The merged cards: a saved choice that names the cards they replaced migrates, motion honours reduced motion and a
// hidden tab, and an empty server gets a readable note in every card.
const MERGED_IDS = ['prefill', 'decode', 'requests', 'latency', 'gpu', 'memory', 'cachereuse', 'cachetiers', 'specaccept', 'length', 'outcomes', 'serial', 'cachewrites', 'specrows'];
async function dashboardV2(browser) {
  const shown = (page) => page.evaluate(() => [...document.querySelectorAll('#dashCharts .chart:not([hidden])')].map((c) => c.dataset.chart));
  const live = (page) => page.waitForFunction(() => /Live/.test(document.getElementById('dashStatus').textContent));
  {
    const { page, errors, context } = await openPage(browser, '/dashboard', { mode: 'advanced' });
    await live(page);
    assert.deepEqual(await page.$$eval('#dashCharts .chart', (c) => c.map((x) => x.dataset.chart)), MERGED_IDS, 'fourteen cards, two per row in this order');
    assert.deepEqual(await shown(page), MERGED_IDS.filter((id) => id !== 'specrows'), 'Advanced shows all but the DSpark-only card until a choice is made');
    const wrapped = await page.$$eval('section.adv.tiles .tile .k', (ks) => ks.filter((k) => k.offsetHeight > 1.5 * parseFloat(getComputedStyle(k).lineHeight)).map((k) => k.textContent));
    assert.deepEqual(wrapped, [], 'every eight-tile label fits on one line (the short form where the full one wraps)');
    await page.click('#dashChartsBtn');
    assert.deepEqual(await page.$$eval('#dashPickList input', (c) => c.map((x) => x.dataset.chart)), MERGED_IDS, 'the picker lists the merged cards');
    assert.deepEqual(errors, []);
    await context.close();
  }
  {
    // A choice saved before the merge: the old ids map to their merged card, which shows if any of its parts showed.
    const old = ['ttft', 'prefill', 'memsplit', 'decode', 'requests', 'gpu', 'memory', 'specaccept', 'specmodes', 'specrows', 'cachetok', 'cachereuse', 'cachetiers', 'cachewrites', 'outcomes', 'latency', 'e2e', 'ptok', 'otok', 'serial', 'bogus'];
    const hidden = ['ttft', 'e2e', 'latency', 'memsplit', 'memory', 'otok', 'specrows', 'cachewrites'];
    const { page, errors, context } = await openPage(browser, '/dashboard', { mode: 'advanced', storage: { 'sushi-dash:charts': JSON.stringify({ order: old, hidden }) } });
    await live(page);
    assert.deepEqual(await shown(page), ['prefill', 'decode', 'requests', 'gpu', 'specaccept', 'cachereuse', 'cachetiers', 'outcomes', 'length', 'serial'], 'merged cards take the first place of any part; one hides only when every part was hidden');
    const saved = await page.evaluate(() => JSON.parse(localStorage.getItem('sushi-dash:charts')));
    assert.deepEqual([...saved.order].sort(), [...MERGED_IDS].sort(), 'the stored order is rewritten with the merged ids only');
    assert.deepEqual([...saved.hidden].sort(), ['cachewrites', 'latency', 'memory', 'specrows'], 'and so is the hidden list');
    const blanks = await page.$$eval('#dashCharts .chart:not([hidden])', (cards) => cards.filter((c) => !c.querySelector('h2 [data-tip]')?.textContent || c.getBoundingClientRect().height < 150).map((c) => c.dataset.chart));
    assert.deepEqual(blanks, [], 'no migrated card is blank');
    await page.click('#dashChartsBtn');
    assert.equal(await page.isChecked('#dashPickList input[data-chart="length"]'), true, 'the picker ticks a merged card one of whose parts showed');
    assert.equal(await page.isChecked('#dashPickList input[data-chart="latency"]'), false);
    assert.deepEqual(errors, []);
    await context.close();
  }
  {
    // An order of removed ids only, with a stale hidden list, still shows every card once.
    const { page, errors, context } = await openPage(browser, '/dashboard', { mode: 'advanced', storage: { 'sushi-dash:charts': JSON.stringify({ order: ['e2e', 'otok', 'memsplit'], hidden: ['ptok'] }) } });
    await live(page);
    const ids = await page.$$eval('#dashCharts .chart', (c) => c.map((x) => x.dataset.chart));
    assert.deepEqual(ids.slice(0, 3), ['latency', 'length', 'memory'], 'the old order leads');
    assert.equal(new Set(ids).size, MERGED_IDS.length);
    assert.ok((await shown(page)).includes('length'), 'otok was shown, so the merged card shows');
    assert.deepEqual(errors, []);
    await context.close();
  }
  // Motion: a busy poll eases the canvas charts over a few frames; reduced motion and a hidden tab draw once.
  const framesAfterPoll = (page, hide) => page.evaluate(async (hideTab) => {
    const polls = () => window.__stub.calls.filter((c) => c.path === '/metrics.json').length;
    const n = polls(); while (polls() === n) await new Promise((r) => setTimeout(r, 10));
    await new Promise((r) => setTimeout(r, 60));
    if (hideTab) { Object.defineProperty(document, 'hidden', { value: true, configurable: true }); document.dispatchEvent(new Event('visibilitychange')); }
    const ctx = document.getElementById('dashCCdec').getContext('2d'), real = ctx.clearRect;
    let clears = 0;
    ctx.clearRect = function (...a) { clears += 1; return real.apply(this, a); };
    await new Promise((r) => setTimeout(r, 200));
    return clears;
  }, hide);
  for (const [name, options, hide, want] of [['busy', {}, false, (n) => n >= 3], ['reduced motion', { reducedMotion: 'reduce' }, false, (n) => n === 0], ['hidden tab', {}, true, (n) => n <= 1]]) {
    const { page, errors, context } = await openPage(browser, '/dashboard', { mode: 'compact', compact: 'decode', contextOptions: options });
    await live(page);
    await page.selectOption('#dashWindow', '60');
    const frames = await framesAfterPoll(page, hide);
    assert.ok(want(frames), `${name}: ${frames} redraws of the reply chart after a busy poll`);
    assert.deepEqual(errors, []);
    await context.close();
  }
  {
    // Bars change in place, so their CSS transition plays: the same element takes the new width.
    const { page, errors, context } = await openPage(browser, '/dashboard', { mode: 'advanced' });
    await page.waitForFunction(() => document.querySelectorAll('#dashWrites .hb').length > 2);
    const before = await page.evaluate(() => { window.__bar = document.querySelector('#dashWrites .hb b'); return window.__bar.textContent; });
    await page.waitForFunction((was) => document.querySelector('#dashWrites .hb b').textContent !== was, before, { timeout: 8000 });
    assert.equal(await page.evaluate(() => document.querySelector('#dashWrites .hb b') === window.__bar), true, 'a poll updates the row it already drew');
    assert.notEqual(await page.evaluate(() => getComputedStyle(document.querySelector('#dashWrites .hb .tr > i')).transitionDuration), '0s', 'and the bar eases');
    assert.deepEqual(errors, []);
    await context.close();
  }
  {
    // A meaning keeps its colour across cards: waiting is the same muted grey in Requests as in Response time.
    const h1 = (count, sum) => ({ count, sum, bounds: [0.1, 1, 10], bucket_counts: [0, count, count, count] });
    const specData = { histograms: { time_to_first_token_seconds: h1(2, 2), prefill_time_seconds: h1(2, 1), decode_time_seconds: h1(2, 4), e2e_request_latency_seconds: h1(2, 6) } };
    const { page, errors, context } = await openPage(browser, '/dashboard', { mode: 'advanced', specData });
    await page.waitForSelector('[data-chart="latency"] .parts span');
    const swatch = (sel, word) => page.$$eval(sel, (spans, w) => spans.filter((s) => new RegExp(`^${w}`, 'i').test(s.textContent.trim())).map((s) => getComputedStyle(s.querySelector('i')).backgroundColor), word);
    const muted = await page.evaluate(() => { const i = document.createElement('i'); i.style.color = 'var(--text-3)'; document.body.append(i); const c = getComputedStyle(i).color; i.remove(); return c; });
    assert.deepEqual(await swatch('[data-chart="latency"] .parts span', 'Waiting'), [muted], 'Response time draws waiting in --text-3');
    assert.deepEqual(await swatch('[data-chart="requests"] .legend span', 'waiting'), [muted], 'and so does the Requests chart');
    assert.deepEqual(errors, []);
    await context.close();
  }
  {
    // A focused Cancel names one request: a poll that puts another request in its row never hands it that request,
    // and a poll that moves its request keeps the focus on it.
    const row = (id, phase) => ({ request_id: id, model: 'org/m', phase, context_length: 8192, context_tokens: 100, cached_tokens: 0, generated_tokens: 0, max_tokens: 100, elapsed_seconds: 1, state_bytes: 1 << 20 });
    const focused = (page) => page.evaluate(() => document.activeElement.dataset.cancel ?? null);
    const { page, errors, context } = await openPage(browser, '/dashboard', { mode: 'advanced', sessions: [row(12, 'prefill')] }, true);
    await adv(page, 3000);
    await page.focus('button[data-cancel="12"]');
    await page.evaluate((rows) => { window.__stub.sessions = rows; }, [row(13, 'prefill')]);
    await adv(page, 3000);
    assert.notEqual(await focused(page), '13', 'the Cancel of a request that left does not become the next request\'s');
    await page.evaluate((rows) => { window.__stub.sessions = rows; }, [5, 6, 7].map((id) => row(id, 'decode')));
    await adv(page, 3000);
    await page.focus('button[data-cancel="6"]');
    await page.evaluate((rows) => { window.__stub.sessions = rows; }, [6, 7, 8].map((id) => row(id, 'decode')));
    await adv(page, 3000);
    assert.equal(await focused(page), '6', 'a request that moved up a row keeps its focused Cancel');
    assert.deepEqual(errors, []);
    await context.close();
  }
  {
    // An empty server: every card says what it is waiting for, readable and as wide as its card.
    const zero = { requests: 0, rounds: 0, drafted: 0, accepted: 0, plain_rounds: 0 }, h0 = (b) => ({ count: 0, sum: 0, bounds: b, bucket_counts: Array(b.length + 1).fill(0) });
    const LAT = [0.01, 0.1, 1, 10], TOK = [32, 512, 8192];
    const specData = { histograms: { time_to_first_token_seconds: h0(LAT), e2e_request_latency_seconds: h0(LAT), prefill_time_seconds: h0(LAT), decode_time_seconds: h0(LAT), prompt_tokens: h0(TOK), output_tokens: h0(TOK) },
      speculative: { mtp: zero, dflash: zero, dspark: zero }, prefix_cache_tiers: { hits: { hot: 0, ssd: 0 }, evictions: { hot: 0, ssd: 0 }, evicted_bytes: {}, hot_bytes: 0 }, requests: [] };
    for (const width of [1280, 390]) {
      const { page, errors, context } = await openPage(browser, '/dashboard', { mode: 'advanced', specData, contextOptions: { viewport: { width, height: 900 } } });
      await live(page);
      await page.waitForFunction(() => (window.__axis.dashCSpec ?? []).length > 0);
      const notes = await page.$$eval('#dashCharts .chart:not([hidden]) .none', (ns) => ns.map((n) => { const c = n.closest('.chart').getBoundingClientRect(), r = n.getBoundingClientRect(); return { card: n.closest('.chart').dataset.chart, text: n.textContent, share: r.width / c.width, lines: Math.round(r.height / parseFloat(getComputedStyle(n).lineHeight)) }; }));
      const by = Object.fromEntries(notes.map((n) => [n.card, n]));
      assert.equal(by.cachereuse?.text, 'No request in this window yet', `${width}: the cache reuse columns say why they are empty`);
      for (const n of notes) assert.ok(n.lines <= 2, `${width}: ${n.card} "${n.text}" is not squeezed (${n.lines} lines)`);
      assert.match(by.latency?.text ?? '', /No finished request yet/, `${width}: response time`);
      assert.match(by.length?.text ?? '', /No finished request yet/, `${width}: prompt and generated length`);
      assert.ok((await axisOf(page, 'dashCSpec')).some((t) => /No request used speculative decoding/.test(t)), `${width}: draft acceptance`);
      assert.match(await textOf(page, 'dashReuseHero'), /^-/, `${width}: no share before a prompt is read`);
      assert.deepEqual(errors, []);
      await context.close();
    }
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
    await dashboardV2(browser);
    await dashboardBugs(browser);
    await chartPicker(browser);
    await chartExtras(browser);
    await chartAudit(browser);
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
      // A gridline's label is that line's value: a 2,500 scale reads 1.875k, never a rounded neighbour (a count chart steps in whole numbers: see chartExtras).
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
      assert.equal(await page.evaluate(() => document.getElementById('dashCCpre').width), 300, 'a hidden dashboard draws nothing');

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
      assert.notEqual(await page.evaluate(() => document.getElementById('dashCCpre').width), 300, 'a visible dashboard draws its charts');

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
      await page.waitForFunction(() => { const o = document.getElementById('dashOff'); return !o.hidden && /not collecting live statistics/.test(o.textContent); });
      await page.waitForFunction(() => document.getElementById('serverDotLabel').textContent.includes('test-model'));
      await context.close();
    }

    {
      const { page, errors, context } = await openPage(browser, '/dashboard', { metricsStatus: 503, key: 'k-12345' });
      await page.waitForFunction(() => { const o = document.getElementById('dashOff'); return !o.hidden && /not collecting live statistics/.test(o.textContent); });
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
      await page.waitForFunction(() => { const o = document.getElementById('dashOff'); return !o.hidden && /not collecting live statistics/.test(o.textContent); });
      await page.click('#dashEnableMetrics');
      await page.waitForFunction(() => /Could not turn on live statistics/.test(document.getElementById('dashStatus').textContent));
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
      await page.waitForFunction(() => { const o = document.getElementById('dashOff'); return !o.hidden && /not collecting live statistics/.test(o.textContent); });
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
      const rows = () => page.evaluate(() => Object.fromEntries([...document.querySelectorAll('#dashWrites .hb')].map((row) => [row.querySelector('span').textContent, row.querySelector('b').textContent])));
      const foot = () => textOf(page, 'dashWritesFoot');
      await page.waitForFunction(() => /Live/.test(document.getElementById('dashStatus').textContent));
      // The rate needs two polls and the ratio needs /props: wait for both, then read one consistent render.
      await page.waitForFunction(() => /SSD writes .*\/s, [\d.]+x written per byte/.test(document.getElementById('dashWritesFoot').innerText), null, { timeout: 15000 });
      const card = await rows();
      for (const label of ['Attention memory', 'Restore points', 'Long-context index', 'Draft state', 'Sliding-window points', 'Indexes and token lists', 'Failed or dropped']) {
        assert.ok(label in card, `the chart has a ${label} bar`);
      }
      assert.equal(card['Failed or dropped'], '3.0 MiB', 'failed bytes are named apart from the written ones');
      const gib = (text) => parseFloat(text) * (/GiB/.test(text) ? 1 : /MiB/.test(text) ? 1 / 1024 : 1 / 1048576);
      assert.ok(Math.abs(parseFloat((await foot()).match(/([\d.]+)x written/)[1]) - gib(card['Attention memory']) / 2) < 0.1, 'written over the 2 GiB on disk');
      assert.deepEqual(errors, []);
      await context.close();
    }

    {
      const { page, errors, context } = await openPage(browser, '/dashboard', { compact: 'decode' });
      const axisTop = (id) => page.evaluate((canvas) => Math.max(0, ...(window.__axis[canvas] ?? []).filter((t) => /^\d+(\.\d+)?k?$/.test(t)).map((t) => parseFloat(t) * (t.endsWith('k') ? 1000 : 1))), id);
      await page.waitForFunction(() => /Live/.test(document.getElementById('dashStatus').textContent));
      const cards = await page.evaluate(() => [...document.querySelectorAll('#dashCharts .chart canvas')].map((cv) => ({ title: cv.closest('.card').querySelector('h2').firstChild.textContent, legends: cv.closest('.card').querySelectorAll('.legend').length, canvas: cv.id })));
      assert.deepEqual(cards.slice(0, 2).map((card) => card.title), ['Prompt speed (tokens/s)', 'Generation speed (tokens/s)'], 'prompt and generation speed are separate cards');
      assert.deepEqual(cards.slice(0, 2).map((card) => card.legends), [0, 0], 'a single series needs no legend');
      assert.equal(new Set(cards.map((card) => card.canvas)).size, cards.length, 'every card draws on its own canvas');
      assert.equal(await page.evaluate(() => /\bdecode\b/i.test(document.getElementById('dashPanel').innerText)), false, 'the dashboard says generation, not decode (that is the technical name in a tooltip)');
      // The Advanced view draws prefill and decode the way the Compact view does: one dot per finished request, decode on its own scale.
      assert.deepEqual(await page.evaluate(() => [...document.querySelectorAll('#dashCharts .chart .sub')].map((e) => e.textContent).slice(0, 2)), ['one point per finished request', 'while generating']);
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
      assert.equal(await text('dashCmState'), compact === 'decode' ? 'Generating' : 'Idle');

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
      const axis = await page.evaluate(() => window.__axis.dashCMem ?? []);
      assert.ok(axis.includes('128') && !axis.includes('200'), `the memory axis ends at the machine's memory (${axis})`);
      assert.ok(axis.some((t) => /128 GiB machine memory/.test(t)), 'and the ceiling is labelled');

      // Narrow: decode and prefill share the first row, the rest stack; the strip still one line, nothing wider than the screen.
      await page.setViewportSize({ width: 390, height: 844 });
      await page.waitForTimeout(300);
      const narrow = await page.evaluate(() => ({ strip: document.querySelector('.cm-strip').getBoundingClientRect().height, wide: document.getElementById('dashPanel').scrollWidth - document.getElementById('dashPanel').clientWidth,
        tiles: new Set([...document.querySelectorAll('.tiles-cmp .tile')].map((t) => Math.round(t.getBoundingClientRect().x))).size,
        firstRow: [...document.querySelectorAll('.tiles-cmp .tile')].filter((t, _, all) => Math.round(t.getBoundingClientRect().y) === Math.round(all[0].getBoundingClientRect().y)).length,
        charts: [...document.querySelectorAll('#dashCharts .chart:not([hidden])')].map((c) => Math.round(c.getBoundingClientRect().x)) }));
      assert.ok(narrow.strip < 70 && narrow.wide <= 0 && narrow.tiles === 2 && narrow.firstRow === 2, `390 px stacks without overflow (${JSON.stringify(narrow)})`);
      assert.equal(new Set(narrow.charts).size, 1, 'the charts stack');
      assert.deepEqual(errors, []);
      await context.close();
    }

    {
      const { page, errors, context } = await openPage(browser, '/dashboard', { mode: 'compact', compact: 'prefill' });
      await page.waitForFunction(() => document.getElementById('dashCmState').textContent === 'Reading prompt', null, { timeout: 15000 });
      assert.match(await page.innerText('#dashCmReqS'), /reading prompt$/, 'the request tile uses the same words');
      assert.deepEqual(errors, []);
      await context.close();
    }

    {
      // Requests that only wait their turn are neither reading a prompt nor generating.
      const { page, errors, context } = await openPage(browser, '/dashboard', { mode: 'compact', compact: 'waiting' });
      await page.waitForFunction(() => !['Connecting', '-', ''].includes(document.getElementById('dashCmState').textContent), null, { timeout: 15000 });
      assert.equal(await page.textContent('#dashCmState'), 'Waiting');
      assert.match(await page.innerText('#dashCmReqS'), /^1 waiting$/, 'the request tile uses the same word');
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
      // The GPU chart draws utilization and the clock as a share of its top on one 0-100% axis (no second scale); the
      // history backfill carries both, and a box with no readable clock says so and keeps the utilization.
      const id = 'dashCGpuc';
      const hoverTip = async (page) => {
        await page.evaluate((i) => document.getElementById(i).scrollIntoView({ block: 'center' }), id);
        const box = await page.evaluate((i) => { const b = document.getElementById(i).getBoundingClientRect(); return { x: b.left, y: b.top, w: b.width, h: b.height }; }, id);
        await page.mouse.move(box.x + box.w - 60, box.y + box.h / 2);
        await page.waitForFunction(() => getComputedStyle(document.getElementById('dashTip')).display === 'block', null, { timeout: 5000 });
        return page.evaluate(() => document.getElementById('dashTip').innerText.replace(/\s+/g, ' '));
      };
      const { page, errors, context } = await openPage(browser, '/dashboard', { mode, history: true });
      await page.waitForFunction(() => /Live/.test(document.getElementById('dashStatus').textContent));
      await page.waitForFunction((i) => (window.__axis[i] ?? []).includes('100%'), id, { timeout: 15000 });
      const axis = (await axisOf(page, id)).filter((t) => /^[\d,]+%?$/.test(t));
      assert.deepEqual(axis, ['0%', '25%', '50%', '75%', '100%'], `${mode}: one axis, 0 to 100% (${axis})`);
      assert.ok(await page.evaluate(() => document.getElementById('dashCGpuc').closest('.card').querySelector('.legend').innerText.includes('clock')), `${mode}: the legend names the clock`);
      const tip = await hoverTip(page);
      assert.match(tip, /GPU clock\s*1,200 MHz \(74%\)/, `${mode}: the tooltip names the clock and its share of the top speed (${tip})`);
      assert.match(tip, /Busy\s*50%/, `${mode}: and the utilization (${tip})`);
      assert.equal(await textOf(page, 'dashCGpucL'), '', `${mode}: no unavailable note`);
      assert.deepEqual(errors, []);
      await context.close();

      const off = await openPage(browser, '/dashboard', { mode, history: true, noClock: true });
      await off.page.waitForFunction((i) => (off => off.length > 4)(window.__axis[i] ?? []), id, { timeout: 15000 });
      const offAxis = await axisOf(off.page, id);
      assert.ok(offAxis.includes('100%'), `${mode}: without a clock the axis is utilization alone (${offAxis})`);
      assert.equal(await off.page.isVisible('#dashCGpucK'), false, `${mode}: and the legend drops the clock`);
      assert.match(await textOf(off.page, 'dashCGpucL'), /Clock speed not reported/, `${mode}: the chart says the clock is unavailable`);
      const offTip = await hoverTip(off.page);
      assert.ok(!/GPU clock/.test(offTip) && /Busy\s*50%/.test(offTip), `${mode}: the tooltip shows utilization only (${offTip})`);
      assert.deepEqual(off.errors, []);
      await off.context.close();
    }

    {
      // Hovering a chart costs one redraw per frame and reads no style per colour (motion off: a busy poll's easing is its own frames).
      const { page, errors, context } = await openPage(browser, '/dashboard', { compact: 'decode', mode: 'compact', contextOptions: { reducedMotion: 'reduce' } });
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
      assert.ok(await page.evaluate(() => document.getElementById('dashHLat').children.length > 0), 'Advanced is filled the moment it is shown');
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
