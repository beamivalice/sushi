// sushi-test: tier=hermetic needs=node,playwright
// The Models and Settings tabs in a real headless browser against a stubbed server: no sushi server, no model.
// Needs Playwright with a Chromium build (global or local install); skips without one.
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
const GiB = 2 ** 30;
const KEYS = [
  { name: 'ctx_size', type: 'int', min: 1, max: 4294967295 }, { name: 'kv_quant', type: 'choice', choices: ['off', '4', '6', '8'] },
  { name: 'mtp', type: 'bool' }, { name: 'mtp_acceptance', type: 'choice', choices: ['exact', 'typical', 'tokenv3'] }, { name: 'mtp_greedy_tail', type: 'bool' },
  { name: 'ssd_budget_gb', type: 'int', min: 1, max: 512 }, { name: 'ssd_prefetch', type: 'int', min: 0, max: 100 },
  { name: 'preserve_thinking', type: 'bool' }, { name: 'think_penalty', type: 'number', min: 0, max: 20 }, { name: 'vision', type: 'bool' },
  { name: 'max_tokens', type: 'int', min: 1, max: 4294967295 },
];
const keyState = (extra = {}) => ({ file: null, flag: null, source: 'default', locked: false, ...extra });
const BASE_KEYS = ['ctx_size', 'kv_quant', 'mtp', 'mtp_acceptance', 'mtp_greedy_tail', 'preserve_thinking', 'think_penalty', 'max_tokens'];
const allKeys = (over = {}) => Object.fromEntries(KEYS.map((k) => [k.name, over[k.name] ?? keyState()]));

// A stub server: its state lives here in node, every call is recorded, and a test can hold a response back.
function makeServer(opts = {}) {
  const calls = [];
  const gates = {};
  const state = {
    failSettings: 0, loadDelay: null, settingsDelay: [], loadFails: false, down: false, hooks: {},
    models: [
      { id: 'Qwen-A', state: 'ready', bytes_on_disk: 38 * GiB, bytes_resident: 40 * GiB, context_length: 262144, input_modalities: ['text', 'image'], keys: [...BASE_KEYS, 'vision'], settings: allKeys({ ctx_size: keyState({ file: 65536, source: 'file' }), kv_quant: keyState({ flag: '8', source: 'flag', locked: true }) }) },
      { id: 'GLM-B', state: 'unloaded', bytes_on_disk: 61 * GiB, context_length: 131072, input_modalities: ['text'], keys: BASE_KEYS, settings: allKeys({ kv_quant: keyState({ file: 'off', source: 'file' }) }) },
      { id: 'Deep-C', state: 'unloaded', bytes_on_disk: 96 * GiB, bytes_linked: 40 * GiB, pack_update: { current: '1.0', latest: '2.0', command: 'sushi pull deepseek-v4.1-flash', repo: 'beamster/Deep-Sushi' }, streaming: true, context_length: 131072, input_modalities: ['text'], keys: [...BASE_KEYS, 'ssd_budget_gb', 'ssd_prefetch'], settings: allKeys() },
      { id: 'Broke-D', state: 'error_state', bytes_on_disk: 20 * GiB, context_length: 8192, input_modalities: ['text'], keys: BASE_KEYS, settings: allKeys() },
    ],
    server: {
      models_dir: { default: '/h/.sushi/models', boot_roots: ['/models'], file: null, flag: '', source: 'default', locked: !!opts.dirLocked, applies: 'models are rescanned at once' },
      prefix_cache_disk_gb: { file: 100, flag: null, source: 'file', locked: false, effective_bytes: 100 * GiB, ask_bytes: 100 * GiB, applies: 'a running tier takes a new size at once and evicts down to it; 0 (off) and removing the key apply at the next load' },
      update_check: { file: null, flag: null, default: true, source: 'default', locked: false, applies: 'applies at once' },
      pack_update_check: { file: null, flag: null, default: true, source: 'default', locked: false, applies: 'applies at once' },
      skip_mem_preflight: { file: null, flag: null, default: false, source: 'default', locked: false, applies: 'applies at once' },
      port: { file: null, flag: null, default: 12345, source: 'default', locked: false, applies: 'the next start; a running server keeps its port' },
    },
    cache: {
      root: '/h/.sushi/kv-cache', volume: { free_bytes: 1399 * GiB, total_bytes: 3722 * GiB }, total_bytes: 71.5 * GiB,
      namespaces: [
        { id: 'aaaaaaaa11111111', bytes: Math.round(41.2 * GiB), entries: 37, newest_mtime: Date.now() / 1000 - 2 * 86400, model: 'Qwen-A', stale: false, held_by_other_process: false, tier: { used_bytes: 41 * GiB, cap_bytes: 100 * GiB, entries: 37, written_bytes: 5 * GiB } },
        { id: 'bbbbbbbb22222222', bytes: Math.round(22.1 * GiB), entries: 9, newest_mtime: Date.now() / 1000 - 3 * 86400, model: null, stale: true, held_by_other_process: false, tier: null },
        { id: 'cccccccc33333333', bytes: Math.round(8.2 * GiB), entries: 3, newest_mtime: Date.now() / 1000 - 7200, model: null, stale: false, held_by_other_process: true, tier: null },
      ],
    },
  };
  const modelsBody = () => ({ data: state.models.map((m) => ({ id: m.id, object: 'model', loaded: m.state === 'ready', state: m.state === 'error_state' ? 'error' : m.state, bytes_resident: m.bytes_resident ?? 0, bytes_on_disk: m.bytes_on_disk, bytes_linked: m.bytes_linked ?? 0, streaming: !!m.streaming, streaming_required: !!m.streaming_required, ...(m.error ? { error: m.error } : {}), ssd_budget_gb: m.ssd_budget_gb, context_length: m.context_length, input_modalities: m.input_modalities, capabilities: ['chat'], meta: { architecture: 'qwen4_exp', quantization: '2.75-bit', ...(m.pack_update ? { pack_update: m.pack_update } : {}) } })) });
  const settingsBody = () => ({ file: '/h/.sushi/model-settings.json', applies: 'next load of the model', model_keys: KEYS, server: state.server,
    models: state.models.map((m) => ({ id: m.id, path: `/models/${m.id}`, state: m.state, bytes_on_disk: m.bytes_on_disk, settings: m.settings, keys: m.keys, loaded_with: m.loaded_with ?? null, defaults: m.defaults ?? null, ...(opts.limits && m.limits ? { limits: m.limits } : {}) })) });
  if (opts.limits) {
    state.models[0].limits = { ctx_size: { max: 131072, why: "This model's context window is 131072 tokens." }, kv_quant: { choices: ['off', '8'], why: '4-bit and 6-bit need a different cache layout.' },
      mtp_acceptance: { requires: { mtp: true }, why: 'Used only while MTP is on.' }, mtp_greedy_tail: { requires: { mtp: true }, why: 'Used only while MTP is on.' } };
    state.models[2].keys = ['ctx_size', 'mtp', 'ssd_budget_gb', 'ssd_prefetch', 'think_penalty', 'vision'];
    state.models[2].limits = { kv_quant: { disabled: true, why: 'This model keeps no KV cache to quantize.' }, preserve_thinking: { disabled: true, why: 'No preserve-thinking switch in this chat template.' },
      mtp: { unset: ['ssd_budget_gb'], why: 'An SSD budget makes this model stream its experts, and its MTP head cannot run while they stream.' } };
  }
  const json = (body, status = 200) => ({ status, contentType: 'application/json', body: JSON.stringify(body) });
  const hold = (name) => new Promise((resolve) => { gates[name] = resolve; });
  async function handle(route) {
    const request = route.request(), url = new URL(request.url()), p = url.pathname, method = request.method();
    if (/^\/(chat|dashboard|models|settings)?$/.test(p)) return route.fulfill({ status: 200, contentType: 'text/html; charset=utf-8', body: html });
    const body = method === 'POST' ? JSON.parse(request.postData() || '{}') : null;
    calls.push({ path: p, method, body, auth: request.headers().authorization });
    if (state.down) return route.abort('failed');
    const hook = state.hooks[p];
    if (hook) { const answer = await hook(body, method); if (answer) return route.fulfill(answer); }
    switch (p) {
      case '/v1/models': return route.fulfill(json(modelsBody()));
      case '/v1/settings':
        if (method === 'POST') {
          const m = state.models.find((x) => x.id === body.model);
          if (opts.limits && body.set && 'think_penalty' in body.set) return route.fulfill(json({ error: { message: 'think_penalty is not supported by this model' } }, 400));
          for (const [k, v] of Object.entries(body.set ?? {})) if (m) m.settings[k] = keyState({ file: v, source: v === null ? 'default' : 'file' });
          for (const [k, v] of Object.entries(body.server ?? {})) { state.server[k] = { ...state.server[k], file: v, source: v === null ? 'default' : 'file' }; if (k === 'prefix_cache_disk_gb') state.server[k].effective_bytes = state.server[k].ask_bytes = (v ?? 20) * GiB; }
          const cacheNote = body.server?.prefix_cache_disk_gb ? 'applied to the running tiers' : 'saved; applies at the next model load';
          return route.fulfill(json({ applied: Object.fromEntries(Object.keys(body.set ?? body.server ?? {}).map((k) => [k, k === 'prefix_cache_disk_gb' ? cacheNote : 'saved'])), model_loaded: m?.state === 'ready' }));
        }
        if (state.failSettings > 0) { state.failSettings -= 1; return route.fulfill({ status: 500, contentType: 'text/html', body: '<html><body><h1>Bad gateway</h1></body></html>' }); }
        { const answer = json(settingsBody()); if (state.settingsDelay.length) await state.settingsDelay.shift(); return route.fulfill(answer); }
      case '/v1/prefix-cache': return route.fulfill(json(state.cache));
      case '/v1/prefix-cache/clear': {
        if (body.scope === 'namespace') { const n = state.cache.namespaces.find((x) => x.id === body.id); state.cache.namespaces = state.cache.namespaces.filter((x) => x !== n); return route.fulfill(json({ stale_namespaces_removed: 1, stale_freed_bytes: n.bytes, skipped_live_namespaces: 0, refused_symlinks: 0 })); }
        const freed = body.scope === 'stale' ? state.cache.namespaces.filter((n) => n.stale).reduce((a, n) => a + n.bytes, 0) : 123;
        if (body.scope === 'stale') state.cache.namespaces = state.cache.namespaces.filter((n) => !n.stale);
        return route.fulfill(json({ freed_bytes: freed }));
      }
      case '/v1/load-model': {
        if (state.loadDelay) await state.loadDelay;
        const m = state.models.find((x) => x.id === body.model);
        if (state.loadFails) return route.fulfill(json({ error: { message: 'out of memory: needs 61 GiB' } }, 503));
        m.state = 'ready'; m.bytes_resident = m.bytes_on_disk;
        return route.fulfill(json({ ok: true }));
      }
      case '/v1/unload-model': { state.models.find((x) => x.id === body.model).state = 'unloaded'; return route.fulfill(json({ ok: true })); }
      case '/v1/models/rescan': return route.fulfill(json({ added: 2 }));
      case '/props': return route.fulfill(json({ settings: { version: '9.9.9', mtp: {}, prefix_cache: {} }, memory: { available_bytes: 5e9, total_bytes: 128 * GiB }, batching: {} }));
      case '/v1/tools': return route.fulfill(json({ edit_default: false }));
      case '/metrics.json': case '/metrics': return route.fulfill(json({ error: { message: 'metrics off' } }, 503));
      default: return route.fulfill(json({}, 404));
    }
  }
  return { state, calls, handle, hold, gates, count: (p, method) => calls.filter((c) => c.path === p && (!method || c.method === method)).length };
}

async function open(browser, server, urlPath, { viewport = { width: 1440, height: 900 }, mode = 'compact', clock = true, scheme = 'dark', storage = {} } = {}) {
  const context = await browser.newContext({ viewport, colorScheme: scheme });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('dialog', (dialog) => { errors.push(`native dialog: ${dialog.message()}`); dialog.dismiss(); });
  await page.route(`${ORIGIN}/**`, server.handle);
  await page.addInitScript(([m, extra]) => { try { localStorage.setItem('sushi-ui-mode', m); for (const [k, v] of Object.entries(extra)) localStorage.setItem(k, v); } catch { /* storage off */ } }, [mode, storage]);
  if (clock) await page.clock.install();
  await page.goto(`${ORIGIN}${urlPath}`);
  return { page, context, errors };
}
const settle = async (page, ms = 0) => { if (ms) await page.clock.runFor(ms); await page.waitForTimeout(150); };
const row = (key) => `.set[data-key="${key}"]`;

async function main() {
  const playwright = loadPlaywright();
  if (!playwright) { console.log('Web UI admin views (browser): skipped (Playwright is not installed)'); return; }
  let browser;
  try { browser = await playwright.chromium.launch(); } catch (error) {
    console.log(`Web UI admin views (browser): skipped (no Chromium: ${String(error.message).split('\n')[0]})`);
    return;
  }
  try {
    {
      // An edit in progress survives the polls, keeps focus, and an unchanged answer touches no DOM.
      const server = makeServer();
      const { page, context, errors } = await open(browser, server, '/models#Qwen-A');
      await page.waitForSelector(`${row('ctx_size')} input`);
      await page.fill(`${row('ctx_size')} input`, '4096');
      await page.evaluate(() => { window.__input = document.querySelector('.set[data-key="ctx_size"] input'); window.__mutations = 0; window.__where = []; new MutationObserver((list) => { for (const m of list.filter((x) => x.type === 'childList')) { window.__mutations += 1; window.__where.push(`${m.target.id || m.target.className || m.target.tagName}`); } }).observe(document.getElementById('modelsPanel'), { subtree: true, childList: true }); });
      await settle(page, 16_000);
      assert.equal(await page.inputValue(`${row('ctx_size')} input`), '4096', 'the typed value survives three polls');
      assert.equal(await page.evaluate(() => document.activeElement === window.__input), true, 'and so does focus');
      assert.equal(await page.evaluate(() => window.__input === document.querySelector('.set[data-key="ctx_size"] input')), true, 'the control is the same element');
      assert.ok(await page.isVisible('.savebar'), 'the save bar shows for a dirty form');
      assert.match(await page.textContent('.savebar'), /1 change/);
      assert.ok(server.count('/v1/settings', 'GET') >= 2, 'the page did poll');
      await page.click('.savebar button:text-is("Discard")');
      assert.equal(await page.inputValue(`${row('ctx_size')} input`), '65536', 'Discard restores the saved value');
      assert.equal(await page.isVisible('.savebar'), false);
      await page.evaluate(() => { window.__mutations = 0; window.__where = []; });
      await settle(page, 16_000);
      assert.equal(await page.evaluate(() => window.__mutations), 0, `identical answers cause no childList mutation (${await page.evaluate(() => window.__where)})`);
      assert.deepEqual(errors, []);
      await context.close();
    }

    for (const viewport of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
      // Every Default says what the server would use; a model with no known default keeps the bare word.
      const server = makeServer();
      server.state.models[0].defaults = { kv_quant: '8', mtp: true, mtp_acceptance: 'exact', mtp_greedy_tail: false, preserve_thinking: true, think_penalty: 0, vision: false, max_tokens: 256000 };
      const { page, context, errors } = await open(browser, server, '/models#Qwen-A', { viewport, mode: 'advanced' });
      await page.waitForSelector(`${row('max_tokens')} input`);
      const first = (key) => page.evaluate((k) => { const r = document.querySelector(`.set[data-key="${k}"]`); const o = r.querySelector('select option'); return o ? o.textContent : r.querySelector('.tri button').textContent; }, key);
      assert.equal(await page.getAttribute(`${row('max_tokens')} input`, 'placeholder'), 'Default (256,000)', 'a number shows its default with separators');
      assert.match(await page.textContent(`${row('max_tokens')} .lab`), /Max generated tokens/);
      assert.equal(await page.getAttribute(`${row('ctx_size')} input`, 'placeholder'), 'Default', 'a default measured at load is not guessed');
      assert.equal(await first('kv_quant'), 'Default (8-bit)', 'a choice shows its label');
      assert.equal(await first('mtp_acceptance'), 'Default (Exact)');
      assert.equal(await first('mtp'), 'Default (On)', 'a switch shows On or Off');
      assert.equal(await first('mtp_greedy_tail'), 'Default (Off)');
      assert.equal(await page.getAttribute(`${row('think_penalty')} input`, 'placeholder'), 'Default (0)');
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true, 'no horizontal overflow');
      await page.evaluate(() => { window.__mutations = 0; window.__where = []; new MutationObserver((list) => { window.__mutations += list.length; for (const m of list) window.__where.push(`${m.type}:${m.attributeName ?? ''}:${m.target.nodeName}.${m.target.className}`); }).observe(document.querySelector('#modelsDetail .rows'), { subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ['placeholder'] }); });
      await settle(page, 16_000);
      assert.equal(await page.evaluate(() => window.__mutations), 0, `an unchanged poll writes nothing (${await page.evaluate(() => window.__where.slice(0, 5))})`);
      server.state.models[0].defaults = { ...server.state.models[0].defaults, kv_quant: 'off', max_tokens: 0, mtp: false };
      await settle(page, 16_000);
      assert.equal(await first('kv_quant'), 'Default (16-bit)');
      assert.equal(await first('mtp'), 'Default (Off)');
      assert.equal(await page.getAttribute(`${row('max_tokens')} input`, 'placeholder'), 'Default (rest of the context window)', 'no cap of its own means the window');
      await page.goto(`${ORIGIN}/models#GLM-B`);
      await page.waitForSelector(`${row('max_tokens')} input`);
      assert.equal(await page.getAttribute(`${row('max_tokens')} input`, 'placeholder'), 'Default', 'a model that is not loaded keeps the bare word');
      assert.equal(await first('kv_quant'), 'Default');
      assert.equal(await first('mtp'), 'Default');
      assert.deepEqual(errors, []);
      await context.close();
    }

    {
      // Save: one POST with only the changed key, a toast that outlives the refresh, then the new source badge.
      const server = makeServer();
      const { page, context, errors } = await open(browser, server, '/models#Qwen-A');
      await page.waitForSelector(`${row('ctx_size')} input`);
      await page.fill(`${row('ctx_size')} input`, '4096');
      await page.click('.savebar button:text-is("Save")');
      await page.waitForSelector('.toast');
      await settle(page, 1000);
      const posts = server.calls.filter((c) => c.path === '/v1/settings' && c.method === 'POST');
      assert.equal(posts.length, 1);
      assert.deepEqual(posts[0].body, { model: 'Qwen-A', set: { ctx_size: 4096 } });
      assert.match(await page.textContent('#toastHost'), /Saved/, 'the confirmation is still on screen after the refresh');
      assert.equal(await page.isVisible('.savebar'), false, 'the form is clean again');
      assert.match(await page.textContent('.callout'), /Reload now|differ/, 'a loaded model offers a reload');
      await settle(page, 16_000);
      assert.equal(await page.inputValue(`${row('ctx_size')} input`), '4096', 'and the saved value is what the page shows');
      assert.match(await page.textContent(row('ctx_size')), /Saved/);
      assert.deepEqual(errors, []);
      await context.close();
    }

    {
      // Half a number is a bad value: no request, a visible reason, the key is not deleted.
      const server = makeServer();
      const { page, context, errors } = await open(browser, server, '/models#Qwen-A');
      await page.waitForSelector(`${row('ctx_size')} input`);
      await page.fill(`${row('ctx_size')} input`, '1e');
      assert.equal(await page.getAttribute(`${row('ctx_size')} input`, 'aria-invalid'), 'true');
      assert.ok(await page.isVisible(`${row('ctx_size')} .bad`));
      assert.equal(await page.isDisabled('.savebar button:text-is("Save")'), true, 'Save waits for a valid value');
      await page.evaluate(() => [...document.querySelectorAll('.savebar button')].find((b) => b.textContent === 'Save').click());
      assert.equal(server.count('/v1/settings', 'POST'), 0, 'nothing is sent');
      assert.deepEqual(errors, []);
      await context.close();
    }

    {
      // A file value stored as the number 0 is shown as "off" and an unrelated Save leaves it alone.
      const server = makeServer();
      const { page, context, errors } = await open(browser, server, '/models#GLM-B');
      await page.waitForSelector(`${row('kv_quant')} select`);
      assert.equal(await page.inputValue(`${row('kv_quant')} select`), 'off');
      await page.click(`${row('mtp')} button:text-is("On")`);
      await page.click('.savebar button:text-is("Save")');
      await page.waitForSelector('.toast');
      assert.deepEqual(server.calls.find((c) => c.method === 'POST' && c.path === '/v1/settings').body, { model: 'GLM-B', set: { mtp: true } });
      assert.deepEqual(errors, []);
      await context.close();
    }

    {
      // The user's choice for the "More settings" fold, and the model they picked, survive polls.
      const server = makeServer();
      const { page, context } = await open(browser, server, '/models#Qwen-A', { mode: 'advanced' });
      await page.waitForSelector('details.more-set');
      assert.equal(await page.evaluate(() => document.querySelector('details.more-set').open), true, 'Advanced opens it');
      await page.click('details.more-set > summary');
      await settle(page, 16_000);
      assert.equal(await page.evaluate(() => document.querySelector('details.more-set').open), false, 'a collapsed fold stays collapsed');
      await context.close();
    }

    {
      // A failing answer shows a short alert, never HTML; the next good answer clears it.
      const server = makeServer();
      const { page, context } = await open(browser, server, '/models#Qwen-A');
      await page.waitForSelector(`${row('ctx_size')} input`);
      server.state.failSettings = 1;
      await settle(page, 16_000);
      assert.match(await page.textContent('#modelsPanel .alert.err'), /The server answered with error 500/);
      assert.equal(await page.evaluate(() => /<html>/.test(document.getElementById('modelsPanel').textContent)), false);
      await settle(page, 16_000);
      assert.equal(await page.isVisible('#modelsPanel .alert.err'), false, 'the alert goes when the server answers again');
      await context.close();
    }

    {
      // Load: one request however fast the double click, the row shows Loading, the poll keeps going, the result is announced.
      const server = makeServer();
      let release;
      server.state.loadDelay = new Promise((resolve) => { release = resolve; });
      const { page, context, errors } = await open(browser, server, '/models#GLM-B');
      await page.waitForSelector('.d-actions button');
      await page.dblclick('.d-actions >> text=Load');
      await settle(page);
      assert.equal(server.count('/v1/load-model'), 1, 'a double click is one request');
      assert.match(await page.textContent('.d-actions'), /Loading/);
      const before = server.count('/v1/models');
      for (let i = 0; i < 5; i++) await settle(page, 2000);
      assert.ok(server.count('/v1/models') - before >= 3, `polling continues while a load is pending (${server.count('/v1/models') - before})`);
      assert.match(await page.textContent('.list [data-id="GLM-B"]'), /Loading/);
      await page.click('.list [data-id="Deep-C"] button');
      await settle(page);
      assert.equal(server.count('/v1/load-model'), 2, 'another model can load meanwhile');
      release();
      await page.waitForFunction(() => /Loaded/.test(document.getElementById('toastHost').textContent));
      await settle(page, 6000);
      assert.match(await page.textContent('.d-actions'), /Unload/);
      assert.deepEqual(errors, []);
      await context.close();
    }

    {
      // A failed load shows the server's reason where the user is looking and leaves Load available.
      const server = makeServer();
      server.state.loadFails = true;
      const { page, context } = await open(browser, server, '/models#GLM-B');
      await page.waitForSelector('.d-actions button');
      await page.click('.d-actions >> text=Load');
      await page.waitForSelector('.detail .alert.err');
      assert.match(await page.textContent('.detail .alert.err'), /out of memory: needs 61 GiB/);
      assert.equal(await page.isEnabled('.d-actions >> text=Load'), true);
      await context.close();
    }

    {
      // The failed model's raw state word never reaches the screen.
      const server = makeServer();
      const { page, context } = await open(browser, server, '/models');
      await page.waitForSelector('.list [data-id="Broke-D"]');
      const text = await page.textContent('.list [data-id="Broke-D"]');
      assert.match(text, /Load failed/);
      assert.doesNotMatch(text, /error_state/);
      assert.ok(await page.isVisible('.list [data-id="Broke-D"] button'), 'a failed model can be loaded again');
      await context.close();
    }

    {
      // The oldest answer never overwrites a newer one.
      const server = makeServer();
      let slow;
      server.state.settingsDelay.push(new Promise((resolve) => { slow = resolve; }));
      const { page, context } = await open(browser, server, '/models#Qwen-A');
      await page.waitForTimeout(300);
      server.state.models[0].settings.ctx_size = keyState({ file: 2048, source: 'file' });
      await page.evaluate(() => window.sushiAdmin.refresh());
      await page.waitForSelector(`${row('ctx_size')} input`);
      assert.equal(await page.inputValue(`${row('ctx_size')} input`), '2048');
      slow();
      await page.waitForTimeout(400);
      assert.equal(await page.inputValue(`${row('ctx_size')} input`), '2048', 'the late, older answer is dropped');
      await context.close();
    }

    {
      // A flag-locked key shows the flag's value and says why; a key the pack cannot use is not listed.
      const server = makeServer();
      const { page, context } = await open(browser, server, '/models#Qwen-A');
      await page.waitForSelector(row('kv_quant'));
      assert.equal(await page.$eval(`${row('kv_quant')} select`, (n) => n.selectedOptions[0].textContent), '8-bit');
      assert.equal(await page.isDisabled(`${row('kv_quant')} select`), true);
      assert.match(await page.textContent(row('kv_quant')), /--kv-quant 8/);
      assert.match(await page.textContent(row('kv_quant')), /Set at launch/);
      assert.equal(await page.isVisible(row('vision')), true, 'an image model lists its vision tower');
      assert.equal(await page.locator(row('ssd_budget_gb')).count(), 0, 'a resident model has no SSD budget row');
      await page.click('.list [data-id="Deep-C"]');
      await page.waitForSelector(row('ssd_budget_gb'));
      assert.equal(await page.isVisible(row('ssd_budget_gb')), true, 'a streamed model shows it in Compact');
      assert.equal(await page.locator(row('vision')).count(), 0, 'a text model has no vision row');
      assert.equal(await page.evaluate(() => location.hash), '#Deep-C');
      // `streaming` only says the pack can stream; with no budget it loads resident.
      assert.match(await page.textContent('.detail .facts'), /can run partly from SSD/);
      assert.match(await page.textContent('.detail .facts'), /96\.0 GiB on disk.*40\.0 GiB linked from elsewhere/);
      assert.doesNotMatch(await page.textContent('.detail .facts'), /sushi pull/, 'an update is never answered with sushi pull');
      await page.click('.list [data-id="Qwen-A"]');
      await page.waitForFunction(() => document.querySelector('#modelsDetail h2')?.textContent === 'Qwen-A');
      assert.doesNotMatch(await page.textContent('.detail .facts'), /linked from elsewhere|Pack .* available/);
      await page.click('.list [data-id="Deep-C"]');
      await page.waitForFunction(() => document.querySelector('#modelsDetail h2')?.textContent === 'Deep-C');
      assert.doesNotMatch(await page.textContent('.list [data-id="Deep-C"]'), /runs partly from SSD/);
      await context.close();
    }

    {
      // Leaving a dirty form asks first; Rescan posts once.
      const server = makeServer();
      const { page, context, errors } = await open(browser, server, '/models#Qwen-A');
      await page.waitForSelector(`${row('ctx_size')} input`);
      await page.fill(`${row('ctx_size')} input`, '4096');
      await page.click('.list [data-id="GLM-B"]');
      assert.equal(await page.evaluate(() => document.getElementById('confirmDialog').open), true);
      assert.match(await page.textContent('#confirmDialog'), /unsaved/);
      await page.click('#confirmDialog >> text=Keep editing');
      assert.equal(await page.inputValue(`${row('ctx_size')} input`), '4096');
      await page.click('.list [data-id="GLM-B"]');
      await page.click('#confirmDialog button:text-is("Discard")');
      await page.waitForFunction(() => location.hash === '#GLM-B');
      await page.click('text=Rescan');
      await page.waitForFunction(() => /2 new models/.test(document.getElementById('toastHost').textContent));
      assert.equal(server.count('/v1/models/rescan'), 1);
      assert.deepEqual(errors, []);
      await context.close();
    }

    {
      // The model's own cache: bytes in the button, and a dialog that names them; a loaded row shows its memory.
      const server = makeServer();
      const { page, context, errors } = await open(browser, server, '/models#Qwen-A');
      await page.waitForSelector('.detail >> text=Clear this model');
      assert.match(await page.textContent('.detail'), /41\.2 GiB/);
      assert.match(await page.textContent('.list [data-id="Qwen-A"]'), /40\.0 GiB in memory/);
      await page.click('.detail >> text=Clear this model');
      assert.match(await page.textContent('#confirmDialog'), /41\.2 GiB/);
      await page.click('#confirmDialog >> text=Cancel');
      assert.equal(server.count('/v1/prefix-cache/clear'), 0);
      await page.click('.detail >> text=Clear this model');
      await page.click('#confirmDialog button.danger-fill');
      await page.waitForFunction(() => /Freed/.test(document.getElementById('toastHost').textContent));
      assert.deepEqual(server.calls.find((c) => c.path === '/v1/prefix-cache/clear').body, { scope: 'model', model: 'Qwen-A' });
      assert.deepEqual(errors, []);
      await context.close();
    }

    {
      // Settings: bulk buttons carry their bytes, confirm before deleting, and say what stays.
      const server = makeServer();
      const { page, context, errors } = await open(browser, server, '/settings');
      await page.waitForSelector('#clearStale');
      assert.match(await page.textContent('#clearStale'), /22\.1 GiB/);
      assert.match(await page.textContent('#clearAll'), /63\.3 GiB/, 'all = everything not held by another server');
      await page.click('#clearStale');
      assert.equal(await page.evaluate(() => document.getElementById('confirmDialog').open), true);
      assert.match(await page.textContent('#confirmDialog'), /22\.1 GiB/);
      assert.equal(await page.evaluate(() => document.activeElement.textContent), 'Cancel', 'focus starts on the safe choice');
      await page.keyboard.press('Escape');
      assert.equal(server.count('/v1/prefix-cache/clear'), 0, 'Escape cancels');
      await page.click('#clearStale');
      await page.click('#confirmDialog button.danger-fill');
      await page.waitForFunction(() => /Freed 22\.1 GiB/.test(document.getElementById('toastHost').textContent));
      assert.deepEqual(server.calls.find((c) => c.path === '/v1/prefix-cache/clear').body, { scope: 'stale' });
      assert.equal(server.count('/v1/prefix-cache/clear'), 1);
      await settle(page, 12_000);
      assert.equal(await page.isDisabled('#clearStale'), true, 'nothing stale is left to clear');
      assert.deepEqual(errors, []);
      await context.close();
    }

    {
      // Assigned space: GiB, 0 turns the cache off only after a confirmation; a locked models folder offers no input.
      const server = makeServer({ dirLocked: true });
      const { page, context, errors } = await open(browser, server, '/settings');
      await page.waitForSelector('#cacheGb');
      assert.equal(await page.inputValue('#cacheGb'), '100');
      assert.match(await page.textContent('#cacheCard'), /GiB/);
      await page.fill('#cacheGb', '0');
      await page.click('#cacheSave');
      assert.match(await page.textContent('#confirmDialog'), /off/i);
      await page.click('#confirmDialog >> text=Cancel');
      assert.equal(server.count('/v1/settings', 'POST'), 0);
      assert.equal(await page.locator('#dirPath').count(), 0, 'a --model-dir launch flag leaves no input');
      assert.match(await page.textContent('#dirCard'), /--model-dir/);
      assert.match(await page.textContent('#dirCard'), /\/models/);
      assert.deepEqual(errors, []);
      await context.close();
    }

    {
      // The 390 px phone: nothing scrolls sideways on the Models and Settings tabs, and the detail replaces the list.
      const server = makeServer();
      const phone = { width: 390, height: 844 };
      for (const view of ['/models', '/settings', '/models#Qwen-A']) {
        const { page, context } = await open(browser, server, view, { viewport: phone, mode: 'advanced' });
        await page.waitForSelector(view === '/models' ? '.list [data-id]' : view === '/settings' ? '#cacheCard' : '.detail .set');
        await page.waitForTimeout(300);
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth), 390, `${view} fits 390 px`);
        await context.close();
      }
      const { page, context } = await open(browser, server, '/models', { viewport: phone });
      await page.waitForSelector('.list [data-id="Qwen-A"]');
      await page.click('.list [data-id="Qwen-A"]');
      assert.equal(await page.isVisible('.list'), false, 'the detail takes the whole screen');
      assert.equal(await page.isVisible('.detail'), true);
      await page.click('[aria-label="Back to models"]');
      assert.equal(await page.isVisible('.list'), true);
      assert.equal(await page.evaluate(() => location.hash), '');
      await context.close();
    }
    {
      // Per-model limits: a value or combination the model cannot run is refused on the row, before and after the save.
      const server = makeServer({ limits: true });
      server.state.models[0].settings.kv_quant = keyState();
      const { page, context, errors } = await open(browser, server, '/models#Qwen-A', { mode: 'advanced' });
      await page.waitForSelector(`${row('ctx_size')} input`);
      await page.fill(`${row('ctx_size')} input`, '200000');
      assert.match(await page.textContent(`${row('ctx_size')} .bad`), /context window is 131072/);
      assert.equal(await page.isDisabled('.savebar button:text-is("Save")'), true, 'a value past the model maximum cannot be saved');
      await page.fill(`${row('ctx_size')} input`, '65536');
      assert.equal(await page.isVisible('.savebar'), false);
      const four = `${row('kv_quant')} option[value="4"]`;
      assert.equal(await page.isDisabled(four), true);
      assert.match(await page.textContent(four), /not for this model/);
      assert.match(await page.textContent(row('kv_quant')), /need a different cache layout/, 'the reason is on the row, not only in a tooltip');
      assert.equal(await page.isDisabled(`${row('mtp_acceptance')} select`), false, 'MTP at its default leaves acceptance usable');
      await page.click(`${row('mtp')} button:text-is("Off")`);
      assert.equal(await page.isDisabled(`${row('mtp_acceptance')} select`), true, 'turning MTP off disables acceptance at once');
      assert.match(await page.textContent(row('mtp_acceptance')), /Used only while MTP is on/);
      assert.equal(await page.isDisabled(`${row('mtp_greedy_tail')} button:text-is("On")`), true);
      await page.click(`${row('mtp')} button:text-is("On")`);
      assert.equal(await page.isDisabled(`${row('mtp_acceptance')} select`), false);
      await page.click(`${row('mtp')} button:text-is("Off")`);
      await page.click('.savebar button:text-is("Save")');
      await settle(page, 1000);
      assert.deepEqual(server.calls.find((c) => c.path === '/v1/settings' && c.method === 'POST').body.set, { mtp: false }, 'only the usable change is sent');
      await page.fill(`${row('think_penalty')} input`, '2');
      await page.click('.savebar button:text-is("Save")');
      await settle(page, 1000);
      assert.match(await page.textContent(`${row('think_penalty')} .bad`), /not supported by this model/, 'a refusal naming a setting shows on its row');
      await page.click('.list [data-id="Deep-C"]');
      await page.click('#confirmDialog button.danger-fill');
      await page.waitForFunction(() => document.querySelector('.detail h2')?.textContent === 'Deep-C');
      assert.equal(await page.isDisabled(`${row('kv_quant')} select`), true, 'Advanced lists a key the model lacks, greyed');
      assert.match(await page.textContent(row('kv_quant')), /no KV cache to quantize/);
      assert.equal(await page.locator(row('mtp_acceptance')).count(), 0, 'a key without a limits entry is not listed');
      assert.equal(await page.isDisabled(`${row('mtp')} button:text-is("On")`), false);
      await page.fill(`${row('ssd_budget_gb')} input`, '8');
      assert.equal(await page.isDisabled(`${row('mtp')} button:text-is("On")`), true, 'an SSD budget switches MTP off at once');
      assert.match(await page.textContent(row('mtp')), /cannot run while they stream/);
      await page.fill(`${row('ssd_budget_gb')} input`, '');
      assert.equal(await page.isDisabled(`${row('mtp')} button:text-is("On")`), false);
      assert.deepEqual(errors, []);
      await context.close();
    }
    {
      // A newer server says which keys apply, what the loaded model started with, and clears one namespace at a time.
      const server = makeServer();
      Object.assign(server.state.server.prefix_cache_disk_gb, { ask_bytes: 30 * GiB });
      Object.assign(server.state.models[0], { keys: ['ctx_size', 'kv_quant', 'mtp', 'vision'], loaded_with: { ctx_size: 65536, kv_quant: null } });
      server.state.cache.namespaces.push({ id: 'dddddddd44444444', bytes: Math.round(3 * GiB), entries: 2, newest_mtime: Date.now() / 1000 - 5 * 86400, model: null, stale: true, held_by_other_process: false, tier: null });
      const { page, context, errors } = await open(browser, server, '/models#Qwen-A', { mode: 'advanced' });
      await page.waitForSelector(`${row('ctx_size')} input`);
      assert.deepEqual(await page.$$eval('.detail .set', (rows) => rows.map((r) => r.dataset.key)), ['ctx_size', 'kv_quant', 'mtp', 'vision'], 'the server decides which keys the model has');
      assert.equal(await page.isVisible('.callout'), false, 'saved equals what the model loaded with: nothing to reload');
      await page.fill(`${row('ctx_size')} input`, '4096');
      await page.click('.savebar button:text-is("Save")');
      await settle(page, 1000);
      assert.equal(await page.isVisible('.callout'), true, 'a saved value the loaded model lacks asks for a reload');
      await page.fill(`${row('ctx_size')} input`, '65536');
      await page.click('.savebar button:text-is("Save")');
      await settle(page, 1000);
      assert.equal(await page.isVisible('.callout'), false, 'saving the loaded value back clears the prompt even after an earlier save');
      await page.click('#navSettings');
      await page.waitForSelector('#cacheRows tr');
      assert.equal(await page.getAttribute('#cacheGb', 'placeholder'), 'Default (30)', 'the placeholder is what the next load starts from');
      const clears = page.locator('#cacheRows button:text-is("Clear…")');
      assert.equal(await clears.count(), 3, 'every stale row can be cleared on its own, and the loaded one');
      await page.click('#cacheRows tr:has-text("22.1") button');
      assert.match(await page.textContent('#confirmDialog'), /22\.1 GiB/);
      await page.click('#confirmDialog button.danger-fill');
      await page.waitForFunction(() => /Freed 22\.1 GiB/.test(document.getElementById('toastHost').textContent));
      assert.deepEqual(server.calls.find((c) => c.path === '/v1/prefix-cache/clear').body, { scope: 'namespace', id: 'bbbbbbbb22222222' });
      assert.deepEqual(errors, []);
      await context.close();
    }
    const failures = [];
    const phone = { width: 390, height: 844 };
    // One bug, one scenario: a failure is recorded and the next scenario still runs.
    const scenario = async (name, cfg, fn) => {
      const server = makeServer(cfg.serverOpts);
      cfg.setup?.(server);
      let context;
      try {
        const o = await open(browser, server, cfg.path, cfg.open);
        context = o.context;
        await fn({ page: o.page, server, errors: o.errors });
        assert.deepEqual(o.errors, [], 'page errors');
      } catch (error) { failures.push(`${name}: ${String(error.message).split('\n').slice(0, 3).join(' ')}`); } finally { await context?.close(); }
    };
    const dialogOpen = (page) => page.evaluate(() => document.getElementById('confirmDialog').open);
    const plainModel = (id) => ({ id, state: 'unloaded', bytes_on_disk: 10 * GiB, context_length: 131072, input_modalities: ['text'], keys: BASE_KEYS, settings: allKeys() });

    await scenario('hash change with unsaved edits: Keep editing holds the model', { path: '/models#Qwen-A' }, async ({ page }) => {
      await page.waitForSelector(`${row('ctx_size')} input`);
      await page.fill(`${row('ctx_size')} input`, '4096');
      await page.evaluate(() => { location.hash = '#GLM-B'; });
      await page.waitForFunction(() => document.getElementById('confirmDialog').open);
      assert.equal(await page.textContent('#modelsDetail h2'), 'Qwen-A', 'the form stays while the prompt is open');
      const length = await page.evaluate(() => history.length);
      await page.click('#confirmDialog >> text=Keep editing');
      await settle(page);
      assert.equal(await page.textContent('#modelsDetail h2'), 'Qwen-A');
      assert.equal(await page.inputValue(`${row('ctx_size')} input`), '4096');
      assert.equal(await page.evaluate(() => location.hash), '#Qwen-A');
      assert.equal(await page.evaluate(() => history.length), length, 'no extra history entry');
    });
    await scenario('Back with unsaved edits: Keep editing, then Discard', { path: '/models#GLM-B', open: { clock: false } }, async ({ page }) => {
      await page.waitForSelector('.list [data-id="Qwen-A"]');
      await page.click('.list [data-id="Qwen-A"]');
      await page.waitForFunction(() => document.querySelector('#modelsDetail h2')?.textContent === 'Qwen-A');
      await page.fill(`${row('ctx_size')} input`, '4096');
      await page.goBack();
      await page.waitForFunction(() => document.getElementById('confirmDialog').open);
      await page.click('#confirmDialog >> text=Keep editing');
      await page.waitForTimeout(200);
      assert.equal(await page.textContent('#modelsDetail h2'), 'Qwen-A');
      assert.equal(await page.inputValue(`${row('ctx_size')} input`), '4096');
      assert.equal(await page.evaluate(() => location.hash), '#Qwen-A');
      await page.click('.list [data-id="GLM-B"]');
      await page.click('#confirmDialog button:text-is("Discard")');
      await page.waitForFunction(() => document.querySelector('#modelsDetail h2')?.textContent === 'GLM-B');
      assert.equal(await page.evaluate(() => location.hash), '#GLM-B');
    });
    await scenario('a vanished model leaves no live detail', { path: '/models#GLM-B' }, async ({ page, server }) => {
      await page.waitForSelector(`${row('ctx_size')} input`);
      server.state.models = server.state.models.filter((m) => m.id !== 'GLM-B');
      await settle(page, 16_000);
      await settle(page, 16_000);
      assert.equal(await page.locator('#modelsDetail .set').count(), 0, 'no editable form for a model that is gone');
      assert.equal(await page.locator('#modelsDetail .d-actions').count(), 0, 'and no Load button');
      assert.match(await page.textContent('#modelsDetail'), /not (found|in the list)/i);
    });
    await scenario('an unknown model on a phone is not a blank screen', { path: '/models#Nope', open: { viewport: phone } }, async ({ page }) => {
      await page.waitForSelector('#modelsMd');
      await settle(page);
      assert.match(await page.innerText('#modelsDetail'), /not (found|in the list)/i);
      assert.equal(await page.isVisible('#modelsDetail'), true);
      await page.click('#modelsDetail >> text=Back to Models');
      assert.equal(await page.evaluate(() => location.hash), '');
      assert.equal(await page.isVisible('#modelsList'), true);
    });
    await scenario('alerts never print null', { path: '/models#Qwen-A' }, async ({ page, server }) => {
      await page.waitForSelector(`${row('ctx_size')} input`);
      server.state.failSettings = 1;
      await settle(page, 16_000);
      assert.doesNotMatch(await page.innerText('#modelsAlert'), /null/);
      assert.equal(await page.locator('#modelsAlert button:text-is("Retry")').count(), 1);
      assert.equal(await page.evaluate(() => document.getElementById('modelsMd').inert), true, 'nothing can be edited while the page cannot reach the server');
      await page.click('#modelsAlert button:text-is("Retry")');
      await page.waitForFunction(() => !document.getElementById('modelsMd').inert);
      server.state.hooks['/v1/settings'] = async (body, method) => (method === 'POST' ? { status: 500, contentType: 'text/html', body: '<html>boom</html>' } : null);
      await page.fill(`${row('ctx_size')} input`, '4096');
      await page.click('.savebar button:text-is("Save")');
      await settle(page, 500);
      const text = await page.innerText('#modelsDetail .alert.err:not([hidden])');
      assert.match(text, /Couldn't save/);
      assert.doesNotMatch(text, /null/);
    });
    await scenario('a Load failed message goes when the model is loaded elsewhere', { path: '/models#GLM-B', setup: (s) => { s.state.loadFails = true; } }, async ({ page, server }) => {
      await page.waitForSelector('.d-actions button');
      await page.click('.d-actions >> text=Load');
      await page.waitForSelector('.detail .alert.err:not([hidden])');
      server.state.loadFails = false;
      Object.assign(server.state.models.find((m) => m.id === 'GLM-B'), { state: 'ready', bytes_resident: 61 * GiB });
      await settle(page, 6000);
      assert.equal(await page.isVisible('.detail .alert.err'), false, 'a loaded model shows no failure');
      assert.doesNotMatch(await page.textContent('.list [data-id="GLM-B"]'), /failed/i);
    });
    await scenario('a flag-pinned assigned space shows the flag, not the file', { path: '/settings', setup: (s) => {
      Object.assign(s.state.server.prefix_cache_disk_gb, { file: 100, flag: '512MB', source: 'flag', locked: true, effective_bytes: 512 * 2 ** 20, ask_bytes: 512 * 2 ** 20 });
      Object.assign(s.state.server.models_dir, { flag: '/models', source: 'flag', locked: true });
    } }, async ({ page }) => {
      await page.waitForSelector('#cacheGb');
      assert.equal(await page.inputValue('#cacheGb'), '512MB');
      assert.equal(await page.isDisabled('#cacheGb'), true);
      assert.match(await page.textContent('#cacheUsage'), /of 512 MiB/);
      assert.match(await page.textContent('#cacheCard'), /--prefix-cache-disk 512MB/);
      assert.match(await page.textContent('#cacheCard'), /Set at launch/);
      assert.doesNotMatch(await page.textContent('.set[data-key="prefix_cache_disk_gb"] .badges'), /Now|Next/, 'a pinned size has no timing badge');
      assert.doesNotMatch(await page.textContent('#dirCard .badges'), /Now|Next/);
    });
    await scenario('a flag that pins MTP acceptance prints its own spelling', { path: '/models#Qwen-A', open: { mode: 'advanced' }, setup: (s) => {
      s.state.models[0].settings.mtp_acceptance = keyState({ flag: '--mtp-tokenv3 0.5', source: 'flag', locked: true });
    } }, async ({ page }) => {
      await page.waitForSelector(row('mtp_acceptance'));
      const text = await page.textContent(row('mtp_acceptance'));
      assert.match(text, /--mtp-tokenv3 0\.5/);
      assert.doesNotMatch(text, /--mtp-typical|\bset\b/);
      assert.equal(await page.$eval(`${row('mtp_acceptance')} select`, (n) => n.selectedOptions[0].textContent), 'TokenV3 (faster)');
    });
    await scenario('a value-only lock on MTP acceptance shows the choice it names', { path: '/models#Qwen-A', open: { mode: 'advanced' }, setup: (s) => {
      s.state.models[0].settings.mtp_acceptance = keyState({ flag: 'typical', source: 'flag', locked: true });
    } }, async ({ page }) => {
      await page.waitForSelector(row('mtp_acceptance'));
      assert.equal(await page.$eval(`${row('mtp_acceptance')} select`, (n) => n.selectedOptions[0].textContent), 'Typical (faster)');
      assert.match(await page.textContent(row('mtp_acceptance')), /Set by a launch flag\./);
    });
    await scenario('a flag the page cannot read shows Set at launch, never "set"', { path: '/models#Qwen-A', open: { mode: 'advanced' }, setup: (s) => {
      s.state.models[0].settings.mtp_acceptance = keyState({ flag: 'set', source: 'flag', locked: true });
      s.state.models[0].settings.mtp_greedy_tail = keyState({ flag: '--mtp-greedy-tail', source: 'flag', locked: true });
    } }, async ({ page }) => {
      await page.waitForSelector(row('mtp_acceptance'));
      assert.equal(await page.$eval(`${row('mtp_acceptance')} select`, (n) => n.selectedOptions[0].textContent), 'Set at launch');
      assert.doesNotMatch(await page.textContent(row('mtp_acceptance')), /--mtp-typical/);
      assert.match(await page.textContent(row('mtp_greedy_tail')), /Set by --mtp-greedy-tail at launch/);
    });
    await scenario('saving one server setting keeps the other one\'s draft', { path: '/settings' }, async ({ page }) => {
      await page.waitForSelector('#dirPath');
      await page.fill('#dirPath', '/tmp/typed');
      await page.fill('#cacheGb', '7');
      await page.click('#cacheSave');
      await page.waitForSelector('.toast');
      await settle(page, 1000);
      assert.equal(await page.inputValue('#dirPath'), '/tmp/typed', 'the folder draft survives a cache save');
      await page.fill('#cacheGb', '9');
      await page.click('#dirSave');
      await settle(page, 1000);
      assert.equal(await page.inputValue('#cacheGb'), '9', 'the size draft survives a folder save');
    });
    await scenario('an auto-selected model stays under an edit', { path: '/models', setup: (s) => { s.state.models[0].state = 'unloaded'; } }, async ({ page }) => {
      await page.waitForSelector(`${row('ctx_size')} input`);
      assert.equal(await page.textContent('#modelsDetail h2'), 'Broke-D');
      await page.fill(`${row('ctx_size')} input`, '4096');
      await page.click('.list [data-id="GLM-B"] button');
      await settle(page, 3000);
      assert.equal(await page.textContent('#modelsDetail h2'), 'Broke-D', 'a load elsewhere does not move the selection');
      assert.equal(await page.inputValue(`${row('ctx_size')} input`), '4096');
    });
    await scenario('a tab switch with unsaved edits asks, and the model is kept on return', { path: '/models#GLM-B' }, async ({ page }) => {
      await page.waitForSelector(`${row('ctx_size')} input`);
      await page.fill(`${row('ctx_size')} input`, '4096');
      await page.click('#navSettings');
      assert.equal(await dialogOpen(page), true, 'leaving a dirty form asks');
      await page.click('#confirmDialog >> text=Keep editing');
      await settle(page);
      assert.equal(await page.isVisible('#modelsPanel'), true);
      assert.equal(await page.inputValue(`${row('ctx_size')} input`), '4096');
      await page.click('#navSettings');
      await page.click('#confirmDialog button:text-is("Discard")');
      await page.waitForSelector('#settingsPanel:not([hidden])');
      await page.click('#navModels');
      await page.waitForSelector('#modelsDetail h2');
      assert.equal(await page.textContent('#modelsDetail h2'), 'GLM-B', 'back on the model that was open');
      assert.equal(await page.evaluate(() => location.hash), '#GLM-B');
    });
    await scenario('Cmd+4 with unsaved edits asks like a click on Settings', { path: '/models#GLM-B' }, async ({ page }) => {
      await page.waitForSelector(`${row('ctx_size')} input`);
      await page.fill(`${row('ctx_size')} input`, '4096');
      await page.keyboard.press('Meta+4');
      assert.equal(await dialogOpen(page), true, 'the shortcut asks before leaving a dirty form');
      await page.click('#confirmDialog >> text=Keep editing');
      await settle(page);
      assert.equal(await page.isVisible('#modelsPanel'), true);
      assert.equal(await page.inputValue(`${row('ctx_size')} input`), '4096');
    });
    await scenario('Available with unsaved edits asks like a tab switch', { path: '/models#GLM-B' }, async ({ page }) => {
      await page.waitForSelector(`${row('ctx_size')} input`);
      await page.fill(`${row('ctx_size')} input`, '4096');
      await page.click('#modelsTabAvailable');
      assert.equal(await dialogOpen(page), true, 'the switch asks before leaving a dirty form');
      await page.click('#confirmDialog >> text=Keep editing');
      await settle(page);
      assert.equal(await page.evaluate(() => location.pathname), '/models');
      assert.equal(await page.inputValue(`${row('ctx_size')} input`), '4096');
    });
    await scenario('a clean tab switch returns to the same model', { path: '/models' }, async ({ page }) => {
      await page.waitForSelector('.list [data-id="GLM-B"]');
      await page.click('.list [data-id="GLM-B"]');
      await page.click('#navSettings');
      await page.waitForSelector('#settingsPanel:not([hidden])');
      await page.click('#navModels');
      await page.waitForSelector('#modelsDetail h2');
      assert.equal(await page.textContent('#modelsDetail h2'), 'GLM-B');
    });
    await scenario('a filter on a hidden box does not strand the list', { path: '/models', setup: (s) => {
      s.state.models = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'Zeta', 'Zulu'].map(plainModel);
    } }, async ({ page, server }) => {
      await page.waitForSelector('.list [data-id="A1"]');
      await page.fill('#modelsFilter', 'Z');
      assert.equal(await page.locator('#modelsList .row').count(), 2);
      server.state.models = server.state.models.filter((m) => !m.id.startsWith('Z') && m.id < 'A4');
      await settle(page, 16_000);
      await settle(page, 16_000);
      assert.equal(await page.locator('#modelsList .row').count(), 3, 'the three left show');
    });
    await scenario('an invalid value in a closed fold opens it and says so', { path: '/models#Qwen-A', open: { mode: 'advanced' } }, async ({ page }) => {
      await page.waitForSelector(`${row('think_penalty')} input`);
      await page.fill(`${row('think_penalty')} input`, 'abc');
      await page.click('#modeToggle [data-mode="compact"]');
      await settle(page);
      assert.equal(await page.evaluate(() => document.querySelector('details.more-set').open), true, 'the fold holding the error is open');
      assert.equal(await page.isVisible(`${row('think_penalty')} .bad`), true);
      assert.match(await page.textContent('details.more-set summary'), /1 invalid/);
    });
    await scenario('a saved value on a disabled key can be removed', { serverOpts: { limits: true }, path: '/models#Deep-C', open: { mode: 'advanced' }, setup: (s) => {
      s.state.models[2].settings.preserve_thinking = keyState({ file: true, source: 'file' });
    } }, async ({ page, server }) => {
      const remove = `${row('preserve_thinking')} button:text-is("Remove saved value")`;
      await page.waitForSelector(remove);
      await page.click(remove);
      await page.waitForFunction(() => /Removed|Saved/.test(document.getElementById('toastHost').textContent));
      assert.deepEqual(server.calls.find((c) => c.method === 'POST' && c.path === '/v1/settings').body, { model: 'Deep-C', set: { preserve_thinking: null } });
      await settle(page, 1000);
      assert.equal(await page.isVisible(remove), false, 'nothing is left to remove');
    });
    await scenario('the load-error alert goes soon after the server is back', { path: '/models#Qwen-A' }, async ({ page, server }) => {
      await page.waitForSelector(`${row('ctx_size')} input`);
      await settle(page, 1000);
      server.state.down = true;
      await settle(page, 6000);
      await page.evaluate(() => window.sushiAdmin.refresh());
      await settle(page);
      assert.equal(await page.isVisible('#modelsAlert'), true);
      server.state.down = false;
      await settle(page, 5500);
      assert.equal(await page.isVisible('#modelsAlert'), false, 'the alert clears once the server answers');
    });
    await scenario('a rebuilt list keeps focus on the same control; rows take arrows and Space', { path: '/models#Qwen-A' }, async ({ page, server }) => {
      await page.waitForSelector('.list [data-id="Deep-C"] button');
      const nested = await page.evaluate(() => {
        const control = 'button, a[href], input, select, textarea, summary, [tabindex], [role=button], [role=option], [role=link]';
        return [...document.querySelectorAll(`#modelsList :is(${control})`)].filter((n) => n.parentElement.closest(`#modelsList :is(${control})`)).length;
      });
      assert.equal(nested, 0, 'no control sits inside another in the Models list');
      assert.equal(await page.evaluate(() => document.querySelectorAll('#modelsList [role=option], #modelsList [role=listbox]').length), 0, 'a list with buttons is not a listbox');
      assert.equal(await page.evaluate(() => document.querySelector('#modelsList .row.selected .main').getAttribute('aria-current')), 'true', 'the open model is marked current');
      await page.focus('.list [data-id="Deep-C"] button');
      server.state.models[0].bytes_resident = 12 * GiB;
      await settle(page, 6000);
      assert.equal(await page.evaluate(() => document.activeElement.matches('.row[data-id="Deep-C"] button.btn')), true, 'the Load button keeps focus');
      await page.focus('.list [data-id="Broke-D"] .main');
      await page.keyboard.press('ArrowDown');
      assert.equal(await page.evaluate(() => document.activeElement.closest('.row').dataset.id), 'Deep-C');
      await page.keyboard.press('Space');
      assert.equal(await page.evaluate(() => location.hash), '#Deep-C');
      await page.keyboard.press('ArrowUp');
      await page.keyboard.press('Enter');
      assert.equal(await page.evaluate(() => location.hash), '#Broke-D', 'Enter opens a row');
      assert.equal(await page.evaluate(() => document.activeElement.matches('.row[data-id="Broke-D"] .main')), true, 'and the focus stays on it');
    });
    await scenario('a file value the loader drops does not ask for a reload forever', { serverOpts: { limits: true }, path: '/models#Qwen-A', setup: (s) => {
      s.state.models[0].settings.kv_quant = keyState({ file: '4', source: 'file' });
      s.state.models[0].loaded_with = { ctx_size: 65536, kv_quant: null };
    } }, async ({ page }) => {
      await page.waitForSelector(`${row('ctx_size')} input`);
      assert.equal(await page.isVisible('.callout'), false);
    });
    await scenario('a click in the dialog padding does not close it', { path: '/settings' }, async ({ page }) => {
      await page.waitForSelector('#clearAll:not([disabled])');
      await page.click('#clearAll');
      const b = await page.locator('#confirmDialog').boundingBox();
      await page.mouse.click(b.x + 6, b.y + 6);
      assert.equal(await dialogOpen(page), true, 'padding is part of the dialog');
      await page.mouse.click(5, 5);
      assert.equal(await dialogOpen(page), false, 'the backdrop still cancels');
    });
    await scenario('a cache row\'s Clear goes pending and cannot be repeated', { path: '/settings', setup: (s) => {
      const gate = new Promise((resolve) => { s.release = resolve; });
      s.state.hooks['/v1/prefix-cache/clear'] = async () => { await gate; return null; };
    } }, async ({ page, server }) => {
      const clear = () => page.locator('#cacheRows tr', { hasText: '22.1' }).locator('button');
      await page.waitForSelector('#cacheRows tr button');
      await clear().click();
      await page.click('#cdOk');
      await settle(page);
      assert.equal(await clear().isDisabled(), true);
      assert.match(await clear().textContent(), /Clearing/);
      await clear().evaluate((b) => b.click());
      assert.equal(await dialogOpen(page), false, 'no second confirmation');
      server.release();
      await settle(page, 1000);
      assert.equal(server.count('/v1/prefix-cache/clear'), 1);
    });
    await scenario('the zero-size toast quotes the server', { path: '/settings' }, async ({ page }) => {
      await page.waitForSelector('#cacheGb');
      await page.fill('#cacheGb', '0');
      await page.click('#cacheSave');
      await page.click('#cdOk');
      await page.waitForSelector('.toast');
      const toast = await page.textContent('#toastHost');
      assert.match(toast, /next model load/i);
      assert.doesNotMatch(toast, /now has 0/);
      await settle(page, 1000);
      assert.match(await page.textContent('.set[data-key="prefix_cache_disk_gb"] .badges'), /Next load/, 'a size of 0 applies at the next load');
    });
    await scenario('the model path shows when /v1/settings answers after /v1/models', { path: '/models#Qwen-A', open: { mode: 'advanced' }, setup: (s) => {
      s.state.settingsDelay.push(new Promise((resolve) => { s.releaseSettings = resolve; }));
    } }, async ({ page, server }) => {
      await page.waitForSelector('#modelsDetail h2');
      await page.waitForTimeout(300);
      server.releaseSettings();
      await page.waitForSelector(`${row('ctx_size')} input`);
      await page.waitForTimeout(300);
      assert.equal(await page.locator('.facts .mono').count(), 1);
    });
    await scenario('065536 over a saved 65536 is not a change', { path: '/models#Qwen-A' }, async ({ page }) => {
      await page.waitForSelector(`${row('ctx_size')} input`);
      await page.fill(`${row('ctx_size')} input`, '065536');
      await settle(page);
      assert.equal(await page.evaluate(() => document.querySelector('.set[data-key="ctx_size"]').classList.contains('dirty')), false);
      assert.doesNotMatch(await page.textContent(`${row('ctx_size')} .hint`), /was/);
      assert.equal(await page.isVisible('.savebar'), false);
    });
    await scenario('an empty models folder shows one Rescan and no empty detail card', { path: '/models', setup: (s) => { s.state.models = []; } }, async ({ page }) => {
      await page.waitForSelector('#modelsList .empty');
      assert.equal(await page.isVisible('#modelsDetail'), false);
      assert.equal(await page.locator('button:visible', { hasText: 'Rescan' }).count(), 1);
      assert.equal(await page.locator('#modelsList .empty >> text=Open Settings').count(), 1);
    });
    await scenario('In use lists the saved folder too', { path: '/settings', setup: (s) => { s.state.server.models_dir.file = '/extra/packs'; } }, async ({ page }) => {
      await page.waitForSelector('#dirKv');
      const text = await page.textContent('#dirKv');
      assert.match(text, /\/models/);
      assert.match(text, /\/extra\/packs/);
    });
    await scenario('Enter submits the size and the folder', { path: '/settings' }, async ({ page, server }) => {
      await page.waitForSelector('#dirPath');
      await page.fill('#cacheGb', '12');
      await page.press('#cacheGb', 'Enter');
      await page.waitForSelector('.toast');
      assert.deepEqual(server.calls.find((c) => c.method === 'POST' && c.path === '/v1/settings').body, { server: { prefix_cache_disk_gb: 12 } });
      await page.fill('#dirPath', '/x/y');
      await page.press('#dirPath', 'Enter');
      await settle(page, 500);
      assert.deepEqual(server.calls.filter((c) => c.method === 'POST' && c.path === '/v1/settings')[1].body, { server: { models_dir: '/x/y' } });
    });
    await scenario('a bulk Clear stays off until the table shows the new bytes', { path: '/settings', setup: (s) => {
      let hold = false;
      const clearGate = new Promise((resolve) => { s.releaseClear = resolve; }), getGate = new Promise((resolve) => { s.releaseGet = resolve; });
      s.state.hooks['/v1/prefix-cache/clear'] = async () => { hold = true; await clearGate; return null; };
      s.state.hooks['/v1/prefix-cache'] = async () => { if (hold) await getGate; return null; };
    } }, async ({ page, server }) => {
      await page.waitForSelector('#clearStale:not([disabled])');
      await page.click('#clearStale');
      await page.click('#cdOk');
      await settle(page);
      server.releaseClear();
      await page.waitForTimeout(400);
      assert.equal(await page.isDisabled('#clearStale'), true, 'still pending while the refreshed bytes are on their way');
      server.releaseGet();
      await settle(page, 500);
      assert.equal(await page.isDisabled('#clearStale'), true);
    });
    await scenario('an assigned-space box follows the file again once the typing matches it', { path: '/settings' }, async ({ page, server }) => {
      await page.waitForSelector('#cacheGb');
      await page.fill('#cacheGb', '5');
      await page.fill('#cacheGb', '100');
      Object.assign(server.state.server.prefix_cache_disk_gb, { file: 80, effective_bytes: 80 * GiB });
      await page.evaluate(() => window.sushiAdmin.refresh());
      await settle(page, 500);
      assert.equal(await page.inputValue('#cacheGb'), '80');
    });
    await scenario('the confirm dialog is described by its body', { path: '/settings' }, async ({ page }) => {
      await page.waitForSelector('#clearAll');
      assert.equal(await page.getAttribute('#confirmDialog', 'aria-describedby'), 'cdBody');
    });
    await scenario('Advanced lists the cache columns and each namespace\'s write rate', { path: '/settings', open: { mode: 'advanced' } }, async ({ page, server }) => {
      await page.waitForSelector('#cacheRows tr button');
      const heads = await page.$$eval('#cacheCard thead th', (ths) => ths.filter((t) => t.offsetParent).map((t) => t.textContent));
      for (const want of ['Entries', 'ID', 'Share used', 'Written', 'Write rate']) assert.ok(heads.includes(want), `column ${want} (${heads})`);
      const cells = await page.$$eval('#cacheRows tr:first-child td', (tds) => tds.map((t) => t.textContent));
      assert.ok(cells.includes('37'), `entries ${cells}`);
      assert.ok(cells.includes('aaaaaaaa'), `id ${cells}`);
      assert.ok(cells.some((c) => /41\.0 GiB \/ 100 GiB/.test(c)), `tier use ${cells}`);
      assert.ok(cells.includes('5.00 GiB'), `written ${cells}`);
      server.state.cache.namespaces[0].tier.written_bytes += 100 * 2 ** 20;
      server.state.cache.total_bytes += 1;
      await settle(page, 10_000);
      const rate = await page.$eval('#cacheRows tr:first-child td:nth-last-child(2)', (td) => td.textContent);
      assert.match(rate, /^\d+(\.\d+)? MiB\/s$/);
      assert.ok(Number.parseFloat(rate) > 5 && Number.parseFloat(rate) < 12, rate);
      assert.match(await page.textContent('#cacheKv'), /Write rate/);
      await page.click('#modeToggle [data-mode="compact"]');
      assert.equal(await page.isVisible('#cacheCard th:text-is("Entries")'), false, 'Compact keeps the short table');
    });
    await scenario('an assigned space beyond the free disk is flagged', { path: '/settings' }, async ({ page }) => {
      await page.waitForSelector('#cacheGb');
      assert.equal(await page.locator('#cacheCard .warn:visible').count(), 0);
      await page.fill('#cacheGb', '5000');
      assert.match(await page.textContent('#cacheCard .warn'), /free/);
      assert.equal(await page.isDisabled('#cacheSave'), false, 'a warning does not block saving');
      await page.fill('#cacheGb', '100');
      assert.equal(await page.locator('#cacheCard .warn:visible').count(), 0);
    });
    await scenario('the automatic update checks show Default (On) and save On or Off', { path: '/settings' }, async ({ page, server }) => {
      await page.waitForSelector('.set[data-key="update_check"]');
      const row = (key) => page.locator(`.set[data-key="${key}"]`);
      assert.equal(await row('update_check').locator('button[data-value="null"]').textContent(), 'Default (On)');
      assert.equal(await row('update_check').locator('button[data-value="null"]').getAttribute('aria-pressed'), 'true');
      await row('pack_update_check').locator('button[data-value="false"]').click();
      await page.waitForFunction(() => document.querySelector('.set[data-key="pack_update_check"] button[data-value="false"]').getAttribute('aria-pressed') === 'true');
      assert.deepEqual(server.calls.find((c) => c.method === 'POST' && c.path === '/v1/settings').body, { server: { pack_update_check: false } });
      assert.equal(await row('update_check').locator('button[data-value="false"]').getAttribute('aria-pressed'), 'false', 'the other check is untouched');
    });
    await scenario('--no-update-check locks both automatic update checks', { path: '/settings', setup: (s) => {
      for (const k of ['update_check', 'pack_update_check']) Object.assign(s.state.server[k], { flag: '--no-update-check', source: 'flag', locked: true });
    } }, async ({ page }) => {
      await page.waitForSelector('.set[data-key="update_check"]');
      assert.equal(await page.locator('.set[data-key="update_check"] button[data-value="true"]').isDisabled(), true);
      assert.equal(await page.locator('.set[data-key="update_check"] button[data-value="false"]').getAttribute('aria-pressed'), 'true');
      assert.match(await page.textContent('#updatesCard'), /Set by --no-update-check at launch/);
      assert.match(await page.textContent('#updatesCard'), /Set at launch/);
    });
    await scenario('the memory-check switch defaults to Off, warns about running out of memory, and saves On', { path: '/settings', open: { mode: 'advanced' } }, async ({ page, server }) => {
      await page.waitForSelector('.set[data-key="skip_mem_preflight"]');
      const row = page.locator('.set[data-key="skip_mem_preflight"]');
      const label = row.locator('.lab > .tip-u');
      assert.equal(await label.textContent(), 'Skip the memory check before loading');
      assert.match(await label.getAttribute('data-tip'), /out of memory/i);
      assert.match(await label.getAttribute('data-tip'), /crash|stall/);
      assert.match(await label.getAttribute('data-term'), /skip_mem_preflight.*--skip-mem-preflight/);
      assert.match(await row.locator('.help').textContent(), /out of memory/i, 'the warning is visible, not only in the tip');
      assert.equal(await row.locator('button[data-value="null"]').textContent(), 'Default (Off)');
      await row.locator('button[data-value="true"]').click();
      await page.waitForFunction(() => document.querySelector('.set[data-key="skip_mem_preflight"] button[data-value="true"]').getAttribute('aria-pressed') === 'true');
      assert.deepEqual(server.calls.find((c) => c.method === 'POST' && c.path === '/v1/settings').body, { server: { skip_mem_preflight: true } });
      assert.match(await row.locator('.badges').textContent(), /Now/);
    });
    await scenario('--skip-mem-preflight locks the memory-check switch on', { path: '/settings', setup: (s) => {
      Object.assign(s.state.server.skip_mem_preflight, { flag: '--skip-mem-preflight', source: 'flag', locked: true });
    } }, async ({ page }) => {
      await page.waitForSelector('.set[data-key="skip_mem_preflight"]');
      const row = page.locator('.set[data-key="skip_mem_preflight"]');
      assert.equal(await row.locator('button[data-value="false"]').isDisabled(), true);
      assert.equal(await row.locator('button[data-value="true"]').getAttribute('aria-pressed'), 'true');
      assert.match(await row.textContent(), /Set by --skip-mem-preflight at launch/);
    });
    await scenario('the default port shows its ghost text, refuses a bad number and saves a good one', { path: '/settings' }, async ({ page, server }) => {
      await page.waitForSelector('#portNum');
      assert.equal(await page.getAttribute('#portNum', 'placeholder'), 'Default (12345)');
      assert.equal(await page.isDisabled('#portSave'), true);
      await page.fill('#portNum', '70000');
      assert.equal(await page.isDisabled('#portSave'), true, 'a port past 65535 cannot be saved');
      assert.match(await page.textContent('.set[data-key="port"]'), /1 to 65535/);
      await page.fill('#portNum', '8080');
      await page.click('#portSave');
      await page.waitForFunction(() => document.querySelector('.set[data-key="port"] .badges').textContent.includes('Saved'));
      assert.deepEqual(server.calls.find((c) => c.method === 'POST' && c.path === '/v1/settings').body, { server: { port: 8080 } });
      await page.fill('#portNum', '');
      assert.equal(await page.textContent('#portSave'), 'Use default', 'an empty box saves the default');
      await page.click('#portSave');
      await page.waitForFunction(() => document.querySelector('#portNum').value === '');
      assert.deepEqual(server.calls.filter((c) => c.method === 'POST' && c.path === '/v1/settings')[1].body, { server: { port: null } });
    });
    await scenario('--port locks the default port and names the flag', { path: '/settings', setup: (s) => {
      Object.assign(s.state.server.port, { flag: '9000', source: 'flag', locked: true });
    } }, async ({ page }) => {
      await page.waitForSelector('#portNum');
      assert.equal(await page.isDisabled('#portNum'), true);
      assert.equal(await page.inputValue('#portNum'), '9000');
      assert.match(await page.textContent('.set[data-key="port"]'), /Set by --port 9000 at launch/);
    });

    const badgesOf = (page, key) => page.textContent(`.set[data-key="${key}"] .badges`);
    await scenario('every setting says when it applies: Now, Next load or Next start; a default row carries no badge', { path: '/settings' }, async ({ page, server }) => {
      await page.waitForSelector('#portNum');
      assert.match(await badgesOf(page, 'port'), /Next start/);
      assert.match(await badgesOf(page, 'update_check'), /Now/);
      assert.match(await badgesOf(page, 'models_dir'), /Now/);
      assert.match(await badgesOf(page, 'prefix_cache_disk_gb'), /Saved.*Now/);
      assert.doesNotMatch(await badgesOf(page, 'port'), /Default/, 'the box already says Default');
      server.state.server.prefix_cache_disk_gb.file = 0;
      await page.evaluate(() => window.sushiAdmin.refresh());
      await settle(page);
      assert.match(await badgesOf(page, 'prefix_cache_disk_gb'), /Next load/);
    });
    // Every setting label and every badge explains itself, and the keyboard reaches each explanation.
    const tipsOf = (page) => page.$$eval('.set', (rows) => rows.filter((r) => r.offsetParent).map((r) => ({
      key: r.dataset.key,
      label: r.querySelector('.lab [data-tip]') ? { tip: r.querySelector('.lab [data-tip]').dataset.tip, tab: r.querySelector('.lab [data-tip]').tabIndex } : null,
      badges: [...r.querySelectorAll('.badges .badge')].map((b) => ({ text: b.textContent, tip: b.dataset.tip ?? '', tab: b.tabIndex })),
    })));
    for (const path of ['/models#Qwen-A', '/settings'])
      await scenario(`every setting and badge on ${path} has a keyboard tooltip`, { path, open: { mode: 'advanced' } }, async ({ page }) => {
        await page.waitForSelector('.set .lab');
        const rows = await tipsOf(page);
        assert.ok(rows.length >= 5, `rows found (${rows.length})`);
        for (const r of rows) {
          assert.ok(r.label?.tip.length > 20 && r.label.tab === 0, `${r.key}: its label has a focusable tooltip`);
          for (const b of r.badges) assert.ok(b.tip.length > 10 && b.tab === 0, `${r.key}: the ${b.text} badge has a focusable tooltip`);
        }
        assert.ok(rows.some((r) => r.badges.length), 'some badge was checked');
        await page.keyboard.press('Tab');
        await page.locator('.set .lab [data-tip]').first().focus();
        await page.waitForFunction(() => document.getElementById('tip').classList.contains('show'), null, { timeout: 3000 });
        assert.equal(await page.evaluate(() => document.getElementById('tip').firstChild.textContent), rows[0].label.tip, 'keyboard focus opens the tooltip');
      });
    // A plain-worded setting or badge keeps its engineering name in the tooltip: the model-settings.json key and the launch flag.
    const FLAGS = { ctx_size: '--ctx-size', kv_quant: '--kv-quant', mtp: '--mtp', mtp_acceptance: null, mtp_greedy_tail: '--mtp-greedy-tail', ssd_budget_gb: '--ssd-budget-gb', ssd_prefetch: '--ssd-prefetch',
      preserve_thinking: '--preserve-thinking', think_penalty: '--think-penalty', vision: '--vision', max_tokens: '--max-tokens', prefix_cache_disk_gb: '--prefix-cache-disk', models_dir: null, port: '--port',
      update_check: '--no-update-check', pack_update_check: '--no-update-check', skip_mem_preflight: '--skip-mem-preflight' };
    const WORDS = { ssd_budget_gb: /expert/, ssd_prefetch: /prefetch/ };
    // A label that is already the common term keeps only the key and flag in its technical name.
    const KEY_ONLY = { kv_quant: 'kv_quant (--kv-quant)', mtp: 'mtp (--mtp)', mtp_acceptance: 'mtp_acceptance', vision: 'vision (--vision)' };
    const BADGE_TERMS = { Saved: /model-settings\.json/, 'Set at launch': /launch flag/, Now: /live/, 'Next load': /model load/, 'Next start': /server start/ };
    const termsOf = (page) => page.$$eval('.set', (rows) => rows.filter((r) => r.offsetParent || r.closest('details')).map((r) => ({ key: r.dataset.key, term: r.querySelector('.lab [data-tip]')?.dataset.term ?? '',
      badges: [...r.querySelectorAll('.badges .badge')].map((b) => [b.textContent, b.dataset.term ?? '']) })));
    for (const path of ['/models#Qwen-A', '/settings'])
      await scenario(`every setting and badge on ${path} names its key, its launch flag and its engineering term`, { path, open: { mode: 'advanced' }, setup: (s) => {
        s.state.models[0].keys = KEYS.map((k) => k.name);
        s.state.models[0].settings.mtp = keyState({ file: true, source: 'file' });
        s.state.server.port = { file: 12400, flag: null, default: 12345, source: 'file', locked: false, applies: 'applies at the next start' };
      } }, async ({ page }) => {
        await page.waitForSelector('.set .lab');
        const rows = await termsOf(page);
        assert.ok(rows.length >= 3, `rows found (${rows.length})`);
        const seen = new Set();
        for (const r of rows) {
          assert.ok(r.term.includes(r.key), `${r.key}: its term names the key (${r.term})`);
          if (FLAGS[r.key]) assert.ok(r.term.includes(FLAGS[r.key]), `${r.key}: its term names ${FLAGS[r.key]} (${r.term})`);
          if (WORDS[r.key]) assert.match(r.term, WORDS[r.key], `${r.key}: its engineering word`);
          if (KEY_ONLY[r.key]) assert.equal(r.term, KEY_ONLY[r.key], `${r.key}: its term does not repeat the label`);
          for (const [text, term] of r.badges) { assert.match(term, BADGE_TERMS[text], `${r.key}: the ${text} badge`); seen.add(text); }
        }
        assert.ok(seen.has(path === '/settings' ? 'Next start' : 'Saved'), `a badge was checked (${[...seen]})`);
        if (path !== '/settings') {
          assert.ok(rows.length >= 11, `every model key was checked (${rows.map((r) => r.key)})`);
          assert.ok(seen.has('Set at launch'), 'a launch-flag badge was checked');
        }
      });
    // The label is the term users know from other LLM apps; the page never says guess or Conversation cache.
    for (const path of ['/models#Qwen-A', '/settings'])
      await scenario(`${path} uses the common terms`, { path, open: { mode: 'advanced' }, setup: (s) => {
        s.state.models[0].keys = KEYS.map((k) => k.name);
        s.state.cache.namespaces.push({ id: 'dddddddd44444444', bytes: Math.round(3 * GiB), entries: 2, newest_mtime: Date.now() / 1000 - 5 * 86400, model: null, stale: true, held_by_other_process: false, tier: null });
      } }, async ({ page }) => {
        await page.waitForSelector('.set .lab');
        if (path === '/settings') assert.equal(await page.textContent('#cacheCard .card-h'), 'Prompt cache (SSD)');
        else {
          const labels = await page.$$eval('.set', (rows) => Object.fromEntries(rows.map((r) => [r.dataset.key, r.querySelector('.lab [data-tip]')?.textContent])));
          assert.deepEqual([labels.kv_quant, labels.mtp, labels.mtp_acceptance, labels.vision], ['KV cache precision', 'Speculative decoding', 'Draft acceptance', 'Vision (image input)']);
          const options = await page.$eval(`${row('mtp_acceptance')} select`, (n) => [...n.options].map((o) => o.textContent).join('|'));
          assert.equal(options, 'Default|Exact (same output)|Typical (faster)|TokenV3 (faster)', options);
          assert.ok((await page.$$eval('.sect h3', (ns) => ns.map((n) => n.textContent))).includes('Prompt cache'), 'the model page names its cache section');
        }
        const words = await page.evaluate(() => [...document.querySelectorAll('body *:not(script):not(style)')].flatMap((n) => [...n.childNodes].filter((c) => c.nodeType === 3).map((c) => c.textContent)
          .concat(['tip', 'term'].map((k) => n.dataset[k]), n.getAttribute('aria-label'), n.title)).filter(Boolean).join('\n'));
        assert.doesNotMatch(words, /guess|conversation cache|conversation memory/i, 'no page says guess or Conversation cache');
      });
    await scenario('a model\'s facts keep their engineering names', { path: '/models#Qwen-A' }, async ({ page }) => {
      await page.waitForSelector('.facts [data-tip]');
      const facts = await page.$$eval('.facts [data-tip]', (ns) => ns.map((n) => [n.textContent, n.dataset.term ?? '']));
      const term = (re) => facts.find(([t]) => re.test(t))?.[1];
      assert.match(term(/on disk/), /bytes_on_disk/);
      assert.match(term(/in memory/), /bytes_resident/);
      assert.match(term(/max context/), /context_length/);
      assert.match(term(/sees images/), /vision/);
    });
    await scenario('a model that can stream from the SSD names expert streaming', { path: '/models#Deep-C' }, async ({ page }) => {
      await page.waitForSelector('.facts [data-tip]');
      const facts = await page.$$eval('.facts [data-tip]', (ns) => ns.map((n) => [n.textContent, n.dataset.term ?? '']));
      assert.match(facts.find(([t]) => /partly from SSD/.test(t))[1], /expert streaming.*ssd_budget_gb/);
      assert.match(facts.find(([t]) => /linked/.test(t))[1], /bytes_linked/);
    });
    await scenario('the cache card names what a plain word was before', { path: '/settings', open: { mode: 'advanced' }, setup: (s) => {
      s.state.cache.namespaces.push({ id: 'dddddddd44444444', bytes: Math.round(3 * GiB), entries: 2, newest_mtime: Date.now() / 1000 - 5 * 86400, model: null, stale: true, held_by_other_process: false, tier: null });
    } }, async ({ page }) => {
      await page.waitForSelector('#cacheRows tr');
      const heads = await page.$$eval('#cacheCard thead [data-tip]', (ns) => Object.fromEntries(ns.map((n) => [n.textContent, n.dataset.term ?? ''])));
      assert.match(heads.Entries, /prefix cache entries/);
      assert.match(heads.ID, /namespace/);
      assert.match(heads['Share used'], /tier/);
      assert.match(await page.getAttribute('#clearStale', 'data-term'), /stale/, 'Clear unused caches is the stale clear');
      assert.match(await page.getAttribute('#cacheRows td [data-term]', 'data-term'), /stale/, 'an Unused cache is stale');
    });
    await scenario('a model row is badged only when saved or pinned', { path: '/models#Qwen-A' }, async ({ page }) => {
      await page.waitForSelector(row('ctx_size'));
      assert.match(await badgesOf(page, 'ctx_size'), /Saved/);
      assert.match(await badgesOf(page, 'kv_quant'), /Set at launch/);
      assert.equal(await badgesOf(page, 'mtp'), '', 'a default row has no badge');
    });
    for (const [path, viewport] of [['/models#Qwen-A', { width: 1440, height: 900 }], ['/settings', { width: 1440, height: 900 }], ['/settings', phone]])
      await scenario(`Default (On) stays on one line on ${path} at ${viewport.width}px`, { path, open: { viewport }, setup: (s) => { s.state.models[0].defaults = { mtp: true, vision: false, preserve_thinking: true }; } }, async ({ page }) => {
        await page.waitForSelector('.tri button');
        const wrapped = await page.$$eval('.tri button', (bs) => bs.filter((b) => b.offsetParent && (b.scrollHeight > b.clientHeight || b.scrollWidth > b.clientWidth)).map((b) => b.textContent));
        assert.deepEqual(wrapped, [], 'no switch label wraps');
        assert.match(await page.textContent('.tri button'), /^Default \(/);
      });
    await scenario('a loaded model\'s cache size shows without a click', { path: '/models#Qwen-A', open: { clock: false } }, async ({ page, server }) => {
      await page.waitForSelector('.detail >> text=Clear this model', { timeout: 3000 });
      assert.match(await page.textContent('.detail'), /41\.2 GiB/);
      assert.ok(server.count('/v1/prefix-cache', 'GET') >= 1);
    });
    await scenario('About shows the version as soon as /props answers', { path: '/settings', setup: (s) => {
      s.state.hooks['/props'] = async () => { await new Promise((resolve) => { s.releaseProps = resolve; }); return null; };
    } }, async ({ page, server }) => {
      await page.waitForSelector('#cacheCard:not([hidden])');
      await page.waitForFunction(() => document.getElementById('aboutVersion').textContent === '-');
      server.releaseProps();
      await page.waitForFunction(() => document.getElementById('aboutVersion').textContent === 'v9.9.9', null, { timeout: 2000 });
    });
    await scenario('Load says what the server unloaded, and the button warns first', { path: '/models#GLM-B', setup: (s) => {
      s.state.hooks['/v1/load-model'] = async () => { s.state.models[0].state = 'unloaded'; return null; };
    } }, async ({ page }) => {
      await page.waitForSelector('.d-actions button');
      assert.match(await page.getAttribute('.d-actions >> text=Load', 'title'), /May unload Qwen-A/);
      assert.match(await page.getAttribute('.list [data-id="GLM-B"] button', 'title'), /May unload Qwen-A/);
      await page.click('.d-actions >> text=Load');
      await page.waitForFunction(() => /Loaded GLM-B · unloaded Qwen-A/.test(document.getElementById('toastHost').textContent));
    });
    await scenario('Load beside nothing else says only Loaded', { path: '/models#GLM-B', setup: (s) => { s.state.models[0].state = 'unloaded'; } }, async ({ page }) => {
      await page.waitForSelector('.d-actions button');
      assert.equal(await page.getAttribute('.d-actions >> text=Load', 'title'), null);
      await page.click('.d-actions >> text=Load');
      await page.waitForFunction(() => /Loaded GLM-B/.test(document.getElementById('toastHost').textContent));
      assert.doesNotMatch(await page.textContent('#toastHost'), /unloaded/);
    });
    await scenario('help text names no control the page lacks and no literal markup', { path: '/settings' }, async ({ page }) => {
      await page.waitForSelector('#portNum');
      const text = await page.textContent('#settingsPanel');
      assert.doesNotMatch(text, /Check for updates|sushi pull|`/);
      assert.match(await page.textContent('.set[data-key="update_check"] .help'), /Off stops automatic checks; a check you start yourself still runs\./);
      assert.match(await page.textContent('.set[data-key="pack_update_check"] .help'), /Off stops automatic checks, the Downloads list's included; Check again on Downloads still looks\./);
      assert.match(await page.textContent('.set[data-key="models_dir"] .help'), /where new downloads are saved/);
    });
    await scenario('a failed load after a reload still says why', { path: '/models#Broke-D', setup: (s) => { s.state.models[3].error = 'OutOfMemory'; } }, async ({ page }) => {
      await page.waitForSelector('.detail .alert.err:not([hidden])');
      assert.match(await page.textContent('.detail .alert.err'), /Load failed: OutOfMemory/);
      assert.doesNotMatch(await page.textContent('.detail .alert.err'), /last load failed/);
    });
    await scenario('an empty Models page points at Downloads', { path: '/models', setup: (s) => { s.state.models = []; } }, async ({ page }) => {
      await page.waitForSelector('#modelsList .empty');
      assert.match(await page.textContent('#modelsList .empty'), /Download/);
      await page.click('#modelsList .empty >> text=See available packs');
      await page.waitForFunction(() => location.pathname === '/downloads');
    });
    await scenario('one Reload at a time: the callout replaces the header button', { path: '/models#Qwen-A', setup: (s) => { s.state.models[0].loaded_with = { ctx_size: 1024 }; } }, async ({ page, server }) => {
      await page.waitForSelector('.callout:visible');
      const reloads = () => page.$$eval('#modelsDetail button', (bs) => bs.filter((b) => b.offsetParent && /^Reload/.test(b.textContent)).map((b) => b.textContent));
      assert.deepEqual(await reloads(), ['Reload now']);
      server.state.models[0].loaded_with = { ctx_size: 65536 };
      await settle(page, 16_000);
      assert.deepEqual(await reloads(), ['Reload'], 'with nothing to apply the header keeps Reload');
    });
    await scenario('a pack update opens Downloads at that pack, never sushi pull', { path: '/models#Deep-C', setup: (s) => {
      const pack = { repo: 'beamster/Deep-Sushi', name: 'deep', tag: '', size_bytes: 96 * GiB, tier_gb: 64, fits: true, streams_experts: true, recommended: false, installed: true, path: '/models/Deep-C', partial_bytes: null, update: { current: '1.0', latest: '2.0' }, launch: null };
      const other = { ...pack, repo: 'beamster/Other-Sushi', name: 'other', update: null, installed: false, path: null };
      s.state.hooks['/v1/downloads'] = async () => ({ status: 200, contentType: 'application/json', body: JSON.stringify({ ram_bytes: 128 * GiB, ram_gb: 128, update_check: true, catalog: 'ready', refreshing: false, dest: { path: '/models', source: 'default', locked: false, free_bytes: 900 * GiB, volume: 'Data' }, jobs: [], packs: [other, pack] }) });
    } }, async ({ page }) => {
      await page.waitForSelector('.callout:visible');
      const text = await page.textContent('#modelsDetail .callout:visible');
      assert.match(text, /newer version of this model \(2\.0\) is available/);
      assert.doesNotMatch(text, /sushi pull/);
      await page.click('#modelsDetail .callout:visible button');
      await page.waitForFunction(() => location.pathname === '/downloads');
      await page.waitForSelector('.dl-row[data-repo="beamster/Deep-Sushi"]');
      assert.equal(await page.evaluate(() => document.querySelector('.dl-row[data-repo="beamster/Deep-Sushi"]').classList.contains('pick')), true, 'the pack is marked');
      assert.equal(await page.evaluate(() => document.querySelector('.dl-row[data-repo="beamster/Other-Sushi"]').classList.contains('pick')), false);
    });
    await scenario('Settings Save appears only for a changed value; empty plus Save sends null; no Use default buttons', { path: '/settings', setup: (s) => {
      s.state.server.port.file = 8080;
      s.state.server.models_dir.file = '/extra/packs';
    } }, async ({ page, server }) => {
      await page.waitForSelector('#portNum');
      for (const id of ['cacheSave', 'dirSave', 'portSave']) assert.equal(await page.isVisible(`#${id}`), false, `${id} waits for a change`);
      for (const id of ['cacheDefault', 'dirDefault', 'portDefault']) assert.equal(await page.locator(`#${id}`).count(), 0, `${id} is gone`);
      await page.fill('#portNum', '');
      assert.equal(await page.textContent('#portSave'), 'Use default');
      await page.click('#portSave');
      await page.fill('#cacheGb', '');
      await page.click('#cacheSave');
      await page.fill('#dirPath', '');
      assert.equal(await page.textContent('#dirSave'), 'Use default');
      await page.click('#dirSave');
      while (server.count('/v1/settings', 'POST') < 3) await page.waitForTimeout(50);
      assert.deepEqual(server.calls.filter((c) => c.method === 'POST' && c.path === '/v1/settings').map((c) => c.body), [{ server: { port: null } }, { server: { prefix_cache_disk_gb: null } }, { server: { models_dir: null } }]);
      await page.fill('#cacheGb', '7');
      assert.equal(await page.textContent('#cacheSave'), 'Save');
      assert.equal(await page.isVisible('#cacheSave'), true);
    });
    await scenario('a refused server setting is shown on its row, once, with no toast', { path: '/settings', setup: (s) => {
      s.state.hooks['/v1/settings'] = (body, method) => method === 'POST' && body.server && 'port' in body.server ? { status: 400, contentType: 'application/json', body: JSON.stringify({ error: { message: 'port 80 needs root; pick 1024 or higher' } }) } : null;
    } }, async ({ page }) => {
      await page.waitForSelector('#portNum');
      await page.fill('#portNum', '80');
      await page.click('#portSave');
      await settle(page, 400);
      assert.match(await page.textContent('.set[data-key="port"] .bad'), /port 80 needs root/);
      assert.equal(await page.locator('#toastHost .toast').count(), 0, 'the row already says it');
      await page.fill('#portNum', '81');
      assert.equal(await page.textContent('.set[data-key="port"] .bad'), '', 'typing clears the refusal');
    });
    await scenario('a refused model setting names a label, not a key, and raises no toast', { serverOpts: { limits: true }, path: '/models#Qwen-A', open: { mode: 'advanced' } }, async ({ page }) => {
      await page.waitForSelector(`${row('think_penalty')} input`);
      await page.fill(`${row('think_penalty')} input`, '2');
      await page.click('.savebar button:text-is("Save")');
      await settle(page, 500);
      const bad = await page.textContent(`${row('think_penalty')} .bad`);
      assert.match(bad, /^Shorter thinking is not supported by this model/);
      assert.doesNotMatch(bad, /think_penalty/);
      assert.equal(await page.locator('#toastHost .toast').count(), 0);
    });
    await scenario('a bad number says "1 or more" and the save bar counts the invalid one', { path: '/models#Qwen-A', open: { mode: 'advanced' } }, async ({ page }) => {
      await page.waitForSelector(`${row('ctx_size')} input`);
      await page.fill(`${row('ctx_size')} input`, '4096');
      await page.fill(`${row('max_tokens')} input`, '12x');
      assert.match(await page.textContent(`${row('max_tokens')} .bad`), /1 or more/);
      assert.doesNotMatch(await page.textContent(`${row('max_tokens')} .bad`), /4294967295/);
      assert.match(await page.textContent('.savebar'), /2 changes · 1 invalid/);
      await page.fill(`${row('max_tokens')} input`, '');
      await page.fill(`${row('think_penalty')} input`, '1');
      assert.match(await page.textContent('.savebar'), /2 changes/);
      assert.doesNotMatch(await page.textContent('.savebar'), /invalid/);
    });
    await scenario('a failed load is announced once on the page that shows it', { path: '/models#GLM-B', setup: (s) => { s.state.loadFails = true; } }, async ({ page }) => {
      await page.waitForSelector('.d-actions button');
      await page.click('.d-actions >> text=Load');
      await page.waitForSelector('.detail .alert.err:not([hidden])');
      assert.equal(await page.locator('#toastHost .toast').count(), 0, 'no toast beside the alert');
      assert.doesNotMatch(await page.textContent('.list [data-id="GLM-B"] .meta'), /out of memory/, 'and the selected row only says it failed');
    });
    await scenario('an empty cache hides its Clear buttons and says Empty', { path: '/settings', setup: (s) => { s.state.cache.namespaces = []; s.state.cache.total_bytes = 0; } }, async ({ page }) => {
      await page.waitForSelector('#cacheUsage');
      assert.match(await page.textContent('#cacheUsage'), /^Empty · 100 GiB assigned$/);
      assert.equal(await page.isVisible('#clearStale'), false);
      assert.equal(await page.isVisible('#clearAll'), false);
      assert.doesNotMatch(await page.innerText('#cacheCard'), /0 KiB|Unused:/);
    });
    await scenario('a locked KV row shows no hint about unavailable choices', { serverOpts: { limits: true }, path: '/models#Qwen-A' }, async ({ page }) => {
      await page.waitForSelector(row('kv_quant'));
      assert.equal(await page.textContent(`${row('kv_quant')} .hint`), '');
    });
    await scenario('offline makes the dimmed areas inert, not just faded', { path: '/models#Qwen-A' }, async ({ page, server }) => {
      await page.waitForSelector(`${row('ctx_size')} input`);
      server.state.down = true;
      await settle(page, 6000);
      await settle(page, 6000);
      assert.equal(await page.evaluate(() => document.getElementById('modelsMd').inert), true);
      assert.equal(await page.evaluate(() => document.getElementById('modelsAlert').closest('[inert]') === null), true, 'Retry stays usable');
      server.state.down = false;
      await settle(page, 6000);
      await settle(page, 6000);
      assert.equal(await page.evaluate(() => document.getElementById('modelsMd').inert), false);
    });
    await scenario('offline on Settings makes every card inert', { path: '/settings' }, async ({ page, server }) => {
      await page.waitForSelector('#cacheCard:not([hidden])');
      server.state.down = true;
      await settle(page, 6000);
      await settle(page, 6000);
      assert.deepEqual(await page.$$eval('#settingsPanel section.card:not([hidden])', (cs) => cs.filter((c) => !c.inert).map((c) => c.id)), []);
    });

    if (failures.length) throw new Error(`${failures.length} admin scenario(s) failed:\n  ${failures.join('\n  ')}`);
    console.log('Web UI admin views (browser): passed');
  } finally {
    await browser.close();
  }
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
