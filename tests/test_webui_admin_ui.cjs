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
];
const keyState = (extra = {}) => ({ file: null, flag: null, source: 'default', locked: false, ...extra });
const BASE_KEYS = ['ctx_size', 'kv_quant', 'mtp', 'mtp_acceptance', 'mtp_greedy_tail', 'preserve_thinking', 'think_penalty'];
const allKeys = (over = {}) => Object.fromEntries(KEYS.map((k) => [k.name, over[k.name] ?? keyState()]));

// A stub server: its state lives here in node, every call is recorded, and a test can hold a response back.
function makeServer(opts = {}) {
  const calls = [];
  const gates = {};
  const state = {
    failSettings: 0, loadDelay: null, settingsDelay: [], loadFails: false,
    models: [
      { id: 'Qwen-A', state: 'ready', bytes_on_disk: 38 * GiB, bytes_resident: 40 * GiB, context_length: 262144, input_modalities: ['text', 'image'], keys: [...BASE_KEYS, 'vision'], settings: allKeys({ ctx_size: keyState({ file: 65536, source: 'file' }), kv_quant: keyState({ flag: '8', source: 'flag', locked: true }) }) },
      { id: 'GLM-B', state: 'unloaded', bytes_on_disk: 61 * GiB, context_length: 131072, input_modalities: ['text'], keys: BASE_KEYS, settings: allKeys({ kv_quant: keyState({ file: 'off', source: 'file' }) }) },
      { id: 'Deep-C', state: 'unloaded', bytes_on_disk: 96 * GiB, streaming: true, context_length: 131072, input_modalities: ['text'], keys: [...BASE_KEYS, 'ssd_budget_gb', 'ssd_prefetch'], settings: allKeys() },
      { id: 'Broke-D', state: 'error_state', bytes_on_disk: 20 * GiB, context_length: 8192, input_modalities: ['text'], keys: BASE_KEYS, settings: allKeys() },
    ],
    server: {
      models_dir: { default: '/h/.sushi/models', boot_roots: ['/models'], file: null, flag: '', source: 'default', locked: !!opts.dirLocked, applies: 'models are rescanned at once' },
      prefix_cache_disk_gb: { file: 100, flag: null, source: 'file', locked: false, effective_bytes: 100 * GiB, ask_bytes: 100 * GiB, applies: 'a running tier takes a new size at once and evicts down to it; 0 (off) and removing the key apply at the next load' },
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
  const modelsBody = () => ({ data: state.models.map((m) => ({ id: m.id, object: 'model', loaded: m.state === 'ready', state: m.state === 'error_state' ? 'error' : m.state, bytes_resident: m.bytes_resident ?? 0, bytes_on_disk: m.bytes_on_disk, streaming: !!m.streaming, streaming_required: !!m.streaming_required, ssd_budget_gb: m.ssd_budget_gb, context_length: m.context_length, input_modalities: m.input_modalities, capabilities: ['chat'], meta: { architecture: 'qwen4_exp', quantization: '2.75-bit' } })) });
  const settingsBody = () => ({ file: '/h/.sushi/model-settings.json', applies: 'next load of the model', model_keys: KEYS, server: state.server,
    models: state.models.map((m) => ({ id: m.id, path: `/models/${m.id}`, state: m.state, bytes_on_disk: m.bytes_on_disk, settings: m.settings, keys: m.keys, loaded_with: m.loaded_with ?? null, ...(opts.limits && m.limits ? { limits: m.limits } : {}) })) });
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
    switch (p) {
      case '/v1/models': return route.fulfill(json(modelsBody()));
      case '/v1/settings':
        if (method === 'POST') {
          const m = state.models.find((x) => x.id === body.model);
          if (opts.limits && body.set && 'think_penalty' in body.set) return route.fulfill(json({ error: { message: 'think_penalty is not supported by this model' } }, 400));
          for (const [k, v] of Object.entries(body.set ?? {})) if (m) m.settings[k] = keyState({ file: v, source: v === null ? 'default' : 'file' });
          for (const [k, v] of Object.entries(body.server ?? {})) { state.server[k] = { ...state.server[k], file: v, source: v === null ? 'default' : 'file' }; if (k === 'prefix_cache_disk_gb') state.server[k].effective_bytes = (v ?? 20) * GiB; }
          return route.fulfill(json({ applied: Object.fromEntries(Object.keys(body.set ?? body.server ?? {}).map((k) => [k, 'saved'])), model_loaded: m?.state === 'ready' }));
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
      assert.match(await page.textContent('.savebar'), /1 unsaved change/);
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
      assert.match(await page.textContent('#modelsPanel .alert.err'), /The server answered 500/);
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
      assert.match(await page.textContent(row('kv_quant')), /Launch flag/);
      assert.equal(await page.isVisible(row('vision')), true, 'an image model lists its vision tower');
      assert.equal(await page.locator(row('ssd_budget_gb')).count(), 0, 'a resident model has no SSD budget row');
      await page.click('.list [data-id="Deep-C"]');
      await page.waitForSelector(row('ssd_budget_gb'));
      assert.equal(await page.isVisible(row('ssd_budget_gb')), true, 'a streamed model shows it in Compact');
      assert.equal(await page.locator(row('vision')).count(), 0, 'a text model has no vision row');
      assert.equal(await page.evaluate(() => location.hash), '#Deep-C');
      // `streaming` only says the pack can stream; with no budget it loads resident.
      assert.match(await page.textContent('.detail .facts'), /can stream experts/);
      assert.doesNotMatch(await page.textContent('.list [data-id="Deep-C"]'), /streams experts/);
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
      assert.equal(await page.getAttribute('#cacheGb', 'placeholder'), '30', 'the placeholder is what the next load starts from');
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
    console.log('Web UI admin views (browser): passed');
  } finally {
    await browser.close();
  }
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
