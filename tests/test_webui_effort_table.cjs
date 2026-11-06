// sushi-test: tier=hermetic needs=node,playwright
// The Settings page's Thinking effort card and the Models page's thinking line, in a headless browser against a stub
// server that merges the tables as the engine does. SUSHI_EFFORT_SHOTS=<dir> also saves screenshots of every tab.
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
const WORDS = ['off', 'low', 'medium', 'high', 'xhigh', 'max'];
// The engine's built-in tables (`effort.builtin`): [offered, sends, cap, fold_to] per word.
const BUILTIN = {
  qwen4_exp: { def: 'medium', rows: { off: [true, null, null], low: [true, 'low', 2048], medium: [true, 'medium', 8192], high: [false, 'xhigh', null, 'xhigh'], xhigh: [true, 'xhigh', null], max: [false, 'xhigh', null, 'xhigh'] } },
  deepseek_v41: { def: 'high', rows: { off: [true, null, null], low: [true, 50, null], medium: [true, 62, null], high: [true, 75, null], xhigh: [true, 87, null], max: [true, 100, null] } },
  glm5_next: { def: 'high', rows: { off: [false, null, null], low: [true, 'low', null], medium: [false, 'high', null, 'high'], high: [true, 'high', null], xhigh: [false, 'max', null, 'max'], max: [true, 'max', null] } },
};
const SENDS = {
  qwen4_exp: { kind: 'word', choices: ['low', 'medium', 'xhigh'] },
  deepseek_v41: { kind: 'scale', min: 1, max: 100, presets: { low: 50, high: 75, max: 100 } },
  glm5_next: { kind: 'word', choices: ['low', 'high', 'max'] },
};
const ARCH = { 'Qwen-A': 'qwen4_exp', 'Deep-C': 'deepseek_v41', 'GLM-B': 'glm5_next', 'MiMo-D': 'mimo_v2' };

function makeServer(opts = {}) {
  const calls = [];
  const state = { file: JSON.parse(JSON.stringify(opts.file ?? {})), think: opts.think ?? null, postError: null };
  const merged = (fam) => {
    const b = BUILTIN[fam], f = state.file[fam] ?? {};
    const rows = Object.fromEntries(WORDS.map((w) => { const [offered, sends, cap, fold] = b.rows[w], e = f[w] ?? {}; return [w, { offered: e.offered ?? offered, sends: 'sends' in e ? e.sends : sends, cap: 'cap' in e ? e.cap : cap, fold_to: 'offered' in e ? null : fold ?? null }]; }));
    return { rows, def: f.default ?? b.def };
  };
  const effortBody = () => ({
    applies: 'applies at once',
    families: Object.keys(BUILTIN).map((fam) => {
      const m = merged(fam), b = BUILTIN[fam], f = state.file[fam] ?? {};
      const locked = !!state.think && m.rows[state.think]?.offered;
      return {
        family: fam, sends: SENDS[fam], off_offerable: fam !== 'glm5_next',
        default: { builtin: b.def, file: f.default ?? null, value: locked ? state.think : m.def, flag: locked ? `--think ${state.think}` : null, locked, source: locked ? 'flag' : f.default ? 'file' : 'default' },
        rows: WORDS.map((w) => ({ word: w, builtin: { offered: b.rows[w][0], sends: b.rows[w][1], cap: b.rows[w][2], fold_to: b.rows[w][3] ?? null }, file: f[w] ?? null, value: m.rows[w] })),
      };
    }),
  });
  const modelRow = (id) => {
    const fam = ARCH[id];
    if (fam === 'mimo_v2') return { reasoning_efforts: ['off', 'on'], reasoning_effort_budgets: {}, reasoning_effort_values: {}, default_reasoning_effort: 'on' };
    const m = merged(fam), offered = WORDS.filter((w) => m.rows[w].offered);
    return {
      reasoning_efforts: offered,
      reasoning_effort_budgets: Object.fromEntries(offered.filter((w) => m.rows[w].cap != null).map((w) => [w, m.rows[w].cap])),
      reasoning_effort_values: Object.fromEntries(offered.filter((w) => w !== 'off').map((w) => [w, m.rows[w].sends])),
      default_reasoning_effort: m.def,
    };
  };
  const models = ['Deep-C', 'GLM-B', 'MiMo-D', 'Qwen-A'];
  const modelsBody = () => ({ data: models.map((id, i) => ({ id, object: 'model', loaded: id === 'Qwen-A', state: id === 'Qwen-A' ? 'ready' : 'unloaded', bytes_resident: id === 'Qwen-A' ? 40 * GiB : 0, bytes_on_disk: (40 + i) * GiB, context_length: 262144, input_modalities: ['text'], capabilities: ['chat'], meta: { architecture: ARCH[id] }, ...modelRow(id) })) });
  const keyState = () => ({ file: null, flag: null, source: 'default', locked: false });
  const settingsBody = () => ({
    file: '/h/.sushi/model-settings.json', applies: 'next load of the model', model_keys: [{ name: 'ctx_size', type: 'int', min: 1, max: 4294967295 }],
    server: {
      models_dir: { default: '/h/.sushi/models', boot_roots: ['/models'], file: null, flag: null, source: 'default', locked: false, applies: 'models are rescanned at once' },
      prefix_cache_disk_gb: { file: null, flag: null, source: 'default', locked: false, effective_bytes: 100 * GiB, ask_bytes: 100 * GiB, default_bytes: 100 * GiB, running_tiers: [], applies: 'a running tier takes a new size at once' },
      update_check: { file: null, flag: null, default: true, source: 'default', locked: false, applies: 'applies at once' },
      pack_update_check: { file: null, flag: null, default: true, source: 'default', locked: false, applies: 'applies at once' },
      port: { file: null, flag: null, default: 12345, source: 'default', locked: false, applies: 'the next start; a running server keeps its port' },
    },
    effort: effortBody(),
    models: models.map((id) => ({ id, path: `/models/${id}`, state: id === 'Qwen-A' ? 'ready' : 'unloaded', bytes_on_disk: 40 * GiB, settings: { ctx_size: keyState() }, keys: ['ctx_size'], loaded_with: null, defaults: null })),
  });
  const json = (body, status = 200) => ({ status, contentType: 'application/json', body: JSON.stringify(body) });
  async function handle(route) {
    const request = route.request(), url = new URL(request.url()), p = url.pathname, method = request.method();
    if (/^\/(chat|dashboard|models|settings)?$/.test(p)) return route.fulfill({ status: 200, contentType: 'text/html; charset=utf-8', body: html });
    const body = method === 'POST' ? JSON.parse(request.postData() || '{}') : null;
    calls.push({ path: p, method, body });
    switch (p) {
      case '/v1/models': return route.fulfill(json(modelsBody()));
      case '/v1/settings':
        if (method === 'POST') {
          if (state.postError) return route.fulfill(json({ error: { message: state.postError, type: 'invalid_setting' } }, 400));
          for (const [fam, edits] of Object.entries(body.effort ?? {})) {
            if (edits === null) { delete state.file[fam]; continue; }
            const sec = state.file[fam] ?? {};
            for (const [k, v] of Object.entries(edits)) { if (v === null) delete sec[k]; else sec[k] = v; }
            if (Object.keys(sec).length) state.file[fam] = sec; else delete state.file[fam];
          }
          return route.fulfill(json({ ok: true, applied: { effort: 'applies at once, from the next request' }, model_loaded: false }));
        }
        return route.fulfill(json(settingsBody()));
      case '/v1/prefix-cache': return route.fulfill(json({ root: '/h/.sushi/kv-cache', volume: { free_bytes: 900 * GiB, total_bytes: 2000 * GiB }, total_bytes: 0, namespaces: [] }));
      case '/props': return route.fulfill(json({ settings: { version: '9.9.9', mtp: {}, prefix_cache: {} }, memory: { available_bytes: 5e9, total_bytes: 128 * GiB }, batching: {} }));
      case '/v1/tools': return route.fulfill(json({ edit_default: false }));
      case '/metrics.json': case '/metrics': return route.fulfill(json({ error: { message: 'metrics off' } }, 503));
      default: return route.fulfill(json({}, 404));
    }
  }
  return { state, calls, handle, posts: () => calls.filter((c) => c.path === '/v1/settings' && c.method === 'POST').map((c) => c.body) };
}

async function open(browser, server, urlPath, { viewport = { width: 1280, height: 900 }, scheme = 'dark' } = {}) {
  const context = await browser.newContext({ viewport, colorScheme: scheme });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.route(`${ORIGIN}/**`, server.handle);
  await page.goto(`${ORIGIN}${urlPath}`);
  return { page, context, errors };
}
const settle = (page) => page.waitForTimeout(200);
const tab = (fam) => `#effortTabs [data-fam="${fam}"]`;
const tr = (word) => `#effortRows tr[data-word="${word}"]`;
const preview = (page) => page.innerText('#effortPreview');
const QWEN_FILE = { qwen4_exp: { high: { offered: true, sends: 'medium', cap: 64 } }, deepseek_v41: { medium: { sends: 40 } } };

async function main() {
  const playwright = loadPlaywright();
  if (!playwright) { console.log('Web UI effort table (browser): skipped (Playwright is not installed)'); return; }
  let browser;
  try { browser = await playwright.chromium.launch(); } catch (error) {
    console.log(`Web UI effort table (browser): skipped (no Chromium: ${String(error.message).split('\n')[0]})`);
    return;
  }
  const failures = [];
  async function scenario(name, opts, body) {
    const server = makeServer(opts.server);
    const { page, context, errors } = await open(browser, server, opts.path ?? '/settings', opts);
    try {
      if ((opts.path ?? '/settings') === '/settings') await page.waitForSelector('#effortCard:not([hidden]) #effortRows tr');
      await body({ page, server });
      assert.deepEqual(errors, [], 'no page errors');
    } catch (e) { failures.push(`${name}: ${e.message.split('\n')[0]}`); } finally { await context.close(); }
  }
  try {
    await scenario('the card sits under Inference with a tab per family, built-in values as placeholders and a Now badge', { server: { file: QWEN_FILE } }, async ({ page }) => {
      assert.equal(await page.isVisible('#secInference'), true);
      assert.equal(await page.evaluate(() => document.getElementById('secInference').nextElementSibling.id), 'effortCard');
      assert.deepEqual(await page.$$eval('#effortTabs button', (b) => b.map((x) => x.textContent)), ['Qwen', 'DeepSeek', 'GLM']);
      assert.equal(await page.getAttribute(tab('qwen4_exp'), 'aria-selected'), 'true');
      assert.deepEqual(await page.$$eval('#effortRows tr', (r) => r.map((x) => x.dataset.word)), ['off', 'low', 'medium', 'high', 'xhigh', 'max']);
      assert.match(await page.innerText('#effortBadges'), /Saved/);
      assert.match(await page.innerText('#effortBadges'), /Now/);
      assert.equal(await page.getAttribute(`${tr('low')} input[id$="-cap"]`, 'placeholder'), '2048', 'a built-in cap is the placeholder');
      assert.equal(await page.inputValue(`${tr('low')} input[id$="-cap"]`), '');
      assert.equal(await page.$eval(`${tr('low')} select`, (s) => s.options[0].textContent), 'low', 'a default value shows plainly');
      assert.equal(await page.isVisible(`${tr('high')} [data-changed]`), true, 'an edited row says so');
      assert.equal(await page.isVisible(`${tr('low')} [data-changed]`), false);
      const words = await page.evaluate(() => { const c = document.getElementById('effortCard'); return [c.innerText, ...[...c.querySelectorAll('option')].map((o) => o.textContent), ...[...c.querySelectorAll('[data-tip],[placeholder]')].map((n) => `${n.dataset.tip ?? ''} ${n.getAttribute('placeholder') ?? ''}`)].join(' | '); });
      assert.doesNotMatch(words, /built-in|fold/i, 'the card speaks of defaults, in its text, options, tips and placeholders');
      assert.equal(await page.inputValue(`${tr('high')} select`), 'medium', 'a saved value shows as itself');
      assert.equal(await page.inputValue(`${tr('high')} input[id$="-cap"]`), '64');
      assert.equal(await page.isChecked(`${tr('high')} input[type=checkbox]`), true);
      assert.equal(await page.isChecked(`${tr('max')} input[type=checkbox]`), true, 'a built-in fold is accepted');
      assert.equal(await page.$eval(`${tr('max')} select`, (s) => s.options[s.selectedIndex].textContent), 'same as xhigh');
      assert.equal(await page.getAttribute(`${tr('max')} input[id$="-cap"]`, 'placeholder'), 'same as xhigh');
      assert.match(await page.getAttribute(`${tr('max')} select`, 'data-tip'), /^Qwen has no max level, so max uses xhigh\./);
      assert.equal(await page.isChecked(`${tr('medium')} input[type=radio]`), true, 'medium is the default');
      for (const col of ['Offered', 'Sends', 'Cap', 'Default']) assert.ok(await page.$(`#effortTbl th span[data-tip]:text-is("${col}")`), `${col} explains itself`);
      const heads = await page.$$eval('#effortTbl th span[data-tip]', (ns) => Object.fromEntries(ns.map((n) => [n.textContent, n.dataset.term ?? ''])));
      assert.match(heads.Offered, /reasoning_efforts/);
      assert.match(heads.Sends, /reasoning_effort/);
      assert.match(heads.Cap, /thinking budget.*reasoning_budget_tokens/);
      assert.match(heads.Default, /default_reasoning_effort/);
      assert.equal(await page.isVisible('#effortReset'), true);
      assert.equal(await page.innerText('#effortReset'), 'Reset Qwen to defaults');
    });

    await scenario('the preview shows the template text and the cap of the selected row, per family', { server: { file: QWEN_FILE } }, async ({ page }) => {
      await page.click(`${tr('xhigh')} th`);
      assert.match(await preview(page), /xhigh sends "xhigh"; no cap\./);
      assert.match(await preview(page), /Reasoning effort is set to xhigh\. Please think carefully/);
      await page.click(`${tr('high')} th`);
      assert.match(await preview(page), /^high sends "medium"; cap 64, then the thought is closed\./);
      assert.match(await preview(page), /the template adds no instruction line/);
      await page.click(`${tr('max')} th`);
      assert.match(await preview(page), /^Qwen has no max level, so max uses xhigh; the reply carries an effort_notice\. xhigh sends "xhigh"/);
      await page.click(`${tr('off')} th`);
      assert.match(await preview(page), /thinking is off/);
      await page.click(tab('deepseek_v41'));
      await page.click(`${tr('medium')} th`);
      assert.match(await preview(page), /^medium sends 40; no cap\./);
      assert.match(await preview(page), /Reasoning Effort: 40 \(range 1-100, the higher the value, the more thorough the reasoning\)/);
      await page.click(tab('glm5_next'));
      await page.click(`${tr('low')} th`);
      assert.match(await preview(page), /Reasoning Effort: Low$/m);
      await page.click(`${tr('medium')} th`);
      assert.match(await preview(page), /^GLM has no medium level, so medium uses high; the reply carries an effort_notice\./);
      assert.equal(await page.$eval(`${tr('xhigh')} select`, (s) => s.options[s.selectedIndex].textContent), 'same as max');
      await page.click(`${tr('off')} th`);
      assert.match(await preview(page), /off is not offered: a request for it is refused\./);
      assert.equal(await page.isDisabled(`${tr('off')} input[type=checkbox]`), true, "GLM's off cannot be offered");
    });

    await scenario('a bad value stays on its row and blocks Save; a fix clears it', {}, async ({ page }) => {
      await page.click(tab('deepseek_v41'));
      await page.fill(`${tr('medium')} input[id$="-sends"]`, '0');
      assert.match(await page.innerText(`${tr('medium')} [data-bad]`), /Sends: enter a whole number from 1 to 100/);
      assert.equal(await page.getAttribute(`${tr('medium')} input[id$="-sends"]`, 'aria-invalid'), 'true');
      assert.equal(await page.innerText(`${tr('low')} [data-bad]`), '', 'other rows stay clean');
      assert.equal(await page.isDisabled('#effortSave'), true);
      await page.fill(`${tr('medium')} input[id$="-sends"]`, '40');
      assert.equal(await page.innerText(`${tr('medium')} [data-bad]`), '');
      assert.equal(await page.isDisabled('#effortSave'), false);
      for (const bad of ['0', '-5', 'abc', '1.5']) {
        await page.fill(`${tr('low')} input[id$="-cap"]`, bad);
        assert.match(await page.innerText(`${tr('low')} [data-bad]`), /Cap: enter a whole number/, `cap ${bad}`);
      }
      await page.fill(`${tr('low')} input[id$="-cap"]`, 'none');
      assert.equal(await page.innerText(`${tr('low')} [data-bad]`), '', 'none is no cap');
      await page.uncheck(`${tr('high')} input[type=checkbox]`);
      assert.match(await page.innerText(`${tr('high')} [data-bad]`), /The default word must be offered/);
      assert.equal(await page.isDisabled('#effortSave'), true);
    });

    await scenario('Save posts every row of the family, null where it is built-in, and the chat menu follows', {}, async ({ page, server }) => {
      await page.check(`${tr('high')} input[type=checkbox]`);
      await page.selectOption(`${tr('high')} select`, 'medium');
      await page.fill(`${tr('high')} input[id$="-cap"]`, '64');
      await page.uncheck(`${tr('low')} input[type=checkbox]`);
      await page.uncheck(`${tr('max')} input[type=checkbox]`);
      await page.click(`${tr('max')} th`);
      assert.match(await preview(page), /^max is not offered: a request for it is refused\./);
      assert.equal(await page.isVisible('#effortSave'), true);
      await page.click('#effortSave');
      await settle(page);
      assert.deepEqual(server.posts(), [{ effort: { qwen4_exp: { default: null, off: null, low: { offered: false }, medium: null, high: { offered: true, sends: 'medium', cap: 64 }, xhigh: null, max: { offered: false } } } }]);
      assert.equal(await page.isVisible('#effortSave'), false, 'nothing left to save');
      assert.equal(await page.inputValue(`${tr('high')} select`), 'medium');
      await page.goto(`${ORIGIN}/chat`);
      await page.waitForSelector('#modelSelect option[value="Qwen-A"]', { state: 'attached' });
      await page.selectOption('#modelSelect', 'Qwen-A');
      await page.waitForSelector('#effortSelect option', { state: 'attached' });
      const options = await page.$$eval('#effortSelect option', (o) => o.map((x) => x.value));
      assert.ok(options.includes('high') && !options.includes('low'), `the chat menu lists the table's offered words (${options})`);
    });

    await scenario('a preset chip sets DeepSeek\'s number; the default radio moves the default', {}, async ({ page, server }) => {
      await page.click(tab('deepseek_v41'));
      await page.click(`${tr('medium')} .presets button:text-is("75")`);
      assert.equal(await page.inputValue(`${tr('medium')} input[id$="-sends"]`), '75');
      assert.equal(await page.getAttribute(`${tr('medium')} .presets button:text-is("75")`, 'aria-pressed'), 'true');
      await page.check(`${tr('low')} input[type=radio]`);
      await page.click('#effortSave');
      await settle(page);
      const body = server.posts()[0].effort.deepseek_v41;
      assert.equal(body.default, 'low');
      assert.deepEqual(body.medium, { sends: 75 });
    });

    await scenario('Reset asks first, then posts the family as null', { server: { file: QWEN_FILE } }, async ({ page, server }) => {
      await page.click('#effortReset');
      await page.waitForSelector('#confirmDialog[open]');
      assert.match(await page.innerText('#cdTitle'), /Reset Qwen to defaults\?/);
      await page.click('#cdOk');
      await settle(page);
      assert.deepEqual(server.posts(), [{ effort: { qwen4_exp: null } }]);
      assert.equal(await page.isVisible('#effortReset'), false, 'a built-in table has nothing to reset');
      assert.equal(await page.inputValue(`${tr('high')} select`), '');
    });

    await scenario('a refusal from the server lands on the row it names', {}, async ({ page, server }) => {
      server.state.postError = 'effort qwen4_exp.high.sends: Qwen takes low, medium or xhigh';
      await page.selectOption(`${tr('high')} select`, 'medium');
      await page.click('#effortSave');
      await settle(page);
      assert.match(await page.innerText(`${tr('high')} [data-bad]`), /Qwen takes low, medium or xhigh/);
      assert.equal(await page.isVisible('#effortErr'), false);
    });

    await scenario('the controls work from the keyboard', {}, async ({ page }) => {
      await page.focus(tab('qwen4_exp'));
      await page.keyboard.press('ArrowRight');
      assert.equal(await page.getAttribute(tab('deepseek_v41'), 'aria-selected'), 'true');
      assert.equal(await page.evaluate(() => document.activeElement.dataset.fam), 'deepseek_v41');
      await page.keyboard.press('End');
      assert.equal(await page.evaluate(() => document.activeElement.dataset.fam), 'glm5_next');
      await page.keyboard.press('ArrowRight');
      assert.equal(await page.evaluate(() => document.activeElement.dataset.fam), 'qwen4_exp', 'the arrows wrap');
      await page.focus(`${tr('max')} input[type=checkbox]`);
      assert.match(await preview(page), /^Qwen has no max level/, 'focus selects the row');
      await page.keyboard.press('Space');
      assert.equal(await page.isChecked(`${tr('max')} input[type=checkbox]`), false);
      assert.equal(await page.evaluate(() => document.activeElement.id), 'eff-max-on', 'focus survives the re-render');
      assert.match(await preview(page), /^max is not offered: a request for it is refused/);
      await page.keyboard.press('Tab');
      assert.equal(await page.evaluate(() => document.activeElement.id), 'eff-max-sends');
    });

    await scenario('--think locks the default column and says so', { server: { think: 'xhigh' } }, async ({ page, server }) => {
      assert.match(await page.innerText('#effortFlag'), /Default set by --think xhigh at launch/);
      assert.equal(await page.isDisabled(`${tr('low')} input[type=radio]`), true);
      assert.equal(await page.isChecked(`${tr('xhigh')} input[type=radio]`), true, 'the Default column shows the word --think serves');
      assert.match(await preview(page), /^xhigh sends "xhigh"/, 'and the preview opens on it');
      await page.uncheck(`${tr('xhigh')} input[type=checkbox]`);
      await settle(page);
      assert.notEqual(await page.innerText(`${tr('xhigh')} [data-bad]`), '', 'unoffering the --think word is refused in its row, as the server refuses it');
      assert.equal(await page.isDisabled('#effortSave'), true);
      await page.check(`${tr('xhigh')} input[type=checkbox]`);
      await page.fill(`${tr('low')} input[id$="-cap"]`, '100');
      await page.click('#effortSave');
      await settle(page);
      assert.deepEqual(server.posts().map((b) => b.effort.qwen4_exp.default), [null], 'a save keeps the file default, not the flag word');
    });

    await scenario('the Models page says how each model thinks and links to its tab', { path: '/models#Qwen-A' }, async ({ page }) => {
      await page.waitForSelector('#modelsDetail .think-line:not([hidden])');
      assert.equal(await page.innerText('#modelsDetail .think-line'), 'Thinking: medium by default · low 2048 / medium 8192 tokens · Edit');
      assert.doesNotMatch(await page.getAttribute('#modelsDetail .think-line a', 'data-tip'), /built-in|fold/i);
      assert.match(await page.getAttribute('#modelsDetail .think-line span[data-tip]', 'data-term'), /default_reasoning_effort.*reasoning_effort_budgets/, 'the Thinking line names its fields');
      await page.goto(`${ORIGIN}/models#MiMo-D`);
      await page.waitForFunction(() => document.querySelector('#modelsDetail h2')?.textContent === 'MiMo-D');
      assert.equal(await page.isVisible('#modelsDetail .think-line'), false, 'a model without an effort table has no line');
      await page.goto(`${ORIGIN}/models#GLM-B`);
      await page.waitForFunction(() => document.querySelector('#modelsDetail h2')?.textContent === 'GLM-B');
      await page.waitForSelector('#modelsDetail .think-line a');
      assert.equal(await page.innerText('#modelsDetail .think-line'), 'Thinking: high by default · no token caps · Edit');
      await page.click('#modelsDetail .think-line a');
      await page.waitForSelector('#settingsPanel:not([hidden]) #effortRows tr');
      assert.equal(await page.getAttribute(tab('glm5_next'), 'aria-selected'), 'true');
    });

    const shots = process.env.SUSHI_EFFORT_SHOTS;
    if (shots) {
      fs.mkdirSync(shots, { recursive: true });
      for (const scheme of ['light', 'dark']) {
        const server = makeServer({ file: QWEN_FILE });
        const { page, context } = await open(browser, server, '/settings', { scheme, viewport: { width: 1100, height: 1000 } });
        await page.waitForSelector('#effortCard:not([hidden]) #effortRows tr');
        for (const [fam, name, word] of [['qwen4_exp', 'qwen', 'high'], ['deepseek_v41', 'deepseek', 'medium'], ['glm5_next', 'glm', 'xhigh']]) {
          await page.click(tab(fam));
          await page.click(`${tr(word)} th`);
          await settle(page);
          const box = await page.evaluate(() => { const a = document.getElementById('secInference').getBoundingClientRect(), b = document.getElementById('effortCard').getBoundingClientRect(); return { x: b.x - 8, y: a.y - 8, width: b.width + 16, height: b.bottom - a.y + 16 }; });
          await page.screenshot({ path: path.join(shots, `effort-${name}-${scheme}.png`), clip: box });
        }
        await page.goto(`${ORIGIN}/models#Qwen-A`);
        await page.waitForSelector('#modelsDetail .think-line:not([hidden])');
        await settle(page);
        await page.screenshot({ path: path.join(shots, `models-thinking-line-${scheme}.png`) });
        await context.close();
      }
    }

    if (failures.length) throw new Error(`${failures.length} effort scenario(s) failed:\n  ${failures.join('\n  ')}`);
    console.log('Web UI effort table (browser): passed');
  } finally {
    await browser.close();
  }
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
