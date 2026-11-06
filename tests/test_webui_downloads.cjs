// sushi-test: tier=hermetic needs=node,playwright
// The Models tab's Available list in a real headless browser against a stubbed server: /downloads as its deep link, the
// pack rows (recommended, too big for this Mac, downloaded and opened from Installed, update, an interrupted pack to
// resume), Download and Stop, the progress bar with speed and time left (and its percent on the Available switch), no DOM
// write on an unchanged poll (also while the folder grows), the "new pack" dot on the Models tab (never seeded from a
// catalog still loading, never under --no-update-check), and no sideways scroll at 320, 390 and 1440 px.
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
const PHONE = { width: 390, height: 844 };
const GiB = 2 ** 30;

const pack = (repo, extra) => ({ repo, name: repo.split('/')[1], tag: '', size_bytes: 40 * GiB, tier_gb: null, fits: null, recommended: false, installed: false, path: null, partial_bytes: null, update: null, launch: null, ...extra });
const PACKS = [
  pack('beamster/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw', { name: 'qwen3.8-flash-next', tag: '2.75bpw', tier_gb: 64, fits: true, recommended: true }),
  pack('beamster/MiMo-V2.6-Flash-Sushi-2.3bpw', { name: 'mimo-v2.6-flash', tag: '2.3bpw', tier_gb: 128, fits: false }),
  pack('beamster/GLM-5.3-Flash-Sushi-2bpw', { name: 'glm-5.3-flash', tag: '2bpw', tier_gb: 96, fits: false, partial_bytes: 3 * GiB }),
  pack('beamster/DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw', { name: 'deepseek-v4.1-flash', tag: '1.75bpw', tier_gb: 96, fits: false, streams_experts: true, installed: true, path: '/m/d', launch: "sushi serve --model '/m/d'", update: { current: '1.0', latest: '2.0' } }),
  pack('beamster/Qwen3.8-Flash-Next-Sushi-4bpw', { name: 'qwen3.8-flash-next', tag: '4bpw', tier_gb: 96, fits: false, installed: true, path: '/m/beamster/Qwen3.8-Flash-Next-Sushi-4bpw', launch: "sushi serve --model '/m/q4' --mtp" }),
];

function makeServer() {
  const calls = [];
  const state = { packs: PACKS.map((p) => ({ ...p })), jobs: [], models: [], catalog: 'ready', updateCheck: true, stopping: false };
  const json = (body, status = 200) => ({ status, contentType: 'application/json', body: JSON.stringify(body) });
  async function handle(route) {
    const request = route.request(), url = new URL(request.url()), p = url.pathname;
    if (/^\/(chat|dashboard|models|settings|downloads)?$/.test(p)) return route.fulfill({ status: 200, contentType: 'text/html; charset=utf-8', body: html });
    calls.push({ path: p, method: request.method(), body: request.postData() });
    switch (p) {
      case '/v1/models': return route.fulfill(json({ data: state.models }));
      case '/props': return route.fulfill(json({ settings: { version: '9.9.9', mtp: {}, prefix_cache: {} }, memory: { available_bytes: 5e9, total_bytes: 64 * GiB }, batching: {} }));
      case '/v1/settings': return route.fulfill(json({ file: '/h/f.json', model_keys: [], applies: 'x', server: { models_dir: { default: '/d', boot_roots: ['/m'], file: null, flag: '', locked: false }, prefix_cache_disk_gb: { file: null, locked: false, effective_bytes: GiB } }, models: [] }));
      case '/metrics.json': return route.fulfill(json({ counters: {}, gauges: {}, histograms: {}, sessions: [] }));
      case '/v1/downloads':
        if (request.method() === 'POST') {
          const repo = JSON.parse(request.postData()).repo;
          state.jobs = [{ repo, dest: `/m/${repo}`, phase: 'queued', files: 0, file_index: 0, total_bytes: 0, done_bytes: 0, file: null, speed_bps: null, eta_s: null, message: null }];
          return route.fulfill(json({ ok: true, queued: true, dest: `/m/${repo}`, warning: null }, 202));
        }
        return route.fulfill(json({ ram_bytes: 64 * GiB, ram_gb: 64, dest: { path: '/Users/me/.sushi/models', source: 'default', locked: false, volume: 'Macintosh HD', free_bytes: 500 * GiB, total_bytes: 2000 * GiB }, catalog: state.catalog, refreshing: false, update_check: state.updateCheck, packs: state.packs, jobs: state.jobs }));
      case '/v1/downloads/cancel': state.jobs = state.jobs.map((j) => (state.stopping ? { ...j, stopping: true } : { ...j, phase: 'cancelled' })); return route.fulfill(json({ ok: true }));
      default: return route.fulfill(json({}, 404));
    }
  }
  return { handle, calls, state };
}

async function open(browser, server, urlPath, { viewport = { width: 1440, height: 900 }, storage = {} } = {}) {
  const context = await browser.newContext({ viewport, colorScheme: 'dark' });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.route(`${ORIGIN}/**`, server.handle);
  await page.addInitScript((seed) => {
    if (sessionStorage.getItem('__seeded')) return;
    sessionStorage.setItem('__seeded', '1');
    for (const [k, v] of Object.entries(seed)) localStorage.setItem(k, v);
  }, storage);
  await page.goto(`${ORIGIN}${urlPath}`);
  return { page, context, errors };
}

const row = (repo) => `.dl-row[data-repo="${repo}"]`;
const seenNow = (page) => page.evaluate(() => JSON.parse(localStorage.getItem('sushi-seen-packs')));

async function main() {
  const playwright = loadPlaywright();
  if (!playwright) { console.log('Web UI downloads: skipped (Playwright is not installed)'); return; }
  let browser;
  try { browser = await playwright.chromium.launch(); } catch (error) {
    console.log(`Web UI downloads: skipped (no Chromium: ${String(error.message).split('\n')[0]})`);
    return;
  }
  try {
    {
      // /downloads opens the Models tab on its Available list; each row says what it is for this Mac, and the button
      // matches what is on disk.
      const server = makeServer();
      const { page, context, errors } = await open(browser, server, '/downloads');
      await page.waitForSelector('.dl-row');
      assert.equal(await page.getAttribute('#navModels', 'aria-current'), 'page', 'the deep link lands on the Models tab');
      assert.equal(await page.getAttribute('#modelsTabAvailable', 'aria-current'), 'page');
      assert.equal(await page.getAttribute('#modelsTabInstalled', 'aria-current'), null);
      assert.equal(await page.isVisible('#modelsList'), false, 'the installed list makes room for the available one');
      assert.equal(await page.$('#navDownloads'), null, 'Downloads is no tab of its own');
      assert.match(await page.textContent('#dlMac'), /64 GB of memory/);
      assert.match(await page.textContent('#dlMac'), /500 GiB free on Macintosh HD/);
      const qwen = row(PACKS[0].repo), mimo = row(PACKS[1].repo), glm = row(PACKS[2].repo), ds = row(PACKS[3].repo);
      assert.match(await page.textContent(qwen), /Recommended for this Mac/);
      assert.match(await page.textContent(`${qwen} .act`), /^Download$/);
      assert.match(await page.textContent(mimo), /Needs 128 GB/);
      assert.match(await page.textContent(mimo), /this Mac has 64 GB/);
      assert.match(await page.textContent(`${glm} .act`), /^Resume$/);
      assert.match(await page.textContent(glm), /3\.00 GiB on disk, not complete/);
      assert.match(await page.textContent(ds), /Downloaded/);
      assert.match(await page.textContent(`${ds} .act`), /^Update$/);
      assert.match(await page.textContent(ds), /Needs 96 GB of memory, reading part of the model from the SSD/);
      assert.doesNotMatch(await page.textContent(ds), /in memory/, 'DeepSeek always streams');
      assert.doesNotMatch(await page.textContent('#downloadsPanel'), /sushi serve/, 'no launch command to copy');
      assert.equal(await page.getAttribute(`${glm} .act button`, 'aria-label'), 'Resume glm-5.3-flash:2bpw');

      // A pack above this Mac's RAM asks first; Cancel sends nothing.
      await page.click(`${mimo} .act button`);
      await page.waitForSelector('#confirmDialog[open]');
      assert.match(await page.textContent('#cdBody'), /Made for Macs with 128 GB of memory or more/);
      await page.click('#cdCancel');
      assert.deepEqual(server.calls.filter((c) => c.path.startsWith('/v1/downloads') && c.method === 'POST'), []);

      await page.click(`${qwen} .act button`);
      await page.waitForFunction((r) => /Remove/.test(document.querySelector(`.dl-row[data-repo="${r}"] .act`).textContent), PACKS[0].repo);
      assert.deepEqual(server.calls.filter((c) => c.path.startsWith('/v1/downloads') && c.method === 'POST').map((c) => [c.path, JSON.parse(c.body)]), [['/v1/downloads', { repo: PACKS[0].repo }]]);
      assert.deepEqual(errors, []);

      // A downloaded pack opens in the installed list, where it loads; the switch goes both ways.
      const q4 = row(PACKS[4].repo);
      assert.equal(await page.getAttribute(`${q4} .act button`, 'aria-label'), 'Open qwen3.8-flash-next:4bpw in Installed');
      await page.click(`${q4} .act button`);
      await page.waitForFunction(() => location.pathname === '/models' && location.hash === '#Qwen3.8-Flash-Next-Sushi-4bpw');
      assert.equal(await page.getAttribute('#modelsTabInstalled', 'aria-current'), 'page');
      assert.equal(await page.isVisible('#downloadsPanel'), false);
      await page.click('#modelsTabAvailable');
      await page.waitForFunction(() => location.pathname === '/downloads');
      await page.waitForSelector('#downloadsPanel .dl-row');
      await page.click('#modelsTabInstalled');
      await page.waitForFunction(() => location.pathname === '/models');
      await page.goBack();
      await page.waitForSelector('#downloadsPanel:visible');
      await context.close();
    }

    {
      // A running download: bar, bytes, speed, time left and the file in flight; Stop sends cancel; an identical poll writes nothing.
      const server = makeServer();
      server.state.jobs = [{ repo: PACKS[0].repo, dest: '/m/q', phase: 'downloading', files: 12, file_index: 3, total_bytes: 40 * GiB, done_bytes: 10 * GiB, file: { path: 'model-00003.safetensors', size: 4 * GiB, done: GiB }, speed_bps: 100 * 2 ** 20, eta_s: 307, message: null }];
      const { page, context } = await open(browser, server, '/downloads');
      const qwen = row(PACKS[0].repo);
      await page.waitForSelector(`${qwen} .prog:not([hidden])`);
      await page.waitForFunction((r) => /left/.test(document.querySelector(`.dl-row[data-repo="${r}"] .prog`).textContent), PACKS[0].repo);
      assert.equal(await page.$eval(`${qwen} .meter > span`, (s) => s.style.width), '25%');
      assert.equal(await page.getAttribute(`${qwen} .meter`, 'aria-valuenow'), '25');
      const text = await page.textContent(`${qwen} .prog`);
      assert.match(text, /model-00003\.safetensors \(3 of 12\)/);
      assert.match(text, /10\.0 GiB of 40\.0 GiB · 100 MiB\/s · 5 min left/);
      await page.evaluate(() => { window.__writes = []; new MutationObserver((rs) => { for (const r of rs) window.__writes.push(`${r.type}:${r.target.className || r.target.nodeName}`); }).observe(document.getElementById('downloadsPanel'), { subtree: true, childList: true, attributes: true, characterData: true }); });
      const before = server.calls.filter((c) => c.path === '/v1/downloads').length;
      await page.waitForTimeout(2600);
      assert.ok(server.calls.filter((c) => c.path === '/v1/downloads').length >= before + 2, 'a running download is polled every second');
      assert.deepEqual(await page.evaluate(() => window.__writes), [], 'an unchanged poll writes nothing');
      // The folder of a running download grows: the row and its Stop button stay the same nodes.
      const stopBefore = await page.$(`${qwen} .act button`);
      server.state.packs[0] = { ...server.state.packs[0], partial_bytes: 11 * GiB };
      await page.waitForTimeout(1300);
      server.state.packs[0] = { ...server.state.packs[0], partial_bytes: 12 * GiB };
      await page.waitForTimeout(1300);
      assert.ok(await stopBefore.evaluate((b) => b.isConnected), 'the Stop button is the same node across polls');
      assert.deepEqual(await page.evaluate(() => window.__writes), [], 'a growing folder writes nothing');
      server.state.jobs[0] = { ...server.state.jobs[0], done_bytes: 30 * GiB, eta_s: 40 };
      await page.waitForFunction((r) => /30\.0 GiB of 40\.0 GiB/.test(document.querySelector(`.dl-row[data-repo="${r}"] .prog`).textContent), PACKS[0].repo);
      assert.equal(await page.$eval(`${qwen} .meter > span`, (s) => s.style.width), '75%');
      // Hours and minutes round as one time: 7,190 s left is 2 h 0 min, never 1 h 60 min.
      server.state.jobs[0] = { ...server.state.jobs[0], eta_s: 7190 };
      await page.waitForFunction((r) => / h .* min left/.test(document.querySelector(`.dl-row[data-repo="${r}"] .prog`).textContent), PACKS[0].repo);
      assert.match(await page.textContent(`${qwen} .prog`), /· 2 h 0 min left/);
      // Stop shows "Stopping…" until the server reports the job stopped.
      server.state.stopping = true;
      await page.click(`${qwen} .act button`);
      await page.waitForFunction((r) => /Stopping/.test(document.querySelector(`.dl-row[data-repo="${r}"] .act`).textContent), PACKS[0].repo);
      assert.equal(await page.isDisabled(`${qwen} .act button`), true);
      server.state.jobs = server.state.jobs.map((j) => ({ ...j, phase: 'cancelled' }));
      await page.waitForFunction((r) => /Resume/.test(document.querySelector(`.dl-row[data-repo="${r}"] .act`).textContent), PACKS[0].repo);
      assert.ok(server.calls.some((c) => c.path === '/v1/downloads/cancel' && JSON.parse(c.body).repo === PACKS[0].repo));
      assert.match(await page.textContent(qwen), /downloaded part is kept/);
      await context.close();
    }

    {
      // On the installed list a running download shows its percent on the Available switch, and the finished pack
      // reaches the list without a reload.
      const server = makeServer();
      server.state.jobs = [{ repo: PACKS[0].repo, dest: '/m/q', phase: 'downloading', files: 12, file_index: 3, total_bytes: 40 * GiB, done_bytes: 10 * GiB, file: null, speed_bps: 1, eta_s: 9, message: null }];
      const { page, context } = await open(browser, server, '/downloads');
      await page.waitForSelector(`${row(PACKS[0].repo)} .prog:not([hidden])`);
      await page.click('#modelsTabInstalled');
      await page.waitForFunction(() => document.getElementById('modelsTabPct').textContent === '25%');
      server.state.jobs[0] = { ...server.state.jobs[0], done_bytes: 30 * GiB };
      await page.waitForFunction(() => document.getElementById('modelsTabPct').textContent === '75%', null, { timeout: 4000 });
      server.state.jobs[0] = { ...server.state.jobs[0], phase: 'done', done_bytes: 40 * GiB };
      server.state.models = [{ id: 'Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw', object: 'model', state: 'unloaded', bytes_on_disk: 40 * GiB }];
      await page.waitForSelector('#modelsList [data-id="Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw"]', { timeout: 4000 });
      assert.equal(await page.textContent('#modelsTabPct'), '', 'no percent once it is done');
      await context.close();
    }

    {
      // The dot: the first full catalog seeds the seen list silently (a catalog still loading, the known packs only,
      // seeds nothing); a pack published later marks the Models tab and its Available switch from any view until the
      // available list is shown; an update of a downloaded pack does the same.
      const server = makeServer();
      server.state.catalog = 'loading';
      const later_pack = pack('beamster/Older-Sushi-1bpw');
      const first = await open(browser, server, '/chat');
      await first.page.waitForTimeout(3600);
      assert.equal(await first.page.evaluate(() => localStorage.getItem('sushi-seen-packs')), null, 'a loading catalog seeds nothing');
      server.state.catalog = 'ready';
      server.state.packs.push(later_pack);
      await first.page.waitForFunction(() => localStorage.getItem('sushi-seen-packs'), null, { timeout: 8000 });
      assert.equal(await first.page.isVisible('#navModelsDot'), false, 'nothing is new on the first visit');
      const seen = await seenNow(first.page);
      assert.ok(seen.includes(PACKS[0].repo) && seen.includes(`${PACKS[3].repo}@2.0`) && seen.includes(later_pack.repo));
      assert.ok(server.calls.some((c) => c.path === '/v1/downloads'), 'another tab asks in the background');
      await first.context.close();

      server.state.packs.push(pack('beamster/Brand-New-Sushi-3bpw'));
      const later = await open(browser, server, '/settings', { storage: { 'sushi-seen-packs': JSON.stringify(seen) } });
      await later.page.waitForSelector('#navModelsDot', { state: 'visible', timeout: 8000 });
      assert.equal(await later.page.textContent('#navModels'), 'Models', 'the dot adds no text');
      assert.equal(await later.page.getAttribute('#navModels', 'aria-label'), 'Models, new packs');
      await later.page.click('#navModels');
      assert.equal(await later.page.isVisible('#modelsTabDot'), true, 'the switch shows where the new pack is');
      assert.equal(await later.page.isVisible('#navModelsDot'), true, 'the installed list alone does not clear it');
      await later.page.click('#modelsTabAvailable');
      await later.page.waitForSelector('#navModelsDot', { state: 'hidden' });
      assert.equal(await later.page.isVisible('#modelsTabDot'), false);
      assert.equal(await later.page.getAttribute('#navModels', 'aria-label'), 'Models');
      assert.ok((await seenNow(later.page)).includes('beamster/Brand-New-Sushi-3bpw'));
      await later.context.close();

      server.state.packs[3].update = { current: '1.0', latest: '3.0' };
      const upd = await open(browser, server, '/chat', { storage: { 'sushi-seen-packs': JSON.stringify([...seen, 'beamster/Brand-New-Sushi-3bpw']) } });
      await upd.page.waitForSelector('#navModelsDot', { state: 'visible', timeout: 8000 });
      await upd.context.close();

      // Under --no-update-check another tab asks once (no Hugging Face behind it), shows no dot and stops asking.
      const quiet = makeServer();
      quiet.state.updateCheck = false;
      quiet.state.packs.push(pack('beamster/Unseen-Sushi-1bpw'));
      const off = await open(browser, quiet, '/chat', { storage: { 'sushi-seen-packs': JSON.stringify(seen) } });
      await off.page.waitForTimeout(4000);
      assert.equal(await off.page.isVisible('#navModelsDot'), false);
      assert.equal(quiet.calls.filter((c) => c.path === '/v1/downloads').length, 1);
      await off.context.close();
    }

    {
      // Offline: the known packs without sizes and a note, never an error wall.
      const server = makeServer();
      server.state.catalog = 'offline';
      server.state.packs = server.state.packs.map((p) => ({ ...p, size_bytes: null }));
      const { page, context } = await open(browser, server, '/downloads');
      await page.waitForSelector('#dlNote:not([hidden])');
      assert.match(await page.textContent('#dlNote'), /can't be reached/);
      assert.match(await page.textContent(row(PACKS[0].repo)), /size unknown/);
      assert.equal(await page.isVisible('#dlAlert'), false);
      await context.close();
    }

    for (const viewport of [{ width: 320, height: 640 }, PHONE, { width: 1440, height: 900 }]) {
      const server = makeServer();
      server.state.jobs = [{ repo: PACKS[0].repo, dest: '/m/q', phase: 'downloading', files: 12, file_index: 3, total_bytes: 40 * GiB, done_bytes: 10 * GiB, file: { path: 'model-00003-of-00012.safetensors', size: 4 * GiB, done: GiB }, speed_bps: 100 * 2 ** 20, eta_s: 307, message: null }];
      const { page, context } = await open(browser, server, '/downloads', { viewport });
      await page.waitForSelector(`${row(PACKS[0].repo)} .prog:not([hidden])`);
      await page.waitForTimeout(300);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth), viewport.width, `Available fits ${viewport.width} px`);
      const spill = await page.evaluate(() => [...document.querySelectorAll('.dl-row')].filter((r) => r.scrollWidth > r.clientWidth + 1).length);
      assert.equal(spill, 0, `no row overflows at ${viewport.width} px`);
      await context.close();
    }
    console.log('Web UI downloads: passed');
  } finally {
    await browser.close();
  }
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
