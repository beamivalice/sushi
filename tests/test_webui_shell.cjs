// sushi-test: tier=hermetic needs=node,playwright
// The page's shell in a real headless browser against a stubbed server: the top bar and the phone layout, the Compact and
// Advanced switch, the shared page frame, the dashboard's offline and metrics-off states, the chat composer and drawers.
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

function makeServer({ metrics = 200, loaded = true, down = false, apiKey = null } = {}) {
  const calls = [];
  const state = { loaded };
  const json = (body, status = 200) => ({ status, contentType: 'application/json', body: JSON.stringify(body) });
  async function handle(route) {
    const request = route.request(), url = new URL(request.url()), p = url.pathname;
    if (/^\/(chat|dashboard|models|settings|downloads)?$/.test(p)) return route.fulfill({ status: 200, contentType: 'text/html; charset=utf-8', body: html });
    calls.push({ path: p, method: request.method() });
    if (apiKey && request.headers().authorization !== `Bearer ${apiKey}`) return route.fulfill(json({ error: { message: 'bad key' } }, 401));
    if (down) return route.abort();
    switch (p) {
      case '/v1/models': return route.fulfill(json({ data: [{ id: 'Chat-A', state: state.loaded ? 'ready' : 'unloaded', loaded: state.loaded, capabilities: ['chat'], context_length: 8192, bytes_on_disk: 2 ** 30, input_modalities: ['text'] }] }));
      case '/v1/load-model': state.loaded = true; return route.fulfill(json({ ok: true }));
      case '/props': return route.fulfill(json({ settings: { version: '9.9.9', mtp: {}, prefix_cache: {} }, memory: { available_bytes: 5e9, total_bytes: 128 * 2 ** 30 }, batching: {} }));
      case '/v1/tools': return route.fulfill(json({ edit_default: false }));
      case '/v1/settings': return route.fulfill(json({ file: '/h/f.json', model_keys: [], applies: 'x', server: { models_dir: { default: '/d', boot_roots: ['/m'], file: null, flag: '', locked: false }, prefix_cache_disk_gb: { file: null, locked: false, effective_bytes: 2 ** 30 } }, models: [] }));
      case '/v1/prefix-cache': return route.fulfill(json({ root: '/c', volume: null, total_bytes: 0, namespaces: [] }));
      case '/v1/downloads': return route.fulfill(json({ ram_bytes: 64 * 2 ** 30, ram_gb: 64, dest: { path: '/m', source: 'default', locked: false, volume: 'Disk', free_bytes: 2 ** 40, total_bytes: 2 ** 41 }, catalog: 'ready', refreshing: false, packs: [{ repo: 'beamster/A-Sushi-2bpw', name: 'a', tag: '2bpw', size_bytes: 2 ** 34, tier_gb: 48, fits: true, recommended: true, installed: false, path: null, partial_bytes: null, update: null, launch: null }], jobs: [] }));
      case '/metrics.json': return metrics === 200 ? route.fulfill(json({ counters: {}, gauges: {}, histograms: {}, sessions: [] })) : route.fulfill(json({ error: { message: 'off' } }, metrics));
      case '/metrics/enable': return route.fulfill(json({ metrics: true }));
      default: return route.fulfill(json({}, 404));
    }
  }
  return { handle, calls, state };
}

async function open(browser, server, urlPath, { viewport = { width: 1440, height: 900 }, mode = 'compact', scheme = 'dark', clock = false, storage = {}, platform = null } = {}) {
  const context = await browser.newContext({ viewport, colorScheme: scheme });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.route(`${ORIGIN}/**`, server.handle);
  await page.addInitScript(([m, seed]) => {
    if (sessionStorage.getItem('__seeded')) return;
    sessionStorage.setItem('__seeded', '1');
    try { localStorage.setItem('sushi-ui-mode', m); for (const [k, v] of Object.entries(seed)) localStorage.setItem(k, v); } catch { /* storage off */ }
  }, [mode, storage]);
  if (platform) await page.addInitScript((p) => Object.defineProperty(Navigator.prototype, 'platform', { get: () => p }), platform);
  if (clock) await page.clock.install();
  await page.goto(`${ORIGIN}${urlPath}`);
  return { page, context, errors };
}

async function main() {
  const playwright = loadPlaywright();
  if (!playwright) { console.log('Web UI shell: skipped (Playwright is not installed)'); return; }
  let browser;
  try { browser = await playwright.chromium.launch(); } catch (error) {
    console.log(`Web UI shell: skipped (no Chromium: ${String(error.message).split('\n')[0]})`);
    return;
  }
  try {
    {
      // The phone: nothing scrolls sideways on any tab, and the More menu holds the view and theme switches.
      const server = makeServer();
      for (const view of ['/chat', '/dashboard', '/models', '/settings', '/downloads']) {
        const { page, context } = await open(browser, server, view, { viewport: PHONE, mode: 'advanced' });
        await page.waitForTimeout(500);
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth), 390, `${view} fits 390 px`);
        assert.equal(await page.isVisible('#serverDotLight'), true, 'the status dot stays');
        await context.close();
      }
      const { page, context } = await open(browser, server, '/chat', { viewport: PHONE });
      assert.equal(await page.isVisible('#modeToggle'), false, 'the segmented switch lives in the menu on a phone');
      await page.click('#moreButton');
      await page.click('#themeMenu [data-theme-mode="day"]');
      assert.equal(await page.evaluate(() => document.documentElement.dataset.theme), 'day');
      await page.click('#modeToggleMenu [data-mode="advanced"]');
      assert.equal(await page.evaluate(() => document.body.dataset.mode), 'advanced');
      await context.close();
    }

    {
      // One switch for the view: it names the current mode, and both Compact and Advanced reach every tab.
      const server = makeServer();
      const { page, context } = await open(browser, server, '/models');
      assert.equal(await page.getAttribute('#modeToggle [data-mode="compact"]', 'aria-pressed'), 'true');
      await page.click('#modeToggle [data-mode="advanced"]');
      assert.equal(await page.getAttribute('#modeToggle [data-mode="advanced"]', 'aria-pressed'), 'true');
      assert.equal(await page.getAttribute('#modeToggle [data-mode="compact"]', 'aria-pressed'), 'false');
      assert.equal(await page.evaluate(() => localStorage.getItem('sushi-ui-mode')), 'advanced');
      await page.click('#navChat');
      assert.equal(await page.isVisible('#modeToggle'), true, 'the switch shows on every tab');
      await page.waitForSelector('#perfStatus', { state: 'visible', timeout: 3000 });
      await page.click('#modeToggle [data-mode="compact"]');
      await page.waitForSelector('#perfStatus', { state: 'hidden', timeout: 3000 });
      assert.match(await page.getAttribute('meta[name="theme-color"]', 'content'), /^#|^rgb/);
      await context.close();
    }

    for (const viewport of [{ width: 1440, height: 900 }, { width: 2000, height: 1050 }]) {
      // Heading and content share one left edge on the tabs that use the page frame.
      const server = makeServer();
      for (const view of ['/dashboard', '/models', '/settings', '/downloads']) {
        const { page, context } = await open(browser, server, view, { viewport });
        await page.waitForSelector(`#${{ '/dashboard': 'dashPanel', '/models': 'modelsPanel', '/settings': 'settingsPanel', '/downloads': 'modelsPanel' }[view]} .page-head h1`);
        await page.waitForTimeout(400);
        const edges = await page.evaluate(() => {
          const panel = [...document.querySelectorAll('.panel')].find((p) => !p.hidden);
          const h1 = panel.querySelector('h1').getBoundingClientRect().left;
          const card = [...panel.querySelectorAll('.card')].find((c) => c.offsetParent !== null).getBoundingClientRect().left;
          return { h1, card };
        });
        assert.ok(Math.abs(edges.h1 - edges.card) <= 1, `${view} at ${viewport.width}: heading ${edges.h1} and first card ${edges.card} share an edge`);
        await context.close();
      }
    }

    {
      // The type scale: the sizes on screen are the seven of the design system (plus chat markdown headings).
      const allowed = new Set([11, 12, 13, 15, 17, 22, 28]);
      const server = makeServer();
      for (const [view, mode] of [['/chat', 'advanced'], ['/dashboard', 'advanced'], ['/dashboard', 'compact'], ['/models', 'advanced'], ['/settings', 'advanced'], ['/downloads', 'advanced']]) {
        const { page, context } = await open(browser, server, view, { mode });
        await page.waitForTimeout(700);
        const sizes = await page.evaluate(() => {
          const found = new Map();
          for (const e of document.querySelectorAll('body *')) {
            if (!e.offsetParent && getComputedStyle(e).position !== 'fixed') continue;
            if (![...e.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim())) continue;
            if (e.closest('script, style, dialog:not([open]), [popover]:not(:popover-open)')) continue;
            const size = parseFloat(getComputedStyle(e).fontSize);
            found.set(size, e.className || e.tagName);
          }
          return [...found];
        });
        for (const [size, who] of sizes) assert.ok(allowed.has(size), `${view} ${mode}: ${size}px on ${who}`);
        await context.close();
      }
    }

    {
      // Dashboard states: offline says so, metrics off is an explained card with the way out, Pause is not an error.
      const { page, context } = await open(browser, makeServer({ down: true }), '/dashboard');
      await page.waitForFunction(() => document.getElementById('dashCmState').textContent === 'Offline', null, { timeout: 15000 });
      assert.equal(await page.textContent('#serverDotLabel'), 'Offline');
      await context.close();
    }
    {
      const server = makeServer({ metrics: 503 });
      const { page, context } = await open(browser, server, '/dashboard');
      await page.waitForSelector('#dashEnableMetrics');
      assert.match(await page.textContent('#dashOff'), /not collecting live statistics/);
      assert.equal(await page.isVisible('.tiles-cmp'), false, 'no empty tiles under the explanation');
      assert.equal(await page.isVisible('.cm-strip'), true, 'the strip stays');
      assert.equal(await page.evaluate(() => document.getElementById('dashDot').className.includes('err')), false, 'metrics off is not an error');
      await context.close();
    }
    {
      const { page, context } = await open(browser, makeServer(), '/dashboard');
      await page.waitForFunction(() => /Live/.test(document.getElementById('dashStatus').textContent));
      await page.click('#dashPause');
      assert.match(await page.getAttribute('#dashDot', 'class'), /\boff\b/, 'Paused is a grey dot');
      assert.equal(await page.textContent('#dashStatus'), 'Paused');
      await context.close();
    }
    {
      const { page, context } = await open(browser, makeServer(), '/dashboard', { viewport: PHONE });
      await page.waitForFunction(() => /Live/.test(document.getElementById('dashStatus').textContent));
      const rows = await page.evaluate(() => new Set([...document.querySelectorAll('.tiles-cmp .tile')].map((t) => Math.round(t.getBoundingClientRect().y))).size);
      assert.equal(rows, 3, 'five Compact tiles take three rows on a phone');
      await context.close();
    }

    {
      // The chat on a phone: chips keep their icons, the drawers have a title, a close button and a scrim.
      const { page, context } = await open(browser, makeServer(), '/chat', { viewport: PHONE });
      await page.waitForSelector('#modelSelect option', { state: 'attached' });
      const chips = await page.evaluate(() => [...document.querySelectorAll('.composer-row .chip-toggle')].filter((c) => c.offsetParent).map((c) => ({ id: c.id, w: c.getBoundingClientRect().width, clipped: c.scrollWidth > c.clientWidth })));
      assert.ok(chips.every((c) => !c.clipped && c.w >= 28), `no clipped chip (${JSON.stringify(chips)})`);
      assert.equal(await page.evaluate(() => document.getElementById('composerEffortName').getBoundingClientRect().width > 0), true, 'the effort word always shows');
      assert.equal(await page.evaluate(() => document.querySelector('.composer-note')), null, 'the hint moved into the placeholder');
      await page.click('#rightbarToggle');
      assert.equal(await page.isVisible('#rightbarClose'), true);
      assert.equal(await page.isVisible('#chatScrim'), true);
      assert.equal(await page.evaluate(() => document.activeElement.id), 'rightbarClose', 'focus moves into the drawer');
      await page.click('#rightbarClose');
      assert.equal(await page.evaluate(() => document.activeElement.id), 'rightbarToggle', 'and back to its button');
      await page.click('#sidebarToggle');
      await page.waitForTimeout(300);
      await page.mouse.click(385, 400);
      assert.equal(await page.evaluate(() => document.getElementById('chatPanel').classList.contains('left-open')), false, 'a tap on the scrim closes the drawer');
      await context.close();
    }

    {
      // A model that is not loaded can be loaded from the chat, once.
      const server = makeServer({ loaded: false });
      const { page, context } = await open(browser, server, '/chat');
      await page.waitForSelector('#modelStatus button');
      await page.dblclick('#modelStatus button');
      await page.waitForFunction(() => /Loaded/.test(document.getElementById('toastHost').textContent));
      assert.equal(server.calls.filter((c) => c.path === '/v1/load-model').length, 1);
      await page.waitForFunction(() => !document.querySelector('#modelStatus button'));
      await context.close();
    }

    {
      // The landing view is the Dashboard, and the top bar lists Dashboard, Chat, Models, Settings, each with its icon.
      const { page, context } = await open(browser, makeServer(), '/');
      await page.waitForSelector('#dashPanel:not([hidden])');
      assert.equal(await page.getAttribute('#navDashboard', 'aria-current'), 'page', 'the root shows the Dashboard');
      assert.equal(await page.evaluate(() => document.getElementById('chatPanel').inert), true);
      assert.deepEqual(await page.$$eval('#viewNav a', (links) => links.map((a) => a.textContent)), ['Dashboard', 'Chat', 'Models', 'Settings']);
      const shown = (p) => p.$$eval('#viewNav a', (links) => links.map((a) => [a.getAttribute('aria-label'), !!a.querySelector('.nav-i')?.getBoundingClientRect().width, !!a.querySelector('.nav-l')?.getBoundingClientRect().width]));
      assert.deepEqual(await shown(page), [['Dashboard', true, true], ['Chat', true, true], ['Models', true, true], ['Settings', true, true]], 'icon and label on a wide screen');
      await page.click('#navChat');
      assert.equal(new URL(page.url()).pathname, '/chat');
      await page.goBack();
      assert.equal(await page.getAttribute('#navDashboard', 'aria-current'), 'page', 'Back from the chat returns to the Dashboard');
      await context.close();
      const phone = await open(browser, makeServer(), '/', { viewport: PHONE });
      assert.deepEqual(await shown(phone.page), [['Dashboard', true, false], ['Chat', true, false], ['Models', true, false], ['Settings', true, false]], 'same order on a phone, as icons named by their labels');
      await phone.context.close();
    }

    {
      // Cmd+1..4 (Ctrl+1..4 off a Mac) switch Dashboard, Chat, Models, Settings, even from the chat prompt, and each link names its key.
      const { page, context } = await open(browser, makeServer(), '/');
      await page.waitForSelector('#dashPanel:not([hidden])');
      const at = () => new URL(page.url()).pathname;
      await page.keyboard.press('Meta+2');
      assert.equal(at(), '/chat', 'Cmd+2 opens the chat');
      await page.click('#promptInput');
      await page.keyboard.type('half a thought');
      await page.keyboard.press('Meta+3');
      assert.equal(at(), '/models', 'Cmd+3 works while typing in the prompt');
      await page.keyboard.press('Control+4');
      assert.equal(at(), '/settings', 'Ctrl+4 works too');
      await page.keyboard.press('Meta+5');
      assert.equal(at(), '/settings', 'Cmd+5 is no tab');
      await page.keyboard.press('Meta+1');
      assert.equal(at(), '/dashboard', 'Cmd+1 returns to the Dashboard');
      await page.keyboard.press('Meta+Shift+2');
      assert.equal(at(), '/dashboard', 'a shifted digit is not a tab switch');
      assert.deepEqual(await page.$$eval('#viewNav a', (links) => links.map((a) => a.getAttribute('aria-keyshortcuts'))), ['Meta+1', 'Meta+2', 'Meta+3', 'Meta+4']);
      await page.keyboard.press('Meta+2');
      assert.equal(await page.inputValue('#promptInput'), 'half a thought', 'the prompt survives the tab round trip');
      await context.close();
      const pc = await open(browser, makeServer(), '/', { platform: 'Win32' });
      assert.deepEqual(await pc.page.$$eval('#viewNav a', (links) => links.map((a) => a.getAttribute('aria-keyshortcuts'))), ['Control+1', 'Control+2', 'Control+3', 'Control+4'], 'off a Mac the links name the key that works');
      await pc.context.close();
      const keyed = await open(browser, makeServer({ apiKey: 'sekret' }), '/chat');
      await keyed.page.waitForSelector('#keyOverlay:not([hidden])');
      await keyed.page.keyboard.press('Meta+1');
      assert.equal(new URL(keyed.page.url()).pathname, '/chat', 'the key overlay keeps the shortcuts from switching the page behind it');
      await keyed.context.close();
    }

    {
      // With --api-key and no stored key, the overlay owns the keyboard: nothing typed or tabbed reaches the page behind it.
      const { page, context } = await open(browser, makeServer({ apiKey: 'sekret' }), '/chat');
      await page.waitForSelector('#keyOverlay:not([hidden])');
      await page.waitForTimeout(500);
      assert.equal(await page.evaluate(() => document.activeElement.id), 'keyInput', 'the key box has the cursor');
      await page.keyboard.type('sek');
      assert.equal(await page.inputValue('#promptInput'), '', 'nothing typed lands in the chat prompt');
      assert.equal(await page.inputValue('#keyInput'), 'sek');
      for (let i = 0; i < 8; i++) {
        await page.keyboard.press('Tab');
        assert.equal(await page.evaluate(() => !!document.activeElement.closest('#appShell')), false, 'Tab stays out of the page behind the overlay');
      }
      await page.fill('#keyInput', 'sekret');
      await page.keyboard.press('Enter');
      await page.waitForSelector('#keyOverlay', { state: 'hidden' });
      await page.waitForFunction(() => document.getElementById('modelSelect').options.length === 1);
      assert.equal(await page.evaluate(() => document.getElementById('appShell').inert), false, 'the page works again after unlocking');
      await context.close();
    }

    {
      // The ⋯ menu carries Forget the API key when a key is stored.
      const { page, context } = await open(browser, makeServer({ apiKey: 'sekret' }), '/chat', { viewport: PHONE, storage: { 'sushi-api-key': 'sekret' } });
      await page.waitForSelector('#modelSelect option', { state: 'attached' });
      await page.click('#moreButton');
      const forget = page.locator('#moreMenu button', { hasText: 'Forget' });
      assert.equal(await forget.count(), 1, 'the menu offers Forget the API key');
      await forget.click();
      await page.waitForSelector('#keyOverlay:not([hidden])');
      assert.equal(await page.evaluate(() => localStorage.getItem('sushi-api-key')), null, 'the stored key is gone');
      await context.close();
      const open2 = await open(browser, makeServer(), '/chat', { viewport: PHONE });
      await open2.page.click('#moreButton');
      assert.equal(await open2.page.locator('#moreMenu button', { hasText: 'Forget' }).isVisible(), false, 'no key, nothing to forget');
      await open2.context.close();
    }

    {
      // Only the current tab is marked current; the others carry no aria-current at all.
      const { page, context } = await open(browser, makeServer(), '/chat');
      await page.click('#navDashboard');
      assert.equal(await page.evaluate(() => document.getElementById('navChat').hasAttribute('aria-current')), false);
      assert.equal(await page.getAttribute('#navDashboard', 'aria-current'), 'page');
      await context.close();
    }

    {
      // Phone drawers: closed means out of the tab order and the accessibility tree, and Escape gives focus back.
      const chat = { id: 'a', title: 'First', updated: Date.now(), messages: [{ role: 'user', content: 'hi' }] };
      const { page, context } = await open(browser, makeServer(), '/chat', { viewport: PHONE, storage: { 'sushi-chats': JSON.stringify([chat]) } });
      await page.waitForSelector('#modelSelect option', { state: 'attached' });
      await page.waitForTimeout(400);
      const reachable = await page.evaluate(() => [...document.querySelectorAll('#chatSidebar, #chatRightbar')].flatMap((drawer) => [...drawer.querySelectorAll('button, input, select, textarea, summary, [tabindex="0"]')].filter((n) => n.checkVisibility({ visibilityProperty: true })).map((n) => n.id || n.className)));
      assert.deepEqual(reachable, [], 'a closed drawer has no focusable control');
      await page.click('#sidebarToggle');
      await page.waitForTimeout(300);
      await page.keyboard.press('Escape');
      assert.equal(await page.evaluate(() => document.activeElement.id), 'sidebarToggle', 'Escape returns focus to the button that opened the drawer');
      await page.click('#sidebarToggle');
      await page.waitForTimeout(300);
      await page.hover('.conv-item');
      await page.click('[aria-label="Rename conversation"]');
      assert.equal(await page.getByRole('textbox', { name: 'Conversation title' }).count(), 1, 'the rename box is named');
      await page.keyboard.press('Escape');
      assert.equal(await page.evaluate(() => document.getElementById('chatPanel').classList.contains('left-open')), true, 'Escape while renaming only ends the rename');
      assert.equal(await page.$('.conv-item input'), null);
      await context.close();
    }

    {
      // The top bar fits a 320 px phone, and so does the Models head with its Installed | Available switch.
      const { page, context } = await open(browser, makeServer(), '/chat', { viewport: { width: 320, height: 640 } });
      await page.waitForSelector('#modelSelect option', { state: 'attached' });
      await page.waitForTimeout(300);
      const fit = await page.evaluate(() => ({ doc: document.documentElement.scrollWidth, more: document.getElementById('moreButton').getBoundingClientRect().right, nav: [...document.querySelectorAll('#viewNav a')].map((a) => a.getBoundingClientRect().right <= 320) }));
      assert.ok(fit.doc <= 320 && fit.more <= 320 && fit.nav.every(Boolean), `top bar fits 320 px (${JSON.stringify(fit)})`);
      await page.click('#navModels');
      await page.click('#modelsTabAvailable');
      await page.waitForSelector('.dl-row');
      const head = await page.evaluate(() => ({ doc: document.documentElement.scrollWidth, tabs: document.getElementById('modelsTabs').getBoundingClientRect().right, spill: [...document.querySelectorAll('#modelsPanel .page-head *')].filter((n) => n.offsetParent && n.getBoundingClientRect().right > 320.5).map((n) => n.id || n.className) }));
      assert.ok(head.doc <= 320 && head.tabs <= 320 && head.spill.length === 0, `Models head fits 320 px (${JSON.stringify(head)})`);
      await context.close();
    }

    {
      // The composer row, Tools on, between the 881 px layout and wide: nothing past the end of the row, no clipped chip.
      for (const width of [881, 900, 960]) {
        const { page, context } = await open(browser, makeServer(), '/chat', { viewport: { width, height: 800 } });
        await page.waitForSelector('#modelSelect option', { state: 'attached' });
        await page.click('#toolsButton');
        const fit = await page.evaluate(() => {
          const row = document.querySelector('.composer-row'), edge = row.getBoundingClientRect().right;
          const parts = [...row.querySelectorAll('.chip-toggle, #sendButton')].filter((n) => n.offsetParent);
          return { spill: parts.filter((n) => n.getBoundingClientRect().right > edge + 0.5).map((n) => n.id), clipped: parts.filter((n) => n.scrollWidth > n.clientWidth).map((n) => n.id) };
        });
        assert.deepEqual(fit, { spill: [], clipped: [] }, `composer row at ${width} px`);
        await context.close();
      }
    }
    console.log('Web UI shell: passed');
  } finally {
    await browser.close();
  }
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
