// sushi-test: tier=hermetic needs=node,playwright
// Theme 2.0 in a real headless browser against a stubbed server: the theme choice survives a reload, salmon text and
// the muted text keep WCAG AA in both themes, a [data-tip] tooltip opens on keyboard focus and on hover without
// covering its anchor, reduced motion turns every transition off, and one logo serves every mark and the favicon.
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

function handler({ down = false } = {}) {
  const json = (body, status = 200) => ({ status, contentType: 'application/json', body: JSON.stringify(body) });
  return async (route) => {
    const p = new URL(route.request().url()).pathname;
    if (/^\/(chat|dashboard|models|settings|downloads)?$/.test(p)) return route.fulfill({ status: 200, contentType: 'text/html; charset=utf-8', body: html });
    if (down) return route.abort();
    switch (p) {
      case '/v1/models': return route.fulfill(json({ data: [{ id: 'Chat-A', state: 'ready', loaded: true, capabilities: ['chat'], context_length: 8192, bytes_on_disk: 2 ** 30, input_modalities: ['text'] }] }));
      case '/props': return route.fulfill(json({ settings: { version: '9.9.9', mtp: {}, prefix_cache: {} }, memory: { available_bytes: 5e9, total_bytes: 128 * 2 ** 30 }, batching: {} }));
      case '/metrics.json': return route.fulfill(json({ counters: {}, gauges: {}, histograms: {}, sessions: [] }));
      default: return route.fulfill(json({}, 404));
    }
  };
}

async function open(browser, urlPath, { scheme = 'light', reducedMotion = 'no-preference', down = false } = {}) {
  const context = await browser.newContext({ viewport: { width: 1280, height: 860 }, colorScheme: scheme, reducedMotion });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.route(`${ORIGIN}/**`, handler({ down }));
  await page.goto(`${ORIGIN}${urlPath}`);
  return { page, context, errors };
}

// WCAG contrast of two tokens, each resolved by the page itself.
const contrastIn = (page, pairs) => page.evaluate((list) => {
  const probe = document.body.appendChild(document.createElement('span'));
  const rgb = (token) => { probe.style.color = `var(${token})`; return getComputedStyle(probe).color.match(/[\d.]+/g).slice(0, 3).map(Number); };
  const lum = (c) => { const [r, g, b] = c.map((v) => { v /= 255; return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; }); return 0.2126 * r + 0.7152 * g + 0.0722 * b; };
  const out = list.map(([fg, bg]) => { const [a, b] = [lum(rgb(fg)), lum(rgb(bg))].sort((x, y) => y - x); return [fg, bg, (a + 0.05) / (b + 0.05)]; });
  probe.remove();
  return out;
}, pairs);

async function tabTo(page, selector) {
  for (let i = 0; i < 60; i++) {
    await page.keyboard.press('Tab');
    if (await page.evaluate((s) => document.activeElement?.matches(s), selector)) return;
  }
  throw new Error(`Tab never reached ${selector}`);
}
const tipState = (page) => page.evaluate(() => {
  const tip = document.getElementById('tip'), anchor = document.querySelector('[aria-describedby="tip"]');
  const r = tip.getBoundingClientRect(), a = anchor?.getBoundingClientRect();
  return {
    shown: tip.classList.contains('show') && getComputedStyle(tip).visibility === 'visible', text: tip.firstChild?.textContent ?? '', anchorTip: anchor?.dataset.tip ?? null,
    overlaps: !!a && r.left < a.right && r.right > a.left && r.top < a.bottom && r.bottom > a.top,
    inside: r.left >= 0 && r.top >= 0 && r.right <= innerWidth && r.bottom <= innerHeight,
    box: { x: r.left + r.width / 2, y: r.top + r.height / 2 },
  };
});

async function main() {
  const playwright = loadPlaywright();
  if (!playwright) { console.log('Web UI theme: skipped (Playwright is not installed)'); return; }
  let browser;
  try { browser = await playwright.chromium.launch(); } catch (error) {
    console.log(`Web UI theme: skipped (no Chromium: ${String(error.message).split('\n')[0]})`);
    return;
  }
  try {
    {
      // The toggle steps System, Light, Dark; the choice and its colours survive a reload.
      const { page, context, errors } = await open(browser, '/dashboard', { scheme: 'light' });
      await page.waitForSelector('#themeToggle');
      assert.equal(await page.evaluate(() => document.documentElement.dataset.theme), 'system');
      const lightBg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
      await page.click('#themeToggle');
      await page.click('#themeToggle');
      assert.equal(await page.evaluate(() => document.documentElement.dataset.theme), 'night');
      await page.reload();
      await page.waitForSelector('#themeToggle');
      assert.equal(await page.evaluate(() => document.documentElement.dataset.theme), 'night', 'the choice survives a reload');
      assert.equal(await page.evaluate(() => document.documentElement.classList.contains('theme-fade')), false, 'a reload does not fade');
      const darkBg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
      assert.notEqual(darkBg, lightBg, 'Dark paints a dark page under a light system scheme');
      assert.match(await page.getAttribute('#themeToggle', 'data-tip'), /Theme: Dark\. Click for System/);
      assert.deepEqual(errors, []);
      await context.close();
    }

    for (const scheme of ['light', 'dark']) {
      // Salmon text and muted text stay AA (4.5:1) on every surface they sit on; ink on the salmon fill too.
      const { page, context } = await open(browser, '/dashboard', { scheme });
      await page.waitForSelector('#themeToggle');
      const pairs = [['--accent', '--surface'], ['--accent', '--surface-2'], ['--accent', '--bg'], ['--accent', '--accent-soft'],
        ['--on-accent-fill', '--accent-fill'], ['--on-accent-fill', '--accent-fill-hover'], ['--on-accent', '--accent'],
        ['--text-2', '--surface-2'], ['--text-3', '--surface'], ['--text-3', '--surface-2'], ['--tip-muted', '--tip-bg']];
      for (const [fg, bg, ratio] of await contrastIn(page, pairs)) assert.ok(ratio >= 4.5, `${scheme}: ${fg} on ${bg} is ${ratio.toFixed(2)}:1`);
      for (const [fg, bg, ratio] of await contrastIn(page, [['--focus', '--surface'], ['--focus', '--bg'], ['--c1', '--surface']])) {
        assert.ok(ratio >= 3, `${scheme}: ${fg} on ${bg} is ${ratio.toFixed(2)}:1, under the 3:1 a focus ring or chart line needs`);
      }
      await context.close();
    }

    {
      // Keyboard focus opens the tooltip at once, beside its anchor and inside the window; Escape closes it and keeps focus.
      const { page, context, errors } = await open(browser, '/dashboard');
      await page.waitForSelector('.tiles-cmp .tile[data-tip]');
      await tabTo(page, '.tiles-cmp .tile[data-tip]');
      const shown = await tipState(page);
      assert.equal(shown.shown, true, 'focus shows the tooltip');
      assert.equal(shown.text, shown.anchorTip, 'it reads its anchor\'s text and the anchor points to it');
      assert.equal(shown.overlaps, false, 'it never covers its anchor');
      assert.equal(shown.inside, true, 'it stays inside the window');
      await page.keyboard.press('Escape');
      assert.equal((await tipState(page)).shown, false, 'Escape hides it');
      assert.equal(await page.evaluate(() => document.activeElement.matches('.tiles-cmp .tile[data-tip]')), true, 'focus stays');
      await page.keyboard.press('Tab');
      assert.equal(await page.evaluate(() => document.querySelectorAll('[aria-describedby="tip"]').length), 1, 'the next anchor takes it over');

      // Hover: shown after a pause, the pointer may move onto it, and leaving both hides it.
      await page.keyboard.press('Escape');
      await page.hover('.tiles-cmp .tile[data-tip] >> nth=1');
      await page.waitForFunction(() => document.getElementById('tip').classList.contains('show'), null, { timeout: 3000 });
      const { box } = await tipState(page);
      await page.mouse.move(box.x, box.y, { steps: 4 });
      await page.waitForTimeout(400);
      assert.equal((await tipState(page)).shown, true, 'the pointer can rest on the tooltip');
      await page.mouse.move(2, 850);
      await page.waitForFunction(() => !document.getElementById('tip').classList.contains('show'), null, { timeout: 3000 });
      assert.deepEqual(errors, []);
      await context.close();
    }

    for (const scheme of ['light', 'dark']) {
      // A data-term adds a "Technical name" line, a child element with the term as text, never markup; no data-term, no line.
      const { page, context, errors } = await open(browser, '/dashboard', { scheme });
      await page.waitForSelector('.tiles-cmp .tile[data-tip]');
      await page.evaluate(() => {
        const add = (id, term) => {
          const a = document.body.appendChild(Object.assign(document.createElement('button'), { id, textContent: id }));
          a.style.cssText = 'position:fixed;left:40px;top:300px';
          a.dataset.tip = `${id} tip`;
          if (term != null) a.dataset.term = term;
        };
        add('withTerm', 'decode (tok/s)');
        add('markupTerm', '<b>bold</b>');
        add('plainTip');
      });
      const probe = async (id) => {
        await page.focus('#plainTip');
        await page.keyboard.press('Escape');
        await page.evaluate((i) => { document.getElementById(i).blur(); }, id);
        await page.keyboard.press('Tab');
        await page.evaluate((i) => document.getElementById(i).focus(), id);
        await page.waitForFunction((i) => document.querySelector('[aria-describedby="tip"]')?.id === i, id);
        return page.evaluate(() => {
          const tip = document.getElementById('tip'), line = tip.querySelector('.tip-term');
          const cs = line && getComputedStyle(line), bg = getComputedStyle(tip).backgroundColor;
          return { text: tip.firstChild.textContent, line: line?.textContent ?? null, tag: line?.tagName ?? null, isChild: line?.parentElement === tip,
            markup: tip.querySelectorAll('b').length, size: cs && parseFloat(cs.fontSize), tipSize: parseFloat(getComputedStyle(tip).fontSize),
            color: cs?.color, bg, term: tip.querySelector('.tip-term-v')?.textContent ?? null };
        });
      };
      const a = await probe('withTerm');
      assert.equal(a.text, 'withTerm tip', `${scheme}: the tip text stays plain`);
      assert.equal(a.line, 'Technical name: decode (tok/s)', `${scheme}: the term line follows the tip`);
      assert.equal(a.term, 'decode (tok/s)');
      assert.equal(a.isChild, true, `${scheme}: it is a child element of the tooltip`);
      assert.ok(a.size < a.tipSize, `${scheme}: the term line is smaller than the tip text`);
      const rgb = (c) => c.match(/[\d.]+/g).slice(0, 3).map(Number);
      const lum = (c) => { const [r, g, b] = rgb(c).map((v) => { v /= 255; return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; }); return 0.2126 * r + 0.7152 * g + 0.0722 * b; };
      const [hi, lo] = [lum(a.color), lum(a.bg)].sort((x, y) => y - x);
      assert.ok((hi + 0.05) / (lo + 0.05) >= 4.5, `${scheme}: the term line keeps AA on the tooltip surface`);
      const m = await probe('markupTerm');
      assert.equal(m.markup, 0, `${scheme}: a term is text, not markup`);
      assert.equal(m.term, '<b>bold</b>');
      const p = await probe('plainTip');
      assert.equal(p.line, null, `${scheme}: no data-term, no term line`);
      assert.equal(p.text, 'plainTip tip');
      assert.deepEqual(errors, []);
      await context.close();
    }

    {
      // In the chat's settings column a field's tip sits beside the column and covers none of its fields.
      const { page, context } = await open(browser, '/chat');
      await page.waitForSelector('#optTopP');
      await page.focus('#optTopK');
      await page.keyboard.press('Shift+Tab');
      assert.equal((await tipState(page)).shown, true, 'focus on Top-p shows its tip');
      const covered = await page.evaluate(() => {
        const t = document.getElementById('tip').getBoundingClientRect();
        return [...document.querySelectorAll('#chatRightbar input, #chatRightbar select, #chatRightbar textarea')].filter((f) => {
          const r = f.getBoundingClientRect();
          return r.width && t.left < r.right && t.right > r.left && t.top < r.bottom && t.bottom > r.top;
        }).map((f) => f.id);
      });
      assert.deepEqual(covered, [], 'no field under the tip');
      await context.close();
    }

    for (const reducedMotion of ['no-preference', 'reduce']) {
      // Under reduced motion nothing moves: no transition, no view fade, no theme cross-fade.
      const { page, context } = await open(browser, '/chat', { reducedMotion });
      await page.waitForSelector('#themeToggle');
      const still = reducedMotion === 'reduce';
      const durations = await page.evaluate(() => ['.btn', '#tip', '#viewNav a', '.icon-btn'].map((s) => getComputedStyle(document.querySelector(s)).transitionDuration));
      for (const d of durations) assert.equal(d.split(',').every((x) => parseFloat(x) === 0), still, `${reducedMotion}: transition ${d}`);
      await page.click('#navDashboard');
      const fade = await page.evaluate(() => getComputedStyle(document.getElementById('dashPanel')).animationDuration);
      assert.equal(parseFloat(fade) === 0, still, `${reducedMotion}: view fade ${fade}`);
      await page.click('#themeToggle');
      assert.equal(await page.evaluate(() => document.documentElement.classList.contains('theme-fade')), !still, `${reducedMotion}: theme cross-fade`);
      await context.close();
    }

    for (const scheme of ['light', 'dark']) {
      // One logo in both themes: every mark and the favicon are the same picture.
      const { page, context } = await open(browser, '/chat', { scheme });
      await page.waitForFunction(() => document.querySelector('.brand .mark').complete && document.querySelector('.brand .mark').naturalWidth > 0);
      const srcs = await page.evaluate(() => [...document.querySelectorAll('img.mark'), document.getElementById('favicon')].map((n) => n.src || n.href));
      assert.equal(srcs.length, 4, 'header, empty chat, key card and favicon');
      assert.equal(new Set(srcs).size, 1, `${scheme}: one logo everywhere`);
      assert.match(srcs[0], /^data:image\//, 'inline, no fetch');
      await context.close();
    }
  } finally {
    await browser.close();
  }
}

main().catch((error) => { console.error(error); process.exit(1); });
