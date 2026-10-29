// sushi-test: tier=tool
// Generate a real-browser renderer regression page. Open the output HTML and expect all PASS.
// No model, network image, npm dependency or private conversation data is needed.
const fs = require('node:fs');
const path = require('node:path');
const html = fs.readFileSync('src/webui/index.html', 'utf8');
const app = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const purifier = html.match(/<script id="dompurify">([\s\S]*?)<\/script>/)?.[1] || '';
const helpers = app.slice(app.indexOf('function icon('), app.indexOf('async function copyText('));
const renderer = app.slice(app.indexOf('function inlineNodes('), app.indexOf('function readSsePayloads('));
const checks = async function () {
  const results = document.getElementById('results');
  let failures = 0;
  const assert = (condition, message) => { if (!condition) throw new Error(message); };
  async function test(name, fn) {
    try { await fn(); results.appendChild(el('p', null, `PASS ${name}`)); }
    catch (error) { failures++; results.appendChild(el('p', null, `FAIL ${name}: ${error.message}`)); }
  }
  await test('SVG is rendered only on click and sanitized', async () => {
    const code = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 60" onload="alert(1)"><script>alert(1)</script><foreignObject><div>unsafe</div></foreignObject><style>rect{fill:red}</style><rect width="120" height="60" fill="orange" style="filter:url(https://invalid.test/x)"/><use href="https://invalid.test/x.svg#x"/><image href="https://invalid.test/x.png"/><circle cx="30" cy="30" r="12" fill="url(https://invalid.test/fill)"/><text x="50" y="35">SVG</text></svg>';
    const block = codeBlockNode('svg', code);
    document.getElementById('examples').appendChild(block);
    assert(!block.querySelector('.svg-preview img[src]'), 'eager SVG loading');
    const button = [...block.querySelectorAll('button')].find((b) => /render svg/i.test(b.textContent));
    assert(button, 'missing Render SVG button');
    button.click();
    const image = block.querySelector('.svg-preview img');
    assert(image?.src.startsWith('data:image/svg+xml'), 'missing sanitized image');
    await image.decode();
    assert(image.naturalWidth > 0, 'SVG did not decode');
    const clean = decodeURIComponent(image.src.split(',').slice(1).join(','));
    assert(!/script|foreignObject|onload|invalid\.test|<style|style=/i.test(clean), 'unsafe SVG survived');
    assert(/rect/.test(clean) && /orange/.test(clean), 'geometry lost');
    assert(!document.querySelector('#examples svg rect'), 'SVG inserted into live DOM');
    assert(block.querySelector('code').textContent === code, 'source was modified');
    button.click();
    assert(!block.querySelector('.svg-preview').classList.contains('open'), 'hide failed');
  });
  await test('SVG preserves local paint references', () => {
    const source = sanitizedSvgSource('<svg xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="paint"><stop offset="0" stop-color="red"/></linearGradient></defs><rect width="10" height="10" fill="url(#paint)"/></svg>');
    assert(decodeURIComponent(source).includes('url(#paint)'), 'local paint reference lost');
  });
  await test('invalid SVG fails closed', () => {
    assert(typeof sanitizedSvgSource === 'function', 'missing sanitizer');
    let rejected = false;
    try { sanitizedSvgSource('<div>not SVG</div>'); } catch { rejected = true; }
    assert(rejected, 'non-SVG accepted');
  });
  await test('Markdown image uses the checked image loader and cache', async () => {
    const row = el('div');
    row.appendChild(inlineNodes('![Demo](https://example.com/photo.png)'));
    document.getElementById('examples').appendChild(row);
    await new Promise((resolve) => setTimeout(resolve, 0));
    const image = row.querySelector('img');
    assert(image?.src.startsWith('data:image/png;base64,'), 'image not loaded through tools');
    await image.decode();
    assert(image.naturalWidth === 1, 'PNG did not decode');
    assert(image.alt === 'Demo', 'alt lost');
    assert(row.querySelector('a').href === 'https://example.com/photo.png', 'source link lost');
    inlineNodes('![Again](https://example.com/photo.png)');
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert(imageRequests === 1, 'duplicate image fetch');
    const bad = inlineNodes('![Bad](javascript:alert%281%29)');
    assert(!bad.querySelector('img'), 'unsafe image URL accepted');
  });
  document.title = failures ? `${failures} FAILED` : 'All renderer checks passed';
  results.prepend(el('h2', null, document.title));
};
const stubs = `let imageRequests = 0;
function copyText() {}
async function callResearchTools(body) {
  if (body.name !== 'view_image' || body.arguments !== JSON.stringify({path_or_url:'https://example.com/photo.png'})) throw new Error('Unexpected image request');
  imageRequests++;
  return { image: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=' };
}`;
const escapeScript = (s) => s.replace(/<\/script/gi, '<\\/script');
const output = process.argv[2];
if (!output) throw new Error('Usage: node tests/test_webui_render.cjs /tmp/sushi-render-tests/index.html');
fs.mkdirSync(path.dirname(output), { recursive: true });
fs.writeFileSync(output, `<!doctype html><meta charset="utf-8"><title>Renderer checks</title><style>${html.match(/<style>([\s\S]*?)<\/style>/)[1]} body{padding:24px;overflow:auto} #examples{max-width:720px}</style><h1>Chat renderer regression checks</h1><div id="results"></div><div id="examples"></div><script>${escapeScript(purifier)}</script><script>${escapeScript(helpers + renderer + stubs + '(' + checks.toString() + ')();')}</script>`);
console.log(output);
