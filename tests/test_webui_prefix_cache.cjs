// sushi-test: tier=hermetic needs=node,playwright
// The chat page's SSD prefix-cache meter, driven through the page's own functions without a browser or model.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync('src/webui/index.html', 'utf8');
const script = source.match(/<script>([\s\S]*?)<\/script>/)[1];
const start = script.indexOf('function prefixCacheMeter(');
const slice = script.slice(start, script.indexOf('async function refreshServerDot(', start));

const nodes = {};
const node = (id) => (nodes[id] ??= { id, hidden: false, textContent: '', style: {}, attrs: {}, setAttribute(k, v) { this.attrs[k] = v; } });
const GiB = 1024 ** 3;
let reads = 0;
let props = {};
const timers = [];
const page = vm.createContext({
  $: node,
  apiJson: async () => { reads += 1; return props; },
  setTimeout: (fn, ms) => timers.push([fn, ms]),
});
vm.runInContext(slice, page);

// Tier off: no budget, nothing to show.
assert.equal(page.prefixCacheMeter({ disk_bytes: 0, disk_used_bytes: 0 }), null);
assert.equal(page.prefixCacheMeter(undefined), null);
page.renderPrefixCache({ disk_bytes: 0 });
assert.equal(node('prefixCacheSection').hidden, true, 'hidden while the SSD tier is off');

// Tier on: the bar is used / budget and the text names both in GB.
page.renderPrefixCache({ disk_bytes: 20 * GiB, disk_used_bytes: 5 * GiB, disk_entries: 3 });
assert.equal(node('prefixCacheSection').hidden, false);
assert.equal(node('prefixCacheBar').style.width, '25%');
assert.equal(node('prefixCacheUsage').textContent, '5.00 / 20.0 GB');
assert.equal(node('prefixCacheEntries').textContent, '3 entries');
assert.equal(node('prefixCacheMeter').attrs['aria-valuenow'], '25');
page.renderPrefixCache({ disk_bytes: 20 * GiB, disk_used_bytes: 1, disk_entries: 1 });
assert.equal(node('prefixCacheEntries').textContent, '1 entry');

// An over-full read never draws past the track; a tier that turns off hides the meter again.
assert.equal(page.prefixCacheMeter({ disk_bytes: 2 * GiB, disk_used_bytes: 3 * GiB }).percent, 100);
page.renderPrefixCache({ disk_bytes: 0 });
assert.equal(node('prefixCacheSection').hidden, true);

// Each turn reads /props at once and again after the server's flush.
props = { settings: { prefix_cache: { disk_bytes: 10 * GiB, disk_used_bytes: 2 * GiB, disk_entries: 2 } } };
page.refreshPrefixCache();
assert.equal(reads, 1);
assert.equal(timers.length, 1);
assert.ok(timers[0][1] >= 1000);
timers[0][0]();
assert.equal(reads, 2);
// apiJson resolves on a microtask; the render lands after it.
Promise.resolve().then(() => Promise.resolve()).then(() => {
  assert.equal(node('prefixCacheUsage').textContent, '2.00 / 10.0 GB');
  console.log('Web UI prefix-cache meter: passed');
});
