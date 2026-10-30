// sushi-test: tier=hermetic needs=node,playwright
// The chat page's SSD prefix-cache meter and the dashboard's write-counter card, driven through the page's own
// functions without a browser or model.
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

// The dashboard card: the rate since the previous poll and the bytes written per byte on disk.
const dash = source.match(/<script id="dashboardScript">([\s\S]*?)<\/script>/)[1];
const writesStart = dash.indexOf('function prefixWrites(');
const writes = vm.createContext({});
vm.runInContext(dash.slice(writesStart, dash.indexOf('function push(', writesStart)) + '\nthis.prefixWrites = prefixWrites;', writes);
const counters = (kv, extra = {}) => ({ written_bytes: { kv, ssm: 0, qsa: 0, spec: 0, ring: 0, meta: 0, ...extra }, failed_bytes: 7 });

assert.equal(writes.prefixWrites(undefined, null, 0, 0), null, 'a server without the counters reports none');
const first = writes.prefixWrites(counters(8 * GiB, { qsa: 2 * GiB }), null, 10_000, 5 * GiB);
assert.equal(first.total, 10 * GiB);
assert.equal(first.rate, null, 'no rate on the first poll');
assert.equal(first.amplification, 2, 'written over what is on disk');
assert.equal(first.failed, 7);
assert.equal(writes.prefixWrites(counters(1), null, 0, 0).amplification, null, 'nothing on disk, no ratio');

const second = writes.prefixWrites(counters(8 * GiB + 100, { qsa: 2 * GiB + 300 }), first, 12_000, 5 * GiB);
assert.equal(second.rate, 200, 'bytes per second between the two polls');
assert.equal(writes.prefixWrites(counters(5), second, 14_000, 0).rate, null, 'a restarted server counts from zero: no negative rate');
assert.equal(writes.prefixWrites(counters(8 * GiB + 100, { qsa: 2 * GiB + 300 }), second, 14_000, 0).rate, 0, 'idle writes nothing');
console.log('Web UI prefix-cache writes: passed');
