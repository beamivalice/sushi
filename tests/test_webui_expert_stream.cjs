// sushi-test: tier=hermetic needs=node
// The dashboard's Expert streaming card reading of `/metrics.json` `expert_stream`, driven through the page's own
// function without a browser or model.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync('src/webui/index.html', 'utf8');
const start = source.indexOf('function expertStream(');
assert.ok(start > 0, 'the page has expertStream');
const page = vm.createContext({});
vm.runInContext(source.slice(start, source.indexOf('\n  }\n', start) + 4) + '\nthis.expertStream = expertStream;', page);

const GiB = 1024 ** 3;
const counters = (routes, hits, more = {}) => ({ engines: 1, cache_bytes: 12 * GiB, forwards: 10, forward_ns: 2e9, routes, hits, fill_bytes: 3 * GiB, fill_ns: 5e8, ssd_read_bytes: 4 * GiB, ...more });

assert.equal(page.expertStream(undefined, null, 0), null, 'a server without the counters reports none');

const first = page.expertStream(counters(1000, 900), null, 10_000);
assert.equal(first.hitRate, 0.9, 'since the server started');
assert.equal(first.recent, null, 'no previous poll, no recent rate');
assert.equal(first.readRate, null);
assert.equal(first.misses, 100);
assert.equal(first.waitShare, 0.25, 'fill time over forward time');
assert.equal(first.cacheBytes, 12 * GiB);
assert.equal(first.streaming, true);

const second = page.expertStream(counters(1200, 1050, { ssd_read_bytes: 4 * GiB + 2e9 }), first, 12_000);
assert.equal(second.recent, 0.75, 'hits over routes between the two polls');
assert.equal(second.readRate, 1e9, 'SSD bytes per second between the two polls');

const idle = page.expertStream(counters(1200, 1050, { ssd_read_bytes: 4 * GiB + 2e9 }), second, 14_000);
assert.equal(idle.recent, null, 'no forward between polls: no recent rate');
assert.equal(idle.readRate, 0, 'idle reads nothing');

const restarted = page.expertStream(counters(5, 5, { ssd_read_bytes: 10 }), idle, 16_000);
assert.equal(restarted.recent, null, 'a restarted server counts from zero: no rate across it');
assert.equal(restarted.readRate, null);

const none = page.expertStream({ engines: 0, cache_bytes: 0, forwards: 0, forward_ns: 0, routes: 0, hits: 0, fill_bytes: 0, fill_ns: 0, ssd_read_bytes: 0 }, null, 0);
assert.equal(none.streaming, false, 'no engine and nothing streamed yet');
assert.equal(none.hitRate, null);
assert.equal(none.waitShare, null);
const resident = page.expertStream(counters(0, 0, { engines: 1 }), null, 0);
assert.equal(resident.streaming, true, 'an engine is up before its first forward');
