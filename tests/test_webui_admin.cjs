// sushi-test: tier=hermetic needs=node
// The Models and Settings views' pure functions (sizes, ages, write rate, control values and the changes a Save sends),
// driven through the page's own script under node:vm, with no browser and no model.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync('src/webui/index.html', 'utf8');
const script = source.match(/<script id="adminScript">([\s\S]*?)<\/script>/)[1];
const slice = script.slice(script.indexOf('function adminBytes('), script.indexOf('(() => {'));
const page = vm.createContext({});
vm.runInContext(slice, page);

const GiB = 1024 ** 3;
assert.equal(page.adminBytes(5 * GiB), '5.00 GB');
assert.equal(page.adminBytes(25 * GiB), '25.0 GB');
assert.equal(page.adminBytes(300 * 1024 ** 2), '300 MB');
assert.equal(page.adminBytes(0), '0 KB');

const now = 1_700_000_000_000;
assert.equal(page.adminAgo(0, now), '-');
assert.equal(page.adminAgo(now / 1000 - 30, now), 'just now');
assert.equal(page.adminAgo(now / 1000 - 600, now), '10 min ago');
assert.equal(page.adminAgo(now / 1000 - 7200, now), '2 h ago');
assert.equal(page.adminAgo(now / 1000 - 3 * 86400, now), '3 days ago');

// Write rate: bytes per second between two readings; none on the first, none across a counter reset.
assert.equal(page.adminWriteRate(undefined, { t: 2000, written: 5 }), null);
assert.equal(page.adminWriteRate({ t: 1000, written: 100 }, { t: 3000, written: 300 }), 100);
assert.equal(page.adminWriteRate({ t: 1000, written: 100 }, { t: 3000, written: 50 }), null);
assert.equal(page.adminWriteRate({ t: 3000, written: 100 }, { t: 3000, written: 150 }), null);

const bool = { name: 'mtp', type: 'bool' };
const int = { name: 'ssd_prefetch', type: 'int', min: 0, max: 100 };
const num = { name: 'think_penalty', type: 'number', min: 0, max: 20 };
const choice = { name: 'kv_quant', type: 'choice', choices: ['off', '4', '6', '8'] };
const path = { name: 'models_dir', type: 'choice' };
const value = (spec, text) => JSON.parse(JSON.stringify(page.adminControlValue(spec, text)));
assert.deepEqual(value(bool, 'true'), { ok: true, value: true });
assert.deepEqual(value(bool, ''), { ok: true, value: null });
assert.equal(value(bool, 'yes').ok, false);
assert.deepEqual(value(int, '0'), { ok: true, value: 0 });
assert.equal(value(int, '101').ok, false);
assert.equal(value(int, '1.5').ok, false);
assert.deepEqual(value(num, '2.5'), { ok: true, value: 2.5 });
assert.equal(value(num, '20.5').ok, false);
assert.deepEqual(value(choice, '8'), { ok: true, value: '8' });
assert.equal(value(choice, '5').ok, false);
assert.equal(value(path, '/models').ok, true);

// A Save sends only the keys whose control differs from the file, skips a flag-locked key, and removes a cleared one.
const specs = [bool, int, num, choice];
const settings = {
  mtp: { file: false, locked: false },
  ssd_prefetch: { file: 40, locked: false },
  think_penalty: { file: null, locked: true, flag: '2' },
  kv_quant: { file: '4', locked: false },
};
const changes = (texts) => JSON.parse(JSON.stringify(page.adminChanges(specs, settings, texts)));
assert.deepEqual(changes({ mtp: 'false', ssd_prefetch: '40', think_penalty: '5', kv_quant: '4' }), { set: {}, bad: null });
assert.deepEqual(changes({ mtp: 'true', ssd_prefetch: '', think_penalty: '5', kv_quant: '8' }), { set: { mtp: true, ssd_prefetch: null, kv_quant: '8' }, bad: null });
assert.deepEqual(changes({ ssd_prefetch: '900' }), { set: {}, bad: 'ssd_prefetch' });
// The dashboard's rates come from the server's samples: exact between neighbours, zero across a gap or on the first.
const rates = (a, b) => JSON.parse(JSON.stringify(page.historyRates(a, b, 7000)));
assert.deepEqual(rates(null, [1000, 10, 5]), { gen: 0, pre: 0 });
assert.deepEqual(rates([1000, 10, 100], [3000, 50, 300]), { gen: 20, pre: 100 });
assert.deepEqual(rates([1000, 50, 300], [3000, 50, 300]), { gen: 0, pre: 0 });
assert.deepEqual(rates([1000, 10, 900], [3000, 10, 40]), { gen: 0, pre: 20 }, 'a prefill counter that fell is a new prefill');
assert.deepEqual(rates([1000, 10, 100], [9000, 500, 900]), { gen: 0, pre: 0 }, 'a gap is not averaged into a rate');
assert.deepEqual(rates([3000, 10, 100], [3000, 20, 200]), { gen: 0, pre: 0 });

// Compact is the default and the only other stored value that counts is "advanced".
assert.equal(page.uiModeOf(null), 'compact');
assert.equal(page.uiModeOf('garbage'), 'compact');
assert.equal(page.uiModeOf('advanced'), 'advanced');
assert.equal(page.uiModeLabel('compact'), 'Advanced view');
assert.equal(page.uiModeLabel('advanced'), 'Compact view');
console.log('Web UI admin views: passed');
