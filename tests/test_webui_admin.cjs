// sushi-test: tier=hermetic needs=node
// The Models and Settings views' pure functions (sizes, ages, control values and the changes a Save sends),
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
assert.equal(page.adminBytes(5 * GiB), '5.00 GiB');
assert.equal(page.adminBytes(25 * GiB), '25.0 GiB');
assert.equal(page.adminBytes(300 * 1024 ** 2), '300 MiB');
assert.equal(page.adminBytes(0), '0 KiB');

const now = 1_700_000_000_000;
assert.equal(page.adminAgo(0, now), '-');
assert.equal(page.adminAgo(now / 1000 - 30, now), 'just now');
assert.equal(page.adminAgo(now / 1000 - 600, now), '10 min ago');
assert.equal(page.adminAgo(now / 1000 - 7200, now), '2 h ago');
assert.equal(page.adminAgo(now / 1000 - 3 * 86400, now), '3 days ago');

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
assert.equal(page.adminBytes(1023.8 * 1024 ** 2), '1.00 GiB', 'no "1024 MiB"');
assert.equal(page.adminBytes(110 * GiB), '110 GiB');
assert.equal(page.adminBytes(1.72 * 1024 ** 2), '1.72 MiB', 'MiB keeps its decimals');
assert.equal(page.adminBytes(12.5 * 1024 ** 2), '12.5 MiB');
assert.equal(page.adminTokens(131072), '128k');
assert.equal(page.adminTokens(1048576), '1M');
assert.equal(page.adminTokens(1000), '1000');
assert.equal(page.adminProblem(int, '1e'), 'Enter a whole number from 0 to 100');
assert.equal(page.adminProblem(int, '5'), null);
assert.equal(page.adminProblem({ name: 'ctx_size', type: 'int', min: 1, max: 4294967295 }, 'x'), 'Enter a whole number of 1 or more', 'a u32 bound is "or more"');
assert.equal(page.adminProblem({ name: 'think_penalty', type: 'number', min: 0, max: 20 }, 'x'), 'Enter a number from 0 to 20');
assert.equal(page.adminFileText('8'), '8');
assert.equal(page.adminFileText('off'), 'off');
assert.equal(page.adminFileText(null), '');
assert.equal(page.adminFileText(false), 'false');
assert.equal(page.adminFileText(40), '40');
// A stored "off" equals the control's "off", so an unrelated Save never rewrites or deletes it.
const offKv = { ...settings, kv_quant: { file: 'off', locked: false } };
const changesKv = (texts, bad) => JSON.parse(JSON.stringify(page.adminChanges(specs, offKv, texts, bad)));
assert.deepEqual(changesKv({ kv_quant: 'off' }), { set: {}, bad: null });
assert.deepEqual(changesKv({ mtp: 'true', kv_quant: 'off' }), { set: { mtp: true }, bad: null }, 'changing another key leaves kv_quant alone');
// A number box holding half a number ("1e", "-") reads as empty; that is a bad value, not "remove the key".
assert.equal(page.adminControlValue(int, '', true).ok, false);
assert.deepEqual(changesKv({ ssd_prefetch: '' }, { ssd_prefetch: true }), { set: {}, bad: 'ssd_prefetch' });
assert.deepEqual(changesKv({ ssd_prefetch: '' }), { set: { ssd_prefetch: null }, bad: null }, 'a truly empty box still removes the key');
// "065536" is the saved 65536, not a change.
const ctxSpec = { name: 'ctx_size', type: 'int', min: 1, max: 4294967295 };
assert.deepEqual(JSON.parse(JSON.stringify(page.adminChanges([ctxSpec], { ctx_size: { file: 65536 } }, { ctx_size: '065536' }))), { set: {}, bad: null });

// Error text: the server's message, or a short line; never a page of HTML.
assert.equal(page.adminErrorText(new Error('400 {"error":{"message":"models_dir is not a folder"}}')), 'models_dir is not a folder');
assert.equal(page.adminErrorText(new Error('502 <html><body>Bad gateway</body></html>')), 'The server answered with error 502. Try again; if it repeats, check the server log.');
assert.equal(page.adminErrorText(new TypeError('Failed to fetch')), 'Cannot reach the server. Check that sushi is running.');
assert.equal(page.adminErrorText(new Error('500 ')), 'The server answered with error 500. Try again; if it repeats, check the server log.');
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
console.log('Web UI admin views: passed');
