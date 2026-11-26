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
const main = source.match(/<script>([\s\S]*?)<\/script>/)[1];
vm.runInContext(main.slice(main.indexOf('function forgetAnsweredLoads('), main.indexOf('const modelStore = {')), page);

const GiB = 1024 ** 3;
assert.equal(page.adminBytes(5 * GiB), '5.00 GiB');
assert.equal(page.adminBytes(25 * GiB), '25.0 GiB');
assert.equal(page.adminBytes(300 * 1024 ** 2), '300 MiB');
assert.equal(page.adminBytes(0), '0 KiB');

// A pack's size line: what memory takes, then the whole pack when it is more (tables read from the SSD); one number when
// the two agree within 5%; no memory claim for a pack that streams from the SSD, only its total.
assert.equal(page.adminSizeLine(76.1 * GiB, 194 * GiB, false), '76.1 GiB in memory · 194 GiB total');
assert.equal(page.adminSizeLine(64.8 * GiB, 66 * GiB, false), '64.8 GiB in memory');
assert.equal(page.adminSizeLine(64.8 * GiB, null, false), '64.8 GiB in memory');
assert.equal(page.adminSizeLine(475 * GiB, 475 * GiB, true), '475 GiB total');
assert.equal(page.adminSizeLine(null, null, false), '');

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
assert.equal(value(int, '1e2').ok, false, 'exponent notation is not a whole number');
assert.equal(value(int, '0x10').ok, false);
assert.equal(value(num, '1e1').ok, false);
assert.equal(value(num, '.5').ok, true);
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
assert.equal(page.adminTokens(131072), '131.1k', 'thousands, as the chat badge reads it');
assert.equal(page.adminTokens(974800), '974.8k');
assert.equal(page.adminTokens(1048576), '1M');
assert.equal(page.adminTokens(8192), '8,192');
assert.equal(page.adminTokens(1000), '1,000');
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

// A failure shown for an action goes when that action is tried again: one key of a Map or an object, or all of them.
const failedMap = new Map([['a', 'x'], ['b', 'y']]);
page.adminAttempt(failedMap, 'a');
assert.deepEqual([...failedMap], [['b', 'y']]);
page.adminAttempt(failedMap);
assert.equal(failedMap.size, 0);
const failedObj = { port: 'needs root', models_dir: 'not a folder' };
page.adminAttempt(failedObj, 'port');
assert.deepEqual(JSON.parse(JSON.stringify(failedObj)), { models_dir: 'not a folder' });
page.adminAttempt(failedObj);
assert.deepEqual(JSON.parse(JSON.stringify(failedObj)), {});
// A refusal note this session recorded goes once the server answers for the model: loaded, loading or failed (its own
// `error` is then the reason), wherever it was loaded. A model still listed unloaded keeps the note.
const loadFailures = new Map([['a', 'out of memory'], ['b', 'bad config'], ['c', 'x'], ['d', 'no room']].map(([id, message]) => [id, { message, seq: 1 }]));
assert.equal(page.forgetAnsweredLoads(loadFailures, [{ id: 'a', state: 'ready' }, { id: 'b', state: 'error' }, { id: 'c', state: 'loading' }, { id: 'd', state: 'unloaded' }], 2), true);
assert.deepEqual([...loadFailures].map(([id, note]) => [id, note.message]), [['d', 'no room']], 'a model the server lists unloaded keeps its refusal');
assert.equal(page.forgetAnsweredLoads(loadFailures, [{ id: 'd', state: 'unloaded' }], 2), false);
// An answer to a read sent before the note was written (seq 5 note, read 3 or 5) says nothing about it.
const noted = new Map([['a', { message: 'out of memory', seq: 5 }]]);
assert.equal(page.forgetAnsweredLoads(noted, [{ id: 'a', state: 'loading' }], 3), false);
assert.equal(page.forgetAnsweredLoads(noted, [{ id: 'a', state: 'loading' }], 5), false);
assert.equal(page.forgetAnsweredLoads(noted, [{ id: 'a', state: 'loading' }], 6), true);
// A Store is the newest answer: staleness is decided before ok or error, a failed read keeps the last good value, and
// text and value arrive together.
const storeChecks = (async () => {
  const store = new (vm.runInContext('Store', page))();
  const gate = () => { let go; const wait = new Promise((r) => { go = r; }); return { wait, go }; };
  const slow = gate();
  const older = store.read(async () => { await slow.wait; throw new Error('502 proxy reset'); });
  const newer = await store.read(async () => '{"v":1}');
  assert.deepEqual([newer.stale, newer.changed, store.value.v, store.text], [false, true, 1, '{"v":1}']);
  slow.go();
  const late = await older;
  assert.equal(late.stale, true, 'an older failed read is dropped');
  assert.equal(store.error, null, 'and sets no error');
  const failed = await store.read(async () => { throw new Error('500 boom'); });
  assert.deepEqual([failed.stale, failed.changed, store.value.v, store.error.message], [false, false, 1, '500 boom'], 'a failed read keeps the good value');
  const same = await store.read(async () => '{"v":1}');
  assert.deepEqual([same.changed, store.error], [false, null], 'the same text is no change and ends the error');
  const bad = await store.read(async () => 'not json');
  assert.deepEqual([bad.changed, store.value.v, store.text, bad.error.name === 'SyntaxError'], [false, 1, '{"v":1}', true], 'text that does not parse is kept nowhere');
  const none = await store.read(async () => null);
  assert.deepEqual([none.changed, store.value.v, store.error], [false, 1, null], 'a read with nothing to keep ends the error');
  const next = await store.read(async () => '{"v":2}');
  assert.deepEqual([next.previous.v, next.value.v], [1, 2]);
})();
// A refused Save names a setting or the form; the next Save and an edit of that value answer it, and it blocks nothing.
const form = () => ({ texts: {}, errors: {}, refused: {}, saveError: '' });
const shown = (P) => JSON.parse(JSON.stringify({ refused: P.refused, saveError: P.saveError, errors: P.errors }));
let P = form();
page.adminSaveRefused(P, 'ctx_size', 'too large');
page.adminSaveRefused(P, null, 'disk full');
assert.deepEqual(shown(P), { refused: { ctx_size: 'too large' }, saveError: 'disk full', errors: {} }, 'a refusal is no input error');
page.adminEdited(P, 'kv_quant');
assert.deepEqual(shown(P), { refused: { ctx_size: 'too large' }, saveError: '', errors: {} }, 'editing another value clears the form message only');
page.adminSaveRefused(P, null, 'disk full');
page.adminSaveStarted(P);
assert.deepEqual(shown(P), { refused: {}, saveError: '', errors: {} }, 'a new Save starts with no earlier refusal');
page.adminSaveRefused(P, 'ctx_size', 'too large');
page.adminEdited(P, 'ctx_size');
assert.deepEqual(shown(P), { refused: {}, saveError: '', errors: {} }, 'editing the refused value answers it');

// Compact is the default and the only other stored value that counts is "advanced".
assert.equal(page.uiModeOf(null), 'compact');
assert.equal(page.uiModeOf('garbage'), 'compact');
assert.equal(page.uiModeOf('advanced'), 'advanced');
assert.match(page.loadFailureText('Failed to load model: ExpertCacheTooSmall'), /Memory for SSD streaming.*--ssd-budget-gb/);
assert.match(page.loadFailureText('Failed to load model: OutOfMemory'), /Not enough memory/);
assert.equal(page.loadFailureText('out of memory: needs 61 GiB'), 'out of memory: needs 61 GiB', 'a message with words stays as the server wrote it');
assert.match(page.loadFailureText('Failed to load model: SomethingNew'), /SomethingNew.*server log/, 'an unknown tag keeps its name and points to the log');
// A refused load's message reaches the Models page, the toast and the chat whole, its lines shown as lines.
const refused = 'Not enough memory to load this model.\nIt needs 20.0 GB, and only 12.0 GB of RAM is free right now. Free about 8.0 GB more.\nTo load it:\n- Quit other apps or unload another model, then retry.\n- Last resort: --skip-mem-preflight skips this check; the load can then fail later or stall the Mac.';
const body = JSON.stringify({ error: { message: refused, type: 'out_of_memory', param: null, code: 503 } });
assert.equal(page.loadFailureText(page.adminErrorText(new Error(`503 ${body}`))), refused);
vm.runInContext(main.slice(main.indexOf('function refusalText('), main.indexOf('// A reply that breaks off')), page);
assert.equal(page.refusalText(503, refused), refused, 'a message with its own actions gets no generic one, and no cut');
// A refusal carries the actions for its own bound: a one-line 400 or 503 reaches the chat as the server wrote it.
for (const [status, msg] of [[503, 'The server is shutting down: start it again, then retry.'], [503, 'This request ran out of GPU memory while it ran and was stopped.'], [400, 'This image needs 5.0 GB of GPU memory to encode, and 2.9 GB is available.']]) {
  assert.equal(page.refusalText(status, msg), msg, `${status}: no generic action is added to a refusal`);
}
assert.equal(page.refusalText(503, 'no period'), 'no period.');
assert.match(page.refusalText(500, 'boom'), /^boom\. Retry; if it happens again, check the server log\.$/);
assert.match(page.refusalText(429, 'busy'), /^busy\. Wait a moment, then Retry\.$/);
for (const rule of ['.alert span', '.toast', '.chat-error span']) {
  assert.match(source, new RegExp(`\\${rule.replace(' ', ' ')}[^{]*\\{[^}]*white-space: pre-line`), `${rule} shows a message's lines`);
}
storeChecks.then(() => console.log('Web UI admin views: passed'), (e) => { console.error(e); process.exitCode = 1; });

// A save that would leave the context above the token budget lowers the context to the budget and says why.
{
  const within = (set, settings) => { const s = { ...set }; return { note: page.adminContextWithinBudget(s, settings), set: s }; };
  const both = within({ ctx_size: 8192, max_kv_tokens: 4096 }, {});
  assert.equal(both.set.ctx_size, 4096);
  assert.match(both.note, /lowered to 4,096 tokens/);
  const budgetCut = within({ max_kv_tokens: 4096 }, { ctx_size: { file: 8192 } });
  assert.equal(budgetCut.set.ctx_size, 4096, 'lowering the budget under a saved context lowers the context too');
  assert.equal(within({ ctx_size: 4096, max_kv_tokens: 4096 }, {}).note, null);
  assert.equal(within({ ctx_size: 8192 }, {}).note, null, 'no budget, no cap');
  assert.equal(within({ max_kv_tokens: null }, { ctx_size: { file: 8192 } }).note, null, 'clearing the budget clears the cap');
  assert.equal(within({ ctx_size: null, max_kv_tokens: 4096 }, {}).note, null, 'an auto context is held by the server');
}
