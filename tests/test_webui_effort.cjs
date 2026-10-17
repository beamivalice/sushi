// The chat page's thinking-effort menu, driven through the page's own functions without a browser or model.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync('src/webui/index.html', 'utf8');
const script = source.match(/<script>([\s\S]*?)<\/script>/)[1];
const slice = (from, to) => {
  const start = script.indexOf(from);
  return script.slice(start, script.indexOf(to, start));
};

// A browser select holds only a value one of its options carries, and starts on its first option.
function fakeSelect() {
  const select = { dataset: {}, disabled: false, options: [], current: '' };
  Object.defineProperty(select, 'textContent', { set() { select.options = []; select.current = ''; } });
  select.appendChild = (option) => {
    select.options.push(option);
    if (select.options.length === 1) select.current = option.value;
  };
  Object.defineProperty(select, 'value', {
    get: () => select.current,
    set: (v) => { select.current = select.options.some((o) => o.value === v) ? v : ''; },
  });
  return select;
}

const select = fakeSelect();
const label = { textContent: '' };
const budget = { value: '', disabled: false, placeholder: '', dataset: {} };
const store = {};
let model = null;
const page = vm.createContext({
  $: (id) => ({ effortSelect: select, composerEffortName: label, optThinkBudget: budget })[id],
  el: (tag, className, text) => ({ tag, text, value: '' }),
  readStore: (key) => store[key] ?? null,
  selectedModel: () => model,
});
vm.runInContext(slice('function renderEffortOptions(', 'function renderModelStatus('), page);
const pick = (word) => { select.value = word; page.applyEffortBudget(); return budget.value; };
const show = (m) => { model = m; page.renderEffortOptions(); return select.value; };

const models = {
  qwen: { id: 'qwen', reasoning_efforts: ['off', 'low', 'medium', 'xhigh'], reasoning_effort_budgets: { low: 2048, medium: 8192 }, default_reasoning_effort: 'off' },
  mimo: { id: 'mimo', reasoning_efforts: ['off', 'on'], reasoning_effort_budgets: {}, default_reasoning_effort: 'on' },
  glm: { id: 'glm', reasoning_efforts: ['low', 'high', 'max'], reasoning_effort_budgets: {}, default_reasoning_effort: 'high' },
};
for (const m of Object.values(models)) {
  assert.equal(show(m), m.default_reasoning_effort, `${m.id} starts on its own default`);
  assert.deepEqual(select.options.map((o) => o.value), m.reasoning_efforts, `${m.id} lists exactly its own words`);
  assert.equal(label.textContent, m.default_reasoning_effort, `${m.id} label names the real word`);
}

// A stored choice holds on a model that takes it; elsewhere that model's default applies.
store['sushi-effort'] = 'off';
assert.equal(show(models.mimo), 'off');
assert.equal(show(models.glm), 'high');
assert.equal(label.textContent, 'high');
assert.equal(show(models.qwen), 'off');
store['sushi-effort'] = 'default';
assert.equal(show(models.mimo), 'on');

// The budget box takes the picked word's budget; an uncapped word leaves it empty, off disables it.
show(models.qwen);
assert.equal(pick('medium'), '8192');
assert.equal(budget.disabled, false);
assert.equal(pick('low'), '2048');
assert.equal(pick('xhigh'), '');
assert.equal(pick('off'), '');
assert.equal(budget.disabled, true);
// A typed budget survives the model poll's re-render, and moves again only when the word or model does.
pick('medium');
budget.value = '3000';
store['sushi-effort'] = 'medium';
show(models.qwen);
assert.equal(budget.value, '3000');
assert.equal(pick('low'), '2048');
show(models.glm);
assert.equal(budget.value, '');
assert.equal(budget.disabled, false, 'an uncapped word can still be capped by hand');

// Every request names the selected effort, and the box's budget when it holds one.
const request = vm.createContext({ $: (id) => ({ effortSelect: select, optThinkBudget: budget })[id] ?? { value: '' } });
vm.runInContext(slice('function wireMessages(', 'function createStreamingView('), request);
store['sushi-effort'] = 'default';
show(models.mimo);
assert.equal(request.buildRequest('mimo', []).reasoning_effort, 'on');
assert.equal('reasoning_budget_tokens' in request.buildRequest('mimo', []), false);
budget.value = '1500';
assert.equal(request.buildRequest('mimo', []).reasoning_budget_tokens, 1500);
show(models.qwen);
assert.equal(request.buildRequest('qwen', []).reasoning_effort, 'off');
assert.equal('reasoning_budget_tokens' in request.buildRequest('qwen', []), false, 'off sends no budget');
pick('medium');
assert.equal(request.buildRequest('qwen', []).reasoning_budget_tokens, 8192);
const meta = vm.createContext({ fmtInt: (n) => String(n) });
vm.runInContext(slice('function replyMeta(', 'async function streamReply('), meta);
assert.equal(meta.replyMeta(null, null, null, false, 0, 'Qwen3.8-Flash-Next has no `high` effort; served at `xhigh`'), 'Qwen3.8-Flash-Next has no `high` effort; served at `xhigh`');
assert.equal(meta.replyMeta({ completion_tokens: 3 }, null, 'stop', false, 0, null), '3 tokens');
console.log('Web UI effort menu: passed');
