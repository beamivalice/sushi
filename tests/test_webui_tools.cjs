// Exercise the actual page functions without a browser or model.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync('src/webui/index.html', 'utf8');
const script = source.match(/<script>([\s\S]*?)<\/script>/)[1];
const context = vm.createContext({
  $: () => ({ value: '' }),
});
const start = script.indexOf('function wireMessages(');
const end = script.indexOf('function createStreamingView(', start);
vm.runInContext(script.slice(start, end), context);
const tools = [{ type: 'function', function: { name: 'web_search' } }];
const history = [
  { role: 'user', content: 'Search' },
  { role: 'assistant', content: '', reasoning: 'Look it up', tool_calls: [{ id: 'c1', type: 'function', function: { name: 'web_search', arguments: '{"query":"sushi"}' } }] },
  { role: 'tool', tool_call_id: 'c1', content: 'Found a page' },
];
assert.equal(context.buildRequest('test', history).tools, undefined);
assert.equal(context.buildRequest('test', history).tool_choice, 'none', 'no tools offered: the server bans call markup');
assert.deepEqual(context.buildRequest('test', history, tools).tools, tools);
assert.equal(context.buildRequest('test', history, tools).tool_choice, undefined);
const wire = context.wireMessages(history);
assert.equal(wire.length, 3);
assert.equal(wire[1].tool_calls[0].id, 'c1');
assert.equal(wire[1].reasoning_content, 'Look it up');
assert.equal(wire[2].tool_call_id, 'c1');
console.log('Web UI tools: passed');
const loopStart = script.indexOf('async function runResearchTurn(');
vm.runInContext(script.slice(loopStart, script.indexOf('async function runTurn(', loopStart)), context);
context.renderTranscript = () => {};
const call = (id) => ({ id, type: 'function', function: { name: 'web_search', arguments: '{}' } });
(async () => {
  let count = 0;
  let executed = 0;
  context.streamReply = async (model, messages, signal, defs) => {
    count++;
    if (count <= 8) {
      assert.equal(defs, tools);
      return { text: '', tool_calls: [call(`c${count}`)] };
    }
    assert.equal(defs, null);
    assert.equal(messages.at(-1).role, 'user');
    assert.match(messages.at(-1).content, /Answer now/);
    return { text: 'Done', tool_calls: [] };
  };
  context.callResearchTools = async (body) => { assert.equal(body.directory, "/selected/chat/folder"); assert.equal(body.write, true); executed++; return { text: 'Result' }; };
  const messages = [];
  await context.runResearchTurn('test', messages, { aborted: false }, tools, false, '/selected/chat/folder', true);
  assert.equal(count, 9);
  assert.equal(executed, 8);
  assert.equal(messages.at(-1).content, 'Done');
  assert.equal(messages.filter((m) => m.role === 'tool').length, 8);
  assert.ok(!messages.some((m) => /Answer now/.test(m.content)), 'the nudge rides the last request only, never the saved chat');

  const signal = { aborted: false };
  context.streamReply = async () => ({ text: '', tool_calls: [call('a'), call('b')] });
  context.callResearchTools = async () => { signal.aborted = true; throw new Error('abort'); };
  const cancelled = [];
  await context.runResearchTurn('test', cancelled, signal, tools, false);
  assert.equal(cancelled.length, 3);
  assert.equal(cancelled[1].tool_call_id, 'a');
  assert.equal(cancelled[2].tool_call_id, 'b');
  assert.equal(cancelled[2].content, 'Tool call cancelled');

  context.streamReply = async (model, messages, signal, defs) => {
    assert.equal(defs, undefined);
    return { text: 'No tools', tool_calls: [] };
  };
  context.callResearchTools = async () => assert.fail('Disabled tools executed');
  await context.runResearchTurn('test', [], { aborted: false }, undefined, false);
  console.log('Web UI tool loop: round limit, cancellation, disabled pack passed');
})().catch((error) => { console.error(error); process.exitCode = 1; });
let prefixCacheReads = 0;
const streaming = vm.createContext({
  $: () => ({ value: '' }), TextDecoder, Uint8Array, performance,
  refreshPrefixCache: () => { prefixCacheReads++; },
  perfMon: { begin() {}, finish() {}, delta() {} },
  createStreamingView: () => ({ text: '', reasoning: '', schedule() {}, finalize() {} }),
  replyMeta: () => null,
});
vm.runInContext(script.slice(start, end), streaming);
const sseStart = script.indexOf('function readSsePayloads(');
vm.runInContext(script.slice(sseStart, script.indexOf('\n/*', sseStart)), streaming);
const streamStart = script.indexOf('async function streamReply(');
vm.runInContext(script.slice(streamStart, script.indexOf('function setChatBusy(', streamStart)), streaming);
(async () => {
  for (const finish of ['tool_calls', 'length']) {
    const events = [
      { choices: [{ delta: { tool_calls: [{ index: 0, id: 'c1', function: { name: 'web_search', arguments: '{"query":' } }] } }] },
      { choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: '"sushi"}' } }] } }] },
      { choices: [{ delta: {}, finish_reason: finish }] },
    ];
    const bytes = Buffer.from(events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join('') + 'data: [DONE]\n\n');
    let offset = 0;
    streaming.api = async () => ({ ok: true, body: { getReader: () => ({
      read: async () => offset >= bytes.length ? { done: true } : { done: false, value: bytes.subarray(offset, offset = Math.min(offset + 7, bytes.length)) },
    }) } });
    const reply = await streaming.streamReply('test', [], { aborted: false }, tools);
    assert.equal(reply.tool_calls.length, finish === 'tool_calls' ? 1 : 0);
    if (finish === 'tool_calls') assert.equal(reply.tool_calls[0].function.arguments, '{"query":"sushi"}');
  }
  assert.equal(prefixCacheReads, 2, 'every reply refreshes the SSD prefix-cache meter');
  console.log('Web UI streaming: fragmented calls and token-limit truncation passed');
})().catch((error) => { console.error(error); process.exitCode = 1; });
const labelStart = script.indexOf('function toolCallLabel(');
vm.runInContext(script.slice(labelStart, script.indexOf('function appendMessageNode(', labelStart)), context);
assert.equal(context.toolCallLabel(history[1].tool_calls[0]), 'web_search · sushi');
assert.equal(context.toolCallLabel({ function: { name: 'fetch_url', arguments: '{"url":"https://example.com/article"}' } }), 'fetch_url · https://example.com/article');
assert.equal(context.toolCallLabel({ function: { name: 'web_search', arguments: '{"query":"one\\ntwo"}' } }), 'web_search · one two');
assert.equal(context.toolCallLabel({ function: { name: 'read_file', arguments: '{"path":"README.md"}' } }), 'read_file · README.md');
assert.equal(context.toolCallLabel({ function: { name: 'write_file', arguments: '{"path":"notes.md","content":"x"}' } }), 'write_file · notes.md');
assert.equal(context.toolCallLabel({ function: { name: 'edit_file', arguments: '{"path":"sub/a.md","old_string":"x"}' } }), 'edit_file · sub/a.md');
assert.equal(context.toolCallLabel({ function: { name: 'fetch_url', arguments: '{' } }), 'fetch_url');
assert.equal(context.toolCallLabel({ function: { name: 'web_search', arguments: 'null' } }), 'web_search');
console.log('Web UI tool labels: passed');
/* The pencil chip: per-chat state, reset to the --edit default when the folder changes. */
const editStart = script.indexOf('function chatDirectory(');
const editNodes = {};
const editEl = (id) => (editNodes[id] ??= { id, textContent: '', title: '', disabled: null, attrs: {}, setAttribute(k, v) { this.attrs[k] = v; } });
const editCtx = vm.createContext({
  $: editEl,
  toolsEnabled: true,
  chatAbort: null,
  currentChat: null,
  draftDirectory: '',
  draftEdit: false,
  editDefault: false,
  touchChat: () => {},
  callResearchTools: async () => ({ edit_default: true }),
});
vm.runInContext(script.slice(editStart, script.indexOf('let folderSelection', editStart)), editCtx);

(async () => {
  editCtx.currentChat = { id: 'c1', directory: '/a', edit: false };
  editCtx.renderDirectoryButton();
  assert.equal(editCtx.chatEdit(), false);
  assert.equal(editNodes.editButton.hidden, false, 'no server flag is needed to switch it');
  editNodes.editButton.onclick();
  assert.equal(editCtx.currentChat.edit, true);
  assert.equal(editNodes.editName.textContent, 'Edit on');
  assert.equal(editNodes.editButton.attrs['aria-pressed'], 'true');
  editCtx.setChatDirectory('/b');
  assert.equal(editCtx.currentChat.directory, '/b');
  assert.equal(editCtx.currentChat.edit, false, 'editing starts over from the default on another folder');

  editCtx.currentChat = null;
  editCtx.draftEdit = true;
  editNodes.editButton.onclick();
  assert.equal(editCtx.draftEdit, false, 'the pencil toggles the draft chat too');

  await editCtx.loadEditDefault();
  assert.equal(editCtx.editDefault, true);
  assert.equal(editCtx.draftEdit, true, '--edit on starts a new chat with editing on');
  editCtx.currentChat = { id: 'c2', directory: '/a', edit: false };
  editCtx.setChatDirectory('/c');
  assert.equal(editCtx.currentChat.edit, true);
  editNodes.editButton.onclick();
  assert.equal(editCtx.currentChat.edit, false, '--edit on is a default, not a lock');

  editCtx.toolsEnabled = false;
  editCtx.renderEditButton();
  assert.equal(editNodes.editButton.hidden, true, 'no tool pack, no edit chip');
  editCtx.toolsEnabled = true;
  editCtx.renderEditButton();
  assert.equal(editNodes.editButton.hidden, false);
  assert.equal(editNodes.editName.textContent, 'Edit off');
  editCtx.chatAbort = {};
  editCtx.renderEditButton();
  assert.equal(editNodes.editButton.disabled, true, 'fixed while a reply runs');
  console.log('Web UI edit chip: per-chat state, folder reset, --edit default passed');
})().catch((error) => { console.error(error); process.exit(1); });
