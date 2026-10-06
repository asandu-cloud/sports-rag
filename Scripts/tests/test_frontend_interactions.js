// Synthetic presentation tests: no provider calls, outcomes or publication.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('Scripts/web_app/static/app.js', 'utf8');
let now = 0, nextId = 0;
const frames = new Map();
const preference = { matches: false, addEventListener() {} };
const context = vm.createContext({
  window: { matchMedia: () => preference }, performance: { now: () => now },
  requestAnimationFrame: fn => { frames.set(++nextId, fn); return nextId; },
  cancelAnimationFrame: id => frames.delete(id),
});
vm.runInContext(source.slice(source.indexOf('const ui = {'), source.indexOf("document.addEventListener('keydown'")) + '\nglobalThis.testUI = ui;', context);
const spring = context.testUI.spring(() => {});
function step() { now += 16; const current = [...frames.values()]; frames.clear(); current.forEach(fn => fn(now)); }
spring.set(1);
for (let i = 0; i < 8; i++) step();
const position = spring.value, velocity = spring.velocity;
assert.ok(position > 0 && position < 1);
spring.set(0);
assert.equal(spring.value, position, 'Retargeting must start from the displayed position');
assert.equal(spring.velocity, velocity, 'Retargeting must preserve velocity');
for (let i = 0; i < 70; i++) step();
assert.equal(spring.value, 0); assert.equal(frames.size, 0);
preference.matches = true; spring.set(1);
assert.equal(spring.value, 1); assert.equal(frames.size, 0);
console.log('Spring reversal, settling and reduced-motion checks passed.');

(async () => {
  const elements = new Map();
  const document = { getElementById(id) {
    if (!elements.has(id)) elements.set(id, { setAttribute(key, value) { this[key] = value; } });
    return elements.get(id);
  } };
  let releaseFirst, firstReached;
  const firstPending = new Promise(resolve => { firstReached = resolve; });
  const fetch = async url => {
    if (url.includes('/best/first')) {
      firstReached();
      return new Promise(resolve => { releaseFirst = () => resolve({ ok: true, json: async () => ({ cards: [{ id: 'old' }] }) }); });
    }
    return { ok: true, json: async () => ({ cards: [{ id: url.endsWith('/first') ? 'old' : 'new' }] }) };
  };
  const start = source.indexOf('  _normaliseScheduledFixture('), end = source.indexOf('  _renderFixtures(', start);
  const board = Function('document', 'fetch', 'ui', 'showToast', `return ({${source.slice(start, end)}})`)(document, fetch, { enter() {} }, () => {});
  let renders = 0;
  Object.assign(board, {
    _requestId: 0, activeLeague: 'all', LEAGUES: [{ id: 'league', name: 'League' }], fixtures: [], bestBets: [], fixtureIndex: {},
    date: 'first', _selectedDateISO() { return this.date; },
    _normaliseMatchReadCard: card => card,
    _indexFixtures(fixtures) { fixtures.forEach(f => { this.fixtureIndex[f.id] = f; }); },
    _renderFixtures() { renders++; }, _renderBestBets() { renders++; },
  });
  const older = board._loadMatchday();
  await firstPending;
  board.date = 'second';
  await board._loadMatchday();
  assert.equal(board.fixtures[0].id, 'new');
  releaseFirst(); await older;
  assert.equal(board.fixtures[0].id, 'new');
  assert.equal(board.bestBets[0].id, 'new', 'Slow shortlist request must not overwrite the newly selected day');
  assert.equal(board.fixtureIndex.old, undefined);
  assert.equal(elements.get('fixturesContainer')['aria-busy'], 'false');
  const before = renders;
  await board._loadMatchday({ background: true });
  assert.equal(renders, before, 'Unchanged polling must retain the DOM and focus');
  console.log('Overlapping day requests and non-disruptive background refresh checks passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
