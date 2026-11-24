// Back to Matchday (D133, owner 24 Nov): a Research Area profile opened from a board card's name (D130) has a back
// button that returns to the board as it was: the browser's own Back, so the date, league tab, search text and scroll
// come back with it. A profile reached any other way keeps "Find another team". research.js runs in the browser only,
// so its wiring is pinned from the source; the router's back step runs here on a fake history.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const static_ = file => fs.readFileSync(path.join(__dirname, '../web_app/static', file), 'utf8');
const app = static_('app.js'), js = static_('research/research.js');

// The board's name links open the profile marked as from the board, in the history entry itself, so Back and
// Forward bring the mark back with the page.
assert.match(app, /router\.go\('app', 'research', null, link\.dataset\.researchRoute, \{ fromBoard: true \}\);/);
assert.match(app, /  go\(page, tab = 'fixtures', restored = null, researchRoute = null, extra = \{\}\) \{/);
assert.match(app, /researchRoute: tab === 'research' \? researchRoute \|\| appModule\.researchRoute \|\| 'players' : null, \.\.\.extra \};/);
// The Research Area is told which way back leads; a research link inside the area never carries the mark.
assert.match(app, /document\.querySelector\('spix-research'\)\.showRoute\(next\.researchRoute \|\| 'players', \{ backToBoard: Boolean\(next\.fromBoard\) \}\)/);
assert.match(app, /document\.addEventListener\('research-navigate', event => \{\n  router\.go\('app', 'research', null, event\.detail\.route\);\n\}\);/);
assert.match(app, /document\.addEventListener\('research-back', \(\) => router\.backToBoard\(\)\);/);

// research.js: the profile's back button reads "Back to Matchday" and asks the page to go back; otherwise it keeps
// "Find another …" and opens the area's search.
assert.match(js, /  async showRoute\(route, options\) \{ await this\.ready; this\.controller\.showRoute\(route, options\); \}/);
assert.match(js, /mountResearch\(root, route => this\.dispatchEvent\(new CustomEvent\('research-navigate', \{bubbles: true, composed: true, detail: \{route\}\}\)\),\n        \(\) => this\.dispatchEvent\(new CustomEvent\('research-back', \{bubbles: true, composed: true\}\)\)\)/);
assert.match(js, /  async function fromRoute\(route = 'players', \{backToBoard = false\} = \{\}\) \{\n    state\.backToBoard = backToBoard;/);
assert.match(js, /\$\('profileBack'\)\.addEventListener\('click', \(\) => \(state\.backToBoard \? back\(\) : go\(state\.area\)\)\);/);
assert.match(js, /\$\('profileBackLabel'\)\.textContent = state\.backToBoard \? 'Back to Matchday' : `Find another \$\{current\(\)\.singular\}`;/);

// The router's back step: history.back() when the page came from the board (match page or profile), else the board.
const from = app.indexOf('  backToBoard() {');
const body = app.slice(from, app.indexOf('\n  },', from) + 4);
const run = state => {
  const calls = [];
  const router = Function('history', `return ({ state: null, go(...args) { this.went = args; },\n${body}\n})`)({ back: () => calls.push('back') });
  router.state = state;
  router.backToBoard();
  return calls.length ? 'history.back' : router.went?.join(',');
};
assert.equal(run({ tab: 'research', fromBoard: true }), 'history.back');
assert.equal(run({ tab: 'match', fromBoard: true }), 'history.back');
assert.equal(run({ tab: 'research' }), 'app,fixtures', 'a direct visit opens the board instead');
assert.equal(run({ tab: 'match' }), 'app,fixtures');
console.log('test_board_back_web: ok');
