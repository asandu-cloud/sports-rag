// Research Area landing (D111, owner 16 Nov): the Recently viewed row has a Clear button. It empties the list for the
// tab in view (players, teams or referees) and the row goes back to the Try suggestions. research.js runs in the
// browser only, so its wiring is pinned from the source, as test_research_upcoming_web.js does for app.js.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const static_ = file => fs.readFileSync(path.join(__dirname, '../web_app/static', file), 'utf8');
const js = static_('research/research.js'), css = static_('research/research.css');

// The row: the chips, then Clear, named for the tab it empties.
assert.match(js, /<span>Recently viewed<\/span>\$\{recent\.map\([^\n]*\.join\(''\)\}<button type="button" class="clear-recent" data-clear-recent aria-label="Clear recently viewed \$\{state\.area\}">Clear<\/button><\/div>`/);

// The click: only this tab's list goes, the page redraws, and focus lands on the first suggestion that replaces it.
assert.match(js, /\n  function clearRecent\(\) \{\n    const all = store\.get\('spix-research-recent', \{\}\);\n    delete all\[state\.area\];\n    store\.set\('spix-research-recent', all\);\n    renderHome\('\[data-suggest\]'\);\n  \}/);
assert.match(js, /\$\('researchHome'\)\.addEventListener\('click', event => \{\n    if \(event\.target\.closest\('\[data-clear-recent\]'\)\) return clearRecent\(\);/);

// A quiet text button, as tall as the chips beside it.
assert.match(css, /\.recent \.clear-recent\{[^}]*min-height:36px[^}]*background:none[^}]*text-decoration:underline/);
assert.match(css, /\.recent \.clear-recent:hover\{color:var\(--accent-l\)\}/);

console.log('test_research_home_web: ok');
