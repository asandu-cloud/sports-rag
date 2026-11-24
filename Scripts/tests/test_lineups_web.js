// Line-ups (D134, owner 25 Nov, version A): the match page's Line-ups section and the Research Area's Team news,
// from the lineups block of /api/research/matches/{event_id} (D135, behind SPIX_LINEUPS). Confirmed elevens are
// drawn on a pitch in their formation with the bench under it; until a team names its eleven, the players with most
// starts in its last 5, labelled as past starts, never as a prediction. The players are Crystal Palace's and Real
// Sociedad's from their matches of 21 and 22 Nov; the shirt numbers, grid and coach are test values.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const STATIC = path.join(__dirname, '../web_app/static');
const source = fs.readFileSync(path.join(STATIC, 'lineups.js'), 'utf8');
const css = fs.readFileSync(path.join(STATIC, 'lineups.css'), 'utf8');
const lu = Function(source + 'return { spixLineups, luShortName, luRows, luState, luRecheckIn };')();
const text = html => html.replace(/<[^>]+>/g, ' ').replace(/&#39;/g, "'").replace(/&amp;/g, '&').replace(/\s+/g, ' ')
  .replace(/ ([.,:])/g, '$1').trim();

const P = (player_id, name, position, number = null, grid = null) => ({player_id, name, position, number, grid});
const palaceXI = [P(174, 'D. Henderson', 'G', 1, '1:1'), P(310, 'Chris Richards', 'D', 26, '2:2'), P(371, 'Tyrick Mitchell', 'D', 3, '2:4'),
  P(500, 'Axel Disasi', 'D', 5, '2:1'), P(501, 'Anan Khalaili', 'D', 2, '2:3'), P(416, 'Adam Wharton', 'M', 20, '3:2'),
  P(94, 'Daichi Kamada', 'M', 18, '3:1'), P(502, 'Quinten Timber', 'M', 8, '3:3'), P(503, 'Dwight McNeil', 'M', 11, '4:1'),
  P(372, 'Yéremy Pino', 'M', 10, '4:2'), P(504, 'Jørgen Strand Larsen', 'F', 9, '5:1')];
const sociedadXI = [P(600, 'Álex Remiro', 'G', 1), P(601, 'Igor Zubeldia', 'D', 5), P(602, 'Mamadou Sarr', 'D', 4), P(603, 'Sergio Gómez', 'D', 17),
  P(604, 'Héctor Fort', 'D', 2), P(605, 'Gonçalo Guedes', 'M', 11), P(606, 'Luka Sučić', 'M', 24), P(607, 'Jon Gorrotxategi', 'M', 18),
  P(608, 'Carlos Soler', 'M', 8), P(609, 'Mikel Oyarzabal', 'F', 10), P(null, 'Orri Óskarsson', 'F', 9)];
const most = (prefix, names) => names.map(([name, position, starts], i) => ({player_id: prefix + i, name, position, starts_last_5: starts}));
const base = (lineups, status = 'NS') => ({
  fixture: {status, kickoff: '2026-11-26T20:00:00+00:00', home: {name: 'Crystal Palace'}, away: {name: 'Real Sociedad'}},
  teams: {home: {team: {name: 'Crystal Palace'}, shooters: most(700, [['Adam Wharton', 'M', 5], ['Chris Richards', 'D', 5]])},
    away: {team: {name: 'Real Sociedad'}, shooters: most(800, [['Mikel Oyarzabal', 'F', 4]])}},
  lineups});
const likely = {home: most(900, [['Adam Wharton', 'M', 5], ['D. Henderson', 'G', 4], ['Jaydee Canvot', 'D', 3]]),
  away: most(950, [['Álex Remiro', 'G', 4], ['Orri Óskarsson', 'F', 4]])};
const confirmed = base({status: 'confirmed', checked_at: '2026-11-26T19:24:00+00:00', likely,
  home: {formation: '4-3-2-1', coach: {name: 'Oliver Glasner'}, start_xi: palaceXI, bench: [P(505, 'Jaydee Canvot', 'D', 44), P(506, 'Eddie Nketiah', 'F', 14)]},
  away: {formation: '4-4-2', coach: 'Sergio Francisco', start_xi: sociedadXI, bench: []}});
const now = new Date('2026-11-26T19:30:00Z');

// Short names on the pitch: initials dropped; a long name keeps its last words.
assert.equal(lu.luShortName('D. Henderson'), 'Henderson');
assert.equal(lu.luShortName('Adam Wharton'), 'Adam Wharton');
assert.equal(lu.luShortName('Jørgen Strand Larsen'), 'Strand Larsen');
assert.equal(lu.luShortName('Jon Gorrotxategi'), 'Gorrotxategi');
assert.equal(lu.luShortName('Pedri'), 'Pedri');

// Pitch rows, keeper first: from API-Football's grid ("row:column") when every starter has one, in column order;
// otherwise by position, keeper, defenders, midfielders, forwards.
assert.deepEqual(lu.luRows(palaceXI).map(row => row.map(p => p.name.split(' ').pop())),
  [['Henderson'], ['Disasi', 'Richards', 'Khalaili', 'Mitchell'], ['Kamada', 'Wharton', 'Timber'], ['McNeil', 'Pino'], ['Larsen']]);
assert.deepEqual(lu.luRows(sociedadXI).map(row => row.length), [1, 4, 4, 2]);
assert.deepEqual(lu.luRows([P(1, 'A', 'G', 1, '1:1'), P(2, 'B', 'D', 2, null), P(3, 'C', null, 3, null)]).map(row => row.map(p => p.name)),
  [['A'], ['B'], ['C']], 'one starter without a grid: by position; an unknown position goes with the midfield');

// States. Both elevens named: confirmed. One named: partial. No block: the switch is off. Not out yet; unavailable;
// a match under way or over without elevens: missed.
assert.equal(lu.luState(confirmed).kind, 'confirmed');
assert.equal(lu.luState(base({...confirmed.lineups, away: {start_xi: [], bench: []}})).kind, 'partial');
assert.equal(lu.luState(base(null)).kind, 'off');
assert.equal(lu.luState(base({status: 'not_out_yet', checked_at: null, home: null, away: null})).kind, 'waiting');
assert.equal(lu.luState(base({status: 'unavailable', home: null, away: null})).kind, 'unavailable');
assert.equal(lu.luState(base(null, '1H')).kind, 'missed');
assert.equal(lu.luState(base({status: 'not_out_yet', home: null, away: null}, 'FT')).kind, 'missed');
assert.equal(lu.luState(base(confirmed.lineups, 'FT')).kind, 'confirmed', 'an eleven named before kickoff still shows');
// The match page knows no status (the board's card has none): with the switch off, the clock says whether it began;
// with it on, the backend's own "unavailable" covers a closed window.
const noStatus = lineups => ({...base(lineups), fixture: {...base(lineups).fixture, status: undefined}});
assert.equal(lu.luState(noStatus(null), new Date('2026-11-26T20:05:00Z')).kind, 'missed');
assert.equal(lu.luState(noStatus(null), new Date('2026-11-26T19:55:00Z')).kind, 'off');
assert.equal(lu.luState(noStatus({status: 'not_out_yet', home: null, away: null}), new Date('2026-11-26T20:05:00Z')).kind, 'waiting');

// Confirmed: the head, both teams on the pitch with formation, coach, numbers and bench.
const html = lu.spixLineups(confirmed, {title: '<h2 class="recs-title">Line-ups</h2>', now});
assert.match(html, /^<div class="lu"><div class="lu-head"><h2 class="recs-title">Line-ups<\/h2><span class="lu-chip ok">Confirmed<\/span><span class="lu-when">Checked \d\d:\d\d<\/span><\/div>/);
assert.match(text(html), /Both teams have named their starting eleven\./);
assert.match(text(html), /Crystal Palace 4-3-2-1 Coach Oliver Glasner/);
assert.match(text(html), /Real Sociedad 4-4-2 Coach Sergio Francisco/, 'a coach given as a plain name');
assert.match(html, /<div class="lu-pitch" role="list" aria-label="Crystal Palace starting eleven">/);
assert.equal((html.match(/class="lu-p" role="listitem"/g) || []).length, 22);
assert.match(html, /<span class="lu-dot" aria-hidden="true">1<\/span><a class="lu-link" href="\/app#research\/players\/174" data-research-route="players\/174" aria-label="D\. Henderson, number 1, goalkeeper">Henderson<\/a>/);
assert.match(html, /<span class="lu-name"><span aria-hidden="true">Óskarsson<\/span><span class="lu-sr">Orri Óskarsson, number 9, forward<\/span><\/span>/, 'no profile: plain text');
assert.match(text(html), /Bench Jaydee Canvot, Eddie Nketiah/);
assert.doesNotMatch(text(html), /Real Sociedad[^]*Bench/, 'an empty bench is left out');
assert.doesNotMatch(text(html), /not a prediction|Most starts/, 'no past-starts list beside two named elevens');

// Waiting: the past starts, keeper first, labelled; the note on when line-ups come and that they show here.
const waiting = lu.spixLineups(base({status: 'not_out_yet', checked_at: '2026-11-26T19:05:00+00:00', likely, home: null, away: null}), {title: '<h2>Line-ups</h2>', now});
assert.match(waiting, /<span class="lu-chip">Not out yet<\/span><span class="lu-when">Checked \d\d:\d\d<\/span>/);
assert.match(text(waiting), /Teams name their line-ups about 20 to 40 minutes before kickoff, and they show here\. Until then, the players who started most of each team's last 5 matches\. This is a record of past starts, not a prediction\./);
assert.match(text(waiting), /Crystal Palace Most starts, last 5 D\. Henderson Goalkeeper 4 of 5 Jaydee Canvot Defender 3 of 5 Adam Wharton Midfielder 5 of 5/);
assert.match(waiting, /<span class="lu-sq" role="img" aria-label="Started 4 of the last 5"><i class="on"><\/i><i class="on"><\/i><i class="on"><\/i><i class="on"><\/i><i><\/i><\/span>/);
assert.doesNotMatch(text(waiting), /Outfield players only/);

// Switch off: no promise that line-ups will show; the 8 outfield shooters when the block carries no list.
const off = lu.spixLineups(base(null), {title: '<h2>Line-ups</h2>', now});
assert.match(text(off), /Not out yet Teams name their line-ups about 20 to 40 minutes before kickoff\. Below, the players who started most of each team's last 5 matches\./);
assert.doesNotMatch(text(off), /they show here|Checked/);
assert.match(text(off), /Chris Richards Defender 5 of 5 Adam Wharton Midfielder 5 of 5 Outfield players only\./);

// Unavailable, partial and missed.
assert.match(text(lu.spixLineups(base({status: 'unavailable', home: null, away: null, likely}), {title: '', now})),
  /^Not available Spix's data provider has no line-ups for this match\. Below, the players who started most/);
const partial = lu.spixLineups(base({...confirmed.lineups, away: {start_xi: [], bench: []}}), {title: '', now});
assert.match(text(partial), /^One named Checked \d\d:\d\d Crystal Palace have named their starting eleven\. Real Sociedad have not named theirs yet: below, the players who started most of their last 5 matches, a record of past starts, not a prediction\./);
assert.match(partial, /aria-label="Crystal Palace starting eleven"/);
assert.match(text(partial), /Real Sociedad Most starts, last 5 Álex Remiro Goalkeeper 4 of 5/);
assert.match(text(lu.spixLineups(base(null, 'FT'), {title: '', now})), /^Not available Spix has no line-ups for this match\. Below, the players who started most of each team's last 5 matches before it\./);
assert.match(text(lu.spixLineups(base({status: 'not_out_yet', home: null, away: null, likely: {home: [], away: []}}), {title: '', now})),
  /Crystal Palace Most starts, last 5 No recent starts recorded\./);

// Escaping: names and the title slot's neighbours.
const odd = base({status: 'confirmed', home: {start_xi: [P(1, '<b>X</b>', 'G')], bench: []}, away: {start_xi: [P(2, 'Y & Z', 'G')], bench: []}});
assert.match(lu.spixLineups(odd, {title: '', now}), /&lt;b&gt;X&lt;\/b&gt;/);
assert.doesNotMatch(lu.spixLineups(odd, {title: '', now}), /<b>X/);

// When the match page asks again: every 5 minutes while an eleven is still to come, from 75 minutes before kickoff
// until 15 minutes after it; never once both are named, with the switch off, or for an unavailable match.
const at = iso => new Date(iso);
const wait = base({status: 'not_out_yet', home: null, away: null});
assert.equal(lu.luRecheckIn(wait, at('2026-11-26T19:30:00Z')), 300000);
assert.equal(lu.luRecheckIn(wait, at('2026-11-26T18:46:00Z')), 300000);
assert.equal(lu.luRecheckIn(wait, at('2026-11-26T18:40:00Z')), null, 'too early');
assert.equal(lu.luRecheckIn(wait, at('2026-11-26T20:16:00Z')), null, 'too late');
assert.equal(lu.luRecheckIn(base({...confirmed.lineups, away: {start_xi: [], bench: []}}), now), 300000);
assert.equal(lu.luRecheckIn(confirmed, now), null);
assert.equal(lu.luRecheckIn(base(null), now), null);
assert.equal(lu.luRecheckIn(base({status: 'unavailable', home: null, away: null}), now), null);

// The note on injuries sits under the past starts, not under two named elevens.
const NOTE = /<p class="lu-note">Injuries and suspensions are not in Spix&#39;s data yet\.<\/p><\/div>$/;
assert.match(waiting, NOTE);
assert.match(off, NOTE);
assert.doesNotMatch(html, /Injuries/);

// The match page (app.js): an empty box under the Match Read; on opening a match it asks the Research Area's route,
// shows the section once it answers (nothing on an error, for a preview card, or once the visitor has moved on),
// binds the names as the board's (D130: the profile's Back returns here) and asks again while an eleven may come.
const STATIC_FILE = file => fs.readFileSync(path.join(STATIC, file), 'utf8');
const app = STATIC_FILE('app.js'), page = STATIC_FILE('index.html');
assert.match(page, /<div id="matchReadSummary"><\/div>\n          <div id="matchLineups"><\/div>/);
assert.match(page, /<link rel="stylesheet" href="\/static\/lineups\.css\?v=lineups-20261125b">/);
assert.match(page, /<script src="\/static\/loaders\.js\?v=loaders-20261124a"><\/script>\n  <script src="\/static\/lineups\.js\?v=lineups-20261125b"><\/script>/);
assert.match(app, /    this\.switchTab\('match'\);\n    this\._loadLineups\(f\);\n  \},/);
const from = app.indexOf('  async _loadLineups(f) {');
const body = app.slice(from, app.indexOf('\n  },', from) + 4);
(async () => {
  const run = async ({ok = true, payload = confirmed, preview = false, leave = false, kit = true} = {}) => {
    const box = {innerHTML: 'old', dataset: {}}, calls = [], timers = [];
    const self = {currentMatch: null, _bindProfileLinks: el => calls.push(['bind', el === box])};
    const f = {id: '1636252', isPreview: preview, home: 'Crystal Palace', away: 'Real Sociedad', kickoff: '2026-11-26T20:00:00+00:00'};
    self.currentMatch = f;
    const win = kit ? {spixLineups: lu.spixLineups, luRecheckIn: (m, n) => lu.luRecheckIn(m, now)} : {};
    const router = {state: {tab: 'match'}};
    const fetch = async url => { calls.push(['fetch', url]); if (leave) self.currentMatch = {}; return {ok, json: async () => ({schema_version: 'research-lineups.v1', event_id: '1636252', lineups: payload.lineups})}; };
    const fn = Function('document', 'window', 'fetch', 'router', 'setTimeout', 'clearTimeout', `return ({${body}})._loadLineups`)(
      {getElementById: id => id === 'matchLineups' ? box : null}, win, fetch, router, (cb, ms) => { timers.push(ms); return 1; }, () => {});
    await fn.call(self, f);
    return {box, calls, timers};
  };
  let r = await run();
  // The light route (D135): the block alone; the names and kickoff come from the board's card.
  assert.deepEqual(r.calls, [['fetch', '/api/research/matches/1636252/lineups'], ['bind', true]]);
  assert.match(r.box.innerHTML, /aria-label="Crystal Palace starting eleven"[^]*aria-label="Real Sociedad starting eleven"/);
  assert.match(r.box.innerHTML, /^<div class="lu"><div class="lu-head"><h2 class="recs-title">Line-ups<\/h2><span class="lu-chip ok">Confirmed/);
  assert.deepEqual(r.timers, [], 'both named: no second ask');
  r = await run({payload: base({status: 'not_out_yet', home: null, away: null})});
  assert.deepEqual(r.timers, [300000]);
  r = await run({ok: false});
  assert.equal(r.box.innerHTML, '', 'an error shows nothing');
  r = await run({preview: true});
  assert.equal(r.box.innerHTML, '');
  assert.deepEqual(r.calls, []);
  r = await run({kit: false});
  assert.deepEqual(r.calls, [], 'without lineups.js, no request');
  r = await run({leave: true});
  assert.equal(r.box.innerHTML, '', 'the visitor opened another match meanwhile');

  // The Research Area: Team news draws the same section; the shadow root adopts lineups.css; a player's name opens
  // the profile inside the area.
  const research = STATIC_FILE('research/research.js');
  assert.match(research, /const kit = kitSheet\('\/static\/loaders\.css'\), lineupSheet = kitSheet\('\/static\/lineups\.css'\);\n    root\.adoptedStyleSheets = \[kit, lineupSheet\]\.filter\(Boolean\);\n    kitAdopted = Boolean\(kit\);/);
  assert.match(research, /\[data-add\],\[data-research-route\]'\);/);
  assert.match(research, /if \(target\.dataset\.researchRoute\) \{ event\.preventDefault\(\); const \[area, id\] = target\.dataset\.researchRoute\.split\('\/'\); return go\(area, id\); \}/);
  const {pathToFileURL} = require('node:url');
  const m = await import(pathToFileURL(path.join(STATIC, 'research/match.js')));
  globalThis.spixLineups = lu.spixLineups;
  const news = m.newsSection(base({status: 'not_out_yet', home: null, away: null, likely}));
  assert.match(news, /^<div class="lu"><div class="lu-head"><h2 class="cond">Team news<\/h2><span class="lu-chip">Not out yet<\/span>/);
  assert.match(m.newsSection(confirmed), /aria-label="Crystal Palace starting eleven"/);
  delete globalThis.spixLineups;
  assert.match(text(m.newsSection(base(null))), /^Team news Likely outfield starters/, 'without the kit, the old list');
  console.log('test_lineups_web: ok');
})().catch(error => { console.error(error); process.exit(1); });

// Styles for every part, a phone layout, and the dark theme comes from the page's tokens only.
for (const cls of ['lu-head', 'lu-chip', 'lu-when', 'lu-lead', 'lu-two', 'lu-team', 'lu-pitch', 'lu-row', 'lu-p', 'lu-dot', 'lu-bench', 'lu-list', 'lu-sq', 'lu-of', 'lu-note'])
  assert.match(css, new RegExp(`\\.${cls}[ ,.{:]`), cls);
assert.match(css, /@media \(max-width: 720px\)/);
assert.doesNotMatch(css, /#[0-9a-fA-F]{3,8}\b/, 'colours come from the tokens');
