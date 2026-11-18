// The Matchday fixture search (D126): matching and the result list, from app.js's pure helpers with synthetic
// schedule rows in the schedule route's shape. No browser, network or database.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../web_app/static/app.js'), 'utf8');
const esc = value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const helpersStart = source.indexOf('// ----- Pure render helpers');
const helpersEnd = source.indexOf('// ----- End of pure render helpers -----');
assert.ok(helpersStart >= 0 && helpersEnd > helpersStart);
const h = Function('esc', source.slice(helpersStart, helpersEnd)
  + 'return { fixtureSearchFold, fixtureSearch, renderFixtureSearch };')(esc);
const text = html => html.replace(/<\/?b>/g, '').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();

const LEAGUES = [
  { id: 'EPL', name: 'Premier League', logo: '/static/league-logos/39.png' },
  { id: 'Ligue1', name: 'Ligue 1', logo: '/static/league-logos/61.png' },
  { id: 'SuperLig', name: 'Süper Lig', logo: '/static/league-logos/203.png' },
  { id: 'UCL', name: 'Champions League', logo: '/static/league-logos/2.png' },
];
const row = (id, league, home, away, kickoff, status = 'NS') => ({
  fixture: { event_id: id, league, home_team: home, away_team: away, kickoff }, status,
  visuals: { home_team_logo: `https://media.example/${id}h.png`, away_team_logo: `https://media.example/${id}a.png` }, live_status: null,
});
const rows = [
  row('4', 'UCL', 'Manchester City', 'Napoli', '2026-11-24T20:00:00+00:00'),
  row('1', 'EPL', 'Manchester City', 'Fulham', '2026-11-21T12:30:00+00:00'),
  row('2', 'EPL', 'Liverpool', 'Manchester United', '2026-11-22T16:30:00+00:00'),
  row('3', 'Ligue1', 'Marseille', 'Le Mans', '2026-11-22T19:45:00+00:00'),
  row('5', 'SuperLig', 'Gençlerbirliği S.K.', 'Gaziantep FK', '2026-11-21T10:30:00+00:00'),
  row('6', 'EPL', 'Newcastle', 'Arsenal', '2026-11-21T17:30:00+00:00'),
  row('7', 'UCL', 'Arsenal', 'Borussia Dortmund', '2026-11-24T20:00:00+00:00'),
  row('8', 'EPL', 'Chelsea', 'Leeds', '2026-11-18T15:00:00+00:00', 'FT'),
  row('9', 'EPL', 'Chelsea', 'Everton', '2026-11-18T19:00:00+00:00', '1H'),
  row('10', 'EPL', 'Chelsea', 'Brentford', '2026-11-19T19:00:00+00:00', 'PST'),
  row('11', 'Ligue1', '1.FC <Test>', 'St. Pauli', '2026-11-19T18:00:00+00:00'),
];
const ids = hits => hits.map(hit => hit.id);
const search = q => h.fixtureSearch(rows, q, LEAGUES);

// Folding: accents and case do not matter.
assert.equal(h.fixtureSearchFold('Gençlerbirliği S.K.'), 'genclerbirligi s.k.');
assert.equal(h.fixtureSearchFold('İstanbul Başakşehir'), 'istanbul basaksehir');

// Each typed word matches the start of a word in the teams' or the league's name; every word must match.
assert.deepEqual(ids(search('man')), ['1', '2', '3', '4']);
assert.deepEqual(ids(search('MAN CITY')), ['1', '4']);
assert.deepEqual(ids(search('city man')), ['1', '4']);
assert.deepEqual(ids(search('arsenal dortmund')), ['7']);
assert.deepEqual(ids(search('genc')), ['5']);
assert.deepEqual(ids(search('süper')), ['5']);
assert.deepEqual(ids(search('ity')), [], 'a word matches the start of a name, not its middle');
assert.deepEqual(ids(search('fc')), ['11'], 'dots, hyphens and apostrophes split words');
assert.deepEqual(ids(search('pauli')), ['11']);
assert.deepEqual(ids(search('champions')), ['7', '4'], 'same kickoff: by home team');
assert.deepEqual(ids(search('')), []);
assert.deepEqual(ids(search('   ')), []);
assert.deepEqual(ids(search('celtic')), []);
// Only matches that have not started: finished, in play and postponed matches are left out.
assert.deepEqual(ids(search('chelsea')), []);
// Sorted by kickoff; a hit carries what the list and the board need.
assert.deepEqual(search('man city')[0], {
  id: '1', league: 'EPL', leagueName: 'Premier League', leagueLogo: '/static/league-logos/39.png',
  home: 'Manchester City', away: 'Fulham', kickoff: '2026-11-21T12:30:00+00:00',
  homeLogo: 'https://media.example/1h.png', awayLogo: 'https://media.example/1a.png', words: ['man', 'city'],
});
// An unknown league keeps its id as its name.
assert.equal(h.fixtureSearch([row('12', 'Allsvenskan', 'Malmo', 'AIK', '2026-11-21T12:00:00+00:00')], 'malmo', LEAGUES)[0].leagueName, 'Allsvenskan');

// The list: grouped by day, the typed starts in bold, the first row active, a count at the foot.
const now = new Date('2026-11-21T09:00:00+00:00');
const list = h.renderFixtureSearch(search('man'), 'man', { now });
assert.match(text(list), /^Today Manchester City v Fulham Premier League \d\d:\d\d Tomorrow Liverpool v Manchester United/);
assert.match(text(list), /Tuesday 24 Nov Manchester City v Napoli Champions League/);
assert.match(text(list), /4 matches in the next 7 days$/);
assert.match(list, /<b>Man<\/b>chester City/);
assert.match(list, /Le <b>Man<\/b>s/);
assert.equal(list.split('role="option"').length - 1, 4);
assert.match(list, /id="fixtureSearchOption0"[^>]*aria-selected="true"/);
assert.match(list, /id="fixtureSearchOption1"[^>]*aria-selected="false"/);
assert.match(list, /data-event-id="1"/);
assert.match(list, /data-kickoff="2026-11-21T12:30:00\+00:00"/);
assert.doesNotMatch(list, /fx-more/);
// The active row follows the arrow keys.
assert.match(h.renderFixtureSearch(search('man'), 'man', { now, active: 2 }), /id="fixtureSearchOption2"[^>]*aria-selected="true"/);
// Accents keep their own letters in bold.
assert.match(h.renderFixtureSearch(search('genc'), 'genc', { now }), /<b>Genç<\/b>lerbirliği/);
// One match reads "1 match".
assert.match(text(h.renderFixtureSearch(search('arsenal dortmund'), 'arsenal dortmund', { now })), /1 match in the next 7 days$/);
// Names are escaped.
const escaped = h.renderFixtureSearch(search('pauli'), 'pauli', { now });
assert.match(escaped, /1\.FC &lt;Test&gt;/);
assert.doesNotMatch(escaped, /<Test>/);
// No match: says so with the escaped query.
const none = h.renderFixtureSearch([], 'celtic <b>', { now });
assert.match(text(none), /^No match for “celtic &lt;b&gt;” in the next 7 days\.$/);
assert.doesNotMatch(none, /role="option"/);
// A long list shows the first 20 and says how many there are.
const many = Array.from({ length: 25 }, (_, i) => row(String(100 + i), 'EPL', `Team ${i}`, 'Rovers', `2026-11-22T${String(10 + (i % 10)).padStart(2, '0')}:00:00+00:00`));
const long = h.renderFixtureSearch(h.fixtureSearch(many, 'rovers', LEAGUES), 'rovers', { now });
assert.equal(long.split('role="option"').length - 1, 20);
assert.match(text(long), /Showing 20 of 25 matches\. Keep typing to narrow it\.$/);

// The loader: the 7 days' schedule once, when the box is first used, kept 10 minutes; the board refresh never calls it.
(async () => {
  const start = source.indexOf('  _fixtureSearchCache: null,');
  const end = source.indexOf('  _initFixtureSearch() {', start);
  assert.ok(start > 0 && end > start);
  let clock = Date.parse('2026-11-18T12:00:00Z');
  const DateStub = { now: () => clock, parse: Date.parse };
  const makeBoard = fetch => Function('fetch', 'Date', `return ({${source.slice(start, end)}
    _selectedDateISO(offset) { return '2026-11-' + String(18 + offset); } })`)(fetch, DateStub);
  const calls = [];
  const days = { '2026-11-21': [rows[1], rows[5]], '2026-11-22': [rows[2], rows[1]] };
  let failing = new Set();
  const fetch = async url => {
    calls.push(url);
    const day = url.split('/').pop();
    if (failing.has(day)) return { ok: false, status: 503 };
    return { ok: true, json: async () => ({ fixtures: days[day] || [] }) };
  };
  const board = makeBoard(fetch);
  const [first, second] = await Promise.all([board._fixtureSearchRows(), board._fixtureSearchRows()]);
  assert.equal(calls.length, 7, 'typing quickly shares one load');
  assert.deepEqual(calls, Array.from({ length: 7 }, (_, i) => `/api/match-reads/schedule/2026-11-${18 + i}`));
  assert.strictEqual(first, second);
  assert.deepEqual(first.map(r => r.fixture.event_id), ['1', '6', '2'], 'a match listed twice is kept once');
  clock += 9 * 60 * 1000;
  await board._fixtureSearchRows();
  assert.equal(calls.length, 7, 'kept for 10 minutes');
  clock += 2 * 60 * 1000;
  await board._fixtureSearchRows();
  assert.equal(calls.length, 14, 'loaded again after 10 minutes');
  // A day that fails: the rest is used, and kept for 1 minute only.
  const partial = makeBoard(fetch);
  failing = new Set(['2026-11-21']);
  calls.length = 0;
  assert.deepEqual((await partial._fixtureSearchRows()).map(r => r.fixture.event_id), ['2', '1']);
  clock += 30 * 1000;
  await partial._fixtureSearchRows();
  assert.equal(calls.length, 7);
  clock += 31 * 1000;
  await partial._fixtureSearchRows();
  assert.equal(calls.length, 14);
  // Every day failing: a plain message, nothing kept.
  const down = makeBoard(async () => ({ ok: false, status: 503 }));
  await assert.rejects(down._fixtureSearchRows(), /The matches could not be loaded\. Please try again shortly\./);
  assert.equal(down._fixtureSearchCache, null);
  // Only the box calls it: the board's own loading never does.
  const boardLoadStart = source.indexOf('  async _loadMatchday(');
  const boardLoad = source.slice(boardLoadStart, source.indexOf('  _renderFixtures(', boardLoadStart));
  assert.ok(boardLoad.length > 1000);
  assert.doesNotMatch(boardLoad, /_fixtureSearchRows/);
  console.log('fixture search web tests passed');
})().catch(error => { console.error(error); process.exit(1); });
