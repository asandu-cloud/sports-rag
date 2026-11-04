// Before-kickoff cards on the board (D71): a started match keeps the picks Spix had for it before kickoff,
// shown quietly after the upcoming matches in its league, with no price, Add button or link. Rendered from
// app.js's pure helpers and the schedule merge with synthetic before-kickoff.v1 entries. No browser or network.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../web_app/static/app.js'), 'utf8');
const esc = value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const helpersStart = source.indexOf('// ----- Pure render helpers');
const helpersEnd = source.indexOf('// ----- End of pure render helpers -----');
assert.ok(helpersStart >= 0 && helpersEnd > helpersStart);
const h = Function('esc', source.slice(helpersStart, helpersEnd)
  + 'return { liveStatusLabel, beforeKickoffFor, renderBeforeKickoffCard, splitStartedFixtures };')(esc);
const text = html => html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();

// The match status: live codes, breaks, finishes and the rare ones; a late start is not live;
// null, a missing status or an unknown code says when it kicked off.
const labels = {
  '1H': 'Live', '2H': 'Live', ET: 'Live', BT: 'Live', P: 'Live', INT: 'Live', LIVE: 'Live',
  HT: 'Half time', FT: 'Full time', AET: 'Full time', PEN: 'Full time',
  PST: 'Postponed', SUSP: 'Suspended', ABD: 'Abandoned', CANC: 'Cancelled', AWD: 'Match awarded', WO: 'Walkover',
  NS: 'Not started yet', TBD: 'Not started yet',
};
for (const [short, label] of Object.entries(labels)) assert.equal(h.liveStatusLabel({ short, elapsed: 67 }, '19:45'), label, short);
assert.equal(h.liveStatusLabel({ short: 'ht' }, '19:45'), 'Half time');
assert.equal(h.liveStatusLabel(null, '19:45'), 'Kicked off 19:45');
assert.equal(h.liveStatusLabel(undefined, '19:45'), 'Kicked off 19:45');
assert.equal(h.liveStatusLabel({ short: 'XYZ' }, '19:45'), 'Kicked off 19:45');

const past = new Date(Date.now() - 90 * 60e3).toISOString();
const future = new Date(Date.now() + 90 * 60e3).toISOString();
const pick = (market, side, line, label, chance) => ({ market, side, line, label, chance });
const entry = (id, league, extra = {}) => ({
  version: 'before-kickoff.v1',
  fixture: { event_id: id, league, home_team: 'Celtic', away_team: 'Beşiktaş', kickoff: past },
  visuals: { home_team_logo: 'https://media.api-sports.io/football/teams/247.png', away_team_logo: '' },
  read_at: '2026-11-04T19:34:00+00:00',
  most_likely: [pick('corners', 'over', 8.5, 'Over 8.5 corners', 0.72), pick('spreads', 'away', 0.5, 'Beşiktaş +0.5', 0.71),
    pick('btts', 'yes', null, 'Both teams to score', 0.64), pick('goals', 'over', 1.5, 'Over 1.5 goals', 0.8)],
  // Prices must never reach the card, even if a field carrying one turns up.
  value: [{ ...pick('goals', 'over', 2.5, 'Over 2.5 goals', 0.62), typical_price: 1.71, worth_it_at: 1.65 },
    pick('corners', 'under', 10.5, 'Under 10.5 corners', 0.55), pick('goals', 'under', 3.5, 'Under 3.5 goals', 0.6)],
  status: null,
  ...extra,
});

// The entry: its version, ?recs=off, at most 3 Most likely and 2 Value, unusable items dropped.
const bk = h.beforeKickoffFor(entry('901', 'UEL'), '', 'UEL');
assert.equal(bk.id, '901');
assert.equal(bk.league, 'UEL');
assert.deepEqual(bk.most_likely.map(o => o.label), ['Over 8.5 corners', 'Beşiktaş +0.5', 'Both teams to score']);
assert.deepEqual(bk.value.map(o => o.label), ['Over 2.5 goals', 'Under 10.5 corners']);
assert.equal(h.beforeKickoffFor(entry('901', 'UEL'), '?recs=off', 'UEL'), null);
assert.equal(h.beforeKickoffFor({ ...entry('901', 'UEL'), version: 'before-kickoff.v2' }, '', 'UEL'), null);
assert.equal(h.beforeKickoffFor(null, '', 'UEL'), null);
assert.equal(h.beforeKickoffFor(entry('902', 'UEL', { most_likely: [], value: [{ label: 'Over 2.5 goals', chance: null }] }), '', 'UEL'), null);
assert.equal(h.beforeKickoffFor(entry('903', 'UEL', { fixture: { league: 'UEL' } }), '', 'UEL'), null, 'no event id, nothing to join');

// The card: status, teams, both lists with chances in words, when the picks were checked; nothing to bet with.
const card = h.renderBeforeKickoffCard({ time: '22:00', home: 'Celtic', away: 'Beşiktaş', scheduleLabel: 'Live', beforeKickoff: bk },
  '<span class="match-card-emblem"></span>', '<span class="match-card-emblem"></span>');
const words = text(card);
assert.match(words, /22:00 Live/);
assert.match(words, /Celtic v Beşiktaş/);
assert.match(words, /Most likely, before kickoff Over 8\.5 corners about 3 in 4 · 72%/);
assert.match(words, /Value, before kickoff Over 2\.5 goals about 3 in 5 · 62%/);
assert.match(words, /Last checked at \d\d:\d\d, before kickoff\./);
for (const banned of ['1.71', '1.65', 'Worth it', 'data-recs-add', '<button', 'href=', 'Add', 'View match', 'Hit', 'Miss'])
  assert.ok(!card.includes(banned), `the card must not carry ${banned}`);
const noValue = h.renderBeforeKickoffCard({ time: '22:00', home: 'A', away: 'B', scheduleLabel: 'Full time',
  beforeKickoff: h.beforeKickoffFor(entry('904', 'UEL', { value: [] }), '', 'UEL') }, '', '');
assert.ok(!noValue.includes('Value, before kickoff'));
const nasty = h.renderBeforeKickoffCard({ time: '22:00', home: '<b>A</b>', away: 'B', scheduleLabel: 'Live',
  beforeKickoff: h.beforeKickoffFor(entry('905', 'UEL', { most_likely: [pick('goals', 'over', 2.5, '<img src=x onerror=alert(1)>', 0.6)] }), '', 'UEL') }, '', '');
assert.ok(!nasty.includes('<img') && !nasty.includes('<b>A'), 'labels and team names are escaped');

// A started match with no picks sits in the same quiet card, saying so, with nothing to bet with either.
const empty = h.renderBeforeKickoffCard({ time: '21:00', home: 'Anderlecht', away: 'Salzburg', scheduleLabel: 'Full time',
  beforeKickoff: null, scheduleNote: 'Spix had no picks for this match before kickoff.' }, '', '');
assert.match(text(empty), /21:00 Full time Anderlecht v Salzburg Spix had no picks for this match before kickoff\./);
assert.ok(!empty.includes('<button') && !empty.includes('before kickoff</span>'));

// The merge: rows joined by event id.
const from = source.indexOf('  _normaliseScheduledFixture(row, analysisAvailable) {');
const to = source.indexOf('\n  async _loadMatchday(', from);
assert.ok(from >= 0 && to > from);
const board = Function('liveStatusLabel', `return ({${source.slice(from, to)}})`)(h.liveStatusLabel);
board._formatFixtureTime = () => '19:45';
const row = (id, league, kickoff, status = 'NS', extra = {}) => ({
  fixture: { event_id: id, league, home_team: `Home ${id}`, away_team: `Away ${id}`, kickoff }, status, visuals: {}, ...extra });
const schedule = [
  row('901', 'UEL', past),                       // had picks before kickoff
  row('906', 'UEL', past),                       // started, no picks
  row('907', 'UEL', future),                     // upcoming
  row('908', 'UECL', past),                      // started, in a league whose list is empty
  row('909', 'UEL', past, 'PST'),                // postponed: not a started match
];
const leagues = new Set(['UEL', 'UECL']);
const byId = list => Object.fromEntries(list.map(f => [f.id, f]));
const today = board._mergeSchedule(schedule, [], leagues);
const picks = new Map([['901', bk]]);
const merged = byId(board._mergeSchedule(schedule, [], leagues, picks));

assert.deepEqual(merged['901'].beforeKickoff, bk);
assert.equal(merged['901'].afterKickoff, true);
assert.equal(merged['901'].scheduleLabel, 'Kicked off 19:45');
assert.equal(merged['906'].beforeKickoff, null);
assert.equal(merged['906'].afterKickoff, true);
assert.equal(merged['906'].scheduleNote, 'Spix had no picks for this match before kickoff.');
assert.equal(merged['906'].scheduleLabel, 'Kicked off 19:45');
assert.deepEqual(merged['907'], byId(today)['907'], 'an upcoming match is untouched');
assert.deepEqual(merged['908'], byId(today)['908'], 'a league with no before-kickoff entries looks as it does today');
assert.deepEqual(merged['909'], byId(today)['909'], 'a postponed match is not a started one');

// A missing or empty list (switch off, ?recs=off) gives exactly today's board, old sentence included.
assert.deepEqual(board._mergeSchedule(schedule, [], leagues, new Map()), today);
assert.equal(byId(today)['906'].scheduleNote, 'No pre-match recommendation is available for this fixture.');
assert.equal(today.some(f => f.afterKickoff || f.beforeKickoff), false);

// The live status: the entry's own, else the schedule row's live_status; null and a missing key alike.
const live = new Map([['901', h.beforeKickoffFor(entry('901', 'UEL', { status: { short: 'HT', elapsed: 45 } }), '', 'UEL')]]);
const withLive = byId(board._mergeSchedule([
  row('901', 'UEL', past), row('906', 'UEL', past, 'NS', { live_status: { short: '2H', elapsed: 67 } }),
  row('910', 'UEL', past, 'NS', { live_status: null }), row('911', 'UEL', past, 'NS', { live_status: { short: 'NS' } }),
], [], leagues, live));
assert.equal(withLive['901'].scheduleLabel, 'Half time');
assert.equal(withLive['906'].scheduleLabel, 'Live');
assert.equal(withLive['910'].scheduleLabel, 'Kicked off 19:45');
assert.equal(withLive['911'].scheduleLabel, 'Not started yet');

// Placement: upcoming first in kickoff order, then the started ones; unchanged when nothing has started.
const order = h.splitStartedFixtures(board._mergeSchedule(schedule, [], leagues, picks).filter(f => f.league === 'UEL'));
assert.deepEqual(order.upcoming.map(f => f.id), ['909', '907'], 'kickoff order, as today');
assert.deepEqual(order.started.map(f => f.id), ['901', '906']);
const plain = h.splitStartedFixtures(today);
assert.deepEqual(plain.upcoming.map(f => f.id), today.map(f => f.id));
assert.deepEqual(plain.started, []);

console.log('Before-kickoff card, join by id, placement, live status and switched-off board checks passed.');
