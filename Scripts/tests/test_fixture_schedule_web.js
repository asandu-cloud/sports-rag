// Synthetic browser data only: pending fixtures must never acquire betting actions.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const source = fs.readFileSync('Scripts/web_app/static/app.js', 'utf8');
const start = source.indexOf('  _normaliseScheduledFixture(');
const end = source.indexOf('  _renderFixtures(', start);
const now = Date.parse('2026-10-06T13:00:00Z');
const DateStub = { now: () => now, parse: Date.parse };
const row = (id, status = 'NS') => ({
  fixture: { event_id: id, league: 'EPL', home_team: '<Home>', away_team: 'Away', kickoff: '2026-10-10T17:00:00Z' },
  status, odds: 10, selections: [{ pick: 'unsafe' }], visuals: {},
});
function makeBoard(fetch) {
  const elements = new Map();
  const document = { getElementById(id) {
    if (!elements.has(id)) elements.set(id, { setAttribute() {} });
    return elements.get(id);
  } };
  const board = Function('document', 'fetch', 'ui', 'showToast', 'Date', `return ({${source.slice(start, end)}})`)(
    document, fetch, { enter() {} }, () => {}, DateStub);
  Object.assign(board, {
    _requestId: 0, activeLeague: 'all', LEAGUES: [{ id: 'EPL', name: 'Premier League' }],
    fixtures: [], bestBets: [], fixtureIndex: {},
    _selectedDateISO: () => '2026-10-10', _formatFixtureTime: () => '18:00',
    _normaliseMatchReadCard: card => card,
    _indexFixtures(items) { items.forEach(f => { this.fixtureIndex[f.id] = f; }); },
    _renderFixtures(state) { this.boardState = state; },
    _renderBestBets(state) { this.bestState = state; },
    _loadLegacyPreview() { throw Error('Never calculate predictions to fill a schedule'); },
  });
  return board;
}
(async () => {
  const published = { id: 'ready', league: 'EPL', kickoff: row('ready').fixture.kickoff,
    isMatchRead: true, status: 'recommended', best: true, odds: 2.075, selections: [{ recommendation_id: 'original-id' }] };
  const noBet = { ...published, id: 'no-bet', status: 'no_bet', best: false, odds: null, selections: [] };
  const board = makeBoard();
  const merged = board._mergeSchedule([row('ready'), row('pending'), row('no-bet')], [published, noBet], new Set(['EPL']));
  assert.equal(merged.length, 3);
  assert.equal(merged.find(f => f.id === 'ready'), published, 'Preserve exact published identity and price');
  assert.equal(merged.find(f => f.id === 'no-bet'), noBet, 'A no-bet decision is not pending analysis');
  const pending = merged.find(f => f.id === 'pending');
  assert.equal(pending.scheduleLabel, 'Analysis pending');
  assert.equal(pending.odds, null); assert.equal(pending.best, false); assert.deepEqual(pending.selections, []);
  assert.equal(board._mergeSchedule([row('ready', 'PST')], [published], new Set(['EPL']))[0].scheduleLabel, 'Postponed');
  assert.equal(board._mergeSchedule([row('ready', 'CANC')], [published], new Set(['EPL']))[0].isScheduleOnly, true);
  assert.equal(board._mergeSchedule([row('pending')], [], new Set())[0].scheduleLabel, 'Analysis temporarily unavailable');
  assert.equal(board._normaliseScheduledFixture({ ...row('started'), fixture: { ...row('started').fixture, kickoff: '2026-10-06T12:00:00Z' } }, true).scheduleLabel, 'Match started');
  assert.equal(board._mergeSchedule([], [published], new Set())[0], published);

  const requests = [];
  const pendingBoard = makeBoard(async url => {
    requests.push(url);
    return { ok: true, json: async () => url.includes('/schedule/') ? { fixtures: [row('pending')] } : { cards: [] } };
  });
  await pendingBoard._loadMatchday();
  assert.equal(pendingBoard.fixtures[0].scheduleLabel, 'Analysis pending');
  assert.deepEqual(pendingBoard.bestBets, []);
  assert.ok(requests.every(url => url.startsWith('/api/match-reads/')));
  assert.ok(!requests.some(url => url.includes('/best/')), 'Pending rows must not enter recommendation ranking');

  const unavailable = makeBoard(async url => ({ ok: url.includes('/schedule/'), json: async () => ({ fixtures: [row('pending')] }) }));
  await unavailable._loadMatchday();
  assert.equal(unavailable.fixtures[0].scheduleLabel, 'Analysis temporarily unavailable');
  assert.match(unavailable.boardState.notice, /could not load its analyses/);

  const publishedOnly = makeBoard(async url => ({ ok: !url.includes('/schedule/'), json: async () => ({ cards: [published] }) }));
  await publishedOnly._loadMatchday();
  assert.deepEqual(publishedOnly.fixtures, [published]);
  assert.deepEqual(publishedOnly.bestBets, [published]);
  assert.match(publishedOnly.boardState.notice, /full fixture schedule could not be loaded/);

  const down = makeBoard(async () => ({ ok: false }));
  await down._loadMatchday();
  assert.match(down.boardState.error, /temporarily unavailable/);
  assert.deepEqual(down.bestBets, []);
  console.log('Schedule merge, pending/no-bet separation, unchanged publication identity, withdrawal status and failure-path checks passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
