// Board links (D130): a board card's team names and referee open their Research Area profiles, from app.js's pure
// helpers and the board's normalisers. A name is a link only when the board's data names its profile.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../web_app/static/app.js'), 'utf8');
const esc = value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const helpersStart = source.indexOf('// ----- Pure render helpers');
const helpersEnd = source.indexOf('// ----- End of pure render helpers -----');
assert.ok(helpersStart >= 0 && helpersEnd > helpersStart);
const h = Function('esc', source.slice(helpersStart, helpersEnd)
  + 'return { boardTeamRoute, boardRefereeRoute, boardProfileLink, renderRecsBoardCard, renderBeforeKickoffCard };')(esc);
const links = html => [...html.matchAll(/<a class="board-profile-link" href="([^"]+)" data-research-route="([^"]+)"[^>]*>([\s\S]*?)<\/a>/g)]
  .map(match => ({ href: match[1], route: match[2], text: match[3] }));

// Routes: a team by its Research Area id, a referee by its key; anything else has none.
assert.equal(h.boardTeamRoute(21), 'teams/21');
assert.equal(h.boardTeamRoute('210'), 'teams/210');
for (const none of [null, undefined, '', 0, -3, 'abc', 1.5, '21x']) assert.equal(h.boardTeamRoute(none), null, String(none));
assert.equal(h.boardRefereeRoute({ name: 'Nick Walsh', key: 'n-walsh-fb1b59d7' }), 'referees/n-walsh-fb1b59d7');
assert.equal(h.boardRefereeRoute({ name: 'A B', key: 'a b/"c' }), 'referees/a%20b%2F%22c', 'the key is encoded');
assert.equal(h.boardRefereeRoute({ name: 'Nick Walsh', key: null }), null);
assert.equal(h.boardRefereeRoute(null), null);
assert.equal(h.boardRefereeRoute(undefined), null);

// The link: a real address that opens in a new tab too, the name kept, and the profile in its accessible name.
assert.equal(h.boardProfileLink('Barcelona', 'teams/21', 'Barcelona'),
  '<a class="board-profile-link" href="/app#research/teams/21" data-research-route="teams/21" aria-label="Barcelona: Research Area profile">Barcelona</a>');
assert.equal(h.boardProfileLink('Barcelona', null, 'Barcelona'), 'Barcelona');
assert.match(h.boardProfileLink('&lt;b&gt;', 'teams/1', '<b>'), /aria-label="&lt;b&gt;: Research Area profile">&lt;b&gt;<\/a>$/);

// The board card: both teams and the referee, when the data names them.
const block = {
  most_likely: [{ label: 'Barcelona win', chance: 0.81, typical_price: 1.2, market: 'result', side: 'away' }], value: [],
  explore: [], markets: [{ market: 'result', options: [{ label: 'Barcelona win', typical_price: 1.2 }] }],
};
const fixture = { id: '1635718', time: '18:45', home: 'Sabah FA', away: 'Barcelona', homeId: 210, awayId: 21,
  referee: { name: 'Nick Walsh', key: 'n-walsh-fb1b59d7' }, recommendations: block };
const full = links(h.renderRecsBoardCard(fixture, '', ''));
assert.deepEqual(full.map(link => link.route), ['teams/210', 'teams/21', 'referees/n-walsh-fb1b59d7']);
assert.deepEqual(full.map(link => link.text), ['Sabah FA', 'Barcelona', '<span class="recs-ref-name">Nick Walsh</span>']);
assert.equal(full[0].href, '/app#research/teams/210');
// Until the board's data carries the team ids, the names stay plain text and the referee still links.
const noIds = h.renderRecsBoardCard({ ...fixture, homeId: undefined, awayId: undefined }, '', '');
assert.deepEqual(links(noIds).map(link => link.route), ['referees/n-walsh-fb1b59d7']);
assert.match(noIds, /<span>Sabah FA<\/span> <span class="recs-board-v">v<\/span> <span>Barcelona<\/span>/);
// TBD, a referee with no key, and a board whose data has no referee field are never links.
assert.deepEqual(links(h.renderRecsBoardCard({ ...fixture, homeId: null, awayId: null, referee: null }, '', '')), []);
assert.deepEqual(links(h.renderRecsBoardCard({ ...fixture, homeId: null, awayId: null, referee: { name: 'Nick Walsh', key: null } }, '', '')), []);
assert.deepEqual(links(h.renderRecsBoardCard({ ...fixture, homeId: null, awayId: null, referee: undefined }, '', '')), []);
// One team without an id: only the other links.
assert.deepEqual(links(h.renderRecsBoardCard({ ...fixture, homeId: null, referee: null }, '', '')).map(link => link.route), ['teams/21']);
// Names are escaped inside the link.
const hostile = h.renderRecsBoardCard({ ...fixture, home: '<img src=x>', referee: { name: '<b>R</b>', key: 'r' } }, '', '');
assert.doesNotMatch(hostile, /<img src=x>|<b>R/);

// The Live tab's card links its teams the same way, and still carries nothing to bet with.
const live = h.renderBeforeKickoffCard({ time: '18:45', home: 'Sabah FA', away: 'Barcelona', homeId: 210, awayId: 21,
  scheduleLabel: 'Live', beforeKickoff: null, scheduleNote: 'Spix had no picks for this match before kickoff.' }, '', '');
assert.deepEqual(links(live).map(link => link.route), ['teams/210', 'teams/21']);
assert.ok(!live.includes('<button'));
assert.deepEqual(links(h.renderBeforeKickoffCard({ time: '18:45', home: 'A', away: 'B', scheduleLabel: 'Live',
  beforeKickoff: null, scheduleNote: '' }, '', '')), []);

// The normalisers carry the ids from the board's data: a Match Read card's fixture and a schedule row.
const cardFrom = source.indexOf('  _normaliseMatchReadCard(');
const cardBody = source.slice(cardFrom, source.indexOf('\n  _indexFixtures(', cardFrom));
assert.match(cardBody, /homeId: fixture\.home_team_id \?\? null, awayId: fixture\.away_team_id \?\? null,/);
const from = source.indexOf('  _normaliseScheduledFixture(row, analysisAvailable) {');
const scheduled = Function(`return ({${source.slice(from, source.indexOf('\n  // A started match', from))}})`)();
scheduled._formatFixtureTime = () => '18:45';
const row = { fixture: { event_id: '1', league: 'UCL', home_team: 'Sabah FA', away_team: 'Barcelona', kickoff: '2099-01-01T18:45:00Z',
  home_team_id: 210, away_team_id: 21 }, status: 'NS', visuals: {} };
assert.deepEqual([scheduled._normaliseScheduledFixture(row, true).homeId, scheduled._normaliseScheduledFixture(row, true).awayId], [210, 21]);
const bare = { ...row, fixture: { ...row.fixture, home_team_id: undefined, away_team_id: undefined } };
assert.deepEqual([scheduled._normaliseScheduledFixture(bare, true).homeId, scheduled._normaliseScheduledFixture(bare, true).awayId], [null, null]);

// A tap on a name opens the profile, not the card: the card's own click ignores the links.
const render = source.slice(source.indexOf('  _createMatchReadFixtureCard('), source.indexOf('  _matchReadQuoteUsable('));
assert.match(render, /card\.addEventListener\('click', event => \{\s*if \(event\.target\.closest\('\[data-research-route\]'\)\) return;/);
assert.match(render, /this\._bindProfileLinks\(card\);\s*if \(f\.isScheduleOnly\) return card;/);
console.log('board links web tests passed');
