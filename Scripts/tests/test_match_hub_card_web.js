// The matchday hub card's pure helpers (option A, 29 Oct): plain outcome names, the fair-price rows,
// Spix beside the market, form squares, head-to-head meetings and the research-team match. No DOM or network.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../web_app/static/app.js'), 'utf8');
const start = source.indexOf('// ----- Match hub card helpers');
const end = source.indexOf('// ----- End of match hub card helpers -----', start);
assert.ok(start >= 0 && end > start, 'helpers block present');
const esc = value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const MARKET_LABELS = { goals: 'Goals', btts: 'BTTS', corners: 'Corners', cards: 'Cards', sot: 'Shots on target', moneyline: 'Moneyline', spreads: 'Handicap' };
const marketLabel = group => MARKET_LABELS[group] || group || '';
const hub = Function('esc', 'marketLabel', `${source.slice(start, end)}
  return { hubOutcomeName, hubPickName, hubFairRows, hubBalance, hubSpixVsMarket, hubFormMarkup,
           hubMeetings, hubMeetingsSummary, hubResearchMatch, hubFirstSentence };`)(esc, marketLabel);

const HOME = 'VfB Stuttgart', AWAY = 'Bayern München';
const lines = [
  { market: 'moneyline', line: null, book_count: 10, outcomes: [
    { side: 'home', line: null, fair_probability: 0.25, fair_odds: 4, minimum_odds: 4.08, spix_probability: 0.31 },
    { side: 'draw', line: null, fair_probability: 0.2, fair_odds: 5, minimum_odds: 5.1, spix_probability: 0.22 },
    { side: 'away', line: null, fair_probability: 0.55, fair_odds: 1 / 0.55, minimum_odds: 1.86, spix_probability: 0.47 }] },
  { market: 'goals', line: 3.5, book_count: 7, outcomes: [
    { side: 'over', line: 3.5, fair_probability: 0.48, fair_odds: 1 / 0.48, minimum_odds: 2.13, spix_probability: null },
    { side: 'under', line: 3.5, fair_probability: 0.52, fair_odds: 1 / 0.52, minimum_odds: 1.97, spix_probability: null }] },
  { market: 'spreads', line: 1.5, book_count: 7, outcomes: [
    { side: 'vfb stuttgart', line: 1.5, fair_probability: 0.7, fair_odds: 1 / 0.7, minimum_odds: 1.46, spix_probability: 0.6 },
    { side: 'bayern münchen', line: -1.5, fair_probability: 0.3, fair_odds: 1 / 0.3, minimum_odds: 3.4, spix_probability: 0.4 }] },
];

// Outcomes in plain words, with the fixture's own team names and signed handicaps.
assert.equal(hub.hubOutcomeName('moneyline', 'home', null, HOME, AWAY), 'VfB Stuttgart to win');
assert.equal(hub.hubOutcomeName('moneyline', 'draw', null, HOME, AWAY), 'Draw');
assert.equal(hub.hubOutcomeName('goals', 'under', 3.5, HOME, AWAY), 'Under 3.5 goals');
assert.equal(hub.hubOutcomeName('btts', 'no', null, HOME, AWAY), 'Both teams score: no');
assert.equal(hub.hubOutcomeName('spreads', 'bayern münchen', -1.5, HOME, AWAY), 'Bayern München −1.5');
assert.equal(hub.hubOutcomeName('spreads', 'vfb stuttgart', 0, HOME, AWAY), 'VfB Stuttgart (draw no bet)', 'a level handicap is draw no bet');

// A pick's stored wording ("Vfb Stuttgart 2") becomes the same plain name.
assert.equal(hub.hubPickName({ market: { group: 'spreads' }, pick: 'Vfb Stuttgart 2', explanation: { line: 2 } }, HOME, AWAY), 'VfB Stuttgart +2');
assert.equal(hub.hubPickName({ market: { group: 'corners' }, pick: 'Over 8.5' }, HOME, AWAY), 'Over 8.5 corners');
assert.equal(hub.hubPickName({ market: { group: 'goals' }, pick: 'Under 4' }, HOME, AWAY), 'Under 4 goals');
assert.equal(hub.hubPickName({ market: { group: 'btts' }, pick: 'BTTS No' }, HOME, AWAY), 'Both teams score: no');
assert.equal(hub.hubPickName({ market: { group: 'moneyline' }, pick: 'Bayern München' }, HOME, AWAY), 'Bayern München to win');
assert.equal(hub.hubPickName({ market: { group: 'cards' }, pick: 'Over 4.5' }, HOME, AWAY), 'Over 4.5 cards');

// Every priced outcome is a row: fair odds and "worth it at" (minimum odds) to the cent.
const rows = hub.hubFairRows(lines, HOME, AWAY);
assert.equal(rows.length, 7);
assert.deepEqual(rows[2], { name: 'Bayern München to win', fair: '1.82', worth: '1.86', market: 'moneyline', side: 'away', line: null, first: false });
assert.equal(rows[3].first, true, 'a new market starts a group');
assert.deepEqual(hub.hubFairRows([{ market: 'goals', outcomes: [{ side: 'over', line: 2.5, fair_odds: null, minimum_odds: 2 }] }], HOME, AWAY), []);

// The market's home / draw / away chances, only when all three exist.
assert.deepEqual(Object.values(hub.hubBalance(lines)).map(Math.round), [25, 20, 55]);
assert.equal(hub.hubBalance(lines.slice(1)), null);

// Spix beside the market: per line the outcome the market rates likelier; the widest gap across all outcomes.
const versus = hub.hubSpixVsMarket(lines, HOME, AWAY);
assert.deepEqual(versus.rows.map(row => [row.name, Math.round(row.spix), Math.round(row.market)]),
  [['Bayern München to win', 47, 55], ['VfB Stuttgart +1.5', 60, 70]], 'lines without Spix numbers are left out');
assert.equal(versus.biggest.name, 'VfB Stuttgart +1.5');

// Form squares, oldest first, named for screen readers; unknown results dropped.
const form = hub.hubFormMarkup(['W', 'D', '?', 'L']);
assert.match(form, /aria-label="won, drew, lost"/);
assert.equal((form.match(/<b /g) || []).length, 3);
assert.equal(hub.hubFormMarkup([]), '');

// Meetings: only the opponent's, newest first, scored; summary from the first team's side.
const log = [
  { date: '2025-03-01T15:00:00Z', home: { id: 7 }, away: { id: 9 }, opponent: { id: 9 }, score: { home: 2, away: 1 } },
  { date: '2025-10-01T15:00:00Z', home: { id: 9 }, away: { id: 7 }, opponent: { id: 9 }, score: { home: 3, away: 3 } },
  { date: '2025-11-01T15:00:00Z', home: { id: 7 }, away: { id: 5 }, opponent: { id: 5 }, score: { home: 0, away: 0 } },
  { date: '2024-04-01T15:00:00Z', home: { id: 9 }, away: { id: 7 }, opponent: { id: 9 }, score: { home: 1, away: 0 } },
  { date: '2024-01-01T15:00:00Z', home: { id: 9 }, away: { id: 7 }, opponent: { id: 9 }, score: { home: null, away: null } },
];
const meetings = hub.hubMeetings(log, 9);
assert.deepEqual(meetings.map(m => m.date.slice(0, 7)), ['2025-10', '2025-03', '2024-04']);
assert.deepEqual(hub.hubMeetingsSummary(meetings, 7), { count: 3, wins: 1, draws: 1, losses: 1, over25: 2, btts: 2 });

// The research profile is the team whose crest file matches the card's, else the exact name; never a guess.
const results = [{ team_id: 72, name: 'VfB Stuttgart II', logo_url: 'https://x/teams/999.png' },
  { team_id: 71, name: 'VfB Stuttgart', logo_url: 'https://x/teams/172.png' }];
assert.equal(hub.hubResearchMatch(results, 'Vfb Stuttgart', 'https://media/teams/172.png').team_id, 71);
assert.equal(hub.hubResearchMatch(results, 'VfB Stuttgart', '').team_id, 71);
assert.equal(hub.hubResearchMatch(results, 'Stuttgart', ''), null);

// The card's one-line reason is the briefing's first sentence; decimals do not end it.
assert.equal(hub.hubFirstSentence('The model projects 3.54 goals. Leverkusen lead.'), 'The model projects 3.54 goals.');
assert.equal(hub.hubFirstSentence(''), '');

console.log('Hub card names, fair rows, balance, Spix vs market, form, meetings, research match and reason checks passed.');
