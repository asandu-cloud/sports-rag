// Research Area overhaul task 3 (D107): the match-by-match view answers the bettor's question first, with the line
// always on screen. Cole Palmer's and Anthony Taylor's real profiles, saved from the live API on 15 Nov.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');

const text = html => html.replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim();
const fixture = name => JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures', name), 'utf8'));

(async () => {
  const h = await import(pathToFileURL(path.join(__dirname, '../web_app/static/research/insight.js')));
  const {catalogues} = await import(pathToFileURL(path.join(__dirname, '../web_app/static/research/data.js')));
  const sot = catalogues.players.metrics.shots_on_target;

  // Lines: a player's counts are 1+, 2+, 3+; the referee's cards are the bookmakers' halves.
  assert.deepEqual(h.linesFor(sot), [1, 2, 3]);
  assert.deepEqual(h.linesFor(catalogues.referees.metrics.cards), [2.5, 3.5, 4.5, 5.5]);
  assert.deepEqual(h.linesFor(catalogues.players.metrics.passes), [25, 30, 35, 40]);
  assert.deepEqual(h.linesFor(catalogues.teams.metrics.xg_for), [], 'no line for expected goals');
  assert.deepEqual(h.linesFor(catalogues.teams.metrics.possession), [], 'no line for a percentage');
  assert.equal(h.defaultLine(sot), 1);
  assert.equal(h.defaultLine(catalogues.referees.metrics.cards), 3.5);

  // Palmer's real last 10 on target: 1 or more in 7 of them; 2 or more in 2.
  const values = fixture('research_player_palmer.json').match_log.slice(-10).map(m => m.values.shots_on_target);
  const one = h.hitCount(values, 1), two = h.hitCount(values, 2);
  const answer = h.statAnswer({noun: sot.noun, line: 1, ...one, missing: 0});
  assert.equal(text(answer), '1 or more shots on target in 7 of their last 10.');
  assert.match(answer, /<mark>7 of their last 10<\/mark>/);
  assert.equal(text(h.statAnswer({noun: sot.noun, line: 2, ...two, missing: 0})), '2 or more shots on target in 2 of their last 10.');
  // Missing statistics are left out and said so.
  const gaps = h.hitCount([2, null, 1, null, 0, null, 1, 1, 0, 3], 1);
  assert.equal(text(h.statAnswer({noun: sot.noun, line: 1, ...gaps, missing: 3})), '1 or more shots on target in 5 of their last 7 with data.');
  // The whole season reads "this season", and a half line reads "Over".
  assert.equal(text(h.statAnswer({noun: 'cards', line: 3.5, hits: 30, of: 39, missing: 0, season: true})), 'Over 3.5 cards in 30 of 39 matches this season.');

  console.log('test_research_stat_web: ok');
})().catch(error => { console.error(error); process.exit(1); });
