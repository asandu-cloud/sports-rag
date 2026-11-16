// Research Area overhaul task 1 (D107): lines, hits and the number rules shared by every research view. Real
// profiles saved from the live API on 15 Nov (Chelsea's Premier League 2026/27, Cole Palmer's); no browser or network.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');

const fixture = name => JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures', name), 'utf8'));
const text = html => html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();

(async () => {
  const h = await import(pathToFileURL(path.join(__dirname, '../web_app/static/research/insight.js')));

  // One decimal at most; a true zero is "0"; only a missing value is a dash.
  assert.equal(h.one(2.666), '2.7');
  assert.equal(h.one(3), '3');
  assert.equal(h.one(0), '0');
  assert.equal(h.one(null), '–');
  assert.equal(h.one(undefined), '–');

  // The spec's lines and defaults.
  assert.deepEqual(h.LINES.cards, [2.5, 3.5, 4.5, 5.5]);
  assert.deepEqual(h.LINES.goals, [1.5, 2.5, 3.5]);
  assert.deepEqual(h.LINES.corners, [8.5, 9.5, 10.5, 11.5]);
  assert.deepEqual(h.LINES.sot, [1, 2, 3]);
  assert.deepEqual(h.DEFAULT_LINE, {cards: 3.5, goals: 2.5, corners: 9.5, sot: 1});

  // A profile row becomes the match view's row shape: Sunderland 1–0 Chelsea on 8 Nov, Chelsea 4 cards, Sunderland 5.
  const chelsea = fixture('research_team_chelsea.json');
  const rows = chelsea.match_log.map(h.fromProfileRow);
  const last = rows.at(-1);
  assert.equal(last.date.slice(0, 10), '2026-11-08');
  assert.equal(last.venue, 'away');
  assert.equal(last.opponent.name, 'Sunderland');
  assert.equal(last.cards_for, 4);
  assert.equal(last.cards_against, 5);
  assert.equal(last.goals_for, 0);
  assert.equal(last.goals_against, 1);
  assert.equal(h.marketTotal(last, 'cards'), 9);
  assert.equal(h.marketTotal(last, 'corners'), 4);
  assert.equal(h.marketTotal(last, 'goals'), 1);
  assert.equal(h.marketTotal({...last, cards_against: null}, 'cards'), null, 'one side missing means no total');

  // hitCount skips nulls: a line of 3.5 counts totals over it; a player's "1+" counts 1 or more.
  assert.deepEqual(h.hitCount([5, null, 2, 4], 3.5), {hits: 2, of: 3});
  assert.deepEqual(h.hitCount([0, 1, 2], 1), {hits: 2, of: 3});
  assert.deepEqual(h.hitCount([], 2.5), {hits: 0, of: 0});
  // Chelsea's real last 10 at 3.5 cards: 9 of 10, only Brentford away (2 cards) stayed under.
  assert.deepEqual(h.hitCount(rows.map(r => h.marketTotal(r, 'cards')), 3.5), {hits: 9, of: 10});
  // Cole Palmer's real last 10: 1 or more on target in 7 of them.
  const palmer = fixture('research_player_palmer.json');
  assert.deepEqual(h.hitCount(palmer.match_log.slice(-10).map(m => m.values.shots_on_target), 1), {hits: 7, of: 10});

  // The sketch's strip: one square a match, newest on the right, filled when it happened, dashed when not recorded.
  // The count sits beside it in the page; the strip carries it for screen readers.
  const strip = h.strip([5, null, 2], v => v > 3.5, ['at FUL', 'v BRI', 'at BRE']);
  const marks = strip.match(/<i\b[^>]*>/g);
  assert.equal(marks.length, 3);
  assert.match(marks[0], /class="y"/);
  assert.match(marks[1], /class="na"/);
  assert.match(marks[2], /class=""/);
  assert.match(marks[0], /title="at FUL: 5"/);
  assert.match(marks[1], /title="v BRI: not recorded"/);
  assert.match(strip, /role="img" aria-label="1 of 2"/);
  assert.equal(h.passes(4, 3.5), true);
  assert.equal(h.passes(3, 3.5), false);
  assert.equal(h.passes(1, 1), true, 'a whole-number line means at least');
  assert.equal(h.passes(null, 1), null);

  // Labelled columns: every value written, a zero drawn as a mark with "0", the line dashed and named in the legend.
  const cols = h.labelledColumns([
    {value: 2, top: 'FUL', bottom: 'A', title: 'a'}, {value: 0, top: 'HUL', bottom: 'H', title: 'b'},
    {value: null, top: 'BRE', bottom: 'A', title: 'c'}, {value: 1, top: 'BOU', bottom: 'H', title: 'd'}],
  {line: 1, label: 'Shots on target by appearance', legend: 'At least 1', noun: 'On target'});
  const colParts = cols.split('class="lc-col ').slice(1).map(part => part.slice(0, part.indexOf('</b>') + 4));
  assert.equal(colParts.length, 4);
  assert.match(colParts[0], /^hit"[\s\S]*<b>2<\/b>$/);
  assert.match(colParts[1], /^zero"[\s\S]*<b>0<\/b>$/);
  assert.match(colParts[2], /^none"[\s\S]*<b>–<\/b>$/);
  assert.match(colParts[3], /^hit"[\s\S]*<b>1<\/b>$/, '1 reaches a 1+ line');
  assert.match(cols, /class="lc-line"/);
  assert.match(text(cols), /At least 1/);
  assert.match(cols, /<details class="tbl">/);
  assert.equal((cols.match(/<tr>/g) || []).length, 5, 'a header row and one row per match');

  console.log('test_research_insight_web: ok');
})().catch(error => { console.error(error); process.exit(1); });
