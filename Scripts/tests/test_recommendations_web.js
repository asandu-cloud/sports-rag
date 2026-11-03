// The recommendation layer's match page and board card (design B, tiles; D58), rendered from app.js's pure
// helpers with a synthetic recommendations.v1 block. No browser, network or database.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../web_app/static/app.js'), 'utf8');
const esc = value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const helpersStart = source.indexOf('// ----- Pure render helpers');
const helpersEnd = source.indexOf('// ----- End of pure render helpers -----');
assert.ok(helpersStart >= 0 && helpersEnd > helpersStart);
const h = Function('esc', source.slice(helpersStart, helpersEnd)
  + 'return { chanceWords, renderRecommendations, renderMarketExplorer, renderRecsBoardCard, recsValueLeg, recsStepperPairs, recommendationsFor };')(esc);
const text = html => html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
const count = (haystack, needle) => haystack.split(needle).length - 1;

// Chance in words: the nearest plain fraction.
assert.equal(h.chanceWords(0.76), 'about 3 in 4');
assert.equal(h.chanceWords(0.17), 'about 1 in 6');
assert.equal(h.chanceWords(0.52), 'about 1 in 2');
assert.equal(h.chanceWords(0.02), 'about 1 in 50');
assert.equal(h.chanceWords(0.98), 'about 49 in 50');

const option = (market, side, line, label, chance, price) => ({ market, side, line, label, chance, typical_price: price, long_shot: chance < 0.25 });
const over25 = option('goals', 'over', 2.5, 'Over 2.5 goals', 0.661, 1.46);
const bttsYes = option('btts', 'yes', null, 'Both teams to score', 0.58, 1.6);
const block = {
  version: 'recommendations.v1',
  prices_as_of: '2026-11-03T11:12:58',
  most_likely: [
    { ...option('corners', 'under', 11.5, 'Under 11.5 corners', 0.722, 1.32), why: 'Spix expects about 8.7 corners.' },
    { ...option('spreads', 'away', 1.5, 'Lille +1.5', 0.673, 1.45), why: 'Spix expects Bodo/Glimt about 2.1 goals and Lille about 1.3.' },
    { ...over25, why: 'Spix expects about 3.4 goals.' },
  ],
  value: [{ position: 1, market: 'corners', pick: 'Over 10', chance: 0.484, worth_it_at: 2.1, published_recommendation_id: 9078 }],
  markets: [
    { market: 'moneyline', suggestion: null, options: [
      option('moneyline', 'home', null, 'Bodo/Glimt to win', 0.5, 1.9),
      option('moneyline', 'draw', null, 'Draw', 0.2, 4.5),
      option('moneyline', 'away', null, 'Lille to win', 0.3, 3.1)] },
    { market: 'goals', suggestion: over25, options: [
      option('goals', 'over', 1.5, 'Over 1.5 goals', 0.86, 1.12), option('goals', 'under', 1.5, 'Under 1.5 goals', 0.14, 6.1),
      over25, option('goals', 'under', 2.5, 'Under 2.5 goals', 0.339, 2.8),
      option('goals', 'over', 3.5, 'Over 3.5 goals', 0.44, 2.2), option('goals', 'under', 3.5, 'Under 3.5 goals', 0.56, 1.7)] },
    { market: 'btts', suggestion: bttsYes, options: [bttsYes, option('btts', 'no', null, 'No', 0.42, 2.3)] },
    { market: 'spreads', suggestion: null, options: [
      option('spreads', 'home', -1.5, 'Bodo/Glimt −1.5', 0.327, 2.78), option('spreads', 'away', 1.5, 'Lille +1.5', 0.673, 1.45),
      option('spreads', 'home', -0.5, 'Bodo/Glimt −0.5', 0.5, 1.9), option('spreads', 'away', 0.5, 'Lille +0.5', 0.5, 1.9)] },
    { market: 'corners', suggestion: null, options: [] },
  ],
  score_grid: { home: 2.09, away: 1.29 },
};
const fixture = { id: '77', home: 'Bodo/Glimt', away: 'Lille', time: '22:00', recommendations: block };
const page = h.renderRecommendations(fixture);

// Most likely: label, chance in words with the percentage beside it, the typical price, the why, Add.
assert.match(page, /Most likely/);
assert.match(page, /Prices as of \d\d:\d\d/);
assert.match(text(page), /Under 11\.5 corners about 3 in 4 72%/);
assert.match(page, />1\.32</);
assert.match(page, /Spix expects about 8\.7 corners\./);
assert.ok(count(page, 'data-recs-add=') >= 4, 'every pick has an Add button');

// Value: the unit is added to the pick, the price floor is shown, no bookmaker is named.
assert.match(text(page), /Over 10 corners/);
assert.match(text(page), /Worth it at 2\.10 or better/);
assert.match(text(page), /Exactly 10 corners: you get your stake back\./);
assert.doesNotMatch(page, /bookmaker/i);
assert.deepEqual(h.recsValueLeg(block.value[0], block),
  { market: 'corners', side: 'over', line: 10, label: 'Over 10 corners', chance: 0.484, worth: 2.1, whole: true });

// Value picks as the engine writes them: "Away", a team name in other case with a hyphen, a missing price floor.
assert.deepEqual(h.recsValueLeg({ market: 'moneyline', pick: 'Away', chance: 0.3, worth_it_at: 3.4 }, block),
  { market: 'moneyline', side: 'away', line: null, label: 'Lille to win', chance: 0.3, worth: 3.4, whole: false });
assert.equal(h.recsValueLeg({ market: 'spreads', pick: 'BODO/GLIMT -1.5', chance: 0.33, worth_it_at: 3.1 }, block).label, 'Bodo/Glimt −1.5');
const unpricedValue = h.renderRecommendations({ ...fixture, recommendations: { ...block,
  value: [{ position: 1, market: 'spreads', pick: 'Bodo/Glimt -1.5', chance: null, worth_it_at: null }] } });
assert.match(text(unpricedValue), /Bodo\/Glimt −1\.5/);
assert.doesNotMatch(unpricedValue, /0\.00|or better|1 in 200/);
const longValue = h.renderRecommendations({ ...fixture, recommendations: { ...block,
  value: [{ position: 1, market: 'moneyline', pick: 'Home', chance: 0.155, worth_it_at: 6.59 }] } });
assert.match(text(longValue), /A long shot, about 1 in 6 · 16%/);

const noValue = h.renderRecommendations({ ...fixture, recommendations: { ...block, value: [] } });
assert.match(noValue, /No value picks on this match today\. The prices are about right\./);

// Explore the match: Result as three tiles, a long shot said once.
const result = h.renderMarketExplorer(fixture, { open: 'moneyline' });
for (const chip of ['Result', 'Goals', 'Both teams to score', 'Handicap', 'Corners']) assert.match(result, new RegExp(`>${chip}<`));
assert.equal(count(result, 'class="recs-tile'), 3);
assert.equal(count(text(result).toLowerCase(), 'a long shot, about 1 in 5'), 1);
assert.match(result, /Nothing stands out here\. Every option can still be added\./);

// Goals open on Spix's suggestion's line, with the pick marked; the stepper moves along the lines.
const goals = h.renderMarketExplorer(fixture, { open: 'goals' });
assert.match(text(goals), /Spix's pick here: Over 2\.5 goals, about 2 in 3\./);
assert.match(text(goals), /2\.5 goals Line 2 of 3/);
assert.equal(count(goals, 'recs-pickmark'), 1);
assert.equal(count(goals, 'data-recs-add='), 2);
assert.match(text(h.renderMarketExplorer(fixture, { open: 'goals', line: { goals: 2 } })), /3\.5 goals Line 3 of 3/);
assert.match(h.renderMarketExplorer(fixture, { open: 'goals', line: { goals: 2 } }), /data-recs-step="1"[^>]*disabled/);

// Handicap lines pair the home side with the away side's opposite line.
const pairs = h.recsStepperPairs(block.markets[3]);
assert.deepEqual(pairs.map(pair => [pair.text, pair.options.map(o => o.label)]),
  [['Bodo/Glimt −1.5', ['Bodo/Glimt −1.5', 'Lille +1.5']], ['Bodo/Glimt −0.5', ['Bodo/Glimt −0.5', 'Lille +0.5']]]);

// A market nobody prices: "Nothing stands out here" and no Add buttons, whichever chip is open.
const empty = { ...fixture, recommendations: { ...block, most_likely: [], value: [], markets: block.markets.map(m => ({ ...m, suggestion: null, options: [] })), score_grid: null } };
for (const m of block.markets) {
  const html = h.renderMarketExplorer(empty, { open: m.market });
  assert.match(html, /Nothing stands out here\./);
  assert.doesNotMatch(html, /data-recs-add=/);
}
const unpriced = h.renderRecommendations(empty);
assert.match(unpriced, /The bookmakers haven't priced this match yet, so there are no suggestions\. They usually appear a day or two before kick-off\./);
assert.doesNotMatch(unpriced, />Value</);
assert.doesNotMatch(unpriced, /data-recs-add=/);

// Added state comes from the caller.
const added = h.renderRecommendations(fixture, { added: leg => leg.label === 'Lille +1.5' });
assert.equal(count(added, 'recs-add added'), 1);

// Text from the block is escaped.
const hostile = { ...fixture, recommendations: { ...block, most_likely: [{ ...block.most_likely[0], why: '<script>x</script>' }] } };
assert.doesNotMatch(h.renderRecommendations(hostile), /<script>/);

// The board card: the first Most likely pick, and the count of suggestions (markets with one, plus value picks).
const board = h.renderRecsBoardCard(fixture, '', '');
assert.match(board, /Most likely/);
assert.match(text(board), /Under 11\.5 corners about 3 in 4 · 72% 1\.32/);
assert.match(text(board), /3 suggestions · 1 value pick/);
assert.match(text(h.renderRecsBoardCard(empty, '', '')), /Not priced yet/);

// ?recs=off hides the block, so the page falls back to the old summary for comparison.
assert.equal(h.recommendationsFor({ recommendations: block }, '?recs=off'), null);
assert.equal(h.recommendationsFor({ recommendations: block }, ''), block);
assert.equal(h.recommendationsFor({}, ''), null);
const from = source.indexOf('  _normaliseMatchReadCard(card, leagueId) {');
const to = source.indexOf('\n  _indexFixtures(fixtures)', from);
assert.ok(from >= 0 && to > from);
const normaliser = Function('briefingFigures', 'recommendationsFor', 'window', `return ({${source.slice(from, to)}})`)(
  () => ({ projectedTotal: null, teamGoals: { home: null, away: null }, resultBalance: null }), h.recommendationsFor,
  { location: { search: '?recs=off' } });
Object.assign(normaliser, { _formatMatchReadSelection: () => '', _formatFixtureTime: () => '22:00' });
assert.equal(normaliser._normaliseMatchReadCard({ id: 1, status: 'no_bet', recommendations: block }, 'UCL').recommendations, null);

console.log('Recommendation layer page, explorer, board card and ?recs=off checks passed.');
