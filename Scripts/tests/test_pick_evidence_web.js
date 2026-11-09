// Pick evidence (D82 data, D83 display): "How Spix reads it" above the picks, and under each Most likely pick a
// verdict, a detail line and past matches at similar odds, each opening a sheet of those matches. Built from the
// owner's approved mockup ("Easier to read"). A card or pick without these keys renders exactly as before.
// No browser or network.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../web_app/static/app.js'), 'utf8');
const esc = value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const helpersStart = source.indexOf('// ----- Pure render helpers');
const helpersEnd = source.indexOf('// ----- End of pure render helpers -----');
assert.ok(helpersStart >= 0 && helpersEnd > helpersStart);
const h = Function('esc', source.slice(helpersStart, helpersEnd)
  + 'return { renderRecommendations, renderRecsBoardCard, renderRecsStory, renderEvidenceSheet };')(esc);
const text = html => html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
const count = (haystack, needle) => haystack.split(needle).length - 1;

// A block like the engine's for Brighton v Brentford (8 Nov), without the evidence (the switch off).
const option = (market, side, line, label, chance, price) => ({ market, side, line, label, chance, typical_price: price, long_shot: false });
const plain = {
  version: 'recommendations.v1',
  prices_as_of: '2026-11-08T13:58:37',
  most_likely: [
    { ...option('spreads', 'away', 1.5, 'Brentford +1.5', 0.82, 1.18), why: 'Spix gives Brentford about 4 in 5.' },
    { ...option('corners', 'over', 8.5, 'Over 8.5 corners', 0.66, 1.42), why: 'Spix expects about 9.5 corners.' },
    { ...option('btts', 'yes', null, 'Both teams to score', 0.6, 1.6), why: 'Spix expects about 3.2 goals.' },
  ],
  value: [],
  markets: [{ market: 'moneyline', suggestion: null, options: [option('moneyline', 'home', null, 'Brighton to win', 0.5, 1.92)] }],
  score_grid: null,
};
const story = {
  lead: 'Brighton are favourites, with goals expected at both ends.',
  result_chances: { home: 0.5054, draw: 0.2418, away: 0.2528 },
  goals_expected: { home: 1.9, away: 1.3 },
  corners_expected: 9.5,
};
const match = (date, season, home, away, prices, goals, corners, extra = {}) => ({
  date, season, competition: 'EPL', home, away,
  home_price: prices[0], draw_price: prices[1], away_price: prices[2],
  home_goals: goals[0], away_goals: goals[1], corners, ...extra,
});
const evidence = {
  version: 'pick-evidence.v1',
  since: '2024-08-01',
  competition: 'EPL',
  tonight: { home: 1.92, draw: 3.9, away: 3.75 },
  groups: [
    { key: 'home', team: 'Brighton', price_from: 1.85, price_to: 1.99, matches: [
      match('2026-05-24', '2025/26', 'Brighton', 'Manchester United', [1.86, 3.98, 3.58], [0, 3], 3, { venue: 'H' }),
      match('2026-02-08', '2025/26', 'Brighton', 'Crystal Palace', [1.96, 3.57, 3.68], [0, 1], 7, { venue: 'H' }),
      match('2025-04-12', '2024/25', 'Leicester', 'Brighton', [4.2, 3.9, 1.9], [2, 2], 11, { venue: 'A', competition: 'Championship' }),
    ] },
    { key: 'away', team: 'Brentford', price_from: 3.4, price_to: 4.21, matches: [
      match('2026-05-24', '2025/26', 'Liverpool', 'Brentford', [1.8, 4.09, 3.73], [1, 1], 16, { venue: 'A' }),
      match('2026-04-27', '2025/26', 'Manchester United', 'Brentford', [1.84, 3.82, 3.84], [2, 1], 15, { venue: 'A' }),
    ] },
    { key: 'league', team: null, price_from: null, price_to: null, matches: [
      match('2026-10-31', '2026/27', 'Chelsea', 'Manchester United', [2.0, 3.77, 3.61], [2, 0], 15),
      match('2026-10-24', '2026/27', 'Chelsea', 'Tottenham', [1.91, 3.87, 3.88], [0, 2], 10),
    ] },
  ],
};
const pickEvidence = [
  { verdict: 'All three groups agree.', detail: 'Brentford won 4, drew 1 and lost 7 of their 12 matches at similar odds.', groups: [
    { key: 'home', won: 3, of: 3, means: 'Brighton did not win by more than 1', results: [true, true, true] },
    { key: 'away', won: 2, of: 2, means: 'Brentford did not lose by more than 1', results: [true, true] },
    { key: 'league', won: 1, of: 2, means: 'the away side did not lose by more than 1', results: [false, true] },
  ] },
  { verdict: 'Two of the three groups agree.', detail: 'Similar matches averaged 9.7 to 11.7 corners.', groups: [
    { key: 'home', won: 1, of: 3, means: 'the match had 9 or more corners', results: [false, false, true] },
    { key: 'away', won: 2, of: 2, means: 'the match had 9 or more corners', results: [true, true] },
    { key: 'league', won: 2, of: 2, means: 'the match had 9 or more corners', results: [true, true] },
  ] },
  { verdict: 'The groups are split.', detail: '', groups: [
    { key: 'home', won: 1, of: 3, means: 'both teams scored', results: [false, false, true] },
    { key: 'away', won: 2, of: 2, means: 'both teams scored', results: [true, true] },
  ] },
];
const full = { ...plain, story, evidence, most_likely: plain.most_likely.map((item, i) => ({ ...item, evidence: pickEvidence[i] })) };
const fixture = block => ({ id: '1557458', league: 'EPL', home: 'Brighton', away: 'Brentford', time: '14:00', recommendations: block });
const NEW = ['recs-story', 'recs-verdict', 'recs-detail', 'recs-ev', 'data-recs-evidence'];

// Switch off: nothing new, and the pick keeps its "why" line.
const before = h.renderRecommendations(fixture(plain));
for (const cls of NEW) assert.equal(count(before, cls), 0, `${cls} without evidence`);
assert.equal(count(before, 'class="recs-why"'), 3);

// Keys that are there but unusable change nothing: a pick's evidence naming a group the card does not have, a pick
// evidence without the card's groups, an evidence block of another version, a story without its chances.
const broken = [
  { ...plain, evidence, most_likely: plain.most_likely.map(item => ({ ...item, evidence: { ...pickEvidence[0], groups: [{ key: 'cup', won: 1, of: 2, means: 'x', results: [true, false] }] } })) },
  { ...plain, most_likely: plain.most_likely.map((item, i) => ({ ...item, evidence: pickEvidence[i] })) },
  { ...plain, evidence: { ...evidence, version: 'pick-evidence.v9' }, most_likely: plain.most_likely.map((item, i) => ({ ...item, evidence: pickEvidence[i] })) },
  { ...plain, evidence, most_likely: plain.most_likely.map(item => ({ ...item, evidence: { verdict: '', detail: '', groups: [] } })) },
  { ...plain, story: { lead: 'x', result_chances: { home: 0.5 } } },
  { ...plain, story: null, evidence: null },
];
for (const block of broken) assert.equal(h.renderRecommendations(fixture(block)), before, JSON.stringify(block).slice(0, 120));

// The board card does not change.
assert.equal(h.renderRecsBoardCard(fixture(full)), h.renderRecsBoardCard(fixture(plain)));

// How Spix reads it: the lead, then each figure once.
const storyHtml = h.renderRecsStory(full, 'Brighton', 'Brentford');
assert.match(storyHtml, /^\s*<section class="recs-story" aria-label="How Spix reads it">/);
assert.match(text(storyHtml), /^How Spix reads it Brighton are favourites, with goals expected at both ends\./);
assert.match(text(storyHtml), /Chance of each result Brighton 51% · Draw 24% · Brentford 25%/);
assert.match(text(storyHtml), /Goals expected Brighton 1\.9 · Brentford 1\.3/);
assert.match(text(storyHtml), /Corners expected 9\.5/);
// A missing figure leaves out its cell; a story without its chances shows nothing.
const noGoals = h.renderRecsStory({ ...full, story: { ...story, goals_expected: null, corners_expected: null } }, 'Brighton', 'Brentford');
assert.doesNotMatch(noGoals, /Goals expected|Corners expected/);
assert.match(text(noGoals), /Chance of each result/);
assert.equal(h.renderRecsStory({ ...full, story: { ...story, result_chances: null } }, 'Brighton', 'Brentford'), '');
assert.equal(h.renderRecsStory(plain, 'Brighton', 'Brentford'), '');
assert.doesNotMatch(h.renderRecsStory({ ...full, story: { ...story, lead: '<b>x</b>' } }, '<i>A</i>', 'B'), /<b>x<\/b>|<i>A<\/i>/);
// A cup card has the story and no evidence: the story shows, the picks are as before.
const cup = h.renderRecommendations(fixture({ ...plain, story }));
assert.equal(count(cup, 'class="recs-story"'), 1);
assert.equal(cup.replace(storyHtml, ''), before);

// With the evidence: the story first, then each pick's verdict, detail and rows instead of its "why".
const after = h.renderRecommendations(fixture(full));
assert.ok(after.indexOf('class="recs-story"') < after.indexOf('id="recsMostLikely"'));
assert.equal(count(after, 'class="recs-why"'), 0);
assert.equal(count(after, 'class="recs-verdict"'), 3);
assert.equal(count(after, 'class="recs-detail"'), 2, 'an empty detail is left out');
assert.equal(count(after, 'data-recs-evidence='), 8);
const first = after.slice(after.indexOf('<article class="recs-card">'), after.indexOf('</article>'));
assert.match(text(first), /^Handicap Brentford \+1\.5 about 5 in 6 82% All three groups agree\. Brentford won 4, drew 1 and lost 7 of their 12 matches at similar odds\. Past matches at similar odds Won/);
assert.match(text(first), /Brighton's 3 win odds 1\.85–1\.99 3 of 3 Brentford's 2 win odds 3\.40–4\.21 2 of 2 2 league matches closest home and away odds 1 of 2/);
assert.match(first, /data-recs-evidence="0:home"[^>]*aria-label="Brighton's 3: 3 of 3\. See the matches"/);
assert.match(first, /data-recs-evidence="0:league"/);
assert.match(first, /aria-haspopup="dialog"/);
assert.match(first, /<i style="width:50\.0%"><\/i>/);
assert.match(first, /<span class="recs-fig">1\.18<\/span>/);
// Value and Explore the match are unchanged.
assert.equal(after.slice(after.indexOf('id="recsValue"')), before.slice(before.indexOf('id="recsValue"')));

// The sheet: the matches behind one row, from the group's side.
const names = code => ({ EPL: 'Premier League', Championship: 'Championship' })[code] || code;
const home = h.renderEvidenceSheet(fixture(full), '0:home', { leagueName: names });
assert.match(text(home), /^Evidence for Brentford \+1\.5 Brighton's 3 matches priced like this/);
assert.match(text(home), /League matches since August 2024\. Brighton's win odds in these matches ran from 1\.85 to 1\.99\. Before this match they were 1\.92\./);
assert.match(home, /<b>3 of 3<\/b>: Brighton did not win by more than 1\./);
// The last column asks whether the pick won: the pick in full on wider screens, "Won?" on phones.
assert.match(text(home), /Date Match Win odds Score Brentford \+1\.5\? Won\?/);
assert.match(home, /<span class="recs-sheet-ask">Brentford \+1\.5\?<\/span><span class="recs-sheet-ask-short">Won\?<\/span>/);
// Each row: the date column (desktop), then the match with the date again under it (phones, where the date column
// is hidden so the price, score and answer stay in view).
assert.match(text(home), /24 May 2026 2025\/26 v Manchester United 24 May 2026 · 2025\/26 1\.86 L 0–3 Yes/);
assert.match(text(home), /12 Apr 2025 2024\/25 · Championship at Leicester 12 Apr 2025 · 2024\/25 · Championship 1\.90 D 2–2 Yes/);
assert.equal(count(home, '<td class="recs-sheet-date">'), 3);
assert.equal(count(home, '<th class="recs-sheet-date">Date</th>'), 1);
assert.equal(count(home, '<span class="recs-sheet-when">'), 3);
assert.match(text(home), /Scores are from Brighton's side: W 2–1 means Brighton won 2–1\./);
assert.match(home, /id="recsSheetTitle"/);
assert.match(home, /data-recs-sheet-close/);
const away = h.renderEvidenceSheet(fixture(full), '0:away', { leagueName: names });
assert.match(text(away), /Before this match they were 3\.75\./);
assert.match(text(away), /at Liverpool 24 May 2026 · 2025\/26 3\.73 D 1–1 Yes/);
assert.match(text(away), /at Manchester United 27 Apr 2026 · 2025\/26 3\.84 L 1–2 Yes/);
const league = h.renderEvidenceSheet(fixture(full), '0:league', { leagueName: names });
assert.match(text(league), /The 2 league matches priced like this/);
assert.match(text(league), /Premier League since August 2024: the matches whose home and away odds were closest to this match's 1\.92 and 3\.75\./);
assert.match(text(league), /Date Match Home \/ away Score/);
assert.match(text(league), /31 Oct 2026 2026\/27 Chelsea v Manchester United 31 Oct 2026 · 2026\/27 2\.00 \/ 3\.61 2–0 No/);
assert.doesNotMatch(league, /Scores are from/);
// Goals and corners show the match's total instead of the score.
const corners = h.renderEvidenceSheet(fixture(full), '1:home', { leagueName: names });
assert.match(text(corners), /Date Match Win odds Corners Over 8\.5\? Won\?/);
assert.match(text(corners), /v Manchester United 24 May 2026 · 2025\/26 1\.86 3 No/);
assert.doesNotMatch(corners, /Scores are from/);
const goalsBlock = { ...full, most_likely: [{ ...option('goals', 'under', 3.5, 'Under 3.5 goals', 0.7, 1.3), evidence: { ...pickEvidence[0], groups: pickEvidence[0].groups.map(g => ({ ...g, means: 'the match had 3 or fewer goals' })) } }] };
assert.match(text(h.renderEvidenceSheet(fixture(goalsBlock), '0:league', { leagueName: names })), /Goals Under 3\.5\? Won\?.*Chelsea v Manchester United 31 Oct 2026 · 2026\/27 2\.00 \/ 3\.61 2 No/);
// Escaped, and nothing for a row that is not there.
const hostile = { ...full, evidence: { ...evidence, groups: evidence.groups.map(g => ({ ...g, team: g.team && '<b>T</b>', matches: g.matches.map(m => ({ ...m, home: '<i>H</i>' })) })) } };
assert.doesNotMatch(h.renderEvidenceSheet(fixture(hostile), '0:home', { leagueName: names }), /<b>T<\/b>|<i>H<\/i>/);
for (const ref of ['2:league', '5:home', '0:cup', 'x', '', null]) assert.equal(h.renderEvidenceSheet(fixture(full), ref, { leagueName: names }), '', String(ref));
assert.equal(h.renderEvidenceSheet(fixture(plain), '0:home', { leagueName: names }), '');

console.log('pick evidence web: ok');
