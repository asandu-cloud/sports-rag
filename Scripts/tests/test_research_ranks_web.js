// Research Area overhaul task 4 (D107): ranks come from the page's own season, and the referee page leads with the
// season's cards and the bettor's line. Real profiles saved from the live API on 15 Nov: Cole Palmer (888 minutes in
// 2026/27, under the 900-minute threshold), Chelsea (2026/27 ranks stale or missing), Anthony Taylor (2025/26).
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');

const text = html => html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').replace(/ ([.,:])/g, '$1').trim();
const fixture = name => JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures', name), 'utf8'));

(async () => {
  const h = await import(pathToFileURL(path.join(__dirname, '../web_app/static/research/insight.js')));
  const data = await import(pathToFileURL(path.join(__dirname, '../web_app/static/research/data.js')));
  const defs = {rankMetrics: data.rankMetrics, rankGroups: data.rankGroups};

  // Palmer, unranked: this season's own values, no dots, how far from the threshold, and last season one link away.
  const palmer = fixture('research_player_palmer.json');
  const card = h.rankCard(palmer, 'players', defs);
  assert.doesNotMatch(card, /class="dot"/, 'no position without a rank');
  assert.match(text(card), /Against Premier League midfielders/);
  assert.match(text(card), /Ranked from 900 minutes · 888 so far/);
  assert.match(card, /data-scope="EPL-2025"[^>]*>2025\/26 ranks ›</);
  assert.match(text(card), /On target per 90 0\.9/, "this season's value, one decimal");
  const prevOnTarget = h.one(palmer.previous_season.metrics.shots_on_target_per_90.value);
  assert.notEqual(prevOnTarget, '0.9');
  assert.ok(!text(card).includes(`On target per 90 ${prevOnTarget}`), 'no 2025/26 value on the card');

  // With B5's early ranks the dots come back with a quiet chip.
  const early = structuredClone(palmer);
  early.ranking.provisional = true;
  for (const m of Object.values(early.metrics)) Object.assign(m, {percentile: 61, rank: 53, peer_count: 135});
  const earlyCard = h.rankCard(early, 'players', defs);
  assert.match(earlyCard, /class="dot"/);
  assert.match(text(earlyCard), /Early season · 888 minutes so far/);
  assert.doesNotMatch(text(earlyCard), /Ranked from 900 minutes/);

  // Chelsea: ranks missing this season, values still shown, said quietly.
  const chelsea = fixture('research_team_chelsea.json');
  const teamCard = h.rankCard(chelsea, 'teams', defs);
  assert.doesNotMatch(teamCard, /class="dot"/);
  assert.match(text(teamCard), /Ranks for 2026\/27 are being rebuilt/);

  // Taylor, ranked in 2025/26: the season's cards a match marked, with how he compares, and the line.
  const taylor = fixture('research_referee_taylor.json');
  const answer = h.refereeAnswer(taylor, {current: false, line: 3.5});
  assert.equal(text(answer.lead), '4.3 cards a match in 2025/26, more than 59% of referees.');
  assert.match(answer.lead, /<mark>4\.3 cards<\/mark>/);
  assert.equal(answer.sub, 'All competitions · 39 matches');
  const last = taylor.match_log.slice(-10).map(m => m.values.cards);
  const {hits, of} = h.hitCount(last, 3.5);
  assert.equal(text(answer.count), `Over 3.5 in ${hits} of ${of}`);
  assert.equal(text(h.refereeAnswer(taylor, {current: true, line: 4.5}).lead), '4.3 cards a match this season, more than 59% of referees.');
  assert.equal((answer.chart.match(/class="lc-col /g) || []).length, 10);
  assert.match(answer.chart, /Line 3\.5 cards/);

  // The opening line from the API's headline, its figure marked; B5 (backend 0c8e675) flags a provisional rank.
  const head = {text: '0.6 goals per 90, more than 80% of midfielders', display_value: '0.6', value: 0.62, provisional: false};
  assert.equal(h.headlineLead(head, palmer.ranking), '<p class="lead"><mark>0.6</mark> goals per 90, more than 80% of midfielders.</p>');
  const earlyLead = h.headlineLead({...head, provisional: true}, {...palmer.ranking, sample_minutes: 450});
  assert.match(text(earlyLead), /^0\.6 goals per 90, more than 80% of midfielders\. Early season · 450 minutes so far Rates move a lot with one match\.$/);
  assert.match(earlyLead, /<p class="quiet-line"><span class="quiet">Early season · 450 minutes so far<\/span>/);
  assert.doesNotMatch(h.headlineLead({...head, provisional: undefined}, palmer.ranking), /Early season/, 'an older payload has no flag');
  assert.equal(h.headlineLead(null, palmer.ranking), '');
  assert.equal(h.headlineLead({text: 'x'}, palmer.ranking), '', 'no figure, no headline');

  console.log('test_research_ranks_web: ok');
})().catch(error => { console.error(error); process.exit(1); });
