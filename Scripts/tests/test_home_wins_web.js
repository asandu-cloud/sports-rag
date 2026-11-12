// The home page's wins reel (D101): last week's picks that came in (home_wins.v1), as cards in an endless reel, with
// the honest record of both lists beside it. Real wins from the backend's 4–8 Nov sample. No browser or network.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const STATIC = path.join(__dirname, '../web_app/static');
const app = fs.readFileSync(path.join(STATIC, 'app.js'), 'utf8');
const helpersStart = app.indexOf('// ----- Pure render helpers');
const helpersEnd = app.indexOf('// ----- End of pure render helpers -----');
assert.ok(helpersStart >= 0 && helpersEnd > helpersStart);
const { chanceWords, chancePercent, RECS_MARKETS } = Function('esc', app.slice(helpersStart, helpersEnd)
  + 'return { chanceWords, chancePercent, RECS_MARKETS };')(String);
// The reel's script, with no page: only its pure builders.
const wins = Function('document', fs.readFileSync(path.join(STATIC, 'home-wins.js'), 'utf8')
  + 'return { HOME_WINS_SPEED, HOME_WINS_MIN, HOME_WINS_FILL, homeWinsSeed, homeWinsWeek, homeWinsAbout, homeWinsRecord,'
  + ' homeWinsCard, homeWinsReady, homeWinsTrack, homeWinsDuration };')(undefined);
const page = fs.readFileSync(path.join(STATIC, 'index.html'), 'utf8');

const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const LEAGUES = { Eredivisie: 'Eredivisie', LaLiga: 'La Liga', Championship: 'Championship' };
const h = { esc, words: chanceWords, percent: chancePercent, markets: RECS_MARKETS, leagueName: code => LEAGUES[code] || code };

const CAMBUUR = { fixture: { event_id: '1552219', league: 'Eredivisie', home_team: 'Cambuur', away_team: 'PSV Eindhoven', kickoff: '2026-11-08T15:45:00Z' },
  visuals: { home_team_logo: 'https://media.api-sports.io/football/teams/420.png', away_team_logo: 'https://media.api-sports.io/football/teams/197.png' },
  score: { home: 2, away: 5 }, list: 'value',
  pick: { label: 'Over 5.5 goals', market: 'goals', side: 'over', line: 5.5, chance: 0.2266, typical_price: 4.1 }, result: '7 goals' };
const OSASUNA = { fixture: { event_id: '1570448', league: 'LaLiga', home_team: 'Osasuna', away_team: 'Athletic Club', kickoff: '2026-11-07T17:30:00Z' },
  visuals: { home_team_logo: 'https://media.api-sports.io/football/teams/727.png', away_team_logo: 'https://media.api-sports.io/football/teams/531.png' },
  score: { home: 1, away: 1 }, list: 'most_likely',
  pick: { label: 'Osasuna +0.5', market: 'spreads', side: 'home', line: 0.5, chance: 0.5278, typical_price: 1.82 }, result: null };
const BIRMINGHAM = { fixture: { event_id: '1563251', league: 'Championship', home_team: 'Birmingham', away_team: 'Swansea', kickoff: '2026-11-07T15:00:00Z' },
  visuals: { home_team_logo: 'https://media.api-sports.io/football/teams/54.png', away_team_logo: 'https://media.api-sports.io/football/teams/76.png' },
  score: { home: 2, away: 1 }, list: 'most_likely',
  pick: { label: 'Both teams to score', market: 'btts', side: 'yes', line: null, chance: 0.5406, typical_price: 1.73 }, result: null };
const RECORD = [
  { list: 'most_likely', picks: 353, won: 248, lost: 105, void: 0, said: 0.6897, expected_wins: 243.5, happened: 0.7025 },
  { list: 'value', picks: 10, won: 3, lost: 5, void: 2, said: 0.4518, expected_wins: 4.5, happened: 0.3 },
];
const WEEK = { from: '2026-11-04', to: '2026-11-08' };
// More wins for the reel's length: the same three, as other matches.
const many = n => Array.from({ length: n }, (_, i) => {
  const w = structuredClone([CAMBUUR, OSASUNA, BIRMINGHAM][i % 3]);
  w.fixture.event_id = String(1600000 + i);
  return w;
});

// A Value win: the dark card, its chance in the site's words and said as a long shot, the typical price, the score
// between the crests and what happened. No bookmaker is named.
const value = wins.homeWinsCard(CAMBUUR, h);
assert.match(value, /<article class="hw-card is-value" role="img" aria-label="Value pick, Cambuur 2, PSV Eindhoven 5: Over 5\.5 goals, a long shot, about 1 in 4, typical price 4\.10\. It came in\.">/);
assert.match(value, /Value pick · Cambuur v PSV Eindhoven/);
assert.match(value, /<span data-fit>Goals<\/span>/);
assert.match(value, /<mark>Over 5\.5<\/mark>/, 'the line is marked; the market word says what it counts');
assert.match(value, /<b>2–5<\/b>/);
assert.match(value, /<p class="hw-happened">7 goals<\/p>/);
assert.match(value, /<b>about 1 in 4<\/b><span>chance: a long shot<\/span>/);
assert.match(value, /<b>4\.10<\/b><span>typical price<\/span>/);
assert.match(value, /Came in · Sun 8 Nov<\/span><span>Eredivisie<\/span>/);
assert.match(value, /src="https:\/\/media\.api-sports\.io\/football\/teams\/420\.png" alt=""/);
// A Most likely win: the light card, with the site's market name and an empty line where nothing extra is said.
const likely = wins.homeWinsCard(OSASUNA, h);
assert.match(likely, /<article class="hw-card" role="img" aria-label="Most likely pick, Osasuna 1, Athletic Club 1: Osasuna \+0\.5, about 1 in 2, typical price 1\.82\. It came in\.">/);
assert.match(likely, /<span data-fit>Handicap<\/span>/);
assert.match(likely, /<mark>Osasuna \+0\.5<\/mark>/);
assert.match(likely, /<b>about 1 in 2<\/b><span>chance<\/span>/);
assert.match(likely, /<p class="hw-happened"><\/p>/);
assert.match(likely, /Came in · Sat 7 Nov<\/span><span>La Liga<\/span>/);
// Both teams to score marks Yes or No under the market's name.
assert.match(wins.homeWinsCard(BIRMINGHAM, h), /<span data-fit>Both teams to score<\/span><\/p>\s*<p class="hw-pick"><mark>Yes<\/mark>/);
const notBoth = structuredClone(BIRMINGHAM);
Object.assign(notBoth.pick, { label: 'Not both teams to score', side: 'no' });
assert.match(wins.homeWinsCard(notBoth, h), /<mark>No<\/mark>/);
// Nothing from the feed reaches the page unescaped; a missing crest leaves an empty circle.
const odd = structuredClone(OSASUNA);
odd.fixture.home_team = 'A <b>"Team"</b>';
odd.visuals.home_team_logo = null;
const oddCard = wins.homeWinsCard(odd, h);
assert.doesNotMatch(oddCard, /<b>"Team"/);
assert.match(oddCard, /A &lt;b&gt;&quot;Team&quot;&lt;\/b&gt; v Athletic Club/);
assert.match(oddCard, /<span class="hw-crest"><\/span><b>1–1<\/b>/);
// Public copy rules: no bookmaker, no promises, no tipster words.
for (const card of [value, likely]) assert.doesNotMatch(card, /bookmaker|profit|ROI|units|guarantee|lock|banker|\bAI\b|bet now/i);

// Each card's tilt (−5° to 5°, in half degrees) and height come from its match, so they never change between visits.
const seeds = many(12).map(w => wins.homeWinsSeed(w.fixture.event_id));
assert.deepEqual(wins.homeWinsSeed('1552219'), wins.homeWinsSeed('1552219'));
for (const s of seeds) {
  assert.ok(s.tilt >= -5 && s.tilt <= 5 && Number.isInteger(s.tilt * 2), `tilt ${s.tilt}`);
  assert.ok(Number.isInteger(s.dy) && s.dy >= -18 && s.dy <= 18, `dy ${s.dy}`);
}
assert.ok(new Set(seeds.map(s => s.tilt)).size >= 6, 'the tilts vary from card to card');

// The week as the heading says it.
assert.equal(wins.homeWinsWeek(WEEK), '4 to 8 Nov');
assert.equal(wins.homeWinsWeek({ from: '2026-10-30', to: '2026-11-03' }), '30 Oct to 3 Nov');
assert.equal(wins.homeWinsWeek({ from: '2026-11-08', to: '2026-11-08' }), '8 Nov');
// An expected count: a half says both whole numbers.
assert.equal(wins.homeWinsAbout(4.5), 'about 4 or 5');
assert.equal(wins.homeWinsAbout(243.5), 'about 243 or 244');
assert.equal(wins.homeWinsAbout(4), 'about 4');
assert.equal(wins.homeWinsAbout(1.2), 'about 1');

// The record beside the reel: both lists, losses counted. Most likely says its chance and what happened; Value, with
// ten picks, gives counts only.
assert.equal(wins.homeWinsRecord({ week: WEEK, record: RECORD }, h),
  'Every pick counts, losses too. From 4 to 8 Nov Spix made <b>353 Most likely picks</b> and said about 2 in 3 would'
  + ' come in (69%). <b>248 did</b> (70%). It made <b>10 Value picks</b> and expected about 4 or 5 to win.'
  + ' <b>3 did</b>, and 2 were void (stake back).');
assert.equal(wins.homeWinsRecord({ week: WEEK, record: [{ ...RECORD[1], void: 1 }] }, h),
  'Every pick counts, losses too. From 4 to 8 Nov Spix made <b>10 Value picks</b> and expected about 4 or 5 to win.'
  + ' <b>3 did</b>, and 1 was void (stake back).');
assert.equal(wins.homeWinsRecord({ week: WEEK, record: [{ ...RECORD[1], picks: 1, won: 1, void: 0, expected_wins: 0.6 }] }, h),
  'Every pick counts, losses too. From 4 to 8 Nov Spix made <b>1 Value pick</b> and expected about 1 to win. <b>1 did</b>.');
assert.equal(wins.homeWinsRecord({ week: WEEK, record: [] }, h), '');

// The section shows only with a real feed and enough wins to fill the reel without looking like more than there were.
assert.equal(wins.HOME_WINS_MIN, 6);
assert.equal(wins.homeWinsReady({ version: 'home_wins.v1', week: WEEK, wins: many(6), record: RECORD }), true);
assert.equal(wins.homeWinsReady({ version: 'home_wins.v1', week: WEEK, wins: many(5), record: RECORD }), false);
assert.equal(wins.homeWinsReady({ version: 'home_wins.v2', week: WEEK, wins: many(12), record: RECORD }), false);
assert.equal(wins.homeWinsReady({ version: 'home_wins.v1', week: WEEK, wins: null, record: RECORD }), false);
assert.equal(wins.homeWinsReady(null), false);

// The reel: the set twice in one track, so moving it left by half loops without a seam. Only the first set is read
// out; the second is hidden and out of the tab order.
const slots = html => (html.match(/<li class="hw-slot"/g) || []).length;
const sets = html => html.split('<ul class="hw-set"').slice(1);
const twelve = wins.homeWinsTrack(many(12), h);
assert.equal(sets(twelve).length, 2);
assert.match(twelve, /^<ul class="hw-set" aria-label="Picks that came in">/);
assert.match(sets(twelve)[1], /^ aria-hidden="true" inert>/);
assert.deepEqual(sets(twelve).map(slots), [12, 12]);
assert.doesNotMatch(sets(twelve)[0], /aria-hidden/, 'every card of the first set is read once');
// Each slot carries its own tilt and height.
const s0 = wins.homeWinsSeed('1600000');
assert.match(twelve, new RegExp(`<li class="hw-slot" style="--tilt:${s0.tilt}deg;--dy:${s0.dy}px">`));
// A shorter week is repeated inside each set until a set is wider than any screen; the repeats are not read out.
assert.equal(wins.HOME_WINS_FILL, 12);
const seven = wins.homeWinsTrack(many(7), h);
assert.deepEqual(sets(seven).map(slots), [12, 12], 'both sets the same length, so the loop still has no seam');
assert.equal((sets(seven)[0].match(/<li class="hw-slot" aria-hidden="true" inert/g) || []).length, 5);
assert.equal((sets(seven)[0].match(/role="img"/g) || []).length, 12);
// A constant speed on any screen: one set's width over 28 px a second.
assert.equal(wins.HOME_WINS_SPEED, 28);
assert.equal(wins.homeWinsDuration(3216), '114.9s');
assert.equal(wins.homeWinsDuration(2760), '98.6s');

// The page: the section sits under "Useful before you decide", hidden until the feed has arrived, with its pause
// control (moving content needs one), 18+ and the help line.
const benefitsEnd = page.indexOf('</section>', page.indexOf('id="how-it-works"'));
const winsAt = page.indexOf('<section class="home-wins"');
assert.ok(benefitsEnd > 0 && winsAt > benefitsEnd && winsAt < page.indexOf('id="pricing"'), 'right under the benefits');
const section = page.slice(winsAt, page.indexOf('</section>', winsAt));
assert.match(section, /^<section class="home-wins" id="homeWins" aria-labelledby="homeWinsTitle" hidden>/);
assert.match(section, /<h2 id="homeWinsTitle">Picks that came in<\/h2>/);
// Its label says what it will do (Pause, then Play), so it is not also a pressed toggle.
assert.match(section, /<button type="button" class="hw-pause" id="homeWinsPause"><svg[^>]*aria-hidden="true"[\s\S]*<span>Pause<\/span><\/button>/);
assert.doesNotMatch(section, /aria-pressed/);
assert.match(section, /18\+/);
assert.match(section, /href="https:\/\/www\.gamcare\.org\.uk\/get-support\/"/);
assert.match(page, /<link rel="stylesheet" href="\/static\/home-wins\.css\?v=wins-20261112a">/);
assert.match(page, /<script src="\/static\/home-wins\.js\?v=wins-20261112a"><\/script>/);
// The feed is read from the backend's endpoint; nothing is written.
const script = fs.readFileSync(path.join(STATIC, 'home-wins.js'), 'utf8');
assert.match(script, /fetch\('\/api\/home\/wins'/);
assert.doesNotMatch(script, /method:\s*'(POST|PUT|DELETE)'/);
