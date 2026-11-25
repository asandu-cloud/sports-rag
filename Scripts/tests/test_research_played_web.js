// Played-match report (D121, task 1): a match from a team's last 10, opened from "Last 10 at a glance". Como's real
// Serie A 2026/27 last 10, saved from the live API on 17 Nov. The picks follow D122's spix_before_kickoff field, filled
// with the picks the board really kept at kickoff (1 and 8 Nov); the outcomes are what those matches settled to.
process.env.TZ = 'Europe/London';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');

const text = html => html.replace(/<[^>]+>/g, ' ').replace(/&#39;/g, "'").replace(/&amp;/g, '&').replace(/\s+/g, ' ').replace(/ ([.,:%])/g, '$1').trim();
const sectionOf = (html, label) => html.match(new RegExp(`<section class="pr-card"[^>]*aria-label="${label}"[\\s\\S]*?</section>`))?.[0] || '';

(async () => {
  const p = await import(pathToFileURL(path.join(__dirname, '../web_app/static/research/played.js')));
  const como = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures/research_team_como.json'), 'utf8'));
  const pick = (label, market, chance, outcome) => ({label, market, chance, outcome});
  const venezia = {read_at: '2026-11-01T11:02:11Z',
    most_likely: [pick('Como to win', 'moneyline', 0.7463, 'won'), pick('Under 4.5 goals', 'goals', 0.7166, 'won'), pick('Venezia +2.5', 'spreads', 0.6889, 'won')],
    value: [pick('Venezia +2', 'spreads', 0.7948, 'won'), pick('Over 7.5 corners', 'corners', 0.8398, 'won')]};
  const inter = {read_at: '2026-11-08T16:51:42Z', value: [],
    most_likely: [pick('Como +1.5', 'spreads', 0.7468, 'lost'), pick('Over 7.5 corners', 'corners', 0.7184, 'lost'), pick('Both teams to score', 'btts', 0.6553, 'lost')]};
  const withReads = structuredClone(como);
  withReads.match_log.forEach((m, i) => { m.spix_before_kickoff = i === 8 ? venezia : i === 9 ? inter : null; });

  // The head: home side first, as the score is written; the team's own venue and the result's status.
  const ven = p.playedReport(withReads, 8, 'corners');
  assert.match(ven, /<div class="mv-head pr-top">[\s\S]*?<h2 class="mv-title" id="matchTitle" tabindex="-1">Como 1–0 Venezia<\/h2>/);
  assert.match(text(ven), /Sun 1 Nov · 11:30 Serie A Como at home Full time/);
  assert.match(p.playedReport(withReads, 9, 'corners'), /<h2[^>]*>Inter 2–0 Como<\/h2>/);
  assert.match(ven, /<button type="button" class="text-button back-button" data-match-back>[\s\S]*?<span>Como<\/span><\/button>/);

  // The last 10, oldest first: each a button with the result from Como's side; a dot where a read was kept.
  const strip = ven.match(/<nav class="pr-strip"[\s\S]*?<\/nav>/)[0];
  const pills = strip.match(/<button[\s\S]*?<\/button>/g);
  assert.equal(pills.length, 10);
  assert.deepEqual(pills.map(text), ['NAP W 2–1', 'GEN W 4–1', 'PAR W 2–1', 'FRO L 0–2', 'ROM D 0–0', 'FIO W 3–1', 'SAS W 2–0', 'TOR W 3–1', 'VEN W 1–0', 'INT L 0–2']);
  assert.match(pills[8], /data-played="8" aria-current="true"/);
  assert.match(pills[0], /data-played="0" aria-current="false"/);
  assert.equal((strip.match(/class="pr-kept"/g) || []).length, 2, 'only the matches with a kept read');
  assert.match(pills[8], /aria-label="v Venezia, Sun 1 Nov: won 1–0, Spix&#39;s read kept"/);
  assert.doesNotMatch(p.playedReport(como, 8, 'corners'), /pr-kept/, 'no field yet, no dots');

  // The markets, as on the team page, and the lead for the open one, against the glance's first row (10+, 4+, 3+).
  assert.match(ven, /<nav class="mkts" aria-label="Markets">[\s\S]*?data-played-market="corners" aria-current="true"/);
  assert.equal(text(ven.match(/<p class="pr-lead">[\s\S]*?<\/p>/)[0]), '9 corners, one short of 10.');
  assert.match(ven, /<p class="pr-lead"><mark>9 corners<\/mark>, one short of 10\.<\/p>/);
  assert.equal(text(ven.match(/<p class="pr-sub">[\s\S]*?<\/p>/)[0]), 'Como won 8 and conceded 1. Over their last 10 they won 6.3 a match and conceded 4.5.');
  const lead = (i, m) => { const html = p.playedReport(como, i, m); return text(html.match(/<p class="pr-lead">[\s\S]*?<\/p>/)[0]) + ' | ' + text(html.match(/<p class="pr-sub">[\s\S]*?<\/p>/)[0]); };
  assert.equal(lead(8, 'cards'), '3 cards, one short of 4. | Como got 1 and Venezia 2. Over their last 10, Como got 1.2 a match and their opponents 1.8.');
  assert.equal(lead(8, 'goals'), '1 goal, under 3. | Como scored 1 and conceded 0. Over their last 10 they scored 1.7 a match and conceded 0.9.');
  assert.equal(lead(5, 'corners'), '16 corners: a 10+ match. | Como won 5 and conceded 11. Over their last 10 they won 6.3 a match and conceded 4.5.');
  assert.equal(lead(9, 'corners').split(' | ')[0], '6 corners, under 10.');
  const blank = structuredClone(como); blank.match_log[8].values.corners_against = null;
  assert.equal(text(p.playedReport(blank, 8, 'corners').match(/<p class="pr-lead">[\s\S]*?<\/p>/)[0]), 'Corners were not recorded for this match.');

  // What Spix said: left out while the field is absent, said plainly when no read was kept, else each pick settled.
  assert.doesNotMatch(p.playedReport(como, 8, 'corners'), /What Spix said/);
  const none = sectionOf(p.playedReport(withReads, 5, 'corners'), 'What Spix said before kickoff');
  assert.match(text(none), /No read from before kickoff was kept for this match, so there are no picks to show\./);
  const said = sectionOf(ven, 'What Spix said before kickoff');
  assert.match(text(said), /^What Spix said before kickoff As the board showed it at kickoff · read at 11:02, 28 minutes before 5 of 5 picks came in Most likely/);
  const rows = said.match(/<div class="pr-pick[^"]*">[\s\S]*?<\/div>/g).map(text);
  assert.deepEqual(rows, ['Como to win 75% Came in 1–0', 'Under 4.5 goals 72% Came in 1 goal', 'Venezia +2.5 69% Came in 1–0',
    'Venezia +2 79% Came in 1–0', 'Over 7.5 corners 84% Came in 9 corners']);
  assert.match(said, /<div class="pr-pick on">[\s\S]*?Over 7\.5 corners/, 'the open market\'s pick is marked');
  assert.match(text(said), /Value Venezia \+2/);
  assert.match(text(said), /The percentage is Spix's chance at the time\. One match says little on its own\./);
  const missed = sectionOf(p.playedReport(withReads, 9, 'goals'), 'What Spix said before kickoff');
  assert.match(text(missed), /None of the 3 picks came in/);
  assert.deepEqual(missed.match(/<div class="pr-pick[^"]*">[\s\S]*?<\/div>/g).map(text),
    ['Como +1.5 75% Did not come in 2–0', 'Over 7.5 corners 72% Did not come in 6 corners', 'Both teams to score 66% Did not come in 2–0']);
  assert.doesNotMatch(text(missed), /Value/, 'no Value heading without Value picks');
  // The other outcomes say what happened to the stake; the summary counts only what came in.
  assert.equal(p.pickSummary([{outcome: 'won'}, {outcome: 'won'}, {outcome: 'push'}, {outcome: 'half_won'}, {outcome: 'half_lost'}]),
    '2 of 5 picks came in, 1 half came in, 1 stake back');
  assert.equal(p.pickSummary([{outcome: 'won'}, {outcome: null}]), '1 of 2 picks came in, 1 not graded');
  // The icon's colour: in for won and half won, out for lost and half lost; a push and an ungraded pick stay neutral.
  const kinds = ['won', 'half_won', 'lost', 'half_lost', 'push', null].map(outcome => {
    const one = JSON.parse(JSON.stringify(como)), row = one.match_log[one.match_log.length - 2];
    row.spix_before_kickoff = {read_at: '2026-11-01T11:02:11Z', most_likely: [{label: 'x', market: 'goals', chance: 0.7, outcome}], value: []};
    return p.playedReport(one, 8, 'goals').match(/class="pr-out ([a-z]+)"/)[1];
  });
  assert.deepEqual(kinds, ['in', 'in', 'out', 'out', 'open', 'open']);
  assert.equal(p.pickSummary([{outcome: 'won'}]), '1 of 1 pick came in');
  assert.deepEqual(['won', 'lost', 'push', 'half_won', 'half_lost', null].map(p.outcomeWords),
    ['Came in', 'Did not come in', 'Stake back', 'Half came in', 'Half lost', 'Not graded']);

  // The match in numbers: Como then the opponent, the bars scaled to the larger of the two, Como's usual beside.
  const nums = sectionOf(ven, 'The match in numbers');
  const lines = nums.match(/<div class="pr-row[^"]*">[\s\S]*?<\/div>/g);
  assert.deepEqual(lines.map(text), ['Goals 1 0 1.7', 'Expected goals 1.63 0.23 1.7', 'Shots 19 7 18.8', 'On target 4 1 5.8',
    'Corners 8 1 6.3', 'Cards 1 2 1.2', 'Possession 66% 34% 62%', 'Fouls 5 – 7.5']);
  assert.match(lines[4], /^<div class="pr-row on">/);
  assert.match(lines[4], /<i class="pr-own" style="width:100%"><\/i>[\s\S]*<i class="pr-opp" style="width:12\.5%"><\/i>/);
  assert.match(lines[5], /<i class="pr-own" style="width:50%"><\/i>[\s\S]*<i class="pr-opp" style="width:100%"><\/i>/);
  assert.match(text(nums), /COM VEN Usual/, 'short codes head the columns, so long names never cut off');
  assert.match(text(nums), /Venezia's fouls were not recorded\./);

  // Earlier and later matches; none past either end.
  const nav = ven.match(/<nav class="pr-nav"[\s\S]*?<\/nav>/)[0];
  assert.deepEqual(nav.match(/<button[\s\S]*?<\/button>/g).map(text), ['‹ Earlier Torino 1–3 Como Tue 27 Oct · 11 corners', 'Later › Inter 2–0 Como Sun 8 Nov · 6 corners']);
  assert.match(nav, /data-played="7"[\s\S]*data-played="9"/);
  assert.doesNotMatch(p.playedReport(como, 0, 'corners'), /‹ Earlier/);
  assert.doesNotMatch(p.playedReport(como, 9, 'corners'), /Later ›/);

  // Names are escaped; the foot line is the match view's.
  const odd = structuredClone(como); odd.match_log[8].away.name = 'V<e>nezia'; odd.match_log[8].opponent.name = 'V<e>nezia';
  assert.doesNotMatch(p.playedReport(odd, 8, 'corners'), /V<e>nezia/);
  assert.match(ven, /<p class="disclaimer">Past matches are records, not a forecast\. Spix's figures come from statistical models\. 18\+ · <a href="https:\/\/www\.gamcare\.org\.uk\/get-support\/"/);

  // research.js routes #research/played/teams/<id>/<scope>/<fixture_id>/<market> to the report, drawn in the match view.
  const research = fs.readFileSync(path.join(__dirname, '../web_app/static/research/research.js'), 'utf8');
  assert.match(research, /if \(parts\[0\] === 'played' && parts\[1\]\) return showPlayed\(/);
  assert.match(research, /import \{playedReport, glanceCard, glanceSheet\} from '\.\/played\.js\?v=research-20261125a'/);
  assert.match(research, /\[data-played\]/);
  assert.match(research, /\[data-played-market\]/);
  assert.match(research, /strip\.scrollLeft = pill\.offsetLeft - \(strip\.clientWidth - pill\.offsetWidth\) \/ 2/, 'a phone shows the open match in the strip');
  const css = fs.readFileSync(path.join(__dirname, '../web_app/static/research/research.css'), 'utf8');
  assert.match(css, /\.match \.pr-top\{flex-wrap:nowrap\}/);
  assert.match(css, /\.match \.pr-top \.mv-title\{flex:0 1 auto;min-width:0\}/);

  // ----- Task 2: the glance squares name their match -----
  const ins = await import(pathToFileURL(path.join(__dirname, '../web_app/static/research/insight.js')));
  const rows10 = como.match_log.map(ins.fromProfileRow);
  // With open, each square is a button for its match (no slow native tooltip); without, the strip is as before.
  const open = ins.glance('Como', rows10, 'corners', {open: true});
  const first = open.split('<div class="g-row">')[1];
  const squares = first.match(/<button[^>]*><\/button>/g);
  assert.equal(squares.length, 10);
  assert.equal(squares[8], '<button type="button" class="" data-glance="8" aria-label="v VEN: 9"></button>');
  assert.equal(squares[2], '<button type="button" class="y" data-glance="2" aria-label="v PAR: 15"></button>');
  assert.match(first, /<span class="strip" role="group" aria-label="5 of 10">/);
  assert.doesNotMatch(open, /title=/);
  assert.doesNotMatch(ins.glance('Como', rows10, 'corners'), /<button/, 'other strips keep their squares');

  // The card: when and where, the score, the market's figure split by side, and what Spix said when D122 says.
  const card = p.glanceCard(withReads, 8, 'corners');
  assert.equal(text(card), "Sun 1 Nov · Serie A · Como at home Como 1–0 Venezia Corners in the match 9 Como 8 · Venezia 1 · one short of 10 Spix before kickoff: 5 of 5 picks came in Click to open the match ›");
  assert.match(text(p.glanceCard(withReads, 9, 'cards')), /Cards in the match 1 Como 0 · Inter 1 · under 4 Spix before kickoff: None of the 3 picks came in/);
  assert.match(text(p.glanceCard(withReads, 5, 'goals')), /Goals in the match 4 Como 3 · Fiorentina 1: a 3\+ match Spix before kickoff: no read kept/);
  assert.doesNotMatch(p.glanceCard(como, 8, 'corners'), /Spix before kickoff/, 'no field yet, no line');

  // The phone sheet: the same match, earlier and later, and a button that opens the report.
  const sheet = p.glanceSheet(withReads, 8, 'corners');
  assert.match(sheet, /^<div class="sheet-bg" data-close-sheet><\/div>/);
  assert.match(sheet, /<div class="sheet gsheet" role="dialog" aria-modal="true" aria-labelledby="sheetTitle" tabindex="-1">/);
  assert.match(sheet, /<button type="button" class="gs-step" data-glance-step="7" aria-label="Earlier match">/);
  assert.match(sheet, /<button type="button" class="gs-step" data-glance-step="9" aria-label="Later match">/);
  assert.match(sheet, /<h3 id="sheetTitle">Como 1–0 Venezia<\/h3>/);
  assert.match(text(sheet), /Match 9 of 10 · Sun 1 Nov/);
  assert.deepEqual(sheet.match(/<div class="gs-tile">[\s\S]*?<\/div>/g).map(text), ['Corners in the match 9 Como 8 · Venezia 1', 'Spix before kickoff 5 of 5 picks came in']);
  assert.match(sheet, /<button type="button" class="gs-open" data-played-open="8">Open the match<\/button>/);
  assert.match(p.glanceSheet(como, 0, 'corners'), /data-glance-step="-1" aria-label="Earlier match" disabled/);
  assert.equal((p.glanceSheet(como, 8, 'corners').match(/class="gs-tile"/g) || []).length, 1, 'no Spix tile without the field');

  // research.js: the team page's glance opens; pointing shows the card; a click opens the report on a screen that can
  // hover, else the sheet; the profile already loaded serves the report without a second fetch.
  assert.match(research, /glance\(name, rows, m, \{open: true\}\)/);
  assert.match(research, /function wireGlance\(/);
  assert.match(research, /matchMedia\('\(hover: hover\) and \(pointer: fine\)'\)\.matches/);
  assert.match(research, /openSheet\(square, glanceSheet\(state\.profile, Number\(square\.dataset\.glance\), state\.teamMarket\)\)/);
  assert.match(research, /if \(!cache\.has\(url\)\) cache\.set\(url, Promise\.resolve\(state\.profile\)\)/);
  assert.match(css, /\.profile \.strip button\{/);
  assert.match(css, /\.profile \.strip button\.col::after\{/);

  console.log('test_research_played_web: ok');
})().catch(error => { console.error(error); process.exit(1); });
