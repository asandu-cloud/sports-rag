// Research Area overhaul task 2 (D107): the team page answers first, as in the owner's approved sketch. Chelsea's real
// Premier League 2026/27 profile, saved from the live API on 15 Nov; no browser or network.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');

const text = html => html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').replace(/ ([.,:])/g, '$1').trim();

(async () => {
  const h = await import(pathToFileURL(path.join(__dirname, '../web_app/static/research/insight.js')));
  const chelsea = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures/research_team_chelsea.json'), 'utf8'));
  const rows = chelsea.match_log.map(h.fromProfileRow);

  // The lead: Chelsea's real last 10 had 4+ cards in 9, so they are card-heavy; the count is the one marked figure.
  const cards = h.teamLead('Chelsea', rows, 'cards');
  assert.equal(text(cards), "Chelsea's matches are card-heavy: 9 of their last 10 had 4 or more cards.");
  assert.match(cards, /<mark>9 of their last 10<\/mark>/);
  assert.equal((cards.match(/<mark>/g) || []).length, 1);
  // Lead sentence uses the real count: six matches say "last 6", never padded to 10.
  assert.equal(text(h.teamLead('Chelsea', rows.slice(0, 6), 'cards')), "Chelsea's matches are card-heavy: 5 of their last 6 had 4 or more cards.");
  // Under 7 in 10 there is no "heavy" word: 6 of the 10 had 3+ goals.
  assert.equal(text(h.teamLead('Chelsea', rows, 'goals')), "Chelsea's matches: 6 of their last 10 had 3 or more goals.");
  assert.equal(text(h.teamLead('Chelsea', rows, 'corners')), "Chelsea's matches are corner-heavy: 7 of their last 10 had 10 or more corners.");
  assert.equal(h.teamLead('Chelsea', [], 'cards'), '', 'no matches, no lead');

  // Who got the cards, a match: Chelsea 2.1, their opponents 3.4 (the sketch's sub-line).
  assert.equal(h.teamLeadSub('Chelsea', rows, 'cards'), 'their opponents got most of the cards (3.4 a match, Chelsea 2.1)');
  assert.equal(h.teamLeadSub('Chelsea', rows, 'goals'), 'Chelsea scored 1.9 a match and conceded 1.6');

  // Last 10 at a glance, cards: the sketch's four rows, each a strip and a written count.
  const glance = h.glance('Chelsea', rows, 'cards');
  const lines = glance.split('<div class="g-row">').slice(1).map(text);
  assert.deepEqual(lines, ['4+ cards in the match 9 of 10', '5+ cards in the match 7 of 10',
    'Chelsea booked 2+ times 5 of 10', 'Opponent booked 3+ times 7 of 10']);
  assert.equal((glance.match(/<span class="strip"/g) || []).length, 4);
  assert.match(glance, /title="at SUN: 9"/, 'each square names its match');
  assert.deepEqual(h.glance('Chelsea', rows, 'goals').split('<div class="g-row">').slice(1).map(text),
    ['3+ goals in the match 6 of 10', 'Both teams scored 5 of 10', 'Chelsea scored 2+ 6 of 10']);
  assert.deepEqual(h.glance('Chelsea', rows, 'corners').split('<div class="g-row">').slice(1).map(text),
    ['10+ corners in the match 7 of 10', 'Chelsea won 5+ corners 4 of 10']);

  // Cards by match, up and down (owner, 17 Nov, D118): the team's bar rises above the line in Spix cobalt, the
  // opponent's hangs below it in grey, and the names sit on the chart, so no colour has to say who is who. The match
  // total moves under each column's label and is marked when it clears the line, since a split column cannot carry it.
  const chart = h.stackedColumns('Chelsea', rows, 'cards');
  const cols = chart.match(/<div class="ud-col"[^>]*>[\s\S]*?<\/div>/g);
  assert.equal(cols.length, 10);
  assert.match(cols[9], /aria-label="at Sunderland, 8 Nov: Chelsea 4, Sunderland 5, 9 in all"/);
  assert.match(cols[9], /<i class="ud-up" style="height:33\.33%"><\/i><i class="ud-down" style="height:41\.67%"><\/i>/, 'Chelsea 4 and Sunderland 5 on a 0-6 scale');
  assert.match(chart, /<p class="ud-name ud-name-up">↑ Chelsea<\/p>/);
  assert.match(chart, /<p class="ud-name ud-name-down">↓ Opponents<\/p>/);
  const xs = chart.match(/<span class="ud-xl">[\s\S]*?<\/span>/g);
  assert.equal(xs[9], '<span class="ud-xl"><b><i>at</i> SUN</b><mark>9</mark></span>', 'on phones v or at takes its own line');
  assert.equal(xs[4], '<span class="ud-xl"><b><i>at</i> BRE</b>2</span>', 'a total under the line is not marked');
  assert.equal(xs.filter(x => x.includes('<mark>')).length, 9);
  assert.doesNotMatch(chart, /sc-line|sc-col/);
  assert.match(text(chart), /Chelsea 2\.1 a match Opponents 3\.4 a match/);
  assert.match(text(chart), /The number under each match is its total, marked when over 3\.5 cards\./);
  const corners = h.stackedColumns('Chelsea', rows, 'corners');
  assert.match(text(corners), /marked when over 9\.5 corners/);
  assert.equal((corners.match(/<mark>/g) || []).length, 7);

  // A match with no figures draws a dash at the line; the scale ticks run both ways from 0.
  const ud = h.upDown({name: 'Bayern München', items: [{up: 4.06, down: 0.79, label: 'v Stuttgart', where: 'v', who: 'STU', sub: '5–1'}, {up: null, down: null, label: 'v Union Berlin', where: 'v', who: 'UNI', sub: '7–0'}]});
  assert.match(ud, /<div class="ud-col" tabindex="0" data-tip="v Union Berlin" role="img" aria-label="v Union Berlin"><span class="ud-miss">–<\/span><\/div>/);
  assert.match(ud, /<span class="ud-xl"><b><i>v<\/i> STU<\/b>5–1<\/span>/);
  assert.deepEqual([...ud.matchAll(/<span style="top:([\d.]+)%">(\d)<\/span>/g)].map(m => [m[2], +m[1]]), [['6', 0], ['4', 16.67], ['2', 33.33], ['0', 50], ['2', 66.67], ['4', 83.33], ['6', 100]]);

  // Spix cobalt for the team in both themes (owner); the name above is text, so it keeps the readable accent.
  const css = fs.readFileSync(path.join(__dirname, '../web_app/static/research/research.css'), 'utf8');
  const style = fs.readFileSync(path.join(__dirname, '../web_app/static/style.css'), 'utf8');
  assert.equal((style.match(/--chart-team: #0246A4;/g) || []).length, 2, 'light and dark');
  assert.match(css, /\.profile \.ud-up\{[^}]*background:var\(--chart-team\)/);
  assert.match(css, /\.profile \.ud-down\{[^}]*background:var\(--chart-muted\)/);
  assert.match(css, /\.profile \.ud-name-up\{[^}]*color:var\(--accent-l\)/);
  assert.match(css, /\.profile \.ud-x b i\{font-style:normal\}/);
  assert.match(css, /@media\(max-width:640px\)\{[^\n]*\.profile \.ud-x b i\{display:block\}/);
  const research = fs.readFileSync(path.join(__dirname, '../web_app/static/research/research.js'), 'utf8');
  assert.match(research, /chart = upDown\(\{name, items: last\.map\(m => \(\{up: m\.values\[forKey\], down: m\.values\[againstKey\]/, 'the chances chart is drawn the same way');
  assert.match(research, /where: venueWord\(m\), who: m\.opponent\?\.abbreviation \|\| short\(m\.opponent\), sub: score\(m\)/);

  // Home and away (owner, 17 Nov): the bar joining each home and away dot is a 4px line on the track. It was named
  // "link", which the text-button rule (.link, min-height 44px) also matched, so it drew as a 44px block.
  const js = research;
  assert.match(js, /<span class="dumb-track" role="img" aria-label="\$\{escape\(text\)\}"><span class="dumb-link" style="left:/);
  assert.doesNotMatch(js, /class="link"/);
  assert.match(css, /\.profile \.dumb-track \.dumb-link\{position:absolute;top:9px;height:4px;/);
  assert.doesNotMatch(css, /\.dumb-track \.link\b/);

  console.log('test_research_team_web: ok');
})().catch(error => { console.error(error); process.exit(1); });
