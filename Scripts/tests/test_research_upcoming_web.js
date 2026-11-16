// Research Area overhaul task 5 (D107): a team's next matches as a rail (desktop), a floating bar and a sheet (phone,
// the owner's option C), and the next match in a player's or referee's head. Chelsea's real next three as B1 gives
// them on 16 Nov (no referee named, no read out yet); the ready state is that same match with a read added here.
process.env.TZ = 'Europe/London'; // the cards show the viewer's local time; the test reads it in London
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');

const text = html => html.replace(/<[^>]+>/g, ' ').replace(/&#39;/g, "'").replace(/&amp;/g, '&').replace(/\s+/g, ' ').replace(/ ([.,:])/g, '$1').trim();
const fixture = name => JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures', name), 'utf8'));
const static_ = file => fs.readFileSync(path.join(__dirname, '../web_app/static', file), 'utf8');

(async () => {
  const u = await import(pathToFileURL(path.join(__dirname, '../web_app/static/research/upcoming.js')));
  const chelsea = fixture('research_upcoming_chelsea.json');
  const [leeds, forest] = chelsea.matches;
  const now = new Date('2026-11-16T12:00:00Z');

  // The read's state.
  assert.equal(u.readState({status: 'ready'}, now), 'Read ready');
  assert.equal(u.readState({status: 'preliminary'}, now), 'Preliminary');
  assert.equal(u.readState({status: 'waiting', opens_at: '2026-11-19T15:00:00+00:00'}, now), 'Read due Thu');
  assert.equal(u.readState(leeds.read, now), 'Read due Tue', 'Leeds: 96 hours before a Saturday 15:00 kickoff');
  assert.equal(u.readState({status: 'waiting', opens_at: '2026-11-16T18:00:00+00:00'}, now), 'Read due today');
  assert.equal(u.readState({status: 'waiting', opens_at: '2026-11-28T19:30:00+00:00'}, now), 'Read due 28 Nov');
  assert.equal(u.readState({status: 'waiting', opens_at: null}, now), 'Read not out yet');

  // A card as the API gives it today: no referee, no read, so no figures.
  const card = u.nextCard(leeds, {now});
  assert.match(card, /data-open-match="1557470"/);
  assert.match(text(card), /Sat 21 Nov · 15:00 Premier League/);
  assert.match(text(card), /v Leeds Home/);
  assert.match(text(card), /Referee not named yet/);
  assert.match(text(card), /Spix's read comes out Tue 17 Nov\. Past numbers are ready now\./);
  assert.doesNotMatch(card, /class="nm-figs"><span><b>/, 'no figures before the read');
  assert.match(text(u.nextCard(forest, {now})), /at Nottingham Forest Away/);
  assert.match(u.nextCard(leeds, {now, active: '1557470'}), /aria-current="true"/);

  // The same match once named and read: the referee and Spix's three figures.
  const ready = structuredClone(leeds);
  ready.referee = {name: 'Anthony Taylor', referee_key: 'a-taylor-68af826d'};
  ready.read = {status: 'ready', read_at: '2026-11-17T15:10:00+00:00', opens_at: null, expected: {goals: 2.74, corners: 10.1, cards: 4.6}};
  const readyCard = text(u.nextCard(ready, {now}));
  assert.match(readyCard, /Referee Anthony Taylor/);
  assert.match(readyCard, /2\.7 goals 10\.1 corners 4\.6 cards Spix expects/);
  assert.match(readyCard, /Read ready/);
  // A referee's appointment names both teams.
  assert.match(text(u.nextCard({...leeds, venue: null}, {now})), /Chelsea v Leeds/);

  // The phone's bar and the rail.
  const bar = u.barHTML(chelsea);
  assert.match(text(bar), /^Next: v Leeds Sat 21 Nov · 15:00 · and 2 more Open ›$/);
  assert.match(bar, /data-open-sheet/);
  const single = u.barHTML({...chelsea, matches: [leeds]});
  assert.doesNotMatch(text(single), /more/);
  assert.match(single, /data-open-match="1557470"/, 'one match opens it straight away');
  const rail = u.railHTML(chelsea, {now});
  assert.equal((rail.match(/class="nm"/g) || []).length, 3);
  assert.match(text(rail), /Next matches Chelsea's next three, in every competition/);
  const sheet = u.sheetHTML(chelsea, {now});
  assert.match(sheet, /role="dialog"/);
  assert.match(sheet, /aria-modal="true"/);
  assert.match(text(sheet), /Chelsea's next matches/);
  assert.equal((sheet.match(/class="nm"/g) || []).length, 3);
  assert.match(text(u.sheetHTML({subject: {name: 'Leeds'}, matches: [leeds]}, {now})), /Leeds' next match/);

  // No upcoming means no rail and no bar.
  for (const empty of [null, undefined, {matches: []}, fixture('research_upcoming_taylor.json')]) {
    assert.equal(u.railHTML(empty), '');
    assert.equal(u.barHTML(empty), '');
    assert.equal(u.sheetHTML(empty), '');
  }

  // The head card on a player's or referee's page.
  assert.match(text(u.headNext(leeds, 'Next match')), /^Next match v Leeds · Sat 21 Nov, 15:00 Open ›$/);
  assert.match(text(u.headNext({...leeds, venue: null}, 'Next appointment')), /^Next appointment Chelsea v Leeds · Sat 21 Nov, 15:00 Open ›$/);
  assert.equal(u.headNext(null, 'Next match'), '');

  // The bar sits over the slip's bar: the CSS reads --slip-bar-h, app.js sets it where it shows the slip bar and
  // marks the research host while the slip is open.
  assert.match(static_('research/research.css'), /\.next-bar\{[^}]*bottom:calc\(var\(--slip-bar-h, ?0px\)/);
  assert.match(static_('research/research.css'), /:host\(\[data-slip-open\]\) \.next-bar\{display:none\}/);
  const app = static_('app.js');
  assert.match(app, /bar\.classList\.toggle\('hidden', n === 0\);\s*\n\s*this\.researchBar\(\);/);
  assert.match(app, /\n  researchBar\(\) \{[\s\S]*?setProperty\('--slip-bar-h'[\s\S]*?toggleAttribute\('data-slip-open', this\.open\)/);
  assert.match(app, /ui\.overlay\('betSlip'[^\n]*\n\s*this\.researchBar\(\);/, 'opening or closing the slip updates it too');

  console.log('test_research_upcoming_web: ok');
})().catch(error => { console.error(error); process.exit(1); });
