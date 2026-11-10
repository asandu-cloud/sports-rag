// The team's next match under a Research Assistant answer (D96, the owner's option B for D95): one more row in the
// For / Against / In the match table, Spix's pick under it in the marker with a link to the match page, the note
// while the read is not out, and the day on the board a match link opens. Real answers; no browser or network.
process.env.TZ = 'Europe/London';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../web_app/static/app.js'), 'utf8');
const esc = value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const helpersStart = source.indexOf('// ----- Pure render helpers');
const helpersEnd = source.indexOf('// ----- End of pure render helpers -----');
assert.ok(helpersStart >= 0 && helpersEnd > helpersStart);
const h = Function('esc', source.slice(helpersStart, helpersEnd)
  + 'return { renderAssistantCard, boardDayOffset, recsClock };')(esc);
const real = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures/research_assistant_next_match.json'), 'utf8'));
// Inline tags (the marker, a bold figure) leave no gap; others read as a space.
const text = html => html.replace(/<\/?(mark|b)\b[^>]*>/g, '').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
const card = key => real[key].answers[0];
// Rows of the answer's own table, not of "The N matches behind this" further down.
const rows = html => html.split('<details')[0].match(/<tr[\s>][\s\S]*?<\/tr>/g) || [];

// Corners: the row says what Spix expects in the match; Spix prices only the whole match's corners, so the team's
// own columns get a grey dash and the match figure is the one in bold.
const corners = h.renderAssistantCard(card('corners'));
const cornerRows = rows(corners);
assert.equal(cornerRows.length, 3, 'the header, the average and the next match');
assert.equal(text(cornerRows[2]), 'Spix expects v Brighton, Sun 8 Nov – – 9.5');
assert.equal((cornerRows[2].match(/class="n ra-dash"/g) || []).length, 2);
assert.match(cornerRows[2], /<td class="n ra-strong">9\.5<\/td>/);
assert.doesNotMatch(cornerRows[2], /marker|ra-mark/, 'no colour on the row');
// Under the table: the pick in the marker, its chance in words and its typical price, then the match page.
assert.match(text(corners), /Spix's pick for that match: Over 8\.5 corners, about 2 in 3, at 1\.36\. Match page/);
assert.match(corners, /<mark class="ra-mark">Over 8\.5 corners<\/mark>/);
assert.match(corners, /data-ra-match="1557458" data-ra-kickoff="2026-11-08T14:00:00\+00:00"[^>]*>Match page</);
assert.ok(corners.indexOf('Spix\'s pick') > corners.indexOf('</table>') && corners.indexOf('Spix\'s pick') < corners.indexOf('ra-note'),
  'the pick sits right under the table');
// The footer says when Spix's figures were taken.
assert.match(text(corners), /Data up to Sat 7 Nov 2026 · Spix's figures for Brighton v Brentford as of Sun 8 Nov, 13:56/);

// Goals: Spix splits the goals by side, so the row has all three figures (Brentford away: 1.3 for, 1.9 against).
const goals = h.renderAssistantCard(card('goals'));
assert.equal(text(rows(goals)[2]), 'Spix expects v Brighton, Sun 8 Nov 1.3 1.9 3.1');
assert.doesNotMatch(rows(goals)[2], /ra-dash/);
assert.match(text(goals), /Spix's pick for that match: Over 2\.5 goals, about 3 in 5, at 1\.57\./);

// A line question has no table: the match figure gets its own short line above the same pick, never the asked
// line's own chance (owner).
const line = h.renderAssistantCard(card('line'));
assert.doesNotMatch(line.split('<details')[0], /<table/);
assert.match(text(line), /Spix expects 9\.5 corners v Brighton, Sun 8 Nov\. Spix's pick for that match: Over 8\.5 corners, about 2 in 3, at 1\.36\./);
assert.doesNotMatch(text(line), /7\.5 corners, about/);

// No pick in that market: the figures stay, the note says so in grey, and the match page is still there.
const noPick = h.renderAssistantCard(card('no_pick'));
assert.equal(rows(noPick).length, 3);
assert.match(noPick, /<p class="ra-next quiet">No pick from Spix in corners for that match\. <button[^>]*data-ra-match="1557458"/);
assert.doesNotMatch(noPick, /ra-mark/);

// Before the read is out: only its note, in grey, under the table. No row, no pick, no link, no figures time.
const waiting = h.renderAssistantCard(card('waiting'));
assert.equal(rows(waiting).length, 2);
assert.match(waiting, /<p class="ra-next quiet">Spix's read of Brentford v Everton \(Mon 23 Nov\) comes out from Thu 19 Nov\.<\/p>/);
assert.doesNotMatch(waiting, /data-ra-match|ra-mark|Spix's figures/);

// No next match (another stat, no fixture, or a failure): the card is as it was.
const { next_match: _, ...plain } = card('corners');
const without = h.renderAssistantCard(plain);
assert.doesNotMatch(without, /ra-next|ra-mark|data-ra-match|Spix's figures/);
assert.equal(rows(without).length, 2);
assert.equal(h.renderAssistantCard({ ...plain, next_match: null }), without);
// A read without figures keeps the pick line and skips the row.
const noFigures = h.renderAssistantCard({ ...card('corners'), next_match: { ...card('corners').next_match, expected: null } });
assert.equal(rows(noFigures).length, 2);
assert.match(text(noFigures), /Spix's pick for that match: Over 8\.5 corners/);
// Whatever the next match says is escaped.
const hostile = { ...card('corners').next_match, label: '<img src=x onerror=alert(1)>', pick: { ...card('corners').next_match.pick, label: '<b>x</b>' },
  fixture: { ...card('corners').next_match.fixture, event_id: '"><script>' } };
assert.doesNotMatch(h.renderAssistantCard({ ...card('corners'), next_match: hostile }), /<img|<b>x|"><script>/);

// The match page opens on the match's day of the board's week: today is 0, the board shows seven days.
const now = new Date('2026-11-06T10:00:00Z');
assert.equal(h.boardDayOffset('2026-11-08T14:00:00+00:00', now), 2);
assert.equal(h.boardDayOffset('2026-11-06T23:30:00+00:00', now), 0);
assert.equal(h.boardDayOffset('2026-11-12T20:00:00+00:00', now), 6);
assert.equal(h.boardDayOffset('2026-11-13T20:00:00+00:00', now), null, 'past the board');
assert.equal(h.boardDayOffset('2026-11-05T20:00:00+00:00', now), null, 'already played');
assert.equal(h.boardDayOffset('not a date', now), null);
// The board's clock is unchanged by the shared parsing.
assert.equal(h.recsClock('2026-11-08T13:56:26'), '13:56');
assert.equal(h.recsClock(''), '');

console.log('research assistant next match web: all checks passed');
