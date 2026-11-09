// The sample match (D84): with ?preview=evidence a saved real card (Brighton v Brentford, 8 Nov, built by the
// production pick-evidence code) joins today's board three hours from now, labelled as a preview, so the pick evidence
// can be seen while no club matches are on. ?preview=off ends it. Nothing is written anywhere. No browser or network.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../web_app/static/app.js'), 'utf8');
const esc = value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const helpersStart = source.indexOf('// ----- Pure render helpers');
const helpersEnd = source.indexOf('// ----- End of pure render helpers -----');
assert.ok(helpersStart >= 0 && helpersEnd > helpersStart);
const h = Function('esc', source.slice(helpersStart, helpersEnd)
  + 'return { previewMode, previewMatchCard, renderPreviewNote, renderRecommendations, renderEvidenceSheet, recommendationsFor };')(esc);
const text = html => html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
const count = (haystack, needle) => haystack.split(needle).length - 1;

// On for the browser session once asked; off when asked; nothing else turns it on.
assert.equal(h.previewMode('?preview=evidence', null), 'evidence');
assert.equal(h.previewMode('', 'evidence'), 'evidence');
assert.equal(h.previewMode('?preview=off', 'evidence'), null);
assert.equal(h.previewMode('', null), null);
assert.equal(h.previewMode('?preview=yes', null), null);
assert.equal(h.previewMode('?preview=1', 'other'), null);

// The saved card is the real one: Brighton v Brentford on 8 Nov, with the story and evidence on every pick.
const saved = JSON.parse(fs.readFileSync(path.join(__dirname, '../web_app/static/preview/pick-evidence-card.json'), 'utf8'));
assert.equal(saved.fixture.home_team, 'Brighton');
assert.equal(saved.fixture.away_team, 'Brentford');
assert.equal(saved.fixture.league, 'EPL');
assert.match(saved.fixture.kickoff, /^2026-11-08T/);
assert.equal(saved.recommendations.version, 'recommendations.v1');
assert.equal(saved.recommendations.evidence.version, 'pick-evidence.v1');
assert.ok(saved.recommendations.story.lead);
assert.ok(saved.recommendations.most_likely.length >= 1 && saved.recommendations.most_likely.every(item => item.evidence?.verdict));

// Moved to three hours from now, on the half hour; the saved card itself is left as it was.
const now = new Date('2026-11-09T17:10:00Z');
const moved = h.previewMatchCard(saved, now);
assert.equal(moved.fixture.kickoff, '2026-11-09T20:30:00Z');
assert.equal(moved.preview, true);
assert.match(saved.fixture.kickoff, /^2026-11-08T/);
assert.equal(saved.preview, undefined);
assert.equal(h.previewMatchCard(saved, new Date('2026-11-09T17:30:00Z')).fixture.kickoff, '2026-11-09T20:30:00Z');
assert.equal(h.previewMatchCard(saved, new Date('2026-11-09T17:31:00Z')).fixture.kickoff, '2026-11-09T21:00:00Z');
assert.equal(moved.recommendations, saved.recommendations, 'the picks and evidence are the saved ones');
for (const bad of [null, undefined, {}, { fixture: null }, 'x']) assert.equal(h.previewMatchCard(bad, now), null);

// The note says what it is, on the board and on the match page.
const note = h.renderPreviewNote({ home: 'Brighton', away: 'Brentford' });
assert.match(note, /^<p class="preview-note">/);
assert.equal(text(note), 'Preview. Brighton v Brentford was played on 8 Nov. It is shown here as a sample match so the pick evidence can be seen while no club matches are on. It is not a real fixture today.');
assert.doesNotMatch(h.renderPreviewNote({ home: '<b>A</b>', away: 'B' }), /<b>A<\/b>/);

// The saved card draws the whole feature with the current helpers: the story, a verdict and rows on every pick, and
// a sheet behind every row.
const f = { id: moved.fixture.event_id, league: 'EPL', home: 'Brighton', away: 'Brentford', recommendations: h.recommendationsFor(moved, '') };
const page = h.renderRecommendations(f);
assert.equal(count(page, 'class="recs-story"'), 1);
assert.equal(count(page, 'class="recs-verdict"'), saved.recommendations.most_likely.length);
saved.recommendations.most_likely.forEach((item, i) => item.evidence.groups.forEach(group => {
  assert.notEqual(h.renderEvidenceSheet(f, `${i}:${group.key}`), '', `${i}:${group.key}`);
}));

console.log('preview match web: ok');
