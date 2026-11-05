// Pick notes (D75 data, D76 display): each Most likely and Value pick may carry note {for, against}. The display is
// layout B, chosen by the owner: a closed "For and against" under the pick, on the match page and the board card,
// opened with a tap. A card without notes (SPIX_PICK_NOTES off) renders exactly as before. No browser or network.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../web_app/static/app.js'), 'utf8');
const esc = value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const helpersStart = source.indexOf('// ----- Pure render helpers');
const helpersEnd = source.indexOf('// ----- End of pure render helpers -----');
assert.ok(helpersStart >= 0 && helpersEnd > helpersStart);
const h = Function('esc', source.slice(helpersStart, helpersEnd)
  + 'return { renderPickNote, renderRecommendations, renderRecsBoardCard, opensMatch, beforeKickoffFor, renderBeforeKickoffCard };')(esc);
const text = html => html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
const count = (haystack, needle) => haystack.split(needle).length - 1;

// The note on its own: closed, both lines tagged, escaped; nothing at all without words.
const note = { for: '4 of Ferencvarosi TC\'s last 5 Europa League matches had 3 or more goals.', against: 'It still loses about 4 times in 10.' };
const closed = h.renderPickNote(note);
assert.match(closed, /^<details class="pick-note">/);
assert.match(closed, /<summary>For and against<\/summary>/);
assert.match(text(closed), /For and against For 4 of Ferencvarosi TC's last 5 Europa League matches had 3 or more goals\. Against It still loses about 4 times in 10\./);
assert.doesNotMatch(closed, / open/);
assert.match(h.renderPickNote(note, { open: true }), /^<details class="pick-note" open>/);
assert.match(h.renderPickNote(note, { ref: 'ml:0' }), /^<details class="pick-note" data-pick-note="ml:0">/);
assert.doesNotMatch(h.renderPickNote({ for: '<script>x</script>', against: 'b' }), /<script>/);
for (const missing of [undefined, null, {}, 'For', [], { for: '', against: '  ' }, { for: 3, against: null }]) {
  assert.equal(h.renderPickNote(missing), '', `no note for ${JSON.stringify(missing)}`);
}
// One line only: the tap names what is there.
const forOnly = h.renderPickNote({ for: 'Won 4 of 5.' });
assert.match(forOnly, /<summary>For<\/summary>/);
assert.equal(count(forOnly, 'class="pick-note-line"'), 1);
assert.match(h.renderPickNote({ against: 'Lost 3 of 5.' }), /<summary>Against<\/summary>/);

// A block like the engine's, without notes (the switch off) and with them.
const option = (market, side, line, label, chance, price) => ({ market, side, line, label, chance, typical_price: price, long_shot: false });
const plain = {
  version: 'recommendations.v1',
  prices_as_of: '2026-11-05T15:02:11',
  most_likely: [
    option('spreads', 'home', -0.5, 'AC Milan −0.5', 0.73, 1.33),
    option('corners', 'over', 7.5, 'Over 7.5 corners', 0.7, 1.36),
    option('goals', 'over', 2.5, 'Over 2.5 goals', 0.6, 1.62),
  ],
  value: [{ position: 1, market: 'corners', side: 'over', line: 10, label: 'Over 10 corners', pick: 'Over 10', chance: 0.48, worth_it_at: 2.1 }],
  markets: [{ market: 'goals', suggestion: null, options: [option('goals', 'over', 2.5, 'Over 2.5 goals', 0.6, 1.62)] }],
  score_grid: null,
};
const notes = [
  { for: 'The likeliest option in its market that still pays at least 1.30.', against: 'Ferencvarosi TC avoided defeat in 5 of their last 5 Europa League matches.' },
  { for: '5 of Ferencvarosi TC\'s last 5 Europa League matches had 8 or more corners.', against: 'It still loses about 3 times in 10.' },
  note,
];
const noted = { ...plain,
  most_likely: plain.most_likely.map((item, i) => ({ ...item, note: notes[i] })),
  value: [{ ...plain.value[0], note: { for: 'Spix gives it 48%.', against: 'It still loses about 1 time in 2.' } }] };
const fixture = block => ({ id: '1', home: 'AC Milan', away: 'Ferencvarosi TC', time: '19:45', recommendations: block });

// Switch off: no note anywhere, and a null note is the same as none.
const off = h.renderRecommendations(fixture(plain));
assert.doesNotMatch(off, /pick-note/);
assert.equal(h.renderRecommendations(fixture({ ...plain, most_likely: plain.most_likely.map(item => ({ ...item, note: null })) })), off);
assert.doesNotMatch(h.renderRecsBoardCard(fixture(plain), '', ''), /pick-note/);

// Switch on, the match page: a closed note under each of the 3 Most likely picks and the Value pick.
const on = h.renderRecommendations(fixture(noted));
assert.equal(count(on, '<details class="pick-note"'), 4);
assert.doesNotMatch(on, /class="pick-note"[^>]* open/);
const firstCard = on.slice(on.indexOf('<article class="recs-card">'), on.indexOf('</article>'));
assert.ok(firstCard.indexOf('data-pick-note="ml:0"') < firstCard.indexOf('recs-card-foot'), 'the note sits above the price and Add');
assert.match(text(firstCard), /Against Ferencvarosi TC avoided defeat in 5 of their last 5 Europa League matches\./);
const valueCard = on.slice(on.indexOf('<article class="recs-value-card">'));
assert.ok(valueCard.indexOf('data-pick-note="value:0"') > 0
  && valueCard.indexOf('data-pick-note="value:0"') < valueCard.indexOf('recs-value-worth'), 'the Value note sits under the pick, before the price floor');
// A note opened before a redraw (Add, the stepper, a tab) stays open.
const kept = h.renderRecommendations(fixture(noted), { notes: { 'ml:1': true, 'value:0': false } });
assert.equal(count(kept, ' open>'), 1);
assert.match(kept, /data-pick-note="ml:1" open>/);

// Switch on, the board card: the first pick's note, closed, under its chance.
const board = h.renderRecsBoardCard(fixture(noted), '', '');
assert.equal(count(board, '<details class="pick-note"'), 1);
assert.ok(board.indexOf('recs-board-chance') < board.indexOf('pick-note') && board.indexOf('pick-note') < board.indexOf('recs-board-foot'));
assert.match(text(board), /For and against For The likeliest option in its market that still pays at least 1\.30\./);

// A tap on the note opens or closes it; anywhere else on the board card opens the match.
const target = inNote => ({ closest: selector => (selector === '.pick-note' && inNote ? {} : null) });
assert.equal(h.opensMatch({ target: target(true) }), false);
assert.equal(h.opensMatch({ target: target(false) }), true);
assert.equal(h.opensMatch(undefined), true);

// Started matches (the Live tab) keep only label and chance, so no note shows there.
const entry = { version: 'before-kickoff.v1', fixture: { event_id: '9', league: 'UEL', home_team: 'A', away_team: 'B', kickoff: '2026-11-05T17:45:00Z' },
  most_likely: [{ market: 'goals', side: 'over', line: 2.5, label: 'Over 2.5 goals', chance: 0.6, note }], value: [], status: { short: '1H', elapsed: 20 } };
const kept_ = h.beforeKickoffFor(entry, '', 'UEL');
assert.ok(kept_ && kept_.most_likely.length === 1);
assert.doesNotMatch(h.renderBeforeKickoffCard({ ...kept_, time: '18:45', home: 'A', away: 'B', liveShort: 'Live', beforeKickoff: kept_ }, '', ''), /pick-note/);

// The match page remembers which notes are open, by pick, across redraws.
const from = source.indexOf('  _recsNoteToggled(f, details) {');
const to = source.indexOf('\n  },', from) + 5;
assert.ok(from >= 0 && to > from);
const methods = Function(`return ({${source.slice(from, to)} _recsView(f) { return (this.views[f.id] = this.views[f.id] || { open: null, line: {} }); }, views: {} })`)();
methods._recsNoteToggled({ id: '1' }, { open: true, dataset: { pickNote: 'ml:2' } });
assert.deepEqual(methods._recsView({ id: '1' }).notes, { 'ml:2': true });
methods._recsNoteToggled({ id: '1' }, { open: false, dataset: { pickNote: 'ml:2' } });
assert.deepEqual(methods._recsView({ id: '1' }).notes, { 'ml:2': false });
methods._recsNoteToggled({ id: '1' }, { open: true, dataset: {} });
assert.deepEqual(methods._recsView({ id: '1' }).notes, { 'ml:2': false });

console.log('Pick note checks passed.');
