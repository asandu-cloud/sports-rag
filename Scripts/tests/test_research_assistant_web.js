// The Research Assistant tab (D93): how a question was read, in words; the "Which one?" pick sent back to /choose;
// the API's error messages; the answer cards drawn from real stats_answer.v1 answers; the box for visitors, members
// and a member at the limit. No browser or network.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../web_app/static/app.js'), 'utf8');
const esc = value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const helpersStart = source.indexOf('// ----- Pure render helpers');
const helpersEnd = source.indexOf('// ----- End of pure render helpers -----');
assert.ok(helpersStart >= 0 && helpersEnd > helpersStart);
const h = Function('esc', source.slice(helpersStart, helpersEnd)
  + 'return { assistantReadAs, assistantChoice, assistantError, renderAssistantCard, renderAssistantResult, renderAssistantAsk };')(esc);
const real = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures/research_assistant_answers.json'), 'utf8'));
const text = html => html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
const call = (type, params) => ({ type, params });

// How the question was read, in words, never the raw parameters.
assert.equal(h.assistantReadAs(real.line_record.calls[0]), 'Brentford · over 9.5 corners in the match · away');
assert.equal(h.assistantReadAs(real.team_average.calls[0]), 'Brentford · corners · away');
assert.equal(h.assistantReadAs(real.leaders.calls[0]), 'La Liga · teams · most corners');
assert.equal(h.assistantReadAs(call('head_to_head', { team: 'Arsenal', opponent: 'Chelsea' })), 'Arsenal v Chelsea · past meetings');
assert.equal(h.assistantReadAs(call('recent_form', { team: 'Tottenham', last: 10 })), 'Tottenham · recent form · last 10 matches');
assert.equal(h.assistantReadAs(call('player', { player: 'Bukayo Saka', stat: 'shots_on_target', at_least: 2, starts_only: true })),
  'Bukayo Saka · shots on target · 2+ a match · starts only');
assert.equal(h.assistantReadAs(call('referee', { referee: 'Anthony Taylor', stat: 'cards', team_filter: 'Arsenal', line: 4.5 })),
  'Anthony Taylor · cards · with Arsenal · over 4.5');
assert.equal(h.assistantReadAs(call('team_average', { team: 'Brentford', stat: 'xg', side: 'against', venue: 'home', competition: 'EPL', season: 2025 })),
  'Brentford · expected goals against · at home · Premier League · 2025/26');
assert.equal(h.assistantReadAs(call('line_record', { team: 'Inter', stat: 'cards', side: 'for', line: 2.5, direction: 'under', competition: 'all' })),
  'Inter · under 2.5 cards · all competitions');
assert.equal(h.assistantReadAs(call('leaders', { kind: 'player', stat: 'yellow_cards', competition: 'SerieA', order: 'fewest' })),
  'Serie A · players · fewest yellow cards');
// A name picked from a "Which one?" card is an "#id": the words use the label picked for that id, never the id.
assert.equal(h.assistantReadAs(call('player', { player: '#276', stat: 'goals' }), { '#276': 'Bernardo Silva (Real Madrid)' }),
  'Bernardo Silva (Real Madrid) · goals');
assert.equal(h.assistantReadAs(call('head_to_head', { team: '#42', opponent: 'Chelsea' }), { '#42': 'Arsenal' }), 'Arsenal v Chelsea · past meetings');
assert.equal(h.assistantReadAs(call('player', { player: '#276', stat: 'goals' })), 'goals');
assert.equal(h.assistantReadAs(call('team_average', { team: 'Brentford', stat: 'big_chances' })), 'Brentford · big chances');
assert.equal(h.assistantReadAs(call('mystery', { team: 'Brentford' })), '');
assert.equal(h.assistantReadAs(null), '');

// A pick sends the card's own call back, unchanged except the picked field.
const which = real.which;
const option = which.answers[0].options[1];
const pick = h.assistantChoice(which.calls[0], option);
assert.equal(pick.type, 'player');
assert.equal(pick.params.player, option.value);
assert.match(option.value, /^#\d+$/);
assert.equal(pick.params.stat, 'goals');
assert.equal(which.calls[0].params.player, 'Silva', 'the shown call is left as it was');
assert.deepEqual(Object.keys(pick.params).sort(), Object.keys(which.calls[0].params).sort());

// The API's message is shown as it comes; anything else gets one plain line.
assert.deepEqual(h.assistantError({ detail: { kind: 'member_limit', message: 'You have asked 1 question today, the most for your account. You can ask again tomorrow.' } }, 429),
  { kind: 'member_limit', message: 'You have asked 1 question today, the most for your account. You can ask again tomorrow.' });
assert.deepEqual(h.assistantError({ detail: [{ msg: 'field required' }] }, 422), { kind: 'error', message: 'Spix could not read that question. Please try again.' });
assert.deepEqual(h.assistantError(null, 500), { kind: 'error', message: 'Spix could not read that question. Please try again.' });
assert.deepEqual(h.assistantError(null, 0), { kind: 'offline', message: 'Spix could not be reached. Check your connection and try again.' });

// An answer card: what was counted, the headline, the details, the matches on request and the data date.
const line = h.renderAssistantCard(real.line_record.answers[0]);
assert.match(text(line), /Brentford away · Premier League 2026\/27 · 5 matches/);
assert.match(text(line), /Over 9\.5 corners in 3 of 5 matches\./);
assert.match(text(line), /Premier League: over 9\.5 in 47 of 100 matches\./);
assert.match(line, /<details class="ra-more"><summary>/);
assert.match(text(line), /The 5 matches behind this/);
assert.match(text(line), /Data up to Sun 8 Nov 2026/);
assert.equal((line.match(/<tr>/g) || []).length, 6, 'a header and the five matches');
assert.match(h.renderAssistantCard(real.line_record.answers[0], { open: true }), /<details class="ra-more" open>/);
const leaders = h.renderAssistantCard(real.leaders.answers[0]);
assert.match(text(leaders), /Villarreal 7\.4/);
assert.match(leaders, /<p class="ra-note">Teams with at least 3 matches\.<\/p>/);
assert.match(h.renderAssistantCard({ ...real.team_average.answers[0], note: 'Small sample: only 3 matches.' }), /<p class="ra-note warn">Small sample/);
// Numbers sit right, words left.
assert.match(leaders, /<th class="n">Corners won a match<\/th>/);
assert.match(leaders, /<th>Name<\/th>/);
// "Which one?": the names as buttons, carrying which option they are; no data date on a question back.
const whichCard = h.renderAssistantCard(which.answers[0], { entry: 3, card: 0 });
assert.equal((whichCard.match(/data-ra-pick=/g) || []).length, 6);
assert.match(whichCard, /data-ra-pick="3:0:1"[^>]*>Bernardo Silva \(Real Madrid\)</);
assert.doesNotMatch(text(whichCard), /Data up to/);
// A betting question points to the matches page.
const declined = h.renderAssistantCard(real.declined.answers[0]);
assert.match(text(declined), /Spix's picks are on the matches page/);
assert.match(declined, /data-ra-fixtures/);
assert.doesNotMatch(text(declined), /Data up to/);
// Whatever the card says is escaped.
assert.doesNotMatch(h.renderAssistantCard({ type: 'invalid', headline: '<img src=x onerror=alert(1)>', detail: [] }), /<img/);

// A result: the question, how it was read (both calls for two answers), then its cards.
const two = h.renderAssistantResult(real.two_answers, 0);
assert.match(text(two), /^Corners for and against Brentford at home and away Read as: Brentford · corners · at home; Brentford · corners against · at home/);
assert.equal((two.match(/class="ra-card/g) || []).length, 2);
assert.doesNotMatch(h.renderAssistantResult(real.declined, 0), /Read as/);

// The box: visitors get the sign-in prompt; members see what is left today; at the limit the box is off.
const visitor = h.renderAssistantAsk({ enabled: true, signed_in: false });
assert.match(text(visitor), /Sign in to ask the Research Assistant\. Free accounts can ask one question a day\. Log in/);
assert.doesNotMatch(visitor, /<input/);
const free = h.renderAssistantAsk({ enabled: true, signed_in: true, plan: 'free', daily_limit: 1, asked_today: 0, remaining: 1 });
assert.match(free, /<input[^>]*maxlength="300"/);
assert.match(text(free), /1 of 1 question left today/);
assert.match(text(h.renderAssistantAsk({ enabled: true, signed_in: true, plan: 'paid', daily_limit: 50, asked_today: 7, remaining: 43 })), /43 of 50 questions left today/);
const done = h.renderAssistantAsk({ enabled: true, signed_in: true, plan: 'free', daily_limit: 1, asked_today: 1, remaining: 0 });
assert.match(done, /<input[^>]*disabled/);
assert.match(text(done), /You have asked 1 question today, the most for your account\. You can ask again tomorrow\./);
assert.match(text(h.renderAssistantAsk({ enabled: true, signed_in: true, plan: 'paid', daily_limit: 50, asked_today: 50, remaining: 0 })),
  /You have asked 50 questions today/);
// The API's own message wins when it has sent one (for example the whole site's limit).
assert.match(text(h.renderAssistantAsk({ enabled: true, signed_in: true, plan: 'paid', daily_limit: 50, asked_today: 3, remaining: 47 },
  { kind: 'site_limit', message: 'The Research Assistant has answered all it can today. Please try again tomorrow.' })),
  /The Research Assistant has answered all it can today/);
assert.match(text(h.renderAssistantAsk({ enabled: false })), /The Research Assistant is not available right now\./);

console.log('research assistant web: all checks passed');
