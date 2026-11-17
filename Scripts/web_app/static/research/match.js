// Research Area overhaul (D107, task 6): one match's research view, from B2's research-match.v1. The head, How Spix
// reads it, the markets row, the open market's section (SECTIONS, filled by tasks 7 to 9) and the foot line. Pure
// functions returning HTML; research.js places them and wires the clicks.
import {escape, one, LINES, DEFAULT_LINE, passes, hitCount, strip, shortDay} from './insight.js?v=research-20261117h';
import {crest, whenText, readState, opensDay, possessive} from './upcoming.js?v=research-20261117h';

export const MARKETS = ['cards', 'sot', 'goals', 'corners', 'result', 'h2h', 'news'];
export const MARKET_LABELS = {cards: 'Cards', sot: 'Shots on target', goals: 'Goals', corners: 'Corners', result: 'Result',
  h2h: 'Head to head', news: 'Team news'};
// market -> (match, view) => html. Tasks 7 to 9 add theirs; a market's button shows once its section exists.
export const SECTIONS = {};

const GAMCARE = 'https://www.gamcare.org.uk/get-support/';
const pct = chance => `${Math.round(chance * 100)}%`;
const back = label => `<button type="button" class="text-button back-button" data-match-back><svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><path d="m14 6-6 6 6 6"/></svg><span>${escape(label)}</span></button>`;

// A match that has kicked off, from the fixture's status (API-Football's short codes, as the board's Live tab reads
// them). The API drops the read once the board does, so nothing may promise one; a read still carried is from before
// kickoff and takes no Add. Any other status, or none, is a match still to come.
const KICKED = "This match has kicked off, so Spix's read is no longer shown. The past numbers below run up to kickoff.";
const BEFORE_ONLY = "Spix's chances and typical prices are shown before kickoff only.";
const PHASES = {
  live: {chip: 'In play', lead: KICKED, note: BEFORE_ONLY},
  played: {chip: 'Full time', lead: "This match has been played, so Spix's read is no longer shown. The past numbers below run up to its kickoff.", note: BEFORE_ONLY},
  PST: {chip: 'Postponed', lead: 'This match was postponed. Spix reads it again once it has a new date.', note: "Spix's chances and typical prices show once the match has a new date."},
  CANC: {chip: 'Cancelled', lead: 'This match was cancelled, so Spix has no read for it.', note: 'No chances or typical prices for a cancelled match.'},
  ABD: {chip: 'Abandoned', lead: KICKED, note: BEFORE_ONLY},
};
const IN_PLAY = ['1H', 'HT', '2H', 'ET', 'BT', 'P', 'LIVE', 'INT', 'SUSP'], FULL_TIME = ['FT', 'AET', 'PEN', 'AWD', 'WO'];
export function matchPhase(fixture) {
  const code = String(fixture?.status || '').toUpperCase();
  return IN_PLAY.includes(code) ? PHASES.live : FULL_TIME.includes(code) ? PHASES.played : PHASES[code] || null;
}

export function matchHead(match, {now = new Date()} = {}) {
  const f = match.fixture, read = f.read || {}, phase = matchPhase(f);
  const referee = f.referee?.name
    ? `<span>Referee <button type="button" class="link" data-referee="${escape(f.referee.referee_key)}">${escape(f.referee.name)}</button></span>`
    : '<span>Referee not named yet</span>';
  const tone = phase ? ' wait' : read.status === 'ready' ? '' : read.status === 'preliminary' ? ' pre' : ' wait';
  return `<div class="mv-head">${crest(f.home, 'md')}<h2 class="mv-title" id="matchTitle" tabindex="-1">${escape(f.home.name)} <span class="vw">v</span> ${escape(f.away.name)}</h2>${crest(f.away, 'md')}</div>
    <p class="mv-meta"><span>${escape(whenText(f.kickoff))}</span><span>${escape(f.competition?.name || '')}</span>${referee}<span class="status${tone}">${escape(phase ? phase.chip : readState(read, now))}</span></p>`;
}

// How Spix reads it: the read's own figures, one marked for the open market. Before the read is out, when it comes.
export function spixReads(match, market) {
  const f = match.fixture, r = match.read, phase = matchPhase(f);
  const open = `<section class="reads" aria-labelledby="readsTitle"><p class="cap" id="readsTitle">${r && phase ? 'How Spix read it before kickoff' : 'How Spix reads it'}</p>`;
  if (!r && phase) return `${open}<p class="rlead">${escape(phase.lead)}</p></section>`;
  if (!r) {
    const when = f.read?.opens_at ? opensDay(f.read.opens_at) : 'nearer the match';
    return `${open}<p class="rlead">Spix's read comes out ${escape(when)}. The past numbers below are ready now.</p></section>`;
  }
  const home = f.home.name, away = f.away.name, res = r.result;
  const marked = (key, html) => key === market ? `<mark>${html}</mark>` : html;
  let lead = '', balance = '';
  if (res) {
    const homeLeads = res.home >= res.away, gap = Math.abs(res.home - res.away);
    lead = gap >= 0.1 ? `${escape(homeLeads ? home : away)} are favourites.` : 'No clear favourite.';
    const fav = gap >= 0.1 ? (homeLeads ? 'home' : 'away') : null;
    const side = (key, label) => `<span>${escape(label)} <b>${fav === key ? marked('result', pct(res[key])) : pct(res[key])}</b></span>`;
    balance = `<div class="bal" aria-hidden="true"><i style="flex:${Number(res.home) || 0};background:var(--chart-1)"></i><i style="flex:${Number(res.draw) || 0};background:var(--border2)"></i><i style="flex:${Number(res.away) || 0};background:var(--chart-2)"></i></div>
      <p class="bal-leg">${side('home', home)}${side('draw', 'Draw')}${side('away', away)}</p>`;
  }
  const figure = (key, value, label, note = '') => value == null ? ''
    : `<div><b>${marked(key, escape(value))}</b>${label}${note ? `<small>${note}</small>` : ''}</div>`;
  const goals = r.goals;
  const figs = [
    figure('goals', goals?.total == null ? null : one(goals.total), 'goals', goals && goals.home != null ? `${escape(home)} ${one(goals.home)} · ${escape(away)} ${one(goals.away)}` : ''),
    figure('corners', r.corners_expected == null ? null : one(r.corners_expected), 'corners'),
    figure('cards', r.cards_expected == null ? null : one(r.cards_expected), 'cards'),
    figure('btts', r.btts_yes == null ? null : pct(r.btts_yes), 'both teams score', "Spix's chance of yes"),
  ].join('');
  const early = r.status === 'preliminary' ? '<p class="pre-note">Preliminary: an early read. Spix refreshes it nearer kickoff.</p>' : '';
  return `${open}${lead ? `<p class="rlead">${lead}</p>` : ''}${balance}${figs ? `<div class="figs">${figs}</div>` : ''}${early}</section>`;
}

export function marketsRow(active, available = MARKETS.filter(key => SECTIONS[key])) {
  const list = MARKETS.filter(key => available.includes(key));
  if (!list.length) return '';
  return `<nav class="mkts" aria-label="Markets">${list.map(key => `<button type="button" data-market="${key}" aria-current="${key === active}">${MARKET_LABELS[key]}</button>`).join('')}</nav>`;
}

export function renderMatch(match, view) {
  const section = SECTIONS[view.market]?.(match, view) || '';
  return `${back(view.back?.label || 'Back')}${matchHead(match, view)}${spixReads(match, view.market)}${marketsRow(view.market)}
    ${section ? `<section class="mk" id="mk-${view.market}" aria-label="${MARKET_LABELS[view.market]}">${section}</section>` : ''}
    <p class="disclaimer">Past matches are records, not a forecast. Spix's figures come from statistical models. 18+ · <a href="${GAMCARE}" target="_blank" rel="noopener">GamCare</a></p>`;
}

export function matchError(status) {
  const [title, body] = status === 503 ? ['Research is updating', 'Try again in a few minutes.']
    : status === 404 ? ['This match could not load', 'It may have been moved or cancelled.']
    : ['This match could not load', 'Please try again in a moment.'];
  return `${back('Back')}<div class="search-empty"><h3>${title}</h3><p>${body}</p></div>`;
}

// ----- Task 7: Cards, and the shared parts the other markets reuse -----
const mean = values => { const known = values.filter(v => v != null); return known.length ? known.reduce((sum, v) => sum + v, 0) / known.length : null; };
const both = (a, b) => a == null || b == null ? null : a + b;
const plural = (n, single, many) => `${n} ${n === 1 ? single : many}`;
const seasonLabel = year => `${year}/${String(year + 1).slice(2)}`;
const initials = name => name.split(/[\s.-]+/).filter(Boolean).map(w => w[0]).join('').slice(0, 2).toUpperCase();
const listWords = items => items.length < 2 ? items.join('') : `${items.slice(0, -1).join(', ')} and ${items.at(-1)}`;
// Cards say where the matches come from; goals and corners what each side made of them.
const SPLIT = {goals: ['scored', 'conceded'], corners: ['won', 'conceded']};
function teamSub(rows, market) {
  if (!SPLIT[market]) return `Last ${rows.length}, every competition`;
  const [mine, theirs] = TEAM_TOTAL[market], [won, gave] = SPLIT[market];
  return `Last ${rows.length} · ${won} ${one(mean(rows.map(m => m[mine])))}, ${gave} ${one(mean(rows.map(m => m[theirs])))}`;
}
const TEAM_TOTAL = {cards: ['cards_for', 'cards_against'], goals: ['goals_for', 'goals_against'], corners: ['corners_for', 'corners_against']};

const segment = (attr, label, options, current) => `<div class="segmented" role="group" aria-label="${label}">${options.map(([v, text]) => `<button type="button" data-${attr}="${v}" aria-pressed="${v === current}">${escape(text)}</button>`).join('')}</div>`;
function lineRow(market, line) {
  return `<div class="lines"><span>Line</span>${segment('mline', `${MARKET_LABELS[market]} line`, LINES[market].map(v => [v, v]), line)}</div>`;
}

// One option: its label, Spix's chance and the typical price; the side over 50% is tagged. Cards carry no Spix
// suggestion (D61), so theirs reads "More likely" (owner, 16 Nov). Add only when the read has the option in a market
// the slip takes (addLegDetail finds it).
const LEAN = {cards: 'More likely'};
function pickCard(match, {label, chance, price, market, side, line}) {
  const lean = chance > 0.5, addable = market && !matchPhase(match.fixture) && addLegDetail(match, market, side, line);
  const button = addable ? `<button type="button" class="mk-add" data-add="${market}|${side}|${line}" aria-label="Add ${escape(label)} to the slip">Add</button>` : '';
  const parts = [chance == null ? '' : `Spix <b>${pct(chance)}</b>`, price == null ? '' : `typical price <b>${Number(price).toFixed(2)}</b>`].filter(Boolean);
  return `<div class="mk-pick${lean ? ' lean' : ''}"><span class="mk-pl">${escape(label)}</span>${lean ? ` <span class="mk-lean">${LEAN[market] || 'Spix leans this way'}</span>` : ''}<span class="mk-pc">${parts.join(' · ')}</span>${button}</div>`;
}
const waiting = match => `<p class="mk-note">${escape(matchPhase(match.fixture)?.note || "Spix's chances and typical prices show once the read is out.")}</p>`;

// Over and Under at the line, from the read's lines.
function pickPair(match, market, line, unit) {
  const r = match.read;
  if (!r) return waiting(match);
  const lines = r.lines?.[market] || [], at = lines.find(l => l.line === line);
  if (!at) return `<p class="mk-note">No typical price for ${line} ${unit} yet.${lines.length ? ` Spix's chances cover ${listWords(lines.map(l => l.line))}.` : ''}</p>`;
  const card = side => at[side] ? pickCard(match, {label: `${side === 'over' ? 'Over' : 'Under'} ${line} ${unit}`, chance: at[side].chance,
    price: at[side].typical_price, market, side, line}) : '';
  return `<div class="mk-pair">${card('over')}${card('under')}</div>`;
}

function evRow(icon, title, sub, values, labels, line) {
  const {hits, of} = hitCount(values, line);
  return `<div class="mk-row">${icon}<span class="mk-who"><strong>${escape(title)}</strong><small>${escape(sub)}</small></span>${strip(values, v => passes(v, line), labels)}<span class="mk-hit">${hits} of ${of}</span><span class="mk-avg"><b>${one(mean(values))}</b> a match</span></div>`;
}

// How often each team's last 10, the referee's last 10 (cards only) and the meetings went over the line.
function historyRows(match, market, line, unit) {
  const rows = [];
  for (const side of ['home', 'away']) {
    const t = match.teams?.[side], [mine, theirs] = TEAM_TOTAL[market];
    if (!t?.last_matches?.length) continue;
    rows.push(evRow(crest(t.team), `${possessive(t.team.name)} matches`, teamSub(t.last_matches, market),
      t.last_matches.map(m => both(m[mine], m[theirs])),
      t.last_matches.map(m => `${shortDay(m.date)} ${m.venue === 'home' ? 'v' : 'at'} ${m.opponent?.name || ''}`), line));
  }
  const ref = match.referee;
  if (market === 'cards' && ref?.last_matches?.length) {
    rows.push(evRow(`<span class="mk-ini" aria-hidden="true">${escape(initials(ref.name))}</span>`, ref.name,
      `Referee, last ${plural(ref.last_matches.length, 'match', 'matches')}`, ref.last_matches.map(m => m.cards),
      ref.last_matches.map(m => `${shortDay(m.date)} ${m.home} v ${m.away}`), line));
  }
  const f = match.fixture, since = seasonLabel(match.data_basis?.h2h_since_season || 2023);
  const meetings = [...(match.h2h || [])].reverse();
  const h2h = meetings.length
    ? evRow('<span class="mk-ini" aria-hidden="true">H2H</span>', `${f.home.name} v ${f.away.name}`, `Meetings since ${since}`,
      meetings.map(m => m[market]), meetings.map(m => `${shortDay(m.date)} ${m.home.name} ${m.score?.home}–${m.score?.away} ${m.away.name}`), line)
    : '';
  const notes = [market === 'cards' && !ref ? '<p class="mk-note">Referee not named yet. Appointments usually come two to three days before kickoff.</p>' : '',
    meetings.length ? '' : `<p class="mk-note">No meetings since ${since}.</p>`].join('');
  return `<div class="mk-ev"><h3>How often their matches went over ${line} ${unit}<small>newest on the right</small></h3><div class="mk-rows">${rows.join('')}${h2h}</div>${notes}</div>`;
}

function whoGetsCards(match) {
  const sides = ['home', 'away'].map(side => match.teams?.[side]).filter(t => t?.last_matches?.length);
  if (!sides.length) return '';
  const figures = sides.map(t => ({team: t.team, mine: mean(t.last_matches.map(m => m.cards_for)), theirs: mean(t.last_matches.map(m => m.cards_against))}));
  const top = figures.flatMap(x => [{...x, kind: 'mine', v: x.mine}, {...x, kind: 'theirs', v: x.theirs}]).filter(x => x.v != null)
    .reduce((best, x) => !best || Math.round(x.v * 10) > Math.round(best.v * 10) ? x : best, null);
  const callout = !top ? '' : top.kind === 'mine'
    ? `<p class="mk-callout">${escape(top.team.name)} pick up the most: ${one(top.v)} cards a match.</p>`
    : `<p class="mk-callout">Teams that play ${escape(top.team.name)} pick up the most: ${one(top.v)} cards a match.</p>`;
  return `<div class="mk-ev"><h3>Who gets the cards<small>Last 10 each</small></h3><div class="mk-two">${figures.map(x => `<div class="mk-mini"><div class="mk-mh">${crest(x.team)}${escape(x.team.name)}</div><dl><div><dt>Booked a match</dt><dd>${one(x.mine)}</dd></div><div><dt>Their opponents booked</dt><dd>${one(x.theirs)}</dd></div></dl></div>`).join('')}</div>${callout}</div>`;
}

function refereeBlock(match) {
  const ref = match.referee;
  if (!ref) return '';
  const small = ref.matches != null && ref.matches < 15 ? `<li>${plural(ref.matches, 'match is', 'matches is')} a small sample; one heated game moves the average.</li>` : '';
  return `<div class="mk-ev"><h3>The referee</h3><div class="mk-two"><div class="mk-mini"><div class="mk-mh"><button type="button" class="link" data-referee="${escape(ref.referee_key)}">${escape(ref.name)}</button></div><dl>
    <div><dt>Cards a match this season</dt><dd>${one(ref.cards_per_match)}<small>${plural(ref.matches ?? 0, 'match', 'matches')}, every competition</small></dd></div>
    <div><dt>${escape(ref.league?.name || 'The league')} this season</dt><dd>${one(ref.league_cards_per_match)}<small>${plural(ref.league_matches ?? 0, 'match', 'matches')}</small></dd></div></dl></div>
    <div class="mk-mini"><div class="mk-mh">Keep in mind</div><ul class="mk-notes">${small}<li>Each yellow and each red counts as one card here. Check how your bookmaker counts a red.</li></ul></div></div></div>`;
}

function bookedBlock(match) {
  const sides = ['home', 'away'].map(side => match.teams?.[side]).filter(Boolean);
  const player = b => `<li class="mk-booked-row"><span>${escape(b.name)}</span><span>${[b.yellow ? `${b.yellow} yellow` : '', b.red ? `${b.red} red` : ''].filter(Boolean).join(' · ')} in ${plural(b.appearances, 'match', 'matches')}</span></li>`;
  return `<div class="mk-ev"><h3>Players booked most often<small>This season, every competition</small></h3><div class="mk-two">${sides.map(t => `<div class="mk-mini"><div class="mk-mh">${crest(t.team)}${escape(t.team.name)}</div>${t.booked?.length ? `<ul class="mk-booked">${t.booked.slice(0, 5).map(player).join('')}</ul>` : '<p class="mk-note">No one booked this season yet.</p>'}</div>`).join('')}</div></div>`;
}

const EXPECTED = {cards: r => r.cards_expected, goals: r => r.goals?.total, corners: r => r.corners_expected};
// A totals market's head: its name, what Spix expects, the line picker, Over and Under, and the history at that line.
function totalsTop(match, market, line) {
  const unit = market, value = match.read ? EXPECTED[market](match.read) : null;
  const expects = value != null ? `<p>Spix expects <b>${one(value)}</b> ${unit}</p>` : '';
  return `<div class="mk-h"><h2 class="cond">${MARKET_LABELS[market]}</h2>${expects}</div>${lineRow(market, line)}${pickPair(match, market, line, unit)}
    ${historyRows(match, market, line, unit)}`;
}
const lineOf = (market, view) => LINES[market].includes(view.line) ? view.line : DEFAULT_LINE[market];

export function cardsSection(match, view = {}) {
  return `${totalsTop(match, 'cards', lineOf('cards', view))}${whoGetsCards(match)}${refereeBlock(match)}${bookedBlock(match)}`;
}
SECTIONS.cards = cardsSection;

// The read's own option for Add, in the shape app.js hands to the slip (the match page's Explore options).
export function addLegDetail(match, market, side, line) {
  const options = match.read?.markets?.find(m => m.market === market)?.options || [];
  const leg = options.find(o => o.side === side && (o.line ?? null) === (line ?? null));
  if (!leg) return null;
  const f = match.fixture;
  return {fixture: {event_id: String(f.event_id), home: f.home.name, away: f.away.name, kickoff: f.kickoff, league: f.competition?.code}, leg, market};
}

// ----- Task 8: Shots on target -----
const POSITIONS = {G: 'Goalkeeper', D: 'Defender', M: 'Midfielder', F: 'Forward'};
const ordinal = n => n + (n % 100 >= 11 && n % 100 <= 13 ? 'th' : ['th', 'st', 'nd', 'rd'][n % 10] || 'th');
const SOT_LINES = [1, 2];
// "in the Premier League", but "in La Liga", "in Serie A", "in Ligue 1".
const inLeague = name => !name ? 'in the league' : /^(La|Serie|Ligue|Liga) /.test(name) ? `in ${name}` : `in the ${name}`;

// 1 = fewest. The top half reads from the fewest, the bottom half from the most.
export function rankWords(rank, peers) {
  if (rank === 1) return `the fewest of ${peers}`;
  if (rank === peers) return `the most of ${peers}`;
  return rank <= peers / 2 ? `${ordinal(rank)} fewest of ${peers}` : `${ordinal(peers - rank + 1)} most of ${peers}`;
}

function allowMini(team, allowed) {
  const body = allowed
    ? `<dl><div><dt>Shots on target faced a match</dt><dd>${one(allowed.per_match)}<small>${rankWords(allowed.rank, allowed.peer_count)} ${escape(inLeague(allowed.competition?.name))}</small></dd></div></dl>`
    : `<p class="mk-note">No league figures for ${escape(team.name)} this season yet.</p>`;
  return `<div class="mk-mini"><div class="mk-mh">${crest(team)}What ${escape(team.name)} allow</div>${body}</div>`;
}

export function sotSection(match, view = {}) {
  const line = SOT_LINES.includes(view.line) ? view.line : SOT_LINES[0];
  const team = ['home', 'away'].includes(view.team) ? view.team : 'both';
  const t = match.teams || {}, names = {home: t.home?.team?.name, away: t.away?.team?.name};
  const sides = team === 'both' ? ['home', 'away'] : [team];
  const players = sides.flatMap(side => (t[side]?.shooters || []).map(p => {
    const values = (p.last_10 || []).map(a => a.sot), minutes = mean((p.last_10 || []).map(a => a.minutes));
    return {p, side, values, minutes, ...hitCount(values, line)};
  })).sort((a, b) => b.hits - a.hits);
  const row = x => `<tr class="mk-sot-row"><td><span class="mk-player">${crest(t[x.side].team)}<span><strong>${escape(x.p.name)}</strong><small>${escape([names[x.side], POSITIONS[x.p.position]].filter(Boolean).join(' · '))}</small></span></span></td>
    <td>${strip(x.values, v => passes(v, line), x.p.last_10.map(a => `${shortDay(a.date)} v ${a.opponent}${a.started ? '' : ', from the bench'}`))} <b class="mk-of">${x.hits} of ${x.of}</b></td>
    <td class="r mk-min">${x.minutes == null ? '' : `${Math.round(x.minutes)} min`}</td></tr>`;
  const table = players.length
    ? `<table class="mk-tbl"><thead><tr><th scope="col">Player</th><th scope="col">Last 10 appearances</th><th scope="col" class="r mk-min">Minutes a match</th></tr></thead><tbody>${players.map(row).join('')}</tbody></table>`
    : '<p class="mk-note">No player figures for these teams yet.</p>';
  // Each side's shooters face the other side's defence.
  const facing = sides.map(side => side === 'home' ? 'away' : 'home').filter(side => t[side]?.team);
  const teams = [['both', 'Both'], ['home', names.home || 'Home'], ['away', names.away || 'Away']];
  return `<div class="mk-h"><h2 class="cond">Shots on target</h2><p>Likely starters, by starts in their last 5</p></div>
    <div class="lines"><span class="mk-ctl"><span>Each player</span>${segment('mline', 'Shots on target line', SOT_LINES.map(v => [v, `${v}+`]), line)}</span><span class="mk-ctl"><span>Team</span>${segment('mteam', 'Team', teams, team)}</span></div>
    ${table}<div class="mk-two">${facing.map(side => allowMini(t[side].team, t[side].allowed)).join('')}</div>
    <p class="mk-note">Line-ups are confirmed about an hour before kickoff; this list comes from recent starts.</p>`;
}
SECTIONS.sot = sotSection;

// The player page's context: the club's next opponent and the shots on target it allows.
export function opponentAllows(match, teamId) {
  const t = match.teams || {}, mine = ['home', 'away'].find(side => t[side]?.team?.id === teamId);
  if (!mine) return '';
  const other = t[mine === 'home' ? 'away' : 'home'], a = other?.allowed;
  if (!a) return '';
  const rank = rankWords(a.rank, a.peer_count);
  return `${other.team.name} allow ${one(a.per_match)} shots on target a match, ${rank.startsWith('the ') ? rank : `the ${rank}`} ${inLeague(a.competition?.name)}.`;
}

// ----- Task 9: Goals, Corners, Result, Head to head, Team news -----
export const goalsSection = (match, view = {}) => totalsTop(match, 'goals', lineOf('goals', view));
export const cornersSection = (match, view = {}) => totalsTop(match, 'corners', lineOf('corners', view));
SECTIONS.goals = goalsSection;
SECTIONS.corners = cornersSection;

const outcome = m => m.goals_for == null || m.goals_against == null ? null
  : m.goals_for > m.goals_against ? 'W' : m.goals_for === m.goals_against ? 'D' : 'L';
const record = rows => { const r = rows.map(outcome); return {W: r.filter(x => x === 'W').length, D: r.filter(x => x === 'D').length, L: r.filter(x => x === 'L').length}; };
const fullDay = iso => new Date(iso).toLocaleDateString('en-GB', {day: 'numeric', month: 'short', year: 'numeric', timeZone: 'Europe/London'});

function formRow(t) {
  const rows = t.last_matches || [], n = record(rows);
  const squares = rows.map(m => { const o = outcome(m);
    return `<i class="${o ? o.toLowerCase() : 'na'}" title="${escape(`${shortDay(m.date)} ${m.venue === 'home' ? 'v' : 'at'} ${m.opponent?.name || ''}: ${m.goals_for ?? '?'}–${m.goals_against ?? '?'}`)}">${o || ''}</i>`; }).join('');
  return `<div class="mk-row mk-form">${crest(t.team)}<span class="mk-who"><strong>${escape(t.team.name)}</strong><small>Last ${rows.length}, every competition</small></span><span class="mk-wdl" role="img" aria-label="${n.W} won, ${n.D} drawn, ${n.L} lost">${squares}</span><span class="mk-hit">${n.W} won, ${n.D} drawn, ${n.L} lost</span></div>`;
}

export function resultSection(match, view = {}) {
  const f = match.fixture, t = match.teams || {};
  const options = match.read?.markets?.find(m => m.market === 'moneyline')?.options || [];
  const order = ['home', 'draw', 'away'];
  const picks = match.read && options.length
    ? `<div class="mk-pair mk-three">${order.map(side => options.find(o => o.side === side)).filter(Boolean).map(o => pickCard(match,
      {label: o.label, chance: o.chance, price: o.typical_price, market: 'moneyline', side: o.side, line: null})).join('')}</div>`
    : waiting(match);
  const venue = (side, where) => {
    const team = t[side]; if (!team) return '';
    const rows = (team.last_matches || []).filter(m => m.venue === where), n = record(rows);
    return `<div class="mk-mini"><div class="mk-mh">${crest(team.team)}${escape(team.team.name)} ${where === 'home' ? 'at home' : 'away'}</div><dl><div><dt>Won, drew, lost</dt><dd>${n.W} · ${n.D} · ${n.L}<small>${plural(rows.length, `${where} match`, `${where} matches`)} in their last ${team.last_matches.length}</small></dd></div></dl></div>`;
  };
  return `<div class="mk-h"><h2 class="cond">Result</h2></div>${picks}
    <div class="mk-ev"><h3>Form<small>newest on the right</small></h3><div class="mk-rows">${['home', 'away'].filter(side => t[side]?.last_matches?.length).map(side => formRow(t[side])).join('')}</div></div>
    <div class="mk-two">${venue('home', 'home')}${venue('away', 'away')}</div>`;
}
SECTIONS.result = resultSection;

export function h2hSection(match, view = {}) {
  const f = match.fixture, list = match.h2h || [], since = seasonLabel(match.data_basis?.h2h_since_season || 2023);
  const head = `<div class="mk-h"><h2 class="cond">Head to head</h2><p>Since ${since}</p></div>`;
  if (!list.length) return `${head}<p class="mk-note">No meetings since ${since}.</p>`;
  const wins = {[f.home.id]: 0, [f.away.id]: 0};
  let draws = 0;
  for (const m of list) {
    const {home, away} = m.score || {};
    if (home == null || away == null) continue;
    if (home === away) draws += 1; else wins[home > away ? m.home.id : m.away.id] = (wins[home > away ? m.home.id : m.away.id] || 0) + 1;
  }
  const sides = [f.home, f.away].sort((a, b) => wins[b.id] - wins[a.id]);
  const summary = `${sides.map(team => `${escape(team.name)} won ${wins[team.id] || 0}`).join(', ')}${draws ? `, ${plural(draws, 'draw', 'draws')}` : ''}.`;
  const facts = m => [[m.cards, 'card', 'cards'], [m.corners, 'corner', 'corners'], [m.goals, 'goal', 'goals']].filter(([v]) => v != null).map(([v, a, b]) => plural(v, a, b)).join(' · ');
  return `${head}<p class="mk-callout">${summary}</p><ul class="mk-h2h">${list.map(m => `<li class="mk-h2h-row"><span class="mk-h2h-when">${escape(fullDay(m.date))}<small>${escape(m.competition?.name || '')}</small></span><span class="mk-h2h-score">${escape(m.home.name)} ${m.score?.home ?? '?'}–${m.score?.away ?? '?'} ${escape(m.away.name)}</span><span class="mk-h2h-facts">${facts(m)}</span></li>`).join('')}</ul>`;
}
SECTIONS.h2h = h2hSection;

// No line-ups in B2 yet (lineups is always null), so the likely starters from recent starts.
export function newsSection(match, view = {}) {
  const t = match.teams || {};
  const side = key => { const team = t[key]; if (!team) return '';
    const list = team.shooters || [];
    return `<div class="mk-mini"><div class="mk-mh">${crest(team.team)}${escape(team.team.name)}</div>${list.length
      ? `<ul class="mk-booked">${list.map(p => `<li class="mk-news-row"><span>${escape(p.name)}</span><span>${escape([POSITIONS[p.position], `started ${p.starts_last_5} of last 5`].filter(Boolean).join(' · '))}</span></li>`).join('')}</ul>`
      : '<p class="mk-note">No recent starts recorded.</p>'}</div>`; };
  return `<div class="mk-h"><h2 class="cond">Team news</h2><p>Likely outfield starters, from starts in the last 5</p></div>
    <div class="mk-two">${side('home')}${side('away')}</div>
    <p class="mk-note">Line-ups are confirmed about an hour before kickoff. Injuries and suspensions are not in Spix's data yet.</p>`;
}
SECTIONS.news = newsSection;
