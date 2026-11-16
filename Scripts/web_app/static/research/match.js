// Research Area overhaul (D107, task 6): one match's research view, from B2's research-match.v1. The head, How Spix
// reads it, the markets row, the open market's section (SECTIONS, filled by tasks 7 to 9) and the foot line. Pure
// functions returning HTML; research.js places them and wires the clicks.
import {escape, one, LINES, DEFAULT_LINE, passes, hitCount, strip, shortDay} from './insight.js?v=research-20261116g';
import {crest, whenText, readState, opensDay, possessive} from './upcoming.js?v=research-20261116g';

export const MARKETS = ['cards', 'sot', 'goals', 'corners', 'result', 'h2h', 'news'];
export const MARKET_LABELS = {cards: 'Cards', sot: 'Shots on target', goals: 'Goals', corners: 'Corners', result: 'Result',
  h2h: 'Head to head', news: 'Team news'};
// market -> (match, view) => html. Tasks 7 to 9 add theirs; a market's button shows once its section exists.
export const SECTIONS = {};

const GAMCARE = 'https://www.gamcare.org.uk/get-support/';
const pct = chance => `${Math.round(chance * 100)}%`;
const back = label => `<button type="button" class="text-button back-button" data-match-back><svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><path d="m14 6-6 6 6 6"/></svg><span>${escape(label)}</span></button>`;

export function matchHead(match, {now = new Date()} = {}) {
  const f = match.fixture, read = f.read || {};
  const referee = f.referee?.name
    ? `<span>Referee <button type="button" class="link" data-referee="${escape(f.referee.referee_key)}">${escape(f.referee.name)}</button></span>`
    : '<span>Referee not named yet</span>';
  const tone = read.status === 'ready' ? '' : read.status === 'preliminary' ? ' pre' : ' wait';
  return `<div class="mv-head">${crest(f.home, 'md')}<h2 class="mv-title" id="matchTitle" tabindex="-1">${escape(f.home.name)} <span class="vw">v</span> ${escape(f.away.name)}</h2>${crest(f.away, 'md')}</div>
    <p class="mv-meta"><span>${escape(whenText(f.kickoff))}</span><span>${escape(f.competition?.name || '')}</span>${referee}<span class="status${tone}">${escape(readState(read, now))}</span></p>`;
}

// How Spix reads it: the read's own figures, one marked for the open market. Before the read is out, when it comes.
export function spixReads(match, market) {
  const f = match.fixture, r = match.read;
  const open = '<section class="reads" aria-labelledby="readsTitle"><p class="cap" id="readsTitle">How Spix reads it</p>';
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
    balance = `<div class="bal" aria-hidden="true"><i style="flex:${res.home};background:var(--chart-1)"></i><i style="flex:${res.draw};background:var(--border2)"></i><i style="flex:${res.away};background:var(--chart-2)"></i></div>
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
const TEAM_TOTAL = {cards: ['cards_for', 'cards_against'], goals: ['goals_for', 'goals_against'], corners: ['corners_for', 'corners_against']};

function lineRow(market, line) {
  return `<div class="lines"><span>Line</span><div class="segmented" role="group" aria-label="${MARKET_LABELS[market]} line">${LINES[market].map(v => `<button type="button" data-mline="${v}" aria-pressed="${v === line}">${v}</button>`).join('')}</div></div>`;
}

// Over and Under at the line: Spix's chance and the typical price from the read. Add only where the slip takes it.
function pickPair(match, market, line, unit, {add = false} = {}) {
  const r = match.read;
  if (!r) return '<p class="mk-note">Spix\'s chances and typical prices show once the read is out.</p>';
  const lines = r.lines?.[market] || [], at = lines.find(l => l.line === line);
  if (!at) return `<p class="mk-note">No typical price for ${line} ${unit} yet.${lines.length ? ` Spix's chances cover ${listWords(lines.map(l => l.line))}.` : ''}</p>`;
  const pick = side => {
    const o = at[side];
    if (!o) return '';
    const label = `${side === 'over' ? 'Over' : 'Under'} ${line} ${unit}`, lean = o.chance > 0.5;
    const price = o.typical_price == null ? '' : ` · typical price <b>${Number(o.typical_price).toFixed(2)}</b>`;
    const button = add ? `<button type="button" class="mk-add" data-add="${market}|${side}|${line}" aria-label="Add ${label} to the slip">Add</button>` : '';
    return `<div class="mk-pick${lean ? ' lean' : ''}"><span class="mk-pl">${label}</span>${lean ? ' <span class="mk-lean">Spix leans this way</span>' : ''}<span class="mk-pc">Spix <b>${pct(o.chance)}</b>${price}</span>${button}</div>`;
  };
  return `<div class="mk-pair">${pick('over')}${pick('under')}</div>`;
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
    rows.push(evRow(crest(t.team), `${possessive(t.team.name)} matches`, `Last ${t.last_matches.length}, every competition`,
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

export function cardsSection(match, view = {}) {
  const line = LINES.cards.includes(view.line) ? view.line : DEFAULT_LINE.cards, r = match.read;
  const expects = r?.cards_expected != null ? `<p>Spix expects <b>${one(r.cards_expected)}</b> cards</p>` : '';
  return `<div class="mk-h"><h2 class="cond">Cards</h2>${expects}</div>${lineRow('cards', line)}${pickPair(match, 'cards', line, 'cards')}
    ${historyRows(match, 'cards', line, 'cards')}${whoGetsCards(match)}${refereeBlock(match)}${bookedBlock(match)}`;
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
