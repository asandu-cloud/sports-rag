// A played match from a team's last 10 (D121), opened from "Last 10 at a glance": the head, the last 10 to move
// between, the market's lead, what Spix said before kickoff (D122's spix_before_kickoff, graded by the backend) and the
// match in numbers. Pure functions returning HTML; research.js places them and wires the clicks.
import {escape, one} from './insight.js?v=research-20261125a';
import {crest, whenText, opensDay, possessive} from './upcoming.js?v=research-20261125a';

export const PLAYED_MARKETS = [['cards', 'Cards'], ['goals', 'Goals'], ['corners', 'Corners']];
const GAMCARE = 'https://www.gamcare.org.uk/get-support/';
// The glance's first row for each market: the lead says where the match fell against it.
const MARK = {cards: 4, goals: 3, corners: 10};
const KEYS = {cards: ['cards', 'opponent_cards'], goals: ['goals_for', 'goals_against'], corners: ['corners_for', 'corners_against']};
const NOUN = {cards: ['card', 'cards'], goals: ['goal', 'goals'], corners: ['corner', 'corners']};
const STATUS = {FT: 'Full time', AET: 'After extra time', PEN: 'After penalties'};
const OUTCOME = {won: 'Came in', lost: 'Did not come in', push: 'Stake back', half_won: 'Half came in', half_lost: 'Half lost'};

export const lastTen = d => (d.match_log || d.last_10 || []).slice(-10);
const count = (n, market) => `${n} ${NOUN[market][n === 1 ? 0 : 1]}`;
const mean = values => { const known = values.filter(v => v != null); return known.length ? known.reduce((a, b) => a + b, 0) / known.length : null; };
const back = label => `<button type="button" class="text-button back-button" data-match-back><svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><path d="m14 6-6 6 6 6"/></svg><span>${escape(label)}</span></button>`;
const scoreLine = m => `${escape(m.home.name)} ${m.score?.home ?? '?'}–${m.score?.away ?? '?'} ${escape(m.away.name)}`;
const code = team => team?.abbreviation || team?.short_code || team?.name?.slice(0, 3).toUpperCase() || '';
const total = (m, market) => { const [a, b] = KEYS[market].map(k => m.values?.[k]); return a == null || b == null ? null : a + b; };
function result(m) {
  const f = m.values?.goals_for, a = m.values?.goals_against;
  if (f == null || a == null) return {letter: '–', word: 'result not recorded', score: ''};
  return {letter: f > a ? 'W' : f < a ? 'L' : 'D', word: f > a ? 'won' : f < a ? 'lost' : 'drew', score: `${f}–${a}`};
}

export const outcomeWords = outcome => OUTCOME[outcome] || 'Not graded';
// Counts what came in; what happened to the rest follows, so a stake back is never read as a loss or a win.
export function pickSummary(picks) {
  const n = picks.length, by = key => picks.filter(p => (p.outcome ?? null) === key).length, won = by('won');
  const head = won === 0 && by(null) === 0 ? `None of the ${n} picks came in` : `${won} of ${n} ${n === 1 ? 'pick' : 'picks'} came in`;
  const rest = [[by('half_won'), 'half came in'], [by('push'), 'stake back'], [by(null), 'not graded']].filter(([k]) => k).map(([k, w]) => `${k} ${w}`);
  return [head, ...rest].join(', ');
}
// The figure a pick was settled on: the market's total, else the score as written.
function settledOn(m, market) {
  const t = KEYS[market] ? total(m, market) : null;
  return t != null ? count(t, market) : `${m.score?.home ?? '?'}–${m.score?.away ?? '?'}`;
}

function spixSection(m, market) {
  if (!('spix_before_kickoff' in m)) return '';
  const sb = m.spix_before_kickoff, open = '<section class="pr-card" aria-label="What Spix said before kickoff"><h3>What Spix said before kickoff</h3>';
  if (!sb) return `${open}<p class="pr-none">No read from before kickoff was kept for this match, so there are no picks to show.</p></section>`;
  const picks = picksOf(sb);
  if (!picks.length) return `${open}<p class="pr-none">Spix made no picks for this match.</p></section>`;
  const minutes = sb.read_at ? Math.round((new Date(m.date) - new Date(sb.read_at)) / 60000) : null;
  const when = sb.read_at ? ` · read at ${whenText(sb.read_at).split(' · ')[1]}${minutes > 0 ? `, ${minutes} ${minutes === 1 ? 'minute' : 'minutes'} before` : ''}` : '';
  const row = p => {
    const done = p.outcome === 'won' || p.outcome === 'half_won', missed = p.outcome === 'lost' || p.outcome === 'half_lost', icon = done
      ? '<svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12 5 5 9-10"/></svg>'
      : missed ? '<svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18"/></svg>' : '<svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><path d="M6 12h12"/></svg>';
    const chance = Number.isFinite(p.chance) ? `${Math.round(p.chance * 100)}%` : '–';
    return `<div class="pr-pick${p.market === market ? ' on' : ''}"><b>${escape(p.label)}</b><span class="pr-ch">${chance}</span><span class="pr-out ${done ? 'in' : missed ? 'out' : 'open'}">${icon}<span>${outcomeWords(p.outcome)}<small>${escape(settledOn(m, p.market))}</small></span></span></div>`;
  };
  const group = (title, list) => list?.length ? `<p class="pr-group">${title}</p>${list.map(row).join('')}` : '';
  return `${open}<p class="pr-cap">As the board showed it at kickoff${when}</p><p class="pr-sum">${pickSummary(picks)}</p>
    ${group('Most likely', sb.most_likely)}${group('Value', sb.value)}
    <p class="pr-note">The percentage is Spix's chance at the time. One match says little on its own.</p></section>`;
}

function numbersSection(d, m, rows, market) {
  const name = d.subject.name, opp = m.opponent?.name || 'Opponent', v = m.values || {};
  const usual = key => mean(rows.map(r => r.values?.[key]));
  const possOpp = v.possession == null ? null : 100 - v.possession;
  const list = [
    ['Goals', 'goals_for', 'goals_against', 'goals'], ['Expected goals', 'xg_for', 'xg_against'], ['Shots', 'shots_for', 'shots_against'],
    ['On target', 'sot_for', 'sot_against'], ['Corners', 'corners_for', 'corners_against', 'corners'], ['Cards', 'cards', 'opponent_cards', 'cards'],
    ['Possession', 'possession', null], ['Fouls', 'fouls', null]];
  const lines = list.filter(([, own]) => v[own] != null).map(([label, own, other, key]) => {
    const a = v[own], b = other ? v[other] ?? null : own === 'possession' ? possOpp : null, pct = own === 'possession' ? '%' : '';
    const top = a != null && b != null ? Math.max(a, b) : 0, width = x => top ? Math.round(x / top * 10000) / 100 : 0;
    const u = usual(own);
    return `<div class="pr-row${key === market ? ' on' : ''}"><span>${label}</span><b>${a}${pct}</b><span class="pr-bars"><span class="pr-l"><i class="pr-own" style="width:${width(a)}%"></i></span><span class="pr-r"><i class="pr-opp" style="width:${b == null ? 0 : width(b)}%"></i></span></span><b>${b == null ? '–' : `${b}${pct}`}</b><span class="pr-u">${u == null ? '–' : one(u) + pct}</span></div>`;
  }).join('');
  const fouls = v.fouls != null ? `<p class="pr-note">${escape(possessive(opp))} fouls were not recorded.</p>` : '';
  return `<section class="pr-card" aria-label="The match in numbers"><h3>The match in numbers</h3>
    <p class="pr-cap">${escape(name)} in blue, ${escape(opp)} in grey · usual is ${escape(possessive(name))} average over the last ${rows.length}</p>
    <div class="pr-head"><span></span><span>${escape(code(d.subject))}</span><span></span><span>${escape(code(m.opponent))}</span><span>Usual</span></div>${lines}${fouls}</section>`;
}

// Where a match's total fell against the glance's first row: reached it, one short, or under.
const against = (t, k, sep) => t >= k ? `: a ${k}+ match` : `${sep}${t === k - 1 ? 'one short of' : 'under'} ${k}`;
const Noun = market => NOUN[market][1][0].toUpperCase() + NOUN[market][1].slice(1);
const picksOf = sb => [...(sb.most_likely || []), ...(sb.value || [])];

function lead(d, m, rows, market) {
  const name = escape(d.subject.name), opp = escape(m.opponent?.name || 'Opponent'), t = total(m, market), k = MARK[market];
  if (t == null) return `<p class="pr-lead">${Noun(market)} were not recorded for this match.</p>`;
  const where = `${against(t, k, ', ')}.`;
  const [ownKey, otherKey] = KEYS[market], a = m.values[ownKey], b = m.values[otherKey], n = rows.length;
  const ua = one(mean(rows.map(r => r.values?.[ownKey]))), ub = one(mean(rows.map(r => r.values?.[otherKey])));
  const sub = market === 'cards' ? `${name} got ${a} and ${opp} ${b}. Over their last ${n}, ${name} got ${ua} a match and their opponents ${ub}.`
    : market === 'goals' ? `${name} scored ${a} and conceded ${b}. Over their last ${n} they scored ${ua} a match and conceded ${ub}.`
    : `${name} won ${a} and conceded ${b}. Over their last ${n} they won ${ua} a match and conceded ${ub}.`;
  return `<p class="pr-lead"><mark>${count(t, market)}</mark>${where}</p><p class="pr-sub">${sub}</p>`;
}

export function playedReport(d, index, market) {
  const rows = lastTen(d), m = rows[index], name = d.subject.name;
  const kept = r => !!r.spix_before_kickoff;
  const pills = rows.map((r, i) => {
    const res = result(r), opp = r.opponent?.name || 'Opponent';
    const label = `${r.venue === 'home' ? 'v' : 'at'} ${opp}, ${opensDay(r.date)}: ${res.word}${res.score ? ' ' + res.score : ''}${kept(r) ? ", Spix's read kept" : ''}`;
    return `<button type="button" class="pr-pill" data-played="${i}" aria-current="${i === index}" aria-label="${escape(label)}">${escape(r.opponent?.abbreviation || r.opponent?.short_code || '?')}<small>${res.letter} ${res.score}</small>${kept(r) ? '<span class="pr-kept"></span>' : ''}</button>`;
  }).join('');
  const chips = PLAYED_MARKETS.map(([key, label]) => `<button type="button" data-played-market="${key}" aria-current="${key === market}">${label}</button>`).join('');
  const step = (i, word) => {
    const r = rows[i];
    if (!r) return '<span></span>';
    const t = total(r, market);
    return `<button type="button" class="pr-step" data-played="${i}"><small>${word}</small><b>${scoreLine(r)}</b><small>${opensDay(r.date)}${t == null ? '' : ` · ${count(t, market)}`}</small></button>`;
  };
  return `${back(name)}
    <div class="mv-head pr-top">${crest(m.home, 'md')}<h2 class="mv-title" id="matchTitle" tabindex="-1">${scoreLine(m)}</h2>${crest(m.away, 'md')}</div>
    <p class="mv-meta"><span>${escape(whenText(m.date))}</span><span>${escape(d.competition?.name || '')}</span><span>${escape(name)} ${m.venue === 'home' ? 'at home' : 'away'}</span><span class="status wait">${STATUS[m.status] || 'Played'}</span></p>
    <nav class="pr-strip" aria-label="${escape(possessive(name))} last ${rows.length}, oldest first">${pills}</nav>
    <nav class="mkts" aria-label="Markets">${chips}</nav>
    ${lead(d, m, rows, market)}
    <div class="pr-grid">${spixSection(m, market)}${numbersSection(d, m, rows, market)}</div>
    <nav class="pr-nav" aria-label="Earlier and later matches">${step(index - 1, '‹ Earlier')}${step(index + 1, 'Later ›')}</nav>
    <p class="disclaimer">Past matches are records, not a forecast. Spix's figures come from statistical models. 18+ · <a href="${GAMCARE}" target="_blank" rel="noopener">GamCare</a></p>`;
}

// The glance card (D121, task 2): what a square stands for, shown while it is pointed at or focused.
export function glanceCard(d, index, market) {
  const m = lastTen(d)[index], name = d.subject.name, opp = m.opponent?.name || 'Opponent', t = total(m, market);
  const [a, b] = KEYS[market].map(k => m.values?.[k]), sb = m.spix_before_kickoff;
  const spix = 'spix_before_kickoff' in m ? `<p class="gc-spix">Spix before kickoff: <b>${sb ? escape(pickSummary(picksOf(sb))) : 'no read kept'}</b></p>` : '';
  return `<p class="gc-when">${opensDay(m.date)} · ${escape(d.competition?.name || '')} · ${escape(name)} ${m.venue === 'home' ? 'at home' : 'away'}</p>
    <p class="gc-score">${scoreLine(m)}</p>
    <p class="gc-stat"><span>${Noun(market)} in the match</span><b>${t ?? '–'}</b></p>
    <p class="gc-split">${escape(name)} ${a ?? '–'} · ${escape(opp)} ${b ?? '–'}${t == null ? '' : against(t, MARK[market], ' · ')}</p>${spix}
    <p class="gc-open">Click to open the match ›</p>`;
}

// The same on a phone, where nothing can be pointed at: a sheet with the match, its neighbours and "Open the match".
export function glanceSheet(d, index, market) {
  const rows = lastTen(d), m = rows[index], name = d.subject.name, opp = m.opponent?.name || 'Opponent', t = total(m, market);
  const [a, b] = KEYS[market].map(k => m.values?.[k]), sb = m.spix_before_kickoff, picks = sb ? picksOf(sb) : [];
  const step = (i, label, glyph) => `<button type="button" class="gs-step" data-glance-step="${i}" aria-label="${label}"${rows[i] ? '' : ' disabled'}>${glyph}</button>`;
  const spix = !('spix_before_kickoff' in m) ? ''
    : `<div class="gs-tile"><small>Spix before kickoff</small><b>${picks.length ? `${picks.filter(p => p.outcome === 'won').length} of ${picks.length}` : '–'}</b><small>${picks.length ? 'picks came in' : sb ? 'no picks' : 'no read kept'}</small></div>`;
  return `<div class="sheet-bg" data-close-sheet></div>
    <div class="sheet gsheet" role="dialog" aria-modal="true" aria-labelledby="sheetTitle" tabindex="-1">
      <div class="grab" aria-hidden="true"></div>
      <div class="gs-nav">${step(index - 1, 'Earlier match', '‹')}<span>Match ${index + 1} of ${rows.length} · ${opensDay(m.date)}</span>${step(index + 1, 'Later match', '›')}</div>
      <h3 id="sheetTitle">${scoreLine(m)}</h3>
      <p class="gs-meta">${STATUS[m.status] || 'Played'} · ${escape(d.competition?.name || '')} · ${escape(name)} ${m.venue === 'home' ? 'at home' : 'away'}</p>
      <div class="gs-tiles"><div class="gs-tile"><small>${Noun(market)} in the match</small><b>${t ?? '–'}</b><small>${escape(name)} ${a ?? '–'} · ${escape(opp)} ${b ?? '–'}</small></div>${spix}</div>
      <button type="button" class="gs-open" data-played-open="${index}">Open the match</button>
    </div>`;
}
