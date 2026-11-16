// Research Area overhaul (D107, task 6): one match's research view, from B2's research-match.v1. The head, How Spix
// reads it, the markets row, the open market's section (SECTIONS, filled by tasks 7 to 9) and the foot line. Pure
// functions returning HTML; research.js places them and wires the clicks.
import {escape, one} from './insight.js?v=research-20261116f';
import {crest, whenText, readState, opensDay} from './upcoming.js?v=research-20261116f';

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
