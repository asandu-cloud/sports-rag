// Research Area overhaul (D107, task 5): a team's next matches, from B1's research-upcoming.v1. The rail on wide
// screens, the floating bar and its sheet on phones (the owner's option C), and the next match in a player's or
// referee's head. Pure functions returning HTML; research.js places them and wires the clicks.
import {escape, one} from './insight.js?v=research-20261116g';

// Kickoffs in the viewer's own time, as the board shows them.
const part = (iso, options) => new Date(iso).toLocaleString('en-GB', options);
const weekday = iso => part(iso, {weekday: 'short'});
const dayMonth = iso => part(iso, {day: 'numeric', month: 'short'});
const clock = iso => part(iso, {hour: '2-digit', minute: '2-digit', hour12: false});
const localDay = iso => part(iso, {year: 'numeric', month: '2-digit', day: '2-digit'});
export const whenText = (iso, joiner = ' · ') => `${weekday(iso)} ${dayMonth(iso)}${joiner}${clock(iso)}`;
export const opensDay = iso => `${weekday(iso)} ${dayMonth(iso)}`;
export const possessive = name => name.endsWith('s') ? `${name}'` : `${name}'s`;
const COUNT = ['', 'match', 'two', 'three', 'four', 'five'];

export function crest(team, size = '') {
  const code = team?.short_code || team?.name?.slice(0, 3).toUpperCase() || '';
  return `<span class="crest${size ? ' ' + size : ''}" aria-hidden="true">${team?.logo_url ? `<img src="${escape(team.logo_url)}" alt="" loading="lazy" data-fallback="${escape(code)}">` : escape(code)}</span>`;
}
// "v Leeds" at home, "at Nottingham Forest" away; a referee's appointment names both teams.
const fixtureWords = m => m.venue === 'home' ? `<span class="vw">v</span> ${escape(m.away.name)}`
  : m.venue === 'away' ? `<span class="vw">at</span> ${escape(m.home.name)}`
  : `${escape(m.home.name)} <span class="vw">v</span> ${escape(m.away.name)}`;
const opponent = m => m.venue === 'away' ? m.home : m.away;

export function readState(read, now = new Date()) {
  if (read?.status === 'ready') return 'Read ready';
  if (read?.status === 'preliminary') return 'Preliminary';
  if (!read?.opens_at) return 'Read not out yet';
  const opens = new Date(read.opens_at), days = (opens - now) / 86400000;
  if (localDay(read.opens_at) === localDay(now.toISOString())) return 'Read due today';
  return days < 6 ? `Read due ${weekday(read.opens_at)}` : `Read due ${dayMonth(read.opens_at)}`;
}

export function nextCard(m, {now = new Date(), active = null} = {}) {
  const read = m.read || {}, expected = read.expected;
  const figures = expected && ['goals', 'corners', 'cards'].some(k => expected[k] != null)
    ? `<span class="nm-figs">${['goals', 'corners', 'cards'].filter(k => expected[k] != null).map(k => `<span><b>${one(expected[k])}</b>${k}</span>`).join('')}<span class="nm-by">Spix expects</span></span>`
    : `<span class="nm-figs"><span>Spix's read comes out ${read.opens_at ? opensDay(read.opens_at) : 'nearer the match'}. Past numbers are ready now.</span></span>`;
  const tone = read.status === 'ready' ? '' : read.status === 'preliminary' ? ' pre' : ' wait';
  const side = m.venue ? `<small>${m.venue === 'home' ? 'Home' : 'Away'}</small>` : '';
  return `<button class="nm" type="button" data-open-match="${escape(m.event_id)}" aria-current="${String(active === m.event_id)}">
    <span class="nm-top"><span><b>${escape(weekday(m.kickoff))} ${escape(dayMonth(m.kickoff))}</b> · ${escape(clock(m.kickoff))}</span><span>${escape(m.competition?.name || '')}</span></span>
    <span class="nm-team">${crest(m.venue ? opponent(m) : m.home)}<span class="cond">${fixtureWords(m)}</span>${side}</span>
    ${figures}
    <span class="nm-foot"><span>${m.referee?.name ? `Referee ${escape(m.referee.name)}` : 'Referee not named yet'}</span><span class="status${tone}">${escape(readState(read, now))}</span></span></button>`;
}

const matchesOf = upcoming => upcoming?.matches?.length ? upcoming.matches : null;

export function railHTML(upcoming, {now = new Date(), active = null} = {}) {
  const list = matchesOf(upcoming);
  if (!list) return '';
  const n = list.length, name = upcoming.subject?.name || 'This team';
  return `<aside class="next-rail" aria-labelledby="railTitle"><h2 id="railTitle" class="cond">Next ${n === 1 ? 'match' : 'matches'}</h2>
    <p class="sub">${escape(possessive(name))} next ${n === 1 ? 'match' : COUNT[n] || n}, in every competition</p>
    ${list.map(m => nextCard(m, {now, active})).join('')}
    <p class="rail-note">Referees appear once they are named; appointments usually come two to three days before.</p></aside>`;
}

export function barHTML(upcoming) {
  const list = matchesOf(upcoming);
  if (!list) return '';
  const m = list[0], more = list.length - 1;
  const open = more ? 'data-open-sheet aria-haspopup="dialog"' : `data-open-match="${escape(m.event_id)}"`;
  return `<button class="next-bar" type="button" ${open}>${crest(m.venue ? opponent(m) : m.home)}
    <span class="nb-t"><span class="cond">Next: ${fixtureWords(m)}</span><small>${escape(whenText(m.kickoff))}${more ? ` · and ${more} more` : ''}</small></span>
    <span class="go">Open ›</span></button>`;
}

export function sheetHTML(upcoming, {now = new Date(), active = null} = {}) {
  const list = matchesOf(upcoming);
  if (!list) return '';
  const title = `${possessive(upcoming.subject?.name || 'The team')} next ${list.length === 1 ? 'match' : 'matches'}`;
  return `<div class="sheet-bg" data-close-sheet></div>
    <div class="sheet" role="dialog" aria-modal="true" aria-labelledby="sheetTitle" tabindex="-1">
      <div class="grab" aria-hidden="true"></div>
      <div class="sheet-h"><h2 id="sheetTitle" class="cond">${escape(title)}</h2><button class="sheet-x" type="button" data-close-sheet aria-label="Close">×</button></div>
      ${list.map(m => nextCard(m, {now, active})).join('')}</div>`;
}

export function headNext(m, label) {
  if (!m) return '';
  return `<button class="head-next" type="button" data-open-match="${escape(m.event_id)}">
    <span class="hn-l">${escape(label)}</span> <span class="hn-t"><b>${fixtureWords(m)}</b> · ${escape(whenText(m.kickoff, ', '))}</span>
    <span class="go">Open ›</span></button>`;
}
