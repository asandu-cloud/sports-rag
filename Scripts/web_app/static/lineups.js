/* Line-ups (D134, owner 25 Nov, version A): one section for the match page and the Research Area's Team news, from
   the lineups block of /api/research/matches/{event_id} (D135, behind SPIX_LINEUPS; null while the switch is off).
   A named eleven is drawn on a pitch in its formation, keeper at the foot, with the bench under it. Until a team names
   its eleven, the players with most starts in its last 5: a record of past starts, never a prediction. The Research
   Area's shadow root calls the same functions through window, and adopts a copy of lineups.css. */
const LU_POSITIONS = {G: 'goalkeeper', D: 'defender', M: 'midfielder', F: 'forward'};
const LU_ORDER = ['G', 'D', 'M', 'F'];
// API-Football's short codes for a match under way or over (as the board's Live tab and the match view read them).
const LU_KICKED = ['1H', 'HT', '2H', 'ET', 'BT', 'P', 'LIVE', 'INT', 'SUSP', 'FT', 'AET', 'PEN', 'AWD', 'WO', 'ABD'];
const LU_RECHECK = 5 * 60 * 1000, LU_FROM = 75 * 60 * 1000, LU_UNTIL = 15 * 60 * 1000;
const PAST_STARTS = "the players who started most of each team's last 5 matches";

const luEsc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const luCap = word => word ? word[0].toUpperCase() + word.slice(1) : '';

// The pitch's name: initials dropped ("D. Henderson"), and a name over 12 letters keeps its last words.
function luShortName(name) {
  const words = String(name || '').split(/\s+/).filter(word => word && !/^\p{L}\.$/u.test(word));
  const full = words.join(' ');
  return full.length > 12 && words.length > 1 ? words.slice(1).join(' ') : full || String(name || '');
}

// Rows from the keeper out: API-Football's grid ("row:column") when every starter has one, else by position.
function luRows(xi) {
  const cells = xi.map(p => /^(\d+):(\d+)$/.exec(p.grid || ''));
  if (xi.length && cells.every(Boolean)) {
    const rows = new Map();
    xi.forEach((p, i) => { const row = Number(cells[i][1]); rows.set(row, [...(rows.get(row) || []), [Number(cells[i][2]), p]]); });
    return [...rows.keys()].sort((a, b) => a - b).map(row => rows.get(row).sort((a, b) => a[0] - b[0]).map(([, p]) => p));
  }
  const line = p => LU_ORDER.includes(p.position) ? p.position : 'M';
  return LU_ORDER.map(key => xi.filter(p => line(p) === key)).filter(row => row.length);
}

const luNamed = side => Boolean(side?.start_xi?.length);
// confirmed | partial | waiting | off (no block: the switch is off) | unavailable | missed (under way or over, no eleven).
// Without a status (the match page: the board's card has none), the clock says whether a match began when the switch
// is off; with it on, the backend's "unavailable" already covers a closed window.
function luState(match, now = new Date()) {
  const l = match.lineups, sides = {home: luNamed(l?.home), away: luNamed(l?.away)};
  const code = String(match.fixture?.status || '').toUpperCase();
  let kind = sides.home && sides.away ? 'confirmed' : sides.home || sides.away ? 'partial'
    : l == null ? 'off' : l.status === 'unavailable' ? 'unavailable' : 'waiting';
  if (code ? LU_KICKED.includes(code) && ['off', 'waiting', 'unavailable'].includes(kind)
    : kind === 'off' && Date.parse(match.fixture?.kickoff) <= now.getTime()) kind = 'missed';
  return {kind, sides};
}

// The match page asks again every 5 minutes while an eleven may still come: 75 minutes before kickoff to 15 after.
function luRecheckIn(match, now = new Date()) {
  if (!['waiting', 'partial'].includes(luState(match, now).kind)) return null;
  const toKickoff = Date.parse(match.fixture?.kickoff) - now.getTime();
  return toKickoff <= LU_FROM && toKickoff >= -LU_UNTIL ? LU_RECHECK : null;
}

const luProfile = p => { const id = p.research_id !== undefined ? p.research_id : p.player_id;
  return /^[1-9]\d*$/.test(String(id ?? '')) ? `players/${id}` : null; };
const luLink = (p, html, label) => { const route = luProfile(p);
  return route ? `<a class="lu-link" href="/app#research/${route}" data-research-route="${route}"${label ? ` aria-label="${luEsc(label)}"` : ''}>${html}</a>` : null; };
const luTeamName = (match, side) => match.teams?.[side]?.team?.name || match.fixture?.[side]?.name || (side === 'home' ? 'Home' : 'Away');

function luPlayer(p) {
  const label = [p.name, p.number != null ? `number ${p.number}` : '', LU_POSITIONS[p.position] || ''].filter(Boolean).join(', ');
  const short = luEsc(luShortName(p.name));
  const name = luLink(p, short, label) || `<span class="lu-name"><span aria-hidden="true">${short}</span><span class="lu-sr">${luEsc(label)}</span></span>`;
  return `<span class="lu-p" role="listitem"><span class="lu-dot" aria-hidden="true">${luEsc(p.number ?? p.position ?? '')}</span>${name}</span>`;
}

function luEleven(match, side) {
  const team = match.lineups[side], name = luTeamName(match, side);
  const coach = typeof team.coach === 'string' ? team.coach : team.coach?.name;
  const bench = (team.bench || []).map(p => luLink(p, luEsc(p.name)) || luEsc(p.name));
  return `<div class="lu-team"><div class="lu-th"><strong>${luEsc(name)}</strong>${team.formation ? `<span>${luEsc(team.formation)}</span>` : ''}</div>
    ${coach ? `<p class="lu-coach">Coach ${luEsc(coach)}</p>` : ''}
    <div class="lu-pitch" role="list" aria-label="${luEsc(name)} starting eleven">${luRows(team.start_xi).map(row => `<div class="lu-row">${row.map(luPlayer).join('')}</div>`).join('')}</div>
    ${bench.length ? `<p class="lu-bench"><b>Bench</b> ${bench.join(', ')}</p>` : ''}</div>`;
}

// The past-starts list: the block's own (keeper and ten outfield) or, without one, the match view's 8 outfield shooters.
function luMost(match, side) {
  const own = match.lineups?.likely?.[side], list = own || match.teams?.[side]?.shooters || [];
  const rows = [...list].sort((a, b) => LU_ORDER.indexOf(a.position) - LU_ORDER.indexOf(b.position) || b.starts_last_5 - a.starts_last_5);
  const row = p => { const starts = Number(p.starts_last_5) || 0;
    return `<li><span>${luLink(p, luEsc(p.name)) || luEsc(p.name)}${LU_POSITIONS[p.position] ? `<small>${luCap(LU_POSITIONS[p.position])}</small>` : ''}</span><span class="lu-sq" role="img" aria-label="Started ${starts} of the last 5">${[0, 1, 2, 3, 4].map(i => i < starts ? '<i class="on"></i>' : '<i></i>').join('')}</span><span class="lu-of" aria-hidden="true">${starts} of 5</span></li>`; };
  return `<div class="lu-team"><div class="lu-th"><strong>${luEsc(luTeamName(match, side))}</strong><span>Most starts, last 5</span></div>
    ${rows.length ? `<ul class="lu-list">${rows.map(row).join('')}</ul>` : '<p class="lu-note">No recent starts recorded.</p>'}
    ${rows.length && !own ? '<p class="lu-note">Outfield players only.</p>' : ''}</div>`;
}

function luLead(match, {kind, sides}) {
  const below = `Below, ${PAST_STARTS}. This is a record of past starts, not a prediction.`;
  if (kind === 'confirmed') return 'Both teams have named their starting eleven.';
  if (kind === 'partial') {
    const [named, other] = sides.home ? ['home', 'away'] : ['away', 'home'];
    return `${luTeamName(match, named)} have named their starting eleven. ${luTeamName(match, other)} have not named theirs yet: below, the players who started most of their last 5 matches, a record of past starts, not a prediction.`;
  }
  if (kind === 'waiting') return `Teams name their line-ups about 20 to 40 minutes before kickoff, and they show here. Until then, ${PAST_STARTS}. This is a record of past starts, not a prediction.`;
  if (kind === 'off') return `Teams name their line-ups about 20 to 40 minutes before kickoff. ${below}`;
  if (kind === 'unavailable') return `Spix's data provider has no line-ups for this match. ${below}`;
  return `Spix has no line-ups for this match. Below, ${PAST_STARTS} before it.`;
}

const LU_CHIPS = {confirmed: 'Confirmed', partial: 'One named', waiting: 'Not out yet', off: 'Not out yet', unavailable: 'Not available', missed: 'Not available'};

// The whole section. title: the caller's heading HTML (the match page's or Team news'), placed as given.
function spixLineups(match, {title = '', now = new Date()} = {}) {
  const state = luState(match, now), checked = match.lineups?.checked_at && state.kind !== 'off' && state.kind !== 'missed'
    ? new Date(match.lineups.checked_at).toLocaleTimeString('en-GB', {hour: '2-digit', minute: '2-digit'}) : '';
  const side = key => state.sides[key] ? luEleven(match, key) : luMost(match, key);
  return `<div class="lu"><div class="lu-head">${title}<span class="lu-chip${state.kind === 'confirmed' ? ' ok' : ''}">${LU_CHIPS[state.kind]}</span>${checked ? `<span class="lu-when">Checked ${checked}</span>` : ''}</div>
    <p class="lu-lead">${luEsc(luLead(match, state))}</p>
    <div class="lu-two">${side('home')}${side('away')}</div>${state.kind === 'confirmed' ? '' : `<p class="lu-note">Injuries and suspensions are not in Spix&#39;s data yet.</p>`}</div>`;
}
