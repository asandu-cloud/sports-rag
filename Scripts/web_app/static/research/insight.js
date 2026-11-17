// Lines, hits and labelled charts shared by every research view (D107). Pure: strings in, HTML strings out.

export const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));

// One decimal at most; a dash only when the value is missing.
export const one = value => Number.isFinite(value) ? String(Math.round(value * 10) / 10) : '–';

export const LINES = {cards: [2.5, 3.5, 4.5, 5.5], goals: [1.5, 2.5, 3.5], corners: [8.5, 9.5, 10.5, 11.5], sot: [1, 2, 3]};
export const DEFAULT_LINE = {cards: 3.5, goals: 2.5, corners: 9.5, sot: 1};

// A half line is "over"; a whole-number line is "at least" (a player's 1+). Null when the value is missing.
export const passes = (value, line) => value == null ? null : Number.isInteger(line) ? value >= line : value > line;

export function hitCount(values, line) {
  const known = values.filter(v => v != null);
  return {hits: known.filter(v => passes(v, line)).length, of: known.length};
}

// A profile's match_log row in the match view's row shape (research-match.v1 last_matches).
export function fromProfileRow(m) {
  const v = m.values || {}, n = key => v[key] ?? null;
  return {event_id: m.event_id ?? null, fixture_id: m.fixture_id ?? null, date: m.date, venue: m.venue ?? null,
    competition: m.competition ?? null, opponent: m.opponent ?? null, score: m.score ?? null,
    cards_for: n('cards'), cards_against: n('opponent_cards'), corners_for: n('corners_for'), corners_against: n('corners_against'),
    goals_for: n('goals_for'), goals_against: n('goals_against'), sot_for: n('sot_for'), sot_against: n('sot_against')};
}

export function marketTotal(row, market) {
  const a = row[`${market}_for`], b = row[`${market}_against`];
  return a == null || b == null ? null : a + b;
}

// One square a match, newest on the right: filled when it happened, dashed when not recorded.
export function strip(values, test, labels = []) {
  const known = values.filter(v => v != null);
  const marks = values.map((v, i) => {
    const name = labels[i] ? `${labels[i]}: ` : '';
    return `<i class="${v == null ? 'na' : test(v) ? 'y' : ''}" title="${escape(name + (v == null ? 'not recorded' : v))}"></i>`;
  }).join('');
  return `<span class="strip" role="img" aria-label="${known.filter(test).length} of ${known.length}">${marks}</span>`;
}

const niceMax = value => {
  const v = Math.max(1, value), step = v <= 4 ? 1 : v <= 8 ? 2 : v <= 20 ? 5 : 10;
  return Math.ceil(v / step) * step;
};

// How far the plot must reserve above its top so each column's number and goal dots stay inside the card.
// cols: {top: the column's height in px, dots: goals}. The number is 11px tall, 4px up; the dots start 22px up, each
// 8px with a 2px ring and 3px apart, at most 4. Returns 0 when everything fits, else the overflow plus 4px.
export function columnHeadroom(cols, height) {
  const reach = cols.map(({top, dots}) => {
    const n = Math.min(4, dots || 0);
    return Math.max(top + 15, n ? top + 22 + n * 8 + (n - 1) * 3 + 2 : 0);
  });
  const over = Math.max(0, ...reach) - height;
  return over > 0 ? over + 4 : 0;
}

// Columns with every value written on top, a zero drawn as a base mark, and the bettor's line dashed across.
// items: {value, top, bottom, title}. A whole-number line sits half a step low, between "1" and "0".
export function labelledColumns(items, {line, label, legend, noun}) {
  const max = niceMax(Math.max(line, ...items.map(it => it.value ?? 0)) + 0.5);
  const at = Number.isInteger(line) ? line - 0.5 : line;
  const cols = items.map(it => {
    const v = it.value, kind = v == null ? 'none' : v === 0 ? 'zero' : passes(v, line) ? 'hit' : 'miss';
    const height = v ? v / max * 100 : 0;
    return `<div class="lc-col ${kind}" title="${escape(it.title)}"><b>${v == null ? '–' : one(v)}</b><i style="height:${height}%"></i></div>`;
  }).join('');
  const n = items.length;
  return `<div class="lc"><div class="lc-plot" role="img" aria-label="${escape(label)}" style="--n:${n}">${cols}<div class="lc-line" style="bottom:${at / max * 100}%"></div></div>
    <div class="lc-x" style="--n:${n}">${items.map(it => `<span>${escape(it.top)}<small>${escape(it.bottom)}</small></span>`).join('')}</div>
    <div class="legend"><span><i class="lg-hit"></i>Reached the line</span><span><i class="lg-miss"></i>Did not</span><span><i class="lg-line"></i>${escape(legend)}</span></div>
    <details class="tbl"><summary>Show as table</summary><div class="tbl-wrap"><table><thead><tr><th>Match</th><th>${escape(noun)}</th></tr></thead><tbody>${items.map(it => `<tr><td>${escape(it.title)}</td><td>${it.value == null ? '–' : one(it.value)}</td></tr>`).join('')}</tbody></table></div></details></div>`;
}

// ----- The team page (task 2) -----
const WORDS = {cards: {noun: 'cards', heavy: 'card-heavy'}, goals: {noun: 'goals', heavy: 'high-scoring'}, corners: {noun: 'corners', heavy: 'corner-heavy'}};
const mean = values => { const known = values.filter(v => v != null); return known.length ? known.reduce((a, b) => a + b, 0) / known.length : null; };
export const shortDay = iso => new Date(iso).toLocaleDateString('en-GB', {day: 'numeric', month: 'short', timeZone: 'Europe/London'});
const where = row => `${row.venue === 'home' ? 'v' : 'at'} ${row.opponent?.abbreviation || row.opponent?.short_code || row.opponent?.name || '?'}`;

// "Chelsea's matches are card-heavy: 9 of their last 10 had 4 or more cards." The heavy word needs 7 in 10.
export function teamLead(name, rows, market, line = DEFAULT_LINE[market]) {
  const {hits, of} = hitCount(rows.map(r => marketTotal(r, market)), line);
  if (!of) return '';
  const w = WORDS[market], heavy = hits >= 0.7 * of;
  return `${escape(name)}'s matches${heavy ? ` are ${w.heavy}` : ''}: <mark>${hits} of their last ${of}</mark> had ${Math.ceil(line)} or more ${w.noun}.`;
}

export function teamLeadSub(name, rows, market) {
  const own = mean(rows.map(r => r[`${market}_for`])), other = mean(rows.map(r => r[`${market}_against`]));
  if (own == null || other == null) return '';
  if (market === 'cards') return other > own ? `their opponents got most of the cards (${one(other)} a match, ${name} ${one(own)})`
    : `${name} got most of the cards (${one(own)} a match, their opponents ${one(other)})`;
  if (market === 'goals') return `${name} scored ${one(own)} a match and conceded ${one(other)}`;
  return `${name} won ${one(own)} corners a match and conceded ${one(other)}`;
}

// Last 10 at a glance: the sketch's rows, each a strip of the matches and a written count.
export function glance(name, rows, market) {
  const total = rows.map(r => marketTotal(r, market)), own = rows.map(r => r[`${market}_for`]), other = rows.map(r => r[`${market}_against`]);
  const both = rows.map(r => r.goals_for == null || r.goals_against == null ? null : r.goals_for > 0 && r.goals_against > 0 ? 1 : 0);
  const list = {
    cards: [['4+ cards in the match', total, v => v >= 4], ['5+ cards in the match', total, v => v >= 5],
      [`${name} booked 2+ times`, own, v => v >= 2], ['Opponent booked 3+ times', other, v => v >= 3]],
    goals: [['3+ goals in the match', total, v => v >= 3], ['Both teams scored', both, v => v === 1], [`${name} scored 2+`, own, v => v >= 2]],
    corners: [['10+ corners in the match', total, v => v >= 10], [`${name} won 5+ corners`, own, v => v >= 5]]
  }[market];
  const labels = rows.map(where);
  return `<div class="glance">${list.map(([label, values, test]) => {
    const known = values.filter(v => v != null);
    return `<div class="g-row"><span class="g-l">${escape(label)}</span><span class="g-s">${strip(values, test, labels)}</span><span class="g-n">${known.filter(test).length} of ${known.length}</span></div>`;
  }).join('')}</div>`;
}

// Cards (or goals, corners) by match: the team's and the opponent's stacked, the total written, the line dashed.
// Up and down (owner, 17 Nov, D118): the team's bars rise above a centre line, the opponents' hang below it, and both
// names sit on the chart, so neither colour nor a legend has to say who is who. items: {up, down, label, where, who,
// sub}: "v" or "at", the opponent's short name, and a line under it (HTML, escaped by the caller). Both values missing
// draws a dash at the line.
export function upDown({name, items, legend = '', note = ''}) {
  const top = Math.max(0, ...items.flatMap(it => [it.up ?? 0, it.down ?? 0]));
  const step = top <= 4 ? 1 : top <= 8 ? 2 : top <= 20 ? 5 : 10, max = Math.max(step, Math.ceil(top / step) * step);
  const round = v => Math.round(v * 100) / 100, at = v => round(v / max * 50);
  const ticks = Array.from({length: max / step}, (_, i) => (i + 1) * step);
  const y = [...[...ticks].reverse().map(t => [t, round(50 - at(t))]), [0, 50], ...ticks.map(t => [t, round(50 + at(t))])];
  const cols = items.map(it => {
    const bars = it.up == null && it.down == null ? '<span class="ud-miss">–</span>'
      : `<i class="ud-up" style="height:${at(it.up || 0)}%"></i><i class="ud-down" style="height:${at(it.down || 0)}%"></i>`;
    return `<div class="ud-col" tabindex="0" data-tip="${escape(it.label)}" role="img" aria-label="${escape(it.label)}">${bars}</div>`;
  }).join('');
  return `<div class="ud" style="--n:${items.length}"><div class="ud-plot"><div class="ud-y"><div class="ud-in">${y.map(([t, p]) => `<span style="top:${p}%">${t}</span>`).join('')}</div></div>
    <div class="ud-area"><p class="ud-name ud-name-up">↑ ${escape(name)}</p><p class="ud-name ud-name-down">↓ Opponents</p><div class="ud-in">${y.filter(([t]) => t).map(([, p]) => `<span class="ud-g" style="top:${p}%"></span>`).join('')}<span class="ud-zero"></span><div class="ud-cols">${cols}</div></div></div></div>
    <div class="ud-x">${items.map(it => `<span class="ud-xl"><b><i>${escape(it.where)}</i> ${escape(it.who)}</b>${it.sub}</span>`).join('')}</div></div>${legend}${note}`;
}

// A market by match, up and down: the team's count above the line, the opponents' below. The match total sits under
// each label, marked when it clears the line, since a split column cannot carry the line itself.
export function stackedColumns(name, rows, market, line = DEFAULT_LINE[market]) {
  const items = rows.map(r => {
    const a = r[`${market}_for`], b = r[`${market}_against`], t = marketTotal(r, market), opp = r.opponent?.name || 'Opponent';
    const where = r.venue === 'home' ? 'v' : 'at';
    const label = `${where} ${opp}, ${shortDay(r.date)}: ${t == null ? 'not recorded' : `${name} ${a}, ${opp} ${b}, ${t} in all`}`;
    const total = t == null ? '–' : passes(t, line) ? `<mark>${t}</mark>` : t;
    return {up: a, down: b, label, where, who: r.opponent?.abbreviation || r.opponent?.short_code || '?', sub: total};
  });
  const own = mean(rows.map(r => r[`${market}_for`])), other = mean(rows.map(r => r[`${market}_against`]));
  return upDown({name, items,
    legend: `<div class="legend sc-legend"><span><i class="own"></i>${escape(name)} ${one(own)} a match</span><span><i class="opp"></i>Opponents ${one(other)} a match</span></div>`,
    note: `<p class="card-note">The number under each match is its total, marked when over ${line} ${WORDS[market].noun}.</p>`});
}

// ----- The match-by-match view (task 3) -----
// A metric's lines: its own list, else around its threshold; none for percentages and decimals (xG).
export function linesFor(metric) {
  if (metric.lines) return metric.lines;
  if (metric.percent || metric.decimals) return [];
  const t = metric.threshold || 1, s = metric.step || 1;
  return [t - s, t, t + s, t + 2 * s].filter(v => v > 0);
}
export const defaultLine = metric => metric.threshold ?? linesFor(metric)[0] ?? 1;

// "1 or more shots on target in 7 of their last 10." A half line reads "Over 3.5 cards".
export function statAnswer({noun, line, hits, of, missing = 0, season = false, unit = 'matches'}) {
  const what = Number.isInteger(line) ? `${line} or more ${noun}` : `Over ${line} ${noun}`;
  const count = season ? `${hits} of ${of} ${unit}` : `${hits} of their last ${of}`;
  return `${escape(what)} in <mark>${count}</mark>${!season && missing ? ' with data' : ''}${season ? ' this season' : ''}.`;
}

// ----- Ranks from one season, the referee page (task 4) -----
const cap = text => text ? text[0].toUpperCase() + text.slice(1) : '';
const shown = m => m == null || m.value == null ? null : m.unit === 'percent' ? m.display_value : one(m.value);
const scopeOf = (kind, d) => kind === 'referees' ? String(d.season.year) : `${d.competition.code}-${d.season.year}`;
const whole = n => Number.isFinite(n) ? n.toLocaleString('en-GB') : '–';

// The rank card uses only this page's season. Unranked, it shows the season's values without positions, says how far
// from the threshold it is, and links to the last season that has ranks; with B5's early ranks it shows them, marked.
// A profile's opening line from the API's headline, its figure marked. B5: a provisional rank (under the minutes
// threshold, among at least 10 ranked peers) says so under it, as the rank card does.
export function headlineLead(h, ranking = {}) {
  if (!h?.text || !h.display_value) return '';
  const shown = h.display_value.endsWith('%') ? h.display_value : one(h.value);
  let text = escape(h.text).replace(escape(h.display_value), `<mark>${escape(shown)}</mark>`);
  if (!/[.!?]$/.test(h.text)) text += '.';
  const so = Number.isFinite(ranking?.sample_minutes) ? ` · ${whole(ranking.sample_minutes)} minutes so far` : '';
  const early = h.provisional === true ? `<p class="quiet-line"><span class="quiet">Early season${so}</span> Rates move a lot with one match.</p>` : '';
  return `<p class="lead">${text}</p>${early}`;
}

// The backend's reasons for ranks it holds back while it rebuilds them; any other reason it may add says only "no
// ranks yet". too_few_peers: the owner's 10-peer floor (16 Nov).
const REBUILDING = ['ranks_not_built', 'ranks_stale_or_missing', 'rank_policy_changed_rebuild_required'];
export function rankCard(d, kind, {rankMetrics, rankGroups}) {
  const r = d.ranking, ranked = !!(r.ranked || r.provisional);
  const defs = rankMetrics[kind], groups = rankGroups[kind], scope = scopeOf(kind, d);
  const order = groups.order[r.peer_group] || groups.order.all || groups.order.F;
  const metrics = Object.values(d.metrics || {});
  const peerLabel = metrics.find(m => m.peer_label)?.peer_label, peers = metrics.find(m => m.peer_count)?.peer_count;
  const title = kind === 'players' ? `Against ${peerLabel || 'players in the same position'}` : kind === 'teams' ? 'Against the league' : 'Against other referees';
  const group = !ranked || !peers ? '' : r.provisional ? `${peers} in the group`
    : kind === 'players' ? `${peers} with ${whole(r.minimum_minutes)}+ minutes` : kind === 'teams' ? `${peers} teams · per match` : `${peers} referees with ${r.minimum_matches}+ matches`;
  const row = key => {
    const m = d.metrics?.[key], def = defs[key], value = shown(m);
    if (!m || !def || value == null) return '';
    const [label, stat] = def;
    if (!ranked || m.percentile == null) {
      return `<button class="rank plain" type="button" data-stat="${stat}" data-scope="${escape(scope)}" aria-label="${escape(`${label}: ${value}.`)} Opens the match-by-match chart."><span class="rank-label">${escape(label)}</span><span class="rank-value">${escape(value)}</span><span class="rank-go" aria-hidden="true">›</span></button>`;
    }
    const phrase = cap(kind === 'teams' ? m.rank_phrase : m.comparison_text) || '';
    const average = kind === 'teams' && Number.isFinite(m.league_average) ? ` League average ${m.unit === 'percent' ? Math.round(m.league_average) + '%' : one(m.league_average)}.` : '';
    const tip = `${label}: ${value}. ${phrase}.${average}`;
    return `<button class="rank" type="button" data-stat="${stat}" data-scope="${escape(scope)}" data-tip="${escape(tip)}" aria-label="${escape(tip)} Opens the match-by-match chart.">
      <span class="rank-label">${escape(label)}</span><span class="rank-value">${escape(value)}</span>
      <span class="track" aria-hidden="true"><span class="fill" style="width:${m.percentile}%"></span><span class="mid"></span><span class="dot" style="left:${m.percentile}%"></span></span>
      <span class="rank-go" aria-hidden="true">›</span><span class="rank-band">${escape(phrase)}</span></button>`;
  };
  const sections = order.map(name => {
    const [label, keys] = groups[name];
    const rows = keys.map(row).join('');
    return rows ? `${order.length > 1 ? `<div class="group-label">${label}</div>` : ''}<div class="ranks">${rows}</div>` : '';
  }).join('');
  if (!sections) return '';
  const sample = kind === 'players' ? `${whole(r.sample_minutes)} minutes so far` : `${whole(r.sample_matches)} matches so far`;
  const prev = d.previous_season;
  const prevLink = prev?.ranking?.ranked ? ` <button type="button" class="text-button rank-prev" data-scope="${escape(scopeOf(kind, prev))}">${escape(prev.season.label)} ranks ›</button>` : '';
  let status = '';
  if (r.provisional) status = `<p class="quiet-line"><span class="quiet">Early season · ${sample}</span> Rates move a lot with one match.</p>`;
  else if (!r.ranked) {
    const why = r.reason === 'below_minutes_threshold' ? `Ranked from ${whole(r.minimum_minutes)} minutes · ${whole(r.sample_minutes)} so far`
      : r.reason === 'below_matches_threshold' ? `Ranked from ${r.minimum_matches} matches · ${r.sample_matches} so far`
      : r.reason === 'unknown_position' ? 'No position recorded, so no fair group to rank against'
      : r.reason === 'too_few_peers' ? `Ranked once ${r.minimum_peers ?? 10} ${peerLabel || 'players in the group'} qualify${Number.isFinite(r.qualified_peers) ? ` · ${whole(r.qualified_peers)} so far` : ''}`
      : REBUILDING.includes(r.reason) ? `Ranks for ${d.season.label} are being rebuilt`
      : `No ranks for ${d.season.label} yet`;
    status = `<p class="quiet-line"><span class="quiet">${escape(why)}</span>${prevLink}</p>`;
  }
  const note = !ranked ? 'Select a row to see it match by match.'
    : kind === 'teams' ? 'Dots further right mean more. For goals and chances conceded, further left is better. Select a row to see it match by match.'
    : 'The dot shows where they sit; the tick in the middle is the typical one in the group. Select a row to see it match by match.';
  return `<section class="card" aria-labelledby="rankTitle"><div class="card-h"><h3 id="rankTitle">${escape(title)}</h3><small>${escape([d.season.label, group].filter(Boolean).join(' · '))}</small></div>${status}
    ${ranked ? '<div class="scale-key" aria-hidden="true"><span></span><span></span><span><span>Fewer</span><span>More</span></span><span></span></div>' : ''}${sections}
    <p class="card-note">${note}</p></section>`;
}

// The referee page leads with the season's cards a match, then the bettor's line over the last 10 matches.
export function refereeAnswer(d, {current, line}) {
  const cards = d.metrics?.cards, season = current ? 'this season' : `in ${d.season.label}`;
  const lead = cards?.value == null ? '' : `<mark>${one(cards.value)} cards</mark> a match ${season}${cards.comparison_text ? `, ${escape(cards.comparison_text)}` : ''}.`;
  const prev = d.previous_season?.metrics?.cards;
  const sub = ['All competitions', Number.isFinite(cards?.league_average) ? `league average ${one(cards.league_average)}` : '',
    `${whole(d.totals?.matches)} matches`, prev?.value != null ? `${d.previous_season.season.label}: ${one(prev.value)} a match` : ''].filter(Boolean).join(' · ');
  const last = (d.match_log || []).slice(-10);
  const {hits, of} = hitCount(last.map(m => m.values?.cards ?? null), line);
  const code = t => t?.short_code || t?.name?.slice(0, 3).toUpperCase() || '?';
  const items = last.map(m => ({value: m.values?.cards ?? null, top: `${code(m.home)}–\u200b${code(m.away)}`, bottom: shortDay(m.date),
    title: `${m.home?.name} v ${m.away?.name}, ${shortDay(m.date)}: ${m.values?.cards ?? 'not recorded'}${m.values?.cards == null ? '' : ' cards'}`}));
  return {lead, sub, count: `Over ${line} in <b>${hits} of ${of}</b>`,
    chart: labelledColumns(items, {line, label: `Cards shown in each of the last ${items.length} matches`, legend: `Line ${line} cards`, noun: 'Cards'})};
}
