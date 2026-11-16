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
