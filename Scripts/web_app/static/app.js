// Spix presentation and Matchday Board.

// ----- Utilities -----
function esc(str) {
  const d = document.createElement('div');
  d.textContent = str;
  // Double quotes are escaped too so the result is safe inside attributes.
  return d.innerHTML.replace(/"/g, '&quot;');
}

// ----- Pure render helpers (no DOM access; also loaded by Scripts/tests) -----

const ICONS = {
  plus: '<svg class="icon" width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M12 5v14M5 12h14" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
  check: '<svg class="icon" width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="m5 12.5 4.5 4.5L19 7.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  close: '<svg class="icon" width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
};
// Content for add-to-slip buttons: icon only on compact buttons, icon plus label on market buttons.
function addButtonContent(added, withLabel) {
  const icon = added ? ICONS.check : ICONS.plus;
  return withLabel
    ? `${icon}<span>${added ? 'Added' : 'Add'}</span>`
    : `${icon}<span class="sr-only">${added ? 'Remove from slip' : 'Add to slip'}</span>`;
}

function pluralise(count, singular, plural = `${singular}s`) {
  return `${count} ${count === 1 ? singular : plural}`;
}

// A Match Read's model figures arrive as structured card metrics, so every card
// draws the projected total and the home/draw/away balance the same way,
// whichever source wrote the briefing prose.
function figuresMarkup(metrics, home, away, className) {
  if (!metrics) return '';
  const parts = [];
  if (Number.isFinite(metrics.projectedTotal)) {
    const split = Number.isFinite(metrics.teamGoals.home) && Number.isFinite(metrics.teamGoals.away)
      ? `<span class="figures-split">${esc(home)} ${metrics.teamGoals.home.toFixed(2)} · ${esc(away)} ${metrics.teamGoals.away.toFixed(2)}</span>`
      : '';
    parts.push(`<div class="figures-total"><div><span class="figures-label">Projected total goals</span>${split}</div><mark class="figure-mark">${metrics.projectedTotal.toFixed(2)}</mark></div>`);
  }
  const balance = metrics.resultBalance;
  if (balance) {
    const pct = value => Math.round(value);
    parts.push(`<div class="figures-balance">
      <span class="sr-only">Result balance: ${esc(home)} ${pct(balance.home)}%, draw ${pct(balance.draw)}%, ${esc(away)} ${pct(balance.away)}%.</span>
      <span class="balance-track" aria-hidden="true"><i style="flex:${balance.home}"></i><i style="flex:${balance.draw}"></i><i style="flex:${balance.away}"></i></span>
      <span class="balance-legend" aria-hidden="true"><span>Home <b>${pct(balance.home)}%</b></span><span>Draw <b>${pct(balance.draw)}%</b></span><span>Away <b>${pct(balance.away)}%</b></span></span>
    </div>`);
  }
  return parts.length ? `<div class="${className}">${parts.join('')}</div>` : '';
}

// Fallback for servers that predate card metrics: recover the model figures
// from the deterministic briefing wording. Free-form prose is never guessed at.
function briefingFigures(bullets, summary) {
  const lines = [...(bullets || []), summary || ''].map(line => String(line || ''));
  const number = text => (text == null ? null : Number(text));
  let resultBalance = null;
  let projectedTotal = null;
  let teamGoals = { home: null, away: null };
  for (const line of lines) {
    const balance = line.match(/^Result balance:.*?(\d+(?:\.\d+)?)%,\s*draw\s+(\d+(?:\.\d+)?)%,.*?(\d+(?:\.\d+)?)%\.?$/i)
      || line.match(/(\d+(?:\.\d+)?)% win chance, with the draw at (\d+(?:\.\d+)?)% and .*? at (\d+(?:\.\d+)?)%/i);
    if (balance && !resultBalance) resultBalance = { home: number(balance[1]), draw: number(balance[2]), away: number(balance[3]) };
    const goals = line.match(/^Goals:\s*(\d+(?:\.\d+)?) projected — .*? (\d+(?:\.\d+)?), .*? (\d+(?:\.\d+)?)\.$/)
      || line.match(/projects (\d+(?:\.\d+)?) total goals, split (\d+(?:\.\d+)?) for .*? and (\d+(?:\.\d+)?) for /);
    if (goals && projectedTotal === null) {
      projectedTotal = number(goals[1]);
      teamGoals = { home: number(goals[2]), away: number(goals[3]) };
    }
  }
  return { projectedTotal, teamGoals, resultBalance };
}

// True when every figure in a briefing line is already drawn by figuresMarkup,
// so the line would only repeat it. Lines without figures are never redundant.
function restatesFigures(text, metrics) {
  if (!metrics) return false;
  const tokens = String(text || '').match(/\d+(?:\.\d+)?%?/g) || [];
  if (!tokens.length) return false;
  const goals = [metrics.projectedTotal, metrics.teamGoals.home, metrics.teamGoals.away].filter(Number.isFinite);
  const percents = metrics.resultBalance ? Object.values(metrics.resultBalance) : [];
  return tokens.every(token => {
    const value = parseFloat(token);
    return token.endsWith('%')
      ? percents.some(percent => Math.abs(percent - value) < 0.5)
      : goals.some(goal => Math.abs(goal - value) < 0.006);
  });
}

// Display names for a market group, shared by every surface that names a pick.
const MARKET_LABELS = { goals: 'Goals', btts: 'BTTS', corners: 'Corners', cards: 'Cards', sot: 'Shots on target', moneyline: 'Moneyline', spreads: 'Handicap' };
function marketLabel(group) {
  return MARKET_LABELS[group] || group || '';
}

// ----- Recommendation layer (recommendations.v1, design B "tiles", D63) -----
// The block is built by the backend; these helpers only draw it. Chances are the
// bookmakers' own view without their margin, so prices sit beside them as figures.
const RECS_MARKETS = { moneyline: 'Result', goals: 'Goals', btts: 'Both teams to score', spreads: 'Handicap', corners: 'Corners' };
const RECS_UNITS = { goals: ' goals', corners: ' corners' };
const RECS_TILE_MARKETS = new Set(['moneyline', 'btts']);
const CHANCE_FRACTIONS = [[1, 10], [1, 8], [1, 6], [1, 5], [1, 4], [1, 3], [2, 5], [1, 2], [3, 5], [2, 3], [3, 4], [4, 5], [5, 6], [9, 10]];

// A chance in words: the nearest plain fraction, or "1 in N" (and "N-1 in N") at the extremes.
function chanceWords(p) {
  const chance = Math.min(0.995, Math.max(0.005, Number(p) || 0));
  if (chance < 0.1) return `about 1 in ${Math.round(1 / chance)}`;
  if (chance > 0.9) {
    const n = Math.round(1 / (1 - chance));
    return `about ${n - 1} in ${n}`;
  }
  let best = CHANCE_FRACTIONS[0];
  for (const f of CHANCE_FRACTIONS) if (Math.abs(f[0] / f[1] - chance) < Math.abs(best[0] / best[1] - chance)) best = f;
  return `about ${best[0]} in ${best[1]}`;
}

function chancePercent(p) {
  return `${Math.min(99, Math.max(1, Math.round(Number(p) * 100)))}%`;
}

// A long shot says so once, before the chance.
function recsWords(option) {
  const longShot = typeof option.long_shot === 'boolean' ? option.long_shot : option.chance < 0.25;
  return (longShot ? 'A long shot, ' : '') + chanceWords(option.chance);
}

function recsPrice(value) {
  const price = Number(value);
  return value != null && Number.isFinite(price) ? price.toFixed(2) : '';
}

function recsMarketName(market) {
  return RECS_MARKETS[market] || marketLabel(market);
}

// The block's prices_as_of is UTC without an offset; it is shown in the viewer's time.
function recsClock(iso) {
  if (!iso) return '';
  const stamp = new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : `${iso}Z`);
  if (Number.isNaN(stamp.getTime())) return '';
  const pad = n => String(n).padStart(2, '0');
  return `${pad(stamp.getHours())}:${pad(stamp.getMinutes())}`;
}

// The block, unless absent or switched off with ?recs=off (to compare with the old page).
function recommendationsFor(card, search) {
  const block = card?.recommendations;
  if (!block || block.version !== 'recommendations.v1') return null;
  return new URLSearchParams(search || '').get('recs') === 'off' ? null : block;
}

function recsPriced(block) {
  return Boolean(block?.most_likely?.length) || (block?.markets || []).some(m => m.options?.length);
}

// Markets with a suggestion, plus value picks.
function recsSuggestionCount(block) {
  return (block?.markets || []).filter(m => m.suggestion).length + (block?.value?.length || 0);
}

// A value item names its side, line and label (D65). Older blocks, and a handicap that
// names neither team (side null), carry only the pick ("Over 10", "Away", "As Roma -1.5"):
// then side and line come from the market's own option when one matches, else the pick's words.
function recsValueLeg(item, block) {
  const unit = RECS_UNITS[item.market] || '';
  const figure = value => (value == null || !Number.isFinite(Number(value)) ? null : Number(value));
  const leg = (side, line, label) => ({
    market: item.market, side, line, label,
    chance: figure(item.chance), worth: figure(item.worth_it_at),
    whole: Boolean(unit) && Number.isInteger(line),
  });
  if (item.side != null) return leg(item.side, item.line ?? null, item.label || String(item.pick || ''));
  const pick = String(item.pick || '');
  const plain = value => String(value).toLowerCase().replace(/\u2212/g, '-').replace(/\s+/g, ' ').trim();
  const options = (block?.markets || []).find(m => m.market === item.market)?.options || [];
  const sideWord = ['home', 'away', 'draw', 'yes', 'no'].find(word => word === plain(pick));
  const known = options.find(o => plain(o.label) === plain(pick) || plain(o.label) === plain(pick + unit))
    || (sideWord ? options.find(o => o.side === sideWord) : null);
  const words = pick.match(/^(over|under)\s+(\d+(?:\.\d+)?)$/i);
  if (known) return leg(known.side, known.line, known.label);
  if (words) return leg(words[1].toLowerCase(), Number(words[2]), pick + unit);
  return leg(sideWord || null, null, pick);
}

// The stepper's lines. Handicap pairs each home line with the away side's opposite line.
function recsStepperPairs(market) {
  const options = market?.options || [];
  if (market?.market === 'spreads') {
    return options.filter(o => o.side === 'home').sort((a, b) => a.line - b.line).map(home => ({
      text: home.label,
      options: [home, options.find(o => o.side === 'away' && o.line === -home.line)].filter(Boolean),
    }));
  }
  const unit = RECS_UNITS[market?.market] || '';
  return [...new Set(options.map(o => o.line).filter(line => line != null))].sort((a, b) => a - b).map(line => ({
    text: `${line}${unit}`,
    options: ['over', 'under'].map(side => options.find(o => o.side === side && o.line === line)).filter(Boolean),
  }));
}

function recsIsSuggestion(market, option) {
  const s = market?.suggestion;
  return Boolean(s) && option.side === s.side && option.line === s.line;
}

// The line shown: the one asked for, else the suggestion's, else the middle one.
function recsLineIndex(market, asked) {
  const pairs = recsStepperPairs(market);
  if (!pairs.length) return 0;
  if (Number.isInteger(asked)) return Math.min(Math.max(asked, 0), pairs.length - 1);
  const at = pairs.findIndex(pair => pair.options.some(o => recsIsSuggestion(market, o)));
  return at >= 0 ? at : Math.floor(pairs.length / 2);
}

function recsAddButton(ref, leg, added) {
  return `<button type="button" class="recs-add${added ? ' added' : ''}" data-recs-add="${esc(ref)}" aria-label="${added ? 'Remove' : 'Add'} ${esc(leg.label)} ${added ? 'from' : 'to'} your bet">${addButtonContent(added, true)}</button>`;
}

// Most likely and Value: the top of the match page.
function renderRecommendations(f, view = {}) {
  const block = f.recommendations;
  if (!block) return '';
  const added = view.added || (() => false);
  const priced = recsPriced(block);
  const clock = priced ? recsClock(block.prices_as_of) : '';
  const cards = (block.most_likely || []).map((o, i) => `
    <article class="recs-card">
      <span class="recs-label">${esc(recsMarketName(o.market))}</span>
      <h3 class="recs-pick">${esc(o.label)}</h3>
      <p class="recs-chance"><span class="recs-words">${esc(recsWords(o))}</span><span class="recs-pct">${chancePercent(o.chance)}</span></p>
      ${o.why ? `<p class="recs-why">${esc(o.why)}</p>` : '<p class="recs-why"></p>'}
      <div class="recs-card-foot"><span class="recs-fig">${recsPrice(o.typical_price)}</span>${recsAddButton(`ml:${i}`, o, added(o))}</div>
    </article>`).join('');
  const mostLikely = `
    <section class="recs-section" aria-labelledby="recsMostLikely">
      <div class="recs-head">
        <h2 class="recs-title" id="recsMostLikely">Most likely</h2>
        ${clock ? `<span class="recs-asof">Prices as of ${clock}</span>` : ''}
      </div>
      ${priced
        ? `<div class="recs-strip">${cards}</div>`
        : '<p class="recs-note">The bookmakers haven\'t priced this match yet, so there are no suggestions. They usually appear a day or two before kick-off.</p>'}
    </section>`;
  if (!priced) return mostLikely;
  const values = (block.value || []).map((item, i) => {
    const leg = recsValueLeg(item, block);
    return `
      <article class="recs-value-card">
        <div class="recs-value-main">
          <span class="recs-label">${esc(recsMarketName(leg.market))}</span>
          <h3 class="recs-pick">${esc(leg.label)}</h3>
          ${leg.chance != null ? `<p class="recs-value-chance">${esc(recsWords(leg))} · ${chancePercent(leg.chance)}</p>` : ''}
          ${leg.whole ? `<p class="recs-value-chance">Exactly ${esc(leg.line)}${esc(RECS_UNITS[leg.market])}: you get your stake back.</p>` : ''}
        </div>
        ${leg.worth != null ? `<div class="recs-value-worth"><span>Worth it at</span><span class="recs-fig">${recsPrice(leg.worth)} or better</span></div>` : ''}
        ${recsAddButton(`value:${i}`, leg, added(leg))}
      </article>`;
  }).join('');
  return `${mostLikely}
    <section class="recs-section" aria-labelledby="recsValue">
      <h2 class="recs-title recs-head" id="recsValue">Value</h2>
      ${values || '<p class="recs-note">No value picks on this match today. The prices are about right.</p>'}
    </section>`;
}

// "Explore the match": every market, Spix's suggestion marked, Add on any option.
function renderMarketExplorer(f, view = {}) {
  const block = f.recommendations;
  const markets = block?.markets || [];
  const open = markets.find(m => m.market === view.open) || markets[0];
  if (!open) return '';
  const added = view.added || (() => false);
  const tabs = markets.map(m => `<button type="button" class="recs-tab${m === open ? ' is-on' : ''}" data-recs-tab="${esc(m.market)}" aria-pressed="${m === open}">${esc(recsMarketName(m.market))}</button>`).join('');
  let body;
  if (!open.options?.length) {
    body = '<p class="recs-note"><b>Nothing stands out here.</b> No bookmaker prices this market yet.</p>';
  } else {
    const suggested = open.suggestion ? open.options.find(o => recsIsSuggestion(open, o)) || open.suggestion : null;
    const summary = suggested
      ? `Spix's pick here: ${suggested.label}, ${chanceWords(suggested.chance)}.`
      : 'Nothing stands out here. Every option can still be added.';
    let shown = open.options;
    let stepper = '';
    if (!RECS_TILE_MARKETS.has(open.market)) {
      const pairs = recsStepperPairs(open);
      const index = recsLineIndex(open, view.line?.[open.market]);
      shown = pairs[index]?.options || [];
      stepper = `
        <div class="recs-stepper">
          <button type="button" class="recs-step" data-recs-step="-1" aria-label="Lower line"${index <= 0 ? ' disabled' : ''}>−</button>
          <div class="recs-line" aria-live="polite"><span class="recs-line-text">${esc(pairs[index]?.text || '')}</span><span class="recs-line-pos">Line ${index + 1} of ${pairs.length}</span></div>
          <button type="button" class="recs-step" data-recs-step="1" aria-label="Higher line"${index >= pairs.length - 1 ? ' disabled' : ''}>+</button>
        </div>`;
    }
    const tiles = shown.map(o => {
      const pick = recsIsSuggestion(open, o);
      return `
        <div class="recs-tile${pick ? ' is-pick' : ''}">
          ${pick ? '<span class="recs-pickmark">Spix\'s pick</span>' : ''}
          <span class="recs-option-label">${esc(o.label)}</span>
          <span class="recs-fig">${recsPrice(o.typical_price)}</span>
          <span class="recs-option-chance">${esc(recsWords(o))} · ${chancePercent(o.chance)}</span>
          ${recsAddButton(`opt:${open.market}:${open.options.indexOf(o)}`, o, added(o))}
        </div>`;
    }).join('');
    body = `<p class="recs-note recs-summary">${esc(summary)}</p><div class="recs-swipe" data-recs-swipe>${stepper}<div class="recs-grid">${tiles}</div></div>`;
  }
  return `
    <section class="recs-section recs-explore" aria-labelledby="recsExplore">
      <h2 class="recs-title" id="recsExplore">Explore the match</h2>
      <div class="recs-tabs" role="group" aria-label="Markets">${tabs}</div>
      <div class="recs-market">${body}</div>
    </section>`;
}

// ----- Your bet: chances of legs together -----
const GRID_MARKETS = new Set(['moneyline', 'goals', 'btts', 'spreads']);

// Does a goal-market leg land at this score? A push (a whole line hit exactly) does not land.
function legLands(leg, home, away) {
  const { side, line } = leg;
  if (leg.market === 'moneyline') return side === 'home' ? home > away : side === 'away' ? away > home : home === away;
  if (leg.market === 'goals') return side === 'over' ? home + away > line : home + away < line;
  if (leg.market === 'btts') return side === 'yes' ? home > 0 && away > 0 : !(home > 0 && away > 0);
  if (leg.market === 'spreads') return side === 'home' ? home + line > away : away + line > home;
  return false;
}

// The chance every leg lands, from the score grid: independent Poisson goals for each
// team, 0-12, renormalised (as market_goal_model.score_probabilities). Null when a leg
// is outside the grid's markets or there is no grid.
function jointChance(legs, grid) {
  if (!grid || !Number.isFinite(grid.home) || !Number.isFinite(grid.away)) return null;
  if (!legs.length || legs.some(leg => !GRID_MARKETS.has(leg.market) || !leg.side)) return null;
  const poisson = (rate, goals) => {
    let value = Math.exp(-rate);
    for (let i = 1; i <= goals; i++) value *= rate / i;
    return value;
  };
  let total = 0, landed = 0;
  for (let home = 0; home <= 12; home++) {
    for (let away = 0; away <= 12; away++) {
      const weight = poisson(grid.home, home) * poisson(grid.away, away);
      total += weight;
      if (legs.every(leg => legLands(leg, home, away))) landed += weight;
    }
  }
  return landed / total;
}

// Why a match's legs can't go together, or null when they can.
function slipRefusal(legs, grid) {
  const corners = legs.filter(leg => leg.market === 'corners');
  const goals = legs.filter(leg => leg.market !== 'corners');
  if (corners.length && goals.length) return "Corners and goals from one match can't be combined yet.";
  if (goals.length < 2) return null;
  if (!grid) return 'Combining bets from one match needs prices for both teams.';
  const chance = jointChance(goals, grid);
  if (chance === null) return "Bets from one match can't be combined here yet.";
  if (chance < 1e-6) return legs.length === 2 ? "These two can't both win." : "These can't all win together.";
  return null;
}

// The chance the whole slip lands: one match's legs through its grid, matches multiplied.
// Null when any leg's chance is unknown.
function slipChance(slipLegs) {
  if (!slipLegs.length) return null;
  const matches = new Map();
  slipLegs.forEach(l => {
    const id = String(l.fixtureId || l.fixture);
    matches.set(id, [...(matches.get(id) || []), l]);
  });
  let chance = 1;
  for (const group of matches.values()) {
    const part = group.length === 1
      ? group[0].leg?.chance
      : jointChance(group.map(l => l.leg), group.find(l => l.grid)?.grid);
    if (part == null || !Number.isFinite(Number(part))) return null;
    chance *= Number(part);
  }
  return chance;
}

// The board card: the first Most likely pick, then the match and how much there is to see.
function renderRecsBoardCard(f, homeEmblem, awayEmblem) {
  const block = f.recommendations;
  const first = block?.most_likely?.[0];
  const time = `<span class="match-card-time">${esc(f.time)}</span>`;
  let top;
  if (first) {
    top = `
      <div class="recs-board-topline"><span class="recs-label">Most likely</span>${time}</div>
      <p class="recs-board-pick">${esc(first.label)}</p>
      <p class="recs-board-chance"><span><b>${esc(recsWords(first))}</b> <span class="recs-pct">· ${chancePercent(first.chance)}</span></span><span class="recs-fig">${recsPrice(first.typical_price)}</span></p>`;
  } else if (recsPriced(block)) {
    top = `<div class="recs-board-topline"><span class="recs-label">Explore the match</span>${time}</div>
      <p class="recs-note">Nothing stands out here. Every option can still be added.</p>`;
  } else {
    top = `<div class="recs-board-topline"><span class="recs-label">Not priced yet</span>${time}</div>
      <p class="recs-note">Suggestions come once the bookmakers price this match.</p>`;
  }
  const values = block?.value?.length || 0;
  const counts = recsPriced(block)
    ? [pluralise(recsSuggestionCount(block), 'suggestion'), values ? pluralise(values, 'value pick') : ''].filter(Boolean).join(' · ')
    : '';
  return `
    <div class="recs-board-body">${top}</div>
    <div class="recs-board-foot">
      <h3 class="recs-board-teams">${homeEmblem}<span>${esc(f.home)}</span> <span class="recs-board-v">v</span> ${awayEmblem}<span>${esc(f.away)}</span></h3>
      <div class="recs-board-row">
        <span class="recs-board-count">${counts}</span>
        <button class="match-card-open" type="button" aria-label="View match: ${esc(f.home)} versus ${esc(f.away)}">View match <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M5 12h14m-6-6 6 6-6 6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg></button>
      </div>
    </div>`;
}

// ----- Before kickoff (D71): a started match keeps the picks Spix had for it, with nothing to bet with. -----
const LIVE_STATUS_LABELS = {
  '1H': 'Live', '2H': 'Live', ET: 'Live', BT: 'Live', P: 'Live', INT: 'Live', LIVE: 'Live',
  HT: 'Half time', FT: 'Full time', AET: 'Full time', PEN: 'Full time',
  PST: 'Postponed', SUSP: 'Suspended', ABD: 'Abandoned', CANC: 'Cancelled', AWD: 'Match awarded', WO: 'Walkover',
  NS: 'Not started yet', TBD: 'Not started yet',
};

// A status the API could not give, or one we do not know, says only when the match kicked off.
function liveStatusLabel(status, time) {
  return LIVE_STATUS_LABELS[String(status?.short || '').toUpperCase()] || `Kicked off ${time}`;
}

// The entry, unless absent, of another version, without picks or switched off with ?recs=off. Labels and
// chances only: anything else an entry carries (a price) never reaches the card.
function beforeKickoffFor(entry, search, league) {
  if (!entry || entry.version !== 'before-kickoff.v1' || entry.fixture?.event_id == null) return null;
  if (new URLSearchParams(search || '').get('recs') === 'off') return null;
  const picks = (list, most) => (Array.isArray(list) ? list : [])
    .filter(item => item?.label && item.chance != null && Number.isFinite(Number(item.chance)))
    .slice(0, most)
    .map(item => ({ label: String(item.label), chance: Number(item.chance) }));
  const mostLikely = picks(entry.most_likely, 3);
  const value = picks(entry.value, 2);
  if (!mostLikely.length && !value.length) return null;
  return { id: String(entry.fixture.event_id), league: entry.fixture.league || league,
    readAt: entry.read_at || null, status: entry.status || null, most_likely: mostLikely, value };
}

function renderBeforeKickoffCard(f, homeEmblem, awayEmblem) {
  const block = f.beforeKickoff;
  const row = item => `<li><span class="bk-pick">${esc(item.label)}</span><span class="bk-chance">${esc(chanceWords(item.chance))} · ${chancePercent(item.chance)}</span></li>`;
  const list = (title, items) => (items.length
    ? `<div class="bk-section"><span class="recs-label">${title}</span><ul class="bk-list">${items.map(row).join('')}</ul></div>` : '');
  const checked = block ? recsClock(block.readAt) : '';
  const body = block
    ? `${list('Most likely, before kickoff', block.most_likely)}${list('Value, before kickoff', block.value)}
    ${checked ? `<p class="bk-foot">Last checked at ${checked}, before kickoff.</p>` : ''}`
    : `<p class="bk-none">${esc(f.scheduleNote)}</p>`;
  return `
    <div class="bk-top"><span class="bk-time">${esc(f.time)}</span><span class="bk-status">${esc(f.scheduleLabel)}</span></div>
    <h3 class="bk-teams">${homeEmblem}<span>${esc(f.home)}</span> <span class="bk-v">v</span> ${awayEmblem}<span>${esc(f.away)}</span></h3>
    ${body}`;
}

// Upcoming matches first, then the ones that have started (D71, the owner's choice B).
function splitStartedFixtures(fixtures) {
  return { upcoming: fixtures.filter(f => !f.afterKickoff), started: fixtures.filter(f => f.afterKickoff) };
}

// ----- End of pure render helpers -----

function showToast(msg, type = 'info') {
  const container = document.getElementById('toastContainer');
  const toast = document.createElement('div');
  toast.className = 'toast ' + type;
  toast.textContent = msg;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 2700);
}

function toggleFaq(btn) {
  const expanded = btn.getAttribute('aria-expanded') === 'true';
  btn.setAttribute('aria-expanded', String(!expanded));
  const answer = btn.nextElementSibling;
  answer.classList.toggle('open', !expanded);
}

// ----- Presentation: motion, focus and navigation (no prediction logic) -----
const ui = {
  reducedMotion: window.matchMedia('(prefers-reduced-motion: reduce)'),
  animations: new WeakMap(),
  modal: null,
  focusable(root) {
    return [...root.querySelectorAll('button, a[href], input, [tabindex="0"]')]
      .filter(el => !el.disabled && !el.closest('[inert]') && el.getClientRects().length);
  },
  enter(el, direction = 1) {
    const running = this.animations.get(el);
    const live = running ? getComputedStyle(el) : null;
    const from = live ? { opacity: live.opacity, transform: live.transform }
      : { opacity: 0, transform: `translateY(${direction * 8}px)` };
    running?.cancel();
    if (this.reducedMotion.matches || !el.animate) return;
    const animation = el.animate([from, { opacity: 1, transform: 'translateY(0)' }],
      { duration: 230, easing: 'cubic-bezier(.2,.8,.2,1)' });
    this.animations.set(el, animation);
    animation.onfinish = () => this.animations.delete(el);
  },
  // Exact critically damped spring. Retargeting preserves position AND velocity.
  spring(draw, settled) {
    const state = { value: 0, velocity: 0, target: 0, frame: 0, last: 0 };
    const tick = time => {
      const dt = Math.min((time - state.last) / 1000, 0.05);
      state.last = time;
      const offset = state.value - state.target;
      const coefficient = state.velocity + 24 * offset;
      const decay = Math.exp(-24 * dt);
      state.value = state.target + (offset + coefficient * dt) * decay;
      state.velocity = (state.velocity - 24 * coefficient * dt) * decay;
      if (Math.abs(state.value - state.target) < .001 && Math.abs(state.velocity) < .01) {
        state.value = state.target; state.velocity = 0; state.frame = 0;
        draw(state.value); settled?.(state.target); return;
      }
      draw(state.value);
      state.frame = requestAnimationFrame(tick);
    };
    state.set = target => {
      state.target = target;
      if (this.reducedMotion.matches) {
        cancelAnimationFrame(state.frame); state.frame = 0;
        state.value = target; state.velocity = 0;
        draw(target); settled?.(target);
      } else if (!state.frame) {
        state.last = performance.now(); state.frame = requestAnimationFrame(tick);
      }
    };
    this.reducedMotion.addEventListener('change', () => state.set(state.target));
    return state;
  },
  overlay(id, backdropId, open, close) {
    const el = document.getElementById(id), backdrop = document.getElementById(backdropId);
    if (!el._motion) {
      const draw = value => {
        backdrop.style.opacity = value;
        if (id === 'betSlip') {
          const axis = window.matchMedia('(max-width: 1100px)').matches ? 'Y' : 'X';
          el.style.transform = `translate${axis}(${(1 - value) * 100}%)`;
        } else if (id === 'mobileMenu') {
          // The panel slides in from the right and pushes the page by its own
          // width; the backdrop dims what remains visible.
          el.style.transform = `translateX(${(1 - value) * 100}%)`;
          document.documentElement.style.setProperty('--menu-shift', `${-value * el.offsetWidth}px`);
          document.body.classList.toggle('menu-shifted', value > 0);
        } else {
          el.style.opacity = value;
          el.style.transform = `translate(-50%, -50%) scale(${.97 + .03 * value})`;
        }
      };
      draw(0);
      el._motion = this.spring(draw, target => {
        if (!target) { el.classList.add('hidden'); backdrop.classList.add('hidden'); }
      });
      window.addEventListener('resize', () => draw(el._motion.value));
    }
    if (open) {
      if (this.modal?.el === el) return;
      this.modal?.close();
      const trigger = document.activeElement;
      el.classList.remove('hidden'); backdrop.classList.remove('hidden');
      el.inert = false;
      // Inert only the background branches; keep the dialog and its scrim usable.
      const disabled = [];
      let branch = el;
      while (branch.parentElement && branch !== document.body) {
        [...branch.parentElement.children].forEach(sibling => {
          if (sibling !== branch && sibling !== backdrop && !sibling.contains(backdrop)
              && !sibling.inert && !['SCRIPT', 'STYLE'].includes(sibling.tagName)) {
            sibling.inert = true; disabled.push(sibling);
          }
        });
        branch = branch.parentElement;
      }
      this.modal = { el, close, trigger, disabled };
      document.body.classList.add('overlay-open');
      (el.querySelector('[autofocus]') || this.focusable(el)[0] || el).focus({ preventScroll: true });
    } else {
      el.inert = true;
      if (this.modal?.el === el) {
        const { trigger, disabled } = this.modal;
        disabled.forEach(node => { node.inert = false; });
        this.modal = null;
        document.body.classList.remove('overlay-open');
        if (trigger?.isConnected && trigger.getClientRects().length) trigger.focus({ preventScroll: true });
      }
    }
    el._motion.set(open ? 1 : 0);
  },
  scrollTo(id) {
    document.getElementById(id)?.scrollIntoView({ behavior: this.reducedMotion.matches ? 'instant' : 'smooth' });
  },
};

document.addEventListener('keydown', event => {
  if (!ui.modal) return;
  if (event.key === 'Escape') { event.preventDefault(); ui.modal.close(); return; }
  if (event.key !== 'Tab') return;
  const items = ui.focusable(ui.modal.el), first = items[0], last = items[items.length - 1];
  if (!first) { event.preventDefault(); return; }
  if (event.shiftKey && (document.activeElement === first || !ui.modal.el.contains(document.activeElement))) {
    event.preventDefault(); last.focus();
  } else if (!event.shiftKey && (document.activeElement === last || !ui.modal.el.contains(document.activeElement))) {
    event.preventDefault(); first.focus();
  }
});

const router = {
  state: null,
  positions: new Map(),
  focusTargets: new Map(),
  sequence: 0,
  key(state) { return `${state.page}:${state.tab}:${state.matchId || ''}:${state.researchRoute || ''}`; },
  remember() {
    if (!this.state) return;
    this.state.scrollY = window.scrollY;
    this.positions.set(this.key(this.state), window.scrollY);
    history.replaceState(this.state, '');
  },
  go(page, tab = 'fixtures', restored = null, researchRoute = null) {
    const previous = this.state;
    if (!restored) {
      this.remember();
      if (previous) this.focusTargets.set(this.key(previous), document.activeElement);
    }
    const next = restored || { spix: true, page, tab,
      matchId: tab === 'match' ? appModule.currentMatch?.id : null,
      researchRoute: tab === 'research' ? researchRoute || appModule.researchRoute || 'players' : null };
    if (previous && this.key(previous) === this.key(next) && !restored) return;
    ui.modal?.close();
    this.state = next;
    const token = ++this.sequence;
    document.body.dataset.page = page;
    document.getElementById('landingPage').classList.toggle('hidden', page !== 'landing');
    document.getElementById('appPage').classList.toggle('hidden', page !== 'app');
    document.querySelectorAll('.landing-nav').forEach(el => el.classList.toggle('hidden', page !== 'landing'));
    document.querySelectorAll('.app-nav').forEach(el => el.classList.toggle('hidden', page !== 'app'));
    auth._updateUI();
    const ready = page === 'app' ? (tab === 'research'
      ? customElements.whenDefined('spix-research').then(() => document.querySelector('spix-research').showRoute(next.researchRoute || 'players'))
      : appModule.init()) : Promise.resolve();
    if (tab === 'research') appModule.researchRoute = next.researchRoute || 'players';
    appModule.currentTab = tab;
    const viewId = { fixtures: 'viewFixtures', 'best-bets': 'viewBestBets', research: 'viewResearch', match: 'viewMatch' }[tab];
    document.querySelectorAll('.app-view').forEach(el => el.classList.toggle('hidden', el.id !== viewId));
    document.querySelectorAll('[data-view]').forEach(el => {
      const active = el.dataset.view === tab;
      el.classList.toggle('active', active);
      if (active) el.setAttribute('aria-current', 'page'); else el.removeAttribute('aria-current');
    });
    const url = page === 'app' ? '/app' + (tab === 'best-bets' ? '#best-bets' : tab === 'research' ? '#research/' + next.researchRoute : '') : '/';
    if (!restored) history[previous ? 'pushState' : 'replaceState'](next, '', url + (previous ? '' : location.search));
    const view = document.getElementById(page === 'app' ? viewId : 'landingPage');
    ui.enter(view, restored ? -1 : 1);
    if (previous) window.spixLogo?.drop(); // D48: the ball drops on a page change, never on first load
    const top = restored?.scrollY ?? this.positions.get(this.key(next)) ?? 0;
    window.scrollTo({ top, behavior: 'instant' });
    const focus = this.focusTargets.get(this.key(next));
    const heading = view.querySelector('h1, .match-teams-title, .page-title');
    if (previous) {
      const target = focus?.isConnected && focus.getClientRects().length ? focus : heading;
      if (target) { if (target === heading) target.tabIndex = -1; target.focus({ preventScroll: true }); }
    }
    ready.then(() => {
      if (token === this.sequence && Math.abs(window.scrollY - top) < 2) {
        window.scrollTo({ top, behavior: 'instant' });
      }
    }).catch(error => console.warn('Page content could not load', error));
  },
  backToBoard() {
    if (this.state?.tab === 'match' && this.state.fromBoard) history.back();
    else this.go('app', 'fixtures');
  },
};
history.scrollRestoration = 'manual';
let scrollRecordTimer;
window.addEventListener('scroll', () => {
  if (router.state) router.state.scrollY = window.scrollY;
  clearTimeout(scrollRecordTimer);
  scrollRecordTimer = setTimeout(() => router.remember(), 150);
}, { passive: true });
window.addEventListener('popstate', event => {
  const state = event.state;
  if (!state?.spix) return;
  if (state.tab === 'match' && appModule.fixtureIndex[state.matchId]
      && !appModule.fixtureIndex[state.matchId].isScheduleOnly) {
    // Render detail before restoring history, without adding a second entry.
    appModule._restoringRoute = state;
    appModule._showMatch(state.matchId);
  } else router.go(state.page, state.tab === 'match' ? 'fixtures' : state.tab,
    state.tab === 'match' ? { ...state, tab: 'fixtures', matchId: null } : state);
});

// Paid membership is not offered until the single-plan price and terms are configured.

// ----- Auth -----
const auth = {
  _token: null,
  _user: null,

  async init() {
    // 1. Check for one-time auth code in URL (from Discord OAuth redirect)
    const params = new URLSearchParams(window.location.search);
    const code = params.get('code');
    if (code) {
      // Clean URL immediately so code doesn't linger in browser history
      window.history.replaceState({}, '', window.location.pathname);
      try {
        const resp = await fetch('/auth/exchange', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ code }),
        });
        if (resp.ok) {
          const data = await resp.json();
          this._token = data.token;
          localStorage.setItem('jwt', data.token);
        }
      } catch (e) {
        console.error('Auth code exchange failed:', e);
      }
    }

    // 2. Check localStorage for existing token
    if (!this._token) {
      this._token = localStorage.getItem('jwt');
    }

    // 3. Validate token by fetching user profile
    if (this._token) {
      try {
        const resp = await fetch('/auth/me', {
          headers: { 'Authorization': 'Bearer ' + this._token },
        });
        if (resp.ok) {
          this._user = await resp.json();
          this._updateUI();
          // If we came from OAuth, go to app view
          if (code) router.go('app');
          return;
        }
      } catch (e) {
        console.error('Auth check failed:', e);
      }
      // Token invalid — clear it
      this._token = null;
      this._user = null;
      localStorage.removeItem('jwt');
    }
  },

  _updateUI() {
    const loginBtn = document.getElementById('loginBtn');
    const badge = document.getElementById('userBadge');
    const avatar = document.getElementById('userAvatar');
    const name = document.getElementById('userName');
    const onLanding = document.body.dataset.page !== 'app';
    if (loginBtn) loginBtn.classList.toggle('hidden', !onLanding || Boolean(this._user));
    if (badge) badge.classList.toggle('hidden', !onLanding || !this._user);
    if (!this._user) return;
    if (avatar) {
      if (this._user.avatar_url) {
        avatar.src = this._user.avatar_url;
        avatar.hidden = false;
      } else {
        // Email users — hide avatar img, the name is enough
        avatar.hidden = true;
      }
    }
    if (name) name.textContent = this._user.username || this._user.email || '';
  },

  isLoggedIn() { return !!this._user; },
  getToken() { return this._token; },
  getTier() { return this._user ? this._user.tier : 'free'; },

  logout() {
    this._token = null;
    this._user = null;
    localStorage.removeItem('jwt');
    const loginBtn = document.getElementById('loginBtn');
    const badge = document.getElementById('userBadge');
    if (loginBtn) loginBtn.classList.remove('hidden');
    if (badge) badge.classList.add('hidden');
    router.go('landing');
    showToast('Logged out', 'info');
  },

  showMenu() {
    if (!this._user) return;
    if (confirm('Log out of Spix?')) this.logout();
  },

  requireLogin() {
    if (this.isLoggedIn()) return true;
    this.showLoginModal();
    return false;
  },

  // ----- Login Modal -----
  _isSignup: false,

  showLoginModal() {
    this._isSignup = false;
    this._renderModalState();
    document.getElementById('authError').classList.add('hidden');
    ui.overlay('authModal', 'authBackdrop', true, () => this.hideLoginModal());
  },

  hideLoginModal() {
    ui.overlay('authModal', 'authBackdrop', false, () => this.hideLoginModal());
    document.getElementById('authError').classList.add('hidden');
    document.getElementById('authForm').reset();
  },

  toggleMode() {
    this._isSignup = !this._isSignup;
    document.getElementById('authError').classList.add('hidden');
    this._renderModalState();
  },

  _renderModalState() {
    const title = document.getElementById('authModalTitle');
    const btn = document.getElementById('authSubmitBtn');
    const toggle = document.getElementById('authToggle');
    const pw = document.getElementById('authPassword');
    if (this._isSignup) {
      title.textContent = 'Create your account';
      btn.textContent = 'Sign Up';
      toggle.innerHTML = 'Already have an account? <a href="#" onclick="event.preventDefault(); auth.toggleMode()">Log in</a>';
      pw.setAttribute('autocomplete', 'new-password');
      pw.setAttribute('minlength', '8');
    } else {
      title.textContent = 'Log in to Spix';
      btn.textContent = 'Log In';
      toggle.innerHTML = 'Don\'t have an account? <a href="#" onclick="event.preventDefault(); auth.toggleMode()">Sign up</a>';
      pw.setAttribute('autocomplete', 'current-password');
      pw.removeAttribute('minlength');
    }
  },

  async submitEmailForm() {
    const email = document.getElementById('authEmail').value.trim();
    const password = document.getElementById('authPassword').value;
    const errorEl = document.getElementById('authError');
    const btn = document.getElementById('authSubmitBtn');

    if (!email || !password) return;

    btn.disabled = true;
    btn.textContent = this._isSignup ? 'Creating account...' : 'Logging in...';
    errorEl.classList.add('hidden');

    try {
      const endpoint = this._isSignup ? '/auth/signup' : '/auth/login';
      const resp = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });

      const data = await resp.json();

      if (!resp.ok) {
        errorEl.textContent = data.detail || 'Something went wrong';
        errorEl.classList.remove('hidden');
        btn.disabled = false;
        this._renderModalState();
        return;
      }

      // Success — store token and load user
      this._token = data.token;
      localStorage.setItem('jwt', data.token);

      const meResp = await fetch('/auth/me', {
        headers: { 'Authorization': 'Bearer ' + this._token },
      });
      if (meResp.ok) {
        this._user = await meResp.json();
        this._updateUI();
      }

      this.hideLoginModal();
      showToast(this._isSignup ? 'Account created!' : 'Welcome back!', 'success');
    } catch (e) {
      errorEl.textContent = 'Connection error — please try again';
      errorEl.classList.remove('hidden');
    }

    btn.disabled = false;
    this._renderModalState();
  }
};

// Run auth init on page load
auth.init();

// ----- Bet Slip -----
const slip = {
  legs: [],
  open: false,
  onChange: null,

  toggle() {
    this.open = !this.open;
    const el = document.getElementById('betSlip');
    el.classList.toggle('open', this.open);
    document.querySelectorAll('[aria-controls="betSlip"]').forEach(button => {
      button.setAttribute('aria-expanded', String(this.open));
    });
    ui.overlay('betSlip', 'slipBackdrop', this.open, () => { if (this.open) this.toggle(); });
  },

  add(fixture, pick, odds, market, options = {}) {
    const key = fixture + '|' + pick;
    if (this.legs.find(l => l.key === key)) { this.remove(key); return false; }
    const fixtureId = String(options.fixtureId || fixture);
    const leg = options.leg || null;
    const sameFixture = this.legs.filter(l => String(l.fixtureId || l.fixture) === fixtureId);
    // Legs from the older flows carry no leg: one selection per fixture, as before.
    if (sameFixture.length && (!leg || sameFixture.some(l => !l.leg))) {
      showToast(
        'This slip accepts one selection per fixture. Same-game combined odds need a verified bookmaker quote.',
        'error',
      );
      return null;
    }
    // One leg per market per match: another line of the same market replaces the first.
    const replaced = leg ? sameFixture.filter(l => l.leg.market === leg.market) : [];
    if (leg) {
      const kept = sameFixture.filter(l => !replaced.includes(l));
      const grid = options.grid || kept.find(l => l.grid)?.grid || null;
      const refusal = slipRefusal(kept.map(l => l.leg).concat([leg]), grid);
      if (refusal) {
        this._notice(refusal);
        return null;
      }
    }
    this.legs = this.legs.filter(l => !replaced.includes(l));
    this.legs.push({ key, fixture, fixtureId, pick, odds, market, leg, grid: options.grid || null });
    this.notice = '';
    this._render();
    this._update();
    if (!this.open) this.toggle();
    return true;
  },

  _notice(text) {
    this.notice = text;
    this._render();
    if (!this.open) this.toggle();
  },

  remove(key) {
    this.legs = this.legs.filter(l => l.key !== key);
    this.notice = '';
    this._render();
    this._update();
    // sync add buttons
    document.querySelectorAll('[data-slip-key]').forEach(btn => {
      if (btn.dataset.slipKey === key) btn.classList.remove('added');
    });
  },

  clear() {
    this.legs = [];
    this.notice = '';
    this._render();
    this._update();
    document.querySelectorAll('.fixture-add-btn.added, .market-add-btn.added').forEach(b => b.classList.remove('added'));
  },

  setStake(v) {
    document.getElementById('slipStake').value = v;
    this._calcPayout();
  },

  _render() {
    const legsEl = document.getElementById('slipLegs');
    const notice = document.getElementById('slipNotice');
    notice.textContent = this.notice || '';
    notice.classList.toggle('hidden', !this.notice);
    if (!this.legs.length) {
      legsEl.innerHTML = '<div class="slip-empty">Tap Add on any option to start your bet. You can mix Spix\'s picks with your own.</div>';
      document.getElementById('slipQuality').classList.add('hidden');
      return;
    }
    legsEl.innerHTML = this.legs.map(l => {
      const price = !l.leg ? esc(String(l.odds))
        : l.odds != null ? recsPrice(l.odds)
        : l.leg.worth != null ? `worth it at ${recsPrice(l.leg.worth)}+` : '';
      return `
      <div class="slip-leg">
        <div class="slip-leg-fixture">${esc(l.fixture)}</div>
        <div class="slip-leg-pick">${esc(l.pick)}</div>
        <div class="slip-leg-odds mono">${price}</div>
        <button class="slip-leg-remove" aria-label="Remove ${esc(l.pick)} from slip" onclick="slip.remove('${esc(l.key).replace(/'/g,"\\'")}')">${ICONS.close}</button>
      </div>
    `;
    }).join('');
  },

  // The prices multiply only when each match has one priced leg: legs from one
  // match are not independent, and no total price is shown for them (D63).
  _totalPrice() {
    if (!this.legs.length) return null;
    const matches = new Set(this.legs.map(l => String(l.fixtureId || l.fixture)));
    if (matches.size !== this.legs.length) return null;
    const prices = this.legs.map(l => (l.odds == null ? NaN : parseFloat(l.odds)));
    return prices.every(Number.isFinite) ? prices.reduce((acc, price) => acc * price, 1) : null;
  },

  _update() {
    if (this.onChange) this.onChange();
    const n = this.legs.length;
    ['slipCount','slipToggleCount','slipMobileCount','menuSlipCount'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.textContent = n;
    });
    const combined = this._totalPrice();
    const oddsStr = combined === null ? '--' : combined.toFixed(2);
    // Without a total price a stake has nothing to multiply, so the stake area steps aside.
    document.getElementById('slipFoot').classList.toggle('no-total', n > 0 && combined === null);
    ['slipOdds','slipMobileOdds'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.textContent = oddsStr;
    });
    // The chance, said once, when every leg's chance is known.
    const chance = slipChance(this.legs);
    document.getElementById('slipChance').classList.toggle('hidden', chance === null);
    if (chance !== null) {
      document.getElementById('slipChanceLabel').textContent = n === 1 ? 'Chance it lands' : n === 2 ? 'Chance both land' : `Chance all ${n} land`;
      document.getElementById('slipChanceWords').textContent = chanceWords(chance);
      document.getElementById('slipChancePct').textContent = chancePercent(chance);
    }
    const check = document.getElementById('slipCheck');
    check.textContent = "Check the price in your bookmaker's app before you place it.";
    check.classList.toggle('hidden', n === 0);
    this._calcPayout();

    // mobile bar
    const bar = document.getElementById('slipMobileBar');
    if (bar) bar.classList.toggle('hidden', n === 0);
  },

  _calcPayout() {
    const stake = parseFloat(document.getElementById('slipStake').value) || 0;
    const combined = this._totalPrice();
    document.getElementById('slipPayout').textContent = combined === null && this.legs.length
      ? '--'
      : '$' + (stake && combined !== null ? (stake * combined).toFixed(2) : '0.00');
  }
};

// ----- Mobile section menu (phones only; the toggle is hidden on wider screens) -----
const mobileMenu = {
  open: false,
  phone: window.matchMedia('(max-width: 640px)'),

  set(open) {
    if (this.open === open) return;
    this.open = open;
    document.querySelectorAll('[aria-controls="mobileMenu"]').forEach(button => {
      button.setAttribute('aria-expanded', String(open));
      button.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
    });
    ui.overlay('mobileMenu', 'menuBackdrop', open, () => this.close());
  },
  toggle() { this.set(!this.open); },
  close() { this.set(false); },
  go(tab) {
    this.close();
    app.switchTab(tab);
  },
  openSlip() {
    this.close();
    if (!slip.open) slip.toggle();
  },
};
// Leaving phone width with the menu open would strand a hidden dialog.
mobileMenu.phone.addEventListener('change', event => { if (!event.matches) mobileMenu.close(); });

document.getElementById('slipStake').addEventListener('input', () => slip._calcPayout());
slip._render();
slip._update();

// ----- App Module -----
const appModule = {
  LEAGUES: [
    { id: 'EPL', name: 'Premier League', color: '#3D195B', logo: '/static/league-logos/39.png' },
    { id: 'LaLiga', name: 'La Liga', color: '#EE8707', logo: '/static/league-logos/140.png' },
    { id: 'SerieA', name: 'Serie A', color: '#024494', logo: '/static/league-logos/135.png' },
    { id: 'Bundesliga', name: 'Bundesliga', color: '#D20515', logo: '/static/league-logos/78.png' },
    { id: 'Ligue1', name: 'Ligue 1', color: '#DAE025', logo: '/static/league-logos/61.png' },
    { id: 'Championship', name: 'Championship', color: '#1D4ED8', logo: '/static/league-logos/40.png' },
    { id: 'SuperLig', name: 'Süper Lig', color: '#E30A17', logo: '/static/league-logos/203.png' },
    { id: 'Eredivisie', name: 'Eredivisie', color: '#F97316', logo: '/static/league-logos/88.png' },
    { id: 'PrimeiraLiga', name: 'Primeira Liga', color: '#15803D', logo: '/static/league-logos/94.png' },
    { id: 'BelgianProLeague', name: 'Jupiler Pro League', color: '#EAB308', logo: '/static/league-logos/144.png' },
    { id: 'UCL', name: 'Champions League', color: '#001489', logo: '/static/league-logos/2.png' },
    { id: 'UEL', name: 'Europa League', color: '#F97316', logo: '/static/league-logos/3.png' },
    { id: 'UECL', name: 'Conference League', color: '#22C55E', logo: '/static/league-logos/848.png' },
  ],

  FIXTURES: [
    { id: 1, league: 'epl', home: 'Arsenal', away: 'Chelsea', time: '20:00', pick: 'Over 2.5 Goals', odds: 1.87, edge: 8.5, conf: 'high',
      markets: {
        goals: [ { pick: 'Over 2.5', odds: 1.87, edge: 8.5, conf: 'high', proj: '2.94 expected goals', rec: true }, { pick: 'Under 2.5', odds: 2.00, edge: -8.5, conf: 'low', proj: '' } ],
        btts: [ { pick: 'BTTS Yes', odds: 1.75, edge: 6.1, conf: 'high', proj: '73% model prob', rec: true }, { pick: 'BTTS No', odds: 2.10, edge: -6.1, conf: 'low', proj: '' } ],
        corners: [ { pick: 'Over 9.5', odds: 1.91, edge: 5.3, conf: 'med', proj: '10.2 expected', rec: false }, { pick: 'Under 9.5', odds: 1.92, edge: -5.3, conf: 'low', proj: '' } ],
        moneyline: [ { pick: 'Arsenal', odds: 2.10, edge: 4.2, conf: 'med', proj: 'Home 48%', rec: false }, { pick: 'Draw', odds: 3.40, edge: -2.1, conf: 'low', proj: 'Draw 27%' }, { pick: 'Chelsea', odds: 3.50, edge: -1.9, conf: 'low', proj: 'Away 25%' } ],
      }
    },
    { id: 2, league: 'laliga', home: 'Barcelona', away: 'Atletico', time: '21:00', pick: 'Over 9.5 Corners', odds: 1.91, edge: 7.2, conf: 'high',
      markets: {
        goals: [ { pick: 'Over 2.5', odds: 1.72, edge: 3.1, conf: 'med', proj: '2.61 expected', rec: false }, { pick: 'Under 2.5', odds: 2.15, edge: -3.1, conf: 'low', proj: '' } ],
        corners: [ { pick: 'Over 9.5', odds: 1.91, edge: 7.2, conf: 'high', proj: '10.8 expected', rec: true }, { pick: 'Under 9.5', odds: 1.91, edge: -7.2, conf: 'low', proj: '' } ],
        btts: [ { pick: 'BTTS Yes', odds: 1.80, edge: 2.1, conf: 'med', proj: '62% model prob', rec: false } ],
        moneyline: [ { pick: 'Barcelona', odds: 1.95, edge: 5.5, conf: 'high', proj: 'Home 51%', rec: true }, { pick: 'Draw', odds: 3.50, edge: -1.0, conf: 'low', proj: 'Draw 26%' }, { pick: 'Atletico', odds: 4.10, edge: -4.0, conf: 'low', proj: 'Away 23%' } ],
      }
    },
    { id: 3, league: 'bundesliga', home: 'Bayern', away: 'Dortmund', time: '18:30', pick: 'BTTS Yes', odds: 1.72, edge: 5.2, conf: 'med',
      markets: {
        goals: [ { pick: 'Over 3.5', odds: 1.85, edge: 6.4, conf: 'high', proj: '3.82 expected', rec: true }, { pick: 'Under 3.5', odds: 2.00, edge: -6.4, conf: 'low', proj: '' } ],
        btts: [ { pick: 'BTTS Yes', odds: 1.72, edge: 5.2, conf: 'med', proj: '68% model prob', rec: true }, { pick: 'BTTS No', odds: 2.20, edge: -5.2, conf: 'low', proj: '' } ],
        corners: [ { pick: 'Over 10.5', odds: 1.95, edge: 3.8, conf: 'med', proj: '11.1 expected', rec: false } ],
        moneyline: [ { pick: 'Bayern', odds: 1.60, edge: 7.2, conf: 'high', proj: 'Home 62%', rec: true }, { pick: 'Draw', odds: 4.20, edge: -2.0, conf: 'low', proj: 'Draw 20%' }, { pick: 'Dortmund', odds: 5.50, edge: -3.5, conf: 'low', proj: 'Away 18%' } ],
      }
    },
    { id: 4, league: 'seriea', home: 'Juventus', away: 'AC Milan', time: '20:45', pick: 'Under 3.5 Cards', odds: 2.05, edge: 9.1, conf: 'high',
      markets: {
        cards: [ { pick: 'Under 3.5 Cards', odds: 2.05, edge: 9.1, conf: 'high', proj: '2.8 expected', rec: true }, { pick: 'Over 3.5 Cards', odds: 1.80, edge: -9.1, conf: 'low', proj: '' } ],
        goals: [ { pick: 'Under 2.5', odds: 1.95, edge: 4.2, conf: 'med', proj: '2.1 expected', rec: false }, { pick: 'Over 2.5', odds: 1.90, edge: -4.2, conf: 'low', proj: '' } ],
        btts: [ { pick: 'BTTS No', odds: 2.00, edge: 3.5, conf: 'med', proj: '55% no prob', rec: false } ],
        moneyline: [ { pick: 'Juventus', odds: 2.30, edge: 2.8, conf: 'med', proj: 'Home 43%', rec: false }, { pick: 'Draw', odds: 3.20, edge: 1.5, conf: 'med', proj: 'Draw 31%' }, { pick: 'AC Milan', odds: 3.10, edge: -4.2, conf: 'low', proj: 'Away 26%' } ],
      }
    },
    { id: 5, league: 'ligue1', home: 'PSG', away: 'Marseille', time: '21:00', pick: 'PSG -1.5', odds: 2.20, edge: 4.8, conf: 'med',
      markets: {
        moneyline: [ { pick: 'PSG', odds: 1.50, edge: 6.1, conf: 'high', proj: 'Home 67%', rec: true }, { pick: 'Draw', odds: 4.50, edge: -3.2, conf: 'low', proj: 'Draw 18%' }, { pick: 'Marseille', odds: 6.00, edge: -4.0, conf: 'low', proj: 'Away 15%' } ],
        goals: [ { pick: 'Over 2.5', odds: 1.65, edge: 7.3, conf: 'high', proj: '3.1 expected', rec: true }, { pick: 'Under 2.5', odds: 2.30, edge: -7.3, conf: 'low', proj: '' } ],
        btts: [ { pick: 'BTTS Yes', odds: 1.90, edge: 3.1, conf: 'med', proj: '61% model prob', rec: false } ],
        corners: [ { pick: 'Over 9.5', odds: 1.88, edge: 4.5, conf: 'med', proj: '10.4 expected', rec: false } ],
      }
    },
    { id: 6, league: 'ucl', home: 'Real Madrid', away: 'Man City', time: '20:00', pick: 'Over 2.5 Goals', odds: 1.75, edge: 6.8, conf: 'high',
      markets: {
        goals: [ { pick: 'Over 2.5', odds: 1.75, edge: 6.8, conf: 'high', proj: '3.2 expected', rec: true }, { pick: 'Under 2.5', odds: 2.10, edge: -6.8, conf: 'low', proj: '' } ],
        btts: [ { pick: 'BTTS Yes', odds: 1.70, edge: 8.2, conf: 'high', proj: '76% model prob', rec: true }, { pick: 'BTTS No', odds: 2.25, edge: -8.2, conf: 'low', proj: '' } ],
        corners: [ { pick: 'Over 10.5', odds: 1.93, edge: 4.1, conf: 'med', proj: '11.3 expected', rec: false } ],
        moneyline: [ { pick: 'Real Madrid', odds: 2.40, edge: 3.5, conf: 'med', proj: 'Home 42%', rec: false }, { pick: 'Draw', odds: 3.30, edge: 1.2, conf: 'med', proj: 'Draw 30%' }, { pick: 'Man City', odds: 3.00, edge: -4.5, conf: 'low', proj: 'Away 28%' } ],
      }
    },
  ],

  currentTab: 'fixtures',
  currentMatch: null,
  currentMarketTab: 'goals',
  activeLeague: 'all',
  activeDate: 0,
  fixtures: [],
  bestBets: [],
  fixtureIndex: {},
  matchReadSource: 'loading',
  initialized: false,
  _requestId: 0,
  _matchReadPollTimer: null,
  _lastMatchReadPollAt: 0,

  async init() {
    if (!this.initialized) {
      this._buildDateStrips();
      this._buildLeagueFilters();
      this._installMatchReadPolling();
      this.initialized = true;
      this._initialLoad = this._loadMatchday();
    }
    return this._initialLoad;
  },

  _installMatchReadPolling() {
    // The worker updates persisted cards; the browser only polls the read-only
    // delivery API. Avoid a web socket/model call and avoid repeatedly hitting
    // the local legacy-preview fallback if the published board is unavailable.
    if (this._matchReadPollTimer || !window.setInterval) return;
    const poll = () => {
      if (document.visibilityState === 'hidden' || router.state?.page !== 'app') return;
      if (!['published', 'not-published', 'schedule', 'unavailable'].includes(this.matchReadSource)) return;
      this._lastMatchReadPollAt = Date.now();
      this._loadMatchday({ background: true });
    };
    this._matchReadPollTimer = window.setInterval(poll, 90 * 1000);
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState !== 'visible') return;
      // A user returning after a worker release should not need a manual
      // refresh, while a quick tab switch should not trigger duplicate calls.
      if (Date.now() - this._lastMatchReadPollAt >= 30 * 1000) poll();
    });
  },

  switchTab(tab) {
    const restored = this._restoringRoute;
    this._restoringRoute = null;
    const fromBoard = tab === 'match' && router.state?.page === 'app'
      && ['fixtures', 'best-bets'].includes(router.state.tab);
    router.go('app', tab, restored);
    if (fromBoard && !restored) {
      router.state.fromBoard = true;
      router.remember();
    }
  },

  _buildDateStrips() {
    const today = new Date();
    ['dateStrip','bestBetsDateStrip'].forEach(id => {
      const strip = document.getElementById(id);
      if (!strip) return;
      strip.innerHTML = '';
      for (let i = 0; i < 7; i++) {
        const d = new Date(today);
        d.setDate(today.getDate() + i);
        const label = i === 0 ? 'Today' : i === 1 ? 'Tomorrow' : d.toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' });
        const btn = document.createElement('button');
        btn.className = 'date-chip' + (i === this.activeDate ? ' active' : '');
        btn.textContent = label;
        btn.dataset.dateOffset = String(i);
        btn.setAttribute('aria-pressed', String(i === this.activeDate));
        btn.onclick = () => this._setActiveDate(i);
        strip.appendChild(btn);
      }
    });
  },

  _setActiveDate(offset) {
    if (this.activeDate === offset) return;
    this.activeDate = offset;
    document.querySelectorAll('.date-chip').forEach(btn => {
      btn.classList.toggle('active', Number(btn.dataset.dateOffset) === offset);
      btn.setAttribute('aria-pressed', String(Number(btn.dataset.dateOffset) === offset));
    });
    this._loadMatchday();
  },

  _buildLeagueFilters() {
    const filters = document.getElementById('leagueFilters');
    if (!filters) return;
    this._leagueResizeObserver?.disconnect();
    filters.innerHTML = `
      <button class="league-chip league-all" data-league="all" aria-label="All leagues">All</button>
      <div class="league-scroll-wrap">
        <div class="league-viewport" id="leagueViewport">
          <div class="league-scroll-content">
            ${this.LEAGUES.map(league => `<button class="league-chip" data-league="${esc(league.id)}">
              <span class="league-logo" aria-hidden="true"><img src="${esc(league.logo)}" width="20" height="20" alt="" decoding="async" draggable="false"></span>
              <span>${esc(league.name)}</span>
            </button>`).join('')}
            <span class="league-selected-line" aria-hidden="true"></span>
          </div>
        </div>
      </div>
      <div class="league-scroll-controls hidden">
        <button class="league-scroll-button league-scroll-prev" aria-label="Scroll to earlier leagues" aria-controls="leagueViewport">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M10 3L5 8l5 5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
        </button>
        <button class="league-scroll-button league-scroll-next" aria-label="Scroll to more leagues" aria-controls="leagueViewport">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M6 3l5 5-5 5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
        </button>
      </div>`;
    filters.querySelectorAll('.league-chip').forEach(button => {
      button.setAttribute('aria-controls', 'fixturesContainer');
      button.addEventListener('click', () => {
        const changed = this.activeLeague !== button.dataset.league;
        this.activeLeague = button.dataset.league;
        this._syncLeagueChips(button);
        this._revealLeague(button);
        if (changed) this._loadMatchday();
      });
    });
    filters.querySelectorAll('.league-logo img').forEach(img => {
      img.addEventListener('error', () => { img.parentElement.style.visibility = 'hidden'; }, { once: true });
    });
    const viewport = filters.querySelector('.league-viewport');
    viewport.addEventListener('scroll', () => this._queueLeagueRailUpdate(), { passive: true });
    filters.querySelector('.league-scroll-prev').onclick = () => this._scrollLeagues(-1);
    filters.querySelector('.league-scroll-next').onclick = () => this._scrollLeagues(1);
    // Toolbar arrows move focus; Enter/Space activate the focused filter.
    // Browsing league names alone does not trigger a stream of data requests.
    filters.onkeydown = event => {
      const button = event.target.closest('.league-chip');
      if (!button || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const buttons = [...filters.querySelectorAll('.league-chip')];
      const current = buttons.indexOf(button);
      const index = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1
        : Math.max(0, Math.min(buttons.length - 1, current + (event.key === 'ArrowRight' ? 1 : -1)));
      buttons.forEach(item => { item.tabIndex = item === buttons[index] ? 0 : -1; });
      buttons[index].focus({ preventScroll: true });
      this._revealLeague(buttons[index]);
    };
    this._syncLeagueChips([...filters.querySelectorAll('.league-chip')]
      .find(button => button.dataset.league === this.activeLeague) || filters.querySelector('.league-all'));
    if (window.ResizeObserver) {
      this._leagueResizeObserver = new ResizeObserver(() => this._queueLeagueRailUpdate());
      this._leagueResizeObserver.observe(viewport);
      this._leagueResizeObserver.observe(filters.querySelector('.league-scroll-content'));
    }
    document.fonts?.ready.then(() => this._queueLeagueRailUpdate());
  },

  _syncLeagueChips(activeBtn) {
    document.querySelectorAll('#leagueFilters .league-chip').forEach(button => {
      const active = button === activeBtn;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
      button.tabIndex = active ? 0 : -1;
    });
    this._queueLeagueRailUpdate();
  },

  _queueLeagueRailUpdate() {
    if (this._leagueRailFrame) return;
    this._leagueRailFrame = requestAnimationFrame(() => {
      this._leagueRailFrame = 0;
      this._updateLeagueRail();
    });
  },

  _updateLeagueRail() {
    const filters = document.getElementById('leagueFilters');
    const viewport = filters?.querySelector('.league-viewport');
    if (!viewport?.clientWidth) return;
    const controls = filters.querySelector('.league-scroll-controls');
    const gap = parseFloat(getComputedStyle(filters).columnGap) || 0;
    // Measure the space available without arrows to avoid making their own
    // width create a permanent overflow state at a breakpoint.
    const available = viewport.clientWidth + controls.offsetWidth + (controls.offsetWidth ? gap : 0);
    const overflow = viewport.scrollWidth > available + 1;
    controls.classList.toggle('hidden', !overflow);
    const maxScroll = viewport.scrollWidth - viewport.clientWidth;
    const atStart = viewport.scrollLeft <= 2;
    const atEnd = viewport.scrollLeft >= maxScroll - 2;
    filters.classList.toggle('can-scroll-left', overflow && !atStart);
    filters.classList.toggle('can-scroll-right', overflow && !atEnd);
    filters.querySelector('.league-scroll-prev').disabled = !overflow || atStart;
    filters.querySelector('.league-scroll-next').disabled = !overflow || atEnd;
    const active = filters.querySelector('.league-scroll-content .league-chip.active');
    const line = filters.querySelector('.league-selected-line');
    line.style.opacity = active ? '1' : '0';
    if (active) line.style.transform = `translateX(${active.offsetLeft}px) scaleX(${active.offsetWidth})`;
  },

  _revealLeague(button) {
    const viewport = document.getElementById('leagueViewport');
    if (!viewport) return;
    const behavior = ui.reducedMotion.matches ? 'instant' : 'smooth';
    if (button.dataset.league === 'all') { viewport.scrollTo({ left: 0, behavior }); return; }
    const target = button.getBoundingClientRect(), windowRect = viewport.getBoundingClientRect();
    if (target.left < windowRect.left + 12) viewport.scrollBy({ left: target.left - windowRect.left - 12, behavior });
    else if (target.right > windowRect.right - 12) viewport.scrollBy({ left: target.right - windowRect.right + 12, behavior });
  },

  _scrollLeagues(direction) {
    const viewport = document.getElementById('leagueViewport');
    if (!viewport) return;
    viewport.scrollBy({ left: direction * Math.max(140, viewport.clientWidth * .7),
      behavior: ui.reducedMotion.matches ? 'instant' : 'smooth' });
  },

  _selectedDateISO() {
    const selected = new Date();
    // Midday avoids DST and UTC-boundary surprises when a user selects a date.
    selected.setHours(12, 0, 0, 0);
    selected.setDate(selected.getDate() + this.activeDate);
    const year = selected.getFullYear();
    const month = String(selected.getMonth() + 1).padStart(2, '0');
    const day = String(selected.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
  },

  _formatFixtureTime(kickoff) {
    if (!kickoff) return 'Time TBC';
    const timestamp = new Date(kickoff);
    if (Number.isNaN(timestamp.getTime())) return 'Time TBC';
    return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
  },

  _formatMarketPick(result) {
    const decision = result.decision || {};
    const quote = decision.quote || {};
    if (decision.status !== 'recommended' || !quote.side) return 'No qualified bet';

    const side = String(quote.side).replace(/\b\w/g, char => char.toUpperCase());
    if (result.market?.group === 'btts') return `BTTS ${side}`;
    if (quote.line === null || quote.line === undefined) return side;
    if (result.market?.group === 'spreads') {
      return `${side} ${Number(quote.line) >= 0 ? '+' : ''}${quote.line}`;
    }
    return `${side} ${quote.line}`;
  },

  _formatProjection(result) {
    const value = result.projection?.value;
    if (value === null || value === undefined || Number.isNaN(Number(value))) return 'Projection unavailable';
    const group = result.market?.group;
    if (group === 'btts' || group === 'moneyline') return `${(Number(value) * 100).toFixed(1)}% model probability`;
    const labels = { goals: 'goals', corners: 'corners', cards: 'cards', sot: 'shots on target', spreads: 'goal difference' };
    return `${Number(value).toFixed(2)} projected ${labels[group] || 'value'}`;
  },

  _formatMatchReadSelection(selection) {
    const raw = selection?.model_probability;
    const probability = raw === null || raw === undefined ? NaN : Number(raw);
    const label = selection?.probability_basis === 'asian_equivalent_non_push'
      ? 'Asian price-comparison probability'
      : selection?.probability_source === 'market' ? 'market probability' : 'model probability';
    if (Number.isFinite(probability)) return `${(probability * 100).toFixed(1)}% ${label}`;
    return 'Model probability unavailable';
  },

  _normaliseMatchReadCard(card, leagueId) {
    const selections = Array.isArray(card.selections) ? card.selections : [];
    const core = selections.find(selection => selection.role === 'core') || selections[0] || null;
    const markets = {};
    selections.forEach(selection => {
      const group = selection.market?.group;
      if (!group) return;
      const odds = selection.odds == null ? NaN : Number(selection.odds);
      const edge = selection.value_edge == null ? NaN : Number(selection.value_edge);
      markets[group] = [{
        pick: selection.pick || 'Selection unavailable',
        odds: Number.isFinite(odds) ? odds : null,
        edge: Number.isFinite(edge) ? edge * 100 : null,
        conf: selection.confidence || 'low',
        rec: card.status === 'recommended' && Number.isFinite(odds),
        proj: this._formatMatchReadSelection(selection),
        reason: selection.role === 'core' ? 'Core Match Read selection.' : 'Supporting Match Read selection.',
        bookmaker: selection.bookmaker || '',
        role: selection.role || 'supporting',
      }];
    });

    const coreOdds = core?.odds == null ? NaN : Number(core.odds);
    const coreEdge = core?.value_edge == null ? NaN : Number(core.value_edge);
    const fixture = card.fixture || {};
    const update = card.update || {};
    const status = card.status || 'unavailable';
    const briefing = card.briefing || {};
    const visuals = card.visuals || {};
    const rawMetrics = card.metrics || {};
    const finite = value => (value == null || !Number.isFinite(Number(value)) ? null : Number(value));
    const probabilities = rawMetrics.result_probabilities || null;
    const fromText = briefingFigures(briefing.bullets, briefing.summary || card.thesis);
    // Structured card metrics first; the deterministic briefing wording only
    // when the API response predates them.
    const balance = probabilities
      && [probabilities.home, probabilities.draw, probabilities.away].every(value => finite(value) !== null)
      ? { home: finite(probabilities.home) * 100, draw: finite(probabilities.draw) * 100, away: finite(probabilities.away) * 100 }
      : fromText.resultBalance;
    const projectedTotal = finite(rawMetrics.projected_total_goals) ?? fromText.projectedTotal;
    const teamGoals = finite(rawMetrics.team_goals?.home) !== null
      ? { home: finite(rawMetrics.team_goals?.home), away: finite(rawMetrics.team_goals?.away) }
      : fromText.teamGoals;
    return {
      id: String(fixture.event_id || card.id),
      matchReadId: card.id,
      matchReadVersion: card.version,
      matchReadStage: card.stage,
      isMatchRead: true,
      league: leagueId || fixture.league,
      home: fixture.home_team || 'Home team TBC',
      away: fixture.away_team || 'Away team TBC',
      kickoff: fixture.kickoff,
      time: this._formatFixtureTime(fixture.kickoff),
      pick: core?.pick || (status === 'no_bet' ? 'No bet' : 'Unavailable'),
      market: core?.market?.group || null,
      odds: Number.isFinite(coreOdds) ? coreOdds : null,
      edge: Number.isFinite(coreEdge) ? coreEdge * 100 : null,
      conf: core?.confidence || status,
      best: status === 'recommended' && Boolean(core) && Number.isFinite(coreOdds),
      status,
      thesis: card.thesis || '',
      briefing: {
        summary: briefing.summary || '',
        bullets: Array.isArray(briefing.bullets) ? briefing.bullets.slice(0, 3) : [],
      },
      metrics: { projectedTotal, teamGoals, resultBalance: balance },
      visuals: {
        homeTeamLogo: visuals.home_team_logo || '',
        awayTeamLogo: visuals.away_team_logo || '',
        leagueLogo: visuals.league_logo || '',
      },
      isPreliminary: Boolean(card.timing?.is_preliminary),
      selections,
      alternatives: Array.isArray(card.game_script?.alternative_candidates)
        ? card.game_script.alternative_candidates
        : [],
      selectionRelationship: card.game_script?.selection_relationship || '',
      tags: Array.isArray(card.game_script?.tags) ? card.game_script.tags : [],
      update,
      markets,
      recommendations: recommendationsFor(card, window.location.search),
    };
  },

  _indexFixtures(fixtures) {
    fixtures.forEach(fixture => {
      if (fixture?.id) this.fixtureIndex[String(fixture.id)] = fixture;
    });
  },

  _allowsLegacyPreview() {
    const host = String(window.location.hostname || '').toLowerCase();
    return host === 'localhost' || host === '127.0.0.1' || host === '::1';
  },

  _normaliseFixture(raw, leagueId) {
    const markets = {};
    (raw.market_results || []).forEach(result => {
      const group = result.market?.group;
      if (!group) return;
      const decision = result.decision || {};
      const quote = decision.quote || {};
      const recommended = decision.status === 'recommended' && Number.isFinite(Number(quote.odds));
      markets[group] = [{
        pick: this._formatMarketPick(result),
        odds: recommended ? Number(quote.odds) : null,
        edge: Number.isFinite(Number(decision.value_edge)) ? Number(decision.value_edge) * 100 : null,
        conf: decision.confidence || 'low',
        rec: recommended,
        proj: this._formatProjection(result),
        reason: decision.reason || '',
        bookmaker: quote.bookmaker || '',
      }];
    });

    const bestBet = raw.best_bet;
    return {
      id: String(raw.event_id),
      league: leagueId,
      home: raw.home_team,
      away: raw.away_team,
      kickoff: raw.kickoff,
      time: this._formatFixtureTime(raw.kickoff),
      pick: bestBet?.pick || 'No qualified bet',
      odds: Number.isFinite(Number(bestBet?.odds)) ? Number(bestBet.odds) : null,
      edge: Number.isFinite(Number(bestBet?.edge)) ? Number(bestBet.edge) * 100 : null,
      conf: bestBet?.confidence || 'no-bet',
      best: Boolean(bestBet && Number.isFinite(Number(bestBet.odds))),
      markets,
    };
  },

  _buildBestBetsFromFixtures(fixtures = this.fixtures) {
    const confidenceRank = { high: 3, medium: 2, low: 1 };
    return fixtures
      .filter(fixture => fixture.best && ['high', 'medium'].includes(fixture.conf))
      .sort((a, b) => (
        (confidenceRank[b.conf] || 0) - (confidenceRank[a.conf] || 0)
        || (b.edge || 0) - (a.edge || 0)
      ))
      .slice(0, 5);
  },

  _normaliseScheduledFixture(row, analysisAvailable) {
    const fixture = row.fixture || {};
    const status = String(row.status || '').toUpperCase();
    const scheduled = ['NS', 'TBD'].includes(status) && Date.parse(fixture.kickoff) > Date.now();
    const labels = { PST: 'Postponed', CANC: 'Cancelled', ABD: 'Abandoned', SUSP: 'Suspended',
      FT: 'Full time', AET: 'Full time', PEN: 'Full time', AWD: 'Match awarded', WO: 'Walkover' };
    const label = scheduled
      ? (analysisAvailable ? 'Analysis pending' : 'Analysis temporarily unavailable')
      : (labels[status] || (['NS', 'TBD', '1H', 'HT', '2H', 'ET', 'BT', 'P', 'LIVE', 'INT'].includes(status)
        ? 'Match started' : 'Schedule status unavailable'));
    return {
      id: String(fixture.event_id), league: fixture.league,
      home: fixture.home_team || 'Home team TBC', away: fixture.away_team || 'Away team TBC',
      kickoff: fixture.kickoff, time: this._formatFixtureTime(fixture.kickoff),
      isScheduleOnly: true, scheduled, status: 'pending', scheduleLabel: label,
      scheduleNote: scheduled
        ? (analysisAvailable ? 'A Match Read will appear here once it is published.' : 'Please check back for the latest Match Read.')
        : 'No pre-match recommendation is available for this fixture.',
      best: false, odds: null, edge: null, selections: [], markets: {},
      visuals: { homeTeamLogo: row.visuals?.home_team_logo || '', awayTeamLogo: row.visuals?.away_team_logo || '' },
    };
  },

  // A started match in a league that sent before-kickoff picks (D71): its picks if it had any, the
  // live status when known, and a true sentence if it had none. Not a postponed or called-off match.
  _afterKickoff(fixture, row, picks) {
    if (['PST', 'CANC', 'AWD', 'WO'].includes(String(row.status || '').toUpperCase())) return fixture;
    const live = picks?.status || row.live_status;
    const label = live ? liveStatusLabel(live, fixture.time)
      : fixture.scheduleLabel === 'Match started' ? `Kicked off ${fixture.time}` : fixture.scheduleLabel;
    return { ...fixture, afterKickoff: true, beforeKickoff: picks || null, scheduleLabel: label,
      scheduleNote: picks ? fixture.scheduleNote : 'Spix had no picks for this match before kickoff.' };
  },

  _mergeSchedule(schedule, cards, availableLeagues, beforeKickoff = new Map()) {
    const published = new Map(cards.map(card => [String(card.id), card]));
    const pickedLeagues = new Set([...beforeKickoff.values()].map(picks => picks.league));
    const merged = new Map();
    schedule.forEach(row => {
      let fixture = this._normaliseScheduledFixture(row, availableLeagues.has(row.fixture?.league));
      if (!fixture.scheduled && pickedLeagues.has(fixture.league)) fixture = this._afterKickoff(fixture, row, beforeKickoff.get(fixture.id));
      const read = published.get(fixture.id);
      // A known postponement/cancellation or kickoff must not become actionable
      // because an earlier published card still exists.
      merged.set(fixture.id, fixture.scheduled && read ? read : fixture);
    });
    // Preserve valid public reads if their fixture is temporarily absent from
    // the schedule response. An unavailable schedule is not a withdrawal.
    cards.forEach(card => { if (!merged.has(String(card.id))) merged.set(String(card.id), card); });
    return [...merged.values()].sort((a, b) => String(a.kickoff).localeCompare(String(b.kickoff))
      || String(a.league).localeCompare(String(b.league)) || String(a.id).localeCompare(String(b.id)));
  },

  async _loadMatchday({ background = false } = {}) {
    const requestId = ++this._requestId;
    const containers = ['fixturesContainer', 'bestBetsContainer'].map(id => document.getElementById(id));
    containers.forEach(el => el?.setAttribute('aria-busy', 'true'));
    try {
      await this._fetchMatchday(requestId, background);
    } finally {
      if (requestId === this._requestId) {
        containers.forEach(el => el?.setAttribute('aria-busy', 'false'));
        if (!background) containers.forEach(el => { if (el) ui.enter(el); });
      }
    }
  },

  async _fetchMatchday(requestId, background) {
    const container = document.getElementById('fixturesContainer');
    const bestBetsContainer = document.getElementById('bestBetsContainer');
    if (!container || !bestBetsContainer) return;

    const targetDate = this._selectedDateISO();
    const requestedLeagues = this.activeLeague === 'all'
      ? this.LEAGUES
      : this.LEAGUES.filter(league => league.id === this.activeLeague);

    if (!background) {
      this.fixtures = [];
      this.bestBets = [];
      this.fixtureIndex = {};
      this.matchReadSource = 'loading';
      this._renderFixtures({ loading: true });
      this._renderBestBets({ loading: true });
    }

    const scheduleUrl = `/api/match-reads/schedule/${targetDate}`
      + (this.activeLeague === 'all' ? '' : `?league=${encodeURIComponent(this.activeLeague)}`);
    const responses = await Promise.allSettled([
      (async () => {
        const response = await fetch(scheduleUrl);
        if (!response.ok) throw new Error('Fixture schedule could not be loaded');
        const payload = await response.json();
        if (!Array.isArray(payload.fixtures)) throw new Error('Invalid fixture schedule');
        return payload.fixtures;
      })(),
      ...requestedLeagues.map(async league => {
        const response = await fetch(`/api/match-reads/${encodeURIComponent(league.id)}/${targetDate}`);
        if (!response.ok) throw new Error(`${league.name} Match Reads could not be loaded (${response.status})`);
        const payload = await response.json();
        if (!Array.isArray(payload.cards)) throw new Error('Invalid Match Read board');
        return { league, payload };
      }),
    ]);
    if (requestId !== this._requestId) return;

    const [scheduleResponse, ...boardResponses] = responses;
    const scheduleLoaded = scheduleResponse.status === 'fulfilled';
    const successfulBoards = boardResponses.filter(result => result.status === 'fulfilled');
    const availableLeagues = new Set(successfulBoards.map(result => result.value.league.id));
    const persistedCards = successfulBoards.flatMap(result => result.value.payload.cards
      .map(card => this._normaliseMatchReadCard(card, result.value.league.id)));
    const beforeKickoff = new Map();
    successfulBoards.forEach(result => {
      const entries = Array.isArray(result.value.payload.before_kickoff) ? result.value.payload.before_kickoff : [];
      entries.forEach(entry => {
        const picks = beforeKickoffFor(entry, window.location.search, result.value.league.id);
        if (picks) beforeKickoff.set(picks.id, picks);
      });
    });
    const fixtures = this._mergeSchedule(scheduleLoaded ? scheduleResponse.value : [], persistedCards, availableLeagues, beforeKickoff);
    const notices = [];
    if (!scheduleLoaded) notices.push('The full fixture schedule could not be loaded. Showing published Match Reads only.');
    const failed = boardResponses.length - successfulBoards.length;
    if (failed) notices.push(`${failed} league${failed === 1 ? '' : 's'} could not load its analyses. Please try again shortly.`);

    let bestBets = [];
    if (persistedCards.length) {
      // The shortlist still comes exclusively from the published-card endpoint.
      // Schedule-only fixtures cannot participate in ranking or bet selection.
      try {
        const response = await fetch(`/api/match-reads/best/${targetDate}`);
        if (!response.ok) throw new Error('Best Match Reads could not be loaded');
        const payload = await response.json();
        bestBets = (Array.isArray(payload.cards) ? payload.cards : [])
          .map(card => this._normaliseMatchReadCard(card, card.fixture?.league));
      } catch (error) {
        console.warn('Persisted Best Match Reads unavailable; using board cards only.', error);
        bestBets = this._buildBestBetsFromFixtures(persistedCards);
      }
    }
    if (requestId !== this._requestId) return;
    // Remove any known non-actionable schedule state from this display too;
    // never replace a published recommendation with an invented selection.
    const closedIds = new Set(fixtures.filter(f => f.isScheduleOnly && !f.scheduled).map(f => f.id));
    bestBets = bestBets.filter(f => !closedIds.has(String(f.id)));
    const notice = notices.join(' ');
    const unchanged = background && JSON.stringify(this.fixtures) === JSON.stringify(fixtures)
      && JSON.stringify(this.bestBets) === JSON.stringify(bestBets) && this._boardNotice === notice;
    this._boardNotice = notice;
    this.fixtures = fixtures;
    this.bestBets = bestBets;
    this.fixtureIndex = {};
    this._indexFixtures(fixtures);
    this._indexFixtures(bestBets);
    this.matchReadSource = persistedCards.length ? 'published' : scheduleLoaded ? 'schedule' : 'unavailable';
    if (unchanged) return;
    if (!scheduleLoaded && !successfulBoards.length) {
      this._renderFixtures({ error: 'The fixture schedule and Match Reads are temporarily unavailable. Please try again shortly.' });
      this._renderBestBets({ error: 'Published Best Bets are temporarily unavailable.' });
      return;
    }
    this._renderFixtures({ notice,
      emptyMessage: scheduleLoaded ? 'No fixtures are listed for this matchday.' : 'No published Match Reads are available.',
      emptyHint: scheduleLoaded ? 'Try another date or league.' : 'The full schedule is temporarily unavailable.',
    });
    this._renderBestBets();
  },

  async _loadLegacyPreview(requestId, targetDate, requestedLeagues) {
    const responses = await Promise.allSettled(requestedLeagues.map(async league => {
      const response = await fetch(`/api/fixtures/${encodeURIComponent(league.id)}/${targetDate}`);
      if (!response.ok) throw new Error(`${league.name} could not be loaded (${response.status})`);
      const payload = await response.json();
      return { league, payload };
    }));
    if (requestId !== this._requestId) return;

    const successful = responses.filter(result => result.status === 'fulfilled');
    if (!successful.length) {
      this._renderFixtures({ error: 'Published Match Reads are unavailable, and the local preview could not be loaded.' });
      this._renderBestBets({ error: 'Best of Today is unavailable while Match Reads are loading.' });
      return;
    }

    this.matchReadSource = 'legacy-preview';
    this.fixtures = successful.flatMap(result => (
      result.value.payload.fixtures.map(fixture => this._normaliseFixture(fixture, result.value.league.id))
    ));
    this._indexFixtures(this.fixtures);
    this.bestBets = this._buildBestBetsFromFixtures();
    const failed = responses.length - successful.length;
    const previewNotice = 'Development preview — persisted Match Reads are unavailable. These legacy calculations are not published recommendations and are not part of the official track record.';
    this._renderFixtures({
      notice: failed
        ? `${previewNotice} ${failed} league${failed === 1 ? '' : 's'} could not be loaded.`
        : previewNotice,
    });
    this._renderBestBets({ notice: 'Development preview only — not published Match Reads.' });
  },

  _renderFixtures(state = {}) {
    const container = document.getElementById('fixturesContainer');
    if (!container) return;
    // A single filtered league lays its cards out in rows on desktop (style.css).
    container.classList.toggle('is-single-league', this.activeLeague !== 'all');
    if (state.loading) {
      container.innerHTML = '<div class="empty-state"><p>Preparing the matchday board…</p><span class="empty-state-hint">Loading the fixture schedule and published Match Reads.</span></div>';
      return;
    }
    if (state.error) {
      container.innerHTML = `<div class="error-state"><p>${esc(state.error)}</p><button class="retry-btn" onclick="appModule._loadMatchday()">Try again</button></div>`;
      return;
    }
    const fixtures = this.fixtures;
    const byLeague = {};
    fixtures.forEach(f => { (byLeague[f.league] = byLeague[f.league] || []).push(f); });
    container.innerHTML = '';
    if (state.notice) {
      const notice = document.createElement('p');
      notice.className = 'empty-state-hint';
      notice.style.margin = '0 0 16px';
      notice.textContent = state.notice;
      container.appendChild(notice);
    }
    if (!fixtures.length) {
      const message = state.emptyMessage || 'No fixtures for this selection.';
      const hint = state.emptyHint || 'There may be no matches in the selected leagues on this date.';
      container.innerHTML += `<div class="empty-state"><div class="empty-state-icon"><svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2"><circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/></svg></div><p>${esc(message)}</p><span class="empty-state-hint">${esc(hint)}</span></div>`;
      return;
    }
    Object.entries(byLeague).forEach(([leagueId, fxs]) => {
      const league = this.LEAGUES.find(l => l.id === leagueId) || { name: leagueId };
      const group = document.createElement('section');
      group.className = 'league-group';
      group.setAttribute('aria-label', league.name);
      const crest = league.logo
        ? `<span class="league-crest" aria-hidden="true"><img src="${esc(league.logo)}" alt=""></span>`
        : '';
      group.innerHTML = `<div class="league-header"><div class="league-header-left">${crest}<h2 class="league-name">${esc(league.name)}</h2></div><span class="league-count">${esc(pluralise(fxs.length, 'match', 'matches'))}</span></div>`;
      const { upcoming, started } = splitStartedFixtures(fxs);
      [...upcoming, ...started].forEach((f, i) => {
        if (started.length && i === upcoming.length) group.insertAdjacentHTML('beforeend', '<p class="bk-divider">Started</p>');
        if (f.isMatchRead || f.isScheduleOnly) {
          group.appendChild(this._createMatchReadFixtureCard(f, league, i));
          return;
        }
        const slipKey = `${f.home} vs ${f.away}|${f.pick}`;
        const inSlip = slip.legs.find(l => l.key === slipKey);
        const row = document.createElement('div');
        row.className = 'fixture-row';
        const confidenceLabel = f.best
          ? f.conf.toUpperCase()
          : f.status === 'unavailable' ? 'UNAVAILABLE' : 'NO BET';
        const confidenceClass = f.best ? f.conf : 'no-bet';
        const supportingCount = f.isMatchRead ? Math.max(0, (f.selections || []).length - 1) : 0;
        const supportingLabel = supportingCount
          ? ` + ${supportingCount} aligned angle${supportingCount === 1 ? '' : 's'}`
          : '';
        const updateLabel = f.isMatchRead && f.update?.is_updated && f.update?.label
          ? `<span class="match-read-update-label">${esc(f.update.label)}</span>`
          : '';
        const addButton = f.best
          ? `<button class="fixture-add-btn${inSlip ? ' added' : ''}" data-fixture-add="${esc(f.id)}">${addButtonContent(Boolean(inSlip), false)}</button>`
          : '<span class="fixture-add-placeholder" aria-hidden="true"></span>';
        row.innerHTML = `
          <span class="fixture-num mono">${i + 1}</span>
          <div class="fixture-match">
            <button class="fixture-teams" type="button" aria-label="Open Match Read: ${esc(f.home)} versus ${esc(f.away)}">${esc(f.home)} <span class="fixture-vs">v</span> ${esc(f.away)}</button>
            <div class="fixture-meta-line"><span class="fixture-time mono">${esc(f.time)}</span>${updateLabel}</div>
          </div>
          <span class="fixture-pick-text">${esc(f.pick)}${esc(supportingLabel)}</span>
          <span class="conf-badge ${esc(confidenceClass)}">${esc(confidenceLabel)}</span>
          <span class="fixture-odds-val mono">${f.odds ? esc(String(f.odds)) : '—'}</span>
          <span class="fixture-edge mono ${f.edge && f.edge > 0 ? 'edge-pos' : ''}">${Number.isFinite(f.edge) ? `${f.edge >= 0 ? '+' : ''}${esc(f.edge.toFixed(1))}%` : '—'}</span>
          ${addButton}
        `;
        // The whole row opens the read for pointer users; the team-name button
        // is the keyboard and screen-reader entry point.
        row.onclick = () => this._showMatch(f.id);
        const add = row.querySelector('[data-fixture-add]');
        if (add) add.addEventListener('click', event => {
          event.stopPropagation();
          this._toggleSlipFromFixture(f.id);
        });
        group.appendChild(row);
      });
      container.appendChild(group);
    });
  },

  _createMatchReadFixtureCard(f, league, index) {
    // The card's content is read in full by assistive technology; the "View
    // Match Read" button is its keyboard entry point, while a click anywhere
    // on the card opens the read for pointer users.
    const card = document.createElement('article');
    card.className = 'match-read-fixture-card';
    card.dataset.fixtureId = String(f.id);
    if (f.isScheduleOnly) card.classList.add('schedule-fixture-card');

    const initials = team => String(team || '?')
      .split(/\s+/)
      .filter(Boolean)
      .slice(0, 2)
      .map(part => part[0])
      .join('')
      .toUpperCase() || '?';
    const emblem = (url, label, className) => {
      const safeUrl = /^https?:\/\//i.test(String(url || '')) ? esc(url) : '';
      const fallback = esc(initials(label));
      return `<span class="match-card-emblem ${className}${safeUrl ? '' : ' is-fallback'}">${safeUrl ? `<img src="${safeUrl}" alt="" loading="lazy">` : ''}<span aria-hidden="true">${fallback}</span></span>`;
    };
    const bullets = Array.isArray(f.briefing?.bullets) ? f.briefing.bullets.slice(0, 3) : [];
    const figures = figuresMarkup(f.metrics, f.home, f.away, 'match-card-figures');
    const notes = bullets.filter(bullet => !restatesFigures(bullet, f.metrics));
    const bulletMarkup = notes.length
      ? `<ul class="match-card-signals">${notes.map(bullet => `<li>${esc(bullet)}</li>`).join('')}</ul>`
      : '';
    const briefingMarkup = bullets.length || figures
      ? figures + bulletMarkup
      : '<ul class="match-card-signals"><li class="match-card-briefing-pending">Full fixture briefing will appear after the next model refresh.</li></ul>';
    const updateLabel = f.update?.is_updated && f.update?.label
      ? `<span class="match-card-updated">${esc(f.update.label)}</span>`
      : '';
    const statusLabel = f.isScheduleOnly ? f.scheduleLabel : f.isPreliminary
      ? 'Preliminary Match Read'
      : f.status === 'recommended'
      ? 'Match Read ready'
      : f.status === 'no_bet' ? 'No bet released' : 'Assessment unavailable';
    const statusClass = f.isScheduleOnly ? 'pending' : f.isPreliminary
      ? 'preliminary'
      : f.status === 'recommended' ? 'ready' : f.status === 'no_bet' ? 'no-bet' : 'unavailable';

    const openButton = f.isScheduleOnly
      ? ''
      : `<button class="match-card-open" type="button" aria-label="View Match Read: ${esc(f.home)} versus ${esc(f.away)}">View Match Read <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M5 12h14m-6-6 6 6-6 6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg></button>`;

    if (f.afterKickoff) {
      card.classList.add('bk-card');
      card.innerHTML = renderBeforeKickoffCard(f, emblem(f.visuals?.homeTeamLogo, f.home, 'home'), emblem(f.visuals?.awayTeamLogo, f.away, 'away'));
    } else if (f.recommendations && !f.isScheduleOnly) {
      card.classList.add('recs-board-card');
      card.innerHTML = renderRecsBoardCard(f, emblem(f.visuals?.homeTeamLogo, f.home, 'home'), emblem(f.visuals?.awayTeamLogo, f.away, 'away'));
    } else card.innerHTML = `
      <div class="match-card-topline">
        <span class="match-card-time mono">${esc(f.time)}</span>
        ${updateLabel}
      </div>
      <h3 class="match-card-teams">
        <span class="match-card-team home">${emblem(f.visuals?.homeTeamLogo, f.home, 'home')}<span>${esc(f.home)}</span></span>
        <span class="sr-only"> versus </span>
        <span class="match-card-team away">${emblem(f.visuals?.awayTeamLogo, f.away, 'away')}<span>${esc(f.away)}</span></span>
      </h3>
      ${f.isScheduleOnly
        ? `<p class="match-card-schedule-note empty-state-hint">${esc(f.scheduleNote)}</p>`
        : briefingMarkup}
      <div class="match-card-footer">
        <span class="match-card-status ${statusClass}">${esc(statusLabel)}</span>
        ${openButton}
      </div>
    `;
    card.querySelectorAll('img').forEach(image => {
      image.addEventListener('error', () => {
        const emblemNode = image.closest('.match-card-emblem');
        if (emblemNode) emblemNode.classList.add('is-fallback');
        image.remove();
      }, { once: true });
    });
    if (f.isScheduleOnly) return card;
    card.addEventListener('click', () => this._showMatch(f.id));
    return card;
  },

  _matchReadQuoteUsable(f, pick, odds) {
    if (f.isScheduleOnly) return false;
    if (!f.isMatchRead) return true;
    const selected = (f.selections || []).find(s => s.pick === pick && Number(s.odds) === odds);
    if (f.status !== 'recommended' || !selected || !Number.isFinite(odds) || odds <= 1) return false;
    const expiry = selected.explanation?.expires_at;
    if (selected.justification && !expiry) return false;
    return !expiry || Date.parse(expiry) > Date.now();
  },

  _toggleSlipFromFixture(fxId) {
    const f = this.fixtures.find(x => String(x.id) === String(fxId)) || this.fixtureIndex[String(fxId)];
    if (!f || !f.best || !Number.isFinite(f.odds)) return;
    if (!this._matchReadQuoteUsable(f, f.pick, f.odds)) {
      showToast('This selection needs a current verified price.', 'error');
      return;
    }
    const slipKey = `${f.home} vs ${f.away}|${f.pick}`;
    const added = slip.add(f.home + ' vs ' + f.away, f.pick, f.odds, 'Best Pick', { fixtureId: f.id });
    document.querySelectorAll('[data-fixture-add]').forEach(btn => {
      if (btn.dataset.fixtureAdd !== String(f.id)) return;
      if (added === false) { btn.innerHTML = addButtonContent(false, false); btn.classList.remove('added'); }
      else if (added === true) { btn.innerHTML = addButtonContent(true, false); btn.classList.add('added'); }
    });
  },

  _showMatch(fxId) {
    const f = this.fixtures.find(x => String(x.id) === String(fxId)) || this.fixtureIndex[String(fxId)];
    if (!f || f.isScheduleOnly) return;
    this.currentMatch = f;
    this.currentMarketTab = Object.keys(f.markets)[0];
    const league = this.LEAGUES.find(l => l.id === f.league) || { name: f.league };

    const crest = league.logo
      ? `<span class="league-crest" aria-hidden="true"><img src="${esc(league.logo)}" alt=""></span>`
      : '';
    document.getElementById('matchHeader').innerHTML = `
      <div class="match-hero">
        <h1 class="match-teams-title">${esc(f.home)}&nbsp;<span class="match-teams-vs">v</span> ${esc(f.away)}</h1>
        <div class="match-meta">${crest}<span>${esc(league.name)}</span><span aria-hidden="true">·</span><span>${esc(f.time)}</span><span aria-hidden="true">·</span><span>${esc(this._selectedDateISO())}</span></div>
      </div>
    `;

    const summary = document.getElementById('matchReadSummary');
    if (f.isMatchRead) {
      // The Match Read itself is the public product.  Do not duplicate its
      // selections in the old all-markets panel, where users could mistake
      // two correlated fixture angles for an independently priced parlay.
      this._renderMatchReadSummary(f);
      const tabs = document.getElementById('marketTabs');
      tabs.innerHTML = '';
      tabs.classList.add('hidden');
      document.getElementById('matchMarkets').innerHTML = '';
    } else {
      if (summary) summary.innerHTML = '';
      this._buildMarketTabs(f);
      if (this.currentMarketTab) this._renderMarkets(f, this.currentMarketTab);
      else document.getElementById('matchMarkets').innerHTML = '<div class="empty-state"><p>No market decisions are available for this fixture yet.</p></div>';
    }
    this.switchTab('match');
  },

  _renderMatchReadSummary(f) {
    const container = document.getElementById('matchReadSummary');
    if (!container) return;
    if (f.recommendations) {
      this._renderRecommendationsPage(f, container);
      return;
    }
    const marketLabels = MARKET_LABELS;
    const selections = Array.isArray(f.selections) ? f.selections : [];
    const updateLabel = f.update?.is_updated && f.update?.label
      ? `<span class="match-read-update-label">${esc(f.update.label)}</span>`
      : `<span class="match-read-stage-label">${f.isPreliminary ? 'Preliminary pre-match read' : 'Pre-match read'}</span>`;
    const noBetCopy = f.status === 'no_bet'
      ? 'No selection was released: the fixture was assessed but no current price qualified.'
      : 'No selection was released because the fixture cannot be assessed safely yet.';
    // Caveats that every selection repeats word for word are shown once,
    // after the selections, instead of under each one.
    const explanationLines = selection => {
      const e = selection.explanation;
      return e ? [e.uncertainty, e.price_conditions].filter(Boolean) : [];
    };
    const sharedLines = selections.length > 1
      ? explanationLines(selections[0]).filter(line => selections.every(s => explanationLines(s).includes(line)))
      : [];
    const actionable = selections.length
      ? `<div class="match-read-selections">${selections.map((selection, index) => {
          const odds = selection.odds == null ? NaN : Number(selection.odds);
          const edge = selection.value_edge == null ? NaN : Number(selection.value_edge);
          const role = selection.role === 'core' ? 'Core' : 'Supporting';
          const market = marketLabels[selection.market?.group] || selection.market?.group || 'Market';
          const slipKey = `${f.home} vs ${f.away}|${selection.pick}`;
          const inSlip = slip.legs.find(leg => leg.key === slipKey);
          const explanation = selection.explanation;
          const restated = String(explanation?.reasoning || '').match(/^.*?@\s*\d+(?:\.\d+)?\s*\(([^)]+)\)\.\s*/);
          const bookmaker = selection.bookmaker || (restated ? restated[1] : '');
          const reasoning = restated ? explanation.reasoning.slice(restated[0].length) : explanation?.reasoning;
          const quoteFresh = !explanation?.expires_at || Date.parse(explanation.expires_at) > Date.now();
          const canAdd = f.status === 'recommended' && Number.isFinite(odds) && odds > 1 && quoteFresh;
          const addButton = canAdd
            ? `<button class="market-add-btn${inSlip ? ' added' : ''}" data-match-read-add="${index}">${addButtonContent(Boolean(inSlip), true)}</button>`
            : '';
          return `<div class="match-read-selection" data-recommendation-id="${esc(selection.recommendation_id || '')}">
            <div class="match-read-selection-main">
              <h3 class="match-read-selection-pick">${esc(market)}: ${esc(selection.pick || 'Selection unavailable')}</h3>
              <span class="match-read-role ${selection.role === 'core' ? 'core' : ''}">${esc(role)}</span>
              <span class="conf-badge ${esc(selection.confidence || 'low')}">${esc(String(selection.confidence || 'low').toUpperCase())}${selection.justification ? ' SUPPORT' : ''}</span>
              ${Number.isFinite(edge) ? `<span class="value-badge ${edge > 0.06 ? 'strong' : 'good'}">${edge >= 0 ? '+' : ''}${esc((edge * 100).toFixed(1))}% probability edge</span>` : ''}
            </div>
            <div class="match-read-selection-quote">
              <span class="match-read-price"><span class="market-line-odds mono">${Number.isFinite(odds) ? esc(String(odds)) : '—'}</span>${bookmaker ? `<span class="match-read-bookmaker">${esc(bookmaker)}</span>` : ''}</span>
              ${addButton}
            </div>
            ${explanation ? `<div class="match-read-selection-explanation">
              ${[reasoning, explanation.uncertainty, explanation.price_conditions]
                .filter(line => line && !sharedLines.includes(line))
                .map(line => `<p>${esc(line)}</p>`).join('')}
              ${!quoteFresh ? '<p>Quote expired. Wait for a current price.</p>' : ''}
            </div>` : ''}
          </div>`;
        }).join('')}</div>${sharedLines.length ? `<div class="match-read-shared"><p class="match-read-shared-title">Applies to every selection</p>${sharedLines.map(line => `<p>${esc(line)}</p>`).join('')}</div>` : ''}`
      : `<p class="match-read-no-selection">${esc(noBetCopy)}</p>`;
    container.innerHTML = `
      <section class="match-read-summary" aria-label="Published Match Read">
        <div class="match-read-summary-head">${updateLabel}</div>
        ${this._matchReadResearch(f)}
        ${actionable}
        ${f.selectionRelationship ? `<p class="match-read-relationship">${esc(f.selectionRelationship)}</p>` : ''}
      </section>
    `;
    container.querySelectorAll('[data-match-read-add]').forEach(button => {
      button.addEventListener('click', () => {
        const selection = selections[Number(button.dataset.matchReadAdd)];
        const odds = Number(selection?.odds);
        if (!selection || f.status !== 'recommended' || !Number.isFinite(odds) || odds <= 1) return;
        if (selection.explanation?.expires_at && !(Date.parse(selection.explanation.expires_at) > Date.now())) {
          button.disabled = true;
          button.textContent = 'Quote expired';
          return;
        }
        const added = slip.add(
          f.home + ' vs ' + f.away,
          selection.pick,
          odds,
          'Match Read',
          { fixtureId: f.id },
        );
        if (added === false) {
          button.innerHTML = addButtonContent(false, true);
          button.classList.remove('added');
        } else if (added === true) {
          button.innerHTML = addButtonContent(true, true);
          button.classList.add('added');
        }
      });
    });
  },

  // The read's own research: thesis, model figures and key points.
  _matchReadResearch(f) {
    // A key point is dropped when every figure in it already appears in the thesis.
    const thesisText = String(f.briefing?.summary || f.thesis || '');
    const allPoints = Array.isArray(f.briefing?.bullets) ? f.briefing.bullets : [];
    const restatesThesis = point => {
      const figures = String(point).match(/\d+(?:\.\d+)?/g) || [];
      return thesisText && figures.length > 0 && figures.every(figure => thesisText.includes(figure));
    };
    const keyPoints = allPoints.filter(point => !restatesThesis(point) && !restatesFigures(point, f.metrics));
    return `
      <p class="match-read-thesis">${esc(f.briefing?.summary || f.thesis || 'No fixture-level briefing is available yet.')}</p>
      ${figuresMarkup(f.metrics, f.home, f.away, 'match-read-figures')}
      ${keyPoints.length ? `<ul class="match-read-key-points">${keyPoints.map(point => `<li>${esc(point)}</li>`).join('')}</ul>` : ''}`;
  },

  // The recommendation layer's match page: Most likely, Value, Explore the match,
  // then the Match Read's research. Its Value list replaces the old selection list.
  _renderRecommendationsPage(f, container) {
    container.innerHTML = `
      <div class="recs" id="recsBlock"></div>
      <section class="match-read-summary" aria-labelledby="recsResearch">
        <h2 class="recs-title recs-head" id="recsResearch">The Match Read</h2>
        ${this._matchReadResearch(f)}
      </section>`;
    this._paintRecommendations(f);
    if (container.dataset.recsBound) return;
    container.dataset.recsBound = '1';
    container.addEventListener('click', event => {
      const match = this.currentMatch;
      if (!match?.recommendations) return;
      const tab = event.target.closest('[data-recs-tab]');
      const step = event.target.closest('[data-recs-step]');
      const add = event.target.closest('[data-recs-add]');
      if (tab) {
        this._recsView(match).open = tab.dataset.recsTab;
        this._paintRecommendations(match);
      } else if (step && !step.disabled) this._recsStep(match, Number(step.dataset.recsStep));
      else if (add) this._recsAdd(match, add.dataset.recsAdd);
    });
    // On phones a sideways swipe over the line moves it, like the − / + buttons.
    let touch = null;
    container.addEventListener('touchstart', event => {
      touch = event.target.closest('[data-recs-swipe]') ? event.touches[0] : null;
      if (touch) touch = { x: touch.clientX, y: touch.clientY };
    }, { passive: true });
    container.addEventListener('touchend', event => {
      if (!touch || !this.currentMatch?.recommendations) return;
      const end = event.changedTouches[0];
      const dx = end.clientX - touch.x, dy = end.clientY - touch.y;
      touch = null;
      if (Math.abs(dx) > 48 && Math.abs(dx) > Math.abs(dy) * 1.5) this._recsStep(this.currentMatch, dx < 0 ? 1 : -1);
    }, { passive: true });
  },

  _recsView(f) {
    this._recsViews = this._recsViews || {};
    return (this._recsViews[f.id] = this._recsViews[f.id] || { open: null, line: {} });
  },

  _recsSlipKey(f, leg) {
    return `${f.home} vs ${f.away}|${leg.label}`;
  },

  // Redraws the lists and the explorer, keeping keyboard focus on the same control.
  _paintRecommendations(f) {
    const host = document.getElementById('recsBlock');
    if (!host || !f.recommendations) return;
    const active = document.activeElement;
    const focusKey = active && host.contains(active)
      ? ['recsTab', 'recsStep', 'recsAdd'].find(key => active.dataset[key] !== undefined)
      : null;
    const focusValue = focusKey ? active.dataset[focusKey] : null;
    const view = { ...this._recsView(f), added: leg => slip.legs.some(l => l.key === this._recsSlipKey(f, leg)) };
    host.innerHTML = renderRecommendations(f, view) + renderMarketExplorer(f, view);
    if (!focusKey) return;
    const attr = focusKey.replace(/[A-Z]/g, c => `-${c.toLowerCase()}`);
    const target = [...host.querySelectorAll(`[data-${attr}]`)].find(el => el.dataset[focusKey] === focusValue && !el.disabled)
      || host.querySelector('[data-recs-step]:not([disabled])');
    if (target) target.focus();
  },

  _recsStep(f, delta) {
    const view = this._recsView(f);
    const markets = f.recommendations.markets || [];
    const market = markets.find(m => m.market === view.open) || markets[0];
    if (!market || RECS_TILE_MARKETS.has(market.market)) return;
    const last = recsStepperPairs(market).length - 1;
    view.line[market.market] = Math.min(Math.max(recsLineIndex(market, view.line[market.market]) + delta, 0), last);
    this._paintRecommendations(f);
  },

  _recsAdd(f, ref) {
    const block = f.recommendations;
    const [kind, first, second] = String(ref).split(':');
    let leg = null;
    if (kind === 'ml') leg = block.most_likely?.[Number(first)];
    else if (kind === 'value' && block.value?.[Number(first)]) leg = recsValueLeg(block.value[Number(first)], block);
    else if (kind === 'opt') leg = (block.markets || []).find(m => m.market === first)?.options?.[Number(second)];
    if (!leg) return;
    slip.add(`${f.home} vs ${f.away}`, leg.label, leg.typical_price ?? null, recsMarketName(leg.market), {
      fixtureId: f.id,
      leg: { market: leg.market, side: leg.side, line: leg.line, chance: leg.chance, worth: leg.worth ?? null },
      grid: block.score_grid || null,
    });
  },

  _buildMarketTabs(f) {
    const tabs = document.getElementById('marketTabs');
    tabs.innerHTML = '';
    const marketLabels = MARKET_LABELS;
    Object.keys(f.markets).forEach(mk => {
      const btn = document.createElement('button');
      btn.className = 'market-tab' + (mk === this.currentMarketTab ? ' active' : '');
      btn.textContent = marketLabels[mk] || mk;
      btn.setAttribute('aria-pressed', String(mk === this.currentMarketTab));
      btn.onclick = () => {
        this.currentMarketTab = mk;
        tabs.querySelectorAll('.market-tab').forEach(b => {
          b.classList.remove('active'); b.setAttribute('aria-pressed', 'false');
        });
        btn.classList.add('active');
        btn.setAttribute('aria-pressed', 'true');
        this._renderMarkets(f, mk);
      };
      tabs.appendChild(btn);
    });
    tabs.classList.toggle('hidden', !Object.keys(f.markets).length);
  },

  _renderMarkets(f, marketKey) {
    const lines = f.markets[marketKey] || [];
    const container = document.getElementById('matchMarkets');
    const marketLabels = MARKET_LABELS;
    if (!lines.length) {
      container.innerHTML = '<div class="empty-state"><p>This market is not available for the fixture.</p></div>';
      return;
    }
    container.innerHTML = `
      <div class="market-card">
        <div class="market-card-header">
          <span class="market-card-title">${esc(marketLabels[marketKey] || marketKey)}</span>
        </div>
        ${lines.map((line, index) => {
          const slipKey = `${f.home} vs ${f.away}|${line.pick}`;
          const inSlip = slip.legs.find(l => l.key === slipKey);
          const confidence = line.rec ? `<span class="conf-badge ${esc(line.conf)}">${esc(line.conf.toUpperCase())}</span>` : '<span class="market-no-bet">NO BET</span>';
          const edge = line.rec && line.edge !== null ? `<span class="value-badge ${line.edge > 6 ? 'strong' : 'good'}">+${esc(line.edge.toFixed(1))}%</span>` : '';
          const addButton = line.rec && Number.isFinite(line.odds)
            ? `<button class="market-add-btn${inSlip ? ' added' : ''}" data-market-add="${index}">${addButtonContent(Boolean(inSlip), true)}</button>`
            : '';
          return `<div class="market-line${line.rec ? ' recommended' : ''}">
            <div class="market-line-left">
              <span class="market-line-pick">${esc(line.pick)}</span>
              ${line.proj ? `<span class="market-line-proj">${esc(line.proj)}</span>` : ''}
              ${confidence}
              ${edge}
              ${line.reason ? `<span class="market-line-reason">${esc(line.reason)}</span>` : ''}
            </div>
            <div class="market-line-right">
              <span class="market-line-odds mono">${line.odds ? esc(String(line.odds)) : '—'}</span>
              ${addButton}
            </div>
          </div>`;
        }).join('')}
      </div>
    `;
    container.querySelectorAll('[data-market-add]').forEach(button => {
      const line = lines[Number(button.dataset.marketAdd)];
      button.addEventListener('click', () => this._toggleSlipFromMarket(f.id, line.pick, line.odds));
    });
  },

  _toggleSlipFromMarket(fxId, pick, odds) {
    const f = this.fixtures.find(x => String(x.id) === String(fxId)) || this.fixtureIndex[String(fxId)];
    if (!f || !Number.isFinite(odds)) return;
    if (!this._matchReadQuoteUsable(f, pick, odds)) {
      showToast('This selection needs a current verified price.', 'error');
      return;
    }
    const slipKey = `${f.home} vs ${f.away}|${pick}`;
    const added = slip.add(f.home + ' vs ' + f.away, pick, odds, 'Market', { fixtureId: f.id });
    document.querySelectorAll('[data-market-add]').forEach(btn => {
      if (added === false) { btn.innerHTML = addButtonContent(false, true); btn.classList.remove('added'); }
      else if (added === true) { btn.innerHTML = addButtonContent(true, true); btn.classList.add('added'); }
    });
  },

  _renderBestBets(state = {}) {
    const container = document.getElementById('bestBetsContainer');
    if (!container) return;
    if (state.loading) {
      container.innerHTML = '<div class="empty-state"><p>Ranking the clearest matchday opportunities…</p></div>';
      return;
    }
    if (state.error) {
      container.innerHTML = `<div class="error-state"><p>${esc(state.error)}</p><button class="retry-btn" onclick="appModule._loadMatchday()">Try again</button></div>`;
      return;
    }
    if (!this.bestBets.length) {
      const message = state.emptyMessage || 'No medium- or high-confidence recommendations for this selection.';
      const hint = state.emptyHint || 'No bet is a valid result when the current lines do not qualify.';
      container.innerHTML = `<div class="empty-state"><p>${esc(message)}</p><span class="empty-state-hint">${esc(hint)}</span></div>`;
      return;
    }
    const league = id => this.LEAGUES.find(l => l.id === id) || { name: id };
    const notice = state.notice
      ? `<p class="empty-state-hint" style="margin:0 0 16px">${esc(state.notice)}</p>`
      : '';
    container.innerHTML = notice + this.bestBets.map(f => {
      const slipKey = `${f.home} vs ${f.away}|${f.pick}`;
      const inSlip = slip.legs.find(l => l.key === slipKey);
      const updateLabel = f.isMatchRead && f.update?.is_updated && f.update?.label
        ? `<span class="match-read-update-label">${esc(f.update.label)}</span>`
        : '';
      const leagueInfo = league(f.league);
      const crest = leagueInfo.logo
        ? `<img src="${esc(leagueInfo.logo)}" alt="">`
        : `<span class="best-bet-crest-text" aria-hidden="true">${esc(String(leagueInfo.name || '').slice(0, 3))}</span>`;
      const market = marketLabel(f.market);
      return `<div class="fixture-row best-bet-row">
        <span class="best-bet-crest" title="${esc(leagueInfo.name)}">${crest}<span class="sr-only">${esc(leagueInfo.name)}</span></span>
        <div class="fixture-match">
          <button class="fixture-teams" type="button" data-best-bet-open aria-label="Open Match Read: ${esc(f.home)} versus ${esc(f.away)}">${esc(f.home)} <span class="fixture-vs">v</span> ${esc(f.away)}</button>
          <div class="fixture-meta-line"><span class="best-bet-pick">${market ? `<span class="best-bet-market">${esc(market)}:</span> ` : ''}<span class="best-bet-value">${esc(f.pick)}</span></span>${updateLabel}</div>
        </div>
        <span class="conf-badge ${esc(f.conf)}">${esc(f.conf.toUpperCase())}</span>
        <span class="fixture-odds-val mono">${Number.isFinite(f.odds) ? esc(String(f.odds)) : '—'}</span>
        <span class="fixture-edge mono edge-pos">${Number.isFinite(f.edge) ? `${f.edge >= 0 ? '+' : ''}${esc(f.edge.toFixed(1))}%` : '—'}</span>
        <button class="fixture-add-btn${inSlip ? ' added' : ''}" data-best-bet-add="${esc(f.id)}">${addButtonContent(Boolean(inSlip), false)}</button>
      </div>`;
    }).join('');
    // Rows open the read on click; the team-name button is the keyboard entry point.
    container.querySelectorAll('.fixture-row').forEach((row, index) => {
      const fixture = this.bestBets[index];
      row.addEventListener('click', () => this._showMatch(fixture.id));
    });
    container.querySelectorAll('[data-best-bet-add]').forEach(button => {
      button.addEventListener('click', event => {
        event.stopPropagation();
        this._toggleSlipFromFixture(button.dataset.bestBetAdd);
      });
    });
  }
};

const app = appModule;
// The match page's Add buttons follow the slip, whichever side changed it.
slip.onChange = () => {
  if (app.currentMatch?.recommendations) app._paintRecommendations(app.currentMatch);
};
document.addEventListener('research-navigate', event => {
  router.go('app', 'research', null, event.detail.route);
});

// ``/app`` is the shareable/direct Matchday Board URL served by FastAPI.
// The same HTML shell powers the landing page, so select the board explicitly
// when someone arrives there directly instead of leaving them on the home
// screen with the league controls hidden.
const initialPath = window.location.pathname.replace(/\/+$/, '') || '/';
const initialAnchor = initialPath === '/app' ? '' : location.hash;
router.go(initialPath === '/app' ? 'app' : 'landing',
  location.hash.startsWith('#research') ? 'research' : location.hash === '#best-bets' ? 'best-bets' : 'fixtures', null,
  location.hash.startsWith('#research/') ? location.hash.slice('#research/'.length) : 'players');

if (initialAnchor && document.getElementById(initialAnchor.slice(1))) {
  history.replaceState(router.state, '', '/' + location.search + initialAnchor);
  requestAnimationFrame(() => document.getElementById(initialAnchor.slice(1)).scrollIntoView({ behavior: 'instant' }));
}
