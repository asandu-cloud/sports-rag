/* ================================================================
   Spick -- Premium Betting Interface
   Stake.com-inspired dark teal design
   ================================================================ */

const API = '/api';

const LEAGUE_NAMES = {
  EPL: 'Premier League',
  LaLiga: 'La Liga',
  SerieA: 'Serie A',
  Bundesliga: 'Bundesliga',
  Ligue1: 'Ligue 1',
  UCL: 'Champions League',
  UEL: 'Europa League',
  UECL: 'Conference League',
};

const LEAGUES = ['EPL', 'LaLiga', 'SerieA', 'Bundesliga', 'Ligue1', 'UCL', 'UEL', 'UECL'];

const MARKET_LABELS = {
  goals: 'Goals Over/Under',
  corners: 'Corners Over/Under',
  cards: 'Cards Over/Under',
  sot: 'Shots on Target',
  btts: 'Both Teams to Score',
  moneyline: 'Match Winner',
  spreads: 'Handicap / Spread',
};

const MARKET_ORDER = ['moneyline', 'goals', 'corners', 'cards', 'sot', 'btts', 'spreads'];

// ================================================================
// STATE
// ================================================================

const state = {
  slip: JSON.parse(localStorage.getItem('spick_slip') || '[]'),
  slipMode: localStorage.getItem('spick_slip_mode') || 'parlay', // 'parlay' | 'single'
  slipStake: parseFloat(localStorage.getItem('spick_slip_stake') || '0') || 0,
  singleStakes: JSON.parse(localStorage.getItem('matchwise_single_stakes') || '{}'),
  activeDate: todayStr(),
  activeView: 'home',
  activeMarketTab: 'popular',
  fixtureCache: {},
  fixtureDataCache: {},
  bestBetsCache: null,
  parlayScore: null,
};

// ================================================================
// HELPERS
// ================================================================

function todayStr() {
  return new Date().toISOString().split('T')[0];
}

function formatDate(dateStr) {
  const d = new Date(dateStr + 'T12:00:00');
  return d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
}

function datePlusDays(n) {
  const d = new Date();
  d.setDate(d.getDate() + n);
  return d.toISOString().split('T')[0];
}

function formatKickoff(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
  } catch (e) {
    return '';
  }
}

function confidenceClass(conf) {
  if (!conf) return 'low';
  const c = conf.toLowerCase();
  if (c === 'high' || c === 'strong' || c === 'strong edge') return 'high';
  if (c === 'medium' || c === 'leaning') return 'medium';
  return 'low';
}

function confidenceLabel(conf) {
  const c = confidenceClass(conf);
  if (c === 'high') return 'Strong Edge';
  if (c === 'medium') return 'Leaning';
  return 'Speculative';
}

function legId(leg) {
  return `${leg.fixture}|${leg.market}|${leg.pick}`;
}

function isInSlip(leg) {
  return state.slip.some(l => legId(l) === legId(leg));
}

function saveSlip() {
  localStorage.setItem('spick_slip', JSON.stringify(state.slip));
  localStorage.setItem('spick_slip_mode', state.slipMode);
  localStorage.setItem('spick_slip_stake', String(state.slipStake));
  localStorage.setItem('matchwise_single_stakes', JSON.stringify(state.singleStakes));
}

function combinedOdds() {
  if (state.slip.length === 0) return 0;
  return state.slip.reduce((acc, l) => acc * (l.odds || 1), 1);
}

function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

// ================================================================
// TOAST
// ================================================================

function toast(msg, type = 'success') {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  $('#toastContainer').appendChild(el);
  setTimeout(() => el.remove(), 2600);
}

// ================================================================
// API
// ================================================================

async function apiFetch(path) {
  const res = await fetch(`${API}${path}`);
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json();
}

async function apiPost(path, body) {
  const res = await fetch(`${API}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json();
}

// ================================================================
// SKELETON / EMPTY / ERROR
// ================================================================

function skeletonRows(n = 4) {
  let html = '';
  for (let i = 0; i < n; i++) {
    html += `<div class="skeleton-row">
      <div class="skeleton-line w60"></div>
      <div class="skeleton-line w80 h20"></div>
      <div class="skeleton-line w40"></div>
    </div>`;
  }
  return html;
}

function errorHTML(retryFnName) {
  return `<div class="error-state">
    <p>Could not load predictions. Is the API running?</p>
    <button class="retry-btn" onclick="${retryFnName}">Retry</button>
  </div>`;
}

function emptyHTML(msg) {
  return `<div class="empty-state"><p>${msg}</p></div>`;
}

// ================================================================
// RENDER HELPERS
// ================================================================

function confBadgeHTML(confidence) {
  const cls = confidenceClass(confidence);
  return `<span class="conf-badge ${cls}">${confidenceLabel(confidence)}</span>`;
}

function edgeBadgeHTML(edge) {
  if (edge == null) return '';
  const val = typeof edge === 'number' && Math.abs(edge) < 1 ? edge * 100 : edge;
  const cls = val > 0 ? 'positive' : 'negative';
  return `<span class="edge-badge ${cls}">${val > 0 ? '+' : ''}${val.toFixed(1)}%</span>`;
}

function oddsBtnHTML(label, value, leg) {
  const hasOdds = value && value !== 0;
  const selected = leg && isInSlip(leg) ? ' selected' : '';
  const noOddsClass = !hasOdds ? ' no-odds' : '';
  const dataAttr = leg ? `data-leg="${encodeURIComponent(JSON.stringify(leg))}"` : '';
  const displayValue = hasOdds ? (typeof value === 'number' ? value.toFixed(2) : value) : 'Model';
  // Still show button for model-only picks, but style differently
  return `<button class="odds-btn${selected}${noOddsClass}" ${dataAttr} onclick="handleOddsBtn(this, event)">
    <span class="odds-label">${label}</span>
    <span class="odds-value">${displayValue}</span>
  </button>`;
}

// ================================================================
// ROUTER
// ================================================================

function navigateTo(view, params) {
  if (view === 'home') {
    window.location.hash = '/';
  } else if (view === 'best-bets') {
    window.location.hash = '/best-bets';
  } else if (view === 'match') {
    window.location.hash = `/match/${params.league}/${params.event_id}`;
  } else if (view === 'track-record') {
    window.location.hash = '/track-record';
  }
}

function handleRoute() {
  const hash = window.location.hash || '#/';
  if (hash === '#/' || hash === '#' || hash === '') {
    showView('home');
    renderHomePage();
  } else if (hash === '#/best-bets') {
    showView('best-bets');
    renderBestBetsPage();
  } else if (hash === '#/track-record') {
    showView('track-record');
  } else if (hash.startsWith('#/match/')) {
    const rest = hash.replace('#/match/', '');
    const slashIdx = rest.indexOf('/');
    if (slashIdx !== -1) {
      showView('match');
      renderMatchPage(rest.substring(0, slashIdx), rest.substring(slashIdx + 1));
    } else {
      navigateTo('home');
    }
  } else {
    showView('home');
    renderHomePage();
  }
}

function showView(view) {
  state.activeView = view;
  $$('.nav-link').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.view === view);
  });
  $('#viewHome').classList.toggle('hidden', view !== 'home');
  $('#viewBestBets').classList.toggle('hidden', view !== 'best-bets');
  $('#viewMatch').classList.toggle('hidden', view !== 'match');
  $('#viewTrackRecord').classList.toggle('hidden', view !== 'track-record');
}

// ================================================================
// DATE STRIP
// ================================================================

function renderDateStrip() {
  const container = $('#homeDateTabs');
  const days = [];
  for (let i = 0; i < 7; i++) {
    const d = datePlusDays(i);
    const label = i === 0 ? 'Today' : i === 1 ? 'Tomorrow' : formatDate(d);
    days.push({ date: d, label });
  }
  container.innerHTML = days.map(d =>
    `<button class="date-chip ${d.date === state.activeDate ? 'active' : ''}" data-date="${d.date}">${d.label}</button>`
  ).join('');
}

function switchDate(date) {
  state.activeDate = date;
  $$('#homeDateTabs .date-chip').forEach(tab => {
    tab.classList.toggle('active', tab.dataset.date === date);
  });
  renderHomePage();
}

// ================================================================
// FETCH FIXTURES
// ================================================================

async function fetchLeagueFixtures(league, date) {
  const key = `${league}:${date}`;
  if (state.fixtureCache[key]) return state.fixtureCache[key];
  const data = await apiFetch(`/fixtures/${league}/${date}`);
  state.fixtureCache[key] = data;
  state.fixtureDataCache[key] = data.fixtures || [];
  return data;
}

// ================================================================
// HOME PAGE
// ================================================================

function findBestMarket(fix) {
  const KEYS = ['goals', 'corners', 'cards', 'sot', 'btts', 'moneyline', 'spreads'];
  const confRank = { high: 3, medium: 2, low: 1 };
  let best = { pick: null, market: null, odds: null, confidence: null, edge: null };
  let bestScore = -1;

  for (const key of KEYS) {
    const m = fix[key];
    if (!m) continue;

    let pick = null;
    let odds = m.odds || null;
    let confidence = m.confidence || null;
    let edge = m.value_edge != null ? m.value_edge : null;

    if (key === 'btts' && m.recommended_side) {
      pick = `BTTS ${m.recommended_side}`;
    } else if (key === 'moneyline' && m.recommended) {
      pick = `${m.recommended} Win`;
    } else if (key === 'spreads' && m.recommended_team) {
      pick = `${m.recommended_team} ${m.recommended_line || ''}`;
    } else if (m.recommended_side) {
      const stat = key === 'goals' ? 'Goals' : key === 'corners' ? 'Corners' : key === 'cards' ? 'Cards' : key === 'sot' ? 'SoT' : '';
      pick = `${m.recommended_side} ${m.recommended_line || '?'} ${stat}`;
    }

    if (!pick) continue;
    const rank = confRank[confidence] || 0;
    const edgeAbs = edge != null ? Math.abs(edge) : 0;
    const score = rank * 100 + edgeAbs * 1000;
    if (score > bestScore) {
      bestScore = score;
      best = { pick, market: key, odds, confidence, edge };
    }
  }
  return best;
}

async function renderHomePage() {
  const container = $('#homeFixtures');
  const date = state.activeDate;
  renderDateStrip();
  container.innerHTML = skeletonRows(6);

  try {
    const results = await Promise.allSettled(
      LEAGUES.map(lg => fetchLeagueFixtures(lg, date))
    );

    let html = '';
    let total = 0;

    for (let i = 0; i < LEAGUES.length; i++) {
      const lg = LEAGUES[i];
      if (results[i].status !== 'fulfilled') continue;
      const fixtures = results[i].value.fixtures || [];
      if (fixtures.length === 0) continue;
      total += fixtures.length;

      html += `<div class="league-group">
        <div class="league-header">
          <div class="league-header-left">
            <span class="league-badge">${lg}</span>
            <span class="league-name">${LEAGUE_NAMES[lg] || lg}</span>
          </div>
          <span class="league-count">${fixtures.length} match${fixtures.length !== 1 ? 'es' : ''}</span>
        </div>
        ${fixtures.map(f => renderFixtureRow(f, lg)).join('')}
      </div>`;
    }

    container.innerHTML = html || emptyHTML(`No fixtures found for ${formatDate(date)}.`);
  } catch (err) {
    console.error('Home page fetch failed:', err);
    container.innerHTML = errorHTML('renderHomePage()');
  }
}

function renderFixtureRow(fix, league) {
  const home = fix.home_team || '?';
  const away = fix.away_team || '?';
  const fixture = `${home} vs ${away}`;
  const eventId = fix.event_id || '';
  const best = findBestMarket(fix);
  const cls = confidenceClass(best.confidence);

  // Build 1X2 odds buttons
  const ml = fix.moneyline || {};
  const homeOdds = ml.home_odds || null;
  const drawOdds = ml.draw_odds || null;
  const awayOdds = ml.away_odds || null;

  const makeLeg = (market, pick, odds) => ({
    fixture, market, pick, odds,
    confidence: best.confidence,
    league, event_id: eventId,
  });

  const homeBtn = homeOdds ? oddsBtnHTML('1', homeOdds, makeLeg('moneyline', `${home} Win`, homeOdds)) : '';
  const drawBtn = drawOdds ? oddsBtnHTML('X', drawOdds, makeLeg('moneyline', 'Draw', drawOdds)) : '';
  const awayBtn = awayOdds ? oddsBtnHTML('2', awayOdds, makeLeg('moneyline', `${away} Win`, awayOdds)) : '';

  // Count available markets
  const marketCount = MARKET_ORDER.filter(k => fix[k]).length;

  // Best pick line
  let pickHTML = '';
  if (best.pick) {
    pickHTML = `<div class="fixture-pick">
      <span class="fixture-pick-dot ${cls}"></span>
      <span class="fixture-pick-conf ${cls}">${confidenceLabel(best.confidence)}</span>
      <span class="fixture-pick-label">${best.pick}${best.odds ? ` @ ${best.odds.toFixed(2)}` : ''}</span>
      ${edgeBadgeHTML(best.edge)}
    </div>`;
  }

  return `<div class="fixture-row" data-league="${league}" data-event-id="${eventId}" onclick="handleFixtureClick(this, event)">
    <div class="fixture-teams">
      <span class="fixture-team">${home}</span>
      <span class="fixture-team">${away}</span>
    </div>
    <span class="fixture-time">${formatKickoff(fix.kickoff)}</span>
    <div class="fixture-odds" onclick="event.stopPropagation()">
      ${homeBtn}${drawBtn}${awayBtn}
    </div>
    <span class="fixture-more">+${marketCount} markets</span>
    ${pickHTML}
  </div>`;
}

function handleFixtureClick(el, e) {
  // Don't navigate if clicking on odds buttons
  if (e.target.closest('.odds-btn')) return;
  const league = el.dataset.league;
  const eventId = el.dataset.eventId;
  if (league && eventId) navigateTo('match', { league, event_id: eventId });
}

// ================================================================
// BEST BETS PAGE
// ================================================================

async function renderBestBetsPage() {
  const container = $('#bestBetsList');
  const dateEl = $('#bestBetsDate');
  const today = todayStr();
  dateEl.textContent = formatDate(today);
  container.innerHTML = skeletonRows(5);

  try {
    const data = await apiFetch(`/best-bets/${today}`);
    state.bestBetsCache = data;
    const bets = data.bets || data.picks || [];

    if (bets.length === 0) {
      container.innerHTML = emptyHTML('No best bets available for today. Check back closer to match time.');
      return;
    }

    container.innerHTML = bets.map(renderBestBetCard).join('');
  } catch (err) {
    console.error('Best bets fetch failed:', err);
    container.innerHTML = errorHTML('renderBestBetsPage()');
  }
}

function renderBestBetCard(bet) {
  const fixture = bet.fixture || `${bet.home_team || ''} vs ${bet.away_team || ''}`;
  const edge = bet.edge != null ? bet.edge : (bet.value_edge != null ? bet.value_edge * 100 : null);
  const modelProb = bet.model_prob || bet.probability || null;
  const impliedProb = bet.odds && bet.odds > 1 ? (1 / bet.odds) : null;

  const leg = {
    fixture, market: bet.market, pick: bet.pick, odds: bet.odds,
    confidence: bet.confidence, league: bet.league, event_id: bet.event_id || '',
  };

  const oddsBtn = bet.odds ? oddsBtnHTML(bet.pick, bet.odds, leg) : '';

  // Stats row
  let statsHTML = '';
  if (modelProb != null || impliedProb != null || edge != null) {
    statsHTML = '<div class="bb-stats">';
    if (modelProb != null) {
      statsHTML += `<div class="bb-stat">
        <span class="bb-stat-label">Model</span>
        <span class="bb-stat-value">${(modelProb * 100).toFixed(1)}%</span>
      </div>`;
    }
    if (impliedProb != null) {
      statsHTML += `<div class="bb-stat">
        <span class="bb-stat-label">Books</span>
        <span class="bb-stat-value">${(impliedProb * 100).toFixed(1)}%</span>
      </div>`;
    }
    if (edge != null) {
      const edgeCls = edge > 0 ? 'positive' : edge < 0 ? 'negative' : '';
      statsHTML += `<div class="bb-stat">
        <span class="bb-stat-label">Edge</span>
        <span class="bb-stat-value ${edgeCls}">${edge > 0 ? '+' : ''}${edge.toFixed(1)}%</span>
      </div>`;
    }
    statsHTML += '</div>';
  }

  return `<div class="best-bet-card">
    <div class="bb-header">
      <div class="bb-header-left">
        <span class="league-badge">${bet.league || ''}</span>
        <span class="bb-fixture">${fixture}</span>
      </div>
      <span class="bb-time">${formatKickoff(bet.kickoff) || bet.time || ''}</span>
    </div>
    <div class="bb-pick-row">
      ${confBadgeHTML(bet.confidence)}
      <span class="bb-pick-label">${bet.pick}</span>
      ${edgeBadgeHTML(edge)}
    </div>
    ${bet.reason ? `<div class="bb-reason">${bet.reason}</div>` : ''}
    ${statsHTML}
    <div class="bb-actions" onclick="event.stopPropagation()">
      ${oddsBtn}
    </div>
  </div>`;
}

// ================================================================
// MATCH DETAIL PAGE
// ================================================================

async function renderMatchPage(league, eventId) {
  const container = $('#matchMarkets');
  const header = $('#matchHeader');
  const tabsEl = $('#marketTabs');
  container.innerHTML = skeletonRows(5);
  header.innerHTML = '';
  tabsEl.innerHTML = '';

  try {
    const date = state.activeDate;
    let fix = null;

    // Search across dates
    const datesToTry = [];
    for (let i = 0; i < 7; i++) datesToTry.push(datePlusDays(i));

    for (const d of datesToTry) {
      try {
        const data = await fetchLeagueFixtures(league, d);
        fix = (data.fixtures || []).find(f => f.event_id === eventId);
        if (fix) break;
      } catch (e) { /* skip */ }
    }

    if (!fix) {
      header.innerHTML = '';
      container.innerHTML = `<div class="error-state">
        <p>Match not found.</p>
        <button class="retry-btn" onclick="navigateTo('home')">Back to Home</button>
      </div>`;
      return;
    }

    const home = fix.home_team || '?';
    const away = fix.away_team || '?';
    const fixture = `${home} vs ${away}`;

    // Render hero header
    header.innerHTML = `<div class="match-hero">
      <div class="match-hero-league">${LEAGUE_NAMES[league] || league}</div>
      <h2 class="match-hero-teams">${home} <span class="match-hero-vs">vs</span> ${away}</h2>
      <div class="match-hero-meta">${formatKickoff(fix.kickoff)}</div>
    </div>`;

    // Build all markets from the fixture data
    const marketsByGroup = {};
    for (const key of MARKET_ORDER) {
      const m = fix[key];
      if (!m) continue;
      marketsByGroup[key] = m;
    }

    const availableGroups = Object.keys(marketsByGroup);
    if (availableGroups.length === 0) {
      container.innerHTML = emptyHTML('No predictions available for this match.');
      return;
    }

    // Render market tabs
    const allTabs = [{ key: 'popular', label: 'Popular' }];
    for (const key of availableGroups) {
      allTabs.push({ key, label: MARKET_LABELS[key] || key });
    }

    state.activeMarketTab = 'popular';
    tabsEl.innerHTML = allTabs.map(t =>
      `<button class="market-tab ${t.key === 'popular' ? 'active' : ''}" data-tab="${t.key}">${t.label}</button>`
    ).join('');

    // Render markets
    renderMarketSections(marketsByGroup, fixture, league, fix.event_id);
  } catch (err) {
    console.error('Match page fetch failed:', err);
    container.innerHTML = errorHTML(`renderMatchPage('${league}','${eventId}')`);
  }
}

function renderMarketSections(marketsByGroup, fixture, league, eventId) {
  const container = $('#matchMarkets');
  const tab = state.activeMarketTab;

  let groupsToShow;
  if (tab === 'popular') {
    groupsToShow = Object.keys(marketsByGroup);
  } else {
    groupsToShow = marketsByGroup[tab] ? [tab] : [];
  }

  if (groupsToShow.length === 0) {
    container.innerHTML = emptyHTML('No data for this market type.');
    return;
  }

  let html = '';
  for (const key of groupsToShow) {
    const m = marketsByGroup[key];
    html += renderMarketSection(key, m, fixture, league, eventId);
  }
  container.innerHTML = html;
}

function renderMarketSection(key, m, fixture, league, eventId) {
  const label = MARKET_LABELS[key] || key;
  let html = `<div class="market-section">
    <div class="market-section-title">${label}</div>`;

  // Model projection
  const projTotal = m.projected_total;
  const projDiff = m.projected_diff;
  if (projTotal != null) {
    html += `<div class="model-projection">
      <span class="model-projection-label">Model projects</span>
      <span class="model-projection-value">${projTotal.toFixed(2)} total</span>
    </div>`;
  } else if (projDiff != null) {
    html += `<div class="model-projection">
      <span class="model-projection-label">Projected goal diff</span>
      <span class="model-projection-value">${projDiff >= 0 ? '+' : ''}${projDiff.toFixed(2)}</span>
    </div>`;
  }

  // Moneyline probability bar
  if (key === 'moneyline' && m.home_prob != null) {
    const hp = (m.home_prob * 100).toFixed(0);
    const dp = ((m.draw_prob || 0) * 100).toFixed(0);
    const ap = ((m.away_prob || 0) * 100).toFixed(0);
    html += `<div class="prob-bar">
      <div class="prob-bar-seg home" style="width:${hp}%">Home ${hp}%</div>
      <div class="prob-bar-seg draw" style="width:${dp}%">Draw ${dp}%</div>
      <div class="prob-bar-seg away" style="width:${ap}%">Away ${ap}%</div>
    </div>`;
  }

  // Build the recommended pick line
  let pick = null;
  let odds = m.odds || null;
  let confidence = m.confidence || null;
  let edge = m.value_edge;
  let modelProb = m.model_prob || m.probability || null;

  if (key === 'btts' && m.recommended_side) {
    pick = `BTTS ${m.recommended_side}`;
    modelProb = m.probability;
  } else if (key === 'moneyline' && m.recommended) {
    pick = `${m.recommended} Win`;
  } else if (key === 'spreads' && m.recommended_team) {
    pick = `${m.recommended_team} ${m.recommended_line || ''}`;
  } else if (m.recommended_side) {
    const stat = key === 'goals' ? 'Goals' : key === 'corners' ? 'Corners' : key === 'cards' ? 'Cards' : key === 'sot' ? 'SoT' : '';
    pick = `${m.recommended_side} ${m.recommended_line || '?'} ${stat}`;
  }

  if (pick) {
    const leg = { fixture, market: key, pick, odds, confidence, league, event_id: eventId };
    const impliedProb = odds && odds > 1 ? 1 / odds : null;

    html += `<div class="market-line" style="background: var(--accent-subtle);">
      <div class="market-line-left">
        <span class="market-line-pick" style="font-weight:700;">${pick}</span>
        ${confBadgeHTML(confidence)}
        ${edgeBadgeHTML(edge)}
      </div>
      <div class="market-line-right">`;

    // Show probability data
    if (modelProb != null) {
      html += `<span class="market-line-projection">Model ${(modelProb * 100).toFixed(0)}%</span>`;
    }
    if (impliedProb != null) {
      html += `<span class="market-line-projection">Books ${(impliedProb * 100).toFixed(0)}%</span>`;
    }

    html += oddsBtnHTML(pick, odds, leg);
    html += `</div></div>`;
  }

  // For moneyline, show all 3 outcomes as odds buttons
  if (key === 'moneyline') {
    const sides = [
      { label: 'Home', odds: m.home_odds, pickLabel: `${fixture.split(' vs ')[0]} Win` },
      { label: 'Draw', odds: m.draw_odds, pickLabel: 'Draw' },
      { label: 'Away', odds: m.away_odds, pickLabel: `${fixture.split(' vs ')[1]} Win` },
    ];
    for (const side of sides) {
      if (!side.odds) continue;
      const leg = { fixture, market: key, pick: side.pickLabel, odds: side.odds, confidence, league, event_id: eventId };
      const isRec = pick && side.pickLabel === pick;
      if (isRec) continue; // Already shown above
      html += `<div class="market-line">
        <div class="market-line-left">
          <span class="market-line-pick">${side.pickLabel}</span>
        </div>
        <div class="market-line-right">
          ${oddsBtnHTML(side.label, side.odds, leg)}
        </div>
      </div>`;
    }
  }

  html += '</div>';
  return html;
}

// ================================================================
// ODDS BUTTON CLICK HANDLER
// ================================================================

function handleOddsBtn(btn, e) {
  e.stopPropagation();
  const legData = btn.dataset.leg;
  if (!legData) return;

  const leg = JSON.parse(decodeURIComponent(legData));

  if (isInSlip(leg)) {
    removeFromSlipByLeg(leg);
    btn.classList.remove('selected');
    toast('Removed from slip', 'error');
  } else {
    addToSlip(leg);
    btn.classList.add('selected');
    // Pulse animation
    btn.classList.add('pulse');
    setTimeout(() => btn.classList.remove('pulse'), 400);
    toast('Added to slip');
  }
}

// ================================================================
// BET SLIP
// ================================================================

function addToSlip(leg) {
  if (isInSlip(leg)) return;
  state.slip.push(leg);
  state.parlayScore = null;
  saveSlip();
  renderSlip();
  updateSlipVisibility();
}

function removeFromSlip(index) {
  state.slip.splice(index, 1);
  state.parlayScore = null;
  saveSlip();
  renderSlip();
  updateSlipVisibility();
  refreshOddsButtons();
}

function removeFromSlipByLeg(leg) {
  const id = legId(leg);
  const idx = state.slip.findIndex(l => legId(l) === id);
  if (idx !== -1) {
    state.slip.splice(idx, 1);
    state.parlayScore = null;
    saveSlip();
    renderSlip();
    updateSlipVisibility();
  }
}

function clearSlip() {
  state.slip = [];
  state.parlayScore = null;
  state.slipStake = 0;
  state.singleStakes = {};
  saveSlip();
  renderSlip();
  updateSlipVisibility();
  refreshOddsButtons();
}

function updateSlipVisibility() {
  const count = state.slip.length;
  const layout = $('#layout');
  const slip = $('#betSlip');
  const indicator = $('#slipIndicator');
  const mobileBar = $('#slipMobileBar');
  const isDesktop = window.innerWidth > 1100;

  if (count > 0) {
    // Show slip
    if (isDesktop) {
      layout.classList.add('slip-open');
      slip.classList.add('open');
    }
    indicator.classList.remove('hidden');
    if (!isDesktop) {
      mobileBar.classList.remove('hidden');
    }
  } else {
    // Hide slip
    layout.classList.remove('slip-open');
    slip.classList.remove('open');
    indicator.classList.add('hidden');
    mobileBar.classList.add('hidden');
    // Close mobile backdrop if open
    $('#slipBackdrop').classList.add('hidden');
  }

  // Update indicator
  $('#slipIndicatorCount').textContent = count;
  const odds = combinedOdds();
  $('#slipIndicatorOdds').textContent = odds > 0 ? `@ ${odds.toFixed(2)}` : '';

  // Update mobile bar
  $('#slipMobileCount').textContent = count;
  $('#slipMobileOdds').textContent = odds > 0 ? odds.toFixed(2) + 'x' : '--';
}

function renderSlip() {
  const count = state.slip.length;
  const legsEl = $('#slipLegs');
  const isParlay = state.slipMode === 'parlay';

  if (count === 0) {
    legsEl.innerHTML = `<div class="empty-state" style="padding:30px 0;">
      <p>No selections yet</p>
      <p class="empty-state-hint">Click odds buttons to add picks</p>
    </div>`;
    $('#slipFoot').style.display = 'none';
    return;
  }

  $('#slipFoot').style.display = '';

  legsEl.innerHTML = state.slip.map((leg, i) => {
    let singleStakeHTML = '';
    if (!isParlay) {
      const singleStake = state.singleStakes[legId(leg)] || 0;
      const singlePayout = singleStake && leg.odds ? (singleStake * leg.odds).toFixed(2) : '0.00';
      singleStakeHTML = `<div class="slip-leg-stake-wrap">
        <input type="number" class="slip-leg-stake-input" placeholder="Stake"
               value="${singleStake || ''}" min="0" step="any"
               data-leg-index="${i}"
               oninput="handleSingleStake(this)" aria-label="Stake for ${leg.pick}">
        <span class="slip-leg-payout">$${singlePayout}</span>
      </div>`;
    }

    return `<div class="slip-leg">
      <div class="slip-leg-fixture">${leg.fixture}</div>
      <div class="slip-leg-pick">${leg.pick}</div>
      <div class="slip-leg-odds">${leg.odds ? leg.odds.toFixed(2) : 'Model only'}</div>
      ${singleStakeHTML}
      <button class="slip-leg-remove" onclick="removeFromSlip(${i})" aria-label="Remove">&times;</button>
    </div>`;
  }).join('');

  // Update footer
  const odds = combinedOdds();

  if (isParlay) {
    $('#slipParlaySection').classList.remove('hidden');
    $('#slipSingleSection').classList.add('hidden');
    $('#slipCombinedOdds').textContent = odds > 0 ? odds.toFixed(2) + 'x' : '--';
    updateParlayPayout();
    // Restore stake input
    const stakeInput = $('#slipStakeInput');
    if (stakeInput && state.slipStake > 0) {
      stakeInput.value = state.slipStake;
    }
  } else {
    $('#slipParlaySection').classList.add('hidden');
    $('#slipSingleSection').classList.remove('hidden');
  }

  // Quality
  if (state.parlayScore) {
    renderQuality(state.parlayScore);
  } else {
    $('#slipQuality').classList.add('hidden');
  }
}

function updateParlayPayout() {
  const odds = combinedOdds();
  const stake = state.slipStake || 0;
  const payout = stake * odds;
  $('#slipPayout').textContent = payout > 0 ? `$${payout.toFixed(2)}` : '$0.00';
}

function handleStakeInput(el) {
  state.slipStake = parseFloat(el.value) || 0;
  saveSlip();
  updateParlayPayout();
}

function handleSingleStake(el) {
  const idx = parseInt(el.dataset.legIndex);
  const leg = state.slip[idx];
  if (!leg) return;
  const amount = parseFloat(el.value) || 0;
  state.singleStakes[legId(leg)] = amount;
  saveSlip();
  // Update payout display
  const payoutEl = el.parentElement.querySelector('.slip-leg-payout');
  if (payoutEl && leg.odds) {
    payoutEl.textContent = `$${(amount * leg.odds).toFixed(2)}`;
  }
}

function handleQuickStake(amount) {
  state.slipStake = amount;
  const input = $('#slipStakeInput');
  if (input) input.value = amount;
  saveSlip();
  updateParlayPayout();
}

function switchSlipMode(mode) {
  state.slipMode = mode;
  $$('.slip-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.mode === mode);
    t.setAttribute('aria-selected', t.dataset.mode === mode);
  });
  saveSlip();
  renderSlip();
}

function refreshOddsButtons() {
  $$('.odds-btn').forEach(btn => {
    try {
      const leg = JSON.parse(decodeURIComponent(btn.dataset.leg));
      btn.classList.toggle('selected', isInSlip(leg));
    } catch (_) {}
  });
}

function renderQuality(score) {
  const el = $('#slipQuality');
  const fill = $('#qualityFill');
  const label = $('#qualityLabel');
  el.classList.remove('hidden');

  const val = score.score || score.quality_score || 0;
  const pct = Math.min(val * 10, 100);

  let color, text;
  if (val >= 8) { color = 'var(--accent)'; text = 'Strong'; }
  else if (val >= 6) { color = 'var(--accent)'; text = 'Good'; }
  else if (val >= 4) { color = 'var(--conf-medium)'; text = 'Fair'; }
  else { color = 'var(--negative)'; text = 'Weak'; }

  fill.style.width = pct + '%';
  fill.style.background = color;
  label.textContent = `${text} (${val.toFixed(1)}/10)`;
  label.style.color = color;
}

function _parseSideAndLine(pick) {
  // Parse "Over 2.5 Goals" → { side: "Over", line: 2.5 }
  // Parse "BTTS Yes" → { side: "Yes", line: null }
  // Parse "Arsenal Win" → { side: "Arsenal Win", line: null }
  const m = pick.match(/^(Over|Under)\s+([\d.]+)/i);
  if (m) return { side: m[1], line: parseFloat(m[2]) };
  if (/^BTTS\s+(Yes|No)/i.test(pick)) return { side: pick.replace(/^BTTS\s+/i, ''), line: null };
  return { side: pick, line: null };
}

async function scoreParlay() {
  if (state.slip.length === 0) return;
  const btn = $('#slipScoreBtn');
  btn.disabled = true;
  btn.textContent = 'Scoring...';

  try {
    // Transform frontend leg format to API format
    const apiLegs = state.slip.map(leg => {
      const parsed = _parseSideAndLine(leg.pick || '');
      return {
        event_id: leg.event_id || '',
        market: leg.market || 'goals',
        side: parsed.side,
        line: parsed.line,
        odds: leg.odds || 1.0,
      };
    });
    const data = await apiPost('/parlay/score', { legs: apiLegs });
    state.parlayScore = data;
    renderQuality(data);
    toast('Parlay scored!');
  } catch (err) {
    console.error('Parlay score failed:', err);
    toast('Could not score parlay', 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Score My Parlay';
  }
}

function shareParlay() {
  if (state.slip.length === 0) return;

  const lines = state.slip.map((l, i) =>
    `${i + 1}. ${l.fixture} -- ${l.pick}${l.odds ? ` @ ${l.odds.toFixed(2)}` : ''}`
  );
  const odds = combinedOdds();
  const qualityStr = state.parlayScore
    ? ` | Quality: ${(state.parlayScore.score || state.parlayScore.quality_score || 0).toFixed(1)}/10`
    : '';

  const text = [
    'Spick Parlay',
    '',
    ...lines,
    '',
    `Combined Odds: ${odds.toFixed(2)}x${qualityStr}`,
    '',
    'Built with Spick',
  ].join('\n');

  navigator.clipboard.writeText(text).then(() => {
    toast('Copied to clipboard!');
  }).catch(() => {
    toast('Could not copy', 'error');
  });
}

// ================================================================
// MOBILE SLIP TOGGLE
// ================================================================

function toggleMobileSlip() {
  const slip = $('#betSlip');
  const backdrop = $('#slipBackdrop');
  const isOpen = slip.classList.contains('open');

  slip.classList.toggle('open', !isOpen);
  backdrop.classList.toggle('hidden', isOpen);
}

// ================================================================
// EVENT BINDINGS
// ================================================================

function bindEvents() {
  // Nav links
  $$('.nav-link').forEach(btn => {
    btn.addEventListener('click', () => navigateTo(btn.dataset.view));
  });

  // Date strip (delegated)
  $('#homeDateTabs').addEventListener('click', e => {
    const chip = e.target.closest('.date-chip');
    if (chip) switchDate(chip.dataset.date);
  });

  // Slip controls
  $('#slipClear').addEventListener('click', clearSlip);
  $('#slipScoreBtn').addEventListener('click', scoreParlay);
  $('#slipShareBtn').addEventListener('click', shareParlay);

  // Slip mode tabs
  $$('.slip-tab').forEach(tab => {
    tab.addEventListener('click', () => switchSlipMode(tab.dataset.mode));
  });

  // Stake input
  const stakeInput = $('#slipStakeInput');
  if (stakeInput) {
    stakeInput.addEventListener('input', function () { handleStakeInput(this); });
  }

  // Quick stake buttons
  $$('.quick-stake').forEach(btn => {
    btn.addEventListener('click', () => handleQuickStake(parseFloat(btn.dataset.amount)));
  });

  // Market tabs (delegated)
  $('#marketTabs').addEventListener('click', e => {
    const tab = e.target.closest('.market-tab');
    if (!tab) return;
    state.activeMarketTab = tab.dataset.tab;
    $$('.market-tab').forEach(t => t.classList.toggle('active', t.dataset.tab === state.activeMarketTab));
    // Re-render market sections from the current match data
    const matchContainer = $('#matchMarkets');
    // Find the current match fixture to re-render
    // We'll trigger a full re-render by re-calling handleRoute
    handleRoute();
  });

  // Mobile slip backdrop
  $('#slipBackdrop').addEventListener('click', toggleMobileSlip);

  // Hash-based routing
  window.addEventListener('hashchange', handleRoute);

  // Resize handler to manage slip visibility
  window.addEventListener('resize', () => updateSlipVisibility());
}

// ================================================================
// INIT
// ================================================================

function init() {
  bindEvents();
  renderSlip();
  updateSlipVisibility();
  handleRoute();
}

document.addEventListener('DOMContentLoaded', init);
