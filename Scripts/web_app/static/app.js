/* ============================================================
   Matchwise — Frontend Application
   ============================================================ */

const API = '/api';

const LEAGUE_NAMES = {
  EPL: 'Premier League',
  LaLiga: 'La Liga',
  SerieA: 'Serie A',
  Bundesliga: 'Bundesliga',
  Ligue1: 'Ligue 1',
};

const LEAGUES = ['EPL', 'LaLiga', 'SerieA', 'Bundesliga', 'Ligue1'];

const MARKET_EMOJI = {
  goals: '\u26BD',
  corners: '\uD83D\uDCD0',
  cards: '\uD83D\uDFE8',
  sot: '\uD83C\uDFAF',
  btts: '\uD83E\uDD1D',
  moneyline: '\uD83C\uDFC6',
  spreads: '\uD83D\uDCCF',
};

const MARKET_LABEL = {
  goals: 'Goals Over/Under',
  corners: 'Corners Over/Under',
  cards: 'Cards Over/Under',
  sot: 'Shots on Target',
  btts: 'Both Teams to Score',
  moneyline: 'Match Winner',
  spreads: 'Handicap / Spread',
};

// ---------- State ----------

const state = {
  slip: JSON.parse(localStorage.getItem('matchwise_slip') || '[]'),
  activeDate: todayStr(),
  activeView: 'home',
  fixtureCache: {},   // keyed by "league:date"
  bestBetsCache: null,
  parlayScore: null,
  // Stores the full fixture data per league+date so match detail can look it up
  fixtureDataCache: {},  // keyed by "league:date" -> array of raw fixture objects
};

// ---------- Helpers ----------

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
    return d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', timeZoneName: 'short' });
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
  const id = legId(leg);
  return state.slip.some(l => legId(l) === id);
}

function saveSlip() {
  localStorage.setItem('matchwise_slip', JSON.stringify(state.slip));
}

function combinedOdds() {
  if (state.slip.length === 0) return 0;
  return state.slip.reduce((acc, l) => acc * (l.odds || 1), 1);
}

function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

// ---------- Toast ----------

function toast(msg, type = 'success') {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  $('#toastContainer').appendChild(el);
  setTimeout(() => el.remove(), 2600);
}

// ---------- API ----------

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

// ---------- Skeleton ----------

function skeletonCards(n = 5) {
  let html = '';
  for (let i = 0; i < n; i++) {
    html += `
      <div class="skeleton-card">
        <div class="skeleton-line w60"></div>
        <div class="skeleton-line w80 h20"></div>
        <div class="skeleton-line w40"></div>
      </div>`;
  }
  return html;
}

function errorCard(retryFnName) {
  return `
    <div class="error-state">
      <p>Could not load predictions. Is the API running?</p>
      <button class="retry-btn" onclick="${retryFnName}">Retry</button>
    </div>`;
}

function emptyCard(msg) {
  return `<div class="empty-state"><p>${msg}</p></div>`;
}

// ---------- Render: Confidence Badge ----------

function badgeHTML(confidence) {
  const cls = confidenceClass(confidence);
  const label = confidenceLabel(confidence);
  return `<span class="confidence-badge ${cls}">${label}</span>`;
}

// ---------- Render: Add Button ----------

function addBtnHTML(leg) {
  const inSlip = isInSlip(leg);
  const data = encodeURIComponent(JSON.stringify(leg));
  const cls = inSlip ? 'add-btn added' : 'add-btn';
  const text = inSlip ? '\u2713 Added' : '+ Add';
  return `<button class="${cls}" data-leg="${data}" onclick="handleAddBtn(this)">${text}</button>`;
}

// ---------- Router ----------

function navigateTo(view, params) {
  if (view === 'home') {
    window.location.hash = '/';
  } else if (view === 'best-bets') {
    window.location.hash = '/best-bets';
  } else if (view === 'match') {
    window.location.hash = `/match/${params.league}/${params.event_id}`;
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
  } else if (hash.startsWith('#/match/')) {
    const rest = hash.replace('#/match/', '');
    const slashIdx = rest.indexOf('/');
    if (slashIdx !== -1) {
      const league = rest.substring(0, slashIdx);
      const eventId = rest.substring(slashIdx + 1);
      showView('match');
      renderMatchPage(league, eventId);
    } else {
      // Invalid match URL, go home
      navigateTo('home');
    }
  } else {
    showView('home');
    renderHomePage();
  }
}

function showView(view) {
  state.activeView = view;

  // Update nav buttons
  $$('.nav-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.view === view);
  });

  // Toggle view sections
  $('#viewHome').classList.toggle('hidden', view !== 'home');
  $('#viewBestBets').classList.toggle('hidden', view !== 'best-bets');
  $('#viewMatch').classList.toggle('hidden', view !== 'match');
}

// ---------- Date Tabs (Home) ----------

function renderHomeDateTabs() {
  const container = $('#homeDateTabs');
  const days = [];
  for (let i = 0; i < 5; i++) {
    const d = datePlusDays(i);
    const label = i === 0 ? 'Today' : i === 1 ? 'Tomorrow' : formatDate(d);
    days.push({ date: d, label });
  }

  container.innerHTML = days.map(d => `
    <button class="date-tab ${d.date === state.activeDate ? 'active' : ''}"
            data-date="${d.date}">${d.label}</button>
  `).join('');
}

function switchHomeDate(date) {
  state.activeDate = date;
  $$('#homeDateTabs .date-tab').forEach(tab => {
    tab.classList.toggle('active', tab.dataset.date === date);
  });
  renderHomePage();
}

// ---------- Fetch Fixtures for a League+Date ----------

async function fetchLeagueFixtures(league, date) {
  const cacheKey = `${league}:${date}`;
  if (state.fixtureCache[cacheKey]) {
    return state.fixtureCache[cacheKey];
  }
  const data = await apiFetch(`/fixtures/${league}/${date}`);
  state.fixtureCache[cacheKey] = data;
  // Also cache the raw fixtures for match detail lookups
  state.fixtureDataCache[cacheKey] = data.fixtures || [];
  return data;
}

// ---------- Home Page ----------

async function renderHomePage() {
  const container = $('#homeFixtures');
  const date = state.activeDate;

  renderHomeDateTabs();
  container.innerHTML = skeletonCards(8);

  try {
    // Fetch all leagues in parallel
    const results = await Promise.allSettled(
      LEAGUES.map(lg => fetchLeagueFixtures(lg, date))
    );

    let html = '';
    let totalFixtures = 0;

    for (let i = 0; i < LEAGUES.length; i++) {
      const lg = LEAGUES[i];
      if (results[i].status !== 'fulfilled') continue;
      const data = results[i].value;
      const fixtures = data.fixtures || [];
      if (fixtures.length === 0) continue;

      totalFixtures += fixtures.length;

      html += `<div class="league-section">
        <h3 class="league-section-title">
          <span class="league-badge">${lg}</span>
          ${LEAGUE_NAMES[lg] || lg}
        </h3>
        <div class="league-fixtures">
          ${fixtures.map(f => renderHomeMatchCard(f, lg)).join('')}
        </div>
      </div>`;
    }

    container.innerHTML = html || emptyCard(`No fixtures found for ${formatDate(date)} across any league.`);
  } catch (err) {
    console.error('Home page fetch failed:', err);
    container.innerHTML = errorCard('renderHomePage()');
  }
}

// ---------- Home Match Card ----------

function findBestMarket(fix) {
  // Look through all market predictions and return the one with highest confidence + edge
  const MARKET_KEYS = ['goals', 'corners', 'cards', 'sot', 'btts', 'moneyline', 'spreads'];
  const confRank = { high: 3, medium: 2, low: 1 };
  let best = { pick: null, market: null, odds: null, confidence: null, edge: null };
  let bestScore = -1;

  for (const key of MARKET_KEYS) {
    const m = fix[key];
    if (!m) continue;

    let pick = null;
    let odds = m.odds || null;
    let confidence = m.confidence || null;
    let edge = m.value_edge != null ? m.value_edge : null;

    // Build pick label
    if (key === 'btts' && m.recommended_side) {
      pick = `BTTS ${m.recommended_side}`;
    } else if (key === 'moneyline' && m.recommended) {
      pick = `${m.recommended} Win`;
    } else if (key === 'spreads' && m.recommended_team) {
      pick = `${m.recommended_team} ${m.recommended_line || ''}`;
    } else if (m.recommended_side) {
      const statLabel = key === 'goals' ? 'Goals' : key === 'corners' ? 'Corners' : key === 'cards' ? 'Cards' : key === 'sot' ? 'SoT' : '';
      pick = `${m.recommended_side} ${m.recommended_line || '?'} ${statLabel}`;
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

function renderHomeMatchCard(fix, league) {
  const homeTeam = fix.home_team || '?';
  const awayTeam = fix.away_team || '?';
  const fixture = `${homeTeam} vs ${awayTeam}`;
  const eventId = fix.event_id || '';
  const best = findBestMarket(fix);
  const cls = confidenceClass(best.confidence);

  const leg = {
    fixture: fixture,
    market: best.market,
    pick: best.pick,
    odds: best.odds,
    confidence: best.confidence,
    league: league,
    event_id: eventId,
  };

  // Use a data attribute for the click handler to avoid escaping issues
  return `
    <div class="home-match-card confidence-${cls}"
         data-league="${league}" data-event-id="${eventId}"
         onclick="handleMatchCardClick(this)">
      <div class="hmc-teams">
        <span class="hmc-home">${homeTeam}</span>
        <span class="hmc-vs">vs</span>
        <span class="hmc-away">${awayTeam}</span>
      </div>
      <div class="hmc-time">${formatKickoff(fix.kickoff)}</div>
      ${best.pick ? `
        <div class="hmc-pick">
          ${badgeHTML(best.confidence)}
          <span class="pick-label">${best.pick}</span>
          ${best.odds ? `<span class="pick-odds">@ ${best.odds.toFixed(2)}</span>` : ''}
        </div>
      ` : '<div class="hmc-pick"></div>'}
      <div class="hmc-add" onclick="event.stopPropagation()">
        ${best.pick ? addBtnHTML(leg) : ''}
      </div>
    </div>`;
}

function handleMatchCardClick(el) {
  const league = el.dataset.league;
  const eventId = el.dataset.eventId;
  if (league && eventId) {
    navigateTo('match', { league, event_id: eventId });
  }
}

// ---------- Best Bets Page ----------

function renderBestBetCard(bet) {
  const fixture = bet.fixture || `${bet.home_team || ''} vs ${bet.away_team || ''}`;
  const edge = bet.edge != null ? bet.edge : (bet.value_edge != null ? bet.value_edge * 100 : null);
  const leg = {
    fixture: fixture,
    market: bet.market,
    pick: bet.pick,
    odds: bet.odds,
    confidence: bet.confidence,
    league: bet.league,
    event_id: bet.event_id || '',
  };
  const cls = confidenceClass(bet.confidence);
  const oddsStr = bet.odds ? `@ ${bet.odds.toFixed(2)}` : 'Model only';
  const edgeStr = edge != null ? `Edge: ${edge > 0 ? '+' : ''}${edge.toFixed(1)}%` : '';

  return `
    <div class="best-bet-card confidence-${cls}">
      <div class="bb-top">
        <span class="card-league">${bet.league || ''}</span>
        <span class="bb-fixture">${fixture}</span>
        <span class="bb-time">${formatKickoff(bet.kickoff) || bet.time || ''}</span>
      </div>
      <div class="bb-pick">
        ${badgeHTML(bet.confidence)}
        <span class="pick-label">${bet.pick}</span>
        <span class="pick-odds">${oddsStr}</span>
        ${edgeStr ? `<span class="pick-edge">${edgeStr}</span>` : ''}
      </div>
      ${bet.reason ? `<div class="bb-reason">${bet.reason}</div>` : ''}
      <div class="bb-bottom">
        ${addBtnHTML(leg)}
      </div>
    </div>`;
}

async function renderBestBetsPage() {
  const container = $('#bestBetsList');
  const dateEl = $('#bestBetsDate');
  const today = todayStr();
  dateEl.textContent = formatDate(today);

  container.innerHTML = skeletonCards(6);

  try {
    const data = await apiFetch(`/best-bets/${today}`);
    state.bestBetsCache = data;

    const bets = data.bets || data.picks || [];
    if (bets.length === 0) {
      container.innerHTML = emptyCard('No best bets available for today. Check back closer to match time.');
      return;
    }

    container.innerHTML = bets.map(renderBestBetCard).join('');
  } catch (err) {
    console.error('Best bets fetch failed:', err);
    container.innerHTML = errorCard('renderBestBetsPage()');
  }
}

// ---------- Match Detail Page ----------

function extractAllMarkets(fix, league) {
  const MARKET_KEYS = ['goals', 'corners', 'cards', 'sot', 'btts', 'moneyline', 'spreads'];
  const markets = [];

  for (const key of MARKET_KEYS) {
    const m = fix[key];
    if (!m) continue;

    let pick = null;
    let odds = m.odds || null;
    let confidence = m.confidence || null;
    let edge = m.value_edge != null ? m.value_edge : null;
    let modelProb = m.model_prob || m.probability || null;
    let impliedProb = null;
    let projectedTotal = m.projected_total || null;
    let bookmaker = m.bookmaker || null;

    // Compute implied probability from odds
    if (odds && odds > 1) {
      impliedProb = 1 / odds;
    }

    // Build pick label
    if (key === 'btts' && m.recommended_side) {
      pick = `BTTS ${m.recommended_side}`;
      modelProb = m.probability;
    } else if (key === 'moneyline' && m.recommended) {
      pick = `${m.recommended} Win`;
    } else if (key === 'spreads' && m.recommended_team) {
      pick = `${m.recommended_team} ${m.recommended_line || ''}`;
      projectedTotal = m.projected_diff;
    } else if (m.recommended_side) {
      const statLabel = key === 'goals' ? 'Goals' : key === 'corners' ? 'Corners' : key === 'cards' ? 'Cards' : key === 'sot' ? 'SoT' : '';
      pick = `${m.recommended_side} ${m.recommended_line || '?'} ${statLabel}`;
    }

    if (!pick) continue;

    const market = {
      group: key,
      pick,
      odds,
      confidence,
      edge,
      model_prob: modelProb,
      implied_prob: impliedProb,
      projected_total: projectedTotal,
      bookmaker,
      event_id: fix.event_id || '',
      league,
      // Extra data for moneyline bar
      home_prob: m.home_prob || null,
      draw_prob: m.draw_prob || null,
      away_prob: m.away_prob || null,
    };

    markets.push(market);
  }

  return markets;
}

function renderDetailMarketCard(market, fixture) {
  const emoji = MARKET_EMOJI[market.group] || '';
  const label = MARKET_LABEL[market.group] || market.group;
  const cls = confidenceClass(market.confidence);

  const leg = {
    fixture,
    market: market.group,
    pick: market.pick,
    odds: market.odds,
    confidence: market.confidence,
    league: market.league,
    event_id: market.event_id || '',
  };

  const oddsStr = market.odds ? `@ ${market.odds.toFixed(2)}` : 'Model only';
  const bookmakerStr = market.bookmaker ? `(${market.bookmaker})` : '';

  // Build stat rows
  let statsHTML = '';

  if (market.model_prob != null) {
    statsHTML += `<div class="stat-row">
      <span class="stat-label">Model probability</span>
      <span class="stat-value">${(market.model_prob * 100).toFixed(1)}%</span>
    </div>`;
  }

  if (market.implied_prob != null && market.odds) {
    statsHTML += `<div class="stat-row">
      <span class="stat-label">Books imply</span>
      <span class="stat-value">${(market.implied_prob * 100).toFixed(1)}%</span>
    </div>`;
  }

  if (market.edge != null) {
    const edgeVal = market.edge * 100;
    const edgeCls = edgeVal > 0 ? 'positive' : edgeVal < 0 ? 'negative' : 'neutral';
    statsHTML += `<div class="stat-row">
      <span class="stat-label">Value edge</span>
      <span class="stat-value ${edgeCls}">${edgeVal > 0 ? '+' : ''}${edgeVal.toFixed(1)}%</span>
    </div>`;
  }

  if (market.projected_total != null) {
    const projLabel = market.group === 'spreads' ? 'Projected goal diff' : `Projected total`;
    const projVal = market.group === 'spreads'
      ? (market.projected_total >= 0 ? '+' : '') + market.projected_total.toFixed(2)
      : market.projected_total.toFixed(1);
    statsHTML += `<div class="stat-row">
      <span class="stat-label">${projLabel}</span>
      <span class="stat-value">${projVal}</span>
    </div>`;
  }

  // Moneyline bar
  let moneylineBarHTML = '';
  if (market.group === 'moneyline' && market.home_prob != null) {
    const hp = (market.home_prob * 100).toFixed(0);
    const dp = ((market.draw_prob || 0) * 100).toFixed(0);
    const ap = ((market.away_prob || 0) * 100).toFixed(0);
    moneylineBarHTML = `
      <div class="moneyline-bar">
        <div class="ml-segment home" style="width: ${hp}%">Home ${hp}%</div>
        <div class="ml-segment draw" style="width: ${dp}%">Draw ${dp}%</div>
        <div class="ml-segment away" style="width: ${ap}%">Away ${ap}%</div>
      </div>`;
  }

  return `
    <div class="detail-market-card">
      <div class="dmc-header">
        <span class="market-emoji">${emoji}</span>
        <span class="market-name">${label}</span>
        ${badgeHTML(market.confidence)}
      </div>
      <div class="dmc-pick">
        <span class="dmc-pick-label">${market.pick}</span>
        <span class="dmc-odds">${oddsStr}</span>
        ${bookmakerStr ? `<span class="dmc-bookmaker">${bookmakerStr}</span>` : ''}
      </div>
      ${moneylineBarHTML}
      ${statsHTML ? `<div class="dmc-stats">${statsHTML}</div>` : ''}
      <div class="dmc-add">
        ${addBtnHTML(leg)}
      </div>
    </div>`;
}

async function renderMatchPage(league, eventId) {
  const container = $('#matchMarkets');
  const header = $('#matchHeader');
  container.innerHTML = skeletonCards(6);
  header.innerHTML = '';

  try {
    // Try to find fixture in cache first, otherwise fetch
    const date = state.activeDate;
    const data = await fetchLeagueFixtures(league, date);
    let fix = (data.fixtures || []).find(f => f.event_id === eventId);

    // If not found in active date, try today and nearby dates
    if (!fix) {
      const datesToTry = [];
      for (let i = 0; i < 5; i++) {
        const d = datePlusDays(i);
        if (d !== date) datesToTry.push(d);
      }
      for (const d of datesToTry) {
        try {
          const altData = await fetchLeagueFixtures(league, d);
          fix = (altData.fixtures || []).find(f => f.event_id === eventId);
          if (fix) break;
        } catch (e) { /* skip */ }
      }
    }

    if (!fix) {
      header.innerHTML = '';
      container.innerHTML = `
        <div class="error-state">
          <p>Match not found. It may have already started or the data is unavailable.</p>
          <button class="retry-btn" onclick="navigateTo('home')">Back to Home</button>
        </div>`;
      return;
    }

    const homeTeam = fix.home_team || '?';
    const awayTeam = fix.away_team || '?';
    const fixture = `${homeTeam} vs ${awayTeam}`;

    // Render header
    header.innerHTML = `
      <div class="match-hero">
        <span class="match-league-badge">${LEAGUE_NAMES[league] || league}</span>
        <h2 class="match-teams">${homeTeam} vs ${awayTeam}</h2>
        <span class="match-kickoff">${formatKickoff(fix.kickoff)}</span>
      </div>`;

    // Render all markets
    const markets = extractAllMarkets(fix, league);
    if (markets.length > 0) {
      container.innerHTML = markets.map(m => renderDetailMarketCard(m, fixture)).join('');
    } else {
      container.innerHTML = emptyCard('No predictions available for this match.');
    }
  } catch (err) {
    console.error('Match page fetch failed:', err);
    container.innerHTML = errorCard(`renderMatchPage('${league}','${eventId}')`);
  }
}

// ---------- Bet Slip ----------

function handleAddBtn(btn) {
  const leg = JSON.parse(decodeURIComponent(btn.dataset.leg));
  if (isInSlip(leg)) {
    removeFromSlipByLeg(leg);
    btn.classList.remove('added');
    btn.textContent = '+ Add';
    toast('Removed from slip', 'error');
  } else {
    addToSlip(leg);
    btn.classList.add('added');
    btn.textContent = '\u2713 Added';
    toast('Added to slip');
  }
}

function addToSlip(leg) {
  if (isInSlip(leg)) return;
  state.slip.push(leg);
  state.parlayScore = null;
  saveSlip();
  renderSlip();
}

function removeFromSlip(index) {
  state.slip.splice(index, 1);
  state.parlayScore = null;
  saveSlip();
  renderSlip();
  refreshAddButtons();
}

function removeFromSlipByLeg(leg) {
  const id = legId(leg);
  const idx = state.slip.findIndex(l => legId(l) === id);
  if (idx !== -1) {
    state.slip.splice(idx, 1);
    state.parlayScore = null;
    saveSlip();
    renderSlip();
  }
}

function clearSlip() {
  state.slip = [];
  state.parlayScore = null;
  saveSlip();
  renderSlip();
  refreshAddButtons();
}

function refreshAddButtons() {
  $$('.add-btn').forEach(btn => {
    try {
      const leg = JSON.parse(decodeURIComponent(btn.dataset.leg));
      if (isInSlip(leg)) {
        btn.classList.add('added');
        btn.textContent = '\u2713 Added';
      } else {
        btn.classList.remove('added');
        btn.textContent = '+ Add';
      }
    } catch (_) {}
  });
}

function renderSlip() {
  const count = state.slip.length;
  const countEls = [$('#slipCount'), $('#slipBadgeNav')];
  countEls.forEach(el => { if (el) el.textContent = count; });
  $('#slipCount').textContent = `(${count})`;

  const emptyEl = $('#slipEmpty');
  const legsEl = $('#slipLegs');
  const footerEl = $('#slipFooter');

  if (count === 0) {
    emptyEl.classList.remove('hidden');
    legsEl.innerHTML = '';
    footerEl.style.display = 'none';
    return;
  }

  emptyEl.classList.add('hidden');
  footerEl.style.display = '';

  legsEl.innerHTML = state.slip.map((leg, i) => `
    <div class="slip-leg">
      <div class="slip-leg-fixture">${leg.fixture}</div>
      <div class="slip-leg-pick">${leg.pick}</div>
      <div class="slip-leg-odds">${leg.odds ? `@ ${leg.odds.toFixed(2)}` : 'Model only'}</div>
      <div class="slip-leg-badge">${badgeHTML(leg.confidence)}</div>
      <button class="slip-leg-remove" onclick="removeFromSlip(${i})" aria-label="Remove leg">\u00D7</button>
    </div>
  `).join('');

  const odds = combinedOdds();
  $('#slipOdds').textContent = odds > 0 ? odds.toFixed(2) + 'x' : '--';
  $('#slipPayout').textContent = odds > 0 ? '$' + (10 * odds).toFixed(2) : '--';

  // Quality display
  if (state.parlayScore) {
    renderQuality(state.parlayScore);
  } else {
    $('#slipQuality').classList.add('hidden');
  }
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
  else if (val >= 4) { color = 'var(--warning)'; text = 'Fair'; }
  else { color = 'var(--negative)'; text = 'Weak'; }

  fill.style.width = pct + '%';
  fill.style.background = color;
  label.textContent = `${text} (${val.toFixed(1)}/10)`;
  label.style.color = color;
}

async function scoreParlay() {
  if (state.slip.length === 0) return;

  const btn = $('#slipScoreBtn');
  btn.disabled = true;
  btn.textContent = 'Scoring...';

  try {
    const data = await apiPost('/parlay/score', { legs: state.slip });
    state.parlayScore = data;
    renderQuality(data);
    toast('Parlay scored!');
  } catch (err) {
    console.error('Parlay score failed:', err);
    toast('Could not score parlay. Is the API running?', 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Score My Parlay';
  }
}

function shareParlay() {
  if (state.slip.length === 0) return;

  const lines = state.slip.map((l, i) =>
    `${i + 1}. ${l.fixture} \u2014 ${l.pick}${l.odds ? ` @ ${l.odds.toFixed(2)}` : ''}`
  );
  const odds = combinedOdds();
  const qualityStr = state.parlayScore
    ? ` | Quality: ${(state.parlayScore.score || state.parlayScore.quality_score || 0).toFixed(1)}/10`
    : '';

  const text = [
    '\uD83C\uDFAF Matchwise Parlay',
    '',
    ...lines,
    '',
    `Combined Odds: ${odds.toFixed(2)}x${qualityStr}`,
    '',
    'Built with Matchwise',
  ].join('\n');

  navigator.clipboard.writeText(text).then(() => {
    toast('Parlay copied to clipboard!');
  }).catch(() => {
    toast('Could not copy to clipboard', 'error');
  });
}

// ---------- Mobile Slip Toggle ----------

function toggleMobileSlip() {
  const slip = $('#betSlip');
  const backdrop = $('#slipBackdrop');
  const isOpen = slip.classList.contains('open');

  slip.classList.toggle('open', !isOpen);
  backdrop.classList.toggle('hidden', isOpen);
}

// ---------- Event Bindings ----------

function bindEvents() {
  // Nav buttons
  $$('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      navigateTo(btn.dataset.view);
    });
  });

  // Home date tabs (delegated)
  $('#homeDateTabs').addEventListener('click', e => {
    const tab = e.target.closest('.date-tab');
    if (tab) switchHomeDate(tab.dataset.date);
  });

  // Slip controls
  $('#slipClear').addEventListener('click', clearSlip);
  $('#slipScoreBtn').addEventListener('click', scoreParlay);
  $('#slipShareBtn').addEventListener('click', shareParlay);

  // Mobile slip toggle
  $('#slipToggleMobile').addEventListener('click', toggleMobileSlip);
  $('#slipBackdrop').addEventListener('click', toggleMobileSlip);

  // Hash-based routing
  window.addEventListener('hashchange', handleRoute);
}

// ---------- Init ----------

function init() {
  bindEvents();
  renderSlip();
  handleRoute();
}

document.addEventListener('DOMContentLoaded', init);
