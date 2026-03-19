/* ================================================================
   Matchwise -- Football Predictions SPA
   Neural Edge design. API-driven, zero hardcoded data.
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

const LEAGUE_COLORS = {
  EPL: '#3D195B',
  LaLiga: '#EE8707',
  SerieA: '#024494',
  Bundesliga: '#D20515',
  Ligue1: '#DAE025',
  UCL: '#001489',
  UEL: '#F37021',
  UECL: '#00FF87',
};

const DOMESTIC_LEAGUES = ['EPL', 'LaLiga', 'SerieA', 'Bundesliga', 'Ligue1'];
const ALL_LEAGUES = ['EPL', 'LaLiga', 'SerieA', 'Bundesliga', 'Ligue1', 'UCL', 'UEL', 'UECL'];

const MARKET_LABELS = {
  goals: 'Goals',
  corners: 'Corners',
  cards: 'Cards',
  sot: 'Shots on Target',
  btts: 'BTTS',
  moneyline: 'Moneyline',
  spreads: 'Handicap',
};

const MARKET_ORDER = ['moneyline', 'goals', 'corners', 'cards', 'sot', 'btts', 'spreads'];

// ================================================================
// STATE
// ================================================================

const state = {
  slip: JSON.parse(localStorage.getItem('matchwise_slip') || '[]'),
  slipStake: parseFloat(localStorage.getItem('matchwise_stake') || '0') || 0,
  activeDate: todayStr(),
  activeLeagueFilter: 'all',
  activeView: 'home',
  activeMarketTab: 'popular',
  fixtureCache: {},
  fixtureDataCache: {},
  parlayScore: null,
  // Store the current match detail data for tab switching
  currentMatchData: null,
  currentMatchMeta: null,
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

function formatDateShort(dateStr) {
  const d = new Date(dateStr + 'T12:00:00');
  const day = d.toLocaleDateString('en-US', { weekday: 'short' });
  const num = d.getDate();
  return { day, num };
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

function formatKickoffLong(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    return d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' }) +
      ', ' + d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
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
  localStorage.setItem('matchwise_slip', JSON.stringify(state.slip));
  localStorage.setItem('matchwise_stake', String(state.slipStake));
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

function toast(msg, type) {
  type = type || 'success';
  var el = document.createElement('div');
  el.className = 'toast ' + type;
  el.textContent = msg;
  $('#toastContainer').appendChild(el);
  setTimeout(function() { el.remove(); }, 2600);
}

// ================================================================
// API
// ================================================================

async function apiFetch(path) {
  const res = await fetch(API + path);
  if (!res.ok) throw new Error('API ' + res.status);
  return res.json();
}

async function apiPost(path, body) {
  const res = await fetch(API + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error('API ' + res.status);
  return res.json();
}

// ================================================================
// SKELETON / EMPTY / ERROR
// ================================================================

function skeletonRows(n) {
  n = n || 4;
  var html = '';
  for (var i = 0; i < n; i++) {
    html += '<div class="skeleton-row">' +
      '<div class="skeleton-line w60"></div>' +
      '<div class="skeleton-line w80 h20"></div>' +
      '<div class="skeleton-line w40"></div>' +
    '</div>';
  }
  return html;
}

function errorHTML(retryFnName) {
  return '<div class="error-state">' +
    '<div class="empty-state-icon"><svg width="48" height="48" viewBox="0 0 48 48" fill="none"><circle cx="24" cy="24" r="20" stroke="currentColor" stroke-width="2"/><path d="M24 16v10M24 30v2" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg></div>' +
    '<p>Could not load data. Is the API running?</p>' +
    '<button class="retry-btn" onclick="' + retryFnName + '">Retry</button>' +
  '</div>';
}

function emptyHTML(msg) {
  return '<div class="empty-state">' +
    '<div class="empty-state-icon"><svg width="48" height="48" viewBox="0 0 48 48" fill="none"><circle cx="24" cy="24" r="20" stroke="currentColor" stroke-width="1.5"/><path d="M24 14v12l8 4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg></div>' +
    '<p>' + msg + '</p>' +
    '<p class="empty-state-hint">Try selecting a different date above.</p>' +
  '</div>';
}

// ================================================================
// RENDER HELPERS
// ================================================================

function confBadgeHTML(confidence) {
  var cls = confidenceClass(confidence);
  return '<span class="conf-badge ' + cls + '">' + confidenceLabel(confidence) + '</span>';
}

function edgeBadgeHTML(edge) {
  if (edge == null) return '';
  var val = typeof edge === 'number' && Math.abs(edge) < 1 ? edge * 100 : edge;
  var cls = val > 0 ? 'positive' : 'negative';
  return '<span class="edge-badge ' + cls + '">' + (val > 0 ? '+' : '') + val.toFixed(1) + '%</span>';
}

function valueBadgeHTML(edge) {
  if (edge == null) return '';
  var absEdge = typeof edge === 'number' && Math.abs(edge) < 1 ? Math.abs(edge) * 100 : Math.abs(edge);
  if (absEdge >= 8) return '<span class="value-badge">Strong Value</span>';
  if (absEdge >= 5) return '<span class="value-badge">Good Value</span>';
  return '';
}

function oddsBtnHTML(label, value, leg) {
  var hasOdds = value && value !== 0;
  var selected = leg && isInSlip(leg) ? ' selected' : '';
  var noOddsClass = !hasOdds ? ' no-odds' : '';
  var dataAttr = leg ? ' data-leg="' + encodeURIComponent(JSON.stringify(leg)) + '"' : '';
  var displayValue = hasOdds ? (typeof value === 'number' ? value.toFixed(2) : value) : 'Model';
  return '<button class="odds-btn' + selected + noOddsClass + '"' + dataAttr + ' onclick="handleOddsBtn(this, event)">' +
    '<span class="odds-label">' + label + '</span>' +
    '<span class="odds-value">' + displayValue + '</span>' +
  '</button>';
}

function addBtnHTML(leg) {
  var inSlip = isInSlip(leg);
  var cls = inSlip ? ' added' : '';
  var label = inSlip ? 'Added' : 'Add to Slip';
  var icon = inSlip
    ? '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>'
    : '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>';
  return '<button class="bb-add-btn' + cls + '" data-leg="' + encodeURIComponent(JSON.stringify(leg)) + '" onclick="handleAddBtn(this, event)">' +
    icon + ' ' + label +
  '</button>';
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
    window.location.hash = '/match/' + params.league + '/' + params.event_id;
  }
}

function handleRoute() {
  var hash = window.location.hash || '#/';
  if (hash === '#/' || hash === '#' || hash === '') {
    showView('home');
    renderHomePage();
  } else if (hash === '#/best-bets') {
    showView('best-bets');
    renderBestBetsPage();
  } else if (hash.indexOf('#/match/') === 0) {
    var rest = hash.replace('#/match/', '');
    var slashIdx = rest.indexOf('/');
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
  $$('.nav-link').forEach(function(btn) {
    btn.classList.toggle('active', btn.dataset.view === view);
  });
  $('#viewHome').classList.toggle('hidden', view !== 'home');
  $('#viewBestBets').classList.toggle('hidden', view !== 'best-bets');
  $('#viewMatch').classList.toggle('hidden', view !== 'match');
  window.scrollTo(0, 0);
}

// ================================================================
// DATE STRIP
// ================================================================

function renderDateStrip() {
  var container = $('#homeDateTabs');
  var days = [];
  for (var i = 0; i < 7; i++) {
    var d = datePlusDays(i);
    var info = formatDateShort(d);
    var label = i === 0 ? 'Today' : i === 1 ? 'Tomorrow' : info.day;
    var num = info.num;
    days.push({ date: d, label: label, num: num });
  }
  container.innerHTML = days.map(function(d) {
    return '<button class="date-chip ' + (d.date === state.activeDate ? 'active' : '') + '" data-date="' + d.date + '">' +
      '<span>' + d.label + '</span> <span style="font-family:var(--font-mono);font-weight:700;">' + d.num + '</span>' +
    '</button>';
  }).join('');
}

function renderLeagueFilters() {
  var container = $('#leagueFilters');
  var chips = [{ code: 'all', name: 'All Leagues', color: null }];
  ALL_LEAGUES.forEach(function(code) {
    chips.push({ code: code, name: LEAGUE_NAMES[code] || code, color: LEAGUE_COLORS[code] || '#6B7394' });
  });
  container.innerHTML = chips.map(function(c) {
    var active = c.code === state.activeLeagueFilter ? ' active' : '';
    var dot = c.color ? '<span class="league-chip__dot" style="background:' + c.color + ';"></span>' : '';
    return '<button class="league-chip' + active + '" data-league="' + c.code + '">' + dot + c.name + '</button>';
  }).join('');
}

function switchDate(date) {
  state.activeDate = date;
  $$('#homeDateTabs .date-chip').forEach(function(tab) {
    tab.classList.toggle('active', tab.dataset.date === date);
  });
  renderHomePage();
}

function switchLeagueFilter(code) {
  state.activeLeagueFilter = code;
  $$('.league-chip').forEach(function(chip) {
    chip.classList.toggle('active', chip.dataset.league === code);
  });
  renderHomePage();
}

// ================================================================
// FETCH FIXTURES
// ================================================================

async function fetchLeagueFixtures(league, date) {
  var key = league + ':' + date;
  if (state.fixtureCache[key]) return state.fixtureCache[key];
  var data = await apiFetch('/fixtures/' + league + '/' + date);
  state.fixtureCache[key] = data;
  return data;
}

// ================================================================
// HOME PAGE
// ================================================================

function findBestMarket(fix) {
  var KEYS = ['goals', 'corners', 'cards', 'sot', 'btts', 'moneyline', 'spreads'];
  var confRank = { high: 3, medium: 2, low: 1 };
  var best = { pick: null, market: null, odds: null, confidence: null, edge: null };
  var bestScore = -1;

  for (var k = 0; k < KEYS.length; k++) {
    var key = KEYS[k];
    var m = fix[key];
    if (!m) continue;

    var pick = null;
    var odds = m.odds || null;
    var confidence = m.confidence || null;
    var edge = m.value_edge != null ? m.value_edge : null;

    if (key === 'btts' && m.recommended_side) {
      pick = 'BTTS ' + m.recommended_side;
    } else if (key === 'moneyline' && m.recommended) {
      pick = m.recommended + ' Win';
    } else if (key === 'spreads' && m.recommended_team) {
      pick = m.recommended_team + ' ' + (m.recommended_line || '');
    } else if (m.recommended_side) {
      var stat = key === 'goals' ? 'Goals' : key === 'corners' ? 'Corners' : key === 'cards' ? 'Cards' : key === 'sot' ? 'SoT' : '';
      pick = m.recommended_side + ' ' + (m.recommended_line || '?') + ' ' + stat;
    }

    if (!pick) continue;
    var rank = confRank[confidence] || 0;
    var edgeAbs = edge != null ? Math.abs(edge) : 0;
    var score = rank * 100 + edgeAbs * 1000;
    if (score > bestScore) {
      bestScore = score;
      best = { pick: pick, market: key, odds: odds, confidence: confidence, edge: edge };
    }
  }
  return best;
}

async function renderHomePage() {
  var container = $('#homeFixtures');
  var date = state.activeDate;
  renderDateStrip();
  renderLeagueFilters();
  container.innerHTML = skeletonRows(6);

  var leaguesToFetch = state.activeLeagueFilter === 'all' ? ALL_LEAGUES : [state.activeLeagueFilter];

  try {
    var results = await Promise.allSettled(
      leaguesToFetch.map(function(lg) { return fetchLeagueFixtures(lg, date); })
    );

    var html = '';
    var total = 0;

    for (var i = 0; i < leaguesToFetch.length; i++) {
      var lg = leaguesToFetch[i];
      if (results[i].status !== 'fulfilled') continue;
      var fixtures = results[i].value.fixtures || [];
      if (fixtures.length === 0) continue;
      total += fixtures.length;

      var leagueColor = LEAGUE_COLORS[lg] || '#6B7394';
      html += '<div class="league-group">' +
        '<div class="league-header">' +
          '<div class="league-header-left">' +
            '<span class="league-stripe" style="background:' + leagueColor + ';"></span>' +
            '<span class="league-badge">' + lg + '</span>' +
            '<span class="league-name">' + (LEAGUE_NAMES[lg] || lg) + '</span>' +
          '</div>' +
          '<span class="league-count">' + fixtures.length + ' match' + (fixtures.length !== 1 ? 'es' : '') + '</span>' +
        '</div>' +
        fixtures.map(function(f) { return renderFixtureRow(f, lg); }).join('') +
      '</div>';
    }

    container.innerHTML = html || emptyHTML('No fixtures found for ' + formatDate(date) + '.');
  } catch (err) {
    console.error('Home page fetch failed:', err);
    container.innerHTML = errorHTML('renderHomePage()');
  }
}

function renderFixtureRow(fix, league) {
  var home = fix.home_team || '?';
  var away = fix.away_team || '?';
  var fixture = home + ' vs ' + away;
  var eventId = fix.event_id || '';
  var best = fix.best_bet || findBestMarket(fix);
  var pick = best.pick || null;
  var confidence = best.confidence || null;
  var cls = confidenceClass(confidence);

  // 1X2 odds from moneyline
  var ml = fix.moneyline || {};
  var homeOdds = ml.home_odds || null;
  var drawOdds = ml.draw_odds || null;
  var awayOdds = ml.away_odds || null;

  function makeLeg(market, pickLabel, odds) {
    return { fixture: fixture, market: market, pick: pickLabel, odds: odds, confidence: confidence, league: league, event_id: eventId };
  }

  var homeBtn = homeOdds ? oddsBtnHTML('1', homeOdds, makeLeg('moneyline', home + ' Win', homeOdds)) : '';
  var drawBtn = drawOdds ? oddsBtnHTML('X', drawOdds, makeLeg('moneyline', 'Draw', drawOdds)) : '';
  var awayBtn = awayOdds ? oddsBtnHTML('2', awayOdds, makeLeg('moneyline', away + ' Win', awayOdds)) : '';

  // Count available markets
  var marketCount = MARKET_ORDER.filter(function(k) { return fix[k]; }).length;

  // Best pick line
  var pickHTML = '';
  if (pick) {
    var bestOdds = best.odds;
    pickHTML = '<div class="fixture-pick">' +
      '<span class="fixture-pick-dot ' + cls + '"></span>' +
      '<span class="fixture-pick-conf ' + cls + '">' + confidenceLabel(confidence) + '</span>' +
      '<span class="fixture-pick-label">' + pick + (bestOdds ? ' @ ' + bestOdds.toFixed(2) : '') + '</span>' +
      edgeBadgeHTML(best.edge) +
    '</div>';
  }

  return '<div class="fixture-row" data-league="' + league + '" data-event-id="' + eventId + '" onclick="handleFixtureClick(this, event)">' +
    '<div class="fixture-teams">' +
      '<span class="fixture-team">' + home + '</span>' +
      '<span class="fixture-team">' + away + '</span>' +
    '</div>' +
    '<span class="fixture-time">' + formatKickoff(fix.kickoff) + '</span>' +
    '<div class="fixture-odds" onclick="event.stopPropagation()">' +
      homeBtn + drawBtn + awayBtn +
    '</div>' +
    '<span class="fixture-more">+' + marketCount + ' markets</span>' +
    pickHTML +
  '</div>';
}

function handleFixtureClick(el, e) {
  if (e.target.closest('.odds-btn')) return;
  var league = el.dataset.league;
  var eventId = el.dataset.eventId;
  if (league && eventId) navigateTo('match', { league: league, event_id: eventId });
}

// ================================================================
// BEST BETS PAGE
// ================================================================

async function renderBestBetsPage() {
  var container = $('#bestBetsList');
  var dateEl = $('#bestBetsDate');
  var today = todayStr();
  dateEl.textContent = formatDate(today);
  container.innerHTML = skeletonRows(5);

  try {
    var data = await apiFetch('/best-bets/' + today);
    var bets = data.picks || data.bets || [];

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
  var fixture = bet.fixture || (bet.home_team || '') + ' vs ' + (bet.away_team || '');
  var edge = bet.edge != null ? bet.edge : (bet.value_edge != null ? bet.value_edge * 100 : null);
  var modelProb = bet.model_prob || bet.probability || null;
  var impliedProb = bet.odds && bet.odds > 1 ? (1 / bet.odds) : null;

  var leg = {
    fixture: fixture,
    market: bet.market,
    pick: bet.pick,
    odds: bet.odds,
    confidence: bet.confidence,
    league: bet.league,
    event_id: bet.event_id || '',
  };

  // Stats row
  var statsHTML = '';
  if (modelProb != null || impliedProb != null || edge != null) {
    statsHTML = '<div class="bb-stats">';
    if (modelProb != null) {
      statsHTML += '<div class="bb-stat">' +
        '<span class="bb-stat-label">Model</span>' +
        '<span class="bb-stat-value">' + (modelProb * 100).toFixed(1) + '%</span>' +
      '</div>';
    }
    if (impliedProb != null) {
      statsHTML += '<div class="bb-stat">' +
        '<span class="bb-stat-label">Books</span>' +
        '<span class="bb-stat-value">' + (impliedProb * 100).toFixed(1) + '%</span>' +
      '</div>';
    }
    if (bet.odds != null) {
      statsHTML += '<div class="bb-stat">' +
        '<span class="bb-stat-label">Odds</span>' +
        '<span class="bb-stat-value" style="color:var(--accent);">' + bet.odds.toFixed(2) + '</span>' +
      '</div>';
    }
    if (edge != null) {
      var edgeCls = edge > 0 ? 'positive' : edge < 0 ? 'negative' : '';
      statsHTML += '<div class="bb-stat">' +
        '<span class="bb-stat-label">Edge</span>' +
        '<span class="bb-stat-value ' + edgeCls + '">' + (edge > 0 ? '+' : '') + edge.toFixed(1) + '%</span>' +
      '</div>';
    }
    statsHTML += '</div>';
  }

  return '<div class="best-bet-card">' +
    '<div class="bb-header">' +
      '<div class="bb-header-left">' +
        '<span class="league-badge">' + (bet.league || '') + '</span>' +
        '<span class="bb-fixture">' + fixture + '</span>' +
      '</div>' +
      '<span class="bb-time">' + (formatKickoff(bet.kickoff) || '') + '</span>' +
    '</div>' +
    '<div class="bb-pick-row">' +
      confBadgeHTML(bet.confidence) +
      '<span class="bb-pick-label">' + bet.pick + '</span>' +
      edgeBadgeHTML(edge) +
      valueBadgeHTML(edge) +
    '</div>' +
    (bet.reason ? '<div class="bb-reason">' + bet.reason + '</div>' : '') +
    statsHTML +
    '<div class="bb-actions">' +
      (bet.odds ? addBtnHTML(leg) : '') +
    '</div>' +
  '</div>';
}

// ================================================================
// MATCH DETAIL PAGE
// ================================================================

async function renderMatchPage(league, eventId) {
  var container = $('#matchMarkets');
  var header = $('#matchHeader');
  var tabsEl = $('#marketTabs');
  container.innerHTML = skeletonRows(5);
  header.innerHTML = '';
  tabsEl.innerHTML = '';
  state.currentMatchData = null;
  state.currentMatchMeta = null;

  try {
    var fix = null;

    // Search across next 7 days to find this fixture
    for (var i = 0; i < 7; i++) {
      var d = datePlusDays(i);
      try {
        var data = await fetchLeagueFixtures(league, d);
        fix = (data.fixtures || []).find(function(f) { return f.event_id === eventId; });
        if (fix) break;
      } catch (e) { /* skip */ }
    }

    if (!fix) {
      header.innerHTML = '';
      container.innerHTML = '<div class="error-state">' +
        '<p>Match not found.</p>' +
        '<button class="retry-btn" onclick="navigateTo(\'home\')">Back to Fixtures</button>' +
      '</div>';
      return;
    }

    var home = fix.home_team || '?';
    var away = fix.away_team || '?';
    var fixture = home + ' vs ' + away;

    // Render hero header
    header.innerHTML = '<div class="match-hero">' +
      '<div class="match-hero-league">' + (LEAGUE_NAMES[league] || league) + '</div>' +
      '<h2 class="match-hero-teams">' + home + ' <span class="match-hero-vs">vs</span> ' + away + '</h2>' +
      '<div class="match-hero-meta">' + formatKickoffLong(fix.kickoff) + '</div>' +
    '</div>';

    // Moneyline probability bar (always shown if available)
    var ml = fix.moneyline;
    if (ml && ml.home_prob != null) {
      var hp = (ml.home_prob * 100).toFixed(0);
      var dp = ((ml.draw_prob || 0) * 100).toFixed(0);
      var ap = ((ml.away_prob || 0) * 100).toFixed(0);
      header.innerHTML += '<div style="max-width:520px;margin:0 auto 12px;">' +
        '<div class="prob-bar" style="margin:0;">' +
          '<div class="prob-bar-seg home" style="width:' + hp + '%">' + home.split(' ')[0] + ' ' + hp + '%</div>' +
          '<div class="prob-bar-seg draw" style="width:' + dp + '%">Draw ' + dp + '%</div>' +
          '<div class="prob-bar-seg away" style="width:' + ap + '%">' + away.split(' ')[0] + ' ' + ap + '%</div>' +
        '</div>' +
      '</div>';
    }

    // Build market groups
    var marketsByGroup = {};
    MARKET_ORDER.forEach(function(key) {
      if (fix[key]) marketsByGroup[key] = fix[key];
    });

    var availableGroups = Object.keys(marketsByGroup);
    if (availableGroups.length === 0) {
      container.innerHTML = emptyHTML('No predictions available for this match.');
      return;
    }

    // Store for tab switching
    state.currentMatchData = { marketsByGroup: marketsByGroup, fixture: fixture, league: league, eventId: eventId, fix: fix };
    state.currentMatchMeta = { home: home, away: away };

    // Render market tabs
    state.activeMarketTab = 'popular';
    var allTabs = [{ key: 'popular', label: 'All Markets' }];
    availableGroups.forEach(function(key) {
      allTabs.push({ key: key, label: MARKET_LABELS[key] || key });
    });

    tabsEl.innerHTML = allTabs.map(function(t) {
      return '<button class="market-tab ' + (t.key === 'popular' ? 'active' : '') + '" data-tab="' + t.key + '">' + t.label + '</button>';
    }).join('');

    renderMarketSections();
  } catch (err) {
    console.error('Match page fetch failed:', err);
    container.innerHTML = errorHTML("renderMatchPage('" + league + "','" + eventId + "')");
  }
}

function renderMarketSections() {
  if (!state.currentMatchData) return;

  var container = $('#matchMarkets');
  var d = state.currentMatchData;
  var tab = state.activeMarketTab;
  var groupsToShow;

  if (tab === 'popular') {
    groupsToShow = Object.keys(d.marketsByGroup);
  } else {
    groupsToShow = d.marketsByGroup[tab] ? [tab] : [];
  }

  if (groupsToShow.length === 0) {
    container.innerHTML = emptyHTML('No data for this market type.');
    return;
  }

  var html = '';
  groupsToShow.forEach(function(key) {
    html += renderMarketSection(key, d.marketsByGroup[key], d.fixture, d.league, d.eventId, d.fix);
  });
  container.innerHTML = html;
}

function renderMarketSection(key, m, fixture, league, eventId, fix) {
  var label = MARKET_LABELS[key] || key;
  var html = '<div class="market-section">' +
    '<div class="market-section-title">' + label + '</div>';

  // Model projection
  if (m.projected_total != null) {
    html += '<div class="model-projection">' +
      '<span class="model-projection-label">Model projects</span>' +
      '<span class="model-projection-value">' + m.projected_total.toFixed(2) + ' total</span>' +
    '</div>';
  } else if (m.projected_diff != null) {
    html += '<div class="model-projection">' +
      '<span class="model-projection-label">Projected goal diff</span>' +
      '<span class="model-projection-value">' + (m.projected_diff >= 0 ? '+' : '') + m.projected_diff.toFixed(2) + '</span>' +
    '</div>';
  } else if (key === 'btts' && m.probability != null) {
    html += '<div class="model-projection">' +
      '<span class="model-projection-label">BTTS probability</span>' +
      '<span class="model-projection-value">' + (m.probability * 100).toFixed(1) + '%</span>' +
    '</div>';
  }

  // Moneyline probability bar (inside section)
  if (key === 'moneyline' && m.home_prob != null) {
    var hp = (m.home_prob * 100).toFixed(0);
    var dp = ((m.draw_prob || 0) * 100).toFixed(0);
    var ap = ((m.away_prob || 0) * 100).toFixed(0);
    html += '<div class="prob-bar">' +
      '<div class="prob-bar-seg home" style="width:' + hp + '%">Home ' + hp + '%</div>' +
      '<div class="prob-bar-seg draw" style="width:' + dp + '%">Draw ' + dp + '%</div>' +
      '<div class="prob-bar-seg away" style="width:' + ap + '%">Away ' + ap + '%</div>' +
    '</div>';
  }

  // Build recommended pick line
  var pick = null;
  var odds = m.odds || null;
  var confidence = m.confidence || null;
  var edge = m.value_edge;
  var modelProb = m.model_prob || m.probability || null;

  if (key === 'btts' && m.recommended_side) {
    pick = 'BTTS ' + m.recommended_side;
    modelProb = m.probability;
  } else if (key === 'moneyline' && m.recommended) {
    pick = m.recommended + ' Win';
  } else if (key === 'spreads' && m.recommended_team) {
    pick = m.recommended_team + ' ' + (m.recommended_line || '');
  } else if (m.recommended_side) {
    var stat = key === 'goals' ? 'Goals' : key === 'corners' ? 'Corners' : key === 'cards' ? 'Cards' : key === 'sot' ? 'SoT' : '';
    pick = m.recommended_side + ' ' + (m.recommended_line || '?') + ' ' + stat;
  }

  if (pick) {
    var leg = { fixture: fixture, market: key, pick: pick, odds: odds, confidence: confidence, league: league, event_id: eventId };
    var impliedProb = odds && odds > 1 ? 1 / odds : null;

    html += '<div class="market-line market-line--recommended">' +
      '<div class="market-line-left">' +
        '<span class="market-line-pick" style="font-weight:700;">' + pick + '</span>' +
        confBadgeHTML(confidence) +
        edgeBadgeHTML(edge) +
        valueBadgeHTML(edge) +
      '</div>' +
      '<div class="market-line-right">';

    if (modelProb != null) {
      html += '<span class="market-line-projection">Model ' + (modelProb * 100).toFixed(0) + '%</span>';
    }
    if (impliedProb != null) {
      html += '<span class="market-line-projection">Books ' + (impliedProb * 100).toFixed(0) + '%</span>';
    }

    html += oddsBtnHTML(pick, odds, leg);
    html += '</div></div>';
  }

  // For moneyline, show all 3 outcomes
  if (key === 'moneyline') {
    var homeName = state.currentMatchMeta ? state.currentMatchMeta.home : fixture.split(' vs ')[0];
    var awayName = state.currentMatchMeta ? state.currentMatchMeta.away : fixture.split(' vs ')[1];
    var sides = [
      { label: 'Home', odds: m.home_odds, pickLabel: homeName + ' Win' },
      { label: 'Draw', odds: m.draw_odds, pickLabel: 'Draw' },
      { label: 'Away', odds: m.away_odds, pickLabel: awayName + ' Win' },
    ];
    sides.forEach(function(side) {
      if (!side.odds) return;
      var isRec = pick && side.pickLabel === pick;
      if (isRec) return;
      var sleg = { fixture: fixture, market: key, pick: side.pickLabel, odds: side.odds, confidence: confidence, league: league, event_id: eventId };
      html += '<div class="market-line">' +
        '<div class="market-line-left">' +
          '<span class="market-line-pick">' + side.pickLabel + '</span>' +
        '</div>' +
        '<div class="market-line-right">' +
          oddsBtnHTML(side.label, side.odds, sleg) +
        '</div>' +
      '</div>';
    });
  }

  // Correct Score Grid (simulated from moneyline probs)
  if (key === 'moneyline' && m.home_prob != null && m.away_prob != null) {
    html += renderCorrectScoreGrid(m.home_prob, m.draw_prob || 0, m.away_prob);
  }

  // Goal intervals for the goals market
  if (key === 'goals' && m.projected_total != null) {
    html += renderGoalIntervals(m.projected_total);
  }

  html += '</div>';
  return html;
}

// Generate top-5 scorelines from Poisson-like model
function renderCorrectScoreGrid(homeProb, drawProb, awayProb) {
  // Approximate expected goals from probabilities
  // P(0 goals) ~ exp(-lambda), so lambda ~ -ln(1 - goalProb) is a rough estimate
  // Use a simpler approach: home expected ~ 1.0-2.5 scaled by homeProb
  var homeExpected = 0.5 + homeProb * 2.0;
  var awayExpected = 0.5 + awayProb * 2.0;

  // Poisson PMF
  function poisson(k, lam) {
    var result = Math.exp(-lam);
    for (var i = 1; i <= k; i++) {
      result *= lam / i;
    }
    return result;
  }

  // Build scoreline probabilities
  var scores = [];
  for (var h = 0; h <= 5; h++) {
    for (var a = 0; a <= 5; a++) {
      var prob = poisson(h, homeExpected) * poisson(a, awayExpected);
      scores.push({ home: h, away: a, prob: prob });
    }
  }

  // Sort by probability descending
  scores.sort(function(a, b) { return b.prob - a.prob; });

  // Take top 5
  var top5 = scores.slice(0, 5);

  var html = '<div class="market-section-title" style="border-top:1px solid var(--border);">Most Likely Scorelines</div>' +
    '<div class="score-grid">';

  top5.forEach(function(s, i) {
    var topClass = i === 0 ? ' score-grid__item--top' : '';
    html += '<div class="score-grid__item' + topClass + '">' +
      '<span class="score-grid__score">' + s.home + '-' + s.away + '</span>' +
      '<span class="score-grid__prob">' + (s.prob * 100).toFixed(1) + '%</span>' +
    '</div>';
  });

  html += '</div>';
  return html;
}

// Goal interval bands from projected total
function renderGoalIntervals(projectedTotal) {
  // Poisson CDF
  function poissonCDF(k, lam) {
    var sum = 0;
    for (var i = 0; i <= k; i++) {
      var p = Math.exp(-lam);
      for (var j = 1; j <= i; j++) p *= lam / j;
      sum += p;
    }
    return sum;
  }

  function intervalProb(low, high, lam) {
    if (high === Infinity) return 1 - poissonCDF(low - 1, lam);
    return poissonCDF(high, lam) - poissonCDF(low - 1, lam);
  }

  var bands = [
    { label: '0-1', low: 0, high: 1 },
    { label: '2-3', low: 2, high: 3 },
    { label: '4-5', low: 4, high: 5 },
    { label: '6+',  low: 6, high: Infinity },
  ];

  var maxProb = 0;
  var bandData = bands.map(function(b) {
    var prob = intervalProb(b.low, b.high, projectedTotal);
    if (prob > maxProb) maxProb = prob;
    return { label: b.label, prob: prob };
  });

  var html = '<div class="market-section-title" style="border-top:1px solid var(--border);">Goal Range Probabilities</div>' +
    '<div class="interval-bars">';

  bandData.forEach(function(b) {
    var pct = (b.prob * 100).toFixed(0);
    var widthPct = maxProb > 0 ? (b.prob / maxProb * 100).toFixed(0) : 0;
    var isMax = b.prob === maxProb;
    var fillClass = isMax ? ' interval-bar__fill--recommended' : '';
    html += '<div class="interval-bar">' +
      '<span class="interval-bar__label">' + b.label + '</span>' +
      '<div class="interval-bar__track">' +
        '<div class="interval-bar__fill' + fillClass + '" style="width:' + widthPct + '%;">' +
          '<span class="interval-bar__pct">' + pct + '%</span>' +
        '</div>' +
      '</div>' +
    '</div>';
  });

  html += '</div>';
  return html;
}

// ================================================================
// ODDS / ADD BUTTON HANDLERS
// ================================================================

function handleOddsBtn(btn, e) {
  e.stopPropagation();
  var legData = btn.dataset.leg;
  if (!legData) return;

  var leg = JSON.parse(decodeURIComponent(legData));

  if (isInSlip(leg)) {
    removeFromSlipByLeg(leg);
    btn.classList.remove('selected');
    toast('Removed from slip', 'error');
  } else {
    addToSlip(leg);
    btn.classList.add('selected');
    btn.classList.add('pulse');
    setTimeout(function() { btn.classList.remove('pulse'); }, 400);
    toast('Added to slip');
  }
}

function handleAddBtn(btn, e) {
  e.stopPropagation();
  var legData = btn.dataset.leg;
  if (!legData) return;

  var leg = JSON.parse(decodeURIComponent(legData));

  if (isInSlip(leg)) {
    removeFromSlipByLeg(leg);
    btn.classList.remove('added');
    btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg> Add to Slip';
    toast('Removed from slip', 'error');
  } else {
    addToSlip(leg);
    btn.classList.add('added');
    btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg> Added';
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
  var id = legId(leg);
  var idx = state.slip.findIndex(function(l) { return legId(l) === id; });
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
  saveSlip();
  renderSlip();
  updateSlipVisibility();
  refreshOddsButtons();
}

function updateSlipVisibility() {
  var count = state.slip.length;
  var layout = $('#layout');
  var slip = $('#betSlip');
  var indicator = $('#slipIndicator');
  var mobileBar = $('#slipMobileBar');
  var isDesktop = window.innerWidth > 1100;

  if (count > 0) {
    if (isDesktop) {
      layout.classList.add('slip-open');
      slip.classList.add('open');
    }
    indicator.classList.remove('hidden');
    if (!isDesktop) {
      mobileBar.classList.remove('hidden');
    }
  } else {
    layout.classList.remove('slip-open');
    slip.classList.remove('open');
    indicator.classList.add('hidden');
    mobileBar.classList.add('hidden');
    $('#slipBackdrop').classList.add('hidden');
  }

  // Update indicator counts
  $('#slipIndicatorCount').textContent = count;
  var odds = combinedOdds();
  $('#slipIndicatorOdds').textContent = odds > 0 ? '@ ' + odds.toFixed(2) : '';

  // Update mobile bar
  $('#slipMobileCount').textContent = count;
  $('#slipMobileOdds').textContent = odds > 0 ? odds.toFixed(2) + 'x' : '--';

  // Update count badge
  $('#slipCountBadge').textContent = count;
}

function renderSlip() {
  var count = state.slip.length;
  var legsEl = $('#slipLegs');

  if (count === 0) {
    legsEl.innerHTML = '<div class="empty-state" style="padding:30px 0;">' +
      '<p>No selections yet</p>' +
      '<p class="empty-state-hint">Click odds buttons to add picks to your slip.</p>' +
    '</div>';
    $('#slipFoot').style.display = 'none';
    return;
  }

  $('#slipFoot').style.display = '';

  legsEl.innerHTML = state.slip.map(function(leg, i) {
    return '<div class="slip-leg">' +
      '<div class="slip-leg-fixture">' + leg.fixture + '</div>' +
      '<div class="slip-leg-pick">' + leg.pick + '</div>' +
      '<div class="slip-leg-odds">' + (leg.odds ? leg.odds.toFixed(2) : 'Model only') + '</div>' +
      '<button class="slip-leg-remove" onclick="removeFromSlip(' + i + ')" aria-label="Remove">&times;</button>' +
    '</div>';
  }).join('');

  // Update footer
  var odds = combinedOdds();
  $('#slipCombinedOdds').textContent = odds > 0 ? odds.toFixed(2) + 'x' : '--';
  updateParlayPayout();

  // Restore stake input
  var stakeInput = $('#slipStakeInput');
  if (stakeInput && state.slipStake > 0) {
    stakeInput.value = state.slipStake;
  }

  // Quality
  if (state.parlayScore) {
    renderQuality(state.parlayScore);
  } else {
    $('#slipQuality').classList.add('hidden');
  }
}

function updateParlayPayout() {
  var odds = combinedOdds();
  var stake = state.slipStake || 0;
  var payout = stake * odds;
  $('#slipPayout').textContent = payout > 0 ? '$' + payout.toFixed(2) : '$0.00';
}

function handleStakeInput(el) {
  state.slipStake = parseFloat(el.value) || 0;
  saveSlip();
  updateParlayPayout();
}

function handleQuickStake(amount) {
  state.slipStake = amount;
  var input = $('#slipStakeInput');
  if (input) input.value = amount;
  saveSlip();
  updateParlayPayout();
}

function refreshOddsButtons() {
  $$('.odds-btn').forEach(function(btn) {
    try {
      var leg = JSON.parse(decodeURIComponent(btn.dataset.leg));
      btn.classList.toggle('selected', isInSlip(leg));
    } catch (e) {}
  });
  $$('.bb-add-btn').forEach(function(btn) {
    try {
      var leg = JSON.parse(decodeURIComponent(btn.dataset.leg));
      var inSlip = isInSlip(leg);
      btn.classList.toggle('added', inSlip);
      if (inSlip) {
        btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg> Added';
      } else {
        btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg> Add to Slip';
      }
    } catch (e) {}
  });
}

function renderQuality(score) {
  var el = $('#slipQuality');
  var fill = $('#qualityFill');
  var label = $('#qualityLabel');
  var warningsEl = $('#qualityWarnings');
  el.classList.remove('hidden');

  var val = score.quality_score || score.score || 0;
  var pct = Math.min(val * 10, 100);
  var qualityLabel = score.quality_label || '';

  var color, text;
  if (val >= 7.5) { color = 'var(--accent)'; text = 'Strong'; }
  else if (val >= 6) { color = 'var(--accent)'; text = 'Good'; }
  else if (val >= 4.5) { color = 'var(--conf-medium)'; text = 'Fair'; }
  else { color = 'var(--negative)'; text = 'Weak'; }

  if (qualityLabel) text = qualityLabel;

  fill.style.width = pct + '%';
  fill.style.background = color;
  label.textContent = text + ' (' + val.toFixed(1) + '/10)';
  label.style.color = color;

  // Warnings
  var warnings = score.warnings || [];
  if (warnings.length > 0) {
    warningsEl.innerHTML = warnings.map(function(w) {
      return '<div style="margin-top:4px;">' + w + '</div>';
    }).join('');
  } else {
    warningsEl.innerHTML = '';
  }
}

function parseSideAndLine(pick) {
  var m = pick.match(/^(Over|Under)\s+([\d.]+)/i);
  if (m) return { side: m[1], line: parseFloat(m[2]) };
  if (/^BTTS\s+(Yes|No)/i.test(pick)) return { side: pick.replace(/^BTTS\s+/i, ''), line: null };
  return { side: pick, line: null };
}

async function scoreParlay() {
  if (state.slip.length === 0) return;
  var btn = $('#slipScoreBtn');
  btn.disabled = true;
  btn.textContent = 'Scoring...';

  try {
    var apiLegs = state.slip.map(function(leg) {
      var parsed = parseSideAndLine(leg.pick || '');
      return {
        event_id: leg.event_id || '',
        market: leg.market || 'goals',
        side: parsed.side,
        line: parsed.line,
        odds: leg.odds || 1.0,
      };
    });
    var data = await apiPost('/parlay/score', { legs: apiLegs });
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

  var lines = state.slip.map(function(l, i) {
    return (i + 1) + '. ' + l.fixture + ' -- ' + l.pick + (l.odds ? ' @ ' + l.odds.toFixed(2) : '');
  });
  var odds = combinedOdds();
  var qualityStr = state.parlayScore
    ? ' | Quality: ' + (state.parlayScore.quality_score || 0).toFixed(1) + '/10'
    : '';

  var text = [
    'Matchwise Parlay',
    '',
  ].concat(lines).concat([
    '',
    'Combined Odds: ' + odds.toFixed(2) + 'x' + qualityStr,
    '',
    'Built with Matchwise',
  ]).join('\n');

  navigator.clipboard.writeText(text).then(function() {
    toast('Copied to clipboard!');
  }).catch(function() {
    toast('Could not copy', 'error');
  });
}

// ================================================================
// MOBILE SLIP TOGGLE
// ================================================================

function toggleMobileSlip() {
  var slip = $('#betSlip');
  var backdrop = $('#slipBackdrop');
  var isOpen = slip.classList.contains('open');

  slip.classList.toggle('open', !isOpen);
  backdrop.classList.toggle('hidden', isOpen);
}

// ================================================================
// EVENT BINDINGS
// ================================================================

function bindEvents() {
  // Nav links
  $$('.nav-link').forEach(function(btn) {
    btn.addEventListener('click', function() { navigateTo(btn.dataset.view); });
  });

  // Date strip (delegated)
  $('#homeDateTabs').addEventListener('click', function(e) {
    var chip = e.target.closest('.date-chip');
    if (chip) switchDate(chip.dataset.date);
  });

  // League filters (delegated)
  $('#leagueFilters').addEventListener('click', function(e) {
    var chip = e.target.closest('.league-chip');
    if (chip) switchLeagueFilter(chip.dataset.league);
  });

  // Slip controls
  $('#slipClear').addEventListener('click', clearSlip);
  $('#slipScoreBtn').addEventListener('click', scoreParlay);
  $('#slipShareBtn').addEventListener('click', shareParlay);

  // Stake input
  var stakeInput = $('#slipStakeInput');
  if (stakeInput) {
    stakeInput.addEventListener('input', function() { handleStakeInput(this); });
  }

  // Quick stake buttons
  $$('.quick-stake').forEach(function(btn) {
    btn.addEventListener('click', function() { handleQuickStake(parseFloat(btn.dataset.amount)); });
  });

  // Market tabs (delegated)
  $('#marketTabs').addEventListener('click', function(e) {
    var tab = e.target.closest('.market-tab');
    if (!tab) return;
    state.activeMarketTab = tab.dataset.tab;
    $$('.market-tab').forEach(function(t) { t.classList.toggle('active', t.dataset.tab === state.activeMarketTab); });
    renderMarketSections();
  });

  // Mobile slip backdrop
  $('#slipBackdrop').addEventListener('click', toggleMobileSlip);

  // Hash-based routing
  window.addEventListener('hashchange', handleRoute);

  // Resize handler
  window.addEventListener('resize', function() { updateSlipVisibility(); });

  // Keyboard: Escape closes mobile slip
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
      var slip = $('#betSlip');
      if (slip.classList.contains('open') && window.innerWidth <= 1100) {
        toggleMobileSlip();
      }
    }
  });
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
