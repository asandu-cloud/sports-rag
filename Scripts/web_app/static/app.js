/* ================================================================
   SpickBot -- SPA Application
   Landing page + predictions dashboard. Vanilla JS.
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

const DOMESTIC = ['EPL', 'LaLiga', 'SerieA', 'Bundesliga', 'Ligue1'];
const ALL_LEAGUES = [...DOMESTIC, 'UCL', 'UEL', 'UECL'];

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
// Utilities
// ================================================================

function $(id) { return document.getElementById(id); }
function qs(sel, parent) { return (parent || document).querySelector(sel); }
function qsa(sel, parent) { return (parent || document).querySelectorAll(sel); }

function todayStr() {
  return new Date().toISOString().slice(0, 10);
}

function dateAdd(base, days) {
  const d = new Date(base);
  d.setDate(d.getDate() + days);
  return d.toISOString().slice(0, 10);
}

function formatDateShort(dateStr) {
  const d = new Date(dateStr + 'T12:00:00');
  const today = todayStr();
  if (dateStr === today) return 'Today';
  if (dateStr === dateAdd(today, 1)) return 'Tomorrow';
  return d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
}

function formatTime(isoStr) {
  if (!isoStr) return '';
  try {
    return new Date(isoStr).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: true });
  } catch { return ''; }
}

function pct(v) {
  if (v == null) return '--';
  return (v * 100).toFixed(1) + '%';
}

function edgeFmt(v) {
  if (v == null) return '--';
  const abs = Math.abs(v * 100);
  return (v >= 0 ? '+' : '-') + abs.toFixed(1) + '%';
}

function oddsFmt(v) {
  if (v == null) return '--';
  return v.toFixed(2);
}

function confClass(c) {
  if (!c) return 'low';
  return c.toLowerCase();
}

function valueBadge(edge) {
  if (edge == null) return '';
  const abs = Math.abs(edge);
  if (abs >= 0.08) return '<span class="value-badge strong">Strong</span>';
  if (abs >= 0.03) return '<span class="value-badge good">Good</span>';
  return '';
}

function edgeClass(v) {
  if (v == null) return '';
  return v >= 0 ? 'edge-pos' : '';
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}


// ================================================================
// Toast
// ================================================================

function showToast(msg, type = 'success') {
  const c = $('toastContainer');
  const t = document.createElement('div');
  t.className = 'toast ' + type;
  t.textContent = msg;
  c.appendChild(t);
  setTimeout(() => t.remove(), 2600);
}


// ================================================================
// FAQ accordion (landing page)
// ================================================================

function toggleFaq(btn) {
  const expanded = btn.getAttribute('aria-expanded') === 'true';
  btn.setAttribute('aria-expanded', String(!expanded));
  const answer = btn.nextElementSibling;
  answer.classList.toggle('open', !expanded);
}


// ================================================================
// Router (landing <-> app)
// ================================================================

const router = {
  current: 'landing',

  go(page) {
    if (page === this.current) return;
    this.current = page;

    const landing = $('landingPage');
    const appPage = $('appPage');
    const landingNavs = qsa('.landing-nav');
    const appNavs = qsa('.app-nav');

    if (page === 'app') {
      landing.classList.add('hidden');
      appPage.classList.remove('hidden');
      landingNavs.forEach(el => el.classList.add('hidden'));
      appNavs.forEach(el => el.classList.remove('hidden'));
      document.title = 'SpickBot - Dashboard';
      window.scrollTo(0, 0);
      app.init();
    } else {
      appPage.classList.add('hidden');
      landing.classList.remove('hidden');
      landingNavs.forEach(el => el.classList.remove('hidden'));
      appNavs.forEach(el => el.classList.add('hidden'));
      document.title = 'SpickBot - AI Football Predictions';
      window.scrollTo(0, 0);
    }

    history.pushState({ page }, '', page === 'app' ? '/app' : '/');
  },

  init() {
    window.addEventListener('popstate', (e) => {
      const page = (e.state && e.state.page) || 'landing';
      router.go(page);
    });

    // Check initial URL
    if (location.pathname === '/app') {
      this.go('app');
    }
  }
};


// ================================================================
// AUTH MODULE
// ================================================================

const auth = {
  user: null,
  token: localStorage.getItem('spickbot_token'),

  async init() {
    // Check for token in URL (redirect from OAuth callback)
    const params = new URLSearchParams(window.location.search);
    const urlToken = params.get('token');
    if (urlToken) {
      this.token = urlToken;
      localStorage.setItem('spickbot_token', urlToken);
      // Clean URL
      window.history.replaceState({}, '', window.location.pathname);
    }
    if (!this.token) return;
    // Validate token with server
    try {
      const resp = await fetch('/auth/me', {
        headers: { 'Authorization': `Bearer ${this.token}` },
      });
      if (resp.ok) {
        this.user = await resp.json();
        this.updateUI();
      } else {
        this.logout();
      }
    } catch (e) {
      console.warn('Auth check failed:', e);
    }
  },

  getHeaders() {
    const h = { 'Content-Type': 'application/json' };
    if (this.token) h['Authorization'] = `Bearer ${this.token}`;
    return h;
  },

  isLoggedIn() { return !!this.user; },

  tier() { return this.user?.tier || 'free'; },

  canAccess(minTier) {
    const hierarchy = { free: 0, starter: 1, pro: 2, elite: 3 };
    return (hierarchy[this.tier()] || 0) >= (hierarchy[minTier] || 0);
  },

  login() { window.location.href = '/auth/discord/login'; },

  logout() {
    this.user = null;
    this.token = null;
    localStorage.removeItem('spickbot_token');
    this.updateUI();
  },

  updateUI() {
    const loginBtn = document.getElementById('loginBtn');
    const userBadge = document.getElementById('userBadge');
    const userName = document.getElementById('userName');
    const userAvatar = document.getElementById('userAvatar');

    if (this.user) {
      if (loginBtn) loginBtn.classList.add('hidden');
      if (userBadge) {
        userBadge.classList.remove('hidden');
        userName.textContent = this.user.username || 'User';
        const discordId = this.user.discord_id || '';
        const avatarHash = this.user.discord_avatar_url || '';
        if (avatarHash && discordId) {
          userAvatar.src = avatarHash.startsWith('http')
            ? avatarHash
            : `https://cdn.discordapp.com/avatars/${discordId}/${avatarHash}.png?size=32`;
        } else {
          userAvatar.src = `https://cdn.discordapp.com/embed/avatars/${parseInt(discordId || '0') % 5}.png`;
        }
      }
    } else {
      if (loginBtn) loginBtn.classList.remove('hidden');
      if (userBadge) userBadge.classList.add('hidden');
    }
  },

  showMenu() {
    const tier = this.tier();
    const tierLabel = { free: 'Free', starter: 'Starter', pro: 'Pro', elite: 'Elite' }[tier] || tier;
    const status = this.user?.status || 'inactive';
    if (confirm(`${this.user?.username}\nPlan: ${tierLabel} (${status})\n\nLog out?`)) {
      this.logout();
    }
  },
};


// ================================================================
// PRICING MODULE
// ================================================================

const pricing = {
  interval: 'monthly',

  setInterval(interval) {
    this.interval = interval;
    // Update toggle buttons
    document.querySelectorAll('.billing-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.interval === interval);
    });
    // Update displayed prices
    document.querySelectorAll('.pricing-amount').forEach(el => {
      el.textContent = el.dataset[interval] || el.textContent;
    });
    document.querySelectorAll('.pricing-period').forEach(el => {
      el.textContent = el.dataset[interval] || el.textContent;
    });
  },

  async subscribe(tier) {
    console.log('Subscribe called:', tier, this.interval, 'loggedIn:', auth.isLoggedIn(), 'token:', !!auth.token);
    // Must be logged in with Discord first
    if (!auth.isLoggedIn()) {
      // If we have a token but user hasn't loaded yet, try init first
      if (auth.token && !auth.user) {
        await auth.init();
      }
      if (!auth.isLoggedIn()) {
        // Store intent, redirect to Discord login
        localStorage.setItem('spickbot_subscribe_intent', JSON.stringify({ tier, interval: this.interval }));
        auth.login();
        return;
      }
    }
    try {
      console.log('Creating checkout session...');
      const resp = await fetch('/api/checkout/create-session', {
        method: 'POST',
        headers: auth.getHeaders(),
        body: JSON.stringify({ tier: tier, interval: this.interval }),
      });
      console.log('Checkout response:', resp.status);
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        alert(err.detail || 'Failed to create checkout session. Please try again.');
        return;
      }
      const data = await resp.json();
      if (data.url) {
        window.location.href = data.url;
      }
    } catch (e) {
      console.error('Checkout error:', e);
      alert('Something went wrong. Please try again.');
    }
  },
};


// ================================================================
// APP STATE
// ================================================================

const state = {
  activeDate: todayStr(),
  activeLeague: 'all',
  activeTab: 'fixtures',
  activeMarketTab: 'all',
  slipLegs: JSON.parse(localStorage.getItem('spickbot_slip') || '[]'),
  slipStake: parseFloat(localStorage.getItem('spickbot_stake') || '0') || 0,
  fixtureCache: {},
  bestBetsCache: {},
  currentMatch: null,
  currentMatchData: null,
  initialized: false,
};


// ================================================================
// APP CONTROLLER
// ================================================================

const app = {
  init() {
    if (state.initialized) {
      this.refreshCurrentView();
      return;
    }
    state.initialized = true;
    this.buildDateStrip('dateStrip');
    this.buildDateStrip('bestBetsDateStrip');
    this.buildLeagueFilters();
    this.loadFixtures();
    slip.init();
  },

  refreshCurrentView() {
    if (state.activeTab === 'fixtures') this.loadFixtures();
    else if (state.activeTab === 'best-bets') this.loadBestBets();
  },

  // -- Date strip --
  buildDateStrip(containerId) {
    const el = $(containerId);
    if (!el) return;
    el.innerHTML = '';
    const today = todayStr();
    for (let i = 0; i < 7; i++) {
      const d = dateAdd(today, i);
      const btn = document.createElement('button');
      btn.className = 'date-chip' + (d === state.activeDate ? ' active' : '');
      btn.textContent = formatDateShort(d);
      btn.setAttribute('role', 'tab');
      btn.setAttribute('aria-selected', d === state.activeDate ? 'true' : 'false');
      btn.onclick = () => {
        state.activeDate = d;
        this.updateDateStrips();
        if (state.activeTab === 'fixtures') this.loadFixtures();
        else this.loadBestBets();
      };
      el.appendChild(btn);
    }
  },

  updateDateStrips() {
    ['dateStrip', 'bestBetsDateStrip'].forEach(id => {
      const el = $(id);
      if (!el) return;
      const today = todayStr();
      Array.from(el.children).forEach((btn, i) => {
        const d = dateAdd(today, i);
        btn.className = 'date-chip' + (d === state.activeDate ? ' active' : '');
        btn.setAttribute('aria-selected', d === state.activeDate ? 'true' : 'false');
      });
    });
  },

  // -- League filters --
  buildLeagueFilters() {
    const el = $('leagueFilters');
    if (!el) return;
    el.innerHTML = '';

    // "All" chip
    const allBtn = document.createElement('button');
    allBtn.className = 'league-chip' + (state.activeLeague === 'all' ? ' active' : '');
    allBtn.textContent = 'All';
    allBtn.onclick = () => { state.activeLeague = 'all'; this.updateLeagueFilters(); this.renderFixtures(); };
    el.appendChild(allBtn);

    ALL_LEAGUES.forEach(code => {
      const btn = document.createElement('button');
      btn.className = 'league-chip' + (state.activeLeague === code ? ' active' : '');
      btn.innerHTML = `<span class="league-dot" style="background:${LEAGUE_COLORS[code]}"></span>${LEAGUE_NAMES[code]}`;
      btn.onclick = () => { state.activeLeague = code; this.updateLeagueFilters(); this.renderFixtures(); };
      el.appendChild(btn);
    });
  },

  updateLeagueFilters() {
    const el = $('leagueFilters');
    if (!el) return;
    Array.from(el.children).forEach((btn, i) => {
      const code = i === 0 ? 'all' : ALL_LEAGUES[i - 1];
      btn.className = 'league-chip' + (state.activeLeague === code ? ' active' : '');
    });
  },

  // -- Tab switching --
  switchTab(tab) {
    state.activeTab = tab;

    $('viewFixtures').classList.toggle('hidden', tab !== 'fixtures');
    $('viewBestBets').classList.toggle('hidden', tab !== 'best-bets');
    $('viewMatch').classList.toggle('hidden', tab !== 'match');

    qsa('.app-nav[data-view]').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.view === tab);
    });

    if (tab === 'fixtures') this.loadFixtures();
    else if (tab === 'best-bets') this.loadBestBets();
  },

  // -- Load fixtures --
  async loadFixtures() {
    const container = $('fixturesContainer');
    if (!container) return;

    const cacheKey = state.activeDate;
    if (state.fixtureCache[cacheKey]) {
      this.renderFixtures();
      return;
    }

    container.innerHTML = this.skeletonHTML(5);

    const allData = {};
    const fetches = ALL_LEAGUES.map(async (league) => {
      try {
        const resp = await fetch(`${API}/fixtures/${league}/${state.activeDate}`);
        if (resp.ok) {
          const data = await resp.json();
          allData[league] = data.fixtures || [];
        }
      } catch (e) {
        console.warn(`Failed to fetch ${league}:`, e);
      }
    });

    await Promise.all(fetches);
    state.fixtureCache[cacheKey] = allData;
    this.renderFixtures();
  },

  renderFixtures() {
    const container = $('fixturesContainer');
    if (!container) return;

    const data = state.fixtureCache[state.activeDate];
    if (!data) {
      container.innerHTML = this.emptyHTML('No data loaded.');
      return;
    }

    const leagues = state.activeLeague === 'all' ? ALL_LEAGUES : [state.activeLeague];
    let totalFixtures = 0;
    let html = '';

    leagues.forEach(league => {
      const fixtures = data[league];
      if (!fixtures || fixtures.length === 0) return;
      totalFixtures += fixtures.length;

      html += `<div class="league-group">`;
      html += `<div class="league-header">
        <div class="league-header-left">
          <div class="league-stripe" style="background:${LEAGUE_COLORS[league]}"></div>
          <span class="league-name">${LEAGUE_NAMES[league]}</span>
        </div>
        <span class="league-count">${fixtures.length} match${fixtures.length !== 1 ? 'es' : ''}</span>
      </div>`;

      fixtures.forEach((fx, i) => {
        const bb = fx.best_bet;
        const conf = bb ? confClass(bb.confidence) : 'low';
        const odds = bb && bb.odds ? oddsFmt(bb.odds) : '--';
        const edge = bb && bb.edge != null ? edgeFmt(bb.edge) : '--';
        const pick = bb ? escHtml(bb.pick) : '--';
        const isInSlip = slip.has(fx.event_id, bb ? bb.market : '', bb ? bb.pick : '');

        html += `<div class="fixture-row" onclick="app.openMatch('${league}', '${fx.event_id}')">
          <span class="fixture-num">${i + 1}</span>
          <div class="fixture-match">
            <span class="fixture-teams">${escHtml(fx.home_team)}<span class="fixture-vs">vs</span>${escHtml(fx.away_team)}</span>
          </div>
          <span class="fixture-pick-text">${pick}</span>
          <span class="conf-badge ${conf}">${(bb && bb.confidence) ? bb.confidence.toUpperCase() : '--'}</span>
          <span class="fixture-odds-val">${odds}</span>
          <span class="fixture-edge ${edgeClass(bb ? bb.edge : null)}">${edge}</span>
          <button class="fixture-add-btn ${isInSlip ? 'added' : ''}" onclick="event.stopPropagation(); app.addBestBet('${league}', '${fx.event_id}')" aria-label="Add to slip">
            ${isInSlip ? '&#10003;' : '+'}
          </button>
        </div>`;
      });

      html += `</div>`;
    });

    if (totalFixtures === 0) {
      container.innerHTML = this.emptyHTML('No fixtures found for this date and league.');
      return;
    }

    container.innerHTML = html;
  },

  addBestBet(league, eventId) {
    const data = state.fixtureCache[state.activeDate];
    if (!data) return;
    const fixtures = data[league];
    if (!fixtures) return;
    const fx = fixtures.find(f => f.event_id === eventId);
    if (!fx || !fx.best_bet) return;

    const bb = fx.best_bet;
    const legId = `${eventId}_${bb.market}_${bb.pick}`;

    if (slip.has(eventId, bb.market, bb.pick)) {
      slip.remove(legId);
    } else {
      slip.add({
        id: legId,
        event_id: eventId,
        fixture: `${fx.home_team} vs ${fx.away_team}`,
        market: bb.market,
        pick: bb.pick,
        side: bb.pick,
        line: null,
        odds: bb.odds || 1.01,
        league,
      });
    }
    this.renderFixtures();
  },

  // -- Match detail --
  async openMatch(league, eventId) {
    const data = state.fixtureCache[state.activeDate];
    if (!data || !data[league]) return;
    const fx = data[league].find(f => f.event_id === eventId);
    if (!fx) return;

    state.currentMatch = { league, eventId, fx };
    state.currentMatchData = fx;
    state.activeMarketTab = 'all';
    this.switchTab('match');
    this.renderMatchDetail();
  },

  renderMatchDetail() {
    const { league, fx } = state.currentMatch;
    const header = $('matchHeader');
    const tabs = $('marketTabs');
    const markets = $('matchMarkets');

    // Header
    header.innerHTML = `<div class="match-hero">
      <div class="match-league-badge">${LEAGUE_NAMES[league]}</div>
      <div class="match-teams-title">${escHtml(fx.home_team)}<span class="match-teams-vs">vs</span>${escHtml(fx.away_team)}</div>
      <div class="match-meta">${formatTime(fx.kickoff)}</div>
    </div>`;

    // Market tabs
    const tabList = [{ key: 'all', label: 'All Markets' }, ...MARKET_ORDER.map(k => ({ key: k, label: MARKET_LABELS[k] }))];
    tabs.innerHTML = tabList.map(t =>
      `<button class="market-tab ${state.activeMarketTab === t.key ? 'active' : ''}" onclick="state.activeMarketTab='${t.key}'; app.renderMatchDetail()">${t.label}</button>`
    ).join('');

    // Markets
    const showMarkets = state.activeMarketTab === 'all' ? MARKET_ORDER : [state.activeMarketTab];
    markets.innerHTML = showMarkets.map(m => this.renderMarketCard(fx, m, league)).join('');
  },

  renderMarketCard(fx, market, league) {
    const data = fx[market];
    if (!data) return `<div class="market-card">
      <div class="market-card-header"><span class="market-card-title">${MARKET_LABELS[market]}</span></div>
      <div class="empty-state"><p>No data available</p></div>
    </div>`;

    if (market === 'moneyline') return this.renderMoneylineCard(fx, data, league);
    if (market === 'btts') return this.renderBttsCard(fx, data, league);
    if (market === 'spreads') return this.renderSpreadsCard(fx, data, league);
    return this.renderTotalsCard(fx, data, market, league);
  },

  renderTotalsCard(fx, data, market, league) {
    if (!data.projected_total) return '';
    const side = data.recommended_side || 'Over';
    const line = data.recommended_line != null ? data.recommended_line : '?';
    const proj = data.projected_total;
    const conf = confClass(data.confidence);
    const odds = oddsFmt(data.odds);
    const mp = pct(data.model_prob);
    const edge = data.value_edge;
    const pick = `${side} ${line}`;
    const isRec = true;
    const legId = `${fx.event_id}_${market}_${pick}`;
    const inSlip = state.slipLegs.some(l => l.id === legId);

    return `<div class="market-card">
      <div class="market-card-header">
        <span class="market-card-title">${MARKET_LABELS[market]}</span>
        <span class="conf-badge ${conf}">${(data.confidence || 'low').toUpperCase()}</span>
      </div>
      <div class="model-proj-row">
        <span class="model-proj-label">Model Projection</span>
        <span class="model-proj-value">${proj.toFixed(2)}</span>
      </div>
      <div class="market-line ${isRec ? 'recommended' : ''}">
        <div class="market-line-left">
          <span class="market-line-pick">${escHtml(pick)}</span>
          <span class="market-line-proj">P: ${mp}</span>
          ${edge != null ? `<span class="market-line-proj ${edgeClass(edge)}">Edge: ${edgeFmt(edge)}</span>` : ''}
          ${valueBadge(edge)}
        </div>
        <div class="market-line-right">
          <span class="market-line-odds">${odds}</span>
          <button class="market-add-btn ${inSlip ? 'added' : ''}" onclick="app.addMarketLeg('${market}', '${escHtml(pick)}', ${data.odds || 0}, '${league}')">
            ${inSlip ? 'Added' : '+ Add'}
          </button>
        </div>
      </div>
    </div>`;
  },

  renderMoneylineCard(fx, data, league) {
    if (data.home_prob == null) return '';
    const hp = data.home_prob * 100;
    const dp = (data.draw_prob || 0) * 100;
    const ap = (data.away_prob || 0) * 100;
    const conf = confClass(data.confidence);
    const rec = data.recommended || fx.home_team;
    const odds = oddsFmt(data.odds);
    const edge = data.value_edge;
    const pick = `${rec} Win`;
    const legId = `${fx.event_id}_moneyline_${pick}`;
    const inSlip = state.slipLegs.some(l => l.id === legId);

    return `<div class="market-card">
      <div class="market-card-header">
        <span class="market-card-title">Moneyline</span>
        <span class="conf-badge ${conf}">${(data.confidence || 'low').toUpperCase()}</span>
      </div>
      <div class="prob-bar">
        <div class="prob-seg home" style="width:${hp}%">${fx.home_team.split(' ').pop()} ${hp.toFixed(0)}%</div>
        <div class="prob-seg draw" style="width:${dp}%">${dp.toFixed(0)}%</div>
        <div class="prob-seg away" style="width:${ap}%">${fx.away_team.split(' ').pop()} ${ap.toFixed(0)}%</div>
      </div>
      <div class="market-line recommended">
        <div class="market-line-left">
          <span class="market-line-pick">${escHtml(pick)}</span>
          ${edge != null ? `<span class="market-line-proj ${edgeClass(edge)}">Edge: ${edgeFmt(edge)}</span>` : ''}
          ${valueBadge(edge)}
        </div>
        <div class="market-line-right">
          <span class="market-line-odds">${odds}</span>
          <button class="market-add-btn ${inSlip ? 'added' : ''}" onclick="app.addMarketLeg('moneyline', '${escHtml(pick)}', ${data.odds || 0}, '${league}')">
            ${inSlip ? 'Added' : '+ Add'}
          </button>
        </div>
      </div>
    </div>`;
  },

  renderBttsCard(fx, data, league) {
    if (data.probability == null) return '';
    const prob = (data.probability * 100).toFixed(0);
    const side = data.recommended_side || 'Yes';
    const conf = confClass(data.confidence);
    const odds = oddsFmt(data.odds);
    const edge = data.value_edge;
    const pick = `BTTS ${side}`;
    const legId = `${fx.event_id}_btts_${pick}`;
    const inSlip = state.slipLegs.some(l => l.id === legId);

    return `<div class="market-card">
      <div class="market-card-header">
        <span class="market-card-title">Both Teams To Score</span>
        <span class="conf-badge ${conf}">${(data.confidence || 'low').toUpperCase()}</span>
      </div>
      <div class="model-proj-row">
        <span class="model-proj-label">BTTS Probability</span>
        <span class="model-proj-value">${prob}%</span>
      </div>
      <div class="market-line recommended">
        <div class="market-line-left">
          <span class="market-line-pick">${escHtml(pick)}</span>
          ${edge != null ? `<span class="market-line-proj ${edgeClass(edge)}">Edge: ${edgeFmt(edge)}</span>` : ''}
          ${valueBadge(edge)}
        </div>
        <div class="market-line-right">
          <span class="market-line-odds">${odds}</span>
          <button class="market-add-btn ${inSlip ? 'added' : ''}" onclick="app.addMarketLeg('btts', '${escHtml(pick)}', ${data.odds || 0}, '${league}')">
            ${inSlip ? 'Added' : '+ Add'}
          </button>
        </div>
      </div>
    </div>`;
  },

  renderSpreadsCard(fx, data, league) {
    if (data.projected_diff == null) return '';
    const diff = data.projected_diff;
    const team = data.recommended_team || fx.home_team;
    const line = data.recommended_line != null ? data.recommended_line : '?';
    const conf = confClass(data.confidence);
    const odds = oddsFmt(data.odds);
    const mp = pct(data.model_prob);
    const edge = data.value_edge;
    const lineStr = line >= 0 ? `+${line}` : `${line}`;
    const pick = `${team} ${lineStr}`;
    const legId = `${fx.event_id}_spreads_${pick}`;
    const inSlip = state.slipLegs.some(l => l.id === legId);

    return `<div class="market-card">
      <div class="market-card-header">
        <span class="market-card-title">Handicap / Spread</span>
        <span class="conf-badge ${conf}">${(data.confidence || 'low').toUpperCase()}</span>
      </div>
      <div class="model-proj-row">
        <span class="model-proj-label">Projected Goal Diff</span>
        <span class="model-proj-value">${diff >= 0 ? '+' : ''}${diff.toFixed(2)}</span>
      </div>
      <div class="market-line recommended">
        <div class="market-line-left">
          <span class="market-line-pick">${escHtml(pick)}</span>
          <span class="market-line-proj">Cover P: ${mp}</span>
          ${edge != null ? `<span class="market-line-proj ${edgeClass(edge)}">Edge: ${edgeFmt(edge)}</span>` : ''}
          ${valueBadge(edge)}
        </div>
        <div class="market-line-right">
          <span class="market-line-odds">${odds}</span>
          <button class="market-add-btn ${inSlip ? 'added' : ''}" onclick="app.addMarketLeg('spreads', '${escHtml(pick)}', ${data.odds || 0}, '${league}')">
            ${inSlip ? 'Added' : '+ Add'}
          </button>
        </div>
      </div>
    </div>`;
  },

  addMarketLeg(market, pick, odds, league) {
    const fx = state.currentMatch.fx;
    const legId = `${fx.event_id}_${market}_${pick}`;

    if (state.slipLegs.some(l => l.id === legId)) {
      slip.remove(legId);
    } else {
      slip.add({
        id: legId,
        event_id: fx.event_id,
        fixture: `${fx.home_team} vs ${fx.away_team}`,
        market,
        pick,
        side: pick,
        line: null,
        odds: odds || 1.01,
        league,
      });
    }
    this.renderMatchDetail();
  },

  // -- Best Bets --
  async loadBestBets() {
    const container = $('bestBetsContainer');
    if (!container) return;

    if (state.bestBetsCache[state.activeDate]) {
      this.renderBestBets(state.bestBetsCache[state.activeDate]);
      return;
    }

    container.innerHTML = this.skeletonHTML(6);

    try {
      const resp = await fetch(`${API}/best-bets/${state.activeDate}?limit=15`);
      if (!resp.ok) throw new Error('Failed to fetch');
      const data = await resp.json();
      state.bestBetsCache[state.activeDate] = data.picks || [];
      this.renderBestBets(data.picks || []);
    } catch (e) {
      container.innerHTML = `<div class="error-state"><p>Failed to load best bets.</p><button class="retry-btn" onclick="app.loadBestBets()">Retry</button></div>`;
    }
  },

  renderBestBets(picks) {
    const container = $('bestBetsContainer');
    if (!container) return;

    if (picks.length === 0) {
      container.innerHTML = this.emptyHTML('No best bets found for this date.');
      return;
    }

    let html = '';
    picks.forEach((p, i) => {
      const conf = confClass(p.confidence);
      const edge = p.value_edge;
      const legId = `${p.event_id}_${p.market}_${p.pick}`;
      const inSlip = state.slipLegs.some(l => l.id === legId);

      html += `<div class="market-card" style="margin-bottom:12px">
        <div style="padding:16px 18px">
          <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:10px; flex-wrap:wrap; gap:8px">
            <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap">
              <span class="league-dot" style="background:${LEAGUE_COLORS[p.league] || '#333'}"></span>
              <span style="font-size:14px; font-weight:700">${escHtml(p.home_team)} vs ${escHtml(p.away_team)}</span>
              <span style="font-family:var(--mono); font-size:11px; color:var(--text3)">${formatTime(p.kickoff)}</span>
            </div>
            <span class="conf-badge ${conf}">${p.confidence.toUpperCase()}</span>
          </div>
          <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px; flex-wrap:wrap">
            <span style="font-size:15px; font-weight:700">${escHtml(p.pick)}</span>
            ${valueBadge(edge)}
          </div>
          <div style="display:flex; gap:24px; margin-bottom:12px; flex-wrap:wrap">
            <div>
              <div style="font-size:11px; color:var(--text3); text-transform:uppercase; letter-spacing:0.3px">Odds</div>
              <div class="mono" style="font-size:14px; font-weight:600">${oddsFmt(p.odds)}</div>
            </div>
            <div>
              <div style="font-size:11px; color:var(--text3); text-transform:uppercase; letter-spacing:0.3px">Model Prob</div>
              <div class="mono" style="font-size:14px; font-weight:600">${pct(p.model_prob)}</div>
            </div>
            <div>
              <div style="font-size:11px; color:var(--text3); text-transform:uppercase; letter-spacing:0.3px">Edge</div>
              <div class="mono ${edgeClass(edge)}" style="font-size:14px; font-weight:600">${edgeFmt(edge)}</div>
            </div>
          </div>
          <div style="font-size:13px; color:var(--text2); margin-bottom:12px">${escHtml(p.reason)}</div>
          <div style="text-align:right">
            <button class="market-add-btn ${inSlip ? 'added' : ''}" onclick="app.addBestBetPick(${JSON.stringify(p).replace(/"/g, '&quot;')})">
              ${inSlip ? 'Added' : '+ Add to Slip'}
            </button>
          </div>
        </div>
      </div>`;
    });

    container.innerHTML = html;
  },

  addBestBetPick(p) {
    const legId = `${p.event_id}_${p.market}_${p.pick}`;
    if (state.slipLegs.some(l => l.id === legId)) {
      slip.remove(legId);
    } else {
      slip.add({
        id: legId,
        event_id: p.event_id,
        fixture: `${p.home_team} vs ${p.away_team}`,
        market: p.market,
        pick: p.pick,
        side: p.pick,
        line: null,
        odds: p.odds || 1.01,
        league: p.league,
      });
    }
    this.renderBestBets(state.bestBetsCache[state.activeDate] || []);
  },

  // -- Helpers --
  skeletonHTML(n) {
    let h = '';
    for (let i = 0; i < n; i++) {
      h += `<div class="skeleton-row">
        <div class="skeleton-line w80 h20"></div>
        <div class="skeleton-line w60"></div>
        <div class="skeleton-line w40"></div>
      </div>`;
    }
    return h;
  },

  emptyHTML(msg) {
    return `<div class="empty-state">
      <div class="empty-state-icon">
        <svg width="48" height="48" viewBox="0 0 48 48" fill="none"><circle cx="24" cy="24" r="20" stroke="currentColor" stroke-width="1.5"/><path d="M16 20h16M20 28h8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
      </div>
      <p>${msg}</p>
      <p class="empty-state-hint">Try selecting a different date or league.</p>
    </div>`;
  },
};


// ================================================================
// BET SLIP
// ================================================================

const slip = {
  init() {
    $('slipStake').addEventListener('input', () => {
      state.slipStake = parseFloat($('slipStake').value) || 0;
      localStorage.setItem('spickbot_stake', String(state.slipStake));
      this.updateSummary();
    });
    $('slipStake').value = state.slipStake || '';
    this.render();
  },

  persist() {
    localStorage.setItem('spickbot_slip', JSON.stringify(state.slipLegs));
  },

  has(eventId, market, pick) {
    const id = `${eventId}_${market}_${pick}`;
    return state.slipLegs.some(l => l.id === id);
  },

  add(leg) {
    if (state.slipLegs.some(l => l.id === leg.id)) return;
    state.slipLegs.push(leg);
    this.persist();
    this.render();
    showToast('Added to slip', 'success');
    if (state.slipLegs.length === 1) this.open();
  },

  remove(id) {
    state.slipLegs = state.slipLegs.filter(l => l.id !== id);
    this.persist();
    this.render();
    // Re-render active view
    if (state.activeTab === 'fixtures') app.renderFixtures();
    else if (state.activeTab === 'best-bets') app.renderBestBets(state.bestBetsCache[state.activeDate] || []);
    else if (state.activeTab === 'match') app.renderMatchDetail();
  },

  clear() {
    state.slipLegs = [];
    this.persist();
    this.render();
    $('slipQuality').classList.add('hidden');
    // Re-render
    if (state.activeTab === 'fixtures') app.renderFixtures();
    else if (state.activeTab === 'best-bets') app.renderBestBets(state.bestBetsCache[state.activeDate] || []);
    else if (state.activeTab === 'match') app.renderMatchDetail();
  },

  setStake(amount) {
    state.slipStake = amount;
    $('slipStake').value = amount;
    localStorage.setItem('spickbot_stake', String(amount));
    this.updateSummary();
  },

  toggle() {
    const el = $('betSlip');
    const layout = $('appLayout');
    const backdrop = $('slipBackdrop');
    const isOpen = el.classList.contains('open');

    if (isOpen) {
      el.classList.remove('open');
      layout.classList.remove('slip-open');
      backdrop.classList.add('hidden');
    } else {
      el.classList.add('open');
      layout.classList.add('slip-open');
      // Show backdrop on mobile
      if (window.innerWidth <= 1100) backdrop.classList.remove('hidden');
    }
  },

  open() {
    const el = $('betSlip');
    if (!el.classList.contains('open')) {
      el.classList.add('open');
      $('appLayout').classList.add('slip-open');
      if (window.innerWidth <= 1100) $('slipBackdrop').classList.remove('hidden');
    }
  },

  render() {
    const legs = state.slipLegs;
    const legsEl = $('slipLegs');
    const count = legs.length;

    // Counts
    $('slipToggleCount').textContent = count;
    $('slipCount').textContent = count;
    $('slipMobileCount').textContent = count;

    // Toggle visibility
    const slipToggle = $('slipToggle');
    if (slipToggle) slipToggle.style.display = count > 0 ? 'flex' : 'none';

    const mobileBar = $('slipMobileBar');
    if (count > 0) mobileBar.classList.remove('hidden');
    else mobileBar.classList.add('hidden');

    // Legs
    if (count === 0) {
      legsEl.innerHTML = '<div class="slip-empty">No selections yet. Click + on any fixture to add a bet.</div>';
    } else {
      legsEl.innerHTML = legs.map(leg => `<div class="slip-leg">
        <div class="slip-leg-fixture">${escHtml(leg.fixture)}</div>
        <div class="slip-leg-pick">${escHtml(leg.pick)}</div>
        <div class="slip-leg-odds">${oddsFmt(leg.odds)}</div>
        <button class="slip-leg-remove" onclick="slip.remove('${leg.id}')" aria-label="Remove">&times;</button>
      </div>`).join('');
    }

    this.updateSummary();
  },

  updateSummary() {
    const legs = state.slipLegs;
    let combined = 1;
    legs.forEach(l => combined *= l.odds);

    const oddsStr = legs.length > 0 ? combined.toFixed(2) : '--';
    $('slipOdds').textContent = oddsStr;
    $('slipMobileOdds').textContent = oddsStr;

    const stake = state.slipStake;
    const payout = legs.length > 0 && stake > 0 ? (stake * combined).toFixed(2) : '0.00';
    $('slipPayout').textContent = '$' + payout;

    $('slipScoreBtn').disabled = legs.length === 0;
  },

  async score() {
    const legs = state.slipLegs;
    if (legs.length === 0) return;

    const btn = $('slipScoreBtn');
    btn.disabled = true;
    btn.textContent = 'Scoring...';

    try {
      const resp = await fetch(`${API}/parlay/score`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          legs: legs.map(l => ({
            event_id: l.event_id,
            market: l.market,
            side: l.side,
            line: l.line,
            odds: l.odds,
          })),
        }),
      });

      if (!resp.ok) throw new Error('Scoring failed');
      const data = await resp.json();

      const qualEl = $('slipQuality');
      const fill = $('qualityFill');
      const label = $('qualityLabel');
      const warns = $('qualityWarnings');

      qualEl.classList.remove('hidden');
      const score = data.quality_score || 0;
      const pct = (score / 10) * 100;
      fill.style.width = pct + '%';

      let color = 'var(--loss)';
      if (score >= 7.5) color = 'var(--green)';
      else if (score >= 6) color = 'var(--accent)';
      else if (score >= 4.5) color = 'var(--amber)';
      fill.style.background = color;

      label.textContent = `${data.quality_label} (${score}/10)`;
      label.style.color = color;

      warns.innerHTML = (data.warnings || []).map(w => `<div>${escHtml(w)}</div>`).join('');

      showToast(`Parlay scored: ${data.quality_label}`, 'success');
    } catch (e) {
      showToast('Failed to score parlay', 'error');
    } finally {
      btn.disabled = false;
      btn.textContent = 'Score My Parlay';
    }
  },
};


// ================================================================
// REVIEWS ANIMATION (duplicate cards for infinite scroll)
// ================================================================

function initReviewsScroll() {
  const slide = $('reviewsSlide');
  if (!slide) return;
  // Clone all cards for seamless loop
  const cards = slide.innerHTML;
  slide.innerHTML = cards + cards;
}


// ================================================================
// INIT
// ================================================================

document.addEventListener('DOMContentLoaded', async () => {
  await auth.init();
  // Check for pending subscribe intent (user was redirected to login first)
  const intent = localStorage.getItem('spickbot_subscribe_intent');
  if (intent && auth.isLoggedIn()) {
    localStorage.removeItem('spickbot_subscribe_intent');
    try {
      const { tier, interval } = JSON.parse(intent);
      pricing.interval = interval || 'monthly';
      pricing.subscribe(tier);
      return; // Don't load the page, we're redirecting to Stripe
    } catch (e) { /* ignore bad intent */ }
  }
  router.init();
  initReviewsScroll();
});
