
// ============================================================
// SpixBot — stub JS for preview (app.js not included)
// ============================================================

// ----- Active users counter -----
(function() {
  const el = document.getElementById("activeUsers");
  if (!el) return;
  let base = 34 + Math.floor(Math.random() * 8);
  el.textContent = base;
  setInterval(() => {
    base += Math.random() > 0.5 ? 1 : -1;
    base = Math.max(28, Math.min(52, base));
    el.textContent = base;
  }, 4000);
})();

// ----- Seamless review carousel -----
(function() {
  const slide = document.getElementById("reviewsSlide");
  if (!slide) return;
  // Clone all cards and append so translateX(-50%) loops perfectly
  const cards = Array.from(slide.children);
  cards.forEach(card => slide.appendChild(card.cloneNode(true)));
})();

// ----- Utilities -----
function esc(str) {
  const d = document.createElement('div');
  d.textContent = str;
  return d.innerHTML;
}

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

// ----- Router -----
const router = {
  go(view) {
    const landing = document.getElementById('landingPage');
    const app = document.getElementById('appPage');
    const landingNavEls = document.querySelectorAll('.landing-nav');
    const appNavEls = document.querySelectorAll('.app-nav');
    if (view === 'app') {
      // Require login to access the app
      if (!auth.isLoggedIn()) {
        auth.showLoginModal();
        return;
      }
      landing.classList.add('hidden');
      app.classList.remove('hidden');
      landingNavEls.forEach(el => el.classList.add('hidden'));
      appNavEls.forEach(el => el.classList.remove('hidden'));
      this._appInit();
    } else {
      app.classList.add('hidden');
      landing.classList.remove('hidden');
      landingNavEls.forEach(el => el.classList.remove('hidden'));
      appNavEls.forEach(el => el.classList.add('hidden'));
    }
  },
  _appInit() {
    appModule.init();
  }
};

// ----- Pricing -----
const pricing = {
  setInterval(interval) {
    document.querySelectorAll('.billing-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.interval === interval);
    });
    document.querySelectorAll('.pricing-amount').forEach(el => {
      el.textContent = el.dataset[interval];
    });
    document.querySelectorAll('.pricing-period').forEach(el => {
      el.textContent = el.dataset[interval];
    });
  },
  async subscribe(plan) {
    // Require Discord login before checkout
    if (!auth.isLoggedIn()) {
      showToast('Please log in with Discord first', 'info');
      window.location.href = '/auth/discord/login';
      return;
    }

    const interval = document.querySelector('.billing-btn.active')?.dataset.interval || 'monthly';
    showToast('Redirecting to checkout...', 'success');

    try {
      const resp = await fetch('/api/checkout/create-session', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer ' + auth.getToken(),
        },
        body: JSON.stringify({ tier: plan, interval }),
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        showToast(err.detail || 'Checkout failed', 'error');
        return;
      }

      const data = await resp.json();
      if (data.url) {
        window.location.href = data.url;
      }
    } catch (e) {
      showToast('Checkout failed — please try again', 'error');
    }
  }
};

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
    if (!this._user) return;
    const loginBtn = document.getElementById('loginBtn');
    const badge = document.getElementById('userBadge');
    const avatar = document.getElementById('userAvatar');
    const name = document.getElementById('userName');
    if (loginBtn) loginBtn.classList.add('hidden');
    if (badge) badge.classList.remove('hidden');
    if (avatar) {
      if (this._user.avatar_url) {
        avatar.src = this._user.avatar_url;
        avatar.style.display = '';
      } else {
        // Email users — hide avatar img, the name is enough
        avatar.style.display = 'none';
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
    if (confirm('Log out of SpixBot?')) this.logout();
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
    document.getElementById('authModal').classList.remove('hidden');
    document.getElementById('authBackdrop').classList.remove('hidden');
    document.getElementById('authEmail').focus();
  },

  hideLoginModal() {
    document.getElementById('authModal').classList.add('hidden');
    document.getElementById('authBackdrop').classList.add('hidden');
    document.getElementById('authError').classList.add('hidden');
    document.getElementById('authForm').reset();
  },

  toggleMode() {
    this._isSignup = !this._isSignup;
    this._renderModalState();
  },

  _renderModalState() {
    const title = document.getElementById('authModalTitle');
    const btn = document.getElementById('authSubmitBtn');
    const toggle = document.getElementById('authToggle');
    const pw = document.getElementById('authPassword');
    document.getElementById('authError').classList.add('hidden');
    if (this._isSignup) {
      title.textContent = 'Create your account';
      btn.textContent = 'Sign Up';
      toggle.innerHTML = 'Already have an account? <a href="#" onclick="event.preventDefault(); auth.toggleMode()">Log in</a>';
      pw.setAttribute('autocomplete', 'new-password');
      pw.setAttribute('minlength', '8');
    } else {
      title.textContent = 'Log in to SpixBot';
      btn.textContent = 'Log In';
      toggle.innerHTML = 'Don\'t have an account? <a href="#" onclick="event.preventDefault(); auth.toggleMode()">Sign up</a>';
      pw.setAttribute('autocomplete', 'current-password');
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

  toggle() {
    this.open = !this.open;
    const el = document.getElementById('betSlip');
    el.classList.toggle('open', this.open);
    document.getElementById('slipBackdrop').classList.toggle('hidden', !this.open);
    document.getElementById('appLayout').classList.toggle('slip-open', this.open);
  },

  add(fixture, pick, odds, market) {
    const key = fixture + '|' + pick;
    if (this.legs.find(l => l.key === key)) { this.remove(key); return false; }
    this.legs.push({ key, fixture, pick, odds, market });
    this._render();
    this._update();
    if (!this.open) this.toggle();
    return true;
  },

  remove(key) {
    this.legs = this.legs.filter(l => l.key !== key);
    this._render();
    this._update();
    // sync add buttons
    document.querySelectorAll('[data-slip-key]').forEach(btn => {
      if (btn.dataset.slipKey === key) btn.classList.remove('added');
    });
  },

  clear() {
    this.legs = [];
    this._render();
    this._update();
    document.querySelectorAll('.fixture-add-btn.added, .market-add-btn.added').forEach(b => b.classList.remove('added'));
  },

  setStake(v) {
    document.getElementById('slipStake').value = v;
    this._calcPayout();
  },

  score() {
    // TODO: BACKEND — score parlay server-side
    // Suggested: POST /api/slip/score { legs: this.legs }
    // Response: { score: 0-100, label: string, warnings: string[] }
    if (!this.legs.length) { showToast('Add picks first', 'error'); return; }
    const q = document.getElementById('slipQuality');
    q.classList.remove('hidden');
    const fill = document.getElementById('qualityFill');
    const pct = Math.min(95, 45 + this.legs.length * 8 + Math.random() * 15);
    fill.style.width = pct + '%';
    fill.style.background = pct > 70 ? 'var(--green)' : pct > 45 ? 'var(--amber)' : 'var(--loss)';
    document.getElementById('qualityLabel').textContent = pct > 70 ? '🔥 Strong Parlay' : pct > 45 ? '⚠️ Average Value' : '❌ Weak Parlay';
    document.getElementById('qualityLabel').style.color = pct > 70 ? 'var(--green)' : pct > 45 ? 'var(--amber)' : 'var(--loss)';
    document.getElementById('qualityWarnings').textContent = this.legs.length > 4 ? 'High leg count reduces probability.' : '';
    showToast('Parlay scored!', 'success');
  },

  _render() {
    const legsEl = document.getElementById('slipLegs');
    if (!this.legs.length) {
      legsEl.innerHTML = '<div class="slip-empty">No picks yet. Browse fixtures to add bets.</div>';
      document.getElementById('slipQuality').classList.add('hidden');
      return;
    }
    legsEl.innerHTML = this.legs.map(l => `
      <div class="slip-leg">
        <div class="slip-leg-fixture">${esc(l.fixture)}</div>
        <div class="slip-leg-pick">${esc(l.pick)}</div>
        <div class="slip-leg-odds mono">${esc(String(l.odds))}</div>
        <button class="slip-leg-remove" onclick="slip.remove('${esc(l.key).replace(/'/g,"\\'")}')">×</button>
      </div>
    `).join('');
  },

  _update() {
    const n = this.legs.length;
    ['slipCount','slipToggleCount','slipMobileCount'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.textContent = n;
    });
    const combined = this.legs.reduce((acc, l) => acc * parseFloat(l.odds), 1);
    const oddsStr = n ? combined.toFixed(2) : '--';
    ['slipOdds','slipMobileOdds'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.textContent = oddsStr;
    });
    this._calcPayout();

    // mobile bar
    const bar = document.getElementById('slipMobileBar');
    if (bar) bar.classList.toggle('hidden', n === 0);
  },

  _calcPayout() {
    const stake = parseFloat(document.getElementById('slipStake').value) || 0;
    const combined = this.legs.reduce((acc, l) => acc * parseFloat(l.odds), 1);
    const payout = stake * combined;
    document.getElementById('slipPayout').textContent = '$' + (stake ? payout.toFixed(2) : '0.00');
  }
};

document.getElementById('slipStake').addEventListener('input', () => slip._calcPayout());
slip._render();
slip._update();

// ----- App Module -----
const appModule = {
  LEAGUES: [
    { id: 'EPL', name: 'Premier League', color: '#3D195B' },
    { id: 'LaLiga', name: 'La Liga', color: '#EE8707' },
    { id: 'SerieA', name: 'Serie A', color: '#024494' },
    { id: 'Bundesliga', name: 'Bundesliga', color: '#D20515' },
    { id: 'Ligue1', name: 'Ligue 1', color: '#DAE025' },
    { id: 'UCL', name: 'Champions League', color: '#001489' },
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
  initialized: false,
  _requestId: 0,

  async init() {
    if (!this.initialized) {
      this._buildDateStrips();
      this._buildLeagueFilters();
      this.initialized = true;
    }
    this.switchTab('fixtures');
    await this._loadMatchday();
  },

  switchTab(tab) {
    this.currentTab = tab;
    document.querySelectorAll('.app-view').forEach(v => v.classList.add('hidden'));
    document.querySelectorAll('.app-nav[data-view]').forEach(b => {
      b.classList.toggle('active', b.dataset.view === tab);
    });
    if (tab === 'fixtures') document.getElementById('viewFixtures').classList.remove('hidden');
    else if (tab === 'best-bets') document.getElementById('viewBestBets').classList.remove('hidden');
    else if (tab === 'match') document.getElementById('viewMatch').classList.remove('hidden');
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
        btn.onclick = () => this._setActiveDate(i);
        strip.appendChild(btn);
      }
    });
  },

  _setActiveDate(offset) {
    this.activeDate = offset;
    document.querySelectorAll('.date-chip').forEach(btn => {
      btn.classList.toggle('active', Number(btn.dataset.dateOffset) === offset);
    });
    this._loadMatchday();
  },

  _buildLeagueFilters() {
    const filters = document.getElementById('leagueFilters');
    if (!filters) return;
    filters.innerHTML = '';
    const allBtn = document.createElement('button');
    allBtn.className = 'league-chip active';
    allBtn.innerHTML = 'All Leagues';
    allBtn.onclick = () => { this.activeLeague = 'all'; this._syncLeagueChips(allBtn); this._loadMatchday(); };
    filters.appendChild(allBtn);
    this.LEAGUES.forEach(l => {
      const btn = document.createElement('button');
      btn.className = 'league-chip';
      btn.innerHTML = `<span class="league-dot" style="background:${l.color}"></span>${l.name}`;
      btn.onclick = () => { this.activeLeague = l.id; this._syncLeagueChips(btn); this._loadMatchday(); };
      filters.appendChild(btn);
    });
  },

  _syncLeagueChips(activeBtn) {
    document.querySelectorAll('#leagueFilters .league-chip').forEach(c => c.classList.remove('active'));
    activeBtn.classList.add('active');
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

  _buildBestBetsFromFixtures() {
    const confidenceRank = { high: 3, medium: 2, low: 1 };
    return this.fixtures
      .filter(fixture => fixture.best && ['high', 'medium'].includes(fixture.conf))
      .sort((a, b) => (
        (confidenceRank[b.conf] || 0) - (confidenceRank[a.conf] || 0)
        || (b.edge || 0) - (a.edge || 0)
      ))
      .slice(0, 10);
  },

  async _loadMatchday() {
    const container = document.getElementById('fixturesContainer');
    const bestBetsContainer = document.getElementById('bestBetsContainer');
    if (!container || !bestBetsContainer) return;

    const requestId = ++this._requestId;
    const targetDate = this._selectedDateISO();
    const requestedLeagues = this.activeLeague === 'all'
      ? this.LEAGUES
      : this.LEAGUES.filter(league => league.id === this.activeLeague);

    this.fixtures = [];
    this.bestBets = [];
    this._renderFixtures({ loading: true });
    this._renderBestBets({ loading: true });

    const responses = await Promise.allSettled(requestedLeagues.map(async league => {
      const response = await fetch(`/api/fixtures/${encodeURIComponent(league.id)}/${targetDate}`);
      if (!response.ok) throw new Error(`${league.name} could not be loaded (${response.status})`);
      const payload = await response.json();
      return { league, payload };
    }));

    // A date/filter could change before a slow prediction request completes.
    if (requestId !== this._requestId) return;

    const successful = responses.filter(result => result.status === 'fulfilled');
    this.fixtures = successful.flatMap(result => (
      result.value.payload.fixtures.map(fixture => this._normaliseFixture(fixture, result.value.league.id))
    ));
    this.bestBets = this._buildBestBetsFromFixtures();

    if (!successful.length) {
      this._renderFixtures({ error: 'Predictions could not be loaded. Please try again.' });
      this._renderBestBets({ error: 'Best bets are unavailable while matchday data is loading.' });
      return;
    }

    const failed = responses.length - successful.length;
    this._renderFixtures({ notice: failed ? `${failed} league${failed === 1 ? '' : 's'} could not be loaded.` : '' });
    this._renderBestBets();
  },

  _renderFixtures(state = {}) {
    const container = document.getElementById('fixturesContainer');
    if (!container) return;
    if (state.loading) {
      container.innerHTML = '<div class="empty-state"><p>Preparing the matchday board…</p><span class="empty-state-hint">Loading live fixtures, prices and model decisions.</span></div>';
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
      container.innerHTML += '<div class="empty-state"><div class="empty-state-icon"><svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2"><circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/></svg></div><p>No fixtures for this selection.</p><span class="empty-state-hint">There may be no matches in the selected leagues on this date.</span></div>';
      return;
    }
    Object.entries(byLeague).forEach(([leagueId, fxs]) => {
      const league = this.LEAGUES.find(l => l.id === leagueId) || { name: leagueId, color: '#666' };
      const group = document.createElement('div');
      group.className = 'league-group';
      group.innerHTML = `<div class="league-header"><div class="league-header-left"><span class="league-stripe" style="background:${esc(league.color)}"></span><span class="league-name">${esc(league.name)}</span></div><span class="league-count mono">${fxs.length} matches</span></div>`;
      fxs.forEach((f, i) => {
        const slipKey = `${f.home} vs ${f.away}|${f.pick}`;
        const inSlip = slip.legs.find(l => l.key === slipKey);
        const row = document.createElement('div');
        row.className = 'fixture-row';
        const confidenceLabel = f.best ? f.conf.toUpperCase() : 'NO BET';
        const addButton = f.best
          ? `<button class="fixture-add-btn${inSlip ? ' added' : ''}" data-fixture-add="${esc(f.id)}">${inSlip ? '✓' : '+'}</button>`
          : '<span class="fixture-add-placeholder" aria-hidden="true"></span>';
        row.innerHTML = `
          <span class="fixture-num mono">${i + 1}</span>
          <div class="fixture-match">
            <div class="fixture-teams">${esc(f.home)} <span class="fixture-vs">vs</span> ${esc(f.away)}</div>
            <div style="font-size:11px;color:var(--text3);margin-top:2px;font-family:var(--mono)">${esc(f.time)}</div>
          </div>
          <span class="fixture-pick-text">${esc(f.pick)}</span>
          <span class="conf-badge ${esc(f.conf)}">${esc(confidenceLabel)}</span>
          <span class="fixture-odds-val mono">${f.odds ? esc(f.odds.toFixed(2)) : '—'}</span>
          <span class="fixture-edge mono ${f.edge && f.edge > 0 ? 'edge-pos' : ''}">${f.edge ? `+${esc(f.edge.toFixed(1))}%` : '—'}</span>
          ${addButton}
        `;
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

  _toggleSlipFromFixture(fxId) {
    const f = this.fixtures.find(x => String(x.id) === String(fxId));
    if (!f || !f.best || !Number.isFinite(f.odds)) return;
    const slipKey = `${f.home} vs ${f.away}|${f.pick}`;
    const added = slip.add(f.home + ' vs ' + f.away, f.pick, f.odds, 'Best Pick');
    document.querySelectorAll('[data-fixture-add]').forEach(btn => {
      if (btn.dataset.fixtureAdd !== String(f.id)) return;
      if (added === false) { btn.textContent = '+'; btn.classList.remove('added'); }
      else { btn.textContent = '✓'; btn.classList.add('added'); }
    });
  },

  _showMatch(fxId) {
    const f = this.fixtures.find(x => String(x.id) === String(fxId));
    if (!f) return;
    this.currentMatch = f;
    this.currentMarketTab = Object.keys(f.markets)[0];
    const league = this.LEAGUES.find(l => l.id === f.league) || { name: f.league };

    document.getElementById('matchHeader').innerHTML = `
      <div class="match-hero">
        <div class="match-league-badge">${esc(league.name)}</div>
        <div class="match-teams-title">${esc(f.home)} <span class="match-teams-vs">vs</span> ${esc(f.away)}</div>
        <div class="match-meta">${esc(f.time)} · ${esc(this._selectedDateISO())}</div>
      </div>
    `;

    this._buildMarketTabs(f);
    if (this.currentMarketTab) this._renderMarkets(f, this.currentMarketTab);
    else document.getElementById('matchMarkets').innerHTML = '<div class="empty-state"><p>No market decisions are available for this fixture yet.</p></div>';
    this.switchTab('match');
  },

  _buildMarketTabs(f) {
    const tabs = document.getElementById('marketTabs');
    tabs.innerHTML = '';
    const marketLabels = { goals: 'Goals', btts: 'BTTS', corners: 'Corners', cards: 'Cards', sot: 'Shots on target', moneyline: 'Moneyline', spreads: 'Handicap' };
    Object.keys(f.markets).forEach(mk => {
      const btn = document.createElement('button');
      btn.className = 'market-tab' + (mk === this.currentMarketTab ? ' active' : '');
      btn.textContent = marketLabels[mk] || mk;
      btn.onclick = () => {
        this.currentMarketTab = mk;
        tabs.querySelectorAll('.market-tab').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this._renderMarkets(f, mk);
      };
      tabs.appendChild(btn);
    });
  },

  _renderMarkets(f, marketKey) {
    const lines = f.markets[marketKey] || [];
    const container = document.getElementById('matchMarkets');
    const marketLabels = { goals: 'Goals', btts: 'BTTS', corners: 'Corners', cards: 'Cards', sot: 'Shots on target', moneyline: 'Moneyline', spreads: 'Handicap' };
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
            ? `<button class="market-add-btn${inSlip ? ' added' : ''}" data-market-add="${index}">${inSlip ? '✓ Added' : '+ Add'}</button>`
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
              <span class="market-line-odds mono">${line.odds ? esc(line.odds.toFixed(2)) : '—'}</span>
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
    const f = this.fixtures.find(x => String(x.id) === String(fxId));
    if (!f || !Number.isFinite(odds)) return;
    const slipKey = `${f.home} vs ${f.away}|${pick}`;
    const added = slip.add(f.home + ' vs ' + f.away, pick, odds, 'Market');
    document.querySelectorAll('[data-market-add]').forEach(btn => {
      if (added === false) { btn.textContent = '+ Add'; btn.classList.remove('added'); }
      else { btn.textContent = '✓ Added'; btn.classList.add('added'); }
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
      container.innerHTML = '<div class="empty-state"><p>No medium- or high-confidence recommendations for this selection.</p><span class="empty-state-hint">No bet is a valid result when the current lines do not qualify.</span></div>';
      return;
    }
    const league = id => this.LEAGUES.find(l => l.id === id) || { name: id, color: '#666' };
    container.innerHTML = this.bestBets.map(f => {
      const slipKey = `${f.home} vs ${f.away}|${f.pick}`;
      const inSlip = slip.legs.find(l => l.key === slipKey);
      return `<div class="fixture-row" style="cursor:pointer">
        <span class="fixture-num mono" style="background:${esc(league(f.league).color)};border-radius:3px;padding:2px 4px;font-size:9px;color:#fff">${esc(league(f.league).name)}</span>
        <div class="fixture-match">
          <div class="fixture-teams">${esc(f.home)} <span class="fixture-vs">vs</span> ${esc(f.away)}</div>
          <div style="font-size:11px;color:var(--text3);margin-top:2px">${esc(f.pick)}</div>
        </div>
        <span class="conf-badge ${esc(f.conf)}">${esc(f.conf.toUpperCase())}</span>
        <span class="fixture-odds-val mono">${esc(f.odds.toFixed(2))}</span>
        <span class="fixture-edge mono edge-pos">+${esc(f.edge.toFixed(1))}%</span>
        <button class="fixture-add-btn${inSlip ? ' added' : ''}" data-best-bet-add="${esc(f.id)}">${inSlip ? '✓' : '+'}</button>
      </div>`;
    }).join('');
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
