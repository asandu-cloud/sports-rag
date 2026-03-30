
// ============================================================
// SpickBot — stub JS for preview (app.js not included)
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
  subscribe(plan) {
    // TODO: BACKEND — redirect to payment session
    // Suggested: POST /api/checkout/session { plan }
    // Then redirect to Stripe/LemonSqueezy checkout URL
    showToast('Redirecting to checkout...', 'success');
    // Example:
    // fetch('/api/checkout/session', { method: 'POST', body: JSON.stringify({ plan }), headers: { 'Content-Type': 'application/json' } })
    //   .then(r => r.json()).then(d => window.location.href = d.url);
  }
};

// ----- Auth stub -----
const auth = {
  // TODO: BACKEND — on page load, fetch /api/me and if logged in:
  //   document.getElementById('loginBtn').classList.add('hidden');
  //   document.getElementById('userBadge').classList.remove('hidden');
  //   document.getElementById('userAvatar').src = user.avatar;
  //   document.getElementById('userName').textContent = user.name;
  showMenu() { showToast('Account menu coming soon', 'info'); }
};

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
        <div class="slip-leg-fixture">${l.fixture}</div>
        <div class="slip-leg-pick">${l.pick}</div>
        <div class="slip-leg-odds mono">${l.odds}</div>
        <button class="slip-leg-remove" onclick="slip.remove('${l.key.replace(/'/g,"\'")}')">×</button>
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
    { id: 'epl', name: 'EPL', color: '#3D195B' },
    { id: 'laliga', name: 'La Liga', color: '#EE8707' },
    { id: 'seriea', name: 'Serie A', color: '#024494' },
    { id: 'bundesliga', name: 'Bundesliga', color: '#D20515' },
    { id: 'ligue1', name: 'Ligue 1', color: '#DAE025' },
    { id: 'ucl', name: 'UCL', color: '#001489' },
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

  init() {
    this._buildDateStrips();
    this._buildLeagueFilters();
    // TODO: BACKEND — replace this._renderFixtures() with live data fetch
    // Suggested: GET /api/fixtures?date=YYYY-MM-DD&league=all
    // Then call this._renderFixtures(data) with the response
    this._renderFixtures();
    this._buildBestBets();
    this.switchTab('fixtures');
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
        btn.onclick = () => {
          strip.querySelectorAll('.date-chip').forEach(c => c.classList.remove('active'));
          btn.classList.add('active');
          this.activeDate = i;
          this._renderFixtures();
        };
        strip.appendChild(btn);
      }
    });
  },

  _buildLeagueFilters() {
    const filters = document.getElementById('leagueFilters');
    if (!filters) return;
    filters.innerHTML = '';
    const allBtn = document.createElement('button');
    allBtn.className = 'league-chip active';
    allBtn.innerHTML = 'All Leagues';
    allBtn.onclick = () => { this.activeLeague = 'all'; this._syncLeagueChips(allBtn); this._renderFixtures(); };
    filters.appendChild(allBtn);
    this.LEAGUES.forEach(l => {
      const btn = document.createElement('button');
      btn.className = 'league-chip';
      btn.innerHTML = `<span class="league-dot" style="background:${l.color}"></span>${l.name}`;
      btn.onclick = () => { this.activeLeague = l.id; this._syncLeagueChips(btn); this._renderFixtures(); };
      filters.appendChild(btn);
    });
  },

  _syncLeagueChips(activeBtn) {
    document.querySelectorAll('#leagueFilters .league-chip').forEach(c => c.classList.remove('active'));
    activeBtn.classList.add('active');
  },

  _renderFixtures() {
    const container = document.getElementById('fixturesContainer');
    if (!container) return;
    const fixtures = this.activeLeague === 'all' ? this.FIXTURES : this.FIXTURES.filter(f => f.league === this.activeLeague);
    const byLeague = {};
    fixtures.forEach(f => { (byLeague[f.league] = byLeague[f.league] || []).push(f); });
    container.innerHTML = '';
    if (!fixtures.length) {
      container.innerHTML = '<div class="empty-state"><div class="empty-state-icon"><svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2"><circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/></svg></div><p>No fixtures for this selection.</p></div>';
      return;
    }
    Object.entries(byLeague).forEach(([leagueId, fxs], gi) => {
      const league = this.LEAGUES.find(l => l.id === leagueId) || { name: leagueId, color: '#666' };
      const group = document.createElement('div');
      group.className = 'league-group';
      group.innerHTML = `<div class="league-header"><div class="league-header-left"><span class="league-stripe" style="background:${league.color}"></span><span class="league-name">${league.name}</span></div><span class="league-count mono">${fxs.length} matches</span></div>`;
      fxs.forEach((f, i) => {
        const slipKey = f.id + '|' + f.pick;
        const inSlip = slip.legs.find(l => l.key === slipKey);
        const row = document.createElement('div');
        row.className = 'fixture-row';
        row.innerHTML = `
          <span class="fixture-num mono">${i + 1}</span>
          <div class="fixture-match">
            <div class="fixture-teams">${f.home} <span class="fixture-vs">vs</span> ${f.away}</div>
            <div style="font-size:11px;color:var(--text3);margin-top:2px;font-family:var(--mono)">${f.time}</div>
          </div>
          <span class="fixture-pick-text">${f.pick}</span>
          <span class="conf-badge ${f.conf}">${f.conf.toUpperCase()}</span>
          <span class="fixture-odds-val mono">${f.odds}</span>
          <span class="fixture-edge mono ${f.edge > 0 ? 'edge-pos' : ''}">+${f.edge}%</span>
          <button class="fixture-add-btn${inSlip ? ' added' : ''}" data-slip-key="${slipKey}"
            onclick="event.stopPropagation();appModule._toggleSlipFromFixture(${f.id})">
            ${inSlip ? '✓' : '+'}
          </button>
        `;
        row.onclick = () => this._showMatch(f.id);
        group.appendChild(row);
      });
      container.appendChild(group);
    });
  },

  _toggleSlipFromFixture(fxId) {
    const f = this.FIXTURES.find(x => x.id === fxId);
    if (!f) return;
    const slipKey = f.id + '|' + f.pick;
    const btn = document.querySelector(`[data-slip-key="${slipKey}"]`);
    const added = slip.add(f.home + ' vs ' + f.away, f.pick, f.odds, 'Best Pick');
    if (btn) {
      if (added === false) { btn.textContent = '+'; btn.classList.remove('added'); }
      else { btn.textContent = '✓'; btn.classList.add('added'); }
    }
  },

  _showMatch(fxId) {
    const f = this.FIXTURES.find(x => x.id === fxId);
    if (!f) return;
    this.currentMatch = f;
    this.currentMarketTab = Object.keys(f.markets)[0];
    const league = this.LEAGUES.find(l => l.id === f.league) || { name: f.league };

    document.getElementById('matchHeader').innerHTML = `
      <div class="match-hero">
        <div class="match-league-badge">${league.name}</div>
        <div class="match-teams-title">${f.home} <span class="match-teams-vs">vs</span> ${f.away}</div>
        <div class="match-meta">${f.time} · Today</div>
      </div>
    `;

    this._buildMarketTabs(f);
    this._renderMarkets(f, this.currentMarketTab);
    this.switchTab('match');
  },

  _buildMarketTabs(f) {
    const tabs = document.getElementById('marketTabs');
    tabs.innerHTML = '';
    const marketLabels = { goals: 'Goals', btts: 'BTTS', corners: 'Corners', cards: 'Cards', moneyline: 'Moneyline' };
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
    const marketLabels = { goals: 'Goals', btts: 'BTTS', corners: 'Corners', cards: 'Cards', moneyline: 'Moneyline' };
    container.innerHTML = `
      <div class="market-card">
        <div class="market-card-header">
          <span class="market-card-title">${marketLabels[marketKey] || marketKey}</span>
        </div>
        ${lines.map(line => {
          const slipKey = f.id + '|' + line.pick;
          const inSlip = slip.legs.find(l => l.key === slipKey);
          return `<div class="market-line${line.rec ? ' recommended' : ''}">
            <div class="market-line-left">
              <span class="market-line-pick">${line.pick}</span>
              ${line.proj ? `<span class="market-line-proj">${line.proj}</span>` : ''}
              ${line.conf ? `<span class="conf-badge ${line.conf}">${line.conf.toUpperCase()}</span>` : ''}
              ${line.edge > 0 ? `<span class="value-badge ${line.edge > 6 ? 'strong' : 'good'}">+${line.edge}%</span>` : ''}
            </div>
            <div class="market-line-right">
              <span class="market-line-odds mono">${line.odds}</span>
              <button class="market-add-btn${inSlip ? ' added' : ''}" data-slip-key="${slipKey}"
                onclick="appModule._toggleSlipFromMarket(${f.id},'${line.pick}',${line.odds})">
                ${inSlip ? '✓ Added' : '+ Add'}
              </button>
            </div>
          </div>`;
        }).join('')}
      </div>
    `;
  },

  _toggleSlipFromMarket(fxId, pick, odds) {
    const f = this.FIXTURES.find(x => x.id === fxId);
    if (!f) return;
    const slipKey = fxId + '|' + pick;
    const btn = document.querySelector(`[data-slip-key="${slipKey}"]`);
    const added = slip.add(f.home + ' vs ' + f.away, pick, odds, 'Market');
    if (btn) {
      if (added === false) { btn.textContent = '+ Add'; btn.classList.remove('added'); }
      else { btn.textContent = '✓ Added'; btn.classList.add('added'); }
    }
  },

  _buildBestBets() {
    const container = document.getElementById('bestBetsContainer');
    if (!container) return;
    const allPicks = [];
    this.FIXTURES.forEach(f => {
      Object.entries(f.markets).forEach(([mk, lines]) => {
        lines.filter(l => l.rec && l.edge > 4).forEach(l => {
          allPicks.push({ fixture: f, market: mk, line: l });
        });
      });
    });
    allPicks.sort((a, b) => b.line.edge - a.line.edge);
    const league = id => this.LEAGUES.find(l => l.id === id) || { name: id, color: '#666' };
    container.innerHTML = allPicks.slice(0, 10).map(({ fixture: f, market, line }) => {
      const slipKey = f.id + '|' + line.pick;
      const inSlip = slip.legs.find(l => l.key === slipKey);
      return `<div class="fixture-row" style="cursor:pointer">
        <span class="fixture-num mono" style="background:${league(f.league).color};border-radius:3px;padding:2px 4px;font-size:9px;color:#fff">${league(f.league).name}</span>
        <div class="fixture-match">
          <div class="fixture-teams">${f.home} <span class="fixture-vs">vs</span> ${f.away}</div>
          <div style="font-size:11px;color:var(--text3);margin-top:2px">${line.pick}</div>
        </div>
        <span class="conf-badge ${line.conf}">${line.conf.toUpperCase()}</span>
        <span class="fixture-odds-val mono">${line.odds}</span>
        <span class="fixture-edge mono edge-pos">+${line.edge}%</span>
        <button class="fixture-add-btn${inSlip ? ' added' : ''}" data-slip-key="${slipKey}"
          onclick="event.stopPropagation();appModule._toggleSlipFromMarket(${f.id},'${line.pick}',${line.odds})">
          ${inSlip ? '✓' : '+'}
        </button>
      </div>`;
    }).join('');
  }
};

const app = appModule;
