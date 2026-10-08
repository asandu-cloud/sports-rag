import {catalogues, rankMetrics, rankGroups, frequencyLabels} from './data.js?v=real-20261008a';

const API = '/api/research';

function mountResearch(root, navigate) {
  const $ = id => root.getElementById(id);
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
  const fmt = (value, digits = 1) => Number.isFinite(value) ? value.toLocaleString('en-GB', {maximumFractionDigits: digits}) : 'Unavailable';
  const fixed = (value, digits = 2) => Number.isFinite(value) ? value.toFixed(digits) : '–';
  const whole = value => Number.isFinite(value) ? value.toLocaleString('en-GB') : '–';
  const day = iso => new Date(iso).toLocaleDateString('en-GB', {day: 'numeric', month: 'short', timeZone: 'UTC'});
  const dayYear = iso => new Date(iso).toLocaleDateString('en-GB', {day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC'});
  const initials = name => String(name || '').split(/[\s.\-]+/).filter(Boolean).map(w => w[0]).join('').slice(0, 2).toUpperCase();
  const cap = text => text ? text[0].toUpperCase() + text.slice(1) : '';
  const plural = (n, word, many = word + 's') => `${whole(n)} ${n === 1 ? word : many}`;
  const seasonLabel = year => `${year}/${String((year + 1) % 100).padStart(2, '0')}`;
  const short = team => team?.short_code || initials(team?.name) || '?';

  let metrics = catalogues.players.metrics, common = catalogues.players.common, initialized = false;
  let searchTimer = 0, searchController = null, routeToken = 0;
  const state = {area: 'players', view: 'search', profileKey: null, profile: null, id: null, scope: null, records: [],
    stat: null, period: '10', venue: 'all', rate: 'match', thresholdOn: false, threshold: 1, compare: false, selected: null};
  const current = () => catalogues[state.area];
  const isPlayer = () => state.area === 'players';
  const isReferee = () => state.area === 'referees';

  // ----- Data -----
  const cache = new Map();
  function load(url) {
    if (!cache.has(url)) {
      cache.set(url, fetch(url, {headers: {Accept: 'application/json'}}).then(response => {
        if (!response.ok) { const error = new Error('Research request failed'); error.status = response.status; throw error; }
        return response.json();
      }).catch(error => { cache.delete(url); throw error; }));
    }
    return cache.get(url);
  }
  const isScope = part => /^([A-Za-z0-9_]+-)?\d{4}$/.test(part);
  function profileUrl(area, id, scope) {
    const params = new URLSearchParams();
    if (scope) {
      const cut = scope.lastIndexOf('-');
      if (cut > 0) params.set('competition', scope.slice(0, cut));
      params.set('season', scope.slice(cut + 1));
    }
    const query = params.toString();
    return `${API}/${area}/${encodeURIComponent(id)}${query ? '?' + query : ''}`;
  }
  const scopeOf = profile => isReferee() ? String(profile.season.year) : `${profile.competition.code}-${profile.season.year}`;
  function go(area, id = null, scope = null, stat = null) {
    navigate([area, id && encodeURIComponent(id), id && scope, id && stat].filter(Boolean).join('/'));
  }

  // ----- Shared pieces -----
  const badgeInner = (src, name) => src ? `<img src="${escape(src)}" alt="" loading="lazy" data-fallback="${escape(initials(name))}">` : escape(initials(name));
  const badge = (src, name, square = false, large = true) =>
    `<span class="avatar${large ? ' avatar-large' : ''}${square ? ' avatar-crest' : ''}" aria-hidden="true">${badgeInner(src, name)}</span>`;
  function wireImages(container) {
    container.querySelectorAll('img[data-fallback]').forEach(img => {
      const fallback = () => { img.parentElement.textContent = img.dataset.fallback; };
      if (img.complete && !img.naturalWidth) fallback(); else img.addEventListener('error', fallback, {once: true});
    });
  }
  const tip = () => $('tip');
  function wireTips(container) {
    container.querySelectorAll('[data-tip]').forEach(el => {
      const show = () => {
        const box = tip(); box.textContent = el.dataset.tip; box.hidden = false;
        const r = el.getBoundingClientRect(), w = box.offsetWidth;
        box.style.left = Math.max(8, Math.min(window.innerWidth - w - 8, r.left + r.width / 2 - w / 2)) + 'px';
        box.style.top = Math.max(8, r.top - box.offsetHeight - 8) + 'px';
      };
      const hide = () => { tip().hidden = true; };
      el.addEventListener('pointerenter', show); el.addEventListener('pointerleave', hide);
      el.addEventListener('focus', show); el.addEventListener('blur', hide);
    });
  }
  function view(name) {
    state.view = name;
    $('searchView').hidden = name !== 'search';
    $('profileView').hidden = name !== 'profile';
    $('playerView').hidden = name !== 'chart';
    root.host.dataset.profile = String(name !== 'search');
    if (tip()) tip().hidden = true;
  }

  // ----- Search -----
  function setArea(area) {
    state.area = catalogues[area] ? area : 'players';
    metrics = current().metrics; common = current().common;
    Object.assign(state, {profileKey: null, profile: null, records: [], stat: current().defaultStat});
    $('playerSearch').value = '';
    $('searchTitle').textContent = `Find a ${current().singular}`;
    $('searchDescription').textContent = current().description;
    $('playerSearch').placeholder = current().search;
    $('playerSearch').setAttribute('aria-label', current().search);
    $('profileBackLabel').textContent = `Find another ${current().singular}`;
    $('venueField').hidden = isReferee();
    $('rateField').hidden = !isPlayer();
    $('filterButton').hidden = isReferee();
    $('contextHeading').textContent = isPlayer() ? 'Time on the pitch' : state.area === 'teams' ? 'Results' : 'Appointments';
    $('researchIntroduction').textContent = current().introduction;
    $('researchTopics').textContent = current().topics;
    $('dnpLegend').hidden = true;
    root.querySelectorAll('[data-area]').forEach(a => { if (a.dataset.area === state.area) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); });
    $('extraStats').innerHTML = current().groups.map(([title, keys]) => `<div><h3>${title}</h3>${keys.map(key => `<button data-stat="${key}">${metrics[key].label}</button>`).join('')}</div>`).join('');
    renderSuggestions();
  }
  function renderSuggestions() {
    $('resultsTitle').textContent = 'Try a search';
    $('resultCount').textContent = '';
    $('searchHelp').textContent = 'Type at least two letters of a name.';
    $('playerResults').innerHTML = `<div class="search-suggestions"><p>Popular searches</p><div>${current().suggestions.map(s => `<button class="suggestion" type="button" data-suggest="${escape(s)}">${escape(s)}</button>`).join('')}</div></div>`;
  }
  function search() {
    const query = $('playerSearch').value.trim();
    clearTimeout(searchTimer); searchController?.abort();
    if (query.length < 2) { $('playerResults').setAttribute('aria-busy', 'false'); renderSuggestions(); return; }
    $('playerResults').setAttribute('aria-busy', 'true');
    const area = state.area;
    searchTimer = setTimeout(async () => {
      const controller = searchController = new AbortController();
      try {
        const response = await fetch(`${API}/search?type=${area}&q=${encodeURIComponent(query)}&limit=12`, {signal: controller.signal});
        if (!response.ok) throw new Error('Search failed');
        const data = await response.json();
        if (controller === searchController && area === state.area) renderResults(query, data.results || []);
      } catch (error) {
        if (error.name === 'AbortError') return;
        $('playerResults').innerHTML = '<div class="search-empty"><h3>Search is unavailable right now</h3><p>Please try again in a moment.</p></div>';
      } finally {
        if (controller === searchController) $('playerResults').setAttribute('aria-busy', 'false');
      }
    }, 200);
  }
  function resultRow(r) {
    let avatar, sub, side = '';
    if (state.area === 'players') {
      avatar = badge(r.api_football_id ? `https://media.api-sports.io/football/players/${r.api_football_id}.png` : null, r.name, false, false);
      sub = [r.club?.name, r.latest_competition && `${r.latest_competition} ${seasonLabel(r.latest_season)}`];
      side = r.position?.label || '';
    } else if (state.area === 'teams') {
      avatar = badge(r.logo_url, r.name, true, false);
      sub = [r.country, r.latest_competition && `${r.latest_competition} ${seasonLabel(r.latest_season)}`];
    } else {
      avatar = badge(null, r.name, false, false);
      sub = [r.country, plural(r.matches, 'match', 'matches')];
      side = r.last_activity_at ? `Last match ${dayYear(r.last_activity_at)}` : '';
    }
    const id = r.id ?? r.referee_key;
    return `<button class="player-result" type="button" data-id="${escape(id)}">${avatar}<span class="player-result-copy"><strong>${escape(r.name)}</strong><small>${escape(sub.filter(Boolean).join(' · '))}</small></span><span class="player-position">${escape(side)}</span><svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><path d="m9 5 7 7-7 7"/></svg></button>`;
  }
  function renderResults(query, results) {
    $('resultsTitle').textContent = 'Search results';
    $('resultCount').textContent = plural(results.length, current().singular, current().label.toLowerCase());
    $('searchHelp').textContent = 'Results are ordered by most recent match.';
    $('playerResults').innerHTML = results.length ? results.map(resultRow).join('')
      : `<div class="search-empty"><h3>No ${current().label.toLowerCase()} match “${escape(query)}”</h3><p>Try a surname or a shorter spelling. Players appear once they have played a match in our leagues or European cups.</p></div>`;
    wireImages($('playerResults'));
  }
  function showSearch() {
    const fromProfile = state.view !== 'search';
    view('search');
    Object.assign(state, {profileKey: null, profile: null, id: null, scope: null});
    if (fromProfile) $('playerSearch').focus({preventScroll: true});
  }

  // ----- Routing -----
  async function fromRoute(route = 'players') {
    const parts = route.split('/').map(part => decodeURIComponent(part));
    const area = catalogues[parts[0]] ? parts[0] : 'players';
    const id = parts[1] || null, rest = parts.slice(2);
    const scope = rest.find(isScope) || null, stat = rest.find(part => !isScope(part)) || null;
    if (state.area !== area || !initialized) { initialized = true; setArea(area); }
    if (!id) return showSearch();
    const token = ++routeToken, key = `${area}/${id}/${scope || ''}`;
    if (state.profileKey !== key) {
      Object.assign(state, {profileKey: key, profile: null, rendered: null, id, scope});
      view('profile');
      $('profileBody').innerHTML = '<p class="profile-loading" role="status">Loading profile…</p>';
    }
    let profile;
    try {
      profile = await load(profileUrl(area, id, scope));
    } catch (error) {
      if (token !== routeToken) return;
      state.profileKey = null;
      view('profile');
      $('profileBody').innerHTML = error.status === 404
        ? `<div class="search-empty"><h3>No finished matches for this selection</h3><p>Try another season or search again.</p></div>`
        : `<div class="search-empty"><h3>This profile could not load</h3><p>Please try again in a moment.</p><button class="text-button" type="button" data-retry>Try again</button></div>`;
      return;
    }
    if (token !== routeToken) return;
    if (state.profile !== profile) { state.profile = profile; state.records = buildRecords(profile); }
    if (stat && metrics[stat]) openChart(stat); else showProfile();
  }

  // ----- Profile -----
  const ROLE = {regular_starter: 'Regular starter', rotation: 'Rotation player', mostly_substitute: 'Mostly a substitute'};
  function showProfile() {
    const changed = state.rendered !== state.profile;
    view('profile');
    if (changed) {
      state.rendered = state.profile;
      $('profileBody').innerHTML = renderProfile(state.profile);
      wireImages($('profileBody')); wireTips($('profileBody'));
      $('profileBody').querySelector('.p-name')?.focus({preventScroll: true});
    }
  }
  function renderProfile(d) {
    const kind = state.area;
    const pair = (a, b) => a && b ? `<div class="grid-2">${a}${b}</div>` : (a || b || '');
    const stack = (...cards) => { const list = cards.filter(Boolean); return list.length > 1 ? `<div class="stack">${list.join('')}</div>` : list[0] || ''; };
    const top = profileHead(d) + profileLead(d) + seasonLine(d) + rankNotice(d);
    let body;
    if (kind === 'players') body = pair(rankCard(d), stack(howOftenCard(d), minutesCard(d))) + pair(lastTenCard(d), funnelCard(d));
    else if (kind === 'teams') body = pair(rankCard(d), stack(howOftenCard(d), leadersCard(d))) + pair(lastTenCard(d), homeAwayCard(d));
    else body = pair(rankCard(d), stack(howOftenCard(d), competitionCard(d))) + lastTenCard(d);
    return top + body + advancedBlock(d) + dataFoot(d);
  }
  function scopePicker(d) {
    const scopes = d.available_scopes || [];
    if (scopes.length < 2) return `<span class="scope-single">${escape(d.season.label)}${d.competition ? ' · ' + escape(d.competition.name) : ''}</span>`;
    const selected = scopeOf(d);
    const options = scopes.map(s => {
      const value = isReferee() ? String(s.season.year) : `${s.competition.code}-${s.season.year}`;
      const label = isReferee() ? `${s.season.label} · ${plural(s.matches, 'match', 'matches')}` : `${s.season.label} · ${s.competition.name} (${s.matches})`;
      return `<option value="${escape(value)}"${value === selected ? ' selected' : ''}>${escape(label)}</option>`;
    }).join('');
    return `<label class="scope-picker"><span>Season</span><select id="scopeSelect">${options}</select></label>`;
  }
  function profileHead(d) {
    const s = d.subject, kind = state.area;
    let pic, meta, chips = [];
    if (kind === 'players') {
      pic = badge(s.photo_url, s.name);
      meta = [s.club?.name, s.position?.label, d.competition?.name];
      const role = d.role_summary;
      if (role?.role) chips.push(`<span class="chip${role.role === 'regular_starter' ? ' good' : ''}">${ROLE[role.role]} · started ${role.starts} of ${role.appearances}</span>`);
      if (Number.isFinite(role?.average_minutes_per_appearance)) chips.push(`<span class="chip">${Math.round(role.average_minutes_per_appearance)} min per appearance</span>`);
    } else if (kind === 'teams') {
      pic = badge(s.logo_url, s.name, true);
      meta = [d.competition?.name, s.country];
      const t = d.totals;
      chips.push(`<span class="chip">${t.wins} won · ${t.draws} drawn · ${t.losses} lost</span>`, `<span class="chip">${t.goals_for} scored · ${t.goals_against} conceded</span>`);
      const form = d.last_5_form || [];
      if (form.length) chips.push(`<span class="form" aria-label="Last ${form.length} results, oldest first: ${form.map(r => ({W: 'won', D: 'drew', L: 'lost'}[r] || 'unknown')).join(', ')}">${form.map(r => `<b class="${(r || '').toLowerCase()}">${escape(r || '?')}</b>`).join('')}</span>`);
    } else {
      pic = badge(null, s.name);
      meta = ['Referee', s.country, `${plural(d.totals.matches, 'match', 'matches')} in ${d.season.label}`];
      chips = (d.competition_breakdown || []).map(c => `<span class="chip">${escape(c.competition)} · ${c.matches}</span>`);
    }
    return `<div class="p-head">${pic}<div class="p-title"><h2 class="p-name" tabindex="-1">${escape(s.name)}</h2><div class="p-meta">${meta.filter(Boolean).map(m => `<span>${escape(m)}</span>`).join('')}</div>${chips.length ? `<div class="p-chips">${chips.join('')}</div>` : ''}</div>${scopePicker(d)}</div>`;
  }
  function profileLead(d) {
    const h = d.headline;
    if (h?.text && h.display_value) {
      let text = escape(h.text).replace(escape(h.display_value), `<mark>${escape(h.display_value)}</mark>`);
      if (!/[.!?]$/.test(h.text)) text += '.';
      return `<p class="lead">${text}</p>`;
    }
    const t = d.totals;
    if (state.area === 'players') return `<p class="lead"><mark>${plural(t.goals, 'goal')}</mark> and ${plural(t.shots, 'shot')} in ${plural(t.appearances, 'appearance')} for ${escape(d.subject.club?.name || 'their club')} in ${escape(d.competition?.name || 'this competition')}, ${escape(d.season.label)}.</p>`;
    if (state.area === 'teams') return `<p class="lead">${plural(t.wins, 'win')}, ${plural(t.draws, 'draw')} and ${plural(t.losses, 'loss', 'losses')} from ${plural(t.matches, 'match', 'matches')}, scoring <mark>${whole(t.goals_for)}</mark> and conceding ${whole(t.goals_against)}.</p>`;
    const cards = d.metrics.cards?.display_value;
    return cards ? `<p class="lead"><mark>${escape(cards)}</mark> cards per match across ${plural(t.matches, 'match', 'matches')} in ${escape(d.season.label)}.</p>` : '';
  }
  function seasonLine(d) {
    const t = d.totals;
    const items = state.area === 'players'
      ? [[t.appearances, 'Appearances'], [t.minutes, 'Minutes'], [t.goals, 'Goals'], [t.assists, 'Assists'], [t.shots, 'Shots'], [t.yellow_cards, 'Yellow cards']]
      : state.area === 'teams'
        ? [[t.matches, 'Matches'], [t.goals_for, 'Scored'], [t.goals_against, 'Conceded'], [t.clean_sheets, 'Clean sheets'], [t.corners_for, 'Corners won'], [t.cards, 'Cards']]
        : [[t.matches, 'Matches'], [t.cards, 'Cards'], [t.yellows, 'Yellows'], [t.reds, 'Reds'], [t.fouls, 'Fouls']];
    return `<div class="season-line">${items.map(([v, label]) => `<div><b>${whole(v)}</b><span>${label}</span></div>`).join('')}</div>`;
  }
  function rankNotice(d) {
    const r = d.ranking;
    if (r.ranked) return '';
    const prev = d.previous_season?.ranking?.ranked ? ` The ranks below are from ${escape(d.previous_season.season.label)} and are labelled.` : '';
    let title, text;
    if (r.reason === 'below_minutes_threshold') {
      title = 'Not ranked yet';
      text = `Ranks start once a player reaches ${whole(r.minimum_minutes)} minutes in one competition. ${escape(d.subject.name)} has ${whole(r.sample_minutes)} so far, so per-90 rates would move a lot with one match.${prev}`;
    } else if (r.reason === 'below_matches_threshold') {
      title = 'Not ranked yet';
      text = `Ranks start after ${r.minimum_matches} matches with full statistics. This ${state.area === 'teams' ? 'team' : 'referee'} has ${r.sample_matches} so far, and averages from a few matches move a lot with one result.${prev}`;
    } else if (r.reason === 'unknown_position') {
      title = 'No peer group';
      text = 'The data does not record a position for this player, so there is no fair group to rank them against.';
    } else {
      title = 'Ranks are being rebuilt';
      text = `Rankings for this season will return after the next data refresh.${prev}`;
    }
    return `<div class="notice"><b>${title}</b><span>${text}</span></div>`;
  }
  function rankSource(d) {
    if (d.ranking.ranked) return {metrics: d.metrics, ranking: d.ranking, label: null, scope: scopeOf(d)};
    const p = d.previous_season;
    if (p?.ranking?.ranked) return {metrics: p.metrics, ranking: p.ranking, label: p.season.label,
      scope: isReferee() ? String(p.season.year) : `${p.competition.code}-${p.season.year}`};
    return null;
  }
  function rankCard(d) {
    const src = rankSource(d);
    if (!src) return '';
    const kind = state.area, defs = rankMetrics[kind], groups = rankGroups[kind];
    const order = groups.order[src.ranking.peer_group] || groups.order.all || groups.order.F;
    const any = Object.values(src.metrics).find(m => m.peer_count);
    if (!any) return '';
    const peers = any.peer_count;
    const title = kind === 'players' ? `Against ${any.peer_label}` : kind === 'teams' ? 'Against the league' : 'Against other referees';
    const small = [src.label, kind === 'players' ? `${peers} with ${whole(src.ranking.minimum_minutes)}+ minutes` : kind === 'teams' ? `${peers} teams · per match` : `${peers} referees with ${src.ranking.minimum_matches}+ matches`].filter(Boolean).join(' · ');
    const sections = order.map(name => {
      const [label, keys] = groups[name];
      const rows = keys.map(key => rankRow(key, src.metrics[key], defs[key], src)).filter(Boolean).join('');
      return rows ? `${order.length > 1 ? `<div class="group-label">${label}</div>` : ''}<div class="ranks">${rows}</div>` : '';
    }).join('');
    if (!sections) return '';
    const note = kind === 'teams' ? 'Dots further right mean more. For goals and chances conceded, further left is better.'
      : `The dot shows where ${kind === 'players' ? 'they sit' : 'this referee sits'}; the tick in the middle is the typical ${kind === 'players' ? (any.peer_label.split(' ').at(-1) || 'player').replace(/s$/, '') : 'referee'}.`;
    return `<section class="card" aria-labelledby="rankTitle"><div class="card-h"><h3 id="rankTitle">${escape(title)}</h3><small>${escape(small)}</small></div>
      <div class="scale-key" aria-hidden="true"><span></span><span></span><span><span>Fewer</span><span>More</span></span><span></span></div>${sections}
      <p class="card-note">${note} Select a row to see it match by match.</p></section>`;
  }
  function rankRow(key, m, def, src) {
    if (!m || !def || m.percentile == null || m.display_value == null) return '';
    const [label, stat] = def;
    const text = cap(state.area === 'teams' ? m.rank_phrase : m.comparison_text) || '';
    const average = state.area === 'teams' && Number.isFinite(m.league_average) ? ` League average ${m.unit === 'percent' ? Math.round(m.league_average) + '%' : fixed(m.league_average)}.` : '';
    const tipText = `${label}: ${m.display_value}. ${text}.${average}`;
    return `<button class="rank" type="button" data-stat="${stat}" data-scope="${escape(src.scope)}" data-tip="${escape(tipText)}" aria-label="${escape(tipText)} Opens the match-by-match chart.">
      <span class="rank-label">${escape(label)}</span><span class="rank-value">${escape(m.display_value)}</span>
      <span class="track" aria-hidden="true"><span class="fill" style="width:${m.percentile}%"></span><span class="mid"></span><span class="dot" style="left:${m.percentile}%"></span></span>
      <span class="rank-go" aria-hidden="true">›</span><span class="rank-band">${escape(text)}</span></button>`;
  }
  function howOftenCard(d) {
    const rows = Object.entries(d.frequencies || {}).filter(([, v]) => v?.denominator).map(([key, v]) => ({label: frequencyLabels[key] || key, n: v.count, of: v.denominator, pct: Math.round(v.count / v.denominator * 100)}));
    if (!rows.length) return '';
    const of = Math.max(...rows.map(r => r.of));
    const note = state.area === 'players' ? 'Counts from past matches. They are not a forecast or a probability for the next game.'
      : state.area === 'teams' ? 'Match totals include both teams. Past matches, not a forecast.' : 'Past appointments, not a forecast. Cards depend on the teams as well as the referee.';
    return `<section class="card"><div class="card-h"><h3>How often</h3><small>In ${isPlayer() ? plural(of, 'appearance') : plural(of, 'match', 'matches')}</small></div>
      <div class="meters">${rows.map(r => `<div class="meter"><span class="meter-label">${escape(r.label)}</span><span class="meter-count"><b>${r.n}</b> of ${r.of} · ${r.pct}%</span><span class="meter-bar" role="img" aria-label="${escape(r.label)}: ${r.n} of ${r.of}"><i style="width:${r.pct}%"></i></span></div>`).join('')}</div>
      <p class="card-note">${note}</p></section>`;
  }
  const niceMax = (value, minimum = 1) => {
    const v = Math.max(minimum, value || 0), step = v <= 4 ? 1 : v <= 8 ? 2 : v <= 20 ? 5 : v <= 50 ? 10 : 30;
    const max = Math.ceil(v / step) * step;
    return {max, ticks: Array.from({length: max / step + 1}, (_, i) => i * step)};
  };
  function columnChart({items, height = 170, minimum = 1, segs, dots, label, tipFor, xLabel, legend, table, maxValue}) {
    const n = items.length;
    const {max, ticks} = niceMax(maxValue ?? Math.max(0, ...items.map(it => segs.reduce((sum, s) => sum + (it[s.key] || 0), 0))), minimum);
    const grid = ticks.slice(1).map(t => `<span class="gridline" style="bottom:${t / max * 100}%"></span>`).join('');
    const cols = items.map(it => {
      const missing = segs.every(s => it[s.key] == null);
      const total = segs.reduce((sum, seg) => sum + (it[seg.key] || 0), 0);
      const stack = missing ? '<span class="col-missing">–</span>' : total === 0 ? '<span class="col-zero"></span>' : segs.map(s => it[s.key] > 0 ? `<i style="height:${it[s.key] / max * height}px;background:var(${s.color})"></i>` : '').join('');
      const goalDots = dots && !missing ? `<span class="goals">${Array.from({length: Math.min(4, dots(it) || 0)}, () => '<i></i>').join('')}</span>` : '';
      const text = tipFor(it);
      return `<div class="ccol"><div class="colwrap" tabindex="0" data-tip="${escape(text)}" aria-label="${escape(text)}">${goalDots}<span class="stack-col">${stack}</span></div></div>`;
    }).join('');
    const gaps = items.some(it => segs.every(seg => it[seg.key] == null)) ? '<span>– No statistics recorded</span>' : '';
    if (gaps) legend = (legend || '') + gaps;
    return `<div class="chart">${legend ? `<div class="legend">${legend}</div>` : ''}
      <div class="plot" role="img" aria-label="${escape(label)}"><div class="yaxis" style="height:${height}px">${ticks.map(t => `<span style="bottom:${t / max * 100}%">${t}</span>`).join('')}</div>
      <div class="bars" style="height:${height}px;grid-template-columns:repeat(${n},minmax(0,1fr))">${grid}${cols}</div></div>
      <div class="xlabels" style="grid-template-columns:repeat(${n},minmax(0,1fr))">${items.map(xLabel).join('')}</div>
      <details class="tbl"><summary>Show as table</summary><div class="tbl-wrap">${table}</div></details></div>`;
  }
  const lastTen = d => (d.match_log || d.last_10 || []).slice(-10);
  const venueWord = m => m.venue === 'home' ? 'v' : 'at';
  const opp = m => m.opponent?.name || 'Opponent';
  function lastTenCard(d) {
    const last = lastTen(d);
    if (!last.length) return '';
    const kind = state.area;
    let chart, title, small;
    if (kind === 'players') {
      const items = last.map(m => ({...m, on: m.values.shots_on_target, off: m.values.shots_off_target ?? (m.values.shots != null && m.values.shots_on_target != null ? m.values.shots - m.values.shots_on_target : null)}));
      title = `Last ${plural(items.length, 'appearance')}`; small = 'Shots, on target and goals';
      chart = columnChart({items, minimum: 4, segs: [{key: 'on', color: '--chart-1'}, {key: 'off', color: '--chart-muted'}], dots: m => m.values.goals,
        label: `Shots in each of the last ${items.length} appearances, split into on target and off target, with goals marked`,
        tipFor: m => `${day(m.date)} ${venueWord(m)} ${opp(m)}: ${m.values.shots ?? '?'} shots, ${m.values.shots_on_target ?? '?'} on target, ${plural(m.values.goals ?? 0, 'goal')}, ${m.values.minutes} min`,
        xLabel: m => `<span><b>${escape(m.opponent?.abbreviation || short(m.opponent))}</b>${m.venue === 'home' ? 'H' : 'A'}</span>`,
        legend: '<span><i style="background:var(--chart-1)"></i>On target</span><span><i style="background:var(--chart-muted)"></i>Off target or blocked</span><span><i class="dot" style="background:var(--text)"></i>Goal</span>',
        table: `<table><thead><tr><th>Date</th><th>Opponent</th><th>Min</th><th>Shots</th><th>On target</th><th>Goals</th></tr></thead><tbody>${last.map(m => `<tr><td>${day(m.date)}</td><td>${escape(opp(m))} (${m.venue === 'home' ? 'H' : 'A'})</td><td>${m.values.minutes ?? '–'}</td><td>${m.values.shots ?? '–'}</td><td>${m.values.shots_on_target ?? '–'}</td><td>${m.values.goals ?? '–'}</td></tr>`).join('')}</tbody></table>`});
    } else if (kind === 'teams') {
      const hasXg = last.some(m => m.values.xg_for != null);
      const [forKey, againstKey, what] = hasXg ? ['xg_for', 'xg_against', 'Expected goals'] : ['goals_for', 'goals_against', 'Goals'];
      title = hasXg ? 'Chances by match' : 'Goals by match'; small = `${what}, last ${last.length}`;
      const {max, ticks} = niceMax(Math.max(...last.flatMap(m => [m.values[forKey] || 0, m.values[againstKey] || 0])), 2);
      const h = 170, name = d.subject.name;
      const score = m => `${m.values.goals_for ?? '?'}–${m.values.goals_against ?? '?'}`;
      const cols = last.map(m => {
        const f = m.values[forKey], a = m.values[againstKey];
        const text = `${day(m.date)} ${venueWord(m)} ${opp(m)}: ${score(m)}.${hasXg ? ` xG ${fixed(f)} for, ${fixed(a)} against` : ''}`;
        const bars = f == null && a == null ? '<span class="col-missing">–</span>' : `<i style="height:${(f || 0) / max * h}px;background:var(--chart-1)"></i><i style="height:${(a || 0) / max * h}px;background:var(--chart-2)"></i>`;
        return `<div class="ccol"><div class="colwrap" tabindex="0" data-tip="${escape(text)}" aria-label="${escape(text)}"><span class="pair">${bars}</span></div></div>`;
      }).join('');
      chart = `<div class="chart"><div class="legend"><span><i style="background:var(--chart-1)"></i>${escape(name)}</span><span><i style="background:var(--chart-2)"></i>Opponent</span>${last.some(m => m.values[forKey] == null && m.values[againstKey] == null) ? '<span>– No statistics recorded</span>' : ''}</div>
        <div class="plot" role="img" aria-label="${what} for and against in each of the last ${last.length} matches"><div class="yaxis" style="height:${h}px">${ticks.map(t => `<span style="bottom:${t / max * 100}%">${t}</span>`).join('')}</div>
        <div class="bars" style="height:${h}px;grid-template-columns:repeat(${last.length},minmax(0,1fr))">${ticks.slice(1).map(t => `<span class="gridline" style="bottom:${t / max * 100}%"></span>`).join('')}${cols}</div></div>
        <div class="xlabels" style="grid-template-columns:repeat(${last.length},minmax(0,1fr))">${last.map(m => `<span><b>${escape(m.opponent?.abbreviation || short(m.opponent))}</b>${score(m)}</span>`).join('')}</div>
        <details class="tbl"><summary>Show as table</summary><div class="tbl-wrap"><table><thead><tr><th>Date</th><th>Opponent</th><th>Score</th><th>xG for</th><th>xG against</th></tr></thead><tbody>${last.map(m => `<tr><td>${day(m.date)}</td><td>${escape(opp(m))} (${m.venue === 'home' ? 'H' : 'A'})</td><td>${score(m)}</td><td>${fixed(m.values.xg_for)}</td><td>${fixed(m.values.xg_against)}</td></tr>`).join('')}</tbody></table></div></details></div>`;
      if (hasXg) chart += `<p class="card-note">Expected goals (xG) estimates the quality of chances. A taller green bar than blue means ${escape(name)} made the better chances.</p>`;
    } else {
      title = `Last ${plural(last.length, 'match', 'matches')}`; small = 'Cards shown, both teams';
      const items = last.map(m => ({...m, cards: m.values.cards}));
      chart = columnChart({items, minimum: 6, segs: [{key: 'cards', color: '--chart-1'}],
        label: `Cards shown in each of the last ${items.length} matches`,
        tipFor: m => `${day(m.date)} ${m.home.name} v ${m.away.name} (${m.competition}): ${m.values.cards ?? '?'} cards, ${m.values.fouls ?? '?'} fouls`,
        xLabel: m => `<span><b>${escape(m.competition)}</b>${day(m.date)}</span>`, legend: '',
        table: `<table><thead><tr><th>Date</th><th>Match</th><th>Competition</th><th>Cards</th><th>Fouls</th></tr></thead><tbody>${last.map(m => `<tr><td>${day(m.date)}</td><td>${escape(m.home.name)} v ${escape(m.away.name)}</td><td>${escape(m.competition)}</td><td>${m.values.cards ?? '–'}</td><td>${m.values.fouls ?? '–'}</td></tr>`).join('')}</tbody></table>`});
    }
    return `<section class="card"><div class="card-h"><h3>${title}</h3><small>${small}</small></div>${chart}</section>`;
  }
  function minutesCard(d) {
    const last = lastTen(d).filter(m => m.values?.minutes != null);
    if (!last.length) return '';
    const top = Math.max(90, ...last.map(m => m.values.minutes)), h = 56;
    const role = d.role_summary;
    const strip = `<div class="chart"><div class="plot" role="img" aria-label="Minutes played in each of the last ${last.length} appearances">
      <div class="yaxis" style="height:${h}px"><span style="bottom:0%">0</span><span style="bottom:${90 / top * 100}%">90</span></div>
      <div class="bars" style="height:${h}px;grid-template-columns:repeat(${last.length},minmax(0,1fr))"><span class="gridline" style="bottom:${90 / top * 100}%"></span>${last.map(m => {
        const text = `${day(m.date)} ${venueWord(m)} ${opp(m)}: ${m.values.minutes} minutes${m.started === false ? ', came off the bench' : ''}`;
        return `<div class="ccol"><div class="colwrap" tabindex="0" data-tip="${escape(text)}" aria-label="${escape(text)}"><span class="stack-col"><i style="height:${m.values.minutes / top * h}px;background:var(${m.started === false ? '--chart-muted' : '--chart-1'})"></i></span></div></div>`;
      }).join('')}</div></div>
      <div class="legend"><span><i style="background:var(--chart-1)"></i>Started</span><span><i style="background:var(--chart-muted)"></i>Off the bench</span></div></div>`;
    return `<section class="card"><div class="card-h"><h3>Minutes</h3><small>Last ${plural(last.length, 'appearance')}</small></div>${strip}<p class="card-note">${role ? `Started ${role.starts} of ${plural(role.appearances, 'appearance')} this season. ` : ''}Per-90 rates only count minutes on the pitch.</p></section>`;
  }
  function funnelCard(d) {
    const f = d.shooting_funnel;
    if (!f?.shots) return '';
    const onT = Math.round(f.shots_on_target / f.shots * 100), conv = Math.round(f.goals / f.shots * 100);
    return `<section class="card"><div class="card-h"><h3>Shooting</h3><small>${escape(d.season.label)} totals</small></div>
      <div class="funnel" role="img" aria-label="Shooting funnel: ${f.shots} shots, ${f.shots_on_target} on target, ${f.goals} goals">
      <div><span>Shots</span><i style="width:100%"></i><b>${f.shots}</b></div><div><span>On target</span><i style="width:${f.shots_on_target / f.shots * 100}%"></i><b>${f.shots_on_target}</b></div><div><span>Goals</span><i style="width:${f.goals / f.shots * 100}%"></i><b>${f.goals}</b></div></div>
      <p class="card-note">${onT}% of shots hit the target and ${conv}% were scored.</p></section>`;
  }
  function homeAwayCard(d) {
    const ha = d.home_away;
    if (!ha?.home?.matches || !ha?.away?.matches) return '';
    const rows = [['Goals scored', 'goals_for'], ['Goals conceded', 'goals_against'], ['Corners won', 'corners_for'], ['Corners conceded', 'corners_against'], ['Cards', 'cards']]
      .map(([label, key]) => [label, ha.home.metrics[key]?.value, ha.away.metrics[key]?.value]).filter(([, h, a]) => Number.isFinite(h) && Number.isFinite(a));
    if (!rows.length) return '';
    const {max} = niceMax(Math.max(...rows.flatMap(([, h, a]) => [h, a])), 2);
    const row = ([label, h, a]) => {
      const lo = Math.min(h, a), hi = Math.max(h, a), text = `${label}: home ${fixed(h)}, away ${fixed(a)} per match`;
      return `<div class="dumb-row"><span>${label}</span><span class="dumb-track" role="img" tabindex="0" aria-label="${escape(text)}" data-tip="${escape(text)}"><span class="link" style="left:${lo / max * 100}%;width:${(hi - lo) / max * 100}%"></span><span class="d" style="left:${h / max * 100}%;background:var(--chart-1)"></span><span class="d" style="left:${a / max * 100}%;background:var(--chart-2)"></span></span></div>`;
    };
    return `<section class="card"><div class="card-h"><h3>Home and away</h3><small>Per match</small></div>
      <div class="legend"><span><i class="dot" style="background:var(--chart-1)"></i>Home (${ha.home.matches})</span><span><i class="dot" style="background:var(--chart-2)"></i>Away (${ha.away.matches})</span></div>
      <div class="dumb">${rows.map(row).join('')}</div><p class="card-note">Scale 0 to ${max}. Hover or focus a row for exact values.</p></section>`;
  }
  function leadersCard(d) {
    const L = d.squad_leaders;
    if (!L) return '';
    const scope = scopeOf(d);
    const rows = [['Top scorer', 'goals', 'goal'], ['Most assists', 'assists', 'assist'], ['Most on target', 'shots_on_target', 'on target', 'on target'], ['Most bookings', 'yellow_cards', 'yellow card']]
      .map(([label, key, one, many]) => { const p = L[key]?.[0]; return p && p.value ? `<div><span>${label}</span><button type="button" class="leader-link" data-player="${p.player_id}" data-scope="${escape(scope)}">${escape(p.name)}</button><span class="num">${plural(p.value, one, many)}</span></div>` : ''; }).join('');
    return rows ? `<section class="card"><div class="card-h"><h3>Squad leaders</h3><small>${escape(d.competition?.name || '')}</small></div><div class="leaders">${rows}</div></section>` : '';
  }
  function competitionCard(d) {
    const list = d.competition_breakdown || [];
    if (!list.length) return '';
    return `<section class="card"><div class="card-h"><h3>By competition</h3><small>${escape(d.season.label)}</small></div><table class="mini-table"><thead><tr><th>Competition</th><th>Matches</th><th>Cards per match</th></tr></thead><tbody>${list.map(c => `<tr><td>${escape(c.competition)}</td><td>${c.matches}</td><td>${fixed(c.metrics.cards?.value)}${c.small_sample ? ' <span class="chip warn">Small sample</span>' : ''}</td></tr>`).join('')}</tbody></table><p class="card-note">Averages from fewer than five matches are marked. One heated game can swing them.</p></section>`;
  }
  function advancedBlock(d) {
    const m = d.metrics, t = d.totals, cell = (label, value, note) => value == null ? '' : `<div><span>${label}</span><b>${escape(value)}</b>${note ? `<p>${escape(note)}</p>` : ''}</div>`;
    const compare = metric => metric?.comparison_text ? cap(metric.comparison_text) : '';
    let cells = '';
    if (state.area === 'players') {
      cells = cell('Shooting accuracy', m.shooting_accuracy_pct?.display_value, compare(m.shooting_accuracy_pct) || 'Share of shots on target')
        + cell('Shot conversion', m.conversion_pct?.display_value, compare(m.conversion_pct) || 'Share of shots scored')
        + cell('Duels won', m.duel_win_pct?.display_value, compare(m.duel_win_pct) || 'Share of one-on-one contests won')
        + cell('Pass accuracy', m.pass_accuracy_pct?.display_value, compare(m.pass_accuracy_pct) || 'Share of passes completed')
        + cell('Fouls won vs committed', `${whole(t.fouls_won)} / ${whole(t.fouls_committed)}`, 'Season totals')
        + cell('Average match rating', m.average_rating?.display_value, compare(m.average_rating) || 'Data provider rating, 0 to 10');
    } else if (state.area === 'teams') {
      const diff = (a, b) => Number.isFinite(a) && Number.isFinite(b) ? (a - b >= 0 ? '+' : '−') + Math.abs(a - b).toFixed(1) : null;
      cells = cell('Goals minus xG', diff(t.goals_for, t.xg_for), 'Positive means more goals than the chances suggested')
        + cell('Conceded minus xG against', diff(t.goals_against, t.xg_against), 'Negative means fewer goals allowed than the chances suggested')
        + cell('Possession', m.possession?.display_value, cap(m.possession?.rank_phrase) || 'Average share of the ball')
        + cell('Shots on target faced', m.shots_on_target_faced?.display_value, cap(m.shots_on_target_faced?.rank_phrase) || 'Per match')
        + cell('Cards per match', m.cards?.display_value, cap(m.cards?.rank_phrase) || 'Yellow and red');
    } else {
      cells = cell('Cards per foul', m.cards_per_foul?.display_value, compare(m.cards_per_foul) || 'How readily fouls become cards')
        + cell('Yellows and reds', `${whole(t.yellows)} / ${whole(t.reds)}`, 'Season totals')
        + cell('Fouls per match', m.fouls?.display_value, compare(m.fouls));
    }
    const open = window.matchMedia('(min-width: 769px)').matches ? ' open' : '';
    return cells ? `<details class="adv"${open}><summary>Advanced <small>Rates, accuracy and context</small></summary><div class="adv-grid">${cells}</div></details>` : '';
  }
  function dataFoot(d) {
    const b = d.data_basis || {};
    const excluded = b.excluded_matches ? `, ${plural(b.excluded_matches, 'match', 'matches')} without full statistics left out` : '';
    const scope = isReferee() ? `all competitions in our data, ${escape(d.season.label)}` : `${escape(d.competition?.name)}, ${escape(d.season.label)}`;
    const phase = isReferee() ? '' : ' Domestic leagues count regular-season matches; European cups count the league or group stage and knockouts.';
    return `<p class="disclaimer">Statistics describe past matches. They are not predictions or recommendations. Data behind this view: ${plural(b.matches ?? 0, 'match', 'matches')}, ${scope}${excluded}.${phase}</p>`;
  }

  // ----- Match-by-match chart -----
  function buildRecords(profile) {
    return (profile.match_log || []).map(m => ({
      id: String(m.fixture_id), date: m.date, venue: m.venue || null, competition: m.competition || null,
      opponent: isReferee() ? `${m.home.name} v ${m.away.name}` : opp(m),
      abbr: isReferee() ? `${short(m.home)}–${short(m.away)}` : (m.opponent?.abbreviation || short(m.opponent)),
      minutes: m.values.minutes ?? null, started: m.started, appeared: true, score: m.score, stats: m.values
    }));
  }
  const value = record => {
    if (!record.appeared || record.stats[state.stat] == null) return null;
    if (state.rate === 'per90') return record.minutes > 0 ? record.stats[state.stat] * 90 / record.minutes : null;
    return record.stats[state.stat];
  };
  const scopedRecords = () => {
    const records = state.records.filter(r => state.venue === 'all' || r.venue === state.venue);
    return state.period === 'season' ? records : records.slice(-Number(state.period));
  };
  const average = records => {
    const known = records.filter(r => value(r) != null);
    if (!known.length) return null;
    const total = known.reduce((n, r) => n + r.stats[state.stat], 0);
    return state.rate === 'per90' ? total * 90 / known.reduce((n, r) => n + r.minutes, 0) : total / known.length;
  };
  const nounFor = (metric, v) => v !== 1 ? metric.noun
    : metric.noun.replace(/\b(pass)es\b|\b(\w+?)s\b/, (m, pass, word) => pass || word);
  const metricValue = v => fmt(v, metrics[state.stat].decimals || 1) + (Number.isFinite(v) && metrics[state.stat].percent ? '%' : '');
  const resultWord = r => {
    if (isReferee()) return r.score?.home != null ? `Final score ${r.score.home}–${r.score.away}` : 'Score unavailable';
    const f = r.stats.goals_for ?? (r.venue === 'home' ? r.score?.home : r.score?.away), a = r.stats.goals_against ?? (r.venue === 'home' ? r.score?.away : r.score?.home);
    return f == null || a == null ? 'Score unavailable' : `${f > a ? 'Won' : f < a ? 'Lost' : 'Drew'} ${f}–${a}`;
  };
  function openChart(stat) {
    const entering = state.view !== 'chart';
    if (entering) Object.assign(state, {period: '10', venue: 'all', rate: 'match', thresholdOn: false, compare: false, selected: null});
    state.stat = stat; state.threshold = metrics[stat].threshold || 1;
    if (metrics[stat].noRate) state.rate = 'match';
    view('chart');
    const d = state.profile, s = d.subject;
    $('playerName').textContent = s.name;
    $('playerMeta').textContent = (isPlayer() ? [s.club?.name, s.position?.label] : state.area === 'teams' ? [s.country] : ['Referee', s.country]).filter(Boolean).join(' · ');
    $('playerAvatar').innerHTML = badgeInner(isPlayer() ? s.photo_url : s.logo_url, s.name);
    $('playerAvatar').classList.toggle('avatar-crest', state.area === 'teams');
    wireImages($('playerAvatar'));
    $('nextFixtureLabel').textContent = 'Showing';
    $('nextFixture').textContent = isReferee() ? `All competitions ${d.season.label}` : `${d.competition.name} ${d.season.label}`;
    $('fixtureHint').textContent = plural(state.records.length, isPlayer() ? 'appearance' : 'match', isPlayer() ? 'appearances' : 'matches');
    $('backSearchLabel').textContent = 'Back to profile';
    $('resetFilters').textContent = `View ${metrics[current().defaultStat].title.toLowerCase()}`;
    if (entering) {
      $('filterPanel').hidden = true; $('filterButton').setAttribute('aria-expanded', 'false');
      $('extraStats').hidden = true; $('moreStats').setAttribute('aria-expanded', 'false');
      $('matchDetails').open = false;
    }
    render();
    if (entering) $('playerName').focus({preventScroll: true});
  }
  function tabs() {
    const keys = common.includes(state.stat) ? common : [...common, state.stat];
    $('statTabs').innerHTML = keys.map(key => `<button id="tab-${key}" role="tab" aria-selected="${key === state.stat}" aria-controls="statPanel" tabindex="${key === state.stat ? 0 : -1}" data-stat="${key}">${metrics[key].label}</button>`).join('');
    $('statPanel').setAttribute('aria-labelledby', `tab-${state.stat}`);
  }
  function selectStat(key, focus = true) {
    $('extraStats').hidden = true; $('moreStats').setAttribute('aria-expanded', 'false');
    go(state.area, state.id, state.scope, key);
    if (focus) $(`tab-${key}`)?.focus({preventScroll: true});
    $(`tab-${key}`)?.scrollIntoView({block: 'nearest', inline: 'nearest', behavior: 'instant'});
  }
  function togglePanel(button, panel) {
    $(panel).hidden = !$(panel).hidden;
    $(button).setAttribute('aria-expanded', String(!$(panel).hidden));
  }
  function renderSelected(record) {
    state.selected = record.id;
    root.querySelectorAll('.match-column').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.record === record.id)));
    const v = value(record);
    const label = v == null ? 'Unavailable' : `${metricValue(v)} ${nounFor(metrics[state.stat], v)}${state.rate === 'per90' ? ' / 90' : ''}`;
    const where = isReferee() ? record.competition : record.venue === 'home' ? 'Home' : 'Away';
    const side = isPlayer() ? (record.minutes == null ? 'Minutes unavailable' : `${record.minutes} minutes${record.started === false ? ' off the bench' : ' played'}`) : resultWord(record);
    $('selectedMatch').innerHTML = `<div><strong>${isReferee() ? '' : venueWord(record) + ' '}${escape(record.opponent)}</strong><small>${dayYear(record.date)} · ${escape(where)}${isPlayer() ? ' · ' + escape(resultWord(record)) : ''}</small></div><div class="selected-value">${label}<span>${escape(side)}</span></div>`;
  }
  function render() {
    if (!state.profile || state.view !== 'chart') return;
    const metric = metrics[state.stat], records = scopedRecords();
    const eligible = records.filter(r => value(r) != null);
    const missing = records.length - eligible.length;
    const recent = average(records), season = average(state.records);
    const unit = isPlayer() ? 'appearances' : 'matches';
    const periodLabel = state.period === 'season' ? `Whole ${state.profile.season.label} season` : `Last ${state.period} ${unit}`;
    tabs();
    for (const [group, key] of [['periodControl', 'period'], ['venueControl', 'venue'], ['rateControl', 'rate']]) {
      $(group).querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset[key] === state[key])));
    }
    $('rateField').hidden = !isPlayer() || !!metric.noRate;
    const filtered = Number(state.venue !== 'all') + Number(state.rate !== 'match');
    $('filterCount').hidden = !filtered; $('filterCount').textContent = String(filtered);
    $('rateHelp').textContent = state.rate === 'per90' ? 'Per 90 adjusts for time played. Short appearances can produce large values; averages use total recorded minutes.' : 'Match counts show what actually happened. Missing values are left out, never filled with zero.';
    $('thresholdToggle').disabled = state.rate === 'per90';
    $('thresholdToggle').title = state.rate === 'per90' ? 'Switch to match counts to explore a historical threshold.' : '';
    $('thresholdToggle').setAttribute('aria-pressed', String(state.thresholdOn && state.rate === 'match'));
    $('thresholdPanel').hidden = !state.thresholdOn || state.rate === 'per90';
    $('thresholdValue').textContent = `${state.threshold}${metric.percent ? '%' : ''}+`;
    $('thresholdDown').disabled = state.threshold <= (metric.step || 1);
    $('thresholdUp').disabled = state.threshold >= (metric.limit || 10);
    $('compareSeason').checked = state.compare;
    $('statTitle').textContent = metric.title;
    $('statDefinition').textContent = metric.definition;
    let insight;
    if (!eligible.length) insight = 'There isn’t enough recorded data for this view.';
    else if (state.rate === 'per90') insight = `Averaged <strong>${fmt(recent)} ${metric.noun}</strong> per 90 minutes.`;
    else if (state.thresholdOn) {
      const count = eligible.filter(r => value(r) >= state.threshold).length;
      insight = `Recorded ${state.threshold}${metric.percent ? '%' : ''}+ ${metric.noun} in <strong>${count} of ${eligible.length}</strong> ${unit} with data.`;
    } else if (!isPlayer() || metric.decimals || metric.noRate) insight = `Averaged <strong>${metricValue(recent)} ${metric.noun}</strong> across ${eligible.length} ${unit} with data.`;
    else {
      const count = eligible.filter(r => value(r) >= 1).length;
      insight = `${metric.verb || 'Recorded ' + metric.noun} in <strong>${count} of ${eligible.length}</strong> ${unit} with data.`;
    }
    $('mainInsight').innerHTML = insight;
    $('sampleContext').textContent = `${periodLabel}${state.venue !== 'all' ? ` · ${state.venue} only` : ''} · ${records.length ? `${dayYear(records[0].date)} – ${dayYear(records.at(-1).date)}` : 'No matches'}${missing ? ` · ${missing} without this statistic` : ''}`;
    $('chartMeasure').textContent = `${metric.title}${metric.percent ? ' (%)' : state.rate === 'per90' ? ' per 90' : ' per match'}`;
    $('chartEmpty').hidden = eligible.length > 0;
    $('chartBlock').hidden = !eligible.length;
    const dataMax = Math.max(1, ...eligible.map(value), state.compare && season != null ? season : 0, state.thresholdOn && state.rate === 'match' ? state.threshold : 0);
    const step = dataMax <= 4 ? 1 : dataMax <= 12 ? 2 : dataMax <= 30 ? 5 : dataMax <= 60 ? 10 : 30;
    const max = metric.percent ? 100 : Math.ceil((dataMax + step * .5) / step) * step;
    const ticks = Array.from({length: Math.floor(max / step) + 1}, (_, i) => ({top: 100 - i * step / max * 100, label: fmt(i * step)}));
    $('plotGrid').innerHTML = ticks.map(t => `<div class="grid-line" style="top:${t.top}%"></div>`).join('');
    $('chartAxis').innerHTML = ticks.map(t => `<span style="top:${t.top}%">${t.label}</span>`).join('');
    const refs = [];
    if (state.compare && season != null) refs.push(`<div class="reference-line average" style="top:${100 - season / max * 100}%"><span class="reference-label">Season ${metricValue(season)}</span></div>`);
    if (state.thresholdOn && state.rate === 'match') refs.push(`<div class="reference-line" style="top:${100 - state.threshold / max * 100}%"><span class="reference-label">${state.threshold}${metric.percent ? '%' : ''}+</span></div>`);
    $('referenceLines').innerHTML = refs.join('');
    const barWidth = isReferee() ? 76 : 44;
    $('chartColumns').style.setProperty('--bar-width', `${barWidth}px`);
    $('chartStage').style.minWidth = `${records.length * barWidth + Math.max(0, records.length - 1) * 8}px`;
    $('chartColumns').style.setProperty('--count', Math.max(records.length, 1));
    $('chartColumns').innerHTML = records.map(r => {
      const v = value(r), height = v == null ? 0 : v / max * 100;
      const title = `${dayYear(r.date)}, ${isReferee() ? '' : r.venue === 'home' ? 'home to ' : 'away at '}${r.opponent}: ${v == null ? 'statistic unavailable' : `${metricValue(v)} ${nounFor(metric, v)}${state.rate === 'per90' ? ' per 90 minutes' : ''}`}; ${isPlayer() ? (r.minutes == null ? 'minutes unavailable' : `${r.minutes} minutes played`) : resultWord(r)}`;
      const bar = v == null ? '<span class="missing-value">—</span>' : v === 0 ? '<span class="zero-value"></span><span class="bar-value">0</span>' : `<span class="stat-bar"></span><span class="bar-value">${metricValue(v)}</span>`;
      return `<button class="match-column" data-record="${r.id}" aria-pressed="false" aria-label="${escape(title)}" title="${escape(title)}" style="--bar-height:${height}%"><span class="bar-area" aria-hidden="true">${bar}</span><span class="opponent-label" aria-hidden="true">${escape(r.abbr)}</span><span class="venue-label" aria-hidden="true">${isReferee() ? day(r.date) : r.venue === 'home' ? 'H' : 'A'}</span></button>`;
    }).join('');
    if (records.length) renderSelected(records.find(r => r.id === state.selected) || records.at(-1));
    if (isPlayer()) {
      const known = records.filter(r => r.minutes != null), full = known.filter(r => r.minutes >= 75).length;
      $('minutesInsight').textContent = known.length ? `Played 75+ minutes in ${full} of ${plural(known.length, 'appearance')} in this view.` : 'No recorded minutes in this view.';
      $('contextFootnote').textContent = 'Short appearances mean fewer opportunities.';
    } else if (state.area === 'teams') {
      const scored = records.filter(r => r.stats.goals_for != null && r.stats.goals_against != null);
      const wins = scored.filter(r => r.stats.goals_for > r.stats.goals_against).length, draws = scored.filter(r => r.stats.goals_for === r.stats.goals_against).length;
      $('minutesInsight').textContent = `${plural(wins, 'win')}, ${plural(draws, 'draw')} and ${plural(scored.length - wins - draws, 'loss', 'losses')} across ${plural(scored.length, 'match', 'matches')} in this view.`;
      $('contextFootnote').textContent = 'Results give context. They do not establish the value of a bet.';
    } else {
      $('minutesInsight').textContent = `${plural(records.length, 'match', 'matches')} in this view. These counts describe the matches this referee took charge of.`;
      $('contextFootnote').textContent = 'Team style, competition and match state matter too. Raw home and away counts do not show bias.';
    }
    $('recentAverageLabel').textContent = state.period === 'season' ? (state.venue === 'all' ? 'This view' : `${state.venue === 'home' ? 'Home' : 'Away'} view`) : `Last ${records.length} ${unit}`;
    $('recentAverage').textContent = metricValue(recent);
    $('seasonAverageLabel').textContent = `Whole ${state.profile.season.label}`;
    $('seasonAverage').textContent = metricValue(season);
    $('comparisonInsight').textContent = state.rate === 'per90' ? 'Per 90, using total counts and minutes from appearances with data. The season figure includes both venues.' : `Average per ${isPlayer() ? 'appearance' : 'match'} with a recorded value. The season figure includes both venues.`;
    $('coverageText').textContent = `${eligible.length} of ${plural(records.length, isPlayer() ? 'appearance' : 'match', unit)} in this view have a recorded value for ${metric.noun}. Missing values are left out of averages, never counted as zero. Extra time is included when a match had it.`;
    $('tableVenue').textContent = isReferee() ? 'Competition' : 'Venue';
    $('tableContext').textContent = isPlayer() ? 'Minutes' : isReferee() ? 'Score' : 'Result';
    $('detailCount').textContent = plural(records.length, isPlayer() ? 'appearance' : 'match', unit);
    $('tableCaption').textContent = `${state.profile.subject.name}: ${metric.title.toLowerCase()} by match`;
    $('tableStat').textContent = `${metric.label}${state.rate === 'per90' ? ' / 90' : ''}`;
    $('matchRows').innerHTML = records.toReversed().map(r => `<tr><td>${escape(r.opponent)}<small>${dayYear(r.date)}</small></td><td>${isReferee() ? escape(r.competition) : r.venue === 'home' ? 'Home' : 'Away'}</td><td>${metricValue(value(r))}</td><td>${isPlayer() ? (r.minutes == null ? 'Unknown' : r.minutes) : escape(resultWord(r))}</td></tr>`).join('');
    requestAnimationFrame(() => {
      const viewport = $('chartViewport');
      $('chartBlock').classList.toggle('chart-overflows', viewport.scrollWidth > viewport.clientWidth + 2);
      viewport.scrollLeft = viewport.scrollWidth;
    });
  }

  // ----- Events -----
  $('playerSearch').addEventListener('input', search);
  $('playerResults').addEventListener('click', event => {
    const item = event.target.closest('[data-id]');
    if (item) return go(state.area, item.dataset.id);
    const suggestion = event.target.closest('[data-suggest]');
    if (suggestion) { $('playerSearch').value = suggestion.dataset.suggest; search(); $('playerSearch').focus(); }
  });
  $('profileBack').addEventListener('click', () => go(state.area));
  $('backSearch').addEventListener('click', () => go(state.area, state.id, state.scope));
  $('profileBody').addEventListener('click', event => {
    const rank = event.target.closest('.rank[data-stat]');
    if (rank) return go(state.area, state.id, rank.dataset.scope, rank.dataset.stat);
    const leader = event.target.closest('[data-player]');
    if (leader) return go('players', leader.dataset.player, leader.dataset.scope);
    if (event.target.closest('[data-retry]')) fromRoute([state.area, state.id, state.scope].filter(Boolean).join('/'));
  });
  $('profileBody').addEventListener('change', event => {
    if (event.target.id === 'scopeSelect') go(state.area, state.id, event.target.value);
  });
  root.querySelectorAll('[data-area]').forEach(a => a.addEventListener('click', event => { event.preventDefault(); go(a.dataset.area); }));
  for (const id of ['statTabs', 'extraStats']) $(id).addEventListener('click', event => { const tab = event.target.closest('[data-stat]'); if (tab) selectStat(tab.dataset.stat); });
  $('statTabs').addEventListener('keydown', event => {
    const list = [...$('statTabs').querySelectorAll('[role=tab]')], index = list.indexOf(root.activeElement);
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % list.length;
    if (event.key === 'ArrowLeft') next = (index - 1 + list.length) % list.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = list.length - 1;
    if (next != null) { event.preventDefault(); selectStat(list[next].dataset.stat); }
  });
  $('moreStats').addEventListener('click', () => togglePanel('moreStats', 'extraStats'));
  $('filterButton').addEventListener('click', () => togglePanel('filterButton', 'filterPanel'));
  for (const [group, key] of [['periodControl', 'period'], ['venueControl', 'venue'], ['rateControl', 'rate']]) $(group).addEventListener('click', event => {
    const button = event.target.closest('button'); if (!button) return;
    state[key] = button.dataset[key]; render();
  });
  $('thresholdToggle').addEventListener('click', () => { state.thresholdOn = !state.thresholdOn; render(); });
  $('thresholdDown').addEventListener('click', () => { state.threshold = Math.max(metrics[state.stat].step || 1, state.threshold - (metrics[state.stat].step || 1)); render(); });
  $('thresholdUp').addEventListener('click', () => { state.threshold = Math.min(metrics[state.stat].limit || 10, state.threshold + (metrics[state.stat].step || 1)); render(); });
  $('compareSeason').addEventListener('change', event => { state.compare = event.target.checked; render(); });
  $('resetFilters').addEventListener('click', () => { Object.assign(state, {venue: 'all', period: '10', rate: 'match', thresholdOn: false}); go(state.area, state.id, state.scope, current().defaultStat); render(); });
  $('chartColumns').addEventListener('click', event => { const column = event.target.closest('[data-record]'); if (column) renderSelected(state.records.find(r => r.id === column.dataset.record)); });
  $('chartColumns').addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    const buttons = [...$('chartColumns').children], index = buttons.indexOf(root.activeElement);
    if (index < 0) return;
    event.preventDefault();
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 : Math.max(0, Math.min(buttons.length - 1, index + (event.key === 'ArrowRight' ? 1 : -1)));
    buttons[next].focus({preventScroll: true}); buttons[next].click(); buttons[next].scrollIntoView({block: 'nearest', inline: 'nearest', behavior: 'instant'});
  });
  root.addEventListener('keydown', event => {
    if (event.key !== 'Escape') return;
    for (const [button, panel] of [['moreStats', 'extraStats'], ['filterButton', 'filterPanel']]) if (!$(panel).hidden) { togglePanel(button, panel); $(button).focus(); break; }
  });
  window.addEventListener('resize', () => { $('chartBlock').classList.toggle('chart-overflows', $('chartViewport').scrollWidth > $('chartViewport').clientWidth + 2); });
  window.addEventListener('scroll', () => { if (tip() && !tip().hidden) tip().hidden = true; }, {passive: true});
  return {showRoute: fromRoute};
}

class SpixResearch extends HTMLElement {
  connectedCallback() {
    if (this.ready) return;
    const root = this.attachShadow({mode: 'open'});
    root.innerHTML = '<p role="status">Loading Research Area…</p>';
    this.ready = Promise.all([
      fetch(new URL('./content.html?v=real-20261008a', import.meta.url)).then(r => { if (!r.ok) throw new Error('Research content unavailable'); return r.text(); }),
      fetch(new URL('./research.css?v=real-20261008a', import.meta.url)).then(r => { if (!r.ok) throw new Error('Research styles unavailable'); return r.text(); })
    ]).then(([html, css]) => {
      root.innerHTML = `<style>${css}</style>${html}`;
      this.controller = mountResearch(root, route => this.dispatchEvent(new CustomEvent('research-navigate', {bubbles: true, composed: true, detail: {route}})));
    }).catch(error => {
      root.innerHTML = '<p role="alert">Research Area could not load. <button type="button">Try again</button></p>';
      root.querySelector('button').addEventListener('click', () => location.reload());
      throw error;
    });
  }
  async showRoute(route) { await this.ready; this.controller.showRoute(route); }
}
customElements.define('spix-research', SpixResearch);
