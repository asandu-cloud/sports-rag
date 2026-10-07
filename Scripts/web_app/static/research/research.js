import {catalogues} from './data.js?v=research-20261007';
function mountResearch(root,navigate) {
  const $=id=>root.getElementById(id);
  const escape=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt=value=>Number.isFinite(value)?value.toLocaleString('en-GB',{maximumFractionDigits:1}):'Unavailable';
  const date=value=>new Date(value+'T12:00:00Z').toLocaleDateString('en-GB',{day:'numeric',month:'short',timeZone:'UTC'});
  let metrics=catalogues.players.metrics,common=catalogues.players.common,initialized=false;
  const state = {area:'players',player:null, stat:'sot', period:'10', venue:'all', rate:'match', thresholdOn:false, threshold:1, compare:false, selected:null};
  const value = record => {
    if (!record.appeared || record.stats[state.stat] == null) return null;
    if (state.rate === 'per90') return record.minutes > 0 ? record.stats[state.stat] * 90 / record.minutes : null;
    return record.stats[state.stat];
  };
  const scopedRecords = () => {
    const records = state.player.records.filter(r => state.venue === 'all' || r.venue === state.venue);
    return state.period === 'season' ? records : records.slice(-Number(state.period));
  };
  const average = records => {
    const known = records.filter(r => value(r) != null);
    if (!known.length) return null;
    const total = known.reduce((n,r) => n + r.stats[state.stat],0);
    return state.rate === 'per90' ? total * 90 / known.reduce((n,r) => n + r.minutes,0) : total / known.length;
  };
  const current = () => catalogues[state.area];
  const isPlayer = () => state.area === 'players';
  const metricValue = v => fmt(v) + (Number.isFinite(v) && metrics[state.stat].percent ? '%' : '');
  function search() {
    const query=$('playerSearch').value.trim().toLocaleLowerCase('en-GB');
    const found=current().entities.filter(p=>`${p.name} ${p.club} ${p.position}`.toLocaleLowerCase('en-GB').includes(query));
    $('resultsTitle').textContent=query?'Search results':`Explore a sample ${current().singular}`;
    $('resultCount').textContent=`${found.length} ${found.length===1?current().singular:current().label.toLowerCase()}`;
    $('playerResults').innerHTML=found.length?found.map(p=>`<button class="player-result" data-player="${p.id}"><span class="avatar" aria-hidden="true">${p.initials}</span><span class="player-result-copy"><strong>${escape(p.name)}</strong><small>${escape(p.club)}</small></span><span class="player-position">${p.position}</span><svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><path d="m9 5 7 7-7 7"/></svg></button>`).join(''):`<div class="search-empty"><h3>No sample ${current().label.toLowerCase()} found</h3><p>This design preview includes ${current().entities.length} fictional ${current().label.toLowerCase()}.</p><button class="text-button" id="clearSearch">Show all sample ${current().label.toLowerCase()}</button></div>`;
  }
  function updateRoute() {navigate(state.area+(state.player?`/${state.player.id}/${state.stat}`:''));}
  function setArea(area) {
    state.area=catalogues[area]?area:'players'; metrics=current().metrics;common=current().common;
    Object.assign(state,{player:null,stat:current().defaultStat,period:'10',venue:'all',rate:'match',thresholdOn:false,compare:false,selected:null});
    $('playerSearch').value='';
    $('searchTitle').textContent=`Find a ${current().singular}`;
    $('searchDescription').textContent=current().description;
    $('playerSearch').placeholder=current().search;
    $('playerSearch').setAttribute('aria-label',current().search);
    $('searchHelp').textContent=`Try a sample ${current().singular} below. All records in this preview are fictional.`;
    $('backSearchLabel').textContent=`Find another ${current().singular}`;
    $('venueField').hidden=state.area==='referees';
    $('rateField').hidden=!isPlayer();
    $('filterButton').hidden=state.area==='referees';
    $('contextHeading').textContent=current().context;
    $('researchIntroduction').textContent=isPlayer()?'See recent performances, time on the pitch and the records behind a player’s numbers.':state.area==='teams'?'Explore what a team produces and concedes. Home and away filters add context to each match.':'Explore an official’s match totals, then look at the individual fixtures behind an average.';
    $('researchTopics').textContent=isPlayer()?'Shots · Goals · Minutes':state.area==='teams'?'Scoring · Chances · Discipline':'Cards · Fouls · Penalties';
    $('dnpLegend').hidden=!isPlayer();
    root.querySelectorAll('[data-area]').forEach(a=>{if(a.dataset.area===state.area)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');});
    $('extraStats').innerHTML=current().groups.map(([title,keys])=>`<div><h3>${title}</h3>${keys.map(key=>`<button data-stat="${key}">${metrics[key].label}</button>`).join('')}</div>`).join('');
    search();showSearch(false);
  }
  function openPlayer(player, fromRoute = false) {
    Object.assign(state,{player,stat:current().defaultStat,period:'10',venue:'all',rate:'match',thresholdOn:false,threshold:metrics[current().defaultStat].threshold||1,compare:false,selected:null});
    $('searchView').hidden = true; $('playerView').hidden = false; root.host.dataset.profile='true';
    $('playerName').textContent = player.name;
    $('playerMeta').textContent = `${player.club} · ${player.position}`;
    $('playerAvatar').textContent = player.initials;
    $('nextFixture').textContent=player.next||'Regulation-time match totals';
    $('nextFixtureLabel').textContent=state.area==='referees'?'Scope of this view':'Example next fixture';
    $('fixtureHint').textContent=state.area==='referees'?'Not bookmaker settlement':isPlayer()?'Lineup not confirmed':'Sample fixture';
    $('filterPanel').hidden = true; $('filterButton').setAttribute('aria-expanded','false');
    $('extraStats').hidden = true; $('moreStats').setAttribute('aria-expanded','false');
    $('matchDetails').open = false;
    if (!fromRoute) updateRoute(true);
    render();
    window.scrollTo({top:0,behavior:'instant'});
    $('playerName').focus({preventScroll:true});
  }
  function showSearch(push = true) {
    state.player = null;
    $('searchView').hidden = false; $('playerView').hidden = true; root.host.dataset.profile='false';
    if (push) updateRoute();
    $('playerSearch').focus({preventScroll:true});
    window.scrollTo({top:0,behavior:'instant'});
  }
  function fromRoute(route='players') {
    const [area,id,stat]=route.split('/');
    if(state.area!==area||!initialized){initialized=true;setArea(area);}
    const player=current().entities.find(p=>p.id===id);
    if(!player)return showSearch(false);
    if(state.player?.id!==id)openPlayer(player,true);
    if(metrics[stat]&&state.stat!==stat){state.stat=stat;state.threshold=metrics[stat].threshold||1;render();}
  }
  function tabs() {
    const keys = common.includes(state.stat) ? common : [...common,state.stat];
    $('statTabs').innerHTML = keys.map(key => `<button id="tab-${key}" role="tab" aria-selected="${key===state.stat}" aria-controls="statPanel" tabindex="${key===state.stat ? 0:-1}" data-stat="${key}">${metrics[key].label}</button>`).join('');
    $('statPanel').setAttribute('aria-labelledby',`tab-${state.stat}`);
  }
  function selectStat(key, focus = true) {
    state.stat = key; state.threshold = metrics[key].threshold || 1;
    $('extraStats').hidden = true; $('moreStats').setAttribute('aria-expanded','false');
    updateRoute(); render();
    if (focus) $(`tab-${key}`).focus({preventScroll:true});
    $(`tab-${key}`).scrollIntoView({block:'nearest',inline:'nearest',behavior:'instant'});
  }
  function togglePanel(button, panel) {
    $(panel).hidden = !$(panel).hidden;
    $(button).setAttribute('aria-expanded',String(!$(panel).hidden));
  }
  function renderSelected(record) {
    state.selected = record.id;
    root.querySelectorAll('.match-column').forEach(button => button.setAttribute('aria-pressed',String(button.dataset.record === record.id)));
    const v = value(record);
    const label = !record.appeared ? 'Did not play' : v == null ? 'Unavailable' : `${metricValue(v)} ${metrics[state.stat].noun}${state.rate === 'per90' ? ' / 90' : ''}`;
    $('selectedMatch').innerHTML = `<div><strong>${state.area==='referees'?'':record.venue === 'home' ? 'v' : 'at'} ${record.opponent}</strong><small>${date(record.date)} · ${state.area==='referees'?'Example League':record.venue === 'home' ? 'Home' : 'Away'} · Sample match</small></div><div class="selected-value">${label}<span>${isPlayer()?(record.minutes == null ? 'Minutes unavailable' : `${record.minutes} minutes played`):'Regulation time'}</span></div>`;
  }
  function render() {
    if (!state.player) return;
    const metric = metrics[state.stat], records = scopedRecords();
    const eligible = records.filter(r => value(r) != null);
    const appearances = records.filter(r => r.appeared);
    const missing = appearances.length - eligible.length;
    const recent = average(records), season = average(state.player.records);
    const periodLabel=state.period==='season'?'Sample season':`Last ${state.period} ${state.area==='referees'?'officiated matches':'team matches'}`;
    tabs();
    for (const [group,key] of [['periodControl','period'],['venueControl','venue'],['rateControl','rate']]) {
      $(group).querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed',String(b.dataset[key]===state[key])));
    }
    const filtered = Number(state.venue !== 'all') + Number(state.rate !== 'match');
    $('filterCount').hidden = !filtered; $('filterCount').textContent = String(filtered);
    $('rateHelp').textContent = state.rate === 'per90' ? 'Per 90 adjusts for time played. Short appearances can produce large values; averages use total recorded minutes.' : 'Match counts show what actually happened. Missing values are excluded, never filled with zero.';
    $('thresholdToggle').disabled = state.rate === 'per90';
    $('thresholdToggle').title = state.rate === 'per90' ? 'Switch to match counts to explore a historical threshold.' : '';
    $('thresholdToggle').setAttribute('aria-pressed',String(state.thresholdOn && state.rate === 'match'));
    $('thresholdPanel').hidden = !state.thresholdOn || state.rate === 'per90';
    $('thresholdValue').textContent=`${state.threshold}${metric.percent?'%':''}+`;
    $('thresholdDown').disabled = state.threshold <= 1;
    $('thresholdUp').disabled = state.threshold >= (metric.limit||(state.stat==='passes'?100:10));
    $('compareSeason').checked = state.compare;
    $('statTitle').textContent = metric.title;
    $('statDefinition').textContent = metric.definition;
    let insight;
    if (!eligible.length) insight = 'There isn’t enough recorded data for this view.';
    else if (state.rate === 'per90') insight = `Averaged <strong>${fmt(recent)} ${metric.noun}</strong> per 90 minutes.`;
    else if(!isPlayer()&&!state.thresholdOn) insight=`Averaged <strong>${metricValue(recent)} ${metric.noun}</strong> across ${eligible.length} recorded matches.`;
    else {
      const count = eligible.filter(r => value(r) >= (state.thresholdOn ? state.threshold : 1)).length;
      insight = state.thresholdOn ? `Recorded ${state.threshold}${metric.percent?'%':''}+ ${metric.noun} in <strong>${count} of ${eligible.length}</strong> ${isPlayer()?'appearances':'matches'} with data.` : `${(metric.verb||'Recorded '+metric.noun)} in <strong>${count} of ${eligible.length}</strong> appearances with data.`;
    }
    $('mainInsight').innerHTML = insight;
    $('sampleContext').textContent = `${periodLabel}${state.venue !== 'all' ? ` · ${state.venue} only` : ''} · ${records.length ? `${date(records[0].date)}–${date(records.at(-1).date)} 2026` : 'No matches'}${missing ? ` · ${missing} unavailable` : ''}${records.length > appearances.length ? ` · ${records.length-appearances.length} did not play` : ''}`;
    $('chartMeasure').textContent = `${metric.title}${metric.percent?' (%)':state.rate === 'per90' ? ' per 90':metric.title.endsWith('per match')?'':' per match'}`;
    $('chartEmpty').hidden = eligible.length > 0;
    $('chartBlock').hidden = !eligible.length;
    const dataMax = Math.max(1,...eligible.map(value),state.compare && season != null ? season : 0,state.thresholdOn && state.rate === 'match' ? state.threshold : 0);
    const step = dataMax <= 4 ? 1 : dataMax <= 12 ? 2 : dataMax <= 30 ? 5 : 20;
    const max = metric.percent ? 100 : Math.ceil((dataMax + step * .5) / step) * step;
    const ticks = Array.from({length:Math.floor(max / step)+1},(_,i) => ({top:100-i*step/max*100, label:fmt(i*step)}));
    $('plotGrid').innerHTML = ticks.map(t => `<div class="grid-line" style="top:${t.top}%"></div>`).join('');
    $('chartAxis').innerHTML = ticks.map(t => `<span style="top:${t.top}%">${t.label}</span>`).join('');
    const refs = [];
    if (state.compare && season != null) refs.push(`<div class="reference-line average" style="top:${100-season/max*100}%"><span class="reference-label">Season ${metricValue(season)}</span></div>`);
    if (state.thresholdOn && state.rate === 'match') refs.push(`<div class="reference-line" style="top:${100-state.threshold/max*100}%"><span class="reference-label">${state.threshold}${metric.percent?'%':''}+</span></div>`);
    $('referenceLines').innerHTML = refs.join('');
    const barWidth=state.area==='referees'?76:44;
    $('chartColumns').style.setProperty('--bar-width',`${barWidth}px`);
    $('chartStage').style.minWidth = `${records.length*barWidth+Math.max(0,records.length-1)*8}px`;
    $('chartColumns').style.setProperty('--count',Math.max(records.length,1));
    $('chartColumns').innerHTML = records.map(r => {
      const v = value(r), height = v == null ? 0 : v/max*100;
      const title = `${date(r.date)}, ${state.area==='referees'?'match':r.venue === 'home' ? 'home to' : 'away at'} ${r.opponent}: ${!r.appeared ? 'did not play' : v == null ? 'statistic unavailable' : `${metricValue(v)} ${metric.noun}${state.rate === 'per90' ? ' per 90 minutes' : ''}`}; ${isPlayer()?(r.minutes == null ? 'minutes unavailable' : `${r.minutes} minutes played`):'regulation time'}`;
      const bar = v == null ? `<span class="missing-value">${r.appeared ? '—' : 'DNP'}</span>` : v === 0 ? '<span class="zero-value"></span><span class="bar-value">0</span>' : `<span class="stat-bar"></span><span class="bar-value">${fmt(v)}</span>`;
      return `<button class="match-column" data-record="${r.id}" aria-pressed="false" aria-label="${escape(title)}" title="${escape(title)}" style="--bar-height:${height}%"><span class="bar-area" aria-hidden="true">${bar}</span><span class="opponent-label" aria-hidden="true">${r.abbr}</span><span class="venue-label" aria-hidden="true">${state.area==='referees'?date(r.date):r.venue === 'home' ? 'H' : 'A'}</span></button>`;
    }).join('');
    if (records.length) renderSelected(records.find(r => r.id === state.selected) || records.at(-1));
    const minutesKnown = appearances.filter(r => r.minutes != null);
    const full = minutesKnown.filter(r => r.minutes >= 75).length;
    if(isPlayer()) {
      $('minutesInsight').textContent=minutesKnown.length?`Played 75+ minutes in ${full} of ${minutesKnown.length} appearances with recorded minutes.`:'No recorded minutes in this view.';
      $('contextFootnote').textContent='Short appearances mean fewer opportunities.';
    } else if(state.area==='teams') {
      const scored=records.filter(r=>r.stats.goals!=null&&r.stats.conceded!=null);
      const wins=scored.filter(r=>r.stats.goals>r.stats.conceded).length,draws=scored.filter(r=>r.stats.goals===r.stats.conceded).length;
      $('minutesInsight').textContent=`${wins} wins, ${draws} draws and ${scored.length-wins-draws} losses across ${scored.length} matches with scores.`;
      $('contextFootnote').textContent='Results provide context. They do not establish the value of a bet.';
    } else {
      $('minutesInsight').textContent=`${records.length} sample appointments in this view. These counts describe the matches this referee oversaw.`;
      $('contextFootnote').textContent='Team style, competition and match state matter too. Raw home/away counts do not establish bias.';
    }
    $('recentAverageLabel').textContent = state.period === 'season' ? (state.venue === 'all' ? 'This view' : `${state.venue === 'home' ? 'Home' : 'Away'} view`) : `Last ${records.length} matches`;
    $('recentAverage').textContent = metricValue(recent);
    $('seasonAverage').textContent = metricValue(season);
    $('comparisonInsight').textContent = state.rate === 'per90' ? 'Per 90, using total counts and minutes from eligible appearances. Season includes both venues.' : isPlayer()?'Average per appearance with a recorded value. Season includes both venues.':state.area==='teams'?'Average per recorded team match. Season includes both venues.':'Average per recorded appointment in the sample season.';
    $('coverageText').textContent = `${eligible.length} usable ${metric.noun} records in ${records.length} team matches. ${missing} appearance${missing===1 ? '' : 's'} unavailable for this measure; ${records.length-appearances.length} did not play. ${appearances.length-minutesKnown.length} appearance${appearances.length-minutesKnown.length===1 ? '' : 's'} with unknown minutes. Unknowns are excluded from the relevant calculation.`;
    if(!isPlayer())$('coverageText').textContent=`${eligible.length} usable ${metric.noun} records in ${records.length} matches. ${missing} missing values excluded. All figures cover regulation time.`;
    $('tableVenue').textContent=state.area==='referees'?'Scope':'Venue';
    $('tableContext').textContent=isPlayer()?'Minutes':state.area==='teams'?'Score':'Competition';
    $('detailCount').textContent = `${records.length} matches`;
    $('tableCaption').textContent = `${state.player.name}: synthetic ${metric.title.toLowerCase()} records for this view`;
    $('tableStat').textContent = `${metric.label}${state.rate === 'per90' ? ' / 90' : ''}`;
    $('matchRows').innerHTML = records.toReversed().map(r => `<tr><td>${r.opponent}<small>${date(r.date)}</small></td><td>${state.area==='referees'?'Both teams':r.venue === 'home' ? 'Home' : 'Away'}</td><td>${!r.appeared ? 'Did not play' : metricValue(value(r))}</td><td>${isPlayer()?(r.minutes == null ? 'Unknown' : r.minutes):state.area==='teams'?`${r.stats.goals}–${r.stats.conceded}`:'Example League'}</td></tr>`).join('');
    requestAnimationFrame(() => {
      const viewport=$('chartViewport');
      $('chartBlock').classList.toggle('chart-overflows',viewport.scrollWidth>viewport.clientWidth+2);
      viewport.scrollLeft=viewport.scrollWidth;
    });
  }
  $('playerSearch').addEventListener('input',search);
  $('playerResults').addEventListener('click',event => {
    const item=event.target.closest('[data-player]');
    if (item) openPlayer(current().entities.find(p=>p.id===item.dataset.player));
    if (event.target.closest('#clearSearch')) {$('playerSearch').value='';search();$('playerSearch').focus();}
  });
  $('backSearch').addEventListener('click',()=>showSearch());
  root.querySelectorAll('[data-area]').forEach(a=>a.addEventListener('click',event=>{event.preventDefault();setArea(a.dataset.area);updateRoute();}));
  for (const id of ['statTabs','extraStats']) $(id).addEventListener('click',event=>{const tab=event.target.closest('[data-stat]');if(tab)selectStat(tab.dataset.stat);});
  $('statTabs').addEventListener('keydown',event=>{
    const list=[...$('statTabs').querySelectorAll('[role=tab]')], index=list.indexOf(root.activeElement);
    let next;
    if (event.key==='ArrowRight') next=(index+1)%list.length;
    if (event.key==='ArrowLeft') next=(index-1+list.length)%list.length;
    if (event.key==='Home') next=0;
    if (event.key==='End') next=list.length-1;
    if(next!=null){event.preventDefault();selectStat(list[next].dataset.stat);}
  });
  $('moreStats').addEventListener('click',()=>togglePanel('moreStats','extraStats'));
  $('filterButton').addEventListener('click',()=>togglePanel('filterButton','filterPanel'));
  for (const [group,key] of [['periodControl','period'],['venueControl','venue'],['rateControl','rate']]) $(group).addEventListener('click',event=>{
    const button=event.target.closest('button');if(!button)return;
    state[key]=button.dataset[key];render();
  });
  $('thresholdToggle').addEventListener('click',()=>{state.thresholdOn=!state.thresholdOn;render();});
  $('thresholdDown').addEventListener('click',()=>{state.threshold=Math.max(1,state.threshold-(metrics[state.stat].step||1));render();});
  $('thresholdUp').addEventListener('click',()=>{state.threshold=Math.min(metrics[state.stat].limit||(state.stat==='passes'?100:10),state.threshold+(metrics[state.stat].step||1));render();});
  $('compareSeason').addEventListener('change',event=>{state.compare=event.target.checked;render();});
  $('resetFilters').addEventListener('click',()=>{Object.assign(state,{venue:'all',period:'10',rate:'match',stat:state.area==='referees'?'fouls':'shots',threshold:1,thresholdOn:false});updateRoute();render();});
  $('chartColumns').addEventListener('click',event=>{const column=event.target.closest('[data-record]');if(column)renderSelected(state.player.records.find(r=>r.id===column.dataset.record));});
  $('chartColumns').addEventListener('keydown',event=>{
    if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
    const buttons=[...$('chartColumns').children],index=buttons.indexOf(root.activeElement);
    if(index<0)return;
    event.preventDefault();
    const next=event.key==='Home'?0:event.key==='End'?buttons.length-1:Math.max(0,Math.min(buttons.length-1,index+(event.key==='ArrowRight'?1:-1)));
    buttons[next].focus({preventScroll:true});buttons[next].click();buttons[next].scrollIntoView({block:'nearest',inline:'nearest',behavior:'instant'});
  });
  root.addEventListener('keydown',event=>{
    if(event.key!=='Escape')return;
    for(const [button,panel] of [['moreStats','extraStats'],['filterButton','filterPanel']]) if(!$(panel).hidden){togglePanel(button,panel);$(button).focus();break;}
  });

  window.addEventListener('resize',()=>{$('chartBlock').classList.toggle('chart-overflows',$('chartViewport').scrollWidth>$('chartViewport').clientWidth+2);});
  return {showRoute:fromRoute};

}
class SpixResearch extends HTMLElement {
  connectedCallback() {
    if(this.ready)return;
    const root=this.attachShadow({mode:'open'});
    root.innerHTML='<p role="status">Loading Research Area…</p>';
    this.ready=Promise.all([
      fetch(new URL('./content.html?v=research-20261007',import.meta.url)).then(r=>{if(!r.ok)throw new Error('Research content unavailable');return r.text();}),
      fetch(new URL('./research.css?v=research-20261007',import.meta.url)).then(r=>{if(!r.ok)throw new Error('Research styles unavailable');return r.text();})
    ]).then(([html,css])=>{
      root.innerHTML=`<style>${css}</style>${html}`;
      this.controller=mountResearch(root,route=>this.dispatchEvent(new CustomEvent('research-navigate',{bubbles:true,composed:true,detail:{route}})));
    }).catch(error=>{
      root.innerHTML='<p role="alert">Research Area could not load. <button type="button">Try again</button></p>';
      root.querySelector('button').addEventListener('click',()=>location.reload());
      throw error;
    });
  }
  async showRoute(route) {await this.ready;this.controller.showRoute(route);}
}
customElements.define('spix-research',SpixResearch);
