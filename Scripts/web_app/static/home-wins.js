/* The home page's wins reel (D101): last week's picks that came in, from both lists, gliding right to left under
   "Useful before you decide", with the honest record of every pick beside them. The feed is home_wins.v1
   (GET /api/home/wins); the section stays hidden until it arrives. Chances are said with the board's own
   chanceWords, markets with its RECS_MARKETS and leagues with its names (app.js). */
const HOME_WINS_SPEED = 28;   // pixels a second, on any screen
const HOME_WINS_MIN = 6;      // fewer wins and the section stays hidden: a short list looped would look like more
const HOME_WINS_FILL = 12;    // a set holds at least this many cards, so it is wider than any screen
const HOME_WINS_LISTS = { value: 'Value', most_likely: 'Most likely' };

// Each card's tilt (−5° to 5°, in half degrees) and height offset (±18 px) come from its match's id, so a card
// looks the same on every visit.
function homeWinsSeed(id) {
  let h = 2166136261;
  for (const c of String(id)) h = Math.imul(h ^ c.charCodeAt(0), 16777619);
  const r = () => (h = Math.imul(h ^ (h >>> 15), 2246822507) >>> 0) / 4294967296;
  return { tilt: Math.round((r() * 10 - 5) * 2) / 2, dy: Math.round(r() * 36 - 18) };
}

// The week as the heading says it: "4 to 8 Nov", or "30 Oct to 3 Nov" across a month.
function homeWinsWeek(week) {
  const parts = iso => {
    const d = new Date(`${iso}T12:00:00Z`);
    return [d.getUTCDate(), d.toLocaleDateString('en-GB', { month: 'short', timeZone: 'UTC' })];
  };
  const [fromDay, fromMonth] = parts(week.from);
  const [toDay, toMonth] = parts(week.to);
  if (week.from === week.to) return `${toDay} ${toMonth}`;
  return fromMonth === toMonth ? `${fromDay} to ${toDay} ${toMonth}` : `${fromDay} ${fromMonth} to ${toDay} ${toMonth}`;
}

// An expected count of wins: a half says both whole numbers ("about 4 or 5").
function homeWinsAbout(x) {
  const n = Math.round(Number(x) * 10) / 10;
  return n % 1 === 0.5 ? `about ${Math.floor(n)} or ${Math.ceil(n)}` : `about ${Math.round(n)}`;
}

// The record beside the reel: every pick of the week from both lists, losses included. Most likely gives its chance
// and what happened; Value, with few picks, gives counts.
function homeWinsRecord(data, h) {
  const list = name => (data.record || []).find(r => r.list === name && r.picks > 0);
  const ml = list('most_likely');
  const va = list('value');
  const picks = (n, name) => `<b>${Number(n)} ${name} pick${n === 1 ? '' : 's'}</b>`;
  const said = [];
  if (ml) {
    said.push(`made ${picks(ml.picks, 'Most likely')} and said ${h.words(ml.said)} would come in (${h.percent(ml.said)}).`
      + ` <b>${Number(ml.won)} did</b> (${h.percent(ml.happened)}).`);
  }
  if (va) {
    const voided = va.void ? `, and ${Number(va.void)} ${va.void === 1 ? 'was' : 'were'} void (stake back)` : '';
    said.push(`made ${picks(va.picks, 'Value')} and expected ${homeWinsAbout(va.expected_wins)} to win.`
      + ` <b>${Number(va.won)} did</b>${voided}.`);
  }
  if (!said.length) return '';
  return `Every pick counts, losses too. From ${homeWinsWeek(data.week)} Spix ${said.join(' It ')}`;
}

// One win as a card: the hero's pack without its band, the score between the crests. A Value pick is the dark card.
function homeWinsCard(w, h) {
  const f = w.fixture;
  const p = w.pick;
  const list = HOME_WINS_LISTS[w.list];
  const longShot = p.chance < 0.25;
  const chance = h.words(p.chance);
  const price = Number(p.typical_price).toFixed(2);
  const score = `${Number(w.score.home)}–${Number(w.score.away)}`;
  // The market's name says what is counted, so the marked pick drops the unit; both teams to score marks Yes or No.
  const marked = p.market === 'btts' ? (p.side === 'yes' ? 'Yes' : 'No') : String(p.label).replace(/ (goals|corners)$/, '');
  const day = new Date(f.kickoff).toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short', timeZone: 'Europe/London' }).replace(',', '');
  const crest = logo => `<span class="hw-crest">${logo ? `<img src="${h.esc(logo)}" alt="" decoding="async">` : ''}</span>`;
  const label = `${list} pick, ${f.home_team} ${Number(w.score.home)}, ${f.away_team} ${Number(w.score.away)}: ${p.label}, `
    + `${longShot ? 'a long shot, ' : ''}${chance}, typical price ${price}. It came in.`;
  return `<article class="hw-card${w.list === 'value' ? ' is-value' : ''}" role="img" aria-label="${h.esc(label)}">
    <div class="hw-body">
      <p class="hw-read">${list} pick · ${h.esc(f.home_team)} v ${h.esc(f.away_team)}</p>
      <p class="hw-market"><span data-fit>${h.esc(h.markets[p.market] || p.market)}</span></p>
      <p class="hw-pick"><mark>${h.esc(marked)}</mark></p>
      <div class="hw-score">${crest(w.visuals?.home_team_logo)}<b>${score}</b>${crest(w.visuals?.away_team_logo)}</div>
      <p class="hw-happened">${h.esc(w.result || '')}</p>
      <div class="hw-facts">
        <div><b>${chance}</b><span>${longShot ? 'chance: a long shot' : 'chance'}</span></div>
        <div><b>${price}</b><span>typical price</span></div>
      </div>
    </div>
    <div class="hw-stub"><span>Came in · ${day}</span><span>${h.esc(h.leagueName(f.league))}</span></div>
  </article>`;
}

// Only a real feed with enough wins is shown.
function homeWinsReady(data) {
  return !!data && data.version === 'home_wins.v1' && !!data.week && Array.isArray(data.wins)
    && data.wins.length >= HOME_WINS_MIN;
}

// The reel's track: the set twice, so moving it left by exactly one set (half the track) loops without a seam. A
// short week repeats inside each set until it fills one; the repeats, and the whole second set, are not read out.
function homeWinsTrack(wins, h) {
  const count = Math.max(wins.length, HOME_WINS_FILL);
  const set = Array.from({ length: count }, (_, i) => {
    const w = wins[i % wins.length];
    const s = homeWinsSeed(w.fixture.event_id);
    return `<li class="hw-slot"${i >= wins.length ? ' aria-hidden="true" inert' : ''} style="--tilt:${s.tilt}deg;--dy:${s.dy}px">${homeWinsCard(w, h)}</li>`;
  }).join('');
  return `<ul class="hw-set" aria-label="Picks that came in">${set}</ul><ul class="hw-set" aria-hidden="true" inert>${set}</ul>`;
}

// The loop's length for a constant speed: one set's width over the speed.
function homeWinsDuration(setWidth, speed = HOME_WINS_SPEED) {
  return `${(setWidth / speed).toFixed(1)}s`;
}

const HOME_WINS_ICONS = {
  pause: '<rect x="2" y="1" width="3.5" height="12" rx="1" fill="currentColor"/><rect x="8.5" y="1" width="3.5" height="12" rx="1" fill="currentColor"/>',
  play: '<path d="M3 1.5v11l9.5-5.5z" fill="currentColor"/>',
};

function homeWinsShow(section, data) {
  const h = { esc, words: chanceWords, percent: chancePercent, markets: RECS_MARKETS,
    leagueName: code => app.LEAGUES.find(l => l.id === code)?.name || code };
  const track = section.querySelector('.hw-track');
  section.querySelector('h2').textContent = `Picks that came in, ${homeWinsWeek(data.week)}`;
  section.querySelector('.hw-record').innerHTML = homeWinsRecord(data, h);
  track.innerHTML = homeWinsTrack(data.wins, h);
  section.hidden = false;

  const pace = () => track.style.setProperty('--dur', homeWinsDuration(track.firstElementChild.getBoundingClientRect().width));
  document.fonts.ready.then(() => {
    // The market's name fills its line, shrinking only when it is long.
    track.querySelectorAll('[data-fit]').forEach(el => {
      let size = 3.3;
      el.style.fontSize = `${size}em`;
      while (el.scrollWidth > el.parentElement.clientWidth && size > 1.4) {
        size -= 0.1;
        el.style.fontSize = `${size}em`;
      }
    });
    pace();
  });
  addEventListener('resize', pace);

  // The visible pause control (moving content that lasts over 5 seconds needs one); the pointer resting on the reel
  // also holds it (CSS). It rests while off screen.
  const pause = document.getElementById('homeWinsPause');
  pause.addEventListener('click', () => {
    const paused = section.classList.toggle('is-paused');
    pause.querySelector('span').textContent = paused ? 'Play' : 'Pause';
    pause.querySelector('svg').innerHTML = paused ? HOME_WINS_ICONS.play : HOME_WINS_ICONS.pause;
  });
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(entries => section.classList.toggle('is-away', !entries[entries.length - 1].isIntersecting))
      .observe(section);
  }
}

if (typeof document !== 'undefined') {
  const section = document.getElementById('homeWins');
  if (section) {
    fetch('/api/home/wins', { headers: { Accept: 'application/json' } })
      .then(r => (r.ok ? r.json() : null))
      .then(data => { if (homeWinsReady(data)) homeWinsShow(section, data); })
      .catch(() => {});
  }
}
