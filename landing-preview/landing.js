/* New Spix landing page: how Spix works, the match panel, today's matches, the theme toggle. */
(() => {
  // Example match for the "Everything in one place" panel: [short version, [[label, value], ...]].
  const RESEARCH = {
    form: ['Arsenal come in on the better run.', [['Arsenal, last five', 'W W D W L'], ['Chelsea, last five', 'D W L W D'], ['Goals per game', '2.1 and 1.6']]],
    news: ['No key absences for Arsenal.', [['Arsenal', 'One doubt in midfield'], ['Chelsea', 'First-choice striker out'], ['Line-ups', 'About an hour before kick-off']]],
    h2h: ['These two have been close lately.', [['Last five meetings', 'Arsenal 2, draws 2, Chelsea 1'], ['Goals per meeting', '2.8'], ['Both teams scored', '3 of 5']]],
    goals: ['Spix expects about 2.9 goals.', [['Over 1.5 goals', '80%'], ['Over 2.5 goals', '55%'], ['Both teams to score', '59%']]],
    set: ['A busy game for corners, average for cards.', [['Expected corners', 'About 10'], ['Expected cards', 'About 4'], ['Referee', 'Average for cards']]],
    players: ['Three players most likely to score.', [['Arsenal striker', '36%'], ['Chelsea winger', '28%'], ['Arsenal winger', '24%']]],
  };

  const LEAGUES = { EPL: 'Premier League', LaLiga: 'La Liga', SerieA: 'Serie A', Bundesliga: 'Bundesliga', Ligue1: 'Ligue 1',
    UCL: 'Champions League', UEL: 'Europa League', UECL: 'Conference League', SuperLig: 'Süper Lig' };
  const TOP_LEAGUES = ['EPL', 'UCL', 'LaLiga', 'SerieA', 'Bundesliga', 'Ligue1', 'UEL'];
  const TODAY_ROWS = 5;

  const $ = (id) => document.getElementById(id);
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const escape = (text) => String(text).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  /* ---- How Spix works: someone using it, on a loop ---- */
  const run = $('run'), ball = $('ball'), slot = $('slot'), track = $('track'), fill = $('fill'), mouse = $('mouse'), roller = $('roll');
  const area = $('area'), teamRoll = $('teamRoll'), words = [...$('rail').children];
  const leg = (name) => run.querySelector(`[data-leg="${name}"]`);
  const RADIUS = 11, LANDED = 620, TICK = 100;

  let ballAt = { x: 0, y: 0, turn: 0 }, anchor = slot, onScreen = false, paused = false, wake = null;
  let mouseAt = { x: 0, y: 0 }, under = null;

  const pose = ({ x, y, turn }, extra = '') => `translate(${x}px, ${y}px) rotate(${turn}rad) ${extra}`;
  const setBall = (next) => { ballAt = next; ball.style.transform = pose(next); };
  const spot = (el) => {
    const p = run.getBoundingClientRect(), b = el.getBoundingClientRect();
    return { x: b.left - p.left + b.width / 2 - RADIUS, y: b.top - p.top + b.height / 2 - RADIUS };
  };
  const onLine = (i) => ({ x: spot(track).x, y: spot(words[i]).y });
  const place = (at) => setBall({ ...at, turn: ballAt.turn + (at.y - ballAt.y) / RADIUS });

  const sleep = (ms) => new Promise(done => setTimeout(done, ms));
  const held = () => paused || !onScreen || document.hidden;
  const free = async () => { while (held()) await new Promise(done => { wake = done; }); };
  // A beat only counts down while the panel is on screen, the tab is open and nobody has paused it.
  const beat = async (ms) => {
    for (let left = ms; left > 0; left -= TICK) {
      await sleep(Math.min(TICK, left));
      await free();
    }
  };
  const rouse = () => { if (wake && !held()) { wake(); wake = null; } };

  const say = (text) => {
    $('say').textContent = text;
    $('say').animate([{ transform: 'translateY(105%)' }, { transform: 'none' }], { duration: 480, easing: 'cubic-bezier(.16,1,.3,1)' });
  };
  const step = (i) => {
    words.forEach((word, k) => word.classList.toggle('hit', k <= i));
    const box = track.getBoundingClientRect(), word = words[i].getBoundingClientRect();
    fill.style.transform = `scaleY(${(word.top + word.height / 2 - box.top) / box.height})`;
  };
  const roll = (i) => {
    const from = ballAt;
    anchor = words[i]; place(onLine(i));
    return ball.animate([{ transform: pose(from) }, { transform: pose(ballAt) }], { duration: 900, easing: 'cubic-bezier(.76,0,.24,1)' }).finished;
  };
  // The logo's drop (D48): up, down onto the bet, one squash and a small rebound.
  const drop = (el) => {
    const from = ballAt, to = spot(el), turn = from.turn;
    const at = (y, extra) => pose({ x: to.x, y, turn }, extra);
    anchor = el; setBall({ ...to, turn });
    return ball.animate([
      { transform: pose(from), easing: 'cubic-bezier(0,.55,.45,1)' },
      { transform: pose({ x: (from.x + to.x) / 2, y: Math.min(from.y, to.y) - 30, turn }), offset: .32, easing: 'cubic-bezier(.55,0,1,.45)' },
      { transform: at(to.y, 'scale(.92,1.1)'), offset: .68 },
      { transform: at(to.y + 2, 'scale(1.28,.72)'), offset: .74, easing: 'cubic-bezier(0,.55,.45,1)' },
      { transform: at(to.y - 12, 'scale(1,1)'), offset: .87, easing: 'cubic-bezier(.55,0,1,.45)' },
      { transform: at(to.y, 'scale(1,1)') },
    ], { duration: 820 }).finished;
  };

  /* The mouse: curved paths at a hand's speed, never quite the same spot twice */
  const setMouse = (at) => { mouseAt = at; mouse.style.transform = `translate(${at.x}px, ${at.y}px)`; };
  const hover = (el) => { if (under) under.classList.remove('hover'); under = el; if (el) el.classList.add('hover'); };
  const aim = (el, across = .5, down = .5) => {
    const p = run.getBoundingClientRect(), b = el.getBoundingClientRect(), wobble = () => (Math.random() - .5) * .08;
    return { x: b.left - p.left + b.width * (across + wobble()), y: b.top - p.top + b.height * (down + wobble()) };
  };
  const smooth = (t) => t * t * t * (10 + t * (6 * t - 15));
  const glide = async (to, pace = 1) => {
    await free();
    const from = mouseAt, far = Math.hypot(to.x - from.x, to.y - from.y);
    const bow = far * .12 * (Math.random() < .5 ? -1 : 1);
    const bend = { x: (from.x + to.x) / 2 + (to.y - from.y) / (far || 1) * bow, y: (from.y + to.y) / 2 - (to.x - from.x) / (far || 1) * bow };
    const frames = Array.from({ length: 25 }, (_, i) => {
      const t = smooth(i / 24), u = 1 - t;
      return { transform: `translate(${u * u * from.x + 2 * u * t * bend.x + t * t * to.x}px, ${u * u * from.y + 2 * u * t * bend.y + t * t * to.y}px)` };
    });
    setMouse(to);
    await mouse.animate(frames, { duration: (280 + 160 * Math.log2(1 + far / 24)) * pace }).finished;
  };
  const press = async (el) => {
    await free();
    const dip = [{ transform: 'scale(1)' }, { transform: 'scale(.84)' }, { transform: 'scale(1)' }];
    mouse.querySelector('.m-hand').animate(dip, { duration: 190 });
    mouse.querySelector('.m-dot').animate(dip, { duration: 190 });
    mouse.querySelector('.m-ring').animate([{ transform: 'scale(.35)', opacity: .9 }, { transform: 'scale(1.2)', opacity: 0 }], { duration: 420, easing: 'cubic-bezier(.16,1,.3,1)' });
    el.classList.add('press');
    await beat(120);
    el.classList.remove('press');
  };
  const click = async (target, part = target, across, down) => {
    await glide(aim(part, across, down));
    hover(target);
    await beat(170);
    await press(target);
  };
  const pass = async (target, part, pause, pace = 1.25) => {
    await glide(aim(part, .4, .5), pace);
    hover(target);
    await beat(pause);
  };

  const count = () => {
    const picked = run.querySelectorAll('.opt.on').length;
    $('count').textContent = picked ? `${picked} leg${picked > 1 ? 's' : ''}` : 'No legs yet';
  };
  const type = async (text) => {
    for (const letter of text) { $('typed').textContent += letter; await beat(110); }
  };
  const tap = async (name) => {
    await click(leg(name), leg(name).querySelector('.o-tick'));
    leg(name).classList.toggle('on');
    count();
  };
  const scroll = async (box, to) => {
    const from = Number(box.dataset.at || 0), notches = 3;
    for (let i = 1; i <= notches; i++) {
      box.style.transform = `translateY(${-(from + (to - from) * i / notches)}px)`;
      await beat(210);
    }
    box.dataset.at = to;
  };
  const deepest = (box) => Math.max(0, box.offsetHeight - box.parentElement.offsetHeight);

  const startOver = () => {
    run.classList.add('hold');
    run.dataset.scene = 'research';
    area.classList.remove('found', 'seen');
    $('field').classList.remove('on');
    $('typed').textContent = '';
    run.querySelectorAll('.opt.on').forEach(option => option.classList.remove('on'));
    $('drawer').classList.remove('open');
    [roller, teamRoll].forEach(box => { box.style.transform = ''; box.dataset.at = 0; });
    hover(null);
    count();
    $('say').textContent = 'Spix does the research';
    step(0);
    anchor = words[0]; setBall({ ...onLine(0), turn: 0 });
    setMouse({ x: run.offsetWidth + 30, y: run.offsetHeight * .72 });
    void run.offsetWidth;
    run.classList.remove('hold');
  };

  const play = async () => {
    for (;;) {
      await beat(1500);

      // Through the Research Area to the statistics
      say('Look up a team');
      await click($('field'), $('field'), .3);
      $('field').classList.add('on');
      hover(null);
      await beat(240);
      await type('Arsenal');
      area.classList.add('found');
      await beat(420);
      await click($('hit'), $('hit').querySelector('b'), .6);
      area.classList.add('seen');
      say('Find the statistics'); hover(null);
      await beat(620);
      await pass($('corners'), $('corners'), 620);
      await pass($('often'), $('often').querySelector('span'), 720);
      hover(null);
      await scroll(teamRoll, $('howOften').offsetTop - teamRoll.offsetTop);
      await pass(run.querySelector('.col:nth-child(3)'), run.querySelector('.col:nth-child(3) .pair'), 620);
      await pass(run.querySelector('.col:nth-child(5)'), run.querySelector('.col:nth-child(5) .pair'), 620);
      hover(null);

      // The Match Read
      say('Read the short version');
      await click($('toRead'));
      run.dataset.scene = 'read'; step(1); roll(1); hover(null);
      await beat(620);
      await glide(aim(run.querySelector('.read-short'), .34), 1.2);
      await beat(420);
      await glide(aim(run.querySelector('.read-long'), .78, .3), 1.4);
      await beat(420);
      await glide(aim(run.querySelector('.run-read ul'), .42, .76), 1.3);
      await beat(420);
      await click($('toPicks'));

      // Looking through the picks
      run.dataset.scene = 'lists'; say('Look through the picks'); step(2); hover(null);
      await roll(2);
      await pass(leg('over15'), leg('over15').querySelector('.o-main'), 400);
      await pass(leg('arsenal'), leg('arsenal').querySelector('.o-main'), 400);
      await pass(leg('btts'), leg('btts').querySelector('.o-main'), 420);
      hover(null);
      await scroll(roller, deepest(roller));
      await pass(leg('striker'), leg('striker').querySelector('.o-main'), 520);
      hover(null);
      await scroll(roller, 0);
      await beat(200);

      say('See why Spix likes it');
      await click(leg('corners'), leg('corners').querySelector('.o-main b'), .4);
      $('drawer').classList.add('open');
      await beat(500);
      await glide(aim($('drawer').querySelector('dl'), .3, .4), 1.2);
      await beat(900);
      await click($('closeWhy'), $('closeWhy'));
      $('drawer').classList.remove('open');
      hover(null);
      await beat(420);

      say('Keep what you like');
      await tap('btts'); await beat(200);
      await tap('arsenal'); await beat(200);
      await tap('corners'); await beat(480);

      say("Drop what you don't");
      await tap('arsenal'); await beat(460);

      await click($('toBet'));
      run.dataset.scene = 'bet'; say('The bet is yours'); step(3); hover(null);
      await beat(380);
      await drop(slot);
      await glide(aim(run.querySelector('.legs'), .36, .3), 1.2);
      await beat(340);
      await glide(aim($('chance'), .8), 1.2);
      await beat(1400);
      await glide({ x: run.offsetWidth + 30, y: run.offsetHeight * .78 }, 1.2);
      await beat(500);

      run.classList.add('fade');
      await sleep(340);
      startOver();
      run.classList.remove('fade');
    }
  };

  ball.hidden = false;
  addEventListener('resize', () => setBall({ ...(words.includes(anchor) ? onLine(words.indexOf(anchor)) : spot(anchor)), turn: ballAt.turn }));
  if (still || !('IntersectionObserver' in window)) {
    setBall({ ...spot(slot), turn: 0 });
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => setBall({ ...spot(slot), turn: 0 }));
  } else {
    run.classList.add('live');
    startOver();
    mouse.hidden = false;
    $('pause').hidden = false;
    $('pause').addEventListener('click', () => {
      paused = !paused;
      run.classList.toggle('paused', paused);
      $('pause').setAttribute('aria-label', `${paused ? 'Resume' : 'Pause'} the example`);
      run.getAnimations({ subtree: true }).forEach(move => (paused ? move.pause() : move.play()));
      rouse();
    });
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { step(0); setBall({ ...onLine(0), turn: 0 }); });
    new IntersectionObserver((seen) => { onScreen = seen[seen.length - 1].isIntersecting; rouse(); }, { threshold: .25 }).observe(run);
    document.addEventListener('visibilitychange', rouse);
    play();
  }

  /* ---- Everything in one place ---- */
  const researchTabs = [...document.querySelectorAll('#researchTabs button')];
  researchTabs.forEach(tab => tab.addEventListener('click', () => {
    const [short, rows] = RESEARCH[tab.dataset.tab];
    researchTabs.forEach(other => other.setAttribute('aria-pressed', String(other === tab)));
    $('researchShort').textContent = short;
    $('researchRows').innerHTML = rows.map(([label, value]) => `<div><dt>${label}</dt><dd>${value}</dd></div>`).join('');
  }));

  /* ---- Today's matches ---- */
  const dayKey = (date) => `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
  const leagueName = (key) => LEAGUES[key] || String(key).replace(/([a-z])([A-Z])/g, '$1 $2');
  const rank = (key) => { const i = TOP_LEAGUES.indexOf(key); return i < 0 ? TOP_LEAGUES.length : i; };
  const upcoming = async (date) => {
    const response = await fetch(`/api/match-reads/schedule/${dayKey(date)}`);
    if (!response.ok) throw new Error('schedule unavailable');
    const { fixtures } = await response.json();
    const now = Date.now();
    return fixtures.map(item => item.fixture).filter(f => new Date(f.kickoff).getTime() > now)
      .sort((a, b) => rank(a.league) - rank(b.league) || new Date(a.kickoff) - new Date(b.kickoff)).slice(0, TODAY_ROWS)
      .sort((a, b) => new Date(a.kickoff) - new Date(b.kickoff));
  };
  const loadToday = async () => {
    const box = $('todayRows');
    try {
      const today = new Date(), next = new Date(today.getTime() + 864e5);
      let fixtures = await upcoming(today), label = "Today's matches";
      if (!fixtures.length) { fixtures = await upcoming(next); label = "Tomorrow's matches"; }
      if (!fixtures.length) { box.innerHTML = '<p class="match-note">No matches in the next two days.</p>'; return; }
      $('todayTitle').textContent = label;
      box.innerHTML = fixtures.map(f => {
        const time = new Date(f.kickoff).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
        return `<a class="match" href="/app"><b>${escape(f.home_team)} v ${escape(f.away_team)}</b><span>${time} · ${escape(leagueName(f.league))}</span><em>Open</em></a>`;
      }).join('');
    } catch (error) {
      box.innerHTML = '<a class="match" href="/app"><b>See the matchday board</b><span>The live list could not be loaded here.</span><em>Open</em></a>';
    }
  };
  loadToday();

  /* ---- Theme, stored the same way as the rest of the site ---- */
  const toggleTheme = $('themeToggle');
  const paintTheme = () => toggleTheme.setAttribute('aria-label', `Switch to ${document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'} theme`);
  toggleTheme.addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('spix-theme', next); } catch (error) { /* the choice just lasts for this visit */ }
    paintTheme();
  });
  paintTheme();

  /* ---- Motion ---- */
  if (!still && 'IntersectionObserver' in window) {
    const reveal = new IntersectionObserver((seen) => seen.forEach(entry => {
      if (entry.isIntersecting) { entry.target.classList.add('show'); entry.target.classList.remove('pre'); reveal.unobserve(entry.target); }
    }), { rootMargin: '0px 0px -8% 0px' });
    document.querySelectorAll('.reveal').forEach((el, i) => {
      if (el.getBoundingClientRect().top > innerHeight) { el.classList.add('pre'); el.style.transitionDelay = `${(i % 3) * 70}ms`; reveal.observe(el); }
    });
  }
})();
