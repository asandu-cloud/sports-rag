/* The hero animation on the home page (D87): someone using Spix on an example match, on a loop. Copied from the
   landing preview (D86); its ids carry an hr- prefix so they never meet the app's. */
(() => {
  const $ = (id) => document.getElementById(`hr-${id}`);
  if (!$('run')) return;
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches;

  const run = $('run'), ball = $('ball'), slot = $('slot'), track = $('track'), fill = $('fill'), mouse = $('mouse');
  const area = $('area'), teamRoll = $('teamRoll'), page = $('page'), strip = $('strip'), sheetRoll = $('sheetRoll');
  const words = [...$('rail').children];
  const leg = (name) => run.querySelector(`[data-leg="${name}"]`);
  const RADIUS = 11, TICK = 100;

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

  // The slip bar rises with the first leg, as in the app.
  const count = () => {
    const picked = run.querySelectorAll('[data-leg].on').length;
    $('count').textContent = picked;
    $('toBet').classList.toggle('up', picked > 0);
  };
  const type = async (text) => {
    for (const letter of text) { $('typed').textContent += letter; await beat(110); }
  };
  const tap = async (name) => {
    const add = leg(name).querySelector('.add');
    await click(add, add, .5);
    leg(name).classList.toggle('on');
    count();
  };
  // Scrolling: three notches of a wheel, to a place measured from the layout (transforms don't move it).
  const scroll = async (box, to) => {
    const from = Number(box.dataset.at || 0), notches = 3;
    for (let i = 1; i <= notches; i++) {
      box.style.transform = `translateY(${-(from + (to - from) * i / notches)}px)`;
      await beat(210);
    }
    box.dataset.at = to;
    await beat(300);
  };
  const top = (el, box) => { let y = 0; for (let n = el; n && n !== box; n = n.offsetParent) y += n.offsetTop; return y; };
  const deepest = (box) => Math.max(0, box.offsetHeight - box.parentElement.offsetHeight);
  const toTop = (box, el, gap = 8) => scroll(box, Math.min(deepest(box), Math.max(0, top(el, box) - gap)));
  // The Most likely strip moves sideways, the hand dragging it; the last tile stops at the right edge.
  const drag = async (tile) => {
    const last = strip.lastElementChild;
    const x = Math.min(tile.offsetLeft, Math.max(0, last.offsetLeft + last.offsetWidth - strip.offsetWidth));
    const from = Number(strip.dataset.at || 0);
    if (x === from) return;
    const r = run.getBoundingClientRect(), box = page.parentElement.getBoundingClientRect(), row = strip.getBoundingClientRect();
    const seen = (Math.max(row.top, box.top) + Math.min(row.bottom, box.bottom)) / 2;
    await glide({ x: box.left - r.left + box.width * (x > from ? .74 : .26), y: seen - r.top });
    strip.style.transform = `translateX(${-x}px)`;
    strip.dataset.at = x;
    await glide({ x: mouseAt.x + (from - x) * .55, y: mouseAt.y + 4 }, .7);
    await beat(300);
  };

  const startOver = () => {
    run.classList.add('hold');
    run.dataset.scene = 'research';
    area.classList.remove('found', 'seen');
    $('field').classList.remove('on');
    $('typed').textContent = '';
    run.querySelectorAll('[data-leg].on').forEach(option => option.classList.remove('on'));
    [$('sheet'), $('dim')].forEach(el => el.classList.remove('open'));
    [teamRoll, page, sheetRoll].forEach(box => { box.style.transform = ''; box.dataset.at = 0; });
    strip.style.transform = ''; strip.dataset.at = 0;
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

      // The match page, and how Spix reads it
      await click($('toRead'), $('toRead').querySelector('span'), .5);
      run.dataset.scene = 'read'; say('See how Spix reads it'); step(1); roll(1); hover(null);
      await beat(800);
      await glide(aim($('lead'), .3), 1.2);
      await beat(700);
      await glide(aim($('kv'), .45, .25), 1.3);
      await beat(900);

      // Looking through the picks
      const [first, second, third] = strip.children;
      run.dataset.scene = 'lists'; say('Look through the picks'); step(2);
      roll(2);
      await toTop(page, $('mostHead'));
      await pass(first, first.querySelector('.t-pick'), 520);
      await pass(first, first.querySelector('.t-verdict'), 620);
      await drag(second);
      await pass(second, second.querySelector('.t-pick'), 620);
      await drag(third);
      await pass(third, third.querySelector('.t-pick'), 620);
      hover(null);
      await toTop(page, $('valueHead'));
      await pass(leg('over25'), leg('over25').querySelector('.v-worth'), 820);
      hover(null);

      // The matches behind one pick
      say('See why Spix likes it');
      await toTop(page, first.querySelector('.t-verdict'), 12);
      await drag(first);
      await click($('evRow'), $('evRow').querySelector('.t-count'), .5);
      $('dim').classList.add('open'); $('sheet').classList.add('open'); hover(null);
      await beat(700);
      await glide(aim($('tally'), .3), 1.2);
      await beat(900);
      await toTop(sheetRoll, $('sheetTable'), 4);
      await glide(aim($('sheetTable'), .3, .45), 1.2);
      await beat(1100);
      await click($('closeSheet'));
      $('sheet').classList.remove('open'); $('dim').classList.remove('open'); hover(null);
      await beat(500);

      say('Keep what you like');
      await tap('arsenal'); await beat(300);
      await drag(third);
      await tap('btts'); await beat(300);
      await toTop(page, $('valueHead'));
      await tap('over25'); await beat(520);

      say("Drop what you don't");
      await tap('over25'); await beat(520);

      await click($('toBet'), $('toBet').querySelector('.bar-left'), .5);
      run.dataset.scene = 'bet'; say('The bet is yours'); step(3); hover(null);
      await beat(380);
      await drop(slot);
      await glide(aim($('legs'), .36, .3), 1.2);
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
  // The home page and the board share one page, so the panel can start hidden: measure again whenever its size
  // changes, and start the loop the first time it is seen.
  const settle = () => setBall({ ...(words.includes(anchor) ? onLine(words.indexOf(anchor)) : spot(anchor)), turn: ballAt.turn });
  if ('ResizeObserver' in window) new ResizeObserver(settle).observe(run); else addEventListener('resize', settle);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(settle);
  if (still || !('IntersectionObserver' in window)) {
    setBall({ ...spot(slot), turn: 0 });
  } else {
    let started = false;
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
    new IntersectionObserver((seen) => {
      onScreen = seen[seen.length - 1].isIntersecting;
      if (onScreen && !started) { started = true; startOver(); play(); }
      rouse();
    }, { threshold: .25 }).observe(run);
    document.addEventListener('visibilitychange', rouse);
  }
})();
