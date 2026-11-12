/* The home page's first section (D98, D100). The corners pack runs by itself through the real lines of one saved
   Match Read (Brighton v Brentford, Sun 8 Nov 2026, prices as of 13:56 UK): each line's chance of going over and its
   typical price, then back to Spix's pick, held for a rest before the next run. Chances are said with the board's
   own chanceWords (app.js). */
const HOME_HERO_LINES = [[6.5, 0.915, 1.07], [7.5, 0.815, 1.19], [8.5, 0.704, 1.36], [9.5, 0.577, 1.65],
  [10.5, 0.45, 2.07], [11.5, 0.327, 2.67]];
const HOME_HERO_PICK = 8.5;
const HOME_HERO_BEAT_MS = 800;     // each line
const HOME_HERO_LAST_MS = 1100;    // before the run comes back to the pick
const HOME_HERO_REST_MS = 4000;    // the pick, held between runs

// The steps of one run: every line in turn, then Spix's pick again. Only the pick is marked.
function homeHeroSteps(words) {
  const steps = HOME_HERO_LINES.map(([line, chance, price]) => ({
    pick: `Over ${line}`, chance: words(chance), price: price.toFixed(2), marked: line === HOME_HERO_PICK,
  }));
  return [...steps, steps.find(step => step.marked)];
}

// The run, over and over: start(delay) shows the first line after delay; stop() goes back to the pick and waits.
// The card stands straight while it runs (data-running) and tilts back on the pick.
function homeHeroLoop(view, words, timers = { set: (fn, ms) => setTimeout(fn, ms), clear: id => clearTimeout(id) }) {
  const steps = homeHeroSteps(words);
  const last = steps.length - 1;
  let timer = null;
  let i = 0;
  const show = step => {
    view.pick.innerHTML = step.marked ? `<mark>${step.pick}</mark>` : step.pick;
    view.chance.textContent = step.chance;
    view.price.textContent = step.price;
  };
  const tick = () => {
    show(steps[i]);
    if (i < last) view.pack.dataset.running = '';
    else delete view.pack.dataset.running;
    i = i < last ? i + 1 : 0;
    timer = timers.set(tick, i === 0 ? HOME_HERO_REST_MS : i === last ? HOME_HERO_LAST_MS : HOME_HERO_BEAT_MS);
  };
  return {
    start(delay = HOME_HERO_REST_MS) {
      if (timer !== null) return;
      i = 0;
      timer = timers.set(tick, delay);
    },
    stop() {
      if (timer !== null) timers.clear(timer);
      timer = null;
      i = 0;
      show(steps[last]);
      delete view.pack.dataset.running;
    },
  };
}

// D100: the owner compares two desktop layouts; ?hero=b shows the second. Remove once one is chosen.
function homeHeroVariant(search) {
  return new URLSearchParams(search).get('hero') === 'b' ? 'b' : null;
}

if (typeof document !== 'undefined') {
  const section = document.querySelector('.home-hero');
  const variant = homeHeroVariant(location.search);
  if (section && variant) section.dataset.hero = variant;
  const pack = document.getElementById('hh-corners');
  const still = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  // It runs only while the card is on screen, the tab is open and no pointer rests on it; reduced motion keeps the pick.
  if (pack && !still && 'IntersectionObserver' in window) {
    const loop = homeHeroLoop({ pack, pick: document.getElementById('hh-cornersPick'),
      chance: document.getElementById('hh-cornersChance'), price: document.getElementById('hh-cornersPrice') }, chanceWords);
    let inView = false;
    let held = false;
    const sync = () => (inView && !held && !document.hidden ? loop.start(2500) : loop.stop());
    new IntersectionObserver(entries => { inView = entries[entries.length - 1].isIntersecting; sync(); }).observe(pack);
    document.addEventListener('visibilitychange', sync);
    pack.addEventListener('mouseenter', () => { held = true; sync(); });
    pack.addEventListener('mouseleave', () => { held = false; sync(); });
  }
}
