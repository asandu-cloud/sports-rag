/* The home page's first section (D98). The play badge runs the corners pack through the real lines of one saved
   Match Read (Brighton v Brentford, Sun 8 Nov 2026, prices as of 13:56 UK): each line's chance of going over and its
   typical price, then back to Spix's pick. Chances are said with the board's own chanceWords (app.js). */
const HOME_HERO_LINES = [[6.5, 0.915, 1.07], [7.5, 0.815, 1.19], [8.5, 0.704, 1.36], [9.5, 0.577, 1.65],
  [10.5, 0.45, 2.07], [11.5, 0.327, 2.67]];
const HOME_HERO_PICK = 8.5;

// The steps of one run: every line in turn, then Spix's pick again. Only the pick is marked.
function homeHeroSteps(words) {
  const steps = HOME_HERO_LINES.map(([line, chance, price]) => ({
    pick: `Over ${line}`, chance: words(chance), price: price.toFixed(2), marked: line === HOME_HERO_PICK,
  }));
  return [...steps, steps.find(step => step.marked)];
}

if (typeof document !== 'undefined') {
  const play = document.getElementById('hh-play');
  const pack = document.getElementById('hh-corners');
  const pick = document.getElementById('hh-cornersPick');
  const chance = document.getElementById('hh-cornersChance');
  const price = document.getElementById('hh-cornersPrice');
  const said = document.getElementById('hh-said');
  let timer = null;
  const show = step => {
    pick.innerHTML = step.marked ? `<mark>${step.pick}</mark>` : step.pick;
    chance.textContent = step.chance;
    price.textContent = step.price;
  };
  play?.addEventListener('click', () => {
    if (timer) return;
    const steps = homeHeroSteps(chanceWords);
    play.setAttribute('aria-pressed', 'true');
    pack.dataset.running = '';
    said.textContent = '';
    let i = 0;
    const next = () => {
      show(steps[i]);
      i += 1;
      if (i < steps.length) {
        timer = setTimeout(next, i === steps.length - 1 ? 1100 : 800);
        return;
      }
      timer = null;
      play.setAttribute('aria-pressed', 'false');
      delete pack.dataset.running;
      said.textContent = `Spix's pick: ${steps[i - 1].pick} corners, ${steps[i - 1].chance}, at a typical price of ${steps[i - 1].price}.`;
    };
    next();
  });
}
