/* The six loading screens (D112, owner 16 Nov). Each loader is HTML from spixLoader(kind, …); loaders.css moves it:
   it shows only after 300 ms, its words step in CSS, and it rests on a still frame with reduced motion. Screen
   readers hear one fixed line. The Research Area's shadow root calls the same functions through window.
     ball    A  the logo's football drops onto the i (the first-load cover, spixCover)
     pitch   B  the pitch markings draw themselves (the Research Area's match view)
     last10  C  ten columns grow and the line drops in (a Research Area profile)
     prices  D  three prices roll and settle (Best Bets)
     card    E  the board card's outline with a sweep (the matchday board)
     roll    F  a football rolls along a touchline (short waits) */
const SPIX_LOADER_KINDS = ['ball', 'pitch', 'last10', 'prices', 'card', 'roll'];

const ldEsc = value => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;').replace(/'/g, '&#39;');

// The pitch, 122 by 82: the outline, halfway line, centre circle, both boxes, both six-yard boxes, both arcs.
const LD_PITCH = ['<rect x="3" y="3" width="116" height="76" rx="1"', '<path d="M61 3V79"', '<circle cx="61" cy="41" r="10"',
  '<path d="M3 21H21V61H3"', '<path d="M119 21H101V61H119"', '<path d="M3 32H9V50H3"', '<path d="M119 32H113V50H119"',
  '<path d="M21 34A9 9 0 0 1 21 48"', '<path d="M101 34A9 9 0 0 0 101 48"'];
const LD_PITCH_DELAY = [0, 1, 2, 3, 3, 4, 4, 5, 5];
// An example shape for the ten columns, not a team's figures: heights in % of the chart, five over the line.
const LD_BARS = [70, 30, 85, 40, 56, 100, 18, 62, 6, 38];
const LD_LINE = 44;
const LD_PRICES = [['Home', '2.10'], ['Draw', '3.40'], ['Away', '3.75']];
const LD_BALL = '<svg viewBox="-12 -12 24 24"><circle r="11" class="ld-ball-skin"/><path class="ld-ball-ink" d="M0-4.2 4-1.3 2.5 3.4H-2.5L-4-1.3Z"/>'
  + '<path class="ld-ball-seam" d="M0-4.2V-10.8M4-1.3 10.3-3.3M2.5 3.4 6.4 8.8M-2.5 3.4-6.4 8.8M-4-1.3-10.3-3.3"/></svg>';

function ldArt(kind, {logo = '', cards = 1, title = ''}) {
  if (kind === 'ball') return logo;
  if (kind === 'pitch') {
    return '<svg class="ld-pitch" viewBox="0 0 122 82"><rect class="ld-grass" x="1" y="1" width="120" height="80" rx="2"/>'
      + LD_PITCH.map((shape, i) => `${shape} class="ld-line" pathLength="1" style="--i:${LD_PITCH_DELAY[i]}"/>`).join('')
      + '<circle class="ld-spot" cx="61" cy="41" r="1"/><circle class="ld-spot" cx="15" cy="41" r=".9"/><circle class="ld-spot" cx="107" cy="41" r=".9"/></svg>';
  }
  if (kind === 'last10') {
    return `<span class="ld-chart">${LD_BARS.map((h, i) => `<i class="ld-bar${h > LD_LINE ? ' over' : ''}" style="--h:${h}%;--i:${i}"></i>`).join('')}`
      + `<span class="ld-mark" style="--y:${LD_LINE}%"></span></span>`;
  }
  if (kind === 'prices') {
    return `<span class="ld-board">${LD_PRICES.map(([side, price]) => `<span class="ld-cell"><small>${side}</small><span class="ld-num">`
      + [...price].map(ch => (ch === '.' ? '.' : `<i style="--d:${ch}"></i>`)).join('') + '</span></span>').join('')}</span>`;
  }
  if (kind === 'card') {
    const card = '<span class="ld-fixture"><span class="ld-row"><i class="ld-sk" style="--w:90px"></i><i class="ld-sk" style="--w:60px"></i></span>'
      + '<span class="ld-row"><i class="ld-sk ld-crest"></i><i class="ld-sk" style="--w:120px"></i></span>'
      + '<span class="ld-row"><i class="ld-sk ld-crest"></i><i class="ld-sk" style="--w:96px"></i></span>'
      + '<span class="ld-tiles"><i class="ld-sk"></i><i class="ld-sk"></i><i class="ld-sk"></i></span></span>';
    return card.repeat(Math.max(1, cards));
  }
  return `${title ? `<p class="ld-title">${ldEsc(title)}</p>` : ''}<span class="ld-track"><span class="ld-trail"></span><span class="ld-rollball">${LD_BALL}</span></span>`;
}

// One fixed line for screen readers; the steps (one or three) are for the eye. Any other count keeps the first.
function spixLoader(kind, {label, steps = [], logo, cards, title} = {}) {
  if (!SPIX_LOADER_KINDS.includes(kind)) throw new Error(`Unknown loader: ${kind}`);
  const words = steps.length === 3 ? steps : [steps[0] || label];
  return `<div class="ld ld-${kind}" role="status"><span class="ld-sr">${ldEsc(label)}</span>`
    + `<div class="ld-art" aria-hidden="true">${ldArt(kind, {logo, cards, title})}</div>`
    + `<p class="ld-steps" aria-hidden="true" style="--n:${words.length}">${words.map(w => `<span>${ldEsc(w)}…</span>`).join('')}</p></div>`;
}

const LD_CLOCK = {now: () => performance.now(), setTimeout: (fn, ms) => setTimeout(fn, ms), clearTimeout: t => clearTimeout(t)};

// The first-load cover (A): drawn only if the load is still running at 300 ms, held at least 600 ms once drawn,
// cleared when the load ends or fails, and given up at 15 s (the board's own loader carries on underneath).
function spixCover(load, {label, steps, logo, doc = document, clock = LD_CLOCK, delay = 300, hold = 600, giveUp = 15000}) {
  let el = null, shownAt = 0, done = false;
  const show = clock.setTimeout(() => {
    if (done) return;
    el = doc.createElement('div');
    el.className = 'ld-cover';
    el.innerHTML = spixLoader('ball', {label, steps, logo});
    doc.body.append(el);
    shownAt = clock.now();
  }, delay);
  const finish = () => {
    if (done) return;
    done = true;
    clock.clearTimeout(show);
    clock.clearTimeout(limit);
    if (!el) return;
    clock.setTimeout(() => {
      el.classList.add('ld-out');
      clock.setTimeout(() => el.remove(), 250);
    }, Math.max(0, hold - (clock.now() - shownAt)));
  };
  const limit = clock.setTimeout(finish, giveUp);
  Promise.resolve(load).then(finish, finish);
}
