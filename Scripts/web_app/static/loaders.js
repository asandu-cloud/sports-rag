/* The six loading screens (D112, owner 16 Nov). Each loader is HTML from spixLoader(kind, …); loaders.css moves it:
   it shows only after 300 ms, its words step in CSS, and it rests on a still frame with reduced motion. Screen
   readers hear one fixed line. The Research Area's shadow root calls the same functions through window.
     ball    A  the logo's football drops onto the i (the first-load cover, spixCover)
     pitch   B  the pitch markings draw themselves (the Research Area's match view)
     last10  C  ten columns grow and the line drops in (a Research Area profile)
     prices  D  three prices roll and settle (Best Bets)
     card    E  the board card's outline with a sweep (the matchday board)
     roll    F  the logo's football rolls through a tube, cobalt filling in behind it (the Research Area) */
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
// The header logo's football (D48), as the logo draws it: the patches turn inside the clip, the shade and highlight
// stay put. Its own clip id, so it never points at the header's.
const LD_BALL = '<svg viewBox="-17.2 -17.2 34.4 34.4"><clipPath id="ld-ball-clip"><circle r="16.36"/></clipPath><circle r="16.36" fill="#FAFDFA"/><g clip-path="url(#ld-ball-clip)"><g class="ld-spin" stroke="#111E17" stroke-linejoin="round" stroke-linecap="round"><g stroke-width="0.49"><path d="M-2.97 1.39 L-2.35 0.88 L-1.71 0.37 L-1.06 -0.16 L-0.4 -0.69 L0.25 -1.21 L0.91 -1.73 L1.55 -2.24 L2.18 -2.74 L1.93 -3.51 L1.67 -4.28 L1.41 -5.04 L1.14 -5.8 L0.87 -6.55 L0.59 -7.27 L0.32 -7.97 L0.05 -8.63 L-0.76 -8.65 L-1.59 -8.64 L-2.42 -8.61 L-3.25 -8.56 L-4.07 -8.49 L-4.87 -8.39 L-5.66 -8.28 L-6.41 -8.14 L-6.71 -7.44 L-6.99 -6.7 L-7.26 -5.93 L-7.5 -5.15 L-7.73 -4.36 L-7.94 -3.55 L-8.12 -2.75 L-8.28 -1.95 L-7.69 -1.54 L-7.06 -1.13 L-6.41 -0.71 L-5.75 -0.28 L-5.06 0.14 L-4.37 0.57 L-3.67 0.98Z" fill="#111E17"/><path d="M3.89 -12.34 L4.68 -12.18 L5.46 -11.98 L6.25 -11.75 L7.02 -11.49 L7.77 -11.19 L8.49 -10.87 L9.19 -10.53 L9.85 -10.16 L10.08 -10.49 L10.28 -10.79 L10.45 -11.08 L10.6 -11.33 L10.72 -11.55 L10.81 -11.75 L10.88 -11.91 L10.91 -12.03 L10.34 -12.56 L9.73 -13.07 L9.09 -13.55 L8.43 -13.99 L7.74 -14.4 L7.03 -14.77 L6.32 -15.09 L5.6 -15.37 L5.1 -15.54 L4.58 -15.66 L4.05 -15.75 L3.5 -15.79 L2.94 -15.8 L2.38 -15.76 L1.82 -15.68 L1.26 -15.57 L1.6 -15.3 L1.95 -14.99 L2.29 -14.64 L2.63 -14.25 L2.96 -13.82 L3.28 -13.35 L3.59 -12.86Z" fill="#111E17"/><path d="M-13.54 -8.39 L-13.1 -8.84 L-12.61 -9.28 L-12.09 -9.7 L-11.53 -10.09 L-10.94 -10.45 L-10.33 -10.79 L-9.69 -11.09 L-9.05 -11.37 L-8.64 -11.94 L-8.21 -12.49 L-7.76 -13.01 L-7.28 -13.5 L-6.78 -13.95 L-6.26 -14.37 L-5.74 -14.75 L-5.21 -15.08 L-5.52 -15.13 L-5.83 -15.14 L-6.12 -15.11 L-6.4 -15.05 L-6.67 -14.94 L-6.92 -14.83 L-7.18 -14.7 L-7.43 -14.58 L-8.15 -14.19 L-8.87 -13.75 L-9.57 -13.27 L-10.25 -12.75 L-10.9 -12.2 L-11.52 -11.62 L-12.1 -11.02 L-12.64 -10.39 L-12.81 -10.18 L-12.97 -9.97 L-13.14 -9.75 L-13.29 -9.53 L-13.41 -9.28 L-13.49 -9 L-13.53 -8.71Z" fill="#111E17"/><path d="M14.22 -8.09 L14.6 -7.4 L14.94 -6.68 L15.25 -5.94 L15.52 -5.19 L15.75 -4.42 L15.95 -3.65 L16.11 -2.88 L16.22 -2.12 L16.28 -1.6 L16.33 -1.06 L16.35 -0.52 L16.36 0.04 L16.35 0.6 L16.32 1.17 L16.27 1.74 L16.2 2.31 L16.25 1.93 L16.29 1.54 L16.32 1.12 L16.35 0.68 L16.36 0.22 L16.36 -0.27 L16.34 -0.77 L16.31 -1.31 L16.2 -2.32 L16.02 -3.31 L15.79 -4.29 L15.5 -5.24 L15.17 -6.14 L14.79 -7 L14.38 -7.8 L13.95 -8.55 L13.99 -8.49 L14.03 -8.42 L14.06 -8.36 L14.1 -8.3 L14.13 -8.25 L14.16 -8.19 L14.19 -8.14Z" fill="#111E17"/><path d="M3.81 9.89 L3.09 9.7 L2.35 9.49 L1.6 9.25 L0.84 8.98 L0.08 8.69 L-0.68 8.38 L-1.42 8.05 L-2.16 7.71 L-2.74 8.15 L-3.33 8.58 L-3.92 8.99 L-4.5 9.38 L-5.07 9.75 L-5.62 10.09 L-6.15 10.4 L-6.66 10.68 L-6.31 11.28 L-5.95 11.86 L-5.57 12.42 L-5.17 12.95 L-4.76 13.45 L-4.33 13.91 L-3.9 14.32 L-3.47 14.7 L-2.69 14.77 L-1.88 14.8 L-1.07 14.8 L-0.24 14.76 L0.58 14.67 L1.4 14.55 L2.21 14.4 L3 14.21 L3.13 13.79 L3.25 13.33 L3.37 12.84 L3.47 12.3 L3.57 11.73 L3.66 11.14 L3.74 10.52Z" fill="#111E17"/><path d="M-11.97 7.34 L-12.18 6.61 L-12.36 5.85 L-12.51 5.07 L-12.63 4.27 L-12.72 3.46 L-12.77 2.64 L-12.79 1.83 L-12.78 1.02 L-13.23 0.63 L-13.65 0.22 L-14.04 -0.19 L-14.39 -0.6 L-14.71 -1.01 L-14.99 -1.42 L-15.22 -1.81 L-15.41 -2.2 L-15.65 -1.67 L-15.86 -1.14 L-16.03 -0.59 L-16.15 -0.04 L-16.23 0.51 L-16.27 1.06 L-16.27 1.6 L-16.22 2.12 L-16.1 2.88 L-15.94 3.65 L-15.73 4.42 L-15.48 5.17 L-15.19 5.92 L-14.86 6.64 L-14.49 7.34 L-14.1 8.02 L-13.95 8 L-13.78 7.97 L-13.56 7.91 L-13.31 7.84 L-13.02 7.74 L-12.7 7.62 L-12.34 7.49Z" fill="#111E17"/><path d="M11.98 -4.27 L11.61 -3.84 L11.2 -3.39 L10.75 -2.93 L10.28 -2.46 L9.77 -1.99 L9.25 -1.51 L8.7 -1.03 L8.15 -0.56 L8.32 0.24 L8.48 1.04 L8.62 1.85 L8.73 2.66 L8.82 3.46 L8.89 4.24 L8.94 5.01 L8.96 5.76 L9.59 5.83 L10.2 5.9 L10.79 5.94 L11.36 5.98 L11.9 5.99 L12.41 5.99 L12.87 5.98 L13.3 5.95 L13.65 5.22 L13.98 4.47 L14.27 3.7 L14.53 2.91 L14.75 2.12 L14.93 1.32 L15.07 0.53 L15.17 -0.25 L14.9 -0.76 L14.6 -1.27 L14.25 -1.79 L13.86 -2.3 L13.44 -2.81 L12.98 -3.31 L12.49 -3.8Z" fill="#111E17"/><path d="M-1.32 16.31 L-1.88 16.25 L-2.44 16.18 L-3 16.09 L-3.54 15.97 L-4.07 15.85 L-4.6 15.7 L-5.1 15.55 L-5.6 15.37 L-6.32 15.09 L-7.04 14.77 L-7.75 14.41 L-8.44 14.02 L-9.12 13.59 L-9.77 13.12 L-10.4 12.63 L-10.99 12.12 L-11.04 12.08 L-11.08 12.04 L-11.13 11.99 L-11.18 11.95 L-11.23 11.9 L-11.28 11.85 L-11.34 11.8 L-11.39 11.75 L-10.76 12.33 L-10.07 12.89 L-9.33 13.44 L-8.53 13.96 L-7.68 14.45 L-6.79 14.89 L-5.87 15.28 L-4.92 15.61 L-4.4 15.76 L-3.91 15.89 L-3.43 16 L-2.97 16.09 L-2.53 16.17 L-2.11 16.23 L-1.7 16.27Z" fill="#111E17"/><path d="M12.49 10.27 L11.95 10.88 L11.37 11.48 L10.76 12.05 L10.12 12.59 L9.45 13.1 L8.76 13.57 L8.05 14.01 L7.34 14.4 L7.13 14.61 L6.91 14.8 L6.67 14.94 L6.41 15.06 L6.14 15.16 L5.88 15.27 L5.61 15.37 L5.34 15.47 L5.94 15.25 L6.54 15 L7.15 14.72 L7.76 14.4 L8.38 14.05 L8.99 13.67 L9.6 13.25 L10.19 12.8 L10.77 12.32 L11.32 11.82 L11.83 11.3 L12.32 10.77 L12.76 10.24 L13.18 9.7 L13.56 9.16 L13.91 8.62 L13.76 8.85 L13.61 9.08 L13.46 9.31 L13.3 9.53 L13.14 9.75 L12.95 9.95 L12.73 10.12Z" fill="#111E17"/></g><path d="M0.05 -8.63 L0.54 -9.18 L1.03 -9.71 L1.52 -10.22 L2.01 -10.71 L2.5 -11.17 L2.98 -11.59 L3.44 -11.99 L3.89 -12.34 M-6.41 -8.14 L-6.8 -8.62 L-7.18 -9.09 L-7.55 -9.54 L-7.89 -9.96 L-8.22 -10.36 L-8.52 -10.73 L-8.79 -11.06 L-9.05 -11.37 M-2.97 1.39 L-2.9 2.2 L-2.81 3.02 L-2.72 3.83 L-2.62 4.64 L-2.51 5.44 L-2.4 6.22 L-2.28 6.98 L-2.16 7.71 M-8.28 -1.95 L-8.93 -1.59 L-9.55 -1.22 L-10.17 -0.85 L-10.75 -0.47 L-11.31 -0.09 L-11.84 0.29 L-12.33 0.66 L-12.78 1.02 M2.18 -2.74 L2.95 -2.49 L3.73 -2.23 L4.5 -1.96 L5.27 -1.68 L6.02 -1.4 L6.76 -1.12 L7.47 -0.84 L8.15 -0.56 M1.26 -15.57 L0.45 -15.64 L-0.36 -15.68 L-1.19 -15.68 L-2.02 -15.64 L-2.84 -15.56 L-3.65 -15.44 L-4.44 -15.27 L-5.21 -15.08 M9.85 -10.16 L10.21 -9.51 L10.55 -8.82 L10.86 -8.11 L11.15 -7.36 L11.4 -6.6 L11.63 -5.83 L11.82 -5.05 L11.98 -4.27 M-13.54 -8.39 L-13.9 -7.69 L-14.23 -6.95 L-14.52 -6.19 L-14.78 -5.41 L-15 -4.61 L-15.18 -3.81 L-15.32 -3 L-15.41 -2.2 M-6.66 10.68 L-7.39 10.35 L-8.11 10 L-8.82 9.61 L-9.51 9.2 L-10.17 8.76 L-10.81 8.3 L-11.41 7.82 L-11.97 7.34 M3.81 9.89 L4.49 9.46 L5.18 8.99 L5.85 8.5 L6.52 7.99 L7.17 7.45 L7.79 6.9 L8.39 6.33 L8.96 5.76 M3 14.21 L3.57 14.36 L4.14 14.48 L4.71 14.56 L5.27 14.6 L5.82 14.61 L6.35 14.58 L6.85 14.5 L7.34 14.4 M13.3 5.95 L13.32 6.55 L13.3 7.14 L13.25 7.72 L13.16 8.28 L13.04 8.82 L12.89 9.33 L12.7 9.82 L12.49 10.27" fill="none" stroke-width="0.74"/></g><path d="M-16.36 0A16.36 16.36 0 1 0 16.36 0A16.36 16.36 0 1 0 -16.36 0ZM-20.62 -4.25A17.02 17.02 0 1 1 13.42 -4.25A17.02 17.02 0 1 1 -20.62 -4.25Z" fill-rule="evenodd" fill="#111E17" fill-opacity="0.16"/><ellipse cx="-6.87" cy="-8.18" rx="3.27" ry="1.64" transform="rotate(-40 -6.87 -8.18)" fill="#FFFFFF" fill-opacity="0.85"/></g><circle r="15.95" fill="none" stroke="#111E17" stroke-width="0.82"/></svg>';

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
  return `${title ? `<p class="ld-title">${ldEsc(title)}</p>` : ''}<span class="ld-tube"><span class="ld-fill"></span><span class="ld-rollball">${LD_BALL}</span></span>`;
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
