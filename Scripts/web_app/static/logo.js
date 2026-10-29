// The logo's ball drops onto the i and bobbles when the user moves to another page (D48):
// never on a loop, never with reduced motion, never with the old logo (?logo=old).
(() => {
  const DROP_MS = 1500;
  const DROP_FALL = 'cubic-bezier(.55,0,1,.45)';
  const DROP_RISE = 'cubic-bezier(0,.55,.45,1)';
  const spixLogo = {
    // A full page load counts as a page change only when it came from another Spix page: not a first visit, not a reload.
    dropsOnLoad(referrer, origin, navigationType) {
      if (navigationType === 'reload') return false;
      try { return new URL(referrer).origin === origin; } catch (error) { return false; }
    },
    drop() {
      if (document.documentElement.dataset.logo === 'old') return false;
      if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return false;
      const svg = document.querySelector('.logo-word');
      const ball = svg?.querySelector('.spx-drop'), spin = svg?.querySelector('.spx-spin'), stem = svg?.querySelector('.spx-stem');
      if (!ball || !spin || !stem || typeof ball.animate !== 'function') return false;
      [ball, spin, stem].forEach(part => part.getAnimations?.().forEach(animation => animation.cancel()));
      ball.animate([
        { offset: 0, transform: 'translateY(-520%) scale(0.9, 1.12)', easing: DROP_FALL },
        { offset: 0.30, transform: 'translateY(0) scale(0.9, 1.12)' },
        { offset: 0.35, transform: 'translateY(0) scale(1.3, 0.7)' },
        { offset: 0.40, transform: 'translateY(0) scale(1, 1)', easing: DROP_RISE },
        { offset: 0.56, transform: 'translateY(-140%) scale(0.97, 1.03)', easing: DROP_FALL },
        { offset: 0.71, transform: 'translateY(0) scale(1, 1)' },
        { offset: 0.75, transform: 'translateY(0) scale(1.14, 0.86)' },
        { offset: 0.79, transform: 'translateY(0) scale(1, 1)', easing: DROP_RISE },
        { offset: 0.87, transform: 'translateY(-32%) scale(1, 1)', easing: DROP_FALL },
        { offset: 0.94, transform: 'translateY(0) scale(1, 1)' },
        { offset: 0.97, transform: 'translateY(0) scale(1.05, 0.95)' },
        { offset: 1, transform: 'translateY(0) scale(1, 1)' },
      ], { duration: DROP_MS, fill: 'backwards' });
      spin.animate([
        { offset: 0, transform: 'rotate(-220deg)' },
        { offset: 0.30, transform: 'rotate(-60deg)' },
        { offset: 0.71, transform: 'rotate(-12deg)' },
        { offset: 1, transform: 'rotate(0deg)' },
      ], { duration: DROP_MS, easing: 'ease-out', fill: 'backwards' });
      stem.animate([
        { offset: 0, transform: 'scale(1, 1)' },
        { offset: 0.30, transform: 'scale(1, 1)' },
        { offset: 0.35, transform: 'scale(1.06, 0.88)' },
        { offset: 0.45, transform: 'scale(1, 1)' },
        { offset: 0.71, transform: 'scale(1, 1)' },
        { offset: 0.75, transform: 'scale(1.02, 0.96)' },
        { offset: 0.80, transform: 'scale(1, 1)' },
        { offset: 1, transform: 'scale(1, 1)' },
      ], { duration: DROP_MS });
      return true;
    },
  };
  window.spixLogo = spixLogo;

  const navigation = performance.getEntriesByType?.('navigation')[0];
  if (spixLogo.dropsOnLoad(document.referrer, location.origin, navigation?.type)) spixLogo.drop();
})();
