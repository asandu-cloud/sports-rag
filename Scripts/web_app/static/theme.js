/* Light/dark theme preference.
   Every visitor starts in the light Chalk theme. Choosing dark with the toggle
   is stored per browser and applied before first paint by the inline snippet
   in each page's <head>. */
(function () {
  const KEY = 'spix-theme';
  const root = document.documentElement;

  const stored = () => {
    try {
      const value = localStorage.getItem(KEY);
      return value === 'light' || value === 'dark' ? value : null;
    } catch (error) {
      return null;
    }
  };
  const effective = () => root.dataset.theme || 'light';

  const CHROME = { light: '#F3F7F3', dark: '#0D1511' };

  function syncButtons() {
    const current = effective();
    const next = current === 'dark' ? 'light' : 'dark';
    document.querySelectorAll('meta[name="theme-color"]').forEach(meta => {
      meta.setAttribute('content', CHROME[current]);
    });
    document.querySelectorAll('[data-theme-toggle]').forEach(button => {
      button.setAttribute('aria-label', `Switch to ${next} theme`);
      button.title = `Switch to ${next} theme`;
    });
  }

  function choose(theme) {
    root.dataset.theme = theme;
    try { localStorage.setItem(KEY, theme); } catch (error) { /* storage unavailable: applies to this page only */ }
    syncButtons();
  }

  const initial = stored();
  if (initial) root.dataset.theme = initial;

  document.addEventListener('click', event => {
    const button = event.target.closest('[data-theme-toggle]');
    if (button) choose(effective() === 'dark' ? 'light' : 'dark');
  });
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', syncButtons);
  else syncButtons();
})();
