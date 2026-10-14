import { hasEffect } from '../effects/registry.ts';
import { lazyMount } from './lazy.ts';
import { hrefFor, onQueryChange, readQuery, startRouter } from './router.ts';
import type { ShellContext, Unmount } from './types.ts';
import './theme.css';
import './app.css';

type Theme = 'auto' | 'light' | 'dark';
const THEMES: readonly Theme[] = ['auto', 'light', 'dark'];
const THEME_KEY = 'wt-theme';

function storedTheme(): Theme {
  try {
    const value = localStorage.getItem(THEME_KEY);
    return value === 'light' || value === 'dark' ? value : 'auto';
  } catch {
    return 'auto';
  }
}

function applyTheme(theme: Theme) {
  if (theme === 'auto') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = theme;
  try {
    if (theme === 'auto') localStorage.removeItem(THEME_KEY);
    else localStorage.setItem(THEME_KEY, theme);
  } catch {
    // Storage can be blocked; the choice then lasts for this page view only.
  }
}

function themeToggle(): HTMLButtonElement {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'wt-button wt-button--quiet';
  let theme = storedTheme();
  const render = () => {
    button.textContent = `Theme: ${theme}`;
    button.setAttribute('aria-label', `Colour theme: ${theme}. Activate to change.`);
  };
  button.addEventListener('click', () => {
    theme = THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length]!;
    applyTheme(theme);
    render();
  });
  render();
  return button;
}

export function mountApp(root: HTMLElement, shell: ShellContext): void {
  const header = document.createElement('header');
  header.className = 'wt-app__header';
  const home = document.createElement('a');
  home.className = 'wt-app__home';
  home.href = hrefFor({}, '');
  home.dataset.link = '';
  home.textContent = 'Water Temple';
  const tagline = document.createElement('span');
  tagline.className = 'wt-app__tagline wt-muted';
  tagline.textContent = 'WebGPU liquid effects, measured';
  const spacer = document.createElement('span');
  spacer.className = 'wt-app__spacer';
  header.append(home, tagline, spacer, themeToggle());

  const outlet = document.createElement('main');
  outlet.className = 'wt-app__outlet';
  root.append(header, outlet);

  let page: Unmount | undefined;
  let shown: string | null | undefined;
  let galleryScroll = 0;

  function show(query: URLSearchParams) {
    const requested = query.get('effect');
    const effect = requested && hasEffect(requested) ? requested : null;
    if (effect === shown) return;
    if (shown === null) galleryScroll = scrollY;
    shown = effect;
    page?.();
    outlet.replaceChildren();
    scrollTo(0, 0);
    page = effect
      ? lazyMount(outlet, 'effect page', async () =>
          (await import('./effect-page.ts')).mountEffectPage(outlet, shell, effect),
        )
      : lazyMount(outlet, 'gallery', async () => {
          const unmount = (await import('./gallery.ts')).mountGallery(outlet, shell);
          // Coming back from an effect, land where the reader left the grid.
          scrollTo(0, galleryScroll);
          return unmount;
        });
  }

  startRouter();
  onQueryChange(show);
  show(readQuery());
}
