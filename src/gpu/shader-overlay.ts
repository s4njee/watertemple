import {
  diagnosticExcerpt,
  getDiagnostics,
  groupDiagnosticsByFile,
  onDiagnosticsChanged,
  type ShaderDiagnostic,
} from './shader.ts';

const CLASS = 'wt-shader-overlay';

let uninstallCurrent: (() => void) | undefined;

function element<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/** Identifies what is on display, so a dismissed panel stays hidden until something changes. */
function signature(diagnostics: readonly ShaderDiagnostic[]): string {
  return diagnostics
    .map((d) => `${d.severity}|${d.shader}|${d.file}|${d.line}|${d.column}|${d.message}`)
    .join('\n');
}

function count(diagnostics: readonly ShaderDiagnostic[], severity: ShaderDiagnostic['severity']) {
  return diagnostics.filter((d) => d.severity === severity).length;
}

function title(diagnostics: readonly ShaderDiagnostic[]): string {
  const errors = count(diagnostics, 'error');
  const warnings = count(diagnostics, 'warning');
  const parts = [];
  if (errors) parts.push(`${errors} shader error${errors === 1 ? '' : 's'}`);
  if (warnings) parts.push(`${warnings} warning${warnings === 1 ? '' : 's'}`);
  return parts.join(', ') || 'Shader notes';
}

function renderItem(diagnostic: ShaderDiagnostic): HTMLElement {
  const item = element('div', `${CLASS}__item ${CLASS}__item--${diagnostic.severity}`);
  const head = element('div', `${CLASS}__message`);
  if (diagnostic.line > 0) {
    head.append(element('span', `${CLASS}__loc`, `${diagnostic.line}:${diagnostic.column}`));
  }
  head.append(
    element(
      'span',
      `${CLASS}__severity`,
      diagnostic.severity === 'info' ? 'note' : diagnostic.severity,
    ),
    element('span', `${CLASS}__text`, diagnostic.message),
  );
  item.append(head);
  if (diagnostic.file !== diagnostic.shader) {
    item.append(element('div', `${CLASS}__via`, `while compiling ${diagnostic.shader}`));
  }
  const excerpt = diagnosticExcerpt(diagnostic);
  if (excerpt) item.append(element('pre', `${CLASS}__source`, excerpt));
  return item;
}

function render(root: HTMLElement, diagnostics: readonly ShaderDiagnostic[], dismiss: () => void) {
  const header = element('div', `${CLASS}__header`);
  const close = element('button', `${CLASS}__close`, '×');
  close.type = 'button';
  close.title = 'Dismiss';
  close.setAttribute('aria-label', 'Dismiss shader diagnostics');
  close.addEventListener('click', dismiss);
  header.append(element('span', `${CLASS}__title`, title(diagnostics)), close);

  const body = element('div', `${CLASS}__body`);
  for (const group of groupDiagnosticsByFile(diagnostics)) {
    const section = element('section', `${CLASS}__file`);
    section.append(element('h2', `${CLASS}__filename`, group.file));
    section.append(...group.diagnostics.map(renderItem));
    body.append(section);
  }
  root.replaceChildren(header, body);
}

/**
 * Shows current shader diagnostics in a dismissible panel pinned to the corner of the page, and
 * removes it again once they are gone. The host decides when to call this (development only).
 * Returns an uninstall function. Installing twice returns the existing panel's uninstall.
 */
export function installShaderOverlay(): () => void {
  if (typeof document === 'undefined') return () => {};
  if (uninstallCurrent) return uninstallCurrent;

  const root = element('aside', CLASS);
  root.setAttribute('role', 'region');
  root.setAttribute('aria-label', 'Shader diagnostics');
  let dismissed: string | undefined;

  const update = () => {
    const diagnostics = getDiagnostics();
    const current = signature(diagnostics);
    if (diagnostics.length === 0 || current === dismissed) {
      root.remove();
      if (diagnostics.length === 0) dismissed = undefined;
      return;
    }
    render(root, diagnostics, () => {
      dismissed = current;
      root.remove();
    });
    if (!root.isConnected) (document.body ?? document.documentElement).append(root);
  };

  const unsubscribe = onDiagnosticsChanged(update);
  update();

  const uninstall = () => {
    unsubscribe();
    root.remove();
    if (uninstallCurrent === uninstall) uninstallCurrent = undefined;
  };
  uninstallCurrent = uninstall;
  return uninstall;
}
