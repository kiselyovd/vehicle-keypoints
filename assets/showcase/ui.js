const debug = new URLSearchParams(location.search).get('debug') === '1';

export function log(event, details = {}) {
  if (debug) console.debug('[showcase]', event, details);
}

export function element(tag, className = '', text = '') {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

export function selectField(label, options, name) {
  const wrapper = element('label', 'sc-field');
  wrapper.append(element('span', '', label));
  const select = element('select');
  select.name = name;
  select.setAttribute('aria-label', label);
  for (const [value, title] of options) {
    const option = element('option', '', title);
    option.value = String(value);
    select.append(option);
  }
  wrapper.append(select);
  return { wrapper, select };
}

export function checkboxField(label, name, checked = true) {
  const wrapper = element('label', 'sc-check');
  const input = element('input');
  input.type = 'checkbox';
  input.name = name;
  input.checked = checked;
  wrapper.append(input, document.createTextNode(label));
  return { wrapper, input };
}

export function assetUrl(path, base) {
  return new URL(path, base).href;
}

export function startShowcase(kind, render, dataUrl) {
  const initialize = async () => {
    const root = document.querySelector(`[data-showcase="${kind}"]`);
    if (!root || root.dataset.ready) return;
    root.dataset.ready = 'loading';
    try {
      log('load:start', { kind });
      const response = await fetch(dataUrl);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      if (data.kind !== kind || data.owner !== 'kiselyovd') throw new Error('Unexpected source');
      await render(root, data, new URL('.', dataUrl));
      root.dataset.ready = 'true';
      log('load:complete', { kind, source: data.source_sha });
    } catch (error) {
      root.dataset.ready = 'error';
      root.querySelector('[role="status"]').textContent = 'The explorer could not load. The source illustration and documentation remain available below.';
      console.warn('[showcase] load failed', { kind, message: error.message });
    }
  };
  if (globalThis.document$?.subscribe) globalThis.document$.subscribe(initialize);
  else if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initialize, { once: true });
  else initialize();
}

export function sourceLink(url, label = 'Inspect the source') {
  const link = element('a', 'sc-source', label);
  link.href = url;
  return link;
}
