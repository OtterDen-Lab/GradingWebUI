// Persistent AI model defaults. The API keeps the provider contract independent
// of this UI so future providers can use the same settings endpoints.
let aiModelSettings = null;

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('ai-settings-btn')?.addEventListener('click', async () => {
    navigateToSection('ai-settings-section');
    await loadAIModelSettings();
  });
  document.getElementById('cancel-ai-settings')?.addEventListener('click', () =>
    navigateToSection('session-section'));
  document.getElementById('save-my-ai-settings')?.addEventListener('click', () => saveAIModelSettings('me'));
  document.getElementById('save-system-ai-settings')?.addEventListener('click', () => saveAIModelSettings('system'));
});

async function loadAIModelSettings() {
  const error = document.getElementById('ai-settings-error');
  error.style.display = 'none';
  try {
    const response = await fetch(`${API_BASE}/ai-settings`, {credentials: 'include'});
    if (!response.ok) throw new Error((await response.json()).detail || 'Could not load settings');
    aiModelSettings = await response.json();
    let modelOptions = [];
    // Model discovery is intentionally live rather than checked into the UI;
    // a provider deprecation therefore needs no deploy to update this list.
    const modelsResponse = await fetch(`${API_BASE}/ai-settings/models/anthropic`, {credentials: 'include'});
    if (modelsResponse.ok) modelOptions = (await modelsResponse.json()).models;
    const form = document.getElementById('ai-settings-form');
    form.innerHTML = `<datalist id="anthropic-model-options">${modelOptions.map(model =>
      `<option value="${escapeAI(model.id)}">${escapeAI(model.display_name)}</option>`).join('')}</datalist>` +
      ['small', 'medium', 'large'].map(tier => {
      const setting = aiModelSettings.settings[tier];
      const personal = setting.source === 'user' ? setting.model_id : '';
      return `<label style="display:block; margin:12px 0; max-width:620px;">
        <strong>${tier[0].toUpperCase() + tier.slice(1)}</strong>
        <small style="display:block;color:var(--gray-700)">Effective: ${escapeAI(setting.model_id)} (${setting.source})</small>
        <input data-ai-tier="${tier}" list="anthropic-model-options" value="${escapeAI(personal)}" placeholder="Use system default" style="width:100%; padding:8px; box-sizing:border-box;">
      </label>`;
    }).join('');
    if (currentUser?.role === 'instructor') await loadOllamaSettings();
  } catch (err) {
    error.textContent = err.message;
    error.style.display = 'block';
  }
}

async function loadOllamaSettings() {
  const container = document.getElementById('ollama-settings-form');
  const response = await fetch(`${API_BASE}/ai-settings/ollama/servers`, {credentials: 'include'});
  if (!response.ok) return;
  const servers = (await response.json()).servers;
  container.innerHTML = `
    <h3>Optional Ollama servers</h3>
    <p style="color:var(--gray-700)">No server or models are bundled. Add a server you operate, then choose one installed model or request a pull by exact tag.</p>
    <label style="display:block; margin:8px 0">Server name <input id="ollama-server-name" placeholder="GPU server" style="margin-left:8px"></label>
    <label style="display:block; margin:8px 0">Server URL <input id="ollama-server-url" placeholder="https://ollama.p.zxqw.dev" style="width:360px; max-width:100%; margin-left:8px"></label>
    <button id="save-ollama-server" class="btn btn-secondary">Add Ollama Server</button>
    <div style="margin-top:16px">${servers.length ? `
      <label>Configured server <select id="ollama-server-select">${servers.map(server =>
        `<option value="${server.id}" data-active-model="${escapeAI(server.active_model)}">${escapeAI(server.name)} — ${escapeAI(server.base_url)}</option>`).join('')}</select></label>
      <button id="refresh-ollama-models" class="btn btn-secondary">Refresh models</button>
      <div id="ollama-model-controls" style="margin-top:12px"></div>` :
      '<small style="color:var(--gray-700)">No Ollama servers configured.</small>'}</div>`;
  document.getElementById('save-ollama-server').onclick = saveOllamaServer;
  if (servers.length) {
    document.getElementById('ollama-server-select').onchange = renderOllamaModels;
    document.getElementById('refresh-ollama-models').onclick = renderOllamaModels;
    await renderOllamaModels();
  }
}

async function saveOllamaServer() {
  const name = document.getElementById('ollama-server-name').value.trim();
  const base_url = document.getElementById('ollama-server-url').value.trim();
  const response = await fetch(`${API_BASE}/ai-settings/ollama/servers`, {
    method: 'POST', credentials: 'include', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name, base_url})
  });
  if (!response.ok) return alert((await response.json()).detail || 'Could not add Ollama server');
  await loadOllamaSettings();
}

async function renderOllamaModels() {
  const serverId = document.getElementById('ollama-server-select').value;
  const controls = document.getElementById('ollama-model-controls');
  controls.textContent = 'Loading installed models…';
  const response = await fetch(`${API_BASE}/ai-settings/ollama/servers/${serverId}/models`, {credentials: 'include'});
  if (!response.ok) {
    controls.textContent = (await response.json()).detail || 'Could not reach Ollama server.';
    return;
  }
  const models = (await response.json()).models;
  const selectedServer = document.getElementById('ollama-server-select').selectedOptions[0];
  const activeModel = selectedServer.dataset.activeModel || '';
  controls.innerHTML = `
    <label>Active model <select id="ollama-active-model"><option value="">None selected</option>${models.map(model =>
      `<option value="${escapeAI(model.id)}">${escapeAI(model.display_name)}</option>`).join('')}</select></label>
    <button id="save-ollama-model" class="btn btn-secondary">Use this model</button>
    <div style="margin-top:10px"><input id="ollama-pull-model" placeholder="e.g. qwen3-vl:30b" style="width:240px; max-width:100%">
      <button id="pull-ollama-model" class="btn btn-secondary">Download model</button></div>`;
  document.getElementById('ollama-active-model').value = activeModel;
  document.getElementById('save-ollama-model').onclick = async () => {
    const model = document.getElementById('ollama-active-model').value;
    const save = await fetch(`${API_BASE}/ai-settings/ollama/servers/${serverId}/active-model`, {
      method: 'PUT', credentials: 'include', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({model})
    });
    if (!save.ok) alert((await save.json()).detail || 'Could not save active model');
    else selectedServer.dataset.activeModel = model;
  };
  document.getElementById('pull-ollama-model').onclick = async () => {
    const model = document.getElementById('ollama-pull-model').value.trim();
    const pull = await fetch(`${API_BASE}/ai-settings/ollama/servers/${serverId}/pull`, {
      method: 'POST', credentials: 'include', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({model})
    });
    if (!pull.ok) return alert((await pull.json()).detail || 'Could not start model download');
    alert('Download requested. Refresh models after Ollama finishes pulling it.');
  };
}

function escapeAI(value) {
  const element = document.createElement('div');
  element.textContent = value || '';
  return element.innerHTML;
}

async function saveAIModelSettings(scope) {
  const models = {};
  document.querySelectorAll('[data-ai-tier]').forEach(input => {
    const tier = input.dataset.aiTier;
    models[tier] = input.value.trim() ||
      (scope === 'system' ? aiModelSettings.settings[tier].model_id : null);
  });
  const response = await fetch(`${API_BASE}/ai-settings/${scope}`, {
    method: 'PUT', credentials: 'include', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({provider: 'anthropic', models})
  });
  if (!response.ok) {
    const error = await response.json();
    alert(error.detail || 'Could not save settings');
    return;
  }
  await loadAIModelSettings();
}
