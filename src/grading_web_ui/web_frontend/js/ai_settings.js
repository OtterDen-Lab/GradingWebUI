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
  } catch (err) {
    error.textContent = err.message;
    error.style.display = 'block';
  }
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
