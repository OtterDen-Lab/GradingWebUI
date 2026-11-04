let aiModelSettings = null;

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('ai-settings-btn')?.addEventListener('click', async () => {
    navigateToSection('ai-settings-section');
    await loadAIModelSettings();
  });
  document.getElementById('cancel-ai-settings')?.addEventListener('click', () =>
    navigateToSection('session-section'));
  document.getElementById('save-my-ai-settings')?.addEventListener('click', () =>
    saveAIModelSettings('me'));
  document.getElementById('save-system-ai-settings')?.addEventListener('click', () =>
    saveAIModelSettings('system'));
});

function escapeAI(value) {
  const element = document.createElement('div');
  element.textContent = value || '';
  return element.innerHTML;
}

async function loadAIModelSettings() {
  const error = document.getElementById('ai-settings-error');
  error.style.display = 'none';
  try {
    const [settingsResponse, modelsResponse] = await Promise.all([
      fetch(`${API_BASE}/ai-settings`, {credentials: 'include'}),
      fetch(`${API_BASE}/ai-settings/models`, {credentials: 'include'}),
    ]);
    if (!settingsResponse.ok) {
      throw new Error((await settingsResponse.json()).detail || 'Could not load settings');
    }
    aiModelSettings = await settingsResponse.json();
    if (!modelsResponse.ok) {
      throw new Error((await modelsResponse.json()).detail || 'Could not load Bifrost models');
    }
    const models = (await modelsResponse.json()).models;
    const selected = aiModelSettings.default_model.model_id || '';
    if (selected && !models.some(model => model.id === selected)) {
      models.unshift({id: selected, display_name: `${selected} (currently selected)`});
    }
    document.getElementById('ai-settings-form').innerHTML = `
      <label style="display:block;max-width:620px;">
        <strong>Default model</strong>
        <small style="display:block;color:var(--gray-700);margin:4px 0 8px">
          Used for handwriting transcription, scanning, and autograding. Current source: ${escapeAI(aiModelSettings.default_model.source)}.
        </small>
        <select id="default-ai-model" style="width:100%;padding:8px;box-sizing:border-box;">
          <option value="">Use system default</option>
          ${models.map(model => `<option value="${escapeAI(model.id)}"${model.id === selected ? ' selected' : ''}>${escapeAI(model.display_name)}</option>`).join('')}
        </select>
      </label>`;
  } catch (err) {
    error.textContent = err.message;
    error.style.display = 'block';
  }
}

async function saveAIModelSettings(scope) {
  const model_id = document.getElementById('default-ai-model').value;
  if (scope === 'system' && !model_id) {
    alert('Choose a system default model.');
    return;
  }
  const response = await fetch(`${API_BASE}/ai-settings/${scope}`, {
    method: 'PUT', credentials: 'include', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({model_id})
  });
  if (!response.ok) {
    const error = await response.json();
    alert(error.detail || 'Could not save default model');
    return;
  }
  if (typeof transcriptionModelOptions !== 'undefined') transcriptionModelOptions = null;
  if (typeof transcriptionCache !== 'undefined') {
    Object.values(transcriptionCache).forEach(cache => delete cache.default);
  }
  await loadAIModelSettings();
}
