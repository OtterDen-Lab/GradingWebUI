// Per-user Canvas API credential settings.
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('canvas-settings-back')?.addEventListener('click', () =>
    navigateToSection('session-section'));
});

function canvasEnvironmentLabel(environment) {
  return environment === 'production' ? 'Production' : 'Development / Beta';
}

async function loadCanvasCredentialSettings() {
  const form = document.getElementById('canvas-settings-form');
  const error = document.getElementById('canvas-settings-error');
  if (!form || !error) return;
  error.style.display = 'none';
  form.textContent = 'Loading Canvas key status…';
  try {
    const response = await fetch(`${API_BASE}/canvas/credentials`, {credentials: 'include'});
    if (!response.ok) throw new Error((await response.json()).detail || 'Could not load Canvas key status');
    const {credentials} = await response.json();
    form.innerHTML = ['development', 'production'].map((environment) => {
      const status = credentials[environment];
      const detail = status.configured
        ? `Configured${status.updated_at ? ` · last updated ${new Date(status.updated_at).toLocaleString()}` : ''}`
        : 'Not configured';
      return `
        <form data-canvas-environment="${environment}" style="background:white; border:1px solid var(--gray-300); border-radius:8px; padding:18px; margin-bottom:14px;">
          <h3 style="margin:0 0 6px;">${canvasEnvironmentLabel(environment)}</h3>
          <p style="margin:0 0 12px; color:${status.configured ? 'var(--success-color)' : 'var(--gray-700)'};">${detail}</p>
          <label style="display:block; margin-bottom:8px;">Canvas API key
            <textarea name="api_key" rows="2" autocomplete="off" autocapitalize="none" spellcheck="false" data-lpignore="true" data-1p-ignore="true" placeholder="Paste a new key to save or replace it" style="display:block; width:100%; margin-top:4px; box-sizing:border-box; resize:vertical; font-family:monospace;" aria-label="Canvas API key"></textarea>
          </label>
          <div style="display:flex; gap:8px;">
            <button type="submit" class="btn btn-primary">Save key</button>
            ${status.configured ? '<button type="button" class="btn btn-danger" data-remove-canvas-key>Remove key</button>' : ''}
          </div>
        </form>`;
    }).join('');
    form.querySelectorAll('form[data-canvas-environment]').forEach((keyForm) => {
      keyForm.addEventListener('submit', saveCanvasCredential);
      keyForm.querySelector('[data-remove-canvas-key]')?.addEventListener('click', removeCanvasCredential);
    });
  } catch (err) {
    error.textContent = err.message;
    error.style.display = 'block';
    form.textContent = '';
  }
}

async function saveCanvasCredential(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const apiKey = form.elements.api_key.value.trim();
  if (!apiKey) return;
  const error = document.getElementById('canvas-settings-error');
  error.style.display = 'none';
  try {
    const response = await fetch(`${API_BASE}/canvas/credentials`, {
      method: 'PUT', headers: {'Content-Type': 'application/json'}, credentials: 'include',
      body: JSON.stringify({environment: form.dataset.canvasEnvironment, api_key: apiKey}),
    });
    if (!response.ok) throw new Error((await response.json()).detail || 'Could not save Canvas API key');
    await loadCanvasCredentialSettings();
  } catch (err) {
    error.textContent = err.message;
    error.style.display = 'block';
  }
}

async function removeCanvasCredential(event) {
  const form = event.currentTarget.closest('form');
  if (!confirm(`Remove your ${canvasEnvironmentLabel(form.dataset.canvasEnvironment)} Canvas API key?`)) return;
  const error = document.getElementById('canvas-settings-error');
  error.style.display = 'none';
  try {
    const response = await fetch(`${API_BASE}/canvas/credentials/${form.dataset.canvasEnvironment}`, {
      method: 'DELETE', credentials: 'include',
    });
    if (!response.ok) throw new Error((await response.json()).detail || 'Could not remove Canvas API key');
    await loadCanvasCredentialSettings();
  } catch (err) {
    error.textContent = err.message;
    error.style.display = 'block';
  }
}
