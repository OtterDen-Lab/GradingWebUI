(() => {
  const form = document.querySelector('#run-form');
  const sessionSelect = document.querySelector('#session');
  const questionSelect = document.querySelector('#question');
  const availableModels = document.querySelector('#available-models');
  const selectedModels = document.querySelector('#selected-models');
  const sampleSize = document.querySelector('#sample-size');
  const startButton = document.querySelector('#start');
  const panel = document.querySelector('#run-panel');
  const progressBar = document.querySelector('#progress-bar');
  const progressText = document.querySelector('#progress-text');
  const resultContainer = document.querySelector('#results');
  const cancelButton = document.querySelector('#cancel-run');
  const exportButton = document.querySelector('#export-csv');
  const filters = document.querySelector('#result-filters');
  let activeRunId = null;
  let pollTimer = null;
  const shownImages = new Map();
  const expandedRawResponses = new Set();
  let modelCatalog = [];
  let selectedTargets = [];
  let currentRun = null;
  const activeFilters = new Set();

  const escapeHtml = value => String(value ?? '').replace(/[&<>'"]/g, character => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
  })[character]);

  async function api(url, options) {
    const response = await fetch(url, options);
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || 'Request failed');
    return body;
  }

  async function loadQuestions() {
    questionSelect.innerHTML = '';
    if (!sessionSelect.value) return;
    const data = await api(`/api/analysis/sessions/${sessionSelect.value}/questions`);
    for (const question of data.questions) {
      const option = document.createElement('option');
      option.value = question.problem_number;
      option.textContent = `Question ${question.problem_number} (${question.response_count} responses)`;
      questionSelect.append(option);
    }
  }

  async function loadSetup() {
    const [sessions, catalog] = await Promise.all([api('/api/sessions'), api('/api/analysis/models')]);
    sessionSelect.innerHTML = '<option value="">Select an exam…</option>';
    for (const session of sessions) {
      const option = document.createElement('option');
      option.value = session.id;
      option.textContent = session.name || `Session ${session.id}`;
      sessionSelect.append(option);
    }
    modelCatalog = catalog.models;
    renderModelPicker();
  }

  function renderModelPicker() {
    if (!modelCatalog.length) {
      availableModels.textContent = 'No configured models are available.';
      selectedModels.textContent = 'Select one or more models.';
      return;
    }
    availableModels.innerHTML = modelCatalog.map(model => `<label class="model-option"><input type="checkbox" value="${escapeHtml(model.id)}" ${selectedTargets.includes(model.id) ? 'checked' : ''}><span><strong>${escapeHtml(model.provider)}</strong> · ${escapeHtml(model.label)}</span></label>`).join('');
    availableModels.querySelectorAll('input').forEach(input => input.addEventListener('change', () => {
      if (input.checked) selectedTargets.push(input.value);
      else selectedTargets = selectedTargets.filter(target => target !== input.value);
      renderModelPicker();
    }));
    selectedModels.innerHTML = selectedTargets.length ? selectedTargets.map((target, index) => {
      const model = modelCatalog.find(item => item.id === target);
      return `<div class="selected-model"><span>${index + 1}. ${escapeHtml(model?.label || target)}</span><span class="selected-actions"><button type="button" data-move="up" data-index="${index}" ${index ? '' : 'disabled'}>↑</button><button type="button" data-move="down" data-index="${index}" ${index < selectedTargets.length - 1 ? '' : 'disabled'}>↓</button></span></div>`;
    }).join('') : 'Select one or more models.';
    selectedModels.querySelectorAll('button').forEach(button => button.addEventListener('click', () => {
      const index = Number(button.dataset.index);
      const neighbor = button.dataset.move === 'up' ? index - 1 : index + 1;
      [selectedTargets[index], selectedTargets[neighbor]] = [selectedTargets[neighbor], selectedTargets[index]];
      renderModelPicker();
    }));
  }

  function resultBadge(result) {
    if (result.error) return '<span class="badge failed">Failed</span>';
    if (result.is_blank) return '<span class="badge blank">AI blank</span>';
    if (result.is_relevant === true) return '<span class="badge relevant">Relevant</span>';
    if (result.is_relevant === false) return '<span class="badge irrelevant">Irrelevant</span>';
    return '<span class="badge">Classification unavailable</span>';
  }

  function render(run) {
    panel.hidden = false;
    currentRun = run;
    const complete = Number(run.completed_items);
    const total = Number(run.total_items);
    progressBar.style.width = `${total ? 100 * complete / total : 0}%`;
    progressText.textContent = `${run.status}: ${complete}/${total} model-response analyses complete; ${run.failed_items} failed.`;
    cancelButton.hidden = !['queued', 'running', 'cancelling'].includes(run.status);
    exportButton.hidden = !run.results.length;
    cancelButton.disabled = run.status === 'cancelling';
    cancelButton.textContent = run.status === 'cancelling' ? 'Cancelling after current request…' : 'Cancel run';
    document.querySelector('#run-title').textContent = `Question ${run.problem_number}: ${run.models.join(' vs ')}`;
    const grouped = new Map();
    for (const result of run.results) {
      if (!grouped.has(result.problem_id)) grouped.set(result.problem_id, []);
      grouped.get(result.problem_id).push(result);
    }
    filters.hidden = !grouped.size;
    const visibleResponses = [...grouped.entries()].filter(([, results]) => {
      if (!activeFilters.size) return true;
      return (activeFilters.has('heuristic_blank') && results[0].heuristic_is_blank) ||
        (activeFilters.has('ai_blank') && results.some(result => result.is_blank)) ||
        (activeFilters.has('not_relevant') && results.some(result => result.is_relevant === false));
    });
    resultContainer.innerHTML = visibleResponses.map(([problemId, results], index) => {
      const heuristic = results[0].heuristic_is_blank ? '<span class="badge blank">Heuristic blank</span>' : '<span class="badge">Heuristic nonblank</span>';
      return `<article class="response"><h3>Response ${index + 1} ${heuristic}</h3>
        <button type="button" class="show-image" data-problem-id="${problemId}">${shownImages.has(String(problemId)) ? 'Crop shown' : 'Show crop'}</button>
        <div class="image-holder">${shownImages.has(String(problemId)) ? `<img class="response-image" alt="Response crop" src="data:image/png;base64,${shownImages.get(String(problemId))}">` : ''}</div><div class="model-grid">${results.sort((left, right) => run.models.indexOf(left.model_id) - run.models.indexOf(right.model_id)).map(result =>
          `<div class="model-result"><strong>${escapeHtml(result.model_id)}</strong> ${resultBadge(result)}
          <p><strong>Transcription</strong></p><pre>${escapeHtml(result.transcription || result.error || '—')}</pre>
          <details class="raw-response" data-result-key="${escapeHtml(`${problemId}:${result.model_id}`)}" ${expandedRawResponses.has(`${problemId}:${result.model_id}`) ? 'open' : ''}><summary>Raw responses and timing (${Math.round(result.duration_ms || 0)} ms)</summary>
          <pre>${escapeHtml(result.raw_transcription_response || 'No transcription response recorded')}</pre>
          <pre>${escapeHtml(result.raw_relevance_response || 'No relevance response recorded')}</pre></details></div>`
        ).join('')}</div></article>`;
    }).join('') || (grouped.size ? '<p>No responses match the active filters.</p>' : '<p>Waiting for the first result…</p>');
    resultContainer.querySelectorAll('.show-image').forEach(button => button.addEventListener('click', () => showImage(button)));
    resultContainer.querySelectorAll('.raw-response').forEach(details => details.addEventListener('toggle', () => {
      if (details.open) expandedRawResponses.add(details.dataset.resultKey);
      else expandedRawResponses.delete(details.dataset.resultKey);
    }));
  }

  async function showImage(button) {
    const problemId = String(button.dataset.problemId);
    if (shownImages.has(problemId)) return;
    button.disabled = true;
    try {
      const image = await api(`/api/analysis/runs/${activeRunId}/problems/${problemId}/image`);
      // Polling redraws the result list while a crop is loading. Keep image
      // data outside the transient DOM so the next redraw displays it too.
      shownImages.set(problemId, image.image_data);
      const holder = button.nextElementSibling;
      if (holder) holder.innerHTML = `<img class="response-image" alt="Response crop" src="data:image/png;base64,${image.image_data}">`;
      button.textContent = 'Crop shown';
    } catch (error) {
      const holder = button.nextElementSibling;
      if (holder) holder.textContent = error.message;
    }
    button.disabled = false;
  }

  async function refreshRun() {
    if (!activeRunId) return;
    try {
      const run = await api(`/api/analysis/runs/${activeRunId}`);
      render(run);
      if (['completed', 'cancelled'].includes(run.status)) clearInterval(pollTimer);
    } catch (error) {
      progressText.textContent = error.message;
      clearInterval(pollTimer);
    }
  }

  sessionSelect.addEventListener('change', () => loadQuestions().catch(error => alert(error.message)));
  form.addEventListener('submit', async event => {
    event.preventDefault();
    const models = selectedTargets;
    if (!models.length) {
      alert('Select at least one model.');
      return;
    }
    startButton.disabled = true;
    try {
      const payload = { session_id: Number(sessionSelect.value), problem_number: Number(questionSelect.value), models };
      if (sampleSize.value) payload.sample_size = Number(sampleSize.value);
      const run = await api('/api/analysis/runs', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
      activeRunId = run.run_id;
      shownImages.clear();
      expandedRawResponses.clear();
      activeFilters.clear();
      filters.querySelectorAll('[data-filter]').forEach(item => item.classList.remove('active'));
      panel.hidden = false;
      resultContainer.innerHTML = '';
      progressText.textContent = 'Queued…';
      clearInterval(pollTimer);
      await refreshRun();
      pollTimer = setInterval(refreshRun, 1000);
    } catch (error) { alert(error.message); }
    startButton.disabled = false;
  });

  cancelButton.addEventListener('click', async () => {
    if (!activeRunId || !confirm('Cancel this analysis? The current model request may finish, but no further work will start.')) return;
    cancelButton.disabled = true;
    try {
      await api(`/api/analysis/runs/${activeRunId}/cancel`, {method: 'POST'});
      await refreshRun();
    } catch (error) {
      alert(error.message);
      cancelButton.disabled = false;
    }
  });

  filters.addEventListener('click', event => {
    const button = event.target.closest('[data-filter]');
    if (!button || !currentRun) return;
    const filter = button.dataset.filter;
    if (filter === 'clear') activeFilters.clear();
    else if (activeFilters.has(filter)) activeFilters.delete(filter);
    else activeFilters.add(filter);
    filters.querySelectorAll('[data-filter]').forEach(item => {
      item.classList.toggle('active', activeFilters.has(item.dataset.filter));
    });
    render(currentRun);
  });

  exportButton.addEventListener('click', () => {
    if (!currentRun) return;
    const columns = ['run_id', 'session_id', 'problem_number', 'problem_id', 'model_id', 'model_label', 'heuristic_is_blank', 'is_blank', 'is_relevant', 'transcription', 'duration_ms', 'error', 'raw_transcription_response', 'raw_relevance_response'];
    const csvValue = value => `"${String(value ?? '').replaceAll('"', '""')}"`;
    const lines = [columns.join(','), ...currentRun.results.map(result => columns.map(column => csvValue(currentRun.id && column === 'run_id' ? currentRun.id : currentRun[column] ?? result[column])).join(','))];
    const blob = new Blob([lines.join('\n')], {type: 'text/csv;charset=utf-8'});
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `handwriting-analysis-${currentRun.id}.csv`;
    link.click();
    URL.revokeObjectURL(link.href);
  });

  async function restoreMostRecentRun() {
    const data = await api('/api/analysis/runs?limit=1');
    if (!data.runs.length) return;
    activeRunId = data.runs[0].id;
    await refreshRun();
    if (!['completed', 'cancelled'].includes(data.runs[0].status)) {
      pollTimer = setInterval(refreshRun, 1000);
    }
  }

  Promise.all([loadSetup(), restoreMostRecentRun()])
    .catch(error => alert(`Could not load analysis setup: ${error.message}`));
})();
