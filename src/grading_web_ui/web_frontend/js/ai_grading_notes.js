// AI Grading Notes (per-problem instructions for autograding)

let currentAiGradingNotes = null;

async function loadAiGradingNotes(sessionId, problemNumber) {
  try {
    const response = await fetch(
      `${API_BASE}/sessions/${sessionId}/ai-grading-notes/${problemNumber}`
    );

    if (!response.ok) {
      throw new Error('Failed to load AI grading notes');
    }

    const data = await response.json();
    currentAiGradingNotes = data.ai_grading_notes;
    displayAiGradingNotes();

    const container = document.getElementById('ai-grading-notes-container');
    if (container) {
      container.style.display = 'block';
    }
  } catch (error) {
    console.error('Failed to load AI grading notes:', error);
    currentAiGradingNotes = null;
    displayAiGradingNotes();
  }
}

function displayAiGradingNotes() {
  const display = document.getElementById('ai-grading-notes-display');
  if (!display) return;

  display.innerHTML = '';

  if (!currentAiGradingNotes) {
    const placeholder = document.createElement('span');
    placeholder.className = 'default-feedback-placeholder';
    placeholder.textContent = 'No AI grading notes set.';
    display.appendChild(placeholder);
  } else {
    display.textContent = currentAiGradingNotes;
  }
}

function showAiGradingNotesDialog() {
  const dialog = document.getElementById('edit-ai-grading-notes-dialog');
  const textarea = document.getElementById('ai-grading-notes-text');
  if (!dialog || !textarea) return;

  textarea.value = currentAiGradingNotes || '';
  dialog.style.display = 'flex';
  textarea.focus();
}

function hideAiGradingNotesDialog() {
  const dialog = document.getElementById('edit-ai-grading-notes-dialog');
  if (dialog) {
    dialog.style.display = 'none';
  }
}

function clearAiGradingNotes() {
  if (!confirm('Clear AI grading notes for this problem?')) {
    return;
  }

  const textarea = document.getElementById('ai-grading-notes-text');
  if (textarea) {
    textarea.value = '';
  }
}

async function saveAiGradingNotes() {
  const textarea = document.getElementById('ai-grading-notes-text');
  if (!textarea) return;

  const notesText = textarea.value.trim();
  if (notesText.length > 2000) {
    alert('AI grading notes must be 2000 characters or less');
    textarea.focus();
    return;
  }

  try {
    const saveBtn = document.getElementById('save-ai-grading-notes-btn');
    saveBtn.disabled = true;
    saveBtn.textContent = 'Saving...';

    const params = new URLSearchParams({
      problem_number: currentProblemNumber
    });

    if (notesText) {
      params.append('ai_grading_notes', notesText);
    }

    const response = await fetch(
      `${API_BASE}/sessions/${currentSession.id}/ai-grading-notes?${params}`,
      { method: 'PUT' }
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to save AI grading notes');
    }

    currentAiGradingNotes = notesText || null;
    displayAiGradingNotes();
    hideAiGradingNotesDialog();
  } catch (error) {
    console.error('Failed to save AI grading notes:', error);
    alert(`Failed to save AI grading notes: ${error.message}`);
  } finally {
    const saveBtn = document.getElementById('save-ai-grading-notes-btn');
    if (saveBtn) {
      saveBtn.disabled = false;
      saveBtn.textContent = 'Save';
    }
  }
}

document.getElementById('edit-ai-grading-notes-btn')?.addEventListener('click', () => {
  showAiGradingNotesDialog();
});

document.getElementById('cancel-ai-grading-notes-btn')?.addEventListener('click', () => {
  hideAiGradingNotesDialog();
});

document.getElementById('clear-ai-grading-notes-btn')?.addEventListener('click', () => {
  clearAiGradingNotes();
});

document.getElementById('save-ai-grading-notes-btn')?.addEventListener('click', () => {
  saveAiGradingNotes();
});
