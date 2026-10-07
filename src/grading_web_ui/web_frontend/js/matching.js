// Name matching functionality

let allSubmissions = [];
let allStudents = [];
let revealCanvasNames = true;
let matchingSectionFilter = 'all';
let matchingImagePreviewBound = false;
let matchingActionStatus = { message: '', type: 'info' };
let matchingPreviewSessionId = null;
const matchingPagePreviewCache = new Map();
let matchingPagePreviewDialog = null;
let matchingPagePreviewRequestId = 0;

async function getMatchingApiError(response, fallbackMessage) {
    try {
        const errorData = await response.json();
        if (errorData && errorData.detail) {
            return String(errorData.detail);
        }
    } catch {
        // Ignore parse errors and fall back to generic text.
    }
    return fallbackMessage;
}

function setMatchingActionStatus(message = '', type = 'info') {
    matchingActionStatus = { message, type };
    const statusEl = document.getElementById('matching-action-status');
    if (!statusEl) return;

    const colorMap = {
        info: '#1d4ed8',
        success: '#047857',
        warning: '#9a3412',
        error: '#b91c1c'
    };
    statusEl.textContent = message;
    statusEl.style.color = colorMap[type] || colorMap.info;
    statusEl.style.display = message ? 'block' : 'none';
}

function setMatchingControlsDisabled(disabled) {
    document.querySelectorAll('.student-select').forEach((select) => {
        select.disabled = disabled;
    });
}

function getMatchingSectionValues() {
    const sections = new Set();
    allStudents.forEach((student) => {
        if (student && student.section) {
            sections.add(student.section);
        }
    });
    return Array.from(sections).sort((a, b) => a.localeCompare(b));
}

function getStudentDisplayLabel(student) {
    if (!student) return '';
    return student.section ? `${student.name} (${student.section})` : student.name;
}

function getVisibleMatchingStudents(submission) {
    const currentMatch = submission.canvas_user_id
        ? allStudents.find((student) => student.user_id === submission.canvas_user_id)
        : null;

    let students = matchingSectionFilter === 'all'
        ? allStudents
        : allStudents.filter((student) => student.section === matchingSectionFilter);

    if (currentMatch && !students.some((student) => student.user_id === currentMatch.user_id)) {
        students = [currentMatch, ...students];
    }

    return students;
}

// Load name matching interface
async function loadNameMatching() {
    if (!currentSession) return;

    try {
        if (matchingPreviewSessionId !== currentSession.id) {
            matchingPagePreviewCache.clear();
            matchingPreviewSessionId = currentSession.id;
        }

        // Fetch all submissions (unmatched first)
        const submissionsResp = await fetch(`${API_BASE}/matching/${currentSession.id}/submissions`);
        if (!submissionsResp.ok) {
            throw new Error(await getMatchingApiError(submissionsResp, 'Failed to load submissions'));
        }
        const submissionsData = await submissionsResp.json();
        allSubmissions = submissionsData.submissions;

        // Fetch all students (unmatched first)
        const revealQuery = revealCanvasNames ? '?reveal_names=true' : '';
        const studentsResp = await fetch(`${API_BASE}/matching/${currentSession.id}/students${revealQuery}`);
        if (!studentsResp.ok) {
            throw new Error(await getMatchingApiError(studentsResp, 'Failed to load Canvas roster'));
        }
        const studentsData = await studentsResp.json();
        allStudents = studentsData.students;

        // Render UI
        renderMatchingList();

    } catch (error) {
        console.error('Failed to load matching data:', error);
        const container = document.getElementById('unmatched-list');
        if (container) {
            container.innerHTML = `<div class="info-box error" style="margin-top: 10px;">${error.message}</div>`;
        }
    }
}

// Render all submissions list
function renderMatchingList() {
    const container = document.getElementById('unmatched-list');
    const canDeleteSubmissions = currentUser && currentUser.role === 'instructor';
    const sectionValues = getMatchingSectionValues();

    if (matchingSectionFilter !== 'all' && !sectionValues.includes(matchingSectionFilter)) {
        matchingSectionFilter = 'all';
    }

    const matchedCount = allSubmissions.filter(s => s.is_matched).length;
    const suggestedCount = allSubmissions.filter(
        s => !s.is_matched && s.suggested_canvas_user_id
    ).length;
    const unmatchedCount = allSubmissions.length - matchedCount - suggestedCount;
    const percentage = allSubmissions.length > 0 ? (matchedCount / allSubmissions.length * 100) : 0;

    // Update progress bar
    document.getElementById('matching-progress-fill').style.width = `${percentage}%`;
    document.getElementById('matching-progress-text').textContent =
        `${matchedCount} confirmed, ${suggestedCount} suggested, ${unmatchedCount} need a selection`;

    let html = `
        <p style="margin-bottom: 20px;">
            <strong>${matchedCount}</strong> confirmed, <strong>${suggestedCount}</strong> AI suggestion(s) awaiting review, and <strong>${unmatchedCount}</strong> submission(s) needing a selection.
        </p>
        <div style="margin-bottom: 20px; text-align: center;">
            <button id="confirm-all-matches-btn" class="btn btn-primary" onclick="confirmAllMatches()" style="padding: 10px 30px; font-size: 16px;">
                Confirm All Matches
            </button>
            <button class="btn btn-secondary" onclick="toggleCanvasNameReveal()" style="padding: 10px 20px; margin-left: 10px; font-size: 14px;">
                ${revealCanvasNames ? 'Hide Real Names' : 'Show Real Names'}
            </button>
            <div id="matching-action-status" style="margin-top: 10px; font-size: 14px; display: none;"></div>
            ${sectionValues.length > 0 ? `
                <div style="margin-top: 14px; display: flex; justify-content: center; gap: 8px; align-items: center; flex-wrap: wrap;">
                    <label for="matching-section-filter" style="font-size: 14px; color: var(--gray-700);">
                        Section:
                    </label>
                    <select id="matching-section-filter" style="min-width: 140px;">
                        <option value="all" ${matchingSectionFilter === 'all' ? 'selected' : ''}>All sections</option>
                        ${sectionValues.map((section) => `
                            <option value="${escapeHtml(section)}" ${matchingSectionFilter === section ? 'selected' : ''}>
                                ${escapeHtml(section)}
                            </option>
                        `).join('')}
                    </select>
                </div>
            ` : ''}
            <p style="margin-top: 10px; color: var(--gray-600); font-size: 14px;">
                Select students from the dropdowns below, review the yellow AI suggestions, then click this button to confirm all changes at once.
            </p>
        </div>
    `;

    allSubmissions.forEach(submission => {
        const suggestedStudent = !submission.is_matched && submission.suggested_canvas_user_id
            ? allStudents.find((student) => student.user_id === submission.suggested_canvas_user_id)
            : null;
        const statusClass = submission.is_matched
            ? 'matched'
            : suggestedStudent ? 'soft-matched' : 'unmatched';
        const statusLabel = submission.is_matched
            ? `✓ Confirmed match: ${submission.student_name}`
            : suggestedStudent
                ? `✓ AI suggestion: ${getStudentDisplayLabel(suggestedStudent)} — review and confirm`
                : 'No AI suggestion — select a student';

        html += `
            <div class="matching-item ${statusClass}" data-submission-id="${submission.id}">
                <div class="matching-info">
                    <div class="matching-preview-row">
                        ${submission.name_image_data ? `
                            <img src="data:image/png;base64,${submission.name_image_data}"
                                 alt="Name area"
                                 title="Click to view full page"
                                 class="matching-name-image">
                        ` : ''}
                        <div style="flex: 1;">
                            <strong>Exam #${submission.document_id + 1}</strong>
                            <div class="detected-name">AI detected: <em>${submission.approximate_name}</em></div>
                            <div class="match-status">${statusLabel}</div>
                        </div>
                    </div>
                </div>
                <div class="matching-control">
                    <select class="student-select" id="select-${submission.id}"
                            ${submission.is_matched ? `data-current-match="${submission.canvas_user_id}"` : ''}
                            onchange="handleStudentSelection(${submission.id})">
                        <option value="">-- Select Canvas Student --</option>
                        ${getVisibleMatchingStudents(submission).map(s => {
                            // Pre-select if this is the actual match OR the suggested match
                            const isSelected = (submission.canvas_user_id === s.user_id) ||
                                             (!submission.canvas_user_id && submission.suggested_canvas_user_id === s.user_id);
                            return `
                            <option value="${s.user_id}"
                                    ${s.is_matched ? 'class="matched-student"' : ''}
                                    ${isSelected ? 'selected' : ''}>
                                ${s.is_matched ? '✓ ' : ''}${getStudentDisplayLabel(s)}
                            </option>
                        `;
                        }).join('')}
                    </select>
                    ${canDeleteSubmissions ? `
                        <div style="margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap;">
                            <button type="button"
                                    class="btn btn-danger btn-small matching-action-btn"
                                    data-action="remove-submission"
                                    data-submission-id="${submission.id}"
                                    data-student-name="${escapeHtml(submission.student_name || `Exam #${submission.document_id + 1}`)}">
                                Erase Exam
                            </button>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
    setMatchingActionStatus(matchingActionStatus.message, matchingActionStatus.type);
    handleStudentSelection();
    bindMatchingImagePreview();

    document.querySelectorAll('.matching-action-btn[data-action="remove-submission"]').forEach((button) => {
        button.addEventListener('click', async () => {
            const submissionId = parseInt(button.dataset.submissionId, 10);
            const studentName = button.dataset.studentName || 'Submission';
            await eraseMatchingSubmission(submissionId, studentName, button);
        });
    });

    const sectionFilter = document.getElementById('matching-section-filter');
    if (sectionFilter) {
        sectionFilter.addEventListener('change', () => {
            matchingSectionFilter = sectionFilter.value || 'all';
            renderMatchingList();
        });
    }
}

function bindMatchingImagePreview() {
    if (matchingImagePreviewBound) return;

    const container = document.getElementById('unmatched-list');
    if (!container) return;

    container.addEventListener('click', async (event) => {
        const image = event.target.closest('.matching-name-image');
        if (!image) return;
        const matchingItem = image.closest('.matching-item');
        if (!matchingItem) return;

        const submissionId = parseInt(matchingItem.dataset.submissionId, 10);
        if (!Number.isInteger(submissionId)) return;

        await openMatchingPagePreview(submissionId);
    });

    matchingImagePreviewBound = true;
}

function ensureMatchingPagePreviewDialog() {
    if (matchingPagePreviewDialog) {
        return matchingPagePreviewDialog;
    }

    const overlay = document.createElement('div');
    overlay.id = 'matching-page-preview-dialog';
    overlay.style.display = 'none';
    overlay.style.position = 'fixed';
    overlay.style.top = '0';
    overlay.style.left = '0';
    overlay.style.width = '100%';
    overlay.style.height = '100%';
    overlay.style.background = 'rgba(0,0,0,0.8)';
    overlay.style.zIndex = '2200';
    overlay.style.alignItems = 'center';
    overlay.style.justifyContent = 'center';
    overlay.innerHTML = `
        <div style="background: white; border-radius: 8px; padding: 20px; max-width: 95vw; max-height: 95vh; overflow: auto;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <h3 id="matching-page-preview-title" style="margin: 0;">Full Page Preview</h3>
                <button id="matching-page-preview-close" class="btn" style="padding: 5px 15px;">&times;</button>
            </div>
            <div id="matching-page-preview-status" style="margin-bottom: 12px; color: var(--gray-700);">Loading full page preview...</div>
            <img id="matching-page-preview-image" alt="Full page preview" style="display: none; max-width: 100%; height: auto;">
        </div>
    `;

    document.body.appendChild(overlay);

    const closeButton = overlay.querySelector('#matching-page-preview-close');
    closeButton.addEventListener('click', () => {
        closeMatchingPagePreviewDialog();
    });

    overlay.addEventListener('click', (event) => {
        if (event.target === overlay) {
            closeMatchingPagePreviewDialog();
        }
    });

    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && overlay.style.display === 'flex') {
            closeMatchingPagePreviewDialog();
        }
    });

    matchingPagePreviewDialog = {
        overlay,
        title: overlay.querySelector('#matching-page-preview-title'),
        status: overlay.querySelector('#matching-page-preview-status'),
        image: overlay.querySelector('#matching-page-preview-image')
    };

    return matchingPagePreviewDialog;
}

function closeMatchingPagePreviewDialog() {
    if (!matchingPagePreviewDialog) return;
    matchingPagePreviewDialog.overlay.style.display = 'none';
}

async function getPagePreviewErrorMessage(response) {
    try {
        const errorData = await response.json();
        if (errorData && errorData.detail) {
            return String(errorData.detail);
        }
    } catch {
        // Fall back to a generic error message.
    }
    return `Request failed (${response.status})`;
}

async function eraseMatchingSubmission(submissionId, studentName, triggerButton) {
    if (!currentSession) return;

    const confirmed = confirm(
        `Erase ${studentName}?\n\nThis permanently deletes the exam and all grading data for this submission.`
    );
    if (!confirmed) {
        return;
    }

    if (triggerButton) {
        triggerButton.disabled = true;
        triggerButton.textContent = 'Erasing...';
    }

    try {
        const response = await fetch(`${API_BASE}/sessions/${currentSession.id}/submissions/${submissionId}`, {
            method: 'DELETE'
        });
        const payload = await response.json();

        if (!response.ok) {
            throw new Error(payload.detail || 'Failed to erase submission');
        }

        await loadNameMatching();

        const remainingCount = allSubmissions.length;
        const unmatchedCount = allSubmissions.filter((submission) => !submission.is_matched).length;
        if (remainingCount > 0 && unmatchedCount === 0) {
            setMatchingActionStatus('All remaining exams are matched. Preparing alignment...', 'info');
            await prepareAlignment();
        } else if (remainingCount === 0) {
            setMatchingActionStatus('All exams have been erased from this session.', 'warning');
        } else {
            setMatchingActionStatus(`Erased ${studentName}.`, 'success');
        }
    } catch (error) {
        console.error('Failed to erase submission:', error);
        setMatchingActionStatus(error.message || 'Failed to erase submission.', 'error');
        alert(error.message || 'Failed to erase submission.');
        if (triggerButton) {
            triggerButton.disabled = false;
            triggerButton.textContent = 'Erase Exam';
        }
    }
}

async function openMatchingPagePreview(submissionId) {
    if (!currentSession) return;

    const submission = allSubmissions.find((s) => s.id === submissionId);
    if (!submission) {
        throw new Error('Submission not found');
    }

    const dialog = ensureMatchingPagePreviewDialog();
    const requestId = ++matchingPagePreviewRequestId;

    dialog.title.textContent = `Exam #${submission.document_id + 1} Full Page`;
    dialog.status.style.display = 'block';
    dialog.status.style.color = 'var(--gray-700)';
    dialog.status.textContent = 'Loading full page preview...';
    dialog.image.style.display = 'none';
    dialog.overlay.style.display = 'flex';

    try {
        let pageImage = matchingPagePreviewCache.get(submissionId);

        if (!pageImage) {
            const response = await fetch(`${API_BASE}/matching/${currentSession.id}/submissions/${submissionId}/page-preview`);
            if (!response.ok) {
                throw new Error(await getPagePreviewErrorMessage(response));
            }
            const result = await response.json();
            pageImage = result.page_image;
            if (!pageImage) {
                throw new Error('No preview image returned');
            }
            matchingPagePreviewCache.set(submissionId, pageImage);
        }

        // Avoid updating the dialog with stale responses if user clicked another image.
        if (requestId !== matchingPagePreviewRequestId) {
            return;
        }

        dialog.image.src = `data:image/png;base64,${pageImage}`;
        dialog.image.style.display = 'block';
        dialog.status.style.display = 'none';
    } catch (error) {
        if (requestId !== matchingPagePreviewRequestId) {
            return;
        }
        dialog.image.removeAttribute('src');
        dialog.image.style.display = 'none';
        dialog.status.style.display = 'block';
        dialog.status.style.color = 'var(--danger-color)';
        dialog.status.textContent = `Unable to load full page preview: ${error.message}`;
        console.error('Failed to load full page preview:', error);
    }
}

async function toggleCanvasNameReveal() {
    revealCanvasNames = !revealCanvasNames;
    await loadNameMatching();
}

// Highlight duplicate current choices.  Saved matches are deliberately not
// considered here: an instructor may be moving a student from one row to
// another in this same review pass.
function handleStudentSelection(submissionId) {
    const selectedIds = new Map();
    document.querySelectorAll('.student-select').forEach((select) => {
        const userId = parseInt(select.value, 10);
        if (!userId) return;
        const selects = selectedIds.get(userId) || [];
        selects.push(select);
        selectedIds.set(userId, selects);
    });

    document.querySelectorAll('.student-select').forEach((select) => {
        const userId = parseInt(select.value, 10);
        const duplicate = userId && (selectedIds.get(userId) || []).length > 1;
        select.style.borderColor = duplicate ? '#ef4444' : '';
        select.style.backgroundColor = duplicate ? '#fee2e2' : '';
    });
}

// Confirm all matches at once (batch operation)
async function confirmAllMatches() {
    // Collect all pending matches
    const pendingMatches = [];
    const selectedStudentIds = new Map();

    for (const submission of allSubmissions) {
        const select = document.getElementById(`select-${submission.id}`);
        const selectedUserId = parseInt(select.value);

        if (selectedUserId && selectedStudentIds.has(selectedUserId)) {
            const otherExam = selectedStudentIds.get(selectedUserId);
            alert(`The same student is selected for Exam #${otherExam} and Exam #${submission.document_id + 1}. Assign each student to only one exam.`);
            return;
        }
        if (selectedUserId) selectedStudentIds.set(selectedUserId, submission.document_id + 1);

        // A blank selection means the existing confirmed match should be
        // cleared; a selected value is the row's intended final assignment.
        if (submission.is_matched && submission.canvas_user_id === selectedUserId) continue;

        pendingMatches.push({
            submission_id: submission.id,
            canvas_user_id: selectedUserId,
            exam_number: submission.document_id + 1
        });
    }

    // Check if all submissions are already matched (even if no pending changes)
    const unmatchedCount = allSubmissions.filter(s => !s.is_matched).length;

    if (pendingMatches.length === 0) {
        // No pending changes, but check if we should proceed
        if (unmatchedCount === 0) {
            console.log('All submissions already matched. Preparing alignment...');
            setMatchingActionStatus('All submissions are matched. Preparing alignment...', 'info');
            await prepareAlignment();
            return;
        }

        alert('No new matches to confirm. Please select students from the dropdowns.');
        return;
    }

    // Show confirmation dialog.
    let confirmMessage = `Confirm ${pendingMatches.length} match(es)?`;

    if (!confirm(confirmMessage)) {
        return;
    }

    // Disable button during processing
    const btn = document.getElementById('confirm-all-matches-btn');
    btn.disabled = true;
    setMatchingControlsDisabled(true);
    btn.textContent = `Processing 0/${pendingMatches.length}...`;
    setMatchingActionStatus(`Saving ${pendingMatches.length} confirmed match(es)...`, 'info');

    try {
        // First release every changed confirmed match.  This makes moves and
        // swaps work in one confirmation pass instead of requiring users to
        // unmatch a row, save, and come back for a second pass.
        const revealQuery = revealCanvasNames ? '?reveal_names=true' : '';
        const changedExistingMatches = pendingMatches.filter((match) => {
            const submission = allSubmissions.find((item) => item.id === match.submission_id);
            return submission && submission.is_matched;
        });
        for (const match of changedExistingMatches) {
            const response = await fetch(`${API_BASE}/matching/${currentSession.id}/unmatch`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ submission_id: match.submission_id })
            });
            if (!response.ok) throw new Error(await getMatchingApiError(response, 'Failed to clear existing match'));
        }

        const queue = [...pendingMatches];
        const total = queue.length;
        let completed = 0;
        let successCount = 0;
        let failCount = 0;

        const worker = async () => {
            while (queue.length > 0) {
                const match = queue.shift();
                if (!match) return;

                if (!match.canvas_user_id) {
                    completed++;
                    continue;
                }

                try {
                    const response = await fetch(`${API_BASE}/matching/${currentSession.id}/match${revealQuery}`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            submission_id: match.submission_id,
                            canvas_user_id: match.canvas_user_id
                        })
                    });

                    if (response.ok) {
                        successCount++;
                    } else {
                        failCount++;
                        console.error(`Failed to match submission ${match.submission_id}`);
                    }
                } catch (error) {
                    failCount++;
                    console.error(`Error matching submission ${match.submission_id}:`, error);
                } finally {
                    completed++;
                    btn.textContent = `Processing ${completed}/${total}...`;
                    setMatchingActionStatus(`Saving matches: ${completed}/${total} complete...`, 'info');
                }
            }
        };

        await worker();

        // Show result
        if (failCount > 0) {
            alert(`Completed with ${successCount} successful and ${failCount} failed matches.`);
            setMatchingActionStatus(
                `Completed with ${successCount} successful and ${failCount} failed match(es).`,
                'warning'
            );
        } else {
            setMatchingActionStatus(`Saved ${successCount} match(es).`, 'success');
        }

        // Reload data to reflect changes
        setMatchingActionStatus('Refreshing match status...', 'info');
        await loadNameMatching();

        // Check if all submissions are matched, then move to alignment
        const unmatchedCount = allSubmissions.filter(s => !s.is_matched).length;

        if (unmatchedCount === 0) {
            console.log(`All ${allSubmissions.length} submissions matched. Preparing alignment...`);
            setMatchingActionStatus('All submissions matched. Preparing alignment...', 'info');
            await prepareAlignment();
        } else {
            // Some submissions still unmatched
            console.log(`${unmatchedCount} submissions still need matching`);
            setMatchingActionStatus(
                `${unmatchedCount} submission(s) still need to be matched before alignment.`,
                'warning'
            );
            alert(`${unmatchedCount} submission(s) still need to be matched. Please select students for all submissions.`);
        }

    } catch (error) {
        console.error('Failed to confirm matches:', error);
        setMatchingActionStatus(`Failed to confirm matches: ${error.message}`, 'error');
        alert('Failed to confirm matches: ' + error.message);
    } finally {
        setMatchingControlsDisabled(false);
        btn.disabled = false;
        btn.textContent = 'Confirm All Matches';
    }
}

// Match a submission to a student (legacy single-match function, kept for compatibility)
async function matchSubmission(submissionId) {
    const select = document.getElementById(`select-${submissionId}`);
    const canvasUserId = parseInt(select.value);

    if (!canvasUserId) {
        alert('Please select a student');
        return;
    }

    // Find the selected student
    const student = allStudents.find(s => s.user_id === canvasUserId);

    // Confirm if reassigning
    if (student && student.is_matched) {
        const currentMatchId = select.dataset.currentMatch;
        if (!currentMatchId || parseInt(currentMatchId) !== canvasUserId) {
            if (!confirm(`"${student.name}" is already matched to another exam. This will unassign them from that exam and assign them to this one. Continue?`)) {
                return;
            }
        }
    }

    try {
        const revealQuery = revealCanvasNames ? '?reveal_names=true' : '';
        const response = await fetch(`${API_BASE}/matching/${currentSession.id}/match${revealQuery}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                submission_id: submissionId,
                canvas_user_id: canvasUserId
            })
        });

        const result = await response.json();

        // Reload data to reflect changes
        await loadNameMatching();

        // If all matched, move to alignment
        if (result.remaining_unmatched === 0) {
            setTimeout(async () => {
                await prepareAlignment();
            }, 1500);
        }

    } catch (error) {
        console.error('Failed to match submission:', error);
        alert('Failed to match submission');
    }
}

// Auto-load data when navigating to sections
document.addEventListener('DOMContentLoaded', () => {
    const originalNavigate = window.navigateToSection;
    window.navigateToSection = function(sectionId) {
        originalNavigate(sectionId);
        if (sectionId === 'matching-section') {
            loadNameMatching();
        } else if (sectionId === 'grading-section') {
            const skipInit = Boolean(window.__skipNextGradingInitialize);
            window.__skipNextGradingInitialize = false;
            if (!skipInit) {
                if (typeof window.initializeGrading === 'function') {
                    window.initializeGrading();
                } else {
                    console.error('initializeGrading is not available on window');
                }
            }
        } else if (sectionId === 'stats-section') {
            loadStatistics();
            // Check if finalization is in progress
            if (currentSession && currentSession.status === 'finalizing') {
                document.getElementById('finalization-progress').style.display = 'block';
                document.getElementById('finalize-btn').disabled = true;
                startFinalizationPolling();
            }
        }
    };
});
