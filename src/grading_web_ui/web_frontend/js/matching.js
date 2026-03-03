// Name matching functionality

let allSubmissions = [];
let allStudents = [];
let revealCanvasNames = true;
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

// Simple fuzzy matching helper (Levenshtein distance)
function fuzzyMatch(str1, str2) {
    const s1 = str1.toLowerCase();
    const s2 = str2.toLowerCase();
    const len1 = s1.length;
    const len2 = s2.length;

    const matrix = [];
    for (let i = 0; i <= len1; i++) {
        matrix[i] = [i];
    }
    for (let j = 0; j <= len2; j++) {
        matrix[0][j] = j;
    }

    for (let i = 1; i <= len1; i++) {
        for (let j = 1; j <= len2; j++) {
            const cost = s1[i - 1] === s2[j - 1] ? 0 : 1;
            matrix[i][j] = Math.min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost
            );
        }
    }

    const maxLen = Math.max(len1, len2);
    const distance = matrix[len1][len2];
    return Math.round((1 - distance / maxLen) * 100);
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

        // Pre-fill suggested matches based on fuzzy matching
        // Only if not already matched
        allSubmissions.forEach(submission => {
            if (!submission.canvas_user_id && submission.approximate_name) {
                let bestScore = 0;
                let bestStudent = null;

                allStudents.forEach(student => {
                    const score = fuzzyMatch(submission.approximate_name, student.name);
                    if (score > bestScore && score >= 98) {  // 98% threshold (same as backend)
                        bestScore = score;
                        bestStudent = student;
                    }
                });

                if (bestStudent) {
                    submission.suggested_canvas_user_id = bestStudent.user_id;
                    console.log(`Suggested match for "${submission.approximate_name}": ${bestStudent.name} (${bestScore}%)`);
                }
            }
        });

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

    const unmatchedCount = allSubmissions.filter(s => !s.is_matched).length;
    const matchedCount = allSubmissions.length - unmatchedCount;
    const percentage = allSubmissions.length > 0 ? (matchedCount / allSubmissions.length * 100) : 0;

    // Update progress bar
    document.getElementById('matching-progress-fill').style.width = `${percentage}%`;
    document.getElementById('matching-progress-text').textContent =
        `${matchedCount} of ${allSubmissions.length} matched (${unmatchedCount} remaining)`;

    let html = `
        <p style="margin-bottom: 20px;">
            <strong>${unmatchedCount}</strong> of <strong>${allSubmissions.length}</strong> submission(s) need manual matching.
        </p>
        <div style="margin-bottom: 20px; text-align: center;">
            <button id="confirm-all-matches-btn" class="btn btn-primary" onclick="confirmAllMatches()" style="padding: 10px 30px; font-size: 16px;">
                Confirm All Matches
            </button>
            <button class="btn btn-secondary" onclick="toggleCanvasNameReveal()" style="padding: 10px 20px; margin-left: 10px; font-size: 14px;">
                ${revealCanvasNames ? 'Hide Real Names' : 'Show Real Names'}
            </button>
            <div id="matching-action-status" style="margin-top: 10px; font-size: 14px; display: none;"></div>
            <p style="margin-top: 10px; color: var(--gray-600); font-size: 14px;">
                Select students from the dropdowns below, then click this button to confirm all changes at once.
            </p>
        </div>
    `;

    allSubmissions.forEach(submission => {
        const statusClass = submission.is_matched ? 'matched' : 'unmatched';
        const statusLabel = submission.is_matched ? `✓ Matched to: ${submission.student_name}` : 'Not matched';

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
                        ${allStudents.map(s => {
                            // Pre-select if this is the actual match OR the suggested match
                            const isSelected = (submission.canvas_user_id === s.user_id) ||
                                             (!submission.canvas_user_id && submission.suggested_canvas_user_id === s.user_id);
                            return `
                            <option value="${s.user_id}"
                                    ${s.is_matched ? 'class="matched-student"' : ''}
                                    ${isSelected ? 'selected' : ''}>
                                ${s.is_matched ? '✓ ' : ''}${s.name}
                            </option>
                        `;
                        }).join('')}
                    </select>
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
    setMatchingActionStatus(matchingActionStatus.message, matchingActionStatus.type);
    bindMatchingImagePreview();
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

// Handle student selection - show warning if student is already matched
function handleStudentSelection(submissionId) {
    const select = document.getElementById(`select-${submissionId}`);
    const selectedUserId = parseInt(select.value);

    if (!selectedUserId) return;

    // Find the selected student
    const student = allStudents.find(s => s.user_id === selectedUserId);

    // Check if this student is already matched
    if (student && student.is_matched) {
        const currentMatchId = select.dataset.currentMatch;

        // Only show warning if reassigning to a different student
        if (!currentMatchId || parseInt(currentMatchId) !== selectedUserId) {
            select.style.borderColor = '#ef4444';
            select.style.backgroundColor = '#fee2e2';
        }
    } else {
        select.style.borderColor = '';
        select.style.backgroundColor = '';
    }
}

// Confirm all matches at once (batch operation)
async function confirmAllMatches() {
    // Collect all pending matches
    const pendingMatches = [];
    const warnings = [];

    for (const submission of allSubmissions) {
        const select = document.getElementById(`select-${submission.id}`);
        const selectedUserId = parseInt(select.value);

        // Skip if no selection or if already matched to the same student
        if (!selectedUserId) continue;
        if (submission.is_matched && submission.canvas_user_id === selectedUserId) continue;

        // Check for warnings (reassignments)
        const student = allStudents.find(s => s.user_id === selectedUserId);
        if (student && student.is_matched) {
            const currentMatchId = select.dataset.currentMatch;
            if (!currentMatchId || parseInt(currentMatchId) !== selectedUserId) {
                warnings.push(`"${student.name}" will be reassigned to Exam #${submission.document_id + 1}`);
            }
        }

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

    // Show confirmation dialog with warnings if any
    let confirmMessage = `Confirm ${pendingMatches.length} match(es)?`;
    if (warnings.length > 0) {
        confirmMessage += '\n\nWarnings:\n' + warnings.join('\n');
    }

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
        // Process matches with bounded concurrency for faster confirmation.
        const revealQuery = revealCanvasNames ? '?reveal_names=true' : '';
        const queue = [...pendingMatches];
        const total = queue.length;
        const concurrency = Math.min(8, total);
        let completed = 0;
        let successCount = 0;
        let failCount = 0;

        const worker = async () => {
            while (queue.length > 0) {
                const match = queue.shift();
                if (!match) return;

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

        await Promise.all(Array.from({ length: concurrency }, () => worker()));

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
