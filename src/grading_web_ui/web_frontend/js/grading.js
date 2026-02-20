// Grading interface logic

let currentProblem = null;
let currentProblemNumber = 1;
let availableProblemNumbers = [];
let lastGradedProblemNumber = null; // Track if we just graded something
let problemMaxPoints = {}; // Cache max points per problem number
let problemHistory = []; // Track navigation history for back button
let historyIndex = -1; // Current position in history
const regeneratedAnswerCache = new Map();
const regeneratedAnswerRequests = new Map();
const MAX_REGENERATED_ANSWER_CACHE = 200;
const regenerationSessionPrefetchStarted = new Set();
const subjectiveSettingsByProblem = new Map();
let currentSubjectiveSettings = null;
let selectedSubjectiveBucketId = null;
let activeSubjectiveBucketFilter = '';
let subjectiveAssignInFlight = false;
const PREFETCH_QUEUE_TARGET = 2;
const prefetchedNextProblems = [];
let prefetchQueueInFlight = false;
let prefetchQueueProblemNumber = null;

const DEFAULT_SUBJECTIVE_BUCKETS = [
    { id: 'perfect', label: 'Perfect', color: '#16a34a' },
    { id: 'excellent', label: 'Excellent', color: '#22c55e' },
    { id: 'good', label: 'Good', color: '#3b82f6' },
    { id: 'passable', label: 'Passable', color: '#f59e0b' },
    { id: 'poor_blank', label: 'Poor/Blank', color: '#ef4444' }
];

function invalidateNextProblemPrefetch() {
    prefetchedNextProblems.length = 0;
    prefetchQueueInFlight = false;
    prefetchQueueProblemNumber = null;
}

function canPrefetchNextProblems() {
    if (!currentSession || !currentProblemNumber || activeSubjectiveBucketFilter) {
        return false;
    }
    const problemNumber = Number(currentProblemNumber);
    if (prefetchQueueProblemNumber === null) {
        prefetchQueueProblemNumber = problemNumber;
    } else if (prefetchQueueProblemNumber !== problemNumber) {
        prefetchedNextProblems.length = 0;
        prefetchQueueProblemNumber = problemNumber;
    }
    return true;
}

function getPrefetchExcludeProblemIds() {
    const ids = new Set();
    if (currentProblem?.id) {
        ids.add(Number(currentProblem.id));
    }
    prefetchedNextProblems.forEach((problem) => {
        if (problem?.id) {
            ids.add(Number(problem.id));
        }
    });
    return Array.from(ids).filter((id) => Number.isInteger(id) && id > 0);
}

async function prefetchNextProblemCandidate() {
    if (!canPrefetchNextProblems()) {
        return false;
    }

    const requestedSessionId = Number(currentSession.id);
    const requestedProblemNumber = Number(currentProblemNumber);
    const requestedFilter = activeSubjectiveBucketFilter;
    const excludeIds = getPrefetchExcludeProblemIds();
    const params = new URLSearchParams();
    if (excludeIds.length > 0) {
        params.set('exclude_problem_ids', excludeIds.join(','));
    }
    const query = params.toString() ? `?${params.toString()}` : '';
    const response = await fetch(
        `${API_BASE}/problems/${requestedSessionId}/${requestedProblemNumber}/next${query}`
    );

    if (response.status === 404) {
        return false;
    }
    if (!response.ok) {
        const message = await response.text();
        console.debug('Prefetch next problem failed:', response.status, message);
        return false;
    }

    const candidate = await response.json();
    if (
        !candidate ||
        Number(currentSession?.id) !== requestedSessionId ||
        Number(currentProblemNumber) !== requestedProblemNumber ||
        activeSubjectiveBucketFilter !== requestedFilter
    ) {
        return false;
    }

    if (
        (currentProblem && candidate.id === currentProblem.id) ||
        prefetchedNextProblems.some((problem) => problem.id === candidate.id)
    ) {
        return false;
    }

    prefetchedNextProblems.push(candidate);
    return true;
}

async function fillNextProblemPrefetchQueue() {
    if (prefetchQueueInFlight || !canPrefetchNextProblems()) {
        return;
    }
    prefetchQueueInFlight = true;
    try {
        while (prefetchedNextProblems.length < PREFETCH_QUEUE_TARGET) {
            const added = await prefetchNextProblemCandidate();
            if (!added) {
                break;
            }
        }
    } finally {
        prefetchQueueInFlight = false;
    }
}

function triggerNextProblemPrefetch() {
    void fillNextProblemPrefetchQueue();
}

function usePrefetchedNextProblemIfAvailable() {
    if (!canPrefetchNextProblems()) {
        return false;
    }

    while (prefetchedNextProblems.length > 0) {
        const candidate = prefetchedNextProblems.shift();
        if (!candidate) continue;
        if (currentProblem && candidate.id === currentProblem.id) continue;

        currentProblem = candidate;
        addToHistory(currentProblem);
        displayCurrentProblem();
        triggerNextProblemPrefetch();
        return true;
    }
    return false;
}

function cacheRegeneratedAnswer(problemId, data) {
    regeneratedAnswerCache.set(problemId, data);
    if (regeneratedAnswerCache.size > MAX_REGENERATED_ANSWER_CACHE) {
        const oldestKey = regeneratedAnswerCache.keys().next().value;
        regeneratedAnswerCache.delete(oldestKey);
    }
}

async function getRegeneratedAnswer(problemId) {
    if (!problemId) {
        throw new Error('No problem loaded');
    }

    if (regeneratedAnswerCache.has(problemId)) {
        return regeneratedAnswerCache.get(problemId);
    }

    if (regeneratedAnswerRequests.has(problemId)) {
        return regeneratedAnswerRequests.get(problemId);
    }

    const request = (async () => {
        const response = await fetch(`${API_BASE}/problems/${problemId}/regenerate-answer`);
        let payload = null;
        try {
            payload = await response.json();
        } catch (error) {
            payload = null;
        }

        if (!response.ok) {
            throw new Error(payload?.detail || 'Failed to load answer');
        }

        cacheRegeneratedAnswer(problemId, payload);
        return payload;
    })();

    regeneratedAnswerRequests.set(problemId, request);
    try {
        return await request;
    } finally {
        regeneratedAnswerRequests.delete(problemId);
    }
}

async function ensureExplanationLoaded(problemId) {
    if (!problemId) {
        return false;
    }

    if (explanationCache[problemId]) {
        return true;
    }

    try {
        const data = await getRegeneratedAnswer(problemId);
        if (data.explanation_html || data.explanation_markdown) {
            explanationCache[problemId] = {
                html: data.explanation_html || null,
                markdown: data.explanation_markdown || null
            };
            return true;
        }
    } catch (error) {
        console.error('Failed to load explanation:', error);
    }

    return false;
}

function invalidateRegeneratedAnswer(problemId) {
    if (!problemId) return;
    regeneratedAnswerCache.delete(problemId);
    regeneratedAnswerRequests.delete(problemId);
}

function startSessionRegenerationPrefetch() {
    if (!currentSession || !currentSession.id) {
        return;
    }

    if (regenerationSessionPrefetchStarted.has(currentSession.id)) {
        return;
    }
    regenerationSessionPrefetchStarted.add(currentSession.id);

    fetch(`${API_BASE}/problems/session/${currentSession.id}/prefetch-regeneration`, {
        method: 'POST'
    })
        .then((response) => response.json().catch(() => null))
        .then((payload) => {
            if (!payload) return;
            if (payload.status === 'started') {
                console.log(`Started regeneration prefetch for session ${currentSession.id} (${payload.total_qr_problems} QR-backed problems)`);
            } else if (payload.status === 'already_running') {
                console.log(`Regeneration prefetch already running for session ${currentSession.id}`);
            } else if (payload.status === 'no_qr_data') {
                console.log(`No QR-backed problems to prefetch for session ${currentSession.id}`);
            }
        })
        .catch((error) => {
            console.debug('Failed to start session regeneration prefetch:', error);
        });
}

// Initialize grading interface when section becomes active
function initializeGrading() {
    if (!currentSession) return;

    startSessionRegenerationPrefetch();
    loadProblemMaxPoints();
    loadProblemNumbers();
    setupGradingControls();
    updateOverallProgress();
    setupProblemImageResize();
}

// Ensure global access for navigation hooks.
window.initializeGrading = initializeGrading;

// Load max points metadata for all problems
async function loadProblemMaxPoints() {
    try {
        const response = await fetch(`${API_BASE}/sessions/${currentSession.id}/problem-max-points-all`);
        const data = await response.json();
        problemMaxPoints = data.max_points || {};
    } catch (error) {
        console.error('Failed to load max points metadata:', error);
        problemMaxPoints = {};
    }
}

// Show notification overlay
function showNotification(message, callback) {
    const overlay = document.getElementById('notification-overlay');
    const messageEl = document.getElementById('notification-message');
    const okBtn = document.getElementById('notification-ok');

    messageEl.textContent = message;
    overlay.style.display = 'flex';

    const dismiss = () => {
        overlay.style.display = 'none';
        document.removeEventListener('keydown', handleNotificationKey);
        if (callback) callback();
    };

    const handleNotificationKey = (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            dismiss();
        }
    };

    okBtn.onclick = dismiss;
    document.addEventListener('keydown', handleNotificationKey);

    // Focus the button for accessibility
    okBtn.focus();
}

// Update max points dropdown based on current problem number
function updateMaxPointsDropdown() {
    const maxPointsInput = document.getElementById('max-points-input');
    const scoreInput = document.getElementById('score-input');
    const cachedMax = problemMaxPoints[currentProblemNumber];

    // Default to 8 if not set
    const maxPoints = cachedMax || 8;

    maxPointsInput.value = maxPoints;
    scoreInput.max = maxPoints;
}

function cloneDefaultSubjectiveBuckets() {
    return DEFAULT_SUBJECTIVE_BUCKETS.map((bucket) => ({ ...bucket }));
}

function sanitizeSubjectiveBucketId(label, fallbackIndex = 1) {
    const slug = (label || '')
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '_')
        .replace(/^_+|_+$/g, '');
    return slug || `bucket_${fallbackIndex}`;
}

function getCurrentProblemMaxPoints() {
    const maxPointsInput = document.getElementById('max-points-input');
    const fromInput = parseFloat(maxPointsInput?.value || '');
    if (!Number.isNaN(fromInput) && fromInput > 0) {
        return fromInput;
    }
    const fromCache = parseFloat(problemMaxPoints[currentProblemNumber]);
    if (!Number.isNaN(fromCache) && fromCache > 0) {
        return fromCache;
    }
    return 8;
}

function updateProblemSelectUngradedCounts(problemStats = []) {
    const select = document.getElementById('problem-select');
    if (!select) return;

    const ungradedByProblem = {};
    if (Array.isArray(problemStats)) {
        problemStats.forEach((ps) => {
            const problemNumber = Number(ps.problem_number);
            if (Number.isNaN(problemNumber)) return;
            const numTotal = Number(ps.num_total || 0);
            const numGraded = Number(ps.num_graded || 0);
            ungradedByProblem[problemNumber] = Math.max(numTotal - numGraded, 0);
        });
    }

    [...select.options].forEach((option) => {
        const problemNumber = Number(option.value);
        if (Number.isNaN(problemNumber)) return;
        const ungradedCount = Number(ungradedByProblem[problemNumber] || 0);
        option.textContent = `Problem ${problemNumber} (${ungradedCount})`;
    });
}

async function loadSubjectiveSettings(problemNumber, force = false) {
    if (!currentSession || !problemNumber) return null;
    if (!force && subjectiveSettingsByProblem.has(problemNumber)) {
        currentSubjectiveSettings = subjectiveSettingsByProblem.get(problemNumber);
        return currentSubjectiveSettings;
    }

    try {
        const response = await fetch(`${API_BASE}/sessions/${currentSession.id}/subjective-settings/${problemNumber}`);
        if (!response.ok) {
            const message = await response.text();
            throw new Error(`Failed to load subjective settings: ${message}`);
        }
        const payload = await response.json();
        const settings = {
            problem_number: problemNumber,
            grading_mode: payload.grading_mode || 'calculation',
            buckets: Array.isArray(payload.buckets) && payload.buckets.length > 0 ? payload.buckets : cloneDefaultSubjectiveBuckets(),
            bucket_usage: payload.bucket_usage || {},
            triaged_count: payload.triaged_count || 0,
            finalized_count: payload.finalized_count || 0,
            untriaged_count: payload.untriaged_count || 0,
            total_count: payload.total_count || 0
        };
        subjectiveSettingsByProblem.set(problemNumber, settings);
        currentSubjectiveSettings = settings;
        return settings;
    } catch (error) {
        console.error('Failed to load subjective settings:', error);
        const fallback = {
            problem_number: problemNumber,
            grading_mode: 'calculation',
            buckets: cloneDefaultSubjectiveBuckets(),
            bucket_usage: {},
            triaged_count: 0,
            finalized_count: 0,
            untriaged_count: 0,
            total_count: 0
        };
        subjectiveSettingsByProblem.set(problemNumber, fallback);
        currentSubjectiveSettings = fallback;
        return fallback;
    }
}

function getCurrentSubjectiveSettings() {
    if (!currentSubjectiveSettings || Number(currentSubjectiveSettings.problem_number) !== Number(currentProblemNumber)) {
        const cached = subjectiveSettingsByProblem.get(Number(currentProblemNumber));
        if (cached) {
            currentSubjectiveSettings = cached;
        }
    }
    return currentSubjectiveSettings;
}

function toNonNegativeNumber(value, fallback = 0) {
    const parsed = Number(value);
    if (Number.isNaN(parsed) || !Number.isFinite(parsed)) {
        return Math.max(0, Number(fallback) || 0);
    }
    return Math.max(0, parsed);
}

function applyLocalSubjectiveStateUpdate({
    triagedCount = null,
    untriagedCount = null,
    finalizedCount = null,
    bucketUsage = null,
    previousBucketId = null,
    bucketId = null
} = {}) {
    const settings = getCurrentSubjectiveSettings();
    if (!settings || settings.grading_mode !== 'subjective') {
        return;
    }

    settings.triaged_count = toNonNegativeNumber(
        triagedCount,
        settings.triaged_count
    );
    settings.untriaged_count = toNonNegativeNumber(
        untriagedCount,
        settings.untriaged_count
    );
    settings.finalized_count = toNonNegativeNumber(
        finalizedCount,
        settings.finalized_count
    );

    if (bucketUsage && typeof bucketUsage === 'object') {
        settings.bucket_usage = { ...bucketUsage };
    } else {
        const usage = { ...(settings.bucket_usage || {}) };
        if (previousBucketId && previousBucketId !== bucketId) {
            usage[previousBucketId] = Math.max(
                0,
                toNonNegativeNumber(usage[previousBucketId], 0) - 1
            );
        }
        if (bucketId && previousBucketId !== bucketId) {
            usage[bucketId] = toNonNegativeNumber(usage[bucketId], 0) + 1;
        }
        settings.bucket_usage = usage;
    }

    if (currentProblem && currentProblem.grading_mode === 'subjective') {
        currentProblem.subjective_triaged_count = settings.triaged_count;
        currentProblem.subjective_untriaged_count = settings.untriaged_count;
    }

    subjectiveSettingsByProblem.set(Number(currentProblemNumber), settings);
    currentSubjectiveSettings = settings;
    renderSubjectiveBucketButtons();
    renderSubjectiveBucketFilter();
    renderSubjectiveBucketHistogram();
    updateSubjectiveFinalizeButton();
}

function renderSubjectiveBucketButtons() {
    const container = document.getElementById('subjective-bucket-buttons');
    const settings = getCurrentSubjectiveSettings();
    if (!container) return;

    container.innerHTML = '';
    if (!settings || !Array.isArray(settings.buckets)) return;

    settings.buckets.forEach((bucket) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'subjective-bucket-btn';
        if (selectedSubjectiveBucketId === bucket.id) {
            button.classList.add('selected');
        }
        if (bucket.color) {
            button.style.borderColor = bucket.color;
            if (selectedSubjectiveBucketId === bucket.id) {
                button.style.boxShadow = `0 0 0 2px ${bucket.color}33`;
            }
        }
        const usage = Number(settings.bucket_usage?.[bucket.id] || 0);
        button.textContent = usage > 0 ? `${bucket.label} (${usage})` : bucket.label;
        button.onclick = async () => {
            selectedSubjectiveBucketId = bucket.id;
            renderSubjectiveBucketButtons();
            if (settings.grading_mode === 'subjective' && currentProblem && !currentProblem.graded) {
                await submitSubjectiveTriage({ triggeredByBucketClick: true });
            }
        };
        container.appendChild(button);
    });
}

function renderSubjectiveBucketFilter() {
    const select = document.getElementById('subjective-view-filter');
    const settings = getCurrentSubjectiveSettings();
    if (!select || !settings) return;

    const previous = activeSubjectiveBucketFilter || '';
    select.innerHTML = '';

    const allOption = document.createElement('option');
    allOption.value = '';
    allOption.textContent = 'All (normal flow)';
    select.appendChild(allOption);

    settings.buckets.forEach((bucket) => {
        const usage = Number(settings.bucket_usage?.[bucket.id] || 0);
        if (usage <= 0) return;
        const option = document.createElement('option');
        option.value = bucket.id;
        option.textContent = `${bucket.label} (${usage})`;
        select.appendChild(option);
    });

    const hasPrevious = [...select.options].some((opt) => opt.value === previous);
    activeSubjectiveBucketFilter = hasPrevious ? previous : '';
    select.value = activeSubjectiveBucketFilter;
}

function renderSubjectiveBucketHistogram() {
    const container = document.getElementById('subjective-bucket-histogram');
    const settings = getCurrentSubjectiveSettings();
    if (!container || !settings || settings.grading_mode !== 'subjective') {
        if (container) container.innerHTML = '';
        return;
    }

    const buckets = settings.buckets || [];
    const usage = settings.bucket_usage || {};
    const triagedTotal = Math.max(Number(settings.triaged_count || 0), 0);
    const untriaged = Math.max(Number(settings.untriaged_count || 0), 0);
    const maxBucketCount = Math.max(
        1,
        ...buckets.map((bucket) => Number(usage[bucket.id] || 0))
    );
    const canAssignFromHistogram = Boolean(
        settings.grading_mode === 'subjective' &&
        currentProblem &&
        !currentProblem.graded
    );

    container.innerHTML = '';

    const title = document.createElement('div');
    title.style.cssText = 'font-size: 12px; color: var(--gray-700); margin-bottom: 6px; font-weight: 600;';
    title.textContent = canAssignFromHistogram
        ? 'Bucket Distribution (click row to assign)'
        : 'Bucket Distribution (triaged responses)';
    container.appendChild(title);

    buckets.forEach((bucket) => {
        const count = Number(usage[bucket.id] || 0);
        const relativePct = (count / maxBucketCount) * 100;
        const sharePct = triagedTotal > 0 ? (count / triagedTotal) * 100 : 0;

        const row = document.createElement('div');
        row.className = 'subjective-histogram-row';
        if (canAssignFromHistogram) {
            row.classList.add('clickable');
        }
        if (selectedSubjectiveBucketId === bucket.id) {
            row.classList.add('selected');
        }
        row.dataset.bucketId = bucket.id;

        const label = document.createElement('div');
        label.textContent = `${bucket.label} (${sharePct.toFixed(0)}%)`;

        const track = document.createElement('div');
        track.className = 'subjective-histogram-track';
        const fill = document.createElement('div');
        fill.className = 'subjective-histogram-fill';
        fill.style.width = `${Math.max(0, Math.min(100, relativePct))}%`;
        if (bucket.color) fill.style.background = bucket.color;
        track.appendChild(fill);

        const value = document.createElement('div');
        value.style.textAlign = 'right';
        value.textContent = String(count);

        row.appendChild(label);
        row.appendChild(track);
        row.appendChild(value);

        if (canAssignFromHistogram) {
            row.onclick = async () => {
                selectedSubjectiveBucketId = bucket.id;
                renderSubjectiveBucketHistogram();
                await submitSubjectiveTriage({ triggeredByBucketClick: true });
            };
        }

        container.appendChild(row);
    });

    if (untriaged > 0) {
        const remaining = document.createElement('div');
        remaining.style.cssText = 'margin-top: 4px; font-size: 12px; color: var(--gray-700);';
        remaining.textContent = `Remaining untriaged: ${untriaged}`;
        container.appendChild(remaining);
    }
}

function renderSubjectiveBucketEditor() {
    const editor = document.getElementById('subjective-bucket-editor');
    const settings = getCurrentSubjectiveSettings();
    if (!editor || !settings) return;

    editor.innerHTML = '';
    settings.buckets.forEach((bucket, index) => {
        const row = document.createElement('div');
        row.className = 'subjective-bucket-row';
        row.dataset.bucketId = bucket.id;

        const labelInput = document.createElement('input');
        labelInput.type = 'text';
        labelInput.value = bucket.label || '';
        labelInput.placeholder = 'Bucket label';

        const colorInput = document.createElement('input');
        colorInput.type = 'text';
        colorInput.value = bucket.color || '';
        colorInput.placeholder = '#hex color';

        const removeBtn = document.createElement('button');
        removeBtn.type = 'button';
        removeBtn.className = 'btn btn-secondary btn-small';
        removeBtn.textContent = 'Remove';
        removeBtn.onclick = () => {
            settings.buckets.splice(index, 1);
            if (selectedSubjectiveBucketId === bucket.id) {
                selectedSubjectiveBucketId = null;
            }
            renderSubjectiveBucketEditor();
            renderSubjectiveBucketButtons();
        };

        row.appendChild(labelInput);
        row.appendChild(colorInput);
        row.appendChild(removeBtn);
        editor.appendChild(row);
    });
}

function collectBucketsFromEditor() {
    const editor = document.getElementById('subjective-bucket-editor');
    if (!editor) return [];
    const rows = [...editor.querySelectorAll('.subjective-bucket-row')];
    const usedIds = new Set();

    return rows.map((row, idx) => {
        const labelInput = row.querySelector('input[type="text"]');
        const colorInput = row.querySelectorAll('input[type="text"]')[1];
        const label = (labelInput?.value || '').trim();
        let bucketId = (row.dataset.bucketId || '').trim();
        if (!bucketId) {
            bucketId = sanitizeSubjectiveBucketId(label, idx + 1);
        }
        while (usedIds.has(bucketId)) {
            bucketId = `${bucketId}_${idx + 1}`;
        }
        usedIds.add(bucketId);
        row.dataset.bucketId = bucketId;
        return {
            id: bucketId,
            label: label || `Bucket ${idx + 1}`,
            color: (colorInput?.value || '').trim() || null
        };
    });
}

async function saveSubjectiveSettings() {
    const settings = getCurrentSubjectiveSettings();
    if (!settings || !currentSession || !currentProblemNumber) return;

    let buckets = collectBucketsFromEditor();
    // When switching into subjective mode, the editor may not be rendered yet.
    // Fall back to in-memory settings/defaults so mode toggle can persist cleanly.
    if (buckets.length === 0) {
        if (Array.isArray(settings.buckets) && settings.buckets.length > 0) {
            buckets = settings.buckets.map((bucket, index) => ({
                id: bucket.id || sanitizeSubjectiveBucketId(bucket.label || '', index + 1),
                label: (bucket.label || '').trim() || `Bucket ${index + 1}`,
                color: (bucket.color || null)
            }));
        } else if (settings.grading_mode === 'subjective') {
            buckets = cloneDefaultSubjectiveBuckets();
        }
    }
    if (buckets.length === 0 && settings.grading_mode === 'subjective') {
        alert('At least one bucket is required in subjective mode. Open "Manage Buckets" to add one.');
        return;
    }

    const response = await fetch(`${API_BASE}/sessions/${currentSession.id}/subjective-settings`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            problem_number: Number(currentProblemNumber),
            grading_mode: settings.grading_mode,
            buckets
        })
    });

    let payload = null;
    try {
        payload = await response.json();
    } catch (_error) {
        payload = null;
    }

    if (!response.ok) {
        throw new Error(payload?.detail || 'Failed to save subjective settings');
    }

    const updated = {
        problem_number: Number(currentProblemNumber),
        grading_mode: payload.grading_mode || settings.grading_mode,
        buckets: payload.buckets || buckets,
        bucket_usage: payload.bucket_usage || {},
        triaged_count: payload.triaged_count || 0,
        finalized_count: payload.finalized_count || 0,
        untriaged_count: payload.untriaged_count || 0,
        total_count: payload.total_count || settings.total_count || 0
    };
    subjectiveSettingsByProblem.set(Number(currentProblemNumber), updated);
    currentSubjectiveSettings = updated;

    if (selectedSubjectiveBucketId && !updated.buckets.some((b) => b.id === selectedSubjectiveBucketId)) {
        selectedSubjectiveBucketId = null;
    }
    renderSubjectiveBucketEditor();
    renderSubjectiveBucketButtons();
    renderSubjectiveBucketFilter();
    renderSubjectiveBucketHistogram();
    updateSubjectiveFinalizeButton();
}

function applyGradingModeUI() {
    const settings = getCurrentSubjectiveSettings();
    const gradingModeSelect = document.getElementById('grading-mode-select');
    const calculationRow = document.getElementById('calculation-input-row');
    const subjectivePanel = document.getElementById('subjective-grading-panel');
    const submitGradeBtn = document.getElementById('submit-grade-btn');
    const subjectiveAssignBtn = document.getElementById('subjective-assign-btn');
    const subjectiveClearBtn = document.getElementById('subjective-clear-btn');
    const subjectiveFinalizeBtn = document.getElementById('subjective-finalize-btn');
    const subjectiveReopenBtn = document.getElementById('subjective-reopen-btn');

    const isSubjective = settings?.grading_mode === 'subjective';
    if (gradingModeSelect) {
        gradingModeSelect.value = isSubjective ? 'subjective' : 'calculation';
    }
    if (calculationRow) {
        calculationRow.style.display = isSubjective ? 'none' : 'flex';
    }
    if (subjectivePanel) {
        subjectivePanel.style.display = isSubjective ? 'block' : 'none';
    }
    if (submitGradeBtn) {
        submitGradeBtn.style.display = isSubjective ? 'none' : 'inline-block';
    }
    if (subjectiveAssignBtn) {
        subjectiveAssignBtn.style.display = isSubjective ? 'inline-block' : 'none';
    }
    if (subjectiveClearBtn) {
        subjectiveClearBtn.style.display = isSubjective ? 'inline-block' : 'none';
    }
    if (subjectiveFinalizeBtn && !isSubjective) {
        subjectiveFinalizeBtn.style.display = 'none';
    }
    if (subjectiveReopenBtn && !isSubjective) {
        subjectiveReopenBtn.style.display = 'none';
    }

    if (isSubjective) {
        renderSubjectiveBucketButtons();
        renderSubjectiveBucketFilter();
        renderSubjectiveBucketHistogram();
        renderSubjectiveBucketEditor();
    }
    updateSubjectiveFinalizeButton();
}

function updateSubjectiveFinalizeButton() {
    const finalizeBtn = document.getElementById('subjective-finalize-btn');
    const reopenBtn = document.getElementById('subjective-reopen-btn');
    const settings = getCurrentSubjectiveSettings();
    if (!finalizeBtn || !reopenBtn) return;

    if (!settings || settings.grading_mode !== 'subjective') {
        finalizeBtn.style.display = 'none';
        finalizeBtn.disabled = true;
        finalizeBtn.title = '';
        reopenBtn.style.display = 'none';
        reopenBtn.disabled = true;
        reopenBtn.title = '';
        return;
    }

    const untriagedCount = Number(settings.untriaged_count || 0);
    const triagedCount = Number(settings.triaged_count || 0);
    const finalizedCount = Number(settings.finalized_count || 0);
    const canFinalize = untriagedCount === 0 && triagedCount > 0;
    const canReopen = finalizedCount > 0;

    finalizeBtn.style.display = canFinalize ? 'inline-block' : 'none';
    finalizeBtn.disabled = !canFinalize;
    if (untriagedCount > 0) {
        finalizeBtn.title = `${untriagedCount} responses still need bucket assignments`;
    } else if (triagedCount === 0) {
        finalizeBtn.title = 'No triaged responses to finalize';
    } else {
        finalizeBtn.title = '';
    }

    reopenBtn.style.display = canReopen ? 'inline-block' : 'none';
    reopenBtn.disabled = !canReopen;
    if (canReopen) {
        reopenBtn.title = `${finalizedCount} finalized responses can be reopened`;
    } else {
        reopenBtn.title = '';
    }
}

// Load available problem numbers
async function loadProblemNumbers() {
    try {
        const [numbersResponse, statsResponse] = await Promise.all([
            fetch(`${API_BASE}/sessions/${currentSession.id}/problem-numbers`),
            fetch(`${API_BASE}/sessions/${currentSession.id}/stats`)
        ]);

        if (!numbersResponse.ok) {
            const message = await numbersResponse.text();
            console.error('Failed to load problem numbers:', numbersResponse.status, message);
            return;
        }

        const data = await numbersResponse.json();
        let stats = { problem_stats: [] };
        if (statsResponse.ok) {
            stats = await statsResponse.json();
        } else {
            const message = await statsResponse.text();
            console.warn('Failed to load stats for problem counts:', statsResponse.status, message);
        }
        availableProblemNumbers = data.problem_numbers;

        const select = document.getElementById('problem-select');
        select.innerHTML = '';

        availableProblemNumbers.forEach(num => {
            const option = document.createElement('option');
            option.value = num;
            option.textContent = `Problem ${num} (0)`;
            select.appendChild(option);
        });
        updateProblemSelectUngradedCounts(stats.problem_stats);

        currentProblemNumber = availableProblemNumbers[0] || 1;
        activeSubjectiveBucketFilter = '';
        invalidateNextProblemPrefetch();
        select.value = currentProblemNumber;
        await loadSubjectiveSettings(currentProblemNumber, true);
        applyGradingModeUI();
        select.onchange = async () => {
            currentProblemNumber = parseInt(select.value);
            activeSubjectiveBucketFilter = '';
            invalidateNextProblemPrefetch();
            await loadSubjectiveSettings(currentProblemNumber, true);
            applyGradingModeUI();
            updateMaxPointsDropdown();
            const progressPromise = updateOverallProgress(); // Update progress bar when changing problems
            await loadProblemOrMostRecent();
            await progressPromise;
        };

        loadNextProblem();
    } catch (error) {
        console.error('Failed to load problem numbers:', error);
    }
}

// Update overall progress display
async function updateOverallProgress() {
    try {
        const response = await fetch(`${API_BASE}/sessions/${currentSession.id}/stats`);
        if (!response.ok) {
            const message = await response.text();
            console.error('Failed to load stats for progress:', response.status, message);
            return;
        }

        const stats = await response.json();
        updateProblemSelectUngradedCounts(stats.problem_stats);
        const clampPercent = (value) => Math.max(0, Math.min(100, value));

        const percentage = clampPercent(stats.progress_percentage || 0);
        document.getElementById('overall-progress-label').textContent =
            `Overall: ${stats.problems_graded} / ${stats.total_problems} (${percentage.toFixed(1)}%)`;

        const sortedProblemStats = [...stats.problem_stats]
            .sort((a, b) => a.problem_number - b.problem_number);
        const segmentContainer = document.getElementById('overall-problem-segments');
        const currentProblemWindow = document.getElementById('current-problem-window');
        const currentProblemWindowGraded = document.getElementById('current-problem-window-graded');
        const currentProblemWindowUngraded = document.getElementById('current-problem-window-ungraded');
        const currentProblemWindowBlank = document.getElementById('current-problem-window-blank');
        segmentContainer.innerHTML = '';

        if (stats.total_problems > 0) {
            let cumulativeTotal = 0;
            sortedProblemStats.forEach((ps) => {
                if (!ps || ps.num_total <= 0) {
                    return;
                }

                const segmentLeftPercent = clampPercent((cumulativeTotal / stats.total_problems) * 100);
                const segmentWidthPercent = clampPercent((ps.num_total / stats.total_problems) * 100);
                const gradedCount = Math.min(Math.max(ps.num_graded || 0, 0), ps.num_total);
                const gradedPercentWithin = clampPercent((gradedCount / ps.num_total) * 100);

                const segment = document.createElement('div');
                segment.className = 'problem-progress-segment';
                segment.style.left = `${segmentLeftPercent}%`;
                segment.style.width = `${segmentWidthPercent}%`;

                const segmentFill = document.createElement('div');
                segmentFill.className = 'problem-progress-segment-fill';
                segmentFill.style.width = `${gradedPercentWithin}%`;

                segment.appendChild(segmentFill);
                segmentContainer.appendChild(segment);

                cumulativeTotal += ps.num_total;
            });
        }

        const clearCurrentProblemWindow = () => {
            currentProblemWindow.style.width = '0%';
            currentProblemWindow.style.left = '0%';
            currentProblemWindowGraded.style.width = '0%';
            currentProblemWindowUngraded.style.width = '0%';
            currentProblemWindowBlank.style.width = '0%';
        };

        const currentProblemStats = sortedProblemStats.find(
            (p) => p.problem_number === Number(currentProblemNumber)
        );
        if (!currentProblemStats || stats.total_problems <= 0 || currentProblemStats.num_total <= 0) {
            clearCurrentProblemWindow();
            return;
        }

        // Current-problem window position uses total work distribution, not graded work.
        const totalBeforeCurrent = sortedProblemStats
            .filter((ps) => ps.problem_number < Number(currentProblemNumber))
            .reduce((sum, ps) => sum + ps.num_total, 0);
        const currentProblemStartPercent = clampPercent((totalBeforeCurrent / stats.total_problems) * 100);
        const currentProblemWidthPercent = clampPercent((currentProblemStats.num_total / stats.total_problems) * 100);

        const gradedCount = Math.min(
            Math.max(currentProblemStats.num_graded || 0, 0),
            currentProblemStats.num_total
        );
        const ungradedCount = Math.max(currentProblemStats.num_total - gradedCount, 0);
        const blankUngradedCount = Math.min(
            Math.max(currentProblemStats.num_blank_ungraded || 0, 0),
            ungradedCount
        );

        const gradedWithinCurrentPercent = clampPercent((gradedCount / currentProblemStats.num_total) * 100);
        const ungradedWithinCurrentPercent = clampPercent((ungradedCount / currentProblemStats.num_total) * 100);
        const blankWithinCurrentPercent = clampPercent((blankUngradedCount / currentProblemStats.num_total) * 100);

        currentProblemWindow.style.left = `${currentProblemStartPercent}%`;
        currentProblemWindow.style.width = `${currentProblemWidthPercent}%`;
        currentProblemWindowGraded.style.width = `${gradedWithinCurrentPercent}%`;
        currentProblemWindowUngraded.style.width = `${ungradedWithinCurrentPercent}%`;
        currentProblemWindowBlank.style.width = `${blankWithinCurrentPercent}%`;
    } catch (error) {
        console.error('Failed to update overall progress:', error);
    }
}

// Upload more exams button
document.getElementById('upload-more-btn').addEventListener('click', () => {
    if (!currentSession) return;
    // Navigate back to upload section with currentSession still set
    navigateToSection('upload-section');
    // Show a message that we're adding to existing session
    document.getElementById('initial-upload-message').style.display = 'block';
    document.getElementById('initial-upload-message').innerHTML =
        `<strong>Adding exams to:</strong> ${currentSession.assignment_name} - ${currentSession.course_name || `Course ${currentSession.course_id}`}`;
});

// Setup score sync between slider and input (slider removed, keeping function for compatibility)
function setupScoreSync() {
    // Slider has been removed - this function is now a no-op
    // Kept for compatibility with existing calls
}

// Setup grading controls
function setupGradingControls() {
    document.getElementById('submit-grade-btn').onclick = submitGrade;
    document.getElementById('subjective-assign-btn').onclick = submitSubjectiveTriage;
    document.getElementById('subjective-clear-btn').onclick = clearSubjectiveTriage;
    document.getElementById('subjective-finalize-btn').onclick = openSubjectiveFinalizeDialog;
    document.getElementById('subjective-reopen-btn').onclick = submitSubjectiveReopen;
    document.getElementById('next-problem-btn').onclick = loadNextProblem;
    document.getElementById('back-problem-btn').onclick = loadPreviousProblem;
    document.getElementById('view-stats-btn').onclick = () => {
        navigateToSection('stats-section');
        loadStatistics();
    };

    // Continue grading button (in stats section)
    document.getElementById('continue-grading-btn').onclick = () => {
        navigateToSection('grading-section');
    };

    // Initial score sync setup
    setupScoreSync();

    // Max points input handler
    const maxPointsInput = document.getElementById('max-points-input');
    maxPointsInput.addEventListener('change', async (e) => {
        const maxPoints = parseFloat(e.target.value);
        if (!isNaN(maxPoints) && maxPoints > 0 && currentProblemNumber) {
            // Update input max
            document.getElementById('score-input').max = maxPoints;

            // Save to cache
            problemMaxPoints[currentProblemNumber] = maxPoints;

            // Save to backend
            try {
                const response = await fetch(`${API_BASE}/sessions/${currentSession.id}/problem-max-points?problem_number=${currentProblemNumber}&max_points=${maxPoints}`, {
                    method: 'PUT'
                });

                if (!response.ok) {
                    throw new Error('Failed to save max points');
                }

                // Update current problem object
                if (currentProblem) {
                    currentProblem.max_points = maxPoints;
                }
            } catch (error) {
                console.error('Failed to save max points:', error);
                alert('Failed to save max points: ' + error.message);
            }
        }
    });

    const gradingModeSelect = document.getElementById('grading-mode-select');
    gradingModeSelect.addEventListener('change', async (e) => {
        const mode = e.target.value === 'subjective' ? 'subjective' : 'calculation';
        const settings = getCurrentSubjectiveSettings() || {
            problem_number: Number(currentProblemNumber),
            grading_mode: mode,
            buckets: cloneDefaultSubjectiveBuckets(),
            bucket_usage: {}
        };
        settings.grading_mode = mode;
        if (mode === 'subjective' && (!Array.isArray(settings.buckets) || settings.buckets.length === 0)) {
            settings.buckets = cloneDefaultSubjectiveBuckets();
        }
        currentSubjectiveSettings = settings;
        subjectiveSettingsByProblem.set(Number(currentProblemNumber), settings);
        applyGradingModeUI();

        try {
            await saveSubjectiveSettings();
            await loadSubjectiveSettings(Number(currentProblemNumber), true);
            applyGradingModeUI();
            invalidateNextProblemPrefetch();
            await loadProblemOrMostRecent();
        } catch (error) {
            console.error('Failed to save grading mode:', error);
            alert(error.message || 'Failed to save grading mode');
            await loadSubjectiveSettings(Number(currentProblemNumber), true);
            applyGradingModeUI();
        }
    });

    document.getElementById('subjective-add-bucket-btn').addEventListener('click', () => {
        const settings = getCurrentSubjectiveSettings();
        if (!settings) return;
        const nextIndex = settings.buckets.length + 1;
        settings.buckets.push({
            id: `bucket_${Date.now()}_${nextIndex}`,
            label: `Bucket ${nextIndex}`,
            color: null
        });
        renderSubjectiveBucketEditor();
        renderSubjectiveBucketButtons();
    });

    document.getElementById('subjective-save-settings-btn').addEventListener('click', async () => {
        try {
            await saveSubjectiveSettings();
            showNotification('Subjective bucket settings saved.');
        } catch (error) {
            console.error('Failed to save subjective settings:', error);
            alert(error.message || 'Failed to save subjective settings');
        }
    });

    const subjectiveFilterSelect = document.getElementById('subjective-view-filter');
    if (subjectiveFilterSelect) {
        subjectiveFilterSelect.addEventListener('change', async (e) => {
            activeSubjectiveBucketFilter = e.target.value || '';
            invalidateNextProblemPrefetch();
            if (!activeSubjectiveBucketFilter) {
                await loadProblemOrMostRecent();
                return;
            }
            try {
                await loadProblemFromActiveBucketFilter('next', true);
            } catch (error) {
                console.error('Failed to apply bucket filter:', error);
                alert('Failed to load filtered responses: ' + error.message);
            }
        });
    }

    document.getElementById('subjective-finalize-cancel-btn').onclick = closeSubjectiveFinalizeDialog;
    document.getElementById('subjective-finalize-confirm-btn').onclick = submitSubjectiveFinalize;

    // Keyboard shortcuts
    document.addEventListener('keydown', handleGradingKeyboard);
}

// Handle keyboard shortcuts for grading
function handleGradingKeyboard(e) {
    // Only handle when grading section is active
    if (!document.getElementById('grading-section').classList.contains('active')) {
        return;
    }

    // Don't handle if notification overlay is visible
    const notificationOverlay = document.getElementById('notification-overlay');
    if (notificationOverlay && notificationOverlay.style.display === 'flex') {
        return;
    }

    // Don't handle if default feedback dialog is open
    const defaultFeedbackDialog = document.getElementById('edit-default-feedback-dialog');
    if (defaultFeedbackDialog && defaultFeedbackDialog.style.display === 'flex') {
        return;
    }

    // Don't handle if add tag dialog is open
    const addTagDialog = document.getElementById('add-tag-dialog');
    if (addTagDialog && addTagDialog.style.display === 'flex') {
        return;
    }

    const manualQrDialog = document.getElementById('manual-qr-dialog');
    if (manualQrDialog && manualQrDialog.style.display === 'flex') {
        return;
    }

    const subjectiveFinalizeDialog = document.getElementById('subjective-finalize-dialog');
    if (subjectiveFinalizeDialog && subjectiveFinalizeDialog.style.display === 'flex') {
        return;
    }

    // Don't handle if typing in textarea
    if (e.target.tagName === 'TEXTAREA') {
        return;
    }

    // Enter key - submit and move to next
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (getCurrentSubjectiveSettings()?.grading_mode === 'subjective') {
            submitSubjectiveTriage();
        } else {
            submitGrade();
        }
    }

    // Number keys 0-9 or dash (-) - quick score entry (but not when typing in rubric table or other inputs)
    // Also ignore if any modifier keys are pressed (Cmd/Ctrl/Alt for browser shortcuts like zoom)
    if ((/^[0-9]$/.test(e.key) || e.key === '-') &&
        !e.metaKey && !e.ctrlKey && !e.altKey && // Ignore if modifier keys are held
        e.target.id !== 'score-input' &&
        e.target.id !== 'feedback-input' &&
        e.target.id !== 'max-points-input' &&
        !e.target.classList.contains('rubric-points') &&
        !e.target.classList.contains('rubric-description')) {
        e.preventDefault();
        document.getElementById('score-input').value = e.key;
        document.getElementById('score-input').focus();
    }
}

// Display the current problem (common display logic)
function displayCurrentProblem() {
    if (!currentProblem) return;

    // Display problem
    const problemImage = document.getElementById('problem-image');
    problemImage.src = `data:image/png;base64,${currentProblem.image_data}`;

    // Auto-size container to fit image when it loads
    problemImage.onload = () => {
        const scrollContainer = document.getElementById('problem-scroll-container');
        if (scrollContainer) {
            // Calculate the full displayed height of the image
            const displayedHeight = problemImage.offsetHeight;
            const fullImageHeight = displayedHeight + 40; // Add padding for borders/margins

            // Store this as the maximum allowed height (so user can always expand to see full image)
            scrollContainer.dataset.maxImageHeight = fullImageHeight;

            // Check if we have a saved height preference
            const savedHeight = localStorage.getItem('problemScrollContainerHeight');
            if (!savedHeight) {
                // No saved preference - default to showing full image
                scrollContainer.style.height = `${fullImageHeight}px`;
            }
            // If there's a saved height, the setupProblemImageResize() function already applied it
        }
    };

    const settings = getCurrentSubjectiveSettings();
    if (settings) {
        settings.grading_mode = currentProblem.grading_mode || settings.grading_mode || 'calculation';
        if (settings.grading_mode === 'subjective') {
            // Keep local subjective counters authoritative to avoid regressions from
            // stale prefetched payloads.
            const triagedCount = toNonNegativeNumber(
                settings.triaged_count,
                currentProblem.subjective_triaged_count || 0
            );
            const untriagedCount = toNonNegativeNumber(
                settings.untriaged_count,
                currentProblem.subjective_untriaged_count || 0
            );
            settings.triaged_count = triagedCount;
            settings.untriaged_count = untriagedCount;
            currentProblem.subjective_triaged_count = triagedCount;
            currentProblem.subjective_untriaged_count = untriagedCount;
        } else {
            settings.triaged_count = 0;
            settings.untriaged_count = 0;
        }
        settings.total_count = currentProblem.total_count || settings.total_count || 0;
        subjectiveSettingsByProblem.set(Number(currentProblemNumber), settings);
        currentSubjectiveSettings = settings;
    }
    selectedSubjectiveBucketId = currentProblem.subjective_bucket_id || null;
    const subjectiveNotesInput = document.getElementById('subjective-notes-input');
    if (subjectiveNotesInput) {
        subjectiveNotesInput.value = currentProblem.subjective_notes || '';
    }

    // Update progress with blank count
    let progressText = `${currentProblem.current_index} / ${currentProblem.total_count}`;
    if (currentProblem.grading_mode === 'subjective') {
        progressText = `${currentProblem.subjective_triaged_count} triaged / ${currentProblem.total_count}`;
        if (currentProblem.subjective_untriaged_count > 0) {
            progressText += ` (${currentProblem.subjective_untriaged_count} remaining)`;
        }
    } else if (currentProblem.ungraded_blank > 0 || currentProblem.ungraded_nonblank > 0) {
        if (currentProblem.ungraded_blank > 0) {
            progressText += ` (${currentProblem.ungraded_blank} blank)`;
        }
    }

    document.getElementById('grading-progress').textContent = progressText;
    applyGradingModeUI();

    // Update max points from cache
    updateMaxPointsDropdown();

    // Re-attach event listeners
    setupScoreSync();

    // Show/hide "Show Answer" button based on QR data availability
    const showAnswerBtn = document.getElementById('show-answer-btn');
    if (currentProblem.has_qr_data) {
        showAnswerBtn.style.display = 'inline-block';
    } else {
        showAnswerBtn.style.display = 'none';
    }

    // Update answer dialog if it's currently visible
    const answerDialog = document.getElementById('answer-dialog');
    if (answerDialog && answerDialog.style.display === 'flex') {
        updateAnswerDialog();
    }

    // Update transcription dialog if it's currently visible
    const transcriptionDialog = document.getElementById('transcription-dialog');
    if (transcriptionDialog && transcriptionDialog.style.display === 'flex') {
        updateTranscriptionDialog();
    }

    // Populate form based on whether it's graded or blank
    if (currentProblem.graded) {
        // Already graded - show existing grade
        document.getElementById('score-input').value = currentProblem.score != null ? currentProblem.score : '';
        document.getElementById('feedback-input').value = currentProblem.feedback || '';

        // Remove blank indicator
        const oldBlankIndicator = document.getElementById('blank-indicator');
        if (oldBlankIndicator) oldBlankIndicator.remove();

        // Remove AI indicator
        const oldAiIndicator = document.getElementById('ai-graded-indicator');
        if (oldAiIndicator) oldAiIndicator.remove();
    } else if (currentProblem.score != null && !currentProblem.graded) {
        // AI-graded suggestion should override heuristic blank flag
        document.getElementById('score-input').value = currentProblem.score != null ? currentProblem.score : '';
        document.getElementById('feedback-input').value = currentProblem.feedback || '';

        // Remove blank indicator
        const oldBlankIndicator = document.getElementById('blank-indicator');
        if (oldBlankIndicator) oldBlankIndicator.remove();

        // Show AI-graded indicator
        const aiIndicator = document.createElement('div');
        aiIndicator.id = 'ai-graded-indicator';
        aiIndicator.style.cssText = `
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
        `;
        aiIndicator.innerHTML = `
            <strong>🤖 AI-Graded (Needs Review)</strong>
            <div style="font-size: 12px; margin-top: 5px; opacity: 0.9;">
                Review and modify the score and feedback as needed, then submit
            </div>
        `;

        const oldAiIndicator = document.getElementById('ai-graded-indicator');
        if (oldAiIndicator) oldAiIndicator.remove();

        const indicatorContainer = document.getElementById('grading-indicators');
        if (indicatorContainer) {
            indicatorContainer.appendChild(aiIndicator);
        }
    } else if (currentProblem.is_blank) {
        const isAiBlank = currentProblem.blank_method === 'ai' || currentProblem.feedback;
        if (isAiBlank) {
            document.getElementById('score-input').value = '-';
            document.getElementById('feedback-input').value = currentProblem.feedback || '';
        } else {
            // Don't auto-populate score for heuristically detected blanks - let user verify
            document.getElementById('score-input').value = '';
            document.getElementById('feedback-input').value = '';
        }

        // Show blank detection indicator
        const blankIndicator = document.createElement('div');
        blankIndicator.id = 'blank-indicator';
        blankIndicator.className = 'blank-indicator';
        blankIndicator.innerHTML = `
            <strong>⚠️ Blank Detected</strong>
            <div style="font-size: 12px; margin-top: 5px;">
                Confidence: ${(currentProblem.blank_confidence * 100).toFixed(0)}%
                (${currentProblem.blank_method || 'heuristic'})
            </div>
        `;

        // Remove old indicator if exists
        const oldIndicator = document.getElementById('blank-indicator');
        if (oldIndicator) oldIndicator.remove();

        // Insert in the grading indicators area
        const indicatorContainer = document.getElementById('grading-indicators');
        if (indicatorContainer) {
            indicatorContainer.appendChild(blankIndicator);
        }
    } else {
        // Remove blank indicator if it exists
        const oldBlankIndicator = document.getElementById('blank-indicator');
        if (oldBlankIndicator) oldBlankIndicator.remove();

        // Clear form for non-AI-graded problems
        document.getElementById('score-input').value = '';
        document.getElementById('feedback-input').value = '';

        // Remove AI indicator if it exists
        const oldAiIndicator = document.getElementById('ai-graded-indicator');
        if (oldAiIndicator) oldAiIndicator.remove();
    }

    // Load feedback tags and default feedback for this problem number
    if (currentSession && currentProblemNumber) {
        loadFeedbackTags(currentSession.id, currentProblemNumber);
        loadDefaultFeedback(currentSession.id, currentProblemNumber);
        loadAiGradingNotes(currentSession.id, currentProblemNumber);
    }

    // Set explanation UI state (avoid eager regeneration fetch on every navigation)
    loadExplanation({ eager: false });
}

// Load problem for current problem number (ungraded if available, otherwise most recent)
async function loadProblemOrMostRecent() {
    try {
        if (shouldUseSubjectiveBucketFilter()) {
            const loaded = await loadProblemFromActiveBucketFilter('next', true);
            if (loaded) return;
        }

        // Try to load next ungraded problem first
        const nextResponse = await fetch(
            `${API_BASE}/problems/${currentSession.id}/${currentProblemNumber}/next`
        );

        if (nextResponse.ok) {
            // Found an ungraded problem, load it directly
            currentProblem = await nextResponse.json();
            await loadSubjectiveSettings(currentProblemNumber);
            addToHistory(currentProblem);
            displayCurrentProblem();
            triggerNextProblemPrefetch();
        } else if (nextResponse.status === 404) {
            // No ungraded problems, load most recently graded
            const prevResponse = await fetch(
                `${API_BASE}/problems/${currentSession.id}/${currentProblemNumber}/previous`
            );

            if (prevResponse.ok) {
                currentProblem = await prevResponse.json();
                await loadSubjectiveSettings(currentProblemNumber);
                addToHistory(currentProblem);
                displayCurrentProblem();
                triggerNextProblemPrefetch();
            } else {
                alert('No problems found for this problem number');
            }
        }
    } catch (error) {
        console.error('Failed to load problem:', error);
        alert('Failed to load problem: ' + error.message);
    }
}

function shouldUseSubjectiveBucketFilter() {
    const settings = getCurrentSubjectiveSettings();
    return Boolean(
        settings &&
        settings.grading_mode === 'subjective' &&
        activeSubjectiveBucketFilter
    );
}

async function loadProblemFromActiveBucketFilter(direction = 'next', reset = false) {
    if (!currentSession || !currentProblemNumber || !activeSubjectiveBucketFilter) return false;
    invalidateNextProblemPrefetch();

    const endpoint = direction === 'previous' ? 'previous' : 'next';
    const params = new URLSearchParams();
    if (!reset && currentProblem?.id) {
        params.set('current_problem_id', String(currentProblem.id));
    }
    const query = params.toString() ? `?${params.toString()}` : '';
    const url = `${API_BASE}/problems/${currentSession.id}/${currentProblemNumber}/bucket/${encodeURIComponent(activeSubjectiveBucketFilter)}/${endpoint}${query}`;

    const response = await fetch(url);
    if (response.status === 404) {
        showNotification(`No responses found in bucket "${activeSubjectiveBucketFilter}" for this problem.`);
        return false;
    }
    if (!response.ok) {
        const text = await response.text();
        throw new Error(text || 'Failed to load bucket-filtered response');
    }

    currentProblem = await response.json();
    await loadSubjectiveSettings(currentProblemNumber);
    addToHistory(currentProblem);
    displayCurrentProblem();
    return true;
}

// Add problem to history
function addToHistory(problem) {
    // If we're in the middle of history, remove everything after current position
    if (historyIndex < problemHistory.length - 1) {
        problemHistory = problemHistory.slice(0, historyIndex + 1);
    }

    // Add new problem to history
    problemHistory.push(problem);
    historyIndex = problemHistory.length - 1;

    // Limit history to last 50 problems to avoid memory issues
    if (problemHistory.length > 50) {
        problemHistory.shift();
        historyIndex--;
    }
}

// Load previous problem from history
async function loadPreviousProblem() {
    if (shouldUseSubjectiveBucketFilter()) {
        try {
            await loadProblemFromActiveBucketFilter('previous', false);
        } catch (error) {
            console.error('Failed to load previous bucket-filtered problem:', error);
            alert('Failed to load previous problem: ' + error.message);
        }
        return;
    }

    if (historyIndex > 0) {
        // Go back in history
        historyIndex--;
        const cachedProblem = problemHistory[historyIndex];
        currentProblem = cachedProblem;

        // History snapshots can be stale (e.g., after grading/feedback edits).
        // Refresh from API so Back always shows the latest saved state.
        if (cachedProblem && cachedProblem.id) {
            try {
                const freshResponse = await fetch(`${API_BASE}/problems/${cachedProblem.id}`);
                if (freshResponse.ok) {
                    const freshProblem = await freshResponse.json();
                    problemHistory[historyIndex] = freshProblem;
                    currentProblem = freshProblem;
                }
            } catch (error) {
                console.warn('Failed to refresh historical problem state, using cached snapshot:', error);
            }
        }

        if (currentProblem && currentProblem.problem_number) {
            currentProblemNumber = currentProblem.problem_number;
            document.getElementById('problem-select').value = currentProblemNumber;
            await loadSubjectiveSettings(currentProblemNumber);
            applyGradingModeUI();
        }
        displayCurrentProblem();
    } else {
        alert('No more previous problems in history');
    }
}

// Find next problem number with ungraded submissions
async function findNextUngradedProblem() {
    // Check each problem number to see if it has ungraded submissions
    for (const problemNum of availableProblemNumbers) {
        try {
            const response = await fetch(
                `${API_BASE}/problems/${currentSession.id}/${problemNum}/next`
            );
            if (response.ok) {
                return problemNum; // Found an ungraded problem
            }
        } catch (error) {
            console.error(`Error checking problem ${problemNum}:`, error);
        }
    }
    return null; // No ungraded problems found
}

async function maybeOfferSubjectiveFinalizeForCurrentProblem() {
    const settings = await loadSubjectiveSettings(Number(currentProblemNumber), true);
    if (!settings || settings.grading_mode !== 'subjective') {
        return false;
    }

    const untriagedCount = Number(settings.untriaged_count || 0);
    const triagedCount = Number(settings.triaged_count || 0);
    const canFinalizeNow = untriagedCount === 0 && triagedCount > 0;
    if (!canFinalizeNow) {
        return false;
    }

    applyGradingModeUI();
    const shouldFinalizeNow = confirm(
        `All responses for Problem ${currentProblemNumber} are bucketed.\n\n` +
        `Finalize subjective scores now?`
    );
    if (shouldFinalizeNow) {
        await openSubjectiveFinalizeDialog();
        return true;
    }
    return false;
}

// Load next ungraded problem
async function loadNextProblem() {
    try {
        if (shouldUseSubjectiveBucketFilter()) {
            await loadProblemFromActiveBucketFilter('next', false);
            return;
        }

        if (usePrefetchedNextProblemIfAvailable()) {
            return;
        }

        const response = await fetch(
            `${API_BASE}/problems/${currentSession.id}/${currentProblemNumber}/next`
        );

        if (response.status === 404) {
            const finalizeHandled = await maybeOfferSubjectiveFinalizeForCurrentProblem();
            if (finalizeHandled) {
                return;
            }

            // No more problems for this number
            // Find next ungraded problem number across all problems
            const nextUngradedProblem = await findNextUngradedProblem();

            if (nextUngradedProblem !== null) {
                // Found ungraded problems in another problem number
                if (lastGradedProblemNumber === currentProblemNumber) {
                    // Show notification if we just graded something
                    lastGradedProblemNumber = null;
                    showNotification(`All submissions for Problem ${currentProblemNumber} are graded! Moving to Problem ${nextUngradedProblem}...`, async () => {
                        invalidateNextProblemPrefetch();
                        currentProblemNumber = nextUngradedProblem;
                        document.getElementById('problem-select').value = currentProblemNumber;
                        await loadSubjectiveSettings(currentProblemNumber, true);
                        applyGradingModeUI();
                        updateMaxPointsDropdown();
                        loadNextProblem();
                    });
                } else {
                    // Silently move to next ungraded problem
                    invalidateNextProblemPrefetch();
                    currentProblemNumber = nextUngradedProblem;
                    document.getElementById('problem-select').value = currentProblemNumber;
                    await loadSubjectiveSettings(currentProblemNumber, true);
                    applyGradingModeUI();
                    updateMaxPointsDropdown();
                    loadNextProblem();
                }
            } else {
                // No more /next items available anywhere. This can mean either:
                // 1) fully graded, or 2) subjective-triaged but not finalized.
                let stats = null;
                try {
                    const statsResponse = await fetch(`${API_BASE}/sessions/${currentSession.id}/stats`);
                    if (statsResponse.ok) {
                        stats = await statsResponse.json();
                    }
                } catch (_error) {
                    stats = null;
                }

                const fullyGraded = stats && stats.problems_graded >= stats.total_problems;
                if (fullyGraded) {
                    if (lastGradedProblemNumber === currentProblemNumber) {
                        lastGradedProblemNumber = null;
                        showNotification('All problems are graded! 🎉', () => {
                            navigateToSection('stats-section');
                            loadStatistics();
                        });
                    } else {
                        navigateToSection('stats-section');
                        loadStatistics();
                    }
                } else {
                    showNotification('No untriaged/ungraded responses remain in this pass. Subjective problems may still need final score assignment.');
                }
            }
            return;
        }

        currentProblem = await response.json();
        await loadSubjectiveSettings(currentProblemNumber);

        // Add to history and display
        addToHistory(currentProblem);
        displayCurrentProblem();
        triggerNextProblemPrefetch();

    } catch (error) {
        console.error('Failed to load problem:', error);
        alert('Failed to load problem');
    }
}

// Submit grade for current problem
async function submitGrade() {
    if (!currentProblem) return;

    // Auto-apply selected tags before submitting
    if (typeof applySelectedTags === 'function' && selectedTagIds && selectedTagIds.size > 0) {
        await applySelectedTags();
    }

    const scoreValue = document.getElementById('score-input').value.trim();
    const maxPoints = problemMaxPoints[currentProblemNumber] || 8;

    // Check if it's a dash (for blank marking) or a number
    let score;
    let isBlank = false;
    if (scoreValue === '-') {
        score = '-';  // Send dash as-is to backend
        isBlank = true;
    } else {
        score = parseFloat(scoreValue);
        if (isNaN(score)) {
            alert('Please enter a valid score or "-" to mark as blank');
            return;
        }
        if (score > maxPoints) {
            alert(`Score cannot exceed ${maxPoints} points`);
            return;
        }
    }

    // Auto-apply default feedback if conditions are met
    if (typeof shouldApplyDefaultFeedback === 'function' && shouldApplyDefaultFeedback(score === '-' ? 0 : score, isBlank)) {
        applyDefaultFeedbackToTextarea();
    }

    let feedback = document.getElementById('feedback-input').value;

    // Auto-include explanation if checkbox is enabled.
    const includeExplanation = document.getElementById('include-explanation-checkbox');
    if (includeExplanation && includeExplanation.checked && currentProblem) {
        if (!explanationCache[currentProblem.id]) {
            await ensureExplanationLoaded(currentProblem.id);
        }
    }
    if (includeExplanation && includeExplanation.checked && currentProblem && explanationCache[currentProblem.id]) {
        const cachedExplanation = explanationCache[currentProblem.id];
        const explanationText = cachedExplanation.markdown || cachedExplanation.html || '';
        const explanationWithDisclaimer = 'Note: The explanation below is automatically generated and might not be correct.\n\n' + explanationText;

        if (explanationText && feedback.trim()) {
            // Append explanation with separator if there's existing feedback
            feedback = feedback + '\n\n---\n\n' + explanationWithDisclaimer;
        } else if (explanationText) {
            // Use explanation alone if no custom feedback
            feedback = explanationWithDisclaimer;
        }
    }

    // Show loading state
    const submitBtn = document.getElementById('submit-grade-btn');
    const originalText = submitBtn.textContent;
    submitBtn.disabled = true;
    submitBtn.textContent = 'Submitting...';

    try {
        const response = await fetch(`${API_BASE}/problems/${currentProblem.id}/grade`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ score, feedback })
        });

        if (!response.ok) {
            throw new Error(`Failed to submit grade: ${response.statusText}`);
        }

        // Mark that we just graded this problem number
        lastGradedProblemNumber = currentProblemNumber;

        // Load next problem first so navigation feels responsive on slower links.
        const progressPromise = updateOverallProgress();
        await loadNextProblem();
        await progressPromise;

        // Restore button state after loading next problem
        submitBtn.disabled = false;
        submitBtn.textContent = originalText;
    } catch (error) {
        console.error('Failed to submit grade:', error);
        alert('Failed to submit grade: ' + error.message);

        // Restore button state on error
        submitBtn.disabled = false;
        submitBtn.textContent = originalText;
    }
}

async function submitSubjectiveTriage(options = {}) {
    if (!currentProblem) return;
    if (getCurrentSubjectiveSettings()?.grading_mode !== 'subjective') {
        alert('Subjective mode is not enabled for this problem.');
        return;
    }
    if (subjectiveAssignInFlight) {
        return;
    }
    if (!selectedSubjectiveBucketId) {
        alert('Select a bucket before assigning.');
        return;
    }

    const notes = document.getElementById('subjective-notes-input').value.trim();
    const previousBucketId = currentProblem.subjective_bucket_id || null;
    const assignBtn = document.getElementById('subjective-assign-btn');
    const originalText = assignBtn.textContent;
    subjectiveAssignInFlight = true;
    assignBtn.disabled = true;
    assignBtn.textContent = options.triggeredByBucketClick ? 'Applying...' : 'Assigning...';

    try {
        const response = await fetch(`${API_BASE}/problems/${currentProblem.id}/subjective-triage`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                bucket_id: selectedSubjectiveBucketId,
                notes: notes || null
            })
        });
        let payload = null;
        try {
            payload = await response.json();
        } catch (_error) {
            payload = null;
        }

        if (!response.ok) {
            throw new Error(payload?.detail || 'Failed to assign subjective bucket');
        }

        applyLocalSubjectiveStateUpdate({
            triagedCount: payload?.triaged_count,
            untriagedCount: payload?.untriaged_count,
            finalizedCount: payload?.finalized_count,
            bucketUsage: payload?.bucket_usage,
            previousBucketId,
            bucketId: selectedSubjectiveBucketId
        });
        currentProblem.subjective_bucket_id = selectedSubjectiveBucketId;
        currentProblem.subjective_notes = notes || '';
        currentProblem.subjective_triaged = true;

        await loadNextProblem();
    } catch (error) {
        console.error('Failed to assign subjective bucket:', error);
        alert(error.message || 'Failed to assign subjective bucket');
    } finally {
        subjectiveAssignInFlight = false;
        assignBtn.disabled = false;
        assignBtn.textContent = originalText;
    }
}

async function clearSubjectiveTriage() {
    if (!currentProblem) return;
    if (getCurrentSubjectiveSettings()?.grading_mode !== 'subjective') {
        return;
    }

    const clearBtn = document.getElementById('subjective-clear-btn');
    const originalText = clearBtn.textContent;
    const previousBucketId = currentProblem.subjective_bucket_id || null;
    clearBtn.disabled = true;
    clearBtn.textContent = 'Clearing...';
    try {
        const response = await fetch(`${API_BASE}/problems/${currentProblem.id}/subjective-triage`, {
            method: 'DELETE'
        });
        let payload = null;
        try {
            payload = await response.json();
        } catch (_error) {
            payload = null;
        }
        if (!response.ok) {
            throw new Error(payload?.detail || 'Failed to clear subjective triage');
        }
        selectedSubjectiveBucketId = null;
        currentProblem.subjective_bucket_id = null;
        currentProblem.subjective_notes = '';
        currentProblem.subjective_triaged = false;
        document.getElementById('subjective-notes-input').value = '';
        applyLocalSubjectiveStateUpdate({
            triagedCount: payload?.triaged_count,
            untriagedCount: payload?.untriaged_count,
            finalizedCount: payload?.finalized_count,
            bucketUsage: payload?.bucket_usage,
            previousBucketId,
            bucketId: null
        });
    } catch (error) {
        console.error('Failed to clear subjective triage:', error);
        alert(error.message || 'Failed to clear subjective triage');
    } finally {
        clearBtn.disabled = false;
        clearBtn.textContent = originalText;
    }
}

function closeSubjectiveFinalizeDialog() {
  const dialog = document.getElementById('subjective-finalize-dialog');
  if (dialog) {
    dialog.style.display = 'none';
  }
}

async function openRandomBucketSample(bucketId) {
    if (!currentSession || !currentProblemNumber || !bucketId) return;
    try {
        const response = await fetch(
            `${API_BASE}/problems/${currentSession.id}/${currentProblemNumber}/bucket/${encodeURIComponent(bucketId)}/sample`
        );
        let payload = null;
        try {
            payload = await response.json();
        } catch (_error) {
            payload = null;
        }
        if (!response.ok) {
            throw new Error(payload?.detail || 'Failed to load bucket sample');
        }

        closeSubjectiveFinalizeDialog();
        currentProblem = payload;
        activeSubjectiveBucketFilter = bucketId;
        invalidateNextProblemPrefetch();
        await loadSubjectiveSettings(currentProblemNumber);
        const filter = document.getElementById('subjective-view-filter');
        if (filter) {
            renderSubjectiveBucketFilter();
            filter.value = bucketId;
        }
        addToHistory(currentProblem);
        displayCurrentProblem();
    } catch (error) {
        console.error('Failed to load bucket sample:', error);
        alert(error.message || 'Failed to load bucket sample');
    }
}

async function submitSubjectiveReopen() {
    if (!currentSession || !currentProblemNumber) return;

    if (!confirm(`Reopen subjective scores for Problem ${currentProblemNumber}?\n\nThis will clear current scores/feedback and return responses to triaged state.`)) {
        return;
    }

    const reopenBtn = document.getElementById('subjective-reopen-btn');
    const originalText = reopenBtn.textContent;
    reopenBtn.disabled = true;
    reopenBtn.textContent = 'Reopening...';

    try {
        const response = await fetch(`${API_BASE}/sessions/${currentSession.id}/subjective-reopen`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                problem_number: Number(currentProblemNumber)
            })
        });
        let payload = null;
        try {
            payload = await response.json();
        } catch (_error) {
            payload = null;
        }
        if (!response.ok) {
            throw new Error(payload?.detail || 'Failed to reopen subjective scores');
        }

        await loadSubjectiveSettings(Number(currentProblemNumber), true);
        await updateOverallProgress();
        invalidateNextProblemPrefetch();
        await loadProblemOrMostRecent();
        showNotification(`Reopened ${payload.reopened_count || 0} responses for Problem ${currentProblemNumber}.`);
    } catch (error) {
        console.error('Failed to reopen subjective scores:', error);
        alert(error.message || 'Failed to reopen subjective scores');
    } finally {
        reopenBtn.disabled = false;
        reopenBtn.textContent = originalText;
    }
}

function renderSubjectiveFinalizeRows(settings) {
    const summary = document.getElementById('subjective-finalize-summary');
    const rowsContainer = document.getElementById('subjective-finalize-rows');
    if (!summary || !rowsContainer) return [];

    const bucketUsage = settings?.bucket_usage || {};
    const activeBuckets = (settings?.buckets || []).filter((bucket) => Number(bucketUsage[bucket.id] || 0) > 0);
    const maxPoints = getCurrentProblemMaxPoints();

    summary.textContent = `Problem ${currentProblemNumber}: ${settings?.triaged_count || 0} triaged response(s), ${activeBuckets.length} active bucket(s), max points ${maxPoints}.`;
    rowsContainer.innerHTML = '';

    activeBuckets.forEach((bucket) => {
        const count = Number(bucketUsage[bucket.id] || 0);
        const row = document.createElement('div');
        row.className = 'subjective-finalize-row';
        row.dataset.bucketId = bucket.id;

        const label = document.createElement('div');
        label.style.paddingTop = '6px';
        label.innerHTML = `<strong>${bucket.label}</strong><div style="font-size: 12px; color: var(--gray-700);">${count} responses</div>`;
        const sampleBtn = document.createElement('button');
        sampleBtn.type = 'button';
        sampleBtn.className = 'btn btn-secondary btn-small';
        sampleBtn.style.marginTop = '6px';
        sampleBtn.textContent = 'View Sample';
        sampleBtn.onclick = async () => {
            await openRandomBucketSample(bucket.id);
        };
        label.appendChild(sampleBtn);

        const scoreInput = document.createElement('input');
        scoreInput.type = 'number';
        scoreInput.step = '0.25';
        scoreInput.min = '0';
        scoreInput.max = String(maxPoints);
        scoreInput.placeholder = 'Score';
        scoreInput.className = 'subjective-finalize-score';

        const feedbackInput = document.createElement('textarea');
        feedbackInput.rows = 2;
        feedbackInput.maxLength = 4000;
        feedbackInput.placeholder = 'Optional feedback applied to all responses in this bucket';
        feedbackInput.className = 'subjective-finalize-feedback';

        row.appendChild(label);
        row.appendChild(scoreInput);
        row.appendChild(feedbackInput);
        rowsContainer.appendChild(row);
    });

    return activeBuckets;
}

async function openSubjectiveFinalizeDialog() {
    const settings = await loadSubjectiveSettings(Number(currentProblemNumber), true);
    if (!settings || settings.grading_mode !== 'subjective') {
        alert('Subjective mode is not enabled for this problem.');
        return;
    }

    const untriagedCount = Number(settings.untriaged_count || 0);
    if (untriagedCount > 0) {
        alert(`Assign buckets to all responses before finalizing. ${untriagedCount} responses remain untriaged.`);
        return;
    }

    const activeBuckets = renderSubjectiveFinalizeRows(settings);
    if (activeBuckets.length === 0) {
        alert('No triaged responses remain to finalize for this problem.');
        return;
    }

    document.getElementById('subjective-finalize-dialog').style.display = 'flex';
}

function collectSubjectiveFinalizePayload() {
    const rows = [...document.querySelectorAll('#subjective-finalize-rows .subjective-finalize-row')];
    const maxPoints = getCurrentProblemMaxPoints();

    const bucketScores = rows.map((row) => {
        const bucketId = row.dataset.bucketId;
        const scoreInput = row.querySelector('.subjective-finalize-score');
        const feedbackInput = row.querySelector('.subjective-finalize-feedback');
        const scoreText = (scoreInput?.value || '').trim();
        const score = parseFloat(scoreText);
        if (!scoreText || Number.isNaN(score)) {
            throw new Error(`Enter a numeric score for bucket "${bucketId}".`);
        }
        if (score < 0 || score > maxPoints) {
            throw new Error(`Score for bucket "${bucketId}" must be between 0 and ${maxPoints}.`);
        }
        return {
            bucket_id: bucketId,
            score,
            feedback: (feedbackInput?.value || '').trim() || null
        };
    });

    return bucketScores;
}

async function submitSubjectiveFinalize() {
    if (!currentSession || !currentProblemNumber) return;

    let bucketScores;
    try {
        bucketScores = collectSubjectiveFinalizePayload();
    } catch (error) {
        alert(error.message || 'Please fill all bucket scores.');
        return;
    }

    if (!confirm(`Apply subjective scores for Problem ${currentProblemNumber}? This will mark all triaged responses as graded.`)) {
        return;
    }

    const confirmBtn = document.getElementById('subjective-finalize-confirm-btn');
    const originalText = confirmBtn.textContent;
    confirmBtn.disabled = true;
    confirmBtn.textContent = 'Applying...';

    try {
        const response = await fetch(`${API_BASE}/sessions/${currentSession.id}/subjective-finalize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                problem_number: Number(currentProblemNumber),
                bucket_scores: bucketScores
            })
        });

        let payload = null;
        try {
            payload = await response.json();
        } catch (_error) {
            payload = null;
        }

        if (!response.ok) {
            throw new Error(payload?.detail || 'Failed to finalize subjective scores');
        }

        closeSubjectiveFinalizeDialog();
        invalidateNextProblemPrefetch();
        await loadSubjectiveSettings(Number(currentProblemNumber), true);
        await updateOverallProgress();
        updateSubjectiveFinalizeButton();
        await loadNextProblem();
        showNotification(`Finalized ${payload.graded_count || 0} responses for Problem ${currentProblemNumber}.`);
    } catch (error) {
        console.error('Failed to finalize subjective scores:', error);
        alert(error.message || 'Failed to finalize subjective scores');
    } finally {
        confirmBtn.disabled = false;
        confirmBtn.textContent = originalText;
    }
}

// Toggle student scores visibility
function toggleStudentScores() {
    const container = document.getElementById('student-scores-container');
    const toggle = document.getElementById('student-scores-toggle');

    if (container.style.display === 'none') {
        container.style.display = 'block';
        toggle.textContent = '▼';
    } else {
        container.style.display = 'none';
        toggle.textContent = '▶';
    }
}

// Load statistics
async function loadStatistics() {
    try {
        const [statsResponse, scoresResponse] = await Promise.all([
            fetch(`${API_BASE}/sessions/${currentSession.id}/stats`),
            fetch(`${API_BASE}/sessions/${currentSession.id}/student-scores`)
        ]);

        const stats = await statsResponse.json();
        const scoresData = await scoresResponse.json();

        const container = document.getElementById('stats-container');

        // Calculate overall statistics based on what's been graded so far
        let examStatsHtml = '';
        const studentsWithGrades = scoresData.students.filter(s => s.total_score !== null && s.graded_problems > 0);

        if (studentsWithGrades.length > 0) {
            // Get raw scores
            const rawScores = studentsWithGrades.map(s => s.total_score);
            const rawMin = Math.min(...rawScores);
            const rawMax = Math.max(...rawScores);
            const rawAvg = rawScores.reduce((sum, s) => sum + s, 0) / rawScores.length;

            // Calculate raw standard deviation
            const rawVariance = rawScores.reduce((sum, s) => sum + Math.pow(s - rawAvg, 2), 0) / rawScores.length;
            const rawStddev = Math.sqrt(rawVariance);

            // Calculate normalized scores (percentage of points earned out of points graded)
            const normalizedScores = studentsWithGrades.map(s => {
                // Calculate max possible points for problems this student has been graded on
                // We need to figure out which problems they've been graded on
                // For now, approximate using their graded_problems count
                const problemsGraded = s.graded_problems;

                // Get the actual max points for problems based on graded count
                // Assume problems are graded in order (1, 2, 3, etc.)
                let maxPossibleForStudent = 0;
                const gradedProblemStats = stats.problem_stats.slice(0, problemsGraded);
                gradedProblemStats.forEach(ps => {
                    maxPossibleForStudent += (ps.max_points || 8);
                });

                // Return normalized score as percentage
                return maxPossibleForStudent > 0 ? (s.total_score / maxPossibleForStudent) * 100 : 0;
            });

            const normAvg = normalizedScores.reduce((sum, s) => sum + s, 0) / normalizedScores.length;

            // Calculate normalized standard deviation
            const normVariance = normalizedScores.reduce((sum, s) => sum + Math.pow(s - normAvg, 2), 0) / normalizedScores.length;
            const normStddev = Math.sqrt(normVariance);

            // Calculate total possible points across all problems
            const totalPossible = stats.problem_stats.reduce((sum, ps) => {
                return sum + (ps.max_points || 8);
            }, 0);

            // Calculate Canvas grade (raw score out of 100)
            const canvasAvg = rawAvg; // Canvas grade is the raw score (out of 100)
            const canvasPercentage = canvasAvg; // Since it's out of 100, the score IS the percentage

            // Calculate blank percentage statistics per student
            const blankPercentages = studentsWithGrades.map(student => {
                // Find all graded problems for this student
                const studentProblems = stats.problem_stats.filter(ps => ps.num_graded > 0);
                if (studentProblems.length === 0) return 0;

                // Count how many problems this student left blank
                // We don't have per-student blank data easily accessible, so we'll calculate from problem_stats
                // For now, use the overall blank percentage as an approximation
                // A better approach would require additional API endpoint for per-student blank counts
                const totalBlankProblems = stats.problem_stats.reduce((sum, ps) => sum + (ps.num_blank || 0), 0);
                const totalGradedProblems = stats.problem_stats.reduce((sum, ps) => sum + ps.num_graded, 0);

                return totalGradedProblems > 0 ? (totalBlankProblems / totalGradedProblems) * 100 : 0;
            });

            // Calculate average blank percentage across all graded problems
            const totalBlankProblems = stats.problem_stats.reduce((sum, ps) => sum + (ps.num_blank || 0), 0);
            const totalGradedProblems = stats.problem_stats.reduce((sum, ps) => sum + ps.num_graded, 0);
            const avgBlankPct = totalGradedProblems > 0 ? (totalBlankProblems / totalGradedProblems) * 100 : 0;

            // Calculate stddev of blank percentages per problem
            const problemBlankPcts = stats.problem_stats
                .filter(ps => ps.num_graded > 0)
                .map(ps => ((ps.num_blank || 0) / ps.num_graded) * 100);
            const blankPctStddev = problemBlankPcts.length > 1
                ? Math.sqrt(problemBlankPcts.reduce((sum, pct) => sum + Math.pow(pct - avgBlankPct, 2), 0) / problemBlankPcts.length)
                : 0;

            examStatsHtml = `
                <h3>Overall Progress Statistics <small style="font-size: 14px; font-weight: normal; color: var(--gray-600);">(${studentsWithGrades.length} students with grades, based on problems graded so far)</small></h3>
                <div class="overall-stats" style="margin-bottom: 30px;">
                    <div class="stat-card">
                        <h3>Average Score</h3>
                        <div class="value">${rawAvg.toFixed(2)} pts</div>
                        <div style="font-size: 14px; color: var(--gray-600); margin-top: 5px;">±${rawStddev.toFixed(2)} pts</div>
                    </div>
                    <div class="stat-card">
                        <h3>Canvas Grade</h3>
                        <div class="value">${canvasPercentage.toFixed(1)}%</div>
                        <div style="font-size: 14px; color: var(--gray-600); margin-top: 5px;">(out of 100)</div>
                    </div>
                    <div class="stat-card">
                        <h3>Normalized Average</h3>
                        <div class="value">${normAvg.toFixed(1)}%</div>
                        <div style="font-size: 14px; color: var(--gray-600); margin-top: 5px;">±${normStddev.toFixed(1)}%</div>
                    </div>
                    <div class="stat-card">
                        <h3>Score Range</h3>
                        <div class="value">${rawMin.toFixed(2)} - ${rawMax.toFixed(2)}</div>
                        <div style="font-size: 14px; color: var(--gray-600); margin-top: 5px;">Min to Max</div>
                    </div>
                    <div class="stat-card">
                        <h3>Blank Rate</h3>
                        <div class="value">${avgBlankPct.toFixed(1)}%</div>
                        <div style="font-size: 14px; color: var(--gray-600); margin-top: 5px;">±${blankPctStddev.toFixed(1)}%</div>
                    </div>
                </div>
            `;
        }

        container.innerHTML = examStatsHtml + `
            <h3>Grading Progress</h3>
            <div class="overall-stats">
                <div class="stat-card">
                    <h3>Total Submissions</h3>
                    <div class="value">${stats.total_submissions}</div>
                </div>
                <div class="stat-card">
                    <h3>Problems Graded</h3>
                    <div class="value">${stats.problems_graded} / ${stats.total_problems}</div>
                </div>
                <div class="stat-card">
                    <h3>Overall Progress</h3>
                    <div class="value">${stats.progress_percentage.toFixed(1)}%</div>
                    <div class="progress-bar-container">
                        <div class="progress-bar-fill" style="width: ${stats.progress_percentage}%"></div>
                    </div>
                </div>
            </div>
        `;

        // Add per-problem stats
        if (stats.problem_stats.length > 0) {
            container.innerHTML += '<h3 style="margin-top: 30px;">Per-Problem Statistics <small style="font-size: 14px; font-weight: normal; color: var(--gray-600);">(click a card to review)</small></h3>';
            const problemStatsHtml = stats.problem_stats.map(ps => {
                const problemProgress = ps.num_total > 0 ? (ps.num_graded / ps.num_total * 100) : 0;

                // Format statistics with fallbacks
                const avgText = ps.avg_score !== null && ps.avg_score !== undefined ? ps.avg_score.toFixed(2) : 'N/A';
                const minText = ps.min_score !== null && ps.min_score !== undefined ? ps.min_score.toFixed(2) : 'N/A';
                const maxText = ps.max_score !== null && ps.max_score !== undefined ? ps.max_score.toFixed(2) : 'N/A';
                const medianText = ps.median_score !== null && ps.median_score !== undefined ? ps.median_score.toFixed(2) : 'N/A';
                const stddevText = ps.stddev_score !== null && ps.stddev_score !== undefined ? ps.stddev_score.toFixed(2) : 'N/A';

                // Format mean ± stddev
                const meanPlusMinusText = (ps.avg_score !== null && ps.avg_score !== undefined && ps.stddev_score !== null && ps.stddev_score !== undefined)
                    ? `${ps.avg_score.toFixed(2)} ± ${ps.stddev_score.toFixed(2)}`
                    : 'N/A';

                // Format normalized mean ± normalized stddev (as percentages)
                const meanNormPlusMinusText = (ps.mean_normalized !== null && ps.mean_normalized !== undefined && ps.stddev_normalized !== null && ps.stddev_normalized !== undefined)
                    ? `${(ps.mean_normalized * 100).toFixed(1)}% ± ${(ps.stddev_normalized * 100).toFixed(1)}%`
                    : 'N/A';

                const maxPointsText = ps.max_points !== null && ps.max_points !== undefined ? ps.max_points.toFixed(1) : 'N/A';

                // Format blank % with highlight if high skip rate
                let pctBlankDisplay;
                const hasHighSkipRate = ps.pct_blank !== null && ps.pct_blank !== undefined && ps.pct_blank > 25;
                if (ps.pct_blank !== null && ps.pct_blank !== undefined) {
                    const blankValue = ps.pct_blank.toFixed(1) + '%';
                    if (hasHighSkipRate) {
                        pctBlankDisplay = `<div style="color: var(--gray-600); font-size: 12px; margin-bottom: 2px;">Blank %</div><div class="blank-pct-highlight">${blankValue}</div>`;
                    } else {
                        pctBlankDisplay = `<div style="color: var(--gray-600); font-size: 12px; margin-bottom: 2px;">Blank %</div><div style="font-weight: 600; font-size: 16px;">${blankValue}</div>`;
                    }
                } else {
                    pctBlankDisplay = '<div style="color: var(--gray-600); font-size: 12px; margin-bottom: 2px;">Blank %</div><div style="font-weight: 600; font-size: 16px;">N/A</div>';
                }

                // Determine CSS classes for visual indicators
                let cssClasses = 'stat-card';

                // Completion indicator - add 'fully-graded' class if all problems graded
                if (ps.num_graded >= ps.num_total && ps.num_total > 0) {
                    cssClasses += ' fully-graded';
                }

                // Performance indicator - add class based on normalized mean
                // Only add if we have valid data
                if (ps.mean_normalized !== null && ps.mean_normalized !== undefined) {
                    if (ps.mean_normalized >= 0.9) {
                        cssClasses += ' performance-excellent';  // 90%+
                    } else if (ps.mean_normalized >= 0.75) {
                        cssClasses += ' performance-good';       // 75-89%
                    } else if (ps.mean_normalized >= 0.6) {
                        cssClasses += ' performance-moderate';   // 60-74%
                    } else if (ps.mean_normalized >= 0.5) {
                        cssClasses += ' performance-poor';       // 50-59%
                    } else {
                        cssClasses += ' performance-verypoor';   // <50%
                    }
                }

                return `
                    <div class="${cssClasses}" data-problem-number="${ps.problem_number}" style="cursor: pointer; transition: all 0.2s;"
                         onmouseenter="this.style.transform='translateY(-4px)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.15)';"
                         onmouseleave="this.style.transform=''; this.style.boxShadow='';"
                         onclick="reviewProblemFromStats(${ps.problem_number})">
                        <h3 style="margin-bottom: 12px; border-bottom: 2px solid var(--primary-color); padding-bottom: 8px;">Problem ${ps.problem_number}</h3>

                        <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 8px; margin-bottom: 8px;">
                            <div style="text-align: left;">
                                <div style="color: var(--gray-600); font-size: 12px; margin-bottom: 2px;">Mean ± Std Dev</div>
                                <div style="font-weight: 600; font-size: 16px;">${meanPlusMinusText}</div>
                            </div>
                            <div style="text-align: right;">
                                <div style="color: var(--gray-600); font-size: 12px; margin-bottom: 2px;">Median</div>
                                <div style="font-weight: 600; font-size: 16px;">${medianText}</div>
                            </div>
                        </div>
                        <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 8px; margin-bottom: 12px;">
                            <div style="text-align: left;">
                                <div style="color: var(--gray-600); font-size: 12px; margin-bottom: 2px;">Normalized</div>
                                <div style="font-weight: 600; font-size: 16px;">${meanNormPlusMinusText}</div>
                            </div>
                            <div style="text-align: right;">
                                ${pctBlankDisplay}
                            </div>
                        </div>

                        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-bottom: 12px; padding-top: 8px; border-top: 1px solid var(--gray-200);">
                            <div style="text-align: center;">
                                <div style="color: var(--gray-600); font-size: 11px;">Min</div>
                                <div style="font-weight: 500;">${minText}</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="color: var(--gray-600); font-size: 11px;">Max</div>
                                <div style="font-weight: 500;">${maxText}</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="color: var(--gray-600); font-size: 11px;">Out of</div>
                                <div style="font-weight: 500;">${maxPointsText}</div>
                            </div>
                        </div>

                        <div style="padding-top: 8px; border-top: 1px solid var(--gray-200); text-align: center;">
                            <div style="color: var(--gray-700); font-size: 13px; font-weight: 500;">
                                Graded: ${ps.num_graded} / ${ps.num_total} (${problemProgress.toFixed(0)}%)
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
            container.innerHTML += '<div class="problem-stats-grid">' + problemStatsHtml + '</div>';
        }

        // Add student scores table (collapsible, hidden by default)
        if (scoresData.students.length > 0) {
            container.innerHTML += `
                <h3 style="margin-top: 30px; cursor: pointer; user-select: none;"
                    id="student-scores-header"
                    onclick="toggleStudentScores()"
                    title="Click to expand/collapse">
                    <span id="student-scores-toggle">▶</span> Student Scores (${scoresData.students.length})
                </h3>
            `;
            // Check if current user is a TA for anonymous grading
            const isTA = currentUser && currentUser.role === 'ta';

            const studentScoresHtml = `
                <div id="student-scores-container" style="display: none;">
                    <table class="student-scores-table">
                        <thead>
                            <tr>
                                ${!isTA ? `
                                <th class="sortable" onclick="sortStudentTable('name')" data-sort="name">
                                    Student Name <span class="sort-indicator"></span>
                                </th>
                                ` : ''}
                                <th class="sortable" onclick="sortStudentTable('progress')" data-sort="progress">
                                    Progress <span class="sort-indicator"></span>
                                </th>
                                <th class="sortable" onclick="sortStudentTable('score')" data-sort="score">
                                    Total Score <span class="sort-indicator"></span>
                                </th>
                            </tr>
                        </thead>
                        <tbody id="student-scores-tbody">
                            ${scoresData.students.map(s => `
                                <tr class="${s.is_complete ? 'complete' : 'incomplete'}"
                                    data-name="${s.student_name || 'Unmatched'}"
                                    data-progress="${s.graded_problems / s.total_problems}"
                                    data-score="${s.total_score || 0}">
                                    ${!isTA ? `<td>${s.student_name || 'Unmatched'}</td>` : ''}
                                    <td>${s.graded_problems} / ${s.total_problems}</td>
                                    <td>${s.total_score ? s.total_score.toFixed(2) : '0.00'}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            `;
            container.innerHTML += studentScoresHtml;
        }

        // Load TA assignments for instructors
        if (typeof loadSessionAssignments === 'function') {
            await loadSessionAssignments();
        }
    } catch (error) {
        console.error('Failed to load statistics:', error);
    }
}

// Change Canvas Target button
document.getElementById('change-canvas-target-btn').onclick = async () => {
    if (!currentSession) return;
    if (currentSession.mock_roster) {
        showNotification('Mock roster sessions cannot update Canvas targets.');
        return;
    }

    const dialog = document.getElementById('canvas-target-dialog');
    const envSelect = document.getElementById('canvas-env-select');
    const courseSelect = document.getElementById('canvas-course-select');
    const assignmentSelect = document.getElementById('canvas-assignment-select');

    // Show dialog
    dialog.style.display = 'flex';

    // Load current settings
    try {
        const response = await fetch(`${API_BASE}/sessions/${currentSession.id}/canvas-info`);
        const info = await response.json();

        // Set current environment
        envSelect.value = info.environment === 'production' ? 'true' : 'false';

        // Load courses for selected environment
        await loadCanvasConfigCourses();

        // Select current course
        courseSelect.value = info.course_id;

        // Load and select current assignment
        await loadCanvasConfigAssignments(info.course_id);
        assignmentSelect.value = info.assignment_id;

    } catch (error) {
        console.error('Failed to load current Canvas config:', error);
    }
};

// Load courses for Canvas config dialog
async function loadCanvasConfigCourses() {
    const envSelect = document.getElementById('canvas-env-select');
    const courseSelect = document.getElementById('canvas-course-select');
    const useProd = envSelect.value === 'true';

    courseSelect.innerHTML = '<option value="">Loading courses...</option>';
    courseSelect.disabled = true;

    try {
        const response = await fetch(`${API_BASE}/canvas/courses?use_prod=${useProd}`);
        const data = await response.json();

        courseSelect.innerHTML = '<option value="">-- Select a Course --</option>';
        data.courses.forEach(course => {
            const option = document.createElement('option');
            option.value = course.id;
            const prefix = course.is_favorite ? '⭐ ' : '';
            option.textContent = prefix + course.name;
            courseSelect.appendChild(option);
        });

        courseSelect.disabled = false;
    } catch (error) {
        console.error('Failed to load courses:', error);
        courseSelect.innerHTML = '<option value="">Failed to load courses</option>';
    }
}

// Load assignments for Canvas config dialog
async function loadCanvasConfigAssignments(courseId) {
    const envSelect = document.getElementById('canvas-env-select');
    const assignmentSelect = document.getElementById('canvas-assignment-select');
    const useProd = envSelect.value === 'true';

    assignmentSelect.innerHTML = '<option value="">Loading assignments...</option>';
    assignmentSelect.disabled = true;

    try {
        const response = await fetch(`${API_BASE}/canvas/courses/${courseId}/assignments?use_prod=${useProd}`);
        const data = await response.json();

        assignmentSelect.innerHTML = '<option value="">-- Select an Assignment --</option>';
        data.assignments.forEach(assignment => {
            const option = document.createElement('option');
            option.value = assignment.id;
            option.textContent = assignment.name;
            assignmentSelect.appendChild(option);
        });

        assignmentSelect.disabled = false;
    } catch (error) {
        console.error('Failed to load assignments:', error);
        assignmentSelect.innerHTML = '<option value="">Failed to load assignments</option>';
    }
}

// Canvas config dialog event handlers
document.getElementById('canvas-env-select').onchange = loadCanvasConfigCourses;
document.getElementById('canvas-course-select').onchange = (e) => {
    if (e.target.value) {
        loadCanvasConfigAssignments(e.target.value);
    }
};

document.getElementById('cancel-canvas-target-btn').onclick = () => {
    document.getElementById('canvas-target-dialog').style.display = 'none';
};

document.getElementById('save-canvas-target-btn').onclick = async () => {
    const courseId = document.getElementById('canvas-course-select').value;
    const assignmentId = document.getElementById('canvas-assignment-select').value;
    const useProd = document.getElementById('canvas-env-select').value === 'true';

    if (!courseId || !assignmentId) {
        alert('Please select both a course and an assignment');
        return;
    }

    try {
        const response = await fetch(
            `${API_BASE}/sessions/${currentSession.id}/canvas-config?course_id=${courseId}&assignment_id=${assignmentId}&use_prod=${useProd}`,
            { method: 'PUT' }
        );

        if (!response.ok) {
            throw new Error('Failed to update Canvas configuration');
        }

        const result = await response.json();
        alert(`Canvas target updated!\n\nEnvironment: ${result.environment}\nCourse: ${result.course_name}\nAssignment: ${result.assignment_name}`);

        // Close dialog and reload session
        document.getElementById('canvas-target-dialog').style.display = 'none';

        // Refresh session data
        const sessionResponse = await fetch(`${API_BASE}/sessions/${currentSession.id}`);
        currentSession = await sessionResponse.json();
        updateSessionInfo();

    } catch (error) {
        console.error('Failed to update Canvas config:', error);
        alert('Failed to update Canvas configuration. Please try again.');
    }
};

// Export session button
document.getElementById('export-session-btn').onclick = async () => {
    if (!currentSession) return;

    try {
        // Fetch export data
        const response = await fetch(`${API_BASE}/sessions/${currentSession.id}/export`);

        if (!response.ok) {
            throw new Error('Export failed');
        }

        // Get filename from Content-Disposition header or generate default
        const contentDisposition = response.headers.get('Content-Disposition');
        let filename = `grading_session_${currentSession.id}.json`;
        if (contentDisposition) {
            const filenameMatch = contentDisposition.match(/filename="?([^"]+)"?/);
            if (filenameMatch) {
                filename = filenameMatch[1];
            }
        }

        // Download the file
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        alert('Session exported successfully! Save this file to resume grading later.');

    } catch (error) {
        console.error('Export failed:', error);
        alert('Failed to export session. Please try again.');
    }
};

// Finalize and upload to Canvas
document.getElementById('finalize-btn').onclick = async () => {
    if (!currentSession) return;
    if (currentSession.mock_roster) {
        showNotification('Mock roster sessions cannot be finalized to Canvas.');
        return;
    }

    // Check if all grading is complete
    try {
        const [statsResponse, canvasInfoResponse] = await Promise.all([
            fetch(`${API_BASE}/sessions/${currentSession.id}/stats`),
            fetch(`${API_BASE}/sessions/${currentSession.id}/canvas-info`)
        ]);

        const stats = await statsResponse.json();
        const canvasInfo = await canvasInfoResponse.json();

        if (stats.problems_graded < stats.total_problems) {
            showNotification(
                `Cannot finalize: ${stats.total_problems - stats.problems_graded} problems still ungraded. Please complete all grading first.`
            );
            return;
        }

        // Confirm finalization with Canvas details
        const confirmMessage = `Ready to finalize and upload ${stats.total_submissions} submissions to Canvas?\n\n` +
            `Canvas Details:\n` +
            `- Environment: ${canvasInfo.environment.toUpperCase()}\n` +
            `- Course: ${canvasInfo.course_name}\n` +
            `- Assignment: ${canvasInfo.assignment_name}\n` +
            `- URL: ${canvasInfo.canvas_url}\n\n` +
            `This will:\n` +
            `- Generate annotated PDFs with scores\n` +
            `- Upload to Canvas with detailed comments\n` +
            `- Mark this session as complete`;

        if (!confirm(confirmMessage)) {
            return;
        }

        // Show progress area IMMEDIATELY to provide feedback
        const progressDiv = document.getElementById('finalization-progress');
        const messageDiv = document.getElementById('finalization-message');
        const progressBar = document.getElementById('finalization-progress-bar');

        progressDiv.style.display = 'block';
        messageDiv.textContent = 'Initializing finalization...';
        progressBar.style.width = '0%';
        document.getElementById('finalize-btn').disabled = true;

        // Start finalization
        const response = await fetch(`${API_BASE}/finalize/${currentSession.id}/finalize`, {
            method: 'POST'
        });

        if (!response.ok) {
            const error = await response.json();
            // Reset UI on error
            progressDiv.style.display = 'none';
            document.getElementById('finalize-btn').disabled = false;
            throw new Error(error.detail || 'Finalization failed');
        }

        // Update message once server responds
        messageDiv.textContent = 'Starting finalization...';

        connectToFinalizationStream();

    } catch (error) {
        console.error('Finalization failed:', error);
        alert('Failed to start finalization: ' + error.message);
    }
};

// Listen for finalization status via SSE
let finalizationEventSource = null;

function connectToFinalizationStream() {
    // Close existing connection if any
    if (finalizationEventSource) {
        finalizationEventSource.close();
    }

    const streamUrl = `${API_BASE}/finalize/${currentSession.id}/finalize-stream`;
    console.log('Connecting to finalization SSE stream:', streamUrl);

    finalizationEventSource = new EventSource(streamUrl);

    finalizationEventSource.addEventListener('connected', (e) => {
        console.log('SSE connected for finalization progress');
    });

    finalizationEventSource.addEventListener('start', (e) => {
        const data = JSON.parse(e.data);
        console.log('Finalization started:', data);
        document.getElementById('finalization-message').textContent = data.message;
    });

    finalizationEventSource.addEventListener('progress', (e) => {
        const data = JSON.parse(e.data);
        console.log('Finalization progress:', data);

        document.getElementById('finalization-message').textContent = data.message;
        document.getElementById('finalization-progress-bar').style.width = `${data.progress}%`;
    });

    finalizationEventSource.addEventListener('complete', (e) => {
        const data = JSON.parse(e.data);
        console.log('Finalization complete:', data);

        finalizationEventSource.close();
        finalizationEventSource = null;

        document.getElementById('finalization-progress-bar').style.width = '100%';
        showNotification('Finalization complete! All grades have been uploaded to Canvas. 🎉', () => {
            location.reload();
        });
    });

    finalizationEventSource.addEventListener('error', (e) => {
        console.error('Finalization SSE error:', e);

        if (finalizationEventSource && finalizationEventSource.readyState === EventSource.CLOSED) {
            console.log('SSE connection closed');
            finalizationEventSource = null;
        } else {
            document.getElementById('finalization-progress').style.backgroundColor = '#fee2e2';
            document.getElementById('finalization-message').textContent = 'Connection error during finalization';
        }
    });
}

// Handwriting Transcription Dialog
const transcriptionDialog = document.getElementById('transcription-dialog');
const transcriptionText = document.getElementById('transcription-text');
const transcriptionActions = document.getElementById('transcription-actions');
const modelUsed = document.getElementById('model-used');
const closeTranscription = document.getElementById('close-transcription');
const decipherBtn = document.getElementById('decipher-btn');
const retryPremiumBtn = document.getElementById('retry-premium-btn');

// Cache for transcriptions: { problemId: { standard: {text, model}, premium: {text, model} } }
const transcriptionCache = {};

// Make dialog draggable
let isDragging = false;
let dragOffsetX = 0;
let dragOffsetY = 0;

document.querySelector('.transcription-header').addEventListener('mousedown', (e) => {
    if (e.target.classList.contains('transcription-close')) return;
    isDragging = true;
    const rect = transcriptionDialog.getBoundingClientRect();
    dragOffsetX = e.clientX - rect.left;
    dragOffsetY = e.clientY - rect.top;
    transcriptionDialog.style.transform = 'none';
});

document.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    transcriptionDialog.style.left = (e.clientX - dragOffsetX) + 'px';
    transcriptionDialog.style.top = (e.clientY - dragOffsetY) + 'px';
});

document.addEventListener('mouseup', () => {
    isDragging = false;
});

// Close dialog
closeTranscription.addEventListener('click', () => {
    transcriptionDialog.style.display = 'none';
});

// Function to fetch transcription (with caching)
async function fetchTranscription(problemId, model = 'default') {
    // Normalize 'default' to 'sonnet' for caching since default routes to Anthropic.
    const cacheKey = model === 'default' ? 'sonnet' : model;

    // Check cache first
    if (transcriptionCache[problemId] && transcriptionCache[problemId][cacheKey]) {
        console.log(`Using cached ${cacheKey} transcription for problem ${problemId}`);
        return transcriptionCache[problemId][cacheKey];
    }

    console.log(`Fetching new ${model} transcription for problem ${problemId}`);

    // Fetch from API
    const url = `${API_BASE}/problems/${problemId}/decipher?model=${model}`;
    const response = await fetch(url, { method: 'POST' });

    if (!response.ok) {
        let detail = 'Transcription failed';
        try {
            const errorPayload = await response.json();
            if (errorPayload?.detail) {
                detail = errorPayload.detail;
            }
        } catch (_error) {
            // Keep default detail
        }
        throw new Error(detail);
    }

    const data = await response.json();

    // Cache the result
    if (!transcriptionCache[problemId]) {
        transcriptionCache[problemId] = {};
    }
    transcriptionCache[problemId][cacheKey] = {
        text: data.transcription,
        model: data.model
    };

    console.log(`Cached ${cacheKey} transcription for problem ${problemId}`);

    return transcriptionCache[problemId][cacheKey];
}

// Function to display transcription in dialog
function displayTranscription(transcription) {
    transcriptionText.textContent = transcription.text;
    modelUsed.textContent = `Model used: ${transcription.model}`;

    // Show model selection buttons
    transcriptionActions.style.display = 'block';
    transcriptionActions.innerHTML = `
        <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px;">
            <button id="retry-sonnet-btn" class="btn-secondary" style="flex: 1; min-width: 120px;">
                Try Sonnet
            </button>
            <button id="retry-opus-btn" class="btn-secondary" style="flex: 1; min-width: 120px;">
                Try Opus (Premium)
            </button>
        </div>
    `;

    // Add event listeners for the new buttons
    document.getElementById('retry-sonnet-btn').addEventListener('click', () => retryWithModel('sonnet'));
    document.getElementById('retry-opus-btn').addEventListener('click', () => retryWithModel('opus'));
}

// Function to retry transcription with a specific model
async function retryWithModel(model) {
    if (!currentProblem) return;

    const modelNames = {
        'default': 'Anthropic',
        'sonnet': 'Sonnet',
        'opus': 'Opus (Premium)'
    };

    // Show loading state
    transcriptionText.innerHTML = `<div class="transcription-loading">Transcribing with ${modelNames[model]}...</div>`;
    transcriptionActions.style.display = 'none';

    try {
        const transcription = await fetchTranscription(currentProblem.id, model);
        displayTranscription(transcription);
    } catch (error) {
        console.error(`Failed to decipher with ${model}:`, error);
        const modelLabel = modelNames[model] || model;
        transcriptionText.innerHTML = `<div style="color: var(--danger-color);">Failed to transcribe with ${modelLabel}. ${error.message}</div>`;
        // Show buttons again so user can retry
        transcriptionActions.style.display = 'block';
    }
}

// Function to update transcription dialog when problem changes
async function updateTranscriptionDialog() {
    if (!currentProblem) {
        transcriptionDialog.style.display = 'none';
        return;
    }

    // Check if we have a cached transcription for this problem (default to sonnet)
    const cacheKey = 'sonnet';
    if (transcriptionCache[currentProblem.id] && transcriptionCache[currentProblem.id][cacheKey]) {
        // Show cached transcription immediately
        console.log(`Showing cached transcription for problem ${currentProblem.id}`);
        displayTranscription(transcriptionCache[currentProblem.id][cacheKey]);
    } else {
        // No cache - fetch new transcription with Anthropic default
        console.log(`No cache found, fetching new transcription for problem ${currentProblem.id}`);
        transcriptionText.innerHTML = '<div class="transcription-loading">Transcribing handwriting with Anthropic...</div>';
        transcriptionActions.style.display = 'none';

        try {
            const transcription = await fetchTranscription(currentProblem.id, 'default');
            displayTranscription(transcription);
        } catch (error) {
            console.error('Failed to auto-fetch transcription:', error);
            transcriptionText.innerHTML = `<div style="color: var(--danger-color);">Failed to transcribe handwriting. ${error.message}</div>`;
            transcriptionActions.style.display = 'block';
        }
    }
}

// Show in Context button
const showContextBtn = document.getElementById('show-context-btn');
const contextDialog = document.getElementById('context-dialog');
const closeContext = document.getElementById('close-context');
const contextPageImage = document.getElementById('context-page-image');
const contextHighlight = document.getElementById('context-highlight');

showContextBtn.addEventListener('click', async () => {
    if (!currentProblem) {
        alert('No problem loaded');
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/problems/${currentProblem.id}/context`);

        if (!response.ok) {
            if (response.status === 400) {
                alert('Context view not available for this problem (uses legacy storage)');
            } else {
                throw new Error(`Failed to fetch context: ${response.statusText}`);
            }
            return;
        }

        const data = await response.json();

        // Display the full page image
        contextPageImage.src = `data:image/png;base64,${data.page_image}`;

        // Wait for image to load before positioning highlight
        contextPageImage.onload = () => {
            const imgNaturalHeight = contextPageImage.naturalHeight;
            const imgNaturalWidth = contextPageImage.naturalWidth;
            const displayHeight = contextPageImage.offsetHeight;
            const displayWidth = contextPageImage.offsetWidth;

            // Scale region coordinates to displayed image size
            const scaleY = displayHeight / imgNaturalHeight;

            const highlightTop = data.problem_region.y_start * scaleY;
            const highlightHeight = (data.problem_region.y_end - data.problem_region.y_start) * scaleY;

            // Position the highlight box
            contextHighlight.style.top = `${highlightTop}px`;
            contextHighlight.style.left = '0px';
            contextHighlight.style.width = `${displayWidth}px`;
            contextHighlight.style.height = `${highlightHeight}px`;
        };

        // Show the dialog
        contextDialog.style.display = 'flex';

    } catch (error) {
        console.error('Failed to show context:', error);
        alert('Failed to load context view. Please try again.');
    }
});

closeContext.addEventListener('click', () => {
    contextDialog.style.display = 'none';
});

// Close on background click
contextDialog.addEventListener('click', (e) => {
    if (e.target === contextDialog) {
        contextDialog.style.display = 'none';
    }
});

// Decipher handwriting button (defaults to Anthropic)
decipherBtn.addEventListener('click', async () => {
    if (!currentProblem) {
        alert('No problem loaded');
        return;
    }

    // Show dialog with loading state
    transcriptionText.innerHTML = '<div class="transcription-loading">Transcribing handwriting with Anthropic...</div>';
    transcriptionActions.style.display = 'none';
    transcriptionDialog.style.display = 'flex';

    try {
        // Default to 'default' which uses Anthropic Sonnet-family candidates
        const transcription = await fetchTranscription(currentProblem.id, 'default');
        displayTranscription(transcription);
    } catch (error) {
        console.error('Failed to decipher handwriting:', error);
        transcriptionText.innerHTML = `<div style="color: var(--danger-color);">Failed to transcribe handwriting. ${error.message}</div>`;

        // Show model selection buttons
        transcriptionActions.style.display = 'block';
        transcriptionActions.innerHTML = `
            <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px;">
                <button id="retry-sonnet-btn" class="btn btn-secondary" style="flex: 1; min-width: 120px;">
                    Try Sonnet
                </button>
                <button id="retry-opus-btn" class="btn btn-primary" style="flex: 1; min-width: 120px;">
                    Try Opus (Premium)
                </button>
            </div>
        `;

        // Add event listeners for the buttons
        document.getElementById('retry-sonnet-btn').addEventListener('click', () => retryWithModel('sonnet'));
        document.getElementById('retry-opus-btn').addEventListener('click', () => retryWithModel('opus'));
    }
});

// =============================================================================
// SHOW ANSWER FUNCTIONALITY
// =============================================================================

const showAnswerBtn = document.getElementById('show-answer-btn');
const answerDialog = document.getElementById('answer-dialog');
const closeAnswerX = document.getElementById('close-answer-x');

// Make answer dialog draggable
let isAnswerDragging = false;
let answerDragOffsetX = 0;
let answerDragOffsetY = 0;

document.querySelector('.answer-header').addEventListener('mousedown', (e) => {
    if (e.target.classList.contains('answer-close')) return;
    isAnswerDragging = true;
    const rect = answerDialog.getBoundingClientRect();
    answerDragOffsetX = e.clientX - rect.left;
    answerDragOffsetY = e.clientY - rect.top;
    answerDialog.style.transform = 'none';
});

document.addEventListener('mousemove', (e) => {
    if (!isAnswerDragging) return;
    answerDialog.style.left = (e.clientX - answerDragOffsetX) + 'px';
    answerDialog.style.top = (e.clientY - answerDragOffsetY) + 'px';
});

document.addEventListener('mouseup', () => {
    isAnswerDragging = false;
});

function setAnswerDialogLoadingState() {
    const answerContent = document.getElementById('answer-content');
    const answerList = document.getElementById('answer-list');
    const answerError = document.getElementById('answer-error');
    const answerMetadata = document.getElementById('answer-metadata');

    answerContent.style.display = 'block';
    answerError.style.display = 'none';
    answerList.innerHTML = '<div style="text-align: center; padding: 20px;">Loading answer...</div>';
    answerMetadata.style.display = 'none';
}

function showAnswerDialogError(message) {
    const answerContent = document.getElementById('answer-content');
    const answerError = document.getElementById('answer-error');
    answerContent.style.display = 'none';
    answerError.style.display = 'block';
    answerError.textContent = message;
}

function renderAnswerDialogData(data) {
    const answerList = document.getElementById('answer-list');
    const answerMetadata = document.getElementById('answer-metadata');

    document.getElementById('answer-question-type').textContent = data.question_type;
    document.getElementById('answer-seed').textContent = data.seed;
    document.getElementById('answer-version').textContent = data.version;
    document.getElementById('answer-max-points').textContent = data.max_points;

    const configWrapper = document.getElementById('answer-config-wrapper');
    const configSpan = document.getElementById('answer-config');
    configSpan.textContent = data.config ? JSON.stringify(data.config) : 'None';
    configWrapper.style.display = 'block';

    answerMetadata.style.display = 'block';

    if (data.answer_key_html) {
        answerList.innerHTML = `<div style="padding: 15px; background: white; border-radius: 4px;">${data.answer_key_html}</div>`;
    } else if (data.answers && data.answers.length > 0) {
        answerList.innerHTML = data.answers.map(answer => {
            let html = `<div style="margin-bottom: 15px; padding: 10px; background: white; border-radius: 4px;">`;
            html += `<div style="font-weight: 600; color: #1e40af; margin-bottom: 5px;">${answer.key}:</div>`;
            if (answer.html) {
                html += `<div style="font-size: 18px; font-family: 'Courier New', monospace;">${answer.html}</div>`;
            } else {
                html += `<div style="font-size: 18px; font-family: 'Courier New', monospace;">${answer.value}</div>`;
            }
            if (answer.tolerance !== undefined && answer.tolerance !== null) {
                html += `<div style="font-size: 12px; color: #6b7280; margin-top: 5px;">Tolerance: ±${answer.tolerance}</div>`;
            }
            html += `</div>`;
            return html;
        }).join('');
    } else {
        answerList.innerHTML = '<div style="color: #6b7280;">No answers available</div>';
    }

    if (typeof MathJax !== 'undefined') {
        MathJax.typesetPromise([answerList]).catch((err) => console.error('MathJax typesetting failed:', err));
    }
}

// Function to update answer dialog with current problem
async function updateAnswerDialog() {
    if (!currentProblem) {
        answerDialog.style.display = 'none';
        return;
    }

    // Check if current problem has QR data
    if (!currentProblem.has_qr_data) {
        showAnswerDialogError('Answer not available for this problem (no QR code data)');
        return;
    }

    setAnswerDialogLoadingState();

    try {
        const data = await getRegeneratedAnswer(currentProblem.id);
        renderAnswerDialogData(data);

    } catch (error) {
        console.error('Failed to load answer:', error);
        showAnswerDialogError(error.message);
    }
}

// Show answer button
showAnswerBtn.addEventListener('click', async () => {
    if (!currentProblem) {
        alert('No problem loaded');
        return;
    }

    answerDialog.style.display = 'flex';
    await updateAnswerDialog();
});

// Close answer dialog
closeAnswerX.addEventListener('click', () => {
    answerDialog.style.display = 'none';
});

// Close on background click
answerDialog.addEventListener('click', (e) => {
    if (e.target === answerDialog) {
        answerDialog.style.display = 'none';
    }
});

// =============================================================================
// MANUAL QR PAYLOAD FUNCTIONALITY
// =============================================================================

const manualQrBtn = document.getElementById('manual-qr-btn');
const manualQrDialog = document.getElementById('manual-qr-dialog');
const manualQrInput = document.getElementById('manual-qr-input');
const manualQrStatus = document.getElementById('manual-qr-status');
const manualQrCancelBtn = document.getElementById('manual-qr-cancel-btn');
const manualQrApplyBtn = document.getElementById('manual-qr-apply-btn');

function setManualQrStatus(message, isError = false) {
    manualQrStatus.style.display = 'block';
    manualQrStatus.style.color = isError ? 'var(--danger-color)' : 'var(--gray-700)';
    manualQrStatus.textContent = message;
}

function resetManualQrDialogState() {
    manualQrStatus.style.display = 'none';
    manualQrStatus.textContent = '';
    manualQrStatus.style.color = 'var(--gray-700)';
}

function closeManualQrDialog() {
    manualQrDialog.style.display = 'none';
    resetManualQrDialogState();
}

manualQrBtn.addEventListener('click', () => {
    if (!currentProblem) {
        alert('No problem loaded');
        return;
    }

    resetManualQrDialogState();
    manualQrDialog.style.display = 'flex';
    manualQrInput.focus();
});

manualQrCancelBtn.addEventListener('click', closeManualQrDialog);

manualQrDialog.addEventListener('click', (e) => {
    if (e.target === manualQrDialog) {
        closeManualQrDialog();
    }
});

manualQrApplyBtn.addEventListener('click', async () => {
    if (!currentProblem) {
        alert('No problem loaded');
        return;
    }

    const payloadText = manualQrInput.value.trim();
    if (!payloadText) {
        setManualQrStatus('Paste a decoded QR JSON payload first.', true);
        return;
    }

    const problemId = currentProblem.id;
    const problemNumber = currentProblem.problem_number;

    manualQrApplyBtn.disabled = true;
    manualQrCancelBtn.disabled = true;
    setManualQrStatus('Applying payload...');

    try {
        const response = await fetch(`${API_BASE}/problems/${problemId}/manual-qr`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ payload_text: payloadText })
        });

        let payload = null;
        try {
            payload = await response.json();
        } catch (_error) {
            payload = null;
        }

        if (!response.ok) {
            throw new Error(payload?.detail || 'Failed to apply manual QR payload');
        }

        invalidateRegeneratedAnswer(problemId);
        problemMaxPoints[problemNumber] = payload.max_points;

        if (currentProblem && currentProblem.id === problemId) {
            currentProblem.max_points = payload.max_points;
            currentProblem.has_qr_data = Boolean(payload.has_qr_data);
            updateMaxPointsDropdown();
            document.getElementById('show-answer-btn').style.display =
                currentProblem.has_qr_data ? 'inline-block' : 'none';

            await loadExplanation({ eager: true });
        }

        closeManualQrDialog();
        showNotification(payload.message || 'Manual QR payload applied.');
    } catch (error) {
        console.error('Failed to apply manual QR payload:', error);
        setManualQrStatus(error.message || 'Failed to apply manual QR payload.', true);
    } finally {
        manualQrApplyBtn.disabled = false;
        manualQrCancelBtn.disabled = false;
    }
});


// =============================================================================
// AUTOGRADING FUNCTIONALITY
// =============================================================================

let autogradingEventSource = null;
let autogradeAllProblems = false;
let autogradingAllRunActive = false;

const startAutogradeBtn = document.getElementById('start-autograde-btn');
const startAutogradeAllBtn = document.getElementById('run-autograde-all-btn');
const autogradingModeModal = document.getElementById('autograding-mode-modal');
const autogradingModeCancelBtn = document.getElementById('autograding-mode-cancel-btn');
const autogradingModeContinueBtn = document.getElementById('autograding-mode-continue-btn');
const autogradingImageModal = document.getElementById('autograding-image-modal');
const autogradingImageCancelBtn = document.getElementById('autograding-image-cancel-btn');
const autogradingImageDryRunBtn = document.getElementById('autograding-image-dry-run-btn');
const autogradingImageStartBtn = document.getElementById('autograding-image-start-btn');
const autogradingImageBatchSize = document.getElementById('autograding-image-batch-size');
const autogradingImageQuality = document.getElementById('autograding-image-quality');
const autogradingImageIncludeAnswer = document.getElementById('autograding-image-include-answer');
const autogradingImageIncludeFeedback = document.getElementById('autograding-image-include-feedback');
const autogradingImageAutoAccept = document.getElementById('autograding-image-auto-accept');

async function startAutogradingTextRubricFlow() {
    if (!currentSession || !currentProblemNumber) return;

    // Show modal with extract phase
    const modal = document.getElementById('autograding-modal');
    const extractPhase = document.getElementById('autograding-extract-phase');
    const verifyPhase = document.getElementById('autograding-verify-phase');
    const progressPhase = document.getElementById('autograding-progress-phase');

    modal.style.display = 'flex';
    extractPhase.style.display = 'block';
    verifyPhase.style.display = 'none';
    progressPhase.style.display = 'none';

    try {
        // Extract question text
        const response = await fetch(`${API_BASE}/ai-grader/${currentSession.id}/extract-question`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ problem_number: currentProblemNumber })
        });

        if (!response.ok) {
            throw new Error('Failed to extract question text');
        }

        const data = await response.json();

        // Show verify phase
        extractPhase.style.display = 'none';
        verifyPhase.style.display = 'block';

        const questionTextArea = document.getElementById('autograding-question-text');
        questionTextArea.value = data.question_text;

        // Try to load existing rubric if available
        try {
            const rubricResponse = await fetch(`${API_BASE}/ai-grader/${currentSession.id}/rubric/${currentProblemNumber}`);
            if (rubricResponse.ok) {
                const rubricData = await rubricResponse.json();
                if (rubricData.rubric) {
                    // Render as table (rubric-table.js provides this function)
                    renderRubricTable(rubricData.rubric);
                }
            }
        } catch (error) {
            console.log('No existing rubric found, starting fresh');
        }

    } catch (error) {
        console.error('Failed to extract question:', error);
        modal.style.display = 'none';
        showNotification(`Failed to extract question: ${error.message}`);
    }
}

// Start Autograding button: choose mode first
startAutogradeBtn.addEventListener('click', () => {
    if (!currentSession || !currentProblemNumber) return;
    autogradeAllProblems = false;
    autogradingModeModal.style.display = 'flex';
});

startAutogradeAllBtn.addEventListener('click', () => {
    if (!currentSession) return;
    autogradeAllProblems = true;
    autogradingModeModal.style.display = 'flex';
});

autogradingModeCancelBtn.addEventListener('click', () => {
    autogradingModeModal.style.display = 'none';
    autogradeAllProblems = false;
});

autogradingModeContinueBtn.addEventListener('click', () => {
    const selected = document.querySelector('input[name="autograding-mode"]:checked');
    const mode = selected ? selected.value : 'text-rubric';

    autogradingModeModal.style.display = 'none';

    if (autogradeAllProblems && mode !== 'image-only') {
        showNotification('Running AI on all problems currently supports image-only mode.');
        autogradeAllProblems = false;
        return;
    }

    if (mode === 'text-rubric') {
        startAutogradingTextRubricFlow();
    } else {
        if (autogradeAllProblems) {
            autogradingImageAutoAccept.checked = true;
        }
        autogradingImageModal.style.display = 'flex';
    }
});

autogradingImageCancelBtn.addEventListener('click', () => {
    autogradingImageModal.style.display = 'none';
    autogradeAllProblems = false;
});

async function startImageOnlyAutograding(dryRun, allProblems = false) {
    if (!currentSession) return;
    if (!allProblems && !currentProblemNumber) return;

    const settings = {
        batch_size: autogradingImageBatchSize.value,
        image_quality: autogradingImageQuality.value,
        include_answer: autogradingImageIncludeAnswer.checked,
        include_default_feedback: autogradingImageIncludeFeedback.checked,
        auto_accept: autogradingImageAutoAccept.checked,
        dry_run: dryRun
    };

    autogradingImageModal.style.display = 'none';
    autogradingAllRunActive = allProblems;
    autogradeAllProblems = false;

    try {
        const modal = document.getElementById('autograding-modal');
        const extractPhase = document.getElementById('autograding-extract-phase');
        const verifyPhase = document.getElementById('autograding-verify-phase');
        const progressPhase = document.getElementById('autograding-progress-phase');

        modal.style.display = 'flex';
        extractPhase.style.display = 'none';
        verifyPhase.style.display = 'none';
        progressPhase.style.display = 'block';

        document.getElementById('autograding-progress-message').textContent =
            dryRun
                ? 'Starting image-only autograding dry run...'
                : (allProblems ? 'Starting image-only autograding for all problems...' : 'Starting image-only autograding...');
        document.getElementById('autograding-progress-bar').style.width = '0%';
        document.getElementById('autograding-current').textContent = '0';
        document.getElementById('autograding-total').textContent = '0';

        connectToAutogradingStream();

        const endpoint = allProblems
            ? `${API_BASE}/ai-grader/${currentSession.id}/autograde-all`
            : `${API_BASE}/ai-grader/${currentSession.id}/autograde`;
        const payload = allProblems
            ? { mode: 'image-only', settings }
            : { mode: 'image-only', problem_number: currentProblemNumber, settings };
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to start image-only autograding');
        }

        const data = await response.json();
        if (data.status === 'not_implemented') {
            modal.style.display = 'none';
            showNotification(data.message);
        }
    } catch (error) {
        console.error('Failed to start image-only autograding:', error);
        document.getElementById('autograding-modal').style.display = 'none';
        showNotification(`Failed to start image-only autograding: ${error.message}`);
        autogradingAllRunActive = false;
    }
}

autogradingImageStartBtn.addEventListener('click', () => {
    startImageOnlyAutograding(false, autogradeAllProblems);
});

autogradingImageDryRunBtn.addEventListener('click', () => {
    startImageOnlyAutograding(true, autogradeAllProblems);
});

// Cancel autograding
document.getElementById('autograding-cancel-btn').onclick = () => {
    const modal = document.getElementById('autograding-modal');
    modal.style.display = 'none';
};

// Generate rubric button
document.getElementById('generate-rubric-btn').onclick = async () => {
    const questionText = document.getElementById('autograding-question-text').value;

    if (!questionText.trim()) {
        alert('Please enter the question text first');
        return;
    }

    // Get max points from the UI
    const maxPointsInput = document.getElementById('max-points-input');
    const maxPoints = parseFloat(maxPointsInput.value) || 8;

    // Show loading state
    const rubricTextarea = document.getElementById('autograding-rubric-text');
    const rubricLoading = document.getElementById('rubric-loading');
    const generateBtn = document.getElementById('generate-rubric-btn');

    rubricLoading.style.display = 'block';
    rubricTextarea.style.display = 'none';
    generateBtn.disabled = true;

    try {
        // Generate rubric
        const response = await fetch(`${API_BASE}/ai-grader/${currentSession.id}/generate-rubric`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                problem_number: currentProblemNumber,
                question_text: questionText,
                max_points: maxPoints,
                num_examples: 3
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to generate rubric');
        }

        const data = await response.json();
        rubricTextarea.value = data.rubric;

    } catch (error) {
        console.error('Failed to generate rubric:', error);
        alert(`Failed to generate rubric: ${error.message}\n\nMake sure you have manually graded at least 3 submissions for this problem first.`);
    } finally {
        rubricLoading.style.display = 'none';
        rubricTextarea.style.display = 'block';
        generateBtn.disabled = false;
    }
};

// Confirm and start autograding
document.getElementById('autograding-confirm-btn').onclick = async () => {
    const questionText = document.getElementById('autograding-question-text').value;
    const rubricText = document.getElementById('autograding-rubric-text').value;

    if (!questionText.trim()) {
        alert('Please enter the question text');
        return;
    }

    // Get max points from the UI
    const maxPointsInput = document.getElementById('max-points-input');
    const maxPoints = parseFloat(maxPointsInput.value) || 8; // Default to 8 if not set

    // Hide verify phase, show progress phase
    const verifyPhase = document.getElementById('autograding-verify-phase');
    const progressPhase = document.getElementById('autograding-progress-phase');
    verifyPhase.style.display = 'none';
    progressPhase.style.display = 'block';

    // Connect to SSE stream before starting
    connectToAutogradingStream();

    try {
        // Save rubric if provided
        if (rubricText.trim()) {
            await fetch(`${API_BASE}/ai-grader/${currentSession.id}/save-rubric`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    problem_number: currentProblemNumber,
                    rubric: rubricText
                })
            });
        }

        // Start autograding
        const response = await fetch(`${API_BASE}/ai-grader/${currentSession.id}/autograde`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                mode: 'text-rubric',
                problem_number: currentProblemNumber,
                question_text: questionText,
                max_points: maxPoints
            })
        });

        if (!response.ok) {
            throw new Error('Failed to start autograding');
        }

        const data = await response.json();
        console.log('Autograding started:', data);

    } catch (error) {
        console.error('Failed to start autograding:', error);
        const modal = document.getElementById('autograding-modal');
        modal.style.display = 'none';
        showNotification(`Failed to start autograding: ${error.message}`);
    }
};

function connectToAutogradingStream() {
    const progressMessage = document.getElementById('autograding-progress-message');
    const progressBar = document.getElementById('autograding-progress-bar');
    const currentEl = document.getElementById('autograding-current');
    const totalEl = document.getElementById('autograding-total');

    // Close existing connection if any
    if (autogradingEventSource) {
        autogradingEventSource.close();
    }

    // Connect to SSE stream
    const streamUrl = `${API_BASE}/ai-grader/${currentSession.id}/autograde-stream`;
    autogradingEventSource = new EventSource(streamUrl);

    autogradingEventSource.addEventListener('connected', (e) => {
        console.log('SSE connected for autograding progress');
    });

    autogradingEventSource.addEventListener('start', (e) => {
        const data = JSON.parse(e.data);
        console.log('Autograding started:', data);
        progressMessage.textContent = data.message;
    });

    autogradingEventSource.addEventListener('progress', (e) => {
        const data = JSON.parse(e.data);
        console.log('Autograding progress:', data);

        progressMessage.textContent = data.message;
        progressBar.style.width = `${data.progress}%`;
        progressBar.textContent = `${data.progress}%`;
        currentEl.textContent = data.current;
        totalEl.textContent = data.total;
    });

    autogradingEventSource.addEventListener('complete', async (e) => {
        const data = JSON.parse(e.data);
        console.log('Autograding complete:', data);

        autogradingEventSource.close();
        autogradingEventSource = null;

        // Complete the progress bar
        progressBar.style.width = '100%';
        progressBar.textContent = '100%';
        progressMessage.textContent = data.message;

        // Close modal after a brief delay
        setTimeout(() => {
            const modal = document.getElementById('autograding-modal');
            modal.style.display = 'none';

            // Show completion message
            if (autogradingAllRunActive) {
                const message = `Autograding complete! ${data.graded} of ${data.total} problems graded.`;
                showNotification(message, async () => {
                    await loadStatistics();
                });
                autogradingAllRunActive = false;
            } else {
                showNotification(`Autograding complete! ${data.graded} of ${data.total} problems graded. Please review the AI suggestions.`, async () => {
                    // Reload current problem to show AI suggestion
                    await loadProblemOrMostRecent();
                });
            }
        }, 2000);
    });

    autogradingEventSource.addEventListener('error', (e) => {
        console.error('SSE error:', e);
        if (autogradingEventSource && autogradingEventSource.readyState === EventSource.CLOSED) {
            console.log('SSE connection closed');
            autogradingEventSource = null;
        } else {
            progressMessage.textContent = 'Connection error - autograding may still be running';
        }
        autogradingAllRunActive = false;
    });
}

// =============================================================================
// PROBLEM IMAGE RESIZE FUNCTIONALITY
// =============================================================================

function setupProblemImageResize() {
    const scrollContainer = document.getElementById('problem-scroll-container');
    const resizeHandle = document.getElementById('problem-resize-handle');

    if (!scrollContainer || !resizeHandle) {
        console.warn('Problem scroll container or resize handle not found');
        return;
    }

    // Load saved height from localStorage
    const savedHeight = localStorage.getItem('problemScrollContainerHeight');
    if (savedHeight) {
        scrollContainer.style.height = savedHeight;
    }

    let isResizing = false;
    let startY = 0;
    let startHeight = 0;

    // Mouse down on resize handle
    resizeHandle.addEventListener('mousedown', (e) => {
        isResizing = true;
        startY = e.clientY;
        startHeight = scrollContainer.offsetHeight;

        // Prevent text selection during resize
        e.preventDefault();
        document.body.style.userSelect = 'none';
        document.body.style.cursor = 'ns-resize';
    });

    // Mouse move - resize
    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;

        const deltaY = e.clientY - startY;
        const newHeight = startHeight + deltaY;

        // Enforce minimum and maximum heights
        const minHeight = 200; // Minimum 200px
        const maxHeight = scrollContainer.dataset.maxImageHeight
            ? parseFloat(scrollContainer.dataset.maxImageHeight)
            : window.innerHeight * 0.9; // Fallback to 90% of viewport if not set

        if (newHeight >= minHeight && newHeight <= maxHeight) {
            scrollContainer.style.height = `${newHeight}px`;
        } else if (newHeight < minHeight) {
            scrollContainer.style.height = `${minHeight}px`;
        } else if (newHeight > maxHeight) {
            scrollContainer.style.height = `${maxHeight}px`;
        }
    });

    // Mouse up - stop resizing and save height
    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            document.body.style.userSelect = '';
            document.body.style.cursor = '';

            // Save the height to localStorage
            const currentHeight = scrollContainer.style.height;
            localStorage.setItem('problemScrollContainerHeight', currentHeight);
            console.log('Saved problem container height:', currentHeight);
        }
    });
}

// =============================================================================
// STUDENT TABLE SORTING
// =============================================================================

let currentSortColumn = null;
let currentSortDirection = 'asc';

function sortStudentTable(column) {
    const tbody = document.getElementById('student-scores-tbody');
    if (!tbody) return;

    const rows = Array.from(tbody.querySelectorAll('tr'));

    // Toggle direction if clicking same column, otherwise default to ascending
    if (currentSortColumn === column) {
        currentSortDirection = currentSortDirection === 'asc' ? 'desc' : 'asc';
    } else {
        currentSortColumn = column;
        currentSortDirection = 'asc';
    }

    // Sort rows based on column and direction
    rows.sort((a, b) => {
        let aVal, bVal;

        if (column === 'name') {
            aVal = a.dataset.name.toLowerCase();
            bVal = b.dataset.name.toLowerCase();
            return currentSortDirection === 'asc'
                ? aVal.localeCompare(bVal)
                : bVal.localeCompare(aVal);
        } else if (column === 'progress') {
            aVal = parseFloat(a.dataset.progress);
            bVal = parseFloat(b.dataset.progress);
        } else if (column === 'score') {
            aVal = parseFloat(a.dataset.score);
            bVal = parseFloat(b.dataset.score);
        }

        // Numerical comparison
        if (currentSortDirection === 'asc') {
            return aVal - bVal;
        } else {
            return bVal - aVal;
        }
    });

    // Re-append rows in sorted order
    rows.forEach(row => tbody.appendChild(row));

    // Update sort indicators
    updateSortIndicators(column);
}

function updateSortIndicators(column) {
    // Clear all indicators
    document.querySelectorAll('.student-scores-table th .sort-indicator').forEach(indicator => {
        indicator.textContent = '';
    });

    // Set current indicator
    const th = document.querySelector(`.student-scores-table th[data-sort="${column}"]`);
    if (th) {
        const indicator = th.querySelector('.sort-indicator');
        if (indicator) {
            indicator.textContent = currentSortDirection === 'asc' ? ' ▲' : ' ▼';
        }
    }
}

// =============================================================================
// EXPLANATION LOADING AND AUTO-INCLUDE IN FEEDBACK
// =============================================================================

// Cache for explanations { problemId: { html?: string, markdown?: string } }
let explanationCache = {};

// Load explanation UI state, optionally warming cache for the current problem.
async function loadExplanation({ eager = false } = {}) {
    const controls = document.getElementById('explanation-controls');
    const viewBtn = document.getElementById('view-explanation-btn');

    if (!currentProblem || !currentProblem.has_qr_data) {
        // No QR data - hide explanation controls
        if (controls) controls.style.display = 'none';
        return;
    }

    if (controls) controls.style.display = 'block';
    if (viewBtn) {
        viewBtn.disabled = false;
        viewBtn.textContent = 'View Explanation';
    }

    if (!eager) {
        return;
    }

    if (viewBtn) {
        viewBtn.disabled = true;
        viewBtn.textContent = 'Loading...';
    }

    const ok = await ensureExplanationLoaded(currentProblem.id);
    if (!ok) {
        if (controls) controls.style.display = 'none';
        return;
    }

    if (viewBtn) {
        viewBtn.disabled = false;
        viewBtn.textContent = 'View Explanation';
    }
}

async function showExplanationDialog() {
    const dialog = document.getElementById('explanation-dialog');
    const content = document.getElementById('explanation-dialog-content');

    if (!currentProblem) {
        showNotification('Explanation not available for this problem.');
        return;
    }

    if (!explanationCache[currentProblem.id]) {
        content.innerHTML = '<div style="padding: 20px; text-align: center;">Loading explanation...</div>';
        dialog.style.display = 'flex';
        const loaded = await ensureExplanationLoaded(currentProblem.id);
        if (!loaded) {
            dialog.style.display = 'none';
            showNotification('Explanation not available for this problem.');
            return;
        }
    }

    const explanation = explanationCache[currentProblem.id];
    const htmlContent = explanation.html
        ? explanation.html
        : marked.parse(explanation.markdown || '');
    content.innerHTML = htmlContent;
    dialog.style.display = 'flex';

    if (typeof MathJax !== 'undefined') {
        MathJax.typesetPromise([content]).catch((err) => console.error('MathJax typesetting failed:', err));
    }
}

document.getElementById('view-explanation-btn')?.addEventListener('click', async () => {
    await showExplanationDialog();
});

document.getElementById('close-explanation-x')?.addEventListener('click', () => {
    const dialog = document.getElementById('explanation-dialog');
    dialog.style.display = 'none';
});
