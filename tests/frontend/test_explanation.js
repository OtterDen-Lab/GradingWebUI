const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const source = fs.readFileSync(path.join(__dirname, '../../src/grading_web_ui/web_frontend/js/grading.js'), 'utf8');
const start = source.indexOf('async function showExplanationDialog()');
const end = source.indexOf("document.getElementById('view-explanation-btn')", start);
const dialog = { style: { display: 'none' } };
const content = { innerHTML: '', textContent: '' };
const pending = new Map();
const context = {
    currentProblem: { id: 1, has_qr_data: true },
    explanationCache: {},
    document: { getElementById: (id) => id === 'explanation-dialog' ? dialog : content },
    ensureExplanationLoaded: (id) => new Promise((resolve) => pending.set(id, resolve)),
    marked: { parse: (text) => text },
    showNotification: () => {},
};
vm.createContext(context);
vm.runInContext(source.slice(start, end), context);

(async () => {
    const first = context.showExplanationDialog();
    context.currentProblem = { id: 2, has_qr_data: true };
    const second = context.showExplanationDialog();
    context.explanationCache[2] = { html: 'Second explanation' };
    pending.get(2)(true);
    await second;
    assert.equal(content.innerHTML, 'Second explanation');
    context.explanationCache[1] = { html: 'First explanation' };
    pending.get(1)(true);
    await first;
    assert.equal(content.innerHTML, 'Second explanation');

    context.currentProblem = { id: 3, has_qr_data: false };
    await context.showExplanationDialog();
    assert.equal(content.textContent, 'Explanation not available for this problem.');

    context.currentProblem = { id: 4, has_qr_data: true };
    const closed = context.showExplanationDialog();
    dialog.style.display = 'none';
    context.explanationCache[4] = { html: 'Late explanation' };
    pending.get(4)(true);
    await closed;
    assert.equal(dialog.style.display, 'none');
})().catch((error) => { console.error(error); process.exitCode = 1; });
