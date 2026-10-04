(() => {
  const sessionsElement = document.querySelector('#sessions');
  const statusFilter = document.querySelector('#status-filter');
  const bucketsElement = document.querySelector('#buckets');
  const resultsPanel = document.querySelector('#results-panel');
  const resultsElement = document.querySelector('#results');
  let sessions = [];
  const selectedSessionIds = new Set();
  let buckets = [
    {name: 'Long response (6 & 8 points)', points: '6, 8'},
    {name: 'Medium response (4 points)', points: '4'},
    {name: 'Short response (1 & 2 points)', points: '1, 2'},
  ];
  const escapeHtml = value => String(value ?? '').replace(/[&<>'"]/g, character => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'})[character]);
  async function api(url, options) { const response = await fetch(url, options); const body = await response.json().catch(() => ({})); if (!response.ok) throw new Error(body.detail || 'Request failed'); return body; }
  function sessionName(session) { return session.session_name || session.assignment_name || `Session ${session.id}`; }
  function renderSessions() { const visible = sessions.filter(session => !statusFilter.value || session.status === statusFilter.value); sessionsElement.innerHTML = visible.map(session => `<label class="session-option"><input type="checkbox" value="${session.id}" ${selectedSessionIds.has(session.id) ? 'checked' : ''}><span><strong>${escapeHtml(sessionName(session))}</strong> <small>(Session ${session.id})</small><br><small>${escapeHtml(session.course_name || 'No course name')} · <span class="status status-${escapeHtml(session.status || 'unknown')}">${escapeHtml(session.status || 'unknown')}</span></small></span></label>`).join('') || 'No sessions match this status.'; sessionsElement.querySelectorAll('input').forEach(input => input.addEventListener('change', () => { const id = Number(input.value); if (input.checked) selectedSessionIds.add(id); else selectedSessionIds.delete(id); })); }
  function renderBuckets() { bucketsElement.innerHTML = buckets.map((bucket, index) => `<div class="bucket"><input data-field="name" data-index="${index}" value="${escapeHtml(bucket.name)}" aria-label="Bucket name"><input data-field="points" data-index="${index}" value="${escapeHtml(bucket.points)}" aria-label="Point values"><button class="danger" data-remove="${index}" type="button">Remove</button></div>`).join(''); bucketsElement.querySelectorAll('input').forEach(input => input.addEventListener('input', () => { buckets[Number(input.dataset.index)][input.dataset.field] = input.value; })); bucketsElement.querySelectorAll('[data-remove]').forEach(button => button.addEventListener('click', () => { buckets.splice(Number(button.dataset.remove), 1); renderBuckets(); })); }
  function fmt(value) { return value === null || value === undefined ? '—' : `${(value * 100).toFixed(1)}%`; }
  function renderResults(rows, examDistributions = []) {
    resultsPanel.hidden = false;
    const sessionOrder = [...new Map(rows.map(row => [row.session.id, row.session])).values()];
    const bucketOrder = [...new Map(rows.map(row => [row.bucket, row])).values()];
    const rowFor = (bucket, sessionId) => rows.find(row => row.bucket === bucket && row.session.id === sessionId);
    const percent = value => value === null || value === undefined ? '—' : `${value.toFixed(1)}%`;
    const metricRow = (label, bucket, display) => `<tr><th class="metric-label">${label}</th>${sessionOrder.map(session => `<td>${display(rowFor(bucket, session.id))}</td>`).join('')}</tr>`;
    const sections = bucketOrder.map(bucketRow => {
      const bucket = bucketRow.bucket;
      const distributionRows = bucketRow.distribution.map((bin, index) => metricRow(`Distribution: ${bin.label}`, bucket, row => {
        if (!row) return '—';
        const binPercentage = row.distribution[index].percentage;
        const cumulative = row.distribution.slice(0, index + 1)
          .reduce((sum, item) => sum + item.percentage, 0);
        return `${percent(binPercentage)} (${percent(cumulative)})`;
      })).join('');
      const pointDescription = bucketRow.point_values.length ? `${bucketRow.point_values.join(', ')} point questions` : 'all positive-point questions';
      return `<tr class="bucket-heading"><td colspan="${sessionOrder.length + 1}">${escapeHtml(bucket)} · ${pointDescription}</td></tr>` +
        metricRow('Questions included', bucket, row => row?.question_numbers.length ? `Q${row.question_numbers.join(', Q')}` : '—') +
        metricRow('Responses', bucket, row => row ? `${row.scored_count} scored / ${row.response_count} total` : '—') +
        metricRow('Normalized score — mean', bucket, row => fmt(row?.mean_normalized)) +
        metricRow('Normalized score — stddev', bucket, row => fmt(row?.stddev_normalized)) +
        metricRow('Blank rate', bucket, row => row ? `${percent(row.blank_percentage)} (${row.blank_count}/${row.response_count})` : '—') +
        metricRow('Distribution (visual)', bucket, row => row ? `<div class="hist" title="${row.distribution.map(bin => `${bin.label}: ${bin.count}`).join(', ')}">${row.distribution.map(bin => `<span style="width:${bin.percentage}%"></span>`).join('')}</div>` : '—') +
        distributionRows;
    }).join('');
    const bar = distribution => `<div class="hist" title="${distribution.map(bin => `${bin.label}: ${bin.count}`).join(', ')}">${distribution.map(bin => `<span style="width:${bin.percentage}%"></span>`).join('')}</div><small>${distribution.map(bin => `${bin.label}: ${bin.percentage.toFixed(0)}%`).join(' · ')}</small>`;
    const histogram = distribution => { const maximum = Math.max(...distribution.map(bin => bin.count), 1); return `<div class="histogram-chart" title="${distribution.map(bin => `${bin.label}: ${bin.count}`).join(', ')}">${distribution.map(bin => `<div class="histogram-bin"><div class="histogram-bar" style="height:${100 * bin.count / maximum}%"></div></div>`).join('')}</div><div class="histogram-labels">${distribution.map(bin => `<span>${escapeHtml(bin.label)}</span>`).join('')}</div>`; };
    const examGraphs = examDistributions.length ? `<h3>Exam score distributions</h3><div class="exam-distributions">${examDistributions.map(exam => `<div class="exam-card"><strong>${escapeHtml(exam.session.name)}</strong><br><small>Session ${exam.session.id} · ${exam.scored_exam_count}/${exam.exam_count} fully scored exams</small><p><strong>Equal-weight normalized score</strong><br>${fmt(exam.mean_normalized)} mean · ± ${fmt(exam.stddev_normalized)} stddev</p>${bar(exam.normalized_distribution)}<p><strong>Actual total score — 10-point bins</strong><br>${exam.mean_actual_score === null ? '—' : `${exam.mean_actual_score.toFixed(1)} / ${exam.max_total_points.toFixed(1)}`} mean · ± ${exam.stddev_actual_score === null ? '—' : exam.stddev_actual_score.toFixed(1)} stddev</p>${histogram(exam.actual_distribution)}</div>`).join('')}</div>` : '';
    resultsElement.innerHTML = `${examGraphs}<h3>Question-response comparison</h3><table><thead><tr><th class="metric-label">Metric</th>${sessionOrder.map(session => `<th>${escapeHtml(session.name)}<br><small>Session ${session.id}</small></th>`).join('')}</tr></thead><tbody>${sections}</tbody></table>`;
  }
  document.querySelector('#add-bucket').addEventListener('click', () => { buckets.push({name: 'New bucket', points: ''}); renderBuckets(); });
  document.querySelector('#compare').addEventListener('click', async () => { const session_ids = [...selectedSessionIds]; const normalizedBuckets = buckets.map(bucket => ({name: bucket.name.trim(), points: bucket.points.split(',').map(value => Number(value.trim())).filter(value => Number.isFinite(value))})).filter(bucket => bucket.name && bucket.points.length); if (!session_ids.length) return alert('Select at least one session.'); if (!normalizedBuckets.length) return alert('Define at least one bucket with point values.'); try { const data = await api('/api/session-comparison', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({session_ids, buckets: normalizedBuckets})}); renderResults(data.rows, data.exam_distributions); } catch (error) { alert(error.message); } });
  statusFilter.addEventListener('change', renderSessions);
  Promise.all([api('/api/sessions')]).then(([loaded]) => { sessions = loaded; [...new Set(sessions.map(session => session.status).filter(Boolean))].sort().forEach(status => { const option = document.createElement('option'); option.value = status; option.textContent = status; statusFilter.append(option); }); renderSessions(); renderBuckets(); }).catch(error => { sessionsElement.textContent = error.message; });
})();
