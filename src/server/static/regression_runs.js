(function () {
  const apiKeyInput = document.getElementById('apiKey');
  const saveBtn = document.getElementById('saveKey');
  const runsList = document.getElementById('runsList');
  const runDetail = document.getElementById('runDetail');

  function getApiKey() {
    return localStorage.getItem('REG_API_KEY') || '';
  }

  function setApiKey(v) {
    localStorage.setItem('REG_API_KEY', v || '');
  }

  saveBtn.addEventListener('click', () => {
    setApiKey(apiKeyInput.value.trim());
    loadRuns();
  });

  apiKeyInput.value = getApiKey();

  async function fetchJson(path) {
    const headers = {};
    const k = getApiKey();
    if (k) headers['X-API-Key'] = k;
    const res = await fetch(path, { headers });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    return await res.json();
  }

  function el(tag, cls) { const e = document.createElement(tag); if (cls) e.className = cls; return e; }

  async function loadRuns() {
    runsList.innerHTML = 'Loading...';
    runDetail.innerHTML = '';
    try {
      const data = await fetchJson('/regression/runs');
      const runs = data.runs || [];
      runsList.innerHTML = '';
      if (!runs.length) runsList.textContent = 'No runs found';
      runs.forEach(r => {
        const item = el('div', 'run-item');
        const title = el('div');
        title.innerHTML = `<strong>${r.run_id}</strong> — ${r.timestamp || ''}`;
        const meta = el('div', 'meta');
        meta.innerHTML = `Pass rate: <span class="${(r.pass_rate||0) >= 1 ? 'pass' : 'fail'}">${((r.pass_rate||0)*100).toFixed(0)}%</span> (${r.passed_cases || 0}/${r.total_cases || 0})`;
        item.appendChild(title);
        item.appendChild(meta);
        item.addEventListener('click', () => { showRun(r.run_id); });
        runsList.appendChild(item);
      });
    } catch (e) {
      runsList.innerHTML = 'Error loading runs: ' + e;
    }
  }

  async function showRun(runId) {
    runDetail.innerHTML = 'Loading run...';
    try {
      const data = await fetchJson(`/regression/runs/${runId}`);
      const html = [];
      html.push(`<h2>Run ${runId}</h2>`);
      html.push(`<div>Pass rate: ${((data.pass_rate||0)*100).toFixed(0)}% (${data.passed_cases||0}/${data.total_cases||0})</div>`);
      html.push('<h3>Failed cases</h3>');
      const failed = (data.results || []).filter(r => !r.passed);
      if (!failed.length) {
        html.push('<div>No failed cases</div>');
      } else {
        html.push('<div>');
        failed.forEach(c => {
          const traceId = c.trace && c.trace.trace_id ? c.trace.trace_id : (c.trace && c.trace.get && c.trace.get('trace_id')) || null;
          let line = `<div class="case"><strong>${c.id}</strong>: ${c.details || ''}`;
          if (traceId) {
            line += ` <a class="trace-link" href="/ui/traces/${traceId}" target="_blank">Open Trace</a>`;
          } else if (c.trace && typeof c.trace === 'object' && c.trace.trace_id) {
            line += ` <span class="trace-link">trace:${c.trace.trace_id}</span>`;
          }
          line += ` <div>${(c.response||'')}</div></div>`;
          html.push(line);
        });
        html.push('</div>');
      }
      runDetail.innerHTML = html.join('\n');
    } catch (e) {
      runDetail.innerHTML = 'Error loading run: ' + e;
    }
  }

  loadRuns();
})();
