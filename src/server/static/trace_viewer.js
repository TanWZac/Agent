(function () {
  function getTraceIdFromPath() {
    const parts = window.location.pathname.split('/').filter(Boolean);
    return parts[parts.length - 1];
  }

  const traceId = getTraceIdFromPath();
  const metaEl = document.getElementById('meta');
  const eventsEl = document.getElementById('events');
  const timelineEl = document.getElementById('timeline');
  const statusEl = document.getElementById('status');
  const delayInput = document.getElementById('delay');
  const startBtn = document.getElementById('startBtn');

  let events = [];
  let startTime = null;
  let endTime = null;
  let es = null;

  function fmt(ts) {
    return new Date(ts * 1000).toISOString();
  }

  function renderMeta() {
    metaEl.innerHTML = `<strong>Trace:</strong> ${traceId}<br/>` +
      (startTime ? `<strong>Start:</strong> ${fmt(startTime)} ` : '') +
      (endTime ? ` <strong>End:</strong> ${fmt(endTime)}` : '');
  }

  function renderEvents() {
    eventsEl.innerHTML = '';
    timelineEl.innerHTML = '';
    if (!startTime) return;
    const nowSec = Date.now() / 1000;
    const total = (endTime || nowSec) - startTime || 1;
    events.forEach(ev => {
      const li = document.createElement('li');
      li.textContent = `[${fmt(ev.ts)}] ${ev.type} - ${JSON.stringify(ev.payload)}`;
      eventsEl.appendChild(li);

      const dot = document.createElement('div');
      dot.className = 'event-dot';
      const pos = Math.max(0, Math.min(1, (ev.ts - startTime) / total));
      dot.style.left = (pos * 100) + '%';
      dot.title = `${ev.type} ${JSON.stringify(ev.payload)}`;
      timelineEl.appendChild(dot);
    });
  }

  async function fetchTrace() {
    try {
      const r = await fetch(`/traces/${traceId}`);
      if (!r.ok) {
        statusEl.textContent = 'Trace not found';
        return;
      }
      const t = await r.json();
      startTime = t.start_time;
      endTime = t.end_time;
      renderMeta();
    } catch (e) {
      statusEl.textContent = 'Error fetching trace: ' + e;
    }
  }

  function startReplay() {
    if (es) try { es.close(); } catch (e) {}
    events = [];
    timelineEl.innerHTML = '';
    eventsEl.innerHTML = '';
    statusEl.textContent = 'Connecting...';
    const delay = parseInt(delayInput.value || '100', 10);
    const url = `/traces/${traceId}/replay?delay_ms=${delay}`;
    es = new EventSource(url);
    es.onopen = () => { statusEl.textContent = 'Connected, replaying'; };
    es.onerror = () => { statusEl.textContent = 'SSE error or closed'; es.close(); };
    es.onmessage = (msg) => {
      try {
        const data = JSON.parse(msg.data);
        if (data.type === 'trace_end') {
          endTime = data.end_time || endTime;
          renderMeta();
          renderEvents();
          statusEl.textContent = 'Replay finished';
          es.close();
          return;
        }
        events.push({ type: data.type, ts: data.ts, payload: data.payload });
        if (!startTime) { startTime = events[0].ts; renderMeta(); }
        renderEvents();
      } catch (e) {
        console.error('parse error', e);
      }
    };
  }

  startBtn.addEventListener('click', startReplay);
  fetchTrace().then(() => { startReplay(); });

})();
