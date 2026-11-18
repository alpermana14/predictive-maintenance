// Operator actions from the mixed-reality view (human in the loop) and the measurement log of AR sessions.
// Every action goes to the same backend endpoints as the dashboard and passes the same approver-token check, so the
// headset never gains more authority than a named approver at the dashboard. Nothing here acts on the machine.

export const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
// Demo deployment (VITE_APPROVAL_MODE=open): the backend skips the token check, as on the dashboard
const OPEN_APPROVAL = import.meta.env.VITE_APPROVAL_MODE === 'open';
// ?measure in the dashboard URL: AR sessions post checkpoints, frame-rate samples and action timings to /api/ar/log
// (the backend stores them only when AR_LOG_ENABLED=1)
export const MEASURE = typeof window !== 'undefined' && new URLSearchParams(window.location.search).has('measure');

// Measurement sessions only: ?measure&token=... stores the approver token for this tab and removes it from the URL,
// so the operator does not have to trigger a dashboard action (which would change the state) to enter it.
if (MEASURE) {
    const params = new URLSearchParams(window.location.search);
    const t = params.get('token');
    if (t) {
        try { sessionStorage.setItem('approverToken', t); } catch { /* storage blocked */ }
        params.delete('token');
        window.history.replaceState(null, '', `${window.location.pathname}?${params.toString()}`);
    }
}

// The approver token the dashboard asked for (kept in sessionStorage for this tab). Typing a token in the headset is
// impractical, so it is entered once on the dashboard before AR starts.
export function approverToken() {
    if (OPEN_APPROVAL) return 'open';
    try { return sessionStorage.getItem('approverToken') || ''; } catch { return ''; }
}

const serverMs = (res) => {
    const m = /app;dur=([\d.]+)/.exec(res.headers.get('Server-Timing') || '');
    return m ? parseFloat(m[1]) : null;
};

// POST with timing: t_cmd (sent), t_resp (response received), server processing time from Server-Timing
export async function timedPost(path, body, { token = true } = {}) {
    const headers = { 'Content-Type': 'application/json' };
    if (token) headers['X-Approver-Token'] = approverToken();
    const t_cmd = performance.now();
    let res, json = null, error = null;
    try {
        res = await fetch(`${API_URL}${path}`, { method: 'POST', headers, body: JSON.stringify(body) });
        try { json = await res.json(); } catch { /* empty body */ }
    } catch (e) {
        error = e?.message || String(e);
    }
    const t_resp = performance.now();
    return {
        ok: !!res?.ok, status: res?.status ?? 0, json, error,
        t_cmd, t_resp, rtt_ms: t_resp - t_cmd, server_ms: res ? serverMs(res) : null,
    };
}

export async function fetchSummary() {
    const res = await fetch(`${API_URL}/api/summary`, { cache: 'no-store' });
    if (!res.ok) throw new Error(`summary ${res.status}`);
    return res.json();
}

// Measurement log: batched, fire-and-forget; never blocks or breaks the AR session
const queue = [];
let timer = null;
export function logRecord(kind, data) {
    if (!MEASURE) return;
    queue.push({ kind, client_ms: performance.now(), client_utc: new Date().toISOString(), ...data });
    if (!timer) timer = setTimeout(flushLog, 1000);
}
export function flushLog() {
    timer = null;
    if (!queue.length) return;
    const batch = queue.splice(0, queue.length);
    fetch(`${API_URL}/api/ar/log`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(batch) })
        .catch(() => { /* measurement only */ });
}
