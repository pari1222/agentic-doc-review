/* ══ Config ══════════════════════════════════════════════════════════════════ */
const API = 'http://localhost:8000/api/v1';

/* ══ State ═══════════════════════════════════════════════════════════════════ */
let lastId  = null;
let lastDoc = '';

/* ══ Boot ════════════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  ping();
  setInterval(ping, 30000);

  const ta = document.getElementById('inputContent');
  const cc = document.getElementById('charCount');
  ta.addEventListener('input', () => {
    const n = ta.value.length;
    cc.textContent = `${n.toLocaleString()} chars`;
    cc.style.color = n < 10 ? 'var(--red)' : 'var(--n400)';
  });

  document.querySelectorAll('.nav-btn').forEach(b =>
    b.addEventListener('click', () => switchTab(b.dataset.tab))
  );
});

/* ══ Tab switching ═══════════════════════════════════════════════════════════ */
const TAB_META = {
  generate: ['Generate Document',  'Convert raw notes into structured professional documents'],
  retrieve: ['Retrieve Document',  'Fetch a previously generated document by its unique ID'],
  approve:  ['Human Approval',     'Approve or reject documents awaiting human review'],
};

function switchTab(tab) {
  document.querySelectorAll('.nav-btn').forEach(b  => b.classList.toggle('active', b.dataset.tab === tab));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.toggle('active', p.id === `tab-${tab}`));
  const [h, d] = TAB_META[tab];
  document.getElementById('pageHeading').textContent = h;
  document.getElementById('pageDesc').textContent    = d;
  if (tab === 'approve'  && lastId) document.getElementById('approveId').value  = lastId;
  if (tab === 'retrieve' && lastId) document.getElementById('retrieveId').value = lastId;
}

/* ══ Health ping ═════════════════════════════════════════════════════════════ */
async function ping() {
  const dot   = document.getElementById('statusDot');
  const label = document.getElementById('statusLabel');
  try {
    const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(4000) });
    dot.className    = `status-indicator ${r.ok ? 'online' : 'offline'}`;
    label.textContent = r.ok ? 'Online' : 'Degraded';
  } catch {
    dot.className     = 'status-indicator offline';
    label.textContent = 'Offline';
  }
}

/* ══ Generate ════════════════════════════════════════════════════════════════ */
async function generateDocument() {
  const content = document.getElementById('inputContent').value.trim();
  const type    = document.getElementById('docType').value;
  const btn     = document.getElementById('generateBtn');

  if (content.length < 10) { toast('Enter at least 10 characters.', true); return; }

  btn.disabled = true;
  document.getElementById('resultPanel').style.display = 'none';
  showLoader(true);
  animateSteps();

  try {
    const body = { content };
    if (type) body.document_type = type;

    const r    = await fetch(`${API}/generate-document`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'Generation failed');

    lastId  = data.document_id;
    lastDoc = data.document;

    renderResult(data);
    toast(`Done! Score ${data.score}/100 ✓`);

  } catch (e) {
    showLoader(false);
    toast(e.message, true);
  } finally {
    btn.disabled = false;
  }
}

/* ══ Loader ══════════════════════════════════════════════════════════════════ */
function showLoader(show) {
  document.getElementById('loaderOverlay').style.display = show ? 'flex' : 'none';
}

function animateSteps() {
  const orbs = ['orb1','orb2','orb3'];
  orbs.forEach(id => { document.getElementById(id).className = 'step-orb idle'; });
  document.getElementById('orb1').className = 'step-orb pending';
  let i = 0;
  const iv = setInterval(() => {
    document.getElementById(orbs[i]).className = 'step-orb done';
    i++;
    if (i < orbs.length) document.getElementById(orbs[i]).className = 'step-orb pending';
    else clearInterval(iv);
  }, 1800);
}

/* ══ Render Result — with animated score counter ════════════════════════════ */
function renderResult(data) {
  showLoader(false);

  const score = data.score ?? 0;
  const cls   = score >= 80 ? '' : score >= 50 ? 'mid' : 'low';

  // ── Score block with animated counter ──────────────────────────────────
  document.getElementById('scoreBlock').innerHTML = `
    <div class="score-block">
      <div class="score-circle">
        <span class="score-num" id="scoreNum">0</span>
        <span class="score-den">/100</span>
      </div>
      <div class="score-right">
        <div class="score-lbl">Quality Score</div>
        <div class="score-bar-track">
          <div class="score-bar-fill ${cls}" id="sfill" style="width:0%"></div>
        </div>
        <div style="margin-top:6px;font-size:.75rem;color:var(--n400)">
          ${score >= 80 ? '✓ Meets quality threshold' : score >= 50 ? '⚠ Below threshold — revision needed' : '✗ Low quality — major revision required'}
        </div>
      </div>
    </div>`;

  // Animate score counter from 0 → score
  let current = 0;
  const step  = Math.ceil(score / 40);
  const iv = setInterval(() => {
    current = Math.min(current + step, score);
    const el = document.getElementById('scoreNum');
    if (el) {
      el.textContent = current;
      el.style.color = current >= 80 ? 'var(--g600)' : current >= 50 ? 'var(--amber)' : 'var(--red)';
    }
    if (current >= score) clearInterval(iv);
  }, 30);

  // Animate bar after short delay
  setTimeout(() => {
    const bar = document.getElementById('sfill');
    if (bar) bar.style.width = `${score}%`;
  }, 100);

  // ── Meta tags ───────────────────────────────────────────────────────────
  const statusPill = {
    approved:           '<span class="pill pill-approved">✓ Approved</span>',
    needs_human_review: '<span class="pill pill-human">⏳ Needs Human Review</span>',
    rejected:           '<span class="pill pill-rejected">✗ Rejected</span>',
    processing:         '<span class="pill pill-revision">⟳ Processing</span>',
  }[data.status] ?? '';

  document.getElementById('metaTags').innerHTML = `
    <div class="meta-tags">
      ${statusPill}
      <span class="pill pill-outline">${(data.document_type||'general').replace(/_/g,' ').toUpperCase()}</span>
      <span class="pill pill-outline">${data.iteration_count} iteration${data.iteration_count !== 1 ? 's' : ''}</span>
      <span class="pill pill-outline" style="opacity:.6;font-size:.68rem">ID: ${data.document_id?.slice(0,8)}…</span>
    </div>`;

  // ── Feedback ────────────────────────────────────────────────────────────
  let fb = '';
  if (data.missing_sections?.length) {
    fb += `<div class="fb-block danger">
      <div class="fb-title">Missing Sections</div>
      <ul class="fb-list">${data.missing_sections.map(s=>`<li>${s}</li>`).join('')}</ul>
    </div>`;
  }
  if (data.feedback?.length) {
    fb += `<div class="fb-block warn">
      <div class="fb-title">Critic Feedback</div>
      <ul class="fb-list">${data.feedback.map(f=>`<li>${f}</li>`).join('')}</ul>
    </div>`;
  }
  document.getElementById('feedbackBlock').innerHTML = fb;

  // ── Document body ───────────────────────────────────────────────────────
  document.getElementById('docBody').textContent = data.document || '(empty)';

  // Show result + download buttons
  const panel = document.getElementById('resultPanel');
  panel.style.display = 'flex';

  // Update download buttons to use this document ID
  document.getElementById('downloadMdBtn').onclick  = () => downloadMarkdown();
  document.getElementById('downloadPdfBtn').onclick = () => downloadPDF(data.document_id);

  panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ══ Retrieve ════════════════════════════════════════════════════════════════ */
async function retrieveDocument() {
  const id = document.getElementById('retrieveId').value.trim();
  const el = document.getElementById('retrieveResult');
  if (!id) { toast('Enter a document ID.', true); return; }

  el.innerHTML = '<p style="color:var(--n400);font-size:.85rem;margin-top:12px">Fetching…</p>';

  try {
    const r    = await fetch(`${API}/document/${encodeURIComponent(id)}`);
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'Not found');

    const score = data.score ?? 0;
    const cls   = score >= 80 ? '' : score >= 50 ? 'mid' : 'low';

    el.innerHTML = `
      <div class="score-block" style="margin-top:16px">
        <div class="score-circle">
          <span class="score-num" style="color:${score>=80?'var(--g600)':score>=50?'var(--amber)':'var(--red)'}">${score}</span>
          <span class="score-den">/100</span>
        </div>
        <div class="score-right">
          <div class="score-lbl">Quality Score</div>
          <div class="score-bar-track">
            <div class="score-bar-fill ${cls}" style="width:${score}%"></div>
          </div>
        </div>
      </div>
      <div class="meta-tags" style="margin:12px 0">
        <span class="pill pill-outline">${(data.document_type||'').replace(/_/g,' ').toUpperCase()}</span>
        <span class="pill pill-outline">${data.iteration_count} iteration(s)</span>
      </div>
      <div style="display:flex;gap:8px;margin-bottom:10px">
        <button class="icon-btn" onclick="downloadPDF('${data.document_id}')">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
          Download PDF
        </button>
      </div>
      <div class="retrieve-doc">${esc(data.document)}</div>`;
  } catch (e) {
    el.innerHTML = `<div class="alert err">${e.message}</div>`;
  }
}

/* ══ Approve ═════════════════════════════════════════════════════════════════ */
async function submitApproval(approved) {
  const id    = document.getElementById('approveId').value.trim();
  const notes = document.getElementById('reviewerNotes').value.trim();
  const el    = document.getElementById('approveResult');
  if (!id) { toast('Enter a document ID.', true); return; }

  try {
    const r    = await fetch(`${API}/approve`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ document_id: id, approved, reviewer_notes: notes || null }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'Request failed');
    el.innerHTML = `<div class="alert ${approved ? 'ok' : 'err'}">${data.message}</div>`;
    toast(data.message);
  } catch (e) {
    el.innerHTML = `<div class="alert err">${e.message}</div>`;
  }
}

/* ══ Download Markdown ═══════════════════════════════════════════════════════ */
function downloadMarkdown() {
  if (!lastDoc) { toast('No document yet.', true); return; }
  const a = Object.assign(document.createElement('a'), {
    href: URL.createObjectURL(new Blob([lastDoc], { type: 'text/markdown' })),
    download: `document-${lastId?.slice(0,8) || 'output'}.md`,
  });
  a.click();
  URL.revokeObjectURL(a.href);
  toast('Markdown downloaded!');
}

/* ══ Download PDF (from backend) ═════════════════════════════════════════════ */
async function downloadPDF(docId) {
  const id = docId || lastId;
  if (!id) { toast('No document to download.', true); return; }

  toast('Generating PDF…');
  try {
    const r = await fetch(`${API}/document/${id}/download`);
    if (!r.ok) {
      const err = await r.json();
      throw new Error(err.detail || 'PDF failed');
    }
    const blob = await r.blob();
    const a = Object.assign(document.createElement('a'), {
      href: URL.createObjectURL(blob),
      download: `document-${id.slice(0,8)}.pdf`,
    });
    a.click();
    URL.revokeObjectURL(a.href);
    toast('PDF downloaded!');
  } catch (e) {
    toast(e.message, true);
  }
}

/* ══ Copy ════════════════════════════════════════════════════════════════════ */
function copyDocument() {
  if (!lastDoc) return;
  navigator.clipboard.writeText(lastDoc).then(() => toast('Copied to clipboard!'));
}

/* ══ Toast ═══════════════════════════════════════════════════════════════════ */
let _tt;
function toast(msg, err = false) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className   = `toast show${err ? ' err' : ''}`;
  clearTimeout(_tt);
  _tt = setTimeout(() => el.classList.remove('show'), 3400);
}

/* ══ Helpers ═════════════════════════════════════════════════════════════════ */
function esc(s) {
  return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
