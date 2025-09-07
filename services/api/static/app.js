const $ = (s) => document.querySelector(s)
const fmt = (n) => new Intl.NumberFormat().format(n)

function apiHeaders(){
  const k = localStorage.getItem('x_api_key')
  return k ? { 'X-API-Key': k } : {}
}

async function api(path, opts={}){
  const headers = Object.assign({}, opts.headers||{}, apiHeaders())
  const url = path.startsWith('http') ? path : path
  console.log('[web] fetch', { method: (opts.method||'GET'), url, headers, hasBody: !!opts.body })
  const res = await fetch(url, Object.assign({}, opts, { headers }))
  const bodyTxt = await res.clone().text().catch(()=>"")
  console.log('[web] response', { status: res.status, ok: res.ok, body: bodyTxt.slice(0, 400) })
  if (!res.ok) throw new Error(bodyTxt || res.statusText)
  try { return JSON.parse(bodyTxt) } catch { return await res.json() }
}

function ext(path){ const i = path.lastIndexOf('.'); return i>=0? path.slice(i+1).toLowerCase(): '' }

function renderTree(node, base=''){
  const full = base ? `${base}/${node.name}` : node.name
  const children = (node.dirs||[]).map(d=>renderTree(d, full)).join('')
  return `<div class="tree-node" data-path="${node.path}">📁 ${node.name}${children?'<div style="margin-left:12px">'+children+'</div>':''}</div>`
}

function renderRows(rows){
  const grid = $('#gridview')
  const list = $('#listview')
  const sorted = [...rows]
  if (list.style.display !== 'table') {
    list.style.display='table'
  }
  grid.style.display='none'
  $('#rows').innerHTML = sorted.map(r=>{
    const name = r.path.split('/').pop()
    const originalSize = r.size || 0
    const holoSize = r.holo_size || 0
    const ratio = holoSize > 0 ? (originalSize / holoSize).toFixed(1) : '—'
    return `<tr data-path="${r.path}" data-doc-id="${r.doc_id}">
      <td><input type="checkbox" class="file-checkbox" data-doc-id="${r.doc_id}" data-filename="${name}"></td>
      <td>📄</td>
      <td>${name}</td>
      <td>${ext(name) || '—'}</td>
      <td>${fmt(originalSize)} bytes</td>
      <td>${fmt(holoSize)} bytes</td>
      <td>${ratio}x</td>
      <td>${r.mtime? new Date(r.mtime*1000).toLocaleString(): '—'}</td>
      <td>
        <button onclick="downloadFile('${r.doc_id}', '${name}')" class="btn small">⬇️</button>
        <button onclick="deleteOne('${r.doc_id}'); return false;" class="btn small btn-danger">🗑️</button>
      </td>
    </tr>`
  }).join('')
  bindRowEvents()
  $('#statusBar').textContent = `${sorted.length} items`
  bindSelection()
}

function bindRowEvents(){
  document.querySelectorAll('#rows tr').forEach(tr=>{
    tr.addEventListener('click', async ()=>{
      const path = tr.getAttribute('data-path')
      const docId = tr.getAttribute('data-doc-id')
      document.querySelectorAll('#rows tr').forEach(x=>x.classList.remove('sel'))
      tr.classList.add('sel')
      
      console.log('[web] previewing', { path, docId })
      await showPreview(path, docId)
    })
  })
}

async function showPreview(path, docId) {
  const filename = path.split('/').pop()
  const fileExt = ext(filename).toLowerCase()
  
  try {
    // Try to download the actual file content for preview
    const res = await fetch(`/download/${docId}`, { headers: apiHeaders() })
    if (!res.ok) {
      $('#preview').innerHTML = `<div class="muted">Cannot preview: ${res.statusText}</div>`
      return
    }
    
    if (fileExt === 'txt' || fileExt === 'md' || fileExt === 'py' || fileExt === 'js' || fileExt === 'json' || fileExt === 'html' || fileExt === 'css') {
      // Text files - show content
      const text = await res.text()
      $('#preview').innerHTML = `<pre style="white-space: pre-wrap; font-family: monospace; font-size: 12px; max-height: 400px; overflow-y: auto;">${text.slice(0, 5000)}</pre>`
    } else if (fileExt === 'png' || fileExt === 'jpg' || fileExt === 'jpeg' || fileExt === 'gif' || fileExt === 'webp') {
      // Images - show full size
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      $('#preview').innerHTML = `<img src="${url}" style="max-width: 100%; max-height: 400px; object-fit: contain;"/>`
    } else if (fileExt === 'pdf') {
      // PDFs - show download link and try thumbnail
      const thumbRes = await fetch(`/thumb?path=${encodeURIComponent(path)}&w=400`, { headers: apiHeaders() })
      if (thumbRes.ok) {
        const thumbBlob = await thumbRes.blob()
        const thumbUrl = URL.createObjectURL(thumbBlob)
        $('#preview').innerHTML = `
          <div>
            <img src="${thumbUrl}" style="max-width: 100%; max-height: 300px; object-fit: contain; border: 1px solid #ddd;"/>
            <p class="muted">PDF preview (first page)</p>
          </div>`
      } else {
        $('#preview').innerHTML = `<div class="muted">PDF file - click download to view</div>`
      }
    } else {
      // Other files - show file info
      const blob = await res.blob()
      $('#preview').innerHTML = `
        <div class="muted">
          <p><strong>${filename}</strong></p>
          <p>Size: ${fmt(blob.size)} bytes</p>
          <p>Type: ${fileExt || 'unknown'}</p>
          <p>Click download button to save file</p>
        </div>`
    }
  } catch (error) {
    console.error('[web] preview error', error)
    $('#preview').innerHTML = `<div class="muted">Preview error: ${error.message}</div>`
  }
}

function showError(msg){
  const b = $('#banner')
  b.textContent = msg
  b.style.display = 'block'
  setTimeout(()=>{ b.style.display='none' }, 6000)
}

function downloadFile(docId, filename){
  console.log('[web] downloading', { docId, filename })
  const url = `/download/${docId}`
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
}

function bindSelection(){
  const toolbar = $('#toolbar')
  const cbAll = $('#select-all')
  if (cbAll) cbAll.onchange = toggleSelectAll
  document.querySelectorAll('.file-checkbox').forEach(cb=>{
    cb.addEventListener('change', updateSelectionCount)
  })
  updateSelectionCount()
}

function toggleSelectAll(){
  const all = $('#select-all').checked
  document.querySelectorAll('.file-checkbox').forEach(cb=>{ cb.checked = all })
  updateSelectionCount()
}

function getSelected(){
  return Array.from(document.querySelectorAll('.file-checkbox:checked')).map(cb=>({ doc_id: cb.dataset.docId, filename: cb.dataset.filename }))
}

async function deleteOne(docId){
  if (!confirm('Delete this file?')) return
  const r = await fetch(`/files/${docId}`, { method: 'DELETE', headers: apiHeaders() })
  if (!r.ok){ showError('Delete failed'); return }
  // remove from current rows and re-render
  window.CURRENT_ROWS = window.CURRENT_ROWS.filter(r=> r.doc_id !== docId)
  renderRows(window.CURRENT_ROWS)
}

async function deleteSelected(){
  const sel = getSelected()
  if (sel.length === 0) return
  if (!confirm(`Delete ${sel.length} files?`)) return
  for (const s of sel){
    try { await fetch(`/files/${s.doc_id}`, { method: 'DELETE', headers: apiHeaders() }) } catch {}
  }
  // refresh list
  await refresh()
}

async function downloadSelected(){
  const sel = getSelected()
  if (sel.length === 0) return
  const body = JSON.stringify({ doc_ids: sel.map(s=>s.doc_id) })
  const r = await fetch('/zip', { method: 'POST', headers: Object.assign({ 'Content-Type': 'application/json' }, apiHeaders()), body })
  if (!r.ok){ showError('Zip download failed'); return }
  const blob = await r.blob(); const url = URL.createObjectURL(blob)
  const a = document.createElement('a'); a.href = url; a.download = 'selected.zip'; document.body.appendChild(a); a.click(); document.body.removeChild(a)
}

function updateSelectionCount(){
  const sel = getSelected(); const cnt = $('#selection-count'); const bar = $('#toolbar')
  if (cnt) cnt.textContent = `${sel.length} files selected`
  if (bar) bar.style.display = sel.length > 0 ? 'flex' : 'none'
}

// wire toolbar buttons
document.addEventListener('DOMContentLoaded', ()=>{
  const bDel = document.getElementById('btnDeleteSel'); if (bDel) bDel.onclick = deleteSelected
  const bZip = document.getElementById('btnZipSel'); if (bZip) bZip.onclick = downloadSelected
})

async function refresh(){
  try {
    console.log('[web] refresh')
    const stats = await api('/stats')
    $('#status').textContent = 'Connected'
    $('#stats').innerHTML = `
      <div class="row">
        <div class="pill">dim ${stats.dimension ?? '—'}</div>
        <div class="pill">files ${stats.files_indexed ?? 0}</div>
        <div class="pill">bytes ${fmt(stats.original_total_bytes ?? 0)}</div>
        <div class="pill">compression x${stats.compression_x ?? '—'}</div>
      </div>`

    const tree = await api('/tree')
    $('#tree').innerHTML = renderTree(tree)
    document.querySelectorAll('.tree-node').forEach(el=>{
      el.addEventListener('click', ()=>{
        const p = el.getAttribute('data-path')
        $('#cwd').textContent = p
        const rows = window.CURRENT_ROWS.filter(r => r.path.startsWith(p))
        renderRows(rows)
      })
    })
    const list = await api('/list')
    window.CURRENT_ROWS = (list.results||[])
    renderRows(window.CURRENT_ROWS)
  } catch (e) {
    console.error('[web] refresh error', e)
    $('#status').textContent = 'API not running'
  }
}

function setupDnD(){
  const drop = $('#drop')
  drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.style.background = '#eef2ff' })
  drop.addEventListener('dragleave', () => { drop.style.background = '#fff' })
  drop.addEventListener('drop', async (e) => {
    e.preventDefault(); drop.style.background = '#fff'
    const files = e.dataTransfer.files
    try {
      for (const f of files) await uploadFile(f)
      await refresh()
    } catch (err) {
      console.error('[web] upload error', err)
      showError(err?.message || 'Upload failed')
    }
  })
  $('#file').addEventListener('change', async (e) => {
    const files = e.target.files
    try {
      for (const f of files) await uploadFile(f)
      await refresh()
    } catch (err) {
      console.error('[web] upload error', err)
      showError(err?.message || 'Upload failed')
    }
  })
}

async function uploadFile(file){
  console.log('[web] uploading', { name: file.name, size: file.size, type: file.type })
  const fd = new FormData()
  fd.append('file', file, file.name)
  const entries = []
  for (const [k, v] of fd.entries()) { entries.push([k, (v instanceof File)? { name: v.name, size: v.size, type: v.type } : String(v)]) }
  console.log('[web] FormData', entries)
  const res = await fetch('/store', { method: 'POST', body: fd, headers: apiHeaders() })
  const txt = await res.clone().text().catch(()=>"")
  console.log('[web] upload response', res.status, txt.slice(0,400))
  if (!res.ok) throw new Error(txt || res.statusText)
  return JSON.parse(txt)
}

function setupSearch(){
  const q = $('#q')
  q.addEventListener('input', async () => {
    const v = q.value.trim()
    if (!v) return refresh()
    const res = await api(`/search?q=${encodeURIComponent(v)}`)
    renderRows(res.results || [])
  })
}

window.addEventListener('DOMContentLoaded', async () => {
  // settings: api key
  $('#apiKey').value = localStorage.getItem('x_api_key') || ''
  $('#saveKey').addEventListener('click', ()=>{
    localStorage.setItem('x_api_key', $('#apiKey').value.trim())
    refresh()
  })
  setupDnD()
  setupSearch()
  // wait for API readiness
  let ok = false
  for (let i=0;i<20;i++){
    try { const r = await fetch('/healthz'); if (r.ok) { ok = true; break } } catch {}
    await new Promise(r=>setTimeout(r, 500))
  }
  const st = $('#status')
  st.textContent = ok? 'Connected' : 'Waiting for API…'
  await refresh()
})
