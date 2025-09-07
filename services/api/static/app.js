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
    return `<tr data-path="${r.path}">
      <td>📄</td>
      <td>${name}</td>
      <td>${ext(name) || '—'}</td>
      <td>${fmt(r.size||0)}</td>
      <td>${r.mtime? new Date(r.mtime*1000).toLocaleString(): '—'}</td>
    </tr>`
  }).join('')
  bindRowEvents()
  $('#statusBar').textContent = `${sorted.length} items`
}

function bindRowEvents(){
  document.querySelectorAll('#rows tr').forEach(tr=>{
    tr.addEventListener('click', async ()=>{
      const path = tr.getAttribute('data-path')
      document.querySelectorAll('#rows tr').forEach(x=>x.classList.remove('sel'))
      tr.classList.add('sel')
      const res = await fetch(`/thumb?path=${encodeURIComponent(path)}&w=320`, { headers: apiHeaders() })
      if (res.ok){
        const blob = await res.blob()
        const url = URL.createObjectURL(blob)
        $('#preview').innerHTML = `<img src="${url}" style="max-width:100%"/>`
      } else {
        $('#preview').innerHTML = `<div class="muted">No preview available</div>`
      }
    })
  })
}

function showError(msg){
  const b = $('#banner')
  b.textContent = msg
  b.style.display = 'block'
  setTimeout(()=>{ b.style.display='none' }, 6000)
}

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

