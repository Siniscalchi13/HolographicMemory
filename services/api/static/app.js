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

    // Append wave visualization with 2D/3D toggle
    try {
      let wave = null; let real = false
      try {
        console.log(`[Wave] Fetching data for docId: ${docId}`)
        const wreal = await fetch(`/wave/${docId}/real`, { headers: apiHeaders() })
        if (wreal.ok) {
          wave = await wreal.json();
          real = true;
          console.log(`[Wave] Got real data:`, wave.magnitudes?.length, 'magnitudes,', wave.phases?.length, 'phases')
        } else {
          console.warn(`[Wave] Real endpoint failed: ${wreal.status}`)
        }
      } catch(e) {
        console.warn(`[Wave] Real endpoint error:`, e)
      }
      if (!wave){
        console.log(`[Wave] Falling back to DFT endpoint`)
        const wres = await fetch(`/wave/${docId}`, { headers: apiHeaders() })
        if (wres.ok) {
          wave = await wres.json();
          console.log(`[Wave] Got DFT data:`, wave.magnitudes?.length, 'magnitudes')
        } else {
          console.warn(`[Wave] DFT endpoint failed: ${wres.status}`)
        }
      }
      if (!wave) throw new Error('No wave data')

      // Create visualization controls
      const controlsDiv = document.createElement('div')
      controlsDiv.style.cssText = 'margin-top: 12px; margin-bottom: 8px; display: flex; gap: 8px; align-items: center;'
      controlsDiv.innerHTML = `
        <div class="muted" style="flex: 1;">${real? '🌌 Authentic holographic wave' : '📊 DFT approximation'} spectrum</div>
        <button id="toggle-2d-3d" style="padding: 4px 8px; background: #6366f1; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px;">Show 3D</button>
        <button id="toggle-animation" style="padding: 4px 8px; background: #10b981; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px;">Animate</button>
      `
      document.getElementById('preview').appendChild(controlsDiv)

      // Create 2D canvas visualization (default)
      const canvas = document.createElement('canvas')
      canvas.id = 'wave-canvas-2d'
      canvas.width = 420; canvas.height = 120
      canvas.style.cssText = 'border: 1px solid #e2e8f0; border-radius: 4px;'

      const ctx = canvas.getContext('2d')
      ctx.fillStyle = '#f8fafc'; ctx.fillRect(0,0,canvas.width,canvas.height)
      const W = canvas.width, H = canvas.height
      const mags = wave.magnitudes || []
      const count = Math.max(1, mags.length)
      const bar = Math.max(1, Math.floor(W / count))
      const phases = wave.phases || new Array(count).fill(0)
      mags.forEach((m,i)=>{
        const h = Math.max(1, Math.floor(m * (H-10)))
        // simple phase->hue mapping
        const hue = Math.floor(((phases[i]||0)/(Math.PI*2) + 1) * 180) % 360
        ctx.fillStyle = `hsl(${hue},70%,55%)`
        ctx.fillRect(i*bar, H-h, Math.max(1, bar-1), h)
      })

      // Create 3D container (hidden by default)
      const container3D = document.createElement('div')
      container3D.id = 'wave-container-3d'
      container3D.style.cssText = 'width: 420px; height: 300px; border: 1px solid #e2e8f0; border-radius: 4px; display: none;'

      // Add both to preview
      document.getElementById('preview').appendChild(canvas)
      document.getElementById('preview').appendChild(container3D)

      // Wire up toggle buttons
      let currentViewer = null
      let is3D = false
      let isAnimating = false

      document.getElementById('toggle-2d-3d').onclick = async () => {
        const btn = document.getElementById('toggle-2d-3d')
        is3D = !is3D

        if (is3D) {
          // Switch to 3D
          canvas.style.display = 'none'
          container3D.style.display = 'block'
          btn.textContent = 'Show 2D'

          // Load 3D viewer if not already loaded
          if (!currentViewer) {
            try {
              const { WaveViewer3D } = await import('./wave-viewer-3d.js')
              currentViewer = new WaveViewer3D()
              await currentViewer.initScene(container3D)
              currentViewer.updateData(wave.magnitudes || [], wave.phases || [])
            } catch (e) {
              console.error('[3D] Failed to load viewer:', e)
              container3D.innerHTML = '<div style="padding: 20px; text-align: center; color: #ef4444;">3D visualization not supported on this device</div>'
            }
          }
        } else {
          // Switch to 2D
          canvas.style.display = 'block'
          container3D.style.display = 'none'
          btn.textContent = 'Show 3D'

          // Stop 3D animation if running
          if (currentViewer && isAnimating) {
            currentViewer.toggleAnimation()
          }
        }
      }

      document.getElementById('toggle-animation').onclick = () => {
        const btn = document.getElementById('toggle-animation')
        isAnimating = !isAnimating
        btn.textContent = isAnimating ? 'Stop' : 'Animate'
        btn.style.background = isAnimating ? '#ef4444' : '#10b981'

        if (currentViewer && is3D) {
          currentViewer.toggleAnimation()
        }
      }

    } catch (e) {
      console.warn('[web] wave viz failed', e)
      // Add fallback message
      const fallback = document.createElement('div')
      fallback.innerHTML = '<div class="muted" style="margin-top: 12px;">Wave visualization unavailable</div>'
      document.getElementById('preview').appendChild(fallback)
    }
  } catch (error) {
    console.error('[web] preview error', error)
    $('#preview').innerHTML = `<div class="muted">Preview error: ${error.message}</div>`
  }
}

function renderMemoryField(fieldData) {
  const canvas = document.getElementById('memory-field-canvas')
  if (!canvas) return

  const ctx = canvas.getContext('2d')
  const width = canvas.width = canvas.offsetWidth || 280
  const height = canvas.height = 120

  // Clear canvas with dark background
  ctx.fillStyle = '#0f0f23'
  ctx.fillRect(0, 0, width, height)

  const amplitudes = fieldData.magnitudes || []
  const phases = fieldData.phases || []

  if (amplitudes.length === 0) {
    ctx.fillStyle = '#6b7280'
    ctx.font = '12px monospace'
    ctx.textAlign = 'center'
    ctx.fillText('No field data', width/2, height/2)
    return
  }

  // Create interference pattern visualization
  const step = Math.max(1, Math.floor(amplitudes.length / width))
  const barWidth = Math.max(1, Math.floor(width / (amplitudes.length / step)))

  for (let i = 0; i < amplitudes.length; i += step) {
    const amp = amplitudes[i] || 0
    const phase = phases[i] || 0

    // Convert phase to hue (0-360)
    const hue = Math.floor(((phase + Math.PI) / (2 * Math.PI)) * 360) % 360

    // Height based on amplitude
    const barHeight = Math.max(1, Math.floor(amp * height * 0.8))

    // Create gradient for interference effect
    const gradient = ctx.createLinearGradient(0, height - barHeight, 0, height)
    gradient.addColorStop(0, `hsl(${hue}, 70%, 60%)`)
    gradient.addColorStop(0.5, `hsl(${(hue + 60) % 360}, 50%, 40%)`)
    gradient.addColorStop(1, `hsl(${(hue + 120) % 360}, 30%, 20%)`)

    ctx.fillStyle = gradient
    ctx.fillRect(i * barWidth / step, height - barHeight, barWidth, barHeight)

    // Add interference lines
    ctx.strokeStyle = `hsl(${hue}, 80%, 70%)`
    ctx.lineWidth = 0.5
    ctx.beginPath()
    ctx.moveTo(i * barWidth / step, height - barHeight)
    ctx.lineTo(i * barWidth / step + barWidth, height - barHeight + Math.sin(phase) * 10)
    ctx.stroke()
  }

  // Add subtle animation
  let animationOffset = 0
  const animateField = () => {
    animationOffset += 0.02
    ctx.globalAlpha = 0.3

    // Add moving interference waves
    for (let x = 0; x < width; x += 20) {
      const wave = Math.sin((x * 0.1) + animationOffset) * 5
      ctx.strokeStyle = '#4c1d95'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(x, height/2 + wave)
      ctx.lineTo(x + 10, height/2 + Math.sin(((x + 10) * 0.1) + animationOffset) * 5)
      ctx.stroke()
    }

    ctx.globalAlpha = 1
    requestAnimationFrame(animateField)
  }
  animateField()
}

function showError(msg){
  const b = $('#banner')
  b.textContent = msg
  b.style.display = 'block'
  setTimeout(()=>{ b.style.display='none' }, 6000)
}

function downloadFile(docId, filename){
  console.log('[web] downloading', { docId, filename })
  // Create a proper download with API headers
  fetch(`/download/${docId}`, { headers: apiHeaders() })
    .then(response => {
      if (!response.ok) throw new Error(`Download failed: ${response.statusText}`)
      return response.blob().then(blob => ({ blob, response }))
    })
    .then(({ blob, response }) => {
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url

      // Use filename from Content-Disposition header if available, otherwise use provided filename
      const contentDisposition = response.headers.get('content-disposition')
      if (contentDisposition && contentDisposition.includes('filename=')) {
        const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/)
        if (filenameMatch && filenameMatch[1]) {
          a.download = filenameMatch[1].replace(/['"]/g, '')
        } else {
          a.download = filename
        }
      } else {
        a.download = filename
      }

      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    })
    .catch(error => {
      console.error('[web] download error', error)
      showError(`Download failed: ${error.message}`)
    })
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

    // Add collective holographic memory field visualization
    const memoryFieldDiv = document.createElement('div')
    memoryFieldDiv.id = 'memory-field-viz'
    memoryFieldDiv.style.cssText = 'margin-top: 20px; padding: 12px; background: linear-gradient(135deg, #1e1b4b, #312e81); border-radius: 8px; color: white;'
    memoryFieldDiv.innerHTML = `
      <h4 style="margin: 0 0 8px 0; font-size: 14px;">🌌 Holographic Memory Field</h4>
      <canvas id="memory-field-canvas" style="width: 100%; height: 120px; background: #0f0f23; border-radius: 4px; border: 1px solid #4c1d95;"></canvas>
      <div style="margin-top: 8px; font-size: 11px; opacity: 0.8;">
        <span id="field-status">Loading field data...</span>
      </div>
    `
    document.getElementById('tree').appendChild(memoryFieldDiv)

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

    // Load collective memory field data AFTER CURRENT_ROWS is populated
    try {
      const docIds = window.CURRENT_ROWS?.map(r => r.doc_id) || []
      if (docIds.length > 0) {
        console.log('[web] Loading collective field for', docIds.length, 'files')
        const collectiveResponse = await fetch('/wave/collective/real', {
          method: 'POST',
          headers: Object.assign({ 'Content-Type': 'application/json' }, apiHeaders()),
          body: JSON.stringify({ doc_ids: docIds.slice(0, 10) }) // Limit to first 10 for performance
        })

        if (collectiveResponse.ok) {
          const fieldData = await collectiveResponse.json()
          console.log('[web] Collective field data loaded:', fieldData.amplitudes?.length, 'amplitudes')
          renderMemoryField(fieldData)
          document.getElementById('field-status').textContent = `${docIds.length} patterns superposed`
        } else {
          console.warn('[web] Collective field response not ok:', collectiveResponse.status)
          document.getElementById('field-status').textContent = 'Field data unavailable'
        }
      } else {
        console.warn('[web] No doc_ids available for collective field')
        document.getElementById('field-status').textContent = 'No files to visualize'
      }
    } catch (e) {
      console.warn('[web] collective field failed:', e)
      document.getElementById('field-status').textContent = 'Field visualization failed'
    }
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

// Expose functions for inline per-row actions (module scope isn't global)
// eslint-disable-next-line no-undef
window.downloadFile = downloadFile
// eslint-disable-next-line no-undef
window.deleteOne = deleteOne
