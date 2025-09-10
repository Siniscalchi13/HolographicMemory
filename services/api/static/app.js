console.log('[DEBUG] JavaScript loading...')

const $ = (s) => document.querySelector(s)
const fmt = (n) => new Intl.NumberFormat().format(n)

// Pagination state
let currentPage = 1
let perPage = 50
let totalPages = 1
let totalFiles = 0

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

// Load files with pagination
async function loadFiles(page = 1, itemsPerPage = 50) {
  try {
    console.log(`[DEBUG] Loading files: page=${page}, per_page=${itemsPerPage}`)
    const response = await api(`/list?page=${page}&per_page=${itemsPerPage}`)
    window.CURRENT_ROWS = response.results || []
    currentPage = response.page || 1
    perPage = response.per_page || 50
    totalPages = response.pages || 1
    totalFiles = response.total || 0
    
    console.log(`[DEBUG] Loaded ${window.CURRENT_ROWS.length} files, total=${totalFiles}, pages=${totalPages}`)
    
    renderRows(window.CURRENT_ROWS)
    renderPagination()
    updateStatusBar()
  } catch (error) {
    console.error('Failed to load files:', error)
    $('#status').textContent = 'Failed to load files'
  }
}

function renderPagination() {
  let pagination = $('#pagination')
  if (!pagination) {
    console.warn('Pagination container missing; creating dynamically')
    pagination = createPaginationElement()
  }
  // Ensure default styling
  try { pagination.classList.add('pagination') } catch {}
  
  console.log(`[DEBUG] Rendering pagination: page=${currentPage}/${totalPages}, total=${totalFiles}`)
  
  // Generate page numbers (show up to 5 pages around current page)
  let pageNumbers = []
  const startPage = Math.max(1, currentPage - 2)
  const endPage = Math.min(totalPages, currentPage + 2)
  
  for (let i = startPage; i <= endPage; i++) {
    pageNumbers.push(i)
  }
  
  let html = `<div class="pagination-controls">
    <div class="pagination-info">
      Showing ${((currentPage - 1) * perPage) + 1}-${Math.min(currentPage * perPage, totalFiles)} of ${fmt(totalFiles)} files
    </div>
    <div class="pagination-buttons">
      <button class="btn btn-sm" ${currentPage <= 1 ? 'disabled' : ''} onclick="loadFiles(1, ${perPage})">⏮️ First</button>
      <button class="btn btn-sm" ${currentPage <= 1 ? 'disabled' : ''} onclick="loadFiles(${currentPage - 1}, ${perPage})">← Previous</button>
      
      ${pageNumbers.map(page => 
        `<button class="btn btn-sm ${page === currentPage ? 'btn-primary' : ''}" onclick="loadFiles(${page}, ${perPage})">${page}</button>`
      ).join('')}
      
      <button class="btn btn-sm" ${currentPage >= totalPages ? 'disabled' : ''} onclick="loadFiles(${currentPage + 1}, ${perPage})">Next →</button>
      <button class="btn btn-sm" ${currentPage >= totalPages ? 'disabled' : ''} onclick="loadFiles(${totalPages}, ${perPage})">Last ⏭️</button>
    </div>
    <div class="per-page-controls">
      <label>Per page:</label>
      <select onchange="changePerPage(this.value)">
        <option value="25" ${perPage === 25 ? 'selected' : ''}>25</option>
        <option value="50" ${perPage === 50 ? 'selected' : ''}>50</option>
        <option value="100" ${perPage === 100 ? 'selected' : ''}>100</option>
        <option value="250" ${perPage === 250 ? 'selected' : ''}>250</option>
      </select>
    </div>
  </div>`
  
  pagination.innerHTML = html
}

function createPaginationElement() {
  const pagination = document.createElement('div')
  pagination.id = 'pagination'
  pagination.className = 'pagination'
  $('#listview').parentNode.insertBefore(pagination, $('#listview'))
  return pagination
}

function changePerPage(newPerPage) {
  perPage = parseInt(newPerPage)
  loadFiles(1, perPage) // Reset to page 1
}

function updateStatusBar() {
  $('#statusBar').textContent = `${fmt(totalFiles)} items (Page ${currentPage}/${totalPages})`
}

function updatePaginationForFiltered(filteredRows) {
  const pagination = $('#pagination')
  if (!pagination) return
  
  const filteredCount = filteredRows.length
  const filteredPages = Math.ceil(filteredCount / perPage)
  
  let html = `<div class="pagination-controls">
    <div class="pagination-info">
      Showing ${filteredCount} filtered files (${fmt(totalFiles)} total)
    </div>
    <div class="pagination-buttons">
      <button class="btn btn-sm" onclick="loadFiles(1, ${perPage})">🔄 Show All Files</button>
    </div>
  </div>`
  
  pagination.innerHTML = html
}

function toAbsPath(path, rootAbs) {
  if (!path) return ''
  if (path.startsWith('/')) return path
  const r = (rootAbs || '').replace(/\/$/, '')
  return r ? `${r}/${path}` : path
}

function relFromRoot(absPath, rootAbs) {
  const r = (rootAbs || '').replace(/\/$/, '')
  const a = (absPath || '').replace(/\/$/, '')
  if (r && a.startsWith(r)) {
    const rel = a.slice(r.length)
    return rel.startsWith('/') ? rel.slice(1) : rel
  }
  return absPath || ''
}

function rowMatchesFolder(rowPath, folderAbs, rootAbs) {
  const rp = toAbsPath(rowPath, rootAbs).replace(/\/$/, '')
  const fp = (folderAbs || '').replace(/\/$/, '')
  return rp === fp || rp.startsWith(fp + '/')
}

function setupFolderClickHandlers() {
  console.log('[DEBUG] Setting up folder click handlers...')
  const rootAbs = window.ROOT_PATH || ''
  document.querySelectorAll('.tree-node').forEach(el=>{
    el.addEventListener('click', async ()=>{
      const folderAbs = el.getAttribute('data-path') || ''
      const folderRel = relFromRoot(folderAbs, rootAbs)
      console.log(`[DEBUG] Folder clicked: abs=${folderAbs} rel=${folderRel}`)

      // Ensure we have a complete set of rows to filter (fetch all if needed)
      try {
        if (Array.isArray(window.CURRENT_ROWS) && window.CURRENT_ROWS.length < (totalFiles || 0)) {
          console.log('[DEBUG] Fetching all rows for folder filter...')
          const all = await api(`/list?page=1&per_page=${Math.max(totalFiles||0, 5000)}`)
          window.CURRENT_ROWS = all.results || []
        }
      } catch (e) { console.warn('[DEBUG] Fetch-all failed; filtering current page only', e) }

      $('#cwd').textContent = folderRel || '/'

      const rows = (window.CURRENT_ROWS || []).filter(r => rowMatchesFolder(r.path || '', folderAbs, rootAbs))
      console.log(`[DEBUG] Filtered ${rows.length} files for folder: ${folderRel}`)

      renderRows(rows)
      updatePaginationForFiltered(rows)
    })
  })
  console.log(`[DEBUG] Set up ${document.querySelectorAll('.tree-node').length} folder click handlers`)
}

// 3D Holographic Memory Field Visualization
let scene, camera, renderer, controls
let holographicField = null

function init3DVisualization() {
  const container = $('#preview')
  if (!container) return
  
  // Create Three.js scene
  scene = new THREE.Scene()
  scene.background = new THREE.Color(0x0a0a0a)
  
  // Camera
  camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000)
  camera.position.set(5, 5, 5)
  
  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setSize(container.clientWidth, container.clientHeight)
  renderer.shadowMap.enabled = true
  renderer.shadowMap.type = THREE.PCFSoftShadowMap
  
  // Clear existing content
  container.innerHTML = ''
  container.appendChild(renderer.domElement)
  
  // Add lighting
  const ambientLight = new THREE.AmbientLight(0x404040, 0.6)
  scene.add(ambientLight)
  
  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8)
  directionalLight.position.set(10, 10, 5)
  directionalLight.castShadow = true
  scene.add(directionalLight)
  
  // Add controls for rotation
  controls = new THREE.OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true
  controls.dampingFactor = 0.05
  
  // Create holographic field visualization
  createHolographicField()
  
  // Start animation loop
  animate()
}

function createHolographicField() {
  if (holographicField) {
    scene.remove(holographicField)
  }
  
  // Create a 3D grid representing the holographic memory field
  const geometry = new THREE.BufferGeometry()
  const vertices = []
  const colors = []
  
  // Create a 32x32x32 grid (representing the 64³ grid)
  const gridSize = 32
  const spacing = 0.1
  
  for (let x = 0; x < gridSize; x++) {
    for (let y = 0; y < gridSize; y++) {
      for (let z = 0; z < gridSize; z++) {
        // Add some randomness to make it look like a holographic field
        const noise = Math.sin(x * 0.1) * Math.cos(y * 0.1) * Math.sin(z * 0.1)
        const intensity = Math.abs(noise) * 0.5 + 0.5
        
        vertices.push(
          (x - gridSize/2) * spacing,
          (y - gridSize/2) * spacing,
          (z - gridSize/2) * spacing
        )
        
        // Color based on intensity (blue to purple to pink)
        colors.push(
          intensity * 0.2,      // R
          intensity * 0.4,      // G
          intensity * 0.8 + 0.2 // B
        )
      }
    }
  }
  
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3))
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))
  
  const material = new THREE.PointsMaterial({
    size: 0.02,
    vertexColors: true,
    transparent: true,
    opacity: 0.8
  })
  
  holographicField = new THREE.Points(geometry, material)
  scene.add(holographicField)
}

function animate() {
  requestAnimationFrame(animate)
  
  if (controls) {
    controls.update()
  }
  
  // Rotate the holographic field slowly
  if (holographicField) {
    holographicField.rotation.y += 0.005
    holographicField.rotation.x += 0.002
  }
  
  if (renderer && scene && camera) {
    renderer.render(scene, camera)
  }
}

// Initialize 3D visualization when page loads (with error handling)
window.addEventListener('load', () => {
  console.log('[DEBUG] Page loaded, skipping 3D visualization for now')
  // Temporarily disable 3D visualization to debug pagination
  // init3DVisualization()
})

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
    const pathName = (r.path || '').split('/').pop() || ''
    const displayName = r.original_filename || pathName.replace(/\.hwp$/i, '') || pathName
    const originalSize = Number(r.size || 0)
    const holoSize = Number(r.holo_size || 0)
    const fmtMagic = r.format || ''
    const recoverable = !!r.recoverable
    // Compute readable ratio; mark expansion and header-only cases
    let ratioTxt = '—'
    if (fmtMagic === 'H4M1' && !recoverable) {
      ratioTxt = 'header-only'
    } else if (holoSize > 0) {
      const rn = originalSize / holoSize
      ratioTxt = rn >= 1 ? `${rn.toFixed(1)}x` : `${rn.toFixed(1)}x (expansion)`
    }
    const typeStr = (()=>{
      const e = ext(displayName) || '—'
      if (!fmtMagic) return e
      if (fmtMagic === 'H4M1' && !recoverable) return `${e} / H4M1`
      if (fmtMagic === 'v3json') return `${e} / v3json`
      return `${e} / ${fmtMagic}`
    })()
    return `<tr data-path="${r.path}" data-doc-id="${r.doc_id}">
      <td><input type="checkbox" class="file-checkbox" data-doc-id="${r.doc_id}" data-filename="${displayName}"></td>
      <td>📄</td>
      <td>${displayName}</td>
      <td>${typeStr}</td>
      <td>${fmt(originalSize)} bytes</td>
      <td>${fmt(holoSize)} bytes</td>
      <td>${ratioTxt}</td>
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
    window.ROOT_PATH = tree?.path || ''
    $('#tree').innerHTML = renderTree(tree)
    $('#cwd').textContent = 'All Files'

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

    // Load files with pagination first
    console.log('[DEBUG] About to load files...')
    await loadFiles(1, 50)
    
    // Set up folder click handlers AFTER files are loaded
    setupFolderClickHandlers()

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
  
  // Show All Files button
  $('#showAllFiles').addEventListener('click', async ()=>{
    console.log('[DEBUG] Show All Files clicked')
    $('#cwd').textContent = 'All Files'
    await loadFiles(1, perPage)
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
// Expose for inline event handlers used by pagination HTML
// eslint-disable-next-line no-undef
window.loadFiles = loadFiles
// eslint-disable-next-line no-undef
window.changePerPage = changePerPage
