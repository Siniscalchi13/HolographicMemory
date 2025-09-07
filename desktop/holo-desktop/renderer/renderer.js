const $ = (s) => document.querySelector(s)
const fmt = (n) => new Intl.NumberFormat().format(n)

let CURRENT_DIR = '/'
let CURRENT_ROWS = []
let VIEW_MODE = 'list'
let SORT_KEY = 'name'
let SORT_DIR = 'asc'

function ext(path){ const i = path.lastIndexOf('.'); return i>=0? path.slice(i+1).toLowerCase(): '' }

function iconFor(path){
  const e = ext(path)
  if(['png','jpg','jpeg','gif','webp','bmp'].includes(e)) return '🖼️'
  if(['pdf'].includes(e)) return '📄'
  if(['mp4','mov','avi','mkv'].includes(e)) return '🎞️'
  if(['mp3','wav','flac','m4a'].includes(e)) return '🎵'
  if(['zip','gz','tar','7z'].includes(e)) return '🗜️'
  if(['txt','md','rtf'].includes(e)) return '📝'
  return '📄'
}

function renderTree(node, base=''){ // node: {name,path,dirs[],files[]}
  const full = base ? `${base}/${node.name}` : node.name
  const children = (node.dirs||[]).map(d=>renderTree(d, full)).join('')
  return `<div class="tree-node" data-path="${node.path}">📁 ${node.name}${children?'<div style="margin-left:12px">'+children+'</div>':''}</div>`
}

function renderRows(rows){
  CURRENT_ROWS = rows
  const sorted = [...rows].sort((a,b)=>{
    const na = a.path.split('/').pop(); const nb = b.path.split('/').pop()
    const ta = (na.lastIndexOf('.')>-1? na.slice(na.lastIndexOf('.')+1).toLowerCase(): '')
    const tb = (nb.lastIndexOf('.')>-1? nb.slice(nb.lastIndexOf('.')+1).toLowerCase(): '')
    let va, vb
    if (SORT_KEY==='name'){ va = na.toLowerCase(); vb = nb.toLowerCase() }
    else if (SORT_KEY==='type'){ va = ta; vb = tb }
    else if (SORT_KEY==='size'){ va = a.size||0; vb = b.size||0 }
    else { va = a.mtime||0; vb = b.mtime||0 }
    const c = va<vb? -1 : (va>vb? 1 : 0)
    return SORT_DIR==='asc'? c : -c
  })
  const grid = document.getElementById('gridview')
  const list = document.getElementById('listview')
  if (VIEW_MODE==='grid'){
    list.style.display='none'
    grid.style.display='grid'
    grid.innerHTML = sorted.map(r=>`<div class="tile" data-path="${r.path}">
      <img id="img_${btoa(r.path)}" />
      <div class="name" title="${r.path}">${r.path.split('/').pop()}</div>
    </div>`).join('')
    sorted.forEach(async r=>{
      const url = await window.holo.thumb(r.path, 256)
      const el = document.getElementById(`img_${btoa(r.path)}`)
      if (el && url) el.src = url
    })
    bindTiles()
  } else {
    grid.style.display='none'
    list.style.display='table'
    document.getElementById('rows').innerHTML = sorted.map(r=>{
      const name = r.path.split('/').pop()
      return `<tr data-path="${r.path}">
        <td>${iconFor(r.path)}</td>
        <td>${name}</td>
        <td>${ext(name) || '—'}</td>
        <td>${fmt(r.size||0)}</td>
        <td>${r.mtime? new Date(r.mtime*1000).toLocaleString(): '—'}</td>
      </tr>`
    }).join('')
    bindRowEvents()
  }
  const sb = document.getElementById('statusBar'); if (sb) sb.textContent = `${sorted.length} items` + (SELECTED_PATH? ` — selected: ${SELECTED_PATH.split('/').pop()}`:'')
}

async function refresh() {
  try {
    const stats = await window.holo.stats()
    $('#status').textContent = 'Connected'
    $('#stats').innerHTML = `
      <div class="row">
        <div class="pill">dim ${stats.dimension ?? '—'}</div>
        <div class="pill">files ${stats.files_indexed ?? 0}</div>
        <div class="pill">bytes ${fmt(stats.original_total_bytes ?? 0)}</div>
        <div class="pill">compression x${stats.compression_x ?? '—'}</div>
      </div>`

    const tree = await window.holo.tree()
    $('#tree').innerHTML = renderTree(tree)
    bindTree()
    const list = await window.holo.list()
    renderRows(list.results || [])
  } catch (e) {
    $('#status').textContent = 'API not running. Start: `make api` or `docker compose up`'
  }
}

function setupDnD() {
  const drop = $('#drop')
  drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.style.background = '#eef2ff' })
  drop.addEventListener('dragleave', () => { drop.style.background = '#fff' })
  drop.addEventListener('drop', async (e) => {
    e.preventDefault(); drop.style.background = '#fff'
    const files = e.dataTransfer.files
    for (const f of files) {
      await window.holo.upload(f)
    }
    await refresh()
  })
  $('#file').addEventListener('change', async (e) => {
    const files = e.target.files
    for (const f of files) await window.holo.upload(f)
    await refresh()
  })
}

function setupSearch() {
  const q = $('#q')
  q.addEventListener('input', async () => {
    const v = q.value.trim()
    if (!v) return refresh()
    const res = await window.holo.search(v)
    renderRows(res.results || [])
  })
}

function bindTree(){
  document.querySelectorAll('.tree-node').forEach(el=>{
    el.addEventListener('click', ()=>{
      const p = el.getAttribute('data-path')
      CURRENT_DIR = p
      $('#cwd').textContent = p
      const rows = CURRENT_ROWS.filter(r => r.path.startsWith(p))
      renderRows(rows)
    })
  })
}

function bindRowEvents(){
  document.querySelectorAll('#rows tr').forEach(tr=>{
    tr.addEventListener('click', async ()=>{
      const path = tr.getAttribute('data-path')
      $('#rows tr').forEach(x=>x.classList.remove('sel'))
      tr.classList.add('sel')
      SELECTED_PATH = path
      const url = await window.holo.thumb(path, 320)
      if (url) {
        $('#preview').innerHTML = `<img src="${url}" style="max-width:100%"/>`
      } else {
        $('#preview').innerHTML = `<div class="muted">No preview available</div>`
      }
      document.getElementById('statusBar').textContent = `${CURRENT_ROWS.length} items — selected: ${path.split('/').pop()}`
    })
    tr.addEventListener('contextmenu', (e)=>{
      e.preventDefault()
      const path = tr.getAttribute('data-path')
      showContextMenu(e.pageX, e.pageY, path)
    })
  })
}

function showContextMenu(x,y,path){
  let menu = document.getElementById('ctx')
  if (!menu){
    menu = document.createElement('div')
    menu.id = 'ctx'
    menu.style.position = 'fixed'
    menu.style.background = '#fff'
    menu.style.border = '1px solid #cbd5e1'
    menu.style.borderRadius = '6px'
    menu.style.boxShadow = '0 6px 24px rgba(0,0,0,0.12)'
    menu.style.zIndex = '1000'
    document.body.appendChild(menu)
    document.addEventListener('click', ()=> menu.remove(), { once:true })
  }
  menu.style.left = x+'px'; menu.style.top = y+'px'
  menu.innerHTML = `<div style="padding:8px 12px; cursor:pointer" id="actDel">Delete</div>
  <div style="padding:8px 12px; cursor:pointer" id="actRen">Rename</div>
  <div style="padding:8px 12px; cursor:pointer" id="actProps">Properties</div>`
  menu.querySelector('#actDel').onclick = async ()=>{ await window.holo.delete(path); menu.remove(); refresh() }
  menu.querySelector('#actRen').onclick = async ()=>{
    const bn = path.split('/').pop()
    const nn = prompt('New name', bn)
    if (nn && nn!==bn){
      const np = path.slice(0, path.length-bn.length)+nn
      await window.holo.rename(path, np)
      menu.remove(); refresh()
    }
  }
  menu.querySelector('#actProps').onclick = async ()=>{
    const info = await window.holo.fileinfo(path).catch(()=>null)
    if (info){
      const body = `
        <div><b>Name:</b> ${path.split('/').pop()}</div>
        <div><b>Path:</b> ${info.path}</div>
        <div><b>Type:</b> ${ext(path)}</div>
        <div><b>Size:</b> ${fmt(info.size||0)} bytes</div>
        <div><b>Modified:</b> ${info.mtime? new Date(info.mtime*1000).toLocaleString(): '—'}</div>
        <div><b>Doc ID:</b> ${info.doc_id}</div>
        <div><b>SHA-256:</b> ${info.sha256||'—'}</div>
        <div><b>HM Preview:</b> ${info.has_preview? 'Yes':'No'}</div>
      `
      document.getElementById('propsBody').innerHTML = body
      document.getElementById('props').style.display='flex'
      document.getElementById('propsClose').onclick = ()=> document.getElementById('props').style.display='none'
    }
    menu.remove()
  }
}

function bindTiles(){
  document.querySelectorAll('.tile').forEach(tile=>{
    tile.addEventListener('click', async ()=>{
      const path = tile.getAttribute('data-path')
      const url = await window.holo.thumb(path, 320)
      if (url) document.getElementById('preview').innerHTML = `<img src="${url}" style="max-width:100%"/>`
      else document.getElementById('preview').innerHTML = `<div class="muted">No preview available</div>`
    })
    tile.addEventListener('contextmenu', (e)=>{
      e.preventDefault()
      const path = tile.getAttribute('data-path')
      showContextMenu(e.pageX, e.pageY, path)
    })
  })
}

function setupViewToggle(){
  const btn = document.getElementById('viewMode')
  btn.addEventListener('click', ()=>{
    VIEW_MODE = VIEW_MODE==='list'? 'grid':'list'
    btn.textContent = VIEW_MODE
    renderRows(CURRENT_ROWS)
  })
  document.querySelectorAll('.sortable').forEach(th=>{
    th.addEventListener('click', ()=>{
      const key = th.getAttribute('data-sort')
      if (SORT_KEY===key) SORT_DIR = SORT_DIR==='asc'? 'desc':'asc'; else { SORT_KEY=key; SORT_DIR='asc' }
      renderRows(CURRENT_ROWS)
    })
  })
}

window.addEventListener('DOMContentLoaded', async () => {
  setupDnD()
  setupSearch()
  setupViewToggle()
  await refresh()
})
