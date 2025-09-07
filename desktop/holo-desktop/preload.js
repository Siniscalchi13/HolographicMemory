import { contextBridge, ipcRenderer } from 'electron'

let API_BASE = process.env.HOLO_API || ''
const API_KEY = process.env.HOLO_API_KEY || ''

async function ping(url, ms = 800) {
  try {
    const ctl = new AbortController()
    const t = setTimeout(() => ctl.abort(), ms)
    const res = await fetch(`${url}/healthz`, { signal: ctl.signal })
    clearTimeout(t)
    return res.ok
  } catch { return false }
}

async function selectApiBase() {
  if (API_BASE) return API_BASE
  const port = process.env.HOLO_API_PORT || '8000'
  const candidates = [
    `http://localhost:${port}`,
    'http://127.0.0.1:8000',
    'http://localhost:8000',
    'http://localhost:8080'
  ]
  for (const c of candidates) {
    if (await ping(c)) { API_BASE = c; return API_BASE }
  }
  // Fallback even if ping fails
  API_BASE = candidates[0]
  return API_BASE
}

async function api(path, opts = {}) {
  const base = await selectApiBase()
  opts.headers = Object.assign({}, opts.headers || {}, API_KEY ? { 'X-API-Key': API_KEY } : {})
  const res = await fetch(`${base}${path}`, opts)
  if (!res.ok) throw new Error(`API ${path} failed: ${res.status}`)
  return res.json()
}

contextBridge.exposeInMainWorld('holo', {
  stats: () => api('/stats'),
  list: () => api('/list'),
  search: (q) => api(`/search?q=${encodeURIComponent(q)}`),
  tree: () => api('/tree'),
  upload: async (file) => {
    const fd = new FormData()
    fd.append('file', file)
    const res = await fetch(`${API_BASE}/store`, { method: 'POST', body: fd, headers: API_KEY ? { 'X-API-Key': API_KEY } : {} })
    if (!res.ok) throw new Error(`Upload failed: ${res.status}`)
    return res.json()
  },
  delete: (path) => api('/delete', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path }) }),
  rename: (path, new_path) => api('/rename', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path, new_path }) }),
  thumb: async (path, w=256) => {
    const res = await fetch(`${API_BASE}/thumb?path=${encodeURIComponent(path)}&w=${w}`, { headers: API_KEY ? { 'X-API-Key': API_KEY } : {} })
    if (!res.ok) return null
    const blob = await res.blob()
    return URL.createObjectURL(blob)
  },
  fileinfo: (path) => api(`/fileinfo?path=${encodeURIComponent(path)}`)
  ,revealInFolder: (path) => ipcRenderer.invoke('revealInFolder', path)
  ,copyToClipboard: (text) => navigator.clipboard.writeText(text)
  ,apiBase: async () => await selectApiBase()
})
