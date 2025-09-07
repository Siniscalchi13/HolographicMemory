import { contextBridge, ipcRenderer } from 'electron'

const API_BASE = process.env.HOLO_API || 'http://localhost:8080'
const API_KEY = process.env.HOLO_API_KEY || ''

async function api(path, opts = {}) {
  opts.headers = Object.assign({}, opts.headers || {}, API_KEY ? { 'X-API-Key': API_KEY } : {})
  const res = await fetch(`${API_BASE}${path}`, opts)
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
})
