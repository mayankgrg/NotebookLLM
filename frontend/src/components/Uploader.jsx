import React, { useState } from 'react'

export default function Uploader({ apiBase, onDone }) {
  const [files, setFiles] = useState([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const onChange = (e) => setFiles(Array.from(e.target.files || []))

  const onUpload = async () => {
    if (!files.length) return
    setBusy(true); setError('')
    const fd = new FormData()
    files.forEach(f => fd.append('files', f))
    try {
      const res = await fetch(`${apiBase}/ingest`, { method: 'POST', body: fd })
      if (!res.ok) throw new Error('Upload failed')
      const data = await res.json()
      onDone(data.docs || [])
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="p-4 bg-white rounded-2xl shadow">
      <h2 className="font-semibold mb-2">Upload documents</h2>
      <input type="file" multiple onChange={onChange} className="mb-3" />
      <button onClick={onUpload} disabled={busy || !files.length} className="px-3 py-2 rounded-xl bg-black text-white disabled:opacity-50">
        {busy ? 'Indexing…' : 'Ingest'}
      </button>
      {error && <p className="text-red-600 text-sm mt-2">{error}</p>}
      <p className="text-xs text-gray-500 mt-2">Supports txt, md, pdf, json, csv and most plain-text files.</p>
    </div>
  )
}