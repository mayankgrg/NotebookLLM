import React, { useState } from 'react'
import Chat from './components/Chat'
import Uploader from './components/Uploader'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

export default function App() {
  const [ingested, setIngested] = useState([])

  return (
    <div className="min-h-screen p-6 max-w-5xl mx-auto">
      <header className="mb-6">
        <h1 className="text-3xl font-bold">Notebook‑LLM‑lite</h1>
        <p className="text-sm text-gray-600">Multi‑doc chatbot. Answers are concise summaries grounded in your files.</p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="md:col-span-1">
          <Uploader apiBase={API_BASE} onDone={setIngested} />
          {ingested.length > 0 && (
            <div className="mt-4 text-sm text-gray-700">
              <h3 className="font-semibold">Indexed files</h3>
              <ul className="list-disc ml-5">
                {ingested.map(d => (
                  <li key={d.filename}>{d.filename} <span className="text-gray-500">({d.n_chunks} chunks)</span></li>
                ))}
              </ul>
            </div>
          )}
        </div>
        <div className="md:col-span-2">
          <Chat apiBase={API_BASE} />
        </div>
      </div>
    </div>
  )
}