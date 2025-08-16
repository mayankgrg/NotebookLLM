import React, { useState } from 'react'

export default function Chat({ apiBase }) {
  const [query, setQuery] = useState('')
  const [busy, setBusy] = useState(false)
  const [messages, setMessages] = useState([])

  const ask = async () => {
    if (!query.trim()) return
    setBusy(true)
    const q = query
    setQuery('')
    setMessages(m => [...m, { role: 'user', content: q }])
    try {
      const res = await fetch(`${apiBase}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, top_k: 6, max_sentences: 6 })
      })
      const data = await res.json()
      setMessages(m => [...m, { role: 'assistant', content: data.answer, citations: data.citations }])
    } catch (e) {
      setMessages(m => [...m, { role: 'assistant', content: 'Error: ' + e.message }])
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="p-4 bg-white rounded-2xl shadow min-h-[500px] flex flex-col">
      <div className="flex-1 overflow-auto space-y-4">
        {messages.map((m, i) => (
          <div key={i} className={m.role === 'user' ? 'text-right' : 'text-left'}>
            <div className={`inline-block px-3 py-2 rounded-2xl ${m.role==='user' ? 'bg-gray-900 text-white' : 'bg-gray-100'}`}>
              <p className="whitespace-pre-wrap">{m.content}</p>
              {m.citations && (
                <details className="mt-1 text-xs text-gray-600">
                  <summary>Citations</summary>
                  <ul className="list-disc ml-5">
                    {m.citations.map((c, j) => (
                      <li key={j}>{c.filename} • chunk {c.chunk} • score {c.score.toFixed(3)}</li>
                    ))}
                  </ul>
                </details>
              )}
            </div>
          </div>
        ))}
      </div>
      <div className="mt-4 flex gap-2">
        <input
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={e => e.key==='Enter' && ask()}
          placeholder="Ask a question about your documents…"
          className="flex-1 border rounded-xl px-3 py-2"
        />
        <button onClick={ask} disabled={busy} className="px-4 py-2 rounded-xl bg-black text-white disabled:opacity-50">Send</button>
      </div>
    </div>
  )
}