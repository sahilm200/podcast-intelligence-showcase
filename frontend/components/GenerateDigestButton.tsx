'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { createClient } from '@/lib/supabase/client'

export default function GenerateDigestButton() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const router = useRouter()

  const generate = async () => {
    setLoading(true)
    setError(null)
    try {
      const supabase = createClient()
      const { data: { session } } = await supabase.auth.getSession()
      if (!session) return

      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/digests/generate`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${session.access_token}` },
      })

      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Error ${res.status}`)
      }

      const data = await res.json()
      if (data.digest_id) {
        router.push(`/dashboard/digest/${data.digest_id}`)
      } else {
        router.refresh()
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to generate digest')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col items-end gap-1">
      <button
        onClick={generate}
        disabled={loading}
        className="inline-flex items-center gap-2 bg-[#0071e3] hover:bg-[#0077ed] active:bg-[#006cdc] disabled:opacity-50 text-white font-medium rounded-full text-sm px-5 py-2 transition-colors"
      >
        {loading ? (
          <>
            <svg className="w-3.5 h-3.5 animate-spin" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
            </svg>
            Generating…
          </>
        ) : (
          'Generate Digest'
        )}
      </button>
      {error && <p className="text-red-500 text-xs">{error}</p>}
    </div>
  )
}

