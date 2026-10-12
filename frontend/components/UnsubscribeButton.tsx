'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { createClient } from '@/lib/supabase/client'

interface Props {
  podcastId: string
  podcastTitle?: string
}

export default function UnsubscribeButton({ podcastId, podcastTitle }: Props) {
  const [loading, setLoading] = useState(false)
  const [confirmed, setConfirmed] = useState(false)
  const router = useRouter()

  const unsubscribe = async () => {
    if (!confirmed) {
      setConfirmed(true)
      setTimeout(() => setConfirmed(false), 3000)
      return
    }
    setLoading(true)
    try {
      const supabase = createClient()
      const { data: { session } } = await supabase.auth.getSession()
      if (!session) return

      await fetch(`${process.env.NEXT_PUBLIC_API_URL}/subscriptions/unsubscribe`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${session.access_token}`,
        },
        body: JSON.stringify({ podcast_id: podcastId }),
      })

      router.refresh()
    } finally {
      setLoading(false)
      setConfirmed(false)
    }
  }

  return (
    <button
      onClick={unsubscribe}
      disabled={loading}
      className={`w-full text-[12px] py-1.5 rounded-xl border transition-all ${
        confirmed
          ? 'border-[#ff3b30] text-[#ff3b30] bg-[rgba(255,59,48,0.06)] hover:bg-[rgba(255,59,48,0.1)]'
          : 'border-[rgba(0,0,0,0.1)] text-[#aeaeb2] hover:text-[#6e6e73] hover:border-[rgba(0,0,0,0.18)]'
      }`}
    >
      {loading ? 'Removing…' : confirmed ? `Remove ${podcastTitle ? `"${podcastTitle.slice(0, 20)}${podcastTitle.length > 20 ? '…' : ''}"` : 'this podcast'}?` : 'Unsubscribe'}
    </button>
  )
}

