import { createClient } from '@/lib/supabase/server'
import { redirect, notFound } from 'next/navigation'
import Link from 'next/link'
import Nav from '@/components/Nav'
import DigestContent from '@/components/DigestContent'

const STATUS_STYLES: Record<string, string> = {
  completed: 'bg-[#d1f0d8] text-[#1a7d3a]',
  generating: 'bg-[#fff3cd] text-[#856404]',
  pending: 'bg-[#e8f0fe] text-[#1a56c4]',
  failed: 'bg-[#fde8e8] text-[#c41a1a]',
}

export default async function DigestPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) redirect('/')

  const { data: digest } = await supabase
    .from('user_digests')
    .select('*')
    .eq('id', id)
    .eq('user_id', user.id)
    .single()

  if (!digest) notFound()

  const dateLabel = new Date(digest.digest_date).toLocaleDateString('en-US', {
    weekday: 'long', year: 'numeric', month: 'long', day: 'numeric',
  })

  return (
    <div className="min-h-screen" style={{ background: '#f5f5f7' }}>
      <Nav email={user.email} />

      {/* Sub-header */}
      <div style={{ background: 'rgba(245,245,247,0.8)', backdropFilter: 'blur(16px)', WebkitBackdropFilter: 'blur(16px)', borderBottom: '1px solid rgba(0,0,0,0.06)' }}
        className="px-6 py-3">
        <div className="max-w-3xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link href="/dashboard" className="text-[13px] text-[#6e6e73] hover:text-[#1d1d1f] transition-colors flex items-center gap-1">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                <path d="M15 18l-6-6 6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              Dashboard
            </Link>
            <span className="text-[#d2d2d7]">/</span>
            <span className="text-[13px] text-[#1d1d1f] font-medium truncate max-w-[200px]">
              {digest.title || dateLabel}
            </span>
          </div>
          <span className={`text-[11px] font-medium px-2.5 py-1 rounded-full ${STATUS_STYLES[digest.status] || STATUS_STYLES.pending}`}>
            {digest.status}
          </span>
        </div>
      </div>

      <main className="max-w-3xl mx-auto px-6 py-10">
        <p className="text-[13px] text-[#aeaeb2] mb-6">{dateLabel}</p>

        {digest.status === 'completed' && digest.content ? (
          <div style={{ background: 'rgba(255,255,255,0.75)', backdropFilter: 'blur(20px)', WebkitBackdropFilter: 'blur(20px)', border: '1px solid rgba(0,0,0,0.06)', boxShadow: '0 2px 24px rgba(0,0,0,0.06)' }}
            className="rounded-2xl p-8">
            <DigestContent content={digest.content} />
          </div>
        ) : digest.status === 'generating' ? (
          <div style={{ background: 'rgba(255,255,255,0.7)', border: '1px solid rgba(0,0,0,0.06)' }}
            className="rounded-2xl p-16 text-center">
            <div className="w-8 h-8 border-2 border-[#0071e3] border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-[#1d1d1f] font-medium">Generating your digest</p>
            <p className="text-[#6e6e73] text-sm mt-1">Refresh to check progress.</p>
          </div>
        ) : (
          <div style={{ background: 'rgba(255,255,255,0.7)', border: '1px solid rgba(0,0,0,0.06)' }}
            className="rounded-2xl p-16 text-center">
            <p className="text-[#1d1d1f] font-medium mb-1">
              {digest.status === 'failed' ? 'Generation failed' : 'Digest pending'}
            </p>
            <p className="text-[#6e6e73] text-sm">
              {digest.status === 'failed'
                ? 'Try generating again from the dashboard.'
                : 'Episodes are still being processed.'}
            </p>
            <Link href="/dashboard" className="inline-block mt-4 text-[13px] text-[#0071e3] hover:underline">
              Back to dashboard
            </Link>
          </div>
        )}
      </main>
    </div>
  )
}

