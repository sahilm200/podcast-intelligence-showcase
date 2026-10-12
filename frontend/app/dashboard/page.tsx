import { createClient } from '@/lib/supabase/server'
import { redirect } from 'next/navigation'
import Link from 'next/link'
import Nav from '@/components/Nav'
import SyncSpotifyButton from '@/components/SyncSpotifyButton'
import GenerateDigestButton from '@/components/GenerateDigestButton'

interface Podcast {
  id: string; title: string; description: string | null
  image_url: string | null; author: string | null
}
interface Subscription {
  id: string; subscribed_at: string; is_active: boolean; podcast: Podcast
}
interface Digest {
  id: string; digest_date: string; title: string | null; status: string
}

const STATUS_STYLES: Record<string, string> = {
  completed: 'bg-[#d1f0d8] text-[#1a7d3a]',
  generating: 'bg-[#fff3cd] text-[#856404]',
  pending: 'bg-[#e8f0fe] text-[#1a56c4]',
  failed: 'bg-[#fde8e8] text-[#c41a1a]',
}

export default async function DashboardPage({
  searchParams,
}: {
  searchParams: Promise<{ synced?: string }>
}) {
  const { synced } = await searchParams
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) redirect('/')

  const [subsResult, digestsResult] = await Promise.all([
    supabase
      .from('user_subscriptions')
      .select('id, subscribed_at, is_active, podcast:podcasts(id, title, description, image_url, author)')
      .eq('is_active', true)
      .order('subscribed_at', { ascending: false })
      .limit(12),
    supabase
      .from('user_digests')
      .select('id, digest_date, title, status')
      .eq('user_id', user.id)
      .order('digest_date', { ascending: false })
      .limit(5),
  ])

  const subscriptions = (subsResult.data || []) as unknown as Subscription[]
  const digests = (digestsResult.data || []) as Digest[]
  const name = user.user_metadata?.full_name?.split(' ')[0] || user.email?.split('@')[0] || 'there'

  return (
    <div className="min-h-screen" style={{ background: '#f5f5f7' }}>
      <Nav active="dashboard" email={user.email} />

      <main className="max-w-6xl mx-auto px-6 py-10 space-y-10">
        {/* Header row */}
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div>
            <h1 className="text-[28px] font-bold text-[#1d1d1f] tracking-tight">Good to see you, {name}.</h1>
            <p className="text-[#6e6e73] text-[15px] mt-1">
              {subscriptions.length > 0
                ? `${subscriptions.length} podcast${subscriptions.length !== 1 ? 's' : ''} in your library`
                : 'Sync your Spotify to get started'}
            </p>
          </div>
          <GenerateDigestButton />
        </div>

        {/* Digests */}
        <section>
          <h2 className="text-[17px] font-semibold text-[#1d1d1f] mb-3">Digests</h2>
          {digests.length === 0 ? (
            <div style={{ background: 'rgba(255,255,255,0.6)', backdropFilter: 'blur(16px)', WebkitBackdropFilter: 'blur(16px)', border: '1px solid rgba(0,0,0,0.06)' }}
              className="rounded-2xl p-8 text-center">
              <p className="text-[#6e6e73] text-sm">No digests yet — hit Generate Digest to create your first one.</p>
            </div>
          ) : (
            <div className="space-y-2">
              {digests.map((d) => (
                <Link key={d.id} href={`/dashboard/digest/${d.id}`}
                  style={{ background: 'rgba(255,255,255,0.7)', backdropFilter: 'blur(16px)', WebkitBackdropFilter: 'blur(16px)', border: '1px solid rgba(0,0,0,0.06)', boxShadow: '0 1px 8px rgba(0,0,0,0.04)' }}
                  className="flex items-center justify-between rounded-2xl px-5 py-4 hover:border-[rgba(0,0,0,0.12)] transition-all group">
                  <div>
                    <p className="text-[15px] font-medium text-[#1d1d1f] group-hover:text-[#0071e3] transition-colors">
                      {d.title || `Digest — ${d.digest_date}`}
                    </p>
                    <p className="text-[13px] text-[#aeaeb2] mt-0.5">
                      {new Date(d.digest_date).toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' })}
                    </p>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className={`text-[11px] font-medium px-2.5 py-1 rounded-full ${STATUS_STYLES[d.status] || STATUS_STYLES.pending}`}>
                      {d.status}
                    </span>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className="text-[#aeaeb2]">
                      <path d="M9 18l6-6-6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </div>
                </Link>
              ))}
            </div>
          )}
        </section>

        {/* Podcasts grid */}
        <section>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-[17px] font-semibold text-[#1d1d1f]">Your Library</h2>
            {subscriptions.length > 0 && (
              <Link href="/dashboard/podcasts" className="text-[13px] text-[#0071e3] hover:underline">
                View all
              </Link>
            )}
          </div>

          {subscriptions.length === 0 ? (
            <div style={{ background: 'rgba(255,255,255,0.6)', backdropFilter: 'blur(16px)', WebkitBackdropFilter: 'blur(16px)', border: '1px dashed rgba(0,0,0,0.12)' }}
              className="rounded-2xl p-12 text-center">
              <p className="text-[#1d1d1f] font-medium mb-1">No podcasts yet</p>
              <p className="text-[#6e6e73] text-sm mb-5">Follow podcasts on Spotify, then sync your library.</p>
              <SyncSpotifyButton autoSync={synced === '1'} />
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
              {subscriptions.map((sub) => (
                <div key={sub.id} style={{ background: 'rgba(255,255,255,0.7)', backdropFilter: 'blur(16px)', WebkitBackdropFilter: 'blur(16px)', border: '1px solid rgba(0,0,0,0.06)', boxShadow: '0 1px 8px rgba(0,0,0,0.04)' }}
                  className="rounded-2xl p-3 flex gap-3 items-center hover:border-[rgba(0,0,0,0.12)] transition-all">
                  {sub.podcast?.image_url ? (
                    <img src={sub.podcast.image_url} alt={sub.podcast.title} className="w-11 h-11 rounded-xl object-cover shrink-0" />
                  ) : (
                    <div className="w-11 h-11 rounded-xl shrink-0 flex items-center justify-center" style={{ background: 'rgba(0,113,227,0.08)' }}>
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                        <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" fill="#0071e3" opacity=".6"/>
                      </svg>
                    </div>
                  )}
                  <div className="min-w-0">
                    <p className="text-[13px] font-medium text-[#1d1d1f] truncate leading-tight">{sub.podcast?.title}</p>
                    <p className="text-[11px] text-[#aeaeb2] truncate mt-0.5">{sub.podcast?.author}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  )
}

