import { createClient } from '@/lib/supabase/server'
import { redirect } from 'next/navigation'
import Nav from '@/components/Nav'
import SyncSpotifyButton from '@/components/SyncSpotifyButton'
import UnsubscribeButton from '@/components/UnsubscribeButton'
import GenerateDigestButton from '@/components/GenerateDigestButton'

interface Podcast {
  id: string; title: string; description: string | null
  image_url: string | null; author: string | null
}
interface Subscription {
  id: string; podcast_id: string; subscribed_at: string; podcast: Podcast
}

export default async function PodcastsPage() {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) redirect('/')

  const { data } = await supabase
    .from('user_subscriptions')
    .select('id, podcast_id, subscribed_at, podcast:podcasts(id, title, description, image_url, author)')
    .eq('is_active', true)
    .order('subscribed_at', { ascending: false })

  const subscriptions = (data || []) as unknown as Subscription[]

  return (
    <div className="min-h-screen" style={{ background: '#f5f5f7' }}>
      <Nav active="podcasts" email={user.email} />

      <main className="max-w-6xl mx-auto px-6 py-10">
        <div className="flex items-start justify-between mb-8 flex-wrap gap-4">
          <div>
            <h1 className="text-[28px] font-bold text-[#1d1d1f] tracking-tight">Your Library</h1>
            <p className="text-[#6e6e73] text-[15px] mt-1">
              {subscriptions.length} podcast{subscriptions.length !== 1 ? 's' : ''} synced from Spotify
            </p>
          </div>
          <div className="flex items-center gap-3">
            <SyncSpotifyButton />
            {subscriptions.length > 0 && <GenerateDigestButton />}
          </div>
        </div>

        {subscriptions.length === 0 ? (
          <div style={{ background: 'rgba(255,255,255,0.6)', backdropFilter: 'blur(16px)', WebkitBackdropFilter: 'blur(16px)', border: '1px dashed rgba(0,0,0,0.12)' }}
            className="rounded-2xl p-16 text-center">
            <p className="text-[#1d1d1f] font-medium mb-1">No podcasts yet</p>
            <p className="text-[#6e6e73] text-sm mb-5">Follow podcasts on Spotify, then sync your library.</p>
            <SyncSpotifyButton />
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {subscriptions.map((sub) => (
              <div key={sub.id}
                style={{ background: 'rgba(255,255,255,0.7)', backdropFilter: 'blur(16px)', WebkitBackdropFilter: 'blur(16px)', border: '1px solid rgba(0,0,0,0.06)', boxShadow: '0 1px 8px rgba(0,0,0,0.04)' }}
                className="rounded-2xl p-4">
                <div className="flex gap-3 items-start mb-3">
                  {sub.podcast?.image_url ? (
                    <img src={sub.podcast.image_url} alt={sub.podcast.title} className="w-14 h-14 rounded-xl object-cover shrink-0" />
                  ) : (
                    <div className="w-14 h-14 rounded-xl shrink-0 flex items-center justify-center" style={{ background: 'rgba(0,113,227,0.08)' }}>
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                        <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" fill="#0071e3" opacity=".6"/>
                      </svg>
                    </div>
                  )}
                  <div className="min-w-0 flex-1 pt-0.5">
                    <p className="text-[14px] font-medium text-[#1d1d1f] leading-tight line-clamp-2">{sub.podcast?.title}</p>
                    <p className="text-[12px] text-[#aeaeb2] mt-0.5 truncate">{sub.podcast?.author}</p>
                  </div>
                </div>
                {sub.podcast?.description && (
                  <p className="text-[12px] text-[#6e6e73] line-clamp-2 mb-3 leading-relaxed">{sub.podcast.description}</p>
                )}
                <UnsubscribeButton podcastId={sub.podcast_id} podcastTitle={sub.podcast?.title} />
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  )
}

