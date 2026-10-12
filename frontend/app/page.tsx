import { createClient } from '@/lib/supabase/server'
import { redirect } from 'next/navigation'
import SpotifyLoginButton from '@/components/SpotifyLoginButton'

export default async function LandingPage() {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()
  if (user) redirect('/dashboard')

  return (
    <main className="min-h-screen" style={{ background: 'linear-gradient(160deg, #e8f0fe 0%, #f5f5f7 40%, #fce8f0 100%)' }}>
      {/* Nav */}
      <nav style={{
        background: 'rgba(245,245,247,0.8)',
        backdropFilter: 'blur(24px) saturate(200%)',
        WebkitBackdropFilter: 'blur(24px) saturate(200%)',
        borderBottom: '1px solid rgba(0,0,0,0.06)',
      }} className="sticky top-0 z-50">
        <div className="max-w-5xl mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 14H9V8h2v8zm4 0h-2V8h2v8z" fill="#0071e3"/>
            </svg>
            <span className="font-semibold text-[15px] text-[#1d1d1f] tracking-tight">Parakeet</span>
          </div>
          <SpotifyLoginButton />
        </div>
      </nav>

      {/* Hero */}
      <div className="max-w-5xl mx-auto px-6 pt-24 pb-16 text-center">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-sm text-[#6e6e73] mb-8" style={{
          background: 'rgba(255,255,255,0.7)',
          backdropFilter: 'blur(16px)',
          WebkitBackdropFilter: 'blur(16px)',
          border: '1px solid rgba(0,0,0,0.07)',
        }}>
          <span className="w-1.5 h-1.5 bg-[#30d158] rounded-full"></span>
          AI-powered podcast intelligence
        </div>

        <h1 className="text-[56px] md:text-[72px] font-bold tracking-tight text-[#1d1d1f] leading-[1.05] mb-6">
          Your podcasts,<br />
          <span style={{ color: '#0071e3' }}>distilled.</span>
        </h1>

        <p className="text-[19px] text-[#6e6e73] max-w-xl mx-auto leading-relaxed mb-10">
          Connect Spotify and get AI-generated digests of every podcast you follow. Key insights, quotes, and takeaways — without the time commitment.
        </p>

        <div className="flex flex-col items-center gap-3">
          <SpotifyLoginButton />
          <p className="text-sm text-[#aeaeb2]">Free · No credit card required</p>
        </div>
      </div>

      {/* Feature cards */}
      <div className="max-w-5xl mx-auto px-6 pb-24">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[
            {
              icon: (
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
                  <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" fill="#0071e3" opacity=".9"/>
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2H3v2a9 9 0 0 0 8 8.94V23h2v-2.06A9 9 0 0 0 21 12v-2h-2z" fill="#0071e3" opacity=".5"/>
                </svg>
              ),
              title: 'Auto-sync',
              desc: 'Connects to your Spotify follows. Your library, always up to date.',
            },
            {
              icon: (
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
                  <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" stroke="#0071e3" strokeWidth="2" strokeLinecap="round"/>
                  <polyline points="16 17 21 12 16 7" stroke="#0071e3" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <line x1="21" y1="12" x2="9" y2="12" stroke="#0071e3" strokeWidth="2" strokeLinecap="round"/>
                </svg>
              ),
              title: 'AI Summaries',
              desc: 'Key insights, memorable quotes, and actionable takeaways from every episode.',
            },
            {
              icon: (
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
                  <rect x="3" y="4" width="18" height="18" rx="2" stroke="#0071e3" strokeWidth="2"/>
                  <line x1="16" y1="2" x2="16" y2="6" stroke="#0071e3" strokeWidth="2" strokeLinecap="round"/>
                  <line x1="8" y1="2" x2="8" y2="6" stroke="#0071e3" strokeWidth="2" strokeLinecap="round"/>
                  <line x1="3" y1="10" x2="21" y2="10" stroke="#0071e3" strokeWidth="2"/>
                </svg>
              ),
              title: 'Daily Digests',
              desc: 'Catch up on everything you missed. Delivered on your schedule.',
            },
          ].map((f) => (
            <div key={f.title} style={{
              background: 'rgba(255,255,255,0.68)',
              backdropFilter: 'blur(20px) saturate(160%)',
              WebkitBackdropFilter: 'blur(20px) saturate(160%)',
              border: '1px solid rgba(255,255,255,0.9)',
              boxShadow: '0 2px 20px rgba(0,0,0,0.055), 0 0 0 0.5px rgba(0,0,0,0.04)',
            }} className="rounded-2xl p-6">
              <div className="w-10 h-10 rounded-xl flex items-center justify-center mb-4" style={{ background: 'rgba(0,113,227,0.08)' }}>
                {f.icon}
              </div>
              <h3 className="font-semibold text-[15px] text-[#1d1d1f] mb-1">{f.title}</h3>
              <p className="text-sm text-[#6e6e73] leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </main>
  )
}

