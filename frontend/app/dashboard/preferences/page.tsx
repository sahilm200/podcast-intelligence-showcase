import { createClient } from '@/lib/supabase/server'
import { redirect } from 'next/navigation'
import Nav from '@/components/Nav'
import PreferencesForm from '@/components/PreferencesForm'

export default async function PreferencesPage() {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) redirect('/')

  const { data: session } = await supabase.auth.getSession()
  const accessToken = session?.session?.access_token ?? ''

  let preferences = null
  if (accessToken) {
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/digests/preferences`, {
        headers: { Authorization: `Bearer ${accessToken}` },
        cache: 'no-store',
      })
      if (res.ok) preferences = await res.json()
    } catch { /* use defaults */ }
  }

  return (
    <div className="min-h-screen" style={{ background: '#f5f5f7' }}>
      <Nav active="preferences" email={user.email} />

      <main className="max-w-2xl mx-auto px-6 py-10">
        <h1 className="text-[28px] font-bold text-[#1d1d1f] tracking-tight mb-1">Preferences</h1>
        <p className="text-[#6e6e73] text-[15px] mb-8">Control how and when your digests are generated.</p>

        <div style={{ background: 'rgba(255,255,255,0.75)', backdropFilter: 'blur(20px)', WebkitBackdropFilter: 'blur(20px)', border: '1px solid rgba(0,0,0,0.06)', boxShadow: '0 2px 24px rgba(0,0,0,0.06)' }}
          className="rounded-2xl p-6">
          <PreferencesForm initialPreferences={preferences} accessToken={accessToken} />
        </div>
      </main>
    </div>
  )
}

