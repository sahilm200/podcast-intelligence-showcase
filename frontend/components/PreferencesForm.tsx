'use client'

import { useState } from 'react'

interface Preferences {
  frequency?: string
  delivery_time?: string
  delivery_timezone?: string
  max_episodes_per_digest?: number
  exclude_listened_on_spotify?: boolean
  email_enabled?: boolean
  email_address?: string
}

interface Props {
  initialPreferences: Preferences | null
  accessToken: string
}

export default function PreferencesForm({ initialPreferences, accessToken }: Props) {
  const [prefs, setPrefs] = useState<Preferences>({
    frequency: 'daily',
    delivery_timezone: 'UTC',
    max_episodes_per_digest: 10,
    exclude_listened_on_spotify: true,
    email_enabled: false,
    email_address: '',
    ...initialPreferences,
  })
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const save = async () => {
    setSaving(true)
    setSaved(false)
    setError(null)
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/digests/preferences`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${accessToken}`,
        },
        body: JSON.stringify(prefs),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Error ${res.status}`)
      }
      setSaved(true)
      setTimeout(() => setSaved(false), 3000)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to save')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Frequency */}
      <div>
        <label className="block text-[13px] font-medium text-[#1d1d1f] mb-2">Digest Frequency</label>
        <div className="flex gap-2">
          {(['daily', 'weekly'] as const).map((f) => (
            <button
              key={f}
              onClick={() => setPrefs(p => ({ ...p, frequency: f }))}
              className={`px-4 py-1.5 rounded-full text-[13px] font-medium border transition-all capitalize ${
                prefs.frequency === f
                  ? 'bg-[#0071e3] text-white border-[#0071e3]'
                  : 'bg-transparent text-[#6e6e73] border-[rgba(0,0,0,0.12)] hover:border-[rgba(0,0,0,0.24)]'
              }`}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {/* Max episodes */}
      <div>
        <label className="block text-[13px] font-medium text-[#1d1d1f] mb-1">Max Episodes per Digest</label>
        <p className="text-[#aeaeb2] text-[12px] mb-2">Limit the number of episodes included in each digest.</p>
        <input
          type="number"
          min={1}
          max={50}
          value={prefs.max_episodes_per_digest ?? 10}
          onChange={(e) => setPrefs(p => ({ ...p, max_episodes_per_digest: parseInt(e.target.value) || 10 }))}
          className="w-24 bg-white border border-[rgba(0,0,0,0.12)] rounded-xl px-3 py-2 text-[13px] text-[#1d1d1f] focus:outline-none focus:border-[#0071e3] transition-colors"
        />
      </div>

      {/* Divider */}
      <div className="border-t border-[rgba(0,0,0,0.06)]" />

      {/* Exclude listened */}
      <div className="flex items-center justify-between">
        <div>
          <p className="text-[13px] font-medium text-[#1d1d1f]">Skip Spotify-listened episodes</p>
          <p className="text-[#aeaeb2] text-[12px] mt-0.5">Exclude episodes you&apos;ve already listened to on Spotify.</p>
        </div>
        <button
          onClick={() => setPrefs(p => ({ ...p, exclude_listened_on_spotify: !p.exclude_listened_on_spotify }))}
          className={`relative w-11 h-6 rounded-full transition-colors ${
            prefs.exclude_listened_on_spotify ? 'bg-[#34c759]' : 'bg-[rgba(0,0,0,0.12)]'
          }`}
        >
          <span className={`absolute top-1 w-4 h-4 bg-white rounded-full shadow-sm transition-transform ${
            prefs.exclude_listened_on_spotify ? 'translate-x-6' : 'translate-x-1'
          }`} />
        </button>
      </div>

      {/* Divider */}
      <div className="border-t border-[rgba(0,0,0,0.06)]" />

      {/* Email */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <div>
            <p className="text-[13px] font-medium text-[#1d1d1f]">Email delivery</p>
            <p className="text-[#aeaeb2] text-[12px] mt-0.5">Receive your digest via email.</p>
          </div>
          <button
            onClick={() => setPrefs(p => ({ ...p, email_enabled: !p.email_enabled }))}
            className={`relative w-11 h-6 rounded-full transition-colors ${
              prefs.email_enabled ? 'bg-[#34c759]' : 'bg-[rgba(0,0,0,0.12)]'
            }`}
          >
            <span className={`absolute top-1 w-4 h-4 bg-white rounded-full shadow-sm transition-transform ${
              prefs.email_enabled ? 'translate-x-6' : 'translate-x-1'
            }`} />
          </button>
        </div>
        {prefs.email_enabled && (
          <input
            type="email"
            placeholder="your@example.com"
            value={prefs.email_address ?? ''}
            onChange={(e) => setPrefs(p => ({ ...p, email_address: e.target.value }))}
            className="w-full bg-white border border-[rgba(0,0,0,0.12)] rounded-xl px-4 py-2.5 text-[13px] text-[#1d1d1f] placeholder-[#aeaeb2] focus:outline-none focus:border-[#0071e3] transition-colors"
          />
        )}
      </div>

      {/* Save */}
      <div className="flex items-center gap-4 pt-2">
        <button
          onClick={save}
          disabled={saving}
          className="px-5 py-2 bg-[#0071e3] hover:bg-[#0077ed] active:bg-[#006cdc] disabled:opacity-50 text-white font-medium rounded-full text-[13px] transition-colors"
        >
          {saving ? 'Saving…' : 'Save Preferences'}
        </button>
        {saved && <span className="text-[#34c759] text-[13px] font-medium">Saved</span>}
        {error && <span className="text-[#ff3b30] text-[13px]">{error}</span>}
      </div>
    </div>
  )
}

