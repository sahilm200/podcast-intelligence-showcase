'use client'

import Link from 'next/link'
import LogoutButton from '@/components/LogoutButton'

interface Props {
  active?: 'dashboard' | 'podcasts' | 'preferences'
  email?: string
}

const navLinks = [
  { key: 'dashboard', label: 'Dashboard', href: '/dashboard' },
  { key: 'podcasts', label: 'Library', href: '/dashboard/podcasts' },
  { key: 'preferences', label: 'Preferences', href: '/dashboard/preferences' },
]

export default function Nav({ active, email }: Props) {
  return (
    <header
      style={{
        background: 'rgba(245,245,247,0.88)',
        backdropFilter: 'blur(24px) saturate(200%)',
        WebkitBackdropFilter: 'blur(24px) saturate(200%)',
        borderBottom: '1px solid rgba(0,0,0,0.06)',
      }}
      className="sticky top-0 z-50 px-6"
    >
      <div className="max-w-6xl mx-auto h-12 flex items-center justify-between">
        {/* Logo */}
        <Link href="/dashboard" className="flex items-center gap-2 shrink-0">
          <div className="w-6 h-6 rounded-lg flex items-center justify-center" style={{ background: '#0071e3' }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
              <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" fill="white"/>
              <path d="M19 10v1a7 7 0 0 1-14 0v-1M12 19v4M8 23h8" stroke="white" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </div>
          <span className="text-[14px] font-semibold text-[#1d1d1f] tracking-tight">Parakeet</span>
        </Link>

        {/* Nav links */}
        <nav className="flex items-center gap-1">
          {navLinks.map((link) => (
            <Link
              key={link.key}
              href={link.href}
              className={`px-3 py-1.5 rounded-full text-[13px] font-medium transition-all ${
                active === link.key
                  ? 'bg-[rgba(0,113,227,0.1)] text-[#0071e3]'
                  : 'text-[#6e6e73] hover:text-[#1d1d1f] hover:bg-[rgba(0,0,0,0.04)]'
              }`}
            >
              {link.label}
            </Link>
          ))}
        </nav>

        {/* Right side */}
        <div className="flex items-center gap-3 shrink-0">
          {email && (
            <span className="text-[12px] text-[#aeaeb2] hidden sm:block truncate max-w-[140px]">{email}</span>
          )}
          <LogoutButton />
        </div>
      </div>
    </header>
  )
}

