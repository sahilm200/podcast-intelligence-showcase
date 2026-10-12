import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Parakeet — AI Podcast Digests',
  description: 'Connect your Spotify and get AI-powered digests of your favorite podcasts.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}

