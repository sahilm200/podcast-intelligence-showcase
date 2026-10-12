import { createServerClient } from '@supabase/ssr'
import { cookies } from 'next/headers'
import { NextResponse } from 'next/server'

type CookieEntry = { name: string; value: string; options?: Record<string, unknown> }

export async function GET(request: Request) {
  const requestUrl = new URL(request.url)
  const code = requestUrl.searchParams.get('code')
  const origin = requestUrl.origin

  if (code) {
    const cookieStore = await cookies()
    const supabase = createServerClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
      {
        cookies: {
          getAll() {
            return cookieStore.getAll()
          },
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          setAll(cookiesToSet: CookieEntry[]) {
            cookiesToSet.forEach(({ name, value, options }) =>
              cookieStore.set(name, value, options as any)
            )
          },
        },
      }
    )

    const { data } = await supabase.auth.exchangeCodeForSession(code)
    const session = data?.session

    // Store Spotify tokens in backend so it can make Spotify API calls
    if (session?.provider_token && process.env.NEXT_PUBLIC_API_URL) {
      try {
        await fetch(`${process.env.NEXT_PUBLIC_API_URL}/auth/store-spotify-token`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${session.access_token}`,
          },
          body: JSON.stringify({
            provider_token: session.provider_token,
            provider_refresh_token: session.provider_refresh_token,
            expires_in: session.expires_in ?? 3600,
            provider_user_id: session.user?.user_metadata?.provider_id ?? '',
          }),
        })
      } catch {
        // Non-fatal: user is logged in, sync can be triggered manually
      }
    }
  }

  return NextResponse.redirect(`${origin}/dashboard?synced=1`)
}

