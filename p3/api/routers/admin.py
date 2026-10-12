"""
Admin endpoints
Manage curated catalog, process catalog requests, and serve the management dashboard.
"""

from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse
from supabase import Client

from p3.api.deps import get_supabase, get_current_admin_user
from p3.api.limiter import limiter
from p3.api.models.admin import (
    CatalogRequestResponse,
    ApproveCatalogRequestRequest,
    RejectCatalogRequestRequest,
    CuratePodcastRequest,
)
from p3.api import queue


router = APIRouter()


@router.get("/catalog-requests", response_model=List[CatalogRequestResponse])
async def list_catalog_requests(
    status_filter: str = Query("pending", description="Filter by status"),
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0),
    supabase: Client = Depends(get_supabase),
    admin_user: dict = Depends(get_current_admin_user),
):
    """
    List podcast catalog requests

    Admin can review user requests to add podcasts to the curated catalog.
    """
    query = (
        supabase.table("podcast_catalog_requests")
        .select("*, users(*)")
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
    )

    if status_filter:
        query = query.eq("status", status_filter)

    result = query.execute()

    return [CatalogRequestResponse(**request) for request in result.data]


@router.post("/catalog-requests/{request_id}/approve")
@limiter.limit("5/minute")
async def approve_catalog_request(
    request: Request,
    request_id: UUID,
    body: ApproveCatalogRequestRequest,
    supabase: Client = Depends(get_supabase),
    admin_user: dict = Depends(get_current_admin_user),
):
    """
    Approve a catalog request

    Creates a curated podcast and marks the request as approved.
    """
    # Get the request
    catalog_request = (
        supabase.table("podcast_catalog_requests")
        .select("*")
        .eq("id", str(request_id))
        .execute()
    )

    if not catalog_request.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Catalog request not found",
        )

    req = catalog_request.data[0]

    if req["status"] != "pending":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request already processed",
        )

    # Check if podcast already exists
    existing = supabase.table("podcasts").select("*").eq("rss_url", req["rss_url"]).execute()

    if existing.data:
        podcast_id = existing.data[0]["id"]
        # Mark as curated if not already
        if not existing.data[0]["is_curated"]:
            supabase.table("podcasts").update({"is_curated": True}).eq("id", podcast_id).execute()
    else:
        # Create new curated podcast
        podcast_data = {
            "title": req.get("title") or body.title,
            "rss_url": req["rss_url"],
            "description": req.get("description") or body.description,
            "category": body.category,
            "is_curated": True,
            "added_by_user_id": None,  # Admin-curated
        }

        podcast_result = supabase.table("podcasts").insert(podcast_data).execute()

        if not podcast_result.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create podcast",
            )

        podcast_id = podcast_result.data[0]["id"]

    # Update request status
    supabase.table("podcast_catalog_requests").update({
        "status": "approved",
        "reviewed_by": admin_user.id,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "review_notes": body.notes,
        "approved_podcast_id": podcast_id,
    }).eq("id", str(request_id)).execute()

    # Queue background job to fetch episodes
    await queue.fetch_podcast_feed(podcast_id)

    return {
        "message": "Catalog request approved",
        "podcast_id": podcast_id,
    }


@router.post("/catalog-requests/{request_id}/reject")
@limiter.limit("5/minute")
async def reject_catalog_request(
    request: Request,
    request_id: UUID,
    body: RejectCatalogRequestRequest,
    supabase: Client = Depends(get_supabase),
    admin_user: dict = Depends(get_current_admin_user),
):
    """
    Reject a catalog request
    """
    # Update request status
    result = (
        supabase.table("podcast_catalog_requests")
        .update({
            "status": "rejected",
            "reviewed_by": admin_user.id,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "review_notes": body.reason,
        })
        .eq("id", str(request_id))
        .eq("status", "pending")
        .execute()
    )

    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pending catalog request not found",
        )

    return {"message": "Catalog request rejected"}


@router.post("/curate-podcast")
@limiter.limit("5/minute")
async def curate_podcast(
    request: Request,
    body: CuratePodcastRequest,
    supabase: Client = Depends(get_supabase),
    admin_user: dict = Depends(get_current_admin_user),
):
    """
    Directly add a podcast to the curated catalog

    Admin can bypass the request system and add podcasts directly.
    """
    # Check if podcast already exists
    existing = supabase.table("podcasts").select("*").eq("rss_url", body.rss_url).execute()

    if existing.data:
        podcast_id = existing.data[0]["id"]
        # Mark as curated if not already
        if not existing.data[0]["is_curated"]:
            supabase.table("podcasts").update({"is_curated": True}).eq("id", podcast_id).execute()

        return {
            "message": "Podcast already exists, marked as curated",
            "podcast_id": podcast_id,
        }

    # Create new curated podcast
    podcast_data = {
        "title": body.title,
        "rss_url": body.rss_url,
        "description": body.description,
        "category": body.category,
        "author": body.author,
        "website_url": body.website_url,
        "spotify_show_id": body.spotify_show_id,
        "is_curated": True,
        "added_by_user_id": None,  # Admin-curated
    }

    result = supabase.table("podcasts").insert(podcast_data).execute()

    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create podcast",
        )

    podcast_id = result.data[0]["id"]

    # Queue background job to fetch episodes
    await queue.fetch_podcast_feed(podcast_id)

    return {
        "message": "Podcast added to curated catalog",
        "podcast_id": podcast_id,
    }


@router.get("/episodes")
async def list_failed_episodes(
    episode_status: str = Query("failed", description="Filter by episode status"),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    supabase: Client = Depends(get_supabase),
    admin_user: dict = Depends(get_current_admin_user),
):
    """
    List episodes by status (default: failed)

    Useful for identifying stuck episodes and triggering retries.
    """
    result = (
        supabase.table("episodes")
        .select("id, podcast_id, title, status, audio_url, created_at, podcasts(title)")
        .eq("status", episode_status)
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )

    return {
        "status_filter": episode_status,
        "count": len(result.data),
        "episodes": result.data,
    }


@router.post("/episodes/{episode_id}/retry")
@limiter.limit("10/minute")
async def retry_episode(
    request: Request,
    episode_id: UUID,
    supabase: Client = Depends(get_supabase),
    admin_user: dict = Depends(get_current_admin_user),
):
    """
    Retry a failed or stuck episode

    Resets episode status to 'pending' and re-enqueues it for download.
    Works on any episode that is not already 'completed'.
    """
    ep_result = (
        supabase.table("episodes")
        .select("id, status, title")
        .eq("id", str(episode_id))
        .execute()
    )

    if not ep_result.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Episode not found",
        )

    episode = ep_result.data[0]
    if episode["status"] == "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Episode is already completed",
        )

    # Reset and re-enqueue from the download step
    supabase.table("episodes").update({"status": "pending"}).eq("id", str(episode_id)).execute()
    await queue.download_episode(str(episode_id))

    return {
        "message": "Episode retry enqueued",
        "episode_id": str(episode_id),
        "previous_status": episode["status"],
    }


@router.post("/podcasts/{podcast_id}/retry-failed")
@limiter.limit("5/minute")
async def retry_failed_episodes(
    request: Request,
    podcast_id: UUID,
    supabase: Client = Depends(get_supabase),
    admin_user: dict = Depends(get_current_admin_user),
):
    """
    Retry all failed episodes for a podcast

    Batch-resets every episode with status 'failed' back to 'pending'
    and re-enqueues them for download.
    """
    failed_result = (
        supabase.table("episodes")
        .select("id")
        .eq("podcast_id", str(podcast_id))
        .eq("status", "failed")
        .execute()
    )

    if not failed_result.data:
        return {"message": "No failed episodes found", "retried": 0}

    episode_ids = [ep["id"] for ep in failed_result.data]

    # Batch reset
    supabase.table("episodes").update({"status": "pending"}).in_(
        "id", episode_ids
    ).execute()

    # Enqueue each
    for ep_id in episode_ids:
        await queue.download_episode(ep_id)

    return {
        "message": f"Retried {len(episode_ids)} failed episodes",
        "retried": len(episode_ids),
        "episode_ids": episode_ids,
    }


@router.get("/stats")
async def get_admin_stats(
    supabase: Client = Depends(get_supabase),
    admin_user: dict = Depends(get_current_admin_user),
):
    """
    Get admin statistics

    Overview of catalog, users, and processing stats.
    """
    # Count podcasts
    total_podcasts = supabase.table("podcasts").select("*", count="exact").execute()
    curated_podcasts = (
        supabase.table("podcasts")
        .select("*", count="exact")
        .eq("is_curated", True)
        .execute()
    )

    # Count episodes
    total_episodes = supabase.table("episodes").select("*", count="exact").execute()
    processed_episodes = (
        supabase.table("episodes")
        .select("*", count="exact")
        .eq("status", "completed")
        .execute()
    )

    # Count pending requests
    pending_requests = (
        supabase.table("podcast_catalog_requests")
        .select("*", count="exact")
        .eq("status", "pending")
        .execute()
    )

    # Count users (need to query auth.users via Supabase admin)
    # For now, we'll get active subscriptions as a proxy
    active_users = (
        supabase.table("user_subscriptions")
        .select("user_id")
        .eq("is_active", True)
        .execute()
    )
    unique_users = len(set(sub["user_id"] for sub in active_users.data))

    return {
        "podcasts": {
            "total": total_podcasts.count,
            "curated": curated_podcasts.count,
            "user_added": total_podcasts.count - curated_podcasts.count,
        },
        "episodes": {
            "total": total_episodes.count,
            "processed": processed_episodes.count,
            "pending": total_episodes.count - processed_episodes.count,
        },
        "catalog_requests": {
            "pending": pending_requests.count,
        },
        "users": {
            "active_subscribers": unique_users,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Rich system-status endpoint (used by the dashboard)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/system-status")
async def get_system_status(
    supabase: Client = Depends(get_supabase),
    admin_user: dict = Depends(get_current_admin_user),
):
    """
    Comprehensive system status for the management dashboard.

    Returns episode pipeline counts by status, digest stats, queue depth,
    worker health, and the last 10 failed episodes.
    """
    from p3.api.queue import get_redis_pool

    # ── Episode pipeline counts ──────────────────────────────────────────────
    pipeline_statuses = ["pending", "downloaded", "transcribing", "transcribed",
                         "processing", "completed", "failed"]
    episode_counts: dict = {}
    for s in pipeline_statuses:
        r = supabase.table("episodes").select("id", count="exact").eq("status", s).execute()
        episode_counts[s] = r.count or 0

    # ── Digest stats ─────────────────────────────────────────────────────────
    today = datetime.now(timezone.utc).date().isoformat()
    digest_total = supabase.table("user_digests").select("id", count="exact").execute()
    digest_today = (
        supabase.table("user_digests")
        .select("id", count="exact")
        .eq("digest_date", today)
        .execute()
    )
    digest_failed = (
        supabase.table("user_digests")
        .select("id", count="exact")
        .eq("status", "failed")
        .execute()
    )

    # ── Podcast / newsletter counts ──────────────────────────────────────────
    podcast_count = (
        supabase.table("podcasts").select("id", count="exact").eq("feed_type", "podcast").execute()
    )
    newsletter_count = (
        supabase.table("podcasts").select("id", count="exact").eq("feed_type", "newsletter").execute()
    )

    # ── Recent failed episodes ───────────────────────────────────────────────
    recent_failed = (
        supabase.table("episodes")
        .select("id, title, status, created_at, podcasts(title, feed_type)")
        .eq("status", "failed")
        .order("created_at", desc=True)
        .limit(10)
        .execute()
    )

    # ── Redis / queue health ─────────────────────────────────────────────────
    redis_ok = False
    queue_depth = 0
    worker_seen_at: Optional[str] = None
    try:
        redis = await get_redis_pool()
        await redis.ping()
        redis_ok = True
        # arq stores pending jobs in a sorted set keyed "arq:queue"
        queue_depth = await redis.zcard("arq:queue") or 0
        # arq worker heartbeat: "arq:health-check" key or similar
        health_keys = await redis.keys("arq:health*")
        if health_keys:
            key = health_keys[0]
            val = await redis.get(key)
            if val:
                worker_seen_at = val.decode() if isinstance(val, bytes) else str(val)
    except Exception as e:
        pass  # redis_ok stays False

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "redis": {"ok": redis_ok, "queue_depth": queue_depth},
        "worker": {"last_heartbeat": worker_seen_at},
        "episodes": {
            **episode_counts,
            "total": sum(episode_counts.values()),
        },
        "digests": {
            "total": digest_total.count or 0,
            "today": digest_today.count or 0,
            "failed": digest_failed.count or 0,
        },
        "feeds": {
            "podcasts": podcast_count.count or 0,
            "newsletters": newsletter_count.count or 0,
        },
        "recent_failures": recent_failed.data,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Control actions
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/trigger-feed-refresh")
@limiter.limit("3/minute")
async def trigger_feed_refresh(
    request: Request,
    admin_user: dict = Depends(get_current_admin_user),
):
    """
    Manually trigger an immediate RSS feed refresh for all podcasts with active subscribers.
    Same logic as the scheduled 6-hourly cron, but on-demand.
    """
    job_id = await queue.enqueue_job("refresh_all_feeds_job")
    return {"message": "Feed refresh enqueued", "job_id": job_id}


@router.post("/retry-all-failed")
@limiter.limit("3/minute")
async def retry_all_failed_episodes(
    request: Request,
    supabase: Client = Depends(get_supabase),
    admin_user: dict = Depends(get_current_admin_user),
):
    """
    Retry every episode currently in 'failed' status across the entire catalog.
    Resets each to 'pending' and re-enqueues the download step.
    """
    failed = (
        supabase.table("episodes")
        .select("id, audio_url")
        .eq("status", "failed")
        .execute()
    )

    if not failed.data:
        return {"message": "No failed episodes found", "retried": 0}

    episode_ids = [ep["id"] for ep in failed.data]

    supabase.table("episodes").update({"status": "pending"}).in_("id", episode_ids).execute()

    # For newsletter articles (no audio_url) skip to summarize; podcasts start from download
    podcast_ids = [ep["id"] for ep in failed.data if ep.get("audio_url")]
    article_ids = [ep["id"] for ep in failed.data if not ep.get("audio_url")]

    for ep_id in podcast_ids:
        await queue.download_episode(ep_id)
    for ep_id in article_ids:
        await queue.enqueue_job("summarize_episode_job", ep_id)

    return {
        "message": f"Retried {len(episode_ids)} failed items",
        "podcast_episodes": len(podcast_ids),
        "newsletter_articles": len(article_ids),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Management dashboard  (GET /admin/dashboard?token=<JWT>)
# ─────────────────────────────────────────────────────────────────────────────

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Parakeet Admin</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#0f172a;color:#e2e8f0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;font-size:14px}
  a{color:#38bdf8}
  h2{font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:#94a3b8;margin-bottom:12px}
  .shell{max-width:1200px;margin:0 auto;padding:24px 20px}

  /* topbar */
  .topbar{display:flex;align-items:center;justify-content:space-between;
          padding:14px 20px;background:#1e293b;border-bottom:1px solid #334155;
          position:sticky;top:0;z-index:10}
  .logo{font-size:18px;font-weight:700;color:#f8fafc}
  .logo span{color:#38bdf8}
  .refresh-info{display:flex;align-items:center;gap:10px;font-size:12px;color:#64748b}
  .dot{width:8px;height:8px;border-radius:50%;display:inline-block}
  .dot-green{background:#22c55e} .dot-red{background:#ef4444} .dot-yellow{background:#f59e0b}

  /* status bar */
  .statusbar{display:flex;gap:16px;padding:10px 20px;background:#162032;
             border-bottom:1px solid #1e293b;font-size:12px;flex-wrap:wrap}
  .sb-item{display:flex;align-items:center;gap:6px}

  /* grid layout */
  .grid{display:grid;gap:16px;margin-top:20px}
  .grid-2{grid-template-columns:1fr 1fr}
  .grid-3{grid-template-columns:1fr 1fr 1fr}
  @media(max-width:768px){.grid-2,.grid-3{grid-template-columns:1fr}}

  /* cards */
  .card{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:20px}

  /* pipeline bar */
  .pipeline{display:flex;gap:3px;margin-top:4px}
  .stage{flex:1;text-align:center;padding:10px 4px;border-radius:6px;font-size:11px}
  .stage .num{font-size:22px;font-weight:700;display:block;margin-bottom:2px}
  .stage-pending{background:#1e3a5f;color:#60a5fa}
  .stage-downloaded{background:#1e3f2f;color:#4ade80}
  .stage-transcribing{background:#3b2800;color:#fb923c}
  .stage-transcribed{background:#2e1a4a;color:#a78bfa}
  .stage-processing{background:#4a2800;color:#fbbf24}
  .stage-completed{background:#14532d;color:#86efac}
  .stage-failed{background:#3b0a0a;color:#f87171}

  /* stat numbers */
  .stat-row{display:flex;justify-content:space-between;align-items:baseline;
            padding:6px 0;border-bottom:1px solid #334155}
  .stat-row:last-child{border-bottom:none}
  .stat-label{color:#94a3b8}
  .stat-val{font-weight:600;font-size:16px}

  /* actions */
  .btn{display:inline-flex;align-items:center;gap:6px;padding:8px 16px;
       border-radius:6px;border:none;font-size:13px;font-weight:500;cursor:pointer;
       transition:opacity .15s}
  .btn:hover{opacity:.85}
  .btn:disabled{opacity:.4;cursor:not-allowed}
  .btn-blue{background:#1d4ed8;color:#fff}
  .btn-amber{background:#92400e;color:#fef3c7}
  .btn-green{background:#166534;color:#dcfce7}
  .btn-red{background:#7f1d1d;color:#fee2e2}
  .actions{display:flex;flex-wrap:wrap;gap:10px}

  /* table */
  table{width:100%;border-collapse:collapse;font-size:12px}
  th{text-align:left;color:#64748b;font-weight:500;padding:6px 8px;
     border-bottom:1px solid #334155;text-transform:uppercase;font-size:11px}
  td{padding:7px 8px;border-bottom:1px solid #1e293b;vertical-align:middle}
  tr:last-child td{border-bottom:none}
  .badge{padding:2px 8px;border-radius:4px;font-size:11px;font-weight:500}
  .b-podcast{background:#14532d;color:#86efac}
  .b-newsletter{background:#1e3a5f;color:#60a5fa}
  .b-failed{background:#3b0a0a;color:#f87171}

  /* toast */
  #toast{position:fixed;bottom:24px;right:24px;background:#22c55e;color:#fff;
         padding:10px 18px;border-radius:6px;font-size:13px;font-weight:500;
         opacity:0;transition:opacity .3s;pointer-events:none}
  #toast.show{opacity:1}
  #toast.err{background:#ef4444}

  .spinner{display:inline-block;width:12px;height:12px;border:2px solid #334155;
           border-top-color:#38bdf8;border-radius:50%;animation:spin .7s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}

  .muted{color:#64748b;font-size:12px}
  .ts{color:#475569;font-size:11px;font-family:monospace}
</style>
</head>
<body>

<div class="topbar">
  <div class="logo">🦜 <span>Parakeet</span> Admin</div>
  <div class="refresh-info">
    <span id="refresh-cd">Refreshing in <b id="cd">30</b>s</span>
    <span id="spinner" class="spinner" style="display:none"></span>
    <button class="btn btn-blue" style="padding:5px 12px;font-size:12px" onclick="load()">↺ Now</button>
  </div>
</div>

<div class="statusbar" id="statusbar">
  <span class="sb-item"><span class="dot dot-yellow"></span> Loading…</span>
</div>

<div class="shell">
  <div id="main-content">
    <div style="text-align:center;padding:60px;color:#475569">Loading dashboard…</div>
  </div>
</div>

<div id="toast"></div>

<script>
const TOKEN = new URLSearchParams(location.search).get('token') || '';
const API   = (path) => fetch(path, {headers: TOKEN ? {Authorization:'Bearer '+TOKEN} : {}});
let countdown = 30;
let timer;

function toast(msg, err=false) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'show' + (err ? ' err' : '');
  setTimeout(() => t.className = '', 3000);
}

async function action(path, label) {
  try {
    toast('Running: ' + label + '…');
    const r = await API(path);
    const d = await r.json();
    toast(d.message || label + ' done');
    setTimeout(load, 800);
  } catch(e) {
    toast('Failed: ' + e.message, true);
  }
}

async function retryEpisode(id) {
  await action('/admin/episodes/'+id+'/retry', 'Retry episode');
}

function pct(n, total) {
  if (!total) return '0%';
  return Math.round(n/total*100) + '%';
}

function fmt(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
}

async function load() {
  document.getElementById('spinner').style.display = '';
  clearInterval(timer);
  countdown = 30;
  document.getElementById('cd').textContent = countdown;

  try {
    const r = await API('/admin/system-status');
    if (r.status === 401 || r.status === 403) {
      document.getElementById('main-content').innerHTML =
        '<div style="text-align:center;padding:60px;color:#f87171">' +
        'Access denied. Add <code>?token=YOUR_JWT</code> to the URL.</div>';
      return;
    }
    const d = await r.json();
    render(d);
  } catch(e) {
    document.getElementById('main-content').innerHTML =
      '<div style="text-align:center;padding:60px;color:#f87171">Failed to load: ' + e.message + '</div>';
  }

  document.getElementById('spinner').style.display = 'none';

  timer = setInterval(() => {
    countdown--;
    document.getElementById('cd').textContent = countdown;
    if (countdown <= 0) load();
  }, 1000);
}

function render(d) {
  // status bar
  const redis = d.redis;
  const qd = redis.queue_depth;
  const hb = d.worker.last_heartbeat;
  document.getElementById('statusbar').innerHTML = `
    <span class="sb-item">
      <span class="dot ${redis.ok ? 'dot-green' : 'dot-red'}"></span>
      Redis ${redis.ok ? 'connected' : 'unreachable'}
    </span>
    <span class="sb-item">
      <span class="dot ${qd > 0 ? 'dot-yellow' : 'dot-green'}"></span>
      ${qd} job${qd !== 1 ? 's' : ''} queued
    </span>
    <span class="sb-item">
      <span class="dot ${hb ? 'dot-green' : 'dot-red'}"></span>
      Worker: ${hb || 'no heartbeat'}
    </span>
    <span class="sb-item muted ts">${new Date(d.timestamp).toLocaleTimeString()}</span>
  `;

  const ep = d.episodes;
  const totalEp = ep.total;
  const di = d.digests;
  const fe = d.feeds;

  document.getElementById('main-content').innerHTML = `

  <!-- Episode Pipeline -->
  <h2 style="margin-top:0">Episode Pipeline</h2>
  <div class="card" style="padding:16px">
    <div class="pipeline">
      ${[
        ['pending',     ep.pending,     'stage-pending'],
        ['downloaded',  ep.downloaded,  'stage-downloaded'],
        ['transcribing',ep.transcribing,'stage-transcribing'],
        ['transcribed', ep.transcribed, 'stage-transcribed'],
        ['processing',  ep.processing,  'stage-processing'],
        ['completed',   ep.completed,   'stage-completed'],
        ['failed',      ep.failed,      'stage-failed'],
      ].map(([lbl,n,cls]) =>
        `<div class="stage ${cls}"><span class="num">${n.toLocaleString()}</span>${lbl}</div>`
      ).join('')}
    </div>
    <div class="muted" style="margin-top:10px;text-align:right">
      Total: ${totalEp.toLocaleString()} &nbsp;·&nbsp;
      Processed: ${pct(ep.completed, totalEp)}
    </div>
  </div>

  <div class="grid grid-3" style="margin-top:16px">

    <!-- Digests -->
    <div class="card">
      <h2>Digests</h2>
      <div class="stat-row"><span class="stat-label">Total</span><span class="stat-val">${di.total.toLocaleString()}</span></div>
      <div class="stat-row"><span class="stat-label">Generated today</span><span class="stat-val">${di.today}</span></div>
      <div class="stat-row"><span class="stat-label" style="color:#f87171">Failed</span>
        <span class="stat-val" style="color:${di.failed > 0 ? '#f87171' : 'inherit'}">${di.failed}</span></div>
    </div>

    <!-- Feeds -->
    <div class="card">
      <h2>Feeds</h2>
      <div class="stat-row"><span class="stat-label">Podcasts</span><span class="stat-val">${fe.podcasts}</span></div>
      <div class="stat-row"><span class="stat-label">Newsletters</span><span class="stat-val">${fe.newsletters}</span></div>
      <div class="stat-row"><span class="stat-label">Total</span>
        <span class="stat-val">${fe.podcasts + fe.newsletters}</span></div>
    </div>

    <!-- Episodes quick stats -->
    <div class="card">
      <h2>In-Progress</h2>
      <div class="stat-row"><span class="stat-label">Downloading</span>
        <span class="stat-val">${ep.downloaded}</span></div>
      <div class="stat-row"><span class="stat-label">Transcribing</span>
        <span class="stat-val">${ep.transcribing + ep.transcribed}</span></div>
      <div class="stat-row"><span class="stat-label">Summarizing</span>
        <span class="stat-val">${ep.processing}</span></div>
    </div>
  </div>

  <!-- Actions -->
  <div class="card" style="margin-top:16px">
    <h2>Quick Actions</h2>
    <div class="actions">
      <button class="btn btn-blue"
        onclick="action('/admin/trigger-feed-refresh','Refresh feeds')">
        ↺ Refresh All Feeds
      </button>
      <button class="btn btn-amber" ${ep.failed === 0 ? 'disabled' : ''}
        onclick="action('/admin/retry-all-failed','Retry all failed')">
        ⚡ Retry All Failed (${ep.failed})
      </button>
      <button class="btn btn-green"
        onclick="action('/admin/trigger-feed-refresh','Refresh feeds')">
        📡 Force Podcast Fetch
      </button>
    </div>
  </div>

  <!-- Recent failures -->
  ${d.recent_failures.length > 0 ? `
  <div class="card" style="margin-top:16px">
    <h2>Recent Failures</h2>
    <table>
      <thead><tr>
        <th>Title</th><th>Source</th><th>Type</th><th>Created</th><th>Action</th>
      </tr></thead>
      <tbody>
        ${d.recent_failures.map(ep => {
          const pod = ep.podcasts || {};
          const ft = pod.feed_type || 'podcast';
          return `<tr>
            <td>${ep.title || '—'}</td>
            <td>${pod.title || '—'}</td>
            <td><span class="badge b-${ft}">${ft}</span></td>
            <td class="ts">${fmt(ep.created_at)}</td>
            <td>
              <button class="btn btn-amber" style="padding:3px 10px;font-size:11px"
                onclick="retryEpisode('${ep.id}')">Retry</button>
            </td>
          </tr>`;
        }).join('')}
      </tbody>
    </table>
  </div>` : ''}

  `;
}

// boot
load();
</script>
</body>
</html>"""


@router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def get_dashboard(token: Optional[str] = None):
    """
    Management dashboard.
    Accepts the JWT via ?token= query param for browser access.
    Authentication is enforced client-side (all API calls use the token).
    """
    return HTMLResponse(content=_DASHBOARD_HTML)

