"""Command-line interface for P³."""

import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
import click
import yaml
from rich.console import Console
from rich.table import Table
from rich.progress import track

from .database import P3Database
from .downloader import PodcastDownloader
from .transcriber import AudioTranscriber
from .cleaner import TranscriptCleaner
from .exporter import DigestExporter
from .writer import BlogWriter
from .emailer import DigestEmailer
from .thread_generator import ThreadGenerator
from .model_manager import ModelManager
from .thesis_generator import ThesisGenerator
from .storage_manager import StorageManager
from .newsletter_fetcher import SubstackFetcher
from .newsletter_summarizer import NewsletterSummarizer
from .audio_digest_generator import AudioDigestGenerator
from .dependency_checker import DependencyChecker
from .llm_client import LLMClient
from .system_doctor import SystemDoctor
from .windows_utils import WindowsHelper

console = Console()

# Apply Windows-specific optimizations at module load
WindowsHelper.setup_windows_environment()


def load_config(config_path: str = "config/feeds.yaml"):
    """Load configuration from YAML file."""
    config_file = Path(config_path)

    if not config_file.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        console.print("Copy config/feeds.yaml.example to config/feeds.yaml and configure your feeds")
        sys.exit(1)

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        sys.exit(1)


def prepare_transcription_config(settings: dict) -> dict:
    """Prepare transcription configuration from settings, handling external APIs.

    Respects compute_mode setting:
    - "local": Use local faster-whisper/whisper
    - "api" or "hybrid": Use transcription API (Groq/OpenAI)

    Args:
        settings: Configuration settings dict from feeds.yaml

    Returns:
        Dict with keys: use_api, api_provider, api_model, api_key_file, api_base_url
    """
    config = {}

    # Check compute_mode first
    compute_mode = settings.get('compute_mode', 'local')

    # Check if transcription API is enabled (explicit or via compute_mode)
    transcription_api = settings.get('transcription_api', {})
    use_api = transcription_api.get('enabled', False)

    # Override based on compute_mode
    if compute_mode in ['api', 'hybrid']:
        use_api = True

    if use_api:
        # Use API transcription
        config['use_api'] = True
        config['api_provider'] = transcription_api.get('provider', 'groq')
        config['api_model'] = transcription_api.get('model', 'whisper-large-v3')
        config['api_fallback_models'] = transcription_api.get('fallback_models')
        config['api_fallback_model'] = transcription_api.get('fallback_model')
        config['api_key_file'] = transcription_api.get('api_key_file')
        config['api_base_url'] = transcription_api.get('base_url')
        config['wait_for_ratelimit'] = transcription_api.get('wait_for_ratelimit', False)
        config['max_wait_retries'] = transcription_api.get('max_wait_retries', 3)

        console.print(f"[yellow]⚙ Using {config['api_provider']} API for transcription (compute_mode: {compute_mode})[/yellow]")
    else:
        # Use local transcription
        config['use_api'] = False
        config['api_provider'] = None
        config['api_model'] = None
        config['api_key_file'] = None
        config['api_base_url'] = None
        config['wait_for_ratelimit'] = False
        config['max_wait_retries'] = 0

    return config


def prepare_tts_config(settings: dict) -> dict:
    """Prepare TTS configuration from settings, handling external APIs.

    Respects compute_mode setting:
    - "local": Use local Piper TTS
    - "api": Use TTS API (Groq/OpenAI)
    - "hybrid": Use local Piper TTS (heavy compute is API, TTS is local)

    Args:
        settings: Configuration settings dict from feeds.yaml

    Returns:
        Dict with keys: use_api, api_provider, api_model, voice, api_key_file
    """
    config = {}

    # Check compute_mode first
    compute_mode = settings.get('compute_mode', 'local')

    # Check if TTS API is enabled (explicit or via compute_mode)
    tts_api = settings.get('tts_api', {})
    use_api = tts_api.get('enabled', False)

    # Override based on compute_mode (only "api" mode enables TTS API, hybrid uses local)
    if compute_mode == 'api':
        use_api = True

    if use_api:
        # Use API TTS
        config['use_api'] = True
        config['api_provider'] = tts_api.get('provider', 'groq')
        config['api_model'] = tts_api.get('model', 'playai-tts')
        config['voice'] = tts_api.get('voice', 'Arista-PlayAI')
        config['api_key_file'] = tts_api.get('api_key_file')

        console.print(f"[yellow]⚙ Using {config['api_provider']} API for TTS (compute_mode: {compute_mode})[/yellow]")
    else:
        # Use local Piper TTS
        config['use_api'] = False
        config['api_provider'] = None
        config['api_model'] = None
        config['voice'] = None  # Will use default from narration config
        config['api_key_file'] = None

    return config


def prepare_llm_config(settings: dict, llm_model: str = None) -> dict:
    """Prepare LLM configuration from settings, handling compute_mode and external APIs.

    Determines whether to use local Ollama or cloud APIs (Groq, OpenAI, etc.)
    based on compute_mode setting or explicit external_apis config.

    Args:
        settings: Configuration settings dict from feeds.yaml
        llm_model: Optional model override from command line

    Returns:
        Dict with keys: provider, model, api_key_file, ollama_base_url, ollama_model, base_url, timeout, max_retries
    """
    config = {}

    # Get local Ollama model for fallback (always needed)
    local_ollama_model = settings.get('llm_model', 'llama3.2:3b')

    # If configured model isn't installed, prefer an installed local model
    try:
        installed_models = ModelManager.get_installed_ollama_models()
        if installed_models and not ModelManager.is_model_installed(local_ollama_model):
            # Prefer a recommended installed model if possible
            fallback_model = next(
                (name for name in ModelManager.RECOMMENDED_MODELS.keys()
                 if ModelManager.is_model_installed(name)),
                installed_models[0]
            )
            console.print(
                f"[yellow]⚠ Ollama model '{local_ollama_model}' not found. "
                f"Using installed model '{fallback_model}' for local fallback.[/yellow]"
            )
            local_ollama_model = fallback_model
    except Exception as e:
        console.print(f"[yellow]⚠ Could not verify Ollama models: {e}[/yellow]")

    # Check compute_mode first
    compute_mode = settings.get('compute_mode', 'local')

    # Check if external APIs are explicitly enabled
    external_apis = settings.get('external_apis', {})
    use_external = external_apis.get('enabled', False)

    # Determine provider based on compute_mode or external_apis
    if compute_mode in ['api', 'hybrid'] or use_external:
        # Use external API for LLM
        config['provider'] = external_apis.get('provider', 'groq')
        # Use command line model if provided, otherwise use config
        config['model'] = llm_model or external_apis.get('model', 'llama-3.1-8b-instant')
        config['api_key_file'] = external_apis.get('api_key_file')
        config['base_url'] = external_apis.get('base_url')
        config['ollama_base_url'] = settings.get('ollama_base_url', 'http://localhost:11434')
        config['ollama_model'] = local_ollama_model  # For fallback when API fails
        config['timeout'] = external_apis.get('timeout', 120)
        config['max_retries'] = external_apis.get('max_retries', 3)
        config['model_fallbacks'] = external_apis.get('model_fallbacks')
        if config['provider'] == 'groq' and not config['model_fallbacks']:
            # Ordered by context window size (largest first) to minimize wasted API calls
            # scout: 30K TPM, versatile: 12K TPM, 8b-instant: 6K TPM
            config['model_fallbacks'] = [
                "meta-llama/llama-4-scout-17b-16e-instruct",
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant"
            ]
            if config['model'] not in config['model_fallbacks']:
                config['model_fallbacks'].insert(0, config['model'])
        elif config['model_fallbacks'] and config['model'] not in config['model_fallbacks']:
            config['model_fallbacks'].insert(0, config['model'])
        console.print(f"[yellow]⚙ Using {config['provider']} API for LLM (compute_mode: {compute_mode})[/yellow]")
    else:
        # Use local Ollama
        config['provider'] = 'ollama'
        # Use command line model if provided, otherwise use config
        config['model'] = llm_model or local_ollama_model
        config['api_key_file'] = None
        config['base_url'] = None
        config['ollama_base_url'] = settings.get('ollama_base_url', 'http://localhost:11434')
        config['ollama_model'] = config['model']  # Same as model when using local Ollama
        config['timeout'] = settings.get('llm_timeout', 120)
        config['max_retries'] = settings.get('llm_max_retries', 3)
        config['model_fallbacks'] = None

    return config


def _filter_relevant_summaries(summaries, topic, llm_client=None):
    """Filter summaries to only those relevant to the given topic.

    Uses keyword matching against stored topics, themes, and company names.
    No LLM call needed - scales to thousands of episodes.
    Returns a subset of summaries ordered by relevance score.
    """
    if len(summaries) <= 3:
        return summaries

    topic_lower = topic.lower()
    # Split topic into keywords, keep meaningful words (>1 char to preserve AI, ML, VC, etc.)
    keywords = [w.strip() for w in topic_lower.replace(',', ' ').replace('-', ' ').split() if len(w.strip()) > 1]

    if not keywords:
        # No usable keywords extracted from topic - can't filter, return all
        return summaries

    scored = []
    for s in summaries:
        score = 0
        # Check stored structured fields (weighted higher - curated by LLM during digest)
        ep_topics = [t.lower() for t in s.get('key_topics', [])]
        ep_themes = [t.lower() for t in s.get('themes', [])]
        ep_companies = [t.lower() for t in s.get('startups', [])]
        ep_title = s.get('episode_title', '').lower()

        for kw in keywords:
            # Exact topic/theme match (high weight)
            if any(kw in t for t in ep_topics):
                score += 3
            if any(kw in t for t in ep_themes):
                score += 2
            # Company match
            if any(kw in c for c in ep_companies):
                score += 2
            # Title match
            if kw in ep_title:
                score += 1
            # Summary text match (lowest weight)
            if kw in (s.get('full_summary', '') or '').lower():
                score += 1

        scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    # Keep episodes with at least 1 match
    relevant = [s for score, s in scored if score > 0]
    return relevant


def _display_sources(summaries):
    """Display source podcasts/newsletters grouped by show name."""
    console.print(f"\n[cyan]Sources ({len(summaries)}):[/cyan]")
    seen_podcasts = {}
    for s in summaries:
        podcast = s.get('podcast_title', 'Unknown Podcast')
        if podcast not in seen_podcasts:
            seen_podcasts[podcast] = []
        seen_podcasts[podcast].append(s.get('episode_title', 'Unknown Episode'))
    for podcast, episodes in seen_podcasts.items():
        console.print(f"  [bold]{podcast}[/bold]")
        for ep in episodes:
            console.print(f"    - {ep}")


def _refine_loop(llm_client, original_prompt, original_result, result_key='text',
                  max_iterations=10):
    """Interactive refinement loop for content generation.

    Args:
        llm_client: LLMClient instance
        original_prompt: The original generation prompt
        original_result: The generated content text
        result_key: Label for the content type (e.g., 'blog post', 'tweet')
        max_iterations: Maximum refinement rounds

    Returns:
        The final refined content string, or original if user exits immediately
    """
    current_content = original_result
    iteration = 0

    while iteration < max_iterations:
        console.print(f"\n[cyan]Refine mode:[/cyan] Enter feedback to improve the {result_key}, or type 'done' to finish.")
        try:
            feedback = click.prompt("Feedback", default="done", show_default=True)
        except (click.Abort, EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Refinement ended.[/yellow]")
            break

        feedback = feedback.strip()
        if not feedback or feedback.lower() == 'done':
            console.print("[green]Refinement complete.[/green]")
            break

        iteration += 1
        refine_prompt = f"""Here is the current {result_key}:

{current_content}

The author wants the following changes:
{feedback}

Please rewrite the {result_key} incorporating this feedback. Return ONLY the revised {result_key}, nothing else."""

        with console.status(f"[bold green]Refining {result_key} (round {iteration})..."):
            refined = llm_client.chat(
                messages=[
                    {"role": "system", "content": f"You are refining a {result_key} based on author feedback. Return only the revised content."},
                    {"role": "user", "content": refine_prompt}
                ],
                temperature=0.4,
                max_tokens=4000
            )

        if refined:
            current_content = refined.strip()
            console.print(f"\n[green]Refined {result_key} (round {iteration}):[/green]")
            console.print("-" * 50)
            preview = current_content[:500]
            if len(current_content) > 500:
                preview += "..."
            console.print(preview)
            console.print("-" * 50)
        else:
            console.print("[red]Refinement failed, keeping previous version.[/red]")

    return current_content


def _record_source_usage(summaries, content_type='unknown'):
    """Record which podcasts/newsletters were used as sources.

    Appends usage data to data/.source_usage.json for tracking
    which sources contribute most to generated content.
    """
    import json as _json
    usage_file = Path("data/.source_usage.json")

    # Load existing data
    usage_data = []
    if usage_file.exists():
        try:
            usage_data = _json.loads(usage_file.read_text(encoding='utf-8'))
        except (_json.JSONDecodeError, ValueError):
            usage_data = []

    # Record this usage event
    sources_used = []
    for s in summaries:
        sources_used.append({
            'podcast': s.get('podcast_title', 'Unknown'),
            'episode': s.get('episode_title', 'Unknown'),
            'source_type': s.get('source_type', 'podcast')
        })

    usage_data.append({
        'timestamp': datetime.now().isoformat(),
        'content_type': content_type,
        'source_count': len(sources_used),
        'sources': sources_used
    })

    # Keep last 200 entries to avoid unbounded growth
    if len(usage_data) > 200:
        usage_data = usage_data[-200:]

    usage_file.parent.mkdir(parents=True, exist_ok=True)
    usage_file.write_text(_json.dumps(usage_data, indent=2), encoding='utf-8')


def _get_source_stats():
    """Get podcast usage stats from source tracking data.

    Returns dict of source_name -> usage count, sorted by count descending.
    """
    import json as _json
    usage_file = Path("data/.source_usage.json")

    if not usage_file.exists():
        return {}

    try:
        usage_data = _json.loads(usage_file.read_text(encoding='utf-8'))
    except (_json.JSONDecodeError, ValueError):
        return {}

    from collections import Counter
    source_counts = Counter()
    for entry in usage_data:
        for src in entry.get('sources', []):
            source_counts[src.get('podcast', 'Unknown')] += 1

    return dict(source_counts.most_common())


def _get_unified_summaries(db, lookback_days):
    """Get both podcast and newsletter summaries merged into one list.

    Newsletter summaries are returned with fields compatible with podcast summaries
    (podcast_title prefixed with [Newsletter], episode_title = article title).
    """
    podcast_summaries = db.get_recent_summaries(lookback_days)
    try:
        newsletter_summaries = db.get_recent_newsletter_summaries(lookback_days)
    except Exception:
        newsletter_summaries = []

    combined = list(podcast_summaries)
    combined.extend(newsletter_summaries)
    return combined


def _detect_convergence(summaries):
    """Detect topics/companies mentioned by 3+ independent sources.

    Returns dict of topic -> list of unique source names that mention it.
    Only returns entries with 3+ unique sources (strong signals).
    """
    from collections import defaultdict
    topic_sources = defaultdict(set)

    for s in summaries:
        source_name = s.get('podcast_title', 'Unknown')
        for topic in s.get('key_topics', []):
            topic_sources[topic.lower().strip()].add(source_name)
        for company in s.get('startups', []):
            topic_sources[company.lower().strip()].add(source_name)
        for theme in s.get('themes', []):
            topic_sources[theme.lower().strip()].add(source_name)

    # Filter to 3+ unique sources
    strong_signals = {
        topic: sorted(sources)
        for topic, sources in topic_sources.items()
        if len(sources) >= 3 and topic  # skip empty strings
    }
    return dict(sorted(strong_signals.items(), key=lambda x: len(x[1]), reverse=True))


@click.group()
@click.option('--config', default="config/feeds.yaml", help='Configuration file path')
@click.option('--db', default="data/p3.duckdb", help='Database file path')
@click.pass_context
def main(ctx, config, db):
    """Parakeet Podcast Processor (P³) - Automated podcast processing.

    \b
    📚 COMMAND GROUPS:

    🎯 Core Pipeline:
      fetch        - Download new episodes from RSS feeds
      transcribe   - Convert audio to text
      digest       - Generate structured summaries
      export       - Export digests to markdown/JSON
      audio-digest - Generate condensed audio from best clips (1/10th length)
      auto         - Run complete pipeline (fetch→transcribe→digest→export→audio→cleanup)

    ✍️  Content Creation:
      write        - Generate blog posts from summaries
      tweet        - Generate tweets or Twitter threads
      thesis       - Generate company/sector analyses for interviews
      ask          - Ask questions about your podcasts (AI-powered search)

    📰 Newsletter Integration:
      newsletter   - Manage Substack newsletter tracking and summaries
                     (add, list, fetch, digest)

    🛠️  Utilities:
      status       - Show processing pipeline status
      doctor       - Comprehensive system health check
      test         - Quick pipeline validation with test sample
      check-deps   - Check system dependencies
      models       - Show available models and defaults
      cleanup      - Clean up old audio files
      init         - Initialize P³ directories and database

    \b
    💡 QUICK START:
      p3 auto                          Run full automated pipeline
      p3 thesis --list-companies       See companies in your data
      p3 models                        Check model configuration
      p3 status                        Check pipeline status

    \b
    📖 For detailed help on any command:
      p3 <command> --help

    \b
    🔗 Documentation:
      README.md          - Overview and setup
      COMMANDS.md        - Complete command reference
      AUTOMATION_GUIDE.md - Windows scheduling setup
    """
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['db_path'] = db
    ctx.obj['db'] = P3Database(db)


@main.command()
@click.option('--max-episodes', default=None, type=int, help='Max episodes per feed')
@click.pass_context
def fetch(ctx, max_episodes):
    """Download new podcast episodes from configured RSS feeds."""
    config = load_config(ctx.obj['config_path'])
    db = ctx.obj['db']
    
    settings = config.get('settings', {})
    max_eps = max_episodes or settings.get('max_episodes_per_feed', 10)
    
    downloader = PodcastDownloader(
        db=db,
        max_episodes=max_eps,
        audio_format=settings.get('audio_format', 'wav')
    )
    
    console.print("[blue]Fetching podcast episodes...[/blue]")
    
    feeds = config.get('feeds', [])
    if not feeds:
        console.print("[yellow]No feeds configured[/yellow]")
        return
    
    total_downloaded = 0
    results = downloader.fetch_all_feeds(feeds)
    
    # Display results table
    table = Table(title="Download Results")
    table.add_column("Podcast", style="cyan")
    table.add_column("New Episodes", style="green", justify="right")
    
    for name, count in results.items():
        table.add_row(name, str(count))
        total_downloaded += count
    
    console.print(table)
    console.print(f"[green]Total downloaded: {total_downloaded} episodes[/green]")


@main.command()
@click.option('--model', default=None, help='Whisper model to use')
@click.option('--episode-id', type=int, help='Transcribe specific episode')
@click.option('--faster-whisper', is_flag=True, help='Use faster-whisper (4x faster, Windows-optimized)')
@click.pass_context
def transcribe(ctx, model, episode_id, faster_whisper):
    """Transcribe downloaded audio files."""
    config = load_config(ctx.obj['config_path'])
    db = ctx.obj['db']

    settings = config.get('settings', {})
    whisper_model = model or settings.get('whisper_model', 'base')
    use_parakeet = settings.get('parakeet_enabled', False)
    use_faster_whisper = faster_whisper or settings.get('use_faster_whisper', False)

    # Prepare transcription API configuration
    trans_config = prepare_transcription_config(settings)

    # Display model being used
    console.print(f"[blue]Transcription Settings:[/blue]")
    if trans_config['use_api']:
        console.print(f"  Provider: {trans_config['api_provider']} API")
        console.print(f"  Model: {trans_config['api_model']}")
    else:
        console.print(f"  Whisper Model: {whisper_model}")
        console.print(f"  Faster-Whisper: {use_faster_whisper}")

    transcriber = AudioTranscriber(
        db=db,
        whisper_model=whisper_model,
        use_parakeet=use_parakeet,
        parakeet_model=settings.get('parakeet_model', 'mlx-community/parakeet-tdt-0.6b-v2'),
        use_faster_whisper=use_faster_whisper,
        use_api=trans_config['use_api'],
        api_provider=trans_config['api_provider'],
        api_model=trans_config['api_model'],
        api_fallback_model=trans_config.get('api_fallback_model'),
        api_fallback_models=trans_config.get('api_fallback_models'),
        api_key_file=trans_config['api_key_file'],
        api_base_url=trans_config['api_base_url'],
        wait_for_ratelimit=trans_config['wait_for_ratelimit'],
        max_wait_retries=trans_config['max_wait_retries']
    )

    if episode_id:
        console.print(f"\n[blue]Transcribing episode {episode_id}...[/blue]")
        success = transcriber.transcribe_episode(episode_id)
        if success:
            console.print(f"[green]✓ Episode {episode_id} transcribed[/green]")
        else:
            console.print(f"[red]✗ Failed to transcribe episode {episode_id}[/red]")
    else:
        console.print(f"\n[blue]Transcribing all pending episodes...[/blue]")
        episodes = db.get_episodes_by_status('downloaded')

        if not episodes:
            console.print("[yellow]No episodes to transcribe[/yellow]")
            return

        transcribed = 0
        for episode in track(episodes, description="Transcribing..."):
            if transcriber.transcribe_episode(episode['id']):
                transcribed += 1

        console.print(f"[green]Transcribed {transcribed} episodes[/green]")


@main.command()
@click.option('--provider', default=None, help='LLM provider (openai, anthropic, ollama)')
@click.option('--model', default=None, help='LLM model to use')
@click.option('--episode-id', type=int, help='Process specific episode')
@click.pass_context
def digest(ctx, provider, model, episode_id):
    """Generate structured summaries from transcripts."""
    config = load_config(ctx.obj['config_path'])
    db = ctx.obj['db']

    settings = config.get('settings', {})

    # Prepare LLM configuration (handles external APIs if enabled)
    llm_config = prepare_llm_config(settings, model)

    # Allow provider override from CLI
    if provider:
        llm_config['provider'] = provider

    # Display model being used
    console.print(f"[blue]LLM Settings:[/blue]")
    console.print(f"  Provider: {llm_config['provider']}")
    console.print(f"  Model: {llm_config['model']}")

    # Auto-download Ollama model if needed
    if llm_config['provider'] == 'ollama':
        if not ModelManager.ensure_model_available(llm_config['model'], console):
            console.print(f"[red]✗ Failed to ensure model {llm_config['model']} is available[/red]")
            return

    cleaner = TranscriptCleaner(
        db=db,
        llm_provider=llm_config['provider'],
        llm_model=llm_config['model'],
        ollama_base_url=llm_config['ollama_base_url'],
        ollama_model=llm_config['ollama_model'],
        model_fallbacks=llm_config.get('model_fallbacks'),
        api_key_file=llm_config['api_key_file'],
        base_url=llm_config['base_url'],
        timeout=llm_config['timeout'],
        max_retries=llm_config['max_retries']
    )

    if episode_id:
        console.print(f"\n[blue]Processing episode {episode_id}...[/blue]")
        result = cleaner.generate_summary(episode_id)
        if result:
            console.print(f"[green]✓ Episode {episode_id} processed[/green]")
        else:
            console.print(f"[red]✗ Failed to process episode {episode_id}[/red]")
    else:
        console.print(f"\n[blue]Processing all transcribed episodes...[/blue]")
        processed = cleaner.process_all_transcribed()
        console.print(f"[green]Processed {processed} episodes[/green]")


@main.command()
@click.option('--date', help='Export date (YYYY-MM-DD)')
@click.option('--format', multiple=True, help='Export format (markdown, json)')
@click.option('--output', help='Output file path')
@click.option('--include-newsletters/--no-newsletters', default=True, help='Include newsletters in export (default: True)')
@click.pass_context
def export(ctx, date, format, output, include_newsletters):
    """Export daily digest summaries."""
    config = load_config(ctx.obj['config_path'])
    db = ctx.obj['db']

    # Parse date
    if date:
        try:
            target_date = datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            console.print("[red]Invalid date format. Use YYYY-MM-DD[/red]")
            return
    else:
        target_date = datetime.now()

    # Get export formats
    formats = list(format) if format else config.get('settings', {}).get('export_format', ['markdown'])

    exporter = DigestExporter(db)

    summaries = db.get_summaries_by_date(target_date)

    # Check for newsletters
    newsletter_summaries = db.get_newsletter_summaries_by_date(target_date) if include_newsletters else []

    if not summaries and not newsletter_summaries:
        console.print(f"[yellow]📭 No content found for {target_date.date()}[/yellow]")
        console.print("\n[cyan]💡 No summaries available to export.[/cyan]")
        console.print("  • Run: [yellow]p3 digest[/yellow] to process transcribed episodes")
        console.print("  • Or run: [yellow]p3 auto[/yellow] to fetch and process new content")
        console.print("  • Check status: [yellow]p3 status[/yellow]")
        return

    console.print(f"[blue]Exporting {len(summaries)} podcast summaries + {len(newsletter_summaries)} newsletter summaries for {target_date.date()}[/blue]")

    # Ensure output directory exists
    output_dir = Path("outputs/digests")
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        if fmt == 'markdown':
            content = exporter.export_markdown(summaries, target_date.date(), include_newsletters)
            filename = output or str(output_dir / f"digest_{target_date.strftime('%Y-%m-%d')}.md")
        elif fmt == 'json':
            content = exporter.export_json(summaries, target_date.date(), include_newsletters)
            filename = output or str(output_dir / f"digest_{target_date.strftime('%Y-%m-%d')}.json")
        else:
            console.print(f"[red]Unsupported format: {fmt}[/red]")
            continue

        # Write to file with UTF-8 encoding (Windows compatibility)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)

        console.print(f"[green]✓ Exported {fmt}: {filename}[/green]")


@main.command(name='export-transcripts')
@click.option('--lookback-days', default=7, type=int, help='Export transcripts from the last N days (default: 7)')
@click.option('--episode-id', type=int, help='Export transcript for a specific episode')
@click.option('--output-dir', default='outputs/transcripts', help='Output directory')
@click.pass_context
def export_transcripts(ctx, lookback_days, episode_id, output_dir):
    """Export readable transcripts as markdown files.

    Creates human-readable transcript files with timestamps,
    speaker context, and episode metadata.

    \b
    Examples:
      p3 export-transcripts                           # Last 7 days
      p3 export-transcripts --lookback-days 30        # Last month
      p3 export-transcripts --episode-id 5            # Specific episode
    """
    db = ctx.obj['db']
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if episode_id:
        # Export single episode
        episodes = db.conn.execute("""
            SELECT e.id, e.title, e.date, p.title as podcast_title
            FROM episodes e
            JOIN podcasts p ON e.podcast_id = p.id
            WHERE e.id = ?
        """, (episode_id,)).fetchall()
    else:
        # Export all transcribed episodes in lookback window
        lookback_days = int(lookback_days)
        episodes = db.conn.execute("""
            SELECT e.id, e.title, e.date, p.title as podcast_title
            FROM episodes e
            JOIN podcasts p ON e.podcast_id = p.id
            WHERE e.status IN ('transcribed', 'processed')
            AND e.date >= CURRENT_DATE - CAST(? AS INTEGER) * INTERVAL 1 DAY
            ORDER BY e.date DESC
        """, (lookback_days,)).fetchall()

    if not episodes:
        console.print("[yellow]No transcribed episodes found for the given range.[/yellow]")
        return

    console.print(f"[blue]Exporting {len(episodes)} transcripts to {output_dir}/[/blue]\n")

    exported = 0
    for ep in episodes:
        ep_id, ep_title, ep_date, podcast_title = ep[0], ep[1], ep[2], ep[3]
        segments = db.get_transcripts_for_episode(ep_id)

        if not segments:
            console.print(f"  [yellow]Skipping {ep_title} (no transcript segments)[/yellow]")
            continue

        # Build readable markdown
        safe_podcast = re.sub(r'[^\w\s-]', '', podcast_title).strip().replace(' ', '_')
        safe_title = re.sub(r'[^\w\s-]', '', ep_title).strip().replace(' ', '_')
        date_str = str(ep_date)[:10] if ep_date else 'unknown'
        filename = f"{safe_podcast}_{safe_title}_{date_str}.md"
        # Truncate filename if too long
        if len(filename) > 200:
            filename = filename[:196] + '.md'

        content = f"# {podcast_title}: {ep_title}\n\n"
        content += f"**Date:** {date_str}\n"
        content += f"**Episode ID:** {ep_id}\n"
        content += f"**Segments:** {len(segments)}\n\n"
        content += "---\n\n"

        # Add transcript with periodic timestamps
        last_timestamp_shown = -60  # Force first timestamp to show
        for seg in segments:
            ts_start = seg.get('timestamp_start', 0) or 0
            text = seg.get('text', '').strip()
            if not text:
                continue

            # Show timestamp every ~60 seconds
            if ts_start - last_timestamp_shown >= 60:
                minutes = int(ts_start // 60)
                seconds = int(ts_start % 60)
                content += f"\n**[{minutes:02d}:{seconds:02d}]**\n\n"
                last_timestamp_shown = ts_start

            content += f"{text} "

        filepath = output_path / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content.strip() + '\n')

        exported += 1
        safe_display = ep_title.replace('[', '\\[').replace(']', '\\]')
        console.print(f"  [green]Exported:[/green] {safe_display}")

    console.print(f"\n[green]Exported {exported} transcripts to {output_dir}/[/green]")


@main.command(name='audio-digest')
@click.option('--date', help='Date to generate digest for (YYYY-MM-DD)')
@click.option('--duration', type=int, help='Target duration in seconds (default: 1/10th of input)')
@click.option('--episodes', help='Comma-separated episode IDs (overrides --date)')
@click.option('--model', help='Ollama model for clip selection')
@click.option('--output', help='Output file path')
@click.pass_context
def audio_digest(ctx, date, duration, episodes, model, output):
    """Generate audio digest by selecting and stitching best podcast clips.

    Creates a condensed audio version containing the most valuable segments
    from podcasts, targeting 1/10th the length of the original content.

    Examples:

      # Generate digest for today's episodes (auto 1/10th duration)
      p3 audio-digest

      # Generate digest for specific date
      p3 audio-digest --date 2024-01-15

      # Generate 5-minute digest for today
      p3 audio-digest --duration 300

      # Generate digest for specific episodes
      p3 audio-digest --episodes 1,2,3
    """
    config = load_config(ctx.obj['config_path'])
    db = ctx.obj['db']

    # Get model from config or option
    if not model:
        model = config.get('settings', {}).get('model', 'llama3.2:3b')

    # Parse date
    if date:
        try:
            target_date = datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            console.print("[red]Invalid date format. Use YYYY-MM-DD[/red]")
            return
    else:
        target_date = datetime.now()

    console.print(f"[blue]🎵 Generating audio digest...[/blue]\n")

    # Get audio digest config settings
    audio_config = config.get('settings', {}).get('audio_digest', {})
    min_clip_duration = audio_config.get('min_clip_duration', 30.0)
    max_clip_duration = audio_config.get('max_clip_duration', 180.0)
    max_candidates_per_episode = audio_config.get('max_candidates_per_episode', 40)
    max_clips_per_episode = audio_config.get('max_clips_per_episode', 20)

    # Get narration config
    narration_config = audio_config.get('narration', {})
    enable_narration = narration_config.get('enabled', True)
    piper_voice = narration_config.get('voice', 'en_US-lessac-medium')  # Local Piper voice for fallback

    # Prepare TTS API configuration
    settings = config.get('settings', {})
    tts_config = prepare_tts_config(settings)

    # Prepare LLM API configuration
    llm_config = prepare_llm_config(settings)

    # Override model from command line if provided
    if model:
        llm_config['model'] = model

    # Determine TTS voice: use API voice if enabled, otherwise use Piper voice
    if tts_config['use_api'] and tts_config['voice']:
        tts_voice = tts_config['voice']  # API-specific voice (e.g., Fritz-PlayAI for Groq)
    else:
        tts_voice = piper_voice  # Use local Piper voice

    # Run pre-flight dependency checks
    console.print("[cyan]Checking dependencies...[/cyan]")
    checks = DependencyChecker.ensure_dependencies(
        settings,
        require_narration=enable_narration,
        use_tts_api=tts_config['use_api'],
        console=console
    )

    critical_failed = False
    for check_name, (passed, message) in checks.items():
        if not passed and check_name in ["ffmpeg", "Ollama"]:
            critical_failed = True
            console.print(f"[red]✗ {check_name}: {message}[/red]")
        elif not passed:
            console.print(f"[yellow]⚠ {check_name}: {message}[/yellow]")

    if critical_failed:
        console.print(f"\n[red]✗ Critical dependencies missing![/red]")
        console.print("[cyan]Run 'p3 check-deps' for installation instructions[/cyan]")
        return

    # Show Python version warning if needed (only for local Piper TTS)
    python_passed, python_msg = checks["Python Version"]
    if python_passed and "3.9+" in python_msg and enable_narration and not tts_config['use_api']:
        console.print(f"[yellow]ℹ TTS narration will be disabled (Python 3.9+ required for local Piper)[/yellow]")
        console.print(f"[dim]Run 'p3 check-deps' for Python upgrade instructions or use tts_api[/dim]")

    console.print("[green]✓ All dependencies satisfied[/green]\n")

    generator = AudioDigestGenerator(
        db,
        model=llm_config['model'],
        min_clip_duration=min_clip_duration,
        max_clip_duration=max_clip_duration,
        max_candidates_per_episode=max_candidates_per_episode,
        max_clips_per_episode=max_clips_per_episode,
        enable_narration=enable_narration,
        tts_voice=tts_voice,
        use_tts_api=tts_config['use_api'],
        tts_api_provider=tts_config['api_provider'],
        tts_api_model=tts_config['api_model'],
        tts_api_key_file=tts_config['api_key_file'],
        llm_provider=llm_config['provider'],
        llm_api_key_file=llm_config['api_key_file'],
        piper_voice=piper_voice,
        ollama_base_url=llm_config['ollama_base_url'],
        ollama_model=llm_config['ollama_model'],
        model_fallbacks=llm_config.get('model_fallbacks')
    )

    try:
        if episodes:
            # Generate for specific episodes
            episode_ids = [int(ep.strip()) for ep in episodes.split(',')]
            console.print(f"[blue]Processing {len(episode_ids)} episodes[/blue]")

            output_file = generator.generate_digest_for_episodes(
                episode_ids,
                target_duration=duration,
                output_path=output
            )
        else:
            # Generate for date
            console.print(f"[blue]Processing episodes for {target_date.date()}[/blue]")

            output_file = generator.generate_digest_for_date(
                target_date,
                target_duration=duration,
                output_dir=output or "outputs/audio_digests"
            )

        if output_file:
            # Get digest metadata
            digest_info = db.get_audio_digest_by_date(target_date)

            # === POST-GENERATION SUMMARY REPORT ===
            console.print("\n" + "="*60)
            console.print("[bold green]✅ AUDIO DIGEST GENERATION COMPLETE[/bold green]")
            console.print("="*60 + "\n")

            # File info
            from pathlib import Path
            output_path = Path(output_file)
            file_size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0

            console.print("[cyan]📁 Output File[/cyan]")
            console.print(f"  Location: {output_file}")
            console.print(f"  Size: {file_size_mb:.2f} MB")
            console.print(f"  Format: MP3")
            console.print()

            # Digest statistics
            if digest_info:
                console.print("[cyan]📊 Digest Statistics[/cyan]")
                console.print(f"  Duration: {digest_info['duration_sec']:.1f}s ({digest_info['duration_sec']//60}m {digest_info['duration_sec']%60:.0f}s)")
                console.print(f"  Target: {digest_info['target_duration_sec']}s ({digest_info['target_duration_sec']//60}m {digest_info['target_duration_sec']%60:.0f}s)")

                # Calculate compression ratio
                if digest_info['target_duration_sec'] > 0:
                    accuracy = (digest_info['duration_sec'] / digest_info['target_duration_sec']) * 100
                    console.print(f"  Target Accuracy: {accuracy:.0f}%")

                console.print(f"  Total Clips: {digest_info['clip_count']}")
                console.print(f"  Source Episodes: {digest_info['episode_count']}")

                # Calculate average clip length
                avg_clip_sec = digest_info['duration_sec'] / digest_info['clip_count'] if digest_info['clip_count'] > 0 else 0
                console.print(f"  Avg Clip Length: {avg_clip_sec:.1f}s")
                console.print()

                # Get source episode details
                console.print("[cyan]🎙️  Source Episodes[/cyan]")
                episodes = db.conn.execute("""
                    SELECT DISTINCT e.id, e.title, p.title as podcast_title
                    FROM episodes e
                    JOIN podcasts p ON e.podcast_id = p.id
                    WHERE date(e.date) = date(?)
                    AND e.status = 'transcribed'
                    ORDER BY e.date DESC
                """, [target_date]).fetchall()

                for i, ep in enumerate(episodes, 1):
                    # Escape title to prevent Rich markup interpretation
                    safe_title = ep[1].replace('[', '\\[').replace(']', '\\]')
                    safe_podcast = ep[2].replace('[', '\\[').replace(']', '\\]')
                    console.print(f"  {i}. {safe_podcast}: {safe_title}")
                console.print()

            # Performance metrics
            import time
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            console.print("[cyan]⏱️  Generation Info[/cyan]")
            console.print(f"  Completed: {current_time}")
            console.print(f"  Date Range: {target_date.date()}")
            console.print()

            # Next steps
            console.print("[cyan]🎯 What's Next?[/cyan]")
            console.print(f"  • Play digest: Open {output_file}")
            console.print(f"  • View clips: [yellow]p3 status[/yellow]")
            console.print(f"  • Export text digest: [yellow]p3 export --date {target_date.strftime('%Y-%m-%d')}[/yellow]")
            console.print(f"  • Generate blog: [yellow]p3 write --topic 'your topic'[/yellow]")
            console.print()
            console.print("="*60)
        else:
            console.print(f"[yellow]⚠️  No audio digest generated[/yellow]")
            console.print("\n[cyan]💡 Possible reasons:[/cyan]")
            console.print("  • No transcribed episodes found for this date")
            console.print("  • Run: [yellow]p3 transcribe[/yellow] to transcribe downloaded episodes")
            console.print("  • Or run: [yellow]p3 auto[/yellow] to run the full pipeline")

    except Exception as e:
        console.print(f"[red]✗ Error generating audio digest: {e}[/red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")


@main.command()
@click.pass_context
def status(ctx):
    """Show processing status of podcasts, newsletters, and outputs."""
    db = ctx.obj['db']
    from datetime import date as date_type

    # === PODCAST STATUS ===
    console.print("[bold cyan]📻 Podcast Processing Status[/bold cyan]\n")

    # Count episodes by status
    statuses = ['downloaded', 'transcribed', 'processed']
    counts = {}
    counts_today = {}
    episode_lists = {}
    today = date_type.today()

    for status in statuses:
        episodes = db.get_episodes_by_status(status)
        counts[status] = len(episodes)
        episode_lists[status] = episodes

        # Count today's episodes (by created_at, i.e., when downloaded/added)
        today_eps = []
        for ep in episodes:
            created = ep.get('created_at')
            if created:
                # Handle both datetime objects and date objects
                if hasattr(created, 'date'):
                    ep_date = created.date()
                else:
                    ep_date = created
                if ep_date == today:
                    today_eps.append(ep)
        counts_today[status] = len(today_eps)

    # Total episode count across all statuses
    total_episodes = sum(counts.values())

    table = Table(title="Episode Status")
    table.add_column("Status", style="cyan")
    table.add_column("Count", style="green", justify="right")
    table.add_column("Today", style="bright_green", justify="right")

    for s in statuses:
        count_str = str(counts[s])
        today_str = str(counts_today[s]) if counts_today[s] > 0 else "-"
        table.add_row(s.title(), count_str, today_str)

    table.add_section()
    table.add_row("[bold]Total[/bold]", f"[bold]{total_episodes}[/bold]", "")

    console.print(table)

    # Database size
    db_path = Path(ctx.obj.get('db_path', 'data/p3.duckdb'))
    if db_path.exists():
        db_size_mb = db_path.stat().st_size / (1024 * 1024)
        console.print(f"  Database: {db_size_mb:.1f} MB ({db_path})")

    # === NEWSLETTER STATUS ===
    newsletter_counts = {
        'tracked': len(db.get_all_newsletters()),
        'fetched': len(db.get_articles_by_status('fetched')),
        'processed': len(db.get_articles_by_status('processed'))
    }

    if newsletter_counts['tracked'] > 0:
        console.print("\n[bold cyan]📰 Newsletter Status[/bold cyan]\n")
        nl_table = Table(title="Newsletter Status")
        nl_table.add_column("Status", style="cyan")
        nl_table.add_column("Count", style="green", justify="right")
        nl_table.add_row("Newsletters Tracked", str(newsletter_counts['tracked']))
        nl_table.add_row("Articles Pending", str(newsletter_counts['fetched']))
        nl_table.add_row("Articles Processed", str(newsletter_counts['processed']))
        console.print(nl_table)

    # === GENERATED OUTPUTS ===
    from glob import glob

    # Helper to filter out example files (files with .example. in the name)
    def count_real_outputs(pattern):
        files = glob(pattern)
        # Exclude files that contain .example. in their name
        real_files = [f for f in files if '.example.' not in f]
        return len(real_files)

    output_counts = {
        'digests': count_real_outputs("outputs/digests/*.md") + count_real_outputs("outputs/digests/*.json"),
        'audio_digests': count_real_outputs("outputs/audio_digests/*.mp3"),
        'blogs': count_real_outputs("outputs/blogs/*.md"),
        'tweets': count_real_outputs("outputs/tweets/*.txt"),
        'thesis': count_real_outputs("outputs/thesis/*.md"),
        'transcripts': count_real_outputs("outputs/transcripts/*.md")
    }

    if any(output_counts.values()):
        console.print("\n[bold cyan]📂 Generated Outputs[/bold cyan]\n")
        out_table = Table(title="Output Files")
        out_table.add_column("Type", style="cyan")
        out_table.add_column("Count", style="green", justify="right")
        out_table.add_column("Location", style="yellow")

        if output_counts['digests'] > 0:
            out_table.add_row("Digests", str(output_counts['digests']), "outputs/digests/")
        if output_counts['audio_digests'] > 0:
            out_table.add_row("Audio Digests", str(output_counts['audio_digests']), "outputs/audio_digests/")
        if output_counts['blogs'] > 0:
            out_table.add_row("Blog Posts", str(output_counts['blogs']), "outputs/blogs/")
        if output_counts['tweets'] > 0:
            out_table.add_row("Tweets", str(output_counts['tweets']), "outputs/tweets/")
        if output_counts['thesis'] > 0:
            out_table.add_row("Analyses", str(output_counts['thesis']), "outputs/thesis/")
        if output_counts['transcripts'] > 0:
            out_table.add_row("Transcripts", str(output_counts['transcripts']), "outputs/transcripts/")

        console.print(out_table)

    # === TOP SOURCES ===
    source_stats = _get_source_stats()
    if source_stats:
        console.print("\n[bold cyan]📊 Top Content Sources[/bold cyan]\n")
        src_table = Table(title="Sources by Usage in Generated Content")
        src_table.add_column("Source", style="cyan")
        src_table.add_column("Times Used", style="green", justify="right")

        total_uses = sum(source_stats.values())
        for source_name, count in list(source_stats.items())[:10]:
            pct = round(count / total_uses * 100) if total_uses else 0
            src_table.add_row(source_name, f"{count} ({pct}%)")

        console.print(src_table)

    # === RECENT SUMMARIES ===
    processed_episodes = episode_lists.get('processed', [])
    if processed_episodes:
        console.print("\n[cyan]📊 Recent Summaries (Ready for Export):[/cyan]")
        for ep in processed_episodes[:5]:  # Show last 5
            podcast_name = db.conn.execute(
                "SELECT title FROM podcasts WHERE id = ?",
                [ep['podcast_id']]
            ).fetchone()[0]
            # Escape title to prevent Rich markup interpretation (e.g., "10x" -> "10[x]")
            safe_title = ep['title'].replace('[', '\\[').replace(']', '\\]')
            console.print(f"  • {podcast_name}: {safe_title}")

        if len(processed_episodes) > 5:
            console.print(f"  ... and {len(processed_episodes) - 5} more")

    # === PENDING WORK ===
    downloaded_episodes = episode_lists.get('downloaded', [])
    transcribed_episodes = episode_lists.get('transcribed', [])
    fetched_articles = newsletter_counts['fetched']

    if downloaded_episodes:
        console.print("\n[yellow]⏳ Pending Transcription:[/yellow]")
        for ep in downloaded_episodes[:5]:
            podcast_name = db.conn.execute(
                "SELECT title FROM podcasts WHERE id = ?",
                [ep['podcast_id']]
            ).fetchone()[0]
            safe_title = ep['title'].replace('[', '\\[').replace(']', '\\]')
            console.print(f"  • {podcast_name}: {safe_title}")

        if len(downloaded_episodes) > 5:
            console.print(f"  ... and {len(downloaded_episodes) - 5} more")

    if transcribed_episodes:
        console.print("\n[blue]📝 Pending Digest Generation:[/blue]")
        for ep in transcribed_episodes[:5]:
            podcast_name = db.conn.execute(
                "SELECT title FROM podcasts WHERE id = ?",
                [ep['podcast_id']]
            ).fetchone()[0]
            safe_title = ep['title'].replace('[', '\\[').replace(']', '\\]')
            console.print(f"  • {podcast_name}: {safe_title}")

        if len(transcribed_episodes) > 5:
            console.print(f"  ... and {len(transcribed_episodes) - 5} more")

    if fetched_articles > 0:
        console.print(f"\n[blue]📰 Pending Newsletter Digests:[/blue] {fetched_articles} articles")

    # === NEXT STEPS ===
    if downloaded_episodes or transcribed_episodes or fetched_articles:
        console.print("\n[cyan]💡 Next Steps:[/cyan]")
        if downloaded_episodes:
            console.print("  • Run: [yellow]p3 transcribe[/yellow]")
        if transcribed_episodes:
            console.print("  • Run: [yellow]p3 digest[/yellow]")
        if fetched_articles:
            console.print("  • Run: [yellow]p3 newsletter digest[/yellow]")
        console.print("  • Or run: [yellow]p3 auto[/yellow] (complete pipeline)")
    elif processed_episodes:
        console.print("\n[green]✓ All content processed![/green]")
        console.print("[cyan]💡 Next Steps:[/cyan]")
        console.print("  • Export digests: [yellow]p3 export[/yellow]")
        console.print("  • Generate audio digest: [yellow]p3 audio-digest[/yellow]")
        console.print("  • Find content ideas: [yellow]p3 ideas[/yellow]")
        console.print("  • Generate blog: [yellow]p3 write --topic 'Your Topic'[/yellow]")
        console.print("  • Create tweet: [yellow]p3 tweet --topic 'Topic'[/yellow]")
        console.print("  • Analyze company: [yellow]p3 thesis --list-companies[/yellow]")
    else:
        console.print("\n[yellow]No content to process yet.[/yellow]")
        console.print("[cyan]💡 Get started:[/cyan]")
        console.print("  • Fetch podcasts: [yellow]p3 fetch[/yellow]")
        console.print("  • Add newsletters: [yellow]p3 newsletter add <url>[/yellow]")
        console.print("  • Run full pipeline: [yellow]p3 auto[/yellow]")

    # === SYSTEM DEPENDENCIES ===
    console.print("\n[bold cyan]🔧 System Dependencies[/bold cyan]\n")

    config = load_config(ctx.obj['config_path'])
    narration_enabled = config.get('settings', {}).get('audio_digest', {}).get('narration', {}).get('enabled', True)

    checks = DependencyChecker.run_pre_flight_checks(require_narration=narration_enabled)

    # Print status
    for check_name, (passed, message) in checks.items():
        status_icon = "✓" if passed else "✗"
        status_color = "green" if passed else "yellow" if check_name == "Piper TTS" else "red"
        console.print(f"[{status_color}]{status_icon}[/{status_color}] {check_name}: {message}")

    # Show Python upgrade instructions if needed
    python_passed, python_msg = checks["Python Version"]
    if python_passed and "3.9+" in python_msg and narration_enabled:
        console.print("\n[yellow]ℹ TTS narration disabled (requires Python 3.9+)[/yellow]")
        console.print("[dim]Run 'p3 check-deps' for upgrade instructions[/dim]")


@main.command()
@click.option('--verbose', is_flag=True, help='Show detailed dependency information')
@click.pass_context
def check_deps(ctx, verbose):
    """Check system dependencies and provide installation guidance."""
    config = load_config(ctx.obj['config_path'])
    narration_enabled = config.get('settings', {}).get('audio_digest', {}).get('narration', {}).get('enabled', True)

    checks = DependencyChecker.run_pre_flight_checks(require_narration=narration_enabled)
    all_passed = DependencyChecker.print_dependency_status(checks, verbose=True)

    # Show Python upgrade instructions if Python < 3.9
    python_passed, python_msg = checks["Python Version"]
    if python_passed and "3.9+" in python_msg:
        print(DependencyChecker.get_python_upgrade_instructions())

    # Show next steps
    if not all_passed:
        critical_missing = []
        for check_name, (passed, _) in checks.items():
            if not passed and check_name in ["ffmpeg", "Ollama"]:
                critical_missing.append(check_name)

        if critical_missing:
            print("\n" + "="*60)
            print("Next Steps:")
            print("="*60)
            if "ffmpeg" in critical_missing:
                print("\n1. Install ffmpeg:")
                print("   Linux: sudo apt install ffmpeg")
                print("   macOS: brew install ffmpeg")
                print("   Windows: https://ffmpeg.org/download.html")
            if "Ollama" in critical_missing:
                print("\n2. Install Ollama:")
                print("   Visit: https://ollama.com")
                print("   Then run: ollama pull llama3.2:3b")
            print()


@main.command()
@click.option('--verbose', is_flag=True, help='Show detailed diagnostic information')
@click.option('--fix', is_flag=True, help='Attempt to fix common issues automatically')
@click.pass_context
def doctor(ctx, verbose, fix):
    """Comprehensive system health check and diagnostics.

    Validates your entire P3 setup including Python version, dependencies,
    storage, memory, database health, and configuration. Provides actionable
    recommendations to fix any issues.

    Examples:
        p3 doctor                    Quick health check
        p3 doctor --verbose          Detailed diagnostics
        p3 doctor --fix              Auto-fix common issues (creates missing directories)
    """
    config = load_config(ctx.obj['config_path'])
    db = ctx.obj['db']

    console.print("[bold blue]🏥 P3 System Health Check[/bold blue]\n")

    # Run diagnostics
    doctor_instance = SystemDoctor(db, config)
    results = doctor_instance.run_full_checkup()

    # === System Information ===
    if verbose:
        console.print("[cyan]💻 System Information[/cyan]")
        sys_info = results['system_info']
        console.print(f"  Platform: {sys_info['platform']} ({sys_info['architecture']})")
        console.print(f"  OS Version: {sys_info['platform_version']}")
        console.print(f"  Processor: {sys_info['processor']}")
        console.print(f"  Python: {sys_info['python_version']} ({sys_info['python_implementation']})")
        console.print()

    # === Python Version ===
    console.print("[cyan]🐍 Python Version[/cyan]")
    python_result = results['python']
    if python_result['status'] == 'pass':
        console.print(f"  [green]✓[/green] Python {python_result['version']}")
    elif python_result['status'] == 'warning':
        console.print(f"  [yellow]⚠[/yellow] Python {python_result['version']}")
    else:
        console.print(f"  [red]✗[/red] Python {python_result['version']}")

    if python_result.get('details'):
        for detail in python_result['details']:
            console.print(f"    {detail}")
    console.print()

    # === Dependencies ===
    console.print("[cyan]🔧 System Dependencies[/cyan]")
    deps = results['dependencies']
    for dep_name, dep_result in deps.items():
        if dep_result['status'] == 'pass':
            console.print(f"  [green]✓[/green] {dep_name}: {dep_result['message']}")
        else:
            console.print(f"  [red]✗[/red] {dep_name}: {dep_result['message']}")
    console.print()

    # === Storage ===
    console.print("[cyan]💾 Storage[/cyan]")
    storage = results['storage']
    if storage['status'] == 'error':
        console.print(f"  [red]✗[/red] Storage check failed: {storage.get('message', 'Unknown error')}")
    else:
        status_icon = "✓" if storage['status'] == 'pass' else "⚠" if storage['status'] == 'warning' else "✗"
        status_color = "green" if storage['status'] == 'pass' else "yellow" if storage['status'] == 'warning' else "red"
        console.print(f"  [{status_color}]{status_icon}[/{status_color}] Disk Space: {storage['free_gb']} GB free of {storage['total_gb']} GB total ({storage['used_percent']}% used)")

        if 'audio_dir_mb' in storage:
            console.print(f"  Audio directory: {storage['audio_dir_mb']} MB")
    console.print()

    # === Memory ===
    console.print("[cyan]🧠 Memory[/cyan]")
    memory = results['memory']
    if memory['status'] == 'error':
        console.print(f"  [red]✗[/red] Memory check failed: {memory.get('message', 'Unknown error')}")
    else:
        status_icon = "✓" if memory['status'] == 'pass' else "⚠" if memory['status'] == 'warning' else "✗"
        status_color = "green" if memory['status'] == 'pass' else "yellow" if memory['status'] == 'warning' else "red"
        console.print(f"  [{status_color}]{status_icon}[/{status_color}] RAM: {memory['available_gb']} GB available of {memory['total_gb']} GB total ({memory['used_percent']}% used)")
    console.print()

    # === Database ===
    console.print("[cyan]🗄️  Database[/cyan]")
    database = results['database']
    if database['status'] == 'pass':
        console.print(f"  [green]✓[/green] Database healthy")
        console.print(f"    Episodes: {database['episode_count']}")
        console.print(f"    Summaries: {database['summary_count']}")

        if database.get('pending_transcription', 0) > 0:
            console.print(f"    [yellow]⏳ Pending transcription: {database['pending_transcription']}[/yellow]")
        if database.get('pending_digest', 0) > 0:
            console.print(f"    [yellow]⏳ Pending digest: {database['pending_digest']}[/yellow]")
    else:
        console.print(f"  [red]✗[/red] Database issues detected")
        if 'error' in database:
            console.print(f"    Error: {database['error']}")
    console.print()

    # === Configuration ===
    console.print("[cyan]⚙️  Configuration[/cyan]")
    cfg = results['config']
    if cfg['status'] == 'pass':
        console.print(f"  [green]✓[/green] Configuration valid")
        console.print(f"    Podcast feeds: {cfg.get('feed_count', 0)}")
        console.print(f"    LLM: {cfg['llm_provider']} / {cfg['llm_model']}")
        console.print(f"    Audio digest: {'enabled' if cfg['audio_digest_enabled'] else 'disabled'}")
        console.print(f"    TTS narration: {'enabled' if cfg['narration_enabled'] else 'disabled'}")
    else:
        console.print(f"  [yellow]⚠[/yellow] Configuration issues")
        for detail in cfg.get('details', []):
            console.print(f"    {detail}")
    console.print()

    # === Directories ===
    console.print("[cyan]📁 Directories[/cyan]")
    dirs = results['directories']
    if dirs['status'] == 'pass':
        console.print(f"  [green]✓[/green] All required directories present ({len(dirs['present'])})")
    else:
        console.print(f"  [yellow]⚠[/yellow] Missing directories: {len(dirs['missing'])}")
        if verbose:
            for missing in dirs['missing']:
                console.print(f"    Missing: {missing}")

        if fix:
            console.print("\n  [cyan]Creating missing directories...[/cyan]")
            for missing_dir in dirs['missing']:
                Path(missing_dir).mkdir(parents=True, exist_ok=True)
                console.print(f"    [green]✓[/green] Created: {missing_dir}")
    console.print()

    # === Models ===
    console.print("[cyan]🤖 LLM Models[/cyan]")
    models = results['models']
    if not models['ollama_available']:
        console.print(f"  [red]✗[/red] Ollama not running")
    else:
        console.print(f"  [green]✓[/green] Ollama running")
        console.print(f"    Installed models: {models['model_count']}")

        if models.get('missing_recommended'):
            console.print(f"    [yellow]⚠[/yellow] Missing configured model: {models['missing_recommended'][0]}")
            console.print(f"    Will auto-download on first use")
    console.print()

    # === Overall Health Status ===
    console.print("="*60)
    overall = results['overall_health']
    if overall == 'healthy':
        console.print("[bold green]✅ Overall Health: HEALTHY[/bold green]")
        console.print("All systems operational. Ready to run P3!")
    elif overall == 'degraded':
        console.print("[bold yellow]⚠️  Overall Health: DEGRADED[/bold yellow]")
        console.print("Some non-critical issues detected. P3 can run but with reduced functionality.")
    else:
        console.print("[bold red]❌ Overall Health: CRITICAL[/bold red]")
        console.print("Critical issues detected. P3 cannot run until these are resolved.")
    console.print("="*60)

    # === Summary ===
    if results['issues'] or results['warnings']:
        console.print()
        if results['issues']:
            console.print("[bold red]🚨 Critical Issues:[/bold red]")
            for issue in results['issues']:
                console.print(f"  • {issue}")
            console.print()

        if results['warnings']:
            console.print("[bold yellow]⚠️  Warnings:[/bold yellow]")
            for warning in results['warnings']:
                console.print(f"  • {warning}")
            console.print()

    # === Recommendations ===
    recommendations = doctor_instance.get_recommendations()
    if recommendations:
        console.print("[bold cyan]💡 Recommendations:[/bold cyan]")
        for i, rec in enumerate(recommendations, 1):
            console.print(f"  {i}. {rec}")
        console.print()

    # Show next steps
    if overall == 'healthy':
        console.print("[cyan]🎯 Next Steps:[/cyan]")
        console.print("  • Run full pipeline: [yellow]p3 auto[/yellow]")
        console.print("  • Check status: [yellow]p3 status[/yellow]")
        console.print("  • Generate audio digest: [yellow]p3 audio-digest[/yellow]")
    elif overall == 'degraded':
        console.print("[cyan]🎯 Next Steps:[/cyan]")
        console.print("  • Address warnings above to unlock full functionality")
        console.print("  • Core pipeline can still run: [yellow]p3 auto[/yellow]")
    else:
        console.print("[cyan]🎯 Next Steps:[/cyan]")
        console.print("  • Address critical issues above before running P3")
        console.print("  • Run [yellow]p3 check-deps[/yellow] for installation instructions")
        console.print("  • Re-run [yellow]p3 doctor[/yellow] after fixes to verify")


@main.command()
@click.option('--skip-download', is_flag=True, help='Use existing episodes instead of downloading test episode')
@click.option('--quick', is_flag=True, help='Skip audio digest generation (faster test)')
@click.pass_context
def test(ctx, skip_download, quick):
    """Quick pipeline validation with test sample.

    Runs a quick end-to-end test of the P3 pipeline to verify everything
    is working correctly. This is useful after installation or system changes.

    The test will:
    1. Check dependencies
    2. Download a test episode (or use existing)
    3. Transcribe audio
    4. Generate digest
    5. Generate audio digest (unless --quick)

    Examples:
        p3 test                 Full pipeline test
        p3 test --quick         Skip audio digest generation
        p3 test --skip-download Use existing episodes
    """
    config = load_config(ctx.obj['config_path'])
    db = ctx.obj['db']
    settings = config.get('settings', {})

    console.print("[bold blue]🧪 P3 Pipeline Test[/bold blue]\n")

    # Track results
    test_results = {
        'dependencies': False,
        'download': False,
        'transcription': False,
        'digest': False,
        'audio_digest': False
    }

    # Step 1: Check dependencies
    console.print("[cyan]Step 1/5: Checking dependencies...[/cyan]")
    from .dependency_checker import DependencyChecker

    checks = DependencyChecker.run_pre_flight_checks(require_narration=False)
    critical_failed = False
    for check_name, (passed, message) in checks.items():
        if not passed and check_name in ["ffmpeg", "Ollama"]:
            critical_failed = True
            console.print(f"  [red]✗ {check_name}: {message}[/red]")

    if critical_failed:
        console.print(f"\n[red]✗ Test failed: Critical dependencies missing[/red]")
        console.print("Run [yellow]p3 doctor[/yellow] for detailed diagnostics")
        return

    test_results['dependencies'] = True
    console.print("  [green]✓ All dependencies available[/green]\n")

    # Step 2: Get or download test episode
    console.print("[cyan]Step 2/5: Preparing test episode...[/cyan]")

    test_episode_id = None

    if skip_download:
        # Use existing downloaded episode
        episodes = db.get_episodes_by_status('downloaded')
        if episodes:
            test_episode_id = episodes[0]['id']
            console.print(f"  [green]✓ Using existing episode: {episodes[0]['title']}[/green]\n")
            test_results['download'] = True
        else:
            console.print("  [yellow]⚠ No downloaded episodes found[/yellow]")
            console.print("  Remove --skip-download flag to fetch a test episode\n")
            return
    else:
        # Download test episode from first configured feed
        feeds = config.get('feeds', [])
        if not feeds:
            console.print("  [red]✗ No feeds configured in config/feeds.yaml[/red]")
            console.print("  Configure at least one feed to run test\n")
            return

        # Use first feed for test
        test_feed = feeds[0]
        console.print(f"  Fetching from: {test_feed['name']}")

        downloader = PodcastDownloader(
            db=db,
            max_episodes=1,
            audio_format=settings.get('audio_format', 'wav')
        )

        try:
            results = downloader.fetch_all_feeds([test_feed])
            downloaded_count = results.get(test_feed['name'], 0)

            if downloaded_count > 0:
                # Get the episode we just downloaded
                episodes = db.get_episodes_by_status('downloaded')
                if episodes:
                    test_episode_id = episodes[0]['id']
                    console.print(f"  [green]✓ Downloaded test episode[/green]\n")
                    test_results['download'] = True
            else:
                console.print("  [yellow]⚠ No new episodes to download (may already be processed)[/yellow]")
                # Try to use an existing transcribed episode
                episodes = db.get_episodes_by_status('transcribed')
                if episodes:
                    test_episode_id = episodes[0]['id']
                    test_results['download'] = True
                    console.print(f"  [green]✓ Using existing episode[/green]\n")
                else:
                    console.print("  Run test again later when new episodes are available\n")
                    return

        except Exception as e:
            console.print(f"  [red]✗ Download failed: {e}[/red]\n")
            return

    # Step 3: Transcribe
    console.print("[cyan]Step 3/5: Transcribing audio...[/cyan]")

    whisper_model = settings.get('whisper_model', 'base')
    use_faster_whisper = settings.get('use_faster_whisper', True)

    # Prepare transcription API configuration
    trans_config = prepare_transcription_config(settings)

    if trans_config['use_api']:
        console.print(f"  Provider: {trans_config['api_provider']} API, Model: {trans_config['api_model']}")
    else:
        console.print(f"  Model: {whisper_model}, Faster-whisper: {use_faster_whisper}")

    transcriber = AudioTranscriber(
        db=db,
        whisper_model=whisper_model,
        use_parakeet=False,
        use_faster_whisper=use_faster_whisper,
        use_api=trans_config['use_api'],
        api_provider=trans_config['api_provider'],
        api_model=trans_config['api_model'],
        api_fallback_model=trans_config.get('api_fallback_model'),
        api_fallback_models=trans_config.get('api_fallback_models'),
        api_key_file=trans_config['api_key_file'],
        api_base_url=trans_config['api_base_url'],
        wait_for_ratelimit=trans_config['wait_for_ratelimit'],
        max_wait_retries=trans_config['max_wait_retries']
    )

    try:
        with console.status("[bold green]Transcribing (this may take a few minutes)..."):
            success = transcriber.transcribe_episode(test_episode_id)

        if success:
            console.print("  [green]✓ Transcription completed[/green]\n")
            test_results['transcription'] = True
        else:
            console.print("  [red]✗ Transcription failed[/red]\n")
            console.print("[yellow]Check logs for details[/yellow]")
            return

    except Exception as e:
        console.print(f"  [red]✗ Transcription error: {e}[/red]\n")
        return

    # Step 4: Generate digest
    console.print("[cyan]Step 4/5: Generating digest...[/cyan]")

    # Prepare LLM configuration (handles external APIs if enabled)
    llm_config = prepare_llm_config(settings)

    console.print(f"  Provider: {llm_config['provider']}")
    console.print(f"  Model: {llm_config['model']}")

    # Auto-download model if needed (only for Ollama)
    from .model_manager import ModelManager
    if llm_config['provider'] == 'ollama':
        if not ModelManager.ensure_model_available(llm_config['model'], console):
            console.print(f"  [red]✗ Failed to get model {llm_config['model']}[/red]\n")
            return

    cleaner = TranscriptCleaner(
        db=db,
        llm_provider=llm_config['provider'],
        llm_model=llm_config['model'],
        ollama_base_url=llm_config['ollama_base_url'],
        ollama_model=llm_config['ollama_model'],
        model_fallbacks=llm_config.get('model_fallbacks'),
        api_key_file=llm_config['api_key_file'],
        base_url=llm_config['base_url'],
        timeout=llm_config['timeout'],
        max_retries=llm_config['max_retries']
    )

    try:
        with console.status("[bold green]Generating digest..."):
            result = cleaner.generate_summary(test_episode_id)

        if result:
            console.print("  [green]✓ Digest generated[/green]")
            test_results['digest'] = True

            # Export text digest to file
            try:
                from .exporter import DigestExporter
                from pathlib import Path

                exporter = DigestExporter(db)
                episode = db.conn.execute(
                    "SELECT date FROM episodes WHERE id = ?",
                    [test_episode_id]
                ).fetchone()

                if episode:
                    episode_date = episode[0]
                    output_dir = Path("outputs/digests")
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # Get summaries for this episode's date
                    summaries = db.get_summaries_for_date(episode_date)

                    if summaries:
                        # Export as markdown
                        content = exporter.export_markdown(summaries, episode_date.date(), include_newsletters=False)
                        filename = output_dir / f"digest_{episode_date.strftime('%Y-%m-%d')}.md"

                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(content)

                        console.print(f"  [green]✓ Text digest saved: {filename}[/green]\n")
            except Exception as e:
                console.print(f"  [yellow]⚠ Could not export text digest: {e}[/yellow]\n")
        else:
            console.print("  [red]✗ Digest generation failed[/red]\n")
            return

    except Exception as e:
        console.print(f"  [red]✗ Digest error: {e}[/red]\n")
        return

    # Step 5: Generate audio digest (optional)
    if not quick:
        console.print("[cyan]Step 5/5: Generating audio digest...[/cyan]")

        # Get audio digest config settings
        audio_config = settings.get('audio_digest', {})
        min_clip_duration = audio_config.get('min_clip_duration', 30.0)
        max_clip_duration = audio_config.get('max_clip_duration', 180.0)
        max_candidates_per_episode = audio_config.get('max_candidates_per_episode', 40)
        max_clips_per_episode = audio_config.get('max_clips_per_episode', 20)

        narration_config = audio_config.get('narration', {})
        enable_narration = narration_config.get('enabled', False)
        piper_voice = narration_config.get('voice', 'en_US-lessac-medium')  # Local Piper voice for fallback

        # Prepare TTS API configuration
        tts_config = prepare_tts_config(settings)

        # Ensure dependencies before audio digest generation
        checks = DependencyChecker.ensure_dependencies(
            settings,
            require_narration=enable_narration,
            use_tts_api=tts_config['use_api'],
            console=console
        )
        critical_failed = False
        for check_name, (passed, message) in checks.items():
            if not passed and check_name in ["ffmpeg", "Ollama"]:
                critical_failed = True
                console.print(f"[red]✗ {check_name}: {message}[/red]")
            elif not passed:
                console.print(f"[yellow]⚠ {check_name}: {message}[/yellow]")

        if critical_failed:
            console.print(f"\n[red]✗ Critical dependencies missing![/red]")
            console.print("[cyan]Run 'p3 check-deps' for installation instructions[/cyan]")
            return

        # Determine TTS voice: use API voice if enabled, otherwise use Piper voice
        if tts_config['use_api'] and tts_config['voice']:
            tts_voice = tts_config['voice']  # API-specific voice
        else:
            tts_voice = piper_voice  # Use local Piper voice

        console.print(f"  Narration: {'enabled' if enable_narration else 'disabled'}{' (API)' if tts_config['use_api'] else ' (local)'}")
        console.print(f"  Clip duration: {min_clip_duration}s - {max_clip_duration}s")
        console.print(f"  Max candidates per episode: {max_candidates_per_episode}")

        generator = AudioDigestGenerator(
            db,
            model=llm_config['model'],
            min_clip_duration=min_clip_duration,
            max_clip_duration=max_clip_duration,
            max_candidates_per_episode=max_candidates_per_episode,
            max_clips_per_episode=max_clips_per_episode,
            enable_narration=enable_narration,
            tts_voice=tts_voice,
            use_tts_api=tts_config['use_api'],
            tts_api_provider=tts_config['api_provider'],
            tts_api_model=tts_config['api_model'],
            tts_api_key_file=tts_config['api_key_file'],
            llm_provider=llm_config['provider'],
            llm_api_key_file=llm_config['api_key_file'],
            piper_voice=piper_voice,
            ollama_base_url=llm_config['ollama_base_url'],
            ollama_model=llm_config['ollama_model'],
            model_fallbacks=llm_config.get('model_fallbacks')
        )

        try:
            # Get episode date
            episode = db.conn.execute(
                "SELECT date FROM episodes WHERE id = ?",
                [test_episode_id]
            ).fetchone()

            if episode:
                episode_date = episode[0]

                with console.status("[bold green]Creating audio digest..."):
                    output_file = generator.generate_digest_for_date(
                        episode_date,
                        target_duration=60  # 1 minute test digest
                    )

                if output_file:
                    console.print(f"  [green]✓ Audio digest created: {output_file}[/green]\n")
                    test_results['audio_digest'] = True
                else:
                    console.print("  [yellow]⚠ Audio digest not generated (may not have enough content)[/yellow]\n")

        except Exception as e:
            console.print(f"  [red]✗ Audio digest error: {e}[/red]\n")
    else:
        console.print("[cyan]Step 5/5: Skipped (--quick mode)[/cyan]\n")
        test_results['audio_digest'] = True  # Mark as passed since we skipped

    # === Test Results Summary ===
    console.print("="*60)
    console.print("[bold]TEST RESULTS SUMMARY[/bold]")
    console.print("="*60)

    all_passed = all(test_results.values())

    for test_name, passed in test_results.items():
        status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
        console.print(f"  {test_name.replace('_', ' ').title()}: {status}")

    console.print("="*60)

    if all_passed:
        console.print("\n[bold green]✅ ALL TESTS PASSED![/bold green]")
        console.print("P3 is working correctly and ready to use.")
        console.print("\n[cyan]🎯 Next Steps:[/cyan]")
        console.print("  • Run full pipeline: [yellow]p3 auto[/yellow]")
        console.print("  • Check status: [yellow]p3 status[/yellow]")
        console.print("  • View this test digest: [yellow]p3 status[/yellow]")
    else:
        console.print("\n[bold red]❌ SOME TESTS FAILED[/bold red]")
        console.print("P3 may not be working correctly.")
        console.print("\n[cyan]🔧 Troubleshooting:[/cyan]")
        console.print("  • Run diagnostics: [yellow]p3 doctor[/yellow]")
        console.print("  • Check dependencies: [yellow]p3 check-deps[/yellow]")
        console.print("  • Review logs in logs/ directory")

    console.print()


@main.command()
@click.option('--topic', required=True, help='Blog post topic/angle')
@click.option('--prompt', default=None, help='Additional instructions for structure, tone, or focus')
@click.option('--date', help='Date (YYYY-MM-DD), defaults to today')
@click.option('--lookback-days', default=None, type=int, help='Use episodes from the last N days instead of a single date')
@click.option('--target-grade', default=91.0, help='Target AP English grade (default: 91)')
@click.option('--limit', default=None, type=int, help='Number of episodes to use (default: all)')
@click.option('--refine', is_flag=True, default=False, help='Enter interactive refinement loop after generation')
@click.pass_context
def write(ctx, topic, prompt, date, lookback_days, target_grade, limit, refine):
    """Generate blog post from podcast summaries (AP English grading).

    Uses Tomasz Tunguz's iterative writing approach. Synthesizes insights
    from multiple podcast episodes into a cohesive blog post.

    \b
    Examples:
      p3 write --topic "AI Trends"
      p3 write --topic "AI Trends" --lookback-days 7
      p3 write --topic "AI Trends" --prompt "Focus on contrarian viewpoints"
      p3 write --topic "AI Trends" --lookback-days 30 --limit 10
    """
    config = load_config(ctx.obj['config_path'])
    db = ctx.obj['db']

    settings = config.get('settings', {})
    llm_config = prepare_llm_config(settings)
    llm_client = LLMClient.from_config(llm_config)

    console.print(f"[blue]LLM Settings:[/blue]")
    console.print(f"  Provider: {llm_config['provider']}")
    console.print(f"  Model: {llm_config['model']}")

    if llm_config['provider'] == 'ollama':
        if not ModelManager.ensure_model_available(llm_config['model'], console):
            console.print(f"[red]Failed to ensure model {llm_config['model']} is available[/red]")
            return

    # Get summaries - either by lookback days or single date
    if lookback_days:
        summaries = _get_unified_summaries(db, lookback_days)
        date_label = f"last {lookback_days} days"
    else:
        if date:
            try:
                target_date = datetime.strptime(date, '%Y-%m-%d')
            except ValueError:
                console.print("[red]Invalid date format. Use YYYY-MM-DD[/red]")
                return
        else:
            target_date = datetime.now()
        summaries = db.get_summaries_by_date(target_date)
        date_label = str(target_date.date())

    if not summaries:
        console.print(f"[yellow]No summaries found for {date_label}[/yellow]")
        console.print("\n[cyan]To generate blog posts, you need summaries first:[/cyan]")
        console.print("  1. Add podcasts: Edit [yellow]config/feeds.yaml[/yellow]")
        console.print("  2. Run pipeline: [yellow]p3 auto[/yellow]")
        if not lookback_days:
            console.print("  Or try: [yellow]p3 write --topic \"...\" --lookback-days 7[/yellow]")
        return

    if limit and limit > 0:
        summaries = summaries[:limit]

    console.print(f"\n[blue]Found {len(summaries)} episodes from {date_label}[/blue]")

    # Filter for relevance using stored metadata (no LLM call)
    relevant = _filter_relevant_summaries(summaries, topic)

    if not relevant:
        console.print(f"[yellow]No episodes matched '{topic}'. Using all {len(summaries)} episodes.[/yellow]")
        relevant = summaries
    elif len(relevant) < len(summaries):
        console.print(f"[green]Using {len(relevant)} of {len(summaries)} episodes matching '{topic}'[/green]")

    _display_sources(relevant)

    writer = BlogWriter(
        db=db,
        llm_provider=llm_config['provider'],
        llm_model=llm_config['model'],
        target_grade=target_grade,
        llm_client=llm_client
    )

    console.print(f"\n[blue]Generating blog post: '{topic}'[/blue]")
    if prompt:
        console.print(f"  Instructions: {prompt}")
    console.print(f"  Target grade: {target_grade}/100")

    with console.status("[bold green]Writing and grading blog post..."):
        blog_result = writer.generate_blog_post_from_digest(topic, relevant, extra_instructions=prompt)

    _record_source_usage(relevant, content_type='blog')
    console.print(f"\n[green]Blog post generated![/green]")
    console.print(f"  Final Grade: {blog_result['final_grade']} ({blog_result['final_score']}/100)")
    console.print(f"  Iterations: {len(blog_result['iterations'])}")
    console.print(f"  Sources: {len(relevant)} episode(s)")

    # Word count and reading time
    word_count = len(blog_result['final_post'].split())
    reading_time = max(1, round(word_count / 238))  # avg adult reading speed
    console.print(f"  Words: {word_count} (~{reading_time} min read)")

    # Show grade progression if multiple iterations
    if len(blog_result['iterations']) > 1:
        grades = [f"{it['grade']}({it['score']})" for it in blog_result['iterations']]
        console.print(f"  Grade progression: {' -> '.join(grades)}")

    file_path = writer.save_blog_post(blog_result)
    console.print(f"  Saved to: {file_path}")

    console.print(f"\n[blue]Generating social media posts...[/blue]")
    social_posts = writer.generate_social_posts(blog_result)

    console.print("\n[cyan]Twitter Posts:[/cyan]")
    for i, post in enumerate(social_posts['twitter'], 1):
        console.print(f"  {i}. {post}")

    console.print("\n[cyan]LinkedIn Posts:[/cyan]")
    for i, post in enumerate(social_posts['linkedin'], 1):
        console.print(f"  {i}. {post[:100]}...")

    console.print(f"\n[cyan]Blog Post Preview:[/cyan]")
    console.print("-" * 50)
    console.print(f"{blog_result['final_post'][:500]}...")
    console.print("-" * 50)
    console.print(f"[green]Complete post saved to: {file_path}[/green]")

    # Interactive refinement
    if refine:
        refined_post = _refine_loop(
            llm_client, original_prompt=None,
            original_result=blog_result['final_post'],
            result_key='blog post'
        )
        if refined_post != blog_result['final_post']:
            blog_result['final_post'] = refined_post
            file_path = writer.save_blog_post(blog_result)
            console.print(f"[green]Refined post saved to: {file_path}[/green]")


@main.command()
@click.option('--topic', required=True, help='Tweet topic/angle')
@click.option('--prompt', default=None, help='Additional instructions for structure, tone, or focus')
@click.option('--date', help='Date (YYYY-MM-DD), defaults to today')
@click.option('--lookback-days', default=None, type=int, help='Use episodes from the last N days instead of a single date')
@click.option('--limit', default=None, type=int, help='Number of episodes to use (default: all)')
@click.option('--thread', 'is_thread', is_flag=True, default=False, help='Generate a multi-tweet thread instead of a single tweet')
@click.option('--style', default='educational', type=click.Choice(['educational', 'contrarian', 'hot-take', 'data-driven']), help='Thread style (only used with --thread)')
@click.option('--max-tweets', default=10, type=int, help='Max tweets in thread (only used with --thread)')
@click.option('--refine', is_flag=True, default=False, help='Enter interactive refinement loop after generation')
@click.pass_context
def tweet(ctx, topic, prompt, date, lookback_days, limit, is_thread, style, max_tweets, refine):
    """Generate a tweet or Twitter thread from podcast summaries.

    By default, generates a single standalone tweet that distills the most
    valuable insight. Use --thread for a multi-tweet thread.

    \b
    Thread Styles (with --thread):
      educational  - Numbered insights and learnings
      contrarian   - Challenges conventional wisdom
      hot-take     - Bold, controversial takes
      data-driven  - Stats, numbers, valuations

    \b
    Examples:
      p3 tweet --topic "AI Trends"
      p3 tweet --topic "AI Trends" --lookback-days 7
      p3 tweet --topic "AI" --prompt "Focus on the GPU supply chain angle"
      p3 tweet --topic "AI Trends" --thread --style educational
      p3 tweet --topic "Why VCs are wrong" --thread --style contrarian --lookback-days 14
    """
    config = load_config(ctx.obj['config_path'])
    db = ctx.obj['db']

    settings = config.get('settings', {})
    llm_config = prepare_llm_config(settings)
    llm_client = LLMClient.from_config(llm_config)

    console.print(f"[blue]LLM Settings:[/blue]")
    console.print(f"  Provider: {llm_config['provider']}")
    console.print(f"  Model: {llm_config['model']}")

    if llm_config['provider'] == 'ollama':
        if not ModelManager.ensure_model_available(llm_config['model'], console):
            console.print(f"[red]Failed to ensure model {llm_config['model']} is available[/red]")
            return

    # Get summaries - either by lookback days or single date
    if lookback_days:
        summaries = _get_unified_summaries(db, lookback_days)
        date_label = f"last {lookback_days} days"
    else:
        if date:
            try:
                target_date = datetime.strptime(date, '%Y-%m-%d')
            except ValueError:
                console.print("[red]Invalid date format. Use YYYY-MM-DD[/red]")
                return
        else:
            target_date = datetime.now()
        summaries = db.get_summaries_by_date(target_date)
        date_label = str(target_date.date())

    if not summaries:
        console.print(f"[yellow]No summaries found for {date_label}[/yellow]")
        console.print("\n[cyan]To generate tweets, you need summaries first:[/cyan]")
        console.print("  1. Add podcasts: Edit [yellow]config/feeds.yaml[/yellow]")
        console.print("  2. Run pipeline: [yellow]p3 auto[/yellow]")
        if not lookback_days:
            console.print("  Or try: [yellow]p3 tweet --topic \"...\" --lookback-days 7[/yellow]")
        return

    if limit and limit > 0:
        summaries = summaries[:limit]

    console.print(f"\n[blue]Found {len(summaries)} episodes from {date_label}[/blue]")

    # Filter for relevance using stored metadata (no LLM call)
    relevant = _filter_relevant_summaries(summaries, topic)

    if not relevant:
        console.print(f"[yellow]No episodes matched '{topic}'. Using all {len(summaries)} episodes.[/yellow]")
        relevant = summaries
    elif len(relevant) < len(summaries):
        console.print(f"[green]Using {len(relevant)} of {len(summaries)} episodes matching '{topic}'[/green]")

    _display_sources(relevant)

    generator = ThreadGenerator(llm_model=llm_config['model'], llm_client=llm_client)

    if is_thread:
        # Thread mode
        console.print(f"\n[blue]Mode: Thread | Style: {style} | Max tweets: {max_tweets}[/blue]")
        if prompt:
            console.print(f"  Instructions: {prompt}")

        with console.status("[bold green]Generating thread..."):
            result = generator.generate_thread(relevant, topic, style, max_tweets, extra_instructions=prompt)

        if 'error' in result:
            console.print(f"[red]{result['error']}[/red]")
            return

        _record_source_usage(relevant, content_type='thread')
        file_path = generator.save_tweet(result)

        console.print(f"\n[green]Twitter thread generated![/green]")
        console.print(f"  Tweets: {result['tweet_count']}")
        console.print(f"  Saved to: {file_path}")

        console.print(f"\n[cyan]COPY-PASTE READY:[/cyan]")
        console.print("=" * 60)
        console.print(result['copy_paste'])
        console.print("=" * 60)

        console.print(f"\n[cyan]Individual Tweets:[/cyan]")
        for i, tw in enumerate(result['tweets'], 1):
            char_count = len(tw)
            color = "green" if char_count <= 280 else "red"
            console.print(f"\n[bold]Tweet {i}:[/bold] [{color}]({char_count} chars)[/{color}]")
            console.print(tw)

        console.print(f"\n[green]Thread saved to: {file_path}[/green]")
        console.print("[yellow]Tip: Copy the 'COPY-PASTE READY' section and paste directly into Twitter[/yellow]")

        # Interactive refinement for thread
        if refine:
            refined_thread = _refine_loop(
                llm_client, original_prompt=None,
                original_result=result['copy_paste'],
                result_key='thread'
            )
            if refined_thread != result['copy_paste']:
                result['copy_paste'] = refined_thread
                result['tweets'] = [t.strip() for t in refined_thread.split('\n\n') if t.strip()]
                result['tweet_count'] = len(result['tweets'])
                file_path = generator.save_tweet(result)
                console.print(f"\n[cyan]REFINED THREAD:[/cyan]")
                console.print("=" * 60)
                console.print(refined_thread)
                console.print("=" * 60)
                console.print(f"[green]Saved to: {file_path}[/green]")
    else:
        # Single tweet mode (default)
        console.print(f"\n[blue]Mode: Single tweet[/blue]")
        if prompt:
            console.print(f"  Instructions: {prompt}")

        with console.status("[bold green]Generating tweet..."):
            result = generator.generate_single_tweet(relevant, topic, extra_instructions=prompt)

        if 'error' in result:
            console.print(f"[red]{result['error']}[/red]")
            return

        _record_source_usage(relevant, content_type='tweet')
        file_path = generator.save_tweet(result)

        char_count = result['char_count']
        color = "green" if char_count <= 280 else "red"

        console.print(f"\n[green]Tweet generated![/green]")
        console.print(f"  Characters: [{color}]{char_count}/280[/{color}]")
        console.print(f"  Sources: {len(relevant)} episode(s)")
        console.print(f"  Saved to: {file_path}")

        console.print(f"\n[cyan]COPY-PASTE READY:[/cyan]")
        console.print("=" * 60)
        console.print(result['tweet'])
        console.print("=" * 60)

        console.print(f"\n[yellow]Tip: Use --thread for a multi-tweet thread[/yellow]")

        # Interactive refinement for single tweet
        if refine:
            refined_tweet = _refine_loop(
                llm_client, original_prompt=None,
                original_result=result['tweet'],
                result_key='tweet'
            )
            if refined_tweet != result['tweet']:
                refined_tweet = refined_tweet.strip().strip('"').strip("'").strip()
                if len(refined_tweet) > 280:
                    refined_tweet = refined_tweet[:277] + '...'
                result['tweet'] = refined_tweet
                result['char_count'] = len(refined_tweet)
                file_path = generator.save_tweet(result)
                char_count = result['char_count']
                color = "green" if char_count <= 280 else "red"
                console.print(f"\n[cyan]REFINED TWEET:[/cyan]")
                console.print("=" * 60)
                console.print(result['tweet'])
                console.print("=" * 60)
                console.print(f"  Characters: [{color}]{char_count}/280[/{color}]")
                console.print(f"[green]Saved to: {file_path}[/green]")


@main.command()
@click.option('--lookback-days', default=7, type=int, help='Days to look back (default: 7)')
@click.option('--count', default=10, type=int, help='Number of ideas to generate (default: 10)')
@click.option('--format', 'output_format', default='all', type=click.Choice(['blog', 'tweet', 'thread', 'thesis', 'all']), help='Suggest ideas for a specific format (default: all)')
@click.option('--go', default=None, type=int, help='Execute idea N from the last ideas run (e.g. --go 3)')
@click.option('--list', 'list_cached', is_flag=True, default=False, help='Show cached ideas from last run without regenerating')
@click.option('--refresh', is_flag=True, default=False, help='Force regenerate ideas even if cache exists')
@click.pass_context
def ideas(ctx, lookback_days, count, output_format, go, list_cached, refresh):
    """Find content ideas and topics from recent podcast episodes.

    Analyzes recent summaries to surface interesting topics, trends,
    and angles for tweets, blog posts, threads, or thesis deep-dives.

    Use --go N to immediately execute an idea from the last run.

    \b
    Examples:
      p3 ideas                                    # Ideas from last 7 days
      p3 ideas --lookback-days 30                 # Ideas from last month
      p3 ideas --lookback-days 14 --format tweet  # Tweet ideas only
      p3 ideas --go 3                             # Execute idea #3
      p3 ideas --list                             # Show cached ideas
      p3 ideas --refresh                          # Force regenerate
    """
    import json as _json

    config = load_config(ctx.obj['config_path'])
    db = ctx.obj['db']
    ideas_file = Path("data/.ideas_cache.json")

    # Handle --list: show cached ideas without regenerating
    if list_cached:
        if not ideas_file.exists():
            console.print("[yellow]No cached ideas. Run 'p3 ideas' to generate.[/yellow]")
            return

        cached = _json.loads(ideas_file.read_text(encoding='utf-8'))
        cached_ideas = cached.get('ideas', [])
        generated_at = cached.get('generated_at', 'unknown')
        ep_count = cached.get('episode_count', '?')
        cached_lb = cached.get('lookback_days', '?')

        # Show age of cache
        age_str = ""
        try:
            gen_time = datetime.fromisoformat(generated_at)
            delta = datetime.now() - gen_time
            if delta.days > 0:
                age_str = f" ({delta.days}d ago)"
            elif delta.seconds >= 3600:
                age_str = f" ({delta.seconds // 3600}h ago)"
            else:
                age_str = f" ({delta.seconds // 60}m ago)"
        except (ValueError, TypeError):
            pass

        console.print(f"[green]Cached Ideas ({len(cached_ideas)} ideas from {ep_count} episodes, last {cached_lb} days){age_str}:[/green]\n")

        for i, idea in enumerate(cached_ideas, 1):
            fmt = idea.get('format', 'tweet').upper()
            title = idea.get('title', 'Untitled')
            angle = idea.get('angle', '')
            sources = idea.get('sources', [])

            console.print(f"  [bold]{i}.[/bold] [{fmt}] {title}")
            if angle:
                console.print(f"     Angle: {angle}")
            if sources:
                console.print(f"     Sources: {', '.join(sources[:3])}")
            console.print()

        console.print(f"[cyan]To execute:[/cyan] p3 ideas --go <number>")
        console.print(f"[cyan]To refresh:[/cyan] p3 ideas --refresh")
        return

    # Handle --go N: load cached ideas and execute
    if go is not None:
        if not ideas_file.exists():
            console.print("[red]No cached ideas found. Run 'p3 ideas' first.[/red]")
            return

        cached = _json.loads(ideas_file.read_text(encoding='utf-8'))
        cached_ideas = cached.get('ideas', [])
        if go < 1 or go > len(cached_ideas):
            console.print(f"[red]Invalid idea number {go}. Valid range: 1-{len(cached_ideas)}[/red]")
            return

        idea = cached_ideas[go - 1]
        idea_format = idea.get('format', 'tweet').lower()
        idea_topic = idea.get('title', '')
        idea_angle = idea.get('angle', '')
        idea_sources = idea.get('sources', [])
        cached_lookback = int(cached.get('lookback_days', 7))

        console.print(f"[green]Executing idea #{go}:[/green]")
        console.print(f"  [{idea_format.upper()}] {idea_topic}")
        console.print(f"  Angle: {idea_angle}")
        if idea_sources:
            console.print(f"  Sources: {', '.join(idea_sources[:5])}")
        console.print()

        # Build the prompt from the idea's angle (None if empty so it's skipped downstream)
        extra_prompt = idea_angle or None

        if idea_format == 'blog':
            ctx.invoke(write, topic=idea_topic, prompt=extra_prompt,
                       date=None, lookback_days=cached_lookback,
                       target_grade=91.0, limit=None)
        elif idea_format == 'thesis':
            ctx.invoke(thesis, company=None, sector=idea_topic,
                       lookback_days=cached_lookback, llm_model=None,
                       list_companies=False, list_sectors=False)
        elif idea_format == 'thread':
            ctx.invoke(tweet, topic=idea_topic, prompt=extra_prompt,
                       date=None, lookback_days=cached_lookback,
                       limit=None, is_thread=True, style='educational', max_tweets=10)
        else:
            # Default to single tweet
            ctx.invoke(tweet, topic=idea_topic, prompt=extra_prompt,
                       date=None, lookback_days=cached_lookback,
                       limit=None, is_thread=False, style='educational', max_tweets=10)
        return

    settings = config.get('settings', {})
    llm_config = prepare_llm_config(settings)
    llm_client = LLMClient.from_config(llm_config)

    if llm_config['provider'] == 'ollama':
        if not ModelManager.ensure_model_available(llm_config['model'], console):
            console.print(f"[red]Failed to ensure model {llm_config['model']} is available[/red]")
            return

    summaries = _get_unified_summaries(db, lookback_days)

    if not summaries:
        console.print(f"[yellow]No summaries found in the last {lookback_days} days[/yellow]")
        console.print("Run [yellow]p3 auto[/yellow] to process podcasts first")
        return

    # Count source types
    podcast_count = sum(1 for s in summaries if s.get('source_type') != 'newsletter')
    newsletter_count = sum(1 for s in summaries if s.get('source_type') == 'newsletter')
    source_label = f"{podcast_count} episodes"
    if newsletter_count > 0:
        source_label += f" + {newsletter_count} articles"

    console.print(f"[blue]Analyzing {source_label} from the last {lookback_days} days...[/blue]\n")

    # Detect convergence (topics mentioned by 3+ independent sources)
    convergence = _detect_convergence(summaries)
    if convergence:
        console.print(f"[bold green]STRONG SIGNALS ({len(convergence)} topics from 3+ sources):[/bold green]")
        for topic, sources in list(convergence.items())[:5]:
            console.print(f"  [bold]{topic}[/bold] ({len(sources)} sources: {', '.join(sources[:4])})")
        console.print()

    # Collect data from summaries
    all_topics, all_themes, all_insights, all_companies, all_takeaways = [], [], [], [], []
    podcast_episodes = []

    for s in summaries:
        all_topics.extend(s.get('key_topics', []))
        all_themes.extend(s.get('themes', []))
        all_insights.extend(s.get('key_insights', []))
        all_companies.extend(s.get('startups', []))
        all_takeaways.extend(s.get('actionable_takeaways', []))
        podcast_episodes.append(f"{s.get('podcast_title', 'Unknown')}: {s.get('episode_title', 'Unknown')}")

    def dedupe(lst):
        seen = set()
        return [x for x in lst if x and x.lower().strip() not in seen and not seen.add(x.lower().strip())]

    all_topics = dedupe(all_topics)
    all_themes = dedupe(all_themes)
    all_companies = dedupe(all_companies)

    # Build format-specific guidance
    format_map = {
        'blog': 'For each idea, set format to "blog". Ideas should suit a 500-word analytical post.',
        'tweet': 'For each idea, set format to "tweet". Ideas should distill one amazing insight into a single tweet.',
        'thread': 'For each idea, set format to "thread". Ideas should have a strong hook and 5-10 supporting points.',
        'thesis': 'For each idea, set format to "thesis". Ideas should be about a specific company, sector, or trend.',
        'all': 'For each idea, choose the best format: "tweet", "blog", "thread", or "thesis". Use "tweet" for punchy single insights, "blog" for analytical posts, "thread" for multi-point explorations, "thesis" for company/sector deep-dives.'
    }
    format_guidance = format_map.get(output_format, format_map['all'])

    # Build convergence context for the LLM
    convergence_text = ""
    if convergence:
        convergence_lines = []
        for topic, sources in list(convergence.items())[:10]:
            convergence_lines.append(f"- {topic} (mentioned by {len(sources)} sources: {', '.join(sources[:4])})")
        convergence_text = f"""
STRONG SIGNALS - Topics mentioned by 3+ independent sources (prioritize these):
{chr(10).join(convergence_lines)}
Mark ideas based on strong signals with "[STRONG SIGNAL]" at the start of their title.

"""

    prompt = f"""Based on recent podcast episodes and newsletters, suggest {count} content ideas.

Source episodes ({len(summaries)}):
{chr(10).join(podcast_episodes[:20])}
{convergence_text}
Recurring topics: {', '.join(all_topics[:20])}
Themes: {', '.join(all_themes[:15])}
Companies mentioned: {', '.join(all_companies[:20])}

Key insights across episodes:
{chr(10).join('- ' + i for i in all_insights[:15])}

Actionable takeaways:
{chr(10).join('- ' + t for t in all_takeaways[:10])}

{format_guidance}

Requirements:
- Find patterns, contrasts, and trends ACROSS episodes
- Prioritize ideas where multiple podcasts discussed the same topic from different angles
- Include specific angles/hooks, not generic topics
- NEVER use emojis, emoji unicode characters, or emoji-like symbols anywhere in any field
- Use plain ASCII text only for all titles, angles, and source names

Return a JSON array of exactly {count} ideas. Each idea must be a JSON object with these exact fields:
{{"format": "tweet"|"blog"|"thread"|"thesis", "title": "Short title in plain text", "angle": "One sentence describing the specific take/hook in plain text", "sources": ["Podcast: Episode", ...]}}

Return ONLY the JSON array, no other text."""

    with console.status("[bold green]Generating content ideas..."):
        result = llm_client.chat(
            messages=[
                {"role": "system", "content": "You are a content strategist. Return only valid JSON arrays. Never use emojis or unicode symbols in your output - use plain ASCII text only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=3000
        )

    if not result:
        console.print("[red]Failed to generate ideas[/red]")
        return

    # Parse structured JSON from response
    parsed_ideas = None
    try:
        # Find JSON array in response
        import re as _re
        json_match = _re.search(r'\[.*\]', result, _re.DOTALL)
        if json_match:
            parsed_ideas = _json.loads(json_match.group())
    except (_json.JSONDecodeError, ValueError):
        pass

    # Strip any emojis the LLM may have included despite instructions
    if parsed_ideas:
        _emoji_pattern = _re.compile(
            "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251"
            "\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF"
            "\U00002600-\U000026FF\U0000FE00-\U0000FE0F\U0000200D]+",
            flags=_re.UNICODE
        )
        for idea in parsed_ideas:
            for key in ('title', 'angle'):
                if key in idea and isinstance(idea[key], str):
                    idea[key] = _emoji_pattern.sub('', idea[key]).strip()

    if not parsed_ideas:
        # Fallback: display raw output
        console.print(f"[green]Content Ideas (from {len(summaries)} episodes, last {lookback_days} days):[/green]\n")
        console.print(result)
        console.print(f"\n[cyan]Quick start:[/cyan]")
        console.print(f"  p3 tweet --topic \"<idea>\" --lookback-days {lookback_days}")
        console.print(f"  p3 write --topic \"<idea>\" --lookback-days {lookback_days}")
        console.print(f"  p3 thesis --sector \"<topic>\" --lookback-days {lookback_days}")
        return

    # Save to cache for --go
    ideas_file.parent.mkdir(parents=True, exist_ok=True)
    ideas_file.write_text(_json.dumps({
        'ideas': parsed_ideas,
        'lookback_days': lookback_days,
        'episode_count': len(summaries),
        'generated_at': datetime.now().isoformat()
    }, indent=2), encoding='utf-8')

    # Display structured ideas
    console.print(f"[green]Content Ideas ({len(parsed_ideas)} ideas from {len(summaries)} episodes, last {lookback_days} days):[/green]\n")

    for i, idea in enumerate(parsed_ideas, 1):
        fmt = idea.get('format', 'tweet').upper()
        title = idea.get('title', 'Untitled')
        angle = idea.get('angle', '')
        sources = idea.get('sources', [])

        console.print(f"  [bold]{i}.[/bold] [{fmt}] {title}")
        if angle:
            console.print(f"     Angle: {angle}")
        if sources:
            console.print(f"     Sources: {', '.join(sources[:3])}")
        console.print()

    console.print(f"[cyan]To execute an idea:[/cyan]")
    console.print(f"  p3 ideas --go <number>                    # Run with defaults")
    console.print(f"  p3 ideas --go 3                           # Execute idea #3")
    console.print(f"\n[cyan]Or run manually with custom options:[/cyan]")
    console.print(f"  p3 tweet --topic \"<title>\" --prompt \"<angle>\" --lookback-days {lookback_days}")
    console.print(f"  p3 tweet --topic \"<title>\" --thread --lookback-days {lookback_days}")
    console.print(f"  p3 write --topic \"<title>\" --prompt \"<angle>\" --lookback-days {lookback_days}")
    console.print(f"  p3 thesis --sector \"<topic>\" --lookback-days {lookback_days}")


@main.command()
@click.pass_context
def init(ctx):
    """Initialize P³ configuration and directories."""
    console.print("[blue]Initializing P³...[/blue]")

    # Create directories
    dirs = [
        'data', 'config', 'logs', 'data/audio', 'exports',
        'outputs/digests', 'outputs/blogs', 'outputs/tweets', 'outputs/thesis', 'outputs/transcripts'
    ]
    for dir_name in dirs:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        console.print(f"✓ Created directory: {dir_name}")

    # Copy example config if it doesn't exist
    config_path = Path("config/feeds.yaml")
    example_path = Path("config/feeds.yaml.example")

    if not config_path.exists() and example_path.exists():
        import shutil
        shutil.copy(example_path, config_path)
        console.print("✓ Created config/feeds.yaml from example")

    # Initialize database
    db = P3Database("data/p3.duckdb")
    db.close()
    console.print("✓ Initialized database")

    console.print("\n[green]✅ P³ initialized successfully![/green]")
    console.print("\n[cyan]📋 Quick Start Guide:[/cyan]")
    console.print("1. Edit [yellow]config/feeds.yaml[/yellow] with your podcast RSS feeds")
    console.print("2. (Optional) Add newsletters: [yellow]p3 newsletter add <substack-url>[/yellow]")
    console.print("3. Run automated pipeline: [yellow]p3 auto[/yellow]")
    console.print("\n[cyan]💡 Or run steps individually:[/cyan]")
    console.print("  p3 fetch          # Download episodes")
    console.print("  p3 transcribe     # Convert audio to text")
    console.print("  p3 digest         # Generate summaries")
    console.print("  p3 export         # Export to markdown/JSON")
    console.print("\n[cyan]📂 Outputs will be saved to:[/cyan]")
    console.print("  outputs/digests/  - Daily summaries")
    console.print("  outputs/blogs/    - Blog posts (p3 write)")
    console.print("  outputs/tweets/   - Tweets and threads (p3 tweet)")
    console.print("  outputs/thesis/   - Company analyses (p3 thesis)")
    console.print("\n[yellow]💡 Tip: Run 'p3 models' to check available LLM models[/yellow]")


@main.command()
@click.option('--company', help='Company to analyze')
@click.option('--sector', help='Sector/theme to analyze (e.g., "AI", "SaaS")')
@click.option('--lookback-days', default=30, type=int, help='Days to search (default: 30)')
@click.option('--llm-model', default=None, help='LLM model to use')
@click.option('--list-companies', is_flag=True, help='List companies mentioned')
@click.option('--list-sectors', is_flag=True, help='List topics/sectors discussed')
@click.option('--refine', is_flag=True, default=False, help='Enter interactive refinement loop after generation')
@click.pass_context
def thesis(ctx, company, sector, lookback_days, llm_model, list_companies, list_sectors, refine):
    """Generate company/sector analysis for interview prep.

    Analyzes full podcast transcripts to help you prepare for job interviews,
    speak intelligently about companies, and understand market dynamics.

    \b
    Examples:
      p3 thesis --list-companies               See available companies
      p3 thesis --list-sectors                 See topics discussed
      p3 thesis --company "OpenAI"             Analyze a company
      p3 thesis --sector "AI"                  Analyze a sector
      p3 thesis --company "X" --llm-model llama3.1:8b   Use better model
    """
    config = load_config(ctx.obj['config_path'])
    db = ctx.obj['db']

    # Initialize thesis generator for list commands
    if list_companies or list_sectors:
        generator = ThesisGenerator(db=db, llm_model="llama3.2:3b")  # Model doesn't matter for listing

        if list_companies:
            console.print("[blue]📊 Companies mentioned in podcasts (last 90 days):[/blue]\n")
            companies = generator.get_available_companies(lookback_days=90)

            if not companies:
                console.print("[yellow]📭 No companies found in summaries[/yellow]")
                console.print("\n[cyan]💡 To analyze companies, you need summaries first:[/cyan]")
                console.print("  1. Add podcasts: Edit [yellow]config/feeds.yaml[/yellow]")
                console.print("  2. Run pipeline: [yellow]p3 auto[/yellow]")
                console.print("  Or run steps: [yellow]p3 fetch[/yellow] → [yellow]p3 transcribe[/yellow] → [yellow]p3 digest[/yellow]")
                return

            # Display in columns
            from rich.columns import Columns
            console.print(Columns(companies, equal=True, expand=True))
            console.print(f"\n[green]Total: {len(companies)} companies[/green]")
            console.print("\n[cyan]💡 Tip: Use 'p3 thesis --company \"Company Name\"' to generate analysis[/cyan]")
            return

        if list_sectors:
            console.print("[blue]📈 Topics/Sectors discussed in podcasts (last 90 days):[/blue]\n")
            sectors = generator.get_available_sectors(lookback_days=90)

            if not sectors:
                console.print("[yellow]No topics found in summaries[/yellow]")
                console.print("Run 'p3 digest' first to generate summaries")
                return

            # Display as table with counts
            sector_table = Table(title="Topics & Mention Count")
            sector_table.add_column("Topic", style="cyan")
            sector_table.add_column("Mentions", style="green", justify="right")

            for topic, count in list(sectors.items())[:30]:  # Show top 30
                sector_table.add_row(topic, str(count))

            console.print(sector_table)
            console.print(f"\n[green]Showing top 30 of {len(sectors)} topics[/green]")
            console.print("\n[cyan]💡 Tip: Use 'p3 thesis --sector \"Topic\"' to generate analysis[/cyan]")
            return

    if not company and not sector:
        console.print("[red]Error: Must specify either --company, --sector, --list-companies, or --list-sectors[/red]")
        console.print("\nExamples:")
        console.print("  p3 thesis --list-companies       # See available companies")
        console.print("  p3 thesis --list-sectors         # See available topics")
        console.print("  p3 thesis --company 'OpenAI'")
        console.print("  p3 thesis --sector 'AI'")
        return

    if company and sector:
        console.print("[red]Error: Cannot specify both --company and --sector[/red]")
        return

    settings = config.get('settings', {})
    llm_config = prepare_llm_config(settings, llm_model=llm_model)
    llm_client = LLMClient.from_config(llm_config)

    # Display model being used
    console.print(f"[blue]LLM Settings:[/blue]")
    console.print(f"  Provider: {llm_config['provider']}")
    console.print(f"  Model: {llm_config['model']}")

    # Auto-download Ollama model if needed
    if llm_config['provider'] == 'ollama':
        if not ModelManager.ensure_model_available(llm_config['model'], console):
            console.print(f"[red]✗ Failed to ensure model {llm_config['model']} is available[/red]")
            return

    # Initialize thesis generator
    generator = ThesisGenerator(db=db, llm_model=llm_config['model'], llm_client=llm_client)

    subject = company or sector
    thesis_type = "company" if company else "sector"

    console.print(f"\n[blue]Generating {thesis_type} analysis: '{subject}'[/blue]")
    console.print(f"Analysis period: Last {lookback_days} days")

    # Generate thesis
    with console.status(f"[bold green]Analyzing transcripts and generating thesis..."):
        if company:
            result = generator.generate_thesis(company, lookback_days)
        else:
            result = generator.generate_sector_thesis(sector, lookback_days)

    # Check for errors
    if "error" in result:
        console.print(f"[red]✗ {result['error']}[/red]")
        console.print("\n[yellow]Suggestions:[/yellow]")
        console.print("  - Try a longer lookback period (--lookback-days 60)")
        console.print("  - Check spelling of company/sector name")
        console.print("  - Run 'p3 digest' first to ensure summaries are generated")
        return

    # Display results
    console.print(f"\n[green]✓ Company analysis generated![/green]")

    # Show peer company warning if applicable
    if result.get('based_on_peers'):
        peers = ', '.join(result.get('peer_companies', [])[:3])
        console.print(f"[yellow]⚠️  Note: {company} was not directly mentioned. Analysis based on peer companies:[/yellow]")
        console.print(f"[yellow]   {peers}[/yellow]")

    console.print(f"Sources: {result['mention_count']} mentions across {result['episode_count']} episodes")

    thesis = result.get('thesis', {})

    # Display executive summary
    console.print(f"\n[cyan]📊 Executive Summary:[/cyan]")
    console.print(thesis.get('executive_summary', 'N/A'))

    console.print(f"\n[cyan]🎯 Confidence Level:[/cyan] {thesis.get('confidence_level', 'N/A')}")

    # Save analysis
    file_path = generator.save_thesis(result)
    console.print(f"\n[green]✓ Full analysis saved to: {file_path}[/green]")

    # Show preview of key sections
    if company:
        console.print(f"\n[cyan]💼 Key Talking Points (For Interviews):[/cyan]")
        for point in thesis.get('key_talking_points', [])[:3]:
            console.print(f"  • {point}")

        console.print(f"\n[cyan]💡 Why It's Interesting:[/cyan]")
        for point in thesis.get('why_its_interesting', [])[:3]:
            console.print(f"  • {point}")

        console.print(f"\n[cyan]⚠️  Potential Concerns:[/cyan]")
        for concern in thesis.get('potential_concerns', [])[:3]:
            console.print(f"  • {concern}")
    else:
        console.print(f"\n[cyan]🚀 Investment Opportunities (Top 3):[/cyan]")
        for opp in thesis.get('investment_opportunities', [])[:3]:
            console.print(f"  • {opp}")

        console.print(f"\n[cyan]📈 Key Trends:[/cyan]")
        for trend in thesis.get('market_trends', [])[:3]:
            console.print(f"  • {trend}")

    console.print(f"\n[yellow]💡 Tip: Read the full analysis in {file_path} for complete interview prep[/yellow]")

    # Interactive refinement
    if refine and not result.get('error'):
        # Read the saved file to get the full markdown content
        thesis_content = file_path.read_text(encoding='utf-8') if file_path.exists() else ""
        if thesis_content:
            refined_content = _refine_loop(
                llm_client, original_prompt=None,
                original_result=thesis_content,
                result_key='thesis analysis'
            )
            if refined_content != thesis_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(refined_content)
                console.print(f"[green]Refined analysis saved to: {file_path}[/green]")


@main.command()
@click.argument('question', nargs=-1, required=True)
@click.option('--lookback-days', default=90, type=int, help='Days to search back (default: 90)')
@click.option('--max-sources', default=5, type=int, help='Max transcript sources to include (default: 5)')
@click.option('--show-sources', is_flag=True, default=False, help='Show which transcript segments were used')
@click.pass_context
def ask(ctx, question, lookback_days, max_sources, show_sources):
    """Ask questions about your podcast transcripts using AI.

    Search across all transcripts and get AI-powered answers with
    source citations. Works like a research assistant for your podcasts.

    \b
    Examples:
      p3 ask "Did anyone discuss GPU supply chains?"
      p3 ask "What did Chamath say about AI regulation?"
      p3 ask "Which companies were mentioned as AI winners?"
      p3 ask "Compare views on interest rates" --lookback-days 30
      p3 ask "What are the best startup ideas mentioned?" --show-sources
    """
    config = load_config(ctx.obj['config_path'])
    db = ctx.obj['db']
    question_text = ' '.join(question)

    settings = config.get('settings', {})
    llm_config = prepare_llm_config(settings)
    llm_client = LLMClient.from_config(llm_config)

    console.print(f"[blue]Searching transcripts for:[/blue] {question_text}")
    console.print(f"  Provider: {llm_config['provider']} | Model: {llm_config['model']}")
    console.print(f"  Search window: last {lookback_days} days\n")

    # Extract search keywords from the question
    stop_words = {
        'did', 'does', 'do', 'was', 'were', 'is', 'are', 'am', 'been', 'being',
        'have', 'has', 'had', 'having', 'will', 'would', 'could', 'should', 'may',
        'might', 'shall', 'can', 'need', 'dare', 'ought', 'used', 'to', 'a', 'an',
        'the', 'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
        'neither', 'each', 'every', 'all', 'any', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'only', 'same', 'than', 'too', 'very', 'just',
        'about', 'above', 'after', 'again', 'against', 'at', 'before', 'below',
        'between', 'by', 'for', 'from', 'in', 'into', 'of', 'off', 'on', 'out',
        'over', 'through', 'under', 'until', 'up', 'with', 'what', 'which', 'who',
        'whom', 'this', 'that', 'these', 'those', 'how', 'when', 'where', 'why',
        'there', 'here', 'they', 'them', 'their', 'he', 'she', 'it', 'his', 'her',
        'its', 'we', 'us', 'our', 'you', 'your', 'my', 'me', 'i', 'ever', 'anyone',
        'say', 'said', 'talk', 'talked', 'discuss', 'discussed', 'mention', 'mentioned',
        'think', 'thought', 'tell', 'compare', 'views', 'opinion', 'opinions'
    }
    words = re.findall(r'\b[a-zA-Z0-9]+\b', question_text.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 1]

    if not keywords:
        keywords = [w for w in words if len(w) > 2]

    if not keywords:
        console.print("[yellow]Could not extract meaningful keywords from the question.[/yellow]")
        return

    console.print(f"[dim]Keywords: {', '.join(keywords)}[/dim]")

    # Search transcripts for matching segments
    lookback_days_int = int(lookback_days)
    matching_episodes = {}

    for kw in keywords:
        results = db.conn.execute("""
            SELECT DISTINCT
                e.id as episode_id,
                e.title as episode_title,
                e.date,
                p.title as podcast_title
            FROM transcripts t
            JOIN episodes e ON t.episode_id = e.id
            JOIN podcasts p ON e.podcast_id = p.id
            WHERE t.text ILIKE ?
            AND e.date >= CURRENT_DATE - CAST(? AS INTEGER) * INTERVAL 1 DAY
        """, (f"%{kw}%", lookback_days_int)).fetchall()

        for row in results:
            ep_id = row[0]
            if ep_id not in matching_episodes:
                matching_episodes[ep_id] = {
                    'episode_id': ep_id,
                    'episode_title': row[1],
                    'date': row[2],
                    'podcast_title': row[3],
                    'keyword_hits': 0
                }
            matching_episodes[ep_id]['keyword_hits'] += 1

    if not matching_episodes:
        # Fall back: try searching summaries
        console.print("[yellow]No transcript matches. Checking summaries...[/yellow]")
        summaries = _get_unified_summaries(db, lookback_days)
        if summaries:
            relevant = _filter_relevant_summaries(summaries, question_text)
            if relevant:
                context_parts = []
                for s in relevant[:max_sources]:
                    source = f"{s.get('podcast_title', 'Unknown')} - {s.get('episode_title', 'Unknown')}"
                    context_parts.append(
                        f"Source: {source}\nSummary: {s.get('full_summary', 'N/A')}\n"
                        f"Key insights: {', '.join(s.get('key_insights', []))}\n"
                    )
                context = "\n---\n".join(context_parts)
                console.print(f"[green]Found {len(relevant)} relevant summaries[/green]\n")
            else:
                console.print("[yellow]No relevant content found for this question.[/yellow]")
                return
        else:
            console.print("[yellow]No content found. Run p3 auto to process podcasts first.[/yellow]")
            return
    else:
        # Sort by keyword hits (most relevant first)
        sorted_episodes = sorted(
            matching_episodes.values(),
            key=lambda x: x['keyword_hits'],
            reverse=True
        )[:max_sources]

        console.print(f"[green]Found matches in {len(matching_episodes)} episodes, using top {len(sorted_episodes)}[/green]\n")

        # Build context from transcript segments
        context_parts = []
        for ep in sorted_episodes:
            segments = db.get_transcripts_for_episode(ep['episode_id'])
            if not segments:
                continue

            source = f"{ep['podcast_title']} - {ep['episode_title']} ({ep['date']})"

            # Find segments containing any keyword and include surrounding context
            full_text_parts = []
            segment_texts = [seg.get('text', '') for seg in segments]
            for idx, seg_text in enumerate(segment_texts):
                seg_lower = seg_text.lower()
                if any(kw in seg_lower for kw in keywords):
                    # Include this segment plus 2 before and 2 after for context
                    start = max(0, idx - 2)
                    end = min(len(segment_texts), idx + 3)
                    chunk = ' '.join(segment_texts[start:end])
                    if chunk not in full_text_parts:
                        full_text_parts.append(chunk)

            if full_text_parts:
                # Limit total context per episode
                episode_context = '\n...\n'.join(full_text_parts[:10])
                if len(episode_context) > 4000:
                    episode_context = episode_context[:4000] + '...'
                context_parts.append(f"Source: {source}\n{episode_context}")

            if show_sources:
                console.print(f"  [cyan]{source}[/cyan] ({ep['keyword_hits']} keyword hits)")

        if not context_parts:
            console.print("[yellow]Found matching episodes but couldn't extract relevant segments.[/yellow]")
            return

        context = "\n\n---\n\n".join(context_parts)

    # Also pull in structured summaries for any matching episodes
    summaries = _get_unified_summaries(db, lookback_days)
    relevant_summaries = _filter_relevant_summaries(summaries, question_text)
    if relevant_summaries:
        summary_context = "\n\nSTRUCTURED SUMMARIES:\n"
        for s in relevant_summaries[:5]:
            source = f"{s.get('podcast_title', 'Unknown')} - {s.get('episode_title', 'Unknown')}"
            summary_context += f"\n{source}:\n"
            summary_context += f"  Topics: {', '.join(s.get('key_topics', []))}\n"
            summary_context += f"  Key insights: {', '.join(s.get('key_insights', []))}\n"
            if s.get('quotes'):
                summary_context += f"  Quotes: {'; '.join(s.get('quotes', [])[:3])}\n"
        context += summary_context

    # Truncate total context if needed
    if len(context) > 25000:
        context = context[:25000] + "\n\n[Context truncated for length]"

    # Ask the LLM
    system_prompt = """You are a research assistant that answers questions based ONLY on podcast transcripts and summaries provided below.

Rules:
- ONLY use information from the provided source material. Do NOT use your general knowledge.
- For every claim or fact, cite the specific podcast and episode it came from using "(Source: Podcast - Episode)".
- If the sources don't contain enough information to answer the question, say so clearly.
- Be specific and quote directly from transcripts when possible.
- If multiple sources discuss the topic, compare their perspectives."""

    user_prompt = f"""Based on the following podcast transcripts and summaries, answer this question:

QUESTION: {question_text}

SOURCE MATERIAL:
{context}

Answer the question thoroughly, citing specific sources for each claim."""

    with console.status("[bold green]Thinking..."):
        answer = llm_client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=4000
        )

    if not answer:
        console.print("[red]Failed to generate an answer. Try again or check your LLM configuration.[/red]")
        return

    console.print("[bold green]Answer:[/bold green]\n")
    console.print(answer.strip())
    console.print()

    # Show follow-up suggestions
    console.print("[dim]Follow-up ideas:[/dim]")
    console.print("[dim]  p3 ask \"Tell me more about [specific topic from answer]\"[/dim]")
    console.print("[dim]  p3 ask \"What else did [person/podcast] say about this?\" --show-sources[/dim]")


@main.command()
@click.option('--retention-days', default=7, type=int, help='Keep audio files for this many days (default: 7)')
@click.option('--dry-run', is_flag=True, help='Show what would be deleted without deleting')
@click.option('--stats', is_flag=True, help='Show storage statistics')
@click.pass_context
def cleanup(ctx, retention_days, dry_run, stats):
    """Clean up old audio files to save disk space.

    Audio files are safe to delete after transcription since:
    - Full transcripts are stored in the database
    - Structured summaries are also stored
    - Original audio can be re-downloaded from RSS feed if needed

    By default, keeps audio files for 7 days after transcription.

    Examples:
        p3 cleanup --stats                    # Show storage stats
        p3 cleanup --dry-run                  # Preview what would be deleted
        p3 cleanup                            # Delete files older than 7 days
        p3 cleanup --retention-days 14        # Keep files for 14 days
    """
    db = ctx.obj['db']
    manager = StorageManager(db=db)

    if stats:
        console.print("[blue]📊 Storage Statistics[/blue]\n")
        stats_data = manager.get_storage_stats()

        stat_table = Table(title="Audio File Storage")
        stat_table.add_column("Metric", style="cyan")
        stat_table.add_column("Value", style="green")

        stat_table.add_row("Total Files", str(stats_data['total_files']))
        stat_table.add_row("Total Size", f"{stats_data['total_size_mb']:.2f} MB")
        stat_table.add_row("Oldest File Age", f"{stats_data['oldest_file_age_days']} days")
        stat_table.add_row("Transcribed", str(stats_data['transcribed_count']))
        stat_table.add_row("Pending Transcription", str(stats_data['pending_transcription']))
        stat_table.add_row("Directory", stats_data['audio_directory'])

        console.print(stat_table)

        # Show cleanup estimate
        console.print(f"\n[cyan]Cleanup Estimate (retention: {retention_days} days):[/cyan]")
        estimate = manager.estimate_cleanup_impact(retention_days)
        console.print(f"  Files to delete: {estimate['deleted_count']}")
        console.print(f"  Space to free: {estimate['freed_space_mb']:.2f} MB")

        if estimate['deleted_count'] > 0:
            console.print(f"\n[yellow]Run 'p3 cleanup --retention-days {retention_days}' to delete these files[/yellow]")
        return

    # Perform cleanup
    action_word = "Would delete" if dry_run else "Deleting"
    console.print(f"[blue]{action_word} audio files older than {retention_days} days...[/blue]\n")

    result = manager.cleanup_old_audio(retention_days=retention_days, dry_run=dry_run, console=console)

    if "error" in result:
        console.print(f"[red]✗ {result['error']}[/red]")
        return

    # Display results
    if result['deleted_count'] == 0:
        console.print("[green]✓ No files to delete[/green]")
        console.print(f"All audio files are newer than {retention_days} days or pending transcription")
    else:
        action_word = "Would be deleted" if dry_run else "Deleted"
        console.print(f"[green]✓ {action_word}: {result['deleted_count']} files[/green]")
        console.print(f"[green]✓ Space freed: {result['freed_space_mb']:.2f} MB[/green]")

        if dry_run:
            console.print(f"\n[yellow]This was a dry run. Run without --dry-run to actually delete files.[/yellow]")

        if result['deleted_count'] <= 10:
            console.print("\n[cyan]Files:[/cyan]")
            for file in result['deleted_files']:
                console.print(f"  • {file}")
        else:
            console.print(f"\n[cyan]First 10 files:[/cyan]")
            for file in result['deleted_files'][:10]:
                console.print(f"  • {file}")
            console.print(f"  ... and {result['deleted_count'] - 10} more")


@main.command()
@click.pass_context
def models(ctx):
    """Show available Ollama models and recommendations.

    Displays which LLM models are currently installed and which are recommended
    but not yet downloaded. Models will be auto-downloaded when you select them.
    """
    config = load_config(ctx.obj['config_path'])
    settings = config.get('settings', {})

    console.print("[bold blue]🤖 Ollama Model Status[/bold blue]\n")

    # Show default models from config
    console.print("[cyan]📋 Default Models (from config):[/cyan]")
    console.print(f"  Whisper Model: {settings.get('whisper_model', 'base')}")
    console.print(f"  Faster-Whisper: {settings.get('use_faster_whisper', False)}")
    console.print(f"  LLM Provider: {settings.get('llm_provider', 'ollama')}")
    console.print(f"  LLM Model: {settings.get('llm_model', 'llama3.2:1b')}")
    console.print("")

    # Check if Ollama is available
    if not ModelManager.check_ollama_available():
        console.print("[red]✗ Ollama not available[/red]")
        console.print("Please install Ollama from https://ollama.com")
        return

    console.print("[green]✓ Ollama is running[/green]\n")

    # Get model status
    model_status = ModelManager.get_model_status()
    default_llm = settings.get('llm_model', 'llama3.2:1b')

    # Create table for LLM models
    llm_table = Table(title="LLM Models (for summarization)")
    llm_table.add_column("Model", style="cyan")
    llm_table.add_column("Status", style="white")
    llm_table.add_column("Size", style="yellow")
    llm_table.add_column("Speed", style="blue")
    llm_table.add_column("Quality", style="magenta")
    llm_table.add_column("Use Case", style="white")

    for model_name, info in model_status.items():
        status_icon = "✓" if info['installed'] else "✗"
        status_color = "green" if info['installed'] else "red"
        status_text = f"[{status_color}]{status_icon}[/{status_color}]"

        # Add asterisk if this is the default model
        display_name = f"{model_name} ⭐" if model_name == default_llm else model_name

        llm_table.add_row(
            display_name,
            status_text,
            info['size'],
            info['speed'],
            info['quality'],
            info['use_case']
        )

    console.print(llm_table)
    console.print("[dim]⭐ = Default model from config[/dim]")

    # Create table for Whisper models
    console.print("\n")
    whisper_table = Table(title="Whisper Models (for transcription)")
    whisper_table.add_column("Model", style="cyan")
    whisper_table.add_column("Quality", style="magenta")
    whisper_table.add_column("Speed", style="blue")
    whisper_table.add_column("Size", style="yellow")

    default_whisper = settings.get('whisper_model', 'base')

    for model_name, info in ModelManager.WHISPER_MODELS.items():
        # Add asterisk if this is the default model
        display_name = f"{model_name} ⭐" if model_name == default_whisper else model_name

        whisper_table.add_row(
            display_name,
            info['quality'],
            info['speed'],
            info['size']
        )

    console.print(whisper_table)
    console.print("\n[dim]⭐ = Default model from config[/dim]")
    console.print("[dim]Note: Whisper models are auto-downloaded on first use by faster-whisper[/dim]")

    # Usage instructions
    console.print("\n[cyan]💡 Usage:[/cyan]")
    console.print("  - LLM models will be auto-downloaded when you select them")
    console.print("  - Example: [bold]p3 auto --llm-model llama3.2:3b[/bold]")
    console.print("  - Example: [bold]p3 digest --model llama3.1:8b[/bold]")
    console.print("  - Whisper models: [bold]p3 auto --whisper-model medium[/bold]")
    console.print("\n[cyan]📚 Task-Specific Recommendations:[/cyan]")

    # Create recommendation table
    rec_table = Table(title="Best Models by Task")
    rec_table.add_column("Task", style="cyan", no_wrap=True)
    rec_table.add_column("Recommended Model", style="yellow")
    rec_table.add_column("Why", style="white")

    rec_table.add_row(
        "Daily Digests",
        "llama3.2:3b",
        "Best balance of speed and quality"
    )
    rec_table.add_row(
        "Thesis Analysis",
        "llama3.1:8b",
        "More nuanced for complex analysis"
    )
    rec_table.add_row(
        "Blog Posts",
        "llama3.1:8b",
        "Better writing quality"
    )
    rec_table.add_row(
        "Tweets/Threads",
        "llama3.2:3b",
        "Fast, punchy, concise"
    )
    rec_table.add_row(
        "Transcription",
        "medium",
        "96-98% accuracy, 2-3 min/hour"
    )
    rec_table.add_row(
        "Fast Transcription",
        "small",
        "90-94% accuracy, 1 min/hour"
    )

    console.print(rec_table)

    # Show installed models
    installed = ModelManager.get_installed_ollama_models()
    console.print(f"\n[green]✓ {len([m for m in model_status.values() if m['installed']])} of {len(model_status)} recommended models installed[/green]")

    if installed:
        console.print("\n[cyan]All installed Ollama models:[/cyan]")
        for model in installed[:10]:  # Show first 10
            console.print(f"  • {model}")
        if len(installed) > 10:
            console.print(f"  ... and {len(installed) - 10} more")


@main.command()
@click.option('--send-email', is_flag=True, help='Send email notification with digest')
@click.option('--max-episodes', default=None, type=int, help='Max episodes per feed to fetch')
@click.option('--whisper-model', default=None, help='Whisper model (tiny, base, small, medium, large)')
@click.option('--llm-model', default=None, help='LLM model for summarization')
@click.option('--faster-whisper', is_flag=True, default=None, help='Use faster-whisper')
@click.option('--skip-fetch', is_flag=True, help='Skip fetching new episodes')
@click.option('--skip-transcribe', is_flag=True, help='Skip transcription step')
@click.option('--skip-digest', is_flag=True, help='Skip digest generation')
@click.option('--skip-export', is_flag=True, help='Skip export step')
@click.option('--skip-audio-digest', is_flag=True, help='Skip audio digest generation')
@click.option('--audio-digest-duration', default=None, type=int, help='Target audio digest duration in seconds (default: 1/10th of input)')
@click.option('--skip-cleanup', is_flag=True, help='Skip audio file cleanup')
@click.option('--cleanup-retention-days', default=None, type=int, help='Days to keep audio files (default: 7, configurable in feeds.yaml)')
@click.option('--include-newsletters/--no-newsletters', default=True, help='Process newsletters (default: True)')
@click.option('--max-newsletter-articles', default=10, type=int, help='Max articles per newsletter (default: 10)')
@click.pass_context
def auto(ctx, send_email, max_episodes, whisper_model, llm_model, faster_whisper,
         skip_fetch, skip_transcribe, skip_digest, skip_export, skip_audio_digest, audio_digest_duration,
         skip_cleanup, cleanup_retention_days, include_newsletters, max_newsletter_articles):
    """Run complete automated pipeline: fetch→transcribe→digest→export→audio-digest→cleanup.

    Perfect for scheduling with Windows Task Scheduler to process podcasts
    automatically. Override config settings with command-line options.

    \b
    Examples:
      p3 auto                              Use config settings
      p3 auto --send-email                 Run and send email
      p3 auto --whisper-model medium       Override transcription model
      p3 auto --llm-model llama3.2:3b      Override LLM model
      p3 auto --max-episodes 5             Limit episodes per feed
      p3 auto --skip-fetch                 Process existing episodes only
    """
    config = load_config(ctx.obj['config_path'])
    db = ctx.obj['db']
    settings = config.get('settings', {})

    console.print("[bold blue]🤖 P³ Automated Pipeline Starting...[/bold blue]\n")

    total_downloaded = 0
    total_transcribed = 0
    total_processed = 0
    transcriber = None

    # Step 1: Fetch new episodes
    if not skip_fetch:
        console.print("[cyan]📥 Step 1/6: Fetching new episodes...[/cyan]")
        feeds = config.get('feeds', [])
        max_eps = max_episodes or settings.get('max_episodes_per_feed', 10)

        downloader = PodcastDownloader(
            db=db,
            max_episodes=max_eps,
            audio_format=settings.get('audio_format', 'wav')
        )

        results = downloader.fetch_all_feeds(feeds)
        total_downloaded = sum(results.values())
        console.print(f"[green]✓ Downloaded {total_downloaded} new episodes[/green]\n")
    else:
        console.print("[yellow]⏭ Skipping fetch step[/yellow]\n")

    # Step 2: Transcribe
    if not skip_transcribe:
        console.print("[cyan]🎙 Step 2/6: Transcribing audio...[/cyan]")

        # Use command-line options if provided, otherwise use config
        whisper_model_to_use = whisper_model or settings.get('whisper_model', 'base')
        use_parakeet = settings.get('parakeet_enabled', False)
        use_faster_whisper_flag = faster_whisper if faster_whisper is not None else settings.get('use_faster_whisper', False)

        # Prepare transcription API configuration
        trans_config = prepare_transcription_config(settings)

        if trans_config['use_api']:
            console.print(f"  Provider: {trans_config['api_provider']} API")
            console.print(f"  Model: {trans_config['api_model']}")
        else:
            console.print(f"  Model: {whisper_model_to_use}")
            console.print(f"  Faster-whisper: {use_faster_whisper_flag}")

        transcriber = AudioTranscriber(
            db=db,
            whisper_model=whisper_model_to_use,
            use_parakeet=use_parakeet,
            parakeet_model=settings.get('parakeet_model', 'mlx-community/parakeet-tdt-0.6b-v2'),
            use_faster_whisper=use_faster_whisper_flag,
            use_api=trans_config['use_api'],
            api_provider=trans_config['api_provider'],
            api_model=trans_config['api_model'],
            api_fallback_model=trans_config.get('api_fallback_model'),
            api_fallback_models=trans_config.get('api_fallback_models'),
            api_key_file=trans_config['api_key_file'],
            api_base_url=trans_config['api_base_url'],
            wait_for_ratelimit=trans_config['wait_for_ratelimit'],
            max_wait_retries=trans_config['max_wait_retries']
        )

        episodes = db.get_episodes_by_status('downloaded')
        for episode in track(episodes, description="Transcribing..."):
            if transcriber.transcribe_episode(episode['id']):
                total_transcribed += 1

        console.print(f"[green]✓ Transcribed {total_transcribed} episodes[/green]\n")
    else:
        console.print("[yellow]⏭ Skipping transcribe step[/yellow]\n")

    # Step 3: Generate summaries
    if not skip_digest:
        console.print("[cyan]🧠 Step 3/6: Generating summaries...[/cyan]")

        # Prepare LLM configuration (handles external APIs if enabled)
        llm_config = prepare_llm_config(settings, llm_model)

        console.print(f"  Provider: {llm_config['provider']}")
        console.print(f"  Model: {llm_config['model']}")

        # Auto-download Ollama model if needed
        if llm_config['provider'] == 'ollama':
            if not ModelManager.ensure_model_available(llm_config['model'], console):
                console.print(f"[red]✗ Failed to ensure model {llm_config['model']} is available[/red]")
                console.print("[yellow]Pipeline stopped at digest step[/yellow]")
                return

        cleaner = TranscriptCleaner(
            db=db,
            llm_provider=llm_config['provider'],
            llm_model=llm_config['model'],
            ollama_base_url=llm_config['ollama_base_url'],
            ollama_model=llm_config['ollama_model'],
            model_fallbacks=llm_config.get('model_fallbacks'),
            api_key_file=llm_config['api_key_file'],
            base_url=llm_config['base_url'],
            timeout=llm_config['timeout'],
            max_retries=llm_config['max_retries'],
            transcriber=transcriber
        )

        total_processed = cleaner.process_all_transcribed()
        console.print(f"[green]✓ Processed {total_processed} episodes[/green]\n")
    else:
        console.print("[yellow]⏭ Skipping digest step[/yellow]\n")

    # Newsletter processing
    total_newsletter_fetched = 0
    total_newsletter_processed = 0

    if include_newsletters and not skip_fetch:
        console.print("[cyan]📰 Fetching newsletter articles...[/cyan]")
        fetcher = SubstackFetcher(db)
        total_newsletter_fetched = fetcher.fetch_articles(max_articles=max_newsletter_articles)
        console.print(f"[green]✓ Fetched {total_newsletter_fetched} new newsletter articles[/green]\n")

    if include_newsletters and not skip_digest:
        console.print("[cyan]📝 Processing newsletter articles...[/cyan]")

        # Use same LLM settings as podcasts (reuse the llm_config from earlier)
        llm_config = prepare_llm_config(settings, llm_model)

        console.print(f"  Provider: {llm_config['provider']}")
        console.print(f"  Model: {llm_config['model']}")

        nl_llm_client = LLMClient.from_config(llm_config)
        summarizer = NewsletterSummarizer(
            db=db,
            llm_provider=llm_config['provider'],
            llm_model=llm_config['model'],
            ollama_base_url=llm_config.get('ollama_base_url', 'http://localhost:11434'),
            api_key_file=llm_config.get('api_key_file'),
            base_url=llm_config.get('base_url'),
            timeout=llm_config.get('timeout', 120),
            max_retries=llm_config.get('max_retries', 3),
            model_fallbacks=llm_config.get('model_fallbacks'),
            llm_client=nl_llm_client
        )

        total_newsletter_processed = summarizer.process_all_fetched()
        console.print(f"[green]✓ Processed {total_newsletter_processed} newsletter articles[/green]\n")

    # Step 4: Export
    if not skip_export:
        console.print("[cyan]📤 Step 4/6: Exporting digests...[/cyan]")
        target_date = datetime.now()
        exporter = DigestExporter(db)
        summaries = db.get_summaries_by_date(target_date)

        if summaries:
            formats = settings.get('export_format', ['markdown', 'json'])
            markdown_file = None
            json_file = None

            # Ensure output directory exists
            output_dir = Path("outputs/digests")
            output_dir.mkdir(parents=True, exist_ok=True)

            for fmt in formats:
                if fmt == 'markdown':
                    content = exporter.export_markdown(summaries, target_date.date(), include_newsletters)
                    filename = output_dir / f"digest_{target_date.strftime('%Y-%m-%d')}.md"
                    markdown_file = filename
                elif fmt == 'json':
                    content = exporter.export_json(summaries, target_date.date(), include_newsletters)
                    filename = output_dir / f"digest_{target_date.strftime('%Y-%m-%d')}.json"
                    json_file = filename
                else:
                    continue

                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                console.print(f"[green]✓ Exported {fmt}: {filename}[/green]")

            console.print("")

            # Send email if requested
            if send_email:
                console.print("[cyan]📧 Sending email notification...[/cyan]")
                email_config = settings.get('email', {})
                emailer = DigestEmailer(email_config)

                if emailer.send_digest(
                    target_date.date(),
                    len(summaries),
                    markdown_file,
                    json_file
                ):
                    console.print("[green]✓ Email sent successfully[/green]\n")
                else:
                    console.print("[yellow]⚠ Email not sent (check configuration)[/yellow]\n")
        else:
            console.print(f"[yellow]No summaries found for {target_date.date()}[/yellow]\n")
    else:
        console.print("[yellow]⏭ Skipping export step[/yellow]\n")

    # Step 5: Generate audio digest
    audio_digest_generated = False
    audio_digest_enabled = settings.get('audio_digest', {}).get('enabled', True)

    if not skip_audio_digest and audio_digest_enabled:
        console.print("[cyan]🎵 Step 5/6: Generating audio digest...[/cyan]")
        target_date = datetime.now()

        # Get audio digest config settings
        audio_config = settings.get('audio_digest', {})
        min_clip_duration = audio_config.get('min_clip_duration', 30.0)
        max_clip_duration = audio_config.get('max_clip_duration', 180.0)
        max_candidates_per_episode = audio_config.get('max_candidates_per_episode', 40)
        max_clips_per_episode = audio_config.get('max_clips_per_episode', 20)

        # Get narration config
        narration_config = audio_config.get('narration', {})
        enable_narration = narration_config.get('enabled', True)
        piper_voice = narration_config.get('voice', 'en_US-lessac-medium')  # Local Piper voice for fallback

        # Prepare TTS API configuration
        tts_config = prepare_tts_config(settings)

        checks = DependencyChecker.ensure_dependencies(
            settings,
            require_narration=enable_narration,
            use_tts_api=tts_config['use_api'],
            console=console
        )
        critical_failed = False
        for check_name, (passed, message) in checks.items():
            if not passed and check_name in ["ffmpeg", "Ollama"]:
                critical_failed = True
                console.print(f"[red]✗ {check_name}: {message}[/red]")
            elif not passed:
                console.print(f"[yellow]⚠ {check_name}: {message}[/yellow]")

        if critical_failed:
            console.print(f"\n[red]✗ Critical dependencies missing![/red]")
            console.print("[cyan]Run 'p3 check-deps' for installation instructions[/cyan]")
            return

        # Prepare LLM API configuration
        llm_config = prepare_llm_config(settings)

        # Override model from command line if provided
        if llm_model:
            llm_config['model'] = llm_model

        # Determine TTS voice: use API voice if enabled, otherwise use Piper voice
        if tts_config['use_api'] and tts_config['voice']:
            tts_voice = tts_config['voice']  # API-specific voice
        else:
            tts_voice = piper_voice  # Use local Piper voice

        try:
            generator = AudioDigestGenerator(
                db,
                model=llm_config['model'],
                min_clip_duration=min_clip_duration,
                max_clip_duration=max_clip_duration,
                max_candidates_per_episode=max_candidates_per_episode,
                max_clips_per_episode=max_clips_per_episode,
                enable_narration=enable_narration,
                tts_voice=tts_voice,
                use_tts_api=tts_config['use_api'],
                tts_api_provider=tts_config['api_provider'],
                tts_api_model=tts_config['api_model'],
                tts_api_key_file=tts_config['api_key_file'],
                llm_provider=llm_config['provider'],
                llm_api_key_file=llm_config['api_key_file'],
                piper_voice=piper_voice,
                ollama_base_url=llm_config['ollama_base_url'],
                ollama_model=llm_config['ollama_model'],
                model_fallbacks=llm_config.get('model_fallbacks')
            )
            output_file = generator.generate_digest_for_date(
                target_date,
                target_duration=audio_digest_duration
            )

            if output_file:
                digest_info = db.get_audio_digest_by_date(target_date)
                if digest_info:
                    console.print(f"[green]✓ Audio digest created: {output_file}[/green]")
                    console.print(f"  Duration: {digest_info['duration_sec']:.1f}s, Clips: {digest_info['clip_count']}\n")
                    audio_digest_generated = True
                else:
                    console.print(f"[green]✓ Audio digest created: {output_file}[/green]\n")
                    audio_digest_generated = True
            else:
                console.print("[yellow]⚠ No audio digest generated (no transcribed episodes for today)[/yellow]\n")
        except Exception as e:
            console.print(f"[yellow]⚠ Audio digest generation failed: {e}[/yellow]\n")
    elif skip_audio_digest:
        console.print("[yellow]⏭ Skipping audio digest step[/yellow]\n")
    else:
        console.print("[yellow]⏭ Audio digest disabled in config (set audio_digest.enabled: true to enable)[/yellow]\n")

    # Step 6: Cleanup old audio files
    cleaned_count = 0
    freed_space = 0

    if not skip_cleanup:
        console.print("[cyan]🗑️  Step 6/6: Cleaning up old audio files...[/cyan]")

        storage_settings = settings.get('storage', {})
        delete_after_transcription = storage_settings.get('delete_audio_after_transcription', False)
        if cleanup_retention_days is not None:
            retention = cleanup_retention_days
        elif delete_after_transcription:
            retention = 0
        else:
            retention = storage_settings.get('audio_retention_days', 7)

        retention_label = "immediate deletion after transcription" if retention == 0 else f"{retention} days"
        console.print(f"  Retention period: {retention_label}")

        storage_mgr = StorageManager(db=db)
        cleanup_result = storage_mgr.cleanup_old_audio(retention_days=retention, console=console)

        if "error" not in cleanup_result:
            cleaned_count = cleanup_result['deleted_count']
            freed_space = cleanup_result['freed_space_mb']

            if cleaned_count > 0:
                console.print(f"[green]✓ Cleaned up {cleaned_count} files ({freed_space:.2f} MB freed)[/green]\n")
            else:
                console.print(f"[green]✓ No old audio files to clean up[/green]\n")
        else:
            console.print(f"[yellow]⚠ Cleanup skipped: {cleanup_result['error']}[/yellow]\n")
    else:
        console.print("[yellow]⏭ Skipping cleanup step[/yellow]\n")

    # Summary
    console.print("[bold green]✅ Automated Pipeline Complete![/bold green]")
    summary_text = f"""
Summary:
  - Episodes Downloaded: {total_downloaded}
  - Episodes Transcribed: {total_transcribed}
  - Episodes Processed: {total_processed}"""

    if include_newsletters:
        summary_text += f"""
  - Newsletter Articles Fetched: {total_newsletter_fetched}
  - Newsletter Articles Processed: {total_newsletter_processed}"""

    summary_text += f"""
  - Audio Files Cleaned: {cleaned_count} ({freed_space:.2f} MB freed)
  - Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    console.print(summary_text)

    # Show output locations
    if total_processed > 0 or total_newsletter_processed > 0:
        console.print("\n[cyan]📁 Output Locations:[/cyan]")
        console.print("  • Digests: [yellow]outputs/digests/[/yellow]")
        console.print("  • Blogs: [yellow]outputs/blogs/[/yellow] (use [yellow]p3 write[/yellow])")
        console.print("  • Tweets: [yellow]outputs/tweets/[/yellow] (use [yellow]p3 tweet[/yellow])")
        console.print("  • Analysis: [yellow]outputs/thesis/[/yellow] (use [yellow]p3 thesis[/yellow])")
        console.print("\n[cyan]💡 Next Steps:[/cyan]")
        console.print("  • Check status: [yellow]p3 status[/yellow]")
        console.print("  • List companies: [yellow]p3 thesis --list-companies[/yellow]")
        console.print("  • Generate content: [yellow]p3 write --topic 'your topic'[/yellow]")


@main.group()
def newsletter():
    """Manage Substack newsletter integration."""
    pass


@newsletter.command('add')
@click.argument('url')
@click.pass_context
def newsletter_add(ctx, url):
    """Add a Substack newsletter to track.

    \b
    Examples:
      p3 newsletter add https://stratechery.substack.com
      p3 newsletter add https://notboring.substack.com
    """
    db = ctx.obj['db']
    fetcher = SubstackFetcher(db)

    try:
        newsletter_id = fetcher.add_newsletter(url)
        console.print(f"[green]✓ Newsletter added with ID: {newsletter_id}[/green]")
    except Exception as e:
        console.print(f"[red]✗ Failed to add newsletter: {e}[/red]")


@newsletter.command('list')
@click.pass_context
def newsletter_list(ctx):
    """List all tracked newsletters."""
    db = ctx.obj['db']
    fetcher = SubstackFetcher(db)

    newsletters = fetcher.list_newsletters()

    if not newsletters:
        console.print("[yellow]No newsletters tracked yet[/yellow]")
        console.print("\nAdd a newsletter with: [cyan]p3 newsletter add <url>[/cyan]")
        return

    table = Table(title="Tracked Newsletters")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Author", style="yellow")
    table.add_column("URL", style="blue")

    for nl in newsletters:
        table.add_row(
            str(nl['id']),
            nl['name'],
            nl['author'] or "Unknown",
            nl['substack_url']
        )

    console.print(table)


@newsletter.command('fetch')
@click.option('--max-articles', default=10, type=int, help='Max articles per newsletter (default: 10)')
@click.pass_context
def newsletter_fetch(ctx, max_articles):
    """Fetch new articles from tracked newsletters."""
    db = ctx.obj['db']
    fetcher = SubstackFetcher(db)

    console.print(f"[blue]Fetching articles from newsletters...[/blue]\n")
    total = fetcher.fetch_articles(max_articles=max_articles)

    console.print(f"\n[green]✓ Fetched {total} new articles[/green]")


@newsletter.command('digest')
@click.option('--provider', default=None, help='LLM provider (openai, anthropic, ollama)')
@click.option('--model', default=None, help='LLM model to use')
@click.pass_context
def newsletter_digest(ctx, provider, model):
    """Generate summaries for fetched newsletter articles."""
    config = load_config(ctx.obj['config_path'])
    db = ctx.obj['db']

    settings = config.get('settings', {})
    llm_config = prepare_llm_config(settings, llm_model=model)
    if provider:
        llm_config['provider'] = provider
    llm_client = LLMClient.from_config(llm_config)

    # Display model being used
    console.print(f"[blue]LLM Settings:[/blue]")
    console.print(f"  Provider: {llm_config['provider']}")
    console.print(f"  Model: {llm_config['model']}\n")

    # Auto-download Ollama model if needed
    if llm_config['provider'] == 'ollama':
        if not ModelManager.ensure_model_available(llm_config['model'], console):
            console.print(f"[red]✗ Failed to ensure model {llm_config['model']} is available[/red]")
            return

    summarizer = NewsletterSummarizer(
        db=db,
        llm_provider=llm_config['provider'],
        llm_model=llm_config['model'],
        ollama_base_url=llm_config.get('ollama_base_url', 'http://localhost:11434'),
        llm_client=llm_client
    )

    console.print(f"[blue]Processing fetched newsletter articles...[/blue]\n")
    processed = summarizer.process_all_fetched()
    console.print(f"\n[green]✓ Processed {processed} articles[/green]")


if __name__ == '__main__':
    main()

