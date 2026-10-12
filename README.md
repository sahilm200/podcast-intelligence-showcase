# Podcast Intelligence Showcase

A source-code showcase of an adapted Parakeet Podcast Processor application. It combines a podcast-processing CLI, a FastAPI service, a Next.js web interface, SQL migrations, and background processing.

## What's included

- Python podcast ingestion, transcription, summarization, and content-generation modules in `p3/`
- API endpoints for authentication, podcast catalogs, subscriptions, digests, and Spotify integration in `p3/api/`
- Redis/ARQ background processing in `p3/api/worker.py`
- A Next.js interface for digests, podcasts, and preferences in `frontend/`
- Database schema migrations in `migrations/`

This is an adaptation of [Parakeet Podcast Processor](https://github.com/haasonsaas/parakeet-podcast-processor), not a claim of sole authorship. See [ATTRIBUTION.md](ATTRIBUTION.md) for provenance and the upstream license declaration.

## Running it

See [SETUP.md](SETUP.md). The web application requires your own service configuration, database, Redis, and any optional provider credentials. No working credentials or hosted service are included.

The CLI can use local or hosted processing providers depending on configuration. External API features may incur charges. Do not treat this repository as a preconfigured deployment or a production security review.

## Publication scope

This fresh source snapshot includes application source and dependency metadata. It excludes prior Git history, private environment/deployment configuration, keys, generated audio, transcripts, and personal outputs. Example email addresses and a source-repository HTTP referer have been replaced with neutral values.

Static secret checks and Python syntax checks were performed for this snapshot. They do not establish that every dependency or runtime path is secure. An end-to-end service deployment and live provider calls were not tested as part of publication.

## Project structure

- `p3/`: processor, CLI, API, and background worker
- `frontend/`: Next.js web interface
- `migrations/`: SQL schema
- `pyproject.toml`: Python package metadata and dependencies

