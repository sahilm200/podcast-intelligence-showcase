# Local setup

This is a source snapshot. Supply your own configuration locally and never commit credentials.

## Python

1. Create and activate a Python virtual environment.
2. Install the package with `python -m pip install -e .`.
3. Use `p3 --help` for CLI commands. `p3 init` creates local processor directories and its database. The original feed configuration and example template are excluded, so create your own `config/feeds.yaml` before running feed-processing commands; `init` cannot copy an omitted template.
4. Local audio processing requires FFmpeg. Local model/transcription options can require Ollama and optional Python dependencies, such as the `local` or `macos` extras.

## API and worker

The API reads environment settings in `p3/api/config.py`. Required values include `SUPABASE_URL`, `SUPABASE_KEY`, `SUPABASE_JWT_SECRET`, and `JWT_SECRET_KEY`. Set a strong, unique JWT secret. Configure Redis through `REDIS_URL` and apply the SQL migrations to a database you control.

Optional features need their corresponding OAuth, email, storage, transcription, or LLM-provider settings. Provider secrets belong only in server-side environment configuration.

Run the API with `p3-api` and the background worker with `p3-worker` after configuring services. Review authentication, redirect URLs, database access rules, and provider costs before exposing the application publicly.

## Frontend

From `frontend/`, run `npm ci`, then `npm run dev`. Inspect the environment variable references in `frontend/lib/` and configure your own API URL and Supabase public client configuration. Never put a service-role key or other server secret in a `NEXT_PUBLIC_` value.

A complete authenticated deployment and external-provider processing were not exercised during source publication.

