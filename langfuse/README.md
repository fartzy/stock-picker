# Local Langfuse (this machine only)

```
docker compose -f langfuse/docker-compose.yml up -d
```

UI: http://127.0.0.1:3100  
Login: `local@localhost` / `local-only`

Telemetry is off. Ports bind `127.0.0.1`. The news judge posts traces here if it is up; if not, scoring continues.

Init keys live in compose as local dummy values for 127.0.0.1 only.
Override with `LANGFUSE_BASE_URL` / `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`.
Set `LANGFUSE_DISABLED=1` to skip tracing. The client refuses non-localhost hosts.
