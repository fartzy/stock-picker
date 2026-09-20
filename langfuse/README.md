# Local Langfuse (this machine only)

```
docker compose -f langfuse/docker-compose.yml up -d
```

UI: http://127.0.0.1:3100  
Login: `local@localhost` / `local-only`

Telemetry is off. Ports bind `127.0.0.1`. The news judge posts traces here if it is up; if not, scoring continues.

Project keys (already in compose, local dummy values):

- public: `pk-lf-local-stockpicker`
- secret: `sk-lf-local-stockpicker`

Override with `LANGFUSE_BASE_URL` / `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` if you rotate them. Set `LANGFUSE_DISABLED=1` to skip tracing. Do not point `LANGFUSE_BASE_URL` at cloud.langfuse.com — the client refuses non-localhost.
