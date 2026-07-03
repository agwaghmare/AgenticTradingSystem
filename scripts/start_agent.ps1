# Start the agent locally (uses .env + native Postgres on localhost:5432)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "Running DB migrations..."
python -m alembic upgrade head

Write-Host "Starting agent on http://localhost:8000 (Discord notifications enabled)"
$env:PYTHONUNBUFFERED = "1"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
