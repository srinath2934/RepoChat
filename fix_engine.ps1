# 🛠️ RepoChat: Endee Engine Fix & Diagnostics
# This script ensures the Docker engine is running and the database is reachable.

Write-Host "🔍 Step 1: Checking Docker Desktop Status..." -ForegroundColor Cyan

# Check if docker is available
try {
    docker version | Out-Null
    Write-Host "✅ Docker Desktop is Running." -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is NOT RUNNING on your system." -ForegroundColor Red
    Write-Host "👉 ACTION: Please open the 'Docker Desktop' application from your Start Menu." -ForegroundColor Yellow
    exit
}

Write-Host "🔍 Step 2: Sychronizing Infrastructure (docker-compose)..." -ForegroundColor Cyan
cd $PSScriptRoot

# Clean up any stale containers and restart with the correct 9999 port mapping
docker-compose down
docker-compose up -d --force-recreate

Write-Host "🔍 Step 3: Verifying Connectivity on Port 9999..." -ForegroundColor Cyan
Start-Sleep -Seconds 5

try {
    $response = Invoke-WebRequest -Uri "http://localhost:9999/api/v1/health" -Method Get -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "🚀 SUCCESS: Endee Engine is now ONLINE at http://localhost:9999" -ForegroundColor Green
        Write-Host "You can now click 'Refresh' in your Streamlit app." -ForegroundColor White
    }
} catch {
    Write-Host "❌ FAILURE: Engine started but is not responding on 9999 yet." -ForegroundColor Red
    Write-Host "Wait 30 seconds and try again. Ensure no other app is using port 9999." -ForegroundColor Yellow
}
