$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    Write-Host '[ERROR] .venv 가 없습니다. 프로젝트 가상환경을 먼저 준비해주세요.' -ForegroundColor Red
    exit 1
}
& $python -c 'import streamlit, cv2, mediapipe, ultralytics, torch' 
if ($LASTEXITCODE -ne 0) {
    Write-Host '[ERROR] 필수 패키지가 없습니다. 기존 requirements.txt를 확인하세요.' -ForegroundColor Red
    exit 1
}
Write-Host 'AURA DRIVE 로컬 대시보드: http://127.0.0.1:8501' -ForegroundColor Cyan
Write-Host '종료하려면 이 PowerShell에서 Ctrl+C' -ForegroundColor Gray
& $python -m streamlit run .\11_streamlit_dashboard.py --server.address 127.0.0.1 --server.port 8501 --server.fileWatcherType none --browser.gatherUsageStats false
