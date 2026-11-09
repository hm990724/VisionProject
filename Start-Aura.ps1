$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$auraPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $auraPython)) {
    throw 'Project .venv is missing. See README.md for setup.'
}
& $auraPython -m streamlit run (Join-Path $PSScriptRoot '11_streamlit_dashboard.py') --server.address=127.0.0.1 --server.fileWatcherType=none --browser.gatherUsageStats=false
