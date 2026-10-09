# 予測 API の起動（ローカル専用。yfinance は非公式のデータ源なので外部に公開しない）
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$cfg = Get-Content (Join-Path $root "config.yaml") -Raw -Encoding utf8
$apiHost = if ($cfg -match '(?m)^\s+host:\s*"?([^"\r\n]+)"?') { $Matches[1] } else { "127.0.0.1" }
$port = if ($cfg -match '(?m)^\s+port:\s*(\d+)') { $Matches[1] } else { "8000" }
& "$root\.venv\Scripts\python.exe" -m uvicorn src.api.main:app --host $apiHost --port $port --workers 1
