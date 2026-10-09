# 毎朝の予測（タスクスケジューラから JST 7:30 に実行する想定）
# 登録例：
#   schtasks /Create /TN "nikkei_forecast" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 07:30 `
#     /TR "powershell -NoProfile -ExecutionPolicy Bypass -File `"<このファイルのフルパス>`""
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$log = Join-Path $root "predictions\run_daily.log"
"===== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') =====" | Out-File -Append -Encoding utf8 $log
& "$root\.venv\Scripts\python.exe" -W ignore -m src.predict_today *>> $log
exit $LASTEXITCODE
