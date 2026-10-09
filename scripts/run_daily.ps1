# 毎朝の予測（タスクスケジューラから JST 7:50 に実行する想定）
# 7:50 にする理由：冬時間は、米国株の確定時刻（引け＋安全マージン）が JST 7:45、CME 先物が 7:30。
# それより前に実行すると、バックテスト（JST 8:00 までに確定したデータ）と同じデータがそろわない。
# 登録例：
#   schtasks /Create /TN "nikkei_forecast" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 07:50 `
#     /TR "powershell -NoProfile -ExecutionPolicy Bypass -File `"<このファイルのフルパス>`""
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$log = Join-Path $root "predictions\run_daily.log"
"===== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') =====" | Out-File -Append -Encoding utf8 $log
& "$root\.venv\Scripts\python.exe" -W ignore -m src.predict_today *>> $log
exit $LASTEXITCODE
