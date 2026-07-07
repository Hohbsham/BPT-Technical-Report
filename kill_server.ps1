$p = Get-NetTCPConnection -LocalPort 8765 -ErrorAction SilentlyContinue
if ($p) { Stop-Process -Id $p.OwningProcess -Force; Write-Host "Killed PID $($p.OwningProcess)" }
else { Write-Host "Port 8765 free" }
