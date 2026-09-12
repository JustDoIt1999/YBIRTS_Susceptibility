# 11-Factor Spatial CV Training & Sweep Runner
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Starting YBI RTS Susceptibility 11-Factor Spatial CV Training & Sweep" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$env:PYTHONPATH = "src"
& "C:\Users\LPF\anaconda3\envs\ML\python.exe" scripts/workflow/run_full_pipeline.py --route dbscan --batch-size 128 --rank-loss-mode ndcg_aware_lambdarank --skip-existing

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "Training and Lambda Selection Finished!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
