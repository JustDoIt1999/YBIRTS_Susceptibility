@echo off
echo ======================================================================
echo Starting YBI RTS Susceptibility 12-Factor Spatial CV Training & Sweep
echo ======================================================================
call C:\Users\LPF\anaconda3\Scripts\activate.bat C:\Users\LPF\anaconda3\envs\ML
cd /d D:\orca\workspaces\YBIRTS_Susceptibility\feature-Slope-unit\YBI_Ranking_Framework
set PYTHONPATH=src
python scripts/workflow/run_full_pipeline.py --route dbscan --batch-size 256 --rank-loss-mode ndcg_aware_lambdarank --skip-existing
echo.
echo ======================================================================
echo Training and Lambda Selection Finished!
echo ======================================================================
pause
