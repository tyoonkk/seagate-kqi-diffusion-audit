$root = "C:\Users\taeyoon\Desktop\김태윤_석사\pythonproject_tabddpm"
$py = Join-Path $root ".venv_gpu\Scripts\python.exe"
$logdir = Join-Path $root "experiments\seagate_kqi\outputs\revision_2026_10"
$c = Start-Process -FilePath $py -ArgumentList @("-u", "experiments\seagate_kqi\revision_2026_10\run_regen_archive_v1.py", "--run-name", "regen_archive_v1_run1", "--generator-config", "experiments\seagate_kqi\revision_2026_10\generator_config_v3_final.json", "--tasks", "8") -WorkingDirectory $root -RedirectStandardOutput (Join-Path $logdir "regen_v1_procC.stdout.log") -RedirectStandardError (Join-Path $logdir "regen_v1_procC.stderr.log") -WindowStyle Hidden -PassThru
"procC PID: $($c.Id)"
