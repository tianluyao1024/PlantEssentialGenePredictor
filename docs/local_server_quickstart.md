# Local server quickstart

This project can be served from a Windows workstation for users on the same
local network.

## Start the server

From PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File E:\PlantEssentialGenePredictor\scripts\webapp\start_local_server_hidden.ps1
```

Or double-click:

```text
E:\PlantEssentialGenePredictor\scripts\webapp\start_local_server.bat
```

Then open:

```text
http://localhost:8501
```

Other users on the same network can usually open:

```text
http://192.168.1.100:8501
```

If other users cannot connect, allow TCP port `8501` through Windows Firewall.

## Stop the server

```powershell
powershell -ExecutionPolicy Bypass -File E:\PlantEssentialGenePredictor\scripts\webapp\stop_local_server.ps1
```

## Clean private uploads

```powershell
D:\Python\Python311\python.exe E:\PlantEssentialGenePredictor\scripts\webapp\cleanup_jobs.py --max-age-hours 24
```

## Notes

- The current website predicts from processed 6,751-dimensional `.npz` feature
  matrices.
- FASTA upload validation is available, but raw sequence probability prediction
  requires the future annotation-light model.
- Public species-level cache files are saved only when the user explicitly
  agrees to share final prediction results.
