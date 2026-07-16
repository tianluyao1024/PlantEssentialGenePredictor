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

- The website predicts from processed 6,751-dimensional `.npz` feature matrices
  and from raw uploads through the released joint feature-profile models.
- For raw FASTA probability prediction, the server administrator must download
  ESM2, ProtBERT and ProtT5 once. Run:

```powershell
python scripts\feature_extraction\download_plm_weights.py --weights-root ..\plm_model_weights
```

- The application detects `../plm_model_weights` automatically. Otherwise set
  `PLANT_EG_PLM_WEIGHTS` to the directory that contains the checkpoint files.
- Public species-level cache files are saved only when the user explicitly
  agrees to share final prediction results.
