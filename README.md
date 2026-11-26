# Q-Sense Studio

A local research application for training Quantum-Train LSTM forecasts for Barangay Wawa, Tanauan City. Import station observations or collect public data, configure the experiment, inspect learning curves and export a classical inference model. All training runs on your computer.

## Launch

Install 64-bit Python **3.11 or newer** with pip. Python 3.12 is the cross-platform CI target. Clone or download this private repository and extract it to a writable folder.

- **macOS:** Double-click **Start Studio.command** or run `bash "Start Studio.command"` from Terminal.
- **Windows 11:** Open PowerShell in the extracted folder and run `.\Start-Studio.ps1`. If local policy prevents running scripts, use the Python commands below in a terminal.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe start.py
```

The first launch installs dependencies. The app opens at **http://127.0.0.1:8765**. Close the browser and reopen that address to return to the same training run. Keep the computer awake during training. A worker checkpoints after every epoch. If the service or computer restarts mid-run, select **Resume** to continue from its latest durable checkpoint. Use `--port 8766` if the default port is in use.

Datasets, provenance, checkpoints, and results live in the ignored `.local/` folder. Set `QSENSE_HOME` to use a different workspace. The prebuilt React interface is included. Node.js is only required to modify the frontend. CPU runs on either platform. Install an appropriate CUDA build into the app's virtual environment for NVIDIA acceleration on Windows. On Apple Silicon the app probes Metal. A device is selectable only after a forward/backward probe. [PyTorch installation guidance](https://pytorch.org/get-started/locally/)

## Train your first experiment

1. Open **Datasets → Import folder** and select the supplied dataset directory. Review the previews and accept. For the national rainfall file the app selects Batangas (PH04010) and final observations. The shorter Batangas extract repeats the tail of the national file. The flood layers are geographic context. The May 2025 Sentinel footprint falls outside Batangas.
2. Select the full historical rainfall CSV, choose **Configure training**, then **Check configuration**. The rainfall preset uses a 12-dekad lookback, 16 hidden units, seed 42, validation from 2014 and test from 2020. For shorter or newer data, clear both date boundaries to use chronological proportions.
3. Choose **Queue experiment**. In **Runs**, select the run to watch optimization, pause, resume, cancel, or duplicate its settings. Each finished job provides observed-versus-predicted series, a vector figure, metrics, an inference checkpoint, and held-out CSV predictions.
4. For Stage A, import hourly rainfall paired with official basin water levels. For Stage B, import ten-minute telemetry. The primary preset uses a six-hour window, 32 hidden units, a three-hour horizon and five seeds. The local training presets require measured local targets. Public hazard maps and rainfall-only observations cannot establish local flood-warning skill.

With the supplied national CSV (SHA-256 `bb41763d1d37f3a4e97c0d58051916e855664dc3ad5582f9a8b93070563`), Python 3.14.7, torch 2.14.0 and NumPy 2.5.5 on CPU, the port exactly reproduced the earlier model's **231** held-out forecasts. QT-LSTM MAE was **32.91 mm**, compared with **32.13 mm** for the dimension-matched LSTM, **32.63 mm** for the seasonal mean and **44.24 mm** for persistence. Dependency versions and hardware can affect reproducibility. These numbers describe next-dekad provincial rainfall.

## Data sources and preparation

**Sources** resolves the full historical WFP CSV through the HDX resource catalog. The Batangas flood polygons are collected from BetterGovPH's archive of Project NOAH data. Links to PAGASA CADS, the hydrometeorological request system, LiPAD and the flood mapping catalog open the provider's access workflow. Use **Attach file** on that source to record provenance when the data arrives. Online collection requires network access. Previously imported data and bundled maps work offline.

- CSV or the active worksheet of XLSX becomes a table. Use **Map columns and join observations** for nonstandard names, input time zone, unit multipliers, or a separate single-station rain series. Joins match exact interval-end timestamps. Use Manila civil time and matched site datums. Rain values are interval millimeters, and stage is centimeters.
- Canonical hourly fields are `timestamp,site_id,qa_flag,stage_cm,rain_mm`. Ten-minute data adds `temp_c,rh_pct,pressure_hpa` and can include `lat,lon`. Local site IDs are C1–C4. Include C3 observations for C2, and C1/C2 observations for C4. The loader derives previous-interval upstream measurements, rate of rise and trailing 1-, 6-, and 24-hour rain. Supply externally quality-checked readings. VALID and causally repaired INTERPOLATED values may enter inputs. Only VALID stage can become a measured target. Leave stale observations masked. A mapped source can be explicitly marked synthetic for Stage A warm starts.
- Supply the SHP, SHX, DBF and PRJ together for shapefiles. GeoJSON uses longitude/latitude. GeoTIFF needs an embedded CRS. Maps display a sampled local preview and geographic bounds. The files retain full precision in local storage.
- Dates, seeds, optimizer, learning rate, patience (zero disables), batch size, model arms, features, network dimensions and compute device are saved with each run. The inference checkpoint stores feature order, units, forecast horizon and train-only normalization. To infer, instantiate `studio.qt_model.Forecast(len(features), hidden)`, load `state_dict`, normalize a physical-unit window in the stored feature order, and undo the target scaling after the forward pass.

## Research protocol

The QT model uses an exact state-vector simulation with trainable RY rotations, CNOT rings, and a basis-index/probability MLP generating all weights. This is an adaptation of Lin et al.'s method. The classical softmax ablation swaps in trainable logits with the identical MLP. The compact LSTM chooses the nearest integer hidden width to the QT offline budget and displays the difference. The dimension-matched LSTM is the primary control. Persistence, rate of rise and an ARX least-squares model provide local-stage baselines. The exported model runs entirely classically. Inference timings are measured on the current computer's CPU. [Lin et al.](https://arxiv.org/html/2407.08617v1) · [Authors' public demonstration](https://github.com/Abeeekoala/Quantum-Train-LSTM-PennyLane-Demo)

For transfer, select an earlier hourly gauge source and its site ID. Each model/seed learns an hourly encoder using the same features and width as its local configuration. Locally exclusive channels are zero during hourly training. The target head is reinitialized for local training; a fixed offset resets the generated head in generator arms. Inputs and stage targets are normalized separately on each training fold. Hourly data are truncated before the first local observation. Scratch and transfer are compared on identical local held-out targets.

Use one verified storm interval per line (`name,start,end`, local ISO times). Test folds are chronological, expand forward, and keep supplied storm intervals together. Each partition has separate windows and local antecedent-rain warmup. The thesis report averages seeds on matched timestamps, then bootstraps paired storm errors (2,000 resamples). Diebold–Mariano checks use a horizon-aware HAC estimate per fold; fold p-values are combined conservatively before secondary Holm adjustment. Event threshold metrics use an approved site threshold when at least three verified events across two test folds qualify. Otherwise local runs try the site-specific 95th percentile of six-hour rise, labeling results as rapid-rise detection. Insufficient evidence produces regression-only results; regional historical runs remain separately labeled. Null denominators are undefined. Inspect sample coverage and stored event-level differences before drawing a research conclusion. [Q-Sense thesis](https://docs.google.com/document/d/1FDjAIXdh12-KGfeJz4DygCe2PYHX-yk4xmht7V23vDA/edit)

## Development and verification

Run `npm ci && npm run build` after editing the React source. Run `python -m unittest discover -s tests -v` and `python tests/http_smoke.py` inside the environment. Tests use generated fixtures in temporary workspaces. GitHub Actions checks macOS and Windows hosted runners, plus the bundled frontend. The Windows matrix includes Windows Server x64 and a separate Windows 11 ARM host running the x64 Python stack. To repeat the desktop smoke check on your own hardware, run the launcher, the same tests, and reopen the browser during a queued CPU run.
