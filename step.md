# Step-by-Step: things YOU need to do

A follow-along checklist. Do them in order. Each step says what to do and how to
know it worked. Project root:
`C:\RMTPROJECTS\dataengineering\crop-type-s2\crop-type-s2\`

---

## Stage 0 — Environment setup (once)

- [ ] **0.1 Open a terminal in the project root**
  ```powershell
  cd C:\RMTPROJECTS\dataengineering\crop-type-s2\crop-type-s2
  ```

- [ ] **0.2 Create and activate a virtual environment**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
  *Worked if:* your prompt now starts with `(.venv)`.
  *If activation is blocked:* run once →
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

- [ ] **0.3 Install dependencies**
  ```powershell
  pip install -r requirements.txt
  ```
  *Worked if:* it finishes with no red errors. (First install downloads PyTorch —
  a few minutes.)

---

## Stage 1 — Sentinel Hub account + credentials

- [ ] **1.1 Create a free account**
  Go to the Copernicus Data Space Ecosystem (https://dataspace.copernicus.eu)
  and register. (Alternatively sentinel-hub.com — 30-day trial.)

- [ ] **1.2 Create an OAuth client**
  In the dashboard → User settings → **OAuth clients** → create new.
  Copy the **client id** and **client secret** immediately (the secret is shown
  once).

- [ ] **1.3 Make your `.env` file**
  ```powershell
  copy .env.example .env
  ```
  Open `.env` and paste your id + secret:
  ```
  SH_CLIENT_ID=your_id_here
  SH_CLIENT_SECRET=your_secret_here
  ```
  Leave the two URL lines as-is (they default to Copernicus Data Space).
  *Worked if:* `.env` exists and both values are filled. **Never commit this file.**

---

## Stage 2 — Get the labels (EuroCrops)

- [ ] **2.1 Download the region archive**
  Go to the Zenodo record → https://zenodo.org/records/10118572
  Recommended for a first run: **Slovenia** — download **`SI_2021.zip`** (213 MB).
  (Optional: `si_2021.csv` — the crop-name → HCAT mapping reference.)

- [ ] **2.2 Unzip into the data folder**
  Unzip `SI_2021.zip` here (create the folders if needed):
  ```
  data\raw\eurocrops\
  ```
  It is a **shapefile set** — several files sharing one name (`.shp`, `.shx`,
  `.dbf`, `.prj`). Keep them together in this folder.
  *Worked if:* you see a `.shp` file inside `data\raw\eurocrops\`.

- [ ] **2.3 Check the filename + column name against the config**
  Open `configs\data.yaml` and confirm two things match your actual file:
  - `eurocrops_file:` → the real path/filename you unzipped.
  - `crop_class_column:` → the column holding the crop **name**
    (EuroCrops usually `EC_hcat_n`).

  Quick way to see the columns (in the activated venv):
  ```powershell
  python -c "import geopandas as g; d=g.read_file(r'data/raw/eurocrops/SI_2021_EC21.shp'); print(d.columns.tolist())"
  ```
  *(swap in your real filename)*.
  *Worked if:* you can see the column list and `crop_class_column` is one of them.

---

## Stage 3 — Build the label table

- [ ] **3.1 Run the label builder**
  ```powershell
  python -m src.data.labels --config configs/data.yaml
  ```
  *Worked if:* it prints class counts and saves `data\processed\labels.parquet`.

- [ ] **3.2 Sanity-check the class counts**
  Look at the printed "sampled" counts. If any class is 0 or tiny, the crop
  names in `class_map` (in `configs\data.yaml`) don't match the file's labels →
  adjust the `class_map` keywords and re-run 3.1.

---

## Stage 4 — Fetch imagery + build the dataset

- [ ] **4.1 Run the full preparation pipeline**
  ```powershell
  python -m src.data.prepare_dataset --config configs/data.yaml
  ```
  This fetches a Sentinel-2 patch per parcel (cached), builds `(C,H,W)` tensors,
  and writes the train/val/test split.
  *Worked if:* you get:
  - `data\raw\patches\*.npy` (raw fetched patches),
  - `data\processed\tensors\*.npy` (model-ready tensors),
  - `data\processed\splits.json` (with train/val/test counts printed).

- [ ] **4.2 Note on re-runs**
  Safe to stop and re-run — it skips parcels already fetched, so you never
  re-spend Sentinel Hub units. If a few parcels error out, re-run to retry them.

- [ ] **4.3 (Optional) Eyeball a patch**
  Open `notebooks\02_patches_and_ndvi.ipynb` (to be added) or quickly:
  ```powershell
  python -c "import numpy as np,glob; a=np.load(glob.glob('data/processed/tensors/*.npy')[0]); print('shape',a.shape,'min',a.min(),'max',a.max())"
  ```
  *Worked if:* shape is `(C, 64, 64)` and values look sane (~0–1).

---

## Stage 5 — Version control (recommended)

- [ ] **5.1 Initialize git**
  ```powershell
  git init
  git add .
  git commit -m "Dataset preparation pipeline"
  ```
  `.gitignore` already excludes `.env` and the `data/` folder, so no secrets or
  large files get committed.

- [ ] **5.2 (Later) Push to GitHub**
  Create an empty repo on GitHub, then follow its "push existing repo" commands.

---

## What comes next (I'll build these — not your manual steps)

- [ ] Model code: `src\models\resnet18_single.py`
- [ ] `src\lit_module.py` (LightningModule) + `src\train.py`
- [ ] Training config `configs\train.yaml`
- [ ] `src\evaluate.py` (metrics + Grad-CAM)
- [ ] Notebooks for EDA + results

Once Stage 4 produces tensors + splits, ping me and I'll wire up training so you
can run `python -m src.train`.

---

### Quick reference — the whole run, once set up

```powershell
cd C:\RMTPROJECTS\dataengineering\crop-type-s2\crop-type-s2
.venv\Scripts\Activate.ps1
python -m src.data.labels --config configs/data.yaml
python -m src.data.prepare_dataset --config configs/data.yaml
```
