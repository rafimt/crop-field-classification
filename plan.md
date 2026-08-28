# Crop Type Classification from Sentinel-2 Patches with ResNet18 (PyTorch Lightning)

Classify the crop growing in a farmland parcel (polygon) from Sentinel-2 imagery
pulled via the **Sentinel Hub API**, using a **ResNet18** CNN backbone in
**PyTorch + PyTorch Lightning**. Two phases: a **minimal** single-image
classifier, then an **extended** multi-temporal version that keeps ResNet18 as a
per-date feature extractor and adds time-series modeling.

---

## 1. Why this project

- **Mirrors real agri-remote-sensing work**: satellite imagery + labeled field
  polygons + crop outcomes — the same shape as production farmland-monitoring
  systems.
- **Modern deep-learning stack, done right**: `LightningModule` +
  `LightningDataModule` + `Trainer` with callbacks, logging, checkpoints, and
  reproducible configs — structured the way DL projects are in industry, not a
  one-off notebook.
- **Real CNN transfer learning**: ResNet18 pretrained on ImageNet, adapted to
  multi-band satellite input — a recognizable, employable skill (torchvision,
  fine-tuning, input-layer surgery, Grad-CAM).
- **Plays to a Geoinformatics background**: working with raster patches, band
  math, cloud masking, and (in Phase 2) the temporal greenness signal of crops.
- **Finishable**: existing labeled parcels + small patches + a small backbone →
  trains in minutes on an RTX 4060.

---

## 2. Data

### Labels — parcel polygons with crop type
- **EuroCrops** (https://github.com/maja601/EuroCrops): harmonized crop-type
  labels on real field polygons across several EU countries. Pick **one region**
  to keep it small.
- Reduce to **4–6 crop classes** (e.g. maize, wheat, barley, rapeseed,
  grassland, sugar beet). Merge rare classes into "other" or drop them.
- Sample **~2,000–4,000 parcels**, as class-balanced as possible.

### Inputs — Sentinel-2 image patches via Sentinel Hub
- Use the **Sentinel Hub Process API** to fetch a small **raster patch** per
  parcel: a fixed window (e.g. **64×64 px @ 10 m** ≈ 640 m) centered on the
  parcel centroid.
- Bands: **B02, B03, B04, B08, B11, B12** (6 channels) + optional derived
  **NDVI**. Add a **parcel mask** channel (1 inside the polygon, 0 outside) so
  the model focuses on the field, not neighbors.
- **Cloud handling**: request a **cloud-free seasonal median composite** (SH can
  mosaic with `mosaickingOrder=leastCC` / least-cloud, or composite over a date
  range) so each patch is clean without manual masking.
- **Cache every patch to disk** (`.npy`/`.tif`) so processing units are spent
  once. Free **Sentinel Hub trial** (30 days of units) covers Phase 1.

### Model input tensor
- **Phase 1:** each parcel → one tensor **`(C, H, W)`** — `C` bands (+ mask), 64×64.
- **Phase 2:** each parcel → a stack **`(T, C, H, W)`** — `T` seasonal
  composites (e.g. monthly), same spatial patch.

---

## 3. Architecture / data flow

```
EuroCrops polygons ──► sample & clean (geopandas)
        │
        ▼
Sentinel Hub Process API ──► per-parcel image patch(es)  (C×H×W)
        │
        ▼
band scaling + parcel mask + (Phase 2) temporal stack
        │
        ▼
LightningDataModule
        │
   Phase 1 ▼                          Phase 2 ▼
ResNet18 (adapted conv1)      ResNet18 shared encoder ──► temporal head
   → linear head                        (attention / LSTM) → linear head
        │                                       │
        └───────────────┬───────────────────────┘
                        ▼
metrics: accuracy, macro-F1, confusion matrix, per-class report (torchmetrics)
```

---

## 4. Tech stack

| Purpose | Library |
|---|---|
| Deep learning | `torch`, `torchvision` (ResNet18), `pytorch-lightning` |
| Metrics | `torchmetrics` |
| Config management | `hydra-core` or `pydantic` + YAML |
| Experiment logging | TensorBoard (`CSVLogger` fallback) |
| Interpretability | `grad-cam` / `captum` |
| Vector / polygons | `geopandas`, `shapely`, `rasterio` |
| EO data access | `sentinelhub` (official Python SDK) |
| Data prep | `numpy`, `pandas` |
| Demo (Phase 2) | `streamlit` + `folium`/`leafmap` |
| Env | `conda`/`venv` + `requirements.txt` |

---

## 5. Repository structure

```
crop-type-s2/
├── README.md
├── plan.md
├── requirements.txt
├── .env.example              # Sentinel Hub client id/secret (never commit real ones)
├── configs/
│   ├── data.yaml            # region, date range, classes, bands, patch size
│   └── train.yaml           # model, lr, batch size, epochs, freeze/unfreeze
├── data/
│   ├── raw/                 # EuroCrops subset + cached SH patches (gitignored)
│   └── processed/           # tensors + split indices
├── src/
│   ├── data/
│   │   ├── labels.py        # load + sample + clean EuroCrops polygons
│   │   ├── sentinelhub_client.py  # Process API patch requests + on-disk caching
│   │   ├── patches.py       # scaling, parcel mask, temporal stacking
│   │   └── datamodule.py    # CropPatchDataset + LightningDataModule (+ augment)
│   ├── models/
│   │   ├── resnet18_single.py  # ResNet18, conv1 adapted for C bands (Phase 1)
│   │   └── resnet18_temporal.py# shared ResNet18 encoder + temporal head (Phase 2)
│   ├── lit_module.py        # LightningModule: shared train/val/test + metrics
│   ├── train.py             # Trainer entry point (reads configs)
│   └── evaluate.py          # load checkpoint, test metrics + plots + Grad-CAM
├── notebooks/
│   ├── 01_explore_labels.ipynb
│   ├── 02_patches_and_ndvi.ipynb   # visualize patches + seasonal NDVI per crop
│   └── 03_results.ipynb
└── app/
    └── streamlit_app.py     # Phase 2 demo
```

Key idea: **one `LightningModule`** with a swappable backbone
(`resnet18_single` / `resnet18_temporal`), so switching models is a config
change, not new training code.

---

## 6. Phase 1 — Minimal (single-image ResNet18)

**Goal:** a clean, reproducible Lightning pipeline classifying crop type from a
single seasonal composite patch with a fine-tuned ResNet18.

1. **Setup** — repo, `requirements.txt`, Sentinel Hub trial account, `.env`,
   config files.
2. **Labels** — download EuroCrops subset, filter to one region, pick classes,
   sample parcels, save a clean `GeoDataFrame`.
3. **Patch fetch** — for each parcel, Process API request for a **cloud-free
   seasonal median composite** 64×64 patch (6 bands + mask); **cache to disk**.
4. **Preprocess** — scale reflectance (e.g. /10000, then normalize with
   **train-split** band means/stds), attach parcel-mask channel, save tensors.
5. **DataModule** — `CropPatchDataset` + `LightningDataModule` with a
   **stratified, spatially-blocked** train/val/test split and light
   augmentation (flips/rotations — valid for top-down imagery).
6. **Model** — `torchvision` **ResNet18** (ImageNet weights); **adapt `conv1`**
   to accept `C` input channels (init new channels from the mean of pretrained
   RGB filters); replace `fc` with a `num_classes` head.
7. **Training strategy** — start with backbone **frozen** (train head), then
   **unfreeze** for fine-tuning with a lower LR; class-weighted cross-entropy;
   `torchmetrics` accuracy + macro-F1.
8. **Trainer** — `EarlyStopping`, `ModelCheckpoint`, `LearningRateMonitor`,
   `precision="16-mixed"`, TensorBoard logging, seeded + `deterministic=True`.
9. **Evaluate** — best checkpoint on test set: accuracy, macro-F1, confusion
   matrix, per-class report; **Grad-CAM** overlays showing which parts of the
   patch drove the prediction.
10. **README** — problem, data, model, results table, how to reproduce.

**Milestone:** `python src/train.py` trains from cached patches and reproduces
the reported metrics on the 4060 in minutes.

---

## 7. Phase 2 — Extended (multi-temporal ResNet18 + demo)

**Goal:** keep ResNet18 but add the crop **time-series** signal, plus richer
inputs and a demo — still lightweight.

1. **Temporal patches** — fetch **T seasonal composites** per parcel (e.g.
   monthly Apr–Oct) → `(T, C, H, W)` stack; cache each.
2. **Temporal model** — ResNet18 as a **shared per-date encoder** (weights
   shared across timesteps) producing a feature vector per date, then a
   **temporal head** aggregating over time:
   - simple: mean/max pooling over T,
   - better: **temporal attention** or a small **LSTM/GRU** over the T features.
3. **Model comparison** — single-image ResNet18 (Phase 1) vs. temporal
   ResNet18, through the same `LightningModule` (accuracy, macro-F1, params,
   train time). Shows the value of the temporal signal.
4. **Training niceties** — mixed precision, cosine LR schedule, freeze/unfreeze
   schedule, early stopping — keeps runs to minutes and the GPU cool.
5. **Sentinel-1 fusion (stretch)** — add radar (VV/VH) channels for
   cloud-robustness.
6. **Interpretability** — **Grad-CAM** per date (where + when the model looks) +
   temporal-attention weights (which dates matter per crop) — a strong,
   geoinformatics-flavored analysis section.
7. **Demo** — Streamlit app: pick a parcel on a map → show its patches / NDVI
   curve, the predicted crop with class probabilities, and Grad-CAM overlays.

**Further stretch (optional):**
- Second region → test spatial generalization.
- Swap ResNet18 for a small ResNet34 / EfficientNet to compare backbones.
- Anomaly flag: parcels whose NDVI curve deviates sharply from their class mean
  (a nod to "crop-health alerting").

---

## 8. Keeping the 4060 comfortable

- **Small patches** (64×64) + **small backbone** (ResNet18, ~11M params) →
  batches of 64–128 fit easily; epochs take seconds–minutes.
- `precision="16-mixed"` + early stopping → low thermal load, no overnight runs.
- **Cache all Sentinel Hub patches** → each parcel fetched once, ever; training
  reads from disk.
- Phase 1 fetch is bounded (a few thousand 64×64×~7 patches ≈ tens–hundreds of
  MB), so download and storage stay modest.

---

## 9. Deliverables

- Public GitHub repo, Lightning-structured, with README + `requirements.txt` +
  configs.
- Reproducible training/eval (`train.py` / `evaluate.py`) + results table.
- Patch + seasonal-NDVI-per-crop visualization (domain-knowledge highlight).
- Phase-2 temporal ResNet18 + model-comparison table + Grad-CAM/attention
  analysis + Streamlit demo.
- Short "Results & Discussion" in the README (what worked, crop confusions,
  limitations).

---

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Sentinel Hub trial units run out | Cache aggressively; modest parcel count; composites (not per-date) in Phase 1. |
| Clouds in composites | Use least-cloud mosaicking / seasonal median; add NDVI/mask; drop bad patches. |
| ImageNet weights ≠ 6-band satellite | Adapt `conv1`, init extra channels from pretrained mean; freeze-then-finetune. |
| Class imbalance | Merge/drop rare classes; class-weighted loss; report macro-F1, not just accuracy. |
| Small dataset → overfitting | Augmentation, dropout, weight decay, freeze backbone early; strong val discipline. |
| Spatial autocorrelation inflates scores | Spatial (block) train/test split; state it in the README. |
| No baseline | Include a trivial baseline (majority-class / NDVI-max heuristic) so gains are honest. |

---

## 11. Rough effort estimate

- **Phase 1 (minimal):** ~4–6 focused sessions → a complete, portfolio-ready
  Lightning + ResNet18 repo.
- **Phase 2 (extended):** ~4–5 more sessions on top.

Ship Phase 1 first — it already stands on its own as a complete deep-learning
project. Add Phase 2 as a second milestone.
