# Project Documentation — Crop Type Classification from Sentinel-2

This document explains the whole project end to end: what each step does, how
the scripts fit together, which settings control training, what the results
mean, where the approach is limited, and how it could be improved.

---

## 1. The goal in one sentence

Given the boundary (polygon) of a farm field, predict **which crop is growing in
it** — one of **Maize, Soybean, Potato, Sunflower** — using free Sentinel-2
satellite imagery.

We do this two ways and compare them:

- **Phase 1 — single image:** one blended "summer" image per field.
- **Phase 2 — temporal:** several images across the year, so the model can see
  how the field *changes over time*.

---

## 2. The big picture (data flow)

```
EuroCrops shapefile            Sentinel Hub (satellite)
 (field polygons + crop name)   (images on demand)
        │                               │
        ▼                               ▼
   labels.py  ───────────────►  prepare_dataset*.py
   (pick fields, 4 classes)     (download image per field,
                                 build number arrays = tensors)
                                         │
                                         ▼
                                 data/processed/tensors*/  +  splits*.json
                                 (one .npy per field + train/val/test lists)
                                         │
                                         ▼
                                  datamodule*.py  (feeds data to the model)
                                         │
                                         ▼
                        train.py / train_temporal.py  (train the model)
                                         │
                                         ▼
                            outputs/results*.txt  (accuracy, F1, confusion)
```

`*` means there are two versions: one plain (Phase 1) and one `_temporal`
(Phase 2).

---

## 3. What each stage actually does

### Stage A — Labels (which fields, which crop)
- **Input:** the EuroCrops shapefile for Slovenia (`data/SI_2021/SI_2021_EC21.shp`).
  Each polygon is a real field with a crop name in the `EC_trans_n` column.
- **What happens:** we keep only the 4 crops we care about, cap each crop at a
  fixed number of fields (so classes are balanced), give each field a simple id
  (`p000000`, `p000001`, …), and save the result.
- **Output:** `data/processed/labels.parquet` — a clean table of
  `parcel_id, crop_class, geometry`.

### Stage B — Download imagery + build tensors
For every field:
1. Compute a **64×64 pixel square** (640 m) centered on the field.
2. Ask **Sentinel Hub** for a **cloud-free composite** over that square:
   - **Phase 1:** ONE composite for the whole Apr–Oct season.
   - **Phase 2:** SIX composites, one per ~2-month window across the crop year.
3. For each composite, stack the channels into an array:
   - 6 satellite bands (B02, B03, B04, B08, B11, B12)
   - + **NDVI** (a "how green" index)
   - + a **parcel mask** (1 inside the field, 0 outside)
4. Save as a `.npy` file (a saved number array — see the README note on `.npy`).

- **Phase 1 tensor shape:** `(8, 64, 64)` = 8 channels × 64 × 64 pixels.
- **Phase 2 tensor shape:** `(6, 8, 64, 64)` = 6 timesteps × the same image.

Everything is **cached to disk**, so re-running never re-downloads.

### Stage C — Train / Val / Test split
- Fields are grouped into ~10 km **grid cells**, and whole cells are assigned to
  train, validation, or test.
- **Why:** neighboring fields look almost identical. If one went to training and
  its neighbor to testing, the model would "cheat." Splitting by location makes
  the test score honest.
- **Output:** `splits*.json` — the lists of field ids per split, plus the class
  number (0–3) for each field.

### Stage D — The model
- **Phase 1 — ResNet18:** a standard image-recognition network, adapted to take
  8 input channels instead of 3, and to output 4 classes. Started from ImageNet
  weights (transfer learning).
- **Phase 2 — ResNet18 + GRU:** the same ResNet18 turns *each* of the 6 images
  into a 512-number summary; a small **GRU** (a network that reads sequences)
  then reads those 6 summaries **in order** and learns the timing pattern of
  each crop. A final layer outputs the 4 class scores.

### Stage E — Training & evaluation
- The model is trained with PyTorch Lightning (handles the training loop).
- Metrics: **accuracy** (% correct) and **macro-F1** (average per-class score,
  fair to every crop even if sizes differ).
- After training, it tests on the held-out test set and writes a text report.
- `evaluate.py` adds a **confusion matrix** (which crops get mixed up) and
  **per-class precision / recall / F1**.

---

## 4. Script pipeline — which script does what

| Order | Script | Role | Run it? |
|------:|--------|------|---------|
| 1 | `src/data/labels.py` | Shapefile → clean field table (4 classes) | yes |
| 2 | `src/data/sentinelhub_client.py` | Talks to Sentinel Hub, downloads + caches images | no (used by others) |
| 3 | `src/data/patches.py` | Geometry + turns a raw image into an `(C,H,W)` tensor | no (used by others) |
| 4 | `src/data/prepare_dataset.py` | Phase 1: build all single-image tensors + split | yes |
| 5 | `src/data/prepare_dataset_temporal.py` | Phase 2: build all `(T,C,H,W)` tensors + split | yes |
| 6 | `src/data/datamodule.py` | Phase 1: feeds tensors to the model | no (used by train) |
| 7 | `src/data/datamodule_temporal.py` | Phase 2: feeds temporal tensors | no (used by train) |
| 8 | `src/models/resnet18_single.py` | Phase 1 model (ResNet18) | no (used by train) |
| 9 | `src/models/resnet18_temporal.py` | Phase 2 model (ResNet18 + GRU) | no (used by train) |
| 10 | `src/lit_module.py` | Shared training logic (loss, metrics, optimizer) | no (used by train) |
| 11 | `src/train.py` | Train Phase 1 | yes |
| 12 | `src/train_temporal.py` | Train Phase 2 | yes |
| 13 | `src/evaluate.py` | Confusion matrix + per-class report | yes |
| — | `src/config_utils.py` | Loads the YAML config files | no (helper) |

**Commands, in order:**
```bash
python -m src.data.labels                 --config configs/data.yaml
python -m src.data.prepare_dataset        --config configs/data.yaml
python -m src.train                       --config configs/train.yaml
python -m src.evaluate                    --config configs/train.yaml
python -m src.data.prepare_dataset_temporal --config configs/data_temporal.yaml
python -m src.train_temporal              --config configs/train_temporal.yaml
```

---

## 5. Training parameters (what controls the model)

These live in the config files so you can change them without touching code.

### Phase 1 — `configs/train.yaml`
| Parameter | Value | What it does |
|-----------|-------|--------------|
| `batch_size` | 32 | How many fields the model sees at once per step |
| `max_epochs` | 30 | Maximum passes over the training data |
| `learning_rate` | 0.0003 | How big each learning step is (Adam optimizer) |
| `pretrained` | true | Start ResNet18 from ImageNet weights (transfer learning) |
| `augment` | true | Random flips of images = free extra training variety |
| `precision` | 16-mixed | 16-bit math on GPU = faster, less memory |
| `num_workers` | 0 | Data-loading processes (0 is safest on Windows) |

### Phase 2 — `configs/train_temporal.yaml`
| Parameter | Value | What it does |
|-----------|-------|--------------|
| `batch_size` | 16 | Smaller than Phase 1 (sequences use more memory) |
| `max_epochs` | 40 | Max passes over the data |
| `learning_rate` | 0.0003 | Learning step size |
| `hidden_size` | 128 | Size of the GRU's memory (temporal head) |
| `pretrained` | true | ResNet18 encoder starts from ImageNet |
| `augment` | true | Random flips (applied the same way to all 6 timesteps) |
| `precision` | 16-mixed | Fast GPU math |

### Data settings — `configs/data.yaml` / `configs/data_temporal.yaml`
| Parameter | Value | What it does |
|-----------|-------|--------------|
| `classes` | Maize, Soybean, Potato, Sunflower | The crops to classify |
| `samples_per_class` | 600 (P1) / 200 (P2 subset) | Fields per crop |
| `bands` | B02,B03,B04,B08,B11,B12 | Satellite bands requested |
| `patch_size_px` | 64 | Image size in pixels |
| `resolution_m` | 10 | Meters per pixel (Sentinel-2 native) |
| `season` / `periods` | Apr–Oct (P1) / 6 windows (P2) | Time window(s) fetched |
| `add_ndvi` | true | Add the NDVI channel |
| `add_parcel_mask` | true | Add the in-field mask channel |

Under the hood, training also uses (fixed in code): **Adam** optimizer, a
**cosine** learning-rate schedule, **cross-entropy** loss, **early stopping**
(stops if validation macro-F1 stops improving), and it keeps the **best**
checkpoint by validation macro-F1.

---

## 6. Results

Test set, held-out fields.

| | Accuracy | Macro-F1 |
|--|---------:|---------:|
| **Phase 1** (single image) | 0.45 | 0.44 |
| **Phase 2** (temporal) | **0.56** | **0.55** |

Per-class F1:

| Crop | Phase 1 | Phase 2 |
|------|--------:|--------:|
| Maize | 0.24 | **0.48** |
| Sunflower | 0.25 | **0.48** |
| Soybean | 0.59 | 0.53 |
| Potato | 0.67 | **0.73** |

**Main finding:** in Phase 1, sunflower and maize were constantly confused (they
look identical in one summer image). Adding the time dimension roughly **doubled**
their F1 — exactly the crops that can only be told apart by *when* they grow.

---

## 7. Limitations

1. **Small, single-region dataset.** Only Slovenia, one year (2021), a few
   hundred fields per crop. The model may not transfer to other countries,
   climates, or years.
2. **Uncontrolled Phase 1 vs Phase 2 comparison.** Phase 1 used 600 fields/class;
   Phase 2 used a 200/class subset (to save download cost). So the headline
   numbers are not a perfect apples-to-apples test — the per-class F1 pattern is
   the trustworthy signal, not the exact percentages.
3. **Clouds and gaps.** Some time windows have few clear satellite passes, so a
   composite can be partly cloudy or empty. This is worse for winter months.
4. **Only 4 easy-ish crops.** Real crop maps have dozens of classes, including
   very similar ones (different cereals), which are much harder.
5. **Median composite loses detail (Phase 1).** Blending a whole season into one
   image throws away the timing information — this is *why* Phase 1 is weak.
6. **Field context, not pure field.** The 64×64 window includes neighbors around
   the field; the mask helps but the model can still be influenced by surroundings.
7. **Moderate accuracy.** 56% is a good *demonstration*, not a production-ready
   classifier.
8. **No radar / weather.** We use only optical Sentinel-2. Clouds block optical
   sensors entirely on some dates.

---

## 8. How it could be improved

Roughly in order of expected payoff:

1. **Fair, larger comparison.** Re-run Phase 2 on the full 600/class so Phase 1
   and Phase 2 are directly comparable, and add more fields overall.
2. **More timesteps / smarter dates.** Use monthly composites, or pick the dates
   that best separate crops (e.g. flowering time). More/better timing = more
   signal.
3. **A stronger temporal model.** Replace the GRU with a **temporal attention**
   model (e.g. LTAE), which is state of the art for crop time-series and can show
   *which dates* it focuses on.
4. **Add Sentinel-1 radar.** Radar sees through clouds, filling the gaps that
   break optical composites — a big robustness gain.
5. **Better cloud handling.** Explicit cloud masking + gap-filling / interpolation
   so each timestep is genuinely clean.
6. **More regions and years.** Train across countries/years to make the model
   generalize instead of memorizing Slovenia-2021.
7. **Explainability.** Grad-CAM (where in the field it looks) and attention
   weights (which dates matter) — turns a number into an analysis.
8. **Class balance & harder classes.** Add more crops and use class weighting so
   rare crops are handled fairly.
9. **A demo.** A small app: click a field on a map → see its NDVI-over-time curve
   and the predicted crop with probabilities.

---

## 9. File map (quick reference)

```
configs/                 all settings (data + training)
data/
  SI_2021/               EuroCrops shapefile (source labels)
  raw/patches*/          cached raw downloads (never re-fetched)
  processed/
    labels.parquet       clean field table
    tensors*/            model-ready .npy arrays
    splits*.json         train/val/test lists + labels
src/
  data/                  labels, sentinel client, patch building, datamodules
  models/                resnet18_single, resnet18_temporal
  lit_module.py          shared training logic
  train*.py              training entry points
  evaluate.py            confusion matrix + per-class report
outputs/
  results.txt            Phase 1 results
  results_temporal.txt   Phase 2 results
```
