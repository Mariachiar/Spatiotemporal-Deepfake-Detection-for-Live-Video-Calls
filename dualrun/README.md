# 🎭 DeepFake Detection – Dual Branch AU + LMK

Questo repository implementa una pipeline completa per la **rilevazione di deepfake** basata su un **modello Dual Encoder** che combina:
- **Action Units (AU)** estratti dal volto
- **Landmarks (LMK)** facciali normalizzati

L’architettura usa **Transformer Encoder + Attention Pooling**, con opzioni avanzate come:
- **Domain-Adversarial Training (DAT)** per la robustezza cross-dataset
- **Sampler bilanciati per tecnica**
- **Temperature scaling** per calibrazione delle probabilità


---

## 📂 Struttura del repository

```
dualrun/
│
├── cli/                     # Script eseguibili da linea di comando
│   ├── best.py              # Analisi e report del miglior checkpoint
│   ├── opts.py              # Parser e gestione argomenti CLI
│   └── run.py               # Entry point principale (training / valutazione)
│
├── data/                    # Dataset e preprocessing
│   ├── compute_norm_stat.py # Calcolo statistiche globali AU/LMK (zscore=global)
│   ├── dataset_dual.py      # Dataset AU+LMK con normalizzazione e augmentation
│   ├── make_au_features.py  # Estrazione AU features da video
│   ├── make_lmk_features.py # Estrazione landmark features da video
│   ├── makeCDF_splits.py    # Generazione split CelebDF (train/val/test)
│   └── makeFF_splits.py     # Generazione split FaceForensics++
│
├── model/
│   └── dual_encoder.py      # Modello DualEncoderAU_LMK (AU branch + LMK branch + DAT opzionale)
│
├── train/                   # Moduli training/validazione
│   ├── engine.py            # Loop training/val/test + early stopping
│   ├── losses.py            # BCE e Focal loss
│   ├── metrics.py           # Metriche (AUC, PR-AUC, acc, f1, ecc.)
│   ├── samplers.py          # Sampler bilanciati per real/fake e leave-one-out
│   └── thresholds.py        # Selezione soglia ROC ottimale
│
└── utils/
    ├── io.py                # Utility di I/O (JSON, checkpoint)
    └── setup.py             # Setup logging, seed e configurazioni comuni
```

---

## ⚙️ Installazione

### 1. Creare un ambiente Python

```bash
conda create -n deepfake python=3.10 -y
conda activate deepfake
```

### 2. Installare le dipendenze principali

```bash
pip install torch torchvision torchaudio
pip install numpy scikit-learn tqdm opencv-python mediapipe matplotlib pandas
```

---

## 🛠️ Preprocessing dei dataset

### Estrazione AU features

```bash
python dualrun/data/make_au_features.py \
  --base datasets/raw_datasets/FaceForensics++_C23 \
  --out datasets/processed_dataset/FaceForensics++_C23
```

### Estrazione LMK features

```bash
python dualrun/data/make_lmk_features.py \
  --base datasets/raw_datasets/FaceForensics++_C23 \
  --out datasets/processed_dataset/FaceForensics++_C23
```

### Generazione split FaceForensics++

```bash
python dualrun/data/makeFF_splits.py \
  --ffpp-root datasets/processed_dataset/FaceForensics++_C23 \
  --outdir splitF/ff++ \
  --ratios-base 0.45,0.20,0.15,0.10,0.10 \
  --ratios-extra 0.00,0.40,0.40,0.10,0.10 \
  --cap_extra_valtest 400
```

### Generazione split CelebDF

```bash
python dualrun/data/makeCDF_splits.py \
  --processed-root datasets/raw_datasets/celebdf_v2 \
  --outdir splitC/celebdf \
  --a2-ratio 0.70 \
  --c-ratio 0.10
```

### Combinare split FF++ + CelebDF

```bash
python data/combine_splits.py \
  --ffpp_dir split_ffpp --celeb_dir split_celebdf --out_dir split_combined
```

### Calcolo statistiche globali (per zscore=global)

```bash
python dualrun/data/compute_norm_stat.py \
  --base datasets/processed_dataset/FaceForensics++_C23 \
  --index-json splitF/ff++/ffpp_phase1_pretrain.json \
  --out split/norm_stats.npz
```

---

## 🔥 Training

Esempio di training standard su FaceForensics++:

```bash
NORM_STATS=split/norm_stats.npz \
python cli/run.py \
  --data datasets/processed_dataset/FaceForensics++_C23 \
  --index split_ffpp/ffpp_phase1_pretrain.json \
  --out runs/ffpp_phase1 \
  --epochs 100 \
  --batch 256 \
  --lr 3e-4 --wd 1e-5 \
  --d_model 192 --heads 6 --layers 3 --ff_dim 768 \
  --dropout 0.3 \
  --zscore global \
  --epoch-samples 60000 \
  --scheduler onecycle --onecycle_pct_start 0.1 --onecycle_final_div 100 \
  --amp --tqdm
```

### Opzioni utili

* **DAT**: `--dat --dat-lambda 0.5 --dat-schedule linear`
* **Freeze encoders**: `--freeze-encoders 5` (congela encoder per 5 epoche)
* **Target FPR**: `--target-fpr 0.1` (sceglie soglia ROC con FPR ≤ 0.1)

---

## 📊 Valutazione

Per valutare il modello con il checkpoint migliore:

```bash
python dualrun/cli/best.py --ckpt runs/ffpp_model/best.py  --index-cache split_ffpp/ffpp_phase1_pretrain.json  --subset test   --use-best-thresh
```

Durante il training vengono salvati:

* `best.pt` → pesi migliori del modello
* `best_threshold.txt` → soglia ROC ottimale clip-level
* `best_video_threshold.txt` (se calcolata) → soglia ottimale video-level
* `args.json` → configurazione completa
* `splits_used.json` → elenco clip usate

E' possibile effettuare tre tipologie di valutazione:
1. Clip-level
```bash
python cli/best.py \
  --ckpt runs/ffpp_phase1/best.pt \
  --index-cache split_ffpp/ffpp_phase1_pretrain.json \
  --subset test \
  --use-best-thresh --zscore global
```

2. Video-level (track aggregation)
```bash
python cli/best.py \
  --ckpt runs/ffpp_phase1/best.pt \
  --index-cache split_ffpp/ffpp_phase1_pretrain.json \
  --subset test \
  --use-best-thresh --zscore global \
  --video-metrics --agg-mode track_mean
```
### Opzioni utili

* **track_mean**: un track è fake se media clip ≥ soglia
* **track_majority**: un track è fake se maggioranza clip fake

---

## 📚 Riferimenti

* **FaceForensics++** – Rössler et al., ICCV 2019
* **CelebDF** – Li et al., CVPR 2020
* **Focal Loss** – Lin et al., ICCV 2017
* **Domain-Adversarial Training (GRL)** – Ganin et al., JMLR 2016

---

## ✨ Note

* Il dataset è gestito in modo modulare: AU e LMK vengono normalizzati e sincronizzati per ogni clip.
* L’uso di **AttentionPooling** migliora la rappresentazione rispetto alla media semplice.
* Gli split sono **deterministici e riproducibili** (seed controllato).
* Gli script sono stati modularizzati per consentire facile estensione o sostituzione di parti (es. nuovo encoder o nuove loss).
