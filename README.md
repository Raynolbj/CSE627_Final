# 🚗 Road Skeletonization with U-Net

This project implements a deep learning-based approach to skeletonize road networks from noisy, thick raster images using OpenStreetMap (OSM) data. It includes data generation, training with a U-Net model, evaluation (both visual and quantitative), and an ablation study framework.

---

## 📁 Directory Structure

```
CSE627_FINAL/
├── cache/                        # Cached or temporary data
├── checkpoints/                  # Trained model checkpoint files (.pt)
├── data/                         # Original image data (optional, legacy)
├── thinning_data/                # Main dataset location
│   ├── inputs/                   # Thick road input images (PNG)
│   └── targets/                  # Ground truth 1px-wide skeletons
├── models/                       # UNet architecture and node metric utils
│   ├── unet.py
│   └── node_metrics.py
├── predictions/                  # (Optional) saved outputs from test runs
├── results/                      # All evaluation outputs (images + CSVs)
│   ├── baseline/                     # From baseline config
│   ├── alt_loss/                     # From alt_loss config
│   ├── low_lr/                       # From low_lr config
│   └── sample/                       # From manual evaluation scripts
├── scripts/                      # All training/evaluation scripts
│   ├── dataset.py
│   ├── train.py
│   ├── train_eval_ablation.py         # Runs all configs & saves checkpoints
│   ├── evaluate_sample.py             # Manual inspection of a single prediction
│   ├── evaluate_debug_valence.py      # Visual/debug view of valence structures
│   └── compare_all_outputs_with_target.py  # Side-by-side visual + metrics
├── requirements.txt
└── README.md
```

---

## 🧩 Setup Instructions

1. **Clone the repository**
2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Generate dataset**:

Run the data generation script (inside `thinning_data/trace/models/thinning/`) to populate `data/thinning/inputs` and `data/thinning/targets`.

```bash
python make_data.py
```

4. **Train the model**:

To train a U-Net model on the generated dataset:
```bash
python scripts/train.py
```

This saves sample predictions and checkpoints under `/checkpoints` and `/predictions`.

5. **Run evaluation on trained model**:
```bash
python scripts/evaluate_sample.py
```

6. **Run full ablation study (3 configs)**:
```bash
python scripts/train_eval_ablation.py
```

This will train 3 variants of U-Net (different loss/learning rate), output:
- 3 qualitative prediction PNGs per run
- Loss, MSE, and valence-based node metrics for each sample

---

## 📊 Output Summary

Results are saved to `results/<config_name>_<timestamp>/` including:
- `qualitative_*.png`: input, target, prediction comparison
- `sample_X/loss.txt`
- `sample_X/mse.txt`
- `sample_X/node_metrics.csv`

---

## 📌 Notes

- Predictions are binarized and cleaned before skeletonization.
- Evaluation includes node-level precision/recall for 1–4 valent nodes.
- The pipeline is modular, so you can rerun evaluation without retraining.

---


