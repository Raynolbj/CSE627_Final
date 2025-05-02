# 🚗 Road Skeletonization with U-Net

This project implements a deep learning-based approach to skeletonize road networks from noisy, thick raster images using OpenStreetMap (OSM) data. It includes data generation, training with a U-Net model, evaluation (both visual and quantitative), and an ablation study framework.

---

## 📁 Directory Structure

```
CSE627_FINAL/
├── cache/                    # Cached OSM data
├── checkpoints/              # Saved model checkpoints
├── data/                     # Image and target PNGs
│   └── thinning/
│       ├── inputs/           # Noisy thick road input images
│       └── targets/          # Ground truth 1px skeletons
├── models/                   # UNet and node metrics implementation
├── predictions/              # Sample outputs during training
├── results/                  # Visuals + metrics from evaluations and ablation
├── scripts/                  # All Python scripts
│   ├── dataset.py
│   ├── train.py
│   ├── train_eval_ablation.py
│   ├── evaluate_sample.py
│   └── evaluate_debug_valence.py
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


