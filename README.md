# 🚗 Deep Learning-Based Road Skeletonization

This project builds a deep learning pipeline to **predict thin road skeletons** from noisy OpenStreetMap (OSM) images using a **U-Net** model.

It covers **data generation**, **model training**, and **sample prediction visualization**.

---

## 🛠 Project Setup

### 1. Install Dependencies

After cloning the repo, install all required Python packages:

```bash
pip install -r requirements.txt
```

(If `requirements.txt` isn't available yet, you can create it by running:  
`pip freeze > requirements.txt`.)

---

### 2. Data Generation

Generate the training dataset from OpenStreetMap data:

```bash
python thinning_data/trace/models/thinning/make_data.py
```

This will:
- Download OSM data for **Oxford, Ohio**
- Rasterize roads into `256x256` grayscale images
- Apply noise and distortions
- Save:
  - **Input images** to `data/thinning/inputs/`
  - **Target skeletons** to `data/thinning/targets/`
  - **GeoJSON metadata** to `data/thinning/geojson/`

---

### 3. Train the U-Net Model

After generating the dataset, train the U-Net:

```bash
python train.py
```

This will:
- Train on all input/target pairs
- Save model checkpoints (`.pt` files) in `checkpoints/`
- Save sample visual predictions each epoch in `predictions/`

---

## 📁 Directory Structure (After Data Generation)

```
data/
└── thinning/
    ├── inputs/
    │   ├── image_00000.png
    │   ├── image_00001.png
    │   └── ...
    ├── targets/
    │   ├── target_00000.png
    │   ├── target_00001.png
    │   └── ...
    └── geojson/
        ├── target_00000.geojson
        ├── target_00001.geojson
        └── ...
```

---

## 🚨 Important Notes

- **Inputs** are thick, noisy road renderings.
- **Targets** are clean, 1-pixel-wide skeletonized masks.
- **Inputs and targets are stored in separate folders** to prevent accidental confusion.
- Model training uses **BCEWithLogitsLoss**.

---

## 💬 Useful Commands

- Regenerate dataset:

```bash
python thinning_data/trace/models/thinning/make_data.py
```

- Start a clean training run:

```bash
python train.py
```

(You may want to delete old `checkpoints/` and `predictions/` first if retraining.)
