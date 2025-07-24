# Cloth Segmentation Pipeline

A robust pipeline for 3D cloth segmentation.

---

## 🚀 Overview
This pipeline processes 3D garment meshes, segments front and back views, labels with region-aware strategies, and produces high-quality overlays and colored meshes. 

---

## 📁 Directory Structure
```
.
├── scripts/                # Main pipeline and utilities
│   ├── run.py              # Main entry: improved segmentation pipeline
│   ├── run_backup.py       # Legacy/backup pipeline
│   └── ...
├── data/                   # Input data
│   ├── labels/             # Ground truth or generated labels
│   ├── meshes/             # Input OBJ meshes
│   └── embeddings/         # Feature embeddings
├── training/               # Training scripts
│   └── train.py
├── utils/                  # Utility modules
│   ├── unwrap_utils.py
│   └── render_utils.py
├── outputs/                # All output results
│   ├── labels/
│   ├── colored_meshes_outputs/
│   ├── overlay_outputs/
│   ├── visualization_output/
│   ├── uv_outputs/
│   └── mask_outputs/
├── debug/                  # Debugging outputs
│   ├── checks/             #check the cloth structure
│   └── debug_masks/
├── configs/ config.json             #configuration summary
├── README.md               # This file
└── ...
```

## 🔧 Usage

### Run the Pipeline
```bash
python scripts/run.py --obj_path data/meshes/shirt.obj
```

#### Optional Arguments
- `--checkpoint`: Path to SAM2 checkpoint
- `--model_cfg`: Path to SAM2 model config
- `--img_h`, `--img_w`: Image height/width
- `--min_mask_area`: Minimum mask area for segmentation
- `--min_vertex_count`: Minimum vertex count for label assignment

#### Example: High-Quality Processing
```bash
python scripts/run.py --obj_path data/meshes/shirt.obj --img_h 1600 --img_w 1200 --min_mask_area 1000
```

#### Example: Fast Processing
```bash
python scripts/run.py --obj_path data/meshes/shirt.obj --img_h 800 --img_w 600 --min_mask_area 5000
```


## ⚙️ Configuration
Edit `config.json` to customize:
- Model checkpoint and config
- Rendering settings (image size, camera positions)
- Segmentation thresholds (mask area, confidence)
- Merging and visualization options

Example:
```json
{
  "model": {
    "checkpoint": "path/to/checkpoint.pt",
    "model_cfg": "path/to/config.yaml"
  },
  "rendering": {
    "img_h": 1200,
    "img_w": 800,
    "focal": [20000, 20000],
    "front_translation": [0.0, 1.0, 28.0],
    "back_translation": [0.0, 1.0, -28.0]
  },
  "segmentation": {
    "points_per_side": 32,
    "pred_iou_thresh": 0.8,
    "stability_score_thresh": 0.85,
    "min_mask_area": 2000,
    "confidence_threshold": 0.8
  }
}
```

## 📦 Output Files
- `outputs/labels/vertex_labels_full.npy`
- `outputs/colored_meshes_outputs/dress_colored_vert.ply` # Colored mesh
- `outputs/overlay_outputs/mask_overlay_front.png` # Front overlay
- `outputs/overlay_outputs/mask_overlay_back.png`  # Back overlay
- Additional overlays, UVs, and debug outputs in respective folders

---

## 🐛 Troubleshooting
- **Missing Model**: Check `checkpoint` and `model_cfg` paths
- **CUDA Out of Memory**: Lower image size or batch size
- **Poor Segmentation**: Adjust confidence/area thresholds

---
