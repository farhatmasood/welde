"""
welde/config.py — Central configuration for WELDE experiments.

All paths, hyperparameters, class definitions, and experiment settings
are defined here.  Paths are resolved relative to the project root so
the code works on any machine without hard-coded drive letters.
"""
from pathlib import Path
import torch
import os

# ──────────────────────────── Paths ────────────────────────────
# The project root is determined from this file's location:
#   <project_root>/welde/config.py  →  project_root = ..
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Users can override paths via environment variables:
DATA_ROOT = Path(os.environ.get("WELDE_DATA_ROOT", PROJECT_ROOT / "data"))
OUTPUT_ROOT = Path(os.environ.get("WELDE_OUTPUT_ROOT", PROJECT_ROOT / "results"))
FIGURE_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR  = OUTPUT_ROOT / "tables"
MODEL_DIR  = OUTPUT_ROOT / "models"
LOG_DIR    = OUTPUT_ROOT / "logs"
FEATURE_DIR = OUTPUT_ROOT / "features"

for d in [OUTPUT_ROOT, FIGURE_DIR, TABLE_DIR, MODEL_DIR, LOG_DIR, FEATURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────── Class Definitions ────────────────────
# Spinal MRI dataset class definitions (YOLO label order)
YOLO_CLASS_NAMES = {
    0: "DDD",
    1: "LDB",
    2: "Normal_IVD",
    3: "SS",
    4: "TDB",
    5: "Spondylolisthesis",
}
NUM_CLASSES = 6

# Paper-order mapping (for table/figure display)
PAPER_CLASS_ORDER = [2, 1, 3, 0, 4, 5]  # IVD, LDB, SS, DDD, TDB, Spon
PAPER_CLASS_ABBR  = ["IVD", "LDB", "SS", "DDD", "TDB", "Spon"]

# ──────────────────── Training Hyperparameters ─────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONE = "resnet50"
PRETRAINED = True
IMG_SIZE = 224          # Resize patches to this square
BATCH_SIZE = 64
NUM_WORKERS = 0 if os.name == "nt" else 4   # 0 for Windows; 4 for Linux/Mac
NUM_EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42
WARMUP_EPOCHS = 3

# ──────────────── Feature-Level Regularisation ─────────────────
# Gaussian noise injection to pre-extracted features during training.
# Helps prevent feature-space overfitting (e.g., AP_SS saturation).
# Set to 0.0 to disable; typical values: 0.01–0.05.
FEATURE_NOISE_STD = 0.0

# ──────────────────── WELDE Hyperparameters ────────────────────
WELDE_ALPHA = 0.01       # Minimum weight floor
WELDE_ETA = 0.1          # Penalty coefficient (relaxed for adapter-based heads)
WELDE_LAMBDA = 0.0       # Diversity disabled (harmful with frozen backbone)
WELDE_S = 0.1            # EMA smoothing factor
WELDE_DELTA_THR = 2.0    # Diversity threshold (unused when lambda=0)
FOCAL_GAMMA = 2.0        # Focal loss focusing parameter
CBL_BETA = 0.999         # CBL effective-number hyperparameter
LDAM_C = 0.5             # LDAM margin constant
HEAD_HIDDEN = 256        # Hidden dimension for classification heads
PROJ_DIM = 512           # Per-head adapter projection dimension
WELDE_LOSS_COMPONENTS = ["CE", "FL", "CBL", "LDAM"]  # Loss ensemble

# ──────────────────── Experiment Definitions ───────────────────
BASELINES = ["CE", "wCE", "FL", "CBL", "LDAM", "DB_Loss", "EqL_v2"]
ABLATIONS = [
    "Full_WELDE", "-Adapters", "-EMA", "+Diversity",
    "-LDAM", "-CBL", "-FL", "-CE_head", "CE_ensemble",
]
SENSITIVITY_PARAMS = {
    "alpha":     [0.001, 0.01, 0.1],
    "eta":       [0.1, 1.0, 10.0],
    "s":         [0.01, 0.1, 0.5],
}
