# MvGPRNet: Robust and Lightweight 3D Reconstruction of Buried Threats

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Pytorch](https://img.shields.io/badge/PyTorch-1.9.0+-red.svg)](https://pytorch.org/)

This is the official implementation of the paper: **"Robust and Lightweight 3D Reconstruction of Buried Threats Using Multi-view GPR Data"** (Published in *IEEE Internet of Things Journal*).

## 📢 Updates
- **[2026/02]** The inference code, pre-trained model weights, and sample datasets have been officially released!

## 📖 Introduction
Ground-penetrating radar (GPR) is a crucial non-intrusive method for detecting buried threats (e.g., pistols, explosives, eavesdropping devices). However, traditional 3D GPR reconstruction methods struggle with low Signal-to-Noise Ratio (SNR) and high computational loads. 

**MvGPRNet** is a robust and lightweight 3D reconstruction framework that:
1. Employs a **range migration** preprocessing step to extract preliminary energy-focused volumes.
2. Utilizes a novel **Golden-angle multi-view projection** strategy to systematically capture 2D spatial features from 3D sparse volumes.
3. Reconstructs high-fidelity 3D voxel models using a NeRF-inspired autoencoder with multi-level feature fusion.

---

## 🛠️ Requirements & Installation

### 1. Python Environment (Deep Learning)
Clone the repository and set up the Conda environment:
```bash
git clone [https://github.com/Shiwen-web/MvGPRNet-3D-Reconstruction.git](https://github.com/Shiwen-web/MvGPRNet-3D-Reconstruction.git)
cd MvGPRNet-3D-Reconstruction

conda create -n mvgprnet python=3.8 -y
conda activate mvgprnet
pip install -r requirements.txt

```

### 2. MATLAB Environment (Data Preprocessing)

For the initial 3D range migration (FK migration) step, you will need:

* MATLAB (with GPU support recommended)
* Parallel Computing Toolbox

---

## 📁 Data Preparation & Model Weights

Due to file size limits, the sample datasets and pre-trained model weights are hosted externally. Please download them from the links below and place them in the corresponding directories.

* **Google Drive**: `[[https://drive.google.com/drive/folders/114Pu5iJR-Pyzj7YVEQGDKPPoguH-4-ts?usp=drive_link ](https://drive.google.com/drive/folders/114Pu5iJR-Pyzj7YVEQGDKPPoguH-4-ts?usp=sharing)]`


## 🚀 Usage Pipeline

### Step 1: 3D Range Migration (MATLAB)

Navigate to the `FK_migration` directory and process the raw GPR `.mat` files to generate 3D volumes.

```matlab
% Set `data_path` in `run_fk_migration.m` to your input `.mat` file, then run:
cd FK_migration
run run_fk_migration.m

```

### Step 2: Multi-view Projection Generation (Python)

Extract 2D multi-view projections from the generated 3D volumes using the golden-angle sampling strategy.

```bash
cd MvGPRNet-3D-Reconstruction

python projection_engine.py \
    --mask-folder /path/to/masks \
    --projection-folder /path/to/output \
    --n-views 16 \
    --sampling-method golden_angle

```

### Step 3: Evaluation & Inference (Python)

To evaluate the pre-trained models across the three different sampling strategies (Golden, Uniform, Random) and calculate the Mean & Std metrics (MSE, MAE, Dice), run the inference script:

```bash
python inference.py \
    --label_folder ./data/test_label \
    --output_dir ./evaluation_results

```

*The evaluation summary will be saved to `./evaluation_results/inference_summary_strategies.csv`.*
