# Pneumonia Detection with CNN Models
This repository contains our CMPT 419 project **PNEUMONET** — a deep learning model that detects pneumonia from chest X-ray images using EfficientNet.  

Our model applies transfer learning with EfficientNet B2 model to classify images as **Normal** or **Pneumonia**, and includes reproducible scripts for training, evaluation, and single-image inference.

---

## Important Links

| [Project report](https://www.overleaf.com/6221845897mmcvmzmnmfxj#2cdf62) |
|-------------------------|

---

## Video/demo/GIF  

The Google Drive link for the video: https://drive.google.com/drive/folders/1aHiXGL7yGwwiwdo-TlOJegmy2ZQKn0zI?usp=drive_link

---

## Table of Contents  
1. [Demo](#1-example-demo)  
2. [Installation](#2-installation)  
3. [Reproducing this project](#3-reproducing-this-project-step-by-step)  
4. [Guidance](#4-guidance)

---

## 1. Example Demo

### Predict a single chest X-ray:
```bash
python run.py --image_path ./example_xray.png --model_path ./models/efficientnet_b2_pneumonia_best.pth
```

### Evaluate trained model:
```bash
python evaluate.py --model_path ./models/efficientnet_b2_pneumonia_best.pth --test_dir ../data/chest_xray/test
```

### Train a EfficientNet B2 model:
```bash
python train.py
```

### Train a Custom CNN model:
Make sure you are in the custom_cnn directory.

```bash
python train.py
```

### Evaluate a Custom CNN model:
Make sure you are in the custom_cnn directory.

```bash
python eval.py best_custom_cnn_model.h5
```


## 2. Installation

### 🧰 Prerequisites
- Python 3.9+
- Anaconda (GPU-enabled setup preferred)
- ~6GB RAM
- ~2GB VRAM (for GPU acceleration)

## 3. Reproducing this project (step-by-step)

### Step 1: Clone the repository
```bash
git clone https://github.com/yourusername/2025_1_project_23.git
cd "C:\Users\Derek\Documents\GitHub\2025_1_project_23"
```

### Step 2: Setup environment (GPU with CUDA 11.8)
For running efficientnet_cnn:

```bash
conda env create -f requirements.yml
conda activate pneumonia-detection

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```



For running custom_cnn:
```bash
conda env create -f environment.yml
conda activate pneumonia-cnn
```



### Step 3: Prepare dataset
Download the dataset from Kaggle: Chest X-Ray Images (Pneumonia)

The Google Drive link for the dataset: For running custom_cnn: https://drive.google.com/file/d/1XvBLpUm-alMSOG1dWUNxef75-9tyn5UB/view

The Kaggle link for the dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Extract the data into the following structure:

```
../data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```
Place chest_xray outside the project folder (../data/... from the repo root).

### Step 4: Train the model
For training efficientnet_cnn:

```bash
cd src
cd efficientnet_cnn
python train.py
```



For training custom_cnn:
```bash
cd src
cd custom_cnn
python data_loader.py
python train.py
```

### Step 5: Evaluate the model
For evaluating efficientnet_cnn:
```bash
python evaluate.py --model_path ./models/efficientnet_b2_pneumonia_best.pth --test_dir ../data/chest_xray/test
```



For evaluating custom_cnn:
```bash
python eval.py best_custom_cnn_model.h5
```

### Step 6: Predict a single image
```bash
python run.py --image_path ./example_xray.png --model_path ./models/efficientnet_b2_pneumonia_best.pth
```



### 🗂️ What to find where
```
repository/
├── data/
├── processed_data/             # Preprocessed input data
├── results/                        # Output results (trained models, visualizations)
├── src/
│   ├── custom_cnn/
│   │   ├── data_loader.py          # Custom CNN dataset loading
│   │   ├── eval.py                 # Evaluation logic for custom CNN
│   │   └── train.py                # Model training logic for custom CNN
│   └── efficientnet_cnn/
│       ├── data_loader.py         # EfficientNet dataset and loaders
│       ├── efficientnet.py        # EfficientNet loading + customization
│       ├── evaluate.py            # Test evaluation + visualizations
│       ├── run.py                 # Single image prediction
│       └── train.py               # Model training logic
├── environment.yml                # Conda environment setup (alternative)
├── requirements.yml               # pip dependencies setup
├── .gitignore                     # Git ignored files list
├── LICENSE                        # Project license
└── README.md                      # You're here
```
If running the code from previous steps do not work, please make sure your repository is structured like above.

## 4. Guidance

### Tools & Best Practices
- Recommended IDE: VSCode
    - If it doesn't work try using Anaconda prompt and cd into the location you cloned the repository
- GPU-accelerated training (CUDA 11.8) is supported

### Tested on:
- Anaconda + Python 3.9
- PyTorch 2.x
- Windows 11
- NVIDIA GPU (CUDA 11.8)
- Apple M1
