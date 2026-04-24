# 🧬 KRASPred

## Multi-Task Deep Neural Network for Prediction of KRas Inhibitors and IC₅₀ Values
---

## 📖 Overview

**KRASPred** is a multi-task deep neural network (DNN)-based predictive framework developed for the identification of **GTPase Kirsten rat sarcoma virus oncogene homologue (GTPase KRas) inhibitors**.

The model performs:

- **Binary Classification** – Predicts whether a molecule is a GTPase KRas inhibitor or non-inhibitor
- **Regression Prediction** – Predicts IC₅₀ values (in µM)

The system accepts **SMILES (Simplified Molecular Input Line Entry System)** representations as input.

This repository provides a **Docker-based, fully reproducible deep learning pipeline**, including:

- Pre-trained multi-task DNN model
- Descriptor normalization model
- Prediction script
- Docker environment specification

This ensures reproducible predictions across different operating systems.

---

## 🧪 Scientific Background

GTPase KRas is a signal-regulating molecular switch; when its GTPase function is disrupted, it becomes permanently active driver of cancer and RASopathies (genetic developmental disorders).

KRAS is one of the most frequently mutated oncogenes for example:
- In pancreatic ductal adenocarcinoma (PDAC): ~90% cases harbor KRAS mutations
- In colorectal cancer (CRC): ~35-45% mutation frequency
- Non-small cell lung cancer (NSCLC): especially in lung adenocarcinoma ~25-30% mutation frequency

Identification of potent and selective GTPase KRas inhibitors remains a major objective in drug discovery.

---

## 📂 Repository Structure

```
KRASBPred/
│
├── TrainingScripts/                   # ML and DL model training scripts
├── sample.csv                         # Sample input SMILES files
├── PaDEL/                             # PaDEL descriptor software
├── padel_config_file/                 # PaDEL descriptor configuration
├── padel_destype.xml                  # Descriptor selection file
├── Dockerfile                         # Docker environment specification
├── kraspred_predict.py              # Prediction pipeline script
├── padel_scaler.pkl                   # Pre-trained normalization model
├── kras_model.h5                    # Trained multi-task DNN model
├── dependencies.txt                   # Python dependencies
├── X_train.csv                        # Training dataset
└── README.md
```

---

## ⚙️ Model Workflow

1. **Descriptor Calculation**
   Molecular descriptors are generated using PaDEL.

2. **Feature Normalization**
   Min–Max scaling using a pre-trained scaler (`padel_scaler.pkl`).

3. **Multi-Task Deep Neural Network**
   - Shared hidden layers
   - Classification head (Inhibitor/Non-Inhibitor)
   - Regression head (IC₅₀ prediction)

4. **Output Generation**
   - Probability score
   - Predicted class
   - Predicted IC₅₀ (µM)


---

## 🐳 Prerequisites

Install Docker:

https://www.docker.com/

No additional software installation is required when using Docker.

---

## 📥 Input File Format

Input must be a **CSV file** containing:

- Two columns
- Header name: `Smiles`,`Name`
- One molecule per row

### Example:

```csv
Smiles,Name
CC1=CC=CC=C1,Mol1
CCN(CC)CCOC(=O)C1=CC=CC=C1,Mol2
```

A reference file (`sample.csv`) is provided.

---

## 🚀 Usage

You can run KRASPred using either a prebuilt Docker image or by building locally.

---

### 🔹 Option 1: Use Prebuilt Docker Image (Recommended)

```bash
docker run --rm -v "${PWD}:/WorkPlace" ghcr.io/pradnyajk/kraspred:latest sample.csv output.csv
```

Replace `sample.csv` with your input file name.

---

### 🔹 Option 2: Build Docker Image Locally

#### 1️⃣ Clone Repository

```bash
git clone https://github.com/PGlab-NIPER/KRASPred.git
cd KRASPred
git lfs pull
```

If Git LFS is not installed:

```bash
git lfs install
```

---

#### 2️⃣ Build Docker Image

```bash
docker build -t kraspred .
```

---

#### 3️⃣ Run Prediction

```bash
docker run --rm -v "${PWD}:/WorkPlace" kraspred sample.csv output.csv
```

---

## 📊 Output

After execution, a file named:

```
output.csv
```

will be generated in the working directory.

### Output Columns

| Column | Description |
|---------|-------------|
| Name | Input molecule |
| Predicted_Prob | Probability of being inhibitor |
| Predicted_Class | Inhibitor / Non-Inhibitor |
| Predicted_IC50_uM | Predicted IC₅₀ value (µM) |
| Predicted_IC50_M | Predicted IC₅₀ value (M)|

---

## 🧪 Python Environment (Inside Docker)

| Package | Version |
|----------|----------|
| Python | 3.12.9 |
| pandas | 2.2.3 |
| numpy | 1.26.4 |
| joblib | 1.4.2 |
| scikit-learn | 1.6.1 |
| rdkit | 2024.9.5 |
| tensorflow | 2.18.0 |

---

## 🧠 Key Features

- Multi-task learning (classification + regression)
- Docker-based reproducibility
- Descriptor normalization included
- Pre-trained model provided
- Fully automated prediction pipeline

---

## 📖 Citation

If you use **KRASPred** in your research, please cite:

```bibtex

```

---
