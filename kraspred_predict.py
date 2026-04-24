#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from rdkit import Chem
from rdkit.Chem import MolStandardize

# ============================================================
# SMILES STANDARDIZATION
# ============================================================
def standardize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None

        try:
            Chem.SanitizeMol(mol)
        except:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                            Chem.SanitizeFlags.SANITIZE_KEKULIZE
            )

        normalizer = MolStandardize.normalize.Normalizer()
        mol = normalizer.normalize(mol)

        reionizer = MolStandardize.charge.Reionizer()
        mol = reionizer.reionize(mol)

        return Chem.MolToSmiles(mol, canonical=True)

    except:
        return None

# ============================================================
# Args
# ============================================================
parser = argparse.ArgumentParser(description="Predict inhibitor and pIC50 from SMILES")
parser.add_argument("input_csv")
parser.add_argument("output_csv")
args = parser.parse_args()


# ============================================================
# Load input
# ============================================================
try:
    df = pd.read_csv(args.input_csv)
    df.columns = df.columns.str.strip()

    df["Name"] = df["Name"].astype(str).str.strip()
    df["Smiles"] = df["Smiles"].astype(str).str.strip()

except Exception:
    sys.exit("ERROR: Cannot read input CSV")

if not {"Name", "Smiles"}.issubset(df.columns):
    sys.exit("ERROR: CSV must contain Name and Smiles")

if len(df) == 0:
    sys.exit("ERROR: Empty input file")

df = df.reset_index(drop=True)
df["_ID"] = [f"Mol_{i}" for i in range(len(df))]


# ============================================================
# SMILES NORMALIZATION
# ============================================================
print("Standardizing SMILES...")
df["Std_SMILES"] = df["Smiles"].apply(standardize_smiles)
df["Final_SMILES"] = df["Std_SMILES"].fillna(df["Smiles"])

# ============================================================
# Paths
# ============================================================
BASE = Path(__file__).resolve().parent

PADEL = BASE / "PaDEL" / "PaDEL-Descriptor.jar"
DESTYPE = BASE / "padel_destype.xml"
XTRAIN = BASE / "X_train.csv"
SCALER = BASE / "padel_scaler.pkl"
MODEL = BASE / "kras_model.h5"

for f in [PADEL, DESTYPE, XTRAIN, SCALER, MODEL]:
    if not f.exists():
        sys.exit(f"ERROR: Missing file {f}")


# ============================================================
# Temp workspace
# ============================================================
tmp = Path(tempfile.mkdtemp())
padel_dir = tmp / "padel"
padel_dir.mkdir()

try:
    # ========================================================
    # Write SMILES
    # ========================================================
    smi = padel_dir / "mol.smi"
    with open(smi, "w") as f:
        for _, r in df.iterrows():
            f.write(f"{r['Final_SMILES']} {r['_ID']}\n")

    # ========================================================
    # Run PaDEL
    # ========================================================
    desc_file = tmp / "desc.csv"
    config = tmp / "padel.cfg"

    with open(config, "w") as f:
        f.write(f"""Compute2D=true
Compute3D=false
ComputeFingerprints=true
Convert3D=No
Directory={padel_dir.as_posix()}
DescriptorFile={desc_file.as_posix()}
DetectAromaticity=true
Log=true
MaxCpdPerFile=0
MaxJobsWaiting=-1
MaxRunTime=600000
MaxThreads=-1
RemoveSalt=true
Retain3D=false
RetainOrder=true
StandardizeNitro=true
StandardizeTautomers=false
UseFilenameAsMolName=false
""")

    subprocess.run(
        ["java", "-jar", str(PADEL), "-config", str(config), "-descriptortypes", str(DESTYPE)],
        check=True
    )

    # ========================================================
    # Validate descriptors
    # ========================================================
    print("\nChecking descriptor file...")

    if not desc_file.exists():
        sys.exit("ERROR: desc.csv not created")

    if desc_file.stat().st_size == 0:
        sys.exit("ERROR: desc.csv is empty")

    desc = pd.read_csv(desc_file)
    print("Descriptor shape:", desc.shape)

    # ========================================================
    # FEATURE ALIGNMENT (FINAL FIX)
    # ========================================================
    print("\nAligning descriptors with training data...")

    X_train = pd.read_csv(XTRAIN)
    X_train = X_train.apply(pd.to_numeric, errors='coerce')

    train_cols = list(X_train.columns)
    train_means = X_train.mean()

    # Drop PaDEL Name column
    padel_features = desc.drop(columns=['Name'], errors='ignore')

    # Convert numeric
    padel_features = padel_features.apply(pd.to_numeric, errors='coerce')

    # Reindex EXACTLY like training
    aligned = padel_features.reindex(columns=train_cols)

    # Fill missing values
    aligned = aligned.replace([np.inf, -np.inf], np.nan)
    aligned = aligned.fillna(train_means)

    if aligned.shape[0] == 0:
        sys.exit("ERROR: No valid descriptors after alignment")

    print("Aligned shape:", aligned.shape)

    # ========================================================
    # Predict
    # ========================================================
    scaler = joblib.load(SCALER)
    model = load_model(MODEL, compile=False)

    Xs = scaler.transform(aligned)
    pred = model.predict(Xs, verbose=0)

    prob = pred[0].ravel()
    pic50 = pred[1].ravel()

    # ========================================================
    # Output reconstruction
    # ========================================================
    label = np.where(prob >= 0.5, "Inhibitor", "Non-inhibitor")

    pic50_clipped = np.clip(pic50, 0.5, 11.5)
    ic50_uM = 10 ** (6 - pic50_clipped)

    ic50_display = [f"{v:.3g}" for v in ic50_uM]

    out = pd.DataFrame({
        "Name": df["Name"],
        "Predicted Probability": prob,
        "Predicted Class": label,
        "Predicted IC50 (μM)": ic50_display
    })

    out.to_csv(args.output_csv, index=False)
    print(f"\nSaved → {args.output_csv}")

finally:
    shutil.rmtree(tmp, ignore_errors=True)
