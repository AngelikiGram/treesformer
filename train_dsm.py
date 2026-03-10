import os
import subprocess
import sys

# Simple wrapper to train with ONLY DSM (3D Point Cloud) signals
# Use: python train_dsm.py [additional flags]

command = [
    sys.executable, "train.py",
    "--modality", "dsm",
] + sys.argv[1:]

print(f"[INFO] Launching DSM-ONLY training...")
subprocess.run(command)
