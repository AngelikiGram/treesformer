import os
import subprocess
import sys

# Simple wrapper to train with ONLY Orthophoto (RGB) signals
# Use: python train_ortho.py [additional flags]

command = [
    sys.executable, "train.py",
    "--modality", "ortho",
] + sys.argv[1:]

print(f"[INFO] Launching ORTHO-ONLY training...")
subprocess.run(command)
