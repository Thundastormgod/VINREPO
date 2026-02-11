#!/usr/bin/env python3
"""Debug label file issue."""

import os
from pathlib import Path

label_file = Path("vin_ocr_data/preprocessed_extended/train/train_labels.txt")
print(f"Label file exists: {label_file.exists()}")
print(f"Label file size: {label_file.stat().st_size} bytes")

with open(label_file, 'r') as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")
print("\nFirst 3 lines:")
for i, line in enumerate(lines[:3]):
    print(f"  Line {i+1}: {repr(line[:100])}")
    parts = line.strip().split('\t')
    print(f"    Parts after split: {len(parts)}")
    if len(parts) >= 1:
        img_path = parts[0]
        print(f"    Image path: {img_path}")
        print(f"    Exists: {os.path.exists(img_path)}")
    if len(parts) >= 2:
        label = parts[1]
        print(f"    Label: {label}")

print("\n\nImages directory check:")
images_dir = Path("vin_ocr_data/preprocessed_extended/train/images")
print(f"Directory exists: {images_dir.exists()}")
if images_dir.exists():
    images = list(images_dir.glob("*"))
    print(f"Image count: {len(images)}")
    print(f"First 3 images: {[img.name for img in images[:3]]}")
