"""
IMCITS — Target Person of Interest (POI) Registration Utility
============================================================
Enrolls reference POI photos into the `targets/` registry.

Each call ADDS a new reference photo for that name (reference_1.jpg,
reference_2.jpg, ...) instead of overwriting the existing one — the
Re-ID gallery in poi_detector_matcher.py averages up to 15 embeddings
per identity, so multiple angles/lighting conditions of the same person
meaningfully improve match reliability, especially with the ImageNet-only
OSNet backbone (see the warning in poi_detector_matcher.py).

Usage:
  python src/register_poi.py --name "John_Doe" --image "path/to/photo.jpg"
"""
from __future__ import annotations

import os
import sys
import argparse
import shutil
import glob
import cv2

def register_target(name: str, image_path: str, targets_dir: str = "targets") -> bool:
    if not os.path.exists(image_path):
        print(f"[ERROR] Target image file not found at: '{image_path}'")
        return False

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Failed to decode image file: '{image_path}'")
        return False

    target_folder = os.path.join(targets_dir, name.strip().replace(" ", "_"))
    os.makedirs(target_folder, exist_ok=True)

    ext = os.path.splitext(image_path)[1].lower() or ".jpg"

    # Find the next free reference_N slot so we never overwrite an existing photo.
    existing = glob.glob(os.path.join(target_folder, "reference_*.*"))
    existing_nums = []
    for f in existing:
        stem = os.path.splitext(os.path.basename(f))[0]  # e.g. "reference_3"
        try:
            existing_nums.append(int(stem.split("_")[-1]))
        except ValueError:
            continue
    next_num = (max(existing_nums) + 1) if existing_nums else 1
    dest_path = os.path.join(target_folder, f"reference_{next_num}{ext}")

    shutil.copy2(image_path, dest_path)
    print("=" * 60)
    print(f" [SUCCESS] POI Reference Photo Added!")
    print(f"   - Name         : {name}")
    print(f"   - Source Image : {image_path}")
    print(f"   - Stored Path  : {dest_path}")
    print(f"   - Target Specs : {img.shape[1]}x{img.shape[0]} px")
    print(f"   - Total refs   : {len(existing) + 1} photo(s) enrolled for this name")
    print("=" * 60)
    return True

def main():
    parser = argparse.ArgumentParser(description="IMCITS POI Target Registration Utility")
    parser.add_argument("--name", type=str, required=True, help="Name or Identifier of the Person of Interest")
    parser.add_argument("--image", type=str, required=True, help="Path to reference image photo")
    parser.add_argument("--targets-dir", type=str, default="targets", help="Targets directory path")

    args = parser.parse_args()
    register_target(args.name, args.image, args.targets_dir)

if __name__ == "__main__":
    main()