"""
Quick test: run the ANPR pipeline on individual extracted frames
and print exactly what EasyOCR detects in each one.
"""
import sys, time
sys.path.insert(0, ".")

import cv2
from app.anpr import _get_reader, _process_frame, _find_plate_regions

FRAMES = [30, 90, 150, 210, 300, 400, 500]

print("Loading EasyOCR reader (first call may take 30-60s)...")
t0 = time.time()
reader = _get_reader()
print(f"Reader ready in {time.time()-t0:.1f}s\n")

for fn in FRAMES:
    path = f"test_frame_{fn}.jpg"
    frame = cv2.imread(path)
    if frame is None:
        print(f"Frame {fn}: file not found, skipping")
        continue

    t1 = time.time()
    plates = _process_frame(frame, reader)
    elapsed = time.time() - t1

    regions = _find_plate_regions(frame)
    print(f"Frame {fn:>3}  ({elapsed:.2f}s)  regions={len(regions)}  plates={plates}")

print("\nDone.")
