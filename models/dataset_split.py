import os
import random
import shutil
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# Reorganized structure: images live in cam_images/*_capture_threaded/, labels in *_labelled/
# (labels share basenames with images)
AMR_IMAGES_DIR = _HERE / "cam_images" / "amr_img_capture_threaded"
AMR_LABELS_DIR = _HERE / "cam_images" / "amr_img_labelled"
WEBCAM_IMAGES_DIR = _HERE / "cam_images" / "img_capture_threaded"
WEBCAM_LABELS_DIR = _HERE / "cam_images" / "img_labelled"

OUTPUT_DIR = str(_HERE / "dataset_split")

for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

def collect_pairs():
    """Return list of (img_path, lbl_path_or_None, domain) from current cam_images layout."""
    pairs = []
    # AMR domain
    if AMR_IMAGES_DIR.exists():
        for img in sorted(AMR_IMAGES_DIR.glob("*.jpg")):
            lbl = AMR_LABELS_DIR / f"{img.stem}.txt"
            pairs.append((img, lbl if lbl.exists() else None, "amr"))
    # Webcam domain (img_*)
    if WEBCAM_IMAGES_DIR.exists():
        for img in sorted(WEBCAM_IMAGES_DIR.glob("*.jpg")):
            lbl = WEBCAM_LABELS_DIR / f"{img.stem}.txt"
            pairs.append((img, lbl if lbl.exists() else None, "webcam"))
    return pairs

random.seed(42)

def split_and_copy(pairs, domain_name):
    if not pairs:
        return
    random.shuffle(pairs)
    total = len(pairs)
    train_end = int(total * 0.7)
    val_end = int(total * 0.9)
    splits = {
        'train': pairs[:train_end],
        'val': pairs[train_end:val_end],
        'test': pairs[val_end:]
    }
    
    for split_type, file_pairs in splits.items():
        for img_path, lbl_path, _ in file_pairs:
            img_name = img_path.name
            shutil.copy(img_path, os.path.join(OUTPUT_DIR, 'images', split_type, img_name))
            if lbl_path is not None and lbl_path.exists():
                shutil.copy(lbl_path, os.path.join(OUTPUT_DIR, 'labels', split_type, lbl_path.name))
    print(f"[{domain_name}] 분할 완료 -> Train: {len(splits['train'])}장, Val: {len(splits['val'])}장, Test: {len(splits['test'])}장")

all_pairs = collect_pairs()
amr_pairs = [p for p in all_pairs if p[2] == "amr"]
webcam_pairs = [p for p in all_pairs if p[2] == "webcam"]

split_and_copy(amr_pairs, "TurtleBot 도메인 (amr_*)")
split_and_copy(webcam_pairs, "Webcam 도메인 (img_*)")
print("\n🔥 모든 도메인이 균등하게 7:2:1 분할 정렬되었습니다.")
