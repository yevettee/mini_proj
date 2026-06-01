import os
import random
import shutil

SOURCE_DIR = "./all_images" 
OUTPUT_DIR = "./dataset_split"

for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

all_files = os.listdir(SOURCE_DIR)
image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
images = [f for f in all_files if f.endswith(image_extensions)]

turtle_images = [img for img in images if img.lower().startswith("amr_")]
webcam_images = [img for img in images if img.lower().startswith("img_")]

etc_images = [img for img in images if img not in turtle_images and img not in webcam_images]
webcam_images.extend(etc_images)

random.seed(42)

def split_and_copy(image_list, domain_name):
    random.shuffle(image_list)
    total = len(image_list)
    if total == 0: return

    train_end = int(total * 0.7)
    val_end = int(total * 0.9)
    splits = {'train': image_list[:train_end], 'val': image_list[train_end:val_end], 'test': image_list[val_end:]}
    
    for split_type, file_list in splits.items():
        for img_name in file_list:
            base_name = os.path.splitext(img_name)[0]
            lbl_name = base_name + ".txt"
            shutil.copy(os.path.join(SOURCE_DIR, img_name), os.path.join(OUTPUT_DIR, 'images', split_type, img_name))
            if os.path.exists(os.path.join(SOURCE_DIR, lbl_name)):
                shutil.copy(os.path.join(SOURCE_DIR, lbl_name), os.path.join(OUTPUT_DIR, 'labels', split_type, lbl_name))
    print(f"[{domain_name}] 분할 완료 -> Train: {len(splits['train'])}장, Val: {len(splits['val'])}장, Test: {len(splits['test'])}장")

split_and_copy(turtle_images, "TurtleBot 도메인 (amr_*)")
split_and_copy(webcam_images, "Webcam 도메인 (img_*)")
print("\n🔥 모든 도메인이 균등하게 7:2:1 분할 정렬되었습니다.")
