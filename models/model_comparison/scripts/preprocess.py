"""
Image Preprocessing Module
- Simulate quality degradation (blur, noise)
- Copy dataset and apply preprocessing
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import yaml


def apply_gaussian_blur(image, sigma=1.5):
    """Apply Gaussian blur"""
    ksize = int(sigma * 6) | 1  # Convert to odd number
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def apply_motion_blur(image, size=15, angle=0):
    """Apply motion blur"""
    kernel = np.zeros((size, size))
    kernel[int((size-1)/2), :] = np.ones(size)
    kernel = kernel / size
    
    # Apply rotation
    M = cv2.getRotationMatrix2D((size/2, size/2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (size, size))
    
    return cv2.filter2D(image, -1, kernel)


def apply_gaussian_noise(image, mean=0, var=0.02):
    """Add Gaussian noise"""
    image = image.astype(np.float32) / 255.0
    noise = np.random.normal(mean, var ** 0.5, image.shape).astype(np.float32)
    noisy = np.clip(image + noise, 0, 1)
    return (noisy * 255).astype(np.uint8)


def apply_brightness_contrast(image, alpha=1.0, beta=0):
    """Adjust brightness/contrast
    alpha: contrast (1.0 = original)
    beta: brightness (-100 ~ 100)
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


PREPROCESS_FUNCTIONS = {
    "gaussian_blur": apply_gaussian_blur,
    "motion_blur": apply_motion_blur,
    "gaussian_noise": apply_gaussian_noise,
    "brightness_contrast": apply_brightness_contrast,
}


def process_single_image(args):
    """Preprocess single image (for multithreading)"""
    src_path, dst_path, preprocess_type, params = args
    
    try:
        image = cv2.imread(str(src_path))
        if image is None:
            print(f"Warning: Cannot read {src_path}")
            return False
        
        func = PREPROCESS_FUNCTIONS.get(preprocess_type)
        if func:
            processed = func(image, **params)
        else:
            processed = image
        
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cv2.imwrite(str(dst_path), processed)
        return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False


def create_preprocessed_dataset(
    original_data_yaml: str,
    preprocess_type: str,
    preprocess_params: dict = None,
    output_suffix: str = None,
    max_workers: int = 4
):
    """Create preprocessed dataset
    
    Args:
        original_data_yaml: Original data.yaml path
        preprocess_type: Preprocessing type (gaussian_blur, motion_blur, gaussian_noise, etc.)
        preprocess_params: Preprocessing parameters
        output_suffix: Output folder suffix (default: preprocess_type)
        max_workers: Number of parallel workers
    
    Returns:
        Newly created data.yaml path
    """
    if preprocess_params is None:
        preprocess_params = {}
    
    if output_suffix is None:
        param_str = "_".join(f"{k}{v}" for k, v in preprocess_params.items())
        output_suffix = f"{preprocess_type}_{param_str}" if param_str else preprocess_type
    
    # Load original yaml
    with open(original_data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    original_base = Path(original_data_yaml).parent
    original_path = original_base / data_config['path']
    
    # New dataset path
    new_dataset_name = f"dataset_{output_suffix}"
    new_dataset_path = original_base / new_dataset_name
    
    # Return existing path if already exists
    new_yaml_path = original_base / f"data_{output_suffix}.yaml"
    if new_yaml_path.exists():
        print(f"Preprocessed dataset already exists: {new_yaml_path}")
        return str(new_yaml_path)
    
    print(f"Creating preprocessed dataset: {preprocess_type} ({preprocess_params})")
    print(f"  Source: {original_path}")
    print(f"  Target: {new_dataset_path}")
    
    # Collect image files and create preprocessing tasks
    tasks = []
    for split in ['train', 'val', 'test']:
        split_path = data_config.get(split, f'images/{split}')
        src_images_dir = original_path / split_path
        dst_images_dir = new_dataset_path / split_path
        
        if not src_images_dir.exists():
            print(f"  Warning: {src_images_dir} not found, skipping")
            continue
        
        for img_file in src_images_dir.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                dst_file = dst_images_dir / img_file.name
                tasks.append((img_file, dst_file, preprocess_type, preprocess_params))
    
    # Parallel processing
    print(f"  Preprocessing {len(tasks)} images...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single_image, tasks))
    
    success_count = sum(results)
    print(f"  Done: {success_count}/{len(tasks)} succeeded")
    
    # Copy label files (unchanged)
    for split in ['train', 'val', 'test']:
        split_path = data_config.get(split, f'images/{split}')
        src_labels_dir = original_path / split_path.replace('images', 'labels')
        dst_labels_dir = new_dataset_path / split_path.replace('images', 'labels')
        
        if src_labels_dir.exists():
            shutil.copytree(src_labels_dir, dst_labels_dir, dirs_exist_ok=True)
    
    # Create new data.yaml
    new_data_config = data_config.copy()
    new_data_config['path'] = f"./{new_dataset_name}"
    
    with open(new_yaml_path, 'w') as f:
        yaml.dump(new_data_config, f, default_flow_style=False)
    
    print(f"  New data.yaml created: {new_yaml_path}")
    return str(new_yaml_path)


def get_data_yaml_for_experiment(experiment_config: dict, base_data_yaml: str):
    """Return data.yaml path for experiment config
    
    Creates/returns preprocessed dataset if preprocessing required
    """
    preprocess = experiment_config.get("preprocess")
    
    if preprocess is None:
        return base_data_yaml
    
    params = experiment_config.get("preprocess_params", {})
    return create_preprocessed_dataset(
        original_data_yaml=base_data_yaml,
        preprocess_type=preprocess,
        preprocess_params=params
    )


if __name__ == "__main__":
    # Test: Create blur dataset
    import sys
    
    if len(sys.argv) > 1:
        data_yaml = sys.argv[1]
    else:
        # robust default after models/ reorg
        data_yaml = str(Path(__file__).resolve().parent.parent.parent / "data.yaml")
    
    # Example: Create Gaussian blur dataset
    new_yaml = create_preprocessed_dataset(
        data_yaml,
        preprocess_type="gaussian_blur",
        preprocess_params={"sigma": 1.5}
    )
    print(f"Created data.yaml: {new_yaml}")
