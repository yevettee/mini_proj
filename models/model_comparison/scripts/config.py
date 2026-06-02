"""
YOLO 모델 비교 실험 설정
- YOLOv8, YOLO11, YOLO26 비교
- 다양한 이미지 크기, 전처리 조건 테스트
"""

# ==================== 공통 설정 ====================
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MODELS_DIR = _HERE.parent.parent  # /.../mini_proj/models

SEED = 42
EPOCHS = 100
PATIENCE = 20
BATCH_SIZE = 16
DATA_YAML = str(_MODELS_DIR / "data.yaml")
PROJECT_ROOT = str(_MODELS_DIR / "model_comparison" / "results")
RUNS_ROOT = str(_MODELS_DIR / "runs" / "model_comparison")

# ==================== 실험 정의 ====================
# 각 실험은 딕셔너리로 정의
# - name: 실험 이름 (결과 폴더명)
# - model: 모델 파일명
# - imgsz: 입력 이미지 크기
# - preprocess: 전처리 종류 (None, 'gaussian_blur', 'motion_blur', 'noise')
# - preprocess_params: 전처리 파라미터 (선택)
# - augment: Mosaic augmentation 사용 여부
# - extra_args: 추가 학습 인자 (선택)

EXPERIMENTS = [
    # ========== 1. 기본 비교 (Baseline) - 640 이미지 ==========
    {
        "name": "baseline_v8n",
        "model": "yolov8n.pt",
        "imgsz": 640,
        "preprocess": None,
        "augment": False,
        "description": "YOLOv8 nano 베이스라인"
    },
    {
        "name": "baseline_v11n",
        "model": "yolo11n.pt",
        "imgsz": 640,
        "preprocess": None,
        "augment": False,
        "description": "YOLO11 nano 베이스라인"
    },
    {
        "name": "baseline_v26n",
        "model": "yolo26n.pt",
        "imgsz": 640,
        "preprocess": None,
        "augment": False,
        "description": "YOLO26 nano 베이스라인 (NMS-free, 소형객체 강점)"
    },
    
    # ========== 2. 이미지 크기 변화 테스트 ==========
    # YOLO26의 소형 객체 탐지 강점 검증
    {
        "name": "v8n_imgsz320",
        "model": "yolov8n.pt",
        "imgsz": 320,
        "preprocess": None,
        "augment": False,
        "description": "YOLOv8n - 작은 이미지 (320)"
    },
    {
        "name": "v11n_imgsz320",
        "model": "yolo11n.pt",
        "imgsz": 320,
        "preprocess": None,
        "augment": False,
        "description": "YOLO11n - 작은 이미지 (320)"
    },
    {
        "name": "v26n_imgsz320",
        "model": "yolo26n.pt",
        "imgsz": 320,
        "preprocess": None,
        "augment": False,
        "description": "YOLO26n - 작은 이미지 (320, 강점 검증)"
    },
    {
        "name": "v8n_imgsz480",
        "model": "yolov8n.pt",
        "imgsz": 480,
        "preprocess": None,
        "augment": False,
        "description": "YOLOv8n - 중간 이미지 (480)"
    },
    {
        "name": "v11n_imgsz480",
        "model": "yolo11n.pt",
        "imgsz": 480,
        "preprocess": None,
        "augment": False,
        "description": "YOLO11n - 중간 이미지 (480)"
    },
    {
        "name": "v26n_imgsz480",
        "model": "yolo26n.pt",
        "imgsz": 480,
        "preprocess": None,
        "augment": False,
        "description": "YOLO26n - 중간 이미지 (480)"
    },
    
    # ========== 3. Blur 테스트 (품질 저하 상황) ==========
    # 카메라 초점 문제, 움직임 등 시뮬레이션
    {
        "name": "v8n_blur_light",
        "model": "yolov8n.pt",
        "imgsz": 640,
        "preprocess": "gaussian_blur",
        "preprocess_params": {"sigma": 1.5},
        "augment": False,
        "description": "YOLOv8n - 약한 블러 (sigma=1.5)"
    },
    {
        "name": "v11n_blur_light",
        "model": "yolo11n.pt",
        "imgsz": 640,
        "preprocess": "gaussian_blur",
        "preprocess_params": {"sigma": 1.5},
        "augment": False,
        "description": "YOLO11n - 약한 블러 (sigma=1.5)"
    },
    {
        "name": "v26n_blur_light",
        "model": "yolo26n.pt",
        "imgsz": 640,
        "preprocess": "gaussian_blur",
        "preprocess_params": {"sigma": 1.5},
        "augment": False,
        "description": "YOLO26n - 약한 블러 (sigma=1.5)"
    },
    {
        "name": "v8n_blur_heavy",
        "model": "yolov8n.pt",
        "imgsz": 640,
        "preprocess": "gaussian_blur",
        "preprocess_params": {"sigma": 2.5},
        "augment": False,
        "description": "YOLOv8n - 강한 블러 (sigma=2.5)"
    },
    {
        "name": "v11n_blur_heavy",
        "model": "yolo11n.pt",
        "imgsz": 640,
        "preprocess": "gaussian_blur",
        "preprocess_params": {"sigma": 2.5},
        "augment": False,
        "description": "YOLO11n - 강한 블러 (sigma=2.5)"
    },
    {
        "name": "v26n_blur_heavy",
        "model": "yolo26n.pt",
        "imgsz": 640,
        "preprocess": "gaussian_blur",
        "preprocess_params": {"sigma": 2.5},
        "augment": False,
        "description": "YOLO26n - 강한 블러 (sigma=2.5)"
    },
    
    # ========== 4. Noise 테스트 (저조도 환경) ==========
    {
        "name": "v8n_noise",
        "model": "yolov8n.pt",
        "imgsz": 640,
        "preprocess": "gaussian_noise",
        "preprocess_params": {"var": 0.02},
        "augment": False,
        "description": "YOLOv8n - 노이즈 추가"
    },
    {
        "name": "v11n_noise",
        "model": "yolo11n.pt",
        "imgsz": 640,
        "preprocess": "gaussian_noise",
        "preprocess_params": {"var": 0.02},
        "augment": False,
        "description": "YOLO11n - 노이즈 추가"
    },
    {
        "name": "v26n_noise",
        "model": "yolo26n.pt",
        "imgsz": 640,
        "preprocess": "gaussian_noise",
        "preprocess_params": {"var": 0.02},
        "augment": False,
        "description": "YOLO26n - 노이즈 추가"
    },
    
    # ========== 5. Augmentation 효과 테스트 ==========
    {
        "name": "v8n_augment",
        "model": "yolov8n.pt",
        "imgsz": 640,
        "preprocess": None,
        "augment": True,
        "description": "YOLOv8n - Mosaic augmentation ON"
    },
    {
        "name": "v11n_augment",
        "model": "yolo11n.pt",
        "imgsz": 640,
        "preprocess": None,
        "augment": True,
        "description": "YOLO11n - Mosaic augmentation ON"
    },
    {
        "name": "v26n_augment",
        "model": "yolo26n.pt",
        "imgsz": 640,
        "preprocess": None,
        "augment": True,
        "description": "YOLO26n - Mosaic augmentation ON"
    },
    
    # ========== 6. 모델 크기 비교 (nano vs small) ==========
    {
        "name": "v8s_baseline",
        "model": "yolov8s.pt",
        "imgsz": 640,
        "preprocess": None,
        "augment": False,
        "description": "YOLOv8 small - 크기 비교"
    },
    {
        "name": "v11s_baseline",
        "model": "yolo11s.pt",
        "imgsz": 640,
        "preprocess": None,
        "augment": False,
        "description": "YOLO11 small - 크기 비교"
    },
    {
        "name": "v26s_baseline",
        "model": "yolo26s.pt",
        "imgsz": 640,
        "preprocess": None,
        "augment": False,
        "description": "YOLO26 small - 크기 비교"
    },
]

# ==================== 실험 그룹 정의 ====================
# 빠른 테스트나 특정 그룹만 실행하고 싶을 때 사용
EXPERIMENT_GROUPS = {
    "baseline": ["baseline_v8n", "baseline_v11n", "baseline_v26n"],
    "imgsz_comparison": [
        "v8n_imgsz320", "v11n_imgsz320", "v26n_imgsz320",
        "v8n_imgsz480", "v11n_imgsz480", "v26n_imgsz480",
    ],
    "blur_test": [
        "v8n_blur_light", "v11n_blur_light", "v26n_blur_light",
        "v8n_blur_heavy", "v11n_blur_heavy", "v26n_blur_heavy",
    ],
    "noise_test": ["v8n_noise", "v11n_noise", "v26n_noise"],
    "augment_test": ["v8n_augment", "v11n_augment", "v26n_augment"],
    "size_comparison": ["v8s_baseline", "v11s_baseline", "v26s_baseline"],
    "quick_test": ["baseline_v8n", "baseline_v26n"],  # 빠른 2개 모델 비교
}

def get_experiments(group=None, names=None):
    """실험 목록 반환
    
    Args:
        group: 실험 그룹명 (EXPERIMENT_GROUPS의 키)
        names: 특정 실험 이름 리스트
    
    Returns:
        실험 설정 리스트
    """
    if names:
        return [exp for exp in EXPERIMENTS if exp["name"] in names]
    elif group:
        target_names = EXPERIMENT_GROUPS.get(group, [])
        return [exp for exp in EXPERIMENTS if exp["name"] in target_names]
    else:
        return EXPERIMENTS


if __name__ == "__main__":
    print(f"총 {len(EXPERIMENTS)}개 실험 정의됨")
    print("\n실험 그룹:")
    for group, names in EXPERIMENT_GROUPS.items():
        print(f"  - {group}: {len(names)}개")
    
    print("\n전체 실험 목록:")
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"  {i:2d}. {exp['name']}: {exp['description']}")
