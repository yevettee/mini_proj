# YOLO 모델 비교 실험 파이프라인

YOLOv8, YOLO11, YOLO26을 다양한 조건에서 비교하는 실험 파이프라인입니다.

## 구조

```
models/model_comparison/scripts/
├── config.py      # 실험 설정 정의
├── preprocess.py  # 이미지 전처리 (blur, noise)
├── runner.py      # 실험 실행기
├── analyzer.py    # 결과 분석 및 시각화
├── __init__.py
└── README.md
```

## 실험 목록

### 1. Baseline 비교
- `baseline_v8n`, `baseline_v11n`, `baseline_v26n`
- 동일 조건(imgsz=640, no augment)에서 순수 모델 성능 비교

### 2. 이미지 크기 변화
- 320, 480, 640 크기에서 각 모델 테스트
- YOLO26의 소형 객체 탐지 강점 검증

### 3. 품질 저하 테스트 (Blur/Noise)
- Gaussian Blur (sigma=1.5, 2.5)
- Gaussian Noise
- 모델 강건성 비교

### 4. Augmentation 효과
- Mosaic augmentation ON/OFF 비교

### 5. 모델 크기 비교
- Nano vs Small 성능/속도 트레이드오프

## 사용법

### 전체 파이프라인 실행

```bash
cd /home/woody/mini_proj
python models/model_comparison/scripts/runner.py
```

### 특정 그룹만 실행

```bash
# Baseline만
python models/model_comparison/scripts/runner.py --group baseline

# 이미지 크기 비교만
python models/model_comparison/scripts/runner.py --group imgsz_comparison

# Blur 테스트만
python models/model_comparison/scripts/runner.py --group blur_test

# 빠른 테스트 (v8n vs v26n)
python models/model_comparison/scripts/runner.py --group quick_test
```

### 특정 실험만 실행

```bash
python models/model_comparison/scripts/runner.py --only baseline_v26n
```

### 에포크/배치 조절

```bash
python models/model_comparison/scripts/runner.py --epochs 50 --batch 32
```

### 실험 목록 확인

```bash
python models/model_comparison/scripts/runner.py --list
python models/model_comparison/scripts/runner.py --list-groups
```

### 결과 분석

```bash
python models/model_comparison/scripts/analyzer.py
```

## 결과 파일

실행 후 `results/` 폴더에 다음 파일들이 생성됩니다:

- `summary.csv` - 전체 실험 결과 통합
- `comparison_charts/` - 비교 차트 이미지
  - `baseline_comparison.png` - 기본 모델 비교
  - `imgsz_comparison.png` - 이미지 크기별 성능
  - `degradation_comparison.png` - Blur/Noise 강건성
  - `augmentation_effect.png` - Augmentation 효과
  - `model_size_comparison.png` - Nano vs Small
  - `comprehensive_heatmap.png` - 전체 결과 히트맵
  - `summary_table.png` - 요약 테이블
  - `best_model_report.txt` - 최고 성능 모델 리포트

## 중단 및 재개

파이프라인은 자동으로 완료된 실험을 스킵합니다.
중단 후 다시 실행하면 남은 실험부터 이어서 진행됩니다.

강제로 다시 실행하려면:
```bash
python models/model_comparison/scripts/runner.py --no-skip
```

## 설정 커스터마이징

`config.py`에서 다음을 수정할 수 있습니다:

- `SEED`: 재현성 시드 (기본: 42)
- `EPOCHS`: 학습 에포크 (기본: 100)
- `PATIENCE`: Early stopping patience (기본: 20)
- `BATCH_SIZE`: 배치 크기 (기본: 16)
- `EXPERIMENTS`: 실험 목록 추가/수정

## 새 실험 추가

`config.py`의 `EXPERIMENTS` 리스트에 추가:

```python
{
    "name": "my_experiment",
    "model": "yolo26n.pt",
    "imgsz": 640,
    "preprocess": "gaussian_blur",  # None, gaussian_blur, motion_blur, gaussian_noise
    "preprocess_params": {"sigma": 2.0},
    "augment": True,
    "description": "내 커스텀 실험"
}
```
