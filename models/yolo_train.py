import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

_HERE = Path(__file__).resolve().parent

# ==================== [라운드별 옵션 변경 파트] ====================
MODEL_NAME = 'yolov8n.pt'           # 예: yolov8n.pt / yolo12n.pt / yolov8l.pt 등
DATA_YAML = str(_HERE / 'data.yaml')  # 데이터셋 설정 yaml 경로 (models/data.yaml)
EPOCHS = 100                       # Round 1~3: 100, Round 4: 150
PATIENCE = 20                      # 조기 종료(Early Stopping) 대기 에포크 수 (개선이 없을 시 학습 종료, 0으로 설정하면 비활성화)
AUGMENT = False                    # Round 1~3: False, Round 4: True (Mosaic 강화)
PROJECT_NAME = 'YOLO_Tournament'   
RUN_NAME = 'Round1_v8_Nano'        # 실험별 세부 결과 폴더명 설정
SEED = 42                          # 재현성(reproducibility)을 위한 시드 값

# --- Seed별 폴더 분리 설정 (seed_42 형태로 통합) ---
# runs 를 models/runs/YOLO_Tournament/ 아래로 저장 (프로젝트 구조 정리 후)
RUNS_BASE = _HERE / "runs" / PROJECT_NAME
TRAIN_PROJECT = str(RUNS_BASE)     # e.g. models/runs/YOLO_Tournament
TRAIN_NAME = RUN_NAME              # 예: Round1_v8_Nano
# 결과 최종 경로 예시: models/runs/YOLO_Tournament/Round1_v8_Nano/
# =================================================================

# ==================== [재현성 Seed 고정] ====================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(SEED)
print(f"-> 재현성 Seed 고정 완료 (SEED={SEED})")
# =================================================================

if torch.cuda.is_available():
    device_setup = 0
    print(f"-> GPU 감지 성공: {torch.cuda.get_device_name(0)} 환경에서 학습을 시작합니다.")
else:
    device_setup = 'cpu'
    print("-> Warning: CPU 환경에서 학습합니다.")

model = YOLO(MODEL_NAME)

print(f"=== [{RUN_NAME}] 학습 시작 (저장 경로: {TRAIN_PROJECT}/{TRAIN_NAME}) ===")
model.train(
    data=DATA_YAML, epochs=EPOCHS, batch=16, imgsz=640,
    mosaic=1.0 if AUGMENT else 0.0, project=TRAIN_PROJECT, name=TRAIN_NAME, device=device_setup, plots=True,
    patience=PATIENCE, seed=SEED
)

print(f"=== [{RUN_NAME}] 최종 Test 데이터셋 검증 시작 ===")
metrics = model.val(data=DATA_YAML, split='test', device=device_setup, save_json=True)

print("\n" + "="*50)
print(f" 실험 [{RUN_NAME}] 최종 성능 리포트  (Seed={SEED})")
print(f" 저장 위치: {TRAIN_PROJECT}/{TRAIN_NAME}/")
print("="*50)
print(f"- mAP50:     {metrics.results_dict['metrics/mAP50(B)']:.3f}")
print(f"- mAP50-95:  {metrics.results_dict['metrics/mAP50-95(B)']:.3f}")
print(f"- Precision: {metrics.results_dict['metrics/precision(B)']:.3f}")
print(f"- Recall:    {metrics.results_dict['metrics/recall(B)']:.3f}")
p = metrics.results_dict['metrics/precision(B)']
r = metrics.results_dict['metrics/recall(B)']
f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0
print(f"- F1-Score:  {f1:.3f}")
print("="*50)

# ==================== [성능지표 저장] ====================
# ultralytics가 실제로 결과를 저장한 폴더 경로 사용 (runs/detect/seed_{SEED}/RUN_NAME)
try:
    save_dir = str(model.trainer.save_dir)
except Exception:
    save_dir = os.path.join(str(_HERE / "runs"), PROJECT_NAME, TRAIN_NAME)
os.makedirs(save_dir, exist_ok=True)

rows = [
    ['항목', '값'],
    ['Run',       RUN_NAME],
    ['Model',     MODEL_NAME],
    ['Epochs',    str(EPOCHS)],
    ['Seed',      str(SEED)],
    ['mAP50',     f"{metrics.results_dict['metrics/mAP50(B)']:.4f}"],
    ['mAP50-95',  f"{metrics.results_dict['metrics/mAP50-95(B)']:.4f}"],
    ['Precision', f"{p:.4f}"],
    ['Recall',    f"{r:.4f}"],
    ['F1-Score',  f"{f1:.4f}"],
]

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.axis('off')
table = ax.table(cellText=rows[1:], colLabels=rows[0],
                 cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.4, 1.6)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#EAF2FB')
    else:
        cell.set_facecolor('#FDFEFE')

ax.set_title(f'[{RUN_NAME}] Test Metrics', fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
table_path = os.path.join(save_dir, 'metrics_table.png')
plt.savefig(table_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"-> 성능지표 표 이미지 저장 완료: {table_path}")
print(f"-> 전체 실험 결과 저장 경로: {save_dir}")
# =========================================================
