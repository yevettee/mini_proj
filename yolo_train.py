import torch
import csv
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ==================== [라운드별 옵션 변경 파트] ====================
MODEL_NAME = 'yolov8n.pt'          # 예: yolov8n.pt / yolov12n.pt / yolov8l.pt 등
DATA_YAML = './data.yaml'          # 데이터셋 설정 yaml 경로 (Round 3 압축 실험 시 압축용 yaml로 변경)
EPOCHS = 100                       # Round 1~3: 100, Round 4: 150
PATIENCE = 20                      # 조기 종료(Early Stopping) 대기 에포크 수 (개선이 없을 시 학습 종료, 0으로 설정하면 비활성화)
AUGMENT = False                    # Round 1~3: False, Round 4: True (Mosaic 강화)
PROJECT_NAME = 'YOLO_Tournament'   
RUN_NAME = 'Round1_v8_Nano'        # 실험별 세부 결과 폴더명 설정
# =================================================================

if torch.cuda.is_available():
    device_setup = 0
    print(f"-> GPU 감지 성공: {torch.cuda.get_device_name(0)} 환경에서 학습을 시작합니다.")
else:
    device_setup = 'cpu'
    print("-> Warning: CPU 환경에서 학습합니다.")

model = YOLO(MODEL_NAME)

print(f"=== [{RUN_NAME}] 학습 시작 ===")
model.train(
    data=DATA_YAML, epochs=EPOCHS, batch=16, imgsz=640,
    mosaic=1.0 if AUGMENT else 0.0, project=PROJECT_NAME, name=RUN_NAME, device=device_setup, plots=True,
    patience=PATIENCE
)

print(f"=== [{RUN_NAME}] 최종 Test 데이터셋 검증 시작 ===")
metrics = model.val(data=DATA_YAML, split='test', device=device_setup)

print("\n" + "="*50)
print(f" 실험 [{RUN_NAME}] 최종 성능 리포트")
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
save_dir = os.path.join(PROJECT_NAME, RUN_NAME)
os.makedirs(save_dir, exist_ok=True)

metric_data = {
    'Run': RUN_NAME,
    'Model': MODEL_NAME,
    'Epochs': EPOCHS,
    'mAP50': round(metrics.results_dict['metrics/mAP50(B)'], 4),
    'mAP50-95': round(metrics.results_dict['metrics/mAP50-95(B)'], 4),
    'Precision': round(p, 4),
    'Recall': round(r, 4),
    'F1-Score': round(f1, 4),
}

# CSV 저장 (누적 기록용)
csv_path = os.path.join(PROJECT_NAME, 'results_summary.csv')
write_header = not os.path.exists(csv_path)
with open(csv_path, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=metric_data.keys())
    if write_header:
        writer.writeheader()
    writer.writerow(metric_data)
print(f"-> CSV 저장 완료: {csv_path}")

# 막대그래프 이미지 저장
labels = ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1-Score']
values = [metric_data['mAP50'], metric_data['mAP50-95'],
          metric_data['Precision'], metric_data['Recall'], metric_data['F1-Score']]
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, values, color=colors, width=0.5)
ax.set_ylim(0, 1.05)
ax.set_ylabel('Score')
ax.set_title(f'[{RUN_NAME}] Test Metrics')
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
chart_path = os.path.join(save_dir, 'metrics_chart.png')
plt.savefig(chart_path, dpi=150)
plt.close()
print(f"-> 차트 이미지 저장 완료: {chart_path}")
# =========================================================
