import os
from ament_index_python.packages import get_package_share_directory


def resolve_model_path(path: str) -> str:
    """절대경로면 그대로, 아니면 패키지 models/ 에서 찾는다."""
    if os.path.isabs(path):
        return path
    return os.path.join(
        get_package_share_directory('minicar_navigator'), 'models', path
    )


def draw_box(img, x1: int, y1: int, x2: int, y2: int,
             label: str, color: tuple, thickness: int = 2) -> None:
    """바운딩박스와 라벨을 이미지에 그린다."""
    import cv2
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(img, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
