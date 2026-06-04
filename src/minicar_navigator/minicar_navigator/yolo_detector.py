#!/usr/bin/env python3
import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from .utils import draw_box, resolve_model_path

cv2.setLogLevel(2)

COLOR_TARGET = (0, 255, 0)
COLOR_OTHER  = (0, 165, 255)
COLOR_NONE   = (0, 0, 255)


class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector')

        self.declare_parameter('model_path',   'best.pt')
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('confidence',   0.86)
        self.declare_parameter('target_class', 'car')
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('image_width',  640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('roi_x_min', 0.0)
        self.declare_parameter('roi_x_max', 1.0)
        self.declare_parameter('roi_y_min', 0.0)
        self.declare_parameter('roi_y_max', 1.0)

        model_path        = resolve_model_path(self.get_parameter('model_path').value)
        self.camera_index = self.get_parameter('camera_index').value
        self.confidence   = self.get_parameter('confidence').value
        self.target_class = self.get_parameter('target_class').value.lower()
        publish_rate      = self.get_parameter('publish_rate').value
        self.image_width  = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        self.roi_x_min    = self.get_parameter('roi_x_min').value
        self.roi_x_max    = self.get_parameter('roi_x_max').value
        self.roi_y_min    = self.get_parameter('roi_y_min').value
        self.roi_y_max    = self.get_parameter('roi_y_max').value

        self.model      = YOLO(model_path)
        self.classNames = self.model.names if hasattr(self.model, 'names') else {}
        self.bridge     = CvBridge()

        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.image_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)

        self.pub_detected = self.create_publisher(Bool,  '/minicar_detected', 10)
        self.pub_image    = self.create_publisher(Image, '/detection_image',  10)

        self.create_timer(1.0 / publish_rate, self._tick)

        self.get_logger().info(
            f'YoloDetector ready | model={model_path} | '
            f'target="{self.target_class}" | conf={self.confidence} | camera={self.camera_index} | '
            f'roi=({self.roi_x_min:.2f},{self.roi_y_min:.2f})-({self.roi_x_max:.2f},{self.roi_y_max:.2f})'
        )

    def _tick(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to read frame')
            return

        results   = self.model(frame, conf=self.confidence, verbose=False)
        triggered = False
        annotated = frame.copy()

        # ROI 픽셀 좌표 계산 및 시각화
        roi_px1 = int(self.roi_x_min * self.image_width)
        roi_py1 = int(self.roi_y_min * self.image_height)
        roi_px2 = int(self.roi_x_max * self.image_width)
        roi_py2 = int(self.roi_y_max * self.image_height)
        cv2.rectangle(annotated, (roi_px1, roi_py1), (roi_px2, roi_py2), (255, 255, 0), 2)

        for result in results:
            for box in result.boxes:
                cls_id   = int(box.cls[0])
                cls_name = self.classNames.get(cls_id, '').lower()
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 박스 중심이 ROI 안에 있어야 유효한 탐지로 처리
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                in_roi = roi_px1 <= cx <= roi_px2 and roi_py1 <= cy <= roi_py2

                is_target = cls_name == self.target_class and in_roi
                if is_target:
                    triggered = True
                color = COLOR_TARGET if is_target else COLOR_OTHER
                draw_box(annotated, x1, y1, x2, y2, f'{cls_name} {conf_val:.2f}', color)

        status = 'CAR DETECTED' if triggered else 'No detection'
        cv2.putText(annotated, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, COLOR_TARGET if triggered else COLOR_NONE, 2)

        self.pub_detected.publish(Bool(data=triggered))
        try:
            msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()
            self.pub_image.publish(msg)
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
