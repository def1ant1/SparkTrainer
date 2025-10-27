"""
Object detection module for SparkTrainer.

Provides integration with YOLOv8 for automated object detection and labeling.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class YOLOv8Detector:
    """
    YOLOv8-based object detector for automated image labeling.

    Supports:
    - Object detection
    - Instance segmentation
    - Classification
    - Pose estimation
    """

    def __init__(self, model_name: str = "yolov8n.pt", device: Optional[str] = None):
        """
        Initialize YOLOv8 detector.

        Args:
            model_name: YOLOv8 model variant (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the YOLOv8 model"""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package not installed. Install with: pip install ultralytics"
            )

        try:
            self.model = YOLO(self.model_name)
            if self.device:
                self.model.to(self.device)
            logger.info(f"Loaded YOLOv8 model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            raise

    def detect(self,
               image_path: Union[str, Path],
               conf_threshold: float = 0.25,
               iou_threshold: float = 0.45,
               max_detections: int = 300) -> List[Dict[str, Any]]:
        """
        Detect objects in an image.

        Args:
            image_path: Path to image file
            conf_threshold: Confidence threshold (0-1)
            iou_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections

        Returns:
            List of detections, each containing:
            - bbox: [x1, y1, x2, y2]
            - confidence: float
            - class_id: int
            - class_name: str
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            results = self.model(
                str(image_path),
                conf=conf_threshold,
                iou=iou_threshold,
                max_det=max_detections,
                verbose=False
            )

            detections = []
            for result in results:
                boxes = result.boxes
                for i in range(len(boxes)):
                    detection = {
                        'bbox': boxes.xyxy[i].cpu().numpy().tolist(),
                        'confidence': float(boxes.conf[i].cpu()),
                        'class_id': int(boxes.cls[i].cpu()),
                        'class_name': result.names[int(boxes.cls[i].cpu())]
                    }
                    detections.append(detection)

            return detections
        except Exception as e:
            logger.error(f"Detection failed for {image_path}: {e}")
            return []

    def detect_batch(self,
                     image_paths: List[Union[str, Path]],
                     conf_threshold: float = 0.25,
                     iou_threshold: float = 0.45,
                     max_detections: int = 300,
                     batch_size: int = 32) -> List[List[Dict[str, Any]]]:
        """
        Detect objects in multiple images.

        Args:
            image_paths: List of image paths
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            max_detections: Maximum detections per image
            batch_size: Batch size for processing

        Returns:
            List of detection lists (one per image)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        all_detections = []

        try:
            # Process in batches
            for i in range(0, len(image_paths), batch_size):
                batch = image_paths[i:i + batch_size]
                results = self.model(
                    [str(p) for p in batch],
                    conf=conf_threshold,
                    iou=iou_threshold,
                    max_det=max_detections,
                    verbose=False
                )

                for result in results:
                    detections = []
                    boxes = result.boxes
                    for j in range(len(boxes)):
                        detection = {
                            'bbox': boxes.xyxy[j].cpu().numpy().tolist(),
                            'confidence': float(boxes.conf[j].cpu()),
                            'class_id': int(boxes.cls[j].cpu()),
                            'class_name': result.names[int(boxes.cls[j].cpu())]
                        }
                        detections.append(detection)
                    all_detections.append(detections)

            return all_detections
        except Exception as e:
            logger.error(f"Batch detection failed: {e}")
            return [[] for _ in image_paths]

    def segment(self,
                image_path: Union[str, Path],
                conf_threshold: float = 0.25,
                iou_threshold: float = 0.45) -> List[Dict[str, Any]]:
        """
        Perform instance segmentation on an image.

        Args:
            image_path: Path to image
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold

        Returns:
            List of segmentations with masks
        """
        # Load segmentation model if not already loaded
        if "seg" not in self.model_name:
            logger.warning("Using detection model for segmentation. Consider using yolov8*-seg.pt")

        try:
            results = self.model(
                str(image_path),
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )

            segmentations = []
            for result in results:
                if hasattr(result, 'masks') and result.masks is not None:
                    boxes = result.boxes
                    masks = result.masks
                    for i in range(len(boxes)):
                        segmentation = {
                            'bbox': boxes.xyxy[i].cpu().numpy().tolist(),
                            'confidence': float(boxes.conf[i].cpu()),
                            'class_id': int(boxes.cls[i].cpu()),
                            'class_name': result.names[int(boxes.cls[i].cpu())],
                            'mask': masks.data[i].cpu().numpy()
                        }
                        segmentations.append(segmentation)

            return segmentations
        except Exception as e:
            logger.error(f"Segmentation failed for {image_path}: {e}")
            return []

    def generate_labels(self,
                       image_path: Union[str, Path],
                       output_format: str = "yolo",
                       conf_threshold: float = 0.25) -> str:
        """
        Generate label annotations for an image.

        Args:
            image_path: Path to image
            output_format: Label format ('yolo', 'coco', 'pascal_voc')
            conf_threshold: Confidence threshold

        Returns:
            Label string in specified format
        """
        detections = self.detect(image_path, conf_threshold=conf_threshold)

        if output_format == "yolo":
            return self._to_yolo_format(detections, image_path)
        elif output_format == "coco":
            return self._to_coco_format(detections, image_path)
        elif output_format == "pascal_voc":
            return self._to_pascal_voc_format(detections, image_path)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def _to_yolo_format(self, detections: List[Dict[str, Any]], image_path: Union[str, Path]) -> str:
        """Convert detections to YOLO format"""
        try:
            from PIL import Image
            img = Image.open(image_path)
            img_width, img_height = img.size
        except:
            logger.warning("Could not get image size, using detections as-is")
            img_width, img_height = 640, 640

        lines = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            # Convert to YOLO format: class_id x_center y_center width height (normalized)
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            lines.append(f"{det['class_id']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        return '\n'.join(lines)

    def _to_coco_format(self, detections: List[Dict[str, Any]], image_path: Union[str, Path]) -> str:
        """Convert detections to COCO JSON format"""
        import json

        try:
            from PIL import Image
            img = Image.open(image_path)
            img_width, img_height = img.size
        except:
            img_width, img_height = 640, 640

        annotations = []
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            annotations.append({
                'id': i,
                'image_id': 0,
                'category_id': det['class_id'],
                'bbox': [x1, y1, x2 - x1, y2 - y1],
                'area': (x2 - x1) * (y2 - y1),
                'iscrowd': 0,
                'score': det['confidence']
            })

        coco_data = {
            'images': [{
                'id': 0,
                'file_name': str(Path(image_path).name),
                'width': img_width,
                'height': img_height
            }],
            'annotations': annotations
        }

        return json.dumps(coco_data, indent=2)

    def _to_pascal_voc_format(self, detections: List[Dict[str, Any]], image_path: Union[str, Path]) -> str:
        """Convert detections to Pascal VOC XML format"""
        try:
            from PIL import Image
            img = Image.open(image_path)
            img_width, img_height = img.size
        except:
            img_width, img_height = 640, 640

        xml_lines = [
            '<annotation>',
            f'  <filename>{Path(image_path).name}</filename>',
            f'  <size>',
            f'    <width>{img_width}</width>',
            f'    <height>{img_height}</height>',
            f'    <depth>3</depth>',
            f'  </size>',
        ]

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            xml_lines.extend([
                '  <object>',
                f'    <name>{det["class_name"]}</name>',
                f'    <bndbox>',
                f'      <xmin>{int(x1)}</xmin>',
                f'      <ymin>{int(y1)}</ymin>',
                f'      <xmax>{int(x2)}</xmax>',
                f'      <ymax>{int(y2)}</ymax>',
                f'    </bndbox>',
                f'    <difficult>0</difficult>',
                '  </object>',
            ])

        xml_lines.append('</annotation>')
        return '\n'.join(xml_lines)

    def get_class_names(self) -> List[str]:
        """Get list of class names the model can detect"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return list(self.model.names.values())

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        return {
            'model_name': self.model_name,
            'device': str(self.device) if self.device else 'auto',
            'num_classes': len(self.model.names),
            'class_names': self.get_class_names(),
            'task': getattr(self.model, 'task', 'detect')
        }


def create_detector(model_name: str = "yolov8n.pt", device: Optional[str] = None) -> YOLOv8Detector:
    """
    Factory function to create a YOLOv8 detector.

    Args:
        model_name: Model variant
        device: Device to use

    Returns:
        YOLOv8Detector instance
    """
    return YOLOv8Detector(model_name=model_name, device=device)
