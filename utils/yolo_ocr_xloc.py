import cv2
import numpy as np
from ultralytics import YOLO
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

def get_yolo_ocr_xloc(
    img_path,
    yolo_model,
    ocr_pipeline,
    num_cropped_boxes=5,
    expand_px=1,
    expand_px_for_first_last_cha=12,
    yolo_imgsz=640,
    yolo_iou=0.1,
    yolo_conf=0.07
):
    """
    Detect character bounding boxes and recognize characters in an image using YOLO and OCR.

    Parameters:
        img_path (str): Path to the input image file.
        yolo_model (YOLO): Instantiated YOLO model for character detection.
        ocr_pipeline (Pipeline): Instantiated ModelScope OCR pipeline for character recognition.
        num_cropped_boxes (int): Number of adjacent boxes to crop for each OCR segment (default: 5).
        expand_px (int): Number of pixels to expand each side of the box for non-edge characters (default: 1).
        expand_px_for_first_last_cha (int): Number of pixels to expand for the first/last character (default: 12).
        yolo_imgsz (int): Image size for YOLO inference (default: 640).
        yolo_iou (float): IOU threshold for YOLO detection (default: 0.1).
        yolo_conf (float): Confidence threshold for YOLO detection (default: 0.07).

    Returns:
        boxes (list of list): List of detected bounding boxes [x1, y1, x2, y2], sorted left-to-right.
        recognized_chars (list of str): List of recognized characters, one per box.
        char_x_centers (list of int): List of x-axis center positions for each character.
    """
    img = cv2.imread(img_path)
    results = yolo_model([img_path], imgsz=yolo_imgsz, iou=yolo_iou, conf=yolo_conf)
    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    boxes = sorted(boxes, key=lambda box: box[0])
    recognized_chars = []
    char_x_centers = []
    n_boxes = len(boxes)
    for j, box in enumerate(boxes):
        if n_boxes <= num_cropped_boxes:
            idxs = list(range(n_boxes))
        else:
            half = num_cropped_boxes // 2
            start = max(0, min(j - half, n_boxes - num_cropped_boxes))
            end = start + num_cropped_boxes
            idxs = list(range(start, end))
        boxes_to_crop = [boxes[idx] for idx in idxs]
        contains_last_char = (n_boxes - 1) in idxs
        if j == 0:
            left_expand = expand_px_for_first_last_cha
        else:
            left_expand = expand_px
        if contains_last_char:
            right_expand = expand_px_for_first_last_cha
        else:
            right_expand = expand_px
        crop_x1 = min(b[0] for b in boxes_to_crop)
        crop_x2 = max(b[2] for b in boxes_to_crop)
        crop_y1 = 0
        crop_y2 = img.shape[0]
        if j == 0:
            crop_x1 = max(crop_x1 - left_expand, 0)
        if contains_last_char:
            crop_x2 = min(crop_x2 + right_expand, img.shape[1])
        segment_img = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        mask = np.zeros(segment_img.shape[:2], dtype=np.uint8)
        for b in boxes_to_crop:
            bx1 = max(b[0] - crop_x1 - expand_px, 0)
            bx2 = min(b[2] - crop_x1 + expand_px, crop_x2 - crop_x1)
            by1 = 0
            by2 = img.shape[0]
            mask[by1:by2, bx1:bx2] = 255
        non_text_mask = cv2.bitwise_not(mask)
        if np.count_nonzero(non_text_mask) > 0:
            mean_color = cv2.mean(segment_img, mask=non_text_mask)[:3]
            mean_color = np.array(mean_color, dtype=np.uint8)
        else:
            mean_color = np.array([255, 255, 255], dtype=np.uint8)
        mean_img = np.full(segment_img.shape, mean_color, dtype=np.uint8)
        blurred_mask = cv2.GaussianBlur(mask, (15, 15), 0)
        alpha = blurred_mask.astype(np.float32) / 255.0
        alpha = np.expand_dims(alpha, axis=2)
        segment_img_masked = (segment_img * alpha + mean_img * (1 - alpha)).astype(np.uint8)
        ocr_result = ocr_pipeline(segment_img_masked)
        segment_text = ocr_result['text'][0] if 'text' in ocr_result else ''
        segment_text = segment_text.replace(' ', '')
        if len(segment_text) == num_cropped_boxes:
            char = segment_text[j - idxs[0]]
        elif len(segment_text) > 0:
            char = segment_text[min(j - idxs[0], len(segment_text)-1)]
        else:
            char = ''
        recognized_chars.append(char)
        x1, _, x2, _ = box
        x_center = (x1 + x2) // 2
        char_x_centers.append(x_center)
    return img[:,:,::-1], boxes, recognized_chars, char_x_centers 