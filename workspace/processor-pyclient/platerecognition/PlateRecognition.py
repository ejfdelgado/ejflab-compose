from ultralytics import YOLO
from typing import Any, Dict
import cv2
import numpy as np

from platerecognition.sort import *
from platerecognition.utils import Util
# from util import get_car, read_license_plate
from PIL import Image
import os

def detect_plates_() -> Dict[str, Any]:
    print(f"--detect_plates ")
    mot_tracker = Sort()
    vehicles = [2, 3, 5, 7]
    results = {}
    frame_nmr = 1
    # load models
    coco_model = YOLO('./models/yolo/yolov8n.pt')
    license_plate_detector = YOLO('./models/license_plate_detector.pt')

    frame = cv2.imread("test_alpr_01.jpg", cv2.IMREAD_COLOR)

    results[frame_nmr] = {}

    # detect vehicles
    detections = coco_model(frame)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # track vehicles
    track_ids = mot_tracker.update(np.asarray(detections_))
    # print(f"--track_ids {track_ids}")
    # detect license plates
    license_plates = license_plate_detector(frame)[0]
    # print(f"--license_plates {license_plates}")
    i = 0
    for license_plate in license_plates.boxes.data.tolist():
        # print(f"--license_plate {license_plate}")
        i += 1
        x1, y1, x2, y2, score, class_id = license_plate
        # print(f"--x1 {x1} x2 {x2} y1 {y1} y2 {y2} score {score} class_id {class_id}")

        # assign license plate to car
        xcar1, ycar1, xcar2, ycar2, car_id = Util.get_car(license_plate, track_ids)
        # print(f"--car_id {car_id}")
        if car_id != -1:

            # crop license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
            # print(f"--license_plate_crop {license_plate_crop}")

            # Crear el nombre del archivo
            output_path = os.path.join("extract", f"plate_{i}.png")
            # Guardar la imagen
            cv2.imwrite(output_path, license_plate_crop)

            # process license plate
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

            output_path_thresh = os.path.join("extract", f"plate_thresh_{i}.png")
            # Guardar la imagen
            cv2.imwrite(output_path_thresh, license_plate_crop_thresh)
            # read license plate number
            # license_plate_text, license_plate_text_score = Util.read_license_plate(license_plate_crop_thresh)
            license_plate_text, license_plate_text_score = Util.read_license_plate(license_plate_crop)
            print(f"--license_plate_text {license_plate_text}")
            if license_plate_text is not None:
                results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                'text': license_plate_text,
                                                                'bbox_score': score,
                                                                'text_score': license_plate_text_score}}
                
    print(f"--results {results}")

def detect_plates(frame) -> Dict[str, Any]:
    print(f"--detect_plates ")
    mot_tracker = Sort()
    vehicles = [2, 3, 5, 7]
    results = {}
    frame_nmr = 1
    # load models
    coco_model = YOLO('./models/yolo/yolov8n.pt')
    license_plate_detector = YOLO('./models/license_plate_detector.pt')

    results[frame_nmr] = {}

    # detect vehicles
    detections = coco_model(frame)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # track vehicles
    track_ids = mot_tracker.update(np.asarray(detections_))
    # print(f"--track_ids {track_ids}")
    # detect license plates
    license_plates = license_plate_detector(frame)[0]
    # print(f"--license_plates {license_plates}")
    i = 0
    for license_plate in license_plates.boxes.data.tolist():
        # print(f"--license_plate {license_plate}")
        i += 1
        x1, y1, x2, y2, score, class_id = license_plate
        # print(f"--x1 {x1} x2 {x2} y1 {y1} y2 {y2} score {score} class_id {class_id}")

        # assign license plate to car
        xcar1, ycar1, xcar2, ycar2, car_id = Util.get_car(license_plate, track_ids)
        # print(f"--car_id {car_id}")
        if car_id != -1:

            # crop license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
            # print(f"--license_plate_crop {license_plate_crop}")

            # Crear el nombre del archivo
            output_path = os.path.join("extract", f"plate_{i}.png")
            # Guardar la imagen
            cv2.imwrite(output_path, license_plate_crop)

            # process license plate
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

            output_path_thresh = os.path.join("extract", f"plate_thresh_{i}.png")
            # Guardar la imagen
            cv2.imwrite(output_path_thresh, license_plate_crop_thresh)
            # read license plate number
            # license_plate_text, license_plate_text_score = Util.read_license_plate(license_plate_crop_thresh)
            license_plate_text, license_plate_text_score = Util.read_license_plate(license_plate_crop)
            print(f"--license_plate_text {license_plate_text}")
            if license_plate_text is not None:
                results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                'text': license_plate_text,
                                                                'bbox_score': score,
                                                                'text_score': license_plate_text_score}}
                
    print(f"--results {results}")