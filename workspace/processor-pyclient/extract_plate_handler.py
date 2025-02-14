from ultralytics import YOLO
from typing import Any, Dict
import cv2
import numpy as np

from platerecognition.sort import *
from platerecognition.utils import Util
# from util import get_car, read_license_plate
from PIL import Image
import os
from datetime import datetime
from opencv_utils import crop_with_gap

class ExtractPlateHandler:

    def __init__ (self, milvus_handler):
        self.milvus_handler = milvus_handler


    async def process(
        self, 
        frame, 
        video_id, 
        timeline, 
        img_path, 
        image_width, 
        image_height, 
        db_data, 
        media,
        img_object_path_template,
        default_arguments
        ):
        print(f"--ExtractPlateHandler process {video_id} {timeline}")
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
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
        j = 0
        plates_array = []
        car_bytes = []
        plate_bytes = []
        for detection in detections.boxes.data.tolist():
            j += 1
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                print(f"--class_id {class_id}")
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
                
                plate_path = f"{img_object_path_template}/plate_{j}.png"
                car_path = f"{img_object_path_template}/car_{j}.png"
                
                roi_x0 = xcar1
                roi_y0 = ycar1
                roi_width = xcar2 - xcar1
                roi_height = ycar2 - ycar1
                car_crop = crop_with_gap(frame, roi_x0, roi_y0, roi_width, roi_height, default_arguments['car_gap'])
                car_crop_bytes = cv2.imencode('.png', car_crop)[1].tobytes()
                # car_output_path = os.path.join("extract", f"{timestamp}_car_{j}.png")
                # Guardar la imagen
                car_bytes.append({
                    'path': car_path,
                    'bytes': car_crop_bytes
                })
                # cv2.imwrite(car_output_path, car_crop)

                # crop license plate
                roi_x0 = x1
                roi_y0 = y1
                roi_width = x2 - x1
                roi_height = y2 - y1
                license_plate_crop = crop_with_gap(frame, roi_x0, roi_y0, roi_width, roi_height, default_arguments['plate_gap'])
                license_plate_crop_bytes = cv2.imencode('.png', license_plate_crop)[1].tobytes()
                # print(f"--license_plate_crop {license_plate_crop}")

                # Crear el nombre del archivo
                # output_path = os.path.join("extract", f"{timestamp}_plate_{i}.png")
                # Guardar la imagen
                # cv2.imwrite(output_path, license_plate_crop)
                plate_bytes.append({
                    'path': plate_path,
                    'bytes': license_plate_crop_bytes
                })

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # output_path_thresh = os.path.join("extract", f"plate_thresh_{i}.png")
                # Guardar la imagen
                # cv2.imwrite(output_path_thresh, license_plate_crop_thresh)
                # read license plate number
                # license_plate_text, license_plate_text_score = Util.read_license_plate(license_plate_crop_thresh)
                license_plate_text, license_plate_text_score = Util.read_license_plate(license_plate_crop)
                print(f"--license_plate_text {license_plate_text}")
                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {
                        'car': {
                            'bbox': [xcar1, ycar1, xcar2, ycar2]
                        },
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_text_score
                        }
                    }
                    
                    plates_array.append({
                        'vehicle': {
                            'bbox': {
                                "x1": xcar1/image_width, 
                                "y1": ycar1/image_height, 
                                "x2": xcar2/image_width, 
                                "y2": ycar2/image_height
                            },
                            'bbox_score': 1,
                            'thumbnail': car_path,
                        },
                        'license_plate': {
                            'bbox': {
                                "x1": x1/image_width, 
                                "y1": y1/image_height, 
                                "x2": x2/image_width, 
                                "y2": y2/image_height
                            },
                            'bbox_score': score,
                            'text_score': license_plate_text_score,
                            'thumbnail': plate_path,
                        },
                        'license_plate_text': license_plate_text, 
                        'license_plate_path': img_path, 
                        'document_id': video_id, 
                        'millis': timeline["t"]*1000 + media['startTime'],
                        'image_width': image_width, 
                        'image_height': image_height
                    })
        
            
        print(f"--ExtractPlateHandler process END ")
        return {
            'plates': plates_array,
            'plate_bytes': plate_bytes,
            'car_bytes': car_bytes
        }
