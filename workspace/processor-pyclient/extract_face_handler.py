from retinaface import RetinaFace
from deepface import DeepFace
import os
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
from base_procesor import register_face_found
from opencv_utils import crop_with_gap

class ExtractFaceHandler:

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
        print(f"--ExtractFaceHandler process {frame} {video_id} {timeline}")
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        detected_faces = RetinaFace.detect_faces(frame)
        frame_ = cv2.imread(frame, cv2.IMREAD_COLOR)
        faces_array = []
        face_bytes = []
        for face_id, face_data in detected_faces.items():
            print(f"Contenido de {face_id}:")
            # print(face_data)
            score = face_data["score"]
            print(f"face score {score}")
            if score < default_arguments["min_face_score"]:
                continue
            facial_area = face_data["facial_area"]
            roi_height = facial_area[3] - facial_area[1]
            roi_width = facial_area[2] - facial_area[0]
            roi_x0 = facial_area[0]
            roi_y0 = facial_area[1]
            face_crop = crop_with_gap(frame_, roi_x0, roi_y0, roi_width, roi_height, default_arguments['face_gap'])
            face_crop_bytes = cv2.imencode('.png', face_crop)[1].tobytes()
            # Guardar la imagen
            face_path = f"{img_object_path_template}/{face_id}.png" 
            face_bytes.append({
                'path': face_path,
                'bytes': face_crop_bytes
            })
            # output_path = os.path.join("extract", f"{video_id}_{timestamp}_{face_id}.png")
            # cv2.imwrite(output_path, face_crop)
            face_obj = {
                "name": f"{video_id}_{timestamp}_{face_id}",
                "path": img_path
            }
            faces_array.append(face_obj)
            # image = cv2.rectangle(frame_, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), (255, 255, 255), 1)
            # cv2.imwrite(output_path, image)
            embedding_objs = DeepFace.represent(
                img_path = face_crop
                , model_name = "ArcFace"
                , detector_backend = "skip"
                , enforce_detection = False)
            
            embedding_vector = embedding_objs[0]["embedding"]
            sql_insert_response = await register_face_found(db_data, {
                'mediaStartTime':timeline["t"]*1000 + media['startTime'],#millis
                'mediaEndTime':timeline["t"]*1000 + media['startTime'],#millis
                'mediaSourceUrl': '',
                'imageSourceUrl': img_path,
                'thumbnail': face_path,
                'frameWidth': image_width,
                'frameHeight': image_height,
                'imageBboxX1':facial_area[0]/image_width,
                'imageBboxY1':facial_area[1]/image_height,
                'imageBboxX2':facial_area[2]/image_width,
                'imageBboxY2':facial_area[3]/image_height,
                'imageBboxScore':face_data['score'],
            })
            # print(sql_insert_response)
            sql_inserted_id = ''
            if (sql_insert_response['inserted'] >= 0):
                sql_inserted_id = sql_insert_response['object']['id']
            face_item = {
                    'document_id': video_id,
                    'face_path': img_path,
                    'millis': int(timeline["t"]*1000 + media['startTime']),
                    'x1': facial_area[0]/image_width,
                    'y1': facial_area[1]/image_height,
                    'x2': facial_area[2]/image_width,
                    'y2': facial_area[3]/image_height,
                    'face_vector': embedding_vector,
                    'ref_id': sql_inserted_id
                } 
            #print(face_item)
            self.milvus_handler.insert(face_item, "faces")
 
        print(f"--ExtractFaceHandler process END ")
        return {
            'faces': faces_array,
            'face_bytes': face_bytes
        }
    
    def search(self, frame, db_data, paging, extra):
        # print(extra)
        detected_faces = RetinaFace.detect_faces(frame)
        frame_ = cv2.imread(frame, cv2.IMREAD_COLOR)
        faces_array = []
        for face_id, face_data in detected_faces.items():
            print(f"Contenido de {face_id}:")
            facial_area = face_data["facial_area"]
            face_crop = frame_[facial_area[1]:facial_area[3], facial_area[0]: facial_area[2], :]
            embedding_objs = DeepFace.represent(
                img_path = face_crop
                , model_name = "ArcFace"
                , detector_backend = "skip"
                , enforce_detection = False)
            
            embedding_vector = embedding_objs[0]["embedding"]

            face_results = self.milvus_handler.search_faces(embedding_vector, "faces", db_data, paging, extra)
            for one_face in face_results:
                faces_array.append(one_face)
        return faces_array
