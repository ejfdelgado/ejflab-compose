import asyncio
import sys
import json
import os
from base_procesor import BaseProcessor
from milvus_proxy import MilvusProxy
from milvus_client import MilvusHandler
#from milvus_client import MilvusHandler
from extract_face_handler import ExtractFaceHandler
from extract_plate_handler import ExtractPlateHandler
import numpy as np
import cv2
from PIL import Image

SUPER_CLASS = MilvusHandler
if 'MILVUS_PROXY' in os.environ and os.environ['MILVUS_PROXY'] == '1':
    SUPER_CLASS = MilvusProxy

class FrameProcessor(SUPER_CLASS):
    async def localConfigure(self):
        pass

    def get_default_arguments(self):
        return {
            'face_gap': 10,
            'car_gap': 10,
            'plate_gap': 10,
            'min_face_score': 0.95
        }

    async def process(self, args, default_arguments):
        method = args['method']
        named_inputs = args['namedInputs']
        inputs = []
        if 'inputs' in args:
            inputs = args['inputs']
            
        if (method == "indexFace"):
            print(f"indexFace...")
            the_bytes = named_inputs['bytes']
            timeline = named_inputs['timeline']
            media = named_inputs['media']
            print(f"--timeline {timeline}")
            room = args['room']
            print(f"--room {room}")
            db_data = args['dbData']
            # print(f"--db_data {db_data}")
            path = self.store_bytes_locally(the_bytes, "png", args, "_face")
            with Image.open(path) as img:
                image_width, image_height = img.size
            self.use_database("searchable", False)
            await self.wait_loaded("faces")
            extract_face_handler = ExtractFaceHandler(self)
            t = timeline["t"]
            offset = 0
            if len(inputs) > 0 and 'dperiod' in timeline:
                offset = inputs[0] * timeline['dperiod']
            t = t - offset
            # Alter the t
            timeline["t"] = t
            img_save_path = f"frames/{room}/frame_{round(t*1000)+media['startTime']}.png"
            img_object_path_template = f"frames/{room}/frame_{round(t*1000)+media['startTime']}"
            process_response = await extract_face_handler.process(
                path, 
                room, 
                timeline, 
                img_save_path, 
                image_width, 
                image_height, 
                db_data, 
                media, 
                img_object_path_template, 
                default_arguments
                )
            faces = process_response['faces']
            face_bytes = process_response['face_bytes']
            return {
                "output": {
                    "path": img_save_path, 
                    "have_face": bool(faces),
                },
                "face_bytes": face_bytes,
            }

        if (method == "indexPlate"):
            print(f"indexPlate...")
            the_bytes = named_inputs['bytes']
            timeline = named_inputs['timeline']
            media = named_inputs['media']
            print(f"--timeline {timeline}")
            room = args['room']
            print(f"--room {room}")
            db_data = args['dbData']
            print(f"--db_data {db_data}")
            path = self.store_bytes_locally(the_bytes, "png", args, "_plate")
            with Image.open(path) as img:
                image_width, image_height = img.size
            t = timeline["t"]
            offset = 0
            if len(inputs) > 0 and 'dperiod' in timeline:
                offset = inputs[0] * timeline['dperiod']
            t = t - offset
            # Alter the t
            timeline["t"] = t
            img_save_path = f"frames/{room}/frame_{round(t*1000)+media['startTime']}.png"
            frame = cv2.imread(path, cv2.IMREAD_COLOR)
            extract_plate_handler = ExtractPlateHandler(self)
            img_object_path_template = f"frames/{room}/frame_{round(t*1000)+media['startTime']}"
            process_response = await extract_plate_handler.process(
                frame, 
                room, 
                timeline, 
                img_save_path, 
                image_width, 
                image_height, 
                db_data, 
                media, 
                img_object_path_template, 
                default_arguments
                )
            plates = process_response['plates']
            plate_bytes = process_response['plate_bytes']
            car_bytes = process_response['car_bytes']
            return {
                "output":{
                    "path": img_save_path, 
                    "plates": plates, 
                    "have_plate": bool(plates),
                },
                "plate_bytes": plate_bytes, 
                "car_bytes": car_bytes, 
            }
        
        elif (method == "read"):
            print("read 2...")
            bytes = named_inputs['bytes']
            db_data = named_inputs['dbData']
            paging = named_inputs['paging']
            extra = named_inputs['extra']
            path = self.store_bytes_locally(bytes, "png", args)
            print(f"--path {path}")
            self.use_database("searchable", False)
            await self.wait_loaded("faces")
            extract_face_handler = ExtractFaceHandler(self)
            print("--db loaded")
            faces = extract_face_handler.search(path, db_data, paging, extra)
            print(f"--faces {faces}")
            return {"results": faces}


async def main():
    processor = FrameProcessor()
    await processor.configure(sys.argv)
    await processor.start_communication()

if __name__ == '__main__':
    asyncio.run(main())