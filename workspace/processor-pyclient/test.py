from retinaface import RetinaFace
from deepface import DeepFace
from platerecognition import PlateRecognition
import numpy as np
from PIL import Image
import os

faces = RetinaFace.extract_faces(img_path = "prueba_01.png", align = True)

output_dir = "extract"
os.makedirs(output_dir, exist_ok=True)

for i, face in enumerate(faces):
    face_np = np.array(face)
    
    # Convertir el array de NumPy a una imagen PIL
    face_image = Image.fromarray(face)
    # Crear el nombre del archivo
    output_path = os.path.join(output_dir, f"face_{i+1}.png")
    # Guardar la imagen
    face_image.save(output_path)
    
    embedding_objs = DeepFace.represent(
        img_path = face_np
        , model_name = "ArcFace"
        , detector_backend = "skip"
        , enforce_detection = False)
    
    print(f"--face ")

PlateRecognition.detect_plates()