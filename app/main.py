from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
from PIL import Image
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# FastAPI uygulaması
app = FastAPI()

# CORS middleware ekleme
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tüm kaynaklara izin vermek için
    allow_credentials=True,
    allow_methods=["*"],  # Tüm HTTP metotlarına izin vermek için
    allow_headers=["*"],  # Tüm başlıklara izin vermek için
)

# TensorFlow Lite modelini yükleme
MODEL_PATH = "model/model.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Giriş ve çıkış tensörlerini alma
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Sınıf isimleri
class_labels = [
    "type_dog", "type_cat", "type_human", "gender_female",
    "gender_male", "hair_short", "hair_long", "hair_light", "hair_dark"
]

# Fotoğraf tahmini için endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Yüklenen fotoğrafı açma
        image = Image.open(file.file).convert("RGB")
        
        # Resmi yeniden boyutlandırma (224x224) ve normalize etme
        image = image.resize((224, 224))
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0).astype(np.float32)
        
        # Modeli çalıştırma
        interpreter.set_tensor(input_details[0]['index'], image_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Tahmin sonuçlarını sınıf isimleriyle eşleştirme
        results = {class_labels[i]: float(predictions[i]) for i in range(len(class_labels))}
        
        return JSONResponse(content={"predictions": results})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
