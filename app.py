from flask import Flask, request, jsonify, render_template
import os
import json
import mediapipe as mp
import numpy as np
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageEnhance
import io
import base64
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from deepface import DeepFace
import random

app = Flask(__name__)
CORS(app)

# Configuración de credenciales y Google Drive
CREDENTIALS_FILE = 'secret.json'
SCOPES = ['https://www.googleapis.com/auth/drive.file']
FOLDER_ID = '1ZNUgneU919vdI1Lg__o4XWGWQx3xVfhz'

def obtener_servicio_drive():
    creds = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def detectar_puntos_y_procesar_imagenes():
    if 'file' not in request.files:
        return jsonify({'error': 'No se recibió correctamente la imagen'})

    archivo = request.files['file']
    if archivo.filename == '':
        return jsonify({'error': 'No se cargó ninguna imagen'})

    imagen_original = archivo.read()
    archivo.seek(0)
    image_np = np.array(Image.open(archivo).convert('RGB'))

    imagen_con_puntos = Image.fromarray(image_np)
    imagen_brillo = ImageEnhance.Brightness(imagen_con_puntos).enhance(random.uniform(1.5, 2))

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(image_np)
        puntos_deseados = [70, 55, 285, 300, 33, 468, 133, 362, 473, 263, 4, 185, 0, 306, 17]

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    if idx in puntos_deseados:
                        h, w, _ = image_np.shape
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        draw_puntos = ImageDraw.Draw(imagen_con_puntos)
                        draw_puntos.ellipse((x-4, y-4, x+4, y+4), fill=(255, 0, 0))

    try:
        resultado_emocion = DeepFace.analyze(img_path=image_np, actions=['emotion'], enforce_detection=False)
        emocion_principal = resultado_emocion[0]['dominant_emotion']
    except Exception:
        emocion_principal = "No detectado"

    def convertir_a_base64(imagen):
        buffered = io.BytesIO()
        imagen.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    img_data_puntos = convertir_a_base64(imagen_con_puntos)

    service = obtener_servicio_drive()
    archivo_drive = MediaIoBaseUpload(io.BytesIO(imagen_original), mimetype='image/png')
    archivo_metadata = {'name': archivo.filename, 'mimeType': 'image/png', 'parents': [FOLDER_ID]}
    archivo_drive_subido = service.files().create(body=archivo_metadata, media_body=archivo_drive).execute()

    return jsonify({
        'image_with_points_base64': img_data_puntos,
        'dominant_emotion': emocion_principal,
        'drive_id': archivo_drive_subido.get('id')
    })

if __name__ == '__main__':
    app.run(debug=True)
