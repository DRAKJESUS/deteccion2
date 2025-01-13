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

# Configuración del archivo JSON de credenciales
CREDENTIALS_FILE = 'secret.json'
SCOPES = ['https://www.googleapis.com/auth/drive.file']
FOLDER_ID = '1ZNUgneU919vdI1Lg__o4XWGWQx3xVfhz'  # Cambiar por tu carpeta de Google Drive

# Inicializar el servicio de Google Drive
def obtener_servicio_drive():
    try:
        creds = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=SCOPES)
        service = build('drive', 'v3', credentials=creds)
        return service
    except Exception as e:
        raise Exception(f"Error al cargar las credenciales: {e}")

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

    if image_np is None:
        return jsonify({'error': 'Error al cargar la imagen'})

    imagen_con_puntos = Image.fromarray(image_np)
    imagen_brillo = ImageEnhance.Brightness(imagen_con_puntos).enhance(random.uniform(1.5, 2))
    imagen_girada_horizontal = imagen_con_puntos.transpose(Image.FLIP_LEFT_RIGHT)
    imagen_girada_vertical = imagen_con_puntos.transpose(Image.FLIP_TOP_BOTTOM)

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
                        size = 8
                        color = (255, 0, 0)
                        thickness = 4
                        draw_puntos = ImageDraw.Draw(imagen_con_puntos)
                        draw_puntos.line((x - size, y - size, x + size, y + size), fill=color, width=thickness)
                        draw_puntos.line((x - size, y + size, x + size, y - size), fill=color, width=thickness)

    TRADUCCION_EMOCIONES = {
        "angry": "enojado",
        "disgust": "disgustado",
        "fear": "miedo",
        "happy": "feliz",
        "sad": "triste",
        "surprise": "sorprendido",
        "neutral": "neutral"
    }

    try:
        resultado_emocion = DeepFace.analyze(img_path=image_np, actions=['emotion'], enforce_detection=False)
        emocion_principal_en = resultado_emocion[0]['dominant_emotion']
        emocion_principal = TRADUCCION_EMOCIONES.get(emocion_principal_en, emocion_principal_en)
    except Exception as e:
        emocion_principal = f"Error detectando emociones: {str(e)}"

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
