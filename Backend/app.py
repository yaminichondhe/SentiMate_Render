import os
import tempfile
import numpy as np
import base64
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import librosa
from pydub import AudioSegment
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
from pydub import AudioSegment
AudioSegment.converter = r"C:\Users\lenovo\Downloads\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe"



app = Flask(__name__)
CORS(app)
API_BASE_URL = '/api'

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_DIR, 'models')

FER_LABELS = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
SER_LABELS = ('Angry', 'Calm', 'Disgust', 'Fear', 'Happy', 'Sad', 'Neutral', 'Surprise')

fer_model = None
ser_model = None

# -------- LOAD MODELS --------
def load_models():
    global fer_model, ser_model
    try:
        fer_model = load_model(os.path.join(MODELS_DIR, 'fer_model.h5'), compile=False)

        with open(os.path.join(MODELS_DIR, 'ser_model.json'), 'r') as f:
            ser_model = model_from_json(f.read())
        ser_model.load_weights(os.path.join(MODELS_DIR, 'ser_model.weights.h5'))

        print("✅ Models loaded successfully")

    except Exception as e:
        print("❌ Error loading models:", e)

# -------- IMAGE --------
@app.route(f'{API_BASE_URL}/fer_predict', methods=['POST'])
def fer_predict():
    try:
        data = request.json
        img_b64 = data['image'].split(',')[1]

        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        img = cv2.resize(img, (96, 96))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        preds = fer_model.predict(img)[0]
        idx = int(np.argmax(preds))

        return jsonify({
            "prediction": FER_LABELS[idx],
            "confidence": float(preds[idx])
        })

    except Exception as e:
        return jsonify({"error": f"FER error: {str(e)}"}), 500

# -------- CAMERA --------
@app.route(f'{API_BASE_URL}/fer_predict_frame', methods=['POST'])
def fer_predict_frame():
    try:
        file = request.files.get("frame")
        file_bytes = np.frombuffer(file.read(), np.uint8)

        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (96, 96))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        preds = fer_model.predict(img)[0]
        idx = int(np.argmax(preds))

        return jsonify({
            "prediction": FER_LABELS[idx],
            "confidence": float(preds[idx])
        })

    except Exception as e:
        return jsonify({"error": f"Frame error: {str(e)}"}), 500

# -------- AUDIO FEATURE --------
def extract_mfcc(audio_path, n_mfcc=40, max_pad_len=200):
    audio, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    if mfcc.shape[1] > max_pad_len:
        mfcc = mfcc[:, :max_pad_len]
    else:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')

    return mfcc.T

# -------- AUDIO (LIVE MIC + FILE FIXED) --------
@app.route(f'{API_BASE_URL}/ser_predict', methods=['POST'])
def ser_predict():
    temp_input = None
    temp_wav = None

    try:
        audio_file = request.files.get("audio")

        if not audio_file:
            return jsonify({"error": "No audio file provided"}), 400

        # Save raw file (webm or wav)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            audio_file.save(tmp.name)
            temp_input = tmp.name

        # 🔥 Convert to WAV
        temp_wav = temp_input.replace(".webm", ".wav")

        audio = AudioSegment.from_file(temp_input)
        audio = audio.set_frame_rate(16000).set_channels(1)

        audio.export(temp_wav, format="wav")

        # Extract features
        features = extract_mfcc(temp_wav)
        features = features[np.newaxis, :, :]

        preds = ser_model.predict(features)[0]
        idx = int(np.argmax(preds))

        return jsonify({
            "prediction": SER_LABELS[idx],
            "confidence": float(preds[idx])
        })

    except Exception as e:
        return jsonify({"error": f"SER error: {str(e)}"}), 500

    finally:
        if temp_input and os.path.exists(temp_input):
            os.remove(temp_input)
        if temp_wav and os.path.exists(temp_wav):
            os.remove(temp_wav)

# -------- MAIN --------
if __name__ == '__main__':
    load_models()
    app.run(host='0.0.0.0', port=5000, debug=True)
