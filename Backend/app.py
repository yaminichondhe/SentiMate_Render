import os
import tempfile
import random
import numpy as np
import cv2
import librosa

from flask import Flask, request, jsonify
from flask_cors import CORS

from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing.image import img_to_array

# Firebase
import firebase_admin
from firebase_admin import credentials, firestore

# ---------------- APP ----------------
app = Flask(__name__)
CORS(app)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_DIR, 'models')

FER_LABELS = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
SER_LABELS = ('Angry', 'Calm', 'Disgust', 'Fear', 'Happy', 'Sad', 'Neutral', 'Surprise')

COMMON_EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ---------------- LOAD MODELS ----------------
fer_model = load_model(os.path.join(MODELS_DIR, 'fer_model.h5'), compile=False)

with open(os.path.join(MODELS_DIR, 'ser_model.json'), 'r') as f:
    ser_model = model_from_json(f.read())

ser_model.load_weights(os.path.join(MODELS_DIR, 'ser_model.weights.h5'))

print("✅ Models loaded")

# ---------------- FIREBASE ----------------
cred = credentials.Certificate(os.path.join(APP_DIR, "firebase-key.json"))
firebase_admin.initialize_app(cred)
db = firestore.client()

print("🔥 Firebase connected")

# ---------------- RESPONSE BANK ----------------
RESPONSES = {
    "Happy": ["You look really happy today! 😊","That smile suits you!","You're radiating positive energy!","It's great to see you this cheerful!","Happiness looks good on you!","You seem in a wonderful mood!","Keep smiling like that!","You look joyful and relaxed!","Such a bright expression!","Looks like something made your day!","You're glowing with happiness!","This is a great moment, enjoy it!","Your positivity is contagious!","Stay this happy always!","You look amazing and cheerful!"],

    "Sad": ["You seem a bit down. I'm here for you.","It's okay to feel sad sometimes.","Do you want to talk about it?","Things will get better.","I'm here if you need support.","Take your time.","You’re not alone.","Sending positive vibes.","It's okay to feel this way.","Take a deep breath.","Hope things improve soon.","You deserve happiness.","Take it easy today.","Everything will be alright.","You’re strong."],

    "Angry": ["I sense anger. Take a breath.","Try to stay calm.","Pause for a moment.","Take a step back.","You might need a break.","Let’s calm down.","It's okay to feel angry.","Release tension slowly.","Breathe deeply.","Stay calm.","Reset yourself.","Relax your mind.","Anger will pass.","You’ve got this.","Shift your focus."],

    "Fear": ["You seem worried.","Everything will be okay.","Take a deep breath.","You're safe.","Calm your thoughts.","It will work out.","You’re stronger than fear.","One step at a time.","You’ve got control.","Relax.","Stay calm.","You can overcome this.","Trust yourself.","You are safe.","Breathe slowly."],

    "Surprise": ["That looks surprising!","Unexpected reaction!","Something caught you off guard!","Hope it's good!","You look amazed!","That was sudden!","You seem startled!","Interesting reaction!","Wow moment!","You look shocked!","That’s unexpected!","Hope it’s pleasant!","You seem surprised!","That’s something!","Quite a reaction!"],

    "Neutral": ["You look calm.","Everything seems normal.","You appear composed.","You look steady.","Balanced mood.","You seem peaceful.","All looks good.","Calm state.","No strong emotion.","Relaxed.","Stable mood.","You look fine.","You seem okay.","All good.","Everything is steady."],

    "Disgust": ["Something seems unpleasant.","You look uncomfortable.","That doesn't seem nice.","Something bothered you.","You seem uneasy.","That looks unpleasant.","Hope it improves.","That reaction says a lot.","Something feels off.","You didn’t like that.","Stay positive.","Hope it passes.","You seem disturbed.","Not a good feeling.","Try to move on."]
}

# ---------------- RESPONSE ----------------
def generate_response(emotion):
    return random.choice(RESPONSES.get(emotion, ["Emotion detected."]))

# ---------------- UTILS ----------------
def normalize(d):
    total = sum(d.values()) + 1e-6
    return {k: v / total for k, v in d.items()}

def soften(d, temp=1.5):
    return {k: v ** (1 / temp) for k, v in d.items()}

# ---------------- AUDIO ----------------
def extract_mfcc(audio_path, n_mfcc=40, max_pad_len=200):
    audio, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    if mfcc.shape[1] > max_pad_len:
        mfcc = mfcc[:, :max_pad_len]
    else:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_pad_len - mfcc.shape[1])), mode='constant')

    return mfcc.T

# ---------------- API ----------------
@app.route('/api/analyze', methods=['POST'])
def analyze():
    temp_audio = None

    try:
        image = request.files.get("image")
        audio = request.files.get("audio")

        if not image or not audio:
            return jsonify({"error": "Image and audio required"}), 400

        # ---------- FER ----------
        img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (96, 96))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        fer_preds = fer_model.predict(img, verbose=0)[0]
        fer_dict = dict(zip(FER_LABELS, fer_preds))

        # ---------- SER ----------
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            audio.save(tmp.name)
            temp_audio = tmp.name

        features = extract_mfcc(temp_audio)
        features = features[np.newaxis, :, :]

        ser_preds = ser_model.predict(features, verbose=0)[0]
        ser_dict = dict(zip(SER_LABELS, ser_preds))

        # ---------- FIX BIAS ----------
        fer_dict = normalize(soften(fer_dict))
        ser_dict = normalize(soften(ser_dict))

        # ---------- DYNAMIC WEIGHT ----------
        conf_fer = np.max(fer_preds)
        conf_ser = np.max(ser_preds)

        w_fer = conf_fer / (conf_fer + conf_ser + 1e-6)
        w_ser = 1 - w_fer

        # ---------- COMBINE ----------
        final_scores = {}
        for emotion in COMMON_EMOTIONS:
            final_scores[emotion] = (
                w_fer * fer_dict.get(emotion, 0) +
                w_ser * ser_dict.get(emotion, 0)
            )

        # ---------- DIVERSITY BOOST ----------
        sorted_emotions = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top2 = sorted_emotions[:2]

        final_emotion = random.choice([top2[0][0], top2[1][0]])

        print("FER:", fer_dict)
        print("SER:", ser_dict)
        print("FINAL:", final_scores)
        print("SELECTED:", final_emotion)

        response_text = generate_response(final_emotion)

        # ---------- SAVE ----------
        db.collection("emotions").add({
            "emotion": final_emotion,
            "response": response_text,
            "timestamp": firestore.SERVER_TIMESTAMP
        })

        return jsonify({
            "final_emotion": final_emotion,
            "response": response_text,
            "scores": final_scores
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if temp_audio and os.path.exists(temp_audio):
            os.remove(temp_audio)

# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
