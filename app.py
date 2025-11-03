"""
Flask Web Application for Face Emotion Detection
Optimized and stable version — handles large models safely.
"""

import os
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import cv2
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ======================================================
# CONFIGURATION
# ======================================================
app = Flask(__name__)
app.secret_key = 'your_secret_key_here_change_in_production'

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Responses
EMOTION_RESPONSES = {
    'Angry': "You look angry! 😠 Take a deep breath and relax.",
    'Disgust': "You seem disgusted! 😖 Is everything okay?",
    'Fear': "You look scared! 😨 Don't worry, everything will be fine.",
    'Happy': "You're smiling! 😊 Keep spreading that positive energy!",
    'Neutral': "You look calm and neutral. 😐 Perfectly balanced!",
    'Sad': "You seem sad. 😢 Remember, tough times don't last, tough people do!",
    'Surprise': "You look surprised! 😲 Hope it's a pleasant surprise!"
}

# ======================================================
# DATABASE INITIALIZATION
# ======================================================
def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            matric_number TEXT NOT NULL,
            email TEXT NOT NULL,
            image_path TEXT NOT NULL,
            detected_emotion TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print("[SUCCESS] Database initialized successfully!")

init_db()

# ======================================================
# LOAD MODEL SAFELY (Prevent Memory Overload)
# ======================================================
print("[INFO] Loading emotion detection model...")
try:
    # Use a TensorFlow graph to make prediction thread-safe
    model = load_model('face_emotionModel.h5')
    model.make_predict_function()
    graph = tf.compat.v1.get_default_graph()
    print("[SUCCESS] Model loaded successfully!")
except Exception as e:
    print(f"[ERROR] Error loading model: {e}")
    model = None

# ======================================================
# HELPER FUNCTIONS
# ======================================================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_emotion(img_path):
    if model is None:
        return "Error", 0.0
    try:
        img = image.load_img(img_path, color_mode='grayscale', target_size=(48, 48))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        with graph.as_default():
            predictions = model.predict(img_array, verbose=0)
            emotion_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][emotion_idx]) * 100
            emotion_label = EMOTIONS[emotion_idx]
            return emotion_label, confidence

    except Exception as e:
        print(f"[ERROR] Emotion detection failed: {e}")
        return "Error", 0.0


def save_to_database(name, matric, email, img_path, emotion, confidence):
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO submissions (name, matric_number, email, image_path, detected_emotion, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (name, matric, email, img_path, emotion, confidence))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[ERROR] Database save failed: {e}")
        return False

# ======================================================
# ROUTES
# ======================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit():
    try:
        name = request.form.get('name', '').strip()
        matric = request.form.get('matric_number', '').strip()
        email = request.form.get('email', '').strip()

        if not all([name, matric, email]):
            flash('Please fill in all fields!', 'error')
            return redirect(url_for('index'))

        if 'photo' not in request.files:
            flash('No photo uploaded!', 'error')
            return redirect(url_for('index'))

        file = request.files['photo']
        if file.filename == '':
            flash('No photo selected!', 'error')
            return redirect(url_for('index'))

        if not allowed_file(file.filename):
            flash('Invalid file type! Please upload PNG, JPG, or JPEG images.', 'error')
            return redirect(url_for('index'))

        # Save file
        filename = secure_filename(f"{matric}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Detect emotion
        emotion, confidence = detect_emotion(filepath)

        if emotion == "Error":
            flash('Error processing image. Please try again.', 'error')
            return redirect(url_for('index'))

        # Save to DB
        if save_to_database(name, matric, email, filepath, emotion, confidence):
            response_msg = EMOTION_RESPONSES.get(emotion, "Emotion detected!")
            flash(f'✅ Submission successful!', 'success')
            flash(f'🎭 Detected Emotion: {emotion} ({confidence:.1f}% confident)', 'info')
            flash(f'💬 {response_msg}', 'message')
            return redirect(url_for('index'))
        else:
            flash('Error saving to database. Please try again.', 'error')
            return redirect(url_for('index'))

    except Exception as e:
        print(f"[ERROR] Submit route failed: {e}")
        flash('An unexpected error occurred. Please try again.', 'error')
        return redirect(url_for('index'))


@app.route('/submissions')
def view_submissions():
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM submissions ORDER BY timestamp DESC LIMIT 50')
        rows = cursor.fetchall()
        conn.close()

        submissions = [{
            'id': r[0],
            'name': r[1],
            'matric': r[2],
            'email': r[3],
            'image_path': r[4],
            'emotion': r[5],
            'confidence': f"{r[6]:.1f}%",
            'timestamp': r[7]
        } for r in rows]

        return render_template('submissions.html', submissions=submissions)
    except Exception as e:
        return f"Error loading submissions: {e}"

# ======================================================
# ERROR HANDLERS
# ======================================================
@app.errorhandler(413)
def file_too_large(e):
    flash('File too large! Max 5MB.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(404)
def not_found(e):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(e):
    flash('An internal error occurred. Please try again.', 'error')
    return redirect(url_for('index'))

# ======================================================
# RUN APP
# ======================================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🌍 FACE EMOTION DETECTION WEB APP")
    print("=" * 60)
    print("🚀 Server running at: http://127.0.0.1:5000")
    print("⏹️  Press CTRL+C to stop\n")

    # Use threaded=True to prevent worker blocking
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
