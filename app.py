"""
Flask Web Application for Face Emotion Detection
This app allows users to upload images and detects their facial emotions.
"""

import os
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ==================== CONFIGURATION ====================
app = Flask(__name__)
app.secret_key = 'your_secret_key_here_change_in_production'  # Change this in production!

# File upload settings
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Emotion labels (must match training order)
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Emotion responses (what to say for each emotion)
EMOTION_RESPONSES = {
    'Angry': "You look angry! 😠 Take a deep breath and relax.",
    'Disgust': "You seem disgusted! 😖 Is everything okay?",
    'Fear': "You look scared! 😨 Don't worry, everything will be fine.",
    'Happy': "You're smiling! 😊 Keep spreading that positive energy!",
    'Neutral': "You look calm and neutral. 😐 Perfectly balanced!",
    'Sad': "You seem sad. 😢 Remember, tough times don't last, tough people do!",
    'Surprise': "You look surprised! 😲 Hope it's a pleasant surprise!"
}

# ==================== DATABASE SETUP ====================
def init_db():
    """Initialize SQLite database and create table if not exists"""
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

# Initialize database on app start
init_db()

# ==================== LOAD MODEL ====================
print("[INFO] Loading emotion detection model...")
try:
    model = load_model('face_emotionModel.h5')
    print("[SUCCESS] Model loaded successfully!")
except Exception as e:
    print(f"[ERROR] Error loading model: {e}")
    print("[WARNING] Make sure you've trained the model first (run model_training.py)")
    model = None

# ==================== HELPER FUNCTIONS ====================
def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_emotion(img_path):
    """
    Detect emotion from an image file
    Returns: (emotion_label, confidence_percentage)
    """
    if model is None:
        return "Error", 0.0
    
    try:
        # Load and preprocess image
        img = image.load_img(img_path, color_mode='grayscale', target_size=(48, 48))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        emotion_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][emotion_idx]) * 100
        
        emotion_label = EMOTIONS[emotion_idx]
        
        return emotion_label, confidence
    
    except Exception as e:
        print(f"Error detecting emotion: {e}")
        return "Error", 0.0

def save_to_database(name, matric, email, img_path, emotion, confidence):
    """Save submission data to SQLite database"""
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
        print(f"Database error: {e}")
        return False

# ==================== ROUTES ====================
@app.route('/')
def index():
    """Home page - shows the upload form"""
    print("Templates folder:", app.template_folder)
    print("Available files:", os.listdir(app.template_folder))
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    """Handle form submission and emotion detection"""
    
    # Get form data
    name = request.form.get('name', '').strip()
    matric = request.form.get('matric_number', '').strip()
    email = request.form.get('email', '').strip()
    
    # Validate form fields
    if not all([name, matric, email]):
        flash('Please fill in all fields!', 'error')
        return redirect(url_for('index'))
    
    # Check if file was uploaded
    if 'photo' not in request.files:
        flash('No photo uploaded!', 'error')
        return redirect(url_for('index'))
    
    file = request.files['photo']
    
    # Check if file has a name
    if file.filename == '':
        flash('No photo selected!', 'error')
        return redirect(url_for('index'))
    
    # Validate file type
    if not allowed_file(file.filename):
        flash('Invalid file type! Please upload PNG, JPG, or JPEG images.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Save uploaded file
        filename = secure_filename(f"{matric}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Detect emotion
        emotion, confidence = detect_emotion(filepath)
        
        if emotion == "Error":
            flash('Error processing image. Please try again.', 'error')
            return redirect(url_for('index'))
        
        # Save to database
        if save_to_database(name, matric, email, filepath, emotion, confidence):
            # Get emotion response message
            response_msg = EMOTION_RESPONSES.get(emotion, "Emotion detected!")
            
            # Show success message with emotion and response
            flash(f'✅ Submission successful!', 'success')
            flash(f'🎭 Detected Emotion: {emotion} ({confidence:.1f}% confident)', 'info')
            flash(f'💬 {response_msg}', 'message')
            
            return redirect(url_for('index'))
        else:
            flash('Error saving to database. Please try again.', 'error')
            return redirect(url_for('index'))
    
    except Exception as e:
        print(f"Error in submit route: {e}")
        flash('An unexpected error occurred. Please try again.', 'error')
        return redirect(url_for('index'))

@app.route('/submissions')
def view_submissions():
    """View all submissions (optional - for testing/admin)"""
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM submissions ORDER BY timestamp DESC LIMIT 50')
        submissions = cursor.fetchall()
        conn.close()
        
        # Format data for display
        formatted_submissions = []
        for row in submissions:
            formatted_submissions.append({
                'id': row[0],
                'name': row[1],
                'matric': row[2],
                'email': row[3],
                'image_path': row[4],
                'emotion': row[5],
                'confidence': f"{row[6]:.1f}%",
                'timestamp': row[7]
            })
        
        return render_template('submissions.html', submissions=formatted_submissions)
    
    except Exception as e:
        return f"Error loading submissions: {e}"

# ==================== ERROR HANDLERS ====================
@app.errorhandler(413)
def file_too_large(e):
    """Handle file size too large error"""
    flash('File is too large! Maximum size is 5MB.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    flash('An internal error occurred. Please try again.', 'error')
    return redirect(url_for('index'))

# ==================== RUN APP ====================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌐 FACE EMOTION DETECTION WEB APP")
    print("="*60)
    print("\n🚀 Starting Flask server...")
    print("📱 Open your browser and go to: http://127.0.0.1:5000")
    print("⏹️  Press CTRL+C to stop the server\n")
    

    app.run(debug=True, host='0.0.0.0', port=5000)
