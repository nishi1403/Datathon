from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import librosa  # For audio processing
import numpy as np
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Path to model and upload folder
MODEL_PATH = r'D:\My Project\Cyber Quest\frontend\public\deepfake_detection_model.keras'  # Ensure this is the correct path to your model
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the deepfake detection model
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {str(e)}")

# Ensure upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess the audio file into the required format for the model
def preprocess_audio(file_path):
    try:
        print(f"Attempting to load audio file from: {file_path}")
        
        # Load the audio file using librosa with a fixed sample rate (e.g., 16000)
        audio, sr = librosa.load(file_path, sr=16000)  # Use the same sample rate as your model training
        print(f"Audio loaded successfully. Sample rate: {sr}, Audio shape: {audio.shape}")

        # Check if audio is empty
        if audio.size == 0:
            raise ValueError("Loaded audio is empty")

        # Convert to mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(audio, sr=sr)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        print(f"Mel spectrogram created successfully. Shape: {mel_spectrogram.shape}")

        # Ensure the mel spectrogram matches the model input dimensions
        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)  # Add batch dimension
        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)  # Add channel dimension if required
        print(f"Preprocessed audio ready for model input. Final shape: {mel_spectrogram.shape}")
        
        return mel_spectrogram

    except Exception as e:
        print(f"Error during audio preprocessing: {str(e)}")
        return None

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            print(f"File saved at: {file_path}")
            if not os.path.exists(file_path):
                return jsonify({'error': 'File not saved properly'}), 500

            # Preprocess the audio file for the model
            preprocessed_audio = preprocess_audio(file_path)
            if preprocessed_audio is None:
                
                return jsonify({'error': 'Error processing the audio file'}), 500

            # Make the prediction using the loaded model
            prediction = model.predict(preprocessed_audio)
            print(f"Prediction result: {prediction}")

            # Assuming binary classification with threshold at 0.5
            result = 'fake' if prediction[0][0] > 0.5 else 'real'

            return jsonify({'result': result}), 200

        except Exception as e:
            print(f"Error during file processing or model inference: {str(e)}")
            return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

    else:
        return jsonify({'error': 'Unsupported file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
