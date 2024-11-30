import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, send_file
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
from PIL import Image

matplotlib.use('Agg')  # Set the backend to non-interactive mode

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

# Temporary folder for file uploads
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to load pickle files
def load_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Function to generate a spectrogram image from the radar data
def generate_spectrogram(data):
    if np.iscomplexobj(data):
        data = np.abs(data)

    if data.ndim > 1:
        data = data.mean(axis=-1).flatten()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.specgram(data, Fs=1.0)
    ax.set_title('Spectrogram')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

# Function to preprocess the spectrogram image for model input
def preprocess_spectrogram(spectrogram_buf):
    img = Image.open(spectrogram_buf).convert("L")  # Convert to grayscale
    img = img.resize((224, 224))  # Resize to match model input size

    # Convert grayscale image to RGB by duplicating the single channel
    img = img.convert("RGB")  # Now it will have 3 channels

    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    return img_array

# Load the trained model
MODEL_PATH = r"serial_network_model_combined.keras"  # Update to your model path
model = tf.keras.models.load_model(MODEL_PATH)

@app.route('/classify', methods=['POST'])
def upload_file_classify():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(temp_file_path)

    try:
        # Load the data from the pickle file
        data = load_pkl_file(temp_file_path)

        if isinstance(data, np.ndarray):
            # Generate spectrogram
            spectrogram_buf = generate_spectrogram(data)

            # Preprocess the spectrogram for classification
            preprocessed_image = preprocess_spectrogram(spectrogram_buf)

            # Classify the spectrogram
            predictions = model.predict(preprocessed_image)
            class_idx = np.argmax(predictions, axis=1)[0]
            class_labels = ["Drone", "Bird"]  # Adjust based on your model's output
            result = class_labels[class_idx]

            return jsonify({"message": "Classification successful", "class": result}), 200
        else:
            return jsonify({"error": "Invalid file content"}), 400
    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)

@app.route('/image', methods=['POST'])
def upload_file_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(temp_file_path)

    try:
        # Load the data from the pickle file
        data = load_pkl_file(temp_file_path)

        if isinstance(data, np.ndarray):
            # Generate spectrogram
            spectrogram_buf = generate_spectrogram(data)

            # Send the generated spectrogram as an image
            return send_file(spectrogram_buf, mimetype='image/png')
        else:
            return jsonify({"error": "Invalid file content"}), 400
    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)

@app.route('/')
def welcome():
    return "Welcome to the Flask Application!"

@app.route('/sih')
def welcome_sih():
    return "Micro Doppler based Target Classification by Team BlackSquad!"


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
