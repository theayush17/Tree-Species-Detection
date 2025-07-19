import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
import cv2
import io
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-key-for-testing')

# Configuration
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = os.path.join('model', 'saved_model')
CLASS_NAMES_PATH = os.path.join('model', 'class_names.json')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and class names
def load_model_and_classes():
    try:
        # Load the model if it exists
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
        else:
            # Use a placeholder model for development
            model = None
            print("Warning: Model not found. Using placeholder.")
        
        # Load class names if they exist
        if os.path.exists(CLASS_NAMES_PATH):
            with open(CLASS_NAMES_PATH, 'r') as f:
                class_names = json.load(f)
        else:
            # Placeholder class names for development
            class_names = [
                "Oak", "Maple", "Pine", "Birch", "Cedar",
                "Spruce", "Willow", "Elm", "Fir", "Redwood"
            ]
            print("Warning: Class names file not found. Using placeholders.")
        
        return model, class_names
    except Exception as e:
        print(f"Error loading model or class names: {e}")
        return None, []

model, class_names = load_model_and_classes()

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess the image for model prediction"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize to [0,1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def get_prediction(image_path):
    """Get model prediction for an image"""
    if model is None:
        # Return mock predictions for development
        return {
            "species": "Oak",
            "confidence": 0.92,
            "info": "Oak trees are characterized by their distinctive lobed leaves and acorns. They are deciduous trees belonging to the genus Quercus and the family Fagaceae.",
            "top_predictions": [
                {"species": "Oak", "confidence": 0.92},
                {"species": "Maple", "confidence": 0.05},
                {"species": "Elm", "confidence": 0.02}
            ]
        }
    
    try:
        # Preprocess the image
        processed_img = preprocess_image(image_path)
        
        # Make prediction
        predictions = model.predict(processed_img)[0]
        
        # Get top 3 predictions
        top_indices = predictions.argsort()[-3:][::-1]
        top_predictions = [
            {"species": class_names[i], "confidence": float(predictions[i])}
            for i in top_indices
        ]
        
        # Get the top prediction
        top_prediction = top_predictions[0]
        
        # Mock information about the species (in a real app, this would come from a database)
        species_info = {
            "Oak": "Oak trees are characterized by their distinctive lobed leaves and acorns. They are deciduous trees belonging to the genus Quercus and the family Fagaceae.",
            "Maple": "Maple trees are known for their distinctive palmate leaves and winged fruits called samaras. They belong to the genus Acer and are popular for their vibrant fall colors.",
            "Pine": "Pine trees are evergreen conifers with needle-like leaves bundled in clusters. They produce cones and belong to the genus Pinus.",
            "Birch": "Birch trees are recognized by their distinctive bark that peels in thin, paper-like layers. They have simple, alternate leaves and belong to the genus Betula.",
            "Cedar": "Cedar trees are evergreen conifers with scale-like or needle-like leaves. They are known for their aromatic wood and belong to several different genera including Cedrus and Thuja.",
            "Spruce": "Spruce trees are evergreen conifers with needle-like leaves attached individually to small, peg-like structures. They belong to the genus Picea and have distinctive hanging cones.",
            "Willow": "Willow trees are known for their slender, flexible branches and narrow leaves. They often grow near water and belong to the genus Salix.",
            "Elm": "Elm trees have distinctive asymmetrical leaves with serrated edges. They belong to the genus Ulmus and typically have a vase-shaped growth form.",
            "Fir": "Fir trees are evergreen conifers with flat, needle-like leaves attached individually to the branches. They belong to the genus Abies and have upright cones.",
            "Redwood": "Redwood trees are massive evergreen conifers known for their great height and longevity. They include the coastal redwood (Sequoia sempervirens) and the giant sequoia (Sequoiadendron giganteum)."
        }
        
        # Get info for the predicted species
        species = top_prediction["species"]
        info = species_info.get(species, "No information available for this species.")
        
        return {
            "species": species,
            "confidence": top_prediction["confidence"],
            "info": info,
            "top_predictions": top_predictions
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": str(e)}

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Get prediction
        result = get_prediction(file_path)
        
        # Render result page
        return render_template('result.html', 
                              result=result, 
                              image_file=os.path.join('uploads', filename))
    else:
        flash('Allowed file types are png, jpg, jpeg')
        return redirect(request.url)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Get prediction
        result = get_prediction(file_path)
        
        return jsonify(result)
    else:
        return jsonify({"error": "Allowed file types are png, jpg, jpeg"}), 400

if __name__ == '__main__':
    app.run(debug=True)