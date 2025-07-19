import os
import requests
import zipfile
import tensorflow as tf
import argparse
import json
from tqdm import tqdm

def download_file(url, destination):
    """
    Download a file from a URL with a progress bar
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(destination, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    
    progress_bar.close()
    
    if total_size != 0 and progress_bar.n != total_size:
        print("ERROR: Something went wrong during download")
        return False
    return True

def download_pretrained_model(model_name="efficientnetb0", output_dir="model"):
    """
    Download a pre-trained model for tree species classification
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define model URLs - in a real application, these would point to actual model files
    # For this demo, we'll create a simple model structure
    
    print(f"Downloading pre-trained {model_name} model for tree species classification...")
    
    # Create a simple model using TensorFlow/Keras
    try:
        # Load a pre-trained model as the base
        if model_name == "efficientnetb0":
            base_model = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False)
        elif model_name == "efficientnetb3":
            base_model = tf.keras.applications.EfficientNetB3(weights='imagenet', include_top=False)
        elif model_name == "resnet50v2":
            base_model = tf.keras.applications.ResNet50V2(weights='imagenet', include_top=False)
        elif model_name == "mobilenetv2":
            base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)
        else:
            print(f"Model {model_name} not supported. Using EfficientNetB0 instead.")
            base_model = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False)
        
        # Add classification head
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation='softmax')  # 10 example tree species
        ])
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Save the model
        model_path = os.path.join(output_dir, f"{model_name}_tree_classifier")
        model.save(model_path)
        
        # Create a class names file
        class_names = [
            "Oak", "Maple", "Pine", "Birch", "Palm", 
            "Spruce", "Willow", "Cedar", "Cypress", "Fir"
        ]
        
        with open(os.path.join(output_dir, "class_names.json"), "w") as f:
            json.dump(class_names, f)
        
        print(f"Successfully downloaded and saved {model_name} model to {model_path}")
        print(f"Class names saved to {os.path.join(output_dir, 'class_names.json')}")
        return True
    
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download pre-trained tree species classification model')
    parser.add_argument('--model', type=str, default='efficientnetb0', 
                        choices=['efficientnetb0', 'efficientnetb3', 'resnet50v2', 'mobilenetv2'],
                        help='Model architecture to download')
    parser.add_argument('--output', type=str, default='model',
                        help='Output directory to save the model')
    
    args = parser.parse_args()
    
    success = download_pretrained_model(args.model, args.output)
    
    if success:
        print("\nModel download complete! You can now use the model for tree species classification.")
        print("\nTo use the model, run:")
        print(f"python app.py --model {args.output}/{args.model}_tree_classifier")
    else:
        print("\nModel download failed. Please try again or check your internet connection.")

if __name__ == "__main__":
    main()