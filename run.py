import os
import argparse
import subprocess
import sys
import webbrowser
import time

def check_dependencies():
    """
    Check if all required dependencies are installed
    """
    try:
        import flask
        import tensorflow as tf
        import numpy as np
        import PIL
        return True
    except ImportError as e:
        print(f"Missing dependency: {str(e)}")
        return False

def install_dependencies():
    """
    Install required dependencies from requirements.txt
    """
    print("Installing dependencies from requirements.txt...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {str(e)}")
        return False

def check_model(model_path):
    """
    Check if the model exists at the specified path
    """
    if model_path and os.path.exists(model_path):
        return True
    return False

def download_model(model_name):
    """
    Download a pre-trained model
    """
    print(f"Downloading pre-trained model: {model_name}...")
    try:
        subprocess.check_call([sys.executable, "download_model.py", "--model", model_name])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading model: {str(e)}")
        return False

def generate_sample_dataset(num_images=50):
    """
    Generate a sample dataset for training
    """
    print(f"Generating sample dataset with {num_images} images per class...")
    try:
        subprocess.check_call([sys.executable, "generate_sample_dataset.py", "--num-images", str(num_images)])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating sample dataset: {str(e)}")
        return False

def train_model(dataset_path, epochs=10, batch_size=32, model_name="efficientnetb0"):
    """
    Train a model on the specified dataset
    """
    print(f"Training model on dataset: {dataset_path}...")
    try:
        subprocess.check_call([
            sys.executable, "model/train.py", 
            "--dataset", dataset_path,
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--model", model_name
        ])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error training model: {str(e)}")
        return False

def run_app(model_path=None, host="127.0.0.1", port=5000, debug=False):
    """
    Run the Flask application
    """
    cmd = [sys.executable, "app.py"]
    if model_path:
        cmd.extend(["--model", model_path])
    if host:
        cmd.extend(["--host", host])
    if port:
        cmd.extend(["--port", str(port)])
    if debug:
        cmd.append("--debug")
    
    print(f"Starting application on http://{host}:{port}...")
    
    # Open the browser after a short delay
    def open_browser():
        time.sleep(2)  # Wait for the server to start
        webbrowser.open(f"http://{host}:{port}")
    
    import threading
    threading.Thread(target=open_browser).start()
    
    # Run the Flask app
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running application: {str(e)}")
        return False
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
        return True

def main():
    parser = argparse.ArgumentParser(description='Tree Species Classification Application')
    
    # Application mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--run', action='store_true', help='Run the application')
    mode_group.add_argument('--train', action='store_true', help='Train a new model')
    mode_group.add_argument('--generate-dataset', action='store_true', help='Generate a sample dataset')
    mode_group.add_argument('--download-model', action='store_true', help='Download a pre-trained model')
    
    # Model options
    parser.add_argument('--model', type=str, default=None, help='Path to the model to use')
    parser.add_argument('--model-name', type=str, default='efficientnetb0', 
                        choices=['efficientnetb0', 'efficientnetb3', 'resnet50v2', 'mobilenetv2'],
                        help='Model architecture to use')
    
    # Dataset options
    parser.add_argument('--dataset', type=str, default='data/tree_dataset', help='Path to the dataset')
    parser.add_argument('--num-images', type=int, default=50, help='Number of images per class for dataset generation')
    
    # Training options
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    
    # Application options
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the application on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the application on')
    parser.add_argument('--debug', action='store_true', help='Run the application in debug mode')
    
    # Dependency options
    parser.add_argument('--install-deps', action='store_true', help='Install dependencies')
    
    args = parser.parse_args()
    
    # Check and install dependencies if requested
    if args.install_deps or not check_dependencies():
        if not install_dependencies():
            print("Failed to install dependencies. Please install them manually.")
            return
    
    # Handle different modes
    if args.generate_dataset:
        generate_sample_dataset(args.num_images)
    elif args.download_model:
        download_model(args.model_name)
    elif args.train:
        if not os.path.exists(args.dataset):
            print(f"Dataset not found at {args.dataset}. Generating sample dataset...")
            generate_sample_dataset(args.num_images)
        train_model(args.dataset, args.epochs, args.batch_size, args.model_name)
    else:  # Default to run mode
        model_path = args.model
        
        # If no model specified, check for default model or download one
        if not model_path:
            default_model_path = f"model/{args.model_name}_tree_classifier"
            if check_model(default_model_path):
                model_path = default_model_path
            else:
                print(f"No model specified and default model not found at {default_model_path}.")
                print("Downloading pre-trained model...")
                if download_model(args.model_name):
                    model_path = default_model_path
        
        # Run the application
        run_app(model_path, args.host, args.port, args.debug)

if __name__ == "__main__":
    main()