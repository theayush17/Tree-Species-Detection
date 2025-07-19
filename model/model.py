import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.optimizers import Adam
import os
import json

def create_model(num_classes, input_shape=(224, 224, 3), base_model_name='EfficientNetB0'):
    """
    Create a transfer learning model for tree species classification
    
    Args:
        num_classes: Number of tree species classes
        input_shape: Input image dimensions (height, width, channels)
        base_model_name: Name of the pre-trained model to use as base
        
    Returns:
        Compiled Keras model
    """
    # Available base models
    base_models = {
        'EfficientNetB0': applications.EfficientNetB0,
        'EfficientNetB3': applications.EfficientNetB3,
        'ResNet50V2': applications.ResNet50V2,
        'MobileNetV2': applications.MobileNetV2,
        'InceptionV3': applications.InceptionV3
    }
    
    # Select base model
    if base_model_name not in base_models:
        raise ValueError(f"Base model {base_model_name} not supported. Choose from: {list(base_models.keys())}")
    
    # Create base model with pre-trained weights
    base_model = base_models[base_model_name](
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Create new model on top
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def fine_tune_model(model, num_layers_to_unfreeze=10):
    """
    Fine-tune the model by unfreezing some top layers of the base model
    
    Args:
        model: The pre-trained model
        num_layers_to_unfreeze: Number of top layers to unfreeze for fine-tuning
        
    Returns:
        Fine-tuned model
    """
    # Unfreeze the base_model for fine-tuning
    base_model = model.layers[0]
    
    # Unfreeze the top N layers
    for layer in base_model.layers[-num_layers_to_unfreeze:]:
        layer.trainable = True
    
    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # Lower learning rate for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def save_model(model, model_dir='saved_model', class_names=None):
    """
    Save the trained model and class names
    
    Args:
        model: Trained Keras model
        model_dir: Directory to save the model
        class_names: List of class names
    """
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model
    model.save(model_dir)
    print(f"Model saved to {model_dir}")
    
    # Save class names if provided
    if class_names is not None:
        class_names_path = os.path.join(os.path.dirname(model_dir), 'class_names.json')
        with open(class_names_path, 'w') as f:
            json.dump(class_names, f)
        print(f"Class names saved to {class_names_path}")

def load_model(model_dir='saved_model', class_names_path=None):
    """
    Load a saved model and class names
    
    Args:
        model_dir: Directory where the model is saved
        class_names_path: Path to the class names JSON file
        
    Returns:
        model: Loaded Keras model
        class_names: List of class names (if available)
    """
    # Load the model
    model = tf.keras.models.load_model(model_dir)
    
    # Load class names if path is provided
    class_names = None
    if class_names_path and os.path.exists(class_names_path):
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
    
    return model, class_names

# Example usage
if __name__ == "__main__":
    # Example: Create a model for 10 tree species
    model = create_model(num_classes=10)
    print(model.summary())
    
    # Example class names
    example_classes = [
        "Oak", "Maple", "Pine", "Birch", "Cedar",
        "Spruce", "Willow", "Elm", "Fir", "Redwood"
    ]
    
    # Example: Save model (commented out to prevent accidental overwriting)
    # save_model(model, 'saved_model', example_classes)