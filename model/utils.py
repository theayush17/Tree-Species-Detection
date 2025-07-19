import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import os

def plot_training_history(history, save_path=None):
    """
    Plot training and validation accuracy/loss
    
    Args:
        history: Training history object from model.fit()
        save_path: Path to save the plot image
    """
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='lower right')
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper right')
    ax2.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    
    plt.show()

def create_confusion_matrix(model, test_generator, class_names, save_path=None):
    """
    Create and plot confusion matrix for model evaluation
    
    Args:
        model: Trained model
        test_generator: Test data generator
        class_names: List of class names
        save_path: Path to save the confusion matrix image
    """
    # Reset the generator
    test_generator.reset()
    
    # Get predictions
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Get true labels
    y_true = test_generator.classes
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess an image for model prediction
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing (height, width)
        
    Returns:
        Preprocessed image ready for model prediction
    """
    # Read image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    
    # Convert to array
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # Expand dimensions to create batch
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess (normalize)
    img_array = img_array / 255.0
    
    return img_array

def predict_image(model, image_path, class_names, target_size=(224, 224), top_k=3):
    """
    Make prediction for a single image
    
    Args:
        model: Trained model
        image_path: Path to the image file
        class_names: List of class names
        target_size: Target size for resizing (height, width)
        top_k: Number of top predictions to return
        
    Returns:
        Dictionary with prediction results
    """
    # Preprocess image
    img_array = preprocess_image(image_path, target_size)
    
    # Make prediction
    predictions = model.predict(img_array)[0]
    
    # Get top k predictions
    top_indices = predictions.argsort()[-top_k:][::-1]
    top_predictions = [
        {"species": class_names[i], "confidence": float(predictions[i])}
        for i in top_indices
    ]
    
    return {
        "top_predictions": top_predictions,
        "species": top_predictions[0]["species"],
        "confidence": top_predictions[0]["confidence"]
    }

def create_data_generators(train_dir, val_dir, test_dir, img_size=224, batch_size=32):
    """
    Create data generators for training, validation, and testing
    
    Args:
        train_dir: Directory containing training data
        val_dir: Directory containing validation data
        test_dir: Directory containing test data
        img_size: Image size (height and width)
        batch_size: Batch size
        
    Returns:
        train_generator: Training data generator
        val_generator: Validation data generator
        test_generator: Test data generator
        class_names: List of class names
    """
    # Data augmentation for training
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation and test
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Get class names
    class_names = list(train_generator.class_indices.keys())
    
    return train_generator, val_generator, test_generator, class_names

def prepare_sample_dataset(output_dir, num_classes=10, samples_per_class=5):
    """
    Create a sample dataset structure for testing
    
    Args:
        output_dir: Directory to create the sample dataset
        num_classes: Number of classes to create
        samples_per_class: Number of sample images per class
        
    Note: This function creates empty directories only, it doesn't generate actual images
    """
    # Sample class names
    class_names = [
        "Oak", "Maple", "Pine", "Birch", "Cedar",
        "Spruce", "Willow", "Elm", "Fir", "Redwood"
    ]
    
    # Ensure we don't exceed available class names
    num_classes = min(num_classes, len(class_names))
    selected_classes = class_names[:num_classes]
    
    # Create directory structure
    for split in ['train', 'validation', 'test']:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        for class_name in selected_classes:
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Create placeholder for sample images
            print(f"Created directory: {class_dir} (add {samples_per_class} images here)")
    
    print(f"\nSample dataset structure created at {output_dir}")
    print("Add actual images to these directories before training.")