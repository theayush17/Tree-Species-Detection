import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import matplotlib.pyplot as plt
import json
import datetime
from model import create_model, fine_tune_model, save_model
from utils import plot_training_history, create_confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser(description='Train a tree species classification model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the dataset')
    parser.add_argument('--output_dir', type=str, default='../model', help='Directory to save the model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for initial training')
    parser.add_argument('--fine_tune_epochs', type=int, default=10, help='Number of epochs for fine-tuning')
    parser.add_argument('--img_size', type=int, default=224, help='Image size (height and width)')
    parser.add_argument('--base_model', type=str, default='EfficientNetB0', 
                        help='Base model architecture (EfficientNetB0, ResNet50V2, etc.)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--fine_tune', action='store_true', help='Whether to fine-tune the model after initial training')
    
    return parser.parse_args()

def prepare_dataset(data_dir, img_size, batch_size):
    """
    Prepare training, validation, and test datasets
    
    Args:
        data_dir: Directory containing the dataset
        img_size: Image size (height and width)
        batch_size: Batch size for training
        
    Returns:
        train_generator: Training data generator
        val_generator: Validation data generator
        test_generator: Test data generator
        class_names: List of class names
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 20% for validation
    )
    
    # Only rescaling for validation and test
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Training generator with data augmentation
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    # Validation generator
    val_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    # Test generator
    test_generator = test_datagen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Get class names
    class_names = list(train_generator.class_indices.keys())
    
    return train_generator, val_generator, test_generator, class_names

def train_model(args):
    """
    Train the tree species classification model
    
    Args:
        args: Command line arguments
    """
    # Prepare datasets
    train_generator, val_generator, test_generator, class_names = prepare_dataset(
        args.data_dir, args.img_size, args.batch_size
    )
    
    # Number of classes
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    # Create model
    model = create_model(
        num_classes=num_classes,
        input_shape=(args.img_size, args.img_size, 3),
        base_model_name=args.base_model
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.join(args.output_dir, 'saved_model')
    logs_dir = os.path.join(args.output_dir, 'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(logs_dir, exist_ok=True)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(args.output_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(log_dir=logs_dir)
    ]
    
    # Initial training
    print("\nStarting initial training...")
    history = model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    # Fine-tuning if requested
    if args.fine_tune:
        print("\nFine-tuning the model...")
        model = fine_tune_model(model)
        
        # Train with fine-tuning
        fine_tune_history = model.fit(
            train_generator,
            epochs=args.fine_tune_epochs,
            validation_data=val_generator,
            callbacks=callbacks
        )
        
        # Combine histories
        for key in fine_tune_history.history:
            history.history[key].extend(fine_tune_history.history[key])
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save model and class names
    save_model(model, model_dir, class_names)
    
    # Plot training history
    history_plot_path = os.path.join(args.output_dir, 'training_history.png')
    plot_training_history(history, history_plot_path)
    
    # Create confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    create_confusion_matrix(model, test_generator, class_names, cm_path)
    
    print(f"\nTraining completed. Model saved to {model_dir}")

if __name__ == "__main__":
    args = parse_args()
    train_model(args)