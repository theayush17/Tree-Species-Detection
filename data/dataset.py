import os
import shutil
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

def split_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split a dataset of images into training, validation, and test sets
    
    Args:
        source_dir: Directory containing class subdirectories with images
        output_dir: Directory to save the split dataset
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        seed: Random seed for reproducibility
    """
    # Verify ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'validation')
    test_dir = os.path.join(output_dir, 'test')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get class directories
    class_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    # Process each class
    for class_name in class_dirs:
        # Create class directories in each split
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
        # Get all images for this class
        class_dir = os.path.join(source_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Shuffle images
        random.seed(seed)
        random.shuffle(images)
        
        # Calculate split indices
        n_images = len(images)
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)
        
        # Split images
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Copy images to respective directories
        for img in tqdm(train_images, desc=f"Copying {class_name} training images"):
            shutil.copy2(
                os.path.join(class_dir, img),
                os.path.join(train_dir, class_name, img)
            )
        
        for img in tqdm(val_images, desc=f"Copying {class_name} validation images"):
            shutil.copy2(
                os.path.join(class_dir, img),
                os.path.join(val_dir, class_name, img)
            )
        
        for img in tqdm(test_images, desc=f"Copying {class_name} test images"):
            shutil.copy2(
                os.path.join(class_dir, img),
                os.path.join(test_dir, class_name, img)
            )
    
    # Print summary
    print("\nDataset split complete!")
    print(f"Training set: {count_images(train_dir)} images")
    print(f"Validation set: {count_images(val_dir)} images")
    print(f"Test set: {count_images(test_dir)} images")

def count_images(directory):
    """
    Count the total number of images in a directory and its subdirectories
    
    Args:
        directory: Directory to count images in
        
    Returns:
        Total number of images
    """
    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                count += 1
    return count

def create_data_generators(img_size=224, batch_size=32):
    """
    Create data generators for training with augmentation
    
    Args:
        img_size: Target image size
        batch_size: Batch size for training
        
    Returns:
        train_datagen: Training data generator with augmentation
        val_test_datagen: Validation/test data generator without augmentation
    """
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Validation and test data generator (only rescaling)
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    return train_datagen, val_test_datagen

def load_data_generators(data_dir, img_size=224, batch_size=32):
    """
    Load data from directory structure using data generators
    
    Args:
        data_dir: Directory containing train, validation, and test subdirectories
        img_size: Target image size
        batch_size: Batch size for training
        
    Returns:
        train_generator: Training data generator
        val_generator: Validation data generator
        test_generator: Test data generator
        class_names: List of class names
    """
    # Create data generators
    train_datagen, val_test_datagen = create_data_generators(img_size, batch_size)
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    # Load validation data
    val_generator = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, 'validation'),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Load test data
    test_generator = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Get class names
    class_names = list(train_generator.class_indices.keys())
    
    return train_generator, val_generator, test_generator, class_names

def visualize_dataset_samples(data_generator, class_names, num_samples=5):
    """
    Visualize random samples from a data generator
    
    Args:
        data_generator: Data generator to visualize samples from
        class_names: List of class names
        num_samples: Number of samples to visualize
    """
    # Get a batch of images
    images, labels = next(data_generator)
    
    # Create a figure
    plt.figure(figsize=(15, num_samples * 3))
    
    # Plot images
    for i in range(min(num_samples, len(images))):
        plt.subplot(num_samples, 1, i + 1)
        plt.imshow(images[i])
        
        # Get class name from one-hot encoded label
        class_idx = np.argmax(labels[i])
        class_name = class_names[class_idx]
        
        plt.title(f"Class: {class_name}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def analyze_dataset(data_dir):
    """
    Analyze a dataset and print statistics
    
    Args:
        data_dir: Directory containing train, validation, and test subdirectories
        
    Returns:
        DataFrame with dataset statistics
    """
    stats = []
    
    # Analyze each split
    for split in ['train', 'validation', 'test']:
        split_dir = os.path.join(data_dir, split)
        
        if not os.path.exists(split_dir):
            continue
        
        # Get class directories
        class_dirs = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        
        # Count images per class
        for class_name in class_dirs:
            class_dir = os.path.join(split_dir, class_name)
            num_images = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            stats.append({
                'Split': split,
                'Class': class_name,
                'Images': num_images
            })
    
    # Create DataFrame
    df = pd.DataFrame(stats)
    
    # Print summary
    print("Dataset Statistics:")
    print(f"Total classes: {len(df['Class'].unique())}")
    print(f"Total images: {df['Images'].sum()}")
    
    # Print split summary
    split_summary = df.groupby('Split')['Images'].sum().reset_index()
    print("\nImages per split:")
    for _, row in split_summary.iterrows():
        print(f"{row['Split']}: {row['Images']} images")
    
    # Print class distribution
    print("\nClass distribution:")
    class_summary = df.groupby('Class')['Images'].sum().sort_values(ascending=False).reset_index()
    for _, row in class_summary.iterrows():
        print(f"{row['Class']}: {row['Images']} images")
    
    return df

# Example usage
if __name__ == "__main__":
    # Example: Split a dataset
    # split_dataset(
    #     source_dir="path/to/original/dataset",
    #     output_dir="path/to/split/dataset"
    # )
    
    # Example: Analyze a dataset
    # analyze_dataset("path/to/dataset")
    
    print("Dataset module loaded successfully.")