import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf

def resize_image(image, target_size=(224, 224)):
    """
    Resize an image to the target size
    
    Args:
        image: Input image (numpy array or PIL Image)
        target_size: Target size (height, width)
        
    Returns:
        Resized image as numpy array
    """
    if isinstance(image, np.ndarray):
        # Convert numpy array to PIL Image
        image = Image.fromarray(image)
    
    # Resize the image
    resized_image = image.resize(target_size, Image.LANCZOS)
    
    # Convert back to numpy array
    return np.array(resized_image)

def normalize_image(image):
    """
    Normalize image pixel values to [0, 1]
    
    Args:
        image: Input image (numpy array)
        
    Returns:
        Normalized image
    """
    return image.astype(np.float32) / 255.0

def apply_augmentation(image, rotation_range=20, flip=True, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2)):
    """
    Apply data augmentation to an image
    
    Args:
        image: Input image (PIL Image)
        rotation_range: Maximum rotation angle in degrees
        flip: Whether to apply horizontal flip
        brightness_range: Range for brightness adjustment
        contrast_range: Range for contrast adjustment
        
    Returns:
        Augmented image as PIL Image
    """
    if isinstance(image, np.ndarray):
        # Convert numpy array to PIL Image
        image = Image.fromarray(image)
    
    # Random rotation
    if rotation_range > 0:
        angle = np.random.uniform(-rotation_range, rotation_range)
        image = image.rotate(angle, resample=Image.BILINEAR, expand=False)
    
    # Random horizontal flip
    if flip and np.random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Random brightness adjustment
    if brightness_range is not None:
        brightness_factor = np.random.uniform(brightness_range[0], brightness_range[1])
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)
    
    # Random contrast adjustment
    if contrast_range is not None:
        contrast_factor = np.random.uniform(contrast_range[0], contrast_range[1])
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)
    
    return image

def preprocess_for_model(image, target_size=(224, 224), normalize=True):
    """
    Preprocess an image for model input
    
    Args:
        image: Input image (file path, PIL Image, or numpy array)
        target_size: Target size (height, width)
        normalize: Whether to normalize pixel values to [0, 1]
        
    Returns:
        Preprocessed image ready for model input
    """
    # Load image if path is provided
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Resize image
    image = cv2.resize(image, target_size)
    
    # Ensure image has 3 channels (RGB)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Normalize pixel values if requested
    if normalize:
        image = image.astype(np.float32) / 255.0
    
    return image

def batch_preprocess_images(input_dir, output_dir, target_size=(224, 224), augment=False, num_augmentations=0):
    """
    Preprocess all images in a directory and save to output directory
    
    Args:
        input_dir: Directory containing images or class subdirectories
        output_dir: Directory to save preprocessed images
        target_size: Target size for resizing (height, width)
        augment: Whether to apply data augmentation
        num_augmentations: Number of augmented versions to create per image
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if input_dir contains class subdirectories
    has_class_dirs = any(os.path.isdir(os.path.join(input_dir, d)) for d in os.listdir(input_dir))
    
    if has_class_dirs:
        # Process each class directory
        class_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
        
        for class_name in class_dirs:
            class_input_dir = os.path.join(input_dir, class_name)
            class_output_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_output_dir, exist_ok=True)
            
            # Process images in this class
            process_directory_images(
                class_input_dir, 
                class_output_dir, 
                target_size, 
                augment, 
                num_augmentations
            )
    else:
        # Process all images in the input directory
        process_directory_images(
            input_dir, 
            output_dir, 
            target_size, 
            augment, 
            num_augmentations
        )

def process_directory_images(input_dir, output_dir, target_size=(224, 224), augment=False, num_augmentations=0):
    """
    Process all images in a directory
    
    Args:
        input_dir: Directory containing images
        output_dir: Directory to save preprocessed images
        target_size: Target size for resizing (height, width)
        augment: Whether to apply data augmentation
        num_augmentations: Number of augmented versions to create per image
    """
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Process each image
    for img_file in tqdm(image_files, desc=f"Processing {os.path.basename(input_dir)}"):
        # Load image
        img_path = os.path.join(input_dir, img_file)
        try:
            image = Image.open(img_path).convert('RGB')
            
            # Save preprocessed original image
            processed_img = preprocess_for_model(image, target_size, normalize=False)
            output_path = os.path.join(output_dir, img_file)
            cv2.imwrite(output_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
            
            # Create augmented versions if requested
            if augment and num_augmentations > 0:
                for i in range(num_augmentations):
                    # Apply augmentation
                    augmented_img = apply_augmentation(image)
                    
                    # Preprocess augmented image
                    processed_aug_img = preprocess_for_model(augmented_img, target_size, normalize=False)
                    
                    # Save augmented image
                    base_name, ext = os.path.splitext(img_file)
                    aug_filename = f"{base_name}_aug_{i+1}{ext}"
                    aug_output_path = os.path.join(output_dir, aug_filename)
                    
                    cv2.imwrite(aug_output_path, cv2.cvtColor(processed_aug_img, cv2.COLOR_RGB2BGR))
        
        except Exception as e:
            print(f"Error processing {img_file}: {e}")

def visualize_preprocessing(image_path, target_size=(224, 224)):
    """
    Visualize the preprocessing steps for an image
    
    Args:
        image_path: Path to the input image
        target_size: Target size for resizing (height, width)
    """
    # Load original image
    original_img = Image.open(image_path).convert('RGB')
    
    # Resize image
    resized_img = original_img.resize(target_size, Image.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(resized_img)
    
    # Normalize image
    normalized_img = img_array.astype(np.float32) / 255.0
    
    # Create augmented versions
    augmented_imgs = [
        apply_augmentation(original_img) for _ in range(3)
    ]
    
    # Visualize
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(original_img)
    plt.title('Original Image')
    plt.axis('off')
    
    # Resized image
    plt.subplot(2, 3, 2)
    plt.imshow(resized_img)
    plt.title(f'Resized to {target_size}')
    plt.axis('off')
    
    # Normalized image
    plt.subplot(2, 3, 3)
    plt.imshow(normalized_img)
    plt.title('Normalized [0, 1]')
    plt.axis('off')
    
    # Augmented images
    for i, aug_img in enumerate(augmented_imgs):
        plt.subplot(2, 3, 4 + i)
        plt.imshow(aug_img)
        plt.title(f'Augmented {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Example: Preprocess a single image
    # image_path = "path/to/image.jpg"
    # preprocessed_img = preprocess_for_model(image_path)
    
    # Example: Batch preprocess images
    # batch_preprocess_images(
    #     input_dir="path/to/input/images",
    #     output_dir="path/to/output/images",
    #     target_size=(224, 224),
    #     augment=True,
    #     num_augmentations=3
    # )
    
    print("Preprocessing module loaded successfully.")