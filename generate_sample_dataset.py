import os
import argparse
import requests
import zipfile
import shutil
import random
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

# Define tree species for the sample dataset
TREE_SPECIES = [
    "Oak", "Maple", "Pine", "Birch", "Palm", 
    "Spruce", "Willow", "Cedar", "Cypress", "Fir"
]

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

def create_synthetic_tree_image(species, size=(224, 224), variation=0):
    """
    Create a synthetic tree image for the given species
    """
    # Create a base image with a sky background
    img = Image.new('RGB', size, color=(135 + random.randint(-20, 20), 
                                       206 + random.randint(-20, 20), 
                                       235 + random.randint(-20, 20)))
    draw = ImageDraw.Draw(img)
    
    # Draw ground
    ground_height = random.randint(40, 60)
    ground_color = (34 + random.randint(-10, 10), 
                    139 + random.randint(-20, 20), 
                    34 + random.randint(-10, 10))
    draw.rectangle([0, size[1] - ground_height, size[0], size[1]], fill=ground_color)
    
    # Draw tree trunk
    trunk_width = random.randint(20, 40)
    trunk_height = random.randint(80, 120)
    trunk_left = (size[0] - trunk_width) // 2
    trunk_top = size[1] - ground_height - trunk_height
    trunk_color = (139 + random.randint(-20, 20), 
                   69 + random.randint(-10, 10), 
                   19 + random.randint(-10, 10))
    draw.rectangle([trunk_left, trunk_top, trunk_left + trunk_width, size[1] - ground_height], 
                   fill=trunk_color)
    
    # Draw tree crown based on species
    crown_color = None
    if species == "Oak":
        # Round, broad crown
        crown_color = (34 + random.randint(-10, 10), 
                       139 + random.randint(-20, 20), 
                       34 + random.randint(-10, 10))
        crown_radius = random.randint(60, 80)
        draw.ellipse([trunk_left + trunk_width//2 - crown_radius, 
                      trunk_top - crown_radius, 
                      trunk_left + trunk_width//2 + crown_radius, 
                      trunk_top], 
                     fill=crown_color)
        
    elif species == "Maple":
        # Maple has a reddish crown
        crown_color = (178 + random.randint(-30, 30), 
                       34 + random.randint(-10, 10), 
                       34 + random.randint(-10, 10))
        crown_radius = random.randint(60, 80)
        draw.ellipse([trunk_left + trunk_width//2 - crown_radius, 
                      trunk_top - crown_radius, 
                      trunk_left + trunk_width//2 + crown_radius, 
                      trunk_top], 
                     fill=crown_color)
        
    elif species == "Pine" or species == "Spruce" or species == "Fir":
        # Conical shape for conifers
        crown_color = (0 + random.randint(0, 30), 
                       100 + random.randint(-20, 20), 
                       0 + random.randint(0, 30))
        # Draw multiple triangles
        triangle_width = random.randint(100, 140)
        triangle_height = random.randint(30, 50)
        layers = random.randint(3, 5)
        
        for i in range(layers):
            layer_width = triangle_width - (i * (triangle_width // layers))
            top_y = trunk_top - (i * triangle_height)
            draw.polygon([
                (trunk_left + trunk_width//2, top_y - triangle_height),
                (trunk_left + trunk_width//2 - layer_width//2, top_y),
                (trunk_left + trunk_width//2 + layer_width//2, top_y)
            ], fill=crown_color)
            
    elif species == "Birch":
        # Birch has a light green crown and white trunk
        crown_color = (144 + random.randint(-20, 20), 
                       238 + random.randint(-30, 30), 
                       144 + random.randint(-20, 20))
        # Overwrite trunk with white
        trunk_color = (245 + random.randint(-10, 10), 
                       245 + random.randint(-10, 10), 
                       220 + random.randint(-10, 10))
        draw.rectangle([trunk_left, trunk_top, trunk_left + trunk_width, size[1] - ground_height], 
                       fill=trunk_color)
        # Add black marks to trunk
        for _ in range(random.randint(5, 10)):
            mark_y = trunk_top + random.randint(0, trunk_height)
            mark_width = random.randint(5, 10)
            draw.rectangle([trunk_left, mark_y, trunk_left + trunk_width, mark_y + 2], 
                           fill=(0, 0, 0))
        
        # Oval crown
        crown_width = random.randint(80, 100)
        crown_height = random.randint(100, 130)
        draw.ellipse([trunk_left + trunk_width//2 - crown_width//2, 
                      trunk_top - crown_height, 
                      trunk_left + trunk_width//2 + crown_width//2, 
                      trunk_top], 
                     fill=crown_color)
        
    elif species == "Palm":
        # Palm tree has a thin trunk and fan-like leaves
        # Redraw trunk to be thinner
        trunk_width = random.randint(10, 20)
        trunk_left = (size[0] - trunk_width) // 2
        trunk_color = (139 + random.randint(-20, 20), 
                       90 + random.randint(-10, 10), 
                       43 + random.randint(-10, 10))
        draw.rectangle([trunk_left, trunk_top, trunk_left + trunk_width, size[1] - ground_height], 
                       fill=trunk_color)
        
        # Draw palm leaves
        crown_color = (0 + random.randint(0, 30), 
                       128 + random.randint(-20, 20), 
                       0 + random.randint(0, 30))
        
        center_x = trunk_left + trunk_width//2
        center_y = trunk_top
        
        # Draw fan-like leaves
        for angle in range(0, 360, 40):
            angle += random.randint(-10, 10)
            length = random.randint(50, 70)
            end_x = center_x + int(length * np.cos(np.radians(angle)))
            end_y = center_y + int(length * np.sin(np.radians(angle)))
            draw.line([center_x, center_y, end_x, end_y], fill=crown_color, width=5)
            
    else:  # Default for other species
        crown_color = (34 + random.randint(-10, 10), 
                       139 + random.randint(-20, 20), 
                       34 + random.randint(-10, 10))
        crown_radius = random.randint(60, 80)
        draw.ellipse([trunk_left + trunk_width//2 - crown_radius, 
                      trunk_top - crown_radius, 
                      trunk_left + trunk_width//2 + crown_radius, 
                      trunk_top], 
                     fill=crown_color)
    
    # Apply some filters and enhancements for more realism
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    
    return img

def generate_dataset(output_dir, num_images_per_class=100, image_size=(224, 224)):
    """
    Generate a synthetic dataset of tree images
    """
    # Create the dataset directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Create train, validation, and test directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "validation")
    test_dir = os.path.join(output_dir, "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Create class subdirectories
    for species in TREE_SPECIES:
        os.makedirs(os.path.join(train_dir, species), exist_ok=True)
        os.makedirs(os.path.join(val_dir, species), exist_ok=True)
        os.makedirs(os.path.join(test_dir, species), exist_ok=True)
    
    # Generate images for each class
    for species in TREE_SPECIES:
        print(f"Generating images for {species}...")
        
        # Calculate split counts (70% train, 15% validation, 15% test)
        train_count = int(num_images_per_class * 0.7)
        val_count = int(num_images_per_class * 0.15)
        test_count = num_images_per_class - train_count - val_count
        
        # Generate training images
        for i in tqdm(range(train_count), desc=f"Training images for {species}"):
            img = create_synthetic_tree_image(species, size=image_size, variation=i)
            img.save(os.path.join(train_dir, species, f"{species.lower()}_{i:04d}.jpg"))
        
        # Generate validation images
        for i in tqdm(range(val_count), desc=f"Validation images for {species}"):
            img = create_synthetic_tree_image(species, size=image_size, variation=i+train_count)
            img.save(os.path.join(val_dir, species, f"{species.lower()}_{i:04d}.jpg"))
        
        # Generate test images
        for i in tqdm(range(test_count), desc=f"Test images for {species}"):
            img = create_synthetic_tree_image(species, size=image_size, variation=i+train_count+val_count)
            img.save(os.path.join(test_dir, species, f"{species.lower()}_{i:04d}.jpg"))
    
    print(f"\nDataset generation complete! Generated {num_images_per_class * len(TREE_SPECIES)} images.")
    print(f"Dataset saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Generate a synthetic dataset of tree images')
    parser.add_argument('--output', type=str, default='data/tree_dataset',
                        help='Output directory to save the dataset')
    parser.add_argument('--num-images', type=int, default=100,
                        help='Number of images to generate per class')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Size of the generated images (square)')
    
    args = parser.parse_args()
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    generate_dataset(args.output, args.num_images, (args.image_size, args.image_size))
    
    print("\nTo train a model on this dataset, run:")
    print(f"python model/train.py --dataset {args.output} --epochs 10 --batch-size 32")

if __name__ == "__main__":
    main()