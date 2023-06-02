import torchvision.transforms as transforms
import os
from PIL import Image
import shutil
import random

# Set the paths
dataset_path = './EarVN1_updated/Images'
train_path = './EarVN1_updated/train'
valid_path = './EarVN1_updated/val'

# Set the percentage of data for validation
validation_split = 0.2

# Iterate through the subject subdirectories
for subject_dir in os.listdir(dataset_path):
    subject_path = os.path.join(dataset_path, subject_dir)
    
    # Create the training and validation subdirectories
    train_subject_path = os.path.join(train_path, subject_dir)
    valid_subject_path = os.path.join(valid_path, subject_dir)
    os.makedirs(train_subject_path, exist_ok=True)
    os.makedirs(valid_subject_path, exist_ok=True)
    
    # Collect the image file paths
    image_paths = [os.path.join(subject_path, image_file) for image_file in os.listdir(subject_path)]
    num_images = len(image_paths)
    
    # Shuffle the image paths
    random.shuffle(image_paths)
    
    # Split the dataset
    num_valid_images = int(num_images * validation_split)
    valid_images = image_paths[:num_valid_images]
    train_images = image_paths[num_valid_images:]
    
    # Move the images to the respective directories
    for image_path in train_images:
        shutil.copy(image_path, train_subject_path)
    
    for image_path in valid_images:
        shutil.copy(image_path, valid_subject_path)

# Training data transform
augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-15, 15)),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5))], p=0.5),
])

for subject in os.listdir(train_path):
    print(subject)
    subject_dir = os.path.join(train_path, subject)

    for image_name in os.listdir(subject_dir):
        image_path = os.path.join(subject_dir, image_name)
        
        image = Image.open(image_path)
        transformed_image = augment_transform(image)
        output_path = os.path.join(subject_dir, 'aug_' + image_name)
        transformed_image.save(output_path)
