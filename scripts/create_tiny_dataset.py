import os
import json
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import numpy as np

def create_tiny_dataset(num_images=5):
    # Create directories if they don't exist
    os.makedirs('data/tiny_coco/images', exist_ok=True)
    os.makedirs('data/tiny_coco/annotations', exist_ok=True)
    
    # Get some test images (we'll use CIFAR10 since it's built into torchvision)
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True
    )
    
    # COCO format annotations
    coco_annotations = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "car"},
            {"id": 2, "name": "bird"},
            {"id": 3, "name": "cat"},
            {"id": 4, "name": "dog"},
        ]
    }
    
    ann_id = 1
    # Create some test images and fake annotations
    for img_id in range(num_images):
        # Get a CIFAR image
        img, _ = dataset[img_id]
        
        # Save image
        img_filename = f"{img_id:012d}.jpg"
        img_path = os.path.join('data/tiny_coco/images', img_filename)
        img.save(img_path)
        
        # Add image info to COCO format
        img_info = {
            "id": img_id,
            "file_name": img_filename,
            "height": 32,
            "width": 32
        }
        coco_annotations["images"].append(img_info)
        
        # Create 2-3 random boxes per image
        num_boxes = np.random.randint(2, 4)
        for _ in range(num_boxes):
            # Random box coordinates (x1, y1, w, h)
            x1 = np.random.randint(0, 20)
            y1 = np.random.randint(0, 20)
            w = np.random.randint(8, 32 - x1)
            h = np.random.randint(8, 32 - y1)
            
            # Random category
            cat_id = np.random.randint(1, 5)
            
            ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0
            }
            coco_annotations["annotations"].append(ann)
            ann_id += 1
    
    # Save annotations
    with open('data/tiny_coco/annotations/instances.json', 'w') as f:
        json.dump(coco_annotations, f)

if __name__ == '__main__':
    create_tiny_dataset() 