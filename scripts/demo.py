import os
import sys
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from models.deco_model import DECOModel
from models.backbone.resnet18 import ResNet18Backbone
from models.encoder.deco_encoder import DECOEncoder
from models.decoder.deco_decoder import DECODecoder
from utils.data_utils import CocoDetection, TrainTransforms, collate_fn

def visualize_batch(images, targets):
    """Helper to visualize a batch of training data"""
    # Create output directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    for idx, (img, target) in enumerate(zip(images, targets)):
        # Denormalize and convert to numpy
        img = img * std + mean
        img = img.permute(1, 2, 0).numpy()
        
        # Plot image
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        
        # Plot boxes (they're in normalized coordinates)
        h, w = img.shape[:2]
        boxes = target['boxes']
        for box in boxes:
            x1, y1, x2, y2 = box.numpy() * np.array([w, h, w, h])
            plt.gca().add_patch(plt.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                fill=False, color='red', linewidth=2
            ))
            
        plt.axis('off')
        plt.savefig(f'outputs/demo_img_{idx}.png')
        plt.close()
        
        if idx >= 2:  # Save max 3 images
            break
    
    print(f"Saved visualization plots to outputs/demo_img_*.png")

def main():
    # 1. Create tiny dataset first
    from create_tiny_dataset import create_tiny_dataset
    create_tiny_dataset(num_images=5)
    
    # 2. Create dataset with tiny COCO
    dataset = CocoDetection(
        img_folder='data/tiny_coco/images',
        ann_file='data/tiny_coco/annotations/instances.json',
        transforms=TrainTransforms(min_size=224, max_size=224)  # smaller size for demo
    )
    
    # 3. Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )
    
    # 4. Test data loading
    print("Testing data loading...")
    batch = next(iter(loader))
    images, targets = batch
    print(f"Batch shapes:")
    print(f"- Images: {images.shape}")
    print(f"- Number of targets: {len(targets)}")
    print(f"- First image targets: {targets[0]['boxes'].shape} boxes, {targets[0]['labels'].shape} labels")
    
    # 5. Visualize batch
    print("\nVisualizing batch with boxes...")
    visualize_batch(images, targets)
    
    # 6. Create model and test forward pass
    print("\nTesting model forward pass...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DECOModel(
        backbone=ResNet18Backbone(pretrained=True),
        encoder=DECOEncoder(in_channels=512, d_model=256),
        decoder=DECODecoder(d_model=256, num_queries=100, num_classes=5)  # 4 classes + background
    ).to(device)
    
    # Test forward pass
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
    
    print("\nModel outputs:")
    print(f"- Class logits shape: {outputs[0].shape}")
    print(f"- Box coordinates shape: {outputs[1].shape}")
    
    print("\nAll tests completed!")

if __name__ == '__main__':
    main()
