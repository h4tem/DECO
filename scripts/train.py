# Training script for DECO (Detection with Convolutions)
import os
import sys
import random
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from models.deco_model import DECOModel
from models.backbone.resnet18 import ResNet18Backbone
from models.encoder.deco_encoder import DECOEncoder
from models.decoder.deco_decoder import DECODecoder
from utils.matcher import HungarianMatcher
from utils.criterion import DETRCriterion
from utils.data_utils import (
    CocoDetection, 
    TrainTransforms, 
    ValTransforms,
    collate_fn
)

def build_model(num_classes=91):  # COCO has 80 classes + background
    # 1. Backbone
    backbone = ResNet18Backbone(pretrained=True)
    
    # 2. DECO Encoder (takes backbone features)
    encoder = DECOEncoder(
        in_channels=512,  # ResNet18 last stage channels
        d_model=256,
        num_layers=3,
        kernel_size=7
    )
    
    # 3. DECO Decoder
    decoder = DECODecoder(
        d_model=256,
        num_queries=100,
        num_layers=6,  # paper's ablation shows 6 is good
        num_classes=num_classes,
        kernel_size=9  # paper's ablation shows 9x9 is best
    )
    
    # 4. Full model
    model = DECOModel(backbone, encoder, decoder)
    return model

def train_one_epoch(model, criterion, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (images, targets) in enumerate(data_loader):
        # Move to device
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        class_logits, pred_boxes = model(images)
        outputs = {
            'pred_logits': class_logits,
            'pred_boxes': pred_boxes
        }
        
        # Compute loss
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        
        if batch_idx % 10 == 0:
            print(f'Batch [{batch_idx}], Loss: {losses.item():.4f}')
            print(f'  - Class loss: {loss_dict["loss_ce"].item():.4f}')
            print(f'  - Box loss: {loss_dict["loss_bbox"].item():.4f}')
            print(f'  - GIoU loss: {loss_dict["loss_giou"].item():.4f}')
            
    return total_loss / len(data_loader)

def main():
    # 1. Setup paths for COCO dataset
    data_dir = project_root / 'data'
    img_folder = data_dir / 'train2017'
    ann_file = data_dir / 'annotations' / 'instances_train2017.json'
    
    if not img_folder.exists() or not ann_file.exists():
        print("Please download COCO dataset first:")
        print("1. Download train2017.zip and annotations_trainval2017.zip")
        print("2. Extract them to data/ directory")
        return
    
    # 2. Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 3. Create dataset
    full_dataset = CocoDetection(
        img_folder=str(img_folder),
        ann_file=str(ann_file),
        transforms=TrainTransforms(min_size=320, max_size=320)  # Slightly larger for real data
    )
    
    # Create a subset of 1000 images
    num_train = 1000
    indices = list(range(len(full_dataset)))
    random.seed(42)  # For reproducibility
    subset_indices = random.sample(indices, num_train)
    train_dataset = Subset(full_dataset, subset_indices)
    print(f"\nUsing {num_train} images from COCO train2017")
    
    # 4. Create dataloader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8,  # Increased for better training
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # 5. Build model (with COCO classes)
    model = build_model(num_classes=91)  # COCO has 80 classes + background
    model = model.to(device)
    
    # 6. Setup criterion
    matcher = HungarianMatcher(
        cost_class=2.0,
        cost_bbox=5.0,
        cost_giou=2.0
    )
    
    criterion = DETRCriterion(
        num_classes=90,  # COCO has 80 classes (matcher expects num_classes-1)
        matcher=matcher,
        eos_coef=0.1,
        weight_dict={'loss_ce': 2.0, 'loss_bbox': 5.0, 'loss_giou': 2.0}
    )
    criterion = criterion.to(device)
    
    # 7. Setup optimizer
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if "backbone" not in n and p.requires_grad]
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if "backbone" in n and p.requires_grad],
            "lr": 1e-5,  # Lower backbone lr
        },
    ]
    optimizer = optim.AdamW(param_dicts, lr=1e-4, weight_decay=1e-4)
    
    # 8. Training loop
    num_epochs = 50  # Reduced epochs since we have more data
    print("\nStarting training...")
    
    os.makedirs('outputs', exist_ok=True)
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        train_loss = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device
        )
        
        print(f'Epoch {epoch+1}, Avg Loss: {train_loss:.4f}')
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }
        
        # Save periodically
        if (epoch + 1) % 5 == 0:
            torch.save(checkpoint, f'outputs/checkpoint_epoch_{epoch+1}.pth')
            print(f'Saved checkpoint at epoch {epoch+1}')
        
        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(checkpoint, 'outputs/best_model.pth')
            print(f'New best model saved! Loss: {train_loss:.4f}')

if __name__ == '__main__':
    main()
