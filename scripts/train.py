import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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

def build_model(num_classes=91):  # COCO has 80 classes but we use 91 (historical reasons)
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
        outputs = model(images)
        
        # Compute loss
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        
        if batch_idx % 50 == 0:
            print(f'Batch [{batch_idx}], Loss: {losses.item():.4f}')
            
    return total_loss / len(data_loader)

def main():
    # 1. Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Create datasets
    train_dataset = CocoDetection(
        img_folder='path/to/train2017',
        ann_file='path/to/annotations/instances_train2017.json',
        transforms=TrainTransforms(min_size=800, max_size=1333)
    )
    
    val_dataset = CocoDetection(
        img_folder='path/to/val2017',
        ann_file='path/to/annotations/instances_val2017.json',
        transforms=ValTransforms(min_size=800)
    )
    
    # 3. Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2,  # start small, increase if memory allows
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # 4. Build model
    model = build_model(num_classes=91)
    model = model.to(device)
    
    # 5. Setup criterion
    matcher = HungarianMatcher(
        cost_class=1.0,
        cost_bbox=5.0,
        cost_giou=2.0
    )
    
    criterion = DETRCriterion(
        num_classes=90,  # 91 - 1 because we don't predict "background" explicitly
        matcher=matcher,
        eos_coef=0.1,
        weight_dict={'loss_ce': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0}
    )
    criterion = criterion.to(device)
    
    # 6. Setup optimizer
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if "backbone" not in n and p.requires_grad]
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if "backbone" in n and p.requires_grad],
            "lr": 1e-5,  # lower lr for backbone
        },
    ]
    optimizer = optim.AdamW(param_dicts, lr=1e-4, weight_decay=1e-4)
    
    # 7. Training loop
    num_epochs = 150  # paper trains for 150 epochs
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}')
        
        # Train one epoch
        train_loss = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device
        )
        
        print(f'Epoch {epoch}, Avg Loss: {train_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')

if __name__ == '__main__':
    main()
