import os
import torch
import torch.utils.data
import torchvision.transforms as T
import torchvision.transforms.functional as F

from PIL import Image
from pycocotools.coco import COCO

class CocoDetection(torch.utils.data.Dataset):
    """
    A minimal COCO dataset class that:
      - Reads images from a folder (train2017 or val2017)
      - Reads annotation JSON via pycocotools
      - Returns images, bounding boxes in (x_min, y_min, x_max, y_max), class labels
    """

    def __init__(self, img_folder, ann_file, transforms=None, remove_crowd=True):
        """
        img_folder: path to COCO images (e.g., 'coco/train2017')
        ann_file: path to JSON (e.g., 'coco/annotations/instances_train2017.json')
        transforms: optional torchvision transforms (if None, will just do ToTensor)
        remove_crowd: if True, ignore any annotation with iscrowd=1
        """
        super().__init__()
        self.img_folder = img_folder
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())  # all image IDs
        self.transforms = transforms
        self.remove_crowd = remove_crowd
        
        # If no transforms provided, just do ToTensor
        if self.transforms is None:
            self.transforms = DetectionTransforms()

        # Create a mapping from category_id to a contiguous label [0..N-1]
        # so that your model sees labels in a [0..num_cats-1] range.
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.catid2label = {cat['id']: idx for idx, cat in enumerate(cats)}

    def __getitem__(self, index):
        img_id = self.ids[index]

        # 1. Load the image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        path = self.coco.imgs[img_id]['file_name']  # e.g. '000000000139.jpg'
        img_path = os.path.join(self.img_folder, path)
        img = Image.open(img_path).convert('RGB')

        # 2. Build target dict
        boxes = []
        labels = []
        for ann in anns:
            if self.remove_crowd and ann.get('iscrowd', 0) == 1:
                # skip crowd
                continue

            # ann['bbox'] is [x_min, y_min, width, height] in absolute coords
            x_min, y_min, w, h = ann['bbox']
            x_max = x_min + w
            y_max = y_min + h
            boxes.append([x_min, y_min, x_max, y_max])

            cat_id = ann['category_id']
            labels.append(self.catid2label[cat_id])  # map to 0..N-1

        # Convert to Tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"]  = boxes    # shape (#obj, 4)
        target["labels"] = labels   # shape (#obj,)

        # 3. (Optional) transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)


def collate_fn(batch):
    """
    Custom collate for object detection:
    returns:
      images: tensor (B, 3, H, W)
      targets: list of length B, each a dict { 'boxes', 'labels' }
    """
    images, targets = list(zip(*batch))

    # images is a tuple of PIL images or Tensors
    # If they're all the same size, we can stack directly.
    # Often, transforms.ToTensor() => each image is (3, H, W).
    # So let's assume they're all Tensors now.
    images = torch.stack(images, dim=0)

    return images, list(targets)


class DetectionTransforms:
    """
    Transforms for DETR/DECO training:
    - Random resize + crop
    - Random horizontal flip
    - Convert to tensors and normalize
    """
    def __init__(self, min_size=800, max_size=1333):
        self.min_size = min_size
        self.max_size = max_size
        
        # ImageNet normalization
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __call__(self, img, target):
        # 1. Convert PIL image to tensor and normalize
        img = F.to_tensor(img)
        img = self.normalize(img)
        
        # 2. Convert boxes to normalized [0, 1] coordinates
        h, w = img.shape[-2:]
        if target is not None and len(target["boxes"]):
            boxes = target["boxes"]
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
            
        return img, target

class TrainTransforms(DetectionTransforms):
    """
    Training transforms with augmentation.
    """
    def __call__(self, img, target):
        # 1. Random horizontal flip
        if torch.rand(1) > 0.5:
            img = F.hflip(img)
            if target is not None and len(target["boxes"]):
                boxes = target["boxes"]
                boxes[:, [0, 2]] = 1 - boxes[:, [2, 0]]  # flip x coordinates
                target["boxes"] = boxes
        
        # 2. Random resize
        h = torch.randint(self.min_size, self.max_size + 1, (1,)).item()
        w = h  # square resize for simplicity
        img = F.resize(img, [h, w])
        
        # 3. Convert to tensor and normalize
        return super().__call__(img, target)

class ValTransforms(DetectionTransforms):
    """
    Validation transforms (no augmentation).
    """
    def __call__(self, img, target):
        # Just resize to min_size
        h = self.min_size
        w = h
        img = F.resize(img, [h, w])
        
        # Convert to tensor and normalize
        return super().__call__(img, target)


