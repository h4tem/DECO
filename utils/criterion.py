import torch
import torch.nn as nn
import torch.nn.functional as F

class DETRCriterion(nn.Module):
    """
    This module computes the loss for DECO/DETR given the Hungarian matches.
    """
    def __init__(self, num_classes, matcher, eos_coef=0.1,
                 weight_dict={'loss_ce':1.0,'loss_bbox':5.0,'loss_giou':2.0}):
        """
        num_classes: number of object classes (not counting 'no-object')
        matcher: HungarianMatcher instance
        eos_coef: coefficient for the no-object class in the classification
        weight_dict: weighting for different loss components
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.weight_dict = weight_dict

        # In DETR, they treat "no-object" as an extra class in the class logits
        # We'll define a weight so that "no-object" gets scaled by eos_coef.
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def forward(self, outputs, targets):
        """
        outputs: dict containing:
          'pred_logits' => (B, num_queries, num_classes)
          'pred_boxes'  => (B, num_queries, 4)
        targets: list of length B, each a dict with:
          'labels' => (num_gt,), 'boxes' => (num_gt, 4)
        Returns:
          A dict of losses: 'loss_ce', 'loss_bbox', 'loss_giou'
        """
        # 1. Perform matching
        indices = self.matcher(outputs, targets)

        # 2. Compute classification loss
        loss_ce = self.loss_labels(outputs, targets, indices)

        # 3. Compute bounding box losses
        loss_bbox, loss_giou = self.loss_boxes(outputs, targets, indices)

        losses = {}
        losses['loss_ce']   = loss_ce   * self.weight_dict['loss_ce']
        losses['loss_bbox'] = loss_bbox * self.weight_dict['loss_bbox']
        losses['loss_giou'] = loss_giou * self.weight_dict['loss_giou']
        return losses

    def loss_labels(self, outputs, targets, indices):
        """
        Classification loss: cross-entropy between predicted class and target label.
        Unmatched predictions => 'no-object' (class index = num_classes).
        """
        pred_logits = outputs['pred_logits']  # (B, num_queries, num_classes)
        bs, num_queries, _ = pred_logits.shape

        # We'll make a (B, num_queries) array of target class indices,
        # defaulting to 'no-object' = self.num_classes
        target_classes = torch.full((bs, num_queries), self.num_classes,
                                    dtype=torch.long, device=pred_logits.device)
        
        for b, (pred_inds, tgt_inds) in enumerate(indices):
            # The matched predictions -> ground-truth labels
            target_classes[b, pred_inds] = targets[b]['labels'][tgt_inds]

        # We do a cross_entropy with 'num_classes+1' possible classes
        # (the extra one is 'no-object').
        pred_logits = pred_logits.transpose(1, 2)  # (B, num_classes, num_queries)
        loss_ce = F.cross_entropy(pred_logits, target_classes, weight=self.empty_weight)
        return loss_ce

    def loss_boxes(self, outputs, targets, indices):
        """
        Compute L1 + GIoU loss on matched boxes only.
        """
        pred_boxes = outputs['pred_boxes']  # (B, num_queries, 4)

        # We'll collect matched predictions and targets
        matched_pred_boxes = []
        matched_tgt_boxes  = []
        for b, (pred_inds, tgt_inds) in enumerate(indices):
            matched_pred_boxes.append(pred_boxes[b, pred_inds])
            matched_tgt_boxes.append(targets[b]['boxes'][tgt_inds])
        if len(matched_pred_boxes) == 0:
            return torch.tensor(0.0, device=pred_boxes.device), torch.tensor(0.0, device=pred_boxes.device)

        matched_pred_boxes = torch.cat(matched_pred_boxes, dim=0)  # (sum(num_gt), 4)
        matched_tgt_boxes  = torch.cat(matched_tgt_boxes, dim=0)   # (sum(num_gt), 4)

        # 1) L1 loss
        loss_bbox = F.l1_loss(matched_pred_boxes, matched_tgt_boxes, reduction='mean')

        # 2) GIoU loss
        loss_giou = 1 - self.batched_giou(matched_pred_boxes, matched_tgt_boxes)
        loss_giou = loss_giou.mean()

        return loss_bbox, loss_giou

    def batched_giou(self, boxes1, boxes2):
        """
        Compute GIoU between corresponding pairs of boxes.
        
        Args:
            boxes1: shape (N, 4) in (x1, y1, x2, y2) format
            boxes2: shape (N, 4) in (x1, y1, x2, y2) format
        Returns:
            giou: shape (N,) containing GIoU for each pair
        """
        # Get box coordinates
        x1, y1, x2, y2 = boxes1.unbind(-1)  # each is (N,)
        x1g, y1g, x2g, y2g = boxes2.unbind(-1)  # each is (N,)

        # Compute areas
        area1 = (x2 - x1) * (y2 - y1)  # (N,)
        area2 = (x2g - x1g) * (y2g - y1g)  # (N,)

        # Find intersection coordinates
        xkis1 = torch.max(x1, x1g)  # (N,)
        ykis1 = torch.max(y1, y1g)  # (N,)
        xkis2 = torch.min(x2, x2g)  # (N,)
        ykis2 = torch.min(y2, y2g)  # (N,)

        # Compute intersection area
        intsctk = torch.zeros_like(xkis1)
        mask = (xkis2 > xkis1) & (ykis2 > ykis1)
        intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])

        # Compute union area
        unionk = area1 + area2 - intsctk

        # Find smallest enclosing box coordinates
        xc1 = torch.min(x1, x1g)  # (N,)
        yc1 = torch.min(y1, y1g)  # (N,)
        xc2 = torch.max(x2, x2g)  # (N,)
        yc2 = torch.max(y2, y2g)  # (N,)

        # Compute area of enclosing box
        area_c = (xc2 - xc1) * (yc2 - yc1)

        # Compute GIoU
        iou = intsctk / unionk
        giou = iou - ((area_c - unionk) / area_c)

        return giou
