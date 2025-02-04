import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class HungarianMatcher:
    """
    This class computes an assignment between targets and predictions
    for a batch using the Hungarian algorithm.
    """

    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        """
        cost_class: weight for classification cost
        cost_bbox: weight for L1 box distance
        cost_giou: weight for GIoU cost
        """
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def __call__(self, outputs, targets):
        """
        Parameters:
        -----------
        outputs: dict with:
            'pred_logits': (B, num_queries, num_classes),
            'pred_boxes':  (B, num_queries, 4)   [format depends on you]
        targets: list of length B, each a dict with:
            'labels': (num_gt,), 'boxes': (num_gt, 4)

        Returns:
        --------
        A list of size B, where each element is (pred_indices, tgt_indices)
        that define the matching between predicted boxes and GT boxes.
        """
        bs = outputs['pred_logits'].shape[0]
        
        # We'll apply a softmax over the class dimension to get probabilities
        out_prob = F.softmax(outputs['pred_logits'], dim=-1)  # (B, num_queries, num_classes)
        out_bbox = outputs['pred_boxes']                      # (B, num_queries, 4)

        indices = []
        for b in range(bs):
            # predicted prob for this batch element => shape (num_queries, num_classes)
            prob = out_prob[b]
            # predicted boxes => shape (num_queries, 4)
            bbox = out_bbox[b]

            # gt labels => shape (num_gt,)
            tgt_ids = targets[b]['labels']
            # gt boxes => shape (num_gt, 4)
            tgt_bbox = targets[b]['boxes']

            num_gt = tgt_bbox.shape[0]
            if num_gt == 0:
                # if no ground truth, just an empty matching
                indices.append((torch.empty(0, dtype=torch.long),
                                torch.empty(0, dtype=torch.long)))
                continue

            # Classification cost = -log(prob_of_gt_class)
            # shape => (num_queries, num_gt)
            cost_class = -prob[:, tgt_ids]

            # L1 distance between predicted boxes and gt boxes
            # shape => (num_queries, num_gt)
            bbox_i = bbox.unsqueeze(1)      # (num_queries, 1, 4)
            tgt_bbox_i = tgt_bbox.unsqueeze(0)  # (1, num_gt, 4)
            cost_bbox = torch.cdist(bbox_i, tgt_bbox_i, p=1).squeeze(1)  # (num_queries, num_gt)

            # GIoU cost => we define a function generalized_iou that returns GIoU
            cost_giou = 1 - generalized_iou(bbox_i, tgt_bbox_i)  # (num_queries, num_gt)

            # Final cost matrix
            cost_matrix = (
                self.cost_class * cost_class 
                + self.cost_bbox  * cost_bbox 
                + self.cost_giou  * cost_giou.squeeze(1)
            )

            cost_matrix = cost_matrix.cpu()

            # Solve the assignment using Hungarian algorithm
            pred_indices, tgt_indices = linear_sum_assignment(cost_matrix)
            pred_indices = torch.as_tensor(pred_indices, dtype=torch.int64)
            tgt_indices  = torch.as_tensor(tgt_indices, dtype=torch.int64)

            indices.append((pred_indices, tgt_indices))

        return indices

def generalized_iou(boxes1, boxes2):
    """
    Compute GIoU for pairs of boxes in (x1, y1, x2, y2) format.

    Input:
      boxes1 => shape (num_queries, 1, 4)
      boxes2 => shape (1, num_gt, 4)
    Output:
      giou => shape (num_queries, num_gt)
    """
    # Get box coordinates
    x1, y1, x2, y2 = boxes1.unbind(-1)  # each is (num_queries, 1)
    x1g, y1g, x2g, y2g = boxes2.unbind(-1)  # each is (1, num_gt)

    # Compute areas of boxes1 and boxes2
    area1 = (x2 - x1) * (y2 - y1)  # (num_queries, 1)
    area2 = (x2g - x1g) * (y2g - y1g)  # (1, num_gt)

    # Find intersection coordinates
    xkis1 = torch.max(x1, x1g)  # (num_queries, num_gt)
    ykis1 = torch.max(y1, y1g)  # (num_queries, num_gt)
    xkis2 = torch.min(x2, x2g)  # (num_queries, num_gt)
    ykis2 = torch.min(y2, y2g)  # (num_queries, num_gt)

    # Compute intersection area
    intsctk = torch.zeros_like(xkis1)
    mask = (xkis2 > xkis1) & (ykis2 > ykis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])

    # Compute union area
    unionk = area1 + area2 - intsctk

    # Find smallest enclosing box coordinates
    xc1 = torch.min(x1, x1g)  # (num_queries, num_gt)
    yc1 = torch.min(y1, y1g)  # (num_queries, num_gt)
    xc2 = torch.max(x2, x2g)  # (num_queries, num_gt)
    yc2 = torch.max(y2, y2g)  # (num_queries, num_gt)

    # Compute area of enclosing box
    area_c = (xc2 - xc1) * (yc2 - yc1)

    # Compute GIoU
    iou = intsctk / unionk
    giou = iou - ((area_c - unionk) / area_c)

    return giou
