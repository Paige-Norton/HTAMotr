import torch
import torch.nn.functional as F

def smooth_l1_loss(prediction, target):
    # Calculate Smooth L1 Loss element-wise
    loss = F.smooth_l1_loss(prediction, target, reduction='none')
    return loss

def match_boxes(predictions, targets, iou_threshold=0.5):
    # Calculate IoU between each predicted box and each target box
    ious = calculate_iou(predictions, targets)

    # Find the best matching target box for each predicted box
    best_target_per_prediction = ious.argmax(dim=1)
    
    # Filter out predictions that do not have a good match
    mask = ious[torch.arange(ious.size(0)), best_target_per_prediction] > iou_threshold
    return best_target_per_prediction, mask

def calculate_iou(boxes1, boxes2):
    # Calculate IoU between two sets of bounding boxes

    intersection_x1 = torch.max(boxes1[:, 0].unsqueeze(1), boxes2[:, 0])
    intersection_y1 = torch.max(boxes1[:, 1].unsqueeze(1), boxes2[:, 1])
    intersection_x2 = torch.min(boxes1[:, 0] + boxes1[:, 2], boxes2[:, 0] + boxes2[:, 2])
    intersection_y2 = torch.min(boxes1[:, 1] + boxes1[:, 3], boxes2[:, 1] + boxes2[:, 3])

    intersection_area = torch.clamp(intersection_x2 - intersection_x1, min=0) * torch.clamp(intersection_y2 - intersection_y1, min=0)

    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]
    union_area = area1.unsqueeze(1) + area2 - intersection_area

    iou = intersection_area / (union_area + 1e-6)

    return iou

def multibox_smooth_l1_loss(predictions, targets, iou_threshold=0.5):
    # predictions: Tensor of shape (N, 4), where N is the number of predicted bounding boxes
    # targets: Tensor of shape (M, 4), where M is the number of target bounding boxes
    
    # Find the best matching target box for each predicted box
    best_target_per_prediction, mask = match_boxes(predictions, targets, iou_threshold=iou_threshold)

    # Use the best matching targets for Smooth L1 Loss calculation
    selected_targets = targets[best_target_per_prediction[mask]]

    # Calculate Smooth L1 Loss
    loss = smooth_l1_loss(predictions[mask], selected_targets)

    return loss.mean()

# Example usage
predictions = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], dtype=torch.float32)
targets = torch.tensor([[2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0]], dtype=torch.float32)
loss = multibox_smooth_l1_loss(predictions, targets)
print("Smooth L1 Loss:", loss.item())
