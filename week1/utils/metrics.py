
# Intersection over Union (IoU)
def iou(pred, target):
    """Calculate the Intersection over Union (IoU) of two segmentation masks.

    Args:
        pred (torch.Tensor): predicted mask, shape (N, H, W)
        target (torch.Tensor): target mask, shape (N, H, W)

    Returns:
        torch.Tensor: IoU, shape (N,)
    """
    intersection = (pred & target).float().sum((1, 2))
    union = (pred | target).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou


# Average Precision (AP) for Object Detection