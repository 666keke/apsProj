import os
import torch
from ultralytics import YOLO
from ultralytics.utils import nms as nms_module
from ultralytics.utils.ops import xywh2xyxy

# Your custom NMS function
def document_aware_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.5,
    reading_order_weight: float = 0.3,
    vertical_align_bonus: float = 0.1,
) -> torch.Tensor:
    """
    Document-aware NMS that considers:
    1. Reading order (top-to-bottom, left-to-right)
    2. Vertical alignment (columns)
    3. Horizontal alignment (rows)
    4. Category relationships
    """
    if boxes.size(0) == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    _, order = scores.sort(descending=True)
    
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        
        i = order[0].item()
        keep.append(i)
        
        remaining_idx = order[1:]
        xx1 = torch.maximum(x1[i], x1[remaining_idx])
        yy1 = torch.maximum(y1[i], y1[remaining_idx])
        xx2 = torch.minimum(x2[i], x2[remaining_idx])
        yy2 = torch.minimum(y2[i], y2[remaining_idx])
        
        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[remaining_idx] - inter + 1e-6)
        
        # Document-aware adjustments
        dx = center_x[remaining_idx] - center_x[i]
        dy = center_y[remaining_idx] - center_y[i]
        reading_order_mask = (dx > 0) & (dy > 0)
        reading_order_adjustment = reading_order_mask.float() * reading_order_weight
        
        horizontal_overlap = (torch.minimum(x2[i], x2[remaining_idx]) - 
                             torch.maximum(x1[i], x1[remaining_idx])) / \
                            (torch.maximum(x2[i], x2[remaining_idx]) - 
                             torch.minimum(x1[i], x1[remaining_idx]) + 1e-6)
        vertical_aligned = horizontal_overlap > 0.7
        vertical_adjustment = vertical_aligned.float() * vertical_align_bonus
        
        vertical_overlap = (torch.minimum(y2[i], y2[remaining_idx]) - 
                           torch.maximum(y1[i], y1[remaining_idx])) / \
                          (torch.maximum(y2[i], y2[remaining_idx]) - 
                           torch.minimum(y1[i], y1[remaining_idx]) + 1e-6)
        horizontal_aligned = vertical_overlap > 0.7
        horizontal_adjustment = horizontal_aligned.float() * vertical_align_bonus
        
        same_category = (labels[i] == labels[remaining_idx]).float()
        category_adjustment = (1 - same_category) * 0.2
        
        adjusted_threshold = (iou_threshold - 
                             reading_order_adjustment - 
                             vertical_adjustment - 
                             horizontal_adjustment - 
                             category_adjustment)
        
        mask = iou <= adjusted_threshold
        order = order[1:][mask]
    
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

# Store original NMS function
original_nms = nms_module.non_max_suppression

def custom_nms_wrapper(
    prediction,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes=None,
    agnostic: bool = False,
    multi_label: bool = False,
    labels=(),
    max_det: int = 300,
    nc: int = 0,
    max_time_img: float = 0.05,
    max_nms: int = 30000,
    max_wh: int = 7680,
    rotated: bool = False,
    end2end: bool = False,
    return_idxs: bool = False,
):
    """Wrapper to use custom NMS in YOLO validation"""
    
    # Handle YOLOv8 validation model output: (inference_out, loss_out)
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]  # select only inference output
    
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)
    
    # If end2end model, return as-is (no NMS needed)
    if end2end or prediction.shape[-1] == 6:
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return (output, None) if return_idxs else output
    
    # Get batch size and number of classes
    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    extra = prediction.shape[1] - nc - 4  # number of extra info (e.g., masks)
    mi = 4 + nc  # mask start index
    
    # Filter by confidence to get candidates
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates
    
    # Transpose from [batch, features, anchors] to [batch, anchors, features]
    prediction = prediction.transpose(-1, -2)  # shape(bs, 84, 6300) -> shape(bs, 6300, 84)
    
    # Convert boxes from xywh to xyxy format
    if not rotated:
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
    
    output = [torch.zeros((0, 6 + extra), device=prediction.device)] * bs
    kept_indices = [torch.zeros((0,), dtype=torch.long, device=prediction.device)] * bs
    
    for xi, x in enumerate(prediction):  # image index
        # Apply confidence filter
        x = x[xc[xi]]
        
        # If none remain, process next image
        if not x.shape[0]:
            continue
        
        # Split into box, cls, mask
        box, cls, mask = x.split((4, nc, extra), 1) if extra else (x[:, :4], x[:, 4:], torch.empty((x.shape[0], 0), device=x.device))
        
        if multi_label:
            # Multi-label: get all classes above threshold
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:
            # Single label: best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]
        
        # Filter by class if specified
        if classes is not None:
            x = x[(x[:, 5:6] == classes).any(1)]
        
        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        
        # Get boxes, scores, and classes for NMS
        boxes = x[:, :4]
        scores = x[:, 4]
        cls_ids = x[:, 5]
        
        # Apply document-aware NMS (per-class by default unless agnostic=True)
        if agnostic:
            # Class-agnostic NMS
            keep = document_aware_nms(
                boxes=boxes,
                scores=scores,
                labels=cls_ids,
                iou_threshold=iou_thres
            )
        else:
            # Per-class NMS using class offsets
            c = cls_ids * (0 if agnostic else max_wh)  # class offsets
            boxes_for_nms = boxes + c[:, None]  # offset boxes by class
            
            # Apply document-aware NMS per class
            keep = []
            for class_id in cls_ids.unique():
                mask = cls_ids == class_id
                if mask.sum() == 0:
                    continue
                
                class_boxes = boxes[mask]
                class_scores = scores[mask]
                class_labels = cls_ids[mask]
                
                class_keep = document_aware_nms(
                    boxes=class_boxes,
                    scores=class_scores,
                    labels=class_labels,
                    iou_threshold=iou_thres
                )
                
                # Map back to original indices
                original_idx = torch.where(mask)[0][class_keep]
                keep.append(original_idx)
            
            if len(keep):
                keep = torch.cat(keep)
                # Sort by confidence
                keep = keep[scores[keep].argsort(descending=True)]
            else:
                keep = torch.zeros((0,), dtype=torch.long, device=x.device)
        
        # Limit to max_det
        keep = keep[:max_det]
        
        output[xi] = x[keep]
        if return_idxs:
            kept_indices[xi] = keep
    
    return (output, kept_indices) if return_idxs else output

# Load model
model = YOLO('/Users/jihaoran/Downloads/model1/test-batch-inference/runs/detect/train4/weights/last.pt')

print("="*50)
print("TESTING WITH CUSTOM DOCUMENT-AWARE NMS")
print("="*50)

# Monkey-patch the NMS function in the correct module
nms_module.non_max_suppression = custom_nms_wrapper

results = model.val(
    data='datasets/hybrid_publaynet_yolo/data.yaml',
    conf=0.01,
    iou=0.5,
    batch=16,
    imgsz=640,
    device='cpu',
    save_json=True,
    save_hybrid=True,
    plots=True
)

# Print metrics
print(f"mAP50: {results.box.map50:.4f}")
print(f"mAP50-95: {results.box.map:.4f}")
print(f"Precision: {results.box.mp:.4f}")
print(f"Recall: {results.box.mr:.4f}")
if results.box.mp + results.box.mr > 0:
    print(f"F1-Score: {2 * results.box.mp * results.box.mr / (results.box.mp + results.box.mr):.4f}")
else:
    print(f"F1-Score: 0.0000")
print("="*50)

# Restore original NMS
nms_module.non_max_suppression = original_nms

print("\n" + "="*50)
print("TESTING WITH ORIGINAL YOLO NMS")
print("="*50)

results = model.val(
    data='datasets/hybrid_publaynet_yolo/data.yaml',
    conf=0.01,
    iou=0.5,
    batch=16,
    imgsz=640,
    device='cpu',
    save_json=True,
    save_hybrid=True,
    plots=True
)

print(f"mAP50: {results.box.map50:.4f}")
print(f"mAP50-95: {results.box.map:.4f}")
print(f"Precision: {results.box.mp:.4f}")
print(f"Recall: {results.box.mr:.4f}")
if results.box.mp + results.box.mr > 0:
    print(f"F1-Score: {2 * results.box.mp * results.box.mr / (results.box.mp + results.box.mr):.4f}")
else:
    print(f"F1-Score: 0.0000")
print("="*50)
