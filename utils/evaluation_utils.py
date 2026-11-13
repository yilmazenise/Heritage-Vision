import torch

def check_acc(img, mask, model, device):
    num_correct = 0
    num_pixel = 0
    
    model.eval()
    
    with torch.no_grad():
        img = img.to(device)
        mask = mask.to(device)
            
        preds = model(img)
        preds = torch.round(preds)
            
        num_correct += (preds == mask).sum()
        num_pixel += torch.numel(preds)
            
    acc = num_correct/num_pixel*100
    
    return num_correct, num_pixel


def calculate_iou(prediction, target, class_label):

    prediction = torch.round(prediction).type(torch.LongTensor).cpu()
    target = target.cpu()
    
    prediction_mask = (prediction == class_label)
    target_mask = (target == class_label)

    intersection = torch.logical_and(prediction_mask, target_mask)
    union = torch.logical_or(prediction_mask, target_mask)

    if union.sum() == 0:
        iou = 1.0
    else:
        iou = intersection.sum() / union.sum()

    return iou


def calculate_miou(predictions, targets, color_mapping):
    iou_per_class = []

    for label in color_mapping.keys():
        iou = calculate_iou(predictions,targets, label)
        iou_per_class.append(iou)

    miou = (sum(iou_per_class) / len(iou_per_class)) * 100.0

    return miou

            
            
    