from tqdm import tqdm
import torch


def calculate_iou_single(pred, label):
    hist = torch.bincount(label.long().view(-1) * 2 + pred.view(-1), minlength=4).view(2, 2)
    iou = hist[1, 1] / (hist[1, 1] + hist[0, 1] + hist[1, 0] + 1e-10)
    return iou


def evaluate(model, dataloader, device):
    print("Evaluating...")
    model.eval()

    hist = torch.zeros(2, 2).to(device)
    for minibatch in tqdm(dataloader, dynamic_ncols=True):
        images = minibatch["data"]
        labels = minibatch["label"]

        modal_xs = minibatch["modal_x"]
        images = [images.to(device), modal_xs.to(device)]
        labels = labels.to(device)
        output = model(images[0], images[1])
        # preds = (torch.sigmoid(output) > 0.5).long()
        preds = torch.argmax(output, dim=1)
        hist += torch.bincount(labels.long().view(-1) * 2 + preds.view(-1), minlength=4).view(2, 2)

    TP = hist[1, 1]
    FP = hist[0, 1]
    FN = hist[1, 0]

    iou = TP / (TP + FP + FN + 1e-10)
    miou = torch.diag(hist) / (hist.sum(1) + hist.sum(0) - torch.diag(hist) + 1e-10)
    miou[miou.isnan()] = 0.
    miou = miou.mean()
    Precision = TP / (TP + FP + 1e-10)
    Recall = TP / (TP + FN + 1e-10)
    F1 = 2 * Precision * Recall / (Precision + Recall + 1e-10)
    print(f'IoU: {float(iou.cpu()):.4f}, MIoU: {float(miou.cpu()):.4f}, Precision: {float(Precision.cpu()):.4f}, '
          f'Recall: {float(Recall.cpu()):.4f}, F1: {float(F1.cpu()):.4f}')

    return round(float(iou.cpu()) * 100, 2)
