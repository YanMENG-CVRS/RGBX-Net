import torch
import torch.nn as nn


class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)

        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (preds * targets).sum(dim=1)
        union = preds.sum(dim=1) + targets.sum(dim=1) - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)
        loss = 1 - iou.mean()
        return loss


class SetCriterion(nn.Module):
    def __init__(self):
        super(SetCriterion, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.iou_loss = IoULoss()

    def forward(self, outputs, targets):
        bce = self.bce_loss(outputs, targets)
        iou = self.iou_loss(outputs, targets)
        loss = bce + iou
        return loss
