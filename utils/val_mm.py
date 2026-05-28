import math

import torch
from torch.nn import functional as F
from tqdm import tqdm

from utils.metrics_new import Metrics


@torch.no_grad()
def evaluate(model, dataloader, config, device, engine):
    model.eval()
    metrics = Metrics(config.num_classes, config.background, device)

    for minibatch in tqdm(dataloader, dynamic_ncols=True):
        images = minibatch["data"].to(device)
        modal_xs = minibatch["modal_x"].to(device)
        labels = minibatch["label"].to(device)

        if len(labels.shape) == 2:
            labels = labels.unsqueeze(0)

        preds = model(images, modal_xs).softmax(dim=1)
        metrics.update(preds, labels)

    if engine.distributed:
        all_metrics = [None for _ in range(engine.world_size)]
        torch.distributed.all_gather_object(all_metrics, metrics)
        return all_metrics
    return metrics


@torch.no_grad()
def evaluate_msf(model, dataloader, config, device, scales, flip, engine):
    model.eval()
    metrics = Metrics(config.num_classes, config.background, device)

    for minibatch in tqdm(dataloader, dynamic_ncols=True):
        images = minibatch["data"].to(device)
        modal_xs = minibatch["modal_x"].to(device)
        labels = minibatch["label"].to(device)
        batch_size, height, width = labels.shape
        scaled_logits = torch.zeros(batch_size, config.num_classes, height, width, device=device)

        for scale in scales:
            scaled_h = int(math.ceil(int(scale * height) / 32)) * 32
            scaled_w = int(math.ceil(int(scale * width) / 32)) * 32
            scaled_images = F.interpolate(images, size=(scaled_h, scaled_w), mode="bilinear", align_corners=True)
            scaled_modal_xs = F.interpolate(modal_xs, size=(scaled_h, scaled_w), mode="bilinear", align_corners=True)

            logits = model(scaled_images, scaled_modal_xs)
            logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=True)
            scaled_logits += logits.softmax(dim=1)

            if flip:
                logits = model(torch.flip(scaled_images, dims=(3,)), torch.flip(scaled_modal_xs, dims=(3,)))
                logits = torch.flip(logits, dims=(3,))
                logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=True)
                scaled_logits += logits.softmax(dim=1)

        metrics.update(scaled_logits, labels)

    if engine.distributed:
        all_metrics = [None for _ in range(engine.world_size)]
        torch.distributed.all_gather_object(all_metrics, metrics)
        return all_metrics
    return metrics
