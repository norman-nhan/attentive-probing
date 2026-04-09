import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn

from decord import VideoReader, cpu

from probing.metrics import ClassMeanRecall

def decode_clip(batch):
    buffers = []
    verb_list = []
    manipulated_list = []
    affected_list = []

    # group samples by video path to avoid reopening same video repeatedly
    data_dict = {}
    for data in batch:
        data_dict.setdefault(data["path"], []).append(data)

    for video_path, data_list in data_dict.items():
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=8)

        for data in data_list:
            frames = vr.get_batch(data["indices"]).asnumpy()   # (T, H, W, C)
            frames = torch.from_numpy(frames)                  # torch tensor
            frames = frames.permute(0, 3, 1, 2)               # (T, C, H, W)

            buffers.append(frames)
            verb_list.append(data["verb_id"])
            manipulated_list.append(data["manipulated_id"])
            affected_list.append(data["affected_id"])

    buffers = torch.stack(buffers, dim=0)                     # (B, T, C, H, W)
    verb = torch.tensor(verb_list, dtype=torch.long)
    mani = torch.tensor(manipulated_list, dtype=torch.long)
    affect = torch.tensor(affected_list, dtype=torch.long)

    return buffers, verb, mani, affect

def fix_random_seed(seed, include_cuda=True):
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_cuda:
        # training: disable cudnn benchmark to ensure the reproducibility
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # this is needed for CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        cudnn.enabled = True
        cudnn.benchmark = True
    return rng_generator

def train_one_epoch(
    model,
    processor,
    classifier,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    num_verb_classes,
    num_mani_classes,
    num_affect_classes,
    criterion,
    logger = None,
    log_freq=10,
):
    device = model.device

    # frozen encoder train classifier
    model.eval()
    classifier.train()
    
    total_loss = 0.0
    total_verb_loss = 0.0
    total_manipulated_loss = 0.0
    total_affected_loss = 0.0
    num_batches = 0

    verb_logger   = ClassMeanRecall(num_verb_classes, device=device, k=5)
    mani_logger   = ClassMeanRecall(num_mani_classes, device=device, k=5)
    affect_logger = ClassMeanRecall(num_affect_classes, device=device, k=5)
    
    progress = tqdm(data_loader)
    
    for step, batch in enumerate(progress):
        buffer, gt_verb, gt_mani, gt_affect = batch

        buffer = processor(
            buffer,
            return_tensors="pt",
            do_resize=False
        )["pixel_values_videos"].float().to(device, non_blocking=True)

        gt_verb = gt_verb.to(device, non_blocking=True)
        gt_mani = gt_mani.to(device, non_blocking=True)
        gt_affect = gt_affect.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # extract feature with frozen encoder
        with torch.no_grad():
            outputs = model.encoder(buffer).last_hidden_state   # (B, N, D)

        outputs = classifier(outputs)

        # calculate losses
        verb_loss = criterion(outputs["verb"], gt_verb)
        manipulated_loss = criterion(outputs["manipulated"], gt_mani)
        affected_loss = criterion(outputs["affected"], gt_affect)

        loss = verb_loss + manipulated_loss + affected_loss

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()
        if wd_scheduler is not None:
            wd_scheduler.step()
    
        total_loss += loss.item()
        total_verb_loss += verb_loss.item()
        total_manipulated_loss += manipulated_loss.item()
        total_affected_loss += affected_loss.item()
        num_batches += 1
        
        verb_metric = verb_logger(outputs["verb"], gt_verb)
        mani_metric = mani_logger(outputs["manipulated"], gt_mani)
        affect_metric = affect_logger(outputs["affected"], gt_affect)

        # print losses every log freq
        if step % log_freq == 0:
            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "verb_loss": f"{verb_loss.item():.4f}",
                "manip_loss": f"{manipulated_loss.item():.4f}",
                "aff_loss": f"{affected_loss.item():.4f}",
            })
            if logger is not None:
                logger.log({
                    "train/step_loss": loss.item(),
                    "train/verb_loss_step": verb_loss.item(),
                    "train/manipulated_loss_step": manipulated_loss.item(),
                    "train/affected_loss_step": affected_loss.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                })
    
    if num_batches == 0:
        raise RuntimeError("data_loader is empty.")
    
    ret = dict(
        total_loss=total_loss / num_batches,
        verb=dict(
            loss=total_verb_loss / num_batches,
            **verb_metric,
        ),
        manipulated=dict(
            loss=total_manipulated_loss / num_batches,
            **mani_metric,
        ),
        affected=dict(
            loss=total_affected_loss / num_batches,
            **affect_metric,
        ),
    )

    return ret

def valid_one_epoch(
    model,
    processor,
    classifier,
    data_loader,
    num_verb_classes,
    num_mani_classes,
    num_affect_classes,
    criterion,
    logger=None,
    log_freq=10,
):
    device = model.device

    model.eval()
    classifier.eval()

    total_loss = 0.0
    total_verb_loss = 0.0
    total_manipulated_loss = 0.0
    total_affected_loss = 0.0
    num_batches = 0

    verb_logger = ClassMeanRecall(num_verb_classes, device=device, k=5)
    mani_logger = ClassMeanRecall(num_mani_classes, device=device, k=5)
    affect_logger = ClassMeanRecall(num_affect_classes, device=device, k=5)

    progress = tqdm(data_loader)

    with torch.no_grad():
        for step, batch in enumerate(progress):
            buffer, gt_verb, gt_mani, gt_affect = batch

            buffer = processor(
                buffer,
                return_tensors="pt",
                do_resize=False,
            )["pixel_values_videos"].float().to(device, non_blocking=True)

            gt_verb = gt_verb.to(device, non_blocking=True)
            gt_mani = gt_mani.to(device, non_blocking=True)
            gt_affect = gt_affect.to(device, non_blocking=True)

            outputs = model.encoder(buffer).last_hidden_state
            outputs = classifier(outputs)

            verb_loss = criterion(outputs["verb"], gt_verb)
            manipulated_loss = criterion(outputs["manipulated"], gt_mani)
            affected_loss = criterion(outputs["affected"], gt_affect)

            loss = verb_loss + manipulated_loss + affected_loss

            total_loss += loss.item()
            total_verb_loss += verb_loss.item()
            total_manipulated_loss += manipulated_loss.item()
            total_affected_loss += affected_loss.item()
            num_batches += 1

            verb_metric = verb_logger(outputs["verb"], gt_verb)
            mani_metric = mani_logger(outputs["manipulated"], gt_mani)
            affect_metric = affect_logger(outputs["affected"], gt_affect)

            if step % log_freq == 0:
                progress.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "verb_loss": f"{verb_loss.item():.4f}",
                    "manip_loss": f"{manipulated_loss.item():.4f}",
                    "aff_loss": f"{affected_loss.item():.4f}",
                })
                if logger is not None:
                    logger.log({
                        "valid/step_loss": loss.item(),
                        "valid/verb_loss_step": verb_loss.item(),
                        "valid/manipulated_loss_step": manipulated_loss.item(),
                        "valid/affected_loss_step": affected_loss.item(),
                    })

    if num_batches == 0:
        raise RuntimeError("data_loader is empty.")

    ret = dict(
        total_loss=total_loss / num_batches,
        verb=dict(
            loss=total_verb_loss / num_batches,
            **verb_metric,
        ),
        manipulated=dict(
            loss=total_manipulated_loss / num_batches,
            **mani_metric,
        ),
        affected=dict(
            loss=total_affected_loss / num_batches,
            **affect_metric,
        ),
    )

    return ret