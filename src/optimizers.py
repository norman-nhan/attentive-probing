import math
import torch

class WarmupCosineLRSchedule:

    def __init__(self, optimizer, total_steps, warmup_steps,
                 start_lr, ref_lr, final_lr):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1

        if self.step_num < self.warmup_steps:
            progress = self.step_num / max(1, self.warmup_steps)
            lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            progress = (self.step_num - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            lr = self.final_lr + 0.5 * (self.ref_lr - self.final_lr) * (
                1 + math.cos(math.pi * progress)
            )

        for group in self.optimizer.param_groups:
            group["lr"] = lr

class CosineWDSchedule:

    def __init__(self, optimizer, total_steps, ref_wd, final_wd):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.step_num = 0

    def step(self):
        self.step_num += 1

        progress = self.step_num / self.total_steps

        wd = self.final_wd + 0.5 * (self.ref_wd - self.final_wd) * (
            1 + math.cos(math.pi * progress)
        )

        for group in self.optimizer.param_groups:
            group["weight_decay"] = wd

def init_opt(
    params, 
    iterations_per_epoch, 
    num_epochs: int,
    warmup: int,
    lr: float,
    start_lr: float,
    final_lr: float,
    weight_decay: float,
    final_weight_decay: float,
    use_bfloat16: bool=True,
):
    total_steps = num_epochs * iterations_per_epoch

    optimizer = torch.optim.AdamW(
        params,
        lr=start_lr,
        weight_decay=weight_decay,
    )

    scheduler = WarmupCosineLRSchedule(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
    )

    wd_scheduler = CosineWDSchedule(
        optimizer=optimizer,
        total_steps=total_steps,
        ref_wd=weight_decay,
        final_wd=final_weight_decay,
    )

    scaler = torch.amp.GradScaler() if use_bfloat16 else None
    # print("optimizer", optimizer)
    # print("scheduler", scheduler)
    # print("wd_scheduler", wd_scheduler)
    # print("scaler", scaler)
    return optimizer, scheduler, wd_scheduler, scaler