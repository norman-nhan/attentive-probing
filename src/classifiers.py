# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from glob import glob

import torch
import torch.nn as nn
from src.utils.attentive_pooler import AttentivePooler

class AttentiveClassifier(nn.Module):
    def __init__(
        self,
        num_verb_classes: int,
        num_manipulated_classes: int,
        num_affected_classes: int,
        embed_dim: int,
        num_heads: int,
        depth: int,
    ):
        assert embed_dim % num_heads == 0, 'num heads must be divisible by embed dim.'
        super().__init__()
        self.embed_dim = embed_dim
        self.num_verb_classes = num_verb_classes
        self.num_mani_classes = num_manipulated_classes
        self.num_affect_classes = num_affected_classes
        
        self.pooler = AttentivePooler(
            num_queries=3,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
        )

        self.verb_classifier   = nn.Linear(embed_dim, self.num_verb_classes, bias=True)
        self.mani_classifier   = nn.Linear(embed_dim, self.num_mani_classes, bias=True)
        self.affect_classifier = nn.Linear(embed_dim, self.num_affect_classes, bias=True)


    def forward(self, x):
        if torch.isnan(x).any():
            print("Nan detected at output of encoder")
            exit(1)

        x = self.pooler(x)  # [B, 3, D]
        x_verb, x_mani, x_affect = x[:, 0, :], x[:, 1, :], x[:, 2, :]
        
        # forward each feature to their cls head
        x_verb = self.verb_classifier(x_verb)
        x_mani = self.mani_classifier(x_mani)
        x_affect = self.affect_classifier(x_affect)
        
        return dict(
            verb=x_verb,
            manipulated=x_mani,
            affected=x_affect,
        )

def init_classifier(
    embed_dim: int,
    num_heads: int,
    num_blocks: int,
    num_verb_classes: int,
    num_mani_classes: int,
    num_affect_classes: int,
    checkpoint: str=None,
):
    classifier = AttentiveClassifier(
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=num_blocks,
        num_verb_classes=num_verb_classes,
        num_manipulated_classes=num_mani_classes,
        num_affected_classes=num_affect_classes,
    ).cuda()
    
    if checkpoint is not None:
        if os.path.isdir(checkpoint):
            ckpt_paths = sorted(glob(os.path.join(checkpoint, "*.pt")))
            ckpt = ckpt_paths[-1]
        elif os.path.isfile(checkpoint):
            ckpt = checkpoint
        ckpt = torch.load(ckpt, map_location="cpu")
        classifier.load_state_dict(ckpt["classifier"])
        del ckpt

    return classifier