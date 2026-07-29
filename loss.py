import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


def normalize(x):
    return x / 255.0


def dice_coeff(prediction, target, threshold=0.5):
    """Dice coefficient (torch version - correct & stable)"""

    prediction = torch.sigmoid(prediction)

    prediction = (prediction > threshold).float()
    target = target.float()

    smooth = 1e-6

    intersection = (prediction * target).sum()
    union = prediction.sum() + target.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return dice.item()


class BCEDiceLoss(nn.Module):
    """Class-balanced BCE plus differentiable Dice loss for binary masks."""

    def __init__(self, pos_weight=20.0, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits, target):
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight
        )

        probabilities = torch.sigmoid(logits)
        probabilities = probabilities.flatten(1)
        target = target.flatten(1)
        intersection = (probabilities * target).sum(dim=1)
        denominator = probabilities.sum(dim=1) + target.sum(dim=1)
        dice_loss = 1.0 - (
            (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        )

        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice_loss.mean()


class FocalLoss(nn.modules.loss._WeightedLoss):

    def __init__(
        self,
        gamma=0,
        size_average=None,
        ignore_index=-100,
        reduce=None,
        balance_param=1.0,
    ):
        super(FocalLoss, self).__init__(size_average)
        self.gamma = gamma
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param

    def forward(self, input, target):
        """Pixel-wise binary focal loss for logits and binary masks."""
        assert len(input.shape) == len(target.shape)
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)

        target = target.float()
        bce = F.binary_cross_entropy_with_logits(input, target, reduction="none")
        pt = torch.exp(-bce)

        focal_loss = ((1 - pt) ** self.gamma) * bce
        balanced_focal_loss = self.balance_param * focal_loss
        return balanced_focal_loss.mean()
