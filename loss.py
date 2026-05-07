import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


def normalize(x):
    return x / 255.0


def dice_coeff(prediction, target):
    # Nếu output là logits (từ FocalLoss), cần qua Sigmoid
    if not torch.all((prediction >= 0) & (prediction <= 1)):
        prediction = torch.sigmoid(prediction)

    # Chuyển về nhị phân 0/1
    prediction = (prediction > 0.5).float()
    target = target.float()

    smooth = 1e-6
    intersect = torch.sum(prediction * target)
    union = torch.sum(prediction) + torch.sum(target)

    dice = (2.0 * intersect + smooth) / (union + smooth)
    return dice.item()  # Trả về số thực đơn thuần


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
        # inputs and targets are assumed to be BatchxClasses
        assert len(input.shape) == len(target.shape)
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)

        # compute the negative likelyhood
        logpt = -F.binary_cross_entropy_with_logits(input, target)
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        balanced_focal_loss = self.balance_param * focal_loss
        return balanced_focal_loss
