import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


def normalize(x):
    return x / 255.0


def dice_coeff(prediction, target):
    """
    Tính toán Dice Coefficient trực tiếp trên Tensor (GPU hoặc CPU).
    prediction: Tensor đầu ra từ model (thường là logits).
    target: Tensor mask thực tế (Ground Truth).
    """
    # 1. Chuyển đổi Prediction thành xác suất 0-1 (Sigmoid)
    # và đưa về nhị phân (0 hoặc 1)
    # Lưu ý: Vì bạn dùng FocalLoss với Logits,
    # nên đầu ra model thường chưa qua Sigmoid.
    if not torch.all((prediction >= 0) & (prediction <= 1)):
        prediction = torch.sigmoid(prediction)

    # Ngưỡng 0.5 để tạo mask nhị phân
    mask = (prediction >= 0.5).float()
    target = target.float()

    # 2. Tính toán Intersection và Union trên toàn bộ Batch
    # Sử dụng sum() của torch thay vì np.sum()
    inter = torch.sum(mask * target)
    union = torch.sum(mask) + torch.sum(target)

    epsilon = 1e-6
    # 3. Tính Dice
    result = (2.0 * inter) / (union + epsilon)

    # Trả về giá trị số (Python scalar) để log
    return result.item()


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
