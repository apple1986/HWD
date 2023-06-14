from cv2 import imread
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from torch.nn.modules.loss import _Loss, _WeightedLoss

class ImageBasedCrossEntropyLoss2d(nn.Module):
    """
    Image Weighted Cross Entropy Loss
    """

    def __init__(self, classes, weight=None, ignore_index=255,
                    upper_bound=100.0, norm=False,  batch_weights=True):
        super(ImageBasedCrossEntropyLoss2d, self).__init__()
        logging.info("Using Per Image based weighted loss")
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss(weight, ignore_index=ignore_index)
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = batch_weights

    def calculate_weights(self, target):
        """
        Calculate weights of classes based on the training crop
        """
        hist = np.histogram(target.flatten(), range(self.num_classes + 1), density=True)[0]
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist

    def forward(self, inputs, targets):

        target_cpu = targets.data.cpu().numpy()
        # target_cpu = targets.numpy()
        if self.batch_weights:
            weights = self.calculate_weights(target_cpu)
            self.nll_loss.weight = torch.Tensor(weights).cuda()
            loss = self.nll_loss(F.log_softmax(inputs, dim=1), targets)

        else:
            loss = 0.0
            for i in range(0, inputs.shape[0]):
                if not self.batch_weights:
                    weights = self.calculate_weights(target_cpu[i])
                    self.nll_loss.weight = torch.Tensor(weights).cuda()

                loss += self.nll_loss(F.log_softmax(inputs[i].unsqueeze(0), dim=1),
                                      targets[i].unsqueeze(0))
        
        return loss