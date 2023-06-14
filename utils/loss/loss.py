import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append("/home/gpxu/vess_seg/vess_efficient")

from utils.losses.lovasz_losses import lovasz_softmax
from torch.nn.modules.loss import _Loss, _WeightedLoss
from torch.nn import NLLLoss2d
import kornia as K


__all__ = ["CSLoss2d", "CrossEntropyLoss2d", "CrossEntropyLoss2dLabelSmooth",
           "FocalLoss2d", "LDAMLoss", "ProbOhemCrossEntropy2d",
           "LovaszSoftmax", "DiceLoss", "MultiLabelSoftMarginLoss2d","TopKLoss",
           "DC_and_Focal_loss","DC_and_CS_loss", "CE_and_FC_loss", "CE_and_FM_loss","Feature_Clust_Loss",
           "Median_CE_Loss","Boundary_CE_Loss","Feature_SSIM_Loss", "Img_SSIM_Loss"]


class CSLoss2d(nn.Module):
    def __init__(self, alpha=0.64, gamma=2, weight=None, ignore_index=255, size_average=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.size_average = size_average
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, output, target):

        if output.dim()>2:
            output = output.contiguous().view(output.size(0), output.size(1), -1)
            output = output.transpose(1,2)
            output = output.contiguous().view(-1, output.size(2)).squeeze()
        if target.dim()==4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1,2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim()==3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        logpt = self.ce_fn(output, target)
        pt = torch.exp(-logpt)
        loss = self.gamma * (torch.cos(pt)-self.alpha*torch.sin(pt)) * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


# class CrossEntropyLoss2d(_WeightedLoss):
#     """
#     Standard pytorch weighted nn.CrossEntropyLoss
#     Note: Do not need the softmax and negative log, because it has done in the function.
#     """

#     def __init__(self, weight=None, ignore_index=255, reduction='mean'):
#         super(CrossEntropyLoss2d, self).__init__()

#         self.nll_loss = nn.CrossEntropyLoss(weight, ignore_index=ignore_index, reduction=reduction)

#     def forward(self, output, target):
#         """
#         Forward pass
#         :param output: torch.tensor (NxC)
#         :param target: torch.tensor (N)
#         :return: scalar
#         """
#         return self.nll_loss(output, target)

########CE + softmax
class CrossEntropyLoss2d(_WeightedLoss):
    """
    Standard pytorch weighted nn.CrossEntropyLoss
    Note: Do not need the softmax and negative log, because it has done in the function.
    """

    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()

        self.nll_loss = nn.CrossEntropyLoss(weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        """
        Forward pass
        :param output: torch.tensor (NxC)
        :param target: torch.tensor (N)
        :return: scalar
        """
        return self.nll_loss(output, target)


# class CrossEntropyLoss2d(nn.Module):
#     '''
#     This file defines a cross entropy loss for 2D images
#     '''
#
#     def __init__(self, weight=None, ignore_index=255):
#         '''
#         :param weight: 1D weight vector to deal with the class-imbalance
#         Obtaining log-probabilities in a neural network is easily achieved by adding a LogSoftmax layer in the last layer of your network.
#         You may use CrossEntropyLoss instead, if you prefer not to add an extra layer.
#         '''
#         super().__init__()
#
#         # self.loss = nn.NLLLoss2d(weight, ignore_index=255)
#         self.loss = nn.NLLLoss(weight, ignore_index=ignore_index)
#
#     def forward(self, outputs, targets):
#         return self.loss(F.log_softmax(outputs, dim=1), targets)

class CrossEntropyLoss2dLabelSmooth(_WeightedLoss):
    """
    Refer from https://arxiv.org/pdf/1512.00567.pdf
    :param target: N,
    :param n_classes: int
    :param eta: float
    :return:
        N x C onehot smoothed vector
    """

    def __init__(self, weight=None, ignore_index=255, epsilon=0.1, reduction='mean'):
        super(CrossEntropyLoss2dLabelSmooth, self).__init__()
        self.epsilon = epsilon
        self.ignore_index = ignore_index
        # self.nll_loss = nn.CrossEntropyLoss(weight, ignore_index=ignore_index, 
        #                     reduction=reduction, label_smoothing=epsilon)

    def forward(self, seg_logit, seg_label):
        """
        Forward pass
        :param output: torch.tensor (NxC)
        :param target: torch.tensor (N)
        :return: scalar
        """
        seg_logit = F.log_softmax(seg_logit, dim=1)	# softmax + log

        b, c, h, w = seg_logit.size()
        # 不计算ignore_index的损失
        seg_label = seg_label.unsqueeze(1)
        valid_mask = torch.where(seg_label!=self.ignore_index)
        seg_logit = seg_logit[valid_mask[0],:,valid_mask[2],valid_mask[3]]
        seg_logit = seg_logit.permute(1,0)
        seg_label = seg_label[valid_mask].unsqueeze(0)
        valid_size = seg_label.size()[1]
        # one-hot编码
        seg_label = torch.zeros([c,valid_size]).scatter(dim=0, index=seg_label, value=1)

        seg_label = (1 - self.epsilon) * seg_label + self.epsilon / c

        loss = -1*torch.sum(seg_label*seg_logit, 1).mean()

        return loss

# class CrossEntropyLoss2dLabelSmooth(_WeightedLoss):
#     """
#     Refer from https://arxiv.org/pdf/1512.00567.pdf
#     :param target: N,
#     :param n_classes: int
#     :param eta: float
#     :return:
#         N x C onehot smoothed vector
#     """

#     def __init__(self, weight=None, ignore_index=255, epsilon=0.1, reduction='mean'):
#         super(CrossEntropyLoss2dLabelSmooth, self).__init__()
#         self.epsilon = epsilon
#         self.nll_loss = nn.CrossEntropyLoss(weight, ignore_index=ignore_index, reduction=reduction)

#     def forward(self, output, target):
#         """
#         Forward pass
#         :param output: torch.tensor (NxC)
#         :param target: torch.tensor (N)
#         :return: scalar
#         """
#         n_classes = output.size(1)
#         # batchsize, num_class = input.size()
#         # log_probs = F.log_softmax(inputs, dim=1)
#         targets = torch.zeros_like(output).scatter_(1, target.unsqueeze(1), 1) ## one-hot
#         targets = (1 - self.epsilon) * targets + self.epsilon / n_classes

#         return self.nll_loss(output, targets) ## wrong , need to cal softmax + log firstly


# class CrossEntropyLoss2dLabelSmooth(nn.Module):
#     ''' Cross Entropy Loss with label smoothing '''
#     def __init__(self, label_smooth=None, class_num=11):
#         super().__init__()
#         self.label_smooth = label_smooth
#         self.class_num = class_num

#     def forward(self, pred, target):
#         ''' 
#         Args:
#             pred: prediction of model output    [N, M]
#             target: ground truth of sampler [N]
#         '''
#         eps = 1e-12
        
#         if self.label_smooth is not None:
#             # cross entropy loss with label smoothing
#             logprobs = F.log_softmax(pred, dim=1)	# softmax + log
#             target = F.one_hot(target, self.class_num)	# 转换成one-hot
            
#             # label smoothing
#             # 实现 1
#             # target = (1.0-self.label_smooth)*target + self.label_smooth/self.class_num 	
#             # 实现 2
#             # implement 2
#             target = torch.clamp(target.float(), min=self.label_smooth/(self.class_num-1), max=1.0-self.label_smooth)
#             loss = -1*torch.sum(target*logprobs, 1)
        
#         else:
#             # standard cross entropy loss
#             loss = -1.*pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred+eps).sum(dim=1))

#         return loss.mean()



"""
https://arxiv.org/abs/1708.02002
# Credit to https://github.com/clcarwin/focal_loss_pytorch
"""
class FocalLoss2d(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255, size_average=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.size_average = size_average
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, output, target):

        if output.dim()>2:
            output = output.contiguous().view(output.size(0), output.size(1), -1)
            output = output.transpose(1,2)
            output = output.contiguous().view(-1, output.size(2)).squeeze()
        if target.dim()==4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1,2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim()==3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        logpt = self.ce_fn(output, target)
        pt = torch.exp(-logpt)
        loss = ((1-pt) ** self.gamma) * self.alpha * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

################################
class DC_and_Focal_loss(nn.Module):
    def __init__(self):
        super(DC_and_Focal_loss, self).__init__()
        self.dc = DiceLoss(num_classes=11)
        self.focal = FocalLoss2d(ignore_index=11)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        focal_loss = self.focal(net_output, target)

        result = dc_loss + focal_loss
        return result

################################
class DC_and_CS_loss(nn.Module):
    def __init__(self):
        super(DC_and_CS_loss, self).__init__()
        self.dc = DiceLoss(num_classes=11)
        self.cs = CSLoss2d(ignore_index=11)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        cs_loss = self.cs(net_output, target)

        result = dc_loss + cs_loss
        return result
"""
https://arxiv.org/pdf/1906.07413.pdf
"""
class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)




# # Adapted from OCNet Repository (https://github.com/PkuRainBow/OCNet)
# class ProbOhemCrossEntropy2d(nn.Module):
#     def __init__(self, ignore_index=255, reduction='mean', thresh=0.6, min_kept=256,
#                  down_ratio=1, use_weight=False):
#         super(ProbOhemCrossEntropy2d, self).__init__()
#         self.ignore_index = ignore_index
#         self.thresh = float(thresh)
#         self.min_kept = int(min_kept)
#         self.down_ratio = down_ratio
#         if use_weight:
#             print("w/ class balance")
#             weight = torch.FloatTensor(
#                 [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
#                  0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
#                  1.0865, 1.1529, 1.0507])
#             # data['classWeights']:  [4.4286523 3.443227  9.627174  2.9082294 6.9883366 5.2119217 9.490196
#             # 9.150478  7.1654587 9.888601  7.5455384]
#             self.criterion = nn.CrossEntropyLoss(reduction=reduction,
#                                                        weight=weight,
#                                                        ignore_index=ignore_index)
#         else:
#             print("w/o class balance")
#             self.criterion = nn.CrossEntropyLoss(reduction=reduction,
#                                                        ignore_index=ignore_index)

#     def forward(self, pred, target):
#         b, c, h, w = pred.size()
#         target = target.view(-1)
#         valid_mask = target.ne(self.ignore_index)
#         target = target * valid_mask.long()
#         num_valid = valid_mask.sum()

#         prob = F.softmax(pred, dim=1)
#         prob = (prob.transpose(0, 1)).reshape(c, -1)

#         if self.min_kept > num_valid:
#             print('Labels: {}'.format(num_valid))
#             pass
#         elif num_valid > 0:
#             prob = prob.masked_fill_(1 - valid_mask, 1)     #
#             mask_prob = prob[
#                 target, torch.arange(len(target), dtype=torch.long)]
#             threshold = self.thresh
#             if self.min_kept > 0:
#                 index = mask_prob.argsort()
#                 threshold_index = index[min(len(index), self.min_kept) - 1]
#                 if mask_prob[threshold_index] > self.thresh:
#                     threshold = mask_prob[threshold_index]
#                 kept_mask = mask_prob.le(threshold)
#                 target = target * kept_mask.long()
#                 valid_mask = valid_mask * kept_mask
#                 print('Valid Mask: {}'.format(valid_mask.sum()))

#         target = target.masked_fill_(1 - valid_mask, self.ignore_index)
#         target = target.view(b, h, w)

#         return self.criterion(pred, target)


# Adapted from OCNet Repository (https://github.com/PkuRainBow/OCNet)
class ProbOhemCrossEntropy2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean', thresh=0.6, min_kept=256,
                 down_ratio=1):
        super(ProbOhemCrossEntropy2d, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio

        self.criterion = nn.CrossEntropyLoss(reduction=reduction,
                                                       weight=weight,
                                                       ignore_index=ignore_index)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index) ## ne: not equal
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            print('Labels: {}'.format(num_valid))
            pass
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)     #
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask
                # print('Valid Mask: {}'.format(valid_mask.sum()))

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target)


# ==========================================================================================================================
# ==========================================================================================================================
class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore_index=255):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        loss = lovasz_softmax(logits, target, ignore=self.ignore_index)
        return loss

#####################aping: DiceLoss
class DiceLoss(nn.Module):
    def __init__(self, num_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = num_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

###############################
class MultiLabelSoftMarginLoss2d(_WeightedLoss):
    """
    Standard pytorch weighted nn.CrossEntropyLoss
    Note: Do not need the softmax and negative log, because it has done in the function.
    """

    def __init__(self, num_classes=11, weight=None, ignore_index=255, reduction='mean'):
        super(MultiLabelSoftMarginLoss2d, self).__init__()
        self.n_classes = num_classes

        self.nll_loss = nn.MultiLabelSoftMarginLoss(weight, reduction=reduction)
    
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, output, target):
        """
        Forward pass
        :param output: torch.tensor (NxC)
        :param target: torch.tensor (N)
        :return: scalar
        """
        target = self._one_hot_encoder(target)
        return self.nll_loss(output, target)

class TopKLoss(nn.CrossEntropyLoss):
    """
    Network has to have NO LINEARITY!
    """
    def __init__(self, weight=None, ignore_index=-100, k=10):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False)

    def forward(self, inp, target):
        # target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()

###################ap-2
class Feature_Clust_Loss(nn.Module):
    def __init__(self):
        super(Feature_Clust_Loss, self).__init__()
        self.mse = nn.MSELoss()

    def _cal_cluster(self, extract_region, gt_onehot):
        smooth = 1e-5
        num_class = gt_onehot.sum(2, keepdim=True).sum(3, keepdim=True)
        center_class = extract_region.sum(2, keepdim=True).sum(3, keepdim=True) / (num_class + smooth)

        return center_class

    def _one_hot_encoder(self, gt):
        tensor_list = []
        for i in range(11): ## 11 is the total class number
            temp_prob = gt == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
        
    def forward(self, pd, gt):
        gt_onehot = self._one_hot_encoder(gt)
        if (gt_onehot.shape != pd.shape):
            gt_onehot = F.interpolate(gt_onehot, size=pd.shape[2:], mode='nearest') 
        extract_region = gt_onehot * pd
        center_class = self._cal_cluster(extract_region, gt_onehot)
        center_class = gt_onehot * center_class
        loss = self.mse(pd, center_class)

        return loss
        

class Feature_Max_Loss(nn.Module):
    def __init__(self):
        super(Feature_Max_Loss, self).__init__()
        self.mse = nn.MSELoss()

    def _cal_cluster(self, extract_region):
        # smooth = 1e-5
        # num_class = gt_onehot.sum(2, keepdim=True).sum(3, keepdim=True)
        # center_class = extract_region.sum(2, keepdim=True).sum(3, keepdim=True) / (num_class + smooth)
        center_class = extract_region.max(2,keepdim=True)[0].max(3, keepdim=True)[0]

        return center_class

    def _one_hot_encoder(self, gt):
        tensor_list = []
        for i in range(11): ## 11 is the total class number
            temp_prob = gt == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
        
    def forward(self, pd, gt):
        gt_onehot = self._one_hot_encoder(gt)
        extract_region = gt_onehot * pd
        center_class = self._cal_cluster(extract_region)
        # loss = self.mse(extract_region, center_class)

        center_class = gt_onehot * center_class
        loss = self.mse(pd, center_class)
        return loss

###################ap-3
class Feature_SSIM_Loss(nn.Module):
    def __init__(self):
        super(Feature_SSIM_Loss, self).__init__()
        self.ssim = K.losses.SSIMLoss(window_size=5)

    def _cal_cluster(self, extract_region, gt_onehot):
        smooth = 1e-5
        num_class = gt_onehot.sum(2, keepdim=True).sum(3, keepdim=True)
        center_class = extract_region.sum(2, keepdim=True).sum(3, keepdim=True) / (num_class + smooth)

        return center_class

    def _one_hot_encoder(self, gt):
        tensor_list = []
        for i in range(11): ## 11 is the total class number
            temp_prob = gt == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
        
    def forward(self, pd, gt):
        gt_onehot = self._one_hot_encoder(gt)
        if (gt_onehot.shape != pd.shape):
            gt_onehot = F.interpolate(gt_onehot, size=pd.shape[2:], mode='nearest') 

        pd_prob = F.softmax(pd, dim=1)
        loss = self.ssim(pd_prob, gt_onehot)

        return loss
###################ap-4
class Img_SSIM_Loss(nn.Module):
    def __init__(self):
        super(Img_SSIM_Loss, self).__init__()
        self.ssim = K.losses.SSIMLoss(window_size=5)
        
    def forward(self, pd, gt):
        # pd = F.softmax(pd, dim=1)
        loss = self.ssim(pd, gt)

        return loss

################################
class CE_and_FC_loss(nn.Module):
    def __init__(self, weight=None, ignore_index=11):
        super(CE_and_FC_loss, self).__init__()
        self.ce = CrossEntropyLoss2d(weight, ignore_index=ignore_index)
        self.fc = Feature_Clust_Loss()

    def forward(self, net_output, target):
        ce_loss = self.ce(net_output, target)
        fc_loss = self.fc(net_output, target)

        result = ce_loss +  fc_loss
        print(ce_loss, '=============', fc_loss)
        return result

class CE_and_FM_loss(nn.Module):
    def __init__(self, weight=None, ignore_index=11):
        super(CE_and_FM_loss, self).__init__()
        self.ce = CrossEntropyLoss2d(weight, ignore_index=ignore_index)
        self.fc = Feature_Max_Loss()

    def forward(self, net_output, target):
        ce_loss = self.ce(net_output, target)
        fc_loss = self.fc(net_output, target)

        result = ce_loss +  4.8*fc_loss
        # print(ce_loss, '=============', fc_loss)
        return result

#######################aping-2
class Median_CE_Loss(nn.Module):
    def __init__(self, weight=None, ignore_index=11):
        super(Median_CE_Loss, self).__init__()
        self.ce = CrossEntropyLoss2d(weight=weight, ignore_index=ignore_index)
        self.median = K.filters.MedianBlur((3,3))

        
    def forward(self, pd, gt):

        pd_med = self.median(pd)
        loss = self.ce(pd_med, gt)

        return loss  
##########################aping-3
# class Boundary_CE_Loss(nn.Module):
#     def __init__(self, weight=None, ignore_index=11):
#         super(Boundary_CE_Loss, self).__init__()
#         self.ce = CrossEntropyLoss2d(weight=weight, ignore_index=ignore_index)
#         self.edge = K.filters.Canny()
#         self.kernel = torch.tensor([[0,1,0],[1,1,1],[0,1,0]]).cuda()

#     def _one_hot_encoder(self, gt):
#         tensor_list = []
#         for i in range(11):
#             temp_prob = gt == i  # * torch.ones_like(input_tensor)
#             tensor_list.append(temp_prob.unsqueeze(1))
#         output_tensor = torch.cat(tensor_list, dim=1)
#         return output_tensor.float()

#     def _get_edge(self, gt_onehot):
#         edge_gt_list = []
#         for n in range(gt_onehot.shape[1]):
#             temp_edge = self.edge(gt_onehot[:,n,::].unsqueeze(1))[1]
#             temp_edge=K.morphology.dilation(temp_edge, self.kernel)
#             edge_gt_list.append(temp_edge)
#         edge_gt = torch.cat(edge_gt_list, dim=1)
#         return edge_gt.float()
        
#     def forward(self, pd, gt):
#         gt_onehot = self._one_hot_encoder(gt)
#         edge_gt = self._get_edge(gt_onehot)
#         boundary_pd = pd + pd.max() * edge_gt

#         loss = self.ce(boundary_pd, gt)

#         return loss 


####bk
# class Boundary_CE_Loss(nn.Module):
#     def __init__(self, weight=None, ignore_index=11):
#         super(Boundary_CE_Loss, self).__init__()
#         # self.ce = CrossEntropyLoss2d(weight=weight, ignore_index=ignore_index)
#         self.edge = K.filters.Canny()
#         self.kernel = torch.tensor([[0,1,0],[1,1,1],[0,1,0]])#.cuda()
#         self.weight = weight

#     def _one_hot_encoder(self, gt):
#         tensor_list = []
#         for i in range(11):
#             temp_prob = gt == i  # * torch.ones_like(input_tensor)
#             tensor_list.append(temp_prob.unsqueeze(1))
#         output_tensor = torch.cat(tensor_list, dim=1)
#         return output_tensor.float()

#     def _get_edge(self, gt_onehot):
#         edge_gt_list = []
#         for n in range(gt_onehot.shape[1]):
#             temp_edge = self.edge(gt_onehot[:,n,::].unsqueeze(1))[1]
#             temp_edge=K.morphology.dilation(temp_edge, self.kernel)
#             edge_gt_list.append(temp_edge)
#         edge_gt = torch.cat(edge_gt_list, dim=1)
#         return edge_gt.float()
        
#     def forward(self, pd, gt):
#         pd_prob = F.log_softmax(pd, dim=1)	# softmax + log

#         gt_onehot = self._one_hot_encoder(gt)
#         edge_gt = self._get_edge(gt_onehot)
#         # boundary_pd = pd + pd.max() * edge_gt
#         boundary_gt = gt_onehot + 2.0 * edge_gt * gt_onehot ##Notice:  edge_gt * gt_onehot

#         # loss = self.ce(boundary_pd, gt)
#         # loss = self.ce(pd, boundary_gt)
#         # loss = -1*torch.sum(boundary_gt*pd_prob, 1).mean()

#         loss = -1*(boundary_gt*pd_prob).sum() / (gt_onehot.sum())

#         return loss    

class Boundary_CE_Loss(nn.Module):
    def __init__(self, weight=None, ignore_index=11):
        super(Boundary_CE_Loss, self).__init__()
        # self.ce = CrossEntropyLoss2d(weight=weight, ignore_index=ignore_index)
        self.edge = K.filters.Canny()
        self.kernel = torch.tensor([[0,1,0],[1,1,1],[0,1,0]]).cuda()
        self.weight = weight.cuda()

    def _one_hot_encoder(self, gt):
        tensor_list = []
        for i in range(11):
            temp_prob = gt == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _get_edge(self, gt_onehot):
        edge_gt_list = []
        for n in range(gt_onehot.shape[1]):
            temp_edge = self.edge(gt_onehot[:,n,::].unsqueeze(1))[1]
            temp_edge=K.morphology.dilation(temp_edge, self.kernel)
            edge_gt_list.append(temp_edge)
        edge_gt = torch.cat(edge_gt_list, dim=1)
        return edge_gt.float()
        
    def forward(self, pd, gt):
        pd_prob = F.log_softmax(pd, dim=1)	# softmax + log

        gt_onehot = self._one_hot_encoder(gt)
        edge_gt = self._get_edge(gt_onehot)
        # boundary_pd = pd_prob + pd_prob.max() * edge_gt
        boundary_gt = gt_onehot + 2.0 * edge_gt * gt_onehot ##Notice:  edge_gt * gt_onehot

        # loss = self.ce(boundary_pd, gt)
        # loss = self.ce(pd, boundary_gt)
        # loss = -1*torch.sum(boundary_gt*pd_prob, 1).mean()

        # loss = -1*(boundary_gt*pd_prob).sum() / (gt_onehot.sum())
        loss = -1 * (self.weight * torch.sum(pd_prob*boundary_gt, dim=(0,2,3))).sum()/ (gt_onehot.sum())

        return loss    

###########################
if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    print(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    pd = torch.randn((1,11,128, 128))
    gt = torch.randint(0, 11, size=(1,128, 128))
    weight = torch.tensor([4.4286523, 3.443227,  9.627174,  2.9082294, 6.9883366, 5.2119217, 9.490196,
 9.150478,  7.1654587, 9.888601,  7.5455384])
    sel_loss = Boundary_CE_Loss(weight=weight)
    out_loss = sel_loss(pd, gt)
    print(out_loss)

    