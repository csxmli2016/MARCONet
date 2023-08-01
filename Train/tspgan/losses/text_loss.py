
import math
import torch
import lpips
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY


def convert_encode(gt_labels=None, for_ctc=True):
    length = []
    result = []
    if for_ctc:
        for i in range(gt_labels.size(0)):
            n = 0
            for item in gt_labels[i]:
                if item < 6735:
                    n += 1
                    result.append(item)
            length.append(n)
        
        return (torch.IntTensor(result), torch.IntTensor(length))
    else:
        for i in range(gt_labels.size(0)):
            for item in gt_labels[i]:
                result.append(item)

        return torch.IntTensor(result)

@LOSS_REGISTRY.register() 
class TextCELoss(nn.Module):
    def __init__(self, loss_weight=1.0, num_cls=6736, **kwargs):
        super(TextCELoss, self).__init__()
        self.loss_weight = loss_weight

        empty_weight = torch.ones(num_cls)
        empty_weight[-1] = 0.1 # default from DETR
        self.register_buffer('empty_weight', empty_weight)


    def forward(self, pred, gt):
        '''
        pred is N*ObjNum*C
        gt is N*ObjNum(16)
        '''
        pred_m = pred.transpose(1,2)
        # target_m = convert_encode(gt, False).to(pred_m)
        target_m = gt
        loss_ce = F.cross_entropy(pred_m, target_m.long(), self.empty_weight) * self.loss_weight
        return loss_ce


@LOSS_REGISTRY.register()
class CTCLoss(nn.Module):
    def __init__(self, loss_weight=1.0, blank=6735, reduction='mean', **kwargs):
        super(CTCLoss, self).__init__()
        self.loss_weight = loss_weight
        self.cri = torch.nn.CTCLoss(blank=blank, reduction=reduction)
    def forward(self, pred, gt):
        '''
        input is N*T*Cls should be softmax
        gt is N*S
        '''
        pred_t = pred.permute(1,0,2)
        pred_s = F.log_softmax(pred_t, dim=2)

        batch_size = pred.size(0)
        targets, target_lengths = convert_encode(gt, True)
        input_lengths = torch.IntTensor([pred_s.size(0)] * batch_size) # timestep * batchsize
        loss = self.cri(pred_s, targets, input_lengths, target_lengths) * self.loss_weight
        return loss



@LOSS_REGISTRY.register()
class LPIPSLossF(nn.Module):
    """ LPIPS loss.
    Args:
        loss_weight (float): Loss weight for LPIPS loss. Default: 1.0
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean'. Default: 'mean'.
    """
    
    def __init__(self, loss_weight=1.0, net='vgg', reduction='mean'):
        super().__init__()
        self.loss_weight = loss_weight
        self.lpips_model = lpips.LPIPS(net=net)
        self.reduction = reduction

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            Warning: both pred and target should be normalized to [-1, 1].
        """
        loss = self.lpips_model(pred, target)  # [N, 1, 1, 1]
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        else:
            raise ValueError()