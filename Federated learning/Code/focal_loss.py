from torch import nn
import torch
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.76, gamma=2, num_classes=2, size_average=True):
        """
        focal_loss, -α(1-yi)**γ *ce_loss(xi,yi)
        This step implements the focal_loss function in detail.
        :param alpha:   Alpha α, category weight. When α is a list, is the category weight,
                        and when α is a constant, the category weight is[α, 1-α, 1-α, ....]
        :param gamma:   Gamma γ, hard and easy sample adjustment parameters. Set to 2 in retainnet
        :param num_classes:
        :param size_average: Loss calculation method, default to take the mean
        """
        super(FocalLoss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes
            # print("Focal_loss alpha = {},".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   # If α is a constant, the effect of the
                             # first category is reduced, which in object detection is the first category
            # print(" --- Focal_loss alpha = {}".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha)

        self.gamma = gamma

    def forward(self, inputs, labels):
        """
        focal_loss
        :param preds:   Category of prediction. size:[B,N,C] or [B,C]
                        Corresponding to and detection and classification tasks, B batch, N number of detection boxes, C number of categories
        :param labels:  Actual categories. size:[B,N] or [B]
        :return:
        """
        preds_softmax = F.softmax(inputs,dim=-1)
        self.alpha = self.alpha.to(preds_softmax.device)
        preds_softmax = preds_softmax.view(-1,preds_softmax.size(-1))
        preds_logsoft = torch.log(preds_softmax)

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) :focal loss中 (1-pt)**γ

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
