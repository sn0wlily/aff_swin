import torch
import torch.nn.functional as F
import torch.nn as nn

from utils.registery import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class ExprLoss(nn.Module):
    def __init__(self, data_weight, config_weight):
        super().__init__()
        w = torch.FloatTensor(config_weight)
        print("loss_weight: "+str(w))
        self.ce = nn.CrossEntropyLoss(weight=w)

    def forward(self, x, y):
        return self.ce(x, y)


@LOSS_REGISTRY.register()
class RDropLoss(nn.Module):
    def __init__(self, weight, alpha=5):
        super().__init__()
        w = torch.FloatTensor(weight)
        self.ce = nn.CrossEntropyLoss(weight=w, reduction='none')
        self.kld = nn.KLDivLoss(reduction='none')
        self.alpha = alpha

    def forward(self, logits1, logits2, gt):
        ce_loss = (self.ce(logits1, gt) + self.ce(logits2, gt)) / 2
        kl_loss1 = self.kld(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1)).sum(-1)
        kl_loss2 = self.kld(F.log_softmax(logits2, dim=-1), F.softmax(logits1, dim=-1)).sum(-1)
        kl_loss = (kl_loss1 + kl_loss2) / 2
        loss = ce_loss + self.alpha * kl_loss

        loss = loss.mean(-1)

        return loss
