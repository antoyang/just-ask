import torch as torch
import torch.nn.functional as F


class Contrastive_Loss(torch.nn.Module):
    def __init__(self):
        super(Contrastive_Loss, self).__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, target):
        return self.ce_loss(x, target)


class LogSoftmax(torch.nn.Module):
    def __init__(self, dim):
        super(LogSoftmax, self).__init__()
        self.dim = dim

    def forward(self, x, a):
        nll = -F.log_softmax(x, self.dim, _stacklevel=5)
        return (nll * a / a.sum(1, keepdim=True).clamp(min=1)).sum(dim=1).mean()


class NCELoss(torch.nn.Module):
    def __init__(self, batch_size=4096):
        super(NCELoss, self).__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        batch_size = len(x)
        target = torch.arange(batch_size).cuda()
        x = torch.cat((x, x.t()), dim=1)
        return self.ce_loss(x, target)
