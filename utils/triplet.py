from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable

#this is trihard loss
class TripletLoss(nn.Module):
    def __init__(self, margin=0, num_instances=None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        # [[a, a, a], [b, b, b], [c, c, c]]
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        # [[a + a, a + b, a + c], [b + a, b + b, b + c], [c + a, c + b, c + c]]
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        # positive is the same type of anchor
        # negative is different
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        # y == 1 indicates the first input should be ranked higher (have a larger value) 
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        # prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        #for n examples if dist_an > dist_ap
        #we should say the example is classified correctly
        correct = (dist_an.data > dist_ap.data).sum()
        dist_p = torch.mean(dist_ap).item()
        dist_n = torch.mean(dist_an).item()
        return loss, correct, dist_p, dist_n
        # prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        # dist_p = torch.mean(dist_ap).item()
        # dist_n = torch.mean(dist_an).item()
        # return loss, prec, dist_p, dist_n
