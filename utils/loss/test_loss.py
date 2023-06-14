import torch
import torch.nn as nn
import torch.nn.functional as F


class LSR(nn.Module):
    def __init__(self, n_classes=10, eps=0.1):
        super(LSR, self).__init__()
        self.n_classes = n_classes
        self.eps = eps

    def forward(self, outputs, labels):
        # labels.shape: [b,]
        assert outputs.size(0) == labels.size(0)
        n_classes = self.n_classes
        one_hot = F.one_hot(labels, n_classes).float()
        mask = ~(one_hot > 0)
        smooth_labels = torch.masked_fill(one_hot, mask, eps / (n_classes - 1))
        smooth_labels = torch.masked_fill(smooth_labels, ~mask, 1 - eps)
        ce_loss = torch.sum(-smooth_labels * F.log_softmax(outputs, 1), dim=1).mean()
        # ce_loss = F.nll_loss(F.log_softmax(outputs, 1), labels, reduction='mean')
        return ce_loss


if __name__ == '__main__':
    labels = [0, 1, 2, 1, 1, 0, 3]
    labels = torch.tensor(labels)
    eps = 0.1
    n_classes = 4
    outputs = torch.rand([7, 4])
    print(outputs)

    LL = LSR(n_classes, eps)
    LL2 = nn.CrossEntropyLoss()
    loss = LL.forward(outputs, labels)
    loss2 = LL2.forward(outputs, labels)
    print(loss)
    print(loss2)

    # # 2D loss example (used, for example, with image inputs)
    # N, C = 5, 4
    # loss = nn.NLLLoss()
    # # input is of size N x C x height x width
    # data = torch.randn(N, 16, 10, 10)
    # conv = nn.Conv2d(16, C, (3, 3))
    # m = nn.LogSoftmax(dim=1)
    # # each element in target has to have 0 <= value < C
    # target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
    # print(m(conv(data)).shape)
    # print(target.shape)
    # print(torch.unique(target))
    # output = loss(m(conv(data)), target)
    # output.backward()
    # print(output)
