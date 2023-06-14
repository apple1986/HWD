import os.path as osp

import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward

from .fcn32s import get_upsampling_weight
# from fcn32s import get_upsampling_weight



class down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.bn_conv_relu = nn.Sequential(nn.BatchNorm2d(in_ch*4),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_ch*4, out_ch, kernel_size=1),                                    
                                    ) 

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)        
        x = self.bn_conv_relu(x)

        return x

class FCN16s_WT(nn.Module):

#    pretrained_model = \
#        osp.expanduser('~/data/models/pytorch/fcn16s_wt_from_caffe.pth')
#
#    @classmethod
#    def download(cls):
#        return fcn.data.cached_download(
#            url='http://drive.google.com/uc?id=0B9P1L--7Wd2vVGE3TkRMbWlNRms',
#            path=cls.pretrained_model,
#            md5='991ea45d30d632a01e5ec48002cac617',
#        )

    def __init__(self, classes=9, channels=1):
        super(FCN16s_WT, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(channels, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        # self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2
        self.pool1 = down_wt(64, 64)

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        # self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4
        self.pool2 = down_wt(128, 128)

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        # self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8
        self.pool3 = down_wt(256, 256)

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        # self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16
        self.pool4 = down_wt(512, 512)

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        # self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32
        self.pool5 = down_wt(512, 512)

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, classes, 1)
        self.score_pool4 = nn.Conv2d(512, classes, 1)

        self.upscore2 = nn.ConvTranspose2d(
            classes, classes, 4, stride=2, bias=False)
        self.upscore16 = nn.ConvTranspose2d(
            classes, classes, 32, stride=16, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c

        h = self.upscore16(h)
        h = h[:, :, 27:27 + x.size()[2], 27:27 + x.size()[3]].contiguous()

        return h

    def copy_params_from_fcn32s(self, fcn32s):
        for name, l1 in fcn32s.named_children():
            try:
                l2 = getattr(self, name)
                l2.weight  # skip ReLU / Dropout
            except Exception:
                continue
            assert l1.weight.size() == l2.weight.size()
            assert l1.bias.size() == l2.bias.size()
            l2.weight.data.copy_(l1.weight.data)
            l2.bias.data.copy_(l1.bias.data)

if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = FCN16s_WT(classes=9, channels=1).to(device)
    model.eval()
    img = torch.randn(1, 1, 256, 256).to(device)
    out = model(img)
    print(out.shape)
    # summary(model,(3,512,1024))