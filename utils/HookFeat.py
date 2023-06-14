import torch.nn as nn

class HookFeat(nn.Module):
    def __init__(self, submodule, layername, upscale=False):
        super(HookFeat, self).__init__()

        self.submodule = submodule
        self.submodule.eval()
        self.layername = layername
        self.outputs = None
        self.inputs = None
        self.upscale = upscale


    def rescale_output_array(self, newsize):
        us = nn.Upsample(size=newsize[2:], mode='bilinear', align_corners=True)
        if isinstance(self.outputs, list):
            for index in range(len(self.outputs)): self.outputs[index] = us(self.outputs[index]).data()
        else:
            self.outputs = us(self.outputs)#.data()

    def get_features_hook(self,m, input, output):
        print("input shape: ", input[0].data.cpu().numpy().shape)
        print("output shape: ", output.data.cpu().numpy().shape)
        self.inputs = input
        self.outputs = output

    def forward(self, x):
        # target_layer = self.submodule._modules.get(self.layername)
        # target_layer = self.submodule._modules['ResNet']._modules.get(self.layername)

        for name, target_layer in self.submodule.named_modules():
            print(name)
            if name == self.layername:
                h_inp = target_layer.register_forward_hook(self.get_features_hook)

        self.submodule(x)
        h_inp.remove()
        # h_out.remove()

        # Rescale the feature-map if it's required
        if self.upscale: self.rescale_output_array(x.size())

        return self.inputs, self.outputs


if __name__ == '__main__':
    import torchvision
    import torch

    res18=torchvision.models.resnet18()
    mod = HookFeat(res18, "conv1", upscale=True)
    img = torch.ones([1,3,224,224])
    fin, fout = mod(img)