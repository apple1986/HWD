import torch
import torch.nn as nn
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
from builders.most_dataset_builder import build_dataset_test
import kornia as K
from kornia import morphology as morph

###########################define hooked layer name
## extract feature map
class HookBasedFeatureExtractor(nn.Module):
    def __init__(self, submodule, layername, upscale=False):
        super(HookBasedFeatureExtractor, self).__init__()

        self.submodule = submodule
        self.submodule.eval()
        self.layername = layername
        self.outputs_size = None
        self.outputs = None
        self.inputs = None
        self.inputs_size = None
        self.upscale = upscale

    def get_input_array(self, m, i, o):
        if isinstance(i, tuple):
            self.inputs = [i[index].data.clone() for index in range(len(i))]
            self.inputs_size = [input.size() for input in self.inputs]
        else:
            self.inputs = i.data.clone()
            self.inputs_size = self.input.size()

    def get_output_array(self, m, i, o):
        if isinstance(o, tuple):
            self.outputs = [o[index].data.clone() for index in range(len(o))]
            self.outputs_size = [output.size() for output in self.outputs]
        else:
            self.outputs = o.data.clone()
            self.outputs_size = self.outputs.size()

    def rescale_output_array(self, newsize):
        us = nn.Upsample(size=newsize[2:], mode='bilinear')
        if isinstance(self.outputs, list):
            for index in range(len(self.outputs)): self.outputs[index] = us(self.outputs[index]).data()
        else:
            self.outputs = us(self.outputs).data()

    def forward(self, x):
        target_layer = self.submodule._modules['encoder']._modules.get(self.layername)

        # Collect the output tensor
        h_inp = target_layer.register_forward_hook(self.get_input_array)
        h_out = target_layer.register_forward_hook(self.get_output_array)
        self.submodule(x)
        h_inp.remove()
        h_out.remove()

        # Rescale the feature-map if it's required
        if self.upscale: self.rescale_output_array(x.size())

        return self.inputs, self.outputs

####################################### load model
root_path = "./most_dataset/"

## define all segmentation model
SMP_MODEL = [smp.Unet, smp.PAN, smp.DeepLabV3Plus,
              smp.Linknet, smp.FPN, smp.PSPNet, smp.UnetPlusPlus,]

## "-_WT" means we use Haar wavelet downsamping (HWD) to downsample feature maps
BACKBONE_NAME = [
                "resnet18_wt",
                "resnet34_wt",  
                 "resnet50_wt", 
                ]

## use the default downsampling methods, like maxpooling, averagepooling, convolution with stride 
# BACKBONE_NAME = [
#                 "resnet18",
#                 "resnet34",  
#                  "resnet50", 
#                 ]

## the layer name to output feature maps
layer_name = "down_wt" ## maxpool, avgpool, conv with stride

################################
def extract_boundary_loop(label, output, kernerl_size=7):
    label = label.to(torch.float32)
    # device = "cuda:" + args.gpus
    kernel = torch.ones(kernerl_size, kernerl_size)#.to("cuda:0")

    gt_sobel: torch.Tensor = K.filters.sobel(label)#[1] ## obtain boundary
    dilated_loop = (gt_sobel > gt_sobel.min()).to(torch.float32)
    gt_loop = dilated_loop * label
    output_loop = dilated_loop * output
    gt_loop = gt_loop.to(torch.long)

    return output_loop, gt_loop

def cal_grad(img, feat):
    pass

def norm_img(img):
    img_norm = (img - torch.min(img)) / (img.max() - img.min())
    return img_norm

def cal_entropy(model, img, gt):
    pd = model(img)
    pd = nn.functional.softmax(pd)
    mask_obj = torch.zeros(gt.shape)
    for idx in range(11):
        ## extract boundary from a mask
        gt_mask = torch.tensor([gt == idx], dtype=torch.float32)

        mask_obj += gt_mask * pd[:,idx,:,:]   
    
    return 

######################save result
fpath = "../eval_ssim_psnr_feat_img_feat"

Channel_Num = 3     
############################cal SSIM and PSNR for each model
for m in range(len(SMP_MODEL)):
    for n in range(len(BACKBONE_NAME)):
        # ### select model
        ENCODER_NAME = BACKBONE_NAME[n]
        MODEL_SEL = str(SMP_MODEL[m]).split('.')[-1][:-2] + '_' + ENCODER_NAME
        print(MODEL_SEL)
        model_path = root_path + MODEL_SEL + "bs8gpu1_trainval_ce_dice/model_200.pth"

        save_txt_name = MODEL_SEL + "_down4_infor_feat_certainty.txt"
        ## build model
        model = SMP_MODEL[m](encoder_name=ENCODER_NAME, encoder_weights=None, 
                                in_channels=Channel_Num, classes=3)

        ## load model
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        model.eval()

        #################### which feature do you want to extract
        feature_extractor = HookBasedFeatureExtractor(model, layer_name, upscale=False)

        ################################# load testing data
        N = 1 ## to contral feature size
        datas, testLoader = build_dataset_test(dataset='most_png_v3', num_workers=1)
        print(len(testLoader))
        all_ssim = []
        all_psnr = []
        all_fei = []
        feat_entropy_loop_infor = []
        for i, (input, label, size, name) in enumerate(testLoader):
            pd = model(input)
            pd = nn.functional.softmax(pd)
            print("pd:", torch.unique(pd))
            print("pd.shape", pd.shape)
            # vis.images(pd[:,1,::],win="pd")
            pd_size = nn.functional.interpolate(pd, size=(128*N, 128*N), mode='nearest', align_corners=None)
            pd_max, _ = torch.max(pd_size, dim=1)
            pd_max = pd_max.unsqueeze(1)
            print(pd_max.shape)

            print(name)
            print(input.shape)
            ## extract features
            label = label.unsqueeze(0)
            print(label.shape)
            label = label.to(torch.float32)
            label = norm_img(label)

            ###mode = nearest bilinear
            gt_down = nn.functional.interpolate(label, size=(128*N, 128*N), mode='nearest', align_corners=None)
            print(torch.unique(gt_down))
            inp_fmap, out_fmap = feature_extractor.forward(input)
            input = (input - torch.min(input)) / (input.max() - input.min())
            img = torch.mean(input, dim=1).unsqueeze(1)
            print('*'*50)
            print(img.shape)

            img_sobel_ori = K.filters.sobel(img)
            print(torch.unique(img_sobel_ori))
            print('-'*30)
            img_sobel_ori_norm = norm_img(img_sobel_ori)
            print(img_sobel_ori.shape)

            img_sobel_ori = nn.functional.interpolate(img_sobel_ori, size=(128*N, 128*N), mode='nearest', align_corners=None)
            img_sobel = norm_img(img_sobel_ori)
            
            ## resize
            img = nn.functional.interpolate(img, size=(128*N, 128*N), mode='nearest', align_corners=None)

            wt_feat1 = inp_fmap[0]
            wt_feat2 = out_fmap[0]
            wt_feat2 = wt_feat2.unsqueeze(0)

            feat_fuse_ori = torch.mean(wt_feat2, axis=1, keepdim=True)
            feat_fuse = norm_img(feat_fuse_ori)
            feat_fuse_sobel_ori = K.filters.sobel(feat_fuse_ori)

            ## boundary loop
            _, gt_edge = K.filters.canny(gt_down, low_threshold=0.1, high_threshold=0.2)
            kernel = torch.ones(5, 5)#
            gt_edge_loop = morph.dilation(gt_edge, kernel=kernel)
            print(torch.unique(gt_edge_loop))
           
            ssim_ori = k.metrics.ssim(feat_fuse_ori, img, 1)
            print(ssim_ori.shape)

            ssim = ssim_ori.mean()*100
            print('SSIM: ', ssim)
            all_ssim.append(ssim)
            psnr = torch.abs(k.metrics.psnr(feat_fuse_ori, img, 1)) 
            print('PSNR: ', psnr)
            all_psnr.append(psnr)

            ## cal FEI metric
            feat_entropy = torch.sum(-1.0 * feat_fuse_ori * torch.log10(pd_max)) #
            print('feat_entropy: ',feat_entropy)
            all_fei.append(feat_entropy.item())

            feat_entropy_loop = torch.sum(-1.0 * feat_fuse_ori*gt_edge_loop * torch.log10(pd_max)) #
            print('feat_entropy_loop: ',feat_entropy_loop)
            feat_entropy_loop_infor.append(feat_entropy_loop.item())

        ## save result
        mean_ssim = np.array(all_ssim).mean()
        mean_psnr = np.array(all_psnr).mean()
        mean_feat_entropy = np.array(all_fei).mean()
        mean_feat_entropy_loop = np.array(feat_entropy_loop_infor).mean()
        print("Mean SSIM: ", mean_ssim)
        print("Mean PSNR: ", mean_psnr)
        print("Mean feat_entrop: ", mean_feat_entropy)

        with open(os.path.join(fpath, save_txt_name), 'w') as f:
            f.write("\nMean Feat Img SSIM  : %.4f" % mean_ssim)
            f.write("\nMean Feat Img PSNR : %.4f" % mean_psnr)
            f.write("\nMean feat_entropy: %.4f" % mean_feat_entropy)
            f.write("\nMean feat_entropy_loop: %.4f" % mean_feat_entropy_loop)







