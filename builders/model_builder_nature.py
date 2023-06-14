import torch
from model.LinkNet import LinkNet
from model.SegNet import SegNet

from model import config
from model.hrnet_seg import HighResolutionNet
from model.HRNet_WT import HighResolutionNet_WT
from model.HRNet_OCR import HighResolutionNet_OCR

from model.fcn32s import FCN32s
from model.fcn16s import FCN16s
from model.fcn8s import FCN8s

from model.fcn32s_wt import FCN32s_WT
from model.fcn16s_wt import FCN16s_WT
from model.fcn8s_wt import FCN8s_WT

from model.SegNet_WT import SegNet_WT

from model.DeepLabV3 import DeepLabV3

from model.UNext import UNext, UNext_S

from model.UNext_WT import UNext_WT
from model.Swin_Unet import SwinTransformerSys
from model.Swin_Unet_WT import SwinTransformerSys_WT
from networks.segformer import SegFormer, SegFormer_WT
from model.convnext_unet_wt import ConvNeXt_UNet_T_WT
from model.convnext_unet import ConvNeXt_UNet_T



def build_model(model_name, num_classes, num_channel=3):
    if model_name == 'LinkNet':
        return LinkNet(classes=num_classes)
    elif model_name == 'SegNet':
        return SegNet(classes=num_classes, channels=num_channel)
    elif model_name == 'SegNet_WT':
        return SegNet_WT(classes=num_classes, channels=num_channel)
    
    elif model_name == 'UNext_WT':
        return UNext_WT(input_channels=num_channel, num_classes=num_classes) 
    elif model_name == 'Swin_Unet':
        return SwinTransformerSys(in_chans=num_channel, num_classes=num_classes) 
    elif model_name == 'Swin_Unet_WT':
        return SwinTransformerSys_WT(in_chans=num_channel, num_classes=num_classes) 
    elif model_name == 'SegFormer':
        return SegFormer(num_classes=num_classes) 
    elif model_name == 'SegFormer_WT':
        return SegFormer_WT(num_classes=num_classes)     

    elif model_name == 'DeepLabV3':
        return DeepLabV3(classes=num_classes, channels=num_channel)

    elif model_name == 'FCN32s':
        return FCN32s(classes=num_classes, channels=num_channel)
    elif model_name == 'FCN16s':
        return FCN16s(classes=num_classes, channels=num_channel)
    elif model_name == 'FCN8s':
        return FCN8s(classes=num_classes, channels=num_channel)

    elif model_name == 'FCN32s_WT':
        return FCN32s_WT(classes=num_classes, channels=num_channel)
    elif model_name == 'FCN16s_WT':
        return FCN16s_WT(classes=num_classes, channels=num_channel)
    elif model_name == 'FCN8s_WT':
        return FCN8s_WT(classes=num_classes, channels=num_channel)



    elif model_name == 'HighResolutionNet':
        return HighResolutionNet(config, classes=num_classes, channels=num_channel)
    elif model_name == 'HighResolutionNet_WT':
        return HighResolutionNet_WT(config, classes=num_classes, channels=num_channel)
    elif model_name == 'HighResolutionNet_OCR':
        return HighResolutionNet_OCR(config, classes=num_classes, channels=num_channel)

    elif model_name == 'ConvNeXt_UNet_T':
        return ConvNeXt_UNet_T(classes=num_classes, channels=num_channel)
    elif model_name == 'ConvNeXt_UNet_T_WT':
        return ConvNeXt_UNet_T_WT(classes=num_classes, channels=num_channel)

    elif model_name == 'UNext':
        return UNext(input_channels=num_channel, num_classes=num_classes) 
    elif model_name == 'UNext_S':
        return UNext_S(input_channels=num_channel, num_classes=num_classes) 
                                
    
                                                            
                                                        
                             
                                                          