import os
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
# user
# from builders.model_builder import build_model
# from builders.most_model_builder import build_model
# from builders.med_model_builder import build_model
from builders.model_builder_nature import build_model
from builders.synapse_dataset_builder import build_dataset_test
from utils.utils import save_predict
from utils.metric.metric import get_iou
from utils.convert_state import convert_state_dict
from utils.metric import metric_all
from utils.utils import setup_seed, init_weight, netParams, test_single_volume
import torch.nn as nn
import logging

import segmentation_models_pytorch as smp

MODELS = ['UNet',  
        "HighResolutionNet_OCR","HighResolutionNet", "HighResolutionNet_WT",
            "FCN32s", "FCN32s_WT", "FCN16s", "FCN16s_WT", "FCN8s", "FCN8s_WT",
            'SegNet', "SegNet_WT", "DeepLabV3", 
             "FCN32s", "UNet_WT_Avg",
            "UNet_WT_SR", "UNet_WT_LH","WTNet","UNet_WT_Max","UNet_WT","HighResolutionNet","ConvNeXt_UNet_T",
            "UNet_SR","UNet_Boundary","UNet_Gauss", "NoGaussNN","GaussResNet","GaussNN",
            "FastSCNN_SR",'W_UNet','UNet_Fuse',
            "FastSCNN_DWT","FastSCNN_Fuse","FastSCNN",  "NF_UNet_v2", "NF_UNet_v1",
            "LGNet","LGNet_Com", 
            "DeepLabV3Plus","BiSeNet","FastAttSegHR","FastAttSeg",
            'SegNet', 'SEUNet',
            'OutlookUNet','PSAUNet', 'SAUNet','A2UNet',
            'SGEUNet', 'SKUNet','ECAUNet', 'CAUNet', 
            'CAUNet_v1','CoTUNet','SE3UNet', 'SEUNetX',
            'SE3UNetX','SAUNetX','DDRNet_23_slim','DDRNet_23',
            'DDRNet_39']
NUM = 1
GPU_NUM = "0"

def parse_args():
    parser = ArgumentParser(description='Efficient semantic segmentation')
    parser.add_argument('--model', default=MODELS[NUM], help="model name: (default ENet)")
    parser.add_argument('--dataset', default="synapse", help="dataset: cityscapes or camvid")
    parser.add_argument('--num_workers', type=int, default=1, help="the number of parallel threads")
    parser.add_argument('--batch_size', type=int, default=1,
                        help=" the batch_size is set to 1 when evaluating or testing")
    parser.add_argument('--checkpoint', type=str,
                        default="/home/gpxu/vess_seg/vess_efficient/checkpoint/synapse/"+MODELS[NUM]+"_bs8gpu1_trainvalce_dice/model_350.pth",
                        help="use the file to load the checkpoint for evaluating or testing ")
    parser.add_argument('--save_seg_dir', type=str, default="./result/",
                        help="saving path of prediction result")
    parser.add_argument('--best', default=False, action='store_true', help="Get the best result among last few checkpoints")
    parser.add_argument('--save', default=True, action='store_true', help="Save the predicted image")
    parser.add_argument('--cuda', default=True, help="run on CPU or GPU")
    parser.add_argument("--gpus", default=GPU_NUM, type=str, help="gpu ids (default: 0)")
    args = parser.parse_args()

    return args



## input image and ouput segmentation result
def test(args, test_loader, model):
    """
    args:
      test_loader: loaded for test dataset
      model: model
    return: class IoU and mean IoU
    """
    # evaluation or test mode
    model.eval()
    total_batches = len(test_loader)
    logging.info("{} test iterations per epoch".format(len(test_loader)))

    data_list = []
    eval_result = []
    time_list = []
    metric_list = 0.0
    for i, (image, label, size, name) in enumerate(test_loader):
        print('subject: ', i)
        #######testing
        h, w = size[:,2], size[:,3]
        image = image.squeeze(0)
        metric_i = test_single_volume(image, label, model, classes=9, patch_size=[224, 224],
                                      #test_save_path='/home/gpxu/vess_seg/vess_efficient/result/synapse/'+args.model, 
                                      case=name, z_spacing=1)
        print('Dice and HD95:', metric_i)
        metric_list += np.array(metric_i)

        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i, name[0], np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    print('total metric list:', metric_list)
    metric_list = metric_list / 12.0
    for i in range(1, 9):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    mean_dice = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (mean_dice, mean_hd95))
            


    return  metric_list, mean_dice, mean_hd95


def test_model(args, model):
    """
     main function for testing
     param args: global arguments
     return: None
    """
    print(args)

    if args.cuda:
        print("=====> use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("no GPU found or wrong gpu id, please run without --cuda")

    # build the model
    # model = build_model(args.model, num_classes=args.classes) # ap: for levit
    # model = build_model(args.model, num_classes=args.classes, num_channel=1)
    init_weight(model, nn.init.kaiming_normal_,
            nn.BatchNorm2d, 1e-3, 0.1,
            mode='fan_in')
    
    # model = smp.DeepLabV3Plus(
    #     encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    #     encoder_weights=None, #"imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    #     in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    #     classes=9,                      # model output channels (number of classes in your dataset)
    #     )


    if args.cuda:
        model = model.cuda()  # using GPU for inference
        cudnn.benchmark = True

    if args.save:
        if not os.path.exists(args.save_seg_dir):
            os.makedirs(args.save_seg_dir)

    # load the test set
    # datas, testLoader = build_dataset_test(args.dataset, args.num_workers)
    datas, testLoader = build_dataset_test(args.dataset, args.num_workers, none_gt=False)

    if not args.best:
        if args.checkpoint:
            if os.path.isfile(args.checkpoint):
                print("=====> loading checkpoint '{}'".format(args.checkpoint))
                checkpoint = torch.load(args.checkpoint)
                model.load_state_dict(checkpoint['model'])
                # model.load_state_dict(convert_state_dict(checkpoint['model']))
            else:
                print("=====> no checkpoint found at '{}'".format(args.checkpoint))
                raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))

        print("=====> beginning validation")
        print("validation set length: ", len(testLoader))
        # mIOU_val, per_class_iu = test(args, testLoader, model)
        metric_list, mean_dice, mean_hd95 = test(args, testLoader, model)
        print(metric_list)
        print(mean_dice)
        print(mean_hd95)

    # Get the best test result among the last 10 model records.
    else:
        if args.checkpoint:
            if os.path.isfile(args.checkpoint):
                dirname, basename = os.path.split(args.checkpoint)
                epoch = int(os.path.splitext(basename)[0].split('_')[1])
                mIOU_val = []
                per_class_iu = []
                for i in range(epoch - 9, epoch + 1):
                    print('Epoch: ', i)
                    basename = 'model_' + str(i) + '.pth'
                    resume = os.path.join(dirname, basename)
                    checkpoint = torch.load(resume)
                    model.load_state_dict(checkpoint['model'])
                    print("=====> beginning test the " + basename)
                    print("validation set length: ", len(testLoader))
                    mIOU_val_0, per_class_iu_0 = test(args, testLoader, model)
                    mIOU_val.append(mIOU_val_0)
                    per_class_iu.append(per_class_iu_0)

                index = list(range(epoch - 9, epoch + 1))[np.argmax(mIOU_val)]
                print("The best mIoU among the last 10 models is", index)
                print(mIOU_val)
                per_class_iu = per_class_iu[np.argmax(mIOU_val)]
                mIOU_val = np.max(mIOU_val)
                print(mIOU_val)
                print(per_class_iu)

            else:
                print("=====> no checkpoint found at '{}'".format(args.checkpoint))
                raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))

    # Save the result
    if not args.best:
        model_path = os.path.splitext(os.path.basename(args.checkpoint))
        args.logFile = 'test_' + model_path[0] + '.txt'
        logFileLoc = os.path.join(os.path.dirname(args.checkpoint), args.logFile)
    else:
        args.logFile = 'test_' + 'best' + str(index) + '.txt'
        logFileLoc = os.path.join(os.path.dirname(args.checkpoint), args.logFile)

    # Save the result
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a+')
        logger.write("Mean Dice: %.4f\t" % mean_dice)
        logger.write("Mean HD95: %.4f" % mean_hd95)
        logger.write("\nPer class Performance: \n")
        # logger.write("Mean List: " , metric_list)
        for i in range(len(metric_list)):
            logger.write("%.4f\t" % metric_list[i][0])
            logger.write("%.4f\n" % metric_list[i][1])
    else:
        logger = open(logFileLoc, 'a+')
        logger.write("Mean Dice: %.4f\t" % mean_dice)
        logger.write("Mean HD95: %.4f" % mean_hd95)
        logger.write("\nPer class Performance: \n")
        # logger.write("Mean List: " , metric_list)
        for i in range(len(metric_list)):
            logger.write("%.4f\t" % metric_list[i][0])
            logger.write("%.4f\n" % metric_list[i][1])
    logger.flush()
    logger.close()


if __name__ == '__main__':

    args = parse_args()

    args.save_seg_dir = os.path.join(args.save_seg_dir, args.dataset, args.model)

    if args.dataset == 'cityscapes':
        args.classes = 19
    elif args.dataset == 'synapse':
        args.classes = 9
        args.input_size = '224,224'
        ignore_label = 11
    else:
        raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, %s is not included" % args.dataset)
    # SMP_MODEL = [smp.UnetPlusPlus]#smp.FPN,  smp.PSPNet,# smp.DeepLabV3Plus, smp.Unet,  smp.PSPNet, smp.Unet, smp.Linknet, smp.PAN, smp.DeepLabV3Plus,
    #                  #smp.PAN, smp.Linknet,]# smp.DeepLabV3,  smp.PAN, smp.Linknet,smp.Unet, smp.FPN, smp.PSPNet, smp.UnetPlusPlus, 
    # SMP_MODEL = [smp.Unet, smp.DeepLabV3Plus, smp.FPN, smp.PSPNet,# smp.PSPNet, smp.Unet, smp.Linknet, smp.PAN, smp.DeepLabV3Plus,
    #                  smp.PAN, smp.Linknet,smp.UnetPlusPlus]
    # SMP_MODEL = [smp.Unet,  smp.FPN, smp.PSPNet,# smp.PSPNet, smp.Unet, smp.Linknet, smp.PAN, smp.DeepLabV3Plus,
    #                   smp.Linknet,smp.UnetPlusPlus]
    SMP_MODEL = [smp.Unet,  smp.PAN]
    # SMP_MODEL = [smp.Unet, smp.FPN]
    # BACKBONE_NAME = [ "resnet50_wt", "resnet101_wt"]#] # "resnet18_wt","resnet34_wt",
    
    # done: smp.Unet,smp.DeepLabV3Plus, smp.PAN,
    # SMP_MODEL = [smp.FPN, smp.PSPNet, smp.UnetPlusPlus,smp.Linknet,] # smp.DeepLabV3,   
    # SMP_MODEL = [smp.PAN, smp.Linknet, smp.DeepLabV3Plus, smp.UnetPlusPlus, smp.Unet, smp.FPN]#smp.PAN, smp.DeepLabV3, smp.Linknet, smp.DeepLabV3Plus] ## Set diconv True
    # BACKBONE_NAME = ["xception_wt", "xception"]#] # 'mobilenet_v2'

    # BACKBONE_NAME = [
    #                 "resnet34_wt", 
    #                  "resnet18_wt",  
    #                  "resnet50_wt", 
    #                    ] # "resnet101_wt",

    # BACKBONE_NAME = [ "resnet50",
    #                     "resnet34", 
    #                     "resnet18", 
    #                   ] # "resnet101_wt",

    # BACKBONE_NAME = ["resnet50" 
    #              ] # "resnet101_wt",

    GPU_NUM = "1"
    args.gpus = GPU_NUM

    # NUM_CHANNEL = 3
    # for name in BACKBONE_NAME:
    #     print(name)
    #     if "_wt" in name:
    #         BACKBONE_NAME = ["mobilenet_v2_wt"] 
    #         NUM_CHANNEL = 3
    #     else:
    #         BACKBONE_NAME = ["mobilenet_v2"]
    #         NUM_CHANNEL = 1

    if GPU_NUM == "0":
        # BACKBONE_NAME = ["resnet34_wt", 
        #              "resnet18_wt",   
        #              "resnet50_wt",  ] # "resnet101_wt", mobilenet_v2
        BACKBONE_NAME = ["resnet50_wt"]  #mobilenet_v2_wt        xception_wt
        NUM_CHANNEL = 3   
    else:
        # BACKBONE_NAME = ["resnet34",  
        #               "resnet18", 
        #              "resnet50", ] # "resnet101_wt",   
        BACKBONE_NAME = ["resnet50"]
        NUM_CHANNEL = 1


    for m in range(len(SMP_MODEL)):
        for n in range(len(BACKBONE_NAME)):
            sel_model = str(SMP_MODEL[m].__name__) + '_' + BACKBONE_NAME[n]
            # print(sel_model)
            args.model = sel_model
            args.checkpoint = "/home/gpxu/vess_seg/vess_efficient/checkpoint/synapse_SMP/"+ \
                                sel_model+"_bs8gpu1_trainvalce_dice"+"/model_350.pth"
            print(args.model)
            model = SMP_MODEL[m](
                encoder_name=BACKBONE_NAME[n],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=NUM_CHANNEL,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=9,                      # model output channels (number of classes in your dataset)
                )

            test_model(args, model)

    # test_model(args)
