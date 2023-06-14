import os
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
# user
from builders.model_builder_nature import build_model
# from builders.med_model_builder import build_model
from builders.dataset_builder import build_dataset_test
from utils.utils import save_predict
from utils.metric.metric import get_iou_v2
from utils.convert_state import convert_state_dict
from utils.utils import setup_seed, init_weight
import torch.nn as nn
import segmentation_models_pytorch as smp


MODELS = ["SMP_UNetPlusPlus_Res50_WT", "SMP_UNetPlusPlus_Res50",
    "SMP_FPN_Res50_WT", "SMP_FPN_Res50", 
        "SMP_UNet_Res50_WT", "SMP_UNet_Res50",
    "HighResolutionNet_WT","HighResolutionNet",
    "SMP_UNetPlusPlus_Res34_WT", "SMP_UNetPlusPlus_Res34",
    "SMP_Deeplabv3plus_Res18_WT", "SMP_Deeplabv3plus_Res18",
    "SMP_PSPNet_Res18_WT", "SMP_PSPNet_Res18",  
    "SMP_FPN_Res18_WT", "SMP_FPN_Res18",
    "SMP_UNet_Res18_WT", "SMP_UNet_Res18",
    
    "SMP_PSPNet_Res50_WT", "SMP_PSPNet_Res50", 
    
    "SMP_UNet_Res50_WT", "SMP_UNet_Res50",
    "FCN8s_WT","HighResolutionNet_OCR","SegNet_WT", "DeepLabV3","FCN16s_WT", "FCN32s_WT",
            "FCN8s", "FCN16s","FCN32s", "UNet_WT_Avg",
            "UNet_WT_SR", "UNet_WT_LH","WTNet","UNet_WT_Max","UNet_WT","HighResolutionNet","ConvNeXt_UNet_T",
            "UNet_SR","UNet_Boundary","UNet_Gauss", "NoGaussNN","GaussResNet","GaussNN",
            "FastSCNN_SR",'W_UNet','UNet_Fuse',
            "FastSCNN_DWT","FastSCNN_Fuse","FastSCNN", 'UNet',  "NF_UNet_v2", "NF_UNet_v1",
            "LGNet","LGNet_Com", "FastSCNN_NF",
            "DDRAttNet_23s","FastAttSegHR","FastAttSeg_Multi","UnetPlusPlus_res50_scse",
            "UnetPlusPlus_res101","UnetPlusPlus_Eff_B7","PAN","PSPNet",
            "DeepLabV3Plus","BiSeNet","FastAttSegHR","FastAttSeg",
            'SegNet', 'SEUNet',
            'OutlookUNet','PSAUNet', 'SAUNet','A2UNet',
            'SGEUNet', 'SKUNet','ECAUNet', 'CAUNet', 
            'CAUNet_v1','CoTUNet','SE3UNet', 'SEUNetX',
            'SE3UNetX','SAUNetX','DDRNet_23_slim','DDRNet_23',
            'DDRNet_39']
NUM = 3
BACKBONE_NAME = ["resnet50_wt", "resnet50","resnet34_wt", "resnet34","resnet18_wt", "resnet18",]
BK_NUM = 1

SAVE_FLAG = ["ImgAug", "ImgAug_Canny","No_Aug","Aug_Scale_Person", "Ori_Aug", "Ori_Aug_plus_Scale",
      "ImgAug_NoWeight", "ImgAug_CopyObj", "ImgAug_CopyRect",
     "ImgAug_CopyObjScale","ImgAug_CopyObjScaleRand",]
SEL_NUM = 0

GPU_NUM = "0"

def parse_args():
    parser = ArgumentParser(description='Efficient semantic segmentation')
    parser.add_argument('--model', default=MODELS[NUM], help="model name: (default ENet)")
    parser.add_argument('--dataset', default="camvid", help="dataset: cityscapes or camvid")
    parser.add_argument('--num_workers', type=int, default=1, help="the number of parallel threads")
    parser.add_argument('--batch_size', type=int, default=1,
                        help=" the batch_size is set to 1 when evaluating or testing")
    parser.add_argument('--checkpoint', type=str,default="/home/gpxu/vess_seg/vess_efficient/checkpoint/camvid/"+
            MODELS[NUM]+"bs8gpu1_trainval_"+SAVE_FLAG[SEL_NUM]+"/model_350.pth",
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

    total_time = []
    data_list = []
    for i, (input, label, size, name) in enumerate(test_loader):
        with torch.no_grad():
            input_var = input.cuda()
        torch.cuda.synchronize()
        start_time = time.time()
        output = model(input_var)
        time_taken = time.time() - start_time
        total_time.append(time_taken*1000) # ms
        print('[%d/%d]  time: %.2f' % (i + 1, total_batches, time_taken*1000))
        output = output.cpu().data[0].numpy()
        gt = np.asarray(label[0].numpy(), dtype=np.uint8)
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        data_list.append([gt.flatten(), output.flatten()])

        # save the predicted image
        if args.save:
            save_predict(output, gt, name[0], args.dataset, args.save_seg_dir,
                         output_grey=False, output_color=True, gt_color=True)

    meanIoU, per_class_iu = get_iou_v2(data_list, args.classes)
    meanPredTime = np.mean(np.array(total_time[1:]).sum() / (total_batches-1)) ### ignore the first sample
    print(np.array(total_time[1:]).shape)
    print(meanPredTime)
    return meanIoU, per_class_iu, meanPredTime


def test_model(args, Seg_Model, Backbone):
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

    # # build the model
    # model = build_model(args.model, num_classes=args.classes,num_channel=3)
    # init_weight(model, nn.init.kaiming_normal_,
    #     nn.BatchNorm2d, 1e-3, 0.1,
    #     mode='fan_in')

    # ### use another package: https://github.com/qubvel/segmentation_models.pytorch#installation
    model = Seg_Model(
        encoder_name=Backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=11,                      # model output channels (number of classes in your dataset)
        )
    init_weight(model, nn.init.kaiming_normal_,
    nn.BatchNorm2d, 1e-3, 0.1,
    mode='fan_in')

    if args.cuda:
        model = model.cuda()  # using GPU for inference
        cudnn.benchmark = True

    if args.save:
        if not os.path.exists(args.save_seg_dir):
            os.makedirs(args.save_seg_dir)

    # load the test set
    datas, testLoader = build_dataset_test(args.dataset, args.num_workers)

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
        mIOU_val, per_class_iu, meanPredTime = test(args, testLoader, model)
        print(mIOU_val)
        print(per_class_iu)

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
    print("done")
    # if os.path.isfile(logFileLoc):
    #     logger = open(logFileLoc, 'w')
    #     logger.write("\nMean Prediction Time (ms): %.4f" % meanPredTime)
    #     logger.write("\nMean IoU: %.4f" % mIOU_val)
    #     logger.write("\nPer class IoU: ")
    #     for i in range(len(per_class_iu)):
    #         logger.write("%.4f\t" % per_class_iu[i])        
    # else:
    #     logger = open(logFileLoc, 'w')
    #     logger.write("\nMean Prediction Time (ms): %.4f" % meanPredTime)
    #     logger.write("\nMean IoU: %.4f" % mIOU_val)
    #     logger.write("\nPer class IoU: ")
    #     for i in range(len(per_class_iu)):
    #         logger.write("%.4f\t" % per_class_iu[i])
    # logger.flush()
    # logger.close()


if __name__ == '__main__':

    args = parse_args()

    # args.save_seg_dir = os.path.join(args.save_seg_dir, args.dataset, args.model)

    if args.dataset == 'cityscapes':
        args.classes = 19
    elif args.dataset == 'camvid':
        args.classes = 11
    else:
        raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, %s is not included" % args.dataset)

    # SMP_MODEL = [ smp.PSPNet, smp.FPN, smp.Unet, smp.PAN]#, smp.UnetPlusPlus, smp.Linknet, smp.DeepLabV3, smp.DeepLabV3Plus]
    # SMP_MODEL = [smp.PAN, smp.UnetPlusPlus, smp.Linknet, smp.DeepLabV3Plus]
    # BACKBONE_NAME = ["resnet18", "resnet34", "resnet50", "resnet101"]

    ## Ori
    SMP_MODEL = [smp.Unet, smp.Linknet, smp.PSPNet, smp.PAN, smp.DeepLabV3Plus, 
                        smp.UnetPlusPlus,  smp.FPN]#,smp.PAN, smp.Unet, smp.FPN, smp.PSPNet, smp.UnetPlusPlus,]# ] # smp.DeepLabV3,  smp.PAN, smp.Linknet
    # BACKBONE_NAME = ["resnet18", "resnet34", "resnet50", "resnet101", ]#"resnet34_wt", "resnet18_wt"]

    # SMP_MODEL = [smp.Unet] # smp.DeepLabV3,  smp.PAN, smp.Linknet,smp.Unet, smp.FPN, smp.PSPNet, smp.UnetPlusPlus, 
    # SMP_MODEL = [smp.DeepLabV3,  smp.PAN, smp.Linknet, smp.DeepLabV3Plus] ## Set diconv True
    
    # BACKBONE_NAME = ["vgg11_bn_wt", "vgg11_bn","vgg13_bn_wt", "vgg13_bn",
    #                     "vgg16_bn_wt", "vgg16_bn","vgg19_bn_wt", "vgg19_bn",]
    # BACKBONE_NAME = ["xception", "xception_wt"]#"resnet18_wt", "resnet34_wt", "resnet50_wt", "resnet101_wt"] #  "mobilenet_v2_wt"
    # BACKBONE_NAME = ["resnet18_wt", "resnet34_wt", "resnet50_wt", "resnet101_wt" ] #
    BACKBONE_NAME = ["resnet18","resnet34", "resnet50", "resnet101", ] #
                        # 'densenet121', 'densenet169', 'densenet201',]
                        #  'vgg11_bn',  'vgg13_bn',  'vgg16_bn', 'vgg19_bn', 
                        #   'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6',
                        #  'xception','mobilenet_v2',]# Now : UNet: densenet201


    GPU_NUM = "0"
    args.gpus = GPU_NUM
    res_root = "result"
    res_save = "camvid_SMP"

    for m in range(len(SMP_MODEL)):
        for n in range(len(BACKBONE_NAME)):
            sel_model = str(SMP_MODEL[m].__name__) + '_' + BACKBONE_NAME[n]
            # print(sel_model)
            args.model = sel_model

            args.checkpoint = "/home/gpxu/vess_seg/vess_efficient/checkpoint/camvid/"+ \
                                sel_model+"bs8gpu1_trainval_"+SAVE_FLAG[SEL_NUM]+"/model_350.pth"
            print(args.model)
            args.save_seg_dir = os.path.join(res_root, res_save, args.model)

            test_model(args, Seg_Model=SMP_MODEL[m], Backbone=BACKBONE_NAME[n])
