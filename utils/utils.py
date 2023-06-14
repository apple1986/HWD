import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from utils.colorize_mask import cityscapes_colorize_mask, camvid_colorize_mask, most_colorize_mask
from medpy import metric
from scipy.ndimage import zoom
import SimpleITK as sitk
import time
import cv2


#################extract boundary loop
import kornia as K
from kornia import morphology as morph
def extract_boundary_loop(label, output):
    label = label[:, None, ::].to(torch.float32)
    # device = "cuda:" + args.gpus
    kernel = torch.ones(7, 7).to("cuda:0")

    gt_sobel: torch.Tensor = K.filters.sobel(label)#[1] ## obtain boundary
    gt_sobel = (gt_sobel > gt_sobel.min()).to(torch.float32)
    dilated_loop = morph.dilation(gt_sobel, kernel) # Dilation
    gt_loop = dilated_loop * label
    output_loop = dilated_loop * output
    gt_loop = gt_loop.squeeze(1).to(torch.long)

    return output_loop, gt_loop

############################
def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                  **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_predict(output, gt, img_name, dataset, save_path, output_grey=False, output_color=True, gt_color=False):
    if output_grey:
        output_grey = Image.fromarray(output)
        output_grey.save(os.path.join(save_path, img_name + '_pd.png'))

        gt_output = Image.fromarray(gt)
        gt_output.save(os.path.join(save_path, img_name + '_gt.png'))

    if output_color:
        if dataset == 'cityscapes':
            output_color = cityscapes_colorize_mask(output)
        elif dataset == 'camvid':
            output_color = camvid_colorize_mask(output)
        elif dataset == 'most_png':
            output_color = most_colorize_mask(output)
        elif dataset == 'most_png_v2':
            output_color = most_colorize_mask(output)
        elif dataset == 'most_png_v3':
            output_color = most_colorize_mask(output)   
        elif dataset == 'ROSE-2':
            output_color = most_colorize_mask(output)      
        elif dataset == 'LN':
            output_color = most_colorize_mask(output)     
        elif dataset == 'Breast':
            output_color = most_colorize_mask(output)      
        elif dataset == 'ThreeCenter':
            output_color = most_colorize_mask(output)      
        elif dataset == 'FiveCenter':
            output_color = most_colorize_mask(output)  
        elif dataset == 'SingleCenter':
            output_color = most_colorize_mask(output)   
        elif dataset == 'isic':
            output_color = most_colorize_mask(output)   
        output_color.save(os.path.join(save_path, img_name + '_color.png'))

    if gt_color:
        if dataset == 'cityscapes':
            gt_color = cityscapes_colorize_mask(gt)
        elif dataset == 'camvid':
            gt_color = camvid_colorize_mask(gt)
        elif dataset == 'most_png':
            gt_color = most_colorize_mask(gt)
        elif dataset == 'most_png_v2':
            gt_color = most_colorize_mask(gt)
        elif dataset == 'most_png_v3':
            gt_color = most_colorize_mask(gt)
        elif dataset == 'ROSE-2':
            gt_color = most_colorize_mask(gt)
        elif dataset == 'LN':
            gt_color = most_colorize_mask(gt)
        elif dataset == 'Breast':
            gt_color = most_colorize_mask(gt)
        elif dataset == 'ThreeCenter':
            gt_color = most_colorize_mask(gt)  
        elif dataset == 'FiveCenter':
            gt_color = most_colorize_mask(gt)
        elif dataset == 'SingleCenter':
            gt_color = most_colorize_mask(gt) 
        elif dataset == 'isic':
            gt_color = most_colorize_mask(gt)
            
        gt_color.save(os.path.join(save_path, img_name + '_gt.png'))


def netParams(model):
    """
    computing total network parameters
    args:
       model: model
    return: the number of parameters
    """
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters

###############aping: evaluation: synapse
def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0

##########ap: cal single volume for each slice
def eval_every_slice(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            print('index:', ind)
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs= net(input) #,_,_,_,_ 
                # np.save('./vis_feature/all_final_384_244x244_'+str(ind)+'.npy', att_feature.detach().cpu().numpy())
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

    ## save the results        
    metric_list = []
    metric_class_all = []
    
    for n in range(image.shape[0]):
        for i in range(1, classes):
            pd_slice = prediction[n]
            gt_slice = label[n]
            metric_class_all.append(calculate_metric_percase(pd_slice == i, gt_slice == i))
        temp_mean = np.mean(np.array(metric_class_all), axis=0)
        temp_std = np.std(np.array(metric_class_all), axis=0)
        # temp_mean = temp_mean.transpose()
        metric_list.append(list([temp_mean,temp_std]))
        metric_class_all = []
    ## write the results into txt
    logFileLoc = "/home/gpxu/vess_seg/vess_efficient/vis_feature/FuseSeg_Synapse/synsape_result_Swin_MLP_F9_F30_v2.txt" #os.path.join(os.path.dirname(args.checkpoint), args.logFile)

    # Save the result

    logger = open(logFileLoc, 'a')
    logger.write("\nSubject Name: %s\tDSC\tHD\tDSC_std\tHD_std\n" % case[0])
    # logger.write(str(metric_list))
    for i in range(len(metric_list)):
        logger.write("%d\t%.4f\t%.4f\t" % (i, metric_list[i][0][0],metric_list[i][0][1]))
        logger.write("%.4f\t%.4f\n" % (metric_list[i][1][0],metric_list[i][1][1]))
    logger.flush()
    logger.close()

    if test_save_path is  not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case[0] + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case[0] + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case[0] + "_gt.nii.gz")
    return None #metric_list  


#########ap: test each volume of 3D volume
## input: H*W*N
def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            print('index:', ind)
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs= net(input) #,_,_,_,_ 
                # np.save('./vis_feature/all_final_384_244x244_'+str(ind)+'.npy', att_feature.detach().cpu().numpy())
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    # metric_list = [1.0]
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case[0] + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case[0] + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case[0] + "_gt.nii.gz")
    return np.array(metric_list)  

########################
## for Gray + 2 FFT images
## ## input: H*W*3*N
def test_single_volume_3d(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]//3):
            slice = image[ind*3:(ind+1)*3, :, :]
            print('index:', ind)
            x, y = slice.shape[1], slice.shape[2]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (1, patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs= net(input) #,_,_,_,_ 
                # np.save('./vis_feature/all_final_384_244x244_'+str(ind)+'.npy', att_feature.detach().cpu().numpy())
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case[0] + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case[0] + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case[0] + "_gt.nii.gz")
    return metric_list    

########################
def test_single_volume_time(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    time_list = []
    total_slice_num = 0
    if len(image.shape) == 3:
        # prediction = np.zeros_like(label)
        total_slice_num = total_slice_num + image.shape[0]
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            print('index:', ind)
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                start_time = time.time()
                outputs= net(input)
                torch.cuda.synchronize()
                time_taken = time.time() - start_time
                time_list.append(time_taken)
                print('[%d/%d]  time: %.7f' % (ind + 1, image.shape[0], time_taken))

    if test_save_path is not None:
        print('saving...')
        # print('time list: ', time_list)
    ## mean prediction time
    time_mean = np.mean(time_list)
    # print("time mean", time_mean)

    return time_mean

#############give more weight near the boudary of an selected object
def weight_boundary_nearby(gt, label_id = 9):
    ## extract boundary from a mask
    gt = (gt * (gt == label_id))
    gt_edge = cv2.Canny(gt,0,20)

    ## dilate the boundary
    kernel = np.ones((9,9), np.float32)
    edge_di = cv2.dilate(gt_edge,kernel)

    ## cal distance from boundary
    skeleton_op = (gt_edge == 0).astype(np.uint8)
    dt,lbl = cv2.distanceTransformWithLabels(skeleton_op, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    ## only for the dilated region
    edge_di_th = (edge_di > 0).astype(np.uint8) ## 0, 1
    dt_sel = edge_di_th * dt
    dt_sel = np.max(dt_sel) - dt_sel + 2
    # ## Gaussian: give the weight of object according to the distance to boundary
    # mu = 0
    # sd = 1.5
    # dt_sel = stats.norm(mu, sd).pdf(dt_sel) + 1
    dt_weight = dt_sel * edge_di_th + 1 ## get the weight map: 

    return dt_weight