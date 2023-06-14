import os.path as osp
import numpy as np
import random
import cv2
from torch.utils import data
import pickle
from torchvision.transforms import InterpolationMode
import torchvision
from PIL import Image
from math import ceil, floor
from skimage import measure
import copy
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

# ia.seed(1)

"""
CamVid is a road scene understanding dataset with 367 training images and 233 testing images of day and dusk scenes. 
The challenge is to segment 11 classes such as road, building, cars, pedestrians, signs, poles, side-walk etc. We 
resize images to 360x480 pixels for training and testing.
"""

## sacle the selected object
## sacle the selected object
def copy_obj(img, gt, label_id=9):
    img_ori = copy.deepcopy(img)
    gt_ori = copy.deepcopy(gt)

    img_change = copy.deepcopy(img)
    gt_change = copy.deepcopy(gt)

    h_img, w_img, _ = img.shape
    # w_img, h_img = img.size
    ## 1. extract the selected object
    gt = (gt * (gt == label_id)) 

    bw = (gt>0).astype(int)
    labels =measure.label(bw)

    if (len(np.unique(labels)) > 1):
        ## get obj coord
        ## select one object
        for n in range(1, labels.max()+1):#labels.max()+1
            obj_num = np.sum(labels == n)
            if (obj_num >= 500) or (obj_num <=20):
                continue

            x, y = (labels == n).nonzero() ## obtain idx
            ## 2. select bounding box: obtain the begining coord
            # x, y = (gt_1 > 0).nonzero()
            # if (any(x)):
            sel_obj = gt_ori[min(x):max(x)+1, min(y):max(y)+1]
            sel_img = img_ori[min(x):max(x)+1, min(y):max(y)+1, :]
            h_obj, w_obj = sel_obj.shape
            ## select a random position
            x_beg_pos = random.randint(0, h_img)
            y_beg_pos = random.randint(0, w_img)
            if (x_beg_pos + h_obj) <= h_img:
                x_end_pos = x_beg_pos + h_obj
            else:
                x_end_pos = h_img
            
            if (y_beg_pos + w_obj) <= w_img:
                y_end_pos = y_beg_pos + w_obj
            else:
                y_end_pos = w_img

            ## copy obj
            gt_change[x_beg_pos:x_end_pos, y_beg_pos:y_end_pos] = sel_obj[0:x_end_pos-x_beg_pos, 0:y_end_pos-y_beg_pos]
            img_change[x_beg_pos:x_end_pos, y_beg_pos:y_end_pos,:] = sel_img[0:x_end_pos-x_beg_pos, 0:y_end_pos-y_beg_pos,:]

    idx = np.where(gt_change != label_id)
    gt_change[idx] = gt_ori[idx]

    for n in range(0,3):
        img_changed_tmp = img_change[:,:,n]
        img_ori_tmp = img_ori[:,:,n]
        img_changed_tmp[idx] = img_ori_tmp[idx]
        img_change[:,:,n] = img_changed_tmp
            # gt_ori[np.argwhere(gt_used == label_id)]=label_id

    return img_change, gt_change

## sacle the selected object
def copy_obj_rect(img, gt, label_id=9):
    img_ori = copy.deepcopy(img)
    gt_ori = copy.deepcopy(gt)

    img_change = copy.deepcopy(img)
    gt_change = copy.deepcopy(gt)

    h_img, w_img, _ = img.shape
    # w_img, h_img = img.size
    ## 1. extract the selected object
    gt = (gt * (gt == label_id)) 

    bw = (gt>0).astype(int)
    labels =measure.label(bw)

    if (len(np.unique(labels)) > 1):
        ## get obj coord
        ## select one object
        for n in range(1, labels.max()+1):#labels.max()+1
            x, y = (labels == n).nonzero()
            ## 2. select bounding box: obtain the begining coord
            # x, y = (gt_1 > 0).nonzero()
            # if (any(x)):
            sel_obj = gt_ori[min(x):max(x)+1, min(y):max(y)+1]
            sel_img = img_ori[min(x):max(x)+1, min(y):max(y)+1, :]
            h_obj, w_obj = sel_obj.shape
            ## select a random position
            x_beg_pos = random.randint(0, h_img)
            y_beg_pos = random.randint(0, w_img)
            if (x_beg_pos + h_obj) <= h_img:
                x_end_pos = x_beg_pos + h_obj
            else:
                x_end_pos = h_img
            
            if (y_beg_pos + w_obj) <= w_img:
                y_end_pos = y_beg_pos + w_obj
            else:
                y_end_pos = w_img

            ## copy obj
            gt_change[x_beg_pos:x_end_pos, y_beg_pos:y_end_pos] = sel_obj[0:x_end_pos-x_beg_pos, 0:y_end_pos-y_beg_pos]
            img_change[x_beg_pos:x_end_pos, y_beg_pos:y_end_pos,:] = sel_img[0:x_end_pos-x_beg_pos, 0:y_end_pos-y_beg_pos,:]


    return img_change, gt_change

## ap: define aug func
def obj_scale_aug_rect(img, gt, label_id, scale_rate=2):
    h_img, w_img, _ = img.shape
    # w_img, h_img = img.size
    ## 1. extract the selected object
    gt_1 = (gt * (gt == label_id)) 

    bw = (gt_1>0).astype(int)
    labels =measure.label(bw) # 0 is background

    # patch_obj_save = []
    # patch_img_save = []
    img_change = copy.deepcopy(img)
    gt_change = copy.deepcopy(gt)
    if (len(np.unique(labels)) > 1):
        for n in range(1, 2):#labels.max()+1
            x, y = (labels == n).nonzero()

        ## 2. select bounding box: obtain the begining coord
        # x, y = (gt_1 > 0).nonzero()
        # if (any(x)):
            sel_obj = gt[min(x):max(x)+1, min(y):max(y)+1]
            sel_img = img[min(x):max(x)+1, min(y):max(y)+1, :]
            h_pre, w_pre = sel_obj.shape
            ## 3. resize       
            sel_img = Image.fromarray(sel_img)
            sel_obj = Image.fromarray(sel_obj)

            shape_aug_img = torchvision.transforms.Resize(size=(int(h_pre*scale_rate), int(w_pre*scale_rate)),
                    interpolation=InterpolationMode.BICUBIC)
            shape_aug_gt = torchvision.transforms.Resize(size=(int(h_pre*scale_rate), int(w_pre*scale_rate)), 
                    interpolation=InterpolationMode.NEAREST)
            
            img_want = shape_aug_img(sel_img)
            gt_want = shape_aug_gt(sel_obj)

            # gt_want = F.interpolate(sel_obj, scale_factor=2)
            # img_want = F.interpolate(sel_img, scale_factor=2)

            

            gt_want = np.array(gt_want)
            img_want = np.array(img_want)
            h_pos, w_pos = gt_want.shape

            ## 4. recover to ori image
            x_now = min(x)-floor((h_pos - h_pre)/2)
            y_now = min(y)-floor((w_pos - w_pre)/2)

            x_end = x_now + h_pos
            y_end = y_now + w_pos

            h_beg = 0
            h_end = h_pos
            w_beg = 0
            w_end = w_pos

            if x_now < 0:
                h_beg = x_now *  (-1) # changed img
                x_now = 0  
            if x_end > h_img:
                h_end = h_beg + (h_img - x_now) # changed img
                x_end = h_img

            # if (x_now >= 0) and (x_end > h_img):
            #     h_end = h_img - x_now # changed img


            if y_now < 0:
                w_beg = y_now * (-1) # changed img
                y_now = 0
            if y_end > w_img:
                w_end = w_beg + (w_img - y_now) # changed img
                y_end = w_img

            # if (y_now >= 0) and (y_end > w_img):
            #     w_end = w_img - y_now # changed img

            gt_change[x_now:x_end, y_now:y_end] = gt_want[h_beg:h_end, w_beg:w_end]
            img_change[x_now:x_end, y_now:y_end, :] = img_want[h_beg:h_end, w_beg:w_end, :]
    

    return img_change, gt_change

## sacle the selected object
def obj_scale_aug(img, gt, label_id, scale_rate=2):
    img_ori = copy.deepcopy(img)
    gt_ori = copy.deepcopy(gt)

    img_change = copy.deepcopy(img)
    gt_change = copy.deepcopy(gt)

    h_img, w_img, _ = img.shape
    # w_img, h_img = img.size
    ## 1. extract the selected object
    gt = (gt * (gt == label_id)) 

    bw = (gt>0).astype(int)
    labels =measure.label(bw)

    # patch_obj_save = []
    # patch_img_save = []
    if (len(np.unique(labels)) > 1):
        for n in range(1, labels.max()+1):#labels.max()+1
            x, y = (labels == n).nonzero()

        ## 2. select bounding box: obtain the begining coord
        # x, y = (gt_1 > 0).nonzero()
        # if (any(x)):
            sel_obj = gt[min(x):max(x)+1, min(y):max(y)+1]
            sel_img = img[min(x):max(x)+1, min(y):max(y)+1, :]
            h_pre, w_pre = sel_obj.shape
            ## 3. resize       
            sel_img = Image.fromarray(sel_img)
            sel_obj = Image.fromarray(sel_obj)

            # print(h_pre, w_pre)
            if (h_pre > 1) and (w_pre > 1):
                shape_aug_img = torchvision.transforms.Resize(size=(int(h_pre*scale_rate), int(w_pre*scale_rate)),
                        interpolation=InterpolationMode.BICUBIC)
                shape_aug_gt = torchvision.transforms.Resize(size=(int(h_pre*scale_rate), int(w_pre*scale_rate)), 
                        interpolation=InterpolationMode.NEAREST)
                
                img_want = shape_aug_img(sel_img)
                gt_want = shape_aug_gt(sel_obj)

                # gt_want = F.interpolate(sel_obj, scale_factor=2)
                # img_want = F.interpolate(sel_img, scale_factor=2)

                

                gt_want = np.array(gt_want)
                img_want = np.array(img_want)
                img_want[:,:,0] = img_want[:,:,0] * (gt_want>0)
                img_want[:,:,1] = img_want[:,:,1] * (gt_want>0)
                img_want[:,:,2] = img_want[:,:,2] * (gt_want>0)

                h_pos, w_pos = gt_want.shape
                ## 4. recover to ori image
                x_now = min(x)-floor((h_pos - h_pre)/2)
                y_now = min(y)-floor((w_pos - w_pre)/2)

                x_end = x_now + h_pos
                y_end = y_now + w_pos

                h_beg = 0
                h_end = h_pos
                w_beg = 0
                w_end = w_pos

                if x_now < 0:
                    h_beg = x_now *  (-1) # changed img
                    x_now = 0  
                if x_end > h_img:
                    h_end = h_beg + (h_img - x_now) # changed img
                    x_end = h_img

                # if (x_now >= 0) and (x_end > h_img):
                #     h_end = h_img - x_now # changed img


                if y_now < 0:
                    w_beg = y_now * (-1) # changed img
                    y_now = 0
                if y_end > w_img:
                    w_end = w_beg + (w_img - y_now) # changed img
                    y_end = w_img

                # if (y_now >= 0) and (y_end > w_img):
                #     w_end = w_img - y_now # changed img

                gt_change[x_now:x_end, y_now:y_end] = gt_want[h_beg:h_end, w_beg:w_end]
                img_change[x_now:x_end, y_now:y_end, :] = img_want[h_beg:h_end, w_beg:w_end, :]
        
        # else:
        #     pass
        idx = np.where(gt_change != label_id)
        gt_change[idx] = gt_ori[idx]

        for n in range(0,3):
            img_changed_tmp = img_change[:,:,n]
            img_ori_tmp = img_ori[:,:,n]
            img_changed_tmp[idx] = img_ori_tmp[idx]
            img_change[:,:,n] = img_changed_tmp
            # gt_ori[np.argwhere(gt_used == label_id)]=label_id

    return img_change, gt_change

## copy and scale 
def copy_obj_scale_gray(img, gt, label_id=9, scale_rate=2):

    # YN = random.randint(0,1)
    # if (YN) :
    #     return img, gt

    img_ori = copy.deepcopy(img)
    gt_ori = copy.deepcopy(gt)

    img_change = copy.deepcopy(img)
    gt_change = copy.deepcopy(gt)

    h_img, w_img = img.shape

    # w_img, h_img = img.size
    ## 1. extract the selected object
    gt = (gt * (gt == label_id)) 

    bw = (gt>0).astype(int)
    labels =measure.label(bw)

    if (len(np.unique(labels)) > 1):
        ## get obj coord
        ## select one object
        for n in range(1, labels.max()+1):#labels.max()+1
            obj_num = np.sum(labels == n)
            if (obj_num >= 500) or (obj_num <=20):
                continue

            x, y = (labels == n).nonzero() ## obtain idx
            ## 2. select bounding box: obtain the begining coord
            # x, y = (gt_1 > 0).nonzero()
            # if (any(x)):
            sel_obj = gt_ori[min(x):max(x)+1, min(y):max(y)+1]
            sel_img = img_ori[min(x):max(x)+1, min(y):max(y)+1]
            h_pre, w_pre = sel_obj.shape
            sel_img = Image.fromarray(sel_img)
            sel_obj = Image.fromarray(sel_obj)
            
            ## sacling
            shape_aug_img = torchvision.transforms.Resize(size=(int(h_pre*scale_rate), int(w_pre*scale_rate)),
                    interpolation=InterpolationMode.BICUBIC)
            shape_aug_gt = torchvision.transforms.Resize(size=(int(h_pre*scale_rate), int(w_pre*scale_rate)), 
                    interpolation=InterpolationMode.NEAREST)
            
            img_want = shape_aug_img(sel_img)
            gt_want = shape_aug_gt(sel_obj)            

            sel_img = np.array(img_want)
            sel_obj = np.array(gt_want)

            h_obj, w_obj = sel_obj.shape
            ## select a random position
            x_beg_pos = random.randint(0, h_img)
            y_beg_pos = random.randint(0, w_img)
            if (x_beg_pos + h_obj) <= h_img:
                x_end_pos = x_beg_pos + h_obj
            else:
                x_end_pos = h_img
            
            if (y_beg_pos + w_obj) <= w_img:
                y_end_pos = y_beg_pos + w_obj
            else:
                y_end_pos = w_img

            ## copy obj
            gt_change[x_beg_pos:x_end_pos, y_beg_pos:y_end_pos] = sel_obj[0:x_end_pos-x_beg_pos, 0:y_end_pos-y_beg_pos]
            img_change[x_beg_pos:x_end_pos, y_beg_pos:y_end_pos] = sel_img[0:x_end_pos-x_beg_pos, 0:y_end_pos-y_beg_pos]

    idx = np.where(gt_change != label_id)
    gt_change[idx] = gt_ori[idx]
    img_change[idx] = img_ori[idx]

    # for n in range(0,3):
    #     img_changed_tmp = img_change[:,:,n]
    #     img_ori_tmp = img_ori[:,:,n]
    #     img_changed_tmp[idx] = img_ori_tmp[idx]
    #     img_change[:,:,n] = img_changed_tmp
            # gt_ori[np.argwhere(gt_used == label_id)]=label_id

    return img_change, gt_change


## ap: aug image
def aug_img(img, gt):
    ## define aug pipeline
    seq = iaa.Sequential([
    # iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
    iaa.Sharpen((0.0, 0.5)),       # sharpen the image
    iaa.Affine(
        scale={"x": (0.8, 2), "y": (0.8, 2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-10, 10),
        shear=(-8, 8)
    ),  # rotate by -45 to 45 degrees (affects segmaps)
    iaa.ElasticTransformation(alpha=1, sigma=1),  # apply water effect (affects segmaps)
    iaa.flip.Fliplr(p=0.3)
                        ], random_order=True)
    segmap = SegmentationMapsOnImage(gt, shape=img.shape)
    img_aug, gt_aug = seq(image=img, segmentation_maps=segmap)
    gt_aug = gt_aug.arr
    gt_aug = np.squeeze(gt_aug, axis=2)
    return img_aug, gt_aug

## give more weight near the boundary
def weight_boundary_nearby(gt, label_id = [7,9]):
    gt_ori = copy.deepcopy(gt)
    dt_weight = np.zeros_like(gt)
    
    for idx in label_id:
        ## extract boundary from a mask
        gt = (gt_ori * (gt_ori == idx))
        if (len(np.unique(gt)) == 1):
            continue

        # gt_edge = cv2.Canny(gt,0,20)
        contours, hierarchy = cv2.findContours((gt>0).astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) 
        gt_edge = np.zeros_like(gt)
        for contour in contours:
            gt_edge[contour[:,:,1], contour[:,:,0]] = 1

        ## dilate the boundary
        kernel = np.ones((6,6), np.float32)
        edge_di = cv2.dilate(gt_edge.astype('uint8'),kernel)

        ## cal distance from boundary
        skeleton_op = (gt_edge == 0).astype(np.uint8)
        dt,lbl = cv2.distanceTransformWithLabels(skeleton_op, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
        ## only for the dilated region
        edge_di_th = (edge_di > 0).astype(np.uint8) ## 0, 1
        # edge_di_th = (((edge_di > 0)*dt_weight)==1).astype(np.uint8) ## remove overlapping

        dt_sel = edge_di_th * dt
        dt_sel = np.max(dt_sel) - dt_sel #+ 2
        # ## Gaussian: give the weight of object according to the distance to boundary
        # mu = 0
        # sd = 1.5
        # dt_sel = stats.norm(mu, sd).pdf(dt_sel) + 1
        dt_weight = dt_weight + (dt_sel+0.0) * edge_di_th #+ 1 ## get the weight map: 
        ## change weight
        # dt_weight = dt_weight + (dt_weight > 1)
    
    # dt_weight = dt_weight * (gt_ori<11)
    dt_weight = dt_weight / (np.max(dt_weight)-np.min(dt_weight)+1e-6) #+ 1 # norm
    dt_weight = dt_weight + 1

    return dt_weight

## give weight according the distance to boundary
def weight_mask_distance(gt, label_id = [7,9]):
    gt_ori = copy.deepcopy(gt)
    dt_weight = np.zeros_like(gt)
    
    for idx in label_id:
        ## extract boundary from a mask
        gt = (gt_ori * (gt_ori == idx))
        if (len(np.unique(gt)) == 1):
            continue
            # return dt_weight
        mask_obj = (gt_ori == idx).astype(np.uint8)

        # save_img = os.path.join(root_path, "gt_temp_2.png")
        # temp = cv2.applyColorMap(gt_edge.astype(np.uint8)*80, cv2.COLORMAP_JET)
        # cv2.imwrite(save_img, temp)

        ## cal distance from boundary
        dt,lbl = cv2.distanceTransformWithLabels(mask_obj, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
        ## only for the dilated region
        # dt_sel = mask_obj * dt
        dt_sel = (np.max(dt) - dt)*mask_obj #+ 2
        # ## Gaussian: give the weight of object according to the distance to boundary
        # mu = 0
        # sd = 1.5
        # dt_sel = stats.norm(mu, sd).pdf(dt_sel) + 1
        dt_weight = dt_weight + dt_sel #+ 1 ## get the weight map: 
    
    dt_weight = dt_weight / (np.max(dt_weight)-np.min(dt_weight)+1e-6) + 1 # norm

    return dt_weight