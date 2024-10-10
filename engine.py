# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import cv2
import math
import os
import sys
from typing import Iterable
import numpy as np
import torch
import util.misc as utils
from torch import Tensor
from datasets.data_prefetcher import data_dict_to_cuda
from util import box_ops
from util.plot_utils import draw_boxes, draw_ref_pts, image_hwc2chw

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def get_rotate_mat(theta):
    '''positive theta value means rotate clockwise'''
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

def train_one_epoch_mot(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    count = 0
    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for data_dict in metric_logger.log_every(data_loader, print_freq, header):
        img_path = data_dict['img_path']
        del data_dict['img_path']
        data_dict = data_dict_to_cuda(data_dict, device)
        outputs = model(data_dict)
###################画出不带角度的output图#######
        # i = 0
        # img = cv2.imread(img_path[0])
        # img= Tensor(img)
        # img_h,img_w,c = img.shape
        # out_bbox = outputs['pred_boxes'][i][0].cpu()
        # boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(boxes)
        # boxes = boxes * scale_fct[None, :]
        # boxes = boxes.tolist()
        # img_show = draw_boxes(img,boxes)
        # filename = f"show/show_{count}.jpg"
        # out_bbox = out_bbox * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        # cv2.imwrite(filename,img_show)
        # count=(count+1)%1000
####################带角度#################
        # for j in range(len(img_path)):
        #     img = cv2.imread(img_path[j])
        #     # img= Tensor(img)
        #     img_h,img_w,c = img.shape
        #     out_bbox = outputs['pred_boxes'][j][0].cpu()
        #     rotates = outputs["pred_rotate"][j][0].cpu()
        #     paths = img_path[j].split('/')
        #     path = paths[-2] + '/' + paths[-1]
        #     filename = "./show/" + path
        #     if not os.path.exists('./show/'+paths[-2]):
        #         os.makedirs('./show/'+paths[-2])
        #     out_bbox = out_bbox * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        #     img = np.array(img)
        #     out_bbox = box_cxcywh_to_xyxy(out_bbox)
        #     for box, rotate in zip(out_bbox, rotates):
        #         x_min,y_min,x_max,y_max = box.detach().numpy() 
        #         rotate_mat = get_rotate_mat(-rotate)
        #         temp_x = np.array([[x_min, x_max, x_max, x_min]]) - (x_min+x_max)/2
        #         temp_y = np.array([[y_min, y_min, y_max, y_max]]) - (y_min+y_max)/2
        #         coordidates = np.concatenate((temp_x, temp_y), axis=0)
        #         res = np.dot(rotate_mat, coordidates)
        #         res[0,:] += (x_min+x_max)/2
        #         res[1,:] += (y_min+y_max)/2
        #         points = np.array([[res[0,0], res[1,0]], [res[0,1], res[1,1]], [res[0,2], res[1,2]], [res[0,3], res[1,3]]], np.int32)
        #         cv2.polylines(img, [points], True, (0,255,0), thickness=1) # 
        #     cv2.imwrite(filename,img)
#######################
        loss_dict = criterion(outputs, data_dict)
        # print("iter {} after model".format(cnt-1))
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)\
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())


        # loss_value = losses_reduced_scaled.item() + SL1_loss*10000
        loss_value = losses_reduced_scaled.item()

        # print("SL1_loss:",SL1_loss*10000)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        # gather the stats from all processes

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
