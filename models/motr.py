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
DETR model and criterion classes.
"""
import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List
# from .Rotated_ROIAlign.roi_align_rotate import ROIAlignRotated
from util import box_ops, checkpoint
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate, get_rank,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from models.structures import Instances, Boxes, pairwise_iou, matched_boxlist_iou

from .backbone import build_backbone
from .matcher import build_matcher
from .deformable_transformer_plus import build_deforamble_transformer, pos2posemb
from .qim import build as build_query_interaction_layer
from .deformable_detr import SetCriterion, MLP, sigmoid_focal_loss
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

def smooth_l1_loss(prediction, target):
    # Calculate Smooth L1 Loss element-wise
    loss = F.smooth_l1_loss(prediction, target, reduction='none')
    return loss

def match_boxes(predictions, targets, iou_threshold=0.5):
    # Calculate IoU between each predicted box and each target box
    ious = calculate_L1_iou(predictions, targets)

    # Find the best matching target box for each predicted box
    best_target_per_prediction = ious.argmax(dim=1)
    
    # Filter out predictions that do not have a good match
    mask = ious[torch.arange(ious.size(0)), best_target_per_prediction] > 0.01
    return best_target_per_prediction, mask

def calculate_L1_iou(boxes1, boxes2):
    # Calculate IoU between two sets of bounding boxes

    intersection_x1 = torch.max(boxes1[:, 0].unsqueeze(1), boxes2[:, 0])
    intersection_y1 = torch.max(boxes1[:, 1].unsqueeze(1), boxes2[:, 1])
    intersection_x2 = torch.min(boxes1[:, 0] + boxes1[:, 2], boxes2[:, 0] + boxes2[:, 2])
    intersection_y2 = torch.min(boxes1[:, 1] + boxes1[:, 3], boxes2[:, 1] + boxes2[:, 3])

    intersection_area = torch.clamp(intersection_x2 - intersection_x1, min=0) * torch.clamp(intersection_y2 - intersection_y1, min=0)

    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]
    union_area = area1.unsqueeze(1) + area2 - intersection_area

    iou = intersection_area / (union_area + 1e-6)

    return iou

def multibox_smooth_l1_loss(predictions, targets, iou_threshold=0.5):
    # predictions: Tensor of shape (N, 4), where N is the number of predicted bounding boxes
    # targets: Tensor of shape (M, 4), where M is the number of target bounding boxes
    
    # Find the best matching target box for each predicted box
    # print(predictions.shape,targets.shape)
    best_target_per_prediction, mask = match_boxes(predictions, targets, iou_threshold=iou_threshold)

    # Use the best matching targets for Smooth L1 Loss calculation
    selected_targets = targets[best_target_per_prediction[mask]]

    # Calculate Smooth L1 Loss
    loss = smooth_l1_loss(predictions[mask], selected_targets)

    return loss.mean()

def calculate_iou(box1, box2):
    x1_1, y1_1, w1_1, h1_1 = box1
    x1_2, y1_2, w1_2, h1_2 = box2

    # 计算框1的右下角坐标
    x2_1 = x1_1 + w1_1
    y2_1 = y1_1 + h1_1

    # 计算框2的右下角坐标
    x2_2 = x1_2 + w1_2
    y2_2 = y1_2 + h1_2

    # 计算交集框的坐标
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    # 计算交集框的宽度和高度
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)

    # 计算交集框的面积
    area_inter = inter_width * inter_height

    # 计算框1和框2的面积
    area_box1 = w1_1 * h1_1
    area_box2 = w1_2 * h1_2

    # 计算并集框的面积
    area_union = area_box1 + area_box2 - area_inter

    # 计算交并比（IoU）
    iou = area_inter / area_union

    # 保留小的框
    if area_box1 < area_box2:
        return iou, True
    else:
        return iou, False

class ClipMatcher(SetCriterion):
    def __init__(self, num_classes,
                        matcher,
                        weight_dict,
                        losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__(num_classes, matcher, weight_dict, losses)
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_loss = True
        self.losses_dict = {}
        self._current_frame_idx = 0

    def initialize_for_single_clip(self, gt_instances: List[Instances]):
        self.gt_instances = gt_instances
        self.num_samples = 0
        self.sample_device = None
        self._current_frame_idx = 0
        self.losses_dict = {}

    def _step(self):
        self._current_frame_idx += 1

    def calc_loss_for_track_scores(self, track_instances: Instances):
        frame_id = self._current_frame_idx - 1
        gt_instances = self.gt_instances[frame_id]
        outputs = {
            'pred_logits': track_instances.track_scores[None],
        }
        device = track_instances.track_scores.device

        num_tracks = len(track_instances)
        src_idx = torch.arange(num_tracks, dtype=torch.long, device=device)
        tgt_idx = track_instances.matched_gt_idxes  # -1 for FP tracks and disappeared tracks

        track_losses = self.get_loss('labels',
                                     outputs=outputs,
                                     gt_instances=[gt_instances],
                                     indices=[(src_idx, tgt_idx)],
                                     num_boxes=1)
        self.losses_dict.update(
            {'frame_{}_track_{}'.format(frame_id, key): value for key, value in
             track_losses.items()})

    def get_num_boxes(self, num_samples):
        num_boxes = torch.as_tensor(num_samples, dtype=torch.float, device=self.sample_device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        return num_boxes

    def get_loss(self, loss, outputs, gt_instances, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'rotate': self.loss_rotate,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, gt_instances, indices, num_boxes, **kwargs)

    def loss_boxes(self, outputs, gt_instances: List[Instances], indices: List[tuple], num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        # We ignore the regression loss of the track-disappear slots.
        #TODO: Make this filter process more elegant.
        filtered_idx = []
        for src_per_img, tgt_per_img in indices:
            keep = tgt_per_img != -1
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
        indices = filtered_idx
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([gt_per_img.boxes[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0)

        # for pad target, don't calculate regression loss, judged by whether obj_id=-1
        target_obj_ids = torch.cat([gt_per_img.obj_ids[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0) # size(16)
        mask = (target_obj_ids != -1)

        loss_bbox = F.l1_loss(src_boxes[mask], target_boxes[mask], reduction='none')
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes[mask]),
            box_ops.box_cxcywh_to_xyxy(target_boxes[mask])))

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_labels(self, outputs, gt_instances: List[Instances], indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # The matched gt for disappear track query is set -1.
        labels = []
        for gt_per_img, (_, J) in zip(gt_instances, indices):
            labels_per_img = torch.ones_like(J)
            # set labels of track-appear slots to 0.
            if len(gt_per_img) > 0:
                labels_per_img[J != -1] = gt_per_img.labels[J[J != -1]]
            labels.append(labels_per_img)
        target_classes_o = torch.cat(labels)
        target_classes[idx] = target_classes_o
        if self.focal_loss:
            gt_labels_target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[:, :, :-1]  # no loss for the last (background) class
            gt_labels_target = gt_labels_target.to(src_logits)
            loss_ce = sigmoid_focal_loss(src_logits.flatten(1),
                                             gt_labels_target.flatten(1),
                                             alpha=0.25,
                                             gamma=2,
                                             num_boxes=num_boxes, mean_in_dim1=False)
            loss_ce = loss_ce.sum()
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses
    
    def loss_rotate(self, outputs, gt_instances: List[Instances], indices, num_boxes, log=False):
        """Classification loss (NLL)
        gt_instances dicts must contain the key "pred_rotate" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_rotate' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_logits = outputs['pred_rotate']
        
        target_rotate = torch.full(src_logits.shape[:2], 0.0,
                                    dtype=torch.float, device=src_logits.device)
        
        ignored_classes = torch.full(src_logits.shape[:2], 1.0,
                                    dtype=torch.long, device=src_logits.device)
        
        # The matched gt for disappear track query is set -1.
        labels = []
        ignored_label = []
        for gt_per_img, (_, J) in zip(gt_instances, indices):
            rotate_per_img = torch.zeros_like(J,dtype=torch.float)
            labels_ignored_per_img = torch.ones_like(J,dtype=torch.long)
            # set labels of track-appear slots to 0.
            
            if len(gt_per_img) > 0:
                rotate_per_img[J != -1] = gt_per_img.rotate[J[J != -1]]
                labels_ignored_per_img[J != -1] = gt_per_img.texts_ignored[J[J != -1]]

            labels.append(rotate_per_img)
            ignored_label.append(labels_ignored_per_img)
            
        target_rotate_o = torch.cat(labels)
        target_rotate[idx] = target_rotate_o
        
        target_classes_o_ignored = torch.cat(ignored_label)
        ignored_classes[idx] = target_classes_o_ignored
        ignored_classes = ignored_classes.float()

        ignored = torch.full(src_logits.shape[:2], 0.0,
                                    dtype=torch.float, device=src_logits.device)
        ignored[idx] = 1.0

        pred_rotate = (src_logits.sigmoid() - 0.5) * math.pi
#         print(ignored_classes)
        angle_loss = 1 - torch.cos(pred_rotate*ignored.unsqueeze(-1)*ignored_classes.unsqueeze(-1) - target_rotate.unsqueeze(-1)*ignored_classes.unsqueeze(-1))
        sum_ = torch.clamp(ignored.sum(),1,10000)
        losses = {'loss_angle': angle_loss.sum()/(sum_*num_boxes)}

        return losses

    def match_for_single_frame(self, outputs: dict):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        ########### 提取这帧的pred放output_i 和gt的id放obj_idxes
        gt_instances_i = self.gt_instances[self._current_frame_idx]  # gt instances of i-th image.
        track_instances: Instances = outputs_without_aux['track_instances']
        pred_logits_i = track_instances.pred_logits  # predicted logits of i-th image.
        pred_boxes_i = track_instances.pred_boxes  # predicted boxes of i-th image.
        pred_rotate_i = track_instances.pred_rotate

        obj_idxes = gt_instances_i.obj_ids
        outputs_i = {
            'pred_logits': pred_logits_i.unsqueeze(0),
            'pred_boxes': pred_boxes_i.unsqueeze(0),
            'pred_rotate': pred_rotate_i.unsqueeze(0),
        }
        ########### 已有目标与新的 Ground Truth 目标之间的匹配过程
        # step1. inherit and update the previous tracks.
        num_disappear_track = 0
        track_instances.matched_gt_idxes[:] = -1
        i, j = torch.where(track_instances.obj_idxes[:, None] == obj_idxes) # 找到已有目标与新的 Ground Truth 目标之间的匹配关系。i 和 j 是两个张量，其中 i 是 track_instances.obj_idxes 中匹配到的索引，j 是 obj_idxes 中匹配到的索引
        track_instances.matched_gt_idxes[i] = j 

        full_track_idxes = torch.arange(len(track_instances), dtype=torch.long, device=pred_logits_i.device)
        matched_track_idxes = (track_instances.obj_idxes >= 0)  # occu 
        prev_matched_indices = torch.stack(
            [full_track_idxes[matched_track_idxes], track_instances.matched_gt_idxes[matched_track_idxes]], dim=1)

        # step2. select the unmatched slots.
        # note that the FP tracks whose obj_idxes are -2 will not be selected here.
        unmatched_track_idxes = full_track_idxes[track_instances.obj_idxes == -1]  # 得到未被匹配的槽位

        # step3. select the untracked gt instances (new tracks).
        tgt_indexes = track_instances.matched_gt_idxes
        tgt_indexes = tgt_indexes[tgt_indexes != -1] # 得到被跟踪的 GT 目标的索引

        tgt_state = torch.zeros(len(gt_instances_i), device=pred_logits_i.device) #创建一个长度为 GT 目标数的零张量 
        tgt_state[tgt_indexes] = 1 #已经匹配的 GT 目标的位置设置为 1
        untracked_tgt_indexes = torch.arange(len(gt_instances_i), device=pred_logits_i.device)[tgt_state == 0] 
        # untracked_tgt_indexes = select_unmatched_indexes(tgt_indexes, len(gt_instances_i))
        untracked_gt_instances = gt_instances_i[untracked_tgt_indexes] # 获取未跟踪的 GT 目标实例


        # 这个辅助函数接受未匹配的输出（unmatched_outputs）和一个匹配器（matcher），
        # 然后使用匹配器将未匹配的输出与未跟踪的 GT 目标进行匹配。返回的 new_track_indices 是一个元组列表，
        # 表示新的匹配关系。接着，通过这些索引创建一个新的匹配关系的张量 new_matched_indices，
        # 其中包含了未匹配的跟踪槽位和未跟踪的 GT 目标的索引。
        # 整个过程的目的是为后续的匹配和损失计算提供正确的索引关系
        def match_for_single_decoder_layer(unmatched_outputs, matcher): # 然后使用匹配器将未匹配的输出与未跟踪的 GT 目标进行匹配
            new_track_indices = matcher(unmatched_outputs,
                                             [untracked_gt_instances])  # list[tuple(src_idx, tgt_idx)]

            src_idx = new_track_indices[0][0]
            tgt_idx = new_track_indices[0][1]
            # concat src and tgt.
            new_matched_indices = torch.stack([unmatched_track_idxes[src_idx], untracked_tgt_indexes[tgt_idx]],
                                              dim=1).to(pred_logits_i.device)
            return new_matched_indices

        # step4. do matching between the unmatched slots and GTs.
        # 从 track_instances 中提取未匹配的槽位的预测输出
        unmatched_outputs = {
            'pred_logits': track_instances.pred_logits[unmatched_track_idxes].unsqueeze(0),
            'pred_boxes': track_instances.pred_boxes[unmatched_track_idxes].unsqueeze(0),
            'pred_rotate': track_instances.pred_rotate[unmatched_track_idxes].unsqueeze(0),
        }
        
        # 使用匹配器（self.matcher）将这些未匹配的槽位与未跟踪的 GT 目标进行匹配，得到新的匹配关系的索引。
        new_matched_indices = match_for_single_decoder_layer(unmatched_outputs, self.matcher) 

        # step5. update obj_idxes according to the new matching result.
        # 将新匹配的 GT 目标的 ID 更新到对应的跟踪槽位中，同时更新匹配关系的信息
        # 未匹配槽位和 GT 之间的匹配，维度相同
        track_instances.obj_idxes[new_matched_indices[:, 0]] = gt_instances_i.obj_ids[new_matched_indices[:, 1]].long()
        track_instances.matched_gt_idxes[new_matched_indices[:, 0]] = new_matched_indices[:, 1]

        # step6. calculate iou.
        #在这一步，首先确定哪些跟踪槽位是“活跃”的，即已经匹配到了目标。
        # 然后，提取这些活跃槽位的预测边界框和相应的 GT 边界框，并将它们转换为 xyxy 格式。
        # 接着，使用 matched_boxlist_iou 函数计算它们之间的 IoU，
        # 并将结果存储在 track_instances.iou 中。

        active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.matched_gt_idxes >= 0)
        active_track_boxes = track_instances.pred_boxes[active_idxes]
        active_track_angle = track_instances.pred_rotate[active_idxes]
        if len(active_track_boxes) > 0:
            gt_boxes = gt_instances_i.boxes[track_instances.matched_gt_idxes[active_idxes]]

            active_track_boxes = box_ops.box_cxcywh_to_xyxy(active_track_boxes)
            gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
            track_instances.iou[active_idxes] = matched_boxlist_iou(Boxes(active_track_boxes), Boxes(gt_boxes))

            gt_angle = gt_instances_i.rotate[track_instances.matched_gt_idxes[active_idxes]]

            active_track_angle = (active_track_angle.sigmoid() - 0.5) * math.pi
            track_instances.angle[active_idxes] = torch.abs(active_track_angle[0] - gt_angle)

        # step7. merge the unmatched pairs and the matched pairs.
        # 最后，将新匹配的索引和之前匹配的索引合并到一个张量中，以便用于后续的处理。
        matched_indices = torch.cat([new_matched_indices, prev_matched_indices], dim=0)

        # step8. calculate losses.
        #通过迭代训练过程中定义的损失函数列表 self.losses，
        # 使用 get_loss 函数计算新的损失。
        # 计算时使用了之前匹配到的槽位和 GT 目标的索引信息。
        # 最后，将计算得到的新损失信息更新到 self.losses_dict 中，其中包括了帧索引信息
        self.num_samples += len(gt_instances_i) + num_disappear_track
        self.sample_device = pred_logits_i.device
        for loss in self.losses:
            new_track_loss = self.get_loss(loss,
                                           outputs=outputs_i,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices[:, 0], matched_indices[:, 1])],
                                           num_boxes=1)
            self.losses_dict.update(
                {'frame_{}_{}'.format(self._current_frame_idx, key): value for key, value in new_track_loss.items()})
        # 如果输出中包含辅助输出（'aux_outputs'），则对每个辅助输出执行类似的损失计算。
        # 首先，提取未匹配的辅助输出信息，然后通过之前定义的匹配函数 match_for_single_decoder_layer 进行匹配。
        # 接着，将新的匹配索引与之前的匹配索引合并，并使用相同的损失函数列表计算新的损失，将结果更新到 self.losses_dict 中
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                unmatched_outputs_layer = {
                    'pred_logits': aux_outputs['pred_logits'][0, unmatched_track_idxes].unsqueeze(0),
                    'pred_boxes': aux_outputs['pred_boxes'][0, unmatched_track_idxes].unsqueeze(0),
                    'pred_rotate': aux_outputs['pred_rotate'][0, unmatched_track_idxes].unsqueeze(0),
                }
                new_matched_indices_layer = match_for_single_decoder_layer(unmatched_outputs_layer, self.matcher)
                matched_indices_layer = torch.cat([new_matched_indices_layer, prev_matched_indices], dim=0)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(loss,
                                           aux_outputs,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices_layer[:, 0], matched_indices_layer[:, 1])],
                                           num_boxes=1, )
                    self.losses_dict.update(
                        {'frame_{}_aux{}_{}'.format(self._current_frame_idx, i, key): value for key, value in
                         l_dict.items()})
        # 类似地，如果输出中包含像素级别的输出（'ps_outputs'），则对每个像素级别的输出执行类似的损失计算。
        # 此处使用了一个简单的 get_loss 函数，计算 'boxes' 类型的损失
        if 'ps_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['ps_outputs']):
                ar = torch.arange(len(gt_instances_i), device=obj_idxes.device)
                l_dict = self.get_loss('boxes',
                                        aux_outputs,
                                        gt_instances=[gt_instances_i],
                                        indices=[(ar, ar)],
                                        num_boxes=1, )
                self.losses_dict.update(
                    {'frame_{}_ps{}_{}'.format(self._current_frame_idx, i, key): value for key, value in
                        l_dict.items()})
        self._step()
        return track_instances
    

    def forward(self, outputs, input_data: dict):
        # losses of each frame are calculated during the model's forwarding and are outputted by the model as outputs['losses_dict].
        losses = outputs.pop("losses_dict")
        num_samples = self.get_num_boxes(self.num_samples)
        for loss_name, loss in losses.items():
            losses[loss_name] /= num_samples
        return losses


class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.6, filter_score_thresh=0.5, miss_tolerance=10):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances):
        device = track_instances.obj_idxes.device

        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        new_obj = (track_instances.obj_idxes == -1) & (track_instances.scores >= self.score_thresh)
        disappeared_obj = (track_instances.obj_idxes >= 0) & (track_instances.scores < self.filter_score_thresh)
        num_new_objs = new_obj.sum().item()

        track_instances.obj_idxes[new_obj] = self.max_obj_id + torch.arange(num_new_objs, device=device)
        self.max_obj_id += num_new_objs

        track_instances.disappear_time[disappeared_obj] += 1
        to_del = disappeared_obj & (track_instances.disappear_time >= self.miss_tolerance)
        track_instances.obj_idxes[to_del] = -1


class TrackerPostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, track_instances: Instances, target_size) -> Instances:
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits = track_instances.pred_logits
        out_bbox = track_instances.pred_boxes
        out_rotate = track_instances.pred_rotate
        rotate = (out_rotate.sigmoid() - 0.5) * math.pi


        scores = out_logits[..., 0].sigmoid()

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = target_size
        scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(boxes)
        boxes = boxes * scale_fct[None, :]

        track_instances.rotate = rotate
        track_instances.boxes = boxes
        track_instances.scores = scores
        track_instances.labels = torch.full_like(scores, 0)

        return track_instances


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Conv_BN_ReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0):
        super(Conv_BN_ReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()
        self.apply(self._init_weights)

    def  _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()    
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class FPEM_v1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPEM_v1, self).__init__()
        planes = out_channels
        self.dwconv3_1 = nn.Conv2d(
                planes,planes,kernel_size=3,stride=1,padding=1,groups=planes)
        self.smooth_layer3_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv2_1 = nn.Conv2d(
                planes,planes,kernel_size=3,stride=1,padding=1,groups=planes)
        self.smooth_layer2_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv1_1 = nn.Conv2d(
                planes,planes,kernel_size=3,stride=1,padding=1, groups=planes)
        self.smooth_layer1_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv2_2 = nn.Conv2d(
                planes,planes, kernel_size=3,stride=2,padding=1,groups=planes)
        self.smooth_layer2_2 = Conv_BN_ReLU(planes, planes)

        self.dwconv3_2 = nn.Conv2d(
                planes,planes,kernel_size=3,stride=2,padding=1,groups=planes)
        self.smooth_layer3_2 = Conv_BN_ReLU(planes, planes)

        self.dwconv4_2 = nn.Conv2d(
                planes,planes,kernel_size=3, stride=2,padding=1,groups=planes)
        self.smooth_layer4_2 = Conv_BN_ReLU(planes, planes)

    def _upsample_add(self, x, y):
        _, _, H, W = y.shape
        return F.upsample(x, size=[H, W], mode='bilinear') + y

    def forward(self, f1, f2, f3, f4):
       
        f3 = self.smooth_layer3_1(self.dwconv3_1(self._upsample_add(f4, f3)))
        f2 = self.smooth_layer2_1(self.dwconv2_1(self._upsample_add(f3, f2)))
        f1 = self.smooth_layer1_1(self.dwconv1_1(self._upsample_add(f2, f1)))
        
        f2 = self.smooth_layer2_2(self.dwconv2_2(self._upsample_add(f2, f1)))
        f3 = self.smooth_layer3_2(self.dwconv3_2(self._upsample_add(f3, f2)))
        f4 = self.smooth_layer4_2(self.dwconv4_2(self._upsample_add(f4, f3)))
        return f1, f2, f3, f4


class FPEM_v2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPEM_v2, self).__init__()
        planes = out_channels
        self.dwconv3_1 = nn.Conv2d(
                planes,planes,kernel_size=3,stride=1,padding=1,groups=planes,bias=False)
        self.smooth_layer3_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv2_1 = nn.Conv2d(
                planes,planes,kernel_size=3,stride=1,padding=1,groups=planes, bias=False)
        self.smooth_layer2_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv1_1 = nn.Conv2d(
                planes,planes,kernel_size=3,stride=1,padding=1,groups=planes,bias=False)
        self.smooth_layer1_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv2_2 = nn.Conv2d(
                planes,planes,kernel_size=3,stride=2,padding=1,groups=planes,bias=False)
        self.smooth_layer2_2 = Conv_BN_ReLU(planes, planes)

        self.dwconv3_2 = nn.Conv2d(
                planes, planes,kernel_size=3,stride=2,padding=1,groups=planes,bias=False)
        self.smooth_layer3_2 = Conv_BN_ReLU(planes, planes)

        self.dwconv4_2 = nn.Conv2d(
                planes,planes,kernel_size=3,stride=2,padding=1,groups=planes,bias=False)
        self.smooth_layer4_2 = Conv_BN_ReLU(planes, planes)

    def _upsample_add(self, x, y):
        _, _, H, W = y.shape
        return F.upsample(x, size=[H, W], mode='bilinear') + y

    def forward(self, f1, f2, f3, f4):

        f3_ = self.smooth_layer3_1(self.dwconv3_1(self._upsample_add(f4, f3)))
        f2_ = self.smooth_layer2_1(self.dwconv2_1(self._upsample_add(f3_, f2)))
        f1_ = self.smooth_layer1_1(self.dwconv1_1(self._upsample_add(f2_, f1)))

        f2_ = self.smooth_layer2_2(self.dwconv2_2(self._upsample_add(f2_,f1_)))
        f3_ = self.smooth_layer3_2(self.dwconv3_2(self._upsample_add(f3_,f2_)))
        f4_ = self.smooth_layer4_2(self.dwconv4_2(self._upsample_add(f4, f3_)))
                
        f1 = f1 + f1_
        f2 = f2 + f2_
        f3 = f3 + f3_
        f4 = f4 + f4_

        return f1, f2, f3, f4

class MOTR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels, criterion, track_embed,
                 aux_loss=True, with_box_refine=False, two_stage=False, memory_bank=None, use_checkpoint=False, query_denoise=0):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.track_embed = track_embed
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_classes = num_classes
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.rotate_embed = nn.Linear(hidden_dim, 1)
        self.num_feature_levels = num_feature_levels
        self.use_checkpoint = use_checkpoint
        self.query_denoise = query_denoise
        self.position = nn.Embedding(num_queries, 4)
        self.yolox_embed = nn.Embedding(1, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if query_denoise:
            self.refine_embed = nn.Embedding(1, hidden_dim)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        # out_chan =256
        # self.fpem1 = FPEM_v1(in_channels=(256, 256, 256, 256),out_channels=out_chan)
        # self.fpem2 = FPEM_v2(in_channels=(246, 256, 256, 256),out_channels=out_chan)



        # self.roirotate = ROIAlignRotated((8,32), spatial_scale = (1.), sampling_ratio = 0)
        

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        self.rotate_embed.bias.data = torch.ones(num_classes) * bias_value

        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        nn.init.uniform_(self.position.weight.data, 0, 1)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.rotate_embed = _get_clones(self.rotate_embed, num_pred)
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            self.rotate_embed = nn.ModuleList([self.rotate_embed for _ in range(num_pred)])
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        self.post_process = TrackerPostProcess()
        self.track_base = RuntimeTrackerBase()
        self.criterion = criterion
        self.memory_bank = memory_bank
        self.mem_bank_len = 0 if memory_bank is None else memory_bank.max_his_length

    def _generate_empty_tracks(self, proposals=None):
        track_instances = Instances((1, 1))
        num_queries, d_model = self.query_embed.weight.shape  # (300, 512)
        device = self.query_embed.weight.device
        if proposals is None:
            track_instances.ref_pts = self.position.weight
            track_instances.query_pos = self.query_embed.weight
        else:
            track_instances.ref_pts = torch.cat([self.position.weight, proposals[:, :4]])
            track_instances.query_pos = torch.cat([self.query_embed.weight, pos2posemb(proposals[:, 4:], d_model) + self.yolox_embed.weight])
        track_instances.output_embedding = torch.zeros((len(track_instances), d_model), device=device)
        track_instances.obj_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.disappear_time = torch.zeros((len(track_instances), ), dtype=torch.long, device=device)
        track_instances.iou = torch.ones((len(track_instances),), dtype=torch.float, device=device)
        track_instances.angle = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros((len(track_instances), 4), dtype=torch.float, device=device)
        track_instances.pred_logits = torch.zeros((len(track_instances), self.num_classes), dtype=torch.float, device=device)
        track_instances.pred_rotate = torch.zeros((len(track_instances), 1), dtype=torch.float, device=device)

        mem_bank_len = self.mem_bank_len
        track_instances.mem_bank = torch.zeros((len(track_instances), mem_bank_len, d_model), dtype=torch.float32, device=device)
        track_instances.mem_padding_mask = torch.ones((len(track_instances), mem_bank_len), dtype=torch.bool, device=device)
        track_instances.save_period = torch.zeros((len(track_instances), ), dtype=torch.float32, device=device)

        return track_instances.to(self.query_embed.weight.device)
    
    # def _generate_empty_tracks(self):
    #     track_instances = Instances((1, 1))
    #     num_queries, d_model = self.query_embed.weight.shape  # (300, 512)
    #     device = self.query_embed.weight.device
    #     track_instances.ref_pts = self.position.weight
    #     track_instances.query_pos = self.query_embed.weight
    #     track_instances.output_embedding = torch.zeros((len(track_instances), d_model), device=device)
    #     track_instances.obj_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
    #     track_instances.matched_gt_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
    #     track_instances.disappear_time = torch.zeros((len(track_instances), ), dtype=torch.long, device=device)
    #     track_instances.iou = torch.ones((len(track_instances),), dtype=torch.float, device=device)
    #     track_instances.angle = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
    #     track_instances.scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
    #     track_instances.track_scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
    #     track_instances.pred_boxes = torch.zeros((len(track_instances), 4), dtype=torch.float, device=device)
    #     track_instances.pred_logits = torch.zeros((len(track_instances), self.num_classes), dtype=torch.float, device=device)
    #     track_instances.pred_rotate = torch.zeros((len(track_instances), 1), dtype=torch.float, device=device)

    #     mem_bank_len = self.mem_bank_len
    #     track_instances.mem_bank = torch.zeros((len(track_instances), mem_bank_len, d_model), dtype=torch.float32, device=device)
    #     track_instances.mem_padding_mask = torch.ones((len(track_instances), mem_bank_len), dtype=torch.bool, device=device)
    #     track_instances.save_period = torch.zeros((len(track_instances), ), dtype=torch.float32, device=device)

    #     return track_instances.to(self.query_embed.weight.device)


    def clear(self):
        self.track_base.clear()

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_rotate):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_rotate':c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_rotate[:-1])]
    
    # def _set_aux_loss(self, outputs_class, outputs_coord):
    #     # this is a workaround to make torchscript happy, as torchscript
    #     # doesn't support dictionary with non-homogeneous values, such
    #     # as a dict having both a Tensor and a list.
    #     return [{'pred_logits': a, 'pred_boxes': b}
    #             for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def _forward_single_image(self, samples, track_instances: Instances, gtboxes=None):
        #hrnet40
        # features, pos, layer1 = self.backbone(samples)
        #resnet50
        features, pos = self.backbone(samples) # [1, 512, 96, 171]/[1, 1024, 48, 86]/[1, 2048, 24, 43]
        src, mask = features[-1].decompose()
        assert mask is not None

        # cxcywh_boxes = gtboxes.clone()  # 复制gtboxes张量，避免修改原始数据
        # # 计算左上角和右下角坐标
        # ltwh_boxes = cxcywh_boxes[:, :2] - cxcywh_boxes[:, 2:] / 2
        # rb_boxes = cxcywh_boxes[:, :2] + cxcywh_boxes[:, 2:] / 2
        # # 构建四点坐标
        # gtboxes = torch.cat((ltwh_boxes, torch.stack((rb_boxes[:, 0], ltwh_boxes[:, 1]), dim=1),
        #                             rb_boxes, torch.stack((ltwh_boxes[:, 0], rb_boxes[:, 1]), dim=1)), dim=1)

        # cx = (track_instances.ref_pts[:, 0] + track_instances.ref_pts[:, 2] + track_instances.ref_pts[:, 4] + track_instances.ref_pts[:, 6]) / 4
        # cy = (track_instances.ref_pts[:, 1] + track_instances.ref_pts[:, 3] + track_instances.ref_pts[:, 5] + track_instances.ref_pts[:, 7]) / 4
        # w = torch.max(track_instances.ref_pts[:, 0:8:2], dim=-1).values - torch.min(track_instances.ref_pts[:, 0:8:2], dim=-1).values
        # h = torch.max(track_instances.ref_pts[:, 1:8:2], dim=-1).values - torch.min(track_instances.ref_pts[:, 1:8:2], dim=-1).values
        # track_instances.ref_pts = torch.stack((cx, cy, w, h), dim=-1)

        srcs = []
        masks = []
        for l, feat in enumerate(features): #对三层特征进行降维
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None  # [1, 256, 96, 171]/[1, 256, 48, 86]/[1, 256, 24, 43]

        if self.num_feature_levels > len(srcs): #从第三层特征提出第四层特征 并降维
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors) # [1, 256, 12, 22]
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l) # [1, 256, 96, 171]/[1, 256, 48, 86]/[1, 256, 24, 43]/[1, 256, 12, 22]

        if gtboxes is not None:
            n_dt = len(track_instances)
            ps_tgt = self.refine_embed.weight.expand(gtboxes.size(0), -1)
            query_embed = torch.cat([track_instances.query_pos, ps_tgt])
            ref_pts = torch.cat([track_instances.ref_pts, gtboxes])
            attn_mask = torch.zeros((len(ref_pts), len(ref_pts)), dtype=bool, device=ref_pts.device)
            attn_mask[:n_dt, n_dt:] = True
        else:
            query_embed = track_instances.query_pos
            ref_pts = track_instances.ref_pts
            attn_mask = None
#######################特征加强#####################################################
        # f1, f2, f3, f4 = srcs
        
        # f1_1, f2_1, f3_1, f4_1 = self.fpem1(f1, f2, f3, f4)
        # f1_2, f2_2, f3_2, f4_2 = self.fpem2(f1_1, f2_1, f3_1, f4_1)

        # f1 = f1_1 + f1_2#[1, 128, 160, 160]
        # f2 = f2_1 + f2_2#[1, 128, 80, 80]
        # f3 = f3_1 + f3_2#[1, 128, 40, 40]
        # f4 = f4_1 + f4_2##[1, 128, 20, 20]

        # srcs[0] = f1
        # srcs[1] = f2
        # srcs[2] = f3
        # srcs[3] = f4

        # p1, p2, p3, p4 = pos
        
        # p1_1, p2_1, p3_1, p4_1 = self.fpem1(p1, p2, p3, p4)
        # p1_2, p2_2, p3_2, p4_2 = self.fpem2(p1_1, p2_1, p3_1, p4_1)

        # p1 = p1_1 + p1_2#[1, 128, 160, 160]
        # p2 = p2_1 + p2_2#[1, 128, 80, 80]
        # p3 = p3_1 + p3_2#[1, 128, 40, 40]
        # p4 = p4_1 + p4_2##[1, 128, 20, 20]

        # pos[0] = p1
        # pos[1] = p2
        # pos[2] = p3
        # pos[3] = p4

#######################################################################
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = \
            self.transformer(srcs, masks, pos, query_embed, ref_pts=ref_pts,
                             mem_bank=track_instances.mem_bank, mem_bank_pad_mask=track_instances.mem_padding_mask, attn_mask=attn_mask)

        outputs_classes = []
        outputs_coords = []
        outputs_rotates = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            outputs_rotate = self.rotate_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_rotates.append(outputs_rotate)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_rotate = torch.stack(outputs_rotates)

        ref_pts_all = torch.cat([init_reference[None][:, :, :, :2], inter_references[:, :, :, :2]], dim=0)
        out = {'pred_logits': outputs_class[-1], 'pred_rotate': outputs_rotate[-1], 'pred_boxes': outputs_coord[-1],'ref_pts': ref_pts_all[2]}
        # out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],'ref_pts': ref_pts_all[2]}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_rotate)
            # out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        out['hs'] = hs[-1]
        return out

    def _post_process_single_image(self, frame_res, track_instances, is_last,proposals):
        if self.query_denoise > 0:
            n_ins = len(track_instances)
            ps_logits = frame_res['pred_logits'][:, n_ins:]
            ps_boxes = frame_res['pred_boxes'][:, n_ins:]
            frame_res['hs'] = frame_res['hs'][:, :n_ins]
            frame_res['pred_logits'] = frame_res['pred_logits'][:, :n_ins]
            frame_res['pred_boxes'] = frame_res['pred_boxes'][:, :n_ins]
            frame_res['pred_rotate'] = frame_res['pred_rotate'][:, :n_ins]
            ps_outputs = [{'pred_logits': ps_logits, 'pred_boxes': ps_boxes}]
            for aux_outputs in frame_res['aux_outputs']:
                ps_outputs.append({
                    'pred_logits': aux_outputs['pred_logits'][:, n_ins:],
                    'pred_boxes': aux_outputs['pred_boxes'][:, n_ins:],
                    'pred_rotate': aux_outputs['pred_rotate'][:, n_ins:],
                })
                aux_outputs['pred_logits'] = aux_outputs['pred_logits'][:, :n_ins]
                aux_outputs['pred_rotate'] = aux_outputs['pred_rotate'][:, :n_ins]
                aux_outputs['pred_boxes'] = aux_outputs['pred_boxes'][:, :n_ins]
            frame_res['ps_outputs'] = ps_outputs

        with torch.no_grad():
            if self.training:
                track_scores = frame_res['pred_logits'][0, :].sigmoid().max(dim=-1).values
            else:
                track_scores = frame_res['pred_logits'][0, :, 0].sigmoid() #预测到追踪分数

        track_instances.scores = track_scores  # 筛选新旧物体的依据
        # print('at1:',track_instances.scores)
        track_instances.pred_logits = frame_res['pred_logits'][0]
        track_instances.pred_boxes = frame_res['pred_boxes'][0]
        track_instances.pred_rotate = frame_res['pred_rotate'][0]
        track_instances.output_embedding = frame_res['hs'][0]
        if self.training:
            frame_res['track_instances'] = track_instances
            track_instances = self.criterion.match_for_single_frame(frame_res)
        else:
            # each track will be assigned an unique global id by the track base.
            # keep = track_instances.scores > self.track_base.filter_score_thresh
            # track_instances = track_instances[keep]
            #########################非极大值抑制交并比大的框#######################
            score_shreld = 0.5
            for id in range(len(track_instances)):
                if track_instances.scores[id] >= score_shreld:
                    for id2 in range(len(track_instances)):
                        id3 = id2
                        if id3 == id:
                            break
                        elif id3<len(track_instances) and track_instances.scores[id3] >= score_shreld:
                            #判断id与id3交是否够大 若是 则保留小框 
                            iou,min_is_id = calculate_iou(track_instances._fields['pred_boxes'][id],track_instances._fields['pred_boxes'][id3]) 
                            if 0.5<=iou and iou <0.95:
                                if min_is_id == True:
                                    track_instances.scores[id3] = 0
                                    # track_instances._fields['scores'][id3] = 0
                                    track_instances._fields['pred_logits'][id3] = -1
                                    # track_instances._fields['pred_boxes'][id3] = torch.Tensor([0, 0, 0, 0])
                                elif min_is_id == False:
                                    track_instances.scores[id] = 0
                                    # track_instances._fields['scores'][id] = 0
                                    track_instances._fields['pred_logits'][id] = -1
                                    # track_instances._fields['pred_boxes'][id3] = torch.Tensor([0, 0, 0, 0])
                                else:
                                    print('出现错误1')
                            elif 0.95<=iou:
                                #有一个有id
                                if track_instances._fields['obj_idxes'][id] >= 0 and track_instances._fields['obj_idxes'][id3] < 0:
                                    track_instances.scores[id3] = 0
                                    # track_instances._fields['scores'][id3] = 0
                                    track_instances._fields['pred_logits'][id3] = -1
                                elif track_instances._fields['obj_idxes'][id3] >= 0 and track_instances._fields['obj_idxes'][id] < 0:
                                    track_instances.scores[id] = 0
                                    # track_instances._fields['scores'][id] = 0
                                    track_instances._fields['pred_logits'][id] = -1
                                #两个都没id
                                elif track_instances._fields['obj_idxes'][id3] < 0 and track_instances._fields['obj_idxes'][id] < 0:
                                    if min_is_id == True:
                                        track_instances.scores[id3] = 0
                                        # track_instances._fields['scores'][id3] = 0
                                        track_instances._fields['pred_logits'][id3] = -1
                                    elif min_is_id == False:
                                        track_instances.scores[id] = 0
                                        # track_instances._fields['scores'][id] = 0
                                        track_instances._fields['pred_logits'][id] = -1
                                    else:
                                        print('出现错误2')
                                #两个都有id   
                                elif track_instances._fields['obj_idxes'][id3] >= 0 and track_instances._fields['obj_idxes'][id] >= 0:
                                    if id>id3:
                                        track_instances.scores[id3] = 0
                                        track_instances._fields['pred_logits'][id3] = -1
                                    elif id<id3:
                                        track_instances.scores[id] = 0
                                        track_instances._fields['pred_logits'][id] = -1
                                    else:
                                        print('出现错误3')
                                else:
                                    print('ERROR:iou大于0.95时出现两新生成的相似框同时有id')
                            elif iou<0.5:
                                continue
                            else:
                                print('出现错误4')
                        elif track_instances.scores[id3] < score_shreld:
                            continue
                        else:
                            print("ERROR")
            ######################################################
            # print('at2:',track_instances.scores)
            self.track_base.update(track_instances)  #新物体的产生 和消失物体的保留
            # print('at3:',track_instances.scores)
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)

#######################################
        tmp = {}
        tmp['track_instances'] = track_instances
        # tmp['init_track_instances'] = self._generate_empty_tracks()
        if not is_last:
            out_track_instances = self.track_embed(tmp) #选择跟踪物体 并更新queryEnbeding
            frame_res['track_instances'] = out_track_instances
        else:
            frame_res['track_instances'] = None
        if self.training:
            return frame_res
        else:
            return frame_res
########################################

        # tmp = {}
        # tmp['init_track_instances'] = self._generate_empty_tracks(proposals)
        # tmp['track_instances'] = track_instances
        # out_track_instances = self.track_embed(tmp)
        # frame_res['track_instances'] = out_track_instances 
        # return frame_res


    @torch.no_grad()
    def inference_single_image(self, img, ori_img_size, track_instances=None, proposals=None):
        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensor_list(img)
        if track_instances is None:
            track_instances = self._generate_empty_tracks(proposals)
        else:
            track_instances = Instances.cat([
                self._generate_empty_tracks(proposals),
                track_instances])
        res = self._forward_single_image(img,
                                         track_instances=track_instances) #单单找出总的query输出的新的query
        res = self._post_process_single_image(res, track_instances, False, proposals)

        track_instances = res['track_instances']
        track_instances = self.post_process(track_instances, ori_img_size) #sgmoid分数
        # print('scores5:',track_instances.scores)
        ret = {'track_instances': track_instances}
        if 'ref_pts' in res:
            ref_pts = res['ref_pts']
            img_h, img_w = ori_img_size
            scale_fct = torch.Tensor([img_w, img_h]).to(ref_pts)
            ref_pts = ref_pts * scale_fct[None]
            ret['ref_pts'] = ref_pts
        return ret

    def forward(self, data: dict):
        if self.training:
            self.criterion.initialize_for_single_clip(data['gt_instances'])
        frames = data['imgs']  # list of Tensor.
        outputs = {
            'pred_logits': [],
            'pred_boxes': [],
            'pred_rotate': [],
        }
        track_instances = None
       
        keys = list(self._generate_empty_tracks()._fields.keys())
        for frame_index, (frame, gt, proposals) in enumerate(zip(frames, data['gt_instances'], data['proposals'])):
            frame.requires_grad = False
            is_last = frame_index == len(frames) - 1

            if self.query_denoise > 0:
                l_1 = l_2 = self.query_denoise
                gtboxes = gt.boxes.clone()
                _rs = torch.rand_like(gtboxes) * 2 - 1
                gtboxes[..., :2] += gtboxes[..., 2:] * _rs[..., :2] * l_1
                gtboxes[..., 2:] *= 1 + l_2 * _rs[..., 2:]
            else:
                gtboxes = None

            if track_instances is None:
                # track_instances = self._generate_empty_tracks()
                track_instances = self._generate_empty_tracks(proposals)
            else:
                # track_instances = Instances.cat([
                #     self._generate_empty_tracks(),
                #     track_instances])
                track_instances = Instances.cat([
                    self._generate_empty_tracks(proposals),
                    track_instances])

            if self.use_checkpoint and frame_index < len(frames) - 1:
                def fn(frame, gtboxes, *args):
                    frame = nested_tensor_from_tensor_list([frame])
                    tmp = Instances((1, 1), **dict(zip(keys, args)))
                    frame_res = self._forward_single_image(frame, tmp, gtboxes)
                    return (
                        frame_res['pred_logits'],
                        frame_res['pred_boxes'],
                        frame_res['hs'],
                        *[aux['pred_logits'] for aux in frame_res['aux_outputs']],
                        *[aux['pred_boxes'] for aux in frame_res['aux_outputs']]
                    )

                args = [frame, gtboxes] + [track_instances.get(k) for k in keys]
                params = tuple((p for p in self.parameters() if p.requires_grad))
                tmp = checkpoint.CheckpointFunction.apply(fn, len(args), *args, *params)
                frame_res = {
                    'pred_logits': tmp[0],
                    'pred_boxes': tmp[1],
                    'hs': tmp[2],
                    'aux_outputs': [{
                        'pred_logits': tmp[3+i],
                        'pred_boxes': tmp[3+5+i],
                    } for i in range(5)],
                }
            else:
                frame = nested_tensor_from_tensor_list([frame])
                frame_res = self._forward_single_image(frame, track_instances, gtboxes)
            frame_res = self._post_process_single_image(frame_res, track_instances, is_last,proposals)
            
            track_instances = frame_res['track_instances']
            outputs['pred_logits'].append(frame_res['pred_logits'])
            outputs['pred_boxes'].append(frame_res['pred_boxes'])
            outputs['pred_rotate'].append(frame_res['pred_rotate'])
            
        if not self.training:
            outputs['track_instances'] = track_instances
        else:
            outputs['losses_dict'] = self.criterion.losses_dict
        return outputs


def build(args):
    dataset_to_num_classes = {
        'coco': 91,
        'coco_panoptic': 250,
        'e2e_mot': 1,
        'e2e_dance': 1,
        'e2e_joint': 1,
        'e2e_static_mot': 1,
        'VideoText':1
    }
    assert args.dataset_file in dataset_to_num_classes
    num_classes = dataset_to_num_classes[args.dataset_file]
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    d_model = transformer.d_model
    hidden_dim = args.dim_feedforward
    query_interaction_layer = build_query_interaction_layer(args, args.query_interaction_layer, d_model, hidden_dim, d_model*2)

    img_matcher = build_matcher(args)
    num_frames_per_batch = max(args.sampler_lengths)
    weight_dict = {}
    for i in range(num_frames_per_batch):
        weight_dict.update({"frame_{}_loss_ce".format(i): args.cls_loss_coef,
                            'frame_{}_loss_bbox'.format(i): args.bbox_loss_coef,
                            'frame_{}_loss_giou'.format(i): args.giou_loss_coef,
                            'frame_{}_loss_angle'.format(i): 50.,
                            })

    # TODO this is a hack
    if args.aux_loss:
        for i in range(num_frames_per_batch):
            for j in range(args.dec_layers - 1):
                weight_dict.update({"frame_{}_aux{}_loss_ce".format(i, j): args.cls_loss_coef,
                                    'frame_{}_aux{}_loss_bbox'.format(i, j): args.bbox_loss_coef,
                                    'frame_{}_aux{}_loss_giou'.format(i, j): args.giou_loss_coef,
                                    'frame_{}_aux{}_loss_angle'.format(i, j): 50.,
                                    })
            for j in range(args.dec_layers):
                weight_dict.update({"frame_{}_ps{}_loss_ce".format(i, j): args.cls_loss_coef,
                                    'frame_{}_ps{}_loss_bbox'.format(i, j): args.bbox_loss_coef,
                                    'frame_{}_ps{}_loss_giou'.format(i, j): args.giou_loss_coef,
                                    'frame_{}_aux{}_loss_angle'.format(i, j): 50.,
                                    })
    if args.memory_bank_type is not None and len(args.memory_bank_type) > 0:
        memory_bank = build_memory_bank(args, d_model, hidden_dim, d_model * 2)
        for i in range(num_frames_per_batch):
            weight_dict.update({"frame_{}_track_loss_ce".format(i): args.cls_loss_coef})
    else:
        memory_bank = None
    losses = ['labels', 'boxes', 'rotate']
    # losses = ['labels', 'boxes']
    criterion = ClipMatcher(num_classes, matcher=img_matcher, weight_dict=weight_dict, losses=losses)
    criterion.to(device)
    postprocessors = {}
    model = MOTR(
        backbone,
        transformer,
        track_embed=query_interaction_layer,
        num_feature_levels=args.num_feature_levels,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        criterion=criterion,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        memory_bank=memory_bank,
        use_checkpoint=args.use_checkpoint,
        query_denoise=args.query_denoise,
    )
    return model, criterion, postprocessors
