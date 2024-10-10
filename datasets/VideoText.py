# ------------------------------------------------------------------------
# Copyright (c) 2021 Zhejiang University-model. All Rights Reserved.
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
MOT dataset which returns image_id for evaluation.
"""
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.utils.data
import os.path as osp
from PIL import Image, ImageDraw
import copy
import datasets.transforms as T
from models.structures import Instances
import os
from datasets.data_tools import get_vocabulary
import mmcv
import math
from PIL import Image, ImageDraw, ImageFont
from util.box_ops import box_cxcywh_to_xyxy
import re
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
import pyclipper
from .config import State
import json

def cv2AddChineseText(image, text, position, textColor=(0, 0, 0), textSize=30):
    
    
    x1,y1 = position
    x2,y2 = len(text)* textSize/2 + x1, y1 + textSize
    
    points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
    mask_1 = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask_1, [points], 1)

    image,rgb = mask_image(image, mask_1,[255,255,255])
    
    
    if (isinstance(image, np.ndarray)):
        image = Image.fromarray(cv2.cvtColor(np.uint8(image.astype(np.float32)), cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    fontStyle = ImageFont.truetype(
        "./tools/simsun.ttc", textSize, encoding="utf-8")
    draw.text(position, text, textColor, font=fontStyle)
    
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                    
    return image

def mask_image(image, mask_2d, rgb=None, valid = False):
    h, w = mask_2d.shape

    # mask_3d = np.ones((h, w), dtype="uint8") * 255
    mask_3d_color = np.zeros((h, w, 3), dtype="uint8")
    # mask_3d[mask_2d[:, :] == 1] = 0
    
        
    image.astype("uint8")
    mask = (mask_2d!=0).astype(bool)
    if rgb is None:
        rgb = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
        
    mask_3d_color[mask_2d[:, :] == 1] = rgb
    image[mask] = image[mask] * 0.5 + mask_3d_color[mask] * 0.5
    
    if valid:
        mask_3d_color[mask_2d[:, :] == 1] = [[0,0,0]]
        kernel = np.ones((5,5),np.uint8)
        mask_2d = cv2.dilate(mask_2d,kernel,iterations = 4)
        mask = (mask_2d!=0).astype(bool)
        image[mask] = image[mask] * 0 + mask_3d_color[mask] * 1
        return image,rgb
        
    return image,rgb


def get_rotate_mat(theta):
    '''positive theta value means rotate clockwise'''
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


class DetMOTDetection:
    def __init__(self, args, data_txt_path: str, seqs_folder, dataset2transform):
        self.args = args
        self.dataset2transform = dataset2transform
        self.num_frames_per_batch = max(args.sampler_lengths)
        self.sample_mode = args.sample_mode
        self.sample_interval = args.sample_interval
        self.vis = args.vis
        self.video_dict = {}
        self.shrink_ratio = 0.4
        self.thresh_min = 0.3
        self.thresh_max = 0.7
        self.min_text_size = 8
        self.shrink_ratio = 0.4
        self.F_object_id = 20000
        self.boxes = []
        with open(data_txt_path, 'r') as file:
            self.img_files = file.readlines()
            self.img_files = [osp.join(seqs_folder, x.strip()) for x in self.img_files]
            
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))
        self.img_files = self.img_files[:int(len(self.img_files))]
#         print(data_txt_path)
        if "BOVText" in data_txt_path:
            self.label_files = [(("/share/wuweijia/Data/VideoText/MOTR/BOVText/labels_with_ids/train" + x.split("/Frames")[1]).replace('.png', '.txt').replace('.jpg', '.txt'))
                            for x in self.img_files]
        if "VideoSynthText" in data_txt_path:
            self.label_files = [(x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt'))
                            for x in self.img_files]
        elif "SynthText" in data_txt_path:
            self.label_files = [(("/mmu-ocr/weijiawu/Data/VideoText/MOTR/SynthText/labels_with_ids/train" + x.split("/SynthText")[1]).replace('.png', '.txt').replace('.jpg', '.txt')) if len(x.split("/SynthText"))>1 else (x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt'))
                            for x in self.img_files]
        elif  "ICDAR15" in data_txt_path:
            self.label_files = [(x.replace('images', 'labels').replace('img','gt_img').replace('.png', '.txt').replace('.jpg', '.txt'))
                            for x in self.img_files]
        else:
            self.label_files = [(x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt'))
                            for x in self.img_files]
        
        # The number of images per sample: 1 + (num_frames - 1) * interval.
        # The number of valid samples: num_images - num_image_per_sample + 1.
        
        self.item_num = len(self.img_files) - (self.num_frames_per_batch - 1) * self.sample_interval

        self._register_videos()

        # video sampler.
        self.sampler_steps: list = args.sampler_steps
        self.lengths: list = args.sampler_lengths
        print("sampler_steps={} lenghts={}".format(self.sampler_steps, self.lengths))
        if self.sampler_steps is not None and len(self.sampler_steps) > 0:
            # Enable sampling length adjustment.
            assert len(self.lengths) > 0
            assert len(self.lengths) == len(self.sampler_steps) + 1
            for i in range(len(self.sampler_steps) - 1):
                assert self.sampler_steps[i] < self.sampler_steps[i + 1]
            self.item_num = len(self.img_files) - (self.lengths[-1] - 1) * self.sample_interval
            self.period_idx = 0
            self.num_frames_per_batch = self.lengths[0]
            self.current_epoch = 0
        if "ICDAR15" in data_txt_path:
            with open(os.path.join("/home/ubuntu/MOTRv2-trans/", "det_db_DBnet_train_IC.json")) as f:
                self.det_db = json.load(f)
        elif "DSText" in data_txt_path:
            with open(os.path.join("/home/ubuntu/MOTRv2-trans/", "det_db_DBnet_train_DS.json")) as f:
                self.det_db = json.load(f)
        # recognition  CHINESE LOWERCASE
        self.use_ctc = True
        self.voc, self.char2id, self.id2char = get_vocabulary('LOWERCASE', use_ctc=self.use_ctc)
        self.max_word_num = 50
        self.max_word_len = 32
        
        
        self.vis = False
        
    def _register_videos(self):
        for label_name in self.label_files:
            video_name = '/'.join(label_name.split('/')[:-1])
            if video_name not in self.video_dict:
                print("register {}-th video: {} ".format(len(self.video_dict) + 1, video_name))
                self.video_dict[video_name] = len(self.video_dict)
                # assert len(self.video_dict) <= 300
    
        
    def _increase(self,box):
        cx, cy, w, h = box

        new_width = 1.5 * w
        new_height = 1.5 * h

        new_x_min = cx - new_width / 2
        new_y_min = cy - new_height / 2
        new_x_max = cx + new_width / 2
        new_y_max = cy + new_height / 2

        return new_x_min,new_y_min,new_x_max,new_y_max
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.sampler_steps is None or len(self.sampler_steps) == 0:
            # fixed sampling length.
            return

        for i in range(len(self.sampler_steps)):
            if epoch >= self.sampler_steps[i]:
                self.period_idx = i + 1
        print("set epoch: epoch {} period_idx={}".format(epoch, self.period_idx))
        self.num_frames_per_batch = self.lengths[self.period_idx]

    def step_epoch(self):
        # one epoch finishes.
        print("Dataset: epoch {} finishes".format(self.current_epoch))
        self.set_epoch(self.current_epoch + 1)

    @staticmethod
    def _targets_to_instances(self,targets: dict, img_shape) -> Instances:
        gt_instances = Instances(tuple(img_shape))
        n_gt = len(targets['labels'])
        gt_instances.boxes = targets['boxes'][:n_gt]
        gt_instances.labels = targets['labels']
        gt_instances.rotate = targets['rotate']
        gt_instances.texts_ignored = targets['texts_ignored']
        gt_instances.word = targets['word']
        gt_instances.obj_ids = targets['obj_ids']
        gt_instances.area = targets['area'][:n_gt]
        return gt_instances
    
    # def _targets_to_instances(self, targets: dict, img_shape) -> Instances:
    #     gt_instances = Instances(tuple(img_shape))
    #     n_gt = len(targets['boxes']) # 245
    #     n_ga = len(targets['labels']) #51
    #     gt_instances.boxes = targets['boxes']

    #     gt_instances.labels = torch.zeros(n_gt, dtype=torch.int64)
    #     gt_instances.labels[:n_ga] = targets['labels']

    #     gt_instances.rotate = torch.zeros(n_gt, dtype=torch.float32)
    #     gt_instances.rotate[:n_ga] = targets['rotate']

    #     gt_instances.texts_ignored = torch.ones(n_gt, dtype=torch.long)
    #     gt_instances.texts_ignored[:n_ga] = targets['texts_ignored']

    #     gt_instances.obj_ids = torch.ones((n_gt), dtype=torch.int64)*-1
    #     gt_instances.obj_ids[:n_ga] = targets['obj_ids']
        
    #     if len(self.boxes) != 0:
    #         for i, box1 in enumerate(targets['boxes'][n_ga:]):
    #             x1_box1, y1_box1, x2_box1, y2_box1 = self._increase(box1)
    #             area_box1 = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
    #             if self.boxes is not None:
    #                 for k,box2 in enumerate(self.boxes):
    #                     x1_box2, y1_box2, x2_box2, y2_box2 = self._increase(box2)
    #                     area_box2 = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)

    #                     x1_intersection = max(x1_box1, x1_box2)
    #                     y1_intersection = max(y1_box1, y1_box2)
    #                     x2_intersection = min(x2_box1, x2_box2)
    #                     y2_intersection = min(y2_box1, y2_box2)
    #                     intersection_area = max(0, x2_intersection - x1_intersection) * max(0, y2_intersection - y1_intersection)
    #                     iou = intersection_area / (area_box1 + area_box2 - intersection_area)

    #                     if iou>0.4 and self.obj_id[k]>0:
    #                         if self.obj_id[k] not in gt_instances.obj_ids:
    #                             gt_instances.obj_ids[n_ga + i] = self.obj_id[k]
    #                             break
                        
    #     for j,val in enumerate(gt_instances.obj_ids[n_ga:]):
    #         if val == -1:
    #             gt_instances.obj_ids[n_ga + j] = self.F_object_id
    #             self.F_object_id = self.F_object_id +1


    #     gt_instances.area = targets['area']

    #     #存放上一帧box信息
    #     self.boxes = gt_instances.boxes[n_ga:]
    #     self.obj_id = gt_instances.obj_ids[n_ga:]

    #     return gt_instances

    def _pre_single_frame(self, idx: int):
        img_path = self.img_files[idx]
        label_path = self.label_files[idx]
        #print(img_path)
        img = Image.open(img_path).convert("RGB")
        targets = {}
        w, h = img._size
        assert w > 0 and h > 0, "invalid image {} with shape {} {}".format(img_path, w, h)
        if osp.isfile(label_path):
            size = os.path.getsize(label_path)

            if size == 0:
                labels = np.array([])
            else:
#                 labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 7)
                labels0 = []
                words = []
                lines = mmcv.list_from_file(label_path)
                
                bboxes = []
                texts = []
                texts_ignored = []  
                for i,line in enumerate(lines):  #最大不超过五十个字
                    if i> self.max_word_num:
                        break
                    text = line.strip('\ufeff').strip('\xef\xbb\xbf').strip().split(' ')

                    # recognition words
                    if "#" in text[-1]:
#                         continue
                        texts_ignored.append(0)
                    else:
                        texts_ignored.append(1)
                    try:
                        labels0.append(list(map(float, text[:11])))
                    except:
                        print(text)
                    
                    if ("#" not in text[-1]) and ('ICDAR2015' in img_path):
                        word = text[0].split(',')[-1].strip()
                        #print(word)
                    
                    else: word = text[-1]

                    word = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])","",word.lower())
                    word = word.lower()
                    gt_word = np.full((self.max_word_len,), self.char2id['PAD'], dtype=np.int)
                    for j, char in enumerate(word):
                        if j > self.max_word_len - 1:
                            break
                        if char in self.char2id:
                            gt_word[j] = self.char2id[char]
                        else:
                            gt_word[j] = self.char2id['UNK']
                    if not self.use_ctc:
                        if len(word) > self.max_word_len - 1:
                            gt_word[-1] = self.char2id['EOS']
                        else:
                            gt_word[len(word)] = self.char2id['EOS']
                    words.append(gt_word)
                 
                if len(labels0)==0:
                    labels = np.array([])
                else:
                    # normalized cewh to pixel xyxy format 将中心点+wh格式转换为xyxy格式
                    labels0 = np.array(labels0)
                    labels = np.array(labels0).copy()
                    labels[:, 2] = w * (labels0[:, 2] - labels0[:, 4] / 2)
                    labels[:, 3] = h * (labels0[:, 3] - labels0[:, 5] / 2)
                    labels[:, 4] = w * (labels0[:, 2] + labels0[:, 4] / 2)
                    labels[:, 5] = h * (labels0[:, 3] + labels0[:, 5] / 2)
        else:
            raise ValueError('invalid label path: {}'.format(label_path))
        
               
        video_name = '/'.join(label_path.split('/')[:-1])
        obj_idx_offset = self.video_dict[video_name] * 1000000  # 1000000 unique ids is enough for a video.
        if 'crowdhuman' in img_path:
            targets['dataset'] = 'CrowdHuman'
        elif 'MOT17' in img_path:
            targets['dataset'] = 'MOT17'
        elif 'ICDAR2015' in img_path:
            targets['dataset'] = 'ICDAR2015'
        elif 'ICDAR15' in img_path:
            targets['dataset'] = 'ICDAR15_Pre'
        elif 'COCOTextV2' in img_path:
            targets['dataset'] = 'COCOTextV2'
        elif 'VideoSynthText' in img_path:
            targets['dataset'] = 'VideoSynthText'
        elif 'FlowTextV2' in img_path:
            targets['dataset'] = 'FlowTextV2'
        elif 'FlowText' in img_path:
            targets['dataset'] = 'FlowText'
        elif 'SynthText' in img_path:
            targets['dataset'] = 'SynthText'
        elif 'VISD' in img_path:
            targets['dataset'] = 'VISD'
        elif 'YVT' in img_path:
            targets['dataset'] = 'YVT'
        elif 'UnrealText' in img_path:
            targets['dataset'] = 'UnrealText'
        elif 'BOVTextV2' in img_path:
            targets['dataset'] = 'BOVText'
        elif 'FlowImage' in img_path:
            targets['dataset'] = 'FlowImage'
        elif 'DSText' in img_path:
            targets['dataset'] = 'DSText'
        else:
            raise NotImplementedError()
        targets['boxes'] = []
        # targets['proposal_boxes'] = []
        targets['img_path'] = img_path
        targets['rotate'] = []
        targets['area'] = []
        targets['iscrowd'] = []
        targets['labels'] = []
        targets['upright_box'] = []
        targets['word'] = []
        targets['texts_ignored'] = []
        targets['obj_ids'] = []
        targets['scores'] = []
        targets['image_id'] = torch.as_tensor(idx)
        targets['size'] = torch.as_tensor([h, w])
        targets['orig_size'] = torch.as_tensor([h, w])

        for i,label in enumerate(labels):
            targets['rotate'].append(label[6])
            targets['boxes'].append(label[2:6].tolist())
            targets['area'].append(label[4] * label[5])
            targets['upright_box'].append(label[7:].tolist())
            targets['word'].append(words[i])
            targets['texts_ignored'].append(texts_ignored[i])
            targets['iscrowd'].append(0)
            targets['labels'].append(0)
            targets['scores'].append(1.)

            obj_id = label[1] + obj_idx_offset if label[1] >= 0 else label[1]
            targets['obj_ids'].append(obj_id)  # relative id
        
        if 'DSText' in label_path:
            txt_key = label_path.replace("labels_with_ids/train", "images/test")
            pattern = '/home/ubuntu/MOTRv2-trans/'
            txt_key = re.sub(pattern, '', txt_key)
        elif 'ICDAR2015' in label_path:
            txt_key = label_path.replace("/home/ubuntu/TransDETR-main/", "").replace("labels/track/train", "images/test")
            pattern = '/home/ubuntu/MOTRv2-trans/'
            txt_key = re.sub(pattern, '', txt_key)

#两点宽高
        for line in self.det_db[txt_key]:
            point_x, point_y, point_w, point_h, point_s = map(float, line.split(',')) #分别是左上角坐标xy和宽高wh
            box=[point_x,point_y,point_x+point_w,point_y+point_h] #转化为左上角坐标和右下角坐标
            targets['boxes'].append(box)
            targets['area'].append(point_w*point_h)
            targets['scores'].append(point_s)
#四点
        # for line in self.det_db[txt_key]:
        #     point_x_1, point_y_1,point_x_2, point_y_2,point_x_3, point_y_3,point_x_4, point_y_4,point_s = map(float, line.split(',')) #分别是左上角坐标xy和宽高wh
            
        #     point_x = [point_x_1, point_x_2, point_x_3, point_x_4]
        #     point_y = [point_y_1, point_y_2, point_y_3, point_y_4]
        #     left_bottom_x = min(point_x)
        #     left_bottom_y = min(point_y)
        #     right_top_x = max(point_x)
        #     right_top_y = max(point_y)

        #     box=[left_bottom_x, left_bottom_y,right_top_x, right_top_y] #转化为左上角坐标和右下角坐标
        #     width = max(point_x_1, point_x_2, point_x_3, point_x_4) - min(point_x_1, point_x_2, point_x_3, point_x_4)
        #     height = max(point_y_1, point_y_2, point_y_3, point_y_4) - min(point_y_1, point_y_2, point_y_3, point_y_4)
        #     area = width * height
        #     targets['boxes'].append(box)
        #     targets['area'].append(area)
        #     targets['scores'].append(point_s)



        # for line in self.det_db[txt_key]:
        #     point_x, point_y, point_w, point_h, point_s = map(float, line.split(',')) #分别是左上角坐标xy和宽高wh
        #     box=[point_x,point_y,point_x+point_w,point_y+point_h] #转化为左上角坐标和右下角坐标
        #     flag = 0
        #     for ex_box in targets["boxes"]:
        #         iou = self.calculate_iou(box, ex_box)
        #         if iou >0.5:
        #             flag = 1
        #             break
        #     if flag == 0:
        #         targets['boxes'].append(box)
        #         targets['area'].append(point_w*point_h)
        #         targets['scores'].append(point_s)


        
        # for line in self.det_db['data/ICDAR2015/images/test/Video_25_5_2/1.txt']:
        #     point_x, point_y, point_w, point_h, point_s = map(float, line.split(',')) #分别是左上角坐标xy和宽高wh
        #     box = [point_x, point_y, point_x + point_w, point_y + point_h] #转化为左上角坐标和右下角坐标
        #     best_iou = 0.5  # 初始化最佳IOU
        #     best_match_index = -1  # 初始化最佳匹配的目标框索引

        #     for i, ex_box in enumerate(targets["boxes"]):
        #         iou = self.calculate_iou(box, ex_box)
        #         if iou > best_iou:
        #             best_iou = iou
        #             best_match_index = i

        #     if best_match_index != -1:
        #         # 将最佳匹配的目标框添加到targets['boxes']中
        #         targets['boxes'].append(targets["boxes"][best_match_index])
        #         # targets['area'].append(point_w * point_h)
        #         targets['scores'].append(point_s)
#######################################创新点1 筛选并取出query###################################################################################
        # mid_box = []
        # mid_score = []
        # for ex_box in targets["boxes"]:
        #     best_iou = 0.5  # 初始化最佳IOU
        #     best_match_index = -1  # 初始化最佳匹配的目标框索引
        #     best_box = None
        #     for i,line in enumerate(self.det_db[txt_key]):
        #         point_x, point_y, point_w, point_h, point_s = map(float, line.split(',')) #分别是左上角坐标xy和宽高wh
        #         box = [point_x, point_y, point_x + point_w, point_y + point_h] #转化为左上角坐标和右下角坐标
        #         iou = self.calculate_iou(box, ex_box)
        #         if iou > best_iou:
        #             best_iou = iou
        #             best_match_index = i
        #             best_box = box
        #             best_score = point_s
                    
        #     if best_match_index != -1:
        #         mid_box.append(best_box)
        #         mid_score.append(best_score)
        # for b,s in zip(mid_box,mid_score):
        #     targets["boxes"].append(b)
        #     targets["scores"].append(s)
############################################################################################################################
        targets['area'] = torch.as_tensor(targets['area'])
        targets['iscrowd'] = torch.as_tensor(targets['iscrowd'])
        targets['labels'] = torch.as_tensor(targets['labels'], dtype=torch.int64)
        targets['obj_ids'] = torch.as_tensor(targets['obj_ids'])
        targets['word'] = torch.as_tensor(targets['word'], dtype=torch.long)
        targets['texts_ignored'] = torch.as_tensor(targets['texts_ignored'], dtype=torch.long)
        targets['rotate'] = torch.as_tensor(targets['rotate'], dtype=torch.float32)
        targets['boxes'] = torch.as_tensor(targets['boxes'], dtype=torch.float32).reshape(-1, 4)
        # targets['proposal_boxes'] = torch.as_tensor(targets['proposal_boxes'], dtype=torch.float32).reshape(-1, 8)
        targets['boxes'][:, 0::2].clamp_(min=0, max=w)
        targets['boxes'][:, 1::2].clamp_(min=0, max=h)
        # targets['proposal_boxes'][:, 0::2].clamp_(min=0, max=w)
        # targets['proposal_boxes'][:, 1::2].clamp_(min=0, max=h)
        targets['scores'] = torch.as_tensor(targets['scores'])
        
        targets['upright_box'] = torch.as_tensor(targets['upright_box'], dtype=torch.float32).reshape(-1, 4)
        targets['upright_box'][:, 0::2].clamp_(min=0, max=w)
        targets['upright_box'][:, 1::2].clamp_(min=0, max=h)
        return img, targets

    def _get_sample_range(self, start_idx):

        # take default sampling method for normal dataset.
        assert self.sample_mode in ['fixed_interval', 'random_interval'], 'invalid sample mode: {}'.format(self.sample_mode)
        if self.sample_mode == 'fixed_interval':
            sample_interval = self.sample_interval
        elif self.sample_mode == 'random_interval':
            sample_interval = np.random.randint(1, self.sample_interval + 1)
        default_range = start_idx, start_idx + (self.num_frames_per_batch - 1) * sample_interval + 1, sample_interval
#         print(self.num_frames_per_batch)
        return default_range
    
    def calculate_iou(self, box1, box2):
        """
        计算两个文字框的IoU（交并比）
        box1 和 box2 分别是左上角和右下角坐标的元组 (x1, y1, x2, y2)
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x_overlap = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
        y_overlap = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
        intersection_area = x_overlap * y_overlap
        
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou

    def pre_continuous_frames(self, start, end, interval=1):
        targets = []
        images = []
        for i in range(start, end, interval):
            img_i, targets_i = self._pre_single_frame(i)
            images.append(img_i)
            targets.append(targets_i)
        return images, targets

    def validate_polygons(self, polygons, ignore_tags, h, w):
        '''
        polygons (numpy.array, required): of shape (num_instances, num_points, 2)
        '''
        if len(polygons) == 0:
            return polygons, ignore_tags
        assert len(polygons) == len(ignore_tags)
        for polygon in polygons:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        for i in range(len(polygons)):
            area = self.polygon_area(polygons[i])
            if abs(area) < 1:
                ignore_tags[i] = True
            if area > 0:
                polygons[i] = polygons[i][::-1, :]
        return polygons, ignore_tags
    
    def polygon_area(self, polygon):
        edge = 0
        for i in range(polygon.shape[0]):
            next_index = (i + 1) % polygon.shape[0]
            edge += (polygon[next_index, 0] - polygon[i, 0]) * (polygon[next_index, 1] + polygon[i, 1])

        return edge / 2.
    
    def draw_border_map(self, polygon, canvas, mask):
        polygon = np.array(polygon)
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2

        polygon_shape = Polygon(polygon)
        distance = polygon_shape.area * \
            (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(l) for l in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND,
                        pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon = np.array(padding.Execute(distance)[0])
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin

        xs = np.broadcast_to(
            np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(
            np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

        distance_map = np.zeros(
            (polygon.shape[0], height, width), dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = self.distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = distance_map.min(axis=0)

        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
            1 - distance_map[
                ymin_valid-ymin:ymax_valid-ymax+height,
                xmin_valid-xmin:xmax_valid-xmax+width],
            canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])

    def distance(self, xs, ys, point_1, point_2):
        '''
        compute the distance from point to a line
        ys: coordinates in the first axis
        xs: coordinates in the second axis
        point_1, point_2: (x, y), the end of the line
        '''
        height, width = xs.shape[:2]
        square_distance_1 = np.square(
            xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(
            xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(
            point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

        cosin = (square_distance - square_distance_1 - square_distance_2) / \
            (2 * np.sqrt(square_distance_1 * square_distance_2))
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_distance_1 * square_distance_2 *
                         square_sin / square_distance)

        result[cosin < 0] = np.sqrt(np.fmin(
            square_distance_1, square_distance_2))[cosin < 0]
        # self.extend_line(point_1, point_2, result)
        return result

    def extend_line(self, point_1, point_2, result):
        ex_point_1 = (int(round(point_1[0] + (point_1[0] - point_2[0]) * (1 + self.shrink_ratio))),
                      int(round(point_1[1] + (point_1[1] - point_2[1]) * (1 + self.shrink_ratio))))
        cv2.line(result, tuple(ex_point_1), tuple(point_1),
                 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
        ex_point_2 = (int(round(point_2[0] + (point_2[0] - point_1[0]) * (1 + self.shrink_ratio))),
                      int(round(point_2[1] + (point_2[1] - point_1[1]) * (1 + self.shrink_ratio))))
        cv2.line(result, tuple(ex_point_2), tuple(point_2),
                 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
        return ex_point_1, ex_point_2

    def __getitem__(self, idx):
        sample_start, sample_end, sample_interval = self._get_sample_range(idx)
        
        images, targets = self.pre_continuous_frames(sample_start, sample_end, sample_interval)
        w, h = images[0].size
        data = {}
        dataset_name = targets[0]['dataset']
        transform = self.dataset2transform[dataset_name]

        if transform is not None:
            images, targets = transform(images, targets)  #随机裁剪功能被删去了 xyxy转化为cxcywh
        
        self.vis = False  # 是否查看GT
        if self.vis:
            import random
            image_icdxx = random.randint(1,100)
            for idx1,(img,ann) in enumerate(zip(images,targets)):
                imge = img.permute(1,2,0)
                imge = (imge.cpu().numpy()*[0.229, 0.224, 0.225]+[0.485, 0.456, 0.406])*255
                # imge = (imge.cpu().numpy()+[0.485, 0.456, 0.406])*255
                h_1, w_1 = imge.shape[:2]
                label = ann["labels"]
                area = ann["area"]
                iscrowd = ann["iscrowd"]
                obj_ids = ann["obj_ids"]
                words = ann["word"]
                texts_ignoreds = ann["texts_ignored"]
                boxes = ann["boxes"] * torch.tensor([w_1, h_1, w_1, h_1], dtype=torch.float32)
                n_gt = boxes.shape[0]
                n_ga = label.shape[0]
                rotates = torch.zeros(n_gt, dtype=torch.float32)
                rotates[:n_ga] = ann['rotate']
                image = imge.copy()
                boxes = box_cxcywh_to_xyxy(boxes)
                img_path = ann['img_path']
                save_path = "./show_GTandProposal/" + img_path.split('/')[-2] 
                if not os.path.exists(save_path):
                    # 创建文件夹
                    os.makedirs(save_path)
                i = 0
                for box,rotate in zip(boxes,rotates):
                    x_min,y_min,x_max,y_max = box.cpu().numpy()
#                     print(x_min,y_min,x_max,y_max)
                    rotate_mat = get_rotate_mat(-rotate)
                    temp_x = np.array([[x_min, x_max, x_max, x_min]]) - (x_min+x_max)/2
                    temp_y = np.array([[y_min, y_min, y_max, y_max]]) - (y_min+y_max)/2
                    coordidates = np.concatenate((temp_x, temp_y), axis=0)
                    res = np.dot(rotate_mat, coordidates)
                    res[0,:] += (x_min+x_max)/2
                    res[1,:] += (y_min+y_max)/2
                    points = np.array([[res[0,0], res[1,0]], [res[0,1], res[1,1]], [res[0,2], res[1,2]], [res[0,3], res[1,3]]], np.int32)
                    if i < n_ga:
                        cv2.polylines(image, [points], True, (0,0,255), thickness=1) # 红色
                    elif i >= n_ga:
                        cv2.polylines(image, [points], True, (0,255,255), thickness=1) # 黄色

                     
                    short_side = min(imge.shape[0],imge.shape[1])
                    text_size = int(short_side * 0.05)
                    i=i+1
                cv2.imwrite(save_path + '/' + img_path.split('/')[-1],image)


        gt_instances = []
        proposals = []
        img_paths = []
        self.boxes = []
        for img_i, targets_i in zip(images, targets):
            gt_instances_i = self._targets_to_instances(self,targets_i, img_i.shape[1:3])
            gt_instances.append(gt_instances_i)
            img_path = targets_i['img_path']
            img_paths.append(img_path)

            n_gt = len(targets_i['labels'])
            proposals.append(torch.cat([
                targets_i['boxes'][n_gt:],
                targets_i['scores'][n_gt:, None],
            ], dim=1))
            

        self.boxes = []
        data.update({
            'imgs': images,
            'gt_instances': gt_instances,
            'proposals': proposals,
            'img_path':img_paths
        })
        if self.args.vis:
            data['ori_img'] = [target_i['ori_img'] for target_i in targets]
        return data

    def __len__(self):
        return self.item_num


class DetMOTDetectionValidation(DetMOTDetection):
    def __init__(self, args, seqs_folder, dataset2transform):
        args.data_txt_path = args.val_data_txt_path
        super().__init__(args, seqs_folder, dataset2transform)



def make_transforms_for_mot17(image_set, args=None):

    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]

    if image_set == 'train':
        return T.MotCompose([
#             T.MotRandomHorizontalFlip(),
#             T.MotRandomResize(scales, max_size=1536),
            T.MotRandomSelect(
                T.MotRandomResize(scales, max_size=1536),
                T.MotCompose([
                    T.MotRandomResize([400, 500, 600]),
                    T.FixedMotRandomCrop(384, 600),
                    T.MotRandomResize(scales, max_size=1536),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.MotCompose([
            T.MotRandomResize([800], max_size=1536),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def make_transforms_for_BOVText(image_set, args=None):

    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]

    if image_set == 'train':
        return T.MotCompose([
#             T.MotRandomHorizontalFlip(),
            T.MotRandomResize(scales, max_size=1536),
#             T.MotRandomSelect(
#                 T.MotRandomResize(scales, max_size=1536),
#                 T.MotCompose([
#                     T.MotRandomResize([400, 500, 600]),
#                     T.FixedMotRandomCrop(384, 600),
#                     T.MotRandomResize(scales, max_size=1536),
#                 ])
#             ),
            normalize,
        ])

    if image_set == 'val':
        return T.MotCompose([
            T.MotRandomResize([800], max_size=1536),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def make_transforms_for_crowdhuman(image_set, args=None):

    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]
#     scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896]
    if image_set == 'train':
        return T.MotCompose([
#             T.MotRandomHorizontalFlip(),
            T.FixedMotRandomShift(bs=1),
#             T.MotRandomResize(scales, max_size=1536),
            T.MotRandomSelect(
                T.MotRandomResize(scales, max_size=1536),
                T.MotCompose([
                    T.MotRandomResize([400, 500, 600]),
                    T.FixedMotRandomCrop(384, 600),
                    T.MotRandomResize(scales, max_size=1536),
                ])
            ),
            normalize,

        ])

    if image_set == 'val':
        return T.MotCompose([
            T.MotRandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build_dataset2transform(args, image_set):
    mot17_train = make_transforms_for_mot17('train', args)
    mot17_test = make_transforms_for_mot17('val', args)
    BOVText_train = make_transforms_for_BOVText('train', args)

    crowdhuman_train = make_transforms_for_crowdhuman('train', args)
    dataset2transform_train = {'ICDAR2015': mot17_train,'DSText': mot17_train,  'VideoSynthText':mot17_train, 'FlowText':mot17_train, 'FlowTextV2':mot17_train,
                       'YVT': mot17_train,
                       'BOVText': BOVText_train, 'SynthText': crowdhuman_train, 'UnrealText': crowdhuman_train,
                       'COCOTextV2': crowdhuman_train, 'VISD': crowdhuman_train, 'FlowImage': crowdhuman_train,
                        "ICDAR15_Pre": crowdhuman_train}
    
    dataset2transform_val = {'ICDAR2015': mot17_test,'DSText': mot17_test, 'VideoSynthText':mot17_test, 'FlowText':mot17_test, 'YVT': mot17_test,
                     'FlowTextV2':mot17_test,
                     'BOVText': mot17_test, 'SynthText': mot17_test, 'UnrealText': mot17_test,
                     'COCOTextV2': mot17_test,'VISD': mot17_test,'FlowImage': mot17_test,"ICDAR15_Pre": crowdhuman_train}
    if image_set == 'train':
        return dataset2transform_train
    elif image_set == 'val':
        return dataset2transform_val
    else:
        raise NotImplementedError()


def build(image_set, args):
    root = Path(args.mot_path)
    assert root.exists(), f'provided MOT path {root} does not exist'
    dataset2transform = build_dataset2transform(args, image_set)
    if image_set == 'train':
        data_txt_path = args.data_txt_path_train
        dataset = DetMOTDetection(args, data_txt_path=data_txt_path, seqs_folder=root, dataset2transform=dataset2transform)
    if image_set == 'val':
        data_txt_path = args.data_txt_path_val
        dataset = DetMOTDetection(args, data_txt_path=data_txt_path, seqs_folder=root, dataset2transform=dataset2transform)
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser('DETR training and evaluation script')
    args = parser.parse_args()
    
    data_txt_path = ""
    root = ""
    dataset2transform = ""
    dataset = DetMOTDetection(args, data_txt_path=data_txt_path, seqs_folder=root, dataset2transform=dataset2transform)

    
