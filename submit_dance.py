# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from copy import deepcopy
import json
import numpy as np
import os
import argparse
import torchvision.transforms.functional as F
import torch
import cv2
from tqdm import tqdm
from PIL import Image, ImageDraw
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from models import build_model
from util.tool import load_model
from main import get_args_parser
from xml.dom.minidom import Document
from collections import OrderedDict
from models.structures import Instances
from torch.utils.data import Dataset, DataLoader
from util.utils import write_result_as_txt,debug, setup_logger,write_lines,MyEncoder
try:
    import xml.etree.cElementTree as ET  #解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET
import math

# def draw_bboxes(ori_img, bbox, scores,identities=None, cvt_color=False, rgbs=None):
#     if cvt_color:
#         ori_img = cv2.cvtColor(np.asarray(ori_img), cv2.COLOR_RGB2BGR)
#     img = ori_img
#     for i, box in enumerate(bbox):
#         x1, y1, x2, y2, x3, y3, x4, y4 = [int(i) for i in box[:8]]
        
#         points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
#         mask_1 = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
#         cv2.fillPoly(mask_1, [points], 1)
        
#         ID = int(identities[i]) if identities is not None else 0

#         score = str(np.array(scores[i]))[:4]

#         img,rgb = mask_image(img, mask_1)

#         r,g,b = rgb[0]
#         r,g,b = int(r),int(g),int(b)
#         cv2.polylines(img, [points], True, (r,g,b), thickness=4)
#         # img=cv2AddChineseText(img,str(ID), (int(x1), int(y1) - 20),((0,0,255)), 45)
# #         print(word)
#         short_side = min(img.shape[0],img.shape[1])
#         text_size = int(short_side * 0.03)
        
#         img=cv2AddChineseText(img, str(ID)+"|"+score, (int(x1), int(y1) - text_size),((255,255,255)), text_size)
#     return img
def draw_bboxes(ori_img, bbox, scores,identities=None, cvt_color=False, rgbs=None):
    if cvt_color:
        ori_img = cv2.cvtColor(np.asarray(ori_img), cv2.COLOR_RGB2BGR)
    img = ori_img
    for i, box in enumerate(bbox):
        x1, y1, x2, y2, x3, y3, x4, y4 = [int(i) for i in box[:8]]
        
        points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
        mask_1 = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask_1, [points], 1)
        
        ID = int(identities[i]) if identities is not None else 0
        # word = words[i]
        score = str(np.array(scores[i]))[:4]
        if ID in rgbs:
            img,rgb = mask_image(img, mask_1,rgbs[ID])
        else:
            img,rgb = mask_image(img, mask_1)
            rgbs[ID] = rgb
        r,g,b = rgb[0]
        r,g,b = int(r),int(g),int(b)
        cv2.polylines(img, [points], True, (r,g,b), thickness=4)
#         img=cv2AddChineseText(img,str(ID), (int(x1), int(y1) - 20),((0,0,255)), 45)
#         print(word)
        short_side = min(img.shape[0],img.shape[1])
        text_size = int(short_side * 0.03)
        
        img=cv2AddChineseText(img, str(ID)+"|"+score, (int(x1), int(y1) - text_size),((255,255,255)), text_size)
    return img



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
        image[mask] = image[mask] * 0.5 + mask_3d_color[mask] * 0.5
        return image,rgb
        
    return image,rgb

def is_chinese(string):
    """
    检查整个字符串是否包含中文
    :param string: 需要检查的字符串
    :return: bool
    """
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True

    return False

def cv2AddChineseText(image, text, position, textColor=(0, 0, 0), textSize=30):
    
    
    x1,y1 = position
    x2,y2 = len(text)* textSize/2 + x1, y1 + textSize
    if is_chinese(text):
        x2,y2 = len(text)* textSize + x1, y1 + textSize
        
    points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
    mask_1 = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask_1, [points], 1)

    image,rgb = mask_image_bg(image, mask_1, rgb = [0,0,0])
    
    
    if (isinstance(image, np.ndarray)):  # 判断是否OpenCV图片类型
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(image)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "./tools/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    # 转换回OpenCV格式
                    
    return image


def mask_image_bg(image, mask_2d, rgb=None, valid = False):
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
        image[mask] = image[mask] * 0.5 + mask_3d_color[mask] * 0.5
        return image,rgb
        
    return image,rgb

def get_rotate_mat(theta):
	'''positive theta value means rotate clockwise'''
	return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

class ListImgDataset(Dataset):
    def __init__(self, mot_path, img_list, det_db) -> None:
        super().__init__()
        self.mot_path = mot_path
        self.img_list = img_list
        self.det_db = det_db

        '''
        common settings
        '''
        self.img_height = 800
        self.img_width = 1536
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def load_img_from_file(self, f_path):

        #icdar
        if 'ICDAR' in f_path:
            cur_img = cv2.imread(f_path)
        
        # DSText
        else:
            cur_img = cv2.imread(os.path.join(self.mot_path, f_path))

        assert cur_img is not None, f_path
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
        proposals = []
        im_h, im_w = cur_img.shape[:2]
        if 'DSText' in self.mot_path:
            for line in self.det_db['data/DSText/images/test/' + f_path[:-4] + '.txt']:
                # l, t, w, h, s = list(map(float, line.split(',')))
                # proposals.append([(l + w / 2) / im_w,
                #                     (t + h / 2) / im_h,
                #                     w / im_w,
                #                     h / im_h,
                #                     s])
                x1, y1, x2, y2, x3, y3, x4, y4, s = list(map(float, line.split(',')))
                cx = (x1 + x2 + x3 + x4) / 4
                cy = (y1 + y2 + y3 + y4) / 4
                w = max(x1, x2, x3, x4) - min(x1, x2, x3, x4)
                h = max(y1, y2, y3, y4) - min(y1, y2, y3, y4)
                proposals.append([cx / im_w, cy / im_h, w / im_w, h / im_h, s])
        else:
            for line in self.det_db['data/ICDAR2015/images/test/' + f_path.split("val/")[1][:-4] + '.txt']:
                l, t, w, h, s = list(map(float, line.split(',')))
                proposals.append([(l + w / 2) / im_w,
                                    (t + h / 2) / im_h,
                                    w / im_w,
                                    h / im_h,
                                    s])
        return cur_img, torch.as_tensor(proposals).reshape(-1, 5)

    def init_img(self, img, proposals):
        ori_img = img.copy()
        self.seq_h, self.seq_w = img.shape[:2]
        scale = self.img_height / min(self.seq_h, self.seq_w)
        # scale = 1
        if max(self.seq_h, self.seq_w) * scale > self.img_width:
            # scale = 1
            scale = self.img_width / max(self.seq_h, self.seq_w)
        target_h = int(self.seq_h * scale)
        target_w = int(self.seq_w * scale)
        img = cv2.resize(img, (target_w, target_h))
        img = F.normalize(F.to_tensor(img), self.mean, self.std)
        img = img.unsqueeze(0)
        return img, ori_img, proposals

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img, proposals = self.load_img_from_file(self.img_list[index])
        # print(self.img_list[index])
        return self.init_img(img, proposals)
    

def draw_points(img: np.ndarray, points: np.ndarray, color=(255, 255, 255)) -> np.ndarray:
    assert len(points.shape) == 2 and points.shape[1] == 2, 'invalid points shape: {}'.format(points.shape)
    for i, (x, y) in enumerate(points):
        if i >= 300:
            color = (0, 255, 0)
        cv2.circle(img, (int(x), int(y)), 2, color=color, thickness=2)
    return img

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


class Detector(object):
    def __init__(self, args, model, vid):
        self.args = args
        self.detr = model

        self.vid = vid
        self.seq_num = os.path.basename(vid)
        # img_list = os.listdir(os.path.join(self.args.mot_path, vid, 'img1'))
        # img_list = [os.path.join(vid, 'img1', i) for i in img_list if 'jpg' in i]
        
        # ICDAR
        if 'ICDAR' in vid:
            img_list = os.listdir(vid) 

        # DSText
        else:
            img_list = os.listdir(os.path.join(self.args.mot_path, vid)) 

        img_list = [os.path.join(vid, i) for i in img_list if 'jpg' in i]
        self.img_list = [os.path.join( self.vid, "{}.jpg".format(_)) for _ in range(1,len(img_list)+1)]

        # self.img_list = sorted(img_list)
        self.img_len = len(self.img_list)

        # self.predict_path = os.path.join(self.args.output_dir, args.exp_name)

        if 'ICDAR' in vid:
            self.seq_num = self.seq_num.replace("/","_")
            xmls = self.seq_num.split("_")
            xml_name = xmls[0].replace('V','v') + "_" + xmls[1]
        else:
            self.seq_num = self.seq_num.replace("/","_")
            xml_name = self.seq_num

        self.predict_path ="/home/ubuntu/MOTRv2-trans/result/eval/preds"
        os.makedirs(self.predict_path, exist_ok=True)
        self.predict_path = os.path.join("/home/ubuntu/MOTRv2-trans/result/eval/preds","res_{}.xml".format(xml_name))

        json_path = os.path.join("/home/ubuntu/MOTRv2-trans/result/eval", 'jons')
        os.makedirs(json_path, exist_ok=True)
        self.json_path = os.path.join(json_path,"{}.json".format(self.seq_num))


    @staticmethod
    def filter_dt_by_score(dt_instances: Instances, prob_threshold: float) -> Instances:
        keep = dt_instances.scores > prob_threshold
        keep &= dt_instances.obj_idxes >= 0
        return dt_instances[keep]

    @staticmethod
    def filter_dt_by_area(dt_instances: Instances, area_threshold: float) -> Instances:
        wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep = areas > area_threshold
        return dt_instances[keep]
    
    def visualize_img_with_bbox(self,img_path, img, dt_instances: Instances, ref_pts=None, gt_boxes=None,rgbs=None):
        img = img.numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        scores = dt_instances.scores.cpu()
        if dt_instances.has('scores'):
            img_show = draw_bboxes(img, dt_instances.boxes,scores = scores,identities = dt_instances.obj_idxes,rgbs=rgbs)
        
        # if ref_pts is not None:
        #     img_show = draw_points(img_show, ref_pts)

        cv2.imwrite(img_path, img_show)

    def to_rotated_rec(self,dt_instances: Instances, filter_word_score=0.5) -> Instances:
        boxes = []
        for box,angle in zip(dt_instances.boxes,dt_instances.rotate):
            x_min,y_min, x_max, y_max = [int(i) for i in box[:4]]
            rotate = angle
            rotate_mat = get_rotate_mat(-rotate)
            temp_x = np.array([[x_min, x_max, x_max, x_min]]) - (x_min+x_max)/2
            temp_y = np.array([[y_min, y_min, y_max, y_max]]) - (y_min+y_max)/2
            coordidates = np.concatenate((temp_x, temp_y), axis=0)
            res = np.dot(rotate_mat, coordidates)
            res[0,:] += (x_min+x_max)/2
            res[1,:] += (y_min+y_max)/2
            boxes.append(np.array([res[0,0], res[1,0], res[0,1], res[1,1], res[0,2], res[1,2],res[0,3], res[1,3]]))
        dt_instances.boxes = np.array(boxes)

        return dt_instances
    
    # def to_rotated_rec(self,dt_instances: Instances, filter_word_score=0.5) -> Instances:
    #     boxes = []
    #     for box in dt_instances.boxes:
    #         x_min,y_min, x_max, y_max = [int(i) for i in box[:4]]
    #         rotate = 0
    #         rotate_mat = get_rotate_mat(-rotate)
    #         temp_x = np.array([[x_min, x_max, x_max, x_min]]) - (x_min+x_max)/2
    #         temp_y = np.array([[y_min, y_min, y_max, y_max]]) - (y_min+y_max)/2
    #         coordidates = np.concatenate((temp_x, temp_y), axis=0)
    #         res = np.dot(rotate_mat, coordidates)
    #         res[0,:] += (x_min+x_max)/2
    #         res[1,:] += (y_min+y_max)/2
    #         boxes.append(np.array([res[0,0], res[1,0], res[0,1], res[1,1], res[0,2], res[1,2],res[0,3], res[1,3]]))
    #     dt_instances.boxes = np.array(boxes)

    #     return dt_instances
    

    def detect(self, prob_threshold=0.5, area_threshold=100, vis=False):
        total_dts = 0
        total_occlusion_dts = 0
        annotation = {}
        rgbs = {}

        track_instances = None
        
        if 'ICDAR2015' in self.vid:
            with open(f'/home/ubuntu/MOTRv2-trans/det_db_DBnet_test_IC.json') as f:
                det_db = json.load(f)
        else:
            with open(f'/home/ubuntu/MOTRv2-trans/det_db_DBnet_test_DS.json') as f:
                det_db = json.load(f)
        # 
        loader = DataLoader(ListImgDataset(self.args.mot_path, self.img_list, det_db), 1, num_workers=2)
        # loader = DataLoader(ListImgDataset('data/', self.img_list, det_db), 1, num_workers=2)
        lines = []
        for i, data in enumerate(tqdm(loader)):
            cur_img, ori_img, proposals = [d[0] for d in data]
            cur_img, proposals = cur_img.cuda(), proposals.cuda()

            # track_instances = None
            if track_instances is not None:
                track_instances.remove('boxes')
                track_instances.remove('labels')
            seq_h, seq_w, _ = ori_img.shape

            res = self.detr.inference_single_image(cur_img, (seq_h, seq_w), track_instances, proposals)
            track_instances = res['track_instances']

            all_ref_pts = tensor_to_numpy(res['ref_pts'][0, :, :2])
            dt_instances = deepcopy(track_instances)


            dt_instances = self.filter_dt_by_score(dt_instances, prob_threshold)
            dt_instances = self.filter_dt_by_area(dt_instances, area_threshold)
            dt_instances = self.to_rotated_rec(dt_instances)  # 字符编码识别为字符串

            total_dts += len(dt_instances)

            bbox_xyxy = dt_instances.boxes.tolist()
            identities = dt_instances.obj_idxes.tolist()
################################
            if not os.path.exists('result/eval/img/' + self.seq_num):
                os.makedirs('result/eval/img/' + self.seq_num)
            cur_vis_img_path = os.path.join('result/eval/img/'+self.seq_num, '{}.jpg'.format(i))
            gt_boxes = None
            self.visualize_img_with_bbox(cur_vis_img_path, ori_img, dt_instances, ref_pts=all_ref_pts,gt_boxes=gt_boxes,rgbs = rgbs)

################################
        #     save_format = '{frame},{id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n'
        #     for xyxy, track_id in zip(bbox_xyxy, identities):
        #         if track_id < 0 or track_id is None:
        #             continue
        #         x1, y1, x2, y2 = xyxy
        #         w, h = x2 - x1, y2 - y1
        #         lines.append(save_format.format(frame=i + 1, id=track_id, x1=x1, y1=y1, w=w, h=h))
        # with open(os.path.join(self.predict_path, f'{self.seq_num}.txt'), 'w') as f:
        #     f.writelines(lines)
        # print("totally {} dts {} occlusion dts".format(total_dts, total_occlusion_dts))

            boxes,IDs,scores = dt_instances.boxes, dt_instances.obj_idxes, dt_instances.scores
            lines = []
            for box,ID,score in zip(boxes,IDs,scores):
                # x1, y1, x2, y2 = [int(i) for i in box[:4]]
                # lines.append([x1, y1, x2, y1, x2, y2, x1, y2, int(ID),"0"])
                x1, y1, x2, y2, x3, y3, x4, y4 = [int(i) for i in box[:8]]
                lines.append([x1, y1, x2, y2, x3, y3, x4, y4,int(ID),"0"])
            annotation.update({str(i+1):lines})  
            # print(self.json_path,self.predict_path)
        Generate_Json_annotation(annotation,self.json_path,self.predict_path)

class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.6, filter_score_thresh=0.5, miss_tolerance=10):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        # self.miss_tolerance = 1
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

class StorageDictionary(object):
    @staticmethod
    def dict2file(file_name, data_dict):
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        # import pickle
        output = open(file_name,'wb')
        pickle.dump(data_dict,output)
        output.close()
        
    @staticmethod
    def file2dict(file_name):
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        # import pickle
        pkl_file = open(file_name, 'rb')
        data_dict = pickle.load(pkl_file)
        pkl_file.close()
        return data_dict
    
    @staticmethod
    def dict2file_json(file_name, data_dict):
        import json, io
        with io.open(file_name, 'w', encoding='utf-8') as fp:
            # fp.write(unicode(json.dumps(data_dict, ensure_ascii=False, indent=4) ) )  #可以解决在文件里显示中文的问题，不加的话是 '\uxxxx\uxxxx'
            fp.write((json.dumps(data_dict, ensure_ascii=False, indent=4) ) )
            
    @staticmethod
    def file2dict_json(file_name):
        import json, io
        with io.open(file_name, 'r', encoding='utf-8') as fp:
            data_dict = json.load(fp)
        return data_dict


def Generate_Json_annotation(TL_Cluster_Video_dict, Outpu_dir,xml_dir_):
    '''   '''
    ICDAR21_DetectionTracks = {}
    text_id = 1
    
    doc = Document()
    video_xml = doc.createElement("Frames")
    
    for frame in TL_Cluster_Video_dict.keys():
        
        doc.appendChild(video_xml)
        aperson = doc.createElement("frame")
        aperson.setAttribute("ID", str(frame))
        video_xml.appendChild(aperson)

        ICDAR21_DetectionTracks[frame] = []
        for text_list in TL_Cluster_Video_dict[frame]:
            ICDAR21_DetectionTracks[frame].append({"points":text_list[:8],"ID":text_list[8],"transcription":text_list[9]})
            
            # xml
            object1 = doc.createElement("object")
            object1.setAttribute("ID", str(text_list[8]))
            object1.setAttribute("Transcription", str(text_list[9]))
            aperson.appendChild(object1)
            
            for i in range(4):
                
                name = doc.createElement("Point")
                object1.appendChild(name)
                # personname = doc.createTextNode("1")
                name.setAttribute("x", str(int(text_list[i*2])))
                name.setAttribute("y", str(int(text_list[i*2+1])))
                
    StorageDictionary.dict2file_json(Outpu_dir, ICDAR21_DetectionTracks)
    
    # xml
    f = open(xml_dir_, "w")
    f.write(doc.toprettyxml(indent="  "))
    f.close()

def sort_key(old_dict, reverse=False):
    """对字典按key排序, 默认升序, 不修改原先字典"""
    # 先获得排序后的key列表
    keys = [int(i) for i in old_dict.keys()]
    keys = sorted(keys, reverse=reverse)
    # 创建一个新的空字典
    new_dict = OrderedDict()
    # 遍历 key 列表
    for key in keys:
        new_dict[str(key)] = old_dict[str(key)]
    return new_dict

def getBboxesAndLabels_icd131(annotations):
    bboxes = []
    labels = []
    polys = []
    bboxes_ignore = []
    labels_ignore = []
    polys_ignore = []
    Transcriptions = []
    IDs = []
    rotates = []
    confidences = []
    # points_lists = [] # does not contain the ignored polygons.
    for annotation in annotations:
        object_boxes = []
        for point in annotation:
            object_boxes.append([int(point.attrib["x"]), int(point.attrib["y"])])

        points = np.array(object_boxes).reshape((-1))
        points = cv2.minAreaRect(points.reshape((4, 2)))
        # 获取矩形四个顶点，浮点型
        points = cv2.boxPoints(points).reshape((-1))         
        IDs.append(annotation.attrib["ID"])
        Transcriptions.append(annotation.attrib["Transcription"])
#         confidences.append(annotation.attrib["confidence"])
        confidences.append(1)
        bboxes.append(points)

    if bboxes:
        IDs = np.array(IDs, dtype=np.int64)
        bboxes = np.array(bboxes, dtype=np.float32)
    else:
        bboxes = np.zeros((0, 8), dtype=np.float32)
        IDs = np.array([], dtype=np.int64)
        Transcriptions = []
        confidences = []
        
    return bboxes, IDs, Transcriptions, confidences


def parse_xml_rec(annotation_path):
    utf8_parser = ET.XMLParser(encoding='gbk')
#     print(annotation_path)
    with open(annotation_path, 'r', encoding='gbk') as load_f:
        tree = ET.parse(load_f, parser=utf8_parser)
    root = tree.getroot()  # 获取树型结构的根
    
    ann_dict = {}
    for idx,child in enumerate(root):
#         image_path = os.path.join(video_path, child.attrib["ID"] + ".jpg")

        bboxes, IDs, Transcriptions, confidences = \
            getBboxesAndLabels_icd131(child)
        ann_dict[child.attrib["ID"]] = [bboxes,IDs,Transcriptions,confidences]
    return ann_dict

def getid_text(new_xml_dir_):
#     new_xml_dir_ = "/share/wuweijia/Code/VideoSpotting/TransDETRe2e/exps/e2e_TransVTS_r50_ICDAR15/jons"
#     new_xml_dir_1 = "/share/wuweijia/Code/VideoSpotting/MOTR/exps/e2e_TransVTS_r50_ICDAR15/e2e_xml_final"
    
    voc_dict = {"res_video_11.xml": "Video_11_4_1_GT_voc.txt", "res_video_15.xml": "Video_15_4_1_GT_voc.txt", "res_video_17.xml": "Video_17_3_1_GT_voc.txt", "res_video_1.xml": "Video_1_1_2_GT_voc.txt", "res_video_20.xml": "Video_20_5_1_GT_voc.txt", "res_video_22.xml": "Video_22_5_1_GT_voc.txt", "res_video_23.xml": "Video_23_5_2_GT_voc.txt", "res_video_24.xml": "Video_24_5_2_GT_voc.txt", "res_video_30.xml": "Video_30_2_3_GT_voc.txt", "res_video_32.xml": "Video_32_2_3_GT_voc.txt", "res_video_34.xml": "Video_34_2_3_GT_voc.txt", "res_video_35.xml": "Video_35_2_3_GT_voc.txt", "res_video_38.xml": "Video_38_2_3_GT_voc.txt", "res_video_39.xml": "Video_39_2_3_GT_voc.txt", "res_video_43.xml": "Video_43_6_4_GT_voc.txt", "res_video_44.xml": "Video_44_6_4_GT_voc.txt", "res_video_48.xml": "Video_48_6_4_GT_voc.txt", "res_video_49.xml": "Video_49_6_4_GT_voc.txt", "res_video_50.xml": "Video_50_7_4_GT_voc.txt", "res_video_53.xml": "Video_53_7_4_GT_voc.txt", "res_video_55.xml": "Video_55_3_2_GT_voc.txt", "res_video_5.xml": "Video_5_3_2_GT_voc.txt", "res_video_6.xml": "Video_6_3_2_GT_voc.txt", "res_video_9.xml": "Video_9_1_1_GT_voc.txt"}
    
    for xml in tqdm(os.listdir(new_xml_dir_)):
        id_trans = {}
        id_cond = {}
        if ".txt" in xml or "ipynb" in xml:
            continue
                
        lines = []
        xml_one = os.path.join(new_xml_dir_,xml)
        ann = parse_xml_rec(xml_one)
        for frame_id_ann in ann:
            points, IDs, Transcriptions,confidences = ann[frame_id_ann]
            for ids, trans, confidence in zip(IDs,Transcriptions,confidences):
                if str(ids) in id_trans:
                    id_trans[str(ids)].append(trans)
                    id_cond[str(ids)].append(float(confidence))
                else:
                    id_trans[str(ids)]=[trans]
                    id_cond[str(ids)]=[float(confidence)]
                    
        id_trans = sort_key(id_trans)
        id_cond = sort_key(id_cond)
#         print(xml)
        for i in id_trans:
            txts = id_trans[i]
            confidences = id_cond[i]
            txt = max(txts,key=txts.count)
            
            lines.append('"'+i+'"'+","+'"'+txt+'"'+"\n")
        write_lines(os.path.join(new_xml_dir_,xml.replace("xml","txt")),lines)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    parser.add_argument('--score_threshold', default=0.5, type=float)
    parser.add_argument('--update_score_threshold', default=0.5, type=float)
    parser.add_argument('--miss_tolerance', default=-1, type=int)
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # load model and weights
    detr, _, _ = build_model(args)
    detr.track_embed.score_thr = args.update_score_threshold
    detr.track_base = RuntimeTrackerBase(args.score_threshold, args.score_threshold, args.miss_tolerance)
    checkpoint = torch.load(args.resume, map_location='cpu')
    detr = load_model(detr, args.resume)
    detr.eval()
    detr = detr.cuda()
    
    # load Vedio sequn

###################### ICDAR2015vedio #############################################
    # args.mot_path = "./data/ICDAR2015/images/track/val"
    # seq_nums = []
    # for seq in os.listdir(args.mot_path):
    #     seq_nums.append(os.path.join(args.mot_path,seq))
######################## DSText #########################################
    args.mot_path = "./data/DSText/images/test"
    seq_nums = []
    for seq in os.listdir(args.mot_path):
        for video_name in os.listdir(os.path.join(args.mot_path,seq)):
            seq_nums.append(os.path.join(seq,video_name))
#####################################################################
    # '''for MOT17 submit''' 
    # sub_dir = 'DanceTrack/test'
    # sub_dir = 'DSText/images/test/Game'
    # sub_dir = 'DSText/images/test/Street_View_Indoor'
    # seq_nums = os.listdir(os.path.join(args.mot_path, sub_dir))

    # if 'seqmap' in seq_nums:
    #     seq_nums.remove('seqmap')
    # vids = [os.path.join('./data/DSText/images/test', seq) for seq in seq_nums]

    rank = int(os.environ.get('RLAUNCH_REPLICA', '0'))
    ws = int(os.environ.get('RLAUNCH_REPLICA_TOTAL', '1'))
    # vids = vids[rank::ws]

    for seq_num in seq_nums:
        det = Detector(args, model=detr, vid=seq_num)
        det.detect(args.score_threshold)
    
    getid_text(os.path.join('/home/ubuntu/MOTRv2-trans/result/eval/preds'))
