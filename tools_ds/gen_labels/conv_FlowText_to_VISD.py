import os.path as osp
import os
import numpy as np
from util.utils import write_result_as_txt,debug, setup_logger,write_lines,MyEncoder
try:
    import xml.etree.cElementTree as ET  # 解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import math
from tqdm import tqdm
import json
import shutil
import imutils

def adjust_box_sort(box):
    start = -1
    _box = list(np.array(box).reshape(-1,2))
    min_x = min(box[0::2])
    min_y = min(box[1::2])
    _box.sort(key=lambda x:(x[0]-min_x)**2+(x[1]-min_y)**2)
    start_point = list(_box[0])
    for i in range(0,8,2):
        x,y = box[i],box[i+1]
        if [x,y] == start_point:
            start = i//2
            break

    new_box = []
    new_box.extend(box[start*2:])
    new_box.extend(box[:start*2])
    return np.array(new_box)


def find_min_rect_angle(vertices):
    '''find the best angle to rotate poly and obtain min rectangle
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the best angle <radian measure>
    '''
            
    angle_interval = 1
    angle_list = list(range(-90, 90, angle_interval))
    vertices = adjust_box_sort(vertices)
    area_list = []
    for theta in angle_list:
        rotated = rotate_vertices(vertices, theta / 180 * math.pi)
        x1, y1, x2, y2, x3, y3, x4, y4 = rotated
        temp_area = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * \
                    (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
        area_list.append(temp_area)

    sorted_area_index = sorted(list(range(len(area_list))), key=lambda k: area_list[k])
    min_error = float('inf')
    best_index = -1
    rank_num = 10
    # find the best angle with correct orientation
    for index in sorted_area_index[:rank_num]:
        rotated = rotate_vertices(vertices, angle_list[index] / 180 * math.pi)
        temp_error = cal_error(rotated)
        if temp_error < min_error:
            min_error = temp_error
            best_index = index
    return angle_list[best_index] / 180 * math.pi

def rotate_vertices(vertices, theta, anchor=None):
    '''rotate vertices around anchor
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        theta   : angle in radian measure
        anchor  : fixed position during rotation
    Output:
        rotated vertices <numpy.ndarray, (8,)>
    '''
    v = vertices.reshape((4, 2)).T
#     print(v)
#     print(anchor)
    if anchor is None:
#         anchor = v[:, :1]
        anchor = np.array([[v[0].sum()],[v[1].sum()]])/4
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)

def get_rotate_mat(theta):
    '''positive theta value means rotate clockwise'''
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

def cal_error(vertices):
    '''default orientation is x1y1 : left-top, x2y2 : right-top, x3y3 : right-bot, x4y4 : left-bot
    calculate the difference between the vertices orientation and default orientation
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        err     : difference measure
    '''
    x_min, x_max, y_min, y_max = get_boundary(vertices)
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    err = cal_distance(x1, y1, x_min, y_min) + cal_distance(x2, y2, x_max, y_min) + \
          cal_distance(x3, y3, x_max, y_max) + cal_distance(x4, y4, x_min, y_max)
    return err

def get_boundary(vertices):
    '''get the tight boundary around given vertices
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the boundary
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return x_min, x_max, y_min, y_max

def cal_distance(x1, y1, x2, y2):
    '''calculate the Euclidean distance'''
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def get_rotate(box):
    # box : x1,y2...,x3,y3
    theta = find_min_rect_angle(box)
    
    rotate_mat = get_rotate_mat(theta)
    rotated_vertices = rotate_vertices(box, theta)
#     print(rotated_vertices)
    x_min, x_max, y_min, y_max = get_boundary(rotated_vertices)
#     print(x_min, x_max, y_min, y_max)
    return np.array([x_min, y_min,x_max-x_min , y_max-y_min]),theta
    
def getBboxesAndLabels_icd13(height, width, annotations):
    bboxes_bonding = []
    labels = []
    polys = []
    bboxes_ignore = []
    labels_ignore = []
    polys_ignore = []
    IDs = []
    rotates = []
    bboxes_box = []

    # points_lists = [] # does not contain the ignored polygons.
    img_paths = annotations['img']
    bboxes = annotations['instances']
    bboxes_all = []
    id_S = []
    for idss in bboxes:
        id_S.append(idss)
        bboxes_all.append(bboxes[idss])
    bboxes_all = np.array(bboxes_all)
    texts = annotations['words']

#     bboxes = np.reshape(bboxes, (bboxes.shape[0], bboxes.shape[1], -1))
# #                     bboxes = bboxes.transpose(2, 1, 0)
#     bboxes = np.reshape(bboxes, (bboxes.shape[0], -1))

#     words = []
#     for idx,text in enumerate(texts):
#         text = text.replace('\n', ' ').replace('\r', ' ')
#         words.extend([w for w in text.split(' ') if len(w) > 0])
    
    words_final = []
    for i, (id_one, content) in enumerate(zip(id_S,bboxes_all)):
        object_boxe = content["coords"]
        word = content["text"]
        
        object_boxe = [np.array(i,dtype=np.int64) for i in object_boxe]
        
        # 最小外接矩形框，有方向角
        rect = cv2.minAreaRect(np.array(object_boxe))
        box = cv2.cv.Boxpoints() if imutils.is_cv2()else cv2.boxPoints(rect)
        box = np.int0(box)
#         print(box)
        points = np.array(box).reshape((-1)).reshape((4, 2))

        points_rotate = cv2.minAreaRect(points)
        # 获取矩形四个顶点，浮点型
        points_rotate = cv2.boxPoints(points_rotate).reshape((-1))
#         rotate_box, rotate = get_rotate(points_rotate)
        
#         x, y, w, h = cv2.boundingRect(points.reshape((4, 2)))
#         box = np.array([x, y, w, h])
        
        Transcription=word
        
        words_final.append(Transcription)    
        bboxes_box.append(points_rotate)
#         IDs.append(int(id_one))
#         rotates.append(rotate)
#         bboxes_bonding.append(box)

    if bboxes_box:
        bboxes_box = np.array(bboxes_box, dtype=np.float32)
#         bboxes_bonding = np.array(bboxes_bonding, dtype=np.float32)
        # filter the coordinates that overlap the image boundaries.
#         bboxes_box[:, 0::2] = np.clip(bboxes_box[:, 0::2], 0, width - 1)
#         bboxes_box[:, 1::2] = np.clip(bboxes_box[:, 1::2], 0, height - 1)
#         IDs = np.array(IDs, dtype=np.int64)
#         rotates = np.array(rotates, dtype=np.float32)
    else:
        bboxes_box = np.zeros((0, 4), dtype=np.float32)
#         bboxes_bonding = np.zeros((0, 4), dtype=np.float32)
        # polygon_point = np.zeros((0, 8), dtype=np.int)
#         IDs = np.array([], dtype=np.int64)
#         rotates = np.array([], dtype=np.float32)
        words_final = []

    return bboxes_box,words_final

def parse_xml(annotation_path,image_path):
    
    with open(ann_path,'r') as load_f:
        load_dict = json.load(load_f)
            
    bboxess, IDss, rotatess, wordss,orignial_bboxess = [], [] , [], [], []
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

            
    for idx,child in enumerate(load_dict):
        
        bboxes, words = \
            getBboxesAndLabels_icd13(height, width, child)
        bboxess.append(bboxes) 
        wordss.append(words)
    return bboxess, wordss

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

def gen_data_path(path,split_train_test="train",data_path_str = "./datasets/data_path/VideoSynthText.train"):
    
    image_path = os.path.join(path,"images",split_train_test)
    ann_path = os.path.join(path,"labels_with_ids",split_train_test)
    
    lines = []
    check_repeat = []
    for video_name in os.listdir(image_path):
        
#         if video_name.split("_")[0]+video_name.split("_")[1] not in check_repeat:
#             check_repeat.append(video_name.split("_")[0]+video_name.split("_")[1])
#         else:
#             continue
            
        frame_path = os.path.join(image_path,video_name)
        annotation_path = os.path.join(ann_path,video_name)
        
        min_frame = 10000
        for ii in os.listdir(frame_path):
            if "jpg" in ii:
                if int(ii.replace(".jpg",""))< min_frame:
                    min_frame = int(ii.replace(".jpg",""))
                        
        frame_list = []
        print(video_name)
        for frame_path_ in os.listdir(frame_path):
            if ".jpg" in frame_path_ and os.path.exists(os.path.join(annotation_path,frame_path_.replace(".jpg",".txt"))):
                frame_list.append(frame_path_)
        for i in range(0,len(frame_list)):
            frame_id = min_frame + i 
            frame_real_path = "VideoSynthText/images/train/" + video_name + "/{}.jpg".format(str(frame_id).zfill(8)) + "\n"
            lines.append(frame_real_path)
    write_lines(data_path_str, lines)  
    

from_label_root = "/mmu-ocr/yuzhong/code/VideoSynthtext/SynthText/gen_data/synthtextvid_activitynet_154fonts_1f_805"
seq_root = "/mmu-ocr/yuzhong/code/VideoSynthtext/SynthText/gen_data/synthtextvid_activitynet_154fonts_1f_805"
# label_root = '/mmu-ocr/weijiawu/Data/VideoText/MOTR/VideoSynthText/labels_with_ids/train'

to_image = "/mmu-ocr/weijiawu/Data/VideoText/MOTR/VideoSynthText_image/train_image"
to_txt = "/mmu-ocr/weijiawu/Data/VideoText/MOTR/VideoSynthText_image/train_gt"


# mkdirs(label_root)
seqs = [s for s in os.listdir(seq_root)]


tid_curr = 0
tid_last = -1
ha = 0
for seq_id,seq in tqdm(enumerate(seqs)):
#     if seq_id>4000:
#         continue
    image_path_frame = osp.join(seq_root,seq)
#     seq_label_root = osp.join(label_root, seq)
#     mkdirs(seq_label_root)
    ha+=1  
    ann_path = os.path.join(from_label_root, seq, "ann.json")
    if not os.path.exists(ann_path):
#         print(ann_path)
        continue
    bboxess, wordss = parse_xml(ann_path,osp.join(image_path_frame,"00000000.jpg"))
    ID_list = {}
    
    for i in range(len(wordss)):
        
        frame_id = i
        label_fpath = osp.join(to_txt, '{}_{}.txt'.format(seq,str(frame_id).zfill(8)))
        to_frame_path_one = osp.join(to_image, '{}_{}.jpg'.format(seq,str(frame_id).zfill(8)))
        
        frame_path_one = osp.join(image_path_frame,"{}.jpg".format(str(frame_id).zfill(8)))
        
        
        lines = []
#         print(wordss[i])
#         if len(wordss[i]) <= 2:
#             continue
            
          
        shutil.copyfile(frame_path_one,to_frame_path_one)
        for bboxes,word in zip(bboxess[i],wordss[i]):
            
            label_str = '{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{}\n'.format(
            bboxes[0], bboxes[1],bboxes[2],bboxes[3],bboxes[4],bboxes[5],bboxes[6],bboxes[7], word)
            lines.append(label_str)
            
        write_lines(label_fpath, lines)     
print(ha)
# gen_data_path(path="/mmu-ocr/weijiawu/Data/VideoText/MOTR/VideoSynthText")