import xml.etree.ElementTree as ET
import numpy as np
import os

def calculate_iou(points1, points2):
    def get_min_max_coordinates(points):
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        return min(x_values), min(y_values), max(x_values), max(y_values)

    x1_min, y1_min, x1_max, y1_max = get_min_max_coordinates(points1)
    x2_min, y2_min, x2_max, y2_max = get_min_max_coordinates(points2)

    intersection_x_min = max(x1_min, x2_min)
    intersection_y_min = max(y1_min, y2_min)
    intersection_x_max = min(x1_max, x2_max)
    intersection_y_max = min(y1_max, y2_max)

    intersection_width = max(0, intersection_x_max - intersection_x_min)
    intersection_height = max(0, intersection_y_max - intersection_y_min)

    area_intersection = intersection_width * intersection_height

    area_box1 = (x1_max - x1_min) * (y1_max - y1_min)
    area_box2 = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = area_box1 + area_box2 - area_intersection

    iou = area_intersection / union_area if union_area > 0 else 0

    return iou

def evaluate_tracking(gt_frames, pred_frames, threshold=0.5):
    num_frames = len(gt_frames)
    total_gt = 0
    total_idtp = 0
    total_idfp = 0
    total_idfn = 0
    total_distance = 0
    total_id_changes = 0  # 新增变量用于记录 ID 变化的次数

    for frame in range(num_frames):
        gt_objects = gt_frames[frame]
        pred_objects = pred_frames[frame]['Objects']

        num_match = 0   # 每一帧正确匹配的数量
        total_iou = 0
        id_changes_in_frame = 0  # 记录每一帧的 ID 变化次数

        # 创建映射以跟踪新生成的 pred_id 与 gt_id 的匹配
        id_mapping = {}

        for gt_object in gt_objects:
            total_gt += 1

        for pred_object in pred_objects:
            gt_id_matched = False
            best_iou = 0
            match_gt_object = None

            # 将 gt_id 匹配改为映射后的 pred_id 与 gt_id 进行匹配
            for gt_object in gt_objects:
                gt_id = gt_object['ID']
                gt_points = [(gt_object['BoundingBox'][0], gt_object['BoundingBox'][1]),
                             (gt_object['BoundingBox'][2], gt_object['BoundingBox'][1]),
                             (gt_object['BoundingBox'][2], gt_object['BoundingBox'][3]),
                             (gt_object['BoundingBox'][0], gt_object['BoundingBox'][3])]

                pred_id = pred_object['ID']

                # 使用映射后的 pred_id 进行匹配
                if pred_id in id_mapping and gt_id == id_mapping[pred_id]:
                    iou = calculate_iou(gt_points, pred_object['Points'])
                    if iou > best_iou:
                        best_iou = iou
                        gt_id_matched = True
                        match_gt_object = gt_object

            if not gt_id_matched and match_gt_object is not None:
                # 如果 gt_id 和 pred_id 不匹配，并且找到了匹配的 gt_object，说明是新生成的 pred_id
                # 更新映射关系，使用匹配的 gt_id
                gt_id_matched = True
                pred_object['ID'] = match_gt_object['ID']
                id_mapping[pred_object['ID']] = gt_objects.index(match_gt_object)
                id_changes_in_frame += 1  # 增加 ID 变化次数

            if not gt_id_matched:
                # 如果 gt_id 和 pred_id 仍然不匹配，找到 IoU 最大的一对进行匹配
                for gt_object in gt_objects:
                    gt_points = [(gt_object['BoundingBox'][0], gt_object['BoundingBox'][1]),
                                 (gt_object['BoundingBox'][2], gt_object['BoundingBox'][1]),
                                 (gt_object['BoundingBox'][2], gt_object['BoundingBox'][3]),
                                 (gt_object['BoundingBox'][0], gt_object['BoundingBox'][3])]

                    iou = calculate_iou(gt_points, pred_object['Points'])
                    if iou > best_iou:
                        best_iou = iou
                        gt_id_matched = True
                        match_gt_object = gt_object

                if gt_id_matched:
                    # 更新映射关系，使用匹配的 gt_id
                    pred_object['ID'] = match_gt_object['ID']
                    id_mapping[pred_object['ID']] = gt_objects.index(match_gt_object)
                    id_changes_in_frame += 1  # 增加 ID 变化次数

            if gt_id_matched:
                total_idtp += 1
                num_match += 1
                total_iou += best_iou

            else:
                total_idfp += 1

        total_idfn += len(gt_objects) - num_match  # 未正确跟踪的目标数
        total_id_changes += id_changes_in_frame  # 累积每一帧的 ID 变化次数

    mota = 1 - (total_idfn + total_idfp + total_id_changes) / total_gt
    motp = total_iou / num_match if num_match > 0 else 0
    idf1 = 2 * num_match / (2 * num_match + total_idfp + total_idfn) if (num_match + total_idfp + total_idfn) > 0 else 0

    return mota, motp, idf1, total_id_changes  # 返回总的 ID 变化次数



def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    frames = []

    for frame in root.findall('frame'):
        frame_id = frame.get('ID')
        objects = []

        for obj in frame.findall('object'):
            object_id = obj.get('ID')
            points = []

            for point in obj.findall('Point'):
                x = int(point.get('x'))
                y = int(point.get('y'))
                points.append((x, y))

            objects.append({'ID': object_id, 'Points': points})

        frames.append({'ID': frame_id, 'Objects': objects})

    return frames

def calculate_average_metrics(gt_all_videos_data, pred_all_videos_data):
    total_mota = 0
    total_motp = 0
    total_idf1 = 0

    num_files = len(gt_all_videos_data)

    for video_name, gt_video_data in gt_all_videos_data.items():
        if video_name in pred_all_videos_data:
            pred_video_data = pred_all_videos_data[video_name]
            mota, motp, idf1,id_changes = evaluate_tracking(gt_video_data, pred_video_data)

        total_mota += mota
        total_motp += motp
        total_idf1 += idf1

    # 计算平均值
    average_mota = total_mota / num_files
    average_motp = total_motp / num_files
    average_idf1 = total_idf1 / num_files

    return average_mota, average_motp, average_idf1,id_changes


def parse_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    frame_objects = []

    for line in lines:
        data = line.strip().split(' ')
        object_id = int(data[1])
        bbox = [float(data[i]) for i in range(7, 11)]
        object_info = {'ID': object_id, 'BoundingBox': bbox}
        frame_objects.append(object_info)

    return frame_objects

def parse_frames_in_folder(folder_path):
    frame_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    frames_data = []

    for frame_file in sorted(frame_files):
        frame_path = os.path.join(folder_path, frame_file)
        frame_objects = parse_txt_file(frame_path)
        frames_data.append(frame_objects)

    return frames_data

def parse_all_videos(gt_parent_folder):
    gt_video_data = {}

    for video_folder in os.listdir(gt_parent_folder):
        video_path = os.path.join(gt_parent_folder, video_folder)
        if os.path.isdir(video_path):
            video_frames_data = parse_frames_in_folder(video_path)
            gt_video_data[video_folder] = video_frames_data

    return gt_video_data

if __name__ == '__main__':
    # 循环读取文件并计算指标
    pred_files_dir = '/home/ubuntu/MOTRv2-trans/result/eval_change_loss_weibiaoqian/preds'  # 替换成包含预测结果文件的目录路径
    gt_folder_path = '/home/ubuntu/MOTRv2-trans/test.file/train'

    gt_frames_data = parse_all_videos(gt_folder_path)
    pred_tracks_list = {}
    pred_files = [os.path.join(pred_files_dir, file) for file in os.listdir(pred_files_dir) if file.endswith('.xml')]
    for file in pred_files:
        pred_tracks_list[file.split('/')[-1].replace('res_','').replace('.xml','')]=parse_xml(file)

    # 计算平均指标
    average_mota, average_motp, average_idf1,id_changes = calculate_average_metrics(gt_frames_data, pred_tracks_list)

    # 输出平均指标
    print(f"Average MOTA: {average_mota}")
    print(f"Average MOTP: {average_motp}")
    print(f"Average IDF1: {average_idf1}")
    print(f"id_changes:{id_changes}")
