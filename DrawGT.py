import os
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# def list_images(folder_path):
#     image_extensions = ['.png', '.jpg']
#     images = []
#     for video_type in os.listdir(folder_path):
#         for video_name in os.listdir(os.path.join(folder_path,video_type)):
#             for file_name in os.listdir(os.path.join(folder_path,video_type,video_name)):
#                 if any(file_name.lower().endswith(ext) for ext in image_extensions):
#                     images.append(os.path.join(folder_path,video_type,video_name,file_name))
#     images.sort()
#     return images

key = "train"

def paige(type, label, path, points):
    image = Image.open(path)
    draw = ImageDraw.Draw(image)
    for point in points:
        draw.polygon(point, outline='red')

    j = os.path.join("/home/ubuntu/MOTRv2-trans/GT_show", type, label.replace('_GT.xml', ""))
    if not os.path.exists(j):
        os.makedirs(j)
    image.save(path.replace(f"/data/DSText/images/{key}_all", "/GT_show"))

# folder_path = '/home/ubuntu/MOTRv2-trans/data/DSText/images/train_all'
# image_paths = list_images(folder_path)

with open('/home/ubuntu/MOTRv2-trans/det_db_DBnet.json', 'w') as file:
    file.write("{")
    score = 1
    for test_type in os.listdir(f"/home/ubuntu/MOTRv2-trans/data/DSText/labels_with_ids/{key}_lable"):
        for test_label in os.listdir(os.path.join(f"/home/ubuntu/MOTRv2-trans/data/DSText/labels_with_ids/{key}_lable", test_type)):
            if test_label.endswith("xml"):
                tree = ET.parse(os.path.join(f"/home/ubuntu/MOTRv2-trans/data/DSText/labels_with_ids/{key}_lable", test_type, test_label))
                print(test_label)
                root = tree.getroot()
                for frame in root.findall('frame'):
                    points = []
                    p = os.path.join(test_type, test_label.replace('_GT.xml', ''))
                    path = f"/home/ubuntu/MOTRv2-trans/data/DSText/images/{key}_all/{p}/{frame.attrib['ID']}.jpg"
                    filename = f"data/DSText/images/test/{p}/{frame.attrib['ID']}.txt"
                    file.write("\"" + str(filename) + "\"" + ": [")

                    for obj in frame.findall('object'):
                        point = [(int(point.attrib['x']), int(point.attrib['y'])) for point in obj.findall('Point')]
                        points.append(point)
                        file.write("\"" + str(point).replace("[", "").replace("]", "").replace("(", "").replace(")", "") +"," +str(f'{score:.2f}')+ "\\n" + "\",")
                    file.write("],")
                    paige(test_type, test_label, path, points)
    file.write("}")
    file.close()








