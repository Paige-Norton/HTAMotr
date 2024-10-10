import cv2
import os
from PIL import Image

# 图像帧的文件夹路径
frames_folder = r'C:\Users\Paige\Desktop\新建文件夹'

# 获取帧的列表
frame_list = [img for img in os.listdir(frames_folder) if img.endswith(".jpg")]
frame_list.sort(key=lambda x: int(x.split('.')[0])) 

frames = []
for i in frame_list:
    filename = os.path.join(frames_folder, i)
    img = Image.open(filename)
    frames.append(img)

# 保存为gif格式图片
frames[0].save('./doc/Video_220_2_0.gif', save_all=True, append_images=frames[1:], loop=0, duration=200)

print("GIF图片已保存")
