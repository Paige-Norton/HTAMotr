# -*- coding: utf-8 -*-
import cv2
import os
import copy
import numpy as np
import math
# import Levenshtein
from cv2 import VideoWriter, VideoWriter_fourcc
import json
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import moviepy
import moviepy.video.io.ImageSequenceClip
import shutil
from moviepy.editor import *



def pics2video(frames_dir="", fps=25):
    im_names = os.listdir(frames_dir)
    num_frames = len(im_names)
    frames_path = []
    for im_name in tqdm(range(0, num_frames)):
        string = os.path.join( frames_dir, str(im_name) + '.jpg')
        frames_path.append(string)
        
#     frames_name = sorted(os.listdir(frames_dir))
#     frames_path = [frames_dir+frame_name for frame_name in frames_name]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frames_path, fps=fps)
    clip.write_videofile(frames_dir+".mp4", codec='libx264')
#     shutil.rmtree(frames_dir)
    
    
if __name__ == "__main__":
#     image_path = "./exps/e2e_TransVTS_r50_COCOTextV2/results"   e2e_TransVTS_r50_SynthText   e2e_TransVTS_r50_ICDAR15   e2e_TransVTS_r50_BOVText
    image_path = "/home/ubuntu/MOTRv2-trans/result_changeProposalfangfa_bigdata/eval_nochange_loss_changeProposal_bigdata_epoch14_lrDrop_5******/img"
    seqs = ["Video_156_5_6"]
    for video_name in os.listdir(image_path):
#         if video_name not in seqs:
#             continue
#         if video_name not in ["Video_11_4_1"]:
#             continue
        if "mp4" in video_name:
            continue
        if "ip" in video_name:
            continue
        
        try:
            video_name_one = os.path.join(image_path,video_name)
            pics2video(frames_dir=video_name_one,fps=25)
        except:
            continue