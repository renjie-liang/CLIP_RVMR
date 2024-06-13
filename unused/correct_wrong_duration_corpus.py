
import pandas as pd
from utils.basic_utils import load_json, save_json
import json
import os
from tqdm import tqdm
import math 

# Load video corpus data
video_corpus_path = "/home/renjie.liang/11_TVR-Ranking/ReLoCLNet/data/TVR_Ranking_v2/video_corpus.json"
new_video_corpus_path = "/home/renjie.liang/11_TVR-Ranking/ReLoCLNet/data/TVR_Ranking_v4/video_corpus.json"
video_path  = "/home/share/rjliang/Dataset/TVR/frames"
video_corpus = load_json(video_corpus_path)

new_video_corpus = []
for video_name, (duration, vid) in tqdm(video_corpus.items()):
    frame_path = os.path.join(video_path, video_name)
    total_frames = len(os.listdir(frame_path)) 
    duration_frames = math.floor(total_frames/3 * 100) / 100
    video_corpus[video_name] = duration_frames
        # print(video_name, duration, duration_frames)
# Save the segment corpus
save_json(video_corpus, new_video_corpus_path)
# print(f"Segment corpus saved to {segment_corpus_path}")
