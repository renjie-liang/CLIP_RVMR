import torch
from utils.utils import load_json, load_jsonl
import numpy as np
import math

import os
import matplotlib.pyplot as plt

pred_result = torch.load("pred_result_val.pt")
video_corpus = load_json("/home/share/rjliang/Dataset/TVR_Ranking_v3/video_corpus.json")
root_dir = 'figures/videos_logit'

seg_duration = 4

def init_logit(video_corpus):
    res = {}
    for video_name, dur_id in video_corpus.items():
        duration = dur_id[0]
        res[video_name] = np.zeros(math.ceil(duration))
    return res


def plot_save(video_name, videos_pred, videos_gt, output_dir, i):
    y1 = videos_pred[video_name]
    y2 = videos_gt[video_name]
    # Create a plot
    plt.figure()
    plt.plot(y1, label="prediction score")
    plt.plot(y2, label="ground truth")
    # plt.ylim(-10, 10)
    plt.title(f'{video_name}')

    output_file = os.path.join(output_dir, f'{i}_{video_name}.jpg')
    plt.savefig(output_file)
    print(f'Plot saved to {output_file}')
    plt.close()


    

for pred in pred_result[:30]:
    videos_pred = init_logit(video_corpus)
    videos_gt = init_logit(video_corpus)

    query_id = pred["query_id"]
    relevant_moment = pred["relevant_moment"]
    predicted_seg_name = pred["predicted_seg_name"]
    predicted_seg_score = pred["predicted_seg_score"]
    video_name = predicted_seg_name[0]
    
    for seg_name, score in zip(predicted_seg_name, predicted_seg_score):
        parts = seg_name.rsplit('_', 1)
        video_name = parts[0]
        seg_idx = int(parts[1])

        start = seg_idx * seg_duration
        end = (seg_idx + 1)* seg_duration
        videos_pred[video_name][start:end] = score
    
    for i in relevant_moment:
        video_name = i["video_name"]
        timestamp = i["timestamp"]
        start = round(timestamp[0])
        end = round(timestamp[1])
        videos_gt[video_name][start:end] = 1
    
        
    
    out_dir = os.path.join(root_dir, str(query_id))
    os.makedirs(out_dir, exist_ok=True)
    video_name_set = set()
    for i in range(50):
        seg_name = predicted_seg_name[i]
        score = predicted_seg_score[i]
        parts = seg_name.rsplit('_', 1)
        video_name = parts[0]
        
        if video_name in video_name_set:
            continue
        else:
            video_name_set.add(video_name)
            plot_save(video_name, videos_pred, videos_gt, out_dir, i)