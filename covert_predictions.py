import torch
from utils.utils import load_json
import numpy as np
import math
from tqdm import tqdm
from utils.ndcg_iou import calculate_ndcg_iou

def init_logit(video_corpus):
    res = {}
    for video_name, dur_id in video_corpus.items():
        duration = dur_id[0]
        res[video_name] = np.zeros(math.ceil(duration))
    return res

SEG_DURATION = 4

all_pred_result = torch.load("/home/share/rjliang/pred_result_val.pt")
new_pred_result_name = "/home/share/rjliang/pred_result_val_video_simi.npy"
video_corpus = load_json("/home/share/rjliang/TVR_Ranking/video_corpus.json")

new_result = {}
for pred_result in tqdm(all_pred_result, desc="grab video similarity"):
    query_id = pred_result["query_id"]
    predicted_seg_name = pred_result["predicted_seg_name"]
    predicted_seg_score = pred_result["predicted_seg_score"]
    relevant_moment = pred_result["relevant_moment"]
    
    video_simi = {}
    video_name_set = set()
    for seg_name in predicted_seg_name:
        parts = seg_name.rsplit('_', 1)
        video_name = parts[0]
        video_name_set.add(video_name)
        duration = video_corpus[video_name]
        video_simi[video_name] = np.zeros(math.ceil(duration))
        
    for seg_name, seg_score in zip(predicted_seg_name, predicted_seg_score):
        parts = seg_name.rsplit('_', 1)
        video_name = parts[0]
        seg_idx = int(parts[1])
        start = seg_idx * SEG_DURATION
        end = min((seg_idx + 1) * SEG_DURATION, len(video_simi[video_name]))  # Ensure end doesn't exceed duration
        video_simi[video_name][start:end] = seg_score
        
    new_result[query_id] = video_simi
    
# Save new_result to new_pred_result_name with NumPy
np.save(new_pred_result_name, new_result)
