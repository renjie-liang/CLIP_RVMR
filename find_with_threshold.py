import torch
from utils.utils import load_json
import numpy as np
import math
from tqdm import tqdm
from  utils.ndcg_iou import calculate_ndcg_iou


def grab_gt_moments(pred_results):
    gt_moments_all = {}
    for pred in tqdm(pred_results, "Grab ground truth moment list:"):
        query_id = pred["query_id"]
        gt_moments= pred["relevant_moment"]
        gt_moments_all[query_id] = gt_moments
    return gt_moments_all
from tqdm import tqdm

def normalize_scores(scores):
    min_score = np.min(scores)
    max_score = np.max(scores)
    normalized_scores = (scores - min_score) / (max_score - min_score)
    return normalized_scores

def average(scores):
    return np.mean(scores)

def top_video_similarity(all_pred_result, all_pred_result_video_simi, n):
    top_video_simi = {}
    for pred_result in tqdm(all_pred_result, desc="grab top video similarity"):
        query_id = pred_result["query_id"]
        predicted_seg_name = pred_result["predicted_seg_name"]
        
        # find topn video_name and init the similarity array
        video_simi = {}
        video_name_set = set()
        for seg_name in predicted_seg_name:
            parts = seg_name.rsplit('_', 1)
            video_name = parts[0]
            video_name_set.add(video_name)
            video_simi[video_name] = all_pred_result_video_simi[query_id][video_name]
            if len(video_name_set) >= n:
                break
        
        top_video_simi[query_id] = video_simi
    return top_video_simi


def find_clip(top_video_simi, threshold):
    pred_moments_all = {}
    for query_id, video_simis in tqdm(top_video_simi.items(), desc="find with threshold:"):
        one_query_preds = []
        for video_name, vs in video_simis.items():
            # Normalize the vs scores
            vs = normalize_scores(np.array(vs))

            # Identify segments that are above the threshold
            segments = []
            current_segment = None
            for i, score in enumerate(vs):
                if score >= threshold:
                    if current_segment is None:
                        current_segment = [i, i]
                    else:
                        current_segment[1] = i
                else:
                    if current_segment is not None:
                        segments.append(current_segment)
                        current_segment = None
            if current_segment is not None:
                segments.append(current_segment)

            # Create segment info for each identified segment
            for segment in segments:
                start, end = segment
                avg_score = average(vs[start:end+1])
                segment_info = {
                    "timestamp": [start-1, end+1],
                    "video_name": video_name,
                    "model_scores": avg_score,
                }
                one_query_preds.append(segment_info)
        pred_moments_all[query_id] = one_query_preds
    return pred_moments_all

SEG_DURATION = 4
KS = [10, 20, 40]
TS = [0.3, 0.5, 0.7]
SIMI_THRESHOLD = 0.8


all_pred_result = torch.load("/home/share/rjliang/pred_result_val.pt")
video_corpus = load_json("/home/share/rjliang/TVR_Ranking/video_corpus.json")
new_pred_result_name = "/home/share/rjliang/pred_result_val_video_simi.npy"
all_pred_result_video_simi = np.load(new_pred_result_name, allow_pickle=True).item()


topn_video = 100
# top_video_simi = {"query_id": {"video_name1": 1*L1, "video_name2": 1*L2}}
top_video_simi = top_video_similarity(all_pred_result, all_pred_result_video_simi, topn_video)
pred_moments_all = find_clip(top_video_simi, SIMI_THRESHOLD)

gt_moments_all = grab_gt_moments(all_pred_result)
average_ndcg = calculate_ndcg_iou(gt_moments_all, pred_moments_all , TS, KS)
for K, vs in average_ndcg.items():
    for T, v in vs.items():
        print(f"VAL NDCG@{K}, IoU={T}: {v:.6f}")

