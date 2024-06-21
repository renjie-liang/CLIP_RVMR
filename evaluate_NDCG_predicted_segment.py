import torch
from utils.utils import load_json, load_jsonl
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from  utils.ndcg_iou import calculate_ndcg_iou
from tqdm import tqdm


def grab_gt_moments(pred_results):
    gt_moments_all = {}
    for pred in tqdm(pred_results, "Grab ground truth moment list:"):
        query_id = pred["query_id"]
        gt_moments= pred["relevant_moment"]
        gt_moments_all[query_id] = gt_moments
    return gt_moments_all
from tqdm import tqdm



### please modify this function with the different strategies
def grab_pred_moments(pred_results):
    """
    Extracts predicted moments from prediction results.

    Args:
        pred_results (list): A list of dictionaries where each dictionary contains:
            - predicted_seg_name (list of str): List of segment names (e.g., ["video1_segidx0", "video1_segidx1", ...])
            - predicted_seg_score (list of float): List of scores corresponding to each segment.
            - query_id (str): The query identifier.

    Returns:
        dict: A dictionary where each key is a query_id and the value is a list of dictionaries. 
              Each dictionary contains:
                - video_name (str): Name of the video.
                - start_time (int): Start time of the segment.
                - end_time (int): End time of the segment.
                - model_scores (float): Score of the segment.
    """
    
    seg_duration = 4
    pred_moments_all = {}

    for pred in tqdm(pred_results, "Grab predicted moment list:"):
        query_id = pred["query_id"]
        predicted_seg_name = pred["predicted_seg_name"]
        predicted_seg_score = pred["predicted_seg_score"]
        
        one_query_preds = []

        # Process top 50 predictions
        for i in range(50):
            seg_name = predicted_seg_name[i]
            score = predicted_seg_score[i]
            parts = seg_name.rsplit('_', 1)
            video_name = parts[0]
            seg_idx = int(parts[1])
            start = seg_idx * seg_duration
            end = (seg_idx + 1) * seg_duration
            
            segment_info = {
                "timestamp": [start-2, end+2],
                "video_name": video_name,
                "model_scores": score,
            }
            
            one_query_preds.append(segment_info)
        pred_moments_all[query_id] = one_query_preds
    return pred_moments_all


KS = [10, 20, 40]
TS = [0.3, 0.5, 0.7]

pred_result = torch.load("/home/share/rjliang/pred_result_val.pt")
gt_moments_all = grab_gt_moments(pred_result)
pred_moments_all = grab_pred_moments(pred_result)
average_ndcg = calculate_ndcg_iou(gt_moments_all, pred_moments_all , TS, KS)
for K, vs in average_ndcg.items():
    for T, v in vs.items():
        print(f"VAL Top {K}, IoU={T}, NDCG: {v:.6f}")


