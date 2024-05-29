from utils.utils import load_json, load_jsonl
import numpy as np
def generate_map(map_path):
    video_map = {}
    map_data = load_json(map_path)
    for video_name, i in map_data.items():
        video_id = i[1]
        video_map[video_id] = video_name
    return video_map

def generate_gt(annotation_path):
    all_gt = {}
    annotations = load_jsonl(annotation_path)
    for record in annotations:
        query_id = record["query_id"]
        
        one_text_gt = []
        for i in record["relevant_moment"]:
            video_name = i["video_name"]
            relevance = i["relevance"]
            if relevance >= 1:
                one_text_gt.append(video_name)
        if len(one_text_gt) == 0:
            video_name = record["relevant_moment"][0]["video_name"]
            one_text_gt.append(video_name)
        all_gt[query_id] = one_text_gt
    return all_gt


test_gt = generate_gt("/home/renjie.liang/11_TVR-Ranking/ReLoCLNet/data/TVR_Ranking_v3/val.jsonl")
    
test_path = "result/ReLocNet/best_val_predictions.json"
test_prediction = load_json(test_path)

video_map = generate_map("result/ReLocNet/video_name_duration_id.json")


test_vr_pred = {}
for record in test_prediction:
    query_id = record["query_id"]
    predictions = record["predictions"]
    tmp = []
    for p in predictions:
        video_id = p[0]
        video_name = video_map[video_id]
        tmp.append(video_name)
    test_vr_pred[query_id] = tmp



topk = 100
recalls = []

for text_idx, preds in test_vr_pred.items():
    gt_videos = test_gt[text_idx]
    preds = preds[:topk]
    recall = sum(1 for gt in gt_videos if gt in preds) / len(gt_videos)
    recalls.append(recall)
    
average_recall = np.mean(recalls)
print(average_recall)