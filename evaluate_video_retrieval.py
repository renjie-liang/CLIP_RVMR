import numpy as np
import torch
from tqdm import tqdm
def grab_corpus_feature(model, corpus_dataloader, device):
    all_video_features, all_video_masks = [], []
    model.eval()
    n = 0
    with torch.no_grad():
        # Process video features
        for batch in tqdm(corpus_dataloader, desc="Grab Videos Feature"):
            batch = [b.to(device) for b in batch]
            videos, video_masks = batch
            visual_output = model.module.get_video_features(videos)
            # Detach and move to CPU
            visual_output = visual_output.detach().cpu()
            video_masks = video_masks.detach().cpu()
            
            all_video_features.append(visual_output)
            all_video_masks.append(video_masks)
            
            # n += 1
            # if n > 10:
            #     break
        # Concatenate all video features and masks
        all_video_features = torch.cat(all_video_features, dim=0)
        all_video_masks = torch.cat(all_video_masks, dim=0)
    return all_video_features, all_video_masks

def eval_epoch(model, eval_dataloader, corpus_feature, device, ground_truth, corpus_videos, topk):
    all_video_features, all_video_masks = corpus_feature
    all_text_features = []
    with torch.no_grad():
        # Process text features
        for batch in tqdm(eval_dataloader, desc="Get Texts Feature"):
            batch = [b.to(device) for b in batch]
            text_input_ids, attention_mask = batch
            text_output = model.module.clip_model.get_text_features(input_ids=text_input_ids, attention_mask=attention_mask)
            # Detach and move to CPU
            text_output = text_output.detach().cpu()
            attention_mask = attention_mask.detach().cpu()
            all_text_features.append(text_output)
        # Concatenate all text features and masks
        all_text_features = torch.cat(all_text_features, dim=0)
    # ----------------------------------
    # 2. calculate the similarity
    # ----------------------------------
    average_recall = calculate_recall_topk(model, all_text_features, all_video_features, all_video_masks, topk, ground_truth, corpus_videos)
    return average_recall


import torch
import numpy as np
from tqdm import tqdm

def calculate_recall_topk(model, all_text_features, all_video_features, all_video_masks, topk, ground_truth, corpus_videos):
    all_text_features = all_text_features.squeeze(1)
    all_text_features = all_text_features.contiguous()
    all_video_features = all_video_features.contiguous()
    # Calculate similarities in a batch
    simi_matrix = model.module.compute_similarity_matrix(all_text_features, all_video_features, all_video_masks)
    simi_matrix = simi_matrix.cpu().detach().numpy()
    recalls = []
    for text_idx in tqdm(range(simi_matrix.shape[0]), desc="Calculate the Recall"):
        simi = simi_matrix[text_idx]
        gt_videos = ground_truth[text_idx]
        # Get top N similar video indices
        top_n_indices = np.argsort(-simi)[:topk]
        top_n_video_names = [corpus_videos[i] for i in top_n_indices]
        # Calculate recall
        recall = sum(1 for gt in gt_videos if gt in top_n_video_names) / len(gt_videos)
        recalls.append(recall)

    average_recall = np.mean(recalls)
    return average_recall
