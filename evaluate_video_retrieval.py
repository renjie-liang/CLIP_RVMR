import numpy as np
import torch
from tqdm import tqdm

def eval_epoch(args, model, eval_dataloader, corpus_dataloader, device, ground_truth):
    # ----------------------------
    # 1. cache the features
    # ----------------------------
    all_video_features, all_video_masks = [], []
    all_text_features, all_text_masks = [], []
    
    model.eval()
    with torch.no_grad():
        # Process video features
        for batch in tqdm(corpus_dataloader, desc="Get Videos Feature"):
            batch = [b.to(device) for b in batch]
            videos, video_masks = batch
            visual_output = model.get_visual_output(videos, video_masks)
            
            # Detach and move to CPU
            visual_output = visual_output.detach().cpu()
            video_masks = video_masks.detach().cpu()
            
            all_video_features.append(visual_output)
            all_video_masks.append(video_masks)
        # Concatenate all video features and masks
        all_video_features = torch.cat(all_video_features, dim=0)
        all_video_masks = torch.cat(all_video_masks, dim=0)

        # Process text features
        for batch in tqdm(eval_dataloader, desc="Get Texts Feature"):
            batch = [b.to(device) for b in batch]
            input_ids, input_mask = batch
            text_output = model.get_sequence_output(input_ids, input_mask)
            
            # Detach and move to CPU
            text_output = text_output.detach().cpu()
            input_mask = input_mask.detach().cpu()
            
            all_text_features.append(text_output)
            all_text_masks.append(input_mask)
        # Concatenate all text features and masks
        all_text_features = torch.cat(all_text_features, dim=0)
        all_text_masks = torch.cat(all_text_masks, dim=0)

    # ----------------------------------
    # 2. calculate the similarity
    # ----------------------------------
    logit_scale = model.clip.logit_scale.exp().item()
    R100 = calculate_recall_topn(all_text_features, all_text_masks, all_video_features, all_video_masks, 100, logit_scale, ground_truth)

    return R100

def mean_pooling_for_similarity_visual(visual_output, video_mask,):
    video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
    visual_output = visual_output * video_mask_un
    video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
    video_mask_un_sum[video_mask_un_sum == 0.] = 1.
    video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
    return video_out

def calculate_simi(text_output, video_output, video_mask,  logit_scale):
    text_output, video_output = text_output.contiguous(), video_output.contiguous()
    video_output = video_output / video_output.norm(dim=-1, keepdim=True)
    video_output = mean_pooling_for_similarity_visual(video_output, video_mask)
    video_output = video_output / video_output.norm(dim=-1, keepdim=True)
    text_output = text_output.squeeze(1)
    text_output = text_output / text_output.norm(dim=-1, keepdim=True)
    retrieve_logits = logit_scale * torch.matmul(text_output, video_output.t())
    return retrieve_logits

def calculate_recall_topn(all_text_features, all_text_masks, all_video_features, all_video_masks, topn, logit_scale, ground_truth):
    recalls = []
    for text_idx, (text_features, _) in enumerate(zip(all_text_features, all_text_masks)):
        simi = calculate_simi(text_features.unsqueeze(0), all_video_features, all_video_masks, logit_scale)
        simi = simi.cpu().detach().numpy().flatten()
        
        gt_videos = ground_truth[text_idx]
        # Get top N similar video indices
        top_n_indices = np.argsort(-simi)[:topn]
        
        # Calculate recall
        recall = sum(1 for gt in gt_videos if gt in top_n_indices) / len(gt_videos)
        recalls.append(recall)

    average_recall = np.mean(recalls)
    return average_recall