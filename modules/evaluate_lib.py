import numpy as np
import torch
from tqdm import tqdm
import h5py

def grab_corpus_feature(model, corpus_dataloader, device):
    
    h5file = "./data/features/segments_clip.h5"
    hf = h5py.File(h5file, 'w')

    all_video_features, all_video_masks = [], []
    model.eval()
    n = 0
    with torch.no_grad():
        # Process video features
        for batch in tqdm(corpus_dataloader, desc="Grab Videos Feature"):
            videos, video_masks = batch[0].to(device), batch[1].to(device)
            segment_names = batch[2]
            visual_output = model.module.get_video_features(videos)
            # Detach and move to CPU
            visual_output = visual_output.detach().cpu()
            video_masks = video_masks.detach().cpu()
            
            all_video_features.append(visual_output)
            all_video_masks.append(video_masks)
            
            for i in range(len(batch)):
                seg_name = segment_names[i]
                vfeat = visual_output[i]
                vmask = video_masks[i]
                
                truncated_vfeat = vfeat[vmask.bool()]
                # print(vfeat.shape, truncated_vfeat.shape)
                # print(vmask)
                # breakpoint()
                hf.create_dataset(f"{seg_name}", data=truncated_vfeat.numpy())
                
            # n += 1
            # if n > 50:
            #     break
        # Concatenate all video features and masks
        all_video_features = torch.cat(all_video_features, dim=0)
        all_video_masks = torch.cat(all_video_masks, dim=0)
    hf.close()

    return all_video_features, all_video_masks

def eval_epoch(model, eval_dataset, eval_dataloader, corpus_feature, device, corpus_videos, topk, set_type):
    
    segment_retrieval_gt = eval_dataset.segment_retrieval_gt 
    relevant_moment_gt = eval_dataset.relevant_moment_gt 
    
    
    all_video_features, all_video_masks = corpus_feature
    all_text_features = []
    model.eval()
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
    average_recall = calculate_recall_topk(model, all_text_features, all_video_features, all_video_masks, topk, segment_retrieval_gt, corpus_videos)
    # average_ndcg = calculate_NDCG(model, all_text_features, all_video_features, all_video_masks, topk, relevant_moment_gt, corpus_videos, set_type)
    
    return average_recall


def eval_video_epoch(model, eval_dataset, eval_dataloader, corpus_feature, device, corpus_videos, topk, set_type):
    
    ground_truth = eval_dataset.ground_truth 
    
    all_video_features, all_video_masks = corpus_feature
    all_text_features = []
    model.eval()
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

def calculate_recall_topk(model, all_text_features, all_video_features, all_video_masks, topks, ground_truth, corpus_videos):
    all_text_features = all_text_features.squeeze(1)
    all_text_features = all_text_features.contiguous()
    all_video_features = all_video_features.contiguous()
    # Calculate similarities in a batch
    simi_matrix = model.module.compute_similarity_matrix(all_text_features, all_video_features, all_video_masks)
    simi_matrix = simi_matrix.cpu().detach()
    recalls_dict = {topk: [] for topk in topks}

    for text_idx in tqdm(range(simi_matrix.shape[0]), desc="Calculate the Recall"):
        simi = simi_matrix[text_idx]
        gt_query = ground_truth[text_idx]
        # Calculate recall
        for topk in topks:
            # Get top N similar video indices
            top_n_indices = np.argsort(-simi)[:topk]
            top_n_simi, top_n_indices = torch.topk(simi, topk)
            top_n_video_names = [corpus_videos[i] for i in top_n_indices]
            # Calculate recall
            recall = sum(1 for gt in gt_query if gt in top_n_video_names) / len(gt_query)
            recalls_dict[topk].append(recall)

    # Calculate the average recall for each topk
    average_recalls = {topk: np.mean(recalls) for topk, recalls in recalls_dict.items()}
    
    return average_recalls



def calculate_NDCG(model, all_text_features, all_video_features, all_video_masks, topks, ground_truth, corpus_videos, set_type):
    all_text_features = all_text_features.squeeze(1)
    all_text_features = all_text_features.contiguous()
    all_video_features = all_video_features.contiguous()
    # Calculate similarities in a batch
    simi_matrix = model.module.compute_similarity_matrix(all_text_features, all_video_features, all_video_masks)
    simi_matrix = simi_matrix.cpu().detach()
    recalls_dict = {topk: [] for topk in topks}
    result = []
    
    for text_idx in tqdm(range(simi_matrix.shape[0]), desc="Calculate the NDCG"):
        simi = simi_matrix[text_idx]
        gt_query = ground_truth[text_idx]
        # Calculate recall
        for topk in topks:
            top_n_simi, top_n_indices = torch.sort(simi, descending=True)
            top_n_seg_names = [corpus_videos[i] for i in top_n_indices]

            per_result = {"query_id": gt_query["query_id"],
                          "relevant_moment": gt_query["relevant_moment"],
                          "predicted_seg_name": top_n_seg_names,
                          "predicted_seg_score": top_n_simi}
            result.append(per_result)
            break
    save_path = f"pred_result_{set_type}.pt"
    torch.save(result, save_path)
    print("save at", save_path)
