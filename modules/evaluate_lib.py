import numpy as np
import torch
from tqdm import tqdm
import h5py
from utils.ndcg_iou import calculate_iou

def grab_corpus_feature(model, corpus_dataloader, device):
    
    # h5file = "./data/features/segments_clip.h5"
    # hf = h5py.File(h5file, 'w')

    all_vfeat, all_vmask = [], []
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
            
            all_vfeat.append(visual_output)
            all_vmask.append(video_masks)
            
            # for i in range(len(batch[0])):
            #     seg_name = segment_names[i]
            #     vfeat = visual_output[i]
            #     vmask = video_masks[i]
                
            #     truncated_vfeat = vfeat[vmask.bool()]
            #     print(vmask)
            #     print(vfeat.shape, truncated_vfeat.shape)
            #     print(seg_name, truncated_vfeat)
            #     breakpoint()
                # hf.create_dataset(f"{seg_name}", data=truncated_vfeat.numpy())
            # if len(all_vfeat) > 100:
            #     break
        # Concatenate all video features and masks
        all_vfeat = torch.cat(all_vfeat, dim=0)
        all_vmask = torch.cat(all_vmask, dim=0)
    # hf.close()
    return all_vfeat, all_vmask

def grab_corpus_feature_from_file(vfeat_path, max_vlen, corpus_list):
    all_vfeat, all_vmask = [], []

    with h5py.File(vfeat_path, 'r') as h5file:
        for seg_name in corpus_list:
        # for seg_name in h5file.keys():
            visual_output = h5file[seg_name][()]
            video_mask = [1] * len(visual_output) + [0] * (max_vlen - len(visual_output))
            video_mask = torch.tensor(video_mask[:max_vlen], dtype=torch.long)

            if len(visual_output) < max_vlen:
                padding = torch.zeros((max_vlen - len(visual_output), visual_output.shape[1]))
                visual_output = torch.cat((torch.tensor(visual_output), padding), dim=0)
            else:
                visual_output = torch.tensor(visual_output[:max_vlen])
            
            all_vfeat.append(visual_output)
            all_vmask.append(video_mask)
        all_vfeat = torch.stack(all_vfeat, dim=0)
        all_vmask = torch.stack(all_vmask, dim=0)
    return all_vfeat, all_vmask

def eval_epoch(model, eval_dataset, eval_dataloader, corpus_feature, device, corpus_list, topk, set_type):
    
    segment_retrieval_gt = eval_dataset.segment_retrieval_gt 
    relevant_moment_gt = eval_dataset.relevant_moment_gt 
    
    all_tfeat, all_query_id = [], []
    model.eval()
    with torch.no_grad():
        # Process text features
        for batch in tqdm(eval_dataloader, desc="Get Texts Feature"):
            
            meta_data, batch_input = batch
            batch_input = {k: v.to(device) for k, v in batch_input.items()}
            text_output = model.module.clip_model.get_text_features(input_ids=batch_input["text_token_id"], 
                                                                    attention_mask=batch_input["text_mask"])
            # Detach and move to CPU
            text_output = text_output.detach().cpu()
            # attention_mask = attention_mask.detach().cpu()
            all_tfeat.append(text_output)
            all_query_id.extend(meta_data["query_id"])
        # Concatenate all text features and masks
        all_tfeat = torch.cat(all_tfeat, dim=0)
    # ----------------------------------
    # 2. calculate the similarity
    # ----------------------------------
    
    # average_recall = calculate_recall(model, all_tfeat, all_vfeat, all_vmask, topk, segment_retrieval_gt, corpus_list)
    average_recall = calculate_recall(model, all_query_id, all_tfeat, corpus_feature, topk, relevant_moment_gt, segment_retrieval_gt, corpus_list, set_type)
    
    return average_recall


# def eval_video_epoch(model, eval_dataset, eval_dataloader, corpus_feature, device, corpus_list, topk, set_type):
#     ground_truth = eval_dataset.ground_truth 
#     all_vfeat, all_vmask = corpus_feature
#     all_tfeat = []
#     model.eval()
#     with torch.no_grad():
#         # Process text features
#         for batch in tqdm(eval_dataloader, desc="Get Texts Feature"):
#             batch = [b.to(device) for b in batch]
#             text_input_ids, attention_mask = batch
#             text_output = model.module.clip_model.get_text_features(input_ids=text_input_ids, attention_mask=attention_mask)
#             # Detach and move to CPU
#             text_output = text_output.detach().cpu()
#             attention_mask = attention_mask.detach().cpu()
#             all_tfeat.append(text_output)
#         # Concatenate all text features and masks
#         all_tfeat = torch.cat(all_tfeat, dim=0)
        
#     average_recall = calculate_recall(model, all_tfeat, all_vfeat, all_vmask, topk, ground_truth, corpus_list)
#     return average_recall


import torch
import numpy as np
from tqdm import tqdm

# def calculate_recall_topk(model, all_tfeat, all_vfeat, all_vmask, topks, ground_truth, corpus_list):
#     all_tfeat = all_tfeat.squeeze(1)
#     all_tfeat = all_tfeat.contiguous()
#     all_vfeat = all_vfeat.contiguous()
#     # Calculate similarities in a batch
#     simi_matrix = model.module.compute_similarity_matrix(all_tfeat, all_vfeat, all_vmask)
#     simi_matrix = simi_matrix.cpu().detach()
#     recalls_dict = {topk: [] for topk in topks}

#     for idx in tqdm(range(simi_matrix.shape[0]), desc="Calculate the Recall"):
#         simi = simi_matrix[idx]
#         gt_query = ground_truth[idx]
#         # Calculate recall
#         for topk in topks:
#             # Get top N similar video indices
#             top_n_indices = np.argsort(-simi)[:topk]
#             top_n_simi, top_n_indices = torch.topk(simi, topk)
#             top_n_seg_names = [corpus_list[i] for i in top_n_indices]
#             # Calculate recall
#             recall = sum(1 for gt in gt_query if gt in top_n_seg_names) / len(gt_query)
#             recalls_dict[topk].append(recall)
#     # Calculate the average recall for each topk
#     average_gt_recall = {topk: np.mean(recalls) for topk, recalls in recalls_dict.items()}
#     print("segment recall", average_gt_recall)
#     return average_gt_recall


def calculate_recall(model, all_query_id, all_tfeat, corpus_feature, topks, gt_moment, gt_segment, corpus_list, set_type):

    '''
    gt_moment = [{"video_name": vname1, "timestamp":  ts1, "relevance":  s1},
                 {"video_name": vname2, "timestamp":  ts2, "relevance":  s2}]
                 
    gt_segment = ["seg_name1", "seg_name2", "seg_name3"]
    '''    
    
    all_vfeat, all_vmask = corpus_feature
    all_tfeat = all_tfeat.squeeze(1)
    all_tfeat = all_tfeat.contiguous()
    all_vfeat = all_vfeat.contiguous()
    # Calculate similarities in a batch
    simi_matrix = model.module.compute_similarity_matrix(all_tfeat, all_vfeat, all_vmask)
    simi_matrix = simi_matrix.cpu().detach()
    gt_recall_video_dict = {topk: [] for topk in topks}
    gt_recall_seg_dict = {topk: [] for topk in topks}
    unique_video_nums_dict = {topk: [] for topk in topks}
    pd_recall_seg_dict = {topk: [] for topk in topks}
    
    for idx in tqdm(range(simi_matrix.shape[0]), desc="Calculate the NDCG"):
        simi = simi_matrix[idx]
        query_id = all_query_id[idx]
        one_gt_seg = gt_segment[query_id]
        one_gt_vid = gt_moment[query_id]
        
        breakpoint()
        # Calculate recall
        for topk in topks:
            # breakpoint()
            top_n_simi, top_n_indices = torch.topk(simi, topk)
            top_n_seg_names = [corpus_list[i] for i in top_n_indices]
            # ========= gt_recall_seg ====================
            pred_video_ts = []
            seg_duration = 4
            for seg_name in top_n_seg_names:
                parts = seg_name.rsplit('_', 1)
                video_name  = parts[0]
                seg_idx = int(parts[1])
                start = seg_idx * seg_duration
                end = (seg_idx + 1) * seg_duration
                pred_video_ts.append([video_name, [start, end]])
            
            recall_seg = 0
            for i in one_gt_vid:
                gt_vn = i["video_name"]
                gt_ts = i["timestamp"]
                for pd_vn, pd_ts in pred_video_ts:
                    if gt_vn == pd_vn and calculate_iou(pd_ts[0], pd_ts[1], gt_ts[0], gt_ts[1]) > 0:
                        recall_seg += 1
                        break
                
            recall_seg = recall_seg / len(one_gt_vid)
            gt_recall_seg_dict[topk].append(recall_seg)
            # ========= gt_recall_seg ====================
            
            
            # ========= gt_recall_video =================
            gt_video_names = [i["video_name"] for i in one_gt_vid]
            top_n_video_names =  [seg_name.rsplit('_', 1)[0] for seg_name in top_n_seg_names]
            top_n_video_names_unique = set(top_n_video_names)
            recall_video = sum(1 for gt in gt_video_names if gt in top_n_video_names) / len(gt_video_names)
            gt_recall_video_dict[topk].append(recall_video)
            unique_video_nums_dict[topk].append(len(top_n_video_names_unique))
            # ========= gt_recall_video =================


            # ========= pd_recall_seg =================
            # Calculate recall
            if len(one_gt_seg) > 0:
                recall = sum(1 for gt in one_gt_seg if gt in top_n_seg_names) / len(one_gt_seg)
                pd_recall_seg_dict[topk].append(recall)
            # ========= pd_recall_seg =================

  
    avg_gt_recall_seg = {topk: np.mean(recall) for topk, recall in gt_recall_seg_dict.items()}
    avg_gt_recall_video = {topk: np.mean(recall) for topk, recall in gt_recall_video_dict.items()}
    average_unique_video_nums = {topk: np.mean(unique_video_nums) for topk, unique_video_nums in unique_video_nums_dict.items()}
    avg_pd_recall_seg = {topk: np.mean(recall) for topk, recall in pd_recall_seg_dict.items()}
    print("ground_truht_recall_by_segment =", avg_gt_recall_seg)
    print("ground_truht_recall_by_video =", avg_gt_recall_video)
    print("unique_video_number =", average_unique_video_nums)
    print("prediction_recall_by_seg =", avg_pd_recall_seg)
    return avg_gt_recall_video


            # per_result = {"query_id": gt_query["query_id"],
            #               "relevant_moment": gt_query["relevant_moment"],
            #               "predicted_seg_name": top_n_seg_names,
            #               "predicted_seg_score": top_n_simi}
            # result.append(per_result)
    # save_path = f"pred_result_{set_type}.pt"
    # torch.save(result, save_path)
    # print("save at", save_path)