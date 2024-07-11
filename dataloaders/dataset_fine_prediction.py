import h5py
import torch

from torch.utils.data import Dataset
from utils.utils import load_json
from utils.tensor_utils import pad_sequences_1d, uniform_feature_sampling
import torch.nn.functional as F


def collate_fn(batch, max_vlen):
    
    meta_data, batch_data = {},  {}
    
    query_feat_mask = pad_sequences_1d([e["query_feat"] for e in batch], dtype=torch.float32, fixed_length=None)
    batch_data["query_feat"] = query_feat_mask[0]
    batch_data["query_mask"] = query_feat_mask[1] 
    
    coarse_mask, _ = pad_sequences_1d([e["coarse_moment_mask"] for e in batch], dtype=torch.float32, fixed_length=max_vlen)
    
    video_feat_mask = pad_sequences_1d([e["video_feat"] for e in batch], dtype=torch.float32, fixed_length=max_vlen)
    batch_data["video_feat"] = video_feat_mask[0] * coarse_mask.unsqueeze(-1) 
    batch_data["video_mask"] = video_feat_mask[1] * coarse_mask
    sub_feat_mask = pad_sequences_1d([e["sub_feat"] for e in batch], dtype=torch.float32, fixed_length=max_vlen)
    batch_data["sub_feat"] = sub_feat_mask[0] * coarse_mask.unsqueeze(-1) 
    batch_data["sub_mask"] = sub_feat_mask[1] * coarse_mask
    
    meta_data["query_id"] = [e["query_id"] for e in batch]
    meta_data["video_name"] = [e["video_name"] for e in batch]
    return  meta_data, batch_data


class FinePredictSegmentDataset(Dataset):
    def __init__(self, annotation_path, coarse_prediction_path, args):
        self.coarse_pred = load_json(coarse_prediction_path)
        self.coarse_pred = self.extend_coarse_pred(self.coarse_pred)
        
        self.annotations = load_json(annotation_path)
        self.corpus = load_json(args.corpus_path)
        
        self.relevant_moment_gt = self.get_relevant_moment_gt(self.annotations)

        self.desc_bert_h5 = h5py.File(args.desc_bert_path, "r")
        self.vid_feat_h5 = h5py.File(args.video_feat_path, "r")
        self.sub_bert_h5 = h5py.File(args.sub_bert_path, "r")
        self.max_vlen = args.max_vlen
        self.max_qlen = args.max_qlen
        
        self.start_offset = args.start_offset
        self.end_offset = args.end_offset
        
    def __len__(self):
        return len(self.coarse_pred)

    def __getitem__(self, idx):
        raw_data = self.coarse_pred[idx]
        
        query_id = int(raw_data["query_id"])
        video_name = raw_data["video_name"]
        duration = self.corpus[video_name]
        st, ed = raw_data["timestamp"]
        
        st = max(st-self.start_offset, 0)
        ed = min(ed-self.end_offset, duration)
        coarse_timestamp = [st, ed]
        
        model_inputs = dict()
        model_inputs["query_feat"] = self.grab_qfeat(self.desc_bert_h5, str(query_id), self.max_qlen)
        
        coarse_moment_mask = self.get_coarse_moment_mask(coarse_timestamp, self.max_vlen)
        
        model_inputs["video_feat"] = self.grab_vfeat(self.vid_feat_h5, video_name, self.max_vlen, coarse_moment_mask)
        model_inputs["sub_feat"] = self.grab_vfeat(self.sub_bert_h5, video_name, self.max_vlen, coarse_moment_mask)

        model_inputs["coarse_moment_mask"] = coarse_moment_mask
        
        model_inputs["query_id"] = query_id
        model_inputs["video_name"] = video_name
        return model_inputs
    
    def grab_qfeat(self, h5_data, key, max_len):
        feat = h5_data[key][...][:max_len]  # Assuming you want to extract all data from h5_data[key]
        feat = torch.from_numpy(feat)
        feat = F.normalize(feat, dim=-1)
        return feat
    
    def grab_vfeat(self, h5_data, key, max_len, mask):
        feat = uniform_feature_sampling(h5_data[key][...], max_len)
        feat = torch.from_numpy(feat)
        feat = feat * mask[:len(feat)].unsqueeze(-1)  # Applying mask to feat
        feat = F.normalize(feat, dim=-1)
        return feat
    
    def get_coarse_moment_mask(self, coarse_timestamp, max_vlen):
        s, e = coarse_timestamp
        coarse_mask = torch.zeros(max_vlen)
        coarse_mask[s:e+1] = 1
        return coarse_mask
    
    def get_relevant_moment_gt(self, annotations):
        gt_all = {}
        for record in annotations:
            gt_all[record["query_id"]] = record["relevant_moment"]
        return gt_all
    
    def extend_coarse_pred(self, coarse_pred):
        new_pred = []
        for k, preds in coarse_pred.items():
            for p in preds:
                p.update({"query_id": k})
                new_pred.append(p)
        return new_pred
