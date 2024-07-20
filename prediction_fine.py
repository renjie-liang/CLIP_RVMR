import torch
from utils.utils import load_json
import numpy as np
import math
from tqdm import tqdm
from  utils.ndcg_iou import calculate_ndcg_iou
from utils.utils import get_logger
from utils.utils_model import prepare_ReLoCLNet, resume_ReLoCLNet
from utils.tensor_utils import pad_sequences_1d, find_max_triples
import torch.nn.functional as F
from utils.setup_ReLoCLNet import get_args
from dataloaders.dataset_fine_prediction import FinePredictSegmentDataset, collate_fn
from torch.utils.data import DataLoader
from easydict import EasyDict
from collections import defaultdict
'''
Todo:
1. mask or truncate
    1.1 mask: 
'''


opt = get_args()
opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
opt.batch_size_eval = 128
opt.num_workers = 4
opt.val_coarse_path     = "./data/coarse_prediction.json"
opt.val_path            = "/home/share/rjliang/TVR_Ranking/val.json"
opt.corpus_path         = "/home/share/rjliang/TVR_Ranking/video_corpus.json"
opt.desc_bert_path      = "/home/renjie.liang/11_TVR-Ranking/ReLoCLNet/data/features/query_bert.h5"
opt.video_feat_path     = "/home/renjie.liang/11_TVR-Ranking/ReLoCLNet/data/features/tvr_i3d_rgb600_avg_cl-1.5.h5"
opt.sub_bert_path       = "/home/renjie.liang/11_TVR-Ranking/ReLoCLNet/data/features/tvr_sub_pretrained_w_sub_query_max_cl-1.5.h5"
opt.checkpoint          = "/home/renjie.liang/11_TVR-Ranking/ReLoCLNet/results/tvr_ranking/top20/best_model.pt"

opt.start_offset = 0
opt.end_offset = 0


SEG_DURATION = 4
KS = [10, 20, 40]
TS = [0.3, 0.5, 0.7]


logger = get_logger(opt.results_path, opt.exp_id)
model = prepare_ReLoCLNet(opt, logger)
model, _, _ = resume_ReLoCLNet(logger, opt, model)
coarse_pred_set = FinePredictSegmentDataset( annotation_path=opt.val_path, coarse_prediction_path = opt.val_coarse_path, args = opt )
coarse_pred_loader = DataLoader(coarse_pred_set, collate_fn=lambda batch: collate_fn(batch, max_vlen=opt.max_vlen), batch_size=opt.batch_size_eval, num_workers=opt.num_workers, shuffle=False, pin_memory=True)


all_fine_pred = defaultdict(list)
all_gt = coarse_pred_set.relevant_moment_gt

for batch in tqdm(coarse_pred_loader, desc="Fine Predict", total=len(coarse_pred_loader)):
    meta_data, batch_input = batch
    batch_input = {k: v.to(opt.device) for k, v in batch_input.items()}
    st_prob, ed_prob = model.infer_onevideo(**batch_input)
    
    st_prob = F.softmax(st_prob, dim=-1) 
    ed_prob = F.softmax(ed_prob, dim=-1)
    
    st_prob = st_prob * batch_input["video_mask"]
    ed_prob = ed_prob * batch_input["video_mask"]

    fine_pred_batch = find_max_triples(st_prob.detach().cpu(), ed_prob.detach().cpu(), top_n=1)
    # fine_pred_batch = find_max_triples(st_prob.detach().cpu().numpy(), ed_prob, top_n=1)
    for query_id, video_name, preds in zip(meta_data["query_id"], meta_data["video_name"], fine_pred_batch):
        for p in preds:
            start, end, model_scores  = p
            # start *= opt.clip_length
            # end *= opt.clip_length
            tmp = {
                    "video_name": video_name, 
                    "timestamp": [start, end],
                    "model_scores": model_scores}
            all_fine_pred[query_id].append(tmp)
            

average_ndcg = calculate_ndcg_iou(all_gt, all_fine_pred , TS, KS)
for K, vs in average_ndcg.items():
    for T, v in vs.items():
        logger.info(f"VAL NDCG@{K}, IoU={T}: {v:.6f}")


# qsub -I -l select=1:ngpus=1 -P gs_slab -q slab_gpu8
# cd /home/renjie.liang/12_RVMR_IR/CLIP_RVMR ; conda activate py11 ; python prediction_fine.py --exp_id fine_prediction
