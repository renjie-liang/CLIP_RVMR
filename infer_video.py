import torch
import torch.nn as nn

from tqdm import tqdm
import json

from utils.utils_model import  save_model, load_model
from utils.setup import get_args, set_seed_logger
from utils.utils import LossTracker, TimeTracker, save_json

from dataloaders.data_dataloaders import prepare_dataloader_segment, prepare_dataloader_video
from modules.evaluate_lib import eval_epoch, grab_corpus_feature, eval_video_epoch
# from modules.modeling import CLIP4Clip
import time
from transformers import CLIPProcessor, CLIPModel
from modules.CLIPFineTuner import CLIPFineTuner
from torch import nn, optim
import torch.nn.functional as F

def main():
    global logger
    args = get_args()
    logger = set_seed_logger(args)
    logger.info("Arguments:\n%s", json.dumps(vars(args), indent=4))

    model = CLIPFineTuner(args.clip_model_name)
    model.freeze_layers(freeze_layer_count=args.freeze_layer_num)
    processor = CLIPProcessor.from_pretrained(args.clip_model_name)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = nn.DataParallel(model)
        model.to(device)
    else:
        device = torch.device("cpu")
        model.to(device)


    if args.data_name == "query_video":
        train_dataloader, corpus_dataloader, corpus_video_list, val_dataset, val_dataloader, test_dataset, test_dataloader  = prepare_dataloader_video(args, processor)
    else:
        train_dataloader, corpus_dataloader, corpus_video_list, val_dataset, val_dataloader, test_dataset, test_dataloader  = prepare_dataloader_segment(args, processor)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if args.checkpoint_path is not None:
        model, optimizer  = load_model(model, args.checkpoint_path, optimizer,  args.optimizer_path)
        logger.info(f"Load model from {args.checkpoint_path}")
        logger.info(f"Load optimizer from {args.optimizer_path}")

    
    model.eval()
    corpus_feature = grab_corpus_feature(model, corpus_dataloader, device) # len(vidoes) * L * 512 
    val_recalls = eval_video_epoch(model, val_dataset, val_dataloader, corpus_feature, device, corpus_video_list, args.recall_topk, "val")
    # test_recalls = eval_epoch(model, test_dataset, test_dataloader, corpus_feature, device, corpus_video_list, args.recall_topk, "test")


    # Log each recall value for the given topk values

    print(f"Search Space: {len(corpus_video_list)}")
    for topk in args.recall_topk:
        logger.info(f"VAL  Recall@{topk}: {val_recalls[topk]:.4f}")
        # logger.info(f"TEST Recall@{topk}: {test_recalls[topk]:.4f}\n")

if __name__ == "__main__":
    main()
