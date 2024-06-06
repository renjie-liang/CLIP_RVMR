import torch
import torch.nn as nn

from tqdm import tqdm
import json

from utils.utils_model import prep_optimizer, save_model, load_model
from utils.setup import get_args, set_seed_logger
from utils.utils import LossTracker, TimeTracker, save_json

from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from dataloaders.data_dataloaders import prepare_dataloader_segment_CLIP, prepare_dataloader_video_CLIP
from evaluate_video_retrieval import eval_epoch, grab_corpus_feature
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

    if args.data_name == "query_segment":
        train_dataloader, corpus_dataloader, corpus_video_list, val_dataloader, val_gt, test_dataloader, test_gt  = prepare_dataloader_segment_CLIP(args, processor)
    elif args.data_name == "query_video_clip":
        train_dataloader, corpus_dataloader, corpus_video_list, val_dataloader, val_gt, test_dataloader, test_gt  = prepare_dataloader_video_CLIP(args, processor)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= 5 * len(train_dataloader))
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size * len(train_dataloader), gamma=args.lr_gamma)

    if args.checkpoint_path is not None:
        model, optimizer  = load_model(model, args.checkpoint_path, optimizer,  args.optimizer_path)
        logger.info(f"Load model from {args.checkpoint_path}")
        logger.info(f"Load optimizer from {args.optimizer_path}")

    best_score = -1.0
    time_tracker = TimeTracker()
    epoch_loss_tracker = LossTracker()
    
    for epoch in range(args.num_epochs):
        model.train()
            
        # torch.cuda.empty_cache()
        time_tracker.start("grab_data")
        for step, batch_data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="TRAIN"):
            step += 1
            time_tracker.stop("grab_data")
            time_tracker.start("to_device")
            batch_data = [b.to(device, non_blocking=True) for b in batch_data]
            text_ids, text_masks, videos, video_masks = batch_data
            optimizer.zero_grad()
            time_tracker.stop("to_device")

            time_tracker.start("forward")
            loss = model(text_ids, text_masks, videos, video_masks)
            time_tracker.stop("forward")

            time_tracker.start("backward")
            if loss.dim() > 0:  # Check if loss is not a scalar
                loss = loss.mean()  # Apply reduction to make it a scalar
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            time_tracker.stop("backward")
            time_tracker.start("grab_data")
            epoch_loss_tracker.update(loss.item())
            
            if step % args.step_log == 0 or step % len(train_dataloader) == 0:
                average_loss = epoch_loss_tracker.average_loss()
                logger.info(f"Epoch: {epoch + 1}/{args.num_epochs}, Step: {step}/{len(train_dataloader)}, Loss: {average_loss:.6f}")
                
                print("-------------------------")
                print(time_tracker.report())
                epoch_loss_tracker.reset()
                time_tracker.reset_all()
                
                for i in range(torch.cuda.device_count()):
                    print(f"Memory Allocated on GPU {i}: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
                    print(f"Memory Cached on GPU {i}: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
                print("-------------------------")
            
            if step % args.step_eval == 0 or step % len(train_dataloader) == 0:
            # if step % 1 == 0 or step % len(train_dataloader) == 0:
                corpus_feature = grab_corpus_feature(model, corpus_dataloader, device) # len(vidoes) * L * 512 
                # Assuming eval_epoch returns a dictionary with recall values for each topk
                val_recalls = eval_epoch(model, val_dataloader, corpus_feature, device, val_gt, corpus_video_list, args.recall_topk)
                test_recalls = eval_epoch(model, test_dataloader, corpus_feature, device, test_gt, corpus_video_list, args.recall_topk)
                model.train()

                # Log each recall value for the given topk values
                for topk in args.recall_topk:
                    logger.info(f"VAL  Recall@{topk}: {val_recalls[topk]:.4f}")
                    logger.info(f"TEST Recall@{topk}: {test_recalls[topk]:.4f}\n")

                # Use the first topk value as the criterion for best score
                first_topk = args.recall_topk[0]
                if val_recalls[first_topk] > best_score:
                    best_score = val_recalls[first_topk]
                    save_model(args, model, optimizer, suffix="best", logger=logger)
                    logger.info(f"BEST VAL  Recall@{first_topk}: {val_recalls[first_topk]:.4f}")
                    logger.info(f"BEST TEST Recall@{first_topk}: {test_recalls[first_topk]:.4f}")

if __name__ == "__main__":
    main()
