import torch
import torch.nn as nn

from tqdm import tqdm
import json

from utils.utils_model import prep_optimizer, save_model, load_model
from utils.setup import get_args, set_seed_logger
from utils.utils import LossTracker, TimeTracker

from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from dataloaders.data_dataloaders import prepare_dataloader_video, prepare_dataloader_segment, prepare_dataloader_video_CLIP
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
    # if args.checkpoint_path is not None:
    #     logger.info(f"Load model from {args.checkpoint_path}")
    #     model = load_model(args, args.checkpoint_path)
    #         # checkpoint = torch.load(args.resume_model, map_location='cpu')
    model = CLIPFineTuner(args.clip_model_name)
    model.freeze_layers(freeze_layer_count=args.freeze_layer_count)
    processor = CLIPProcessor.from_pretrained(args.clip_model_name)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = nn.DataParallel(model)
        model.to(device)
    else:
        device = torch.device("cpu")
        model.to(device)

    if args.data_name == "tvrr_segment":
        train_dataloader, val_dataloader, test_dataloader, corpus_dataloader, corpus_videos, val_gt, test_gt  = prepare_dataloader_segment(args, tokenizer)
    elif args.data_name == "tvrr_video":
        train_dataloader, val_dataloader, test_dataloader, corpus_dataloader, corpus_videos, val_gt, test_gt  = prepare_dataloader_video(args, tokenizer)
    elif args.data_name == "query_video_clip":
        train_dataloader, corpus_dataloader, corpus_video_list, val_dataloader, val_gt, test_dataloader, test_gt  = prepare_dataloader_video_CLIP(args, processor)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    best_score = -1.0
    time_tracker = TimeTracker()
    epoch_loss_tracker = LossTracker()
    
    model.train()
    for epoch in range(args.num_epochs):
        # torch.cuda.empty_cache()
        time_tracker.start("grab_data")
        for step, batch_data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="TRAIN"):
            step += 1
            
            time_tracker.stop("grab_data")
            time_tracker.start("to_device")
            batch_data = [b.to(device) for b in batch_data]
            text_ids, text_masks, videos, video_masks, sim_masks = batch_data
            # breakpoint()
            optimizer.zero_grad()
            time_tracker.stop("to_device")

            time_tracker.start("forward")
            # text_features, video_features = model(text_ids, text_masks, videos, video_masks, sim_masks)
            # sim_matrix = model.module.compute_similarity_matrix(text_features, video_features, video_masks)

            loss = model(text_ids, text_masks, videos, video_masks, sim_masks)
            # move the sim_matrix and sim_masks into a same device, because the sim_mask will be split to <batch_size, batch_size/n_gpus>
            # sim_matrix = sim_matrix.to("cuda:0")
            # sim_masks = sim_masks.to("cuda:0")
            # sim_matrix = torch.stack(sim_matrix, dim=0)
            # print("sim_matrix", sim_matrix.shape)
            # print("sim_masks", sim_masks.shape) 
            # sim_matrix_masked = sim_matrix * sim_masks
            # Compute the contrastive loss
            # loss_text_to_video = F.cross_entropy(sim_matrix_masked, sim_masks.max(1)[1])
            # loss_video_to_text = F.cross_entropy(sim_matrix_masked.t(), sim_masks.max(0)[1])
            # loss = (loss_text_to_video + loss_video_to_text) / 2
            
            
            time_tracker.stop("forward")
            time_tracker.start("backward")
            if loss.dim() > 0:  # Check if loss is not a scalar
                loss = loss.mean()  # Apply reduction to make it a scalar
            loss.backward()
            optimizer.step()
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
                val_recall = eval_epoch(model, val_dataloader, corpus_feature, device, val_gt, corpus_video_list, args.recall_topk)
                test_recall = eval_epoch(model, test_dataloader, corpus_feature, device, test_gt, corpus_video_list, args.recall_topk)
                
                logger.info()
                logger.info(f"VAL  Recall@{args.recall_topk}: {val_recall:.4f}")
                logger.info(f"TEST Recall@{args.recall_topk}: {test_recall:.4f}\n")

                if val_recall > best_score:
                    best_score = val_recall
                    save_model(args, model, optimizer, suffix="best", logger=logger)
                    logger.info(f"BEST VAL  Recall@{args.recall_topk}: {val_recall:.4f}")
                    logger.info(f"BEST TEST Recall@{args.recall_topk}: {test_recall:.4f}")
        scheduler.step()

if __name__ == "__main__":
    main()
