import torch
import torch.nn as nn

from tqdm import tqdm

from utils.utils_model import prep_optimizer, save_model, load_model
from utils.setup import get_args, set_seed_logger

from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from dataloaders.data_dataloaders import dataloader_TVRR_train, dataloader_TVRR_eval, dataloader_TVRR_video_corpus
from evaluate_video_retrieval import eval_epoch, grab_corpus_feature
from modules.modeling import CLIP4Clip
import time

def main():
    global logger
    args = get_args()
    logger = set_seed_logger(args)
    logger.info(vars(args))
    
    tokenizer = ClipTokenizer()
    if args.checkpoint_path is not None:
        logger.info(f"Load model from {args.checkpoint_path}")
        model = load_model(args, args.checkpoint_path)
            # checkpoint = torch.load(args.resume_model, map_location='cpu')
    else:
        model = CLIP4Clip.from_pretrained(task_config=args)
    
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = nn.DataParallel(model)
        model.to(device)
    else:
        device = torch.device("cpu")
        model.to(device)
            

    ## ####################################
    # freeze testing
    ## ####################################
    assert args.freeze_layer_num <= 12 and args.freeze_layer_num >= -1
    if hasattr(model, "clip") and args.freeze_layer_num > -1:
        for name, param in model.clip.named_parameters():
            # top layers always need to train
            if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                    or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0:
                continue    # need to train
            elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= args.freeze_layer_num:
                    continue    # need to train

            if args.linear_patch == "3d" and name.find("conv2."):
                continue
            else:
                # paramenters which < freeze_layer_num will be freezed
                param.requires_grad = False

    train_dataloader, train_dataset = dataloader_TVRR_train(args.train_path, args, tokenizer)
    val_dataloader, val_dataset = dataloader_TVRR_eval(args.val_path, args, tokenizer)
    test_dataloader, test_dataset = dataloader_TVRR_eval(args.test_path, args, tokenizer)
    corpus_dataloader = dataloader_TVRR_video_corpus(args.corpus_path, args)

    num_train_optimization_steps = len(train_dataloader) * args.epochs
    optimizer = prep_optimizer(args, model, num_train_optimization_steps, logger, coef_lr=args.coef_lr)

        
    best_score = -1.0
    time_grab_data = 0
    time_to_divice = 0
    time_forward = 0
    time_backward = 0
    start_time = time.time()
    
    
    ## ####################################
    # train and eval
    ## ####################################
    for epoch in range(args.epochs):
        # torch.cuda.empty_cache()
        model.train()
        for step, batch_data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="TRAIN"):
            step += 1
            optimizer.zero_grad()
            
            time_grab_data += time.time()  - start_time
            start_time = time.time()
            
            batch_data = [b.to(device) for b in batch_data]
            text_ids, text_masks, videos, video_masks = batch_data
            
            time_to_divice += time.time()  - start_time
            start_time = time.time()
            loss = model(text_ids, text_masks, videos, video_masks)
            time_forward += time.time()  - start_time
            start_time = time.time()
            
            if loss.dim() > 0:  # Check if loss is not a scalar
                loss = loss.mean()  # Apply reduction to make it a scalar
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            time_backward += time.time()  - start_time
            start_time = time.time()

            if step % args.step_log == 0 or step % len(train_dataloader) == 0:
                logger.info(f"Epoch: {epoch}/{args.epochs}, Step: {step}/{len(train_dataloader)}, Loss: {loss.item():.6f}")
                print(f"time_grab_data: {time_grab_data:.4f}")
                print(f"time_to_divice: {time_to_divice:.4f}")
                print(f"time_forward: {time_forward:.4f}")
                print(f"time_backward: {time_backward:.4f}")
            
            if step % args.step_eval == 0 or step % len(train_dataloader) == 0:
            # if step % 1 == 0 or step % len(train_dataloader) == 0:
                corpus_feature = grab_corpus_feature(model, corpus_dataloader, device)
                val_r100 = eval_epoch(model, val_dataloader, corpus_feature, device, val_dataset.ground_truth)
                logger.info(f"\nVAL Recall@100: {val_r100:.4f}\n")
                test_r100 = eval_epoch(model, test_dataloader, corpus_feature, device, test_dataset.ground_truth)
                logger.info(f"\nTEST Recall@100: {test_r100:.4f}\n")

                if val_r100 > best_score:
                    best_score = val_r100
                    save_model(args, model, optimizer, suffix="best", logger=logger)
                    logger.info(f"BEST VAL Recall@100: {val_r100:.4f}")
                    logger.info(f"BEST TEST Recall@100: {test_r100:.4f}")

if __name__ == "__main__":
    main()
