import torch
from tqdm import tqdm

from utils.utils_model import init_model, prep_optimizer
from utils.setup import get_args, set_seed_logger, init_device

from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from dataloaders.data_dataloaders import dataloader_TVRR_train, dataloader_TVRR_eval, dataloader_TVRR_video_corpus
from evaluate_video_retrieval import eval_epoch


def main():
    global logger
    args = get_args()
    args, logger = set_seed_logger(args)
    device = init_device(logger)
    tokenizer = ClipTokenizer()
    model = init_model(args, device)
    logger.info("---------------------------------")
    logger.info(vars(args))
    logger.info("---------------------------------")
    
    
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

    ## ####################################
    # train and eval
    ## ####################################
    num_train_optimization_steps = len(train_dataloader) * args.epochs
    coef_lr = args.coef_lr
    optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, coef_lr=coef_lr)

    best_score = 0.00001
    best_output_model_file = "None"
    ## ##############################################################
    # resume optimizer state besides loss to continue train
    ## ##############################################################
    resumed_epoch = 0
    if args.resume_model:
        checkpoint = torch.load(args.resume_model, map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        resumed_epoch = checkpoint['epoch']+1
        resumed_loss = checkpoint['loss']
    
    global_step = 0
    for epoch in range(resumed_epoch, args.epochs):
        
        # torch.cuda.empty_cache()
        model.train()
        log_step = args.step_log
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="TRAIN"):
            batch = [b.to(device) for b in batch]
            text_ids, text_masks, videos, video_masks = batch
            loss = model(text_ids, text_masks, videos, video_masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scheduler is not None:
                scheduler.step() 
            optimizer.step()
            optimizer.zero_grad()

            for i in range(torch.cuda.device_count()):
                print(f"Memory Allocated on GPU {i}: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
                print(f"Memory Cached on GPU {i}: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")

            global_step += 1
            if global_step % log_step == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader), "-".join([str('%.9f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss))
    #++++++++++++++++++++++++++++++++
        
            if global_step % args.step_eval == 0:
                # logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
                val_r100 = eval_epoch(args, model, val_dataloader, corpus_dataloader, device, val_dataset.ground_truth)
                logger.info(f"VAL Recall@100: {val_r100}")
                test_r100 = eval_epoch(args, model, test_dataloader, corpus_dataloader, device, test_dataset.ground_truth)
                logger.info(f"TEST Recall@100: {test_r100}")


if __name__ == "__main__":
    main()
