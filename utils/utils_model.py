import torch, os
from modules.optimization import BertAdam
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CLIP4Clip
import torch.nn as nn


def save_model(args, model, optimizer, suffix, logger):
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(args.output_dir, f"{suffix}_model.bin")
    optimizer_state_file = os.path.join(args.output_dir, f"{suffix}_optimizer.bin")
    torch.save(model_to_save.state_dict(), output_model_file)
    torch.save(optimizer.state_dict(), optimizer_state_file)
    
    logger.info(f"Model saved to {output_model_file}")
    logger.info(f"Optimizer saved to {optimizer_state_file}")
    return output_model_file


def load_model(model, ckpt_path, optimizer=None, optimizer_path=None):
    state_dict = torch.load(ckpt_path)
    if hasattr(model, 'module'):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    
    optimizer_data = torch.load(optimizer_path)
    optimizer.load_state_dict(optimizer_data)
    return model, optimizer


def prep_optimizer(args, model, num_train_optimization_steps, logger):

    # if hasattr(model, 'module'):
    #     model = model.module

    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    # decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    # no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    # decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    # decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]

    # no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    # no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]

    weight_decay = 0.2
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': args.lr * coef_lr},
    #     {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
    #     {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
    #     {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0}
    # ]

    optimizer = BertAdam(model.param, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)
    if args.optimizer_path is not None:
        optimizer_data = torch.load(args.optimizer_path, map_location='cpu')
        optimizer.load_state_dict(optimizer_data)                                           
        logger.info(f"Load optimizer from {args.optimizer_path}")
    return optimizer

