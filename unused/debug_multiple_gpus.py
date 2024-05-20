
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from modules.modeling import CLIP4Clip
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import os
from easydict import EasyDict
from modules.optimization import BertAdam


task_config = {"cross_model": 'cross-base',
          "max_words": 32,
          "max_frames": 12,
          "max_position_embeddings": 128,
          "loose_type": True,
          "pretrained_clip_name": "ViT-B/16",
          "local_rank": 0,
        }
task_config = EasyDict(task_config)

model = CLIP4Clip.from_pretrained(task_config)
if torch.cuda.is_available():
    device = torch.device("cuda")
    model = nn.DataParallel(model)  # This uses all available GPUs
    model.to(device)
else:
    device = torch.device("cpu")
    model.to(device)

# Create dummy dataset and dataloader
dataset = TensorDataset(torch.randint(0, 1000, (100, 32)),
                        torch.randint(1, 2, (100,32)),
                        torch.randint(0, 256, (100, 1, 12, 1, 3, 224, 224), dtype=float),
                        torch.randint(1, 2, (100,12)),
                        )
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)


def prep_optimizer(model):
    coef_lr = 1
    lr = 0.0001
    # if hasattr(model, 'module'):
    #     model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]

    weight_decay = 0.2
    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': lr * coef_lr},
        {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
        {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': lr * coef_lr},
        {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters, lr=lr,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         weight_decay=weight_decay, max_grad_norm=1.0)
    return optimizer

optimizer =  prep_optimizer(model)
# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for batch_data in dataloader:
        text_ids, text_masks, videos, video_masks = batch_data
        text_ids, text_masks, videos, video_masks = text_ids.to(device), text_masks.to(device), videos.to(device), video_masks.to(device)

        optimizer.zero_grad()
        loss = model(text_ids, text_masks, videos, video_masks)
        print(loss)
        if loss.dim() > 0:  # Check if loss is not a scalar
            loss = loss.mean()  # Apply reduction to make it a scalar
        loss.backward()
        optimizer.step()

        print("-------------------------")
        for i in range(torch.cuda.device_count()):
            print(f"Memory Allocated on GPU {i}: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"Memory Cached on GPU {i}: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
        print("-------------------------")
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')



# qsub -I -l select=1:ngpus=2 -P gs_slab -q slab_gpu8
# cd /home/renjie.liang/12_RVMR_IR/CLIP4Clip ; conda activate py11; python debug_multiple_gpus.py 
# python -m torch.distributed.launch --nproc_per_node=2  debug_multiple_gpus.py 