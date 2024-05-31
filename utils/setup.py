import random
import torch
import os
import numpy as np
import argparse

from utils.utils import get_logger

import argparse

def get_args(description='CLIP4Clip on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    
    # Paths
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training dataset")
    parser.add_argument("--val_path", type=str, required=True, help="Path to the validation dataset")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing video frames")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to the video corpus JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    # parser.add_argument("--checkpoint_path", type=str, help="Path to save model checkpoints")
    # parser.add_argument("--optimizer_path", type=str, help="Path to save optimizer state")

    # Experiment settings
    parser.add_argument("--experiment_remark", type=str, help="Remarks for the current experiment")
    parser.add_argument("--data_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    parser.add_argument('--recall_topk', type=int, help="Top K recall value")

    # Model and Training parameters
    parser.add_argument('--clip_model_name', type=str, default="openai/clip-vit-base-patch32", help="Name of the CLIP model")
    parser.add_argument('--learning_rate', type=float, required=True, help="Learning rate for training")
    parser.add_argument('--coef_lr', type=float, default=1.0, help="Coefficient for learning rate adjustment")
    parser.add_argument('--lr_step_size', type=int, required=True, help="Step size for learning rate decay")
    parser.add_argument('--lr_gamma', type=float, required=True, help="Gamma for learning rate decay")
    parser.add_argument('--num_epochs', type=int, default=20, help="Number of epochs to train")
    parser.add_argument('--freeze_layer_num', type=int, default=10, help="Number of layers to freeze in the CLIP model")
    parser.add_argument('--warmup_proportion', type=float, default=0.01, help="Proportion of training for warm-up")

    # Data handling
    parser.add_argument('--frame_dim', type=int, default=224, help="Dimension of video frames")
    parser.add_argument('--max_words', type=int, default=20, help="Maximum number of words in a text query")
    parser.add_argument('--max_frame_count', type=int, default=100, help="Maximum number of frames to use per video")
    parser.add_argument('--segment_second', type=int, help="Segment length in seconds")
    parser.add_argument('--fps', type=int, help="Frames per second for video processing")
    parser.add_argument('--read_video_from_tensor', action='store_true', help="Flag to read videos from tensor files")

    # Logging and evaluation
    parser.add_argument('--step_log', type=int, default=100, help="Frequency of logging training information")
    parser.add_argument("--step_eval", type=int, required=True, help="Frequency of evaluation during training")

    # System settings
    parser.add_argument('--num_workers', type=int, default=16, help="Number of worker threads for data loading")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training")
    parser.add_argument('--seed', type=int, default=2024, help="Random seed for reproducibility")

    args = parser.parse_args()
    return args



def set_seed_logger(args):
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger(args.output_dir, args.experiment_remark)
    return logger


# def init_device(logger):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     n = torch.cuda.device_count()
#     logger.info(f"Using {n} {device}!")
#     return device

