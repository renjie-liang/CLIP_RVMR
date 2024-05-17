
import os
from torch.utils.data import Dataset
import numpy as np
import json
import math
from utils.utils import load_jsonl, load_json
import torch
import cv2

    

class TVRR_Base_DataLoader(Dataset):
    def __init__(self):
        pass
    
    def _prepare_text(self, sentence):
        # Tokenize the sentence and add special tokens
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + self.tokenizer.tokenize(sentence) + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
        
        # Truncate if necessary
        if len(words) > self.max_words:
            words = words[:self.max_words-1] + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
        
        # Convert tokens to IDs and pad if necessary
        input_ids = self.tokenizer.convert_tokens_to_ids(words)
        padding_length = self.max_words - len(input_ids)
        input_ids.extend([0] * padding_length)
        
        # Create input mask
        input_mask = [1] * len(words) + [0] * padding_length
        
        # Ensure both input_ids and input_mask are of length self.max_words
        input_ids = input_ids[:self.max_words]
        input_mask = input_mask[:self.max_words]
        
        # Convert to torch tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        
        return input_ids, input_mask
    
    def _get_frames(self, frame_path, n, HW):
        # Get a list of all frame files
        frame_files = sorted([f for f in os.listdir(frame_path)])
        
        # Select frames with the same step
        total_frames = len(frame_files)
        step = max(1, total_frames // n)
        selected_frames = frame_files[::step][:n]
        
        # Load frames and resize to 224x224
        frames = []
        for frame_file in selected_frames:
            frame = cv2.imread(os.path.join(frame_path, frame_file))
            if not (frame.shape[1] == HW and frame.shape[2] == HW):
                frame = cv2.resize(frame, (HW, HW))
            frames.append(frame)
        return frames
       
    def _prepare_video(self, video_id):
        frame_path = os.path.join(self.video_path, video_id)
        image_size = self.image_resolution
        frames =  self._get_frames(frame_path, self.max_frames, image_size)
        
        total_frames = len(frames)
        # Pad if there are fewer frames than self.max_frames
        if len(frames) < self.max_frames:
            padding_frames = [np.zeros((image_size, image_size, 3), dtype=np.uint8)] * (self.max_frames - len(frames))
            frames.extend(padding_frames)    
        
        
        # Convert frames to tensor with shape [1 x self.max_frames x 1 x 3 x H x W]
        frames = np.stack(frames, axis=0)  # [self.max_frames x 224 x 224 x 3]
        frames = frames.transpose(0, 3, 1, 2)  # [self.max_frames x 3 x 224 x 224]
        frames = np.expand_dims(frames, axis=0)  # [1 x self.max_frames x 3 x 224 x 224]
        frames = np.expand_dims(frames, axis=2)  # [1 x self.max_frames x 1 x 3 x 224 x 224]
        frames = torch.tensor(frames, dtype=torch.float)
        
        # Generate video_mask: 1 for real frames, 0 for padding
        video_mask = [1] * min(total_frames, self.max_frames) + [0] * (self.max_frames - min(total_frames, self.max_frames))
        video_mask = torch.tensor(video_mask, dtype=torch.long)
        video_mask = torch.unsqueeze(video_mask, axis=0)  # [1 x self.max_frames]
        
        assert frames.shape == (1, self.max_frames, 1, 3, 224, 224)
        assert video_mask.shape == (1, self.max_frames)
        
        return frames, video_mask




class TVRR_DataLoader(TVRR_Base_DataLoader):
    def __init__(self, annotation_path, video_path, tokenizer, dataset_type,
                max_words=30, feature_framerate=1.0, max_frames=100,
                image_resolution=224, frame_order=0, slice_framepos=0,
    ):
        self.annotation = load_jsonl(annotation_path)
        self.video_path = video_path
        self.dataset_type = dataset_type
        
        # self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.image_resolution = image_resolution
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        # self.frame_order = frame_order
        # assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        # self.slice_framepos = slice_framepos
        # assert self.slice_framepos in [0, 1, 2]

        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        anno = self.annotation[idx]
        text = anno["query"]
        video_id = anno["video_name"]
        simi = anno["similarity"]

        text_id, text_mask = self._prepare_text(text)
        video, video_mask = self._prepare_video(video_id)
        return text_id, text_mask, video, video_mask





class TVRR_Corpus_DataLoader(TVRR_Base_DataLoader):
    def __init__(self, corpus_path, video_path,
                feature_framerate=1.0, max_frames=100,
                image_resolution=224, frame_order=0, slice_framepos=0,
    ):
        self.corpus_map = load_json(corpus_path)
        self.corpus = list(self.corpus_map.keys())
        self.video_path = video_path
        self.max_frames = max_frames
        self.image_resolution = image_resolution
        
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        video_id = self.corpus["video_name"]
        video, video_mask = self._prepare_video(video_id)
        return video, video_mask
    