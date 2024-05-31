
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
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
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
        frame_path = os.path.join(self.video_dir, video_id)
        image_size = self.image_resolution
        frames =  self._get_frames(frame_path, self.max_frame_count, image_size)
        
        total_frames = len(frames)
        # Pad if there are fewer frames than self.max_frame_count
        if len(frames) < self.max_frame_count:
            padding_frames = [np.zeros((image_size, image_size, 3), dtype=np.uint8)] * (self.max_frame_count - len(frames))
            frames.extend(padding_frames)    
        
        
        # Convert frames to tensor with shape [1 x self.max_frame_count x 1 x 3 x H x W]
        frames = np.stack(frames, axis=0)  # [self.max_frame_count x 224 x 224 x 3]
        frames = frames.transpose(0, 3, 1, 2)  # [self.max_frame_count x 3 x 224 x 224]
        frames = torch.tensor(frames, dtype=torch.float)
        
        # Generate video_mask: 1 for real frames, 0 for padding
        video_mask = [1] * min(total_frames, self.max_frame_count) + [0] * (self.max_frame_count - min(total_frames, self.max_frame_count))
        video_mask = torch.tensor(video_mask, dtype=torch.long)
        
        assert frames.shape == (self.max_frame_count, 3, 224, 224)
        assert len(video_mask) == self.max_frame_count
        return frames, video_mask




class TVRR_DataLoader_train(TVRR_Base_DataLoader):
    def __init__(self, annotation_path, video_dir, tokenizer,
                max_words=30, feature_framerate=1.0, max_frame_count=100,
                image_resolution=224, frame_order=0, slice_framepos=0,
    ):
        super().__init__()
        
        self.annotation = load_jsonl(annotation_path)
        self.video_dir = video_dir
        
        self.max_words = max_words
        self.max_frame_count = max_frame_count
        self.image_resolution = image_resolution
        self.tokenizer = tokenizer
        
        
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        
        anno = self.annotation[idx]
        text = anno["query"]
        video_id = anno["video_name"]
        # simi = anno["similarity"]
        text_id, text_mask = self._prepare_text(text)
        video, video_mask = self._prepare_video(video_id)
        return text_id, text_mask, video, video_mask

        
        
class TVRR_DataLoader_eval(TVRR_Base_DataLoader):
    def __init__(self, annotation_path, corpus_path, video_dir, tokenizer,
                max_words=30, feature_framerate=1.0, max_frame_count=100,
                image_resolution=224, frame_order=0, slice_framepos=0,
    ):
        super().__init__()
        self.annotation = load_jsonl(annotation_path)
        self.video_dir = video_dir
        self.max_words = max_words
        self.max_frame_count = max_frame_count
        self.image_resolution = image_resolution
        self.tokenizer = tokenizer
        self.ground_truth = self.generate_gt()
        
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        anno = self.annotation[idx]
        text = anno["query"]
        text_id, text_mask = self._prepare_text(text)
        return text_id, text_mask
        
    def generate_gt(self):
        all_gt = []
        for record in self.annotation:
            one_text_gt = []
            for i in record["relevant_moment"]:
                video_name = i["video_name"]
                relevance = i["relevance"]
                if relevance >= 1:
                    one_text_gt.append(video_name)
            if len(one_text_gt) == 0:
                video_name = record["relevant_moment"][0]["video_name"]
                one_text_gt.append(video_name)
            all_gt.append(one_text_gt)
        return all_gt



class TVRR_DataLoader_corpus(TVRR_Base_DataLoader):
    def __init__(self, corpus_path, video_dir,
                feature_framerate=1.0, max_frame_count=100,
                image_resolution=224, frame_order=0, slice_framepos=0,
    ):
        self.corpus_map = load_json(corpus_path)
        self.corpus = list(self.corpus_map.keys())
        self.video_dir = video_dir
        self.max_frame_count = max_frame_count
        self.image_resolution = image_resolution
        
    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        video_id = self.corpus[idx]
        video, video_mask = self._prepare_video(video_id)
        return video, video_mask




