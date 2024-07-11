
import os
from torch.utils.data import Dataset
import numpy as np
import json
import math
from utils.utils import load_jsonl, load_json
import torch
import cv2

class TVRR_Base_DataLoader_segment(Dataset):
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
    
    def _prepare_segment(self, video_name, segment_idx, duration, segment_second, fps):
        frame_path = os.path.join(self.video_dir, video_name)
        # fist check the fps, duration, and the number frame in frame_path
        # number_frames = len(os.listdir(frame_path))
        # number_expectation = math.floor(duration * self.fps)
        # assert  abs(number_frames - number_expectation) < 3, f"{video_name} {duration} {number_frames}"

        image_size = self.image_resolution
        frames =  self._get_segment_frames(frame_path, segment_idx, image_size, segment_second, fps)
        
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

    def _get_segment_frames(self, frame_path, segment_idx, HW, segment_second, fps):
        n = self.max_frame_count
        start_frame = segment_idx * segment_second * fps
        end_frame = (segment_idx + 1) * segment_second * fps
        
        frame_files = sorted([f for f in os.listdir(frame_path)])
        total_frames = len(frame_files)
        
        # Ensure end_frame does not exceed total_frames
        end_frame = min(end_frame, total_frames)
        
        # Select frames within the start_frame and end_frame
        segment_frame_files = frame_files[int(start_frame):int(end_frame)]
        
        # Sample n frames from the segment
        step = max(1, len(segment_frame_files) // n)
        selected_frames = segment_frame_files[::step][:n]
        
        frames = []
        for frame_file in selected_frames:
            frame = cv2.imread(os.path.join(frame_path, frame_file))
            if frame.shape[0] != HW or frame.shape[1] != HW:
                frame = cv2.resize(frame, (HW, HW))
            frames.append(frame)
        return frames
    

class TVRR_DataLoader_corpus_segment(TVRR_Base_DataLoader_segment):
    def __init__(self, corpus_path, video_dir,
                feature_framerate=1.0, max_frame_count=100,
                image_resolution=224, frame_order=0, slice_framepos=0,
    ):
        
        self.corpus = load_jsonl(corpus_path)
        self.corpus_video_list = [i["video_name"] for i in self.corpus]
        self.video_dir = video_dir
        self.max_frame_count = max_frame_count
        self.image_resolution = image_resolution
        self.segment_second = 4 
        self.fps = 3
    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        anno = self.corpus[idx]
        video_name = anno["video_name"]
        segment_idx = anno["segment_idx"]
        duration = anno["duration"]
        video, video_mask = self._prepare_segment(video_name, segment_idx, duration, self.segment_second, self.fps)
        return video, video_mask


class TVRR_DataLoader_train_segment(TVRR_Base_DataLoader_segment):
    def __init__(self, annotation_path, video_dir, tokenizer,
                max_words=30, feature_framerate=1.0, max_frame_count=100,
                image_resolution=224, frame_order=0, slice_framepos=0,
    ):
        super().__init__()
        
        self.annotation = load_jsonl(annotation_path)
        self.annotation = self.expand_annotation(self.annotation)
        self.video_dir = video_dir
        self.max_words = max_words
        self.max_frame_count = max_frame_count
        self.image_resolution = image_resolution
        self.tokenizer = tokenizer
        self.fps = 3
        self.segment_second = 4
        
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        anno = self.annotation[idx]
        text = anno["query"]
        text_id, text_mask = self._prepare_text(text)

        video_name = anno["video_name"]
        segment_idx = anno["segment_idx"]
        duration = anno["duration"]
        video, video_mask = self._prepare_segment(video_name, segment_idx, duration, self.segment_second, self.fps)
        return text_id, text_mask, video, video_mask

    def expand_annotation(self, annotation):
        new_annotation = []
        for i in annotation:
            query = i["query"]
            relevant_segment = i["relevant_segment"]
            for segment in relevant_segment:
                segment.update({'query': query})
                new_annotation.append(segment)
        return new_annotation