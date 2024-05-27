import os
from torch.utils.data import Dataset
import numpy as np
import torch
import cv2
from utils.utils import load_jsonl, load_json

class BaseVideoDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.read_video_from_tensor = args.read_video_from_tensor
        self.video_dir = args.video_dir
        self.max_frame_count = args.max_frame_count
        self.frame_dim = args.frame_dim

    def _prepare_video_frames(self, video_id):
        frame_path = os.path.join(self.video_dir, video_id)
        if self.read_video_from_tensor:
            frame_path = frame_path + ".pt"
            frames, video_mask = self._extract_frames_tensor(frame_path, self.max_frame_count)
        else:
            frames = self._extract_frames(frame_path, self.max_frame_count)
            # Create a mask indicating valid frames
            video_mask = [1] * len(frames) + [0] * (self.max_frame_count - len(frames))
            video_mask = torch.tensor(video_mask, dtype=torch.long)
            while len(frames) < self.max_frame_count:
                frames.append(torch.zeros_like(frames[0]))
            frames = torch.stack(frames)
        return frames, video_mask      

    def _extract_frames_tensor(self, frame_path, num_frames):
        frames_tensor = torch.load(frame_path)
        frames = frames_tensor['frames']
        video_mask = frames_tensor['video_mask']
        assert len(frames) == num_frames
        assert len(video_mask) == num_frames
        return frames, video_mask      
        
    def _extract_frames(self, frame_path, num_frames, start_frame=None, end_frame=None):
        # Get a list of all frame files
        frame_files = sorted([f for f in os.listdir(frame_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Filter frames by start_frame and end_frame if specified
        if start_frame is not None and end_frame is not None:
            total_frames = len(frame_files)
            end_frame = min(end_frame, total_frames)
            frame_files = frame_files[int(start_frame):int(end_frame)]
        
        # Calculate step size to select num_frames frames
        step = max(1, len(frame_files) // num_frames)
        selected_frames = frame_files[::step][:num_frames]

        # Read and convert frames to tensors
        frames = []
        for frame_file in selected_frames:
            frame = cv2.imread(os.path.join(frame_path, frame_file))
            frame = cv2.resize(frame, (self.frame_dim, self.frame_dim))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frame_tensor = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)  # Convert to CxHxW format
            frames.append(frame_tensor)
        return frames

class TrainVideoDataset(BaseVideoDataset):
    def __init__(self, annotation_path, args):
        super().__init__(args)
        self.annotations = load_jsonl(annotation_path)
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        anno = self.annotations[idx]
        text = anno["query"]
        query_id = anno["query_id"]
        video_id = anno["video_name"]
        similarity = anno["similarity"]
        frames, video_mask = self._prepare_video_frames(video_id)
        return text, frames, video_mask, (query_id, video_id, similarity)

class CorpusVideoDataset(BaseVideoDataset):
    def __init__(self, corpus_path, args):
        super().__init__(args)
        corpus_data = load_json(corpus_path)
        self.corpus = list(corpus_data.keys())
        self.corpus_video_list = self.corpus
        
    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        video_id = self.corpus[idx]
        video, video_mask = self._prepare_video_frames(video_id)
        return video, video_mask


     
class EvalVideoDataset(BaseVideoDataset):
    def __init__(self, annotation_path, args):
        super().__init__(args)
        self.annotations = load_jsonl(annotation_path)
        self.ground_truth = self.generate_gt()
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        anno = self.annotations[idx]
        text = anno["query"]
        return text
    
    def generate_gt(self):
        all_gt = []
        for record in self.annotations:
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
