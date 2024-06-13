import os
from torch.utils.data import Dataset
import numpy as np
import torch
import cv2
from utils.utils import load_json, load_json

class BaseDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.read_video_from_tensor = args.read_video_from_tensor
        self.video_dir = args.video_dir
        self.max_frame_count = args.max_frame_count
        self.frame_dim = args.frame_dim


    def _prepare_video(self, video_id):
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

    def _extract_frames(self, frame_path, num_frames, start_frame=None, end_frame=None):
        # Get a list of all frame files
        frame_files = sorted([f for f in os.listdir(frame_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Filter frames by start_frame and end_frame if specified
        if start_frame is not None and end_frame is not None:
            end_frame = min(end_frame, len(frame_files))
            frame_files_seg = frame_files[int(start_frame):int(end_frame)]
        # Calculate step size to select num_frames frames
        step = max(1, len(frame_files_seg) // num_frames)
        selected_frames = frame_files_seg[::step][:num_frames]
        # keep at least one frame the in input
        if len(selected_frames) == 0:
            selected_frames = frame_files[:1]
        # Read and convert frames to tensors
        
        frames = []
        for frame_file in selected_frames:
            frame = cv2.imread(os.path.join(frame_path, frame_file))
            frame = cv2.resize(frame, (self.frame_dim, self.frame_dim))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frame_tensor = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)  # Convert to CxHxW format, 3, 244, 244
            frames.append(frame_tensor)
        return frames
    
    def _extract_frames_tensor(self, frame_path, num_frames):
        frames_tensor = torch.load(frame_path)
        frames = frames_tensor['frames']
        video_mask = frames_tensor['video_mask']
        frames = frames.permute(0, 3, 1, 2)  # Convert to CxHxW format
        assert len(frames) == num_frames
        assert len(video_mask) == num_frames
        return frames, video_mask     


    def _prepare_segment(self, video_name, segment_idx, segment_second, fps):
        frame_path = os.path.join(self.video_dir, video_name)
        start_frame = segment_idx * segment_second * fps
        end_frame = (segment_idx + 1) * segment_second * fps
        frames = self._extract_frames(frame_path, self.max_frame_count, start_frame, end_frame)
        
        

            # Create a mask indicating valid frames
        video_mask = [1] * len(frames) + [0] * (self.max_frame_count - len(frames))
        video_mask = torch.tensor(video_mask, dtype=torch.long)
        while len(frames) < self.max_frame_count:
            frames.append(torch.zeros([3, self.frame_dim, self.frame_dim]))
        frames = torch.stack(frames)
        return frames, video_mask
    
    
class TrainVideoDataset(BaseDataset):
    def __init__(self, annotation_path, args):
        super().__init__(args)
        self.annotations = load_json(annotation_path)
        self.annotations = self.expand_annotations(self.annotations)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        anno = self.annotations[idx]
        query = anno["query"]
        query_id = anno["query_id"]
        video_name = anno["video_name"]
        frames, frames_mask = self._prepare_video(video_name)
        return query, frames, frames_mask, (query_id, video_name, 1)

    def expand_annotations(self, annotations):
        new_annotations = []
        for i in annotations:
            query = i["query"]
            query_id = i["query_id"]
            for moment in  i["relevant_moment"]:
                moment.update({'query': query, 'query_id': query_id})
                new_annotations.append(moment)
        return new_annotations
    
class CorpusVideoDataset(BaseDataset):
    def __init__(self, corpus_path, args):
        super().__init__(args)
        corpus_data = load_json(corpus_path)
        self.corpus = list(corpus_data.keys())
        self.corpus_video_list = self.corpus
        
    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        video_id = self.corpus[idx]
        video, video_mask = self._prepare_video(video_id)
        return video, video_mask


     
class EvalVideoDataset(BaseDataset):
    def __init__(self, annotation_path, args):
        super().__init__(args)
        self.annotations = load_json(annotation_path)
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



class TrainSegmentDataset(BaseDataset):
    def __init__(self, annotation_path, args):
        super().__init__(args)
        self.annotations = load_json(annotation_path)
        self.annotations = self.expand_annotations(self.annotations)

        self.segment_second = args.segment_second
        self.fps = args.fps
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        anno = self.annotations[idx]
        query = anno["query"]
        query_id = anno["query_id"]
        video_name = anno["video_name"]
        segment_idx = anno["segment_idx"]
        segment_name = video_name + "_" + str(segment_idx)
        frames, frames_mask = self._prepare_segment(video_name, segment_idx, self.segment_second, self.fps)
        return query, frames, frames_mask, (query_id, segment_name, 1)
    
    
    def expand_annotations(self, annotations):
        new_annotations = []
        for i in annotations:
            query = i["query"]
            query_id = i["query_id"]
            relevant_segment = i["relevant_segment"]
            for segment in relevant_segment:
                segment.update({'query': query, 'query_id': query_id})
                new_annotations.append(segment)
        return new_annotations
    

class CorpusSegmentDataset(BaseDataset):
    def __init__(self, corpus_path, args):
        super().__init__(args)
        self.corpus = load_json(corpus_path)
        self.corpus_segment_list = [i["video_name"] + "_" + str(i["segment_idx"]) for i in self.corpus]
        self.segment_second = args.segment_second
        self.fps = args.fps
        
    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        anno = self.corpus[idx]
        video_name = anno["video_name"]
        segment_idx = anno["segment_idx"]
        frames, frames_mask = self._prepare_segment(video_name, segment_idx, self.segment_second, self.fps)
        return frames, frames_mask



     
class EvalSegmentDataset(BaseDataset):
    def __init__(self, annotation_path, args):
        super().__init__(args)
        self.annotations = load_json(annotation_path)
        self.segment_retrieval_gt = self.get_segment_retrieval_gt()
        self.relevant_moment_gt = self.get_relevant_moment_gt()
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        anno = self.annotations[idx]
        text = anno["query"]
        return text
    
    def get_segment_retrieval_gt(self):
        gt_all = []
        for record in self.annotations:
            gt_per_query = []
            for i in record["relevant_segment"]:
                video_name = i["video_name"]
                segment_idx = i["segment_idx"]
                relevance = i["relevance"]
                segment_name = video_name + "_" + str(segment_idx)
                if relevance >= 1:
                    gt_per_query.append(segment_name)
                    
            if len(gt_per_query) == 0:
                i = record["relevant_segment"][0]
                video_name = i["video_name"]
                segment_idx = i["segment_idx"]
                segment_name = video_name + "_" + str(segment_idx)
                gt_per_query.append(segment_name)
            gt_all.append(gt_per_query)
        return gt_all


    def get_relevant_moment_gt(self):
        gt_all = []
        for record in self.annotations:
            gt_all.append({
                "query_id": record["query_id"],
                "relevant_moment": record["relevant_moment"]})
        return gt_all