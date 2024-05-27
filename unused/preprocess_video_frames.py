import os
import torch
import cv2
from tqdm import tqdm 
from torch.utils.data import Dataset

class BaseVideoDataset(Dataset):
    def __init__(self, video_dir, max_frame_count, frame_dim=224):
        super().__init__()
        self.video_dir = video_dir
        self.max_frame_count = max_frame_count
        self.frame_dim = frame_dim

    def _prepare_video_frames(self, video_id):
        frame_path = os.path.join(self.video_dir, video_id)
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
            frame_tensor = torch.tensor(frame, dtype=torch.float32)
            frames.append(frame_tensor)
        return frames

def preprocess_and_save_frames(video_dir, tensor_dir, max_frame_count, frame_dim=224):
    if not os.path.exists(tensor_dir):
        os.makedirs(tensor_dir)
    
    video_ids = [d for d in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, d))]
    base_dataset = BaseVideoDataset(video_dir, max_frame_count, frame_dim)
    
    for video_id in tqdm(video_ids):

        tensor_path = os.path.join(tensor_dir, video_id + '.pt')
        tmp = torch.load(tensor_path)
        # if isinstance(tmp, torch.Tensor):
            # pass
        if  isinstance(tmp, dict):
            continue
        frames, video_mask = base_dataset._prepare_video_frames(video_id)
        torch.save({'frames': frames, 'video_mask': video_mask}, tensor_path)

# Example usage
video_dir = '/home/share/rjliang/Dataset/TVR/frames'
tensor_dir = '/home/share/rjliang/Dataset/frames_tensor_12_224'
max_frame_count = 12
frame_dim = 224

preprocess_and_save_frames(video_dir, tensor_dir, max_frame_count, frame_dim)
