
import pandas as pd
from utils.utils import load_json, save_jsonl
import json
import os
from tqdm import tqdm

# Load video corpus data
video_corpus_path = "/home/renjie.liang/11_TVR-Ranking/ReLoCLNet/data/TVR_Ranking_v3/video_corpus.json"
segment_corpus_path = "data/TVR_Ranking/segment_corpus_1seconds.jsonl"
video_dir  = "/home/share/rjliang/Dataset/TVR/frames"

video_corpus = load_json(video_corpus_path)
segment_corpus = []
# Define segment duration
segment_second = 1  # seconds
fps = 3 

# Process each video in the video corpus
for video_name, (duration, _) in tqdm(video_corpus.items()):
    # Calculate the number of segments
    num_segments = int(duration // segment_second) + (1 if duration % segment_second > 0 else 0)
    frame_path = os.path.join(video_dir, video_name)
        
    for segment_idx in range(num_segments):
        start_frame = segment_idx * segment_second * fps
        end_frame = (segment_idx + 1) * segment_second * fps
        
        # frame_files = sorted([f for f in os.listdir(frame_path)])
        # total_frames = len(frame_files)
        # end_frame = min(end_frame, total_frames)
        # segment_frame_files = frame_files[int(start_frame):int(end_frame)]
        # step = max(1, len(segment_frame_files) // n)
        # selected_frames = segment_frame_files[::step][:n]
        
        # if len(selected_frames) == 0:
        #     print(video_name, duration, total_frames)
        #     breakpoint()

        segment = {
            "segment_id": len(segment_corpus),  # Assign a unique ID for each segment
            "video_name": video_name,
            "segment_idx": segment_idx,
            "duration": duration
        }
        segment_corpus.append(segment)

# Save the segment corpus
save_jsonl(segment_corpus, segment_corpus_path)
print(f"Segment corpus saved to {segment_corpus_path}")
