import pandas as pd
import math
from utils.utils import load_jsonl, save_jsonl

input_path = "/home/renjie.liang/11_TVR-Ranking/ReLoCLNet/data/TVR_Ranking_v3/test.jsonl"
output_path = "data/TVR_Ranking_Segment/test_segment.jsonl"

def split_into_segments(relevant_moments):
    segments = []
    for moment in relevant_moments:
        video_name = moment['video_name']
        start_time, end_time = moment['timestamp']
        duration = moment['duration']
        relevance = moment['relevance']
        
        similarity = round(moment['similarity'], 4)

        start_segment = math.floor(start_time / 4)
        end_segment = math.floor(end_time / 4)

        for segment_idx in range(start_segment, end_segment + 1):
            segment_start_time = segment_idx * 4
            segment_end_time = (segment_idx + 1) * 4
            if start_time < segment_end_time and end_time > segment_start_time:
                segments.append({
                    "video_name": video_name,
                    "duration": duration,
                    "segment_idx": segment_idx,
                    "moment_timestamp": [start_time, end_time],
                    "similarity": similarity,
                    "relevance": relevance,
                })
    return segments

in_data = load_jsonl(input_path)
processed_data = []

for item in in_data:
    query = item['query']
    query_id = item['query_id']
    
    relevant_segments = split_into_segments(item['relevant_moment'])
    processed_data.append({"query": query, "query_id": query_id, "relevant_segment": relevant_segments})

save_jsonl(processed_data, output_path)
print(f"Processed data saved to {output_path}")
