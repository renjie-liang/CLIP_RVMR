
import pandas as pd
from utils.basic_utils import load_jsonl, save_json, load_json
import json
import os
from tqdm import tqdm

# Load video corpus data
data_path = "/home/renjie.liang/11_TVR-Ranking/ReLoCLNet/data/TVR_Ranking_v2/train_top40.jsonl"
new_data_path = "/home/renjie.liang/11_TVR-Ranking/ReLoCLNet/data/TVR_Ranking_v4/train_top40.json"
data = load_jsonl(data_path)
video_corpus_path = "/home/renjie.liang/11_TVR-Ranking/ReLoCLNet/data/TVR_Ranking_v4/video_corpus.json"
video_corpus = load_json(video_corpus_path)


for item in tqdm(data):
    query = item['query']
    relevant_moment = item['relevant_moment']
    for moment_info in relevant_moment:
        video_name = moment_info["video_name"]
        new_druation = video_corpus[video_name]
        moment_info["duration"] = new_druation
        moment_info["similarity"] = round(moment_info["similarity"], 4)
        s, e = moment_info["timestamp"]
        if e > new_druation:
            moment_info["timestamp"] = [s, new_druation]
            assert s < new_druation
            print(e, new_druation)
save_json(data, new_data_path)
