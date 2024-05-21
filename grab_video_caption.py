import pandas as pd
from utils.utils import load_jsonl, save_jsonl

# Load the data
input_path = "/home/renjie.liang/11_TVR-Ranking/ReLoCLNet/data/TVR_Ranking_v3/train_top20.jsonl"
output_path = "data/TVR_Ranking/train_top20_video_caption.jsonl"

in_data = load_jsonl(input_path)
df = pd.DataFrame(in_data)

# Extract relevant information and flatten the relevant_moment entries
expanded_rows = []
for _, row in df.iterrows():
    query_id = row['query_id']
    query = row['query']
    for moment in row['relevant_moment']:
        moment.update({'query_id': query_id, 'query': query})
        expanded_rows.append(moment)

expanded_df = pd.DataFrame(expanded_rows)
expanded_df = expanded_df.drop_duplicates(subset=['video_name', 'query'])

video_caption_dict = {}
# Group by video_name and concatenate captions sorted by timestamp using a for loop
for video_name, group in expanded_df.groupby('video_name'):
    
    sorted_group = group.sort_values(by='timestamp')
    concatenated_caption = " ".join(group['query'])
    video_caption_dict[video_name] = concatenated_caption
    # print(sorted_group[["video_name", "query", "timestamp"]])
    # breakpoint()
    
# Convert the dictionary to a DataFrame
result_df = pd.DataFrame(list(video_caption_dict.items()), columns=['video_name', 'query'])
output_data = result_df.to_dict(orient='records')
save_jsonl(output_data, output_path)
print(f"Data successfully saved to {output_path}")
