from utils.basic_utils import load_jsonl, save_jsonl
import pandas as pd

# Define input and output paths
in_path = "data/TVR_Ranking/val.jsonl"
out_path = "data/TVR_Ranking_v2/val.jsonl"

# Load the data
in_data = load_jsonl(in_path)

# Create a DataFrame
df = pd.DataFrame(in_data)

# Function to reorganize the data
def reorganize_data(df):
    result = []

    grouped = df.groupby(['query_id', 'query'])
    for (query_id, query), group in grouped:
        relevant_moments = []
        for _, row in group.iterrows():
            relevant_moments.append({
                "pair_id": row["pair_id"],
                "video_name": row['video_name'],
                "timestamp": row['timestamp'],
                "duration": row['duration'],
                "caption": row['caption'],
                "similarity": row['similarity'],
                "relevance": row['relevance']
            })
   
        tmp = {
            "query_id": int(query_id),
            "query": query,
            "relevant_moment": relevant_moments
        }
        result.append(tmp)
    
    return result

# Reorganize the data
reorganized_data = reorganize_data(df)

# Save the reorganized data to a new JSONL file
save_jsonl(reorganized_data, out_path)
print(f"Reorganized data saved to {out_path}")
