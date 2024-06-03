from utils.utils import load_jsonl

# train_data = load_jsonl("data/TVR_Ranking/train_top20_segment.jsonl")
# print(train_data[0])

# corpus_data = load_jsonl("data/TVR_Ranking/segment_corpus_4seconds.jsonl")
# print(corpus_data[0])

total_number = 0

val_data = load_jsonl("/home/share/rjliang/Dataset/TVR_Ranking_v3/val.jsonl")
for i in val_data:
    total_number += len(i["relevant_moment"])
    
print(total_number) # 14382
    
val_data = load_jsonl("/home/share/rjliang/Dataset/TVR_Ranking_v3/test.jsonl")
for i in val_data:
    total_number += len(i["relevant_moment"])
print(total_number) # 80060
