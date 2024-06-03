from utils.utils import load_jsonl

# train_data = load_jsonl("data/TVR_Ranking/train_top20_segment.jsonl")
# print(train_data[0])

# corpus_data = load_jsonl("data/TVR_Ranking/segment_corpus_4seconds.jsonl")
# print(corpus_data[0])


val_data = load_jsonl("data/TVR_Ranking_Segment/val_segment.jsonl")
max_len = 0
total_len = 0
for i in val_data:
    l = len(i["relevant_segment"])
    total_len += l
    max_len = max(max_len, l)

avg_len = total_len/len(val_data)
print(max_len) # 476
print(avg_len) # 90