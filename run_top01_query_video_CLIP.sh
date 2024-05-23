python main_video_retrieval.py \
    --train_path /home/renjie.liang/datasets/TVR_Ranking/train_top01.jsonl \
    --val_path /home/renjie.liang/11_TVR-Ranking/ReLoCLNet/data/TVR_Ranking_v3/val.jsonl \
    --test_path /home/renjie.liang/11_TVR-Ranking/ReLoCLNet/data/TVR_Ranking_v3/test.jsonl \
    --video_dir /home/share/rjliang/Dataset/TVR/frames \
    --corpus_path /home/renjie.liang/11_TVR-Ranking/ReLoCLNet/data/TVR_Ranking_v3/video_corpus.json \
    --output_dir result/tvrr_video_query \
    --data_name query_video_clip \
    --experiment_remark  freeze_10 \
    --recall_topk 100 \
    --batch_size 64  --batch_size_val 64 \
    --learning_rate 1e-4 --lr_step_size 5 --lr_gamma 0.1\
    --step_log=100 --step_eval=1000000 \
    --max_words 32 --max_frame_count 12 \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0  --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header meanP \
    --pretrained_clip_name ViT-B/16


# qsub -I -l select=1:ngpus=2 -P gs_slab -q gpu8
# cd /home/renjie.liang/12_RVMR_IR/CLIP4Clip ; conda activate py11 ; sh run_top01_query_video_CLIP.sh