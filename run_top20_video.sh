python main_video_retrieval.py \
    --train_path    /home/share/rjliang/Dataset/TVR_Ranking_v3/train_top20.jsonl \
    --val_path      /home/share/rjliang/Dataset/TVR_Ranking_v3/val.jsonl \
    --test_path     /home/share/rjliang/Dataset/TVR_Ranking_v3/test.jsonl \
    --corpus_path   /home/share/rjliang/Dataset/TVR_Ranking_v3/video_corpus.json \
    --video_dir     /home/share/rjliang/Dataset/TVR/frames \
    --output_dir    result/tvrr_video_query_top20 \
    --data_name     query_video_clip \
    --recall_topk 100 \
    --batch_size 64 --num_workers 8  \
    --learning_rate 1e-4 --lr_step_size 5 --lr_gamma 0.1 --coef_lr 1e-3 \
    --step_log=100 --step_eval=2000 \
    --max_words 32 --max_frame_count 12 \
    --experiment_remark  unique_video_text
    

# qsub -I -l select=1:ngpus=1 -P gs_slab -q gpu8
# cd /home/renjie.liang/12_RVMR_IR/CLIP_RVMR ; conda activate py11 ; sh run_top20_video.sh

# gpu8, slab_gpu8, q32, q64, q64_enri, q128, qintel_wfly, tl-gpu, gpu_a40, gpu_a100