python main_video_retrieval.py \
    --train_path    data/TVR_Ranking_Segment/train_top20_segment.jsonl \
    --val_path      data/TVR_Ranking_Segment/val_segment.jsonl \
    --test_path     data/TVR_Ranking_Segment/test_segment.jsonl \
    --corpus_path   data/TVR_Ranking_Segment/segment_corpus_4seconds.jsonl \
    --video_dir     /home/share/rjliang/Dataset/TVR/frames \
    --output_dir    result/tvrr_segment_top20 \
    --data_name     query_segment \
    --step_log 100     --step_eval 10000  --recall_topk 100 500 1000 \
    --batch_size 256 --num_workers 8 \
    --learning_rate 1e-6 --lr_step_size 50 --lr_gamma 0.1 \
    --max_words 32 --max_frame_count 4 --freeze_layer_num 10 \
    --segment_second 4 --fps 3 \
    --experiment_remark  correct_contractive_loss


# qsub -I -l select=1:ngpus=2 -P gs_slab -q gpu8
# cd /home/renjie.liang/12_RVMR_IR/CLIP_RVMR ; conda activate py11 ; sh run_top20_segment.sh



# gpu8, slab_gpu8, q32, q64, q64_enri, q128, qintel_wfly, tl-gpu, gpu_a40, gpu_a100
# qsub  run_top20_segment.pbs
# watch -n 1 qstat 2147774
# qstat -Q
# qdel 2147774