python train.py \
    --train_path    /home/share/rjliang/TVR_Ranking_Segment/train_top20_segment.json \
    --val_path      /home/share/rjliang/TVR_Ranking_Segment/val_segment.json \
    --test_path     /home/share/rjliang/TVR_Ranking_Segment/test_segment.json \
    --corpus_path   /home/share/rjliang/TVR_Ranking_Segment/segment_corpus_4seconds.json \
    --video_dir     /home/share/rjliang/Dataset/TVR/frames \
    --output_dir    result/tvrr_segment_top20 \
    --data_name     query_segment \
    --step_log 100     --step_eval 10000  --recall_topk 100 500 1000 \
    --batch_size 256 --num_workers 8 \
    --learning_rate 1e-4 --lr_step_size 1 --lr_gamma 0.1 \
    --max_words 32 --max_frame_count 4 --freeze_layer_num 10 \
    --segment_second 4 --fps 3 \
    --seed 2024 \
    --experiment_remark  CosineAnnealingLR

# qsub -I -l select=1:ngpus=1 -P gs_slab -q slab_gpu8
# cd /home/renjie.liang/12_RVMR_IR/CLIP_RVMR ; conda activate py11 ; sh run_top20_segment.sh



# gpu8, slab_gpu8, q32, q64, q64_enri, q128, qintel_wfly, tl-gpu, gpu_a40, gpu_a100
# qsub  run_top20_segment.pbs
# watch -n 1 qstat 2147774
# qstat -Q
# qdel 2147774