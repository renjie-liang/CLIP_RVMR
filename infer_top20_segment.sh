python infer.py \
    --train_path    /home/share/rjliang/TVR_Ranking_Segment/train_top20_segment.json \
    --val_path      /home/share/rjliang/TVR_Ranking_Segment/val_segment.json \
    --test_path     /home/share/rjliang/TVR_Ranking_Segment/test_segment.json \
    --corpus_path   /home/share/rjliang/TVR_Ranking_Segment/segment_corpus_4seconds.json \
    --video_dir     /home/share/rjliang/Dataset/TVR/frames \
    --output_dir    result/tvrr_segment_top20 \
    --data_name     query_segment \
    --step_log 100     --step_eval 10000  --recall_topk 10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000 \
    --batch_size 128 --num_workers 4 \
    --learning_rate 1e-4 --lr_step_size 1 --lr_gamma 0.1 \
    --max_words 32 --max_frame_count 4 --freeze_layer_num 10 \
    --segment_second 4 --fps 3 \
    --seed 2024 \
    --checkpoint_path   result/tvrr_segment_top20/best_model.bin \
    --optimizer_path    result/tvrr_segment_top20/best_optimizer.bin \
    --experiment_remark  infer

# qsub -I -l select=1:ngpus=1 -P gs_slab -q slab_gpu8
# cd /home/renjie.liang/12_RVMR_IR/CLIP_RVMR ; conda activate py11 ; sh infer_top20_segment.sh