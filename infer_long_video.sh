python infer_video.py \
    --val_path      data/long_video_dataset/s01e02/test.json \
    --corpus_path   data/long_video_dataset/s01e02/video_corpus.json \
    --video_dir     /home/share/rjliang/Dataset/TVR/frames \
    --output_dir    result/long_video \
    --data_name     query_video \
    --recall_topk 1 3 10 \
    --batch_size 4 --num_workers 2 \
    --learning_rate 1e-4 --lr_step_size 1 --lr_gamma 0.1 \
    --max_words 32 --max_frame_count 12 --freeze_layer_num 10 \
    --seed 2024 \
    --checkpoint_path   result/tvrr_video_query_top20/best_model.bin \
    --optimizer_path    result/tvrr_video_query_top20/best_optimizer.bin \
    --experiment_remark  infer_long_video

# qsub -I -l select=1:ngpus=1 -P gs_slab -q slab_gpu8
# cd /home/renjie.liang/12_RVMR_IR/CLIP_RVMR ; conda activate py11 ; sh infer_long_video.sh
