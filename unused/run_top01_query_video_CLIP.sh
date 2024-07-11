python main_video_retrieval.py \
    --train_path ./data/TVR_Ranking_old/train_top01.jsonl \
    --val_path   ./data/TVR_Ranking/val.jsonl \
    --test_path  ./data/TVR_Ranking/test.jsonl \
    --video_dir   /home/share/rjliang/Dataset/frames_tensor_12_224 \
    --corpus_path /home/renjie.liang/11_TVR-Ranking/ReLoCLNet/data/TVR_Ranking_v3/video_corpus.json \
    --output_dir  ./result/tvrr_video_query_top01 \
    --data_name query_video_clip \
    --experiment_remark  constractive_loss_sim_mask \
    --recall_topk 100 \
    --batch_size 64   \
    --learning_rate 1e-4 --lr_step_size 5 --lr_gamma 0.1\
    --step_log=100 --step_eval=1000000 \
    --max_words 32 --max_frame_count 12 \
    --freeze_layer_count 10  \
    --read_video_from_tensor 
    # --debug  # set batch_size=4, number_works=1


# qsub -I -l select=1:ngpus=1 -P gs_slab -q gpu8
# cd /home/renjie.liang/12_RVMR_IR/CLIP_RVMR ; conda activate py11 ; sh run_top01_query_video_CLIP.sh
