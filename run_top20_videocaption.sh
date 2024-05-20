export CUDA_VISIBLE_DEVICES=0,1,2,3

python main_video_retrieval.py \
    --train_path ./data/TVR_Ranking/train_top20_video_caption.jsonl \
    --val_path /home/renjie.liang/11_TVR-Ranking/ReLoCLNet/data/TVR_Ranking_v2/val.jsonl \
    --test_path /home/renjie.liang/11_TVR-Ranking/ReLoCLNet/data/TVR_Ranking_v2/test.jsonl \
    --video_path /home/share/rjliang/Dataset/TVR/frames \
    --corpus_path /home/renjie.liang/11_TVR-Ranking/ReLoCLNet/data/TVR_Ranking_v2/video_corpus.json \
    --output_dir result/tvrr_video_videocaption_top20 \
    --experiment_remark  top20 \
    --batch_size 16  --batch_size_val 16 --lr 1e-4 \
    --step_log=200 --step_eval=1000 \
    --max_words 64 --max_frames 64 \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0  --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header meanP \
    --pretrained_clip_name ViT-B/16

    # --checkpoint_path result/tvrr_video_query/best_model.bin \
    # --optimizer_path result/tvrr_video_query/best_optimizer.bin

# qsub -I -l select=1:ngpus=4 -P gs_slab -q slab_gpu8
# cd /home/renjie.liang/12_RVMR_IR/CLIP4Clip ; conda activate py11 ; sh run_top20_videocaption.sh