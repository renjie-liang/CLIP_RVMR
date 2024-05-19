export CUDA_VISIBLE_DEVICES=0,1,2,3

python main_video_retrieval.py \
    --train_path /home/renjie.liang/datasets/TVR_Ranking/train_top01.jsonl \
    --val_path /home/renjie.liang/11_TVR-Ranking/ReLoCLNet/data/TVR_Ranking_v2/val.jsonl \
    --test_path /home/renjie.liang/11_TVR-Ranking/ReLoCLNet/data/TVR_Ranking_v2/test.jsonl \
    --video_path /home/share/rjliang/Dataset/TVR/frames \
    --corpus_path /home/renjie.liang/11_TVR-Ranking/ReLoCLNet/data/TVR_Ranking_v2/video_corpus.json \
    --output_dir ckpts/tvrr_tmp \
    --experiment_remark debug \
    --batch_size 24  --batch_size_val 24 --lr 1e-4 \
    --step_log=200 --step_eval=1000 \
    --max_words 32 --max_frames 12 \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0  --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header meanP \
    --pretrained_clip_name ViT-B/16

# qsub -I -l select=1:ngpus=4 -P gs_slab -q slab_gpu8
# cd /home/renjie.liang/12_RVMR_IR/CLIP4Clip ; conda activate py11 ; sh run.sh