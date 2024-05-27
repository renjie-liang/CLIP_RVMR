
# qsub -I -l select=1:ngpus=1 -P gs_slab -q slab_gpu8
# cd /home/renjie.liang/12_RVMR_IR/CLIP_RVMR; conda activate py11 ; sh run_top20_query_video_CLIP.sh

# deepspeed train.py --demo_deepspeed configs/demo_ds_config.json --freeze_layer_count 1 --recall_topk 10
# deepspeed demo_deepspeed.py --deepspeed_config configs/demo_ds_config.json