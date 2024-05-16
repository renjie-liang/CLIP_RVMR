python main_task_retrieval.py --do_train --num_thread_reader=0 \
--epochs=5 --batch_size=16 --n_display=50 \
--train_csv /home/renjie.liang/datasets/MSRVTT/msrvtt_data/MSRVTT_train.9k.csv \
--val_csv /home/renjie.liang/datasets/MSRVTT/msrvtt_data/MSRVTT_JSFUSION_test.csv \
--data_path /home/renjie.liang/datasets/MSRVTT/msrvtt_data/MSRVTT_data.json \
--features_path /home/renjie.liang/datasets/MSRVTT/MSRVTT/videos/all \
--output_dir ckpts/ckpt_msrvtt_retrieval_looseType \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
--datatype msrvtt --expand_msrvtt_sentences  \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/16

# qsub -I -l select=1:ngpus=1 -P gs_slab -q gpu8
# cd /home/renjie.liang/12_RVMR_IR/CLIP4Clip ; conda activate py11 ; sh run.sh