# CLIP_RVMR

Build a dense retrieval framework on Ranking Video Moment Retrieval (RVMR) based on CLIP.

### Quick Run

```
  qsub -I -l select=1:ngpus=1 -P gs_slab -q slab_gpu8
  cd /home/renjie.liang/12_RVMR_IR/CLIP_RVMR ; conda activate py11 ;
  sh run_top20_video.sh
```

### Performance
top20 training set:

VAL  Recall@100: 0.2748
TEST Recall@100: 0.2712