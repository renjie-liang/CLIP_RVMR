# if batch size = 64
# when using single GPU, the dimension is correct as follows
class CLIPFineTuner(nn.Module):
    def forward(text_feature, video_feature, sim_masks):
        # text_feature:  (64 x d)
        # video_feature:  (63 x d)
        # sim_masks:  (64 x 63)
        sim_matrix = text_feature * video_feature.T
        # sim_matrix: (64 x 63)
        sim_matrix_masked = sim_matrix * sim_masks 
        # (64 x 63) * (64 x 63)

# however, when using 2 GPUs, the dimension can not match.
class CLIPFineTuner(nn.Module):
    def forward(text_feature, video_feature, sim_masks):
        # text_feature:  (32 x d), (32 x d)
        # video_feature: (32 x d), (31 x d)
        # sim_masks:  (32 x 63), (32 x 63)
        sim_matrix = text_feature * video_feature.T
        # sim_matrix: (32 x 32), (32 x 31), 
        sim_matrix_masked = sim_matrix * sim_masks 
        # (32 x 63) * (32 x 32),  (32 x 63) * (32 x 31)
        # the dimension mismached! 