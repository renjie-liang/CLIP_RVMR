from transformers import CLIPModel
import torch
from torch import nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, similarity_matrix):
        logpt = F.log_softmax(similarity_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        contrastive_loss = nce_loss.mean()
        return contrastive_loss


class CLIPFineTuner(nn.Module):
    def __init__(self, clip_model_name):
        super(CLIPFineTuner, self).__init__()
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.loss_function = ContrastiveLoss()
        
        
    def freeze_layers(self, freeze_layer_count):
        # Get the layers of the model
        layers = list(self.clip_model.vision_model.encoder.layers) + list(self.clip_model.text_model.encoder.layers)
        
        # Freeze the first `freeze_layer_count` layers
        for layer in layers[:freeze_layer_count]:
            for param in layer.parameters():
                param.requires_grad = False
    def forward(self, text_input_ids, attention_mask, video_frames, video_mask, sim_masks):
        text_features = self.clip_model.get_text_features(input_ids=text_input_ids, attention_mask=attention_mask)
        video_features = self.get_video_features(video_frames)
        sim_matrix = self.compute_similarity_matrix(text_features, video_features, video_mask)
        sim_matrix_masked = sim_matrix * sim_masks
        # Compute the contrastive loss
        loss_text_to_video = F.cross_entropy(sim_matrix_masked, sim_masks.max(1)[1])
        loss_video_to_text = F.cross_entropy(sim_matrix_masked.t(), sim_masks.max(0)[1])
        total_loss = (loss_text_to_video + loss_video_to_text) / 2
        return total_loss
        
    def get_video_features(self, video_tensor):
        batch_size, num_frames, channels, height, width = video_tensor.shape
        video_tensor = video_tensor.view(batch_size * num_frames, channels, height, width)
        video_features = self.clip_model.get_image_features(pixel_values=video_tensor)
        video_features = video_features.view(batch_size, num_frames, video_features.size(-1))
        return video_features

    def mean_pooling(self, visual_output, video_mask):
        video_mask = video_mask.to(dtype=torch.float32).unsqueeze(-1)
        visual_output = visual_output * video_mask
        mask_sum = video_mask.sum(dim=1)
        mask_sum[mask_sum == 0] = 1  # avoid division by zero
        visual_output = visual_output.sum(dim=1) / mask_sum
        return visual_output

    def compute_similarity_matrix(self, text_features, video_features, video_mask):
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        video_features = self.mean_pooling(video_features, video_mask)
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.clip_model.logit_scale.exp().to(text_features.device)
        similarity_matrix = logit_scale * torch.matmul(text_features, video_features.t())
        return similarity_matrix


# Example usage:
# clip_model_name = "openai/clip-vit-base-patch32"
# model = CLIPFineTuner(clip_model_name)
# text_input_ids, attention_mask, video_frames, video_mask = ...  # Your data loading logic
# loss = model(text_input_ids, attention_mask, video_frames, video_mask)


# ChatGPT version
# class CrossEn(nn.Module):
#     def __init__(self):
#         super(CrossEn, self).__init__()
#         self.loss = nn.CrossEntropyLoss()

#     def forward(self, sim_matrix):
#         b = sim_matrix.size(0)
#         labels = torch.arange(b, device=sim_matrix.device)
#         loss = self.loss(sim_matrix, labels)
#         return loss