import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_tvrr_retrieval import TVRR_DataLoader_train, TVRR_DataLoader_eval, TVRR_DataLoader_corpus
from dataloaders.dataloader_tvrr_segment import TVRR_DataLoader_train_segment, TVRR_DataLoader_corpus_segment
from dataloaders.dataloader_tvrr_CLIP import TrainVideoDataset, CorpusVideoDataset, EvalVideoDataset
def collate_fn(processor, batch, task='train'):
    if task == 'train':
        texts, frames, video_masks, text_id_video_id_simis = zip(*batch)
        unique_text_ids = []
        unique_texts = []
        for text, tmp in zip(texts, text_id_video_id_simis):
            text_id = tmp[0]
            if text_id not in unique_text_ids:
                unique_texts.append(text)
                unique_text_ids.append(text_id)
        unique_text_inputs = processor(text=unique_texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
        
        # Extract unique video IDs
        unique_video_ids = []
        unique_video_frames = []
        unique_video_masks = []
        for tmp, frame_list, video_mask in zip(text_id_video_id_simis, frames, video_masks):
            video_id = tmp[1]
            if video_id not in unique_video_ids:
                unique_video_ids.append(video_id)
                unique_video_frames.append(frame_list)
                unique_video_masks.append(video_mask)
        unique_video_inputs = processor(images=[frame for frame_list in unique_video_frames for frame in frame_list], return_tensors="pt", padding=True)
        unique_video_inputs['pixel_values'] = unique_video_inputs['pixel_values'].view(len(unique_video_ids), *frames[0].shape)
        unique_video_inputs['pixel_values'] = unique_video_inputs['pixel_values'].permute(0, 1, 4, 2, 3)
        # Generate video masks (this part might need further adjustment based on actual implementation)
        video_masks = torch.stack(unique_video_masks)

        # Generate similarity mask
        text_id_to_index = {text_id: idx for idx, text_id in enumerate(unique_text_ids)}
        video_id_to_index = {video_id: idx for idx, video_id in enumerate(unique_video_ids)}
        
        sim_masks = torch.zeros((len(unique_texts), len(unique_video_ids)), dtype=torch.float32)
        
        for query_id, video_id, similarity in text_id_video_id_simis:
            text_index = text_id_to_index[query_id]
            video_index = video_id_to_index[video_id]
            sim_masks[text_index, video_index] = 1
        return unique_text_inputs['input_ids'], unique_text_inputs['attention_mask'], unique_video_inputs['pixel_values'], video_masks, sim_masks

    elif task == 'corpus':
        frames, video_masks = zip(*batch)
        video_inputs = processor(images=[frame for video in frames for frame in video], return_tensors="pt", padding=True)
        video_inputs['pixel_values'] = video_inputs['pixel_values'].view(len(frames), *frames[0].shape)
        video_inputs['pixel_values'] = video_inputs['pixel_values'].permute(0, 1, 4, 2, 3)
        video_masks = torch.stack(video_masks)
        return video_inputs['pixel_values'], video_masks
    elif task == 'eval':
        texts = batch
        text_inputs = processor(text=list(texts), return_tensors="pt", padding=True, truncation=True, max_length=77)
        return text_inputs['input_ids'], text_inputs['attention_mask']

def prepare_dataloader_video_CLIP(args, processor):
    
    train_dataset = TrainVideoDataset(annotation_path=args.train_path, args=args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True, collate_fn=lambda batch: collate_fn(processor, batch, task='train'))

    corpus_dataset = CorpusVideoDataset(corpus_path=args.corpus_path, args=args)
    corpus_dataloader = DataLoader(corpus_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False, collate_fn=lambda batch: collate_fn(processor, batch, task='corpus'))
    corpus_video_list = corpus_dataset.corpus
    
    val_dataset = EvalVideoDataset(annotation_path=args.val_path, args=args)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False, collate_fn=lambda batch: collate_fn(processor, batch, task='eval'))
    val_ground_truth = val_dataset.ground_truth
    
    test_dataset = EvalVideoDataset(annotation_path=args.test_path, args=args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False, collate_fn=lambda batch: collate_fn(processor, batch, task='eval'))
    test_ground_truth = test_dataset.ground_truth
    
    return train_dataloader, corpus_dataloader, corpus_video_list, val_dataloader, val_ground_truth, test_dataloader, test_ground_truth

def prepare_dataloader_video(args, tokenizer):
    corpus_dataset = TVRR_DataLoader_corpus(corpus_path = args.corpus_path, 
                                            video_dir = args.video_dir, max_frame_count=args.max_frame_count,)
    corpus_dataloader = DataLoader(corpus_dataset, batch_size=args.batch_size, 
                                   num_workers=args.num_workers, pin_memory=True, shuffle=False)
    
    train_dataset = TVRR_DataLoader_train(annotation_path=args.train_path, video_dir=args.video_dir,
                                          tokenizer=tokenizer, max_words=args.max_words, max_frame_count=args.max_frame_count)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                   num_workers=args.num_workers, pin_memory=True, shuffle=True)
    
    val_dataset = TVRR_DataLoader_eval(annotation_path=args.val_path, corpus_path = args.corpus_path, video_dir=args.video_dir,
                                      tokenizer=tokenizer, max_words=args.max_words, max_frame_count=args.max_frame_count)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, 
                                   num_workers=args.num_workers, pin_memory=True, shuffle=False)
    
    test_dataset = TVRR_DataLoader_eval(annotation_path=args.test_path, corpus_path = args.corpus_path, video_dir=args.video_dir,
                                      tokenizer=tokenizer, max_words=args.max_words, max_frame_count=args.max_frame_count)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, 
                                   num_workers=args.num_workers, pin_memory=True, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader, corpus_dataloader, corpus_dataset.corpus, val_dataset.ground_truth, test_dataset.ground_truth



def prepare_dataloader_segment(args, tokenizer):
    train_dataset = TVRR_DataLoader_train_segment(annotation_path=args.train_path, video_dir=args.video_dir,
                                          tokenizer=tokenizer, max_words=args.max_words, max_frame_count=args.max_frame_count)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                   num_workers=args.num_workers, pin_memory=True, shuffle=True)
    
    
    corpus_dataset = TVRR_DataLoader_corpus_segment(corpus_path = args.corpus_path, 
                                            video_dir = args.video_dir, max_frame_count=args.max_frame_count,)
    corpus_dataloader = DataLoader(corpus_dataset, batch_size=args.batch_size, 
                                   num_workers=args.num_workers, pin_memory=True, shuffle=False)
    

    val_dataset = TVRR_DataLoader_eval(annotation_path=args.val_path, corpus_path = args.corpus_path, video_dir=args.video_dir,
                                      tokenizer=tokenizer, max_words=args.max_words, max_frame_count=args.max_frame_count)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, 
                                   num_workers=args.num_workers, pin_memory=True, shuffle=False)
    
    test_dataset = TVRR_DataLoader_eval(annotation_path=args.test_path, corpus_path = args.corpus_path, video_dir=args.video_dir,
                                      tokenizer=tokenizer, max_words=args.max_words, max_frame_count=args.max_frame_count)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, 
                                   num_workers=args.num_workers, pin_memory=True, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader, corpus_dataloader, corpus_dataset.corpus_video_list, val_dataset.ground_truth, test_dataset.ground_truth

