import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_tvrr_retrieval import TVRR_DataLoader_train, TVRR_DataLoader_eval, TVRR_DataLoader_corpus
from dataloaders.dataloader_tvrr_segment import TVRR_DataLoader_train_segment, TVRR_DataLoader_corpus_segment
from dataloaders.dataloader_tvrr_CLIP import TrainVideoDataset, CorpusVideoDataset, EvalVideoDataset



def collate_fn(processor, batch, task='train'):
    if task == 'train':
        texts, frames, video_masks = zip(*batch)
        text_inputs = processor(text=list(texts), return_tensors="pt", padding=True, truncation=True, max_length=77)
        video_inputs = processor(images=[frame for video in frames for frame in video], return_tensors="pt", padding=True)
        video_inputs['pixel_values'] = video_inputs['pixel_values'].view(len(texts), -1, 3, 224, 224)
        video_masks = torch.stack(video_masks)
        return text_inputs['input_ids'], text_inputs['attention_mask'], video_inputs['pixel_values'], video_masks
    elif task == 'corpus':
        frames, video_masks = zip(*batch)
        video_inputs = processor(images=[frame for video in frames for frame in video], return_tensors="pt", padding=True)
        video_inputs['pixel_values'] = video_inputs['pixel_values'].view(len(video_masks), -1, 3, 224, 224)
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

