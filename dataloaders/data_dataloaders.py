import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_tvrr_retrieval import TVRR_DataLoader_train, TVRR_DataLoader_eval, TVRR_DataLoader_corpus
from dataloaders.dataloader_tvrr_segment import TVRR_DataLoader_train_segment, TVRR_DataLoader_corpus_segment



def prepare_dataloader_video(args, tokenizer):
    corpus_dataset = TVRR_DataLoader_corpus(corpus_path = args.corpus_path, 
                                            video_path = args.video_path, max_frames=args.max_frames,)
    corpus_dataloader = DataLoader(corpus_dataset, batch_size=args.batch_size, 
                                   num_workers=args.num_workers, pin_memory=False, shuffle=False)
    
    train_dataset = TVRR_DataLoader_train(annotation_path=args.train_path, video_path=args.video_path,
                                          tokenizer=tokenizer, max_words=args.max_words, max_frames=args.max_frames)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                   num_workers=args.num_workers, pin_memory=False, shuffle=True)
    
    val_dataset = TVRR_DataLoader_eval(annotation_path=args.val_path, corpus_path = args.corpus_path, video_path=args.video_path,
                                      tokenizer=tokenizer, max_words=args.max_words, max_frames=args.max_frames)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, 
                                   num_workers=args.num_workers, pin_memory=False, shuffle=False)
    
    test_dataset = TVRR_DataLoader_eval(annotation_path=args.test_path, corpus_path = args.corpus_path, video_path=args.video_path,
                                      tokenizer=tokenizer, max_words=args.max_words, max_frames=args.max_frames)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, 
                                   num_workers=args.num_workers, pin_memory=False, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader, corpus_dataloader, corpus_dataset.corpus, val_dataset.ground_truth, test_dataset.ground_truth



def prepare_dataloader_segment(args, tokenizer):
    train_dataset = TVRR_DataLoader_train_segment(annotation_path=args.train_path, video_path=args.video_path,
                                          tokenizer=tokenizer, max_words=args.max_words, max_frames=args.max_frames)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                   num_workers=args.num_workers, pin_memory=False, shuffle=True)
    
    
    corpus_dataset = TVRR_DataLoader_corpus_segment(corpus_path = args.corpus_path, 
                                            video_path = args.video_path, max_frames=args.max_frames,)
    corpus_dataloader = DataLoader(corpus_dataset, batch_size=args.batch_size, 
                                   num_workers=args.num_workers, pin_memory=False, shuffle=False)
    

    val_dataset = TVRR_DataLoader_eval(annotation_path=args.val_path, corpus_path = args.corpus_path, video_path=args.video_path,
                                      tokenizer=tokenizer, max_words=args.max_words, max_frames=args.max_frames)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, 
                                   num_workers=args.num_workers, pin_memory=False, shuffle=False)
    
    test_dataset = TVRR_DataLoader_eval(annotation_path=args.test_path, corpus_path = args.corpus_path, video_path=args.video_path,
                                      tokenizer=tokenizer, max_words=args.max_words, max_frames=args.max_frames)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, 
                                   num_workers=args.num_workers, pin_memory=False, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader, corpus_dataloader, corpus_dataset.corpus_video_list, val_dataset.ground_truth, test_dataset.ground_truth

