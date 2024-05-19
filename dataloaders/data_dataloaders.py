import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_tvrr_retrieval import TVRR_DataLoader_train, TVRR_DataLoader_eval, TVRR_Corpus_DataLoader


def dataloader_TVRR_video_corpus(corpus_path, args):
    tvrr_dataset = TVRR_Corpus_DataLoader(
        corpus_path = corpus_path,
        video_path = args.video_path,
        max_frames=args.max_frames,
        # feature_framerate=args.feature_framerate,
        # frame_order=args.train_frame_order,
        # slice_framepos=args.slice_framepos,
    )
    dataloader = DataLoader(
        tvrr_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        shuffle=False,
    )
    return dataloader


def dataloader_TVRR_train(annotation_path, args, tokenizer):
    tvrr_dataset = TVRR_DataLoader_train(
        annotation_path = annotation_path,
        video_path = args.video_path,
        tokenizer=tokenizer,
        max_words=args.max_words,
        max_frames=args.max_frames,
    )
    dataloader = DataLoader(
        tvrr_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        shuffle=True,
    )
    return dataloader, tvrr_dataset

def dataloader_TVRR_eval(annotation_path, args, tokenizer):
    tvrr_dataset = TVRR_DataLoader_eval(
        annotation_path = annotation_path,
        corpus_path = args.corpus_path,
        video_path = args.video_path,
        tokenizer=tokenizer,
        max_words=args.max_words,
        max_frames=args.max_frames,
    )
    dataloader = DataLoader(
        tvrr_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        shuffle=False,
    )
    return dataloader, tvrr_dataset



