import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_tvrr_retrieval import TVRR_DataLoader, TVRR_Corpus_DataLoader


def dataloader_TVRR_video_corpus(annotation_path, args, tokenizer):
    tvrr_dataset = TVRR_Corpus_DataLoader(
        annotation_path = annotation_path,
        video_path = args.video_path,
        dataset_type = "train",
        tokenizer=tokenizer,
        max_words=args.max_words,
        max_frames=args.max_frames,
        # feature_framerate=args.feature_framerate,
        # frame_order=args.train_frame_order,
        # slice_framepos=args.slice_framepos,
    )

    dataloader = DataLoader(
        tvrr_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=False,
    )

    return dataloader, len(tvrr_dataset)


def dataloader_TVRR_train(annotation_path, args, tokenizer):
    tvrr_dataset = TVRR_DataLoader(
        annotation_path = annotation_path,
        video_path = args.video_path,
        dataset_type = "train",
        tokenizer=tokenizer,
        max_words=args.max_words,
        max_frames=args.max_frames,
        # feature_framerate=args.feature_framerate,
        # frame_order=args.train_frame_order,
        # slice_framepos=args.slice_framepos,
    )

    dataloader = DataLoader(
        tvrr_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=False,
    )

    return dataloader, len(tvrr_dataset)






def dataloader_TVRR_train(annotation_path, args, tokenizer):
    tvrr_dataset = TVRR_DataLoader(
        annotation_path = annotation_path,
        video_path = args.video_path,
        dataset_type = "train",
        tokenizer=tokenizer,
        max_words=args.max_words,
        max_frames=args.max_frames,
        # feature_framerate=args.feature_framerate,
        # frame_order=args.train_frame_order,
        # slice_framepos=args.slice_framepos,
    )

    dataloader = DataLoader(
        tvrr_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=False,
    )

    return dataloader, len(tvrr_dataset)


def dataloader_TVRR_train(annotation_path, args, tokenizer):
    tvrr_dataset = TVRR_DataLoader(
        annotation_path = annotation_path,
        video_path = args.video_path,
        dataset_type = "train",
        tokenizer=tokenizer,
        max_words=args.max_words,
        max_frames=args.max_frames,
        # feature_framerate=args.feature_framerate,
        # frame_order=args.train_frame_order,
        # slice_framepos=args.slice_framepos,
    )

    dataloader = DataLoader(
        tvrr_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=False,
    )

    return dataloader, len(tvrr_dataset)
