from ast import literal_eval
from braceexpand import braceexpand
import io
import logging

from torch.utils.data import Dataset, DataLoader
import webdataset as wds
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)
import pandas as pd
import numpy as np


class DinoGeneratedDataset(Dataset):
    def __init__(self, parquet_file: str):
        df = pd.read_parquet(parquet_file)
        self.captions = df['caption'].to_list()
        self.boxes = df['boxes'].to_list()

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        bboxes = [literal_eval(bbox) for bbox in self.boxes[idx]]
        bbox_str = str([(phrase, literal_eval(box)) for phrase, box in bboxes])
        return "Caption: " + self.captions[idx] + "\nObjects: " + bbox_str


def get_dino_dataloader(args, tokenizer):
    dataset = DinoGeneratedDataset(args.captions)

    def collate_fn(batch):
        targets = tokenizer(
            batch[1],
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors='pt',
            return_attention_mask=True
        )

        return targets

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.workers,
        pin_memory=True
    )

    return dataloader


# Referencing https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/train/data.py
def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    if "No images in sample" in str(exn) or "Only one image in sample" in str(
        exn
    ):  # Avoid spamming logs with these
        return True
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def group_by_keys_nothrow(
    data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None
):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if (
            current_sample is None
            or prefix != current_sample["__key__"]
            or suffix in current_sample
        ):
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def samples_from_parquet(data):
    for sample in data:
        if "parquet" in sample:
            df = pd.read_parquet(io.BytesIO(sample['parquet']))
            for i in range(len(df)):
                # Split caption and narrative into different samples
                for text_name, boxes_name in [('caption', 'caption_boxes'), ('narrative', 'narrative_boxes')]:
                    yield (
                        df[text_name][i],
                        df[boxes_name][i],
                    )


def preprocess_text(sample):
    caption, bboxes = sample
    bboxes = [literal_eval(bbox) for bbox in bboxes]

    bbox_str = str([(phrase, (np.array(literal_eval(box)) / 512).round(2).tolist()) for phrase, box in bboxes])
    return (sample, "Caption: " + caption + "\nObjects: " + bbox_str)


def get_narratives_dataloader(
    args,
    tokenizer,
):
    dataset = wds.DataPipeline(
        wds.SimpleShardList(braceexpand(args.captions)),
        wds.shuffle(),
        wds.split_by_worker,
        tarfile_to_samples_nothrow,
    ).compose(samples_from_parquet, wds.map(preprocess_text), wds.shuffle(1000), wds.batched(args.batch_size))

    def collate_fn(batch):
        targets = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors='pt',
            return_attention_mask=True
        )

        return targets

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.workers,
        persistent_workers=True,
    )

    return dataloader
