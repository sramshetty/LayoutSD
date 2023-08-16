from ast import literal_eval

import torch
from torch.utils.data import Dataset, DataLoader

from wds_utils import *


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


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


def get_dino_data(args, tokenizer):
    dataset = DinoGeneratedDataset(args.captions)

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

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.workers,
        pin_memory=True
    )

    dataloader.num_samples = args.num_samples

    return DataInfo(dataloader=dataloader)


MIN_KB = 10

def preprocess_image(sample, image_processor):
    image = [image_processor(s).unsqueeze(0) for s in sample]
    image = torch.cat(image, dim=0)
    # apply random horizontal flip
    # image = torchvision.transforms.RandomHorizontalFlip(p=0.5)(image)
    # NOTE: potentially move jitter into the image_preprocessor before normalization
    # image = torchvision.transforms.ColorJitter(brightness=0.5, hue=0.3)(image)
    return image


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


def get_image_data(args, image_processor, epoch=0, floor=False):
    input_shards = args.shards
    assert input_shards is not None

    num_samples, num_shards = get_dataset_size(input_shards)
    num_samples = None
    if not num_samples:
        num_samples = args.image_count
        if not num_samples:
            raise RuntimeError(
                "Currently, number of dataset samples must be specified for dataset. "
                "Please specify via `--image_count` if no dataset length info present."
            )

    # create a shared epoch store to sync epoch to dataloader worker proc
    pipeline = [wds.SimpleShardList(input_shards)]

    # create preprocess function that take in the passed in image_processor
    # preprocess_image_fn = functools.partial(
    #     preprocess_image, image_processor=image_processor
    # )

    pipeline.extend(
        [
            wds.split_by_node,
            wds.split_by_worker,
            tarfile_to_samples_nothrow,
            wds.select(filter_no_caption_or_no_image),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.to_tuple("jpg;png;jpeg", "__key__", "__url__", handler=log_and_continue),
            wds.batched(args.batch_size, partial=False),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)
    assert (
        num_shards >= args.workers
    ), "number of shards must be >= total workers"
    # roll over and repeat a few samples to get same number of full batches on each node
    round_fn = math.floor if floor else math.ceil
    num_batches = round_fn(num_samples / args.batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * args.batch_size
    # each worker is iterating over this
    dataset = dataset.with_epoch(num_worker_batches)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader)


def get_narratives_data(
    args,
    tokenizer,
    epoch=0,
    floor=False
):
    shared_epoch = SharedEpoch(epoch=epoch)

    dataset = wds.DataPipeline(
        wds.SimpleShardList(braceexpand(args.captions)),
        detshuffle2(
            bufsize=_SHARD_SHUFFLE_SIZE,
            initial=_SHARD_SHUFFLE_INITIAL,
            seed=args.seed,
            epoch=shared_epoch,
        ),
        wds.split_by_worker,
        tarfile_to_samples_nothrow,
    ).compose(
        samples_from_parquet,
        wds.map(preprocess_text),
        wds.shuffle(
            bufsize=_SAMPLE_SHUFFLE_SIZE,
            initial=_SAMPLE_SHUFFLE_INITIAL,
        ),
        wds.batched(args.batch_size, partial=False)
    )

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


    round_fn = math.floor if floor else math.ceil
    global_batch_size = args.batch_size * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    # each worker is iterating over this
    dataset = dataset.with_epoch(num_worker_batches)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.workers,
        persistent_workers=True,
    )

    dataloader.num_samples = args.num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_dataset_fn(dataset_type, args, image_processor, tokenizer, epoch=0):
    if dataset_type == "generation_wds":
        return get_image_data(args, image_processor, epoch)
    elif dataset_type == "dino_texts":
        return get_dino_data(args, tokenizer)
    elif dataset_type == "narrative_wds":
        return get_narratives_data(args, tokenizer, epoch)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, image_processor=None, tokenizer=None, dataset_type="generation_wds", epoch=0):
    return get_dataset_fn(dataset_type, args, image_processor, tokenizer, epoch)