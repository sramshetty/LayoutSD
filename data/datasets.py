from ast import literal_eval

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import box_iou

from .wds_utils import *


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


def get_dino_data(args, tokenizer, epoch):
    dataset = DinoGeneratedDataset(args.data_path)

    shared_epoch = SharedEpoch(epoch=epoch)

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

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


MIN_KB = 10

def preprocess_image(sample, image_processor):
    image = [image_processor(s).unsqueeze(0) for s in sample]
    image = torch.cat(image, dim=0)
    # apply random horizontal flip
    # image = torchvision.transforms.RandomHorizontalFlip(p=0.5)(image)
    # NOTE: potentially move jitter into the image_preprocessor before normalization
    # image = torchvision.transforms.ColorJitter(brightness=0.5, hue=0.3)(image)
    return image


def samples_from_narratives(data):
    for sample in data:
        if "parquet" in sample:
            df = pd.read_parquet(io.BytesIO(sample['parquet']))
            for i in range(len(df)):
                # Split caption and narrative into different samples
                for text_name, boxes_name in [('caption', 'caption_boxes'), ('narrative', 'narrative_boxes')]:
                    
                    filtered_bboxes = []
                    for phrase, bbox in [literal_eval(bbox) for bbox in df[boxes_name][i]]:
                        bbox = (np.array(literal_eval(bbox))).tolist()
                        # bbox should be less than 450 pixels and more than 20 pixels in either dimension
                        if (abs(bbox[2] - bbox[0]) > 450) or (abs(bbox[2] - bbox[0]) < 20) or (abs(bbox[3] - bbox[1]) > 450) or (abs(bbox[3] - bbox[1]) < 20):
                            continue

                        # if coordinates are flipped, correct them as to avoid throwing away a sample
                        if bbox[2] - bbox[0] < 0:
                            bbox[0], bbox[2] = bbox[2], bbox[0]
                        if bbox[3] - bbox[1] < 0:
                            bbox[1], bbox[3] = bbox[3], bbox[1]

                        filtered_bboxes.append((phrase, bbox))

                    if len(filtered_bboxes) == 0:
                        continue
                    else:
                        # avoid having boxes that overlap too much
                        tensor_boxes = torch.Tensor([box for _, box in filtered_bboxes])
                        ious = box_iou(tensor_boxes, tensor_boxes)
                        
                        # iterate over bottom triangle to adjust boxes with too much overlap
                        duplicate_ids = set()
                        for r in range(len(filtered_bboxes)):
                            for c in range(r):
                                if ious[r][c] > 0.50:
                                    duplicate_ids.add(c)
                        filtered_bboxes = [fbox for idx, fbox in enumerate(filtered_bboxes) if idx not in duplicate_ids]

                    yield (
                        df[text_name][i],
                        filtered_bboxes,
                        512,
                        512
                    )


def samples_from_grit(data):
    for sample in data:
        if "parquet" in sample:
            df = pd.read_parquet(io.BytesIO(sample['parquet']))
            for caption, width, height, chunks in zip(df['caption'], df['width'], df['height'], df['noun_chunks']):
                filtered_bboxes = []

                for (start, end, x_min, y_min, x_max, y_max, _) in chunks:
                    phrase = caption[start:end]
                    bbox = list(np.around(np.array([x_min, y_min, x_max, y_max]), 2))
                    filtered_bboxes.append((phrase, bbox))

                yield (
                    caption,
                    filtered_bboxes,
                    width,
                    height
                )


def preprocess_text(sample):
    caption, bboxes, width, height = sample
    bbox_str = str([(phrase, bbox) for phrase, bbox in bboxes])
    return (sample, f"Caption: {caption.strip()}\nSize: {width}, {height}\nObjects: {bbox_str.strip()}")


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
    shared_epoch = SharedEpoch(epoch=epoch)

    pipeline = [wds.SimpleShardList(input_shards)]

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
    global_batch_size = args.batch_size * args.world_size
    num_batches = round_fn(args.num_samples / global_batch_size)
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
        num_workers=args.workers,
        persistent_workers=True,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_samples = num_samples
    dataloader.num_batches = num_batches

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_wds_data(
    args,
    tokenizer,
    epoch=0,
    sampling_fn=samples_from_narratives,
    floor=False,
    support=False
):
    data_path = args.supp_data_path if support else args.data_path
    batch_size = args.supp_batch_size if support else args.batch_size
    num_samples, num_shards = get_dataset_size(data_path)
    num_samples = None
    if not num_samples:
        num_samples = args.supp_num_samples if support else args.num_samples
        if not num_samples:
            raise RuntimeError(
                "Currently, number of dataset samples must be specified for training dataset. "
                "Please specify via `--train-num-samples` if no dataset length info present."
            )
        
    shared_epoch = SharedEpoch(epoch=epoch)
    if args.resampled:
        shard_list = ResampledShards2(data_path, deterministic=True, epoch=shared_epoch)
    else:
        shard_list = wds.SimpleShardList(data_path)

    dataset = wds.DataPipeline(
        shard_list,
        detshuffle2(
            bufsize=_SHARD_SHUFFLE_SIZE,
            initial=_SHARD_SHUFFLE_INITIAL,
            seed=args.seed,
            epoch=shared_epoch,
        ),
        wds.split_by_worker,
        tarfile_to_samples_nothrow,
    ).compose(
        sampling_fn,
        wds.map(preprocess_text),
        wds.shuffle(
            bufsize=_SAMPLE_SHUFFLE_SIZE,
            initial=_SAMPLE_SHUFFLE_INITIAL,
        ),
        wds.batched(batch_size, partial=False)
    )

    def collate_fn(batch):
        tokens = tokenizer(
            batch[1],
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors='pt',
            return_attention_mask=True
        )

        return tokens["input_ids"], tokens["attention_mask"]

    if not args.resampled:
        assert (
            num_shards >= args.workers * args.world_size
        ), "number of shards must be >= total workers"

    round_fn = math.floor if floor else math.ceil
    global_batch_size = batch_size * args.world_size
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

    dataloader.num_samples = num_samples
    dataloader.num_batches = num_batches

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_dataset_fn(dataset_type, args, image_processor, tokenizer, epoch=0):
    support = args.supp_dataset_type == dataset_type and dataset_type is not None
    if dataset_type == "generation_wds":
        return get_image_data(args, image_processor, epoch)
    elif dataset_type == "dino_texts":
        return get_dino_data(args, tokenizer, epoch)                                   
    elif dataset_type == "narrative_wds":
        return get_wds_data(args, tokenizer, epoch, sampling_fn=samples_from_narratives, support=support)
    elif dataset_type == "grit":
        return get_wds_data(args, tokenizer, epoch, sampling_fn=samples_from_grit, support=support)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, image_processor=None, tokenizer=None, epoch=0, dataset_type=None):
    if dataset_type is None:
        return None
    return get_dataset_fn(dataset_type, args, image_processor, tokenizer, epoch)