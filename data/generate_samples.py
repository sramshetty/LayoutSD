import argparse
import os
from glob import glob
from PIL import Image
import requests

import braceexpand
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

import open_clip
import supervision as sv
from groundingdino.util.inference import load_model, load_image, predict, box_convert
import groundingdino.datasets.transforms as T

from data import get_data


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, help="path to image directory or txt file with urls")
    parser.add_argument("--shards", type=str, help="webdataset shards")
    parser.add_argument("--image_count", type=int)

    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--workers", default=6, type=int)

    parser.add_argument("--output_parquet", default="image_layouts.parquet", type=str, help="path to save results")
    
    parser.add_argument("--clip_model", default="coca_ViT-L-14", type=str)
    parser.add_argument("--clip_pretrained", default="mscoco_finetuned_laion2B-s13B-b90k", type=str)
    
    parser.add_argument("--det_config_path", default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", type=str)
    parser.add_argument("--det_weights_path", default="weights/groundingdino_swint_ogc.pth", type=str)
    parser.add_argument("--box_threshold", default=0.35, type=float)
    parser.add_argument("--text_threshold", default=0.25, type=float)

    return parser


def generate_caption(args, model, transform, dataset=None):
    url = False
    if args.shards is not None:
        image_paths = dataset.dataloader
    elif os.path.isdir(args.images):
        image_paths = glob(args.images + "/*.png") + glob(args.images + "/*.jpg") + glob(args.images + "/*.jpeg")
    elif args.images.endswith(".txt"):
        with open(args.images, 'r') as f:
            image_paths = f.readlines()
        url = True
    else:
        raise ValueError("images argument must be a directory with images or a text file with image urls")
    
    if args.images is not None:
        image_paths = image_paths[:args.image_count]
    
    image_count = args.image_count if args.image_count is not None else len(image_paths)
    captions = []
    paths = []
    with torch.no_grad(), torch.cuda.amp.autocast():  
        for im_path in tqdm(image_paths, total=image_count):
            if args.shards is None:
                if url:
                    im = Image.open(requests.get(im_path, stream=True).raw).convert("RGB")
                else:
                    im = Image.open(im_path).convert("RGB")
                im = transform(im).unsqueeze(0)
            else:
                im = transform(im_path[0][0]).unsqueeze(0)
                # Store some information about image when using webdataset
                paths.append(str((im_path[1][0], im_path[2][0])))

            generated = model.generate(im.to(device))  
            captions.append(open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", ""))
    
    if args.shards is not None:
        image_paths = paths

    data = {
        "image_path": image_paths,
        "caption": captions
    }
    
    return pd.DataFrame(data)


def fantasize_bboxes(args, model, caption_df, dataset=None):
    BOX_TRESHOLD = args.box_threshold
    TEXT_TRESHOLD = args.text_threshold

    img_boxes = []
    paths = caption_df["image_path"] if dataset is None else dataset.dataloader
    for im_path, caption in tqdm(zip(paths, caption_df["caption"]), total=len(list(caption_df["caption"]))):
        if dataset is None:
            _, image = load_image(im_path)
        else:
            # Replicate transformation in "load_image()"
            det_transform = T.Compose(
                [
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            image, _ = det_transform(im_path[0][0], None)

        boxes, _, phrases = predict(
            model=model,
            image=image,
            caption=caption,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
        # convert to xywh format, scale to image size of 512 (don't overlap boundary), and represent as integers
        xywh = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xywh").numpy()
        xywh = np.around(xywh * 510, 0).astype(np.int32) + 1
        img_boxes.append([str((p, np.array2string(box, separator=", "))) for p, box in zip(phrases, xywh)])   

    caption_df["boxes"] = img_boxes
    return caption_df
            

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    assert args.images is not None or args.shards is not None, "must provide images or shards"

    # Currently, haven't implemented batched generation
    if args.batch_size != 1:
        args.batch_size = 1

    global devive
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = None

    clip_model, _, transform = open_clip.create_model_and_transforms(
        model_name=args.clip_model,
        pretrained=args.clip_pretrained
    )
    clip_model.to(device)

    print("Generating caption for each image...")
    if args.shards is not None:
        # Important to avoid shuffling the data here since we split processing into two parts
        dataset = get_data(args, transform)
    captions_df = generate_caption(args, clip_model, transform, dataset)
    del clip_model
    
    print("Saving caption results...")
    captions_df.to_parquet(args.output_parquet, index=False)

    det_model = load_model(args.det_config_path, args.det_weights_path)
    det_model.to(device)

    print("Predicting boxes for each image...")
    captions_df = fantasize_bboxes(args, det_model, captions_df, dataset)
    
    print("Saving results...")
    captions_df.to_parquet(args.output_parquet, index=False)

    print("Done!")