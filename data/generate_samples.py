import argparse
import os
from glob import glob
from PIL import Image
import requests
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import open_clip
import supervision as sv
from groundingdino.util.inference import load_model, load_image, predict, box_convert


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, help="path to image directory or txt file with urls")
    parser.add_argument("--image_count", type=int)
    parser.add_argument("--output_parquet", default="image_layouts.parquet", type=str, help="path to save results")
    parser.add_argument("--clip_model", default="coca_ViT-L-14", type=str)
    parser.add_argument("--clip_pretrained", default="mscoco_finetuned_laion2B-s13B-b90k", type=str)
    parser.add_argument("--det_config_path", default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", type=str)
    parser.add_argument("--det_weights_path", default="weights/groundingdino_swint_ogc.pth", type=str)

    return parser


def generate_caption(args, model, transform):
    url = False
    if os.path.isdir(args.images):
        image_paths = glob(args.images + "/*.png") + glob(args.images + "/*.jpg") + glob(args.images + "/*.jpeg")
    elif args.images.endswith(".txt"):
        with open(args.images, 'r') as f:
            image_paths = f.readlines()
        url = True
    else:
        raise ValueError("images argument must be a directory with images or a text file with image urls")
          
    captions = []
    with torch.no_grad(), torch.cuda.amp.autocast():  
        for im_path in tqdm(image_paths[:args.image_count], total=min(len(image_paths), args.image_count)):
            if url:
                im = Image.open(requests.get(im_path, stream=True).raw).convert("RGB")
            else:
                im = Image.open(im_path).convert("RGB")

            generated = model.generate(transform(im).unsqueeze(0))  
            captions.append(open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", ""))

    return pd.DataFrame([image_paths, captions], columns=["image_path", "caption"])


def fantasize_bboxes(args, model, caption_df):
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    img_boxes = []
    for im_path, caption in tqdm(zip(caption_df["image_path"], caption_df["caption"]), total=len(list(caption_df["caption"]))):
        image_source, image = load_image(im_path)

        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=caption,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
        # convert to xyxy format, scale to image size of 512, and represent as integers
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        xyxy = np.around(xyxy * 512, 0).astype(np.int32)
        img_boxes.append([(p, box) for p, box in zip(phrases, xyxy)])   

    caption_df["boxes"] = img_boxes
    return caption_df
            

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _, transform = open_clip.create_model_and_transforms(
        model_name=args.clip_model,
        pretrained=args.clip_pretrained
    )
    clip_model.to(device)
    print("Generating caption for each image...")
    captions_df = generate_caption(args, clip_model, transform)
    del clip_model

    det_model = load_model(args.det_config_path, args.det_weights_path)
    det_model.to(device)
    print("Predicting boxes for each image...")
    captions_df = fantasize_bboxes(args, det_model, captions_df)
    
    print("Saving results...")
    captions_df.to_parquet(args.output_parquet, index=False)