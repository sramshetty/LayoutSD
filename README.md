# Layout Diffusion (WIP)

A fully open-source implementation of [LLM-grounded Diffusion: Enhancing Prompt Understanding of Text-to-Image Diffusion Models with Large Language Models](https://arxiv.org/abs/2305.13655).


## Installation

Note: Update requirements with your desired CUDA version

```
git clone https://github.com/sramshetty/LayoutSD.git
cd LayoutSD

conda create -n "LayoutSD" python=3.9
pip install -r requirements.txt

git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/
pip install -e .

cd ..
mkdir weights/
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```


## Finetuning

```
python finetune.py \
    --caption_file generated_data/image_layouts_20K.parquet \
    --batch_size 2 \
    --workers 0 \
    --lr 0.00003 \
    --decay 0.01 \
    --epochs 5 \
    --warmup 5000 \
    --checkpoint_path checkpoints/
```


## TODO:
- [x] Generate samples for finetuning (currently only 20k)
    - [ ] Larger set may be nice or necessary
- [x] Finetune LM on samples to generate new bounding boxes given a caption
- [x] Per-Box Masked Latent Inversion
    - [x] Generate per-box image w/ cross-attention masking
    - [x] Get mask using cross-attention map for object token
    - [x] Get inverted latent of image
    - [x] Use mask to get object specific inverted latent 
- [x] Compose Masked Latents w/ Background Latent
- [x] Pipeline these steps

Roadmap:
- [ ] LLM for example generation
- [ ] Improve per-box image generation -> more consistency
- [ ] Refine masking method
- [ ] Better compositional generation strategy?
- [ ] Larger SD model?


## Acknowledgements

Original Paper

```bibtex
@article{lian2023llmgrounded,
    title={LLM-grounded Diffusion: Enhancing Prompt Understanding of Text-to-Image Diffusion Models with Large Language Models}, 
    author={Lian, Long and Li, Boyi and Yala, Adam and Darrell, Trevor},
    journal={arXiv preprint arXiv:2305.13655},
    year={2023}
}
```

We thank these additional works for their helpful resources/repositories:

```bibtex
@article{chen2023trainingfree,
      title={Training-Free Layout Control with Cross-Attention Guidance}, 
      author={Minghao Chen and Iro Laina and Andrea Vedaldi},
      journal={arXiv preprint arXiv:2304.03373},
      year={2023}
}

@article{hertz2022prompt,
  title = {Prompt-to-Prompt Image Editing with Cross Attention Control},
  author = {Hertz, Amir and Mokady, Ron and Tenenbaum, Jay and Aberman, Kfir and Pritch, Yael and Cohen-Or, Daniel},
  journal = {arXiv preprint arXiv:2208.01626},
  year = {2022},
}
```

