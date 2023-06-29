# Layout Diffusion

Open Source implementation of [LLM-grounded Diffusion: Enhancing Prompt Understanding of Text-to-Image Diffusion Models with Large Language Models](https://arxiv.org/abs/2305.13655).


## Installation

```
conda create -n "LayoutSD" python=3.9
pip install -r requirements.txt

git clone https://github.com/sramshetty/LayoutSD.git
cd LayoutSD

git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/
pip install -e .

cd ..
mkdir weights/
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```


## TODO:
- [x] Generate samples for finetuning (currently only 20k)
    - [ ] Larger set may be nice or necessary
- [ ] Finetune LM on samples to generate new bounding boxes given a caption
- [ ] Implement masking
