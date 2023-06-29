### Usage

Note: May need to follow this [issue](https://github.com/IDEA-Research/GroundingDINO/issues/8#issuecomment-1541892708) to resolve detection errors when using GPU.

Example:

Assumes that your working directory is `LayoutSD` with the installation described [here](../README.md). 
```
python data/generate_samples.py --images ../Data/coco_train2017 --image_count 10
```

For different directory organizations, you can change the paths as seen below:
```
python generate_samples.py --shards "laion400m-data/{00000..00001}.tar" --det_weights_path <path to weights> --det_config_path <path to config> --image_count 10
```