# Simple dataset pipelines

There exists some rich dataset pipeline APIis like `tf.data.Dataset` or `torch.utils.dataset`.
While they provide many useful method, using them inject some library dependency into your code and can cause a boaring error in the future.

This scripts implement simple dataset pipelines with only few library dependencies, and can be wrapped by those famous APIs.

# Example usage

## Segmentation task

```
from image_segmentation import Dataset, build_dataloader

import albumentations as A

preprocessing = A.Compose([
	A.Resize(64, 64),
])
albumentation = A.Compose([
	A.HorizontalFlip(),
	A.VerticalFlip(),
])

dataset = Dataset(
	image_paths=['image_dir1/*.png', 'image_dir2/*.png'],
	mask_paths=['mask_dir1/*.png', 'mask_dir2/*.png],
	preprocessing=preprocessing,
	augmentation=augmentation
)

loader = build_dataloader(dataset, batch_size=4, shuffle=True)

for image, mask in loader:
	# Your awesome training logic ...
```

## Transformation

```
from transforms import build_transform

import yaml

with open('sample_transformation.yml', 'wb') as f:
	transform_config = yaml.load(f) # See contents in sample files.

# Build albumentations.Compose object
transform = build_transform(**transform_config)

This object can be treated as preprocessing/augmentation argument of above dataset class.
```
