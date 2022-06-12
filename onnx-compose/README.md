# Compose PyTorch components into a single ONNX model

When we serve a ML model, there often exists somepre/post-processing along with the main model. Composing those components into a single file is useful for serving the model regardless of platform differences.

This example shows how to compose pre/post-processing with the main vision model (EfficientNet).

## Assets

`Assets` directory contains some files for check:

- `dog1.jpg`: from [TorchVision](https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog1.jpg)
- `imagenet_classes.txt`: from [TorchHub](https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt)

## Usage

```bash
docker compose --build up onnx-compose
# launch container and print JupyterLab link
```

Open and run the `compose_onnx_ops.ipynb` and you would see the composed ONNX model under `onnx_models` directory :).
