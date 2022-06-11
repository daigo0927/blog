# Export MMDetection model to ONNX

Sample export step for MMDetection to ONNX.

Though MMDetection provides a [sample script for ONNX export](https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html), some models is not supported because PyTorch operations are not fully compatible with ONNX opset.

This example shows how to export PanopticFPN model (a vision model performing [panoptic segmentation task](https://paperswithcode.com/task/panoptic-segmentation)) to ONNX. MMDetection PanopticFPN is not exportable to ONNX because:

1. using `F.grid_sample` operation not supported by PyTorch2ONNX export (merged pytorch:master recently, may be available soon as a stable version).
2. `mmdet.models.PanopticFPN` class does not implement `export_onnx` method.

For 1. this example uses Microsoft's contrib operation as a counterpart for `F.grid_sample`. For 2. I have commit `export_onnx` method to my [forked MMDetection repository](https://github.com/daigo0927/mmdetection/tree/panopticfpn_onnx). This change would be merged to the original repository :).

## Export steps

Download PanopticFPN checkpoint from MMDetection: [https://download.openmmlab.com/mmdetection/v2.0/panoptic_fpn/panoptic_fpn_r50_fpn_1x_coco/panoptic_fpn_r50_fpn_1x_coco_20210821_101153-9668fd13.pth](DL link), place it under `checkpoints` directory.

Build and run docker container:

```shell
docker compose up --build mmdet2onnx
# launches jupyter-lab and shows its link
```

Enter jupyter-lab and run `panopticfpn_onnx.ipynb`.
