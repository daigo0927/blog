# PanopticFPN ONNX export via MMDeploy

[MMDeploy](https://github.com/open-mmlab/mmdeploy) is a framework for deploying ML Models of open-mmlab.

This is an example workload for exporting PanopticFPN model into ONNX format and also tests my Pull Request that implementing PanopticFPN exportation.

## Prerequisite

`sample.ipynb` shows ONNX export for Mask R-CNN and PanopticFPN. This uses the original PyTorch weights available at MMDetection. 

Download from below links and place under `checkpoints` directory.

- Mask R-CNN: [DL link](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth)
- PanopticFPN: [DL link](https://download.openmmlab.com/mmdetection/v2.0/panoptic_fpn/panoptic_fpn_r50_fpn_1x_coco/panoptic_fpn_r50_fpn_1x_coco_20210821_101153-9668fd13.pth)
- Other weights would work with a few line of code change.

## Run

Build and run the container. This shows a link of JupyterLab.

```bash
docker compose up (--build) mmdeploy-export
```

Open the `sample.ipynb` at JupyterLab, run all cells to export PanopticFPN to ONNX format. Output ONNX model would be placed at `onnx_models` directory.

### Note

`sample.ipynb` exports all the ONNX model with a name `end2end.onnx` by default.

If you are like to save the model with individual names, you can specify the name via the deployment config. See details at the [MMDeploy document](https://mmdeploy.readthedocs.io/en/latest/02-how-to-run/convert_model.html)
