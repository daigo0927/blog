# Fine-tune object detection model on TensorFlow

A script for fine-tuning an object detection model (RetinaNet) via [TensorFlow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).

This script is based on the [official fine-tuning tutorial](https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/eager_few_shot_od_training_tf2_colab.ipynb), but brushed up for ease of understanding.

## Setup

### Install TensorFlow Object Detection API

Follow [official instruction](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md):

Note: Above instruction automatically install the Object Detection API with the **latest** TensorFlow version (mentioned in [this issue](https://github.com/tensorflow/models/issues/9634)). If you want to use a specific version of TensorFlow, you are required to manually reinstall `tensorflow, tf-models-official` after the official instruction:

``` bash
pip uninstall -y tf-models-official
pip install tensorflow==<TARGET_VERSION> tf-models-official==<TARGET_VERSION>
```

### Prepare data

Here I use PascalVOC dataset because this is popular and relatively light enough to demonstrate fine-tuning.

``` bash
curl -OL http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtrainval_06-Nov-2007.tar
```

You can use any object detection dataset just implementing data loading procedures.

### Prepare pre-trained weights

TensorFlow object detection hosts various models, see the [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) for details. Now I use RetinaNet following the [official tutorial](https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/eager_few_shot_od_training_tf2_colab.ipynb)

``` bash
curl -OL http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz .
tar -xvf ssd_resnet50_v1_fpn_640x640_coco17_tpu.tar.gz
```

## Train

TODO
