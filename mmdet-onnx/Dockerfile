FROM python:3.8-slim

WORKDIR /root

RUN apt-get update -y \
    && apt-get install -y build-essential git wget python3-opencv \
    && apt-get -y clean all

RUN pip install torch==1.11+cpu torchvision==0.12+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html \
    && pip install protobuf==3.20 onnx==1.11 onnxruntime==1.11 \
    && pip install jupyter-client==6.1.12 jupyter jupyterlab
    
RUN pip install mmcv-full==1.5.2 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.11/index.html

RUN git clone -b panopticfpn_onnx https://github.com/daigo0927/mmdetection.git  \
    && cd mmdetection \
    && pip install -r requirements/build.txt \
    && pip install -v -e .
