import os
import sys
import yaml
import json
import hydra
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import xml.etree.ElementTree as ET
from glob import glob
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple

import tensorflow as tf
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.eval_util import object_detection_evaluation


def seed_everything(seed: int) -> None:
    ''' Fix random seed

    Args:
      seed (int): random seed
    '''
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_voc_annotation(path: str, label_map: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    ''' Parse VOC annotation XML file
    
    Args:
      path (str): xml path
      label_map (dict): transfer class name to class id **1-INDEXING**

    Returns:
      boxes (np.ndarray): bounding box with shape (N, 4),
        with Pascal VOC format: (xmin, ymin, xmax, ymax)
      labels (np.ndarray): object label with shape (N) **1-INDEXING**
    '''
    with open(path, 'r') as f:
        root = ET.fromstring(f.read())

    boxes, labels = [], []
    for obj in root.iter('object'):
        label_name = obj.find('name').text
        label = label_map[label_name]
        labels.append(label)

        box = [
            int(obj.find('bndbox/xmin').text),
            int(obj.find('bndbox/ymin').text),
            int(obj.find('bndbox/xmax').text),
            int(obj.find('bndbox/ymax').text)
        ]
        boxes.append(box)
    return np.array(boxes), np.array(labels)


def plot_detections(image: np.ndarray,
                    boxes: np.ndarray,
                    labels: np.ndarray,
                    scores: np.ndarray,
                    category_index: Dict[int, Dict],
                    min_score_thresh: float = 0.8,
                    figsize: Tuple[float, float] = (6, 8),
                    savepath: str = None) -> None:
    ''' Wrapper function to visualize detections
    
    Args:
      image (np.ndarray): target image with shape (height, width, 3)
      boxes (np.ndarray): predicted bounding box with shape (N, 4),
        with TF object detection format: (ymin, xmin, ymax, xmax)
        and coordinates are normalized by image size
      labels (np.ndarray): predicted labels with shape (N), 0-INDEXING
      scores (np.ndarray): prediction confidence scores with shape (N)
      category_index (dict): a dict containing category dictionaries
        (each holding category index `id` and category name `name`)
        keyed by category indices.
      min_score_thresh (float): minimum score of box to be visualized
      savepath (str): path to save figure
    '''
    image_with_annotations = image.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_with_annotations,
        boxes,
        labels,
        scores,
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=100,
        min_score_thresh=min_score_thresh)
    if savepath:
        plt.imsave(savepath, image_with_annotations)
    else:
        plt.imshow(image_with_annotations)


def convert_format(image: np.ndarray,
                   boxes: np.ndarray,
                   labels: np.ndarray,
                   num_classes: int,
                   label_id_offset: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    ''' Convert numpy data into tensor objects
    and VOC bbox format (xmin, ymin, xmax, ymax) into TF object detection format
    (ymin, xmin, ymax, xmax) normalized with height and width

    Args:
      image (np.ndarray): image with shape (height, width, 3)
      boxes (np.ndarray): bounding box with shape (N, 4)
        with Pascal VOC format: (xmin, ymin, xmax, ymax)
      labels (np.ndarray): labels with shape (N), 1-indexing
      num_classes (int): number of unique classes 
      label_id_offset (int): offset value for shift 1-indexed labels (usually 1)

    Returns:
      image: image tensor with shape (height, width, 3)
      boxes: bounding box tensor with shape (N, 4)
        with TF object detection format: (ymin, xmin, ymax, xmax)
        and coordinates are normalized by image size
      labels: one-hot encoded labels with shape (N, num_classes)
    '''
    h, w, _ = image.shape
    xmins, ymins, xmaxs, ymaxs = np.split(boxes, 4, axis=-1)
    boxes = np.concatenate([ymins/h, xmins/w, ymaxs/h, xmaxs/w], axis=-1)
    
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32) - label_id_offset
    labels = tf.one_hot(labels, num_classes)
    return image, boxes, labels
        

class VOCLoader:
    ''' Pascal VOC data loader class '''
    def __init__(self,
                 image_paths,
                 batch_size=1,
                 shuffle=False,
                 augment=None):
        ''' Constructor

        Args:
          image_paths (list): list of image paths
          batch_size (int): batch size
          shuffle (bool): whether to shuffle the dataset on iteration
          augment (Transform): albumentations transform object
        '''
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment

        self.num_classes = 20
        self.label_id_offset = 1

        self.label_map = {
            "aeroplane": 1,
            "bicycle": 2,
            "bird": 3,
            "boat": 4,
            "bottle": 5,
            "bus": 6,
            "car": 7,
            "cat": 8,
            "chair": 9,
            "cow": 10,
            "diningtable": 11,
            "dog": 12,
            "horse": 13,
            "motorbike": 14,
            "person": 15,
            "pottedplant": 16,
            "sheep": 17,
            "sofa": 18,
            "train": 19,
            "tvmonitor": 20
            }        

        if shuffle:
            np.random.shuffle(self.image_paths)

    def __len__(self):
        return np.ceil(len(self.image_paths)/self.batch_size).astype(int)

    def __getitem__(self, idx):
        images, boxes_list, labels_list = [], [], []
        for image_path in self.image_paths[self.batch_size*idx: self.batch_size*(idx+1)]:
            image = cv2.imread(image_path)[:, :, ::-1]

            annot_path = image_path.replace('JPEGImages', 'Annotations')\
                                   .replace('jpg', 'xml')
            boxes, labels = load_voc_annotation(annot_path, self.label_map)

            if self.augment:
                augmented = self.augment(image=image, bboxes=boxes, class_labels=labels)
                image = augmented['image']
                boxes = np.array(augmented['bboxes'])
                labels = np.array(augmented['class_labels'])

            image, boxes, labels = convert_format(
                image=image, boxes=boxes, labels=labels,
                num_classes=self.num_classes,
                label_id_offset=self.label_id_offset
            )

            images.append(image)
            boxes_list.append(boxes)
            labels_list.append(labels)

        if idx+1 >= len(self) and self.shuffle:
            np.random.shuffle(self.image_paths)

        return images, boxes_list, labels_list
        

def build_detection_model(pretrained_dir,
                          num_classes,
                          max_detections_per_class=100,
                          max_total_detections=100):
    ''' Build detection_model
    
    Args:
      pretrained_dir (str): directory containing pretrained model
      num_classes (int): number of classes
      finetune_weights_keys (list): 
      max_detections_per_class (int): maximum number of detections per class
      max_total_detections (int): maximum number of total detections

    Returns:
      detection_model: built model
      target_size: (height, width) of the detection model
      checkpoint (tf.train.Checkpoint): checkpoint object for overall weights
    '''
    tf.keras.backend.clear_session()
    pipeline_config = f'{pretrained_dir}/pipeline.config'
    checkpoint_path = f'{pretrained_dir}/checkpoint/ckpt-0'

    # Override config
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    model_config.ssd.num_classes = num_classes
    model_config.ssd.freeze_batchnorm = True
    model_config.ssd.post_processing.batch_non_max_suppression.max_detections_per_class \
        = max_detections_per_class
    model_config.ssd.post_processing.batch_non_max_suppression.max_total_detections \
        = max_total_detections
    # Initialize detection model    
    detection_model = model_builder.build(model_config=model_config,
                                          is_training=True)

    # Restore model weights except for label prediction head to be finetuned
    box_predictor = tf.train.Checkpoint(
        _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
        _box_predictor_head=detection_model._box_predictor._box_prediction_head
    )
    model = tf.train.Checkpoint(
        _feature_extractor=detection_model._feature_extractor,
        _box_predictor=box_predictor
    )
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(checkpoint_path).expect_partial()

    # Initialize classification weights via dummy inference
    target_size = (
        model_config.ssd.image_resizer.fixed_shape_resizer.height,
        model_config.ssd.image_resizer.fixed_shape_resizer.width
    )
    image, shapes = detection_model.preprocess(tf.zeros([1, *target_size, 3]))
    prediction_dict = detection_model.predict(image, shapes)
    _ = detection_model.postprocess(prediction_dict, shapes)

    checkpoint = tf.train.Checkpoint(detection_model=detection_model)
    tf.keras.backend.set_learning_phase(True)

    return detection_model, target_size, checkpoint


def get_train_step(model, optimizer, target_weights, target_size):
    ''' Construct train step function

    Args:
      model (DetectionModel): object detection model
      optimizer (Optimizer): tensorflow optimizer object
      target_weights (list): list of weights to be finetuned
      target_size (tuple): (height, width) of the detection model

    Returns:
      function of single train step, receives batch of image, boxes, labels,
        updates weights and returns total_loss
    '''
    @tf.function(experimental_relax_shapes=True)
    def train_step(images, boxes_list, labels_list):
        ''' A function of single train step
        
        Args:
          images (list): list of input images
          boxes_list (list): list of ground truth bounding boxes,
            coordinates should be normalized
          labels_list (list): list of ground truth labels,
            should be one-hot encoded

        Returns:
          total_loss (tensor): total loss value
        '''
        shapes = tf.constant(len(images)*[[*target_size, 3]], dtype=tf.int32)
        model.provide_groundtruth(
            groundtruth_boxes_list=boxes_list,
            groundtruth_classes_list=labels_list
        )
        with tf.GradientTape() as tape:
            # add batch dimension and preprocess image
            preprocessed_images = tf.concat([
                model.preprocess(image[None])[0] for image in images
            ], axis=0)
            prediction_dict = model.predict(preprocessed_images, shapes)
            losses_dict = model.loss(prediction_dict, shapes)
            total_loss = losses_dict['Loss/localization_loss'] \
                + losses_dict['Loss/classification_loss']
        gradients = tape.gradient(total_loss, target_weights)
        optimizer.apply_gradients(zip(gradients, target_weights))
        return total_loss
    return train_step


def validate(model, loader, num_visualize=10):
    batch_size = loader.batch_size
    label_id_offset = loader.label_id_offset

    if num_visualize > 0:
        figure_dir = 'figures'
        os.mkdir(figure_dir)
    
    evaluator = object_detection_evaluation.ObjectDetectionEvaluator(
        categories=[{'id': v, 'name': k} for k, v in loader.label_map.items()],
        evaluate_precision_recall=True
    )
    for i, batch in enumerate(loader):
        for j, (image, boxes, labels) in enumerate(zip(*batch)):
            preprocessed_image, shape = model.preprocess(image[None])
            prediction_dict = model.predict(preprocessed_image, shape)
            detection = model.postprocess(prediction_dict, shape)

            evaluator.add_eval_dict({
                'key': [batch_size*i+j],
                'groundtruth_boxes': [boxes.numpy()],
                'groundtruth_classes': [labels.numpy().argmax(axis=-1) + label_id_offset],
                'detection_boxes': [detection['detection_boxes'][0].numpy()],
                'detection_scores': [detection['detection_scores'][0].numpy()],
                'detection_classes': [detection['detection_classes'][0].numpy()]
            })

    return evaluator.evaluate()


def show_progress(epoch: int, batch: int, batch_total: int, **kwargs) -> None:
    ''' Show training progress with arbitrary values

    Args:
      epoch (int): 0-indexing epoch 
      batch (int): 0-indexing batch 
      batch_tatal (int): total number of batch
      kwargs (dict): additional keyword arguments to be shown
    '''
    message = f'\r{epoch+1} epoch: [{batch+1}/{batch_total}batches'
    for key, item in kwargs.items():
        message += f', {key:}: {item:.4f}'
    sys.stdout.write(message + ']')
    sys.stdout.flush()


def visualize(model, loader, num_visualize=10, logdir='samples'):
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    category_index = {v: {'id': v, 'name': k} for k, v in loader.label_map.items()}
    for i in range(num_visualize):
        image_path = loader.image_paths[i]
        image = cv2.imread(image_path)[:, :, ::-1]
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        preprocessed_image, shape = model.preprocess(image[None])
        prediction_dict = model.predict(preprocessed_image, shape)
        detection = model.postprocess(prediction_dict, shape)

        plot_detections(
            image=image.numpy().astype(np.uint8),
            boxes=detection['detection_boxes'][0].numpy(),
            labels=detection['detection_classes'][0].numpy(),
            scores=detection['detection_scores'][0].numpy(),
            category_index=category_index,
            savepath=f'{logdir}/sample_{i+1}.png'
        )


@hydra.main(config_name="config")
def run(config):
    config['data']['voc_dir'] = '../../' + config['data']['voc_dir']
    config['model']['pretrained_dir'] = '../../' + config['model']['pretrained_dir']
    
    seed_everything(config['seed'])

    # Build data loader
    data_config = config['data']
    voc_dir = data_config['voc_dir']
    image_paths = glob(f'{voc_dir}/JPEGImages/*.jpg')
    image_paths_train, image_paths_val \
        = train_test_split(image_paths, test_size=data_config['val_ratio'])
    
    augment = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ColorJitter()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    train_loader = VOCLoader(image_paths_train,
                             batch_size=data_config['batch_size'],
                             shuffle=True,
                             augment=augment)
    val_loader = VOCLoader(image_paths_val,
                           batch_size=data_config['batch_size'])
    
    # Build detection model
    model, target_size, ckpt = build_detection_model(**config['model'])
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Get single train step function
    train_config = config['train']
    optimizer = tf.keras.optimizers.Adam(train_config['learning_rate'])
    target_weights = [w for w in model.trainable_weights if 'Head' in w.name]
    train_step = get_train_step(
        model=model,
        optimizer=optimizer,
        target_weights=target_weights,
        target_size=target_size
    )

    for e in range(train_config['n_epochs']):
        for i, (images, boxes_list, labels_list) in enumerate(train_loader):
            loss = train_step(images, boxes_list, labels_list)
            show_progress(epoch=e, batch=i, batch_total=len(train_loader),
                          loss=loss.numpy())

        val_result = validate(model, val_loader)
        ckpt.save(f'{model_dir}/ckpt-{e}')

    visualize(model, val_loader, num_visualize=3)
    

if __name__ == '__main__':
    run()
