import os
import cv2
import time
import numpy as np
import scipy.io
import logging
import argparse
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import albumentations as A
from glob import glob
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)

SEED = 42
N_CLASSES = 120
IMAGE_SIZE = (128, 128)


class StanfordDogs(Dataset):
    def __init__(self, datadir, split, preprocess, augment=None):
        self.datadir = datadir
        if split not in ['train', 'test']:
            raise ValueError('split should be either train or test')
        self.split = split
        self.preprocess = preprocess
        self.augment = augment

        self.lists = scipy.io.loadmat(f'{datadir}/{split}_list.mat')

    def __len__(self):
        return len(self.lists['file_list'])

    def __getitem__(self, idx):
        image_file = self.lists['file_list'][idx,0][0]
        image_path = f'{self.datadir}/Images/{image_file}'
        label = self.lists['labels'][idx,0] - 1

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.preprocess(image=image)['image']

        if self.augment:
            image = self.augment(image=image)['image']

        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image)
        label = torch.as_tensor(label, dtype=torch.long)
        return image, label


class EfficientNet(nn.Module):
    def __init__(self, backbone: str, n_classes: int):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.n_classes = n_classes

        self.fc = nn.Linear(self.backbone.bn2.num_features, n_classes)

    def forward(self, x):
        feature = self.backbone(x)
        logit = self.fc(feature)
        return logit


def setup(rank, n_gpus):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=n_gpus)


def cleanup():
    dist.destroy_process_group()

    
def fit(rank, datadir, n_gpus, epochs, batch_size, learning_rate):
    preprocess = A.Compose([
        A.LongestMaxSize(max(IMAGE_SIZE)),
        A.PadIfNeeded(*IMAGE_SIZE),
        A.Normalize()
    ])

    augment = A.Compose([
        A.RandomBrightness(0.3),
        A.RandomContrast(0.2),
        A.HorizontalFlip()
    ])

    ds_train = StanfordDogs(datadir, split='train',
                            preprocess=preprocess, augment=augment)
    ds_val = StanfordDogs(datadir, split='test',
                          preprocess=preprocess)
    
    sampler_train = DistributedSampler(ds_train, num_replicas=n_gpus, rank=rank,
                                       shuffle=True)
    sampler_val = DistributedSampler(ds_val, num_replicas=n_gpus, rank=rank,
                                     shuffle=False)
    dl_train = DataLoader(ds_train, batch_size=batch_size, sampler=sampler_train)
    dl_val = DataLoader(ds_val, batch_size=batch_size, sampler=sampler_val)

    model = EfficientNet(backbone='efficientnet_b2', n_classes=N_CLASSES)
    model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for e in range(epochs):
        model.train()
        for i, (images, labels) in enumerate(dl_train):
            images, labels = images.to(rank), labels.to(rank)
            logits = model(images)
            criterion(logits, labels).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for images, labels in dl_val:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)        

    cleanup()


def run(datadir, n_gpus, epochs, batch_size, learning_rate):
    t_train_start = time.time()    
    mp.spawn(fit,
             args=(datadir, n_gpus, epochs, batch_size, learning_rate),
             nprocs=n_gpus,
             join=True)
    t_train = time.time() - t_train_start
    logger.info(f'Training finished with {t_train:.4}s')    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch DDP training')
    parser.add_argument('datadir', type=str,
                        help='Path to the Stanford dogs dataset directory')
    parser.add_argument('-n', '--n-gpus', type=int, default=8,
                        help='Number of gpus to use, [8] default')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='Number of epochs, [10] default')
    parser.add_argument('-bs', '--batch-size', type=int, default=1024,
                        help='Batch size, [1024] default')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.001,
                        help='Learning rate, [0.001] default')
    args = parser.parse_args()

    run(**vars(args))
    
