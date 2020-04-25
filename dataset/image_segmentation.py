''' Simple image segmenation dataset pipeline with less library-dependency '''

from glob import glob
import cv2
import numpy as np


def glob_files(queries):
    ''' A function to correct target files with glob function.

    Args:
      queries: A list or a str indicating the target files. Applied to glob function.
      
    Returns:
      A list containining the corrected paths.
    '''
    if isinstance(queries, list):
        pass
    elif isinstance(queries, str):
        queries = [queries]
    else:
        raise TypeError(
            f'queries are expected to be a list or a str, {queries} ware given.'
        )

    files = []
    for q in queries:
        files += glob(q)
    return sorted(files)


class Dataset:
    """Standard Dataset class. Read images,
    apply preprocessings and augmentations.
    
    Args:
      image_files: List files indicating image path
      mask_files: List of files indicating mask path.
        If None, dataset returns image for both input/output.
      preprocessing (albumentations.Compose): data preprocessing
        (e.g. noralization, shape manipulation, etc.)
      augmentation (albumentations.Compose):
        data transfromation pipeline (e.g. flip, scale, etc.)
    """
    def __init__(self,
                 image_files,
                 mask_files=None,
                 preprocessing=None,
                 augmentation=None):
        self.image_files = image_files
        self.mask_files = mask_files
        self.preprocessing = preprocessing
        self.augmentation = augmentation

    def __getitem__(self, i):
        image = cv2.imread(self.image_files[i], 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = None
        if self.mask_files:
            mask = cv2.imread(self.mask_files[i], 0)
            mask = mask[:, :, np.newaxis]

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)

        if mask is not None:
            return image, mask
        else:
            return image, image


def build_dataloader(dataset, batch_size=1, shuffle=False):
    ''' A function that builds a dataloader.

    Args:
      dataset: A Dataset object iterating samples.
      batch_size=1: Integer value specifying batch size.
      shuffle=False: Boolean value specifying whether to shuffle or not.

    Returns:
      An iterator yielding samples.
    '''
    while True:
        indexes = np.arange(len(dataset))
        if shuffle:
            indexes = np.random.permutation(indexes)

        batch_idx = 0
        xs, ys = [], []
        for i in indexes:
            x, y = dataset[i]
            xs.append(x)
            ys.append(y)

            batch_idx += 1
            if batch_idx % batch_size == 0:
                yield (np.stack(xs), np.stack(ys))
                xs, ys = [], []
