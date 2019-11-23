
from ssd.config.path_catlog import DatasetCatalog
from .voc_dataset import VOCDataset
from .coco_dataset import COCODataset
from .voc_model_dataset import VOCModelDataset
from .detection_subset_dataset import DetectionSubset
from .detection_concat_dataset import DetectionConcatDataset
import numpy as np

_DATASETS = {
    'VOCDataset': VOCDataset,
    'COCODataset': COCODataset,
    'VOCModelDataset': VOCModelDataset
}


def get_train_val_splits(dataset, split_val_size=10):
    """used to produce a split of a dataset in train set and validation set
    :param dataset: to be used for the split
    :param split_val_size: percentage of the dataset to be used for the validation split
    :return: tuple (train_set, validation_set)
    """
    ds_size = len(dataset)
    # set the seed of the random function for the next call so that it returns always the same indices
    np.random.seed(0)
    val_indices = np.random.choice(ds_size, int((split_val_size/100) * ds_size), replace=False)
    ia = np.indices((ds_size,))
    train_indices = np.setxor1d(ia, val_indices)
    train_set = DetectionSubset(dataset, train_indices)
    val_set = DetectionSubset(dataset, val_indices)

    return train_set, val_set


def build_dataset(dataset_list, transform=None, target_transform=None, is_test=False, split=False, split_val_size=10):
    assert len(dataset_list) > 0
    datasets = []
    for dataset_name in dataset_list:
        data = DatasetCatalog.get(dataset_name)
        args = data['args']
        factory = _DATASETS[data['factory']]
        args['transform'] = transform
        args['target_transform'] = target_transform
        if factory == VOCDataset or factory == VOCModelDataset:
            args['keep_difficult'] = is_test
        elif factory == COCODataset:
            args['remove_empty'] = not is_test
        dataset = factory(**args)
        datasets.append(dataset)
    # for testing, return a list of datasets
    if is_test:
        return datasets
    if len(datasets) > 1:
        dataset = DetectionConcatDataset(datasets)
        if split:
            return get_train_val_splits(dataset, split_val_size)
    else:
        dataset = datasets[0]
        if split:
            return get_train_val_splits(dataset, split_val_size)

    return dataset
