from torch.utils.data import Dataset


class DetectionSubset(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.class_names = dataset.class_names

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def get_image(self, idx, apply_transform=True):
        return self.dataset.get_image(self.indices[idx], apply_transform)

    def get_annotation(self, idx):
        return self.dataset.get_annotation(self.indices[idx])

    def __len__(self):
        return len(self.indices)
