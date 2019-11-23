import torch.utils.data as data
from ...modeling.self_supervision_scramble import get_scrambler

import copy


class SelfSupervisedDataset(data.Dataset):
    def __init__(self, dataset, cfg):
        self.dataset = dataset
        self.scrambler = get_scrambler(cfg, 'cpu')
        if cfg.MODEL.SELF_SUPERVISOR.TYPE == 'rotation':
            self.aux_func = self.scrambler.rotate_single
        else:
            raise Exception

    def __getitem__(self, index):
        image_t, boxes, labels, orig_boxes, orig_labels = self.dataset[index]

        image_j = copy.deepcopy(image_t)
        j_image, order = self.aux_func(image_j)

        return image_t, boxes, labels, j_image, int(order), orig_boxes, orig_labels

    def __len__(self):
        return len(self.dataset)
