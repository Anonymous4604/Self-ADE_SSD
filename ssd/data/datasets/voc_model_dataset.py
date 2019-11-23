import os
import torch.utils.data
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image


class VOCModelDataset(torch.utils.data.Dataset):
    class_names = ('__background__',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self, data_dir, split, desired_classes_subset=None, transform=None, target_transform=None, keep_difficult=False):
        """Dataset for data with structure similar to VOC data.
        Args:
            data_dir: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        self.keep_difficult = keep_difficult

        if desired_classes_subset is None:
            self.keep_all_classes = True
        else:
            self.keep_all_classes = False
            self.subset_class_names = ['__background__']
            self.subset_class_names.extend(desired_classes_subset)
            self.subset_class_dict = {class_name: i for i, class_name in enumerate(self.subset_class_names)}

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

        if not self.keep_all_classes:
            self.class_names = self.subset_class_names

        self.complete_ids_list = []

        if isinstance(self.split, tuple):
            for spl in self.split:
                img_set_split = os.path.join(self.data_dir, "ImageSets", "Main", "%s.txt" % spl)
                self.complete_ids_list.extend(VOCModelDataset._read_image_ids(img_set_split))
        else:
            image_sets_file = os.path.join(self.data_dir, "ImageSets", "Main", "%s.txt" % self.split)
            self.complete_ids_list = VOCModelDataset._read_image_ids(image_sets_file)

        # now we make a list with only the ids of the images containing objects of the subset
        # of the classes we are interested in
        if not self.keep_all_classes:
            self.ids = self._filter_ids()
        else:
            self.ids = self.complete_ids_list

    def _filter_ids(self):
        """
        Iterating on the list of ids self.complete_ids_list we build up
        a new list containing only the ids of the images having at least one
        element of one of the subset of the classes we are interested in
        """
        self.inverted_subset_dict = {}
        for name in self.subset_class_names:
            self.inverted_subset_dict[self.class_dict[name]] = name

        filtered_ids = []

        for image_id in self.complete_ids_list:
            boxes, labels, is_difficult = self._get_annotation(image_id)

            found = False
            for idx, lbl in enumerate(labels):
                if lbl in self.inverted_subset_dict and is_difficult[idx] == 0:
                    found = True
                    break

            if found:
                filtered_ids.append(image_id)

        return filtered_ids

    def _filter_instances(self, boxes, labels, is_difficult):
        """
        removes from a list of labels (and the corresponding list of boxes and the list
        called is_difficult) the labels that are not in the subset of classes we
        are interested in
        """
        nboxes = []
        nlabels = []
        nis_difficult = []
        for idx, lbl in enumerate(labels):
            if lbl in self.inverted_subset_dict:
                nboxes.append(boxes[idx])
                nlabels.append(lbl)
                nis_difficult.append(is_difficult[idx])

        labels = self._translate_labels(nlabels)
        
        return (np.array(nboxes, dtype=np.float32), 
                np.array(labels, dtype=np.int64),
                np.array(nis_difficult, dtype=np.uint8))

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)

        if not self.keep_all_classes:
            # filter instances
            boxes, labels, is_difficult = self._filter_instances(boxes, labels, is_difficult)

        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        
        image = self._read_image(image_id)

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)

        orig_boxes = boxes
        orig_labels = labels

        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)

        # when passing element back we need to make sure that each
        # element has always the same size
        # this is the case for images (3x300x300), for boxes (8732x4), for labels (8732)
        # and needs to be the same for orig_boxes and orig_labels.
        # we have to use a bigger array than the one strictly needed and fill it
        # only in the first positions
        padded_labels = np.zeros(labels.shape)
        padded_boxes = np.zeros(boxes.shape)
        padded_labels[:orig_labels.shape[0]] = orig_labels
        padded_boxes[:orig_boxes.shape[0], :orig_boxes.shape[1]] = orig_boxes

        return image, boxes, labels, padded_boxes, padded_labels

    def get_image(self, index, apply_transform=True):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform and apply_transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        # the function "_get_annotation" does not filter out difficult instances
        # this is ok, because this function ("get_annotation") is called only during
        # evaluation and at test time we effectively want to keep difficult instances

        if not self.keep_all_classes:
            filtered_annotation = self._filter_instances(*(self._get_annotation(image_id)))
        else:
            filtered_annotation = self._get_annotation(image_id)

        return image_id, filtered_annotation

    def __len__(self):
        return len(self.ids)

    def _translate_labels(self, labels):
        """
        this is needed to translate labels defined using the self.class_dict in labels define using
        self.subset_class_dict
        """
        for pos, idc in enumerate(labels):
            class_name = self.inverted_subset_dict[idc]
            idn = self.subset_class_dict[class_name]
            labels[pos] = idn

        return labels

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
        annotation_file = os.path.join(self.data_dir, "Annotations", "%s.xml" % image_id)
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for obj in objects:
            class_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_dict[class_name])
            is_difficult_str = obj.find('difficult').text
            is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, image_id):
        image_file = os.path.join(self.data_dir, "JPEGImages", "%s.jpg" % image_id)
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image
