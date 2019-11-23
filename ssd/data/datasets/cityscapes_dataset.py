import os
import torch.utils.data
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image


class CityscapesDataset(torch.utils.data.Dataset):
    class_names = ('__background__',
                   'person', 'rider', 'car', 'truck',
                   'bus', 'train', 'motorcycle', 'bicycle')

    def __init__(self, data_dir, split, transform=None, target_transform=None, keep_difficult=False):
        """Dataset for VOC data.
        Args:
            data_dir: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.images_rootdir = os.path.join(self.data_dir, 'leftImg8bit_trainvaltest', 'leftImg8bit', split)
        self.annotations_rootdir = os.path.join(self.data_dir, 'gtFine_trainvaltest', 'gtFine', split)
        self.filenames, self.annotations = self._retrieve_paths(self.images_rootdir, self.annotations_rootdir)

        image_sets_file = os.path.join(self.data_dir, "ImageSets", "Main", "%s.txt" % self.split)
        self.ids = VOCDataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def _retrieve_paths(images_rootdir, annotations_rootdir):
        filenames, annotations = [], []
        for dir in os.listdir(images_rootdir):
            city_folder = os.path.join(images_rootdir, dir)
            image_files = os.listdir(city_folder)
            annotation_files = [image.split.('_')[:-1] for image in image_files]
            for af in annotation_files:
                af.append('gtFine_polygons.json')
            annotation_files = ['_'.join(af) for af in annotation_files]

            image_files = [os.path.join(city_folder, im_f) for im_f in image_files]
            annotation_files = [os.path.join(annotations_rootdir, dir, af) for af in annotation_files]

            filenames.extend(image_files)
            annotations.extend(annotation_files)
        return filenames, annotations

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)
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

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

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
