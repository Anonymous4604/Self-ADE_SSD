import os


class DatasetCatalog:
    DATA_DIR = 'datasets'
    DATASETS = {
        'voc_2007_train': {
            "data_dir": "VOC2007",
            "split": "train"
        },
        'voc_2007_val': {
            "data_dir": "VOC2007",
            "split": "val"
        },
        'voc_2007_trainval': {
            "data_dir": "VOC2007",
            "split": "trainval"
        },
        'voc_2007_trainval_subset': {
            "data_dir": "VOC2007",
            "split": "trainval",
            "desired_classes_subset": ('bicycle', 'bird', 'car', 'cat', 'dog', 'person')
        },
        'voc_2007_test': {
            "data_dir": "VOC2007",
            "split": "test"
        },
        'voc_2007_test_subset': {
            "data_dir": "VOC2007",
            "split": "test",
            "desired_classes_subset": ('bicycle', 'bird', 'car', 'cat', 'dog', 'person')
        },
        'voc_2012_train': {
            "data_dir": "VOC2012",
            "split": "train"
        },
        'voc_2012_val': {
            "data_dir": "VOC2012",
            "split": "val"
        },
        'voc_2012_trainval': {
            "data_dir": "VOC2012",
            "split": "trainval"
        },
        'voc_2012_trainval_subset': {
            "data_dir": "VOC2012",
            "split": "trainval",
            "desired_classes_subset": ('bicycle', 'bird', 'car', 'cat', 'dog', 'person')
        },
        'voc_2012_test': {
            "data_dir": "VOC2012",
            "split": "test"
        },
        'coco_2014_valminusminival': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_valminusminival2014.json"
        },
        'coco_2014_minival': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_minival2014.json"
        },
        'coco_2014_train': {
            "data_dir": "train2014",
            "ann_file": "annotations/instances_train2014.json"
        },
        'coco_2014_val': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_val2014.json"
        },
        'clipart_train_subset': {
            "data_dir": "clipart",
            "split": "train",
            "desired_classes_subset": ('bicycle', 'bird', 'car', 'cat', 'dog', 'person')
        },
        'clipart_test_subset': {
            "data_dir": "clipart",
            "split": "test",
            "desired_classes_subset": ('bicycle', 'bird', 'car', 'cat', 'dog', 'person')
        },
        'clipart_test_train_subset': {
            "data_dir": "clipart",
            "split": ("test", "train"),
            "desired_classes_subset": ('bicycle', 'bird', 'car', 'cat', 'dog', 'person')
        }
        ,'clipart_test': {
            "data_dir": "clipart",
            "split": "test"
        }
        ,'clipart_test_train': {
            "data_dir": "clipart",
            "split": ("test", "train"),
        },
        'watercolor_train_subset': {
            "data_dir": "watercolor",
            "split": "train",
            "desired_classes_subset": ('bicycle', 'bird', 'car', 'cat', 'dog', 'person')
        },
        'watercolor_test_subset': {
            "data_dir": "watercolor",
            "split": "test",
            "desired_classes_subset": ('bicycle', 'bird', 'car', 'cat', 'dog', 'person')
        },
        'watercolor_test_train_subset': {
            "data_dir": "watercolor",
            "split": "instance_level_annotated",
            "desired_classes_subset": ('bicycle', 'bird', 'car', 'cat', 'dog', 'person')
        },
        'watercolor_test': {
            "data_dir": "watercolor",
            "split": "test"
        },
        'comic_train_subset': {
            "data_dir": "comic",
            "split": "train",
            "desired_classes_subset": ('bicycle', 'bird', 'car', 'cat', 'dog', 'person')
        },
        'comic_test_subset': {
            "data_dir": "comic",
            "split": "test",
            "desired_classes_subset": ('bicycle', 'bird', 'car', 'cat', 'dog', 'person')
        },
        'comic_test_train_subset': {
            "data_dir": "comic",
            "split": "instance_level_annotated",
            "desired_classes_subset": ('bicycle', 'bird', 'car', 'cat', 'dog', 'person')
        },
        'comic_test': {
            "data_dir": "comic",
            "split": "test"
        },
    }

    @staticmethod
    def get(name):
        if "subset" in name:

            root_dir = DatasetCatalog.DATA_DIR

            if "voc_2007" in name:
                if 'VOC07_ROOT' in os.environ:
                    root_dir = os.environ['VOC07_ROOT']
            elif "voc_2012" in name:
                if 'VOC12_ROOT' in os.environ:
                    root_dir = os.environ['VOC12_ROOT']
            elif "clipart" in name:
                if 'CLIPART_ROOT' in os.environ:
                    root_dir = os.environ['CLIPART_ROOT']
            elif "watercolor" in name:
                if 'WATERCOLOR_ROOT' in os.environ:
                    root_dir = os.environ['WATERCOLOR_ROOT']
            elif "comic" in name:
                if 'COMIC_ROOT' in os.environ:
                    root_dir = os.environ['COMIC_ROOT']

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(root_dir, attrs["data_dir"]),
                split=attrs["split"],
                desired_classes_subset=attrs["desired_classes_subset"]
            )
            return dict(factory="VOCModelDataset", args=args)

        elif "voc" in name or "clipart" or "comic" in name or "watercolor" in name:
             voc_root = DatasetCatalog.DATA_DIR
             if 'VOC_ROOT' in os.environ:
                 voc_root = os.environ['VOC_ROOT']
        
             attrs = DatasetCatalog.DATASETS[name]
             args = dict(
                 data_dir=os.path.join(voc_root, attrs["data_dir"]),
                 split=attrs["split"],
             )
             return dict(factory="VOCModelDataset", args=args)
        elif "coco" in name:
            coco_root = DatasetCatalog.DATA_DIR
            if 'COCO_ROOT' in os.environ:
                coco_root = os.environ['COCO_ROOT']

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(coco_root, attrs["data_dir"]),
                ann_file=os.path.join(coco_root, attrs["ann_file"]),
            )
            return dict(factory="COCODataset", args=args)

        raise RuntimeError("Dataset not available: {}".format(name))
