import argparse
import logging
import os

import torch
import torch.utils.data

from ssd.config import cfg
from ssd.engine.inference import do_evaluation
from ssd.modeling.vgg_ssd import build_ssd_model
from ssd.utils import distributed_util
from ssd.utils.logger import setup_logger
from ssd.data.datasets import build_dataset
from ssd.modeling.data_preprocessing import TrainAugmentation
from ssd.modeling.ssd import MatchPrior
from ssd.module.prior_box import PriorBox


def _create_val_datasets(args, cfg, logger):
    dslist = {}
    val_set_dict = {}

    train_transform = TrainAugmentation(cfg.INPUT.IMAGE_SIZE, cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)
    target_transform = MatchPrior(PriorBox(cfg)(), cfg.MODEL.CENTER_VARIANCE, cfg.MODEL.SIZE_VARIANCE,
                                  cfg.MODEL.THRESHOLD)

    default_domain_dataset, default_domain_val_set = build_dataset(
        dataset_list=cfg.DATASETS.DG_SETTINGS.DEFAULT_DOMAIN,
        transform=train_transform, target_transform=target_transform, split=True)
    val_set_dict["Default domain"] = default_domain_val_set
    logger.info("Default domain: train split has {} elements, test split has {} elements".format(
        len(default_domain_dataset), len(default_domain_val_set)))

    dslist["Default domain"] = default_domain_dataset
    for element in cfg.DATASETS.DG_SETTINGS.SOURCE_DOMAINS:
        if not isinstance(element, tuple):
            sets = (element,)
        else:
            sets = element

        if args.eval_mode == "val":
            ds, val_set = build_dataset(dataset_list=sets, transform=train_transform,
                                        target_transform=target_transform, split=True)
            val_set_dict[element] = val_set
            logger.info(
                "Domain {}: train split has {} elements, test split has {} elements".format(str(element), len(ds),
                                                                                            len(val_set)))
        else:
            ds = build_dataset(dataset_list=sets, transform=train_transform,
                               target_transform=target_transform)

        dslist[element] = ds

    return val_set_dict


def evaluation(cfg, args, weights_file, output_dir, distributed):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    device = torch.device(cfg.MODEL.DEVICE)
    model = build_ssd_model(cfg)
    model.load(weights_file)
    logger = logging.getLogger("SSD.inference")
    logger.info('Loaded weights from {}.'.format(weights_file))
    model.to(device)

    if args.eval_mode == "test":
        do_evaluation(cfg, model, output_dir, distributed)
    else:
        dataset_metrics = do_evaluation(cfg, model, cfg.OUTPUT_DIR, distributed, datasets_dict=_create_val_datasets(args, cfg, logger))
        count = len(dataset_metrics)
        map_sum = 0
        for k,v in dataset_metrics.items():
            #logger.info("mAP on {}: {:.3f}".format(k, v.info["mAP"]))
            map_sum += v.info["mAP"]

        avg_map = map_sum/count
        print("'Model': '{}', 'Avg_mAP': {}".format(weights_file, avg_map))

def main():
    parser = argparse.ArgumentParser(description='SSD Evaluation on VOC and COCO dataset.')
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--weights", type=str, help="Trained weights.")
    parser.add_argument("--output_dir", default="eval_results", type=str, help="The directory to store evaluation results.")
    parser.add_argument("--eval_mode", default="test", type=str,
                        help='Use defined test datasets for final evaluation or use a validation split. Default: "test", alternative "val"')

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = setup_logger("SSD", distributed_util.get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    evaluation(cfg, args, weights_file=args.weights, output_dir=args.output_dir, distributed=distributed)


if __name__ == '__main__':
    main()
