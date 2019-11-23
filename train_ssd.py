import argparse
import logging
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from ssd.config import cfg
from ssd.data import samplers
from ssd.data.datasets import build_dataset
from ssd.data.datasets.self_supervised import SelfSupervisedDataset
from ssd.engine.inference import do_evaluation
from ssd.engine.trainer import do_train
from ssd.modeling.data_preprocessing import TrainAugmentation
from ssd.modeling.ssd import MatchPrior
from ssd.modeling.vgg_ssd import build_ssd_model
from ssd.module.prior_box import PriorBox
from ssd.utils import distributed_util
from ssd.utils.logger import setup_logger
from ssd.utils.lr_scheduler import WarmupMultiStepLR
from ssd.utils.misc import str2bool
from ssd.utils.checkpoint_util import restore_training_checkpoint

try:
    # Capirca uses Google's abseil-py library, which uses a Google-specific
    # wrapper for logging. That wrapper will write a warning to sys.stderr if
    # the Google command-line flags library has not been initialized.
    #
    # https://github.com/abseil/abseil-py/blob/pypi-v0.7.1/absl/logging/__init__.py#L819-L825
    #
    # This is not right behavior for Python code that is invoked outside of a
    # Google-authored main program. Use knowledge of abseil-py to disable that
    # warning; ignore and continue if something goes wrong.
    import absl.logging

    # https://github.com/abseil/abseil-py/issues/99
    logging.root.removeHandler(absl.logging._absl_handler)
    # https://github.com/abseil/abseil-py/issues/102
    absl.logging._warn_preinit_stderr = False
except Exception:
    pass

try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')


def train(cfg, args):
    logger = logging.getLogger('SSD.trainer')
    # -----------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------
    model = build_ssd_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    # -----------------------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------------------
    lr = cfg.SOLVER.LR * args.num_gpus  # scale by num gpus
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg.SOLVER.MOMENTUM,
                                weight_decay=cfg.SOLVER.WEIGHT_DECAY)

    # -----------------------------------------------------------------------------
    # Scheduler
    # -----------------------------------------------------------------------------
    milestones = [step // args.num_gpus for step in cfg.SOLVER.LR_STEPS]
    scheduler = WarmupMultiStepLR(optimizer=optimizer,
                                  milestones=milestones,
                                  gamma=cfg.SOLVER.GAMMA,
                                  warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
                                  warmup_iters=cfg.SOLVER.WARMUP_ITERS)

    # -----------------------------------------------------------------------------
    # Load weights or restore checkpoint
    # -----------------------------------------------------------------------------
    if args.resume:
        logger.info("Resume from the model {}".format(args.resume))
        restore_training_checkpoint(logger, model, args.resume, optimizer=optimizer, scheduler=scheduler)
    else:
        logger.info("Init from base net {}".format(args.vgg))
        model.init_from_base_net(args.vgg)
    
    # Initialize mixed-precision training
    use_mixed_precision = cfg.USE_AMP
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,find_unused_parameters=True)
   
    # -----------------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------------
    train_transform = TrainAugmentation(cfg.INPUT.IMAGE_SIZE, cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)
    target_transform = MatchPrior(PriorBox(cfg)(), cfg.MODEL.CENTER_VARIANCE, cfg.MODEL.SIZE_VARIANCE,
                                  cfg.MODEL.THRESHOLD)

    if cfg.DATASETS.DG:
        if args.eval_mode == "val":
            dslist, val_set_dict = _create_dg_datasets(args, cfg, logger, target_transform, train_transform)
        else:
            dslist = _create_dg_datasets(args, cfg, logger, target_transform, train_transform)

        logger.info("Sizes of sources datasets:")
        for k, v in dslist.items():
            logger.info("{} size: {}".format(k, len(v)))

        dataloaders = []
        for name, train_dataset in dslist.items():
            sampler = torch.utils.data.RandomSampler(train_dataset)
            batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=sampler, batch_size=cfg.SOLVER.BATCH_SIZE,
                                                                  drop_last=True)

            batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, num_iterations=cfg.SOLVER.MAX_ITER)

            if cfg.MODEL.SELF_SUPERVISED:
                ss_dataset = SelfSupervisedDataset(train_dataset, cfg)
                train_loader = DataLoader(ss_dataset, num_workers=args.num_workers, batch_sampler=batch_sampler, pin_memory=True)
            else:
                train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_sampler=batch_sampler, pin_memory=True)
            dataloaders.append(train_loader)

        if args.eval_mode == "val":
            if args.return_best:
                return do_train(cfg, model, dataloaders, optimizer, scheduler, device, args, val_set_dict)
            else:
                return do_train(cfg, model, dataloaders, optimizer, scheduler, device, args)
        else:
            return do_train(cfg, model, dataloaders, optimizer, scheduler, device, args)

    # No DG:
    if args.eval_mode == "val":
        train_dataset, val_dataset = build_dataset(dataset_list=cfg.DATASETS.TRAIN, transform=train_transform,
                                      target_transform=target_transform,split=True)
    else:
        train_dataset = build_dataset(dataset_list=cfg.DATASETS.TRAIN, transform=train_transform,
                                  target_transform=target_transform)
    logger.info("Train dataset size: {}".format(len(train_dataset)))
    if args.distributed:
        sampler = torch.utils.data.DistributedSampler(train_dataset)
    else:
        sampler = torch.utils.data.RandomSampler(train_dataset)
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=sampler, batch_size=cfg.SOLVER.BATCH_SIZE,
                                                          drop_last=False)
    batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler,
                                                        num_iterations=cfg.SOLVER.MAX_ITER // args.num_gpus)

    if cfg.MODEL.SELF_SUPERVISED:
        ss_dataset = SelfSupervisedDataset(train_dataset, cfg)
        train_loader = DataLoader(ss_dataset, num_workers=args.num_workers, batch_sampler=batch_sampler, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_sampler=batch_sampler, pin_memory=True)

    if args.eval_mode == "val":
        return do_train(cfg, model, train_loader, optimizer, scheduler, device, args, {"validation_split": val_dataset})
    else:
        return do_train(cfg, model, train_loader, optimizer, scheduler, device, args)


def _create_dg_datasets(args, cfg, logger, target_transform, train_transform):
    dslist = {}
    if args.eval_mode == "val":
        val_set_dict = {}
        default_domain_dataset, default_domain_val_set = build_dataset(
            dataset_list=cfg.DATASETS.DG_SETTINGS.DEFAULT_DOMAIN,
            transform=train_transform, target_transform=target_transform, split=True)
        val_set_dict["Default domain"] = default_domain_val_set
        logger.info("Default domain: train split has {} elements, test split has {} elements".format(
            len(default_domain_dataset), len(default_domain_val_set)))
    else:
        default_domain_dataset = build_dataset(dataset_list=cfg.DATASETS.DG_SETTINGS.DEFAULT_DOMAIN,
                                               transform=train_transform, target_transform=target_transform)
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

    if args.eval_mode == "val":
        return dslist, val_set_dict
    else:
        return dslist


def main():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With PyTorch')
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--vgg',
                        help='Pre-trained vgg model path, download from https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth')
    parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--log_step', default=50, type=int, help='Print logs every log_step')
    parser.add_argument('--save_step', default=5000, type=int, help='Save checkpoint every save_step')
    parser.add_argument('--eval_step', default=0, type=int,
                        help='Evaluate dataset every eval_step, disabled when eval_step <= 0. Default: disabled')
    parser.add_argument('--use_tensorboard', default=True, type=str2bool)
    parser.add_argument("--num_workers", default=4, type=int, help="Number of workers to use for data loaders")
    parser.add_argument("--eval_mode", default="test", type=str,
                        help='Use defined test datasets for periodic evaluation or use a validation split. Default: "test", alternative "val"')
    parser.add_argument("--return_best", default=False, type=str2bool,
                        help="If false (default) tests on the target the last model. If true tests on the target the model with the best performance on the validation set")
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    args.num_gpus = num_gpus

    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    logger = setup_logger("SSD", distributed_util.get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if not os.path.exists(cfg.OUTPUT_DIR):
        if not args.distributed or (args.distributed and distributed_util.is_main_process()):
            os.makedirs(cfg.OUTPUT_DIR)

    model = train(cfg, args)

    if not args.skip_test:
        logger.info('Start evaluating...')
        torch.cuda.empty_cache()  # speed up evaluating after training finished
        do_evaluation(cfg, model, cfg.OUTPUT_DIR, distributed=args.distributed)


if __name__ == '__main__':
    main()
