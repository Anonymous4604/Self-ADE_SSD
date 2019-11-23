import argparse
import datetime
import logging
import time

from tqdm import tqdm


import torch


from apex import amp
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler

from ssd.config import cfg
from ssd.utils.checkpoint_util import restore_training_checkpoint
from ssd.modeling.vgg_ssd import build_ssd_model
from ssd.utils.logger import setup_logger
from ssd.data.datasets import build_dataset
from ssd.data.datasets.self_supervised import SelfSupervisedDataset

from ssd.modeling.data_preprocessing import TrainAugmentation
from ssd.modeling.ssd import MatchPrior
from ssd.module.prior_box import PriorBox

from self_ade.os_batch_sampler import OneSampleBatchSampler
from self_ade.incremental_eval import IncrementalEval


def setup_self_ade(cfg, args):
    logger = logging.getLogger("self_ade.setup")
    logger.info("Starting self_ade setup")

    # build model from config
    model = build_ssd_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    train_transform = TrainAugmentation(cfg.INPUT.IMAGE_SIZE, cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)

    target_transform = MatchPrior(PriorBox(cfg)(), cfg.MODEL.CENTER_VARIANCE, cfg.MODEL.SIZE_VARIANCE,
                                  cfg.MODEL.THRESHOLD)

    test_dataset = build_dataset(dataset_list=cfg.DATASETS.TEST, is_test=True)[0]
    self_ade_dataset = build_dataset(dataset_list=cfg.DATASETS.TEST, transform=train_transform, target_transform=target_transform)
    ss_dataset = SelfSupervisedDataset(self_ade_dataset, cfg)

    test_sampler = SequentialSampler(test_dataset)
    os_sampler = OneSampleBatchSampler(test_sampler, cfg.SOLVER.BATCH_SIZE, args.self_ade_iterations)

    self_ade_dataloader = DataLoader(ss_dataset, batch_sampler=os_sampler, num_workers=args.num_workers)

    effective_lr = args.learning_rate * args.self_ade_weight

    optimizer = torch.optim.SGD(model.parameters(), lr=effective_lr, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)

    # Initialize mixed-precision training
    use_mixed_precision = cfg.USE_AMP
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    execute_self_ade(cfg, args, test_dataset, self_ade_dataloader, model, optimizer)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def execute_self_ade(cfg, args, test_dataset, self_ade_dataloader, model, optimizer):
    logger = logging.getLogger("self_ade.execution")
    logger.info('Starting self_ade eval')
    logger.info('Dataset size: Test {}'.format(len(test_dataset)))

    self_ade_breakpoints = (*cfg.MODEL.SELF_SUPERVISOR.SELF_ADE_BREAKPOINTS, args.self_ade_iterations)

    self_ade_dataiter = iter(cycle(self_ade_dataloader))

    start_time = time.time()

    device = torch.device(cfg.MODEL.DEVICE)

    if not args.skip_no_self_ade_eval:
        no_self_ade_eval = IncrementalEval(cfg, test_dataset, "no_self_ade")
    self_ade_eval = [IncrementalEval(cfg, test_dataset, "self_ade_" + str(breakpoint) + "its_" + str(args.learning_rate) + "lr")
                 for breakpoint in self_ade_breakpoints]

    for sample_idx in tqdm(range(len(test_dataset))):
        # load model again
        restore_training_checkpoint(logger, model, args.weights, log_enabled=False)
        if not args.skip_no_self_ade_eval:
            model.eval()
            no_self_ade_eval.add_element(model, sample_idx)
        model.train()

        for b_idx in range(args.self_ade_iterations):
            model.train()
            optimizer.zero_grad()
            _, _, _, j_images, j_index, _, _ = next(self_ade_dataiter)
            j_images = j_images.to(device)
            j_index = j_index.to(device)
            loss_dict_j = model(j_images, targets=j_index, auxiliary_task=True)
            loss_j = sum(loss for loss in loss_dict_j.values())

            if b_idx < args.warmup_steps:
                loss_j = loss_j * ((b_idx + 1) / args.warmup_steps)

            with amp.scale_loss(loss_j, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            if (b_idx + 1) in self_ade_breakpoints:
                model.eval()
                self_ade_eval[self_ade_breakpoints.index(b_idx+1)].add_element(model, sample_idx)

    total_training_time = int(time.time() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total time for self_ade eval: " + total_time_str)

    if not args.skip_no_self_ade_eval:
        logger.info("No self_ade results:")
        no_self_ade_eval.compute_map()

    for breakpoint in self_ade_breakpoints:
        logger.info("self_ade {} results:".format(breakpoint))
        self_ade_eval[self_ade_breakpoints.index(breakpoint)].compute_map()


def main():
    parser = argparse.ArgumentParser(description='self_ade on SSD')
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument('--weights', default=None, type=str, help='Checkpoint state_dict file to use for self_ade')
    parser.add_argument("--self_ade_iterations", default=50, type=int, help="Number of adaptation iterations to perform for each target")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of workers to use for data loaders")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate to be used for adaptation steps")
    parser.add_argument("--self_ade_weight", default=0.8, type=float, help="The weight to be applied to the loss of the self_ade adaptation task")
    parser.add_argument("--warmup_steps", default=20, type=int, help="Steps to linearly increase learning rate from 0 to learning_rate")
    parser.add_argument("--skip_no_self_ade_eval", action='store_true', help="Skips no self_ade evaluation for speed")

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    setup_logger("SSD", 0)
    logger = setup_logger("self_ade", 0)

    logger.info(args)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    logger.info("Loaded configuration file {}".format(args.config_file))

    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    setup_self_ade(cfg, args)


if __name__ == '__main__':
    main()
