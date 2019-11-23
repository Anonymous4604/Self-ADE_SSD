import datetime
import logging
import os
import time
import torch
import torch.distributed as dist
import numpy as np
from torch.nn.parallel import DistributedDataParallel

from ssd.engine.inference import do_evaluation
from ssd.utils import distributed_util
from ssd.utils.checkpoint_util import save_training_checkpoint
from ssd.data.datasets import VOCDataset
from ssd.utils.viz import draw_bounding_boxes
from ..transforms.transforms import DeNormalize
from ..transforms.transforms import ToPixelDomain

import tensorflow as tf

try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = distributed_util.get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def _save_model(logger, model, model_path):
    vgg_model = model
    if isinstance(model, DistributedDataParallel):
        vgg_model = model.module
    vgg_model.save(model_path)
    logger.info("Saved checkpoint to {}".format(model_path))


def _scale_back_image(cfg, img):
    orig_image = img.numpy()
    t1 = np.transpose(orig_image, (1, 2, 0))
    transform1 = DeNormalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)
    transform2 = ToPixelDomain()
    orig_image, _, _ = transform2(transform1(t1))
    orig_image = orig_image.astype(np.uint8)
    return orig_image


def do_train(cfg, model,
             data_loader,
             optimizer,
             scheduler,
             device,
             args,
             val_sets_dict=None):
    logger = logging.getLogger("SSD.trainer")
    logger.info("Start training")
    model.train()
    save_to_disk = distributed_util.get_rank() == 0
    if args.use_tensorboard and save_to_disk:
        import tensorboardX

        summary_writer = tensorboardX.SummaryWriter(log_dir=cfg.OUTPUT_DIR)
        tf_writer = tf.compat.v1.summary.FileWriter(cfg.OUTPUT_DIR)
    else:
        summary_writer = None


    if cfg.DATASETS.DG:
        dataloaders = data_loader
        max_iter = len(data_loader[0])
        dataiters = [iter(dataloader) for dataloader in dataloaders]
    else:
        max_iter = len(data_loader)
        data_loader = iter(data_loader)

    start_training_time = time.time()
    trained_time = 0
    tic = time.time()
    end = time.time()

    if args.return_best:
        best_map = 0
    
    for iteration in range(scheduler.last_epoch, max_iter):
        if cfg.DATASETS.DG:
            # domain generalization settings
            # we need to read images from different sources
            images = torch.ones(cfg.SOLVER.BATCH_SIZE * len(dataloaders), 3, cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE)

            for j in range(len(dataloaders)):
                if cfg.MODEL.SELF_SUPERVISED:
                    d_images, d_boxes, d_labels, d_j_images, d_j_index, d_orig_boxes, d_orig_labels = next(dataiters[j])
                else:
                    d_images, d_boxes, d_labels, d_orig_boxes, d_orig_labels = next(dataiters[j])

                start_bs = cfg.SOLVER.BATCH_SIZE * j
                end_bs = start_bs + cfg.SOLVER.BATCH_SIZE

                images[start_bs:end_bs, :, :, :] = d_images

                if j == 0:
                    boxes = d_boxes
                    labels = d_labels
                    orig_boxes = d_orig_boxes
                    orig_labels = d_orig_labels

                    if cfg.MODEL.SELF_SUPERVISED:
                        j_images = d_j_images
                        j_index = d_j_index
                else:
                    boxes = torch.cat((boxes, d_boxes))
                    labels = torch.cat((labels, d_labels))
                    orig_boxes = torch.cat((orig_boxes, d_orig_boxes))
                    orig_labels = torch.cat((orig_labels, d_orig_labels))

                    if cfg.MODEL.SELF_SUPERVISED:
                        j_images = torch.cat((j_images, d_j_images))
                        j_index = torch.cat((j_index, d_j_index))
        else:
            if cfg.MODEL.SELF_SUPERVISED:
                images, boxes, labels, j_images, j_index, orig_boxes, orig_labels = next(data_loader)
            else:
                images, boxes, labels, orig_boxes, orig_labels = next(data_loader)

        # it is not a problem if we increment iteration because it will be reset in the loop
        iteration = iteration + 1

        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss_dict = model(images, targets=(boxes, labels))
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss = sum(loss for loss in loss_dict.values())

        # loss.backward() becomes:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        if cfg.MODEL.SELF_SUPERVISED:
            j_images = j_images.to(device)
            j_index = j_index.to(device)
            loss_dict_j = model(j_images, targets=j_index, auxiliary_task=True)
            loss_dict_reduced_j = reduce_loss_dict(loss_dict_j)
            losses_reduced_j = sum(loss for loss in loss_dict_reduced_j.values())
            loss_j = sum(loss for loss in loss_dict_j.values())
            # apply reduction factor for auxiliary loss
            loss_j = loss_j * cfg.MODEL.SELF_SUPERVISOR.WEIGHT

            # loss.backward() becomes:
            with amp.scale_loss(loss_j, optimizer) as scaled_loss:
                scaled_loss.backward()

            # append this loss to the dictionary of losses
            loss_dict.update(loss_dict_j)
            losses_reduced += losses_reduced_j

        optimizer.step()
        scheduler.step()

        trained_time += time.time() - end
        end = time.time()
        if iteration % args.log_step == 0:
            eta_seconds = int((trained_time / iteration) * (max_iter - iteration))
            log_str = [
                "Iter: {:06d}, Lr: {:.5f}, Cost: {:.2f}s, Eta: {}".format(iteration,
                                                                          optimizer.param_groups[0]['lr'],
                                                                          time.time() - tic,
                                                                          str(datetime.timedelta(seconds=eta_seconds))),
                "total_loss: {:.3f}".format(losses_reduced.item())
            ]
            for loss_name, loss_item in loss_dict_reduced.items():
                log_str.append("{}: {:.3f}".format(loss_name, loss_item.item()))
            log_str = ', '.join(log_str)
            logger.info(log_str)
            if summary_writer:
                global_step = iteration
                summary_writer.add_scalar('losses/total_loss', losses_reduced, global_step=global_step)
                for loss_name, loss_item in loss_dict_reduced.items():
                    summary_writer.add_scalar('losses/{}'.format(loss_name), loss_item, global_step=global_step)
                summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

                if cfg.MODEL.SELF_SUPERVISED:
                    _log_images_tensorboard(cfg, global_step, images, orig_boxes, orig_labels, summary_writer, j_images=j_images)
                else:
                    _log_images_tensorboard(cfg, global_step, images, orig_boxes, orig_labels, summary_writer)
                #for tag, value in model.named_parameters():
                #    tag = tag.replace('.', '/')
                #    if 'ss_classifier' in tag:
                #        print(tag, value)
                #_log_network_params(tf_writer, model, global_step)

            tic = time.time()

        if save_to_disk and iteration % args.save_step == 0:
            model_path = os.path.join(cfg.OUTPUT_DIR,
                                      "ssd{}_vgg_iteration_{:06d}.pth".format(cfg.INPUT.IMAGE_SIZE, iteration))
            save_training_checkpoint(logger, model, scheduler, optimizer, model_path)
        # Do eval when training, to trace the mAP changes and see whether or not performance improved
        # if args.return_best = True the model returned should be the one that gave best performances on the val set
        if args.eval_step > 0 and iteration % args.eval_step == 0 and (not iteration == max_iter or args.return_best):
            dataset_metrics = do_evaluation(cfg, model, cfg.OUTPUT_DIR, distributed=args.distributed, datasets_dict=val_sets_dict)
            
            model.train()

            if args.distributed and not distributed_util.is_main_process():
                continue
            
            avg_map = _compute_avg_map(dataset_metrics)
            
            if args.return_best:

                if avg_map > best_map:
                    best_map = avg_map
                    logger.info("With iteration {} passed the best! New best avg map: {:4f}".format(iteration, best_map))
                    model_path = os.path.join(cfg.OUTPUT_DIR, "ssd{}_vgg_best.pth".format(cfg.INPUT.IMAGE_SIZE))
                    _save_model(logger, model, model_path)
                else:
                    logger.info("With iteration {} the best has not been reached. Best avg map: {:4f}, Current avg mAP: {:4f}".format(iteration, best_map, avg_map))
                
            # logging
            if summary_writer:
                global_step = iteration

                summary_writer.add_scalar("val_avg_map", avg_map, global_step=global_step)

                for dataset_name, metrics in dataset_metrics.items():
                    for metric_name, metric_value in metrics.get_printable_metrics().items():
                        summary_writer.add_scalar('/'.join(['val', dataset_name, metric_name]), metric_value,
                                                  global_step=global_step)

    if save_to_disk:
        model_path = os.path.join(cfg.OUTPUT_DIR, "ssd{}_vgg_final.pth".format(cfg.INPUT.IMAGE_SIZE))
        _save_model(logger, model, model_path)
    # compute training time
    total_training_time = int(time.time() - start_training_time)
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))
    if args.return_best:
        model.load(os.path.join(cfg.OUTPUT_DIR, "ssd{}_vgg_best.pth".format(cfg.INPUT.IMAGE_SIZE)))
    return model


def _compute_avg_map(dataset_metrics):
    count = len(dataset_metrics)
    map_sum = 0
    for k, v in dataset_metrics.items():
        # logger.info("mAP on {}: {:.3f}".format(k, v.info["mAP"]))
        map_sum += v.info["mAP"]
    avg_map = map_sum / count
    return avg_map


def _log_images_tensorboard(cfg, global_step, images, orig_boxes, orig_labels, summary_writer, j_images=None):
    from random import randrange
    size = images.shape[0]
    i = randrange(size)
    imagei = images[i]
    imagei = imagei.cpu()
    labels_i = orig_labels[i]
    labels_i = labels_i.numpy()
    nz_indices_i = (np.nonzero(labels_i))[0]
    boxes_i = orig_boxes[i]
    boxes_i = boxes_i.numpy()
    boxes_i = boxes_i[nz_indices_i]
    labels_i = labels_i[nz_indices_i]
    labels_i = labels_i.astype(int)
    boxes_i = boxes_i * cfg.INPUT.IMAGE_SIZE
    boxes_i = boxes_i.astype(int)
    rescaled = _scale_back_image(cfg, imagei)
    drawn_image = draw_bounding_boxes(rescaled, boxes_i, labels=labels_i,
                                      class_name_map=VOCDataset.class_names).astype(np.uint8)
    drawn_image = np.transpose(drawn_image, (2, 0, 1))
    summary_writer.add_image('imaugmall', drawn_image, global_step=global_step)
    if cfg.MODEL.SELF_SUPERVISED:
        j_imagei = j_images[i]
        j_imagei = j_imagei.cpu()
        rescaled2 = _scale_back_image(cfg, j_imagei)
        drawn_image2 = np.transpose(rescaled2, (2, 0, 1))
        summary_writer.add_image('aux_task_image', drawn_image2, global_step=global_step)


def _log_network_params(writer, model, global_step):
    # (2) Log values and gradients of the parameters (histogram)
    for tag, value in model.named_parameters():
        tag = tag.replace('.', '/')
        _histo_summary(writer, tag, to_np(value), global_step)


def _histo_summary(writer, tag, values, step, bins=1000):
    """Log a histogram of the tensor of values."""

    # Create a histogram using numpy
    counts, bin_edges = np.histogram(values, bins=bins)

    # Fill the fields of the histogram proto
    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values ** 2))

    # Drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)

    # Create and write Summary
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
    writer.add_summary(summary, step)
    writer.flush()


def to_np(x):
    return x.data.cpu().numpy()
