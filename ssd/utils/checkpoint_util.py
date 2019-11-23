from torch.nn.parallel import DistributedDataParallel
import torch
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')

def save_training_checkpoint(logger, model, scheduler, optimizer, model_path):
    vgg_model = model
    if isinstance(model, DistributedDataParallel):
        vgg_model = model.module
    torch.save({
        'model_state_dict': vgg_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'amp': amp.state_dict()
    }, model_path)
    logger.info("Saved checkpoint to {}. Last iteration: {}".format(model_path, scheduler.last_epoch))


def restore_training_checkpoint(logger, model, model_path, optimizer=None, scheduler=None, log_enabled=True):
    checkpoint = torch.load(model_path)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        return
   
    amp.load_state_dict(checkpoint['amp'])

    if log_enabled:
        logger.info("Restored model from checkpoint {}.".format(model_path))

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if log_enabled:
            logger.info("Resuming training from iteration: {}".format(scheduler.last_epoch))
