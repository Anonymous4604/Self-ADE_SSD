import torch
import os

from ssd.data.datasets.evaluation import evaluate
from ssd.modeling.predictor import Predictor


class IncrementalEval(object):
    """
    Necessary to compute the mAP of a model on a certain dataset when the predictions are not executed all together
    """

    def __init__(self, cfg, test_dataset, output_name):
        main_output_dir = cfg.OUTPUT_DIR
        self.output_dir = os.path.join(main_output_dir, output_name)
        if not os.path.exists(self.output_dir):
            try:
                os.makedirs(self.output_dir)
            except:
                logger = logging.getLogger("self_ade.eval")
                logger.info("Output dir {} exists".format(self.output_dir))

        self.test_dataset = test_dataset
        self.predictions = {}
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.cpu_device = torch.device("cpu")
        self.predictor = Predictor(cfg=cfg, iou_threshold=cfg.TEST.NMS_THRESHOLD, score_threshold=cfg.TEST.CONFIDENCE_THRESHOLD, device=self.device)

    def add_element(self, model, sample_idx):
        """
        Compute a prediction using a certain model on the image at position sample_idx in the dataset
        :param model: to be used to perform prediction
        :param sample_idx: to identify the image
        """
        image = self.test_dataset.get_image(sample_idx, apply_transform=False)
        output = self.predictor.predict(image, model=model)
        boxes, labels, scores = [o.to(self.cpu_device).numpy() for o in output]
        self.predictions[sample_idx] = (boxes, labels, scores)

    def compute_map(self):
        image_ids = list(sorted(self.predictions.keys()))
        predictions = [self.predictions[i] for i in image_ids]
        return evaluate(dataset=self.test_dataset, predictions=predictions, output_dir=self.output_dir)
