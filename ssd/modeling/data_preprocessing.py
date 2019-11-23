from ..transforms.transforms import *

class TrainAugmentation:
    def __init__(self, size, mean=0, std=1):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            PhotometricDistort(),
            #SSDColorJitter(0.5, 0.5, 0.5, 0),
            Expand(self.mean),
            RandomSampleCrop(),
            # random horizontal mirroring (probability 0.5)
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean),
            #ToNormalDomain(),
            #Normalize(mean, std),
            ToTensor(),
        ])

    def __call__(self, img, boxes, labels):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.augment(img, boxes, labels)

class PredictionTransform:
    def __init__(self, size, mean=0.0, std=1):
        self.transform = Compose([
            Resize(size),
            SubtractMeans(mean),
            #ToNormalDomain(),
            #Normalize(mean, std),
            ToTensor()
        ])

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image
