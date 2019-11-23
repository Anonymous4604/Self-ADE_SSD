from torch._six import int_classes as _int_classes

from torch.utils.data.sampler import Sampler


class OneSampleBatchSampler(Sampler):

    def __init__(self, sampler, batch_size, onesample_iterations, drop_last=False):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.onesample_iterations = onesample_iterations
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        for idx in self.sampler:
            for _ in range(self.onesample_iterations):
                yield [idx] * self.batch_size

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

