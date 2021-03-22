import torch as th


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = th.FloatTensor(mean).view(1, 3, 1, 1)
        self.std = th.FloatTensor(std).view(1, 3, 1, 1)

    def __call__(self, tensor):
        tensor = (tensor - self.mean) / (self.std + 1e-8)
        return tensor


class Preprocessing(object):
    def __init__(self, num_frames=16):
        self.norm = Normalize(mean=[110.6, 103.2, 96.3], std=[1.0, 1.0, 1.0])
        self.num_frames = num_frames

    def _zero_pad(self, tensor, size):
        n = size - len(tensor) % size
        if n == size:
            return tensor
        else:
            z = th.zeros(n, tensor.shape[1], tensor.shape[2], tensor.shape[3])
            return th.cat((tensor, z), 0)

    def __call__(self, tensor):
        tensor = tensor[: len(tensor) - len(tensor) % self.num_frames]
        tensor = tensor / 255.0
        tensor = tensor.view(
            -1, self.num_frames, tensor.shape[1], tensor.shape[2], tensor.shape[3]
        )
        tensor = tensor.transpose(1, 2)
        return tensor
