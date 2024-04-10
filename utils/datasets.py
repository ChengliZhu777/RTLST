
from torch.utils.data import Dataset

from utils.torch_utils import torch_distributed_zero_first


class LoadImageAndLabels(Dataset):
    def __init__(self, paths, hypers, long_size=1280, short_size=768, patch_size=64, is_train=False,
                 is_augment=False, is_recognize=False, is_visible=False, read_type='pil', prefix='Dataset'):
        if is_augment:
            self.random_scale_ratio = np.array(hypers['random_scale_ratio'])
            self.random_scale_aspect = np.array(hypers['random_scale_aspect'])
            self.random_rotate_angle = np.array(hypers['random_rotate_angle'])
        self.patch_thresh = hypers['patch_thresh']
        self.kernel_scale = hypers['kernel_scale']

        self.long_size, self.short_size, self.patch_size = long_size, short_size, patch_size
        self.is_train, self.is_augment, self.is_recognize, self.is_visible = \
            is_train, is_augment, is_recognize, is_visible
        self.read_type = read_type

        self.max_label_num, self.max_label_len = 200, 32
        self.chars, self.char2id, self.id2char = get_chars('lowercase')


def create_dataloader(paths, hypers, long_size, short_size, patch_size, batch_size,
                      is_train=False, is_augment=False, is_recognize=False, is_visible=False,
                      rank=-1, read_type='pil', prefix='Dataset'):
    with torch_distributed_zero_first(rank):
        dataset = LoadImageAndLabels(paths=paths, hypers=hypers,
                                     long_size=long_size, short_size=short_size,
                                     patch_size=patch_size, batch_size=batch_size,
                                     is_train=is_train, is_augment=is_augment,
                                     is_recognize=is_recognize, is_visible=is_visible,
                                     read_type=read_type, prefix=prefix)
      
