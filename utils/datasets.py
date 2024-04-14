import glob

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset

from utils.general import colorstr
from utils.torch_utils import torch_distributed_zero_first
from utils.dataset_utils import get_chars, image2label_paths

image_formats = ('bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo')


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

        self.image_files, self.label_files, self.shapes = [], [], tuple()
        self.bboxes, self.words, self.gt_words, self.word_mask = [], [], [], []

        for dataset_format, dataset_path in paths.items():
            try:
                dataset_path = Path(dataset_path)
                if dataset_path.is_dir():
                    image_paths = [img_path for img_path in glob.glob(str(dataset_path) + '/*.*')
                                   if img_path.endswith(image_formats)]
                elif dataset_path.is_file():
                    parent = str(dataset_path.parent.absolute()) + os.sep
                    with open(dataset_path, 'r') as mf:
                        image_paths = [img_path.replace('./', parent) if img_path.startswith('./') else img_path
                                       for img_path in mf.read().strip().splitlines()
                                       if img_path.split('.')[-1].lower() in image_formats]
                else:
                    raise Exception(f'({prefix}) Error: {dataset_path} dose not exist.')
                    
                image_paths = sorted([image_path.replace('/', os.sep) for image_path in image_paths])
                assert len(image_paths), f'({prefix}) Error: no images found at dataset {dataset_path}.'
                label_paths = image2label_paths(image_paths, dataset_format)

                cache_path = Path(dataset_path)
                cache_path = (cache_path if cache_path.is_file()
                              else Path(label_paths[0]).parent.parent.parent).with_suffix('.cache')

                if cache_path.exists():
                    cache = torch.load(cache_path)
                    num_found, num_miss, num_empty, num_duplicate = cache.pop('results')
                    desc = f"({prefix}) Scanning {dataset_format} at '{cache_path.parent / cache_path.stem}' ... " \
                           f"{num_found} found, {num_miss} missing, {num_empty} empty, {num_duplicate} corrupted."
                    tqdm(None, desc=desc, total=len(image_paths), initial=len(image_paths))
                    assert num_found > 0, f'({prefix}) Error: No labels in {cache_path}, can not train without labels.'
                else:
                    cache = self.cache_dataset(image_paths, label_paths, cache_path, dataset_format, prefix)
                    cache.pop('results')

    def cache_dataset(self, image_paths, label_paths, cache_path,
                      dataset_format='RTLSTD', prefix='Dataset'):
        labels = {}
        num_found, num_miss, num_empty, num_corrupted = 0, 0, 0, 0
        pbar = tqdm(zip(image_paths, label_paths), desc=f'({prefix}) Scanning {dataset_format}', total=len(image_paths))

        for image_path, label_path in pbar:
            try:
                image = Image.open(image_path)
                image.verify()
                image_size = image.size  # width, height
                try:
                    image_exif = image._getexif().items()
                    image_size = get_exif_size(image_size, image_exif)
                except AttributeError:
                    pass

                assert max(image_size) > 32, f'({prefix}) Error: image size {image_size} < 32 pixels.'
                assert image.format.lower() in image_formats, f'({prefix}) Error: invalid image format {image.format}.'


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
      
