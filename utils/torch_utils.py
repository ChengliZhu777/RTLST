import torch
import logging

logger = logging.getLogger(__name__)


def select_torch_device(device='', batch_size=None, prefix=''):
    info = f'{prefix}torch {torch.__version__}'
    is_cpu = device.lower() == 'cpu'

    if is_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available(), f"Error: cuda unavailable, invalid device '{device}' requested."
    else:
        raise ValueError("Error: torch device must be specified, e.g, 'cpu', '0', or '0, 1, 2'.")
      

def init_torch_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = False, True  # using default conv, more reproducible


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)
    
