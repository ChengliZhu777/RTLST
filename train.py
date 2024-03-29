import yaml
import wandb
import logging
import argparse

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from utils.general import set_logging, get_options, get_latest_run, increment_path, \
    check_filepath, colorstr, load_file, check_image_size
from utils.torch_utils import select_torch_device, init_seeds


logger = logging.getLogger(__name__)
init_seeds(666)


def train(opts, evolved_hypers=None):

    hypers, model_cfg, data_list = load_file(opts.hyp), load_file(opts.model_cfg), load_file(opts.data)
    save_dir = Path(opts.save_dir)
    if evolved_hypers is not None:
        evolve_cfg = ''
        for para_name in evolved_hypers.keys():
            if para_name in ['patch_size']:
                model_cfg['model'][para_name] = evolved_hypers[para_name]
            else:
                hypers[para_name] = evolved_hypers[para_name]
            evolve_cfg += (str(evolved_hypers[para_name]) + '_')
        save_dir = save_dir / evolve_cfg[:-1]

    save_dir.mkdir(parents=True, exist_ok=False)

    weight_dir = save_dir / 'weights'
    weight_dir.mkdir(parents=True, exist_ok=False)
    best_weight = weight_dir / 'best.pth'

    epochs, batch_size, read_type = hypers['epochs'], hypers['batch_size'], hypers['read_type']
    patch_size, is_recognize = model_cfg['model']['patch_size'], model_cfg['model']['is_recognize']

    logger.info(colorstr('hyper-parameters: ') + ', '.join(f'{k}={v}' for k, v in hypers.items()))
    device = select_torch_device(hypers['device'], batch_size=['batch_size'], prefix=colorstr('Device: '))

    logger.info(f"{colorstr('Tensorboard: ')}Start with 'tensorboard --logdir {opts.save_dir}',"
                f"view at http://localhost:6006/")
    tb_writer = SummaryWriter(str(opts.save_dir))

    results_writer = open(save_dir / 'results.txt', 'a')

    with open(save_dir / 'opt.yaml', 'w') as f1:
        yaml.dump(vars(opts), f1, sort_keys=False)
    with open(save_dir / 'hyp.yaml', 'w') as f1:
        yaml.dump(hypers, f1, sort_keys=False)

    train_data_paths, valid_data_paths = data_list['train'], data_list['valid']
    train_long_size, train_short_size = [check_image_size(size, patch_size) for size in hypers['image_size']]


if __name__ == '__main__':
    set_logging(-1)

    train_options = get_options(evolve=True,
                                name='train-1',
                                exist_ok=False)

    if train_options.resume and not train_options.evolve:
        train_ckpt = train_options.resume if isinstance(train_options.resume, str) else get_latest_run()
        with open(Path(train_ckpt).parent.parent / 'opt.yaml') as f:
            train_options = argparse.ArgumentParser(**yaml.load(f, Loader=yaml.SafeLoader))
        train_options.weights, train_options.hyp, train_options.resume = \
            train_ckpt, str(Path(train_ckpt).parent.parent / 'hyp.yaml'), True
    else:
        train_options.hyp = check_filepath(train_options.hyp)
        train_options.model_cfg = check_filepath(train_options.model_cfg)
        train_options.evolve_hyp = check_filepath(train_options.evolve_hyp)
        train_options.save_dir = increment_path(Path(train_options.project) / train_options.name,
                                                exist_ok=train_options.exist_ok)

    evolved_hyper_list = None
    if not train_options.evolve:
        train(train_options)
    else:
        wandb.login()
        sweep_config = {'method': 'grid'}

        metric = {'name': 'val_pos', 'goal': 'maximize'}
        sweep_config['metric'] = metric

        evolved_hyper_list = load_file(train_options.evolve_hyp)

        parameters_dict = {}
        for hyp_name in evolved_hyper_list.keys():
            parameters_dict[hyp_name] = {'values': evolved_hyper_list[hyp_name]}

        sweep_config['parameters'] = parameters_dict
        sweep_id = wandb.sweep(sweep_config, project=Path(train_options.save_dir).name)

        wandb.agent(sweep_id, wandb_sweep)
        
