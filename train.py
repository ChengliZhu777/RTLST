import yaml
import wandb
import logging
import argparse

from pathlib import Path

from utils.general import set_logging, get_options, get_latest_run, increment_path, \
    check_filepath, colorstr, load_file
from utils.torch_utils import select_torch_device


logger = logging.getLogger(__name__)


def train(opts, evolve_paras=None):
    pass


def wandb_sweep():
    with wandb.init(config=None):
        cur_sweep_config = wandb.config

        evolved_hypers = {}
        for _hyp_name in evolved_hyper_list.keys():
            print(_hyp_name)
            evolved_hypers[_hyp_name] = cur_sweep_config[_hyp_name]

        train(train_options, evolved_hypers)
        

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
        train_options.save_dir = increment_path(Path(train_options.project) / train_options.name,
                                                exist_ok=train_options.exist_ok)
        train_options.hyp = check_filepath(train_options.hyp)

    logger.info(colorstr('Options: ') + str(train_options))

    evolved_hyper_list = None
    if not train_options.evolve:
        train(train_options)
    else:
        wandb.login()
        sweep_config = {'method': 'grid'}

        metric = {'name': 'val_pos', 'goal': 'maximize'}
        sweep_config['metric'] = metric

        evolved_hyper_list = load_file(train_options.evolve_hyp, 'evolve-hyp')

        parameters_dict = {}
        for hyp_name in evolved_hyper_list.keys():
            parameters_dict[hyp_name] = {'values': evolved_hyper_list[hyp_name]}

        sweep_config['parameters'] = parameters_dict
        sweep_id = wandb.sweep(sweep_config, project=Path(train_options.save_dir).name)

        wandb.agent(sweep_id, wandb_sweep)
        
