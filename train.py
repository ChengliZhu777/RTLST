import yaml

from pathlib import Path

from utils.general import set_logging, get_options, get_latest_run, increment_path


if __name__ == '__main__':
    set_logging(-1)
  
    train_options = get_options(evolve=True,
                                name='train',
                                exist_ok=False)

    if train_options.resume and not train_options.evolve:
        train_ckpt = train_options.resume if isinstance(train_options.resume, str) else get_latest_run()
        with open(Path(train_ckpt).parent.parent / 'opt.yaml') as f:
            train_options = argparse.ArgumentParser(**yaml.load(f, Loader=yaml.SafeLoader))
        train_options.weights, train_ckpt.hyp, train_options.resume = \
            train_ckpt, str(Path(train_ckpt).parent.parent / 'opt.yaml'), True
    else:
        train_options.save_dir = increment_path(Path(train_options.project) / train_options.name,
                                                exist_ok=train_options.exist_ok)

