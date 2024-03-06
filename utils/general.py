import logging
import argparse


def set_logging(rank=-1):
    """
    %(message)s : output log content
    %(asctime)s : output logging time
    """
    logging.basicConfig(format='%(message)s',
                        level=logging.INFO if rank in [-1, 0] else logging.WARN)


def get_options(hyp_path='hyp/hyp.base.yaml',
                data_path='data/rtls.yaml',
                model_cfg='models/cfg/rtlstd.yaml',
                weights=None, resume=False,
                evolve=False, visible=False,
                exist_ok=False, project='RTLSTD-Runs', name='Train'):

    parser = argparse.ArgumentParser(description='RTLSTD Train/Test Options')
    parser.add_argument('--hyp', type=str, default=hyp_path, help='Required hyper-parameters, i.e, patch-size.')
    parser.add_argument('--data', type=str, default=data_path, help='Train/Test dataset paths.')
    parser.add_argument('--model-cfg', type=str, default=model_cfg, help='RTLSTD architecture.')
    parser.add_argument('--weights', type=str, default=weights, help='Trained weight path.')
    parser.add_argument('--resume', action='store_true', default=resume, help='Resume most recent training results.')
    parser.add_argument('--evolve', action='store_true', default=evolve, help='Evolve specified hyper-parameters.')
    parser.add_argument('--visible', action='store_true', default=visible, help='Visualizing detecting results.')
    parser.add_argument('--exist-ok', action='store_true', default=exist_ok,
                        help='Allow the directory project/name existing, and use it to run this program.')
    parser.add_argument('--project', type=str, default=project, help='save to project/name')
    parser.add_argument('--name', type=str, default=name, help='save to project/name')

    return parser.parse_args()

