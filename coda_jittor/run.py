"""Command-line entry point for the Jittor CODA-Prompt reproduction."""
from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Any, Dict, List

import jittor as jt
import numpy as np
import yaml

from .trainer import Trainer


def create_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0], help="List of GPU ids; negative for CPU")
    parser.add_argument('--log_dir', type=str, default="outputs/out", help="Directory to store logs and checkpoints")
    parser.add_argument('--learner_type', type=str, default='prompt', help="Learner module name inside coda_jittor.learners")
    parser.add_argument('--learner_name', type=str, default='PromptLearner', help="Learner class name")
    parser.add_argument('--debug_mode', type=int, default=0, help="Enable debug toggles inside learners")
    parser.add_argument('--repeat', type=int, default=1, help="Number of independent repetitions")
    parser.add_argument('--overwrite', type=int, default=0, help="Ignore cached runs if set")

    parser.add_argument('--oracle_flag', default=False, action='store_true', help="Activate oracle training regime")
    parser.add_argument('--upper_bound_flag', default=False, action='store_true', help="Upper bound mode trains on full label space")
    parser.add_argument('--memory', type=int, default=0, help="Memory size for rehearsal")
    parser.add_argument('--temp', type=float, default=2.0, help="Distillation temperature")
    parser.add_argument('--DW', default=False, action='store_true', help="Dataset balancing flag")
    parser.add_argument('--batch_size', type=int, default=128, help="Override training batch size")
    parser.add_argument('--prompt_param', nargs="+", type=float, default=[1, 1, 1], help="Prompt hyper-parameters")

    parser.add_argument('--config', type=str, default="configs/config.yaml", help="YAML configuration path")
    return parser


def get_args(argv: List[str]) -> argparse.Namespace:
    parser = create_args()
    args = parser.parse_args(argv)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    cli_args = vars(args)
    for key, value in cli_args.items():
        if key == 'config':
            continue
        config[key] = value
    return argparse.Namespace(**config)


class Logger:
    """Mirror stdout into a log file."""

    def __init__(self, file_path: str) -> None:
        self.terminal = sys.stdout
        self.log = open(file_path, "a")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        self.log.flush()


def _set_device(gpuid: List[int]) -> None:
    if gpuid and max(gpuid) >= 0:
        try:
            import cupy  # noqa: F401
        except ModuleNotFoundError:
            print('Warning: cupy not found, falling back to CPU execution.')
            jt.flags.use_cuda = 0
            return
        jt.flags.use_cuda = 1
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", ",".join(str(g) for g in gpuid if g >= 0))
    else:
        jt.flags.use_cuda = 0


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)


if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    _set_device(args.gpuid)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    log_out = os.path.join(args.log_dir, 'output.log')
    sys.stdout = Logger(log_out)

    with open(os.path.join(args.log_dir, 'args.yaml'), 'w') as yaml_file:
        yaml.dump(vars(args), yaml_file, default_flow_style=False)

    metric_keys = ['acc', 'time']
    save_keys = ['global', 'pt', 'pt-local']
    global_only = {'time'}
    avg_metrics: Dict[str, Dict[str, np.ndarray]] = {k: {s: [] for s in save_keys} for k in metric_keys}

    if args.overwrite:
        start_r = 0
    else:
        start_r = 0
        for mkey in metric_keys:
            for skey in save_keys:
                if mkey in global_only and skey != 'global':
                    continue
                result_file = os.path.join(args.log_dir, f'results-{mkey}', f'{skey}.yaml')
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        data = yaml.safe_load(f)
                    if data and 'history' in data:
                        avg_metrics.setdefault(mkey, {})
                        history = np.asarray(data['history'])
                        avg_metrics[mkey][skey] = history
                        start_r = history.shape[-1]

    for r in range(start_r, args.repeat):
        print('************************************')
        print(f'* STARTING TRIAL {r + 1}')
        print('************************************')

        _set_seeds(r)
        trainer = Trainer(args, r, metric_keys, save_keys)

        max_task = trainer.max_task
        if r == 0:
            for mkey in metric_keys:
                if mkey not in avg_metrics or not avg_metrics[mkey]['global']:
                    avg_metrics[mkey]['global'] = np.zeros((max_task, args.repeat))
                if mkey not in global_only:
                    avg_metrics[mkey]['pt'] = np.zeros((max_task, max_task, args.repeat))
                    avg_metrics[mkey]['pt-local'] = np.zeros((max_task, max_task, args.repeat))

        avg_metrics = trainer.train(avg_metrics)
        avg_metrics = trainer.evaluate(avg_metrics)

        for mkey in metric_keys:
            target_dir = os.path.join(args.log_dir, f'results-{mkey}')
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            for skey in save_keys:
                if mkey in global_only and skey != 'global':
                    continue
                result = avg_metrics[mkey][skey]
                result_struct: Dict[str, Any] = {}
                if result.ndim > 2:
                    result_struct['mean'] = result[:, :, :r + 1].mean(axis=2).tolist()
                    if r > 1:
                        result_struct['std'] = result[:, :, :r + 1].std(axis=2).tolist()
                else:
                    result_struct['mean'] = result[:, :r + 1].mean(axis=1).tolist()
                    if r > 1:
                        result_struct['std'] = result[:, :r + 1].std(axis=1).tolist()
                result_struct['history'] = result[..., :r + 1].tolist()
                with open(os.path.join(target_dir, f'{skey}.yaml'), 'w') as f:
                    yaml.dump(result_struct, f, default_flow_style=False)

        print(f'===Summary of experiment repeats: {r + 1} / {args.repeat}===')
        for mkey in metric_keys:
            global_last = avg_metrics[mkey]['global'][-1, :r + 1]
            print(mkey, '| mean:', float(global_last.mean()), 'std:', float(global_last.std()))
