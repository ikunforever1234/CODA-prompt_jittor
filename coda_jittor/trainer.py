"""Trainer coordinating continual learning experiments with Jittor."""
from __future__ import annotations

import os
import random
from collections import OrderedDict
from typing import Dict, List

import numpy as np
import jittor as jt

from . import dataloaders, learners


class Trainer:
    """Mirror of the original Trainer but targeting Jittor modules."""

    def __init__(self, args, seed: int, metric_keys: List[str], save_keys: List[str]):
        self.args = args
        self.seed = seed
        self.metric_keys = metric_keys
        self.save_keys = save_keys
        self.log_dir = args.log_dir
        self.batch_size = args.batch_size
        self.workers = args.workers
        self.model_top_dir = args.log_dir
        self.top_k = 1

        dataset_entry = dataloaders.get_dataset_entry(args.dataset)
        if dataset_entry is None:
            raise ValueError(f'Dataset {args.dataset} not implemented for Jittor port.')
        DatasetCls, num_classes, dataset_size = dataset_entry
        self.dataset_size = dataset_size
        base_model_kwargs = getattr(args, 'model_kwargs', None)
        if base_model_kwargs is None:
            self.model_kwargs = {}
        elif isinstance(base_model_kwargs, dict):
            self.model_kwargs = dict(base_model_kwargs)
        else:
            raise ValueError('model_kwargs must be a mapping if provided.')
        short_edge = min(dataset_size[:2]) if isinstance(dataset_size, (list, tuple)) else dataset_size
        if short_edge <= 64:
            self.model_kwargs.setdefault('img_size', short_edge)
            self.model_kwargs.setdefault('patch_size', 4)
            self.model_kwargs.setdefault('embed_dim', 384)
            self.model_kwargs.setdefault('depth', 8)
            self.model_kwargs.setdefault('num_heads', 8)
        else:
            self.model_kwargs.setdefault('img_size', short_edge)
            if short_edge >= 128:
                # Reduce backbone width/depth for high-resolution datasets to stay within 6GB GPUs.
                self.model_kwargs.setdefault('embed_dim', 384)
                self.model_kwargs.setdefault('depth', 6)
                self.model_kwargs.setdefault('num_heads', 6)

        if args.upper_bound_flag:
            args.other_split_size = num_classes
            args.first_split_size = num_classes

        class_order = np.arange(num_classes).tolist()
        class_order_logits = np.arange(num_classes).tolist()
        if self.seed > 0 and args.rand_split:
            print('=============================================')
            print('Shuffling class order...')
            print('pre-shuffle:' + str(class_order))
            random.seed(self.seed)
            random.shuffle(class_order)
            print('post-shuffle:' + str(class_order))
            print('=============================================')
        self.tasks: List[List[int]] = []
        self.tasks_logits: List[List[int]] = []
        p = 0
        while p < num_classes and (args.max_task == -1 or len(self.tasks) < args.max_task):
            inc = args.other_split_size if p > 0 else args.first_split_size
            self.tasks.append(class_order[p:p + inc])
            self.tasks_logits.append(class_order_logits[p:p + inc])
            p += inc
        self.num_tasks = len(self.tasks)
        self.task_names = [str(i + 1) for i in range(self.num_tasks)]
        self.max_task = min(args.max_task, len(self.task_names)) if args.max_task > 0 else len(self.task_names)

        resize_imnet = args.model_name.startswith('vit')
        self.train_transform = dataloaders.build_transform(args.dataset, 'train', args.train_aug, resize_imnet)
        self.test_transform = dataloaders.build_transform(args.dataset, 'test', args.train_aug, resize_imnet)

        self.train_dataset = DatasetCls(
            root=args.dataroot,
            train=True,
            transform=self.train_transform,
            download_flag=True,
            lab=True,
            tasks=self.tasks,
            seed=self.seed,
            rand_split=args.rand_split,
            validation=args.validation,
        )
        self.test_dataset = DatasetCls(
            root=args.dataroot,
            train=False,
            transform=self.test_transform,
            download_flag=False,
            lab=True,
            tasks=self.tasks,
            seed=self.seed,
            rand_split=args.rand_split,
            validation=args.validation,
        )

        self.oracle_flag = args.oracle_flag
        self.add_dim = 0
        self.learner_config: Dict[str, object] = {
            'num_classes': num_classes,
            'lr': args.lr,
            'debug_mode': args.debug_mode == 1,
            'momentum': args.momentum,
            'weight_decay': args.weight_decay,
            'schedule': args.schedule,
            'schedule_type': args.schedule_type,
            'model_type': args.model_type,
            'model_name': args.model_name,
            'optimizer': args.optimizer,
            'gpuid': args.gpuid,
            'memory': args.memory,
            'temp': args.temp,
            'out_dim': num_classes,
            'overwrite': args.overwrite == 1,
            'DW': args.DW,
            'batch_size': args.batch_size,
            'upper_bound_flag': args.upper_bound_flag,
            'tasks': self.tasks_logits,
            'top_k': self.top_k,
            'prompt_param': [self.num_tasks, args.prompt_param],
            'model_kwargs': self.model_kwargs,
        }
        self.learner_type = args.learner_type
        self.learner_name = args.learner_name
        self.learner = learners.build_learner(self.learner_type, self.learner_name, self.learner_config)

    def _build_loader(self, dataset, batch_size: int, shuffle: bool, drop_last: bool, num_workers: int):
        return dataloaders.build_loader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    def task_eval(self, t_index: int, local: bool = False, task: str = 'acc'):
        val_name = self.task_names[t_index]
        print('validation split name:', val_name)
        self.test_dataset.load_dataset(t_index, train=True)
        test_loader = self._build_loader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
        if local:
            return self.learner.validation(test_loader, task_in=self.tasks_logits[t_index], task_metric=task)
        return self.learner.validation(test_loader, task_metric=task)

    def train(self, avg_metrics):
        temp_table: Dict[str, List[float]] = {mkey: [] for mkey in self.metric_keys}
        temp_dir = os.path.join(self.log_dir, 'temp')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        for i in range(self.max_task):
            self.current_t_index = i
            train_name = self.task_names[i]
            print('======================', train_name, '=======================')

            task = self.tasks_logits[i]
            if self.oracle_flag:
                self.train_dataset.load_dataset(i, train=False)
                self.learner = learners.build_learner(self.learner_type, self.learner_name, self.learner_config)
                self.add_dim += len(task)
            else:
                self.train_dataset.load_dataset(i, train=True)
                self.add_dim = len(task)

            self.learner.set_task_id(i)
            self.learner.add_valid_output_dim(self.add_dim)
            self.train_dataset.append_coreset(only=False)
            attempt = 0
            while True:
                train_loader = self._build_loader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=int(self.workers))

                if i > 0:
                    self.learner.process_task_count()

                self.test_dataset.load_dataset(i, train=False)
                test_loader = self._build_loader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
                model_save_dir = os.path.join(self.model_top_dir, 'models', f'repeat-{self.seed + 1}', f'task-{self.task_names[i]}')
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                try:
                    avg_train_time = self.learner.learn_batch(train_loader, self.train_dataset, model_save_dir, test_loader)
                    break
                except RuntimeError as err:
                    message = str(err).lower()
                    if ('cuda' in message or 'memory' in message) and 'memory' in message and self.batch_size > 1:
                        self.batch_size = max(1, self.batch_size // 2)
                        self.learner.batch_size = self.batch_size
                        self.learner_config['batch_size'] = self.batch_size
                        attempt += 1
                        try:
                            jt.sync_all(True)
                        except Exception:
                            pass
                        try:
                            import gc
                            gc.collect()
                        except Exception:
                            pass
                        print(f'Caught OOM during task {self.task_names[i]}, reducing batch size to {self.batch_size} and retrying (attempt {attempt}).')
                        continue
                    raise

            self.learner.save_model(model_save_dir)

            acc_table = []
            for j in range(i + 1):
                acc_table.append(self.task_eval(j))
            temp_table['acc'].append(float(np.mean(np.asarray(acc_table))))
            np.savetxt(os.path.join(temp_dir, 'acc.csv'), np.asarray(temp_table['acc']), delimiter=",", fmt='%.2f')

            if avg_train_time is not None:
                avg_metrics['time']['global'][i] = avg_train_time

        return avg_metrics

    def summarize_acc(self, acc_dict, acc_table, acc_table_pt):
        avg_acc_all = acc_dict['global']
        avg_acc_pt = acc_dict['pt']
        avg_acc_pt_local = acc_dict['pt-local']

        avg_acc_history = [0.0] * self.max_task
        for i in range(self.max_task):
            train_name = self.task_names[i]
            cls_acc_sum = 0.0
            for j in range(i + 1):
                val_name = self.task_names[j]
                cls_acc = acc_table[val_name][train_name]
                cls_acc_sum += cls_acc
                avg_acc_pt[j, i, self.seed] = cls_acc
                avg_acc_pt_local[j, i, self.seed] = acc_table_pt[val_name][train_name]
            avg_acc_history[i] = cls_acc_sum / (i + 1)
        avg_acc_all[:, self.seed] = avg_acc_history
        return {'global': avg_acc_all, 'pt': avg_acc_pt, 'pt-local': avg_acc_pt_local}

    def evaluate(self, avg_metrics):
        self.learner = learners.build_learner(self.learner_type, self.learner_name, self.learner_config)

        metric_table: Dict[str, Dict[str, OrderedDict]] = {mkey: {} for mkey in self.metric_keys}
        metric_table_local: Dict[str, Dict[str, OrderedDict]] = {mkey: {} for mkey in self.metric_keys}

        for i in range(self.max_task):
            if i > 0:
                self.learner.process_task_count()

            model_save_dir = os.path.join(self.model_top_dir, 'models', f'repeat-{self.seed + 1}', f'task-{self.task_names[i]}')
            self.learner.task_count = i
            self.learner.add_valid_output_dim(len(self.tasks_logits[i]))
            self.learner.pre_steps()
            self.learner.load_model(model_save_dir)
            self.learner.set_task_id(i)

            metric_table['acc'][self.task_names[i]] = OrderedDict()
            metric_table_local['acc'][self.task_names[i]] = OrderedDict()
            for j in range(i + 1):
                val_name = self.task_names[j]
                metric_table['acc'][val_name][self.task_names[i]] = self.task_eval(j)
            for j in range(i + 1):
                val_name = self.task_names[j]
                metric_table_local['acc'][val_name][self.task_names[i]] = self.task_eval(j, local=True)

        avg_metrics['acc'] = self.summarize_acc(avg_metrics['acc'], metric_table['acc'], metric_table_local['acc'])
        return avg_metrics
