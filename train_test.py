"""
The code implementation of the paper:

A. Rasouli, I. Kotseruba, T. Kunic, and J. Tsotsos, "PIE: A Large-Scale Dataset and Models for Pedestrian Intention Estimation and
Trajectory Prediction", ICCV 2019.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import os
import sys
import argparse
import csv
import json
import random
import re
import shutil
import numpy as np

from pie_intent import PIEIntent
from pie_predict import PIEPredict

from pie_data import PIE

import keras.backend as K
import tensorflow as tf

from prettytable import PrettyTable

dim_ordering = K.image_dim_ordering()


def parse_int_list(arg_value, default_values):
    if arg_value is None:
        return list(default_values)
    values = [v.strip() for v in str(arg_value).split(',') if v.strip()]
    return [int(v) for v in values]


def maybe_subset_sequence_data(sequence_data, max_tracks=None):
    if not max_tracks:
        return sequence_data
    subset = {}
    for key, values in sequence_data.items():
        if isinstance(values, list):
            subset[key] = values[:max_tracks]
        else:
            subset[key] = values
    return subset


def summarize_metrics(rows):
    groups = {}
    for row in rows:
        group_key = (row.get('model_name'), row.get('K'))
        groups.setdefault(group_key, []).append(row)

    summary_rows = []
    metric_keys = ['minADE', 'minFDE', 'MSE', 'ADE', 'FDE', 'F1', 'AUC']
    for (model_name, k), group_rows in sorted(groups.items(), key=lambda x: x[0][1]):
        summary = {'model_name': model_name, 'K': k, 'num_runs': len(group_rows)}
        for metric_key in metric_keys:
            values = [r[metric_key] for r in group_rows if r.get(metric_key) is not None]
            if values:
                summary['%s_mean' % metric_key] = float(np.mean(values))
                summary['%s_std' % metric_key] = float(np.std(values, ddof=0))
            else:
                summary['%s_mean' % metric_key] = None
                summary['%s_std' % metric_key] = None
        summary_rows.append(summary)
    return summary_rows


def copy_if_exists(src, dst):
    if src and os.path.exists(src):
        shutil.copy2(src, dst)
        return True
    return False


def read_num_hypotheses_from_model(traj_model_path):
    if not traj_model_path:
        return None
    config_path = os.path.join(traj_model_path, 'configs.txt')
    if not os.path.exists(config_path):
        return None
    pattern = re.compile(r'^\s*num_hypotheses\s*:\s*(\d+)\s*$')
    with open(config_path, 'r') as f_cfg:
        for line in f_cfg:
            match = pattern.match(line.strip())
            if match:
                return int(match.group(1))
    return None


def save_run_artifacts(run_dir, run_config, run_metrics, intent_model_path, speed_model_path, traj_model_path,
                       num_hypotheses_from_model=None):
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    run_config_snapshot = dict(run_config)
    run_config_snapshot['num_hypotheses_from_model'] = num_hypotheses_from_model

    with open(os.path.join(run_dir, 'config_snapshot.json'), 'w') as f_cfg:
        json.dump(run_config_snapshot, f_cfg, indent=2, sort_keys=True)

    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f_met:
        json.dump(run_metrics, f_met, indent=2, sort_keys=True)

    ckpt_dir = os.path.join(run_dir, 'best_checkpoints')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    copied = {}
    for name, model_path in [('intent', intent_model_path), ('speed', speed_model_path), ('trajectory', traj_model_path)]:
        model_ckpt = os.path.join(model_path, 'model.h5') if model_path else ''
        target = os.path.join(ckpt_dir, '%s_model.h5' % name)
        copied[name] = copy_if_exists(model_ckpt, target)

    with open(os.path.join(run_dir, 'checkpoint_manifest.json'), 'w') as f_ckpt:
        json.dump(copied, f_ckpt, indent=2, sort_keys=True)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    try:
        tf.set_random_seed(seed)
    except AttributeError:
        tf.random.set_seed(seed)


def write_eval_table(rows, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    csv_path = os.path.join(output_dir, 'evaluation_summary.csv')
    json_path = os.path.join(output_dir, 'evaluation_summary.json')
    fields = sorted(set(k for row in rows for k in row.keys()))
    with open(csv_path, 'w') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with open(json_path, 'w') as f_json:
        json.dump(rows, f_json, indent=2, sort_keys=True)
    return csv_path, json_path


def write_summary_table(rows, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    csv_path = os.path.join(output_dir, 'evaluation_summary_mean_std.csv')
    json_path = os.path.join(output_dir, 'evaluation_summary_mean_std.json')
    fields = sorted(set(k for row in rows for k in row.keys()))
    with open(csv_path, 'w') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with open(json_path, 'w') as f_json:
        json.dump(rows, f_json, indent=2, sort_keys=True)
    return csv_path, json_path


def validate_trajectory_paths(ks, traj_model_path_map):
    unique_paths = {}
    for k in ks:
        if k not in traj_model_path_map:
            raise ValueError('Missing trajectory model path mapping for K={}'.format(k))
        model_path = traj_model_path_map[k]
        if not model_path:
            raise ValueError('Empty trajectory model path for K={}'.format(k))
        real_path = os.path.realpath(model_path)
        if real_path in unique_paths:
            raise ValueError('Trajectory model paths must be independent directories, but K={} and K={} '
                             'both map to {}'.format(unique_paths[real_path], k, model_path))
        unique_paths[real_path] = k


def run_consistency_check(row):
    expected_k = row.get('K')
    from_model = row.get('num_hypotheses_from_model')
    if from_model is None:
        return False, 'Missing num_hypotheses_from_model in checkpoint config'
    if from_model != expected_k:
        return False, 'K={} but checkpoint config num_hypotheses={}'.format(expected_k, from_model)
    return True, ''


def train_predict(dataset='pie',
                  train_test=2,
                  intent_model_path='data/pie/intention/context_loc_pretrained',
                  traj_model_path='data/pie/trajectory/loc_intent_speed_pretrained',
                  speed_model_path='data/pie/speed/speed_pretrained',
                  num_hypotheses=5,
                  batch_size=64,
                  eval_split='test',
                  max_tracks=None):
    data_opts = {'fstride': 1,
                 'sample_type': 'all',
                 'height_rng': [0, float('inf')],
                 'squarify_ratio': 0,
                 'data_split_type': 'default',  # kfold, random, default
                 'seq_type': 'trajectory',
                 'min_track_size': 61,
                 'random_params': {'ratios': None,
                                 'val_data': True,
                                 'regen_data': False},
                 'kfold_params': {'num_folds': 5, 'fold': 1}}

    t_traj = PIEPredict(num_hypotheses=num_hypotheses)
    t_speed = PIEPredict()
    pie_path = os.environ.copy()['PIE_PATH']

    if dataset == 'pie':
        imdb = PIE(data_path=pie_path)

    traj_model_opts = {'normalize_bbox': True,
                       'track_overlap': 0.5,
                       'observe_length': 15,
                       'predict_length': 45,
                       'enc_input_type': ['bbox'],
                       'dec_input_type': ['intention_prob', 'obd_speed'],
                       'prediction_type': ['bbox'] 
                       }

    speed_model_opts = {'normalize_bbox': True,
                       'track_overlap': 0.5,
                       'observe_length': 15,
                       'predict_length': 45,
                       'enc_input_type': ['obd_speed'], 
                       'dec_input_type': [],
                       'prediction_type': ['obd_speed'] 
                       }

    if train_test < 2:
        beh_seq_val = imdb.generate_data_trajectory_sequence('val', **data_opts)
        beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
        beh_seq_val = maybe_subset_sequence_data(beh_seq_val, max_tracks=max_tracks)
        beh_seq_train = maybe_subset_sequence_data(beh_seq_train, max_tracks=max_tracks)
        traj_model_path = t_traj.train(beh_seq_train, beh_seq_val, batch_size=batch_size, **traj_model_opts)
        speed_model_path = t_speed.train(beh_seq_train, beh_seq_val, batch_size=batch_size, **speed_model_opts)

    if train_test > 0:
        beh_seq_test = imdb.generate_data_trajectory_sequence(eval_split, **data_opts)
        beh_seq_test = maybe_subset_sequence_data(beh_seq_test, max_tracks=max_tracks)

        perf_final = t_traj.test_final(beh_seq_test,
                                       traj_model_path=traj_model_path,
                                       speed_model_path=speed_model_path,
                                       intent_model_path=intent_model_path)

        required_perf_keys = ['mse-45', 'c-mse-45', 'ade', 'fde', 'minADE', 'minFDE']
        for metric_key in required_perf_keys:
            if metric_key not in perf_final:
                print('WARNING: Missing metric "{}" from test_final output. '
                      'Filling with None for stable CSV/report output.'.format(metric_key))
                perf_final[metric_key] = None

        t = PrettyTable(['MSE', 'C_MSE', 'ADE', 'FDE', 'minADE', 'minFDE'])
        t.title = 'Trajectory prediction model (loc + PIE_intent + PIE_speed)'
        t.add_row([perf_final['mse-45'],
                   perf_final['c-mse-45'],
                   perf_final['ade'],
                   perf_final['fde'],
                   perf_final['minADE'],
                   perf_final['minFDE']])
        
        print(t)
        return perf_final

#train models with data up to critical point
#only for PIE
#train_test = 0 (train only), 1 (train-test), 2 (test only)
def train_intent(train_test=1,
                 model_path='data/pie/intention/context_loc_pretrained',
                 eval_split='test',
                 max_tracks=None):

    data_opts = {'fstride': 1,
            'sample_type': 'all', 
            'height_rng': [0, float('inf')],
            'squarify_ratio': 0,
            'data_split_type': 'default',  #  kfold, random, default
            'seq_type': 'intention', #  crossing , intention
            'min_track_size': 0, #  discard tracks that are shorter
            'max_size_observe': 15,  # number of observation frames
            'max_size_predict': 5,  # number of prediction frames
            'seq_overlap_rate': 0.5,  # how much consecutive sequences overlap
            'balance': True,  # balance the training and testing samples
            'crop_type': 'context',  # crop 2x size of bbox around the pedestrian
            'crop_mode': 'pad_resize',  # pad with 0s and resize to VGG input
            'encoder_input_type': [],
            'decoder_input_type': ['bbox'],
            'output_type': ['intention_binary']
            }


    t = PIEIntent(num_hidden_units=128,
                  regularizer_val=0.001,
                  lstm_dropout=0.4,
                  lstm_recurrent_dropout=0.2,
                  convlstm_num_filters=64,
                  convlstm_kernel_size=2)

    saved_files_path = ''

    imdb = PIE(data_path=os.environ.copy()['PIE_PATH'])

    pretrained_model_path = model_path

    if train_test < 2:  # Train
        beh_seq_val = imdb.generate_data_trajectory_sequence('val', **data_opts)
        beh_seq_val = maybe_subset_sequence_data(beh_seq_val, max_tracks=max_tracks)
        beh_seq_val = imdb.balance_samples_count(beh_seq_val, label_type='intention_binary')

        beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
        beh_seq_train = maybe_subset_sequence_data(beh_seq_train, max_tracks=max_tracks)
        beh_seq_train = imdb.balance_samples_count(beh_seq_train, label_type='intention_binary')

        saved_files_path = t.train(data_train=beh_seq_train,
                                   data_val=beh_seq_val,
                                   epochs=400,
                                   loss=['binary_crossentropy'],
                                   metrics=['accuracy'],
                                   batch_size=128,
                                   optimizer_type='rmsprop',
                                   data_opts=data_opts)

        print(data_opts['seq_overlap_rate'])

    metrics = {}
    if train_test > 0:  # Test
        if saved_files_path == '':
            saved_files_path = pretrained_model_path
        beh_seq_test = imdb.generate_data_trajectory_sequence(eval_split, **data_opts)
        beh_seq_test = maybe_subset_sequence_data(beh_seq_test, max_tracks=max_tracks)
        acc, f1, auc = t.test_chunk(beh_seq_test, data_opts, saved_files_path, False)
        
        t = PrettyTable(['Acc', 'F1', 'AUC'])
        t.title = 'Intention model (local_context + bbox)'
        t.add_row([acc, f1, auc])
        
        print(t)

        K.clear_session()
        tf.reset_default_graph()
        metrics = {'Accuracy': acc, 'F1': f1, 'AUC': auc}
    return saved_files_path, metrics


def main(dataset='pie', train_test=2, batch_size=64,
         seeds=(42, 43, 44), ks=(1, 5, 10),
         protocol='quick', quick_max_tracks=512,
         intent_model_path='data/pie/intention/context_loc_pretrained',
         speed_model_path='data/pie/speed/speed_pretrained',
         traj_model_path_k1='',
         traj_model_path_k5='',
         traj_model_path_k10='',
         eval_output_dir='data/pie/eval_reports'):

      if protocol == 'quick':
          eval_split = 'val'
          max_tracks = quick_max_tracks
      else:
          eval_split = 'test'
          max_tracks = None

      rows = []
      invalid_rows = []
      traj_model_path_map = {1: traj_model_path_k1, 5: traj_model_path_k5, 10: traj_model_path_k10}
      model_name_map = {1: 'baseline', 5: 'multi-future', 10: 'multi-future'}
      validate_trajectory_paths(ks, traj_model_path_map)

      path_hypothesis_map = {}
      for k, model_path in traj_model_path_map.items():
          if k in ks:
              path_hypothesis_map[k] = read_num_hypotheses_from_model(model_path)
              print('Trajectory checkpoint K={} path={} config num_hypotheses={}'.format(
                  k, model_path, path_hypothesis_map[k]))

      for seed in seeds:
          set_global_seed(seed)
          print('Running seed={} protocol={} eval_split={} max_tracks={}'.format(seed, protocol, eval_split, max_tracks))

          intent_path_seed, intent_metrics = train_intent(train_test=train_test,
                                                          model_path=intent_model_path,
                                                          eval_split=eval_split,
                                                          max_tracks=max_tracks)
          for k in ks:
              traj_model_path = traj_model_path_map[k]
              num_hypotheses_from_model = path_hypothesis_map.get(k)
              model_name = model_name_map.get(k, 'model-k{}'.format(k))
              print('Trajectory num_hypotheses (K): {}'.format(k))
              t_metrics = train_predict(dataset=dataset,
                                        train_test=train_test,
                                        intent_model_path=intent_path_seed,
                                        traj_model_path=traj_model_path,
                                        speed_model_path=speed_model_path,
                                        num_hypotheses=k,
                                        batch_size=batch_size,
                                        eval_split=eval_split,
                                        max_tracks=max_tracks)
              if train_test > 0 and t_metrics is not None:
                  row = {'model_name': model_name,
                         'protocol': protocol,
                         'eval_split': eval_split,
                         'K': k,
                         'seed': seed,
                         'max_tracks': max_tracks,
                         'minADE': t_metrics.get('minADE'),
                         'minFDE': t_metrics.get('minFDE'),
                         'F1': intent_metrics.get('F1'),
                         'AUC': intent_metrics.get('AUC'),
                         'MSE': t_metrics.get('mse-45'),
                         'ADE': t_metrics.get('ade'),
                         'FDE': t_metrics.get('fde'),
                         'traj_model_path': traj_model_path,
                         'num_hypotheses_from_model': num_hypotheses_from_model}
                  is_valid, invalid_reason = run_consistency_check(row)
                  row['consistency_valid'] = is_valid
                  row['consistency_message'] = invalid_reason
                  if is_valid:
                      rows.append(row)
                  else:
                      invalid_rows.append(row)
                      print('WARNING: Invalid run removed from report (seed={}, K={}): {}'.format(
                          seed, k, invalid_reason))

                  run_dir = os.path.join(eval_output_dir, protocol, 'K{}_seed{}'.format(k, seed))
                  run_config = {'dataset': dataset,
                                'train_test': train_test,
                                'batch_size': batch_size,
                                'seed': seed,
                                'K': k,
                                'protocol': protocol,
                                'eval_split': eval_split,
                                'max_tracks': max_tracks,
                                'intent_model_path': intent_path_seed,
                                'speed_model_path': speed_model_path,
                                'traj_model_path': traj_model_path,
                                'consistency_valid': is_valid,
                                'consistency_message': invalid_reason}
                  save_run_artifacts(run_dir=run_dir,
                                     run_config=run_config,
                                     run_metrics=row,
                                     intent_model_path=intent_path_seed,
                                     speed_model_path=speed_model_path,
                                     traj_model_path=traj_model_path,
                                     num_hypotheses_from_model=num_hypotheses_from_model)

      if rows:
          csv_path, json_path = write_eval_table(rows, os.path.join(eval_output_dir, protocol))
          summary_rows = summarize_metrics(rows)
          summary_csv_path, summary_json_path = write_summary_table(summary_rows, os.path.join(eval_output_dir, protocol))
          print('Saved evaluation table: {}'.format(csv_path))
          print('Saved evaluation table: {}'.format(json_path))
          print('Saved aggregate table (mean/std): {}'.format(summary_csv_path))
          print('Saved aggregate table (mean/std): {}'.format(summary_json_path))
      if invalid_rows:
          invalid_csv_path, invalid_json_path = write_eval_table(
              invalid_rows, os.path.join(eval_output_dir, protocol, 'invalid_runs'))
          print('Saved invalid run table: {}'.format(invalid_csv_path))
          print('Saved invalid run table: {}'.format(invalid_json_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('legacy_train_test', nargs='?', type=int, choices=[0, 1, 2],
                        help='[Deprecated] 旧版位置参数 1/2 已废弃。请改用 --train_test {0,1,2}。')
    parser.add_argument('--dataset', type=str, default='pie')
    parser.add_argument('--train_test', type=int, default=2,
                        help='0 - train only, 1 - train and test, 2 - test only. 旧版位置参数 1/2 已废弃。')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Trajectory/speed training batch size. Reduce this if GPU OOM occurs.')
    parser.add_argument('--seeds', type=str, default='42,43,44',
                        help='Comma-separated random seeds for reproducible experiments, e.g., 42,43,44.')
    parser.add_argument('--ks', type=str, default='1,5,10',
                        help='Comma-separated K values for baseline and multi-future models, e.g., 1,5,10.')
    parser.add_argument('--protocol', type=str, default='quick', choices=['quick', 'formal'],
                        help='quick: reproducible subset on validation split; formal: full test split.')
    parser.add_argument('--quick_max_tracks', type=int, default=512,
                        help='Max number of tracks used per split in quick protocol.')
    parser.add_argument('--intent_model_path', type=str, default='data/pie/intention/context_loc_pretrained')
    parser.add_argument('--speed_model_path', type=str, default='data/pie/speed/speed_pretrained')
    parser.add_argument('--traj_model_path_k1', type=str, required=True,
                        help='Trajectory checkpoint directory for K=1 (must be an independent path).')
    parser.add_argument('--traj_model_path_k5', type=str, required=True,
                        help='Trajectory checkpoint directory for K=5 (must be an independent path).')
    parser.add_argument('--traj_model_path_k10', type=str, required=True,
                        help='Trajectory checkpoint directory for K=10 (must be an independent path).')
    parser.add_argument('--eval_output_dir', type=str, default='data/pie/eval_reports')

    args = parser.parse_args()
    if args.legacy_train_test is not None:
        print('WARNING: Legacy positional argument "{}" is deprecated. '
              'Please use --train_test {}.'.format(args.legacy_train_test, args.legacy_train_test))
        args.train_test = args.legacy_train_test

    main(dataset=args.dataset,
         train_test=args.train_test,
         batch_size=args.batch_size,
         seeds=parse_int_list(args.seeds, [42, 43, 44]),
         ks=parse_int_list(args.ks, [1, 5, 10]),
         protocol=args.protocol,
         quick_max_tracks=args.quick_max_tracks,
         intent_model_path=args.intent_model_path,
         speed_model_path=args.speed_model_path,
         traj_model_path_k1=args.traj_model_path_k1,
         traj_model_path_k5=args.traj_model_path_k5,
         traj_model_path_k10=args.traj_model_path_k10,
         eval_output_dir=args.eval_output_dir)
