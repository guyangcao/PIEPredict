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
import numpy as np

from pie_intent import PIEIntent
from pie_predict import PIEPredict

from pie_data import PIE

import keras.backend as K
import tensorflow as tf

from prettytable import PrettyTable

dim_ordering = K.image_dim_ordering()


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
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


def train_predict(dataset='pie',
                  train_test=2,
                  intent_model_path='data/pie/intention/context_loc_pretrained',
                  traj_model_path='data/pie/trajectory/loc_intent_speed_pretrained',
                  speed_model_path='data/pie/speed/speed_pretrained',
                  num_hypotheses=5,
                  batch_size=64,
                  eval_split='test'):
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
        traj_model_path = t_traj.train(beh_seq_train, beh_seq_val, batch_size=batch_size, **traj_model_opts)
        speed_model_path = t_speed.train(beh_seq_train, beh_seq_val, batch_size=batch_size, **speed_model_opts)

    if train_test > 0:
        beh_seq_test = imdb.generate_data_trajectory_sequence(eval_split, **data_opts)

        perf_final = t_traj.test_final(beh_seq_test,
                                       traj_model_path=traj_model_path,
                                       speed_model_path=speed_model_path,
                                       intent_model_path=intent_model_path)

        t = PrettyTable(['MSE', 'C_MSE', 'ADE', 'FDE', 'minADE', 'minFDE'])
        t.title = 'Trajectory prediction model (loc + PIE_intent + PIE_speed)'
        t.add_row([perf_final['mse-45'],
                   perf_final['c-mse-45'],
                   perf_final.get('ade'),
                   perf_final.get('fde'),
                   perf_final.get('minADE'),
                   perf_final.get('minFDE')])
        
        print(t)
        return perf_final

#train models with data up to critical point
#only for PIE
#train_test = 0 (train only), 1 (train-test), 2 (test only)
def train_intent(train_test=1,
                 model_path='data/pie/intention/context_loc_pretrained',
                 eval_split='test'):

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
        beh_seq_val = imdb.balance_samples_count(beh_seq_val, label_type='intention_binary')

        beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
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
        acc, f1, auc = t.test_chunk(beh_seq_test, data_opts, saved_files_path, False)
        
        t = PrettyTable(['Acc', 'F1', 'AUC'])
        t.title = 'Intention model (local_context + bbox)'
        t.add_row([acc, f1, auc])
        
        print(t)

        K.clear_session()
        tf.reset_default_graph()
        metrics = {'Accuracy': acc, 'F1': f1, 'AUC': auc}
    return saved_files_path, metrics


def main(dataset='pie', train_test=2, num_hypotheses=5, batch_size=64,
         seed=42, eval_split='val',
         intent_model_path='data/pie/intention/context_loc_pretrained',
         speed_model_path='data/pie/speed/speed_pretrained',
         traj_model_path_k1='data/pie/trajectory/loc_intent_speed_pretrained',
         traj_model_path_k5='data/pie/trajectory/loc_intent_speed_pretrained',
         traj_model_path_k10='data/pie/trajectory/loc_intent_speed_pretrained',
         eval_output_dir='data/pie/eval_reports'):

      set_global_seed(seed)
      intent_model_path, intent_metrics = train_intent(train_test=train_test,
                                                       model_path=intent_model_path,
                                                       eval_split=eval_split)
      rows = []
      traj_models = [('baseline', 1, traj_model_path_k1),
                     ('multi-future', 5, traj_model_path_k5),
                     ('multi-future', 10, traj_model_path_k10)]
      for model_name, k, traj_model_path in traj_models:
          print('Trajectory num_hypotheses (K): {}'.format(k))
          t_metrics = train_predict(dataset=dataset,
                                    train_test=train_test,
                                    intent_model_path=intent_model_path,
                                    traj_model_path=traj_model_path,
                                    speed_model_path=speed_model_path,
                                    num_hypotheses=k,
                                    batch_size=batch_size,
                                    eval_split=eval_split)
          if train_test > 0 and t_metrics is not None:
              row = {'model_name': model_name,
                     'K': k,
                     'seed': seed,
                     'minADE': t_metrics.get('minADE'),
                     'minFDE': t_metrics.get('minFDE'),
                     'F1': intent_metrics.get('F1'),
                     'AUC': intent_metrics.get('AUC'),
                     'MSE': t_metrics.get('mse-45'),
                     'ADE': t_metrics.get('ade'),
                     'FDE': t_metrics.get('fde')}
              rows.append(row)

      if rows:
          csv_path, json_path = write_eval_table(rows, eval_output_dir)
          print('Saved evaluation table: {}'.format(csv_path))
          print('Saved evaluation table: {}'.format(json_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pie')
    parser.add_argument('--train_test', type=int, default=2,
                        help='0 - train only, 1 - train and test, 2 - test only')
    parser.add_argument('--num_hypotheses', type=int, default=5, choices=[5, 10],
                        help='Number of trajectory hypotheses (K). Use K=5 by default to reduce OOM risk.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Trajectory/speed training batch size. Reduce this if GPU OOM occurs.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible testing/evaluation.')
    parser.add_argument('--eval_split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Data split used for reporting metrics.')
    parser.add_argument('--intent_model_path', type=str, default='data/pie/intention/context_loc_pretrained')
    parser.add_argument('--speed_model_path', type=str, default='data/pie/speed/speed_pretrained')
    parser.add_argument('--traj_model_path_k1', type=str, default='data/pie/trajectory/loc_intent_speed_pretrained')
    parser.add_argument('--traj_model_path_k5', type=str, default='data/pie/trajectory/loc_intent_speed_pretrained')
    parser.add_argument('--traj_model_path_k10', type=str, default='data/pie/trajectory/loc_intent_speed_pretrained')
    parser.add_argument('--eval_output_dir', type=str, default='data/pie/eval_reports')

    args = parser.parse_args()
    main(dataset=args.dataset,
         train_test=args.train_test,
         num_hypotheses=args.num_hypotheses,
         batch_size=args.batch_size,
         seed=args.seed,
         eval_split=args.eval_split,
         intent_model_path=args.intent_model_path,
         speed_model_path=args.speed_model_path,
         traj_model_path_k1=args.traj_model_path_k1,
         traj_model_path_k5=args.traj_model_path_k5,
         traj_model_path_k10=args.traj_model_path_k10,
         eval_output_dir=args.eval_output_dir)
