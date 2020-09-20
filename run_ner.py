# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""BERT NER finetuning runner in TF 2.x."""
from __future__ import absolute_import, division, print_function

import functools
import json
import math
import os

import tensorflow as tf
from absl import app, flags, logging

import gin
from official.modeling import performance
from official.nlp import optimization
from official.nlp.bert import bert_models, common_flags
from official.nlp.bert import configs as bert_configs
from official.nlp.bert import model_saving_utils, tokenization
from official.utils.misc import distribution_utils, keras_utils

from data.dataset import NERDataset
from utils.model_utils import get_dataset_fn, run_keras_compile_fit
from model.ner_bert import ner_model
from utils.losses import get_ner_loss_fn
from utils.callback import NERF1Metrics

flags.DEFINE_enum(
    'mode', 'train_and_eval', ['train_and_eval', 'export_only', 'predict'],
    'One of {"train_and_eval", "export_only", "predict"}. `train_and_eval`: '
    'trains the model and evaluates in the meantime. '
    '`export_only`: will take the latest checkpoint inside '
    'model_dir and export a `SavedModel`. `predict`: takes a checkpoint and '
    'restores the model to output predictions on the test set.')
flags.DEFINE_string('train_data_path', None,
                    'Path to training data for BERT classifier.')
flags.DEFINE_string('eval_data_path', None,
                    'Path to evaluation data for BERT classifier.')
flags.DEFINE_string('predict_checkpoint_path', None,
                    'Path to the checkpoint for predictions.')
flags.DEFINE_integer(
    'num_eval_per_epoch', 1,
    'Number of evaluations per epoch. The purpose of this flag is to provide '
    'more granular evaluation scores and checkpoints. For example, if original '
    'data has N samples and num_eval_per_epoch is n, then each epoch will be '
    'evaluated every N/n samples.')
flags.DEFINE_integer('train_batch_size', 32, 'Batch size for training.')
flags.DEFINE_integer('eval_batch_size', 32, 'Batch size for evaluation.')

# 额外的flags
flags.DEFINE_integer('max_seq_length', 512, 'Max sequence length, default 512, you can use smaller number to save memory')
flags.DEFINE_string('vocab_file', None,
                    'The vocabulary file that the BERT model was trained on.')
flags.DEFINE_string('label_file', None,
                    'The label file that to map the labels.')
flags.DEFINE_integer('train_data_size', None, 'size of training dataset.')
flags.DEFINE_integer('eval_data_size', None, 'size of evaluation dataset.')
common_flags.define_common_bert_flags()

FLAGS = flags.FLAGS


def run_bert_ner(strategy,
                bert_config,
                input_meta_data,
                model_dir,
                epochs,
                steps_per_epoch,
                steps_per_loop,
                eval_steps,
                warmup_steps,
                initial_lr,
                init_checkpoint,
                train_input_fn,
                eval_input_fn,
                training_callbacks=True,
                custom_callbacks=None,
                custom_metrics=None):
  """Run BERT NER training using low-level API."""
  max_seq_length = input_meta_data['max_seq_length']
  num_classes = input_meta_data['num_labels']
  id2label = input_meta_data['id2label']
  logging.info(f'class num {num_classes}')

  def _get_model():
    """Gets a ner model."""
    model, core_model = (
      ner_model(
          bert_config,
          num_classes))
    optimizer = optimization.create_optimizer(initial_lr,
                                              steps_per_epoch * epochs,
                                              warmup_steps, FLAGS.end_lr,
                                              FLAGS.optimizer_type)
    model.optimizer = performance.configure_optimizer(
        optimizer,
        use_float16=common_flags.use_float16(),
        use_graph_rewrite=common_flags.use_graph_rewrite())
    return model, core_model

  # tf.keras.losses objects accept optional sample_weight arguments (eg. coming
  # from the dataset) to compute weighted loss, as used for the regression
  # tasks. The classification tasks, using the custom get_loss_fn don't accept
  # sample weights though.
  loss_fn = get_ner_loss_fn(num_classes)

  # Defines evaluation metrics function, which will create metrics in the
  # correct device and strategy scope.
  monitor = None
  if custom_metrics:
    metric_fn = custom_metrics
  else:
    metric_fn = functools.partial(
      tf.keras.metrics.SparseCategoricalAccuracy,
      'accuracy',
      dtype=tf.float32)

  f1_callback = NERF1Metrics(id2label,eval_input_fn(),model_dir=model_dir)
  if custom_callbacks:
    custom_callbacks.append(f1_callback)
  else:
    custom_callbacks = [f1_callback]
  
  if not monitor:
    monitor = 'val_accuracy'

  # Start training using Keras compile/fit API.
  logging.info('Training using TF 2.x Keras compile/fit API with '
               'distribution strategy.')
  return run_keras_compile_fit(
      model_dir,
      strategy,
      _get_model,
      train_input_fn,
      eval_input_fn,
      loss_fn,
      metric_fn,
      init_checkpoint,
      epochs,
      steps_per_epoch,
      steps_per_loop,
      eval_steps,
      monitor=monitor,
      training_callbacks=training_callbacks,
      custom_callbacks=custom_callbacks)


def run_bert(strategy,
             input_meta_data,
             model_config,
             train_input_fn=None,
             eval_input_fn=None,
             init_checkpoint=None):
  # Enables XLA in Session Config. Should not be set for TPU.
  keras_utils.set_session_config(FLAGS.enable_xla)
  performance.set_mixed_precision_policy(common_flags.dtype())

  epochs = FLAGS.num_train_epochs * FLAGS.num_eval_per_epoch
  train_data_size = (
      input_meta_data['train_data_size'] // FLAGS.num_eval_per_epoch)
  steps_per_epoch = int(train_data_size / FLAGS.train_batch_size)
  warmup_steps = int(epochs * train_data_size * 0.1 / FLAGS.train_batch_size)
  eval_steps = int(
      math.ceil(input_meta_data['eval_data_size'] / FLAGS.eval_batch_size))

  if not strategy:
    raise ValueError('Distribution strategy has not been specified.')
  
  custom_callbacks = []
  if FLAGS.log_steps:
    custom_callbacks.append(
        keras_utils.TimeHistory(
            batch_size=FLAGS.train_batch_size,
            log_steps=FLAGS.log_steps,
            logdir=FLAGS.model_dir))
  trained_model, _ = run_bert_ner(
      strategy,
      model_config,
      input_meta_data,
      FLAGS.model_dir,
      epochs,
      steps_per_epoch,
      FLAGS.steps_per_loop,
      eval_steps,
      warmup_steps,
      FLAGS.learning_rate,
      init_checkpoint or FLAGS.init_checkpoint,
      train_input_fn,
      eval_input_fn,
      custom_callbacks=custom_callbacks)

  if FLAGS.model_export_path:
    model_saving_utils.export_bert_model(
        FLAGS.model_export_path, model=trained_model)
  return trained_model

def main(_):
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
  if not FLAGS.model_dir:
    FLAGS.model_dir = '/tmp/bert20/'

  bert_config = bert_configs.BertConfig.from_json_file(FLAGS.bert_config_file)
  tokeninzer = tokenization.FullTokenizer(FLAGS.vocab_file, do_lower_case=True)
  if FLAGS.mode == 'export_only':
    raise NotImplementedError()
    return

  strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=FLAGS.distribution_strategy,
      num_gpus=FLAGS.num_gpus)

  eval_dataset = NERDataset(
    tokeninzer, 
    FLAGS.eval_data_path, 
    FLAGS.mode, 
    FLAGS.label_file, 
    FLAGS.max_seq_length)
  eval_input_fn = get_dataset_fn(
      eval_dataset,
      FLAGS.eval_batch_size,
      is_training=False)
  
  if FLAGS.mode != 'train_and_eval':
    raise ValueError('Unsupported mode is specified: %s' % FLAGS.mode)
  train_dataset = NERDataset(
    tokeninzer, 
    FLAGS.train_data_path,
    FLAGS.mode, 
    FLAGS.label_file, 
    FLAGS.max_seq_length)
  train_input_fn = get_dataset_fn(
      train_dataset,
      FLAGS.train_batch_size,
      is_training=True)

  input_meta_data = {
    "max_seq_length": FLAGS.max_seq_length,
    'num_labels':train_dataset.label_num,
    "train_data_size": FLAGS.train_data_size if FLAGS.train_data_size else train_dataset.data_size,
    "eval_data_size": FLAGS.eval_data_size if FLAGS.eval_data_size else eval_dataset.data_size,
    "id2label": train_dataset.id2label_map
  }

  run_bert(
      strategy,
      input_meta_data,
      bert_config,
      train_input_fn,
      eval_input_fn,)



if __name__ == '__main__':
  flags.mark_flag_as_required('bert_config_file')
  flags.mark_flag_as_required('model_dir')
  app.run(main)