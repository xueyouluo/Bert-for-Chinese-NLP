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
"""BERT classification or regression finetuning runner in TF 2.x."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import math
import os

from absl import app
from absl import flags
from absl import logging
import gin
import tensorflow as tf
from official.modeling import performance
from official.nlp import optimization
from official.nlp.bert import bert_models
from official.nlp.bert import common_flags
from official.nlp.bert import tokenization
from official.nlp.bert import configs as bert_configs
from official.nlp.bert import model_saving_utils
from official.utils.misc import distribution_utils
from official.utils.misc import keras_utils

from data.dataset import SiameseDataset, SentencesDataset
from data import input_pipeline
from model import siamese_bert

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
flags.DEFINE_enum('model_type', 'bert', ['bert','siamese'],'model type')
flags.DEFINE_integer('max_seq_length', 512, 'Max sequence length, default 512, you can use smaller number to save memory')
flags.DEFINE_string('vocab_file', None,
                    'The vocabulary file that the BERT model was trained on.')
flags.DEFINE_string('label_file', None,
                    'The label file that to map the labels.')
flags.DEFINE_integer('train_data_size', None, 'size of training dataset.')
flags.DEFINE_integer('eval_data_size', None, 'size of evaluation dataset.')

common_flags.define_common_bert_flags()

FLAGS = flags.FLAGS



def get_loss_fn(num_classes):
  """Gets the classification loss function."""

  def classification_loss_fn(labels, logits):
    """Classification loss."""
    labels = tf.squeeze(labels)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    one_hot_labels = tf.one_hot(
        tf.cast(labels, dtype=tf.int32), depth=num_classes, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(
        tf.cast(one_hot_labels, dtype=tf.float32) * log_probs, axis=-1)
    return tf.reduce_mean(per_example_loss)

  return classification_loss_fn

def get_dataset_fn(raw_dataset,
                   global_batch_size,
                   is_training):
  """Gets a closure to create a dataset."""

  def _dataset_fn(ctx=None):
    """Returns tf.data.Dataset for distributed BERT pretraining."""
    batch_size = ctx.get_per_replica_batch_size(
        global_batch_size) if ctx else global_batch_size
    dataset = input_pipeline.create_siamese_dataset(
        raw_dataset,
        batch_size,
        is_training=is_training,
        input_pipeline_context=ctx)
    return dataset

  return _dataset_fn

def run_bert_classifier(strategy,
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
  """Run BERT siamese training using low-level API."""
  max_seq_length = input_meta_data['max_seq_length']
  num_classes = input_meta_data.get('num_labels', 1)
  logging.info(f'class num {num_classes}')
  is_regression = num_classes == 1

  def _get_model():
    """Gets a siamese model."""
    if FLAGS.model_type == 'siamese':
      model, core_model = (
        siamese_bert.siamese_model(
            bert_config,
            num_classes))
    else:
      model, core_model = (
        bert_models.classifier_model(
            bert_config,
            num_classes,
            max_seq_length))
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
  loss_fn = (tf.keras.losses.MeanSquaredError() if is_regression
             else get_loss_fn(num_classes))

  # Defines evaluation metrics function, which will create metrics in the
  # correct device and strategy scope.
  if custom_metrics:
    metric_fn = custom_metrics
  elif is_regression:
    metric_fn = functools.partial(
        tf.keras.metrics.MeanSquaredError,
        'mean_squared_error',
        dtype=tf.float32)
  else:
    metric_fn = functools.partial(
        tf.keras.metrics.SparseCategoricalAccuracy,
        'accuracy',
        dtype=tf.float32)

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
      training_callbacks=training_callbacks,
      custom_callbacks=custom_callbacks)

def run_keras_compile_fit(model_dir,
                          strategy,
                          model_fn,
                          train_input_fn,
                          eval_input_fn,
                          loss_fn,
                          metric_fn,
                          init_checkpoint,
                          epochs,
                          steps_per_epoch,
                          steps_per_loop,
                          eval_steps,
                          training_callbacks=True,
                          custom_callbacks=None):
  """Runs BERT classifier model using Keras compile/fit API."""
  # tf.config.set_soft_device_placement(True)
  # tf.debugging.experimental.enable_dump_debug_info(
  #   '/tmp/my-tfdbg-dumps', tensor_debug_mode="FULL_HEALTH")
  with strategy.scope():
    training_dataset = train_input_fn()
    evaluation_dataset = eval_input_fn() if eval_input_fn else None
    bert_model, sub_model = model_fn()

    optimizer = bert_model.optimizer

    if init_checkpoint:
      checkpoint = tf.train.Checkpoint(model=sub_model)
      checkpoint.restore(init_checkpoint).assert_existing_objects_matched()

    if not isinstance(metric_fn, (list, tuple)):
      metric_fn = [metric_fn]
    bert_model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[fn() for fn in metric_fn])
        # steps_per_loop这个是个坑，我在训练的时候没有设置，一直报init_value的错误，我还一直以为是模型的错误
        # -_-!!!
        # experimental_steps_per_execution=steps_per_loop

    summary_dir = os.path.join(model_dir, 'summaries')
    summary_callback = tf.keras.callbacks.TensorBoard(summary_dir)
    checkpoint = tf.train.Checkpoint(model=bert_model, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=model_dir,
        max_to_keep=3,
        step_counter=optimizer.iterations,
        checkpoint_interval=0)
    checkpoint_callback = keras_utils.SimpleCheckpoint(checkpoint_manager)

    #save best
    best_dir = os.path.join(model_dir, 'best/model')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=best_dir,
      save_weights_only=True,
      monitor='val_accuracy',
      mode='max',
      save_best_only=True)

    if training_callbacks:
      if custom_callbacks is not None:
        custom_callbacks += [summary_callback, checkpoint_callback, model_checkpoint_callback]
      else:
        custom_callbacks = [summary_callback, checkpoint_callback, model_checkpoint_callback]
    logging.info('start to train')
    history = bert_model.fit(
        x=training_dataset,
        validation_data=evaluation_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_steps=eval_steps,
        callbacks=custom_callbacks)
    stats = {'total_training_steps': steps_per_epoch * epochs}
    if 'loss' in history.history:
      stats['train_loss'] = history.history['loss'][-1]
    if 'val_accuracy' in history.history:
      stats['eval_metrics'] = history.history['val_accuracy'][-1]
    return bert_model, stats

def run_bert(strategy,
             input_meta_data,
             model_config,
             train_input_fn=None,
             eval_input_fn=None,
             init_checkpoint=None,
             custom_callbacks=None,
             custom_metrics=None):
  """Run BERT training."""
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

  if not custom_callbacks:
    custom_callbacks = []

  if FLAGS.log_steps:
    custom_callbacks.append(
        keras_utils.TimeHistory(
            batch_size=FLAGS.train_batch_size,
            log_steps=FLAGS.log_steps,
            logdir=FLAGS.model_dir))

  trained_model, _ = run_bert_classifier(
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
      custom_callbacks=custom_callbacks,
      custom_metrics=custom_metrics)

  if FLAGS.model_export_path:
    model_saving_utils.export_bert_model(
        FLAGS.model_export_path, model=trained_model)
  return trained_model


def custom_main(custom_callbacks=None, custom_metrics=None):
  """Run classification or regression.

  Args:
    custom_callbacks: list of tf.keras.Callbacks passed to training loop.
    custom_metrics: list of metrics passed to the training loop.
  """
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

  if FLAGS.model_type == 'siamese':
    data_fn = SiameseDataset
  else:
    data_fn = SentencesDataset
  
  eval_dataset = data_fn(
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
  train_dataset = data_fn(
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
    "eval_data_size": FLAGS.eval_data_size if FLAGS.eval_data_size else eval_dataset.data_size

  }
  run_bert(
      strategy,
      input_meta_data,
      bert_config,
      train_input_fn,
      eval_input_fn,
      custom_callbacks=custom_callbacks,
      custom_metrics=custom_metrics)


def main(_):
  custom_main(custom_callbacks=None, custom_metrics=None)


if __name__ == '__main__':
  flags.mark_flag_as_required('bert_config_file')
  flags.mark_flag_as_required('model_dir')
  app.run(main)