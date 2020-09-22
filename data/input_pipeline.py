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
"""BERT model input pipelines."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import tensorflow as tf


def convert_example(example, suffix=''):
  input_ids, segment_ids = example
  input_mask = tf.ones_like(input_ids)
  return {
    'input_word_ids' + suffix: input_ids,
    'input_mask' + suffix: input_mask,
    'input_type_ids' + suffix: segment_ids
  }

def convert_bert_input_to_list(example):
  '''Bert的dict输入改变成list的输入，与bert encoder的输入顺序保持一致'''
  input_ids, segment_ids = example
  input_mask = tf.ones_like(input_ids)
  return [input_ids,input_mask,segment_ids]

def create_dataset(dataset,
                  batch_size,
                  is_training=True,
                  input_pipeline_context=None,
                  padding_values=0):
  dataset = tf.data.Dataset.from_generator(
    dataset.make_generator,
    args=[True],
    output_types=dataset.output_type,
    output_shapes=dataset.output_shape
    )
  
  # The dataset is always sharded by number of hosts.
  # num_input_pipelines is the number of hosts rather than number of cores.
  if input_pipeline_context and input_pipeline_context.num_input_pipelines > 1:
    dataset = dataset.shard(input_pipeline_context.num_input_pipelines,
                            input_pipeline_context.input_pipeline_id)
  
  if is_training:
    dataset = dataset.shuffle(1000)
    dataset = dataset.repeat()

  def _process_record(record):
    x1 = convert_example(record['text_a'])
    y = record['label']

    if 'text_b' in record:
      x2 = convert_example(record['text_b'])
      return ((x1,x2),y)
    return (x1,y)

  dataset = dataset.map(
    _process_record,
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
  
  dataset = dataset.padded_batch(batch_size, drop_remainder=is_training,padding_values=padding_values)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset

