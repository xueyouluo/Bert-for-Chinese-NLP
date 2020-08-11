# pylint: disable=g-classes-have-attributes

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import gin
import tensorflow as tf

from official.modeling import tf_utils
from official.nlp.bert import configs
from official.nlp.bert import bert_models
from official.nlp.modeling import models
from official.nlp.modeling import networks

from utils import model_utils

class LSTMSiamese(tf.keras.Model):
  def __init__(self, vocab_size, dim, dropout_rate=0.1, **kwargs):
    '''LSTM的baseline模型，用于验证整个pipeline能够跑通，方便调试'''
    self._self_setattr_tracking = False
    self.model = tf.keras.Sequential([
      tf.keras.layers.Embedding(vocab_size, dim, mask_zero=True),
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(dim, return_sequences=True,dropout=dropout_rate)),
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(dim)),
      tf.keras.layers.Dense(dim),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    # 为了保持输入pipeline不需要改动，这里输入加入了多余的字段
    word_ids = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, name='input_word_ids')
    mask = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, name='input_mask')
    type_ids = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, name='input_type_ids')
    
    output = self.model(word_ids)
    super(LSTMSiamese, self).__init__(inputs=[word_ids,mask,type_ids],outputs=output,**kwargs)

@tf.keras.utils.register_keras_serializable(package='Text')
class BertSiamese(tf.keras.Model):
  def __init__(self, encoder, pooling_type='MEAN', dropout_rate=0.1, **kwargs):
    self._self_setattr_tracking = False
    self._encoder = encoder
    self._config = {
        'encoder': encoder,
        'pooling_type': pooling_type,
    }

    # bert的输入是[word_ids, mask, type_ids]，这个列表是有序的
    # WARNING: 由于keras.Model其实是不支持dict形式的输入的，因此我们必须保证输入的顺序，不然可能就会得到错误的结果
    inputs = encoder.inputs
    input_word_ids,input_mask,input_type_ids = inputs
    sequence_output, cls_output = encoder(inputs)

    # 我们这里使用MEAN作为pooling
    # 我们也可以直接使用cls_output
    if pooling_type.lower() == 'mean':
      input_mask = tf.cast(input_mask, sequence_output.dtype)
      pooled_output = tf.reduce_sum(sequence_output * tf.expand_dims(input_mask,-1), axis=1) / (
        tf.reduce_sum(input_mask,axis=1,keepdims=True) + 1e-10
      )
    elif pooling_type.lower() == 'cls':
      pooled_output = cls_output
    output = tf.keras.layers.Dropout(rate=dropout_rate)(
        pooled_output)
    super(BertSiamese, self).__init__(inputs=inputs,outputs=output,**kwargs)

  @property
  def checkpoint_items(self):
    return dict(encoder=self._network)

  def get_config(self):
    return self._config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

class SiameseModel(tf.keras.Model):
  def __init__(self, siamese_model, num_labels, dropout_rate=0.1, **kwargs):
    super(SiameseModel,self).__init__(**kwargs)
    self.encoder = siamese_model

    self._config = {
      'siamese_model':siamese_model,
      "dropout_rate":dropout_rate,
      "num_labels":num_labels,
    }
    self.dense = tf.keras.layers.Dense(num_labels)

  def call(self, inputs, training=False):
    i1,i2 = inputs
    o1 = self.encoder(i1,training=training)
    o2 = self.encoder(i2,training=training)
    outputs = tf.concat([o1,o2,tf.abs(o2-o1)],axis=-1)
    outputs = self.dense(outputs)
    return outputs

  @property
  def checkpoint_items(self):
    return dict(encoder=self._network)

  def get_config(self):
    return self._config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

def siamese_model(bert_config,
                  num_labels,
                  pooling_type='MEAN',
                  mode='train'):
  encoder = bert_models.get_transformer_encoder(bert_config)
  bert_siamese = BertSiamese(
    encoder=encoder,
    pooling_type=pooling_type,
    dropout_rate=bert_config.hidden_dropout_prob)
  # bert_siamese = encoder = LSTMSiamese(bert_config.vocab_size, bert_config.hidden_size)
  bert_siamese = SiameseModel(
    bert_siamese, 
    num_labels=num_labels, 
    dropout_rate=bert_config.hidden_dropout_prob)
  return bert_siamese, encoder
