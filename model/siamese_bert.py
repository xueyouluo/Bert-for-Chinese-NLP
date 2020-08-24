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
  def __init__(self, vocab_size, dim, dropout_rate=0.1, norm=False, **kwargs):
    '''基于LSTM的baseline模型，用于验证整个pipeline能够跑通，方便调试'''
    logging.info('Using LSTM model...')
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
    if norm:
      output = tf.nn.l2_normalize(output)
    super(LSTMSiamese, self).__init__(inputs=[word_ids,mask,type_ids],outputs=output,**kwargs)

@tf.keras.utils.register_keras_serializable(package='Text')
class BertSiamese(tf.keras.Model):
  def __init__(self, encoder, pooling_type='MEAN', dropout_rate=0.1, norm=False, **kwargs):
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
    if norm:
      pooled_output = tf.nn.l2_normalize(pooled_output)
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

class SiameseAMSModel(tf.keras.Model):
  '''AMS is short for Additive Margin Softmax'''
  def __init__(self, encoder, **kwargs):
    super().__init__(**kwargs)
    self.encoder = encoder

  def call(self, inputs):
    assert len(inputs) == 2, 'inputs must have 2 values'
    a,b = inputs
    a = self.encoder(a)
    b = self.encoder(b)
    concat_val = tf.concat([a,b],axis=-1)
    return concat_val

class SiameseClassifierModel(tf.keras.Model):
  def __init__(self, encoder, num_labels, dropout_rate=0.1, **kwargs):
    super(SiameseClassifierModel,self).__init__(**kwargs)
    self.encoder = encoder

    self._config = {
      'encoder':encoder,
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

class SiameseContrastiveModel(tf.keras.Model):
  def __init__(self, encoder, **kwargs):
    super().__init__(**kwargs)
    self.encoder = encoder

  def call(self, inputs, training=False):
    assert len(inputs) == 2, 'inputs must have 2 values'
    a,b = inputs
    a = self.encoder(a,training=training)
    b = self.encoder(b,training=training)
    val = tf.linalg.norm(a-b,axis=1)
    return val

class SiameseTripletModel(tf.keras.Model):
  def __init__(self, encoder, **kwargs):
    super().__init__(**kwargs)
    self.encoder = encoder

  def call(self, inputs, training=False):
    if len(inputs) == 3:
      anchor,pos,neg = inputs
      anchor = self.encoder(anchor,training=training)
      pos = self.encoder(pos,training=training)
      neg = self.encoder(neg,training=training)
    elif len(inputs) == 2:
      anchor, pos = inputs
      anchor = self.encoder(anchor,training=training)
      pos = self.encoder(pos,training=training)

      batch_size = tf.shape(pos)[0]
      expand_pos = tf.expand_dims(pos,axis=0)
      # B * B * D
      expand_pos = tf.tile(expand_pos,[batch_size,1,1])
      # B * 1 * D
      expand_anchor = tf.expand_dims(anchor,axis=1)
      # B * B
      dist = tf.reduce_sum(tf.square(expand_anchor-expand_pos),axis=-1)
      # B * B
      mask = tf.eye(batch_size) * 1e20
      mask_dist = mask + dist
      indices = tf.argmin(mask_dist,axis=-1)
      # TODO：不确定是否需要stop gradient
      incices = tf.stop_gradient(indices)
      neg = tf.gather(pos,indices)
    else:
      raise ValueError('inputs length must be 2 or 3')
    concat_val = tf.concat([anchor,pos,neg],axis=-1)
    return concat_val

def siamese_model(bert_config,
                  num_labels,
                  siamese_type='classify',
                  pooling_type='CLS'):
  encoder = bert_models.get_transformer_encoder(bert_config)
  bert_siamese = BertSiamese(
    encoder=encoder,
    pooling_type=pooling_type,
    dropout_rate=bert_config.hidden_dropout_prob,
    norm=True if siamese_type=='ams' else False
  )

  # Uncomment following line to get a baseline model to debug your network
  # bert_siamese = encoder = LSTMSiamese(
  #   bert_config.vocab_size, bert_config.hidden_size,norm=True if siamese_type=='ams' else False)
  if siamese_type == 'classify':
    bert_siamese = SiameseClassifierModel(
      bert_siamese, 
      num_labels=num_labels, 
      dropout_rate=bert_config.hidden_dropout_prob)
  elif siamese_type == 'triplet':
    bert_siamese = SiameseTripletModel(bert_siamese)
  elif siamese_type == 'contrastive':
    bert_siamese = SiameseContrastiveModel(bert_siamese)
  elif siamese_type == 'ams':
    bert_siamese = SiameseAMSModel(bert_siamese)
  else:
    raise ValueError(f'Siamese type {siamese_type} not supported!!')
  return bert_siamese, encoder
