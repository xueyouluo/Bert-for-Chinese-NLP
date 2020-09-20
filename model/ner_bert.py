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


def ner_model(bert_config,
              num_labels):
  encoder_network = bert_models.get_transformer_encoder(bert_config)
  model = models.BertTokenClassifier(
        network=encoder_network,
        num_classes=num_labels,
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=0.02),
        dropout_rate=bert_config.hidden_dropout_prob,
        output='logits')
  return model, encoder_network