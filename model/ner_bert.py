# pylint: disable=g-classes-have-attributes

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import gin
import tensorflow as tf
import tensorflow_addons as tfa

from official.modeling import tf_utils
from official.nlp.bert import configs
from official.nlp.bert import bert_models
from official.nlp.modeling import models
from official.nlp.modeling import networks

from utils import model_utils
from utils.losses import _masked_labels_and_weights, ner_loss_fn

class BertNERCRFModel(tf.keras.Model):
    def __init__(self,
                network,
                num_classes,
                initializer='glorot_uniform',
                dropout_rate=0.1,
                use_crf=True,
                **kwargs):
        super(BertNERCRFModel,self).__init__(**kwargs)
        self.network = network
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.classifier = tf.keras.layers.Dense(
            num_classes,
            activation=None,
            kernel_initializer=initializer,
            name='predictions/transform/logits')
        self.use_crf = use_crf
        self.transition_params = self.add_weight(
            shape=(num_classes,num_classes),
            initializer=initializer,
            name='predictions/transform/transition'
        )
        self.num_classes = num_classes

    def call(self, inputs, training=False):
        sequence_output, _ = self.network(inputs)
        if training:
            sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        if self.use_crf:
            input_mask = inputs['input_mask']
            mask = tf.cast(tf.equal(input_mask, 1),tf.int32)
            seq_len = tf.reduce_sum(mask,axis=-1)
            decode_tags,_ = tfa.text.crf_decode(logits,self.transition_params,seq_len)
        else:
            decode_tags = tf.argmax(logits, axis=-1)
        # return {
        #     "logits":logits,
        #     "decode_tags":decode_tags
        # }
        return tf.one_hot(decode_tags,self.num_classes), logits

    def train_step(self, data):
        x,y = data
        input_mask = x['input_mask']
        mask = tf.cast(tf.equal(input_mask, 1),tf.int32)
        seq_len = tf.reduce_sum(mask,axis=-1)
        masked_labels, masked_weights = _masked_labels_and_weights(y)
        with tf.GradientTape() as tape:
            decode_tags, logits = self(x,training=True)
            if self.use_crf:
                log_likelihood,_ = tfa.text.crf_log_likelihood(
                    logits,
                    masked_labels,
                    seq_len,
                    transition_params=self.transition_params)
                loss = tf.reduce_mean(-log_likelihood)
            else:
                loss = ner_loss_fn(y,logits)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, decode_tags)

        # Return a dict mapping metric names to current value
        metrics = {m.name: m.result() for m in self.metrics}
        metrics['loss'] = loss
        return metrics

def ner_model(bert_config,
              num_labels,
              use_crf=False):
  encoder_network = bert_models.get_transformer_encoder(bert_config)

  if use_crf:
    model = BertNERCRFModel(encoder_network,num_labels)
  else:
    model = models.BertTokenClassifier(
        network=encoder_network,
        num_classes=num_labels,
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=0.02),
        dropout_rate=bert_config.hidden_dropout_prob,
        output='logits')
  return model, encoder_network