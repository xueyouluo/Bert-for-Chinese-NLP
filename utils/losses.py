
'''一些标准的loss和自定义的loss'''
import tensorflow as tf


def get_contrastive_loss_fn(margin=1.0):
  def contrastive_loss(y_true,y_pred):
    # refer to https://github.com/tensorflow/addons/blob/v0.10.0/tensorflow_addons/losses/contrastive.py#L26
    y_true = tf.cast(y_true, y_pred.dtype)
    return y_true * tf.math.square(y_pred) + (1.0 - y_true) * tf.math.square(
      tf.math.maximum(margin-y_pred,0.0)
    )
  return contrastive_loss

def get_classification_loss_fn(num_classes):
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

def get_triplet_loss_fn(margin=1.0):
  def triplet_loss_fn(y_true,y_pred):
    # y_true没有被使用到
    dim = tf.shape(y_pred)[-1]//3
    anchor,pos,neg = y_pred[:,:dim], y_pred[:,dim:2*dim], y_pred[:,2*dim:]
    pos_dist = tf.reduce_sum(tf.square(anchor-pos),axis=-1,keepdims=True)
    neg_dist = tf.reduce_sum(tf.square(anchor-neg),axis=-1,keepdims=True)

    basic_loss = pos_dist - neg_dist + margin
    loss = tf.maximum(0.0,basic_loss)
    return loss
  return triplet_loss_fn