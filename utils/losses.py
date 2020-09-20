
'''一些标准的loss和自定义的loss'''
import tensorflow as tf

def _masked_labels_and_weights(y_true):
  """Masks negative values from token level labels.

  Args:
    y_true: Token labels, typically shape (batch_size, seq_len), where tokens
      with negative labels should be ignored during loss/accuracy calculation.

  Returns:
    (masked_y_true, masked_weights) where `masked_y_true` is the input
    with each negative label replaced with zero and `masked_weights` is 0.0
    where negative labels were replaced and 1.0 for original labels.
  """
  # Ignore the classes of tokens with negative values.
  mask = tf.greater_equal(y_true, 0)
  # Replace negative labels, which are out of bounds for some loss functions,
  # with zero.
  masked_y_true = tf.where(mask, y_true, 0)
  return masked_y_true, tf.cast(mask, tf.float32)


def get_additive_margin_softmax_loss(margin=0.3, scale=30.0):
  # refer to the paper "Language-agnostic BERT Sentence Embedding"
  # TODO: 实现有问题，需要修改
  def ams(a,b):
    batch_size = tf.shape(a)[0]
    # B * 1 * D
    expand_a = tf.expand_dims(a,axis=1)
    expand_b = tf.expand_dims(b,axis=0)
    # B * B * D
    expand_b = tf.tile(expand_b,[batch_size,1,1])
    # B * B
    # cos_sim = -1 * tf.keras.losses.cosine_similarity(expand_a,expand_b,axis=-1)
    cos_sim = tf.reduce_sum(expand_a*expand_b,axis=-1)
    # B * B
    mask = tf.eye(batch_size) * margin
    cos_sim = cos_sim - mask
    loss = tf.keras.losses.categorical_crossentropy(tf.eye(batch_size),scale * cos_sim, from_logits=True)
    return tf.reduce_mean(loss)

  def ams_loss(y_true, y_pred):
    dim = tf.shape(y_pred)[-1] // 2
    a,b = y_pred[:,:dim], y_pred[:,dim:]
    loss_forward = ams(a,b)
    loss_backward = ams(b,a)
    return loss_forward + loss_backward

  return ams_loss


def get_contrastive_loss_fn(margin=1.0):
  def contrastive_loss(y_true,y_pred):
    # refer to https://github.com/tensorflow/addons/blob/v0.10.0/tensorflow_addons/losses/contrastive.py#L26
    y_true = tf.cast(y_true, y_pred.dtype)
    loss = y_true * tf.math.square(y_pred) + (1.0 - y_true) * tf.math.square(
      tf.math.maximum(margin-y_pred,0.0)
    )
    return tf.reduce_mean(loss)
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
    return tf.reduce_mean(loss)
  return triplet_loss_fn

def get_ner_loss_fn(num_classes):
  def ner_loss_fn(y_true, y_pred):
    model_outputs = tf.cast(y_pred, tf.float32)
    masked_labels, masked_weights = _masked_labels_and_weights(y_true)
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        masked_labels, model_outputs, from_logits=True)
    numerator_loss = tf.reduce_sum(loss * masked_weights)
    denominator_loss = tf.reduce_sum(masked_weights)
    loss = tf.math.divide_no_nan(numerator_loss, denominator_loss)
    return loss
  
  return ner_loss_fn