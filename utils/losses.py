
'''一些标准的loss和自定义的loss'''
import tensorflow as tf


def get_additive_margin_softmax_loss(margin=0.3, scale=10.0):
  def ams(a,b):
    batch_size = tf.shape(a)[0]
    # B * 1 * D
    expand_a = tf.expand_dims(a,axis=1)
    expand_b = tf.expand_dims(b,axis=0)
    # B * B * D
    expand_b = tf.tile(expand_b,[batch_size,1,1])
    # B * B
    # TODO: 这里有个疑问，论文没有说清楚，cosine是怎么计算的
    # - scale后的vector再计算cosine，这样的话cosine的值是不变的
    # - scale后直接通过multipy再求sum，那么这个值就会扩大scale^2倍了，margin的差异就很小
    # 所以我这里还是按照cosine的方法计算，这样的作用相当于在梯度上进行了scale（跟直接loss上乘以scale有区别吗？）
    cos_sim = -1 * tf.keras.losses.cosine_similarity(expand_a,expand_b,axis=-1)
    # cos_sim = tf.reduce_sum(expand_a*expand_b,axis=-1)
    # B * B
    mask = tf.eye(batch_size) * margin
    cos_sim = cos_sim - mask
    loss = tf.keras.losses.categorical_crossentropy(tf.eye(batch_size),cos_sim, from_logits=True)
    return tf.reduce_mean(loss)

  def ams_loss(y_true, y_pred):
    dim = tf.shape(y_pred)[-1] // 2
    a,b = y_pred[:,:dim], y_pred[:,dim:]
    # 不确定这个是否有用
    a = scale * a
    b = scale * b
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