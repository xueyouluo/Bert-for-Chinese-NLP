'''自定义的metrics'''
import tensorflow as tf



def get_contrastive_distance_fn(feature_val=1):
  def contrastive_distance_fn(y_true,y_pred):
    return tf.reduce_mean(y_pred[y_true==feature_val])
  if feature_val == 1:
    contrastive_distance_fn.__name__ = 'postive_distance'
  else:
    contrastive_distance_fn.__name__ = 'negative_distance'
  return contrastive_distance_fn

def get_contrastive_metric_fn(margin=1.0):
  class ContrastiveAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='accuracy', **kwargs):
      super().__init__(name=name, **kwargs)
      self.total_cnt = self.add_weight(name='cnt',initializer='zeros')
      self.true_cnt = self.add_weight(name='true_cnt',initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
      '''根据margin判断相似性，大于margin认为不相似'''
      y_pred = tf.reshape(y_pred, shape=(-1, 1))
      batch_size = tf.cast(tf.shape(y_true)[0],'float32')
      y_pred = tf.cast(y_pred < margin,y_true.dtype)
      values = y_pred == y_true
      values = tf.cast(values,'float32')
      self.total_cnt.assign_add(batch_size)
      self.true_cnt.assign_add(tf.reduce_sum(values))

    def result(self):
      return self.true_cnt / self.total_cnt

    def reset_states(self):
      self.total_cnt.assign(0.0)
      self.true_cnt.assign(0.0)

  def contrastive_metric_fn(y_true, y_pred):
    y_pred = tf.reshape(y_pred, shape=(-1, 1))
    batch_size = tf.cast(tf.shape(y_true)[0],'float32')
    y_pred = tf.cast(y_pred < margin,y_true.dtype)
    values = (y_pred == y_true)
    values = tf.cast(values,'float32')
    return tf.reduce_sum(values) / batch_size
  contrastive_metric_fn.__name__ = 'accuracy'
  return contrastive_metric_fn


def get_triplet_metric_fn():
  def triplet_metric_fn(y_true, y_pred):
    '''简单的metric, 在同一个batch中，与anchor最相似的应该是对应的pos'''
    dim = tf.shape(y_pred)[-1]//3
    batch_size = tf.shape(y_pred)[0]
    anchor,pos,neg = y_pred[:,:dim], y_pred[:,dim:2*dim], y_pred[:,2*dim:]
    expand_pos = tf.expand_dims(pos,axis=0)
    # B * B * D
    expand_pos = tf.tile(expand_pos,[batch_size,1,1])
    # B * 1 * D
    expand_anchor = tf.expand_dims(anchor,axis=1)
    # B * B
    dist = tf.reduce_sum(tf.square(expand_anchor-expand_pos),axis=-1)
    truth = tf.range(batch_size)
    closest = tf.argmin(dist,axis=-1,output_type=tf.int32)
    return tf.reduce_sum(tf.cast(tf.equal(truth,closest),tf.int32)) / batch_size * 100
  # 这是为了让fit的metric名称比较直观，不然会使用函数本身的名称
  triplet_metric_fn.__name__ = 'accuracy'
  return triplet_metric_fn

def get_ams_metric_fn(forward=True):
  def ams_metric_fn(y_true,y_pred):
    dim = tf.shape(y_pred)[-1] // 2
    a,b = y_pred[:,:dim], y_pred[:,dim:]
    batch_size = tf.shape(a)[0]
    def accuracy(a,b):
      # B * 1 * D
      expand_a = tf.expand_dims(a,axis=1)
      expand_b = tf.expand_dims(b,axis=0)
      # B * B * D
      expand_b = tf.tile(expand_b,[batch_size,1,1])
      # B * B
      cos_dist = 1 - tf.reduce_sum(expand_a*expand_b,axis=-1)
      # cos_dist = 1 + tf.keras.losses.cosine_similarity(expand_a,expand_b,axis=-1)
      truth = tf.range(batch_size,dtype=tf.int64)
      closest = tf.argmin(cos_dist, axis=-1)
      return tf.reduce_sum(tf.cast(tf.equal(truth,closest),tf.int32)) / batch_size * 100
    
    if forward:
      return accuracy(a,b)
    else:
      return accuracy(b,a)

  ams_metric_fn.__name__ = ('forward_' if forward else 'backward_') + 'accuracy'
  return ams_metric_fn


