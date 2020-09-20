import os
import numpy as np
import tensorflow as tf

from seqeval.metrics import f1_score, classification_report


class NERF1Metrics(tf.keras.callbacks.Callback):
  def __init__(self, id2label, dataset, pad_value=0, digits=4, model_dir=None):
    super().__init__()
    self.id2label = id2label
    self.pad_value = pad_value
    self.digits = digits
    self.dataset = dataset
    self.best_f1 = -1.0
    self.best_save_file = None if not model_dir else os.path.join(model_dir,'best_f1')

  def convert_idx_to_name(self, y, array_indexes):
    """Convert label index to name.
    Args:
        y (np.ndarray): label index 2d array.
        array_indexes (list): list of valid index arrays for each row.
    Returns:
        y: label name list.
    """
    y = [[self.id2label[idx] for idx in row[row_indexes]] for
          row, row_indexes in zip(y, array_indexes)]
    return y

  def score(self, y_true, y_pred):
    """Calculate f1 score.
    Args:
        y_true (list): true sequences.
        y_pred (list): predicted sequences.
    Returns:
        score: f1 score.
    """
    score = f1_score(y_true, y_pred)
    print(' - f1: {:04.2f}'.format(score * 100))
    if self.digits:
        print(classification_report(y_true, y_pred, digits=self.digits))
    return  score

  def on_epoch_end(self, epoch, logs=None):
    true = []
    pred = []
    for x,y in self.dataset.as_numpy_iterator():
      y_pred = self.model.predict_on_batch(x)
      y_true = y
      y_pred = np.argmax(y_pred, -1)
      non_pad_indexes = [np.nonzero(y_true_row != self.pad_value)[0] for y_true_row in y_true]
      y_true = self.convert_idx_to_name(y_true, non_pad_indexes)
      y_pred = self.convert_idx_to_name(y_pred, non_pad_indexes)
      true.extend(y_true)
      pred.extend(y_pred)

    score = self.score(true,pred)
    if score > self.best_f1:
      if self.best_save_file:
        self.model.save_weights(self.best_save_file,save_format='tf')
      self.best_f1 = score
    print('- current best f1:{:04.2f}'.format(self.best_f1 * 100))

    logs['f1'] = score
