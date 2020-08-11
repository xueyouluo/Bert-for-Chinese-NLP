import tensorflow as tf


def duplicate_model_inputs(model,name_suffix='_v2'):
  inputs = []
  for tensor,name in zip(model.inputs,model.input_names):
    inputs.append(tf.keras.layers.Input(tensor.shape[1:],dtype=tensor.dtype,name=name+name_suffix))
  return inputs