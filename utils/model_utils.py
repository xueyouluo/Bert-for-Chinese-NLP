import os
import tensorflow as tf
from data import input_pipeline
from absl import app, flags, logging
from official.utils.misc import distribution_utils, keras_utils


def get_dataset_fn(raw_dataset,
                   global_batch_size,
                   is_training,
                   pad_value=0):
  """Gets a closure to create a dataset."""

  def _dataset_fn(ctx=None):
    """Returns tf.data.Dataset for distributed BERT pretraining."""
    batch_size = ctx.get_per_replica_batch_size(
        global_batch_size) if ctx else global_batch_size
    dataset = input_pipeline.create_dataset(
        raw_dataset,
        batch_size,
        is_training=is_training,
        input_pipeline_context=ctx,
        padding_values=pad_value)
    return dataset

  return _dataset_fn

def duplicate_model_inputs(model,name_suffix='_v2'):
  inputs = []
  for tensor,name in zip(model.inputs,model.input_names):
    inputs.append(tf.keras.layers.Input(tensor.shape[1:],dtype=tensor.dtype,name=name+name_suffix))
  return inputs


def run_keras_compile_fit(model_dir,
                          strategy,
                          model_fn,
                          train_input_fn,
                          eval_input_fn,
                          loss_fn,
                          metric_fn,
                          init_checkpoint,
                          epochs,
                          steps_per_epoch,
                          steps_per_loop,
                          eval_steps,
                          monitor='val_loss',
                          training_callbacks=True,
                          custom_callbacks=None):
  """Runs BERT classifier model using Keras compile/fit API."""
  # tf.config.set_soft_device_placement(True)
  # tf.debugging.experimental.enable_dump_debug_info(
  #   '/tmp/my-tfdbg-dumps', tensor_debug_mode="FULL_HEALTH")
  with strategy.scope():
    training_dataset = train_input_fn()
    evaluation_dataset = eval_input_fn() if eval_input_fn else None
    bert_model, sub_model = model_fn()

    optimizer = bert_model.optimizer

    if init_checkpoint:
      logging.info('Restore from {}'.format(init_checkpoint))
      checkpoint = tf.train.Checkpoint(model=sub_model)
      checkpoint.restore(init_checkpoint).assert_existing_objects_matched()

    if metric_fn and not isinstance(metric_fn, (list, tuple)):
      metric_fn = [metric_fn]
    bert_model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[fn() for fn in metric_fn] if metric_fn else None)
        # steps_per_loop这个是个坑，我在训练的时候没有设置，一直报init_value的错误，我还一直以为是模型的错误
        # -_-!!!
        # experimental_steps_per_execution=steps_per_loop

    summary_dir = os.path.join(model_dir, 'summaries')
    summary_callback = tf.keras.callbacks.TensorBoard(summary_dir)
    checkpoint = tf.train.Checkpoint(model=bert_model, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=model_dir,
        max_to_keep=3,
        step_counter=optimizer.iterations,
        checkpoint_interval=0)
    checkpoint_callback = keras_utils.SimpleCheckpoint(checkpoint_manager)

    #save best
    best_dir = os.path.join(model_dir, 'best/model')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=best_dir,
      save_weights_only=True,
      monitor=monitor,
      mode='min' if ('loss' in monitor or 'error' in monitor) else 'max',
      save_best_only=True)

    if training_callbacks:
      if custom_callbacks is not None:
        custom_callbacks += [summary_callback, checkpoint_callback, model_checkpoint_callback]
      else:
        custom_callbacks = [summary_callback, checkpoint_callback, model_checkpoint_callback]
    logging.info('start to train')
    history = bert_model.fit(
        x=training_dataset,
        validation_data=evaluation_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_steps=eval_steps,
        callbacks=custom_callbacks)
    stats = {'total_training_steps': steps_per_epoch * epochs}
    if 'loss' in history.history:
      stats['train_loss'] = history.history['loss'][-1]
    if 'val_accuracy' in history.history:
      stats['eval_metrics'] = history.history['val_accuracy'][-1]
    return bert_model, stats