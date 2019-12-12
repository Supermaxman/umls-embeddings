import tensorflow as tf
import os
from tqdm import tqdm
import time
import numpy as np

from .Trainable import Trainable
from .ModelSaver import ModelSaver, Policy


def train(config, session, model, saver,
          train_post_step=None,
          train_post_epoch=None,
          val_post_step=None,
          val_post_epoch=None,
          global_step=0,
          max_batches_per_epoch=None):
  """
  Trains and validates
  :param config: config
  :type config: dict
  :param session: tf session
  :param model: Trainable
  :type model: Trainable
  :param train_post_step: functions to execute after each train step
  :param saver: ModelSaver
  :type saver: ModelSaver
  :param train_post_step: functions to execute after each train step
  :param train_post_epoch: functions to execute after each train epoch
  :param val_post_step: functions to execute after each validation step
  :param val_post_epoch: functions to execute after each validation epoch
  :param global_step: optional argument to pass a non-zero global step
  :param max_batches_per_epoch: optional argument to limit a training epoch to some number of batches
  """
  # init summary directories and summary writers
  if not os.path.exists(os.path.join(config['summaries_dir'], 'train')):
    os.makedirs(os.path.join(config['summaries_dir'], 'train'))
  train_summary_writer = tf.summary.FileWriter(os.path.join(config['summaries_dir'], 'train'))
  if not os.path.exists(os.path.join(config['summaries_dir'], 'val')):
    os.makedirs(os.path.join(config['summaries_dir'], 'val'))
  val_summary_writer = tf.summary.FileWriter(os.path.join(config['summaries_dir'], 'val'))

  # train
  for ep in range(config['num_epochs']):
    print('\nBegin training epoch %d' % ep)
    global_step = train_epoch(config, session, model, train_summary_writer, train_post_step, global_step, saver)
    saver.save(global_step, Policy.EPOCH, epoch=ep)
    if train_post_epoch:
      for post_epoch in train_post_epoch:
        post_epoch()

    print('\nDone epoch %d. Begin validation' % ep)
    validate(config, session, model, val_summary_writer, val_post_step, global_step)
    if val_post_epoch:
      for post_epoch in val_post_epoch:
        post_epoch()


def train_epoch(config, session, model, summary_writer, post_step, global_step, saver):
  console_update_interval = config['progress_update_interval']
  pbar = tqdm(total=console_update_interval)
  start = time.time()
  np.random.shuffle(model.data_provider.train_idx)
  session.run(
    model.data_provider.iterator.initializer,
    feed_dict={
      model.data_provider.data_indices_placeholder: model.data_provider.train_idx
    }
  )

  b = 0
  try:
    while True:
      verbose_batch = b > 0 and b % console_update_interval == 0
      # training batch
      fetched = session.run(model.fetches(True, verbose=verbose_batch))
      # update tensorboard summary
      summary_writer.add_summary(fetched[0], global_step)
      global_step += 1

      # perform post steps
      # if post_step is not None:
      #   for step in post_step:
      #     step(global_step, batch)

      # udpate progress bar
      pbar.set_description("Training Batch: %d. Loss: %.4f" % (b, fetched[1]))
      pbar.update()

      if verbose_batch:
        pbar.close()
        model.progress_update(None, fetched)
        pbar = tqdm(total=console_update_interval)

      saver.save(global_step, Policy.TIMED, seconds_since_last_save=(time.time() - start))
      b += 1
  except tf.errors.OutOfRangeError:
    pass
  pbar.close()

  return global_step


def validate(config, session, model, summary_writer, post_step, global_step):
  console_update_interval = config['val_progress_update_interval']
  pbar = tqdm(total=console_update_interval)

  session.run(
    model.data_provider.iterator.initializer,
    feed_dict={
      model.data_provider.data_indices_placeholder: model.data_provider.val_idx
    }
  )
  # validation epoch
  b = 0
  try:
    while True:
      verbose_batch = b > 0 and b % console_update_interval == 0

      fetched = session.run(model.fetches(False, verbose=verbose_batch))

      # update tensorboard summary
      summary_writer.add_summary(fetched[0], global_step)
      global_step += 1

      # perform post steps
      # if post_step is not None:
      #   for step in post_step:
      #     step(b, batch)

      # udpate progress bar
      pbar.set_description("Validation Batch: %d. Loss: %.4f" % (b, fetched[1]))
      pbar.update()

      if verbose_batch:
        pbar.close()
        model.progress_update(None, fetched)
        pbar = tqdm(total=console_update_interval)
      b += 1
  except tf.errors.OutOfRangeError:
    pass
  pbar.close()

  return global_step
