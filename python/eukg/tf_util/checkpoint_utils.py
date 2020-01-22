
import tensorflow as tf
from bert import modeling
# import hedgedog.tf.models.bert as modeling


def init_from_checkpoint(init_checkpoint):
  t_vars = tf.trainable_variables()
  (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
    t_vars,
    init_checkpoint
  )
  tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

  for trainable_var in t_vars:
    init_string = ""
    if trainable_var.name in initialized_variable_names:
      init_string = '*INIT_FROM_CKPT*'
    print(f'{trainable_var.name}: {trainable_var.get_shape()} {init_string}')

