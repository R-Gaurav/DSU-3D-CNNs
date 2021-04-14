#
# Author: Ramashish Gaurav
#

import sys
import tensorflow as tf
import tensorflow_addons as tfa

from .base_utils import log

# Set the memory growth on GPU.
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

def _get_cnn_block(conv, num_filters, ker_params, exp_cfg, pool_depth=2, dp_out=0):
  """
  Returns a block of conv layer followed by pool layer.

  Args:
    conv (Tensor): The convolution object.
    num_filter (int): Number of filters in this conv layer.
    ker_params (tuple): A tuple of `ker_depth`, `ker_rows`, `ker_cols` as below->
      ker_depth (int): Number of previous frames to consider in kernel
      ker_rows (int): Number of frame rows to consider in kernel.
      ker_cols (int): Number of frame cols to consider in kernel.
    exp_cfg (dict): A dictionary of experimental configurations.
    pool_depth(int): The 3D Max Pool's temporal depth.
    dp_out (float): Fraction of input units to be dropped.
  """
  log.INFO("Layer Name: %s" % conv.name)
  dpout_lyr = conv
  if exp_cfg["inc_dpout"] and dp_out:
    log.INFO("Including dropout with dropping fraction: %s" % dp_out)
    dpout_lyr = tf.keras.layers.Dropout(dp_out)(conv)

  if exp_cfg["include_kernel_regularizer"]:
    conv = tf.keras.layers.Conv3D(
        num_filters, ker_params, padding="same", data_format="channels_last",
        activation='relu', kernel_initializer='he_uniform',
        kernel_regularizer=tf.keras.regularizers.l2(exp_cfg["rf"]))(dpout_lyr)
  else:
    conv = tf.keras.layers.Conv3D(
        num_filters, ker_params, padding="same", data_format="channels_last",
        activation='relu', kernel_initializer='he_uniform')(dpout_lyr)

  if exp_cfg["include_max_pooling"]:
    conv = tf.keras.layers.MaxPool3D(
        pool_size=(pool_depth, 2, 2), data_format="channels_last")(conv)
  else:
    log.ERROR("No downsampling method mentioned in 3D CNN!")
    sys.exit()

  return conv

def _get_dense_block(block, nn_dlyr, exp_cfg, dp_out=0, actvn="relu"):
  """
  Returns a block of dense layer.

  Args:
    block (Tensor): The model's block.
    nn_dlyr (int): Number of neurons in dense layer.
    exp_cfg (dict): A dictionary of experimental configurations.
    dp_out (float): Fraction of input units to be dropped.
    actvn (str): Activation function.
  """
  log.INFO("Layer Name: %s" % block.name)
  dpout_lyr = block
  if exp_cfg["inc_dpout"] and dp_out:
    log.INFO("Including dropout with dropping fraction: %s" % dp_out)
    dpout_lyr = tf.keras.layers.Dropout(dp_out)(block)

  if exp_cfg["include_kernel_regularizer"]:
    dense = tf.keras.layers.Dense(
        nn_dlyr, activation=actvn, kernel_initializer="he_uniform",
        kernel_regularizer=tf.keras.regularizers.l2(exp_cfg["rf"]))(dpout_lyr)
  else:
    dense = tf.keras.layers.Dense(
        nn_dlyr, activation=actvn, kernel_initializer="he_uniform")(dpout_lyr)

  return dense

def get_3d_cnn_model(inpt_shape, num_clss, exp_cfg):
  """
  Build and returns a 3D CNN model. Note that you need to compile the model
  after getting it from this function.

  Args:
    inpt_shape (tuple): A tuple of (frames_depth, img_rows, img_cols, num_chnls).
    num_clss (int): Number of classes.
    exp_cfg (dict): A dictionary of experimental configurations.

  Returns:
    training.Model
  """
  METRICS = [
      "accuracy",
      tf.keras.metrics.Precision(name="precision"),
      tf.keras.metrics.Recall(name="recall"),
      tf.keras.metrics.AUC(name="auc")
  ]

  inpt = tf.keras.Input(shape=inpt_shape)

  # 1st Conv layer..
  conv0 = _get_cnn_block(inpt, 64, (3, 3, 3), exp_cfg, pool_depth=1)
  # 2nd Conv layer.
  conv1 = _get_cnn_block(conv0, 128, (3, 3, 3), exp_cfg, pool_depth=2) #dp_out=0.1)
  # 3rd Conv layer.
  conv2 = _get_cnn_block(conv1, 256, (3, 3, 3), exp_cfg, pool_depth=2) #dp_out=0.1)
  # 4th Conv layer.
  conv3 = _get_cnn_block(conv2, 256, (3, 3, 3), exp_cfg, pool_depth=2) #dp_out=0.2)
  # 5th Conv layer.
  conv4 = _get_cnn_block(conv3, 256, (3, 3, 3), exp_cfg, pool_depth=2) #dp_out=0.2)

  # Flat layer.
  flat = tf.keras.layers.Flatten(data_format="channels_last")(conv4)

  # 1st Dense layer.
  dense0 = _get_dense_block(flat, exp_cfg["nn_dlyr"], exp_cfg, dp_out=0.25)
  # 2nd Dense layer.
  dense1 = _get_dense_block(dense0, exp_cfg["nn_dlyr"], exp_cfg, dp_out=0.25)
  # Output layer.
  output = _get_dense_block(dense1, num_clss, exp_cfg, dp_out=0.25, actvn="softmax")

  # Model.
  model = tf.keras.Model(inputs=inpt, outputs=output)
  return model
