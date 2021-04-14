#
# Author: Ramashish Gaurav
#
# To execute this file: python hdd_train_test_eval_3d_cnn.py --run=1
#

import argparse
import datetime
import pickle
import numpy as np
import pathlib
import sys
import tensorflow as tf
import tensorflow_addons as tfa

import _init_paths

from exp_settings import hdd_exp_config
from utils.base_utils.consts import (
    HDD_TRAIN_SIDS, HDD_TEST_SIDS, LOGS_PATH, GLOBAL_BATCH_SIZE, VIDEO_FRAME_SPCS)
from utils.base_utils.data_prep_utils import get_batches_of_sequence_frames
from utils.base_utils.exp_utils import get_map_of_unique_classes_for_layer
from utils.base_utils import log
from utils.cnn_utils import get_3d_cnn_model

def train_test_eval_model(exp_cfg, load_model_epoch=None, do_eval=False):
  """
  Args:
    exp_cfg (dict): Experimental Configurations.
    load_model_epoch (int): Epoch number of the saved model to be loaded.
    do_eval (bool): Evaluate the model on test data if True else don't.
  """
  num_clss, map_lbl_to_cls, map_cls_to_lbl = get_map_of_unique_classes_for_layer(
      HDD_TRAIN_SIDS, exp_cfg["layer"])

  log.INFO("*"*80)
  log.INFO("EXPERIMENT SETTINGS: %s" % exp_cfg)
  log.INFO("For layer: %s, obtained number of classes: %s" % (
           exp_cfg["layer"], num_clss))
  log.INFO("Map of labels to classes: %s" % map_lbl_to_cls)
  log.INFO("Map of classes to labels: %s" % map_cls_to_lbl)
  log.INFO("Number of classes: %s" % num_clss)
  log.INFO("Do evaluation?: %s" % do_eval)
  log.INFO("*"*80)

  # Create a tf distribute strategy for data parallel training.
  strategy = tf.distribute.MirroredStrategy()

  log.INFO("Creating the 3D CNN training model to be executed in parallel...")
  with strategy.scope():

    # Define the appropriate loss.
    if exp_cfg["use_focal_loss"]:
      log.INFO("Using focal loss")
      loss_ob = tfa.losses.focal_loss.SigmoidFocalCrossEntropy() # Dflt Reduction None
    else:
      log.ERROR("No loss function mentioned. Exiting...")
      sys.exit()

    # Compute the loss (averaged across the entire global batch).
    def compute_loss(true_labels, pred_labels):
      per_example_loss = loss_ob(true_labels, pred_labels)
      return tf.nn.compute_average_loss(
          per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

    frm_spcs = VIDEO_FRAME_SPCS["hdd"]
    # Get the model.
    if load_model_epoch:
      log.INFO("Loading model saved at epoch: %s" % load_model_epoch)
      path = exp_cfg["exp_otpt_dir"] +"/model_%s" % load_model_epoch
      # Compile info not required as the model weights are updated explicitly
      # in `train_step` function.
      model = tf.keras.models.load_model(path, compile=False)
      start_epoch = load_model_epoch + 1
    else:
      model = get_3d_cnn_model(
        (exp_cfg["sequence_size"], int(frm_spcs["frm_hght"]*exp_cfg["dsp"]), int(
        frm_spcs["frm_wdth"]*exp_cfg["dsp"]), 3), num_clss, exp_cfg)
      start_epoch = 1
    # Get the optimizer.
    optimizer = tf.keras.optimizers.Adam(lr=exp_cfg["lr"])

    # Define the training result matrices.
    train_accuracy = tf.keras.metrics.Accuracy()
    train_precision = tf.keras.metrics.Precision(name="precision")
    train_recall = tf.keras.metrics.Recall(name="recall")

  def train_step(inputs):
    video_clips, labels = inputs

    # Pass the video data through the model to obtain predictions on it and loss.
    with tf.GradientTape() as tape:
      predictions = model(video_clips, training=True)
      loss = compute_loss(labels, predictions)

    # Get the gradients and update the trainable params.
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_accuracy.update_state(labels, predictions)
    train_precision.update_state(labels, predictions)
    train_recall.update_state(labels, predictions)

    return loss

  # Following function initiates distributed training on all the available GPUs.
  @tf.function
  def distributed_train_step(dataset_inputs):
    per_replica_losses = strategy.run(train_step, args=(dataset_inputs, ))
    return strategy.reduce(
        tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

  log.INFO("Starting training...")
  for epoch in range(start_epoch, exp_cfg["epochs"]+1):
    total_loss, num_batches = 0.0, 0
    for session_id in HDD_TRAIN_SIDS:
      log.INFO("Starting data prep... for session: %s" % session_id)
      genrtr_dataset = tf.data.Dataset.from_generator(
          get_batches_of_sequence_frames,
          output_types=(tf.float64, tf.float32),
          output_shapes=([exp_cfg["batch_size"], exp_cfg["sequence_size"],
                       int(frm_spcs["frm_hght"] * exp_cfg["dsp"]), int(frm_spcs[
                       "frm_wdth"] * exp_cfg["dsp"]), 3], [exp_cfg["batch_size"],
                       num_clss]),
          args=(session_id, exp_cfg["batch_size"], exp_cfg["sequence_size"],
                exp_cfg["layer"], exp_cfg["option"], exp_cfg["dsp"]))
      # Get the distributed training dataset.
      train_dist_dataset = strategy.experimental_distribute_dataset(
          genrtr_dataset)

      log.INFO("Distributed dataset obtained. Fitting the model on session: %s"
               % session_id)
      for train_data in train_dist_dataset:
        total_loss += distributed_train_step(train_data)
        num_batches += 1
      log.INFO("Epoch: %s, Done for session: %s" % (epoch, session_id))

    average_train_loss = total_loss / num_batches
    log.INFO("End of Epoch: %s, Accuracy: %s, Precision: %s, Recall: %s, Loss: %s"
             % (epoch, train_accuracy.result()*100, train_precision.result()*100,
                train_recall.result()*100, average_train_loss))
    train_accuracy.reset_states()
    train_precision.reset_states()
    train_recall.reset_states()

    if epoch % 1 == 0:
      log.INFO("Saving model and weights...")
      model.save(exp_cfg["exp_otpt_dir"]+"/model_%s" % epoch)
      model.save_weights(
          exp_cfg["exp_otpt_dir"]+"/model_weights_%s/weights" % epoch)
      epoch_results = {}
      for session_id in HDD_TEST_SIDS:
        log.INFO("Testing for session: %s" % session_id)
        pred_class, true_class, pred_scores = [], [], []
        batches = get_batches_of_sequence_frames(
            session_id, exp_cfg["batch_size"], exp_cfg["sequence_size"],
            exp_cfg["layer"], exp_cfg["option"], exp_cfg["dsp"],
            create_batch=True, do_shuffle=False)
        for batch in batches:
          pred_scr = model.predict_on_batch(batch[0])
          pred = np.argmax(pred_scr, axis=-1)
          true = np.argmax(batch[1], axis=-1)
          pred_class.extend(pred.tolist())
          true_class.extend(true.tolist())
          pred_scores.extend(pred_scr)

        epoch_results[session_id] = (
            pred_class, true_class, np.array(pred_scores))

        if do_eval:
          log.INFO("Evaluating model on test session ID: %s" % session_id)
          batches = get_batches_of_sequence_frames(
              session_id, exp_cfg["batch_size"], exp_cfg["sequence_size"],
              exp_cfg["layer"], exp_cfg["option"], exp_cfg["dsp"],
              create_batch=True, do_shuffle=False)
          eval_history = model.evaluate(batches, return_dict=True)
          log.INFO("Evaluate History: %s" % eval_history)

      log.INFO("Saving epoch: %s output results..." % epoch)
      pickle.dump(epoch_results,
                  open(exp_cfg["exp_otpt_dir"] + "/epoch_%s_results.p" % epoch,
                  "wb"))

if __name__=="__main__":
  # Accept the experiment run number.
  parser = argparse.ArgumentParser()
  parser.add_argument("--run", type=int, required=True, help="Experiment Run #?")
  args = parser.parse_args()
  # Change the output directory.
  hdd_exp_config["exp_otpt_dir"] = (
      hdd_exp_config["exp_otpt_dir"] + "/layer_%s/run_%s/"
      % (hdd_exp_config["layer"], args.run))

  # Create the output directory if it doesn't exist.
  pathlib.Path(hdd_exp_config["exp_otpt_dir"]).mkdir(parents=True, exist_ok=True)

  log.configure_log_handler(
      "%s_%s.log" % (
      hdd_exp_config["exp_otpt_dir"] + __file__, datetime.datetime.now()))
  train_test_eval_model(
      hdd_exp_config, load_model_epoch=hdd_exp_config["train_load_model_epoch"],
      do_eval=False)
