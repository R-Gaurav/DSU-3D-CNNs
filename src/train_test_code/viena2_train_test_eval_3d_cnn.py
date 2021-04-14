#
# Author: Ramashish Gaurav
#
# To execute this file: python viena2_train_test_eval_3d_cnn.py --run=1
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

from exp_settings import viena2_exp_config
from utils.base_utils import log
from utils.base_utils.consts import (
    GLOBAL_BATCH_SIZE, VIDEO_FRAME_SPCS, VIENA2_SCN_MAP)
from utils.base_utils.data_prep_utils import get_batches_of_viena2_clips
from utils.cnn_utils import get_3d_cnn_model

def train_test_model(exp_cfg):
  """
  Args:
    exp_cfg (dict): VIENA2 Experimental config.
  """
  log.INFO("*" * 80)
  log.INFO("VIENA2 EXPERIMENT SETTINGS: %s" % exp_cfg)
  log.INFO("*" * 80)

  # Create a tf distribute strategy for data parallel training.
  strategy = tf.distribute.MirroredStrategy()

  log.INFO("Creating the 3D CNN training model to be executed in parallel...")
  with strategy.scope():

    # Define the appropriate loss.
    if exp_cfg["use_focal_loss"]:
      log.INFO("Using focal loss")
      loss_ob = tfa.losses.focal_loss.SigmoidFocalCrossEntropy() # Dflt Reduction None.
    else:
      log.ERROR("No loss function mentioned. Exiting...")
      sys.exit()

    # Compute the loss (averaged across the entire global batch).
    def compute_loss(true_labels, pred_labels):
      per_example_loss = loss_ob(true_labels, pred_labels)
      return tf.nn.compute_average_loss(
          per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

    # Get the model.
    frm_spcs = VIDEO_FRAME_SPCS["viena2"]
    num_clss = len(VIENA2_SCN_MAP[exp_cfg["scenario"]].keys())
    model = get_3d_cnn_model(
        (16, exp_cfg["rows"], exp_cfg["cols"], 3), num_clss, exp_cfg)
    # Get the optimizer.
    optimizer = tf.keras.optimizers.Adam(lr=exp_cfg["lr"])

    # Define the training result metrics.
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

    log.INFO("Updating metrics...")
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
  for epoch in range(1, exp_cfg["epochs"]+1):
    total_loss, num_batches = 0.0, 0
    genrtr_dataset = tf.data.Dataset.from_generator(
        get_batches_of_viena2_clips,
        output_types=(tf.float64, tf.float32),
        output_shapes=([exp_cfg["batch_size"], 16, exp_cfg["rows"],
                       exp_cfg["cols"], 3], [exp_cfg["batch_size"], num_clss]),
        args=(exp_cfg["scenario"], exp_cfg["split"], "training",
              exp_cfg["batch_size"], exp_cfg["option"], exp_cfg["rows"],
              exp_cfg["cols"]))
    # Get the distributed training dataset.
    train_dist_dataset = strategy.experimental_distribute_dataset(
        genrtr_dataset)

    log.INFO("Distributed dataset obtained. Fittting the model... ")
    for train_data in train_dist_dataset:
      total_loss += distributed_train_step(train_data)
      num_batches += 1

    average_train_loss = total_loss / num_batches
    log.INFO("Epoch: %s done. Accuracy: %s, Precision: %s, Recall: %s, Loss: %s"
              % (epoch, train_accuracy.result()*100, train_precision.result()*100,
              train_recall.result()*100, average_train_loss))
    train_accuracy.reset_states()
    train_precision.reset_states()
    train_recall.reset_states()

    if epoch % 1 == 0:
      log.INFO("Saving model and weights...")
      model.save(exp_cfg["exp_otpt_dir"] + "/model_%s" % epoch)
      model.save_weights(
          exp_cfg["exp_otpt_dir"] + "/model_weights_%s/weights" % epoch)
      batches = get_batches_of_viena2_clips(
          exp_cfg["scenario"], exp_cfg["split"], "test", exp_cfg["batch_size"],
          exp_cfg["option"], exp_cfg["rows"], exp_cfg["cols"])
      pred_clss, true_clss, pred_scores = [], [], []

      for batch in batches:
        pred_scr = model.predict_on_batch(batch[0])
        pred = np.argmax(pred_scr, axis=-1)
        pred_clss.extend(pred.tolist())
        true_clss.extend(batch[1].tolist())
        pred_scores.extend(pred_scr)

      log.INFO("Saving epoch: %s output results..." % epoch)
      pickle.dump((pred_clss, true_clss, np.array(pred_scores)),
                  open(exp_cfg["exp_otpt_dir"] + "/epoch_%s_results.p" % epoch,
                       "wb"))

if __name__ == "__main__":
  # Accept the experiment run number.
  parser = argparse.ArgumentParser()
  parser.add_argument("--run", type=int, required=True, help="Experiment Run #?")
  args = parser.parse_args()
  #viena2_exp_config["exp_otpt_dir"] = (
  #    viena2_exp_config["exp_otpt_dir"]+"/%s/%s_run_%s/"
  #    % (viena2_exp_config["scenario"], viena2_exp_config["split"], args.run))
  exp_otpt_dir = viena2_exp_config["exp_otpt_dir"]
  if args.run == 0: # Run for all the 5 Runs.
    for r in range(1, 6):
      # Change the output directory..
      viena2_exp_config["exp_otpt_dir"] = (
          "%s/%s/%s_run_%s/" % (exp_otpt_dir, viena2_exp_config["scenario"],
                                viena2_exp_config["split"], r))

      # Create the output directory if it doesn't exist.
      pathlib.Path(viena2_exp_config["exp_otpt_dir"]).mkdir(
          parents=True, exist_ok=True)

      log.configure_log_handler(
        "%s_%s.log" % (
        viena2_exp_config["exp_otpt_dir"] + __file__, datetime.datetime.now()))
      train_test_model(viena2_exp_config)
