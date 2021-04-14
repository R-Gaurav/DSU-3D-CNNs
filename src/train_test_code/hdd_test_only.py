#
# Author: Ramashish Gaurav
#

import argparse
import datetime
import pickle
import numpy as np
import pathlib
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

def test_model(exp_cfg):
  """
  Args:
    exp_cfg (dict): Experimental Configurations.
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
  log.INFO("*"*80)
  epoch_start = exp_cfg["test_epoch_start"]
  epoch_end = exp_cfg["test_epoch_end"]

  for epoch in range(epoch_start, epoch_end+1):
    path = exp_cfg["exp_otpt_dir"] +"/model_%s" % epoch
    model = tf.keras.models.load_model(path, compile=False)

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

    log.INFO("Saving epoch: %s output results..." % epoch)
    pathlib.Path(
        hdd_exp_config["exp_otpt_dir"] + "/ns_test_only/").mkdir(
        parents=True, exist_ok=True)
    pickle.dump(epoch_results,
                open(exp_cfg["exp_otpt_dir"] +
                "/ns_test_only/epoch_%s_results_ns_test_only.p" % epoch, "wb"))

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
  test_model(hdd_exp_config)
