#
# Author: Ramashish Gaurav
#

import datetime

import _init_paths

from utils.base_utils.consts import (HDD_TRAIN_SIDS, HDD_TEST_SIDS, LOGS_PATH,
    DSP_0_15, DSP_0_10, DSP_0_05, VIENA2_DATA_DIR)
from utils.base_utils.data_prep_utils import (get_down_samp_frames_of_session,
    get_down_samp_frames_of_clip)
from utils.base_utils.exp_utils import (get_map_of_unique_classes_for_layer,
    get_viena2_train_test_clip_ids)
from utils.base_utils import log

if __name__ == "__main__":

  ############################ FOR HDD DATASET ###############################
  #log.configure_log_handler(
  #    "%s_%s.log" % (LOGS_PATH + __file__, datetime.datetime.now()))
  #for session_id in HDD_TRAIN_SIDS + HDD_TEST_SIDS:
  #  ds_frames = get_down_samp_frames_of_session(
  #      session_id, "resize", down_samp_prop=DSP_0_05)

  #get_map_of_unique_classes_for_layer(HDD_TRAIN_SIDS, 0)
  ############################################################################


  ########################### FOR VIENA2 DATASET #############################
  scenario, split, rows, cols = "Scenario5", "random", 108, 192
  LOGS_PATH = VIENA2_DATA_DIR + "/frames/frames_dsp_0_15/%s/" % scenario
  log.configure_log_handler(
      "%s_%s.log" % (LOGS_PATH + __file__, datetime.datetime.now()))
  log.INFO("VIENA2 Dataset creation for scenario: %s" % scenario)
  train_ids, _, test_ids, _ = get_viena2_train_test_clip_ids(scenario, split)
  for clip_id in train_ids + test_ids:
    clip_ds_frames = get_down_samp_frames_of_clip(
        clip_id, scenario, "resize", rows, cols)
    if clip_ds_frames.shape[0] != 16:
      log.WARN("Clip ID: %s does not have 16 frames but: %s number of frames."
               % (clip_id, clip_ds_frames.shape[0]))
  ############################################################################
