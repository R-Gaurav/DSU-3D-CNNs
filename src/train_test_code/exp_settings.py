#
# Author: Ramashish Gaurav
#
import _init_paths

from utils.base_utils.consts import (DSP_0_15, DSP_0_10, DSP_0_05,
                                     EXP_OTPT_DIR, GLOBAL_BATCH_SIZE)

hdd_exp_config = {
  "layer": 0,    # The layer of HDD dataset.
  "option": "resize",     # One of "resize" or "rescale".
  "train_load_model_epoch": None, # Train Load Model Epoch.`None` if train from scratch.

  # Make sure all the configs match for `batch_size`, `sequence_size`, `dsp`,
  # and `exp_otpt_dir`.
  "batch_size": GLOBAL_BATCH_SIZE,
  "sequence_size": 16,    # Temporal depth.
  "dsp": DSP_0_15, # DownSampling Proportion.
  "exp_otpt_dir": EXP_OTPT_DIR + (
      "/hdd_results/visuals_only/cnn_3d_outputs_dsp_0_15_ss_16/"),

  "nn_dlyr": 2048,    # Number of neurons in dense layer.
  "epochs": 16,     # Number of epochs.
  "lr": 1e-4,     # The learning rate of the model.
  "rf": 5e-5,     # Regularization factor  (Kernel Decay).
  "inc_dpout": True,      # Include dropout layer?
  "gamma": 2.0,     # Gamma parameter for focal loss.

  # Include kernel regularization in Dense and Conv if True else don't.
  "include_kernel_regularizer": True,

  # Downsampling methods. Only one of the three should be True.
  "include_max_pooling": True,

  "use_focal_loss": True, # Use focal loss with class weights if True.

  # Test Epochs
  "test_epoch_start": 7,
  "test_epoch_end": 7
}

viena2_exp_config = {
  "scenario": "Scenario1",
  "split": "random",
  "option": "resize",

  "batch_size": GLOBAL_BATCH_SIZE,
  "dsp": DSP_0_15,
  "rows": 108,
  "cols": 192,
  "exp_otpt_dir": EXP_OTPT_DIR + "/viena2_results/visuals_only/frames_dsp_0_15/",

  "nn_dlyr": 2048,
  "lr": 1e-4,
  "rf": 5e-5,
  "inc_dpout": True,
  "gamma": 2.0,
  "epochs": 64,

  "include_kernel_regularizer": True,

  "include_max_pooling": True,

  "use_focal_loss": True,
}
