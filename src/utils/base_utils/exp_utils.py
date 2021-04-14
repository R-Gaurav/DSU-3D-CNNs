#
# This file contains the general utility functions for the experiment.
#
# Author: Ramashish Gaurav
#


from collections import defaultdict, Counter
from datetime import datetime, timedelta
from joblib import Parallel, delayed
from sklearn.metrics import (
    average_precision_score, precision_score, recall_score)
from sklearn.preprocessing import label_binarize

import dateutil.parser as dp
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import random
import re
import seaborn as sn
import subprocess
import sys

from . import log
from .consts import (HDD_SAVED_INDEX_PKL_PATH, HDD_VIDEO_PATH, HDD_EVENT_TYPES,
                     HDD_ALL_FRAMES_ALL_VIDS_LABELS_PATH, HDD_TRAIN_SIDS,
                     HDD_TEST_SIDS, MAP_LBL_TO_CLS_LYR_0, MAP_LBL_TO_CLS_LYR_1,
                     COMMON_DATA_DIR, VIENA2_TRAIN_TEST_SPLITS, VIENA2_DATA_DIR,
                     VIENA2_SCN_MAP, HDD, VIENA2, EXP_OTPT_DIR, HDD_LYR_CLS_MAP,
                     HDD_DATA_BASE_DIR)

def get_saved_index_pkl():
  return pickle.load(open(HDD_SAVED_INDEX_PKL_PATH, "rb"))

def get_layer_and_event_type_df(layer, event_type=None):
  """
  Returns the data frame of session_ids (each video as a unique session_id), the
  corresponding start time and end time of the `layer` and `event_type`. The
  start time and end time are mentioned in milliseconds.

  Args:
    layer (int): The layer id of the driver behaviour. Ex: 0 => Goal Oriented.
    event_type (int): The corresponding event type in `layer`. For 0 layer i.e.
        Goal Oriented action, the even type could be 7 i.e. left turn.

  Note that the `event_type` if given should fall in `layer`, if not, the data
  frame returned would be empty.

  Returns:
    pandas.DataFrame
  """
  sidx = get_saved_index_pkl()
  df = sidx["events_pd"]
  if event_type is not None:
    return df[(df["layer"] == layer) & (df["event_type"] == event_type)]
  else:
    return df[df["layer"] == layer]

def get_epochs_from_session_id_st_en(session_id, start, end):
  """
  Returns epochs for the start and end of an event for a particular session.

  Note that `session_id_fname` stores the actual time of the start of the
  session. At approximately the same time the CAN bus data starts to be stored.

  Args:
    session_id (str): The session id of video.
    start (int): The start time in milliseconds of the clip.
    end (int): The end time in milliseconds of the clip.

  Returns:
    int, int : The epoch for start, The epoch for end.
  """
  session_id_fname = os.listdir(HDD_VIDEO_PATH.format(session_id))[0]
  year, month, day, hour, mins, secs = session_id_fname.split("_")[0].split("-")
  sid_dt = datetime(
      int(year), int(month), int(day), int(hour), int(mins), int(secs))
  start_epoch = int((sid_dt + timedelta(seconds=start/1000)).strftime("%s"))
  end_epoch = int((sid_dt + timedelta(seconds=end/1000)).strftime("%s"))
  return start_epoch, end_epoch

def get_epochs_from_iso_timestamps_lst(iso_ts_lst):
  """
  Returns a list of epochs for ISO timestamps. Note that this function get rids
  of the TimeZone. The ISO time format it accepts is:
    "YYYY-MM-DDThh:mm:ss.sTZD" e.g.: 1997-07-16T19:20:30.45+01:00
  Here TZD is the Time Zone Designator (Z or +hh:mm or -hh:mm) and this function
  gets rid of this TZD or +hh:mm or -hh:mm.

  Args:
    iso_ts_lst ([str]): A list of ISO timestamps.

  Returns:
    numpy.ndarray: An array of corresponding epoch.
  """
  length = len(iso_ts_lst)
  iso_epoch_arr =  np.zeros(length)
  for i in range(length):
    iso_epoch_arr[i] = int(dp.parse(iso_ts_lst[i][:-6]).strftime("%s"))

  return iso_epoch_arr

def basic_plot(y_lst, y_legend_lst, title, y_label="", x_label="", fs=(6, 4),
               close=False, fnt_sze=14, put_title=True, put_grid=False,
               x_ticks_lst=None, lloc=None, invisible_indices=[], thick_lgnd=None):
  """
  Plots a basic plot. Now it evolved to become complex... ;)

  Args:
    y_lst ([[float]]): A list of numbers to be plotted.
    y_legend_lst ([str]): A list of corresponding legends.
    title (str): Title of the plot.
    y_label (str): Y-axis label of the numbers to be plotted.
    x_label (str): X-axis label.
    fs (tuple): A tuple of two integers denoting the figure size.
    close (bool): Close the plot and not show it?
  """
  fig, ax = plt.subplots(figsize=fs)
  fig.subplots_adjust(top=0.92)
  if put_title:
    fig.suptitle(title, fontsize=fnt_sze)
  for y, legend in zip(y_lst, y_legend_lst):
    if legend == thick_lgnd:
      ax.plot(y, label=legend, linewidth=3, color="k")
    else:
      ax.plot(y, label=legend, linewidth=2)
  if lloc:
    ax.legend(loc=lloc, prop={"size":fnt_sze}, framealpha=0.5)
  else:
    ax.legend(prop={"size":fnt_sze}, framealpha=0.5)
  ax.set_ylabel(y_label, fontsize=fnt_sze, labelpad=1)
  ax.set_xlabel(x_label, fontsize=fnt_sze, labelpad=1)
  if x_ticks_lst:
    plt.xticks(x_ticks_lst, fontsize=fnt_sze)
    if invisible_indices:
      xticks = ax.xaxis.get_major_ticks()
      for i in invisible_indices:
        print(i, "Invisible")
        xticks[i].label1.set_visible(False)
  else:
    plt.xticks(fontsize=fnt_sze)
  plt.yticks(fontsize=fnt_sze)
  if put_grid:
    plt.grid()
  plt.savefig(EXP_OTPT_DIR+"/exp_otpt_plots/"+"".join(title.split()), dpi=500)
  if close:
    plt.close()
  else:
    plt.show()

def get_event_types_for_layer_lst(layer):
  """
  Returns a list of event types for the corresponding layer.

  Args:
    layer (int): The layer, e.g. 0 => Operation_Goal-Oriented.

  Returns:
    [[]]: A list of lists.
  """
  event_types_lst = []
  for tpl in HDD_EVENT_TYPES:
    if tpl[0] == layer:
      event_types_lst.append(tpl[1])

  return event_types_lst

def plot_conf_mat_heat_map(y_true, y_pred, metric="precision", y_annot=True):
  """
  Plots a heat map of the confusion matrix. Note: `normalize` over 'index' is
  Precision and over 'columns' is Recall.

  Args:
    y_true (np.array(int)): True labels of the test samples.
    y_pred (np.array(int)): Predicted labels of the test samples.
    metric (str): "precision"|"recall" -> Metric to be reported.
    y_annot (bool): Plot the `metric` on heatmap if True else not.
  """
  if metric=="precision":
    normalize = "index"
  elif metric=="recall":
    normalize = "columns"
  else:
    log.ERROR("Invalid metric: %s for calculating heat map." % metric)
    sys.exit()

  conf_mat = pd.crosstab(y_pred, y_true, rownames=["Predicted"],
                         colnames=["True"], normalize=normalize).round(4)*100
  #print(conf_mat)
  plt.figure(figsize=(8,6))
  sn.heatmap(conf_mat, annot=y_annot)
  plt.show()

def get_exp_output_metrics(y_true, y_pred, pred_scores, num_clss, avg=None):
  """
  Returns the metrics: Precision, Recall, Average Precision (AP), Mean AP.

  Args:
    y_true ([int]): A list of integers denoting true classes.
    y_pred ([int]): A list of integers denoting predicted classes.
    pred_scores (numpy.ndarray): A 2D array of predicted scores.
    num_clss (int): The number of unique classes in y_true and y_pred.
    avg (str|None): Averaging mechanism for calculating average precision score.

  Returns:
    numpy.ndarray, numpy.ndarray, numpy.ndarray
  """
  classes = list(range(num_clss))
  # Check if y_true has classes same as in a particular layer.
  assert np.unique(y_true).tolist() == classes
  # y_true_binary = label_binarize(y_true, classes=classes)
  y_true_binary = np.eye(num_clss)[y_true]

  prcsn_score = precision_score(y_true, y_pred, average=avg)
  recll_score = recall_score(y_true, y_pred, average=avg)
  avg_prcsn_score = average_precision_score(
      y_true_binary, pred_scores, average=avg)

  return prcsn_score, recll_score, avg_prcsn_score

def get_duration_of_video_in_secs(filename):
  """
  Returns the duration of video `filename` in seconds.

  Args:
    filename (str): Path/to/video.mp4

  Returns:
    float : Seconds.
  """
  result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration", "-of",
                          "default=noprint_wrappers=1:nokey=1", filename],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  return float(result.stdout)

def get_map_of_unique_classes_for_layer(session_ids, layer):
  """
  Returns the total number of unique classes in a layer and the dict of mapping
  of HDD layer labels to experiment classes (in range 0 to total labels -1 ).
  Also makes sure that the labels (keys) in map are in the qualified range of
  labels for the layer (mentioned in consts.py).

  Args:
    session_ids [str]: Session IDs of videos.
    layer (int): The layer.

  Returns:
    int, dict, dict: Number of unique labels (which is equal to number of
        classes), map of label to class, map of class to label.
  """
  if os.path.exists(COMMON_DATA_DIR + "/layer_%s_class_label_maps.p" % layer):
    log.INFO("Found the maps of classes and labels, returning...")
    return pickle.load(
        open(COMMON_DATA_DIR + "/layer_%s_class_label_maps.p" % layer, "rb"))

  all_unique_labels = set()
  map_cls_to_lbl = defaultdict(list)
  if layer == 0:
    map_lbl_to_cls = MAP_LBL_TO_CLS_LYR_0
  elif layer == 1:
    map_lbl_to_cls = MAP_LBL_TO_CLS_LYR_1

  all_qualified_labels = get_event_types_for_layer_lst(layer)
  for session_id in session_ids:
    labels = np.load(
        HDD_ALL_FRAMES_ALL_VIDS_LABELS_PATH + "/session_%s_layer_%s.npy" %
        (session_id, layer), allow_pickle=True)
    for label in labels:
      for ql in all_qualified_labels:
        # The saved `labels` are event_type + 1 with BG 0 and `ql` are event_type.
        if label-1 in ql:
          all_unique_labels.add(label)
          break
    log.INFO("Done for session: %s" % session_id)

  log.INFO("All unique labels of layer: %s are: %s" % (layer, all_unique_labels))
  # Map the cls 0 to label 0 where label 0 is always the background by default.
  map_cls_to_lbl[0] = [0]
  for lbl in all_unique_labels:
    cls = map_lbl_to_cls[lbl]
    map_cls_to_lbl[cls].append(lbl)

  pickle.dump((len(map_cls_to_lbl.keys()), map_lbl_to_cls, map_cls_to_lbl),
              open(COMMON_DATA_DIR + "/layer_%s_class_label_maps.p" % layer,
              "wb"))
  return len(map_cls_to_lbl.keys()), map_lbl_to_cls, map_cls_to_lbl

def get_accuracy_of_predicted_classes(pred_clss, test_clss, ignore_clss=[]):
  """
  Returns the accuracy of predicted classes. If `ignore_clss` is given, then
  while calculating the accuracy, those test classes are ignored.

  Args:
    pred_clss [int]: Predicted classes.
    test_clss [int]: Actual classes
    ignore_clss [int]: The classes to be ignored.

  Returns:
    float: The accuracy.
  """
  total_num_clss = 0
  acc = 0
  for pred_cls, test_cls in zip(pred_clss, test_clss):
    if test_cls in ignore_clss:
      continue
    total_num_clss += 1
    if test_cls == pred_cls:
      acc += 1

  if total_num_clss == 0:
    return 0.0

  return acc/total_num_clss

def construct_all_y_pred_y_actual(files_dir, session_ids=HDD_TEST_SIDS):
  """
  Constructs the tuple of predicted output and actual output for all the sessions.

  Args:
    files_dir (str): The pickle files directory,
    session_ids ([str]): A list of session IDs.

  Returns:
    {}: {epoch_num: [List of predicted classes], [List of actual classes]}.
  """
  files = os.listdir(files_dir)
  epochs_output = {}

  for f in files: # Each file corresponds to each epoch.
    if f.endswith(".p"): # i.e. it is a pickle file containing labels.
      epoch = int(f.split("_")[1])
      epoch_output = pickle.load(open(files_dir + "/%s" % f, "rb"))
      pred_clss, actual_clss, pred_scores = [], [], []
      for session_id in session_ids:
        if session_id not in epoch_output.keys():
          print("Session ID: %s output not found in pickle file: %s"
                % (session_id, f))
          continue
        session_output = epoch_output[session_id]
        for pred_cls, actual_cls, pred_score in zip(
              session_output[0], session_output[1], session_output[2]):
          pred_clss.append(pred_cls)
          actual_clss.append(actual_cls)
          pred_scores.append(pred_score)

      epochs_output[epoch] = (pred_clss, actual_clss, pred_scores)

  return epochs_output

def get_viena2_train_test_sample_splits(scenario, split_type):
  """
  Returns a tuple of lists storing the train and test splits for a particular
  `scenario` and a particular `split_type`.

  Args:
    scenario (str): The scenario e.g. "Scenario1", "Scenario2", ...
    split_type (str): One of "random", "daytime", "weather".

  Returns:
    [str], [str]: Training samples, Test samples.
  """
  train_split_type = ""
  test_split_type = ""

  if split_type == "random":
    train_split_type = 1
    test_split_type = 1
  else:
    train_split_type = split_type
    test_split_type = split_type

  scenario_train_split = (
      VIENA2_TRAIN_TEST_SPLITS.format(scenario) + scenario.lower() + "_train_"
      + ("1" if split_type == "random" else split_type)) + ".txt"
  scenario_test_split = (
      VIENA2_TRAIN_TEST_SPLITS.format(scenario) + scenario.lower() + "_test_"
      + ("1" if split_type == "random" else split_type)) + ".txt"
  #scenario_train_split = (
  #    VIENA2_TRAIN_TEST_SPLITS.format(scenario) + scenario.lower() + "_train_"
  #    + (train_split_type)) + ".txt"
  #scenario_test_split = (
  #    VIENA2_TRAIN_TEST_SPLITS.format(scenario) + scenario.lower() + "_test_"
  #    + (test_split_type)) + ".txt"
  train_lst, test_lst = [], []

  with open(os.path.abspath(scenario_train_split)) as f:
    lines = f.readlines()
    for line in lines:
      train_lst.append((line.split(" ")[0], line.split(" ")[1][:-1]))

  with open(os.path.abspath(scenario_test_split)) as f:
    lines = f.readlines()
    for line in lines:
      test_lst.append((line.split(" ")[0], line.split(" ")[1][:-1]))

  return train_lst, test_lst

def get_shuffled_lists_in_unison(lst_a, lst_b):
  """
  Shuffles two passed lists in unison.

  Args:
    lst_a ([]): 1st list.
    lst_b ([]): 2nd list.

  Returns:
    [], []
  """
  lst_f = list(zip(lst_a, lst_b))
  random.shuffle(lst_f)
  lst_a, lst_b = zip(*lst_f)
  return list(lst_a), list(lst_b)

def get_parallelly_processed_data(function, data):
  """
  This function parallelizes the for loop whose body contains the `function` call
  over each element in the data.

  Args:
    function (function): A python function to be implemented on data's elements.
    data ([]): An iterable, it can be numpy.ndarray too.

  Returns:
    []: List of the same length as data.
  """
  output = Parallel(n_jobs=-1)(delayed(function)(datum) for datum in data)
  return output

def get_viena2_train_test_clip_ids(scenario, split):
  """
  Returns the training and test clip IDs and labels in a scenario for a split
  type.

  Args:
    scenario (str): One of "Scenario1" | "Scenario2" ... | "Scenario5".
    split (str): One of "random" | "daytime" | "weather".

  Returns:
    [str], [int], [str], [int]: train_clip_ids, train_clss, test_clip_ids, test_clss
  """
  train_lst, test_lst = get_viena2_train_test_sample_splits(scenario, split)
  log.INFO("Train lst and Test lst obtained. Length: %s %s respectively"
           % (len(train_lst), len(test_lst)))
  def _get_name_and_clss(lst):
    clip_ids_lst, clip_clss_lst = [], []
    for clip_info in lst:
      clip_name, clip_cls = clip_info
      if scenario == "Scenario1":
        if clip_name[:-12] == "13_122_RR": # This clip has just 11 frames.
          continue
      clip_ids_lst.append(clip_name[:-12])
      clip_clss_lst.append(VIENA2_SCN_MAP[scenario][clip_cls])

    return clip_ids_lst, clip_clss_lst

  train_clip_ids, train_clss = _get_name_and_clss(train_lst)
  test_clip_ids, test_clss = _get_name_and_clss(test_lst)

  return train_clip_ids, train_clss, test_clip_ids, test_clss

def _get_results_dir_num_clss_ignore_clss(dataset, exp_type, kwargs, cls=None):
  """
  Returns the number of classes in a dataset and the classes to be ignored
  while creation of accuracy or average precision results.

  Args:
    dataset (str): One of HDD | VIENA2 macros defined in consts.py.
    exp_type (str): "visuals_only".
    kwargs (dict): `dataset` specific details.
    cls [int]: Classes for which results are to be created and rest discarded.

  Returns:
    str, int, [int]: Epoch results dir, Number of classes, classes to be ignored.
  """
  ignore_clss = []

  if dataset == HDD:
    layer = kwargs["layer"]
    results_dir = (
        EXP_OTPT_DIR + "/hdd_results/%s/layer_%s/" % (exp_type, layer))
    num_clss = len(HDD_LYR_CLS_MAP["layer%s" % layer].keys())
    if cls != None:
      ignore_clss = list(set(range(0, num_clss)) - set(cls))
  elif dataset == VIENA2:
    scenario, split = "Scenario%s" % kwargs["scenario"], kwargs["split"]
    results_dir = (
        EXP_OTPT_DIR + "/viena2_results/%s/%s/%s_" % (exp_type, scenario, split))
    num_clss = len(VIENA2_SCN_MAP[scenario].keys())
    if cls != None:
      ignore_clss = list(set(range(0, num_clss)) - set(cls))
  else:
    print("Wrong dataset type.")
    sys.exit()

  return results_dir, num_clss, ignore_clss

def get_acc_result_stats_tuple(dataset, exp_type, epoch_limit, runs, kwargs,
                               cls=None):
  """
  Returns the mean and std of results.

  Args:
    dataset (str): One of HDD | VIENA2 macros defined in consts.py.
    exp_type (str): One of "visuals_only".
    epoch_limit (int): Number of epochs up till which results are considered.
    runs (int): Number of experiment runs considered for result stats calculation.
    kwargs (dict): `dataset` specific details.
    cls [int]: Classes for which results are to be created and rest discarded.

  Returns:
    [float], [float]: mean of accuracies, std of accuracies.
  """
  agrt_result = np.zeros((epoch_limit, runs))
  results_dir, _, ignore_clss = _get_results_dir_num_clss_ignore_clss(
      dataset, exp_type, kwargs, cls)

  for run in range(1, runs+1):
    run_results_dir = results_dir + "run_%s/" % run
    if dataset == HDD:
      epochs_result = construct_all_y_pred_y_actual(run_results_dir)
    for epoch in range(1, epoch_limit+1):
      if dataset == HDD:
        epoch_result = epochs_result[epoch]
      else:
        epoch_result = pickle.load(
            open(run_results_dir + "epoch_%s_results.p" % epoch, "rb"))
      agrt_result[epoch-1, run-1] = (
          get_accuracy_of_predicted_classes(epoch_result[0], epoch_result[1],
                                            ignore_clss))

  return np.mean(agrt_result, axis=1).tolist(), np.std(agrt_result, axis=1).tolist()

def get_prcsn_recll_avg_prcsn_stats_tuple(dataset, exp_type, epoch_limit, kwargs,
                                          runs=1):
  """
  Returns the precision, recall, average precision and std deviation of
  correponding metrics.

  Args:
    dataset (str): One of HDD | VIENA2 macros defined in consts.py.
    exp_type (str): One of "visuals_only".
    epoch_limit (int): Number of epochs up till which results are considered.
    kwargs (dict): `dataset` specific details.
    runs (int): Number of experiment runs considered for result stats calculation.

  Returns:
    dict: {
      "precision": { "mean_values": [[float]], "std_values: [[float]] },
      "recall": { "mean_values": [[float]], "std_values": [[float]] },
      "average_precision": {"mean_values": [[float]], "std_values": [[float]]}
    }
    Note: List at index i (in the external list) corresponds to epoch i+1.
  """
  results_dir, num_clss, _ = _get_results_dir_num_clss_ignore_clss(
      dataset, exp_type, kwargs)
  agrt_avg_prcsn_results = np.zeros((epoch_limit, num_clss, runs))
  agrt_prcsn_results = np.zeros((epoch_limit, num_clss, runs))
  agrt_recll_results = np.zeros((epoch_limit, num_clss, runs))

  for run in range(1, runs+1):
    run_results_dir = results_dir + "run_%s/" % run
    if dataset == HDD:
      epochs_output = construct_all_y_pred_y_actual(run_results_dir)
    for epoch in range(1, epoch_limit+1):
      if dataset == HDD:
        if epoch in epochs_output:
          epoch_result = epochs_output[epoch]
      else:
        epoch_result = pickle.load(
            open(run_results_dir + "epoch_%s_results.p" % epoch, "rb"))

      if epoch in epochs_output:
        prcsn_score, recall_score, avg_prcsn_score = get_exp_output_metrics(
            epoch_result[1], epoch_result[0], epoch_result[2], num_clss)
        agrt_avg_prcsn_results[epoch-1, :, run-1] = avg_prcsn_score
        agrt_prcsn_results[epoch-1, :, run-1] = prcsn_score
        agrt_recll_results[epoch-1, :, run-1] = recall_score

  results = {
    "precision": {"mean_values": np.mean(agrt_prcsn_results, axis=2).tolist(),
                  "std_values": np.std(agrt_prcsn_results, axis=2).tolist() },
    "recall": {"mean_values": np.mean(agrt_recll_results, axis=2).tolist(),
               "std_values": np.std(agrt_recll_results, axis=2).tolist()},
    "average_precision": {
        "mean_values": np.mean(agrt_avg_prcsn_results, axis=2).tolist(),
        "std_values": np.std(agrt_avg_prcsn_results, axis=2).tolist()}
  }

  return results

def get_viena2_class_counts(scenario, split):
  """
  Returns the counter of classes in viena2.

  Args:
    scenario (str): One of "Scenario1" | "Scenario2" | ... "Scenario5".
    split (str): One of "random" | "daytime" | "weather".

  Returns:
    collections.Counter()
  """
  _, train_clss, _, test_clss = get_viena2_train_test_clip_ids(
      scenario, split)
  return Counter(train_clss), Counter(test_clss)

def get_event_type_stats(layer, event_type_lst):
  df = get_saved_index_pkl()["events_pd"]
  df = df[(df["layer"] ==  layer) & (df["event_type"].isin(event_type_lst))]
  time_diff_secs = (df["end"] - df["start"])/1000  # Times are mentioned in ms.
  return dict(time_diff_secs.describe())

def cal_acc_at_x(numrtr_xframes, dnmrtr_xframes, K, true_clss, pred_clss):
  num_true_clss = len(true_clss)
  fst = 0
  for i in range(1, num_true_clss):
    #if true_clss[i] == 0:
    #  continue
    if true_clss[i] != true_clss[i-1]:
      fst = 0
    if fst <= K:
      dnmrtr_xframes[fst] += 1
      if pred_clss[i] == true_clss[i]:
        numrtr_xframes[fst] += 1
    fst += 1

def get_acc_for_nframe_since_scene_transition(K, exp_type, kwargs, epoch,
                                              run, session_ids=HDD_TEST_SIDS):
  results_dir, _, _ = _get_results_dir_num_clss_ignore_clss(HDD, exp_type, kwargs)
  results_dir = results_dir + "/run_%s/ns_test_only/" % run
  otpt = pickle.load(
      open(results_dir + "epoch_%s_results_ns_test_only.p" % epoch, "rb"))

  numrtr_xframes = np.zeros(K+1)
  dnmrtr_xframes = np.zeros(K+1)
  for session_id in session_ids:
    cal_acc_at_x(numrtr_xframes, dnmrtr_xframes, K, otpt[session_id][1],
                 otpt[session_id][0])
  #return numrtr_xframes, dnmrtr_xframes
  return numrtr_xframes / dnmrtr_xframes
