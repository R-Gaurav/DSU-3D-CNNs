#
# Author: Ramashish Gaurav
#

from datetime import datetime, timedelta
from collections import defaultdict
from skimage.transform import resize, rescale

import pandas as pd
import numpy as np
import os
import random
import pickle
import subprocess
import sys
import csv
import math
import time

from . import log
from .consts import (HDD_VIDEO_PATH, FPS, HDD_ALL_FRAMES_ALL_VIDS_LABELS_PATH,
                     HDD_TRAIN_SIDS, LOGS_PATH, DSP_0_15, HDD_DSP_0_15_FRAMES,
                     COMMON_DATA_DIR, DSP_0_10, HDD_DSP_0_10_FRAMES, DSP_0_05,
                     HDD_DSP_0_05_FRAMES, VIDEO_FRAME_SPCS, VIENA2_FRAMES_DIR,
                     HDD_DATA_BASE_DIR, VIENA2_DATA_DIR, HDD, VIENA2, VIENA2_SCN_MAP)
from .exp_utils import (
    get_shuffled_lists_in_unison, get_parallelly_processed_data,
    get_map_of_unique_classes_for_layer, get_viena2_train_test_clip_ids)

def extract_frames(video_path, frm_wdth, frm_hght, start=None, end=None,
                   fps_rate=FPS):
  """
  Extract frames from the videos. Reads a *.mp4 video and a matrix of frames.
  Note:
    HDD videos frames shape: 1280 x 720 x 3 (frm_width x frm_hght x 3)
    VIENA2 videos frames shape: 1920 x 1080 x 3 (frm_width x frm_hght x 3)

  Args:
    video_path (str): path/to/video.
    frm_wdth (int): Frames width or number of pixel columns.
    frm_hght (int): Frames height or the number of pixel rows.
    start (float): Start of the video sequence in seconds.
    end (float): End of the video sequence in seconds.
    fps_rate (str): Number of frames to be extracted every second, e.g. "30/1" =>
        thirty frames extracted every second.

  Returns:
    numpy.ndarray : A matrix of frames of shape (# frames, 720, 1280, 3)
  """

  if start and end:
    command = ["ffmpeg",
              "-ss", str(start),
              "-i", video_path,
              "-t", str(end-start),
              # Make sure all frames are of same scale for uniformity in CNN inpt.
              "-vf", "fps=%s,scale=%s:%s" % (fps_rate, frm_wdth, frm_hght),
              "-f", "image2pipe",
              "-pix_fmt", "rgb24",
              "-vcodec", "rawvideo",
              "-"]
  else:
    command = ["ffmpeg",
               "-i", video_path,
               "-vf", "fps=%s,scale=%s:%s" % (fps_rate, frm_wdth, frm_hght),
               "-f", "image2pipe",
               "-pix_fmt", "rgb24",
               "-vcodec", "rawvideo",
               "-"]

  process = subprocess.Popen(
      command, stdout=subprocess.PIPE, stderr=open(os.devnull, 'w'),
      bufsize=10 ** 7)

  nbytes = 3*frm_hght*frm_wdth # Number of pixels in image.
  frames = []
  frame_count = 0

  while True:
    byte_string = process.stdout.read(nbytes)
    if len(byte_string) != nbytes:
      break
    else:
      frame = np.fromstring(byte_string, dtype="uint8")
      frame = frame.reshape(frm_hght, frm_wdth, 3) # Shape the frame as desired.
      frames.append(frame)
      frame_count += 1

  process.wait()
  del process
  if len(frames) == 0:
    log.INFO("No frames found, perhaps the video_path is wrong. Check it.")
  return np.asarray(frames, dtype="uint8")

def get_all_frames_labels_all_vids(session_ids_lst, layer, fps):
  """
  Creates labels of each frames in all videos.

  Args:
    session_ids_lst ([int]): A list of videos' session IDs.
    layer (int): Layer of the HDD dataset.
    fps (str): Frame rate per second e.g. "3/1" => 3 frames per second.
  """
  samp_freq = int(fps.split("/")[0])
  saved_index = get_saved_index_pkl()
  events_pd = saved_index["events_pd"]
  layer_events_pd = events_pd[events_pd["layer"] == layer]

  for session_id in session_ids_lst:
    files = os.listdir(HDD_VIDEO_PATH.format(session_id))
    total_duration_secs = get_duration_of_video_in_secs(
        HDD_VIDEO_PATH.format(session_id)+"/"+files[0])
    num_frames = int(samp_freq * total_duration_secs)
    session_labels = np.zeros(num_frames)

    session_rows = layer_events_pd[layer_events_pd["session_id"] == session_id]
    for row in session_rows.iterrows():
      start, end = (int(row[1]["start"]/1000 * samp_freq),
                    int(row[1]["end"] / 1000 * samp_freq))
      # Since the background (i.e. frames with no event_type) is labelled 0,
      # and right_turn is also labelled 0, increment each event_type label by 1.
      # Make sure while interpreting results, predicted event_type labels are
      # reduced by 1. This is generalized to all the layers.
      session_labels[start:end] = row[1]["event_type"] + 1

    np.save(HDD_ALL_FRAMES_ALL_VIDS_LABELS_PATH + "/session_%s_layer_%s.npy" % (
            session_id, layer), session_labels)
    log.INFO("Labels for session: %s, layer: %s done." % (session_id, layer))

def _do_down_sampling(video_path, option, video_source, file_name, directory,
                      down_samp_prop=None, start=None, end=None, rows=None,
                      cols=None, fps_rate=FPS):
  """
  Calls extract_frames function and does downsampling.

  Args:
    video_path (str): path/to/video
    option (str): One of "rescale" or "resize".
    video_source (str): One of "hdd" | "viena2" for choosing frame shape.
    down_samp_prop (float): Downsampling proportion.
    file_name (str): The output file name to be saved.
    directory (str): Output/path/where/downsampled/frames/are/saved
    start (float): Start of the video sequence in seconds.
    end (float): End of the video sequence in seconds.
    fps_rate (str): Number of frames to be extracted every second, e.g. "30/1" =>
        thirty frames extracted every second.
  """
  frm_wdth = VIDEO_FRAME_SPCS[video_source]["frm_wdth"]
  frm_hght = VIDEO_FRAME_SPCS[video_source]["frm_hght"]
  log.INFO("Extracting frames for video... : %s" % video_path)
  single_ssn_all_frames = extract_frames(
      video_path, frm_wdth, frm_hght, start, end, fps_rate)
  total_num_frames = single_ssn_all_frames.shape[0]
  img_rows, img_cols = (
      single_ssn_all_frames[0].shape[0], single_ssn_all_frames[0].shape[1])

  if option == "resize":
    if down_samp_prop:
      rows, cols = int(img_rows*down_samp_prop), int(img_cols*down_samp_prop)
    def _resize_image(img):
      return resize(img, (rows, cols), anti_aliasing=True)
    log.INFO("Resizing frames... to (rows, cols): (%s, %s)" % (rows, cols))
    single_ssn_ds_all_frames = get_parallelly_processed_data(
        _resize_image, single_ssn_all_frames)
  elif option == "rescale":
    def _rescale_image(img):
      return rescale(img, down_samp_prop, anti_aliasing=True)
    log.INFO("Rescaling frames... to proportion: %s" % down_samp_prop)
    single_ssn_ds_all_frames = get_parallelly_processed_data(
        _rescale_image, single_ssn_all_frames)
  else:
    log.ERROR("Downsamling Option: %s not supported." % option)
    sys.exit()

  single_ssn_ds_all_frames = np.array(single_ssn_ds_all_frames)
  log.INFO("Saving downsampled frames... of session: %s" % file_name)
  np.save(directory + "/" + file_name, single_ssn_ds_all_frames)

  return single_ssn_ds_all_frames

def get_down_samp_frames_of_clip(clip_id, scenario, option, rows, cols,
                                 fps_rate=FPS):
  """
  Note: This function is for the VIENA2 dataset.

  Args:
    clip_id (str): The clip ID of the `scenario`'s video clip.
    scenario (str): One of "Scenario1" | "Scenario2" | ... | "Scenario5".
    option (str): One of "resize" | "rescale".
    rows (int): Number of rows of pixels in down sampled frames.
    cols (int): Number of columnss of pixels in down sampled frames.
    fps_rate (str): Number of frames to be extracted every second, e.g. "30/1" =>
        thirty frames extracted every second.
  """
  file_name = "%s_snippet_%sd.npy" % (clip_id, option)
  directory = VIENA2_FRAMES_DIR["%s %s" % (rows, cols)] + "/" + scenario

  if os.path.exists(directory + "/" + file_name):
    return np.load(directory + "/" + file_name, allow_pickle=True)
  else:
    video_path = VIENA2_DATA_DIR + "/raw_data/video/" + scenario + "/" + (
        "%s_snippet.avi" % clip_id)
    single_clip_ds_all_frames = _do_down_sampling(
        video_path, option, VIENA2, file_name, directory, rows=rows, cols=cols,
        fps_rate=fps_rate)

    return single_clip_ds_all_frames

def get_down_samp_frames_of_session(session_id, option, down_samp_prop=DSP_0_15,
                                    start=None, end=None, fps_rate=FPS):
  """
  Note: This function is for the HDD dataset.

  Obtains the frames of a session and rescales or resizes it to a desired size.
  `anti_aliasing` is set to True to smooth the image and avoid aliasing artifacts.

  Args:
    session_id (str): Session ID of the video.mp4
    option (str): One of "rescale" or "resize".
    down_samp_prop (float): Downsampling proportion.
    start (float): Start of the video sequence in seconds.
    end (float): End of the video sequence in seconds.
    fps_rate (str): Number of frames to be extracted every second, e.g. "30/1" =>
        thirty frames extracted every second.

  Returns:
    numpy.ndarray: Shape-> num_frames x ds_rows x ds_cols x 3
  """
  log.INFO("Using DSP: %s" % down_samp_prop)
  file_name = "session_%s_%sd_frames.npy" % (session_id, option)
  if down_samp_prop == DSP_0_15:
    directory = HDD_DSP_0_15_FRAMES
  elif down_samp_prop == DSP_0_10:
    directory = HDD_DSP_0_10_FRAMES
  elif down_samp_prop == DSP_0_05:
    directory = HDD_DSP_0_05_FRAMES

  if os.path.exists(directory + "/" + file_name):
    log.INFO("File: %s found. Returning it..." % file_name)
    return np.load(directory + "/" + file_name, allow_pickle=True)
  else:
    log.INFO("Creating file... : %s" % file_name)
    files = os.listdir(HDD_VIDEO_PATH.format(session_id))
    video_path = HDD_VIDEO_PATH.format(session_id)+"/"+files[0]
    single_ssn_ds_all_frames = _do_down_sampling(
        video_path, option, HDD, file_name, directory,
        down_samp_prop=down_samp_prop, start=start, end=end, fps_rate=fps_rate)

    return single_ssn_ds_all_frames

def get_qualified_frames_and_clss(session_id, layer, map_lbl_to_cls, option,
                                  dsp=DSP_0_15):
  """
  Returns the qualified frames and their corresponding labels for a particular
  `layer`. The qualified frames and labels are the ones mentioned in consts.py
  for the passed `layer`.

  Args:
    session_id (str): The session ID of the video.mp4.
    layer (int): The layer of action recognition.
    map_lbl_to_cls (dict): The map of qualified labels to classes.
    option (str): The downsampling option, one of "resize" or "rescale".
    dsp (float): The image downsampling proportion.

  Returns:
    numpy.ndarray, numpy.ndarray: Qualified frames, Qualified classes
  """
  single_ssn_all_frames = get_down_samp_frames_of_session(session_id, option, dsp)
  single_ssn_all_labels = np.load(
      HDD_ALL_FRAMES_ALL_VIDS_LABELS_PATH + "/session_%s_layer_%s.npy" %
      (session_id, layer), allow_pickle=True)

  log.INFO("Downsampled frames and labels obtained for session: %s" % session_id)
  # Make sure that frames and labels are same in number.
  min_len = min(single_ssn_all_frames.shape[0], single_ssn_all_labels.shape[0])
  single_ssn_all_frames = single_ssn_all_frames[:min_len]
  single_ssn_all_labels = single_ssn_all_labels[:min_len]

  qualified_all_frames, qualified_all_clss = [], []
  # Map the qualified labels to their classes.
  log.INFO("Creating qualified frames and classes...")
  for frame, label in zip(single_ssn_all_frames, single_ssn_all_labels):
    if label in map_lbl_to_cls.keys():
      qualified_all_frames.append(frame)
      qualified_all_clss.append(map_lbl_to_cls[label])

  return np.array(qualified_all_frames), np.array(qualified_all_clss)

def get_batches_of_sequence_frames(session_id, batch_size, depth_size, layer,
                                   option, dsp, create_batch=True,
                                   do_shuffle=True):
  """
  Args:
    session_id (str): The session ID of the video.mp4.
    batch_size (int): The number of elements in a batch.
    depth_size (int): The number of frames in one element of the batch. This is
                      the sequence size or the temporal dimension.
    layer (int): The layer of the action recognition.
    option (str): One of "rescale" or "resize".
    dsp (float): The image downsampling proportion.
    create_batch (bool): If True create batches else return the whole data.
    do_shuffle (bool): If True shuffle the data else not.

  Returns:
    if create_batch == True:
      A generator of:
        numpy.ndarray: 5D Shape -> batch_size, depth_size, img_rows, img_cols, 3
        numpy.ndarray: 2D Shape -> batch_size, num_clss
    else:
      numpy.ndarray, numpy.ndarray
  """
  if isinstance(dsp, np.float32):
    dsp = round(dsp.item(), 2)
  if isinstance(session_id, bytes):
    session_id = session_id.decode("utf-8")
  if isinstance(option, bytes):
    option = option.decode("utf-8")

  num_clss, map_lbl_to_cls, _ = get_map_of_unique_classes_for_layer(
      HDD_TRAIN_SIDS, layer)
  single_ssn_all_frames, single_ssn_all_clss = get_qualified_frames_and_clss(
      session_id, layer, map_lbl_to_cls, option, dsp)
  single_ssn_all_clss = np.eye(num_clss, dtype=np.float)[single_ssn_all_clss]

  log.INFO("Qualified frames and classes obtained for session: %s" % session_id)

  # Note: Frames in single_ssn_all_frames are all arranged with starting frame
  # at 0th index. Shape is: num_frames x rows x cols x 3
  num_frames = single_ssn_all_frames.shape[0]
  frame_sqnc, class_sqnc = [], []

  log.INFO("Creating frame sequence...")
  for i in range(depth_size, num_frames):
    # Flip the frames.
    frame_sqnc.append(np.flip(single_ssn_all_frames[i-depth_size:i], 0))
    # Eg: For a `depth_size` of 16, and `i` = 16, the frames to be flipped above
    # are from 0 to 15 inclusive, thus we need the label of i-1 i.e. 15th frame.
    class_sqnc.append(single_ssn_all_clss[i-1])

  del single_ssn_all_frames
  del single_ssn_all_clss
  # Shuffle the order of sequences if required.
  if do_shuffle:
    log.INFO("Shuffling the data ...")
    frame_sqnc, class_sqnc = get_shuffled_lists_in_unison(frame_sqnc, class_sqnc)

  if not create_batch:
    frame_sqnc = np.array(frame_sqnc)
    class_sqnc = np.array(class_sqnc)
    log.INFO("Returning whole data, frame_sqnc.shape: %s, class_sqnc.shape: %s"
             % (frame_sqnc.shape, class_sqnc.shape))
    return frame_sqnc, class_sqnc

  log.INFO("Creating batches...")
  # Create the batch data.
  fs_size = len(frame_sqnc)
  for start in range(0, fs_size, batch_size):
    end = min(fs_size, start+batch_size)
    if end-start != batch_size:
      continue
    yield (np.array(frame_sqnc[start:end]), np.array(class_sqnc[start:end]))

def get_batches_of_viena2_clips(scenario, split, mode, batch_size, option,
                               rows, cols, depth_size=16, do_shuffle=True):
  """
  Generates a batch of training and test frames for the VIENA2 experiment.

  Args
    scenario (str): One of "Scenario1" | "Scenario2" | ... "Scenario5".
    split (str): One of "random" | "daytime" | "weather".
    mode (str): One of "training" | "test".
    batch_size (int): Batch size.
    option (str): One of "resize" | "rescale".
    depth_size (int): Temporal Depth of the sequence of frames.
    do_shuffle (bool): If True then shuffle the entire training/test data.

  Returns:
    A generator of:
      numpy.ndarray: 5D Shape -> batch_size, depth_size, img_rows, img_cols, 3
      numpy.ndarray: 2D Shape -> batch_size, num_clss
  """
  if isinstance(rows, np.float32):
    #rows = round(dsp.item(), 2)
    rows = int(rows.item())
  if isinstance(cols, np.float32):
    cols = int(cols.item())
  if isinstance(scenario, bytes):
    scenario = scenario.decode("utf-8")
  if isinstance(split, bytes):
    split = split.decode("utf-8")
  if isinstance(mode, bytes):
    mode = mode.decode("utf-8")
  if isinstance(option, bytes):
    option = option.decode("utf-8")

  log.INFO("Mode: %s" % mode)
  num_clss = len(VIENA2_SCN_MAP[scenario].keys())
  train_ids, train_clss, test_ids, test_clss = get_viena2_train_test_clip_ids(
      scenario, split)

  def _batch_generator(clip_ids, clip_clss):
    num_ids = len(clip_ids)
    for start in range(0, num_ids, batch_size):
      end = min(num_ids, start + batch_size)
      if end-start != batch_size and mode == "training":
        continue
      clip_batch, cls_batch = [], []
      for idx in range(start, end):
        clip = get_down_samp_frames_of_clip(
            clip_ids[idx], scenario, option, rows, cols)
        clip_batch.append(np.flip(clip, 0))
        cls_batch.append(clip_clss[idx])

      clip_batch, cls_batch = np.array(clip_batch), np.array(cls_batch)
      log.INFO("Clip Batch Shape: {0}, Cls Batch Shape: {1}".format(
               clip_batch.shape, cls_batch.shape))
      log.INFO("Files: %s" % clip_ids[start:end])
      yield (clip_batch, cls_batch)

  if mode == "training":
    # Shuffle training clip and class in unison.
    if do_shuffle:
      log.INFO("Shuffling the training data...")
      train_ids, train_clss = get_shuffled_lists_in_unison(train_ids, train_clss)

    train_clss = np.eye(num_clss, dtype=np.float32)[train_clss]
    log.INFO("Creating training batches...")
    return _batch_generator(train_ids, train_clss)

  else:
    log.INFO("Creating test batches...")
    return _batch_generator(test_ids, test_clss)
