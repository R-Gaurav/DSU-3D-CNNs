# DSU-3D-CNNs
This repository contains code for Driving Scene Understanding using 3D-CNNs. The
code presented here has been cleaned up and executed again for training and
testing. In case something breaks up or doesn't work on your machine, please feel
free to contact us (contact email IDs in research paper).

# Research Paper
**Driving Scene Understanding: How much temporal context and spatial resolution is
necessary?**

Accepted at [Canadian AI 2021](https://www.caiac.ca/en/conferences/canadianai-2021/home)

## Abstract
Driving Scene Understanding is a broad field which addresses the problem of
recognizing a variety of on-road situations; namely driver behaviour/intention
recognition, driver-action causal reasoning, pedestrians’ and nearby vehicles’
intention recognition, etc. Many existing works propose excellent AI based
solutions to these interesting problems by leveraging visual data along with
other modalities. However, very few researchers venture into determining the
necessary metadata of the visual inputs to their models. This work attempts to
put forward some useful insights about the required spatial resolution and
temporal context/depth of the visual data for Driving Scene Understanding.

## Datasets Used
* Honda Research Institute Driving Dataset ([HDD](https://usa.honda-ri.com/HDD))
* [VIENA2](https://sites.google.com/view/viena2-project/home)

# Instructions to use our Code
Following are the general instructions to use our code (since a variety of
experiments were executed). In case you are having troubles to set up the
experiment environment, please feel free to contact us.

* You will be required to create directories to store the extracted frames,
experiment outputs etc. and set appropriate experiment constants; this can be
done by creating a file `DSU-3D-CNNs/src/utils/base_utils/consts.py` and mentioning
your constants there (we have not uploaded this file since it is specific to our
experiment environment).
* You can execute `DSU-3D-CNNs/src/data_creators/data_prep_utils.py` file to
extract frames of required spatial resolution.
* Once done with frame data creation, you can then set appropriate model specific
and dataset specific constants in file `DSU-3D-CNNs/src/train_test_code/exp_settings.py`.
Do note that you would need a system with nearly 180GB of RAM to extract frames
of the HDD dataset videos.
* Then execute the file `DSU-3D-CNNs/src/train_test_code/hdd_train_test_eval_3d_cnn.py`
to train/test/evaluate our 3D-CNN model (note that with the change in dataset,
you also need to change the file to be executed).
* In case you have saved your trained models and want to test one such trained
model obtained at the end of a certain epoch, you can execute the file
`DSU-3D-CNNs/src/train_test_code/hdd_test_only.py` to obtain the results.
* For obtaining the average precision results, you can use functions
`get_exp_output_metrics` and `get_prcsn_recll_avg_prcsn_stats_tuple` in file
`DSU-3D-CNNs/src/utils/base_utils/exp_utils.py`.
* For obtaining the **ASiST@x** scores, you can use the function
`get_acc_for_nframe_since_scene_transition` defined in file
`DSU-3D-CNNs/src/utils/base_utils/exp_utils.py`.

## Experiment Environment
* Python 3.7.7
* TensorFlow 2.2.0
* TensorFlow Addons 0.9.1
* Pandas 1.0.5
* Scipy 1.4.1
* Skimage 0.16.2
* Sklearn 0.23.1
* Numpy 1.18.5
* Joblib 0.16.0
* Matplotlib 3.2.2
* Seaborn 0.10.1

## System Requirements
It is recommended that you have access to a system with 4 32GB GPUs to train and
test the code within a period of 24 hours (for one experiment with 108 x 192
pixels spatial resolution and 16 frames temporal depth).
