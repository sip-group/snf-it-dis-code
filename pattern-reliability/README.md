# Predictor patent project

## 1. General description

This repository contains all codes and simulations used in the project of the TIFS paper on pattern-based authentication.

![Predictor algorithm](predictor_algorithm.png)

## 2. Files and folder organization

### Main files

There are 4 main python notebook which run the main experiments :

* measure_codebook.ipynb :
  * Trains a codebook on a dataset triplet (t,x,f)
  * Stores the results in results/codebooks/codebook_measures/


* visualize_codebooks.iypnb :
  * Reads datas from the result folder of codebook_measure.ipynb
  * Generates graphics and stores them in results/codebooks/


* measure_weighted_metrics.ipynb :
  * Trains a codebook on a dataset pair (t,x)
  * Uses this codebook to learn a mask with various threshold values $\mu$
  * Computes weighted metrics using the learned mask on a dataset triplet (t,x,f)
  * Stores the results in results/metrics/{measures_data/, roc_curves_data/}

* visualize_metrics.ipynb:
  * Reads datas from the result folder of weighted_metrics_measures.ipynb
  * Generates graphics and stores them in results/metrics/graphics/

### Library files

There are 3 pure python files which are used as libraries by the python-notebooks :

* cdp_metrics.py :
  * Implements all different kinds of metrics : MSE, PCOR, NC-PCOR, L1, DHAMM, LLS
  * There is one general function batch_metric() which loads them all by only specifying the name.


* Dataset_cdp.py :
  * This file defines a Dataset object from the Pytorch library.
  * It is used to load all 3 different datasets 'scanner', 'iphone' and 'samsung'.
  * It manages missing samples automatically.


* predictor_functions.py :
  * This file implements the main functions of the predictor algorithm : train_codebook() and predict()
  * They also come in a multiprocessing version for multi-CPU acceleration.
  * We also implement here the binarization function Otsu + majority voting.

## 3. Datasets

The datasets we use in this experiment are located on the server gallager.unige.ch

Main folder : /ndata/chaban/cdp/images/d1_ps1_dens0.5_rep1

* Scanner dataset :
  * Originals : orig_scan/HPI55_printdpi812.8_printrun1_session0_InvercoteG/scanrun1_scandpi2400/rcod
  * Fakes : fake_scan/HPI55_printdpi812.8_printrun1_session0_InvercoteG_EHPI55/scanrun1_scandpi2400/rcod

* iPhone dataset :
  * Originals : orig_phone/HPI55_printdpi812.8_printrun1_session1_InvercoteG/iPhone12Pro_run1_ss100_focal12_apperture1/rcod
  * Fakes : fake_phone/HPI55_printdpi812.8_printrun1_session1_InvercoteG_EHPI55/iPhone12Pro_run1_ss100_focal12_apperture1/rcod

* Samsung dataset :
    * Originals : orig_phone/HPI55_printdpi812.8_printrun1_session1_InvercoteG/SamsungGN20U_run1_ss100_focal12_apperture1/rcod
    * Fakes : fake_phone/HPI55_printdpi812.8_printrun1_session1_InvercoteG_EHPI55/SamsungGN20U_run1_ss100_focal12_apperture1/rcod

