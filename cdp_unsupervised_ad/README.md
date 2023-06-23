# CDP authentication based on synthetic physical templates

## Description
This repository collects the methods to do unsupervised anomaly detection on our Copy Detection Patterns (CDP) dataset.

## Requirements
The code is meant to be run on Python 3.8. Code dependencies are specified in the `requirements.txt` file.

## Usage
A model can be trained or tested using the following command:

`python main.py --conf {conf_file.json}`

Where `conf_file.json` is a configuration file containing all program arguments:

 - `mode` - Modality to use. Checkout different modalities in the [modalities section](#modalities)
 - `t_dir` - Path to the directory containing templates.
 - `x_dirs` - List of paths to the directories containing **original** printed codes.
 - `f_dirs` - List of directories containing **fake** printed codes (used for testing only).
 -  `result_dir` - Directory where all results and trained models will be stored.
 -  `no_train` - Boolean which tells whether to skip train (true) or not (false). Pre-trained models are loaded from the `result_dir`.
 - `epochs` - Number of epochs used for training.
 -  `lr` - Learning rate for training.
 -  `bs` - Batch size used for training.
 -  `tp` - Percentage of codes used for training.
 -  `vp` - Percentage of codes used for validation. Note that the percentage of testing images is `1 - (tp+vp)`.
 -  `seed` - Random seed for the experiment.

### Modalities

There are currently 6 possible modalities:
 - **t2x** The model is trained to produce a printed CDP given the template (```t2x```).
 - **t2xa** The model is trained to produce a printed CDP and a confidence map given the template (```t2xa```).
 - **x2t** The model is trained to produce a template CDP given the printed (```x2t```).
 - **x2ta** The model is trained to produce a template CDP and a confidence map given the printed (```x2ta```).
 - **both** Two models are trained: The first model is trained as in **1** and the second as in **3**. A Cycle-consistent term in the loss is also used (```both```).
 - **both_a** Two models are trained: The first model is trained as in **2** and the second as in **4**. A Cycle-consistent term in the loss is also used (```both_a```).

