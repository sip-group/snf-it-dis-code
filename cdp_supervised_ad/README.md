## CDP Supervised AD

Repository of the "Authentication of Copy Detection Patterns under Machine Learning Attacks: A Supervised Approach" [paper](https://arxiv.org/abs/2206.11793)

## Requirements
All requirements are specified in the ```requirements.txt``` file. The code was deceloped using Python 3.8 on a MacOS Big Sur (Version 11.6) and is meant to work with MacOS and Linux systems.

## Usage
### Training
Simply run

```python3 main.py {path/to/conf.json}```

Where the JSON file specifies configuration values. Check the ```src/configuration/default_conf.json``` file for an example configuration.

The models are stored under a new directory ```new/``` and are tested on part of the data (50% by default). The embedding representations (vectors) are stored under the ```new/lv/``` sub-folder.
Tensorboard logs of the training and validation loss are available under the ```/main/tensorboard_logs/``` folder, and are accessible through the ```tensorboard --logdir /main/tensorboard_logs/``` command (navigate to ```localhost:6006```).


### Testing
Testing is done by obtaining ResNet-18 latent vectors (lvs) with ```main.py``` and then running a summary with 
```results_summary.py```

**Testing on the same test set used when training**

```python3 main.py {path/to/conf.json}```
```python3 results_summary.py dirs ${DIR_1} ${DIR_2} ${DIR_3}```

Where the _conf.json_ file containes the same directories structures, percentages of train and validation, store directory and seed.
The directories passed to ```results_summary.py``` will be used to find the stored lvs and summarize results.
<br/><br/><br/>

**Testing on completely unseen fakes and originals**

```python3 main.py {path/to/conf.json}```

Where the configuration now contains folders in parameters ```ADDITIONAL_TEST_ORIGINALS``` and (unused) ```ADDITIONAL_TEST_FAKES```.
The directories passed to ```results_summary.py``` will be used to find the stored lvs and summarize results.