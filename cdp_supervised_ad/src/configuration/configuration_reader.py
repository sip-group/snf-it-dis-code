import os
import json
import warnings
import argparse
import shutil

# Arguments MACROS
TEMPLATE_DIR = 'TEMPLATE_DIR'
ORIGINAL_DIR = 'ORIGINAL_DIR'
FAKE_DIR = 'FAKE_DIR'

# Arguments MACROS (optional)
ADDITIONAL_ORIGINALS = 'ADDITIONAL_ORIGINALS'
ADDITIONAL_FAKES = 'ADDITIONAL_FAKES'
ADDITIONAL_TEST_ORIGINALS = 'ADDITIONAL_TEST_ORIGINALS'
ADDITIONAL_TEST_FAKES = 'ADDITIONAL_TEST_FAKES'
BATCH_SIZE = 'BATCH_SIZE'
BAD_INDICES = 'BAD_INDICES'
CLASSIFICATION = 'CLASSIFICATION'
EPOCHS = 'EPOCHS'
LATENT_DIMENSION = 'LATENT_DIMENSION'
MARGIN = 'MARGIN'
MLP = 'MLP'
MODEL = 'MODEL'
MULTI_GPU = 'MULTI_GPU'
PATIENCE = 'PATIENCE'
PERCENT_TRAIN = 'PERCENT_TRAIN'
PERCENT_VAL = 'PERCENT_VAL'
RUNS = 'RUNS'
SEED = 'SEED'
STACK = 'STACK'
STORE_DIR = 'STORE_DIR'
TEMPLATE_LR = 'TEMPLATE_LR'
VERIFICATION_LR = 'VERIFICATION_LR'

# All macros
ALL_MACROS = [
    TEMPLATE_DIR, ORIGINAL_DIR, FAKE_DIR, ADDITIONAL_ORIGINALS, ADDITIONAL_FAKES, ADDITIONAL_TEST_ORIGINALS,
    ADDITIONAL_TEST_FAKES, BATCH_SIZE, BAD_INDICES, CLASSIFICATION, EPOCHS, LATENT_DIMENSION, MARGIN, MLP, MODEL,
    MULTI_GPU, PATIENCE, PERCENT_TRAIN, PERCENT_VAL, RUNS, SEED, STACK, STORE_DIR, TEMPLATE_LR, VERIFICATION_LR
]

DEFAULT_ARGS = {
    TEMPLATE_DIR: "./t_dir",
    ORIGINAL_DIR: "./x_dir",
    FAKE_DIR: "./f_dir",

    ADDITIONAL_ORIGINALS: ["./add_x"],
    ADDITIONAL_FAKES: ["./add_f"],
    ADDITIONAL_TEST_ORIGINALS: [],
    ADDITIONAL_TEST_FAKES: [],
    BATCH_SIZE: 8,
    BAD_INDICES: False,
    CLASSIFICATION: False,
    EPOCHS: 300,
    LATENT_DIMENSION: 100,
    MARGIN: 100,
    MLP: False,
    MODEL: "resnet18",
    MULTI_GPU: False,
    PATIENCE: 20,
    PERCENT_TRAIN: 0.4,
    PERCENT_VAL: 0.1,
    RUNS: 1,
    SEED: 0,
    STACK: True,
    STORE_DIR: "./result",
    TEMPLATE_LR: 0.01,
    VERIFICATION_LR: 0.01
}


def get_conf_path():
    """Sets the path to the configuration file as a program argument"""
    parser = argparse.ArgumentParser()
    parser.add_argument("conf", type=str, default='conf.json', help="Path to configuration file.")
    return vars(parser.parse_args())['conf']


class ConfigurationReader:
    def __init__(self, file_path, print_args=True):
        """Returns the program arguments as a dictionary accessible through the arguments MACROS"""
        self.file_path = file_path
        self.args = self._parse_conf_from_file(self.file_path)

        if print_args:
            # Printing args
            print(self.args)

        # Creating storing directory if it does not exist
        self.testing = False
        if not os.path.isdir(self.args[STORE_DIR]):
            os.mkdir(self.args[STORE_DIR])
        else:
            self.testing = True

    def log_conf(self):
        """Copies the configuration file to the result directory"""
        shutil.copy(self.file_path, os.path.join(self.args[STORE_DIR]))

    @staticmethod
    def _parse_conf_from_file(file_path):
        assert os.path.isfile(file_path)
        assert file_path.lower().endswith(".json")

        file = open(file_path)
        args = json.load(file)
        file.close()

        missing_keys = []
        for macro in ALL_MACROS:
            if macro not in args.keys():
                missing_keys.append(macro)
                args[macro] = DEFAULT_ARGS[macro]

        if len(missing_keys) > 0:
            warnings.warn(f"File {file_path} misses the following keys: {missing_keys}\n"
                          f"Using the default configurations for such keys.")

        return args

    def __getitem__(self, key):
        return self.args[key]
