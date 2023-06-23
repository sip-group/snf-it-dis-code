#!/bin/sh

PROJECT_DIR=indigo_supervised_ad

# Running multiple configurations
echo Running multiple configurations
for conf_file in "$@"
do
  # For reproducibility with multiple GPUs
  # CUBLAS_WORKSPACE_CONFIG=:16:8 python3 -u ${HOME}/${PROJECT_DIR}/src/main.py "$conf_file"

  # Non reproducible with multiple GPUs
  python3 -u ${HOME}/${PROJECT_DIR}/src/main.py "$conf_file"
done
echo All configurations were run
