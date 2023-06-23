#!/bin/sh

PROJECT_DIR=cdp_unsupervised_ad
CONF_DIR=$HOME/$PROJECT_DIR/confs/wifs

for f in "$CONF_DIR"/*
do
      srun python3 -u \
      "${HOME}"/${PROJECT_DIR}/src/main.py \
      --conf "${f}"
done
