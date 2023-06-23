#!/bin/sh

PROJECT_DIR=cdp_unsupervised_ad
CONF_DIR=$HOME/Projects/$PROJECT_DIR/confs/cross

for f in "$CONF_DIR"/*
do
      srun python3 -u \
      "${HOME}"/Projects/${PROJECT_DIR}/src/main.py \
      --conf "${f}"
done
