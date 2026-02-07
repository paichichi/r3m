#!/usr/bin/env bash

python -m evaluation.r3meval.core.hydra_launcher --multirun \
  hydra/launcher=basic hydra/sweeper=basic hydra/output=local \
  env="kitchen_micro_open-v3" \
  camera="left_cap" \
  pixel_based=true \
  embedding=resnet50 \
  num_demos=5 \
  env_kwargs.load_path=r3m \
  bc_kwargs.finetune=false \
  proprio=9 \
  job_name=try_r3m

