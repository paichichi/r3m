#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=""
export GPUS=""

python -m r3meval.core.hydra_launcher --multirun \
  hydra/launcher=local hydra/output=local \
  hydra.sweep.subdir='job_${hydra.job.num}' \
  device=cpu \
  eval_frequency=1000 \
  env=kitchen_knob1_on-v3,kitchen_light_on-v3,kitchen_sdoor_open-v3,kitchen_ldoor_open-v3,kitchen_micro_open-v3 \
  camera=left_cap \
  pixel_based=true \
  embedding=resnet50 \
  num_demos=1 \
  env_kwargs.load_path=r3m \
  bc_kwargs.finetune=false \
  proprio=9 \
  job_name=try_r3m




#env=kitchen_knob1_on-v3,kitchen_light_on-v3,kitchen_sdoor_open-v3,kitchen_ldoor_open-v3,kitchen_micro_open-v3 \