#!/usr/bin/env bash

python eval_v2/core/hydra_launcher.py --multirun \
  hydra/launcher=basic hydra/output=local \
  hydra.sweep.subdir='job_${hydra.job.num}' \
  device=cuda \
  eval_frequency=1000 \
  env=FrankaKitchen-v1 \
  task="microwave" \
  pixel_based=true \
  embedding=resnet50 \
  num_demos=1 \
  env_kwargs.load_path=r3m \
  bc_kwargs.finetune=false \
  proprio=9 \
  job_name=new_eval_v2_test


#task="bottom burner","light switch","slide cabinet","hinge cabinet","microwave"