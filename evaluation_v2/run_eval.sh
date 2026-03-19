#!/usr/bin/env bash

# 运行 Hydra 实验
python eval_v2/core/hydra_launcher.py --multirun \
  hydra/launcher=local hydra/output=local \
  device=cuda \
  eval_frequency=1000 \
  env=FrankaKitchen-v1 \
  task="bottom burner","light switch" \
  pixel_based=true \
  embedding=resnet50 \
  num_demos=1 \
  env_kwargs.load_path=r3m \
  bc_kwargs.finetune=false \
  proprio=9 \
  job_name=new_eval_v2_test


#task="bottom burner","light switch","slide cabinet","hinge cabinet","microwave"
# task="bottom burner, light switch, slide cabinet, hinge cabinet, microwave"