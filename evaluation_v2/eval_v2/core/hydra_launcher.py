# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time as timer
import hydra
import multiprocessing
from omegaconf import DictConfig, OmegaConf
import torch
from eval_v2.utils.device import pick_device

from eval_v2.core.train_loop import bc_train_loop

cwd = os.getcwd()

# ===============================================================================
# Process Inputs and configure job
# ===============================================================================
@hydra.main(config_name="BC_config", config_path="config")

def configure_jobs(job_data:dict) -> None:
    os.environ['GPUS'] = os.environ.get('SLURM_STEP_GPUS', '0')
    
    print("========================================")
    print("Job Configuration")
    print("========================================")

    # ---- device debug ----
    requested = str(job_data.get("device", "auto"))
    device = pick_device(requested)

    print("requested device:", requested)
    print("selected device :", str(device))
    print("torch version   :", torch.__version__)
    print("cuda available  :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda device cnt :", torch.cuda.device_count())
        print("cuda name[0]    :", torch.cuda.get_device_name(0))
    print("mps available   :", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())

    job_data["device"] = str(device)
    if "env_kwargs" in job_data:
        job_data["env_kwargs"]["device"] = str(device)
    # ----------------------
    # job_data = OmegaConf.structured(OmegaConf.to_yaml(job_data))

    job_data = OmegaConf.to_container(job_data, resolve=True)

    job_data['cwd'] = cwd
    with open('job_config.json', 'w') as fp:
        OmegaConf.save(config=job_data, f=fp.name)
    # print(OmegaConf.to_yaml(job_data))

    bc_train_loop(job_data)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    configure_jobs()