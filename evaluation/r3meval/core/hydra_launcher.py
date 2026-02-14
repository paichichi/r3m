# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time as timer
import hydra
import multiprocessing
from omegaconf import DictConfig, OmegaConf
from .train_loop import bc_train_loop

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

    # job_data = OmegaConf.structured(OmegaConf.to_yaml(job_data))
    print("before:", type(job_data))
    job_data = OmegaConf.structured(OmegaConf.to_yaml(job_data))
    print("after:", type(job_data))

    print("device:", OmegaConf.select(job_data, "device"))

    job_data['cwd'] = cwd
    OmegaConf.save(config=job_data, f="job_config.yaml")

    print(OmegaConf.to_yaml(job_data))

    bc_train_loop(job_data)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    configure_jobs()