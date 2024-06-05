#!/bin/bash

# Change to the directory containing the Python script
cd ../

# Run the Python script (Train 3 seeds of the baseline approach. Equivalent to gymnasium)
CUDA_VISIBLE_DEVICES=10 python cleanrl/ppo_atari.py --seed 42 --track --capture-video --wandb-project-name Boxing_HA --env-id ALE/Boxing --exp-name baseline_ppo_42 
CUDA_VISIBLE_DEVICES=10 python cleanrl/ppo_atari.py --seed 73 --track --capture-video --wandb-project-name Boxing_HA --env-id ALE/Boxing --exp-name baseline_ppo_73 
CUDA_VISIBLE_DEVICES=10 python cleanrl/ppo_atari.py --seed 91 --track --capture-video --wandb-project-name Boxing_HA --env-id ALE/Boxing --exp-name baseline_ppo_91

# Modif1: One Armed Boxing
CUDA_VISIBLE_DEVICES=10 python cleanrl/ppo_atari.py --seed 42 --track --capture-video --wandb-project-name Boxing_HA --env-id ALE/Boxing --exp-name OA_ppo_42 --modifs "one_armed"
CUDA_VISIBLE_DEVICES=10 python cleanrl/ppo_atari.py --seed 73 --track --capture-video --wandb-project-name Boxing_HA --env-id ALE/Boxing --exp-name OA_ppo_73 --modifs "one_armed"
CUDA_VISIBLE_DEVICES=10 python cleanrl/ppo_atari.py --seed 91 --track --capture-video --wandb-project-name Boxing_HA --env-id ALE/Boxing --exp-name OA_ppo_91 --modifs "one_armed"

# Modif1: Drunken Boxing
CUDA_VISIBLE_DEVICES=10 python cleanrl/ppo_atari.py --seed 42 --track --capture-video --wandb-project-name Boxing_HA --env-id ALE/Boxing --exp-name DB_ppo_42 --modifs "drunken_boxing"
CUDA_VISIBLE_DEVICES=10 python cleanrl/ppo_atari.py --seed 73 --track --capture-video --wandb-project-name Boxing_HA --env-id ALE/Boxing --exp-name DB_ppo_73 --modifs "drunken_boxing"
CUDA_VISIBLE_DEVICES=10 python cleanrl/ppo_atari.py --seed 91 --track --capture-video --wandb-project-name Boxing_HA --env-id ALE/Boxing --exp-name DB_ppo_91 --modifs "drunken_boxing"