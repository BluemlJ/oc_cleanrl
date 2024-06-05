#!/bin/bash

# Change to the directory containing the Python script
cd ../

# Run the Python script (Train 3 seeds of the baseline approach. Equivalent to gymnasium)
CUDA_VISIBLE_DEVICES=10 python cleanrl/ppo_atari.py --seed 42 --track --capture-video --wandb-project-name Pong_HA --env-id ALE/Pong --exp-name baseline_ppo_42 
CUDA_VISIBLE_DEVICES=10 python cleanrl/ppo_atari.py --seed 73 --track --capture-video --wandb-project-name Pong_HA --env-id ALE/Pong --exp-name baseline_ppo_73 
CUDA_VISIBLE_DEVICES=10 python cleanrl/ppo_atari.py --seed 91 --track --capture-video --wandb-project-name Pong_HA --env-id ALE/Pong --exp-name baseline_ppo_91

# Modif1: Lazy Enemy Pong
CUDA_VISIBLE_DEVICES=10 python cleanrl/ppo_atari.py --seed 42 --track --capture-video --wandb-project-name Pong_HA --env-id ALE/Pong --exp-name LE_ppo_42 --modifs "lazy_enemy"
CUDA_VISIBLE_DEVICES=10 python cleanrl/ppo_atari.py --seed 73 --track --capture-video --wandb-project-name Pong_HA --env-id ALE/Pong --exp-name LE_ppo_73 --modifs "lazy_enemy"
CUDA_VISIBLE_DEVICES=10 python cleanrl/ppo_atari.py --seed 91 --track --capture-video --wandb-project-name Pong_HA --env-id ALE/Pong --exp-name LE_ppo_91 --modifs "lazy_enemy"
