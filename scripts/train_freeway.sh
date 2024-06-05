#!/bin/bash

# Change to the directory containing the Python script
cd ../

# Run the Python script (Train 3 seeds of the baseline approach. Equivalent to gymnasium)
CUDA_VISIBLE_DEVICES=10 python cleanrl/ppo_atari.py --seed 42 --track --capture-video --wandb-project-name Freeway_HA --env-id ALE/Freeway --exp-name baseline_ppo_42 
CUDA_VISIBLE_DEVICES=10 python cleanrl/ppo_atari.py --seed 73 --track --capture-video --wandb-project-name Freeway_HA --env-id ALE/Freeway --exp-name baseline_ppo_73 
CUDA_VISIBLE_DEVICES=10 python cleanrl/ppo_atari.py --seed 91 --track --capture-video --wandb-project-name Freeway_HA --env-id ALE/Freeway --exp-name baseline_ppo_91

# Modif1: Mono-Colored Freeway
CUDA_VISIBLE_DEVICES=10 python cleanrl/ppo_atari.py --seed 42 --track --capture-video --wandb-project-name Freeway_HA --env-id ALE/Freeway --exp-name c1_ppo_42 --modifs "color1"
CUDA_VISIBLE_DEVICES=10 python cleanrl/ppo_atari.py --seed 73 --track --capture-video --wandb-project-name Freeway_HA --env-id ALE/Freeway --exp-name c1_ppo_73 --modifs "color1"
CUDA_VISIBLE_DEVICES=10 python cleanrl/ppo_atari.py --seed 91 --track --capture-video --wandb-project-name Freeway_HA --env-id ALE/Freeway --exp-name c1_ppo_91 --modifs "color1"

# Modif2: Aligned Cars Freeway
CUDA_VISIBLE_DEVICES=10 python cleanrl/ppo_atari.py --seed 42 --track --capture-video --wandb-project-name Freeway_HA --env-id ALE/Freeway --exp-name ac_ppo_42 --modifs "stop2"
CUDA_VISIBLE_DEVICES=10 python cleanrl/ppo_atari.py --seed 73 --track --capture-video --wandb-project-name Freeway_HA --env-id ALE/Freeway --exp-name ac_ppo_73 --modifs "stop2"
CUDA_VISIBLE_DEVICES=10 python cleanrl/ppo_atari.py --seed 91 --track --capture-video --wandb-project-name Freeway_HA --env-id ALE/Freeway --exp-name ac_ppo_91 --modifs "stop2"

# Modif3: Stopped Cars Freeway
CUDA_VISIBLE_DEVICES=10 python cleanrl/ppo_atari.py --seed 42 --track --capture-video --wandb-project-name Freeway_HA --env-id ALE/Freeway --exp-name sc_ppo_42 --modifs "stop3"
CUDA_VISIBLE_DEVICES=10 python cleanrl/ppo_atari.py --seed 73 --track --capture-video --wandb-project-name Freeway_HA --env-id ALE/Freeway --exp-name sc_ppo_73 --modifs "stop3"
CUDA_VISIBLE_DEVICES=10 python cleanrl/ppo_atari.py --seed 91 --track --capture-video --wandb-project-name Freeway_HA --env-id ALE/Freeway --exp-name sc_ppo_91 --modifs "stop3"
