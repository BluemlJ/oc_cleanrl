#!/bin/bash

# Attention. This scirpt works after downloading the models. It can be necessary to download and rename them in your case.

# Change to the directory containing the Python script
cd ../submodules/HackAtari/hackatari/

# PPO Baselines
python eval.py -g Pong -p models/Pong/baseline_ppo_42.cleanrl_model 
python eval.py -g Pong -p models/Pong/baseline_ppo_73.cleanrl_model 
python eval.py -g Pong -p models/Pong/baseline_ppo_91.cleanrl_model 

python eval.py -g Freeway -p models/Freeway/baseline_ppo_42.cleanrl_model 
python eval.py -g Freeway -p models/Freeway/baseline_ppo_73.cleanrl_model 
python eval.py -g Freeway -p models/Freeway/baseline_ppo_91.cleanrl_model 

python eval.py -g Boxing -p models/Boxing/baseline_ppo_42.cleanrl_model 
python eval.py -g Boxing -p models/Boxing/baseline_ppo_73.cleanrl_model 
python eval.py -g Boxing -p models/Boxing/baseline_ppo_91.cleanrl_model 

# Var - Var
python eval.py -g Pong -p models/Pong/LE_ppo_42.cleanrl_model -m "lazy_enemy"
python eval.py -g Pong -p models/Pong/LE_ppo_73.cleanrl_model -m "lazy_enemy"
python eval.py -g Pong -p models/Pong/LE_ppo_91.cleanrl_model -m "lazy_enemy"

python eval.py -g Boxing -p models/Boxing/OA_ppo_42.cleanrl_model -m "one_armed"
python eval.py -g Boxing -p models/Boxing/OA_ppo_73.cleanrl_model -m "one_armed"
python eval.py -g Boxing -p models/Boxing/OA_ppo_91.cleanrl_model -m "one_armed"

python eval.py -g Freeway -p models/Freeway/C1_ppo_42.cleanrl_model -m "color1"
python eval.py -g Freeway -p models/Freeway/C1_ppo_73.cleanrl_model -m "color1"
python eval.py -g Freeway -p models/Freeway/C1_ppo_91.cleanrl_model -m "color1"

# Org - Var

python eval.py -g Pong -p models/Pong/baseline_ppo_42.cleanrl_model -m "lazy_enemy"
python eval.py -g Pong -p models/Pong/baseline_ppo_73.cleanrl_model -m "lazy_enemy"
python eval.py -g Pong -p models/Pong/baseline_ppo_91.cleanrl_model -m "lazy_enemy"

python eval.py -g Freeway -p models/Freeway/baseline_ppo_42.cleanrl_model -m "color1"
python eval.py -g Freeway -p models/Freeway/baseline_ppo_73.cleanrl_model -m "color1"
python eval.py -g Freeway -p models/Freeway/baseline_ppo_91.cleanrl_model -m "color1"

python eval.py -g Boxing -p models/Boxing/baseline_ppo_42.cleanrl_model -m "one_armed"
python eval.py -g Boxing -p models/Boxing/baseline_ppo_73.cleanrl_model -m "one_armed"
python eval.py -g Boxing -p models/Boxing/baseline_ppo_91.cleanrl_model -m "one_armed"
