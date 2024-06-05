# Training

Training follows the basic usage explained in the cleanRL documentation (https://docs.cleanrl.dev/).
To log the experiments, we used Weights and Biases (https://wandb.ai).

To start a training run without modifications, we used the following command:

```
CUDA_VISIBLE_DEVICES=0 python cleanrl/ppo_atari.py --seed 42 \
    --track --capture-video --wandb-project-name Hackatari_frostbite \
    --env-id ALE/Frostbite --exp-name baseline_ppo_42 
```

 To use a game variation as explained in the paper, one can add modifications to the environment

```
CUDA_VISIBLE_DEVICES=0 python cleanrl/ppo_atari.py --seed 42 \
   --track --capture-video --wandb-project-name Hackatari_frostbite \
   --env-id ALE/Frostbite --exp-name baseline_ppo_42 \
   --modifs static60
```

Models can be safed in the process.
Currently the ppo_atari and the c51_atari are modified to work with HackAtari. Feel free to adapt other training scripts accordingly. 

# Testing or Playing

To test trained agents within a HackAtari environment, one need access to the modelfile created within the training process. The final model can also be found over wandb (should tracking be enabled). 

The easiest way to test a model is the eval.py within the submodule/HackAtari/hackatari folder. 

```
python eval.py -g ALE/Frostbite -p models/baseline_ppo.cleanrl_model -m static60
```

If you want to test a game variation for yourself, you can also run the run.py scirpt instead

```
python run.py -g ALE/Frostbite -hu -m static60
```


