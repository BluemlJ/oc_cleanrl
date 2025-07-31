# OC-CleanRL

This fork enables the usage of OCAtari and HackAtari wrappers for Gymnasium instead of pure Gymnasium. OCAtari and HackAtari offer advanced wrappers that extract and use object-centered representations, enabling more interpretable observations and potentially improving training efficiency compared to raw pixel-based inputs. The goal is to use object-centered input representations instead of pure pixel-based ones. Currently, our experiments focus on the prominent Atari environment, particularly games like Pong, Breakout, and Space Invaders, to evaluate the effectiveness of object-centered representations.

> **ℹ️ Support for Gymnasium and v5**  
> [Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium) is the next generation of [`openai/gym`](https://github.com/openai/gym) that will continue to be maintained and introduce new features. Please see their [announcement](https://farama.org/Announcing-The-Farama-Foundation) for further details. CleanRL has already migrated to `gymnasium` (see [vwxyzjn/cleanrl#277](https://github.com/vwxyzjn/cleanrl/pull/277)). Primarily, we base our training on the new v5 versions of the Atari games; see [ALE-Farama](https://ale.farama.org/environments/).

---

## About OCAtari

You can find the OCAtari repository at [OCAtari GitHub](https://github.com/BluemlJ/OCAtari).

OCAtari is a specialized wrapper designed for Atari environments that transforms pixel-based observations into object-centered representations. This allows agents to interact with the environment using more interpretable and structured inputs. By extracting meaningful object-level features, OCAtari enhances the efficiency and robustness of reinforcement learning models, especially in tasks where pixel-level noise can hinder performance.

---

## CleanRL (Clean Implementation of RL Algorithms) - by Costa Huang

CleanRL is a Deep Reinforcement Learning library that provides high-quality, single-file implementations with research-friendly features. The implementation is clean and simple yet scalable to run thousands of experiments using AWS Batch. Key features of CleanRL include:

- **📜 Single-file implementation**
  - All details about an algorithm variant are in a single, standalone file.
  - For example, `ppo_atari.py` has only 340 lines of code but contains all implementation details on how PPO works with Atari games. This makes it a great reference implementation for those who do not want to read an entire modular library.

- **📊 Benchmarked Implementations**
  - Explore 7+ algorithms and 34+ games at [CleanRL Benchmarks](https://benchmark.cleanrl.dev).

- **📈 TensorBoard Logging**
- **🪛 Local Reproducibility via Seeding**
- **🎮 Gameplay Video Capturing**
- **🧫 Experiment Management with [Weights and Biases](https://wandb.ai/site)**
- **💸 Cloud Integration**
  - Docker and AWS support for seamless scaling.

We keep this fork up to date with the original CleanRL master branch to enable further adaptations of algorithms for object-centered representations. For more details, you can:

> ⚠️ **NOTE**: CleanRL is *not* a modular library and therefore it is not meant to be imported. At the cost of duplicate code, we make all implementation details of a DRL algorithm variant easy to understand, so CleanRL comes with its own pros and cons. You should consider using CleanRL if you want to 1) understand all implementation details of an algorithm's variant or 2) prototype advanced features that other modular DRL libraries do not support (CleanRL has minimal lines of code so it gives you great debugging experience and you don't have do a lot of subclassing like sometimes in modular DRL libraries).

> **⚠️ Note**  
> CleanRL is *not* a modular library. This means it is not meant to be imported as a library. Instead, it is designed to make all implementation details of a DRL algorithm variant easy to understand, at the cost of duplicate code. Use CleanRL if you want to:
> 1. Understand all implementation details of an algorithm variant.
> 2. Prototype advanced features that modular DRL libraries may not support. CleanRL’s minimal lines of code enable easier debugging and avoid extensive subclassing required in modular libraries.

---

## Getting Started with OC-CleanRL

### Prerequisites
- Python >= 3.9, < 3.13
- pip

### Running Experiments Locally

1. **Clone the repository**

   ```bash
   git clone git@github.com:BluemlJ/oc_cleanrl.git && cd oc_cleanrl
   ```

2. **Install dependencies**

   ```bash
   # Core dependencies
   pip install -r requirements/requirements.txt
   
   # Atari-specific dependencies
   pip install -r requirements/requirements-atari.txt
   ```

3. **Enable OCAtari/HackAtari**

   ```bash
   pip install ocatari
   ```

4. **Start a training run**

   ```bash
   python cleanrl/ppo_atari_occam.py --env-id ALE/Pong-v5 --obs_mode masked_dqn_planes --architecture PPO --backend OCAtari
   ```

---

## Tracking Results with W&B

You can track the results of training runs using [Weights and Biases](https://wandb.ai/): W&B allows you to visualize key metrics, compare runs across different experiments, and easily share results with collaborators. For instance, you can monitor training progress, analyze model performance, and debug issues more effectively using W&B's interactive dashboards.

```bash
python cleanrl/ppo_atari_occam.py \
  --env-id ALE/${game_name}-v5 \
  --backend OCAtari \
  --obs_mode masked_dqn_planes \
  --architecture PPO \
  --track \
  --capture_video \
  --wandb-project-name OCAtari \
  --exp-name "OCCAM_ppo"
```

Additional W&B settings can be adjusted directly in the training scripts.

---

## Next Steps and Contributing

If you have any questions or need support, feel free to reach out by creating an issue on the [GitHub repository](https://github.com/BluemlJ/oc_cleanrl/issues).

### Next Steps
- Experiment with different Atari environments to explore object-centered representations further.
- Compare the performance of pixel-based and object-centered models across various tasks.
- Enable envpool and more methods to use object-centered inputs

### Contributing
We welcome contributions to OC-CleanRL! If you'd like to contribute:
1. Fork the repository and create a new branch for your feature or bugfix.
2. Follow the existing coding style and add relevant tests.
3. Submit a pull request and include a detailed description of your changes.
