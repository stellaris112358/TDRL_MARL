# TdRL

repo for Test-driven Reinforcement Learning

## Quick Start

Recommand: Ubuntu 24.04, python>=3.10, 4 core CPU or higher, 16GB RAM or higher, NVIDIA RTX 3060 or higher.

install system requirements:

```shell
sudo apt install patchelf libosmesa6-dev
sudo apt install libegl1-mesa libegl1-mesa-dev libgl1-mesa-glx libgl1-mesa-dev libgles2-mesa-dev
sudo apt install ffmpeg
```

download mujoco:

```shell
wget https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
mkdir -p ~/.mujoco
tar -xvf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
rm mujoco210-linux-x86_64.tar.gz
```

add following lines to `~/.bashrc` or `~/.zshrc`:

```shell
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
```

create conda env:

```shell
conda create -n tdrl python=3.10
```

install python requirements:

```shell
pip install -r requirements.txt
cd dmc2gym
pip install .
```

train agent:

```shell
python train_tdrl.py env=walker_run seed=0 device=cuda num_train_steps=1000_000
```

## Config

We use `hydra` to config the algorithm settings. You can change the config by modify the `yaml` files in `conf` directory, or add the option in cmds.
