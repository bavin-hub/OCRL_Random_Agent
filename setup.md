# Setup Guide

## Table of Contents

- [1. MiniConda Setup](#1-miniconda-setup)
- [2. Create Environment](#2-create-environment)
- [3. Clone Repository](#3-clone-repository)
- [4. Baseline Policy](#4-baseline-policy)
  - [Download Pretrained Weights](#41-download-pretrained-weights)
  - [Test Baseline Policy](#42-test-baseline-policy)
  - [Train from Scratch](#43-train-from-scratch)
- [5. Rollout Sampling](#5-rollout-sampling)
  - [Use Existing Rollouts](#51-use-existing-rollouts)
  - [Sample Your Own Rollouts](#52-sample-your-own-rollouts)


---

## 1. MiniConda Setup

```bash
cd
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init --all
source ~/.bashrc
```

---

## 2. Create Environment

```bash
conda create -n random_agent python=3.11
conda activate random_agent
```

---

## 3. Clone Repository

```bash
git clone git@github.com:bavin-hub/OCRL_Random_Agent.git
cd OCRL_Random_Agent
git submodule update --init --recursive
```

Install dependencies for the `unitree_rl_mjlab` submodule:

```bash
cd data/baseline_policies/unitree_rl_mjlab
git checkout baseline_policy
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
pip install -e .
cd ../../..
```

---

## 4. Baseline Policy

### 4.1 Download Pretrained Weights

1. Download the zip from: `<unitree_rl_mjlab_weights_link>`
2. Place it inside `data/baseline_policies/unitree_rl_mjlab/`
3. Unzip and clean up:

```bash
unzip data/baseline_policies/unitree_rl_mjlab/logs.zip
sudo rm -rf data/baseline_policies/unitree_rl_mjlab/logs.zip
```

### 4.2 Test Baseline Policy

```bash
cd data/baseline_policies/unitree_rl_mjlab
python scripts/play.py Unitree-G1-Flat \
  --checkpoint_file=logs/rsl_rl/g1_velocity/2026-xx-xx_xx-xx-xx/model_xx.pt
```

### 4.3 Train from Scratch

```bash
cd data/baseline_policies/unitree_rl_mjlab
python scripts/train.py Unitree-G1-Flat \
  --env.scene.num-envs=4096
```

---

## 5. Rollout Sampling

Two options are available: use pre-collected rollout data or sample your own.

### 5.1 Use Existing Rollouts

1. Create the rollouts directory:

```bash
mkdir -p data/pretraining_rollouts
```

2. Download the zip from:
   > https://drive.google.com/file/d/1yV5Nhvdgugw56TS_FYcb-22JIRg18YL_/view?usp=drive_link

3. Place it inside `data/pretraining_rollouts/`, then unzip and clean up:

```bash
unzip data/pretraining_rollouts/1000000_transitions.zip -d data/pretraining_rollouts
sudo rm -rf ${PWD}/data/pretraining_rollouts/1000000_transitions.zip
```

### 5.2 Sample Your Own Rollouts

1. Copy the `base.py` viewer file into the mjlab package:

```bash
cp base.py ~/miniconda3/envs/random_agent/lib/python3.11/site-packages/mjlab/viewer/
```

2. Run the [test baseline policy](#42-test-baseline-policy) command — rollouts will be saved automatically to `data/pretraining_rollouts/`.

> **Note:** `num_transitions` and `num_trajectories` values can be configured inside `base.py`.


