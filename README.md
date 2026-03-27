# 🤖 Robotic World Model

> A learned internal model that enables robots to understand, predict, and interact with their environment through experience.

---

## 📖 Table of Contents

- [Robotic World Model](#-robotic-world-model)
- [Setup](#-setup)
- [Train](#-train)
- [Inference](#-inference)
- [TODOs](#-todos)

---

## 🧠 Robotic World Model

A **Robotic World Model** is a learned internal representation that allows a robot to simulate and reason about the physical world — without requiring live sensor input at every decision step. Rather than reacting purely to raw observations, the robot learns a compressed, structured understanding of how the world behaves over time.

At its core, the world model ingests sequences of observations (e.g., joint states, IMU readings, camera frames) and actions, then learns to predict future states. This enables the robot to "imagine" the consequence of taking an action before executing it — dramatically improving sample efficiency and planning capability.

### How It Works

```
Observation (oₜ) ──┐
                   ├──▶  Encoder  ──▶  Latent State (zₜ)  ──▶  GRU  ──▶  zₜ₊₁  ──▶  Decoder  ──▶  Predicted Obs (ô)
Action     (aₜ) ──┘
```

1. **Encoder** — Maps high-dimensional observations into a compact latent space `zₜ`.
2. **Recurrent Core (GRU)** — Maintains a hidden state that captures temporal dynamics across timesteps.
3. **Transition Model** — Predicts the next latent state `zₜ₊₁` given the current state and action `aₜ`.
4. **Decoder** — Reconstructs the predicted observation `ô` from the latent state for supervision and interpretability.

### Why It Matters

| Capability | Benefit |
|---|---|
| **State prediction** | Robot can anticipate future conditions before they occur |
| **Model-based planning** | Enables rollout simulation for safer, more efficient policy search |
| **Data efficiency** | Learns rich representations from offline transition data |
| **Generalization** | Latent structure captures physics that transfers across terrains and tasks |
| **Reduced real-world risk** | Test behaviors in imagination before deploying on hardware |

### Architecture: `wm_gru`

This implementation uses a **GRU-based world model** (`wm_gru`) trained on prerecorded locomotion rollouts. The GRU hidden state serves as the robot's "memory," encoding recent history to produce accurate multi-step predictions even under partial observability — a critical property for real legged robots operating in noisy, unstructured environments.

---

## ⚙️ Setup

### 1. Clone the Repository

```bash
git clone <repo_link>
```

### 2. Initialize Submodules

```bash
git submodule update --init --recursive
```

### 3. Configure Base File

Copy and paste the `base.py` file into the appropriate directory as required by your environment configuration:

```bash
cp base.py <target_directory>/base.py
```

> **Note:** Ensure your Python environment is set up with the required dependencies before proceeding. It is recommended to use a virtual environment (e.g., `conda` or `venv`).

---

## 🏋️ Train

Train the GRU-based world model on prerecorded transition data:

```bash
python3 main.py \
  --db_dir_name pretraining_rollouts/1000000_transitions \
  --run_mode train \
  --model_type wm_gru
```

| Argument | Description |
|---|---|
| `--db_dir_name` | Path to the rollout dataset directory |
| `--run_mode` | Set to `train` to run the training loop |
| `--model_type` | Model architecture to use (e.g., `wm_gru`) |

Training checkpoints will be saved automatically and timestamped for versioning.

---

## 🔍 Inference

Run inference using a pretrained world model checkpoint:

```bash
python3 main.py \
  --db_dir_name pretraining_rollouts/1000000_transitions \
  --run_mode train \
  --model_type wm_gru \
  --model_dir_name wm_gru_2026-03-25_23:26:29/wm_gru-epoch_50.pth
```

| Argument | Description |
|---|---|
| `--model_dir_name` | Path to the saved model checkpoint (`.pth` file) |

> Replace `wm_gru_2026-03-25_23:26:29/wm_gru-epoch_50.pth` with the path to your own trained checkpoint.

---

## 📋 TODOs

Planned features and research directions for future development:

- [ ] **Fine-tunable** — Support task-specific fine-tuning of the pretrained world model on downstream locomotion tasks
- [ ] **Obstacle avoidance & environment adaptation** — Enable the model to predict and respond to dynamic obstacles and changing terrain conditions
- [ ] **Human-like gait** — Learn and reproduce naturalistic, energy-efficient bipedal movement patterns inspired by human locomotion
- [ ] **Diverse gait patterns** — Generalize across multiple locomotion modes (walk, trot, bound, crawl) within a single unified model

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">Built with curiosity and a lot of robot falls 🦾</p>
