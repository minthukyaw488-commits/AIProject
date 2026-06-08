<div align="center">

# CADS-VLA: Critical Re-evaluation Text for Dual-System Vision-Language-Action Models

**A study repository exploring how a small re-evaluation Actor can improve a frozen Planner on long-horizon robotic manipulation.**

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Study--Use-lightgrey?style=flat-square)](#license)
[![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)](#)

[Overview](#overview) · [Architecture](#architecture) · [Setup](#setup) · [Usage](#usage) · [My Contributions](#my-contributions) · [References](#references)

</div>

---

## Overview

> **TL;DR.** A frozen OpenVLA-7B Planner proposes initial robot actions. A lightweight SmolVLM-500M Actor critically re-evaluates and refines them, with LoRA + GRPO training driven by LIBERO-Long task rewards. The two models run in separate environments and communicate via zeroMQ.

Single-model VLA systems perform well on short-horizon tasks but degrade sharply on long-horizon ones — OpenVLA reports **~85% success on LIBERO-Spatial but only ~54% on LIBERO-Long**. This project tests whether a second, smaller VLM acting as a *critic* of the Planner's proposals can close that gap without retraining the Planner.

### Key Ideas

| | |
| --- | --- |
|  **Dual-system design** | Frozen Planner + LoRA-trained Actor — only the Actor learns |
|  **Critical text** | Actor produces a short critique before emitting the refined action tokens |
|  **GRPO on rewards** | Group-relative policy optimization over LIBERO success/failure |
|  **Lightweight Actor** | SmolVLM-500M chosen for 8GB-class GPU feasibility |
|  **Process isolation** | Planner (CPU) and Actor (GPU) decoupled via zeroMQ |

---
### Motivation & Problem Statement
Existing single-model VLAs map vision and language directly to robot actions — with no intermediate step where the model checks whether its own proposed action is appropriate. In long-horizon tasks, one early bad decision can cascade into total failure.
This isn't speculation — it shows up quantitatively in benchmarks. The CoT-VLA paper (Zhao et al., 2025) evaluates several VLAs under identical conditions (3 seeds × 500 episodes). Models without a re-evaluation step score high on short tasks, but collapse to ~50% on LIBERO-Long.
LIBERO Benchmark — Success Rate by Task (identical evaluation conditions)
ModelReasoning stepSpatialObjectGoalLongDiffusion PolicyNone78.3%92.5%68.3%50.5%OctoNone78.9%85.7%84.6%51.1%OpenVLA-7BNone84.7%88.4%79.2%53.7%CoT-VLA-7BYes (visual CoT)87.5%91.6%87.6%69.0%

All four numbers are from the same Table 1 in the CoT-VLA paper, evaluated under identical conditions.
Across different architectures (autoregressive, diffusion), models without an intermediate reasoning step collapse to ~50% on Long tasks. CoT-VLA, which inserts a visual chain-of-thought step, reaches 69.0% on Long — closing most of the gap with shorter tasks. The variable is "presence of a review step before acting."
Source: Zhao et al., "CoT-VLA: Visual Chain-of-Thought Reasoning for VLA Models", arXiv:2503.22020, Table 1.

→ In short: the presence or absence of a "review and correct" step before acting materially affects long-horizon success.
This project implements that review step as role separation through a second model (Actor).
The cause is well-understood — supervised imitation learning is fragile over long sequences because errors compound. A mid-sequence step that re-examines the model's own judgment can break that compounding cycle.
This project addresses it through role separation:

Planner (OpenVLA) — quickly proposes a first-pass action
Actor (SmolVLM-500M) — reviews the proposal with critical re-evaluation text and corrects it

A dual-system architecture mirroring a human "think → check → act" flow.

### Project Status
The architecture and pipeline are fully implemented. Training was carried through to the stages reachable under available compute/time constraints; further training is future work.
StageStatusDual-system architecture design✅ CompletePlanner (OpenVLA) inference pipeline✅ CompleteActor (SmolVLM-500M) — tokenizer extension + projection layer✅ CompleteZeroMQ inter-process communication✅ CompleteSFT — Actor action-token output + text generation🟡 Partial, qualitatively verifiedGRPO reinforcement learning⬜ Not run (code implemented, large-scale training deferred)LIBERO-Spatial quantitative evaluation⬜ Not run (compute constraints)

The project reached the milestone of "a system that runs end-to-end with its core learning step verified to work as intended."
Quantitative benchmark results are deferred to future work.

## Architecture

### Model Structure

![Figure 1 — Model Structure](https://github.com/user-attachments/assets/edbb073d-d40e-452a-873f-a4de54bc41b6)

The Planner (OpenVLA-7B, frozen) emits action token IDs from the input image and instruction. These IDs are translated into the Actor's embedding space through OpenVLA's action embedding table and a learned projection layer, then concatenated with SmolVLM's image and text embeddings. The Actor's LLM produces a refined output — a short critical text plus 7 action tokens — which is detokenized into an action vector for the robot. The robot's success/failure on the LIBERO task forms the reward signal that drives GRPO training.

### Process Flow

![Figure 2 — Model Process](https://github.com/user-attachments/assets/15b2c0a8-c4bb-4968-bf58-09bde7475a99)

Both models process image and text through their own vision encoders and LLMs. The Planner's action tokens are mapped through the frozen embedding table, dimension-aligned via the projection layer, and concatenated with the Actor's processor output. SmolVLM's tokenizer has been extended with OpenVLA's 256 action tokens. After transformer attention, the output is split: text via the detokenizer, action tokens as a 7-dimensional action vector.

### Training Flow

![Figure 3 — Flow Chart](https://github.com/user-attachments/assets/e5adbf70-5ef6-4e22-94d6-8828ed1dc6be)

All image and text data come from the LIBERO environment. The Actor communicates with the Planner server via zeroMQ. During tokenizer initialization, 256 action tokens are imported from OpenVLA (add → embedding table → resize → projection layer). The GRPO training loop follows the RLinf framework: `collect_rollout` gathers group-sized trajectories from LIBERO, critical text and action tokens are inferred, rewards and losses are computed, and weights are updated — training only the LoRA adapter on the Actor.

### Actor Tokenizer Process

![Figure 4 — Actor Tokenizer](https://github.com/user-attachments/assets/1c0fbf23-fe96-475a-8827-3fd545224f3e)

Each input modality is converted into embeddings via its own encoder. The input merge and action injection hook project them into a shared embedding space and concatenate them. Image and text pass through SmolVLM's processor at this stage. The merged representation then flows through the transformer to produce the output.

### Supervised Fine-Tuning

![Figure 5 — SFT](https://github.com/user-attachments/assets/f106f294-3e26-4129-ab98-1bde2140310e)

Starting GRPO from scratch is unstable — the Actor doesn't yet know the expected output format (critical text + action tokens), causing constant penalties and convergence failure. A 4-stage SFT phase teaches the Actor the basic output format first, after which GRPO refines the policy.

---

## Components

| Component | Configuration |
| --- | --- |
| **Planner** | OpenVLA-7B (fine-tuned on libero-10), 4-bit, frozen, CPU inference |
| **Actor (primary)** | SmolVLM-500M, 4-bit, LoRA, GRPO ✅ |
| **Actor (alt)** | Qwen2.5-VL-3B, 4-bit, LoRA, GRPO |
| **Simulator** | LIBERO `libero_10` (LIBERO-Long, 10 long-horizon tasks) |
| **Image resolution** | 224 × 224 (matches SmolVLM input) |
| **Communication** | zeroMQ REQ/REP, localhost |

---

## Repository Structure

```
.
├── openvla_planner/         # Frozen Planner — runs on CPU, opens zeroMQ server
│   ├── openvla_inference_code.py
│   └── action_tokenizer.py
│
├── qwen_actor/              # Alternative Actor (Qwen2.5-VL 3B)
│   ├── actor_action_tokenizer.py
│   ├── projection_layer.py
│   └── actor_model.py
│
├── SmolVLM_actor/           # Primary Actor (SmolVLM 500M) ✅
│   ├── smol_action_tokenizer.py
│   ├── smol_projection_layer.py
│   └── smol_actor_model.py
│
├── train_file/              # Training pipeline
│   ├── train.py             # main GRPO + LIBERO loop
│   ├── smol_train.py        # SmolVLM-specific training (RLinf-style)
│   └── smol_sft.py          # 4-stage SFT pretraining
│
├── SFT/                     # SFT setup
├── assets/
│   └── make_embeddings.py   # Run once, downloads OpenVLA action embeddings
├── checkpoints/
│   ├── sft/                 # (legacy — vision encoder bug, not used)
│   └── sft2/                # ✅ vision-encoder version, in use
└── logs/                    # Training run logs
```

---

## Setup

> Two separate conda environments are required because OpenVLA and SmolVLM cannot share a single PyTorch installation.

**Planner environment (OpenVLA, CPU)**

```bash
conda create -n openvla python=3.10 -y
conda activate openvla
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements_openvla.txt
```

**Actor environment (Qwen / SmolVLM, GPU)**

```bash
conda create -n qwen python=3.10 -y
conda activate qwen
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements_qwen.txt --no-deps
git clone https://github.com/Lifelong-Robot-Learning/LIBERO && cd LIBERO && pip install -e . && cd ..
```

**One-time setup**

```bash
conda activate openvla
python assets/make_embeddings.py    # downloads OpenVLA action embeddings
```

---

## Usage

The Planner and Actor run as two separate processes connected via zeroMQ. Start the Planner **first**, then the Actor.

**Terminal 1 — Planner**

```bash
conda activate openvla
python openvla_planner/openvla_inference_code.py
```

**Terminal 2 — Actor + training**

```bash
conda activate qwen
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train_file/smol_train.py
```

### Monitoring

```bash
# Live GPU usage (refresh every 0.5s)
watch -n 0.5 nvidia-smi

# Per-process GPU usage
nvidia-smi pmon -c 1

# Log VRAM to CSV during training
nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 1 >> vram_log.csv &

# Combined training + log capture
python train_file/smol_train.py 2>&1 | tee train_log.txt
```

### Stable long runs over SSH (tmux)

VS Code's remote SSH drops periodically — use `tmux` so training survives disconnects.

```bash
tmux new -s planner   # start Planner inside its own session
tmux new -s train     # start Actor inside another
# detach: Ctrl-b then d
# reattach: tmux attach -t <name>
```

---

## My Contributions

> This repository is my personal study copy of a team research project. The codebase was implemented by the team; this section documents what I personally contributed and what I'm using this repo to study.

**My contributions to the project:**

-  **Framework research** — Investigated RLinf-VLA and identified that it does not directly support Qwen-as-Actor (Qwen2.5-VL is supported only as a VLM in RLinf, not as a robotic action model), informing the team's decision to apply RLinf's rollout/training separation idea rather than swap frameworks.
-  **VRAM analysis & figures** — Produced VRAM breakdown analysis (activation + gradient as dominant cost), before/after comparison charts, architecture diagrams, and the zeroMQ communication diagram used in team materials.
-  **Documentation** — Caught and listed typos in the original README (e.g. `setpu` → `setup`, `colliect_rollout` → `collect_rollout`); proposed a cleaned-up, table-driven README structure.
-  **Presentation materials** — Built the LIBERO + zeroMQ section of the project presentation, including slides, a Korean speaker script, anticipated Q&A, and supporting figures.
-  **Translation & accessibility** — Maintained an English version of the project README (this document) for international readers.

**What I'm using this repo to study:**

- Dual-system VLA architectures and the role of a re-evaluation Actor
- The mechanics of GRPO and reward function design for robot manipulation
- LoRA + 4-bit quantization for fitting large VLMs on small GPUs
- Token-level loss weighting (critical text tokens vs. action tokens)
- Inter-process communication patterns (REQ/REP via zeroMQ) for mixed CPU/GPU pipelines
- LIBERO benchmark internals — `OffScreenRenderEnv`, BDDL task definitions, `get_state`/`set_state` for fair GRPO group sampling

---

## Why These Choices

A short rationale for the more interesting design decisions:

**Why LIBERO?** It is the de-facto standard benchmark for VLA models (OpenVLA, RT-2, π₀ all report on it), so results are comparable to published work. LIBERO-Long specifically — where OpenVLA shows its largest gap — is the suite most likely to expose the value of a re-evaluation Actor.

**Why SmolVLM-500M instead of Qwen2.5-VL-3B?** Adding the vision encoder pushed VRAM beyond 8GB on the available hardware. SmolVLM is purpose-built for low-VRAM scenarios while retaining VLM capabilities, making the dual-system feasible on consumer GPUs without sacrificing the project's core idea.

**Why REQ/REP (not PUB/SUB)?** Each training step requires *one specific action* from the Planner, with the Actor blocking until it arrives. PUB/SUB's broadcast semantics don't fit — Actor needs a guaranteed reply, not a stream.

**Why SFT before GRPO?** Starting GRPO without format pretraining produces constant penalties for malformed outputs (no critical text, wrong number of action tokens) and prevents convergence. SFT teaches the output format first; GRPO then refines the policy. This mirrors the approach used by DeepSeek.

---

## References

**Models**

- OpenVLA — [openvla/openvla](https://github.com/openvla/openvla)
- SmolVLM — [huggingface/smollm](https://github.com/huggingface/smollm)
- Qwen2.5-VL — [HuggingFace transformers/qwen2_5_vl](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen2_5_vl)

**Benchmark**

- LIBERO — [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)

**Frameworks & Techniques**

- RLinf-VLA — [RLinf/RLinf](https://github.com/RLinf/RLinf)
- LLaVA (projection layer reference) — [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/multimodal_projector/builder.py)
- GRPO (Group Relative Policy Optimization) — DeepSeek-Math, DeepSeek-R1

---

## License

This repository is maintained as a personal study copy. The original codebase is the work of the project team; I do not claim authorship of the implementation. Code in this repository is for educational and research-study purposes.

---

<div align="center">

*Last updated — study copy maintained by MIN THU KYAW*

</div>
