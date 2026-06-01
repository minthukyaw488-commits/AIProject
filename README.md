<h1> Dual-System VLA Model with Critical Re-evaluation Text for Planner Performance Improvement</h1>

**Critical text And Dual System Vision Language Action Model (CADS-VLA)**

<a href="https://pytorch.org/get-started/locally/">
  <img src="https://img.shields.io/badge/PYTORCH-Qwen%202.6.0%20cu124-brightgreen?style=flat-square&label=PYTORCH&labelColor=%23eeeeee&color=%23d63f3a" height="40"/>
</a>
&nbsp;
<a href="https://pytorch.org/get-started/locally/">
  <img src="https://img.shields.io/badge/PYTORCH-openvla%20%2Bcpu-brightgreen?style=flat-square&label=PYTORCH&labelColor=%23eeeeee&color=%23d63f3a" height="40"/>
</a>
&nbsp;
<a href="https://www.python.org/">
  <img src="https://img.shields.io/badge/Python-3.10-brightgreen?style=flat-square&label=Python&labelColor=%23eeeeee&color=%2355adf4" height="40"/>
</a>

---

## Model Configuration

| Role | Model | Setup |
| --- | --- | --- |
| **Planner** | OpenVLA 7B (fine-tuned) | 4bit · Frozen · CPU inference |
| **Actor 1** | Qwen2.5-VL 3B | 4bit · LoRA · GRPO |
| **Actor 2** ✅ (in use) | SmolVLM 500M | 4bit · LoRA · GRPO |
| **Simulator** | LIBERO (libero_10) | — |

---

## Backbone Model References

- **OpenVLA** : [openvla](https://github.com/openvla/openvla.git)
- **Qwen2.5-VL** : [Qwen2.5VL](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen2_5_vl)
- **SmolVLM** : [SmolVLM](https://github.com/huggingface/smollm/tree/main)
- **LIBERO** : [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO.git)

---

## 🖼️ Model Figures

![Figure1](https://github.com/user-attachments/assets/edbb073d-d40e-452a-873f-a4de54bc41b6)

### 1. Model Structure

A high-level diagram of the overall model architecture. The Planner uses OpenVLA in frozen mode. The Actor uses SmolVLM 500M as its backbone — a very lightweight choice among large vision-language models. The Planner produces action token IDs and passes them to the Actor. The Actor receives these through the OpenVLA embedding table, passes them through a projection layer to match dimensions, and concatenates them with the image and text embeddings produced by SmolVLM's processor. The resulting embeddings go through the LLM and are detokenized back into text, with the robot then executing the action. Based on whether the robot's action succeeds or fails, the model receives a reward and is trained through the GRPO learning curriculum.

---

![Figure2](https://github.com/user-attachments/assets/15b2c0a8-c4bb-4968-bf58-09bde7475a99)

### 2. Model Process

A visualization of the full model process. The Planner processes image and text through OpenVLA's own LLM and ViT, then infers action tokens and outputs action token IDs. The Actor processes image and text through SmolVLM's own LLM and ViT. The action tokens received from the Planner are converted into embeddings through the OpenVLA action embedding table (frozen), then transformed into SmolVLM's embedding dimension via the projection layer, and concatenated with the processor output. At this point, SmolVLM's tokenizer has had OpenVLA's 256 action tokens added to it. After attention computation in the transformer, text is recovered through the detokenizer, while action tokens are sent to the robot as action vectors to execute.

---

![Figure3](https://github.com/user-attachments/assets/e5adbf70-5ef6-4e22-94d6-8828ed1dc6be)

### 3. Flow Chart

The overall data flow through the model. The Planner and Actor structures are omitted here since they were covered above. All image and text data used by both the Actor and the Planner are collected from the LIBERO environment, and the Actor sends them to the Planner server via zeroMQ. In the Actor action tokenizer initialization step shown above, 256 action tokens are pulled from OpenVLA via the sequence: add → action embedding table → resize → projection layer. The GRPO training process shown below reflects the RLinf framework. The `collect_rollout` step gathers data from the LIBERO environment up to the group size. During this collection, critical text and action tokens are inferred to perform actions, with the corresponding reward and loss calculated, and weight updates performed in the `compute` step. These updates train the LoRA weights.

---

![Figure4](https://github.com/user-attachments/assets/1c0fbf23-fe96-475a-8827-3fd545224f3e)

### 4. Actor Tokenizer Process

A detailed view of the implemented action tokenizer's structure. Each input is converted into embedding form through its respective vision encoder, LLM, or projection layer. Through the input merge and action injection hook, these embeddings are projected into the same space and concatenated. Image and text inputs go through SmolVLM's processor at this stage. The merged representation then passes through the transformer to produce the output.

---

![Figure5](https://github.com/user-attachments/assets/f106f294-3e26-4129-ab98-1bde2140310e)

### 5. Supervised Fine-Tuning

If we go straight into model training, the model does not yet know the expected text format or how to produce action tokens. This would lead to constant penalties during GRPO training and likely prevent convergence. To address this, SFT is used first to teach the model basic capabilities, so that it can then learn critical text generation and the corresponding action token corrections. SFT consists of 4 stages, trained sequentially.

---

## Repository File Structure

### 📁 openvla_planner

- **openvla_inference_code**
  > Inference-only code for OpenVLA.
  > Integrated with zeroMQ to open a server.
  > Uses CPU for inference, so CUDA is not used.
  > Loaded via the transformers library.
  > Model: `openvla-7b-finetuned-libero-10`

- **action_tokenizer**
  > The original action tokenizer from OpenVLA.
  > Kept for convenience during code development and reference.

---

### 📁 qwen_actor

- **actor_action_tokenizer**
  > Adds OpenVLA's 256 action tokens to the LLM tokenizer.
  > Pulls the Planner's embedding table and uses the projection layer to match Qwen's dimensions.
  > Includes concatenation with the Qwen processor.
  > Refer to the `setup` and `forward` functions.

- **projection_layer**
  > A layer to align the embedding spaces of OpenVLA and Qwen2.5-VL, which differ.
  > Directly inserting OpenVLA's 4096-dim action embeddings causes a dimension mismatch.
  > Implemented with reference to LLaVA's projection layer.
  >
  > **LLaVA** : [LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/multimodal_projector/builder.py)

- **actor_model**
  > Executable file for the Actor: applies 4bit quantization + LoRA to Qwen, and integrates zeroMQ.

---

### 📁 SmolVLM_actor

- **smol_action_tokenizer**
  > Adds OpenVLA's action tokens to SmolVLM's LLM tokenizer.
  > Initializes the tokenizer when the model is run.

- **smol_actor_model**
  > Applies 4bit quantization + LoRA + zeroMQ to SmolVLM.

- **smol_projection_layer**
  > Aligns the embedding spaces of OpenVLA and SmolVLM.
  > Implemented with reference to LLaVA's projection layer.

---

### 📁 train_file

- **train**
  > The main training script. Runs GRPO and LIBERO.

- **smol_train**
  > Training script for SmolVLM.
  > GRPO is implemented via the RLinf framework.
  >
  > **RLinf** : [RLinf](https://github.com/RLinf/RLinf)
  >
  > Training proceeds through the `collect_rollout` and `compute_grpo_loss` functions.
  > Run this file to execute the model.

- **smol_sft**
  > Pre-trains the model via SFT before GRPO reinforcement learning begins.
  > This approach follows the method used by DeepSeek.
  > It has been shown to improve final performance of RL-trained models.
  > Here, SFT teaches the model two abilities: generating critical text and generating the 7 action tokens.

---

### 📁 SFT

- **SFT**
  > Pre-training step before GRPO, used to teach the Actor the expected text format.

---

### 📁 assets

- **make_embeddings.py**
  > Downloads the OpenVLA action embedding parameters.
  > Must be run once before running the model.

---

### 📁 checkpoints

- **sft**
  > Checkpoint from a completed SFT run.
  > Used to retrain from this point if the training result is unsatisfactory.
  > ⚠️ This checkpoint has a bug — the vision encoder was not used. **Not used.**

- **sft2** ✅ (in use)
  > SFT checkpoint that teaches the basic ability to infer critical text and action tokens.
  > This is the vision-encoder version — **this is the file in use**.

---

### 📁 logs

- Training logs are stored here, kept as a record of how training progressed.

---

## ⏬ Installation

**Qwen environment**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install requirements_qwen.txt --no-deps
git clone https://github.com/Lifelong-Robot-Learning/LIBERO
```

**OpenVLA environment (CPU)**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install requirements_openvla.txt
```

> SmolVLM can also be run inside the pre-configured Qwen environment.

---

## Getting Started

**1. Generate the embedding file (required, run once)**

Before running the model, the OpenVLA embeddings file must be generated. Enter the OpenVLA environment:

```bash
git clone https://github.com/imgonnago/AIproject
conda activate openvla
python make_embeddings.py    # generates the embedding file
python openvla_planner/openvla_inference_code.py
```

**2. Enter the Actor environment and run training**

SmolVLM also uses the Qwen environment.

```bash
conda activate qwen

# Start OpenVLA first so the zeroMQ server is open, then start the Actor side.
python train/train.py
```

---

## Other Settings

**Logging VRAM usage and training output**

```bash
# VRAM log
nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 1 >> vram_log.csv &

# Training log
python train/train.py >> train_log.txt 2>&1
# or
python train/smol_train.py >> train_log.txt 2>&1
```

**Real-time GPU monitoring** (change the number to control refresh interval in seconds):

```bash
watch -n 0.5 nvidia-smi
```

**Check which processes are using the GPU:**

```bash
nvidia-smi pmon -c 1
```

**When the VS Code SSH connection keeps dropping (use tmux on Ubuntu)**

Move into the project folder before running each command.

```bash
# Planner session
tmux new -s planner
conda activate openvla
python openvla_planner/openvla_inference_code.py

# Training session
tmux new -s train
conda activate qwen
python train/smol_train.py
```

**Helpful VRAM environment variable** (set before running the model):

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**Combined VRAM logging + training run:**

```bash
nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 1 >> vram_log.csv &
python train/smol_train.py 2>&1 | tee train_log.txt
```
