# ⚡ NVIDIA Nemotron 3.5 Lightning 30B A3B NVFP4

<p align="center">

**A 30B-parameter Mixture-of-Experts reasoning model with only 3B active parameters**

**Hybrid Mamba-2 + MoE + Attention Architecture • 1M Token Context • NVFP4**

<br>

<img src="https://img.shields.io/badge/NVIDIA-Nemotron-76B900?style=for-the-badge&logo=nvidia&logoColor=white">
<img src="https://img.shields.io/badge/30B-Parameters-76B900?style=for-the-badge">
<img src="https://img.shields.io/badge/3B-Active-00AEEF?style=for-the-badge">
<img src="https://img.shields.io/badge/NVFP4-Quantized-8A2BE2?style=for-the-badge">
<img src="https://img.shields.io/badge/Context-1M%20Tokens-FF6B35?style=for-the-badge">

</p>

---

## 🧠 Overview

**NVIDIA Nemotron 3.5 Lightning 30B A3B NVFP4** is a high-efficiency large language model designed for **reasoning, coding, tool use, long-context workloads, and autonomous AI agents**.

The model combines:

> **Mamba-2 + Mixture-of-Experts + Transformer Attention + Multi-Token Prediction**

into a hybrid architecture optimized for long-running agentic workloads.

Despite having **30 billion total parameters**, only approximately **3 billion parameters are active per inference step**, enabling a substantially more efficient compute profile than a dense 30B model.

The model supports context windows of up to **1 million tokens** and is particularly targeted at:

* 🤖 Autonomous AI agents
* 🧩 Multi-agent systems
* 🔧 Tool-calling workflows
* 💻 Software engineering agents
* 📚 Long-context reasoning
* 🔎 RAG systems
* 🧠 Sub-agent workloads
* 💬 Conversational AI
* ⚙️ Enterprise AI applications

The uploaded model specification identifies the intended use as general-purpose reasoning/chat with strong emphasis on agent systems, chatbots, RAG, and instruction-following applications.

---

# 🚀 Why Lightning?

Traditional large language models often activate the majority of their parameters for every token.

Nemotron 3.5 Lightning takes a different approach.

```text
                    ┌─────────────────────────┐
                    │   User / Agent Request   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   1M Token Context      │
                    └────────────┬────────────┘
                                 │
                                 ▼
              ┌─────────────────────────────────────┐
              │       Hybrid Lightning Core         │
              │                                     │
              │   Mamba-2 ── MoE ── Attention       │
              │       │       │        │             │
              │       └───────┼────────┘             │
              │               ▼                      │
              │       3B Active Parameters           │
              └────────────────┬────────────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │ Multi-Token Prediction  │
                  └────────────┬────────────┘
                               │
                               ▼
                    ┌─────────────────────────┐
                    │ Agent / Tool / Response │
                    └─────────────────────────┘
```

The model contains **30B total parameters with 3B active parameters**, and its architecture is described as a hybrid Mamba + Transformer / MoE design.

---

# 📊 Model at a Glance

| Property                    | Specification                               |
| --------------------------- | ------------------------------------------- |
| **Model**                   | NVIDIA Nemotron 3.5 Lightning 30B A3B NVFP4 |
| **Developer**               | NVIDIA Corporation                          |
| **Parameters**              | 30B total                                   |
| **Active Parameters**       | 3B                                          |
| **Architecture**            | Hybrid Mamba-2 + MoE + Attention            |
| **Network**                 | Nemotron-3-Lightning + MTP                  |
| **Context Length**          | Up to 1M tokens                             |
| **Quantization**            | NVFP4                                       |
| **Runtime**                 | PyTorch                                     |
| **Acceleration**            | Dynamo + vLLM                               |
| **Primary Hardware**        | NVIDIA GPU                                  |
| **Recommended Temperature** | 1.0                                         |
| **Recommended Top-P**       | 0.95                                        |
| **Primary Focus**           | Agentic AI / Reasoning / Coding             |
| **Release**                 | August 11, 2026                             |
| **License**                 | OpenMDW License Agreement 1.1               |

## The official specification reports a 30B/3B parameter configuration, 1M context, PyTorch runtime, and NVIDIA Blackwell/Hopper/Ampere compatibility under the listed precision modes.

# 🏗️ Architecture

Nemotron 3.5 Lightning uses a **hybrid Mixture-of-Experts architecture**.

```text
                 NVIDIA Nemotron 3.5 Lightning
                              │
                              ▼
                  ┌───────────────────────┐
                  │   Input Token Stream  │
                  └───────────┬───────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │      Mamba-2        │
                   │ Long-range sequence │
                   │      modeling       │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │        MoE          │
                   │ Sparse expert       │
                   │     routing         │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │     Attention       │
                   │ Context reasoning  │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ Multi-Token         │
                   │ Prediction (MTP)   │
                   └──────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Final Generation   │
                    └────────────────────┘
```

## The model uses **interleaved Mamba-2 and MoE layers with selected Attention layers**, together with Multi-Token Prediction to improve training signals and generation efficiency.

# ⚡ Multi-Token Prediction

A key component of the Lightning architecture is **Multi-Token Prediction (MTP)**.

Instead of learning only to predict the immediate next token, MTP heads learn to predict multiple future tokens.

```text
Traditional:

Token₁ → Token₂ → Token₃ → Token₄


Lightning MTP:

Token₁ ─────┬────→ Token₂
            ├────→ Token₃
            └────→ Token₄
```

This provides richer training signals and is also used to accelerate rollout generation during reinforcement learning.

The model documentation describes a dedicated continued-pretraining stage for aligning the MTP heads with the base model distribution.

---

# 🧪 Training Pipeline

The training process consists of multiple stages.

```text
┌───────────────────────────┐
│       01 Pre-Training     │
│                           │
│ Code • Math • Science     │
│ General Knowledge         │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│    02 MTP Training        │
│                           │
│ Multi-Token Prediction    │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│   03 Supervised FT        │
│                           │
│ Code • Math • Tools       │
│ Structured Outputs        │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│  04 Reinforcement Learning│
│                           │
│ GRPO + Multi-Environment  │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│       05 PTQ              │
│                           │
│ NVFP4 / W4A16             │
└─────────────┬─────────────┘
              │
              ▼
        🚀 Final Model
```

The documented training pipeline includes pre-training, MTP continued pre-training, supervised fine-tuning, GRPO-based reinforcement learning, and post-training quantization.

---

# 📚 Training Data

The model was pretrained on **more than 20 trillion tokens**.

The training corpus combines:

* Curated datasets
* Synthetic data
* Web content
* Code
* Mathematics
* Science
* General knowledge
* Dialogue
* Articles
* QA data
* Alignment-oriented data

The documented corpus spans multiple domains and programming languages, with automated, manually collected, and synthetic data sources.

Synthetic post-training data was generated through teacher-model distillation, trajectory generation, solution generation, translation, and automated verification pipelines.

---

# 📈 Benchmark Results

## Reasoning & General Intelligence

| Benchmark          |  BF16 |     NVFP4 |
| ------------------ | ----: | --------: |
| **MMLU Pro**       | 81.94 | **81.62** |
| **AA-Omniscience** | 17.50 |     16.63 |
| **GPQA Diamond**   | 75.44 | **75.57** |
| **HLE**            | 11.72 |     10.47 |
| **SciCode**        | 32.60 |     31.38 |

## Coding & Agentic Workloads

| Benchmark                  |  BF16 |     NVFP4 |
| -------------------------- | ----: | --------: |
| **SWE-bench Verified**     | 51.56 | **52.80** |
| **SWE-bench Multilingual** | 39.33 |     36.47 |
| **Terminal-Bench 2.1**     | 24.58 |     23.46 |
| **PinchBench**             | 85.37 |     83.43 |
| **BrowseComp**             | 36.97 |     36.81 |
| **τ³-bench Banking**       |  9.28 |  **9.48** |
| **GDPval-AA-V2**           |   832 |   **865** |

## Instruction Following

| Benchmark   |  BF16 |     NVFP4 |
| ----------- | ----: | --------: |
| **IFBench** | 71.88 | **72.88** |

## Long Context

| Benchmark  |  BF16 | NVFP4 |
| ---------- | ----: | ----: |
| **AA-LCR** | 52.00 | 49.19 |

The benchmark values above are reproduced from the uploaded model specification, which notes that these results apply to the official NVFP4 checkpoint.

---

# 🎯 Designed For Agentic AI

Nemotron 3.5 Lightning is particularly interesting for **AI agent infrastructure**.

### 🤖 Autonomous Agents

```text
User
 │
 ▼
Planner Agent
 │
 ├── Research Agent
 │
 ├── Coding Agent
 │
 ├── Browser Agent
 │
 ├── Data Agent
 │
 └── Verification Agent
          │
          ▼
      Nemotron
          │
          ▼
     Final Result
```

Potential workloads include:

* Autonomous software engineering
* Browser agents
* Research agents
* Coding copilots
* RAG agents
* Multi-agent orchestration
* Tool-calling systems
* Long-running workflows
* Enterprise assistants

---

# 🔍 Long-Context Intelligence

With support for up to **1 million tokens**, the model is designed for workloads where conventional context windows become restrictive.

Example:

```text
                    1M TOKEN CONTEXT
┌──────────────────────────────────────────────────────┐
│                                                      │
│ Repository                                            │
│ Documentation                                         │
│ API Specifications                                    │
│ Test Cases                                            │
│ Logs                                                  │
│ Architecture Docs                                     │
│ Tickets                                               │
│ Previous Agent Runs                                   │
│ Knowledge Base                                        │
│                                                       │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
                    Nemotron Lightning
                           │
                           ▼
                   Contextual Reasoning
                           │
                           ▼
                    Agentic Decision
```

This makes the architecture particularly relevant to **large repositories, enterprise documentation, multi-document RAG, and long-running agent sessions**.

---

# 🧮 NVFP4 Quantization

The released checkpoint uses NVIDIA's **NVFP4 post-training quantization** approach.

The documented recipe includes:

* Four Over Six NVFP4
* Static MSE calibration
* W4A16 for routed and shared experts
* FP8 dynamic scaling for selected Mamba projections
* FP8 dynamic scaling for KV cache
* 1,000 calibration samples
* 32K token calibration length

These details are from the model's documented PTQ recipe.

---

# 🖥️ Hardware Compatibility

The model is optimized for NVIDIA GPU-accelerated systems.

### Supported Architectures

| GPU Architecture     | Precision               |
| -------------------- | ----------------------- |
| **NVIDIA Blackwell** | NVFP4 / supported modes |
| **NVIDIA Hopper**    | NVFP4 / W4A16           |
| **NVIDIA Ampere**    | W4A16                   |

The model specification explicitly lists Blackwell, Hopper, and Ampere compatibility under the corresponding precision configurations.

### Tested Inference Stack

```text
Application
     │
     ▼
┌───────────────┐
│    Agent      │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│    vLLM       │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   Dynamo      │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ NVIDIA GPU    │
│ H100 / etc.   │
└───────────────┘
```

The documented inference acceleration stack is **Dynamo + vLLM**, with **NVIDIA Hopper H100** listed as test hardware.

---

# 🌍 Supported Languages

The model supports:

* 🇺🇸 English
* 🇪🇸 Spanish
* 🇫🇷 French
* 🇩🇪 German
* 🇮🇹 Italian
* 🇯🇵 Japanese

It is also trained across a substantially broader multilingual corpus and supports programming languages.

---

# ⚙️ Recommended Sampling

For general inference:

```text
Temperature: 1.0
Top-P:       0.95
```

These are the recommended sampling parameters in the supplied model specification.

---

# 🧩 Example Use Cases

## 1. Software Engineering Agent

```text
GitHub Repository
       │
       ▼
Code Understanding
       │
       ▼
Issue Analysis
       │
       ▼
Planning
       │
       ▼
Code Generation
       │
       ▼
Testing
       │
       ▼
Validation
       │
       ▼
Pull Request
```

## 2. RAG Agent

```text
Documents
   │
   ▼
Embedding / Retrieval
   │
   ▼
Long Context
   │
   ▼
Nemotron
   │
   ▼
Reasoning
   │
   ▼
Grounded Answer
```

## 3. Multi-Agent System

```text
                    ┌───────────────┐
                    │ Orchestrator  │
                    └───────┬───────┘
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
     Research Agent    Coding Agent     Browser Agent
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ▼
                    Nemotron Lightning
                            │
                            ▼
                       Final Output
```

---

# 🧪 Evaluation & Validation

Large language models should be evaluated beyond simple accuracy.

A production evaluation strategy should consider:

```text
                  MODEL EVALUATION
                         │
       ┌─────────────────┼─────────────────┐
       ▼                 ▼                 ▼
   Reasoning          Coding            Agentic
       │                 │                 │
       ▼                 ▼                 ▼
   Knowledge         SWE Tasks         Tool Use
       │                 │                 │
       └─────────────────┼─────────────────┘
                         ▼
                   Long Context
                         │
                         ▼
                  Safety / Quality
                         │
                         ▼
                Production Validation
```

The supplied specification also emphasizes use-case-specific testing and iterative validation at both unit and system levels before deployment.

---

# 📸 Model Playground / UI

## NVIDIA Nemotron Playground

The following screenshots demonstrate the model interface and associated environment.

<p align="center">
<img width="1361" src="https://github.com/user-attachments/assets/7a34a4b2-de3b-46da-9309-f73c16e9847b">
</p>

<p align="center">
<img width="1366" src="https://github.com/user-attachments/assets/352c342c-b699-4181-9847-06ead9f47fa3">
</p>

<p align="center">
<img width="1360" src="https://github.com/user-attachments/assets/70c3bbbc-9e7e-433e-8c8c-e9861425de28">
</p>

---

# 🔬 Research & Engineering Focus

This model is especially relevant for research into:

### Agentic AI

* Autonomous planning
* Tool use
* Multi-agent orchestration
* Long-running agent loops
* Agent memory

### Software Engineering AI

* Code generation
* Code review
* Repository understanding
* Automated debugging
* Test generation
* SWE agents

### Long-Context AI

* Repository-scale reasoning
* Enterprise knowledge bases
* Multi-document reasoning
* Large RAG systems
* Long-running conversations

### Efficient Inference

* Mixture-of-Experts routing
* NVFP4 quantization
* Multi-token prediction
* Sparse activation
* GPU-optimized serving

---

# 📌 Model Facts

```text
┌─────────────────────────────────────────┐
│         NEMOTRON LIGHTNING 30B          │
├─────────────────────────────────────────┤
│ Total Parameters       : 30B            │
│ Active Parameters      : 3B             │
│ Context                : 1M Tokens      │
│ Architecture           : Hybrid MoE     │
│ Sequence Modeling      : Mamba-2        │
│ Attention              : Transformer    │
│ Prediction             : MTP            │
│ Quantization           : NVFP4          │
│ Runtime                : PyTorch        │
│ Serving                : vLLM + Dynamo  │
│ Primary Focus          : Agentic AI     │
└─────────────────────────────────────────┘
```

---

# 🗓️ Release Information

| Field                    | Value              |
| ------------------------ | ------------------ |
| **Model Release**        | August 11, 2026    |
| **Model Version**        | 1.0-preview        |
| **Pre-training Cutoff**  | September 2025     |
| **Post-training Cutoff** | May 2026           |
| **Total Training Data**  | >20T tokens        |
| **Developer**            | NVIDIA Corporation |

## The supplied documentation lists the model version as **1.0-preview (08/11/2026)** and gives the respective pre-training and post-training data cutoffs.

# 📜 License

This model is governed by the:

**OpenMDW License Agreement, Version 1.1**

The supplied model documentation states that use of the model is governed by the OpenMDW License Agreement 1.1.

> Always review the applicable NVIDIA license and terms before commercial or production deployment.

---

# 👨‍💻 Project / Documentation

### Built, curated and documented by

# **Harsha Vardhan Upadrasta**

AI Automation Engineer • AI/ML Researcher • Agentic AI Enthusiast

---

# ⭐ Key Takeaways

**Nemotron 3.5 Lightning 30B A3B NVFP4 brings together:**

```text
             30B Total Parameters
                       │
                       ▼
                3B Active Params
                       │
                       ▼
          ┌────────────────────────┐
          │ Hybrid Architecture    │
          │                        │
          │ Mamba-2 + MoE +       │
          │ Attention + MTP        │
          └───────────┬────────────┘
                      │
                      ▼
                1M Context
                      │
                      ▼
              Agentic Intelligence
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       Coding       RAG       Multi-Agent
          │           │           │
          └───────────┼───────────┘
                      ▼
              Efficient Inference
                      │
                      ▼
                    NVFP4
```

> **A high-capacity model architecture designed to make large-scale reasoning and agentic workloads more efficient through sparse activation, hybrid sequence modeling, long-context capability, and optimized inference.**

---

<p align="center">

### ⚡ NVIDIA Nemotron 3.5 Lightning

### **30B Total • 3B Active • 1M Context • NVFP4**

<br>

**Developed / Curated by Harsha Vardhan Upadrasta**

</p>


<img width="1361" height="616" alt="image" src="https://github.com/user-attachments/assets/7a34a4b2-de3b-46da-9309-f73c16e9847b" />

<img width="1366" height="639" alt="image" src="https://github.com/user-attachments/assets/352c342c-b699-4181-9847-06ead9f47fa3" />

<img width="1360" height="694" alt="image" src="https://github.com/user-attachments/assets/70c3bbbc-9e7e-433e-8c8c-e9861425de28" />
