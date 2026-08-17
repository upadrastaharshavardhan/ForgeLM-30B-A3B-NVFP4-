# ⚡ ForgeLM-30B-A3B-NVFP4

<p align="center">

### **30B Total Parameters • 3B Active Parameters • 1M Token Context • NVFP4**

**Hybrid Mamba-2 + Mixture-of-Experts + Attention Architecture**

**Built for Reasoning • Coding • RAG • Tool Calling • Autonomous Agents • Long-Context AI**

<br>

<img src="https://img.shields.io/badge/ForgeLM-30B--A3B-76B900?style=for-the-badge">
<img src="https://img.shields.io/badge/30B-Total%20Parameters-76B900?style=for-the-badge">
<img src="https://img.shields.io/badge/3B-Active%20Parameters-00AEEF?style=for-the-badge">
<img src="https://img.shields.io/badge/NVFP4-Quantized-8A2BE2?style=for-the-badge">
<img src="https://img.shields.io/badge/1M-Context-FF6B35?style=for-the-badge">
<img src="https://img.shields.io/badge/Agentic-AI-FF4B4B?style=for-the-badge">

</p>

---

## 📖 Table of Contents

* [Overview](#-overview)
* [Model Summary](#-model-summary)
* [Model Identity](#-model-identity)
* [Architecture](#-architecture)
* [Multi-Token Prediction](#-multi-token-prediction)
* [Training Pipeline](#-training-pipeline)
* [Training Data](#-training-data)
* [Benchmarks](#-benchmarks)
* [Evaluation Methodology](#-evaluation-methodology)
* [Use Cases](#-use-cases)
* [Long-Context Capability](#-long-context-capability)
* [Input Specification](#-input-specification)
* [Output Specification](#-output-specification)
* [Hardware Compatibility](#-hardware-compatibility)
* [Software Stack](#-software-stack)
* [Inference](#-inference)
* [Testing & Validation](#-testing--validation)
* [Dataset Considerations](#-dataset-considerations)
* [Model Version](#-model-version)
* [Release Information](#-release-information)
* [License](#-license)
* [Screenshots](#-screenshots)
* [Project Attribution](#-project-attribution)

---

# 🧠 Overview

**ForgeLM-30B-A3B-NVFP4** is an advanced large-language-model engineering and deployment project centered around a highly efficient **30-billion-parameter / 3-billion-active-parameter hybrid architecture**.

The architecture combines:

* 🧬 **Mamba-2**
* 🧠 **Mixture-of-Experts (MoE)**
* 🔭 **Selective Transformer Attention**
* ⚡ **Multi-Token Prediction (MTP)**
* 📦 **NVFP4 post-training quantization**
* 📚 **Up to 1M-token context**

The system is designed for demanding AI workloads including:

* Autonomous AI agents
* Long-running agent workflows
* Sub-agent deployments
* Coding agents
* Tool-calling systems
* RAG applications
* Chatbots
* Instruction following
* Long-context reasoning
* Multi-document aggregation
* Software-engineering workflows

The underlying technical specification describes a 30B total / 3B active configuration and identifies agentic workflows and long-running autonomous workloads as primary use cases.

---

# 📊 Model Summary

| Property                    | Specification                                         |
| --------------------------- | ----------------------------------------------------- |
| **Project Name**            | ForgeLM-30B-A3B-NVFP4                                 |
| **Total Parameters**        | 30B                                                   |
| **Active Parameters**       | 3B                                                    |
| **Architecture**            | Hybrid Mixture-of-Experts                             |
| **Sequence Architecture**   | Mamba-2 + MoE + Attention                             |
| **Network Architecture**    | Lightning-style architecture + Multi-Token Prediction |
| **Context Length**          | Up to 1M tokens                                       |
| **Input**                   | Text                                                  |
| **Output**                  | Text                                                  |
| **Quantization**            | NVFP4                                                 |
| **Runtime**                 | PyTorch                                               |
| **Inference Stack**         | Dynamo + vLLM                                         |
| **Test Hardware**           | NVIDIA Hopper H100                                    |
| **Operating System**        | Linux                                                 |
| **Recommended Temperature** | 1.0                                                   |
| **Recommended Top-P**       | 0.95                                                  |
| **Primary Domain**          | Agentic AI                                            |
| **Model Type**              | General-purpose reasoning and chat                    |
| **Release Date**            | August 11, 2026                                       |
| **Model Version**           | 1.0-preview                                           |
| **License**                 | OpenMDW License Agreement, Version 1.1                |

The source specification reports English and several additional supported languages, including Spanish, French, German, Italian, and Japanese.

---

# 🎯 Model Identity

## Project

**ForgeLM-30B-A3B-NVFP4**

## Model Category

```text
Large Language Model
        │
        ├── Reasoning
        ├── Coding
        ├── Agentic AI
        ├── Tool Calling
        ├── RAG
        ├── Long Context
        └── Instruction Following
```

## Design Philosophy

ForgeLM focuses on combining:

> **Large total model capacity + sparse activation + efficient sequence modeling + long-context reasoning + optimized inference**

The architecture contains **30B total parameters**, while approximately **3B parameters are active** during inference.

---

# 🏗️ Architecture

ForgeLM uses a hybrid architecture combining multiple complementary mechanisms.

```text
                         ┌─────────────────────┐
                         │     Input Text      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Context Engine    │
                         │    Up to 1M Tokens  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                  ┌─────────────────────────────────┐
                  │       Hybrid Model Core         │
                  │                                 │
                  │   ┌─────────┐   ┌───────────┐  │
                  │   │ Mamba-2 │   │    MoE    │  │
                  │   └────┬────┘   └─────┬─────┘  │
                  │        │              │        │
                  │        └──────┬───────┘        │
                  │               ▼                │
                  │        ┌────────────┐           │
                  │        │ Attention  │           │
                  │        └─────┬──────┘           │
                  └──────────────┼──────────────────┘
                                 │
                                 ▼
                       ┌────────────────────┐
                       │ Multi-Token        │
                       │ Prediction (MTP)   │
                       └─────────┬──────────┘
                                 │
                                 ▼
                       ┌────────────────────┐
                       │ 3B Active Compute  │
                       └─────────┬──────────┘
                                 │
                                 ▼
                       ┌────────────────────┐
                       │ Generated Output   │
                       └────────────────────┘
```

The documented architecture uses interleaved **Mamba-2 and MoE layers**, selected Attention layers, and Multi-Token Prediction.

---

# 🧬 Mamba-2 + MoE + Attention

## Mamba-2

Mamba-2 provides sequence-modeling capabilities within the hybrid architecture.

## Mixture-of-Experts

The MoE architecture provides sparse expert activation.

Instead of activating the entire 30B parameter set for every operation, the architecture operates with approximately **3B active parameters**.

```text
30B Total Parameters
        │
        ├──────── Expert 1
        ├──────── Expert 2
        ├──────── Expert 3
        ├──────── Expert 4
        ├──────── Expert 5
        ├──────── ...
        │
        ▼
   Router / Gating
        │
        ▼
  Selected Experts
        │
        ▼
 ~3B Active Parameters
```

## Attention

Selective Attention layers complement the Mamba-2 and MoE components for context-dependent reasoning.

The model documentation explicitly characterizes the network as a **Mixture-of-Experts hybrid using Mamba + Transformer components**.

---

# ⚡ Multi-Token Prediction

**Multi-Token Prediction (MTP)** is an important component of the model architecture.

Traditional next-token prediction can be represented as:

```text
Token 1
   │
   ▼
Token 2
   │
   ▼
Token 3
   │
   ▼
Token 4
```

MTP expands the prediction objective:

```text
                  Token 1
                 /   |   \
                ▼    ▼    ▼
            Token 2 Token 3 Token 4
```

The MTP layers learn to predict multiple future tokens, providing richer training signals.

A dedicated continued-pretraining stage was used to train and align the MTP layers with the base model distribution.

MTP is also leveraged during reinforcement-learning rollouts to accelerate generation.

---

# 🧪 Training Pipeline

The model development process consists of five major stages.

```text
                 ┌───────────────────────┐
                 │  Stage 1              │
                 │  PRE-TRAINING         │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │  Stage 2              │
                 │  MTP PRE-TRAINING    │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │  Stage 3              │
                 │  SUPERVISED FT        │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │  Stage 4              │
                 │  REINFORCEMENT        │
                 │  LEARNING             │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │  Stage 5              │
                 │  POST-TRAINING        │
                 │  QUANTIZATION         │
                 └───────────┬───────────┘
                             │
                             ▼
                    ForgeLM NVFP4
```

---

## Stage 1 — Pre-Training

The model was pretrained using an **NVFP4 recipe** on:

* Code
* Mathematics
* Science
* General knowledge
* Curated data
* Synthetic data

The pre-training software identified in the source is **Megatron-LM**.

The model was pretrained on **more than 20 trillion tokens**.

---

## Stage 2 — Continued Pre-Training for MTP

The second stage focuses on Multi-Token Prediction.

MTP heads are trained to predict multiple future tokens and are subsequently aligned with the base model's distribution.

---

## Stage 3 — Supervised Fine-Tuning

The supervised fine-tuning stage incorporates data covering:

* Synthetic code
* Mathematics
* Science
* Tool calling
* Instruction following
* Structured outputs
* General knowledge
* Long-range retrieval
* Multi-document aggregation

---

## Stage 4 — Reinforcement Learning

The reinforcement-learning stage uses:

**GRPO — Group Relative Policy Optimization**

Training environments include:

* Mathematics
* Coding
* Science
* Instruction following
* Multi-step tool use
* Multi-turn conversations
* Structured outputs

The documented RL architecture uses asynchronous training that separates training from inference and uses MTP to accelerate rollout generation.

Software:

```text
NeMo RL
NeMo Gym
```

---

## Stage 5 — Post-Training Quantization

The final checkpoint uses **Post-Training Quantization (PTQ)**.

The documented recipe includes:

```text
Four Over Six NVFP4
        │
        ▼
Static MSE Calibration
        │
        ▼
W4A16
        │
        ├── Routed Experts
        └── Shared Experts

FP8 Dynamic Scaling
        │
        ├── Mamba in_proj
        ├── Mamba out_proj
        └── KV Cache
```

Calibration details include:

* 1,000 calibration samples
* 32K token calibration length
* Subset of the Nemotron Ultra validation set

---

# 📚 Training Data

## Scale

**More than 20 trillion tokens**

The documented training-data collection period spans **2013 to December 2025**.

## Data Modality

```text
Text
```

## Collection Methods

The training corpus combines:

* Automated collection
* Manual collection
* Synthetic generation

## Labeling Methods

The source identifies:

* Automated labeling
* Manual labeling
* Synthetic labeling

---

# 🌍 Languages & Programming Languages

The documented pre-training corpus includes:

* English
* 19 other spoken languages
* 43 programming languages

The post-training corpus primarily includes:

* English
* French
* German
* Italian
* Japanese
* Spanish
* Chinese

---

# 📖 Training Corpus Domains

The documented sources cover:

* Web pages
* Dialogue
* Articles
* Written materials
* Legal information
* Mathematics
* Science
* Finance
* Question answering
* Alignment-oriented data

The training corpus is described as a combination of high-quality curated and synthetically generated data.

---

# 🧬 Synthetic Data Pipeline

Synthetic post-training data is generated through:

```text
Teacher Models
      │
      ▼
Trajectory Generation
      │
      ▼
Solution Generation
      │
      ▼
Translation
      │
      ▼
Best-of-N Selection
      │
      ▼
Automated Verification
      │
      ▼
Quality Filtering
      │
      ▼
Post-Training Dataset
```

Verification mechanisms include:

* Compilers
* Numerical checks
* Language identification
* Structural validation

The source describes synthetic-data generation and automated verification across mathematics, code, science, long-context, and multilingual workflows.

---

# 🧹 Data Quality & Filtering

The documented filtering pipeline includes:

### Structural Validation

Malformed examples are removed using structural checks.

For example:

```text
Missing tool definitions
        │
        ▼
Invalid Sample
        │
        ▼
Removed
```

### Repetition Filtering

Reasoning traces exhibiting pathological repetition are filtered.

The pipeline checks repeated n-grams across sliding windows and complete trajectories.

### Additional Filtering

The source also documents targeted filtering of generated trajectories exhibiting unwanted political or nationalistic narratives.

---

# 📊 Benchmarks

## Reasoning Benchmark Evaluations

| Benchmark      |  BF16 |     NVFP4 |
| -------------- | ----: | --------: |
| MMLU Pro       | 81.94 |     81.62 |
| AA-Omniscience | 17.50 |     16.63 |
| GPQA Diamond   | 75.44 | **75.57** |
| HLE            | 11.72 |     10.47 |
| SciCode        | 32.60 |     31.38 |

---

## Coding & Agentic Evaluation

| Benchmark              |  BF16 |     NVFP4 |
| ---------------------- | ----: | --------: |
| SWE-bench Verified     | 51.56 | **52.80** |
| SWE-bench Multilingual | 39.33 |     36.47 |
| Terminal-Bench 2.1     | 24.58 |     23.46 |
| PinchBench             | 85.37 |     83.43 |
| BrowseComp             | 36.97 |     36.81 |
| τ³-bench (Banking)     |  9.28 |  **9.48** |
| GDPval-AA-V2           |   832 |   **865** |

---

## Instruction Following

| Benchmark       |  BF16 |     NVFP4 |
| --------------- | ----: | --------: |
| IFBench (loose) | 71.88 | **72.88** |

---

## Long Context

| Benchmark |  BF16 | NVFP4 |
| --------- | ----: | ----: |
| AA-LCR    | 52.00 | 49.19 |

The supplied benchmark table reports these BF16 and official NVFP4 checkpoint results.

---

# 🔬 Evaluation Methodology

The evaluation suite covers:

```text
Knowledge
   │
Reasoning
   │
Instruction Following
   │
Coding
   │
Agentic Tasks
   │
Tool Use
   │
Long Context
```

The source states that evaluation recipes, installation instructions, commands, benchmark-specific containers, prompts, inference parameters, parser configurations, and scoring settings were published through the associated evaluation tooling.

Most evaluations use **NeMo Gym-native harnesses**, while selected evaluations such as SWE-Bench and Terminal-Bench use **NeMo Evaluator** natively.

---

# 🤖 Use Cases

## Autonomous AI Agents

```text
                 User Goal
                    │
                    ▼
             Agent Planner
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
   Research       Coding       Browser
     Agent         Agent        Agent
       │            │            │
       └────────────┼────────────┘
                    ▼
                ForgeLM
                    │
                    ▼
              Tool Calling
                    │
                    ▼
               Validation
                    │
                    ▼
               Final Result
```

---

## 💻 Coding Agents

Potential workflows include:

* Code generation
* Repository understanding
* Code reasoning
* Debugging
* Tool execution
* Software engineering tasks
* Automated development workflows

---

## 🔎 RAG Systems

The long context capability makes the model suitable for:

* Document retrieval
* Multi-document reasoning
* Enterprise knowledge systems
* Long-form context aggregation
* Retrieval-augmented generation

---

## 🧩 Multi-Agent Systems

ForgeLM can serve as a model component inside:

```text
Orchestrator
    │
    ├── Research Agent
    ├── Coding Agent
    ├── Browser Agent
    ├── Data Agent
    ├── Testing Agent
    └── Verification Agent
             │
             ▼
          ForgeLM
```

---

# 📚 Long-Context Capability

The architecture supports **up to 1 million tokens** of context.

A long-context workload can combine:

```text
Repository
Documentation
API Specifications
Test Cases
Logs
Tickets
Architecture Documents
Previous Agent Runs
Knowledge Base
        │
        ▼
   1M Token Context
        │
        ▼
      ForgeLM
        │
        ▼
Contextual Reasoning
```

This enables workflows requiring large amounts of information to remain available within a single context.

---

# 📥 Input Specification

| Property                | Value                                                                           |
| ----------------------- | ------------------------------------------------------------------------------- |
| **Input Type**          | Text                                                                            |
| **Input Format**        | String                                                                          |
| **Input Structure**     | One-dimensional sequences                                                       |
| **Maximum Context**     | Up to 1M tokens                                                                 |
| **Supported Languages** | English, Spanish, French, German, Italian, Japanese + broader training coverage |

---

# 📤 Output Specification

| Property             | Value                     |
| -------------------- | ------------------------- |
| **Output Type**      | Text                      |
| **Output Format**    | String                    |
| **Output Structure** | One-dimensional sequences |
| **Maximum Context**  | Up to 1M tokens           |

---

# 🎛️ Recommended Sampling

```text
Temperature = 1.0
Top-P       = 0.95
```

These values are listed as the recommended sampling configuration in the supplied specification.

---

# 🖥️ Hardware Compatibility

The model is designed and optimized for NVIDIA GPU-accelerated systems.

## Supported GPU Architectures

| Architecture         | Supported Configuration |
| -------------------- | ----------------------- |
| **NVIDIA Blackwell** | NVFP4                   |
| **NVIDIA Hopper**    | NVFP4 / W4A16           |
| **NVIDIA Ampere**    | W4A16                   |

---

# 🧰 Software Stack

```text
┌─────────────────────────────┐
│       Application / Agent   │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│            vLLM             │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│           Dynamo            │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│      PyTorch / CUDA         │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│       NVIDIA GPU            │
└─────────────────────────────┘
```

## The documented runtime is **PyTorch**, while the inference acceleration engine is **Dynamo + vLLM**.

# ⚡ Inference

## Acceleration Engine

```text
Dynamo + vLLM
```

## Test Hardware

```text
NVIDIA Hopper
H100
```

---

# 🧪 Testing & Validation

Large language model deployment requires more than benchmark scores.

The supplied specification recommends iterative testing and validation at:

* Unit level
* System level
* Use-case-specific level

The validation approach follows a **V-model methodology** to help address:

* Technical requirements
* Functional requirements
* Safety
* Effectiveness
* Compliance
* Ethical considerations

---

# 🧪 Testing Datasets

The documented testing datasets use:

### Collection

```text
Automated
Manual
Synthetic
```

### Labeling

```text
Automated
Manual
Synthetic
```

### Evaluation Focus

The testing corpus includes standard benchmarks and modern agentic-AI test suites covering:

* Tool calling
* Instruction following
* Agentic behavior
* Modern AI capabilities

---

# ⚖️ Dataset Considerations

The supplied documentation notes that the datasets do not exhaustively represent all demographic groups.

It specifically discusses representation limitations across age, gender, and ethnicity-related mentions and recommends evaluation approaches such as:

* Bias audits
* Demographically balanced fine-tuning datasets
* Counterfactual data augmentation
* Additional evaluation techniques

The documentation describes a 3,000-sample subset per dataset used in the referenced evaluation context.

---

# 📦 Dataset Availability

The project documentation states that final pre-training and post-training data are released alongside the model.

The source describes:

* An ungated sample set
* Gated code data
* Gated mathematics data
* Gated multilingual data
* Approval requirements for certain datasets
* Permissive licensing for model-training purposes

---

# 🗂️ Model Version

```text
ForgeLM-30B-A3B-NVFP4
Version: 1.0-preview
Release: 08/11/2026
```

The supplied source lists the corresponding model version as **1.0-preview** with an August 11, 2026 release date.

---

# 🗓️ Release Information

| Field                          | Value                |
| ------------------------------ | -------------------- |
| **Release Date**               | August 11, 2026      |
| **Model Version**              | 1.0-preview          |
| **Pre-Training Data Cutoff**   | September 2025       |
| **Post-Training Data Cutoff**  | May 2026             |
| **Training Data**              | >20T tokens          |
| **Training Collection Period** | 2013 – December 2025 |
| **Deployment Geography**       | Global               |

---

# 📜 License

## OpenMDW License Agreement

**Version 1.1**

The supplied specification identifies the governing model license as the **OpenMDW License Agreement, version 1.1**.

> Review the applicable license terms before using the model in commercial or production environments.

---

# 📸 Screenshots

## Playground / Interface

<p align="center">
<img width="1361" height="616" alt="ForgeLM Interface" src="https://github.com/user-attachments/assets/7a34a4b2-de3b-46da-9309-f73c16e9847b">
</p>

<p align="center">
<img width="1366" height="639" alt="ForgeLM Playground" src="https://github.com/user-attachments/assets/352c342c-b699-4181-9847-06ead9f47fa3">
</p>

<p align="center">
<img width="1360" height="694" alt="ForgeLM Model Interface" src="https://github.com/user-attachments/assets/70c3bbbc-9e7e-433e-8c8c-e9861425de28">
</p>

The three interface images above are retained from the original uploaded README.

---

# 🧭 End-to-End Architecture

```text
                              ┌───────────────┐
                              │     USER      │
                              └───────┬───────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │   APPLICATION / AGENT  │
                         └────────────┬───────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │   1M TOKEN CONTEXT     │
                         └────────────┬───────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────┐
                    │          FORGELM CORE             │
                    │                                  │
                    │   ┌────────┐ ┌──────┐ ┌───────┐ │
                    │   │Mamba-2 │ │  MoE │ │Attention│
                    │   └────┬───┘ └──┬───┘ └───┬───┘ │
                    │        └────────┼──────────┘     │
                    │                 ▼                │
                    │          3B ACTIVE PARAMS        │
                    │                 │                │
                    │                 ▼                │
                    │              MTP                  │
                    └────────────────┬─────────────────┘
                                     │
                                     ▼
                              ┌──────────────┐
                              │    NVFP4     │
                              └──────┬───────┘
                                     │
                                     ▼
                            ┌──────────────────┐
                            │ Dynamo + vLLM    │
                            └────────┬─────────┘
                                     │
                                     ▼
                              NVIDIA GPU
                                     │
                                     ▼
                              Agent Response
```

---

# 🔥 Why ForgeLM?

| Capability        | Value                              |
| ----------------- | ---------------------------------- |
| **30B Capacity**  | Large model capacity               |
| **3B Active**     | Sparse active computation          |
| **Mamba-2**       | Hybrid sequence modeling           |
| **MoE**           | Expert-based sparse computation    |
| **Attention**     | Context-dependent reasoning        |
| **MTP**           | Multi-token prediction             |
| **1M Context**    | Long-context workloads             |
| **NVFP4**         | Optimized low-precision deployment |
| **Agentic Focus** | Autonomous workflows               |
| **Tool Calling**  | Multi-step tool use                |
| **RAG**           | Knowledge-grounded applications    |
| **Coding**        | Software-engineering workloads     |
| **Long Context**  | Large documents and repositories   |

---

# 🚀 Potential Applications

```text
ForgeLM
│
├── 🤖 Autonomous Agents
│
├── 🧑‍💻 Coding Agents
│
├── 🔎 RAG Systems
│
├── 🧠 Reasoning Systems
│
├── 🔧 Tool-Calling Agents
│
├── 🕸️ Browser Agents
│
├── 🧩 Multi-Agent Systems
│
├── 📚 Long-Context Applications
│
├── 💬 Enterprise Assistants
│
└── ⚙️ AI Automation Platforms
```

---

# 📌 Technical Snapshot

```text
╔══════════════════════════════════════════════════╗
║             FORGELM-30B-A3B-NVFP4                ║
╠══════════════════════════════════════════════════╣
║ Total Parameters       │ 30B                     ║
║ Active Parameters      │ 3B                      ║
║ Context                │ Up to 1M Tokens         ║
║ Architecture           │ Mamba-2 + MoE + Attn.   ║
║ Prediction             │ Multi-Token Prediction  ║
║ Precision              │ NVFP4                   ║
║ Runtime                │ PyTorch                 ║
║ Serving                │ Dynamo + vLLM           ║
║ Test GPU               │ NVIDIA H100             ║
║ GPU Families           │ Blackwell/Hopper/Ampere ║
║ Training               │ >20T Tokens             ║
║ RL                     │ GRPO                    ║
║ Primary Focus          │ Agentic AI              ║
║ Version                │ 1.0-preview             ║
║ Release                │ August 11, 2026         ║
╚══════════════════════════════════════════════════╝
```

---

# 👨‍💻 Project Attribution

<p align="center">

## **ForgeLM**

### AI Model Engineering • Agentic AI • Long-Context Intelligence

<br>

**Developed / Curated / Documented by**

# **Harsha Vardhan Upadrasta**

**AI Automation Engineer • AI/ML Engineer • Agentic AI Researcher**

</p>

---

# ⭐ Final Summary

**ForgeLM-30B-A3B-NVFP4** brings together:

```text
                 30B TOTAL PARAMETERS
                          │
                          ▼
                  3B ACTIVE PARAMETERS
                          │
                          ▼
              ┌───────────────────────┐
              │    HYBRID ARCHITECTURE│
              │                       │
              │ MAMBA-2 + MoE +       │
              │ ATTENTION + MTP       │
              └───────────┬───────────┘
                          │
                          ▼
                    1M TOKEN CONTEXT
                          │
                          ▼
                     NVFP4
                          │
                          ▼
                  EFFICIENT INFERENCE
                          │
                          ▼
                 ┌─────────────────┐
                 │   AGENTIC AI    │
                 └────────┬────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
       CODING            RAG         MULTI-AGENT
          │               │               │
          └───────────────┼───────────────┘
                          ▼
                  LONG-RUNNING AI
                     WORKFLOWS
```

> ## ⚡ ForgeLM
>
> ### **Forge Intelligence. Build Agents. Scale Reasoning.**

---

<p align="center">

**30B Total • 3B Active • 1M Context • NVFP4**

<br>

### Built for the next generation of agentic AI systems.

<br>

**© 2026 Harsha Vardhan Upadrasta**

</p>


<img width="1361" height="616" alt="image" src="https://github.com/user-attachments/assets/7a34a4b2-de3b-46da-9309-f73c16e9847b" />

<img width="1366" height="639" alt="image" src="https://github.com/user-attachments/assets/352c342c-b699-4181-9847-06ead9f47fa3" />

<img width="1360" height="694" alt="image" src="https://github.com/user-attachments/assets/70c3bbbc-9e7e-433e-8c8c-e9861425de28" />
