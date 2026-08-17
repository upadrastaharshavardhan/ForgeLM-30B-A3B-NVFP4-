NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
Model Summary
Total Parameters	30B (3B active)
Architecture	MoE - Mamba-2 + MoE + Attention hybrid
Context Length	Up to 1M tokens
Supported Languages	English (and coding languages), Spanish, French, German, Italian, Japanese
Recommended Sampling	Temperature 1.0, Top_P 0.95
Best For	Long-running autonomous agents, sub-agent workhorse deployments, and agentic workflows
License	OpenMDW License Agreement, version 1.1
Release Date	August 11, 2026
Model Overview
Model Developer: NVIDIA Corporation

Model Dates: December 2025 - May 2026

Data Freshness:

The pre-training data has a cutoff date of September 2025.
The post-training data has a cutoff date of May 2026.
What is Nemotron?
NVIDIA Nemotron™ is a family of open models with open weights, training data, and recipes, delivering leading efficiency and accuracy for building specialized AI agents.

Description
NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 is a large language model (LLM) trained by NVIDIA.

The model employs a hybrid Mixture-of-Experts architecture, utilizing interleaved Mamba-2 and MoE layers, along with select Attention layers. The Lightning 3.5 model is released alongside a number of speculative decoding methods for faster text generation. The model has 3B active parameters and 30B parameters in total.

This model is ready for commercial use.

License and Terms of Use:
GOVERNING TERMS: The trial service is governed by the NVIDIA API Trial Terms of Service. Use of this model is governed by the OpenMDW License Agreement, version 1.1.

Benchmarks
Reasoning Benchmark Evaluations
We evaluated our model on the following benchmarks:

Task	Nemotron-3.5-Lightning-30B-A3B-BF16	Nemotron-3.5-Lightning-30B-A3B-NVFP4
General Knowledge		
MMLU Pro	81.94	81.62
AA-Omniscience	17.50	16.63
Reasoning		
GPQA Diamond (no tools)	75.44	75.57
HLE (text-only, no tools)	11.72	10.47
SciCode	32.60	31.38
Coding & Agentic		
SWE-bench Verified	51.56	52.80
SWE-bench Multilingual	39.33	36.47
Terminal-Bench 2.1	24.58	23.46
PinchBench	85.37	83.43
BrowseComp	36.97	36.81
τ³-bench (Banking)	9.28	9.48
GDPval-AA-V2	832	865
Instruction Following		
IFBench (loose)	71.88	72.88
Long Context		
AA-LCR	52.00	49.19
For reproducibility, the evaluation recipes, installation instructions, and commands for NVIDIA Nemotron 3.5 Lightning were collected and published in NeMo Gym. The reported results cover the release evaluation suite, including knowledge and reasoning, instruction following, coding, agentic, tool-use, and long-context. Most evaluations use NeMo Gym-native harnesses while a small subset, including SWE-Bench and Terminal-Bench, used NeMo Evaluator natively. The published recipes specify the benchmark-specific containers, prompts, inference parameters, parser configurations, and scoring settings used to produce the results.

These numbers were measured with and apply to the official NVFP4 checkpoint

Deployment Geography: Global
Use Case
NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 is a general purpose reasoning and chat model intended to be used in English and coding languages. Other non-English languages (Spanish, French, German, Italian, Japanese) are also supported. Intended for developers designing AI Agent systems, chatbots, RAG systems, and other AI-powered applications. Also suitable for typical instruction-following tasks.

Release Date
Hugging Face — 08/11/2026
Build.NVIDIA.com — 08/11/2026
Model Architecture
Architecture Type: Mixture-of-Experts Hybrid (Mamba + Transformer)
Network Architecture: Nemotron-3-Lightning + Multi-Token Prediction (MTP)
Number of model parameters: 30B Total / 3B Active
Model Design
The model was pre-trained with over 20T tokens and supports up to 1M context length. The pre-training phase used an NVFP4 recipe. The model includes Multi-Token Prediction (MTP) layers, which predict multiple future tokens to provide richer training signals.

Training Methodology
Stage 1: Pre-Training

NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 model was pre-trained using an NVFP4 recipe with crawled and synthetic code, math, science, and general knowledge data.
Software used for pre-training: Megatron-LM
Stage 2: Continued Pre-Training for Multi-Token Prediction (MTP)

The model underwent a continued pre-training phase to train its Multi-Token Prediction (MTP) layers. In this stage, MTP heads learn to predict multiple future tokens, providing richer training signals to the base model. This phase aligns the MTP layers with the base model's distribution.
Stage 3: Supervised Fine-Tuning

The model was further fine-tuned on synthetic code, math, science, tool calling, instruction following, structured outputs, and general knowledge data. This stage incorporated data designed to support long-range retrieval and multi-document aggregation.
Stage 4: Reinforcement Learning

The model underwent multi-environment reinforcement learning using GRPO (Group Relative Policy Optimization) across math, code, science, instruction following, multi-step tool use, multi-turn conversations, and structured output environments. It utilized an asynchronous RL architecture that decouples training from inference and leverages MTP to accelerate rollout generation.
Software used for reinforcement learning: NeMo RL, NeMo Gym
Stage 5: Post-training Quantization (PTQ)

We performed post-training quantization (PTQ) with Nvidia Model Optimizer using the following recipe: Four Over Six NVFP4 (a variant of static MSE calibration) W4A16 on routed and shared experts, FP8 per-tensor dynamic scales on mamba in_proj/out_proj and KV cache. We used a subset of the Nemotron Ultra validation set for calibration with 1000 samples at 32k token length.
NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 is a result of the above work.

Input
Input Type(s): Text
Input Format(s): String
Input Parameters: One-Dimensional (1D): Sequences
Other Properties Related to Input: Maximum context length up to 1M tokens. Supported languages include English, Spanish, French, German, Italian, and Japanese.
Output
Output Type(s): Text
Output Format: String
Output Parameters: One-Dimensional (1D): Sequences
Other Properties Related to Output: Maximum context length up to 1M tokens
Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

Software Integration
Runtime Engine(s): PyTorch
Supported Hardware Microarchitecture Compatibility: NVIDIA Blackwell; NVIDIA Hopper (NVFP4 / W4A16); NVIDIA Ampere (W4A16)
Preferred/Supported Operating System(s): Linux
The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

Model Version(s)
NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 — 1.0-preview (08/11/2026)
Training, Testing, and Evaluation Datasets
Training
Data Modality: Text Training Data Size: More than 20 Trillion Tokens Dataset partition: Training [100%], testing [0%], validation [0%] Time period for training data collection: 2013 to December 2025 Time period for testing data collection: 2013 to December 2025 Time period for validation data collection: 2013 to December 2025 Data Collection Method by dataset: Hybrid: Automated, Manually-Collected, Synthetic Labeling Method by dataset: Hybrid: Automated, Manually-Labeled, Synthetic

NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 is pre-trained on a large corpus of high-quality curated and synthetically-generated data. It is trained in the English language, as well as 19 other spoken languages and 43 programming languages. Our sources cover a variety of document types such as: webpages, dialogue, articles, and other written materials. The corpus spans domains including legal, math, science, finance, and more. We also include a small portion of question-answering, and alignment style data to improve model accuracy. The model was pre-trained for more than 20 trillion tokens.

The post-training corpus for NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 consists of high-quality curated and synthetically-generated data. Primary languages used for post-training include English, French, German, Italian, Japanese, Spanish, and Chinese.

These datasets, such as FinePDFs, EssentialWeb, HotpotQA, SQuAD, and HelpSteer3, do not collectively or exhaustively represent all demographic groups (and proportionally therein). For instance, these datasets do not contain explicit mentions of demographic classes such as age, gender, or ethnicity in 64-99% of samples, depending on the source. In the subset where such terms are present, document-based datasets (FinePDFs and EssentialWeb) contain representational skews, such as references to "male" outnumbering those to "female", and mentions of "White" as the most frequent among ethnic identifiers (comprising 43-44% of ethnicity mentions). To mitigate these imbalances, we recommend considering evaluation techniques such as bias audits, fine-tuning with demographically balanced datasets, and mitigation strategies like counterfactual data augmentation to align with the desired model behavior. This evaluation used a 3,000-sample subset per dataset, identified as the optimal threshold for maximizing embedder accuracy.

During post-training, we generate synthetic data by distilling trajectories, solutions, and translations from strong teacher models and agent systems, often grounded in real tasks or documents and aggressively filtered for quality. For math, code, and science, we start from curated problem sets and use open source permissive models such as GPT-OSS-120B to produce step-by-step reasoning traces, candidate solutions, best-of-n selection traces, and verified CUDA kernels. For long-context and science, we build synthetic QA and reasoning data by retrieving passages from long documents, generating MCQ/OpenQA questions and answers, and paraphrasing them into multiple prompt/response formats to ensure diversity. Across all pipelines we stack automated verification—compilers, numerical checks, language identification—to ensure our data is high quality.

For all domains, we apply a unified data filtering pipeline to ensure that only high-quality, license-compliant, and verifiable samples are used for post-training. We first discard malformed examples using structural checks (e.g., missing tool definitions when tool calls are present). We then aggressively filter reasoning traces exhibiting pathological repetition, such as repeated n-grams within a sliding window or across the entire trajectory, which we found to be a strong indicator of malformed or low-quality reasoning. Finally, based on internal audits of synthetically generated datasets, we observed that some teacher models occasionally produce reasoning traces and final responses that implicitly align with specific political entities or promote nationalistic narratives. To mitigate this, we apply targeted keyword- and regex-based filters and remove all trajectories matching such behavior.

Alongside the model, we release our final pre-training and post-training data, as outlined in this section. For ease of analysis, there is a sample set that is ungated. For all remaining code, math and multilingual data, gating and approval is required, and the dataset is permissively licensed for model training purposes.

For Detailed Dataset Information: Click here!
Testing Datasets:
Data Collection Method by dataset

Hybrid: Automated, Manually-Collected, Synthetic
Labeling Method by dataset
Hybrid: Automated, Manually-Labeled, Synthetic
Properties: This corpus comprises a mix of high-quality standard benchmarks and test suites for modern agentic AI. These benchmarks test model capabilities on tasks such as tool-calling and instruction following.
Evaluation Datasets:
Data Collection Method by dataset

Hybrid: Automated, Manually-Collected, Synthetic
Labeling Method by dataset
Hybrid: Automated, Manually-Labeled, Synthetic
Properties: This corpus comprises a mix of high-quality standard benchmarks and test suites for modern agentic AI. These benchmarks test model capabilities on tasks such as tool-calling and instruction following.
Inference
Acceleration Engine: Dynamo + vLLM
Test Hardware: NVIDIA Hopper - H100

<img width="1361" height="616" alt="image" src="https://github.com/user-attachments/assets/7a34a4b2-de3b-46da-9309-f73c16e9847b" />

<img width="1366" height="639" alt="image" src="https://github.com/user-attachments/assets/352c342c-b699-4181-9847-06ead9f47fa3" />

<img width="1360" height="694" alt="image" src="https://github.com/user-attachments/assets/70c3bbbc-9e7e-433e-8c8c-e9861425de28" />
