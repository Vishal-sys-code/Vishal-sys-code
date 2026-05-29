<div align="center">

<h1>Vishal S. Pandey</h1>

<p><b>Research Engineer &nbsp;·&nbsp; Founder, DeepBrain Labs</b></p>

<p><i>Building systems at the intersection of neuromorphic computation, efficient sequence modeling, and adaptive decision-making.</i></p>

<p>
  <a href="mailto:pandeyvishal.mlprof@gmail.com"><img src="https://img.shields.io/badge/Email-black?style=flat-square&logo=gmail&logoColor=white" /></a>
  <a href="https://scholar.google.com/citations?user=tVAsb08AAAAJ&hl=en"><img src="https://img.shields.io/badge/Google%20Scholar-4285F4?style=flat-square&logo=google-scholar&logoColor=white" /></a>
  <!-- <a href="https://arxiv.org/search/?query=vishal+pandey&searchtype=author"><img src="https://img.shields.io/badge/arXiv-B31B1B?style=flat-square&logo=arxiv&logoColor=white" /></a> -->
  <a href="https://www.kaggle.com/tmleyncodes"><img src="https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white" /></a>
  <!-- <a href="https://vayishu.bearblog.dev"><img src="https://img.shields.io/badge/Blog-FF5722?style=flat-square&logo=rss&logoColor=white" /></a> -->
  <a href="https://vayishu.bearblog.dev"><img src="https://img.shields.io/badge/Portfolio-543DE0?style=flat-square&logo=About.me&logoColor=white"></a>
  <a href="https://x.com/its_vayishu"><img src="https://img.shields.io/badge/X-000000?style=flat-square&logo=x&logoColor=white" /></a>
</p>

</div>

<br>

---

## About

I work on methods for efficient, adaptive, and interpretable machine learning, with a particular focus on where biological computation meets modern deep learning. My recent work spans neuromorphic sequence models, stable long-context attention, interpretable clinical ML, and multimodal sensing for real-world systems. I publish as an independent researcher and run [DeepBrain Labs](https://github.com/deepbrain-labs), a small effort building adaptive cognitive systems.

---

## Research Interests

- **Neuromorphic and spiking architectures**: energy-efficient inference via spike-based computation and biologically motivated learning rules
- **Attention mechanisms and long-context modeling**: stable associative memory, linear-time recurrent formulations, principled memory management
- **Reinforcement learning and sequential decision-making**: return-conditioned sequence models, adaptive control, and agentic pipelines
- **Multimodal and applied ML**: fusing visual and structured signals; interpretable clinical prediction with SHAP-grounded analysis
- **Systems-level efficiency**: JAX and PyTorch implementations tuned for research-to-deployment fidelity

---

## Publications

> Preprints on arXiv. Click a badge to open the paper.

| Year | Title | Authors | Links |
|------|-------|---------|-------|
| 2026 | **Variational Linear Attention: Stable Associative Memory for Long-Context Transformers** | Pandey, Singh | [![arXiv](https://img.shields.io/badge/arXiv-2605.11196-B31B1B?style=flat-square)](https://arxiv.org/abs/2605.11196) |
| 2026 | **A Unified Three-Stage Machine Learning Framework for Diabetes Detection, Subtype Discrimination, and Cognitive-Metabolic Hypothesis Testing** | Pandey, Laskar, Tewari | [![arXiv](https://img.shields.io/badge/arXiv-2605.13464-B31B1B?style=flat-square)](https://arxiv.org/abs/2605.13464) |
| 2025 | **Spiking Decision Transformers: Local Plasticity, Phase-Coding, and Dendritic Routing for Low-Power Sequence Control** | Pandey, Biswas | [![arXiv](https://img.shields.io/badge/arXiv-2508.21505-B31B1B?style=flat-square)](https://arxiv.org/abs/2508.21505) [![Code](https://img.shields.io/badge/Code-181717?style=flat-square&logo=github)](https://github.com/Vishal-sys-code/neuromorphic_decision_transformer) |
| 2025 | **AgroSense: A Multimodal Framework for Crop and Soil Analysis in Precision Agriculture** | Pandey, Das, Biswas | [![arXiv](https://img.shields.io/badge/arXiv-2509.01344-B31B1B?style=flat-square)](https://arxiv.org/abs/2509.01344) [![Code](https://img.shields.io/badge/Code-181717?style=flat-square&logo=github)](https://github.com/Vishal-sys-code/AgroSense) |

<details>
<summary><b>Abstract highlights</b></summary>

<br>

**Variational Linear Attention (2605.11196):** Linear attention reduces softmax attention's quadratic cost to O(T), but its memory state grows as O(T) in Frobenius norm, causing progressive interference between stored associations. VLA reframes the memory update as an online regularised least-squares problem with an adaptive penalty matrix maintained via the Sherman-Morrison rank-1 formula. Provably stable recurrence: Jacobian spectral norm equals exactly 1 for all sequence lengths and head dimensions.

**Diabetes Detection Framework (2605.13464):** A reproducible three-stage ML pipeline covering binary detection (SVM-RBF / LR achieving ROC-AUC 0.825), silhouette-validated subtype clustering without ground-truth labels, and statistical analysis of the Ohio Longitudinal Cognitive Dataset (n=373) revealing a significant glycaemic-cognitive association (rho=0.208, p<0.0001) surviving Holm correction. SHAP identifies Glucose, BMI, and Age as dominant biomarkers.

**Spiking Decision Transformers (2508.21505):** Return-conditioned sequence modeling with Leaky Integrate-and-Fire neurons in each self-attention block, trained end-to-end via surrogate gradients. Matches or exceeds standard DT performance on CartPole-v1, MountainCar-v0, Acrobot-v1, and Pendulum-v1 while emitting fewer than ten spikes per decision — suggesting over four orders-of-magnitude reduction in per-inference energy.

**AgroSense (2509.01344):** Multimodal framework integrating visual and tabular signals for crop and soil analysis. Designed for precision agriculture workflows where heterogeneous sensor streams must be fused reliably across conditions.

</details>

---

## Research Codebases

| Repository | Description |
|------------|-------------|
| [**neuromorphic_decision_transformer**](https://github.com/Vishal-sys-code/neuromorphic_decision_transformer) | Code for *Spiking Decision Transformers* (arXiv:2508.21505). Spike-based self-attention with surrogate gradient training, LIF neurons, and three-factor plasticity. |
| [**AgroSense**](https://github.com/Vishal-sys-code/AgroSense) | Multimodal vision + tabular fusion for precision agriculture (arXiv:2509.01344). |
| [**variational-linear-attention**](https://github.com/Vishal-sys-code/variational-linear-attention) | Code for *Variational Linear Attention* (arXiv:2605.11196). Sherman-Morrison rank-1 memory update with provably unit Jacobian spectral norm. |
| [**diabetes-type-prediction**](https://github.com/Vishal-sys-code/diabetes-type-prediction) | Code for the three-stage diabetes ML framework (arXiv:2605.13464). Detection, subtype clustering, and cognitive-metabolic analysis with SHAP explainability. |

---

## Systems and Tooling

| Repository | Description |
|------------|-------------|
| [**SmartTraffic-RL**](https://github.com/Vishal-sys-code/SmartTraffic-RL) | Gym-style RL environment for adaptive urban traffic signal control with macroscopic flow models and optional SUMO integration. |
| [**AgenticCyberOps**](https://github.com/Vishal-sys-code/AgenticCyberOps) | Autonomous pipeline for security analysis, penetration testing, and threat triage using LLM-backed agentic orchestration. |

> 56+ repositories total, including paper re-implementations, RL experiments, generative model prototypes, and systems utilities.

---

## Stack

**Languages**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-00599C?style=flat-square&logo=c%2B%2B&logoColor=white)
![C](https://img.shields.io/badge/C-A8B9CC?style=flat-square&logo=c&logoColor=black)

**Frameworks**

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![JAX](https://img.shields.io/badge/JAX-8A2BE2?style=flat-square&logo=google&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)

**Infrastructure**

![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat-square&logo=linux&logoColor=black)
![Git](https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white)

**Research domains:** Transformers · Spiking Neural Networks · Reinforcement Learning · Multimodal Fusion · RAG Systems · Linear Attention · Interpretable ML · Neuromorphic Computing

---

## DeepBrain Labs

<div align="center">

[**DeepBrain Labs**](https://github.com/deepbrain-labs) is a small independent research effort focused on building adaptive cognitive systems and computational models of intelligence. Current directions include associative memory architectures, neuromorphic control, and learned world models for planning under uncertainty.

*Open to research collaborations. Reach out via [email](mailto:pandeyvishal.mlprof@gmail.com) or [X](https://x.com/its_vayishu).*

</div>
