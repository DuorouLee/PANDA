# PANDA: Progressive Adaptive Noise Denoising Architecture for Graph-Based Multimodal Fake News Detection

## Description

This repository contains the implementation and experiment materials for
PANDA, a progressive adaptive noise denoising architecture for graph-based
multimodal fake news detection. PANDA is designed to improve the robustness of
LLM-assisted graph learning by reducing the effect of noisy pseudo-labels and
uncertain multimodal evidence during message passing.

The repository includes scripts for preprocessing, graph construction, model
training, ablation studies, robustness analysis, and result generation for the
experiments reported in the manuscript.

## Dataset Information

The experiments use three third-party benchmark datasets:

- PHEME: a multi-event rumor detection dataset used for rumor/non-rumor
  classification.
- Twitter: an English fake news detection benchmark used in prior social-media
  fake news studies.
- Weibo: a Chinese social-media fake news detection benchmark.

The original datasets are not redistributed in this repository because they are
third-party datasets and may be subject to platform terms, privacy constraints,
or redistribution restrictions. Users should obtain the datasets from their
original sources and follow the applicable terms of use.

The manuscript cites the original dataset references:

- Zubiaga et al., "Analysing how people orient to and spread rumours in social
  media by looking at conversational threads" (PHEME).
- Ma et al., "Detecting rumors from microblogs with recurrent neural networks"
  (Twitter).
- Qi et al., "Exploiting multi-domain visual information for fake news
  detection" (Weibo).

After obtaining the datasets, place the processed files under:

```text
script/dataset/
├── pheme/
├── twitter/
└── weibo/
```

Each dataset directory is expected to contain files such as:

```text
dataforGCN_train.csv
dataforGCN_test.csv
TweetEmbeds.pt
TweetTextEmbeds.pt
TweetImageEmbeds.pt
TweetGraph.pt
*_analysis_results.csv
```

## Code Information

The main repository components are:

```text
script/      PANDA training, preprocessing, pseudo-label, and analysis scripts
baselines/   Baseline method implementations used for comparison
results/     Experiment summaries and generated figures
logs/        Training and evaluation logs
paper/       Manuscript source files and submission materials
```

Important scripts include:

```text
script/construction.py                 Build multimodal embeddings and graphs
script/main.py                         GLPN-LLM baseline
script/main_step1_dceloss.py           PANDA Stage 1: DCE loss
script/main_step2_zpd.py               PANDA Stage 2: ZPD selection
script/main_step3_prototype.py         PANDA full model
script/run_ablation.py                 Ablation experiments
script/run_loss_comparison.py          Loss comparison experiments
script/run_noise_robustness.py         Noise robustness experiments
script/run_zpd_coverage.py             ZPD coverage analysis
```

## Requirements

The code was implemented in Python and requires PyTorch. Core dependencies
include:

```text
numpy
pandas
scikit-learn
torch
torch-geometric
tqdm
```

Install dependencies in an environment with a compatible PyTorch and CUDA
version:

```bash
pip install numpy pandas scikit-learn tqdm
pip install torch torch-geometric
```

If using a CUDA GPU, install the PyTorch and PyTorch Geometric builds that
match your CUDA version.

## Usage Instructions

Run commands from the `script/` directory.

Build graph inputs after preparing the raw dataset files:

```bash
cd script
python construction.py --dataset_name pheme --save_csv
```

Run the GLPN-LLM baseline:

```bash
cd script
python main.py --dataset_name pheme --seed 123 --device cuda
```

Run the full PANDA model:

```bash
cd script
python main_step3_prototype.py --dataset_name pheme --seed 123 --device cuda
```

Use `--device cpu` if CUDA is unavailable. To run other datasets, replace
`pheme` with `twitter` or `weibo`.

The scripts report accuracy, precision, recall, and macro-F1. Output logs and
result CSV files are written under `script/logs/`, `script/results/`, or the
dataset-specific result directory depending on the script.

## Methodology

The experimental pipeline is:

1. Load text-image fake news samples from a selected dataset.
2. Encode multimodal content and construct a graph representation.
3. Load or generate LLM pseudo-labels for unlabeled or weakly supervised nodes.
4. Train the graph model with Distributional Confidence Estimation.
5. Select reliable pseudo-labeled nodes using Zone of Proximal Development
   filtering.
6. Apply prototype-guided contrastive rectification for uncertain nodes.
7. Evaluate the model on held-out test data using classification metrics.

## Reproducibility Notes

The manuscript reports results over multiple random seeds. Example seed values
used in the experiment scripts include:

```text
123, 133, 223, 323, 333
```

Because the third-party datasets are not redistributed here, exact
reproduction requires obtaining the original datasets and preparing them into
the expected directory structure.

## Citation

If you use this code, please cite the associated manuscript:

```text
Liangni Li. PANDA: Progressive Adaptive Noise Denoising Architecture for
Graph-Based Multimodal Fake News Detection. PeerJ Computer Science.
Manuscript under review.
```

Please also cite the original dataset papers when using PHEME, Twitter, or
Weibo.

## License

This repository is provided for academic research and peer review purposes.
The third-party datasets are governed by their original licenses and terms of
use and are not redistributed in this repository.

## Contact

For questions about the code or manuscript, please contact the corresponding
author listed in the manuscript submission system.
