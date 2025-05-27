# Think&Cite

This is the repo for our paper: [Think&Cite: Improving Attributed Text Generation with Self-Guided Tree Search and Progress Reward Modeling]([https://arxiv.org/abs/2305.11747](https://arxiv.org/pdf/2412.14860)). The repo contains:

- The [data and model](#required-data-and-models) used in our work.
- The code for [model inference](#inference-and-evaluation).
- The code for [model evaluation](#inference-and-evaluation).

## Overview

Existing approaches to attributed text generation adopt an auto-regressive generation paradigm that can be characterized as "System 1", a mode of thinking which is fast and instinctive, but less accurate. Inspired by research on complex reasoning, we aim to develop models in the "System 2" mode for attribution to external evidence, requiring more in-depth, deliberative, and logical thinking. Second, attributed text generation often involves long text generation. We argue that the absence of explicit generation planning in previous work hinders advances in such systems.

In this paper, we propose Think&Cite, a novel framework integrating search algorithms into attributed text generation. We conceptualize the generation task as a multi-step reasoning problem, where the model generates a sentence in each step through an iterative think-verbalize-cite paradigm. To enhance this generation process, we propose Self-Guided Monte Carlo Tree Search (SG-MCTS), which extends the classic MCTS with two innovations. First, our approach leverages the self-reflection capability of LLMs to reflect on the intermediate states of MCTS in real time, so as to guide the tree expansion process and proactively avoid inadequate reasoning paths.  Second, we propose Progress Reward Models (PRM) to measure the progress of tree search from the root to the current state from two aspects, i.e., generation progress and attribution progress.

# Required Data and Models 

We utilize the ALCE evaluation benchmark proposed in EMNLP 2023, which can be download from this [repository](https://github.com/princeton-nlp/ALCE). We also adopt the GTR retriever and passage embeddings from the repository. 

To measure the generation progress during the MCTS process, we employ [LLaMA-3-8B-SFR-Iterative-DPO-R](https://huggingface.co/Salesforce/LLaMA-3-8B-SFR-Iterative-DPO-R) as the reward model. Besides, to compute the attribution progress reward, we use [t5_xxl_true_nli_mixture](https://huggingface.co/google/t5_xxl_true_nli_mixture) to perform natural language inference. 

# Inference and Evaluation

You can run the following bash file in /mcts directory for inference:

```
bash run.bash
```

You can also run the following bash file in /mcts directory for evaluation:

```
bash run_eval.bash
```

# Citation

Please cite the repo if you use the data or code in this repo.

```
@inproceedings{Think-Cite,
  author = {Junyi Li and Hwee Tou Ng},
  title = {Think&Cite: Improving Attributed Text Generation with Self-Guided Tree Search and Progress Reward Modeling},
  year = {2025},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational
                  Linguistics, {ACL} 2025, 2025},
}
```
