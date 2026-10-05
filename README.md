# AHNL-KGC

**AHNL-KGC: An Adaptive Hard Negative Learning for Aircraft Fault Knowledge Graph Completion**

Official PyTorch implementation of the adaptive hard-negative learning method for knowledge
graph completion (AHNL-KGC), which integrates **multi-trajectory diffusion generation** with
**adaptive reliability filtering** to improve both decision-boundary coverage and supervision
reliability.

---

## Abstract

Hard-negative learning improves knowledge graph completion by introducing more challenging
negative samples. However, excessive reliance on model scores can concentrate candidates in a
few local regions, limiting the coverage of complex decision boundaries. Moreover, the
incompleteness of knowledge graphs means that some high-scoring candidates may correspond to
valid but unobserved facts, introducing erroneous supervision when treated as negatives.

To address these challenges, we propose an adaptive hard-negative learning method for knowledge
graph completion (AHNL-KGC), which integrates multi-trajectory diffusion generation with
adaptive reliability filtering to improve both boundary coverage and supervision reliability.
Specifically, self-adversarial weighted ranking provides base supervision, while conditional
diffusion guided by query structure generates virtual hard-negative candidates. Multi-trajectory
repulsion constraints enhance candidate coverage and directional diversity near decision
boundaries. Adaptive filtering jointly considers boundary position, diffusion timestep
reliability, and relational semantic consistency to suppress supervision noise from potential
false negatives. Dynamic boundary-consistency optimization further balances the training
contributions of discrete and virtual negative samples.

Experiments on **FB15k-237**, **WN18**, and **WN18RR** yield best mean reciprocal rank (MRR)
scores of **0.3400**, **0.9552**, and **0.4830**, respectively. The proposed method provides an
effective approach to constructing high-quality negative samples for incomplete knowledge
graphs, enhancing the discriminative power of learned representations and the accuracy of
missing-fact prediction.

---

## Method Overview

| Component in the paper | Implementation |
| --- | --- |
| Self-adversarial weighted ranking (base supervision) | `model.KGEModel.train_step`, `negative_adversarial_sampling` / `adversarial_temperature` args |
| Conditional diffusion guided by query structure | `model_cond.Diffusion_Cond` (`q_sample`, `p_losses`, `p_sample`, `p_sample_loop_keep`), geometric/query conditions in `model.get_geometric_condition` / `get_inverse_geometric_condition` |
| Multi-trajectory repulsion constraints | `model_cond.Diffusion_Cond.sample_multi_seed_from_anchor`, `model.seed_repulsion_loss` |
| Adaptive reliability filtering (boundary position + timestep reliability + relational semantic consistency) | virtual-negative safe gating in `model.KGEModel.train_step` (`virt_*` args) and relation-prototype gating `model.proto_gating_weight` |
| Dynamic boundary-consistency optimization | `virt_loss_weight` / `virt_warmup_steps` scheduling and `proj_consistency_weight` in `model.KGEModel.train_step` |

Supported scoring functions: `TransE`, `DistMult`, `ComplEx`, `RotatE`, `pRotatE`.

---

## Repository Structure

```
AHNL/
├── run.py          # entry point: argument parsing, data loading, train/valid/test loop
├── model.py        # KGEModel (KGE scorers + train_step with hard-negative learning), diffusion helpers
├── model_cond.py   # conditional diffusion model (Diffusion_Cond): schedules, denoiser, sampling
├── dataloader.py   # TrainDataset / TestDataset / BidirectionalOneShotIterator
├── requirements.txt
└── README.md
```

---

## Requirements

- Python >= 3.8
- PyTorch >= 1.10 (CUDA build recommended)
- NumPy, scikit-learn, tqdm

```bash
pip install -r requirements.txt
```

---

## Data Preparation

The code follows the standard 1-N scoring data layout. For each dataset place the following
tab-separated files under a single directory:

```
data/FB15k-237/
├── entities.dict    # <entity_id>\t<entity_name>
├── relations.dict   # <relation_id>\t<relation_name>
├── train.txt        # <head>\t<relation>\t<tail>
├── valid.txt
└── test.txt
```

Preprocessed **FB15k-237**, **WN18**, and **WN18RR** in this format can be obtained from the
[OpenKE](https://github.com/thunlp/OpenKE) or [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)
repositories.

---

## Training

Example on FB15k-237:

```bash
python run.py \
  --data_path data/FB15k-237 \
  --do_train --do_valid --do_test \
  --cuda \
  --model TransE \
  -n 1024 -d 500 -g 12.0 -b 1024 -lr 0.01 \
  -adv -a 1.0 \
  --max_steps 100000 --save_checkpoint_steps 10000 --valid_steps 10000 --log_steps 100 \
  --save_path ./ckpt_fb15k237 \
  --timesteps 30 --d_epoch 3 --diff_lr 1e-3 --diff_weight 1.0 \
  --num_seeds 5 \
  --diff_keep_start 6 --diff_keep_end 15 \
  --virt_loss_weight 0.05 --virt_warmup_steps 20000 \
  --proto_beta 1.0 --proto_tau 0.3
```

Key options (see `run.py --help` for the full list):

| Argument | Default | Description |
| --- | --- | --- |
| `--model` | `TransE` | Scoring function (`TransE`/`DistMult`/`ComplEx`/`RotatE`/`pRotatE`) |
| `-n` | `1024` | Negative sample size |
| `-d` | `500` | Hidden (embedding) dimension |
| `-g` | `12.0` | Gamma margin |
| `-adv` / `-a` | off / `1.0` | Self-adversarial negative sampling and its temperature |
| `--timesteps` | `30` | Reverse diffusion steps |
| `--d_epoch` | `3` | Diffusion training iterations per KGE step |
| `--num_seeds` | `5` | Number of multi-trajectory diffusion seeds per query (repulsion) |
| `--div_weight` | `0.0` | Seed-repulsion loss weight (`0` disables) |
| `--diff_keep_start` / `--diff_keep_end` | `6` / `15` | Kept intermediate diffusion indices (reliability band) |
| `--anchor_cond_alpha` | `0.3` | Anchor blending weight in the raw-space condition |
| `--diff_init_sigma` | `0.1` | Noise scale for anchor-centered diffusion init |
| `--virt_loss_weight` | `0.05` | Max weight of the virtual (diffusion) negative loss |
| `--virt_warmup_steps` | `20000` | Warm-up steps for the virtual loss weight |
| `--proto_beta` | `1.0` | Strength of relation-prototype gating (`0` disables) |
| `--proto_path` | `None` | Relation-prototype `.npz`; defaults to `<init_checkpoint>/relation_prototypes_M32.npz` |
| `--eval_degree_buckets` | off | Report long-tail degree-bucket metrics during evaluation |

---

## Evaluation

```bash
python run.py \
  --data_path data/FB15k-237 \
  --do_valid --do_test \
  --cuda --model TransE \
  -de -d 500 -g 12.0 -b 1024 -n 1024 \
  --init_checkpoint ./ckpt_fb15k237
```

Metrics reported: MRR, MR, Hits@1, Hits@3, Hits@10 (filtered ranking).

---

## Results

| Dataset | MRR |
| --- | --- |
| FB15k-237 | 0.3400 |
| WN18 | 0.9552 |
| WN18RR | 0.4830 |

---

## Citation

```bibtex
@article{ahnlkgc,
  title   = {AHNL-KGC: An Adaptive Hard Negative Learning for Aircraft Fault Knowledge Graph Completion},
  author  = {Meng, X. and others},
  journal = {[venue]},
  year    = {[year]}
}
```

> Please update the citation entry once the paper is published.

---

## Acknowledgements

The KGE backbone and data pipeline follow the conventions of
[RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding) and
[OpenKE](https://github.com/thunlp/OpenKE).

## License

Released for academic and research use.
