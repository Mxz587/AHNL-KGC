# AHNL-KGC

Code for **AHNL-KGC: An Adaptive Hard Negative Learning for Aircraft Fault Knowledge Graph Completion**.

AHNL-KGC improves knowledge graph completion by generating harder negative samples through
conditional diffusion guided by query structure, and by filtering unreliable candidates with
adaptive reliability constraints. Multi-trajectory repulsion enhances candidate coverage near
decision boundaries, while safe gating suppresses supervision noise from potential false
negatives.

## Files

| File | Description |
| --- | --- |
| `run.py` | Entry point: arguments, data loading, train/valid/test loop |
| `model.py` | `KGEModel`: KGE scorers and the hard-negative training step |
| `model_cond.py` | `Diffusion_Cond`: conditional diffusion model and multi-seed sampling |
| `dataloader.py` | Dataset and negative-sampling iterators |

## Requirements

```bash
pip install -r requirements.txt
```

## Data

Each dataset directory should contain tab-separated `entities.dict`, `relations.dict`,
`train.txt`, `valid.txt`, `test.txt`. Preprocessed FB15k-237, WN18 and WN18RR are available
from [OpenKE](https://github.com/thunlp/OpenKE) or
[RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding).

## Usage

Train (FB15k-237 example):

```bash
python run.py \
  --data_path data/FB15k-237 \
  --do_train --do_valid --do_test \
  --cuda --model TransE \
  -n 1024 -d 500 -g 12.0 -b 1024 -lr 0.01 -adv -a 1.0 \
  --max_steps 100000 --save_path ./ckpt_fb15k237 \
  --timesteps 30 --d_epoch 3 --diff_lr 1e-3 --num_seeds 5
```

Evaluate:

```bash
python run.py \
  --data_path data/FB15k-237 \
  --do_valid --do_test \
  --cuda --model TransE \
  -de -d 500 -g 12.0 -b 1024 -n 1024 \
  --init_checkpoint ./ckpt_fb15k237
```

Run `python run.py --help` for the full option list.

## Results

| Dataset | MRR |
| --- | --- |
| FB15k-237 | 0.3400 |
| WN18 | 0.9552 |
| WN18RR | 0.4830 |
