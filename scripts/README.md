# Training Scripts

This directory contains training and inference scripts for the recommendation pipeline.

## Scripts

### `train_retrieval.py`
Train Stage 1 retrieval models (e.g., LRURec, MMGCN).

**Usage:**
```bash
python scripts/train_retrieval.py
```

**Features:**
- Trains retrieval model on training data
- Evaluates on validation and test sets
- Saves retrieved candidates for Stage 2 reranking
- Exports results to `experiments/retrieval/{method}/{dataset}/seed{seed}/`

### `train_rerank.py`
Train Stage 2 reranking models (e.g., Qwen LLM).

**Usage:**
```bash
python scripts/train_rerank.py
```

**Features:**
- Trains reranking model on candidates from Stage 1
- Uses LLM (Qwen) for reranking
- Evaluates reranking performance

### `train_pipeline.py` ⭐ NEW
Train end-to-end two-stage pipeline.

**Usage:**
```bash
python scripts/train_pipeline.py \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method identity \
    --rerank_top_k 50
```

**Features:**
- Trains both Stage 1 and Stage 2 in sequence
- Evaluates full pipeline performance
- Compares Stage 1 only vs Full pipeline

**Arguments:**
- `--retrieval_method`: Retrieval method (lrurec, mmgcn)
- `--retrieval_top_k`: Number of candidates from Stage 1
- `--rerank_method`: Rerank method (identity, random, qwen)
- `--rerank_top_k`: Number of final recommendations
- `--metric_k`: Cutoff for evaluation metrics (default: 10)

## Workflow

### 1. Train Stage 1 only:
```bash
python scripts/train_retrieval.py
```

### 2. Train Stage 2 (using Stage 1 candidates):
```bash
python scripts/train_rerank.py
```

### 3. Train end-to-end:
```bash
python scripts/train_pipeline.py \
    --retrieval_method lrurec \
    --rerank_method qwen
```

## Output

All scripts save results to `experiments/` directory:
- `experiments/retrieval/{method}/{dataset}/seed{seed}/` - Stage 1 results
- `experiments/rerank/{method}/{dataset}/seed{seed}/` - Stage 2 results
- `experiments/pipeline/{retrieval}_{rerank}/{dataset}/seed{seed}/` - Full pipeline results

