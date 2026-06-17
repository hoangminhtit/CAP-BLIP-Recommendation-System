# Chat Reranking on This Repo's Datasets

This folder contains the original chat-reranking experiment plus adapters for the
CSV dataset format produced by this repo.

## 1. Prepare the base dataset

From the repo root:

```bash
python data_prepare.py --dataset_code beauty --min_rating 3 --min_uc 20 --min_sc 20 --use_text
```

Optional but recommended: run a retrieval model first so ChatGPT reranks real
stage-1 candidates.

```bash
python scripts/train_retrieval.py --dataset_code beauty --min_rating 3 --min_uc 20 --min_sc 20 --retrieval_method lrurec
```

## 2. Export chat-reranking artifacts

```bash
python experiments/chat-reranking/prepare_seqrec_dataset.py \
  --dataset_code beauty \
  --min_rating 3 --min_uc 20 --min_sc 20 \
  --retrieval_method lrurec \
  --seed 42 \
  --split test \
  --top_m 20
```

For a quick smoke test before retrieval is available, add `--fallback_popularity`.

The prepared files are written to:

```text
experiments/chat-reranking/prepared/beauty/
```

## 3. Run the LLM reranker

OpenAI example:

```bash
python experiments/chat-reranking/llm_reranker.py \
  --datasetpath experiments/chat-reranking/prepared/beauty \
  --domain "beauty product" \
  --fold 0 \
  --model gpt-4o-mini \
  --promptpath experiments/chat-reranking/prompts/template_beauty.json \
  --prompt_id 1 \
  --baseline_recs lrurec-test-top20.tsv \
  --rerank_top_m 20 \
  --top_n 10 \
  --run_with_sample_users 0 \
  --debug_mode 0 \
  --request_sleep 0
```

Set `OPENAI_API_KEY` in the environment, or pass `--openai_key`.

HuggingFace Llama 2 example:

```bash
python experiments/chat-reranking/llm_reranker.py \
  --datasetpath experiments/chat-reranking/prepared/beauty \
  --domain "beauty product" \
  --fold 0 \
  --model meta-llama/Llama-2-7b-chat-hf \
  --promptpath experiments/chat-reranking/prompts/template_beauty.json \
  --prompt_id 7 \
  --baseline_recs lrurec-test-top20.tsv \
  --rerank_top_m 20 \
  --top_n 10 \
  --run_with_sample_users 0 \
  --debug_mode 0
```

For this gated Meta model, set `HF_TOKEN` in the environment or pass
`--hf_auth_token`. The token must belong to a HuggingFace account that has
accepted access to `meta-llama/Llama-2-7b-chat-hf`.

## 4. Evaluate output

```bash
python experiments/chat-reranking/evaluate_outputs.py \
  --datasetpath experiments/chat-reranking/prepared/beauty \
  --output_json experiments/chat-reranking/prepared/beauty/recs/reranked/gpt-4o-mini-div-p1-lrurec-test-top20.tsv.json \
  --split test \
  --ks 1,5,10,20
```
