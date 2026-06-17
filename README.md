# Sequential Reranking for Recommendation

Hệ thống hai giai đoạn: **Retrieval** sinh candidates và **Reranking** sắp xếp lại bằng LLM/multimodal. Qwen đã được hợp nhất: `--rerank_method qwen` dùng chung cho 3 mode (`text_only`, `caption`, `VIU`); `qwen3vl` chỉ còn là alias. `raw_image` không còn hỗ trợ.

## Tổng quan nhanh
- Retrieval: LRURec, MMGCN, VBPR, BM3.
- Rerank: Qwen (unified), VIP5; alias `qwen3vl` trỏ về Qwen unified.
- Multimodal: image, text, caption (BLIP2), VIU (Qwen3-VL, ở dạng text đã sinh sẵn).
- Train độc lập từng stage hoặc end-to-end; metric: Recall/NDCG/Hit @5/10/20; early stopping.

## Cài đặt
```bash
pip install -r requirements.txt
# cần transformers mới cho Qwen3-VL pipeline sinh summary
pip install git+https://github.com/huggingface/transformers
```

## Chuẩn bị dữ liệu
```bash
# chỉ ratings
python data_prepare.py --dataset_code beauty --min_rating 3 --min_uc 20 --min_sc 20

# thêm image + text (cho MMGCN/VBPR/BM3 và để sinh caption/summary)
python data_prepare.py --dataset_code beauty --min_rating 3 --min_uc 20 --min_sc 20 --use_image --use_text

# sinh caption (BLIP2)
python data_prepare.py --dataset_code beauty --min_rating 3 --min_uc 20 --min_sc 20 --use_image --generate_caption

# sinh VIU (Qwen3-VL, lưu text)
python data_prepare.py --dataset_code beauty --min_rating 3 --min_uc 20 --min_sc 20 --use_image --generate_viu
```

## Train Retrieval (Stage 1)
```bash
python scripts/train_retrieval.py --retrieval_method lrurec
python scripts/train_retrieval.py --retrieval_method mmgcn
python scripts/train_retrieval.py --retrieval_method vbpr
python scripts/train_retrieval.py --retrieval_method bm3
```

## Train Rerank (Stage 2, standalone)
```bash
# Qwen unified (3 mode)
python scripts/train_rerank_standalone.py --rerank_method qwen --qwen_mode text_only --mode ground_truth
python scripts/train_rerank_standalone.py --rerank_method qwen --qwen_mode caption --mode ground_truth
python scripts/train_rerank_standalone.py --rerank_method qwen --qwen_mode VIU --mode ground_truth
# alias (backward-compat)
python scripts/train_rerank_standalone.py --rerank_method qwen3vl --qwen_mode caption --mode ground_truth

# VIP5
python scripts/train_rerank_standalone.py --rerank_method vip5 --mode ground_truth
```

## Rerank với candidates từ retrieval
```bash
python scripts/train_rerank_standalone.py \
  --rerank_method qwen --qwen_mode text_only \
  --mode retrieval --retrieval_method lrurec \
  --retrieval_top_k 200 --rerank_top_k 50
```

## End-to-end (Stage 1 + Stage 2)
```bash
python scripts/train_pipeline.py \
  --retrieval_method lrurec --retrieval_top_k 200 \
  --rerank_method qwen --qwen_mode text_only \
  --rerank_top_k 50 --rerank_mode retrieval
```

## Bảng phương pháp
### Retrieval
| Method | Yêu cầu | Command |
|--------|---------|---------|
| lrurec | ratings | `--retrieval_method lrurec` |
| mmgcn  | image/text + CLIP | `--retrieval_method mmgcn` |
| vbpr   | image + CLIP | `--retrieval_method vbpr` |
| bm3    | image + text + CLIP | `--retrieval_method bm3` |

### Rerank
| Method | Mô tả | Data cần | Command |
|--------|-------|----------|---------|
| qwen / qwen3vl (alias) | Unified LLM 3 mode | text; thêm caption/VIU nếu dùng | `--rerank_method qwen --qwen_mode <mode>` |
| vip5 | Multimodal T5 | image + CLIP | `--rerank_method vip5` |

### Qwen modes (unified)
| Mode | Mô tả | Model gợi ý | Data prep |
|------|-------|-------------|-----------|
| text_only | Chỉ description | `qwen3-0.6b`, `qwen3-1.6b` | text |
| caption | Thêm BLIP2 caption | `qwen3-0.6b`, `qwen3-1.6b`, `qwen3-2bvl` | `--use_image --generate_caption` |
| VIU | Thêm Qwen3-VL VIU (text) | `qwen3-0.6b`, `qwen3-1.6b`, `qwen3-2bvl` | `--use_image --generate_viu` |

Lưu ý: cả 3 mode dùng chung `LLMModel`; không hỗ trợ `raw_image` trong unified.

### Rerank modes
| Mode | Dùng cho | Mô tả |
|------|----------|-------|
| ground_truth | Đánh giá độc lập | 1 GT + (N-1) negatives, N=`rerank_eval_candidates` |
| retrieval | Đánh giá theo candidates Stage 1 | Dùng output retrieval |

## Cấu hình chính (config.py / CLI)
- Retrieval: `--retrieval_epochs`, `--retrieval_lr`, `--batch_size_retrieval`, `--retrieval_patience`.
- Rerank: `--rerank_epochs`, `--rerank_lr`, `--rerank_batch_size`, `--rerank_patience`, `--rerank_eval_candidates` (default 50).
- Qwen: `--qwen_mode` (`text_only|caption|VIU`), `--qwen_model` (`qwen3-0.6b`, `qwen3-1.6b`, `qwen3-2bvl`), `--qwen_max_candidates`, `--qwen_max_history`, `--qwen_max_seq_length`, `--qwen_temperature`.
- Alias: `--qwen3vl_mode` deprecated, giữ để tương thích; dùng `--qwen_mode`.
- Hiệu năng: `--use_quantization` (4-bit Unsloth), `--use_torch_compile`.

## Output
```
data/preprocessed/{dataset_code}_.../
├── dataset_single_export.csv
├── clip_embeddings.pt
├── blip2_captions.pt
└── qwen3vl_viu.pt

experiments/
├── retrieval/{method}/{dataset_code}/seed{seed}/
│   ├── retrieved.csv
│   └── retrieved_metrics.json
└── rerank/{method}/{dataset_code}/seed{seed}/
    ├── model.pt
    └── metrics.json
```

## Troubleshooting nhanh
- OOM: giảm `--rerank_batch_size`, giảm `--qwen_max_candidates`, bật `--use_quantization`.
- Prompt bị cắt: tăng `--qwen_max_seq_length`.
- Thiếu caption/VIU: chạy `data_prepare.py` với `--generate_caption` / `--generate_viu`.
- Dùng alias `--rerank_method qwen3vl`: vẫn gọi unified Qwen.

## Cập nhật mới nhất
- Hợp nhất Qwen reranker: `qwen_reranker_unified.py` thay cho `qwen_reranker.py`/`qwen3vl_reranker.py`; `qwen3vl` là alias.
- Tất cả mode Qwen dùng `LLMModel`; bỏ `raw_image` trong unified.
- README và scripts đồng bộ `--qwen_mode`.

## License
[Add your license here]
