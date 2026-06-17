#hhh%%writefile config.py
import argparse
import os


EXPERIMENT_ROOT = 'experiments'


parser = argparse.ArgumentParser(description='Configuration for the project.')

parser.add_argument('--rerank_top_k', type=int, default=None, help='Number of final recommendations (used by train_pipeline.py)')
parser.add_argument('--metric_k', type=int, default=None, help='Cutoff for evaluation metrics (used by train_pipeline.py)')
parser.add_argument('--retrieval_top_k', type=int, default=None, help='Number of candidates from Stage 1 (used by train_pipeline.py)')
parser.add_argument('--rerank_method', type=str, default=None, help='Rerank method (used by train_pipeline.py)')
parser.add_argument('--rerank_mode', type=str, default=None, help='Rerank mode (used by train_pipeline.py)')

#=========================================================================
# Data preparation arguments
#=========================================================================
parser.add_argument('--data_path', type=str, default=None,
                    help='Path to data folder. If None, uses default "data" folder.')
parser.add_argument('--dataset_code', type=str, default='beauty')
parser.add_argument('--min_rating', type=int, default=4)    # minimum rating to consider positive
parser.add_argument('--min_uc', type=int, default=6)   # min rating/user
parser.add_argument('--min_sc', type=int, default=6)   # min rating/item
parser.add_argument('--use_image', action='store_true', default=False, help='Filter out items without image')
parser.add_argument('--use_text', action='store_true', default=False, help='Filter out items without text')
parser.add_argument('--generate_caption', action='store_true', default=False, 
                    help='Generate BLIP2 captions for images and save to CSV')
parser.add_argument('--generate_viu', action='store_true', default=False,
                    help='Generate Qwen3 VL VIU for images and save to CSV')
parser.add_argument('--viu_batch_size', type=int, default=8,
					help='Batch size for VIU generation (increase if GPU memory allows, recommended: 8-16 for T4)')
parser.add_argument('--viu_max_tokens', type=int, default=128,
					help='Maximum tokens for VIU generation (default: 64, reduced from 128 for speed)')
parser.add_argument('--use_quantization', action='store_true', default=True,
					help='Use 4-bit quantization for models (saves memory, may slightly reduce accuracy)')
parser.add_argument('--use_torch_compile', action='store_true', default=True,
					help='Use torch.compile() for faster inference (requires PyTorch 2.0+)')
parser.add_argument('--preload_all_images', action='store_true', default=True,
					help='Pre-load all images into memory before processing (faster but uses more RAM)')

#===========================================================================
# Training retrieval arguments
#===========================================================================
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--retrieval_epochs', type=int, default=100,
					help='Number of training epochs for Stage-1 retrieval models (e.g., neural LRURec).')
parser.add_argument('--batch_size_retrieval', type=int, default=512,
					help='Batch size for Stage-1 retrieval model training.')
parser.add_argument('--num_workers_retrieval', type=int, default=4,
					help='Number of DataLoader workers for Stage-1 retrieval model.')
parser.add_argument('--retrieval_patience', type=int, default=10,
					help='Early stopping patience (epochs without val improvement) for Stage-1 retrieval.')
parser.add_argument('--retrieval_lr', type=float, default=1e-3,
					help='Learning rate for all retrieval methods (default: 1e-4). Note: VBPR paper uses 5e-4, MMGCN may need 1e-3 for better performance.')

#===========================================================================
# MMGCN-specific hyperparameters
#===========================================================================
parser.add_argument('--mmgcn_dim_x', type=int, default=64,
					help='MMGCN embedding dimension (default: 64, recommended: 128-256 for better performance)')
parser.add_argument('--mmgcn_num_layer', type=int, default=2,
					help='MMGCN number of GCN layers (default: 2, max: 3)')
parser.add_argument('--mmgcn_concate', action='store_true', default=False,
					help='MMGCN concat features (default: False, recommended: True for better performance)')
parser.add_argument('--mmgcn_reg_weight', type=float, default=1e-4,
					help='MMGCN regularization weight (default: 1e-4, range: 1e-5 to 1e-3)')
parser.add_argument('--mmgcn_aggr_mode', type=str, default='add',
					choices=['add', 'mean', 'max'],
					help='MMGCN aggregation mode (default: add, usually best for recommendation)')


#===========================================================================
# VBPR-specific hyperparameters
#===========================================================================
parser.add_argument('--vbpr_dim_gamma', type=int, default=128,
					help='VBPR dimension of user/item latent factors (default: 20, recommended: 64 for better performance)')
parser.add_argument('--vbpr_dim_theta', type=int, default=128,
					help='VBPR dimension of visual projection (default: 20, recommended: 64 for better performance)')
parser.add_argument('--vbpr_lambda_reg', type=float, default=0.01,
					help='VBPR regularization weight (default: 0.01, range: 1e-4 to 0.1)')
parser.add_argument('--vbpr_optimizer', type=str, default='adam',
					choices=['adam', 'sgd'],
					help='VBPR optimizer (default: adam, recommended for better convergence)')

#===========================================================================
# BM3-specific hyperparameters
#===========================================================================
parser.add_argument('--bm3_embed_dim', type=int, default=128,
					help='BM3 embedding dimension (default: 64, recommended: 128-256 for better performance)')
parser.add_argument('--bm3_layers', type=int, default=2,
					help='BM3 number of MLP layers for feature fusion (default: 1, recommended: 2-3 for better performance)')
parser.add_argument('--bm3_dropout', type=float, default=0.05,
					help='BM3 dropout rate (default: 0.1, range: 0.0-0.5, lower dropout may improve performance)')
parser.add_argument('--bm3_reg_weight', type=float, default=1e-4,
					help='BM3 regularization weight (default: 1e-4, range: 1e-5 to 1e-3)')

#===========================================================================
# BERT4Rec-specific hyperparameters
#===========================================================================
parser.add_argument('--bert4rec_hidden_size', type=int, default=256,
					help='BERT4Rec hidden dimension (default: 64, recommended: 256 for better performance)')
parser.add_argument('--bert4rec_num_hidden_layers', type=int, default=2,
					help='BERT4Rec number of transformer layers (default: 2, recommended: 2-4)')
parser.add_argument('--bert4rec_num_attention_heads', type=int, default=4,
					help='BERT4Rec number of attention heads (default: 2, recommended: 4-8 for better performance)')
parser.add_argument('--bert4rec_intermediate_size', type=int, default=1024,
					help='BERT4Rec feed-forward intermediate size (default: 256, recommended: 1024 for better performance)')
parser.add_argument('--bert4rec_max_seq_length', type=int, default=200,
					help='BERT4Rec maximum sequence length (default: 200, increase if users have long histories)')
parser.add_argument('--bert4rec_attention_dropout', type=float, default=0.2,
					help='BERT4Rec attention dropout rate (default: 0.2, range: 0.0-0.5)')
parser.add_argument('--bert4rec_hidden_dropout', type=float, default=0.2,
					help='BERT4Rec hidden dropout rate (default: 0.2, range: 0.0-0.5)')
parser.add_argument('--bert4rec_warmup_steps', type=int, default=1000,
					help='BERT4Rec learning rate warmup steps (default: 100, recommended: 1000-2000 for better convergence)')

#===========================================================================
# Training reranking arguments
#===========================================================================
parser.add_argument('--rerank_epochs', type=int, default=1,
					help='Number of training epochs for rerank models (e.g., BERT4Rec).')
parser.add_argument('--rerank_batch_size', type=int, default=16,
					help='Batch size for rerank model training.')
parser.add_argument('--rerank_lr', type=float, default=5e-4,
					help='Learning rate for rerank models (default: 1e-4).')
parser.add_argument('--rerank_patience', type=int, default=2,
					help='Early stopping patience (epochs without val improvement) for rerank models.')
parser.add_argument('--rerank_eval_candidates', type=int, default=50,
					help='Number of candidates for reranker evaluation and data preparation (default: 20). Used for both validation and pre-generating candidates in data_prepare.py.')
parser.add_argument('--qwen_max_candidates', type=int, default=50,
					help='Maximum number of candidates for Qwen reranker during inference (default: 20). If None, uses retrieval_top_k from pipeline config.')
parser.add_argument('--qwen_max_seq_length', type=int, default=2048,
					help='Maximum sequence length for Qwen LLM models (default: 2048). Increase for longer prompts (e.g., 4096 for multimodal with images).')
parser.add_argument('--vip5_max_candidates', type=int, default=50,
					help='Maximum number of candidates for VIP5 Direct Task training (default: 100). Number of negatives + 1 positive = total candidates.')
parser.add_argument('--vip5_max_text_length', type=int, default=128,
					help='Maximum text sequence length for VIP5 (default: 128, increase for longer prompts)')
parser.add_argument('--retrieval_eval_mode', type=str, default='full_ranking',
					choices=['full_ranking', 'candidate_list'],
					help='Evaluation mode for retrieval models: full_ranking (evaluate on all items) or candidate_list (evaluate only on pre-generated candidates, default: full_ranking).')
parser.add_argument('--qwen_mode', type=str, default='text_only',
					choices=['text_only', 'caption', 'VIU'],
					help='Prompt mode for Qwen reranker: text_only (description only), caption, VIU')
parser.add_argument('--qwen_model', type=str, default='qwen3-0.6b',
					help='Model for Qwen reranker. Can be: qwen3-0.6b, qwen3-2bvl, qwen3-1.7b, qwen3-4b, or any HuggingFace model name (e.g., Qwen/Qwen2.5-0.5B-Instruct)')
parser.add_argument('--qwen_max_history', type=int, default=5,
					help='Maximum number of items in user history to use for Qwen reranker prompts (default: 5). History will be truncated to the last N items if longer.')
parser.add_argument('--qwen_gradient_accumulation_steps', type=int, default=1,
					help='Gradient accumulation steps for Qwen LLM training (default: 2). Effective batch size = batch_size * gradient_accumulation_steps.')
parser.add_argument('--qwen_warmup_steps', type=int, default=10,
					help='Number of warmup steps for Qwen LLM training (default: 20).')
parser.add_argument('--qwen_lora_r', type=int, default=8,
					help='LoRA rank (r) for Qwen LLM fine-tuning (default: 8). Higher values = more parameters but better capacity.')
parser.add_argument('--qwen_lora_alpha', type=int, default=16,
					help='LoRA alpha for Qwen LLM fine-tuning (default: 16). Usually set to 2 * lora_r.')
parser.add_argument('--qwen_lora_dropout', type=float, default=0.05,
					help='LoRA dropout rate for Qwen LLM fine-tuning (default: 0.05).')
parser.add_argument('--qwen_verbose', type=int, default=1,
					choices=[0, 1, 2],
					help='Verbosity level for Qwen reranker: 0 (minimal), 1 (normal), 2 (verbose/debug, default: 1).')
parser.add_argument('--qwen_disable_progress_bar', action='store_true',
					help='Disable progress bars during Qwen reranker evaluation (default: False). Progress bars are shown by default.')
parser.add_argument('--qwen_temperature', type=float, default=1.0,
					help='Temperature for probability scaling in Qwen reranker (default: 1.0). Temperature < 1.0 makes distribution sharper, > 1.0 makes it smoother. Only affects probability extraction, not generation.')
parser.add_argument('--rerank_action', type=str, default='train',
					choices=['train', 'eval'],
					help='Action for rerank: train (train model) or eval (load pretrained model and evaluate only, default: train). When eval, pass model path to --qwen_model and Unsloth will automatically load the adapter.')
# Legacy: Keep qwen3vl_mode for backward compatibility (will be removed in future)
parser.add_argument('--qwen3vl_mode', type=str, default=None,
					choices=['caption', 'VIU', 'viu_small'],
					help='[DEPRECATED] Use --qwen_mode instead. This argument is kept for backward compatibility only.')

parser.add_argument('--max_text_length', type=int, default=256,
					help='Maximum text length in characters for item metadata (default: 512, range: 256-512). Text will be truncated from the end if longer.')

#===========================================================================
# Script-specific arguments (not used by config, but added to avoid "unrecognized arguments" errors)
#===========================================================================
parser.add_argument('--retrieval_method', type=str, default=None, help='Retrieval method (used by train_retrieval.py)')
parser.add_argument('--mode', type=str, default=None, help='Training mode (used by train_rerank_standalone.py)')
arg = parser.parse_args()

# Set RAW_DATASET_ROOT_FOLDER based on data_path argument
# If data_path is provided, use it; otherwise use default "data" folder
RAW_DATASET_ROOT_FOLDER = arg.data_path if arg.data_path is not None else 'data'


