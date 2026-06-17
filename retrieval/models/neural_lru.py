"""Lightweight neural LRU-style sequential recommender.

This is an independent re-implementation inspired by the LRURec model in
`LlamaRec/model/lru.py`, but lives entirely under `retrieval/` and does not
import anything from `LlamaRec/`.

Design goals:
- Keep the core idea: item embeddings + LRU-style sequence layer + feed-forward.
- Expose a simple interface: `NeuralLRURec(args)` with the fields we need.
- Stay reasonably light so it can train on a modest machine (small hidden size,
  few blocks, configurable via args).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NeuralLRUConfig:
	num_items: int
	hidden_units: int = 64
	num_blocks: int = 2
	dropout: float = 0.1
	attn_dropout: float = 0.1


class NeuralLRURec(nn.Module):
	"""Top-level model: embedding + LRU blocks + output bias.

	Input:  LongTensor [B, L] of item ids.
	Output: FloatTensor [B, L, num_items+1] of logits per position.
	"""

	def __init__(self, cfg: NeuralLRUConfig) -> None:
		super().__init__()
		self.cfg = cfg
		self.embedding = _LRUEmbedding(cfg)
		self.model = _LRUStack(cfg)
		self._truncated_normal_init()

	def _truncated_normal_init(self, mean: float = 0.0, std: float = 0.02, lower: float = -0.04, upper: float = 0.04) -> None:
		"""Truncated normal init for all parameters (except layer norms)."""
		with torch.no_grad():
			l = (1.0 + math.erf(((lower - mean) / std) / math.sqrt(2.0))) / 2.0
			u = (1.0 + math.erf(((upper - mean) / std) / math.sqrt(2.0))) / 2.0

			for name, p in self.named_parameters():
				if "layer_norm" in name:
					continue
				if torch.is_complex(p):
					p.real.uniform_(2 * l - 1, 2 * u - 1)
					p.imag.uniform_(2 * l - 1, 2 * u - 1)
					p.real.erfinv_()
					p.imag.erfinv_()
					p.real.mul_(std * math.sqrt(2.0))
					p.imag.mul_(std * math.sqrt(2.0))
					p.real.add_(mean)
					p.imag.add_(mean)
				else:
					p.uniform_(2 * l - 1, 2 * u - 1)
					p.erfinv_()
					p.mul_(std * math.sqrt(2.0))
					p.add_(mean)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: [B, L]
		x, mask = self.embedding(x)
		logits = self.model(x, self.embedding.token.weight, mask)
		return logits


class _LRUEmbedding(nn.Module):
	def __init__(self, cfg: NeuralLRUConfig) -> None:
		super().__init__()
		vocab_size = cfg.num_items + 1
		embed_size = cfg.hidden_units
		self.token = nn.Embedding(vocab_size, embed_size)
		self.layer_norm = nn.LayerNorm(embed_size)
		self.embed_dropout = nn.Dropout(cfg.dropout)

	def _get_mask(self, x: torch.Tensor) -> torch.Tensor:
		return x > 0

	def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		mask = self._get_mask(x)
		x = self.token(x)
		return self.layer_norm(self.embed_dropout(x)), mask


class _LRUStack(nn.Module):
	def __init__(self, cfg: NeuralLRUConfig) -> None:
		super().__init__()
		self.cfg = cfg
		self.hidden_size = cfg.hidden_units
		layers = cfg.num_blocks
		self.blocks = nn.ModuleList([_LRUBlock(cfg) for _ in range(layers)])
		self.bias = nn.Parameter(torch.zeros(cfg.num_items + 1))

	def forward(self, x: torch.Tensor, embedding_weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
		# Left-pad to power-of-two length for the parallel LRU algorithm.
		seq_len = x.size(1)
		log2_L = int(np.ceil(np.log2(seq_len)))
		pad_len = 2 ** log2_L - seq_len
		if pad_len > 0:
			x = F.pad(x, (0, 0, pad_len, 0, 0, 0))
			mask = F.pad(mask, (pad_len, 0, 0, 0))

		for block in self.blocks:
			x = block(x, mask)

		# Remove left padding and compute scores against item embedding.
		x = x[:, -seq_len:]
		scores = torch.matmul(x, embedding_weight.t()) + self.bias
		return scores


class _LRUBlock(nn.Module):
	def __init__(self, cfg: NeuralLRUConfig) -> None:
		super().__init__()
		self.lru_layer = _LRULayer(d_model=cfg.hidden_units, dropout=cfg.attn_dropout)
		self.ffn = _PositionwiseFeedForward(d_model=cfg.hidden_units, d_ff=cfg.hidden_units * 4, dropout=cfg.dropout)

	def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
		x = self.lru_layer(x, mask)
		x = self.ffn(x)
		return x


class _LRULayer(nn.Module):
	"""Parallel LRU layer as in the original LRURec paper/code.

	This is a compact adaptation using complex weights, similar in spirit to the
	original but implemented locally so we don't depend on `LlamaRec/`.
	"""

	def __init__(
		self,
		d_model: int,
		dropout: float = 0.1,
		use_bias: bool = True,
		r_min: float = 0.8,
		r_max: float = 0.99,
	) -> None:
		super().__init__()
		self.embed_size = d_model
		self.hidden_size = 2 * d_model
		self.use_bias = use_bias

		# Initialize nu, theta, gamma following the original idea.
		u1 = torch.rand(self.hidden_size)
		u2 = torch.rand(self.hidden_size)
		nu_log = torch.log(-0.5 * torch.log(u1 * (r_max**2 - r_min**2) + r_min**2))
		theta_log = torch.log(u2 * torch.tensor(np.pi * 2.0))
		diag_lambda = torch.exp(torch.complex(-torch.exp(nu_log), torch.exp(theta_log)))
		gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))
		self.params_log = nn.Parameter(torch.vstack((nu_log, theta_log, gamma_log)))

		self.in_proj = nn.Linear(self.embed_size, self.hidden_size, bias=use_bias).to(torch.cfloat)
		self.out_proj = nn.Linear(self.hidden_size, self.embed_size, bias=use_bias).to(torch.cfloat)
		self.out_vector: nn.Module = nn.Identity()
		self.dropout = nn.Dropout(dropout)
		self.layer_norm = nn.LayerNorm(self.embed_size)

	def _lru_parallel(
		self,
		i: int,
		h: torch.Tensor,
		lamb: torch.Tensor,
		mask: torch.Tensor,
		B: int,
		L: int,
		D: int,
	) -> tuple[torch.Tensor, torch.Tensor]:
		# Parallel prefix-style recursion over the sequence length.
		segment_len = 2 ** i
		h = h.reshape(B * L // segment_len, segment_len, D)
		mask_ = mask.reshape(B * L // segment_len, segment_len)
		h1, h2 = h[:, : segment_len // 2], h[:, segment_len // 2 :]

		if i > 1:
			lamb = torch.cat((lamb, lamb * lamb[-1]), dim=0)
		h2 = h2 + lamb * h1[:, -1:] * mask_[:, segment_len // 2 - 1 : segment_len // 2].unsqueeze(-1)
		h = torch.cat([h1, h2], dim=1)
		return h, lamb

	def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
		# Compute bu and lambda parameters.
		nu, theta, gamma = torch.exp(self.params_log).split((1, 1, 1))
		lamb = torch.exp(torch.complex(-nu, theta))
		h = self.in_proj(x.to(torch.cfloat)) * gamma

		log2_L = int(np.ceil(np.log2(h.size(1))))
		B, L, D = h.size(0), h.size(1), h.size(2)
		for i in range(log2_L):
			h, lamb = self._lru_parallel(i + 1, h, lamb, mask, B, L, D)

		x_out = self.dropout(self.out_proj(h).real) + self.out_vector(x)
		return self.layer_norm(x_out)


class _PositionwiseFeedForward(nn.Module):
	def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
		super().__init__()
		self.w_1 = nn.Linear(d_model, d_ff)
		self.w_2 = nn.Linear(d_ff, d_model)
		self.activation = nn.GELU()
		self.dropout = nn.Dropout(dropout)
		self.layer_norm = nn.LayerNorm(d_model)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x_ = self.dropout(self.activation(self.w_1(x)))
		return self.layer_norm(self.dropout(self.w_2(x_)) + x)
