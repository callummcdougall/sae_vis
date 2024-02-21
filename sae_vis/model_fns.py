import torch
from torch import Tensor
import pprint
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import einops
from jaxtyping import Float
import transformers
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from typing import Any, Dict, Callable
import inspect
from huggingface_hub import hf_hub_download
import json

def select_compatible_kwargs(
    kwargs_dict: Dict[str, Any], callable: Callable
) -> Dict[str, Any]:
    """Return a dict with the elements kwargs_dict that are parameters of callable"""
    return {
        k: v
        for k, v in kwargs_dict.items()
        if k in inspect.getfullargspec(callable).args
    }


def download_file_from_hf(
    repo_name,
    file_name,
    subfolder = ".",
    cache_dir = transformers.TRANSFORMERS_CACHE,
    force_is_torch = False,
    **kwargs,
):
    """
    Helper function to download files from the HuggingFace Hub, from subfolder/file_name in repo_name, saving locally to
	cache_dir and returning the loaded file (if a json or Torch object) and the file path otherwise.

    If it's a Torch file without the ".pth" extension, set force_is_torch=True to load it as a Torch object.
    """
    file_path = hf_hub_download(
        repo_id = repo_name,
        filename = file_name,
        subfolder = subfolder,
        cache_dir = cache_dir,
        **select_compatible_kwargs(kwargs, hf_hub_download),
    )

    if file_path.endswith(".pth") or force_is_torch:
        return torch.load(file_path, map_location="cpu")
    elif file_path.endswith(".json"):
        return json.load(open(file_path, "r"))
    else:
        print("File type not supported:", file_path.split(".")[-1])
        return file_path



DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

@dataclass
class AutoEncoderConfig:
    '''Class for storing configuration parameters for the autoencoder'''
    seed: int = 42
    batch_size: int = 32
    buffer_mult: int = 384
    epochs: int = 10
    lr: float = 1e-3
    num_tokens: int = int(2e9)
    l1_coeff: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.99
    dict_mult: int = 8
    seq_len: int = 128
    d_mlp: int = 2048
    enc_dtype: str = "fp32"
    remove_rare_dir: bool = False
    model_batch_size: int = 64

    def __post_init__(self):
        '''Using kwargs, so that we can pass in a dict of parameters which might be
        a superset of the above, without error.'''
        self.buffer_size = self.batch_size * self.buffer_mult
        self.buffer_batches = self.buffer_size // self.seq_len
        self.dtype = DTYPES[self.enc_dtype]
        self.d_hidden = self.d_mlp * self.dict_mult



class AutoEncoder(nn.Module):
    def __init__(self, cfg: AutoEncoderConfig):
        super().__init__()

        self.cfg = cfg
        torch.manual_seed(cfg.seed)

        # W_enc has shape (d_mlp, d_encoder), where d_encoder is a multiple of d_mlp (cause dictionary learning; overcomplete basis)
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg.d_mlp, cfg.d_hidden, dtype=cfg.dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg.d_hidden, cfg.d_mlp, dtype=cfg.dtype)))
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_hidden, dtype=cfg.dtype))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_mlp, dtype=cfg.dtype))
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.to("cuda")

    def forward(self, x: torch.Tensor):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
        l1_loss = self.cfg.l1_coeff * (acts.float().abs().sum())
        loss = l2_loss + l1_loss
        return loss, x_reconstruct, acts, l2_loss, l1_loss

    @torch.no_grad()
    def remove_parallel_component_of_grads(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj

    @classmethod
    def load_from_hf(cls, version, verbose=False):
        """
        Loads the saved autoencoder from HuggingFace. 

        Note, this is a classmethod, because we'll be using it as `auto_encoder = AutoEncoder.load_from_hf("run1")`

        Version is expected to be an int, or "run1" or "run2"

        version 25 is the final checkpoint of the first autoencoder run,
        version 47 is the final checkpoint of the second autoencoder run.
        """

        assert version in ["run1", "run2"]
        version = 25 if version=="run1" else 47

        cfg: dict = download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version}_cfg.json")
        # There are some unnecessary params in cfg cause they're defined in post_init for config dataclass; we remove them
        cfg.pop("buffer_batches", None)
        cfg.pop("buffer_size", None)

        if verbose: pprint.pprint(cfg)
        cfg = AutoEncoderConfig(**cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version}.pt", force_is_torch=True))
        return self

    def __repr__(self):
        return f"AutoEncoder(d_mlp={self.cfg.d_mlp}, dict_mult={self.cfg.dict_mult})"




# ==============================================================
# ! TRANSFORMERS
# This returns the activations & resid_pre as well (optionally)
# ==============================================================


@dataclass
class DemoTransformerConfig:
	d_head: int
	d_mlp: int
	d_model: int
	d_vocab: int
	device: torch.device
	dtype: torch.dtype
	n_ctx: int
	n_heads: int
	n_layers: int
	normalization_type: str	
	act_fn: str = "gelu"
	layer_norm_eps: float = 1e-5
	

class LayerNorm(nn.Module):
	def __init__(self, cfg: DemoTransformerConfig):
		super().__init__()
		self.cfg = cfg
		self.w = nn.Parameter(torch.ones(cfg.d_model))
		self.b = nn.Parameter(torch.zeros(cfg.d_model))

	def forward(self, residual):
		residual_mean = residual.mean(dim=-1, keepdim=True)
		residual_std = (residual.var(dim=-1, keepdim=True, unbiased=False) + self.cfg.layer_norm_eps).sqrt()

		residual = (residual - residual_mean) / residual_std
		return residual * self.w + self.b


class LayerNormPre(nn.Module):
	def __init__(self, cfg: DemoTransformerConfig):
		super().__init__()
		self.cfg = cfg

	def forward(self, residual):
		residual_mean = residual.mean(dim=-1, keepdim=True)
		residual_std = (residual.var(dim=-1, keepdim=True, unbiased=False) + self.cfg.layer_norm_eps).sqrt()

		return (residual - residual_mean) / residual_std


class Embed(nn.Module):
	def __init__(self, cfg: DemoTransformerConfig):
		super().__init__()
		self.cfg = cfg
		self.W_E = nn.Parameter(torch.empty((cfg.d_vocab, cfg.d_model)))

	def forward(self, tokens):
		return self.W_E[tokens]


class PosEmbed(nn.Module):
	def __init__(self, cfg: DemoTransformerConfig):
		super().__init__()
		self.cfg = cfg
		self.W_pos = nn.Parameter(torch.empty((cfg.n_ctx, cfg.d_model)))

	def forward(self, tokens):
		batch, seq_len = tokens.shape
		return einops.repeat(self.W_pos[:seq_len], "seq d_model -> batch seq d_model", batch=batch)


class Attention(nn.Module):
	IGNORE: Float[Tensor, ""]

	def __init__(self, cfg: DemoTransformerConfig):
		super().__init__()
		self.cfg = cfg
		self.W_Q = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
		self.W_K = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
		self.W_V = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
		self.W_O = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
		self.b_Q = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
		self.b_K = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
		self.b_V = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
		self.b_O = nn.Parameter(torch.zeros((cfg.d_model)))
		self.register_buffer("IGNORE", torch.tensor(-1e5, dtype=torch.float32, device=cfg.device))

	def forward(self, normalized_resid_pre):
		# Calculate query, key and value vectors
		q = einops.einsum(
			normalized_resid_pre, self.W_Q,
			"batch posn d_model, nheads d_model d_head -> batch posn nheads d_head", 
		) + self.b_Q
		k = einops.einsum(
			normalized_resid_pre, self.W_K,
			"batch posn d_model, nheads d_model d_head -> batch posn nheads d_head", 
		) + self.b_K
		v = einops.einsum(
			normalized_resid_pre, self.W_V,
			"batch posn d_model, nheads d_model d_head -> batch posn nheads d_head", 
		) + self.b_V

		# Calculate attention scores, then scale and mask, and apply softmax to get probabilities
		attn_scores = einops.einsum(
			q, k,
			"batch posn_Q nheads d_head, batch posn_K nheads d_head -> batch nheads posn_Q posn_K", 
		)
		attn_scores_masked = self.apply_causal_mask(attn_scores / self.cfg.d_head ** 0.5)
		attn_pattern = attn_scores_masked.softmax(-1)

		# Take weighted sum of value vectors, according to attention probabilities
		z = einops.einsum(
			v, attn_pattern,
			"batch posn_K nheads d_head, batch nheads posn_Q posn_K -> batch posn_Q nheads d_head", 
		)

		# Calculate output (by applying matrix W_O and summing over heads, then adding bias b_O)
		attn_out = einops.einsum(
			z, self.W_O,
			"batch posn_Q nheads d_head, nheads d_head d_model -> batch posn_Q d_model", 
		) + self.b_O

		return attn_out

	def apply_causal_mask(self, attn_scores):
		'''
		Applies a causal mask to attention scores, and returns masked scores.
		'''
		# Define a mask that is True for all positions we want to set probabilities to zero for
		all_ones = torch.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device)
		mask = torch.triu(all_ones, diagonal=1).bool()
		# Apply the mask to attention scores, then return the masked scores
		attn_scores.masked_fill_(mask, self.IGNORE)
		return attn_scores


class MLP(nn.Module):
	def __init__(self, cfg: DemoTransformerConfig):
		super().__init__()
		self.cfg = cfg
		self.W_in = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp)))
		self.W_out = nn.Parameter(torch.empty((cfg.d_mlp, cfg.d_model)))
		self.b_in = nn.Parameter(torch.zeros((cfg.d_mlp)))
		self.b_out = nn.Parameter(torch.zeros((cfg.d_model)))
		self.act_fn = {"gelu": F.gelu, "relu": F.relu}[cfg.act_fn]

	def forward(self, normalized_resid_mid):
		pre = einops.einsum(
			normalized_resid_mid, self.W_in,
			"batch position d_model, d_model d_mlp -> batch position d_mlp", 
		) + self.b_in
		post = self.act_fn(pre)
		mlp_out = einops.einsum(
			post, self.W_out,
			"batch position d_mlp, d_mlp d_model -> batch position d_model", 
		) + self.b_out
		return mlp_out, post


class TransformerBlock(nn.Module):
	def __init__(self, cfg: DemoTransformerConfig):
		super().__init__()
		self.cfg = cfg
		ln = {"LN": LayerNorm, "LNPre": LayerNormPre}[cfg.normalization_type]
		self.ln1 = ln(cfg)
		self.attn = Attention(cfg)
		self.ln2 = ln(cfg)
		self.mlp = MLP(cfg)

	def forward(self, resid_pre):
		resid_mid = self.attn(self.ln1(resid_pre)) + resid_pre
		mlp_out, post = self.mlp(self.ln2(resid_mid))
		resid_post = mlp_out + resid_mid
		return resid_post, post
		

class Unembed(nn.Module):
	def __init__(self, cfg: DemoTransformerConfig):
		super().__init__()
		self.cfg = cfg
		self.W_U = nn.Parameter(torch.empty((cfg.d_model, cfg.d_vocab)))
		self.b_U = nn.Parameter(torch.zeros((cfg.d_vocab), requires_grad=False))

	def forward(
		self, normalized_resid_final: Float[Tensor, "batch position d_model"]
	) -> Float[Tensor, "batch position d_vocab"]:
		return einops.einsum(
			normalized_resid_final, self.W_U,
			"batch posn d_model, d_model d_vocab -> batch posn d_vocab",
		) + self.b_U


class DemoTransformer(nn.Module):

	def __init__(self, cfg: DemoTransformerConfig, tokenizer: PreTrainedTokenizerFast):
		super().__init__()
		self.cfg = cfg
		self.embed = Embed(cfg)
		self.pos_embed = PosEmbed(cfg)
		self.blocks: list[TransformerBlock] = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
		ln = {"LN": LayerNorm, "LNPre": LayerNormPre}[cfg.normalization_type]
		self.ln_final = ln(cfg)
		self.unembed = Unembed(cfg)
		self.tokenizer = tokenizer
		self.to(cfg.device)

	def forward(self, tokens, return_logits: bool = True):
		residual = self.embed(tokens) + self.pos_embed(tokens)
		for block in self.blocks:
			residual, post = block(residual)
			if not(return_logits):
				return residual, post
		logits = self.unembed(self.ln_final(residual))
		return logits, residual, post

	@property
	def W_U(self):
		return self.unembed.W_U

	@property
	def W_out(self):
		return torch.stack([block.mlp.W_out for block in self.blocks])