import torch
from torch import Tensor
import pprint
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from jaxtyping import Float, Int, Bool
import transformers
from transformer_lens import utils, HookedTransformer
from transformer_lens.hook_points import HookPoint
import re


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
    d_in: int = 2048
    enc_dtype: str = "fp32"
    remove_rare_dir: bool = False
    model_batch_size: int = 64

    def __post_init__(self):
        '''Using kwargs, so that we can pass in a dict of parameters which might be
        a superset of the above, without error.'''
        self.buffer_size = self.batch_size * self.buffer_mult
        self.buffer_batches = self.buffer_size // self.seq_len
        self.dtype = DTYPES[self.enc_dtype]
        self.d_hidden = self.d_in * self.dict_mult



class AutoEncoder(nn.Module):
    def __init__(self, cfg: AutoEncoderConfig):
        super().__init__()

        self.cfg = cfg
        torch.manual_seed(cfg.seed)

        # W_enc has shape (d_in, d_encoder), where d_encoder is a multiple of d_in (cause dictionary learning; overcomplete basis)
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg.d_in, cfg.d_hidden, dtype=cfg.dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg.d_hidden, cfg.d_in, dtype=cfg.dtype)))
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_hidden, dtype=cfg.dtype))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_in, dtype=cfg.dtype))
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

        cfg: dict = utils.download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version}_cfg.json")
        # There are some unnecessary params in cfg cause they're defined in post_init for config dataclass; we remove them
        cfg.pop("buffer_batches", None)
        cfg.pop("buffer_size", None)
        # Also, we're calling it d_in rather than d_mlp
        cfg["d_in"] = cfg.pop("d_mlp")

        if verbose: pprint.pprint(cfg)
        cfg = AutoEncoderConfig(**cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(utils.download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version}.pt", force_is_torch=True))
        return self

    def __repr__(self):
        return f"AutoEncoder(d_in={self.cfg.d_in}, dict_mult={self.cfg.dict_mult})"




# # ==============================================================
# # ! TRANSFORMERS
# # This returns the activations & resid_pre as well (optionally)
# # ==============================================================


class TransformerLensWrapper(nn.Module):
    '''
    This class wraps around & extends the TransformerLens model, so that we can make sure things like the forward
    function have a standardized signature.
    '''
    def __init__(self, model: HookedTransformer, hook_point: str):
        super().__init__()
        assert hook_point in model.hook_dict, f"Error: hook_point={hook_point!r} must be in model.hook_dict"
        self.model = model
        self.hook_point = hook_point

        # Get the layer (so we can do the early stopping in our forward pass)
        assert re.match(r'blocks\.\d+\.', hook_point), "Error: expecting hook_point to be 'blocks.{layer}.{...}'"
        self.hook_layer = int(re.search(r'blocks\.(\d+)\.', hook_point).group(1))

        # Get the hook names for the residual stream (final) and residual stream (immediately after hook_point)
        self.hook_point_resid = utils.get_act_name("resid_post", self.hook_layer)
        self.hook_point_resid_final = utils.get_act_name("resid_post", self.model.cfg.n_layers-1)
        assert self.hook_point_resid in model.hook_dict
        assert self.hook_point_resid_final in model.hook_dict

    
    def forward(self, tokens: Int[Tensor, "batch seq"], return_logits: bool = True):
        '''
        Forward pass on tokens. Returns (logits, residual, activation) by default, or just (residual, activation) if
        return_logits=False.
        '''

        # Run with hook functions to store the activations & final value of residual stream
        # If return_logits is False, then we compute the last residual stream value but not the logits
        output = self.model.run_with_hooks(
            tokens, 
            # stop_at_layer = (None if return_logits else self.hook_layer),
            fwd_hooks = [
                (self.hook_point, self.hook_fn_store_act),
                (self.hook_point_resid_final, self.hook_fn_store_act)
            ]
        )

        # The hook functions work by storing data in model's hook context, so we pop them back out
        activation = self.model.hook_dict[self.hook_point].ctx.pop("activation")
        residual = self.model.hook_dict[self.hook_point_resid_final].ctx.pop("activation")

        if return_logits:
            return output, residual, activation
        return residual, activation

    def hook_fn_store_act(self, activation: torch.Tensor, hook: HookPoint):
        hook.ctx["activation"] = activation

    @property
    def tokenizer(self):
        return self.model.tokenizer

    @property
    def W_U(self):
        return self.model.W_U

    @property
    def W_out(self):
        return self.model.W_out



def to_resid_dir(dir: Float[Tensor, "feats d_in"], model: TransformerLensWrapper):
    """
    Takes a direction (eg. in the post-ReLU MLP activations) and returns the corresponding dir in the residual stream.

    Args:
        dir: 
            The direction in the activations, i.e. shape (feats, d_in) where d_in could be d_model, d_mlp, etc.
        model:
            The model, which should be a HookedTransformerWrapper or similar.
    """
    # If this SAE was trained on the residual stream, then we don't need to do anything
    if "resid" in model.hook_point:
        return dir
    
    # If it was trained on the MLP layer, then we apply the W_out map
    elif ("pre" in model.hook_point) or ("post" in model.hook_point):
        return dir @ model.W_out[model.hook_layer]
    
    # Others not yet supported
    else:
        raise NotImplementedError("The hook your SAE was trained on isn't yet supported")
