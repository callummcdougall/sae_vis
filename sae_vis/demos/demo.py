import os
import sys

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gc
from pathlib import Path

from datasets import load_dataset
from IPython import get_ipython
from sae_lens import SAE, HookedSAETransformer

ipython = get_ipython()
ipython.run_line_magic("load_ext", "autoreload")  # type: ignore
ipython.run_line_magic("autoreload", "2")  # type: ignore

DEMOS_PATH = Path(__file__).parent
SAE_VIS_PATH = DEMOS_PATH.parent.parent
if str(SAE_VIS_PATH) not in sys.path:
    sys.path.insert(0, str(SAE_VIS_PATH))

import torch
from huggingface_hub import hf_hub_download

from sae_vis.data_config_classes import SaeVisConfig, SaeVisLayoutConfig
from sae_vis.data_storing_fns import SaeVisData
from sae_vis.model_fns import (
    load_demo_model_saes_and_data,
    load_othello_vocab,
)
from sae_vis.utils_fns import get_device

device = get_device()
torch.set_grad_enabled(False)
assert torch.cuda.is_available()

# * [1/5] Feature-centric, vanilla

# > Load model

SEQ_LEN = 128
DATASET_PATH = "NeelNanda/c4-code-20k"
MODEL_NAME = "gelu-1l"
HOOK_NAME = "blocks.0.mlp.hook_post"

# For this, it's just a 1L model from Neel's library
sae, sae_B, model, all_tokens = load_demo_model_saes_and_data(SEQ_LEN, str(device))

# prompt = "'key': 'True'}),\n            'public': ('"
# prompt = "'}),\n            'is_active': ('"
# answer = "django"
# utils.test_prompt(prompt, answer, model, prepend_space_to_answer=False)
# with model.saes([sae]):
#     utils.test_prompt(prompt, answer, model, prepend_space_to_answer=False)

# > Create vis

sae_vis_data = SaeVisData.create(
    sae=sae,
    sae_B=sae_B,
    model=model,
    tokens=all_tokens[:2048],  # 8192
    cfg=SaeVisConfig(
        features=range(64),  # 256
    ),
    verbose=True,
)
sae_vis_data.save_feature_centric_vis(filename=str(DEMOS_PATH / "_demo_feature_vis.html"), feature=8)

# * [2/5] Feature-centric, custom

from sae_vis.data_config_classes import (
    ActsHistogramConfig,
    Column,
    FeatureTablesConfig,
    SeqMultiGroupConfig,
)

layout = SaeVisLayoutConfig(
    columns=[
        Column(
            SeqMultiGroupConfig(buffer=None, n_quantiles=0, top_acts_group_size=30),
            width=1000,
        ),
        Column(ActsHistogramConfig(), FeatureTablesConfig(n_rows=5), width=500),
    ],
    height=1000,
)
layout.help()

sae_vis_data_custom = SaeVisData.create(
    sae=sae,
    sae_B=sae_B,
    model=model,
    tokens=all_tokens[:4096, :48],  # 4096
    cfg=SaeVisConfig(
        features=range(256),  # 256
        feature_centric_layout=layout,
    ),
    verbose=True,
)
sae_vis_data_custom.save_feature_centric_vis(
    filename=str(DEMOS_PATH / "_demo_feature_vis_custom.html"), feature=8, verbose=True
)

# * [3/5] Prompt-centric
# This is done on top of a pre-existing SaeVisData object (because most of the data-gathering is already done!)

prompt = "'first_name': ('django.db.models.fields"
seq_pos = model.tokenizer.tokenize(prompt).index("Ġ('")  # type: ignore
metric = "act_quantile"

sae_vis_data.save_prompt_centric_vis(
    filename=str(DEMOS_PATH / "_demo_prompt_vis.html"),
    prompt=prompt,
    seq_pos=seq_pos,
    metric=metric,
)

# * [4/5] Othello

# > Load model

hf_repo_id = "callummcdougall/arena-demos-othellogpt"
sae_id = "blocks.5.mlp.hook_post-v1"
model_name = "othello-gpt"

othellogpt: HookedSAETransformer = HookedSAETransformer.from_pretrained(model_name)

othellogpt_sae = SAE.from_pretrained(release=hf_repo_id, sae_id=sae_id, device=str(device))[0]


def hf_othello_load(filename: str):
    path = hf_hub_download(repo_id=hf_repo_id, filename=filename)
    return torch.load(path, weights_only=True, map_location=device)


othello_tokens = hf_othello_load("tokens.pt")[:5000]
othello_target_logits = hf_othello_load("target_logits.pt")[:5000]
othello_linear_probes = hf_othello_load("linear_probes.pt")
print(f"{othello_tokens.shape=}")

# Get live features
_, cache = othellogpt.run_with_cache_with_saes(
    othello_tokens[:5_000],
    saes=[othellogpt_sae],
    names_filter=lambda x: "hook_sae" in x,
)
acts = cache[f"{othellogpt_sae.cfg.hook_name}.hook_sae_acts_post"]
alive_feats = (acts[:, 5:-5].flatten(0, 1) > 1e-8).any(dim=0).nonzero().squeeze().tolist()
print(f"Alive features: {len(alive_feats)}/{othellogpt_sae.cfg.d_sae}\n")
del cache

# > Create vis

sae_vis_data = SaeVisData.create(
    sae=othellogpt_sae,
    model=othellogpt,  # type: ignore
    linear_probes=[
        ("input", "theirs vs mine", othello_linear_probes["theirs vs mine"]),
        ("output", "theirs vs mine", othello_linear_probes["theirs vs mine"]),
        ("input", "empty", othello_linear_probes["empty"]),
        ("output", "empty", othello_linear_probes["empty"]),
    ],
    tokens=othello_tokens,
    target_logits=othello_target_logits,
    cfg=SaeVisConfig(
        features=alive_feats[:16],
        seqpos_slice=(5, -5),
        feature_centric_layout=SaeVisLayoutConfig.default_othello_layout(),
    ),
    vocab_dict=load_othello_vocab(),
    verbose=True,
    clear_memory_between_batches=True,
)
sae_vis_data.save_feature_centric_vis(
    filename=str(SAE_VIS_PATH / "sae_vis/demos/_demo_othello_vis.html"),
    verbose=True,
)


# * [5/5] Attention models

# > Load model

attn_model: HookedSAETransformer = HookedSAETransformer.from_pretrained("attn-only-2l-demo")
hf_repo_id = "callummcdougall/arena-demos-attn2l"
# sae_id = "blocks.1.attn.hook_z"
sae_id = "blocks.0.attn.hook_z-v2"
attn_sae = SAE.from_pretrained(release=hf_repo_id, sae_id=sae_id, device=str(device))[0]

original_dataset = load_dataset(attn_sae.cfg.dataset_path, split="train", streaming=True, trust_remote_code=True)
batch_size = 4096
seq_len = 256
seq_list = [x["input_ids"][: seq_len - 1] for (_, x) in zip(range(batch_size), original_dataset)]
tokens = torch.tensor(seq_list, device=device)
assert attn_model.tokenizer is not None
bos_token = torch.tensor([attn_model.tokenizer.bos_token_id for _ in range(batch_size)], device=device)
tokens = torch.cat([bos_token.unsqueeze(1), tokens], dim=1)
assert tokens.shape == (batch_size, seq_len)

# Get live features
_, cache = attn_model.run_with_cache_with_saes(
    tokens[:512],
    saes=[attn_sae],
    names_filter=(post_acts_hook := f"{attn_sae.cfg.hook_name}.hook_sae_acts_post"),
    stop_at_layer=attn_sae.cfg.hook_layer + 1,
)
acts = cache[post_acts_hook]
alive_feats = (acts.flatten(0, 1) > 1e-8).any(dim=0).nonzero().squeeze().tolist()
print(f"Alive features: {len(alive_feats)}/{attn_sae.cfg.d_sae}\n")
del cache
torch.cuda.empty_cache()
gc.collect()

# > Create vis

sae_vis_data = SaeVisData.create(
    sae=attn_sae,
    model=attn_model,
    tokens=tokens,
    cfg=SaeVisConfig(
        features=alive_feats[:32],
    ),
    verbose=True,
    clear_memory_between_batches=True,
)
sae_vis_data.save_feature_centric_vis(filename=str(DEMOS_PATH / "_demo_feature_vis_attn2l-v3.html"))
