# %%

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from sae_lens import SAE, HookedSAETransformer

from sae_vis.data_config_classes import (
    ActsHistogramConfig,
    Column,
    FeatureTablesConfig,
    SaeVisConfig,
    SaeVisLayoutConfig,
    SeqMultiGroupConfig,
)
from sae_vis.data_storing_fns import SaeVisData
from sae_vis.model_fns import load_demo_model_saes_and_data, load_othello_vocab

torch.set_grad_enabled(False)
assert torch.cuda.is_available(), "This demo won't run well on CPU"
device = torch.device("cuda")

# %%

# * [1/5] Feature-centric, vanilla
# First we run setup code, for loading in the model & SAE as well as the dataset

SEQ_LEN = 128
DATASET_PATH = "NeelNanda/c4-code-20k"
MODEL_NAME = "gelu-1l"
HOOK_NAME = "blocks.0.mlp.hook_post"
sae, sae_B, model, all_tokens = load_demo_model_saes_and_data(SEQ_LEN, str(device))

sae_vis_data = SaeVisData.create(
    sae=sae,
    sae_B=sae_B,
    model=model,
    tokens=all_tokens[:4096],  # 8192
    cfg=SaeVisConfig(features=range(128)),  # 256
    verbose=True,
)
sae_vis_data.save_feature_centric_vis(filename="demo_feature_vis.html", feature=8)

# %%

# * [2/5] Feature-centric, custom

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
layout.help()  # this prints out what our vis will look like

sae_vis_data_custom = SaeVisData.create(
    sae=sae,
    sae_B=sae_B,
    model=model,
    tokens=all_tokens[:4096, :48],
    cfg=SaeVisConfig(
        features=range(256), feature_centric_layout=layout
    ),  # specify layout here
    verbose=True,
)
sae_vis_data_custom.save_feature_centric_vis(
    filename="demo_feature_vis_custom.html", feature=8, verbose=True
)

# %%

# * [3/5] Prompt-centric
# This is done on top of a pre-existing SaeVisData object (because most of the data-gathering is already done!)

prompt = "'first_name': ('django.db.models.fields"

sae_vis_data.save_prompt_centric_vis(
    filename="demo_prompt_vis.html",
    prompt=prompt,
    seq_pos=model.tokenizer.tokenize(prompt).index("Ġ('"),  # type: ignore
    metric="act_quantile",
)

# %%

# * [4/5] Othello
# First we run setup code, for loading in the model & SAE as well as the dataset

hf_repo_id = "callummcdougall/arena-demos-othellogpt"
othellogpt = HookedSAETransformer.from_pretrained("othello-gpt")
othellogpt_sae = SAE.from_pretrained(
    release=hf_repo_id, sae_id="blocks.5.mlp.hook_post-v1", device=str(device)
)[0]

batch_size = 5000
batch_size_for_computing_alive_feats = 1000

hf_othello_load = lambda x: torch.load(
    hf_hub_download(repo_id=hf_repo_id, filename=x), device, weights_only=True
)
othello_tokens = hf_othello_load("tokens.pt")[:batch_size]
othello_target_logits = hf_othello_load("target_logits.pt")[:batch_size]
othello_linear_probes = hf_othello_load("linear_probes.pt")
print(f"Tokens loaded from Othello dataset: {othello_tokens.shape=}")

_, cache = othellogpt.run_with_cache_with_saes(
    othello_tokens[:batch_size_for_computing_alive_feats],
    saes=[othellogpt_sae],
    names_filter=lambda x: "hook_sae" in x,
)
othello_acts = cache[f"{othellogpt_sae.cfg.hook_name}.hook_sae_acts_post"]
othello_alive_feats = (
    (othello_acts[:, 5:-5].flatten(0, 1) > 1e-8).any(dim=0).nonzero().squeeze().tolist()
)
print(f"Alive features: {len(othello_alive_feats)}/{othellogpt_sae.cfg.d_sae}\n")

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
        features=othello_alive_feats[:16],
        seqpos_slice=(5, -5),
        feature_centric_layout=SaeVisLayoutConfig.default_othello_layout(),
    ),
    vocab_dict=load_othello_vocab(),
    verbose=True,
    clear_memory_between_batches=True,
)
sae_vis_data.save_feature_centric_vis(filename="demo_othello_vis.html", verbose=True)

# %%

# * [5/5] Attention models
# First we run setup code, for loading in the model & SAE as well as the dataset

attn_model = HookedSAETransformer.from_pretrained("attn-only-2l-demo")
attn_sae = SAE.from_pretrained(
    "callummcdougall/arena-demos-attn2l", "blocks.0.attn.hook_z-v2", str(device)
)[0]

batch_size = 4096
batch_size_for_computing_alive_feats = 512
seq_len = 256

original_dataset = load_dataset(
    attn_sae.cfg.dataset_path, split="train", streaming=True, trust_remote_code=True
)
attn_tokens_as_list = [
    x["input_ids"][: seq_len - 1] for (_, x) in zip(range(batch_size), original_dataset)
]
attn_tokens = torch.tensor(attn_tokens_as_list, device=device)
bos_token = torch.tensor(
    [attn_model.tokenizer.bos_token_id for _ in range(batch_size)], device=device
)  # type: ignore
attn_tokens = torch.cat([bos_token.unsqueeze(1), attn_tokens], dim=1)
print(f"Tokens loaded for attn-only model: {attn_tokens.shape=}")

_, cache = attn_model.run_with_cache_with_saes(
    attn_tokens[:batch_size_for_computing_alive_feats],
    saes=[attn_sae],
    names_filter=(post_acts_hook := f"{attn_sae.cfg.hook_name}.hook_sae_acts_post"),
    stop_at_layer=attn_sae.cfg.hook_layer + 1,
)
attn_acts = cache[post_acts_hook]
attn_alive_feats = (
    (attn_acts.flatten(0, 1) > 1e-8).any(dim=0).nonzero().squeeze().tolist()
)
print(f"Alive features: {len(attn_alive_feats)}/{attn_sae.cfg.d_sae}\n")

sae_vis_data = SaeVisData.create(
    sae=attn_sae,
    model=attn_model,
    tokens=attn_tokens,
    cfg=SaeVisConfig(features=attn_alive_feats[:32]),
    verbose=True,
    clear_memory_between_batches=True,
)
sae_vis_data.save_feature_centric_vis(filename="demo_feature_vis_attn2l-v3.html")

# %%
