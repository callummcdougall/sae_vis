import gc
import os
import sys

from IPython import get_ipython

ipython = get_ipython()
ipython.run_line_magic("load_ext", "autoreload")  # type: ignore
ipython.run_line_magic("autoreload", "2")  # type: ignore

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
if (SAE_VIS_PATH := "C:/Users/calsm/Documents/AI Alignment/sae-vis") not in sys.path:
    sys.path.insert(0, SAE_VIS_PATH)

DEMOS_PATH = SAE_VIS_PATH + "/sae_vis/demos/"

import torch

from sae_vis.data_config_classes import SaeVisConfig, SaeVisLayoutConfig
from sae_vis.data_storing_fns import SaeVisData
from sae_vis.model_fns import (
    load_demo_model_saes_and_data,
    load_demo_othello_model_saes_and_data,
    load_othello_linear_probes,
    load_othello_vocab,
)
from sae_vis.utils_fns import get_device

device = get_device()
torch.set_grad_enabled(False)

# ! Load data / models

SEQ_LEN = 128
DATASET_PATH = "NeelNanda/c4-code-20k"
MODEL_NAME = "gelu-1l"
HOOK_NAME = "blocks.0.mlp.hook_post"

# For this, it's just a 1L model from Neel's library
sae, sae_B, model, all_tokens = load_demo_model_saes_and_data(SEQ_LEN, str(device))

# For data, we're using a custom filtered dataset (all games with illegal moves removed - there was a mistake in original dataset here I think?!)
othello_sae, othello_model, othello_tokens, othello_target_logits = (
    load_demo_othello_model_saes_and_data(str(device))
)
othello_linear_probes = load_othello_linear_probes(device=device)

# * [1/4] Feature-centric, vanilla

# > Vanilla

torch.cuda.empty_cache()
gc.collect()

# TODO - the table time estimates are all underestimates, why?
sae_vis_data = SaeVisData.create(
    sae=sae,
    sae_B=sae_B,
    model=model,
    tokens=all_tokens[:8192],  # type: ignore
    cfg=SaeVisConfig(
        hook_point=HOOK_NAME,
        features=range(256),
    ),
    verbose=True,
)
sae_vis_data.save_feature_centric_vis(
    filename=f"{DEMOS_PATH}_demo_feature_vis.html", feature=8
)

# * [2/4] Feature-centric, custom

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
    tokens=all_tokens[:4096, :48],  # type: ignore
    cfg=SaeVisConfig(
        hook_point=HOOK_NAME,
        features=range(256),
        feature_centric_layout=layout,
    ),
    verbose=True,
)
sae_vis_data_custom.save_feature_centric_vis(
    filename=f"{DEMOS_PATH}_demo_feature_vis_custom.html", feature=8, verbose=True
)

# * [3/4] Prompt-centric
# This is done on top of a pre-existing SaeVisData object (because most of the work is already done for us!)

prompt = "'first_name': ('django.db.models.fields"
seq_pos = model.tokenizer.tokenize(prompt).index("Ġ('")  # type: ignore
metric = "act_quantile"

sae_vis_data.save_prompt_centric_vis(
    filename=f"{DEMOS_PATH}_demo_prompt_vis.html",
    prompt=prompt,
    seq_pos=seq_pos,
    metric=metric,
)

# * [4/4] Othello

# Get live features
# > tokens, from cache
_, cache = othello_model.run_with_cache_with_saes(
    othello_tokens[:500], saes=[othello_sae], names_filter=lambda x: "hook_sae" in x
)
acts = cache[f"{othello_sae.cfg.hook_name}.hook_sae_acts_post"]
alive_feats = (
    (acts[:, 5:-5].flatten(0, 1) > 1e-8).any(dim=0).nonzero().squeeze().tolist()
)
print(f"Alive features: {len(alive_feats)}/{othello_sae.cfg.d_sae}\n")
del cache

# Create vis
sae_vis_data = SaeVisData.create(
    sae=othello_sae,
    model=othello_model,  # type: ignore
    linear_probes_input=othello_linear_probes,
    tokens=othello_tokens,
    target_logits=othello_target_logits,
    cfg=SaeVisConfig(
        hook_point=othello_sae.cfg.hook_name,
        features=alive_feats[:256],
        seqpos_slice=(5, -5),
        feature_centric_layout=SaeVisLayoutConfig.default_othello_layout(boards=True),
    ),
    vocab_dict=load_othello_vocab(),
    verbose=True,
)
sae_vis_data.save_feature_centric_vis(
    filename=f"{SAE_VIS_PATH}/sae_vis/demos/_demo_othello_vis.html",
    verbose=True,
)
