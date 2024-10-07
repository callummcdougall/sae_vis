import itertools
import math
import time
from collections import defaultdict
from typing import Literal

import einops
import numpy as np
import torch
from eindex import eindex
from jaxtyping import Float, Int
from rich import print as rprint
from rich.table import Table
from sae_lens import SAE, HookedSAETransformer
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import ActivationCache, HookedTransformer, utils

from sae_vis.data_config_classes import (
    SaeVisConfig,
    SaeVisLayoutConfig,
    SeqMultiGroupConfig,
)
from sae_vis.data_storing_fns import (
    ActsHistogramData,
    FeatureTablesData,
    LogitsHistogramData,
    LogitsTableData,
    ProbeLogitsTableData,
    SaeVisData,
    SeqGroupData,
    SeqMultiGroupData,
    SequenceData,
)
from sae_vis.model_fns import resid_final_pre_layernorm_to_logits, to_resid_dir
from sae_vis.utils_fns import (
    METRIC_TITLES,
    FeatureStatistics,
    RollingCorrCoef,
    TopK,
    VocabType,
    cross_entropy_loss,
    get_device,
    index_with_buffer,
    k_largest_indices,
    random_range_indices,
)

Arr = np.ndarray

device = get_device()


@torch.inference_mode()
def parse_feature_data(
    model: HookedSAETransformer,
    cfg: SaeVisConfig,
    sae: SAE,
    sae_B: SAE | None,
    tokens: Int[Tensor, "batch seq"],
    feature_indices: int | list[int],
    feature_resid_dir: Float[Tensor, "feats d_model"],
    feature_resid_dir_input: Float[Tensor, "feats d"],
    cache: ActivationCache,
    feature_out_dir: Float[Tensor, "feats d_out"] | None = None,
    corrcoef_neurons: RollingCorrCoef | None = None,
    corrcoef_sae: RollingCorrCoef | None = None,
    corrcoef_sae_B: RollingCorrCoef | None = None,
    linear_probes: list[tuple[Literal["input", "output"], str, Float[Tensor, "d_model d_vocab_out"]]] = [],
    target_logits: Float[Tensor, "batch seq d_vocab_out"] | None = None,
    vocab_dict: dict[VocabType, dict[int, str]] | None = None,
    progress: list[tqdm] | None = None,
) -> tuple[SaeVisData, dict[str, float]]:
    """Convert generic activation data into a SaeVisData object, which can be used to create the feature-centric vis.

    Returns:
        sae_vis_data: SaeVisData
            Containing data for creating each feature visualization, as well as data for rank-ordering the feature
            visualizations when it comes time to make the prompt-centric view (the `feature_act_quantiles` attribute).

        time_logs: dict[str, float]
            A dictionary containing the time taken for each step of the computation. This is optionally printed at the
            end of the `get_feature_data` function, if `verbose` is set to True.

    # TODO - there's redundant docstrings here, each argument should only have a docstring in one function (the outer-most one)

    # TODO - this function was originally written so that there could be a fn that didn't use the saes and the models. But I don't know if that's really necessary any more, and maybe there's more funcs than we need?
    """
    acts_post_hook_name = f"{sae.cfg.hook_name}.hook_sae_acts_post"
    all_feat_acts = cache[acts_post_hook_name]

    time_logs = {
        "(2) Getting data for sequences": 0.0,
        "(3) Getting data for non-sequence components": 0.0,
    }
    t0 = time.monotonic()

    if target_logits is not None:
        assert (
            target_logits.shape[-1] < 1000
        ), "Not recommended to use target logits with a very large vocab size (this is intended for toy models e.g. OthelloGPT)"
        target_logits = target_logits.to(device)

    # Make feature_indices a list, for convenience
    if isinstance(feature_indices, int):
        feature_indices = [feature_indices]

    assert feature_resid_dir.shape[0] == len(
        feature_indices
    ), f"Num features in feature_resid_dir ({feature_resid_dir.shape[0]}) doesn't match {len(feature_indices)=}"

    if feature_out_dir is not None:
        assert feature_out_dir.shape[0] == len(
            feature_indices
        ), f"Num features in feature_out_dir ({feature_resid_dir.shape[0]}) doesn't match {len(feature_indices)=}"

    # ! Data setup code (defining the main objects we'll eventually return)
    feature_data_dict = {feat: {} for feat in feature_indices}

    # We're using `cfg.feature_centric_layout` to figure out what data we'll need to calculate during this function
    layout = cfg.feature_centric_layout
    assert isinstance(
        layout, SaeVisLayoutConfig
    ), f"Error: cfg.feature_centric_layout must be a SaeVisLayoutConfig object, got {type(layout)}"

    # ! Feature tables (i.e. left hand of vis)

    if layout.feature_tables_cfg is not None and feature_out_dir is not None:
        # Store kwargs (makes it easier to turn the tables on and off individually)
        feature_tables_data = {}

        # Table 1: neuron alignment, based on decoder weights
        if layout.feature_tables_cfg.neuron_alignment_table:
            top3_neurons_aligned = TopK(tensor=feature_out_dir, k=layout.feature_tables_cfg.n_rows, largest=True)
            feature_out_l1_norm = feature_out_dir.abs().sum(dim=-1, keepdim=True)
            pct_of_l1: Arr = np.absolute(top3_neurons_aligned.values) / utils.to_numpy(feature_out_l1_norm)
            feature_tables_data.update(
                neuron_alignment_indices=top3_neurons_aligned.indices.tolist(),
                neuron_alignment_values=top3_neurons_aligned.values.tolist(),
                neuron_alignment_l1=pct_of_l1.tolist(),
            )

        # Table 2: neurons correlated with this feature, based on their activations
        if isinstance(corrcoef_neurons, RollingCorrCoef):
            neuron_indices, neuron_pearson, neuron_cossim = corrcoef_neurons.topk_pearson(
                k=layout.feature_tables_cfg.n_rows,
            )
            feature_tables_data.update(
                correlated_neurons_indices=neuron_indices,
                correlated_neurons_pearson=neuron_pearson,
                correlated_neurons_cossim=neuron_cossim,
            )

        # Table 3: primary sae features correlated with this feature, based on their activations
        if isinstance(corrcoef_sae, RollingCorrCoef):
            enc_indices, enc_pearson, enc_cossim = corrcoef_sae.topk_pearson(
                k=layout.feature_tables_cfg.n_rows,
            )
            feature_tables_data.update(
                correlated_features_indices=enc_indices,
                correlated_features_pearson=enc_pearson,
                correlated_features_cossim=enc_cossim,
            )

        # Table 4: sae-B features correlated with this feature, based on their activations
        if isinstance(corrcoef_sae_B, RollingCorrCoef):
            encB_indices, encB_pearson, encB_cossim = corrcoef_sae_B.topk_pearson(
                k=layout.feature_tables_cfg.n_rows,
            )
            feature_tables_data.update(
                correlated_b_features_indices=encB_indices,
                correlated_b_features_pearson=encB_pearson,
                correlated_b_features_cossim=encB_cossim,
            )

        # Add all this data to the list of FeatureTablesData objects
        for i, feat in enumerate(feature_indices):
            feature_data_dict[feat]["featureTables"] = FeatureTablesData(
                **{k: v[i] for k, v in feature_tables_data.items()}
            )

    # ! Histograms & logit tables & optional othello probes (i.e. middle column of vis)

    # Get the logits of all features (i.e. the directions this feature writes to the logit output)
    logits = einops.einsum(feature_resid_dir, model.W_U, "feats d_model, d_model d_vocab -> feats d_vocab")
    probe_names_and_values = [
        (
            f"PROBE {name!r}, {mode.upper()} SPACE",
            einops.einsum(
                feature_resid_dir if mode == "output" else feature_resid_dir_input,
                probe,
                "feats d_model, d_model d_vocab_out -> feats d_vocab_out",
            ),
        )
        for mode, name, probe in linear_probes
    ]

    if any(x is not None for x in [layout.act_hist_cfg, layout.logits_hist_cfg, layout.logits_table_cfg]):
        for i, feat in enumerate(feature_indices):
            # Get logits histogram data (no title)
            if layout.logits_hist_cfg is not None:
                feature_data_dict[feat]["logitsHistogram"] = LogitsHistogramData.from_data(
                    data=logits[i],
                    n_bins=layout.logits_hist_cfg.n_bins,
                    tickmode="5 ticks",
                    title=None,
                )

            # Get data for feature activations histogram (including the title!)
            if layout.act_hist_cfg is not None:
                feat_acts = all_feat_acts[..., i]
                nonzero_feat_acts = feat_acts[feat_acts > 0]
                frac_nonzero = nonzero_feat_acts.numel() / feat_acts.numel()
                feature_data_dict[feat]["actsHistogram"] = ActsHistogramData.from_data(
                    data=nonzero_feat_acts,
                    n_bins=layout.act_hist_cfg.n_bins,
                    tickmode="5 ticks",
                    title=f"ACTIVATIONS<br><span style='color:#666;font-weight:normal'>DENSITY = {frac_nonzero:.3%}</span>",
                )

            # Get logits table data
            if layout.logits_table_cfg is not None:
                feature_data_dict[feat]["logitsTable"] = LogitsTableData.from_data(
                    logits[i], k=layout.logits_table_cfg.n_rows
                )

            # Optionally get probes data
            if layout.probe_logits_table_cfg is not None:
                feature_data_dict[feat]["probeLogitsTables"] = ProbeLogitsTableData.from_data(
                    probe_names_and_values=probe_names_and_values,
                    k=layout.probe_logits_table_cfg.n_rows,
                )

    # ! Get stats (including quantiles, which will be useful for the prompt-centric visualisation)
    feature_stats = FeatureStatistics.create(data=einops.rearrange(all_feat_acts, "b s feats -> feats (b s)"))

    time_logs["(3) Getting data for non-sequence components"] = time.monotonic() - t0
    t0 = time.monotonic()

    # ! Sequences (i.e. right hand of vis)

    if layout.seq_cfg is not None:
        for i, feat in enumerate(feature_indices):
            # Add this feature's sequence data to the list
            feature_data_dict[feat]["seqMultiGroup"] = get_sequences_data(
                tokens=tokens,
                feat_idx=i,
                feat_logits=logits[i],
                cache=cache,
                feature_resid_dir=feature_resid_dir[i],
                model=model,
                sae=sae,
                target_logits=target_logits,
                seq_cfg=layout.seq_cfg,
            )
            # Update the 2nd progress bar (fwd passes & getting sequence data dominates the runtime of these computations)
            if progress is not None:
                progress[1].update(1)

    time_logs["(2) Getting data for sequences"] = time.monotonic() - t0
    t0 = time.monotonic()

    # ! Return the output, as a dict of FeatureData items
    sae_vis_data = SaeVisData(
        feature_data_dict=feature_data_dict,
        prompt_data_dict={},
        feature_stats=feature_stats,
        cfg=cfg,
        model=model,
        sae=sae,
        sae_B=sae_B,
        vocab_dict=vocab_dict,
    )
    return sae_vis_data, time_logs


@torch.inference_mode()
def _get_feature_data(
    sae: SAE,
    model: HookedSAETransformer,
    tokens: Int[Tensor, "batch seq"],
    feature_indices: int | list[int],
    cfg: SaeVisConfig,
    sae_B: SAE | None,
    linear_probes: list[tuple[Literal["input", "output"], str, Float[Tensor, "d_model d_vocab_out"]]] = [],
    target_logits: Float[Tensor, "batch seq d_vocab_out"] | None = None,
    vocab_dict: dict[VocabType, dict[int, str]] | None = None,
    progress: list[tqdm] | None = None,
    clear_memory_between_batches: bool = False,
) -> tuple[SaeVisData, dict[str, float]]:
    """
    Gets data that will be used to create the sequences in the feature-centric HTML visualisation.

    Note - this function isn't called directly by the user, it actually gets called by the `get_feature_data` function
    which does exactly the same thing except it also batches this computation by features (in accordance with the
    arguments `features` and `minibatch_size_features` from the SaeVisConfig object).

    Returns:
        sae_vis_data: SaeVisData
            Containing data for creating each feature visualization, as well as data for
            rank-ordering the feature visualizations when it comes time to make the prompt-centric
            view (the `feature_act_quantiles` attribute).

        time_log: dict[str, float]
            A dictionary containing the time taken for each step of the computation. This is
            optionally printed at the end of the `get_feature_data` function, if `verbose=True`.
    """
    # ! Boring setup code
    time_logs = {"(1) Forward passes to gather model activations": 0.0}
    batch_size, seq_len = tokens.shape

    # Make feature_indices a list, for convenience
    if isinstance(feature_indices, int):
        feature_indices = [feature_indices]

    # Get tokens into minibatches, for the fwd pass. Same for target logits, if using them
    token_minibatches_list = (tokens,) if cfg.minibatch_size_tokens is None else tokens.split(cfg.minibatch_size_tokens)
    token_minibatches = [tok.to(device) for tok in token_minibatches_list]

    # ! Data setup code (defining the main objects we'll eventually return, for each of 5 possible vis components)

    # Create tensors to store the feature activations & final values of the residual stream
    seqpos_slice = slice(*cfg.seqpos_slice)
    resid_final_hook_name = utils.get_act_name("resid_post", model.cfg.n_layers - 1)
    acts_post_hook_name = f"{sae.cfg.hook_name}.hook_sae_acts_post"
    sae_input_hook_name = f"{sae.cfg.hook_name}.hook_sae_input"
    v_hook_name = utils.get_act_name("v", sae.cfg.hook_layer)
    pattern_hook_name = utils.get_act_name("pattern", sae.cfg.hook_layer)
    cache_dict = {
        resid_final_hook_name: torch.zeros(batch_size, seq_len, model.cfg.d_model),
        acts_post_hook_name: torch.zeros(batch_size, seq_len, len(feature_indices)),
        # sae_input_hook_name: torch.zeros(batch_size, seq_len, sae.cfg.d_in),
    }
    using_dfa = (
        (cfg.feature_centric_layout.seq_cfg is not None)
        and (cfg.feature_centric_layout.seq_cfg.dfa_for_attn_saes)
        and (sae.cfg.hook_name.endswith("_z"))
    )
    if using_dfa:
        cache_dict[v_hook_name] = torch.zeros(batch_size, seq_len, model.cfg.n_heads, model.cfg.d_head)
        cache_dict[pattern_hook_name] = torch.zeros(batch_size, model.cfg.n_heads, seq_len, seq_len)

    # Create objects to store the data for computing rolling stats
    sae_input_is_privileged = any([sae.cfg.hook_name.endswith(x) for x in ["mlp.hook_pre", "mlp.hook_post"]])
    corrcoef_neurons = RollingCorrCoef() if sae_input_is_privileged else None
    corrcoef_sae = RollingCorrCoef(indices=feature_indices, with_self=True)
    corrcoef_sae_B = RollingCorrCoef() if sae_B is not None else None

    # Get sae & decoder directions
    feature_out_dir = sae.W_dec[feature_indices]  # [feats d_sae]
    feature_resid_dir = to_resid_dir(feature_out_dir, sae, model)  # [feats d_model]
    feature_in_dir = sae.W_enc.T[feature_indices]  # [feats d_in]
    feature_resid_dir_input = to_resid_dir(feature_in_dir, sae, model, input=True)  # [feats d_model]

    # ! Compute & concatenate together all feature activations & post-activation function values

    start = 0
    for minibatch in token_minibatches:
        # Fwd pass, get model activations
        t0 = time.monotonic()

        sae.use_error_term = True
        _, cache = model.run_with_cache_with_saes(
            minibatch,
            saes=[sae],
            stop_at_layer=model.cfg.n_layers,
            names_filter=list(set(cache_dict.keys()) | {sae_input_hook_name}),
        )
        sae.use_error_term = False
        feat_acts_all = cache[acts_post_hook_name]  # [batch seq d_sae]
        feat_acts = cache[acts_post_hook_name][..., feature_indices]  # [batch seq d_sae]
        sae_input = cache[sae_input_hook_name]  # [batch seq d_in]

        time_logs["(1) Forward passes to gather model activations"] += time.monotonic() - t0

        if clear_memory_between_batches:
            t0 = time.monotonic()
            torch.cuda.empty_cache()
            time_logs["(1.5) Clearing memory"] = time_logs.get("(1.5) Clearing memory", 0.0) + time.monotonic() - t0

        # Update the CorrCoef object between feature activation & neurons
        if corrcoef_neurons is not None:
            corrcoef_neurons.update(feat_acts, sae_input)

        # Update the CorrCoef object between pairwise feature activations
        if corrcoef_sae is not None:
            corrcoef_sae.update(feat_acts, feat_acts_all)

        # Update the CorrCoef object between feature activation & sae-B features
        if corrcoef_sae_B is not None:
            assert sae_B is not None
            feat_acts_B = sae_B.encode(sae_input)  # [batch seq d_sae]
            corrcoef_sae_B.update(feat_acts, feat_acts_B)

        # Put these values into the tensors
        batch_slice = slice(start, start + len(minibatch))
        cache_dict[resid_final_hook_name][batch_slice, seqpos_slice] = cache[resid_final_hook_name][
            :, seqpos_slice
        ].cpu()
        cache_dict[acts_post_hook_name][batch_slice, seqpos_slice] = feat_acts[:, seqpos_slice].cpu()
        if using_dfa:
            cache_dict[v_hook_name][batch_slice, seqpos_slice] = cache[v_hook_name][:, seqpos_slice].cpu()
            cache_dict[pattern_hook_name][batch_slice, :, seqpos_slice, seqpos_slice] = cache[pattern_hook_name][
                :, seqpos_slice, seqpos_slice
            ].cpu()

        # Update the 1st progress bar (fwd passes & getting sequence data dominates the runtime of these computations)
        if progress is not None:
            progress[0].update(1)

        start += len(minibatch)

    torch.cuda.empty_cache()

    cache = ActivationCache(cache_dict, model=model)

    # ! Use the data we've collected to make a MultiFeatureData object
    sae_vis_data, _time_logs = parse_feature_data(
        model=model,
        cfg=cfg,
        sae=sae,
        sae_B=sae_B,
        tokens=tokens,
        feature_indices=feature_indices,
        feature_resid_dir=feature_resid_dir,
        feature_resid_dir_input=feature_resid_dir_input,
        cache=cache,
        feature_out_dir=feature_out_dir,
        corrcoef_neurons=corrcoef_neurons,
        corrcoef_sae=corrcoef_sae,
        corrcoef_sae_B=corrcoef_sae_B,
        linear_probes=linear_probes,
        target_logits=target_logits,
        vocab_dict=vocab_dict,
        progress=progress,
    )

    assert (
        set(time_logs.keys()) & set(_time_logs.keys()) == set()
    ), f"Invalid keys: {set(time_logs.keys()) & set(_time_logs.keys())} should have zero overlap"

    time_logs.update(_time_logs)

    return sae_vis_data, time_logs


@torch.inference_mode()
def get_feature_data(
    sae: SAE,
    model: HookedTransformer,
    tokens: Int[Tensor, "batch seq"],
    cfg: SaeVisConfig,
    sae_B: SAE | None = None,
    linear_probes: list[tuple[Literal["input", "output"], str, Float[Tensor, "d_model d_vocab_out"]]] = [],
    target_logits: Float[Tensor, "batch seq d_vocab_out"] | None = None,
    vocab_dict: dict[VocabType, dict[int, str]] | None = None,
    verbose: bool = False,
    clear_memory_between_batches: bool = False,
) -> SaeVisData:
    """
    This is the main function which users will run to generate the feature visualization data. It
    batches this computation over features, in accordance with the arguments in the SaeVisConfig
    object (we don't want to compute all the features at once, since might give OOMs).

    See the `_get_feature_data` function for an explanation of the arguments, as well as a more
    detailed explanation of what this function is doing.

    The return object is the merged SaeVisData objects returned by the `_get_feature_data` function.
    """
    T0 = time.monotonic()

    # Apply random seed
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    # Create objects to store all the data we'll get from `_get_feature_data`
    sae_vis_data = SaeVisData(
        model=model,
        cfg=cfg,
        sae=sae,
        sae_B=sae_B,
        linear_probes=linear_probes,
        vocab_dict=vocab_dict,
    )
    time_logs = defaultdict(float)

    # Get a feature list (need to deal with the case where `cfg.features` is an int, or None)
    if cfg.features is None:
        assert isinstance(sae.cfg.d_sae, int)
        features_list = list(range(sae.cfg.d_sae))
    elif isinstance(cfg.features, int):
        features_list = [cfg.features]
    else:
        features_list = list(cfg.features)

    # Break up the features into batches
    feature_batches = [x.tolist() for x in torch.tensor(features_list).split(cfg.minibatch_size_features)]
    # Calculate how many minibatches of tokens there will be (for the progress bar)
    n_token_batches = 1 if (cfg.minibatch_size_tokens is None) else math.ceil(len(tokens) / cfg.minibatch_size_tokens)
    # Get the denominator for each of the 2 progress bars
    totals = (n_token_batches * len(feature_batches), len(features_list))

    # Optionally add two progress bars (one for the forward passes, one for getting the sequence data)
    if verbose:
        progress = [
            tqdm(total=totals[0], desc="Forward passes to cache data for vis"),
            tqdm(total=totals[1], desc="Extracting vis data from cached data"),
        ]
    else:
        progress = None

    # If the model is from TransformerLens, we need to apply a wrapper to it for standardization
    assert isinstance(model, HookedSAETransformer)
    assert isinstance(cfg.hook_point, str)

    # For each batch of features: get new data and update global data storage objects
    for features in feature_batches:
        new_feature_data, new_time_logs = _get_feature_data(
            sae=sae,
            model=model,
            tokens=tokens,
            feature_indices=features,
            cfg=cfg,
            sae_B=sae_B,
            linear_probes=linear_probes,
            target_logits=target_logits,
            vocab_dict=vocab_dict,
            progress=progress,
            clear_memory_between_batches=clear_memory_between_batches,
        )
        sae_vis_data.update(new_feature_data)
        for key, value in new_time_logs.items():
            time_logs[key] += value

    # Now exited, make sure the progress bar is at 100%
    if progress is not None:
        for pbar in progress:
            pbar.n = pbar.total

    # If verbose, then print the output
    if verbose:
        time_logs["(?) Unaccounted time"] = time.monotonic() - T0 - sum(time_logs.values())
        total_time = sum(time_logs.values())
        table = Table("Task", "Time", "Pct %")
        for task, duration in time_logs.items():
            table.add_row(task, f"{duration:.2f}s", f"{duration/total_time:.1%}")
        rprint(table)

    return sae_vis_data


@torch.inference_mode()
def get_sequences_data(
    tokens: Int[Tensor, "batch seq"],
    feat_idx: int,  # the index in the batch, not the same as the feature id in the overall SAE
    feat_logits: Float[Tensor, "d_vocab"],
    cache: ActivationCache,
    feature_resid_dir: Float[Tensor, "d_model"],
    model: HookedSAETransformer,
    sae: SAE,
    target_logits: Float[Tensor, "batch seq d_vocab_out"] | None,
    seq_cfg: SeqMultiGroupConfig,
) -> SeqMultiGroupData:
    """
    This function returns the data which is used to create the top activating sequence visualizations. Steps are:

        (1) Get the indices of all example tokens we'll be taking (i.e. the bold tokens in the vis). This includes the
            top activations and all the quantiles.
        (2) Get the token IDs from all these indices.
        (3) Get other values at these positions & the surrounding buffer: feature acts, residual stream values
        (4) Compute the logit & logprob effect of this feature. Use this to also get the top affected tokens by this
            feat (i.e. the vis hoverdata).
        (5) If necessary, get the direct feature attribution info.
        (6) Return all this data as a SeqMultiGroupData object.

    Args:
        tokens:
            The tokens we'll be extracting sequence data from.
        feat_acts:
            The activations of the feature we're interested in, for each token in the batch.
        feat_logits:
            The logit vector for this feature (used to generate histogram, and is needed here for the line-on-hover).
        resid_final:
            The residual stream values before final layernorm, for each token in the batch.
        feature_resid_dir:
            The direction this feature writes to the logit output (i.e. the direction we'll be erasing from resid_final).
        feature_resid_dir_input:
            The input direction (i.e. we dot this with residual stream to get this feature's activation).
        W_U:
            The model's unembedding matrix, which we'll use to get the logits.
        cfg:
            Feature visualization parameters, containing some important params e.g. num sequences per group.

    Returns:
        SeqMultiGroupData
            This is a dataclass which contains a dict of SeqGroupData objects, where each SeqGroupData object
            contains the data for a particular group of sequences (i.e. the top-k, bottom-k, and the quantile groups).
    """
    resid_final_hook_name = utils.get_act_name("resid_post", layer=model.cfg.n_layers - 1)
    acts_post_hook_name = f"{sae.cfg.hook_name}.hook_sae_acts_post"
    # sae_acts_pre_hook_name = f"{sae.cfg.hook_name}.hook_sae_acts_pre"
    v_hook_name = utils.get_act_name("v", layer=sae.cfg.hook_layer)
    pattern_hook_name = utils.get_act_name("pattern", layer=sae.cfg.hook_layer)

    resid_post = cache[resid_final_hook_name]
    feat_acts = cache[acts_post_hook_name][..., feat_idx]

    # ! (1) Find the tokens from each group

    # We define our full buffer going 1 token further back than all the visible tokens. This is because suppose we have
    # visible tokens [0, 1, 2, 3]: we need to compute feature activations from [0, 1, 2, 3] to highlight the tokens, but
    # we also need to compute residual stream & feature acts from [-1, 0, 1, 2] so we can ablate that feature from the
    # residual stream & see loss effect on all visible tokens.
    buffer_to_exclude_from_ex = (0, -1) if seq_cfg.buffer is None else (seq_cfg.buffer[0] + 1, -seq_cfg.buffer[1])
    feat_acts_max = feat_acts[:, buffer_to_exclude_from_ex[0] : buffer_to_exclude_from_ex[1]].max()

    # Get the top-activating tokens
    indices = k_largest_indices(
        feat_acts,
        k=seq_cfg.top_acts_group_size,
        buffer=buffer_to_exclude_from_ex,
    )
    use_dfa = seq_cfg.dfa_for_attn_saes and sae.cfg.hook_name.endswith("hook_z")
    first_title = "TOP ACTIVATIONS (right) & DFA (left)" if use_dfa else "TOP ACTIVATIONS"
    indices_dict = {
        f"{first_title}<br><span style='color:#666;font-weight:normal'>MAX ACT = {feat_acts_max:.3f}</span>": indices
    }

    # Get all possible indices. Note, we need to be able to look 1 back (feature activation on prev token is needed for
    # computing loss effect on this token)
    if seq_cfg.n_quantiles > 0:
        quantiles = torch.linspace(0, feat_acts_max.item(), seq_cfg.n_quantiles + 1)
        quantiles[0] = 1e-6
        n_active = (feat_acts > 1e-6).float().sum()
        for i in range(seq_cfg.n_quantiles - 1, -1, -1):
            lower, upper = quantiles[i : i + 2].tolist()
            pct = ((feat_acts >= lower) & (feat_acts <= upper)).float().sum() / n_active
            indices = random_range_indices(
                feat_acts,
                k=seq_cfg.quantile_group_size,
                bounds=(lower, upper),
                buffer=buffer_to_exclude_from_ex,
            )
            indices_dict[
                f"INTERVAL {lower:.3f} - {upper:.3f}<br><span style='color:#666;font-weight:normal'>CONTAINS {pct:.3%}</span>"
            ] = indices

    # Concat all the indices together (in the next steps we do all groups at once). Shape of this object is [n_ex 2],
    # i.e. the [i, :]-th element are the batch and sequence dimensions for the i-th example.
    indices_ex = torch.concat(list(indices_dict.values())).cpu()
    n_ex = indices_ex.shape[0]

    # ! (2) Get our top tokens, and what we're displaying

    if seq_cfg.buffer is not None:
        # Index into the sequence with a buffer. We get all tokens in buffer, plus the token 1 further back (although we
        # won't be displaying this token in the final vis; it's just for getting effect on loss).
        buffer_full = (seq_cfg.buffer[0] + 1, -seq_cfg.buffer[1])
        token_ids = index_with_buffer(tokens, indices_ex, buffer=buffer_full)
        token_ids_to_display = token_ids[:, 1:]
    else:
        # If we don't specify a sequence, then do all of the seq positions in each seq we pick. In this case, our
        # display seqs are literally just the full sequences from bold tokens.
        buffer_full = None
        token_ids = index_with_buffer(tokens, indices_ex, buffer=None)
        token_ids_to_display = token_ids

    # ! (3) Extract feature activations & residual stream values for those positions
    # Note - the reason we split on cases here is that when computing the buffer, we need activations & loss effect for
    # all tokens (and loss effect requires us to compute activations & resid values 1 token back). But when we aren't
    # computing the buffer, we only need activations for bold token & loss effect on the token after that one.

    if seq_cfg.compute_buffer:
        # Get tokens we'll use to index correct logits (all the visible ones)
        token_ids_for_computing_loss = token_ids[:, 1:]
        # Get feature acts for all sequence positions (all visible used for coloring, 1 back used for loss)
        feat_acts_buf = index_with_buffer(feat_acts, indices_ex, buffer=buffer_full).to(device)
        feat_acts_pre_ablation = feat_acts_buf[:, :-1]
        feat_acts_coloring = feat_acts_buf[:, 1:]
        feat_acts_idx = [None for _ in range(n_ex)]
        # Get residual stream for all sequence positions that come immediately before a visible token (used for loss)
        resid_post = index_with_buffer(resid_post, indices_ex, buffer=buffer_full)[:, :-1].to(device)
    else:
        # Get tokens we'll use to index correct logits (after the bold ones)
        token_ids_for_computing_loss = index_with_buffer(tokens, indices_ex, buffer=0, offset=1).unsqueeze(1).to(device)
        # Get feature acts for just the bold tokens (used for coloring on bold token & loss on token after bold token)
        feat_acts_pre_ablation = index_with_buffer(feat_acts, indices_ex, buffer=0).unsqueeze(-1).to(device)
        feat_acts_coloring = feat_acts_pre_ablation
        feat_acts_idx = indices_ex[:, 1].tolist()  # need to remember which one is the bold one!
        # Get residual stream for just the bold tokens (used for loss on token after bold token)
        resid_post = index_with_buffer(resid_post, indices_ex, buffer=0).unsqueeze(1).to(device)

    # ! (4) Compute the logit effect if this feature is ablated

    # Get this feature's output vector, using an outer product over the feature activations for all tokens
    resid_post_feature_effect = einops.einsum(
        feat_acts_pre_ablation, feature_resid_dir.to(device), "n_ex buf, d_model -> n_ex buf d_model"
    )

    # Contribution to logits is computed without normalization
    logit_contribution = resid_post_feature_effect @ model.W_U
    # Do the ablations, and get difference in logprobs (direct effect)
    new_logits = resid_final_pre_layernorm_to_logits(resid_post - resid_post_feature_effect, model)
    orig_logits = resid_final_pre_layernorm_to_logits(resid_post, model)
    logprobs_contribution = orig_logits.log_softmax(-1) - new_logits.log_softmax(-1)
    orig_prob = orig_logits.softmax(-1)
    new_prob = new_logits.softmax(-1)

    # The TopK function can improve efficiency by masking the features which are zero
    acts_nonzero = feat_acts_pre_ablation.abs() > 1e-5  # shape [batch buf]
    top_logit_contribution = TopK(logprobs_contribution, k=seq_cfg.top_logits_hoverdata, tensor_mask=acts_nonzero)
    bottom_logit_contribution = TopK(
        logprobs_contribution, k=seq_cfg.top_logits_hoverdata, tensor_mask=acts_nonzero, largest=False
    )

    if target_logits is None:
        # loss_cont[b, s] = -logprobs_cont[b, s, token_ids_for_computing_loss[b, s]]
        loss_contribution = -eindex(logprobs_contribution, token_ids_for_computing_loss, "batch seq [batch seq]")
        logit_contribution = eindex(logit_contribution, token_ids_for_computing_loss, "batch seq [batch seq]")
        orig_prob = eindex(orig_prob, token_ids_for_computing_loss, "batch seq [batch seq]")
        new_prob = eindex(new_prob, token_ids_for_computing_loss, "batch seq [batch seq]")
    else:
        assert (
            not seq_cfg.compute_buffer
        ), "Not expecting to compute buffer if using target logits (it's more indexing hassle)"
        target_logits_bold = eindex(target_logits, indices_ex, "[n_ex 0] [n_ex 1] d_vocab").unsqueeze(1)
        loss_orig = cross_entropy_loss(orig_logits, target_logits_bold)
        loss_new = cross_entropy_loss(new_logits, target_logits_bold)
        loss_contribution = loss_orig - loss_new
        logit_contribution = orig_prob = new_prob = None

    # ! (5) If this is an attention SAE, then do DFA
    if use_dfa:
        assert seq_cfg.dfa_buffer is not None
        indices_batch, indices_dest = indices_ex.unbind(dim=-1)
        v = cache[v_hook_name][indices_batch].to(device)  # [k src n_heads d_head]
        pattern = cache[pattern_hook_name][indices_batch, :, indices_dest].to(device)  # [k n_heads src]
        v_weighted = (v * einops.rearrange(pattern, "k n_heads src -> k src n_heads 1")).flatten(-2)  # [k src d_in]
        dfa = v_weighted @ sae.W_enc[:, feat_idx]  # [k src]
        indices_src = dfa.argmax(dim=-1).to(indices_ex.device)  # [k,]
        indices_ex_src = torch.stack([indices_batch, indices_src], dim=-1)
        indices_ex_src[indices_ex_src[:, 1] < seq_cfg.dfa_buffer[0]] = seq_cfg.dfa_buffer[0]
        dfa_buffer = (seq_cfg.dfa_buffer[0], -seq_cfg.dfa_buffer[1])
        dfa_token_ids = index_with_buffer(tokens, indices_ex_src, buffer=dfa_buffer)
        indices_ex_src_for_values = torch.stack([torch.arange(len(indices_src)), indices_src.cpu()], dim=-1)
        dfa_values = index_with_buffer(dfa, indices_ex_src_for_values, buffer=dfa_buffer)
    else:
        dfa_token_ids = dfa_values = indices_ex_src = None

    # ! (6) Store the results in a SeqMultiGroupData object
    sequence_groups_data = []
    group_sizes_cumsum = np.cumsum([0] + [len(indices) for indices in indices_dict.values()]).tolist()
    buffer_range = (
        range(-seq_cfg.buffer[0], seq_cfg.buffer[1] + 1) if seq_cfg.buffer is not None else range(tokens.shape[1])
    )
    dfa_buffer_range = (
        range(-seq_cfg.dfa_buffer[0], seq_cfg.dfa_buffer[1] + 1)
        if seq_cfg.dfa_buffer is not None
        else range(tokens.shape[1])
    )
    for group_idx, group_name in enumerate(indices_dict.keys()):
        seq_data = [
            SequenceData(
                token_ids=token_ids_to_display[i].tolist(),
                token_posns=[f"({indices_ex[i, 0]}, {indices_ex[i, 1] + j})" for j in buffer_range],
                feat_acts=[round(f, 4) for f in feat_acts_coloring[i].tolist()],
                feat_acts_idx=feat_acts_idx[i],
                loss_contribution=loss_contribution[i].tolist(),
                logit_contribution=logit_contribution[i].tolist() if logit_contribution is not None else None,
                orig_prob=orig_prob[i].tolist() if orig_prob is not None else None,
                new_prob=new_prob[i].tolist() if new_prob is not None else None,
                token_logits=feat_logits[token_ids_to_display[i]].tolist(),
                top_token_ids=top_logit_contribution.indices[i].tolist(),
                top_logits=top_logit_contribution.values[i].tolist(),
                bottom_token_ids=bottom_logit_contribution.indices[i].tolist(),
                bottom_logits=bottom_logit_contribution.values[i].tolist(),
                dfa_token_ids=dfa_token_ids[i].tolist() if dfa_token_ids is not None else [],
                dfa_values=dfa_values[i].tolist() if dfa_values is not None else [],
                dfa_token_posns=[f"({indices_ex_src[i, 0]}, {indices_ex_src[i, 1] + j})" for j in dfa_buffer_range]
                if indices_ex_src is not None
                else [],
            )
            for i in range(group_sizes_cumsum[group_idx], group_sizes_cumsum[group_idx + 1])
        ]
        sequence_groups_data.append(SeqGroupData(seq_data, title=group_name))

    return SeqMultiGroupData(sequence_groups_data)


@torch.inference_mode()
def parse_prompt_data(
    tokens: Int[Tensor, "batch seq"],
    str_toks: list[str],
    sae_vis_data: SaeVisData,
    feat_acts: Float[Tensor, "seq feats"],
    feature_resid_dir: Float[Tensor, "feats d_model"],
    resid_post: Float[Tensor, "seq d_model"],
    model: HookedTransformer,
    feature_idx: list[int] | None = None,
    num_top_features: int = 10,
) -> tuple[
    dict[str, list[dict[Literal["feature", "title"], int | str]]],
    dict[int, SeqMultiGroupData],
]:
    """
    Gets data needed to create the sequences in the prompt-centric vis (displaying dashboards for the most relevant
    features on a prompt).

    Returns:
        scores_dict: dict[str, list[dict[Literal["feature", "title"], int | str]]]
            A dictionary mapping keys like "act_quantile|'django' (0)" to a list of tuples, each of
            the form (feature_idx, title).

        prompt_data_dict: dict[int, SeqMultiGroupData]
            A dictionary mapping feature index to the sequence multigroup data for that feature. Note
            that it will only contain data for a single prompt, but we wrap it in this for consistency.
    """
    # Populate our initial dictionaries
    seq_keys = [f"{t!r} ({i})" for i, t in enumerate(str_toks)]
    metrics = ["act_size", "act_quantile", "loss_effect"]
    scores_dict: dict[str, list[dict[Literal["feature", "title"], int | str]]] = {
        f"{metric}|{seq_key}": [] for metric, seq_key in itertools.product(metrics, seq_keys)
    }
    prompt_data_dict: dict[int, SeqMultiGroupData] = {}

    if feature_idx is None:
        feature_idx = list(sae_vis_data.feature_data_dict.keys())
    n_feats = len(feature_idx)
    assert (
        feature_resid_dir.shape[0] == n_feats
    ), f"The number of features in feature_resid_dir ({feature_resid_dir.shape[0]}) does not match the number of feature indices ({n_feats})"

    assert (
        feat_acts.shape[1] == n_feats
    ), f"The number of features in feat_acts ({feat_acts.shape[1]}) does not match the number of feature indices ({n_feats})"

    feats_loss_contribution = torch.empty(size=(n_feats, tokens.shape[1] - 1), device=device)
    # Some logit computations which we only need to do once
    # correct_token_unembeddings = model_wrapped.W_U[:, tokens[0, 1:]] # [d_model seq]
    orig_logits = (resid_post / resid_post.std(dim=-1, keepdim=True)) @ model.W_U  # [seq d_vocab]
    raw_logits = feature_resid_dir @ model.W_U  # [feats d_vocab]

    for i, feat in enumerate(feature_idx):
        # ! Calculate the sequence data for each feature, and store it as FeatureData.prompt_data

        # Get this feature's output vector, using an outer product over the feature activations for all tokens
        resid_post_feature_effect = einops.einsum(feat_acts[:, i], feature_resid_dir[i], "seq, d_model -> seq d_model")

        # Ablate the output vector from the residual stream, and get logits post-ablation
        new_resid_post = resid_post - resid_post_feature_effect
        new_logits = (new_resid_post / new_resid_post.std(dim=-1, keepdim=True)) @ model.W_U

        # Get the top5 & bottom5 changes in logits (don't bother with `efficient_topk` cause it's small)
        contribution_to_logprobs = orig_logits.log_softmax(dim=-1) - new_logits.log_softmax(dim=-1)
        top_contribution_to_logits = TopK(contribution_to_logprobs[:-1], k=5)
        bottom_contribution_to_logits = TopK(contribution_to_logprobs[:-1], k=5, largest=False)

        # Get the change in logprobs (unnormalized) and loss (which is negative of change of logprobs for correct token)
        logit_contribution = einops.einsum(
            resid_post_feature_effect[:-1], model.W_U[:, tokens[0, 1:]], "seq d_model, d_model seq -> seq"
        )
        loss_contribution = eindex(-contribution_to_logprobs[:-1], tokens[0, 1:], "seq [seq]")
        feats_loss_contribution[i, :] = loss_contribution

        # Store the sequence data, wrapped in a multi-group which only has one element
        prompt_data_dict[feat] = SeqMultiGroupData(
            [
                SeqGroupData(
                    [
                        SequenceData(
                            token_ids=tokens.squeeze(0).tolist(),
                            feat_acts=[round(f, 4) for f in feat_acts[:, i].tolist()],
                            loss_contribution=loss_contribution.tolist(),
                            logit_contribution=logit_contribution.tolist(),
                            token_logits=raw_logits[i, tokens.squeeze(0)].tolist(),
                            top_token_ids=top_contribution_to_logits.indices.tolist(),
                            top_logits=top_contribution_to_logits.values.tolist(),
                            bottom_token_ids=bottom_contribution_to_logits.indices.tolist(),
                            bottom_logits=bottom_contribution_to_logits.values.tolist(),
                        )
                    ]
                )
            ],
            is_prompt=True,
        )

    # ! Lastly, return a dictionary mapping each key like 'act_quantile|"django" (0)' to a list of feature indices & scores

    def title(metric: str, feat: int, score_str: str):
        return f"<h3>#{feat}<br>{METRIC_TITLES[metric]} = {score_str}</h3><hr>"

    for seq_pos, seq_key in enumerate(seq_keys):
        # Filter the feature activations, since we only need the ones that are non-zero
        feat_acts_nonzero_filter = utils.to_numpy(feat_acts[seq_pos] > 0)
        feat_acts_nonzero_locations = np.nonzero(feat_acts_nonzero_filter)[0].tolist()
        _feat_acts = feat_acts[seq_pos, feat_acts_nonzero_filter]  # [feats_filtered,]
        _feature_idx = np.array(feature_idx)[feat_acts_nonzero_filter]

        if feat_acts_nonzero_filter.sum() > 0:
            k = min(num_top_features, _feat_acts.numel())

            # Get the top features by activation size. This is just applying a TopK function to feat
            # acts (which were stored by the code before this)
            metric = "act_size"
            topk = TopK(_feat_acts, k=k, largest=True)
            scores_dict[f"{metric}|{seq_key}"] = [
                {"feature": feat, "title": title(metric, feat, score)}
                for feat, score in zip(
                    _feature_idx[topk.indices].tolist(),
                    [f"{v:.3f}" for v in topk.values.tolist()],
                )
            ]

            # Get the top features by activation quantile. We do this using the `feature_act_quantiles` object, which
            # was stored `sae_vis_data`. This quantiles object has a method to return quantiles for a given set of
            # data, as well as the precision (we make the precision higher for quantiles closer to 100%, because these
            # are usually the quantiles we're interested in, and it lets us to save space in `feature_act_quantiles`).
            metric = "act_quantile"
            act_quantile, act_precision = sae_vis_data.feature_stats.get_quantile(
                _feat_acts, feat_acts_nonzero_locations
            )
            topk = TopK(act_quantile, k=k, largest=True)
            act_formatting = [f".{act_precision[i]-2}%" for i in topk.indices]
            scores_dict[f"{metric}|{seq_key}"] = [
                {"feature": feat, "title": title(metric, feat, score)}
                for feat, score in zip(
                    _feature_idx[topk.indices].tolist(),
                    [f"{v:{f}}" for v, f in zip(topk.values.tolist(), act_formatting)],
                )
            ]

        # We don't measure loss effect on the first token
        if seq_pos == 0:
            continue

        # Filter the loss effects, since we only need the ones which have non-zero feature acts on the tokens before them
        prev_feat_acts_nonzero_filter = utils.to_numpy(feat_acts[seq_pos - 1] > 0)
        _loss_contribution = feats_loss_contribution[prev_feat_acts_nonzero_filter, seq_pos - 1]  # [feats_filtered,]
        _feature_idx_prev = np.array(feature_idx)[prev_feat_acts_nonzero_filter]

        if prev_feat_acts_nonzero_filter.sum() > 0:
            k = min(num_top_features, _loss_contribution.numel())

            # Get the top features by loss effect. This is just applying a TopK function to the loss effects (which were
            # stored by the code before this). The loss effects are formatted to 3dp. We look for the most negative
            # values, i.e. the most loss-reducing features.
            metric = "loss_effect"
            topk = TopK(_loss_contribution, k=k, largest=False)
            scores_dict[f"{metric}|{seq_key}"] = [
                {"feature": feat, "title": title(metric, feat, score)}
                for feat, score in zip(
                    _feature_idx_prev[topk.indices].tolist(),
                    [f"{v:+.3f}" for v in topk.values.tolist()],
                )
            ]

    return scores_dict, prompt_data_dict


@torch.inference_mode()
def get_prompt_data(
    sae_vis_data: SaeVisData,
    prompt: str,
    num_top_features: int,
) -> dict[str, list[dict[Literal["feature", "title"], int | str]]]:
    """
    Does 2 things:
        (1) Adds "promptData" into the SaeVisData object's feature dict, so it can be used in the prompt-centric layout
        (2) Returns a dictionary mapping score keys (stringified) to a sorted list of feature IDs & their titles

    We do this simultaneously for every scoring metric & every seq pos in the prompt.
    """

    features = list(sae_vis_data.feature_data_dict.keys())
    sae = sae_vis_data.sae
    assert isinstance(sae, SAE)
    model = sae_vis_data.model
    assert isinstance(model, HookedSAETransformer)
    cfg = sae_vis_data.cfg
    assert isinstance(cfg.hook_point, str), f"{cfg.hook_point=}, expected a string"

    str_toks: list[str] = model.tokenizer.tokenize(prompt)  # type: ignore
    tokens = model.tokenizer.encode(prompt, return_tensors="pt").to(device)  # type: ignore
    assert isinstance(tokens, torch.Tensor)

    feature_act_dir = sae.W_enc[:, features]  # [d_in feats]
    feature_out_dir = sae.W_dec[features]  # [feats d_in]
    feature_resid_dir = to_resid_dir(feature_out_dir, sae, model)  # [feats d_model]
    assert feature_act_dir.T.shape == feature_out_dir.shape == (len(features), sae.cfg.d_in)

    resid_final_hook_name = utils.get_act_name("resid_post", model.cfg.n_layers - 1)
    acts_post_hook_name = f"{sae.cfg.hook_name}.hook_sae_acts_post"
    sae.use_error_term = True
    _, cache = model.run_with_cache_with_saes(
        tokens,
        saes=[sae],
        stop_at_layer=model.cfg.n_layers,
        names_filter=[acts_post_hook_name, resid_final_hook_name],
        remove_batch_dim=True,
    )
    sae.use_error_term = False

    scores_dict, prompt_data_dict = parse_prompt_data(
        tokens=tokens,
        str_toks=str_toks,
        sae_vis_data=sae_vis_data,
        feat_acts=cache[acts_post_hook_name][:, features],
        feature_resid_dir=feature_resid_dir,
        resid_post=cache[resid_final_hook_name],
        model=model,
        feature_idx=features,
        num_top_features=num_top_features,
    )

    # Set prompt data in feature_data_dict, and return scores dict

    for feature_idx, prompt_data in prompt_data_dict.items():
        sae_vis_data.feature_data_dict[feature_idx]["prompt"] = prompt_data

    return scores_dict
