import gc
import itertools
import math
import time
from collections import defaultdict
from typing import Literal

import einops
import numpy as np
import torch
import torch.nn.functional as F
from eindex import eindex
from jaxtyping import Float, Int
from rich import print as rprint
from rich.table import Table
from sae_lens import SAE
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer, utils

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
from sae_vis.model_fns import HookedTransformerWrapper, to_resid_dir
from sae_vis.utils_fns import (
    METRIC_TITLES,
    FeatureStatistics,
    RollingCorrCoef,
    TopK,
    VocabType,
    cross_entropy_loss,
    get_device,
    k_largest_indices,
    random_range_indices,
)

Arr = np.ndarray

device = get_device()


def _get_encodings(sae: SAE, model_acts: Float[Tensor, "batch seq d_in"]):
    # TODO - replace this with a function from SAELens
    sae_in = sae.run_time_activation_norm_fn_in(model_acts)
    sae_in_cent = sae_in - sae.b_dec * sae.cfg.apply_b_dec_to_input

    assert sae.cfg.activation_fn_str == "relu", "Only ReLU activation functions are supported."

    if sae.cfg.architecture == "standard":
        feat_acts = F.relu(
            einops.einsum(sae_in_cent, sae.W_enc, "batch seq d_in, d_in feats -> batch seq feats") + sae.b_enc
        )

    else:
        assert sae.cfg.architecture == "gated", "Only standard & gated architectures are supported."
        gating_pre_activation = (
            einops.einsum(sae_in_cent, sae.W_enc, "batch seq d_in, d_in feats -> batch seq feats") + sae.b_gate
        )

        active_features = (gating_pre_activation > 0).float()

        # magnitude_pre_activation = sae_in @ (self.W_enc * self.r_mag.exp()) + self.b_mag
        feature_magnitudes = F.relu(
            einops.einsum(
                sae_in_cent,
                sae.W_enc * sae.r_mag.exp(),
                "batch seq d_in, d_in feats -> batch seq feats",
            )
            + sae.b_mag
        )
        feat_acts = active_features * feature_magnitudes

    return feat_acts


def compute_feat_acts(
    model_acts: Float[Tensor, "batch seq d_in"],
    feature_idx: list[int],
    sae: SAE,
    sae_B: SAE | None = None,
    corrcoef_neurons: RollingCorrCoef | None = None,
    corrcoef_sae: RollingCorrCoef | None = None,
    corrcoef_sae_B: RollingCorrCoef | None = None,
) -> Float[Tensor, "batch seq feats"]:
    """
    This function computes the feature activations, given a bunch of model data. It also updates the rolling correlation
    coefficient objects, if they're given.

    Args:
        model_acts: Float[Tensor, "batch seq d_in"]
            The activations of the model, which the SAE was trained on.
        feature_idx: list[int]
            The features we're computing the activations for. This will be used to index the sae's weights.
        sae: SAE
            The sae object, which we use to calculate the feature activations.
        sae_B: Optional[SAE]
            The sae-B object, which we use to calculate the feature activations.
        corrcoef_neurons: Optional[RollingCorrCoef]
            The object storing the minimal data necessary to compute corrcoef between feature activations & neurons.
        corrcoef_sae: Optional[RollingCorrCoef]
            The object storing the minimal data necessary to compute corrcoef between pairwise feature activations.
        corrcoef_sae_B: Optional[RollingCorrCoef]
            The object storing minimal data to compute corrcoef between feature activations & sae-B features.
    """
    # Get the feature act direction by indexing sae.W_enc, and the bias by indexing sae.b_enc

    # Calculate & store feature activations (we need to store them so we can get the sequence & histogram vis later)

    feat_acts_all = _get_encodings(sae, model_acts)
    feat_acts = feat_acts_all[..., feature_idx]

    # Update the CorrCoef object between feature activation & neurons
    if corrcoef_neurons is not None:
        corrcoef_neurons.update(
            einops.rearrange(feat_acts, "batch seq feats -> feats (batch seq)"),
            einops.rearrange(model_acts, "batch seq d_in -> d_in (batch seq)"),
        )

    # Update the CorrCoef object between pairwise feature activations
    if corrcoef_sae is not None:
        corrcoef_sae.update(
            einops.rearrange(feat_acts, "batch seq feats -> feats (batch seq)"),
            einops.rearrange(feat_acts_all, "batch seq feats -> feats (batch seq)"),
        )

    # Calculate sae-B feature acts (we don't need to store sae-B acts; it's just for left-hand feature tables)
    if corrcoef_sae_B is not None:
        assert (
            sae_B is not None
        ), "Error: you need to supply an sae-B object if you want to calculate sae-B feature activations."
        feat_acts_B = _get_encodings(sae_B, model_acts)

        # Update the CorrCoef object between feature activation & sae-B features
        corrcoef_sae_B.update(
            einops.rearrange(feat_acts, "batch seq feats -> feats (batch seq)"),
            einops.rearrange(feat_acts_B, "batch seq d_hidden -> d_hidden (batch seq)"),
        )

    return feat_acts


@torch.inference_mode()
def parse_feature_data(
    model: HookedTransformerWrapper,
    cfg: SaeVisConfig,
    sae: SAE,
    sae_B: SAE | None,
    tokens: Int[Tensor, "batch seq"],
    feature_indices: int | list[int],
    all_feat_acts: Float[Tensor, "... feats"],
    feature_resid_dir: Float[Tensor, "feats d_model"],
    feature_resid_dir_input: Float[Tensor, "feats d"],
    all_resid_post: Float[Tensor, "... d_model"],
    feature_out_dir: Float[Tensor, "feats d_out"] | None = None,
    corrcoef_neurons: RollingCorrCoef | None = None,
    corrcoef_sae: RollingCorrCoef | None = None,
    corrcoef_sae_B: RollingCorrCoef | None = None,
    linear_probes_input: dict[str, Float[Tensor, "d_model d_vocab_out"]] = {},
    linear_probes_output: dict[str, Float[Tensor, "d_model d_vocab_out"]] = {},
    target_logits: Float[Tensor, "batch seq d_vocab_out"] | None = None,
    vocab_dict: dict[VocabType, dict[int, str]] | None = None,
    progress: list[tqdm] | None = None,
) -> tuple[SaeVisData, dict[str, float]]:
    """Convert generic activation data into a SaeVisData object, which can be used to create the feature-centric vis.

    This function exists so that feature dashboards can be generated without using our SAE or
    TransformerLens(Wrapper) classes. We pass through W_U & other data needed for computing the logit lens, so we don't
    need the models.

    Returns:
        sae_vis_data: SaeVisData
            Containing data for creating each feature visualization, as well as data for rank-ordering the feature
            visualizations when it comes time to make the prompt-centric view (the `feature_act_quantiles` attribute).

        time_logs: dict[str, float]
            A dictionary containing the time taken for each step of the computation. This is optionally printed at the
            end of the `get_feature_data` function, if `verbose` is set to True.

    # TODO - there's redundant docstrings here, each argument should only have a docstring in one function (the outer-most one)
    """
    time_logs = {
        "(4) Getting data for tables": 0.0,
        "(5) Getting data for histograms": 0.0,
        "(6) Getting data for sequences": 0.0,
        "(7) Getting data for quantiles": 0.0,
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

    # ! [1/3] Feature tables (i.e. left hand of vis)

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

    time_logs["(4) Getting data for tables"] = time.monotonic() - t0
    t0 = time.monotonic()

    # ! [2/3] Histograms & logit tables & optional othello probes (i.e. middle column of vis)

    # Get the logits of all features (i.e. the directions this feature writes to the logit output)
    logits = einops.einsum(feature_resid_dir, model.W_U, "feats d_model, d_model d_vocab -> feats d_vocab")
    probe_logits = {
        name: einops.einsum(
            feature_resid_dir,
            probe,
            "feats d_model, d_model d_vocab_out -> feats d_vocab_out",
        )
        for name, probe in linear_probes_output.items()
    }
    probe_acts = {
        name: einops.einsum(
            feature_resid_dir_input,
            probe,
            "feats d_model, d_model d_vocab_out -> feats d_vocab_out",
        )
        for name, probe in linear_probes_input.items()
    }

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
                    probe_logits={name: pl[i] for name, pl in probe_logits.items()},
                    probe_acts={name: pl[i] for name, pl in probe_acts.items()},
                    k=layout.probe_logits_table_cfg.n_rows,
                )

    time_logs["(5) Getting data for histograms"] = time.monotonic() - t0
    t0 = time.monotonic()

    # ! [3/3] Sequences (i.e. right hand of vis)

    if layout.seq_cfg is not None:
        for i, feat in enumerate(feature_indices):
            # Add this feature's sequence data to the list
            feature_data_dict[feat]["seqMultiGroup"] = get_sequences_data(
                tokens=tokens,
                feat_acts=all_feat_acts[..., i],
                feat_logits=logits[i],
                resid_post=all_resid_post,
                feature_resid_dir=feature_resid_dir[i],
                W_U=model.W_U,
                target_logits=target_logits,
                seq_cfg=layout.seq_cfg,
            )
            # Update the 2nd progress bar (fwd passes & getting sequence data dominates the runtime of these computations)
            if progress is not None:
                progress[1].update(1)

    time_logs["(6) Getting data for sequences"] = time.monotonic() - t0
    t0 = time.monotonic()

    # ! Get stats (including quantiles, which will be useful for the prompt-centric visualisation)
    feature_stats = FeatureStatistics.create(data=einops.rearrange(all_feat_acts, "b s feats -> feats (b s)"))
    time_logs["(7) Getting data for quantiles"] = time.monotonic() - t0
    t0 = time.monotonic()

    # ! Return the output, as a dict of FeatureData items
    sae_vis_data = SaeVisData(
        feature_data_dict,
        {},
        feature_stats,
        cfg=cfg,
        model=model.model,
        sae=sae,
        sae_B=sae_B,
        vocab_dict=vocab_dict,
    )
    return sae_vis_data, time_logs


@torch.inference_mode()
def _get_feature_data(
    sae: SAE,
    model: HookedTransformerWrapper,
    tokens: Int[Tensor, "batch seq"],
    feature_indices: int | list[int],
    cfg: SaeVisConfig,
    sae_B: SAE | None,
    linear_probes_input: dict[str, Float[Tensor, "d_model d_vocab_out"]] = {},
    linear_probes_output: dict[str, Float[Tensor, "d_model d_vocab_out"]] = {},
    target_logits: Float[Tensor, "batch seq d_vocab_out"] | None = None,
    vocab_dict: dict[VocabType, dict[int, str]] | None = None,
    progress: list[tqdm] | None = None,
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
    time_logs = {
        "(1) Initialization": 0.0,
        "(2) Forward passes to gather model activations": 0.0,
        "(3) Computing feature acts from model acts": 0.0,
    }

    t0 = time.monotonic()

    # Make feature_indices a list, for convenience
    if isinstance(feature_indices, int):
        feature_indices = [feature_indices]

    # Get tokens into minibatches, for the fwd pass. Same for target logits, if using them
    token_minibatches_list = (tokens,) if cfg.minibatch_size_tokens is None else tokens.split(cfg.minibatch_size_tokens)
    token_minibatches = [tok.to(device) for tok in token_minibatches_list]

    # ! Data setup code (defining the main objects we'll eventually return, for each of 5 possible vis components)

    # Create tensors to store the feature activations & final values of the residual stream
    seqpos_slice = slice(*cfg.seqpos_slice)
    all_resid_post = torch.zeros(*tokens.shape, model.cfg.d_model, device=device)
    all_feat_acts = torch.zeros(*tokens.shape, len(feature_indices), device=device)

    # Create objects to store the data for computing rolling stats
    corrcoef_neurons = RollingCorrCoef()
    corrcoef_sae = RollingCorrCoef(indices=feature_indices, with_self=True)
    corrcoef_sae_B = RollingCorrCoef() if sae_B is not None else None

    # Get sae & decoder directions
    feature_out_dir = sae.W_dec[feature_indices]  # [feats d_sae]
    feature_resid_dir = to_resid_dir(feature_out_dir, model)  # [feats d_model]
    feature_in_dir = sae.W_enc.T[feature_indices]  # [feats d_in]
    feature_resid_dir_input = to_resid_dir(feature_in_dir, model, input=True)  # [feats d_model]

    time_logs["(1) Initialization"] = time.monotonic() - t0
    batch_start = 0

    # ! Compute & concatenate together all feature activations & post-activation function values

    for minibatch in token_minibatches:
        # Fwd pass, get model activations
        t0 = time.monotonic()
        _, residual, model_acts = model.forward(minibatch)
        time_logs["(2) Forward passes to gather model activations"] += time.monotonic() - t0

        # Compute feature activations from this
        t0 = time.monotonic()
        feat_acts = compute_feat_acts(
            model_acts=model_acts,
            feature_idx=feature_indices,
            sae=sae,
            sae_B=sae_B,
            corrcoef_neurons=corrcoef_neurons,
            corrcoef_sae=corrcoef_sae,
            corrcoef_sae_B=corrcoef_sae_B,
        )
        # Put these values into the tensors
        # TODO - figure out if this is okay (anything outside seqpos_slice is zeroed)
        all_feat_acts[batch_start : batch_start + len(minibatch), seqpos_slice] = feat_acts[:, seqpos_slice]
        all_resid_post[batch_start : batch_start + len(minibatch), seqpos_slice] = residual[:, seqpos_slice]
        time_logs["(3) Computing feature acts from model acts"] += time.monotonic() - t0

        # Update the 1st progress bar (fwd passes & getting sequence data dominates the runtime of these computations)
        if progress is not None:
            progress[0].update(1)

        gc.collect()
        torch.cuda.empty_cache()
        batch_start += len(minibatch)

    # ! Use the data we've collected to make a MultiFeatureData object
    sae_vis_data, _time_logs = parse_feature_data(
        model=model,
        cfg=cfg,
        sae=sae,
        sae_B=sae_B,
        tokens=tokens,
        feature_indices=feature_indices,
        all_feat_acts=all_feat_acts,
        feature_resid_dir=feature_resid_dir,
        feature_resid_dir_input=feature_resid_dir_input,
        all_resid_post=all_resid_post,
        feature_out_dir=feature_out_dir,
        corrcoef_neurons=corrcoef_neurons,
        corrcoef_sae=corrcoef_sae,
        corrcoef_sae_B=corrcoef_sae_B,
        linear_probes_input=linear_probes_input,
        linear_probes_output=linear_probes_output,
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
    linear_probes_input: dict[str, Float[Tensor, "d_model d_vocab_out"]] = {},
    linear_probes_output: dict[str, Float[Tensor, "d_model d_vocab_out"]] = {},
    target_logits: Float[Tensor, "batch seq d_vocab_out"] | None = None,
    vocab_dict: dict[VocabType, dict[int, str]] | None = None,
    verbose: bool = False,
) -> SaeVisData:
    """
    This is the main function which users will run to generate the feature visualization data. It
    batches this computation over features, in accordance with the arguments in the SaeVisConfig
    object (we don't want to compute all the features at once, since might give OOMs).

    See the `_get_feature_data` function for an explanation of the arguments, as well as a more
    detailed explanation of what this function is doing.

    The return object is the merged SaeVisData objects returned by the `_get_feature_data` function.
    """
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
        linear_probes_input=linear_probes_input,
        linear_probes_output=linear_probes_output,
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
    assert isinstance(model, HookedTransformer)
    assert isinstance(cfg.hook_point, str)
    model_wrapper = HookedTransformerWrapper(model, cfg.hook_point)

    # For each batch of features: get new data and update global data storage objects
    for features in feature_batches:
        new_feature_data, new_time_logs = _get_feature_data(
            sae=sae,
            model=model_wrapper,
            tokens=tokens,
            feature_indices=features,
            cfg=cfg,
            sae_B=sae_B,
            linear_probes_input=linear_probes_input,
            linear_probes_output=linear_probes_output,
            target_logits=target_logits,
            vocab_dict=vocab_dict,
            progress=progress,
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
        total_time = sum(time_logs.values())
        table = Table("Task", "Time", "Pct %")
        for task, duration in time_logs.items():
            table.add_row(task, f"{duration:.2f}s", f"{duration/total_time:.1%}")
        rprint(table)

    return sae_vis_data


@torch.inference_mode()
def get_sequences_data(
    tokens: Int[Tensor, "batch seq"],
    feat_acts: Float[Tensor, "batch seq"],
    feat_logits: Float[Tensor, "d_vocab"],
    resid_post: Float[Tensor, "batch seq d_model"],
    feature_resid_dir: Float[Tensor, "d_model"],
    W_U: Float[Tensor, "d_model d_vocab"],
    target_logits: Float[Tensor, "batch seq d_vocab_out"] | None,
    seq_cfg: SeqMultiGroupConfig,
) -> SeqMultiGroupData:
    """
    This function returns the data which is used to create the sequence visualizations (i.e. the
    right-hand column of the visualization). This is a multi-step process (the 4 steps are annotated
    in the code):

        (1) Find all the token groups (i.e. topk, bottomk, and quantile groups of activations). These
            are bold tokens.
        (2) Get the indices of all tokens we'll need data from, which includes a buffer around each
            bold token. Index these token IDs.
        (3) Compute the feature activations & residual stream values for all relevant positions.
        (4) Compute the logit effect if this feature is ablated.
            (4A) Use this to compute the most affected tokens by this feature (i.e. the vis hoverdata)
            (4B) Use this to compute the loss effect if this feature is ablated (i.e. the blue/red underlining)
        (5) Return all this data as a SeqMultiGroupData object.

    Args:
        tokens:
            The tokens we'll be extracting sequence data from.
        feat_acts:
            The activations of the feature we're interested in, for each token in the batch.
        feat_logits:
            The logit vector for this feature (used to generate histogram, and is needed here for the line-on-hover).
        resid_post:
            The residual stream values before final layernorm, for each token in the batch.
        feature_resid_dir:
            The direction this feature writes to the logit output (i.e. the direction we'll be erasing from resid_post).
        W_U:
            The model's unembedding matrix, which we'll use to get the logits.
        cfg:
            Feature visualization parameters, containing some important params e.g. num sequences per group.

    Returns:
        SeqMultiGroupData
            This is a dataclass which contains a dict of SeqGroupData objects, where each SeqGroupData object
            contains the data for a particular group of sequences (i.e. the top-k, bottom-k, and the quantile groups).
    """

    # ! (1) Find the tokens from each group

    # Get buffer, s.t. we're looking for bold tokens in the range `buffer[0] : buffer[1]`. For each bold token, we need
    # to see `seq_cfg.buffer[0]+1` behind it (plus 1 because we need the prev token to compute loss effect), and we need
    # to see `seq_cfg.buffer[1]` ahead of it.
    _, seq_length = tokens.shape
    if seq_cfg.buffer is None:
        buffer = (0, -1)
        buf_size = seq_length - 1
    else:
        buffer = (seq_cfg.buffer[0] + 1, -seq_cfg.buffer[1])
        buf_size = seq_cfg.buffer[0] + seq_cfg.buffer[1] + 1

    # Get the top-activating tokens
    indices = k_largest_indices(feat_acts, k=seq_cfg.top_acts_group_size, buffer=buffer)
    indices_dict = {
        f"TOP ACTIVATIONS<br><span style='color:#666;font-weight:normal'>MAX = {feat_acts.max():.3f}</span>": indices
    }

    # Get all possible indices. Note, we need to be able to look 1 back (feature activation on prev
    # token is needed for computing loss effect on this token)
    if seq_cfg.n_quantiles > 0:
        quantiles = torch.linspace(0, feat_acts.max().item(), seq_cfg.n_quantiles + 1)
        for i in range(seq_cfg.n_quantiles - 1, -1, -1):
            lower, upper = quantiles[i : i + 2].tolist()
            pct = ((feat_acts >= lower) & (feat_acts <= upper)).float().mean()
            indices = random_range_indices(
                feat_acts,
                k=seq_cfg.quantile_group_size,
                bounds=(lower, upper),
                buffer=buffer,
            )
            indices_dict[
                f"INTERVAL {lower:.3f} - {upper:.3f}<br><span style='color:#666;font-weight:normal'>CONTAINS {pct:.3%}</span>"
            ] = indices

    # Concat all the indices together (in the next steps we do all groups at once). Shape of this
    # object is [n_ex 2], # i.e. the [i, :]-th element are the batch and sequence dimensions for
    # the i-th example.
    indices_ex = torch.concat(list(indices_dict.values())).cpu()
    indices_ex_batch, indices_ex_seq = indices_ex.unbind(dim=-1)
    n_ex = indices_ex.shape[0]

    # ! (2) Get the positions of our tokens in top seqs, and get those tokens

    if seq_cfg.buffer is not None:
        # Get the buffer indices, by adding a broadcasted arange object. In this case, indices_buf
        # contains 1 more token than the length of the sequences we'll see (because it also contains
        # the token before the sequence starts). So we slice `token_ids` to get our display seqs.
        buffer_tensor = torch.arange(-seq_cfg.buffer[0] - 1, seq_cfg.buffer[1] + 1, device=indices_ex.device)
        indices_buf = torch.stack(
            [
                einops.repeat(indices_ex_batch, "n_ex -> n_ex seq", seq=buf_size + 1),
                einops.repeat(indices_ex_seq, "n_ex -> n_ex seq", seq=buf_size + 1) + buffer_tensor,
            ],
            dim=-1,
        )
        token_ids = eindex(
            tokens,
            indices_buf,
            "[n_ex buf_plus1 0] [n_ex buf_plus1 1] -> n_ex buf_plus1",
        )
        token_ids_to_display = token_ids[:, 1:]
    else:
        # If we don't specify a sequence, then do all of the seq positions in each seq we pick. In
        # this case, our display seqs are literally just the full sequences from bold tokens.
        indices_buf = torch.stack(
            [
                einops.repeat(indices_ex_batch, "n_ex -> n_ex seq", seq=seq_length),  # batch indices of bold tokens
                einops.repeat(torch.arange(seq_length), "seq -> n_ex seq", n_ex=n_ex),  # all sequence indices
            ],
            dim=-1,
        )
        token_ids = eindex(
            tokens,
            indices_buf,
            "[n_ex buf_plus1 0] [n_ex buf_plus1 1] -> n_ex buf_plus1",
        )
        token_ids_to_display = token_ids

    assert indices_buf.shape == (
        n_ex,
        buf_size + 1,
        2,
    ), f"Error: {indices_buf.shape=}, {n_ex=}, {buf_size+1=}"

    # ! (3) Extract feature activations & residual stream values for those positions

    # Now, we split into cases depending on whether we're computing the buffer or not. One kinda
    # weird thing: when we're computing the buffer we need feat_acts[:, 1:] for coloring and
    # feat_acts[:, :-1] for ablation, but when no buffer we only need feat_acts[:, bold] for both.
    # So we split on cases here.
    if seq_cfg.compute_buffer:
        feat_acts_buf = eindex(
            feat_acts,
            indices_buf,
            "[n_ex buf_plus1 0] [n_ex buf_plus1 1] -> n_ex buf_plus1",
        )
        feat_acts_pre_ablation = feat_acts_buf[:, :-1]
        feat_acts_coloring = feat_acts_buf[:, 1:]
        feat_acts_idx = [None for _ in range(n_ex)]
        resid_post = eindex(
            resid_post,
            indices_buf[:, :-1],
            "[n_ex buf 0] [n_ex buf 1] d_model -> n_ex buf d_model",
        )
        correct_tokens = token_ids[:, 1:]
    else:
        feat_acts_pre_ablation = eindex(feat_acts, indices_ex_batch, indices_ex_seq, "[n_ex] [n_ex] -> n_ex").unsqueeze(
            1
        )
        feat_acts_coloring = feat_acts_pre_ablation
        feat_acts_idx = indices_ex_seq.tolist()
        resid_post = eindex(
            resid_post,
            indices_ex_batch,
            indices_ex_seq,
            "[n_ex] [n_ex] d_model -> n_ex d_model",
        ).unsqueeze(1)
        # The tokens we'll use to index correct logits are the ones *after* the bold ones
        correct_tokens = eindex(tokens, indices_ex_batch, indices_ex_seq + 1, "[n_ex] [n_ex] -> n_ex").unsqueeze(1)

    # ! (4) Compute the logit effect if this feature is ablated

    # Get this feature's output vector, using an outer product over the feature activations for all tokens
    resid_post_feature_effect = einops.einsum(
        feat_acts_pre_ablation,
        feature_resid_dir,
        "n_ex buf, d_model -> n_ex buf d_model",
    )

    # Do the ablations, and get difference in logprobs
    new_resid_post = resid_post - resid_post_feature_effect
    new_logits = (new_resid_post / new_resid_post.std(-1, keepdim=True)) @ W_U
    orig_logits = (resid_post / resid_post.std(-1, keepdim=True)) @ W_U
    contribution_to_logprobs = orig_logits.log_softmax(-1) - new_logits.log_softmax(-1)
    orig_prob = orig_logits.softmax(-1)
    new_prob = new_logits.softmax(-1)

    # ! (4A) Use this to compute the most affected tokens by this feature
    # The TopK function can improve efficiency by masking the features which are zero
    acts_nonzero = feat_acts_pre_ablation.abs() > 1e-5  # shape [batch buf]
    top_contribution_to_logits = TopK(
        contribution_to_logprobs,
        k=seq_cfg.top_logits_hoverdata,
        tensor_mask=acts_nonzero,
    )
    bottom_contribution_to_logits = TopK(
        contribution_to_logprobs,
        k=seq_cfg.top_logits_hoverdata,
        tensor_mask=acts_nonzero,
        largest=False,
    )

    # ! (4B) Use this to compute the loss effect if this feature is ablated
    # which is just the negative of the change in logprobs
    if target_logits is None:
        # loss_cont[b, s] = -logprobs_cont[b, s, correct_tokens[b, s]]
        loss_contribution = eindex(-contribution_to_logprobs, correct_tokens, "batch seq [batch seq]")
        orig_prob = eindex(orig_prob, correct_tokens, "batch seq [batch seq]")
        new_prob = eindex(new_prob, correct_tokens, "batch seq [batch seq]")
    else:
        assert (
            not seq_cfg.compute_buffer
        ), "Not expecting to compute buffer if using target logits (it's more indexing hassle)"
        target_logits_bold = eindex(target_logits, indices_ex, "[n_ex 0] [n_ex 1] d_vocab").unsqueeze(1)
        loss_orig = cross_entropy_loss(orig_logits, target_logits_bold)
        loss_new = cross_entropy_loss(new_logits, target_logits_bold)
        loss_contribution = loss_orig - loss_new
        orig_prob = None
        new_prob = None

    # ! (5) Store the results in a SeqMultiGroupData object

    # Now that we've indexed everything, construct the batch of SequenceData objects
    sequence_groups_data = []
    group_sizes_cumsum = np.cumsum([0] + [len(indices) for indices in indices_dict.values()]).tolist()
    for group_idx, group_name in enumerate(indices_dict.keys()):
        seq_data = [
            SequenceData(
                token_ids=token_ids_to_display[i].tolist(),
                feat_acts=[round(f, 4) for f in feat_acts_coloring[i].tolist()],
                feat_acts_idx=feat_acts_idx[i],
                loss_contribution=loss_contribution[i].tolist(),
                orig_prob=orig_prob[i].tolist() if orig_prob is not None else None,
                new_prob=new_prob[i].tolist() if new_prob is not None else None,
                token_logits=feat_logits[token_ids_to_display[i]].tolist(),
                top_token_ids=top_contribution_to_logits.indices[i].tolist(),
                top_logits=top_contribution_to_logits.values[i].tolist(),
                bottom_token_ids=bottom_contribution_to_logits.indices[i].tolist(),
                bottom_logits=bottom_contribution_to_logits.values[i].tolist(),
            )
            for i in range(group_sizes_cumsum[group_idx], group_sizes_cumsum[group_idx + 1])
        ]
        sequence_groups_data.append(SeqGroupData(seq_data, group_name))

    return SeqMultiGroupData(sequence_groups_data)


@torch.inference_mode()
def parse_prompt_data(
    tokens: Int[Tensor, "batch seq"],
    str_toks: list[str],
    sae_vis_data: SaeVisData,
    feat_acts: Float[Tensor, "seq feats"],
    feature_resid_dir: Float[Tensor, "feats d_model"],
    resid_post: Float[Tensor, "seq d_model"],
    W_U: Float[Tensor, "d_model d_vocab"],
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
    orig_logits = (resid_post / resid_post.std(dim=-1, keepdim=True)) @ W_U  # [seq d_vocab]
    raw_logits = feature_resid_dir @ W_U  # [feats d_vocab]

    for i, feat in enumerate(feature_idx):
        # ! Calculate the sequence data for each feature, and store it as FeatureData.prompt_data

        # Get this feature's output vector, using an outer product over the feature activations for all tokens
        resid_post_feature_effect = einops.einsum(feat_acts[:, i], feature_resid_dir[i], "seq, d_model -> seq d_model")

        # Ablate the output vector from the residual stream, and get logits post-ablation
        new_resid_post = resid_post - resid_post_feature_effect
        new_logits = (new_resid_post / new_resid_post.std(dim=-1, keepdim=True)) @ W_U

        # Get the top5 & bottom5 changes in logits (don't bother with `efficient_topk` cause it's small)
        contribution_to_logprobs = orig_logits.log_softmax(dim=-1) - new_logits.log_softmax(dim=-1)
        top_contribution_to_logits = TopK(contribution_to_logprobs[:-1], k=5)
        bottom_contribution_to_logits = TopK(contribution_to_logprobs[:-1], k=5, largest=False)

        # Get the change in loss (which is negative of change of logprobs for correct token)
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
                            loss_contribution=[0.0] + loss_contribution.tolist(),
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

    feature_idx = list(sae_vis_data.feature_data_dict.keys())
    sae = sae_vis_data.sae
    assert isinstance(sae, SAE)
    model = sae_vis_data.model
    assert isinstance(model, HookedTransformer)
    cfg = sae_vis_data.cfg
    assert isinstance(cfg.hook_point, str), f"{cfg.hook_point=}, expected a string"

    str_toks: list[str] = model.tokenizer.tokenize(prompt)  # type: ignore
    tokens = model.tokenizer.encode(prompt, return_tensors="pt").to(device)  # type: ignore
    assert isinstance(tokens, torch.Tensor)

    model_wrapped = HookedTransformerWrapper(model, cfg.hook_point)

    feature_act_dir = sae.W_enc[:, feature_idx]  # [d_in feats]
    feature_out_dir = sae.W_dec[feature_idx]  # [feats d_in]
    feature_resid_dir = to_resid_dir(feature_out_dir, model_wrapped)  # [feats d_model]
    assert feature_act_dir.T.shape == feature_out_dir.shape == (len(feature_idx), sae.cfg.d_in)

    _, resid_post, act_post = model_wrapped(tokens)
    resid_post: Tensor = resid_post.squeeze(0)
    feat_acts = compute_feat_acts(act_post, feature_idx, sae).squeeze(0)  # [seq feats]

    scores_dict, prompt_data_dict = parse_prompt_data(
        tokens=tokens,
        str_toks=str_toks,
        sae_vis_data=sae_vis_data,
        feat_acts=feat_acts,
        feature_resid_dir=feature_resid_dir,
        resid_post=resid_post,
        W_U=model.W_U,
        feature_idx=feature_idx,
        num_top_features=num_top_features,
    )

    # Set prompt data in feature_data_dict, and return scores dict

    for feature_idx, prompt_data in prompt_data_dict.items():
        sae_vis_data.feature_data_dict[feature_idx]["prompt"] = prompt_data

    return scores_dict
