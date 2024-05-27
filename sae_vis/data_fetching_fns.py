import math
import time
from collections import defaultdict

import einops
import numpy as np
import torch
import torch.nn.functional as F
from eindex import eindex
from jaxtyping import Float, Int
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer, utils

from sae_vis.data_config_classes import (
    SaeVisConfig,
    SaeVisLayoutConfig,
    SequencesConfig,
)

# RollingStats,
from sae_vis.data_storing_fns import (
    ActsHistogramData,
    FeatureData,
    FeatureTablesData,
    LogitsHistogramData,
    LogitsTableData,
    SaeVisData,
    SequenceData,
    SequenceGroupData,
    SequenceMultiGroupData,
)
from sae_vis.model_fns import (
    AutoEncoder,
    TransformerLensWrapper,
    to_resid_dir,
)
from sae_vis.utils_fns import (
    FeatureStatistics,
    RollingCorrCoef,
    TopK,
    get_device,
    k_largest_indices,
    random_range_indices,
)

Arr = np.ndarray

device = get_device()


def compute_feat_acts(
    model_acts: Float[Tensor, "batch seq d_in"],
    feature_idx: list[int],
    encoder: AutoEncoder,
    encoder_B: AutoEncoder | None = None,
    corrcoef_neurons: RollingCorrCoef | None = None,
    corrcoef_encoder: RollingCorrCoef | None = None,
    corrcoef_encoder_B: RollingCorrCoef | None = None,
) -> Float[Tensor, "batch seq feats"]:
    """
    This function computes the feature activations, given a bunch of model data. It also updates the rolling correlation
    coefficient objects, if they're given.

    Args:
        model_acts: Float[Tensor, "batch seq d_in"]
            The activations of the model, which the SAE was trained on.
        feature_idx: list[int]
            The features we're computing the activations for. This will be used to index the encoder's weights.
        encoder: AutoEncoder
            The encoder object, which we use to calculate the feature activations.
        encoder_B: Optional[AutoEncoder]
            The encoder-B object, which we use to calculate the feature activations.
        corrcoef_neurons: Optional[RollingCorrCoef]
            The object storing the minimal data necessary to compute corrcoef between feature activations & neurons.
        corrcoef_encoder: Optional[RollingCorrCoef]
            The object storing the minimal data necessary to compute corrcoef between pairwise feature activations.
        corrcoef_encoder_B: Optional[RollingCorrCoef]
            The object storing minimal data to compute corrcoef between feature activations & encoder-B features.
    """
    # Get the feature act direction by indexing encoder.W_enc, and the bias by indexing encoder.b_enc
    feature_act_dir = encoder.W_enc[:, feature_idx]  # (d_in, feats)
    feature_bias = encoder.b_enc[feature_idx]  # (feats,)

    # Calculate & store feature activations (we need to store them so we can get the sequence & histogram vis later)
    x_cent = model_acts -  encoder.b_dec * encoder.cfg.apply_b_dec_to_input
    feat_acts_pre = einops.einsum(
        x_cent, feature_act_dir, "batch seq d_in, d_in feats -> batch seq feats"
    )
    feat_acts = F.relu(feat_acts_pre + feature_bias)

    # Update the CorrCoef object between feature activation & neurons
    if corrcoef_neurons is not None:
        corrcoef_neurons.update(
            einops.rearrange(feat_acts, "batch seq feats -> feats (batch seq)"),
            einops.rearrange(model_acts, "batch seq d_in -> d_in (batch seq)"),
        )

    # Update the CorrCoef object between pairwise feature activations
    if corrcoef_encoder is not None:
        corrcoef_encoder.update(
            einops.rearrange(feat_acts, "batch seq feats -> feats (batch seq)"),
            einops.rearrange(feat_acts, "batch seq feats -> feats (batch seq)"),
        )

    # Calculate encoder-B feature acts (we don't need to store encoder-B acts; it's just for left-hand feature tables)
    if corrcoef_encoder_B is not None:
        assert (
            encoder_B is not None
        ), "Error: you need to supply an encoder-B object if you want to calculate encoder-B feature activations."
        x_cent_B = model_acts - encoder_B.b_dec * encoder_B.cfg.apply_b_dec_to_input
        feat_acts_pre_B = einops.einsum(
            x_cent_B,
            encoder_B.W_enc,
            "batch seq d_in, d_in d_hidden -> batch seq d_hidden",
        )
        feat_acts_B = F.relu(feat_acts_pre_B + encoder.b_enc)

        # Update the CorrCoef object between feature activation & encoder-B features
        corrcoef_encoder_B.update(
            einops.rearrange(feat_acts, "batch seq feats -> feats (batch seq)"),
            einops.rearrange(feat_acts_B, "batch seq d_hidden -> d_hidden (batch seq)"),
        )

    return feat_acts


@torch.inference_mode()
def parse_feature_data(
    tokens: Int[Tensor, "batch seq"],
    feature_indices: int | list[int],
    all_feat_acts: Float[Tensor, "... feats"],
    feature_resid_dir: Float[Tensor, "feats d_model"],
    all_resid_post: Float[Tensor, "... d_model"],
    W_U: Float[Tensor, "d_model d_vocab"],
    cfg: SaeVisConfig,
    feature_out_dir: Float[Tensor, "feats d_out"] | None = None,
    corrcoef_neurons: RollingCorrCoef | None = None,
    corrcoef_encoder: RollingCorrCoef | None = None,
    corrcoef_encoder_B: RollingCorrCoef | None = None,
    progress: list[tqdm] | None = None,
) -> tuple[SaeVisData, dict[str, float]]:
    """Convert generic activation data into a SaeVisData object, which can be used to create the feature-centric vis.

    This function exists so that feature dashboards can be generated without using our AutoEncoder or
    TransformerLens(Wrapper) classes. We pass through W_U & other data needed for computing the logit lens, so we don't
    need the models.

    Args:
        tokens: Int[Tensor, "batch seq"]
            The tokens we'll be using to get the feature activations.

        feature_indices: Union[int, list[int]]
            The features we're actually computing. These might just be a subset of the model's full features.

        all_feat_acts: Float[Tensor, "... feats"]
            The activations values of the features across the batch & sequence.

        feature_resid_dir: Float[Tensor, "feats d_model"]
            The directions that each feature writes to the residual stream.
            For example, feature_resid_dir = encoder.W_dec[feature_indices] # [feats d_autoencoder]

        all_resid_post: Float[Tensor, "... d_model"]
            The activations of the final layer of the model before the unembed.

        W_U: Float[Tensor, "d_model d_vocab"]
            The model's unembed weights for the logit lens.

        cfg: SaeVisConfig
            Feature visualization parameters, containing a bunch of other stuff. See the SaeVisConfig docstring for
            more information.

        feature_out_dir: Optional[Float[Tensor, "feats d_out"]]
            The directions that each SAE feature writes to the residual stream. This will be the same as
            feature_resid_dir if the SAE is in the residual stream (as we will assume if it not provided)
            For example, feature_out_dir = encoder.W_dec[feature_indices] # [feats d_autoencoder]

        corrcoef_neurons: Optional[RollingCorrCoef]
            The object storing the minimal data necessary to compute corrcoef between feature activations & neurons.

        corrcoef_encoder: Optional[RollingCorrCoef]
            The object storing the minimal data necessary to compute corrcoef between feature activations (pairwise).

        corrcoef_encoder_B: Optional[RollingCorrCoef]
            The object storing minimal data to compute corrcoef between feature activations & encoder-B features.

        progress: Optional[list[tqdm]]
            An optional list containing progress bars for the forward passes and the sequence data. This is used to
            update the progress bars as the computation runs.

    Returns:
        sae_vis_data: SaeVisData
            Containing data for creating each feature visualization, as well as data for rank-ordering the feature
            visualizations when it comes time to make the prompt-centric view (the `feature_act_quantiles` attribute).

        time_logs: dict[str, float]
            A dictionary containing the time taken for each step of the computation. This is optionally printed at the
            end of the `get_feature_data` function, if `cfg.verbose` is set to True.
    """
    time_logs = {
        "(4) Getting data for tables": 0.0,
        "(5) Getting data for histograms": 0.0,
        "(6) Getting data for sequences": 0.0,
        "(7) Getting data for quantiles": 0.0,
    }
    t0 = time.time()

    # Make feature_indices a list, for convenience
    if isinstance(feature_indices, int):
        feature_indices = [feature_indices]

    assert (
        feature_resid_dir.shape[0] == len(feature_indices)
    ), f"Num features in feature_resid_dir ({feature_resid_dir.shape[0]}) doesn't match {len(feature_indices)=}"

    if feature_out_dir is not None:
        assert (
            feature_out_dir.shape[0] == len(feature_indices)
        ), f"Num features in feature_out_dir ({feature_resid_dir.shape[0]}) doesn't match {len(feature_indices)=}"

    # ! Data setup code (defining the main objects we'll eventually return)
    feature_data_dict: dict[int, FeatureData] = {
        feat: FeatureData() for feat in feature_indices
    }

    # We're using `cfg.feature_centric_layout` to figure out what data we'll need to calculate during this function
    layout = cfg.feature_centric_layout
    assert isinstance(
        layout, SaeVisLayoutConfig
    ), f"Error: cfg.feature_centric_layout must be a SaeVisLayoutConfig object, got {type(layout)}"

    # ! Calculate all data for the left-hand column visualisations, i.e. the 3 tables

    if layout.feature_tables_cfg is not None and feature_out_dir is not None:
        # Store kwargs (makes it easier to turn the tables on and off individually)
        feature_tables_data = {}

        # Table 1: neuron alignment, based on decoder weights
        if layout.feature_tables_cfg.neuron_alignment_table:
            top3_neurons_aligned = TopK(
                tensor=feature_out_dir, k=layout.feature_tables_cfg.n_rows, largest=True
            )
            feature_out_l1_norm = feature_out_dir.abs().sum(dim=-1, keepdim=True)
            pct_of_l1: Arr = np.absolute(top3_neurons_aligned.values) / utils.to_numpy(
                feature_out_l1_norm
            )
            feature_tables_data.update(
                neuron_alignment_indices=top3_neurons_aligned.indices.tolist(),
                neuron_alignment_values=top3_neurons_aligned.values.tolist(),
                neuron_alignment_l1=pct_of_l1.tolist(),
            )

        # Table 2: neurons correlated with this feature, based on their activations
        if isinstance(corrcoef_neurons, RollingCorrCoef):
            neuron_indices, neuron_pearson, neuron_cossim = (
                corrcoef_neurons.topk_pearson(
                    k=layout.feature_tables_cfg.n_rows,
                )
            )
            feature_tables_data.update(
                correlated_neurons_indices=neuron_indices,
                correlated_neurons_pearson=neuron_pearson,
                correlated_neurons_cossim=neuron_cossim,
            )

        # Table 3: primary encoder features correlated with this feature, based on their activations
        if isinstance(corrcoef_encoder, RollingCorrCoef):
            enc_indices, enc_pearson, enc_cossim = corrcoef_encoder.topk_pearson(
                k=layout.feature_tables_cfg.n_rows,
            )
            feature_tables_data.update(
                correlated_features_indices=enc_indices,
                correlated_features_pearson=enc_pearson,
                correlated_features_cossim=enc_cossim,
            )

        # Table 4: encoder-B features correlated with this feature, based on their activations
        if isinstance(corrcoef_encoder_B, RollingCorrCoef):
            encB_indices, encB_pearson, encB_cossim = corrcoef_encoder_B.topk_pearson(
                k=layout.feature_tables_cfg.n_rows,
            )
            feature_tables_data.update(
                correlated_b_features_indices=encB_indices,
                correlated_b_features_pearson=encB_pearson,
                correlated_b_features_cossim=encB_cossim,
            )

        # Add all this data to the list of FeatureTablesData objects
        for i, feat in enumerate(feature_indices):
            feature_data_dict[feat].feature_tables_data = FeatureTablesData(
                **{k: v[i] for k, v in feature_tables_data.items()}
            )

    time_logs["(4) Getting data for tables"] = time.time() - t0
    t0 = time.time()

    # ! Get all data for the middle column visualisations, i.e. the two histograms & the logit table

    # Get the logits of all features (i.e. the directions this feature writes to the logit output)
    logits = einops.einsum(
        feature_resid_dir, W_U, "feats d_model, d_model d_vocab -> feats d_vocab"
    )
    if any(
        x is not None
        for x in [layout.act_hist_cfg, layout.logits_hist_cfg, layout.logits_table_cfg]
    ):
        for i, (feat, logit_vector) in enumerate(zip(feature_indices, logits)):
            # Get logits histogram data (no title)
            if layout.logits_hist_cfg is not None:
                feature_data_dict[
                    feat
                ].logits_histogram_data = LogitsHistogramData.from_data(
                    data=logit_vector,
                    n_bins=layout.logits_hist_cfg.n_bins,
                    tickmode="5 ticks",
                    title=None,
                )

            # Get data for feature activations histogram (including the title!)
            if layout.act_hist_cfg is not None:
                feat_acts = all_feat_acts[..., i]
                nonzero_feat_acts = feat_acts[feat_acts > 0]
                frac_nonzero = nonzero_feat_acts.numel() / feat_acts.numel()
                feature_data_dict[
                    feat
                ].acts_histogram_data = ActsHistogramData.from_data(
                    data=nonzero_feat_acts,
                    n_bins=layout.act_hist_cfg.n_bins,
                    tickmode="5 ticks",
                    title=f"ACTIVATIONS<br>DENSITY = {frac_nonzero:.3%}",
                )

            if layout.logits_table_cfg is not None:
                # Get logits table data
                top_logits = TopK(
                    logit_vector, k=layout.logits_table_cfg.n_rows, largest=True
                )
                bottom_logits = TopK(
                    logit_vector, k=layout.logits_table_cfg.n_rows, largest=False
                )

                top_logits, top_token_ids = (
                    top_logits.values.tolist(),
                    top_logits.indices.tolist(),
                )
                bottom_logits, bottom_token_ids = (
                    bottom_logits.values.tolist(),
                    bottom_logits.indices.tolist(),
                )

                # Create a MiddlePlotsData object from this, and add it to the dict
                feature_data_dict[feat].logits_table_data = LogitsTableData(
                    bottom_logits=bottom_logits,
                    bottom_token_ids=bottom_token_ids,
                    top_logits=top_logits,
                    top_token_ids=top_token_ids,
                )

    time_logs["(5) Getting data for histograms"] = time.time() - t0
    t0 = time.time()

    # ! Calculate all data for the right-hand visualisations, i.e. the sequences

    if layout.seq_cfg is not None:
        for i, feat in enumerate(feature_indices):
            # Add this feature's sequence data to the list
            feature_data_dict[feat].sequence_data = get_sequences_data(
                tokens=tokens,
                feat_acts=all_feat_acts[..., i],
                feat_logits=logits[i],
                resid_post=all_resid_post,
                feature_resid_dir=feature_resid_dir[i],
                W_U=W_U,
                seq_cfg=layout.seq_cfg,
            )
            # Update the 2nd progress bar (fwd passes & getting sequence data dominates the runtime of these computations)
            if progress is not None:
                progress[1].update(1)

    time_logs["(6) Getting data for sequences"] = time.time() - t0
    t0 = time.time()

    # ! Get stats (including quantiles, which will be useful for the prompt-centric visualisation)
    feature_stats = FeatureStatistics.create(
        data=einops.rearrange(all_feat_acts, "b s feats -> feats (b s)")
    )
    time_logs["(7) Getting data for quantiles"] = time.time() - t0
    t0 = time.time()

    # ! Return the output, as a dict of FeatureData items
    sae_vis_data = SaeVisData(feature_data_dict, feature_stats, cfg)
    return sae_vis_data, time_logs


@torch.inference_mode()
def _get_feature_data(
    encoder: AutoEncoder,
    encoder_B: AutoEncoder | None,
    model: TransformerLensWrapper,
    tokens: Int[Tensor, "batch seq"],
    feature_indices: int | list[int],
    cfg: SaeVisConfig,
    progress: list[tqdm] | None = None,
) -> tuple[SaeVisData, dict[str, float]]:
    """
    Gets data that will be used to create the sequences in the feature-centric HTML visualisation.

    Note - this function isn't called directly by the user, it actually gets called by the `get_feature_data` function
    which does exactly the same thing except it also batches this computation by features (in accordance with the
    arguments `features` and `minibatch_size_features` from the SaeVisConfig object).

    Args:
        encoder: AutoEncoder
            The encoder whose features we'll be analyzing.

        encoder_B: AutoEncoder
            The encoder we'll be using as a reference (i.e. finding the B-features with the highest correlation). This
            is only necessary if we're generating the left-hand tables (i.e. cfg.include_left_tables=True).

        model: TransformerLensWrapper
            The model we'll be using to get the feature activations. It's a wrapping of the base TransformerLens model.

        tokens: Int[Tensor, "batch seq"]
            The tokens we'll be using to get the feature activations.

        feature_indices: Union[int, list[int]]
            The features we're actually computing. These might just be a subset of the model's full features.

        cfg: SaeVisConfig
            Feature visualization parameters, containing a bunch of other stuff. See the SaeVisConfig docstring for
            more information.

        progress: Optional[list[tqdm]]
            An optional list containing progress bars for the forward passes and the sequence data. This is used to
            update the progress bars as the computation runs.

    Returns:
        sae_vis_data: SaeVisData
            Containing data for creating each feature visualization, as well as data for rank-ordering the feature
            visualizations when it comes time to make the prompt-centric view (the `feature_act_quantiles` attribute).

        time_log: dict[str, float]
            A dictionary containing the time taken for each step of the computation. This is optionally printed at the
            end of the `get_feature_data` function, if `cfg.verbose` is set to True.
    """
    # ! Boring setup code
    time_logs = {
        "(1) Initialization": 0.0,
        "(2) Forward passes to gather model activations": 0.0,
        "(3) Computing feature acts from model acts": 0.0,
    }

    t0 = time.time()

    # Make feature_indices a list, for convenience
    if isinstance(feature_indices, int):
        feature_indices = [feature_indices]

    # Get tokens into minibatches, for the fwd pass
    token_minibatches = (
        (tokens,)
        if cfg.minibatch_size_tokens is None
        else tokens.split(cfg.minibatch_size_tokens)
    )
    token_minibatches = [tok.to(device) for tok in token_minibatches]

    # ! Data setup code (defining the main objects we'll eventually return, for each of 5 possible vis components)

    # Create lists to store the feature activations & final values of the residual stream
    all_resid_post = []
    all_feat_acts = []

    # Create objects to store the data for computing rolling stats
    corrcoef_neurons = RollingCorrCoef()
    corrcoef_encoder = RollingCorrCoef(indices=feature_indices, with_self=True)
    corrcoef_encoder_B = RollingCorrCoef() if encoder_B is not None else None

    # Get encoder & decoder directions
    feature_out_dir = encoder.W_dec[feature_indices]  # [feats d_autoencoder]
    feature_resid_dir = to_resid_dir(feature_out_dir, model)  # [feats d_model]

    time_logs["(1) Initialization"] = time.time() - t0

    # ! Compute & concatenate together all feature activations & post-activation function values

    for minibatch in token_minibatches:
        # Fwd pass, get model activations
        t0 = time.time()
        residual, model_acts = model.forward(minibatch, return_logits=False)
        time_logs["(2) Forward passes to gather model activations"] += time.time() - t0

        # Compute feature activations from this
        t0 = time.time()
        feat_acts = compute_feat_acts(
            model_acts=model_acts,
            feature_idx=feature_indices,
            encoder=encoder,
            encoder_B=encoder_B,
            corrcoef_neurons=corrcoef_neurons,
            corrcoef_encoder=corrcoef_encoder,
            corrcoef_encoder_B=corrcoef_encoder_B,
        )
        time_logs["(3) Computing feature acts from model acts"] += time.time() - t0

        # Add these to the lists (we'll eventually concat)
        all_feat_acts.append(feat_acts)
        all_resid_post.append(residual)

        # Update the 1st progress bar (fwd passes & getting sequence data dominates the runtime of these computations)
        if progress is not None:
            progress[0].update(1)

    all_feat_acts = torch.cat(all_feat_acts, dim=0)
    all_resid_post = torch.cat(all_resid_post, dim=0)

    # ! Use the data we've collected to make a MultiFeatureData object
    sae_vis_data, _time_logs = parse_feature_data(
        tokens=tokens,
        feature_indices=feature_indices,
        all_feat_acts=all_feat_acts,
        feature_resid_dir=feature_resid_dir,
        all_resid_post=all_resid_post,
        W_U=model.W_U,
        cfg=cfg,
        feature_out_dir=feature_out_dir,
        corrcoef_neurons=corrcoef_neurons,
        corrcoef_encoder=corrcoef_encoder,
        corrcoef_encoder_B=corrcoef_encoder_B,
        progress=progress,
    )

    assert (
        set(time_logs.keys()) & set(_time_logs.keys()) == set()
    ), f"Invalid keys: {set(time_logs.keys()) & set(_time_logs.keys())} should have zero overlap"

    time_logs.update(_time_logs)

    return sae_vis_data, time_logs


@torch.inference_mode()
def get_feature_data(
    encoder: AutoEncoder,
    model: HookedTransformer,
    tokens: Int[Tensor, "batch seq"],
    cfg: SaeVisConfig,
    encoder_B: AutoEncoder | None = None,
) -> SaeVisData:
    """
    This is the main function which users will run to generate the feature visualization data. It batches this
    computation over features, in accordance with the arguments in the SaeVisConfig object (we don't want to compute all
    the features at once, since might give OOMs).

    See the `_get_feature_data` function for an explanation of the arguments, as well as a more detailed explanation
    of what this function is doing.

    The return object is the merged SaeVisData objects returned by the `_get_feature_data` function.
    """
    # Apply random seed
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    # Create objects to store all the data we'll get from `_get_feature_data`
    sae_vis_data = SaeVisData()
    time_logs = defaultdict(float)

    # Slice tokens, if we're only doing a subset of them
    if cfg.batch_size is None:
        tokens = tokens
    else:
        tokens = tokens[: cfg.batch_size]

    # Get a feature list (need to deal with the case where `cfg.features` is an int, or None)
    if cfg.features is None:
        assert isinstance(encoder.cfg.d_hidden, int)
        features_list = list(range(encoder.cfg.d_hidden))
    elif isinstance(cfg.features, int):
        features_list = [cfg.features]
    else:
        features_list = list(cfg.features)

    # Break up the features into batches
    feature_batches = [
        x.tolist()
        for x in torch.tensor(features_list).split(cfg.minibatch_size_features)
    ]
    # Calculate how many minibatches of tokens there will be (for the progress bar)
    n_token_batches = (
        1
        if (cfg.minibatch_size_tokens is None)
        else math.ceil(len(tokens) / cfg.minibatch_size_tokens)
    )
    # Get the denominator for each of the 2 progress bars
    totals = (n_token_batches * len(feature_batches), len(features_list))

    # Optionally add two progress bars (one for the forward passes, one for getting the sequence data)
    if cfg.verbose:
        progress = [
            tqdm(total=totals[0], desc="Forward passes to cache data for vis"),
            tqdm(total=totals[1], desc="Extracting vis data from cached data"),
        ]
    else:
        progress = None

    # If the model is from TransformerLens, we need to apply a wrapper to it for standardization
    assert isinstance(
        model, HookedTransformer
    ), "Error: non-HookedTransformer models are not yet supported."
    assert isinstance(cfg.hook_point, str), "Error: cfg.hook_point must be a string"
    model_wrapper = TransformerLensWrapper(model, cfg.hook_point)

    # For each batch of features: get new data and update global data storage objects
    for features in feature_batches:
        new_feature_data, new_time_logs = _get_feature_data(
            encoder, encoder_B, model_wrapper, tokens, features, cfg, progress
        )
        sae_vis_data.update(new_feature_data)
        for key, value in new_time_logs.items():
            time_logs[key] += value

    # Now exited, make sure the progress bar is at 100%
    if progress is not None:
        for pbar in progress:
            pbar.n = pbar.total

    # If verbose, then print the output
    if cfg.verbose:
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
    seq_cfg: SequencesConfig,
) -> SequenceMultiGroupData:
    """
    This function returns the data which is used to create the sequence visualizations (i.e. the right-hand column of
    the visualization). This is a multi-step process (the 4 steps are annotated in the code):

        (1) Find all the token groups (i.e. topk, bottomk, and quantile groups of activations). These are bold tokens.
        (2) Get the indices of all tokens we'll need data from, which includes a buffer around each bold token.
        (3) Extract the token IDs, feature activations & residual stream values for those positions
        (4) Compute the logit effect if this feature is ablated
            (4A) Use this to compute the most affected tokens by this feature (i.e. the vis hoverdata)
            (4B) Use this to compute the loss effect if this feature is ablated (i.e. the blue/red underlining)
        (5) Return all this data as a SequenceMultiGroupData object

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
        SequenceMultiGroupData
            This is a dataclass which contains a dict of SequenceGroupData objects, where each SequenceGroupData object
            contains the data for a particular group of sequences (i.e. the top-k, bottom-k, and the quantile groups).
    """

    # ! (1) Find the tokens from each group

    # Get buffer, s.t. we're looking for bold tokens in the range `buffer[0] : buffer[1]`. For each bold token, we need
    # to see `seq_cfg.buffer[0]+1` behind it (plus 1 because we need the prev token to compute loss effect), and we need
    # to see `seq_cfg.buffer[1]` ahead of it.
    buffer = (
        (seq_cfg.buffer[0] + 1, -seq_cfg.buffer[1])
        if seq_cfg.buffer is not None
        else None
    )
    _batch_size, seq_length = tokens.shape
    padded_buffer_width = (
        seq_cfg.buffer[0] + seq_cfg.buffer[1] + 2
        if seq_cfg.buffer is not None
        else seq_length
    )

    # Get the top-activating tokens
    indices = k_largest_indices(feat_acts, k=seq_cfg.top_acts_group_size, buffer=buffer)
    indices_dict = {f"TOP ACTIVATIONS<br>MAX = {feat_acts.max():.3f}": indices}

    # Get all possible indices. Note, we need to be able to look 1 back (feature activation on prev token is needed for
    # computing loss effect on this token)
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
                f"INTERVAL {lower:.3f} - {upper:.3f}<br>CONTAINS {pct:.3%}"
            ] = indices

    # Concat all the indices together (in the next steps we do all groups at once). Shape of this object is [n_bold 2],
    # i.e. the [i, :]-th element are the batch and sequence dimensions for the i-th bold token.
    indices_bold = torch.concat(list(indices_dict.values())).cpu()
    n_bold = indices_bold.shape[0]

    # ! (2) Get the buffer indices

    if seq_cfg.buffer is not None:
        # Get the buffer indices, by adding a broadcasted arange object. At this point, indices_buf contains 1 more token
        # than the length of the sequences we'll see (because it also contains the token before the sequence starts).
        buffer_tensor = torch.arange(
            -seq_cfg.buffer[0] - 1, seq_cfg.buffer[1] + 1, device=indices_bold.device
        )
        indices_buf = einops.repeat(
            indices_bold,
            "n_bold two -> n_bold seq two",
            seq=seq_cfg.buffer[0] + seq_cfg.buffer[1] + 2,
        )
        indices_buf = torch.stack(
            [indices_buf[..., 0], indices_buf[..., 1] + buffer_tensor], dim=-1
        )
    else:
        # If we don't specify a sequence, then do all of the indices.
        indices_buf = torch.stack(
            [
                einops.repeat(
                    indices_bold[:, 0], "n_bold -> n_bold seq", seq=seq_length
                ),  # batch indices of bold tokens
                einops.repeat(
                    torch.arange(seq_length), "seq -> n_bold seq", n_bold=n_bold
                ),  # all sequence indices
            ],
            dim=-1,
        )

    assert indices_buf.shape == (n_bold, padded_buffer_width, 2)

    # ! (3) Extract the token IDs, feature activations & residual stream values for those positions

    # Get the tokens which will be in our sequences
    token_ids = eindex(
        tokens, indices_buf[:, 1:], "[n_bold seq 0] [n_bold seq 1]"
    )  # shape [batch buf]

    # Now, we split into cases depending on whether we're computing the buffer or not. One kinda weird thing: we get
    # feature acts for 2 different reasons (token coloring & ablation), and in the case where we're computing the buffer
    # we need [:, 1:] for coloring and [:, :-1] for ablation, but when we're not we only need [:, bold] for both. So
    # we split on cases here.
    if seq_cfg.compute_buffer:
        feat_acts_buf = eindex(
            feat_acts,
            indices_buf,
            "[n_bold buf_plus1 0] [n_bold buf_plus1 1] -> n_bold buf_plus1",
        )
        feat_acts_pre_ablation = feat_acts_buf[:, :-1]
        feat_acts_coloring = feat_acts_buf[:, 1:]
        resid_post_pre_ablation = eindex(
            resid_post, indices_buf[:, :-1], "[n_bold buf 0] [n_bold buf 1] d_model"
        )
        # The tokens we'll use to index correct logits are the same as the ones which will be in our sequence
        correct_tokens = token_ids
    else:
        feat_acts_pre_ablation = eindex(
            feat_acts, indices_bold, "[n_bold 0] [n_bold 1]"
        ).unsqueeze(1)
        feat_acts_coloring = feat_acts_pre_ablation
        resid_post_pre_ablation = eindex(
            resid_post, indices_bold, "[n_bold 0] [n_bold 1] d_model"
        ).unsqueeze(1)
        # The tokens we'll use to index correct logits are the ones after bold
        indices_bold_next = torch.stack(
            [indices_bold[:, 0], indices_bold[:, 1] + 1], dim=-1
        )
        correct_tokens = eindex(
            tokens, indices_bold_next, "[n_bold 0] [n_bold 1]"
        ).unsqueeze(1)

    # ! (4) Compute the logit effect if this feature is ablated

    # Get this feature's output vector, using an outer product over the feature activations for all tokens
    resid_post_feature_effect = (
        feat_acts_pre_ablation[..., None] * feature_resid_dir
    )  # shape [batch buf d_model]

    # Do the ablations, and get difference in logprobs
    new_resid_post = resid_post_pre_ablation - resid_post_feature_effect
    new_logits = (new_resid_post / new_resid_post.std(dim=-1, keepdim=True)) @ W_U
    orig_logits = (
        resid_post_pre_ablation / resid_post_pre_ablation.std(dim=-1, keepdim=True)
    ) @ W_U
    contribution_to_logprobs = orig_logits.log_softmax(dim=-1) - new_logits.log_softmax(
        dim=-1
    )

    # ! (4A) Use this to compute the most affected tokens by this feature
    # The TopK function can improve efficiency by masking the features which are zero
    acts_nonzero = feat_acts_pre_ablation.abs() > 1e-5  # shape [batch buf]
    top_contribution_to_logits = TopK(
        contribution_to_logprobs,
        k=seq_cfg.top_logits_hoverdata,
        largest=True,
        tensor_mask=acts_nonzero,
    )
    bottom_contribution_to_logits = TopK(
        contribution_to_logprobs,
        k=seq_cfg.top_logits_hoverdata,
        largest=False,
        tensor_mask=acts_nonzero,
    )

    # ! (4B) Use this to compute the loss effect if this feature is ablated
    # which is just the negative of the change in logprobs
    loss_contribution = eindex(
        -contribution_to_logprobs, correct_tokens, "batch seq [batch seq]"
    )

    # ! (5) Store the results in a SequenceMultiGroupData object

    # Now that we've indexed everything, construct the batch of SequenceData objects
    sequence_groups_data = []
    group_sizes_cumsum = np.cumsum(
        [0] + [len(indices) for indices in indices_dict.values()]
    ).tolist()
    for group_idx, group_name in enumerate(indices_dict.keys()):
        seq_data = [
            SequenceData(
                token_ids=token_ids[i].tolist(),
                feat_acts=[round(f, 4) for f in feat_acts_coloring[i].tolist()],
                loss_contribution=loss_contribution[i].tolist(),
                token_logits=feat_logits[token_ids[i]].tolist(),
                top_token_ids=top_contribution_to_logits.indices[i].tolist(),
                top_logits=top_contribution_to_logits.values[i].tolist(),
                bottom_token_ids=bottom_contribution_to_logits.indices[i].tolist(),
                bottom_logits=bottom_contribution_to_logits.values[i].tolist(),
            )
            for i in range(
                group_sizes_cumsum[group_idx], group_sizes_cumsum[group_idx + 1]
            )
        ]
        sequence_groups_data.append(SequenceGroupData(group_name, seq_data))

    return SequenceMultiGroupData(sequence_groups_data)


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
) -> dict[str, tuple[list[int], list[str]]]:
    """
    Gets data needed to create the sequences in the prompt-centric vis (displaying dashboards for the most relevant
    features on a prompt).

    This function exists so that prompt dashboards can be generated without using our AutoEncoder or
    TransformerLens(Wrapper) classes.

    Args:
        tokens: Int[Tensor, "batch seq"]
            The tokens we'll be using to get the feature activations. Note that we might not be using all of them; the
            number used is determined by `fvp.total_batch_size`.

        str_toks:  list[str]
            The tokens as a list of strings, so that they can be visualized in HTML.

        sae_vis_data: SaeVisData
             The object storing all data for each feature. We'll set each `feature_data.prompt_data` to the
             data we get from `prompt`.

        feat_acts: Float[Tensor, "seq feats"]
            The activations values of the features across the sequence.

        feature_resid_dir: Float[Tensor, "feats d_model"]
            The directions that each feature writes to the residual stream.

        resid_post: Float[Tensor, "seq d_model"]
            The activations of the final layer of the model before the unembed.

        W_U: Float[Tensor, "d_model d_vocab"]
            The model's unembed weights for the logit lens.

        feature_idx: list[int] or None
            The features we're actually computing. These might just be a subset of the model's full features.

        num_top_features: int
            The number of top features to display in this view, for any given metric.

    Returns:
        scores_dict: dict[str, tuple[list[int], list[str]]]
            A dictionary mapping keys like "act_quantile|'django' (0)" to a tuple of lists, where the first list is the
            feature indices, and the second list is the string-formatted values of the scores.

    As well as returning this dictionary, this function will also set `FeatureData.prompt_data` for each feature in
    `sae_vis_data` (this is necessary for getting the prompts in the prompt-centric vis). Note this design choice could
    have been done differently (i.e. have this function return a list of the prompt data for each feature). I chose this
    way because it means the FeatureData._get_html_data_prompt_centric can work fundamentally the same way as
    FeatureData._get_html_data_feature_centric, rather than treating the prompt data object as a different kind of
    component in the vis.
    """
    if feature_idx is None:
        feature_idx = list(sae_vis_data.feature_data_dict.keys())
    n_feats = len(feature_idx)
    assert (
        feature_resid_dir.shape[0] == n_feats
    ), f"The number of features in feature_resid_dir ({feature_resid_dir.shape[0]}) does not match the number of feature indices ({n_feats})"

    assert (
        feat_acts.shape[1] == n_feats
    ), f"The number of features in feat_acts ({feat_acts.shape[1]}) does not match the number of feature indices ({n_feats})"

    feats_loss_contribution = torch.empty(
        size=(n_feats, tokens.shape[1] - 1), device=device
    )
    # Some logit computations which we only need to do once
    # correct_token_unembeddings = model_wrapped.W_U[:, tokens[0, 1:]] # [d_model seq]
    orig_logits = (
        resid_post / resid_post.std(dim=-1, keepdim=True)
    ) @ W_U  # [seq d_vocab]
    raw_logits = feature_resid_dir @ W_U  # [feats d_vocab]

    for i, feat in enumerate(feature_idx):
        # ! Calculate the sequence data for each feature, and store it as FeatureData.prompt_data

        # Get this feature's output vector, using an outer product over the feature activations for all tokens
        resid_post_feature_effect = einops.einsum(
            feat_acts[:, i], feature_resid_dir[i], "seq, d_model -> seq d_model"
        )

        # Ablate the output vector from the residual stream, and get logits post-ablation
        new_resid_post = resid_post - resid_post_feature_effect
        new_logits = (new_resid_post / new_resid_post.std(dim=-1, keepdim=True)) @ W_U

        # Get the top5 & bottom5 changes in logits (don't bother with `efficient_topk` cause it's small)
        contribution_to_logprobs = orig_logits.log_softmax(
            dim=-1
        ) - new_logits.log_softmax(dim=-1)
        top_contribution_to_logits = TopK(contribution_to_logprobs[:-1], k=5)
        bottom_contribution_to_logits = TopK(
            contribution_to_logprobs[:-1], k=5, largest=False
        )

        # Get the change in loss (which is negative of change of logprobs for correct token)
        loss_contribution = eindex(
            -contribution_to_logprobs[:-1], tokens[0, 1:], "seq [seq]"
        )
        feats_loss_contribution[i, :] = loss_contribution

        # Store the sequence data
        sae_vis_data.feature_data_dict[feat].prompt_data = SequenceData(
            token_ids=tokens.squeeze(0).tolist(),
            feat_acts=[round(f, 4) for f in feat_acts[:, i].tolist()],
            loss_contribution=[0.0] + loss_contribution.tolist(),
            token_logits=raw_logits[i, tokens.squeeze(0)].tolist(),
            top_token_ids=top_contribution_to_logits.indices.tolist(),
            top_logits=top_contribution_to_logits.values.tolist(),
            bottom_token_ids=bottom_contribution_to_logits.indices.tolist(),
            bottom_logits=bottom_contribution_to_logits.values.tolist(),
        )

    # ! Lastly, return a dictionary mapping each key like 'act_quantile|"django" (0)' to a list of feature indices & scores

    # Get a dict with keys like f"act_quantile|'My' (1)" and values (feature indices list, feature score values list)
    scores_dict: dict[str, tuple[list[int], list[str]]] = {}

    for seq_pos, seq_key in enumerate([f"{t!r} ({i})" for i, t in enumerate(str_toks)]):
        # Filter the feature activations, since we only need the ones that are non-zero
        feat_acts_nonzero_filter = utils.to_numpy(feat_acts[seq_pos] > 0)
        feat_acts_nonzero_locations = np.nonzero(feat_acts_nonzero_filter)[0].tolist()
        _feat_acts = feat_acts[seq_pos, feat_acts_nonzero_filter]  # [feats_filtered,]
        _feature_idx = np.array(feature_idx)[feat_acts_nonzero_filter]

        if feat_acts_nonzero_filter.sum() > 0:
            k = min(num_top_features, _feat_acts.numel())

            # Get the top features by activation size. This is just applying a TopK function to the feat acts (which
            # were stored by the code before this). The feat acts are formatted to 3dp.
            act_size_topk = TopK(_feat_acts, k=k, largest=True)
            top_features = _feature_idx[act_size_topk.indices].tolist()
            formatted_scores = [f"{v:.3f}" for v in act_size_topk.values]
            scores_dict[f"act_size|{seq_key}"] = (top_features, formatted_scores)

            # Get the top features by activation quantile. We do this using the `feature_act_quantiles` object, which
            # was stored `sae_vis_data`. This quantiles object has a method to return quantiles for a given set of
            # data, as well as the precision (we make the precision higher for quantiles closer to 100%, because these
            # are usually the quantiles we're interested in, and it lets us to save space in `feature_act_quantiles`).
            act_quantile, act_precision = sae_vis_data.feature_stats.get_quantile(
                _feat_acts, feat_acts_nonzero_locations
            )
            act_quantile_topk = TopK(act_quantile, k=k, largest=True)
            act_formatting = [
                f".{act_precision[i]-2}%" for i in act_quantile_topk.indices
            ]
            top_features = _feature_idx[act_quantile_topk.indices].tolist()
            formatted_scores = [
                f"{v:{f}}" for v, f in zip(act_quantile_topk.values, act_formatting)
            ]
            scores_dict[f"act_quantile|{seq_key}"] = (top_features, formatted_scores)

        # We don't measure loss effect on the first token
        if seq_pos == 0:
            continue

        # Filter the loss effects, since we only need the ones which have non-zero feature acts on the tokens before them
        prev_feat_acts_nonzero_filter = utils.to_numpy(feat_acts[seq_pos - 1] > 0)
        _loss_contribution = feats_loss_contribution[
            prev_feat_acts_nonzero_filter, seq_pos - 1
        ]  # [feats_filtered,]
        _feature_idx_prev = np.array(feature_idx)[prev_feat_acts_nonzero_filter]

        if prev_feat_acts_nonzero_filter.sum() > 0:
            k = min(num_top_features, _loss_contribution.numel())

            # Get the top features by loss effect. This is just applying a TopK function to the loss effects (which were
            # stored by the code before this). The loss effects are formatted to 3dp. We look for the most negative
            # values, i.e. the most loss-reducing features.
            loss_contribution_topk = TopK(_loss_contribution, k=k, largest=False)
            top_features = _feature_idx_prev[loss_contribution_topk.indices].tolist()
            formatted_scores = [f"{v:+.3f}" for v in loss_contribution_topk.values]
            scores_dict[f"loss_effect|{seq_key}"] = (top_features, formatted_scores)
    return scores_dict


@torch.inference_mode()
def get_prompt_data(
    sae_vis_data: SaeVisData,
    prompt: str,
    num_top_features: int,
) -> dict[str, tuple[list[int], list[str]]]:
    """
    Gets data that will be used to create the sequences in the prompt-centric HTML visualisation, i.e. an object of
    type SequenceData for each of our features.

    Args:
        sae_vis_data:     The object storing all data for each feature. We'll set each `feature_data.prompt_data` to the
                          data we get from `prompt`.
        prompt:           The prompt we'll be using to get the feature activations.#
        num_top_features: The number of top features we'll be getting data for.

    Returns:
        scores_dict:      A dictionary mapping keys like "act_quantile|0" to a tuple of lists, where the first list is
                          the feature indices, and the second list is the string-formatted values of the scores.

    As well as returning this dictionary, this function will also set `FeatureData.prompt_data` for each feature in
    `sae_vis_data`. This is because the prompt-centric vis will call `FeatureData._get_html_data_prompt_centric` on each
    feature data object, so it's useful to have all the data in once place! Even if this will get overwritten next
    time we call `get_prompt_data` for this same `sae_vis_data` object.
    """

    # ! Boring setup code
    feature_idx = list(sae_vis_data.feature_data_dict.keys())
    encoder = sae_vis_data.encoder
    assert isinstance(encoder, AutoEncoder)
    model = sae_vis_data.model
    assert isinstance(model, HookedTransformer)
    cfg = sae_vis_data.cfg
    assert isinstance(cfg.hook_point, str), f"{cfg.hook_point=}, expected a string"

    str_toks: list[str] = model.tokenizer.tokenize(prompt)  # type: ignore
    tokens = model.tokenizer.encode(prompt, return_tensors="pt").to(device)  # type: ignore
    assert isinstance(tokens, torch.Tensor)

    model_wrapped = TransformerLensWrapper(model, cfg.hook_point)

    feature_act_dir = encoder.W_enc[:, feature_idx]  # [d_in feats]
    feature_out_dir = encoder.W_dec[feature_idx]  # [feats d_in]
    feature_resid_dir = to_resid_dir(feature_out_dir, model_wrapped)  # [feats d_model]
    assert (
        feature_act_dir.T.shape
        == feature_out_dir.shape
        == (len(feature_idx), encoder.cfg.d_in)
    )

    # ! Define hook functions to cache all the info required for feature ablation, then run those hook fns

    resid_post, act_post = model_wrapped(tokens, return_logits=False)
    resid_post: Tensor = resid_post.squeeze(0)
    feat_acts = compute_feat_acts(act_post, feature_idx, encoder).squeeze(
        0
    )  # [seq feats]

    # ! Use the data we've collected to make the scores_dict and update the sae_vis_data
    scores_dict = parse_prompt_data(
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

    return scores_dict
