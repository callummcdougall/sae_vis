import time
import gc
import numpy as np
from pathlib import Path
from typing import List
import torch
from torch import Tensor
from eindex import eindex
from typing import Optional, List, Dict, Tuple, Union
import torch.nn.functional as F
import einops
from jaxtyping import Float, Int
from tqdm import tqdm
from functools import partial
import os
from IPython.display import clear_output
from dataclasses import dataclass
from rich import print as rprint
from rich.table import Table

Arr = np.ndarray

from sae_vis.model_fns import AutoEncoder, DemoTransformer
from sae_vis.utils_fns import (
    k_largest_indices,
    random_range_indices,
    create_vocab_dict,
    QuantileCalculator,
    TopK,
    efficient_topk,
)
from sae_vis.data_storing_fns import (
    FeatureVizParams,
    BatchedCorrCoef,
    SequenceData,
    SequenceGroupData,
    SequenceMultiGroupData,
    LeftTablesData,
    MiddlePlotsData,
    FeatureData,
    MultiFeatureData,
    HistogramData,
    PromptData,
    MultiPromptData,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")








def compute_feat_acts(
    act_post: Float[Tensor, "batch seq d_mlp"],
    feature_idx: Int[Tensor, "feats"],
    encoder: AutoEncoder,
    encoder_B: Optional[AutoEncoder] = None,
    corrcoef_neurons: Optional[BatchedCorrCoef] = None,
    corrcoef_encoder_B: Optional[BatchedCorrCoef] = None,
) -> Float[Tensor, "batch seq feats"]:
    '''
    This function computes the feature activations, given a bunch of model data. It also updates the rolling correlation
    coefficient objects, if they're given.

    Args:
        act_post: Float[Tensor, "batch seq d_mlp"]
            The neuron values post-activation function.
        feature_act_dir: Float[Tensor, "d_mlp feats"]
            The SAE's encoder weights for the feature(s) which we're interested in.
        feature_bias: Float[Tensor, "feats"]
            The bias of the encoder, which we add to the feature activations before ReLU'ing.
        encoder: AutoEncoder
            The encoder object, which we use to calculate the feature activations.
        encoder_B: Optional[AutoEncoder]
            The encoder-B object, which we use to calculate the feature activations.
        corrcoef_neurons: Optional[BatchedCorrCoef]
            The object which stores the rolling correlation coefficients between feature activations & neurons.
        corrcoef_encoder_B: Optional[BatchedCorrCoef]
            The object which stores the rolling correlation coefficients between feature activations & encoder-B features.
    '''
    # Get the feature act direction by indexing encoder.W_enc, and the bias by indexing encoder.b_enc
    feature_act_dir = encoder.W_enc[:, feature_idx] # (d_mlp, feats)
    feature_bias = encoder.b_enc[feature_idx] # (feats,)

    # Calculate & store the feature activations (we need to store them so we can get the right-hand visualisations later)
    x_cent = act_post - encoder.b_dec
    feat_acts_pre = einops.einsum(x_cent, feature_act_dir, "batch seq d_mlp, d_mlp feats -> batch seq feats")
    feat_acts = F.relu(feat_acts_pre + feature_bias)

    # Update the CorrCoef object between feature activation & neurons
    if corrcoef_neurons is not None:
        corrcoef_neurons.update(
            einops.rearrange(feat_acts, "batch seq feats -> feats (batch seq)"),
            einops.rearrange(act_post, "batch seq d_mlp -> d_mlp (batch seq)"),
        )
        
    # Calculate encoder-B feature activations (we don't need to store them, cause it's just for the left-hand visualisations)
    if (encoder_B is not None) and (corrcoef_encoder_B is not None):
        x_cent_B = act_post - encoder_B.b_dec
        feat_acts_pre_B = einops.einsum(x_cent_B, encoder_B.W_enc, "batch seq d_mlp, d_mlp d_hidden -> batch seq d_hidden")
        feat_acts_B = F.relu(feat_acts_pre_B + encoder.b_enc)

        # Update the CorrCoef object between feature activation & encoder-B features
        corrcoef_encoder_B.update(
            einops.rearrange(feat_acts, "batch seq feats -> feats (batch seq)"),
            einops.rearrange(feat_acts_B, "batch seq d_hidden -> d_hidden (batch seq)"),
        )

    return feat_acts




@torch.inference_mode()
def _get_feature_data(
    encoder: AutoEncoder,
    encoder_B: AutoEncoder,
    model: DemoTransformer,
    tokens: Int[Tensor, "batch seq"],
    feature_indices: Union[int, List[int]],
    fvp: FeatureVizParams,
) -> MultiFeatureData:
    '''
    Gets data that will be used to create the sequences in the feature-centric HTML visualisation.
    
    Note - this function isn't called directly by the user, it actually gets called by the `get_feature_data` function
    which does exactly the same thing except it also batches this computation by features (in accordance with the
    arguments `features` and `minibatch_size_features` from the FeatureVizParams object).

    Args:
        encoder: AutoEncoder
            The encoder whose features we'll be analyzing.
        encoder_B: AutoEncoder
            The encoder we'll be using as a reference (i.e. finding the B-features with the highest correlation).
        model: DemoTransformer
            The model we'll be using to get the feature activations.
        tokens: Int[Tensor, "batch seq"]
            The tokens we'll be using to get the feature activations. Note that we might not be using all of them; the
            number used is determined by `fvp.total_batch_size`.
        feature_indices: Union[int, List[int]]
            The features we're actually computing. These might just be a subset of the model's full features.
        fvp:
            Feature visualization parameters, containing a bunch of other stuff. See the FeatureVizParams docstring for
            more information.

    Returns object of class MultiFeatureData (i.e. containing data for creating each feature visualization, as well as
    data for rank-ordering the feature visualizations when it comes time to make the prompt-centric view).
    '''
    # ! Boring setup code

    t0 = time.time()

    # Make feature_indices a list, for convenience
    if isinstance(feature_indices, int): feature_indices = [feature_indices]
    n_feats = len(feature_indices)

    # Chunk the tokens, for less memory usage
    batch_size, seq_len = tokens.shape
    tokens = tokens[:fvp.total_batch_size]
    all_tokens = (tokens,) if fvp.minibatch_size is None else tokens.split(fvp.minibatch_size)
    all_tokens = [tok.to(device) for tok in all_tokens]

    # Get the vocab dict, which we'll use at the end
    vocab_dict = create_vocab_dict(model.tokenizer)

    # Helper function to create an iterator (if verbose we wrap it in a tqdm)
    def create_iterator(iterator, desc: Optional[str] = None):
        return tqdm(iterator, desc=desc, leave=False) if fvp.verbose else iterator
    
    # 
    if fvp.include_left_tables == False:
        encoder_B = None




    # ! Setup code (defining the main objects we'll eventually return)

    sequence_data_dict: Dict[int, SequenceMultiGroupData] = {}
    left_tables_data_dict: Dict[int, LeftTablesData] = {}
    middle_plots_data_dict: Dict[int, MiddlePlotsData] = {}

    # Create tensors which we'll concatenate our data to, as we generate it
    all_resid_post = []
    all_feat_acts = []

    # Create objects to store the rolling correlation coefficients
    corrcoef_neurons = BatchedCorrCoef()
    corrcoef_encoder_B = BatchedCorrCoef()

    # Get encoder & decoder directions
    feature_out_dir = encoder.W_dec[feature_indices] # (feats, d_mlp)
    feature_mlp_out_dir = feature_out_dir @ model.W_out[0] # (feats, d_model)

    t1 = time.time()



    # ! Compute & concatenate together all feature activations & post-activation function values

    iterator = create_iterator(all_tokens, desc="Running forward passes")
    
    for _tokens in iterator:
        residual, act_post = model(_tokens, return_logits=False)
        feat_acts = compute_feat_acts(act_post, feature_indices, encoder, encoder_B, corrcoef_neurons, corrcoef_encoder_B)
        all_feat_acts.append(feat_acts)
        all_resid_post.append(residual)
    all_feat_acts = torch.cat(all_feat_acts, dim=0)
    all_resid_post = torch.cat(all_resid_post, dim=0)

    t2 = time.time()



    # ! Calculate all data for the left-hand column visualisations, i.e. the 3 tables

    if fvp.include_left_tables:
        # Table 1: neuron alignment, based on decoder weights
        top3_neurons_aligned = TopK(feature_out_dir.topk(dim=-1, k=fvp.rows_in_left_tables, largest=True))
        pct_of_l1 = np.absolute(top3_neurons_aligned.values) / feature_out_dir.abs().sum(dim=-1, keepdim=True).cpu().numpy()
        # Table 2: neurons correlated with this feature, based on their activations
        pearson_topk_neurons, cossim_values_neurons = corrcoef_neurons.topk_pearson(k=fvp.rows_in_left_tables)
        # Table 3: encoder-B features correlated with this feature, based on their activations
        pearson_topk_encoderB, cossim_values_encoderB = corrcoef_encoder_B.topk_pearson(k=fvp.rows_in_left_tables)
        # Add all this data to the list of LeftTablesData objects
        for i, feat in enumerate(feature_indices):
            left_tables_data_dict[feat] = LeftTablesData(
                neuron_alignment = (top3_neurons_aligned[i], pct_of_l1[i]),
                neurons_correlated = (pearson_topk_neurons[i], cossim_values_neurons[i]),
                b_features_correlated = (pearson_topk_encoderB[i], cossim_values_encoderB[i]),
            )
    else:
        left_tables_data_dict = {feat: None for feat in feature_indices}

    t3 = time.time()



    # ! Calculate all data for the right-hand visualisations, i.e. the sequences

    iterator = create_iterator(list(enumerate(feature_indices)), desc="Getting sequence data")

    for i, feat in iterator:

        # Add this feature's sequence data to the list
        sequence_data_dict[feat] = get_sequences_data(
            tokens = tokens,
            feat_acts = all_feat_acts[..., i],
            resid_post = all_resid_post,
            feature_mlp_out_dir = feature_mlp_out_dir[i],
            W_U = model.W_U,
            fvp = fvp,
        )

    t4 = time.time()
    

    # ! Get all data for the middle column visualisations, i.e. the two histograms & the logit table

    # Get the logits of all features (i.e. the directions this feature writes to the logit output)
    logits = einops.einsum(feature_mlp_out_dir, model.W_U, "feats d_model, d_model d_vocab -> feats d_vocab")

    for i, feat in enumerate(feature_indices):
        _logits = logits[i]

        # Get data for logits (the histogram, and the table)
        logits_histogram_data = HistogramData(_logits, n_bins=40, tickmode="5 ticks")
        bottom10_logits = TopK(_logits.topk(k=10, largest=False))
        top10_logits = TopK(_logits.topk(k=10))

        # Get data for feature activations histogram (the title, and the histogram)
        feat_acts = all_feat_acts[..., i]
        nonzero_feat_acts = feat_acts[feat_acts > 0]
        frac_nonzero = nonzero_feat_acts.numel() / feat_acts.numel()
        freq_histogram_data = HistogramData(nonzero_feat_acts, n_bins=40, tickmode="ints")

        # Create a MiddlePlotsData object from this, and add it to the dict
        middle_plots_data_dict[feat] = MiddlePlotsData(
            bottom10_logits = bottom10_logits,
            top10_logits = top10_logits,
            logits_histogram_data = logits_histogram_data,
            freq_histogram_data = freq_histogram_data,
            frac_nonzero = frac_nonzero,
        )

    t5 = time.time()


    # ! Return the output, as a dict of FeatureData items

    feature_data = {
        feat: FeatureData(
            # Data-containing inputs (for the feature-centric visualisation)
            sequence_data = sequence_data_dict[feat],
            middle_plots_data = middle_plots_data_dict[feat],
            left_tables_data = left_tables_data_dict[feat],
            # Non data-containing inputs
            feature_idx = feat,
            vocab_dict = vocab_dict,
            fvp = fvp,
        )
        for feat in feature_indices
    }
    feature_act_quantiles = QuantileCalculator(data=einops.rearrange(all_feat_acts, "b s feats -> feats (b s)"))

    t6 = time.time()



    # ! If verbose, try to estimate time it will take to generate data for all features, plus storage space

    if fvp.verbose:
        n_feats_total = encoder.cfg.dict_mult * encoder.cfg.d_mlp
        total_time = t6 - t0
        table = Table("Task", "Time", "Pct %")
        for task, _time in zip(
            ["Setup code", "Fwd passes", "Left-hand tables", "Right-hand sequences", "Middle column", "Creating dict"],
            [t1-t0, t2-t1, t3-t2, t4-t3, t5-t4]
        ):
            frac = _time / total_time
            table.add_row(task, f"{_time:.2f}s", f"{frac:.1%}")
        rprint(table)
        est = ((t2 - t0) + (n_feats_total / n_feats) * (t6 - t2) / 60)
        print(f"Estimated time for all {n_feats_total} features = {est:.0f} minutes\n")


    return MultiFeatureData(feature_data, feature_act_quantiles)




@torch.inference_mode()
def get_feature_data(
    encoder: AutoEncoder,
    encoder_B: AutoEncoder,
    model: DemoTransformer,
    tokens: Int[Tensor, "batch seq"],
    fvp: FeatureVizParams,
) -> MultiFeatureData:
    '''
    This is the main function which users will run to generate the feature visualization data. It batches this
    computation over features, in accordance with the arguments in the FeatureVizParams object (we don't want to
    compute all the features at once, since might be too memory-intensive).

    See the `_get_feature_data` function for an explanation of the arguments, as well as a more detailed explanation
    of what this function is doing.
    '''
    # Create objects to store all the data we'll get from `_get_feature_data`
    feature_data = MultiFeatureData()

    # Get a feature list (need to deal with the case where `fvp.features` is an int, or None)
    if fvp.features is None:
        features_list = list(range(encoder.cfg.d_hidden))
    elif isinstance(fvp.features, int):
        features_list = list(range(fvp.features))
    else:
        features_list = list(fvp.features)

    # Break up the features into batches, and get data for each feature batch at once
    feature_indices_batches = [x.tolist() for x in torch.tensor(features_list).split(fvp.minibatch_size_features)]
    for feature_indices in feature_indices_batches:
        new_feature_data = _get_feature_data(encoder, encoder_B, model, tokens, feature_indices, fvp)
        feature_data.update(new_feature_data)

    return feature_data




@torch.inference_mode()
def get_sequences_data(
    tokens: Int[Tensor, "batch seq"],
    feat_acts: Float[Tensor, "batch seq"],
    resid_post: Float[Tensor, "batch seq d_model"],
    feature_mlp_out_dir: Float[Tensor, "d_model"],
    W_U: Float[Tensor, "d_model d_vocab"],
    fvp: FeatureVizParams,
) -> SequenceMultiGroupData:
    '''
    This function returns the data which is used to create the sequence visualizations (i.e. the right-hand column of
    the visualization). This is a multi-step process (the 4 steps are annotated in the code):

        (1) Find the tokens which will be in each group (i.e. the top-k activations, bottom-k, and the quantile groups).
        (2) Calculate the loss effect if this feature is ablated from resid_post (this is the blue/red underlined text).
        (3) Calculate the most-affected logits by this feature (i.e. the hoverdata in the visualization).
        (4) Return all this data as a SequenceMultiGroupData object

    Args:
        tokens:
            The tokens we'll be extracting sequence data from.
        feat_acts:
            The activations of the feature we're interested in, for each token in the batch.
        resid_post:
            The residual stream values before final layernorm, for each token in the batch.
        feature_mlp_out_dir:
            The direction this feature writes to the logit output (i.e. the direction we'll be erasing from resid_post).
        W_U:
            The model's unembedding matrix, which we'll use to get the logits.
        fvp:
            Feature visualization parameters, containing some important params e.g. num sequences per group.

    Returns:
        SequenceMultiGroupData
            This is a dataclass which contains a dict of SequenceGroupData objects, where each SequenceGroupData object
            contains the data for a particular group of sequences (i.e. the top-k, bottom-k, and the quantile groups).
    '''
    # ! (1) Find the tokens from each group

    indices_dict = {
        f"TOP ACTIVATIONS<br>MAX = {feat_acts.max():.3f}": k_largest_indices(feat_acts, k=fvp.first_group_size),
    }

    if fvp.n_groups > 0:
        quantiles = torch.linspace(0, feat_acts.max(), fvp.n_groups+1)
        for i in range(fvp.n_groups-1, -1, -1):
            lower, upper = quantiles[i:i+2]
            pct = ((feat_acts >= lower) & (feat_acts <= upper)).float().mean()
            indices = random_range_indices(feat_acts, k=fvp.other_groups_size, bounds=(lower, upper))
            indices_dict[f"INTERVAL {lower:.3f} - {upper:.3f}<br>CONTAINS {pct:.3%}"] = indices

    # Concat all the indices together (in the next steps we do all groups at once)
    indices_full = torch.concat(list(indices_dict.values())).cpu()


    # ! (2) & (3) Calculate the loss effect, and the most-affected logits

    # For each token index [batch, seq], we actually want [[batch, seq-buffer[0]], ..., [batch, seq], ..., [batch, seq+buffer[1]]]
    # We get one extra dimension at the start, because we need to see the effect on loss of the first token
    buffer_tensor = torch.arange(-fvp.buffer[0] - 1, fvp.buffer[1] + 1, device=indices_full.device)
    indices_full = einops.repeat(indices_full, "g two -> g buf two", buf=fvp.buffer[0] + fvp.buffer[1] + 2)
    indices_full = torch.stack([indices_full[..., 0], indices_full[..., 1] + buffer_tensor], dim=-1)

    # Use `eindex` for some delicate indexing operations, getting the data for each chosen sequence of tokens in each group
    tokens_group = eindex(tokens, indices_full[:, 1:], "[g buf 0] [g buf 1]") # [g buf]
    feat_acts_group = eindex(feat_acts, indices_full, "[g buf 0] [g buf 1]")
    resid_post_group: torch.Tensor = eindex(resid_post, indices_full, "[g buf 0] [g buf 1] d_model")

    # Get this feature's output vector, using an outer product over the feature activations for all tokens
    resid_post_feature_effect = einops.einsum(feat_acts_group, feature_mlp_out_dir, "g buf, d_model -> g buf d_model")

    # Get new ablated logits, and old ones
    new_resid_post = resid_post_group - resid_post_feature_effect
    new_logits = (new_resid_post / new_resid_post.std(dim=-1, keepdim=True)) @ W_U
    orig_logits = (resid_post_group / resid_post_group.std(dim=-1, keepdim=True)) @ W_U

    # Get the top5 & bottom5 changes in logits (using `efficient_topk` which saves time by ignoring tokens where feat_act=0)
    contribution_to_logprobs = orig_logits.log_softmax(dim=-1) - new_logits.log_softmax(dim=-1)
    acts_nonzero = feat_acts_group[:, :-1].abs() > 1e-5 # [g buf]
    top5_contribution_to_logits = efficient_topk(contribution_to_logprobs[:, :-1], acts_nonzero, k=5, largest=True)
    bottom5_contribution_to_logits = efficient_topk(contribution_to_logprobs[:, :-1], acts_nonzero, k=5, largest=False)

    # Get the change in loss (which is negative of change of logprobs for correct token)
    contribution_to_loss = eindex(-contribution_to_logprobs[:, :-1], tokens_group, "g buf [g buf]")


    # ! (4) Store the results in a SequenceMultiGroupData object

    # Now that we've indexed everything, construct the batch of SequenceData objects
    sequence_groups_data = []
    group_sizes_cumsum = np.cumsum([0] + [len(indices) for indices in indices_dict.values()]).tolist()
    for group_idx, group_name in enumerate(indices_dict.keys()):
        seq_data = [
            SequenceData(
                token_ids = tokens_group[i].tolist(),
                feat_acts = feat_acts_group[i, 1:].tolist(),
                contribution_to_loss = contribution_to_loss[i].tolist(),
                top5_token_ids = top5_contribution_to_logits.indices[i].tolist(),
                top5_logit_contributions = top5_contribution_to_logits.values[i].tolist(),
                bottom5_token_ids = bottom5_contribution_to_logits.indices[i].tolist(),
                bottom5_logit_contributions = bottom5_contribution_to_logits.values[i].tolist(),
            )
            for i in range(group_sizes_cumsum[group_idx], group_sizes_cumsum[group_idx+1])
        ]
        sequence_groups_data.append(SequenceGroupData(group_name, seq_data))

    return SequenceMultiGroupData(sequence_groups_data)





@torch.inference_mode()
def get_prompt_data(
    encoder: AutoEncoder,
    model: DemoTransformer,
    prompt: str,
    feature_data: MultiFeatureData,
    num_top_features: int,
    verbose: bool = False,
) -> MultiPromptData:
    '''
    Gets data that will be used to create the sequences in the prompt-centric HTML visualisation.
    
    Args:
        encoder: AutoEncoder
            The encoder whose features we'll be analyzing.
        model: DemoTransformer
            The model we'll be using to get the feature activations.
        tokens: Int[Tensor, "batch seq"]
            The tokens we'll be using to get the feature activations. Note that we might not be using all of them; the
            number used is determined by `fvp.total_batch_size`.
        feature_indices: Union[int, List[int]]
            The features we're actually computing. These might just be a subset of the model's full features.
        fvp:
            Feature visualization parameters, containing a bunch of other stuff. See the FeatureVizParams docstring for
            more information.

    Returns object of class MultiFeatureData (i.e. containing data for creating each feature visualization, as well as
    data for rank-ordering the feature visualizations when it comes time to make the prompt-centric view).


    Similar to get_feature_data, except it just gets the data relevant for a particular sequence (i.e. a custom one that
    the user inputs on their own). This means it ditches most of the complex indexing to get sequence groups, since we're
    only keeping a fraction of the 'full HTML'.

    feature_data is useful because we want to extract the top sequences for this feature, and display them underneath the
    column.
    '''

    # ! Boring setup code

    torch.cuda.empty_cache()

    feature_idx = list(feature_data.feature_data_dict.keys())
    n_feats = len(feature_idx)

    str_toks: List[str] = model.tokenizer.tokenize(prompt)
    tokens: torch.Tensor = model.tokenizer.encode(prompt, return_tensors="pt").to(device)
    batch, seq_len = tokens.shape

    def create_iterator(iterator, desc: Optional[str] = None):
        return tqdm(iterator, desc=desc, leave=False) if verbose else iterator

    sequence_data_dict: Dict[int, SequenceData] = {}

    feature_act_dir = encoder.W_enc[:, feature_idx] # [d_mlp feats]
    feature_out_dir = encoder.W_dec[feature_idx] # [feats d_mlp]
    feature_mlp_out_dir = feature_out_dir @ model.W_out[0] # [feats d_model]
    assert feature_act_dir.T.shape == feature_out_dir.shape == (len(feature_idx), encoder.cfg.d_mlp)

    # ! Define hook functions to cache all the info required for feature ablation, then run those hook fns

    resid_post, act_post = model(tokens, return_logits=False)
    resid_post: Tensor = resid_post.squeeze(0)
    feat_acts = compute_feat_acts(act_post, feature_idx, encoder).squeeze(0)

    feats_contribution_to_loss = torch.empty(size=(n_feats, seq_len-1), device=device)

    # Some logit computations which we only need to do once
    correct_token_unembeddings = model.W_U[:, tokens[0, 1:]] # [d_model seq]
    orig_logits = (resid_post / resid_post.std(dim=-1, keepdim=True)) @ model.W_U # [seq d_vocab]

    iterator = create_iterator(list(enumerate(feature_idx)))

    for i, feat in iterator:

        # ! Calculate all data for the sequences (this is the only truly 'new' bit of calculation we need to do)

        # Get this feature's output vector, using an outer product over the feature activations for all tokens
        resid_post_feature_effect = einops.einsum(feat_acts[:, i], feature_mlp_out_dir[i], "seq, d_model -> seq d_model")
        
        # Ablate the output vector from the residual stream, and get logits post-ablation
        new_resid_post = resid_post - resid_post_feature_effect
        new_logits = (new_resid_post / new_resid_post.std(dim=-1, keepdim=True)) @ model.W_U

        # Get the top5 & bottom5 changes in logits (don't bother with `efficient_topk` cause it's small)
        contribution_to_logprobs = orig_logits.log_softmax(dim=-1) - new_logits.log_softmax(dim=-1)
        top5_contribution_to_logits = TopK(contribution_to_logprobs[:-1].topk(k=5, dim=-1))
        bottom5_contribution_to_logits = TopK(contribution_to_logprobs[:-1].topk(k=5, dim=-1, largest=False))

        # Get the change in loss (which is negative of change of logprobs for correct token)
        contribution_to_loss = eindex(-contribution_to_logprobs[:-1], tokens[0, 1:], "seq [seq]")
        feats_contribution_to_loss[i, :] = contribution_to_loss

        # Store the sequence data
        sequence_data_dict[feat] = SequenceData(
            token_ids = tokens.squeeze(0).tolist(),
            feat_acts = feat_acts[:, i].tolist(),
            contribution_to_loss = [0.0] + contribution_to_loss.tolist(),
            top5_token_ids = top5_contribution_to_logits.indices.tolist(),
            top5_logit_contributions = top5_contribution_to_logits.values.tolist(),
            bottom5_token_ids = bottom5_contribution_to_logits.indices.tolist(),
            bottom5_logit_contributions = bottom5_contribution_to_logits.values.tolist(),
        )

        # Get the logits for the correct tokens
        logits_for_correct_tokens = einops.einsum(
            feature_mlp_out_dir[i], correct_token_unembeddings,
            "d_model, d_model seq -> seq"
        )

        # Add the annotations data (feature activations and logit effect) to the histograms
        freq_line_posn = feat_acts[:, i].tolist()
        freq_line_text = [f"\\'{str_tok}\\'<br>{act:.3f}" for str_tok, act in zip(str_toks[1:], freq_line_posn)]
        feature_data[feat].middle_plots_data.freq_histogram_data.line_posn = freq_line_posn
        feature_data[feat].middle_plots_data.freq_histogram_data.line_text = freq_line_text
        logits_line_posn = logits_for_correct_tokens.tolist()
        logits_line_text = [f"\\'{str_tok}\\'<br>{logits:.3f}" for str_tok, logits in zip(str_toks[1:], logits_line_posn)]
        feature_data[feat].middle_plots_data.logits_histogram_data.line_posn = logits_line_posn
        feature_data[feat].middle_plots_data.logits_histogram_data.line_text = logits_line_text



    # ! Lastly, use the 3 possible criteria (act size, act quantile, loss effect) to find all the top-scoring features

    # Construct a scores dict, which maps from things like ("act_quantile", seq_pos) to a list of the top-scoring features
    scores_dict: Dict[Tuple[str, str], TopK] = {}

    for seq_pos, str_tok in enumerate(str_toks):

        # Filter the feature activations, since we only need the ones that are non-zero
        feat_acts_nonzero_filter = (feat_acts[seq_pos] > 0).detach().cpu().numpy()
        feat_acts_nonzero_locations = np.nonzero(feat_acts_nonzero_filter)[0].tolist()
        _feat_acts = feat_acts[seq_pos, feat_acts_nonzero_filter] # [feats_filtered,]
        _feature_idx = np.array(feature_idx)[feat_acts_nonzero_filter]

        if feat_acts_nonzero_filter.sum() > 0:
            k = min(num_top_features, _feat_acts.numel())
            
            # Get the "act_size" scores (we return it as a TopK object)
            act_size_topk = TopK(_feat_acts.topk(k=k, largest=True))
            # Replace the indices with feature indices (these are different when feature_idx argument is not [0, 1, 2, ...])
            act_size_topk.indices[:] = _feature_idx[act_size_topk.indices]
            scores_dict[("act_size", seq_pos)] = (act_size_topk, ".3f")

            # Get the "act_quantile" scores, which is just the fraction of cached feat acts that it is larger than
            act_quantile, act_precision = feature_data.feature_act_quantiles.get_quantile(_feat_acts, feat_acts_nonzero_locations)
            act_quantile_topk = TopK(act_quantile.topk(k=k, largest=True, dim=-1))
            act_formatting_topk = [f".{act_precision[i]-2}%" for i in act_quantile_topk.indices]
            # Replace the indices with feature indices (these are different when feature_idx argument is not [0, 1, 2, ...])
            act_quantile_topk.indices[:] = _feature_idx[act_quantile_topk.indices]
            scores_dict[("act_quantile", seq_pos)] = (act_quantile_topk, act_formatting_topk)

        # We don't measure loss effect on the first token
        if seq_pos == 0:
            continue
        
        # Filter the loss effects, since we only need the ones which have non-zero feature acts on the tokens before them
        prev_feat_acts_nonzero_filter = (feat_acts[seq_pos - 1] > 0).detach().cpu().numpy()
        _contribution_to_loss = feats_contribution_to_loss[prev_feat_acts_nonzero_filter, seq_pos-1] # [feats_filtered,]
        _feature_idx_prev = np.array(feature_idx)[prev_feat_acts_nonzero_filter]

        if prev_feat_acts_nonzero_filter.sum() > 0:
            k = min(num_top_features, _contribution_to_loss.numel())

            # Get the "loss_effect" scores, which are just the min of features' contributions to loss (min because we're
            # looking for helpful features, not harmful ones)
            contribution_to_loss_topk = TopK(_contribution_to_loss.topk(k=k, largest=False))
            # Replace the indices with feature indices (these are different when feature_idx argument is not [0, 1, 2, ...])
            contribution_to_loss_topk.indices[:] = _feature_idx_prev[contribution_to_loss_topk.indices]
            scores_dict[("loss_effect", seq_pos)] = (contribution_to_loss_topk, ".3f")

    # Get all the features which are required (i.e. all the sequence position indices)
    feature_idx_required = set()
    for (score_topk, formatting_topk) in scores_dict.values():
        feature_idx_required.update(set(score_topk.indices.tolist()))

    prompt_data_dict = {
        feat: PromptData(
            prompt_data = sequence_data_dict[feat],
            sequence_data = feature_data[feat].sequence_data[0],
            middle_plots_data = feature_data[feat].middle_plots_data,
        )
        for feat in feature_idx_required
    }

    return MultiPromptData(
        prompt_str_toks = str_toks,
        prompt_data_dict = prompt_data_dict,
        scores_dict = scores_dict,
    )

