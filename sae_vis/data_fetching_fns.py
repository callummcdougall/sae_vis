import time
import numpy as np
from typing import List
import math
import torch
from torch import nn, Tensor
from eindex import eindex
from typing import Optional, List, Dict, Tuple, Union
import torch.nn.functional as F
import einops
from jaxtyping import Float, Int
from rich import print as rprint
from rich.table import Table
from collections import defaultdict
from transformer_lens import utils, HookedTransformer
from tqdm.auto import tqdm

Arr = np.ndarray

from sae_vis.model_fns import (
    AutoEncoder,
    TransformerLensWrapper,
    to_resid_dir,
)
from sae_vis.utils_fns import (
    k_largest_indices,
    random_range_indices,
    create_vocab_dict,
    QuantileCalculator,
    TopK,
    device,
)
from sae_vis.data_storing_fns import (
    FeatureVisParams,
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




def compute_feat_acts(
    model_acts: Float[Tensor, "batch seq d_in"],
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
        model_acts: Float[Tensor, "batch seq d_in"]
            The activations of the model, which the SAE was trained on.
        feature_idx: Int[Tensor, "feats"]
            The features we're computing the activations for. This will be used to index the encoder's weights.
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
    feature_act_dir = encoder.W_enc[:, feature_idx] # (d_in, feats)
    feature_bias = encoder.b_enc[feature_idx] # (feats,)

    # Calculate & store the feature activations (we need to store them so we can get the right-hand visualisations later)
    x_cent = model_acts - encoder.b_dec
    feat_acts_pre = einops.einsum(x_cent, feature_act_dir, "batch seq d_in, d_in feats -> batch seq feats")
    feat_acts = F.relu(feat_acts_pre + feature_bias)

    # Update the CorrCoef object between feature activation & neurons
    if corrcoef_neurons is not None:
        corrcoef_neurons.update(
            einops.rearrange(feat_acts, "batch seq feats -> feats (batch seq)"),
            einops.rearrange(model_acts, "batch seq d_in -> d_in (batch seq)"),
        )
        
    # Calculate encoder-B feature activations (we don't need to store them, cause it's just for the left-hand visualisations)
    if corrcoef_encoder_B is not None:
        x_cent_B = model_acts - encoder_B.b_dec
        feat_acts_pre_B = einops.einsum(x_cent_B, encoder_B.W_enc, "batch seq d_in, d_in d_hidden -> batch seq d_hidden")
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
    encoder_B: Optional[AutoEncoder],
    model: TransformerLensWrapper,
    tokens: Int[Tensor, "batch seq"],
    feature_indices: Union[int, List[int]],
    fvp: FeatureVisParams,
    progress_bars: Dict[str, tqdm],
) -> Tuple[MultiFeatureData, Dict[str, int], Dict[str, float]]:
    '''
    Gets data that will be used to create the sequences in the feature-centric HTML visualisation.
    
    Note - this function isn't called directly by the user, it actually gets called by the `get_feature_data` function
    which does exactly the same thing except it also batches this computation by features (in accordance with the
    arguments `features` and `minibatch_size_features` from the FeatureVisParams object).

    Args:
        encoder: AutoEncoder
            The encoder whose features we'll be analyzing.

        encoder_B: AutoEncoder
            The encoder we'll be using as a reference (i.e. finding the B-features with the highest correlation). This
            is only necessary if we're generating the left-hand tables (i.e. fvp.include_left_tables=True).

        model: TransformerLensWrapper
            The model we'll be using to get the feature activations. It's a wrapping of the base TransformerLens model.

        tokens: Int[Tensor, "batch seq"]
            The tokens we'll be using to get the feature activations.

        feature_indices: Union[int, List[int]]
            The features we're actually computing. These might just be a subset of the model's full features.

        fvp: FeatureVisParams
            Feature visualization parameters, containing a bunch of other stuff. See the FeatureVisParams docstring for
            more information.

        progress_bars: Dict[str, tqdm]
            A dictionary containing the progress bars for the forward passes and the sequence data. This is used to
            update the progress bars as the computation progresses.

    Returns:
        MultiFeatureData
            Containing data for creating each feature visualization, as well as data for rank-ordering the feature
            visualizations when it comes time to make the prompt-centric view (the `feature_act_quantiles` attribute).

        vocab_dict: Dict[str, int]
            A dictionary containing the vocabulary of the model, which is used to convert token IDs to strings.

        time_log: Dict[str, float]
            A dictionary containing the time taken for each step of the computation. This is optionally printed at the
            end of the `get_feature_data` function, if `fvp.verbose` is set to True.
    '''
    # ! Boring setup code

    t0 = time.time()

    # Make feature_indices a list, for convenience
    if isinstance(feature_indices, int): feature_indices = [feature_indices]
    n_feats = len(feature_indices)

    # Get the vocab dict, which we'll use at the end
    vocab_dict = create_vocab_dict(model.tokenizer)

    # Get tokens into minibatches, for the fwd pass
    token_minibatches = (tokens,) if fvp.minibatch_size_tokens is None else tokens.split(fvp.minibatch_size_tokens)
    token_minibatches = [tok.to(device) for tok in token_minibatches]



    # ! Data setup code (defining the main objects we'll eventually return)

    sequence_data_dict: Dict[int, SequenceMultiGroupData] = {}
    left_tables_data_dict: Dict[int, LeftTablesData] = {}
    middle_plots_data_dict: Dict[int, MiddlePlotsData] = {}

    # Create lists to store the feature activations & final values of the residual stream
    all_resid_post = []
    all_feat_acts = []

    # Create objects to store the rolling correlation coefficients (for left tables)
    corrcoef_neurons = BatchedCorrCoef() if fvp.include_left_tables else None
    corrcoef_encoder_B = BatchedCorrCoef() if (fvp.include_left_tables and (encoder_B is not None)) else None

    # Get encoder & decoder directions
    feature_out_dir = encoder.W_dec[feature_indices] # [feats d_autoencoder]
    feature_resid_dir = to_resid_dir(feature_out_dir, model) # [feats d_model]

    t1 = time.time()



    # ! Compute & concatenate together all feature activations & post-activation function values

    for minibatch in token_minibatches:
        residual, model_acts = model.forward(minibatch, return_logits=False)
        feat_acts = compute_feat_acts(model_acts, feature_indices, encoder, encoder_B, corrcoef_neurons, corrcoef_encoder_B)
        all_feat_acts.append(feat_acts)
        all_resid_post.append(residual)
        # prog.tasks[0].advance(1)
        progress_bars["tokens"].update(1)

    all_feat_acts = torch.cat(all_feat_acts, dim=0)
    all_resid_post = torch.cat(all_resid_post, dim=0)

    t2 = time.time()



    # ! Calculate all data for the left-hand column visualisations, i.e. the 3 tables

    if fvp.include_left_tables:

        # Table 1: neuron alignment, based on decoder weights
        top3_neurons_aligned = TopK(feature_out_dir, k=fvp.rows_in_left_tables, largest=True)
        pct_of_l1: Arr = np.absolute(top3_neurons_aligned.values) / utils.to_numpy(feature_out_dir.abs().sum(dim=-1, keepdim=True))
        
        # Table 2: neurons correlated with this feature, based on their activations
        pearson_topk_neurons, cossim_values_neurons = corrcoef_neurons.topk_pearson(k=fvp.rows_in_left_tables)
        
        # Table 3: encoder-B features correlated with this feature, based on their activations
        # Note, we deal with this differently, because supplying this argument is optional
        using_B = encoder_B is not None
        if using_B:
            pearson_topk_encoderB, cossim_values_encoderB = corrcoef_encoder_B.topk_pearson(k=fvp.rows_in_left_tables)
        
        # Add all this data to the list of LeftTablesData objects
        for i, feat in enumerate(feature_indices):
            left_tables_data_dict[feat] = LeftTablesData(
                neuron_alignment_indices = top3_neurons_aligned[i].indices.tolist(),
                neuron_alignment_values = top3_neurons_aligned[i].values.tolist(),
                neuron_alignment_l1 = pct_of_l1[i].tolist(),
                correlated_neurons_indices = pearson_topk_neurons[i].indices.tolist(),
                correlated_neurons_pearson = pearson_topk_neurons[i].values.tolist(),
                correlated_neurons_l1 = cossim_values_neurons[i].tolist(),
                correlated_features_indices = pearson_topk_encoderB[i].indices.tolist() if using_B else None,
                correlated_features_pearson = pearson_topk_encoderB[i].values.tolist() if using_B else None,
                correlated_features_l1 = cossim_values_encoderB[i].tolist() if using_B else None,
            )
    else:
        left_tables_data_dict = {feat: None for feat in feature_indices}

    t3 = time.time()



    # ! Calculate all data for the right-hand visualisations, i.e. the sequences

    for i, feat in enumerate(feature_indices):

        # Add this feature's sequence data to the list
        sequence_data_dict[feat] = get_sequences_data(
            tokens = tokens,
            feat_acts = all_feat_acts[..., i],
            resid_post = all_resid_post,
            feature_resid_dir = feature_resid_dir[i],
            W_U = model.W_U,
            fvp = fvp,
        )
        progress_bars["feats"].update(1)

    t4 = time.time()
    

    # ! Get all data for the middle column visualisations, i.e. the two histograms & the logit table

    # Get the logits of all features (i.e. the directions this feature writes to the logit output)
    logits = einops.einsum(feature_resid_dir, model.W_U, "feats d_model, d_model d_vocab -> feats d_vocab")

    for i, (feat, logit_vector) in enumerate(zip(feature_indices, logits)):

        # Get data for logits (the histogram, and the table)
        logits_histogram_data = HistogramData.from_data(logit_vector, n_bins=40, tickmode="5 ticks")
        top10_logits = TopK(logit_vector, k=10, largest=True)
        bottom10_logits = TopK(logit_vector, k=10, largest=False)

        # Get data for feature activations histogram (the title, and the histogram)
        feat_acts = all_feat_acts[..., i]
        nonzero_feat_acts = feat_acts[feat_acts > 0]
        frac_nonzero = nonzero_feat_acts.numel() / feat_acts.numel()
        freq_histogram_data = HistogramData.from_data(nonzero_feat_acts, n_bins=40, tickmode="ints")

        # Create a MiddlePlotsData object from this, and add it to the dict
        middle_plots_data_dict[feat] = MiddlePlotsData(
            bottom10_logits = bottom10_logits.values.tolist(),
            bottom10_token_ids = bottom10_logits.indices.tolist(),
            top10_logits = top10_logits.values.tolist(),
            top10_token_ids = top10_logits.indices.tolist(),
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
            fvp = fvp,
        )
        for feat in feature_indices
    }

    # Also get the quantiles, which will be useful for the prompt-centric visualisation
    feature_act_quantiles = QuantileCalculator.from_data(data=einops.rearrange(all_feat_acts, "b s feats -> feats (b s)"))

    t6 = time.time()

    time_logs = {
        "Forward passes to gather data": t2 - t1,
        "Getting data for left tables": t3 - t2,
        "Getting data for middle histograms": t5 - t4,
        "Getting data for right sequences": t4 - t3,
        "Other": (t1 - t0) + (t6 - t5),
    }

    return MultiFeatureData(feature_data, feature_act_quantiles), vocab_dict, time_logs




@torch.inference_mode()
def get_feature_data(
    encoder: AutoEncoder,
    model: HookedTransformer,
    tokens: Int[Tensor, "batch seq"],
    fvp: FeatureVisParams,
    encoder_B: Optional[AutoEncoder] = None,
) -> Tuple[MultiFeatureData, Dict[str, int]]:
    '''
    This is the main function which users will run to generate the feature visualization data. It batches this
    computation over features, in accordance with the arguments in the FeatureVisParams object (we don't want to
    compute all the features at once, since might be too memory-intensive).

    See the `_get_feature_data` function for an explanation of the arguments, as well as a more detailed explanation
    of what this function is doing.

    The return objects are the MultiFeatureData and vocab_dict objects returned by the `_get_feature_data` function.
    '''
    # Apply random seed
    if fvp.seed is not None:
        torch.manual_seed(fvp.seed)
        np.random.seed(fvp.seed)

    # Create objects to store all the data we'll get from `_get_feature_data`
    feature_data = MultiFeatureData()
    time_logs = defaultdict(float)

    # Get a feature list (need to deal with the case where `fvp.features` is an int, or None)
    if fvp.features is None:
        features_list = list(range(encoder.cfg.d_hidden))
    elif isinstance(fvp.features, int):
        features_list = [fvp.features]
    else:
        features_list = list(fvp.features)

    # Break up the features into batches
    feature_batches = [x.tolist() for x in torch.tensor(features_list).split(fvp.minibatch_size_features)]
    # Calculate how many minibatches of tokens there will be (for the progress bar)
    n_token_batches = 1 if (fvp.minibatch_size_tokens is None) else math.ceil(len(tokens) / fvp.minibatch_size_tokens)

    # Add two progress bars (one for the forward passes, one for getting the sequence data)
    progress_bars = {
        "tokens": tqdm(total=n_token_batches*len(feature_batches), desc="Forward passes to gather data"),
        "feats": tqdm(total=len(features_list), desc="Getting sequence data"),
    }

    # If the model is from TransformerLens, we need to apply a wrapper to it for standardization
    assert isinstance(model, HookedTransformer), "Error: non-HookedTransformer models are not yet supported."
    model = TransformerLensWrapper(model, fvp.hook_point)

    # For each feat: get new data and update global data storage objects
    for features in feature_batches:
        new_feature_data, vocab_dict, new_time_logs = _get_feature_data(
            encoder, encoder_B, model, tokens, features, fvp, progress_bars
        )
        feature_data.update(new_feature_data)
        for key, value in new_time_logs.items():
            time_logs[key] += value

    # If verbose, then print the output
    if fvp.verbose:
        total_time = sum(time_logs.values())
        table = Table("Task", "Time", "Pct %")
        for task, duration in time_logs.items():
            table.add_row(task, f"{duration:.2f}s", f"{duration/total_time:.1%}")
        rprint(table)

    return feature_data, vocab_dict




@torch.inference_mode()
def get_sequences_data(
    tokens: Int[Tensor, "batch seq"],
    feat_acts: Float[Tensor, "batch seq"],
    resid_post: Float[Tensor, "batch seq d_model"],
    feature_resid_dir: Float[Tensor, "d_model"],
    W_U: Float[Tensor, "d_model d_vocab"],
    fvp: FeatureVisParams,
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
        feature_resid_dir:
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

    # For each token index [group, seq], we want the seqpos slice [[group, seq-buffer[0]], ..., [group, seq+buffer[1]]]
    # We get one extra dimension at the start, because we need to see the effect on loss of the first token
    buffer_tensor = torch.arange(-fvp.buffer[0] - 1, fvp.buffer[1] + 1, device=indices_full.device)
    indices_full = einops.repeat(indices_full, "batch two -> batch seq two", seq=fvp.buffer[0] + fvp.buffer[1] + 2)
    indices_full = torch.stack([indices_full[..., 0], indices_full[..., 1] + buffer_tensor], dim=-1)
    indices_batch, indices_seq = indices_full.unbind(dim=-1)

    # Use indices_full to get the feature activations & resid post values for the sequences in question
    feat_acts_group = feat_acts[indices_batch, indices_seq]
    resid_post_group = resid_post[indices_batch, indices_seq]

    # Get this feature's output vector, using an outer product over the feature activations for all tokens
    resid_post_feature_effect = einops.einsum(feat_acts_group, feature_resid_dir, "group buf, d_model -> group buf d_model")

    # Get new ablated logits, and old ones
    new_resid_post = resid_post_group - resid_post_feature_effect
    new_logits = (new_resid_post / new_resid_post.std(dim=-1, keepdim=True)) @ W_U
    orig_logits = (resid_post_group / resid_post_group.std(dim=-1, keepdim=True)) @ W_U

    # Get the top5 & bottom5 changes in logits
    # Note, we use TopK's efficient function which takes in a mask, and ignores tokens where all feature acts are zero
    contribution_to_logprobs = orig_logits.log_softmax(dim=-1) - new_logits.log_softmax(dim=-1)
    acts_nonzero = feat_acts_group[:, :-1].abs() > 1e-5 # [group buf]
    top5_contribution_to_logits = TopK(contribution_to_logprobs[:, :-1], k=5, largest=True, tensor_mask=acts_nonzero)
    bottom5_contribution_to_logits = TopK(contribution_to_logprobs[:, :-1], k=5, largest=False, tensor_mask=acts_nonzero)

    # Get the change in loss (which is negative of change of logprobs for correct tokens)
    token_ids = tokens[indices_batch[:, 1:], indices_seq[:, 1:]] # [group seq-1]
    contribution_to_loss = eindex(-contribution_to_logprobs[:, :-1], token_ids, "group buf [group buf]")


    # ! (4) Store the results in a SequenceMultiGroupData object

    # Now that we've indexed everything, construct the batch of SequenceData objects
    sequence_groups_data = []
    group_sizes_cumsum = np.cumsum([0] + [len(indices) for indices in indices_dict.values()]).tolist()
    for group_idx, group_name in enumerate(indices_dict.keys()):
        seq_data = [
            SequenceData(
                token_ids = token_ids[i].tolist(),
                feat_acts = feat_acts_group[i, 1:].tolist(),
                contribution_to_loss = contribution_to_loss[i].tolist(),
                top5_token_ids = top5_contribution_to_logits.indices[i].tolist(),
                top5_logits = top5_contribution_to_logits.values[i].tolist(),
                bottom5_token_ids = bottom5_contribution_to_logits.indices[i].tolist(),
                bottom5_logits = bottom5_contribution_to_logits.values[i].tolist(),
                filter = True,
            )
            for i in range(group_sizes_cumsum[group_idx], group_sizes_cumsum[group_idx+1])
        ]
        sequence_groups_data.append(SequenceGroupData(group_name, seq_data))

    return SequenceMultiGroupData(sequence_groups_data)





@torch.inference_mode()
def get_prompt_data(
    encoder: AutoEncoder,
    model: HookedTransformer,
    prompt: str,
    feature_data: MultiFeatureData,
    fvp: FeatureVisParams,
    num_top_features: int = 10,
) -> MultiPromptData:
    '''
    Gets data that will be used to create the sequences in the prompt-centric HTML visualisation.
    
    Args:
        encoder: AutoEncoder
            The encoder whose features we'll be analyzing.
        model: HookedTransformer
            The model we'll be using to get the feature activations. It'll be wrapped in a TransformerLensWrapper.
        tokens: Int[Tensor, "batch seq"]
            The tokens we'll be using to get the feature activations. Note that we might not be using all of them; the
            number used is determined by `fvp.total_batch_size`.
        feature_indices: Union[int, List[int]]
            The features we're actually computing. These might just be a subset of the model's full features.
        fvp: FeatureVisParams
            Feature visualization parameters, containing a bunch of other stuff. See the FeatureVisParams docstring for
            more information.
        num_top_features: int
            The number of top features to display in this view, for any given metric.

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

    sequence_data_dict: Dict[int, SequenceData] = {}

    # If the model is from TransformerLens, we need to apply a wrapper to it for standardization
    assert isinstance(model, HookedTransformer), "Error: non-HookedTransformer models are not yet supported."
    model = TransformerLensWrapper(model, fvp.hook_point)

    feature_act_dir = encoder.W_enc[:, feature_idx] # [d_in feats]
    feature_out_dir = encoder.W_dec[feature_idx] # [feats d_in]
    feature_resid_dir = to_resid_dir(feature_out_dir, model) # [feats d_model]
    assert feature_act_dir.T.shape == feature_out_dir.shape == (len(feature_idx), encoder.cfg.d_in)

    # ! Define hook functions to cache all the info required for feature ablation, then run those hook fns

    resid_post, act_post = model(tokens, return_logits=False)
    resid_post: Tensor = resid_post.squeeze(0)
    feat_acts = compute_feat_acts(act_post, feature_idx, encoder).squeeze(0)

    feats_contribution_to_loss = torch.empty(size=(n_feats, seq_len-1), device=device)

    # Some logit computations which we only need to do once
    correct_token_unembeddings = model.W_U[:, tokens[0, 1:]] # [d_model seq]
    orig_logits = (resid_post / resid_post.std(dim=-1, keepdim=True)) @ model.W_U # [seq d_vocab]

    print("Put rich progress bar here!")

    for i, feat in enumerate(feature_idx):

        # ! Calculate all data for the sequences (this is the only truly 'new' bit of calculation we need to do)

        # Get this feature's output vector, using an outer product over the feature activations for all tokens
        resid_post_feature_effect = einops.einsum(feat_acts[:, i], feature_resid_dir[i], "seq, d_model -> seq d_model")
        
        # Ablate the output vector from the residual stream, and get logits post-ablation
        new_resid_post = resid_post - resid_post_feature_effect
        new_logits = (new_resid_post / new_resid_post.std(dim=-1, keepdim=True)) @ model.W_U

        # Get the top5 & bottom5 changes in logits (don't bother with `efficient_topk` cause it's small)
        contribution_to_logprobs = orig_logits.log_softmax(dim=-1) - new_logits.log_softmax(dim=-1)
        top5_contribution_to_logits = TopK(contribution_to_logprobs[:-1], k=5)
        bottom5_contribution_to_logits = TopK(contribution_to_logprobs[:-1], k=5, largest=False)

        # Get the change in loss (which is negative of change of logprobs for correct token)
        contribution_to_loss = eindex(-contribution_to_logprobs[:-1], tokens[0, 1:], "seq [seq]")
        feats_contribution_to_loss[i, :] = contribution_to_loss

        # Store the sequence data
        sequence_data_dict[feat] = SequenceData(
            token_ids = tokens.squeeze(0).tolist(),
            feat_acts = feat_acts[:, i].tolist(),
            contribution_to_loss = [0.0] + contribution_to_loss.tolist(),
            top5_token_ids = top5_contribution_to_logits.indices.tolist(),
            top5_logits = top5_contribution_to_logits.values.tolist(),
            bottom5_token_ids = bottom5_contribution_to_logits.indices.tolist(),
            bottom5_logits = bottom5_contribution_to_logits.values.tolist(),
            filter = True,
        )

        # Get the logits for the correct tokens
        logits_for_correct_tokens = einops.einsum(
            feature_resid_dir[i], correct_token_unembeddings,
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

    # Construct a scores dict, which maps from things like f"act_quantile-{seq_pos}" to a list of the top-scoring
    # features and their scores
    scores_dict: Dict[str, Tuple[List[int], List[str]]] = {}

    for seq_pos, str_tok in enumerate(str_toks):

        # Filter the feature activations, since we only need the ones that are non-zero
        feat_acts_nonzero_filter = utils.to_numpy(feat_acts[seq_pos] > 0)
        feat_acts_nonzero_locations = np.nonzero(feat_acts_nonzero_filter)[0].tolist()
        _feat_acts = feat_acts[seq_pos, feat_acts_nonzero_filter] # [feats_filtered,]
        _feature_idx = np.array(feature_idx)[feat_acts_nonzero_filter]

        if feat_acts_nonzero_filter.sum() > 0:
            k = min(num_top_features, _feat_acts.numel())
            
            # Get the "act_size" scores (we return it as a TopK object)
            act_size_topk = TopK(_feat_acts, k=k, largest=True)
            # Store the actual feature indices, and the string representations of the scores
            scores_dict[f"act_size-{seq_pos}"] = (
                _feature_idx[act_size_topk.indices].tolist(), # these are the feature indices
                [f"{v:.3f}" for v in act_size_topk.values], # these are the formatted scores
            )

            # Get the "act_quantile" scores, which is just the fraction of cached feat acts that it is larger than
            act_quantile, act_precision = feature_data.feature_act_quantiles.get_quantile(_feat_acts, feat_acts_nonzero_locations)
            act_quantile_topk = TopK(act_quantile, k=k, largest=True)
            act_formatting = [f".{act_precision[i]-2}%" for i in act_quantile_topk.indices]
            # Store the actual feature indices, and the string representations of the scores
            scores_dict[f"act_quantile-{seq_pos}"] = (
                _feature_idx[act_quantile_topk.indices].tolist(), # these are the feature indices
                [f"{v:{f}}" for v, f in zip(act_quantile_topk.values, act_formatting)], # these are the formatted scores
            )

        # We don't measure loss effect on the first token
        if seq_pos == 0:
            continue
        
        # Filter the loss effects, since we only need the ones which have non-zero feature acts on the tokens before them
        prev_feat_acts_nonzero_filter = utils.to_numpy((feat_acts[seq_pos - 1] > 0))
        _contribution_to_loss = feats_contribution_to_loss[prev_feat_acts_nonzero_filter, seq_pos-1] # [feats_filtered,]
        _feature_idx_prev = np.array(feature_idx)[prev_feat_acts_nonzero_filter]

        if prev_feat_acts_nonzero_filter.sum() > 0:
            k = min(num_top_features, _contribution_to_loss.numel())

            # Get the "loss_effect" scores, which are just the min of features' contributions to loss (min because we're
            # looking for helpful features, not harmful ones)
            contribution_to_loss_topk = TopK(_contribution_to_loss, k=k, largest=False)
            # Store the actual feature indices, and the string representations of the scores
            scores_dict[f"loss_effect-{seq_pos}"] = (
                _feature_idx_prev[contribution_to_loss_topk.indices].tolist(), # these are the feature indices
                [f"{v:.3f}" for v in contribution_to_loss_topk.values], # these are the formatted scores
            )

    # Get all the features which are required (i.e. all the sequence position indices)
    feature_idx_required = set()
    for (feature_indices, score_strings) in scores_dict.values():
        feature_idx_required.update(set(feature_indices))

    prompt_data_dict = {
        feat: PromptData(
            prompt_data = sequence_data_dict[feat],
            middle_plots_data = feature_data[feat].middle_plots_data,
            sequence_data = feature_data[feat].sequence_data[0],
        )
        for feat in feature_idx_required
    }

    return MultiPromptData(
        prompt_str_toks = str_toks,
        prompt_data_dict = prompt_data_dict,
        scores_dict = scores_dict,
    )

