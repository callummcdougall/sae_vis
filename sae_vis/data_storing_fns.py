import json
import random
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from jaxtyping import Float, Int
from sae_lens import SAE, HookedSAETransformer
from torch import Tensor
from tqdm.auto import tqdm

from sae_vis.data_config_classes import (
    PromptConfig,
    SaeVisConfig,
    SaeVisLayoutConfig,
    SeqMultiGroupConfig,
)
from sae_vis.utils_fns import (
    FeatureStatistics,
    HistogramData,
    TopK,
    VocabType,
    compute_othello_board_state_and_valid_moves,
    get_decode_html_safe_fn,
    max_or_1,
    to_str_tokens,
    unprocess_str_tok,
)

PRECISION = 4


def round_1d_list(lst: list[float], precision: int = PRECISION) -> list[float]:
    return [round(f, precision) for f in lst]


def round_2d_list(
    lst: list[list[float]], precision: int = PRECISION
) -> list[list[float]]:
    return [[round(f, precision) for f in floats] for floats in lst]


@dataclass
class FeatureTablesData:
    """
    This contains all the data necessary to make the left-hand tables in prompt-centric visualization. See diagram
    in readme:

        https://github.com/callummcdougall/sae_vis#data_storing_fnspy

    Inputs:
        neuron_alignment...
            The data for the neuron alignment table (each of its 3 cols). In other words, the data
            containing which neurons in the transformer the sae feature is most aligned with.

        correlated_neurons...
            The data for the correlated neurons table (each of its 3 cols). In other words, the data
            containing which neurons in the transformer are most correlated with the sae feature.

        correlated_features...
            The data for the correlated features table (each of its 3 cols). In other words, the data
            containing which features in this sae are most correlated with each other.

        correlated_b_features...
            The data for the correlated features table (each of its 3 cols). In other words, the data
            containing which features in sae-B are most correlated with those in the original sae. Note,
            this one might be absent if we're not using a B-sae.
    """

    neuron_alignment_indices: list[int] = field(default_factory=list)
    neuron_alignment_values: list[float] = field(default_factory=list)
    neuron_alignment_l1: list[float] = field(default_factory=list)
    correlated_neurons_indices: list[int] = field(default_factory=list)
    correlated_neurons_pearson: list[float] = field(default_factory=list)
    correlated_neurons_cossim: list[float] = field(default_factory=list)
    correlated_features_indices: list[int] = field(default_factory=list)
    correlated_features_pearson: list[float] = field(default_factory=list)
    correlated_features_cossim: list[float] = field(default_factory=list)
    correlated_b_features_indices: list[int] = field(default_factory=list)
    correlated_b_features_pearson: list[float] = field(default_factory=list)
    correlated_b_features_cossim: list[float] = field(default_factory=list)

    def data(
        self,
        layout: SaeVisLayoutConfig,
        decode_fn: Callable[[int | list[int], VocabType], str | list[str]],
        component_specific_kwargs: dict[str, Any] = {},
    ) -> dict[str, Any]:
        """
        Returns the HTML for the left-hand tables, wrapped in a 'grid-column' div.

        Note, we only ever use this obj in the context of the left-hand column of the feature-centric vis, and it's
        always the same width & height, which is why there's no customization available for this function.
        """
        cfg = layout.feature_tables_cfg
        assert (
            cfg is not None
        ), "Calling `FeatureTablesData.data`, but with no vis config for that component."
        data = {}

        # Store the neuron alignment data, if it exists
        if len(self.neuron_alignment_indices) > 0:
            assert len(self.neuron_alignment_indices) >= cfg.n_rows, "Not enough rows!"
            data["neuronAlignment"] = [
                {"index": I, "value": f"{V:+.3f}", "percentageL1": f"{L:.1%}"}
                for I, V, L in zip(
                    self.neuron_alignment_indices,
                    self.neuron_alignment_values,
                    self.neuron_alignment_l1,
                )
            ]

        # Store the other 3, if they exist (they're all in the same format, so we can do it in a for loop)
        for name, js_name in zip(
            ["correlated_neurons", "correlated_features", "correlated_b_features"],
            ["correlatedNeurons", "correlatedFeatures", "correlatedFeaturesB"],
        ):
            if len(getattr(self, f"{name}_indices")) > 0:
                # assert len(getattr(self, f"{name}_indices")) >= cfg.n_rows, "Not enough rows!"
                data[js_name] = [
                    {"index": I, "value": f"{P:+.3f}", "percentageL1": f"{C:+.3f}"}
                    for I, P, C in zip(
                        getattr(self, f"{name}_indices")[: cfg.n_rows],
                        getattr(self, f"{name}_pearson")[: cfg.n_rows],
                        getattr(self, f"{name}_cossim")[: cfg.n_rows],
                    )
                ]

        return data


@dataclass
class ActsHistogramData(HistogramData):
    def data(
        self,
        layout: SaeVisLayoutConfig,
        decode_fn: Callable[[int | list[int], VocabType], str | list[str]],
        component_specific_kwargs: dict[str, Any] = {},
    ) -> dict[str, Any]:
        """
        Converts data -> HTML object, for the feature activations histogram (i.e. the histogram over all sampled tokens,
        showing the distribution of activations for this feature).
        """
        cfg = layout.act_hist_cfg
        assert (
            cfg is not None
        ), "Calling `ActsHistogramData.data`, but with no vis config for that component."

        # We can't post-hoc change the number of bins, so check this wasn't changed in the config
        # assert cfg.n_bins == len(self.bar_heights),\
        #     "Can't post-hoc change `n_bins` in histogram config - you need to regenerate data."

        return {
            "y": self.bar_heights,
            "x": self.bar_values,
            "ticks": self.tick_vals,
            "title": self.title if self.title is not None else False,
        }


@dataclass
class LogitsHistogramData(HistogramData):
    def data(
        self,
        layout: SaeVisLayoutConfig,
        decode_fn: Callable[[int | list[int], VocabType], str | list[str]],
        component_specific_kwargs: dict[str, Any] = {},
    ) -> dict[str, Any]:
        """
        Converts data -> HTML object, for the logits histogram (i.e. the histogram over all tokens in the vocab, showing
        the distribution of direct logit effect on that token).
        """
        cfg = layout.logits_hist_cfg
        assert (
            cfg is not None
        ), "Calling `LogitsHistogramData.data`, but with no vis config for that component."

        # We can't post-hoc change the number of bins, so check this wasn't changed in the config
        # assert cfg.n_bins == len(self.bar_heights),\
        #     "Can't post-hoc change `n_bins` in histogram config - you need to regenerate data."

        return {
            "y": self.bar_heights,
            "x": self.bar_values,
            "ticks": self.tick_vals,
            "title": self.title if self.title is not None else False,
        }


@dataclass
class LogitsTableData:
    bottom_token_ids: list[int] = field(default_factory=list)
    bottom_logits: list[float] = field(default_factory=list)
    top_token_ids: list[int] = field(default_factory=list)
    top_logits: list[float] = field(default_factory=list)
    vocab_type: VocabType = "unembed"
    max_logits: float | None = None

    @classmethod
    def from_data(
        cls, logits: Float[Tensor, "d_vocab"], k: int, max_logits: float | None = None
    ) -> "LogitsTableData":
        # Get logits table data
        top_logits = TopK(logits, k)
        bottom_logits = TopK(logits, k, largest=False)
        return LogitsTableData(
            bottom_logits=bottom_logits.values.tolist(),
            bottom_token_ids=bottom_logits.indices.tolist(),
            top_logits=top_logits.values.tolist(),
            top_token_ids=top_logits.indices.tolist(),
            vocab_type="unembed",
            max_logits=max_logits,
        )

    def data(
        self,
        layout: SaeVisLayoutConfig,
        decode_fn: Callable[[int | list[int], VocabType], str | list[str]],
        component_specific_kwargs: dict[str, Any] = {},
    ) -> dict[str, Any]:
        """
        Converts data -> HTML object, for the logits table (i.e. the top and bottom affected tokens by this feature).
        """
        cfg = layout.logits_table_cfg
        assert (
            cfg is not None
        ), "Calling `LogitsTableData.data`, but with no vis config for that component."

        # Crop the lists to `cfg.n_rows` (first checking the config doesn't ask for more rows than we have)
        assert cfg.n_rows <= len(self.bottom_logits)

        # Get the string tokens, using the decode function (unembed mode or probes mode!)
        neg_str = to_str_tokens(
            self.bottom_token_ids[: cfg.n_rows], decode_fn, self.vocab_type
        )
        pos_str = to_str_tokens(
            self.top_token_ids[: cfg.n_rows], decode_fn, self.vocab_type
        )

        # Get max loss (might be part of table data)
        max_logits = self.max_logits or max(
            0.0, self.top_logits[0], -self.bottom_logits[0]
        )

        # Get data for the tables of pos/neg logits
        return {
            "negLogits": [
                {
                    "symbol": unprocess_str_tok(neg_str[i]),
                    "value": round(self.bottom_logits[i], 2),
                }
                for i in range(len(neg_str))
            ],
            "posLogits": [
                {
                    "symbol": unprocess_str_tok(pos_str[i]),
                    "value": round(self.top_logits[i], 2),
                }
                for i in range(len(pos_str))
            ],
            "maxLogits": round(max_logits, PRECISION),
        }


@dataclass
class ProbeLogitsTableData:
    """
    Basically a wrapper for LogitsTableData, used when we have multiple probes (each with different
    names), but we treat those probes basically like we treat the unembedding when we're getting top
    logits.
    """

    probe_logits_data: dict[str, LogitsTableData] = field(default_factory=dict)

    @classmethod
    def from_data(
        cls,
        probe_names_and_values: list[tuple[str, Float[Tensor, "d_vocab"]]],
        k: int,
        max_logits: float | None = None,
    ) -> "ProbeLogitsTableData":
        """
        Each value in the logits dict is a single logit vector for the corresponding probe.

        # TODO - I like this method, I should make more methods work like this! Nice to have the complexity in classes, not functions. Don't really care about saving these classes, cause I save the JSON data now instead. Also, even if I wanted to save classes, it's still fine to have this logic in a classmethod!
        """
        probe_logits_data = {}

        for name, logits in probe_names_and_values:
            top_logits = TopK(logits, k)
            bottom_logits = TopK(logits, k, largest=False)
            probe_logits_data[name] = LogitsTableData(
                bottom_logits=bottom_logits.values.tolist(),
                bottom_token_ids=bottom_logits.indices.tolist(),
                top_logits=top_logits.values.tolist(),
                top_token_ids=top_logits.indices.tolist(),
                vocab_type="probes",
                max_logits=max_logits or logits.abs().max().item(),
                # max_logits=max_logits or max_or_1([L.abs().max().item() for L in probes.values()]),
            )
        return cls(probe_logits_data=probe_logits_data)

    def data(
        self,
        layout: SaeVisLayoutConfig,
        decode_fn: Callable[[int | list[int], VocabType], str | list[str]],
        component_specific_kwargs: dict[str, Any] = {},
    ) -> dict[str, Any]:
        return {
            name: probe_data.data(layout, decode_fn, component_specific_kwargs)
            for name, probe_data in self.probe_logits_data.items()
        }


@dataclass
class SequenceData:
    """
    This contains all the data necessary to make a sequence of tokens in the vis. See diagram in readme:

        https://github.com/callummcdougall/sae_vis#data_storing_fnspy

    Always-visible data:
        token_ids:          List of token IDs in the sequence
        token_posns:        Strings of "(batch_idx, seq_idx)" for our tokens (only for vis)
        feat_acts:          Sizes of activations on this sequence
        feat_acts_idx:      When feat_acts is a length-1 list, this is that index (we need it!)
        loss_contribution:  Effect on loss of this feature, for this particular token (neg = helpful)
        logit_contribution: Effect on correct logits (pos = helpful)
        orig_prob/new_prob: Original/new prediction for this feature, to help understand better

    Data which is visible on hover:
        token_logits:       The logits of the particular token in that sequence (used for line on logits histogram)
        top_token_ids:     List of the top 5 logit-boosted tokens by this feature
        top_logits:        List of the corresponding 5 changes in logits for those tokens
        bottom_token_ids:  List of the bottom 5 logit-boosted tokens by this feature
        bottom_logits:     List of the corresponding 5 changes in logits for those tokens

    Data which is specific to certain model or SAE classes:
        dfa_token_ids:     List of the buffer of top DFA source tokens for dest this token (centering on the top one)
        dfa_values:        List of the corresponding DFA values for those tokens
        dfa_token_posns:   Strings of "(batch_idx, seq_idx)" for our dfa tokens (only for vis)
    """

    token_ids: list[int] = field(default_factory=list)
    token_posns: list[str] = field(default_factory=list)
    feat_acts: list[float] = field(default_factory=list)
    feat_acts_idx: int | None = None
    loss_contribution: list[float] = field(default_factory=list)
    logit_contribution: list[float] | None = None
    orig_prob: list[float] | None = None
    new_prob: list[float] | None = None

    token_logits: list[float] = field(default_factory=list)
    top_token_ids: list[list[int]] = field(default_factory=list)
    top_logits: list[list[float]] = field(default_factory=list)
    bottom_token_ids: list[list[int]] = field(default_factory=list)
    bottom_logits: list[list[float]] = field(default_factory=list)

    dfa_token_ids: list[int] = field(default_factory=list)
    dfa_values: list[float] = field(default_factory=list)
    dfa_token_posns: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """
        Filters the logits & token IDs by removing any elements which are zero (this saves space in the eventual
        JavaScript).
        """
        self.seq_len = len(self.token_ids)
        self.top_logits, self.top_token_ids = self._filter(
            self.top_logits, self.top_token_ids
        )
        self.bottom_logits, self.bottom_token_ids = self._filter(
            self.bottom_logits, self.bottom_token_ids
        )

    def _filter(
        self, float_list: list[list[float]], int_list: list[list[int]]
    ) -> tuple[list[list[float]], list[list[int]]]:
        """
        Filters the list of floats and ints, by removing any elements which are zero. Note - the absolute values of the
        floats are monotonic non-increasing, so we can assume that all the elements we keep will be the first elements
        of their respective lists. Also reduces precisions of feature activations & logits.
        """
        # Next, filter out zero-elements and reduce precision
        float_list = [
            [round(f, PRECISION) for f in floats if abs(f) > 1e-6]
            for floats in float_list
        ]
        int_list = [ints[: len(floats)] for ints, floats in zip(int_list, float_list)]
        return float_list, int_list

    def _get_seq_data(
        self,
        cfg: PromptConfig | SeqMultiGroupConfig,
        decode_fn: Callable[[int | list[int], VocabType], str | list[str]],
        component_specific_kwargs: dict[str, Any] = {},
    ) -> dict[Literal["seqData", "dfaSeqData", "seqMetadata"], Any]:
        """
        Args:

        Returns:
            js_data: list[dict[str, Any]]
                The data for this sequence, in the form of a list of dicts for each token (where the dict stores things
                like token, feature activations, etc).
        """
        assert isinstance(
            cfg, (PromptConfig, SeqMultiGroupConfig)
        ), f"Invalid config type: {type(cfg)}"

        seq_group_id = component_specific_kwargs.get("seq_group_id", None)
        bold_idx = component_specific_kwargs.get("bold_idx", None)
        permanent_line = component_specific_kwargs.get("permanent_line", False)
        hover_above = component_specific_kwargs.get("hover_above", False)

        # If we didn't supply a sequence group ID, then we assume this sequence is on its own, and give it a unique ID
        if seq_group_id is None:
            seq_group_id = f"prompt-{random.randint(0, 999999):06d}"

        if isinstance(cfg, SeqMultiGroupConfig) and cfg.othello:
            # In this case, return a dict containing {board, valid, lastMove, etc}
            assert (
                bold_idx == "max"
                and len(self.feat_acts) == len(self.loss_contribution) == 1
            ), "Othello expects bold_idx='max' and only 1 feature act, because we see all 59 game states and only highlight one."
            assert self.feat_acts_idx is not None
            results = compute_othello_board_state_and_valid_moves(
                moves := self.token_ids[: self.feat_acts_idx + 1]
            )
            assert not isinstance(results, str), f"Error: {results!r} in {moves}"
            act = self.feat_acts[0]
            loss = self.loss_contribution[0]

            seq_data = {
                "board": results["board"],
                "valid": results["valid"],
                "move": results["move"],
                "captured": results["captured"],
                "act": act,
                "loss": loss,
            }
            dfa_seq_data = []
        else:
            # If we didn't specify bold_idx, then set it to be the midpoint
            if bold_idx is None:
                bold_idx = self.seq_len // 2

            def process_data_fn(x: Any):
                """
                Handles cases where fields have some entries missing (because we consciously didn't get data for them).
                """
                if (not isinstance(x, list)) or (len(x) == 0):
                    return x
                # If we only had data for the bold token, then pad the list out.
                x = deepcopy(x)
                default = [] if isinstance(x[0], list) else 0.0
                if isinstance(cfg, SeqMultiGroupConfig) and (not cfg.compute_buffer):
                    assert (
                        bold_idx != "max"
                    ), "Don't know how to deal with this case yet."
                    x = [
                        x[0] if (i == bold_idx) + 1 else default
                        for i in range(self.seq_len)
                    ]
                # If we're computing the whole sequence not a buffer, then data for the first token might need padding
                if (
                    isinstance(cfg, SeqMultiGroupConfig) and (cfg.buffer is None)
                ) or isinstance(cfg, PromptConfig):
                    if len(x) < self.seq_len:  # TODO - when is this true?
                        x = [default] + x
                # At this point, we expect the length of everything to match the length of `self.token_ids`
                assert len(x) == len(
                    self.token_ids
                ), f"Error: {len(x)=}, {len(self.token_ids)=}"
                return x

            feat_acts = process_data_fn(self.feat_acts)
            loss_contribution = process_data_fn(self.loss_contribution)
            logit_contribution = process_data_fn(self.logit_contribution)
            orig_prob = process_data_fn(self.orig_prob)
            new_prob = process_data_fn(self.new_prob)
            pos_ids = process_data_fn(self.top_token_ids)
            neg_ids = process_data_fn(self.bottom_token_ids)
            pos_val = process_data_fn(self.top_logits)
            neg_val = process_data_fn(self.bottom_logits)
            dfa_values = process_data_fn(self.dfa_values)

            # Process the tokens to get str toks
            toks = to_str_tokens(self.token_ids, decode_fn, "embed")
            pos_toks = [to_str_tokens(pos, decode_fn, "unembed") for pos in pos_ids]
            neg_toks = [to_str_tokens(neg, decode_fn, "unembed") for neg in neg_ids]

            # Get list of data dicts for each token
            seq_data = []
            dfa_seq_data = []

            # Get DFA data (there's way less of this than for the main sequences)
            if len(self.dfa_token_ids) > 0:
                dfa_toks = to_str_tokens(self.dfa_token_ids, decode_fn, "embed")
                for i in range(len(self.dfa_token_ids)):
                    kwargs_dfa = dict(
                        tok=unprocess_str_tok(dfa_toks[i]),
                        tokID=self.dfa_token_ids[i],
                        tokPosn=self.dfa_token_posns[i],
                        dfaValue=dfa_values[i],
                        # isBold=i == np.argmax(dfa_values).item(),
                    )
                    dfa_seq_data.append(kwargs_dfa)

            # Get data for primary sequences
            for i in range(len(self.token_ids)):
                # We might store a bunch of different case-specific data in the JavaScript object for each token. This is
                # done in the form of a disjoint union over different dictionaries (which can each be empty or not), this
                # minimizes the size of the overall JavaScript object. See function in `tokens_script.js` for more.
                kwargs_bold: dict[str, bool] = {}
                kwargs_hide: dict[str, bool] = {}
                kwargs_this_token_active: dict[str, Any] = {}
                kwargs_prev_token_active: dict[str, Any] = {}
                kwargs_hover_above: dict[str, bool] = {}

                # Get args if this is the bolded token (we make it bold, and maybe add permanent line to histograms)
                if bold_idx is not None:
                    kwargs_bold["isBold"] = (bold_idx == i) or (
                        bold_idx == "max" and i == np.argmax(feat_acts).item()
                    )
                    if kwargs_bold["isBold"] and permanent_line:
                        kwargs_bold["permanentLine"] = True

                # If we only have data for the bold token, we hide all other tokens' hoverdata (and skip other kwargs)
                if (
                    (isinstance(cfg, SeqMultiGroupConfig) and (not cfg.compute_buffer))
                    and isinstance(bold_idx, int)
                    and (i not in {bold_idx, bold_idx + 1})
                ):
                    kwargs_hide["hide"] = True

                else:
                    # Get args if we're making the tooltip hover above token (default is below)
                    if hover_above:
                        kwargs_hover_above["hoverAbove"] = True

                    # If feature active on this token, get background color and feature act (for hist line)
                    if abs(feat_acts[i]) > 1e-8:
                        kwargs_this_token_active = dict(
                            featAct=round(feat_acts[i], PRECISION)
                        )

                    # If prev token active, get the top/bottom logits table, underline color, and loss effect (for hist line)
                    if len(pos_toks[i]) + len(neg_toks[i]) > 0:
                        kwargs_prev_token_active = dict(
                            posToks=pos_toks[i],
                            negToks=neg_toks[i],
                            posVal=pos_val[i],
                            negVal=neg_val[i],
                            lossEffect=round(loss_contribution[i], PRECISION),
                            logitEffect=round(logit_contribution[i], PRECISION),
                        )
                        if (orig_prob is not None) and (new_prob is not None):
                            kwargs_prev_token_active |= dict(
                                origProb=round(orig_prob[i], PRECISION + 2),
                                newProb=round(new_prob[i], PRECISION + 2),
                            )

                # pyright 1.1.373 freaks out if rounding is done in-line below for some reason
                # this is likely a bug in 1.1.373, but rounding here is equivalent anyway
                token_logit = round(self.token_logits[i], PRECISION)
                seq_data.append(
                    dict(
                        tok=unprocess_str_tok(toks[i]),
                        tokID=self.token_ids[i],
                        tokPosn=self.token_posns[i] if self.token_posns else "",
                        tokenLogit=token_logit,
                        **kwargs_bold,
                        **kwargs_this_token_active,
                        **kwargs_prev_token_active,
                        **kwargs_hover_above,
                    )
                )

        return {
            "seqData": seq_data,
            "dfaSeqData": dfa_seq_data,
            "seqMetadata": {},
        }


@dataclass
class SeqGroupData:
    """
    This contains all the data necessary to make a single group of sequences (e.g. a quantile in prompt-centric
    visualization). See diagram in readme:

        https://github.com/callummcdougall/sae_vis#data_storing_fnspy

    Inputs:
        title:      The title that this sequence group will have, if any.
        seq_data:   The data for the sequences in this group.
    """

    seq_data: list[SequenceData] = field(default_factory=list)
    title: str = ""
    # dfa_title: str = ""

    def __len__(self) -> int:
        return len(self.seq_data)

    @property
    def max_feat_act(self) -> float:
        """Returns maximum value of feature activation over all sequences in this group."""
        return max_or_1([act for seq in self.seq_data for act in seq.feat_acts])

    @property
    def max_loss_contribution(self) -> float:
        """Returns maximum value of loss contribution over all sequences in this group."""
        return max_or_1(
            [loss for seq in self.seq_data for loss in seq.loss_contribution], abs=True
        )

    @property
    def max_dfa_value(self) -> float:
        """Returns maximum value of DFA values over all sequences in this group."""
        return max_or_1(
            [value for seq in self.seq_data for value in seq.dfa_values], abs=True
        )

    def _get_seq_group_data(
        self,
        cfg: SeqMultiGroupConfig | PromptConfig,
        decode_fn: Callable[[int | list[int], VocabType], str | list[str]],
        component_specific_kwargs: dict[str, Any] = {},
        # These default values should be correct when we only have one sequence group, because when we call this from
        # a SeqMultiGroupData we'll override them
    ) -> dict[Literal["seqGroupData", "seqGroupMetadata"], Any]:
        """
        This creates a single group of sequences, i.e. title plus some number of vertically stacked sequences.

        Args (from component-specific kwargs):
            seq_group_id:   The id of the sequence group div. This will usually be passed as e.g. "seq-group-001".
            group_size:     Max size of sequences in the group (i.e. we truncate after this many, if argument supplied).
            max_feat_act:   If supplied, then we use this as the most extreme value (for coloring by feature act).

        Returns:
            html_obj:       Object containing the HTML and JavaScript data for this seq group.
        """

        # Get metadata for this sequence group
        seq_group_id = (
            component_specific_kwargs.get("seq_group_id", None)
            or f"seq-group-{random.randint(0, 999999):06d}"
        )
        group_size = component_specific_kwargs.get("group_size", None)
        max_feat_act = component_specific_kwargs.get("max_feat_act", None) or round(
            max_or_1([act for seq in self.seq_data for act in seq.feat_acts]), PRECISION
        )
        max_loss_contribution = component_specific_kwargs.get(
            "max_loss_contribution", None
        ) or round(
            max_or_1([loss for seq in self.seq_data for loss in seq.loss_contribution]),
            PRECISION,  # abs=True?
        )
        max_dfa_value = component_specific_kwargs.get("max_dfa_value", None) or round(
            max_or_1([value for seq in self.seq_data for value in seq.dfa_values]),
            PRECISION,
        )

        seqGroupMetadata = {
            "title": self.title,
            "seqGroupId": seq_group_id,
            "maxAct": max_feat_act,
            "maxLoss": max_loss_contribution,
            "maxDFA": max_dfa_value,
        }
        # Deal with prompt cfg vs sequence cfg mode (we don't need to supply all arguments!)
        if isinstance(cfg, SeqMultiGroupConfig):
            seqGroupMetadata["nBoardsPerRow"] = cfg.n_boards_per_row

        # Get data for this sequence group
        seqGroupData = [
            seq._get_seq_data(
                cfg,
                decode_fn,
                component_specific_kwargs=dict(
                    bold_idx="max"
                    if isinstance(cfg, PromptConfig) or cfg.buffer is None
                    else cfg.buffer[0],
                    permanent_line=False,  # in a group, we're never showing a permanent line (only for single seqs)
                    seq_group_id=seq_group_id,
                ),
            )
            for seq in self.seq_data[:group_size]
        ]
        return {"seqGroupData": seqGroupData, "seqGroupMetadata": seqGroupMetadata}


@dataclass
class SeqMultiGroupData:
    """
    This contains all the data necessary to make multiple groups of sequences (e.g. the different quantiles in the
    prompt-centric visualization). See diagram in readme:

        https://github.com/callummcdougall/sae_vis#data_storing_fnspy
    """

    seq_group_data: list[SeqGroupData] = field(default_factory=list)
    is_prompt: bool = False

    def __getitem__(self, idx: int) -> SeqGroupData:
        return self.seq_group_data[idx]

    def data(
        self,
        layout: SaeVisLayoutConfig,
        decode_fn: Callable[[int | list[int], VocabType], str | list[str]],
        component_specific_kwargs: dict[str, Any] = {},
    ) -> list[dict[Literal["seqGroupData", "seqGroupMetadata"], Any]]:
        """
        Args:
            decode_fn:                  Mapping from token IDs to string tokens.
            component_specific_kwargs:  Contains any specific kwargs that could be used to customize this component.

        Returns:
            html_obj:  Object containing the HTML and JavaScript data for these multiple seq groups.
        """
        cfg = layout.prompt_cfg if self.is_prompt else layout.seq_cfg
        assert (
            cfg is not None
        ), "Calling `SeqMultiGroupData.data`, but with no vis config for that component."

        # Get max activation value & max loss contributions, over all sequences in all groups
        max_feat_act = component_specific_kwargs.get(
            "max_feat_act",
            max_or_1([seq_group.max_feat_act for seq_group in self.seq_group_data]),
        )
        max_loss_contribution = component_specific_kwargs.get(
            "max_loss_contribution", self.seq_group_data[0].max_loss_contribution
        )
        max_dfa_value = component_specific_kwargs.get(
            "max_dfa_value", self.seq_group_data[0].max_dfa_value
        )

        group_sizes = cfg.group_sizes if isinstance(cfg, SeqMultiGroupConfig) else [1]

        return [
            sequences_group._get_seq_group_data(
                cfg=cfg,
                decode_fn=decode_fn,
                component_specific_kwargs=dict(
                    group_size=group_size,
                    max_feat_act=max_feat_act,
                    max_loss_contribution=max_loss_contribution,
                    max_dfa_value=max_dfa_value,
                    seq_group_id=f"seq-group-{i}",
                ),
            )
            for i, (group_size, sequences_group) in enumerate(
                zip(group_sizes, self.seq_group_data)
            )
        ]


GenericData = (
    FeatureTablesData
    | ActsHistogramData
    | LogitsTableData
    | LogitsHistogramData
    | SeqMultiGroupData
)


@dataclass
class SaeVisData:
    """
    # TODO - check if this docstring is still correct

    This contains all the data necessary for constructing the feature-centric visualization, over multiple
    features (i.e. being able to navigate through them).

    Args:
        feature_data_dict:  Contains data for each individual feature-centric vis. For each feature, this looks like a
                            dict like {"seqMultiGroup": seqMultiGroupData, ...}.
        prompt_data_dict:   Contains data for each prompt-centric vis. For each feature, rather than keys being
                            component names, they're tuple-ified prompts, and the values are the corresponding
                            SeqMultiGroupData objects (only containing 1 prompt).

        feature_stats:      Contains stats over all features (e.g. activation quantiles for each feature, used for
                            rank-ordering features in the prompt-centric vis)
        cfg:                The vis config, used for the both the data gathering and the vis layout

        model:              Model that our sae was trained on
        sae:                Used to get the feature activations
        sae_B:              Used to get the feature activations for the second model (if applicable)
        linear_probes:      Used to get logits for probe directions in e.g. Othello models
    """

    feature_data_dict: dict[int, dict[str, GenericData]] = field(default_factory=dict)
    prompt_data_dict: dict[int, dict[tuple[str | int, ...], SeqMultiGroupData]] = field(
        default_factory=dict
    )

    feature_stats: FeatureStatistics = field(default_factory=FeatureStatistics)
    cfg: SaeVisConfig = field(default_factory=SaeVisConfig)

    model: HookedSAETransformer | None = None
    sae: SAE | None = None
    sae_B: SAE | None = None
    linear_probes: list[
        tuple[Literal["input", "output"], str, Float[Tensor, "d_model d_vocab_out"]]
    ] = field(default_factory=list)

    vocab_dict: dict[VocabType, dict[int, str]] | None = None

    def __post_init__(self):
        # TODO - actually use the unembedding mode
        if self.vocab_dict is None:
            assert self.model is not None
            self.vocab_dict = {v: k for k, v in self.model.tokenizer.vocab.items()}  # type: ignore
            self.vocab_dict = {
                k: self.vocab_dict for k in ["embed", "unembed", "probes"]
            }  # type: ignore
        self.decode_fn = get_decode_html_safe_fn(self.vocab_dict)  # type: ignore

    def update(self, other: "SaeVisData") -> None:
        """
        Updates SaeVisData with the data from another SaeVisData object. This is useful during the `get_feature_data`
        function, since this function is broken up into different groups of features then merged together.
        """
        if other is None:
            return
        self.feature_data_dict.update(other.feature_data_dict)
        self.feature_stats.update(other.feature_stats)

    @classmethod
    def create(
        cls,
        sae: SAE,
        model: HookedSAETransformer,
        tokens: Int[Tensor, "batch seq"],
        cfg: SaeVisConfig,
        # optional
        sae_B: SAE | None = None,
        linear_probes: list[
            tuple[Literal["input", "output"], str, Float[Tensor, "d_model d_vocab_out"]]
        ] = [],
        target_logits: Float[Tensor, "batch seq d_vocab_out"] | None = None,
        vocab_dict: dict[VocabType, dict[int, str]] | None = None,
        verbose: bool = False,
        clear_memory_between_batches: bool = False,
    ) -> "SaeVisData":
        """
        Optional args:

            sae_B: SAE
                Extra SAE for computing stuff like top feature correlations between the 2 SAEs.

            linear_probes: list[tuple[str, str, Tensor]]
                A list of linear probes, which will be used to create tables that go below the logit tables. Each linear
                probe consists of:
                    label:   either "input" or "output". If "input" then the table will tell us which probe dirs the sae
                             fires strongly on; if "output" then it tells us which dirs the sae writes to strongly.
                    name:    the name of the probe, which will be displayed in the table
                    weights: the weights of the linear probe, which will be used to compute the table values
                For example, an OthelloGPT input probe for the "theirs vs mine" direction would tell us if the latent
                is firing strongly when the model thinks the square is "theirs", and an output probe for that direction
                would tell us if the latent is writing strongly to this direction when it fires.

            target_logits: Tensor
                If supplied, then rather than comparing logits[:, 1:] to tokens[:, :-1], we'll compare logits[:, 1:] to
                target_logits, where target_logits is assumed to be a tensor of logprobs. This is useful when e.g. we
                have models like OthelloGPT whose target distribution isn't deterministic (they match a uniform distn
                over legal moves). Note, this isn't practical to use for language models because the vocab is massive;
                it's more of a special thing for a certain kind of toy models.
        """
        from sae_vis.data_fetching_fns import get_feature_data

        return get_feature_data(
            sae=sae,
            model=model,
            tokens=tokens,
            cfg=cfg,
            sae_B=sae_B,
            linear_probes=linear_probes,
            target_logits=target_logits,
            vocab_dict=vocab_dict,
            verbose=verbose,
            clear_memory_between_batches=clear_memory_between_batches,
        )

    def save_feature_centric_vis(
        self,
        filename: str,
        feature: int | None = None,
        verbose: bool = False,
    ) -> None:
        """
        Saves the HTML page for feature-centric vis. By default it will open on `feature`, or on the
        smallest feature if none is specified.
        """
        layout = self.cfg.feature_centric_layout

        # Set the default argument for the dropdown (i.e. when the page first loads)
        if feature not in self.feature_data_dict:
            first_feature = min(self.feature_data_dict)
            if feature is not None:
                print(f"Feat {feature} not found, defaulting to {first_feature}")
            feature = first_feature
        assert isinstance(feature, int)

        all_feature_data = list(self.feature_data_dict.items())
        if verbose:
            all_feature_data = tqdm(all_feature_data, desc="Saving feature-centric vis")

        DATA = {
            str(feat): {
                comp_name: components_dict[comp_name].data(
                    layout=layout, decode_fn=self.decode_fn
                )
                for comp_name in layout.components
            }
            for feat, components_dict in all_feature_data
        }

        html = self._create_html_file(layout, str(feature), DATA)
        with open(filename, "w") as f:
            f.write(html)

    def save_prompt_centric_vis(
        self,
        filename: str | Path,
        prompt: str,
        metric: str | None = None,
        seq_pos: int | None = None,
        num_top_features: int = 10,
        verbose: bool = False,
    ):
        """
        Saves the HTML page for prompt-centric vis. If supplied then it will open on the `seq_pos`-
        indexed sequence position, and the metric `metric`, unless these aren't supplied in which
        case it'll open on the first listed (seq_pos, metric) in the dictionary returned from
        `get_prompt_data`.
        """
        layout = self.cfg.prompt_centric_layout

        # Run forward passes on our prompt, and store the data within each FeatureData object as `self.prompt_data` as
        # well as returning the scores_dict (which maps from score hash to a list of feature indices & formatted scores)
        from sae_vis.data_fetching_fns import get_prompt_data

        # This function populates "self.feature_data" by adding the prompt data component, and also
        # returns dict mapping stringified metric keys to the top features for that metric
        PROMPT_DATA = get_prompt_data(
            sae_vis_data=self, prompt=prompt, num_top_features=num_top_features
        )
        assert len(PROMPT_DATA) > 0, "No active feats found for any prompt tokens"

        # Get all possible values for dropdowns
        str_toks = self.model.tokenizer.tokenize(prompt)  # type: ignore
        str_toks = [
            t.replace("|", "│") for t in str_toks
        ]  # vertical line -> pipe (hacky, so key splitting on | works)
        seq_keys = [f"{t!r} ({i})" for i, t in enumerate(str_toks)]

        # Get default values for dropdowns
        if PROMPT_DATA.get(first_key := f"{metric}|{seq_keys[seq_pos or 0]}", []) == []:
            valid_keys = [k for k, v in PROMPT_DATA.items() if len(v) > 0]
            assert len(valid_keys) > 0, "No active feats found for any prompt tokens"
            first_key = valid_keys[0]
            first_metric = first_key.split("|")[0]
            first_seq_pos_match = re.search(r"\((\d+)\)", first_key.split("|")[1])
            assert first_seq_pos_match is not None
            first_seq_pos = int(first_seq_pos_match.group(1))
            if metric is not None and seq_pos is not None:
                print(
                    f"Invalid choice of {metric=} and {seq_pos=}, defaulting to metric={first_metric!r} and seq_pos={first_seq_pos}"
                )

        all_feature_data = list(self.feature_data_dict.items())
        if verbose:
            all_feature_data = tqdm(all_feature_data, desc="Saving feature-centric vis")

        DATA = {
            str(feat): {
                comp_name: components_dict[comp_name].data(
                    layout=layout, decode_fn=self.decode_fn
                )
                for comp_name in layout.components
            }
            for feat, components_dict in all_feature_data
        }

        # TODO - do I need `self.feature_stats.aggdata`?

        html = self._create_html_file(layout, first_key, DATA, PROMPT_DATA)
        with open(filename, "w") as f:
            f.write(html)

    def _create_html_file(
        self,
        layout: SaeVisLayoutConfig,
        start_key: str,
        data: dict[str, Any],
        prompt_data: dict[str, Any] = {},
    ):
        init_js_str = (Path(__file__).parent / "init.js").read_text()
        style_css_str = (Path(__file__).parent / "style.css").read_text()

        return f"""
<div id='dropdown-container'></div>
<div class='grid-container'></div>

<script src="https://d3js.org/d3.v6.min.js"></script>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

<style>
{style_css_str}
</style>

<script>
const START_KEY = {json.dumps(start_key)};
const METADATA = {json.dumps(layout.metadata)};
const DATA = defineData();
const PROMPT_DATA = definePromptData();

{init_js_str}

function defineData() {{
    return {json.dumps(data)};
}}

function definePromptData() {{
    return {json.dumps(prompt_data)};
}}
</script>
"""
