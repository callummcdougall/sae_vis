import numpy as np
import itertools
from pathlib import Path
from torch import Tensor
from typing import Optional, Union, Any, Callable
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from jaxtyping import Int
import re
from copy import deepcopy

from transformer_lens import HookedTransformer

from sae_vis.model_fns import AutoEncoder

from sae_vis.utils_fns import (
    to_str_tokens,
    QuantileCalculator,
    HistogramData,
    SaveableDataclass,
    get_decode_html_safe_fn,
    unprocess_str_tok,
    max_or_1,
)
from sae_vis.html_fns import (
    bgColorMap,
    uColorMap,
    HTML,
)
from sae_vis.data_config_classes import (
    SaeVisConfig,
    FeatureTablesConfig,
    ActsHistogramConfig,
    LogitsTableConfig,
    LogitsHistogramConfig,
    SequencesConfig,
    PromptConfig,
    GenericConfig,
)

METRIC_TITLES = {
    "act_size": "Activation Size",
    "act_quantile": "Activation Quantile",
    "loss_effect": "Loss Effect",
}
PRECISION = 4

'''
This file contains all the dataclasses which store data used to create the visualizations. 

There are 2 special classes (SaeVisData, FeatureData), and 8 other component dataclasses.

TLDR:

    - SaeVisData is the class that users interact with
        - It contains a bunch of FeatureData objects (one for each feature) which it uses when creating both the
            feature-centric and prompt-centric visualizations.
    - The other 8 component dataclasses store data for a given component, e.g. logits table / hist or sequence group.
    - FeatureData is the middleman:
        - It stores a bunch of components, enough to make the feature-centric vis for a single feature, or a single
            column of the prompt-centric vis for a particular feature
        - The SaeVisData object has methods to create the feature-centric and prompt-centric vis, and these will call
            methods of all its FeatureData objects & merge the results

WAY MORE DETAIL THAN YOU EVER THOUGHT YOU NEEDED:

SaeVisData

    This is the most important class (it's the one that users interact with). The 3 user-exposed methods are:
    - `create` (classmethod) which is how we gather the data in the first place
    - `save_html_feature_centric` which is how we get the HTML for the feature-centric view
    - `save_html_prompt_centric` which is how we get the HTML for the prompt-centric view

FeatureData

    This is maybe the most complicated class to understand, since it's basically a middle-ground between SaeVisData and
    the 8 component dataclasses. To explain this middle-ground role:

        - It contains a bunch of component dataclasses (e.g. FeatureTablesData, ActsHistogramData, etc). This is all the
          data necessary to make the feature-centric vis, for a single feature (or to make a single column of the
          prompt-centric vis, for a particular feature).
        - SaeVisData contains a bunch of FeatureData objects (one for each feature)

    In feature-centric vis:
        `SaeVisData.save_html_feature_centric` will call `FeatureData._get_html_data_feature_centric` for each of its
        features (i.e. returning a bunch of full-page objects), then merge the results into a single HTML object which
        becomes the feature-centric vis.

    In prompt-centric vis:
        Calling `SaeVisData.save_html_prompt_centric` will call `FeatureData._get_html_data_prompt_centric` for every
        choice of tuple (metric function, token in user's prompt) and every one of the top-scoring features for that
        tuple (i.e. returning a bunch of column objects), then merge across features into a full-page object, then merge
        these full-page objects into a single HTML object which becomes the prompt-centric vis.

8 component dataclasses

The 8 component dataclasses correspond to particular components in the vis (e.g. logits table, feature activations
histogram, or sequence groups), which might appear more than once in any given vis. Each of these classes has a method
called `_get_html_data`, which returns an HTML object (we'll union over all these HTML objects for any given vis). The
5 arguments for each `_get_html_data` method are always the same:
    
    - `cfg`, which is the config object for the visualization. Every different type of component has its own config, see
        `data_config_classes.py` for more. Example: the `FeatureTablesConfig` object determines how many rows to include
        in the tables.
    
    - `decode_fn`, which is a function that takes a token ID and returns a string token. This is created from the model
        vocabulary, but we use this as an argument so we don't have to create it more than once.
    
    - `id_suffix`, which is a string that is appended to the end of the ID of the HTML object. This is used to make sure
        that components have a unique ID, and it also helps us match up token hoverlines with the right histogram.
    
    - `column`, which is the index of the column in the visualization. This is also used to give components a unique ID.

    - `component_specific_kwargs`, which is a dictionary of any other arguments that are specific to the component. This
        is currently only used for the prompt data in the prompt-centric view, because we need to specify which token is
        bolded (and that it has a permanent line). It's good to be able to pass all these arguments into a single dict,
        since this way I can still keep the same type signature (these 5 inputs) for all components. Also, I can do
        things like pass `bold_idx` into this dictionary without checking which components it's for, since it'll just
        be ignored if it's not needed for the component in question.

A full list of these 8 classes (and what data they contain):

    FeatureTablesData:      Basic data about features, e.g. correlated neurons (left-hand tables in default view)
    ActsHistogramData:      Data for the feature activations histogram (middle top histogram in default view)
    LogitsTableData:        Data for the logits table (middle table in default view)
    LogitsHistogramData:    Data for the logits histogram (middle bottom histogram in default view)
    SequenceData:           Data for a single sequence of tokens (right-hand side of default view)
    SequenceGroupData:      Data for a group of sequences (e.g. a quantile in prompt-centric view)
    SequenceMultiGroupData: Data for multiple groups of sequences (e.g. all quantiles in prompt-centric view)
'''




@dataclass_json
@dataclass
class FeatureTablesData:
    '''
    This contains all the data necessary to make the left-hand tables in prompt-centric visualization. See diagram
    in readme: 

        https://github.com/callummcdougall/sae_vis#data_storing_fnspy

    Inputs:
        neuron_alignment...
            The data for the neuron alignment table (each of its 3 columns). In other words, the data containing which
            neurons in the transformer the encoder feature is most aligned with.

        correlated_neurons...
            The data for the correlated neurons table (each of its 3 columns). In other words, the data containing which
            neurons in the transformer are most correlated with the encoder feature.

        correlated_features...
            The data for the correlated features table (each of its 3 columns). In other words, the data containing
            which features in encoder-B are most correlated with those in the original encoder. Note, this one might be
            absent if we're not using a B-encoder.
    '''
    neuron_alignment_indices: list[int] = field(default_factory=list)
    neuron_alignment_values: list[float] = field(default_factory=list)
    neuron_alignment_l1: list[float] = field(default_factory=list)
    correlated_neurons_indices: list[int] = field(default_factory=list)
    correlated_neurons_pearson: list[float] = field(default_factory=list)
    correlated_neurons_cossim: list[float] = field(default_factory=list)
    correlated_features_indices: list[int] = field(default_factory=list)
    correlated_features_pearson: list[float] = field(default_factory=list)
    correlated_features_cossim: list[float] = field(default_factory=list)

    def _get_html_data(
        self,
        cfg: FeatureTablesConfig,
        decode_fn: Callable[[int | list[int]], str | list[str]],
        id_suffix: str,
        column: int | tuple[int, int],
        component_specific_kwargs: dict[str, Any] = {},
    ) -> HTML:
        '''
        Returns the HTML for the left-hand tables, wrapped in a 'grid-column' div.

        Note, we only ever use this obj in the context of the left-hand column of the feature-centric vis, and it's
        always the same width & height, which is why there's no customization available for this function.
        '''
        # Crop the lists to `cfg.n_rows` (first checking the config doesn't ask for more rows than we have)
        assert cfg.n_rows <= len(self.neuron_alignment_indices)
        neuron_alignment_indices = self.neuron_alignment_indices[:cfg.n_rows]
        neuron_alignment_values = self.neuron_alignment_values[:cfg.n_rows]
        neuron_alignment_l1 = self.neuron_alignment_l1[:cfg.n_rows]
        correlated_neurons_indices = self.correlated_neurons_indices[:cfg.n_rows]
        correlated_neurons_pearson = self.correlated_neurons_pearson[:cfg.n_rows]
        correlated_neurons_cossim = self.correlated_neurons_cossim[:cfg.n_rows]

        # Read HTML from file, and replace placeholders with real ID values
        html_str = (Path(__file__).parent / "html" / "feature_tables_template.html").read_text()
        html_str = html_str.replace("FEATURE_TABLES_ID", f"feature-tables-{id_suffix}")

        # Add the neuron alignment and neuron correlation data to the object which will be turned into JavaScript
        data: dict[str, Any] = {
            "neuronAlignment": [
                {"index": I, "value": f"{V:+.3f}", "percentageL1": f"{L:.1%}"}
                for I, V, L in zip(neuron_alignment_indices, neuron_alignment_values, neuron_alignment_l1)
            ],
            "correlatedNeurons": [
                {"index": I, "value": f"{P:+.3f}", "percentageL1": f"{C:+.3f}"}
                for I, P, C in zip(correlated_neurons_indices, correlated_neurons_pearson, correlated_neurons_cossim)
            ],
        }

        # If we have correlated features from encoder_B, add that to the JavaScript data
        if len(self.correlated_features_indices) > 0:
            correlated_features_indices = self.correlated_features_indices[:cfg.n_rows]
            correlated_features_pearson = self.correlated_features_pearson[:cfg.n_rows]
            correlated_features_cossim = self.correlated_features_cossim[:cfg.n_rows]

            assert (correlated_features_pearson is not None) and (correlated_features_cossim is not None), "All or none"
            data["correlatedFeaturesBEncoder"] = [
                {"index": I, "value": f"{P:+.3f}", "percentageL1": f"{C:+.3f}"}
                for I, P, C in zip(correlated_features_indices, correlated_features_pearson, correlated_features_cossim)
            ]
        # If not, remove that table from the HTML string
        else:
            html_str = re.sub(r'<h4>CORRELATED FEATURES \(B-ENCODER\)</h4>.*?</table>', "", html_str, flags=re.DOTALL)

        return HTML(
            html_data = {column: html_str},
            js_data = {"featureTablesData": {id_suffix: data}}
        )


@dataclass_json
@dataclass
class ActsHistogramData(HistogramData):
    def _get_html_data(
        self,
        cfg: ActsHistogramConfig,
        decode_fn: Callable[[int | list[int]], str | list[str]],
        id_suffix: str,
        column: int | tuple[int, int],
        component_specific_kwargs: dict[str, Any] = {},
    ) -> HTML:
        '''
        Converts data -> HTML object, for the feature activations histogram (i.e. the histogram over all sampled tokens,
        showing the distribution of activations for this feature).
        '''
        # We can't post-hoc change the number of bins, so check this wasn't changed in the config
        # assert cfg.n_bins == len(self.bar_heights),\
        #     "Can't post-hoc change `n_bins` in histogram config - you need to regenerate data."
        
        # Read HTML from file, and replace placeholders with real ID values
        html_str = (Path(__file__).parent / "html" / "acts_histogram_template.html").read_text()
        html_str = html_str.replace("HISTOGRAM_ACTS_ID", f"histogram-acts-{id_suffix}")

        # Process colors for frequency histogram; it's darker at higher values
        bar_values_normed = [(0.4 * max(self.bar_values) + 0.6 * v) / max(self.bar_values) for v in self.bar_values]
        bar_colors = [bgColorMap(v) for v in bar_values_normed]
        
        # Next we create the data dict
        data: dict[str, Any] = {
            "y": self.bar_heights,
            "x": self.bar_values,
            "ticks": self.tick_vals,
            "colors": bar_colors,
        }
        if self.title is not None:
            data["title"] = self.title

        return HTML(
            html_data = {column: html_str},
            js_data = {"actsHistogramData": {id_suffix: data}},
        )


@dataclass_json
@dataclass
class LogitsHistogramData(HistogramData):
    def _get_html_data(
        self,
        cfg: LogitsHistogramConfig,
        decode_fn: Callable[[int | list[int]], str | list[str]],
        id_suffix: str,
        column: int | tuple[int, int],
        component_specific_kwargs: dict[str, Any] = {},
    ) -> HTML:
        '''
        Converts data -> HTML object, for the logits histogram (i.e. the histogram over all tokens in the vocab, showing
        the distribution of direct logit effect on that token).
        '''
        # We can't post-hoc change the number of bins, so check this wasn't changed in the config
        # assert cfg.n_bins == len(self.bar_heights),\
        #     "Can't post-hoc change `n_bins` in histogram config - you need to regenerate data."

        # Read HTML from file, and replace placeholders with real ID values
        html_str = (Path(__file__).parent / "html" / "logits_histogram_template.html").read_text()
        html_str = html_str.replace("HISTOGRAM_LOGITS_ID", f"histogram-logits-{id_suffix}")

        data: dict[str, Any] = {
            "y": self.bar_heights,
            "x": self.bar_values,
            "ticks": self.tick_vals,
        }
        if self.title is not None:
            data["title"] = self.title

        return HTML(
            html_data = {column: html_str},
            js_data = {"logitsHistogramData": {id_suffix: data}},
        )


@dataclass_json
@dataclass
class LogitsTableData:

    bottom_token_ids: list[int] = field(default_factory=list)
    bottom_logits: list[float] = field(default_factory=list)
    top_token_ids: list[int] = field(default_factory=list)
    top_logits: list[float] = field(default_factory=list)

    def _get_html_data(
        self,
        cfg: LogitsTableConfig,
        decode_fn: Callable[[int | list[int]], str | list[str]],
        id_suffix: str,
        column: int | tuple[int, int],
        component_specific_kwargs: dict[str, Any] = {},
    ) -> HTML:
        '''
        Converts data -> HTML object, for the logits table (i.e. the top and bottom affected tokens by this feature).
        '''
        # Crop the lists to `cfg.n_rows` (first checking the config doesn't ask for more rows than we have)
        assert cfg.n_rows <= len(self.bottom_logits)
        bottom_token_ids = self.bottom_token_ids[:cfg.n_rows]
        bottom_logits = self.bottom_logits[:cfg.n_rows]
        top_token_ids = self.top_token_ids[:cfg.n_rows]
        top_logits = self.top_logits[:cfg.n_rows]
        
        # Get the negative and positive background values (darkest when equals max abs)
        max_value = max(max(top_logits[:cfg.n_rows]), -min(bottom_logits[:cfg.n_rows]))
        neg_bg_values = np.absolute(bottom_logits[:cfg.n_rows]) / max_value
        pos_bg_values = np.absolute(top_logits[:cfg.n_rows]) / max_value

        # Get the string tokens, using the decode function
        neg_str = to_str_tokens(decode_fn, bottom_token_ids[:cfg.n_rows])
        pos_str = to_str_tokens(decode_fn, top_token_ids[:cfg.n_rows])

        # Read HTML from file, and replace placeholders with real ID values
        html_str = (Path(__file__).parent / "html" / "logits_table_template.html").read_text()
        html_str = html_str.replace("LOGITS_TABLE_ID", f"logits-table-{id_suffix}")

        # Create object for storing JS data
        data: dict[str, list] = {"negLogits": [], "posLogits": []}

        # Get data for the tables of pos/neg logits
        for i in range(len(neg_str)):
            data["negLogits"].append({
                "symbol": unprocess_str_tok(neg_str[i]),
                "value": round(top_logits[i], 2),
                "color": f"rgba(255,{int(255*(1-neg_bg_values[i]))},{int(255*(1-neg_bg_values[i]))},0.5)"
            })
            data["posLogits"].append({
                "symbol": unprocess_str_tok(pos_str[i]),
                "value": round(top_logits[i], 2),
                "color": f"rgba({int(255*(1-pos_bg_values[i]))},{int(255*(1-pos_bg_values[i]))},255,0.5)"})
            
        return HTML(
            html_data = {column: html_str},
            js_data = {"logitsTableData": {id_suffix: data}},
        )





@dataclass_json
@dataclass
class SequenceData:
    '''
    This contains all the data necessary to make a sequence of tokens in the vis. See diagram in readme: 

        https://github.com/callummcdougall/sae_vis#data_storing_fnspy

    Always-visible data:
        token_ids:              List of token IDs in the sequence
        feat_acts:              Sizes of activations on this sequence
        loss_contribution:   Effect on loss of this feature, for this particular token (neg = helpful)

    Data which is visible on hover:
        token_logits:       The logits of the particular token in that sequence (used for line on logits histogram)
        top_token_ids:     List of the top 5 logit-boosted tokens by this feature
        top_logits:        List of the corresponding 5 changes in logits for those tokens
        bottom_token_ids:  List of the bottom 5 logit-boosted tokens by this feature
        bottom_logits:     List of the corresponding 5 changes in logits for those tokens
    '''
    token_ids: list[int] = field(default_factory=list)
    feat_acts: list[float] = field(default_factory=list)
    loss_contribution: list[float] = field(default_factory=list)

    token_logits: list[float] = field(default_factory=list)
    top_token_ids: list[list[int]] = field(default_factory=list)
    top_logits: list[list[float]] = field(default_factory=list)
    bottom_token_ids: list[list[int]] = field(default_factory=list)
    bottom_logits: list[list[float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        '''
        Filters the logits & token IDs by removing any elements which are zero (this saves space in the eventual
        JavaScript).
        '''
        self.seq_len = len(self.token_ids)
        self.top_logits, self.top_token_ids = self._filter(self.top_logits, self.top_token_ids)
        self.bottom_logits, self.bottom_token_ids = self._filter(self.bottom_logits, self.bottom_token_ids)

    def _filter(
        self,
        float_list: list[list[float]],
        int_list: list[list[int]]
    ) -> tuple[list[list[float]], list[list[int]]]:
        '''
        Filters the list of floats and ints, by removing any elements which are zero. Note - the absolute values of the
        floats are monotonic non-increasing, so we can assume that all the elements we keep will be the first elements
        of their respective lists. Also reduces precisions of feature activations & logits.
        '''
        # Next, filter out zero-elements and reduce precision
        float_list = [[round(f, PRECISION) for f in floats if abs(f) > 1e-6] for floats in float_list]
        int_list = [ints[:len(floats)] for ints, floats in zip(int_list, float_list)]
        return float_list, int_list
    

    def _get_html_data(
        self,
        cfg: PromptConfig | SequencesConfig,
        decode_fn: Callable[[int | list[int]], str | list[str]],
        id_suffix: str,
        column: int | tuple[int, int],
        component_specific_kwargs: dict[str, Any] = {},
    ) -> HTML:
        '''
        Args:

        Returns:
            js_data: list[dict[str, Any]]
                The data for this sequence, in the form of a list of dicts for each token (where the dict stores things
                like token, feature activations, etc).
        '''
        assert isinstance(cfg, (PromptConfig, SequencesConfig)), f"Invalid config type: {type(cfg)}"
        seq_group_id = component_specific_kwargs.get("seq_group_id", None)
        max_act_color = component_specific_kwargs.get("max_act_color", None)
        max_loss_color = component_specific_kwargs.get("max_loss_color", None)
        bold_idx = component_specific_kwargs.get("bold_idx", None)
        permanent_line = component_specific_kwargs.get("permanent_line", False)
        first_in_group = component_specific_kwargs.get("first_in_group", True)
        title = component_specific_kwargs.get("title", None)
        hover_above = component_specific_kwargs.get("hover_above", False)

        # If we didn't supply a sequence group ID, then we assume this sequence is on its own, and give it a unique ID
        if seq_group_id is None: seq_group_id = f"prompt-{column:03d}"
        
        # If we didn't specify bold_idx, then set it to be the midpoint
        if bold_idx is None: bold_idx = self.seq_len // 2

        # If we only have data for the bold token, we pad out everything with zeros or empty lists
        only_bold = isinstance(cfg, SequencesConfig) and not(cfg.compute_buffer)
        if only_bold:
            feat_acts = [self.feat_acts[0] if i == bold_idx else 0.0 for i in range(self.seq_len)]
            loss_contribution = [self.loss_contribution[0] if i == bold_idx + 1 else 0.0 for i in range(self.seq_len)]
            pos_ids = [self.top_token_ids[0] if i == bold_idx + 1 else [] for i in range(self.seq_len)]
            neg_ids = [self.bottom_token_ids[0] if i == bold_idx + 1 else [] for i in range(self.seq_len)]
            pos_val = [self.top_logits[0] if i == bold_idx + 1 else [] for i in range(self.seq_len)]
            neg_val = [self.bottom_logits[0] if i == bold_idx + 1 else [] for i in range(self.seq_len)]
        else:
            feat_acts = deepcopy(self.feat_acts)
            loss_contribution = deepcopy(self.loss_contribution)
            pos_ids = deepcopy(self.top_token_ids)
            neg_ids = deepcopy(self.bottom_token_ids)
            pos_val = deepcopy(self.top_logits)
            neg_val = deepcopy(self.bottom_logits)

        # Get values for converting into colors later
        bg_denom = max_or_1(self.feat_acts) if (max_act_color is None) else max_act_color
        u_denom = max_or_1(self.loss_contribution, abs=True) if (max_loss_color is None) else max_loss_color
        bg_values = (np.maximum(feat_acts, 0.0) / max(1e-4, bg_denom)).tolist()
        u_values = (np.array(loss_contribution) / max(1e-4, u_denom)).tolist()

        # If we sent in a prompt rather than this being sliced from a longer sequence, then the pos_ids etc will be shorter
        # than the token list by 1, so we need to pad it at the first token
        if isinstance(cfg, PromptConfig):
            assert len(pos_ids) == len(neg_ids) == len(pos_val) == len(neg_val) == len(self.token_ids) - 1,\
                "If this is a single prompt, these lists must be the same length as token_ids or 1 less"
            pos_ids = [[]] + pos_ids
            neg_ids = [[]] + neg_ids
            pos_val = [[]] + pos_val
            neg_val = [[]] + neg_val
        assert len(pos_ids) == len(neg_ids) == len(pos_val) == len(neg_val) == len(self.token_ids),\
            "If this is part of a sequence group etc are given, they must be the same length as token_ids"

        # Process the tokens to get str toks
        toks = to_str_tokens(decode_fn, self.token_ids)
        pos_toks = [to_str_tokens(decode_fn, pos) for pos in pos_ids]
        neg_toks = [to_str_tokens(decode_fn, neg) for neg in neg_ids]

        # Define the JavaScript object which will be used to populate the HTML string
        js_data_list = []

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
            if bold_idx is not None and bold_idx == i:
                kwargs_bold["isBold"] = True
                if permanent_line: kwargs_bold["permanentLine"] = True

            # If we only have data for the bold token, we hide all other tokens' hoverdata (and skip other kwargs)
            if only_bold and isinstance(bold_idx, int) and (i not in {bold_idx, bold_idx + 1}):
                kwargs_hide["hide"] = True

            else:
                # Get args if we're making the tooltip hover above token (default is below)
                if hover_above:
                    kwargs_hover_above["hoverAbove"] = True

                # If feature active on this token, get background color and feature act (for hist line)
                if abs(feat_acts[i]) > 1e-8:
                    kwargs_this_token_active = dict(
                        featAct = round(feat_acts[i], PRECISION),
                        bgColor = bgColorMap(bg_values[i]),
                    )

                # If prev token active, get the top/bottom logits table, underline color, and loss effect (for hist line)
                pos_toks_i, neg_toks_i, pos_val_i, neg_val_i = pos_toks[i], neg_toks[i], pos_val[i], neg_val[i]
                if len(pos_toks_i) + len(neg_toks_i) > 0:
                    # Create dictionary
                    kwargs_prev_token_active = dict(
                        posToks = pos_toks_i,
                        negToks = neg_toks_i,
                        posVal = pos_val_i,
                        negVal = neg_val_i,
                        lossEffect = round(loss_contribution[i], PRECISION),
                        uColor = uColorMap(u_values[i]),
                    )

            js_data_list.append(dict(
                tok = unprocess_str_tok(toks[i]),
                tokID = self.token_ids[i],
                tokenLogit = round(self.token_logits[i], PRECISION),
                **kwargs_bold,
                **kwargs_this_token_active,
                **kwargs_prev_token_active,
                **kwargs_hover_above,
            ))

        # Create HTML string (empty by default since sequences are added by JavaScript) and JS data    
        html_str = ""
        js_seq_group_data: dict[str, Any] = {"data": [js_data_list]}

        # Add group-specific stuff if this is the first sequence in the group
        if first_in_group:

            # Read HTML from file, replace placeholders with real ID values
            html_str = (Path(__file__).parent / "html" / "sequences_group_template.html").read_text()
            html_str = html_str.replace("SEQUENCE_GROUP_ID", seq_group_id)

            # Get title of sequence group, and the idSuffix to match up with a histogram
            js_seq_group_data["idSuffix"] = id_suffix
            if title is not None: js_seq_group_data["title"] = title

        return HTML(
            html_data = {column: html_str},
            js_data = {"tokenData": {seq_group_id: js_seq_group_data}}
        )


@dataclass_json
@dataclass
class SequenceGroupData:
    '''
    This contains all the data necessary to make a single group of sequences (e.g. a quantile in prompt-centric
    visualization). See diagram in readme: 

        https://github.com/callummcdougall/sae_vis#data_storing_fnspy

    Inputs:
        title:      The title that this sequence group will have, if any. This is used in `_get_html_data`. The titles
                    will actually be in the HTML strings, not in the JavaScript data.
        seq_data:   The data for the sequences in this group.
    '''
    title: str = ""
    seq_data: list[SequenceData] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.seq_data)
    
    def _get_html_data(
        self,
        cfg: SequencesConfig,
        decode_fn: Callable[[int | list[int]], str | list[str]],
        id_suffix: str,
        column: int | tuple[int, int],
        component_specific_kwargs: dict[str, Any] = {}

        # These default values should be correct when we only have one sequence group, because when we call this from
        # a SequenceMultiGroupData we'll override them)
    ) -> HTML:
        '''
        This creates a single group of sequences, i.e. title plus some number of vertically stacked sequences.

        Note, `column` is treated specially here, because it might overflow (could be a tuple).

        Args:
            decode_fn:      Mapping from token IDs to string tokens.
            bold_idx:       If supplied, then we bold the token at this index. Default is the middle token. Note, this
                            argument is only allowed if the sequence group contains just one sequence.
            group_size:     Max size of sequences in the group (i.e. we truncate after this many, if argument supplied).
            max_act_color:  If supplied, then we use this as the most extreme value (for coloring by feature act).
            permanent_line: If True, the bolded token for first seq in group will have a permanent line on histograms.
            seq_group_id:   The id of the sequence group div. This will usually be passed as e.g. "seq-group-001".
            column:         The index of this column. Note that it can be a tuple, if our SequenceMultiGroupData object
                            has overflowed into a new column.
        
        Returns:
            html_obj:       Object containing the HTML and JavaScript data for this seq group.
        '''
        seq_group_id = component_specific_kwargs.get("seq_group_id", None)
        group_size = component_specific_kwargs.get("group_size", None)
        max_act_color = component_specific_kwargs.get("max_act_color", None)
        max_loss_color = component_specific_kwargs.get("max_loss_color", None)

        # Get the data that will go into the div (list of list of dicts, i.e. containing all data for seqs in group). We
        # start with the title.
        html_obj = HTML()

        # If seq_group_id is not supplied, then we assume this is the only sequence in the column, and we name the group
        # after the column
        if seq_group_id is None: seq_group_id = f"seq-group-{column:03d}"

        # If max_act_color not supplied, use the max over all sequences in this group. Same for loss color
        if max_act_color is None: max_act_color = max_or_1([act for seq in self.seq_data for act in seq.feat_acts])
        if max_loss_color is None: max_loss_color = max_or_1([loss for seq in self.seq_data for loss in seq.loss_contribution], abs=True)
        
        # Accumulate the HTML data for each sequence in this group
        for i, seq in enumerate(self.seq_data[:group_size]):
            html_obj += seq._get_html_data(
                cfg = cfg,
                # pass in a PromptConfig object
                decode_fn = decode_fn,
                id_suffix = id_suffix,
                column = column,
                component_specific_kwargs = dict(
                    bold_idx = cfg.buffer[0],
                    permanent_line = False, # in a group, we're never showing a permanent line (only for single seqs)
                    max_act_color = max_act_color,
                    max_loss_color = max_loss_color,
                    seq_group_id = seq_group_id,
                    first_in_group = (i == 0),
                    title = self.title,
                )
            )
        
        return html_obj



@dataclass_json
@dataclass
class SequenceMultiGroupData:
    '''
    This contains all the data necessary to make multiple groups of sequences (e.g. the different quantiles in the
    prompt-centric visualization). See diagram in readme: 

        https://github.com/callummcdougall/sae_vis#data_storing_fnspy
    '''
    seq_group_data: list[SequenceGroupData] = field(default_factory=list)

    def __getitem__(self, idx: int) -> SequenceGroupData:
        return self.seq_group_data[idx]

    def _get_html_data(
        self,
        cfg: SequencesConfig,
        decode_fn: Callable[[int | list[int]], str | list[str]],
        id_suffix: str,
        column: int | tuple[int, int],
        component_specific_kwargs: dict[str, Any] = {},
    ) -> HTML:
        '''
        Args:

        Returns:
            html_obj:  Object containing the HTML and JavaScript data for these multiple seq groups.
        '''
        assert isinstance(column, int)

        # Get max activation value & max loss contributions, over all sequences in all groups
        max_act_color = max_or_1([
            act for group in self.seq_group_data for seq in group.seq_data for act in seq.feat_acts
        ])
        max_loss_color = max_or_1([
            loss for group in self.seq_group_data for seq in group.seq_data for loss in seq.loss_contribution
        ], abs=True)

        # Get the correct column indices for the sequence groups, depending on how group_wrap is configured. Note, we 
        # deal with overflowing columns by extending the dictionary, i.e. our column argument isn't just `column`, but
        # is a tuple of `(column, x)` where `x` is the number of times we've overflowed. For instance, if we have mode
        # 'stack-none' then our columns are `(column, 0), (column, 1), (column, 1), (column, 1), (column, 2), ...`
        n_groups = len(self.seq_group_data)
        n_quantile_groups = n_groups - 1
        match cfg.stack_mode:
            case "stack-all":
                # Here, we stack all groups into 1st column
                cols = [column for i in range(n_groups)]
            case "stack-quantiles":
                # Here, we give 1st group its own column, and stack all groups into second column
                cols = [(column, 0)] + [(column, 1) for _ in range(n_quantile_groups)]
            case "stack-none":
                # Here, we stack groups into columns as [1, 3, 3, ...]
                cols = [(column, 0), *[(column, 1 + int(i/3)) for i in range(n_quantile_groups)]]
            case _:
                raise ValueError(f"Invalid stack_mode: {cfg.stack_mode}. Expected in 'stack-x' for x='all', 'quantiles', 'none'")

        # Create the HTML object, and add all the sequence groups to it, possibly across different columns
        html_obj = HTML()
        for i, (col, group_size, sequences_group) in enumerate(zip(cols, cfg.group_sizes, self.seq_group_data)):
            html_obj += sequences_group._get_html_data(
                cfg = cfg,
                decode_fn = decode_fn,
                id_suffix = id_suffix,
                column = col,
                component_specific_kwargs = dict(
                    group_size = group_size,
                    max_act_color = max_act_color,
                    max_loss_color = max_loss_color,
                    seq_group_id = f"seq-group-{column}-{i}", # we label our sequence groups with (index, column)
                )
            )

        return html_obj



GenericData = Union[
    FeatureTablesData,
    ActsHistogramData,
    LogitsTableData,
    LogitsHistogramData,
    SequenceMultiGroupData,
    SequenceData,
]


@dataclass_json
@dataclass
class FeatureData(SaveableDataclass):
    '''
    This contains all the data necessary to make the feature-centric visualization, for a single feature. See
    diagram in readme: 

        https://github.com/callummcdougall/sae_vis#data_storing_fnspy

    Args:
        feature_idx:    Index of the feature in question (not used within this class's methods, but used elsewhere).
        cfg:            Contains layout parameters which are important in the `get_html` function.
        
        The other args are the 6 possible components we might have in the feature-centric vis, i.e. this is where we
        store the actual data. Note that one of these arguments is `prompt_data` which is only applicable in the prompt-
        centric view.

    This is used in both the feature-centric and prompt-centric views. In the feature-centric view, a single one
    of these objects creates the HTML for a single feature (i.e. a full screen). In the prompt-centric view, a single
    one of these objects will create one column of the full screen vis.
    '''
    feature_idx: int
    cfg: SaeVisConfig

    feature_tables_data: FeatureTablesData = field(default_factory = lambda: FeatureTablesData())
    acts_histogram_data: ActsHistogramData = field(default_factory = lambda: ActsHistogramData())
    logits_table_data: LogitsTableData = field(default_factory = lambda: LogitsTableData())
    logits_histogram_data: LogitsHistogramData = field(default_factory = lambda: LogitsHistogramData())
    sequence_data: SequenceMultiGroupData = field(default_factory = lambda: SequenceMultiGroupData())
    prompt_data: SequenceData = field(default_factory = lambda: SequenceData())

    def get_component_from_config(self, config: GenericConfig) -> GenericData:
        '''
        Given a config object, returns the corresponding data object stored by this instance. For instance, if the input
        is an `FeatureTablesConfig` instance, then this function returns `self.feature_tables_data`.
        '''
        CONFIG_CLASS_MAP = {
            FeatureTablesConfig.__name__: self.feature_tables_data,
            ActsHistogramConfig.__name__: self.acts_histogram_data,
            LogitsTableConfig.__name__: self.logits_table_data,
            LogitsHistogramConfig.__name__: self.logits_histogram_data,
            SequencesConfig.__name__: self.sequence_data,
            PromptConfig.__name__: self.prompt_data,
        }
        config_class_name = config.__class__.__name__
        assert config_class_name in CONFIG_CLASS_MAP, f"Invalid component config: {config_class_name}"
        return CONFIG_CLASS_MAP[config_class_name]
    
    def _get_html_data_feature_centric(
        self,
        decode_fn: Callable,
    ) -> HTML:
        '''
        Returns the HTML object for a single feature-centric view. These are assembled together into the full feature-
        centric view.

        Args:
            decode_fn:  We use this function to decode the token IDs into string tokens. 

        Returns:
            html_obj.html_data:
                Contains a dictionary with keys equal to columns, and values equal to the HTML strings. These will be
                turned into grid-column elements, and concatenated.
            html_obj.js_data:
                Contains a dictionary with keys = component names, and values = JavaScript data that will be used by the
                scripts we'll eventually dump in.
        '''
        # Create object to store all HTML
        html_obj = HTML()

        # For every column in this feature-centric layout, we add all the components in that column
        for column_idx, column_components in self.cfg.feature_centric_layout.columns.items():
            for component_config in column_components:
                component = self.get_component_from_config(component_config)
                html_obj += component._get_html_data(
                    cfg = component_config,
                    decode_fn = decode_fn,
                    column = column_idx,
                    id_suffix = "0", # we only use this if we have >1 set of histograms, i.e. prompt-centric vis
                )

        return html_obj

    def _get_html_data_prompt_centric(
        self,
        decode_fn: Callable,
        column_idx: int,
        bold_idx: int,
        title: str,
    ) -> HTML:
        '''
        Returns the HTML object for a single column of the prompt-centric view. These are assembled together into a full
        screen of a prompt-centric view, and then they're further assembled together into the full prompt-centric view.

        Args:
            decode_fn:  We use this function to decode the token IDs into string tokens. 
            column_idx: This method only gives us a single column (of the prompt-centric vis), so we need to know which
                        column this is (for the JavaScript data).
            title:      The title for this column, which will be used in the JavaScript data.

        Returns:
            html_obj.html_data:
                Contains a dictionary with the single key `str(column_idx)`, representing the single column. This will
                become a single grid-column element, and will get concatenated with others of these.
            html_obj.js_data:
                Contains a dictionary with keys = component names, and values = JavaScript data that will be used by the
                scripts we'll eventually dump in.
        '''
        # Create object to store all HTML
        html_obj = HTML()

        # Verify that we only have a single column
        layout = self.cfg.prompt_centric_layout
        assert layout.columns.keys() == {0},\
            f"cfg.prompt_centric_layout should only have 1 column, instead found cols {layout.columns.keys()}"
        assert layout.prompt_cfg is not None,\
            "cfg.prompt_centric_layout should include a PromptConfig, but found None"
        if layout.seq_cfg is not None:
            assert (layout.seq_cfg.n_quantiles == 0) or (layout.seq_cfg.stack_mode == "stack-all"),\
            "cfg.prompt_centric_layout should have stack_mode='stack-all' if n_quantiles > 0, so that it fits in 1 col"

        # Iterate over all the components in this single column, and add them all to the HTML object
        for component_config in layout.columns[0]:
            component = self.get_component_from_config(component_config)
            html_obj += component._get_html_data(
                cfg = component_config,
                decode_fn = decode_fn,
                column = column_idx,
                id_suffix = str(column_idx),
                component_specific_kwargs = dict(  # only used for the SequenceData (the prompt)
                    bold_idx = bold_idx,
                    permanent_line = True,
                    hover_above = True,
                ),
            )
        
        # Add the title in JavaScript, and the empty title element in HTML
        html_obj.html_data[column_idx] = f"<div id='column-{column_idx}-title'></div>\n{html_obj.html_data[column_idx]}"
        html_obj.js_data["gridColumnTitlesData"] = {str(column_idx): title}

        return html_obj




@dataclass_json
@dataclass
class SaeVisData(SaveableDataclass):
    '''
    This contains all the data necessary for constructing the feature-centric visualization, over multiple 
    features (i.e. being able to navigate through them). See diagram in readme:

        https://github.com/callummcdougall/sae_vis#data_storing_fnspy
     
    feature_data_dict:      Contains the data for each individual feature-centric vis.
    feature_act_quantiles:  Contains the quantiles of activation values for each feature (used for rank-ordering feats
                            in the prompt-centric vis).
    cfg:                    The vis config, used for the both the data gathering and the vis layout.
    '''
    feature_data_dict: dict[int, FeatureData] = field(default_factory=dict)
    feature_act_quantiles: QuantileCalculator = field(default_factory=QuantileCalculator)
    cfg: SaeVisConfig = field(default_factory=SaeVisConfig)

    # Some more attributes which we won't save to disk when we save this class
    model: Optional[HookedTransformer] = None
    encoder: Optional[AutoEncoder] = None
    encoder_B: Optional[AutoEncoder] = None

    def update(self, other: "SaeVisData") -> None:
        '''
        Updates a SaeVisData object with the data from another SaeVisData object. This is useful during the
        `get_feature_data` function, since this function is broken up into different groups of features then merged
        together.
        '''
        if other is None: return
        self.feature_data_dict.update(other.feature_data_dict)
        self.feature_act_quantiles.update(other.feature_act_quantiles)

    @classmethod
    def create(
        cls,
        encoder: AutoEncoder,
        model: HookedTransformer,
        tokens: Int[Tensor, "batch seq"],
        cfg: SaeVisConfig,
        encoder_B: Optional[AutoEncoder] = None,
    ) -> "SaeVisData":
        
        from sae_vis.data_fetching_fns import get_feature_data

        sae_vis_data = get_feature_data(
            encoder = encoder,
            model = model,
            tokens = tokens,
            cfg = cfg,
            encoder_B = encoder_B,
        )
        sae_vis_data.cfg = cfg
        sae_vis_data.model = model
        sae_vis_data.encoder = encoder
        sae_vis_data.encoder_B = encoder_B

        return sae_vis_data

    def save_feature_centric_vis(
       self,
       filename: str | Path,
       feature_idx: Optional[int] = None,
    ) -> None:
        '''
        Returns the HTML string for the view which lets you navigate between different features.

        Args:
            model:          Used to get the tokenizer (for converting token IDs to string tokens).
            filename:       The HTML filepath we'll save the visualization to.
            feature_idx:    This is the default feature index we'll start on. If None, we use the first feature.
        '''
        # Initialize the object we'll eventually get_html from
        HTML_OBJ = HTML()

        # Set the default argument for the dropdown (i.e. when the page first loads)
        first_feature = next(iter(self.feature_data_dict)) if (feature_idx is None) else feature_idx

        # Get tokenize function (we only need to define it once)
        assert self.model is not None
        decode_fn = get_decode_html_safe_fn(self.model.tokenizer)

        # For each FeatureData object, we get the html_obj for it's feature-centric vis, and merge it with HTML_OBJ
        for feature, feature_data in self.feature_data_dict.items():

            # Get the HTML object for a single-feature view
            html_obj = feature_data._get_html_data_feature_centric(decode_fn)

            # Add the JavaScript, and arbitrarily set html_str to be the first feature's html_str (they're all same!)
            HTML_OBJ.js_data[str(feature)] = deepcopy(html_obj.js_data)
            if feature == first_feature: HTML_OBJ.html_data = deepcopy(html_obj.html_data)
        
        # Save our full HTML
        HTML_OBJ.get_html(
            layout = self.cfg.feature_centric_layout,
            filename = filename,
            first_key = str(first_feature),
        )

    def save_prompt_centric_vis(
        self,
        prompt: str,
        filename: str | Path,
        metric: Optional[str] = None,
        seq_pos: Optional[int] = None,
        num_top_features: int = 10,
    ):
        '''
        Returns the HTML string for the view which lets you navigate between different features.

        Args:
            prompt:     The user-input prompt.
            model:      Used to get the tokenizer (for converting token IDs to string tokens).
            filename:   The HTML filepath we'll save the visualization to.
            metric:     This is the default scoring metric we'll start on. If None, we use 'act_quantile'.
            seq_pos:    This is the default seq pos we'll start on. If None, we use 0.
        '''
        # Initialize the object we'll eventually get_html from
        HTML_OBJ = HTML()

        # Run forward passes on our prompt, and store the data within each FeatureData object as `self.prompt_data` as
        # well as returning the scores_dict (which maps from score hash to a list of feature indices & formatted scores)
        from sae_vis.data_fetching_fns import get_prompt_data
        scores_dict = get_prompt_data(
            sae_vis_data = self,
            prompt = prompt,
            num_top_features = num_top_features,
        )

        # Get all possible values for dropdowns
        str_toks = self.model.tokenizer.tokenize(prompt) # type: ignore
        str_toks = [t.replace("|", "│") for t in str_toks] # vertical line -> pipe (hacky, so key splitting on | works)
        str_toks_list = [f"{t!r} ({i})" for i, t in enumerate(str_toks)]
        metric_list = ["act_quantile", "act_size", "loss_effect"]

        # Get default values for dropdowns
        first_metric = "act_quantile" or metric
        first_seq_pos = str_toks_list[0 if seq_pos is None else seq_pos]
        first_key = f"{first_metric}|{first_seq_pos}"

        # Get tokenize function (we only need to define it once)
        assert self.model is not None
        decode_fn = get_decode_html_safe_fn(self.model.tokenizer)

        # For each (metric, seqpos) object, we merge the prompt-centric views of each of the top features, then we merge
        # these all together into our HTML_OBJ
        for (_metric, _seq_pos) in itertools.product(metric_list, range(len(str_toks))):

            # Create the key for this given combination of metric & seqpos, and get our top features & scores
            key = f"{_metric}|{str_toks_list[_seq_pos]}"
            if key not in scores_dict:
                continue
            feature_idx_list, scores_formatted = scores_dict[key]
            
            # Create HTML object, to store each feature column for all the top features for this particular key
            html_obj = HTML()

            for i, (feature_idx, score_formatted) in enumerate(zip(feature_idx_list, scores_formatted)):
                
                # Get HTML object at this column (which includes JavaScript to dynamically set the title)
                html_obj += self.feature_data_dict[feature_idx]._get_html_data_prompt_centric(
                    decode_fn = decode_fn,
                    column_idx = i,
                    bold_idx = _seq_pos,
                    title = f"<h3>#{feature_idx}<br>{METRIC_TITLES[_metric]} = {score_formatted}</h3><hr>",
                )

            # Add the JavaScript (which includes the titles for each column)
            HTML_OBJ.js_data[key] = deepcopy(html_obj.js_data)

            # Set the HTML data to be the one with the most columns (since different options might have fewer cols)
            if len(HTML_OBJ.html_data) < len(html_obj.html_data):
                HTML_OBJ.html_data = deepcopy(html_obj.html_data)
        
        # Check our first key is in the scores_dict (if not, we should pick a different key)
        assert first_key in scores_dict, f"Key {first_key} not found in {scores_dict.keys()=}. Have you tried \
computing your initial data with more features and/or tokens, to make sure you have enough positive examples?"

        # Save our full HTML
        HTML_OBJ.get_html(
            layout = self.cfg.prompt_centric_layout,
            filename = filename,
            first_key = first_key,
        )


