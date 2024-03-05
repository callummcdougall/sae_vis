import numpy as np
import json
from typing import List
import torch
from torch import Tensor
from eindex import eindex
from typing import Optional, List, Dict, Tuple, Literal, Union
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
import einops
from jaxtyping import Float
import re
from rich import print as rprint
from rich.table import Table
from transformer_lens import utils
import time

Arr = np.ndarray

from sae_vis.utils_fns import (
    to_str_tokens,
    QuantileCalculator,
    TopK,
    merge_lists,
    extract_and_remove_scripts,
    device,
)
from sae_vis.html_fns import (
    generate_seq_html,
    generate_left_tables_html,
    generate_middle_plots_html,
    CSS,
    JS_HOVERTEXT_SCRIPT,
    adjust_hovertext,
    grid_item,
)




FEATURE_VIZ_PARAMS = {
    "hook_point": "The hook point we're using to extract the activations (if we're using a TransformerLens model). \
This should be the full-length hook name, i.e. the thing returned by `utils.get_act_name`.",
    "features": "The feature(s) we're analyzing. If None, we analyze all of the AutoEncoder's features.",
    "minibatch_size_features": "Num features in each batch of calculations (i.e. we break up the features to avoid OOM \
errors).",
    "minibatch_size_tokens": "Number of seqs in each forward pass (i.e. we break up the tokens in our batch to avoid \
OOM errors). Note, this is lower-level than breaking up by features (i.e. we break up the calculation by features \
first, then within each feature group we break it up by tokens).",
    "include_left_tables": "Whether to include the left-hand tables in the main visualization.",
    "rows_in_left_tables": "Number of rows in the tables on the left hand side of the main visualization.",
    "buffer": "How many tokens to add as context to each sequence. We also avoid choosing tokens from within these \
buffer positions.",
    "n_groups": "Number of quantile groups for the sequences on the right hand side (not including top-k).",
    "first_group_size": "Number of sequences in the top-k group.",
    "other_groups_size": "Number of sequences in each of the quantile groups.",
    "border": "Whether to include the shadow border around the main visualization.",
    "seq_width": "The max width of the sequences in the main visualization. If None, they'll be full-width (to contain \
the longest seq).",
    "seq_height": "The max height of the sequences in the main visualization. If None, they'll be full-height (to contain \
all the sequences).",
    "seed": "Seed for random number generation (used for choosing the top-k sequences).",
    "verbose": "Whether to print out the time taken for each task, and the estimated time for all features (note, this \
can be very noisy when the number of features is small).",
}


def save_json(
    obj,
    filename: Optional[str] = None,
    return_dict: bool = False,
):
    '''
    Saves the object to a JSON file (and optionally returns dict).

    Args:
        obj: the object to be saved
        filename: if None then we don't save, if string then we save here
        return_dict: if True then we return the dict, if False then we return nothing
    '''
    t0 = time.time()
    assert (filename is not None) or return_dict,\
        "You're doing nothing in this function. Either specify a filename or set return_dict=True."

    # Convert the object to a dictionary, and save it if necessary
    obj_to_dict = obj.to_dict()
    if filename is not None:
        assert filename.endswith(".json"), "Expected filename to end with '.json'"
        json.dump(obj_to_dict, open(filename, "w"))
        # print(f"Saved to JSON in {time.time() - t0:.2f} seconds")
    
    # If we're returning the dict, then return it here
    if return_dict:
        # print(f"Converted to dict in {time.time() - t0:.2f} seconds total")
        return obj_to_dict


# Function to load a dataclass object from a dictionary
def load_json(
    dataclass_type,
    data_dict: Optional[Dict] = None,
    filename: Optional[str] = None,
):
    '''
    Loads a dataclass object from a saved filename (or directly from a dict).

    Args:
        dataclass_type: the dataclass type we're loading
        data_dict: dictionary containing the data (if not None)
        filename: filename we're loading data from (if not None)
    '''
    t0 = time.time()
    assert int(filename is None) + int(data_dict is None) == 1,\
        "Expected exactly one of filename or data_dict should be supplied."
    
    # If we're using a filename, then load in the data (as a dictionary)
    if filename is not None:
        data_dict = json.load(open(filename, "r"))
        # print(f"Loaded in {time.time() - t0:.2f} seconds")
    
    # Get dataclass object from dictionary
    data_dict = dataclass_type.from_dict(data_dict)
    # print(f"Converted to dataclass in {time.time() - t0:.2f} seconds total")
    return data_dict


@dataclass_json
@dataclass
class FeatureVisParams:

    hook_point: Optional[str] = None

    features: Optional[Union[int, List[int]]] = None
    minibatch_size_features: int = 256
    minibatch_size_tokens: int = 64

    include_left_tables: bool = True
    rows_in_left_tables: int = 3
    buffer: tuple = (5, 5)
    n_groups: int = 10
    first_group_size: int = 20
    other_groups_size: int = 5

    border: bool = True
    seq_width: Optional[int] = 440
    seq_height: Optional[int] = None

    seed: Optional[int] = 0
    verbose: bool = True

    def help(self) -> None:
        '''
        Prints out the values & meanings of all the parameters. Also highlights when they're different to default.
        '''
        default_fvp = FeatureVisParams()
        table = Table("Parameter", "Value", "Meaning", title="FeatureVisParams", show_lines=True)
        for param, meaning in FEATURE_VIZ_PARAMS.items():
            value = str(getattr(self, param))
            default_value = str(getattr(default_fvp, param))
            value_formatted = value if value == default_value else f"[b dark_orange]{value}[/]"
            table.add_row(param, value_formatted, meaning)
        rprint(table)


@dataclass_json
@dataclass
class HistogramData:
    '''
    Class for storing all the data necessary to construct a histogram, like we see in the middle plots.

    Args:
        data: Tensor
            The data we're going to be plotting in the histogram.
        n_bins: int
            The number of bins we want to use in the histogram.
        tickmode: str
            How we want to choose the tick values for the histogram. This is pretty hacky atm.
        
        line_posn: Optional[Union[float, List[float]]]
            The possible positions of vertical lines we want to put on the histogram. This is a list because in the
            prompt-centric visualisation there will be a different line for each token.
        line_text: Optional[Union[str]]
            The text we want to display on the line.

    Unlike many other classes here, this class actually does computation on its inputs. This is because
    we don't need to store the entire `data` tensor, only the height of each bar. This is why we eventually
    get the following 3 attributes, in place of the `data` tensor:

        bar_heights (List[float]): The height of each bar in the histogram.
        bar_values (List[float]): The value of each bar in the histogram.
        tick_vals (List[float]): The tick values we want to use for the histogram.
    '''
    bar_heights: List[float] = field(default_factory=list)
    bar_values: List[float] = field(default_factory=list)
    tick_vals: List[float] = field(default_factory=list)
    line_posn: Optional[List[float]] = None
    line_text: Optional[str] = None

    @classmethod
    def from_data(
        cls,
        data: Tensor,
        n_bins: int,
        tickmode: str,
        line_posn: Optional[List[float]] = None,
        line_text: Optional[str] = None,
    ) -> "HistogramData":
        '''
        Returns a HistogramData object, with data computed from the inputs. This is to support the goal of only storing
        the minimum necessary data (and making it serializable, for JSON saving).
        '''
        # There might be no data, if the feature never activates
        if data.numel() == 0:
            return HistogramData()

        # Get min and max of data
        max_value = data.max().item()
        min_value = data.min().item()

        # Divide range up into 40 bins
        bin_size = (max_value - min_value) / n_bins
        bin_edges = torch.linspace(min_value, max_value, n_bins + 1)
        # Calculate the heights of each bin
        bar_heights = torch.histc(data, bins=n_bins).int().tolist()
        bar_values = [round(x, 5) for x in (bin_edges[:-1] + bin_size / 2).tolist()]
        
        # Choose tickvalues (super hacky and terrible, should improve this!)
        assert tickmode in ["ints", "5 ticks"]

        if tickmode == "ints":
            top_tickval = int(max_value)
            tick_vals = torch.arange(0, top_tickval + 1, 1).tolist()
        elif tickmode == "5 ticks":
            # Ticks chosen in multiples of 0.1, so we have 3 on the longer side
            if max_value > -min_value:
                tickrange = 0.1 * int(1e-4 + max_value / (3 * 0.1)) + 1e-6
                num_positive_ticks = 3
                num_negative_ticks = int(-min_value / tickrange)
            else:
                tickrange = 0.1 * int(1e-4 + -min_value / (3 * 0.1)) + 1e-6
                num_negative_ticks = 3
                num_positive_ticks = int(max_value / tickrange)
            tick_vals = merge_lists(
                reversed([-tickrange * i for i in range(1, 1+num_negative_ticks)]), # negative values (if exist)
                [0], # zero (always is a tick)
                [tickrange * i for i in range(1, 1+num_positive_ticks)] # positive values
            )
            tick_vals = [round(t, 1) for t in tick_vals]

        return HistogramData(
            bar_heights = bar_heights,
            bar_values = bar_values,
            tick_vals = tick_vals,
            line_posn = line_posn,
            line_text = line_text,
        )



@dataclass_json
@dataclass
class SequenceData:
    '''
    Class to store data for a given sequence, which will be turned into an HTML visulisation. See the README for a
    diagram of how this class fits into the overall visualization.

    Always-visible data:
        token_ids: list of token IDs in the sequence
        feat_acts: sizes of activations on this sequence
        contribution_to_loss: effect on loss of this feature, for this particular token (neg = helpful)

    Data which is visible on hover:
        top5_token_ids: list of the top 5 logit-boosted tokens by this feature
        top5_logits: list of the corresponding 5 changes in logits for those tokens
        bottom5_token_ids: list of the bottom 5 logit-boosted tokens by this feature
        bottom5_logits: list of the corresponding 5 changes in logits for those tokens

    Metadata:
        filter: if this is true, we filter out the data which is zero (this saves space when we save the data).
    '''
    token_ids: List[int] = field(default_factory=list)
    feat_acts: List[float] = field(default_factory=list)
    contribution_to_loss: List[float] = field(default_factory=list)
    top5_token_ids: Optional[List[List[int]]] = None
    top5_logits: Optional[List[List[float]]] = None
    bottom5_token_ids: Optional[List[List[str]]] = None
    bottom5_logits: Optional[List[List[float]]] = None
    filter: bool = False

    def __post_init__(self) -> None:
        '''
        Filters the list of floats and ints, by removing any elements which are zero. Note - the absolute values of the
        floats are monotonic non-increasing, so we can assume that all the elements we keep will be the first elements
        of their respective lists.

        Also reduces precisions of feature activations & logits to 4dp.
        '''
        if self.filter:
            self.feat_acts = [round(f, 4) for f in self.feat_acts]
            self.top5_logits, self.top5_token_ids = self._filter(self.top5_logits, self.top5_token_ids)
            self.bottom5_logits, self.bottom5_token_ids = self._filter(self.bottom5_logits, self.bottom5_token_ids)

    def _filter(self, float_list: Optional[List[List[float]]], int_list: Optional[List[List[int]]]):
        if float_list is None:
            return None, None
        
        float_list = [[round(f, 4) for f in floats if abs(f) > 1e-6] for floats in float_list]
        int_list = [ints[:len(floats)] for ints, floats in zip(int_list, float_list)]

        for floats, ints in zip(float_list, int_list):
            assert len(floats) == len(ints), "Expected the same number of floats and ints"

        return float_list, int_list

    def __len__(self) -> int:
        return len(self.token_ids)
    
    def get_html(
        self,
        vocab_dict: Dict[int, str],
        hovertext: bool = True,
        bold_idx: Optional[int] = None,
        overflow_x: Literal["break", None] = None,
        max_act_color: Optional[float] = None,
    ) -> str:
        '''
        Args:
            vocab_dict: Dict[int, str]
                Used for mapping token IDs to string tokens
            hovertext: bool
                Determines if we add hovertext to this HTML (yes if the sequence is on its own, no otherwise)
            bold_idx: Optional[int]
                If supplied, then we bold the token at this index. Default is the middle token
            overflow_x: Literal["break", None]
                If supplied, then we use this as the value for the overflow-x CSS property (useful for prompt-centric
                view, where we wrap user-input prompt)
            max_act_color: Optional[float]:
                We use this as the most extreme value, for coloring tokens by activation
        '''
        html_str = generate_seq_html(
            vocab_dict,
            token_ids = self.token_ids,
            feat_acts = self.feat_acts,
            contribution_to_loss = self.contribution_to_loss,
            bold_idx = bold_idx if bold_idx is not None else len(self.token_ids) // 2, # bold the middle token by default
            pos_ids = self.top5_token_ids,
            neg_ids = self.bottom5_token_ids,
            pos_val = self.top5_logits,
            neg_val = self.bottom5_logits,
            overflow_x = overflow_x,
            max_act_color = max_act_color,
        )
        if hovertext: html_str += f"<script>{JS_HOVERTEXT_SCRIPT}</script>"
        return html_str


@dataclass_json
@dataclass
class SequenceGroupData:
    '''
    Class to store data for a given sequence group, which will be turned into an HTML visulisation. See the README for a
    diagram of how this class fits into the overall visualization.

    All the arguments are equivalent to those for SequenceData (except for `title` which is the header of the group),
    so see the SequenceData class for more information.
    '''
    title: str = ""
    seq_data: List[SequenceData] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.seq_data)
    
    def get_html(
        self,
        vocab_dict: Dict[int, str],
        group_size: Optional[int] = None,
        hovertext: bool = True,
        width: Optional[int] = 440,
        max_act_color: Optional[float] = None,
    ) -> str:
        '''
        This creates a single group of sequences, i.e. title plus some number of vertically stacked sequences.

        Args:
            vocab_dict: Dict[int, str]
                Used for converting token indices to string tokens.
            group_size: Optional[int]
                If supplied, we only use this many sequences from the group. If not supplied, we use all of them.
            hovertext: bool
                This adds the JavaScript hovertext. If this is the only sequence group in our visualization then we need
                to add the hovertext, but if we're generating multiple sequence groups then we add the hovertext at a
                higher level.
            width: Optional[int]
                If not None, then the sequence data will be wrapped in a div with this width. If None, then the
                sequences will be full-length (i.e. each column containing sequences will have variable width, to match
                the sequences).
            max_act_color: Optional[float]
                If supplied, then we use this as the most extreme value, for coloring the tokens by activation value.
        '''
        # Get the styles for the div
        width_style: str = "" if width is None else f"width:{width}px;"

        # Get the data that will go into the div
        html_contents = "".join([
            seq.get_html(vocab_dict, hovertext=False, max_act_color=max_act_color)
            for seq in self.seq_data[:group_size]
        ])

        # Assemble the full div
        html_str: str = f'''
<h4>{self.title}</h4>

<div class="seq-scroll" style="{width_style}">
    {html_contents}
</div>
'''
        # If necessary, add hovertext
        if hovertext:
            html_str += f"<script>{JS_HOVERTEXT_SCRIPT}</script>"

        return html_str



@dataclass_json
@dataclass
class SequenceMultiGroupData:
    '''
    Class to store data for multiple sequence groups, which will be turned into an HTML visulisation. See the README for
    a diagram of how this class fits into the overall visualization.

    See the SequenceGroupData and SequenceData classes for more information on the arguments.
    '''
    seq_group_data: List[SequenceGroupData] = field(default_factory=list)

    def __getitem__(self, idx: int) -> SequenceGroupData:
        return self.seq_group_data[idx]

    def get_html(
        self,
        vocab_dict: Dict[int, str],
        width: Optional[int] = 440,
        height: Optional[int] = None,
        hovertext: bool = True,
    ) -> str:
        '''
        Returns all the sequence groups' HTML, wrapped in grid-items (plus the JavaScript code at the end).
        '''
        # Get max activation value, over all sequences
        max_act_color: float = max([
            max(seq.feat_acts) for seq_group in self.seq_group_data for seq in seq_group.seq_data
        ])

        # Get the HTML for all the sequence groups (the first one is the top activations, the rest are quantiles)
        html_top, *html_quantiles = [
            sequences_group.get_html(
                vocab_dict = vocab_dict,
                max_act_color = max_act_color,
                width = width,
                hovertext = False,
            )
            for sequences_group in self.seq_group_data
        ]

        # Create a grid item for the first group, plus a grid item for every 3 quantiles, until we've used them all
        sequences_html = grid_item(html_top, height=height)
        while len(html_quantiles) > 0:
            L = min(3, len(html_quantiles))
            html_next, html_quantiles = html_quantiles[:L], html_quantiles[L:]
            sequences_html += grid_item(''.join(html_next), height=height)

        # If necessary, add the javascript
        return sequences_html + (f"<script>{JS_HOVERTEXT_SCRIPT}</script>" if hovertext else "")



@dataclass_json
@dataclass
class LeftTablesData:
    '''
    Class to store all the data used in the left-hand tables (i.e. neuron alignment, correlated neurons & features).

    Args:
        neuron_alignment...
            The data for the neuron alignment table (each of its 3 columns). In other words, the data containing which
            neurons in the transformer the encoder feature is most aligned with.

        correlated_neurons...
            The data for the correlated neurons table (each of its 3 columns). In other words, the data containing which
            neurons in the transformer are most correlated with the encoder feature.

        correlated_features...
            The data for the correlated features table (each of its 3 columns). In other words, the data containing
            which features in encoder-B are most correlated with those in the original encoder.
    '''
    neuron_alignment_indices: List[int] = field(default_factory=list)
    neuron_alignment_values: List[float] = field(default_factory=list)
    neuron_alignment_l1: List[float] = field(default_factory=list)
    correlated_neurons_indices: List[int] = field(default_factory=list)
    correlated_neurons_pearson: List[float] = field(default_factory=list)
    correlated_neurons_l1: List[float] = field(default_factory=list)
    correlated_features_indices: List[int] = field(default_factory=list)
    correlated_features_pearson: List[float] = field(default_factory=list)
    correlated_features_l1: List[float] = field(default_factory=list)

    def get_html(self) -> str:
        '''
        The `get_html` function returns the HTML for the left-hand tables, wrapped in a 'grid-item' div.
        '''
        # Generate & return the left-hand HTML tables
        html_str: str = generate_left_tables_html(
            self.neuron_alignment_indices,
            self.neuron_alignment_values,
            self.neuron_alignment_l1,
            self.correlated_neurons_indices,
            self.correlated_neurons_pearson,
            self.correlated_neurons_l1,
            self.correlated_features_indices,
            self.correlated_features_pearson,
            self.correlated_features_l1,
        )
        return grid_item(html_str)



@dataclass_json
@dataclass
class MiddlePlotsData:
    '''
    Class to store all the data used in the middle plots (i.e. activation density, logits table & histogram).

    Inputs:
        bottom10_token_ids: List[int]
            The token IDs corresponding to the 10 most negative logits.
        
        bottom10_logits: List[float]
            The 10 most negative logits (corresponding to the token IDs above).

        top10_token_ids: List[int]
            The token IDs corresponding to the 10 most positive logits.

        top10_logits: List[float]
            The 10 most positive logits (corresponding to the token IDs above).

        logits_histogram_data: HistogramData
            Used to generate the logits histogram.

        freq_histogram_data: HistogramData
            Used to generate the activation density histogram.

        frac_nonzero: float
            Used to generate the title of the activation density histogram.
    '''
    bottom10_token_ids: List[int]
    bottom10_logits: List[float]
    top10_token_ids: List[int]
    top10_logits: List[float]
    logits_histogram_data: HistogramData
    freq_histogram_data: HistogramData
    frac_nonzero: float

    def get_html(
        self,
        vocab_dict: Dict[int, str],
        compact: bool = False,
        histogram_line_idx: Optional[int] = None,
    ) -> str:
        '''
        Args:
            vocab_dict: Dict[int, str]
                Used for converting token indices to string tokens.
            compact: bool
                If True, then we make it horizontally compact (by putting the table and charts side by side).

            histogram_line_idx: Optional[int]
                If supplied, then we use this index to choose which line to add to the plots. If not supplied, we don't
                add a line.
        '''
        kwargs_for_line_annotations = {}
        if histogram_line_idx is not None:
            assert isinstance(self.logits_histogram_data.line_posn, list), "Expected list of line positions"
            assert isinstance(self.freq_histogram_data.line_posn, list), "Expected list of line positions"
            kwargs_for_line_annotations.update(dict(
                logits_line_posn = self.logits_histogram_data.line_posn[histogram_line_idx],
                logits_line_text = self.logits_histogram_data.line_text[histogram_line_idx],
                freq_line_posn = self.freq_histogram_data.line_posn[histogram_line_idx],
                freq_line_text = self.freq_histogram_data.line_text[histogram_line_idx],
            ))

        # Get the negative and positive background values (darkest when equals max abs). Easier when in tensor form
        max_value = max(np.absolute(self.bottom10_logits).max(), np.absolute(self.top10_logits).max())
        neg_bg_values = np.absolute(self.bottom10_logits) / max_value
        pos_bg_values = np.absolute(self.top10_logits) / max_value

        html_str = generate_middle_plots_html(
            neg_str = to_str_tokens(vocab_dict, self.bottom10_token_ids),
            neg_values = self.bottom10_logits,
            neg_bg_values = neg_bg_values,
            pos_str = to_str_tokens(vocab_dict, self.top10_token_ids),
            pos_values = self.top10_logits,
            pos_bg_values = pos_bg_values,

            freq_hist_data_bar_heights = self.freq_histogram_data.bar_heights,
            freq_hist_data_bar_values = self.freq_histogram_data.bar_values,
            freq_hist_data_tick_vals = self.freq_histogram_data.tick_vals,
            logits_hist_data_bar_heights = self.logits_histogram_data.bar_heights,
            logits_hist_data_bar_values = self.logits_histogram_data.bar_values,
            logits_hist_data_tick_vals = self.logits_histogram_data.tick_vals,

            frac_nonzero = self.frac_nonzero,
            compact = compact,
            **kwargs_for_line_annotations,
        )

        if grid_item:
            if compact:
                assert len(html_str) == 2, f"Expected 2 HTML strings, got {len(html_str)}"
                return "\n".join([
                    grid_item(html_str[0]),
                    grid_item(html_str[1], width=380)
                ])
            else:
                return grid_item(html_str)
        else:
            return html_str








@dataclass_json
@dataclass
class PromptData:
    '''
    This class contains all the data necessary to make a single prompt-centric visualization, i.e. each column here:

        https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/sae-demo-1.png

    Args:
        prompt_data: Dict[str, SequenceGroupData]
            This contains all the data which will be used to construct the sequences (at the bottom of the column).

        middle_plots_data: MiddlePlotsData
            This contains all the data which will be used to construct the middle plots (act density histogram, logits
            tables, and logits histogram).

        sequence_data: Dict[str, SequenceGroupData]
            This contains all the data which will be used to construct the sequences (at the bottom of the column).
    '''
    prompt_data: SequenceData
    middle_plots_data: MiddlePlotsData
    sequence_data: SequenceGroupData

    def get_html(
        self,
        title: str,
        vocab_dict: Dict[int, str],
        hovertext: bool = True,
        bold_idx: Optional[int] = None,
        width: Optional[int] = 440,
        histogram_line_idx: Optional[int] = None,
    ) -> str:
        '''
        Gets all the HTML for multiple sequences.
        '''
        seq_width: int | None = width - 20 if width is not None else None

        html_contents = f"""
<h3>{title}</h3>

{self.prompt_data.get_html(vocab_dict, hovertext, bold_idx, overflow_x="break")}
{self.middle_plots_data.get_html(vocab_dict, histogram_line_idx=histogram_line_idx)}
{self.sequence_data.get_html(vocab_dict, group_size=10, width=seq_width, hovertext=hovertext)}
"""
        return grid_item(html_contents, width=width)
        

@dataclass_json
@dataclass
class MultiPromptData:
    '''
    This class contains all the data necessary to make a full prompt-centric visualization, i.e. this thing:

        https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/sae-demo-1.png

    Args:
        prompt_str_toks: List[str]
            List of string-tokens of the prompt.

        prompt_data_dict: Dict[int, PromptData]
            Contains all the different possible PromptData objects we might be using in this visualisation,
            indexed by feature index.

        scores_dict:
            Maps every combination of (scoring metric, sequence posn) to a list of highest-scoring features.

            keys: str
                Concatenated (score_name, seq_pos), separated by a dash.
                E.g. "act_size-0", "act_quantile-1". "loss_effect-2".

            values: Tuple[List[int], List[str]]
                The 2 elements are the feature indices, and scores (as strings).

    '''
    prompt_str_toks: List[str] = field(default_factory=list)
    prompt_data_dict: Dict[int, PromptData] = field(default_factory=dict)
    scores_dict: Dict[str, Tuple[List[int], List[str]]] = field(default_factory=dict)

    def __getitem__(self, idx: int) -> PromptData:
        return self.prompt_data_dict[idx]

    def get_html(
        self,
        seq_pos: int,
        score_name: Literal["act_size", "act_quantile", "loss_effect"],
        vocab_dict: Dict[int, str],
        width: Optional[int] = 440,
    ) -> str:
        
        # Check arguments are valid, and if they are then get the score title (i.e. the how it'll appear in HTML)
        assert score_name in ["act_size", "act_quantile", "loss_effect"], f"Invalid score_name: {score_name!r}"
        assert 0 <= seq_pos < len(self.prompt_str_toks)
        if score_name == "loss_effect":
            assert seq_pos > 0, "Can't look at the loss effect on the first token in the prompt."
        score_title = {
            "act_size": "Activation Size",
            "act_quantile": "Activation Quantile",
            "loss_effect": "Loss Effect",
        }[score_name]

        key = f"{score_name}-{seq_pos}"
        feature_indices, scores_str = self.scores_dict[key]
        k = len(feature_indices)

        grid_items = ""

        # Iterate through each of the features we're creating a box for
        for (feature_idx, score_str) in zip(feature_indices, scores_str):

            # Add the HTML box for this feature
            grid_items += self[feature_idx].get_html(
                title = f"Feature #<a href='demo'>{feature_idx}</a><br>{score_title} = {score_str}<br>",
                vocab_dict = vocab_dict,
                hovertext = False,
                bold_idx = seq_pos,
                width = width,
                histogram_line_idx = seq_pos - 1 if (score_name == "loss_effect") else seq_pos,
            )

        # Change the ids from e.g. `histogram-logits` to `histogram-logits-1`, `histogram-logits-2`, ... so that the JavaScript works
        # TODO - change this to be something less hacky?
        for histogram_id in ["histogram-logits", "histogram-acts"]:
            k_list = [i for i in range(1, k+1) for _ in range(2)]
            grid_items = re.sub(histogram_id, lambda m: f"{histogram_id}-{k_list.pop(0)}", grid_items)

        html_str = f"""
<style>
    {CSS}
</style>

<div class='grid-container'>
    {grid_items}
</div>

<script>
    {JS_HOVERTEXT_SCRIPT}
</script>
"""
        # This saves a lot of space, by removing the hidden items from display
        # TODO - this is also kinda hacky and should be improved
        pattern = r'<div class="half-width-container" style="display: none;">(.*?)</div>'
        html_str = re.sub(pattern, "", html_str, flags=re.DOTALL)

        # TODO - debug why I need to do these things
        html_str = html_str.replace("Ċ", "&bsol;n")
        html_str = adjust_hovertext(html_str)
        return html_str





@dataclass_json
@dataclass
class FeatureData:
    '''
    This class contains all the data necessary to make a single prompt-centric visualization, like this one:

        https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/sae-demo-1.png

    Args (data-containing):
        sequence_data: Dict[str, SequenceGroupData]
            This contains all the data which will be used to construct the right-hand sequences.

        left_tables_data: LeftTablesData
            This contains all the data which will be used to construct the left-hand tables (neuron alignment, correlated
            neurons, and correlated features).

        middle_plots_data: MiddlePlotsData
            This contains all the data which will be used to construct the middle plots (act density histogram, logits
            tables, and logits histogram).

    Args (non-data-containing):
        feature_idx: int
            Index of the feature in question.

        vocab_dict: Dict[int, str]
            Used for the string tokens in the sequence visualisations.

        fvp: FeatureVisParams
            Contains several other parameters which are important in the visualisation.
    '''
    sequence_data: SequenceMultiGroupData
    left_tables_data: Optional[LeftTablesData]
    middle_plots_data: MiddlePlotsData

    feature_idx: int
    fvp: FeatureVisParams

    def get_html(
        self,
        vocab_dict: Dict[int, str],
        split_scripts: bool = False,
    ) -> str:
        '''
        Args:
            width: Optional[int]
                If not None, then the sequence data will be wrapped in a div with this width. If sequences overflow past
                this width, then there is a horizontal scrollbar. If it is None, then the sequences will be full-length
                (i.e. each column containing sequences will be a variable width, to match the sequences).
            split_scripts: bool
                If True, then we return a tuple of (javascript, html without javascript). This was useful when creating
                the visualization for my blog (perfectlynormal.co.uk/blog-sae) because the javascript has to be inserted
                in in a specific way to make it run).
        '''
        # Get sequence data HTML
        sequence_html: str = self.sequence_data.get_html(
            vocab_dict,
            width = self.fvp.seq_width,
            height = self.fvp.seq_height,
            hovertext = True,
        )

        # Get other HTML (split depending on whether we include the left tables or not)
        if self.fvp.include_left_tables:
            left_tables_html: str = self.left_tables_data.get_html()
            middle_plots_data: str = self.middle_plots_data.get_html(vocab_dict)
        else:
            left_tables_html = ""
            middle_plots_data: str = self.middle_plots_data.get_html(vocab_dict, compact=True)

        # Get the CSS, and make appropriate edits to it
        css = CSS
        # If no border, then delete it from the CSS
        if not(self.fvp.border):
            css = css.replace("border: 1px solid #e6e6e6;", "").replace("box-shadow: 0 5px 5px rgba(0, 0, 0, 0.25);", "")

        html_str = f"""
<style>
    {css}
</style>

<div class='grid-container'>
    {left_tables_html}
    {middle_plots_data}
    {sequence_html}
</div>

<script>
    {JS_HOVERTEXT_SCRIPT}
</script>
"""
        # This saves a lot of space, by removing the hidden items from display
        pattern = r'<div class="half-width-container" style="display: none;">(.*?)</div>'
        html_str = re.sub(pattern, "", html_str, flags=re.DOTALL)
        
        # idk why this bug is here, for representing newlines the wrong way
        html_str = html_str.replace("Ċ", "&bsol;n")
        html_str = adjust_hovertext(html_str)
        
        if split_scripts:
            return extract_and_remove_scripts(html_str) # (scripts, html_str)
        else:
            return html_str
    
@dataclass_json
@dataclass
class MultiFeatureData:
    '''
    Class for storing data for multiple different features, as well as data which is used to sort the features (for now
    this just means quantile data on the feature activations and loss effects).
    '''
    feature_data_dict: Dict[int, FeatureData] = field(default_factory=dict)
    feature_act_quantiles: QuantileCalculator = field(default_factory=QuantileCalculator)

    # TODO - think about adding loss_quantiles: QuantileCalculator
    # They're potentially slower because we need to compute logits-after-ablation for all tokens rather than just the
    # tokens in our sequence groups (by default we don't compute logits for all tokens)

    def __getitem__(self, idx: int) -> FeatureData:
        return self.feature_data_dict[idx]
    
    def keys(self) -> List[int]:
        return list(self.feature_data_dict.keys())

    def values(self) -> List[FeatureData]:
        return list(self.feature_data_dict.values())

    def items(self) -> List[Tuple[int, FeatureData]]:
        return list(self.feature_data_dict.items())
    
    def __len__(self) -> int:
        return len(self.feature_data_dict)

    def update(self, other: "MultiFeatureData") -> None:
        '''
        Updates a MultiFeatureData object with the data from another MultiFeatureData object.
        '''
        if other is None: return
        self.feature_data_dict.update(other.feature_data_dict)
        self.feature_act_quantiles.update(other.feature_act_quantiles)



class BatchedCorrCoef:
    '''
    This class allows me to calculate corrcoef (both Pearson and cosine sim) between two
    batches of vectors without needing to store them all in memory.

    x.shape = (X, N), y.shape = (Y, N), and we calculate every pairwise corrcoef between
    the X*Y pairs of vectors.

    It's based on the following formulas (for vectors).

        cos_sim(x, y) = xy_sum / ((x2_sum ** 0.5) * (y2_sum ** 0.5))

        pearson_corrcoef(x, y) = num / denom

            num = n * xy_sum - x_sum * y_sum
            denom = (n * x2_sum - x_sum ** 2) ** 0.5 * (n * y2_sum - y_sum ** 2) ** 0.5

        ...and all these quantities (x_sum, xy_sum, etc) can be tracked on a rolling basis.

    When we take `.topk`, we're taking this over the y-tensor, in other words we want to keep
    the X dimension preserved (usually because X is our num_feats dimension).
    '''
    def __init__(self):
        self.n = 0
        self.x_sum = 0
        self.y_sum = 0
        self.xy_sum = 0
        self.x2_sum = 0
        self.y2_sum = 0

    def update(self, x: Float[Tensor, "X N"], y: Float[Tensor, "Y N"]):
        assert x.ndim == 2 and y.ndim == 2, "Both x and y should be 2D"
        assert x.shape[-1] == y.shape[-1], "x and y should have the same size in the last dimension"
        
        self.n += x.shape[-1]
        self.x_sum += einops.reduce(x, "X N -> X", "sum")
        self.y_sum += einops.reduce(y, "Y N -> Y", "sum")
        self.xy_sum += einops.einsum(x, y, "X N, Y N -> X Y")
        self.x2_sum += einops.reduce(x ** 2, "X N -> X", "sum")
        self.y2_sum += einops.reduce(y ** 2, "Y N -> Y", "sum")

    def corrcoef(self) -> Tuple[Float[Tensor, "X Y"], Float[Tensor, "X Y"]]:
        cossim_numer = self.xy_sum
        cossim_denom = torch.sqrt(torch.outer(self.x2_sum, self.y2_sum)) + 1e-6
        cossim = cossim_numer / cossim_denom

        pearson_numer = self.n * self.xy_sum - torch.outer(self.x_sum, self.y_sum)
        pearson_denom = torch.sqrt(torch.outer(self.n * self.x2_sum - self.x_sum ** 2, self.n * self.y2_sum - self.y_sum ** 2)) + 1e-6
        pearson = pearson_numer / pearson_denom

        return pearson, cossim

    def topk_pearson(self, k: int, largest: bool = True) -> Tuple[TopK, Arr]:
        '''
        First element
        Returns the topk corrcoefs, using Pearson (and taking this over the y-tensor)
        '''
        pearson, cossim = self.corrcoef()
        X, Y = cossim.shape
        # Get pearson topk by actually taking topk
        pearson_topk = TopK(pearson, k, largest) # shape (X, k)
        # Get cossim topk by indexing into cossim with the indices of the pearson topk: cossim[X, pearson_indices[X, k]]
        cossim_values = utils.to_numpy(eindex(cossim, pearson_topk.indices, "X [X k]"))
        return pearson_topk, cossim_values




