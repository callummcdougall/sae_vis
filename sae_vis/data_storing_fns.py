import numpy as np
from typing import List
import torch
from torch import Tensor
from eindex import eindex
from typing import Optional, List, Dict, Tuple, Literal, Union
from dataclasses import dataclass
import einops
from jaxtyping import Float
import re
from rich import print as rprint
from rich.table import Table

Arr = np.ndarray

from sae_vis.utils_fns import (
    to_str_tokens,
    QuantileCalculator,
    TopK,
    merge_lists,
    extract_and_remove_scripts,
)
from sae_vis.html_fns import (
    generate_seq_html,
    generate_left_tables_html,
    generate_middle_plots_html,
    CSS,
    JS_HOVERTEXT_SCRIPT,
    adjust_hovertext,
)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")




FEATURE_VIZ_PARAMS = {
    "total_batch_size": "Total number of sequences in our batch",
    "minibatch_size": "Number of seqs in each forward pass (i.e. we break up the total_batch_size to avoid OOM errors)",
    "total_features": "Total number of features we're analyzing",
    "minibatch_size_features": "Num features in each forward pass (i.e. we break up the total_features to avoid OOM errors)",
    "include_left_tables": "Whether to include the left-hand tables in the main visualization",
    "rows_in_left_tables": "Number of rows in the tables on the left hand side of the main visualization",
    "buffer": "How many posns to avoid at the start & end of sequences (so we see the surrounding context)",
    "n_groups": "Number of quantile groups for the sequences on the right hand side (not including top-k and bottom-k)",
    "first_group_size": "Number of sequences in the top-k and bottom-k groups",
    "other_groups_size": "Number of sequences in the other groups (i.e. the `n_groups` groups of quantiles)",
    "border": "Whether to include the shadow border around the main visualization",
    "verbose": "Whether to print out the time taken for each task, and the estimated time for all features",
}


@dataclass
class FeatureVizParams:

    total_batch_size: int = 2048
    minibatch_size: int = 64

    total_features: int = 1024
    minibatch_size_features: int = 256

    include_left_tables: bool = True
    rows_in_left_tables: int = 3
    buffer: tuple = (5, 5)
    n_groups: int = 10
    first_group_size: int = 20
    other_groups_size: int = 5
    border: bool = True

    verbose: bool = False

    def help(self) -> None:
        '''
        Prints out the values & meanings of all the parameters. Also highlights when they're different to default.
        '''
        default_fvp = FeatureVizParams()
        table = Table("Parameter", "Value", "Meaning", title="FeatureVizParams", show_lines=True)
        for param, meaning in FEATURE_VIZ_PARAMS.items():
            value = str(getattr(self, param))
            default_value = str(getattr(default_fvp, param))
            value_formatted = value if value == default_value else f"[b dark_orange]{value}[/]"
            table.add_row(param, value_formatted, meaning)
        rprint(table)


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
    def __init__(
        self,
        data: Tensor,
        n_bins: int,
        tickmode: str,
        line_posn: Optional[List[float]] = None,
        line_text: Optional[str] = None,
    ):
        self.line_posn = line_posn
        self.line_text = line_text

        if data.numel() == 0:
            self.bar_heights = []
            self.bar_values = []
            self.tick_vals = []
            return

        # Get min and max of data
        max_value = data.max().item()
        min_value = data.min().item()

        # divide range up into 40 bins
        bin_size = (max_value - min_value) / n_bins
        bin_edges = torch.linspace(min_value, max_value, n_bins + 1)
        # calculate the heights of each bin
        bar_heights = torch.histc(data, bins=n_bins)
        bar_values = bin_edges[:-1] + bin_size / 2
        
        # choose tickvalues (super hacky and terrible, should improve this)
        assert tickmode in ["ints", "5 ticks"]

        if tickmode == "ints":
            top_tickval = int(max_value)
            tick_vals = torch.arange(0, top_tickval + 1, 1).tolist()
        elif tickmode == "5 ticks":
            # ticks chosen in multiples of 0.1, so we have 3 on the longer side
            if max_value > -min_value:
                tickrange = 0.1 * int(1e-4 + max_value / (3 * 0.1))
                num_positive_ticks = 3
                num_negative_ticks = int(-min_value / tickrange)
            else:
                tickrange = 0.1 * int(1e-4 + -min_value / (3 * 0.1))
                num_negative_ticks = 3
                num_positive_ticks = int(max_value / tickrange)
            tick_vals = merge_lists(
                reversed([-tickrange * i for i in range(1, 1+num_negative_ticks)]), # negative values (if exist)
                [0], # zero (always is a tick)
                [tickrange * i for i in range(1, 1+num_positive_ticks)] # positive values
            )

        self.bar_heights = bar_heights.tolist()
        self.bar_values = bar_values.tolist()
        self.tick_vals = tick_vals



@dataclass
class SequenceData:
    '''
    Class to store data for a given sequence, which will be turned into a JavaScript visulisation. See here for a
    diagram of how this class fits into the overall visualization: https://shorturl.at/rCQ04

    Always-visible data:
        str_tokens: list of string tokens in the sequence
        feat_acts: sizes of activations on this sequence
        contribution_to_loss: effect on loss of this feature, for this particular token (neg = helpful)

    Data which is visible on hover:
        top5_str_tokens: list of the top 5 logit-boosted tokens by this feature
        top5_logit_changes: list of the corresponding 5 changes in logits for those tokens
        bottom5_str_tokens: list of the bottom 5 logit-boosted tokens by this feature
        bottom5_logit_changes: list of the corresponding 5 changes in logits for those tokens
    '''
    token_ids: List[str]
    feat_acts: List[float]
    contribution_to_loss: List[float]
    top5_token_ids: Optional[List[List[str]]] = None
    top5_logit_contributions: Optional[List[List[float]]] = None
    bottom5_token_ids: Optional[List[List[str]]] = None
    bottom5_logit_contributions: Optional[List[List[float]]] = None

    def __len__(self):
        return len(self.token_ids)

    def __str__(self):
        return f"SequenceData({''.join(self.token_ids)})"
    
    def __post_init__(self):
        '''Filters down the data, by deleting the "on hover" information if the activations are zero.'''
        self.top5_logit_contributions, self.top5_token_ids = self._filter(self.top5_logit_contributions, self.top5_token_ids)
        self.bottom5_logit_contributions, self.bottom5_token_ids = self._filter(self.bottom5_logit_contributions, self.bottom5_token_ids)

    def _filter(self, float_list: List[List[float]], int_list: List[List[str]]):
        if float_list is None:
            return None, None
        float_list = [[f for f in floats if f != 0] for floats in float_list]
        int_list = [[i for i, f in zip(ints, floats)] for ints, floats in zip(int_list, float_list)]
        return float_list, int_list
    
    def get_html(
        self,
        vocab_dict: Dict[int, str],
        hovertext: bool = True,
        bold_idx: Optional[int] = None,
        overflow_x: Literal["break", None] = None,
    ) -> str:
        '''
        hovertext determines whether we add hovertext to this HTML (yes if the sequence is on its own, no otherwise).
        '''
        html_str = generate_seq_html(
            vocab_dict,
            token_ids = self.token_ids,
            feat_acts = self.feat_acts,
            contribution_to_loss = self.contribution_to_loss,
            bold_idx = bold_idx if bold_idx is not None else len(self.token_ids) // 2, # bold the middle token by default
            pos_ids = self.top5_token_ids,
            neg_ids = self.bottom5_token_ids,
            pos_val = self.top5_logit_contributions,
            neg_val = self.bottom5_logit_contributions,
            overflow_x = overflow_x,
        )
        if hovertext:
            html_str += f"<script>{JS_HOVERTEXT_SCRIPT}</script>"
        return html_str



class SequenceGroupData:
    '''
    Class to store data for a given sequence group, which will be turned into a JavaScript visulisation. See here for a
    diagram of how this class fits into the overall visualization: https://shorturl.at/rCQ04

    All the arguments are equivalent to those for SequenceData (except for `title` which is the header of the group),
    so see the SequenceData class for more information.
    '''
    def __init__(self, title: str, seq_data: List[SequenceData]):
        self.title = title
        self.seq_data = seq_data
    
    def __len__(self) -> int:
        return len(self.seq_data)
    
    def get_html(
        self,
        vocab_dict: Dict[int, str],
        group_size: Optional[int] = None,
        hovertext: bool = True,
        overflow_x: Literal["scroll", "hidden"] = "scroll",
        width: Optional[int] = 420,
    ):
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
            overflow_x: Literal["scroll", "hidden"]
                If "scroll", then we add a horizontal scrollbar to the div containing the sequences. If "hidden", then
                we don't add a scrollbar (and the sequences will be cut off if they're too long).
            width: Optional[int]
                If not None, then the sequence data will be wrapped in a div with this width. If None, then the
                sequences will be full-length (i.e. each column containing sequences will have variable width, to match
                the sequences).
        '''
        style = "" if width is None else f"style='width:{width}px;'"
        html_str = f'<h4>{self.title}</h4><div class="seq-{overflow_x}" {style}>'
        
        seqs = self.seq_data if group_size is None else self.seq_data[:group_size]
        for seq in seqs:
            # We return the HTML for seq without hovertext JavaScript (cause we only need to do hovertext once)
            html_str += seq.get_html(vocab_dict=vocab_dict, hovertext=False)
        
        html_str += "</div>"

        if hovertext:
            html_str += f"<script>{JS_HOVERTEXT_SCRIPT}</script>"

        return html_str




class SequenceMultiGroupData:
    '''
    Class to store data for multiple sequence groups, which will be turned into a JavaScript visulisation. See here for
    a diagram of how this class fits into the overall visualization: https://shorturl.at/rCQ04

    See the SequenceGroupData and SequenceData classes for more information on the arguments.
    '''
    def __init__(self, seq_group_data: List[SequenceGroupData]):
        self.seq_group_data = seq_group_data

    def __getitem__(self, idx: int) -> SequenceGroupData:
        return self.seq_group_data[idx]

    def get_html(
        self,
        vocab_dict: Dict[int, str],
        hovertext: bool = True
    ) -> str:
        '''
        Returns all the sequence groups' HTML, wrapped in grid-items (plus the JavaScript code at the end).
        '''
        # Get the HTML for all the sequence groups (the first one is the top activations, the rest are quantiles)
        html_top, *html_quantiles = [
            sequences_group.get_html(vocab_dict=vocab_dict, hovertext=False)
            for sequences_group in self
        ]

        # Create a grid item for the first group, plus a grid item for every 3 quantiles, until we've used them all
        sequences_html = f"<div class='grid-item'>{html_top}</div>"
        while len(html_quantiles) > 0:
            L = min(3, len(html_quantiles))
            html_next, html_quantiles = html_quantiles[:L], html_quantiles[L:]
            sequences_html += f"<div class='grid-item'>{''.join(html_next)}</div>"

        # If necessary, add the javascript
        return sequences_html + (f"<script>{JS_HOVERTEXT_SCRIPT}</script>" if hovertext else "")




class LeftTablesData:
    '''
    Class to store all the data used in the left-hand tables (i.e. neuron alignment, correlated neurons & features).

    Args:
        neuron_alignment: Tuple[TopK, Arr]
            First element is the topk by value (i.e. first two cols of first table), second is the % L1 norm (third col).
        
        neurons_correlated: Tuple[TopK, Arr]
            TopK correlated neurons by pearson corr coef, and the cosine sim values of those topk.

        b_features_correlated: Tuple[TopK, Arr]
            Same as neurons_correlated, but for the encoder-B features.

    In other words, each of these 3 inputs creates one of the 3 tables: the TopK object gives us indices and values (the
    first two columns of the table), and the Arr object gives us the third column.
    '''
    def __init__(self, neuron_alignment, neurons_correlated, b_features_correlated):
        self.neuron_alignment: Tuple[TopK, Arr] = neuron_alignment
        self.neurons_correlated: Tuple[TopK, Arr] = neurons_correlated
        self.b_features_correlated: Tuple[TopK, Arr] = b_features_correlated

    def get_html(self) -> str:
        '''
        The `get_html` function returns the HTML for the left-hand tables, wrapped in a 'grid-item' div.
        '''
        # Generate the left-hand HTML tables
        html_str = generate_left_tables_html(
            neuron_alignment_indices = self.neuron_alignment[0].indices.tolist(),
            neuron_alignment_values = self.neuron_alignment[0].values.tolist(),
            neuron_alignment_l1 = self.neuron_alignment[1].tolist(),
            correlated_neurons_indices = self.neurons_correlated[0].indices.tolist(),
            correlated_neurons_pearson = self.neurons_correlated[0].values.tolist(),
            correlated_neurons_l1 = self.neurons_correlated[1].tolist(),
            correlated_features_indices = self.b_features_correlated[0].indices.tolist(),
            correlated_features_pearson = self.b_features_correlated[0].values.tolist(),
            correlated_features_l1 = self.b_features_correlated[1].tolist(),
        )
        # Return both items (we'll be wrapping them in 'grid-item' later)
        return f"<div class='grid-item'>{html_str}</div>"




class MiddlePlotsData:
    '''
    Class to store all the data used in the middle plots (i.e. activation density, logits table & histogram).

    Inputs:
        bottom10_logits: TopK
            For the most negative logits, used to generate the logits table.

        top10_logits: TopK
            For the most positive logits, used to generate the logits table.

        logits_histogram_data: HistogramData
            Used to generate the logits histogram.

        freq_histogram_data: HistogramData
            Used to generate the activation density histogram.

        frac_nonzero: float
            Used to generate the title of the activation density histogram.
    '''
    def __init__(
        self,
        bottom10_logits,
        top10_logits,
        logits_histogram_data, 
        freq_histogram_data,
        frac_nonzero,
    ):
        self.bottom10_logits: TopK = bottom10_logits
        self.top10_logits: TopK = top10_logits
        self.logits_histogram_data: HistogramData = logits_histogram_data
        self.freq_histogram_data: HistogramData = freq_histogram_data
        self.frac_nonzero: float = frac_nonzero

    def get_html(
        self,
        vocab_dict: Dict[int, str],
        grid_item: bool = True,
        compact: bool = False,
        histogram_line_idx: Optional[int] = None,
    ) -> str:
        '''
        Args:
            vocab_dict: Dict[int, str]
                Used for converting token indices to string tokens.

            grid_item: bool
                Whether to wrap the HTML in a 'grid-item' div.

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
        max_value = max(np.absolute(self.bottom10_logits.values).max(), np.absolute(self.top10_logits.values).max())
        neg_bg_values = np.absolute(self.bottom10_logits.values) / max_value
        pos_bg_values = np.absolute(self.top10_logits.values) / max_value

        html_str = generate_middle_plots_html(
            neg_str = to_str_tokens(vocab_dict, self.bottom10_logits.indices),
            neg_values = self.bottom10_logits.values.tolist(),
            neg_bg_values = neg_bg_values,
            pos_str = to_str_tokens(vocab_dict, self.top10_logits.indices),
            pos_values = self.top10_logits.values.tolist(),
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
                    f"<div class='grid-item'>{html_str[0]}</div>",
                    f"<div class='grid-item' style='width:380px'>{html_str[1]}</div>",
                ])
            else:
                return f"<div class='grid-item'>{html_str}</div>"
        else:
            return html_str








@dataclass
class PromptData:
    '''
    This class contains all the data necessary to make a single prompt-centric visualization, i.e. each column here:

        https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/sae-demo-1.png

    Args:
        prompt_data: Dict[str, SequenceGroupData]
            This contains all the data which will be used to construct the sequences (at the bottom of the column).

        sequence_data: Dict[str, SequenceGroupData]
            This contains all the data which will be used to construct the sequences (at the bottom of the column).

        middle_plots_data: MiddlePlotsData
            This contains all the data which will be used to construct the middle plots (act density histogram, logits
            tables, and logits histogram).
    '''
    prompt_data: SequenceData
    sequence_data: SequenceGroupData
    middle_plots_data: MiddlePlotsData

    def get_html(
        self,
        title: str,
        vocab_dict: Dict[int, str],
        hovertext: bool = True,
        bold_idx: Optional[int] = None,
        width: Optional[int] = 420,
        histogram_line_idx: Optional[int] = None,
    ) -> str:
        '''
        Gets all the HTML for multiple sequences.
        '''
        grid_item_style = "" if width is None else f"style='width:{width}px;'"
        seq_width = width - 20 if width is not None else None

        return f"""
<div class='grid-item' {grid_item_style}>
    <h3>{title}</h3>
    {self.prompt_data.get_html(vocab_dict, hovertext, bold_idx, overflow_x="break")}
    {self.middle_plots_data.get_html(vocab_dict, grid_item=False, histogram_line_idx=histogram_line_idx)}
    {self.sequence_data.get_html(vocab_dict, group_size=10, width=seq_width, hovertext=hovertext)}
</div>
"""
        

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
            keys: Tuple[str, int]
                The first element is the scoring metric (e.g. "act_size", "act_quantile", "loss_effect"), and the
                second element is the sequence position (0-indexed).
            values: Tuple[TopK, List[str]]
                The first element is the topk object (indices are the feature indices, values are the scores). The
                second element is a string or list of strings, containing prompt formatting.

    '''
    prompt_str_toks: List[str]
    prompt_data_dict: Dict[int, PromptData]
    scores_dict: Dict[Tuple[str, str], Tuple[TopK, List[str]]]

    def __getitem__(self, idx: int) -> PromptData:
        return self.prompt_data_dict[idx]

    def get_html(
        self,
        seq_pos: int,
        str_score: Literal["act_size", "act_quantile", "loss_effect"],
        vocab_dict: Dict[int, str],
        width: Optional[int] = 420,
    ) -> str:
        
        # Check arguments are valid, and if they are 
        assert str_score in ["act_size", "act_quantile", "loss_effect"], f"Invalid str_score: {str_score!r}"
        assert 0 <= seq_pos < len(self.prompt_str_toks)
        if str_score == "act_size":
            str_score_formatted = "Activation Size"
        elif str_score == "act_quantile":
            str_score_formatted = "Activation Quantile"
        elif str_score == "loss_effect":
            str_score_formatted = "Loss Effect"
            assert seq_pos > 0, "Can't look at the loss effect on the first token in the prompt."

        
        scores, formatting = self.scores_dict[(str_score, seq_pos)]
        if isinstance(formatting, str):
            formatting = [formatting] * len(scores)
        k = len(scores)

        grid_items = ""

        # Iterate through each of the features we're creating a box for
        for (feat, score, fmt) in zip(scores.indices.tolist(), scores.values.tolist(), formatting):

            # Add the HTML box for this feature
            grid_items += self[feat].get_html(
                title = f"Feature #<a href='demo'>{feat}</a><br>{str_score_formatted} = {score:{fmt}}<br>",
                vocab_dict = vocab_dict,
                hovertext = False,
                bold_idx = seq_pos,
                width = width,
                histogram_line_idx = seq_pos - 1 if (str_score == "loss_effect") else seq_pos,
            )

        # Change the ids from e.g. `histogram-logits` to `histogram-logits-1`, `histogram-logits-2`, ... so that the JavaScript works
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
        pattern = r'<div class="half-width-container" style="display: none;">(.*?)</div>'
        html_str = re.sub(pattern, "", html_str, flags=re.DOTALL)

        html_str = html_str.replace("Ċ", "&bsol;n")
        html_str = adjust_hovertext(html_str)
        return html_str





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

        fvp: FeatureVizParams
            Contains several other parameters which are important in the visualisation.
    '''
    sequence_data: SequenceMultiGroupData
    left_tables_data: Optional[LeftTablesData]
    middle_plots_data: MiddlePlotsData

    feature_idx: int
    vocab_dict: Dict[int, str]
    fvp: FeatureVizParams

    def get_html(
        self,
        width: Optional[int] = 420,
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
        sequence_html = self.sequence_data.get_html(self.vocab_dict, hovertext=True)

        # Get other HTML (split depending on whether we include the left tables or not)
        if self.fvp.include_left_tables:
            left_tables_html = self.left_tables_data.get_html()
            middle_plots_data = self.middle_plots_data.get_html(self.vocab_dict)
        else:
            left_tables_html = ""
            middle_plots_data = self.middle_plots_data.get_html(self.vocab_dict, compact=True)

        # Get the CSS, and make appropriate edits to it
        css = CSS
        # If no border, then delete it from the CSS
        if not(self.fvp.border):
            css = css.replace("border: 1px solid #e6e6e6;", "").replace("box-shadow: 0 5px 5px rgba(0, 0, 0, 0.25);", "")
        # If width is specified, then replace all the sequence html objects with the correct width
        if width is not None:
            sequence_html = sequence_html.replace("class='grid-item'", f"class='grid-item' style='width:{width}px;'")

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
    


class MultiFeatureData:
    '''
    Class for storing data for multiple different features, as well as data which is used to sort the features (for now
    this just means quantile data on the feature activations and loss effects).
    '''
    feature_data_dict: Dict[int, FeatureData]
    feature_act_quantiles: QuantileCalculator

    # TODO - think about adding loss_quantiles: QuantileCalculator
    # They're potentially slower because we need to compute logits-after-ablation for all tokens rather than just the
    # tokens in our sequence groups (by default we don't compute logits for all tokens)

    def __init__(self, feature_data_dict=None, feature_act_quantiles=None):
        self.feature_data_dict = {} if feature_data_dict is None else feature_data_dict
        self.feature_act_quantiles = QuantileCalculator() if feature_act_quantiles is None else feature_act_quantiles

    def __getitem__(self, idx: int) -> FeatureData:
        return self.feature_data_dict[idx]

    def update(self, other: "MultiFeatureData") -> None:
        '''
        Updates a MultiFeatureData object with the data from another MultiFeatureData object.
        '''
        if other is None: return
        # assert isinstance(other, MultiFeatureData), f"Can only update with another MultiFeatureData object, not {type(other)}"
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
        pearson_topk = TopK(pearson.topk(dim=-1, k=k, largest=largest)) # shape (X, k)
        # Get cossim topk by indexing into cossim with the indices of the pearson topk: cossim[X, pearson_indices[X, k]]
        cossim_values = eindex(cossim.cpu().numpy(), pearson_topk.indices, "X [X k]")
        return pearson_topk, cossim_values




