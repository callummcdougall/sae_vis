from matplotlib import colors
import copy
import numpy as np
from typing import List, Optional, Tuple, Literal
from pathlib import Path
import re

from sae_vis.utils_fns import to_str_tokens, split_string

'''
Key feature of these functions: the arguments should be descriptive of their role in the actual HTML
visualisation. If the arguments are super arcane features of the model data, this is bad!
'''

ROOT_DIR = Path(__file__).parent
CSS_DIR = ROOT_DIR / "css"
HTML_DIR = ROOT_DIR / "html"
JS_DIR = ROOT_DIR / "js"

CSS = "\n".join([
    (CSS_DIR / "general.css").read_text(),
    (CSS_DIR / "sequences.css").read_text(),
    (CSS_DIR / "tables.css").read_text(),
])

HTML_TOKEN = (HTML_DIR / "token_template.html").read_text()
HTML_LEFT_TABLES = (HTML_DIR / "left_tables_template.html").read_text()
HTML_MIDDLE_PLOTS = (HTML_DIR / "middle_plots.html").read_text()

JS_HOVERTEXT_SCRIPT = (JS_DIR / "hovertext.js").read_text()

HISTOGRAM_LINE = """
    shapes: [{
        type: 'line',
        x0: X_VALUE,
        y0: 0,
        x1: X_VALUE,
        y1: LINE_Y,
        xref: 'x',
        yref: 'y',
        line: {
            color: 'black',
            width: 2,
        },
    }],
    annotations: [{
        x: X_VALUE,
        y: 0.9,
        xref: 'x',
        yref: 'paper',
        text: 'X_STR',
        showarrow: false,
        xshift: 28,
        align: 'left',
    }],
"""


BG_COLOR_MAP = colors.LinearSegmentedColormap.from_list("bg_color_map", ["white", "darkorange"])


def generate_tok_html(
    vocab_dict: dict,
    
    this_token: str,
    underline_color: str,
    bg_color: str,
    is_bold: bool = False,

    feat_act: float = 0.0,
    contribution_to_loss: float = 0.0,
    pos_ids: Optional[List[int]] = None,
    pos_val: Optional[List[float]] = None,
    neg_ids: Optional[List[int]] = None,
    neg_val: Optional[List[float]] = None,
):
    '''
    Creates a single sequence visualisation, by reading from the `token_template.html` file.

    Currently, a bunch of things are randomly chosen rather than actually calculated (we're going for
    proof of concept here).
    '''
    html_str = (
        HTML_TOKEN
        .replace("this_token", to_str_tokens(vocab_dict, this_token))
        .replace("feat_activation", f"{feat_act:+.3f}")
        .replace("feature_ablation", f"{contribution_to_loss:+.3f}")
        .replace("font_weight", "bold" if is_bold else "normal")
        .replace("bg_color", bg_color)
        .replace("underline_color", underline_color)
    )

    # If pos_ids is None, this means we don't want to show the positive/negative contributions, so our job is easier!
    if pos_ids is None:
        # Remove the pos/neg contributions table
        html_str = re.sub(r'<br>\n            <div class="half-width-container">.*?</div>', '', html_str, flags=re.DOTALL)
        html_str = html_str.replace("<br>", "")
        html_str = html_str.replace(
            '<div class="tooltip" style="height:275px; width:325px; align-items: center; text-align: center;">',
            '<div class="tooltip" style="height:100px; width:250px; align-items: center; text-align: center;">'
        )
        return html_str

    # Figure out if the activations were zero on previous token, i.e. no predictions were affected
    is_empty = len(pos_ids) + len(neg_ids) == 0
    
    # Get the string tokens
    pos_str = [to_str_tokens(vocab_dict, i) for i in pos_ids]
    neg_str = [to_str_tokens(vocab_dict, i) for i in neg_ids]

    # Pad out the pos_str etc lists, because they might be short
    pos_str.extend([""] * 5)
    neg_str.extend([""] * 5)
    pos_val.extend([0.0] * 5)
    neg_val.extend([0.0] * 5)
    
    # Make all the substitutions
    html_str = re.sub("pos_str_(\d)", lambda m: pos_str[int(m.group(1))].replace(" ", "&nbsp;"), html_str)
    html_str = re.sub("neg_str_(\d)", lambda m: neg_str[int(m.group(1))].replace(" ", "&nbsp;"), html_str)
    html_str = re.sub("pos_val_(\d)", lambda m: f"{pos_val[int(m.group(1))]:+.3f}", html_str)
    html_str = re.sub("neg_val_(\d)", lambda m: f"{neg_val[int(m.group(1))]:+.3f}", html_str)

    # If the effect on loss is nothing (because feature isn't active), replace the HTML output with smth saying this
    if is_empty:
        html_str = (
            html_str
            .replace('<div class="half-width-container">', '<div class="half-width-container" style="display: none;">') # TODO - regex remove, not have display None?
            .replace('<!-- No effect! -->', '<div style="font-size:0.8em;">Feature not active on prev token;<br>no predictions were affected.</div>')
            .replace(
                '<div class="tooltip" style="height:275px; width:325px; align-items: center; text-align: center;">',
                '<div class="tooltip" style="height:175px; width:250px; align-items: center; text-align: center;">'
            )
        )
    # Also, delete the columns as appropriate if the number is between 0 and 5
    else:
        html_str = html_str.replace('<tr><td class="right-aligned"><code></code></td><td class="left-aligned">+0.000</td></tr>', "")

    return html_str



def generate_seq_html(
    vocab_dict: dict,
    token_ids: List[str],
    feat_acts: List[float],
    contribution_to_loss: List[float],
    pos_ids: Optional[List[List[int]]] = None,
    neg_ids: Optional[List[List[int]]] = None,
    pos_val: Optional[List[List[float]]] = None,
    neg_val: Optional[List[List[float]]] = None,
    bold_idx: Optional[int] = None,
    overflow_x: Literal["break", None] = "break",
    max_act_color: Optional[float] = None,
) -> str:
    assert len(token_ids) == len(feat_acts) == len(contribution_to_loss), f"All input lists must be of the same length, not {len(token_ids)}, {len(feat_acts)}, {len(contribution_to_loss)}"

    # If max_act_color is None, we set it to be the max of feature_acts
    bg_denom = max_act_color if max_act_color is not None else float(np.max(feat_acts))
    bg_values = np.maximum(feat_acts, 0.0) / bg_denom
    underline_values = np.clip(contribution_to_loss, -1, 1).tolist()

    classname = "seq" if (overflow_x is None) else "seq-break"
    html_output = f'<div class="{classname}">'

    # Sometimes, pos_ids etc might be 1 shorter than token_ids, so we pad them at the start
    pos_val = copy.deepcopy(pos_val)
    neg_val = copy.deepcopy(neg_val)
    if (pos_ids is not None) and (len(pos_ids) == len(token_ids) - 1):
        pos_ids = [None] + pos_ids
        pos_val = [None] + pos_val
        neg_ids = [None] + neg_ids
        neg_val = [None] + neg_val

    for i in range(len(token_ids)):

        # Get background color, which is {0: transparent, +1: darkorange}
        bg_val = bg_values[i]
        bg_color = colors.rgb2hex(BG_COLOR_MAP(bg_val))

        # Get underline color, which is {-1: blue, 0: transparent, +1: red}
        underline_val = underline_values[i]
        if underline_val < 0:
            v = int(255 * (underline_val + 1))
            underline_color = f"rgb({v}, {v}, 255)"
        else:
            v = int(255 * (1 - underline_val))
            underline_color = f"rgb(255, {v}, {v})"

        html_output += generate_tok_html(
            vocab_dict = vocab_dict,
            this_token = token_ids[i],
            underline_color = underline_color,
            bg_color = bg_color,
            pos_ids = None if pos_ids is None else pos_ids[i],
            neg_ids = None if neg_ids is None else neg_ids[i],
            pos_val = None if pos_val is None else pos_val[i],
            neg_val = None if neg_val is None else neg_val[i],
            is_bold = (bold_idx is not None) and (bold_idx == i),
            feat_act = feat_acts[i],
            contribution_to_loss = contribution_to_loss[i],
        )

    html_output += '</div>'

    return html_output




def generate_left_tables_html(
    neuron_alignment_indices: List[int],
    neuron_alignment_values: List[float],
    neuron_alignment_l1: List[float],
    correlated_neurons_indices: List[int],
    correlated_neurons_pearson: List[float],
    correlated_neurons_l1: List[float],
    correlated_features_indices: Optional[List[int]] = None,
    correlated_features_pearson: Optional[List[float]] = None,
    correlated_features_l1: Optional[List[float]] = None,
):
    html_output = HTML_LEFT_TABLES

    # If we don't have the correlated features from encoder_B, remove that table
    if correlated_features_indices is None:
        html_output = re.sub(r'<h4>CORRELATED FEATURES \(B-ENCODER\)</h4>.*?</table>', "", html_output, flags=re.DOTALL)

    for (letter, mylist, myformat) in zip(
        "IVLIPCIPC",
        [
            neuron_alignment_indices,
            neuron_alignment_values,
            neuron_alignment_l1,
            correlated_neurons_indices,
            correlated_neurons_pearson,
            correlated_neurons_l1,
            correlated_features_indices,
            correlated_features_pearson,
            correlated_features_l1,
        ],
        [None, "+.2f", ".1%", None, "+.2f", "+.2f", None, "+.2f", "+.2f"]
    ):
        if mylist is None: 
            continue
        fn = lambda m: str(mylist[int(m.group(1))]) if myformat is None else format(mylist[int(m.group(1))], myformat)
        html_output = re.sub(letter + "(\d)", fn, html_output, count=3)
    
    return html_output
    


# def format_list(mylist: List[float], fmt: str) -> str:
#     '''
#     Formats a list of floats as a string, with a given format.
#     '''
#     return ", ".join([format(x, fmt) for x in mylist])



def generate_middle_plots_html(
    neg_str: List[str],
    neg_values: List[float],
    neg_bg_values: List[float],
    pos_str: List[str],
    pos_values: List[float],
    pos_bg_values: List[float],
    freq_hist_data_bar_heights: List[float],
    freq_hist_data_bar_values: List[float],
    freq_hist_data_tick_vals: List[float],
    logits_hist_data_bar_heights: List[float],
    logits_hist_data_bar_values: List[float],
    logits_hist_data_tick_vals: List[float],
    frac_nonzero: float,
    freq_line_posn: Optional[float] = None,
    logits_line_posn: Optional[float] = None,
    logits_line_text: Optional[str] = None,
    freq_line_text: Optional[str] = None,
    compact: bool = False,
) -> Tuple[str, str]:
    '''This generates all the middle data at once, because it comes from a single file.'''

    html_str = HTML_MIDDLE_PLOTS

    # ! Populate the HTML with the logit tables

    # Define the background colors (starts of dark red / blue, gets lighter)
    neg_bg_colors = [f"rgba(255, {int(255 * (1 - v))}, {int(255 * (1 - v))}, 0.5)" for v in neg_bg_values]
    pos_bg_colors = [f"rgba({int(255 * (1 - v))}, {int(255 * (1 - v))}, 255, 0.5)" for v in pos_bg_values]

    # Sub in all the values, tokens, and background colors
    for (letter, mylist) in zip("SVCSVC", [neg_str, neg_values, neg_bg_colors, pos_str, pos_values, pos_bg_colors]):
        if letter == "S":
            fn = lambda m: str(mylist[int(m.group(1))]).replace(" ", "&nbsp;")
        elif letter == "V":
            fn = lambda m: format(mylist[int(m.group(1))], "+.2f")
        elif letter == "C":
            fn = lambda m: str(mylist[int(m.group(1))])
        html_str = re.sub(letter + "(\d)", fn, html_str, count=10)

    # ! Populate the HTML with the activations density histogram data & the logits histogram data

    # Start off high, cause we want closer to orange than white for the left-most bars
    freq_bar_values = freq_hist_data_bar_values
    freq_bar_values_clipped = [(0.4 * max(freq_bar_values) + 0.6 * v) / max(freq_bar_values) for v in freq_bar_values]
    freq_bar_colors = [colors.rgb2hex(BG_COLOR_MAP(v)) for v in freq_bar_values_clipped]

    html_str = (
        html_str
        # Fill in the freq histogram data
        .replace("BAR_HEIGHTS_FREQ", str(list(freq_hist_data_bar_heights)))
        .replace("BAR_VALUES_FREQ", str(list(freq_hist_data_bar_values)))
        .replace("BAR_COLORS_FREQ", str(list(freq_bar_colors)))
        .replace("TICK_VALS_FREQ", str(list(freq_hist_data_tick_vals)))
        # Fill in the logits histogram data
        .replace("BAR_HEIGHTS_LOGITS", str(list(logits_hist_data_bar_heights)))
        .replace("BAR_VALUES_LOGITS", str(list(logits_hist_data_bar_values)))
        .replace("TICK_VALS_LOGITS", str(list(logits_hist_data_tick_vals)))
        # Other things
        .replace("FRAC_NONZERO", f"{frac_nonzero:.3%}")
    )

    # If line_posn is supplied, then we add a vertical line to the freq histogram / logits histogram, with label
    if freq_line_posn is not None:
        line = (
            HISTOGRAM_LINE
            .replace("X_VALUE", f"{freq_line_posn:.3f}")
            .replace("X_STR", freq_line_text)
            .replace("LINE_Y", str(max(freq_hist_data_bar_heights)))
        )
        html_str = html_str.replace(
            "var layoutFreq = {",
            "var layoutFreq = {" + line,
        )
    if logits_line_posn is not None:
        line = (
            HISTOGRAM_LINE
            .replace("X_VALUE", f"{logits_line_posn:.3f}")
            .replace("X_STR", logits_line_text)
            .replace("LINE_Y", str(max(logits_hist_data_bar_heights)))
        )
        html_str = html_str.replace(
            "var layoutLogits = {",
            "var layoutLogits = {" + line,
        )

    # If compact, we extract the table, and return (table, bar charts) rather than both together
    if compact:
        html_str = split_string(html_str, str1=r"<!-- Logits table -->", str2=r"<!-- Logits histogram -->")
        # print([len(x) for x in html_str])
    
    return html_str
    



def adjust_hovertext(html_str):
    '''
    Annoying HTML thing: I need to make the tooltip for tokens appear outside the bounding box
    for the sequences, rather than inside (because otherwise it gets cut off, that is it gets
    cut off if I want to enable x-scrolling on my sequences, which I do!).

    This fixes that by using regex operations to do the following:

        (1) Giving IDs to the class="tooltip" and class="token" objects (increasing, i.e. tooltip-1 ...).
        (2) Extracting the tooltip objects from the sequences, and placing them at the end of the HTML.
    
    Note that I've already added CSS (in sequences.css) and JavaScript (in hovertext.html) code
    which works on the assumption that this function has been run, i.e. it won't look good if this fn
    doesn't get run.
    '''
    def replace_fn(match: re.Match) -> str:
        replace_fn.counter += 1
        return f"class=\"tooltip\" id=\"tooltip-{replace_fn.counter:04}\""
    replace_fn.counter = 0
    html_str = re.sub(r'class="tooltip"', replace_fn, html_str)
    def replace_fn(match: re.Match) -> str:
        replace_fn.counter += 1
        return f"class=\"hover-text\" data-tooltip-id=\"tooltip-{replace_fn.counter:04}\""
    replace_fn.counter = 0
    html_str = re.sub(r'class="hover-text"', replace_fn, html_str)

    # Move the tooltip objects to the end of the HTML
    def extract_tooltip(match: re.Match):
        extract_tooltip.s += "\n" + match.group(1)
        return ''
    extract_tooltip.s = ""
    tooltip_pattern = r'(<div class="tooltip" id="tooltip-\d{4}".*?</div>)<!-- tooltip end -->'
    html_str = re.sub(tooltip_pattern, extract_tooltip, html_str, flags=re.DOTALL)
    html_str += f'<div class="tooltip-container">{extract_tooltip.s}</div>'

    return html_str




def grid_item(
    html_contents: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> str:
    '''
    Wraps the HTML contents in a 'grid-item' element.

    If width or height are given, the grid item will be rendered at a fixed size (with scrollbars).
    '''
    width_str: str = f"width: {width}px;" if width is not None else ""
    height_str: str = f"height: {height}px;" if height is not None else ""
    if height is not None:
        print(height_str)
    
    return f'<div class="grid-item" style="{width_str} {height_str}">{html_contents}</div>'

