# %%

import re
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Sequence,
    Type,
    TypeVar,
    overload,
)

import einops
import numpy as np
import torch
from dataclasses_json import dataclass_json
from eindex import eindex
from jaxtyping import Bool, Float, Int
from torch import Tensor
from tqdm import tqdm
from transformer_lens import utils
from transformers import PreTrainedTokenizerBase

T = TypeVar("T")

# from rich.progress import ProgressColumn, Task # MofNCompleteColumn
# from rich.text import Text
# from rich.table import Column


def get_device() -> torch.device:
    """
    Helper function to return the correct device (cuda, mps, or cpu).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


# # Depreciated - we no longer use a global device variable
# device = get_device()

Arr = np.ndarray

MAIN = __name__ == "__main__"


def create_iterator(
    iterator: Iterable[T], verbose: bool, desc: str | None = None
) -> Iterable[T]:
    """
    Returns an iterator, useful for reducing code repetition.
    """
    return tqdm(iterator, desc=desc, leave=False) if verbose else iterator


def k_largest_indices(
    x: Float[Tensor, "rows cols"],
    k: int,
    largest: bool = True,
    buffer: tuple[int, int] | None = (5, -5),
) -> Int[Tensor, "k 2"]:
    """
    Args:
        x:
            2D array of floats (these will be the values of feature activations or losses for each
            token in our batch)
        k:
            Number of indices to return
        largest:
            Whether to return the indices for the largest or smallest values
        buffer:
            Positions to avoid at the start / end of the sequence, i.e. we can include the slice buffer[0]: buffer[1].
            If None, then we use all sequences

    Returns:
        The indices of the top or bottom `k` elements in `x`. In other words, output[i, :] is the (row, column) index of
        the i-th largest/smallest element in `x`.
    """
    if buffer is None:
        buffer = (0, x.size(1))
    x = x[:, buffer[0] : buffer[1]]
    indices = x.flatten().topk(k=k, largest=largest).indices
    rows = indices // x.size(1)
    cols = indices % x.size(1) + buffer[0]
    return torch.stack((rows, cols), dim=1)


def sample_unique_indices(
    large_number: int, small_number: int
) -> Int[Tensor, "small_number"]:
    """
    Samples a small number of unique indices from a large number of indices.

    This is more efficient than using `torch.permutation`, because we don't need to shuffle everything.
    """
    weights = torch.ones(large_number)  # Equal weights for all indices
    sampled_indices = torch.multinomial(weights, small_number, replacement=False)
    return sampled_indices


def random_range_indices(
    x: Float[Tensor, "batch seq"],
    k: int,
    bounds: tuple[float, float],
    buffer: tuple[int, int] | None = (5, -5),
) -> Int[Tensor, "k 2"]:
    """
    Args:
        x:
            2D array of floats (these will be the values of feature activations or losses for each
            token in our batch)
        k:
            Number of indices to return
        bounds:
            The range of values to consider (so we can get quantiles)
        buffer:
            Positions to avoid at the start / end of the sequence, i.e. we can include the slice buffer[0]: buffer[1]

    Returns:
        Same thing as `k_largest_indices`, but the difference is that we're using quantiles rather than top/bottom k.
    """
    if buffer is None:
        buffer = (0, x.size(1))

    # Limit x, because our indices (bolded words) shouldn't be too close to the left/right of sequence
    x = x[:, buffer[0] : buffer[1]]

    # Creat a mask for where x is in range, and get the indices as a tensor of shape (k, 2)
    mask = (bounds[0] <= x) & (x <= bounds[1])
    indices = torch.stack(torch.where(mask), dim=-1)

    # If we have more indices than we need, randomly select k of them
    if len(indices) > k:
        indices = indices[sample_unique_indices(len(indices), k)]

    # Adjust indices to account for the buffer
    return indices + torch.tensor([0, buffer[0]]).to(indices.device)


# TODO - solve the `get_decode_html_safe_fn` issue
# The verion using `tokenizer.decode` is much slower, but Stefan's raised issues about it not working correctly for e.g.
# Cyrillic characters. I think patching the `vocab_dict` in some way is the best solution.

# def get_decode_html_safe_fn(tokenizer, html: bool = False) -> Callable[[int | list[int]], str | list[str]]:
#     '''
#     Creates a tokenization function on single integer token IDs, which is HTML-friendly.
#     '''
#     def decode(token_id: int | list[int]) -> str | list[str]:
#         '''
#         Check this is a single token
#         '''
#         if isinstance(token_id, int):
#             str_tok = tokenizer.decode(token_id)
#             return process_str_tok(str_tok, html=html)
#         else:
#             str_toks = tokenizer.batch_decode(token_id)
#             return [process_str_tok(str_tok, html=html) for str_tok in str_toks]

#     return decode


def get_decode_html_safe_fn(
    tokenizer: PreTrainedTokenizerBase, html: bool = False
) -> Callable[[int | list[int]], str | list[str]]:
    vocab_dict = {v: k for k, v in tokenizer.vocab.items()}  # type: ignore

    def decode(token_id: int | list[int]) -> str | list[str]:
        """
        Check this is a single token
        """
        if isinstance(token_id, int):
            str_tok = vocab_dict.get(token_id, "UNK")
            return process_str_tok(str_tok, html=html)
        else:
            if isinstance(token_id, torch.Tensor):
                token_id = token_id.tolist()
            return [decode(tok) for tok in token_id]  # type: ignore

    return decode


# # Code to test this function:
# from transformer_lens import HookedTransformer
# model = HookedTransformer.from_pretrained("gelu-1l")
# unsafe_token = "<"
# unsafe_token_id = model.tokenizer.encode(unsafe_token, return_tensors="pt")[0].item() # type: ignore
# assert get_decode_html_safe_fn(model.tokenizer)(unsafe_token_id) == "<"
# assert get_decode_html_safe_fn(model.tokenizer, html=True)(unsafe_token_id) == "&lt;"


HTML_CHARS = {
    "\\": "&bsol;",
    "<": "&lt;",
    ">": "&gt;",
    ")": "&#41;",
    "(": "&#40;",
    "[": "&#91;",
    "]": "&#93;",
    "{": "&#123;",
    "}": "&#125;",
}
HTML_ANOMALIES = {
    "âĢĶ": "&mdash;",
    "âĢĵ": "&ndash;",
    "âĢĭ": "&#8203;",
    "âĢľ": "&ldquo;",
    "âĢĿ": "&rdquo;",
    "âĢĺ": "&lsquo;",
    "âĢĻ": "&rsquo;",
    "Ġ": "&nbsp;",
    "Ċ": "&bsol;n",
    "ĉ": "&bsol;t",
}
HTML_ANOMALIES_REVERSED = {
    "&mdash": "—",
    "&ndash": "–",
    # "&#8203": "​", # TODO: this is actually zero width space character. what's the best way to represent it?
    "&ldquo": "“",
    "&rdquo": "”",
    "&lsquo": "‘",
    "&rsquo": "’",
    "&nbsp;": " ",
    "&bsol;": "\\",
}
HTML_QUOTES = {
    "'": "&apos;",
    '"': "&quot;",
}
HTML_ALL = {**HTML_CHARS, **HTML_QUOTES, " ": "&nbsp;"}

HTML_ALL_REVERSED = {
    **{v: k for k, v in HTML_CHARS.items()},
    **HTML_ANOMALIES_REVERSED,
}


def process_str_tok(str_tok: str, html: bool = True) -> str:
    """
    Takes a string token, and does the necessary formatting to produce the right HTML output. There are 2 things that
    might need to be changed:

        (1) Anomalous chars like Ġ should be replaced with their normal Python string representations
            e.g. "Ġ" -> " "
        (2) Special HTML characters like "<" should be replaced with their HTML representations
            e.g. "<" -> "&lt;", or " " -> "&nbsp;"

    We always do (1), the argument `html` determines whether we do (2) as well.
    """
    for k, v in HTML_ANOMALIES.items():
        str_tok = str_tok.replace(k, v)

    if html:
        # Get rid of the quotes and apostrophes, and replace them with their HTML representations
        for k, v in HTML_QUOTES.items():
            str_tok = str_tok.replace(k, v)
        # repr turns \n into \\n, while slicing removes the quotes from the repr
        str_tok = repr(str_tok)[1:-1]

        # Apply the map from special characters to their HTML representations
        for k, v in HTML_CHARS.items():
            str_tok = str_tok.replace(k, v)

    return str_tok


def unprocess_str_tok(str_tok: str) -> str:
    """
    Performs the reverse of the `process_str_tok` function, i.e. maps from HTML representations back to their original
    characters. This is useful when e.g. our string is inside a <code>...</code> element, because then we have to use
    the literal characters.
    """
    for k, v in HTML_ALL_REVERSED.items():
        str_tok = str_tok.replace(k, v)

    return str_tok


@overload
def to_str_tokens(
    decode_fn: Callable[[int | list[int]], str | list[str]],
    tokens: int,
) -> str: ...


@overload
def to_str_tokens(
    decode_fn: Callable[[int | list[int]], str | list[str]],
    tokens: list[int],
) -> list[str]: ...


def to_str_tokens(
    decode_fn: Callable[[int | list[int]], str | list[str]],
    tokens: int | list[int] | torch.Tensor,
) -> str | Any:
    """
    Helper function which converts tokens to their string representations, but (if tokens is a tensor) keeps
    them in the same shape as the original tensor (i.e. nested lists).
    """
    # Deal with the int case separately
    if isinstance(tokens, int):
        return decode_fn(tokens)

    # If the tokens are a (possibly nested) list, turn them into a tensor
    if isinstance(tokens, list):
        tokens = torch.tensor(tokens)

    # Get flattened list of tokens
    str_tokens = [decode_fn(t) for t in tokens.flatten().tolist()]

    # Reshape
    return np.reshape(str_tokens, tokens.shape).tolist()


class TopK:
    """
    This function implements a version of torch.topk over the last dimension. It offers the following:

        (1) Nicer type signatures (the default obj returned by torck.topk isn't well typed)
        (2) Helper functions for indexing & other standard tensor operations like .ndim, .shape, etc.
        (3) An efficient topk calculation, which doesn't bother applying it to the zero elements of a tensor.
    """

    values: Arr
    indices: Arr

    def __init__(
        self,
        tensor: Float[Tensor, "... d"],
        k: int,
        largest: bool = True,
        tensor_mask: Bool[Tensor, "..."] | None = None,
    ):
        self.k = k
        self.largest = largest
        self.values, self.indices = self.topk(tensor, tensor_mask)

    def __getitem__(self, item: int) -> "TopK":
        new_topk = TopK.__new__(TopK)
        new_topk.k = self.k
        new_topk.largest = self.largest
        new_topk.values = self.values[item]
        new_topk.indices = self.indices[item]
        return new_topk

    def __len__(self) -> int:
        return len(self.values)

    @property
    def ndim(self) -> int:
        return self.values.ndim

    @property
    def shape(self) -> tuple[int]:
        return tuple(self.values.shape)  # type: ignore

    def numel(self) -> int:
        return self.values.size

    def topk(
        self,
        tensor: Float[Tensor, "... d"],
        tensor_mask: Bool[Tensor, "..."] | None = None,
    ) -> tuple[Arr, Arr]:
        """
        This is an efficient version of `torch.topk(..., dim=-1)`. It saves time by only doing the topk calculation over
        the bits of `tensor` where `tensor_mask=True`. This is useful when `tensor` is very sparse, e.g. it has shape
        (batch, seq, d_vocab) and its elements are zero if the corresponding token has feature activation zero. In this
        case, we don't want to waste time taking topk over a tensor of zeros.
        """
        # If no tensor mask is provided, then we just return the topk values and indices
        if tensor_mask is None or not tensor_mask.any():
            k = min(self.k, tensor.shape[-1])
            topk = tensor.topk(k=k, largest=self.largest)
            return utils.to_numpy(topk.values), utils.to_numpy(topk.indices)

        # Get the topk of the tensor, but only computed over the values of the tensor which are nontrivial
        assert (
            tensor_mask.shape == tensor.shape[:-1]
        ), "Error: unexpected shape for tensor mask."
        tensor_nontrivial_values = tensor[tensor_mask]  # shape [rows d]
        k = min(self.k, tensor_nontrivial_values.shape[-1])
        k = self.k
        topk = tensor_nontrivial_values.topk(
            k=k, largest=self.largest
        )  # shape [rows k]

        # Get an array of indices and values (with unimportant elements) which we'll index into using the topk object
        topk_shape = (*tensor_mask.shape, k)
        topk_indices = torch.zeros(topk_shape).to(tensor.device).long()  # shape [... k]
        topk_indices[tensor_mask] = topk.indices
        topk_values = torch.zeros(topk_shape).to(tensor.device)  # shape [... k]
        topk_values[tensor_mask] = topk.values

        return utils.to_numpy(topk_values), utils.to_numpy(topk_indices)


def merge_lists(*lists: Iterable[T]) -> list[T]:
    """
    Merges a bunch of lists into a single list.
    """
    return [item for sublist in lists for item in sublist]


def extract_and_remove_scripts(html_content: str) -> tuple[str, str]:
    """
    Extracts JavaScript from script tags in the HTML content, and returns it as a single string,
    along with the original content with the script tags removed.
    """
    # Pattern to find <script>...</script> tags and capture content inside
    pattern = r"<script[^>]*>(.*?)</script>"

    # Find all script tags and extract content
    scripts = re.findall(pattern, html_content, re.DOTALL)

    # Remove script tags from the original content
    html_without_scripts = re.sub(pattern, "", html_content, flags=re.DOTALL)

    # Join extracted JavaScript code
    javascript = "\n".join(scripts)

    return javascript, html_without_scripts


def pad_with_zeros(
    x: list[float],
    n: int,
    side: Literal["left", "right"] = "left",
) -> list[float]:
    """
    Pads a list with zeros to make it the correct length.
    """
    assert len(x) <= n, "Error: x must have fewer than n elements."

    if side == "right":
        return x + [0.0] * (n - len(x))
    else:
        return [0.0] * (n - len(x)) + x


# %%

# This defines the number of decimal places we'll use. It's assumed to refer to values in the range [0, 1] rather than
# pct, e.g. precision of 5 would be 99.497% = 0.99497. In other words, decimal_places = precision - 2.

SYMMETRIC_RANGES_AND_PRECISIONS: list[tuple[list[float], int]] = [
    ([0.0, 0.01], 5),
    ([0.01, 0.05], 4),
    ([0.05, 0.95], 3),
    ([0.95, 0.99], 4),
    ([0.99, 1.0], 5),
]
ASYMMETRIC_RANGES_AND_PRECISIONS: list[tuple[list[float], int]] = [
    ([0.0, 0.95], 3),
    ([0.95, 0.99], 4),
    ([0.99, 1.0], 5),
]


@dataclass_json
@dataclass
class FeatureStatistics:
    """
    This object (which used to be called QuantileCalculator) stores stats about a dataset.

    The quantiles are a bit complicated because we store a higher precision for values closer to 100%. Most of the
    other stats are pretty straightforward.

    We create these objects using the `create` method. We assume data supplid is 2D, where each row is a different
    dataset that we want to compute the stats for.
    """

    # Stats: max, frac_nonzero, skew, kurtosis
    max: list[float] = field(default_factory=list)
    frac_nonzero: list[float] = field(default_factory=list)
    skew: list[float] = field(default_factory=list)
    kurtosis: list[float] = field(default_factory=list)

    # Quantile data
    quantile_data: list[list[float]] = field(default_factory=list)
    quantiles: list[float] = field(default_factory=list)
    ranges_and_precisions: list[tuple[list[float], int]] = field(
        default_factory=lambda: ASYMMETRIC_RANGES_AND_PRECISIONS
    )

    @property
    def aggdata(
        self,
        precision: int = 5,
    ) -> dict[str, list[float]]:
        return {
            "max": [round(x, precision) for x in self.max],
            "frac_nonzero": [round(x, precision) for x in self.frac_nonzero],
            "skew": [round(x, precision) for x in self.skew],
            "kurtosis": [round(x, precision) for x in self.kurtosis],
        }

    @classmethod
    def create(
        cls,
        data: Float[Tensor, "batch data"] | None = None,
        ranges_and_precisions: list[
            tuple[list[float], int]
        ] = ASYMMETRIC_RANGES_AND_PRECISIONS,
    ) -> "FeatureStatistics":
        # Generate quantiles from the ranges_and_precisions list
        quantiles = []
        for r, p in ranges_and_precisions:
            start, end = r
            step = 10**-p
            quantiles.extend(np.arange(start, end - 0.5 * step, step))

        # If data is None, then set the quantiles and quantile_data to None, and return
        skew = []
        kurtosis = []
        if data is None:
            _max = []
            frac_nonzero = []
            quantiles = []
            quantile_data = []
        # Else, get the actual stats & quantile values (which we'll use to calculate the quantiles of any new data)
        else:
            _max = data.max(dim=-1).values.tolist()
            frac_nonzero = (data.abs() > 1e-6).float().mean(dim=-1).tolist()
            quantiles_tensor = torch.tensor(quantiles, dtype=data.dtype).to(data.device)
            quantile_data = torch.quantile(data, quantiles_tensor, dim=-1).T.tolist()

        quantiles = [round(q, 6) for q in quantiles + [1.0]]
        quantile_data = [[round(q, 6) for q in qd] for qd in quantile_data]

        # Now, strip out the quantile data prefixes which are all zeros
        for i, qd in enumerate(quantile_data):
            first_nonzero = next(
                (i for i, x in enumerate(qd) if abs(x) > 1e-6), len(qd)
            )
            quantile_data[i] = qd[first_nonzero:]

        return cls(
            max=_max,
            frac_nonzero=frac_nonzero,
            skew=skew,
            kurtosis=kurtosis,
            quantile_data=quantile_data,
            quantiles=quantiles,
            ranges_and_precisions=ranges_and_precisions,
        )

    def update(self, other: "FeatureStatistics"):
        """
        Merges two FeatureStatistics objects together (changing self inplace). This is useful when we're batching our
        calculations over different groups of features, and we want to merge them together at the end.

        Note, we also deal with the special case where self has no data.
        """
        assert (
            self.ranges_and_precisions == other.ranges_and_precisions
        ), "Error: can't merge two FeatureStatistics objects with different ranges."

        self.max.extend(other.max)
        self.frac_nonzero.extend(other.frac_nonzero)
        self.skew.extend(other.skew)
        self.kurtosis.extend(other.kurtosis)
        self.quantiles.extend(other.quantiles)
        self.quantile_data.extend(other.quantile_data)

    def get_quantile(
        self,
        values: Float[Tensor, "batch *data_dim"],
        batch_indices: list[int] | None = None,
    ) -> tuple[Float[Tensor, "batch *data_dim"], Int[Tensor, "batch *data_dim"]]:
        """
        Args:
            values:
                Tensor of values for which we want to compute the quantiles. If this is 1D then it is interpreted as a
                single value for each dataset (i.e. for each row of the reference data), if it's 2D then it's a row of
                values for each dataset.
            batch_indices:
                If not None, then this should be a list of batch indices we're actually using, in other words we should
                index `self.quantiles` down to only these indices. This is useful because often we're only doing this
                calculation on a small set of features (the ones which are non-zero).

        Returns:
            quantiles:
                The quantiles of `values` within the respective rows of the reference data.
            precisions:
                The precision of the quantiles (i.e. how many decimal places we're accurate to).
        """
        rp = self.ranges_and_precisions
        ranges = torch.tensor([r[0] for (r, _p) in rp] + [1.0]).to(values.device)
        precisions = torch.tensor([rp[0][1]] + [p for (_r, p) in rp] + [rp[-1][1]]).to(
            values.device
        )

        # For efficient storage, we remove the zeros from quantile_data (it may start with zeros). So when converting it
        # back to a tensor, we need to pad it with zeros again.
        n_buckets = len(self.quantiles) - 1
        quantiles = torch.tensor(self.quantiles).to(values.device)
        quantile_data = torch.tensor(
            [pad_with_zeros(x, n_buckets) for x in self.quantile_data]
        ).to(values.device)

        values_is_1d = values.ndim == 1
        if values_is_1d:
            values = values.unsqueeze(1)

        # Get an object to slice into the tensor (along batch dimension)
        my_slice = slice(None) if batch_indices is None else batch_indices

        # Find the quantiles of these values (i.e. the values between 0 and 1)
        quantile_indices = torch.searchsorted(
            quantile_data[my_slice], values
        )  # shape [batch data_dim]
        quantiles = quantiles[quantile_indices]

        # Also get the precisions (which we do using a separate searchsorted, only over the range dividers)
        precision_indices = torch.searchsorted(
            ranges, quantiles
        )  # shape [batch data_dim]
        precisions = precisions[precision_indices]

        # If values was 1D, we want to return the result as 1D also (for convenience)
        if values_is_1d:
            quantiles = quantiles.squeeze(1)
            precisions = precisions.squeeze(1)

        return quantiles, precisions


# Example usage
if MAIN:
    # 2D data: each row represents the activations data of a different feature. We set some of it to zero, so we can
    # test the "JSON doesn't store zeros" feature of the FeatureStatistics class.
    device = get_device()
    N = 100_000
    data = torch.stack(
        [torch.rand(N).masked_fill(torch.rand(N) < 0.5, 0.0), torch.rand(N)]
    ).to(device)
    qc = FeatureStatistics.create(data)
    print(f"Total datapoints stored = {sum(len(x) for x in qc.quantile_data):_}")
    print(f"Total datapoints used to compute quantiles = {data.numel():_}\n")

    # 2D values tensor: each row applies to a different dataset
    values = torch.tensor([[0.0, 0.005, 0.02, 0.25], [0.75, 0.98, 0.995, 1.0]]).to(
        device
    )
    quantiles, precisions = qc.get_quantile(values)

    print("When 50% of data is 0, and 50% is Unif[0, 1]")
    for v, q, p in zip(values[0], quantiles[0], precisions[0]):
        print(f"Value: {v:.3f}, Precision: {p}, Quantile: {q:.{p-2}%}")
    print("\nWhen 100% of data is Unif[0, 1]")
    for v, q, p in zip(values[1], quantiles[1], precisions[1]):
        print(f"Value: {v:.3f}, Precision: {p}, Quantile: {q:.{p-2}%}")


# %%


def split_string(
    input_string: str,
    str1: str,
    str2: str,
) -> tuple[str, str]:
    assert (
        str1 in input_string and str2 in input_string
    ), "Error: str1 and str2 must be in input_string"
    pattern = f"({re.escape(str1)}.*?){re.escape(str2)}"
    match = re.search(pattern, input_string, flags=re.DOTALL)
    if match:
        between_str1_str2 = match.group(1)
        remaining_string = input_string.replace(between_str1_str2, "")
        return between_str1_str2, remaining_string
    else:
        return "", input_string


# Example usage
if MAIN:
    input_string = "The quick brown fox jumps over the lazy dog"
    str1 = "quick"
    str2 = "jumps"
    print(split_string(input_string, str1, str2))

    input_string = (
        "Before table <!-- Logits table --> Table <!-- Logits histogram --> After table"
    )
    str1 = r"<!-- Logits table -->"
    str2 = r"<!-- Logits histogram -->"
    print(split_string(input_string, str1, str2))


# %%


def apply_indent(
    text: str,
    prefix: str,
    first_line_indented: bool = True,
) -> str:
    """
    Indents a string at every new line (e.g. by spaces or tabs). This is useful for formatting when we're dumping things
    into an HTML file.

    Args:
        text:
            The text to indent
        prefix:
            The string to add at the start of each line
        first_line_indented:
            Whether the first line should be indented. If False, then the first line will be left as it is.
    """
    text_indented = "\n".join(prefix + line for line in text.strip().split("\n"))
    if not first_line_indented:
        text_indented = text_indented[len(prefix) :]

    return text_indented


# %%


def deep_union(
    dict1: dict[Any, Any], dict2: dict[Any, Any], path: str = ""
) -> dict[Any, Any]:
    """
    Returns a deep union of dictionaries (recursive operation). In other words, if `dict1` and `dict2` have the same
    keys then the value of that key will be the deep union of the values.

    Also, base case where one of the values is a list: we concatenate the lists together

    Examples:
        # Normal union
        deep_union({1: 2}, {3: 4}) == {1: 2, 3: 4}

        # 1-deep union
        deep_union(
            {1: {2: [3, 4]}},
            {1: {3: [3, 4]}}
        ) == {1: {2: [3, 4], 3: [3, 4]}}

        # 2-deep union
        assert deep_union(
            {"x": {"y": {"z": 1}}},
            {"x": {"y": {"w": 2}}},
        ) == {"x": {"y": {"z": 1, "w": 2}}}

        # list concatenation
        assert deep_union(
            {"x": [1, 2]},
            {"x": [3, 4]},
        ) == {"x": [1, 2, 3, 4]}

    The `path` accumulates the key/value paths from the recursive calls, so that we can see the full dictionary path
    which caused problems (not just the end-nodes).
    """
    result = dict1.copy()

    # For each new key & value in dict2
    for key2, value2 in dict2.items():
        # If key not in result, then we have a simple case: just add it to the result
        if key2 not in result:
            result[key2] = value2

        # If key in result, both should values be either dicts (then we recursively merge) or lists (then we concat). If
        # not, then we throw an error unconditionally (even if values are the same).
        else:
            value1 = result[key2]

            # Both dicts
            if isinstance(value1, dict) and isinstance(value2, dict):
                result[key2] = deep_union(value1, value2, path=f"{path}[{key2!r}]")

            # Both lists
            elif isinstance(value1, list) and isinstance(value2, list):
                result[key2] = value1 + value2

            # Error
            else:
                path1 = f"dict{path}[{key2!r}] = {value1!r}"
                path2 = f"dict{path}[{key2!r}] = {value2!r}"
                raise ValueError(f"Merge failed. Conflicting paths:\n{path1}\n{path2}")

    return result


if MAIN:
    # Normal union
    assert deep_union({1: 2}, {3: 4}) == {1: 2, 3: 4}

    # 1-deep union
    assert deep_union({1: {2: [3, 4]}}, {1: {3: [3, 4]}}) == {1: {2: [3, 4], 3: [3, 4]}}

    # 2-deep union
    assert deep_union(
        {"x": {"y": {"z": 1}}},
        {"x": {"y": {"w": 2}}},
    ) == {"x": {"y": {"z": 1, "w": 2}}}

    # list concatenation
    assert deep_union(
        {"x": [1, 2]},
        {"x": [3, 4]},
    ) == {"x": [1, 2, 3, 4]}

# %%


# class RollingStats:
#     '''
#     This class helps us compute rolling stats of a dataset as we feed in activations, without ever having to store the
#     entire batch in data.
#     '''
#     def __init__(self):
#         self.n = 0
#         self.x_sum = 0.0
#         self.x2_sum = 0.0
#         self.x3_sum = 0.0
#         self.x4_sum = 0.0
#         self.frac_nonzero = 0.0
#         self.max = 0.0

#     def update(self, x: Tensor):
#         x_frac_nonzero = x.nonzero().size(0) / x.numel()
#         x_n = x.numel()
#         self.frac_nonzero = (self.n * self.frac_nonzero + x_n * x_frac_nonzero) / (self.n + x_n)
#         self.n += x.numel()
#         self.x_sum += x.sum().item()
#         self.x2_sum += x.pow(2).sum().item()
#         self.x3_sum += x.pow(3).sum().item()
#         self.x4_sum += x.pow(4).sum().item()
#         self.max = max(self.max, x.max().item())

#     @property
#     def skew(self) -> float:
#         raise NotImplementedError

#     @property
#     def kurtosis(self) -> float:
#         raise NotImplementedError


class RollingCorrCoef:
    """
    This class helps compute corrcoef (Pearson & cosine sim) between 2 batches of vectors, without having to store the
    entire batch in memory.

    How exactly does it work? We exploit the formula below (x, y assumed to be vectors here), which writes corrcoefs in
    terms of scalars which can be computed on a rolling basis.

        cos_sim(x, y) = xy_sum / ((x2_sum ** 0.5) * (y2_sum ** 0.5))

        pearson_corrcoef(x, y) = num / denom
            num = n * xy_sum - x_sum * y_sum
            denom = (n * x2_sum - x_sum ** 2) ** 0.5 * (n * y2_sum - y_sum ** 2) ** 0.5

    This class batches this computation, i.e. x.shape = (X, N), y.shape = (Y, N), where (for example) we have:
        N = batch_size * seq_len, i.e. it's the number of datapoints we have
        x = features of our original encoder
        y = features of our encoder-B, or neurons of our original model (the thing we're topk-ing over)

    So we can e.g. compute the correlation coefficients for every combination of feature in encoder and model neurons,
    then take topk to find the most correlated neurons for each feature.
    """

    def __init__(
        self,
        indices: list[int] | None = None,
        with_self: bool = False,
    ) -> None:
        """
        Args:
            indices: list[int]
                If supplied, we map y indices (from 0 to y.shape) to these values. Useful when we're working with e.g.
                a dataset which didn't start from 0, and we want the "true indices".
            with_self: bool
                If True, then we take X and Y as coming from the same dataset. This saves us some computation, and it
                also means we exclude the diagonal from final topk (since correlation with self is always 1).
        """
        self.n = 0
        self.X = None
        self.Y = None
        self.indices = indices
        self.with_self = with_self

    def update(self, x: Float[Tensor, "X N"], y: Float[Tensor, "Y N"]) -> None:
        # Get values of x and y, and check for consistency with each other & with previous values
        assert x.ndim == 2 and y.ndim == 2, "Both x and y should be 2D"
        X, Nx = x.shape
        Y, Ny = y.shape
        assert (
            Nx == Ny
        ), "Error: x and y should have the same size in the last dimension"
        if self.with_self:
            assert X == Y, "If with_self is True, then x and y should be the same shape"
        if self.X is not None:
            assert (
                X == self.X
            ), "Error: updating a corrcoef object with different sized dataset."
        if self.Y is not None:
            assert (
                Y == self.Y
            ), "Error: updating a corrcoef object with different sized dataset."
        self.X = X
        self.Y = Y

        # If this is the first update step, then we need to initialise the sums
        if self.n == 0:
            self.x_sum = torch.zeros(X, device=x.device)
            self.xy_sum = torch.zeros(X, Y, device=x.device)
            self.x2_sum = torch.zeros(X, device=x.device)
            if not self.with_self:
                self.y_sum = torch.zeros(Y, device=y.device)
                self.y2_sum = torch.zeros(Y, device=y.device)

        # Next, update the sums
        self.n += x.shape[-1]
        self.x_sum += einops.reduce(x, "X N -> X", "sum")
        self.xy_sum += einops.einsum(x, y, "X N, Y N -> X Y")
        self.x2_sum += einops.reduce(x**2, "X N -> X", "sum")
        if not self.with_self:
            self.y_sum += einops.reduce(y, "Y N -> Y", "sum")
            self.y2_sum += einops.reduce(y**2, "Y N -> Y", "sum")

    def corrcoef(
        self,
    ) -> tuple[Float[Tensor, "X Y"], Float[Tensor, "X Y"]]:
        """
        Computes the correlation coefficients between x and y, using the formulae given in the class docstring.
        """
        # Get y_sum and y2_sum (to deal with the cases when with_self is True/False)
        if self.with_self:
            self.y_sum = self.x_sum
            self.y2_sum = self.x2_sum

        # Compute cosine sim
        cossim_numer = self.xy_sum
        cossim_denom = torch.sqrt(torch.outer(self.x2_sum, self.y2_sum)) + 1e-6
        cossim = cossim_numer / cossim_denom

        # Compute pearson corrcoef
        pearson_numer = self.n * self.xy_sum - torch.outer(self.x_sum, self.y_sum)
        pearson_denom = (
            torch.sqrt(
                torch.outer(
                    self.n * self.x2_sum - self.x_sum**2,
                    self.n * self.y2_sum - self.y_sum**2,
                )
            )
            + 1e-6
        )
        pearson = pearson_numer / pearson_denom

        # If with_self, we exclude the diagonal
        if self.with_self:
            d = cossim.shape[0]
            cossim[range(d), range(d)] = 0.0
            pearson[range(d), range(d)] = 0.0

        return pearson, cossim

    def topk_pearson(
        self,
        k: int,
        largest: bool = True,
    ) -> tuple[list[list[int]], list[list[float]], list[list[float]]]:
        """
        Takes topk of the pearson corrcoefs over the y-dimension (e.g. giving us the most correlated neurons or most
        correlated encoder-B features for each encoder feature).

        Args:
            k: int
                Number of top indices to take (usually 3, for the left-hand tables)
            largest: bool
                If True, then we take the largest k indices. If False, then we take the smallest k indices.

        Returns:
            pearson_indices: list[list[int]]
                y-indices which are most correlated with each x-index (in terms of pearson corrcoef)
            pearson_values: list[list[float]]
                Values of pearson corrcoef for each of the topk indices
            cossim_values: list[list[float]]
                Values of cosine similarity for each of the topk indices
        """
        # Get correlation coefficient, using the formula from corrcoef method
        pearson, cossim = self.corrcoef()

        # Get the top pearson values
        pearson_topk = TopK(tensor=pearson, k=k, largest=largest)  # shape (X, k)

        # Get the cossim values for the top pearson values, i.e. cossim_values[X, k] = cossim[X, pearson_indices[X, k]]
        cossim_values = eindex(cossim, pearson_topk.indices, "X [X k]")

        # If we've supplied indices, use them to offset the returned pearson topk indices
        indices = pearson_topk.indices.tolist()
        if self.indices is not None:
            indices = [[self.indices[i] for i in x] for x in indices]

        return indices, pearson_topk.values.tolist(), cossim_values.tolist()


@dataclass_json
@dataclass
class HistogramData:
    """
    This class contains all the data necessary to construct a single histogram (e.g. the logits or feat acts histogram).
    See diagram in readme:

        https://github.com/callummcdougall/sae_vis#data_storing_fnspy

    We don't need to store the entire `data` tensor, so we initialize instances of this class using the `from_data`
    method, which computes statistics from the input data tensor then discards it.

        bar_heights: The height of each bar in the histogram
        bar_values: The value of each bar in the histogram
        tick_vals: The tick values we want to use for the histogram
    """

    bar_heights: list[float] = field(default_factory=list)
    bar_values: list[float] = field(default_factory=list)
    tick_vals: list[float] = field(default_factory=list)
    title: str | None = None

    @classmethod
    def from_data(
        cls: Type[T],
        data: Tensor,
        n_bins: int,
        tickmode: Literal["ints", "5 ticks"],
        title: str | None,
    ) -> T:
        """
        Args:
            data: 1D tensor of data which will be turned into histogram
            n_bins: Number of bins in the histogram
            line_posn: list of possible positions of vertical lines we want to put on the histogram

        Returns a HistogramData object, with data computed from the inputs. This is to support the goal of only storing
        the minimum necessary data (and making it serializable, for JSON saving).
        """
        # There might be no data, if the feature never activates
        if data.numel() == 0:
            return cls()

        # Get min and max of data
        max_value = data.max().item()
        min_value = data.min().item()

        # Divide range up into 40 bins
        bin_size = (max_value - min_value) / n_bins
        bin_edges = torch.linspace(min_value, max_value, n_bins + 1)
        # Calculate the heights of each bin
        bar_heights = torch.histc(data, bins=n_bins).int().tolist()
        bar_values = [round(x, 5) for x in (bin_edges[:-1] + bin_size / 2).tolist()]

        # Choose tickvalues
        # TODO - improve this, it's currently a bit hacky (currently I only use the 5 ticks mode)
        assert tickmode in ["ints", "5 ticks"]
        if tickmode == "ints":
            top_tickval = int(max_value)
            tick_vals = torch.arange(0, top_tickval + 1, 1).tolist()
        elif tickmode == "5 ticks":
            # Ticks chosen in multiples of 0.1, set to ensure the longer side of {positive, negative} is 3 ticks long
            if max_value > -min_value:
                tickrange = 0.1 * int(1e-4 + max_value / (3 * 0.1)) + 1e-6
                num_positive_ticks = 3
                num_negative_ticks = int(-min_value / tickrange)
            else:
                tickrange = 0.1 * int(1e-4 + -min_value / (3 * 0.1)) + 1e-6
                num_negative_ticks = 3
                num_positive_ticks = int(max_value / tickrange)
            # Tick values = merged list of negative ticks, zero, positive ticks
            tick_vals = merge_lists(
                reversed([-tickrange * i for i in range(1, 1 + num_negative_ticks)]),
                [0],  # zero (always is a tick)
                [tickrange * i for i in range(1, 1 + num_positive_ticks)],
            )
            tick_vals = [round(t, 1) for t in tick_vals]

        return cls(  # type: ignore
            bar_heights=bar_heights,
            bar_values=bar_values,
            tick_vals=tick_vals,
            title=title,
        )


# %%


def max_or_1(mylist: Sequence[float | int], abs: bool = False) -> float | int:
    """
    Returns max of a list, or 1 if the list is empty.

    Args:
        mylist: list of numbers
        abs: If True, then we take the max of the absolute values of the list
    """
    if len(mylist) == 0:
        return 1

    if abs:
        return max(max(x, -x) for x in mylist)
    else:
        return max(mylist)
