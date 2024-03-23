from dataclasses import dataclass, asdict
from dataclasses_json import dataclass_json
from typing import Optional, Iterable, Union, Literal
from rich import print as rprint
from rich.tree import Tree
from rich.table import Table

'''
This file contains dataclasses which are used to configure the SAE visualisation.

The main class is `SaeVisConfig`, which contains all the parameters for the visualisation. There are 2 kinds of params
that are stored here:

    - Data params, e.g. batch size and minibatch size
    - Layout params

There are 2 layout params: both of class `SaeVisLayoutConfig`, one for the feature-centric vis and one for the
prompt-centric vis. The former is used when generating data for the first time in `SaeVisData.create(...)`, and each
are used when calling the methods `SaeVisData.save_feature_centric_vis` and `SaeVisData.save_prompt_centric_vis`
respectively.

You can look at the objects DEFAULT_LAYOUT_FEATURE_VIS, DEFAULT_LAYOUT_PROMPT_VIS to see what layout objects lead to
the default appearance of the feature-centric and prompt-centric views. Also, these layout objects have a `help` method
which prints out a tree showing what your vis layout will look like, and what each argument does (that's the purpose of
the _HELP dictionaries defined immediately below this text).
'''


SEQUENCES_CONFIG_HELP = dict(
    buffer = "How many tokens to add as context to each sequence. The tokens chosen for the top acts / quantile groups \
can't be outside the buffer range.",
    compute_buffer = "If False, then we don't compute the loss effect, activations, or any other data for tokens \
other than the bold tokens in our sequences (saving time).",
    n_quantiles = "Number of quantile groups for the sequences. If zero, we only show top activations, no quantile \
groups.",
    top_acts_group_size = "Number of sequences in the 'top activating sequences' group.",
    quantile_group_size = "Number of sequences in each of the sequence quantile groups.",
    top_logits_hoverdata = "Number of top/bottom logits to show in the hoverdata for each token.",
    stack_mode = "How to stack the sequence groups.\n  'stack-all' = all groups are stacked in a single column \
(scrolls vertically if it overflows)\n  'stack-quantiles' = first col contains top acts, second col contains all \
quantile groups\n  'stack-none' = we stack in a way which ensures no vertical scrolling.",
    hover_below = "Whether the hover information about a token appears below or above the token.",
)

ACTIVATIONS_HISTOGRAM_CONFIG_HELP = dict(
    n_bins = "Number of bins for the histogram.",
)

LOGITS_HISTOGRAM_CONFIG_HELP = dict(
    n_bins = "Number of bins for the histogram.",
)

LOGITS_TABLE_CONFIG_HELP = dict(
    n_rows = "Number of top/bottom logits to show in the table.",
)

FEATURE_TABLES_CONFIG_HELP = dict(
    n_rows = "Number of rows to show for each feature table."
)

@dataclass
class BaseConfig:

    def data_is_contained_in(self, other) -> bool:
        '''
        This returns False only when the data that was computed based on `other` wouldn't be enough to show the data
        that was computed based on `self`. For instance, if `self` was a config object with 10 rows, and `other` had
        just 5 rows, then this would return False. A less obvious example: if `self` was a histogram config with 50 bins
        then `other` would need to have exactly 50 bins (because we can't change the bins after generating them).
        '''
        return True

    @property
    def help_dict(self) -> dict[str, str]:
        return {}

@dataclass
class PromptConfig(BaseConfig):
    pass

@dataclass
class SequencesConfig(BaseConfig):
    buffer: tuple[int, int] = (5, 5)
    compute_buffer: bool = True
    n_quantiles: int = 10
    top_acts_group_size: int = 20
    quantile_group_size: int = 5
    top_logits_hoverdata: int = 5
    stack_mode: Literal["stack-all", "stack-quantiles", "stack-none"] = "stack-all"
    hover_below: bool = True

    def data_is_contained_in(self, other) -> bool:
        assert isinstance(other, self.__class__)
        return all([
            self.buffer[0] <= other.buffer[0], # the buffer needs to be <=
            self.buffer[1] <= other.buffer[1],
            int(self.compute_buffer) <= int(other.compute_buffer), # we can't compute the buffer if we didn't in `other`
            self.n_quantiles in {0, other.n_quantiles}, # we actually need the quantiles identical (or one to be zero)
            self.top_acts_group_size <= other.top_acts_group_size, # group size needs to be <=
            self.quantile_group_size <= other.quantile_group_size, # each quantile group needs to be <=
            self.top_logits_hoverdata <= other.top_logits_hoverdata, # hoverdata rows need to be <=
        ])
        

    def __post_init__(self):
        # Get list of group lengths, based on the config params
        self.group_sizes = [self.top_acts_group_size] + [self.quantile_group_size] * self.n_quantiles

    @property
    def help_dict(self) -> dict[str, str]:
        return SEQUENCES_CONFIG_HELP

    # def help(self, return_table: bool):
    #     return super().help(SEQUENCES_CONFIG_HELP, return_table=return_table)

@dataclass
class ActsHistogramConfig(BaseConfig):
    n_bins: int = 50

    def data_is_contained_in(self, other) -> bool:
        assert isinstance(other, self.__class__)
        return self.n_bins == other.n_bins

    @property
    def help_dict(self) -> dict[str, str]:
        return ACTIVATIONS_HISTOGRAM_CONFIG_HELP

@dataclass
class LogitsHistogramConfig(BaseConfig):
    n_bins: int = 50

    def data_is_contained_in(self, other) -> bool:
        assert isinstance(other, self.__class__)
        return self.n_bins == other.n_bins

    @property
    def help_dict(self) -> dict[str, str]:
        return LOGITS_HISTOGRAM_CONFIG_HELP

@dataclass
class LogitsTableConfig(BaseConfig):
    n_rows: int = 10

    def data_is_contained_in(self, other) -> bool:
        assert isinstance(other, self.__class__)
        return self.n_rows <= other.n_rows

    @property
    def help_dict(self) -> dict[str, str]:
        return LOGITS_TABLE_CONFIG_HELP

@dataclass
class FeatureTablesConfig(BaseConfig):
    n_rows: int = 3

    def data_is_contained_in(self, other) -> bool:
        assert isinstance(other, self.__class__)
        return self.n_rows <= other.n_rows

    @property
    def help_dict(self) -> dict[str, str]:
        return FEATURE_TABLES_CONFIG_HELP

GenericConfig = Union[PromptConfig, SequencesConfig, ActsHistogramConfig, LogitsHistogramConfig, LogitsTableConfig, FeatureTablesConfig]

class Column:
    def __init__(self, *args, width: Optional[int] = None):
        self.components = args
        self.width = width
    
    def __iter__(self):
        return iter(self.components)

    def __getitem__(self, idx: int):
        return self.components[idx]
    
    def __len__(self):
        return len(self.components)


class SaeVisLayoutConfig:
    '''
    This object allows you to set all the ways the feature vis will be laid out.

    The init method also verifies no components are duplicated, and once it does this it will store each component
    as an attribute of the object. This means the config for a particular object can be accessed, regardless of which
    column it's in.
    '''
    columns: dict[int | tuple[int, int], Column]

    prompt_cfg: Optional[PromptConfig]
    seq_cfg: Optional[SequencesConfig]
    act_hist_cfg: Optional[ActsHistogramConfig]
    logits_hist_cfg: Optional[LogitsHistogramConfig]
    logits_table_cfg: Optional[LogitsTableConfig]
    feat_tables_cfg: Optional[FeatureTablesConfig]

    def __init__(self, columns: list[Column], **kwargs):
        # Define the columns
        self.columns = {idx: col for idx, col in enumerate(columns)}

        # Get the height (i.e. other kwargs)
        self.height = kwargs.get("height", 750)

        # By default, each component is None
        self.seq_cfg = None
        self.act_hist_cfg = None
        self.logits_hist_cfg = None
        self.logits_table_cfg = None
        self.feat_tables_cfg = None

        # Get a list of all components, verify there's no duplicates
        all_components = [comp for comps in self.columns.values() for comp in comps]
        all_component_names = [comp.__class__.__name__.rstrip("Config") for comp in all_components]
        assert len(all_component_names) == len(set(all_component_names)), "Duplicate components in layout config"
        self.components: dict[str, BaseConfig] = {name: comp for name, comp in zip(all_component_names, all_components)}

        # Once we've verified this, store each config component as an attribute
        for comp, comp_name in zip(all_components, all_component_names):
            match comp_name:
                case "Prompt":
                    self.prompt_cfg = comp
                case "Sequences":
                    self.seq_cfg = comp
                case "ActsHistogram":
                    self.act_hist_cfg = comp
                case "LogitsHistogram":
                    self.logits_hist_cfg = comp
                case "LogitsTable":
                    self.logits_table_cfg = comp
                case "FeatureTables":
                    self.feat_tables_cfg = comp
                case _:
                    raise ValueError(f"Unknown component name {comp_name}")

    def data_is_contained_in(self, other: "SaeVisLayoutConfig") -> bool:
        '''
        Returns True if `self` uses only data that would already exist in `other`. This is useful because our prompt-
        centric vis needs to only use data that was already computed as part of our initial data gathering. For example,
        if our SaeVisData object only contains the first 10 rows of the logits table, then we can't show the top 15 rows
        in the prompt centric view!
        '''
        for (comp_name, comp) in self.components.items():
            # If the component in `self` is not present in `other`, return False
            if comp_name not in other.components:
                return False
            # If the component in `self` is present in `other`, but the `self` component is larger, then return False
            comp_other = other.components[comp_name]
            if not comp.data_is_contained_in(comp_other):
                return False
        
        return True

    def help(
        self,
        title: str = "SaeVisLayoutConfig",
        key: bool = True,
    ) -> Tree | None:
        '''
        This prints out a tree showing the layout of the vis, by column (as well as the values of the arguments for each
        config object, plus their default values if they changed, and the descriptions of each arg).
        '''
        
        # Create tree (with title and optionally the key explaining arguments)
        if key: title += "\n\n" + KEY_LAYOUT_VIS
        tree = Tree(title)

        n_columns = len(self.columns)

        # For each column, add a tree node
        for i, (column, vis_components) in enumerate(self.columns.items()):
            
            n_components = len(vis_components)
            tree_column = tree.add(f"Column {column}")

            # For each component in that column, add a tree node
            for vis_component in vis_components:

                # Add tree node for the component
                tree_component = tree_column.add(f"{vis_component.__class__.__name__}".rstrip("Config"))

                # For each config parameter of that component
                for j, (param, value) in enumerate(asdict(vis_component).items()):

                    # Get line break at the end of the component descriptions (not of the entire table)
                    suffix = "\n" if (j == n_components - 1) and (i < n_columns - 1) else ""

                    # Get argument description, and its default value
                    desc = vis_component.help_dict.get(param, "")
                    value_default = getattr(vis_component.__class__, param, "no default")

                    # Add tree node (appearance is different if value is changed from default)
                    if value != value_default:
                        info = f"[b dark_orange]{param}: {value!r}[/] ({value_default!r}) \n[i]{desc}[/i]{suffix}"
                    else:
                        info = f"[b #00aa00]{param}: {value!r}[/] \n[i]{desc}[/i]{suffix}"
                    tree_component.add(info)

        rprint(tree)


KEY_LAYOUT_VIS = """Key: 
  the tree shows which components will be displayed in each column (from left to right)
  arguments are [b #00aa00]green[/]
  arguments changed from their default are [b dark_orange]orange[/], with default in brackets
  argument descriptions are in [i]italics[/i]
"""

KEY_VIS = """Key: 
  arguments are [b #00aa00]green[/]
  arguments changed from their default are [b dark_orange]orange[/], with default in brackets
  argument descriptions are in [i]italics[/i]
"""


DEFAULT_LAYOUT_FEATURE_VIS = SaeVisLayoutConfig(
    columns = [
        Column(FeatureTablesConfig()),
        Column(ActsHistogramConfig(), LogitsTableConfig(), LogitsHistogramConfig()),
        Column(SequencesConfig(stack_mode='stack-none')),
    ],
    height = 750,
)

DEFAULT_LAYOUT_PROMPT_VIS = SaeVisLayoutConfig(
    columns = [
        Column(PromptConfig(), ActsHistogramConfig(), LogitsTableConfig(n_rows=5), SequencesConfig(top_acts_group_size=10, n_quantiles=0), width=450),
    ],
    height = 1000,
)




SAE_CONFIG_DICT = dict(
    hook_point = "The hook point to use for the SAE",
    features = "The set of features which we'll be gathering data for. If an integer, we only get data for 1 feature",
    batch_size = "The number of sequences we'll gather data for. If supplied then it can't be larger than `tokens[0]`, \
if not then we use all of `tokens`",
    minibatch_size_tokens = "The minibatch size we'll use to split up the full batch during forward passes, to avoid \
OOMs.",
    minibatch_size_features = "The feature minibatch size we'll use to split up our features, to avoid OOM errors",
    seed = "Random seed, for reproducibility (e.g. sampling quantiles)",
    verbose = "Whether to print out progress messages and other info during the data gathering process",
)



@dataclass_json
@dataclass
class SaeVisConfig:

    # Data
    hook_point: Optional[str] = None
    features: Optional[int | Iterable[int]] = None
    batch_size: Optional[int] = None
    minibatch_size_features: int = 256
    minibatch_size_tokens: int = 64

    # Vis
    feature_centric_layout: SaeVisLayoutConfig = DEFAULT_LAYOUT_FEATURE_VIS
    prompt_centric_layout: SaeVisLayoutConfig = DEFAULT_LAYOUT_PROMPT_VIS

    # # Misc
    seed: Optional[int] = 0
    verbose: bool = False

    # # Can't use post init like this, because dataclass json won't support it
    # def __post_init__(self):
    #     assert isinstance(self.hook_point, str), f"{self.hook_point=}, should be a string."

    def help(self, title: str = "SaeVisConfig"):
        '''
        Performs the `help` method for each of the layout object, as well as for the non-layout-based configs.
        '''
        # Create table for all the non-layout-based params
        table = Table("Param", "Value (default)", "Description", title=title, show_lines=True)

        # Populate table (middle row is formatted based on whether value has changed from default)
        for param, desc in SAE_CONFIG_DICT.items():
            value = getattr(self, param)
            value_default = getattr(self.__class__, param, "no default")
            if value != value_default:
                value_default_repr = "no default" if value_default == "no default" else repr(value_default)
                value_str = f"[b dark_orange]{value!r}[/]\n({value_default_repr})"
            else:
                value_str = f"[b #00aa00]{value!r}[/]"
            table.add_row(param, value_str, f"[i]{desc}[/]")

        # Print table, and print the help trees for the layout objects
        rprint(table)
        self.feature_centric_layout.help(title="SaeVisLayoutConfig: feature-centric vis", key=False)
        self.prompt_centric_layout.help(title="SaeVisLayoutConfig: prompt-centric vis", key=False)



