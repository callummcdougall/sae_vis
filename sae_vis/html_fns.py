from matplotlib import colors
from typing import Optional, Any
from pathlib import Path
import re
import json

from sae_vis.utils_fns import (
    apply_indent,
    deep_union,
)
from sae_vis.data_config_classes import (
    SaeVisLayoutConfig,
    Column,
)

'''
This file contains all functions which do HTML-specific things. Mostly, this is the `HTML` class, which is the return
type for the `_get_html_data` methods in the classes in `data_storing_fns.py`. This class contains the HTML string as
well as JavaScript data in the form of dictionaries. 

Why was the choice made to separate these, rather than just concatenating HTML strings? Because it's useful to keep the
JavaScript data stored as data, so we can do things like merge it or save it all to a single file / dump it into our
HTML file as a single variable.

The rough structure of the HTML file returned by `HTML.get_html` is as follows:

```html
<div id='dropdown-container'></div> # for containing our dropdowns, to navigate between different views
        
<div class='grid-container'>
    <div id='column-0' class="grid-column">
        ... # HTML string for column 0 (i.e. containing a bunch of components, with IDs distinguishable by suffix)
    </div>
    ... # more columns
</div>

<style>
... # CSS
</style>

<script src="https://d3js.org/d3.v6.min.js"></script>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

<script>
document.addEventListener("DOMContentLoaded", function(event) {{
    const DATA = defineData();      # load in our data
    createDropdowns(DATA);          # create dropdowns from the data, and also create the vis for the first time
}});


function createDropdowns(DATA) {{
    const START_KEY = {json.dumps(first_key)};      
    ...

    function updateDropdowns() {
        ... 
        # find the currently selected key, then run createVis(DATA[selectedKey]); 
    }

    select.on('change', function() {
        updateDropdowns();      # rebuild vis when dropdown changes
    });

    updateDropdowns();          # initially build vis
}}

function createVis(DATA) {{
    ...
    # create the vis from the data (this is where all the JavaScript files in this repo get dumped into)
}}

function defineData() {{
    const DATA = {json.dumps(self.js_data)};
    return DATA;
}}
# DATA is a nested dictionary, with the following levels:
#  - 1st level: The keys are "|"-separated options for the dropdowns
#  - 2nd level: The keys are names like `tokenData`, `featureTablesData`, etc.
#               These keys correspond to JS files like `tokenScript.js`, `featureTablesScript.js`, etc.
#               Those scripts use `DATA.tokenData`, `DATA.featureTablesData`, etc.
</script>
```
'''




BG_COLOR_MAP = colors.LinearSegmentedColormap.from_list("bg_color_map", ["white", "darkorange"])

def bgColorMap(x: float):
    '''
    Returns background color, which is a linear interpolation of x as follows:

        0: white
        1: darkorange
    '''
    # assert min(x, 1-x) > -1e-6, f"Expected 0 <= x <= 1, but got {x}"
    x2 = max(0.0, min(1.0, x))
    return colors.rgb2hex(BG_COLOR_MAP(x2))


def uColorMap(x: float) -> str:
    '''
    Returns underline color, which is a linear interpolation of x as follows:

        -1: blue
        0 : transparent
        +1: red
    '''
    # assert min(x-1, 1-x) > -1e-6, f"Expected -1 <= x <= 1, but got {x}"
    x2 = max(-1.0, min(1.0, x))

    if x2 < 0:
        v = int(255 * (x2 + 1))
        return f"rgb({v},{v},255)"
    else:
        v = int(255 * (1 - x2))
        return f"rgb(255,{v},{v})"


class HTML:
    html_data: dict[int | tuple[int, int], str]
    js_data: dict[str, dict[str, Any]]
    '''
    Contains HTML strings & JavaScript data to populate them.

    Args:
        html_data: 
            Keys are the columns we'll be storing that HTML in; values are the HTML strings. When we `__add__` 2 objects
            together, we concat these across columns. When we use `get_html`, again we'll concat across columns
            (wrapping each column's contents in a `grid-item` div), and wrap everything in a `grid-container`.

        js_data:
            Keys are names like `tokenData`, `featureTablesData`, etc. Each name has a corresponding JavaScript file
            like `tokenScript.js`, `featureTablesScript.js`, etc, and those scripts will use `DATA.tokenData`. So the
            `js_data` object is a standin for this `DATA` dict. When we `__add__` 2 objects together, we merge the
            dicts. When we call `get_html`, all the scripts will be wrapped in a JavaScript function `load` which has
            a single argument `DATA`, and we'll handle the data by:
                
                - Loading in DATA, either as a direct `DATA = {json.dumps(...)}` or using `fetch`

                - Calling `load`, either on DOMContentLoaded & with `DATA` argument if `js_data_index=False`, or on
                  slider change & with slider value as argument if `js_data_index=True`

    This is the object returned by each of the `_get_html_data` methods in the classes in `data_storing_fns.py`. It
    helps standardize the way we create & add to our final HTML vis.
    '''
    def __init__(
        self,
        html_data: Optional[dict[int | tuple[int, int], str]] = None,
        js_data: Optional[dict[str, dict[str, Any]]] = None,
    ) -> None:
        self.html_data = html_data if (html_data is not None) else {}
        self.js_data = js_data if (js_data is not None) else {}        

    def __add__(self, other: "HTML") -> "HTML":
        '''
        Merges the JavaScript data and the HTML string together, and returns a new HTML object.

        Further explanation of how this works, for each of the HTML components:

            html_data:
                The HTML data for the same column is concatenated together. Note that the column keys can be ints like 1
                or tuples like (1, 0), (1, 1) ... (the latter is what we do when content in a single column overflows
                onto multiple columns).


        This is how we take separate returned objects from the classes in `data_storing_fns.py` and merge them together
        into a single object.
        '''
        # Merge HTML data by concatenating strings in every column
        html_data = self.html_data.copy()
        for k, v in other.html_data.items():
            html_data[k] = html_data.get(k, "") + v

        # Merge JavaScript data by taking union over data dicts for every different file
        js_data = deep_union(self.js_data, other.js_data)

        return HTML(html_data, js_data)
    
    def get_html(
        self,
        layout: SaeVisLayoutConfig,
        filename: str | Path,
        first_key: str,
    ) -> None:
        '''
        Returns the HTML string, together with JavaScript and CSS. 

        Args:
            layout:         The `SaeVisLayoutConfig` object which contains important data about the full layout (not
                            component-specific because this has already been handled; either column-specific e.g. width
                            or applying to the whole vis e.g. height).
            filename:       The name of the file to save the HTML to. If `single_file` is False, then we'll take the
                            stem of this file but with "json" suffix to save the data.
            first_key:      The key that our vis will be initialized with.

        Further explanation of how this works, for each of the HTML components:

            html_data:
                We'll concat all the strings for each column together, each each wrapped inside a `grid-column` div, and
                then the whole thing will be wrapped in `grid-container`.

            js_data:
                The js_data is initially a dict mapping JavaScript filenames "<name>Script.js" to dictionaries which 
                will be dumped into the first line of those files
        '''
        html_str = ""

        # Check arguments
        if isinstance(filename, str): filename = Path(filename)
        assert filename.suffix == ".html", f"Expected {filename.resolve()!r} to have .html suffix"
        assert filename.parent.exists(), f"Expected {filename.parent.resolve()!r} to exist"

        # ! JavaScript

        # Get path where we store all template JavaScript files
        js_path = Path(__file__).parent / "js"
        assert all(file.suffix == ".js" for file in js_path.iterdir()),\
            f"Expected all files in {js_path.resolve()} to have .js suffix"
        
        # Define the contents of the `createVis` function, which takes in some `DATA[key]` object, and uses it to fill
        # in HTML. We create this by concatenating all files referred to in the keys of the `DATA[key]` object, since
        # its keys are just JS filenames with the `Script.js` suffix removed. Lastly, note that we put the histograms
        # scripts first, because hovering over tokens might require a relayout on a histogram, so it needs to exist!
        js_data_filenames = next(iter(self.js_data.values())).keys()
        js_data_filenames = [name.replace("Data", "Script.js") for name in js_data_filenames]
        js_data_filenames = sorted(js_data_filenames, key=lambda x: "histograms" not in x.lower())
        js_create_vis = "\n".join([(js_path / js_name).read_text() for js_name in js_data_filenames])
        
        # Read in the code which will dynamically create dropdowns from the DATA keys
        js_create_dropdowns = (js_path / "_createDropdownsScript.js").read_text()
        
        # Put everything together, in the correct order (including defining DATA & creating dropdowns from it, which
        # is done in DOMContentLoaded). Note, double curly braces are used to escape single curly braces in f-strings.
        js_str = f"""
document.addEventListener("DOMContentLoaded", function(event) {{
    // Define our data (this is basically where we've dumped the full DATA object; at the end of this file)
    const DATA = defineData();
    // Create dropdowns (or not) based on this object, and also create the vis for the first time
    createDropdowns(DATA);
}});

// Create dropdowns from DATA object, and trigger creation of the vis (using `START_KEY` as initial key)
function createDropdowns(DATA) {{
    const START_KEY = {json.dumps(first_key)};
{apply_indent(js_create_dropdowns, "    ")}
}}

function createVis(DATA) {{
{apply_indent(js_create_vis, "    ")}
}}

function defineData() {{
    const DATA = {json.dumps(self.js_data)};
    return DATA;
}}
"""
                
        # ! CSS

        # We simply merge the different CSS files together (they're only kept separate for modularity)
        css_str = "\n".join([
            file.read_text()
            for file in (Path(__file__).parent / "css").iterdir()
        ])

        # ! HTML

        # Get the HTML string (by column)
        for col_idx, html_str_column in self.html_data.items():

            # Ideally layout.columns[col_idx] would be the column object, but we have to deal with 2 special cases:
            #   (1) we're doing the prompt-centric view, so we always want to use layout.columns[0]
            #   (2) our column overflowed, then col_idx is actually a tuple (x, y), and we want layout.columns[x]
            
            if isinstance(col_idx, int):
                # Deal with case (1) here, as well as the general case
                column = layout.columns[min(len(layout.columns) - 1, col_idx)]
                column_id = f"column-{col_idx}"
            elif isinstance(col_idx, tuple):
                # Deal with case (2) here
                column = layout.columns[col_idx[0]]
                column_id = "column-" + "-".join(map(str, col_idx))
            html_str += "\n\n" + grid_column(html_str_column, id=column_id, column=column, layout=layout)

        # Remove empty style attributes
        html_str = re.sub(r' style=""', "", html_str)
        
        # Create the full HTML string: wrap everything in `grid-container`, and also create object for holding dropdowns
        full_html_str = f"""
<div id='dropdown-container'></div>
        
<div class='grid-container'>
{apply_indent(html_str, " " * 4)}
</div>

<style>
{css_str}
</style>

<script src="https://d3js.org/d3.v6.min.js"></script>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

<script>
{js_str}
</script>
"""

        # TODO - this CSS thing about the border
        # css_str = CSS
        # if not(border):
        #     lines_to_remove = ["border: 1px solid #e6e6e6;", "box-shadow: 0 5px 5px rgba(0, 0, 0, 0.25);"]
        #     for line in lines_to_remove:
        #         assert line in css_str, f"Unexpected CSS in css_str: should contain {line!r}"
        #         css_str = css_str.replace(line, "")

        filename.write_text(full_html_str)
        


def grid_column(
    html_contents: str,
    column: Column,
    layout: SaeVisLayoutConfig,
    id: Optional[str] = None,
    indent: str = " " * 4,
) -> str:
    '''
    Wraps the HTML contents in a 'grid-column' element.

    Args:
        html_contents:  The string we're wrapping
        column:         The `Column` object which contains important data about the column, e.g. width
        layout:         The `SaeVisLayoutConfig` object which contains important data about the full layout (not specific
                        to a single column), e.g. height
        id:             The id of the `grid-column` element (this will usually be `column-0`, `column-1`, etc.)

    We pass the `Column` object to this function, because it contains important data about the column, such as its
    width. We also pass the `SaeVisLayoutConfig` object, because it contains important data about the full layout (not
    column-specific).
    '''
    # height_str = f"height: {height}px; " if height is not None else ""
    # margin_str = f"margin-left: {left_margin}px;" if left_margin is not None else ""


    style_str = ""
    if (column.width is not None) or (layout.height is not None):
        width_str = f"width: {column.width}px; " if column.width is not None else ""
        height_str = f"max-height: {layout.height}px; " if layout.height is not None else ""
        style_str = f"style='{width_str}{height_str}'"

    id_str = f"id='{id}' " if id is not None else ""
    
    return f'''<div {id_str}class="grid-column" {style_str}>
{apply_indent(html_contents, indent)}</div>'''



# # TODO - why doesn't fetch work on browsers? Annoying security thing, means I need to have the data in just one file
# filename_json = filename.with_suffix(".json")
# filename_json.write_text(json.dumps(self.js_data))
# js_str_data_dump = r"""
# fetch('FILENAME'})
#   .then(response => response.json())
#   .then(data => {var DATA = data;})
#   .catch(error => console.error('Error loading the JSON file:', error));
# """.replace("FILENAME", filename_json.name)

