# TLDR 

This codebase was designed to replicate Anthropic's sparse autoencoder visualisations, which you can see [here](https://transformer-circuits.pub/2023/monosemantic-features/vis/a1.html). The codebase provides 2 different views: a **feature-centric view** (which is like the one in the link, i.e. we look at one particular feature and see things like which tokens fire strongest on that feature) and a **prompt-centric view** (where we look at once particular prompt and see which features fire strongest on that prompt according to a variety of different metrics).

You can use my [Colab](https://colab.research.google.com/drive/1rEHvywBZdnkHoie6YW88DxrHN1wnqhlF?usp=sharing) to generate more of these visualisations. You can use this [sae visualiser](https://www.perfectlynormal.co.uk/blog-sae) to navigate through the first thousand features of the aforementioned autoencoder.

There are 2 main sections of this readme: **views** (which explains the 2 different views that this library can create, using an example) and **repository structure** (which explains the different files in this repository and how they interact). If you just want to be able to run the code and don't care about exactly hwo it works, we recommend you read **views** and then jump into the Colab. Only read **repository structure** if you're curious, or if you want to modify the code in some way.

# Views

## [Feature-centric view](https://github.com/callummcdougall/sae_vis/blob/main/feature_vis_demo_repo.html)

Here is an example output (with some discussion). Note that I'm assuming basic familiarity with what SAEs and features are, if you don't have this then I'd recommend reading some prerequisites (e.g. sections 1, 6 and 7 of the [ARENA exercises](https://arena3-chapter1-transformer-interp.streamlit.app/[1.4]_Superposition_&_SAEs)).

This visualization (and the next one) was created using the GELU-1l model from Neel Nanda's HuggingFace library, as well as an autoencoder which he trained on its single layer of neuron activations (see [this Colab](https://colab.research.google.com/drive/1u8larhpxy8w4mMsJiSBddNOzFGj7_RTn) from Neel).

Feature #8 (below) seems to be a fuzzy skip trigram, with the pattern being `(django syntax)`, ..., ` ('` -> `django`. To verify that it was indeed responding to django syntax, I copied some of the text immediately preceding the bracket (e.g. `created_on` or `first_name`) into GPT4 and asked it to identify which library is being used - it correctly identified these as instances of Django syntax. Furthermore, we can see that this feature boosts `django` a lot more than any other token.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/sae-demo-1.png" width="1200">

## [Prompt-centric view](https://github.com/callummcdougall/sae_vis/blob/main/prompt_vis_demo_repo.html)

Here is a prompt-centric visualization, containing some django syntax. The features are sorted by their loss effect on the `django` token (i.e. how much the output of this particular feature changes the logprobs for the correct token). We can see that feature #8 from above is by far the most loss-reducing. We can also sort the features by activation size or activation quantile, this again shows that feature #8 is dominant.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/sae-demo-2.png" width="800">

# Models

This repo currently only supports transformerlens models, because there are a bunch of utilities that come with TL models (specifically hooks and a standardised architecture). However we might expand to supporting general models in the future.

# Repository structure

Here is a summary of each of the important files in this repository, as well as an explanation of what purpose they serve. Hopefully this will make it easier for other people to take and modify this code.

If you just want to run the code and don't care as much about how it works, I recommend you stop reading here, and instead go to the [Colab](https://colab.research.google.com/drive/1rEHvywBZdnkHoie6YW88DxrHN1wnqhlF?usp=sharing).

```
sae-vis
│
├── css
│   ├── general.css
│   ├── sequences.css
│   └── tables.css
│
├── html
│   ├── hovertext.html
│   ├── left_tables_template.html
│   ├── middle_plots.html
│   └── token_template.html
│
├── data_fetching_fns.py
├── data_storing_fns.py
├── html_fns.py
├── demo.ipynb
└── utils_fns.py
```

## Short descriptions

I'll start with one-sentence summaries of each folder or file. I'll have longer descriptions in the next section

* `css`, `html`, `js` - these contain raw CSS / HTML / JavaScript files which are read by `html_fns.py` and processed to create visualisations.
* `data_fetching_fns.py` - this returns data from the transformer (i.e. running batches on it and getting things like feature activations, ablation results, etc).
* `data_storing_fns.py` - this contains classes for storing data, which are then used to create visualisations (or save/load the data in JSON format). It's basically the middleman between `data_fetching_fns.py` and `html_fns.py`.
* `html_fns.py` - this contains functions which produce HTML visualisations. No computation is done here, everything is assumed to be of types like `str`, `float`, `int` or lists of these (i.e. no tensors or numpy arrays).
* `demo.ipynb` - this contains a demo you can run to actually generate visualisations.
* `utils_fns.py` - basic helper functions.

In other words, the basic workflow here is:

* Fetch the data using forward passes, via `data_fetching_fns.py`, and store them in datastructures defined in `data_storing_fns.py`,
* Call the `get_html` methods of these datastructures, which in turn call functions in `html_fns.py`, which read the raw HTML/CSS/JS files and insert the data into them using regex functions.
* Use the `save_json` method to save the data as a JSON file, or `load_json` classmethod to load it back in.

## Long descriptions

Now, I'll include some longer descriptions for each of these, along with explanations for why they are structured the way they are.

### `css`, `html` and `js`

These files contain templates for different parts of the visualization. For example, the `html` folder contains 3 HTML files: one is a template for the left-side tables (i.e. the ones with information like neuron alignment), one is for the middle set of plots (i.e. the bar charts and table) and one is for a single token in the sequences which make up the right hand side plots. Later, I'll discuss in more detail how the hierarchical structure of the visualization works.

Why did I choose to structure things like this? There were some alternatives, e.g. using open-source libraries like `jinja` or tools like `pysvelte`. I chose this process because it was simple and transparent, and it also allowed me to directly write HTML and CSS code rather than only writing Python strings (this is very useful because of the syntax highlighting and copilot).

As for why the files are separate, there are 2 reasons:

* **Hierarchy**. Some of the HTML template files will be used multiple times (e.g. the `token_template.html` file is just for a single token in our sequences, and obviously there are a lot of these!).
* **Modularity**. If I want to analyze a particular sequence then I only need the left visualizations (tables) and middle visualizations (bar charts), not the things on the right hand size. If I had all the CSS or all the HTML in a single file, I couldn't do something like this.

### `data_fetching_fns.py`

This contains functions which actually return the data that gets stored in the objects defined by the file below. This requires doing forward passes, hooks, ablations, etc. - all the computationally expensive stuff.

### `data_storing_fns.py`

This contains classes for storing data. The classes are structured hierarchically, with the highest-level object for the main image above being `FeatureData`. 

Many of these classes have a `get_html` method, which is called recursively by the top-level instance to generate the HTML string for the entire visualisation.

Here is an illustration of all the dataclasses we use, which also shows why it's handy to make these classes modular and hierarchical:

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/sae-vis-dataclass-hierarchy-1.png" width="1000">

Not illustrated is the class `MultiFeatureData`, which contains several `FeatureData` objects as well as information which is used to calculate the quantiles for the prompt-centric visualisation.

There is also a different high-level object `MultiPromptData`, which is designed to (for a single user-given input prompt) show a set of features which score very highly in some metric in this particular prompt (e.g. the features are sorted by activation on the final token in this prompt). This has a different hierarchical structure, but still with a lot of overlap:

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/sae-vis-dataclass-hierarchy-2.png" width="700">

### `html_fns.py`

This file contains functions which produce HTML visualisations. Every function in this file only takes the most basic possible datatype (i.e. strings, floats, ints, or lists of strings, floats, ints). They never take tensors, arrays, etc. This means we can keep these functions separate from those in the next 2 files.

The functions in this file read the HTML and CSS files in the `html` and `css` directories, and then insert the data into them using regex functions.

### `model_fns.py`

This contains code for the AutoEncoder class. Your autoencoder doesn't have to be exactly the same as this one; the library should hopefully be "plug and play".

### `utils_fns.py`

Helper functions which are used in the other files.
