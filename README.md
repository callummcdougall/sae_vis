Note - I'm still open to accepting PRs on this library, and am very happy for other people to build on it, but I won't be actively maintaining it going forwards since I'll be focusing on my job. The [SAELens](https://github.com/jbloomAus/SAELens) library will continue to have more development and iteration, and it uses a fork of this repo as well as containing a much larger suite of tools for working with SAEs, so depending on your use case you might find that library preferable!

---

This codebase was designed to replicate Anthropic's sparse autoencoder visualisations, which you can see [here](https://transformer-circuits.pub/2023/monosemantic-features/vis/a1.html). The codebase provides 2 different views: a **feature-centric view** (which is like the one in the link, i.e. we look at one particular feature and see things like which tokens fire strongest on that feature) and a **prompt-centric view** (where we look at once particular prompt and see which features fire strongest on that prompt according to a variety of different metrics).

Install with `pip install sae-vis`. Link to PyPI page [here](https://pypi.org/project/sae-vis/).

See [here](https://colab.research.google.com/drive/1SuoFIjLvzOAuSg1nkqNbtkwXr8EvQ7SY?usp=drive_link) for a demo Colab notebook (all the code to produce it is also in this repo, in the file `sae_vis/demos/demo.py`, as well as the files containing the created visualizations).

The library supports two types of visualizations:

1. **Feature-centric vis**, where you look at a single feature and see e.g. which sequences in a large dataset this feature fires strongest on.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/feature-vis-video.gif" width="800">

2. **Prompt-centric vis**, where you input a custom prompt and see which features score highest on that prompt, according to a variety of possible metrics.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/prompt-vis-video.gif" width="800">

# Citing this work

To cite this work, you can use this bibtex citation:

```
@misc{sae_vis,
    title  = {{SAE Visualizer}},
    author = {Callum McDougall},
    howpublished    = {\url{https://github.com/callummcdougall/sae_vis}},
    year   = {2024}
}
```

# Contributing

This project is uses [Poetry](https://python-poetry.org/) for dependency management. After cloning the repo, install dependencies with `poetry install`.

This project uses [Ruff](https://docs.astral.sh/ruff/) for formatting and linting, [Pyright](https://github.com/microsoft/pyright) for type-checking, and [Pytest](https://docs.pytest.org/) for tests. If you submit a PR, make sure that your code passes all checks. You can run all checks with `make check-all`.

# Version history (recording started at `0.2.9`)

- `0.2.9` - added table for pairwise feature correlations (not just encoder-B correlations)
- `0.2.10` - fix some anomalous characters
- `0.2.11` - update PyPI with longer description
- `0.2.12` - fix height parameter of config, add videos to PyPI description
- `0.2.13` - add to dependencies, and fix SAELens section
- `0.2.14` - fix mistake in dependencies
- `0.2.15` - refactor to support eventual scatterplot-based feature browser, fix `&rsquo;` HTML
- `0.2.16` - allow disabling buffer in feature generation, fix demo notebook, fix sae-lens compatibility & type checking
- `0.2.17` - use main branch of `sae-lens`
- `0.2.18` - remove circular dependency with `sae-lens`
- `0.2.19` - formatting, error-checking
- `0.2.20` - fix bugs, remove use of `batch_size` in config
- `0.2.21` - formatting
- `0.3.0` - major refactor which makes several improvements, removing complexity and adding new features:
    - OthelloGPT SAEs with linear probes (input / output space)
    - Attention output SAEs with max DFA visualized
    - Tokens labelled with their `(batch, seq)` indices as well as the change in correct-token probability on feature ablation, when hovered over
- `0.3.1` - fix transformerlens dependency
- `0.3.2` - adjust pyright type-checking
- `0.3.3` - remove pyright type-checking
- `0.3.6` - remove tests
