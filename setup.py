from setuptools import setup, find_packages

setup(
    name = 'sae-vis',
    version = '0.2.8',
    packages = find_packages(),
    install_requires = [
        'torch',
        'einops',
        'datasets',
        'dataclasses-json',
        'jaxtyping',
    ],
    include_package_data = True,
    author = 'Callum McDougall',
    author_email = 'cal.s.mcdougall@gmail.com',
    description = "Open-source SAE visualizer, based on Anthropic's published visualizer.",
    url = 'https://github.com/callummcdougall/sae_vis',
    classifiers = [
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    long_description="""
This notebook was created to demo my open-source sparse autoencoder visualizer, as can be seen [here](https://www.perfectlynormal.co.uk/blog-sae). Other useful links:

* [GitHub repo](https://github.com/callummcdougall/sae_vis)
* [Developer guide](https://docs.google.com/document/d/10ctbiIskkkDc5eztqgADlvTufs7uzx5Wj8FE_y5petk/edit#heading=h.t3sp1uj6qghd), for people looking to understand the codebase so they can contribute
* [User guide](https://docs.google.com/document/d/1QGjDB3iFJ5Y0GGpTwibUVsvpnzctRSHRLI-0rm6wt_k/edit#heading=h.t3sp1uj6qghd), for people looking to understand how to use all the features of this codebase (although obviously reading through this notebook is another option, it should be mostly self-explanatory)

Version history (started to record at 0.2.8)

* `0.2.8` - added table for pairwise feature correlations (not just encoder-B correlations) 
""".strip(),
    long_description_content_type='text/markdown',
)