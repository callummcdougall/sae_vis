from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name = 'sae-vis',
    version = '0.2.15',
    packages = find_packages(),
    install_requires = [
        'torch',
        'einops',
        'eindex-callum',
        'rich',
        'transformer_lens',
        'datasets',
        'dataclasses-json',
        'jaxtyping',
        'pytest',
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
    long_description=long_description,
    long_description_content_type='text/markdown',
)
