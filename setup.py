from setuptools import setup, find_packages

setup(
    name = 'sae-vis',
    version = '0.2.4',
    packages = find_packages(),
    install_requires = [
        'torch',
        'einops',
        'datasets',
        'dataclasses-json',
        'jaxtyping',
        # 'eindex',
    ],
    # dependency_links = [
    #     'git+https://github.com/callummcdougall/eindex.git#egg=eindex'  # Add the link with egg specification
    # ],
    include_package_data = True,
    author = 'Callum McDougall',
    author_email = 'cal.s.mcdougall@gmail.com',
    description = "Open-source SAE visualizer, based on Anthropic's published visualizer.",
    url = 'https://github.com/callummcdougall/sae_visualizer',
    classifiers = [
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)