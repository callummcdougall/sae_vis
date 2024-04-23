import json
from pathlib import Path

from transformer_lens import HookedTransformer

from sae_vis.data_config_classes import SaeVisConfig
from sae_vis.data_storing_fns import SaeVisData
from sae_vis.model_fns import AutoEncoder

ROOT_DIR = Path(__file__).parent.parent


def test_SaeVisData_create_results_look_reasonable(
    model: HookedTransformer, autoencoder: AutoEncoder
):
    cfg = SaeVisConfig(hook_point="blocks.2.hook_resid_pre", minibatch_size_tokens=2)
    tokens = model.to_tokens(
        [
            "But what about second breakfast?" * 3,
            "Nothing is cheesier than cheese." * 3,
        ]
    )
    data = SaeVisData.create(encoder=autoencoder, model=model, tokens=tokens, cfg=cfg)

    assert data.encoder == autoencoder
    assert data.model == model
    assert data.cfg == cfg
    # kurtosis and skew are both empty, is this itentional?
    assert len(data.feature_stats.max) == 128
    assert len(data.feature_stats.frac_nonzero) == 128
    assert len(data.feature_stats.quantile_data) == 128
    assert len(data.feature_stats.quantiles) > 1000
    for val in data.feature_stats.max:
        assert val >= 0
    for val in data.feature_stats.frac_nonzero:
        assert 0 <= val <= 1
    for prev_val, next_val in zip(
        data.feature_stats.quantiles[:-1], data.feature_stats.quantiles[1:]
    ):
        assert prev_val <= next_val
    for bounds, prec in data.feature_stats.ranges_and_precisions:
        assert len(bounds) == 2
        assert bounds[0] <= bounds[1]
        assert prec > 0
    # each feature should get its own key
    assert set(data.feature_data_dict.keys()) == set(range(128))


def test_SaeVisData_create_and_save_feature_centric_vis(
    model: HookedTransformer,
    autoencoder: AutoEncoder,
    tmp_path: Path,
):
    cfg = SaeVisConfig(hook_point="blocks.2.hook_resid_pre", minibatch_size_tokens=2)
    tokens = model.to_tokens(
        [
            "But what about second breakfast?" * 3,
            "Nothing is cheesier than cheese." * 3,
        ]
    )
    data = SaeVisData.create(encoder=autoencoder, model=model, tokens=tokens, cfg=cfg)
    save_path = tmp_path / "feature_centric_vis.html"
    data.save_feature_centric_vis(save_path)
    assert (save_path).exists()
    with open(save_path) as f:
        html_contents = f.read()

    # all the CSS should be in the HTML
    css_files = (ROOT_DIR / "sae_vis" / "css").glob("*.css")
    assert len(list(css_files)) > 0
    for css_file in css_files:
        with open(css_file) as f:
            assert f.read() in html_contents

    # all the JS should be in the HTML
    js_files = (ROOT_DIR / "sae_vis" / "js").glob("*.js")
    assert len(list(js_files)) > 0
    for js_file in js_files:
        with open(js_file) as f:
            assert f.read() in html_contents

    # all the HTML templates should be in the HTML
    html_files = (ROOT_DIR / "sae_vis" / "html").glob("*.html")
    assert len(list(html_files)) > 0
    for html_file in html_files:
        with open(html_file) as f:
            assert f.read() in html_contents

    assert json.dumps(data.feature_stats.aggdata) in html_contents
