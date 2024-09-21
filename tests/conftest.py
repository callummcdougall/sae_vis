import pytest
from transformer_lens import HookedTransformer


@pytest.fixture
def model() -> HookedTransformer:
    model = HookedTransformer.from_pretrained("tiny-stories-1M", device="cpu")
    model.eval()
    return model


# TODO - similar test for autoencoder
