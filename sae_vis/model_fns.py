import json
import re
import zipfile
from pathlib import Path

import einops
import numpy as np
import torch
import torch as t
import torch.nn as nn
from datasets import IterableDataset, load_dataset
from datasets.arrow_dataset import Dataset
from huggingface_hub import hf_hub_download, snapshot_download
from jaxtyping import Float, Int
from sae_lens import SAE, HookedSAETransformer, SAEConfig
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint

from sae_vis.utils_fns import VocabType


class HookedTransformerWrapper(nn.Module):
    """
    This class wraps around & extends the TransformerLens model, so that we can make sure things like the forward
    function have a standardized signature.

    # TODO - this can just be a HookedSAETransformer, and I can get rid of the feature batching (or maybe add a PR to SAELens to allow for feature batching on their side?)
    """

    def __init__(self, model: HookedTransformer, hook_point: str):
        super().__init__()
        assert (
            hook_point in model.hook_dict
        ), f"Error: hook_point={hook_point!r} must be in model.hook_dict"
        self.model = model
        self.hook_point = hook_point
        self.cfg = model.cfg

        # Get the layer (so we can do the early stopping in our forward pass)
        layer_match = re.match(r"blocks\.(\d+)\.", hook_point)
        assert layer_match, f"Error: expecting hook_point to be 'blocks.{{layer}}.{{...}}', but got {hook_point!r}"
        self.hook_layer = int(layer_match.group(1))

        # Get the hook names for the residual stream (final) and residual stream (immediately after hook_point)
        self.hook_point_resid = utils.get_act_name("resid_post", self.hook_layer)
        self.hook_point_resid_final = utils.get_act_name(
            "resid_post", self.model.cfg.n_layers - 1
        )
        assert self.hook_point_resid in model.hook_dict
        assert self.hook_point_resid_final in model.hook_dict

    def forward(
        self, tokens: Int[Tensor, "batch seq"]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Inputs:
            tokens: Int[Tensor, "batch seq"]
                The input tokens, shape (batch, seq)
        """
        # Run with hook functions to store the activations & final value of residual stream
        logits: Tensor = self.model.run_with_hooks(
            tokens,
            # stop_at_layer = (None if return_logits else self.hook_layer),
            fwd_hooks=[
                (self.hook_point, self.hook_fn_store_act),
                (self.hook_point_resid_final, self.hook_fn_store_act),
            ],
        )

        # The hook functions work by storing data in model's hook context, so we pop them back out
        activation = self.model.hook_dict[self.hook_point].ctx.pop("activation")
        if self.hook_point_resid_final == self.hook_point:
            residual = activation
        else:
            residual = self.model.hook_dict[self.hook_point_resid_final].ctx.pop(
                "activation"
            )

        return logits, residual, activation

    def hook_fn_store_act(self, activation: torch.Tensor, hook: HookPoint):
        hook.ctx["activation"] = activation.detach().clone()

    @property
    def tokenizer(self):
        return self.model.tokenizer

    @property
    def W_U(self):
        return self.model.W_U

    @property
    def W_out(self):
        return self.model.W_out

    @property
    def W_in(self):
        return self.model.W_in


def to_resid_dir(
    dir: Float[Tensor, "feats d"], model: HookedTransformerWrapper, input: bool = False
):
    """
    Converts a batch of feature directions to residual stream directions (either reading or
    writing). Examples:

        dir = vectors in residual stream
            -> returns dir

        dir = MLP layer activations, input = False
            -> returns dir @ W_out for that layer

        dir = MLP layer activations, input = False
            -> returns dir @ W_in for that layer

    Args:
        dir:
            The direction in the activations, i.e. shape (feats, d_in) where d_in could be d_model, d_mlp, etc.
        model:
            The model, which should be a HookedTransformerWrapper or similar.
    """
    # If this SAE was trained on the residual stream or attn/mlp out, then we don't need to do anything
    if any(
        x in model.hook_point
        for x in ["resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out"]
    ):
        return dir

    # If it was trained on the MLP layer, then we apply the W_out map
    elif any(x in model.hook_point for x in ["mlp.hook_pre", "mlp.hook_post"]):
        return dir @ (
            model.W_in[model.hook_layer].T if input else model.W_out[model.hook_layer]
        )

    # Others not yet supported
    else:
        raise NotImplementedError(
            "The hook your SAE was trained on isn't yet supported"
        )


# ! Othello stuff


def load_othello_data(remove_last_move: bool = True) -> Int[t.Tensor, "batch seq"]:
    OTHELLO_ROOT = Path(__file__).parent.parent / "othello_world"
    OTHELLO_MECHINT_ROOT = (OTHELLO_ROOT / "mechanistic_interpretability").resolve()
    assert OTHELLO_MECHINT_ROOT.exists()

    board_seqs_int = t.tensor(
        np.load(OTHELLO_MECHINT_ROOT / "board_seqs_int_small.npy"), dtype=t.long
    )
    board_seqs_string = t.tensor(
        np.load(OTHELLO_MECHINT_ROOT / "board_seqs_string_small.npy"), dtype=t.long
    )

    assert all([middle_sq not in board_seqs_string for middle_sq in [27, 28, 35, 36]])
    assert board_seqs_int.max() == 60

    if remove_last_move:
        board_seqs_int = board_seqs_int[:, :-1]

    print(f"{board_seqs_int.shape = } = (num_games, length_of_game)")

    return board_seqs_int


def load_othello_dataset_as_in_colab(remove_last_move: bool = True) -> IterableDataset:
    dataset = load_dataset(
        "taufeeque/othellogpt",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    assert isinstance(dataset, IterableDataset)

    if remove_last_move:

        class CustomDataset(IterableDataset):
            def __init__(self, original_dataset: IterableDataset):
                self.original_dataset = original_dataset

            def __iter__(self):
                for item in self.original_dataset:
                    yield {"tokens": item["tokens"][:-1]}

        dataset = CustomDataset(dataset)

    return dataset


def load_othello_data_as_in_colab(
    n_games: int, remove_last_move: bool = True
) -> Tensor:
    dataset = load_othello_dataset_as_in_colab(remove_last_move=remove_last_move)

    # Extract games from dataset, turn into tensor
    batch = []
    for i, example in enumerate(dataset):
        if i >= n_games:
            break
        batch.append(example["tokens"])

    return t.tensor(batch, dtype=t.long)


def load_othello_sae(dir_name: str) -> SAE:
    """
    Loads the SAE from where I'm manually saving mine.
    """
    OTHELLO_SAE_PATH = (
        "C:/Users/calsm/Documents/AI Alignment/sae-vis/trained_saes/othello"
    )
    filenames = [x.name for x in Path(OTHELLO_SAE_PATH).iterdir()]
    assert dir_name in filenames
    with open(f"{OTHELLO_SAE_PATH}/{dir_name}/cfg.json") as f:
        cfg_dict = json.load(f)
    cfg = SAEConfig.from_dict(cfg_dict)

    with safe_open(
        f"{OTHELLO_SAE_PATH}/{dir_name}/sae_weights.safetensors", framework="pt"
    ) as f:  # type: ignore
        # expected_keys = {"W_dec", "W_enc", "b_dec", "b_enc"}
        # assert set(f.keys()) == expected_keys, f"Unexpected keys: {f.keys()}"
        state_dict = {key: f.get_tensor(key) for key in f.keys()}
    for name, param in state_dict.items():
        print(f"{name}, shape={tuple(param.shape)}")
    sae = SAE(cfg)
    sae.load_state_dict(state_dict)

    return sae


def load_othello_vocab() -> dict[VocabType, dict[int, str]]:
    """
    Returns vocab dicts (embedding and unembedding) for OthelloGPT, i.e. token_id -> token_str.

    This means ["pass"] + ["A0", "A1", ..., "H7"].

    If probes=True, then this is actually the board squares (including middle ones)
    """

    all_squares = [r + c for r in "ABCDEFGH" for c in "01234567"]
    legal_squares = [sq for sq in all_squares if sq not in ["D3", "D4", "E3", "E4"]]

    vocab_dict_probes = {
        token_id: str_token for token_id, str_token in enumerate(all_squares)
    }
    vocab_dict = {
        token_id: str_token
        for token_id, str_token in enumerate(["pass"] + legal_squares)
    }
    return {
        "embed": vocab_dict,
        "unembed": vocab_dict,
        "probes": vocab_dict_probes,
    }


def load_othello_linear_probes(
    device: t.device = t.device("cuda" if t.cuda.is_available() else "cpu"),
) -> dict[str, Float[Tensor, "d_model d_vocab_out"]]:
    """
    Loads linear probe from paper & rearranges it to the correct format.

    Interpretation:
        - Initial linear probe has shape (3, d_model, rows, cols, 3) where:
            - 0th dim = different move parity probes (black to play / odd, white / even, both)
            - Last dim = classification of empty / black / white squares
        - We create 3 new probes in a different basis (the "empty / theirs / mine" basis rather
          than "empty / black / white"), by averaging over the old probes.
            - Each new probe has shape (d_model, rows*cols=d_vocab_out).
    """
    OTHELLO_ROOT = Path(__file__).parent.parent / "othello_world"
    OTHELLO_MECHINT_ROOT = OTHELLO_ROOT / "mechanistic_interpretability"
    assert OTHELLO_MECHINT_ROOT.exists()
    linear_probe = t.load(OTHELLO_MECHINT_ROOT / "main_linear_probe.pth").to(device)

    black_to_play, white_to_play, _ = 0, 1, 2
    square_is_empty, square_is_white, square_is_black = 0, 1, 2

    linear_probe = einops.rearrange(
        linear_probe,
        "probes d_model rows cols classes -> probes classes d_model (rows cols)",
    )

    # Change of basis 1: from "blank/black/white" to "blank/mine/theirs"
    linear_probe = {
        "theirs": linear_probe[
            [black_to_play, white_to_play], [square_is_white, square_is_black]
        ].mean(0),
        "mine": linear_probe[
            [black_to_play, white_to_play], [square_is_black, square_is_white]
        ].mean(0),
        "empty": linear_probe[
            [black_to_play, white_to_play], [square_is_empty, square_is_empty]
        ].mean(0),
    }

    # Change of basis 2: get a "mine vs theirs" direction & "blank" direction
    linear_probe = {
        "mine vs theirs": linear_probe["mine"] - linear_probe["theirs"],
        "empty": linear_probe["empty"]
        - 0.5 * (linear_probe["mine"] + linear_probe["theirs"]),
    }

    # Normalize
    linear_probe = {k: v / v.norm(dim=0).mean() for k, v in linear_probe.items()}

    # important thing: when these probes say "mine" they mean what just got moved, not who is to move!
    linear_probe = {
        "theirs vs mine": linear_probe["mine vs theirs"],
        "empty": linear_probe["empty"],
    }

    # the middle 4 squares being empty is meaningless
    linear_probe["empty"][:, [27, 28, 35, 36]] = 0.0

    return linear_probe


# ! Code to download the paper's models (this is more code than I actually needed)
def load_paper_othello_models(
    filename: str = "othello-trained_model-layer_5-2024-05-23.zip",
    model_name: str = "othello-trained_model-layer_5-standard/trainer39-v2",
    dir_name: str = "othello-paper-resid5",
) -> None:
    """
    This code downloads the Othello models from the paper and saves them (along with configs) in
    a way which can be loaded by `load_othello_sae()`.
    """

    # Define paths for loading
    othello_models = Path(__file__).parent.parent.parent / "trained_saes/othello"
    othello_paper_model = (
        othello_models / f"{filename.removesuffix('.zip')}/{model_name}"
    )

    # Download & extract model
    repo_id = "adamkarvonen/othello_saes"
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    print(f"Model downloaded to: {model_path}")
    with zipfile.ZipFile(model_path, "r") as zip_ref:
        zip_ref.extractall(str(othello_models))
    print(f"Model extracted to: {othello_models}")

    # Load config
    with open(str(othello_paper_model / "config.json")) as f:
        print(json.load(f))

    # Load weights, and rename them
    ae_weights = t.load(str(othello_paper_model / "ae.pt"), map_location="cpu")
    for k, v in ae_weights.items():
        print(k, v.shape)
    rename_dict = {
        "bias": "b_dec",
        "encoder.weight": "W_enc",
        "encoder.bias": "b_enc",
        "decoder.weight": "W_dec",
    }
    ae_weights_renamed = {
        rename_dict[k]: (v if v.ndim == 1 else v.T).contiguous()
        for k, v in ae_weights.items()
    }

    # Create SAE config and load weights
    sae_cfg = SAEConfig(
        architecture="standard",
        d_in=512,
        d_sae=8192,
        dtype="float32",
        device="cuda",
        model_name="othello-gpt",
        # model_name="Baidicoot/Othello-GPT-Transformer-Lens",
        hook_name="blocks.6.hook_resid_pre",
        hook_layer=6,
        hook_head_index=None,
        activation_fn_str="relu",
        activation_fn_kwargs={},
        apply_b_dec_to_input=True,
        finetuning_scaling_factor=False,
        sae_lens_training_version="3.19.4",
        prepend_bos=True,
        dataset_path="taufeeque/othellogpt",
        dataset_trust_remote_code=True,
        context_size=59,
        normalize_activations="none",
        neuronpedia_id=None,
        model_from_pretrained_kwargs={"center_writing_weights": False},
    )
    # sae = SAE(sae_cfg)
    # sae.load_state_dict(ae_weights_renamed)

    othello_dir = othello_models / dir_name
    othello_dir.mkdir()
    with open(othello_dir / "cfg.json", "w") as f:
        json.dump(sae_cfg.to_dict(), f)
    save_file(ae_weights_renamed, othello_dir / "sae_weights.safetensors")

    print(f"Saved config & weights to {othello_dir}")
    othello_paper_model.rmdir()

    # Save SAE, and config


def load_demo_model_saes_and_data(seq_len: int, device: str):
    """
    This loads in the SAEs (and dataset) we'll be using for our demo examples.
    """

    SEQ_LEN = seq_len
    DATASET_PATH = "NeelNanda/c4-code-20k"
    MODEL_NAME = "gelu-1l"
    HOOK_NAME = "blocks.0.mlp.hook_post"
    saes: list[SAE] = []

    model = HookedTransformer.from_pretrained(MODEL_NAME)

    for version in [25, 47]:
        state_dict = utils.download_file_from_hf(
            "NeelNanda/sparse_autoencoder", f"{version}.pt", force_is_torch=True
        )
        assert isinstance(state_dict, dict)
        assert set(state_dict.keys()) == {"W_enc", "W_dec", "b_enc", "b_dec"}
        d_in, d_sae = state_dict["W_enc"].shape

        # Create autoencoder
        cfg = SAEConfig(
            architecture="standard",
            # forward pass details.
            d_in=d_in,
            d_sae=d_sae,
            activation_fn_str="relu",
            apply_b_dec_to_input=True,
            finetuning_scaling_factor=False,
            # dataset it was trained on details.
            context_size=SEQ_LEN,
            model_name=MODEL_NAME,
            hook_name=HOOK_NAME,
            hook_layer=0,
            hook_head_index=None,
            prepend_bos=True,
            dataset_path=DATASET_PATH,
            dataset_trust_remote_code=False,
            normalize_activations="None",
            # misc
            sae_lens_training_version=None,
            dtype="float32",
            device=str(device),
        )
        sae = SAE(cfg)
        sae.load_state_dict(state_dict)
        saes.append(sae)

    sae, sae_B = saes

    # Load in the data (it's a Dataset object)
    data = load_dataset(DATASET_PATH, split="train")
    assert isinstance(data, Dataset)

    dataset = load_dataset(path=DATASET_PATH, split="train", streaming=False)

    tokenized_data = utils.tokenize_and_concatenate(
        dataset=dataset,  # type: ignore
        tokenizer=model.tokenizer,  # type: ignore
        streaming=True,
        max_length=sae.cfg.context_size,
        add_bos_token=sae.cfg.prepend_bos,
    )
    tokenized_data = tokenized_data.shuffle(42)
    all_tokens = tokenized_data["tokens"]
    assert isinstance(all_tokens, torch.Tensor)
    print(all_tokens.shape)

    return sae, sae_B, model, all_tokens


def load_demo_othello_model_saes_and_data(device: str):
    """
    This loads in the SAEs (and dataset) we'll be using for our demo examples, in Othello.
    """
    model = (
        HookedSAETransformer.from_pretrained(
            "othello-gpt", center_writing_weights=False
        )
        .to(device)
        .eval()
    )

    local_dir = snapshot_download("callummcdougall/othello-sae")
    print(f"All files downloaded to: {local_dir}")

    cfg_dict = json.loads(Path(local_dir + "/cfg.json").read_text())
    state_dict = load_file(local_dir + "/sae_weights.safetensors")
    sae = SAE.from_dict(cfg_dict)
    sae.load_state_dict(state_dict)

    # tokens.shape = (20k, 59) because we exclude the last move
    # target_logits.shape = (20k, 58, 61) because for all token positions except the last one, we predict over the set of (pass plus each of the 60 legal moves)
    tokens: Tensor = torch.load(local_dir + "/tokens.pt")
    target_logits: Tensor = (
        torch.load(local_dir + "/target_logits.pt").float().to(device)
    )
    print(f"{tokens.shape=}, {target_logits.shape=}")

    return sae, model, tokens, target_logits
