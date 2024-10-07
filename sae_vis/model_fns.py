import einops
import torch
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from jaxtyping import Float
from sae_lens import SAE, HookedSAETransformer, SAEConfig
from torch import Tensor
from transformer_lens import HookedTransformer, utils

from sae_vis.utils_fns import VocabType

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_resid_dir(dir: Float[Tensor, "feats d"], sae: SAE, model: HookedSAETransformer, input: bool = False):
    """
    Converts a batch of feature directions to residual stream directions (i.e. for writing to the
    model's residual stream). For example, if the SAE was trained on the residual stream then this
    is just the identity function, but if the SAE was trained on the MLP activations or attn head
    z-values then we return dir @ W_out or dir @ W_O.flatten(-2, -1) respectively.

    We also have the `input` argument: if True then we'll be returning the reading direction rather
    than writing direction (i.e. @ W_in or @ W_V.flatten(-2, -1) respectively).
    """

    match sae.cfg.hook_name.split(".hook_")[-1]:
        case "resid_pre" | "resid_mid" | "resid_post" | "attn_out" | "mlp_out":
            return dir
        case "pre" | "post":
            return dir @ (model.W_in[sae.cfg.hook_layer].T if input else model.W_out[sae.cfg.hook_layer])
        case "z":
            return dir @ (
                einops.rearrange(
                    model.W_V[sae.cfg.hook_layer],
                    "n_heads d_model d_head -> (n_heads d_head) d_model",
                )
                if input
                else einops.rearrange(
                    model.W_O[sae.cfg.hook_layer],
                    "n_heads d_head d_model -> (n_heads d_head) d_model",
                )
            )
        case _:
            raise ValueError(f"Unexpected hook name: {model.hook_point}")


def resid_final_pre_layernorm_to_logits(x: Tensor, model: HookedTransformer):
    x = x - x.mean(-1, keepdim=True)  # [batch, pos, length]
    scale = x.pow(2).mean(-1, keepdim=True) + model.cfg.eps
    x_normalized = x / scale
    return x_normalized @ model.W_U + model.b_U


def load_othello_vocab() -> dict[VocabType, dict[int, str]]:
    """
    Returns vocab dicts (embedding and unembedding) for OthelloGPT, i.e. token_id -> token_str.

    This means ["pass"] + ["A0", "A1", ..., "H7"].

    If probes=True, then this is actually the board squares (including middle ones)
    """

    all_squares = [r + c for r in "ABCDEFGH" for c in "01234567"]
    legal_squares = [sq for sq in all_squares if sq not in ["D3", "D4", "E3", "E4"]]

    vocab_dict_probes = {token_id: str_token for token_id, str_token in enumerate(all_squares)}
    vocab_dict = {token_id: str_token for token_id, str_token in enumerate(["pass"] + legal_squares)}
    return {
        "embed": vocab_dict,
        "unembed": vocab_dict,
        "probes": vocab_dict_probes,
    }


# def load_othello_linear_probes(
#     device: str = str(device),
# ) -> dict[str, Float[Tensor, "d_model d_vocab_out"]]:
#     """
#     Loads linear probe from paper & rearranges it to the correct format.

#     Interpretation:
#         - Initial linear probe has shape (3, d_model, rows, cols, 3) where:
#             - 0th dim = different move parity probes (black to play / odd, white / even, both)
#             - Last dim = classification of empty / black / white squares
#         - We create 3 new probes in a different basis (the "empty / theirs / mine" basis rather
#           than "empty / black / white"), by averaging over the old probes.
#             - Each new probe has shape (d_model, rows*cols=d_vocab_out).
#     """
#     OTHELLO_ROOT = Path(__file__).parent.parent / "othello_world"
#     OTHELLO_MECHINT_ROOT = OTHELLO_ROOT / "mechanistic_interpretability"
#     assert OTHELLO_MECHINT_ROOT.exists()
#     linear_probe = t.load(OTHELLO_MECHINT_ROOT / "main_linear_probe.pth", map_location=device)

#     black_to_play, white_to_play, _ = 0, 1, 2
#     square_is_empty, square_is_white, square_is_black = 0, 1, 2

#     linear_probe = einops.rearrange(
#         linear_probe,
#         "probes d_model rows cols classes -> probes classes d_model (rows cols)",
#     )

#     # Change of basis 1: from "blank/black/white" to "blank/mine/theirs"
#     linear_probe = {
#         "theirs": linear_probe[[black_to_play, white_to_play], [square_is_white, square_is_black]].mean(0),
#         "mine": linear_probe[[black_to_play, white_to_play], [square_is_black, square_is_white]].mean(0),
#         "empty": linear_probe[[black_to_play, white_to_play], [square_is_empty, square_is_empty]].mean(0),
#     }

#     # Change of basis 2: get a "mine vs theirs" direction & "blank" direction
#     linear_probe = {
#         "mine vs theirs": linear_probe["mine"] - linear_probe["theirs"],
#         "empty": linear_probe["empty"] - 0.5 * (linear_probe["mine"] + linear_probe["theirs"]),
#     }

#     # Normalize
#     linear_probe = {k: v / v.norm(dim=0).mean() for k, v in linear_probe.items()}

#     # important thing: when these probes say "mine" they mean what just got moved, not who is to move!
#     linear_probe = {
#         "theirs vs mine": linear_probe["mine vs theirs"],
#         "empty": linear_probe["empty"],
#     }

#     # the middle 4 squares being empty is meaningless
#     linear_probe["empty"][:, [27, 28, 35, 36]] = 0.0

#     return linear_probe


def load_demo_model_saes_and_data(seq_len: int, device: str) -> tuple[SAE, SAE, HookedSAETransformer, Tensor]:
    """
    This loads in the SAEs (and dataset) we'll be using for our demo examples.
    """

    SEQ_LEN = seq_len
    DATASET_PATH = "NeelNanda/c4-code-20k"
    MODEL_NAME = "gelu-1l"
    HOOK_NAME = "blocks.0.mlp.hook_post"
    saes: list[SAE] = []

    model = HookedSAETransformer.from_pretrained(MODEL_NAME)

    for version in [25, 47]:
        state_dict = utils.download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version}.pt", force_is_torch=True)
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
