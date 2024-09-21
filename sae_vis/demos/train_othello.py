import copy
import os
from pathlib import Path

import torch as t
from datasets import Dataset, IterableDataset, load_dataset
from IPython.display import clear_output, display
from sae_lens import (
    SAE,
    ActivationsStore,
    LanguageModelSAERunnerConfig,
    SAEConfig,
    SAETrainingRunner,
)
from transformer_lens import HookedTransformer

dataset_path = "taufeeque/othellogpt"
model_name = "othello-gpt"

model = HookedTransformer.from_pretrained(model_name)

device = t.device("cuda" if t.cuda.is_available() else "cpu")


training_tokens = int(5e7)
train_batch_size_tokens = 2048
n_steps = int(training_tokens / train_batch_size_tokens)

runner_cfg = LanguageModelSAERunnerConfig(
    #
    # general
    model_name=model_name,
    hook_name="blocks.6.hook_resid_pre",
    hook_layer=6,
    dataset_path=dataset_path,
    device=str(model.cfg.device),
    dtype="float32",
    seed=42,
    prepend_bos=False,
    #
    # architecture
    architecture="gated",
    context_size=59,
    d_in=model.cfg.d_model,
    b_dec_init_method="mean",
    expansion_factor=4,
    #
    # dataset / batches
    n_batches_in_buffer=32,
    store_batch_size_prompts=32,
    training_tokens=training_tokens,
    train_batch_size_tokens=train_batch_size_tokens,
    seqpos_slice=(5, -5),
    #
    # learning rates / schedulers
    lr=0.0005,
    lr_scheduler_name="constant",
    l1_coefficient=2.5,
    lr_warm_up_steps=int(0.2 * n_steps),
    lr_decay_steps=int(0.2 * n_steps),
    l1_warm_up_steps=int(0.2 * n_steps),
    feature_sampling_window=500,
    dead_feature_window=int(1e6),
    #
    # logging
    log_to_wandb=True,
    wandb_project="othello_gpt_sae_16_09",
    wandb_log_frequency=30,
    n_checkpoints=0,
    checkpoint_path="checkpoints",
)


class CustomDataset(IterableDataset):
    def __init__(self, original_dataset: Dataset):
        self.original_dataset = original_dataset

    def __iter__(self):
        for item in self.original_dataset:
            yield {"tokens": item["tokens"][:-1]}  # type: ignore


original_dataset = load_dataset(
    dataset_path,
    split="train",
    streaming=True,
    trust_remote_code=True,
)
override_dataset = CustomDataset(original_dataset)  # type: ignore

# ! test my activations shape will be right (now I've changed some of the code)

activations_store = ActivationsStore.from_config(model, runner_cfg, override_dataset)
batch = activations_store.get_batch_tokens()
acts = activations_store.get_activations(batch)
print(f"{batch.shape=}, {acts.shape=}")

activations_store.seqpos_slice = slice(5, -5)
batch = activations_store.get_batch_tokens()
acts = activations_store.get_activations(batch)
print(f"{tuple(batch.shape)=}, {tuple(acts.shape)=}")


# gated, mean bias, longer
sae = SAETrainingRunner(runner_cfg, override_dataset=override_dataset).run()
activations = sae.encode_standard(batch)
# Path(path := "/content/trained-sae-gated-bias").mkdir(exist_ok=True)
# sae.save_model(path=path)


s1 = (None, None, None)
slice(*s1)

s2 = (5, -5)
slice(*s2)
