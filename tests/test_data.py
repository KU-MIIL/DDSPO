"""CPU smoke tests for the dataset and collate function."""

import json
import os

import torch
from safetensors.torch import save_file

from ddspo.data import SelfTrainingDataset, collate_fn


def _make_dataset(tmp_path, n=3, c=4, h=8, w=8):
    latent_dir = tmp_path / "latents"
    latent_dir.mkdir()
    entries = []
    for i in range(n):
        pid, nid = f"{i}.safetensors", f"{i}_neg.safetensors"
        save_file({"latent": torch.randn(c, h, w)}, str(latent_dir / pid))
        save_file({"latent": torch.randn(c, h, w)}, str(latent_dir / nid))
        entries.append({"id": str(i), "prompt": f"a photo number {i}",
                        "neg_prompt": f"a photo {i}", "pos_file": pid, "neg_file": nid})
    with open(tmp_path / "metadata.jsonl", "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return SelfTrainingDataset(data_dir=str(tmp_path), n_T=1000)


class _FakeTokenizer:
    model_max_length = 8

    def __call__(self, prompts, **kwargs):
        ids = torch.arange(len(prompts) * self.model_max_length).reshape(len(prompts), self.model_max_length)

        class _Out:
            input_ids = ids
        return _Out()


def test_getitem_shapes(tmp_path):
    ds = _make_dataset(tmp_path)
    assert len(ds) == 3
    latents, prompt, neg_prompt, timestep, paired = ds[0]
    assert latents.shape == (8, 8, 8)  # (2*C, H, W)
    assert isinstance(prompt, str) and isinstance(neg_prompt, str)
    assert timestep.shape == (1,) and paired.item() == 1


def test_collate_raw_strings(tmp_path):
    ds = _make_dataset(tmp_path)
    batch = collate_fn()([ds[0], ds[1]])
    assert batch["latents"].shape == (2, 8, 8, 8)
    assert batch["timesteps"].shape == (2,) and batch["paireds"].shape == (2,)
    assert batch["pos_prompts"] == ["a photo number 0", "a photo number 1"]
    assert "pos_input_ids" not in batch


def test_collate_tokenized(tmp_path):
    ds = _make_dataset(tmp_path)
    tok = _FakeTokenizer()
    batch = collate_fn(tok, tok)([ds[0], ds[1]])
    assert batch["pos_input_ids"].shape == (2, 8)
    assert batch["neg_input_ids"].shape == (2, 8)
    assert batch["pos_input_ids_2"].shape == (2, 8)
