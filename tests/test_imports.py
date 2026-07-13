"""Import-graph and argument-parsing smoke tests.

These verify that every adapter module imports cleanly (no syntax / name errors
across the whole package) and that ``parse_args`` applies the right per-model
defaults. They do NOT load any pretrained weights.
"""

import sys

import pytest


def test_all_modules_import():
    import ddspo.args  # noqa: F401
    import ddspo.data  # noqa: F401
    import ddspo.dpo  # noqa: F401
    import ddspo.train  # noqa: F401
    from ddspo.adapters import get_adapter  # noqa: F401
    import ddspo.adapters.sd  # noqa: F401
    import ddspo.adapters.sd3  # noqa: F401
    import ddspo.adapters.sana  # noqa: F401


@pytest.mark.parametrize("model_type,cls_name,sdxl", [
    ("sd15", "SDAdapter", False),
    ("sdxl", "SDAdapter", True),
    ("sd3", "SD3Adapter", None),
    ("sana", "SANAAdapter", None),
])
def test_factory_returns_adapter(model_type, cls_name, sdxl):
    from ddspo.adapters import get_adapter
    adapter = get_adapter(model_type)
    assert type(adapter).__name__ == cls_name
    if sdxl is not None:
        assert adapter.sdxl is sdxl


def _parse(argv):
    from ddspo.args import parse_args
    old = sys.argv
    sys.argv = ["train.py"] + argv
    try:
        return parse_args()
    finally:
        sys.argv = old


def test_args_defaults_sd15():
    args = _parse(["--model_type", "sd15", "--pretrained_model_name_or_path", "x",
                   "--train_data_dir", "d"])
    assert args.resolution == 512
    assert args.max_sequence_length == 77
    assert args.resume_from_checkpoint is None  # no accidental auto-resume


def test_args_defaults_sana():
    args = _parse(["--model_type", "sana", "--pretrained_model_name_or_path", "x",
                   "--train_data_dir", "d"])
    assert args.resolution == 1024
    assert args.max_sequence_length == 300
