"""sim_adapter — drop-in for dummy_nodes using real xpyd-sim.

The default model_name points to the test tokenizer bundled in this repo.
Override via make_sim_app(model_name=...) for custom models.
"""

from pathlib import Path

from xpyd_sim.server import ServerConfig, create_app

_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_MODEL = str(_REPO_ROOT / "tokenizers/DeepSeek-R1")


def make_sim_app(model_name=None, mode="dual"):
    """Create a real xpyd-sim app. Defaults to test tokenizer model."""
    return create_app(ServerConfig(
        mode=mode, model_name=model_name or _DEFAULT_MODEL, prefill_delay_ms=0,
        kv_transfer_delay_ms=0, decode_delay_per_token_ms=0,
        eos_min_ratio=1.0, max_model_len=131072,
    ))


# Default apps for subprocess (uvicorn sim_adapter:prefill_app)
prefill_app = make_sim_app(mode="prefill")
decode_app = make_sim_app(mode="decode")
