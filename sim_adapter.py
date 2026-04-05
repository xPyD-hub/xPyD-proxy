"""sim_adapter — drop-in for dummy_nodes using real xpyd-sim."""

import os

from xpyd_sim.server import ServerConfig, create_app

_DEFAULT_MODEL = os.environ.get("SIM_MODEL_NAME", "dummy")


def make_sim_app(model_name=None, mode="dual"):
    """Create a real xpyd-sim app. model_name defaults to SIM_MODEL_NAME env or 'dummy'."""
    return create_app(ServerConfig(
        mode=mode, model_name=model_name or _DEFAULT_MODEL, prefill_delay_ms=0,
        kv_transfer_delay_ms=0, decode_delay_per_token_ms=0,
        eos_min_ratio=1.0, max_model_len=131072,
    ))


prefill_app = make_sim_app(mode="prefill")
decode_app = make_sim_app(mode="decode")
