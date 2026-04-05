"""sim_adapter — drop-in for dummy_nodes using real xpyd-sim."""
from xpyd_sim.server import ServerConfig, create_app


def make_sim_app(model_name="dummy", mode="dual"):
    return create_app(ServerConfig(
        mode=mode, model_name=model_name, prefill_delay_ms=0,
        kv_transfer_delay_ms=0, decode_delay_per_token_ms=0,
        eos_min_ratio=1.0, max_model_len=131072,
    ))

prefill_app = make_sim_app("dummy", "prefill")
decode_app = make_sim_app("dummy", "decode")
