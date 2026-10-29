# SPDX-License-Identifier: Apache-2.0
"""Route registration aggregator for MicroDisaggregatedProxy."""

from collections.abc import Awaitable, Callable

from fastapi import APIRouter

from xpyd.interfaces import (
    AsyncRequestHandler,
    GetFromInstance,
    PostToInstance,
    RequestHandler,
)
from xpyd.registry import InstanceInfo

from .chat_completions import register as register_chat_completions
from .completions import register as register_completions
from .detokenize import register as register_detokenize
from .embeddings import register as register_embeddings
from .health import register as register_health
from .instances import register as register_instances
from .invocations import register as register_invocations
from .metrics import register as register_metrics
from .models import register as register_models
from .pooling import register as register_pooling
from .rerank import register as register_rerank
from .score import register as register_score
from .status import register as register_status
from .tokenize import register as register_tokenize
from .version import register as register_version


def register_routes(
    router: APIRouter,
    *,
    completion: AsyncRequestHandler,
    chat_completion: AsyncRequestHandler,
    forward: PostToInstance,
    fetch: GetFromInstance,
    add_instance: Callable[[str, str], Awaitable[bool]],
    remove_instance: Callable[[str, str, float], Awaitable[None]],
    prefill_instances: list[str],
    decode_instances: list[str],
    get_instances: Callable[[], list[InstanceInfo]] | None = None,
    get_registered_models: Callable[[], list[str]] | None = None,
    custom_completion: RequestHandler | None = None,
    custom_chat_completion: RequestHandler | None = None,
) -> None:
    """Wire each endpoint to only the operations and data it needs."""
    register_completions(router, execute=completion, custom_handler=custom_completion)
    register_chat_completions(
        router, execute=chat_completion, custom_handler=custom_chat_completion
    )
    register_tokenize(router, forward=forward)
    register_detokenize(router, forward=forward)
    register_embeddings(router, forward=forward)
    register_pooling(router, forward=forward)
    register_score(router, forward=forward)
    register_rerank(router, forward=forward)
    register_invocations(router, forward=forward)
    register_status(
        router,
        prefill_instances=prefill_instances,
        decode_instances=decode_instances,
        get_instances=get_instances,
    )
    register_instances(
        router, add_instance=add_instance, remove_instance=remove_instance
    )
    register_health(router, fetch=fetch)
    register_models(router, fetch=fetch, get_registered_models=get_registered_models)
    register_version(router, fetch=fetch)
    register_metrics(router)
