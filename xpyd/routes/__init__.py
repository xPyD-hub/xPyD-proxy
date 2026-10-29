# SPDX-License-Identifier: Apache-2.0
"""Route registration aggregator for MicroDisaggregatedProxy."""

from typing import TYPE_CHECKING

from fastapi import APIRouter

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

if TYPE_CHECKING:
    from xpyd.proxy import Proxy


def register_routes(router: APIRouter, server: "Proxy") -> None:
    """Register all route handlers on *router*, using *server* for state."""
    register_completions(router, server)
    register_chat_completions(router, server)
    register_tokenize(router, server)
    register_detokenize(router, server)
    register_embeddings(router, server)
    register_pooling(router, server)
    register_score(router, server)
    register_rerank(router, server)
    register_invocations(router, server)
    register_status(router, server)
    register_instances(router, server)
    register_health(router, server)
    register_models(router, server)
    register_version(router, server)
    register_metrics(router, server)
