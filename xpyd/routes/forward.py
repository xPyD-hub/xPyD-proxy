# SPDX-License-Identifier: Apache-2.0
"""Proxy / forward-to-instance route handlers."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse


def register(router: APIRouter, server) -> None:
    """Register proxy-forward routes on *router*."""

    async def post_tokenize(request: Request):
        return await server.post_to_instance(request, "/tokenize", {"model": "", "prompt": ""})

    async def post_detokenize(request: Request):
        return await server.post_to_instance(request, "/detokenize", {"model": "", "tokens": []})

    async def post_embeddings(request: Request):
        return await server.post_to_instance(request, "/v1/embeddings", {"model": "", "input": ""})

    async def post_pooling(request: Request):
        return await server.post_to_instance(request, "/pooling", {"model": "", "messages": ""})

    async def post_score(request: Request):
        return await server.post_to_instance(request, "/score", {"model": "", "text_1": "", "text_2": "", "predictions": ""})

    async def post_scorev1(request: Request):
        return await server.post_to_instance(request, "/v1/score", {"model": "", "text_1": "", "text_2": "", "predictions": ""})

    async def post_rerank(request: Request):
        return await server.post_to_instance(request, "/rerank", {"model": "", "query": "", "documents": ""})

    async def post_rerankv1(request: Request):
        return await server.post_to_instance(request, "/v1/rerank", {"model": "", "query": "", "documents": ""})

    async def post_rerankv2(request: Request):
        return await server.post_to_instance(request, "/v2/rerank", {"model": "", "query": "", "documents": ""})

    async def post_invocations(request: Request):
        return await server.post_to_instance(request, "/invocations", {"model": "", "prompt": ""})

    router.post("/tokenize", response_class=JSONResponse)(post_tokenize)
    router.post("/detokenize", response_class=JSONResponse)(post_detokenize)
    router.post("/v1/embeddings", response_class=JSONResponse)(post_embeddings)
    router.post("/pooling", response_class=JSONResponse)(post_pooling)
    router.post("/score", response_class=JSONResponse)(post_score)
    router.post("/v1/score", response_class=JSONResponse)(post_scorev1)
    router.post("/rerank", response_class=JSONResponse)(post_rerank)
    router.post("/v1/rerank", response_class=JSONResponse)(post_rerankv1)
    router.post("/v2/rerank", response_class=JSONResponse)(post_rerankv2)
    router.post("/invocations", response_class=JSONResponse)(post_invocations)

    router.options("/tokenize")(lambda: None)
    router.options("/detokenize")(lambda: None)
    router.options("/v1/embeddings")(lambda: None)
    router.options("/pooling")(lambda: None)
    router.options("/score")(lambda: None)
    router.options("/v1/score")(lambda: None)
    router.options("/rerank")(lambda: None)
    router.options("/v1/rerank")(lambda: None)
    router.options("/v2/rerank")(lambda: None)
    router.options("/invocations")(lambda: None)
