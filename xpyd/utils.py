# SPDX-License-Identifier: Apache-2.0
"""Shared utility functions."""

import logging

logger = logging.getLogger("xpyd.proxy")


def get_total_token_length(tokenizer, prompt):
    """Compute total token length for various prompt formats.

    Supports all OpenAI API prompt types:
    - ``str`` — plain text
    - ``list[str]`` — multiple text segments
    - ``list[int]`` — pre-tokenized token IDs (single sequence)
    - ``list[list[int]]`` — pre-tokenized token IDs (multiple sequences)
    - ``list[dict]`` — multimodal content array (extracts ``"text"`` parts)
    - ``None`` — returns 0

    Parameters
    ----------
    tokenizer : callable
        HuggingFace-style tokenizer that returns ``{"input_ids": [...]}``.
    prompt : str | list | None
        The prompt in any supported format.

    Returns
    -------
    int
        Total number of tokens.
    """
    fake_len = 100
    if prompt is None:
        return 0
    if isinstance(prompt, str):
        return len(tokenizer(prompt)["input_ids"])
    elif isinstance(prompt, list):
        if len(prompt) == 0:
            return 0
        # Single flat list of ints — already tokenized token IDs
        if all(isinstance(x, int) for x in prompt):
            return len(prompt)
        if all(isinstance(p, str) for p in prompt):
            return sum(len(tokenizer(p)["input_ids"]) for p in prompt)
        if all(
            isinstance(p, list) and all(isinstance(x, int) for x in p)
            for p in prompt
        ):
            # Nested list of ints — multiple already-tokenized sequences
            return sum(len(p) for p in prompt)
        if all(isinstance(p, dict) for p in prompt):
            # Multimodal content array — extract text parts only
            total = 0
            for p in prompt:
                if "text" in p:
                    total += len(tokenizer(p["text"])["input_ids"])
            return total
        logger.error(
            "Unsupported prompt format: %s / nested types. Value: %r",
            type(prompt),
            prompt,
        )
        return fake_len
    else:
        logger.error("Unsupported prompt type: %s", type(prompt))
        return fake_len


def query_instance_model_len(instances, timeout=5.0):
    """Query each instance for its max_model_len.

    Parameters
    ----------
    instances : list[str]
        Instance addresses in ``host:port`` format.
    timeout : float
        HTTP request timeout in seconds.

    Returns
    -------
    list[int]
        Max model length for each instance. Falls back to 131072 on failure.
    """
    import requests

    _DEFAULT_MODEL_LEN = 131072
    model_lens = []
    for inst in instances:
        try:
            url = f"http://{inst}/v1/models"
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()["data"][0]
            max_len = data.get("max_model_len", _DEFAULT_MODEL_LEN)
            model_lens.append(max_len)
            logger.info("Instance %s model_len: %d", inst, max_len)
        except Exception as e:
            logger.warning(
                "Failed to get model_len from %s, using default %d: %s",
                inst,
                _DEFAULT_MODEL_LEN,
                e,
            )
            model_lens.append(_DEFAULT_MODEL_LEN)
    return model_lens
