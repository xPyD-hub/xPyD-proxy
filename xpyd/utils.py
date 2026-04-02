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
