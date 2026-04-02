# SPDX-License-Identifier: Apache-2.0
"""Load-balanced scheduling policy."""

import itertools
import logging
from typing import Optional

from xpyd.scheduler.scheduler_base import SchedulingPolicy

logger = logging.getLogger("xpyd.proxy")

try:
    from xpyd.proxy import query_instance_model_len
except ImportError:

    def query_instance_model_len(instances, timeout=5.0):
        return [131072] * len(instances)

# Ensure logger has a handler for scheduler output
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class LoadBalancedScheduler(SchedulingPolicy):
    """Select the least-loaded instance, respecting model-length limits."""

    def __init__(self, prefill_instances: list[str], decode_instances: list[str], registry=None):
        self.prefill_utils_counter = [0] * len(prefill_instances)
        self.prefill_bs_counter = [0] * len(prefill_instances)
        self.decode_kv_utils_counter = [0] * len(decode_instances)
        self.decode_bs_counter = [0] * len(decode_instances)

        self.prefill_instances = prefill_instances
        self.decode_instances = decode_instances
        logger.info(
            "LoadBalancedScheduler, prefill/decode instance counts: prefill=%d, decode=%d",
            len(self.prefill_bs_counter),
            len(self.decode_bs_counter),
        )
        logger.info(
            "LoadBalancedScheduler, self.prefill_instances=%s",
            self.prefill_instances,
        )
        logger.info(
            "LoadBalancedScheduler, self.decode_instances=%s",
            self.decode_instances,
        )
        self.prefill_schedule_index = 0
        self.prefill_schedule_completion_index = 0
        self.decode_schedule_index = 0
        self.decode_schedule_completion_index = 0

        self.prefill_model_len = query_instance_model_len(prefill_instances)
        self.decode_model_len = query_instance_model_len(decode_instances)

        logger.info("Prefill instance model lens: %s", self.prefill_model_len)
        logger.info("Decode instance model lens: %s", self.decode_model_len)
        super().__init__(registry=registry)

    def schedule(
        self,
        cycler: itertools.cycle,
        is_prompt: Optional[bool] = None,
        request_len: Optional[int] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Optional[str]:
        with self.lock:
            if is_prompt:
                return self._schedule_prefill(request_len, max_tokens)
            else:
                return self._schedule_decode(request_len, max_tokens)

    def _schedule_prefill(self, request_len, max_tokens):
        available = None
        if self._registry is not None:
            available = set(self._registry.get_available_instances("prefill"))
        candidates = [
            i
            for i, max_len in enumerate(self.prefill_model_len)
            if request_len + max_tokens <= max_len
            and (available is None or self.prefill_instances[i] in available)
        ]
        if not candidates:
            logger.warning(
                "No prefill instance can handle request_len=%d, max_tokens=%d",
                request_len,
                max_tokens,
            )
            return None

        min_value = min(self.prefill_utils_counter[i] for i in candidates)
        min_index = next(
            i for i in candidates if self.prefill_utils_counter[i] == min_value
        )

        self.prefill_bs_counter[min_index] += 1
        self.prefill_utils_counter[min_index] += request_len
        self.prefill_schedule_index += 1
        logger.info(
            f"<schedule prefill {self.prefill_schedule_index}> "
            f"instance = {min_index}, min_tokens = {min_value}",
        )
        return self.prefill_instances[min_index]

    def _schedule_decode(self, request_len, max_tokens):
        available = None
        if self._registry is not None:
            available = set(self._registry.get_available_instances("decode"))
        candidates = [
            i
            for i, max_len in enumerate(self.decode_model_len)
            if request_len + max_tokens <= max_len
            and (available is None or self.decode_instances[i] in available)
        ]
        if not candidates:
            logger.warning(
                "No decode instance can handle request_len=%d, max_tokens=%d",
                request_len,
                max_tokens,
            )
            return None

        min_value = min(self.decode_bs_counter[i] for i in candidates)
        if min_value == 0:
            min_index = next(
                i for i in candidates if self.decode_bs_counter[i] == 0
            )
        else:
            min_indices = [
                i for i in candidates if self.decode_bs_counter[i] == min_value
            ]
            min_index = min(
                min_indices, key=lambda i: self.decode_kv_utils_counter[i]
            )

        self.decode_bs_counter[min_index] += 1
        self.decode_kv_utils_counter[min_index] += request_len
        self.decode_schedule_index += 1
        logger.info(
            f"<schedule decode {self.decode_schedule_index}> "
            f"instance = {min_index}, min_batch = {min_value}",
        )
        logger.info(
            f"<schedule decode> decode_bs_counter: {self.decode_bs_counter}"
        )
        logger.info(
            f"<schedule decode> "
            f"decode_kv_utils_counter: {self.decode_kv_utils_counter}",
        )
        return self.decode_instances[min_index]

    def schedule_completion(
        self,
        prefill_instance: Optional[str] = None,
        decode_instance: Optional[str] = None,
        req_len: Optional[int] = None,
    ) -> None:
        with self.lock:
            if prefill_instance:
                self._complete_prefill(prefill_instance, req_len)
            if decode_instance:
                self._complete_decode(decode_instance, req_len)

    def _complete_prefill(self, prefill_instance, req_len):
        index = self.prefill_instances.index(prefill_instance)
        if self.prefill_bs_counter[index] == 0:
            logger.warning("No alive requests for prefill instance, skipping...")
            return

        self.prefill_schedule_completion_index += 1
        logger.info(
            f"<Prefill completed {self.prefill_schedule_completion_index}> "
            f"instance = {index}, req_len={req_len}",
        )

        self.prefill_bs_counter[index] -= 1
        if all(c == 0 for c in self.prefill_bs_counter):
            logger.warning("<Prefill in idle state>")
            self.prefill_utils_counter = [0] * len(self.prefill_instances)
        else:
            self.prefill_utils_counter[index] -= req_len

    def _complete_decode(self, decode_instance, req_len):
        index = self.decode_instances.index(decode_instance)
        if self.decode_bs_counter[index] == 0:
            logger.warning("No alive requests for decode instance, skipping...")
            return

        self.decode_schedule_completion_index += 1
        logger.info(
            f"<Decode completed {self.decode_schedule_completion_index}> "
            f"instance = {index}, req_len={req_len}",
        )

        self.decode_bs_counter[index] -= 1
        if all(c == 0 for c in self.decode_bs_counter):
            logger.warning("<Decode in idle state>")
            self.decode_kv_utils_counter = [0] * len(self.decode_instances)
        else:
            self.decode_kv_utils_counter[index] -= req_len
            logger.info(
                f"<schedule_completion decode> "
                f"decode_bs_counter: {self.decode_bs_counter}",
            )
            logger.info(
                f"<schedule_completion decode> "
                f"decode_kv_utils_counter: {self.decode_kv_utils_counter}",
            )
