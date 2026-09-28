# SPDX-License-Identifier: Apache-2.0
"""Load-balanced scheduling policy."""

import logging
from typing import Optional, Sequence

from xpyd.scheduler.scheduler_base import Candidate, SchedulingContext, SchedulingPolicy
from xpyd.utils import query_instance_model_len

logger = logging.getLogger("xpyd.proxy")


class LoadBalancedScheduler(SchedulingPolicy):
    """Select the least-loaded instance, respecting model-length limits."""

    def __init__(
        self, prefill_instances: list[str], decode_instances: list[str], registry=None
    ):
        self.prefill_utils_counter = [0] * len(prefill_instances)
        self.prefill_bs_counter = [0] * len(prefill_instances)
        self.decode_kv_utils_counter = [0] * len(decode_instances)
        self.decode_bs_counter = [0] * len(decode_instances)

        self.prefill_instances = prefill_instances
        self.decode_instances = decode_instances
        logger.info(
            "LoadBalancedScheduler, prefill/decode instance counts: "
            "prefill=%d, decode=%d",
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

    @classmethod
    def from_config(
        cls,
        *,
        prefill_instances=(),
        decode_instances=(),
        workers=(),
        registry=None,
        tokenizer=None,
        **options,
    ):
        return cls(prefill_instances, decode_instances, registry=registry, **options)

    def select_node(
        self, context: SchedulingContext, candidates: Sequence[Candidate]
    ) -> Optional[str]:
        if not candidates:
            return None
        if context.role == "aggregated":
            return min(
                candidates, key=lambda candidate: candidate.active_requests
            ).address
        available = {candidate.address for candidate in candidates}
        with self.lock:
            select = (
                self._schedule_prefill
                if context.role == "prefill"
                else self._schedule_decode
            )
            return select(context.request_len, context.max_tokens, available)

    def on_instance_added(self, role, address, max_model_len):
        """Extend load-tracking arrays before shared membership is appended."""
        if role == "aggregated":
            return
        with self.lock:
            if role == "prefill":
                self.prefill_utils_counter.append(0)
                self.prefill_bs_counter.append(0)
                self.prefill_model_len.append(max_model_len)
            else:
                self.decode_kv_utils_counter.append(0)
                self.decode_bs_counter.append(0)
                self.decode_model_len.append(max_model_len)

    def on_instance_removed(self, role, address, index):
        """Remove load-tracking entries before a runtime instance is deleted."""
        if role == "aggregated":
            return
        with self.lock:
            if role == "prefill":
                del self.prefill_utils_counter[index]
                del self.prefill_bs_counter[index]
                del self.prefill_model_len[index]
            else:
                del self.decode_kv_utils_counter[index]
                del self.decode_bs_counter[index]
                del self.decode_model_len[index]

    def _schedule_prefill(self, request_len, max_tokens, available):
        candidates = [
            i
            for i, max_len in enumerate(self.prefill_model_len)
            if request_len + max_tokens <= max_len
            and self.prefill_instances[i] in available
        ]
        if not candidates:
            logger.warning(
                "No prefill instance available",
                extra={"request_len": request_len, "max_tokens": max_tokens},
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
            "Schedule prefill",
            extra={
                "schedule_index": self.prefill_schedule_index,
                "instance": min_index,
                "min_tokens": min_value,
            },
        )
        return self.prefill_instances[min_index]

    def _schedule_decode(self, request_len, max_tokens, available):
        candidates = [
            i
            for i, max_len in enumerate(self.decode_model_len)
            if request_len + max_tokens <= max_len
            and self.decode_instances[i] in available
        ]
        if not candidates:
            logger.warning(
                "No decode instance available",
                extra={"request_len": request_len, "max_tokens": max_tokens},
            )
            return None

        min_value = min(self.decode_bs_counter[i] for i in candidates)
        if min_value == 0:
            min_index = next(i for i in candidates if self.decode_bs_counter[i] == 0)
        else:
            min_indices = [
                i for i in candidates if self.decode_bs_counter[i] == min_value
            ]
            min_index = min(min_indices, key=lambda i: self.decode_kv_utils_counter[i])

        self.decode_bs_counter[min_index] += 1
        self.decode_kv_utils_counter[min_index] += request_len
        self.decode_schedule_index += 1
        logger.info(
            "Schedule decode",
            extra={
                "schedule_index": self.decode_schedule_index,
                "instance": min_index,
                "min_batch": min_value,
            },
        )
        logger.info(
            "Decode counters",
            extra={
                "bs_counter": list(self.decode_bs_counter),
                "kv_utils_counter": list(self.decode_kv_utils_counter),
            },
        )
        return self.decode_instances[min_index]

    def on_request_finished(self, context: SchedulingContext, address: str) -> None:
        with self.lock:
            if context.role == "prefill":
                self._complete_prefill(address, context.request_len)
            elif context.role == "decode":
                self._complete_decode(address, context.request_len)

    def _complete_prefill(self, prefill_instance, req_len):
        index = self.prefill_instances.index(prefill_instance)
        if self.prefill_bs_counter[index] == 0:
            logger.warning("No alive requests for prefill instance, skipping...")
            return

        self.prefill_schedule_completion_index += 1
        logger.info(
            "Prefill completed",
            extra={
                "completion_index": self.prefill_schedule_completion_index,
                "instance": index,
                "req_len": req_len,
            },
        )

        self.prefill_bs_counter[index] -= 1
        if all(c == 0 for c in self.prefill_bs_counter):
            logger.warning("Prefill in idle state")
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
            "Decode completed",
            extra={
                "completion_index": self.decode_schedule_completion_index,
                "instance": index,
                "req_len": req_len,
            },
        )

        self.decode_bs_counter[index] -= 1
        if all(c == 0 for c in self.decode_bs_counter):
            logger.warning("Decode in idle state")
            self.decode_kv_utils_counter = [0] * len(self.decode_instances)
        else:
            self.decode_kv_utils_counter[index] -= req_len
            logger.info(
                "Decode completion counters",
                extra={
                    "bs_counter": list(self.decode_bs_counter),
                    "kv_utils_counter": list(self.decode_kv_utils_counter),
                },
            )
